"""
scone_sensitivity.py — SCONE-inspired causal edge sensitivity scoring.

SCONE (Soft-intervention Causal ON-line Ensemble, Reisach et al. 2024)
is designed for the unknown soft-intervention setting — exactly what we have:
CHIP mutations, eQTLs, and drug exposures are all soft perturbations whose
exact targets and strengths are uncertain.

Key ideas imported here:
  1. Cross-regime sensitivity Γ_ij: how much does the gene_i → program_j
     relationship change between "perturbed" (disease/GWAS) and "background"
     (population-level expression) regimes?  Large Γ_ij = likely causal.

  2. PolyBIC scoring: BIC penalty for the number of proposed causal edges,
     scaled by evidence quality (Tier 1 edges have lower effective complexity).

  3. Bootstrap aggregation: resample β values within their uncertainty bounds
     N times and retain edges whose Ota γ exceeds the threshold in > 50% of
     samples.  Produces edge confidence rather than a hard threshold.

In the absence of matched control expression matrices (requires downloading
GTEx v8 or OneK1K), we approximate Γ_ij analytically:

  Γ_ij ≈ |β_ij| × |γ_j| × tier_quality(β) × tier_quality(γ)

This collapses the cross-regime shift to the product of causal strengths,
weighted by evidence quality — a conservative lower bound on the true Γ.

References:
  - SCONE: Reisach et al. NeurIPS 2024
  - AVICI: Lorch et al. NeurIPS 2022
  - PolyBIC: Chickering 2002 (JMLR)
"""
from __future__ import annotations

import math
import random
from typing import Any


# Evidence quality weights (higher = more certain β/γ estimate)
_TIER_QUALITY: dict[str, float] = {
    "Tier1_Interventional": 1.00,
    "Tier2_Convergent":     0.80,
    "Tier3_Provisional":    0.50,
    "moderate_transferred": 0.50,
    "moderate_grn":         0.40,
    "provisional_virtual":  0.10,
}

# PolyBIC complexity cost per edge tier
# Lower-quality evidence pays a higher BIC complexity penalty
_BIC_COMPLEXITY: dict[str, float] = {
    "Tier1_Interventional": 1.0,
    "Tier2_Convergent":     1.5,
    "Tier3_Provisional":    2.5,
    "provisional_virtual":  8.0,
}

BOOTSTRAP_N        = 50    # number of bootstrap samples
BOOTSTRAP_FRACTION = 0.50  # edge must survive this fraction to be retained
GAMMA_THRESHOLD    = 0.01  # minimum |ota_gamma| to count an edge in bootstraps


def compute_cross_regime_sensitivity(
    beta_matrix: dict[str, dict[str, Any]],
    gamma_matrix: dict[str, dict[str, float]],
    evidence_tier_per_gene: dict[str, str],
    gamma_tier: str = "Tier3_Provisional",
) -> dict[str, dict[str, float]]:
    """
    Compute the SCONE cross-regime sensitivity Γ_ij for each (gene, program) pair.

    Γ_ij ≈ |β_ij| × |γ_j_trait| × Q(β_tier) × Q(γ_tier)

    where Q() is the tier quality weight (0.1 – 1.0).

    Args:
        beta_matrix:            {gene: {program: {"beta": float, "evidence_tier": str} | None}}
        gamma_matrix:           {program: {trait: float}} — from PROVISIONAL_GAMMAS
        evidence_tier_per_gene: {gene: str} — best tier per gene
        gamma_tier:             Evidence tier for the γ estimates

    Returns:
        {gene: {program: gamma_ij_sensitivity_score}}
    """
    q_gamma = _TIER_QUALITY.get(gamma_tier, 0.5)
    sensitivity: dict[str, dict[str, float]] = {}

    for gene, prog_betas in beta_matrix.items():
        gene_tier  = evidence_tier_per_gene.get(gene, "provisional_virtual")
        q_beta_max = _TIER_QUALITY.get(gene_tier, 0.1)
        sensitivity[gene] = {}

        for program, beta_entry in prog_betas.items():
            if beta_entry is None:
                sensitivity[gene][program] = 0.0
                continue

            # Extract β and its tier
            if isinstance(beta_entry, dict):
                beta_val  = beta_entry.get("beta") or 0.0
                beta_tier = beta_entry.get("evidence_tier", gene_tier)
            else:
                beta_val  = float(beta_entry) if beta_entry is not None else 0.0
                beta_tier = gene_tier

            q_beta = _TIER_QUALITY.get(beta_tier, q_beta_max)

            # Max |γ| across traits for this program
            # Supports both {trait: float} and {trait: dict} gamma_matrix shapes.
            prog_gammas = gamma_matrix.get(program, {})
            max_gamma   = max(
                (abs(v.get("gamma", 0.0) if isinstance(v, dict) else v)
                 for v in prog_gammas.values()),
                default=0.0,
            )

            gamma_ij = abs(beta_val) * max_gamma * q_beta * q_gamma
            sensitivity[gene][program] = round(gamma_ij, 6)

    return sensitivity


def polybic_score(
    ota_gamma: float,
    evidence_tier: str,
    n_samples: int = 1000,
) -> float:
    """
    PolyBIC score for a proposed causal edge.

    BIC = log|γ_ota| - (k/2) × log(n)

    where k = complexity cost (tier-dependent) and n = effective sample size.

    Higher score = stronger evidence for edge existence.

    Args:
        ota_gamma:      Ota composite γ_{gene→trait}
        evidence_tier:  Evidence tier of the dominant β/γ
        n_samples:      Effective GWAS sample size (default 1000 as conservative minimum)

    Returns:
        BIC score (higher = better supported edge)
    """
    if ota_gamma == 0.0:
        return float("-inf")
    k = _BIC_COMPLEXITY.get(evidence_tier, 5.0)
    log_likelihood = math.log(abs(ota_gamma) + 1e-10)
    penalty = (k / 2.0) * math.log(max(n_samples, 2))
    return log_likelihood - penalty


def bootstrap_edge_confidence(
    gene: str,
    beta_matrix_row: dict[str, Any],
    gamma_matrix: dict[str, dict[str, float]],
    ota_gamma_fn: Any,
    n_bootstrap: int = BOOTSTRAP_N,
    noise_scale: float = 0.15,
) -> dict[str, float]:
    """
    Bootstrap confidence for each (gene, program→trait) edge.

    Adds Gaussian noise to β values (proportional to tier uncertainty) and
    recomputes Ota γ n_bootstrap times.  Returns the fraction of samples
    in which |ota_gamma| > GAMMA_THRESHOLD.

    Args:
        gene:             Gene symbol
        beta_matrix_row:  {program: {"beta": float, "evidence_tier": str} | None}
        gamma_matrix:     {program: {trait: float}}
        ota_gamma_fn:     `compute_ota_gamma` function from ota_gamma_estimation
        n_bootstrap:      Number of bootstrap samples
        noise_scale:      Relative noise added per tier (0.15 = 15% of |β|)

    Returns:
        {trait: confidence_fraction}  — fraction of bootstraps where edge survived
    """
    # Collect all traits
    traits: set[str] = set()
    for prog_gammas in gamma_matrix.values():
        traits.update(prog_gammas.keys())
    if not traits:
        return {}

    edge_counts: dict[str, int] = {t: 0 for t in traits}

    for _ in range(n_bootstrap):
        # Perturb β values by tier-scaled noise
        noisy_betas: dict[str, Any] = {}
        for prog, entry in beta_matrix_row.items():
            if entry is None:
                noisy_betas[prog] = None
                continue
            if isinstance(entry, dict):
                beta_val  = entry.get("beta") or 0.0
                beta_tier = entry.get("evidence_tier", "provisional_virtual")
            else:
                beta_val  = float(entry)
                beta_tier = "provisional_virtual"

            # Noise proportional to tier uncertainty (Tier1 = 15%, Virtual = 60%)
            tier_noise_mult = {
                "Tier1_Interventional": 0.15,
                "Tier2_Convergent":     0.25,
                "Tier3_Provisional":    0.40,
                "provisional_virtual":  0.60,
            }.get(beta_tier, 0.40)

            noise = random.gauss(0, abs(beta_val) * tier_noise_mult + 1e-6)
            noisy_betas[prog] = {
                "beta":          beta_val + noise,
                "evidence_tier": beta_tier,
            }

        # Build tier-matched γ format for compute_ota_gamma
        # Handles both {trait: float} and {trait: dict} gamma_matrix shapes.
        for trait in traits:
            trait_gammas: dict[str, dict] = {}
            for prog, prog_gammas in gamma_matrix.items():
                g_entry = prog_gammas.get(trait, 0.0)
                if isinstance(g_entry, dict):
                    trait_gammas[prog] = g_entry  # already full dict with evidence_tier
                else:
                    trait_gammas[prog] = {"gamma": float(g_entry or 0.0), "evidence_tier": "Tier3_Provisional"}

            try:
                result = ota_gamma_fn(
                    gene=gene,
                    trait=trait,
                    beta_estimates=noisy_betas,
                    gamma_estimates=trait_gammas,
                )
                if abs(result.get("ota_gamma", 0.0)) > GAMMA_THRESHOLD:
                    edge_counts[trait] += 1
            except Exception:
                pass

    return {t: round(c / n_bootstrap, 3) for t, c in edge_counts.items()}


def apply_scone_reweighting(
    gene_gamma_records: dict[str, dict],
    sensitivity_matrix: dict[str, dict[str, float]],
    bic_scores: dict[str, float],
    bootstrap_confidence: dict[str, dict[str, float]],
    min_bootstrap_confidence: float = BOOTSTRAP_FRACTION,
    anchor_gene_set: set[str] | None = None,
) -> dict[str, dict]:
    """
    Apply SCONE reweighting to Ota γ edge records.

    Modifies each record's `ota_gamma` by multiplying with the SCONE confidence
    composite:  γ_scone = γ_ota × Γ_sensitivity × sigmoid(BIC) × bootstrap_confidence

    Edges that fail the bootstrap confidence threshold are downgraded to
    provisional_virtual and their γ is zeroed out.

    Disease anchor genes are exempt from bootstrap zeroing — they have established
    genetic/clinical evidence even when Perturb-seq β is provisional/virtual, so
    the low bootstrap confidence reflects β uncertainty, not false-positive risk.

    Args:
        gene_gamma_records:   {gene__trait: rec} from causal_discovery_agent
        sensitivity_matrix:   {gene: {program: Γ_ij}} from compute_cross_regime_sensitivity
        bic_scores:           {gene__trait: BIC_score}
        bootstrap_confidence: {gene: {trait: confidence}}
        min_bootstrap_confidence: edges below this are zeroed (anchor genes exempt)
        anchor_gene_set:      Set of disease anchor gene symbols exempt from bootstrap filter

    Returns:
        Updated gene_gamma_records dict (same structure, modified in place copy)
    """
    import math
    updated = {}
    _anchors = anchor_gene_set or set()

    for key, rec in gene_gamma_records.items():
        gene  = rec["gene"]
        trait = rec["trait"]
        ota   = rec["ota_gamma"]

        # Bootstrap confidence filter
        bc = bootstrap_confidence.get(gene, {}).get(trait, 1.0)
        if bc < min_bootstrap_confidence:
            # Anchor genes are exempt: their disease association is well-validated
            # even when virtual β is noisy. Pass through with explicit flag.
            if gene in _anchors:
                # Continue to SCONE reweighting below (do not zero out)
                pass
            else:
                # Demote below threshold — zero the edge and mark provisional
                new_rec = dict(rec)
                new_rec["ota_gamma"]     = 0.0
                new_rec["dominant_tier"] = "provisional_virtual"
                new_rec["scone_confidence"] = bc
                new_rec["scone_flags"] = ["bootstrap_rejected"]
                updated[key] = new_rec
                continue

        # Aggregate sensitivity — max Γ_ij across programs for this gene
        gene_sens = sensitivity_matrix.get(gene, {})
        max_sens  = max(gene_sens.values(), default=1.0) or 1.0
        # Normalize to [0.5, 1.0] range — never zero a strong edge entirely
        sens_factor = 0.5 + 0.5 * min(max_sens / (max_sens + 0.1), 1.0)

        # Anchor genes bypass the BIC multiplicative penalty.
        # Their disease association is validated by independent GWAS/clinical
        # evidence; the virtual β reflects missing Perturb-seq data, not an
        # uncertain causal claim.  Apply only sensitivity × bootstrap scaling.
        if gene in _anchors and rec.get("dominant_tier") == "provisional_virtual":
            scone_gamma = ota * sens_factor * bc
            bic_factor  = float("nan")  # not applicable
            scone_flags = ["anchor_scone_exempt"]
        else:
            # BIC-based confidence boost (sigmoid of BIC)
            bic = bic_scores.get(key, 0.0)
            bic_factor = 1.0 / (1.0 + math.exp(-bic * 0.5))  # sigmoid
            scone_gamma = ota * bic_factor * sens_factor * bc
            scone_flags = []

        new_rec = dict(rec)
        new_rec["ota_gamma"]         = round(scone_gamma, 6)
        new_rec["ota_gamma_raw"]     = round(ota, 6)
        new_rec["scone_confidence"]  = bc
        new_rec["scone_bic_factor"]  = round(bic_factor, 4) if not math.isnan(bic_factor) else None
        new_rec["scone_sensitivity"] = round(max_sens, 4)
        new_rec["scone_flags"]       = scone_flags
        if bc < min_bootstrap_confidence and gene in _anchors:
            new_rec["scone_flags"].append("anchor_bootstrap_exempt")
        if bc < 0.7:
            new_rec["scone_flags"].append("low_bootstrap_confidence")
        updated[key] = new_rec

    return updated
