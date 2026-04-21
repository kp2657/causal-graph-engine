"""
scone_sensitivity.py — Soft-intervention Causal ON-line Ensemble (SCONE).

Implementation of the Reisach et al. (2024) SCONE framework for learning 
causal structure from two observational regimes with unknown soft interventions.

In this pipeline:
  - Regime 1 (Observational): Baseline population gene expression.
  - Regime 2 (Soft Intervention): eQTL-mediated expression shift (NES).
  - Target: Validating the causal edge beta_{gene -> program}.

Key components:
  1. Soft Intervention Strength: Derived from eQTL Normalized Effect Sizes.
  2. Cross-Regime Sensitivity (Gamma_ij): Measures the stability of the 
     gene -> program relationship when the gene is "nudged" by an eQTL.
  3. Consistency Check: Reconciles Perturb-seq beta with eQTL-mediated shifts.

References:
  - SCONE: Reisach et al. NeurIPS 2024 (github.com/v-i-s-h-n-u/scone)
"""
from __future__ import annotations

import math
import numpy as np
from typing import Any, Dict, List, Optional


_TIER_SENSITIVITY_MULTIPLIER: Dict[str, float] = {
    "Tier1_Interventional":  1.0,
    "Tier1_Perturb_seq":     1.0,
    "Tier2_Convergent":      0.8,
    "Tier2_eQTL_MR":         0.8,
    "Tier3_Provisional":     0.6,
    "provisional_virtual":   0.3,
}


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def compute_cross_regime_sensitivity(
    beta_or_matrix,
    gamma_or_gamma_matrix,
    tiers_or_shift=None,
):
    """
    Compute SCONE Cross-Regime Sensitivity.

    Supports two calling conventions:

    Scalar (original):
        compute_cross_regime_sensitivity(beta_perturbseq: float,
                                         eqtl_nes: float,
                                         observed_program_shift: float | None)
        → float   (single sensitivity value)

    Matrix (pipeline):
        compute_cross_regime_sensitivity(beta_matrix: dict,
                                         gamma_matrix: dict,
                                         evidence_tier_per_gene: dict)
        → dict    ({gene: {program: sensitivity_float}})

    Matrix algorithm:
        For each (gene, program) pair with non-None beta:
            sensitivity = sigmoid(beta_val) × tier_multiplier
        Missing pairs get 0.0.
    """
    if isinstance(beta_or_matrix, dict):
        beta_matrix = beta_or_matrix
        gamma_matrix = gamma_or_gamma_matrix
        tier_map: Dict[str, str] = tiers_or_shift or {}

        result: Dict[str, Dict[str, float]] = {}
        all_programs = set(gamma_matrix.keys())

        for gene, prog_dict in beta_matrix.items():
            tier = tier_map.get(gene, "provisional_virtual")
            tier_mult = _TIER_SENSITIVITY_MULTIPLIER.get(tier, 0.5)
            gene_sens: Dict[str, float] = {}
            for prog in all_programs:
                binfo = prog_dict.get(prog)
                if binfo is None:
                    gene_sens[prog] = 0.0
                    continue
                beta_val = binfo.get("beta") if isinstance(binfo, dict) else float(binfo)
                if beta_val is None:
                    gene_sens[prog] = 0.0
                    continue
                gene_sens[prog] = round(_sigmoid(float(beta_val)) * tier_mult, 4)
            result[gene] = gene_sens
        return result

    # --- Scalar path (original) ---
    beta_perturbseq = float(beta_or_matrix)
    eqtl_nes        = float(gamma_or_gamma_matrix)
    observed_program_shift = tiers_or_shift

    agreement = beta_perturbseq * eqtl_nes
    if observed_program_shift is not None:
        discrepancy = abs(agreement - float(observed_program_shift))
        return math.exp(-discrepancy)
    return _sigmoid(agreement)


def apply_scone_refinement(
    ota_gamma: float,
    beta_info: Dict[str, Any],
    eqtl_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Refine an Ota edge using SCONE cross-regime logic.
    
    Inputs:
      - ota_gamma: The composite effect size.
      - beta_info: Mechanistic effect from Perturb-seq.
      - eqtl_info: The "Soft Intervention" parameters (NES).
    """
    beta_val = beta_info.get("beta", 0.0)
    nes_val = eqtl_info.get("nes", 0.0) if eqtl_info else 0.0
    
    # Calculate Sensitivity
    # If no eQTL is present, sensitivity is neutral (0.5)
    if nes_val != 0:
        sensitivity = compute_cross_regime_sensitivity(beta_val, nes_val)
    else:
        sensitivity = 0.5
        
    # Reweight Gamma
    # High sensitivity (consistency across regimes) boosts the score
    # Low sensitivity (regime contradiction) penalizes the score
    refined_gamma = ota_gamma * (2 * sensitivity)
    
    return {
        "refined_gamma": round(refined_gamma, 6),
        "scone_sensitivity": round(sensitivity, 4),
        "regime_consistency": "high" if sensitivity > 0.7 else "low",
        "intervention_type": "soft_eqtl" if nes_val != 0 else "none"
    }


def polybic_score(
    ota_gamma: float,
    evidence_tier: str,
    n_samples: int = 1000,
) -> float:
    """
    Compute PolyBIC score for a single causal edge.

    Score = log(|gamma| + ε) - (k/2) * log(n_samples)

    Higher is better.  Tier 1 edges have lower complexity penalty (k=1) so
    the same gamma yields a higher score than provisional_virtual (k=10).

    Args:
        ota_gamma:     OTA gamma value for the edge
        evidence_tier: Evidence tier string
        n_samples:     Effective sample size (default 1000)

    Returns:
        Float PolyBIC score
    """
    k = {
        "Tier1_Interventional":  1.0,
        "Tier1_Perturb_seq":     1.0,
        "Tier2_Convergent":      2.0,
        "Tier2_eQTL_MR":         2.0,
        "Tier3_Provisional":     4.0,
        "provisional_virtual":  10.0,
    }.get(evidence_tier, 5.0)
    return math.log(abs(ota_gamma) + 1e-10) - (k / 2.0) * math.log(max(n_samples, 2))


def bootstrap_edge_confidence(
    gene: str,
    beta_matrix_row: Dict[str, Any],
    gamma_matrix: Dict[str, Any],
    ota_gamma_fn: Any,
    n_bootstrap: int = 30,
) -> Dict[str, float]:
    """
    Estimate bootstrap confidence interval for the OTA gamma of a gene.

    Resamples beta values with Gaussian noise (scaled to beta magnitude) and
    recomputes OTA gamma n_bootstrap times, returning mean and 95% CI.

    Args:
        gene:             Gene symbol
        beta_matrix_row:  Dict of {program_id: beta_info} for this gene
        gamma_matrix:     Program gamma estimates
        ota_gamma_fn:     Callable(gene, beta_row, gamma_estimates) → float
        n_bootstrap:      Number of bootstrap replicates

    Returns:
        {"mean": float, "ci_lower": float, "ci_upper": float, "cv": float}
    """
    gammas: List[float] = []
    for _ in range(n_bootstrap):
        # Perturb each beta by ±20% Gaussian noise
        noisy_row: Dict[str, Any] = {}
        for pid, binfo in beta_matrix_row.items():
            if isinstance(binfo, dict):
                b = binfo.get("beta")
                if b is not None:
                    noise = np.random.normal(0, abs(b) * 0.20 + 1e-6)
                    noisy_row[pid] = {**binfo, "beta": b + noise}
                else:
                    noisy_row[pid] = binfo
            else:
                noisy_row[pid] = binfo
        try:
            g = ota_gamma_fn(gene, noisy_row, gamma_matrix)
            if isinstance(g, (int, float)) and not math.isnan(g):
                gammas.append(float(g))
        except Exception:
            pass

    if not gammas:
        return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "cv": 1.0}

    gammas_arr = np.array(gammas)
    mean = float(np.mean(gammas_arr))
    ci_lower = float(np.percentile(gammas_arr, 2.5))
    ci_upper = float(np.percentile(gammas_arr, 97.5))
    cv = float(np.std(gammas_arr) / (abs(mean) + 1e-10))
    return {"mean": round(mean, 4), "ci_lower": round(ci_lower, 4),
            "ci_upper": round(ci_upper, 4), "cv": round(cv, 4)}


_BOOTSTRAP_CONFIDENCE_THRESHOLD = 0.50


def apply_scone_reweighting(
    gene_gamma_records: Dict[str, Any],
    sensitivity_matrix: Any,
    bic_scores: Dict[str, float],
    bootstrap_confidence: Dict[str, Any],
    anchor_gene_set: Optional[set] = None,
) -> Dict[str, Any]:
    """
    Reweight gene-gamma records using SCONE BIC scores and bootstrap confidence.

    Bootstrap rejection:
        Keys are expected in "GENE__TRAIT" format.
        If bootstrap_confidence[gene][trait] < 0.50, the record is rejected:
        ota_gamma → 0.0, "bootstrap_rejected" added to scone_flags.

    BIC reweighting (for non-rejected records):
        reweight_factor = sigmoid(bic_score)
        adjusted_gamma  = ota_gamma * clamp(reweight_factor, 0.1, 2.0)

    Anchor protection:
        Anchor genes are never reduced below 80% of original gamma.

    Args:
        gene_gamma_records:   Dict keyed by "GENE__TRAIT" strings
        sensitivity_matrix:   Output of compute_cross_regime_sensitivity (used for
                              future directional adjustment; currently informational)
        bic_scores:           {key: polybic_score} per record
        bootstrap_confidence: {gene: {trait: confidence_float}} or
                              {gene: {mean, ci_lower, ci_upper, cv}}
        anchor_gene_set:      Set of anchor gene symbols (optional, default empty)

    Returns:
        Adjusted gene_gamma_records dict (same keys, ota_gamma updated in-place copy)
    """
    if anchor_gene_set is None:
        anchor_gene_set = set()

    adjusted: Dict[str, Any] = {}
    for key, rec in gene_gamma_records.items():
        rec = dict(rec)  # shallow copy
        rec.setdefault("scone_flags", [])

        # Parse gene and trait from key ("GENE__TRAIT")
        if "__" in str(key):
            gene, trait = str(key).split("__", 1)
        else:
            gene = rec.get("gene", str(key))
            trait = rec.get("trait", "")

        ota_gamma = rec.get("ota_gamma", 0.0)

        # --- Bootstrap rejection check ---
        bc_gene = bootstrap_confidence.get(gene, {})
        # Support {trait: float} and {mean: float, ...} formats
        if isinstance(bc_gene, dict) and trait in bc_gene:
            confidence = float(bc_gene[trait])
        elif isinstance(bc_gene, dict) and "mean" in bc_gene:
            confidence = float(bc_gene.get("mean", 1.0))
        else:
            confidence = 1.0  # no bootstrap data → assume passes

        # Always record bootstrap confidence on the edge record
        rec["scone_confidence"] = round(confidence, 4)

        if confidence < _BOOTSTRAP_CONFIDENCE_THRESHOLD:
            rec["ota_gamma"] = 0.0
            rec["scone_flags"].append("bootstrap_rejected")
            adjusted[key] = rec
            continue

        # --- BIC reweighting ---
        bic = bic_scores.get(key, 0.0)
        reweight = max(0.1, min(_sigmoid(bic), 2.0))

        adjusted_gamma = ota_gamma * reweight

        # Protect anchors: floor at 80% of original
        if gene in anchor_gene_set:
            adjusted_gamma = max(adjusted_gamma, ota_gamma * 0.80)

        rec["ota_gamma"] = round(adjusted_gamma, 6)
        rec["scone_reweight_factor"] = round(reweight, 4)
        adjusted[key] = rec

    return adjusted


def polybic_selection(
    edges: List[Dict[str, Any]],
    n_samples: int,
) -> List[Dict[str, Any]]:
    """
    Implement SCONE PolyBIC selection to filter out over-complex causal structures.
    
    Penalty = (k/2) * log(n) where k is the edge complexity weighted by tier.
    """
    selected_edges = []
    for edge in edges:
        # Support both "refined_gamma" (apply_scone_refinement output) and
        # "ota_gamma" (causal_discovery_agent stores refined value under ota_gamma)
        gamma = edge.get("refined_gamma")
        if gamma is None or (isinstance(gamma, float) and math.isnan(gamma)):
            gamma = edge.get("ota_gamma", 0.0)
        if gamma is None or (isinstance(gamma, float) and math.isnan(gamma)):
            gamma = 0.0
        tier = edge.get("evidence_tier") or edge.get("dominant_tier") or "provisional_virtual"
        
        # Complexity cost: Tier 1 (Interventional) is "cheaper" than Virtual
        k = {
            "Tier1_Interventional": 1.0,
            "Tier2_Convergent": 2.0,
            "Tier3_Provisional": 4.0,
            "provisional_virtual": 10.0
        }.get(tier, 5.0)
        
        score = math.log(abs(gamma) + 1e-10) - (k / 2.0) * math.log(n_samples)
        
        if score > -10: # Significance threshold
            edge["polybic_score"] = round(score, 4)
            selected_edges.append(edge)
            
    return selected_edges
