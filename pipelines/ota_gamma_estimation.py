"""
ota_gamma_estimation.py — Ota framework Layer 3: program → trait γ estimation.

The Ota et al. (Nature 2026) framework:
  γ_{gene→trait} = Σ_P (β_{gene→P} × γ_{P→trait})

This pipeline estimates γ_{P→trait}: the causal effect of a cellular program
on a disease trait.

γ estimation sources (in decreasing reliability):
  1. Live OT L2G score proxy: mean locus→gene score across program genes
  2. Hypergeometric GWAS enrichment: program gene overlap with GWAS hits
  3. Provisional fallback: zero (no evidence)
"""
from __future__ import annotations

import logging
import math
import sys
from pathlib import Path
from typing import Any, TypedDict

logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.evidence import GeneTraitAssociation
from config.scoring_thresholds import SLDSC_BYSTANDER_WEIGHT, SLDSC_TAU_SIGNIFICANT_P

# Minimum mean Open Targets L2G score to accept as genetic evidence for γ estimation.
# L2G (Locus-to-Gene; Mountjoy et al., Nature Genetics 2021) scores represent the
# posterior probability that a gene is the causal target at a GWAS locus, trained on
# fine-mapped variants and functional features. The paper recommends ≥0.1 for high-
# confidence prediction; we use ≥0.05 to be inclusive in loci with distributed signal
# across multiple genes (common in complement and lipid GWAS regions). Genes below
# this threshold fall through to provisional γ estimation from LINCS or structural data.
_OT_GENETIC_SCORE_MIN = 0.05


# ---------------------------------------------------------------------------
# TypedDicts — define the expected shape of beta/gamma estimate dicts so
# downstream code (compute_ota_gamma, ota_gamma_calculator) doesn't rely
# on defensive isinstance checks.
# ---------------------------------------------------------------------------

class BetaInfo(TypedDict, total=False):
    """Expected shape of a single beta estimate from ota_beta_estimation."""
    beta: float | None         # NaN = no data; None = not estimated
    beta_se: float | None
    ci_lower: float | None
    ci_upper: float | None
    evidence_tier: str
    data_source: str
    gene: str
    program: str
    tier_used: int


class GammaInfo(TypedDict, total=False):
    """Expected shape of a single gamma estimate from estimate_gamma."""
    gamma: float
    gamma_se: float | None
    evidence_tier: str
    data_source: str
    program: str
    trait: str
    pmid: str


# PROVISIONAL_GAMMAS removed — all γ values must be data-derived via
# estimate_gamma_live (aggregated OT L2G) or compute_cnmf_gamma
# (GWAS enrichment). No hardcoded fallbacks.
PROVISIONAL_GAMMAS: dict[tuple[str, str], dict] = {}


def compute_cnmf_gamma(
    program_gene_set: set[str],
    gwas_hits: set[str],
    program_id: str,
    trait: str,
    n_genome_genes: int = 20_000,
) -> dict | None:
    """
    Compute γ_{program→disease} for a data-driven cNMF program via GWAS enrichment.

    Uses a one-sided Fisher's exact test: are GWAS-implicated genes over-represented
    in the cNMF program gene set relative to the genome background?

    This is NOT circular: cNMF programs are defined from expression data (Perturb-seq /
    scRNA-seq), independent of GWAS. GWAS hits are only used to *score* the programs,
    not to *define* them. Compare to PROVISIONAL_GAMMAS which use MSigDB Hallmarks
    (curated from the literature using the same disease knowledge as GWAS).

    Args:
        program_gene_set: Genes in the cNMF program (from cnmf_runner gene_loadings)
        gwas_hits:        Genes implicated by GWAS (finemapped or p < 5e-8 nearest gene)
        program_id:       Identifier for this program
        trait:            Disease/trait name
        n_genome_genes:   Background genome size for Fisher's test

    Returns:
        {"gamma", "p_enrichment", "odds_ratio", "n_overlap", "evidence_tier", "data_source"}
        or None if insufficient overlap (n_overlap < 2).
    """
    if not program_gene_set or not gwas_hits:
        return None

    overlap   = program_gene_set & gwas_hits
    n_overlap = len(overlap)
    if n_overlap < 2:
        return None

    # Fisher's exact test (one-sided: enrichment)
    a = n_overlap                              # program ∩ GWAS
    b = len(program_gene_set) - n_overlap      # program \ GWAS
    c = len(gwas_hits) - n_overlap             # GWAS \ program
    d = n_genome_genes - a - b - c             # neither

    if b < 0 or c < 0 or d < 0:
        return None

    try:
        import math as _m
        # Use hypergeometric approximation for p-value
        # P(X >= k) where X ~ Hypergeometric(N, K, n)
        # N=total genes, K=GWAS hits, n=program size, k=overlap
        N = n_genome_genes
        K = len(gwas_hits)
        n = len(program_gene_set)
        k = n_overlap

        # Log-probability using log-factorials (Stirling approximation via lgamma)
        def _log_hyper_pmf(k: int, N: int, K: int, n: int) -> float:
            from math import lgamma
            if k < 0 or k > min(K, n):
                return float("-inf")
            return (
                lgamma(K + 1) - lgamma(k + 1) - lgamma(K - k + 1)
                + lgamma(N - K + 1) - lgamma(n - k + 1) - lgamma(N - K - n + k + 1)
                - lgamma(N + 1) + lgamma(n + 1) + lgamma(N - n + 1)
            )

        # P-value = P(X >= k) = sum_{j=k}^{min(K,n)} P(X=j)
        log_probs = [_log_hyper_pmf(j, N, K, n) for j in range(k, min(K, n) + 1)]
        max_lp = max(log_probs)
        p_enrich = min(1.0, _m.exp(max_lp) * sum(_m.exp(lp - max_lp) for lp in log_probs))

        # Odds ratio (small-sample)
        odds_ratio = (a * d) / ((b + 1e-6) * (c + 1e-6))

        # γ = clamp(-log10(p) × sign(OR - 1), 0, 1) — positive enrichment only
        if odds_ratio <= 1.0 or p_enrich >= 0.1:
            return None  # no meaningful enrichment

        gamma_raw = min(1.0, -_m.log10(max(p_enrich, 1e-10)) / 5.0)  # saturates at p=1e-5

        tier = "Tier3_Provisional" if p_enrich < 0.05 else "provisional_virtual"

        return {
            "gamma":          round(gamma_raw, 4),
            "p_enrichment":   round(p_enrich, 6),
            "odds_ratio":     round(odds_ratio, 3),
            "n_overlap":      n_overlap,
            "overlap_genes":  sorted(overlap)[:10],
            "evidence_tier":  tier,
            "data_source":    "cNMF_GWAS_enrichment",
            "program":        program_id,
            "trait":          trait,
        }
    except (ValueError, ArithmeticError, OverflowError) as _e:
        logger.warning("hypergeometric enrichment math error (program=%r): %s", program_id, _e)
        return None



def estimate_gamma(
    program: str,
    trait: str,
    program_gene_set: set[str] | None = None,
    efo_id: str | None = None,
) -> dict:
    """
    Estimate γ_{program→trait} from best available evidence.

    Priority:
      1. Live OT L2G score proxy
      2. Zero (no evidence)
    """
    # Live OT genetic evidence
    live = estimate_gamma_live(program, trait, program_gene_set, efo_id)
    if live is not None:
        return live

    # No evidence found — gamma=None signals "unknown" to compute_ota_gamma,
    # which excludes programs with None gamma from the OTA sum.
    return {
        "gamma":         None,
        "evidence_tier": "no_evidence",
        "data_source":   "none",
        "program":       program,
        "trait":         trait,
    }


def compute_ota_gamma(
    gene: str,
    trait: str,
    beta_estimates: dict[str, dict],
    gamma_estimates: dict[str, dict],
    skip_programs: frozenset[str] | None = None,
) -> dict:
    """
    Compute the Ota composite γ_{gene→trait} = Σ_P (β_{gene→P} × γ_{P→trait}).

    Args:
        gene:             Gene symbol
        trait:            Disease trait
        beta_estimates:   {program → {beta, evidence_tier, ...}}
        gamma_estimates:  {program → {gamma, evidence_tier, ...}}
        skip_programs:    Optional frozenset of program IDs to exclude from the sum.
                          Pass HALLMARK_PROGRAM_IDS | {"__protein_channel__"} to
                          restrict the OTA sum to cNMF + state-transition programs only.

    Returns:
        {
            "gene": str,
            "trait": str,
            "ota_gamma": float,
            "program_contributions": list[{program, beta, gamma, contribution}],
            "dominant_tier": str,
            "n_programs_contributing": int,
        }
    """
    contributions = []
    ota_gamma = 0.0
    tiers_used = []

    for program in beta_estimates:
        if skip_programs and program in skip_programs:
            continue
        beta_info = beta_estimates[program]
        if beta_info is None:
            continue
        gamma_info = gamma_estimates.get(program, {})

        beta_val  = beta_info.get("beta") if isinstance(beta_info, dict) else beta_info
        gamma_val = gamma_info.get("gamma") if isinstance(gamma_info, dict) else (float(gamma_info) if gamma_info is not None else None)

        if gamma_val is None and isinstance(beta_info, dict):
            gamma_val = beta_info.get("program_gamma")

        if beta_val is None or gamma_val is None:
            continue
        # NaN beta means "no data" — skip rather than contribute 0 × γ,
        # which would silently erase program→trait evidence.
        if isinstance(beta_val, float) and math.isnan(beta_val):
            continue

        # Cap β magnitude at ±2.0: values beyond this indicate non-specific
        # global perturbation effects (e.g. ribosomal KO, global stress response)
        # rather than program-specific causal effects.
        beta_val = max(-2.0, min(2.0, float(beta_val)))

        # γ SE-weighted contribution: programs with high GWAS uncertainty (few L2G
        # genes, noisy enrichment) are down-weighted relative to well-grounded ones.
        # Weight = 1/(1 + gamma_se): ranges from ~1.0 (se≈0, tight L2G cluster) to
        # ~0.5 (se=gamma, dispersed signal). Preserves direction; just scales magnitude.
        gamma_se = gamma_info.get("gamma_se") if isinstance(gamma_info, dict) else None
        if gamma_se is None:
            gamma_se = abs(gamma_val) * 0.50  # fallback: 50% SE assumed when unknown
        gamma_weight = 1.0 / (1.0 + float(gamma_se))

        # S-LDSC bystander filter: programs with τ ≤ 0 (no GWAS heritability enrichment)
        # are reactive/bystander programs — cellular response to disease, not causal drivers.
        # When S-LDSC data is available (gamma_source_type="s_ldsc"), programs with
        # non-positive τ already return None from gamma_loader and are skipped above.
        # For programs with OT L2G / h5ad DEG gamma (no S-LDSC), check tau if present.
        _tau = gamma_info.get("tau") if isinstance(gamma_info, dict) else None
        _gamma_source = gamma_info.get("gamma_source_type", "") if isinstance(gamma_info, dict) else ""
        _is_bystander = False
        if _tau is not None and _gamma_source == "s_ldsc" and float(_tau) <= 0:
            # Should have been filtered in gamma_loader, but guard defensively
            gamma_weight *= SLDSC_BYSTANDER_WEIGHT
            _is_bystander = True
        elif _tau is not None and _gamma_source != "s_ldsc" and float(_tau) <= 0:
            # Non-S-LDSC gamma for a program where we happen to know τ ≤ 0 — down-weight
            gamma_weight *= SLDSC_BYSTANDER_WEIGHT
            _is_bystander = True

        _tau_p = gamma_info.get("tau_p") if isinstance(gamma_info, dict) else None
        _heritability_significant = (
            _tau is not None and float(_tau) > 0
            and _tau_p is not None and float(_tau_p) < SLDSC_TAU_SIGNIFICANT_P
        )

        contribution = beta_val * gamma_val * gamma_weight
        ota_gamma += contribution
        tiers_used.append(beta_info.get("evidence_tier", "unknown") if isinstance(beta_info, dict) else "unknown")

        if abs(contribution) > 1e-6:  # only include non-trivial contributions
            contributions.append({
                "program":                  program,
                "beta":                     round(beta_val, 4),
                "gamma":                    round(gamma_val, 4),
                "gamma_weight":             round(gamma_weight, 4),
                "contribution":             round(contribution, 4),
                "beta_tier":                beta_info.get("evidence_tier") if isinstance(beta_info, dict) else None,
                "gamma_source":             gamma_info.get("data_source") if isinstance(gamma_info, dict) else None,
                "tau":                      round(_tau, 4) if _tau is not None else None,
                "heritability_significant": _heritability_significant,
                "bystander_discounted":     _is_bystander,
            })

    # Sort by absolute contribution
    contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)

    # Dominant tier = lowest (most reliable) tier
    tier_priority = {
        "Tier1_Interventional": 1,
        "Tier2_Convergent": 2,
        "Tier3_Provisional": 3,
        "moderate_transferred": 3,
        "moderate_grn": 3,
        "provisional_virtual": 4,
    }
    if tiers_used:
        dominant_tier = min(tiers_used, key=lambda t: tier_priority.get(t, 99))
    else:
        dominant_tier = "provisional_virtual"

    return {
        "gene":                    gene,
        "trait":                   trait,
        "ota_gamma":               round(ota_gamma, 4),
        "program_contributions":   contributions[:10],  # top 10
        "top_programs":            {c["program"]: c["contribution"] for c in contributions[:5]},
        "dominant_tier":           dominant_tier,
        "n_programs_contributing": len(contributions),
        "n_programs_total":        len(beta_estimates),
    }


def build_gamma_matrix(
    programs: list[str],
    traits: list[str],
) -> dict[str, dict[str, dict]]:
    """
    Build the full γ_{program→trait} matrix.

    Args:
        programs:  List of cNMF program names
        traits:    List of disease/trait names

    Returns:
        Nested dict: {program → {trait → gamma_dict}}
    """
    matrix: dict[str, dict[str, dict]] = {}
    for program in programs:
        matrix[program] = {}
        for trait in traits:
            matrix[program][trait] = estimate_gamma(program, trait)
    return matrix


# ---------------------------------------------------------------------------
# Convenience: CAD γ estimation
# ---------------------------------------------------------------------------

def estimate_cad_gammas() -> dict:
    """
    Build γ matrix for CAD-relevant programs × CAD traits.
    Returns provisional estimates from PROVISIONAL_GAMMAS.
    """
    from mcp_servers.burden_perturb_server import get_cnmf_program_info

    programs_info = get_cnmf_program_info()
    programs = programs_info["programs"]
    traits = ["CAD", "LDL-C", "RA", "SLE", "MS", "T1D", "CRP"]

    matrix = build_gamma_matrix(programs=programs, traits=traits)

    # Summarize non-zero, non-None entries
    nonzero = [
        {"program": p, "trait": t, "gamma": (matrix[p][t] or {}).get("gamma")}
        for p in programs for t in traits
        if (matrix[p][t] or {}).get("gamma") is not None and (matrix[p][t] or {}).get("gamma") != 0.0
    ]
    nonzero.sort(key=lambda x: abs(x["gamma"]), reverse=True)

    return {
        "programs":   programs,
        "traits":     traits,
        "matrix":     matrix,
        "top_program_trait_pairs": nonzero[:15],
        "note":       "Provisional γ values. Refine with live OT L2G data after GWAS download.",
    }


def estimate_gamma_live(
    program: str,
    trait: str,
    program_gene_set: set[str] | None,
    efo_id: str | None,
) -> dict | None:
    """
    Live γ_{program→trait} from aggregated Open Targets **L2G** (locus→gene) scores.

    Replaces the hardcoded PROVISIONAL_GAMMAS table with a data-driven proxy:

      γ_proxy = mean( best L2G per gene across OT GWAS studies for the EFO trait )
                × 0.65

    L2G is queried via ``gwas_genetics_server`` (credibleSets → l2GPredictions), not
    via disease ``associatedTargets`` GraphQL. Colocalisation is not used.


    Args:
        program:           cNMF program name
        trait:             Disease/trait name (used for display)
        program_gene_set:  Set of gene symbols defining the program
        efo_id:            EFO trait ID for OT queries (required)

    Returns:
        gamma estimate dict (same schema as estimate_gamma()) or None if
        data insufficient / efo_id not provided.
    """
    if not efo_id or not program_gene_set:
        return None

    # Live γ from aggregated L2G (locus→gene) across OT GWAS studies — no coloc,
    # no disease.associatedTargets genetic_association scores.
    try:
        from mcp_servers.gwas_genetics_server import aggregate_l2g_scores_for_program_genes
        genes = sorted(program_gene_set)  # sorted for deterministic query order
        ot_result = aggregate_l2g_scores_for_program_genes(efo_id, genes)
        mean_score = float(ot_result.get("mean_genetic_score") or ot_result.get("mean_l2g_score") or 0.0)
        n_with_data = int(ot_result.get("n_genes_with_data") or 0)
    except Exception:
        return None

    if mean_score < _OT_GENETIC_SCORE_MIN or n_with_data == 0:
        return None

    # Scale L2G [0,1] to PROVISIONAL_GAMMAS range (same scaling as former OT score proxy)
    gamma_value = round(mean_score * 0.65, 4)
    # SE: proportional to inverse of n_with_data (more genes → tighter estimate)
    gamma_se = round(gamma_value * 0.5 / max(n_with_data, 1) ** 0.5, 4)

    evidence_tier = "Tier3_Provisional"
    data_source   = f"OT_L2G_mean_{n_with_data}_genes"

    return {
        "gamma":         gamma_value,
        "gamma_se":      gamma_se,
        "evidence_tier": evidence_tier,
        "data_source":   data_source,
        "program":       program,
        "trait":         trait,
        "note":          (
            f"Live estimate: mean Open Targets L2G score across "
            f"{n_with_data}/{len(genes)} program genes. "
            "Replaces provisional hardcoded value when L2G data available."
        ),
    }


def compute_ota_gamma_with_uncertainty(
    gene: str,
    trait: str,
    beta_estimates: dict[str, dict],
    gamma_estimates: dict[str, dict],
    genebayes_result: dict | None = None,
    skip_programs: frozenset[str] | None = None,
) -> dict:
    """
    Compute Ota composite γ with 95% CI and optional GeneBayes grounding.

    Propagates β and γ uncertainties via the delta method, and optionally
    performs a Bayesian update using direct GeneBayes (Dataset 1) posteriors.

    1. Causal Logic: γ_ota = Σ_P (β_P × γ_P)
    2. Variance: Var(γ_ota) ≈ Σ_P [γ_P² σ²_β_P + β_P² σ²_γ_P]
    3. Bayesian Fusion (Dataset 1 + Dataset 2):
       If GeneBayes direct γ is provided, we treat it as the prior and the
       Ota mechanistic sum as the likelihood evidence.

    Returns:
        Dict with ota_gamma, ota_gamma_sigma, and fused_gamma (if GB provided).
    """
    _TIER_SIGMA_FRAC: dict[str, float] = {
        "Tier1_Interventional": 0.15,
        "Tier2_Convergent":     0.25,
        "Tier3_Provisional":    0.35,
        "moderate_transferred": 0.35,
        "moderate_grn":         0.40,
        "provisional_virtual":  0.70,
    }

    base = compute_ota_gamma(gene, trait, beta_estimates, gamma_estimates,
                             skip_programs=skip_programs)

    variance = 0.0
    for prog, beta_info in beta_estimates.items():
        if skip_programs and prog in skip_programs:
            continue
        if not isinstance(beta_info, dict):
            continue

        b = beta_info.get("beta")
        if b is None or (isinstance(b, float) and math.isnan(b)):
            continue
        b = float(b)

        # β uncertainty
        tier = beta_info.get("evidence_tier", "provisional_virtual")
        default_frac = _TIER_SIGMA_FRAC.get(tier, 0.50)
        s_b = beta_info.get("beta_sigma")
        if s_b is None:
            s_b = abs(b) * default_frac
        s_b = float(s_b)

        # γ value and uncertainty
        g_info = gamma_estimates.get(prog, {})
        if isinstance(g_info, dict):
            g_raw = g_info.get("gamma")
            if g_raw is None:
                continue
            g = float(g_raw)
            s_g = g_info.get("gamma_se")
            if s_g is None:
                s_g = abs(g) * 0.30
            s_g = float(s_g)
        else:
            if g_info is None:
                continue
            g = float(g_info)
            s_g = abs(g) * 0.30

        variance += g * g * s_b * s_b + b * b * s_g * s_g

    sigma_ota = math.sqrt(variance) if variance > 0 else 0.5 # wide prior if no data
    ota_gamma = base["ota_gamma"]

    # --- Dataset 1 Convergence: GeneBayes Grounding ---
    fused_gamma = ota_gamma
    sigma_fused = sigma_ota
    gb_note = ""

    if genebayes_result:
        # Weiner et al. (Nature 2023) GeneBayes posterior mean and SE
        gb_mean = genebayes_result.get("burden_beta")
        gb_se   = genebayes_result.get("burden_se")
        
        if gb_mean is not None and gb_se is not None and gb_se > 0:
            # Precision-weighted fusion (Bayesian Update)
            w_gb  = 1.0 / (gb_se ** 2)
            w_ota = 1.0 / (sigma_ota ** 2)
            
            fused_gamma = (gb_mean * w_gb + ota_gamma * w_ota) / (w_gb + w_ota)
            sigma_fused = math.sqrt(1.0 / (w_gb + w_ota))
            gb_note = f"Fused with GeneBayes direct prior (beta={gb_mean:.3f}, se={gb_se:.3f})"

    return {
        **base,
        "ota_gamma":         round(fused_gamma, 4),
        "ota_gamma_raw":     round(ota_gamma, 4),
        "ota_gamma_sigma":    round(sigma_fused, 4),
        "ota_gamma_ci_lower": round(fused_gamma - 1.96 * sigma_fused, 4),
        "ota_gamma_ci_upper": round(fused_gamma + 1.96 * sigma_fused, 4),
        "genebayes_grounded": bool(genebayes_result),
        "genebayes_note":     gb_note,
    }
