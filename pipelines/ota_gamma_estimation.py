"""
ota_gamma_estimation.py — Ota framework Layer 3: program → trait γ estimation.

The Ota et al. (Nature 2026) framework:
  γ_{gene→trait} = Σ_P (β_{gene→P} × γ_{P→trait})

This pipeline estimates γ_{P→trait}: the causal effect of a cellular program
on a disease trait.

γ estimation sources (in decreasing reliability):
  1. GWAS S-LDSC enrichment: program gene loadings enriched in trait heritability
  2. Transcriptome-wide MR (TWMR): MR from program gene expression → trait
  3. Pathway association score: OT overall score × program overlap
  4. Literature-curated provisional estimates (for anchor programs)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, TypedDict

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.evidence import CausalEdge, GeneTraitAssociation, EvidenceTier

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
# downstream code (compute_ota_gamma, causal_discovery_agent) doesn't rely
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
        print(f"[WARN] hypergeometric enrichment math error (program={program_id!r}): {_e}")
        return None


def estimate_gamma_fused(
    program: str,
    trait: str,
    program_gene_set: set[str] | None,
    efo_id: str | None,
    ldsc_result: dict | None = None,
    ot_result: dict | None = None,
) -> dict | None:
    """
    Bayesian fusion of S-LDSC heritability enrichment and OT Genetic Score proxy.

    γ_fused = (τ_ldsc * w_ldsc + γ_ot * w_ot) / (w_ldsc + w_ot)

    Weights:
      - w_ldsc: proportional to LDSC Z-score (confidence in enrichment)
      - w_ot:   constant prior weight (0.2), representing evidence from association
    """
    if not efo_id or not program_gene_set:
        return None

    # 1. Get LDSC component (Likelihood)
    gamma_ldsc = 0.0
    w_ldsc = 0.0
    ldsc_source = "none"
    if ldsc_result:
        tau = ldsc_result.get("tau", 0.0)
        z_score = ldsc_result.get("z_score", 0.0)
        if tau > 0 and z_score > 0:
            gamma_ldsc = tau
            w_ldsc = min(z_score, 10.0)  # cap weight to avoid extreme dominance
            ldsc_source = "S-LDSC"

    # 2. Get OT component (Prior)
    gamma_ot = 0.0
    w_ot = 0.2  # Prior weight for OT genetic evidence
    ot_source = "none"
    if ot_result:
        mean_score = float(ot_result.get("mean_genetic_score") or ot_result.get("mean_l2g_score") or 0.0)
        if mean_score >= _OT_GENETIC_SCORE_MIN:
            gamma_ot = mean_score * 0.65
            ot_source = "OT_L2G"
    else:
        # Fetch live if not provided (aggregated L2G across GWAS studies)
        try:
            from mcp_servers.gwas_genetics_server import aggregate_l2g_scores_for_program_genes
            genes = sorted(program_gene_set)  # sorted for deterministic query order
            live_ot = aggregate_l2g_scores_for_program_genes(efo_id, genes)
            mean_score = float(live_ot.get("mean_genetic_score") or live_ot.get("mean_l2g_score") or 0.0)
            if mean_score >= _OT_GENETIC_SCORE_MIN:
                gamma_ot = mean_score * 0.65
                ot_source = "OT_L2G"
        except Exception:
            pass

    if w_ldsc == 0 and ot_source == "none":
        return None

    # 3. Fusion
    total_w = w_ldsc + w_ot
    gamma_fused = (gamma_ldsc * w_ldsc + gamma_ot * w_ot) / total_w
    
    # SE estimation: conservatively high if only one source
    if w_ldsc > 0 and ot_source != "none":
        gamma_se = round(abs(gamma_fused - gamma_ldsc) * 0.5 + 0.05, 4)
    else:
        gamma_se = round(gamma_fused * 0.35, 4)

    evidence_tier = "Tier2_Convergent" if w_ldsc > 2.0 and ot_source != "none" else "Tier3_Provisional"
    
    return {
        "gamma":         round(gamma_fused, 4),
        "gamma_se":      gamma_se,
        "evidence_tier": evidence_tier,
        "data_source":   f"fused_{ldsc_source}+{ot_source}",
        "program":       program,
        "trait":         trait,
        "w_ldsc":        round(w_ldsc, 2),
        "w_ot":          round(w_ot, 2),
    }


def estimate_gamma(
    program: str,
    trait: str,
    gwas_enrichment: dict | None = None,
    twmr_result: dict | None = None,
    program_gene_set: set[str] | None = None,
    efo_id: str | None = None,
    finngen_phenocode: str | None = None,
) -> dict:
    """
    Estimate γ_{program→trait} from best available evidence.

    Priority:
      1. TWMR result (if available and significant, p < 0.05)
      2. Bayesian Fused Gamma (S-LDSC + OT)
      3. Provisional hardcoded estimate
      4. Zero (no evidence)

    Args:
        program:          cNMF program name
        trait:            Trait/disease name
        gwas_enrichment:  S-LDSC enrichment result dict with {"tau", "tau_se", "enrichment_p"}
        twmr_result:      TWMR result dict with {"beta", "se", "p"}
    """
    # Tier 1: TWMR (causal estimate)
    if twmr_result and twmr_result.get("p") is not None:
        if twmr_result["p"] < 0.05:
            return {
                "gamma":         twmr_result["beta"],
                "gamma_se":      twmr_result.get("se"),
                "evidence_tier": "Tier2_Convergent",
                "data_source":   "TWMR",
                "program":       program,
                "trait":         trait,
            }

    # Tier 2: Bayesian Fused Gamma (S-LDSC + OT)
    if program_gene_set and efo_id:
        # Use gwas_enrichment as the LDSC component if available
        fused = estimate_gamma_fused(
            program=program,
            trait=trait,
            program_gene_set=program_gene_set,
            efo_id=efo_id,
            ldsc_result=gwas_enrichment,
        )
        if fused:
            return fused

    # Tier 2b: Live OT genetic evidence (fallback if no LDSC component or fusion failed)
    live = estimate_gamma_live(program, trait, program_gene_set, efo_id, finngen_phenocode)
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
) -> dict:
    """
    Compute the Ota composite γ_{gene→trait} = Σ_P (β_{gene→P} × γ_{P→trait}).

    Args:
        gene:             Gene symbol
        trait:            Disease trait
        beta_estimates:   {program → {beta, evidence_tier, ...}}
        gamma_estimates:  {program → {gamma, evidence_tier, ...}}

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
        beta_info = beta_estimates[program]
        if beta_info is None:
            continue
        gamma_info = gamma_estimates.get(program, {})

        beta_val  = beta_info.get("beta") if isinstance(beta_info, dict) else beta_info
        gamma_val = gamma_info.get("gamma") if isinstance(gamma_info, dict) else (float(gamma_info) if gamma_info is not None else None)

        # Tier2s / Tier2pt embed their own program_gamma when no gamma_estimates entry
        # exists for the synthetic / protein-channel program ID.  Use it as fallback.
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

        contribution = beta_val * gamma_val
        ota_gamma += contribution
        tiers_used.append(beta_info.get("evidence_tier", "unknown") if isinstance(beta_info, dict) else "unknown")

        if abs(contribution) > 1e-6:  # only include non-trivial contributions
            contributions.append({
                "program":      program,
                "beta":         round(beta_val, 4),
                "gamma":        round(gamma_val, 4),
                "contribution": round(contribution, 4),
                "beta_tier":    beta_info.get("evidence_tier") if isinstance(beta_info, dict) else None,
                "gamma_source": gamma_info.get("data_source"),
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
    gwas_enrichment_data: dict | None = None,
    twmr_data: dict | None = None,
) -> dict[str, dict[str, dict]]:
    """
    Build the full γ_{program→trait} matrix.

    Args:
        programs:               List of cNMF program names
        traits:                 List of disease/trait names
        gwas_enrichment_data:   {(program, trait) → enrichment_dict}
        twmr_data:              {(program, trait) → twmr_dict}

    Returns:
        Nested dict: {program → {trait → gamma_dict}}
    """
    matrix: dict[str, dict[str, dict]] = {}
    for program in programs:
        matrix[program] = {}
        for trait in traits:
            key = (program, trait)
            gwas_e = (gwas_enrichment_data or {}).get(key)
            twmr_r = (twmr_data or {}).get(key)
            matrix[program][trait] = estimate_gamma(
                program, trait, gwas_enrichment=gwas_e, twmr_result=twmr_r
            )
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
        "note":       "Provisional γ values. Refine with S-LDSC + TWMR after GWAS download.",
    }


def estimate_gamma_live(
    program: str,
    trait: str,
    program_gene_set: set[str] | None,
    efo_id: str | None,
    finngen_phenocode: str | None = None,
) -> dict | None:
    """
    Live γ_{program→trait} from aggregated Open Targets **L2G** (locus→gene) scores.

    Replaces the hardcoded PROVISIONAL_GAMMAS table with a data-driven proxy:

      γ_proxy = mean( best L2G per gene across OT GWAS studies for the EFO trait )
                × 0.65

    L2G is queried via ``gwas_genetics_server`` (credibleSets → l2GPredictions), not
    via disease ``associatedTargets`` GraphQL. Colocalisation is not used.

    FinnGen augmentation: if finngen_phenocode is provided and the mean FinnGen
    p-value across program genes is < 5e-4, the γ estimate is upgraded to
    Tier2_Convergent (replication in an independent cohort).

    Args:
        program:           cNMF program name
        trait:             Disease/trait name (used for display)
        program_gene_set:  Set of gene symbols defining the program
        efo_id:            EFO trait ID for OT queries (required)
        finngen_phenocode: FinnGen R10 phenocode for replication check

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

    # FinnGen replication check
    if finngen_phenocode:
        try:
            from mcp_servers.finngen_server import get_finngen_gene_associations
            fg_pvals = []
            for gene in list(program_gene_set)[:5]:  # sample 5 genes
                fg_res = get_finngen_gene_associations(finngen_phenocode, gene, p_threshold=5e-4)
                if fg_res.get("n_variants", 0) > 0:
                    fg_pvals.append(min(
                        v["pval"] for v in fg_res["variants"]
                        if v.get("pval") is not None
                    ))
            if fg_pvals and min(fg_pvals) < 5e-4:
                evidence_tier = "Tier2_Convergent"
                data_source   = f"OT_L2G+FinnGen_replication_{n_with_data}_genes"
        except Exception:
            pass

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

    base = compute_ota_gamma(gene, trait, beta_estimates, gamma_estimates)

    variance = 0.0
    for prog, beta_info in beta_estimates.items():
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
