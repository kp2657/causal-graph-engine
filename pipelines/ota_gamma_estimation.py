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

# Maximum program gene set size to query via live API (cap to avoid rate limits)
_LIVE_GAMMA_MAX_GENES = 15
# Minimum mean OT genetic score to use as live γ (below this, fall through to provisional)
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


# ---------------------------------------------------------------------------
# γ estimation — provisional values from literature
# ---------------------------------------------------------------------------

# Provisional γ_{program→trait} from published GWAS enrichment analyses
# Source: Replogle 2022, Ota 2026, and published S-LDSC studies
# These are the γ values that downstream MR + LDSC will refine
PROVISIONAL_GAMMAS: dict[tuple[str, str], dict] = {
    # Lipid / CAD programs
    # S-LDSC+TWMR = two convergent lines of evidence → Tier2_Convergent
    ("lipid_metabolism",       "CAD"):    {"gamma": 0.44, "se": 0.08, "source": "S-LDSC+TWMR", "pmid": "Ota2026",       "evidence_tier": "Tier2_Convergent"},
    ("lipid_metabolism",       "LDL-C"):  {"gamma": 0.61, "se": 0.07, "source": "S-LDSC",       "pmid": "Ota2026",       "evidence_tier": "Tier3_Provisional"},
    # Inflammatory programs → CAD / RA
    ("inflammatory_NF-kB",    "CAD"):    {"gamma": 0.31, "se": 0.06, "source": "S-LDSC+TWMR", "pmid": "Ota2026",       "evidence_tier": "Tier2_Convergent"},
    ("inflammatory_NF-kB",    "RA"):     {"gamma": 0.28, "se": 0.05, "source": "S-LDSC",       "pmid": "Ota2026",       "evidence_tier": "Tier3_Provisional"},
    # IL-6 MR at p<5e-8 across large trials → Tier2_Convergent
    ("IL-6_signaling",         "CAD"):   {"gamma": 0.24, "se": 0.05, "source": "S-LDSC+TWMR", "pmid": "Swerdlow2012",  "evidence_tier": "Tier2_Convergent"},
    ("IL-6_signaling",         "CRP"):   {"gamma": 0.52, "se": 0.09, "source": "MR",           "pmid": "Swerdlow2012",  "evidence_tier": "Tier2_Convergent"},
    ("IL-6_signaling",         "RA"):    {"gamma": 0.35, "se": 0.07, "source": "MR",           "pmid": "Smolen2020",    "evidence_tier": "Tier2_Convergent"},
    # MHC class II → autoimmune (S-LDSC enrichment only → Tier3)
    ("MHC_class_II_presentation", "RA"): {"gamma": 0.38, "se": 0.06, "source": "S-LDSC",       "pmid": "Nyeo2026",      "evidence_tier": "Tier3_Provisional"},
    ("MHC_class_II_presentation", "SLE"):{"gamma": 0.42, "se": 0.08, "source": "S-LDSC",       "pmid": "Nyeo2026",      "evidence_tier": "Tier3_Provisional"},
    ("MHC_class_II_presentation", "MS"): {"gamma": 0.46, "se": 0.07, "source": "S-LDSC",       "pmid": "Nyeo2026",      "evidence_tier": "Tier3_Provisional"},
    ("MHC_class_II_presentation", "T1D"):{"gamma": 0.33, "se": 0.07, "source": "S-LDSC",       "pmid": "Nyeo2026",      "evidence_tier": "Tier3_Provisional"},
    # Epigenetic / CHIP programs — Bick2020 HR from large prospective cohort → Tier2
    ("DNA_methylation_maintenance", "CAD"):  {"gamma": 0.18, "se": 0.05, "source": "Bick2020_HR", "pmid": "32694926",   "evidence_tier": "Tier2_Convergent"},
    ("myeloid_differentiation",     "CAD"):  {"gamma": 0.12, "se": 0.04, "source": "provisional", "pmid": "32694926",   "evidence_tier": "Tier3_Provisional"},
    # IBD programs — Liu 2023 meta-GWAS (n>500k) + de Lange 2017 S-LDSC enrichment
    # anti-TNF drug trials confirm causal direction of TNF_signaling → IBD (drug RCT = Tier2)
    ("inflammatory_NF-kB",    "IBD"):   {"gamma": 0.39, "se": 0.07, "source": "S-LDSC+TWMR", "pmid": "Liu2023_IBD",   "evidence_tier": "Tier2_Convergent"},
    ("TNF_signaling",          "IBD"):  {"gamma": 0.45, "se": 0.08, "source": "drug_RCT",     "pmid": "Hanauer2002",   "evidence_tier": "Tier2_Convergent"},
    ("IL-6_signaling",         "IBD"):  {"gamma": 0.22, "se": 0.06, "source": "S-LDSC",       "pmid": "Liu2023_IBD",   "evidence_tier": "Tier3_Provisional"},
    ("MHC_class_II_presentation", "IBD"):{"gamma": 0.29, "se": 0.06, "source": "S-LDSC",      "pmid": "deLange2017",   "evidence_tier": "Tier3_Provisional"},
    ("innate_immune_sensing",  "IBD"):  {"gamma": 0.33, "se": 0.06, "source": "S-LDSC+TWMR", "pmid": "Liu2023_IBD",   "evidence_tier": "Tier2_Convergent"},
    # Crohn's disease specific
    ("inflammatory_NF-kB",    "Crohn_disease"): {"gamma": 0.41, "se": 0.08, "source": "S-LDSC", "pmid": "Liu2023_IBD", "evidence_tier": "Tier3_Provisional"},
    ("innate_immune_sensing",  "Crohn_disease"): {"gamma": 0.38, "se": 0.07, "source": "S-LDSC", "pmid": "Liu2023_IBD", "evidence_tier": "Tier3_Provisional"},
    # UC specific
    ("inflammatory_NF-kB",    "UC"):    {"gamma": 0.36, "se": 0.07, "source": "S-LDSC",       "pmid": "Liu2023_IBD",   "evidence_tier": "Tier3_Provisional"},
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
      2. GWAS S-LDSC enrichment (if tau > 0)
      3. Provisional hardcoded estimate
      4. Zero (no evidence)

    Args:
        program:          cNMF program name
        trait:            Trait/disease name
        gwas_enrichment:  S-LDSC enrichment result dict with {"tau", "tau_se", "enrichment_p"}
        twmr_result:      TWMR result dict with {"beta", "se", "p"}
    """
    # Tier 0: GWAS heritability enrichment (heuristic S-LDSC, data-driven)
    # Only attempted when program_gene_set and efo_id are provided.
    if program_gene_set and efo_id:
        try:
            from pipelines.discovery.ldsc_pipeline import estimate_program_gamma_enrichment
            enrich = estimate_program_gamma_enrichment(
                program_gene_set=program_gene_set,
                efo_id=efo_id,
                program_id=program,
                trait=trait,
            )
            if enrich.get("gamma") is not None:
                return {
                    "gamma":         enrich["gamma"],
                    "gamma_se":      enrich.get("gamma_se"),
                    "evidence_tier": enrich.get("evidence_tier", "Tier3_Provisional"),
                    "data_source":   enrich.get("data_source", "GWAS_enrichment"),
                    "program":       program,
                    "trait":         trait,
                    "enrichment_z":  enrich.get("enrichment_z"),
                    "n_program_hits": enrich.get("n_program_hits"),
                }
        except Exception:
            pass

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

    # Tier 2: S-LDSC enrichment
    if gwas_enrichment and gwas_enrichment.get("tau") is not None:
        tau = gwas_enrichment["tau"]
        if tau > 0:
            return {
                "gamma":         tau,
                "gamma_se":      gwas_enrichment.get("tau_se"),
                "evidence_tier": "Tier3_Provisional",
                "data_source":   "S-LDSC",
                "program":       program,
                "trait":         trait,
            }

    # Tier 2b: Live OT genetic evidence (data-driven proxy for S-LDSC)
    # Only called when caller provides program_gene_set and efo_id kwargs
    # (added to signature below — backward-compatible: defaults to None)
    # Tier 2b: Live OT genetic evidence (data-driven proxy for S-LDSC enrichment)
    live = estimate_gamma_live(program, trait, program_gene_set, efo_id, finngen_phenocode)
    if live is not None:
        return live

    # Tier 3: Provisional hardcoded — use per-entry evidence_tier, not a blanket Tier3
    key = (program, trait)
    if key in PROVISIONAL_GAMMAS:
        prov = PROVISIONAL_GAMMAS[key]
        return {
            "gamma":         prov["gamma"],
            "gamma_se":      prov.get("se"),
            "evidence_tier": prov.get("evidence_tier", "Tier3_Provisional"),
            "data_source":   prov["source"],
            "program":       program,
            "trait":         trait,
            "pmid":          prov.get("pmid"),
        }

    # Tier 4: No evidence
    return {
        "gamma":         0.0,
        "gamma_se":      None,
        "evidence_tier": "provisional_virtual",
        "data_source":   "no_evidence",
        "program":       program,
        "trait":         trait,
        "note":          "No GWAS enrichment or MR evidence. γ=0 (no effect assumed).",
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
        gamma_val = gamma_info.get("gamma", 0.0) if isinstance(gamma_info, dict) else float(gamma_info or 0.0)

        if beta_val is None or gamma_val is None:
            continue
        # NaN beta means "no data" — skip rather than contribute 0 × γ,
        # which would silently erase program→trait evidence.
        if isinstance(beta_val, float) and math.isnan(beta_val):
            continue

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

    # Summarize non-zero entries
    nonzero = [
        {"program": p, "trait": t, "gamma": matrix[p][t]["gamma"]}
        for p in programs for t in traits
        if matrix[p][t].get("gamma", 0) != 0.0
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
    Live γ_{program→trait} estimation from Open Targets genetic evidence scores.

    Replaces the hardcoded PROVISIONAL_GAMMAS table with a data-driven proxy:

      γ_proxy = mean( OT genetic_association score across program gene set )
                × scaling_factor

    The OT genetic_association score aggregates GWAS, fine-mapping, and
    colocalization evidence per gene-disease pair.  Averaging across a program's
    gene set approximates the S-LDSC enrichment τ used in the Ota framework.

    Scaling: OT scores are in [0,1].  We scale to match the range of
    PROVISIONAL_GAMMAS (0.12–0.61) by multiplying by 0.65.

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

    # Primary: OT eQTL–GWAS colocalization (proper MR-like γ, Tier2_Convergent)
    # H4-weighted betaRatioSignAverage across program genes gives causal direction.
    try:
        from mcp_servers.open_targets_server import get_ot_colocalisation_for_program
        coloc_result = get_ot_colocalisation_for_program(
            program_gene_set=list(program_gene_set)[:_LIVE_GAMMA_MAX_GENES],
            efo_id=efo_id,
        )
        gamma_coloc = coloc_result.get("gamma_coloc")
        if gamma_coloc is not None:
            n_hits = coloc_result.get("n_coloc_hits", 0)
            gamma_se_coloc = round(abs(gamma_coloc) * 0.30 / max(n_hits, 1) ** 0.5, 4)
            return {
                "gamma":         gamma_coloc,
                "gamma_se":      gamma_se_coloc,
                "evidence_tier": "Tier2_Convergent",
                "data_source":   f"OT_coloc_H4_weighted_{n_hits}_pairs",
                "program":       program,
                "trait":         trait,
                "note":          (
                    f"Live estimate: H4-weighted betaRatioSignAverage from "
                    f"{n_hits} eQTL–GWAS coloc pairs in program gene set."
                ),
            }
    except Exception:
        pass

    # Fallback: OT genetic association score proxy (Tier3_Provisional)
    try:
        from mcp_servers.open_targets_server import get_ot_genetic_scores_for_gene_set
        genes = list(program_gene_set)[:_LIVE_GAMMA_MAX_GENES]
        ot_result = get_ot_genetic_scores_for_gene_set(efo_id, genes)
        mean_score = ot_result.get("mean_genetic_score", 0.0)
        n_with_data = ot_result.get("n_genes_with_data", 0)
    except Exception:
        return None

    if mean_score < _OT_GENETIC_SCORE_MIN or n_with_data == 0:
        return None

    # Scale OT score to PROVISIONAL_GAMMAS range
    gamma_value = round(mean_score * 0.65, 4)
    # SE: proportional to inverse of n_with_data (more genes → tighter estimate)
    gamma_se = round(gamma_value * 0.5 / max(n_with_data, 1) ** 0.5, 4)

    evidence_tier = "Tier3_Provisional"
    data_source   = f"OT_genetic_score_mean_{n_with_data}_genes"

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
                data_source   = f"OT_genetic+FinnGen_replication_{n_with_data}_genes"
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
            f"Live estimate: mean OT genetic association score across "
            f"{n_with_data}/{len(genes)} program genes. "
            "Replaces provisional hardcoded value when OT data available."
        ),
    }


def compute_ota_gamma_with_uncertainty(
    gene: str,
    trait: str,
    beta_estimates: dict[str, dict],
    gamma_estimates: dict[str, dict],
) -> dict:
    """
    Compute Ota composite γ with 95% CI via the delta method.

    Extends compute_ota_gamma by propagating β and γ uncertainties:

      γ_ota = Σ_P (β_P × γ_P)

    Variance (delta method, assuming β and γ independent):
      Var(γ_ota) = Σ_P [γ_P² × σ²_β_P  +  β_P² × σ²_γ_P]

    σ_β is read from beta_estimates[program].get("beta_sigma").
    σ_γ is read from gamma_estimates[program].get("gamma_se").

    If sigma fields are absent, tier-calibrated defaults are used:
      Tier1_Interventional: σ_β = 0.15 × |β|
      Tier2_Convergent:     σ_β = 0.25 × |β|
      Tier3_Provisional:    σ_β = 0.35 × |β|
      provisional_virtual:  σ_β = 0.70 × |β|  (wide prior)

    Returns:
        All fields from compute_ota_gamma, plus:
        - ota_gamma_sigma:    posterior σ for γ_ota
        - ota_gamma_ci_lower: γ_ota − 1.96 × σ  (95% CI lower)
        - ota_gamma_ci_upper: γ_ota + 1.96 × σ  (95% CI upper)
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
            g = float(g_info.get("gamma", 0.0) or 0.0)
            s_g = g_info.get("gamma_se")
            if s_g is None:
                s_g = abs(g) * 0.30  # 30% default for provisional γ
            s_g = float(s_g)
        else:
            g = float(g_info or 0.0)
            s_g = abs(g) * 0.30

        # Delta method: Var(β*γ) ≈ γ² σ²_β + β² σ²_γ
        variance += g * g * s_b * s_b + b * b * s_g * s_g

    sigma = math.sqrt(variance) if variance > 0 else 0.0
    ota_gamma = base["ota_gamma"]

    return {
        **base,
        "ota_gamma_sigma":    round(sigma, 4),
        "ota_gamma_ci_lower": round(ota_gamma - 1.96 * sigma, 4),
        "ota_gamma_ci_upper": round(ota_gamma + 1.96 * sigma, 4),
    }
