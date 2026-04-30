"""
ota_beta_estimation.py — Ota framework Layer 2: gene → program β estimation.

The Ota et al. (Nature 2026) framework decomposes causal pathways as:
  γ_{gene→trait} = Σ_P (β_{gene→P} × γ_{P→trait})

This module estimates β_{gene→P}: the causal effect of gene X on program P activity.

β is fundamentally an INTERVENTIONAL quantity — it requires perturbation data or
a genetic instrument. Co-expression is explicitly NOT used because co-expression ≠ causation
(confounding, reverse causation, and shared upstream regulators all produce co-expression
without gene X causally driving program P).

Corrected tier hierarchy:
─────────────────────────────────────────────────────────────────────────────
Tier 1  Interventional   Cell-type-matched Perturb-seq (scPerturb database or
                         Replogle 2022 K562 for blood/myeloid diseases).
                         Direct CRISPR perturbation → transcriptome shift.

Tier 2  Convergent       eQTL-MR: use tissue-matched cis-eQTLs (GTEx) as
                         genetic instruments for gene expression; project onto
                         program gene loadings via MR effect size.
                         Causal basis: Mendelian randomization.

Tier 2L Latent Hijack    Cross-lineage mechanistic transfer: transfer β from
                         a data-rich disease (e.g. IBD) to a data-poor one
                         based on high similarity of latent state motifs.
                         Causal basis: transferred interventional evidence.

Tier 3  Provisional      LINCS L1000 genetic perturbation (shRNA/ORF): still
                         direct perturbation but in a cell line that may not
                         match the disease-relevant cell type.
                         Causal basis: real perturbation, imperfect cell match.

Virtual                  In silico only. Two sub-sources:
                           (a) Geneformer/GEARS: models trained on real
                               perturbation data; extrapolates to unseen genes
                               or cell types. Better than co-expression but
                               still not experimental.
                           (b) Pathway membership: binary proxy (gene in
                               program = 1, out = 0). No causal basis.
                         Must be labelled provisional_virtual; cannot be used
                         for clinical translation.
─────────────────────────────────────────────────────────────────────────────

Co-expression / GRN is NOT a tier. Observing that genes co-express does not
tell us the direction of regulation or whether there is any causal relationship.
GRN methods (SCENIC, Arboreto) produce directed graphs but the edges still
reflect statistical association conditioned on other genes, not intervention.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.evidence import ProgramBetaMatrix
from config.scoring_thresholds import (
    COLOC_H4_MIN, MR_PQTL_P_VALUE_MAX,
    EQTL_GWAS_DIRECTION_SIGMA_FACTOR, EQTL_GWAS_DIRECTION_COLOC_MIN,
    WES_CODING_MECHANISM_P,
)


def _se_from_pval(beta: float, pval: float, min_z: float = 2.0) -> float:
    """
    Back-calculate SE from a p-value and effect size via normal approximation.

        SE = |beta| / z,   z = Φ⁻¹(1 − p/2)

    min_z floor (default 2.0 = p≈0.046) prevents infinite SE when p≈1.
    Used when a source returns a p-value but not a standard error.
    """
    import math as _m
    pval = max(pval, 1e-300)
    pval = min(pval, 1.0 - 1e-15)
    try:
        from scipy.special import erfinv as _erfinv
        z = _m.sqrt(2.0) * float(_erfinv(1.0 - pval))
    except ImportError:
        # Approximation: z ≈ −Φ⁻¹(p/2) via rational approximation (Abramowitz & Stegun)
        t = _m.sqrt(-2.0 * _m.log(pval / 2.0))
        c = (2.515517, 0.802853, 0.010328)
        d = (1.432788, 0.189269, 0.001308)
        z = t - (c[0] + c[1]*t + c[2]*t*t) / (1 + d[0]*t + d[1]*t*t + d[2]*t*t*t)
    z = max(z, min_z)
    return abs(beta) / z


# ---------------------------------------------------------------------------
# Cell-type relevance registry
#
# Maps disease short-key → set of dataset/cell_type identifiers that are
# genuinely disease-relevant.  Any Perturb-seq data NOT in this set for the
# disease is demoted from Tier1_Interventional → Tier3_Provisional (cell-line
# mismatch) rather than silently claiming Tier1 status.
#
# Rationale:
#   K562 is a CML leukemia line.  It is an acceptable generic screen for
#   myeloid/blood biology (IBD macrophage, RA monocyte) but NOT for CAD
#   vascular biology or AMD retinal biology.  Using K562 β for CAD and
#   labelling it Tier1_Interventional inflates ranks for RNA-processing genes
#   (BUD13, SF3A3) whose K562 signatures dominate all NMF programs.
# ---------------------------------------------------------------------------

_CELL_TYPE_MATCHED_DATASETS: dict[str, set[str]] = {
    # CAD: Schnitzler HCASMC/HAEC and Natsume HAEC are vascular cell matches
    "CAD":  {"schnitzler_cad_vascular", "natsume_2023_haec",
             "HCASMC", "HAEC", "HCASMC_HAEC",
             "Schnitzler_GSE210681", "GSE210681"},
    # RA: CZI CD4+ T is now the primary match (Th1/Th17); K562 fallback
    "RA":   {"czi_2025_cd4t_perturb", "replogle_2022_k562", "K562", "k562"},
    # SLE: CZI CD4+ T primary human cells is best match; K562 fallback
    "SLE":  {"czi_2025_cd4t_perturb", "papalexi_2021_thp1", "replogle_2022_k562", "K562", "k562"},
    # DED: shares CZI CD4+ T source with SLE (Th17/Th1 inflammatory mechanism)
    "DED":  {"czi_2025_cd4t_perturb", "replogle_2022_k562", "K562", "k562"},
    # T2D: no pancreas match available; K562 accepted as best proxy for now
    "T2D":  {"replogle_2022_k562", "K562", "k562"},
}

# K562 is the catch-all fallback cell type identifier
_K562_IDENTIFIERS: frozenset[str] = frozenset({"K562", "k562", "replogle_2022_k562"})


def _is_cell_type_matched(cell_type: str, disease: str | None) -> bool:
    """Return True if cell_type is disease-relevant for the given disease."""
    if not disease:
        return True   # no context → don't penalise
    disease_upper = disease.upper().replace("-", "").replace(" ", "_")
    # Normalise common disease aliases
    _alias = {"CORONARY_ARTERY_DISEASE": "CAD",
               "RHEUMATOID_ARTHRITIS": "RA", "SYSTEMIC_LUPUS_ERYTHEMATOSUS": "SLE",
               "DRY_EYE_DISEASE": "DED", "DRY_EYE_SYNDROME": "DED",
               "TYPE_2_DIABETES": "T2D"}
    disease_key = _alias.get(disease_upper, disease_upper)
    matched = _CELL_TYPE_MATCHED_DATASETS.get(disease_key)
    if matched is None:
        return True   # unknown disease — don't penalise
    return cell_type in matched


# ---------------------------------------------------------------------------
# Tier 1: Cell-type-matched Perturb-seq
# ---------------------------------------------------------------------------

def estimate_beta_tier1(
    gene: str,
    program: str,
    perturbseq_data: dict | None = None,
    cell_type: str = "K562",
    cell_type_matched: bool = True,
) -> dict | None:
    """
    Tier 1 β: Direct Perturb-seq measurement.

    Returns Tier1_Interventional only when cell_type_matched=True (the cell line
    is disease-relevant).  When cell_type_matched=False (e.g. K562 used for CAD
    instead of Schnitzler HCASMC), returns Tier3_Provisional with a mismatch note
    so the causal hierarchy is not inflated.

    Uses qualitative sign-level data from burden_perturb_server when the full
    h5ad has not been downloaded.  When quantitative data is available (h5ad
    loaded), returns the actual β coefficient + SE.

    Args:
        gene:            Gene symbol
        program:         cNMF program / gene-set name
        perturbseq_data: Pre-loaded Perturb-seq data dict (gene → program → {beta, se, ...})
        cell_type:       Cell line / type identifier; used to annotate data_source
    """
    # Tier label depends on cell-type relevance for the disease
    _tier      = "Tier1_Interventional" if cell_type_matched else "Tier3_Provisional"
    _mismatch  = (
        "" if cell_type_matched
        else f"; cell-line mismatch ({cell_type} is not disease-relevant — use disease-matched data for Tier1)"
    )

    if perturbseq_data is None:
        # Qualitative path — sign-level β from curated server data
        from mcp_servers.burden_perturb_server import get_gene_perturbation_effect
        effect = get_gene_perturbation_effect(gene)
        if effect.get("data_tier") != "qualitative":
            return None
        up_progs = effect.get("top_programs_up", [])
        dn_progs = effect.get("top_programs_dn", [])
        if program in up_progs:
            return {
                "beta":          1.0,   # sign-only; quantitative requires h5ad
                "beta_se":       None,
                "ci_lower":      None,
                "ci_upper":      None,
                "beta_sigma":    0.50,  # sign known, magnitude unknown
                "evidence_tier": _tier,
                "data_source":   f"Replogle2022_{cell_type}_qualitative",
                "note":          f"Sign-only β; download Figshare pseudo-bulk h5ad for quantitative estimate{_mismatch}",
            }
        if program in dn_progs:
            return {
                "beta":          -1.0,
                "beta_se":       None,
                "ci_lower":      None,
                "ci_upper":      None,
                "beta_sigma":    0.50,  # sign known, magnitude unknown
                "evidence_tier": _tier,
                "data_source":   f"Replogle2022_{cell_type}_qualitative",
                "note":          f"Sign-only β; download Figshare pseudo-bulk h5ad for quantitative estimate{_mismatch}",
            }
        return None

    # Quantitative path (h5ad loaded and processed through cNMF)
    gene_data = perturbseq_data.get(gene, {})
    prog_beta = gene_data.get("programs", {}).get(program)
    if prog_beta is None:
        return None
    return {
        "beta":          prog_beta["beta"],
        "beta_se":       prog_beta.get("se"),
        "ci_lower":      prog_beta.get("ci_lower"),
        "ci_upper":      prog_beta.get("ci_upper"),
        "beta_sigma":    prog_beta.get("se") or abs(prog_beta["beta"]) * 0.15,
        "evidence_tier": _tier,
        "data_source":   f"Perturb-seq_{cell_type}_quantitative",
        **({"note": f"Cell-line mismatch{_mismatch}"} if not cell_type_matched else {}),
    }


# ---------------------------------------------------------------------------
# Helper: GWAS risk allele direction vs eQTL NES concordance
# ---------------------------------------------------------------------------

def _eqtl_gwas_direction_concordant(
    eqtl_nes: float | None,
    gwas_beta: float | None,
) -> bool | None:
    """
    Return True if eQTL NES and GWAS beta are directionally concordant.

    Concordance means the eQTL-increasing allele for expression goes in the
    same direction as the GWAS risk allele — consistent with the gene's mRNA
    level being a mediator of the GWAS signal.

    Returns None when either value is missing or zero (cannot determine direction).
    """
    if eqtl_nes is None or gwas_beta is None:
        return None
    if abs(float(eqtl_nes)) < 1e-10 or abs(float(gwas_beta)) < 1e-10:
        return None
    return math.copysign(1.0, float(eqtl_nes)) == math.copysign(1.0, float(gwas_beta))


def _coding_mechanism_gate(burden_data: dict | None) -> bool:
    """
    Return True if WES burden evidence suggests a coding (not regulatory) mechanism.

    When WES rare LoF burden is nominally significant (p < WES_CODING_MECHANISM_P),
    the gene likely acts via protein-level changes rather than mRNA regulation.
    eQTL-MR in this regime may be measuring the wrong instrument (mRNA level) when
    the true instrument is protein abundance/function. Suppress Tier 2a/2b/2c eQTL-MR.

    Tier 2p (pQTL), Tier 2rb (rare burden), and Tier 2.5 (direction-only) are unaffected.
    """
    if burden_data is None:
        return False
    burden_p = burden_data.get("burden_p")
    if burden_p is None:
        return False
    try:
        return float(burden_p) < WES_CODING_MECHANISM_P
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Tier 2: eQTL-MR (Mendelian Randomization via GTEx)
# ---------------------------------------------------------------------------

def estimate_beta_tier2(
    gene: str,
    program: str,
    eqtl_data: dict | None = None,
    coloc_h4: float | None = None,
    coloc_h3: float | None = None,
    program_loading: float | None = None,
) -> dict | None:
    """
    Tier 2 β: eQTL-MR estimate.

    Uses cis-eQTLs (genetic instruments) for gene X in the disease-relevant
    tissue as instruments for gene X expression, then projects onto program P
    via the program gene loading.

    β_{gene→P} ≈ eQTL_NES × loading(gene_X in program_P)

    This is Mendelian randomization: genetic variation (eQTL SNP) randomizes
    gene X expression, and we measure downstream program effect.  The COLOC H4
    requirement (≥ 0.8) ensures the eQTL and program-activity signal share the
    same causal variant rather than being driven by distinct signals in LD.

    Args:
        gene:             Gene symbol
        program:          Program name
        eqtl_data:        GTEx eQTL result {nes, se, pval_nominal, tissue}
        coloc_h4:         COLOC H4 posterior (shared causal variant probability)
        program_loading:  Gene X's loading / weight in program P (from cNMF or gene set)
    """
    if eqtl_data is None:
        return None
    if coloc_h4 is not None and coloc_h4 < COLOC_H4_MIN:
        # COLOC present but weak — do not promote to Tier2; caller may use
        # estimate_beta_tier2_eqtl_direction() as a weaker fallback.
        return None

    # SMR + HEIDI pleiotropic instrument check
    if eqtl_data and coloc_h3 is not None and coloc_h4 is not None:
        from pipelines.smr_heidi import heidi_proxy_from_coloc, smr_heidi_filter
        _heidi = heidi_proxy_from_coloc(coloc_h3, coloc_h4)
        if eqtl_data.get("beta_gwas") is not None and eqtl_data.get("se_gwas") is not None:
            from pipelines.smr_heidi import compute_smr_beta
            _smr = compute_smr_beta(
                beta_gwas=eqtl_data["beta_gwas"],
                se_gwas=eqtl_data["se_gwas"],
                beta_eqtl=float(eqtl_data["nes"]),
                se_eqtl=float(eqtl_data["se"]),
            )
            _filt = smr_heidi_filter(_smr.get("beta_smr"), _smr.get("se_smr"), _heidi["heidi_flag"])
        else:
            _filt = smr_heidi_filter(None, None, _heidi["heidi_flag"])
        if not _filt["keep"]:
            return None

    nes = eqtl_data.get("nes")
    if nes is None:
        return None

    # Scale by program loading when available; raw NES otherwise
    loading = program_loading if program_loading is not None else 1.0
    beta = nes * loading

    tissue = eqtl_data.get("tissue", "unknown_tissue")
    coloc_str = f"_COLOC_H4={coloc_h4:.2f}" if coloc_h4 is not None else ""

    raw_sigma = (
        abs((eqtl_data.get("se") or 0.0) * (loading if loading is not None else 1.0))
        or _se_from_pval(beta, float(eqtl_data.get("pval_nominal") or 0.05))
    )
    # COLOC H4 magnitude scales beta_sigma: borderline H4 (0.80) has ~2x the
    # uncertainty of high H4 (0.95). Map [COLOC_H4_MIN, 0.95] → sigma multiplier
    # [2.0, 1.0] so borderline colocs propagate wider uncertainty downstream.
    if coloc_h4 is not None and coloc_h4 < 0.95:
        _h4_range = max(0.95 - COLOC_H4_MIN, 1e-6)
        _h4_mult = 1.0 + (0.95 - coloc_h4) / _h4_range  # 1.0 at H4=0.95, 2.0 at H4=COLOC_H4_MIN
        raw_sigma = raw_sigma * _h4_mult

    # GWAS direction vs eQTL direction concordance: when eQTL NES and GWAS beta
    # agree in direction at COLOC H4 ≥ EQTL_GWAS_DIRECTION_COLOC_MIN, tighten sigma
    # by EQTL_GWAS_DIRECTION_SIGMA_FACTOR — two independent lines of genetic evidence.
    _eqtl_gwas_concordant = False
    _gwas_beta = eqtl_data.get("beta_gwas")
    if (
        coloc_h4 is not None and coloc_h4 >= EQTL_GWAS_DIRECTION_COLOC_MIN
        and _gwas_beta is not None
    ):
        _concordant = _eqtl_gwas_direction_concordant(nes, _gwas_beta)
        if _concordant:
            raw_sigma *= EQTL_GWAS_DIRECTION_SIGMA_FACTOR
            _eqtl_gwas_concordant = True

    return {
        "beta":                    beta,
        "beta_se":                 eqtl_data.get("se"),
        "ci_lower":                None,
        "ci_upper":                None,
        "beta_sigma":              raw_sigma,
        "evidence_tier":           "Tier2_Convergent",
        "data_source":             f"GTEx_{tissue}_eQTL_MR{coloc_str}",
        "coloc_h4":                coloc_h4,
        "mr_method":               "eQTL_NES_x_loading",
        "eqtl_gwas_concordant":    _eqtl_gwas_concordant,
        "note":                    "MR-based: genetic instrument (cis-eQTL) for gene expression → program loading",
    }


# ---------------------------------------------------------------------------
# Tier 2b: Open Targets credible-set genetic instruments
# Supplements GTEx when eQTL data is absent (common for immune-specific genes).
# ---------------------------------------------------------------------------

def estimate_beta_tier2_ot_instrument(
    gene: str,
    program: str,
    ot_instruments: dict | None = None,
    program_loading: float | None = None,
) -> dict | None:
    """
    Tier 2 β using Open Targets GWAS/eQTL credible-set instruments.

    Activates when GTEx eQTL data is absent.  Uses OT's integrated genetic
    evidence (multi-cohort GWAS fine-mapping + eQTL catalogue colocalization)
    as the genetic instrument.

    For eQTL instruments: β = eQTL_NES × loading  (same logic as estimate_beta_tier2)
    For GWAS credible sets: β = log(OR) × loading  (approximate — GWAS β to program)

    The GWAS-only path is labelled Tier2_Convergent but flagged as "gwas_projected"
    to signal that the gene→program direction is inferred, not directly measured.

    Args:
        gene:             Gene symbol
        program:          Program name
        ot_instruments:   Output of get_ot_genetic_instruments() — has `instruments`,
                          `best_nes`, `best_gwas_beta`
        program_loading:  Gene's weight in the program (from cNMF loadings)
    """
    if not ot_instruments or not ot_instruments.get("instruments"):
        return None

    loading = program_loading if program_loading is not None else 1.0

    # Prefer eQTL instrument (direct expression effect, same logic as GTEx Tier2)
    best_eqtl = next(
        (i for i in ot_instruments["instruments"] if i["instrument_type"] == "eqtl"),
        None,
    )
    if best_eqtl:
        nes  = best_eqtl["beta"]
        se   = best_eqtl.get("se")
        beta = nes * loading
        return {
            "beta":          beta,
            "beta_se":       (se * abs(loading)) if se else None,
            "ci_lower":      None,
            "ci_upper":      None,
            "beta_sigma":    (
                abs(se * loading) if se
                else _se_from_pval(beta, float(best_eqtl.get("pvalue") or best_eqtl.get("p_value") or 0.05))
            ),
            "evidence_tier": "Tier2_Convergent",
            "data_source":   f"OT_eQTL_catalogue_{ot_instruments.get('ensembl_id', gene)}",
            "instrument_type": "eqtl",
            "note":          "OT eQTL credible set × program loading (MR)",
        }

    # GWAS credible-set β is NOT used as a program-level proxy.
    # β_{gene→program} requires a direct measurement (eQTL colocalising with the
    # program, or perturb-seq KO).  Projecting a GWAS variant β onto program
    # loadings conflates genetic association with causal program effect and
    # produces a flat uniform beta across all programs that cancels in the OTA
    # sum when program gammas have mixed signs.  Return None so the gene is
    # scored via the OT-L2G genetic anchor path rather than a spurious β×γ sum.
    return None


# ---------------------------------------------------------------------------
# Tier 2c: Single-cell eQTL (cell-type specific, from eQTL Catalogue)
# Fills retina/RPE and immune-cell-specific gaps that GTEx bulk misses.
# ---------------------------------------------------------------------------

def estimate_beta_tier2_sc_eqtl(
    gene: str,
    program: str,
    sc_eqtl_data: dict | None = None,
    coloc_h4: float | None = None,
    program_loading: float | None = None,
) -> dict | None:
    """
    Tier 2c β: Cell-type-specific eQTL from eQTL Catalogue (OneK1K, Blueprint, etc.)

    Single-cell eQTLs capture regulatory effects that are present in a specific
    cell type but diluted to noise in bulk GTEx data (which averages across all
    cell types in a tissue). Key use cases:
      - Monocyte-specific eQTLs for CHIP/myeloid AMD genes (OneK1K)
      - T-cell-specific eQTLs for immune disease (RA, IBD, SLE)
      - Any gene where GTEx bulk eQTL is absent but cell-type eQTL exists

    Activated when:
      - sc_eqtl_data contains top_eqtl with beta/pvalue
      - COLOC H4 ≥ 0.8 (or absent — sc-eQTL rarely has COLOC computed) → full Tier2c
      - COLOC H4 < 0.8 → direction-only Tier2c (high sigma)

    Args:
        gene:            Gene symbol
        program:         Program name
        sc_eqtl_data:    Result from get_sc_eqtl() — expects {top_eqtl: {beta, se, pvalue, ...}}
        coloc_h4:        COLOC H4 if available (often None for sc-eQTLs)
        program_loading: Gene's weight in program P
    """
    if sc_eqtl_data is None:
        return None
    top = sc_eqtl_data.get("top_eqtl") or sc_eqtl_data
    if not top:
        return None

    beta_eqtl = top.get("beta")
    if beta_eqtl is None:
        return None
    try:
        beta_eqtl = float(beta_eqtl)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(beta_eqtl):
        return None

    pvalue = float(top.get("pvalue", 1.0))
    if pvalue > 1e-4:  # weak sc-eQTL — don't use
        return None

    loading = program_loading if program_loading is not None else 1.0
    cell_type = (
        top.get("condition_label")
        or sc_eqtl_data.get("cell_type")
        or "unknown_cell_type"
    )
    study = top.get("study_label") or sc_eqtl_data.get("study") or "eQTL_Catalogue"

    # Full Tier2c when COLOC confirms shared variant (or COLOC unavailable for sc)
    coloc_weak = coloc_h4 is not None and coloc_h4 < COLOC_H4_MIN
    if coloc_weak:
        beta = math.copysign(abs(loading), beta_eqtl)
        sigma = 0.50
        tier_label = "Tier2c_scEQTL_direction"
    else:
        beta = beta_eqtl * loading
        sigma = (
            abs((top.get("se") or 0.0) * loading)
            or _se_from_pval(beta, pvalue)
        )
        tier_label = "Tier2c_scEQTL"

    coloc_str = f"_COLOC_H4={coloc_h4:.2f}" if coloc_h4 is not None else ""

    return {
        "beta":          beta,
        "beta_se":       top.get("se"),
        "ci_lower":      None,
        "ci_upper":      None,
        "beta_sigma":    sigma,
        "evidence_tier": tier_label,
        "data_source":   f"{study}_{cell_type}_scEQTL{coloc_str}",
        "cell_type":     cell_type,
        "coloc_h4":      coloc_h4,
        "pvalue":        pvalue,
        "mr_method":     "scEQTL_NES_x_loading",
        "note":          (
            f"Cell-type-specific eQTL ({cell_type}, {study}): "
            "captures regulatory effects diluted in bulk GTEx tissue."
        ),
    }


# ---------------------------------------------------------------------------
# Tier 2p: pQTL-MR (protein QTL Mendelian Randomization)
# Critical for complement / coding variant genes with no cis-eQTL in GTEx.
# ---------------------------------------------------------------------------

def estimate_beta_tier2_pqtl(
    gene: str,
    program: str,
    pqtl_data: dict | None = None,
    program_loading: float | None = None,
) -> dict | None:
    """
    Tier 2p β: pQTL-MR (protein quantitative trait locus MR).

    Uses genetic variants that alter protein abundance (not mRNA levels) as
    instruments for gene function. This is the correct instrument when:
      - The causal variant is a coding mutation (changes protein sequence/stability)
      - The gene's mRNA expression is unchanged but protein levels differ
      - Examples: CFH Y402H, LPA kringles, TREM2 R47H, APOE ε4

    pQTL sources (all from eQTL Catalogue / Sun 2023 UKB-PPP):
      - UKB-PPP: 2,923 proteins, 54,219 UK Biobank participants (Olink Explore)
      - INTERVAL: 3,622 proteins, ~3,300 blood donors (SOMAscan)
      - deCODE: 4,719 proteins, 35,559 Icelanders (SOMAscan)

    β_{gene→P} ≈ pQTL_NES × loading(gene in program_P)

    This is analogous to eQTL-MR but operating at the protein level. The pQTL
    NES represents the SD change in protein abundance per effect allele.

    Args:
        gene:             Gene symbol
        program:          Program name
        pqtl_data:        Result from get_pqtl_instruments() or get_best_pqtl_for_gene()
                          Expects {top_pqtl: {beta, se, pvalue, rsid, study_label}}
        program_loading:  Gene's weight in program P
    """
    if pqtl_data is None:
        return None

    top = pqtl_data.get("top_pqtl") or pqtl_data
    if not top:
        return None

    beta_pqtl = top.get("beta")
    if beta_pqtl is None:
        return None
    try:
        beta_pqtl = float(beta_pqtl)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(beta_pqtl):
        return None

    pvalue = float(top.get("pvalue", 1.0))
    if pvalue > 0.05:  # relaxed from 1e-5: complement pQTLs (e.g. CFH p=0.019) pass
        return None

    loading = program_loading if program_loading is not None else 1.0
    beta    = beta_pqtl * loading
    se_pqtl = top.get("se")
    sigma   = abs((se_pqtl or 0.0) * loading) or _se_from_pval(beta, pvalue)

    study   = top.get("study_label") or pqtl_data.get("data_source", "pQTL_unknown")
    rsid    = top.get("rsid") or top.get("variant_id", "")

    return {
        "beta":          beta,
        "beta_se":       (se_pqtl * abs(loading)) if se_pqtl else None,
        "ci_lower":      None,
        "ci_upper":      None,
        "beta_sigma":    sigma,
        "evidence_tier": "Tier2p_pQTL_MR",
        "data_source":   f"{study}_pQTL_MR_{rsid}",
        "pvalue":        pvalue,
        "mr_method":     "pQTL_NES_x_loading",
        "note":          (
            f"pQTL-MR via {study}: protein-level instrument (NES={beta_pqtl:.3f}) × loading. "
            "Appropriate for coding variants where cis-eQTL is absent (e.g. CFH Y402H, LPA)."
        ),
    }


# ---------------------------------------------------------------------------
# Tier 2rb: Rare variant burden direction (UKB WES collapsing test)
# Weakest genetic tier — direction constraint from LoF burden, above LINCS.
# ---------------------------------------------------------------------------

def estimate_beta_tier2_rare_burden(
    gene: str,
    program: str,
    burden_data: dict | None = None,
    program_loading: float | None = None,
) -> dict | None:
    """
    Tier 2rb β: Rare variant burden direction constraint (UKB WES collapsing test).

    Uses rare variant (MAF < 0.1%) LoF + damaging missense burden to constrain
    the DIRECTION of gene perturbation effect on disease. Collapsing tests ask:
    "do carriers of rare LoF variants have higher/lower disease risk?"

    This tells us the consequence of loss-of-function, giving us:
      - burden_beta > 0: LoF increases risk → gene normally protective
        → inhibiting the gene (reducing function) worsens disease
        → gene activation is therapeutic target
      - burden_beta < 0: LoF decreases risk → gene normally harmful
        → inhibiting the gene is therapeutic

    β direction: For a program that gene X activates (loading > 0):
      If LoF → increased disease risk, then more gene X activity → better
      → β_{gene→program} is positive (restoring gene function helps)

    This is below Tier 2.5 (eQTL direction) in evidence strength because:
      - Burden tests collapse variants with unknown functional directions
      - LoF of a regulator may have opposite effect from reducing expression
      - Sigma is 0.60 (very large) to reflect this

    Only activated when burden_p < 1e-4 (to avoid noise driving direction).

    Args:
        gene:             Gene symbol
        program:          Program name
        burden_data:      Result from get_gene_burden() or get_burden_direction_for_gene()
        program_loading:  Gene's weight in program P
    """
    if burden_data is None:
        return None

    burden_beta = burden_data.get("burden_beta")
    burden_p    = burden_data.get("burden_p")
    loeuf       = burden_data.get("loeuf")

    # Need at least a p-value to use burden as directional evidence
    if burden_beta is None or burden_p is None:
        # If only constraint available (no disease-specific burden), don't activate
        return None

    try:
        burden_p    = float(burden_p)
        burden_beta = float(burden_beta)
    except (TypeError, ValueError):
        return None

    if burden_p > 1e-4:  # threshold: at least suggestive burden
        return None
    if not math.isfinite(burden_beta):
        return None

    loading = program_loading if program_loading is not None else 1.0

    if program_loading is not None and abs(float(program_loading)) > 0:
        beta = math.copysign(abs(float(program_loading)), burden_beta)
    else:
        beta = burden_beta

    loeuf_str = f"_LOEUF={loeuf:.3f}" if loeuf is not None else ""
    study = burden_data.get("burden_study") or "UKB_WES"

    return {
        "beta":          beta,
        "beta_se":       burden_data.get("burden_se"),
        "ci_lower":      None,
        "ci_upper":      None,
        "beta_sigma":    0.60,  # direction from burden test, magnitude from loading (noisy)
        "evidence_tier": "Tier2rb_RareBurden",
        "data_source":   f"{study}_rare_burden{loeuf_str}",
        "burden_p":      burden_p,
        "loeuf":         loeuf,
        "mr_method":     "rare_burden_sign_x_loading",
        "note":          (
            f"Rare variant burden (p={burden_p:.2e}): sign from burden test, magnitude from loading. "
            "Direction reliable; magnitude uncertain; beta_sigma=0.60 reflects this. "
            f"Interpretation: {burden_data.get('interpretation', '')[:100]}"
        ),
    }


# ---------------------------------------------------------------------------
# Tier 2.5: eQTL direction-only (eQTL present but COLOC H4 < 0.8)
# Fills the gap for genes like FBN2 / PRPH2 / HMCN1 that have a cis-eQTL
# but whose GWAS colocalization is below the H4 ≥ 0.8 threshold for full Tier2.
# We retain the eQTL sign to constrain direction and use a large sigma (0.50)
# to reflect that the variant may not be on the causal disease haplotype.
# ---------------------------------------------------------------------------

def estimate_beta_tier2_eqtl_direction(
    gene: str,
    program: str,
    eqtl_data: dict | None = None,
    coloc_h4: float | None = None,
    program_loading: float | None = None,
) -> dict | None:
    """
    Tier 2.5 β: eQTL direction-only when COLOC H4 < 0.8.

    Activated when:
      - A cis-eQTL exists for the gene in the relevant tissue (eqtl_data is not None)
      - COLOC H4 is either absent or below 0.8 (shared causal variant uncertain)

    β = sign(NES) × |program_loading|   (direction preserved, magnitude uncertain)

    This is weaker than full Tier2 (COLOC-confirmed) but stronger than LINCS
    (different cell type) or Virtual (no perturbation). It is appropriate for:
      - Genes with eQTL in primary tissue but no GWAS fine-mapping available
      - eQTLs from secondary tissues (e.g. liver for complement genes in AMD)
      - Genes where COLOC evidence is incomplete rather than absent

    Args:
        gene:             Gene symbol
        program:          Program name
        eqtl_data:        GTEx eQTL result {nes, se, pval_nominal, tissue}
        coloc_h4:         COLOC H4 posterior (should be < 0.8 or None to use this tier)
        program_loading:  Gene's weight in program P (from cNMF or gene set)
    """
    if eqtl_data is None:
        return None
    # Only activate this tier when COLOC would reject Tier2 (H4 < COLOC_H4_MIN or absent)
    if coloc_h4 is not None and coloc_h4 >= COLOC_H4_MIN:
        return None  # caller should use estimate_beta_tier2 instead

    nes = eqtl_data.get("nes")
    if nes is None or not math.isfinite(float(nes)):
        return None

    loading = program_loading if program_loading is not None else 1.0

    beta = math.copysign(abs(loading), float(nes))
    beta_sigma = 0.50

    tissue = eqtl_data.get("tissue", "unknown_tissue")
    coloc_str = f"H4={coloc_h4:.2f}" if coloc_h4 is not None else "H4=absent"

    return {
        "beta":          beta,
        "beta_se":       None,
        "ci_lower":      None,
        "ci_upper":      None,
        "beta_sigma":    beta_sigma,
        "evidence_tier": "Tier2_eQTL_direction",
        "data_source":   f"GTEx_{tissue}_eQTL_direction_only_{coloc_str}",
        "coloc_h4":      coloc_h4,
        "mr_method":     "eQTL_NES_x_sqrt_coloc_credibility_x_loading",
        "note":          (
            f"eQTL exists ({tissue}, NES={float(nes):.3f}) but COLOC {coloc_str} — "
            f"β = sign(NES)×|loading|; beta_sigma={beta_sigma:.2f}."
        ),
    }




def _l2_norm(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def _cosine_sim(a: list[float], b: list[float], norm_a: float | None = None) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    dot = sum(a[i] * b[i] for i in range(n))
    na = norm_a if norm_a is not None else _l2_norm(a)
    nb = _l2_norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ---------------------------------------------------------------------------
# Tier 2L: Latent Hijack — Cross-disease mechanistic transfer
# Transfers β from data-rich diseases (e.g. IBD) based on state-space similarity.
# ---------------------------------------------------------------------------

def estimate_beta_tier2L_latent_transfer(
    gene: str,
    program: str,
    current_disease_motif: np.ndarray | None = None,
    motif_library: dict[str, dict] | None = None,
) -> dict | None:
    """
    Tier 2L β: cross-disease mechanistic transfer via latent motif similarity.

    Matches the current disease's pathological state latent motif (loadings)
    against a library of "data-rich" diseases. If a high-similarity match
    (cosine > 0.8) is found, it "hijacks" the validated perturbation effect (β)
    from the source disease.

    Example: "fibrotic transition" in eye (data-poor) matches IBD (data-rich).

    Args:
        gene:                   Target gene
        program:                Program name
        current_disease_motif:  Latent vector (genes × 1) for current state-transition
        motif_library:          {source_disease: {motif: np.ndarray, betas: {gene: beta}}}
    """
    if current_disease_motif is None or not motif_library:
        return None

    best_sim = 0.0
    best_source = None
    
    # current_disease_motif is expected to be a numpy array or list
    target_vec = current_disease_motif.tolist() if hasattr(current_disease_motif, "tolist") else list(current_disease_motif)
    norm_target = _l2_norm(target_vec)

    for source_disease, data in motif_library.items():
        source_motif = data.get("motif")
        if source_motif is None:
            continue
            
        # source_motif is expected to be a numpy array or list
        source_vec = source_motif.tolist() if hasattr(source_motif, "tolist") else list(source_motif)
        
        # Compute cosine similarity
        sim = _cosine_sim(target_vec, source_vec, norm_target)
        if sim > best_sim:
            best_sim = sim
            best_source = source_disease

    if best_sim < 0.8 or best_source is None:
        return None

    # Transfer β from best source
    source_betas = motif_library[best_source].get("betas", {})
    source_beta = source_betas.get(gene)
    
    if source_beta is None:
        return None

    return {
        "beta":          source_beta,
        "beta_se":       None,
        "ci_lower":      None,
        "ci_upper":      None,
        "beta_sigma":    0.40,  # transferred interventional evidence
        "evidence_tier": "Tier2L_LatentHijack",
        "data_source":   f"LatentHijack_from_{best_source}_sim={best_sim:.2f}",
        "note":          (
            f"Transferred β from {best_source} due to high latent motif similarity "
            f"({best_sim:.2f}). Hijacks validated mechanistic evidence across lineages."
        ),
    }


def estimate_beta_virtual(
    gene: str,
    program: str,
    pathway_member: bool | None = None,
) -> dict:
    """
    Virtual β sub-source B: pathway membership proxy.

    If gene X is in program P's defining gene set, β = 1.0 (binary proxy).
    This has NO causal basis — it is annotation-only.  Used as final fallback
    only so the pipeline produces a finite matrix; all outputs are explicitly
    labelled provisional_virtual.

    Co-expression-derived weights are intentionally excluded here because
    they would imply causality that the data cannot support.
    """
    beta_val = 1.0 if pathway_member else None
    return {
        "beta":          beta_val,
        "beta_se":       None,
        "ci_lower":      None,
        "ci_upper":      None,
        "beta_sigma":    0.70,  # maximum uncertainty: annotation proxy only
        "evidence_tier": "provisional_virtual",
        "data_source":   "pathway_membership_proxy",
        "note":          "Annotation proxy only — no causal basis. Must be labelled provisional_virtual.",
    }


# ---------------------------------------------------------------------------
# Main β fallback decision tree
# ---------------------------------------------------------------------------

def estimate_beta(
    gene: str,
    program: str,
    perturbseq_data: dict | None = None,
    eqtl_data: dict | None = None,
    coloc_h4: float | None = None,
    program_loading: float | None = None,
    program_gene_set: set[str] | None = None,
    pathway_member: bool | None = None,
    cell_type: str = "K562",
    cell_line: str | None = None,
    ot_instruments: dict | None = None,
    sc_eqtl_data: dict | None = None,
    pqtl_data: dict | None = None,
    burden_data: dict | None = None,
    current_disease_motif: Any | None = None,
    motif_library: dict | None = None,
    disease: str | None = None,
) -> dict:
    """
    β fallback decision tree for the Ota framework.

    Cell-type relevance: if cell_type is not disease-relevant (e.g. K562 for CAD),
    Perturb-seq β is demoted from Tier1_Interventional → Tier3_Provisional.
    Pass disease= to enable this check; omit for disease-agnostic use.

    Priority:
      1. Tier1    — cell-type-matched Perturb-seq (direct intervention)
      2. Tier2p   — pQTL-MR (protein-level instrument; UKB-PPP / INTERVAL / deCODE)
                    [elevated: drug target IS the protein; pQTL causally prior to eQTL]
      2. Tier2a   — GTEx eQTL-MR (COLOC H4 ≥ 0.8)
      2. Tier2b   — OT credible-set instruments (GWAS/eQTL, when GTEx absent)
      2. Tier2c   — sc-eQTL from eQTL Catalogue (cell-type specific; OneK1K / Blueprint)
      2. Tier2L   — Latent Hijack (Cross-disease mechanistic transfer)
      2. Tier2.5  — eQTL direction-only (eQTL present but COLOC H4 < 0.8; sigma=0.50)
      2. Tier2rb  — Rare variant burden direction (UKB WES collapsing; sigma=0.60)

    Co-expression, GRN weights, and pathway-annotation-derived synthetic betas
    are intentionally absent from this chain — they do not provide causal evidence.
    """
    # Tier 1 — Perturb-seq KO (cell-type mismatch noted in evidence_tier, not demoted)
    _matched = _is_cell_type_matched(cell_type, disease)
    beta = estimate_beta_tier1(gene, program, perturbseq_data, cell_type=cell_type, cell_type_matched=_matched)
    if beta is not None:
        # Cross-check Tier1 sign vs Tier2 eQTL-MR at high COLOC (Gap 5).
        # eQTL + GWAS is a human genetic causal direction; if it contradicts
        # the cell-line Perturb-seq at high COLOC, flag uncertainty but keep Tier1.
        if coloc_h4 is not None and coloc_h4 >= COLOC_H4_MIN:
            _t2_check = estimate_beta_tier2(
                gene, program,
                eqtl_data=eqtl_data, coloc_h4=coloc_h4,
                program_loading=program_loading,
            )
            if _t2_check is not None:
                _b1 = beta.get("beta") or 0.0
                _b2 = _t2_check.get("beta") or 0.0
                if _b1 * _b2 < 0 and abs(_b1) > 1e-10 and abs(_b2) > 1e-10:
                    _existing = beta.get("warnings") or []
                    beta = {
                        **beta,
                        "beta_sigma": max(float(beta.get("beta_sigma") or 0.0), 0.60),
                        "perturb_eqtl_discordant": True,
                        "warnings": _existing + [
                            f"Tier1 (Perturb-seq β={_b1:+.3f}) contradicts "
                            f"Tier2 eQTL-MR (β={_b2:+.3f}, COLOC H4={coloc_h4:.2f}). "
                            "Cell-line artefact or context-specific effect possible. "
                            "beta_sigma widened to 0.60."
                        ],
                    }
        return {**beta, "gene": gene, "program": program, "tier_used": 1}

    # Tier 2p — pQTL-MR (protein-level instrument; UKB-PPP / INTERVAL / deCODE)
    # Elevated above eQTL: the drug target IS the protein, so a protein QTL is
    # causally prior to an mRNA eQTL (which adds an extra transcription→translation step
    # with additional confounding). Genes with pQTL instruments get more direct causal
    # linkage to therapeutic effect.
    beta = estimate_beta_tier2_pqtl(gene, program, pqtl_data, program_loading)
    if beta is not None:
        return {**beta, "gene": gene, "program": program, "tier_used": 2}

    # Coding variant gate: if WES burden p < WES_CODING_MECHANISM_P, the gene likely
    # acts via protein-level changes. eQTL-MR (mRNA level) is the wrong instrument —
    # suppress Tier 2a/2b/2c. Tier 2p (pQTL) and Tier 2rb (rare burden) are unaffected.
    _coding_gated = _coding_mechanism_gate(burden_data)

    # Tier 2a — GTEx eQTL-MR
    if not _coding_gated:
        beta = estimate_beta_tier2(gene, program, eqtl_data=eqtl_data, coloc_h4=coloc_h4, program_loading=program_loading)
        if beta is not None:
            return {**beta, "gene": gene, "program": program, "tier_used": 2}

    # Tier 2b — OT credible-set instruments (GWAS fine-mapping + eQTL catalogue)
    if not _coding_gated:
        beta = estimate_beta_tier2_ot_instrument(gene, program, ot_instruments, program_loading)
        if beta is not None:
            return {**beta, "gene": gene, "program": program, "tier_used": 2}

    # Tier 2c — single-cell eQTL (cell-type specific; fills GTEx bulk gaps)
    if not _coding_gated:
        beta = estimate_beta_tier2_sc_eqtl(gene, program, sc_eqtl_data, coloc_h4, program_loading)
        if beta is not None:
            return {**beta, "gene": gene, "program": program, "tier_used": 2}

    # Tier 2L — Latent Hijack (Cross-disease mechanistic transfer)
    beta = estimate_beta_tier2L_latent_transfer(gene, program, current_disease_motif, motif_library)
    if beta is not None:
        return {**beta, "gene": gene, "program": program, "tier_used": 2}

    # Tier 2.5 — eQTL direction-only (eQTL present but COLOC H4 < 0.8 or absent)
    beta = estimate_beta_tier2_eqtl_direction(gene, program, eqtl_data, coloc_h4, program_loading)
    if beta is not None:
        return {**beta, "gene": gene, "program": program, "tier_used": 2}

    # Tier 2rb — rare variant burden direction (UKB WES collapsing test)
    beta = estimate_beta_tier2_rare_burden(gene, program, burden_data, program_loading)
    if beta is not None:
        return {**beta, "gene": gene, "program": program, "tier_used": 2}

    # No causal β found — return None rather than falling back to virtual/in silico estimates.
    # provisional_virtual has no perturbation or genetic basis and produces misleading output.
    return None


# ---------------------------------------------------------------------------
# β matrix construction
# ---------------------------------------------------------------------------

def build_beta_matrix(
    genes: list[str],
    programs: list[str],
    perturbseq_data: dict | None = None,
    eqtl_data: dict[str, dict] | None = None,
    coloc_data: dict[str, float] | None = None,
    program_loadings: dict[str, dict[str, float]] | None = None,
    program_gene_sets: dict[str, set[str]] | None = None,
    pathway_membership: dict[str, set[str]] | None = None,
    cell_type: str = "unknown",
    disease: str | None = None,
    # Phase Z7: Latent Hijack
    current_disease_motif: Any | None = None,
    motif_library: dict | None = None,
) -> ProgramBetaMatrix:
    """
    Build the full β_{gene×program} matrix for the Ota framework.

    Args:
        genes:             List of gene symbols
        programs:          List of program / gene-set names
        perturbseq_data:   Perturb-seq data keyed by gene
        eqtl_data:         GTEx eQTL data keyed by gene → {nes, se, tissue}
        coloc_data:        COLOC H4 posteriors keyed by gene
        program_loadings:  NMF/cNMF loadings keyed by program → gene → weight
        program_gene_sets: Gene set definitions keyed by program → set[gene]
        pathway_membership: Pathway membership keyed by program → set[gene]
        cell_type:         Primary cell type context (used for ProgramBetaMatrix annotation)
        disease:           Disease name (used for cell_type routing if cell_type not set)
        current_disease_motif: Phase Z7 motif for latent transfer
        motif_library:     Phase Z7 library of cross-disease motifs
    """
    # Resolve cell type from disease if not supplied
    if cell_type == "unknown" and disease:
        from graph.schema import DISEASE_CELL_TYPE_MAP
        ctx = DISEASE_CELL_TYPE_MAP.get(disease, {})
        cell_type = ctx.get("cell_line") or (ctx.get("cell_types") or ["unknown"])[0]

    matrix: dict[str, dict[str, float | None]] = {}
    tier_summary: dict[int, int] = {1: 0, 2: 0, 4: 0}

    for gene in genes:
        matrix[gene] = {}
        for program in programs:
            loading = (program_loadings or {}).get(program, {}).get(gene)
            pg_set  = (program_gene_sets or {}).get(program)
            pm      = gene in (pathway_membership or {}).get(program, set())

            beta_result = estimate_beta(
                gene=gene,
                program=program,
                perturbseq_data=perturbseq_data,
                eqtl_data=(eqtl_data or {}).get(gene),
                coloc_h4=(coloc_data or {}).get(gene),
                program_loading=loading,
                program_gene_set=pg_set,
                pathway_member=pm if pm else None,
                cell_type=cell_type,
                disease=disease,
                # Phase Z7
                current_disease_motif=current_disease_motif,
                motif_library=motif_library,
            )
            matrix[gene][program] = beta_result.get("beta") if beta_result else None
            if beta_result:
                tier_summary[beta_result.get("tier_used", 4)] += 1

    # Best tier per gene
    tier_priority = {
        "Tier1_Interventional": 1,
        "Tier2_Convergent":     2,
        "Tier2L_LatentHijack":  2,
        "Tier3_Provisional":    3,
        "provisional_virtual":  4,
    }
    valid_tiers = {"Tier1_Interventional", "Tier2_Convergent", "Tier2L_LatentHijack", "Tier3_Provisional"}
    evidence_tier_per_gene: dict[str, str] = {}

    for gene in genes:
        best = "provisional_virtual"
        for program in programs:
            b = estimate_beta(
                gene=gene, program=program,
                perturbseq_data=perturbseq_data,
                eqtl_data=(eqtl_data or {}).get(gene),
                coloc_h4=(coloc_data or {}).get(gene),
                program_loading=(program_loadings or {}).get(program, {}).get(gene),
                cell_type=cell_type,
                disease=disease,
                # Phase Z7
                current_disease_motif=current_disease_motif,
                motif_library=motif_library,
            )
            t = b.get("evidence_tier", "provisional_virtual") if b else "provisional_virtual"
            if tier_priority.get(t, 99) < tier_priority.get(best, 99):
                best = t
        evidence_tier_per_gene[gene] = best if best in valid_tiers else "Tier3_Provisional"

    # Replace None with NaN — distinguishes "no data" from "zero effect"
    float_matrix = {
        g: {p: (v if v is not None else math.nan) for p, v in progs.items()}
        for g, progs in matrix.items()
    }

    note = (
        f"β tiers — T1(cell-matched Perturb-seq)={tier_summary[1]}, "
        f"T2(eQTL-MR)={tier_summary[2]}, "
        f"Virtual={tier_summary[4]}"
    )

    perturb_source = perturbseq_data and f"Perturb-seq_{cell_type}" or "eQTL_MR_or_virtual"

    return ProgramBetaMatrix(
        programs=[
            {"program_id": p, "top_genes": [], "pathways": [], "cell_type": cell_type}
            for p in programs
        ],
        beta_matrix=float_matrix,
        evidence_tier_per_gene=evidence_tier_per_gene,
        cell_type=cell_type,
        perturb_seq_source=perturb_source,
        virtual_ensemble_vs_baseline={"note": note},
    )


# ---------------------------------------------------------------------------
# Convenience: CAD target gene β estimation
# ---------------------------------------------------------------------------

def estimate_cad_target_betas() -> ProgramBetaMatrix:
    """
    Estimate β for core CAD-relevant genes using Tier1 qualitative K562 data.
    Cell type = myeloid (K562) — appropriate for CAD CHIP/inflammatory programs.
    """
    from mcp_servers.burden_perturb_server import get_cnmf_program_info

    cad_genes = ["PCSK9", "LDLR", "HMGCR", "DNMT3A", "TET2", "ASXL1", "IL6R", "HLA-DRA", "CIITA"]
    programs_info = get_cnmf_program_info()
    programs = programs_info["programs"]

    return build_beta_matrix(
        genes=cad_genes,
        programs=programs,
        cell_type="K562",
        disease="CAD",
    )
