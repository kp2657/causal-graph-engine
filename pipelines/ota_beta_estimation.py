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

from models.evidence import CausalEdge, ProgramBetaMatrix, EvidenceTier
from config.scoring_thresholds import COLOC_H4_MIN, MR_PQTL_P_VALUE_MAX


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
    # AMD: RPE1 is retinal pigment epithelium — correct tissue match
    # Include all known dataset IDs and cell_line strings that refer to RPE1 data
    "AMD":  {"replogle_2022_rpe1", "RPE1", "rpe1", "RPE1_essential"},
    # CAD: Schnitzler HCASMC/HAEC and Natsume HAEC are vascular cell matches
    # Include dataset IDs, GEO accessions, and cell_line strings
    "CAD":  {"schnitzler_cad_vascular", "natsume_2023_haec",
             "HCASMC", "HAEC", "HCASMC_HAEC",
             "Schnitzler_GSE210681", "GSE210681"},
    # IBD: THP-1 monocyte and BMDC are gut-immune matches; K562 is also myeloid → accept
    "IBD":  {"papalexi_2021_thp1", "dixit_2016_bmdc", "replogle_2022_k562", "K562", "k562"},
    # RA: monocyte/immune match; K562 acceptable as myeloid proxy
    "RA":   {"papalexi_2021_thp1", "frangieh_2021_a375", "replogle_2022_k562", "K562", "k562"},
    # SLE: immune context; K562 acceptable as myeloid proxy
    "SLE":  {"papalexi_2021_thp1", "frangieh_2021_a375", "replogle_2022_k562", "K562", "k562"},
    # AD: iPSC neurons first; RPE1 and K562 are mismatched
    "AD":   {"ursu_2022_ipsc_neuron"},
    # T2D: no liver/pancreas match available; K562 accepted as best proxy for now
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
    _alias = {"CORONARY_ARTERY_DISEASE": "CAD", "AGE_RELATED_MACULAR_DEGENERATION": "AMD",
               "MACULAR_DEGENERATION": "AMD", "INFLAMMATORY_BOWEL_DISEASE": "IBD",
               "RHEUMATOID_ARTHRITIS": "RA", "SYSTEMIC_LUPUS_ERYTHEMATOSUS": "SLE",
               "ALZHEIMERS": "AD", "ALZHEIMERS_DISEASE": "AD", "TYPE_2_DIABETES": "T2D"}
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
# Tier 2: eQTL-MR (Mendelian Randomization via GTEx)
# ---------------------------------------------------------------------------

def estimate_beta_tier2(
    gene: str,
    program: str,
    eqtl_data: dict | None = None,
    coloc_h4: float | None = None,
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

    nes = eqtl_data.get("nes")
    if nes is None:
        return None

    # Scale by program loading when available; raw NES otherwise
    loading = program_loading if program_loading is not None else 1.0
    beta = nes * loading

    tissue = eqtl_data.get("tissue", "unknown_tissue")
    coloc_str = f"_COLOC_H4={coloc_h4:.2f}" if coloc_h4 is not None else ""

    return {
        "beta":          beta,
        "beta_se":       eqtl_data.get("se"),
        "ci_lower":      None,
        "ci_upper":      None,
        "beta_sigma":    (
            abs((eqtl_data.get("se") or 0.0) * (loading if loading is not None else 1.0))
            or abs(beta) * 0.25
        ),
        "evidence_tier": "Tier2_Convergent",
        "data_source":   f"GTEx_{tissue}_eQTL_MR{coloc_str}",
        "coloc_h4":      coloc_h4,
        "mr_method":     "eQTL_NES_x_loading",
        "note":          "MR-based: genetic instrument (cis-eQTL) for gene expression → program loading",
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
            "beta_sigma":    (abs(se * loading) if se else abs(beta) * 0.25),
            "evidence_tier": "Tier2_Convergent",
            "data_source":   f"OT_eQTL_catalogue_{ot_instruments.get('ensembl_id', gene)}",
            "instrument_type": "eqtl",
            "note":          "OT eQTL credible set × program loading (MR)",
        }

    # Fall back to GWAS credible set instrument
    best_gwas = next(
        (i for i in ot_instruments["instruments"] if i["instrument_type"] == "gwas_credset"),
        None,
    )
    if best_gwas:
        gwas_beta = best_gwas["beta"]
        se        = best_gwas.get("se")
        # Project GWAS variant effect onto program loading
        beta = gwas_beta * loading
        return {
            "beta":          beta,
            "beta_se":       (se * abs(loading)) if se else None,
            "ci_lower":      None,
            "ci_upper":      None,
            "beta_sigma":    (abs(se * loading) if se else abs(beta) * 0.35),
            "evidence_tier": "Tier2_Convergent",
            "data_source":   f"OT_GWAS_credset_{best_gwas.get('study_id', 'unknown')}",
            "instrument_type": "gwas_projected",
            "note":          (
                "GWAS credible-set beta × program loading. "
                "Direction inferred from genetic association, not direct perturbation."
            ),
        }

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
        sigma = abs((top.get("se") or 0.0) * loading) or abs(beta) * 0.25
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
    sigma   = abs((se_pqtl or 0.0) * loading) or abs(beta) * 0.30

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
# Tier 2s: Synthetic pathway program (Reactome enrichment-derived)
# For orphan genes absent from Perturb-seq screens.
# Position: after Tier2rb (rare burden), before Tier3 (LINCS).
# σ = 0.45 (weaker than eQTL but stronger than LINCS/Virtual).
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Tier 2pt: Protein Channel — direct gene→disease arc for protein-level mechanisms
#
# Captures genes whose GWAS causal variant changes protein function/level rather
# than mRNA.  Examples: CFH Y402H (complement), PCSK9 coding (lipid), LPA (CAD).
# These genes have n_programs≈0 in Perturb-seq because the mechanism is
# post-translational; eQTL tiers miss them for the same reason.
#
# Rather than routing through a transcriptional program, Tier2pt embeds its own
# program_gamma derived from (GWAS effect size × mechanism confidence).
# compute_ota_gamma uses program_gamma as fallback when gamma_estimates has no
# entry for the program — so the contribution is β × γ_protein directly.
# ---------------------------------------------------------------------------

def estimate_beta_tier2pt(
    gene: str,
    program: str,
    ot_instruments: dict | None = None,
    pqtl_data: dict | None = None,
    ot_score: float = 0.0,
) -> dict | None:
    """
    Tier 2pt β: direct protein-channel arc for protein-level disease mechanisms.

    Activates when:
      - gene has a GWAS credset instrument with p ≤ 1e-5
      - pQTL evidence OR OT L2G score ≥ 0.3 supports protein-level mechanism
      - Mechanism confidence threshold ≥ 0.15

    γ formula:
      gwas_strength    = clamp(-log10(gwas_p) / 15, 0, 1)
      mechanism_conf   = pqtl_conf × 0.8 + ot_score × 0.4   (if pQTL present)
                       = ot_score × 0.5                       (pQTL absent)
      program_gamma    = gwas_strength × mechanism_conf       (capped at 0.85)

    β = pQTL effect | GWAS beta | OT score  (in priority order, signed)

    The program_gamma is stored in the return dict so compute_ota_gamma can use
    it directly (gamma_estimates has no entry for this virtual program slot).

    Args:
        gene:         Gene symbol
        program:      Current Perturb-seq program (annotation only; not used for scoring)
        ot_instruments: Output of get_ot_genetic_instruments() — contains GWAS credsets
        pqtl_data:    pQTL data dict {top_pqtl: {beta, se, pvalue}} or flat {beta, pvalue}
        ot_score:     OT L2G score (causal gene probability at GWAS locus)

    Returns:
        β dict with evidence_tier="Tier2pt_ProteinChannel" and embedded program_gamma,
        or None if gene lacks sufficient protein-level evidence.
    """
    if not ot_instruments:
        return None

    instruments = ot_instruments.get("instruments", [])
    gwas_insts = [
        i for i in instruments
        if i.get("study_type") == "gwas" or i.get("instrument_type") == "gwas_credset"
    ]
    # Best GWAS credset by p-value
    if gwas_insts:
        best_gwas = min(gwas_insts, key=lambda x: float(x.get("p_value", 1.0) or 1.0))
        gwas_pvalue = float(best_gwas.get("p_value", 1.0) or 1.0)
        gwas_beta_raw = best_gwas.get("beta")
        if gwas_pvalue > 1e-5:
            return None  # no genome-wide-level signal
        gwas_strength = min(-math.log10(gwas_pvalue) / 15.0, 1.0)
    elif float(ot_score) >= 0.5:
        # No credible set instruments but high L2G confidence.
        # L2G ≥ 0.5 is only assigned to genes at GWAS-significant loci, so a
        # genome-wide significant signal exists — use 5e-8 as a conservative floor.
        # This rescues complement genes (CFH L2G=0.86) whose loci are too complex
        # for clean credible-set attribution (CFH/CFHR1-5 deletion polymorphism).
        gwas_pvalue  = 5e-8
        gwas_beta_raw = None
        gwas_strength = min(-math.log10(gwas_pvalue) / 15.0, 1.0)  # ≈ 0.503
    else:
        return None  # no genome-wide signal

    # pQTL evidence
    pqtl_conf = 0.0
    pqtl_pvalue: float | None = None
    pqtl_beta_val: float | None = None
    if pqtl_data is not None:
        top = pqtl_data.get("top_pqtl") or pqtl_data
        if isinstance(top, dict):
            raw_p = top.get("pvalue") or top.get("p_value")
            raw_b = top.get("beta")
            if raw_p is not None:
                try:
                    pqtl_pvalue = float(raw_p)
                    if math.isfinite(pqtl_pvalue) and pqtl_pvalue < MR_PQTL_P_VALUE_MAX:
                        pqtl_conf = max(0.0, 1.0 - pqtl_pvalue / MR_PQTL_P_VALUE_MAX)
                except (TypeError, ValueError):
                    pass
            if raw_b is not None:
                try:
                    pqtl_beta_val = float(raw_b)
                    if not math.isfinite(pqtl_beta_val):
                        pqtl_beta_val = None
                except (TypeError, ValueError):
                    pass

    causal_prob = float(ot_score) if ot_score and ot_score > 0 else 0.4

    if pqtl_conf > 0:
        mechanism_conf = min(pqtl_conf * 0.8 + causal_prob * 0.4, 1.0)
    else:
        mechanism_conf = causal_prob * 0.5

    if mechanism_conf < 0.15:
        return None

    program_gamma = round(min(gwas_strength * mechanism_conf, 0.85), 4)

    # Beta: priority pQTL beta → GWAS beta → OT score
    if pqtl_beta_val is not None:
        beta_val = max(-1.0, min(1.0, pqtl_beta_val))
        sigma = 0.35
    elif gwas_beta_raw is not None and abs(float(gwas_beta_raw)) >= 0.05:
        # Use GWAS beta only when it's on an interpretable effect scale (≥0.05).
        # OT credset betas are per-allele log-odds (~0.018 for AMD/CFH) which are
        # not comparable to pQTL effect sizes — using them directly as β in the OTA
        # formula produces ~50× underestimated gamma. Fall through to OT L2G proxy
        # when the allelic beta is too small to be useful.
        try:
            beta_val = max(-1.0, min(1.0, float(gwas_beta_raw)))
        except (TypeError, ValueError):
            beta_val = float(causal_prob)
        sigma = 0.40
    else:
        beta_val = float(causal_prob)
        sigma = 0.45

    parts = [f"GWAS_p={gwas_pvalue:.1e}"]
    if pqtl_pvalue is not None and pqtl_pvalue < MR_PQTL_P_VALUE_MAX:
        parts.append(f"pQTL_p={pqtl_pvalue:.3f}")
    parts.append(f"L2G={causal_prob:.2f}")

    return {
        "beta":          beta_val,
        "beta_se":       None,
        "ci_lower":      None,
        "ci_upper":      None,
        "beta_sigma":    sigma,
        "evidence_tier": "Tier2pt_ProteinChannel",
        "program_gamma": program_gamma,
        "data_source":   f"ProteinChannel:{','.join(parts)}",
        "note": (
            f"Protein channel γ={program_gamma:.3f} "
            f"(gwas_str={gwas_strength:.2f}, mech_conf={mechanism_conf:.2f}"
            + (f", pQTL_conf={pqtl_conf:.2f}" if pqtl_conf > 0 else "") + ")"
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

    # β direction: LoF burden positive → gene normally protective → activation helps
    # We use sign(burden_beta) as the direction signal
    # magnitude = |loading| (we don't trust the burden beta magnitude for program effects)
    beta = math.copysign(abs(loading), burden_beta)

    loeuf_str = f"_LOEUF={loeuf:.3f}" if loeuf is not None else ""
    study = burden_data.get("burden_study") or "UKB_WES"

    return {
        "beta":          beta,
        "beta_se":       burden_data.get("burden_se"),
        "ci_lower":      None,
        "ci_upper":      None,
        "beta_sigma":    0.60,  # large: direction from burden, magnitude from loading only
        "evidence_tier": "Tier2rb_RareBurden",
        "data_source":   f"{study}_rare_burden{loeuf_str}",
        "burden_p":      burden_p,
        "loeuf":         loeuf,
        "mr_method":     "rare_burden_direction_x_loading",
        "note":          (
            f"Rare variant burden (p={burden_p:.2e}): LoF/damaging direction × loading. "
            "Direction-only; magnitude uncertain (sigma=0.60). "
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
    # Direction from eQTL sign, magnitude from |loading| (not scaled by NES magnitude
    # because NES calibration is uncertain without confirmed COLOC)
    beta = math.copysign(abs(loading), float(nes))

    tissue = eqtl_data.get("tissue", "unknown_tissue")
    coloc_str = f"H4={coloc_h4:.2f}" if coloc_h4 is not None else "H4=absent"

    return {
        "beta":          beta,
        "beta_se":       None,
        "ci_lower":      None,
        "ci_upper":      None,
        "beta_sigma":    0.50,  # large: direction known, magnitude uncertain (COLOC weak)
        "evidence_tier": "Tier2_eQTL_direction",
        "data_source":   f"GTEx_{tissue}_eQTL_direction_only_{coloc_str}",
        "coloc_h4":      coloc_h4,
        "mr_method":     "eQTL_sign_x_loading",
        "note":          (
            f"eQTL exists ({tissue}, NES={float(nes):.3f}) but COLOC {coloc_str} — "
            "direction preserved, magnitude set to |loading| with beta_sigma=0.50."
        ),
    }


# ---------------------------------------------------------------------------
# Tier 3: LINCS L1000 genetic perturbation
# ---------------------------------------------------------------------------

def estimate_beta_tier3(
    gene: str,
    program: str,
    lincs_signature: dict | None = None,
    program_gene_set: set[str] | None = None,
    cell_line: str | None = None,
) -> dict | None:
    """
    Tier 3 β: LINCS L1000 genetic perturbation signature.

    LINCS L1000 measures transcriptional response to shRNA knockdown or ORF
    overexpression across ~9 cancer cell lines.  This is direct perturbation
    data (not co-expression) but in a cell line that may not match the
    disease-relevant cell type.

    β_{gene→P} = overlap score between gene X knockdown signature and
                 program P gene set (Jaccard or weighted mean log2FC).

    Args:
        gene:             Gene symbol
        program:          Program name
        lincs_signature:  L1000 differential expression dict:
                          {gene_symbol → {log2fc, z_score}} for gene X KD
        program_gene_set: Set of gene symbols defining program P
        cell_line:        LINCS cell line used (A375, HT29, MCF7, etc.)
    """
    if lincs_signature is None or program_gene_set is None:
        return None

    # Compute weighted overlap: mean log2fc of program genes in KD signature.
    # Handles both {gene: float} (iLINCS flat) and {gene: {"log2fc": float}} shapes.
    def _extract_log2fc(v: object) -> float:
        if isinstance(v, dict):
            return float(v.get("log2fc") or v.get("Value") or 0.0)
        return float(v)

    program_hits = {
        g: _extract_log2fc(lincs_signature[g])
        for g in program_gene_set
        if g in lincs_signature
    }
    if not program_hits:
        return None

    # Sign-preserving mean effect of KD on program genes
    beta = sum(program_hits.values()) / len(program_gene_set)
    coverage = len(program_hits) / len(program_gene_set)

    if coverage < 0.05:  # < 5% of program genes measured — too sparse
        return None

    line_str = cell_line or "unknown_cell_line"
    return {
        "beta":          beta,
        "beta_se":       None,
        "ci_lower":      None,
        "ci_upper":      None,
        "beta_sigma":    abs(beta) * 0.35,
        "evidence_tier": "Tier3_Provisional",
        "data_source":   f"LINCS_L1000_{line_str}_shRNA",
        "coverage":      round(coverage, 3),
        "note":          (
            f"Direct perturbation (shRNA KD) in {line_str}; "
            "cell line may not match disease-relevant cell type"
        ),
    }


# ---------------------------------------------------------------------------
# Tier 4 / Virtual: In silico prediction
# Two sub-sources — both labelled provisional_virtual
# ---------------------------------------------------------------------------

def estimate_beta_geneformer(
    gene: str,
    program: str,
    geneformer_result: dict | None = None,
) -> dict | None:
    """
    Virtual β sub-source A: Geneformer / GEARS in silico perturbation.

    Models trained on real Perturb-seq data (Replogle K562 + Norman K562)
    that extrapolate to unseen genes or cell types.  Superior to co-expression
    because the model learned from actual perturbation distributions, but still
    extrapolation — not experimental.

    STUB — wire to Geneformer API or local model when available.
    """
    if geneformer_result is None:
        return None
    beta = geneformer_result.get("delta_program_activity")
    if beta is None:
        return None
    return {
        "beta":          beta,
        "beta_se":       geneformer_result.get("se"),
        "ci_lower":      None,
        "ci_upper":      None,
        "beta_sigma":    geneformer_result.get("se") or abs(beta) * 0.50,
        "evidence_tier": "provisional_virtual",
        "data_source":   "Geneformer_virtual_perturbation",
        "note":          "In silico prediction from Geneformer; not experimental — label provisional_virtual",
    }


def estimate_beta_foundation_model(
    gene: str,
    program: str,
    known_tier12_betas: dict[str, float] | None = None,
    program_gene_symbols: list[str] | None = None,
) -> dict | None:
    """
    Virtual-A (upgraded): Expression-similarity transfer from known Tier1/2 genes.

    Approximates what scGPT / Geneformer v2 / Universal Cell Embeddings do:
    find nearest neighbors in expression embedding space, then transfer their
    perturbation responses.  Here we use GTEx v10 median TPM profiles as the
    embedding (54 tissues × gene → cosine similarity).

    This gives a proper uncertainty-weighted prior for genes with no direct
    Perturb-seq coverage, rather than binary pathway membership (0/1).

    β_prior = Σ_i cos_sim(gene, known_i) × β_known_i / Σ cos_sim
    σ_prior = max(0.5 × |β_prior|, 0.30)   [calibrated: min 0.30]

    Args:
        gene:                 Target gene to estimate β for
        program:              Program name
        known_tier12_betas:   {gene_symbol: beta_value} for genes in the same
                              program that already have Tier1 or Tier2 evidence
        program_gene_symbols: All gene symbols defining program (for context)
    """
    if not known_tier12_betas:
        return None

    from mcp_servers.single_cell_server import _query_gtex_v10_median_expression

    # Fetch embedding for target gene
    target_tpm = _query_gtex_v10_median_expression(gene)
    if not target_tpm or len(target_tpm) < 3:
        return None

    target_norm = _l2_norm(target_tpm)
    if target_norm == 0:
        return None

    # Weighted transfer from known Tier1/2 genes
    weighted_sum = 0.0
    weight_total = 0.0
    for known_gene, known_beta in known_tier12_betas.items():
        if known_gene == gene:
            continue
        known_tpm = _query_gtex_v10_median_expression(known_gene)
        if not known_tpm or len(known_tpm) < 3:
            continue
        # Cosine similarity in GTEx expression space
        cos_sim = _cosine_sim(target_tpm, known_tpm, target_norm)
        if cos_sim <= 0:
            continue
        weighted_sum  += cos_sim * known_beta
        weight_total  += cos_sim

    if weight_total == 0:
        return None

    beta_prior = weighted_sum / weight_total
    sigma      = max(0.50 * abs(beta_prior), 0.30)

    return {
        "beta":          beta_prior,
        "beta_se":       None,
        "ci_lower":      None,
        "ci_upper":      None,
        "beta_sigma":    sigma,
        "evidence_tier": "provisional_virtual",
        "data_source":   f"foundation_model_GTEx_transfer_{len(known_tier12_betas)}_donors",
        "note":          (
            "Cosine-similarity transfer from known Tier1/2 genes in GTEx v10 "
            "expression space. Approximates scGPT/UCE nearest-neighbor transfer. "
            "Labelled provisional_virtual; represents uncertainty-weighted prior."
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
    lincs_signature: dict | None = None,
    program_gene_set: set[str] | None = None,
    geneformer_result: dict | None = None,
    known_tier12_betas: dict[str, float] | None = None,
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
      2. Tier2a   — GTEx eQTL-MR (COLOC H4 ≥ 0.8)
      2. Tier2b   — OT credible-set instruments (GWAS/eQTL, when GTEx absent)
      2. Tier2c   — sc-eQTL from eQTL Catalogue (cell-type specific; OneK1K / Blueprint)
      2. Tier2p   — pQTL-MR (protein-level instrument; UKB-PPP / INTERVAL / deCODE)
      2. Tier2L   — Latent Hijack (Cross-disease mechanistic transfer)
      2. Tier2.5  — eQTL direction-only (eQTL present but COLOC H4 < 0.8; sigma=0.50)
      2. Tier2rb  — Rare variant burden direction (UKB WES collapsing; sigma=0.60)
      3. Tier3    — LINCS L1000 perturbation (direct but cell-line mismatched)
      4. Virtual-A — Geneformer in silico (trained on perturbation data)
      4. Virtual-A upgraded — foundation model cosine-similarity transfer
      5. Virtual-B — pathway membership (annotation proxy, no causal basis)

    Co-expression, GRN weights, and pathway-annotation-derived synthetic betas
    are intentionally absent from this chain — they do not provide causal evidence.
    """
    # Tier 1 — demote to Tier3 if cell type is not disease-relevant
    _matched = _is_cell_type_matched(cell_type, disease)
    beta = estimate_beta_tier1(gene, program, perturbseq_data, cell_type=cell_type, cell_type_matched=_matched)
    if beta is not None:
        _tier_used = 1 if _matched else 3
        return {**beta, "gene": gene, "program": program, "tier_used": _tier_used}

    # Tier 2a — GTEx eQTL-MR
    beta = estimate_beta_tier2(gene, program, eqtl_data, coloc_h4, program_loading)
    if beta is not None:
        return {**beta, "gene": gene, "program": program, "tier_used": 2}

    # Tier 2b — OT credible-set instruments (GWAS fine-mapping + eQTL catalogue)
    beta = estimate_beta_tier2_ot_instrument(gene, program, ot_instruments, program_loading)
    if beta is not None:
        return {**beta, "gene": gene, "program": program, "tier_used": 2}

    # Tier 2c — single-cell eQTL (cell-type specific; fills GTEx bulk gaps)
    beta = estimate_beta_tier2_sc_eqtl(gene, program, sc_eqtl_data, coloc_h4, program_loading)
    if beta is not None:
        return {**beta, "gene": gene, "program": program, "tier_used": 2}

    # Tier 2p — pQTL-MR (protein-level instrument; UKB-PPP / INTERVAL / deCODE)
    beta = estimate_beta_tier2_pqtl(gene, program, pqtl_data, program_loading)
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

    # Tier 3
    beta = estimate_beta_tier3(gene, program, lincs_signature, program_gene_set, cell_line)
    if beta is not None:
        return {**beta, "gene": gene, "program": program, "tier_used": 3}

    # Virtual-A: Geneformer
    beta = estimate_beta_geneformer(gene, program, geneformer_result)
    if beta is not None:
        return {**beta, "gene": gene, "program": program, "tier_used": 4}

    # Virtual-A upgraded: foundation model cosine-similarity transfer
    beta = estimate_beta_foundation_model(gene, program, known_tier12_betas, program_gene_set and list(program_gene_set))
    if beta is not None:
        return {**beta, "gene": gene, "program": program, "tier_used": 4}

    # Virtual-B: pathway membership proxy
    return {
        **estimate_beta_virtual(gene, program, pathway_member),
        "gene": gene, "program": program, "tier_used": 4,
    }


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
    lincs_data: dict[str, dict] | None = None,
    program_gene_sets: dict[str, set[str]] | None = None,
    geneformer_data: dict[str, dict] | None = None,
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
        lincs_data:        L1000 KD signatures keyed by gene → {gene_symbol → {log2fc}}
        program_gene_sets: Gene set definitions keyed by program → set[gene]
        geneformer_data:   Geneformer outputs keyed by gene → program → {delta_activity}
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
    tier_summary: dict[int, int] = {1: 0, 2: 0, 3: 0, 4: 0}

    for gene in genes:
        matrix[gene] = {}
        for program in programs:
            loading = (program_loadings or {}).get(program, {}).get(gene)
            pg_set  = (program_gene_sets or {}).get(program)
            pm      = gene in (pathway_membership or {}).get(program, set())
            lincs_sig = (lincs_data or {}).get(gene)

            gf_result = None
            if geneformer_data and gene in geneformer_data:
                gf_result = geneformer_data[gene].get(program)

            beta_result = estimate_beta(
                gene=gene,
                program=program,
                perturbseq_data=perturbseq_data,
                eqtl_data=(eqtl_data or {}).get(gene),
                coloc_h4=(coloc_data or {}).get(gene),
                program_loading=loading,
                lincs_signature=lincs_sig,
                program_gene_set=pg_set,
                geneformer_result=gf_result,
                pathway_member=pm if pm else None,
                cell_type=cell_type,
                disease=disease,
                # Phase Z7
                current_disease_motif=current_disease_motif,
                motif_library=motif_library,
            )
            matrix[gene][program] = beta_result.get("beta")
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
            t = b.get("evidence_tier", "provisional_virtual")
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
        f"T3(LINCS L1000)={tier_summary[3]}, "
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
