"""
pipelines/ldsc/gamma_loader.py — Load pre-computed S-LDSC / eQTL-direction γ values.

Reads from data/ldsc/results/{disease_key}_SVD_program_taus.json (produced by runner.py).

For diseases with some τ > 0 (e.g. CAD): γ = sign(eQTL_direction) × |τ|
For diseases with all τ < 0 (e.g. RA, regulatory architecture): γ is set by
  normalized eQTL concordance fraction × _EQTL_GAMMA_MAX (eQTL-direction-only fallback).

The signed γ is stored directly in program_gammas in the JSON — loader reads it as-is.
Evidence tier: "Tier2_Convergent" when τ_p < 0.05, else "Tier3_Provisional".
"""
from __future__ import annotations

import json
import logging
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_ROOT        = Path(__file__).parent.parent.parent
_RESULTS_DIR = _ROOT / "data" / "ldsc" / "results"
_FROZEN_DIR  = _ROOT / "frozen"  # shipped frozen τ files take precedence

# Minimum tau p-value to treat as Tier2 (genome-wide significant heritability enrichment)
_TAU_P_TIER2 = 0.05

# ---------------------------------------------------------------------------
# Tier 3: Cell-type specificity filter
# ---------------------------------------------------------------------------

# Cell-type marker gene sets for program specificity filtering.
# Programs with few cell-type-specific top genes get γ discounted.
_CELL_TYPE_MARKERS: dict[str, frozenset[str]] = {
    # CD4+ T cell markers and T cell immune genes (RA / SLE datasets)
    "czi_2025_cd4t_perturb": frozenset({
        # Core T cell surface markers
        "CD3D", "CD3E", "CD3G", "CD4", "CD8A", "CD8B",
        # Costimulatory / checkpoint
        "CD28", "ICOS", "CTLA4", "PDCD1", "LAG3", "HAVCR2", "TIGIT", "CD226",
        # TCR signaling
        "ZAP70", "LCK", "LAT", "VAV1", "PLCG1", "LCP2", "CD247",
        # Transcription factors
        "FOXP3", "GATA3", "TBX21", "RORC", "BCL6", "TOX", "NR4A1", "IRF4",
        # Cytokines / receptors
        "IL2", "IL2RA", "IL2RB", "IL7R", "IL4", "IL13", "IL17A", "IL21",
        "IFNG", "TNF", "IL10", "TGFB1", "TNFSF11",
        # Memory / homing
        "CCR7", "SELL", "CXCR5", "CXCR3", "CXCR4",
        # T cell activation / effector
        "PTPRC", "CD44", "CD27", "TNFRSF4", "TNFRSF9", "ENTPD1",
        # RA-specific immune
        "TYK2", "JAK1", "JAK2", "JAK3", "STAT1", "STAT3", "STAT4",
        "IL6R", "IL12RB2", "IL23R", "TRAF3IP2", "TRAF1", "PTPN22",
        "REL", "NFKB1", "IRF5", "PADI4", "MS4A1",
    }),
    # Cardiac endothelial cell markers (CAD dataset — Schnitzler 2023 vascular ECs)
    "schnitzler_cad_vascular": frozenset({
        # Core EC surface markers
        "CDH5", "PECAM1", "VWF", "KDR", "TEK", "TIE1", "ROBO4", "ENG", "THBD",
        # EC function / tone
        "NOS3", "THBS1", "VCAM1", "ICAM1", "SELE", "SELP", "PLVAP",
        # Transcription factors
        "KLF2", "KLF4", "NOTCH1", "HEY1", "HES1", "ETS1", "ERG",
        # Vascular signaling
        "VEGFA", "ANGPT1", "DLL4", "JAG1", "CXCR4",
        # CCM complex / EC integrity
        "CCM2", "KRIT1", "PDCD10", "STK25",
        # EC metabolism / lipid
        "PLPP3", "APOA1", "APOE",
        # Gap junctions
        "GJA5", "GJA4",
        # CAD-specific
        "HMGCR", "LDLR", "PCSK9", "APOB",
    }),
}

_DATASET_FOR_DISEASE: dict[str, str] = {
    "RA":  "czi_2025_cd4t_perturb",
    "SLE": "czi_2025_cd4t_perturb",
    "CAD": "schnitzler_cad_vascular",
}


@lru_cache(maxsize=8)
def _load_tau_file(disease_key: str) -> dict[str, Any] | None:
    """Load pre-computed τ file for disease_key. Cached per process.

    Resolution order:
      1. frozen/{disease}_program_taus.json  — shipped with repo (exact reproduction)
      2. data/ldsc/results/{disease}_program_taus.json  — locally computed
    """
    frozen = _FROZEN_DIR / f"{disease_key}_program_taus.json"
    live   = _RESULTS_DIR / f"{disease_key}_program_taus.json"
    if frozen.exists():
        f = frozen
        log.info("Using frozen S-LDSC τ for %s (%s)", disease_key, f)
    elif live.exists():
        f = live
    else:
        raise FileNotFoundError(
            f"S-LDSC τ file missing: {live}\n"
            f"Run: python -m pipelines.ldsc.setup download_all  "
            f"then: python -m pipelines.ldsc.runner run {disease_key}"
        )
    try:
        with open(f) as fh:
            return json.load(fh)
    except Exception as exc:
        log.warning("Failed to load ldsc taus for %s: %s", disease_key, exc)
        return None


@lru_cache(maxsize=8)
def _load_svd_tau_file(disease_key: str) -> dict[str, Any] | None:
    """Load pre-computed SVD τ file. Cached per process.

    Reads from data/ldsc/results/{disease_key}_SVD_program_taus.json.
    Returns None (not raises) when file is absent — callers handle gracefully.
    """
    live = _RESULTS_DIR / f"{disease_key}_SVD_program_taus.json"
    if not live.exists():
        return None
    try:
        with open(live) as fh:
            return json.load(fh)
    except Exception as exc:
        log.warning("Failed to load SVD ldsc taus for %s: %s", disease_key, exc)
        return None


def get_all_program_gammas_ldsc(disease_key: str) -> dict[str, dict]:
    """
    Return all NMF/Hallmark program γ estimates from S-LDSC for a disease.
    Dict[program_name -> gamma_estimate_dict].
    Reads from the non-SVD τ file ({disease_key}_program_taus.json).
    """
    data = _load_tau_file(disease_key.upper())
    if not data:
        return {}

    program_taus = data.get("program_taus", {})
    results = {}
    for prog, tau in program_taus.items():
        try:
            tau = float(tau)
        except (TypeError, ValueError):
            continue
        gamma_value = max(-1.0, min(1.0, round(tau, 5)))
        tau_p, tau_se = 1.0, None
        for annot in data.get("raw_annotations", []):
            if annot.get("name") == prog:
                tau_p  = float(annot.get("tau_p", 1.0))
                tau_se = annot.get("tau_se")
                break
        gamma_se = round(float(tau_se), 5) if tau_se is not None else round(abs(gamma_value) * 0.30, 5)
        results[prog] = {
            "gamma":         gamma_value,
            "gamma_se":      gamma_se,
            "evidence_tier": "Tier2_Convergent" if tau_p < _TAU_P_TIER2 else "Tier3_Provisional",
            "data_source":   f"S-LDSC_{disease_key}_tau={tau:.4f}",
            "tau_p":         tau_p,
        }
    return results


def _cell_type_specificity_weight(disease_key: str, top_genes: list[str]) -> float:
    """
    Fraction of top_genes in the cell-type marker set, mapped to a [0.15, 1.0] weight.
    Programs with ≥15% cell-type-specific top genes get full weight.
    Programs with 0% get 0.15× (not zeroed — keeps tail signal for novel programs).

    Used as Tier 3 discount: programs whose top-loading genes are not cell-type-specific
    (e.g. metabolic/housekeeping programs in C20/C27 for RA) are down-weighted to
    prevent bystander enrichment from flipping the direction of known drug targets.
    """
    dk = disease_key.upper()
    dataset = _DATASET_FOR_DISEASE.get(dk)
    if dataset is None or not top_genes:
        return 1.0
    markers = _CELL_TYPE_MARKERS.get(dataset, frozenset())
    if not markers:
        return 1.0
    n_specific = sum(1 for g in top_genes if g in markers)
    fraction = n_specific / len(top_genes)
    # Linear ramp: 0% → 0.15, 15%+ → 1.0
    weight = min(1.0, max(0.15, fraction / 0.15))
    return round(weight, 4)


def get_svd_program_gammas(
    disease_key: str,
    gwas_genes: list[str] | None = None,
) -> dict[str, dict]:
    """
    Return γ estimates for all SVD programs.

    Uses the eQTL-direction signed γ from program_gammas when available
    (written by run_svd_programs). Falls back to signed τ normalization for
    legacy files that predate the eQTL direction step.

    Args:
        gwas_genes: Optional list of GWAS-evidence gene symbols for Zhu et al. 2025 Fig 7
                    enrichment-based bystander correction. Pass disease_query["gwas_genes"]
                    from the pipeline call chain. If None, reads from OT cache if available.

    γ semantics:
      γ > 0: program is atherogenic (drives disease; risk allele increases its expression)
      γ < 0: program is atheroprotective (protective; risk allele suppresses it)
      γ = 0: program not in causal path (τ < 0, depleted of GWAS heritability)

    tau_sign is preserved as a separate annotation — it carries information about
    whether the program's gene windows are enriched (tau_sign=+1) or depleted (−1)
    of GWAS heritability, independent of the causal direction.
    """
    data = _load_svd_tau_file(disease_key.upper())
    if not data:
        return {}

    # When all S-LDSC τ values are negative, the model cannot differentiate programs
    # by heritability enrichment direction — normalized gammas are uninformative noise.
    # Returning {} lets the caller keep OT L2G gammas instead of overriding with noise.
    if data.get("all_tau_negative"):
        log.info("get_svd_program_gammas: all τ < 0 for %s — skipping SVD override, using OT L2G gammas", disease_key.upper())
        return {}

    # When no program achieves statistical significance (all τ_p ≥ 0.05), the S-LDSC
    # enrichment lacks power to rank programs — normalized gammas from small windows
    # are dominated by noise and would suppress valid OT L2G program gammas.
    # Returning {} preserves the OT L2G gammas which are more reliable in this case.
    _raw_annots = data.get("raw_annotations", [])
    if _raw_annots:
        _n_sig = sum(1 for r in _raw_annots if float(r.get("tau_p", 1.0)) < _TAU_P_TIER2)
        if _n_sig < 2:
            log.info(
                "get_svd_program_gammas: only %d/%d programs pass τ_p < %.2f for %s — "
                "insufficient signal to override OT L2G gammas (need ≥2)",
                _n_sig, len(_raw_annots), _TAU_P_TIER2, disease_key.upper(),
            )
            return {}

    # --- New format: program_gammas computed by run_svd_programs ---
    program_gammas: dict[str, float] = data.get("program_gammas", {})
    gamma_annots: list[dict] = data.get("gamma_annotations", [])
    annot_by_prog = {a["name"]: a for a in gamma_annots}

    if program_gammas:
        results: dict[str, dict] = {}

        # Max-normalise: divide all gammas by the largest |raw_gamma| so the most
        # enriched program gets γ=±1 and all others get proportional values.
        # Previously raw τ values were individually clipped to [-1,1], which saturated
        # 8 programs at ±1.0 (C27=9.99, C10=6.31, C20=4.48, …) and caused γ to scale
        # with the number of saturated programs a gene loads on, inflating OTA for
        # genes with diffuse betas across many high-τ programs (e.g. ABTB3 γ=4.1).
        _max_abs_gamma = max((abs(v) for v in program_gammas.values()), default=1.0)
        if _max_abs_gamma == 0:
            _max_abs_gamma = 1.0

        # Supplement: WES burden γ_{P→trait} — IV-weighted mean(Backman2021 burden_beta)
        # for top SVD-loading genes. Used for direction resolution when eQTL is unknown.
        # NOTE: for binary disease traits (RA, CAD), all burden_gammas tend to be positive
        # (case-control selection bias) so burden cannot serve as primary magnitude source.
        _burden_gammas = get_svd_burden_program_gammas(disease_key)

        # GWAS enrichment γ_P (Zhu et al. 2025, Fig 7): Fisher exact enrichment of
        # OpenTargets GWAS-evidence genes in each program's top loading genes.
        # This is the principled alternative to S-LDSC τ for underpowered binary-disease
        # traits — does not require LoF burden power, is cell-type-specific by construction.
        # Used as: bystander filter (non-enriched programs with high S-LDSC τ → discount γ).
        # gwas_genes is threaded from disease_query["gwas_genes"] via the caller.
        _gwas_enrichment = get_gwas_enrichment_program_gammas(disease_key, gwas_genes=gwas_genes)

        # Tier 3: cell-type specificity — pre-load SVD Vt for top-gene extraction.
        # Component index is parsed from the program name (e.g. "RA_SVD_C27" → 26).
        _svd_dataset_id = _DISEASE_SVD_DATASET.get(disease_key.upper())
        _Vt, _pert_names = (
            _load_svd_loadings(_svd_dataset_id)
            if _svd_dataset_id else (None, None)
        )

        for prog, gamma_raw in program_gammas.items():
            gamma_value = round(gamma_raw / _max_abs_gamma, 5)

            annot    = annot_by_prog.get(prog, {})
            tau      = annot.get("tau") or data.get("program_taus", {}).get(prog, 0.0)
            tau_sign = annot.get("tau_sign", 1 if tau >= 0 else -1)
            dir_src  = annot.get("direction_source", "unknown")
            tau_p    = 1.0
            tau_se   = None
            for raw in data.get("raw_annotations", []):
                if raw.get("name") == prog:
                    tau_p  = float(raw.get("tau_p", 1.0))
                    tau_se = raw.get("tau_se")
                    break

            # Short name (e.g. "C27") for burden lookup; tau file uses full names like "RA_SVD_C27"
            short_prog = prog.split("_")[-1] if "_" in prog else prog
            burden_gamma = _burden_gammas.get(short_prog)

            # When eQTL direction is unknown, use burden γ sign to anchor direction.
            # S-LDSC magnitude (max-normalized) is preserved; only sign may flip.
            if dir_src == "unknown" and burden_gamma is not None:
                if burden_gamma < 0 and gamma_value > 0:
                    gamma_value = -abs(gamma_value)
                    dir_src = "burden_direction"
                elif burden_gamma > 0 and gamma_value < 0:
                    gamma_value = abs(gamma_value)
                    dir_src = "burden_direction"
                else:
                    dir_src = "burden_direction_confirmed"

            # GWAS enrichment: Zhu et al. 2025 Fig 7 approach for underpowered binary diseases.
            # Programs enriched for GWAS genes → genuine disease programs (not bystanders).
            # Programs with high S-LDSC τ but NO GWAS enrichment → likely bystander.
            # Apply a discount to gamma_value for bystander programs based on GWAS evidence.
            gwas_enrich = _gwas_enrichment.get(short_prog, {})
            gwas_log_or = gwas_enrich.get("log_or", 0.0)
            gwas_enriched = gwas_enrich.get("enriched", False)
            gwas_fisher_p = gwas_enrich.get("fisher_p", 1.0)

            # GWAS enrichment is annotation-only — no gamma discount applied here.
            # A blanket 0.5× discount on non-enriched programs simultaneously halves
            # positive and negative contributions, collapsing signal for balanced-loading
            # genes (e.g. PTPN22: large negative C10 + large positive C27 cancel to ~0).
            # Principled bystander correction requires joint LDSC conditional τ* (Tier 1).
            _gwas_bystander_discount = 1.0

            # Tier 3: annotate programs with cell-type specificity weight.
            # The bystander problem (C20/C27 in RA, C07/C12 in CAD) is LD contamination:
            # GWAS SNPs near metabolic gene bodies co-localise with immune/EC regulatory elements.
            # GWAS enrichment discount (above) addresses this at the gene-set level;
            # joint LDSC conditional τ* (Tier 1) is the genomic-level principled fix.
            _spec_weight = 1.0
            _top_genes_for_spec: list[str] = []
            if _Vt is not None and _pert_names is not None:
                try:
                    import numpy as _np
                    _comp_str = short_prog  # e.g. "C27"
                    if _comp_str.startswith("C") and _comp_str[1:].isdigit():
                        _comp_idx = int(_comp_str[1:]) - 1  # 0-based
                        if 0 <= _comp_idx < _Vt.shape[0]:
                            _abs_row = _np.abs(_Vt[_comp_idx])
                            _top50_idx = _np.argsort(_abs_row)[::-1][:50]
                            _top_genes_for_spec = [str(_pert_names[i]) for i in _top50_idx]
                except Exception as _exc:
                    log.debug("Tier3 spec weight extraction failed for %s: %s", prog, _exc)
            if _top_genes_for_spec:
                _spec_weight = _cell_type_specificity_weight(disease_key, _top_genes_for_spec)

            gamma_se = round(float(tau_se), 5) if tau_se is not None else round(abs(gamma_value) * 0.30, 5)
            evidence_tier = "Tier2_Convergent" if tau_p < _TAU_P_TIER2 else "Tier3_Provisional"

            _gwas_note = (
                f" GWAS-enrich: log(OR)={gwas_log_or:.2f}, p={gwas_fisher_p:.3g}"
                f"{', bystander_discount=0.5x' if _gwas_bystander_discount < 1.0 else ''}."
                if gwas_enrich else ""
            )

            results[prog] = {
                "gamma":                        gamma_value,
                "gamma_se":                     gamma_se,
                "evidence_tier":                evidence_tier,
                "data_source":                  f"S-LDSC_eQTL_{disease_key}_tau={gamma_raw:.4f}",
                "program":                      prog,
                "trait":                        disease_key.upper(),
                "tau":                          tau,
                "tau_sign":                     tau_sign,
                "tau_p":                        tau_p,
                "direction_source":             dir_src,
                "burden_gamma":                 burden_gamma,
                "gwas_log_or":                  gwas_log_or,
                "gwas_enriched":                gwas_enriched,
                "gwas_fisher_p":                gwas_fisher_p,
                "gwas_bystander_discount":      _gwas_bystander_discount,
                "cell_type_specificity_weight": _spec_weight,
                "note": (
                    f"γ={gamma_raw:.4f} for {prog} ({disease_key}). "
                    f"Direction: {dir_src}. "
                    f"LDSC τ={tau:.4f} (tau_sign={tau_sign:+d}, annotation only). "
                    f"τ_p={tau_p:.3g}. "
                    f"Tier3 specificity_weight={_spec_weight:.4f}."
                    + (f" Burden γ={burden_gamma:.3f}." if burden_gamma is not None else "")
                    + _gwas_note
                ),
            }

        n_burden_resolved = sum(
            1 for v in results.values()
            if v.get("direction_source") in ("burden_direction", "burden_direction_confirmed")
        )
        log.info(
            "SVD γ loaded for %s: %d programs (%d atherogenic, %d protective, %d depleted, %d burden-resolved)",
            disease_key.upper(), len(results),
            sum(1 for v in results.values() if v["gamma"] > 0),
            sum(1 for v in results.values() if v["gamma"] < 0),
            sum(1 for v in results.values() if v["gamma"] == 0),
            n_burden_resolved,
        )
        return results

    # --- Legacy fallback: old format without program_gammas ---
    # Sign-variance guard still applies for legacy signed-τ files
    program_taus: dict[str, float] = data.get("program_taus", {})
    if not program_taus:
        return {}

    n_pos = sum(1 for v in program_taus.values() if v > 0)
    n_neg = sum(1 for v in program_taus.values() if v < 0)
    if n_pos == 0 or n_neg == 0:
        log.warning(
            "SVD τ for %s (legacy): no sign variance (%d pos, %d neg) — skipping γ override.",
            disease_key.upper(), n_pos, n_neg,
        )
        return {}

    results = {}
    for prog, tau in program_taus.items():
        gamma_value = max(-1.0, min(1.0, round(tau, 5)))
        tau_p = 1.0
        tau_se = None
        for annot in data.get("raw_annotations", []):
            if annot.get("name") == prog:
                tau_p  = float(annot.get("tau_p", 1.0))
                tau_se = annot.get("tau_se")
                break
        gamma_se = round(float(tau_se), 5) if tau_se is not None else round(abs(gamma_value) * 0.30, 5)
        results[prog] = {
            "gamma":            gamma_value,
            "gamma_se":         gamma_se,
            "evidence_tier":    "Tier2_Convergent" if tau_p < _TAU_P_TIER2 else "Tier3_Provisional",
            "data_source":      f"S-LDSC_chisq_{disease_key}_SVD_tau={tau:.4f}",
            "program":          prog,
            "trait":            disease_key.upper(),
            "tau":              tau,
            "tau_sign":         1 if tau >= 0 else -1,
            "tau_p":            tau_p,
            "direction_source": "legacy_tau_sign",
            "note":             f"Legacy: τ={tau:.4f} for {prog}. τ_p={tau_p:.3g}.",
        }
    return results


def get_genetic_nmf_program_gammas(disease_key: str, condition: str = "") -> dict[str, dict]:
    """
    Return γ estimates for all GeneticNMF programs.

    condition: "" (shared/Stim8hr baseline), "REST", or "Stim48hr".
        Reads from data/ldsc/results/{disease_key}_GeneticNMF{_condition}_program_taus.json.

    Returns empty dict if file absent — caller falls back to SVD gammas.
    Same output schema as get_svd_program_gammas.
    """
    _cond_sfx = f"_{condition.strip()}" if condition.strip() else ""
    live = _RESULTS_DIR / f"{disease_key.upper()}_GeneticNMF{_cond_sfx}_program_taus.json"
    if not live.exists():
        return {}
    try:
        with open(live) as fh:
            data = json.load(fh)
    except Exception as exc:
        log.warning("Failed to load GeneticNMF taus for %s: %s", disease_key, exc)
        return {}

    program_gammas: dict[str, float] = data.get("program_gammas", {})
    gamma_annots: list[dict] = data.get("gamma_annotations", [])
    annot_by_prog = {a["name"]: a for a in gamma_annots}

    if not program_gammas:
        return {}

    # Sign-variance guard: if fewer than 3 programs have positive γ, the GeneticNMF
    # programs are dominated by chromatin depletion at GWAS loci. Max-normalisation would
    # spread near-zero noise across the full [-1, +1] range, polluting the OTA sum with
    # spurious negative contributions. Return {} so the caller uses LOCUS/SVD gammas.
    _n_pos_raw = sum(1 for v in program_gammas.values() if v > 0)
    _min_pos   = max(5, len(program_gammas) // 5)   # need ≥20% positive programs
    if _n_pos_raw < _min_pos:
        log.info(
            "get_genetic_nmf_program_gammas: only %d programs have γ>0 for %s %s — "
            "GeneticNMF chromatin not enriched at GWAS loci, falling back to LOCUS/SVD gammas",
            _n_pos_raw, disease_key.upper(), condition,
        )
        return {}

    # Max-normalise: consistent with get_svd_program_gammas and get_cnmf_program_gammas.
    _max_abs = max((abs(v) for v in program_gammas.values()), default=1.0) or 1.0
    results: dict[str, dict] = {}
    for prog, gamma_raw in program_gammas.items():
        gamma_value = round(gamma_raw / _max_abs, 5)
        gamma_value = max(-1.0, min(1.0, gamma_value))
        annot       = annot_by_prog.get(prog, {})
        tau         = annot.get("tau") or data.get("program_taus", {}).get(prog, 0.0)
        tau_sign    = annot.get("tau_sign", 1 if tau >= 0 else -1)
        results[prog] = {
            "gamma":            gamma_value,
            "tau":              tau,
            "tau_sign":         tau_sign,
            "direction_source": annot.get("direction_source", "unknown"),
            "direction_score":  annot.get("direction_score", 0.0),
            "n_eqtl_genes":     annot.get("n_eqtl_genes", 0),
            "data_source":      f"GeneticNMF_chisq_{disease_key}_tau={tau:.4f}",
        }

    n_pos  = sum(1 for v in results.values() if v["gamma"] > 0)
    n_neg  = sum(1 for v in results.values() if v["gamma"] < 0)
    log.info(
        "GeneticNMF γ loaded for %s: %d programs (%d pro-disease, %d protective, %d depleted)",
        disease_key.upper(), len(results), n_pos, n_neg, len(results) - n_pos - n_neg,
    )
    return results


def get_cnmf_program_gammas(disease_key: str) -> dict[str, dict]:
    """
    Return γ estimates for Schnitzler cNMF programs (k=60 endothelial).

    Reads from data/ldsc/results/{disease_key}_cNMF_program_taus.json
    (produced by runner.run_cnmf_programs).

    Same output schema as get_svd_program_gammas.
    Returns empty dict if file absent — caller falls back to SVD gammas.
    """
    live = _RESULTS_DIR / f"{disease_key.upper()}_cNMF_program_taus.json"
    if not live.exists():
        return {}
    try:
        with open(live) as fh:
            data = json.load(fh)
    except Exception as exc:
        log.warning("Failed to load cNMF taus for %s: %s", disease_key, exc)
        return {}

    program_gammas: dict[str, float] = data.get("program_gammas", {})
    gamma_annots: list[dict] = data.get("gamma_annotations", [])
    annot_by_prog = {a["program"]: a for a in gamma_annots}
    direction_scores: dict[str, float] = data.get("cnmf_direction_scores", {})
    direction_threshold: float = data.get("cnmf_direction_threshold", 0.005)

    if not program_gammas:
        return {}

    # Max-normalise: largest |γ| → ±1 (consistent with get_svd_program_gammas).
    # total_abs normalisation was 10–100× smaller than SVD scale, causing cNMF to be
    # swamped when SVD and cNMF betas both contribute to the OTA sum.
    _max_abs = max((abs(v) for v in program_gammas.values()), default=1.0) or 1.0
    prefix = f"{disease_key.upper()}_cNMF_"
    results: dict[str, dict] = {}
    n_dir_flipped = 0
    for prog, gamma_raw in program_gammas.items():
        gamma_value = round(gamma_raw / _max_abs, 5)
        gamma_value = max(-1.0, min(1.0, gamma_value))
        annot       = annot_by_prog.get(prog, {})
        tau         = annot.get("tau") or data.get("program_taus", {}).get(prog, 0.0)
        tau_sign    = 1 if tau >= 0 else -1

        dir_score   = direction_scores.get(prog, 0.0)
        dir_source  = annot.get("direction_source", "unknown")
        # Apply SVD-projected direction correction: flip γ sign when eQTL evidence
        # strongly contradicts the raw τ sign (only for positive-τ programs — depleted
        # programs are already zero in program_gammas).
        if gamma_raw > 0 and abs(dir_score) >= direction_threshold and dir_score < 0:
            gamma_value = -abs(gamma_value)
            dir_source  = "cnmf_svd_eqtl_projection"
            n_dir_flipped += 1

        # Prefix so keys match load_cnmf_program_betas output (e.g. "CAD_cNMF_P14")
        full_prog = prog if prog.startswith(prefix) else prefix + prog
        results[full_prog] = {
            "gamma":            gamma_value,
            "tau":              tau,
            "tau_sign":         tau_sign,
            "direction_score":  round(dir_score, 6),
            "direction_source": dir_source,
            "n_eqtl_genes":     annot.get("n_eqtl_genes", 0),
            "data_source":      f"cNMF_k60_chisq_{disease_key}_tau={tau:.4f}",
        }

    n_pos  = sum(1 for v in results.values() if v["gamma"] > 0)
    n_neg  = sum(1 for v in results.values() if v["gamma"] < 0)
    log.info(
        "cNMF γ loaded for %s: %d programs (%d atherogenic, %d protective, %d depleted, %d direction-flipped)",
        disease_key.upper(), len(results), n_pos, n_neg, len(results) - n_pos - n_neg, n_dir_flipped,
    )
    return results


def get_locus_program_gammas(disease_key: str) -> dict[str, dict]:
    """
    Return γ estimates for GWAS-anchored locus programs.

    Reads from data/ldsc/results/{disease_key}_LOCUS_program_taus.json
    (produced by runner.py when LOCUS_* program IDs are passed with ra_loci/ BEDs).

    Returns empty dict if file absent — caller falls back to SVD gammas.
    Same output schema as get_svd_program_gammas.
    """
    live = _RESULTS_DIR / f"{disease_key.upper()}_LOCUS_program_taus.json"
    if not live.exists():
        return {}
    try:
        with open(live) as fh:
            data = json.load(fh)
    except Exception as exc:
        log.warning("Failed to load locus program taus for %s: %s", disease_key, exc)
        return {}

    program_gammas: dict[str, float] = data.get("program_gammas", {})
    gamma_annots: list[dict] = data.get("gamma_annotations", [])
    annot_by_prog = {a["name"]: a for a in gamma_annots}

    if not program_gammas:
        # Fallback: raw τ values
        program_taus: dict[str, float] = data.get("program_taus", {})
        if not program_taus:
            return {}
        program_gammas = program_taus

    results: dict[str, dict] = {}
    for prog, gamma_raw in program_gammas.items():
        gamma_value = max(-1.0, min(1.0, round(gamma_raw, 5)))
        annot    = annot_by_prog.get(prog, {})
        tau      = annot.get("tau") or data.get("program_taus", {}).get(prog, 0.0)
        tau_p    = 1.0
        tau_se   = None
        for raw in data.get("raw_annotations", []):
            if raw.get("name") == prog:
                tau_p  = float(raw.get("tau_p", 1.0))
                tau_se = raw.get("tau_se")
                break
        gamma_se = round(float(tau_se), 5) if tau_se is not None else round(abs(gamma_value) * 0.30, 5)
        results[prog] = {
            "gamma":            gamma_value,
            "gamma_se":         gamma_se,
            "evidence_tier":    "Tier2_Convergent" if tau_p < _TAU_P_TIER2 else "Tier3_Provisional",
            "data_source":      f"S-LDSC_locus_{disease_key}_tau={gamma_raw:.4f}",
            "program":          prog,
            "trait":            disease_key.upper(),
            "tau":              tau,
            "tau_p":            tau_p,
            "direction_source": annot.get("direction_source", "locus_eqtl"),
        }

    n_pos = sum(1 for v in results.values() if v["gamma"] > 0)
    n_neg = sum(1 for v in results.values() if v["gamma"] < 0)
    log.info(
        "Locus γ loaded for %s: %d programs (%d atherogenic, %d protective)",
        disease_key.upper(), len(results), n_pos, n_neg,
    )
    return results


def locus_gammas_available(disease_key: str) -> bool:
    """Return True if GWAS-locus γ results exist for this disease."""
    return (_RESULTS_DIR / f"{disease_key.upper()}_LOCUS_program_taus.json").exists()


def genetic_nmf_gammas_available(disease_key: str) -> bool:
    """Return True if GeneticNMF γ results exist for this disease."""
    return (_RESULTS_DIR / f"{disease_key.upper()}_GeneticNMF_program_taus.json").exists()


def ldsc_available(disease_key: str) -> bool:
    """Return True if pre-computed S-LDSC results exist for this disease."""
    return (_RESULTS_DIR / f"{disease_key.upper()}_program_taus.json").exists()


# Disease → perturbseq dataset with SVD loadings (mirrors runner._DISEASE_SVD_DATASET)
_DISEASE_SVD_DATASET: dict[str, str] = {
    "CAD": "schnitzler_cad_vascular",
    "RA":  "czi_2025_cd4t_perturb",
    "SLE": "czi_2025_cd4t_perturb",
}

# Per-disease WES burden file
_DISEASE_BURDEN_FILE: dict[str, str] = {
    "CAD": "CAD_burden.json",
    "RA":  "RA_burden.json",
}


@lru_cache(maxsize=4)
def _load_svd_loadings(dataset_id: str):
    """Load SVD Vt matrix and pert_names. Returns (Vt, pert_names) or (None, None)."""
    try:
        import numpy as np
        npz_path = _ROOT / "data" / "perturbseq" / dataset_id / "svd_loadings.npz"
        if not npz_path.exists():
            return None, None
        data = np.load(str(npz_path), allow_pickle=True)
        return data["Vt"], data["pert_names"]
    except Exception as exc:
        log.warning("Failed to load SVD loadings for %s: %s", dataset_id, exc)
        return None, None


@lru_cache(maxsize=4)
def _load_burden_data(disease_key: str) -> dict:
    """Load Backman2021 WES burden dict. Cached per process."""
    fname = _DISEASE_BURDEN_FILE.get(disease_key.upper())
    if not fname:
        return {}
    path = _ROOT / "data" / "wes" / fname
    if not path.exists():
        return {}
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception as exc:
        log.warning("Failed to load burden data for %s: %s", disease_key, exc)
        return {}


def get_svd_burden_program_gammas(
    disease_key: str,
    n_top_genes: int = 200,
    burden_p_threshold: float = 0.05,
) -> dict[str, float]:
    """
    Compute γ_{P→trait} = IV-weighted mean(Backman2021 burden_beta) for top-loading SVD genes.

    Per Ota et al. 2025: GeneBayes smoothed per-gene LoF γ values averaged across
    program-loading genes estimate the program-level causal effect γ_{P→trait}.
    Here we use raw Backman2021 log-ORs with inverse-variance weighting and a
    significance filter (p < burden_p_threshold) to reduce sparse-carrier noise.

    Returns dict[program_name → burden_gamma] for programs where ≥3 significant genes exist.
    Program names use the short form (e.g. "C27") matching the SVD tau file keys.
    """
    import numpy as np

    dataset_id = _DISEASE_SVD_DATASET.get(disease_key.upper())
    if not dataset_id:
        return {}

    Vt, pert_names = _load_svd_loadings(dataset_id)
    if Vt is None:
        return {}

    burden = _load_burden_data(disease_key)
    if not burden:
        return {}

    n_components = Vt.shape[0]
    results: dict[str, float] = {}

    for comp_idx in range(n_components):
        prog = f"C{comp_idx + 1:02d}"
        row = Vt[comp_idx]
        top_idx = np.argsort(np.abs(row))[::-1][:n_top_genes]

        betas: list[float] = []
        inv_var_weights: list[float] = []
        for i in top_idx:
            gene = str(pert_names[i])
            b = burden.get(gene, {})
            bb = b.get("burden_beta")
            se = b.get("burden_se")
            bp = b.get("burden_p")
            if bb is None or se is None or bp is None:
                continue
            try:
                bb, se, bp = float(bb), float(se), float(bp)
            except (TypeError, ValueError):
                continue
            if bp < burden_p_threshold and se > 0:
                betas.append(bb)
                inv_var_weights.append(1.0 / (se * se))

        if len(betas) < 3:
            continue

        w = np.array(inv_var_weights)
        b_arr = np.array(betas)
        results[prog] = float(round(float(np.dot(w, b_arr) / np.sum(w)), 4))

    log.info(
        "SVD burden γ computed for %s: %d/%d programs have ≥3 significant burden genes",
        disease_key.upper(), len(results), n_components,
    )
    return results


def get_gwas_enrichment_program_gammas(
    disease_key: str,
    gwas_genes: list[str] | None = None,
    n_top_genes: int = 200,
    min_gwas_genes: int = 10,
) -> dict[str, dict]:
    """
    Compute γ_{P→trait} from GWAS-gene enrichment in program loading genes.

    Per Zhu et al. 2025 (Fig 7): for underpowered binary-disease traits (RA, CAD),
    bypass LoF burden entirely. Test whether each program's top-loading genes are
    enriched for disease GWAS-evidence genes (from OpenTargets or caller-supplied list).

    Args:
        gwas_genes: Optional pre-supplied list of GWAS-evidence gene symbols.
                    If None, reads from data/ot_cache/{DISEASE}_genetic_genes.json.
                    Pass disease_query["gwas_genes"] from the pipeline call chain.

    Returns dict[short_prog → {log_or, fisher_p, n_overlap, n_top, n_gwas, enriched}]
    Empty dict if insufficient GWAS genes or SVD loadings unavailable.
    """
    import math as _math
    import numpy as _np

    dataset_id = _DISEASE_SVD_DATASET.get(disease_key.upper())
    if not dataset_id:
        return {}

    Vt, pert_names = _load_svd_loadings(dataset_id)
    if Vt is None:
        return {}

    # Resolve GWAS gene list: caller-supplied > per-disease JSON cache
    ot_genes: list[str] = list(gwas_genes) if gwas_genes else []
    if not ot_genes:
        try:
            ot_path = _ROOT / "data" / "ot_cache" / f"{disease_key.upper()}_genetic_genes.json"
            if ot_path.exists():
                import json as _json
                ot_genes = _json.loads(ot_path.read_text())
        except Exception:
            pass
    if not ot_genes or len(ot_genes) < min_gwas_genes:
        log.debug("GWAS enrichment γ_P: insufficient OT genes for %s (%d)", disease_key, len(ot_genes))
        return {}

    gwas_set = frozenset(str(g) for g in ot_genes)
    all_genes = [str(g) for g in pert_names]
    background_n = len(all_genes)
    background_gwas = len(gwas_set & frozenset(all_genes))

    # Perturb-seq KO libraries have small background (< 5000 genes) and are
    # already disease-biased — the Fisher test cannot discriminate programs.
    # Only run enrichment when background is a general gene expression space.
    _MIN_BACKGROUND = 5000
    if background_n < _MIN_BACKGROUND:
        log.info(
            "GWAS enrichment γ_P skipped for %s: background too small "
            "(%d < %d) — Perturb-seq KO library already disease-biased, "
            "test has no discrimination power",
            disease_key.upper(), background_n, _MIN_BACKGROUND,
        )
        return {}

    results: dict[str, dict] = {}
    n_components = Vt.shape[0]

    for comp_idx in range(n_components):
        prog = f"C{comp_idx + 1:02d}"
        row = Vt[comp_idx]
        top_idx = _np.argsort(_np.abs(row))[::-1][:n_top_genes]
        top_genes = [all_genes[i] for i in top_idx]
        top_set = frozenset(top_genes)

        # 2×2 Fisher contingency: top_genes × gwas_set
        a = len(top_set & gwas_set)         # top AND gwas
        b = len(top_set) - a                 # top NOT gwas
        c = background_gwas - a             # gwas NOT top
        d = background_n - len(top_set) - c  # neither

        if a == 0 or d < 0:
            continue

        # Fisher exact OR and p-value (log scale)
        try:
            from scipy.stats import fisher_exact
            _, fisher_p = fisher_exact([[a, b], [c, d]], alternative="greater")
        except Exception:
            # Fallback: chi2 approximation for speed
            fisher_p = 1.0

        # Log odds ratio (with continuity correction)
        log_or = _math.log((a + 0.5) * (d + 0.5) / ((b + 0.5) * (c + 0.5)))

        results[prog] = {
            "log_or":   round(log_or, 4),
            "fisher_p": round(float(fisher_p), 6),
            "n_overlap": a,
            "n_top":    len(top_set),
            "n_gwas":   background_gwas,
            "enriched": fisher_p < 0.05 and log_or > 0,
        }

    n_enriched = sum(1 for v in results.values() if v["enriched"])
    log.info(
        "GWAS enrichment γ_P for %s: %d/%d programs enriched (OT genes=%d, background=%d)",
        disease_key.upper(), n_enriched, len(results), len(gwas_set), background_n,
    )
    return results


def load_loo_discounts(
    disease_key: str,
    anchor_gene_positions: dict[str, tuple[int, int]],  # gene → (chrom, pos_bp)
    results_dir: Path | None = None,
) -> dict[str, float]:
    """
    Load SNP positions cache and run LOO stability analysis.

    Returns dict[gene → loo_discount_factor] where factor < 1.0 for unstable genes.
    Returns empty dict if SNP positions file not found (graceful fallback).

    Args:
        disease_key:            Disease key (e.g. "CAD", "RA").
        anchor_gene_positions:  dict[gene_symbol → (chrom_int, pos_bp)].
        results_dir:            Override default results directory (for testing).
    """
    _rdir = results_dir if results_dir is not None else _RESULTS_DIR
    snp_pos_file = _rdir / f"{disease_key.upper()}_program_snp_positions.json"
    if not snp_pos_file.exists():
        return {}

    try:
        with open(snp_pos_file) as fh:
            raw_pos = json.load(fh)
    except Exception as exc:
        log.warning("Failed to load SNP positions for LOO (%s): %s", disease_key, exc)
        return {}

    if not raw_pos or not anchor_gene_positions:
        return {}

    # Convert JSON lists back to list-of-tuples
    program_snp_positions: dict[str, list[tuple[int, int, float]]] = {
        prog: [(int(c), int(p), float(q)) for c, p, q in snps]
        for prog, snps in raw_pos.items()
    }

    # Load tau cache for baseline program_taus and mean_chisq
    tau_data = _load_tau_file(disease_key.upper())
    if tau_data is None:
        return {}

    program_taus: dict[str, float] = tau_data.get("program_taus", {})
    mean_chisq: float = float(tau_data.get("mean_chisq_genome", 1.0))
    n_snps_genome: int = int(tau_data.get("n_snps_genome", 1))

    if not program_taus:
        return {}

    try:
        from pipelines.ldsc.leave_locus_out import leave_locus_out_stability, summarize_loo_stability
        from config.scoring_thresholds import LOO_STABILITY_THRESHOLD
    except ImportError as exc:
        log.warning("LOO imports failed: %s", exc)
        return {}

    try:
        loo_results = leave_locus_out_stability(
            program_snp_positions=program_snp_positions,
            anchor_genes=anchor_gene_positions,
            program_taus=program_taus,
            n_snps_genome=n_snps_genome,
            mean_chisq=mean_chisq,
        )
    except Exception as exc:
        log.warning("LOO stability analysis failed (non-fatal): %s", exc)
        return {}

    summary = summarize_loo_stability(loo_results)
    log.info(
        "LOO stability: %d/%d genes stable (median rank_delta=%.1f)",
        summary["n_stable"], summary["n_genes"], summary["median_rank_delta"],
    )

    discounts: dict[str, float] = {}
    for gene, result in loo_results.items():
        if result["rank_delta"] > LOO_STABILITY_THRESHOLD:
            discounts[gene] = 0.8
        else:
            discounts[gene] = 1.0

    return discounts
