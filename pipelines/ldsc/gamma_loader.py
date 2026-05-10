"""
pipelines/ldsc/gamma_loader.py — Load pre-computed S-LDSC γ values.

γ(P→trait) = τ* (signed S-LDSC enrichment coefficient).
Positive τ*: program chromatin is enriched for GWAS heritability → included in OTA sum.
Negative τ*: program chromatin is depleted of heritability → excluded (not a signal carrier).

Programs with τ* ≤ 0 are filtered out before returning; only heritability-enriched programs
contribute to OTA γ = Σ_P β(gene→P) × τ*(P→trait).

Evidence tier: "Tier2_Convergent" when τ_p < 0.05, else "Tier3_Provisional".
"""
from __future__ import annotations

import json
import logging
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


def _gnmf_spec_weights(disease_key: str, condition: str, prog_names: list[str]) -> dict[str, float]:
    """
    Compute cell-type specificity weight for each GeneticNMF program.

    Loads genetic_nmf_loadings{_condition}.npz, takes top-50 genes per component,
    and scores against the disease cell-type marker set.
    Returns dict[prog_name → weight ∈ [0.15, 1.0]]. Empty dict on any failure.
    """
    dk = disease_key.upper()
    dataset = _DATASET_FOR_DISEASE.get(dk)
    if not dataset:
        return {}

    _cond_sfx = f"_{condition.lower()}" if condition.strip() else ""
    npz_candidates = [
        _ROOT / "data" / "perturbseq" / dataset / f"genetic_nmf_loadings{_cond_sfx}.npz",
        _ROOT / "data" / "perturbseq" / dataset / "genetic_nmf_loadings.npz",
    ]
    npz_path = next((p for p in npz_candidates if p.exists()), None)
    if npz_path is None:
        return {}

    try:
        import numpy as _np
        d = _np.load(str(npz_path), allow_pickle=True)
        Vt = d["Vt"]          # (k × n_genes)
        gene_names = [str(g) for g in d.get("pert_names", d.get("gene_names", []))]
    except Exception as exc:
        log.debug("gnmf_spec_weights: failed to load %s: %s", npz_path.name, exc)
        return {}

    result: dict[str, float] = {}
    for prog in prog_names:
        # Parse component index from name suffix (e.g. "RA_GeneticNMF_Stim48hr_C03" → 2)
        try:
            suffix = prog.split("_")[-1]  # "C03"
            comp_idx = int(suffix[1:]) - 1
        except (ValueError, IndexError):
            continue
        if comp_idx < 0 or comp_idx >= Vt.shape[0]:
            continue
        row = Vt[comp_idx]
        top_idx = _np.argsort(_np.abs(row))[::-1][:50]
        top_genes = [gene_names[i] for i in top_idx if i < len(gene_names)]
        result[prog] = _cell_type_specificity_weight(dk, top_genes)

    return result


def get_genetic_nmf_program_gammas(
    disease_key: str,
    condition: str = "",
    prefer_raw_taus: bool = False,
) -> dict[str, dict]:
    """
    Return γ estimates for all GeneticNMF programs.

    γ = τ* (signed), max-normalised by largest positive τ*.
    Programs with τ* ≤ 0 (heritability-depleted) are excluded — they do not carry GWAS signal.
    Each entry also carries `spec_weight` (cell-type specificity, [0.15, 1.0])
    for the bystander-filtered parallel track.

    condition: "" (shared baseline), "REST", or "Stim48hr".
    prefer_raw_taus: if True, use raw τ* (program_taus key) for γ magnitude.
        Use for diseases where eQTL direction correction is unreliable (e.g. RA — weak
        CD4+ T eQTL concordance flips valid programs negative). For CAD, eQTL direction
        correction is validated (CCM2/PLPP3 sign confirmed) so prefer_raw_taus=False.
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

    if prefer_raw_taus:
        # Raw τ* — direction from β signs; avoids excluding programs flipped negative by
        # unreliable eQTL direction correction (RA C01/C12/C04 pattern).
        program_gammas: dict[str, float] = (
            data.get("program_taus") or data.get("program_gammas") or {}
        )
    else:
        # eQTL-direction-corrected γ — validated for CAD (CCM2/PLPP3 sign correct).
        program_gammas = (
            data.get("program_gammas") or data.get("program_taus") or {}
        )
    gamma_annots: list[dict] = data.get("gamma_annotations", [])
    annot_by_prog = {a["name"]: a for a in gamma_annots}

    if not program_gammas:
        return {}

    # Max-normalise by largest positive τ*; skip heritability-depleted programs (τ*≤0).
    _max_pos = max((v for v in program_gammas.values() if v > 0), default=None)
    if _max_pos is None:
        log.warning(
            "GeneticNMF γ for %s %s: all τ* ≤ 0 — no heritability-enriched programs",
            disease_key.upper(), condition or "combined",
        )
        return {}
    results: dict[str, dict] = {}
    for prog, gamma_raw in program_gammas.items():
        if gamma_raw <= 0:
            continue  # heritability-depleted — skip
        gamma_value = round(gamma_raw / _max_pos, 5)
        annot = annot_by_prog.get(prog, {})
        tau   = annot.get("tau") or data.get("program_taus", {}).get(prog, 0.0)
        results[prog] = {
            "gamma":       gamma_value,
            "tau":         tau,
            "data_source": f"GeneticNMF_chisq_{disease_key}_tau={tau:.4f}",
            "spec_weight": 1.0,  # filled in below
        }

    # Compute per-program cell-type specificity weights (bystander track).
    spec_weights = _gnmf_spec_weights(disease_key, condition, list(results.keys()))
    for prog, sw in spec_weights.items():
        if prog in results:
            results[prog]["spec_weight"] = sw

    n_specific = sum(1 for v in results.values() if v["spec_weight"] >= 0.5)
    log.info(
        "GeneticNMF γ loaded for %s %s: %d programs with τ*>0 (depleted excluded), %d cell-type-specific (spec≥0.5)",
        disease_key.upper(), condition or "combined", len(results), n_specific,
    )
    return results


def get_cnmf_program_gammas(disease_key: str) -> dict[str, dict]:
    """
    Return γ estimates for Schnitzler cNMF programs (k=60 endothelial).

    γ = τ* (signed), max-normalised by the largest positive τ*.
    Programs with τ* ≤ 0 (heritability-depleted) are excluded from the OTA sum.
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

    if not program_gammas:
        return {}

    # Only programs with positive τ* carry GWAS heritability; normalise by largest positive τ*.
    _max_pos = max((v for v in program_gammas.values() if v > 0), default=None)
    if _max_pos is None:
        log.warning("cNMF γ for %s: all τ* ≤ 0 — no heritability-enriched programs", disease_key.upper())
        return {}
    prefix = f"{disease_key.upper()}_cNMF_"
    results: dict[str, dict] = {}
    for prog, gamma_raw in program_gammas.items():
        if gamma_raw <= 0:
            continue  # heritability-depleted program — skip
        gamma_value = round(gamma_raw / _max_pos, 5)
        annot = annot_by_prog.get(prog, {})
        tau   = annot.get("tau") or data.get("program_taus", {}).get(prog, 0.0)
        full_prog = prog if prog.startswith(prefix) else prefix + prog
        results[full_prog] = {
            "gamma":       gamma_value,
            "tau":         tau,
            "data_source": f"cNMF_k60_chisq_{disease_key}_tau={tau:.4f}",
        }

    log.info(
        "cNMF γ loaded for %s: %d programs with τ*>0 (signed, depleted programs excluded)",
        disease_key.upper(), len(results),
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
