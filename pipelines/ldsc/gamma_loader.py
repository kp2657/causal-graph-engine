"""
pipelines/ldsc/gamma_loader.py — Load pre-computed S-LDSC τ coefficients as γ values.

Reads from data/ldsc/results/{disease_key}_program_taus.json (produced by runner.py).
Converts τ (per-SNP heritability enrichment) to a normalized γ ∈ [0, 1].

Normalization:
    γ_ldsc = τ_prog / (Σ|τ| across all programs + ε)   if τ > 0
    γ_ldsc = 0                                           if τ ≤ 0 (no enrichment)

This gives programs proportional γ based on their relative heritability contribution.
A program with 3× more heritability enrichment gets 3× higher γ_ldsc.

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


def get_program_gamma_ldsc(
    program: str,
    disease_key: str,
    trait: str | None = None,
) -> dict | None:
    """
    Return S-LDSC-derived γ for (program, disease) as a gamma estimate dict.

    Returns None when:
    - No pre-computed τ file exists (S-LDSC hasn't been run yet)
    - τ ≤ 0 for this program (no heritability enrichment)

    When available, returns a dict compatible with OTA γ estimation schema.
    """
    data = _load_tau_file(disease_key.upper())
    if not data:
        return None

    program_taus: dict[str, float] = data.get("program_taus", {})
    if not program_taus:
        return None

    tau = program_taus.get(program)
    if tau is None:
        return None

    # Normalize τ to γ ∈ [-1, 1] using total absolute enrichment as scale.
    # Negative τ (heritability-depleted) → negative γ (anti-disease program).
    total_abs = sum(abs(v) for v in program_taus.values()) or 1.0
    gamma_value = tau / (total_abs + 1e-8)
    gamma_value = max(-1.0, min(1.0, gamma_value))
    gamma_value = round(gamma_value, 5)

    # Find τ_p and τ_SE for this program from raw_annotations
    tau_p = 1.0
    tau_se: float | None = None
    for annot in data.get("raw_annotations", []):
        prog_slug = program.replace(" ", "_").replace("-", "_")
        if prog_slug in annot.get("name", ""):
            tau_p = annot.get("tau_p", 1.0)
            tau_se = annot.get("tau_se")
            break

    # Delta method: SE_γ = SE_τ / (Σ|τ| + ε)
    if tau_se is not None:
        gamma_se = round(float(tau_se) / (total_abs + 1e-8), 5)
    else:
        gamma_se = round(abs(gamma_value) * 0.30, 5)

    evidence_tier = "Tier2_Convergent" if tau_p < _TAU_P_TIER2 else "Tier3_Provisional"
    h2 = data.get("h2")

    return {
        "gamma":         gamma_value,
        "gamma_se":      gamma_se,
        "evidence_tier": evidence_tier,
        "data_source":   f"S-LDSC_PLAtlas_{disease_key}_tau={tau:.4f}",
        "program":       program,
        "trait":         trait or disease_key,
        "tau":           tau,
        "tau_p":         tau_p,
        "h2_total":      h2,
        "note": (
            f"S-LDSC heritability enrichment τ={tau:.4f} for {program} in {disease_key}. "
            f"PLAtlas MVP+UKB+FinnGen meta-analysis N≈820k. "
            f"τ_p={tau_p:.3g}."
        ),
    }


def get_all_program_gammas_ldsc(disease_key: str) -> dict[str, dict]:
    """
    Return all program γ estimates from S-LDSC for a disease.
    Dict[program_name -> gamma_estimate_dict].
    """
    data = _load_tau_file(disease_key.upper())
    if not data:
        return {}

    program_taus = data.get("program_taus", {})
    results = {}
    for prog, tau in program_taus.items():
        est = get_program_gamma_ldsc(prog, disease_key)
        if est is not None:
            results[prog] = est
    return results


def get_svd_program_gammas(disease_key: str) -> dict[str, dict]:
    """
    Return γ estimates for all SVD programs.

    Uses the eQTL-direction signed γ from program_gammas when available
    (written by run_svd_programs). Falls back to signed τ normalization for
    legacy files that predate the eQTL direction step.

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

    # --- New format: program_gammas computed by run_svd_programs ---
    program_gammas: dict[str, float] = data.get("program_gammas", {})
    gamma_annots: list[dict] = data.get("gamma_annotations", [])
    annot_by_prog = {a["name"]: a for a in gamma_annots}

    if program_gammas:
        total_abs = sum(abs(v) for v in program_gammas.values()) or 1.0
        results: dict[str, dict] = {}

        for prog, gamma_raw in program_gammas.items():
            # Normalize to [-1, 1]
            gamma_value = gamma_raw / (total_abs + 1e-8)
            gamma_value = max(-1.0, min(1.0, round(gamma_value, 5)))

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

            gamma_se = round(float(tau_se) / (total_abs + 1e-8), 5) if tau_se is not None else round(abs(gamma_value) * 0.30, 5)
            evidence_tier = "Tier2_Convergent" if tau_p < _TAU_P_TIER2 else "Tier3_Provisional"

            results[prog] = {
                "gamma":            gamma_value,
                "gamma_se":         gamma_se,
                "evidence_tier":    evidence_tier,
                "data_source":      f"S-LDSC_eQTL_{disease_key}_gamma={gamma_raw:.4f}",
                "program":          prog,
                "trait":            disease_key.upper(),
                "tau":              tau,
                "tau_sign":         tau_sign,     # +1 enriched, -1 depleted — annotation only
                "tau_p":            tau_p,
                "direction_source": dir_src,
                "note": (
                    f"γ={gamma_raw:.4f} for {prog} ({disease_key}). "
                    f"Direction: {dir_src}. "
                    f"LDSC τ={tau:.4f} (tau_sign={tau_sign:+d}, annotation only). "
                    f"τ_p={tau_p:.3g}."
                ),
            }

        log.info(
            "SVD γ loaded for %s: %d programs (%d atherogenic, %d protective, %d depleted)",
            disease_key.upper(), len(results),
            sum(1 for v in results.values() if v["gamma"] > 0),
            sum(1 for v in results.values() if v["gamma"] < 0),
            sum(1 for v in results.values() if v["gamma"] == 0),
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

    total_abs = sum(abs(v) for v in program_taus.values()) or 1.0
    results = {}
    for prog, tau in program_taus.items():
        gamma_value = tau / (total_abs + 1e-8)
        gamma_value = max(-1.0, min(1.0, round(gamma_value, 5)))
        tau_p = 1.0
        tau_se = None
        for annot in data.get("raw_annotations", []):
            if annot.get("name") == prog:
                tau_p  = float(annot.get("tau_p", 1.0))
                tau_se = annot.get("tau_se")
                break
        gamma_se = round(float(tau_se) / (total_abs + 1e-8), 5) if tau_se is not None else round(abs(gamma_value) * 0.30, 5)
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


def svd_ldsc_available(disease_key: str) -> bool:
    """Return True if pre-computed SVD S-LDSC results exist for this disease."""
    return (_RESULTS_DIR / f"{disease_key.upper()}_SVD_program_taus.json").exists()


def ldsc_available(disease_key: str) -> bool:
    """Return True if pre-computed S-LDSC results exist for this disease."""
    return (_RESULTS_DIR / f"{disease_key.upper()}_program_taus.json").exists()


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
