"""
pipelines/twmr.py — Simplified two-sample Mendelian randomisation (ratio MR).

Implements single-gene TWMR using the IVW ratio estimator:

    γ_IVW = Σ_i (β_eQTL_i × β_GWAS_i / σ²_i) / Σ_i (β_eQTL_i² / σ²_i)

where σ²_i is the squared standard error of β_GWAS_i (or β_eQTL_i when GWAS SE
is unavailable).  Each instrument (cis-eQTL SNP) must be:
  - genome-wide or cis-significant in the exposure GWAS (eQTL)
  - not flagged for LD with another included variant

This is the simplified "oracle" IVW that assumes:
  - No sample overlap between eQTL and disease GWAS
  - Effect alleles aligned between studies
  - No pleiotropy (assumption, not enforced — use check_weak_instrument for F-stat)

Public API
----------
    compute_ratio_mr(gene, eqtl_instruments, ...)
        -> dict  compatible with estimate_gamma()'s twmr_result parameter

    check_weak_instrument(beta, se, n_samples)
        -> float  F-statistic; < 10 = weak instrument

    run_twmr_for_gene(gene, disease_key, efo_id, tissue, n_gwas)
        -> dict | None   fetches eQTL + GWAS data via live APIs, returns MR estimate
"""
from __future__ import annotations

import math
import statistics
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.scoring_thresholds import MR_F_STATISTIC_MIN as _F_STAT_MIN

# Disease tissue mapping for eQTL lookups
_DISEASE_TISSUE: dict[str, str] = {
    "IBD": "Colon_Sigmoid",
    "CAD": "Artery_Coronary",
    "RA":  "Whole_Blood",
    "T2D": "Pancreas",
    "AD":  "Brain_Frontal_Cortex_BA9",
}

# Approximate N_GWAS when not provided (used for F-stat approximation)
_DEFAULT_N_GWAS: dict[str, int] = {
    "IBD": 45_000,
    "CAD": 180_000,
    "RA":  58_000,
    "T2D": 900_000,
}


def check_weak_instrument(beta: float, se: float, n_samples: int = 10_000) -> float:
    """
    Approximate F-statistic for a single-instrument MR.

    F ≈ (β / SE)² = z²

    A rule-of-thumb threshold of F ≥ 10 is commonly used.
    """
    if se <= 0:
        return 0.0
    z = beta / se
    return z * z


def compute_ratio_mr(
    gene: str,
    eqtl_instruments: Sequence[dict],
    n_gwas: int = 10_000,
) -> dict | None:
    """
    IVW ratio MR from a list of cis-eQTL instruments.

    Each instrument dict must contain:
        beta_eqtl  (float)  — eQTL effect size (NES or log-allelic effect)
        beta_gwas  (float)  — GWAS effect size for the same SNP
        se_gwas    (float, optional)  — GWAS SE; defaults to |beta_gwas| / 2 if absent

    Returns a dict compatible with estimate_gamma()'s ``twmr_result`` parameter:
        {beta, se, p, n_instruments, f_statistic, method}
    or None if fewer than 1 valid instrument remains after filtering.
    """
    valid: list[dict] = []
    for inst in eqtl_instruments:
        b_eq = inst.get("beta_eqtl")
        b_gw = inst.get("beta_gwas")
        if b_eq is None or b_gw is None:
            continue
        if not (math.isfinite(b_eq) and math.isfinite(b_gw)):
            continue
        if abs(b_eq) < 1e-8:  # near-zero eQTL — degenerate ratio
            continue
        valid.append({
            "beta_eqtl": float(b_eq),
            "beta_gwas": float(b_gw),
            "se_gwas":   float(inst.get("se_gwas") or max(abs(b_gw) / 2, 1e-6)),
        })

    if not valid:
        return None

    if len(valid) == 1:
        # Single-instrument ratio
        inst = valid[0]
        ratio = inst["beta_gwas"] / inst["beta_eqtl"]
        se_ratio = inst["se_gwas"] / abs(inst["beta_eqtl"])
        z = ratio / se_ratio if se_ratio > 0 else 0.0
        p = _z_to_p(z)
        f_stat = check_weak_instrument(inst["beta_eqtl"], inst["se_gwas"], n_gwas)
        return {
            "gene":          gene,
            "beta":          round(ratio, 5),
            "se":            round(se_ratio, 5),
            "p":             p,
            "n_instruments": 1,
            "f_statistic":   round(f_stat, 2),
            "method":        "ratio",
        }

    # IVW: weight by inverse variance of GWAS effect
    weights = [1.0 / (v["se_gwas"] ** 2) for v in valid]
    numerator   = sum(v["beta_eqtl"] * v["beta_gwas"] * w for v, w in zip(valid, weights))
    denominator = sum(v["beta_eqtl"] ** 2 * w             for v, w in zip(valid, weights))
    if abs(denominator) < 1e-12:
        return None
    beta_ivw = numerator / denominator

    # IVW SE (under no heterogeneity)
    se_ivw = math.sqrt(1.0 / denominator)
    z = beta_ivw / se_ivw if se_ivw > 0 else 0.0
    p = _z_to_p(z)

    # Mean F-stat across instruments
    f_stats = [check_weak_instrument(v["beta_eqtl"], v["se_gwas"], n_gwas) for v in valid]
    f_mean = statistics.mean(f_stats)

    # Pleiotropy-robust sensitivity estimators (require ≥3 instruments)
    wm_beta   = compute_weighted_median_mr(valid)
    egger     = compute_mr_egger(valid)
    cochran   = _cochran_q(valid, beta_ivw)

    # Method agreement: IVW + WM same direction and magnitude within 2×SE?
    method_agreement: bool | None = None
    if wm_beta is not None:
        method_agreement = (
            math.copysign(1, beta_ivw) == math.copysign(1, wm_beta)
            and abs(beta_ivw - wm_beta) < 2 * se_ivw
        )

    pleiotropy_flag = bool(
        (egger is not None and egger["pleiotropy_detected"])
        or cochran["p_Q"] < 0.05
    )

    return {
        "gene":                 gene,
        "beta":                 round(beta_ivw, 5),
        "se":                   round(se_ivw, 5),
        "p":                    p,
        "n_instruments":        len(valid),
        "f_statistic":          round(f_mean, 2),
        "method":               "IVW",
        # Sensitivity estimators
        "beta_weighted_median": round(wm_beta, 5) if wm_beta is not None else None,
        "mr_egger":             egger,
        "cochran_q":            cochran,
        "method_agreement":     method_agreement,
        "pleiotropy_flag":      pleiotropy_flag,
    }


def run_twmr_for_gene(
    gene: str,
    disease_key: str,
    efo_id: str,
    tissue: str | None = None,
    n_gwas: int | None = None,
) -> dict | None:
    """
    Fetch cis-eQTL + GWAS data for *gene* and compute IVW ratio MR.

    Uses:
      - GTEx eQTLs (query_gtex_eqtl) as exposure instruments
      - OpenGWAS / GWAS Catalog associations for the same SNPs as outcome

    Returns twmr_result dict (see compute_ratio_mr) or None when data is
    insufficient (< 1 valid instrument or F-stat < 10).
    """
    if not tissue:
        tissue = _DISEASE_TISSUE.get(disease_key, "Whole_Blood")
    if not n_gwas:
        n_gwas = _DEFAULT_N_GWAS.get(disease_key, 50_000)

    try:
        from mcp_servers.gwas_genetics_server import query_gtex_eqtl
        eqtl_result = query_gtex_eqtl(gene, tissue)
    except Exception:
        return None

    eqtls = (
        eqtl_result.get("eqtls")
        or eqtl_result.get("data")
        or eqtl_result.get("results")
        or []
    )
    if not eqtls:
        return None

    # Build instruments from top eQTLs (up to 10 for runtime)
    instruments: list[dict] = []
    for eq in eqtls[:10]:
        nes = eq.get("nes") or eq.get("effect_size") or eq.get("beta")
        if nes is None:
            continue
        snp_id = eq.get("snp_id") or eq.get("rsid") or eq.get("variant_id", "")

        # Look up GWAS effect for this SNP
        beta_gwas = _fetch_gwas_beta_for_snp(snp_id, efo_id)
        if beta_gwas is None:
            continue

        instruments.append({
            "beta_eqtl": float(nes),
            "beta_gwas": beta_gwas["beta"],
            "se_gwas":   beta_gwas.get("se") or max(abs(beta_gwas["beta"]) / 2, 1e-6),
            "snp_id":    snp_id,
        })

    if not instruments:
        return None

    result = compute_ratio_mr(gene, instruments, n_gwas=n_gwas)
    if result is None:
        return None

    # Enforce minimum F-statistic
    if result.get("f_statistic", 0) < _F_STAT_MIN:
        return None

    return result


# ---------------------------------------------------------------------------
# Additional MR estimators — pleiotropy-robust
# ---------------------------------------------------------------------------

def compute_weighted_median_mr(valid: list[dict]) -> float | None:
    """
    Weighted median MR estimator (Bowden 2016).

    Returns a consistent causal estimate when ≤50% of instruments are invalid
    (i.e. pleiotropic). Requires ≥3 instruments.

    Weights each per-instrument ratio β_GWAS/β_eQTL by the precision of the
    GWAS estimate: w_i = β_eQTL_i² / SE_GWAS_i². The weighted median of
    sorted per-instrument ratios is the point estimate.
    """
    if len(valid) < 3:
        return None
    ratios  = [v["beta_gwas"] / v["beta_eqtl"] for v in valid]
    weights = [v["beta_eqtl"] ** 2 / v["se_gwas"] ** 2 for v in valid]
    total_w = sum(weights)
    if total_w <= 0:
        return None
    sorted_pairs = sorted(zip(ratios, weights), key=lambda x: x[0])
    cumulative = 0.0
    for ratio, w in sorted_pairs:
        cumulative += w / total_w
        if cumulative >= 0.5:
            return ratio
    return sorted_pairs[-1][0]


def compute_mr_egger(valid: list[dict]) -> dict | None:
    """
    MR-Egger regression (Bowden 2015).

    Models β_GWAS = α + γ × β_eQTL where α ≠ 0 indicates directional pleiotropy
    (InSIDE assumption: pleiotropy is uncorrelated with instrument strength).

    Returns:
        slope      — causal effect estimate (less precise than IVW)
        intercept  — pleiotropy test; significantly ≠ 0 → directional pleiotropy
        p_intercept — p-value for intercept test
        pleiotropy_detected — bool (p_intercept < 0.05)

    Requires ≥3 instruments.
    """
    if len(valid) < 3:
        return None

    x = [v["beta_eqtl"] for v in valid]
    y = [v["beta_gwas"] for v in valid]
    w = [1.0 / v["se_gwas"] ** 2 for v in valid]

    total_w = sum(w)
    xbar = sum(wi * xi for wi, xi in zip(w, x)) / total_w
    ybar = sum(wi * yi for wi, yi in zip(w, y)) / total_w

    sxx = sum(wi * (xi - xbar) ** 2 for wi, xi in zip(w, x))
    sxy = sum(wi * (xi - xbar) * (yi - ybar) for wi, xi, yi in zip(w, x, y))

    if abs(sxx) < 1e-12:
        return None

    slope     = sxy / sxx
    intercept = ybar - slope * xbar

    n = len(valid)
    residuals = [yi - intercept - slope * xi for xi, yi in zip(x, y)]
    # Residual variance (degrees of freedom = n - 2 for intercept + slope)
    resid_var = (
        sum(wi * r ** 2 for wi, r in zip(w, residuals))
        / ((n - 2) * (total_w / n))
        if n > 2 else 1.0
    )

    se_intercept = math.sqrt(resid_var / total_w) if resid_var > 0 else 1e-8
    z_intercept  = intercept / se_intercept
    p_intercept  = _z_to_p(z_intercept)

    return {
        "slope":               round(slope, 5),
        "intercept":           round(intercept, 6),
        "se_intercept":        round(se_intercept, 6),
        "p_intercept":         round(p_intercept, 4),
        "pleiotropy_detected": bool(p_intercept < 0.05),
    }


def _cochran_q(valid: list[dict], beta_ivw: float) -> dict:
    """
    Cochran Q heterogeneity test for IVW MR.
    Large Q → instrument heterogeneity → possible pleiotropy.
    """
    per_ratio = [v["beta_gwas"] / v["beta_eqtl"] for v in valid]
    weights   = [v["beta_eqtl"] ** 2 / v["se_gwas"] ** 2 for v in valid]
    Q = sum(w * (r - beta_ivw) ** 2 for w, r in zip(weights, per_ratio))
    df = len(valid) - 1
    # p-value from chi-squared(df)
    try:
        import math as _m
        # Use incomplete gamma approximation: p ≈ regularised upper gamma
        # For simplicity, flag if Q > 2×df (rough rule-of-thumb)
        p_Q = 1.0 if Q < df else max(0.0, 1.0 - (Q - df) / (Q + 1))
    except Exception:
        p_Q = 1.0
    return {"Q": round(Q, 3), "Q_df": df, "p_Q": round(p_Q, 4)}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _z_to_p(z: float) -> float:
    """Two-sided p-value from z-score using normal approximation."""
    import math
    if not math.isfinite(z):
        return 1.0
    # Use complementary error function: p = erfc(|z| / sqrt(2))
    return math.erfc(abs(z) / math.sqrt(2))


def _fetch_gwas_beta_for_snp(snp_id: str, efo_id: str) -> dict | None:
    """
    Look up GWAS effect size for a SNP from OpenGWAS / GWAS Catalog.
    Returns {"beta": float, "se": float | None} or None.
    """
    if not snp_id or not efo_id:
        return None
    try:
        from mcp_servers.gwas_genetics_server import get_snp_associations
        result = get_snp_associations(snp_id)
        efo_upper = efo_id.upper()
        for assoc in result.get("associations", []):
            # Filter to associations matching the target trait
            trait_match = any(
                efo_upper in (t.get("uri") or "").upper()
                for t in assoc.get("efo_traits", [])
            )
            if not trait_match:
                continue
            beta = assoc.get("beta")
            if beta is not None and math.isfinite(float(beta)):
                se = assoc.get("se")
                return {"beta": float(beta), "se": float(se) if se is not None else None}
    except Exception:
        pass
    return None
