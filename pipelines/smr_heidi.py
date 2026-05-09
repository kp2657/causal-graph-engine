"""
smr_heidi.py — Summary-data Mendelian Randomization (SMR) + HEIDI heterogeneity test.

SMR detects pleiotropic instruments: when the top eQTL SNP drives both gene
expression and disease risk through SEPARATE pathways rather than mediation
(gene expression → disease).

Key formulas:
    β_SMR  = β_GWAS / β_eQTL          (ratio estimator)
    SE_SMR = |β_SMR| × sqrt(se_eQTL²/β_eQTL² + se_GWAS²/β_GWAS²)

HEIDI proxy (no LD data):
    When full LD panel is unavailable, approximate HEIDI using COLOC posteriors.
    High H3 (linkage broken — two distinct causal variants) relative to H4
    (shared causal variant) signals heterogeneity / pleiotropy.
    heidi_frac = H3 / (H3 + H4 + 1e-8)
    heidi_flag = heidi_frac > threshold (default 0.3)

Reference:
    Zhu et al. (2016) Nature Genetics 48:959–964.
    Wu et al. (2023) Nature Methods (HEIDI extension).
"""
from __future__ import annotations

import math


# ---------------------------------------------------------------------------
# SMR ratio estimator
# ---------------------------------------------------------------------------

def compute_smr_beta(
    beta_gwas: float,
    se_gwas: float,
    beta_eqtl: float,
    se_eqtl: float,
) -> dict:
    """
    Compute SMR ratio estimate.

    β_SMR  = β_GWAS / β_eQTL
    SE_SMR = |β_SMR| × sqrt(se_eQTL²/β_eQTL² + se_GWAS²/β_GWAS²)

    Uses normal approximation for p-value:
        p_SMR = 2 × (1 - Φ(|z|))
        where Φ(x) is approximated via erfc(x/√2)/2

    Returns:
        dict with keys: beta_smr, se_smr, z_smr, p_smr
        All values are None if inputs are invalid (zero denominators, non-finite).
    """
    _null = {"beta_smr": None, "se_smr": None, "z_smr": None, "p_smr": None}

    # Validate inputs
    try:
        beta_gwas = float(beta_gwas)
        se_gwas   = float(se_gwas)
        beta_eqtl = float(beta_eqtl)
        se_eqtl   = float(se_eqtl)
    except (TypeError, ValueError):
        return _null

    for v in (beta_gwas, se_gwas, beta_eqtl, se_eqtl):
        if not math.isfinite(v):
            return _null

    # Guard: zero eQTL beta (division by zero) or zero GWAS beta (delta method breaks)
    if beta_eqtl == 0.0 or beta_gwas == 0.0:
        return _null

    # Guard: non-positive SEs
    if se_gwas <= 0.0 or se_eqtl <= 0.0:
        return _null

    beta_smr = beta_gwas / beta_eqtl

    # Delta-method SE for ratio estimator
    rel_var = (se_eqtl / beta_eqtl) ** 2 + (se_gwas / beta_gwas) ** 2
    se_smr  = abs(beta_smr) * math.sqrt(rel_var)

    if se_smr == 0.0:
        return _null

    z_smr = beta_smr / se_smr

    # p-value via normal approximation: p = erfc(|z|/√2)
    p_smr = math.erfc(abs(z_smr) / math.sqrt(2))

    return {
        "beta_smr": beta_smr,
        "se_smr":   se_smr,
        "z_smr":    z_smr,
        "p_smr":    p_smr,
    }


# ---------------------------------------------------------------------------
# HEIDI proxy from COLOC posteriors
# ---------------------------------------------------------------------------

def heidi_proxy_from_coloc(
    h3: float,
    h4: float,
    threshold: float = 0.3,
) -> dict:
    """
    Approximate HEIDI test from COLOC posteriors.

    H3 = probability that eQTL and GWAS signals are driven by two DISTINCT
         causal variants (linkage but not shared variant) — heterogeneity.
    H4 = probability of a SINGLE shared causal variant — mediation (desired).

    heidi_frac = H3 / (H3 + H4 + 1e-8)
    heidi_flag = heidi_frac > threshold

    Interpretation:
        heidi_flag=True  → pleiotropic instrument — eQTL and GWAS driven by
                           different variants; SMR ratio estimate is unreliable.
        heidi_flag=False → instrument passes HEIDI; proceed with SMR β.

    Args:
        h3:        COLOC posterior H3 (two distinct causal variants)
        h4:        COLOC posterior H4 (shared causal variant)
        threshold: heidi_frac cutoff for flagging pleiotropy (default 0.3)

    Returns:
        dict with keys: heidi_flag, heidi_frac, h3, h4
    """
    try:
        h3 = float(h3)
        h4 = float(h4)
    except (TypeError, ValueError):
        return {"heidi_flag": True, "heidi_frac": float("nan"), "h3": h3, "h4": h4}

    if not (math.isfinite(h3) and math.isfinite(h4)):
        return {"heidi_flag": True, "heidi_frac": float("nan"), "h3": h3, "h4": h4}

    # Clamp to [0, 1] to handle numerical noise from COLOC
    h3 = max(0.0, min(1.0, h3))
    h4 = max(0.0, min(1.0, h4))

    heidi_frac = h3 / (h3 + h4 + 1e-8)
    heidi_flag = heidi_frac > threshold

    return {
        "heidi_flag": bool(heidi_flag),
        "heidi_frac": heidi_frac,
        "h3":         h3,
        "h4":         h4,
    }


# ---------------------------------------------------------------------------
# SMR + HEIDI combined filter
# ---------------------------------------------------------------------------

def smr_heidi_filter(
    beta_smr: float | None,
    se_smr: float | None,
    heidi_flag: bool,
    min_smr_z: float = 3.0,
) -> dict:
    """
    Decide whether to keep the eQTL→trait link based on SMR + HEIDI.

    Decision logic:
        Keep if: |z_smr| >= min_smr_z  AND  NOT heidi_flag

    Cases:
        1. heidi_flag=True              → reject (pleiotropic instrument)
        2. beta_smr/se_smr are None     → reject if heidi_flag, else keep with caveat
        3. |z_smr| < min_smr_z          → reject (SMR signal too weak)
        4. |z_smr| >= min_smr_z + pass  → keep

    Args:
        beta_smr:   SMR effect estimate (None if unavailable)
        se_smr:     SMR standard error (None if unavailable)
        heidi_flag: True if HEIDI test flags pleiotropy
        min_smr_z:  Minimum |z| to consider SMR signal significant (default 3.0)

    Returns:
        dict with keys: keep (bool), reason (str), beta_smr, se_smr
    """
    # HEIDI rejection takes priority — pleiotropic instrument is invalid regardless
    # of how strong the raw SMR association appears.
    if heidi_flag:
        return {
            "keep":     False,
            "reason":   "HEIDI heterogeneity test flagged pleiotropic instrument (H3/(H3+H4) > threshold)",
            "beta_smr": beta_smr,
            "se_smr":   se_smr,
        }

    # If SMR stats are unavailable (e.g. no GWAS beta in eqtl_data), HEIDI passed
    # but we cannot compute a z-score — keep with a caveat note.
    if beta_smr is None or se_smr is None:
        return {
            "keep":     True,
            "reason":   "HEIDI passed; SMR z not computable (missing GWAS beta/se) — kept",
            "beta_smr": beta_smr,
            "se_smr":   se_smr,
        }

    try:
        z = float(beta_smr) / float(se_smr)
    except (TypeError, ValueError, ZeroDivisionError):
        return {
            "keep":     False,
            "reason":   "SMR z-score computation failed (invalid beta_smr or se_smr)",
            "beta_smr": beta_smr,
            "se_smr":   se_smr,
        }

    if abs(z) < min_smr_z:
        return {
            "keep":     False,
            "reason":   f"SMR |z|={abs(z):.2f} < min_smr_z={min_smr_z:.1f} — weak instrument",
            "beta_smr": beta_smr,
            "se_smr":   se_smr,
        }

    return {
        "keep":     True,
        "reason":   f"SMR |z|={abs(z):.2f} >= {min_smr_z:.1f} and HEIDI passed",
        "beta_smr": beta_smr,
        "se_smr":   se_smr,
    }
