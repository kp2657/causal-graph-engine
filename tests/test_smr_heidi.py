"""
tests/test_smr_heidi.py — Unit tests for SMR + HEIDI pleiotropic instrument filter.

Tests cover:
  1. Basic SMR ratio estimator arithmetic
  2. Zero eQTL beta guard (no division by zero)
  3. Strong z-score produces expected keep decision
  4. HEIDI flag when H3 is high relative to H4
  5. HEIDI flag off when H3 is low
  6. smr_heidi_filter: keep when strong z + no HEIDI flag
  7. smr_heidi_filter: reject when HEIDI flag is True
  8. estimate_beta_tier2 integration: pleiotropic instrument → returns None
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipelines.smr_heidi import (
    compute_smr_beta,
    heidi_proxy_from_coloc,
    smr_heidi_filter,
)
from pipelines.ota_beta_estimation import estimate_beta_tier2


# ---------------------------------------------------------------------------
# Test 1: Basic SMR β arithmetic
# ---------------------------------------------------------------------------

def test_smr_beta_basic():
    """β_GWAS=0.1, β_eQTL=0.5 → β_SMR=0.2 (exact ratio)."""
    result = compute_smr_beta(
        beta_gwas=0.1,
        se_gwas=0.02,
        beta_eqtl=0.5,
        se_eqtl=0.05,
    )
    assert result["beta_smr"] is not None
    assert abs(result["beta_smr"] - 0.2) < 1e-10, (
        f"Expected β_SMR=0.2, got {result['beta_smr']}"
    )
    assert result["se_smr"] is not None and result["se_smr"] > 0
    assert result["z_smr"] is not None
    assert result["p_smr"] is not None and 0.0 <= result["p_smr"] <= 1.0


# ---------------------------------------------------------------------------
# Test 2: Zero β_eQTL → all-None (guard against division by zero)
# ---------------------------------------------------------------------------

def test_smr_beta_zero_eqtl():
    """β_eQTL=0 must return all-None dict — no ZeroDivisionError."""
    result = compute_smr_beta(
        beta_gwas=0.1,
        se_gwas=0.02,
        beta_eqtl=0.0,
        se_eqtl=0.05,
    )
    assert result["beta_smr"] is None
    assert result["se_smr"] is None
    assert result["z_smr"] is None
    assert result["p_smr"] is None


# ---------------------------------------------------------------------------
# Test 3: Strong z → expect keep=True from smr_heidi_filter
# ---------------------------------------------------------------------------

def test_smr_z_strong():
    """|z_smr| > 3 should pass the SMR filter when HEIDI is not flagged."""
    # β_SMR = 1.0 / 2.0 = 0.5; SE_SMR ≈ |0.5| × sqrt((0.1/2)² + (0.05/1)²)
    result = compute_smr_beta(
        beta_gwas=1.0,
        se_gwas=0.05,
        beta_eqtl=2.0,
        se_eqtl=0.01,
    )
    assert result["z_smr"] is not None
    assert abs(result["z_smr"]) > 3.0, (
        f"Expected |z|>3, got {result['z_smr']}"
    )
    filt = smr_heidi_filter(result["beta_smr"], result["se_smr"], heidi_flag=False)
    assert filt["keep"] is True


# ---------------------------------------------------------------------------
# Test 4: High H3 → heidi_flag=True
# ---------------------------------------------------------------------------

def test_heidi_flag_high_h3():
    """H3=0.4, H4=0.4 → heidi_frac=0.5 > 0.3 threshold → heidi_flag=True."""
    result = heidi_proxy_from_coloc(h3=0.4, h4=0.4)
    assert result["heidi_flag"] is True
    assert result["heidi_frac"] > 0.3
    assert result["h3"] == pytest.approx(0.4)
    assert result["h4"] == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# Test 5: Low H3 → heidi_flag=False
# ---------------------------------------------------------------------------

def test_heidi_flag_low_h3():
    """H3=0.01, H4=0.8 → heidi_frac ≈ 0.012 < 0.3 → heidi_flag=False."""
    result = heidi_proxy_from_coloc(h3=0.01, h4=0.8)
    assert result["heidi_flag"] is False
    assert result["heidi_frac"] < 0.3
    assert result["h3"] == pytest.approx(0.01)
    assert result["h4"] == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# Test 6: smr_heidi_filter — keep when strong z + no HEIDI flag
# ---------------------------------------------------------------------------

def test_smr_heidi_filter_keep():
    """Strong z (|z|=5) and heidi_flag=False → keep=True."""
    # beta_smr=1.0, se_smr=0.2 → z=5.0
    result = smr_heidi_filter(beta_smr=1.0, se_smr=0.2, heidi_flag=False, min_smr_z=3.0)
    assert result["keep"] is True
    assert result["beta_smr"] == pytest.approx(1.0)
    assert result["se_smr"] == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# Test 7: smr_heidi_filter — reject when heidi_flag=True
# ---------------------------------------------------------------------------

def test_smr_heidi_filter_reject_heidi():
    """heidi_flag=True → keep=False, reason must mention 'HEIDI'."""
    result = smr_heidi_filter(beta_smr=1.0, se_smr=0.2, heidi_flag=True, min_smr_z=3.0)
    assert result["keep"] is False
    assert "HEIDI" in result["reason"], (
        f"Expected 'HEIDI' in reason, got: {result['reason']}"
    )


# ---------------------------------------------------------------------------
# Test 8: estimate_beta_tier2 rejects pleiotropic instrument
# ---------------------------------------------------------------------------

def test_estimate_beta_tier2_rejects_pleiotropic():
    """
    When H3=0.4 and H4=0.4, estimate_beta_tier2 should return None because
    the HEIDI proxy flags this as a pleiotropic instrument.

    eqtl_data provides both beta_gwas/se_gwas (triggering full SMR path) and
    nes/se/tissue for the eQTL instrument.  coloc_h4=0.85 passes the COLOC gate,
    but H3/(H3+H4)=0.5 > 0.3 triggers HEIDI rejection.
    """
    eqtl_data = {
        "nes":        0.5,
        "se":         0.05,
        "pval_nominal": 1e-10,
        "tissue":     "Whole_Blood",
        "beta_gwas":  0.1,
        "se_gwas":    0.02,
    }
    result = estimate_beta_tier2(
        gene="TESTGENE",
        program="HALLMARK_INFLAMMATORY_RESPONSE",
        eqtl_data=eqtl_data,
        coloc_h4=0.85,   # passes COLOC gate
        coloc_h3=0.4,    # H3/(H3+H4) = 0.5 > 0.3 → HEIDI flag
        program_loading=0.8,
    )
    assert result is None, (
        f"Expected None (pleiotropic instrument rejected), got: {result}"
    )
