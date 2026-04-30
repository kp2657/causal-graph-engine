"""
Tests for the OT-genetic fallback `top_programs` reconstruction in
`steps/tier3_causal/ota_gamma_calculator.py`.

Regression target: AMD Run 2 produced `program_flag: no_program_data` for
every top target because the fallback explicitly set `top_programs = []`.
With Tier 2 `gene_program_overlap` piped through, the fallback now
reconstructs a `{program: contribution}` dict using the best-available β/γ
combination per program.
"""
from __future__ import annotations

import math

import pytest

from steps.tier3_causal.ota_gamma_calculator import (
    _build_fallback_top_programs,
    _extract_beta_for_program,
    _extract_gamma_for_trait,
)
from steps.tier4_translation.target_ranker import (
    _classify_program_drivers,
)


# ---------------------------------------------------------------------------
# _extract_beta_for_program / _extract_gamma_for_trait shape handling
# ---------------------------------------------------------------------------

def test_extract_beta_handles_scalar_and_dict_forms():
    bm = {"CFH": {"P1": 0.42, "P2": {"beta": -0.3}, "P3": None, "P4": float("nan")}}
    assert _extract_beta_for_program(bm, "CFH", "P1") == pytest.approx(0.42)
    assert _extract_beta_for_program(bm, "CFH", "P2") == pytest.approx(-0.3)
    assert _extract_beta_for_program(bm, "CFH", "P3") is None
    assert _extract_beta_for_program(bm, "CFH", "P4") is None
    assert _extract_beta_for_program(bm, "UNKNOWN", "P1") is None


def test_extract_gamma_handles_all_shapes():
    assert _extract_gamma_for_trait(0.4, "AMD") == pytest.approx(0.4)
    assert _extract_gamma_for_trait({"AMD": 0.5, "CAD": 0.1}, "AMD") == pytest.approx(0.5)
    assert _extract_gamma_for_trait({"gamma": 0.33}, "AMD") == pytest.approx(0.33)
    assert _extract_gamma_for_trait({"AMD": {"gamma": 0.7}}, "AMD") == pytest.approx(0.7)
    assert _extract_gamma_for_trait(None, "AMD") is None
    assert _extract_gamma_for_trait({"CAD": 0.1}, "AMD") is None


# ---------------------------------------------------------------------------
# _build_fallback_top_programs — three branches
# ---------------------------------------------------------------------------

def test_fallback_branch1_beta_and_gamma_known():
    """β × γ used directly; sum scales to ota_gamma."""
    overlap = {"CFH": ["P_inflam", "P_comp"]}
    betas = {"CFH": {"P_inflam": 0.6, "P_comp": 0.4}}
    gammas = {"P_inflam": {"AMD": 0.5}, "P_comp": {"AMD": 0.2}}
    out = _build_fallback_top_programs(
        gene="CFH", trait="AMD", ota_gamma=0.15,
        gene_program_overlap=overlap, gamma_estimates=gammas, beta_matrix=betas,
    )
    assert isinstance(out, dict) and len(out) == 2
    assert sum(out.values()) == pytest.approx(0.15, rel=1e-6)
    assert abs(out["P_inflam"]) > abs(out["P_comp"])   # β=0.6×γ=0.5 > β=0.4×γ=0.2


def test_fallback_branch2_only_gamma_known():
    overlap = {"CFH": ["P1", "P2"]}
    gammas = {"P1": {"AMD": 0.8}, "P2": {"AMD": 0.2}}
    out = _build_fallback_top_programs(
        gene="CFH", trait="AMD", ota_gamma=0.1,
        gene_program_overlap=overlap, gamma_estimates=gammas, beta_matrix={},
    )
    assert set(out.keys()) == {"P1", "P2"}
    # Proportional to γ: 0.8/1.0 * 0.1 = 0.08; 0.2/1.0 * 0.1 = 0.02
    assert out["P1"] == pytest.approx(0.08)
    assert out["P2"] == pytest.approx(0.02)


def test_fallback_branch3_neither_known_equal_split():
    overlap = {"LIPC": ["P_lipid", "P_metab", "P_inflam"]}
    out = _build_fallback_top_programs(
        gene="LIPC", trait="AMD", ota_gamma=0.09,
        gene_program_overlap=overlap, gamma_estimates={}, beta_matrix={},
    )
    assert set(out.keys()) == {"P_lipid", "P_metab", "P_inflam"}
    for v in out.values():
        assert v == pytest.approx(0.03)


def test_fallback_preserves_negative_sign():
    overlap = {"X": ["P1", "P2"]}
    gammas = {"P1": {"AMD": 0.5}}  # only P1 has γ
    out = _build_fallback_top_programs(
        gene="X", trait="AMD", ota_gamma=-0.08,
        gene_program_overlap=overlap, gamma_estimates=gammas, beta_matrix={},
    )
    # Branch 2 fires: only γ known for P1. |P1| gets full |ota_gamma| with sign.
    assert out["P1"] == pytest.approx(-0.08)


def test_fallback_empty_when_no_overlap():
    assert _build_fallback_top_programs(
        gene="ORPHAN", trait="AMD", ota_gamma=0.2,
        gene_program_overlap={}, gamma_estimates={}, beta_matrix={},
    ) == {}


def test_fallback_empty_when_ota_gamma_is_zero_or_nan():
    overlap = {"G": ["P1"]}
    assert _build_fallback_top_programs(
        gene="G", trait="AMD", ota_gamma=0.0,
        gene_program_overlap=overlap, gamma_estimates={}, beta_matrix={},
    ) == {}
    assert _build_fallback_top_programs(
        gene="G", trait="AMD", ota_gamma=float("nan"),
        gene_program_overlap=overlap, gamma_estimates={}, beta_matrix={},
    ) == {}


# ---------------------------------------------------------------------------
# Downstream integration: classifier no longer emits no_program_data sentinel
# ---------------------------------------------------------------------------

def test_classifier_escapes_no_program_data_with_fallback_dict():
    """Regression: AMD top targets like CFH used to land here with `[]` and
    the classifier returned `no_program_data`.  After the fix the fallback
    emits a non-empty dict, so program_flag must be non-sentinel."""
    overlap = {"CFH": ["COMPLEMENT", "INFLAMMATION"]}
    gammas = {"COMPLEMENT": {"AMD": 0.4}, "INFLAMMATION": {"AMD": 0.1}}
    tp = _build_fallback_top_programs(
        gene="CFH", trait="AMD", ota_gamma=0.12,
        gene_program_overlap=overlap, gamma_estimates=gammas, beta_matrix={},
    )
    drivers = _classify_program_drivers(tp, ota_gamma=0.12, disease_key="AMD")
    assert drivers["program_flag"] != "no_program_data"
    assert drivers["top_program"] in {"COMPLEMENT", "INFLAMMATION"}
