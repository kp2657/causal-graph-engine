"""
Tests for program-level β/γ extraction utilities in
`steps/tier3_causal/causal_filters.py` and program driver classification
in `steps/tier4_translation/target_ranker.py`.
"""
from __future__ import annotations

import math

import pytest

from steps.tier3_causal.ota_gamma_calculator import (
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


