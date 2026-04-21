"""
tests/test_phase_s_counterfactual.py — Phase S: OTA counterfactual simulation.

Tests:
  - simulate_perturbation output schema
  - Knockout (delta=-1.0) reduces γ to 0
  - Partial inhibition reduces γ proportionally
  - Over-expression increases γ
  - No perturbation returns identical gammas
  - delta_gamma = perturbed - baseline (arithmetic identity)
  - percent_change computation
  - dominant_program detection
  - program_contributions_perturbed vs baseline lengths match
  - None entry in beta_estimates handled gracefully
  - Interpretation strings for different perturbation types
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipelines.counterfactual import simulate_perturbation


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _simple_betas() -> dict:
    """Two programs with known beta values."""
    return {
        "PROG_A": {"beta": 0.4, "evidence_tier": "Tier1_Interventional"},
        "PROG_B": {"beta": 0.2, "evidence_tier": "Tier2_Convergent"},
    }


def _simple_gammas() -> dict:
    """Matching gamma estimates per program."""
    return {
        "PROG_A": {"gamma": 0.5, "evidence_tier": "Tier1_Interventional"},
        "PROG_B": {"gamma": 0.3, "evidence_tier": "Tier2_Convergent"},
    }


def _baseline_gamma(betas, gammas) -> float:
    """Expected: sum(beta × gamma) = 0.4×0.5 + 0.2×0.3 = 0.20 + 0.06 = 0.26"""
    total = 0.0
    for prog, b_info in betas.items():
        b = b_info["beta"]
        g = gammas.get(prog, {}).get("gamma", 0.0)
        total += b * g
    return total


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

class TestSimulatePerturbationSchema:

    def test_required_keys(self):
        result = simulate_perturbation("NOD2", -0.5, _simple_betas(), _simple_gammas(), "IBD")
        required = {
            "gene", "trait", "delta_beta_fraction",
            "baseline_gamma", "perturbed_gamma", "delta_gamma", "percent_change",
            "interpretation", "dominant_program",
            "program_contributions_baseline", "program_contributions_perturbed",
        }
        assert required.issubset(result.keys())

    def test_gene_and_trait_passthrough(self):
        result = simulate_perturbation("NOD2", -0.5, _simple_betas(), _simple_gammas(), "IBD")
        assert result["gene"] == "NOD2"
        assert result["trait"] == "IBD"
        assert result["delta_beta_fraction"] == pytest.approx(-0.5)


# ---------------------------------------------------------------------------
# Arithmetic correctness
# ---------------------------------------------------------------------------

class TestKnockout:

    def test_knockout_zeros_gamma(self):
        # delta = -1.0 → scale_factor = 0.0 → all betas = 0 → γ = 0
        result = simulate_perturbation("NOD2", -1.0, _simple_betas(), _simple_gammas(), "IBD")
        assert result["perturbed_gamma"] == pytest.approx(0.0, abs=1e-9)

    def test_knockout_baseline_nonzero(self):
        result = simulate_perturbation("NOD2", -1.0, _simple_betas(), _simple_gammas(), "IBD")
        assert abs(result["baseline_gamma"]) > 1e-6

    def test_knockout_delta_gamma_equals_minus_baseline(self):
        result = simulate_perturbation("NOD2", -1.0, _simple_betas(), _simple_gammas(), "IBD")
        assert result["delta_gamma"] == pytest.approx(-result["baseline_gamma"], abs=1e-6)

    def test_knockout_percent_change_100(self):
        result = simulate_perturbation("NOD2", -1.0, _simple_betas(), _simple_gammas(), "IBD")
        assert result["percent_change"] == pytest.approx(-100.0, abs=0.01)


class TestPartialInhibition:

    def test_50pct_inhibition_halves_gamma(self):
        result = simulate_perturbation("NOD2", -0.5, _simple_betas(), _simple_gammas(), "IBD")
        expected = _baseline_gamma(_simple_betas(), _simple_gammas()) * 0.5
        assert result["perturbed_gamma"] == pytest.approx(expected, rel=1e-4)

    def test_50pct_inhibition_percent_change(self):
        result = simulate_perturbation("NOD2", -0.5, _simple_betas(), _simple_gammas(), "IBD")
        assert result["percent_change"] == pytest.approx(-50.0, abs=0.01)

    def test_delta_gamma_arithmetic(self):
        result = simulate_perturbation("NOD2", -0.5, _simple_betas(), _simple_gammas(), "IBD")
        expected_delta = result["perturbed_gamma"] - result["baseline_gamma"]
        assert result["delta_gamma"] == pytest.approx(expected_delta, abs=1e-9)


class TestOverExpression:

    def test_2x_overexpression_doubles_gamma(self):
        result = simulate_perturbation("NOD2", 1.0, _simple_betas(), _simple_gammas(), "IBD")
        expected = _baseline_gamma(_simple_betas(), _simple_gammas()) * 2.0
        assert result["perturbed_gamma"] == pytest.approx(expected, rel=1e-4)

    def test_overexpression_positive_delta(self):
        result = simulate_perturbation("NOD2", 1.0, _simple_betas(), _simple_gammas(), "IBD")
        assert result["delta_gamma"] > 0

    def test_overexpression_positive_percent_change(self):
        result = simulate_perturbation("NOD2", 1.0, _simple_betas(), _simple_gammas(), "IBD")
        assert result["percent_change"] == pytest.approx(100.0, abs=0.01)


class TestNoChange:

    def test_zero_delta_returns_identical_gammas(self):
        result = simulate_perturbation("NOD2", 0.0, _simple_betas(), _simple_gammas(), "IBD")
        assert result["baseline_gamma"] == pytest.approx(result["perturbed_gamma"], abs=1e-9)
        assert result["delta_gamma"] == pytest.approx(0.0, abs=1e-9)
        assert result["percent_change"] == pytest.approx(0.0, abs=0.01)


# ---------------------------------------------------------------------------
# Dominant program
# ---------------------------------------------------------------------------

class TestDominantProgram:

    def test_dominant_program_is_highest_contribution(self):
        # PROG_A: 0.4×0.5=0.20, PROG_B: 0.2×0.3=0.06 → PROG_A dominates
        result = simulate_perturbation("NOD2", 0.0, _simple_betas(), _simple_gammas(), "IBD")
        assert result["dominant_program"] == "PROG_A"

    def test_dominant_program_none_when_no_betas(self):
        result = simulate_perturbation("GENE", 0.0, {}, {}, "IBD")
        assert result["dominant_program"] is None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_none_beta_entry_skipped(self):
        betas = {
            "PROG_A": {"beta": 0.4, "evidence_tier": "Tier1_Interventional"},
            "PROG_NULL": None,
        }
        result = simulate_perturbation("NOD2", -0.5, betas, _simple_gammas(), "IBD")
        assert result["perturbed_gamma"] == pytest.approx(0.4 * 0.5 * 0.5, abs=1e-6)

    def test_scalar_beta_entry(self):
        betas = {"PROG_A": 0.4}  # scalar, not dict
        gammas = {"PROG_A": {"gamma": 0.5}}
        result = simulate_perturbation("NOD2", -1.0, betas, gammas, "IBD")
        assert result["perturbed_gamma"] == pytest.approx(0.0, abs=1e-9)

    def test_empty_betas_zero_gamma(self):
        result = simulate_perturbation("GENE", -0.5, {}, {}, "IBD")
        assert result["baseline_gamma"] == pytest.approx(0.0, abs=1e-9)
        assert result["perturbed_gamma"] == pytest.approx(0.0, abs=1e-9)

    def test_program_contributions_lengths_consistent(self):
        result = simulate_perturbation("NOD2", -0.5, _simple_betas(), _simple_gammas(), "IBD")
        # Both contribution lists should have same length (same programs)
        assert len(result["program_contributions_baseline"]) == len(
            result["program_contributions_perturbed"]
        )


# ---------------------------------------------------------------------------
# Interpretation strings
# ---------------------------------------------------------------------------

class TestInterpretation:

    def test_zero_delta_interpretation(self):
        result = simulate_perturbation("NOD2", 0.0, _simple_betas(), _simple_gammas(), "IBD")
        assert "No perturbation" in result["interpretation"]

    def test_knockout_interpretation(self):
        result = simulate_perturbation("NOD2", -1.0, _simple_betas(), _simple_gammas(), "IBD")
        interp = result["interpretation"]
        assert "knockout" in interp.lower() or "complete" in interp.lower()
        assert "NOD2" in interp

    def test_partial_inhibition_interpretation(self):
        result = simulate_perturbation("NOD2", -0.5, _simple_betas(), _simple_gammas(), "IBD")
        interp = result["interpretation"]
        assert "50%" in interp or "inhibition" in interp.lower()

    def test_activation_interpretation(self):
        result = simulate_perturbation("NOD2", 0.5, _simple_betas(), _simple_gammas(), "IBD")
        interp = result["interpretation"]
        assert "activation" in interp.lower() or "50%" in interp
