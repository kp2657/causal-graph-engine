"""
tests/test_finngen_validation.py — Unit tests for FinnGen R10 holdout validation.

All tests use synthetic data — no real network downloads required.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipelines.ldsc.finngen_validation import (
    FINNGEN_R10_ENDPOINTS,
    get_finngen_url,
    spearman_rank_correlation,
    run_finngen_holdout_validation,
)


class TestFinngenUrl:
    def test_get_finngen_url_cad(self):
        """URL for CAD should reference I9_CHD endpoint."""
        url = get_finngen_url("CAD")
        assert "I9_CHD" in url
        assert url.startswith("https://")
        assert url.endswith(".gz")

    def test_get_finngen_url_ra(self):
        """URL for RA should reference M13_RHEUMA endpoint."""
        url = get_finngen_url("RA")
        assert "M13_RHEUMA" in url
        assert url.startswith("https://")
        assert url.endswith(".gz")


class TestSpearmanCorrelation:
    def test_spearman_perfect_correlation(self):
        """Identical dicts should yield rho = 1.0."""
        gamma = {"PROG_A": 0.5, "PROG_B": 0.3, "PROG_C": 0.8, "PROG_D": 0.1}
        result = spearman_rank_correlation(gamma, gamma)
        assert result["n"] == 4
        assert abs(result["rho"] - 1.0) < 1e-6

    def test_spearman_inverse_correlation(self):
        """Perfectly reversed ranking should yield rho close to -1.0."""
        gamma_a = {"A": 1.0, "B": 0.8, "C": 0.6, "D": 0.4, "E": 0.2}
        gamma_b = {"A": 0.2, "B": 0.4, "C": 0.6, "D": 0.8, "E": 1.0}
        result = spearman_rank_correlation(gamma_a, gamma_b)
        assert result["n"] == 5
        assert result["rho"] <= -0.9, f"Expected rho near -1.0, got {result['rho']}"

    def test_spearman_empty_overlap(self):
        """No shared keys → n=0, graceful return with NaN rho."""
        import math
        gamma_a = {"X": 0.5, "Y": 0.3}
        gamma_b = {"A": 0.9, "B": 0.1}
        result = spearman_rank_correlation(gamma_a, gamma_b)
        assert result["n"] == 0
        assert math.isnan(result["rho"])
        assert math.isnan(result["p_approx"])


class TestRunFinngenHoldout:
    def test_run_finngen_holdout_missing_file(self, tmp_path):
        """If the sumstats file doesn't exist, return skipped=True."""
        nonexistent = tmp_path / "finngen_R10_I9_CHD.gz"
        result = run_finngen_holdout_validation(
            disease_key="CAD",
            platlas_gammas={"PROG_A": 0.8, "PROG_B": 0.4},
            program_snp_sets={"PROG_A": {"rs123"}, "PROG_B": {"rs456"}},
            sumstats_path=nonexistent,
        )
        assert result.get("skipped") is True
        assert "reason" in result
        assert result["disease_key"] == "CAD"
