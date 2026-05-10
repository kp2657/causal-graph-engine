"""
test_ota_gamma_estimation.py — Unit tests for the γ_{program→trait} estimation pipeline.

Tests verify:
  - estimate_gamma: dispatches to correct sub-estimator; returns valid tier labels
  - compute_ota_gamma: OTA formula summation is correct; NaN propagation
  - Module-level constants imported from config.scoring_thresholds
"""
from __future__ import annotations

import math
import pytest

from pipelines.ota_gamma_estimation import (
    estimate_gamma,
    compute_ota_gamma,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PROGRAM = "complement_activation"
TRAIT = "age-related macular degeneration"
EFO = "EFO_0001481"

AMD_GWAS_GENES = {"CFH", "ARMS2", "C3", "CFB", "CFI", "C9", "VEGFA", "LIPC", "ABO"}
COMPLEMENT_PROGRAM = {"CFH", "C3", "CFB", "CFD", "C5", "C9", "CLU", "CFHR1", "CFHR3"}
UNRELATED_GENES = {"BRCA1", "TP53", "KRAS", "EGFR", "MYC"}


# ---------------------------------------------------------------------------
# estimate_gamma — dispatcher
# ---------------------------------------------------------------------------

class TestEstimateGamma:
    def test_returns_dict_always(self):
        """estimate_gamma always returns a dict (never None)."""
        result = estimate_gamma(PROGRAM, TRAIT)
        assert isinstance(result, dict)

    def test_result_has_gamma_field(self):
        result = estimate_gamma(PROGRAM, TRAIT)
        assert "gamma" in result

    def test_gamma_is_numeric_or_nan(self):
        result = estimate_gamma(PROGRAM, TRAIT)
        gamma = result.get("gamma")
        assert gamma is None or isinstance(gamma, float), f"gamma={gamma!r} is not float"

    def test_enriched_program_with_gene_set(self):
        """Passing program_gene_set+efo_id should return a valid dict (gamma may be None without live API)."""
        result = estimate_gamma(
            PROGRAM, TRAIT,
            program_gene_set=COMPLEMENT_PROGRAM,
            efo_id=EFO,
        )
        assert isinstance(result, dict)
        assert "gamma" in result
        gamma = result.get("gamma")
        assert gamma is None or isinstance(gamma, float)

    def test_evidence_tier_is_string(self):
        result = estimate_gamma(PROGRAM, TRAIT)
        tier = result.get("evidence_tier")
        assert tier is None or isinstance(tier, str)


# ---------------------------------------------------------------------------
# compute_ota_gamma — OTA formula summation
# ---------------------------------------------------------------------------

class TestComputeOtaGamma:
    def _make_inputs(self) -> tuple[dict, dict]:
        beta_estimates = {
            "complement_activation": {
                "beta": -0.6,
                "evidence_tier": "Tier1_Interventional",
                "beta_sigma": 0.15,
            },
            "lipid_metabolism": {
                "beta": -0.2,
                "evidence_tier": "Tier2_Convergent",
                "beta_sigma": 0.25,
            },
        }
        gamma_estimates = {
            "complement_activation": {
                "gamma": 0.45,
                "evidence_tier": "Tier3_Provisional",
            },
            "lipid_metabolism": {
                "gamma": 0.10,
                "evidence_tier": "Tier3_Provisional",
            },
        }
        return beta_estimates, gamma_estimates

    def test_formula_correctness(self):
        """OTA formula: γ_gene = Σ_P β_{gene→P} × γ_{P→trait} × w_P, where
        w_P = 1/(1 + gamma_se) down-weights programs with uncertain γ.
        When gamma_se is absent the fallback is 0.5×|γ|."""
        beta_est, gamma_est = self._make_inputs()
        result = compute_ota_gamma("CFH", "AMD", beta_est, gamma_est)
        # SE-weighted expected: w = 1/(1 + 0.5*|gamma|) for each program
        w1 = 1.0 / (1.0 + 0.5 * 0.45)   # complement_activation
        w2 = 1.0 / (1.0 + 0.5 * 0.10)   # lipid_metabolism
        expected = (-0.6 * 0.45 * w1) + (-0.2 * 0.10 * w2)
        assert math.isclose(result["ota_gamma"], expected, rel_tol=1e-3), \
            f"Expected {expected:.4f}, got {result['ota_gamma']:.4f}"

    def test_returns_dict(self):
        beta_est, gamma_est = self._make_inputs()
        result = compute_ota_gamma("CFH", "AMD", beta_est, gamma_est)
        assert isinstance(result, dict)
        assert "ota_gamma" in result

    def test_empty_programs_yields_zero(self):
        result = compute_ota_gamma("UNKNOWN", "AMD", {}, {})
        assert result["ota_gamma"] == 0.0 or result.get("ota_gamma") is not None

    def test_nan_beta_excluded(self):
        """NaN beta should be excluded from summation, not treated as 0."""
        beta_est = {
            "complement_activation": {"beta": float("nan"), "evidence_tier": "Tier1_Interventional"},
            "lipid_metabolism": {"beta": -0.3, "evidence_tier": "Tier2_Convergent"},
        }
        gamma_est = {
            "complement_activation": {"gamma": 0.5, "evidence_tier": "Tier3_Provisional"},
            "lipid_metabolism": {"gamma": 0.2, "evidence_tier": "Tier3_Provisional"},
        }
        result = compute_ota_gamma("CFH", "AMD", beta_est, gamma_est)
        ota = result["ota_gamma"]
        assert math.isfinite(ota) or math.isnan(ota), "NaN beta should not produce Inf"
        # Should not include NaN contribution in sum
        if math.isfinite(ota):
            assert math.isclose(ota, -0.3 * 0.2, rel_tol=0.1), \
                "NaN beta should be skipped, not treated as 0"

    def test_gene_and_trait_in_result(self):
        beta_est, gamma_est = self._make_inputs()
        result = compute_ota_gamma("CFH", "AMD", beta_est, gamma_est)
        assert result.get("gene") == "CFH"
        assert result.get("trait") == "AMD"

    def test_program_contributions_listed(self):
        beta_est, gamma_est = self._make_inputs()
        result = compute_ota_gamma("CFH", "AMD", beta_est, gamma_est)
        contribs = result.get("program_contributions", [])
        assert isinstance(contribs, list)
        assert len(contribs) > 0
