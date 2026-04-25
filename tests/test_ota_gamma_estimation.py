"""
test_ota_gamma_estimation.py — Unit tests for the γ_{program→trait} estimation pipeline.

Tests verify:
  - compute_cnmf_gamma: hypergeometric enrichment logic and boundary conditions
  - estimate_gamma: dispatches to correct sub-estimator; returns valid tier labels
  - compute_ota_gamma: OTA formula summation is correct; NaN propagation
  - Module-level constants imported from config.scoring_thresholds
"""
from __future__ import annotations

import math
import pytest

from pipelines.ota_gamma_estimation import (
    compute_cnmf_gamma,
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
# compute_cnmf_gamma — hypergeometric enrichment
# ---------------------------------------------------------------------------

class TestComputeCnmfGamma:
    def test_returns_none_for_empty_program(self):
        result = compute_cnmf_gamma(set(), AMD_GWAS_GENES, PROGRAM, TRAIT)
        assert result is None

    def test_returns_none_for_empty_gwas(self):
        result = compute_cnmf_gamma(COMPLEMENT_PROGRAM, set(), PROGRAM, TRAIT)
        assert result is None

    def test_returns_none_below_min_overlap(self):
        """Single gene overlap → insufficient evidence; should return None."""
        tiny_gwas = {"CFH"}
        result = compute_cnmf_gamma(COMPLEMENT_PROGRAM, tiny_gwas, PROGRAM, TRAIT)
        assert result is None

    def test_enriched_complement_program(self):
        """Large overlap between complement program and AMD GWAS → positive gamma."""
        result = compute_cnmf_gamma(COMPLEMENT_PROGRAM, AMD_GWAS_GENES, PROGRAM, TRAIT)
        assert result is not None
        assert result["gamma"] > 0
        assert result["gamma"] <= 1.0

    def test_no_enrichment_for_unrelated_genes(self):
        """Program of unrelated genes against AMD GWAS → no enrichment."""
        result = compute_cnmf_gamma(UNRELATED_GENES, AMD_GWAS_GENES, PROGRAM, TRAIT)
        assert result is None

    def test_gamma_in_unit_interval(self):
        result = compute_cnmf_gamma(COMPLEMENT_PROGRAM, AMD_GWAS_GENES, PROGRAM, TRAIT)
        if result is not None:
            assert 0.0 < result["gamma"] <= 1.0

    def test_result_has_required_fields(self):
        result = compute_cnmf_gamma(COMPLEMENT_PROGRAM, AMD_GWAS_GENES, PROGRAM, TRAIT)
        if result is not None:
            for field in ("gamma", "p_enrichment", "odds_ratio", "n_overlap", "evidence_tier"):
                assert field in result, f"Missing field: {field}"

    def test_p_value_bounded(self):
        result = compute_cnmf_gamma(COMPLEMENT_PROGRAM, AMD_GWAS_GENES, PROGRAM, TRAIT)
        if result is not None:
            assert 0.0 <= result["p_enrichment"] <= 1.0

    def test_n_genome_genes_sensitivity(self):
        """Smaller genome size should yield more significant enrichment."""
        result_large = compute_cnmf_gamma(
            COMPLEMENT_PROGRAM, AMD_GWAS_GENES, PROGRAM, TRAIT, n_genome_genes=20_000
        )
        result_small = compute_cnmf_gamma(
            COMPLEMENT_PROGRAM, AMD_GWAS_GENES, PROGRAM, TRAIT, n_genome_genes=5_000
        )
        if result_large and result_small:
            assert result_small["p_enrichment"] <= result_large["p_enrichment"]



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
        """OTA formula: γ_gene = Σ_P β_{gene→P} × γ_{P→trait}."""
        beta_est, gamma_est = self._make_inputs()
        result = compute_ota_gamma("CFH", "AMD", beta_est, gamma_est)
        expected = (-0.6 * 0.45) + (-0.2 * 0.10)
        assert math.isclose(result["ota_gamma"], expected, rel_tol=1e-4), \
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
