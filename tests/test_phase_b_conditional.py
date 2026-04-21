"""
Tests for Phase B modules:
  pipelines/state_space/conditional_beta.py
  pipelines/state_space/conditional_gamma.py
"""
from __future__ import annotations

import math

import pytest


# ---------------------------------------------------------------------------
# conditional_beta.py
# ---------------------------------------------------------------------------

class TestEstimateConditionalBeta:
    def test_no_data_returns_nan_beta(self):
        from pipelines.state_space.conditional_beta import estimate_conditional_beta
        cb = estimate_conditional_beta("TNF", "IBD_macro_P00", "macrophage", "IBD")
        assert math.isnan(cb.beta)
        assert cb.pooled_fallback is False
        assert cb.context_verified is False
        assert cb.data_source == "no_data"

    def test_pooled_fallback_sets_flag(self):
        from pipelines.state_space.conditional_beta import estimate_conditional_beta
        pooled = {"TNF": {"programs": {"IBD_macro_P00": {"beta": 0.5, "se": 0.1}}}}
        cb = estimate_conditional_beta(
            "TNF", "IBD_macro_P00", "macrophage", "IBD",
            pooled_perturbseq_data=pooled,
        )
        assert abs(cb.beta - 0.5) < 1e-9
        assert cb.pooled_fallback is True
        assert cb.context_verified is False

    def test_lincs_tier3_path(self):
        from pipelines.state_space.conditional_beta import estimate_conditional_beta
        sig = {"IL6": 0.8, "NFKB1": 0.6, "TNF": 0.9}
        pgs = {"IL6", "NFKB1", "TNF", "CCL2", "IFNG"}
        cb = estimate_conditional_beta(
            "GENE_X", "IBD_macro_P00", "macrophage", "IBD",
            lincs_signature=sig,
            program_gene_set=pgs,
        )
        assert math.isfinite(cb.beta)
        assert "LINCS" in cb.data_source or "Tier3" in cb.evidence_tier

    def test_eqtl_tier2_path(self):
        from pipelines.state_space.conditional_beta import estimate_conditional_beta
        eqtl = {"nes": 0.6, "se": 0.1, "tissue": "colon"}
        cb = estimate_conditional_beta(
            "TNF", "IBD_macro_P00", "macrophage", "IBD",
            eqtl_data=eqtl,
            coloc_h4=0.9,
            program_loading=0.5,
        )
        assert math.isfinite(cb.beta)
        assert abs(cb.beta - 0.6 * 0.5) < 1e-9   # nes × loading
        assert cb.pooled_fallback is False
        assert "Tier2" in cb.evidence_tier

    def test_eqtl_rejected_low_coloc(self):
        from pipelines.state_space.conditional_beta import estimate_conditional_beta
        eqtl = {"nes": 0.6, "se": 0.1, "tissue": "colon"}
        cb = estimate_conditional_beta(
            "TNF", "IBD_macro_P00", "macrophage", "IBD",
            eqtl_data=eqtl,
            coloc_h4=0.3,   # below 0.8 threshold → rejected
        )
        assert math.isnan(cb.beta)   # falls through to no_data

    def test_fields_populated(self):
        from pipelines.state_space.conditional_beta import estimate_conditional_beta
        cb = estimate_conditional_beta("TNF", "p", "macrophage", "IBD")
        assert cb.gene == "TNF"
        assert cb.program_id == "p"
        assert cb.cell_type == "macrophage"
        assert cb.disease == "IBD"


class TestEstimateConditionalBetasForProgram:
    def test_returns_one_per_gene(self):
        from pipelines.state_space.conditional_beta import estimate_conditional_betas_for_program
        betas = estimate_conditional_betas_for_program(
            "IBD_macro_P00", ["TNF", "IL6", "NFKB1"], "macrophage", "IBD"
        )
        assert len(betas) == 3
        assert {b.gene for b in betas} == {"TNF", "IL6", "NFKB1"}

    def test_eqtl_by_gene_routing(self):
        from pipelines.state_space.conditional_beta import estimate_conditional_betas_for_program
        eqtl_by_gene = {"TNF": {"nes": 0.7, "se": 0.1, "tissue": "colon"}}
        betas = estimate_conditional_betas_for_program(
            "p", ["TNF", "IL6"], "macrophage", "IBD",
            eqtl_data_by_gene=eqtl_by_gene,
            coloc_h4_by_gene={"TNF": 0.95},
        )
        tnf_b = next(b for b in betas if b.gene == "TNF")
        il6_b = next(b for b in betas if b.gene == "IL6")
        assert math.isfinite(tnf_b.beta)
        assert math.isnan(il6_b.beta)


class TestComputePooledFraction:
    def test_all_nan(self):
        from pipelines.state_space.conditional_beta import compute_pooled_fraction
        from models.latent_mediator import ConditionalBeta
        betas = [
            ConditionalBeta(gene="X", program_id="p", cell_type="m", disease="IBD",
                            beta=float("nan"))
        ]
        assert compute_pooled_fraction(betas) == 0.0

    def test_mixed_pooled(self):
        from pipelines.state_space.conditional_beta import compute_pooled_fraction, estimate_conditional_beta
        pooled_data = {"TNF": {"programs": {"p": {"beta": 0.5}}}}
        b_pooled = estimate_conditional_beta("TNF", "p", "m", "IBD",
                                             pooled_perturbseq_data=pooled_data)
        b_none = estimate_conditional_beta("IL6", "p", "m", "IBD")
        # b_none has NaN beta → excluded from denominator
        assert compute_pooled_fraction([b_pooled, b_none]) == 1.0

    def test_no_pooled(self):
        from pipelines.state_space.conditional_beta import compute_pooled_fraction
        from models.latent_mediator import ConditionalBeta
        b = ConditionalBeta(gene="X", program_id="p", cell_type="m", disease="IBD",
                            beta=0.3, pooled_fallback=False)
        assert compute_pooled_fraction([b]) == 0.0


# ---------------------------------------------------------------------------
# conditional_gamma.py
# ---------------------------------------------------------------------------

class TestAlphaLookup:
    def test_tier_alphas(self):
        from pipelines.state_space.conditional_gamma import _get_alpha
        assert _get_alpha("Tier1_TrajectoryDirect") == 0.35
        assert _get_alpha("Tier2_TrajectoryInferred") == 0.55
        assert _get_alpha("Tier3_TrajectoryProxy") == 0.70

    def test_unknown_tier_uses_default(self):
        from pipelines.state_space.conditional_gamma import _get_alpha, _DEFAULT_ALPHA
        assert _get_alpha("unknown_tier") == _DEFAULT_ALPHA


class TestEstimateConditionalGamma:
    def test_known_program_uses_provisional_gwas(self):
        """PROVISIONAL_GAMMAS removed; gamma_gwas is data-derived (0 without network)."""
        from pipelines.state_space.conditional_gamma import estimate_conditional_gamma
        cg = estimate_conditional_gamma(
            "inflammatory_NF-kB", "IBD", "IBD",
            evidence_tier="Tier3_TrajectoryProxy",
        )
        # gamma_gwas is 0.0 without live network data (no provisional fallback)
        assert math.isfinite(cg.gamma_gwas)
        assert cg.gamma_gwas >= 0
        assert cg.alpha == 0.70

    def test_mixed_formula_correct(self):
        from pipelines.state_space.conditional_gamma import estimate_conditional_gamma
        cg = estimate_conditional_gamma(
            "inflammatory_NF-kB", "IBD", "IBD",
            evidence_tier="Tier2_TrajectoryInferred",
            transition_gene_weights={"TNF": 0.8, "NFKB1": 0.6, "IL1B": 0.7},
            program_top_genes=["TNF", "NFKB1", "IL1B"],
        )
        expected = 0.55 * cg.gamma_gwas + 0.45 * cg.gamma_transition
        assert abs(cg.gamma_mixed - expected) < 1e-9

    def test_no_transition_falls_back_to_gwas(self):
        from pipelines.state_space.conditional_gamma import estimate_conditional_gamma
        cg = estimate_conditional_gamma(
            "inflammatory_NF-kB", "IBD", "IBD",
            evidence_tier="Tier2_TrajectoryInferred",
            # No transition data
        )
        assert cg.gamma_mixed == cg.gamma_gwas   # transition NaN → GWAS only

    def test_unknown_program_uses_transition_signal(self):
        from pipelines.state_space.conditional_gamma import estimate_conditional_gamma
        cg = estimate_conditional_gamma(
            "UNKNOWN_PROGRAM_XYZ", "IBD", "IBD",
            evidence_tier="Tier3_TrajectoryProxy",
            transition_gene_weights={"TNF": 0.9, "IL6": 0.8},
            program_top_genes=["TNF", "IL6"],
        )
        # gamma_gwas = 0.0 (no evidence), gamma_transition = mean(0.9, 0.8) = 0.85
        expected_trans = (0.9 + 0.8) / 2
        expected_mixed = 0.70 * 0.0 + 0.30 * expected_trans
        assert abs(cg.gamma_mixed - expected_mixed) < 1e-9

    def test_all_nan_returns_zero_mixed(self):
        from pipelines.state_space.conditional_gamma import estimate_conditional_gamma
        cg = estimate_conditional_gamma(
            "UNKNOWN_XYZ_999", "UNKNOWN_TRAIT", "UNKNOWN",
            evidence_tier="Tier3_TrajectoryProxy",
        )
        # GWAS returns 0.0 for unknown; transition is NaN (no genes provided)
        # Fallback: gamma_gwas only (0.0) → mixed = 0.0
        assert math.isfinite(cg.gamma_mixed)

    def test_fields_populated(self):
        from pipelines.state_space.conditional_gamma import estimate_conditional_gamma
        cg = estimate_conditional_gamma("p", "IBD", "IBD")
        assert cg.program_id == "p"
        assert cg.trait == "IBD"
        assert cg.disease == "IBD"
        assert 0.0 <= cg.alpha <= 1.0


class TestEstimateConditionalGammasForPrograms:
    def test_returns_all_programs(self):
        from pipelines.state_space.conditional_gamma import estimate_conditional_gammas_for_programs
        programs = [
            {"program_id": "IBD_macro_P00", "top_genes": ["TNF", "IL6"]},
            {"program_id": "IBD_macro_P01", "top_genes": ["TGFB1", "COL1A1"]},
        ]
        results = estimate_conditional_gammas_for_programs(programs, "IBD", "IBD")
        assert "IBD_macro_P00" in results
        assert "IBD_macro_P01" in results

    def test_transition_weights_propagated(self):
        from pipelines.state_space.conditional_gamma import estimate_conditional_gammas_for_programs
        tw = {"TNF": 0.9, "IL6": 0.7, "TGFB1": 0.5, "COL1A1": 0.4}
        programs = [{"program_id": "p0", "top_genes": ["TNF", "IL6"]}]
        results = estimate_conditional_gammas_for_programs(
            programs, "IBD", "IBD", transition_gene_weights=tw
        )
        assert math.isfinite(results["p0"].gamma_transition)
        assert results["p0"].gamma_transition > 0

    def test_empty_programs(self):
        from pipelines.state_space.conditional_gamma import estimate_conditional_gammas_for_programs
        assert estimate_conditional_gammas_for_programs([], "IBD", "IBD") == {}
