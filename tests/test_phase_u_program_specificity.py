"""
tests/test_phase_u_program_specificity.py

Tests for:
  - compute_program_disease_weights  (program_specificity.py)
  - compute_beta_amd_concentration   (program_specificity.py)
  - compute_ota_gamma program_weights param  (ota_gamma_estimation.py)
  - compute_ota_gamma_with_uncertainty program_weights param
"""
from __future__ import annotations

import math
import unittest
from unittest.mock import MagicMock

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adata(n_amd: int, n_healthy: int, n_genes: int = 20,
                amd_multiplier: float = 3.0) -> "AnnData":
    """Build a minimal AnnData with AMD and healthy cells."""
    try:
        import anndata as ad
    except ImportError:
        raise unittest.SkipTest("anndata not installed")

    n_cells = n_amd + n_healthy
    rng = np.random.default_rng(42)

    # AMD cells have higher expression for genes 0..4 (first half)
    X = rng.uniform(0.1, 1.0, size=(n_cells, n_genes)).astype(np.float32)
    X[:n_amd, :5] *= amd_multiplier  # AMD-specific upregulation in first 5 genes

    disease_labels = (
        ["age-related macular degeneration"] * n_amd +
        ["normal"] * n_healthy
    )
    obs = pd.DataFrame({"disease": disease_labels})
    var_names = [f"GENE{i}" for i in range(n_genes)]
    var = pd.DataFrame(index=var_names)

    adata = ad.AnnData(X=X, obs=obs, var=var)
    return adata


def _make_programs() -> list[dict]:
    """Two programs: one AMD-specific (GENE0-4), one generic (GENE5-9)."""
    return [
        {
            "program_id": "amd_complement",
            "gene_set": [f"GENE{i}" for i in range(5)],  # AMD-upregulated
        },
        {
            "program_id": "generic_housekeeping",
            "gene_set": [f"GENE{i}" for i in range(5, 10)],  # not AMD-specific
        },
    ]


# ---------------------------------------------------------------------------
# compute_program_disease_weights
# ---------------------------------------------------------------------------

class TestComputeProgramDiseaseWeights(unittest.TestCase):

    def setUp(self):
        try:
            import anndata  # noqa: F401
        except ImportError:
            self.skipTest("anndata not installed")

    def test_amd_specific_program_gets_high_weight(self):
        from pipelines.state_space.program_specificity import compute_program_disease_weights
        adata = _make_adata(n_amd=50, n_healthy=50, amd_multiplier=4.0)
        programs = _make_programs()
        weights = compute_program_disease_weights(adata, programs)

        self.assertIn("amd_complement", weights)
        self.assertIn("generic_housekeeping", weights)
        # AMD-specific program should have higher weight
        self.assertGreater(weights["amd_complement"], weights["generic_housekeeping"])
        self.assertGreater(weights["amd_complement"], 0.5)

    def test_generic_program_gets_low_weight(self):
        from pipelines.state_space.program_specificity import compute_program_disease_weights
        adata = _make_adata(n_amd=50, n_healthy=50, amd_multiplier=4.0)
        programs = _make_programs()
        weights = compute_program_disease_weights(adata, programs)
        # Generic program (no AMD-specific upregulation) should get low weight
        self.assertLess(weights["generic_housekeeping"], 0.3)

    def test_weights_bounded_zero_one(self):
        from pipelines.state_space.program_specificity import compute_program_disease_weights
        adata = _make_adata(n_amd=50, n_healthy=50, amd_multiplier=10.0)
        programs = _make_programs()
        weights = compute_program_disease_weights(adata, programs)
        for pid, w in weights.items():
            self.assertGreaterEqual(w, 0.0, f"{pid} weight {w} < 0")
            self.assertLessEqual(w, 1.0, f"{pid} weight {w} > 1")

    def test_fallback_when_no_disease_column(self):
        from pipelines.state_space.program_specificity import compute_program_disease_weights
        import anndata as ad
        adata = ad.AnnData(
            X=np.ones((10, 5), dtype=np.float32),
            obs=pd.DataFrame({"cell_type": ["RPE"] * 10}),
            var=pd.DataFrame(index=[f"GENE{i}" for i in range(5)]),
        )
        weights = compute_program_disease_weights(adata, _make_programs())
        # Should return empty dict (no disease column)
        self.assertEqual(weights, {})

    def test_fallback_with_none_adata(self):
        from pipelines.state_space.program_specificity import compute_program_disease_weights
        weights = compute_program_disease_weights(None, _make_programs())
        self.assertEqual(weights, {})

    def test_too_few_cells_returns_uniform(self):
        from pipelines.state_space.program_specificity import compute_program_disease_weights
        adata = _make_adata(n_amd=5, n_healthy=3)  # below _MIN_CELLS_PER_GROUP=10
        programs = _make_programs()
        weights = compute_program_disease_weights(adata, programs)
        # Should return uniform 1.0 weights (not enough contrast)
        for w in weights.values():
            self.assertEqual(w, 1.0)

    def test_gene_loadings_mode(self):
        """Programs defined via gene_loadings dict work correctly."""
        from pipelines.state_space.program_specificity import compute_program_disease_weights
        adata = _make_adata(n_amd=50, n_healthy=50, amd_multiplier=4.0)
        programs = [
            {
                "program_id": "weighted_amd",
                "gene_loadings": {f"GENE{i}": float(i + 1) for i in range(5)},
            }
        ]
        weights = compute_program_disease_weights(adata, programs)
        self.assertIn("weighted_amd", weights)
        self.assertGreater(weights["weighted_amd"], 0.5)


# ---------------------------------------------------------------------------
# compute_beta_amd_concentration
# ---------------------------------------------------------------------------

class TestComputeBetaAmdConcentration(unittest.TestCase):

    def test_all_in_amd_programs(self):
        from pipelines.state_space.program_specificity import compute_beta_amd_concentration
        beta = {
            "amd_complement": {"beta": 0.8},
            "generic_housekeeping": {"beta": 0.1},
        }
        weights = {"amd_complement": 1.0, "generic_housekeeping": 0.0}
        conc = compute_beta_amd_concentration(beta, weights)
        # 0.8 / (0.8 + 0.1) ≈ 0.889
        self.assertAlmostEqual(conc, 0.8 / 0.9, places=2)

    def test_uniform_beta_low_concentration(self):
        """JAZF1-like gene: equal β across all programs, generic programs dominate."""
        from pipelines.state_space.program_specificity import compute_beta_amd_concentration
        beta = {f"prog_{i}": {"beta": 1.0} for i in range(10)}
        weights = {f"prog_{i}": (1.0 if i == 0 else 0.0) for i in range(10)}
        conc = compute_beta_amd_concentration(beta, weights)
        # Only 1/10 of β-mass in AMD-specific program
        self.assertAlmostEqual(conc, 0.1, places=2)

    def test_nan_beta_skipped(self):
        from pipelines.state_space.program_specificity import compute_beta_amd_concentration
        beta = {"prog_amd": {"beta": float("nan")}, "prog_gen": {"beta": 0.5}}
        weights = {"prog_amd": 1.0, "prog_gen": 0.0}
        conc = compute_beta_amd_concentration(beta, weights)
        self.assertEqual(conc, 0.0)  # NaN skipped, only generic with w=0

    def test_empty_inputs(self):
        from pipelines.state_space.program_specificity import compute_beta_amd_concentration
        self.assertEqual(compute_beta_amd_concentration({}, {}), 0.0)
        self.assertEqual(compute_beta_amd_concentration({"p": {"beta": 1.0}}, {}), 0.0)


class TestEstimateBetaTier25(unittest.TestCase):
    """Tests for Tier 2.5: eQTL direction-only fallback."""

    def setUp(self):
        from pipelines.ota_beta_estimation import estimate_beta_tier2_eqtl_direction
        self.fn = estimate_beta_tier2_eqtl_direction

    def test_none_when_no_eqtl(self):
        result = self.fn("CFH", "complement_activation", eqtl_data=None, coloc_h4=0.3)
        self.assertIsNone(result)

    def test_none_when_coloc_passes_tier2_threshold(self):
        # coloc_h4 >= 0.8 → should use Tier2, not Tier2.5
        eqtl = {"nes": 0.5, "tissue": "Retina"}
        result = self.fn("LIPC", "lipid_metabolism", eqtl_data=eqtl, coloc_h4=0.85)
        self.assertIsNone(result)

    def test_activates_when_coloc_below_threshold(self):
        eqtl = {"nes": 0.4, "tissue": "Liver"}
        result = self.fn("FBN2", "ECM_remodeling", eqtl_data=eqtl, coloc_h4=0.45, program_loading=0.7)
        self.assertIsNotNone(result)
        self.assertEqual(result["evidence_tier"], "Tier2_eQTL_direction")
        self.assertAlmostEqual(result["beta_sigma"], 0.50)

    def test_activates_when_coloc_absent(self):
        eqtl = {"nes": -0.3, "tissue": "Whole_Blood"}
        result = self.fn("PRPH2", "photoreceptor_integrity", eqtl_data=eqtl, coloc_h4=None)
        self.assertIsNotNone(result)
        self.assertEqual(result["evidence_tier"], "Tier2_eQTL_direction")

    def test_sign_preserved_positive(self):
        eqtl = {"nes": 0.6, "tissue": "Retina"}
        result = self.fn("GENE1", "prog1", eqtl_data=eqtl, coloc_h4=0.2, program_loading=0.8)
        self.assertGreater(result["beta"], 0)

    def test_sign_preserved_negative(self):
        eqtl = {"nes": -0.6, "tissue": "Retina"}
        result = self.fn("GENE1", "prog1", eqtl_data=eqtl, coloc_h4=0.2, program_loading=0.8)
        self.assertLess(result["beta"], 0)

    def test_magnitude_from_loading_not_nes(self):
        # beta = sign(NES) × |loading|  — magnitude is loading, not NES
        eqtl_small_nes = {"nes": 0.1, "tissue": "Retina"}
        eqtl_large_nes = {"nes": 0.9, "tissue": "Retina"}
        loading = 0.5
        r1 = self.fn("G", "P", eqtl_data=eqtl_small_nes, coloc_h4=None, program_loading=loading)
        r2 = self.fn("G", "P", eqtl_data=eqtl_large_nes, coloc_h4=None, program_loading=loading)
        self.assertAlmostEqual(abs(r1["beta"]), abs(r2["beta"]), places=5)
        self.assertAlmostEqual(abs(r1["beta"]), loading, places=5)

    def test_fallback_chain_uses_tier25(self):
        """estimate_beta() should reach Tier2.5 when eQTL present but COLOC weak."""
        from pipelines.ota_beta_estimation import estimate_beta
        eqtl = {"nes": 0.35, "tissue": "Liver"}
        result = estimate_beta(
            gene="HMCN1",
            program="ECM_remodeling",
            eqtl_data=eqtl,
            coloc_h4=0.55,
            program_loading=0.6,
        )
        self.assertEqual(result["evidence_tier"], "Tier2_eQTL_direction")
        self.assertEqual(result["tier_used"], 2)


if __name__ == "__main__":
    unittest.main()
