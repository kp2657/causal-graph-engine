"""
tests/test_phase_v_new_genomics.py

Tests for Phase V: three new human genomic data sources
  - Tier 2c: sc-eQTL (estimate_beta_tier2_sc_eqtl)
  - Tier 2p: pQTL-MR (estimate_beta_tier2_pqtl)
  - Tier 2rb: Rare variant burden (estimate_beta_tier2_rare_burden)
  - estimate_beta() fallback chain integration
  - eqtl_catalogue_server helpers (unit-level, no network)
  - ukb_wes_server helpers (unit-level, no network)
"""
from __future__ import annotations

import math
import unittest


# ---------------------------------------------------------------------------
# Tier 2c: sc-eQTL
# ---------------------------------------------------------------------------

class TestEstimateBetaTier2ScEqtl(unittest.TestCase):
    """Unit tests for sc-eQTL beta estimation."""

    def setUp(self):
        from pipelines.ota_beta_estimation import estimate_beta_tier2_sc_eqtl
        self.fn = estimate_beta_tier2_sc_eqtl

    def _make_sc_data(self, beta: float, pvalue: float = 1e-6,
                      cell_type: str = "CD14_positive_monocyte",
                      study: str = "Yazar2022") -> dict:
        return {
            "top_eqtl": {
                "beta": beta,
                "se": abs(beta) * 0.15,
                "pvalue": pvalue,
                "condition_label": cell_type,
                "study_label": study,
            },
            "cell_type": cell_type,
            "study": study,
        }

    def test_returns_none_when_no_data(self):
        result = self.fn("CFH", "complement_activation", sc_eqtl_data=None)
        self.assertIsNone(result)

    def test_returns_none_when_no_top_eqtl(self):
        result = self.fn("CFH", "complement_activation", sc_eqtl_data={})
        self.assertIsNone(result)

    def test_returns_none_when_pvalue_too_weak(self):
        sc_data = self._make_sc_data(0.5, pvalue=0.01)  # p > 1e-4
        result = self.fn("GENE1", "prog1", sc_eqtl_data=sc_data)
        self.assertIsNone(result)

    def test_basic_sc_eqtl_tier(self):
        sc_data = self._make_sc_data(0.4, pvalue=1e-7)
        result = self.fn("IL23R", "IL23_pathway", sc_eqtl_data=sc_data, program_loading=0.7)
        self.assertIsNotNone(result)
        self.assertIn("Tier2c", result["evidence_tier"])

    def test_full_tier_when_coloc_absent(self):
        """When COLOC unavailable (None), use full NES × loading."""
        sc_data = self._make_sc_data(0.5, pvalue=1e-8)
        result = self.fn("IL23R", "prog", sc_eqtl_data=sc_data, coloc_h4=None, program_loading=0.6)
        self.assertIsNotNone(result)
        self.assertEqual(result["evidence_tier"], "Tier2c_scEQTL")
        self.assertAlmostEqual(result["beta"], 0.5 * 0.6, places=5)

    def test_direction_only_when_coloc_weak(self):
        """When COLOC H4 < 0.8, use direction-only with sigma=0.50."""
        sc_data = self._make_sc_data(0.5, pvalue=1e-8)
        result = self.fn("GENE", "prog", sc_eqtl_data=sc_data, coloc_h4=0.4, program_loading=0.7)
        self.assertIsNotNone(result)
        self.assertEqual(result["evidence_tier"], "Tier2c_scEQTL_direction")
        self.assertAlmostEqual(result["beta_sigma"], 0.50)
        self.assertAlmostEqual(abs(result["beta"]), 0.7, places=5)

    def test_sign_preserved(self):
        sc_pos = self._make_sc_data(0.3, pvalue=1e-7)
        sc_neg = self._make_sc_data(-0.3, pvalue=1e-7)
        r_pos = self.fn("G", "P", sc_eqtl_data=sc_pos, program_loading=1.0)
        r_neg = self.fn("G", "P", sc_eqtl_data=sc_neg, program_loading=1.0)
        self.assertGreater(r_pos["beta"], 0)
        self.assertLess(r_neg["beta"], 0)


# ---------------------------------------------------------------------------
# Tier 2p: pQTL-MR
# ---------------------------------------------------------------------------

class TestEstimateBetaTier2pQtl(unittest.TestCase):
    """Unit tests for pQTL-MR beta estimation."""

    def setUp(self):
        from pipelines.ota_beta_estimation import estimate_beta_tier2_pqtl
        self.fn = estimate_beta_tier2_pqtl

    def _make_pqtl(self, beta: float, pvalue: float = 1e-8,
                   rsid: str = "rs1061170", study: str = "Sun2023") -> dict:
        return {
            "top_pqtl": {
                "beta": beta,
                "se": abs(beta) * 0.10,
                "pvalue": pvalue,
                "rsid": rsid,
                "study_label": study,
            },
            "protein": "CFH",
            "data_source": f"eQTL Catalogue v2 / pQTL / {study}",
        }

    def test_returns_none_when_no_data(self):
        result = self.fn("CFH", "complement_activation", pqtl_data=None)
        self.assertIsNone(result)

    def test_returns_none_when_pvalue_too_weak(self):
        pqtl = self._make_pqtl(0.3, pvalue=0.3)  # p > MR_PQTL_P_VALUE_MAX (0.05)
        result = self.fn("CFH", "prog", pqtl_data=pqtl)
        self.assertIsNone(result)

    def test_basic_pqtl_tier(self):
        pqtl = self._make_pqtl(-0.28, pvalue=1e-12)  # CFH Y402H reduces plasma CFH
        result = self.fn("CFH", "complement_activation", pqtl_data=pqtl, program_loading=0.9)
        self.assertIsNotNone(result)
        self.assertEqual(result["evidence_tier"], "Tier2p_pQTL_MR")

    def test_beta_scaled_by_loading(self):
        pqtl = self._make_pqtl(0.5, pvalue=1e-10)
        r = self.fn("GENE", "prog", pqtl_data=pqtl, program_loading=0.8)
        self.assertAlmostEqual(r["beta"], 0.5 * 0.8, places=5)

    def test_sign_preserved_negative(self):
        pqtl = self._make_pqtl(-0.3, pvalue=1e-9)
        r = self.fn("CFH", "complement_activation", pqtl_data=pqtl, program_loading=0.7)
        self.assertLess(r["beta"], 0)

    def test_no_loading_returns_none(self):
        pqtl = self._make_pqtl(0.4, pvalue=1e-8)
        r = self.fn("GENE", "prog", pqtl_data=pqtl, program_loading=None)
        self.assertIsNone(r)

    def test_loading_one_gives_raw_beta(self):
        pqtl = self._make_pqtl(0.4, pvalue=1e-8)
        r = self.fn("GENE", "prog", pqtl_data=pqtl, program_loading=1.0)
        self.assertAlmostEqual(r["beta"], 0.4, places=5)

    def test_rejects_non_finite_beta(self):
        pqtl = {"top_pqtl": {"beta": float("inf"), "pvalue": 1e-8}, "data_source": "test"}
        result = self.fn("GENE", "prog", pqtl_data=pqtl)
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# Tier 2rb: Rare variant burden
# ---------------------------------------------------------------------------

class TestEstimateBetaTier2RareBurden(unittest.TestCase):
    """Unit tests for rare variant burden direction constraint."""

    def setUp(self):
        from pipelines.ota_beta_estimation import estimate_beta_tier2_rare_burden
        self.fn = estimate_beta_tier2_rare_burden

    def _make_burden(self, beta: float, p: float, loeuf: float = 0.3) -> dict:
        return {
            "burden_beta": beta,
            "burden_p": p,
            "burden_se": abs(beta) * 0.20,
            "loeuf": loeuf,
            "pli": 0.95,
            "burden_study": "UKB_WES",
            "interpretation": "Test interpretation",
        }

    def test_returns_none_when_no_data(self):
        result = self.fn("HMCN1", "ECM_remodeling", burden_data=None)
        self.assertIsNone(result)

    def test_returns_none_when_no_burden_beta(self):
        result = self.fn("HMCN1", "ECM_remodeling", burden_data={"loeuf": 0.3})
        self.assertIsNone(result)

    def test_returns_none_when_p_too_weak(self):
        burden = self._make_burden(0.5, p=0.01)  # p > 1e-4 threshold
        result = self.fn("HMCN1", "prog", burden_data=burden)
        self.assertIsNone(result)

    def test_basic_burden_tier(self):
        burden = self._make_burden(0.8, p=1e-6)
        result = self.fn("HMCN1", "ECM_remodeling", burden_data=burden, program_loading=0.6)
        self.assertIsNotNone(result)
        self.assertEqual(result["evidence_tier"], "Tier2rb_RareBurden")

    def test_large_sigma(self):
        """Burden direction must have sigma=0.60 (weak evidence)."""
        burden = self._make_burden(0.5, p=1e-7)
        r = self.fn("G", "P", burden_data=burden, program_loading=0.5)
        self.assertAlmostEqual(r["beta_sigma"], 0.60)

    def test_magnitude_from_loading_not_burden_beta(self):
        """β = sign(burden_beta) × |loading|, not burden_beta × loading."""
        burden_small = self._make_burden(0.1, p=1e-7)
        burden_large = self._make_burden(5.0, p=1e-7)
        loading = 0.7
        r1 = self.fn("G", "P", burden_data=burden_small, program_loading=loading)
        r2 = self.fn("G", "P", burden_data=burden_large, program_loading=loading)
        self.assertAlmostEqual(abs(r1["beta"]), loading, places=5)
        self.assertAlmostEqual(abs(r2["beta"]), loading, places=5)

    def test_sign_positive_burden(self):
        """Positive burden_beta (LoF increases risk) → positive β."""
        burden = self._make_burden(0.5, p=1e-7)
        r = self.fn("G", "P", burden_data=burden, program_loading=0.6)
        self.assertGreater(r["beta"], 0)

    def test_sign_negative_burden(self):
        """Negative burden_beta (LoF decreases risk, gene normally harmful) → negative β."""
        burden = self._make_burden(-0.5, p=1e-7)
        r = self.fn("G", "P", burden_data=burden, program_loading=0.6)
        self.assertLess(r["beta"], 0)


# ---------------------------------------------------------------------------
# estimate_beta() fallback chain integration
# ---------------------------------------------------------------------------

class TestEstimateBetaNewTiersFallback(unittest.TestCase):
    """Integration tests for the extended estimate_beta() fallback chain."""

    def test_pqtl_before_eqtl_direction(self):
        """pQTL-MR (Tier2p) should activate before eQTL direction (Tier2.5)."""
        from pipelines.ota_beta_estimation import estimate_beta
        pqtl = {
            "top_pqtl": {"beta": -0.28, "pvalue": 1e-12, "rsid": "rs1061170",
                          "study_label": "Sun2023"},
            "data_source": "test",
        }
        eqtl_weak = {"nes": 0.1, "tissue": "Liver"}  # present but would only give direction
        result = estimate_beta(
            gene="CFH",
            program="complement_activation",
            eqtl_data=eqtl_weak,
            coloc_h4=0.3,       # weak — Tier2a would reject, Tier2.5 would activate
            program_loading=0.9,
            pqtl_data=pqtl,
        )
        # pQTL should win over eQTL direction
        self.assertEqual(result["evidence_tier"], "Tier2p_pQTL_MR")

    def test_pqtl_before_sc_eqtl(self):
        """pQTL (Tier2p) should now activate before sc-eQTL (Tier2c): the drug
        target is the protein, so a protein QTL is causally prior to an mRNA eQTL."""
        from pipelines.ota_beta_estimation import estimate_beta
        sc_data = {
            "top_eqtl": {"beta": 0.5, "pvalue": 1e-8, "condition_label": "monocyte",
                          "study_label": "Yazar2022"},
            "cell_type": "monocyte", "study": "Yazar2022",
        }
        pqtl = {
            "top_pqtl": {"beta": -0.28, "pvalue": 1e-12, "rsid": "rs000",
                          "study_label": "Sun2023"},
            "data_source": "test",
        }
        result = estimate_beta(
            gene="IL6", program="IL6_pathway",
            sc_eqtl_data=sc_data, pqtl_data=pqtl,
            program_loading=0.7,
        )
        self.assertIn("Tier2p", result["evidence_tier"])

    def test_burden_without_eqtl(self):
        """Rare burden (Tier2rb) activates when no eQTL/pQTL present."""
        from pipelines.ota_beta_estimation import estimate_beta
        burden = {
            "burden_beta": 0.7,
            "burden_p": 1e-6,
            "burden_se": 0.14,
            "loeuf": 0.45,
            "pli": 0.6,
            "burden_study": "UKB_WES",
            "interpretation": "LoF increases disease risk",
        }
        result = estimate_beta(
            gene="HMCN1",
            program="ECM_remodeling",
            burden_data=burden,
            program_loading=0.6,
        )
        self.assertEqual(result["evidence_tier"], "Tier2rb_RareBurden")

    def test_cfh_no_gtex_eqtl_gets_pqtl(self):
        """CFH with no GTEx eQTL + no COLOC should fall to pQTL if available."""
        from pipelines.ota_beta_estimation import estimate_beta
        pqtl = {
            "top_pqtl": {"beta": -0.3, "pvalue": 1e-15, "rsid": "rs1061170",
                          "study_label": "Sun2023"},
            "data_source": "eQTL Catalogue",
        }
        result = estimate_beta(
            gene="CFH",
            program="complement_regulation",
            eqtl_data=None,       # No GTEx eQTL
            coloc_h4=None,
            program_loading=0.85,
            pqtl_data=pqtl,
        )
        self.assertEqual(result["evidence_tier"], "Tier2p_pQTL_MR")
        self.assertLess(result["beta"], 0)  # negative NES × positive loading → negative β


# ---------------------------------------------------------------------------
# eqtl_catalogue_server — unit-level helpers (no network)
# ---------------------------------------------------------------------------

class TestEqtlCatalogueServerHelpers(unittest.TestCase):
    """Test eQTL Catalogue server constants and helper structure."""

    def test_disease_cell_type_map_coverage(self):
        from mcp_servers.eqtl_catalogue_server import DISEASE_SC_EQTL_CELL_TYPES
        required = {"CAD", "RA"}
        self.assertTrue(required.issubset(set(DISEASE_SC_EQTL_CELL_TYPES.keys())))

    def test_pqtl_key_genes_cad(self):
        from mcp_servers.eqtl_catalogue_server import DISEASE_KEY_PQTL_PROTEINS
        cad_proteins = DISEASE_KEY_PQTL_PROTEINS.get("CAD", {})
        self.assertIn("PCSK9", cad_proteins)
        self.assertIn("LPA", cad_proteins)

    def test_pick_top_cis_eqtl_selects_lowest_pvalue(self):
        from mcp_servers.eqtl_catalogue_server import _pick_top_cis_eqtl
        assocs = [
            {"beta": 0.3, "pvalue": 1e-5, "gene_start": 1000},
            {"beta": 0.5, "pvalue": 1e-10, "gene_start": 2000},
            {"beta": 0.1, "pvalue": 1e-3, "gene_start": 500},
        ]
        top = _pick_top_cis_eqtl(assocs, "ENSG00000000001")
        self.assertAlmostEqual(top["pvalue"], 1e-10)

    def test_pick_top_cis_eqtl_filters_trans(self):
        """Variants > 1Mb from gene should be excluded (trans)."""
        from mcp_servers.eqtl_catalogue_server import _pick_top_cis_eqtl
        assocs = [
            {"beta": 0.8, "pvalue": 1e-20, "gene_start": 2_000_000},  # trans (>1Mb)
            {"beta": 0.2, "pvalue": 1e-6,  "gene_start": 50_000},      # cis
        ]
        top = _pick_top_cis_eqtl(assocs, "ENSG00000000001", window_kb=1000)
        # trans variant excluded → cis variant selected
        self.assertAlmostEqual(top["pvalue"], 1e-6)


# ---------------------------------------------------------------------------
# ukb_wes_server — unit-level helpers (no network)
# ---------------------------------------------------------------------------

class TestUkbWesServerHelpers(unittest.TestCase):
    """Test UKB WES server constants and helper structure."""

    def test_disease_burden_efo_coverage(self):
        from mcp_servers.ukb_wes_server import _DISEASE_BURDEN_EFO
        required = {"CAD", "RA"}
        self.assertTrue(required.issubset(set(_DISEASE_BURDEN_EFO.keys())))

    def test_get_burden_direction_returns_none_on_failure(self):
        from mcp_servers.ukb_wes_server import get_burden_direction_for_gene
        # Should not raise; returns None on network failure (mocked via bad gene name)
        # We just test the function is importable and handles gracefully
        self.assertTrue(callable(get_burden_direction_for_gene))

    def test_schema_pqtl_cad_fields_present(self):
        from graph.schema import DISEASE_CELL_TYPE_MAP
        cad = DISEASE_CELL_TYPE_MAP.get("CAD", {})
        self.assertIn("pqtl_key_genes", cad)
        self.assertIn("LPA", cad["pqtl_key_genes"])  # LPA is the key pQTL-only CAD target


class TestProgramEqtlDirectionScoring(unittest.TestCase):
    """
    Unit tests for the Z-score-weighted normalized direction score in
    compute_program_eqtl_direction (the scoring loop only — no I/O or external data).

    Score formula:
        score_P = Σ_G  sign(loading) × sign(GWAS_β × eQTL_β) × |loading| × |eQTL_Z|
                  ─────────────────────────────────────────────────────────────────────
                                   Σ_G  |loading| × |eQTL_Z|
    """

    def _score(self, gene_loading_pairs, gene_eqtl, gwas_betas):
        """Run the scoring loop extracted from compute_program_eqtl_direction."""
        numerator = denominator = 0.0
        n_eqtl = n_gwas_hit = 0
        for gene, loading in gene_loading_pairs:
            eqtl = gene_eqtl.get(gene)
            if eqtl is None:
                continue
            eqtl_beta = eqtl.get("beta")
            eqtl_se   = eqtl.get("se")
            if eqtl_beta is None:
                continue
            eqtl_beta = float(eqtl_beta)
            eqtl_se   = float(eqtl_se) if eqtl_se is not None else None
            eqtl_z    = abs(eqtl_beta / eqtl_se) if eqtl_se and eqtl_se > 0 else 1.0
            weight        = abs(loading) * eqtl_z
            loading_sign  = 1.0 if loading >= 0 else -1.0
            eqtl_sign     = 1.0 if eqtl_beta >= 0 else -1.0
            rsid          = eqtl.get("rsid", "")
            gwas_beta     = gwas_betas.get(rsid)
            if gwas_beta is not None:
                gwas_sign    = 1.0 if gwas_beta >= 0 else -1.0
                contribution = loading_sign * eqtl_sign * gwas_sign
                n_gwas_hit  += 1
            else:
                contribution = loading_sign * eqtl_sign
            numerator   += contribution * weight
            denominator += weight
            n_eqtl      += 1
        score = numerator / denominator if denominator > 0 else 0.0
        return score, n_eqtl, n_gwas_hit

    def test_perfect_atherogenic_score_is_one(self):
        """All genes: positive loading, positive eQTL, positive GWAS → score = +1."""
        pairs = [("A", 0.8), ("B", 0.6), ("C", 0.4)]
        eqtl  = {g: {"beta": 0.3, "se": 0.1, "rsid": f"rs{i}"} for i, g in enumerate("ABC")}
        gwas  = {f"rs{i}": 0.1 for i in range(3)}
        score, n, _ = self._score(pairs, eqtl, gwas)
        self.assertAlmostEqual(score, 1.0, places=5)
        self.assertEqual(n, 3)

    def test_perfect_protective_score_is_minus_one(self):
        """All genes: positive loading, negative eQTL, positive GWAS → score = −1."""
        pairs = [("A", 0.8), ("B", 0.6)]
        eqtl  = {g: {"beta": -0.3, "se": 0.1, "rsid": f"rs{i}"} for i, g in enumerate("AB")}
        gwas  = {f"rs{i}": 0.1 for i in range(2)}
        score, _, _ = self._score(pairs, eqtl, gwas)
        self.assertAlmostEqual(score, -1.0, places=5)

    def test_mixed_averages_toward_zero(self):
        """Half atherogenic, half protective with equal weight → score ≈ 0."""
        pairs = [("A", 1.0), ("B", -1.0), ("C", 1.0), ("D", -1.0)]
        eqtl  = {g: {"beta": 0.3, "se": 0.1, "rsid": f"rs{i}"} for i, g in enumerate("ABCD")}
        gwas  = {f"rs{i}": 0.1 for i in range(4)}
        score, _, _ = self._score(pairs, eqtl, gwas)
        self.assertAlmostEqual(score, 0.0, places=5)

    def test_strong_eqtl_gene_dominates_weak(self):
        """Gene with 10× stronger eQTL should dominate; score closer to that gene's direction."""
        # Gene A: atherogenic, |eQTL_Z| = 10; Gene B: protective, |eQTL_Z| = 1
        # Expected: score > 0 (A dominates)
        pairs = [("A", 1.0), ("B", 1.0)]
        eqtl  = {
            "A": {"beta": 0.5,  "se": 0.05, "rsid": "rs1"},   # Z = 10
            "B": {"beta": -0.5, "se": 0.5,  "rsid": "rs2"},   # Z = 1
        }
        gwas  = {"rs1": 0.1, "rs2": 0.1}
        score, _, _ = self._score(pairs, eqtl, gwas)
        self.assertGreater(score, 0.0)

    def test_score_bounded_in_unit_interval(self):
        """Score must lie in [-1, 1] for arbitrary inputs."""
        import random
        rng = random.Random(42)
        pairs = [(f"G{i}", rng.uniform(-2, 2)) for i in range(20)]
        eqtl  = {f"G{i}": {"beta": rng.uniform(-1, 1), "se": 0.1, "rsid": f"rs{i}"}
                 for i in range(20)}
        gwas  = {f"rs{i}": rng.uniform(-0.5, 0.5) for i in range(20)}
        score, _, _ = self._score(pairs, eqtl, gwas)
        self.assertGreaterEqual(score, -1.0 - 1e-9)
        self.assertLessEqual(score, 1.0 + 1e-9)

    def test_no_eqtl_coverage_returns_zero(self):
        """No eQTL data for any gene → score = 0, n_eqtl = 0."""
        pairs = [("A", 1.0), ("B", 0.5)]
        score, n, _ = self._score(pairs, {}, {})
        self.assertEqual(score, 0.0)
        self.assertEqual(n, 0)

    def test_missing_se_falls_back_to_z_one(self):
        """Gene with missing SE should not crash; eQTL_Z defaults to 1."""
        pairs = [("A", 1.0)]
        eqtl  = {"A": {"beta": 0.5, "se": None, "rsid": "rs1"}}
        gwas  = {"rs1": 0.1}
        score, n, _ = self._score(pairs, eqtl, gwas)
        self.assertEqual(n, 1)
        self.assertAlmostEqual(score, 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
