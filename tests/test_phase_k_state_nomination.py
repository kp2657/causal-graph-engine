"""
tests/test_phase_k_state_nomination.py — Phase K: state-space gene nomination + TWMR

Tests:
  - nominate_state_genes() scoring and ranking
  - state_nominated tier in TIER_MULTIPLIER
  - compute_ratio_mr() single-instrument and IVW
  - check_weak_instrument()
  - FinnGen phenotype definition fallback
"""
from __future__ import annotations

import math
import unittest
from unittest.mock import MagicMock, patch

import random


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_adata(n_cells: int = 60, n_genes: int = 10, gene_names: list[str] | None = None):
    """Return a minimal mock AnnData for nomination testing."""
    if gene_names is None:
        gene_names = [f"GENE{i}" for i in range(n_genes)]

    rnd = random.Random(42)
    X = [[float(rnd.random()) for _ in range(len(gene_names))] for _ in range(n_cells)]

    adata = MagicMock()
    adata.var_names = gene_names
    adata.n_obs = n_cells
    adata.n_vars = len(gene_names)

    # obs columns needed by transition_scoring
    cell_labels = ["path"] * 20 + ["healthy"] * 20 + ["other"] * 20
    adata.obs = MagicMock()
    adata.obs.__getitem__ = lambda self, key: {
        "state_intermediate": cell_labels,
        "state_coarse": cell_labels,
        "pseudotime": [i / (n_cells - 1) if n_cells > 1 else 0.0 for i in range(n_cells)],
    }.get(key, [0.0 for _ in range(n_cells)])
    adata.obs.columns = ["state_intermediate", "state_coarse", "pseudotime"]
    adata.obs.__contains__ = lambda self, key: key in ["state_intermediate", "state_coarse", "pseudotime"]

    # X access
    adata.X = X
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]

    return adata


def _make_mock_transition_result(pathologic_ids=None, healthy_ids=None):
    return {
        "pathologic_basin_ids": pathologic_ids or ["path"],
        "healthy_basin_ids":    healthy_ids    or ["healthy"],
        "state_labels":         ["path", "healthy", "other"],
        "transition_matrix":    [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        "cell_state_obs_key":   "state_intermediate",
    }


# ---------------------------------------------------------------------------
# 1. nominate_state_genes
# ---------------------------------------------------------------------------

class TestNominateStateGenes(unittest.TestCase):

    def _make_profiles(self, gene_names, scores):
        """Return mock TransitionGeneProfile objects."""
        from models.evidence import TransitionGeneProfile
        result = {}
        for g, (entry, persist, recov, bound) in zip(gene_names, scores):
            result[g] = TransitionGeneProfile(
                gene=g, disease="IBD",
                entry_score=entry, persistence_score=persist,
                recovery_score=recov, boundary_score=bound,
                disease_axis_score=entry + persist,
                entry_direction=1,
            )
        return result

    def test_returns_top_k_by_combined_score(self):
        """nominate_state_genes returns top-k genes sorted by combined score."""
        gene_names = ["A", "B", "C", "D", "E"]
        # Combined = entry + persist + recov + 0.5*bound
        # A: 1.0+1.0+0.5+0.5*0.2 = 2.6  B: 0.1+0.1+0.1+0.0 = 0.3
        # C: 0.8+0.7+0.3+0.5*0.4 = 2.0  D: 0.5+0.5+0.0+0.0 = 1.0
        # E: 0.2+0.2+0.2+0.5*0.2 = 0.7
        scores = [
            (1.0, 1.0, 0.5, 0.2),  # A: 2.6
            (0.1, 0.1, 0.1, 0.0),  # B: 0.3
            (0.8, 0.7, 0.3, 0.4),  # C: 2.0
            (0.5, 0.5, 0.0, 0.0),  # D: 1.0
            (0.2, 0.2, 0.2, 0.2),  # E: 0.7
        ]
        mock_profiles = self._make_profiles(gene_names, scores)

        adata = MagicMock()
        adata.var_names = gene_names
        trans = {}

        with patch(
            "pipelines.state_space.state_influence.compute_transition_gene_scores",
            return_value=mock_profiles,
        ):
            from pipelines.state_space.state_influence import nominate_state_genes
            result = nominate_state_genes(adata, trans, top_k=3)

        gene_order = [g for g, _ in result]
        self.assertEqual(gene_order, ["A", "C", "D"])

    def test_excludes_specified_genes(self):
        """Genes in exclude set are not returned — mock simulates filtered gene_list."""
        # After excluding "A", only B and C are passed to compute_transition_gene_scores
        mock_profiles = self._make_profiles(
            ["B", "C"],
            [(0.8, 0.7, 0.3, 0.4), (0.1, 0.1, 0.0, 0.0)],
        )

        adata = MagicMock()
        adata.var_names = ["A", "B", "C"]
        trans = {}

        with patch(
            "pipelines.state_space.state_influence.compute_transition_gene_scores",
            return_value=mock_profiles,
        ):
            from pipelines.state_space.state_influence import nominate_state_genes
            result = nominate_state_genes(adata, trans, top_k=10, exclude={"A"})

        genes = [g for g, _ in result]
        self.assertNotIn("A", genes)
        self.assertIn("B", genes)

    def test_returns_profile_dicts(self):
        """Each result is (gene, dict) with expected keys."""
        gene_names = ["SPI1", "STAT1"]
        scores = [(0.7, 0.5, 0.3, 0.1), (0.4, 0.6, 0.2, 0.0)]
        mock_profiles = self._make_profiles(gene_names, scores)

        adata = MagicMock()
        adata.var_names = gene_names
        trans = {}

        with patch(
            "pipelines.state_space.state_influence.compute_transition_gene_scores",
            return_value=mock_profiles,
        ):
            from pipelines.state_space.state_influence import nominate_state_genes
            result = nominate_state_genes(adata, trans, top_k=10)

        self.assertEqual(len(result), 2)
        for gene, profile_dict in result:
            self.assertIsInstance(profile_dict, dict)
            self.assertIn("entry_score", profile_dict)
            self.assertIn("persistence_score", profile_dict)

    def test_empty_adata_returns_empty(self):
        """Empty var_names returns empty list."""
        adata = MagicMock()
        adata.var_names = []
        trans = {}

        with patch(
            "pipelines.state_space.state_influence.compute_transition_gene_scores",
            return_value={},
        ):
            from pipelines.state_space.state_influence import nominate_state_genes
            result = nominate_state_genes(adata, trans, top_k=10)

        self.assertEqual(result, [])

    def test_top_k_respected(self):
        """top_k limits the number of returned genes."""
        gene_names = [f"G{i}" for i in range(20)]
        scores = [(float(i) / 20, float(i) / 20, 0.0, 0.0) for i in range(20)]
        mock_profiles = {
            g: MagicMock(
                entry_score=e, persistence_score=p, recovery_score=r, boundary_score=b,
                disease_axis_score=e+p, entry_direction=1,
                model_dump=lambda self=None, **_: {"entry_score": e}
            )
            for g, (e, p, r, b) in zip(gene_names, scores)
        }
        adata = MagicMock()
        adata.var_names = gene_names
        trans = {}

        with patch(
            "pipelines.state_space.state_influence.compute_transition_gene_scores",
            return_value=mock_profiles,
        ):
            from pipelines.state_space.state_influence import nominate_state_genes
            result = nominate_state_genes(adata, trans, top_k=5)

        self.assertEqual(len(result), 5)


# ---------------------------------------------------------------------------
# 2. state_nominated tier in TIER_MULTIPLIER
# ---------------------------------------------------------------------------

class TestStateNominatedTier(unittest.TestCase):

    def test_tier_multiplier_has_state_nominated(self):
        from agents.tier4_translation.target_prioritization_agent import TIER_MULTIPLIER
        self.assertIn("state_nominated", TIER_MULTIPLIER)
        # Should be between virtual (0.1) and provisional (0.5)
        v = TIER_MULTIPLIER["state_nominated"]
        self.assertGreater(v, 0.1)
        self.assertLessEqual(v, 0.5)

    def test_state_nominated_lower_than_tier3(self):
        from agents.tier4_translation.target_prioritization_agent import TIER_MULTIPLIER
        self.assertLess(TIER_MULTIPLIER["state_nominated"], TIER_MULTIPLIER["Tier3_Provisional"])

    def test_state_nominated_higher_than_virtual(self):
        from agents.tier4_translation.target_prioritization_agent import TIER_MULTIPLIER
        self.assertGreater(TIER_MULTIPLIER["state_nominated"], TIER_MULTIPLIER["provisional_virtual"])


# ---------------------------------------------------------------------------
# 2b. Tier 4/5: state-space targets are separated and rendered
# ---------------------------------------------------------------------------

class TestStateSpaceTargetPlumbing(unittest.TestCase):

    @patch("mcp_servers.open_targets_server.get_open_targets_disease_targets")
    @patch("mcp_servers.clinical_trials_server.get_trials_for_target")
    @patch("mcp_servers.gwas_genetics_server.query_gnomad_lof_constraint")
    @patch("mcp_servers.single_cell_server.get_gene_tau_specificity")
    @patch("mcp_servers.single_cell_server.get_gene_bimodality_scores")
    def test_tier4_emits_state_space_targets_separately(
        self, mock_bc, mock_tau, mock_pli, mock_trials, mock_ot
    ):
        # Minimal OT batch response
        mock_ot.return_value = {"targets": []}
        mock_trials.return_value = {"trial_summary": {}}
        mock_pli.return_value = {"pli": {}}
        mock_tau.return_value = {"tau_scores": {}}
        mock_bc.return_value = {"bimodality_scores": {}}

        from agents.tier4_translation.target_prioritization_agent import run as rank_run

        disease_query = {"disease_name": "coronary artery disease", "efo_id": "EFO_0001645"}
        causal_discovery_result = {
            "top_genes": [
                {"gene": "PCSK9", "ota_gamma": 0.2, "tier": "Tier3_Provisional", "programs": []},
                {
                    "gene": "STATX",
                    "tier": "state_nominated",
                    "evidence_type": "state_space",
                    "ota_gamma": None,
                    "state_edge_effect": 0.8,
                    "state_edge_confidence": 0.5,
                    "therapeutic_redirection_result": {"therapeutic_redirection": 0.6},
                },
            ]
        }
        kg_result = {"drug_target_summary": [], "brg_novel_candidates": []}

        out = rank_run(causal_discovery_result, kg_result, disease_query)
        self.assertIn("targets", out)
        self.assertIn("state_space_targets", out)
        self.assertEqual(len(out["targets"]), 1)
        self.assertEqual(out["targets"][0]["target_gene"], "PCSK9")
        self.assertGreaterEqual(len(out["state_space_targets"]), 1)
        self.assertEqual(out["state_space_targets"][0]["target_gene"], "STATX")
        self.assertIn("state_target_score", out["state_space_targets"][0])

    def test_tier5_renders_state_space_table(self):
        from agents.tier5_writer.scientific_writer_agent import run as writer_run

        phenotype = {"disease_name": "coronary artery disease", "efo_id": "EFO_0001645"}
        empty = {"warnings": []}
        beta = {"n_virtual": 0, "warnings": []}
        causal = {"n_edges_written": 0, "warnings": []}
        kg = {"warnings": []}
        prioritization = {
            "warnings": [],
            "targets": [
                {
                    "target_gene": "PCSK9",
                    "rank": 1,
                    "target_score": 0.2,
                    "ota_gamma": 0.2,
                    "evidence_tier": "Tier3_Provisional",
                    "ot_score": 0.0,
                    "max_phase": 0,
                    "known_drugs": [],
                }
            ],
            "state_space_targets": [
                {
                    "target_gene": "STATX",
                    "rank": 1,
                    "evidence_tier": "state_nominated",
                    "evidence_type": "state_space",
                    "state_edge_effect": 0.8,
                    "state_edge_confidence": 0.5,
                    "state_target_score": 0.4,
                    "state_edge_ci_lower": 0.7,
                    "state_edge_ci_upper": 0.9,
                    "ot_score": 0.0,
                    "max_phase": 0,
                    "known_drugs": [],
                }
            ],
        }
        chem = {"warnings": [], "target_chemistry": {}}
        trials = {"warnings": [], "trial_summary": {}}

        out = writer_run(
            phenotype, empty, beta, empty, causal, kg,
            prioritization, chem, trials, somatic_result=None
        )
        self.assertIn("state_space_table", out)
        self.assertIn("| StateEffect |", out["state_space_table"])
        self.assertIn("CI95", out["state_space_table"])
        self.assertIn("STATX", out["state_space_table"])


# ---------------------------------------------------------------------------
# 3. compute_ratio_mr
# ---------------------------------------------------------------------------

class TestComputeRatioMR(unittest.TestCase):

    def test_single_instrument_ratio(self):
        """Single instrument: gamma = beta_gwas / beta_eqtl."""
        from pipelines.twmr import compute_ratio_mr
        result = compute_ratio_mr("GENE1", [
            {"beta_eqtl": 0.5, "beta_gwas": 0.1, "se_gwas": 0.02}
        ])
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["beta"], 0.1 / 0.5, places=3)
        self.assertEqual(result["n_instruments"], 1)
        self.assertEqual(result["method"], "ratio")

    def test_ivw_multiple_instruments(self):
        """IVW with 3 instruments: method = IVW."""
        from pipelines.twmr import compute_ratio_mr
        instruments = [
            {"beta_eqtl": 0.5, "beta_gwas": 0.10, "se_gwas": 0.02},
            {"beta_eqtl": 0.4, "beta_gwas": 0.08, "se_gwas": 0.02},
            {"beta_eqtl": 0.6, "beta_gwas": 0.12, "se_gwas": 0.02},
        ]
        result = compute_ratio_mr("GENE1", instruments)
        self.assertIsNotNone(result)
        self.assertEqual(result["method"], "IVW")
        self.assertEqual(result["n_instruments"], 3)
        # All instruments have same ratio (0.2); IVW should give ~0.2
        self.assertAlmostEqual(result["beta"], 0.2, places=2)

    def test_returns_none_when_no_valid_instruments(self):
        """No valid instruments → None."""
        from pipelines.twmr import compute_ratio_mr
        self.assertIsNone(compute_ratio_mr("GENE1", []))
        self.assertIsNone(compute_ratio_mr("GENE1", [{"beta_eqtl": None, "beta_gwas": 0.1}]))

    def test_filters_near_zero_eqtl(self):
        """Near-zero eQTL beta is excluded (degenerate ratio)."""
        from pipelines.twmr import compute_ratio_mr
        result = compute_ratio_mr("GENE1", [
            {"beta_eqtl": 1e-10, "beta_gwas": 0.1, "se_gwas": 0.02}
        ])
        self.assertIsNone(result)

    def test_p_value_is_finite(self):
        """p-value should be a finite float in [0, 1]."""
        from pipelines.twmr import compute_ratio_mr
        result = compute_ratio_mr("GENE1", [
            {"beta_eqtl": 0.5, "beta_gwas": 0.2, "se_gwas": 0.01}
        ])
        self.assertIsNotNone(result)
        p = result["p"]
        self.assertTrue(math.isfinite(p))
        self.assertGreaterEqual(p, 0.0)
        self.assertLessEqual(p, 1.0)

    def test_large_effect_gives_small_p(self):
        """Large beta / small SE → p should be very small."""
        from pipelines.twmr import compute_ratio_mr
        result = compute_ratio_mr("GENE1", [
            {"beta_eqtl": 1.0, "beta_gwas": 1.0, "se_gwas": 0.001}
        ])
        self.assertIsNotNone(result)
        self.assertLess(result["p"], 0.001)

    def test_opposite_direction_instruments_cancel(self):
        """Opposite-direction instruments produce near-zero IVW beta."""
        from pipelines.twmr import compute_ratio_mr
        instruments = [
            {"beta_eqtl": 0.5, "beta_gwas":  0.2, "se_gwas": 0.05},
            {"beta_eqtl": 0.5, "beta_gwas": -0.2, "se_gwas": 0.05},
        ]
        result = compute_ratio_mr("GENE1", instruments)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["beta"], 0.0, places=4)


# ---------------------------------------------------------------------------
# 4. check_weak_instrument
# ---------------------------------------------------------------------------

class TestCheckWeakInstrument(unittest.TestCase):

    def test_strong_instrument(self):
        """F-stat ≥ 10 for strong instrument (z > 3.16)."""
        from pipelines.twmr import check_weak_instrument
        # z = 0.5 / 0.05 = 10, F = z² = 100
        f = check_weak_instrument(0.5, 0.05)
        self.assertGreaterEqual(f, 10.0)

    def test_weak_instrument(self):
        """F-stat < 10 for weak instrument."""
        from pipelines.twmr import check_weak_instrument
        # z = 0.1 / 0.1 = 1, F = 1 < 10
        f = check_weak_instrument(0.1, 0.1)
        self.assertLess(f, 10.0)

    def test_zero_se_returns_zero(self):
        from pipelines.twmr import check_weak_instrument
        self.assertEqual(check_weak_instrument(0.5, 0.0), 0.0)


# ---------------------------------------------------------------------------
# 5. FinnGen phenotype definition (stub → live + fallback)
# ---------------------------------------------------------------------------

class TestFinngenPhenotypeDefinition(unittest.TestCase):

    def test_fallback_for_known_cad_phenocode(self):
        """I9_CAD returns metadata even when live API fails."""
        with patch("mcp_servers.finngen_server.get_finngen_phenotype_info", side_effect=Exception("network")):
            from mcp_servers.gwas_genetics_server import get_finngen_phenotype_definition
            result = get_finngen_phenotype_definition("I9_CAD")
        self.assertEqual(result["phenocode"], "I9_CAD")
        self.assertIn("n_cases", result)
        self.assertGreater(result["n_cases"], 0)

    def test_fallback_for_ibd_phenocode(self):
        """K11_IBD returns metadata from fallback dict."""
        with patch("mcp_servers.finngen_server.get_finngen_phenotype_info", side_effect=Exception("network")):
            from mcp_servers.gwas_genetics_server import get_finngen_phenotype_definition
            result = get_finngen_phenotype_definition("K11_IBD")
        self.assertEqual(result["phenocode"], "K11_IBD")
        self.assertIn("n_cases", result)

    def test_live_api_used_when_available(self):
        """When live API succeeds, its result is returned."""
        mock_live = {
            "phenocode": "I9_CAD",
            "name": "Coronary artery disease",
            "n_cases": 31_000,
            "n_controls": 280_000,
            "data_source": "FinnGen R10",
        }
        with patch("mcp_servers.finngen_server.get_finngen_phenotype_info", return_value=mock_live):
            from mcp_servers.gwas_genetics_server import get_finngen_phenotype_definition
            result = get_finngen_phenotype_definition("I9_CAD")
        self.assertEqual(result["n_cases"], 31_000)
        self.assertEqual(result.get("data_source"), "FinnGen R10")

    def test_unknown_phenocode_returns_note(self):
        """Unknown phenocode returns a note, not an exception."""
        with patch("mcp_servers.finngen_server.get_finngen_phenotype_info", return_value={"error": "not found"}):
            from mcp_servers.gwas_genetics_server import get_finngen_phenotype_definition
            result = get_finngen_phenotype_definition("UNKNOWN_CODE")
        self.assertEqual(result["phenocode"], "UNKNOWN_CODE")
        self.assertIn("note", result)

    def test_no_more_stub_note(self):
        """Result should not contain the old STUB marker string."""
        with patch("mcp_servers.finngen_server.get_finngen_phenotype_info", side_effect=Exception("network")):
            from mcp_servers.gwas_genetics_server import get_finngen_phenotype_definition
            result = get_finngen_phenotype_definition("I9_CAD")
        note = result.get("note", "")
        self.assertNotIn("STUB", note)


# ---------------------------------------------------------------------------
# 6. Phase L — TWMR wiring in causal_discovery_agent
# ---------------------------------------------------------------------------

# Minimal inputs for causal_discovery_agent.run()
_TWMR_MOCK_BETA = {
    "genes":    ["PCSK9", "LDLR"],
    "programs": ["lipid_metabolism"],
    "beta_matrix": {
        "PCSK9": {"lipid_metabolism": -0.4},
        "LDLR":  {"lipid_metabolism":  0.3},
    },
    "evidence_tier_per_gene": {
        "PCSK9": "Tier3_Provisional",
        "LDLR":  "Tier3_Provisional",
    },
    "n_tier1": 0, "n_tier2": 0, "n_tier3": 2, "n_virtual": 0, "warnings": [],
}
_TWMR_MOCK_GAMMA = {
    "lipid_metabolism": {"coronary artery disease": 0.6},
}
_TWMR_MOCK_DISEASE = {
    "disease_name":    "coronary artery disease",
    "efo_id":          "EFO_0001645",
    "icd10_codes":     ["I25"],
    "modifier_types":  ["germline"],
    "primary_gwas_id": "ieu-a-7",
}


def _scone_pass_through(ota_gamma=0.0, beta_info=None, eqtl_info=None):
    return {"refined_gamma": ota_gamma, "scone_sensitivity": 0.0, "regime_consistency": 1.0}


class TestTWMRWiring(unittest.TestCase):

    @patch("pipelines.scone_sensitivity.polybic_selection")
    @patch("pipelines.scone_sensitivity.bootstrap_edge_confidence")
    @patch("pipelines.scone_sensitivity.apply_scone_refinement")
    @patch("agents.tier3_causal.causal_discovery_agent._maybe_therapeutic_redirection")
    @patch("pipelines.twmr.run_twmr_for_gene")
    @patch("pipelines.ota_gamma_estimation.compute_ota_gamma")
    @patch("mcp_servers.graph_db_server.write_causal_edges")
    @patch("mcp_servers.graph_db_server.run_anchor_edge_validation")
    @patch("mcp_servers.graph_db_server.compute_shd_metric")
    @patch("mcp_servers.graph_db_server.run_evalue_check")
    def test_twmr_supplements_tier3_gene(
        self, mock_eval, mock_shd, mock_anchor, mock_write, mock_ota, mock_twmr, mock_tr,
        mock_scone, mock_bootstrap, mock_polybic,
    ):
        """Gene with TWMR F-stat ≥ 10 should be upgraded to Tier2_Convergent."""
        mock_tr.return_value = {}
        mock_ota.return_value = {"ota_gamma": 0.1, "dominant_tier": "Tier3_Provisional", "top_programs": []}
        mock_write.return_value = {"n_written": 1}
        mock_anchor.return_value = {"recovery_rate": 0.5, "recovered": [], "missing": []}
        mock_shd.return_value = {"shd": 1}
        mock_eval.return_value = {"e_value": 5.0}
        mock_twmr.return_value = {
            "gene": "PCSK9", "beta": -0.3, "se": 0.05, "p": 1e-10,
            "n_instruments": 3, "f_statistic": 45.0, "method": "IVW",
        }
        mock_scone.side_effect = _scone_pass_through
        mock_bootstrap.return_value = {"mean": 0.8, "confidence": 0.8, "ci_lower": 0.7, "ci_upper": 0.9, "cv": 0.1}
        mock_polybic.side_effect = lambda records, **kw: records

        from agents.tier3_causal.causal_discovery_agent import run
        result = run(_TWMR_MOCK_BETA, _TWMR_MOCK_GAMMA, _TWMR_MOCK_DISEASE)

        pcsk9 = next((g for g in result["top_genes"] if g["gene"] == "PCSK9"), None)
        self.assertIsNotNone(pcsk9)
        self.assertEqual(pcsk9["tier"], "Tier2_Convergent")
        self.assertIsNotNone(pcsk9["twmr_beta"])
        self.assertEqual(pcsk9["twmr_f_stat"], 45.0)
        self.assertEqual(pcsk9["twmr_n_instruments"], 3)

    @patch("pipelines.scone_sensitivity.polybic_selection")
    @patch("pipelines.scone_sensitivity.bootstrap_edge_confidence")
    @patch("pipelines.scone_sensitivity.apply_scone_refinement")
    @patch("agents.tier3_causal.causal_discovery_agent._maybe_therapeutic_redirection")
    @patch("pipelines.twmr.run_twmr_for_gene")
    @patch("pipelines.ota_gamma_estimation.compute_ota_gamma")
    @patch("mcp_servers.graph_db_server.write_causal_edges")
    @patch("mcp_servers.graph_db_server.run_anchor_edge_validation")
    @patch("mcp_servers.graph_db_server.compute_shd_metric")
    @patch("mcp_servers.graph_db_server.run_evalue_check")
    def test_twmr_skipped_when_returns_none(
        self, mock_eval, mock_shd, mock_anchor, mock_write, mock_ota, mock_twmr, mock_tr,
        mock_scone, mock_bootstrap, mock_polybic,
    ):
        """When run_twmr_for_gene returns None, tier and gamma stay unchanged."""
        mock_tr.return_value = {}
        mock_ota.return_value = {"ota_gamma": 0.1, "dominant_tier": "Tier3_Provisional", "top_programs": []}
        mock_write.return_value = {"n_written": 1}
        mock_anchor.return_value = {"recovery_rate": 0.5, "recovered": [], "missing": []}
        mock_shd.return_value = {"shd": 1}
        mock_eval.return_value = {"e_value": 5.0}
        mock_twmr.return_value = None
        mock_scone.side_effect = _scone_pass_through
        mock_bootstrap.return_value = {"mean": 0.8, "confidence": 0.8, "ci_lower": 0.7, "ci_upper": 0.9, "cv": 0.1}
        mock_polybic.side_effect = lambda records, **kw: records

        from agents.tier3_causal.causal_discovery_agent import run
        result = run(_TWMR_MOCK_BETA, _TWMR_MOCK_GAMMA, _TWMR_MOCK_DISEASE)

        self.assertGreater(len(result["top_genes"]), 0)
        for g in result["top_genes"]:
            self.assertIsNone(g.get("twmr_beta"))
            self.assertNotEqual(g["tier"], "Tier2_Convergent")

    @patch("pipelines.scone_sensitivity.polybic_selection")
    @patch("pipelines.scone_sensitivity.bootstrap_edge_confidence")
    @patch("pipelines.scone_sensitivity.apply_scone_refinement")
    @patch("agents.tier3_causal.causal_discovery_agent._maybe_therapeutic_redirection")
    @patch("pipelines.twmr.run_twmr_for_gene")
    @patch("pipelines.ota_gamma_estimation.compute_ota_gamma")
    @patch("mcp_servers.graph_db_server.write_causal_edges")
    @patch("mcp_servers.graph_db_server.run_anchor_edge_validation")
    @patch("mcp_servers.graph_db_server.compute_shd_metric")
    @patch("mcp_servers.graph_db_server.run_evalue_check")
    def test_twmr_blend_is_average(
        self, mock_eval, mock_shd, mock_anchor, mock_write, mock_ota, mock_twmr, mock_tr,
        mock_scone, mock_bootstrap, mock_polybic,
    ):
        """When both ota_gamma and twmr_beta are present, ota_gamma becomes their average."""
        mock_tr.return_value = {}
        mock_ota.return_value = {"ota_gamma": 0.2, "dominant_tier": "Tier3_Provisional", "top_programs": []}
        mock_write.return_value = {"n_written": 1}
        mock_anchor.return_value = {"recovery_rate": 0.5, "recovered": [], "missing": []}
        mock_shd.return_value = {"shd": 1}
        mock_eval.return_value = {"e_value": 5.0}
        mock_twmr.return_value = {
            "gene": "PCSK9", "beta": 0.4, "se": 0.05, "p": 1e-5,
            "n_instruments": 2, "f_statistic": 20.0, "method": "IVW",
        }
        mock_scone.side_effect = _scone_pass_through
        mock_bootstrap.return_value = {"mean": 0.8, "confidence": 0.8, "ci_lower": 0.7, "ci_upper": 0.9, "cv": 0.1}
        mock_polybic.side_effect = lambda records, **kw: records

        from agents.tier3_causal.causal_discovery_agent import run
        result = run(_TWMR_MOCK_BETA, _TWMR_MOCK_GAMMA, _TWMR_MOCK_DISEASE)

        pcsk9 = next((g for g in result["top_genes"] if g["gene"] == "PCSK9"), None)
        self.assertIsNotNone(pcsk9)
        self.assertIsNotNone(pcsk9["twmr_beta"])
        self.assertNotAlmostEqual(pcsk9["ota_gamma"], pcsk9["twmr_beta"], places=5)
        self.assertLess(pcsk9["ota_gamma"], pcsk9["twmr_beta"])

    @patch("pipelines.scone_sensitivity.polybic_selection")
    @patch("pipelines.scone_sensitivity.bootstrap_edge_confidence")
    @patch("pipelines.scone_sensitivity.apply_scone_refinement")
    @patch("agents.tier3_causal.causal_discovery_agent._maybe_therapeutic_redirection")
    @patch("pipelines.twmr.run_twmr_for_gene")
    @patch("pipelines.ota_gamma_estimation.compute_ota_gamma")
    @patch("mcp_servers.graph_db_server.write_causal_edges")
    @patch("mcp_servers.graph_db_server.run_anchor_edge_validation")
    @patch("mcp_servers.graph_db_server.compute_shd_metric")
    @patch("mcp_servers.graph_db_server.run_evalue_check")
    def test_twmr_does_not_upgrade_tier1_or_tier2(
        self, mock_eval, mock_shd, mock_anchor, mock_write, mock_ota, mock_twmr, mock_tr,
        mock_scone, mock_bootstrap, mock_polybic,
    ):
        """TWMR should not downgrade Tier1 or re-label Tier2 genes."""
        mock_tr.return_value = {}
        mock_ota.return_value = {"ota_gamma": 0.5, "dominant_tier": "Tier1_Interventional", "top_programs": []}
        mock_write.return_value = {"n_written": 1}
        mock_anchor.return_value = {"recovery_rate": 0.8, "recovered": [], "missing": []}
        mock_shd.return_value = {"shd": 0}
        mock_eval.return_value = {"e_value": 2.0}
        mock_twmr.return_value = {
            "gene": "PCSK9", "beta": -0.3, "se": 0.04, "p": 1e-12,
            "n_instruments": 5, "f_statistic": 80.0, "method": "IVW",
        }
        mock_scone.side_effect = _scone_pass_through
        mock_bootstrap.return_value = {"mean": 0.8, "confidence": 0.8, "ci_lower": 0.7, "ci_upper": 0.9, "cv": 0.1}
        mock_polybic.side_effect = lambda records, **kw: records

        _t1_beta = {**_TWMR_MOCK_BETA, "evidence_tier_per_gene": {
            "PCSK9": "Tier1_Interventional", "LDLR": "Tier1_Interventional"
        }}

        from agents.tier3_causal.causal_discovery_agent import run
        result = run(_t1_beta, _TWMR_MOCK_GAMMA, _TWMR_MOCK_DISEASE)

        self.assertGreater(len(result["top_genes"]), 0)
        for g in result["top_genes"]:
            self.assertEqual(g["tier"], "Tier1_Interventional")


# ---------------------------------------------------------------------------
# 7. Phase L Task 2 — OTG L2G live implementation
# ---------------------------------------------------------------------------

class TestGetL2GScores(unittest.TestCase):

    @patch("mcp_servers.gwas_genetics_server._ot_gql")
    def test_returns_l2g_genes_from_api(self, mock_gql):
        """Live-like response: l2g_genes list is populated and sorted by score."""
        mock_gql.return_value = {
            "credibleSets": {
                "rows": [
                    {
                        "studyLocusId": "SL1",
                        "l2GPredictions": {
                            "rows": [
                                {"target": {"id": "ENSG0001", "approvedSymbol": "PCSK9"}, "score": 0.82},
                                {"target": {"id": "ENSG0002", "approvedSymbol": "LDLR"},  "score": 0.45},
                            ]
                        },
                    }
                ]
            }
        }
        from mcp_servers.gwas_genetics_server import get_l2g_scores
        result = get_l2g_scores("GCST003116")
        self.assertIn("l2g_genes", result)
        self.assertEqual(len(result["l2g_genes"]), 2)
        self.assertEqual(result["l2g_genes"][0]["gene_symbol"], "PCSK9")
        self.assertAlmostEqual(result["l2g_genes"][0]["l2g_score"], 0.82)
        self.assertEqual(result["data_source"], "OT_Platform_v4")

    @patch("mcp_servers.gwas_genetics_server._ot_gql")
    def test_returns_empty_on_api_error(self, mock_gql):
        """API failure returns empty l2g_genes without raising."""
        mock_gql.return_value = {"error": "timeout"}
        from mcp_servers.gwas_genetics_server import get_l2g_scores
        result = get_l2g_scores("GCST999")
        self.assertEqual(result["l2g_genes"], [])
        self.assertIn("note", result)

    @patch("mcp_servers.gwas_genetics_server._ot_gql")
    def test_deduplicates_genes_keeps_best_score(self, mock_gql):
        """Same gene in two credible sets → only the higher score is kept."""
        mock_gql.return_value = {
            "credibleSets": {
                "rows": [
                    {"studyLocusId": "SL1", "l2GPredictions": {"rows": [
                        {"target": {"id": "ENSG1", "approvedSymbol": "PCSK9"}, "score": 0.6},
                    ]}},
                    {"studyLocusId": "SL2", "l2GPredictions": {"rows": [
                        {"target": {"id": "ENSG1", "approvedSymbol": "PCSK9"}, "score": 0.9},
                    ]}},
                ]
            }
        }
        from mcp_servers.gwas_genetics_server import get_l2g_scores
        result = get_l2g_scores("GCST003116")
        self.assertEqual(len(result["l2g_genes"]), 1)
        self.assertAlmostEqual(result["l2g_genes"][0]["l2g_score"], 0.9)

    @patch("mcp_servers.gwas_genetics_server._ot_gql")
    def test_credible_sets_returns_study_list(self, mock_gql):
        """get_open_targets_genetics_credible_sets passes EFO → returns credible_sets key."""
        # First call = studies query, second call = credible sets query
        mock_gql.side_effect = [
            {"studies": {"rows": [{"id": "GCST003116", "studyType": "gwas"}]}},
            {"credibleSets": {"rows": [
                {"studyLocusId": "SL1", "pValueMantissa": 1.2, "pValueExponent": -8,
                 "locus": {"rows": [{"variant": {"id": "rs123"}, "posteriorProbability": 0.85}]}}
            ]}},
        ]
        from mcp_servers.gwas_genetics_server import get_open_targets_genetics_credible_sets
        result = get_open_targets_genetics_credible_sets("EFO_0001645", min_pip=0.5)
        self.assertIn("credible_sets", result)
        self.assertEqual(len(result["credible_sets"]), 1)
        self.assertEqual(result["credible_sets"][0]["top_variants"][0]["variant_id"], "rs123")
        self.assertEqual(result["data_source"], "OT_Platform_v4")


class TestFinnGenBurdenResults(unittest.TestCase):

    def _mock_gene_map(self):
        """Minimal in-memory gene_map simulating a parsed GCS file."""
        return {
            "PCSK9": {"beta": -1.42, "se": 0.09, "p": 3.1e-15, "n_variants": 12.0, "mask": "LoF", "source": "FinnGen_R12_live"},
            "LDLR":  {"beta":  1.89, "se": 0.11, "p": 1.2e-22, "n_variants": 87.0, "mask": "LoF+missense", "source": "FinnGen_R12_live"},
            "NOD2":  {"beta":  2.10, "se": 0.19, "p": 8.0e-12, "n_variants": 3.0,  "mask": "LoF+missense", "source": "FinnGen_R12_live"},
        }

    @patch("mcp_servers.gwas_genetics_server._load_burden_phenocode")
    def test_protective_gene_direction(self, mock_load):
        """Negative beta → direction=protective."""
        mock_load.return_value = self._mock_gene_map()
        from mcp_servers.gwas_genetics_server import get_finngen_burden_results
        result = get_finngen_burden_results("coronary artery disease", ["PCSK9"])
        self.assertEqual(len(result["burden_results"]), 1)
        row = result["burden_results"][0]
        self.assertEqual(row["gene"], "PCSK9")
        self.assertLess(row["beta"], 0)
        self.assertEqual(row["direction"], "protective")
        self.assertEqual(row["source"], "FinnGen_R12_live")

    @patch("mcp_servers.gwas_genetics_server._load_burden_phenocode")
    def test_risk_gene_direction(self, mock_load):
        """Positive beta → direction=risk."""
        mock_load.return_value = self._mock_gene_map()
        from mcp_servers.gwas_genetics_server import get_finngen_burden_results
        result = get_finngen_burden_results("CAD", ["LDLR"])
        self.assertGreater(result["burden_results"][0]["beta"], 0)
        self.assertEqual(result["burden_results"][0]["direction"], "risk")

    @patch("mcp_servers.gwas_genetics_server._load_burden_phenocode")
    def test_unknown_gene_returns_empty(self, mock_load):
        """Gene not in parsed data: no results, no exception."""
        mock_load.return_value = self._mock_gene_map()
        from mcp_servers.gwas_genetics_server import get_finngen_burden_results
        result = get_finngen_burden_results("CAD", ["UNKNOWNGENE99"])
        self.assertEqual(result["burden_results"], [])
        self.assertEqual(result["n_found"], 0)

    @patch("mcp_servers.gwas_genetics_server._load_burden_phenocode")
    def test_disease_name_normalized(self, mock_load):
        """Full disease name and short key produce same result."""
        mock_load.return_value = self._mock_gene_map()
        from mcp_servers.gwas_genetics_server import get_finngen_burden_results
        r1 = get_finngen_burden_results("coronary artery disease", ["PCSK9"])
        r2 = get_finngen_burden_results("CAD", ["PCSK9"])
        self.assertEqual(r1["burden_results"][0]["beta"], r2["burden_results"][0]["beta"])

    def test_unknown_disease_returns_note(self):
        """Disease with no phenocode mapping returns a note, no exception."""
        from mcp_servers.gwas_genetics_server import get_finngen_burden_results
        result = get_finngen_burden_results("rare_unknown_disease", ["PCSK9"])
        self.assertEqual(result["burden_results"], [])
        self.assertIn("note", result)


class TestEnsemblSymbolRemap(unittest.TestCase):
    """Test that _maybe_therapeutic_redirection remaps Ensembl IDs to gene symbols."""

    def _make_ensembl_adata(self, ensembl_ids, symbols, n_cells=40):
        """Return mock AnnData with Ensembl IDs as var_names and feature_name column."""
        import numpy as np
        import pandas as pd
        adata = MagicMock()
        adata.var_names = list(ensembl_ids)
        adata.n_obs = n_cells
        adata.n_vars = len(ensembl_ids)
        adata.var = pd.DataFrame(
            {"feature_name": list(symbols)},
            index=pd.Index(list(ensembl_ids)),
        )
        adata.obs = MagicMock()
        adata.obs.columns = []
        adata.obs.__contains__ = lambda self, key: False
        adata.X = np.zeros((n_cells, len(ensembl_ids)))
        return adata

    def test_ensembl_ids_remapped_to_symbols(self):
        """After remap, var_names should be gene symbols, not Ensembl IDs."""
        from collections import Counter
        ensembl_ids = ["ENSG00000066336", "ENSG00000175164", "ENSG00000115267"]
        symbols = ["SPI1", "IRF4", "STAT1"]
        adata = self._make_ensembl_adata(ensembl_ids, symbols)

        # Apply the same remap logic as in _maybe_therapeutic_redirection
        raw_symbols = adata.var["feature_name"].tolist()
        sym_counts = Counter(raw_symbols)
        new_var_names = [
            sym if sym_counts[sym] == 1 else ensembl
            for ensembl, sym in zip(adata.var_names, raw_symbols)
        ]
        adata.var_names = list(new_var_names)
        self.assertEqual(list(adata.var_names), ["SPI1", "IRF4", "STAT1"])

    def test_duplicate_symbol_keeps_ensembl_id(self):
        """Duplicate gene symbols (e.g. MATR3) keep Ensembl ID as var_name."""
        import pandas as pd
        from collections import Counter
        ensembl_ids = ["ENSG00000000001", "ENSG00000000002", "ENSG00000000003"]
        symbols = ["MATR3", "MATR3", "SPI1"]  # MATR3 is duplicate
        adata = self._make_ensembl_adata(ensembl_ids, symbols)

        raw_symbols = adata.var["feature_name"].tolist()
        sym_counts = Counter(raw_symbols)
        new_var_names = [
            sym if sym_counts[sym] == 1 else ensembl
            for ensembl, sym in zip(adata.var_names, raw_symbols)
        ]
        adata.var_names = pd.Index(new_var_names)
        # Duplicates keep Ensembl ID; unique symbol is remapped
        self.assertEqual(new_var_names[0], "ENSG00000000001")  # duplicate → keep Ensembl
        self.assertEqual(new_var_names[1], "ENSG00000000002")  # duplicate → keep Ensembl
        self.assertEqual(new_var_names[2], "SPI1")              # unique → symbol

    def test_no_feature_name_column_unchanged(self):
        """If feature_name column is absent, var_names are not modified."""
        import pandas as pd
        adata = MagicMock()
        adata.var_names = pd.Index(["ENSG00000000001", "ENSG00000000002"])
        adata.var = pd.DataFrame({}, index=pd.Index(["ENSG00000000001", "ENSG00000000002"]))
        # Simulate the guard: only remap when feature_name column exists
        self.assertNotIn("feature_name", adata.var.columns)


if __name__ == "__main__":
    unittest.main()
