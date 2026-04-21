"""
test_new_components.py — Tests for the 5 major pipeline upgrades:
  1. Tahoe-100M MCP server
  2. LINCS L1000 MCP server
  3. SCONE sensitivity pipeline
  4. Scientific Reviewer Agent
  5. Cell-type specificity (tau + bimodality) in target scoring
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ===========================================================================
# 1. Tahoe-100M MCP server
# ===========================================================================

class TestTahoeServer:
    def test_known_gene_returns_summary(self):
        from mcp_servers.tahoe_server import get_tahoe_perturbation_signature
        result = get_tahoe_perturbation_signature("PCSK9", cell_line="K562")
        assert result["gene"] == "PCSK9"
        assert result["cell_line"] == "K562"
        assert "lipid_metabolism" in result["top_programs_dn"]
        assert result["data_tier"] == "curated_summary"
        assert result["evidence_tier"] in ("Tier1_Interventional", "Tier3_Provisional")

    def test_unknown_gene_returns_not_found(self):
        from mcp_servers.tahoe_server import get_tahoe_perturbation_signature
        result = get_tahoe_perturbation_signature("FAKEGENEXYZ")
        assert result["data_tier"] == "not_found"
        assert result["evidence_tier"] == "provisional_virtual"
        assert result["top_programs_up"] == []

    def test_disease_context_selects_correct_cell_line(self):
        from mcp_servers.tahoe_server import get_tahoe_perturbation_signature
        # IBD → HT29 (colon)
        result = get_tahoe_perturbation_signature("NOD2", disease_context="IBD")
        assert result["cell_line"] == "HT29"
        assert "inflammatory_NF-kB" in result["top_programs_up"]

    def test_list_cell_lines_returns_all(self):
        from mcp_servers.tahoe_server import list_tahoe_cell_lines
        result = list_tahoe_cell_lines()
        names = [cl["name"] for cl in result["cell_lines"]]
        assert "K562" in names
        assert "HT29" in names
        assert "HEPG2" in names

    def test_best_cell_line_for_cad(self):
        from mcp_servers.tahoe_server import get_tahoe_best_cell_line
        result = get_tahoe_best_cell_line("CAD")
        assert "K562" in [cl["cell_line"] for cl in result["ranked_cell_lines"]]

    def test_k562_is_tier1_for_cad(self):
        """K562 (myeloid cell line) should be disease-matched for CAD/CHIP biology."""
        from mcp_servers.tahoe_server import TAHOE_CELL_LINES
        assert TAHOE_CELL_LINES["K562"]["tier_cap"] == "Tier1_Interventional"


# ===========================================================================
# 2. Perturbation server (Perturb-seq → Enrichr cascade)
# ===========================================================================

class TestLincsServer:
    def test_list_data_sources(self):
        from mcp_servers.lincs_server import list_perturbation_data_sources
        result = list_perturbation_data_sources()
        assert "sources" in result
        assert "cascade" in result
        names = [s["name"] for s in result["sources"]]
        assert any("Perturb-seq" in n for n in names)
        assert any("Enrichr" in n for n in names)

    @patch("mcp_servers.lincs_server._query_enrichr_lincs")
    @patch("mcp_servers.perturbseq_server._load_cached_signatures", return_value=None)
    def test_signature_returns_tier3(self, mock_ps, mock_enrichr):
        mock_enrichr.return_value = {
            "gene": "PCSK9", "cell_line": "L1000_consensus",
            "signature": {"LDLR": 1.0}, "n_genes_measured": 1,
            "source": "enrichr_lincs_directional",
        }
        from mcp_servers.lincs_server import get_lincs_gene_signature
        result = get_lincs_gene_signature("PCSK9")
        assert result["evidence_tier"] == "Tier3_Provisional"
        assert result["gene"] == "PCSK9"

    @patch("mcp_servers.lincs_server._query_enrichr_lincs")
    @patch("mcp_servers.perturbseq_server._load_cached_signatures", return_value=None)
    def test_program_beta_coverage_below_5pct_returns_none(self, mock_ps, mock_enrichr):
        """Beta should be None when < 5% of program genes are in signature."""
        mock_enrichr.return_value = {
            "gene": "PCSK9", "cell_line": "L1000_consensus",
            "signature": {"GENE_A": 0.5},
            "source": "enrichr_lincs_directional",
        }
        from mcp_servers.lincs_server import compute_lincs_program_beta
        program_genes = [f"GENE_{i}" for i in range(50)]  # 50 genes, only 1 in sig = 2%
        result = compute_lincs_program_beta("PCSK9", program_genes)
        assert result["beta"] is None
        assert result["program_coverage"] < 0.05

    @patch("mcp_servers.lincs_server._query_enrichr_lincs")
    @patch("mcp_servers.perturbseq_server._load_cached_signatures", return_value=None)
    def test_program_beta_computed_correctly(self, mock_ps, mock_enrichr):
        """Beta = mean log2fc of program genes in signature."""
        mock_enrichr.return_value = {
            "gene": "TET2", "cell_line": "L1000_consensus",
            "signature": {"NFKB1": 1.2, "RELA": 0.8, "IL6": 1.0},
            "source": "enrichr_lincs_directional",
        }
        from mcp_servers.lincs_server import compute_lincs_program_beta
        result = compute_lincs_program_beta("TET2", ["NFKB1", "RELA", "IL6"])
        assert result["beta"] is not None
        assert abs(result["beta"] - (1.2 + 0.8 + 1.0) / 3) < 0.01
        assert result["evidence_tier"] == "Tier3_Provisional"


# ===========================================================================
# 3. SCONE sensitivity pipeline
# ===========================================================================

class TestSconeSensitivity:
    def _make_beta_matrix(self):
        return {
            "PCSK9": {
                "lipid_metabolism": {"beta": -0.8, "evidence_tier": "Tier2_Convergent"},
                "inflammatory_NF-kB": None,
            },
            "TET2": {
                "lipid_metabolism": None,
                "inflammatory_NF-kB": {"beta": 0.6, "evidence_tier": "Tier1_Interventional"},
            },
        }

    def _make_gamma_matrix(self):
        return {
            "lipid_metabolism":      {"CAD": 0.35},
            "inflammatory_NF-kB":   {"CAD": 0.40},
        }

    def test_cross_regime_sensitivity_produces_matrix(self):
        from pipelines.scone_sensitivity import compute_cross_regime_sensitivity
        beta  = self._make_beta_matrix()
        gamma = self._make_gamma_matrix()
        tiers = {"PCSK9": "Tier2_Convergent", "TET2": "Tier1_Interventional"}

        result = compute_cross_regime_sensitivity(beta, gamma, tiers)

        assert "PCSK9" in result
        assert "TET2" in result
        # PCSK9 → lipid_metabolism should have positive sensitivity
        assert result["PCSK9"]["lipid_metabolism"] > 0
        # TET2 → inflammatory_NF-kB should have positive sensitivity
        assert result["TET2"]["inflammatory_NF-kB"] > 0
        # PCSK9 → inflammatory_NF-kB is None → should be 0.0
        assert result["PCSK9"]["inflammatory_NF-kB"] == 0.0

    def test_tier1_higher_sensitivity_than_virtual(self):
        """Tier1 evidence should produce higher sensitivity than virtual."""
        from pipelines.scone_sensitivity import compute_cross_regime_sensitivity

        beta_t1 = {"GENE_A": {"prog1": {"beta": 0.5, "evidence_tier": "Tier1_Interventional"}}}
        beta_vt = {"GENE_B": {"prog1": {"beta": 0.5, "evidence_tier": "provisional_virtual"}}}
        gamma   = {"prog1": {"trait": 0.4}}

        s1 = compute_cross_regime_sensitivity(beta_t1, gamma, {"GENE_A": "Tier1_Interventional"})
        sv = compute_cross_regime_sensitivity(beta_vt, gamma, {"GENE_B": "provisional_virtual"})

        assert s1["GENE_A"]["prog1"] > sv["GENE_B"]["prog1"]

    def test_polybic_higher_for_stronger_effect(self):
        from pipelines.scone_sensitivity import polybic_score
        strong = polybic_score(0.5, "Tier1_Interventional")
        weak   = polybic_score(0.05, "Tier1_Interventional")
        assert strong > weak

    def test_polybic_penalizes_virtual_more_than_tier1(self):
        from pipelines.scone_sensitivity import polybic_score
        bic_t1 = polybic_score(0.3, "Tier1_Interventional")
        bic_vt = polybic_score(0.3, "provisional_virtual")
        assert bic_t1 > bic_vt, "Tier1 edges should have lower BIC penalty"

    def test_apply_scone_reweighting_zeroes_rejected_edges(self):
        from pipelines.scone_sensitivity import apply_scone_reweighting

        records = {
            "GENE1__CAD": {"gene": "GENE1", "trait": "CAD", "ota_gamma": 0.4,
                           "dominant_tier": "Tier1_Interventional", "tier": "Tier1_Interventional"},
            "GENE2__CAD": {"gene": "GENE2", "trait": "CAD", "ota_gamma": 0.3,
                           "dominant_tier": "Tier2_Convergent", "tier": "Tier2_Convergent"},
        }
        sensitivity = {"GENE1": {"prog1": 0.3}, "GENE2": {"prog1": 0.2}}
        bic = {"GENE1__CAD": 2.0, "GENE2__CAD": 1.0}
        bootstrap = {
            "GENE1": {"CAD": 0.80},   # passes
            "GENE2": {"CAD": 0.30},   # fails < 0.50
        }

        result = apply_scone_reweighting(records, sensitivity, bic, bootstrap)
        # GENE2 should be zeroed (bootstrap confidence < 0.50)
        assert result["GENE2__CAD"]["ota_gamma"] == 0.0
        assert "bootstrap_rejected" in result["GENE2__CAD"]["scone_flags"]
        # GENE1 should survive (confidence = 0.80)
        assert result["GENE1__CAD"]["ota_gamma"] > 0.0


# ===========================================================================
# 4. Scientific Reviewer Agent
# ===========================================================================

MOCK_DISEASE_QUERY = {"disease_name": "coronary artery disease", "efo_id": "EFO_0001645"}

def _make_pipeline_outputs(
    verdict_tier="Tier2_Convergent",
    anchor_rate=0.85,
    ota_gamma=0.35,
    tractability="small_molecule",
):
    return {
        "beta_matrix_result": {
            "evidence_tier_per_gene": {"PCSK9": verdict_tier, "TET2": verdict_tier},
        },
        "causal_result": {
            "anchor_recovery": {"recovery_rate": anchor_rate, "missing": []},
            "top_genes": [
                {"gene": "PCSK9", "ota_gamma": ota_gamma, "tier": verdict_tier,
                 "scone_confidence": 0.8, "scone_flags": []},
                {"gene": "TET2",  "ota_gamma": ota_gamma * 0.8, "tier": verdict_tier,
                 "scone_confidence": 0.7, "scone_flags": []},
            ],
        },
        "prioritization_result": {
            "targets": [
                {"target_gene": "PCSK9", "rank": 1, "target_score": 0.72,
                 "ota_gamma": ota_gamma, "evidence_tier": verdict_tier,
                 "max_phase": 4, "scone_confidence": 0.8, "scone_flags": [],
                 "tau_specificity": 0.81, "bimodality_coeff": 0.78},
                {"target_gene": "TET2", "rank": 2, "target_score": 0.60,
                 "ota_gamma": ota_gamma * 0.8, "evidence_tier": verdict_tier,
                 "max_phase": 0, "scone_confidence": 0.7, "scone_flags": [],
                 "tau_specificity": 0.45, "bimodality_coeff": 0.58},
            ],
        },
        "chemistry_result": {
            "target_chemistry": {
                "PCSK9": {"tractability": tractability, "best_ic50_nM": 5.0, "max_phase": 4},
                "TET2":  {"tractability": "difficult", "best_ic50_nM": None, "max_phase": 0},
            },
        },
        "graph_output": {},
    }


class TestScientificReviewerAgent:
    def test_approve_on_clean_pipeline(self):
        from agents.tier5_writer.scientific_reviewer_agent import run
        outputs = _make_pipeline_outputs()
        result = run(outputs, MOCK_DISEASE_QUERY)
        assert result["verdict"] == "APPROVE"
        assert result["n_critical"] == 0

    def test_revise_on_virtual_in_top5(self):
        from agents.tier5_writer.scientific_reviewer_agent import run
        outputs = _make_pipeline_outputs(verdict_tier="provisional_virtual")
        result = run(outputs, MOCK_DISEASE_QUERY)
        assert result["verdict"] == "REVISE"
        assert result["n_critical"] >= 1
        # Should point to perturbation_genomics_agent
        critical = [i for i in result["issues"] if i["severity"] == "CRITICAL"]
        assert any(i["agent_to_revisit"] == "perturbation_genomics_agent" for i in critical)

    def test_revise_on_low_anchor_recovery(self):
        from agents.tier5_writer.scientific_reviewer_agent import run
        outputs = _make_pipeline_outputs(anchor_rate=0.60)
        outputs["causal_result"]["anchor_recovery"]["missing"] = ["PCSK9→CAD"]
        result = run(outputs, MOCK_DISEASE_QUERY)
        assert result["verdict"] == "REVISE"
        issues = result["issues"]
        assert any(i["check"] == "B_anchor_recovery" for i in issues)

    def test_revise_on_zero_ota_gamma(self):
        from agents.tier5_writer.scientific_reviewer_agent import run
        outputs = _make_pipeline_outputs(ota_gamma=0.0)
        result = run(outputs, MOCK_DISEASE_QUERY)
        assert result["n_major"] >= 1
        assert any(i["check"] == "D_missing_effect_size" for i in result["issues"])

    def test_revise_on_scone_bootstrap_rejected(self):
        from agents.tier5_writer.scientific_reviewer_agent import run
        outputs = _make_pipeline_outputs()
        # Mark PCSK9 as bootstrap-rejected
        outputs["prioritization_result"]["targets"][0]["scone_flags"] = ["bootstrap_rejected"]
        outputs["prioritization_result"]["targets"][0]["scone_confidence"] = 0.25
        result = run(outputs, MOCK_DISEASE_QUERY)
        assert any(i["check"] == "E_scone_bootstrap_rejected" for i in result["issues"])

    def test_approved_vs_flagged_targets_split(self):
        from agents.tier5_writer.scientific_reviewer_agent import run
        outputs = _make_pipeline_outputs(ota_gamma=0.0)
        result = run(outputs, MOCK_DISEASE_QUERY)
        # Both genes have zero gamma so should be flagged
        assert len(result["flagged_targets"]) > 0

    def test_summary_narrative_present(self):
        from agents.tier5_writer.scientific_reviewer_agent import run
        result = run(_make_pipeline_outputs(), MOCK_DISEASE_QUERY)
        assert len(result["summary"]) > 20
        assert "APPROVE" in result["summary"] or "REVISE" in result["summary"]


# ===========================================================================
# 5. Cell-type specificity (tau + bimodality)
# ===========================================================================

class TestCellTypeSpecificity:
    def test_tau_query_returns_known_genes(self):
        from mcp_servers.single_cell_server import get_gene_tau_specificity
        result = get_gene_tau_specificity(["PCSK9", "TET2", "IL6R"])
        scores = result["tau_scores"]
        assert "PCSK9" in scores
        assert scores["PCSK9"]["tau"] == pytest.approx(0.81, abs=0.01)
        assert scores["PCSK9"]["interpretation"] == "highly_specific"

    def test_tau_unknown_gene_returns_none(self):
        from mcp_servers.single_cell_server import get_gene_tau_specificity
        result = get_gene_tau_specificity(["NONEXISTENTGENE"])
        entry = result["tau_scores"]["NONEXISTENTGENE"]
        assert entry["tau"] is None
        assert entry["interpretation"] == "unknown"

    def test_bimodality_score_returned(self):
        from mcp_servers.single_cell_server import get_gene_bimodality_scores
        result = get_gene_bimodality_scores(["PCSK9", "HLA-DRA"])
        scores = result["bimodality_scores"]
        assert scores["PCSK9"]["bc"] == pytest.approx(0.79, abs=0.01)
        assert scores["PCSK9"]["bimodal"] is True

    def test_bimodality_threshold_at_0555(self):
        """BC > 0.555 → bimodal, ≤ 0.555 → not."""
        from mcp_servers.single_cell_server import get_gene_bimodality_scores
        # ASXL1 BC = 0.53 in GTEx v10 fallback — below the 0.555 bimodal threshold
        result = get_gene_bimodality_scores(["ASXL1"])
        assert result["bimodality_scores"]["ASXL1"]["bimodal"] is False

    @patch("mcp_servers.single_cell_server.get_gene_tau_specificity")
    @patch("mcp_servers.single_cell_server.get_gene_bimodality_scores")
    @patch("mcp_servers.gwas_genetics_server.query_gnomad_lof_constraint")
    @patch("mcp_servers.open_targets_server.get_open_targets_disease_targets")
    def test_target_prioritization_uses_specificity(
        self, mock_ot, mock_gnomad, mock_bc, mock_tau
    ):
        """Target prioritization composite score includes specificity_score field."""
        mock_ot.return_value = {"targets": []}
        mock_gnomad.return_value = {"genes": []}
        mock_tau.return_value = {
            "tau_scores": {
                "PCSK9": {"tau": 0.81, "interpretation": "highly_specific"},
                "TET2":  {"tau": 0.45, "interpretation": "moderately_specific"},
            }
        }
        mock_bc.return_value = {
            "bimodality_scores": {
                "PCSK9": {"bc": 0.78, "bimodal": True},
                "TET2":  {"bc": 0.58, "bimodal": True},
            }
        }

        causal_out = {
            "top_genes": [
                {"gene": "PCSK9", "ota_gamma": 0.45, "tier": "Tier2_Convergent",
                 "programs": ["lipid_metabolism"], "scone_confidence": None, "scone_flags": []},
                {"gene": "TET2",  "ota_gamma": 0.30, "tier": "Tier1_Interventional",
                 "programs": ["inflammatory_NF-kB"], "scone_confidence": None, "scone_flags": []},
            ]
        }
        kg_out  = {"drug_target_summary": []}
        dq      = {"efo_id": "EFO_0001645", "disease_name": "coronary artery disease"}

        from agents.tier4_translation.target_prioritization_agent import run
        result = run(causal_out, kg_out, dq)

        targets = result["targets"]
        pcsk9   = next(t for t in targets if t["target_gene"] == "PCSK9")

        assert "tau_specificity"   in pcsk9
        assert "bimodality_coeff"  in pcsk9
        assert "specificity_score" in pcsk9
        assert pcsk9["tau_specificity"] == pytest.approx(0.81, abs=0.01)
        # PCSK9 should have higher specificity score than TET2
        tet2    = next(t for t in targets if t["target_gene"] == "TET2")
        assert pcsk9["specificity_score"] > tet2["specificity_score"]
        # highly_specific flag should be set
        assert "highly_specific" in pcsk9["flags"]
        # gamma CI fields should be present
        assert "ota_gamma_sigma"    in pcsk9
        assert "ota_gamma_ci_lower" in pcsk9
        assert "ota_gamma_ci_upper" in pcsk9


# ===========================================================================
# 6. BRG Diffusion (BioPathNet-style inductive KG completion)
# ===========================================================================

class TestBRGDiffusion:
    """Tests for pipelines/biopath_gnn.py — BRG adjacency + RWR diffusion."""

    def test_build_brg_from_ppi_edges(self):
        """PPI edges create bidirectional adjacency entries."""
        from pipelines.biopath_gnn import build_brg
        ppi = [{"protein_a": "PCSK9", "protein_b": "LDLR", "score": 850}]
        adj = build_brg(ppi, {}, [], [])
        assert "PCSK9" in adj
        assert "LDLR" in adj
        # Bidirectional
        pcsk9_neighbors = {n for n, _ in adj["PCSK9"]}
        assert "LDLR" in pcsk9_neighbors

    def test_build_brg_pathway_relay_node(self):
        """Reactome pathway creates PW: relay node and bidirectional gene edges."""
        from pipelines.biopath_gnn import build_brg
        pathway_map = {"R-HSA-163200": ["PCSK9", "LDLR", "HMGCR"]}
        adj = build_brg([], pathway_map, [], [])
        assert "PW:R-HSA-163200" in adj
        pw_neighbors = {n for n, _ in adj["PW:R-HSA-163200"]}
        assert "PCSK9" in pw_neighbors
        assert "LDLR"  in pw_neighbors

    def test_rwr_converges_and_seeds_high(self):
        """RWR: seed gene should score higher than unconnected gene."""
        from pipelines.biopath_gnn import run_rwr
        adj = {
            "PCSK9": [("LDLR", 0.6), ("HMGCR", 0.4)],
            "LDLR":  [("PCSK9", 1.0)],
            "HMGCR": [("PCSK9", 1.0)],
        }
        seed = {"PCSK9": 1.0}
        scores = run_rwr(adj, seed, n_iter=20)
        # Seed gene should retain a non-trivial score
        assert scores["PCSK9"] > 0
        # Connected genes should have non-zero scores
        assert scores["LDLR"] > 0

    def test_novel_link_scoring_excludes_seed_genes(self):
        """score_novel_links must exclude genes in known_genes set."""
        from pipelines.biopath_gnn import score_novel_links
        rwr_scores = {"PCSK9": 0.5, "LDLR": 0.3, "NOVEL_GENE": 0.2, "PW:ABC": 0.9}
        known = {"PCSK9", "LDLR"}
        candidates = score_novel_links(rwr_scores, known, top_k=10)
        gene_names = [c["gene"] for c in candidates]
        assert "PCSK9"     not in gene_names
        assert "LDLR"      not in gene_names
        assert "PW:ABC"    not in gene_names  # pathway nodes excluded
        assert "NOVEL_GENE" in gene_names

    def test_brg_run_returns_expected_structure(self):
        """biopath_gnn.run() returns the expected dict schema."""
        from unittest.mock import patch as mpatch
        from pipelines.biopath_gnn import run as brg_run

        causal_in = {
            "top_genes": [
                {"gene": "PCSK9", "ota_gamma": 0.44, "tier": "Tier2_Convergent"},
                {"gene": "IL6R",  "ota_gamma": 0.24, "tier": "Tier2_Convergent"},
            ]
        }
        kg_in = {
            "pathway_gene_map": {"R-HSA-163200": ["PCSK9", "LDLR"]},
            "drug_target_summary": [{"target": "PCSK9", "drug": "evolocumab", "max_phase": 4}],
        }
        dq = {"disease_name": "coronary artery disease"}

        with mpatch(
            "mcp_servers.pathways_kg_server.get_string_interactions",
            return_value={"interactions": []},
        ):
            result = brg_run(causal_in, kg_in, dq)

        assert "novel_candidates"   in result
        assert "n_seed_genes"       in result
        assert "n_brg_edges"        in result
        assert "n_novel_candidates" in result
        assert isinstance(result["novel_candidates"], list)


# ===========================================================================
# 7. Beta uncertainty propagation
# ===========================================================================

class TestBetaUncertainty:
    """Tests for beta_sigma fields and gamma delta-method CI."""

    def test_tier1_sigma_lower_than_virtual(self):
        """Tier1 qualitative sigma (0.5) < Virtual sigma (0.7)."""
        from pipelines.ota_beta_estimation import estimate_beta_tier1, estimate_beta_virtual
        t1 = estimate_beta_tier1("PCSK9", "lipid_metabolism")
        virtual = estimate_beta_virtual("PCSK9", "lipid_metabolism", pathway_member=True)
        if t1 is not None:
            assert t1["beta_sigma"] < virtual["beta_sigma"]
        assert virtual["beta_sigma"] == pytest.approx(0.70, abs=0.01)

    def test_tier2_sigma_uses_eqtl_se(self):
        """Tier2 sigma = |se * loading| when eQTL SE is provided."""
        from pipelines.ota_beta_estimation import estimate_beta_tier2
        result = estimate_beta_tier2(
            "PCSK9", "lipid_metabolism",
            eqtl_data={"nes": 0.4, "se": 0.1},
            program_loading=0.8,
        )
        assert result is not None
        assert result["beta_sigma"] == pytest.approx(0.1 * 0.8, abs=0.01)

    def test_delta_method_ci_wider_for_virtual(self):
        """Virtual-tier gamma CI should be wider than Tier1 CI."""
        from pipelines.ota_gamma_estimation import compute_ota_gamma_with_uncertainty

        beta_t1 = {
            "lipid_metabolism": {
                "beta": 0.5, "evidence_tier": "Tier1_Interventional", "beta_sigma": 0.075
            }
        }
        beta_virt = {
            "lipid_metabolism": {
                "beta": 0.5, "evidence_tier": "provisional_virtual", "beta_sigma": 0.35
            }
        }
        gamma_est = {
            "lipid_metabolism": {"gamma": 0.44, "gamma_se": 0.08}
        }

        r_t1   = compute_ota_gamma_with_uncertainty("PCSK9", "CAD", beta_t1,   gamma_est)
        r_virt = compute_ota_gamma_with_uncertainty("PCSK9", "CAD", beta_virt, gamma_est)

        ci_width_t1   = r_t1["ota_gamma_ci_upper"]   - r_t1["ota_gamma_ci_lower"]
        ci_width_virt = r_virt["ota_gamma_ci_upper"] - r_virt["ota_gamma_ci_lower"]
        assert ci_width_virt > ci_width_t1

    def test_ci_contains_point_estimate(self):
        """95% CI must contain the point estimate ota_gamma."""
        from pipelines.ota_gamma_estimation import compute_ota_gamma_with_uncertainty

        beta_est = {
            "lipid_metabolism": {
                "beta": 0.5, "evidence_tier": "Tier2_Convergent", "beta_sigma": 0.125
            }
        }
        gamma_est = {"lipid_metabolism": {"gamma": 0.44, "gamma_se": 0.08}}
        result = compute_ota_gamma_with_uncertainty("PCSK9", "CAD", beta_est, gamma_est)

        assert result["ota_gamma_ci_lower"] <= result["ota_gamma"] <= result["ota_gamma_ci_upper"]

    def test_target_record_has_gamma_ci_fields(self):
        """Target prioritization output includes ota_gamma_sigma and CI fields."""
        from unittest.mock import patch as mpatch

        with (
            mpatch("mcp_servers.open_targets_server.get_open_targets_disease_targets",
                   return_value={"targets": []}),
            mpatch("mcp_servers.clinical_trials_server.get_trials_for_target",
                   return_value={"trials": []}),
            mpatch("mcp_servers.gwas_genetics_server.query_gnomad_lof_constraint",
                   return_value={"genes": []}),
            mpatch("mcp_servers.single_cell_server.get_gene_tau_specificity",
                   return_value={"tau_scores": {}}),
            mpatch("mcp_servers.single_cell_server.get_gene_bimodality_scores",
                   return_value={"bimodality_scores": {}}),
        ):
            from agents.tier4_translation.target_prioritization_agent import run
            result = run(
                causal_discovery_result={
                    "top_genes": [
                        {"gene": "PCSK9", "ota_gamma": 0.44, "tier": "Tier2_Convergent",
                         "programs": [], "scone_confidence": None, "scone_flags": [],
                         "ota_gamma_sigma": 0.09, "ota_gamma_ci_lower": 0.26,
                         "ota_gamma_ci_upper": 0.62},
                    ]
                },
                kg_completion_result={"drug_target_summary": [], "brg_novel_candidates": []},
                disease_query={"efo_id": "EFO_0001645", "disease_name": "coronary artery disease"},
            )
        targets = result["targets"]
        assert len(targets) == 1
        t = targets[0]
        assert t["ota_gamma_sigma"]    == pytest.approx(0.09, abs=0.01)
        assert t["ota_gamma_ci_lower"] == pytest.approx(0.26, abs=0.01)
        assert t["ota_gamma_ci_upper"] == pytest.approx(0.62, abs=0.01)

    def test_brg_novel_candidate_flag_set(self):
        """BRG candidate gene gets 'brg_novel_candidate' flag in target record."""
        from unittest.mock import patch as mpatch

        with (
            mpatch("mcp_servers.open_targets_server.get_open_targets_disease_targets",
                   return_value={"targets": []}),
            mpatch("mcp_servers.clinical_trials_server.get_trials_for_target",
                   return_value={"trials": []}),
            mpatch("mcp_servers.gwas_genetics_server.query_gnomad_lof_constraint",
                   return_value={"genes": []}),
            mpatch("mcp_servers.single_cell_server.get_gene_tau_specificity",
                   return_value={"tau_scores": {}}),
            mpatch("mcp_servers.single_cell_server.get_gene_bimodality_scores",
                   return_value={"bimodality_scores": {}}),
        ):
            from agents.tier4_translation.target_prioritization_agent import run
            result = run(
                causal_discovery_result={
                    "top_genes": [
                        {"gene": "LDLR", "ota_gamma": 0.30, "tier": "Tier3_Provisional",
                         "programs": [], "scone_confidence": None, "scone_flags": [],
                         "ota_gamma_sigma": 0.10, "ota_gamma_ci_lower": 0.10,
                         "ota_gamma_ci_upper": 0.50},
                    ]
                },
                kg_completion_result={
                    "drug_target_summary": [],
                    "brg_novel_candidates": [{"gene": "LDLR", "brg_score": 0.045, "novel": False}],
                },
                disease_query={"efo_id": "EFO_0001645", "disease_name": "coronary artery disease"},
            )
        t = result["targets"][0]
        assert "brg_novel_candidate" in t["flags"]
        assert t["brg_score"] == pytest.approx(0.045, abs=0.001)
