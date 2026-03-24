"""
tests/test_orchestrator.py — Unit tests for orchestration layer.

Tests:
  - ScientificReviewer: approve/reject/warn logic
  - ContradictionAgent: demotion decision tree
  - ChiefOfStaff: pipeline dispatch + retry
  - PIOrchestrator: quality gates + escalation
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

VALID_TIER1_EDGE = {
    "from_node":     "PCSK9",
    "from_type":     "gene",
    "to_node":       "LDL-C",
    "to_type":       "trait",
    "effect_size":   -0.40,
    "ci_lower":      -0.55,
    "ci_upper":      -0.25,
    "evidence_type": "germline",
    "evidence_tier": "Tier1_Interventional",
    "method":        "mr",
    "data_source":   "10.1038/ng.2797",   # DOI for PCSK9 paper
    "data_source_version": "2013",
    "e_value":       8.5,
    "mr_ivw":        -0.40,
    "mr_egger_intercept": 0.85,
}

VALID_TIER2_EDGE = {
    **VALID_TIER1_EDGE,
    "evidence_tier": "Tier2_Convergent",
    "data_source":   "10.1093/hmg/ddu124",
}

VALID_TIER3_EDGE = {
    **VALID_TIER1_EDGE,
    "evidence_tier": "Tier3_Provisional",
    "data_source":   "bick2020_table3",
    "ci_lower":       None,
    "ci_upper":       None,
}

MOCK_DISEASE_QUERY = {
    "disease_name":    "coronary artery disease",
    "efo_id":          "EFO_0001645",
    "modifier_types":  ["germline", "somatic_chip", "drug"],
    "primary_gwas_id": "ieu-a-7",
}


# ===========================================================================
# Scientific Reviewer
# ===========================================================================

class TestScientificReviewer:

    def test_approve_valid_tier1(self):
        from orchestrator.scientific_reviewer import review_edge
        result = review_edge(VALID_TIER1_EDGE)
        assert result["decision"] == "APPROVE"
        assert result["rejection_reason"] is None

    def test_approve_valid_tier2(self):
        from orchestrator.scientific_reviewer import review_edge
        result = review_edge(VALID_TIER2_EDGE)
        assert result["decision"] in ("APPROVE", "APPROVE_WITH_WARNING")

    def test_reject_memory_source(self):
        from orchestrator.scientific_reviewer import review_edge
        bad_edge = {**VALID_TIER1_EDGE, "data_source": "memory"}
        result = review_edge(bad_edge)
        assert result["decision"] == "REJECT"
        assert "memory" in result["rejection_reason"].lower()

    def test_reject_unknown_source(self):
        from orchestrator.scientific_reviewer import review_edge
        bad_edge = {**VALID_TIER1_EDGE, "data_source": "unknown"}
        result = review_edge(bad_edge)
        assert result["decision"] == "REJECT"

    def test_reject_null_effect_size(self):
        from orchestrator.scientific_reviewer import review_edge
        bad_edge = {**VALID_TIER1_EDGE, "effect_size": None}
        result = review_edge(bad_edge)
        assert result["decision"] == "REJECT"
        assert "None" in result["rejection_reason"]

    def test_reject_nan_effect_size(self):
        from orchestrator.scientific_reviewer import review_edge
        import math
        bad_edge = {**VALID_TIER1_EDGE, "effect_size": float("nan")}
        result = review_edge(bad_edge)
        assert result["decision"] == "REJECT"

    def test_reject_tier1_without_doi(self):
        from orchestrator.scientific_reviewer import review_edge
        bad_edge = {**VALID_TIER1_EDGE, "data_source": "my_spreadsheet.xlsx"}
        result = review_edge(bad_edge)
        assert result["decision"] == "REJECT"
        assert "PMID or DOI" in result["rejection_reason"]

    def test_reject_tier1_weak_instrument(self):
        from orchestrator.scientific_reviewer import review_edge
        bad_edge = {**VALID_TIER1_EDGE, "f_statistic": 5.0}
        result = review_edge(bad_edge)
        assert result["decision"] == "REJECT"
        assert "F-statistic" in result["rejection_reason"]

    def test_warn_low_evalue(self):
        from orchestrator.scientific_reviewer import review_edge
        edge = {**VALID_TIER1_EDGE, "e_value": 1.5}
        result = review_edge(edge)
        assert result["decision"] == "APPROVE_WITH_WARNING"
        assert any("E-value" in w for w in result["warnings"])

    def test_warn_missing_ci(self):
        from orchestrator.scientific_reviewer import review_edge
        edge = {**VALID_TIER2_EDGE, "ci_lower": None, "ci_upper": None}
        result = review_edge(edge)
        assert result["decision"] == "APPROVE_WITH_WARNING"
        assert any("confidence interval" in w for w in result["warnings"])

    def test_warn_pleiotropy(self):
        from orchestrator.scientific_reviewer import review_edge
        edge = {**VALID_TIER1_EDGE, "mr_egger_intercept": 0.02}  # p < 0.05 = pleiotropy
        result = review_edge(edge)
        assert result["decision"] == "APPROVE_WITH_WARNING"
        assert any("pleiotropy" in w.lower() for w in result["warnings"])

    def test_warn_tier3_provisional(self):
        from orchestrator.scientific_reviewer import review_edge
        result = review_edge(VALID_TIER3_EDGE)
        assert result["decision"] == "APPROVE_WITH_WARNING"
        assert any("provisional" in w.lower() or "single-source" in w.lower() for w in result["warnings"])

    def test_batch_review_counts(self):
        from orchestrator.scientific_reviewer import review_batch
        edges = [
            VALID_TIER1_EDGE,
            {**VALID_TIER1_EDGE, "data_source": "memory"},    # reject
            {**VALID_TIER1_EDGE, "e_value": 1.0},             # warn
        ]
        result = review_batch(edges)
        assert result["n_rejected"] == 1
        assert result["n_approved_with_warning"] == 1
        assert result["n_approved"] == 1
        assert len(result["approved_edges"]) == 2
        assert len(result["rejected_edges"]) == 1


# ===========================================================================
# Contradiction Agent
# ===========================================================================

class TestContradictionAgent:

    @patch("mcp_servers.graph_db_server.query_graph_for_disease")
    def test_no_contradiction_when_no_existing_edges(self, mock_query):
        mock_query.return_value = {"edges": []}
        from orchestrator.contradiction_agent import check_contradiction
        result = check_contradiction(VALID_TIER1_EDGE, "coronary artery disease")
        assert result["has_contradiction"] is False
        assert result["action"] == "NONE"

    @patch("mcp_servers.graph_db_server.query_graph_for_disease")
    def test_no_contradiction_same_direction(self, mock_query):
        existing = {**VALID_TIER1_EDGE, "effect_size": -0.3}  # same negative direction
        mock_query.return_value = {"edges": [existing]}
        from orchestrator.contradiction_agent import check_contradiction
        result = check_contradiction(VALID_TIER1_EDGE, "coronary artery disease")
        assert result["has_contradiction"] is False

    @patch("mcp_servers.graph_db_server.query_graph_for_disease")
    def test_tier2_vs_tier1_contradiction_soft_warn(self, mock_query):
        # Existing Tier1, new Tier2 with opposite sign
        existing = {**VALID_TIER1_EDGE, "effect_size": -0.4, "evidence_tier": "Tier1_Interventional"}
        new_edge = {**VALID_TIER2_EDGE, "effect_size": 0.2}   # positive = opposite
        mock_query.return_value = {"edges": [existing]}
        from orchestrator.contradiction_agent import check_contradiction
        result = check_contradiction(new_edge, "coronary artery disease")
        assert result["has_contradiction"] is True
        assert result["action"] == "SOFT_WARN"

    @patch("mcp_servers.graph_db_server.query_graph_for_disease")
    def test_tier1_vs_tier1_conditional_demotion(self, mock_query):
        # Both Tier1 with opposite signs → conditional demotion
        existing = {**VALID_TIER1_EDGE, "effect_size": -0.4, "evidence_tier": "Tier1_Interventional"}
        new_edge = {**VALID_TIER1_EDGE, "effect_size": 0.3, "evidence_tier": "Tier1_Interventional"}
        mock_query.return_value = {"edges": [existing]}
        from orchestrator.contradiction_agent import check_contradiction
        result = check_contradiction(new_edge, "coronary artery disease")
        assert result["has_contradiction"] is True
        assert result["action"] == "DEMOTE_EXISTING"

    @patch("mcp_servers.graph_db_server.query_graph_for_disease")
    def test_lower_tier_ignored_vs_tier1(self, mock_query):
        # Tier3 contradicting Tier1 → IGNORE
        existing = {**VALID_TIER1_EDGE, "effect_size": -0.4, "evidence_tier": "Tier1_Interventional"}
        new_edge = {**VALID_TIER3_EDGE, "effect_size": 0.1}
        mock_query.return_value = {"edges": [existing]}
        from orchestrator.contradiction_agent import check_contradiction
        result = check_contradiction(new_edge, "coronary artery disease")
        assert result["has_contradiction"] is True
        assert result["action"] == "IGNORE"

    @patch("mcp_servers.graph_db_server.query_graph_for_disease")
    def test_run_batch_counts(self, mock_query):
        mock_query.return_value = {"edges": []}
        from orchestrator.contradiction_agent import run as check_contradictions
        result = check_contradictions([VALID_TIER1_EDGE, VALID_TIER2_EDGE], "cad")
        assert result["n_contradictions_found"] == 0
        assert len(result["safe_to_write"]) == 2
        assert result["graph_integrity_score"] == 1.0

    def test_check_pleiotropy_demotion_returns_none_when_ok(self):
        from orchestrator.contradiction_agent import check_pleiotropy_demotion
        result = check_pleiotropy_demotion(VALID_TIER1_EDGE, mr_egger_p=0.8)
        assert result is None

    def test_check_pleiotropy_demotion_returns_dict_when_significant(self):
        from orchestrator.contradiction_agent import check_pleiotropy_demotion
        result = check_pleiotropy_demotion(VALID_TIER1_EDGE, mr_egger_p=0.02)
        assert result is not None
        assert "Tier3" in result["suggested_tier"]


# ===========================================================================
# Chief of Staff
# ===========================================================================

class TestChiefOfStaff:

    @patch("agents.tier1_phenomics.phenotype_architect.run")
    @patch("agents.tier1_phenomics.statistical_geneticist.run")
    @patch("agents.tier1_phenomics.somatic_exposure_agent.run")
    @patch("agents.tier2_pathway.perturbation_genomics_agent.run")
    @patch("agents.tier2_pathway.regulatory_genomics_agent.run")
    @patch("pipelines.ota_gamma_estimation.estimate_gamma")
    @patch("agents.tier3_causal.causal_discovery_agent.run")
    @patch("agents.tier3_causal.kg_completion_agent.run")
    @patch("agents.tier4_translation.target_prioritization_agent.run")
    @patch("agents.tier4_translation.chemistry_agent.run")
    @patch("agents.tier4_translation.clinical_trialist_agent.run")
    @patch("agents.tier5_writer.scientific_writer_agent.run")
    def test_full_pipeline_success(
        self,
        mock_writer, mock_ct, mock_chem, mock_tpa, mock_kgc, mock_cda,
        mock_gamma, mock_rga, mock_pga, mock_sea, mock_sg, mock_pa,
    ):
        # Patch all agents to return minimal valid dicts
        mock_pa.return_value = {**MOCK_DISEASE_QUERY, "warnings": []}
        mock_sg.return_value = {
            "instruments": [], "anchor_genes_validated": {}, "warnings": []
        }
        mock_sea.return_value = {
            "chip_edges": [], "viral_edges": [], "drug_edges": [],
            "summary": {"n_chip_genes": 0, "n_viral_viruses": 0, "n_drug_targets": 0},
            "warnings": [],
        }
        mock_pga.return_value = {
            "genes": ["PCSK9"], "programs": ["lipid_metabolism"],
            "beta_matrix": {"PCSK9": {"lipid_metabolism": -0.4}},
            "evidence_tier_per_gene": {"PCSK9": "Tier1_Interventional"},
            "n_tier1": 1, "n_tier2": 0, "n_tier3": 0, "n_virtual": 0, "warnings": [],
        }
        mock_rga.return_value = {
            "gene_eqtl_summary": {}, "gene_program_overlap": {},
            "tier2_upgrades": [], "warnings": [],
        }
        mock_gamma.return_value = {
            "lipid_metabolism": {"coronary artery disease": 0.8}
        }
        mock_cda.return_value = {
            "n_edges_written": 1, "n_edges_rejected": 0,
            "top_genes": [{"gene": "PCSK9", "ota_gamma": -0.32, "tier": "Tier1_Interventional", "programs": ["lipid_metabolism"]}],
            "anchor_recovery": {"recovery_rate": 0.83, "recovered": [], "missing": []},
            "shd": 2, "warnings": [],
        }
        mock_kgc.return_value = {
            "n_pathway_edges_added": 5, "n_ppi_edges_added": 3,
            "n_drug_target_edges_added": 2, "n_primekg_edges_added": 1,
            "top_pathways": [], "drug_target_summary": [],
            "contradictions_flagged": 0, "warnings": [],
        }
        mock_tpa.return_value = {
            "targets": [{
                "target_gene": "PCSK9", "rank": 1, "target_score": 0.73,
                "ota_gamma": -0.32, "evidence_tier": "Tier1_Interventional",
                "ot_score": 0.89, "max_phase": 4, "known_drugs": ["evolocumab"],
                "pli": 0.02, "flags": ["genetic_anchor"], "top_programs": [],
                "key_evidence": [], "safety_flags": [],
            }],
            "warnings": [],
        }
        mock_chem.return_value = {"target_chemistry": {}, "repurposing_candidates": [], "warnings": []}
        mock_ct.return_value = {
            "trial_summary": {}, "key_trials": [], "development_risk": {},
            "repurposing_opportunities": [], "warnings": [],
        }
        mock_writer.return_value = {
            "disease_name": "coronary artery disease", "efo_id": "EFO_0001645",
            "target_list": [], "anchor_edge_recovery": 0.83,
            "n_tier1_edges": 1, "n_tier2_edges": 0, "n_tier3_edges": 0, "n_virtual_edges": 0,
            "executive_summary": "Test summary.", "target_table": "",
            "top_target_narratives": [], "evidence_quality": {}, "limitations": "",
            "pipeline_version": "0.1.0", "generated_at": "2026-01-01T00:00:00Z", "warnings": [],
        }

        from orchestrator.chief_of_staff import run_pipeline
        result = run_pipeline("coronary artery disease")

        assert result["pipeline_status"] == "SUCCESS"
        assert "graph_output" in result
        assert "phenotype_result" in result

    @patch("agents.tier1_phenomics.phenotype_architect.run")
    def test_pipeline_halts_on_phenotype_failure(self, mock_pa):
        mock_pa.side_effect = Exception("API down")
        from orchestrator.chief_of_staff import run_pipeline
        result = run_pipeline("coronary artery disease")
        # After 2 retries, should return stub_fallback
        assert result.get("phenotype_result", {}).get("stub_fallback") is True
        assert "FAILED_TIER1_PHENOTYPE" in result.get("pipeline_status", "")

    @patch("agents.tier1_phenomics.phenotype_architect.run")
    @patch("agents.tier1_phenomics.statistical_geneticist.run")
    @patch("agents.tier1_phenomics.somatic_exposure_agent.run")
    @patch("agents.tier2_pathway.perturbation_genomics_agent.run")
    @patch("agents.tier2_pathway.regulatory_genomics_agent.run")
    @patch("pipelines.ota_gamma_estimation.estimate_gamma")
    @patch("agents.tier3_causal.causal_discovery_agent.run")
    def test_pipeline_halts_on_low_anchor_recovery(
        self, mock_cda, mock_gamma, mock_rga, mock_pga, mock_sea, mock_sg, mock_pa
    ):
        mock_pa.return_value = {**MOCK_DISEASE_QUERY, "warnings": []}
        mock_sg.return_value = {"instruments": [], "anchor_genes_validated": {}, "warnings": []}
        mock_sea.return_value = {"chip_edges": [], "viral_edges": [], "drug_edges": [], "summary": {}, "warnings": []}
        mock_pga.return_value = {"genes": [], "programs": [], "beta_matrix": {}, "evidence_tier_per_gene": {}, "n_tier1": 0, "n_tier2": 0, "n_tier3": 0, "n_virtual": 0, "warnings": []}
        mock_rga.return_value = {"gene_eqtl_summary": {}, "gene_program_overlap": {}, "tier2_upgrades": [], "warnings": []}
        mock_gamma.return_value = {}
        mock_cda.return_value = {
            "n_edges_written": 0, "n_edges_rejected": 0, "top_genes": [],
            "anchor_recovery": {"recovery_rate": 0.5, "recovered": [], "missing": ["PCSK9→LDL-C"]},
            "shd": 5, "warnings": ["CRITICAL: anchor recovery 50%"],
        }

        from orchestrator.chief_of_staff import run_pipeline
        result = run_pipeline("coronary artery disease")
        assert "QUALITY_GATE_FAILED" in result.get("pipeline_status", "")


# ===========================================================================
# PI Orchestrator
# ===========================================================================

class TestPIOrchestrator:

    def _mock_pipeline_success(self):
        return {
            "pipeline_status":   "SUCCESS",
            "pipeline_duration_s": 12.5,
            "all_warnings": [],
            "phenotype_result": MOCK_DISEASE_QUERY,
            "genetics_result": {"instruments": [{"f_statistic": 80.0}], "anchor_genes_validated": {}, "warnings": []},
            "causal_result": {
                "n_edges_written": 3,
                "anchor_recovery": {"recovery_rate": 0.83, "recovered": [], "missing": []},
                "warnings": [],
            },
            "prioritization_result": {
                "targets": [
                    {"target_gene": "PCSK9", "rank": 1, "target_score": 0.73,
                     "evidence_tier": "Tier1_Interventional", "safety_flags": []},
                ],
                "warnings": [],
            },
            "graph_output": {
                "disease_name": "coronary artery disease", "efo_id": "EFO_0001645",
                "target_list": [], "anchor_edge_recovery": 0.83,
                "n_tier1_edges": 1, "n_tier2_edges": 0, "n_tier3_edges": 0, "n_virtual_edges": 0,
                "executive_summary": "Test.", "target_table": "",
                "top_target_narratives": [], "evidence_quality": {}, "limitations": "",
                "pipeline_version": "0.1.0", "generated_at": "2026-01-01T00:00:00Z", "warnings": [],
            },
        }

    @patch("orchestrator.chief_of_staff.run_pipeline")
    def test_analyze_disease_success(self, mock_pipeline):
        mock_pipeline.return_value = self._mock_pipeline_success()
        from orchestrator.pi_orchestrator import analyze_disease
        result = analyze_disease("coronary artery disease")
        assert result["pipeline_status"] == "SUCCESS"
        assert result["pi_reviewed"] is True
        assert result["disease_name"] == "coronary artery disease"

    @patch("orchestrator.chief_of_staff.run_pipeline")
    def test_analyze_disease_weak_instruments_escalation(self, mock_pipeline):
        outputs = self._mock_pipeline_success()
        # Inject weak F-stat
        outputs["genetics_result"]["instruments"] = [{"f_statistic": 3.0, "exposure": "LDL-C"}]
        mock_pipeline.return_value = outputs
        from orchestrator.pi_orchestrator import analyze_disease
        result = analyze_disease("coronary artery disease")
        # Should have an escalation warning
        assert result["n_escalations"] >= 1

    @patch("orchestrator.chief_of_staff.run_pipeline")
    def test_analyze_disease_halts_on_low_anchor_recovery(self, mock_pipeline):
        outputs = self._mock_pipeline_success()
        outputs["causal_result"]["anchor_recovery"]["recovery_rate"] = 0.50
        outputs["causal_result"]["anchor_recovery"]["missing"] = ["PCSK9→LDL-C"]
        mock_pipeline.return_value = outputs
        from orchestrator.pi_orchestrator import analyze_disease
        result = analyze_disease("coronary artery disease")
        # Should halt with HALTED status
        assert "HALTED" in result["pipeline_status"]

    @patch("orchestrator.chief_of_staff.run_pipeline")
    def test_all_virtual_targets_escalate(self, mock_pipeline):
        outputs = self._mock_pipeline_success()
        outputs["prioritization_result"]["targets"] = [
            {"target_gene": f"GENE{i}", "rank": i, "target_score": 0.1,
             "evidence_tier": "provisional_virtual", "safety_flags": []}
            for i in range(1, 6)
        ]
        mock_pipeline.return_value = outputs
        from orchestrator.pi_orchestrator import analyze_disease
        result = analyze_disease("coronary artery disease")
        assert result["n_escalations"] >= 1

    def test_review_single_edge_delegates_to_scientific_reviewer(self):
        from orchestrator.pi_orchestrator import review_single_edge
        result = review_single_edge(VALID_TIER1_EDGE)
        assert "decision" in result
        assert "warnings" in result

    @patch("mcp_servers.graph_db_server.query_graph_for_disease")
    def test_check_contradictions_delegates_to_contradiction_agent(self, mock_query):
        mock_query.return_value = {"edges": []}
        from orchestrator.pi_orchestrator import check_graph_contradictions
        result = check_graph_contradictions([VALID_TIER1_EDGE], "cad")
        assert "n_contradictions_found" in result
        assert "graph_integrity_score" in result
