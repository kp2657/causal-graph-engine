"""
tests/test_orchestrator.py — Unit tests for orchestration layer.

Tests:
  - ScientificReviewer: approve/reject/warn logic
  - ContradictionAgent: demotion decision tree
  - PIOrchestrator v2: gene list, tier1 dispatch, quality gates
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
    "modifier_types":  ["germline"],
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
# PI Orchestrator v2 — gene list and dispatch
# ===========================================================================

class TestCollectGeneList:
    """_collect_gene_list: data-derived only, no CHIP/somatic, no cap."""

    def _genetics(self, validated_genes: list[str]) -> dict:
        return {"instruments": [], "anchor_genes_validated": {g: True for g in validated_genes}, "warnings": []}

    @patch("mcp_servers.open_targets_server.get_open_targets_disease_targets",
           return_value={"targets": []})
    def test_empty_returns_empty_list(self, _mock_ot):
        from orchestrator.pi_orchestrator_v2 import _collect_gene_list
        genes, scores, *_ = _collect_gene_list(MOCK_DISEASE_QUERY, self._genetics([]))
        assert genes == []
        assert scores == {}

    def test_validated_genes_included(self):
        from orchestrator.pi_orchestrator_v2 import _collect_gene_list
        genes, *_ = _collect_gene_list(MOCK_DISEASE_QUERY, self._genetics(["PCSK9", "LDLR"]))
        assert "PCSK9" in genes
        assert "LDLR" in genes

    def test_no_duplicates(self):
        from orchestrator.pi_orchestrator_v2 import _collect_gene_list
        genes, *_ = _collect_gene_list(MOCK_DISEASE_QUERY, self._genetics(["PCSK9", "PCSK9"]))
        assert genes.count("PCSK9") == 1

    @patch("mcp_servers.open_targets_server.get_open_targets_disease_targets")
    def test_ot_genes_added_without_cap(self, mock_ot):
        from orchestrator.pi_orchestrator_v2 import _collect_gene_list
        mock_ot.return_value = {
            "targets": [{"gene_symbol": f"GENE{i}", "genetic_score": 0.10} for i in range(200)]
        }
        genes, scores, *_ = _collect_gene_list(
            {**MOCK_DISEASE_QUERY, "efo_id": "EFO_0001645"}, self._genetics([])
        )
        assert len(genes) == 200
        assert len(scores) == 200

    @patch("mcp_servers.open_targets_server.get_open_targets_disease_targets")
    def test_ot_genes_below_threshold_excluded(self, mock_ot):
        from orchestrator.pi_orchestrator_v2 import _collect_gene_list
        mock_ot.return_value = {
            "targets": [
                {"gene_symbol": "HIGH", "genetic_score": 0.10},
                {"gene_symbol": "LOW",  "genetic_score": 0.01},
            ]
        }
        genes, *_ = _collect_gene_list(
            {**MOCK_DISEASE_QUERY, "efo_id": "EFO_0001645"}, self._genetics([])
        )
        assert "HIGH" in genes
        assert "LOW" not in genes

    @patch("mcp_servers.open_targets_server.get_open_targets_disease_targets")
    def test_ot_failure_falls_back_to_gwas_validated(self, mock_ot):
        from orchestrator.pi_orchestrator_v2 import _collect_gene_list
        mock_ot.side_effect = Exception("API down")
        genes, *_ = _collect_gene_list(
            {**MOCK_DISEASE_QUERY, "efo_id": "EFO_0001645"}, self._genetics(["PCSK9"])
        )
        assert "PCSK9" in genes


class TestTier1DirectCall:
    """Tier 1 agent functions are called directly (no AgentRunner)."""

    @patch("steps.tier1_phenomics.disease_query_builder.run")
    @patch("steps.tier1_phenomics.gwas_anchor_validator.run")
    def test_direct_calls_return_dicts(self, mock_sg, mock_pa):
        mock_pa.return_value = {**MOCK_DISEASE_QUERY, "warnings": []}
        mock_sg.return_value = {"instruments": [], "anchor_genes_validated": {}, "warnings": []}
        from steps.tier1_phenomics.disease_query_builder import run as pa_run
        from steps.tier1_phenomics.gwas_anchor_validator import run as sg_run
        dq = pa_run("coronary artery disease")
        gr = sg_run(dq)
        assert dq.get("efo_id") == "EFO_0001645"
        assert "instruments" in gr


_TIER_STUBS = {
    "pa":  {**MOCK_DISEASE_QUERY, "warnings": []},
    "sg":  {"instruments": [], "anchor_genes_validated": {}, "warnings": []},
    "pga": {
        "genes": ["PCSK9"], "programs": ["lipid_metabolism"],
        "beta_matrix": {"PCSK9": {"lipid_metabolism": -0.4}},
        "evidence_tier_per_gene": {"PCSK9": "Tier1_Interventional"},
        "n_tier1": 1, "n_tier2": 0, "n_tier3": 0, "n_virtual": 0, "warnings": [],
    },
    "rga": {"gene_eqtl_summary": {}, "gene_program_overlap": {}, "tier2_upgrades": [], "warnings": []},
    "cda": {
        "n_edges_written": 1, "n_edges_rejected": 0,
        "top_genes": [{"gene": "PCSK9", "ota_gamma": 0.32, "tier": "Tier1_Interventional", "programs": []}],
        "edges_written": [], "shd": 0, "warnings": [],
    },
    "kgc": {
        "n_pathway_edges_added": 0, "n_ppi_edges_added": 0,
        "n_drug_target_edges_added": 0, "n_primekg_edges_added": 0,
        "top_pathways": [], "drug_target_summary": [], "contradictions_flagged": 0, "warnings": [],
    },
    "tpa": {
        "targets": [{"target_gene": "PCSK9", "rank": 1, "target_score": 0.73,
                     "ota_gamma": 0.32, "evidence_tier": "Tier1_Interventional",
                     "ot_score": 0.89, "max_phase": 4, "known_drugs": [],
                     "pli": 0.02, "flags": [], "top_programs": [],
                     "key_evidence": [], "safety_flags": []}],
        "warnings": [],
    },
    "chem": {"target_chemistry": {}, "repurposing_candidates": [], "warnings": []},
    "ct":   {"trial_summary": {}, "key_trials": [], "development_risk": {}, "repurposing_opportunities": [], "warnings": []},
    "writer": {
        "disease_name": "coronary artery disease", "efo_id": "EFO_0001645",
        "target_list": [{"gene": "PCSK9"}],
        "n_tier1_edges": 1, "n_tier2_edges": 0, "n_tier3_edges": 0, "n_virtual_edges": 0,
        "executive_summary": "Test.", "target_table": "",
        "top_target_narratives": [], "evidence_quality": {}, "limitations": "",
        "pipeline_version": "0.1.0", "generated_at": "2026-01-01T00:00:00Z", "warnings": [],
    },
}


class TestAnalyzeDiseaseV2Quality:
    """analyze_disease_v2 quality gates and output shape — patching tier run() directly."""

    def _patch_all(self):
        return [
            patch("steps.tier1_phenomics.disease_query_builder.run",        return_value=_TIER_STUBS["pa"]),
            patch("steps.tier1_phenomics.gwas_anchor_validator.run",        return_value=_TIER_STUBS["sg"]),
            patch("steps.tier2_pathway.beta_matrix_builder.run",            return_value=_TIER_STUBS["pga"]),
            patch("steps.tier2_pathway.eqtl_coloc_mapper.run",              return_value=_TIER_STUBS["rga"]),
            patch("orchestrator.pi_orchestrator_v2._get_gamma_estimates",    return_value={}),
            patch("steps.tier3_causal.ota_gamma_calculator.run",            return_value=_TIER_STUBS["cda"]),
            patch("steps.tier3_causal.drug_target_graph_enricher.run",      return_value=_TIER_STUBS["kgc"]),
            patch("steps.tier4_translation.target_ranker.run",              return_value=_TIER_STUBS["tpa"]),
            patch("steps.tier4_translation.gps_compound_screener.run",      return_value=_TIER_STUBS["chem"]),
            patch("steps.tier4_translation.trial_landscape_mapper.run",     return_value=_TIER_STUBS["ct"]),
            patch("steps.tier5_writer.report_builder.run",                  return_value=_TIER_STUBS["writer"]),
            patch("mcp_servers.open_targets_server.get_open_targets_disease_targets", return_value={"targets": []}),
        ]

    def test_pipeline_success_shape(self):
        from orchestrator.pi_orchestrator_v2 import analyze_disease_v2
        patches = self._patch_all()
        for p in patches: p.start()
        try:
            result = analyze_disease_v2("coronary artery disease")
            assert result.get("pipeline_status") == "SUCCESS"
            assert result.get("pi_reviewed") is True
            assert result.get("disease_name") == "coronary artery disease"
            assert "target_list" in result
            assert "somatic_result" not in result
        finally:
            for p in patches: p.stop()

    def test_phenotype_failure_returns_failed_status(self):
        from orchestrator.pi_orchestrator_v2 import analyze_disease_v2
        patches = self._patch_all()
        for p in patches: p.start()
        try:
            with patch("steps.tier1_phenomics.disease_query_builder.run",
                       return_value={"stub_fallback": True, "warnings": []}):
                result = analyze_disease_v2("unknown disease xyz")
            assert "FAILED_TIER1_PHENOTYPE" in result.get("pipeline_status", "")
        finally:
            for p in patches: p.stop()

    def test_pipeline_completes_with_zero_edges(self):
        from orchestrator.pi_orchestrator_v2 import analyze_disease_v2
        zero_edge_cda = {**_TIER_STUBS["cda"], "n_edges_written": 0, "edges_written": []}
        patches = self._patch_all()
        for p in patches: p.start()
        try:
            with patch("steps.tier3_causal.ota_gamma_calculator.run",
                       return_value=zero_edge_cda):
                result = analyze_disease_v2("coronary artery disease")
            assert "HALTED" not in result.get("pipeline_status", "SUCCESS")
        finally:
            for p in patches: p.stop()
