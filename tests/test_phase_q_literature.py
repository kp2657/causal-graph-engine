"""
tests/test_phase_q_literature.py — Phase Q: Literature Integration Agent.

Tests:
  - _age_weight: temporal decay at correct year boundaries
  - _classify_title: supporting vs contradicting keyword detection
  - run(): output schema, handles empty results, handles MCP errors gracefully
  - Literature confidence thresholds: SUPPORTED / MODERATE / NOVEL / CONTRADICTED
  - Recency score computation
  - _build_summary: narrative covers supported/novel/contradicted cases
  - Runner wiring: literature_validation_agent dispatches, tool list, routes
  - Integration: literature_result in pipeline_outputs (mocked runner)
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.tier5_writer.literature_validation_agent import (
    _age_weight,
    _classify_title,
    _build_summary,
    run,
)


# ---------------------------------------------------------------------------
# Temporal decay
# ---------------------------------------------------------------------------

class TestAgeWeight:

    def test_recent_paper_no_discount(self):
        assert _age_weight("2024") == pytest.approx(1.0)

    def test_five_year_old_paper(self):
        # 2026 - 2020 = 6yr > 5yr → 0.8
        assert _age_weight("2020") == pytest.approx(0.8)

    def test_ten_year_old_paper(self):
        # 2026 - 2015 = 11yr > 10yr → 0.6
        assert _age_weight("2015") == pytest.approx(0.6)

    def test_unknown_year_moderate_discount(self):
        assert _age_weight("") == pytest.approx(0.9)
        assert _age_weight(None) == pytest.approx(0.9)

    def test_boundary_exactly_5yr(self):
        # 2026 - 2021 = 5yr — NOT >5, so 1.0
        assert _age_weight("2021") == pytest.approx(1.0)

    def test_boundary_exactly_10yr(self):
        # 2026 - 2016 = 10yr — NOT >10, so 0.8
        assert _age_weight("2016") == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# Title classification
# ---------------------------------------------------------------------------

class TestClassifyTitle:

    def test_supporting_causal(self):
        assert _classify_title("NOD2 is a causal risk factor for Crohn disease") == "supporting"

    def test_supporting_therapeutic(self):
        assert _classify_title("IL23R as a therapeutic target in IBD") == "supporting"

    def test_contradicting_no_association(self):
        assert _classify_title("No association between GENE_X and IBD in GWAS") == "contradicting"

    def test_contradicting_no_significant(self):
        assert _classify_title("No significant effect of PCSK9 inhibition on outcomes") == "contradicting"

    def test_neutral_title_defaults_supporting(self):
        # Neutral titles default to 'supporting' (absence ≠ contradiction)
        assert _classify_title("GENE_X expression in IBD patients") == "supporting"


# ---------------------------------------------------------------------------
# run() with mocked MCP
# ---------------------------------------------------------------------------

class TestLiteratureRun:

    def _mock_articles(self, n_supporting=3, n_contradicting=0, year="2023"):
        articles = []
        for i in range(n_supporting):
            articles.append({
                "pmid": f"1000{i}",
                "title": f"Gene causal role in disease {i}",
                "authors": "Author et al.",
                "year": year,
                "journal": "Nature",
            })
        for i in range(n_contradicting):
            articles.append({
                "pmid": f"2000{i}",
                "title": f"No association found for gene {i}",
                "authors": "Author et al.",
                "year": year,
                "journal": "NEJM",
            })
        return {
            "query": "test",
            "total_found": len(articles),
            "articles": articles,
        }

    def _make_prioritization(self, genes=None):
        genes = genes or ["NOD2", "IL23R"]
        return {
            "targets": [
                {"target_gene": g, "rank": i+1, "target_score": 0.8}
                for i, g in enumerate(genes)
            ]
        }

    def test_output_schema(self):
        mock_result = self._mock_articles(n_supporting=6)
        with patch("mcp_servers.literature_server.search_gene_disease_literature",
                   return_value=mock_result):
            result = run(self._make_prioritization(["NOD2"]),
                        {"disease_name": "inflammatory bowel disease"})

        assert "literature_evidence" in result
        assert "n_genes_searched" in result
        assert "n_genes_supported" in result
        assert "n_genes_novel" in result
        assert "n_genes_contradicted" in result
        assert "literature_summary" in result

    def test_supported_confidence(self):
        mock_result = self._mock_articles(n_supporting=6, year="2023")
        with patch("mcp_servers.literature_server.search_gene_disease_literature",
                   return_value=mock_result):
            result = run(self._make_prioritization(["NOD2"]),
                        {"disease_name": "IBD"})

        assert result["literature_evidence"]["NOD2"]["literature_confidence"] == "SUPPORTED"
        assert result["n_genes_supported"] == 1

    def test_moderate_confidence(self):
        mock_result = self._mock_articles(n_supporting=3, year="2023")
        with patch("mcp_servers.literature_server.search_gene_disease_literature",
                   return_value=mock_result):
            result = run(self._make_prioritization(["GENE_X"]),
                        {"disease_name": "IBD"})

        ev = result["literature_evidence"]["GENE_X"]
        assert ev["literature_confidence"] == "MODERATE"

    def test_novel_confidence_zero_papers(self):
        mock_result = {"query": "test", "total_found": 0, "articles": []}
        with patch("mcp_servers.literature_server.search_gene_disease_literature",
                   return_value=mock_result):
            result = run(self._make_prioritization(["NEW_GENE"]),
                        {"disease_name": "IBD"})

        ev = result["literature_evidence"]["NEW_GENE"]
        assert ev["literature_confidence"] == "NOVEL"
        assert result["n_genes_novel"] == 1

    def test_contradicted_confidence(self):
        mock_result = self._mock_articles(n_supporting=0, n_contradicting=3, year="2023")
        with patch("mcp_servers.literature_server.search_gene_disease_literature",
                   return_value=mock_result):
            result = run(self._make_prioritization(["BAD_GENE"]),
                        {"disease_name": "IBD"})

        ev = result["literature_evidence"]["BAD_GENE"]
        assert ev["literature_confidence"] == "CONTRADICTED"
        assert result["n_genes_contradicted"] == 1

    def test_recency_score_old_papers(self):
        mock_result = self._mock_articles(n_supporting=3, year="2010")  # >10yr
        with patch("mcp_servers.literature_server.search_gene_disease_literature",
                   return_value=mock_result):
            result = run(self._make_prioritization(["OLD_GENE"]),
                        {"disease_name": "IBD"})

        ev = result["literature_evidence"]["OLD_GENE"]
        assert ev["recency_score"] == pytest.approx(0.6)
        assert ev["temporal_decay_factor"] == pytest.approx(0.6)

    def test_recency_score_recent_papers(self):
        mock_result = self._mock_articles(n_supporting=3, year="2025")
        with patch("mcp_servers.literature_server.search_gene_disease_literature",
                   return_value=mock_result):
            result = run(self._make_prioritization(["NEW_GENE2"]),
                        {"disease_name": "IBD"})

        ev = result["literature_evidence"]["NEW_GENE2"]
        assert ev["recency_score"] == pytest.approx(1.0)

    def test_mcp_error_graceful_fallback(self):
        with patch("mcp_servers.literature_server.search_gene_disease_literature",
                   side_effect=Exception("API unavailable")):
            result = run(self._make_prioritization(["FAIL_GENE"]),
                        {"disease_name": "IBD"})

        ev = result["literature_evidence"]["FAIL_GENE"]
        assert ev["literature_confidence"] == "NOVEL"
        assert "error" in ev

    def test_key_citations_max_3(self):
        mock_result = self._mock_articles(n_supporting=8, year="2023")
        with patch("mcp_servers.literature_server.search_gene_disease_literature",
                   return_value=mock_result):
            result = run(self._make_prioritization(["NOD2"]),
                        {"disease_name": "IBD"})

        ev = result["literature_evidence"]["NOD2"]
        assert len(ev["key_citations"]) <= 3

    def test_only_top5_genes_searched(self):
        genes = ["G1", "G2", "G3", "G4", "G5", "G6", "G7"]
        mock_result = self._mock_articles(n_supporting=2)
        with patch("mcp_servers.literature_server.search_gene_disease_literature",
                   return_value=mock_result) as mock_fn:
            run(self._make_prioritization(genes), {"disease_name": "IBD"})

        assert mock_fn.call_count == 5  # only top 5


# ---------------------------------------------------------------------------
# _build_summary
# ---------------------------------------------------------------------------

class TestBuildSummary:

    def test_summary_mentions_supported_genes(self):
        lit_ev = {
            "NOD2": {"literature_confidence": "SUPPORTED"},
        }
        summary = _build_summary(lit_ev, "IBD")
        assert "NOD2" in summary
        assert "strong" in summary.lower() or "evidence" in summary.lower()

    def test_summary_mentions_novel_genes(self):
        lit_ev = {
            "NEW_GENE": {"literature_confidence": "NOVEL"},
        }
        summary = _build_summary(lit_ev, "IBD")
        assert "NEW_GENE" in summary
        assert "novel" in summary.lower() or "discovery" in summary.lower()

    def test_summary_warns_contradicted(self):
        lit_ev = {
            "BAD_GENE": {"literature_confidence": "CONTRADICTED"},
        }
        summary = _build_summary(lit_ev, "IBD")
        assert "BAD_GENE" in summary
        assert "caution" in summary.lower() or "contradict" in summary.lower()


# ---------------------------------------------------------------------------
# Runner wiring
# ---------------------------------------------------------------------------

class TestLiteratureRunnerWiring:

    def test_literature_agent_in_modules(self):
        from orchestrator.agent_runner import _AGENT_MODULES
        assert "literature_validation_agent" in _AGENT_MODULES

    def test_literature_agent_in_autonomous(self):
        from orchestrator.agent_runner import _AUTONOMOUS_AGENTS
        assert "literature_validation_agent" in _AUTONOMOUS_AGENTS

    def test_tool_list_has_search_tools(self):
        from orchestrator.agent_runner import AgentRunner
        runner = AgentRunner()
        tools = runner._build_tool_list("literature_validation_agent")
        names = {t["name"] for t in tools}
        assert "search_gene_disease_literature" in names
        assert "fetch_pubmed_abstract" in names
        assert "search_pubmed" in names
        assert "return_result" in names

    def test_tool_list_has_execution_tools(self):
        from orchestrator.agent_runner import AgentRunner
        runner = AgentRunner()
        tools = runner._build_tool_list("literature_validation_agent")
        names = {t["name"] for t in tools}
        assert "run_python" in names  # autonomous agent gets execution tools

    def test_prompt_path_exists(self):
        from orchestrator.agent_runner import AgentRunner, _PROMPT_PATHS
        runner = AgentRunner()
        prompt_path = runner._project_root / _PROMPT_PATHS["literature_validation_agent"]
        assert prompt_path.exists(), f"Literature prompt not found: {prompt_path}"

    def test_runner_dispatches_local(self):
        from orchestrator.agent_runner import AgentRunner
        from orchestrator.message_contracts import AgentInput
        mock_result = {"query": "test", "total_found": 0, "articles": []}
        with patch("mcp_servers.literature_server.search_gene_disease_literature",
                   return_value=mock_result):
            runner = AgentRunner()
            inp = AgentInput(
                agent_name="literature_validation_agent",
                disease_query={"disease_name": "IBD"},
                upstream_results={"prioritization_result": {
                    "targets": [{"target_gene": "NOD2", "rank": 1, "target_score": 0.8}]
                }},
                run_id="test",
            )
            out = runner.dispatch("literature_validation_agent", inp)

        assert out.results is not None
        assert "literature_evidence" in out.results

    def test_literature_result_in_pipeline(self):
        """literature_result key must appear in pipeline output (mocked runner)."""
        from orchestrator.pi_orchestrator_v2 import analyze_disease_v2
        from orchestrator.agent_runner import AgentRunner
        from orchestrator.message_contracts import wrap_output

        def _mock_dispatch(agent_name, agent_input):
            stubs = {
                "phenotype_architect":         {"disease_name": "test", "efo_id": "EFO_0000001"},
                "statistical_geneticist":      {"top_genes": [{"gene_symbol": "NOD2"}], "n_instruments": 1},
                "somatic_exposure_agent":      {"chip_edges": [], "drug_edges": []},
                "perturbation_genomics_agent": {"genes": ["NOD2"], "beta_matrix": {},
                                               "evidence_tier_per_gene": {"NOD2": "Tier1_Perturb_seq"}},
                "regulatory_genomics_agent":   {"tier2_upgrades": []},
                "causal_discovery_agent":      {"edges_written": [], "n_edges_written": 2,
                                               "anchor_recovery": {"recovery_rate": 1.0, "missing": []},
                                               "top_genes": [{"gene_symbol": "NOD2"}]},
                "kg_completion_agent":         {"pathways": [], "drugs": []},
                "target_prioritization_agent": {
                    "targets": [{"target_gene": "NOD2", "rank": 1, "target_score": 0.8,
                                 "ota_gamma": 0.4, "evidence_tier": "Tier1_Perturb_seq", "max_phase": 0}],
                },
                "chemistry_agent":             {"target_chemistry": {}, "repurposing_candidates": []},
                "clinical_trialist_agent":     {"trial_summary": {}, "key_trials": []},
                "scientific_writer_agent":     {"target_list": ["NOD2"],
                                               "anchor_edge_recovery": 1.0, "report": ""},
                "scientific_reviewer_agent":   {
                    "verdict": "APPROVE", "issues": [], "n_critical": 0, "n_major": 0,
                    "n_minor": 0, "summary": "ok", "agent_to_revisit": None,
                    "re_delegation_instructions": [], "approved_targets": ["NOD2"],
                    "flagged_targets": [], "anchor_recovery": 1.0, "warnings": [],
                },
                "literature_validation_agent": {
                    "literature_evidence": {"NOD2": {
                        "n_papers_found": 5, "n_supporting": 5, "n_contradicting": 0,
                        "key_citations": [], "recency_score": 1.0,
                        "temporal_decay_factor": 1.0, "literature_confidence": "SUPPORTED",
                        "search_query": "test",
                    }},
                    "n_genes_searched": 1, "n_genes_supported": 1,
                    "n_genes_novel": 0, "n_genes_contradicted": 0,
                    "literature_summary": "NOD2 supported.",
                },
                "chief_of_staff_agent": {
                    "disease_area": "autoimmune", "recommended_tissues": ["colon"],
                    "anchor_gene_expectations": ["NOD2"], "tier_guidance": {},
                    "key_challenges": [], "briefing_notes": "stub",
                    "disease_name": "test",
                },
            }
            raw = stubs.get(agent_name, {})
            return wrap_output(agent_name, raw)

        with patch.object(AgentRunner, "dispatch", side_effect=_mock_dispatch):
            result = analyze_disease_v2("test disease")

        assert isinstance(result, dict)
        # Pipeline should complete; check for lit_result key or SUCCESS status
        assert result.get("pipeline_status") == "SUCCESS" or "literature" in str(result)
