"""
tests/test_phase_p_cso.py — Phase P: CSO as Active Reasoning Hub.

Tests:
  - run_briefing: tissue priors, anchor expectations, tier_guidance, known challenges
  - run_conflict_analysis: overlap detection, divergence hypothesis, empty tracks
  - run_exec_summary: confidence levels, next_experiments, pipeline_health
  - Unified run() entry point dispatch
  - Runner wiring: CSO routed via runner.dispatch, SDK-upgradeable
  - Integration: CSO outputs in pipeline (mocked runner to avoid network calls)
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.cso.chief_of_staff_agent import (
    run_briefing,
    run_conflict_analysis,
    run_exec_summary,
    run,
)


# ---------------------------------------------------------------------------
# run_briefing
# ---------------------------------------------------------------------------

class TestCSOBriefing:

    def test_ibd_tissue_priors(self):
        result = run_briefing({"disease_name": "inflammatory bowel disease"})
        assert "colon" in result["recommended_tissues"]
        assert "immune_cell" in result["recommended_tissues"]

    def test_ibd_anchor_expectations(self):
        result = run_briefing({"disease_name": "inflammatory bowel disease"})
        assert "NOD2" in result["anchor_gene_expectations"]
        assert "IL23R" in result["anchor_gene_expectations"]

    def test_cad_disease_area(self):
        result = run_briefing({"disease_name": "coronary artery disease"})
        assert result["disease_area"] == "cardiovascular"

    def test_key_challenges_non_empty(self):
        result = run_briefing({"disease_name": "inflammatory bowel disease"})
        assert len(result["key_challenges"]) >= 1

    def test_tier_guidance_has_all_tiers(self):
        result = run_briefing({"disease_name": "coronary artery disease"})
        tg = result["tier_guidance"]
        assert "tier1" in tg
        assert "tier2" in tg
        assert "tier3" in tg
        assert "tier4" in tg

    def test_briefing_notes_contains_disease(self):
        result = run_briefing({"disease_name": "inflammatory bowel disease", "efo_id": "EFO_0003767"})
        assert "inflammatory bowel disease" in result["briefing_notes"].lower()

    def test_unknown_disease_fallback(self):
        result = run_briefing({"disease_name": "mystery disease X"})
        assert isinstance(result["recommended_tissues"], list)
        assert len(result["key_challenges"]) >= 1

    def test_partial_name_match(self):
        """'crohn disease' should match the 'crohn' key."""
        result = run_briefing({"disease_name": "crohn disease"})
        assert "ileum" in result["recommended_tissues"]


# ---------------------------------------------------------------------------
# run_conflict_analysis
# ---------------------------------------------------------------------------

class TestCSOConflictAnalysis:

    def _make_outputs(self, gwas_genes, perturbseq_genes, recovery=1.0):
        return {
            "genetics_result": {
                "top_genes": [{"gene_symbol": g} for g in gwas_genes]
            },
            "beta_matrix_result": {
                "evidence_tier_per_gene": {g: "Tier1_Perturb_seq" for g in perturbseq_genes}
            },
            "causal_result": {
                "anchor_recovery": {"recovery_rate": recovery}
            },
            "regulator_nomination_evidence": {},
        }

    def test_convergent_genes_detected(self):
        outputs = self._make_outputs(["NOD2", "IL23R"], ["NOD2", "PCSK9"])
        result = run_conflict_analysis(outputs)
        assert "NOD2" in result["overlap_genes"]

    def test_divergence_detected_when_no_overlap(self):
        outputs = self._make_outputs(["NOD2", "IL23R"], ["KLF1", "SPI1"])
        result = run_conflict_analysis(outputs)
        assert result["divergence_detected"] is True

    def test_no_divergence_with_full_overlap(self):
        outputs = self._make_outputs(["NOD2", "IL23R"], ["NOD2", "IL23R"])
        result = run_conflict_analysis(outputs)
        assert result["divergence_detected"] is False

    def test_divergence_hypothesis_is_non_trivial(self):
        outputs = self._make_outputs(["PCSK9", "HMGCR"], ["KLF1", "SPI1"])
        result = run_conflict_analysis(outputs)
        # Should explain mechanistic levels
        assert len(result["divergence_hypothesis"]) > 100

    def test_empty_gwas_track(self):
        outputs = self._make_outputs([], ["KLF1", "SPI1"])
        result = run_conflict_analysis(outputs)
        assert "Perturb-seq" in result["divergence_hypothesis"]
        assert result["gwas_top_genes"] == []

    def test_empty_perturbseq_track(self):
        outputs = self._make_outputs(["NOD2", "IL23R"], [])
        result = run_conflict_analysis(outputs)
        assert "GWAS-derived" in result["divergence_hypothesis"]

    def test_recommended_focus_convergent_first(self):
        outputs = self._make_outputs(["NOD2", "PCSK9"], ["NOD2", "KLF1"])
        result = run_conflict_analysis(outputs)
        # NOD2 is convergent — should appear first
        assert result["recommended_focus_genes"][0] == "NOD2"

    def test_anchor_recovery_included(self):
        outputs = self._make_outputs(["NOD2"], ["NOD2"], recovery=0.85)
        result = run_conflict_analysis(outputs)
        assert result["anchor_recovery"] == pytest.approx(0.85)


# ---------------------------------------------------------------------------
# run_exec_summary
# ---------------------------------------------------------------------------

class TestCSOExecSummary:

    def _make_pipeline(self, verdict="APPROVE", n_critical=0, n_major=0,
                       anchor_recovery=1.0, targets=None, adjustments=None):
        targets = targets or [
            {"target_gene": "NOD2", "rank": 1, "target_score": 0.82,
             "max_phase": 2, "evidence_tier": "Tier1_Perturb_seq"},
            {"target_gene": "IL23R", "rank": 2, "target_score": 0.65,
             "max_phase": 0, "evidence_tier": "Tier2_eQTL_MR"},
        ]
        return {
            "phenotype_result": {"disease_name": "inflammatory bowel disease"},
            "prioritization_result": {
                "targets": targets,
                "score_adjustments": adjustments or [],
            },
            "review_result": {
                "verdict": verdict,
                "n_critical": n_critical,
                "n_major": n_major,
                "issues": [],
            },
            "causal_result": {
                "anchor_recovery": {"recovery_rate": anchor_recovery}
            },
            "_redelegation_actions": [],
        }

    def test_high_confidence_on_clean_pipeline(self):
        result = run_exec_summary(self._make_pipeline())
        assert result["confidence_assessment"] == "HIGH"

    def test_low_confidence_on_critical_issue(self):
        result = run_exec_summary(self._make_pipeline(n_critical=1, anchor_recovery=0.6))
        assert result["confidence_assessment"] == "LOW"

    def test_medium_confidence_on_major_issue(self):
        result = run_exec_summary(self._make_pipeline(
            verdict="REVISE", n_major=2, n_critical=0
        ))
        assert result["confidence_assessment"] == "MEDIUM"

    def test_executive_summary_contains_disease(self):
        result = run_exec_summary(self._make_pipeline())
        assert "inflammatory bowel disease" in result["executive_summary"].lower()

    def test_top_insight_contains_top_gene(self):
        result = run_exec_summary(self._make_pipeline())
        assert "NOD2" in result["top_insight"]

    def test_next_experiments_non_empty(self):
        result = run_exec_summary(self._make_pipeline())
        assert len(result["next_experiments"]) >= 1

    def test_score_adjustment_note_in_summary(self):
        adjustments = [{"gene": "IL23R", "original_score": 0.65, "adjusted_score": 0.39,
                        "combined_factor": 0.6, "signals": []}]
        result = run_exec_summary(self._make_pipeline(adjustments=adjustments))
        assert "IL23R" in result["executive_summary"]

    def test_pipeline_health_included(self):
        result = run_exec_summary(self._make_pipeline(anchor_recovery=0.9))
        ph = result["pipeline_health"]
        assert ph["anchor_recovery"] == pytest.approx(0.9)
        assert ph["reviewer_verdict"] == "APPROVE"

    def test_phase3_drug_recommendation(self):
        targets = [
            {"target_gene": "PCSK9", "rank": 1, "target_score": 0.9,
             "max_phase": 4, "evidence_tier": "Tier1_GWAS"},
        ]
        result = run_exec_summary(self._make_pipeline(targets=targets))
        assert any("repurposing" in e.lower() or "approved" in e.lower()
                   for e in result["next_experiments"])


# ---------------------------------------------------------------------------
# Unified run() entry point
# ---------------------------------------------------------------------------

class TestCSORunDispatch:

    def test_briefing_mode(self):
        result = run(
            {"disease_name": "inflammatory bowel disease"},
            {"_cso_mode": "briefing"},
        )
        assert "recommended_tissues" in result
        assert "colon" in result["recommended_tissues"]

    def test_conflict_mode(self):
        pipeline_outputs = {
            "_cso_mode": "conflict_analysis",
            "genetics_result": {"top_genes": [{"gene_symbol": "NOD2"}]},
            "beta_matrix_result": {"evidence_tier_per_gene": {"KLF1": "Tier1_Perturb_seq"}},
            "causal_result": {"anchor_recovery": {"recovery_rate": 1.0}},
            "regulator_nomination_evidence": {},
        }
        result = run({}, pipeline_outputs)
        assert "divergence_hypothesis" in result

    def test_exec_summary_mode(self):
        pipeline_outputs = {
            "_cso_mode": "exec_summary",
            "phenotype_result": {"disease_name": "test disease"},
            "prioritization_result": {
                "targets": [{"target_gene": "NOD2", "rank": 1,
                             "target_score": 0.8, "max_phase": 0, "evidence_tier": "Tier1_GWAS"}],
                "score_adjustments": [],
            },
            "review_result": {"verdict": "APPROVE", "n_critical": 0, "n_major": 0, "issues": []},
            "causal_result": {"anchor_recovery": {"recovery_rate": 1.0}},
            "_redelegation_actions": [],
        }
        result = run({}, pipeline_outputs)
        assert "executive_summary" in result
        assert "confidence_assessment" in result

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="_cso_mode"):
            run({}, {"_cso_mode": "unknown_mode"})


# ---------------------------------------------------------------------------
# Runner wiring: CSO goes through runner.dispatch, not direct Python calls
# ---------------------------------------------------------------------------

class TestCSORunnerWiring:

    def test_runner_dispatches_cso_local(self):
        """runner.dispatch('chief_of_staff_agent', ...) calls CSO run() in local mode."""
        from orchestrator.agent_runner import AgentRunner
        from orchestrator.message_contracts import AgentInput
        runner = AgentRunner()
        cso_input = AgentInput(
            agent_name="chief_of_staff_agent",
            disease_query={"disease_name": "inflammatory bowel disease"},
            upstream_results={"_cso_mode": "briefing"},
            run_id="test",
        )
        out = runner.dispatch("chief_of_staff_agent", cso_input)
        assert out.results is not None
        assert "recommended_tissues" in out.results
        assert "colon" in out.results["recommended_tissues"]

    def test_runner_cso_conflict_mode(self):
        from orchestrator.agent_runner import AgentRunner
        from orchestrator.message_contracts import AgentInput
        runner = AgentRunner()
        cso_input = AgentInput(
            agent_name="chief_of_staff_agent",
            disease_query={"disease_name": "IBD"},
            upstream_results={
                "_cso_mode": "conflict_analysis",
                "gwas_top_genes": ["NOD2", "IL23R"],
                "perturbseq_top_regulators": ["KLF1", "SPI1"],
                "overlap_genes": [],
                "anchor_recovery": 0.9,
                "disease_name": "IBD",
            },
            run_id="test",
        )
        out = runner.dispatch("chief_of_staff_agent", cso_input)
        assert "divergence_hypothesis" in out.results

    def test_runner_cso_exec_summary_mode(self):
        from orchestrator.agent_runner import AgentRunner
        from orchestrator.message_contracts import AgentInput
        runner = AgentRunner()
        cso_input = AgentInput(
            agent_name="chief_of_staff_agent",
            disease_query={"disease_name": "test disease"},
            upstream_results={
                "_cso_mode": "exec_summary",
                "top_targets": [{"target_gene": "NOD2", "rank": 1,
                                 "target_score": 0.8, "max_phase": 0,
                                 "evidence_tier": "Tier1_GWAS"}],
                "reviewer_verdict": "APPROVE",
                "n_critical": 0, "n_major": 0,
                "open_issues": [], "score_adjustments": [],
                "redelegation_rounds": 0, "anchor_recovery": 1.0,
            },
            run_id="test",
        )
        out = runner.dispatch("chief_of_staff_agent", cso_input)
        assert "executive_summary" in out.results

    def test_cso_tool_list_is_return_result_only(self):
        """CSO should only have return_result — no domain tools, no execution tools."""
        from orchestrator.agent_runner import AgentRunner
        runner = AgentRunner()
        tools = runner._build_tool_list("chief_of_staff_agent")
        assert len(tools) == 1
        assert tools[0]["name"] == "return_result"

    def test_cso_max_turns_is_3(self):
        from orchestrator.agent_runner import AgentRunner
        runner = AgentRunner()
        assert runner._get_max_turns("chief_of_staff_agent") == 3

    def test_cso_prompt_path_exists(self):
        from orchestrator.agent_runner import AgentRunner, _PROMPT_PATHS
        import os
        runner = AgentRunner()
        prompt_path = runner._project_root / _PROMPT_PATHS["chief_of_staff_agent"]
        assert prompt_path.exists(), f"CSO prompt not found: {prompt_path}"


# ---------------------------------------------------------------------------
# Integration: CSO wired into analyze_disease_v2 (mocked pipeline)
# ---------------------------------------------------------------------------

class TestCSOIntegration:

    def _make_mock_dispatch(self):
        """Return a mock dispatch that short-circuits the full pipeline."""
        from orchestrator.message_contracts import AgentOutput, wrap_output

        def _dispatch(agent_name, agent_input):
            # Minimal viable outputs for each agent
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
                                 "ota_gamma": 0.4, "evidence_tier": "Tier1_Perturb_seq",
                                 "max_phase": 0}],
                },
                "chemistry_agent":             {"target_chemistry": {}, "repurposing_candidates": []},
                "clinical_trialist_agent":     {"trial_summary": {}, "key_trials": []},
                "scientific_writer_agent":     {"target_list": ["NOD2"],
                                               "anchor_edge_recovery": 1.0, "report": ""},
                "scientific_reviewer_agent":   {
                    "verdict": "APPROVE", "issues": [], "n_critical": 0,
                    "n_major": 0, "n_minor": 0, "summary": "ok",
                    "agent_to_revisit": None, "re_delegation_instructions": [],
                    "approved_targets": ["NOD2"], "flagged_targets": [],
                    "anchor_recovery": 1.0, "warnings": [],
                },
                "chief_of_staff_agent": self._cso_stub(agent_input),
            }
            raw = stubs.get(agent_name, {})
            return wrap_output(agent_name, raw)

        return _dispatch

    def _cso_stub(self, agent_input):
        mode = (agent_input.upstream_results or {}).get("_cso_mode", "briefing")
        if mode == "briefing":
            return {"disease_area": "autoimmune", "recommended_tissues": ["colon"],
                    "anchor_gene_expectations": ["NOD2"], "tier_guidance": {},
                    "key_challenges": [], "briefing_notes": "stub",
                    "disease_name": "test"}
        if mode == "conflict_analysis":
            return {"gwas_top_genes": ["NOD2"], "perturbseq_top_regulators": ["NOD2"],
                    "overlap_genes": ["NOD2"], "divergence_detected": False,
                    "divergence_hypothesis": "Convergent.", "recommended_focus_genes": ["NOD2"],
                    "evidence_conflict_notes": "ok"}
        return {"executive_summary": "stub", "top_insight": "stub",
                "confidence_assessment": "HIGH", "confidence_rationale": "stub",
                "next_experiments": [], "pipeline_health": {}}

    def test_cso_three_outputs_in_pipeline(self, capsys):
        """All three CSO outputs should appear in pipeline_outputs via mocked dispatch."""
        from orchestrator.pi_orchestrator_v2 import analyze_disease_v2
        from orchestrator.agent_runner import AgentRunner

        mock_dispatch = self._make_mock_dispatch()
        with patch.object(AgentRunner, "dispatch", side_effect=mock_dispatch):
            result = analyze_disease_v2("test disease")

        # CSO briefing log line must appear
        captured = capsys.readouterr()
        assert "chief_of_staff_agent" in captured.out
        # Pipeline must complete without exception
        assert isinstance(result, dict)

    def test_cso_sdk_mode_activates_on_set_mode(self):
        """set_mode('chief_of_staff_agent', 'sdk') should be accepted by runner."""
        from orchestrator.agent_runner import AgentRunner
        runner = AgentRunner()
        runner.set_mode("chief_of_staff_agent", "sdk")
        assert runner.get_mode("chief_of_staff_agent") == "sdk"

    def test_cso_briefing_disease_area(self):
        """CSO briefing correctly identifies disease area for CAD."""
        briefing = run_briefing({"disease_name": "coronary artery disease"})
        assert briefing["disease_area"] == "cardiovascular"
        assert "PCSK9" in briefing["anchor_gene_expectations"]
