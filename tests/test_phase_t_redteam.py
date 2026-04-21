"""
tests/test_phase_t_redteam.py — Phase T: Red Team Adversarial Agent + SCONE surfacing.

Tests:
  - red_team_agent.run() output schema
  - confidence_level: HIGH/MODERATE/LOW/REJECTED mapping
  - rank_stability: STABLE/FRAGILE/TIER-DEPENDENT
  - red_team_verdict: PROCEED/CAUTION/DEPRIORITIZE logic
  - counterargument content for bootstrap_rejected target
  - counterargument content for CONTRADICTED literature
  - counterargument content for high-pLI gene
  - literature_flag populated for CONTRADICTED targets
  - n_flagged_caution / n_flagged_deprioritize counts
  - overall_confidence aggregation
  - only top-5 targets assessed (7 targets → 5 assessments)
  - SCONE fields surfaced in scientific_writer_agent target_list
  - Runner wiring: red_team_agent in _AGENT_MODULES, prompt exists, max_turns=3
  - Integration: red_team_result in pipeline outputs (mocked runner)
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.tier5_writer.red_team_agent import (
    run,
    _confidence_level,
    _rank_stability,
    _red_team_verdict,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_target(
    gene: str = "NOD2",
    rank: int = 1,
    tier: str = "Tier1_Interventional",
    scone_confidence: float | None = 0.85,
    scone_flags: list | None = None,
    pli: float | None = None,
    target_score: float = 0.7,
    max_phase: int = 0,
    ci_lower: float | None = 0.10,
    ci_upper: float | None = 0.40,
) -> dict:
    return {
        "target_gene":        gene,
        "rank":               rank,
        "evidence_tier":      tier,
        "scone_confidence":   scone_confidence,
        "scone_flags":        scone_flags or [],
        "pli":                pli,
        "target_score":       target_score,
        "max_phase":          max_phase,
        "ota_gamma":          0.25,
        "ota_gamma_raw":      0.25,
        "ota_gamma_sigma":    0.05,
        "ota_gamma_ci_lower": ci_lower,
        "ota_gamma_ci_upper": ci_upper,
        "flags":              [],
        "top_programs":       [],
        "key_evidence":       [],
        "known_drugs":        [],
        "ot_score":           0.3,
    }


def _make_prioritization(targets: list[dict]) -> dict:
    return {"targets": targets}


def _make_lit_result(gene: str, confidence: str) -> dict:
    return {
        "literature_evidence": {
            gene: {
                "literature_confidence": confidence,
                "n_papers_found": 3,
                "n_supporting": 0 if confidence == "CONTRADICTED" else 3,
                "n_contradicting": 3 if confidence == "CONTRADICTED" else 0,
                "key_citations": [],
                "recency_score": 1.0,
                "temporal_decay_factor": 1.0,
                "search_query": "test",
            }
        }
    }


# ---------------------------------------------------------------------------
# _confidence_level
# ---------------------------------------------------------------------------

class TestConfidenceLevel:

    def test_high(self):
        assert _confidence_level(0.85, []) == "HIGH"

    def test_moderate(self):
        assert _confidence_level(0.65, []) == "MODERATE"

    def test_low(self):
        assert _confidence_level(0.30, []) == "LOW"

    def test_rejected_flag_overrides(self):
        # Even with high scone_confidence, bootstrap_rejected → REJECTED
        assert _confidence_level(0.90, ["bootstrap_rejected"]) == "REJECTED"

    def test_none_defaults_moderate(self):
        assert _confidence_level(None, []) == "MODERATE"

    def test_boundary_high(self):
        assert _confidence_level(0.80, []) == "HIGH"

    def test_boundary_low(self):
        assert _confidence_level(0.50, []) == "MODERATE"  # boundary is ≥ 0.50


# ---------------------------------------------------------------------------
# _rank_stability
# ---------------------------------------------------------------------------

class TestRankStability:

    def test_tier1_high_confidence_stable(self):
        target = _make_target(tier="Tier1_Interventional", scone_confidence=0.90)
        label, _ = _rank_stability(target)
        assert label == "STABLE"

    def test_bootstrap_rejected_fragile(self):
        target = _make_target(scone_flags=["bootstrap_rejected"])
        label, _ = _rank_stability(target)
        assert label == "FRAGILE"

    def test_provisional_virtual_tier_dependent(self):
        target = _make_target(tier="provisional_virtual", scone_confidence=0.70)
        label, _ = _rank_stability(target)
        assert label == "TIER-DEPENDENT"

    def test_low_confidence_fragile(self):
        target = _make_target(
            tier="Tier2_Convergent",
            scone_confidence=0.30,
        )
        label, _ = _rank_stability(target)
        assert label == "FRAGILE"

    def test_rationale_contains_tier(self):
        target = _make_target(tier="provisional_virtual")
        _, rationale = _rank_stability(target)
        assert "provisional_virtual" in rationale


# ---------------------------------------------------------------------------
# _red_team_verdict
# ---------------------------------------------------------------------------

class TestRedTeamVerdict:

    def test_proceed_high_stable(self):
        assert _red_team_verdict("HIGH", "STABLE", "SUPPORTED") == "PROCEED"

    def test_caution_low_confidence(self):
        assert _red_team_verdict("LOW", "TIER-DEPENDENT", "MODERATE") == "CAUTION"

    def test_caution_fragile_stability(self):
        assert _red_team_verdict("MODERATE", "FRAGILE", None) == "CAUTION"

    def test_deprioritize_rejected(self):
        assert _red_team_verdict("REJECTED", "FRAGILE", None) == "DEPRIORITIZE"

    def test_deprioritize_contradicted_literature(self):
        assert _red_team_verdict("HIGH", "STABLE", "CONTRADICTED") == "DEPRIORITIZE"

    def test_proceed_novel_literature(self):
        # NOVEL ≠ CONTRADICTED, should not deprioritise
        assert _red_team_verdict("HIGH", "STABLE", "NOVEL") == "PROCEED"


# ---------------------------------------------------------------------------
# run() output schema
# ---------------------------------------------------------------------------

class TestRedTeamRunSchema:

    def test_required_keys(self):
        targets = [_make_target("NOD2", rank=1)]
        result = run(_make_prioritization(targets))
        assert "red_team_assessments" in result
        assert "n_targets_assessed" in result
        assert "n_flagged_caution" in result
        assert "n_flagged_deprioritize" in result
        assert "overall_confidence" in result
        assert "red_team_summary" in result

    def test_assessment_keys(self):
        targets = [_make_target("NOD2", rank=1)]
        result = run(_make_prioritization(targets))
        a = result["red_team_assessments"][0]
        for key in [
            "target_gene", "rank", "scone_confidence", "scone_flags",
            "ota_gamma_ci_lower", "ota_gamma_ci_upper", "ci_width",
            "confidence_level", "evidence_vulnerability", "counterargument",
            "rank_stability", "rank_stability_rationale", "literature_flag",
            "red_team_verdict", "counterfactual",
        ]:
            assert key in a, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# run() logic
# ---------------------------------------------------------------------------

class TestRedTeamRunLogic:

    def test_only_top5_assessed(self):
        targets = [_make_target(f"GENE{i}", rank=i) for i in range(1, 8)]
        result = run(_make_prioritization(targets))
        assert result["n_targets_assessed"] == 5

    def test_bootstrap_rejected_deprioritize(self):
        target = _make_target("FRAG", rank=1, scone_flags=["bootstrap_rejected"])
        result = run(_make_prioritization([target]))
        assert result["red_team_assessments"][0]["red_team_verdict"] == "DEPRIORITIZE"
        assert result["n_flagged_deprioritize"] == 1

    def test_contradicted_literature_deprioritize(self):
        target = _make_target("BAD_GENE", rank=1)
        lit = _make_lit_result("BAD_GENE", "CONTRADICTED")
        result = run(_make_prioritization([target]), lit)
        assert result["red_team_assessments"][0]["red_team_verdict"] == "DEPRIORITIZE"

    def test_literature_flag_set_for_contradicted(self):
        target = _make_target("BAD_GENE", rank=1)
        lit = _make_lit_result("BAD_GENE", "CONTRADICTED")
        result = run(_make_prioritization([target]), lit)
        lit_flag = result["red_team_assessments"][0]["literature_flag"]
        assert lit_flag is not None
        assert "CONTRADICTED" in lit_flag

    def test_literature_flag_none_for_supported(self):
        target = _make_target("NOD2", rank=1)
        lit = _make_lit_result("NOD2", "SUPPORTED")
        result = run(_make_prioritization([target]), lit)
        assert result["red_team_assessments"][0]["literature_flag"] is None

    def test_high_pli_counterargument(self):
        target = _make_target("ESSENTIAL_GENE", rank=1, pli=0.95)
        result = run(_make_prioritization([target]))
        counterarg = result["red_team_assessments"][0]["counterargument"]
        assert "pLI" in counterarg or "essential" in counterarg.lower()

    def test_ci_width_computed(self):
        target = _make_target("NOD2", rank=1, ci_lower=0.10, ci_upper=0.40)
        result = run(_make_prioritization([target]))
        assert result["red_team_assessments"][0]["ci_width"] == pytest.approx(0.30, abs=1e-4)

    def test_ci_width_none_when_missing(self):
        target = _make_target("NOD2", rank=1, ci_lower=None, ci_upper=None)
        result = run(_make_prioritization([target]))
        assert result["red_team_assessments"][0]["ci_width"] is None

    def test_overall_confidence_high_all_proceed(self):
        targets = [
            _make_target(f"G{i}", rank=i, scone_confidence=0.90,
                         tier="Tier1_Interventional")
            for i in range(1, 4)
        ]
        result = run(_make_prioritization(targets))
        assert result["overall_confidence"] == "HIGH"

    def test_overall_confidence_low_with_deprioritize(self):
        targets = [_make_target("F1", rank=1, scone_flags=["bootstrap_rejected"])]
        result = run(_make_prioritization(targets))
        assert result["overall_confidence"] == "LOW"

    def test_red_team_summary_nonempty(self):
        targets = [_make_target("NOD2", rank=1)]
        result = run(_make_prioritization(targets))
        assert len(result["red_team_summary"]) > 0

    def test_summary_mentions_gene_name(self):
        targets = [_make_target("NOD2", rank=1, scone_confidence=0.90,
                                tier="Tier1_Interventional")]
        result = run(_make_prioritization(targets))
        assert "NOD2" in result["red_team_summary"]

    def test_empty_targets_no_crash(self):
        result = run({"targets": []})
        assert result["n_targets_assessed"] == 0
        assert result["red_team_summary"] == "No targets assessed."


# ---------------------------------------------------------------------------
# Counterfactual integration in red_team_agent
# ---------------------------------------------------------------------------

class TestCounterfactualIntegration:

    def _make_beta_matrix_result(self, gene: str) -> dict:
        return {
            "genes": [gene],
            "beta_matrix": {
                gene: {
                    "PROG_A": {"beta": 0.4, "evidence_tier": "Tier1_Interventional"},
                    "PROG_B": {"beta": 0.2, "evidence_tier": "Tier2_Convergent"},
                }
            },
            "evidence_tier_per_gene": {gene: "Tier1_Interventional"},
        }

    def _make_gamma_estimates(self) -> dict:
        return {
            "PROG_A": {"IBD": {"gamma": 0.5, "evidence_tier": "Tier1_Interventional"}},
            "PROG_B": {"IBD": {"gamma": 0.3, "evidence_tier": "Tier2_Convergent"}},
        }

    def test_counterfactual_key_present(self):
        target = _make_target("NOD2", rank=1)
        result = run(
            _make_prioritization([target]),
            disease_query={"disease_name": "inflammatory bowel disease"},
            beta_matrix_result=self._make_beta_matrix_result("NOD2"),
            gamma_estimates=self._make_gamma_estimates(),
        )
        a = result["red_team_assessments"][0]
        assert "counterfactual" in a

    def test_counterfactual_has_inhibition_and_knockout(self):
        target = _make_target("NOD2", rank=1)
        result = run(
            _make_prioritization([target]),
            disease_query={"disease_name": "inflammatory bowel disease"},
            beta_matrix_result=self._make_beta_matrix_result("NOD2"),
            gamma_estimates=self._make_gamma_estimates(),
        )
        cf = result["red_team_assessments"][0]["counterfactual"]
        assert cf is not None
        assert "inhibition_50pct" in cf
        assert "knockout" in cf
        assert "primary_trait" in cf

    def test_counterfactual_inhibition_reduces_gamma(self):
        target = _make_target("NOD2", rank=1)
        result = run(
            _make_prioritization([target]),
            disease_query={"disease_name": "inflammatory bowel disease"},
            beta_matrix_result=self._make_beta_matrix_result("NOD2"),
            gamma_estimates=self._make_gamma_estimates(),
        )
        inh = result["red_team_assessments"][0]["counterfactual"]["inhibition_50pct"]
        assert inh["delta_gamma"] < 0
        assert inh["percent_change"] == pytest.approx(-50.0, abs=0.1)

    def test_counterfactual_knockout_zeroes_gamma(self):
        target = _make_target("NOD2", rank=1)
        result = run(
            _make_prioritization([target]),
            disease_query={"disease_name": "inflammatory bowel disease"},
            beta_matrix_result=self._make_beta_matrix_result("NOD2"),
            gamma_estimates=self._make_gamma_estimates(),
        )
        ko = result["red_team_assessments"][0]["counterfactual"]["knockout"]
        assert ko["perturbed_gamma"] == pytest.approx(0.0, abs=1e-9)

    def test_counterfactual_none_when_no_beta_matrix(self):
        # No beta_matrix_result → counterfactual should be None
        target = _make_target("NOD2", rank=1)
        result = run(_make_prioritization([target]))
        assert result["red_team_assessments"][0]["counterfactual"] is None

    def test_counterfactual_interpretation_in_output(self):
        target = _make_target("NOD2", rank=1)
        result = run(
            _make_prioritization([target]),
            disease_query={"disease_name": "inflammatory bowel disease"},
            beta_matrix_result=self._make_beta_matrix_result("NOD2"),
            gamma_estimates=self._make_gamma_estimates(),
        )
        inh = result["red_team_assessments"][0]["counterfactual"]["inhibition_50pct"]
        assert "interpretation" in inh
        assert "NOD2" in inh["interpretation"]


# ---------------------------------------------------------------------------
# SCONE fields surfaced in scientific_writer target_list
# ---------------------------------------------------------------------------

class TestSconeFieldsInWriter:

    def test_scone_confidence_in_target_list(self):
        """scientific_writer_agent.run() must propagate scone_confidence to target_list."""
        from agents.tier5_writer.scientific_writer_agent import run as writer_run

        # Build minimal prioritization result with SCONE fields
        targets = [{
            "target_gene":        "NOD2",
            "rank":               1,
            "target_score":       0.75,
            "ota_gamma":          0.30,
            "ota_gamma_raw":      0.30,
            "ota_gamma_sigma":    0.04,
            "ota_gamma_ci_lower": 0.12,
            "ota_gamma_ci_upper": 0.48,
            "scone_confidence":   0.85,
            "scone_flags":        ["anchor_scone_exempt"],
            "evidence_tier":      "Tier1_Interventional",
            "ot_score":           0.4,
            "max_phase":          0,
            "known_drugs":        [],
            "pli":                None,
            "flags":              ["genetic_anchor"],
            "top_programs":       [],
            "key_evidence":       [],
            "safety_flags":       [],
            "brg_score":          None,
        }]
        prio = {"targets": targets, "warnings": []}

        result = writer_run(
            phenotype_result={"disease_name": "IBD", "efo_id": "EFO_0003767"},
            genetics_result={"top_genes": [{"gene_symbol": "NOD2"}], "n_instruments": 1},
            somatic_result={},
            beta_matrix_result={"genes": ["NOD2"], "beta_matrix": {}, "evidence_tier_per_gene": {}},
            regulatory_result={},
            causal_result={
                "top_genes": [{"gene": "NOD2", "ota_gamma": 0.3, "tier": "Tier1_Interventional"}],
                "edges_written": [],
                "n_edges_written": 1,
                "anchor_recovery": {"recovery_rate": 1.0, "missing": []},
            },
            kg_result={},
            prioritization_result=prio,
            chemistry_result={},
            trials_result={},
        )

        tl = result.get("target_list", [])
        assert len(tl) > 0, "target_list is empty"
        rec = tl[0]
        assert "scone_confidence" in rec
        assert rec["scone_confidence"] == pytest.approx(0.85)
        assert "scone_flags" in rec
        assert rec["ota_gamma_ci_lower"] == pytest.approx(0.12)
        assert rec["ota_gamma_ci_upper"] == pytest.approx(0.48)


# ---------------------------------------------------------------------------
# Runner wiring
# ---------------------------------------------------------------------------

class TestRedTeamRunnerWiring:

    def test_red_team_agent_in_modules(self):
        from orchestrator.agent_runner import _AGENT_MODULES
        assert "red_team_agent" in _AGENT_MODULES

    def test_red_team_agent_prompt_exists(self):
        from orchestrator.agent_runner import _PROMPT_PATHS, AgentRunner
        runner = AgentRunner()
        path = runner._project_root / _PROMPT_PATHS["red_team_agent"]
        assert path.exists(), f"Red team prompt not found: {path}"

    def test_red_team_max_turns(self):
        from orchestrator.agent_runner import AgentRunner
        runner = AgentRunner()
        assert runner._get_max_turns("red_team_agent") == 3

    def test_red_team_tool_list_only_return_result(self):
        from orchestrator.agent_runner import AgentRunner
        runner = AgentRunner()
        tools = runner._build_tool_list("red_team_agent")
        names = {t["name"] for t in tools}
        assert "return_result" in names
        # Pure-reasoning: should NOT have execution tools
        assert "run_python" not in names

    def test_red_team_in_agent_name_literal(self):
        from orchestrator.message_contracts import AgentName
        assert "red_team_agent" in AgentName.__args__

    def test_runner_dispatches_local(self):
        from orchestrator.agent_runner import AgentRunner
        from orchestrator.message_contracts import AgentInput
        runner = AgentRunner()
        inp = AgentInput(
            agent_name="red_team_agent",
            disease_query={"disease_name": "IBD"},
            upstream_results={
                "prioritization_result": {
                    "targets": [_make_target("NOD2", rank=1)]
                },
                "literature_result": {},
            },
            run_id="test",
        )
        out = runner.dispatch("red_team_agent", inp)
        assert out.results is not None
        assert "red_team_assessments" in out.results


# ---------------------------------------------------------------------------
# Integration: red_team_result in pipeline output
# ---------------------------------------------------------------------------

class TestRedTeamIntegration:

    def test_red_team_result_in_pipeline(self):
        from orchestrator.pi_orchestrator_v2 import analyze_disease_v2
        from orchestrator.agent_runner import AgentRunner
        from orchestrator.message_contracts import wrap_output

        def _mock_dispatch(agent_name, agent_input):
            stubs = {
                "phenotype_architect":         {"disease_name": "test", "efo_id": "EFO_0000001"},
                "statistical_geneticist":      {"top_genes": [{"gene_symbol": "NOD2"}], "n_instruments": 1},
                "somatic_exposure_agent":      {"chip_edges": [], "drug_edges": []},
                "perturbation_genomics_agent": {
                    "genes": ["NOD2"], "beta_matrix": {},
                    "evidence_tier_per_gene": {"NOD2": "Tier1_Interventional"},
                },
                "regulatory_genomics_agent":   {"tier2_upgrades": []},
                "causal_discovery_agent":      {
                    "edges_written": [], "n_edges_written": 2,
                    "anchor_recovery": {"recovery_rate": 1.0, "missing": []},
                    "top_genes": [{"gene_symbol": "NOD2"}],
                },
                "kg_completion_agent":         {"pathways": [], "drugs": []},
                "target_prioritization_agent": {
                    "targets": [_make_target("NOD2", rank=1)],
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
                "red_team_agent": {
                    "red_team_assessments": [{
                        "target_gene": "NOD2", "rank": 1,
                        "scone_confidence": 0.85, "scone_flags": [],
                        "ota_gamma_ci_lower": 0.10, "ota_gamma_ci_upper": 0.40,
                        "ci_width": 0.30,
                        "confidence_level": "HIGH",
                        "evidence_vulnerability": "No major vulnerability.",
                        "counterargument": "OTA γ predicts population-level causality.",
                        "rank_stability": "STABLE",
                        "rank_stability_rationale": "Tier1 evidence.",
                        "literature_flag": None,
                        "red_team_verdict": "PROCEED",
                    }],
                    "n_targets_assessed": 1, "n_flagged_caution": 0,
                    "n_flagged_deprioritize": 0, "overall_confidence": "HIGH",
                    "red_team_summary": "NOD2 passed red team review.",
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
        # Pipeline should complete and red_team_result surfaced
        assert result.get("pipeline_status") == "SUCCESS" or "red_team" in str(result)
