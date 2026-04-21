"""
tests/test_phase_o_interagent.py — Phase O: Inter-agent communication + reviewer re-delegation.

Tests:
  - ReDelegationInstruction / AgentFeedback dataclasses
  - build_feedback_from_reviewer: groups issues by agent, correct priority
  - Re-delegation loop: correct agents re-run, max_rounds cap, non-redelegatable logged
  - scientific_reviewer_agent: re_delegation_instructions in output
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrator.agent_messages import (
    AgentFeedback,
    ReDelegationInstruction,
    build_feedback_from_reviewer,
)


# ---------------------------------------------------------------------------
# ReDelegationInstruction + AgentFeedback
# ---------------------------------------------------------------------------

class TestAgentMessageTypes:

    def test_redelegation_instruction_to_dict(self):
        instr = ReDelegationInstruction(
            agent_name="chemistry_agent",
            priority="MAJOR",
            issues=["GPS putative target mapping incomplete for PCSK9"],
            instruction="Re-run GPS screen and normalize putative targets to HGNC.",
            context={"genes": ["PCSK9"]},
        )
        d = instr.to_dict()
        assert d["agent_name"] == "chemistry_agent"
        assert d["priority"] == "MAJOR"
        assert "PCSK9" in d["issues"][0]

    def test_agent_feedback_for_agent(self):
        feedback = AgentFeedback(
            run_id="test-run",
            reviewer_verdict="REVISE",
            instructions=[
                ReDelegationInstruction("chemistry_agent", "MAJOR", [], "Fix chem.", {}),
                ReDelegationInstruction("causal_discovery_agent", "CRITICAL", [], "Fix causal.", {}),
            ],
        )
        assert feedback.for_agent("chemistry_agent") is not None
        assert feedback.for_agent("chemistry_agent").priority == "MAJOR"
        assert feedback.for_agent("scientific_writer_agent") is None

    def test_agent_feedback_has_critical(self):
        feedback = AgentFeedback(
            run_id="test",
            reviewer_verdict="REVISE",
            instructions=[
                ReDelegationInstruction("chemistry_agent", "MAJOR", [], "", {}),
            ],
        )
        assert feedback.has_critical() is False

        feedback.instructions.append(
            ReDelegationInstruction("causal_discovery_agent", "CRITICAL", [], "", {})
        )
        assert feedback.has_critical() is True

    def test_agents_to_revisit_critical_first(self):
        feedback = AgentFeedback(
            run_id="test",
            reviewer_verdict="REVISE",
            instructions=[
                ReDelegationInstruction("chemistry_agent", "MAJOR", [], "", {}),
                ReDelegationInstruction("causal_discovery_agent", "CRITICAL", [], "", {}),
            ],
        )
        order = feedback.agents_to_revisit()
        assert order[0] == "causal_discovery_agent"
        assert order[1] == "chemistry_agent"

    def test_agents_to_revisit_deduplicates(self):
        feedback = AgentFeedback(
            run_id="test",
            reviewer_verdict="REVISE",
            instructions=[
                ReDelegationInstruction("chemistry_agent", "CRITICAL", [], "", {}),
                ReDelegationInstruction("chemistry_agent", "MAJOR", [], "", {}),
            ],
        )
        assert feedback.agents_to_revisit().count("chemistry_agent") == 1


# ---------------------------------------------------------------------------
# build_feedback_from_reviewer
# ---------------------------------------------------------------------------

class TestBuildFeedbackFromReviewer:

    def _make_reviewer_result(self, issues):
        return {"issues": issues, "verdict": "REVISE"}

    def test_groups_by_agent(self):
        reviewer_result = self._make_reviewer_result([
            {"severity": "CRITICAL", "check": "B_anchor_recovery",
             "gene": None, "agent_to_revisit": "causal_discovery_agent",
             "description": "Anchor recovery 60%"},
            {"severity": "MAJOR", "check": "C_tractability_unsupported",
             "gene": "PCSK9", "agent_to_revisit": "chemistry_agent",
             "description": "No tractability for PCSK9"},
            {"severity": "MAJOR", "check": "C_tractability_unsupported",
             "gene": "IL6R", "agent_to_revisit": "chemistry_agent",
             "description": "No tractability for IL6R"},
        ])
        feedback = build_feedback_from_reviewer(reviewer_result, "test-run")
        assert len(feedback.instructions) == 2
        chem_instr = feedback.for_agent("chemistry_agent")
        assert chem_instr is not None
        assert len(chem_instr.context["genes"]) == 2  # PCSK9 + IL6R grouped

    def test_critical_priority_from_critical_issue(self):
        reviewer_result = self._make_reviewer_result([
            {"severity": "CRITICAL", "check": "B_anchor_recovery",
             "gene": None, "agent_to_revisit": "causal_discovery_agent",
             "description": "Critical anchor failure"},
        ])
        feedback = build_feedback_from_reviewer(reviewer_result, "test-run")
        instr = feedback.for_agent("causal_discovery_agent")
        assert instr.priority == "CRITICAL"

    def test_minor_issues_excluded(self):
        reviewer_result = self._make_reviewer_result([
            {"severity": "MINOR", "check": "F_low_specificity",
             "gene": "LYZ", "agent_to_revisit": "scientific_writer_agent",
             "description": "Low tau"},
        ])
        feedback = build_feedback_from_reviewer(reviewer_result, "test-run")
        # Minor issues should not generate re-delegation instructions
        assert len(feedback.instructions) == 0

    def test_approve_verdict_no_redelegation(self):
        reviewer_result = {"issues": [], "verdict": "APPROVE"}
        feedback = build_feedback_from_reviewer(reviewer_result, "test-run")
        assert len(feedback.instructions) == 0
        assert feedback.reviewer_verdict == "APPROVE"

    def test_instruction_text_is_actionable(self):
        reviewer_result = self._make_reviewer_result([
            {"severity": "CRITICAL", "check": "B_anchor_recovery",
             "gene": None, "agent_to_revisit": "causal_discovery_agent",
             "description": "Anchor recovery 62% < 80%"},
        ])
        feedback = build_feedback_from_reviewer(reviewer_result, "test-run")
        instr = feedback.for_agent("causal_discovery_agent")
        assert len(instr.instruction) > 20   # non-trivial instruction


# ---------------------------------------------------------------------------
# scientific_reviewer_agent: re_delegation_instructions in output
# ---------------------------------------------------------------------------

class TestReviewerReDelegationOutput:

    def _make_pipeline_outputs(self, anchor_recovery=1.0, top_tier="Tier1_Interventional"):
        return {
            "beta_matrix_result": {
                "evidence_tier_per_gene": {"NOD2": top_tier}
            },
            "causal_result": {
                "anchor_recovery": {"recovery_rate": anchor_recovery, "missing": []}
            },
            "prioritization_result": {
                "targets": [
                    {"target_gene": "NOD2", "rank": 1, "target_score": 0.5,
                     "ota_gamma": 0.3, "evidence_tier": top_tier, "max_phase": 2},
                ]
            },
            "chemistry_result": {"gps_disease_reversers": [], "gps_putative_hgnc": {"genes": []}},
            "graph_output": {"target_list": ["NOD2"], "anchor_edge_recovery": anchor_recovery},
        }

    def test_reviewer_emits_re_delegation_instructions(self):
        from agents.tier5_writer.scientific_reviewer_agent import run as reviewer_run
        pipeline_outputs = self._make_pipeline_outputs(
            anchor_recovery=0.5,   # triggers CRITICAL B_anchor_recovery
        )
        result = reviewer_run(pipeline_outputs, {"disease_name": "IBD"})
        assert "re_delegation_instructions" in result
        assert isinstance(result["re_delegation_instructions"], list)

    def test_critical_anchor_failure_has_causal_instruction(self):
        from agents.tier5_writer.scientific_reviewer_agent import run as reviewer_run
        pipeline_outputs = self._make_pipeline_outputs(anchor_recovery=0.5)
        result = reviewer_run(pipeline_outputs, {"disease_name": "IBD"})
        agents = [i["agent_name"] for i in result["re_delegation_instructions"]]
        assert "causal_discovery_agent" in agents

    def test_clean_pipeline_no_critical_instructions(self):
        from agents.tier5_writer.scientific_reviewer_agent import run as reviewer_run
        pipeline_outputs = self._make_pipeline_outputs(anchor_recovery=1.0)
        result = reviewer_run(pipeline_outputs, {"disease_name": "IBD"})
        # Good pipeline: no CRITICAL re-delegation instructions
        critical = [i for i in result["re_delegation_instructions"] if i["priority"] == "CRITICAL"]
        assert len(critical) == 0


    # Downstream score-adjustment plumbing was removed; OTA ranking is owned by
    # target_prioritization_agent and Tier 4 agents provide evidence-only outputs.
