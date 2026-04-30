"""
Tests for Phase 2 agentic infrastructure:
    agentic_orchestrator (lifecycle, stubs, pause routing)
    chief_of_staff_agent (formatters, decision parsing, prompt assembly)
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from orchestrator.agentic.agent_contracts import (
    CSOOutput,
    DiscoveryRefinementOutput,
    RedelegationRecord,
    ReviewerOutput,
    TokenUsage,
    VirginTarget,
)
from orchestrator.agentic.agentic_orchestrator import (
    _run_chemistry_agent,
    _run_clinical_trialist,
    _run_convergent_evidence_agent,
    _run_discovery_refinement_agent,
    _run_red_team,
    _run_reviewer,
    _run_statistical_geneticist,
    load_checkpoint,
    run_agentic,
)
from orchestrator.agentic.chief_of_staff_agent import (
    _build_system_prompt,
    _fmt_briefing_context,
    _fmt_conflict_context,
    _fmt_exec_context,
    _parse_decision,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_checkpoint():
    return {
        "disease_name": "coronary artery disease",
        "efo_id": "EFO_0001645",
        "target_list": [
            {"target_gene": "ATP2B1", "ota_gamma": -3.842, "dominant_tier": "tier2", "max_phase": 0, "l2g_score": 0.8},
            {"target_gene": "PCSK9", "ota_gamma": -2.1, "dominant_tier": "tier1", "max_phase": 4, "l2g_score": 0.95},
            {"target_gene": "IL6R", "ota_gamma": 1.5, "dominant_tier": "tier1", "max_phase": 4, "l2g_score": 0.7},
        ],
        "gps_disease_state_reversers": [{"compound": "atorvastatin"}],
        "data_completeness": {
            "perturb_seq_dataset": "GSE210681",
            "h5ad_disease_sig_loaded": True,
            "gps_screen_run": True,
            "n_gps_reversers": 1,
        },
        "pipeline_warnings": [],
    }


# ---------------------------------------------------------------------------
# Decision parsing
# ---------------------------------------------------------------------------

def test_parse_decision_valid():
    text = 'Analysis...\n<decision>\n{"pause_run": true, "pause_reason": "low coverage", "downgraded_candidates": ["GENE1"]}\n</decision>'
    d = _parse_decision(text)
    assert d["pause_run"] is True
    assert d["pause_reason"] == "low coverage"
    assert d["downgraded_candidates"] == ["GENE1"]


def test_parse_decision_no_block():
    d = _parse_decision("no decision block here")
    assert d["pause_run"] is False
    assert d["downgraded_candidates"] == []


def test_parse_decision_malformed_json():
    d = _parse_decision("<decision>not json</decision>")
    assert d["pause_run"] is False


def test_parse_decision_continue():
    text = '<decision>\n{"pause_run": false, "pause_reason": "", "downgraded_candidates": []}\n</decision>'
    d = _parse_decision(text)
    assert d["pause_run"] is False


# ---------------------------------------------------------------------------
# Briefing context formatter
# ---------------------------------------------------------------------------

def test_fmt_briefing_includes_top_targets(minimal_checkpoint):
    ctx = _fmt_briefing_context(minimal_checkpoint)
    assert "ATP2B1" in ctx
    assert "PCSK9" in ctx


def test_fmt_briefing_includes_data_completeness(minimal_checkpoint):
    ctx = _fmt_briefing_context(minimal_checkpoint)
    assert "GSE210681" in ctx
    assert "GPS" in ctx or "gps" in ctx.lower()


def test_fmt_briefing_empty_checkpoint():
    ctx = _fmt_briefing_context({})
    assert "unknown" in ctx or "none" in ctx.lower()


# ---------------------------------------------------------------------------
# System prompt assembly
# ---------------------------------------------------------------------------

def test_build_system_prompt_contains_preamble():
    prompt = _build_system_prompt()
    assert "How you work" in prompt
    assert "Decision authority" in prompt
    assert "<decision>" in prompt


def test_build_system_prompt_with_lens():
    prompt = _build_system_prompt(extra_lens="## Extra\nSome lens")
    assert "Extra" in prompt


# ---------------------------------------------------------------------------
# Agent stubs return correct types
# ---------------------------------------------------------------------------

def test_statistical_geneticist_returns_correct_type(minimal_checkpoint):
    with patch("orchestrator.agentic.agents.statistical_geneticist.run_agent_with_tools",
               return_value=_mock_agent_response({"validated_anchors": ["PCSK9"], "library_gaps": []})):
        out = _run_statistical_geneticist(minimal_checkpoint, "r1")
    assert out.agent_name == "statistical_geneticist"
    assert "PCSK9" in out.validated_anchors


def test_convergent_evidence_agent_returns_correct_type(minimal_checkpoint):
    from orchestrator.agentic.agent_contracts import StatisticalGeneticistOutput
    sg = StatisticalGeneticistOutput(agent_name="statistical_geneticist", run_id="r1")
    with patch("orchestrator.agentic.agents.convergent_evidence_agent.run_agent_with_tools",
               return_value=_mock_agent_response({"confirmed_genes": ["GENE1"], "no_evidence_genes": []})):
        out = _run_convergent_evidence_agent(minimal_checkpoint, sg, "r1")
    assert out.agent_name == "convergent_evidence_agent"
    assert out.confirmed_genes == ["GENE1"]


def test_discovery_refinement_agent_returns_correct_type(minimal_checkpoint):
    from orchestrator.agentic.agent_contracts import ConvergentEvidenceOutput, ClinicalTrialistOutput
    ce = ConvergentEvidenceOutput(agent_name="convergent_evidence_agent", run_id="r1")
    ct = ClinicalTrialistOutput(agent_name="clinical_trialist", run_id="r1", no_active_development=[])
    with patch("orchestrator.agentic.agents.discovery_refinement_agent.run_agent_with_tools",
               return_value=_mock_agent_response({"virgin_targets": [], "downgraded_to_known": []})):
        out = _run_discovery_refinement_agent(minimal_checkpoint, ce, ct, "r1")
    assert out.agent_name == "discovery_refinement_agent"
    assert out.virgin_targets == []


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def test_load_checkpoint_missing_raises():
    with pytest.raises(FileNotFoundError):
        load_checkpoint("nonexistent disease", data_dir="/tmp/no_such_dir")


def test_load_checkpoint_reads_json(tmp_path):
    slug = "test_disease"
    data = {"disease_name": "test disease", "target_list": []}
    (tmp_path / f"analyze_{slug}.json").write_text(json.dumps(data))
    result = load_checkpoint("test disease", data_dir=str(tmp_path))
    assert result["disease_name"] == "test disease"


# ---------------------------------------------------------------------------
# Full orchestrator run (mocked LLM calls)
# ---------------------------------------------------------------------------

def _mock_cso_response(pause: bool = False) -> tuple[str, TokenUsage]:
    decision = json.dumps({
        "pause_run": pause,
        "pause_reason": "test" if pause else "",
        "downgraded_candidates": [],
    })
    text = f"CSO analysis content.\n<decision>\n{decision}\n</decision>"
    return text, TokenUsage(input_tokens=100, output_tokens=200)


def _mock_agent_response(output_dict: dict | None = None) -> tuple[str, TokenUsage]:
    """Mock for run_agent / run_agent_with_tools — returns empty structured output."""
    payload = output_dict or {}
    text = f"Agent analysis.\n<output>\n{json.dumps(payload)}\n</output>"
    return text, TokenUsage(input_tokens=50, output_tokens=100)


# Modules that use run_agent_with_tools (bound locally via `from ... import`)
_TOOL_AGENT_MODULES = [
    "orchestrator.agentic.agents.statistical_geneticist",
    "orchestrator.agentic.agents.convergent_evidence_agent",
    "orchestrator.agentic.agents.chemistry_agent",
    "orchestrator.agentic.agents.clinical_trialist",
    "orchestrator.agentic.agents.discovery_refinement_agent",
]
# Modules that use plain run_agent
_PLAIN_AGENT_MODULES = [
    "orchestrator.agentic.agents.red_team",
    "orchestrator.agentic.agents.scientific_reviewer",
]


def _patch_all_agents():
    """Context manager that mocks all LLM calls in the full agentic run."""
    from contextlib import ExitStack
    stack = ExitStack()
    stack.enter_context(patch(
        "orchestrator.agentic.chief_of_staff_agent.run_agent",
        side_effect=[_mock_cso_response(), _mock_cso_response(), _mock_cso_response()],
    ))
    for mod in _TOOL_AGENT_MODULES:
        stack.enter_context(patch(f"{mod}.run_agent_with_tools", return_value=_mock_agent_response()))
    for mod in _PLAIN_AGENT_MODULES:
        stack.enter_context(patch(f"{mod}.run_agent", return_value=_mock_agent_response()))
    return stack


def test_run_agentic_auto_continue(minimal_checkpoint, tmp_path):
    with _patch_all_agents():
        result = run_agentic(
            "coronary artery disease",
            run_id="test001",
            checkpoint=minimal_checkpoint,
            auto_continue=True,
            data_dir=str(tmp_path),
        )

    assert result["status"] == "complete"
    assert "journal_dir" in result
    assert Path(result["journal_dir"]).exists()
    assert (Path(result["journal_dir"]) / "journal.json").exists()
    assert (Path(result["journal_dir"]) / "token_usage.json").exists()


def test_run_agentic_abort_at_pause1(minimal_checkpoint, tmp_path):
    with patch("orchestrator.agentic.chief_of_staff_agent.run_agent", return_value=_mock_cso_response()):
        with patch("builtins.input", return_value="ABORT"):
            result = run_agentic(
                "coronary artery disease",
                run_id="abort_test",
                checkpoint=minimal_checkpoint,
                auto_continue=False,
                data_dir=str(tmp_path),
            )
    assert result["status"] == "aborted"
    assert result["pause"] == 1


def test_run_agentic_token_usage_recorded(minimal_checkpoint, tmp_path):
    with _patch_all_agents():
        result = run_agentic(
            "coronary artery disease",
            run_id="token_test",
            checkpoint=minimal_checkpoint,
            auto_continue=True,
            data_dir=str(tmp_path),
        )

    token_data = json.loads(
        (Path(result["journal_dir"]) / "token_usage.json").read_text()
    )
    # Should have non-zero token usage from agents
    assert token_data["total"]["input_tokens"] > 0
    assert token_data["total"]["output_tokens"] > 0


def test_run_agentic_journal_has_pauses(minimal_checkpoint, tmp_path):
    with _patch_all_agents():
        result = run_agentic(
            "coronary artery disease",
            run_id="pause_test",
            checkpoint=minimal_checkpoint,
            auto_continue=True,
            data_dir=str(tmp_path),
        )

    journal = json.loads(
        (Path(result["journal_dir"]) / "journal.json").read_text()
    )
    assert len(journal["human_in_loop_pauses"]) == 3
    pause_ids = [p["pause_id"] for p in journal["human_in_loop_pauses"]]
    assert 1 in pause_ids and 2 in pause_ids and 3 in pause_ids
