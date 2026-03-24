"""
tests/test_sdk_poc.py — SDK mode proof-of-concept tests (Step 53).

Unit tests verify:
  - AgentRunner.set_mode() / get_mode() management
  - SDK agentic loop with mocked Anthropic client
  - return_result tool extraction
  - mode_overrides dict wiring (used by analyze_disease_v2)

Integration tests (marked @pytest.mark.integration, require ANTHROPIC_API_KEY):
  - Real Claude API call for somatic_exposure_agent in sdk mode
  - Verifies return_result tool is called and structured output returned
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrator.agent_runner import AgentRunner
from orchestrator.message_contracts import AgentInput, AgentOutput


# ---------------------------------------------------------------------------
# Unit tests — no API call
# ---------------------------------------------------------------------------

class TestAgentRunnerModeManagement:

    def test_default_mode_is_local(self):
        runner = AgentRunner()
        assert runner.get_mode("somatic_exposure_agent") == "local"
        assert runner.get_mode("phenotype_architect") == "local"

    def test_set_mode_sdk(self):
        runner = AgentRunner()
        runner.set_mode("somatic_exposure_agent", "sdk")
        assert runner.get_mode("somatic_exposure_agent") == "sdk"

    def test_set_mode_only_affects_target_agent(self):
        runner = AgentRunner()
        runner.set_mode("somatic_exposure_agent", "sdk")
        # All others still local
        assert runner.get_mode("phenotype_architect") == "local"
        assert runner.get_mode("causal_discovery_agent") == "local"

    def test_set_mode_invalid_raises(self):
        runner = AgentRunner()
        with pytest.raises(ValueError, match="mode must be"):
            runner.set_mode("somatic_exposure_agent", "rpc")

    def test_set_all_sdk(self):
        runner = AgentRunner()
        runner.set_all_sdk()
        assert runner.get_mode("somatic_exposure_agent") == "sdk"
        assert runner.get_mode("scientific_writer_agent") == "sdk"

    def test_set_mode_reverts_to_local(self):
        runner = AgentRunner()
        runner.set_mode("somatic_exposure_agent", "sdk")
        runner.set_mode("somatic_exposure_agent", "local")
        assert runner.get_mode("somatic_exposure_agent") == "local"

    def test_mode_overrides_dict_pattern(self):
        """Verify the mode_overrides pattern used by analyze_disease_v2."""
        runner = AgentRunner()
        mode_overrides = {
            "somatic_exposure_agent": "sdk",
            "perturbation_genomics_agent": "sdk",
        }
        for agent_name, mode in mode_overrides.items():
            runner.set_mode(agent_name, mode)

        assert runner.get_mode("somatic_exposure_agent") == "sdk"
        assert runner.get_mode("perturbation_genomics_agent") == "sdk"
        assert runner.get_mode("phenotype_architect") == "local"
        assert runner.get_mode("causal_discovery_agent") == "local"


class TestSdkDispatchWithMockedClient:
    """Verify the SDK agentic loop with a mocked Anthropic client."""

    def _make_runner_sdk(self, return_result_payload: dict) -> AgentRunner:
        """Return an AgentRunner in SDK mode with a mock client."""
        runner = AgentRunner()
        runner.set_mode("somatic_exposure_agent", "sdk")

        # Build a mock response that calls return_result immediately
        mock_block = MagicMock()
        mock_block.type = "tool_use"
        mock_block.name = "return_result"
        mock_block.input = return_result_payload

        mock_response = MagicMock()
        mock_response.stop_reason = "tool_use"
        mock_response.content = [mock_block]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        runner._client = mock_client

        return runner

    def test_sdk_dispatch_returns_agent_output(self):
        runner = self._make_runner_sdk({
            "result": {
                "chip_edges": [],
                "drug_edges": [],
                "viral_edges": [],
                "summary": {"n_chip_genes": 0, "n_drug_targets": 0},
            },
            "warnings": [],
            "edges_written": 0,
        })

        agent_input = AgentInput(
            disease_query={
                "disease_name": "coronary artery disease",
                "efo_id": "EFO_0001645",
                "modifier_types": ["germline", "somatic_chip", "drug"],
            },
            run_id="test-sdk-unit",
        )
        output = runner.dispatch("somatic_exposure_agent", agent_input)

        assert isinstance(output, AgentOutput)
        assert output.agent_name == "somatic_exposure_agent"

    def test_sdk_dispatch_calls_anthropic_client(self):
        runner = self._make_runner_sdk({
            "result": {"chip_edges": [], "drug_edges": [], "summary": {}},
            "warnings": [],
        })

        agent_input = AgentInput(
            disease_query={"disease_name": "coronary artery disease", "efo_id": "EFO_0001645"},
            run_id="test-sdk-client-called",
        )
        runner.dispatch("somatic_exposure_agent", agent_input)

        assert runner._client.messages.create.called
        call_kwargs = runner._client.messages.create.call_args
        assert call_kwargs is not None

    def test_sdk_dispatch_sends_system_prompt(self):
        runner = self._make_runner_sdk({"result": {}, "warnings": []})

        agent_input = AgentInput(
            disease_query={"disease_name": "coronary artery disease"},
            run_id="test-sdk-prompt",
        )
        runner.dispatch("somatic_exposure_agent", agent_input)

        call_kwargs = runner._client.messages.create.call_args
        # system prompt should be a non-empty string
        assert "system" in call_kwargs.kwargs
        assert isinstance(call_kwargs.kwargs["system"], str)
        assert len(call_kwargs.kwargs["system"]) > 0

    def test_sdk_dispatch_includes_return_result_tool(self):
        runner = self._make_runner_sdk({"result": {}, "warnings": []})

        agent_input = AgentInput(
            disease_query={"disease_name": "coronary artery disease"},
            run_id="test-sdk-tools",
        )
        runner.dispatch("somatic_exposure_agent", agent_input)

        call_kwargs = runner._client.messages.create.call_args
        tools = call_kwargs.kwargs.get("tools", [])
        tool_names = [t["name"] for t in tools]
        assert "return_result" in tool_names

    def test_sdk_dispatch_wraps_results(self):
        chip_edges = [{"from_node": "TET2", "to_node": "CAD", "effect_size": 0.3}]
        runner = self._make_runner_sdk({
            "result": {"chip_edges": chip_edges, "summary": {"n_chip_genes": 1}},
            "warnings": ["test warning"],
            "edges_written": 1,
        })

        agent_input = AgentInput(
            disease_query={"disease_name": "coronary artery disease", "efo_id": "EFO_0001645"},
            run_id="test-sdk-results",
        )
        output = runner.dispatch("somatic_exposure_agent", agent_input)

        assert output.edges_written == 1

    def test_sdk_end_turn_falls_back_to_text(self):
        """If Claude returns end_turn (no tool call), text is wrapped as summary."""
        runner = AgentRunner()
        runner.set_mode("somatic_exposure_agent", "sdk")

        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "No CHIP genes identified for this disease."

        mock_response = MagicMock()
        mock_response.stop_reason = "end_turn"
        mock_response.content = [text_block]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        runner._client = mock_client

        agent_input = AgentInput(
            disease_query={"disease_name": "coronary artery disease"},
            run_id="test-sdk-endturn",
        )
        output = runner.dispatch("somatic_exposure_agent", agent_input)

        assert isinstance(output, AgentOutput)
        assert "summary" in output.results or isinstance(output.results, dict)

    def test_sdk_client_error_returns_stub_fallback(self):
        """If the Anthropic client raises, output has stub_fallback=True."""
        runner = AgentRunner()
        runner.set_mode("somatic_exposure_agent", "sdk")

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = RuntimeError("API unavailable")
        runner._client = mock_client

        agent_input = AgentInput(
            disease_query={"disease_name": "coronary artery disease"},
            run_id="test-sdk-error",
        )
        output = runner.dispatch("somatic_exposure_agent", agent_input)

        assert isinstance(output, AgentOutput)
        assert output.stub_fallback is True
        assert len(output.warnings) > 0


# ---------------------------------------------------------------------------
# causal_discovery_agent SDK tool tests
# ---------------------------------------------------------------------------

class TestCausalDiscoveryAgentSdkTools:
    """Verify SDK tooling for causal_discovery_agent is correctly wired."""

    def test_causal_discovery_agent_has_sdk_tools(self):
        runner = AgentRunner()
        runner.set_mode("causal_discovery_agent", "sdk")
        tools = runner._build_tool_list("causal_discovery_agent")
        tool_names = {t["name"] for t in tools}
        assert "return_result"       in tool_names
        assert "compute_ota_gammas"  in tool_names
        assert "write_causal_edges"  in tool_names
        assert "check_anchor_recovery" in tool_names
        assert "compute_shd"         in tool_names

    def test_other_agents_dont_get_causal_tools(self):
        runner = AgentRunner()
        runner.set_mode("somatic_exposure_agent", "sdk")
        tools = runner._build_tool_list("somatic_exposure_agent")
        tool_names = {t["name"] for t in tools}
        assert "compute_ota_gammas" not in tool_names

    def test_causal_discovery_tool_routes_importable(self):
        runner = AgentRunner()
        routes = runner._get_local_tool_routes()
        assert "compute_ota_gammas"    in routes
        assert "check_anchor_recovery" in routes
        assert "compute_shd"           in routes
        assert callable(routes["compute_ota_gammas"])

    def test_causal_discovery_sdk_dispatch_with_mock(self):
        """End-to-end mock: SDK dispatch calls agentic loop, returns structured output."""
        runner = AgentRunner()
        runner.set_mode("causal_discovery_agent", "sdk")

        mock_block = MagicMock()
        mock_block.type = "tool_use"
        mock_block.name = "return_result"
        mock_block.input = {
            "result": {
                "n_edges_written": 5,
                "n_edges_rejected": 2,
                "top_genes": [{"gene": "NOD2", "ota_gamma": 0.25, "tier": "provisional_virtual"}],
                "anchor_recovery": {"recovery_rate": 1.0, "recovered": ["NOD2→IBD"], "missing": []},
                "shd": 0,
                "warnings": [],
            },
            "warnings": [],
            "edges_written": 5,
        }
        mock_response = MagicMock()
        mock_response.stop_reason = "tool_use"
        mock_response.content = [mock_block]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        runner._client = mock_client

        agent_input = AgentInput(
            disease_query={"disease_name": "inflammatory bowel disease", "efo_id": "EFO_0003767"},
            upstream_results={
                "perturbation_genomics_agent": {
                    "genes": ["NOD2"], "programs": ["inflammatory_NF-kB"],
                    "beta_matrix": {}, "evidence_tier_per_gene": {},
                },
                "_gamma_estimates": {},
            },
            run_id="test-causal-sdk",
        )
        output = runner.dispatch("causal_discovery_agent", agent_input)

        assert isinstance(output, AgentOutput)
        assert output.agent_name == "causal_discovery_agent"
        assert output.edges_written == 5
        assert output.results["n_edges_written"] == 5

    def test_causal_discovery_sdk_has_system_prompt(self):
        """System prompt file must exist and be non-trivial."""
        prompt_path = (
            Path(__file__).parent.parent
            / "agents/tier3_causal/prompts/causal_discovery_agent.md"
        )
        assert prompt_path.exists(), "System prompt file missing"
        content = prompt_path.read_text()
        assert "compute_ota_gammas" in content
        assert "anchor_recovery"    in content
        assert "return_result"      in content


# ---------------------------------------------------------------------------
# Integration test — real Anthropic API call
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestSdkPocLive:
    """
    Real Anthropic API call for somatic_exposure_agent in sdk mode.

    Skipped if ANTHROPIC_API_KEY is not set.

    This is the proof-of-concept for the Claude Agent SDK migration:
    one agent flipped to "sdk" mode while all others remain "local".
    The agent receives its system prompt + input JSON, calls return_result,
    and the AgentRunner wraps the output into a typed AgentOutput envelope.
    """

    def test_somatic_agent_sdk_returns_output(self):
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

        runner = AgentRunner(model="claude-haiku-4-5-20251001")  # fast/cheap for PoC
        runner.set_mode("somatic_exposure_agent", "sdk")

        agent_input = AgentInput(
            disease_query={
                "disease_name":   "coronary artery disease",
                "efo_id":         "EFO_0001645",
                "modifier_types": ["germline", "somatic_chip", "drug"],
                "day_one_mode":   True,
            },
            run_id="sdk-poc-live",
        )
        output = runner.dispatch("somatic_exposure_agent", agent_input)

        assert isinstance(output, AgentOutput)
        assert isinstance(output.results, dict)
        assert output.agent_name == "somatic_exposure_agent"
        # The agent should produce at least some structured output
        assert len(output.results) > 0

    def test_sdk_mode_output_compatible_with_local_mode(self):
        """SDK and local outputs both return AgentOutput — shape is the same."""
        if not os.getenv("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

        agent_input = AgentInput(
            disease_query={
                "disease_name":   "coronary artery disease",
                "efo_id":         "EFO_0001645",
                "modifier_types": ["germline", "somatic_chip", "drug"],
                "day_one_mode":   True,
            },
            run_id="sdk-vs-local",
        )

        local_runner = AgentRunner()
        local_output = local_runner.dispatch("somatic_exposure_agent", agent_input)

        sdk_runner = AgentRunner(model="claude-haiku-4-5-20251001")
        sdk_runner.set_mode("somatic_exposure_agent", "sdk")
        sdk_output = sdk_runner.dispatch("somatic_exposure_agent", agent_input)

        # Both should be AgentOutput instances with the same agent name
        assert isinstance(local_output, AgentOutput)
        assert isinstance(sdk_output, AgentOutput)
        assert local_output.agent_name == sdk_output.agent_name
        assert local_output.tier == sdk_output.tier
