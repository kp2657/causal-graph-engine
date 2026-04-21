"""
tests/test_phase_n_autonomous.py — Phase N: Autonomous agent execution tools.

Tests:
  - run_python: stdout capture, stderr capture, returncode, timeout enforcement
  - read_project_file: content read, path sandboxing (rejects ../../../etc/passwd)
  - list_project_files: glob results, count, sandbox
  - AgentRunner: autonomous agents get execution tools, non-autonomous don't
  - AgentRunner: autonomous agents get max_turns=40, others get 20
  - Mock SDK dispatch: Claude calls run_python, gets result, then return_result
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrator.execution_tools import (
    PROJECT_ROOT,
    list_project_files,
    read_project_file,
    run_python,
)
from orchestrator.agent_runner import AgentRunner, _AUTONOMOUS_AGENTS
from orchestrator.message_contracts import AgentInput, AgentOutput


# ---------------------------------------------------------------------------
# run_python tests
# ---------------------------------------------------------------------------

class TestRunPython:

    def test_captures_stdout(self):
        result = run_python("print('hello world')")
        assert result["success"] is True
        assert result["returncode"] == 0
        assert "hello world" in result["stdout"]

    def test_captures_stderr(self):
        result = run_python("import sys; sys.stderr.write('error msg')")
        assert "error msg" in result["stderr"]

    def test_nonzero_returncode_on_exception(self):
        result = run_python("raise ValueError('boom')")
        assert result["success"] is False
        assert result["returncode"] != 0
        assert "ValueError" in result["stderr"] or "boom" in result["stderr"]

    def test_json_output_pattern(self):
        """Agents typically print JSON to stdout — verify this works."""
        code = "import json; print(json.dumps({'gene': 'NOD2', 'gamma': 0.31}))"
        result = run_python(code)
        assert result["success"] is True
        import json
        parsed = json.loads(result["stdout"].strip())
        assert parsed["gene"] == "NOD2"
        assert parsed["gamma"] == pytest.approx(0.31)

    def test_project_imports_work(self):
        """Code in run_python can import from the project."""
        result = run_python(
            "from orchestrator.execution_tools import PROJECT_ROOT; "
            "print(str(PROJECT_ROOT))"
        )
        assert result["success"] is True
        assert "causal-graph-engine" in result["stdout"]

    def test_timeout_enforced(self):
        result = run_python("import time; time.sleep(999)", timeout=2)
        assert result["success"] is False
        assert result["returncode"] == -1
        assert "Timeout" in result["stderr"] or "timeout" in result["stderr"].lower()

    def test_stdout_truncated_when_large(self):
        """Output larger than _STDOUT_LIMIT is truncated, not dropped."""
        code = "print('x' * 20000)"
        result = run_python(code)
        assert result["success"] is True
        assert len(result["stdout"]) <= 8500   # limit + small truncation header
        assert "truncated" in result["stdout"] or "x" * 100 in result["stdout"]

    def test_syntax_error_captured(self):
        result = run_python("def broken(:\n    pass")
        assert result["success"] is False
        assert "SyntaxError" in result["stderr"]

    def test_empty_code_succeeds(self):
        result = run_python("")
        assert result["success"] is True
        assert result["returncode"] == 0


# ---------------------------------------------------------------------------
# read_project_file tests
# ---------------------------------------------------------------------------

class TestReadProjectFile:

    def test_reads_existing_file(self):
        result = read_project_file("orchestrator/execution_tools.py")
        assert result["error"] is None
        assert "run_python" in result["content"]

    def test_path_relative_to_project_root(self):
        result = read_project_file("orchestrator/execution_tools.py")
        assert str(PROJECT_ROOT) in result["path"]

    def test_missing_file_returns_error(self):
        result = read_project_file("nonexistent/path/file.txt")
        assert result["error"] is not None
        assert "not found" in result["error"].lower() or "No such" in result["error"]
        assert result["content"] == ""

    def test_rejects_path_outside_project(self):
        result = read_project_file("../../etc/passwd")
        assert result["error"] is not None
        assert "denied" in result["error"].lower() or "outside" in result["error"].lower()
        assert result["content"] == ""

    def test_rejects_absolute_path_outside_project(self):
        result = read_project_file("/etc/passwd")
        # /etc/passwd resolved will be outside PROJECT_ROOT
        assert result["error"] is not None
        assert result["content"] == ""

    def test_large_file_truncated(self):
        # STATE.md is large enough to test truncation behaviour at 20k chars
        result = read_project_file("STATE.md")
        assert result["error"] is None
        assert len(result["content"]) <= 20_100  # limit + small header


# ---------------------------------------------------------------------------
# list_project_files tests
# ---------------------------------------------------------------------------

class TestListProjectFiles:

    def test_returns_file_list(self):
        result = list_project_files("orchestrator/*.py")
        assert result["error"] is None
        assert result["count"] > 0
        assert any("agent_runner" in f for f in result["files"])

    def test_recursive_glob(self):
        result = list_project_files("tests/**/*.py")
        assert result["count"] > 5
        assert all(f.startswith("tests/") for f in result["files"])

    def test_nonexistent_pattern_returns_empty(self):
        result = list_project_files("nonexistent_dir_xyz/*.json")
        assert result["error"] is None
        assert result["count"] == 0
        assert result["files"] == []

    def test_relative_paths_returned(self):
        result = list_project_files("orchestrator/*.py")
        for f in result["files"]:
            assert not f.startswith("/")   # relative, not absolute
            assert str(PROJECT_ROOT) not in f


# ---------------------------------------------------------------------------
# AgentRunner: execution tool dispatch
# ---------------------------------------------------------------------------

class TestAgentRunnerExecutionTools:

    def test_autonomous_agents_get_execution_tools(self):
        runner = AgentRunner()
        for agent in _AUTONOMOUS_AGENTS:
            tools = runner._build_tool_list(agent)
            tool_names = {t["name"] for t in tools}
            assert "run_python" in tool_names, f"{agent} missing run_python"
            assert "read_project_file" in tool_names, f"{agent} missing read_project_file"
            assert "list_project_files" in tool_names, f"{agent} missing list_project_files"

    def test_non_autonomous_agents_no_execution_tools(self):
        runner = AgentRunner()
        non_autonomous = ["phenotype_architect", "target_prioritization_agent",
                          "scientific_writer_agent", "scientific_reviewer_agent",
                          "kg_completion_agent"]
        for agent in non_autonomous:
            tools = runner._build_tool_list(agent)
            tool_names = {t["name"] for t in tools}
            assert "run_python" not in tool_names, f"{agent} should NOT have run_python"

    def test_autonomous_agents_get_max_turns_40(self):
        runner = AgentRunner()
        for agent in _AUTONOMOUS_AGENTS:
            assert runner._get_max_turns(agent) == 40, f"{agent} should have max_turns=40"

    def test_non_autonomous_agents_get_max_turns_20(self):
        runner = AgentRunner()
        assert runner._get_max_turns("phenotype_architect") == 20
        assert runner._get_max_turns("scientific_writer_agent") == 20

    def test_execution_tools_in_local_routes(self):
        runner = AgentRunner()
        routes = runner._get_local_tool_routes()
        assert "run_python" in routes
        assert "read_project_file" in routes
        assert "list_project_files" in routes
        assert callable(routes["run_python"])
        assert callable(routes["read_project_file"])
        assert callable(routes["list_project_files"])

    def test_autonomous_agents_retain_domain_tools(self):
        """Execution tools are additive — domain tools still present."""
        runner = AgentRunner()
        cda_tools = {t["name"] for t in runner._build_tool_list("causal_discovery_agent")}
        assert "compute_ota_gammas" in cda_tools
        assert "run_python" in cda_tools

        chem_tools = {t["name"] for t in runner._build_tool_list("chemistry_agent")}
        assert "get_chembl_target_activities" in chem_tools
        assert "run_python" in chem_tools

    def test_execution_tool_schemas_valid(self):
        runner = AgentRunner()
        tools = runner._build_tool_list("chemistry_agent")
        exec_tools = [t for t in tools if t["name"] in
                      ("run_python", "read_project_file", "list_project_files")]
        assert len(exec_tools) == 3
        for t in exec_tools:
            assert "description" in t
            assert "input_schema" in t
            assert t["input_schema"]["type"] == "object"


# ---------------------------------------------------------------------------
# Mock SDK dispatch: agent calls run_python then return_result
# ---------------------------------------------------------------------------

class TestAutonomousDispatchWithRunPython:
    """Verify the agentic loop correctly handles a run_python tool call mid-loop."""

    def _make_two_turn_runner(self, agent_name: str) -> tuple[AgentRunner, MagicMock]:
        """
        Set up a runner whose mock client does two turns:
          Turn 1: call run_python (intermediate tool)
          Turn 2: call return_result (final output)
        """
        runner = AgentRunner()
        runner.set_mode(agent_name, "sdk")

        # Turn 1: call run_python
        run_python_block = MagicMock()
        run_python_block.type = "tool_use"
        run_python_block.name = "run_python"
        run_python_block.id = "tool_call_001"
        run_python_block.input = {"code": "print('NOD2 gamma=0.31')"}

        turn1_response = MagicMock()
        turn1_response.stop_reason = "tool_use"
        turn1_response.content = [run_python_block]

        # Turn 2: call return_result
        return_result_block = MagicMock()
        return_result_block.type = "tool_use"
        return_result_block.name = "return_result"
        return_result_block.id = "tool_call_002"
        return_result_block.input = {
            "result": {"genes_scored": ["NOD2"], "investigation_notes": ["used run_python"]},
            "warnings": [],
            "edges_written": 0,
        }

        turn2_response = MagicMock()
        turn2_response.stop_reason = "tool_use"
        turn2_response.content = [return_result_block]

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = [turn1_response, turn2_response]
        runner._client = mock_client

        return runner, mock_client

    def test_run_python_call_executed_and_result_fed_back(self):
        runner, mock_client = self._make_two_turn_runner("chemistry_agent")

        agent_input = AgentInput(
            disease_query={"disease_name": "inflammatory bowel disease"},
            run_id="test-autonomous-runpy",
        )
        output = runner.dispatch("chemistry_agent", agent_input)

        # Two turns: one for run_python, one for return_result
        assert mock_client.messages.create.call_count == 2
        assert isinstance(output, AgentOutput)
        assert output.agent_name == "chemistry_agent"

    def test_run_python_result_included_in_second_turn(self):
        """The tool result from run_python must appear in the second API call's messages."""
        runner, mock_client = self._make_two_turn_runner("causal_discovery_agent")

        agent_input = AgentInput(
            disease_query={"disease_name": "inflammatory bowel disease"},
            upstream_results={
                "perturbation_genomics_agent": {"genes": ["NOD2"], "beta_matrix": {}, "programs": []},
                "_gamma_estimates": {},
            },
            run_id="test-autonomous-feedback",
        )
        runner.dispatch("causal_discovery_agent", agent_input)

        second_call_args = mock_client.messages.create.call_args_list[1]
        messages = second_call_args.kwargs.get("messages", [])
        # Last message should be the tool result from run_python
        tool_result_msgs = [
            m for m in messages
            if isinstance(m.get("content"), list)
            and any(c.get("type") == "tool_result" for c in m["content"])
        ]
        assert len(tool_result_msgs) >= 1

    def test_return_result_terminates_loop(self):
        runner, mock_client = self._make_two_turn_runner("statistical_geneticist")

        agent_input = AgentInput(
            disease_query={"disease_name": "coronary artery disease"},
            run_id="test-autonomous-terminate",
        )
        output = runner.dispatch("statistical_geneticist", agent_input)

        assert isinstance(output, AgentOutput)
        # Loop should have stopped at return_result, not consumed all 40 turns
        assert mock_client.messages.create.call_count == 2
