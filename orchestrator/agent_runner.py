"""
agent_runner.py — Claude Agent SDK runner for subagent dispatch.

Supports two modes:
  - "local":  call the existing agent run() function directly (current behavior)
  - "sdk":    call Claude via the Anthropic API with tool_use for structured output

The mode can be set per-agent, enabling gradual migration from local → sdk
without breaking the working pipeline.

Typical usage (chief_of_staff):
    runner = AgentRunner()

    # Current: local mode (no API call, just wraps existing run())
    output = runner.dispatch("somatic_exposure_agent", input_env)

    # Future: SDK mode (real Claude subagent)
    runner.set_mode("somatic_exposure_agent", "sdk")
    output = runner.dispatch("somatic_exposure_agent", input_env)
"""
from __future__ import annotations

import importlib
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrator.message_contracts import AgentInput, AgentOutput, wrap_output

# ---------------------------------------------------------------------------
# Agent module registry — maps agent_name → importable module path
# ---------------------------------------------------------------------------

_AGENT_MODULES: dict[str, str] = {
    "phenotype_architect":         "agents.tier1_phenomics.phenotype_architect",
    "statistical_geneticist":      "agents.tier1_phenomics.statistical_geneticist",
    "somatic_exposure_agent":      "agents.tier1_phenomics.somatic_exposure_agent",
    "perturbation_genomics_agent": "agents.tier2_pathway.perturbation_genomics_agent",
    "regulatory_genomics_agent":   "agents.tier2_pathway.regulatory_genomics_agent",
    "causal_discovery_agent":      "agents.tier3_causal.causal_discovery_agent",
    "kg_completion_agent":         "agents.tier3_causal.kg_completion_agent",
    "target_prioritization_agent": "agents.tier4_translation.target_prioritization_agent",
    "chemistry_agent":             "agents.tier4_translation.chemistry_agent",
    "clinical_trialist_agent":     "agents.tier4_translation.clinical_trialist_agent",
    "scientific_writer_agent":     "agents.tier5_writer.scientific_writer_agent",
    "scientific_reviewer_agent":   "agents.tier5_writer.scientific_reviewer_agent",
}

# ---------------------------------------------------------------------------
# Per-agent system prompt paths
# ---------------------------------------------------------------------------

_PROMPT_PATHS: dict[str, str] = {
    "phenotype_architect":         "agents/tier1_phenomics/prompts/phenotype_architect.md",
    "statistical_geneticist":      "agents/tier1_phenomics/prompts/statistical_geneticist.md",
    "somatic_exposure_agent":      "agents/tier1_phenomics/prompts/somatic_exposure_agent.md",
    "perturbation_genomics_agent": "agents/tier2_pathway/prompts/perturbation_genomics_agent.md",
    "regulatory_genomics_agent":   "agents/tier2_pathway/prompts/regulatory_genomics_agent.md",
    "causal_discovery_agent":      "agents/tier3_causal/prompts/causal_discovery_agent.md",
    "kg_completion_agent":         "agents/tier3_causal/prompts/kg_completion_agent.md",
    "target_prioritization_agent": "agents/tier4_translation/prompts/target_prioritization_agent.md",
    "chemistry_agent":             "agents/tier4_translation/prompts/chemistry_agent.md",
    "clinical_trialist_agent":     "agents/tier4_translation/prompts/clinical_trialist_agent.md",
    "scientific_writer_agent":     "agents/tier5_writer/prompts/scientific_writer_agent.md",
}

# ---------------------------------------------------------------------------
# Per-agent MCP tool category assignments (ToolUniverse categories)
# Used when building the tool list for SDK mode.
# ---------------------------------------------------------------------------

AGENT_TOOL_CATEGORIES: dict[str, list[str]] = {
    "phenotype_architect": [
        "opentarget", "disease_target_score", "orphanet", "gnomad",
    ],
    "statistical_geneticist": [
        "gwas", "gnomad", "gtex_v2", "ensembl",
    ],
    "somatic_exposure_agent": [
        "cbioportal", "civic", "epigenomics",
    ],
    "perturbation_genomics_agent": [
        "hpa", "software_single_cell", "ensembl",
    ],
    "regulatory_genomics_agent": [
        "reactome", "intact", "proteins_api", "ensembl",
    ],
    "causal_discovery_agent": [
        "opentarget", "disease_target_score", "gnomad", "gwas",
    ],
    "kg_completion_agent": [
        "openalex", "opentarget", "reactome", "intact",
    ],
    "target_prioritization_agent": [
        "opentarget", "disease_target_score", "clinical_trials", "gnomad",
    ],
    "chemistry_agent": [
        "chembl", "pubchem", "admetai",
    ],
    "clinical_trialist_agent": [
        "clinical_trials", "ada_aha_nccn", "guidelines",
    ],
    "scientific_writer_agent": [
        "openalex",
    ],
}

# ---------------------------------------------------------------------------
# Tool schemas for causal_discovery_agent SDK mode
# ---------------------------------------------------------------------------

_CAUSAL_DISCOVERY_TOOLS: list[dict] = [
    {
        "name": "compute_ota_gammas",
        "description": (
            "Run the full Ota composite γ computation + SCONE sensitivity reweighting "
            "for all genes. Returns gene_gamma_records (one per gene×trait), "
            "anchor_gene_set (validated disease anchors), required_anchors, and warnings. "
            "Call this first to get all scored edge candidates."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "beta_matrix_result": {
                    "type": "object",
                    "description": "Output of perturbation_genomics_agent: {genes, beta_matrix, evidence_tier_per_gene, programs}",
                },
                "gamma_estimates": {
                    "type": "object",
                    "description": "Program→trait γ matrix: {program: {trait: float}}",
                },
                "disease_query": {
                    "type": "object",
                    "description": "Disease context dict with disease_name, efo_id, etc.",
                },
            },
            "required": ["beta_matrix_result", "gamma_estimates", "disease_query"],
        },
    },
    {
        "name": "write_causal_edges",
        "description": (
            "Write a list of selected causal edges to the Kùzu graph database. "
            "Each edge must have from_node, from_type, to_node, to_type, effect_size, "
            "evidence_type, evidence_tier, method, data_source fields. "
            "Call after selecting which gene_gamma_records pass your inclusion criteria."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "edges": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "List of CausalEdge dicts to write.",
                },
                "disease_name": {
                    "type": "string",
                    "description": "Disease name, e.g. 'coronary artery disease'",
                },
            },
            "required": ["edges", "disease_name"],
        },
    },
    {
        "name": "check_anchor_recovery",
        "description": (
            "Check what fraction of required disease anchor edges are present in written_edges. "
            "Also queries the DB for previously-written edges (somatic/CHIP from Tier 1). "
            "Returns recovery_rate, recovered list, missing list, required_anchors. "
            "Minimum acceptable recovery_rate is 0.80."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "written_edges": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "CausalEdge dicts written so far (from_node, to_node required).",
                },
                "disease_query": {
                    "type": "object",
                    "description": "Disease context dict.",
                },
            },
            "required": ["written_edges", "disease_query"],
        },
    },
    {
        "name": "compute_shd",
        "description": (
            "Compute Structural Hamming Distance between the predicted causal graph "
            "and the reference anchor graph for this disease. "
            "Returns shd, extra_edges (false positives), missing_edges (false negatives)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "predicted_edges": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "List of {from_node, to_node} dicts.",
                },
                "disease_query": {
                    "type": "object",
                    "description": "Disease context dict.",
                },
            },
            "required": ["predicted_edges", "disease_query"],
        },
    },
]


# Structured output tool — every SDK agent returns its result via this tool
_RETURN_RESULT_TOOL = {
    "name": "return_result",
    "description": (
        "Return the final structured result for this agent. "
        "Call this once when you have completed your analysis."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "result": {
                "type": "object",
                "description": "The agent's complete output as a JSON object.",
            },
            "warnings": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Any warnings or quality flags from this agent.",
            },
            "edges_written": {
                "type": "integer",
                "description": "Number of causal edges written to the graph DB.",
                "default": 0,
            },
        },
        "required": ["result"],
    },
}


# ---------------------------------------------------------------------------
# AgentRunner
# ---------------------------------------------------------------------------

class AgentRunner:
    """
    Dispatch agent calls in local or SDK mode.

    Local mode:  import and call existing agent run() functions.
    SDK mode:    call Claude via Anthropic API with tool_use.

    Agents default to local mode. Flip individual agents to SDK mode via
    set_mode() to enable gradual migration.
    """

    DEFAULT_MODEL = "claude-opus-4-6"
    MAX_RETRIES = 2

    def __init__(
        self,
        model: str | None = None,
        project_root: str | Path | None = None,
    ) -> None:
        self._model = model or self.DEFAULT_MODEL
        self._project_root = Path(project_root or Path(__file__).parent.parent)
        self._modes: dict[str, str] = {}       # agent_name → "local" | "sdk"
        self._client: Any = None               # lazy-init Anthropic client

    # ------------------------------------------------------------------
    # Mode management
    # ------------------------------------------------------------------

    def set_mode(self, agent_name: str, mode: str) -> None:
        """Set an agent to 'local' or 'sdk' mode."""
        if mode not in ("local", "sdk"):
            raise ValueError(f"mode must be 'local' or 'sdk', got {mode!r}")
        self._modes[agent_name] = mode

    def set_all_sdk(self) -> None:
        """Switch every agent to SDK mode (full multiagent deployment)."""
        for name in _AGENT_MODULES:
            self._modes[name] = "sdk"

    def get_mode(self, agent_name: str) -> str:
        return self._modes.get(agent_name, "local")

    # ------------------------------------------------------------------
    # Primary dispatch
    # ------------------------------------------------------------------

    def dispatch(self, agent_name: str, agent_input: AgentInput) -> AgentOutput:
        """
        Run an agent and return a typed AgentOutput envelope.

        Args:
            agent_name:   One of the registered agent names.
            agent_input:  Typed AgentInput envelope.

        Returns:
            AgentOutput with results, warnings, edges_written, etc.
        """
        mode = self.get_mode(agent_name)
        if mode == "sdk":
            return self._dispatch_sdk(agent_name, agent_input)
        return self._dispatch_local(agent_name, agent_input)

    # ------------------------------------------------------------------
    # Local dispatch (wraps existing run() functions)
    # ------------------------------------------------------------------

    def _dispatch_local(self, agent_name: str, agent_input: AgentInput) -> AgentOutput:
        """Call the agent's existing run() function with retry."""
        module_path = _AGENT_MODULES.get(agent_name)
        if not module_path:
            # Unknown agent name can't pass the AgentName Literal validator —
            # construct the envelope directly, bypassing field validation.
            return AgentOutput.model_construct(
                tier="tier1",
                agent_name=agent_name,
                results={"error": f"Unknown agent: {agent_name}"},
                edges_written=0,
                warnings=[f"Unknown agent: {agent_name}"],
                escalate=False,
                escalation_reason=None,
                duration_s=None,
                stub_fallback=True,
            )

        last_exc: Exception | None = None
        for attempt in range(self.MAX_RETRIES):
            try:
                t0 = time.time()
                raw = self._call_local(agent_name, module_path, agent_input)
                duration = time.time() - t0
                edges = raw.get("n_edges_written", 0) + raw.get("edges_written", 0)
                return wrap_output(
                    agent_name, raw,
                    edges_written=edges,
                    duration_s=round(duration, 2),
                )
            except Exception as exc:
                last_exc = exc
                if attempt < self.MAX_RETRIES - 1:
                    print(f"[RETRY] {agent_name}: {exc}")

        return wrap_output(
            agent_name,
            {"error": str(last_exc), "warnings": [f"{agent_name} failed: {last_exc}"]},
            stub_fallback=True,
        )

    def _call_local(self, agent_name: str, module_path: str, inp: AgentInput) -> dict:
        """Import and call the agent's run() function with the right arguments."""
        mod = importlib.import_module(module_path)
        run_fn = mod.run
        dq = inp.disease_query
        up = inp.upstream_results

        # Match each agent's existing run() signature
        if agent_name == "phenotype_architect":
            return run_fn(dq.get("disease_name", ""))

        if agent_name == "statistical_geneticist":
            return run_fn(dq)

        if agent_name == "somatic_exposure_agent":
            return run_fn(dq)

        if agent_name == "perturbation_genomics_agent":
            gene_list = up.get("_gene_list", [])
            return run_fn(gene_list, dq)

        if agent_name == "regulatory_genomics_agent":
            gene_list = up.get("_gene_list", [])
            return run_fn(gene_list, dq)

        if agent_name == "causal_discovery_agent":
            beta_result = up.get("perturbation_genomics_agent", {})
            gamma_estimates = up.get("_gamma_estimates", {})
            return run_fn(beta_result, gamma_estimates, dq)

        if agent_name == "kg_completion_agent":
            causal_result = up.get("causal_discovery_agent", {})
            return run_fn(causal_result, dq)

        if agent_name == "target_prioritization_agent":
            causal_result = up.get("causal_discovery_agent", {})
            kg_result = up.get("kg_completion_agent", {})
            return run_fn(causal_result, kg_result, dq)

        if agent_name == "chemistry_agent":
            prioritization_result = up.get("target_prioritization_agent", {})
            return run_fn(prioritization_result, dq)

        if agent_name == "clinical_trialist_agent":
            prioritization_result = up.get("target_prioritization_agent", {})
            return run_fn(prioritization_result, dq)

        if agent_name == "scientific_writer_agent":
            writer_keys = [
                "phenotype_result", "genetics_result", "somatic_result",
                "beta_matrix_result", "regulatory_result", "causal_result",
                "kg_result", "prioritization_result", "chemistry_result",
                "trials_result",
            ]
            return run_fn(**{k: up.get(k, {}) for k in writer_keys})

        if agent_name == "scientific_reviewer_agent":
            # Reviewer receives the full pipeline_outputs dict + disease_query
            return run_fn(pipeline_outputs=up, disease_query=dq)

        raise ValueError(f"No local dispatch mapping for agent: {agent_name}")

    # ------------------------------------------------------------------
    # SDK dispatch (Claude Agent SDK via Anthropic API)
    # ------------------------------------------------------------------

    def _dispatch_sdk(self, agent_name: str, agent_input: AgentInput) -> AgentOutput:
        """
        Call Claude as a subagent using the Anthropic messages API.

        The agent receives its system prompt + input JSON, calls ToolUniverse
        and local MCP tools, then returns structured output via return_result().
        """
        system_prompt = self._load_system_prompt(agent_name)
        tools = self._build_tool_list(agent_name)
        user_message = json.dumps(agent_input.model_dump(), indent=2, default=str)

        messages = [{"role": "user", "content": user_message}]

        last_exc: Exception | None = None
        for attempt in range(self.MAX_RETRIES):
            try:
                client = self._get_client()
                t0 = time.time()
                raw = self._agentic_loop(
                    client, system_prompt, messages, tools, agent_name
                )
                duration = time.time() - t0
                edges = raw.get("edges_written", 0)
                return wrap_output(
                    agent_name, raw.get("result", raw),
                    edges_written=edges,
                    duration_s=round(duration, 2),
                )
            except Exception as exc:
                last_exc = exc
                if attempt < self.MAX_RETRIES - 1:
                    print(f"[SDK_RETRY] {agent_name}: {exc}")

        return wrap_output(
            agent_name,
            {"error": str(last_exc), "warnings": [f"{agent_name} SDK call failed: {last_exc}"]},
            stub_fallback=True,
        )

    def _agentic_loop(
        self,
        client: Any,
        system_prompt: str,
        messages: list[dict],
        tools: list[dict],
        agent_name: str,
        max_turns: int = 20,
    ) -> dict:
        """
        Run the agent loop: model calls tools until return_result is invoked.

        Each tool call is executed and the result is fed back. The loop ends
        when the model calls return_result (structured output) or max_turns is
        reached.
        """
        current_messages = list(messages)

        for _ in range(max_turns):
            response = client.messages.create(
                model=self._model,
                max_tokens=8192,
                system=system_prompt,
                tools=tools,
                messages=current_messages,
            )

            # Append assistant turn
            current_messages.append({
                "role": "assistant",
                "content": response.content,
            })

            # Check stop reason
            if response.stop_reason == "end_turn":
                # No tool call — extract text and wrap as result
                text = " ".join(
                    block.text for block in response.content
                    if hasattr(block, "text")
                )
                return {"result": {"summary": text}, "warnings": []}

            if response.stop_reason != "tool_use":
                break

            # Process tool calls
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue

                if block.name == "return_result":
                    # Agent is done — return its structured output
                    return block.input

                # Execute other tool calls (ToolUniverse / local MCPs)
                tool_result = self._execute_tool(block.name, block.input, agent_name)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(tool_result, default=str),
                })

            if tool_results:
                current_messages.append({"role": "user", "content": tool_results})

        return {"result": {}, "warnings": [f"{agent_name}: max_turns={max_turns} reached"]}

    def _execute_tool(self, tool_name: str, tool_input: dict, agent_name: str) -> dict:
        """
        Execute a tool call made by the Claude subagent.

        Routes to local MCP server functions. ToolUniverse tools are handled
        by the MCP server process and don't need manual routing here.
        """
        # Local MCP server routing
        local_routes = self._get_local_tool_routes()
        if tool_name in local_routes:
            try:
                return local_routes[tool_name](**tool_input)
            except Exception as exc:
                return {"error": str(exc), "tool": tool_name}

        return {"error": f"Unknown tool: {tool_name}", "tool": tool_name}

    def _get_local_tool_routes(self) -> dict[str, Any]:
        """Return a map of local MCP tool names to callable functions."""
        routes: dict[str, Any] = {}
        try:
            from mcp_servers import graph_db_server as gdb
            routes.update({
                "write_causal_edges":         gdb.write_causal_edges,
                "query_graph_for_disease":    gdb.query_graph_for_disease,
                "run_anchor_edge_validation": gdb.run_anchor_edge_validation,
                "compute_shd_metric":         gdb.compute_shd_metric,
                "run_evalue_check":           gdb.run_evalue_check,
            })
        except ImportError:
            pass
        try:
            from mcp_servers import gwas_genetics_server as gwas
            routes.update({
                "get_gwas_catalog_associations": gwas.get_gwas_catalog_associations,
                "query_gnomad_lof_constraint":   gwas.query_gnomad_lof_constraint,
                "query_gtex_eqtl":               gwas.query_gtex_eqtl,
            })
        except ImportError:
            pass
        try:
            from mcp_servers import viral_somatic_server as vs
            routes.update({
                "get_chip_disease_associations": vs.get_chip_disease_associations,
                "get_drug_exposure_mr":          vs.get_drug_exposure_mr,
            })
        except ImportError:
            pass
        # causal_discovery_agent SDK tools
        try:
            from agents.tier3_causal.sdk_tools import (
                compute_ota_gammas,
                check_anchor_recovery,
                compute_shd,
            )
            routes.update({
                "compute_ota_gammas":    compute_ota_gammas,
                "check_anchor_recovery": check_anchor_recovery,
                "compute_shd":           compute_shd,
            })
        except ImportError:
            pass
        return routes

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic()
            except ImportError as exc:
                raise RuntimeError(
                    "anthropic package required for SDK mode: pip install anthropic"
                ) from exc
            except Exception as exc:
                # Covers AuthenticationError (missing/bad ANTHROPIC_API_KEY) and
                # any other client-init failure — let the retry loop handle it.
                raise RuntimeError(
                    f"Anthropic client init failed (check ANTHROPIC_API_KEY): {exc}"
                ) from exc
        return self._client

    def _load_system_prompt(self, agent_name: str) -> str:
        prompt_path = self._project_root / _PROMPT_PATHS.get(agent_name, "")
        if prompt_path.exists():
            return prompt_path.read_text()
        return (
            f"You are the {agent_name} agent in a causal genomics pipeline. "
            "Analyze the input data and return structured results using the return_result tool."
        )

    def _build_tool_list(self, agent_name: str) -> list[dict]:
        """
        Build the tool list for an SDK agent call.

        Includes return_result (required) plus per-agent tool schemas.
        """
        tools = [_RETURN_RESULT_TOOL]
        if agent_name == "causal_discovery_agent":
            tools.extend(_CAUSAL_DISCOVERY_TOOLS)
        return tools
