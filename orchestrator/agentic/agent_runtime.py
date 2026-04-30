"""
agent_runtime.py — Anthropic SDK wrapper with token tracking for the agentic pipeline.

Supports both simple single-turn calls and tool-use loops.
Each agent call accumulates token usage and returns (final_text, TokenUsage).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import anthropic

from orchestrator.agentic.agent_config import AGENTIC_MODEL, estimate_cost
from orchestrator.agentic.agent_contracts import TokenUsage

logger = logging.getLogger(__name__)

_MAX_TOOL_ITERATIONS = 10


@dataclass
class AgentSession:
    """Accumulates token usage across multiple API calls for a single agent invocation."""
    agent_name: str
    input_tokens: int = 0
    output_tokens: int = 0
    _client: anthropic.Anthropic = field(default_factory=anthropic.Anthropic, repr=False)

    def call(
        self,
        system: str,
        messages: list[dict[str, Any]],
        max_tokens: int = 8192,
        tools: list[dict] | None = None,
    ) -> anthropic.types.Message:
        kwargs: dict[str, Any] = dict(
            model=AGENTIC_MODEL,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
        )
        if tools:
            kwargs["tools"] = tools
        response = self._client.messages.create(**kwargs)
        self.input_tokens += response.usage.input_tokens
        self.output_tokens += response.usage.output_tokens
        return response

    def run_with_tools(
        self,
        system: str,
        user_message: str,
        tools: list[dict],
        tool_functions: dict[str, Callable],
        max_tokens: int = 8192,
    ) -> str:
        """
        Tool-use loop. Executes tool calls until stop_reason == 'end_turn'.
        Returns the final text content from the assistant.
        """
        messages: list[dict[str, Any]] = [{"role": "user", "content": user_message}]

        for _ in range(_MAX_TOOL_ITERATIONS):
            response = self.call(system, messages, max_tokens=max_tokens, tools=tools)

            if response.stop_reason == "end_turn":
                texts = [b.text for b in response.content if hasattr(b, "text")]
                return "\n".join(texts)

            if response.stop_reason == "tool_use":
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        fn = tool_functions.get(block.name)
                        if fn is None:
                            result = {"error": f"unknown tool: {block.name}"}
                        else:
                            try:
                                result = fn(**block.input)
                            except Exception as exc:
                                result = {"error": str(exc)}
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result, default=str),
                        })
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})
            else:
                # Unexpected stop reason
                break

        # If we exit the loop without end_turn, return whatever text we have
        texts = [b.text for b in response.content if hasattr(b, "text")]
        return "\n".join(texts)

    @property
    def token_usage(self) -> TokenUsage:
        return TokenUsage(
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            cost_usd=estimate_cost(self.input_tokens, self.output_tokens),
        )


def run_agent(
    agent_name: str,
    system_prompt: str,
    user_message: str,
    max_tokens: int = 8192,
) -> tuple[str, TokenUsage]:
    """Single-turn agent call (no tools). Returns (response_text, token_usage)."""
    session = AgentSession(agent_name=agent_name)
    response = session.call(
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
        max_tokens=max_tokens,
    )
    texts = [b.text for b in response.content if hasattr(b, "text")]
    return "\n".join(texts), session.token_usage


def run_agent_with_tools(
    agent_name: str,
    system_prompt: str,
    user_message: str,
    tools: list[dict],
    tool_functions: dict[str, Callable],
    max_tokens: int = 8192,
) -> tuple[str, TokenUsage]:
    """Tool-use agent call. Returns (final_text, token_usage)."""
    session = AgentSession(agent_name=agent_name)
    text = session.run_with_tools(
        system=system_prompt,
        user_message=user_message,
        tools=tools,
        tool_functions=tool_functions,
        max_tokens=max_tokens,
    )
    return text, session.token_usage
