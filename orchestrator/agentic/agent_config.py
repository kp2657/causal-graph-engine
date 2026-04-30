"""
agent_config.py — Configuration constants for the agentic pipeline extension.

AGENT_MODE=agentic activates the agent layer on top of the deterministic pipeline.
AGENT_MODE=local (default) runs the deterministic pipeline only, no API cost.
"""
from __future__ import annotations

import os
from typing import Literal

# ---------------------------------------------------------------------------
# Mode flag
# ---------------------------------------------------------------------------

AGENT_MODE: Literal["local", "agentic"] = os.getenv("AGENT_MODE", "local")  # type: ignore[assignment]


def is_agentic() -> bool:
    return AGENT_MODE == "agentic"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

AGENTIC_MODEL = "claude-sonnet-4-6"

# ---------------------------------------------------------------------------
# Pricing — Sonnet 4.6 as of 2026-04 (USD per token)
# Update if Anthropic changes pricing.
# ---------------------------------------------------------------------------

INPUT_COST_PER_TOKEN: float = 3.00 / 1_000_000
OUTPUT_COST_PER_TOKEN: float = 15.00 / 1_000_000


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens * INPUT_COST_PER_TOKEN) + (output_tokens * OUTPUT_COST_PER_TOKEN)


# ---------------------------------------------------------------------------
# Agent names — used as identifiers in journal and token tracking
# ---------------------------------------------------------------------------

MASTER_ORCHESTRATOR = "master_orchestrator"
CHIEF_OF_STAFF = "chief_of_staff"
STATISTICAL_GENETICIST = "statistical_geneticist"
CONVERGENT_EVIDENCE_AGENT = "convergent_evidence_agent"
CHEMISTRY_AGENT = "chemistry_agent"
CLINICAL_TRIALIST = "clinical_trialist"
DISCOVERY_REFINEMENT_AGENT = "discovery_refinement_agent"
RED_TEAM = "red_team"
SCIENTIFIC_REVIEWER = "scientific_reviewer"

ALL_AGENTS = [
    MASTER_ORCHESTRATOR,
    CHIEF_OF_STAFF,
    STATISTICAL_GENETICIST,
    CONVERGENT_EVIDENCE_AGENT,
    CHEMISTRY_AGENT,
    CLINICAL_TRIALIST,
    DISCOVERY_REFINEMENT_AGENT,
    RED_TEAM,
    SCIENTIFIC_REVIEWER,
]
