"""
agent_contracts.py — Pydantic v2 models for agentic pipeline I/O.

Every agent receives an AgentRunInput and returns a subclass of AgentRunOutput.
Shared evidence types enforce the tier hierarchy defined in the design doc.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared vocabulary
# ---------------------------------------------------------------------------

Confidence = Literal["HIGH", "MEDIUM", "LOW"]
Priority = Literal["HIGH", "MEDIUM", "LOW"]
ReviewerVerdict = Literal["APPROVE", "REVISE", "HARD_REJECT"]

# Evidence tier hierarchy — convergent evidence requires legs from different tiers.
# Tier A: genetic / clinical (strongest causal grounding)
# Tier B: orthogonal perturbation in a different cell system
# Tier C: correlational / same experimental ecosystem (GPS, DepMap, more Perturb-seq)
# GPS reversal + Perturb-seq β = both Tier C = replication, not convergence.
EvidenceTier = Literal["A", "B", "C"]

RedelegationTag = Literal["T1", "T2", "T4-chem", "T4-trial", "DRA"]


# ---------------------------------------------------------------------------
# Token tracking
# ---------------------------------------------------------------------------

class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        from orchestrator.agentic.agent_config import estimate_cost
        inp = self.input_tokens + other.input_tokens
        out = self.output_tokens + other.output_tokens
        return TokenUsage(input_tokens=inp, output_tokens=out, cost_usd=estimate_cost(inp, out))


class AgentTokenUsage(BaseModel):
    agent_name: str
    usage: TokenUsage


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------

class EvidenceItem(BaseModel):
    gene: str
    modality_name: str
    tier: EvidenceTier
    source: str
    confidence: Confidence
    description: str
    accession: str | None = None


# ---------------------------------------------------------------------------
# Journal entry types
# ---------------------------------------------------------------------------

class MethodsChoice(BaseModel):
    agent: str
    choice: str
    rationale: str
    alternatives_considered: list[str] = Field(default_factory=list)


class PathReasoning(BaseModel):
    agent: str
    statement: str


class HumanPause(BaseModel):
    pause_id: Literal[1, 2, 3]
    cso_output: str
    user_input: str | None = None


class LibraryGap(BaseModel):
    gene: str
    known_causal_reason: str
    suggested_modalities: list[str] = Field(default_factory=list)


class AugmentationRecommendation(BaseModel):
    agent: str
    gene: str
    accession: str
    source_db: str
    description: str


class VirginTarget(BaseModel):
    gene: str
    evidence_items: list[EvidenceItem] = Field(default_factory=list)
    priority: Priority
    gate1_confirmed: bool = False
    gate2_confirmed: bool = False
    notes: str = ""

    @property
    def has_convergent_evidence(self) -> bool:
        tiers_present = {e.tier for e in self.evidence_items if e.confidence in ("HIGH", "MEDIUM")}
        # Requires at least one Tier A or B leg (not just Tier C)
        has_strong_leg = bool(tiers_present & {"A", "B"})
        # Requires legs from at least 2 different tiers
        multi_tier = len(tiers_present) >= 2
        return has_strong_leg and multi_tier


class RedelegationRecord(BaseModel):
    round_num: int
    tier_tag: RedelegationTag
    instruction: str
    target_agent: str


# ---------------------------------------------------------------------------
# Agent I/O base models
# ---------------------------------------------------------------------------

class AgentRunInput(BaseModel):
    disease_key: str
    run_id: str
    pipeline_checkpoint: dict = Field(default_factory=dict)
    prior_agent_outputs: dict[str, dict] = Field(default_factory=dict)


class AgentRunOutput(BaseModel):
    agent_name: str
    run_id: str
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    methods_choices: list[MethodsChoice] = Field(default_factory=list)
    path_reasoning: list[PathReasoning] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Per-agent output contracts
# ---------------------------------------------------------------------------

class StatisticalGeneticistOutput(AgentRunOutput):
    validated_anchors: list[str] = Field(default_factory=list)
    rejected_anchors: list[str] = Field(default_factory=list)
    upgraded_anchors: list[str] = Field(default_factory=list)
    library_gaps: list[LibraryGap] = Field(default_factory=list)
    anchor_set_suspect: bool = False
    anchor_set_suspect_hypothesis: str = ""


class ConvergentEvidenceOutput(AgentRunOutput):
    confirmed_genes: list[str] = Field(default_factory=list)
    no_evidence_genes: list[str] = Field(default_factory=list)
    evidence_items: list[EvidenceItem] = Field(default_factory=list)
    augmentation_recommendations: list[AugmentationRecommendation] = Field(default_factory=list)


class ChemistryAgentOutput(AgentRunOutput):
    mechanism_linked: list[str] = Field(default_factory=list)
    mechanism_unknown: list[str] = Field(default_factory=list)
    rejected_repurposing: list[str] = Field(default_factory=list)
    tractability: dict[str, Literal["chemically_tractable", "needs_chemical_probe"]] = Field(default_factory=dict)


class ClinicalTrialistOutput(AgentRunOutput):
    efficacy_failures: list[str] = Field(default_factory=list)
    priority_repurposing: list[str] = Field(default_factory=list)
    no_active_development: list[str] = Field(default_factory=list)
    active_development_found: list[str] = Field(default_factory=list)
    insufficient_data: list[str] = Field(default_factory=list)


class DiscoveryRefinementOutput(AgentRunOutput):
    virgin_targets: list[VirginTarget] = Field(default_factory=list)
    downgraded_to_known: list[str] = Field(default_factory=list)


class RedTeamOutput(AgentRunOutput):
    hard_rejects: list[str] = Field(default_factory=list)
    counterarguments: list[dict] = Field(default_factory=list)


class ReviewerOutput(AgentRunOutput):
    verdict: ReviewerVerdict = "APPROVE"
    redelegation_instructions: list[RedelegationRecord] = Field(default_factory=list)
    hard_rejected_targets: list[str] = Field(default_factory=list)


class CSOOutput(AgentRunOutput):
    mode: Literal["briefing", "conflict_analysis", "exec_summary"]
    content: str = ""
    run_paused: bool = False
    pause_reason: str = ""
    downgraded_candidates: list[str] = Field(default_factory=list)
    finngen_auc: float | None = None
