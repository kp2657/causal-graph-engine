"""
message_contracts.py — Typed pydantic v2 envelopes for all tier agents.

Every agent in the multiagent system accepts an AgentInput and returns an
AgentOutput. These contracts are enforced at dispatch time so format
violations are caught before they reach the graph, not during anchor
validation at the end.

Usage (local mode — existing pipeline):
    from orchestrator.message_contracts import wrap_output, T1SomaticOutput
    result = run(disease_query)
    output = wrap_output("somatic_exposure_agent", "tier1", result)

Usage (Agent SDK mode — multiagent):
    from orchestrator.agent_runner import AgentRunner
    runner = AgentRunner()
    output = runner.run("somatic_exposure_agent", input_envelope)
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Shared envelope base classes
# ---------------------------------------------------------------------------

AgentName = Literal[
    "phenotype_architect",
    "statistical_geneticist",
    "somatic_exposure_agent",
    "perturbation_genomics_agent",
    "regulatory_genomics_agent",
    "causal_discovery_agent",
    "kg_completion_agent",
    "target_prioritization_agent",
    "chemistry_agent",
    "clinical_trialist_agent",
    "scientific_writer_agent",
    "scientific_reviewer_agent",
    "literature_validation_agent",
    "chief_of_staff_agent",
    "red_team_agent",
    "discovery_refinement_agent",
]

TierName = Literal["tier1", "tier2", "tier3", "tier4", "tier5"]


class AgentInput(BaseModel):
    """Base input envelope passed to every agent."""
    disease_query: dict                    # DiseaseQuery.model_dump()
    upstream_results: dict = {}            # outputs from prior tiers, keyed by agent name
    graph_snapshot: dict = {}             # lightweight Kùzu state summary
    run_id: str = Field(default_factory=lambda: datetime.now(tz=timezone.utc).isoformat())
    mode: Literal["local", "sdk"] = "local"  # local = call run(); sdk = Claude API


class AgentOutput(BaseModel):
    """Base output envelope returned by every agent."""
    tier: TierName
    agent_name: AgentName
    results: dict                          # tier-specific payload (see typed subclasses)
    edges_written: int = 0
    warnings: list[str] = []
    escalate: bool = False
    escalation_reason: str | None = None
    duration_s: float | None = None
    stub_fallback: bool = False

    @model_validator(mode="after")
    def set_escalate_from_warnings(self) -> "AgentOutput":
        if not self.escalate:
            self.escalate = any(
                kw in w for w in self.warnings
                for kw in ("ESCALATE", "CRITICAL", "HALT")
            )
        return self


# ---------------------------------------------------------------------------
# Tier 1 — Phenomics
# ---------------------------------------------------------------------------

class T1PhenotypeOutput(AgentOutput):
    tier: TierName = "tier1"
    agent_name: AgentName = "phenotype_architect"

    class Results(BaseModel):
        disease_name: str
        efo_id: str | None = None
        icd10_codes: list[str] = []
        modifier_types: list[str] = []
        primary_gwas_id: str | None = None
        n_gwas_studies: int = 0
        finngen_phenocode: str | None = None
        use_precomputed_only: bool = True
        day_one_mode: bool = True

    results: dict  # validated downstream as T1PhenotypeOutput.Results


class T1GeneticsOutput(AgentOutput):
    tier: TierName = "tier1"
    agent_name: AgentName = "statistical_geneticist"

    class Results(BaseModel):
        instruments: list[dict] = []
        anchor_genes_validated: dict[str, bool] = {}
        n_gw_significant_hits: int = 0
        warnings: list[str] = []

    results: dict


class T1SomaticOutput(AgentOutput):
    tier: TierName = "tier1"
    agent_name: AgentName = "somatic_exposure_agent"

    class Results(BaseModel):
        chip_edges: list[dict] = []
        viral_edges: list[dict] = []
        drug_edges: list[dict] = []
        summary: dict = {}
        warnings: list[str] = []

    results: dict


# ---------------------------------------------------------------------------
# Tier 2 — Pathway
# ---------------------------------------------------------------------------

class T2PerturbationOutput(AgentOutput):
    tier: TierName = "tier2"
    agent_name: AgentName = "perturbation_genomics_agent"

    class Results(BaseModel):
        genes: list[str] = []
        programs: list[dict] = []
        beta_matrix: dict[str, dict] = {}
        evidence_tier_per_gene: dict[str, str] = {}
        n_tier1: int = 0
        n_tier2: int = 0
        n_tier3: int = 0
        n_virtual: int = 0
        warnings: list[str] = []

    results: dict


class T2RegulatoryOutput(AgentOutput):
    tier: TierName = "tier2"
    agent_name: AgentName = "regulatory_genomics_agent"

    class Results(BaseModel):
        gene_eqtl_summary: dict[str, Any] = {}
        gene_program_overlap: dict[str, Any] = {}
        tier2_upgrades: list[dict] = []
        warnings: list[str] = []

    results: dict


# ---------------------------------------------------------------------------
# Tier 3 — Causal
# ---------------------------------------------------------------------------

class T3CausalOutput(AgentOutput):
    tier: TierName = "tier3"
    agent_name: AgentName = "causal_discovery_agent"

    class Results(BaseModel):
        n_edges_written: int = 0
        n_edges_rejected: int = 0
        top_genes: list[dict] = []
        anchor_recovery: dict = {}         # {recovery_rate, recovered, missing}
        shd: int = 0
        warnings: list[str] = []

    results: dict


class T3KGOutput(AgentOutput):
    tier: TierName = "tier3"
    agent_name: AgentName = "kg_completion_agent"

    class Results(BaseModel):
        n_pathway_edges_added: int = 0
        n_ppi_edges_added: int = 0
        n_drug_target_edges_added: int = 0
        n_primekg_edges_added: int = 0
        top_pathways: list[dict] = []
        drug_target_summary: list[dict] = []
        contradictions_flagged: list[dict] = []
        warnings: list[str] = []

    results: dict


# ---------------------------------------------------------------------------
# Tier 4 — Translation
# ---------------------------------------------------------------------------

class T4PrioritizationOutput(AgentOutput):
    tier: TierName = "tier4"
    agent_name: AgentName = "target_prioritization_agent"

    class Results(BaseModel):
        targets: list[dict] = []           # list of TargetRecord-compatible dicts
        warnings: list[str] = []

    results: dict


class T4ChemistryOutput(AgentOutput):
    tier: TierName = "tier4"
    agent_name: AgentName = "chemistry_agent"

    class Results(BaseModel):
        target_chemistry: list[dict] = []
        repurposing_candidates: list[dict] = []
        warnings: list[str] = []

    results: dict


class T4ClinicalOutput(AgentOutput):
    tier: TierName = "tier4"
    agent_name: AgentName = "clinical_trialist_agent"

    class Results(BaseModel):
        trial_summary: dict = {}
        key_trials: list[dict] = []
        development_risk: dict = {}
        repurposing_opportunities: list[dict] = []
        warnings: list[str] = []

    results: dict


# ---------------------------------------------------------------------------
# Tier 5 — Writer
# ---------------------------------------------------------------------------

class T5WriterOutput(AgentOutput):
    tier: TierName = "tier5"
    agent_name: AgentName = "scientific_writer_agent"

    class Results(BaseModel):
        disease_name: str = ""
        efo_id: str | None = None
        target_list: list[dict] = []
        anchor_edge_recovery: float = 0.0
        n_tier1_edges: int = 0
        n_tier2_edges: int = 0
        n_tier3_edges: int = 0
        n_virtual_edges: int = 0
        executive_summary: str = ""
        target_table: list[dict] = []
        top_target_narratives: list[dict] = []
        evidence_quality: dict = {}
        limitations: list[str] = []
        pipeline_version: str = "0.1.0"
        generated_at: str = ""
        warnings: list[str] = []

    results: dict


# ---------------------------------------------------------------------------
# Convenience: wrap a raw agent result dict into a typed AgentOutput
# ---------------------------------------------------------------------------

_OUTPUT_CLASSES: dict[str, type[AgentOutput]] = {
    "phenotype_architect":        T1PhenotypeOutput,
    "statistical_geneticist":     T1GeneticsOutput,
    "somatic_exposure_agent":     T1SomaticOutput,
    "perturbation_genomics_agent": T2PerturbationOutput,
    "regulatory_genomics_agent":  T2RegulatoryOutput,
    "causal_discovery_agent":     T3CausalOutput,
    "kg_completion_agent":        T3KGOutput,
    "target_prioritization_agent": T4PrioritizationOutput,
    "chemistry_agent":            T4ChemistryOutput,
    "clinical_trialist_agent":    T4ClinicalOutput,
    "scientific_writer_agent":    T5WriterOutput,
    "scientific_reviewer_agent":  T5WriterOutput,  # reuse writer output shape (generic results dict)
}

_TIER_MAP: dict[str, TierName] = {
    "phenotype_architect":        "tier1",
    "statistical_geneticist":     "tier1",
    "somatic_exposure_agent":     "tier1",
    "perturbation_genomics_agent": "tier2",
    "regulatory_genomics_agent":  "tier2",
    "causal_discovery_agent":     "tier3",
    "kg_completion_agent":        "tier3",
    "target_prioritization_agent": "tier4",
    "chemistry_agent":            "tier4",
    "clinical_trialist_agent":    "tier4",
    "scientific_writer_agent":    "tier5",
    "scientific_reviewer_agent":  "tier5",
}


def wrap_output(
    agent_name: str,
    raw_result: dict,
    *,
    edges_written: int = 0,
    duration_s: float | None = None,
    stub_fallback: bool = False,
) -> AgentOutput:
    """
    Wrap a raw agent result dict into a typed AgentOutput envelope.

    Args:
        agent_name:    One of the AgentName literals.
        raw_result:    The dict returned by the agent's run() function.
        edges_written: Number of graph edges written (if known).
        duration_s:    Wall-clock time for the agent call.
        stub_fallback: True if the agent returned a fallback result.

    Returns:
        Typed AgentOutput subclass for this agent.
    """
    cls = _OUTPUT_CLASSES.get(agent_name, AgentOutput)
    tier = _TIER_MAP.get(agent_name, "tier1")

    warnings = raw_result.get("warnings", [])
    if stub_fallback:
        warnings = [raw_result.get("error", "unknown error")] + warnings

    return cls(
        tier=tier,
        agent_name=agent_name,  # type: ignore[arg-type]
        results=raw_result,
        edges_written=edges_written,
        warnings=warnings,
        duration_s=duration_s,
        stub_fallback=stub_fallback,
    )
