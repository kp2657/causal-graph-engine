"""
agentic_orchestrator.py — Run lifecycle for the agentic pipeline extension.

Flow:
    1. CSO briefing          → PAUSE 1 (user reviews, decides to proceed)
    2. Statistical Geneticist + Convergent Evidence Agent  (Phase 3)
    3. CSO conflict_analysis → PAUSE 2
    4. Chemistry Agent + Clinical Trialist                 (Phase 4)
    5. Discovery Refinement Agent + Red Team + Reviewer    (Phase 5)
    6. CSO exec_summary      → PAUSE 3

Agents not yet implemented return empty stub outputs. The orchestrator structure
and pause points are fully live from Phase 2 onward.

Usage:
    from orchestrator.agentic.agentic_orchestrator import run_agentic
    result = run_agentic("coronary artery disease", auto_continue=False)
"""
from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path
from typing import Any

from orchestrator.agentic.agent_config import (
    CHEMISTRY_AGENT,
    CHIEF_OF_STAFF,
    CLINICAL_TRIALIST,
    CONVERGENT_EVIDENCE_AGENT,
    DISCOVERY_REFINEMENT_AGENT,
    RED_TEAM,
    SCIENTIFIC_REVIEWER,
    STATISTICAL_GENETICIST,
)
from orchestrator.agentic.agent_contracts import (
    AgentTokenUsage,
    ChemistryAgentOutput,
    ClinicalTrialistOutput,
    ConvergentEvidenceOutput,
    DiscoveryRefinementOutput,
    HumanPause,
    RedTeamOutput,
    ReviewerOutput,
    StatisticalGeneticistOutput,
)
from orchestrator.agentic.agents.chemistry_agent import run as run_chemistry
from orchestrator.agentic.agents.clinical_trialist import run as run_trialist
from orchestrator.agentic.agents.convergent_evidence_agent import run as run_convergent_evidence
from orchestrator.agentic.agents.discovery_refinement_agent import run as run_dra
from orchestrator.agentic.agents.red_team import run as run_red_team_agent
from orchestrator.agentic.agents.scientific_reviewer import run as run_reviewer_agent
from orchestrator.agentic.agents.statistical_geneticist import run as run_stat_gen
from orchestrator.agentic.chief_of_staff_agent import (
    run_briefing,
    run_conflict_analysis,
    run_exec_summary,
)
from orchestrator.agentic.run_journal import RunJournal


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_checkpoint(disease_name: str, data_dir: str = "data") -> dict:
    slug = disease_name.lower().replace(" ", "_").replace("-", "_")
    path = Path(data_dir) / f"analyze_{slug}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No pipeline output found at {path}. "
            f"Run `analyze_disease_v2` or `run_tier4` for '{disease_name}' first."
        )
    with path.open() as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Pause mechanism
# ---------------------------------------------------------------------------

def _pause(
    journal: RunJournal,
    pause_id: int,
    cso_output_text: str,
    auto_continue: bool,
) -> str:
    print(f"\n{'='*70}")
    print(f"PAUSE {pause_id} — Chief of Staff")
    print(f"{'='*70}")
    print(cso_output_text)
    print(f"\n{'='*70}")

    if auto_continue:
        user_input = "CONTINUE"
        print(f"[auto_continue=True] Proceeding automatically.")
    else:
        print("Enter your response (or 'ABORT' to stop the run):")
        user_input = input("> ").strip()

    journal.log_pause(HumanPause(
        pause_id=pause_id,  # type: ignore[arg-type]
        cso_output=cso_output_text,
        user_input=user_input,
    ))
    return user_input


def _record_token_usage(journal: RunJournal, agent_name: str, usage: Any) -> None:
    journal.log_token_usage(AgentTokenUsage(agent_name=agent_name, usage=usage))


# ---------------------------------------------------------------------------
# Agent dispatch — thin wrappers for consistent call signature in orchestrator
# ---------------------------------------------------------------------------

def _run_statistical_geneticist(
    checkpoint: dict, run_id: str
) -> StatisticalGeneticistOutput:
    return run_stat_gen(checkpoint, run_id)


def _run_convergent_evidence_agent(
    checkpoint: dict, sg_output: StatisticalGeneticistOutput, run_id: str
) -> ConvergentEvidenceOutput:
    return run_convergent_evidence(checkpoint, sg_output, run_id)


def _run_chemistry_agent(checkpoint: dict, run_id: str) -> ChemistryAgentOutput:
    return run_chemistry(checkpoint, run_id)


def _run_clinical_trialist(
    checkpoint: dict, candidate_genes: list[str], run_id: str
) -> ClinicalTrialistOutput:
    return run_trialist(checkpoint, candidate_genes, run_id)


def _run_discovery_refinement_agent(
    checkpoint: dict,
    ce_output: ConvergentEvidenceOutput,
    trialist_output: ClinicalTrialistOutput,
    run_id: str,
) -> DiscoveryRefinementOutput:
    return run_dra(checkpoint, ce_output, trialist_output, run_id)


def _run_red_team(
    checkpoint: dict,
    dra_output: DiscoveryRefinementOutput,
    run_id: str,
) -> RedTeamOutput:
    return run_red_team_agent(dra_output, run_id)


def _run_reviewer(
    checkpoint: dict,
    dra_output: DiscoveryRefinementOutput,
    rt_output: RedTeamOutput,
    run_id: str,
) -> ReviewerOutput:
    return run_reviewer_agent(dra_output, rt_output, run_id)


# ---------------------------------------------------------------------------
# Re-delegation routing
# ---------------------------------------------------------------------------

_REDELEGATION_ROUND_LIMIT = 2

def _handle_redelegation(
    journal: RunJournal,
    reviewer_output: ReviewerOutput,
    checkpoint: dict,
    agent_outputs: dict,
    run_id: str,
    round_counts: dict[str, int],
) -> tuple[DiscoveryRefinementOutput, RedTeamOutput, ReviewerOutput]:
    """Route REVISE instructions back to the correct tier agent. Max 2 rounds per tier."""
    for rec in reviewer_output.redelegation_instructions:
        tag = rec.tier_tag
        round_counts[tag] = round_counts.get(tag, 0) + 1
        if round_counts[tag] > _REDELEGATION_ROUND_LIMIT:
            print(f"[orchestrator] Max re-delegation rounds reached for {tag} — skipping")
            continue

        from orchestrator.agentic.agent_contracts import RedelegationRecord
        journal.log_redelegation(RedelegationRecord(
            round_num=round_counts[tag],
            tier_tag=tag,
            instruction=rec.instruction,
            target_agent=rec.target_agent,
        ))
        print(f"[orchestrator] Re-delegating {tag} → {rec.target_agent}: {rec.instruction}")

    # Re-run DRA → Red Team → Reviewer after any re-delegation
    dra = _run_discovery_refinement_agent(
        checkpoint,
        agent_outputs.get("convergent_evidence_agent", ConvergentEvidenceOutput(agent_name=CONVERGENT_EVIDENCE_AGENT, run_id=run_id)),
        agent_outputs.get("clinical_trialist", ClinicalTrialistOutput(agent_name=CLINICAL_TRIALIST, run_id=run_id)),
        run_id,
    )
    rt = _run_red_team(checkpoint, dra, run_id)
    rev = _run_reviewer(checkpoint, dra, rt, run_id)
    return dra, rt, rev


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_agentic(
    disease_name: str,
    run_id: str | None = None,
    checkpoint: dict | None = None,
    auto_continue: bool = False,
    data_dir: str = "data",
) -> dict:
    """
    Run the full agentic extension on top of a completed deterministic pipeline run.

    Args:
        disease_name:   e.g. "coronary artery disease"
        run_id:         unique run identifier (generated if not provided)
        checkpoint:     pipeline output dict; loaded from data/ if not provided
        auto_continue:  skip interactive pauses (for testing / CI)
        data_dir:       where analyze_{slug}.json lives

    Returns:
        dict with virgin_targets, journal_dir, agent_outputs, token_summary
    """
    if run_id is None:
        run_id = uuid.uuid4().hex[:8]

    if checkpoint is None:
        checkpoint = load_checkpoint(disease_name, data_dir)

    disease_key = disease_name.upper().replace(" ", "_")[:8]
    journal = RunJournal(disease_key=disease_key, run_id=run_id)

    agent_outputs: dict[str, dict] = {}

    # ------------------------------------------------------------------ PAUSE 1
    print(f"\n[agentic] Starting run {run_id} for '{disease_name}'")
    briefing = run_briefing(checkpoint, run_id)
    _record_token_usage(journal, CHIEF_OF_STAFF, briefing.token_usage)
    for mc in briefing.methods_choices:
        journal.log_methods_choice(mc)

    user_response = _pause(journal, 1, briefing.content, auto_continue)
    if user_response.upper() == "ABORT":
        print("[agentic] Run aborted by user at PAUSE 1.")
        journal.close()
        return {"status": "aborted", "pause": 1, "journal_dir": str(journal.run_dir)}

    if briefing.run_paused and not auto_continue:
        print(f"[agentic] CSO recommends pause: {briefing.pause_reason}")

    # --------------------------------------------------------- Phase 3 agents
    print("\n[agentic] Phase 3 — Statistical Geneticist + Convergent Evidence Agent")
    sg = _run_statistical_geneticist(checkpoint, run_id)
    _record_token_usage(journal, STATISTICAL_GENETICIST, sg.token_usage)
    for gap in sg.library_gaps:
        journal.log_library_gap(gap)

    ce = _run_convergent_evidence_agent(checkpoint, sg, run_id)
    _record_token_usage(journal, CONVERGENT_EVIDENCE_AGENT, ce.token_usage)
    for rec in ce.augmentation_recommendations:
        journal.log_augmentation_recommendation(rec)

    agent_outputs["statistical_geneticist"] = sg.model_dump()
    agent_outputs["convergent_evidence_agent"] = ce.model_dump()

    # ------------------------------------------------------------------ PAUSE 2
    conflict = run_conflict_analysis(checkpoint, agent_outputs, run_id)
    _record_token_usage(journal, CHIEF_OF_STAFF, conflict.token_usage)

    user_response = _pause(journal, 2, conflict.content, auto_continue)
    if user_response.upper() == "ABORT":
        print("[agentic] Run aborted by user at PAUSE 2.")
        journal.close()
        return {"status": "aborted", "pause": 2, "journal_dir": str(journal.run_dir)}

    # --------------------------------------------------------- Phase 4 agents
    print("\n[agentic] Phase 4 — Chemistry Agent + Clinical Trialist")
    chem = _run_chemistry_agent(checkpoint, run_id)
    _record_token_usage(journal, CHEMISTRY_AGENT, chem.token_usage)

    # Pass confirmed genes from CE agent so trialist can run the active-trial gate
    trialist = _run_clinical_trialist(checkpoint, ce.confirmed_genes, run_id)
    _record_token_usage(journal, CLINICAL_TRIALIST, trialist.token_usage)

    agent_outputs["chemistry_agent"] = chem.model_dump()
    agent_outputs["clinical_trialist"] = trialist.model_dump()

    # --------------------------------------------------------- Phase 5 agents
    print("\n[agentic] Phase 5 — DRA + Red Team + Reviewer")
    dra = _run_discovery_refinement_agent(checkpoint, ce, trialist, run_id)
    _record_token_usage(journal, DISCOVERY_REFINEMENT_AGENT, dra.token_usage)

    rt = _run_red_team(checkpoint, dra, run_id)
    _record_token_usage(journal, RED_TEAM, rt.token_usage)

    reviewer = _run_reviewer(checkpoint, dra, rt, run_id)
    _record_token_usage(journal, SCIENTIFIC_REVIEWER, reviewer.token_usage)

    # Re-delegation loop
    round_counts: dict[str, int] = {}
    if reviewer.verdict == "REVISE" and reviewer.redelegation_instructions:
        dra, rt, reviewer = _handle_redelegation(
            journal, reviewer, checkpoint, agent_outputs, run_id, round_counts
        )

    for vt in dra.virgin_targets:
        journal.log_virgin_target(vt)
    journal.log_reviewer_verdict(reviewer.verdict, reviewer.redelegation_instructions)

    agent_outputs["discovery_refinement_agent"] = dra.model_dump()
    agent_outputs["red_team"] = rt.model_dump()
    agent_outputs["scientific_reviewer"] = reviewer.model_dump()

    # Add finngen AUC to checkpoint if available (for exec summary)
    if "finngen_auc" in checkpoint:
        pass  # already in checkpoint

    # ------------------------------------------------------------------ PAUSE 3
    exec_summary = run_exec_summary(checkpoint, agent_outputs, run_id)
    _record_token_usage(journal, CHIEF_OF_STAFF, exec_summary.token_usage)

    user_response = _pause(journal, 3, exec_summary.content, auto_continue)

    # --------------------------------------------------------- Finalise
    run_dir = journal.close()
    print(f"\n[agentic] Run complete. Journal: {run_dir}")

    return {
        "status": "complete",
        "run_id": run_id,
        "journal_dir": str(run_dir),
        "virgin_targets": [vt.model_dump() for vt in dra.virgin_targets],
        "agent_outputs": agent_outputs,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m orchestrator.agentic.agentic_orchestrator <disease_name> [--auto]")
        sys.exit(1)

    _disease = sys.argv[1]
    _auto = "--auto" in sys.argv
    run_agentic(_disease, auto_continue=_auto)
