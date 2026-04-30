"""
chief_of_staff_agent.py — CSO in three modes: briefing, conflict_analysis, exec_summary.

Each mode receives relevant pipeline checkpoint slices, reasons about them using the
Kathiresan/CSO persona, and returns a CSOOutput with a machine-readable decision block.
"""
from __future__ import annotations

import json
import re

from orchestrator.agentic.agent_config import CHIEF_OF_STAFF
from orchestrator.agentic.agent_contracts import CSOOutput, MethodsChoice, TokenUsage
from orchestrator.agentic.agent_runtime import run_agent


# ---------------------------------------------------------------------------
# Shared prompt assembly
# ---------------------------------------------------------------------------

_SHARED_PREAMBLE = """\
## How you work
You receive deterministic pipeline outputs as your starting point. Your job is not to
re-execute what the pipeline already computed. Your job is to find what it missed,
challenge what it got wrong, and surface what it cannot see.

Before drawing any conclusion, state in one sentence:
  "The most likely failure mode here is [X] because [Y]."
Then pursue that line of inquiry. Stop when you have sufficient grounds to take a position.

## Evidence confidence levels
- HIGH: ≥2 independent sources (different modalities).
- MEDIUM: 1 strong source. Note what would upgrade to HIGH.
- LOW: inference or extrapolation. Flag explicitly.

## Escalation criterion
Genuine scientific ambiguity that changes the conclusion of the entire run — escalate
to the PI with a one-sentence question. Do not escalate uncertainty you can resolve yourself.\
"""

_CSO_PERSONA = """\
## Who you are
You are Chief of Staff. You have made enough high-stakes resource allocation decisions
to know that weak evidence dressed up as strong is how programs die. You ask hard
questions before anything moves forward.\
"""

_KATHIRESAN_LENS = """\
## Briefing lens
You think as Sekar Kathiresan. Human genetics is the starting point. You are not
convinced by biology that lacks a genetic instrument anchoring it to the trait.\
"""

_CSO_AUTHORITY = """\
## Decision authority
- In briefing: pause the run if projected anchor coverage is < 60%, or if the data
  completeness picture suggests the run will produce low-yield output.
- In conflict_analysis: downgrade a candidate's confidence unilaterally if you identify
  a data quality concern. State the specific concern.
- In exec_summary: declare "not ready" and trigger re-delegation if the virgin target
  evidence does not hold up under scrutiny.\
"""

_DECISION_FORMAT = """\
End your response with a machine-readable decision block in this exact format:
<decision>
{
  "pause_run": false,
  "pause_reason": "",
  "downgraded_candidates": []
}
</decision>\
"""


def _build_system_prompt(extra_lens: str = "") -> str:
    parts = [_CSO_PERSONA]
    if extra_lens:
        parts.append(extra_lens)
    parts.extend([_SHARED_PREAMBLE, _CSO_AUTHORITY, _DECISION_FORMAT])
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Checkpoint formatters — extract relevant slices per mode
# ---------------------------------------------------------------------------

def _top_targets(checkpoint: dict, n: int = 15) -> list[dict]:
    targets = checkpoint.get("target_list") or []
    return sorted(targets, key=lambda t: abs(t.get("ota_gamma") or 0.0), reverse=True)[:n]


def _anchor_genes(checkpoint: dict, n: int = 20) -> list[dict]:
    targets = checkpoint.get("target_list") or []
    return [t for t in targets if t.get("l2g_score", 0) >= 0.5][:n]


def _fmt_target_row(t: dict) -> str:
    gene = t.get("target_gene") or t.get("gene_symbol") or "?"
    gamma = t.get("ota_gamma")
    gamma_str = f"{gamma:+.3f}" if gamma is not None else "N/A"
    tier = t.get("dominant_tier") or t.get("tier") or "?"
    phase = t.get("max_phase") or 0
    return f"  {gene:<12} γ={gamma_str}  tier={tier}  max_phase={phase}"


def _fmt_briefing_context(checkpoint: dict) -> str:
    disease = checkpoint.get("disease_name", "unknown")
    efo = checkpoint.get("efo_id", "")
    anchors = _anchor_genes(checkpoint)
    top = _top_targets(checkpoint)
    completeness = checkpoint.get("data_completeness", {})
    warnings = checkpoint.get("pipeline_warnings", [])
    gps_n = len(checkpoint.get("gps_disease_state_reversers") or [])

    anchor_lines = "\n".join(_fmt_target_row(a) for a in anchors[:15]) or "  (none)"
    top_lines = "\n".join(_fmt_target_row(t) for t in top) or "  (none)"
    warn_lines = "\n".join(f"  - {w}" for w in warnings[:10]) or "  (none)"

    return f"""\
Disease: {disease} ({efo})

## GWAS anchors (L2G ≥ 0.5)
{anchor_lines}

## Top 15 candidates by |OTA γ|
{top_lines}

## Data completeness
  Perturb-seq dataset: {completeness.get('perturb_seq_dataset', 'unknown')}
  h5ad disease signature loaded: {completeness.get('h5ad_disease_sig_loaded')}
  GPS screen run: {completeness.get('gps_screen_run')}  ({gps_n} disease reversers)

## Pipeline warnings
{warn_lines}"""


def _fmt_conflict_context(checkpoint: dict, agent_outputs: dict) -> str:
    top = _top_targets(checkpoint, n=20)
    top_lines = "\n".join(_fmt_target_row(t) for t in top)

    sg_out = agent_outputs.get("statistical_geneticist", {})
    ce_out = agent_outputs.get("convergent_evidence_agent", {})

    sg_library_gaps = sg_out.get("library_gaps", [])
    sg_suspect = sg_out.get("anchor_set_suspect", False)
    sg_hypothesis = sg_out.get("anchor_set_suspect_hypothesis", "")
    ce_confirmed = ce_out.get("confirmed_genes", [])

    gap_lines = "\n".join(f"  {g['gene']}: {g['known_causal_reason']}" for g in sg_library_gaps[:10]) or "  (none identified)"

    return f"""\
## Top 20 candidates by |OTA γ|
{top_lines}

## Statistical Geneticist findings
  Anchor set suspect: {sg_suspect}
  Hypothesis: {sg_hypothesis or '(none)'}
  Library gaps (known causal genes absent from Perturb-seq):
{gap_lines}

## Convergent Evidence Agent findings
  Genes with confirmed convergent evidence: {ce_confirmed or '(none — agents not yet run)'}"""


def _fmt_exec_context(checkpoint: dict, agent_outputs: dict) -> str:
    dra_out = agent_outputs.get("discovery_refinement_agent", {})
    virgin_targets = dra_out.get("virgin_targets", [])

    rt_out = agent_outputs.get("red_team", {})
    hard_rejects = rt_out.get("hard_rejects", [])

    rev_out = agent_outputs.get("scientific_reviewer", {})
    verdict = rev_out.get("verdict", "(not yet run)")

    top = _top_targets(checkpoint, n=10)
    top_lines = "\n".join(_fmt_target_row(t) for t in top)

    vt_lines = "\n".join(
        f"  {v['gene']} ({v.get('priority','?')} priority) — "
        f"gate1={v.get('gate1_confirmed')} gate2={v.get('gate2_confirmed')}"
        for v in virgin_targets
    ) or "  (none nominated)"

    finngen_auc = checkpoint.get("finngen_auc")
    auc_str = f"{finngen_auc:.3f}" if finngen_auc is not None else "not computed this run"

    return f"""\
## Virgin targets nominated
{vt_lines}

## Red Team hard rejects: {hard_rejects or '(none)'}
## Reviewer verdict: {verdict}

## Top 10 candidates (for context)
{top_lines}

## FinnGen holdout AUC: {auc_str}
Note: anchor recovery on known targets (PCSK9, IL6R, etc.) measures pipeline completeness,
not prospective validity of novel nominations. The prospective test is the next experiment."""


# ---------------------------------------------------------------------------
# Decision parsing
# ---------------------------------------------------------------------------

def _parse_decision(text: str) -> dict:
    match = re.search(r"<decision>(.*?)</decision>", text, re.DOTALL)
    if not match:
        return {"pause_run": False, "pause_reason": "", "downgraded_candidates": []}
    try:
        return json.loads(match.group(1).strip())
    except json.JSONDecodeError:
        return {"pause_run": False, "pause_reason": "", "downgraded_candidates": []}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_briefing(
    checkpoint: dict,
    run_id: str,
) -> CSOOutput:
    system = _build_system_prompt(extra_lens=_KATHIRESAN_LENS)
    user_msg = (
        "You are running the pre-run briefing for this agentic pipeline run.\n\n"
        "Review the pipeline outputs below. Identify the most likely failure modes for "
        "this disease run, assess anchor coverage, orient the downstream agents, and "
        "decide whether to pause before proceeding.\n\n"
        + _fmt_briefing_context(checkpoint)
    )

    text, usage = run_agent(CHIEF_OF_STAFF, system, user_msg)
    decision = _parse_decision(text)

    return CSOOutput(
        agent_name=CHIEF_OF_STAFF,
        run_id=run_id,
        mode="briefing",
        content=text,
        token_usage=usage,
        run_paused=decision.get("pause_run", False),
        pause_reason=decision.get("pause_reason", ""),
        downgraded_candidates=decision.get("downgraded_candidates", []),
        methods_choices=[
            MethodsChoice(
                agent=CHIEF_OF_STAFF,
                choice="briefing_complete",
                rationale=f"pause_run={decision.get('pause_run')}",
            )
        ],
    )


def run_conflict_analysis(
    checkpoint: dict,
    agent_outputs: dict,
    run_id: str,
) -> CSOOutput:
    system = _build_system_prompt()
    user_msg = (
        "You are running post-Tier-3 conflict analysis.\n\n"
        "Explain the divergence between GWAS genetic evidence and Perturb-seq functional "
        "evidence for the top candidates. Take a position on which candidates matter and "
        "why. For each top program driving nominations, assess whether it has a genetic "
        "instrument — if not, state this prominently.\n\n"
        + _fmt_conflict_context(checkpoint, agent_outputs)
    )

    text, usage = run_agent(CHIEF_OF_STAFF, system, user_msg)
    decision = _parse_decision(text)

    return CSOOutput(
        agent_name=CHIEF_OF_STAFF,
        run_id=run_id,
        mode="conflict_analysis",
        content=text,
        token_usage=usage,
        run_paused=decision.get("pause_run", False),
        pause_reason=decision.get("pause_reason", ""),
        downgraded_candidates=decision.get("downgraded_candidates", []),
    )


def run_exec_summary(
    checkpoint: dict,
    agent_outputs: dict,
    run_id: str,
) -> CSOOutput:
    system = _build_system_prompt()
    user_msg = (
        "You are running the final executive summary for this agentic run.\n\n"
        "List virgin targets separately and prominently. Recommend one next experiment "
        "per top candidate. Report the FinnGen holdout AUC. State explicitly what anchor "
        "recovery does and does not prove about prospective discovery validity.\n\n"
        + _fmt_exec_context(checkpoint, agent_outputs)
    )

    text, usage = run_agent(CHIEF_OF_STAFF, system, user_msg)
    decision = _parse_decision(text)

    return CSOOutput(
        agent_name=CHIEF_OF_STAFF,
        run_id=run_id,
        mode="exec_summary",
        content=text,
        token_usage=usage,
        run_paused=decision.get("pause_run", False),
        pause_reason=decision.get("pause_reason", ""),
        downgraded_candidates=decision.get("downgraded_candidates", []),
    )
