"""
scientific_reviewer.py — Hard-reject checklist and tier-tagged re-delegation.

Nature Genetics methods editor persona. Enforces formal criteria:
  F < 10 → reject (weak genetic instrument)
  β coexpression-derived → reject
  Evidence legs not independent → reject
Issues specific re-delegation instructions, not generic ones.
"""
from __future__ import annotations

from orchestrator.agentic.agent_config import SCIENTIFIC_REVIEWER
from orchestrator.agentic.agent_contracts import (
    DiscoveryRefinementOutput,
    PathReasoning,
    RedelegationRecord,
    RedTeamOutput,
    ReviewerOutput,
)
from orchestrator.agentic.agent_runtime import run_agent
from orchestrator.agentic.agents._base import (
    build_system_prompt,
    parse_output_block,
)

_PERSONA = """\
Nature Genetics methods editor. Enforces standards. The checklist is not a suggestion —
F < 10 is a reject. Not looking to fail things: identifies the one or two specific issues
that, if fixed, would make this output publication-quality.\
"""

_AUTHORITY = """\
- APPROVE (no re-delegation needed)
- REVISE with tier-specific re-delegation instruction (tag: T1/T2/T4-chem/T4-trial/DRA)
- HARD_REJECT a virgin target if: evidence legs are not independent, β is coexpression-derived,
  or F < 10 on the genetic instrument
Re-delegation instructions must be specific (name the gene, name the specific issue), not generic.\
"""

_OUTPUT_SCHEMA = """\
<output>
{
  "verdict": "APPROVE",
  "hard_rejected_targets": [],
  "redelegation_instructions": [
    {
      "tier_tag": "T1",
      "instruction": "Re-investigate LOF burden for GENE; current instrument F=8",
      "target_agent": "statistical_geneticist"
    }
  ]
}
</output>\
"""

_TAG_TO_AGENT = {
    "T1": "statistical_geneticist",
    "T2": "convergent_evidence_agent",
    "T4-chem": "chemistry_agent",
    "T4-trial": "clinical_trialist",
    "DRA": "discovery_refinement_agent",
}


def _fmt_context(
    dra_output: DiscoveryRefinementOutput,
    rt_output: RedTeamOutput,
) -> str:
    vt_lines = []
    for vt in dra_output.virgin_targets:
        ev_summary = "; ".join(
            f"{e.modality_name}(Tier{e.tier},{e.confidence})"
            for e in vt.evidence_items
        )
        vt_lines.append(
            f"  {vt.gene} [{vt.priority}]: {ev_summary or 'see notes'}\n"
            f"    gate1={vt.gate1_confirmed} gate2={vt.gate2_confirmed}"
        )

    reject_lines = "\n".join(f"  HARD_REJECT: {g}" for g in rt_output.hard_rejects) or "  (none)"
    arg_lines = "\n".join(
        f"  {a.get('gene','?')} [{a.get('type','?')}]: {a.get('argument','')}"
        for a in rt_output.counterarguments
    ) or "  (none)"

    return f"""\
## Virgin target candidates
{chr(10).join(vt_lines) or "  (none)"}

## Red Team hard rejects
{reject_lines}

## Red Team counterarguments
{arg_lines}

Review against formal criteria:
- F-statistic ≥ 10 on genetic instrument (if instrument present)
- β not derived from coexpression
- Convergent evidence legs from genuinely different tiers
- Re-delegation instructions must name the gene and the specific issue to fix
"""


def run(
    dra_output: DiscoveryRefinementOutput,
    rt_output: RedTeamOutput,
    run_id: str,
) -> ReviewerOutput:
    system = build_system_prompt(_PERSONA, _AUTHORITY, _OUTPUT_SCHEMA)
    user_msg = _fmt_context(dra_output, rt_output)

    text, usage = run_agent(SCIENTIFIC_REVIEWER, system, user_msg)
    parsed = parse_output_block(text)

    verdict_raw = parsed.get("verdict", "APPROVE")
    if verdict_raw not in ("APPROVE", "REVISE", "HARD_REJECT"):
        verdict_raw = "APPROVE"

    redelegation = []
    for r in parsed.get("redelegation_instructions", []):
        if not isinstance(r, dict):
            continue
        tag = r.get("tier_tag", "")
        if tag not in _TAG_TO_AGENT:
            continue
        redelegation.append(RedelegationRecord(
            round_num=1,
            tier_tag=tag,  # type: ignore[arg-type]
            instruction=r.get("instruction", ""),
            target_agent=_TAG_TO_AGENT[tag],
        ))

    return ReviewerOutput(
        agent_name=SCIENTIFIC_REVIEWER,
        run_id=run_id,
        token_usage=usage,
        verdict=verdict_raw,  # type: ignore[arg-type]
        redelegation_instructions=redelegation,
        hard_rejected_targets=parsed.get("hard_rejected_targets", []),
        path_reasoning=[PathReasoning(
            agent=SCIENTIFIC_REVIEWER,
            statement=text[:300],
        )],
    )
