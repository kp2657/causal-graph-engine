"""
red_team.py — Adversarial challenge per virgin target candidate.

No tool calls — reasons on evidence items already collected. Checks:
  - Evidence independence (tier hierarchy — GPS + Perturb-seq is not convergent)
  - Program→trait causal grounding (does the program have a genetic instrument?)
  - Structural causal / druggability / safety arguments
"""
from __future__ import annotations

from orchestrator.agentic.agent_config import RED_TEAM
from orchestrator.agentic.agent_contracts import (
    DiscoveryRefinementOutput,
    PathReasoning,
    RedTeamOutput,
)
from orchestrator.agentic.agent_runtime import run_agent
from orchestrator.agentic.agents._base import (
    build_system_prompt,
    parse_output_block,
)

_PERSONA = """\
You find the flaw in the argument — not to be difficult, but because finding it here
costs less than finding it in Phase III.\
"""

_AUTHORITY = """\
- HARD_REJECT if convergent evidence legs are not actually independent (same-tier evidence
  is replication, not convergence; GPS + Perturb-seq = both Tier C)
- HARD_REJECT if the program→trait link has no genetic grounding (no eQTL-GWAS colocalization,
  no LOF burden) — the β is real but the program may not cause the trait
- PASS without comment if no structural argument exists — do not generate weak arguments
- Type each counterargument: causal / druggability / safety\
"""

_OUTPUT_SCHEMA = """\
<output>
{
  "hard_rejects": ["GENE1"],
  "counterarguments": [
    {
      "gene": "GENE2",
      "type": "causal",
      "argument": "both evidence legs are Tier C — GPS reversal + Perturb-seq β from same cell system"
    }
  ]
}
</output>\
"""


def _fmt_context(dra_output: DiscoveryRefinementOutput) -> str:
    vt_lines = []
    for vt in dra_output.virgin_targets:
        ev_summary = "; ".join(
            f"{e.modality_name}(Tier{e.tier},{e.confidence})"
            for e in vt.evidence_items
        )
        vt_lines.append(
            f"  {vt.gene} [{vt.priority}]: {ev_summary or 'evidence details not available'}\n"
            f"    gate1={vt.gate1_confirmed} gate2={vt.gate2_confirmed}  notes={vt.notes}"
        )

    return f"""\
## Virgin target candidates from Discovery Refinement Agent
{chr(10).join(vt_lines) or "  (none)"}

For each candidate: look for the structural flaw. Is the convergent evidence actually from
different tiers? Does the program→trait link have a genetic instrument grounding it?
Are there safety or druggability signals that would kill this in the clinic?
Pass without comment if no structural argument exists.
"""


def run(
    dra_output: DiscoveryRefinementOutput,
    run_id: str,
) -> RedTeamOutput:
    system = build_system_prompt(_PERSONA, _AUTHORITY, _OUTPUT_SCHEMA)
    user_msg = _fmt_context(dra_output)

    text, usage = run_agent(RED_TEAM, system, user_msg)
    parsed = parse_output_block(text)

    return RedTeamOutput(
        agent_name=RED_TEAM,
        run_id=run_id,
        token_usage=usage,
        hard_rejects=parsed.get("hard_rejects", []),
        counterarguments=parsed.get("counterarguments", []),
        path_reasoning=[PathReasoning(
            agent=RED_TEAM,
            statement=text[:300],
        )],
    )
