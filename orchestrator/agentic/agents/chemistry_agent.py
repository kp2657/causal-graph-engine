"""
chemistry_agent.py — Extends GPS results via ChEMBL. Classifies mechanism linkage.
Flags virgin targets as chemically tractable or needing a probe.
"""
from __future__ import annotations

from orchestrator.agentic.agent_config import CHEMISTRY_AGENT
from orchestrator.agentic.agent_contracts import (
    ChemistryAgentOutput,
    PathReasoning,
)
from orchestrator.agentic.agent_runtime import run_agent_with_tools
from orchestrator.agentic.agents._base import (
    TOOL_GET_CHEMBL_ACTIVITIES,
    TOOL_GET_OT_TARGET_INFO,
    TOOL_RUN_ADMET,
    TOOL_SEARCH_CHEMBL_COMPOUND,
    _get_tool_functions,
    build_system_prompt,
    parse_output_block,
)

_PERSONA = """\
You are excited by unexplored targets and skeptical of claims about undruggability.
A compound in a dish is not a drug program.\
"""

_AUTHORITY = """\
- Classify each GPS reverser as "mechanism_linked" or "mechanism_unknown"
- Reject a repurposing candidate if GPS Z-score is borderline AND ChEMBL shows low selectivity
  (kinase panel hit, PAINS structure)
- Flag a virgin target as "chemically_tractable" (PDB structure + ChEMBL activity < 1µM)
  or "needs_chemical_probe"\
"""

_OUTPUT_SCHEMA = """\
<output>
{
  "mechanism_linked": ["COMPOUND1"],
  "mechanism_unknown": ["COMPOUND2"],
  "rejected_repurposing": ["COMPOUND3"],
  "tractability": {
    "GENE1": "chemically_tractable",
    "GENE2": "needs_chemical_probe"
  },
  "notes": {"COMPOUND1": "direct ChEMBL activity on GENE1 target, IC50=45nM"}
}
</output>\
"""

_TOOLS = [
    TOOL_GET_CHEMBL_ACTIVITIES,
    TOOL_SEARCH_CHEMBL_COMPOUND,
    TOOL_RUN_ADMET,
    TOOL_GET_OT_TARGET_INFO,
]


def _fmt_context(checkpoint: dict) -> str:
    disease = checkpoint.get("disease_name", "unknown")
    reversers = checkpoint.get("gps_disease_state_reversers") or []
    targets = checkpoint.get("target_list") or []
    top_genes = [
        t.get("target_gene", "?")
        for t in sorted(targets, key=lambda t: abs(t.get("ota_gamma") or 0.0), reverse=True)[:15]
    ]

    reverser_lines = "\n".join(
        f"  {r.get('compound_name', r.get('compound','?'))}  "
        f"Z={r.get('z_score', r.get('Z_RGES','?'))}  "
        f"target={r.get('putative_target','unknown')}"
        for r in reversers[:20]
    ) or "  (none — GPS screen may not have run)"

    return f"""\
Disease: {disease}

## GPS disease-state reversers (baseline from pipeline)
{reverser_lines}

## Top pipeline candidates (assess chemical tractability)
{chr(10).join(f"  - {g}" for g in top_genes)}

For each GPS reverser:
1. Look up the compound in ChEMBL — is there direct target engagement data for the putative target?
2. If yes: mechanism_linked. If not: mechanism_unknown.
3. Reject borderline Z-score compounds with low ChEMBL selectivity.

For top novel candidates (max_phase=0 in pipeline output): assess chemical tractability.
"""


def run(checkpoint: dict, run_id: str) -> ChemistryAgentOutput:
    system = build_system_prompt(_PERSONA, _AUTHORITY, _OUTPUT_SCHEMA)
    user_msg = _fmt_context(checkpoint)
    tool_fns = _get_tool_functions([t["name"] for t in _TOOLS])

    text, usage = run_agent_with_tools(
        CHEMISTRY_AGENT, system, user_msg, _TOOLS, tool_fns
    )

    parsed = parse_output_block(text)

    tractability_raw = parsed.get("tractability", {})
    tractability = {
        k: v for k, v in tractability_raw.items()
        if v in ("chemically_tractable", "needs_chemical_probe")
    }

    return ChemistryAgentOutput(
        agent_name=CHEMISTRY_AGENT,
        run_id=run_id,
        token_usage=usage,
        mechanism_linked=parsed.get("mechanism_linked", []),
        mechanism_unknown=parsed.get("mechanism_unknown", []),
        rejected_repurposing=parsed.get("rejected_repurposing", []),
        tractability=tractability,  # type: ignore[arg-type]
        path_reasoning=[PathReasoning(
            agent=CHEMISTRY_AGENT,
            statement=text[:300],
        )],
    )
