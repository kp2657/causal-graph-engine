"""
clinical_trialist.py — Clinical trial history and active development gate.

Two jobs:
1. Investigate terminated/active trials for top targets; classify why trials failed
2. Confirm no active ClinicalTrials.gov entry for virgin target gate candidates
"""
from __future__ import annotations

from orchestrator.agentic.agent_config import CLINICAL_TRIALIST
from orchestrator.agentic.agent_contracts import (
    ClinicalTrialistOutput,
    PathReasoning,
)
from orchestrator.agentic.agent_runtime import run_agent_with_tools
from orchestrator.agentic.agents._base import (
    TOOL_GET_TRIAL_DETAILS,
    TOOL_GET_TRIALS_FOR_TARGET,
    TOOL_SEARCH_CLINICAL_TRIALS,
    _get_tool_functions,
    build_system_prompt,
    parse_output_block,
)

_PERSONA = """\
You read a terminated trial the way a detective reads a crime scene.
Stopped-for-efficacy is not the same story as stopped-for-safety.\
"""

_AUTHORITY = """\
- Flag "development_risk_efficacy_failure" if trial stopped for lack of efficacy in same indication
- Identify "priority_repurposing" if Phase 3 drug in another indication hits the gene and mechanism is plausible
- Confirm "no_active_development" (required for virgin target 2nd gate) or "active_development_found" (disqualifies)
- Declare "insufficient_data" after two failed search strategies — do not fabricate context\
"""

_OUTPUT_SCHEMA = """\
<output>
{
  "efficacy_failures": ["GENE1"],
  "priority_repurposing": ["GENE2"],
  "no_active_development": ["GENE3", ...],
  "active_development_found": ["GENE4"],
  "insufficient_data": ["GENE5"],
  "trial_notes": {"GENE1": "NCT001 stopped phase 3 for futility 2019"}
}
</output>\
"""

_TOOLS = [
    TOOL_GET_TRIALS_FOR_TARGET,
    TOOL_GET_TRIAL_DETAILS,
    TOOL_SEARCH_CLINICAL_TRIALS,
]


def _fmt_context(checkpoint: dict, candidate_genes: list[str]) -> str:
    disease = checkpoint.get("disease_name", "unknown")
    targets = checkpoint.get("target_list") or []
    top_genes = [
        t.get("target_gene", "?")
        for t in sorted(targets, key=lambda t: abs(t.get("ota_gamma") or 0.0), reverse=True)[:20]
    ]

    return f"""\
Disease: {disease}

## Top pipeline candidates (check trial history)
{chr(10).join(f"  - {g}" for g in top_genes)}

## Genes requiring virgin target gate confirmation (active trial check required)
{chr(10).join(f"  - {g}" for g in candidate_genes) or "  (will be determined after CE agent run)"}

For each top candidate:
1. Search for terminated trials in this indication — classify why they stopped
2. Search for active trials (status=RECRUITING or ACTIVE_NOT_RECRUITING) in this indication

For virgin target gate candidates specifically: confirm no active entry exists.
A gene with an active trial is disqualified from virgin target status.
"""


def run(
    checkpoint: dict,
    candidate_genes: list[str],
    run_id: str,
) -> ClinicalTrialistOutput:
    system = build_system_prompt(_PERSONA, _AUTHORITY, _OUTPUT_SCHEMA)
    user_msg = _fmt_context(checkpoint, candidate_genes)
    tool_fns = _get_tool_functions([t["name"] for t in _TOOLS])

    text, usage = run_agent_with_tools(
        CLINICAL_TRIALIST, system, user_msg, _TOOLS, tool_fns
    )

    parsed = parse_output_block(text)

    return ClinicalTrialistOutput(
        agent_name=CLINICAL_TRIALIST,
        run_id=run_id,
        token_usage=usage,
        efficacy_failures=parsed.get("efficacy_failures", []),
        priority_repurposing=parsed.get("priority_repurposing", []),
        no_active_development=parsed.get("no_active_development", []),
        active_development_found=parsed.get("active_development_found", []),
        insufficient_data=parsed.get("insufficient_data", []),
        path_reasoning=[PathReasoning(
            agent=CLINICAL_TRIALIST,
            statement=text[:300],
        )],
    )
