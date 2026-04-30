"""
discovery_refinement_agent.py — Applies the 2-gate virgin target definition.

Receives CE Agent + Clinical Trialist outputs. Applies gates:
  Gate 1: convergent evidence confirmed by CE Agent
  Gate 2: no active development confirmed by Clinical Trialist

For each passing gene: checks literature for "novel" vs "actually known",
assesses biological plausibility, assigns priority tier.
"""
from __future__ import annotations

from orchestrator.agentic.agent_config import DISCOVERY_REFINEMENT_AGENT
from orchestrator.agentic.agent_contracts import (
    ConvergentEvidenceOutput,
    ClinicalTrialistOutput,
    DiscoveryRefinementOutput,
    EvidenceItem,
    PathReasoning,
    VirginTarget,
)
from orchestrator.agentic.agent_runtime import run_agent_with_tools
from orchestrator.agentic.agents._base import (
    TOOL_GET_OT_TARGET_INFO,
    TOOL_SEARCH_PUBMED,
    _get_tool_functions,
    build_system_prompt,
    parse_output_block,
)

_PERSONA_CAD = "You know this disease's biology deeply. You know what a real cardiovascular target looks like and what a false positive smells like."
_PERSONA_RA = "You know this disease's biology deeply. You know what a real autoimmune target looks like and what a false positive smells like."

_AUTHORITY = """\
- Nominate a virgin target directly — this is the primary output of the agentic run
- Downgrade a gene from "novel" to "known" if PubMed returns a disease-specific mechanistic paper
- Assign priority tier HIGH/MEDIUM/LOW to each virgin target based on convergent evidence strength\
"""

_OUTPUT_SCHEMA = """\
<output>
{
  "virgin_targets": [
    {
      "gene": "GENE1",
      "priority": "HIGH",
      "gate1_confirmed": true,
      "gate2_confirmed": true,
      "notes": "strong Tier A + Tier B evidence; no trial history"
    }
  ],
  "downgraded_to_known": ["GENE2"],
  "reasoning": "summary of key decisions"
}
</output>\
"""

_TOOLS = [
    TOOL_SEARCH_PUBMED,
    TOOL_GET_OT_TARGET_INFO,
]


def _get_persona(disease_name: str) -> str:
    dl = disease_name.lower()
    if "coronary" in dl or "cad" in dl or "cardiovascular" in dl:
        return _PERSONA_CAD
    if "rheumatoid" in dl or " ra " in dl:
        return _PERSONA_RA
    return "You know this disease's biology deeply. You know what a real target looks like and what a false positive smells like."


def _fmt_context(
    checkpoint: dict,
    ce_output: ConvergentEvidenceOutput,
    trialist_output: ClinicalTrialistOutput,
) -> str:
    disease = checkpoint.get("disease_name", "unknown")

    confirmed = set(ce_output.confirmed_genes)
    no_dev = set(trialist_output.no_active_development)
    active_dev = set(trialist_output.active_development_found)

    gate1_pass = list(confirmed)
    gate1_and_2 = [g for g in gate1_pass if g in no_dev]
    disqualified = [g for g in gate1_pass if g in active_dev]

    evidence_by_gene: dict[str, list[EvidenceItem]] = {}
    for e in ce_output.evidence_items:
        evidence_by_gene.setdefault(e.gene, []).append(e)

    candidate_lines = []
    for gene in gate1_and_2:
        items = evidence_by_gene.get(gene, [])
        ev_summary = "; ".join(
            f"{e.modality_name}(Tier{e.tier},{e.confidence})" for e in items
        )
        candidate_lines.append(f"  {gene}: {ev_summary or 'evidence confirmed by CE agent'}")

    return f"""\
Disease: {disease}

## Genes passing Gate 1 (convergent evidence confirmed by CE Agent)
{chr(10).join(f"  - {g}" for g in gate1_pass) or "  (none)"}

## Genes passing Gate 1 AND Gate 2 (no active development confirmed)
{chr(10).join(candidate_lines) or "  (none)"}

## Disqualified by active development
{chr(10).join(f"  - {g}" for g in disqualified) or "  (none)"}

For each gene passing both gates:
1. Search PubMed for disease-specific mechanistic papers. If a mechanistic paper exists,
   downgrade from "novel" to "known".
2. Assess biological plausibility in the context of {disease} biology.
3. Assign priority: HIGH = strong multi-tier evidence + biologically plausible;
   MEDIUM = convergent evidence but mechanism less clear; LOW = passes gates but weak.
"""


def run(
    checkpoint: dict,
    ce_output: ConvergentEvidenceOutput,
    trialist_output: ClinicalTrialistOutput,
    run_id: str,
) -> DiscoveryRefinementOutput:
    disease = checkpoint.get("disease_name", "unknown")
    persona = _get_persona(disease)
    system = build_system_prompt(persona, _AUTHORITY, _OUTPUT_SCHEMA)
    user_msg = _fmt_context(checkpoint, ce_output, trialist_output)
    tool_fns = _get_tool_functions([t["name"] for t in _TOOLS])

    text, usage = run_agent_with_tools(
        DISCOVERY_REFINEMENT_AGENT, system, user_msg, _TOOLS, tool_fns
    )

    parsed = parse_output_block(text)

    evidence_by_gene: dict[str, list[EvidenceItem]] = {}
    for e in ce_output.evidence_items:
        evidence_by_gene.setdefault(e.gene, []).append(e)

    virgin_targets = []
    for vt_raw in parsed.get("virgin_targets", []):
        if not isinstance(vt_raw, dict) or not vt_raw.get("gene"):
            continue
        gene = vt_raw["gene"]
        priority = vt_raw.get("priority", "LOW")
        if priority not in ("HIGH", "MEDIUM", "LOW"):
            priority = "LOW"
        virgin_targets.append(VirginTarget(
            gene=gene,
            evidence_items=evidence_by_gene.get(gene, []),
            priority=priority,  # type: ignore[arg-type]
            gate1_confirmed=vt_raw.get("gate1_confirmed", False),
            gate2_confirmed=vt_raw.get("gate2_confirmed", False),
            notes=vt_raw.get("notes", ""),
        ))

    return DiscoveryRefinementOutput(
        agent_name=DISCOVERY_REFINEMENT_AGENT,
        run_id=run_id,
        token_usage=usage,
        virgin_targets=virgin_targets,
        downgraded_to_known=parsed.get("downgraded_to_known", []),
        path_reasoning=[PathReasoning(
            agent=DISCOVERY_REFINEMENT_AGENT,
            statement=parsed.get("reasoning", text[:300]),
        )],
    )
