"""
statistical_geneticist.py — Tier 1 agentic extension.

Starts from the deterministic pipeline's anchor gene list and:
1. Finds anchor gaps the L2G filter missed
2. Audits Perturb-seq library coverage (known causal genes absent from library)
3. Investigates LOF burden signals for promoted candidates
"""
from __future__ import annotations

import json

from orchestrator.agentic.agent_config import STATISTICAL_GENETICIST
from orchestrator.agentic.agent_contracts import (
    LibraryGap,
    MethodsChoice,
    PathReasoning,
    StatisticalGeneticistOutput,
)
from orchestrator.agentic.agent_runtime import run_agent_with_tools
from orchestrator.agentic.agents._base import (
    TOOL_GET_GENE_BURDEN_STATS,
    TOOL_GET_OT_DISEASE_TARGETS,
    TOOL_GET_OT_TARGET_INFO,
    TOOL_SEARCH_EUROPE_PMC,
    TOOL_SEARCH_PUBMED,
    _get_tool_functions,
    build_system_prompt,
    parse_output_block,
)

_PERSONA = """\
Pritchard + Veera Rajagopal. You sit at the intersection of population genetics and
cell biology and you are not satisfied until the evidence spans both. You know the
difference between a signal that is real and a signal that is well-measured.\
"""

_AUTHORITY = """\
- Reject an anchor gene if the eQTL tissue is biologically implausible for this disease
- Upgrade a state-nominated gene to anchor status only if LOF burden p < 0.05 AND
  GWAS-eQTL colocalization evidence exists. LOF alone is insufficient.
- Flag anchor set as suspect if > 20% of expected anchors absent; state a specific hypothesis
- Identify known causal genes absent from the Perturb-seq library (library_gaps)\
"""

_OUTPUT_SCHEMA = """\
<output>
{
  "reasoning": "one paragraph on the most important finding",
  "validated_anchors": ["GENE1", ...],
  "rejected_anchors": ["GENE2"],
  "upgraded_anchors": ["GENE3"],
  "library_gaps": [
    {"gene": "GENE4", "known_causal_reason": "...", "suggested_modalities": ["lof_burden", "animal_model"]}
  ],
  "anchor_set_suspect": false,
  "anchor_set_suspect_hypothesis": ""
}
</output>\
"""

_TOOLS = [
    TOOL_SEARCH_PUBMED,
    TOOL_SEARCH_EUROPE_PMC,
    TOOL_GET_GENE_BURDEN_STATS,
    TOOL_GET_OT_DISEASE_TARGETS,
    TOOL_GET_OT_TARGET_INFO,
]


def _fmt_context(checkpoint: dict) -> str:
    disease = checkpoint.get("disease_name", "unknown")
    efo = checkpoint.get("efo_id", "")
    targets = checkpoint.get("target_list") or []
    anchors = [t for t in targets if t.get("l2g_score", 0) >= 0.5]
    perturb = checkpoint.get("data_completeness", {}).get("perturb_seq_dataset", "unknown")
    warnings = checkpoint.get("pipeline_warnings", [])

    anchor_lines = "\n".join(
        f"  {t.get('target_gene','?')}  L2G={t.get('l2g_score','?')}  γ={t.get('ota_gamma','?')}"
        for t in anchors[:20]
    ) or "  (none)"

    warn_lines = "\n".join(f"  - {w}" for w in warnings[:5]) or "  (none)"

    return f"""\
Disease: {disease} ({efo})
Perturb-seq library: {perturb}
Total targets in pipeline output: {len(targets)}

## Anchor genes found by pipeline (L2G ≥ 0.5)
{anchor_lines}

## Pipeline warnings
{warn_lines}

Your tasks:
1. Identify known causal genes for {disease} that are absent from the anchor list above.
   Use PubMed/Europe PMC to find established disease genes, then check if they are absent.
2. For each absent known-causal gene, note whether it is in the Perturb-seq library (library_gap).
3. Investigate LOF burden for any genes the pipeline promoted without genetic instrument support.
4. Flag any anchor genes where the supporting eQTL tissue is biologically implausible.
"""


def run(checkpoint: dict, run_id: str) -> StatisticalGeneticistOutput:
    system = build_system_prompt(_PERSONA, _AUTHORITY, _OUTPUT_SCHEMA)
    user_msg = _fmt_context(checkpoint)
    tool_fns = _get_tool_functions([t["name"] for t in _TOOLS])

    text, usage = run_agent_with_tools(
        STATISTICAL_GENETICIST, system, user_msg, _TOOLS, tool_fns
    )

    parsed = parse_output_block(text)

    library_gaps = [
        LibraryGap(
            gene=g["gene"],
            known_causal_reason=g.get("known_causal_reason", ""),
            suggested_modalities=g.get("suggested_modalities", []),
        )
        for g in parsed.get("library_gaps", [])
        if isinstance(g, dict) and g.get("gene")
    ]

    return StatisticalGeneticistOutput(
        agent_name=STATISTICAL_GENETICIST,
        run_id=run_id,
        token_usage=usage,
        validated_anchors=parsed.get("validated_anchors", []),
        rejected_anchors=parsed.get("rejected_anchors", []),
        upgraded_anchors=parsed.get("upgraded_anchors", []),
        library_gaps=library_gaps,
        anchor_set_suspect=parsed.get("anchor_set_suspect", False),
        anchor_set_suspect_hypothesis=parsed.get("anchor_set_suspect_hypothesis", ""),
        path_reasoning=[PathReasoning(
            agent=STATISTICAL_GENETICIST,
            statement=parsed.get("reasoning", text[:200]),
        )],
    )
