"""
convergent_evidence_agent.py — Finds independent second-leg evidence for single-leg genes.

Starts from the deterministic pipeline's target list. For genes with strong Perturb-seq β
OR eQTL (not both), finds evidence from a different experimental tier and integrates inline.

Also attempts evidence recovery for library_gap genes flagged by the Statistical Geneticist.
"""
from __future__ import annotations

from orchestrator.agentic.agent_config import CONVERGENT_EVIDENCE_AGENT
from orchestrator.agentic.agent_contracts import (
    AugmentationRecommendation,
    ConvergentEvidenceOutput,
    EvidenceItem,
    PathReasoning,
    StatisticalGeneticistOutput,
)
from orchestrator.agentic.agent_runtime import run_agent_with_tools
from orchestrator.agentic.agents._base import (
    TOOL_GET_GENE_BURDEN_STATS,
    TOOL_GET_OT_TARGET_INFO,
    TOOL_SEARCH_EUROPE_PMC,
    TOOL_SEARCH_PUBMED,
    _get_tool_functions,
    build_system_prompt,
    parse_output_block,
)

_PERSONA = """\
You do not repeat what has already been done. You find the angle no one tried yet.\
"""

_AUTHORITY = """\
- Classify a gene as "convergent_confirmed" (feeds virgin target gate) or "no_evidence_found"
- Write GEO/Zenodo accessions to augmentation_recommendations for any dataset you find
- Declare search exhausted after 2 failed attempts of different modality types
- Do not count GPS reversal + Perturb-seq as convergent — both are Tier C\
"""

_OUTPUT_SCHEMA = """\
<output>
{
  "confirmed_genes": ["GENE1", ...],
  "no_evidence_genes": ["GENE2", ...],
  "evidence_items": [
    {
      "gene": "GENE1",
      "modality_name": "LOF burden gnomAD",
      "tier": "A",
      "source": "gnomAD v4",
      "confidence": "MEDIUM",
      "description": "pLI=0.98, LOF o/e=0.12",
      "accession": null
    }
  ],
  "augmentation_recommendations": [
    {
      "gene": "GENE1",
      "accession": "GSE123456",
      "source_db": "GEO",
      "description": "CRISPRa screen in CD4+ T cells"
    }
  ]
}
</output>\
"""

_TOOLS = [
    TOOL_SEARCH_PUBMED,
    TOOL_SEARCH_EUROPE_PMC,
    TOOL_GET_GENE_BURDEN_STATS,
    TOOL_GET_OT_TARGET_INFO,
]


def _single_leg_genes(checkpoint: dict, top_n: int = 30) -> list[dict]:
    """Genes with Perturb-seq β OR eQTL evidence but not both."""
    targets = checkpoint.get("target_list") or []
    single_leg = []
    for t in targets:
        has_perturb = bool(t.get("perturb_beta") or t.get("perturb_novel"))
        has_eqtl = bool(t.get("eqtl_beta") or t.get("eqtl_direction"))
        gamma = abs(t.get("ota_gamma") or 0.0)
        if (has_perturb ^ has_eqtl) and gamma > 0:
            single_leg.append(t)
    return sorted(single_leg, key=lambda t: abs(t.get("ota_gamma") or 0.0), reverse=True)[:top_n]


def _fmt_context(checkpoint: dict, sg_output: StatisticalGeneticistOutput) -> str:
    disease = checkpoint.get("disease_name", "unknown")
    single_leg = _single_leg_genes(checkpoint)
    library_gaps = sg_output.library_gaps

    gene_lines = "\n".join(
        f"  {t.get('target_gene','?')}  γ={t.get('ota_gamma','?'):.3f}  "
        f"perturb={bool(t.get('perturb_beta'))}  eqtl={bool(t.get('eqtl_beta'))}"
        for t in single_leg[:20]
    ) or "  (none)"

    gap_lines = "\n".join(
        f"  {g.gene}: {g.known_causal_reason} — suggested: {g.suggested_modalities}"
        for g in library_gaps[:10]
    ) or "  (none)"

    return f"""\
Disease: {disease}

## Genes with single-leg evidence (need independent second leg)
{gene_lines}

## Library gap genes (known causal, absent from Perturb-seq — attempt recovery)
{gap_lines}

For each gene, reason about which 1-2 modalities are most likely to yield independent
evidence given that gene's biology. Search those first. Stop when you have HIGH confidence
(≥2 independent sources from different tiers). Remember: GPS reversal and Perturb-seq are
both Tier C — combining them is NOT convergent evidence.

Tier A: LOF burden (gnomAD/UKB), GWAS colocalization, Mendelian disease gene
Tier B: CRISPRa screen, bulk RNA-seq KO in different system, animal KO phenotype, patient iPSC
Tier C: GPS, co-essentiality, more Perturb-seq (DO NOT use as second leg)

Use search_pubmed / search_europe_pmc to find CRISPRa screens, animal model studies, or
patient data. Use get_gene_burden_stats for LOF burden evidence (Tier A).
"""


def run(
    checkpoint: dict,
    sg_output: StatisticalGeneticistOutput,
    run_id: str,
) -> ConvergentEvidenceOutput:
    system = build_system_prompt(_PERSONA, _AUTHORITY, _OUTPUT_SCHEMA)
    user_msg = _fmt_context(checkpoint, sg_output)
    tool_fns = _get_tool_functions([t["name"] for t in _TOOLS])

    text, usage = run_agent_with_tools(
        CONVERGENT_EVIDENCE_AGENT, system, user_msg, _TOOLS, tool_fns
    )

    parsed = parse_output_block(text)

    evidence_items = [
        EvidenceItem(
            gene=e["gene"],
            modality_name=e.get("modality_name", ""),
            tier=e.get("tier", "C"),  # type: ignore[arg-type]
            source=e.get("source", ""),
            confidence=e.get("confidence", "LOW"),  # type: ignore[arg-type]
            description=e.get("description", ""),
            accession=e.get("accession"),
        )
        for e in parsed.get("evidence_items", [])
        if isinstance(e, dict) and e.get("gene")
    ]

    augmentations = [
        AugmentationRecommendation(
            agent=CONVERGENT_EVIDENCE_AGENT,
            gene=r["gene"],
            accession=r.get("accession", ""),
            source_db=r.get("source_db", ""),
            description=r.get("description", ""),
        )
        for r in parsed.get("augmentation_recommendations", [])
        if isinstance(r, dict) and r.get("gene") and r.get("accession")
    ]

    return ConvergentEvidenceOutput(
        agent_name=CONVERGENT_EVIDENCE_AGENT,
        run_id=run_id,
        token_usage=usage,
        confirmed_genes=parsed.get("confirmed_genes", []),
        no_evidence_genes=parsed.get("no_evidence_genes", []),
        evidence_items=evidence_items,
        augmentation_recommendations=augmentations,
        path_reasoning=[PathReasoning(
            agent=CONVERGENT_EVIDENCE_AGENT,
            statement=text[:300],
        )],
    )
