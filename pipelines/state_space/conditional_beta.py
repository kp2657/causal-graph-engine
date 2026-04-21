"""
pipelines/state_space/conditional_beta.py

Estimate ConditionalBeta (gene → LatentProgram β) per cell type.

Resolution order per (gene, program, cell_type):
  1. Tier1 — cell-type-matched Perturb-seq (best)
  2. Tier2 — eQTL-MR via GTEx (tissue-matched)
  3. Tier3 — LINCS L1000 genetic perturbation (cell-line mismatched)
  4. Pooled fallback — best available beta from any cell type; sets pooled_fallback=True

Design rule: if > 50% of betas for a gene come from pooled fallback,
a context_confidence_warning is added to TherapeuticRedirectionResult.

Public API:
    estimate_conditional_beta(gene, program_id, cell_type, disease, ...)
    estimate_conditional_betas_for_program(program, genes, cell_type, disease, ...)
"""
from __future__ import annotations

import math
from typing import Any

from models.latent_mediator import ConditionalBeta

# Alpha values per StateEvidenceTier (used downstream in conditional_gamma)
TIER_ALPHA: dict[str, float] = {
    "Tier1_TrajectoryDirect":   0.35,
    "Tier2_TrajectoryInferred": 0.55,
    "Tier3_TrajectoryProxy":    0.70,
}


def estimate_conditional_beta(
    gene: str,
    program_id: str,
    cell_type: str,
    disease: str,
    *,
    perturbseq_data: dict[str, Any] | None = None,
    eqtl_data: dict[str, Any] | None = None,
    coloc_h4: float | None = None,
    program_loading: float | None = None,
    lincs_signature: dict[str, Any] | None = None,
    program_gene_set: set[str] | None = None,
    pooled_perturbseq_data: dict[str, Any] | None = None,
) -> ConditionalBeta:
    """
    Estimate β for one (gene, program, cell_type) triplet.

    Tries each tier in order; on complete miss tries pooled data.
    Always returns a ConditionalBeta — NaN beta means unmeasured.

    Args:
        gene:                   Gene symbol.
        program_id:             LatentProgram.program_id.
        cell_type:              Target cell type.
        disease:                Short disease key.
        perturbseq_data:        Cell-type-matched Perturb-seq dict {gene → {programs → {beta,...}}}.
        eqtl_data:              GTEx eQTL result for this gene/tissue.
        coloc_h4:               COLOC H4 for eQTL ∩ GWAS.
        program_loading:        NMF loading for gene in this program (scales eQTL-MR beta).
        lincs_signature:        LINCS L1000 differential expression for gene KD:
                                {gene_symbol → log2fc or {"log2fc": float}}.
        program_gene_set:       Set of gene symbols defining the program (for LINCS overlap).
        pooled_perturbseq_data: Pooled (cross-cell-type) Perturb-seq; used as fallback.

    Returns:
        ConditionalBeta with beta=NaN if all sources unavailable.
    """
    from pipelines.ota_beta_estimation import (
        estimate_beta_tier1,
        estimate_beta_tier2,
        estimate_beta_tier3,
    )

    # Tier 1 — cell-type-matched Perturb-seq
    t1 = estimate_beta_tier1(gene, program_id, perturbseq_data, cell_type=cell_type)
    if t1 is not None and math.isfinite(t1.get("beta", float("nan"))):
        return ConditionalBeta(
            gene=gene,
            program_id=program_id,
            cell_type=cell_type,
            disease=disease,
            beta=float(t1["beta"]),
            beta_se=t1.get("beta_se"),
            pooled_fallback=False,
            context_verified=True,
            evidence_tier=t1.get("evidence_tier", "Tier1_TrajectoryDirect"),
            data_source=t1.get("data_source", "Perturb-seq"),
        )

    # Tier 2 — eQTL-MR
    t2 = estimate_beta_tier2(gene, program_id, eqtl_data, coloc_h4, program_loading)
    if t2 is not None and math.isfinite(t2.get("beta", float("nan"))):
        return ConditionalBeta(
            gene=gene,
            program_id=program_id,
            cell_type=cell_type,
            disease=disease,
            beta=float(t2["beta"]),
            beta_se=t2.get("beta_se"),
            pooled_fallback=False,
            context_verified=False,
            evidence_tier=t2.get("evidence_tier", "Tier2_TrajectoryInferred"),
            data_source=t2.get("data_source", "GTEx_eQTL_MR"),
        )

    # Tier 3 — LINCS L1000
    t3 = estimate_beta_tier3(gene, program_id, lincs_signature, program_gene_set)
    if t3 is not None and math.isfinite(t3.get("beta", float("nan"))):
        return ConditionalBeta(
            gene=gene,
            program_id=program_id,
            cell_type=cell_type,
            disease=disease,
            beta=float(t3["beta"]),
            beta_se=t3.get("beta_se"),
            pooled_fallback=False,
            context_verified=False,
            evidence_tier=t3.get("evidence_tier", "Tier3_TrajectoryProxy"),
            data_source=t3.get("data_source", "LINCS_L1000"),
        )

    # Pooled fallback — cross-cell-type data
    if pooled_perturbseq_data is not None:
        tp = estimate_beta_tier1(gene, program_id, pooled_perturbseq_data, cell_type="pooled")
        if tp is not None and math.isfinite(tp.get("beta", float("nan"))):
            return ConditionalBeta(
                gene=gene,
                program_id=program_id,
                cell_type=cell_type,
                disease=disease,
                beta=float(tp["beta"]),
                beta_se=tp.get("beta_se"),
                pooled_fallback=True,
                context_verified=False,
                evidence_tier="Tier3_TrajectoryProxy",
                data_source=tp.get("data_source", "Perturb-seq_pooled"),
            )

    # No data
    return ConditionalBeta(
        gene=gene,
        program_id=program_id,
        cell_type=cell_type,
        disease=disease,
        beta=float("nan"),
        beta_se=None,
        pooled_fallback=False,
        context_verified=False,
        evidence_tier="Tier3_TrajectoryProxy",
        data_source="no_data",
    )


def estimate_conditional_betas_for_program(
    program_id: str,
    genes: list[str],
    cell_type: str,
    disease: str,
    *,
    perturbseq_data: dict[str, Any] | None = None,
    eqtl_data_by_gene: dict[str, dict] | None = None,
    coloc_h4_by_gene: dict[str, float] | None = None,
    program_loadings: dict[str, float] | None = None,
    lincs_signatures_by_gene: dict[str, dict] | None = None,
    program_gene_set: set[str] | None = None,
    pooled_perturbseq_data: dict[str, Any] | None = None,
) -> list[ConditionalBeta]:
    """
    Estimate ConditionalBeta for all genes in a program.

    Args:
        program_id:          LatentProgram.program_id.
        genes:               List of gene symbols to estimate β for.
        cell_type:           Target cell type.
        disease:             Short disease key.
        perturbseq_data:     Cell-type-matched Perturb-seq {gene → {...}}.
        eqtl_data_by_gene:   {gene: GTEx eQTL result dict}.
        coloc_h4_by_gene:    {gene: COLOC H4 float}.
        program_loadings:    {gene: nmf_loading} — passed to eQTL-MR scaling.
        lincs_signatures_by_gene: {gene: LINCS L1000 signature dict}.
        program_gene_set:    Gene set for the program (for LINCS overlap scoring).
        pooled_perturbseq_data: Pooled fallback data.

    Returns:
        List[ConditionalBeta], one per gene. Genes with beta=NaN are included.
    """
    results: list[ConditionalBeta] = []
    for gene in genes:
        cb = estimate_conditional_beta(
            gene=gene,
            program_id=program_id,
            cell_type=cell_type,
            disease=disease,
            perturbseq_data=perturbseq_data,
            eqtl_data=(eqtl_data_by_gene or {}).get(gene),
            coloc_h4=(coloc_h4_by_gene or {}).get(gene),
            program_loading=(program_loadings or {}).get(gene),
            lincs_signature=(lincs_signatures_by_gene or {}).get(gene),
            program_gene_set=program_gene_set,
            pooled_perturbseq_data=pooled_perturbseq_data,
        )
        results.append(cb)
    return results


def compute_pooled_fraction(betas: list[ConditionalBeta]) -> float:
    """
    Fraction of betas (with finite values) that came from pooled fallback.
    Returns 0.0 if no finite betas.
    """
    finite = [b for b in betas if math.isfinite(b.beta)]
    if not finite:
        return 0.0
    pooled = sum(1 for b in finite if b.pooled_fallback)
    return pooled / len(finite)
