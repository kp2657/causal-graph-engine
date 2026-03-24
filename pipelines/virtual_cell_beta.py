"""
virtual_cell_beta.py — 4-tier β fallback decision tree implementation.

Wraps ota_beta_estimation.py with a clean interface for agent use.
The full β fallback tree:
  Tier 1: Cell-type-matched Perturb-seq (direct measurement)
  Tier 2: eQTL-MR via GTEx (convergent, Mendelian randomization)
  Tier 3: LINCS L1000 perturbation signature (provisional, cell-line mismatched)
  Virtual: Geneformer/GEARS in silico OR pathway membership proxy
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipelines.ota_beta_estimation import estimate_beta, build_beta_matrix


def run_virtual_cell_beta_pipeline(
    genes: list[str],
    programs: list[str],
    perturbseq_data: dict | None = None,
    eqtl_data: dict | None = None,
    coloc_data: dict | None = None,
    lincs_data: dict | None = None,
    program_loadings: dict | None = None,
    program_gene_sets: dict | None = None,
    geneformer_data: dict | None = None,
    pathway_membership: dict | None = None,
    cell_type: str = "unknown",
    disease: str | None = None,
) -> dict:
    """
    Run the full β fallback pipeline for a gene × program matrix.

    Returns a summary with tier breakdown and the full β matrix.
    """
    result = build_beta_matrix(
        genes=genes,
        programs=programs,
        perturbseq_data=perturbseq_data,
        eqtl_data=eqtl_data,
        coloc_data=coloc_data,
        lincs_data=lincs_data,
        program_loadings=program_loadings,
        program_gene_sets=program_gene_sets,
        geneformer_data=geneformer_data,
        pathway_membership=pathway_membership,
        cell_type=cell_type,
        disease=disease,
    )

    # Count virtual entries (provisional_virtual)
    n_total = len(genes) * len(programs)
    n_virtual = sum(
        1 for g in genes for p in programs
        if result.beta_matrix.get(g, {}).get(p) is None
    )

    return {
        "genes":       genes,
        "programs":    programs,
        "beta_matrix": result.beta_matrix,
        "n_entries":   n_total,
        "n_virtual":   n_virtual,
        "pct_virtual": round(n_virtual / n_total * 100, 1) if n_total > 0 else 0,
        "note":        (result.virtual_ensemble_vs_baseline or {}).get("note", ""),
    }
