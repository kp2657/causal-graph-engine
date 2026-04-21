"""
pipelines/state_space/program_loading.py

Compute P_loading for each (gene, program, cell_type) triplet.

  P_loading = 0.7 × nmf_loading + 0.3 × transition_de_signal

  nmf_loading:          row-normalised NMF H-matrix weight (gene's contribution to program).
                        Taken from cnmf_runner output (gene_loadings per program).
  transition_de_signal: |mean_disease - mean_healthy| per gene, clipped to [0, 1].
                        Taken from compute_transition_gene_weights() output.

Design decisions (locked):
  - Separate latent spaces per cell type → separate P_loading tables per cell type.
  - Only genes present in both nmf_loading AND transition_de_signal get a full P_loading.
  - Genes only in nmf_loading get: p_loading = 0.7 × nmf_loading (transition_de_signal=0).
  - Genes only in transition_de_signal are NOT included (no program membership).

Public API:
    compute_program_loading(nmf_result, transition_gene_weights, disease, cell_type)
    compute_program_loading_multi_celltype(nmf_results, tw_by_ct, disease)
"""
from __future__ import annotations

import math

from models.latent_mediator import ProgramLoading

# Weights (locked design decisions)
_W_NMF = 0.7
_W_DE  = 0.3


def _normalise_loadings(gene_loadings: dict[str, float]) -> dict[str, float]:
    """L1-normalise a gene_loadings dict (sum → 1), preserving zeros."""
    total = sum(abs(v) for v in gene_loadings.values())
    if total == 0:
        return gene_loadings.copy()
    return {g: v / total for g, v in gene_loadings.items()}


def compute_program_loading(
    nmf_result: dict,
    transition_gene_weights: dict[str, float],
    disease: str,
    cell_type: str,
) -> list[ProgramLoading]:
    """
    Compute P_loading for every (gene, program) pair found in nmf_result.

    Args:
        nmf_result:               Output of cnmf_runner.run_nmf_programs.
                                  Each program dict must contain "gene_loadings":
                                  {gene: raw_loading_float}.
        transition_gene_weights:  Output of compute_transition_gene_weights:
                                  {gene: de_signal ∈ [0, 1]}.
        disease:                  Short disease key.
        cell_type:                Cell type this NMF was built on.

    Returns:
        List of ProgramLoading objects sorted by (program_id, p_loading desc).
    """
    programs = nmf_result.get("programs", [])
    results: list[ProgramLoading] = []

    for i, prog in enumerate(programs):
        raw_id = prog.get("program_id", f"P{i:02d}")
        program_id = f"{disease}_{cell_type.replace(' ', '_')}_{raw_id}"
        gene_loadings_raw: dict[str, float] = prog.get("gene_loadings", {})

        if not gene_loadings_raw:
            continue

        gene_loadings = _normalise_loadings(gene_loadings_raw)

        for gene, nmf_load in gene_loadings.items():
            if not math.isfinite(nmf_load) or nmf_load <= 0:
                continue

            de_signal = transition_gene_weights.get(gene, 0.0)
            if not math.isfinite(de_signal):
                de_signal = 0.0

            p_load = _W_NMF * nmf_load + _W_DE * de_signal

            results.append(ProgramLoading(
                gene=gene,
                program_id=program_id,
                cell_type=cell_type,
                disease=disease,
                nmf_loading=float(nmf_load),
                transition_de_signal=float(de_signal),
                p_loading=float(p_load),
            ))

    results.sort(key=lambda x: (x.program_id, -x.p_loading))
    return results


def compute_program_loading_multi_celltype(
    nmf_results_by_ct: dict[str, dict],
    transition_weights_by_ct: dict[str, dict[str, float]],
    disease: str,
) -> dict[str, list[ProgramLoading]]:
    """
    Compute P_loading for each cell type independently.

    Args:
        nmf_results_by_ct:       {cell_type: nmf_result}
        transition_weights_by_ct: {cell_type: {gene: de_signal}}
        disease:                 Short disease key.

    Returns:
        {cell_type: list[ProgramLoading]}
    """
    result: dict[str, list[ProgramLoading]] = {}
    for cell_type, nmf_result in nmf_results_by_ct.items():
        tw = transition_weights_by_ct.get(cell_type, {})
        result[cell_type] = compute_program_loading(
            nmf_result=nmf_result,
            transition_gene_weights=tw,
            disease=disease,
            cell_type=cell_type,
        )
    return result
