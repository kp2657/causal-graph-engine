"""
pipelines/state_space/conditional_gamma.py

Compute ConditionalGamma (program → trait γ) as a weighted mix of:
  - gamma_GWAS:       anchored from ota_gamma_estimation.estimate_gamma()
  - gamma_transition: derived from the pseudotime transition matrix

Mixed formula (locked):
  gamma_conditional = α × gamma_GWAS + (1-α) × gamma_transition

  α is chosen by StateEvidenceTier:
    T1_TrajectoryDirect   → α = 0.35  (strong transition data → low GWAS weight)
    T2_TrajectoryInferred → α = 0.55
    T3_TrajectoryProxy    → α = 0.70  (weak transition → lean on GWAS prior)

gamma_transition derivation:
  Given a T matrix (n_states × n_states) from infer_state_transition_graph,
  gamma_transition for program P = dot product of:
    - program's state affinity vector (how strongly P is expressed in each state)
    - sum of [T_perturbed - T_baseline] expected under redirecting toward healthy states

  In practice (Phase B v1): uses the pathological state fraction × program affinity
  as a heuristic gamma_transition when full T_perturbed is not yet available.

Public API:
    estimate_conditional_gamma(program_id, trait, disease, ...)
    estimate_conditional_gammas_for_programs(programs, trait, disease, ...)
"""
from __future__ import annotations

import math
from typing import Any

from models.latent_mediator import ConditionalGamma

# α per StateEvidenceTier (locked design decision)
_TIER_ALPHA: dict[str, float] = {
    "Tier1_TrajectoryDirect":   0.35,
    "Tier2_TrajectoryInferred": 0.55,
    "Tier3_TrajectoryProxy":    0.70,
}

_DEFAULT_ALPHA = 0.70   # fallback when tier unknown


def _get_alpha(evidence_tier: str) -> float:
    return _TIER_ALPHA.get(evidence_tier, _DEFAULT_ALPHA)


def _estimate_gamma_transition_from_state_affinity(
    program_top_genes: list[str],
    transition_result: dict[str, Any],
    transition_gene_weights: dict[str, float],
) -> float | None:
    """
    Heuristic gamma_transition from program gene overlap with transition DE signal.

    gamma_transition ≈ mean P_loading × (pathological_basin_fraction weighted by
    transition direction).

    For Phase B v1: use the mean transition_de_signal of the top program genes
    as a proxy for how strongly the program drives the pathological state.
    Returns None if no data available.
    """
    if not program_top_genes or not transition_gene_weights:
        return None

    scores = [
        transition_gene_weights[g]
        for g in program_top_genes
        if g in transition_gene_weights
    ]
    if not scores:
        return None

    # Mean transition DE signal of top program genes → proxy gamma_transition
    # Sign is positive: high transition signal = program drives disease state
    return float(sum(scores) / len(scores))


def estimate_conditional_gamma(
    program_id: str,
    trait: str,
    disease: str,
    *,
    program_top_genes: list[str] | None = None,
    program_gene_set: set[str] | None = None,
    transition_result: dict[str, Any] | None = None,
    transition_gene_weights: dict[str, float] | None = None,
    evidence_tier: str = "Tier3_TrajectoryProxy",
    efo_id: str | None = None,
) -> ConditionalGamma:
    """
    Estimate γ for one (program, trait) pair.

    Args:
        program_id:              LatentProgram.program_id.
        trait:                   Trait/disease name (e.g. "IBD", "CAD").
        disease:                 Short disease key (same as trait for most uses).
        program_top_genes:       Top genes of the program (for transition DE heuristic).
        program_gene_set:        Full gene set (for OT L2G enrichment).
        transition_result:       Output of infer_state_transition_graph.
        transition_gene_weights: Output of compute_transition_gene_weights.
        evidence_tier:           StateEvidenceTier for α selection.
        efo_id:                  EFO ID for live OT GWAS lookup.

    Returns:
        ConditionalGamma with gamma_mixed computed.
    """
    from pipelines.ota_gamma_estimation import estimate_gamma

    # Derive the canonical program label for GWAS lookup.
    # program_id is formatted as "{disease}_{cell_type}_{raw_id}"; strip prefix for lookup.
    # Also try the raw program_id directly in PROVISIONAL_GAMMAS.
    program_label = program_id   # passed through; estimate_gamma handles unknown keys

    # gamma_GWAS — from ota_gamma_estimation
    gamma_result = estimate_gamma(
        program=program_label,
        trait=trait,
        program_gene_set=program_gene_set,
        efo_id=efo_id,
    )
    gamma_gwas_raw: float = ((gamma_result or {}).get("gamma", 0.0) or 0.0)
    if not math.isfinite(gamma_gwas_raw):
        gamma_gwas_raw = float("nan")

    gwas_tier = (gamma_result or {}).get("evidence_tier", "Tier3_Provisional")

    # gamma_transition — heuristic from transition DE signal
    gamma_trans_raw = _estimate_gamma_transition_from_state_affinity(
        program_top_genes=program_top_genes or [],
        transition_result=transition_result or {},
        transition_gene_weights=transition_gene_weights or {},
    )
    if gamma_trans_raw is None or not math.isfinite(gamma_trans_raw):
        gamma_trans_raw = float("nan")

    # Alpha selection
    alpha = _get_alpha(evidence_tier)

    # Mixed gamma — handle NaN gracefully
    gwas_ok  = math.isfinite(gamma_gwas_raw)
    trans_ok = math.isfinite(gamma_trans_raw)

    if gwas_ok and trans_ok:
        gamma_mixed = alpha * gamma_gwas_raw + (1 - alpha) * gamma_trans_raw
    elif gwas_ok:
        gamma_mixed = gamma_gwas_raw        # fallback: GWAS only
    elif trans_ok:
        gamma_mixed = gamma_trans_raw       # fallback: transition only
    else:
        gamma_mixed = float("nan")

    return ConditionalGamma(
        program_id=program_id,
        trait=trait,
        disease=disease,
        gamma_gwas=gamma_gwas_raw if gwas_ok else float("nan"),
        gamma_transition=gamma_trans_raw if trans_ok else float("nan"),
        alpha=alpha,
        gamma_mixed=gamma_mixed if math.isfinite(gamma_mixed) else 0.0,
        evidence_tier=evidence_tier,
        data_source=f"GWAS:{(gamma_result or {}).get('data_source','unknown')}|transition:de_heuristic",
    )


def estimate_conditional_gammas_for_programs(
    programs: list[dict],
    trait: str,
    disease: str,
    *,
    transition_result: dict[str, Any] | None = None,
    transition_gene_weights: dict[str, float] | None = None,
    evidence_tier: str = "Tier3_TrajectoryProxy",
    efo_id: str | None = None,
) -> dict[str, ConditionalGamma]:
    """
    Estimate ConditionalGamma for a list of program dicts.

    Args:
        programs:               List of dicts with keys: program_id, top_genes, gene_set.
                                (Compatible with cnmf_runner output + label_programs output.)
        trait:                  Trait/disease name.
        disease:                Short disease key.
        transition_result:      infer_state_transition_graph output.
        transition_gene_weights: compute_transition_gene_weights output.
        evidence_tier:          Applied uniformly; override per program not yet supported.
        efo_id:                 EFO ID for live GWAS lookup (optional).

    Returns:
        {program_id: ConditionalGamma}
    """
    results: dict[str, ConditionalGamma] = {}
    for prog in programs:
        # Accept both LatentProgram objects and plain dicts
        if isinstance(prog, dict):
            pid = prog.get("program_id", "")
            top_genes = prog.get("top_genes", [])
            gene_set = prog.get("gene_set") or (set(top_genes) if top_genes else None)
        else:
            pid = getattr(prog, "program_id", "")
            top_genes = getattr(prog, "top_genes", [])
            gene_set = set(top_genes) if top_genes else None

        results[pid] = estimate_conditional_gamma(
            program_id=pid,
            trait=trait,
            disease=disease,
            program_top_genes=top_genes,
            program_gene_set=gene_set,
            transition_result=transition_result,
            transition_gene_weights=transition_gene_weights,
            evidence_tier=evidence_tier,
            efo_id=efo_id,
        )
    return results
