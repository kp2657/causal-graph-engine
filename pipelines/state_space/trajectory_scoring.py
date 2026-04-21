"""
pipelines/state_space/trajectory_scoring.py

DEPRECATED — Phase C refactor (2026-03-25).

Trajectory scoring as an additive bonus is replaced by therapeutic_redirection
as the primary OTA score component.  Use:
  pipelines.state_space.therapeutic_redirection.compute_therapeutic_redirection()

This module is kept as a backward-compatibility stub so existing imports do not
crash.  All public functions still work but emit DeprecationWarning.
score_gene() and score_all_genes() delegate to the old implementation for now;
callers should migrate to therapeutic_redirection.py.
"""
from __future__ import annotations

from models.evidence import TrajectoryRedirectionScore, CellState
from pipelines.state_space.schemas import (
    PATHOLOGICAL_ENRICHMENT_THRESHOLD,
    HEALTHY_ENRICHMENT_THRESHOLD,
    ESCAPE_STABILITY_THRESHOLD,
    W_TRAJECTORY_REDIRECTION,
    W_DURABILITY,
    W_ESCAPE_PENALTY,
    W_NEGATIVE_MEMORY,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pathological_occupancy(
    states: list[CellState],
    basin_assignments: dict[str, str],
) -> float:
    """
    Weighted fraction of cells in pathological basins.
    Returns [0, 1].
    """
    total = sum(s.n_cells for s in states)
    if total == 0:
        return 0.0
    path_cells = sum(
        s.n_cells for s in states
        if basin_assignments.get(s.state_id) == "pathological"
    )
    return path_cells / total


def _expected_reduction_after_perturbation(
    gene: str,
    transition_result: dict,
    state_result: dict,
    resolution: str = "intermediate",
) -> float:
    """
    Estimate how much the perturbation of `gene` would reduce pathological occupancy.

    Uses a simple heuristic:
      - Identify which states have `gene` as a top marker gene
      - If those states are pathological, perturbation could reduce their occupancy
        by their proportional contribution to total pathological cells
      - If those states are healthy, perturbation could harm them (negative contribution)

    Returns a score in [0, 1]: 0 = no reduction expected, 1 = full pathological reduction.
    """
    import numpy as np

    states = state_result.get(resolution, [])
    basin_assignments = transition_result.get("basin_assignments", {})
    total_path = sum(
        s.n_cells for s in states
        if basin_assignments.get(s.state_id) == "pathological"
    )
    if total_path == 0:
        return 0.0

    reduction = 0.0
    for s in states:
        basin = basin_assignments.get(s.state_id, "mixed")
        if gene in s.marker_genes:
            if basin == "pathological":
                reduction += s.n_cells / total_path
            elif basin == "healthy":
                reduction -= 0.1  # penalise disrupting healthy state

    return float(np.clip(reduction, 0.0, 1.0))


def _durability_score(
    gene: str,
    transition_result: dict,
    state_result: dict,
    resolution: str = "intermediate",
) -> float:
    """
    Durability: do the states where `gene` is a marker connect to stable healthy basins?

    High durability: gene-marked states transition toward high-stability healthy basins.
    Low durability: transitions lead to escape or mixed basins.

    Returns [0, 1].
    """
    states = state_result.get(resolution, [])
    basin_assignments = transition_result.get("basin_assignments", {})
    transitions = transition_result.get("transitions", [])
    healthy_ids = set(transition_result.get("healthy_basin_ids", []))

    gene_states = {s.state_id for s in states if gene in s.marker_genes}
    if not gene_states:
        return 0.5  # no information → neutral

    durable_count = 0
    total_count = 0
    for t in transitions:
        if t.from_state in gene_states:
            total_count += 1
            if t.to_state in healthy_ids:
                # Check that target state is stable (not escape)
                target_basin = basin_assignments.get(t.to_state, "mixed")
                if target_basin == "healthy":
                    durable_count += 1

    if total_count == 0:
        return 0.5
    return durable_count / total_count


def _escape_risk(
    gene: str,
    transition_result: dict,
    state_result: dict,
    resolution: str = "intermediate",
) -> float:
    """
    Escape risk: probability of landing in escape basins after gene perturbation.

    Returns [0, 1]; higher = higher escape risk.
    """
    states = state_result.get(resolution, [])
    escape_ids = set(transition_result.get("escape_basin_ids", []))
    transitions = transition_result.get("transitions", [])

    gene_states = {s.state_id for s in states if gene in s.marker_genes}
    if not gene_states or not escape_ids:
        return 0.0

    escape_flow = 0.0
    total_flow = 0.0
    for t in transitions:
        if t.from_state in gene_states:
            total_flow += t.baseline_probability
            if t.to_state in escape_ids:
                escape_flow += t.baseline_probability

    if total_flow == 0:
        return 0.0
    return min(1.0, escape_flow / total_flow)


def _non_response_risk(
    gene: str,
    transition_result: dict,
    state_result: dict,
    resolution: str = "intermediate",
) -> float:
    """
    Non-response risk: fraction of pathological states where `gene` is NOT a marker gene.
    High = perturbation won't reach these cells at all.

    Returns [0, 1].
    """
    states = state_result.get(resolution, [])
    basin_assignments = transition_result.get("basin_assignments", {})
    path_states = [s for s in states if basin_assignments.get(s.state_id) == "pathological"]

    if not path_states:
        return 0.0

    unreached = sum(1 for s in path_states if gene not in s.marker_genes)
    return unreached / len(path_states)


# ---------------------------------------------------------------------------
# Public API  (deprecated — use therapeutic_redirection.py instead)
# ---------------------------------------------------------------------------

def score_gene(
    gene: str,
    transition_result: dict,
    state_result: dict,
    failure_records: list | None = None,
    disease: str = "",
    resolution: str = "intermediate",
) -> TrajectoryRedirectionScore:
    """
    DEPRECATED. Use pipelines.state_space.therapeutic_redirection instead.
    """
    import warnings
    warnings.warn(
        "trajectory_scoring.score_gene() is deprecated (Phase C refactor). "
        "Use pipelines.state_space.therapeutic_redirection.compute_therapeutic_redirection().",
        DeprecationWarning, stacklevel=2,
    )
    """
    Compute TrajectoryRedirectionScore for a single gene.

    Args:
        gene:              Gene symbol
        transition_result: Output of infer_state_transition_graph()
        state_result:      Output of define_cell_states()
        failure_records:   FailureRecord list from build_failure_records() (optional)
        disease:           Disease key (IBD, CAD, …)
        resolution:        Which resolution level to score at

    Returns:
        TrajectoryRedirectionScore
    """
    from pipelines.state_space.failure_memory import failure_penalty_score

    path_red = _expected_reduction_after_perturbation(gene, transition_result, state_result, resolution)
    dur      = _durability_score(gene, transition_result, state_result, resolution)
    esc_risk = _escape_risk(gene, transition_result, state_result, resolution)
    nr_risk  = _non_response_risk(gene, transition_result, state_result, resolution)

    # Failure memory: escape + non_responder modes
    neg_mem = 0.0
    if failure_records:
        neg_mem = failure_penalty_score(gene, failure_records)

    # Composite score = weighted sum, penalised by escape and memory
    composite = (
        W_TRAJECTORY_REDIRECTION * path_red
        + W_DURABILITY            * dur
        - W_ESCAPE_PENALTY        * esc_risk
        - W_NEGATIVE_MEMORY       * neg_mem
    )
    # Clip to [0, 1] — composite can go slightly negative for bad targets
    import numpy as np
    composite = float(np.clip(composite, 0.0, 1.0))

    return TrajectoryRedirectionScore(
        entity_id                   = gene,
        entity_type                 = "gene",
        disease                     = disease or transition_result.get("disease", ""),
        expected_pathology_reduction = path_red,
        durable_redirection_score   = dur,
        escape_risk_score           = esc_risk,
        non_response_risk_score     = nr_risk,
        negative_memory_penalty     = neg_mem,
    )


def score_all_genes(
    genes: list[str],
    transition_result: dict,
    state_result: dict,
    failure_records: list | None = None,
    disease: str = "",
    resolution: str = "intermediate",
) -> list[TrajectoryRedirectionScore]:
    """
    DEPRECATED. Use pipelines.state_space.therapeutic_redirection instead.
    Score all genes; sort by expected_pathology_reduction descending.
    """
    import warnings
    warnings.warn(
        "trajectory_scoring.score_all_genes() is deprecated (Phase C refactor). "
        "Use pipelines.state_space.therapeutic_redirection.compute_therapeutic_redirection().",
        DeprecationWarning, stacklevel=2,
    )
    scores = [
        score_gene(g, transition_result, state_result, failure_records, disease, resolution)
        for g in genes
    ]
    return sorted(scores, key=lambda s: s.expected_pathology_reduction, reverse=True)
