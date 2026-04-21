"""
pipelines/state_space/transition_graph.py

Infer a directed state-level transition graph from a latent embedding.

Phase 1 (v1) algorithm:
  - Pseudotime + kNN flow: for each kNN edge (i → j) where pseudotime[j] > pseudotime[i],
    count it as evidence for a transition from state[i] to state[j].
  - Normalise counts row-wise to obtain a Markov transition matrix.
  - Assign basins: pathological = states with high disease-cell fraction;
    healthy = states with high control-cell fraction; others = mixed / escape candidates.
  - All inferred transitions are explicitly marked provenance="pseudotime_inferred".

Upgrade path (Phase 2):
  - Replace kNN flow with direct perturbational transition evidence when Step B data
    (Papalexi PBMC Perturb-seq) is available.
  - mode="velocity" will enable RNA velocity via scvelo.
  - mode="optimal_transport" will use ot.emd2 between condition distributions.

Public API:
    infer_state_transition_graph(latent_result, state_result, disease, resolution, mode)
    get_basin_summary(transition_result)
"""
from __future__ import annotations

from typing import Any

import numpy as np

from models.evidence import CellState, StateTransition
from pipelines.state_space.schemas import (
    PATHOLOGICAL_ENRICHMENT_THRESHOLD,
    HEALTHY_ENRICHMENT_THRESHOLD,
    ESCAPE_STABILITY_THRESHOLD,
    PROV_PSEUDOTIME_INFERRED,
    PROV_KNN_FLOW,
)

_MIN_TRANSITION_PROB = 0.02   # transitions below this are pruned (noise threshold)


# ---------------------------------------------------------------------------
# Optimal Transport transition matrix (Phase 2)
# ---------------------------------------------------------------------------

def _build_ot_transition_matrix(
    embeddings: np.ndarray,     # (n_cells, n_dims) — PCA/UMAP coordinates
    assignments: np.ndarray,    # (n_cells,) str state labels
    state_labels: list,         # ordered unique state labels
    reg: float = 0.05,          # entropic regularisation for Sinkhorn (0 = exact EMD)
) -> np.ndarray:
    """
    Build a state-level transition matrix using Optimal Transport.

    For each ordered pair of states (i, j), compute the Earth Mover's Distance
    between the two empirical distributions in embedding space.  The transition
    probability T[i,j] is proportional to exp(-EMD(i,j) / median_EMD).

    Diagonal is set to 0 before row-normalisation (no self-loops).

    Uses POT (Python Optimal Transport, pot >= 0.9) via `ot.sinkhorn2` when
    reg > 0 (faster, approximate) or `ot.emd2` when reg=0 (exact Wasserstein).

    Args:
        embeddings:   PCA/UMAP cell coordinates.
        assignments:  Per-cell state labels (string).
        state_labels: Ordered list of unique state labels.
        reg:          Entropic regularisation (Sinkhorn). Use 0 for exact EMD.

    Returns:
        T: (n_states × n_states) row-normalised float array.
    """
    try:
        import ot as pot
    except ImportError:
        raise ImportError(
            "Python Optimal Transport (pot) required for mode='optimal_transport'. "
            "Install with: pip install POT"
        )

    n_states = len(state_labels)
    label_to_idx = {str(s): i for i, s in enumerate(state_labels)}
    assignments_str = np.asarray(assignments, dtype=str)

    # Group cell embeddings by state
    state_embeddings: list[np.ndarray] = []
    for label in state_labels:
        mask = assignments_str == str(label)
        if mask.sum() > 0:
            state_embeddings.append(embeddings[mask].astype(np.float64))
        else:
            state_embeddings.append(np.zeros((1, embeddings.shape[1]), dtype=np.float64))

    # Compute pairwise EMD (or Sinkhorn) between state distributions
    # Result: EMD_matrix[i,j] = cost of transporting state i distribution to state j
    EMD = np.zeros((n_states, n_states), dtype=float)
    for i in range(n_states):
        Ai = state_embeddings[i]
        ni = len(Ai)
        for j in range(n_states):
            if i == j:
                continue
            Bj = state_embeddings[j]
            nj = len(Bj)
            # Uniform weights over each state's cells
            a = np.ones(ni, dtype=np.float64) / ni
            b = np.ones(nj, dtype=np.float64) / nj
            # Squared Euclidean cost matrix
            M = pot.dist(Ai, Bj, metric="sqeuclidean")
            M /= M.max() + 1e-10  # normalise to [0,1]
            try:
                if reg > 0:
                    emd_val = float(pot.sinkhorn2(a, b, M, reg=reg))
                else:
                    emd_val = float(pot.emd2(a, b, M))
            except Exception:
                emd_val = 1.0  # fallback: maximum distance
            EMD[i, j] = emd_val

    # Convert distance → similarity: T[i,j] ∝ exp(-EMD[i,j] / σ)
    # σ = median of non-zero off-diagonal distances
    off_diag = EMD[EMD > 0]
    sigma = float(np.median(off_diag)) if len(off_diag) > 0 else 1.0
    T = np.exp(-EMD / (sigma + 1e-10))
    np.fill_diagonal(T, 0.0)

    # Row-normalise
    row_sums = T.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    T = T / row_sums

    return T


# ---------------------------------------------------------------------------
# Transition matrix computation
# ---------------------------------------------------------------------------

def _build_pseudotime_transition_matrix(
    assignments: np.ndarray,    # (n_cells,) int or str state labels
    pseudotime: np.ndarray,     # (n_cells,) float, range [0, 1]
    connectivity: Any,          # sparse or dense (n_cells × n_cells)
    state_labels: list,         # ordered unique state labels
) -> np.ndarray:
    """
    Build a state-level transition matrix from pseudotime-directed kNN flow.

    For each directed edge (i, j) where pseudotime[j] > pseudotime[i]:
      T[state[i], state[j]] += 1

    Row-normalise to give probability estimates.
    Diagonal (self-transitions) is set to 0 before normalisation so dwell
    probability is not conflated with transition probability.

    Returns:
        T: (n_states × n_states) float array, row-normalised.
    """
    n_states = len(state_labels)
    label_to_idx = {str(s): i for i, s in enumerate(state_labels)}
    T = np.zeros((n_states, n_states), dtype=float)

    # Normalise pseudotime to [0, 1] if needed
    pt = np.asarray(pseudotime, dtype=float)
    pt = np.where(np.isfinite(pt), pt, 0.0)  # guard NaN/inf before normalization
    pt_range = pt.max() - pt.min()
    if pt_range > 0:
        pt = (pt - pt.min()) / pt_range

    # Iterate over kNN edges
    if hasattr(connectivity, "tocoo"):
        coo = connectivity.tocoo()
        edge_iter = zip(coo.row, coo.col)
    else:
        rows_i, cols_j = np.where(np.asarray(connectivity) > 0)
        edge_iter = zip(rows_i, cols_j)

    assignments_str = np.asarray(assignments, dtype=str)

    for i, j in edge_iter:
        if i == j:
            continue
        if pt[j] > pt[i]:   # directed: low pseudotime → high pseudotime
            si = label_to_idx.get(assignments_str[i])
            sj = label_to_idx.get(assignments_str[j])
            if si is not None and sj is not None and si != sj:
                T[si, sj] += 1.0

    # Zero diagonal, row-normalise
    np.fill_diagonal(T, 0.0)
    row_sums = T.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    T = T / row_sums

    return T


def _compute_state_pseudotime(
    assignments: np.ndarray,
    pseudotime: np.ndarray,
    state_labels: list,
) -> dict[str, float]:
    """Mean pseudotime per state."""
    assignments_str = np.asarray(assignments, dtype=str)
    result = {}
    for label in state_labels:
        mask = assignments_str == str(label)
        if mask.sum() > 0:
            result[str(label)] = float(pseudotime[mask].mean())
        else:
            result[str(label)] = float("nan")
    return result


_BASIN_FALLBACK_TOP_FRACTION = 0.25   # top/bottom fraction used when absolute threshold fails


def _assign_basins(
    states: list[CellState],
) -> dict[str, str]:
    """
    Classify each state into a basin type.

    Returns {state_id: basin_type} where basin_type is one of:
      pathological | healthy | escape | repair | mixed

    Primary pass: absolute thresholds from schemas.py (works for balanced cohorts).
    Fallback pass: when no state reaches the absolute pathological threshold (e.g.
    heavily imbalanced AMD h5ad with ~1% disease cells), use relative ranking —
    top BASIN_FALLBACK_TOP_FRACTION by disease fraction → pathological,
    bottom BASIN_FALLBACK_TOP_FRACTION → healthy. Guarantees non-empty basins so
    that compute_net_trajectory_improvement can fire for TR scoring.
    """
    basin_map: dict[str, str] = {}
    for s in states:
        ps = s.pathological_score
        stab = s.stability_score

        if ps is None:
            basin_map[s.state_id] = "mixed"
            continue

        if ps >= PATHOLOGICAL_ENRICHMENT_THRESHOLD:
            # Unstable pathological state may be an escape basin
            if stab is not None and stab < ESCAPE_STABILITY_THRESHOLD:
                basin_map[s.state_id] = "escape"
            else:
                basin_map[s.state_id] = "pathological"
        elif ps <= (1.0 - HEALTHY_ENRICHMENT_THRESHOLD):
            basin_map[s.state_id] = "healthy"
        else:
            basin_map[s.state_id] = "mixed"

    # Fallback: absolute threshold produced no pathological basins (imbalanced cohort).
    # Classify by relative enrichment so TR can fire.
    if not any(bt == "pathological" for bt in basin_map.values()):
        scored = sorted(
            [(s.state_id, s.pathological_score)
             for s in states if s.pathological_score is not None],
            key=lambda x: x[1],
        )
        if len(scored) >= 2:
            n = len(scored)
            top_n = max(1, round(n * _BASIN_FALLBACK_TOP_FRACTION))
            bot_n = max(1, round(n * _BASIN_FALLBACK_TOP_FRACTION))
            for sid, _ in scored[-top_n:]:
                basin_map[sid] = "pathological"
            for sid, _ in scored[:bot_n]:
                basin_map[sid] = "healthy"

    return basin_map


# ---------------------------------------------------------------------------
# StateTransition object builder
# ---------------------------------------------------------------------------

def _build_state_transitions(
    T: np.ndarray,
    state_labels: list,
    state_pseudotime: dict[str, float],
    disease: str,
    min_prob: float = _MIN_TRANSITION_PROB,
) -> list[StateTransition]:
    """Convert a transition matrix into a list of StateTransition objects."""
    transitions: list[StateTransition] = []
    n = len(state_labels)
    for i in range(n):
        for j in range(n):
            prob = T[i, j]
            if prob < min_prob:
                continue
            si = str(state_labels[i])
            sj = str(state_labels[j])
            dwell = state_pseudotime.get(si)
            transitions.append(StateTransition(
                from_state          = si,
                to_state            = sj,
                disease             = disease,
                baseline_probability = float(prob),
                uncertainty         = None,   # Phase 2: bootstrap estimate
                dwell_time          = dwell,
                direction_evidence  = [PROV_PSEUDOTIME_INFERRED, PROV_KNN_FLOW],
                context_tags        = ["phase1_v1"],
            ))
    return transitions


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def infer_state_transition_graph(
    latent_result: dict,
    state_result: dict[str, list[CellState]],
    disease: str,
    resolution: str = "intermediate",
    mode: str = "auto",
) -> dict:
    """
    Infer a directed state transition graph from pseudotime-ordered kNN flow.

    Args:
        latent_result: Output of build_disease_latent_space().
        state_result:  Output of define_cell_states().
        disease:       Short disease key.
        resolution:    Which resolution layer to use (coarse/intermediate/fine).
        mode:          "auto" → pseudotime + kNN flow (Phase 1).
                       "velocity", "optimal_transport" → Phase 2 (not yet implemented).

    Returns dict with:
        transitions          — list[StateTransition]
        transition_matrix    — np.ndarray (n_states × n_states)
        state_labels         — list[str]  (ordered; index aligns with matrix)
        basin_assignments    — dict {state_id: basin_type}
        healthy_basin_ids    — list[str]
        pathologic_basin_ids — list[str]
        escape_basin_ids     — list[str]
        state_pseudotime     — dict {state_id: mean_pseudotime}
        confidence_summary   — dict with provenance metadata
        disease              — str
        resolution           — str
    """
    if latent_result.get("error"):
        raise ValueError(
            f"infer_state_transition_graph: latent_result has error: {latent_result['error']}"
        )

    adata = latent_result.get("adata")
    if adata is None:
        raise ValueError("infer_state_transition_graph: latent_result['adata'] is None")

    if mode not in ("auto", "pseudotime", "optimal_transport"):
        raise NotImplementedError(
            f"mode='{mode}' not supported. "
            "Supported: 'auto', 'pseudotime', 'optimal_transport'."
        )

    states = state_result.get(resolution)
    if not states:
        raise ValueError(
            f"infer_state_transition_graph: no states found for resolution='{resolution}'"
        )

    # --- Gather per-cell assignments at chosen resolution -------------------
    key_added = f"state_{resolution}"
    if key_added not in adata.obs.columns:
        raise ValueError(
            f"infer_state_transition_graph: '{key_added}' not found in adata.obs. "
            "Run define_cell_states() first."
        )

    assignments = np.asarray(adata.obs[key_added].values, dtype=str)

    # Map state_ids back to cluster labels (state_id = f"{disease}_{resolution}_{label}")
    prefix = f"{disease}_{resolution}_"
    state_label_map = {}   # label → state_id
    for s in states:
        label = s.state_id.replace(prefix, "", 1)
        state_label_map[label] = s.state_id

    # Unique cluster labels in assignment array (raw labels, not state_ids)
    unique_labels = sorted(set(assignments), key=str)

    # Pseudotime
    pseudotime = latent_result.get("pseudotime")
    if pseudotime is None:
        pseudotime = np.zeros(adata.n_obs, dtype=float)
        confidence_note = "no_pseudotime_available"
    else:
        pseudotime = np.asarray(pseudotime, dtype=float)
        confidence_note = PROV_PSEUDOTIME_INFERRED

    # Connectivity (needed only for pseudotime mode)
    conn = adata.obsp.get("connectivities")

    # --- Build transition matrix -------------------------------------------
    if mode == "optimal_transport":
        # Use PCA embedding from latent_result; fall back to raw obs if unavailable
        embeddings = latent_result.get("pca_embedding")
        if embeddings is None and hasattr(adata, "obsm") and "X_pca" in adata.obsm:
            embeddings = np.asarray(adata.obsm["X_pca"])
        if embeddings is None:
            # Last resort: use raw HVG expression (may be slow)
            X = adata.X
            if hasattr(X, "toarray"):
                X = X.toarray()
            embeddings = np.asarray(X, dtype=np.float32)

        T = _build_ot_transition_matrix(
            embeddings   = np.asarray(embeddings),
            assignments  = assignments,
            state_labels = unique_labels,
        )
        confidence_note = "optimal_transport_sinkhorn"
    else:
        # auto / pseudotime: kNN flow
        if conn is None:
            raise ValueError(
                "infer_state_transition_graph: no 'connectivities' matrix in adata.obsp. "
                "Ensure latent_model ran kNN step."
            )
        T = _build_pseudotime_transition_matrix(
            assignments  = assignments,
            pseudotime   = pseudotime,
            connectivity = conn,
            state_labels = unique_labels,
        )

    # Map raw labels to state_ids for output
    state_ids_ordered = [state_label_map.get(str(lbl), f"{prefix}{lbl}") for lbl in unique_labels]

    # --- State pseudotime (mean per state) ---------------------------------
    raw_state_pt = _compute_state_pseudotime(assignments, pseudotime, unique_labels)
    # Re-key to state_ids
    state_pseudotime_by_id = {
        state_label_map.get(str(lbl), f"{prefix}{lbl}"): v
        for lbl, v in raw_state_pt.items()
    }

    # --- Build StateTransition objects (keyed by state_id) ----------------
    transitions = _build_state_transitions(
        T            = T,
        state_labels = state_ids_ordered,
        state_pseudotime = state_pseudotime_by_id,
        disease      = disease,
    )

    # --- Basin assignment --------------------------------------------------
    basin_assignments = _assign_basins(states)

    healthy_basin_ids    = [sid for sid, bt in basin_assignments.items() if bt == "healthy"]
    pathologic_basin_ids = [sid for sid, bt in basin_assignments.items() if bt == "pathological"]
    escape_basin_ids     = [sid for sid, bt in basin_assignments.items() if bt == "escape"]

    # Detect whether relative fallback was used (no state met absolute threshold)
    scored_ps = [s.pathological_score for s in states if s.pathological_score is not None]
    _used_fallback = bool(pathologic_basin_ids) and (
        not scored_ps or max(scored_ps) < PATHOLOGICAL_ENRICHMENT_THRESHOLD
    )

    # Rebuild T with state_id labels for clarity in output
    confidence_summary = {
        "mode":                 mode,
        "provenance":           PROV_PSEUDOTIME_INFERRED,
        "pseudotime_note":      confidence_note,
        "n_states":             len(states),
        "n_transitions":        len(transitions),
        "n_healthy_basins":     len(healthy_basin_ids),
        "n_pathologic_basins":  len(pathologic_basin_ids),
        "n_escape_basins":      len(escape_basin_ids),
        "backend":              latent_result.get("backend", "unknown"),
        "phase":                "phase2_ot" if mode == "optimal_transport" else "phase1_v1",
        "basin_classification": (
            f"relative_fallback(top/bottom {int(_BASIN_FALLBACK_TOP_FRACTION*100)}%)"
            if _used_fallback else "absolute_threshold"
        ),
        "upgrade_note": (
            "Phase 2: replace with perturbation-direct transitions when "
            "Step B Perturb-seq data is available."
        ),
    }

    return {
        "transitions":          transitions,
        "transition_matrix":    T,
        "state_labels":         state_ids_ordered,
        "basin_assignments":    basin_assignments,
        "healthy_basin_ids":    healthy_basin_ids,
        "pathologic_basin_ids": pathologic_basin_ids,
        "escape_basin_ids":     escape_basin_ids,
        "state_pseudotime":     state_pseudotime_by_id,
        "confidence_summary":   confidence_summary,
        "disease":              disease,
        "resolution":           resolution,
    }


def get_basin_summary(transition_result: dict) -> dict:
    """
    Summarise basins as a human-readable dict.
    Useful for logging and report generation.
    """
    return {
        "n_states":              len(transition_result.get("state_labels", [])),
        "n_transitions":         len(transition_result.get("transitions", [])),
        "healthy_basins":        transition_result.get("healthy_basin_ids", []),
        "pathologic_basins":     transition_result.get("pathologic_basin_ids", []),
        "escape_basins":         transition_result.get("escape_basin_ids", []),
        "provenance":            transition_result.get("confidence_summary", {}).get("provenance"),
    }


# ---------------------------------------------------------------------------
# Transition gene weights (Phase A)
# ---------------------------------------------------------------------------

def compute_transition_gene_weights(
    adata: Any,
    transition_result: dict,
    pathologic_basin_ids: list[str] | None = None,
    healthy_basin_ids: list[str] | None = None,
    n_top_genes: int = 200,
) -> dict[str, float]:
    """
    Compute per-gene transition DE signal: contribution to pathological-vs-healthy
    attractor separation.  Used as the transition_de_signal term in P_loading.

    Algorithm:
      1. Identify cells in pathological basins vs healthy basins.
      2. Compute per-gene |mean_disease - mean_healthy| on log-normalised counts.
      3. Clip to [0, 1] by dividing by the 99th-percentile delta.
      4. Return dict: {gene_name: clipped_de_signal}.

    Returns {} if fewer than 10 cells in either group or adata lacks X / obs.

    Args:
        adata:                   Preprocessed AnnData (obs has "leiden_*" or state column).
        transition_result:       Output of infer_state_transition_graph.
        pathologic_basin_ids:    Override list of pathological state IDs.
        healthy_basin_ids:       Override list of healthy state IDs.
        n_top_genes:             Number of top genes to return (by |delta|).

    Returns:
        {gene: clipped_de_signal}  where 0 ≤ de_signal ≤ 1.
    """
    if adata is None or adata.n_obs == 0:
        return {}

    path_ids  = set(pathologic_basin_ids or transition_result.get("pathologic_basin_ids", []))
    health_ids = set(healthy_basin_ids  or transition_result.get("healthy_basin_ids",   []))

    if not path_ids and not health_ids:
        return {}

    # Identify which obs column carries state assignments
    state_col: str | None = None
    for col in adata.obs.columns:
        if col.startswith("leiden_"):
            state_col = col
            break
    if state_col is None:
        return {}

    state_labels_obs = adata.obs[state_col].astype(str)
    path_mask   = state_labels_obs.isin(path_ids).values
    health_mask = state_labels_obs.isin(health_ids).values

    if path_mask.sum() < 10 or health_mask.sum() < 10:
        return {}

    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)

    mean_path   = X[path_mask].mean(axis=0)
    mean_health = X[health_mask].mean(axis=0)
    delta = np.abs(mean_path - mean_health)

    # Clip to [0, 1] using 99th percentile
    p99 = float(np.percentile(delta, 99))
    if p99 == 0:
        return {}
    delta_clipped = np.clip(delta / p99, 0.0, 1.0)

    # Get gene names
    gene_names: list[str]
    if hasattr(adata.var, "index"):
        gene_names = list(adata.var.index)
    else:
        gene_names = [str(i) for i in range(adata.n_vars)]

    if len(gene_names) != len(delta_clipped):
        return {}

    # Return top-N by delta
    top_idx = np.argsort(delta_clipped)[::-1][:n_top_genes]
    return {gene_names[i]: float(delta_clipped[i]) for i in top_idx}
