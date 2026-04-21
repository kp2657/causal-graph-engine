"""
pipelines/state_space/transition_scoring.py

Transition-landscape gene scoring — Phase G.

Replaces the mean-difference disease_axis_score (DAS) as the primary mechanistic
signal.  DAS is retained as an annotation-only field and no longer enters any
composite formula.

Four independent scores per gene
---------------------------------
entry_score          Expression enrichment in cells entering pathological basins.
                     Reference: all healthy-basin cells.

persistence_score    Expression enrichment in cells dwelling in pathological basins
                     (low exit probability).
                     Reference: recovery cells (cells actively exiting).

recovery_score       Expression enrichment in cells exiting toward healthy basins
                     (high exit probability).
                     Reference: persistence cells (cells stuck).

boundary_score       max(boundary_knn_score, boundary_pseudotime_score)
  boundary_knn_score     Enrichment at the spatial healthy/pathological interface.
                         Defined as cells with ≥30% of kNN neighbors in the
                         opposite disease class.
  boundary_pseudotime_score  Enrichment near the temporal inflection point —
                         the pseudotime bin where the healthy→pathological
                         fraction gradient is steepest.

Each score carries a direction (+1/-1/0):
  +1  gene elevated in category cells   (e.g. entry_direction=+1 → drives/marks entry)
  -1  gene depleted in category cells   (e.g. recovery_direction=-1 → its absence enables recovery)
   0  no detectable signal

Category
--------
  trigger     entry_score dominant
  maintenance persistence_score dominant
  recovery    recovery_score dominant
  mixed       no single score dominates

Transition inference
--------------------
Uses T_baseline (state-level Markov transition matrix from transition_graph.py)
as the primary signal for identifying entry/persistence/recovery cells.
kNN directionality is NOT used for cell categorisation — only for boundary_knn_score.

Public API
----------
    compute_transition_gene_scores(adata, transition_result, gene_list=None)
        -> dict[str, TransitionGeneProfile]
"""
from __future__ import annotations

from typing import Any

import numpy as np

from models.evidence import TransitionGeneProfile

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BOUNDARY_KNN_THRESHOLD  = 0.30   # fraction of opposite-class neighbors → boundary cell
_FLOW_QUARTILE           = 0.75   # top/bottom cut for cluster outflow classification
_MIN_CATEGORY_CELLS      = 5      # min cells in a category to compute a score
_CATEGORY_SCORE_THRESHOLD = 0.15  # min dominant score to assign a non-mixed category


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_gene_index(adata: Any) -> dict[str, int]:
    """Gene symbol/ID → column index. Handles CELLxGENE Ensembl var_names."""
    gene_idx: dict[str, int] = {g: i for i, g in enumerate(adata.var_names)}
    if "feature_name" in adata.var.columns:
        for i, sym in enumerate(adata.var["feature_name"]):
            if sym and str(sym) not in gene_idx:
                gene_idx[str(sym)] = i
    return gene_idx


def _get_state_col(adata: Any) -> str | None:
    """Find state assignment column in adata.obs (prefer intermediate resolution)."""
    for candidate in ("state_intermediate", "state_coarse", "state_fine"):
        if candidate in adata.obs.columns:
            return candidate
    for col in adata.obs.columns:
        if col.startswith("state_"):
            return col
    return None


def _cell_category_masks(
    adata: Any,
    transition_result: dict,
) -> dict[str, np.ndarray] | None:
    """
    Identify per-cell category masks using T_baseline and the kNN graph.

    Returns dict of boolean (n_cells,) arrays:
        path_mask       cells in pathological basins
        health_mask     cells in healthy basins
        entry           healthy cells in clusters with high outbound pathological flow
        persistence     pathological cells in clusters with low outbound healthy flow
        recovery        pathological cells in clusters with high outbound healthy flow
        boundary_knn    cells with ≥ _BOUNDARY_KNN_THRESHOLD opposite-class kNN neighbors
        boundary_pt     cells near the temporal pseudotime inflection point

    Returns None if basin data or state column is missing.
    """
    path_ids   = set(transition_result.get("pathologic_basin_ids", []))
    health_ids = set(transition_result.get("healthy_basin_ids",    []))
    if not path_ids and not health_ids:
        return None

    state_col = _get_state_col(adata)
    if state_col is None:
        return None

    disease    = transition_result.get("disease",    "")
    resolution = transition_result.get("resolution", "intermediate")
    prefix     = f"{disease}_{resolution}_"

    def _raw(sid: str) -> str:
        return sid[len(prefix):] if sid.startswith(prefix) else sid

    obs_labels  = adata.obs[state_col].astype(str).values
    path_mask   = np.isin(obs_labels, [_raw(s) for s in path_ids])
    health_mask = np.isin(obs_labels, [_raw(s) for s in health_ids])

    if path_mask.sum() < _MIN_CATEGORY_CELLS or health_mask.sum() < _MIN_CATEGORY_CELLS:
        return None

    # ---- Cluster-level outflow from T_baseline ----------------------------
    T        = transition_result.get("transition_matrix")
    s_labels = transition_result.get("state_labels", [])
    sid_to_idx: dict[str, int] = {sid: i for i, sid in enumerate(s_labels)}

    path_outflow:   dict[str, float] = {}
    health_outflow: dict[str, float] = {}

    if T is not None and len(s_labels) > 0:
        path_col_idx   = [sid_to_idx[s] for s in path_ids   if s in sid_to_idx]
        health_col_idx = [sid_to_idx[s] for s in health_ids if s in sid_to_idx]
        for sid in s_labels:
            i = sid_to_idx[sid]
            path_outflow[sid]   = float(T[i, path_col_idx].sum())   if path_col_idx   else 0.0
            health_outflow[sid] = float(T[i, health_col_idx].sum()) if health_col_idx else 0.0

    # ---- Entry cells: healthy clusters with top-quartile outbound path flow
    entry_mask = np.zeros(adata.n_obs, dtype=bool)
    h_flows = [(sid, path_outflow[sid]) for sid in health_ids if sid in path_outflow]
    if len(h_flows) >= 2:
        threshold = float(np.quantile([f for _, f in h_flows], _FLOW_QUARTILE))
        high_entry_raw = {_raw(sid) for sid, f in h_flows if f >= threshold}
        entry_mask = health_mask & np.isin(obs_labels, list(high_entry_raw))
    if entry_mask.sum() < _MIN_CATEGORY_CELLS:
        entry_mask = health_mask  # degenerate fallback

    # ---- Persistence / Recovery cells from pathological cluster outflow ---
    persistence_mask = np.zeros(adata.n_obs, dtype=bool)
    recovery_mask    = np.zeros(adata.n_obs, dtype=bool)
    p_flows = [(sid, health_outflow[sid]) for sid in path_ids if sid in health_outflow]
    if len(p_flows) >= 2:
        threshold_low  = float(np.quantile([f for _, f in p_flows], 1.0 - _FLOW_QUARTILE))
        threshold_high = float(np.quantile([f for _, f in p_flows], _FLOW_QUARTILE))
        low_exit_raw   = {_raw(sid) for sid, f in p_flows if f <= threshold_low}
        high_exit_raw  = {_raw(sid) for sid, f in p_flows if f >= threshold_high}
        persistence_mask = path_mask & np.isin(obs_labels, list(low_exit_raw))
        recovery_mask    = path_mask & np.isin(obs_labels, list(high_exit_raw))
    if persistence_mask.sum() < _MIN_CATEGORY_CELLS:
        persistence_mask = path_mask
    if recovery_mask.sum() < _MIN_CATEGORY_CELLS:
        recovery_mask = path_mask

    # ---- Boundary kNN: sparse matrix-vector product (memory-efficient) ----
    boundary_knn_mask = np.zeros(adata.n_obs, dtype=bool)
    conn = adata.obsp.get("connectivities")
    if conn is not None:
        try:
            conn_csr = conn.tocsr() if hasattr(conn, "tocsr") else conn
            health_vec = health_mask.astype(float)
            path_vec   = path_mask.astype(float)
            healthy_nbr_count = np.asarray(conn_csr.dot(health_vec)).ravel()
            path_nbr_count    = np.asarray(conn_csr.dot(path_vec)).ravel()
            total_nbr         = np.asarray(conn_csr.sum(axis=1)).ravel()
            total_nbr         = np.where(total_nbr == 0, 1.0, total_nbr)
            boundary_knn_mask = (
                (path_mask   & (healthy_nbr_count / total_nbr >= _BOUNDARY_KNN_THRESHOLD)) |
                (health_mask & (path_nbr_count    / total_nbr >= _BOUNDARY_KNN_THRESHOLD))
            )
        except Exception:
            pass  # graceful degradation — boundary_knn remains zeros

    # ---- Boundary pseudotime: cells near temporal inflection point --------
    boundary_pt_mask = np.zeros(adata.n_obs, dtype=bool)
    pt_series = adata.obs.get("dpt_pseudotime")
    if pt_series is not None:
        try:
            pt_arr = np.asarray(pt_series, dtype=float)
            n_bins = 10
            bin_edges    = np.percentile(pt_arr, np.linspace(0, 100, n_bins + 1))
            bin_assign   = np.digitize(pt_arr, bin_edges[1:-1])  # 0-indexed 0..n_bins-1
            path_fracs   = np.array([
                path_mask[bin_assign == b].mean() if (bin_assign == b).sum() > 0 else 0.0
                for b in range(n_bins)
            ])
            if len(path_fracs) > 1:
                # Inflection = bin before steepest absolute gradient
                inflection_bin = int(np.argmax(np.abs(np.diff(path_fracs))))
                boundary_bins  = {
                    max(0, inflection_bin - 1),
                    inflection_bin,
                    min(n_bins - 1, inflection_bin + 1),
                    min(n_bins - 1, inflection_bin + 2),
                }
                boundary_pt_mask = np.isin(bin_assign, list(boundary_bins))
        except Exception:
            pass

    return {
        "path_mask":    path_mask,
        "health_mask":  health_mask,
        "entry":        entry_mask,
        "persistence":  persistence_mask,
        "recovery":     recovery_mask,
        "boundary_knn": boundary_knn_mask,
        "boundary_pt":  boundary_pt_mask,
    }


def _gene_enrichment(
    X,
    cat_mask: np.ndarray,
    ref_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-gene enrichment: normalised |mean_cat - mean_ref| with direction sign.

    X may be a dense ndarray or a scipy sparse matrix — both are handled.

    Returns:
        scores     (n_genes,) float  [0, 1]
        directions (n_genes,) int    {+1, -1, 0}
    """
    n_genes = X.shape[1]
    if cat_mask.sum() < _MIN_CATEGORY_CELLS or ref_mask.sum() < _MIN_CATEGORY_CELLS:
        return np.zeros(n_genes), np.zeros(n_genes, dtype=int)

    # Use integer indexing — safe for both sparse and dense matrices.
    # np.asarray + flatten converts scipy sparse .mean() result (numpy.matrix shape 1×n)
    # to a plain 1-D ndarray so boolean assignment into `dirs` works correctly.
    cat_idx = np.where(cat_mask)[0]
    ref_idx = np.where(ref_mask)[0]
    mean_cat = np.asarray(X[cat_idx].mean(axis=0)).flatten()
    mean_ref = np.asarray(X[ref_idx].mean(axis=0)).flatten()
    delta    = mean_cat - mean_ref
    abs_delta = np.abs(delta)

    try:
        p99 = max(float(np.percentile(abs_delta, 99)), 1e-8)
    except Exception:
        p99 = max(float(abs_delta.max()) * 0.1 + 1e-8, 1e-8)

    scores = np.clip(abs_delta / p99, 0.0, 1.0)
    dirs   = np.zeros(n_genes, dtype=int)
    dirs[delta >  1e-6] =  1
    dirs[delta < -1e-6] = -1

    return scores, dirs


def _compute_das(
    X: np.ndarray,
    path_mask: np.ndarray,
    health_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Legacy DAS: |mean_path - mean_health| normalised by p99. Annotation only."""
    return _gene_enrichment(X, path_mask, health_mask)


def _assign_category(
    entry: float,
    persistence: float,
    recovery: float,
    boundary: float,
) -> str:
    """Assign mechanistic category from dominant score."""
    candidates = {"trigger": entry, "maintenance": persistence, "recovery": recovery}
    best_name  = max(candidates, key=candidates.__getitem__)
    best_score = candidates[best_name]
    if best_score >= _CATEGORY_SCORE_THRESHOLD:
        return best_name
    return "mixed"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_transition_gene_scores(
    adata: Any,
    transition_result: dict,
    gene_list: list[str] | None = None,
) -> dict[str, "TransitionGeneProfile"]:
    """
    Compute transition-landscape scores for each gene.

    Args:
        adata:             AnnData with state obs columns, pseudotime, and kNN graph.
        transition_result: Output of infer_state_transition_graph().
        gene_list:         Genes to score.  None → all genes in adata.

    Returns:
        {gene: TransitionGeneProfile}
        Returns empty-profile entries for genes not found in adata.
        Returns all-zero profiles (with message in disease field) if data is
        insufficient for scoring.
    """
    if adata is None or adata.n_obs == 0:
        return {}

    disease = transition_result.get("disease", "")

    # Keep X as-is (sparse or dense) — toarray() would allocate ~800MB for 50k×2k float64.
    # _gene_enrichment handles both representations via integer indexing + np.asarray flatten.
    X = adata.X

    gene_idx = _build_gene_index(adata)
    targets  = gene_list if gene_list is not None else list(adata.var_names)

    # Cell category masks
    masks = _cell_category_masks(adata, transition_result)
    if masks is None:
        return {g: TransitionGeneProfile(gene=g, disease=disease) for g in targets}

    # Interior = cells not at any boundary
    any_boundary = masks["boundary_knn"] | masks["boundary_pt"]
    interior_mask = ~any_boundary
    ref_interior  = interior_mask if interior_mask.sum() >= _MIN_CATEGORY_CELLS else np.ones(adata.n_obs, dtype=bool)

    # Compute enrichment vectors (one call per category pair)
    entry_s,       entry_d       = _gene_enrichment(X, masks["entry"],       masks["health_mask"])
    persist_s,     persist_d     = _gene_enrichment(X, masks["persistence"], masks["recovery"])
    recovery_s,    recovery_d    = _gene_enrichment(X, masks["recovery"],    masks["persistence"])
    bknn_s,        bknn_d        = _gene_enrichment(X, masks["boundary_knn"], ref_interior)
    bpt_s,         bpt_d         = _gene_enrichment(X, masks["boundary_pt"],  ref_interior)
    das_s,         das_d         = _compute_das(X, masks["path_mask"], masks["health_mask"])

    # Cell counts (same for all genes in this context)
    n_entry       = int(masks["entry"].sum())
    n_persistence = int(masks["persistence"].sum())
    n_recovery    = int(masks["recovery"].sum())
    n_boundary    = int(any_boundary.sum())

    result: dict[str, TransitionGeneProfile] = {}

    for gene in targets:
        idx = gene_idx.get(gene)
        if idx is None:
            result[gene] = TransitionGeneProfile(gene=gene, disease=disease)
            continue

        e_s  = float(entry_s[idx])
        p_s  = float(persist_s[idx])
        r_s  = float(recovery_s[idx])
        bk_s = float(bknn_s[idx])
        bp_s = float(bpt_s[idx])
        b_s  = max(bk_s, bp_s)

        result[gene] = TransitionGeneProfile(
            gene                      = gene,
            disease                   = disease,
            entry_score               = e_s,
            persistence_score         = p_s,
            recovery_score            = r_s,
            boundary_knn_score        = bk_s,
            boundary_pseudotime_score = bp_s,
            boundary_score            = b_s,
            entry_direction           = int(entry_d[idx]),
            persistence_direction     = int(persist_d[idx]),
            recovery_direction        = int(recovery_d[idx]),
            boundary_direction        = int(bknn_d[idx]),
            mechanistic_category      = _assign_category(e_s, p_s, r_s, b_s),
            disease_axis_score        = float(das_s[idx]),
            n_entry_cells             = n_entry,
            n_persistence_cells       = n_persistence,
            n_recovery_cells          = n_recovery,
            n_boundary_cells          = n_boundary,
        )

    return result
