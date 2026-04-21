"""
pipelines/state_space/state_definition.py

Convert a latent embedding (from latent_model.py) into multi-resolution CellState objects.

One h5ad produces BOTH:
  - NMF programs (via cnmf_runner.py)  — annotation / interpretability layer
  - Cell states (this module)           — primary causal substrate

cNMF program labels are overlaid onto states as annotations; they do not define state identity.

Clustering hierarchy (in descending preference):
  1. scanpy Leiden         — if scanpy is installed
  2. scipy KMeans (kmeans2) — always available
  (sklearn KMeans is not used — not installed in causal-graph env)

Resolutions default to DEFAULT_RESOLUTIONS from schemas.py:
  coarse=0.20, intermediate=0.50, fine=1.00

Per-state annotations computed:
  - marker_genes         : top N genes by mean log-fold-change vs. rest
  - pathological_score   : fraction of cells in state labelled "disease"
  - stability_score      : kNN purity (fraction of each cell's neighbors in same state)
  - program_labels       : cNMF / Hallmark programs enriched in this state
  - context_tags         : disease | control | mixed + resolution tag

Caching:
  CellState objects are serialised as JSON alongside the latent cache, keyed off the
  latent h5ad mtime.  Pass use_cache=False to force recompute.
  Cache file: state_cache_{latent_stem}_{res_hash}.json
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from models.evidence import CellState
from pipelines.state_space.schemas import (
    DEFAULT_RESOLUTIONS,
    PATHOLOGICAL_ENRICHMENT_THRESHOLD,
    HEALTHY_ENRICHMENT_THRESHOLD,
)

_N_MARKER_GENES = 20      # top marker genes to report per state
_MIN_CELLS_PER_STATE = 5  # states smaller than this are flagged in context_tags


# ---------------------------------------------------------------------------
# State-definition cache helpers
# ---------------------------------------------------------------------------

def _res_hash(resolutions: dict[str, float]) -> str:
    """Short hash of the resolution dict for cache keying."""
    key = json.dumps(sorted(resolutions.items()))
    return hashlib.md5(key.encode()).hexdigest()[:8]


def _state_cache_paths(
    latent_result: dict,
    resolutions: dict[str, float],
) -> tuple[Path, Path] | tuple[None, None]:
    """
    Return (state_json_path, meta_json_path) co-located with the latent cache,
    or (None, None) if no anchor path is available.
    """
    provenance = latent_result.get("provenance", {})
    anchor: str | None = provenance.get("cache_file")
    if not anchor:
        dataset_paths = provenance.get("dataset_paths", [])
        if dataset_paths:
            anchor = dataset_paths[0]
    if not anchor:
        return None, None

    cache_dir  = Path(anchor).parent
    stem       = Path(anchor).stem
    res_tag    = _res_hash(resolutions)
    state_p    = cache_dir / f"state_cache_{stem}_{res_tag}.json"
    meta_p     = cache_dir / f"state_cache_{stem}_{res_tag}_meta.json"
    return state_p, meta_p


def _latent_anchor_mtime(latent_result: dict) -> float | None:
    """Return mtime of the latent cache h5ad, or None if unavailable."""
    provenance = latent_result.get("provenance", {})
    cache_file = provenance.get("cache_file")
    if cache_file and Path(cache_file).exists():
        return Path(cache_file).stat().st_mtime
    # Fall back to first dataset path
    for dp in provenance.get("dataset_paths", []):
        p = Path(dp)
        if p.exists():
            return p.stat().st_mtime
    return None


def _is_state_cache_valid(
    state_path: Path,
    meta_path: Path,
    latent_mtime: float | None,
) -> bool:
    """True if cache exists and was built from a latent embedding with the same mtime."""
    if latent_mtime is None:
        return False
    if not state_path.exists() or not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text())
        return abs(meta.get("latent_mtime", -1) - latent_mtime) < 1.0
    except Exception:
        return False


def _obs_cache_path(state_path: Path) -> Path:
    """Path to the obs-label sidecar (numpy .npz) co-located with the state JSON cache."""
    return state_path.with_suffix(".obs.npz")


def _save_obs_cache(adata: Any, state_path: Path, resolutions: dict[str, float]) -> None:
    """Save per-resolution obs cluster labels to a .npz sidecar.  Failures are silent."""
    try:
        import numpy as np
        arrays = {}
        for res_name in resolutions:
            key = f"state_{res_name}"
            if key in adata.obs.columns:
                arrays[key] = adata.obs[key].values.astype(str)
        if arrays:
            np.savez_compressed(str(_obs_cache_path(state_path)), **arrays)
    except Exception:
        pass


def _load_obs_cache(adata: Any, state_path: Path, resolutions: dict[str, float]) -> bool:
    """
    Restore obs cluster labels from the .npz sidecar into adata.obs.

    Returns True if all requested resolution columns were successfully restored,
    False if the sidecar is missing or incomplete (caller should fall back to clustering).
    """
    obs_path = _obs_cache_path(state_path)
    if not obs_path.exists():
        return False
    try:
        import numpy as np
        data = np.load(str(obs_path), allow_pickle=False)
        restored = 0
        for res_name in resolutions:
            key = f"state_{res_name}"
            if key in data:
                adata.obs[key] = data[key]
                restored += 1
        return restored == len(resolutions)
    except Exception:
        return False


def _save_state_cache(
    result: dict[str, list],
    state_path: Path,
    meta_path: Path,
    latent_mtime: float,
) -> None:
    """Serialise CellState lists to JSON.  Failures are silent."""
    try:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        serialised = {
            res: [cs.model_dump() for cs in states]
            for res, states in result.items()
        }
        state_path.write_text(json.dumps(serialised))
        meta_path.write_text(json.dumps({
            "latent_mtime": latent_mtime,
            "created_at":   time.time(),
        }))
    except Exception:
        pass


def _load_state_cache(state_path: Path) -> dict[str, list] | None:
    """Load and reconstruct CellState objects from cache JSON.  Returns None on failure."""
    try:
        raw = json.loads(state_path.read_text())
        return {
            res: [CellState.model_validate(cs) for cs in states]
            for res, states in raw.items()
        }
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def _cluster_leiden(adata: Any, resolution: float, key_added: str) -> None:
    """Try scanpy Leiden clustering. Raises ImportError if scanpy absent."""
    import scanpy as sc
    sc.tl.leiden(adata, resolution=resolution, key_added=key_added, flavor="igraph", n_iterations=2, directed=False)


def _cluster_kmeans(adata: Any, n_clusters: int, key_added: str) -> None:
    """scipy KMeans2 fallback — no sklearn required."""
    import numpy as np
    from scipy.cluster.vq import kmeans2, whiten

    X = adata.obsm.get("X_pca")
    if X is None:
        X = adata.obsm.get("X_diffmap")
    if X is None:
        raise ValueError("No latent coordinates in adata.obsm (expected X_pca or X_diffmap)")

    X_w = whiten(X.astype(float) + 1e-8)
    n_clusters = min(n_clusters, adata.n_obs)
    _, labels = kmeans2(X_w, n_clusters, seed=42, minit="points", iter=50)
    adata.obs[key_added] = labels.astype(str)


def _cluster_states(adata: Any, resolution_value: float, key_added: str) -> None:
    """
    Cluster cells into states at a given resolution.
    Tries Leiden, falls back to KMeans (scipy).
    """
    n_clusters = max(2, int(resolution_value * 15))  # coarse mapping resolution→k
    try:
        _cluster_leiden(adata, resolution_value, key_added)
    except (ImportError, Exception):
        _cluster_kmeans(adata, n_clusters, key_added)


# ---------------------------------------------------------------------------
# State characterisation helpers
# ---------------------------------------------------------------------------

def _compute_marker_genes(
    adata: Any,
    state_mask: Any,          # boolean array (n_cells,)
    n_markers: int = _N_MARKER_GENES,
) -> list[str]:
    """
    Top marker genes: mean log-FC of in-state cells vs. out-of-state cells.
    Falls back to top expressed genes if expression matrix is not accessible.
    """
    import numpy as np

    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = X.astype(float)

    in_mask  = state_mask.astype(bool)
    out_mask = ~in_mask

    if in_mask.sum() == 0 or out_mask.sum() == 0:
        return []

    mean_in  = X[in_mask].mean(axis=0)
    mean_out = X[out_mask].mean(axis=0)
    lfc = mean_in - mean_out   # log-space: this is Δ log-counts ≈ log-FC

    top_idx = np.argsort(lfc)[::-1][:n_markers]
    gene_names = list(adata.var_names)
    return [gene_names[i] for i in top_idx if lfc[i] > 0]


def _compute_pathological_score(
    adata: Any,
    state_mask: Any,
    disease_col: str = "disease_condition",
    disease_label: str = "disease",
) -> float | None:
    """
    Fraction of cells in this state labelled as disease.

    Supports two column conventions:
    - ``disease_condition`` with values ``"disease"`` / ``"healthy"`` (internal)
    - ``disease`` from CELLxGENE census (MONDO labels; ``"normal"`` = healthy,
      everything else = disease)
    """
    import numpy as np

    if disease_col in adata.obs.columns:
        labels = adata.obs[disease_col].values[state_mask.astype(bool)]
        if len(labels) == 0:
            return None
        return float((labels == disease_label).sum() / len(labels))

    # CELLxGENE census fallback: "disease" column with MONDO ontology labels
    if "disease" in adata.obs.columns:
        labels = adata.obs["disease"].values[state_mask.astype(bool)]
        if len(labels) == 0:
            return None
        return float((labels != "normal").sum() / len(labels))

    return None


def _compute_stability_score(
    adata: Any,
    assignments: Any,         # per-cell cluster labels (str array)
    state_label: str,
) -> float | None:
    """
    kNN purity for this state: fraction of each cell's neighbors assigned to same state.
    Returns None if connectivity matrix is absent.
    """
    import numpy as np

    conn = adata.obsp.get("connectivities")
    if conn is None:
        return None

    state_mask = (assignments == state_label)
    if state_mask.sum() == 0:
        return None

    if hasattr(conn, "toarray"):
        conn_dense = conn.toarray()
    else:
        conn_dense = conn

    cell_indices = np.where(state_mask)[0]
    purities = []
    for idx in cell_indices:
        neighbors = np.where(conn_dense[idx] > 0)[0]
        if len(neighbors) == 0:
            continue
        same_state = (assignments[neighbors] == state_label).sum()
        purities.append(same_state / len(neighbors))

    return float(np.mean(purities)) if purities else None


def _compute_centroid(adata: Any, state_mask: Any) -> list[float] | None:
    """Mean latent coordinate for cells in this state."""
    import numpy as np

    latent = adata.obsm.get("X_diffmap")
    if latent is None:
        latent = adata.obsm.get("X_pca")
    if latent is None:
        return None
    mask = state_mask.astype(bool)
    if mask.sum() == 0:
        return None
    return latent[mask].mean(axis=0).tolist()


def _annotate_with_programs(
    state_genes: list[str],
    program_labels_source: dict[str, list[str]],
    min_overlap: int = 3,
) -> list[str]:
    """
    Return program IDs whose gene sets overlap with state marker genes by >= min_overlap.
    program_labels_source: {program_id: [gene, ...]} from cNMF or MSigDB.
    """
    matched = []
    state_set = set(state_genes)
    for prog_id, prog_genes in program_labels_source.items():
        overlap = len(state_set & set(prog_genes))
        if overlap >= min_overlap:
            matched.append(prog_id)
    return matched


def _build_context_tags(
    n_cells: int,
    pathological_score: float | None,
    resolution: str,
) -> list[str]:
    tags = [f"resolution:{resolution}"]
    if n_cells < _MIN_CELLS_PER_STATE:
        tags.append("sparse_state")
    if pathological_score is not None:
        if pathological_score >= PATHOLOGICAL_ENRICHMENT_THRESHOLD:
            tags.append("pathological")
        elif pathological_score <= (1.0 - HEALTHY_ENRICHMENT_THRESHOLD):
            tags.append("healthy")
        else:
            tags.append("mixed")
    return tags


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def define_cell_states(
    latent_result: dict,
    disease: str,
    resolutions: dict[str, float] | None = None,
    program_labels_source: dict[str, list[str]] | None = None,
    use_cache: bool = True,
) -> dict[str, list[CellState]]:
    """
    Convert a latent embedding into multi-resolution CellState objects.

    Args:
        latent_result:        Output of build_disease_latent_space().
        disease:              Short disease key, e.g. "IBD".
        resolutions:          Override default resolution dict {name: float}.
                              Defaults to DEFAULT_RESOLUTIONS.
        program_labels_source: {program_id: [gene, ...]} from cNMF / MSigDB.
                              Used to annotate states with program overlaps.
        use_cache:            Load/save CellState JSON alongside the latent cache.
                              Set False to force recompute.

    Returns:
        {resolution_name: list[CellState]}

    Raises:
        ValueError if latent_result contains an error or no adata.
    """
    if latent_result.get("error"):
        raise ValueError(
            f"define_cell_states: latent_result has error: {latent_result['error']}"
        )

    adata = latent_result.get("adata")
    if adata is None:
        raise ValueError("define_cell_states: latent_result['adata'] is None")

    resolutions = resolutions or DEFAULT_RESOLUTIONS
    program_labels_source = program_labels_source or {}

    # --- Cache check ---------------------------------------------------------
    if use_cache:
        latent_mtime = _latent_anchor_mtime(latent_result)
        state_path, meta_path = _state_cache_paths(latent_result, resolutions)
        if state_path is not None and _is_state_cache_valid(state_path, meta_path, latent_mtime):
            cached = _load_state_cache(state_path)
            if cached is not None:
                # Cache hit: restore obs labels from sidecar (fast, µs).
                # If sidecar is absent (old cache), fall back to one-time clustering
                # and immediately save the sidecar for future runs.
                obs_restored = _load_obs_cache(adata, state_path, resolutions)
                if not obs_restored:
                    for res_name, res_value in resolutions.items():
                        key_added = f"state_{res_name}"
                        if key_added not in adata.obs.columns:
                            _cluster_states(adata, res_value, key_added)
                    _save_obs_cache(adata, state_path, resolutions)
                return cached

    result: dict[str, list[CellState]] = {}

    for res_name, res_value in resolutions.items():
        key_added = f"state_{res_name}"
        _cluster_states(adata, res_value, key_added)

        import numpy as np
        assignments = adata.obs[key_added].values
        unique_states = sorted(set(assignments), key=lambda x: str(x))

        states: list[CellState] = []
        for label in unique_states:
            state_mask = (assignments == label)
            n_cells_state = int(state_mask.sum())

            marker_genes     = _compute_marker_genes(adata, state_mask)
            pathological     = _compute_pathological_score(adata, state_mask)
            stability        = _compute_stability_score(adata, assignments, label)
            centroid         = _compute_centroid(adata, state_mask)
            prog_labels      = _annotate_with_programs(marker_genes, program_labels_source)
            context_tags     = _build_context_tags(n_cells_state, pathological, res_name)

            # Cell type: dominant cell_type in state if column exists
            cell_type = "unknown"
            if "cell_type" in adata.obs.columns:
                ct_counts = adata.obs.loc[state_mask, "cell_type"].value_counts()
                if len(ct_counts) > 0:
                    cell_type = str(ct_counts.index[0])

            state_id = f"{disease}_{res_name}_{label}"

            states.append(CellState(
                state_id          = state_id,
                disease           = disease,
                cell_type         = cell_type,
                resolution        = res_name,
                n_cells           = n_cells_state,
                centroid          = centroid,
                marker_genes      = marker_genes,
                program_labels    = prog_labels,
                context_tags      = context_tags,
                stability_score   = stability,
                pathological_score = pathological,
                evidence_sources  = latent_result.get("provenance", {}).get("dataset_paths", []),
            ))

        result[res_name] = states

    # --- Cache save ----------------------------------------------------------
    if use_cache:
        latent_mtime = _latent_anchor_mtime(latent_result)
        state_path, meta_path = _state_cache_paths(latent_result, resolutions)
        if state_path is not None and latent_mtime is not None:
            _save_state_cache(result, state_path, meta_path, latent_mtime)
            _save_obs_cache(adata, state_path, resolutions)  # sidecar for fast obs restore

    return result


def get_pathological_states(
    state_result: dict[str, list[CellState]],
    resolution: str = "intermediate",
) -> list[CellState]:
    """Return states at the given resolution with pathological_score >= threshold."""
    return [
        s for s in state_result.get(resolution, [])
        if s.pathological_score is not None
        and s.pathological_score >= PATHOLOGICAL_ENRICHMENT_THRESHOLD
    ]


def get_healthy_states(
    state_result: dict[str, list[CellState]],
    resolution: str = "intermediate",
) -> list[CellState]:
    """Return states at the given resolution with pathological_score <= (1 - threshold)."""
    return [
        s for s in state_result.get(resolution, [])
        if s.pathological_score is not None
        and s.pathological_score <= (1.0 - HEALTHY_ENRICHMENT_THRESHOLD)
    ]
