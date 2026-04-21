"""
pipelines/state_space/latent_model.py

Build a disease-specific latent embedding from sc-RNA h5ad files.

Two computation backends:
  1. _NumpyPCABackend  — always available (numpy + scipy only); used for tests and
                         environments where scanpy is not installed.
  2. PCADiffusionBackend — preferred production backend: scanpy PCA + diffusion map
                            + DPT pseudotime.
  3. ScVIBackend        — stub for Phase 2 (GPU environment); raises NotImplementedError.

Backend selection:
  get_backend(use_scvi=False)  → PCADiffusionBackend (or _NumpyPCABackend if scanpy absent)
  get_backend(use_scvi=True)   → ScVIBackend

The public API is build_disease_latent_space(), which accepts an optional
_adata_override for testing (skips disk load, uses the supplied AnnData directly).

Caching:
  Processed AnnData (post-preprocessing + post-embedding) is saved as h5ad alongside
  the source file: data/cellxgene/{disease}/latent_cache_{stem}_{backend}.h5ad
  Invalidated when source h5ad mtime changes.  Pass use_cache=False to force recompute.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class LatentBackend(Protocol):
    name: str

    def fit_transform(self, adata: Any, **kwargs) -> dict[str, Any]:
        """
        Compute latent embedding on a preprocessed AnnData.

        Must store results directly into adata.obsm and adata.obs:
          adata.obsm["X_pca"]          — PCA coordinates
          adata.obsm["X_diffmap"]      — diffusion / latent coordinates (primary)
          adata.obs["dpt_pseudotime"]  — pseudotime [0, 1] per cell
          adata.obsp["connectivities"] — sparse kNN connectivity matrix

        Returns a dict of keys used, e.g.:
          {"latent_key": "X_diffmap", "pca_key": "X_pca",
           "pseudotime_key": "dpt_pseudotime", "backend": self.name}
        """
        ...


# ---------------------------------------------------------------------------
# Backend 1: _NumpyPCABackend — no scanpy dependency
# ---------------------------------------------------------------------------

class _NumpyPCABackend:
    """
    Minimal PCA backend using only numpy + scipy.
    Suitable for unit tests and environments without scanpy.

    PCA    — numpy SVD
    kNN    — scipy.spatial.KDTree
    Pseudo — Euclidean distance from root cell in PC space (monotone proxy)
    Latent — top n_pcs PCA components (stored as X_diffmap for API compatibility)
    """
    name: str = "numpy_pca"

    def fit_transform(
        self,
        adata: Any,
        n_pcs: int = 20,
        n_neighbors: int = 15,
        **kwargs,
    ) -> dict[str, Any]:
        import numpy as np
        from scipy.spatial import KDTree
        from scipy.sparse import csr_matrix

        X = adata.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = X.astype(float)

        # Library-size normalise + log1p
        row_sums = X.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        X = np.log1p(X / row_sums * 1e4)

        # PCA via truncated SVD
        n_comps = min(n_pcs, adata.n_obs - 1, adata.n_vars - 1)
        X_centered = X - X.mean(axis=0)
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        X_pca = (U[:, :n_comps] * S[:n_comps]).astype(float)
        adata.obsm["X_pca"] = X_pca

        # kNN connectivity (symmetric)
        k = min(n_neighbors, adata.n_obs - 1)
        tree = KDTree(X_pca)
        _, nn_idx = tree.query(X_pca, k=k + 1)  # +1: first result is self
        nn_idx = nn_idx[:, 1:]

        n_cells = X_pca.shape[0]
        rows = np.repeat(np.arange(n_cells), k)
        cols = nn_idx.ravel()
        conn = csr_matrix(
            (np.ones(len(rows)), (rows, cols)),
            shape=(n_cells, n_cells),
        )
        conn = (conn + conn.T)
        conn.data[:] = 1.0
        adata.obsp["connectivities"] = conn

        # Pseudotime: normalised Euclidean distance from root (min PC1 cell)
        root_idx = int(np.argmin(X_pca[:, 0]))
        dists = np.linalg.norm(X_pca - X_pca[root_idx], axis=1)
        max_d = dists.max()
        pseudotime = dists / max_d if max_d > 0 else dists
        adata.obs["dpt_pseudotime"] = pseudotime.astype(float)

        # Latent = top 10 PCs (aliased as X_diffmap for API compatibility)
        n_latent = min(10, n_comps)
        adata.obsm["X_diffmap"] = X_pca[:, :n_latent]

        return {
            "latent_key":    "X_diffmap",
            "pca_key":       "X_pca",
            "pseudotime_key": "dpt_pseudotime",
            "backend":       self.name,
        }


# ---------------------------------------------------------------------------
# Backend 2: PCADiffusionBackend — scanpy-native (preferred)
# ---------------------------------------------------------------------------

class PCADiffusionBackend:
    """
    Production backend: scanpy PCA + neighbor graph + diffusion map + DPT pseudotime.
    Requires: scanpy, anndata.
    """
    name: str = "pca_diffusion"

    def fit_transform(
        self,
        adata: Any,
        n_pcs: int = 50,
        n_neighbors: int = 15,
        n_diffusion_components: int = 10,
        root_cell_idx: int | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        try:
            import scanpy as sc
            import numpy as np
        except ImportError as exc:
            raise ImportError(
                "PCADiffusionBackend requires scanpy: pip install scanpy anndata"
            ) from exc

        n_pcs_safe = max(1, min(n_pcs, adata.n_obs - 1, adata.n_vars - 1))
        sc.pp.pca(adata, n_comps=n_pcs_safe)
        sc.pp.neighbors(adata, n_pcs=n_pcs_safe, n_neighbors=n_neighbors)
        n_dc = min(n_diffusion_components, adata.n_obs - 1)
        sc.tl.diffmap(adata, n_comps=n_dc)

        # Determine pseudotime root: lowest diffusion component 1 value
        if root_cell_idx is None:
            root_cell_idx = int(np.argmin(adata.obsm["X_diffmap"][:, 0]))
        adata.uns["iroot"] = root_cell_idx
        sc.tl.dpt(adata)

        return {
            "latent_key":    "X_diffmap",
            "pca_key":       "X_pca",
            "pseudotime_key": "dpt_pseudotime",
            "backend":       self.name,
        }


# ---------------------------------------------------------------------------
# Backend 3: ScVIBackend — Phase 2 stub
# ---------------------------------------------------------------------------

class ScVIBackend:
    """
    Variational autoencoder backend via scvi-tools.
    Phase 2: full implementation requires GPU environment.
    Stub raises NotImplementedError until activated.
    """
    name: str = "scvi"

    def fit_transform(self, adata: Any, **kwargs) -> dict[str, Any]:
        try:
            import scvi  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "ScVIBackend requires scvi-tools: pip install scvi-tools"
            ) from exc
        raise NotImplementedError(
            "ScVIBackend: full implementation deferred to Phase 2 (GPU environment). "
            "Use get_backend(use_scvi=False) for CPU execution."
        )


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------

def get_backend(use_scvi: bool = False) -> LatentBackend:
    """
    Return the appropriate backend.

    use_scvi=False (default):
      → PCADiffusionBackend if scanpy is importable
      → _NumpyPCABackend otherwise (always works)

    use_scvi=True:
      → ScVIBackend (will raise NotImplementedError until Phase 2)
    """
    if use_scvi:
        return ScVIBackend()

    try:
        import scanpy  # noqa: F401
        return PCADiffusionBackend()
    except ImportError:
        return _NumpyPCABackend()


# ---------------------------------------------------------------------------
# Latent-space cache helpers
# ---------------------------------------------------------------------------

def _cache_paths(
    disease: str,
    dataset_paths: list[str],
    backend_name: str,
    cell_type_filter: list[str] | None,
) -> tuple[Path, Path]:
    """Return (h5ad_cache_path, meta_json_path) co-located with the first source file."""
    stem = Path(dataset_paths[0]).stem if dataset_paths else "unknown"
    if cell_type_filter:
        tag = "-".join(sorted(cell_type_filter))[:32]
        stem = f"{stem}_{tag}"
    cache_dir = Path(dataset_paths[0]).parent if dataset_paths else Path(f"data/cellxgene/{disease}")
    h5ad_p = cache_dir / f"latent_cache_{stem}_{backend_name}.h5ad"
    meta_p  = cache_dir / f"latent_cache_{stem}_{backend_name}.json"
    return h5ad_p, meta_p


def _is_cache_valid(
    h5ad_cache: Path,
    meta_path: Path,
    source_paths: list[str],
) -> bool:
    """True if cache exists and all source files have unchanged mtimes."""
    if not h5ad_cache.exists() or not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text())
        cached_mtimes: dict = meta.get("source_mtimes", {})
        for sp in source_paths:
            src = Path(sp)
            if not src.exists():
                return False
            if abs(src.stat().st_mtime - cached_mtimes.get(sp, -1)) > 1.0:
                return False
        return True
    except Exception:
        return False


def _save_latent_cache(
    adata: Any,
    h5ad_cache: Path,
    meta_path: Path,
    source_paths: list[str],
    backend_name: str,
) -> None:
    """Write processed AnnData + metadata to cache files.  Failures are silent."""
    try:
        import time
        h5ad_cache.parent.mkdir(parents=True, exist_ok=True)
        adata.write_h5ad(str(h5ad_cache))
        meta = {
            "source_mtimes": {
                sp: Path(sp).stat().st_mtime
                for sp in source_paths if Path(sp).exists()
            },
            "backend": backend_name,
            "n_cells": adata.n_obs,
            "n_genes": adata.n_vars,
            "created_at": time.time(),
        }
        meta_path.write_text(json.dumps(meta, indent=2))
    except Exception:
        pass  # caching is best-effort; never block the pipeline


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_and_concat_h5ads(dataset_paths: list[str]) -> Any:
    """Load one or more h5ad files; concatenate if multiple."""
    try:
        import anndata
    except ImportError as exc:
        raise ImportError(
            "latent_model requires anndata: pip install anndata"
        ) from exc

    adatas = []
    for p in dataset_paths:
        path = Path(p)
        if not path.exists():
            raise FileNotFoundError(f"h5ad not found: {p}")
        adatas.append(anndata.read_h5ad(str(path)))

    if len(adatas) == 1:
        return adatas[0]

    # Concatenate; ensure obs_names are unique
    for i, adata in enumerate(adatas):
        adata.obs_names = [f"ds{i}_{name}" for name in adata.obs_names]
    return anndata.concat(adatas, merge="same")


def _filter_cell_types(adata: Any, cell_type_filter: list[str]) -> Any:
    """Subset adata to rows where obs['cell_type'] is in the filter list."""
    col = None
    for candidate in ("cell_type", "cell_type_ontology_term_id", "cell_type_label"):
        if candidate in adata.obs.columns:
            col = candidate
            break
    if col is None:
        return adata  # no cell_type column — return as-is with a warning

    mask = adata.obs[col].isin(cell_type_filter)
    return adata[mask].copy()


def _preprocess_for_latent(adata: Any, warnings: list[str]) -> Any:
    """
    Standard preprocessing before embedding.
    Tries scanpy pipeline; falls back to minimal numpy normalisation.
    Does not modify the original adata in-place.
    """
    try:
        import scanpy as sc
    except ImportError:
        warnings.append("scanpy not available — using numpy-only preprocessing (all genes retained)")
        return adata.copy()

    try:
        adata = adata.copy()

        # Adaptive thresholds — avoid filtering everything away on small datasets
        min_genes_thresh = min(200, max(1, adata.n_vars // 4))
        min_cells_thresh = min(10, max(1, adata.n_obs // 20))
        n_top_hvg = min(2000, adata.n_vars)

        sc.pp.filter_cells(adata, min_genes=min_genes_thresh)
        sc.pp.filter_genes(adata, min_cells=min_cells_thresh)

        if adata.n_obs == 0 or adata.n_vars == 0:
            warnings.append(
                f"Preprocessing filtered all cells/genes "
                f"({adata.n_obs} cells, {adata.n_vars} genes) — skipping HVG selection"
            )
            return adata

        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        if n_top_hvg < adata.n_vars:
            try:
                sc.pp.highly_variable_genes(adata, n_top_genes=n_top_hvg, flavor="seurat_v3")
            except Exception:
                # seurat_v3 may require additional deps; fall back to seurat flavor
                sc.pp.highly_variable_genes(adata, n_top_genes=n_top_hvg, flavor="seurat")
            adata = adata[:, adata.var["highly_variable"]].copy()
        return adata
    except Exception as exc:
        warnings.append(f"scanpy preprocessing failed ({exc}) — using unprocessed data")
        return adata.copy()


# ---------------------------------------------------------------------------
# Phase Z7: Transition Motif Extraction (for Latent Hijack)
# ---------------------------------------------------------------------------

def extract_disease_transition_motif(
    adata: Any,
    pathological_state_id: str | None = None,
    healthy_state_id: str | None = None,
) -> np.ndarray | None:
    """
    Extract a "Transition Motif" from the latent space.

    The motif is defined as the mean expression difference (log2FC) between
    the pathological state(s) and healthy state(s) across all genes in the
    HVG-filtered latent space.

    Args:
        adata:                 AnnData with obs['state'] or similar.
        pathological_state_id: Label in obs for pathological cells.
        healthy_state_id:      Label in obs for healthy cells.

    Returns:
        np.ndarray (n_genes × 1) representing the motif vector.
    """
    import numpy as np

    # Resolve state column
    col = None
    for candidate in ("state", "disease", "status", "condition"):
        if candidate in adata.obs.columns:
            col = candidate
            break
    if col is None:
        return None

    # Resolve IDs if not provided
    if pathological_state_id is None:
        # Heuristic: any label containing "path", "dis", "case", "stim"
        for val in adata.obs[col].unique():
            v = str(val).lower()
            if any(x in v for x in ("path", "dis", "case", "stim", "amd", "ibd", "cad")):
                pathological_state_id = val
                break
    if healthy_state_id is None:
        # Heuristic: any label containing "heal", "cont", "norm", "ctrl"
        for val in adata.obs[col].unique():
            v = str(val).lower()
            if any(x in v for x in ("heal", "cont", "norm", "ctrl", "base")):
                healthy_state_id = val
                break

    if pathological_state_id is None or healthy_state_id is None:
        return None

    mask_p = adata.obs[col] == pathological_state_id
    mask_h = adata.obs[col] == healthy_state_id
    
    if mask_p.sum() < 5 or mask_h.sum() < 5:
        return None

    # Compute mean expression in HVG space
    # adata.X is already log1p-normalised from _preprocess_for_latent
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    
    mean_p = np.mean(X[mask_p], axis=0)
    mean_h = np.mean(X[mask_h], axis=0)
    
    motif = mean_p - mean_h
    return motif


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_disease_latent_space(
    disease: str,
    dataset_paths: list[str],
    cell_type_filter: list[str] | None = None,
    use_scvi: bool = False,
    backend: LatentBackend | None = None,
    _adata_override: Any | None = None,
    use_cache: bool = True,
) -> dict:
    """
    Build a latent embedding from disease-matched sc-RNA h5ad files.

    Args:
        disease:          Short disease key, e.g. "IBD", "CAD".
        dataset_paths:    Paths to h5ad files.  Ignored when _adata_override is set.
        cell_type_filter: If provided, restrict to these cell_type values before embedding.
        use_scvi:         Request ScVI backend (Phase 2, GPU).
        backend:          Explicit backend override — used for testing.
        _adata_override:  Pre-built AnnData; bypasses disk load (testing).

    Returns dict with keys:
        adata             — processed AnnData (obsm contains X_pca, X_diffmap; obs has dpt_pseudotime)
        latent_matrix     — np.ndarray  (n_cells × n_latent)
        cell_metadata     — pd.DataFrame (adata.obs)
        gene_metadata     — pd.DataFrame (adata.var)
        neighbors_graph   — {"connectivities": sparse_matrix}
        pseudotime        — np.ndarray | None
        provenance        — dict with backend, disease, cell_type_filter, n_cells, n_genes
        integration_warnings — list[str]
        backend           — str  (backend name)
        disease           — str

    On error (missing file, import failure) returns:
        {"error": str, "disease": disease, "adata": None, ...}
    """
    import numpy as np

    warnings_list: list[str] = []
    selected_backend = backend or get_backend(use_scvi=use_scvi)

    # --- Cache check (skip when _adata_override supplied) --------------------
    if use_cache and _adata_override is None and dataset_paths:
        h5ad_cache, meta_cache = _cache_paths(
            disease, dataset_paths, selected_backend.name, cell_type_filter
        )
        if _is_cache_valid(h5ad_cache, meta_cache, dataset_paths):
            try:
                import anndata as _anndata
                adata = _anndata.read_h5ad(str(h5ad_cache))
                warnings_list.append(f"[cache] loaded from {h5ad_cache.name}")
                latent_key = "X_diffmap" if "X_diffmap" in adata.obsm else "X_pca"
                latent_matrix = adata.obsm.get(latent_key)
                pseudotime_arr = adata.obs.get("dpt_pseudotime")
                pseudotime_arr = np.asarray(pseudotime_arr, dtype=float) if pseudotime_arr is not None else None
                
                # Phase Z7: Motif
                motif = extract_disease_transition_motif(adata)

                return {
                    "adata":               adata,
                    "latent_matrix":       latent_matrix,
                    "cell_metadata":       adata.obs,
                    "gene_metadata":       adata.var,
                    "neighbors_graph":     {"connectivities": adata.obsp.get("connectivities")},
                    "pseudotime":          pseudotime_arr,
                    "disease_transition_motif": motif,
                    "provenance": {
                        "disease":          disease,
                        "backend":          selected_backend.name,
                        "n_cells":          adata.n_obs,
                        "n_genes":          adata.n_vars,
                        "cell_type_filter": cell_type_filter,
                        "dataset_paths":    dataset_paths,
                        "from_cache":       True,
                        "cache_file":       str(h5ad_cache),
                    },
                    "integration_warnings": warnings_list,
                    "backend":             selected_backend.name,
                    "disease":             disease,
                }
            except Exception as cache_exc:
                warnings_list.append(f"[cache] load failed ({cache_exc}), recomputing")

    # --- Load data -----------------------------------------------------------
    try:
        if _adata_override is not None:
            adata = _adata_override.copy()
        else:
            adata = _load_and_concat_h5ads(dataset_paths)
    except (FileNotFoundError, ImportError) as exc:
        return {
            "error":               str(exc),
            "disease":             disease,
            "adata":               None,
            "latent_matrix":       None,
            "cell_metadata":       None,
            "gene_metadata":       None,
            "neighbors_graph":     {},
            "pseudotime":          None,
            "provenance":          {"disease": disease, "backend": selected_backend.name},
            "integration_warnings": [str(exc)],
            "backend":             selected_backend.name,
        }

    # --- Cell type filter ----------------------------------------------------
    if cell_type_filter:
        n_before = adata.n_obs
        adata = _filter_cell_types(adata, cell_type_filter)
        if adata.n_obs < n_before:
            warnings_list.append(
                f"Cell type filter retained {adata.n_obs}/{n_before} cells "
                f"(filter: {cell_type_filter})"
            )
        if adata.n_obs == 0:
            return {
                "error":    f"No cells remain after cell_type_filter={cell_type_filter}",
                "disease":  disease,
                "adata":    None,
                "latent_matrix": None,
                "cell_metadata": None,
                "gene_metadata": None,
                "neighbors_graph": {},
                "pseudotime": None,
                "provenance": {"disease": disease, "backend": selected_backend.name},
                "integration_warnings": warnings_list,
                "backend": selected_backend.name,
            }

    # --- Preprocess ----------------------------------------------------------
    adata = _preprocess_for_latent(adata, warnings_list)

    # --- Embed ---------------------------------------------------------------
    try:
        backend_result = selected_backend.fit_transform(adata)
    except (ImportError, NotImplementedError) as exc:
        warnings_list.append(f"{selected_backend.name} failed ({exc}); falling back to _NumpyPCABackend")
        fallback = _NumpyPCABackend()
        backend_result = fallback.fit_transform(adata)
        selected_backend = fallback

    # --- Extract outputs -----------------------------------------------------
    latent_key    = backend_result.get("latent_key", "X_diffmap")
    pseudo_key    = backend_result.get("pseudotime_key", "dpt_pseudotime")

    latent_matrix = adata.obsm.get(latent_key)
    if latent_matrix is None and "X_pca" in adata.obsm:
        latent_matrix = adata.obsm["X_pca"]
        warnings_list.append(f"latent_key '{latent_key}' missing; using X_pca as fallback")

    pseudotime = adata.obs.get(pseudo_key)
    if pseudotime is not None:
        pseudotime = np.asarray(pseudotime, dtype=float)

    provenance = {
        "disease":          disease,
        "backend":          selected_backend.name,
        "n_cells":          adata.n_obs,
        "n_genes":          adata.n_vars,
        "cell_type_filter": cell_type_filter,
        "dataset_paths":    dataset_paths if _adata_override is None else ["_adata_override"],
        "from_cache":       False,
    }

    # --- Save cache (skip when _adata_override or use_cache=False) -----------
    if use_cache and _adata_override is None and dataset_paths:
        h5ad_cache, meta_cache = _cache_paths(
            disease, dataset_paths, selected_backend.name, cell_type_filter
        )
        _save_latent_cache(adata, h5ad_cache, meta_cache, dataset_paths, selected_backend.name)
        provenance["cache_file"] = str(h5ad_cache)

    return {
        "adata":               adata,
        "latent_matrix":       latent_matrix,
        "cell_metadata":       adata.obs,
        "gene_metadata":       adata.var,
        "neighbors_graph":     {"connectivities": adata.obsp.get("connectivities")},
        "pseudotime":          pseudotime,
        "disease_transition_motif": extract_disease_transition_motif(adata),
        "provenance":          provenance,
        "integration_warnings": warnings_list,
        "backend":             selected_backend.name,
        "disease":             disease,
    }


# ---------------------------------------------------------------------------
# Multi-cell-type API (Phase A)
# ---------------------------------------------------------------------------

def build_multi_celltype_latent_space(
    disease: str,
    cell_type_h5ad_map: dict[str, str],
    use_scvi: bool = False,
    backend: LatentBackend | None = None,
) -> dict[str, dict]:
    """
    Build separate latent spaces per cell type.

    Each cell type gets its own independent latent embedding — this preserves
    context-specific gene programs and enables cross-context sign-flip detection.

    Args:
        disease:             Short disease key (IBD, CAD, …).
        cell_type_h5ad_map:  {cell_type: h5ad_path} — one h5ad per cell type.
        use_scvi:            Use ScVI backend (Phase 2).
        backend:             Explicit backend override for testing.

    Returns:
        {cell_type: build_disease_latent_space result dict}
        Results keyed by cell_type.  Failed cell types contain an "error" key.
    """
    results: dict[str, dict] = {}
    for cell_type, h5ad_path in cell_type_h5ad_map.items():
        results[cell_type] = build_disease_latent_space(
            disease=disease,
            dataset_paths=[h5ad_path],
            cell_type_filter=None,   # h5ad already contains only this cell type
            use_scvi=use_scvi,
            backend=backend,
        )
        results[cell_type]["cell_type"] = cell_type
    return results


def merge_multi_celltype_adata(
    latent_results: dict[str, dict],
    disease: str,
) -> dict:
    """
    Concatenate processed AnnDatas from build_multi_celltype_latent_space.

    Adds obs column "cell_type_source" so downstream code can split back by
    cell type.  gene_metadata is taken from the first successful result
    (assumes same gene universe after HVG selection — use only as a convenience;
    per-cell-type latent spaces remain in latent_results).

    Args:
        latent_results:  Output of build_multi_celltype_latent_space.
        disease:         Short disease key.

    Returns:
        {
            "adata":          concatenated AnnData (obs has "cell_type_source"),
            "n_cells_total":  int,
            "cell_types":     list[str],
            "warnings":       list[str],
        }
    """
    try:
        import anndata
    except ImportError as exc:
        return {"error": str(exc), "adata": None, "n_cells_total": 0,
                "cell_types": [], "warnings": [str(exc)]}

    adatas = []
    warnings: list[str] = []
    cell_types_ok: list[str] = []

    for cell_type, result in latent_results.items():
        if result.get("error") or result.get("adata") is None:
            warnings.append(
                f"Skipping {cell_type}: {result.get('error', 'adata is None')}"
            )
            continue
        adata = result["adata"].copy()
        adata.obs["cell_type_source"] = cell_type
        adatas.append(adata)
        cell_types_ok.append(cell_type)

    if not adatas:
        return {
            "error": "No valid AnnDatas to merge",
            "adata": None,
            "n_cells_total": 0,
            "cell_types": [],
            "warnings": warnings,
        }

    merged = anndata.concat(adatas, merge="same", index_unique="-")
    return {
        "adata":         merged,
        "n_cells_total": merged.n_obs,
        "cell_types":    cell_types_ok,
        "warnings":      warnings,
    }
