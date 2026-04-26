"""
pipelines/state_space/program_precedence.py

Pseudotime-based program precedence test for causal genomics pipeline.

Determines whether each NMF cellular program activates *before* or *after*
the normal→disease branch point in pseudotime.

  - Pre-branch programs  → candidate causal mediators (label="mediator")
  - Post-branch programs → likely downstream consequences (label="consequence")
  - Ambiguous programs   → flat activity or mixed signal (label="ambiguous")

The precedence label is then used to optionally discount γ_{P→trait} for
consequence programs, preventing downstream artefacts from inflating the
causal γ of the OTA formula.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Discount applied to γ of each label
_PRECEDENCE_DISCOUNTS: dict[str, float] = {
    "mediator":    1.0,
    "ambiguous":   1.0,
    "consequence": 0.5,
}


# ---------------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------------

def get_precedence_discount(label: str) -> float:
    """Return γ discount for a program given its precedence label.

    mediator   → 1.0  (full weight — causal candidate)
    ambiguous  → 1.0  (full weight — insufficient evidence to penalise)
    consequence→ 0.5  (half weight — downstream effect, not cause)
    """
    return _PRECEDENCE_DISCOUNTS.get(label, 1.0)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_pseudotime(adata: Any) -> str:
    """Ensure adata.obs has a pseudotime column; compute via scanpy DPT if absent.

    Returns the name of the pseudotime column.
    Raises ImportError if scanpy unavailable and pseudotime is missing.
    """
    for pt_col in ("pseudotime", "dpt_pseudotime"):
        if pt_col in adata.obs.columns:
            return pt_col

    # Must compute it — requires scanpy
    import scanpy as sc
    import numpy as np

    # Determine root cell: use disease-enriched cell if possible
    root_idx = _find_disease_root(adata)

    sc.pp.neighbors(adata)
    sc.tl.diffmap(adata)
    adata.uns["iroot"] = root_idx
    sc.tl.dpt(adata)
    return "dpt_pseudotime"


def _find_disease_root(adata: Any) -> int:
    """Find index of the most disease-enriched cell to use as DPT root.

    Strategy:
    1. If obs has a 'disease' column, use any non-normal cell (first found).
    2. Otherwise, fall back to the cell with minimum PC1 value
       (standard latent_model.py convention).
    """
    import numpy as np

    if "disease" in adata.obs.columns:
        labels = adata.obs["disease"].astype(str).values
        disease_mask = (labels != "normal") & (~(labels == ""))
        if disease_mask.any():
            return int(np.where(disease_mask)[0][0])

    # Fallback: min PC1 (same as PCADiffusionBackend)
    if "X_pca" in adata.obsm:
        return int(np.argmin(adata.obsm["X_pca"][:, 0]))
    return 0


def _get_pca_coords(adata: Any) -> "np.ndarray":
    """Return PCA coordinates, computing them if needed."""
    import numpy as np

    if "X_pca" in adata.obsm:
        return np.asarray(adata.obsm["X_pca"])

    # Compute minimal PCA
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=float)
    n_comps = min(10, adata.n_obs - 1, adata.n_vars - 1)
    X_c = X - X.mean(axis=0)
    U, S, _ = np.linalg.svd(X_c, full_matrices=False)
    coords = (U[:, :n_comps] * S[:n_comps]).astype(float)
    adata.obsm["X_pca"] = coords
    return coords


def _compute_branching_probability(
    adata: Any,
    pca_coords: "np.ndarray",
) -> "np.ndarray | None":
    """Compute BP per cell from normal/disease centroids.

    BP_i = 1 − |d_normal_i − d_disease_i| / (d_normal_i + d_disease_i + ε)

    Returns None if 'disease' column absent (caller uses median pseudotime instead).
    """
    import numpy as np

    if "disease" not in adata.obs.columns:
        return None

    labels = adata.obs["disease"].astype(str).values
    normal_mask  = labels == "normal"
    disease_mask = ~normal_mask

    if not normal_mask.any() or not disease_mask.any():
        return None

    normal_centroid  = pca_coords[normal_mask].mean(axis=0)
    disease_centroid = pca_coords[disease_mask].mean(axis=0)

    d_n = np.linalg.norm(pca_coords - normal_centroid, axis=1)
    d_d = np.linalg.norm(pca_coords - disease_centroid, axis=1)
    bp = 1.0 - np.abs(d_n - d_d) / (d_n + d_d + 1e-8)
    return bp


def _ensure_normalized(adata: Any) -> Any:
    """Return AnnData with log-normalised expression in adata.X.

    Checks adata.uns for prior normalisation flags to avoid double-normalising.
    Returns a copy — does NOT mutate the input.
    """
    import numpy as np

    # Check if already normalised
    uns_keys = set(adata.uns.keys())
    already_norm = (
        "log1p" in uns_keys
        or adata.uns.get("pp_log1p", False)
        or adata.uns.get("normalized", False)
    )
    if already_norm:
        return adata

    try:
        import scanpy as sc
        adata2 = adata.copy()
        sc.pp.normalize_total(adata2, target_sum=1e4)
        sc.pp.log1p(adata2)
        return adata2
    except Exception:
        # Manual fallback
        adata2 = adata.copy()
        X = adata2.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.asarray(X, dtype=float)
        row_sums = X.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        X = np.log1p(X / row_sums * 1e4)
        from scipy.sparse import csr_matrix as _csr
        adata2.X = _csr(X)
        return adata2


def _pearson_r(x: "np.ndarray", y: "np.ndarray") -> float:
    """Safe Pearson r; returns 0.0 if fewer than 3 observations or zero variance."""
    import numpy as np

    if len(x) < 3:
        return 0.0
    xv = x - x.mean()
    yv = y - y.mean()
    denom = np.sqrt((xv ** 2).sum() * (yv ** 2).sum())
    if denom < 1e-12:
        return 0.0
    return float(np.dot(xv, yv) / denom)


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def compute_program_precedence(
    h5ad_path: "str | Path",
    program_loadings: "dict[str, dict[str, float]]",
    n_bootstrap: int = 200,
    seed: int = 42,
    disease_key: str | None = None,
) -> "dict[str, dict]":
    """
    For each program, test whether it activates before or after the branch point.

    Args:
        h5ad_path:        Path to h5ad file.
        program_loadings: {program_id → {gene → loading}}.
        n_bootstrap:      Number of bootstrap resamples for confidence estimation.
        seed:             Random seed for reproducibility.
        disease_key:      Optional disease key; if provided, result is cached to
                          data/ldsc/results/{disease_key}_program_precedence.json.

    Returns:
        dict[program_id → {
            "label":            "mediator" | "consequence" | "ambiguous",
            "pre_branch_r":     float,
            "post_branch_r":    float,
            "branch_pseudotime": float,
            "n_cells_pre":      int,
            "n_cells_post":     int,
            "confidence":       float,
        }]
    """
    import numpy as np
    import anndata

    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # 1. Load h5ad
    # ------------------------------------------------------------------
    adata = anndata.read_h5ad(str(h5ad_path))

    # ------------------------------------------------------------------
    # 2. Normalise expression
    # ------------------------------------------------------------------
    adata = _ensure_normalized(adata)

    # ------------------------------------------------------------------
    # 3. Ensure pseudotime is present
    # ------------------------------------------------------------------
    try:
        pt_col = _ensure_pseudotime(adata)
    except Exception as exc:
        log.warning("compute_program_precedence: pseudotime computation failed (%s). "
                    "Using rank-order proxy.", exc)
        # Fallback: use gene-expression PC1 rank as pseudotime proxy
        coords = _get_pca_coords(adata)
        pt = np.argsort(coords[:, 0]).argsort().astype(float)
        pt /= max(pt.max(), 1.0)
        adata.obs["_fallback_pseudotime"] = pt
        pt_col = "_fallback_pseudotime"

    pseudotime = np.asarray(adata.obs[pt_col], dtype=float)
    # Replace NaN pseudotime with median (scanpy sometimes sets NaN for disconnected cells)
    nan_mask = np.isnan(pseudotime)
    if nan_mask.any():
        pseudotime[nan_mask] = np.nanmedian(pseudotime)
        adata.obs[pt_col] = pseudotime

    # ------------------------------------------------------------------
    # 4. Branching probability → branch point pseudotime
    # ------------------------------------------------------------------
    pca_coords = _get_pca_coords(adata)
    bp = _compute_branching_probability(adata, pca_coords)

    if bp is not None:
        branch_cell_idx = int(np.argmax(bp))
        branch_pseudotime = float(pseudotime[branch_cell_idx])
    else:
        # No disease column → use median pseudotime as branch point
        branch_pseudotime = float(np.median(pseudotime))
        log.info("compute_program_precedence: no 'disease' column found; "
                 "using median pseudotime (%.3f) as branch point.", branch_pseudotime)

    # ------------------------------------------------------------------
    # 5. Pre / post branch masks
    # ------------------------------------------------------------------
    pre_mask  = pseudotime <= branch_pseudotime
    post_mask = pseudotime >  branch_pseudotime
    pt_pre  = pseudotime[pre_mask]
    pt_post = pseudotime[post_mask]
    n_pre  = int(pre_mask.sum())
    n_post = int(post_mask.sum())

    # ------------------------------------------------------------------
    # 6. Gene names from adata (try feature_name column first)
    # ------------------------------------------------------------------
    if "feature_name" in adata.var.columns:
        var_names = list(adata.var["feature_name"])
    else:
        var_names = list(adata.var_names)
    var_name_index = {g: i for i, g in enumerate(var_names)}

    # Dense expression matrix for vectorised activity computation
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=float)

    # ------------------------------------------------------------------
    # 7. Per-program computation
    # ------------------------------------------------------------------
    results: dict[str, dict] = {}

    for prog_id, loadings in program_loadings.items():
        # Intersect program genes with adata genes
        shared_genes = [(g, w) for g, w in loadings.items() if g in var_name_index]

        if not shared_genes:
            # No overlapping genes — return ambiguous without crash
            results[prog_id] = {
                "label":              "ambiguous",
                "pre_branch_r":       0.0,
                "post_branch_r":      0.0,
                "branch_pseudotime":  branch_pseudotime,
                "n_cells_pre":        n_pre,
                "n_cells_post":       n_post,
                "confidence":         0.0,
                "warning":            "no_shared_genes",
            }
            continue

        # Compute per-cell program activity as weighted sum of gene expression
        gene_indices = [var_name_index[g] for g, _ in shared_genes]
        weights      = np.array([w for _, w in shared_genes], dtype=float)
        activity     = X[:, gene_indices] @ weights   # shape: (n_cells,)

        act_pre  = activity[pre_mask]
        act_post = activity[post_mask]

        pre_r  = _pearson_r(pt_pre,  act_pre)
        post_r = _pearson_r(pt_post, act_post)

        # ------------------------------------------------------------------
        # 8. Label logic
        # ------------------------------------------------------------------
        if pre_r > 0.15 and pre_r > post_r + 0.10:
            label = "mediator"
        elif post_r > 0.15 and post_r > pre_r + 0.10:
            label = "consequence"
        else:
            label = "ambiguous"

        # ------------------------------------------------------------------
        # 9. Bootstrap confidence
        # ------------------------------------------------------------------
        n_total = len(activity)
        boot_diffs: list[float] = []
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n_total, n_total)
            pt_b  = pseudotime[idx]
            act_b = activity[idx]
            pre_b  = pt_b <= branch_pseudotime
            post_b = ~pre_b
            if pre_b.sum() < 3 or post_b.sum() < 3:
                boot_diffs.append(pre_r - post_r)
                continue
            r_pre_b  = _pearson_r(pt_b[pre_b],  act_b[pre_b])
            r_post_b = _pearson_r(pt_b[post_b], act_b[post_b])
            boot_diffs.append(r_pre_b - r_post_b)

        sd = float(np.std(boot_diffs))
        delta = abs(pre_r - post_r)
        confidence = float(1.0 - min(1.0, 2.0 * sd / (delta + 1e-8)))

        results[prog_id] = {
            "label":              label,
            "pre_branch_r":       round(pre_r,  4),
            "post_branch_r":      round(post_r, 4),
            "branch_pseudotime":  round(branch_pseudotime, 4),
            "n_cells_pre":        n_pre,
            "n_cells_post":       n_post,
            "confidence":         round(confidence, 4),
        }

    # ------------------------------------------------------------------
    # 10. Optional caching
    # ------------------------------------------------------------------
    if disease_key:
        _results_dir = Path(__file__).parent.parent.parent / "data" / "ldsc" / "results"
        _results_dir.mkdir(parents=True, exist_ok=True)
        _cache_path = _results_dir / f"{disease_key}_program_precedence.json"
        try:
            _cache_path.write_text(json.dumps({"programs": results}, indent=2))
            log.info("compute_program_precedence: cached to %s", _cache_path)
        except Exception as exc:
            log.warning("compute_program_precedence: cache write failed (%s)", exc)

    return results
