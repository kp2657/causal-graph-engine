"""
replogle_parser.py — Pseudo-bulk Perturb-seq β extractor (Replogle 2022 K562).

Reads the Replogle et al. 2022 K562 essential screen pseudo-bulk h5ad
(Figshare file 35770934, ~357 MB) and computes quantitative β estimates
for the Ota framework:

  β_{gene→program} = pseudobulk_log2FC(gene KO) projected onto program loadings

The h5ad pseudo-bulk format (Replogle 2022):
  obs_names : perturbed gene symbols (one row per guide/perturbation)
  var_names : output gene symbols measured after perturbation
  X         : normalized expression matrix (log1p CPM or similar)
  obs.cols  : 'gene', 'n_cells', 'guide_id', etc.

Algorithm:
  1. Identify non-targeting (control) observations by looking for obs where
     gene ∈ {"non-targeting", "AAVS1", "control", "safe_harbor"}.
  2. Compute control mean expression per output gene.
  3. For each perturbed gene KO:
       log2FC[gene_KO, output_gene] = mean_KO[output_gene] − control_mean[output_gene]
       SE[gene_KO, output_gene]   = pooled_SE estimated from within-KO variance
  4. Project log2FC vector onto each program's gene loadings:
       β_{gene_KO → program} = Σ_g (log2FC[gene_KO, g] × loading[program, g])
       SE_β = sqrt(Σ_g (SE[gene_KO, g]² × loading[program, g]²))
  5. Return dict keyed by perturbed gene:
       {gene: {"programs": {program_name: {beta, se, ci_lower, ci_upper}}}}

Caching:
  Parsed results are cached to data/replogle_cache.json on first run.
  Subsequent calls load from cache in <1s.

Usage:
  from pipelines.replogle_parser import load_replogle_betas
  perturbseq_data = load_replogle_betas(program_gene_sets=my_programs)
  # Then pass to estimate_beta_tier1(gene, program, perturbseq_data=...)
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent.parent
_H5AD_PATH = _ROOT / "data" / "replogle_2022_k562_essential.h5ad"
_CACHE_PATH = _ROOT / "data" / "replogle_cache.json"

# Non-targeting control obs names (checked case-insensitively)
_CONTROL_LABELS = frozenset({
    "non-targeting", "nontargeting", "non_targeting",
    "aavs1", "control", "safe_harbor", "safeharbor",
    "scramble", "scrambled",
})

# Minimum cells per perturbation to compute a reliable mean
_MIN_CELLS = 5

# 95% CI multiplier
_Z95 = 1.96


def _is_control(obs_name: str) -> bool:
    """Return True if this observation is a non-targeting control."""
    base = obs_name.lower().split("_")[0]
    return base in _CONTROL_LABELS or obs_name.lower() in _CONTROL_LABELS


def _load_h5ad_matrix(h5ad_path: Path) -> tuple[Any, list[str], list[str], dict]:
    """
    Load h5ad and return (X_dense, obs_names, var_names, obs_meta).

    Handles both sparse (scipy CSR/CSC) and dense X matrices.
    Returns X as a numpy float32 array (n_obs × n_var).
    """
    import anndata

    adata = anndata.read_h5ad(str(h5ad_path))

    # Densify if sparse
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)

    obs_names = list(adata.obs_names)
    var_names = list(adata.var_names)

    # Extract obs metadata
    obs_meta: dict[str, dict] = {}
    obs_df = adata.obs
    for obs_name in obs_names:
        row = obs_df.loc[obs_name] if obs_name in obs_df.index else None
        meta: dict[str, Any] = {}
        if row is not None:
            # n_cells for pooled pseudo-bulk rows
            for col in ("n_cells", "ncells", "num_cells", "n_umis"):
                if col in obs_df.columns:
                    meta["n_cells"] = int(row.get(col, 1))
                    break
            # gene target label
            for col in ("gene", "target_gene", "perturbation", "guide_gene"):
                if col in obs_df.columns:
                    meta["gene"] = str(row.get(col, obs_name))
                    break
        if "gene" not in meta:
            meta["gene"] = obs_name
        if "n_cells" not in meta:
            meta["n_cells"] = 1
        obs_meta[obs_name] = meta

    return X, obs_names, var_names, obs_meta


def _compute_log2fc(
    X: np.ndarray,
    obs_names: list[str],
    obs_meta: dict,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """
    Compute log2FC matrix relative to pooled non-targeting controls.

    Returns:
        log2fc      : (n_pert, n_var) float32 array
        se_matrix   : (n_pert, n_var) float32 standard error
        pert_genes  : list of perturbed gene symbols (length n_pert)
        ctrl_mean   : (n_var,) control mean vector
    """
    n_var = X.shape[1]

    # Partition indices into control and perturbation rows
    ctrl_idx = [i for i, name in enumerate(obs_names) if _is_control(obs_meta[name]["gene"])]
    pert_idx = [i for i, name in enumerate(obs_names) if not _is_control(obs_meta[name]["gene"])]

    if not ctrl_idx:
        raise ValueError(
            "No non-targeting control observations found in h5ad. "
            f"First 5 obs names: {obs_names[:5]}"
        )

    # Control statistics
    ctrl_X = X[ctrl_idx, :]                         # (n_ctrl, n_var)
    ctrl_mean = ctrl_X.mean(axis=0)                 # (n_var,)
    ctrl_var = ctrl_X.var(axis=0, ddof=1) if ctrl_X.shape[0] > 1 else np.ones(n_var, dtype=np.float32)

    # Group perturbation rows by gene target
    gene_to_rows: dict[str, list[int]] = {}
    for i in pert_idx:
        gene = obs_meta[obs_names[i]]["gene"]
        gene_to_rows.setdefault(gene, []).append(i)

    # Filter by min cells
    valid_genes = sorted(
        g for g, rows in gene_to_rows.items()
        if sum(obs_meta[obs_names[r]]["n_cells"] for r in rows) >= _MIN_CELLS
    )

    n_pert = len(valid_genes)
    log2fc = np.zeros((n_pert, n_var), dtype=np.float32)
    se_matrix = np.zeros((n_pert, n_var), dtype=np.float32)

    for j, gene in enumerate(valid_genes):
        rows = gene_to_rows[gene]
        pert_X = X[rows, :]                          # (n_guide, n_var)
        pert_mean = pert_X.mean(axis=0)              # (n_var,)
        log2fc[j, :] = pert_mean - ctrl_mean         # already log-scale

        # Pooled SE: sqrt((ctrl_var + pert_var) / n)
        n_pert_rows = pert_X.shape[0]
        pert_var = pert_X.var(axis=0, ddof=1) if n_pert_rows > 1 else ctrl_var
        n_ctrl = ctrl_X.shape[0]
        pooled_se = np.sqrt(
            ctrl_var / max(n_ctrl, 1) + pert_var / max(n_pert_rows, 1)
        )
        se_matrix[j, :] = pooled_se.astype(np.float32)

    return log2fc, se_matrix, valid_genes, ctrl_mean


def _project_onto_programs(
    log2fc: np.ndarray,
    se_matrix: np.ndarray,
    pert_genes: list[str],
    var_names: list[str],
    program_gene_sets: dict[str, list[str]],
) -> dict[str, dict]:
    """
    Project gene-level log2FC vectors onto program loadings.

    program_gene_sets: {program_name: [gene_symbol, ...]}
    Uses uniform unit loadings (1/sqrt(n)) within each program;
    weights can be refined once cNMF decomposition runs on the h5ad.

    Returns:
        {perturbed_gene: {"programs": {program_name: {beta, se, ci_lower, ci_upper}}}}
    """
    var_set = {g: i for i, g in enumerate(var_names)}

    # Pre-compute program indices and unit loadings
    prog_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for prog_name, gene_list in program_gene_sets.items():
        idx = [var_set[g] for g in gene_list if g in var_set]
        if not idx:
            continue
        idxarr = np.array(idx, dtype=np.int32)
        n = len(idxarr)
        loadings = np.ones(n, dtype=np.float32) / math.sqrt(n)
        prog_data[prog_name] = (idxarr, loadings)

    result: dict[str, dict] = {}
    for j, gene in enumerate(pert_genes):
        fc_vec = log2fc[j, :]
        se_vec = se_matrix[j, :]

        prog_betas: dict[str, dict] = {}
        for prog_name, (idxarr, loadings) in prog_data.items():
            beta_val = float(np.dot(fc_vec[idxarr], loadings))
            se_val = float(np.sqrt(np.dot(se_vec[idxarr] ** 2, loadings ** 2)))

            ci_lower = beta_val - _Z95 * se_val
            ci_upper = beta_val + _Z95 * se_val
            prog_betas[prog_name] = {
                "beta":     round(beta_val, 6),
                "se":       round(se_val, 6),
                "ci_lower": round(ci_lower, 6),
                "ci_upper": round(ci_upper, 6),
            }
        if prog_betas:
            result[gene] = {"programs": prog_betas}

    return result


def load_replogle_betas(
    program_gene_sets: dict[str, list[str]],
    h5ad_path: Path | None = None,
    cache_path: Path | None = None,
    force_recompute: bool = False,
) -> dict[str, dict]:
    """
    Main entry point: load/compute quantitative Perturb-seq β matrix.

    Args:
        program_gene_sets: {program_name: [gene_list]} — the programs to project onto.
            Use cnmf_programs.get_msigdb_hallmark_programs() or your cNMF programs.
        h5ad_path:   Path to the Replogle 2022 K562 essential h5ad.
                     Defaults to data/replogle_2022_k562_essential.h5ad.
        cache_path:  Path to JSON cache. Defaults to data/replogle_cache.json.
        force_recompute: Skip cache and recompute from h5ad.

    Returns:
        {gene: {"programs": {program_name: {beta, se, ci_lower, ci_upper}}}}
        Compatible with estimate_beta_tier1(perturbseq_data=...) quantitative path.

    Raises:
        FileNotFoundError: if h5ad file not present. Download with:
            wget "https://figshare.com/ndownloader/files/35770934" \\
                 -O data/replogle_2022_k562_essential.h5ad
    """
    h5ad_path = h5ad_path or _H5AD_PATH
    cache_path = cache_path or _CACHE_PATH

    # --- Cache hit ---
    if not force_recompute and cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    # --- Verify h5ad present ---
    if not h5ad_path.exists():
        raise FileNotFoundError(
            f"Replogle 2022 h5ad not found at {h5ad_path}.\n"
            "Download with:\n"
            '  wget "https://figshare.com/ndownloader/files/35770934" \\\n'
            "       -O data/replogle_2022_k562_essential.h5ad"
        )

    # --- Load and compute ---
    print(f"[replogle_parser] Loading h5ad from {h5ad_path} ...")
    X, obs_names, var_names, obs_meta = _load_h5ad_matrix(h5ad_path)
    print(f"[replogle_parser] Matrix shape: {X.shape} ({len(obs_names)} obs × {len(var_names)} genes)")

    print("[replogle_parser] Computing log2FC relative to non-targeting controls ...")
    log2fc, se_matrix, pert_genes, _ctrl_mean = _compute_log2fc(X, obs_names, obs_meta)
    print(f"[replogle_parser] {len(pert_genes)} perturbation genes with ≥{_MIN_CELLS} cells")

    print(f"[replogle_parser] Projecting onto {len(program_gene_sets)} programs ...")
    betas = _project_onto_programs(log2fc, se_matrix, pert_genes, var_names, program_gene_sets)
    print(f"[replogle_parser] β computed for {len(betas)} perturbed genes")

    # --- Write cache ---
    with open(cache_path, "w") as f:
        json.dump(betas, f)
    print(f"[replogle_parser] Cache written to {cache_path}")

    return betas


def invalidate_cache(cache_path: Path | None = None) -> None:
    """Remove cache file so next call recomputes from h5ad."""
    cp = cache_path or _CACHE_PATH
    if cp.exists():
        cp.unlink()
        print(f"[replogle_parser] Cache invalidated: {cp}")
