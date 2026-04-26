"""
replogle_parser.py — Disease/dataset-aware Perturb-seq β extractor.

Reads a Replogle 2022 pseudo-bulk h5ad and computes quantitative β estimates
for the Ota framework:

  β_{gene→program} = pseudobulk_log2FC(gene KO) projected onto program loadings

Supported datasets (keyed by scperturb_dataset in DISEASE_CELL_TYPE_MAP):
  replogle_2022_rpe1  — RPE1 retinal epithelial, 2393 knockouts  (AMD)
  replogle_2022_k562  — K562 CML, 9866 knockouts                 (generic)

The h5ad pseudo-bulk format (Replogle 2022):
  obs_names : perturbed gene symbols (one row per guide/perturbation)
  var_names : output gene symbols measured after perturbation
  X         : normalized expression matrix (log1p CPM or similar)

Algorithm:
  1. Identify non-targeting (control) observations.
  2. Compute control mean expression per output gene.
  3. For each perturbed gene KO:
       log2FC[gene_KO, output_gene] = mean_KO[output_gene] − control_mean[output_gene]
       SE[gene_KO, output_gene]   = pooled_SE estimated from within-KO variance
  4. Project log2FC vector onto each program's gene loadings:
       β_{gene_KO → program} = Σ_g (log2FC[gene_KO, g] × loading[program, g])
  5. Return dict keyed by perturbed gene.

Caching:
  Results cached to data/perturbseq/{dataset_id}/beta_cache_{programs_hash}.json.
  Cache is automatically invalidated when program gene sets change.

Usage:
  from pipelines.replogle_parser import load_replogle_betas
  betas = load_replogle_betas(
      program_gene_sets=my_programs,
      dataset_id="replogle_2022_rpe1",
  )
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import sys
import urllib.request
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent.parent
_PERTURBSEQ_DIR = _ROOT / "data" / "perturbseq"

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

# ---------------------------------------------------------------------------
# Dataset registry — maps scperturb_dataset → h5ad access info
# Keeps replogle_parser self-contained (no MCP import needed).
# ---------------------------------------------------------------------------
_DATASET_H5AD_REGISTRY: dict[str, dict] = {
    "replogle_2022_rpe1": {
        "figshare_file_id": 35775512,
        "filename":         "rpe1_normalized_bulk_01.h5ad",
        "size_gb":          0.10,
    },
    "replogle_2022_k562": {
        "figshare_file_id": 35773217,
        "filename":         "K562_gwps_normalized_bulk_01.h5ad",
        "size_gb":          0.37,
    },
}

# Datasets whose betas come from pre-computed signatures (not h5ad matrices).
# Maps dataset_id → path to signatures.json.gz relative to _PERTURBSEQ_DIR.
_DATASET_SIGNATURES_REGISTRY: dict[str, Path] = {
    "Schnitzler_GSE210681": _ROOT / "data" / "perturbseq" / "schnitzler_cad_vascular" / "signatures.json.gz",
}

# Keep module-level path for backward compat (old callers that don't pass dataset_id)
_H5AD_PATH = _ROOT / "data" / "replogle_2022_k562_essential.h5ad"


def _get_h5ad_path(dataset_id: str) -> Path:
    """Resolve the local h5ad path for a dataset_id."""
    reg = _DATASET_H5AD_REGISTRY.get(dataset_id)
    if reg:
        return _PERTURBSEQ_DIR / dataset_id / reg["filename"]
    # Fallback: legacy K562 path — but only for h5ad-backed datasets.
    # Signatures-backed datasets (e.g. Schnitzler) should never fall through here.
    return _H5AD_PATH


def _load_from_signatures(
    signatures_path: Path,
    program_gene_sets: dict[str, list[str]],
) -> dict[str, dict]:
    """
    Compute beta projections from a pre-computed signatures.json.gz file.

    Signatures format: {pert_gene: {output_gene: mean_log2fc}}
    Uses the same uniform-loading projection as _project_onto_programs:
        β_{gene→prog} = Σ_g (log2fc[g] × 1/√n)  for g in program gene set

    Returns same format as load_replogle_betas:
        {perturbed_gene: {"programs": {program_name: {beta, se, ci_lower, ci_upper}}}}
    """
    import gzip as _gz

    with _gz.open(str(signatures_path), "rt") as fh:
        signatures: dict[str, dict[str, float]] = json.load(fh)

    result: dict[str, dict] = {}
    for pert_gene, fc_map in signatures.items():
        prog_betas: dict[str, dict] = {}
        for prog_name, gene_list in program_gene_sets.items():
            overlap = [g for g in gene_list if g in fc_map]
            if not overlap:
                continue
            n = len(overlap)
            loading = 1.0 / math.sqrt(n)
            beta_val = sum(fc_map[g] * loading for g in overlap)
            prog_betas[prog_name] = {
                "beta":     round(beta_val, 6),
                "se":       0.0,
                "ci_lower": round(beta_val, 6),
                "ci_upper": round(beta_val, 6),
            }
        if prog_betas:
            result[pert_gene] = {"programs": prog_betas}

    return result


def _get_cache_path(
    dataset_id: str,
    program_gene_sets: dict[str, list[str]],
    h5ad_path: "Path | None" = None,
) -> Path:
    """
    Compute a cache path keyed by dataset_id, actual program gene content, and
    h5ad source path. Changing any gene in any program OR switching the source
    h5ad will produce a new hash and force recomputation.
    """
    parts = []
    for k, v in sorted(program_gene_sets.items()):
        parts.append(f"{k}:{','.join(sorted(v))}")
    if h5ad_path is not None:
        parts.append(f"h5ad:{Path(h5ad_path).resolve()}")
    programs_hash = hashlib.md5("|".join(parts).encode()).hexdigest()[:12]
    cache_dir = _PERTURBSEQ_DIR / dataset_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"beta_cache_{programs_hash}.json"


def _download_h5ad(dataset_id: str) -> Path:
    """
    Download the bulk h5ad for a dataset from figshare if not already present.
    Returns the local path.
    """
    reg = _DATASET_H5AD_REGISTRY.get(dataset_id)
    if not reg:
        raise ValueError(f"Unknown dataset_id: {dataset_id!r}. "
                         f"Known: {list(_DATASET_H5AD_REGISTRY)}")

    dest_dir = _PERTURBSEQ_DIR / dataset_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / reg["filename"]

    if dest.exists():
        return dest

    file_id = reg["figshare_file_id"]
    # Use ndownloader.figshare.com — the figshare.com/ndownloader/... alias can
    # return AWS WAF challenge pages (HTTP 202 + empty body) to non-browser clients.
    url = f"https://ndownloader.figshare.com/files/{file_id}"
    size_gb = reg["size_gb"]
    logger.info("Downloading %s h5ad (%.2f GB) from figshare ...", dataset_id, size_gb)
    logger.info("  URL: %s", url)
    logger.info("  Destination: %s", dest)

    try:
        urllib.request.urlretrieve(url, str(dest))
        logger.info("Download complete: %s", dest)
    except Exception as exc:
        if dest.exists():
            dest.unlink()
        raise FileNotFoundError(
            f"Failed to download {dataset_id} h5ad from {url}: {exc}\n"
            f"Manual download:\n"
            f"  wget '{url}' -O '{dest}'"
        ) from exc

    return dest


def _is_control(obs_name: str) -> bool:
    """Return True if this observation is a non-targeting control."""
    base = obs_name.lower().split("_")[0]
    return base in _CONTROL_LABELS or obs_name.lower() in _CONTROL_LABELS


def _is_control_gene_label(label: str) -> bool:
    """True if an explicit obs['gene'] (or parsed symbol) names a control perturbation."""
    low = label.lower()
    if low in _CONTROL_LABELS:
        return True
    # e.g. "10755_non-targeting_..." stored in a gene column as "non-targeting"
    toks = low.split("_")
    return any(t in _CONTROL_LABELS for t in toks)


def _infer_pert_gene_from_obs_name(obs_name: str, meta_gene: str) -> str:
    """
    Resolve a perturbation gene symbol for pseudo-bulk rows.

    Replogle RPE1 obs_names look like:
        ``{guide_id}_{GENE}_{phase}_ENSG...``
    When no ``adata.obs['gene']`` column exists, callers previously fell back to
    the full obs string, which breaks control detection and per-gene pooling.
    """
    if meta_gene and meta_gene != obs_name:
        return meta_gene
    parts = str(obs_name).split("_")
    if len(parts) >= 2 and parts[1]:
        return parts[1]
    return obs_name


def _safe_int_n_cells(value: Any) -> int:
    """Coerce obs cell-count columns to int; NaN/invalid → 1."""
    if value is None:
        return 1
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return 1
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return 1


def _obs_row_is_control(obs_name: str, meta: dict[str, Any]) -> bool:
    if meta.get("core_control") is True:
        return True
    if _is_control(obs_name):
        return True
    gene_lbl = str(meta.get("gene", ""))
    if gene_lbl and gene_lbl != obs_name and _is_control_gene_label(gene_lbl):
        return True
    return False


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
    raw_var_names = list(adata.var_names)
    # Prefer gene symbols over Ensembl IDs (programs use symbols)
    if raw_var_names and raw_var_names[0].startswith("ENSG") and "gene_name" in adata.var.columns:
        var_names = list(adata.var["gene_name"].astype(str).replace("nan", ""))
    else:
        var_names = raw_var_names

    # Extract obs metadata
    obs_meta: dict[str, dict] = {}
    obs_df = adata.obs
    for obs_name in obs_names:
        row = obs_df.loc[obs_name] if obs_name in obs_df.index else None
        meta: dict[str, Any] = {}
        if row is not None:
            if "core_control" in obs_df.columns:
                v = row.get("core_control")
                if isinstance(v, (bool, np.bool_)):
                    meta["core_control"] = bool(v)
                else:
                    meta["core_control"] = str(v).lower() in {"true", "1", "yes"}
            for col in ("n_cells", "ncells", "num_cells", "n_umis", "num_cells_filtered"):
                if col in obs_df.columns:
                    meta["n_cells"] = _safe_int_n_cells(row.get(col, 1))
                    break
            for col in ("gene", "target_gene", "perturbation", "guide_gene"):
                if col in obs_df.columns:
                    meta["gene"] = str(row.get(col, obs_name))
                    break
        if "gene" not in meta:
            meta["gene"] = obs_name
        if "n_cells" not in meta:
            meta["n_cells"] = 1
        # Normalize gene label when obs lacks an explicit gene column (RPE1 bulk)
        meta["gene"] = _infer_pert_gene_from_obs_name(obs_name, meta["gene"])
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

    ctrl_idx = [i for i, name in enumerate(obs_names) if _obs_row_is_control(name, obs_meta[name])]
    pert_idx = [i for i, name in enumerate(obs_names) if not _obs_row_is_control(name, obs_meta[name])]

    if not ctrl_idx:
        raise ValueError(
            "No non-targeting control observations found in h5ad. "
            f"First 5 obs names: {obs_names[:5]}"
        )

    ctrl_X = X[ctrl_idx, :]
    ctrl_mean = ctrl_X.mean(axis=0)
    ctrl_var = ctrl_X.var(axis=0, ddof=1) if ctrl_X.shape[0] > 1 else np.ones(n_var, dtype=np.float32)

    gene_to_rows: dict[str, list[int]] = {}
    for i in pert_idx:
        gene = obs_meta[obs_names[i]]["gene"]
        gene_to_rows.setdefault(gene, []).append(i)

    valid_genes = sorted(
        g for g, rows in gene_to_rows.items()
        if sum(obs_meta[obs_names[r]]["n_cells"] for r in rows) >= _MIN_CELLS
    )

    n_pert = len(valid_genes)
    log2fc = np.zeros((n_pert, n_var), dtype=np.float32)
    se_matrix = np.zeros((n_pert, n_var), dtype=np.float32)

    for j, gene in enumerate(valid_genes):
        rows = gene_to_rows[gene]
        pert_X = X[rows, :]
        pert_mean = pert_X.mean(axis=0)
        log2fc[j, :] = pert_mean - ctrl_mean

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
    Uses uniform unit loadings (1/sqrt(n)) within each program.

    Returns:
        {perturbed_gene: {"programs": {program_name: {beta, se, ci_lower, ci_upper}}}}
    """
    var_set = {g: i for i, g in enumerate(var_names)}

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
    dataset_id: str | None = None,
    h5ad_path: Path | None = None,
    cache_path: Path | None = None,
    force_recompute: bool = False,
    auto_download: bool = True,
) -> dict[str, dict]:
    """
    Main entry point: load/compute quantitative Perturb-seq β matrix.

    Args:
        program_gene_sets: {program_name: [gene_list]} — programs to project onto.
        dataset_id:     scperturb_dataset key from DISEASE_CELL_TYPE_MAP
                        (e.g. "replogle_2022_rpe1" for AMD, "replogle_2022_k562" generic).
                        When None, falls back to legacy K562 h5ad path.
        h5ad_path:      Override h5ad path (bypasses registry lookup).
        cache_path:     Override cache path (bypasses auto-keying).
        force_recompute: Skip cache and recompute from h5ad.
        auto_download:  If True and h5ad not present, download from figshare.

    Returns:
        {gene: {"programs": {program_name: {beta, se, ci_lower, ci_upper}}}}
        Compatible with estimate_beta_tier1(perturbseq_data=...) quantitative path.

    Raises:
        FileNotFoundError: if h5ad file not present and auto_download=False.
    """
    # --- Signatures-backed datasets (e.g. Schnitzler) skip h5ad entirely ---
    sig_path = _DATASET_SIGNATURES_REGISTRY.get(dataset_id or "")
    if sig_path is not None:
        if cache_path is None:
            cache_path = _get_cache_path(dataset_id, program_gene_sets, sig_path)
        if not force_recompute and cache_path.exists():
            with open(cache_path) as f:
                data = json.load(f)
            logger.info("Cache hit (signatures): %s (%d genes)", cache_path, len(data))
            return data
        if not sig_path.exists():
            raise FileNotFoundError(
                f"Signatures file not found: {sig_path}. "
                f"Run: python -m mcp_servers.perturbseq_server preprocess {dataset_id} <log2fc_path>"
            )
        logger.info("Loading signatures: %s", sig_path)
        betas = _load_from_signatures(sig_path, program_gene_sets)
        logger.info("β computed for %d perturbed genes (signatures)", len(betas))
        with open(cache_path, "w") as f:
            json.dump(betas, f)
        logger.info("Cache written: %s", cache_path)
        return betas

    # Resolve paths
    if h5ad_path is None:
        h5ad_path = _get_h5ad_path(dataset_id) if dataset_id else _H5AD_PATH
    if cache_path is None:
        if dataset_id:
            cache_path = _get_cache_path(dataset_id, program_gene_sets, h5ad_path)
        else:
            # Legacy: single flat cache (no program-set invalidation)
            cache_path = _ROOT / "data" / "replogle_cache.json"

    # --- Cache hit ---
    if not force_recompute and cache_path.exists():
        with open(cache_path) as f:
            data = json.load(f)
        logger.info("Cache hit: %s (%d genes)", cache_path, len(data))
        return data

    # --- Ensure h5ad is present ---
    if not h5ad_path.exists():
        if auto_download and dataset_id and dataset_id in _DATASET_H5AD_REGISTRY:
            h5ad_path = _download_h5ad(dataset_id)
        else:
            reg = _DATASET_H5AD_REGISTRY.get(dataset_id or "")
            if reg:
                file_id = reg["figshare_file_id"]
                dest = _PERTURBSEQ_DIR / dataset_id / reg["filename"]
                raise FileNotFoundError(
                    f"Replogle h5ad not found at {h5ad_path}.\n"
                    "Download with:\n"
                    f'  wget "https://figshare.com/ndownloader/files/{file_id}" \\\n'
                    f"       -O '{dest}'"
                )
            else:
                raise FileNotFoundError(
                    f"Replogle 2022 h5ad not found at {h5ad_path}.\n"
                    "Pass dataset_id='replogle_2022_rpe1' or 'replogle_2022_k562' "
                    "to enable auto-download."
                )

    # --- Load and compute ---
    logger.info("Loading h5ad: %s", h5ad_path)
    X, obs_names, var_names, obs_meta = _load_h5ad_matrix(h5ad_path)
    logger.info("Matrix: %s (%d obs × %d genes)", X.shape, len(obs_names), len(var_names))

    logger.info("Computing log2FC vs non-targeting controls ...")
    log2fc, se_matrix, pert_genes, _ctrl_mean = _compute_log2fc(X, obs_names, obs_meta)
    logger.info("%d perturbation genes with ≥%d cells", len(pert_genes), _MIN_CELLS)

    logger.info("Projecting onto %d programs ...", len(program_gene_sets))
    betas = _project_onto_programs(log2fc, se_matrix, pert_genes, var_names, program_gene_sets)
    logger.info("β computed for %d perturbed genes", len(betas))

    # --- Write cache ---
    with open(cache_path, "w") as f:
        json.dump(betas, f)
    logger.info("Cache written: %s", cache_path)

    return betas


