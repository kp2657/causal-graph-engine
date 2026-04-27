"""
cnmf_runner.py — Run NMF on sc-RNA h5ad to discover gene expression programs.

Two computation modes:
  1. sklearn NMF (always available) — fast approximation, no extra deps
  2. cNMF package (preferred) — consensus NMF with stability selection

Output: data/cnmf_programs/{disease}_programs.json
Format compatible with get_programs_for_disease() in cnmf_programs.py.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

_CNMF_OUTPUT_ROOT = Path("./data/cnmf_programs")


def run_nmf_programs(
    h5ad_path: str,
    disease: str,
    n_programs: int = 20,
    n_top_genes: int = 50,
    min_cells: int = 200,
    force: bool = False,
) -> dict:
    """
    Run NMF on an h5ad to extract gene expression programs.

    Preprocessing:
      - Filter low-quality cells (< 200 genes detected)
      - Normalize to 10k counts per cell (scanpy normalize_total)
      - Log1p transform
      - Select top 2000 highly variable genes

    Factorization:
      - If `cnmf` package installed: consensus NMF (30 iterations, stability selection)
      - Otherwise: sklearn NMF (single run, fast)

    Output saved to data/cnmf_programs/{disease}_programs.json.

    Args:
        h5ad_path:   Path to h5ad file (from cellxgene_downloader)
        disease:     Short disease name, used for output filename
        n_programs:  Number of programs (NMF rank K)
        n_top_genes: Number of top genes to record per program
        min_cells:   Minimum cells required; returns error if below threshold
        force:       Recompute even if output exists

    Returns:
        {
            "programs": list[{program_id, gene_set, top_genes, gene_loadings}],
            "n_programs": int,
            "source": "cNMF" | "sklearn_NMF",
            "disease": str,
            "cell_type": str,
            "n_cells": int,
        }
    """
    output_path = _CNMF_OUTPUT_ROOT / f"{disease.upper()}_programs.json"
    if output_path.exists() and not force:
        return load_computed_programs(disease)

    if not Path(h5ad_path).exists():
        return {
            "programs": [], "n_programs": 0, "disease": disease,
            "source": "error", "note": f"h5ad not found: {h5ad_path}",
        }

    try:
        import anndata
        import numpy as np
        adata = anndata.read_h5ad(h5ad_path)
    except ImportError:
        return {
            "programs": [], "n_programs": 0, "disease": disease,
            "source": "error", "note": "Install: pip install anndata",
        }

    if adata.n_obs < min_cells:
        return {
            "programs": [], "n_programs": 0, "disease": disease,
            "source": "error",
            "note": f"Too few cells ({adata.n_obs} < {min_cells}) for reliable NMF",
        }

    # Preprocessing
    adata = _preprocess(adata)
    if adata is None:
        return {"programs": [], "n_programs": 0, "disease": disease,
                "source": "error", "note": "Preprocessing failed"}

    # Try cnmf package first, fall back to sklearn
    try:
        import cnmf  # noqa: F401
        programs_result = _run_cnmf_package(adata, disease, n_programs, n_top_genes)
    except ImportError:
        programs_result = _run_sklearn_nmf(adata, disease, n_programs, n_top_genes)

    # Save to disk
    _CNMF_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(programs_result, indent=2))

    return programs_result


def _preprocess(adata: Any) -> Any | None:
    """Normalize + HVG selection for NMF input."""
    try:
        import scanpy as sc
        import numpy as np

        # Filter low-quality cells
        sc.pp.filter_cells(adata, min_genes=200)
        sc.pp.filter_genes(adata, min_cells=10)

        # Normalize
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        # Highly variable genes — the NMF feature set
        sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat_v3")
        adata = adata[:, adata.var["highly_variable"]].copy()

        return adata
    except ImportError:
        # scanpy not available — use raw counts with basic normalization
        try:
            import numpy as np
            X = adata.X
            if hasattr(X, "toarray"):
                X = X.toarray()
            X = X.astype(float)
            # Library-size normalization
            row_sums = X.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1, row_sums)
            X = X / row_sums * 1e4
            # Log1p
            X = np.log1p(X)
            # Select top 2000 most variable genes
            variances = X.var(axis=0)
            top_idx = np.argsort(variances)[-2000:]
            adata = adata[:, top_idx].copy()
            adata.X = X[:, top_idx]
            return adata
        except Exception:
            return None
    except Exception:
        return None


def _select_k_elbow(X: "np.ndarray", k_min: int = 4, k_max: int = 20, n_iter: int = 50) -> int:
    """Select NMF k by reconstruction-error second-derivative elbow."""
    import numpy as np
    from sklearn.decomposition import NMF

    k_vals = list(range(k_min, min(k_max + 1, X.shape[0], X.shape[1] + 1)))
    if len(k_vals) < 3:
        return k_vals[0] if k_vals else k_min

    errors = []
    for k in k_vals:
        m = NMF(n_components=k, init="nndsvda", max_iter=n_iter, random_state=42)
        m.fit(X)
        errors.append(float(m.reconstruction_err_))

    err = np.array(errors)
    diff2 = np.diff(np.diff(err))
    knee_idx = int(np.argmax(diff2)) + 1
    knee_idx = max(0, min(knee_idx, len(k_vals) - 1))
    k_chosen = int(k_vals[knee_idx])
    print(f"[cnmf_runner] k elbow: searched {k_min}–{k_max}, selected k={k_chosen}", flush=True)
    return k_chosen


def _empiric_program_size(loading_vec: "np.ndarray", min_genes: int = 20, max_genes: int = 500) -> int:
    """
    Find the natural cut-off in a program's loading distribution.

    NMF loadings are heavy-tailed: a core set of genes have substantial weight,
    then there is a sharp drop to near-zero noise genes. We find that drop via
    the second-derivative elbow on the sorted (descending) cumulative-loading
    curve, then return the index as the empiric gene-set size.

    Bounds: [min_genes, max_genes] — never smaller than needed for GPS screens,
    never so large that noise genes dominate downstream enrichment tests.
    """
    import numpy as np

    pos = loading_vec[loading_vec > 0]
    if len(pos) <= min_genes:
        return min(len(pos), max_genes)

    sorted_loads = np.sort(pos)[::-1]
    cumsum = np.cumsum(sorted_loads)
    cumsum_norm = cumsum / cumsum[-1]   # 0→1 cumulative fraction of total loading

    n = len(cumsum_norm)
    if n < 3:
        return min(n, max_genes)

    # Second derivative of the cumulative curve — elbow = where marginal gain drops fastest
    diff2 = np.diff(np.diff(cumsum_norm))
    # We want the *negative* elbow (curve bends downward = diminishing returns)
    knee_idx = int(np.argmin(diff2)) + 2   # +2 to account for two diffs
    knee_idx = max(min_genes, min(knee_idx, max_genes, n))

    return int(knee_idx)


def _run_sklearn_nmf(adata: Any, disease: str, k: int, n_top: int) -> dict:
    """NMF via sklearn with empirical k and empirical per-program gene-set size."""
    import numpy as np
    try:
        from sklearn.decomposition import NMF
    except ImportError:
        return {
            "programs": [], "n_programs": 0, "disease": disease,
            "source": "error",
            "note": "Install scikit-learn: pip install scikit-learn",
        }

    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = X.astype(float)
    X = np.clip(X, 0, None)

    raw_var_names = list(adata.var_names)
    # Resolve Ensembl IDs → HGNC symbols at write time so the cache stores symbols.
    # Downstream beta/gamma matching uses symbols; storing Ensembl IDs causes 0 overlap.
    try:
        from pipelines.static_lookups import get_lookups as _get_lk
        _lk = _get_lk()
        gene_names = [
            (_lk.get_symbol_from_ensembl(g) or g) if g.startswith("ENSG") else g
            for g in raw_var_names
        ]
    except Exception:
        gene_names = raw_var_names

    # Empirical k selection — elbow on reconstruction error curve
    k_empiric = _select_k_elbow(X, k_min=4, k_max=k, n_iter=50)

    model = NMF(n_components=k_empiric, init="nndsvda", max_iter=500, random_state=42)
    W = model.fit_transform(X)   # cells × programs
    H = model.components_         # programs × genes

    programs = []
    for i in range(k_empiric):
        loading_vec = H[i]
        # Empiric cut-off: elbow on cumulative loading curve, bounded [100, 500]
        # min_genes=100 ensures GPS L1000 signatures have sufficient overlap (~978 landmark genes)
        n_genes = _empiric_program_size(loading_vec, min_genes=100, max_genes=500)
        top_idx = np.argsort(loading_vec)[::-1][:n_genes]
        top_genes = [gene_names[j] for j in top_idx if loading_vec[j] > 0]
        gene_loadings = {
            gene_names[j]: float(loading_vec[j])
            for j in top_idx if loading_vec[j] > 0
        }
        print(f"[cnmf_runner] program P{i+1:02d}: empiric size={n_genes} "
              f"(top gene: {top_genes[0] if top_genes else 'none'})", flush=True)
        programs.append({
            "program_id":    f"{disease.upper()}_NMF_P{i+1:02d}",
            "gene_set":      top_genes,
            "top_genes":     top_genes,
            "gene_loadings": gene_loadings,
            "n_genes":       len(top_genes),
            "cell_type":     getattr(adata, "_discovery_cell_type", "unknown"),
            "source":        "sklearn_NMF",
        })

    return {
        "programs":   programs,
        "n_programs": k_empiric,
        "k":          k_empiric,
        "disease":    disease,
        "source":     "sklearn_NMF",
        "n_cells":    adata.n_obs,
        "n_genes":    adata.n_vars,
        "note":       f"sklearn NMF, empiric k={k_empiric}, empiric gene-set size per program.",
    }


def _run_cnmf_package(adata: Any, disease: str, k: int, n_top: int) -> dict:
    """cNMF consensus NMF — preferred when package is installed."""
    import tempfile
    import os
    import numpy as np

    from cnmf import cNMF  # type: ignore

    gene_names = list(adata.var_names)
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Write h5ad for cnmf
        h5ad_tmp = os.path.join(tmpdir, "input.h5ad")
        adata.write_h5ad(h5ad_tmp)

        cnmf_obj = cNMF(output_dir=tmpdir, name=f"{disease}_cnmf")
        cnmf_obj.prepare(
            counts_fn=h5ad_tmp,
            components=[k],
            n_iter=30,
            seed=42,
            num_highvar_genes=2000,
        )
        cnmf_obj.factorize(worker_i=0, total_workers=1)
        cnmf_obj.combine(components=k, skip_missing_files=True)
        usage, spectra, top_genes_df = cnmf_obj.load_results(
            K=k, density_threshold=2.0
        )

    programs = []
    for i in range(k):
        loading_vec = spectra.iloc[i].values
        top_idx = np.argsort(loading_vec)[::-1][:n_top]
        top_genes = [gene_names[j] for j in top_idx if loading_vec[j] > 0]
        gene_loadings = {
            gene_names[j]: float(loading_vec[j])
            for j in top_idx if loading_vec[j] > 0
        }
        programs.append({
            "program_id":    f"{disease.upper()}_cNMF_P{i+1:02d}",
            "gene_set":      top_genes[:n_top],
            "top_genes":     top_genes[:n_top],
            "gene_loadings": gene_loadings,
            "cell_type":     "unknown",
            "source":        "cNMF",
        })

    return {
        "programs":   programs,
        "n_programs": k,
        "disease":    disease,
        "source":     "cNMF",
        "n_cells":    adata.n_obs,
        "n_genes":    adata.n_vars,
    }


def load_computed_programs(disease: str) -> dict:
    """
    Load pre-computed NMF programs from disk.
    Falls back to MSigDB Hallmark programs if no computed programs exist.
    """
    output_path = _CNMF_OUTPUT_ROOT / f"{disease.upper()}_programs.json"
    if output_path.exists():
        try:
            data = json.loads(output_path.read_text())
            if data.get("programs"):
                return data
        except Exception:
            pass

    # Fallback: MSigDB Hallmark (universal, always available)
    from pipelines.cnmf_programs import get_programs_for_disease
    return get_programs_for_disease(disease)


def get_program_gene_loadings_from_disk(program_id: str) -> dict | None:
    """
    Look up gene loadings for a discovered program from disk cache.
    Returns None if not found (caller should fall back to registry).
    """
    for json_path in _CNMF_OUTPUT_ROOT.glob("*_programs.json"):
        try:
            data = json.loads(json_path.read_text())
            for prog in data.get("programs", []):
                if prog.get("program_id") == program_id:
                    return {
                        "program_name": program_id,
                        "top_genes":    prog.get("top_genes", []),
                        "gene_loadings": prog.get("gene_loadings", {}),
                        "n_returned":   len(prog.get("top_genes", [])),
                        "data_tier":    "computed_nmf",
                        "source":       prog.get("source", "NMF"),
                    }
        except Exception:
            continue
    return None
