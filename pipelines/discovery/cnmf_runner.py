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


def _run_sklearn_nmf(adata: Any, disease: str, k: int, n_top: int) -> dict:
    """NMF via sklearn — single run (no stability selection)."""
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
    # Clip negatives (can appear after log1p if X had negative values pre-log)
    X = np.clip(X, 0, None)

    gene_names = list(adata.var_names)

    model = NMF(n_components=k, init="nndsvda", max_iter=500, random_state=42)
    W = model.fit_transform(X)   # cells × programs
    H = model.components_         # programs × genes

    programs = []
    for i in range(k):
        loading_vec = H[i]
        top_idx = np.argsort(loading_vec)[::-1][:n_top]
        top_genes = [gene_names[j] for j in top_idx if loading_vec[j] > 0]
        gene_loadings = {
            gene_names[j]: float(loading_vec[j])
            for j in top_idx if loading_vec[j] > 0
        }
        programs.append({
            "program_id":    f"{disease.upper()}_NMF_P{i+1:02d}",
            "gene_set":      top_genes[:n_top],
            "top_genes":     top_genes[:20],
            "gene_loadings": gene_loadings,
            "cell_type":     getattr(adata, "_discovery_cell_type", "unknown"),
            "source":        "sklearn_NMF",
        })

    return {
        "programs":   programs,
        "n_programs": k,
        "disease":    disease,
        "source":     "sklearn_NMF",
        "n_cells":    adata.n_obs,
        "n_genes":    adata.n_vars,
        "note":       "Single-run NMF. Install cnmf for consensus NMF with stability selection.",
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
            "top_genes":     top_genes[:20],
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
