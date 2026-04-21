"""
cellxgene_downloader.py — Download disease-matched sc-RNA from CELLxGENE census.

Uses the CELLxGENE census Python API to query and download annotated single-cell
RNA-seq data for the disease-relevant cell type. Downloads are cached in
data/cellxgene/{disease}/ and reused across pipeline runs.

Requires: pip install cellxgene-census
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ---------------------------------------------------------------------------
# Disease → CELLxGENE cell type + tissue filter map
# ---------------------------------------------------------------------------

# Maps short disease name → filters for CELLxGENE census query.
# cell_type: ontology term matching CL (Cell Ontology) labels in CELLxGENE
# tissue:    tissue context (used to narrow the query to relevant cells)
# mondo_id:  MONDO disease ontology ID for disease-associated cells
DISEASE_CELLXGENE_MAP: dict[str, dict] = {
    "IBD": {
        "cell_types":  ["macrophage", "enterocyte", "colonocyte",
                        "CD4-positive, alpha-beta memory T cell",
                        "CD8-positive, alpha-beta memory T cell",
                        "B cell", "dendritic cell", "monocyte"],
        "tissues":     ["small intestine", "large intestine", "colon"],
        "description": "IBD: intestinal macrophage + epithelial + T cell programs",
        "priority_cell": "macrophage",
        "max_cells":   50_000,
    },
    "RA": {
        "cell_types":  ["fibroblast", "macrophage", "T cell", "B cell", "dendritic cell"],
        "tissues":     ["synovium", "blood", "joint"],
        "description": "Rheumatoid arthritis: synovial fibroblast + immune cell programs",
        "priority_cell": "fibroblast",
        "max_cells":   30_000,
    },
    "SLE": {
        "cell_types":  ["plasmacytoid dendritic cell", "B cell", "T cell", "monocyte"],
        "tissues":     ["blood", "kidney"],
        "description": "SLE: pDC interferon programs + B cell activation",
        "priority_cell": "plasmacytoid dendritic cell",
        "max_cells":   30_000,
    },
    "AD": {
        "cell_types":  ["microglia", "astrocyte", "neuron", "oligodendrocyte"],
        "tissues":     ["brain", "cerebral cortex", "hippocampus"],
        "description": "Alzheimer's: microglia + astrocyte reactive programs",
        "priority_cell": "microglia",
        "max_cells":   50_000,
    },
    "T2D": {
        "cell_types":  ["pancreatic beta cell", "hepatocyte", "adipocyte"],
        "tissues":     ["pancreas", "liver", "adipose tissue"],
        "description": "T2D: beta cell UPR + hepatocyte metabolic programs",
        "priority_cell": "pancreatic beta cell",
        "max_cells":   30_000,
    },
    "AMD": {
        # RPE = primary lesion site in dry AMD; Müller glia = disease-reactive support cells.
        # Hepatocyte included because complement proteins (CFH, C3, CFD, CFB, CFI, C5) are
        # synthesised in the liver — not the retina. Liver h5ad enables state-space scoring
        # for complement targets and provides the tissue context for liver eQTL instruments.
        "cell_types":  ["retinal pigment epithelial cell", "Mueller cell",
                        "rod photoreceptor cell", "hepatocyte"],
        "tissues":     ["retina", "eye", "liver"],
        "description": "AMD: RPE + Müller glia (retinal programs) + hepatocyte (complement programs)",
        "priority_cell": "retinal pigment epithelial cell",
        "max_cells":   30_000,
        # Stratified disease hints: fetch all AMD disease cells first, fill rest with normal.
        # RPE: "age related macular degeneration 7" — only label with RPE AMD cells in census.
        # Mueller: "macular degeneration" — 1,115 AMD cells in dataset f5be9ed2.
        "stratified_disease_hints": {
            "retinal pigment epithelial cell": "age related macular degeneration 7",
            "Mueller cell": "macular degeneration",
        },
    },
    "CAD": {
        "cell_types":  ["smooth muscle cell", "macrophage", "cardiac endothelial cell",
                        "monocyte", "hepatocyte"],
        "tissues":     ["vasculature", "heart", "aorta", "blood", "liver"],
        "description": "CAD: SMC (atherosclerosis) + hepatocyte (PCSK9/LDLR/HMGCR) + macrophage foam cells",
        "priority_cell": "smooth muscle cell",
        "max_cells":   50_000,
        # Stratified disease hints: downloader fetches ALL available disease cells first,
        # then fills remaining slots with normal cells. Value = disease label to prioritise.
        # GPS _build_sig_from_h5ad then splits by obs disease label for DEG computation.
        "stratified_disease_hints": {
            "smooth muscle cell": "atherosclerosis",
        },
    },
}

_CACHE_ROOT = Path("./data/cellxgene")


def download_disease_scrna(
    disease: str,
    max_cells: int | None = None,
    force: bool = False,
) -> dict:
    """
    Download sc-RNA from CELLxGENE census for the disease-matched cell type.

    Downloads are cached — subsequent calls with the same disease return the
    cached h5ad path without re-downloading.

    Args:
        disease:   Short disease name (CAD, IBD, RA, SLE, AD, T2D)
        max_cells: Override cell count limit (defaults from DISEASE_CELLXGENE_MAP)
        force:     Re-download even if cache exists

    Returns:
        {
            "h5ad_path": str | None,   # path to saved h5ad (None if download failed)
            "n_cells": int,
            "cell_type": str,
            "disease": str,
            "source": str,
            "status": "cached" | "downloaded" | "unavailable",
        }
    """
    ctx = DISEASE_CELLXGENE_MAP.get(disease.upper(), DISEASE_CELLXGENE_MAP.get(disease, {}))
    if not ctx:
        return {
            "h5ad_path": None, "n_cells": 0,
            "cell_type": "unknown", "disease": disease,
            "source": "CELLxGENE_census",
            "status": "unavailable",
            "note": f"Disease '{disease}' not in DISEASE_CELLXGENE_MAP",
        }

    cell_count = max_cells or ctx["max_cells"]
    priority_cell = ctx["priority_cell"]
    h5ad_dir = _CACHE_ROOT / disease.upper()
    h5ad_path = h5ad_dir / f"{disease.upper()}_{priority_cell.replace(' ', '_')}.h5ad"

    # Return cache hit
    if h5ad_path.exists() and not force:
        try:
            import anndata
            adata = anndata.read_h5ad(h5ad_path)
            return {
                "h5ad_path": str(h5ad_path),
                "n_cells":   adata.n_obs,
                "cell_type": priority_cell,
                "disease":   disease,
                "source":    "CELLxGENE_census_cache",
                "status":    "cached",
            }
        except Exception:
            pass  # Cache corrupted — re-download

    # Attempt live download
    try:
        import cellxgene_census  # noqa: F401
    except ImportError:
        return {
            "h5ad_path": None, "n_cells": 0,
            "cell_type": priority_cell, "disease": disease,
            "source": "CELLxGENE_census",
            "status": "unavailable",
            "note": "Install: pip install cellxgene-census",
        }

    try:
        return _download_from_census(
            disease=disease,
            ctx=ctx,
            h5ad_path=h5ad_path,
            cell_count=cell_count,
            priority_cell=priority_cell,
        )
    except Exception as exc:
        return {
            "h5ad_path": None, "n_cells": 0,
            "cell_type": priority_cell, "disease": disease,
            "source": "CELLxGENE_census",
            "status": "unavailable",
            "note": str(exc),
        }


def _download_from_census(
    disease: str,
    ctx: dict,
    h5ad_path: Path,
    cell_count: int,
    priority_cell: str,
) -> dict:
    """
    Live download from CELLxGENE census using axis_query.

    Avoids get_anndata / to_anndata entirely (segfaults in tiledbsoma 2.3.0
    when passed a NumPy array of obs_coords).

    Two-phase approach:
      Phase 1 — obs-only axis_query (fast metadata scan, no X data).
                 Subsample soma_joinids to cell_count as a Python list.
      Phase 2 — axis_query with coords=(python_list,) for only the N cells.
                 Read obs / var / X separately, build AnnData in Python.
    """
    import cellxgene_census
    import tiledbsoma
    import anndata as ad

    h5ad_path.parent.mkdir(parents=True, exist_ok=True)

    # Filter: priority cell type, optional tissue constraint.
    _stratified_hints: dict = ctx.get("stratified_disease_hints", {})
    _disease_hint: str | None = _stratified_hints.get(priority_cell)

    if ctx.get("tissues"):
        tissue_filter = " or ".join(
            f'tissue_general == "{t}"' for t in ctx["tissues"]
        )
        base_filter = f'cell_type == "{priority_cell}" and ({tissue_filter})'
    else:
        base_filter = f'cell_type == "{priority_cell}"'

    var_filter = "feature_type == 'protein_coding'"

    with cellxgene_census.open_soma(census_version="2025-11-08") as census:
        human = census["census_data"]["homo_sapiens"]

        def _get_ids(obs_filter: str, limit: int) -> "list[int]":
            with human.axis_query(
                measurement_name="RNA",
                obs_query=tiledbsoma.AxisQuery(value_filter=obs_filter),
            ) as q:
                meta = q.obs(column_names=["soma_joinid"]).concat().to_pandas()
            n = min(len(meta), limit)
            return meta["soma_joinid"].iloc[:n].tolist()

        if _disease_hint:
            # Stratified: all available disease cells + fill with normal cells
            disease_ids = _get_ids(
                f'{base_filter} and disease == "{_disease_hint}"', cell_count
            )
            n_normal = max(0, cell_count - len(disease_ids))
            normal_ids = _get_ids(
                f'{base_filter} and disease == "normal"', n_normal
            ) if n_normal > 0 else []
            selected_ids: list[int] = disease_ids + normal_ids
            print(
                f"[downloader] stratified: {len(disease_ids)} {_disease_hint!r} + "
                f"{len(normal_ids)} normal {priority_cell} cells",
                flush=True,
            )
        else:
            all_ids = _get_ids(base_filter, cell_count)
            selected_ids = all_ids

        if not selected_ids:
            return {
                "h5ad_path": None, "n_cells": 0,
                "cell_type": priority_cell, "disease": disease,
                "source": "CELLxGENE_census",
                "status": "unavailable",
                "note": f"No cells matched: {base_filter}",
            }

        # ------------------------------------------------------------------
        # Phase 2: targeted axis_query for selected cells only
        # ------------------------------------------------------------------
        with human.axis_query(
            measurement_name="RNA",
            obs_query=tiledbsoma.AxisQuery(coords=(selected_ids,)),
            var_query=tiledbsoma.AxisQuery(value_filter=var_filter),
        ) as query:

            obs_df = (
                query.obs(
                    column_names=["soma_joinid", "cell_type", "tissue_general",
                                  "disease", "donor_id", "sex", "dataset_id"]
                )
                .concat()
                .to_pandas()
            )

            var_df = (
                query.var(column_names=["soma_joinid", "feature_name", "feature_id"])
                .concat()
                .to_pandas()
            )

            # X: .coos().concat().to_scipy() returns scipy COO with global soma_joinids
            # as row/col indices — remap to local 0-based indices.
            import numpy as np
            import scipy.sparse as sp
            import pandas as pd

            obs_id_map = pd.Series(
                np.arange(len(obs_df), dtype=np.int32),
                index=obs_df["soma_joinid"].values,
            )
            var_id_map = pd.Series(
                np.arange(len(var_df), dtype=np.int32),
                index=var_df["soma_joinid"].values,
            )

            coo_scipy = query.X("raw").coos().concat().to_scipy()
            # coo_scipy.row / .col are global soma_joinids; remap to local
            rows = obs_id_map[coo_scipy.row].values.astype(np.int32)
            cols = var_id_map[coo_scipy.col].values.astype(np.int32)
            X = sp.csr_matrix(
                (coo_scipy.data.astype(np.float32), (rows, cols)),
                shape=(len(obs_df), len(var_df)),
                dtype=np.float32,
            )

    if X.shape[0] == 0:
        return {
            "h5ad_path": None, "n_cells": 0,
            "cell_type": priority_cell, "disease": disease,
            "source": "CELLxGENE_census",
            "status": "unavailable",
            "note": "X matrix is empty",
        }

    # Build AnnData from pieces
    # obs: index = soma_joinid strings; index.name=None avoids AnnData column/name conflict
    obs_df = obs_df.reset_index(drop=True)
    obs_df.index = obs_df["soma_joinid"].astype(str)
    obs_df.index.name = None
    # var: use feature_id (Ensembl ID) as index — unique; keep feature_name as column
    var_df = var_df.reset_index(drop=True)
    var_df.index = var_df["feature_id"].astype(str)
    var_df.index.name = None

    adata = ad.AnnData(X=X, obs=obs_df, var=var_df)
    adata.write_h5ad(str(h5ad_path))

    return {
        "h5ad_path": str(h5ad_path),
        "n_cells":   adata.n_obs,
        "n_genes":   adata.n_vars,
        "cell_type": priority_cell,
        "disease":   disease,
        "source":    "CELLxGENE_census",
        "status":    "downloaded",
        "note":      f"Saved {adata.n_obs} cells × {adata.n_vars} genes to {h5ad_path}",
    }


# Phase A cell types per disease (therapeutically relevant subset only).
# macrophage is already ✓ for IBD + CAD; these are the additional downloads needed.
PHASE_A_CELL_TYPES: dict[str, list[str]] = {
    "IBD": [
        "macrophage",
        "enterocyte",
        "CD4-positive, alpha-beta memory T cell",
    ],
    "CAD": [
        "macrophage",
        "smooth muscle cell",
        "monocyte",
        "hepatocyte",           # LDLR, PCSK9, HMGCR — liver lipid programs
    ],
    "AMD": [
        "retinal pigment epithelial cell",
        "Mueller cell",
        "hepatocyte",           # CFH, C3, CFD, CFB, CFI, C5 — complement synthesised in liver
    ],
}


def download_all_cell_types(
    disease: str,
    cell_types: list[str] | None = None,
    max_cells_per_type: int = 50_000,
    force: bool = False,
) -> dict[str, dict]:
    """
    Download separate h5ads for each therapeutically relevant cell type.

    Phase A default cell types come from PHASE_A_CELL_TYPES[disease].
    Each cell type gets its own h5ad: data/cellxgene/{DISEASE}/{DISEASE}_{cell_type}.h5ad

    Args:
        disease:             Short disease name (IBD, CAD, …)
        cell_types:          Override list of cell type strings (CL ontology labels).
                             If None, uses PHASE_A_CELL_TYPES[disease].
        max_cells_per_type:  Cell count cap per cell type.
        force:               Re-download even if cache exists.

    Returns:
        Dict mapping cell_type → download result dict (same schema as download_disease_scrna).
    """
    disease_key = disease.upper()
    ctx = DISEASE_CELLXGENE_MAP.get(disease_key, DISEASE_CELLXGENE_MAP.get(disease, {}))
    if not ctx:
        return {
            "__error__": {
                "status": "unavailable",
                "note": f"Disease '{disease}' not in DISEASE_CELLXGENE_MAP",
            }
        }

    requested_types: list[str] = cell_types or PHASE_A_CELL_TYPES.get(disease_key, [ctx["priority_cell"]])

    try:
        import cellxgene_census  # noqa: F401
    except ImportError:
        return {
            ct: {
                "h5ad_path": None, "n_cells": 0,
                "cell_type": ct, "disease": disease,
                "source": "CELLxGENE_census",
                "status": "unavailable",
                "note": "Install: pip install cellxgene-census",
            }
            for ct in requested_types
        }

    results: dict[str, dict] = {}
    for cell_type in requested_types:
        safe_name = cell_type.replace(" ", "_").replace(",", "").replace("-", "_")
        h5ad_dir = _CACHE_ROOT / disease_key
        h5ad_path = h5ad_dir / f"{disease_key}_{safe_name}.h5ad"

        # Return cached if available and not forced
        if h5ad_path.exists() and not force:
            try:
                import anndata
                adata = anndata.read_h5ad(h5ad_path)
                results[cell_type] = {
                    "h5ad_path": str(h5ad_path),
                    "n_cells":   adata.n_obs,
                    "cell_type": cell_type,
                    "disease":   disease,
                    "source":    "CELLxGENE_census_cache",
                    "status":    "cached",
                }
                continue
            except Exception:
                pass  # Cache corrupted — re-download

        try:
            results[cell_type] = _download_from_census(
                disease=disease,
                ctx=ctx,
                h5ad_path=h5ad_path,
                cell_count=max_cells_per_type,
                priority_cell=cell_type,
            )
        except Exception as exc:
            results[cell_type] = {
                "h5ad_path": None, "n_cells": 0,
                "cell_type": cell_type, "disease": disease,
                "source": "CELLxGENE_census",
                "status": "unavailable",
                "note": str(exc),
            }

    return results


def list_cached_downloads() -> dict:
    """Return all cached sc-RNA downloads."""
    if not _CACHE_ROOT.exists():
        return {"downloads": [], "cache_dir": str(_CACHE_ROOT)}
    cached = []
    for h5ad in _CACHE_ROOT.rglob("*.h5ad"):
        cached.append({"path": str(h5ad), "size_mb": round(h5ad.stat().st_size / 1e6, 1)})
    return {"downloads": cached, "n_cached": len(cached), "cache_dir": str(_CACHE_ROOT)}
