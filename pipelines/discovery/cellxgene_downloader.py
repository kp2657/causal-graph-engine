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
    "RA": {
        # CellxGene Census 2025-11-08: dataset d18736c3 has 23,947 PBMC CD4+ T cells
        # from RA patients. Using CD4+ T aligns with CZI Perturb-seq β source (czi_2025_cd4t_perturb).
        "cell_types":  ["naive thymus-derived CD4-positive, alpha-beta T cell",
                        "CD4-positive, alpha-beta T cell"],
        "tissues":     ["blood"],
        "description": "RA: PBMC CD4+ T cells (Census d18736c3, 23,947 cells; matches CZI Perturb-seq β)",
        "priority_cell": "naive thymus-derived CD4-positive, alpha-beta T cell",
        "max_cells":   30_000,
        "stratified_disease_hints": {
            "naive thymus-derived CD4-positive, alpha-beta T cell": "rheumatoid arthritis",
            "CD4-positive, alpha-beta T cell": "rheumatoid arthritis",
        },
    },
    "SLE": {
        # CELLxGENE verified labels 2026-04-24: pDC confirmed; "B cell" → use naive B cell
        "cell_types":  ["plasmacytoid dendritic cell", "naive B cell", "classical monocyte",
                        "naive thymus-derived CD4-positive, alpha-beta T cell"],
        "tissues":     ["blood"],
        "description": "SLE: pDC interferon programs (CellxGENE 436154da) + B cell + monocyte",
        "priority_cell": "plasmacytoid dendritic cell",
        "max_cells":   30_000,
        "stratified_disease_hints": {
            "plasmacytoid dendritic cell": "systemic lupus erythematosus",
        },
    },
    "T2D": {
        "cell_types":  ["pancreatic beta cell", "hepatocyte", "adipocyte"],
        "tissues":     ["pancreas", "liver", "adipose tissue"],
        "description": "T2D: beta cell UPR + hepatocyte metabolic programs",
        "priority_cell": "pancreatic beta cell",
        "max_cells":   30_000,
    },
    "DED": {
        # Ocular surface cells for state-space embedding. Corneal/conjunctival cells
        # carry the secretory-mucin and glycocalyx programs altered in DED. T cells
        # drive the inflammatory signal but CELLxGENE conjunctival data is scarce;
        # use corneal epithelial + CD4+ T as joint context.
        "cell_types":  ["corneal epithelial cell", "conjunctival cell",
                        "naive thymus-derived CD4-positive, alpha-beta T cell"],
        "tissues":     ["eye", "cornea", "conjunctiva", "blood"],
        "description": "DED: corneal/conjunctival epithelial programs + T cell inflammatory axis",
        "priority_cell": "corneal epithelial cell",
        "max_cells":   30_000,
        "stratified_disease_hints": {
            "corneal epithelial cell": "dry eye syndrome",
        },
    },
    "CAD": {
        # priority_cell = cardiac endothelial cell — matches GSE210681 (Schnitzler 2023) which
        # is predominantly HAEC (Human Aortic Endothelial Cells). Using endothelial h5ad
        # for state-space and GPS disease sig aligns cell context with Perturb-seq β source.
        # Download: python -m pipelines.discovery.cellxgene_downloader download_all CAD
        # CAD_smooth_muscle_cell.h5ad is still used as fallback when endothelial not present.
        "cell_types":  ["cardiac endothelial cell", "smooth muscle cell",
                        "macrophage", "monocyte", "hepatocyte"],
        "tissues":     ["vasculature", "heart", "aorta", "blood", "liver"],
        "description": "CAD: endothelial (HAEC, matches GSE210681) + SMC (atherosclerosis plaque) + hepatocyte (PCSK9/LDLR/HMGCR)",
        "priority_cell": "cardiac endothelial cell",
        "max_cells":   50_000,
        # Stratified disease hints: downloader fetches ALL available disease cells first,
        # then fills remaining slots with normal cells. Value = disease label to prioritise.
        # Disease labels verified in census 2025-11-08:
        #   cardiac endothelial cell:           30,564 MI + 22,981 myocarditis cells
        #   cardiac blood vessel endothelial:   3,052 coronary artery disorder
        #   endothelial cell (generic):         11,950 atherosclerosis
        #   smooth muscle cell:                 atherosclerosis label
        "stratified_disease_hints": {
            "cardiac endothelial cell": "myocardial infarction",
            "smooth muscle cell":       "atherosclerosis",
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

    requested_types: list[str] = cell_types or PHASE_A_CELL_TYPES.get(disease_key, ctx.get("cell_types", [ctx["priority_cell"]]))

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


if __name__ == "__main__":
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format="%(levelname)s %(message)s")

    _argv = sys.argv[1:]
    if not _argv:
        print("Usage:")
        print("  python -m pipelines.discovery.cellxgene_downloader download_all <DISEASE>")
        print("  python -m pipelines.discovery.cellxgene_downloader list")
        print(f"\nKnown diseases: {sorted(DISEASE_CELLXGENE_MAP.keys())}")
        sys.exit(0)

    _cmd = _argv[0]
    if _cmd == "download_all":
        if len(_argv) < 2:
            print("Usage: download_all <DISEASE>")
            sys.exit(1)
        _disease = _argv[1].upper()
        if _disease not in DISEASE_CELLXGENE_MAP:
            print(f"Unknown disease '{_disease}'. Known: {sorted(DISEASE_CELLXGENE_MAP.keys())}")
            sys.exit(1)
        print(f"Downloading CellxGene scRNA-seq for {_disease}...")
        _results = download_all_cell_types(_disease)
        for _ct, _r in _results.items():
            _status = _r.get("status", "?")
            _n = _r.get("n_cells", 0)
            _path = _r.get("h5ad_path", "")
            print(f"  {_ct}: {_status} | {_n:,} cells | {_path}")
    elif _cmd == "list":
        import json as _json
        print(_json.dumps(list_cached_downloads(), indent=2))
    else:
        print(f"Unknown command: {_cmd}")
