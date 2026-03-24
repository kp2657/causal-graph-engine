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
    "CAD": {
        "cell_types":  ["monocyte", "macrophage", "cardiomyocyte", "smooth muscle cell"],
        "tissues":     ["heart", "aorta", "blood"],
        "description": "Coronary artery disease: smooth muscle + macrophage foam cells",
        "priority_cell": "macrophage",
        "max_cells":   50_000,
    },
    "IBD": {
        "cell_types":  ["macrophage", "intestinal epithelial cell", "T cell", "B cell",
                        "dendritic cell", "monocyte"],
        "tissues":     ["colon", "small intestine", "intestine"],
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
    """Live download from CELLxGENE census API."""
    import cellxgene_census

    h5ad_path.parent.mkdir(parents=True, exist_ok=True)

    # Build cell type filter — prefer priority cell, fall back to all listed types
    cell_types_to_query = [priority_cell] + [
        ct for ct in ctx["cell_types"] if ct != priority_cell
    ]

    with cellxgene_census.open_soma() as census:
        # Query human cells for this cell type + tissue combination
        value_filter_parts = []
        # Cell type filter
        ct_filter = " or ".join(
            f'cell_type == "{ct}"' for ct in cell_types_to_query[:3]
        )
        value_filter_parts.append(f"({ct_filter})")

        # Tissue filter
        if ctx.get("tissues"):
            tissue_filter = " or ".join(
                f'tissue_general == "{t}"' for t in ctx["tissues"]
            )
            value_filter_parts.append(f"({tissue_filter})")

        value_filter = " and ".join(value_filter_parts)

        # Download as AnnData
        adata = cellxgene_census.get_anndata(
            census=census,
            organism="Homo sapiens",
            obs_value_filter=value_filter,
            var_value_filter="feature_biotype == 'protein_coding'",
        )

    if adata.n_obs == 0:
        return {
            "h5ad_path": None, "n_cells": 0,
            "cell_type": priority_cell, "disease": disease,
            "source": "CELLxGENE_census",
            "status": "unavailable",
            "note": f"No cells matched filter: {value_filter}",
        }

    # Subsample if needed
    if adata.n_obs > cell_count:
        import numpy as np
        idx = np.random.choice(adata.n_obs, size=cell_count, replace=False)
        adata = adata[idx].copy()

    adata.write_h5ad(str(h5ad_path))

    return {
        "h5ad_path": str(h5ad_path),
        "n_cells":   adata.n_obs,
        "n_genes":   adata.n_vars,
        "cell_type": priority_cell,
        "disease":   disease,
        "source":    "CELLxGENE_census",
        "status":    "downloaded",
        "note":      f"Saved {adata.n_obs} cells to {h5ad_path}",
    }


def list_cached_downloads() -> dict:
    """Return all cached sc-RNA downloads."""
    if not _CACHE_ROOT.exists():
        return {"downloads": [], "cache_dir": str(_CACHE_ROOT)}
    cached = []
    for h5ad in _CACHE_ROOT.rglob("*.h5ad"):
        cached.append({"path": str(h5ad), "size_mb": round(h5ad.stat().st_size / 1e6, 1)})
    return {"downloads": cached, "n_cached": len(cached), "cache_dir": str(_CACHE_ROOT)}
