"""
gps_server.py — MCP server for GPS compound reversal scoring.

GPS (Gene expression Profile predictor on chemical Structures)
Xing et al., Cell 2026. doi:10.1016/j.cell.2026.02.016
GitHub: https://github.com/Bin-Chen-Lab/GPS

What GPS does
-------------
Given:
  - A disease transcriptional signature (up/down DEGs from bulk or single-cell RNA-seq)
  - A chemical library (SMILES strings)

GPS:
  1. RCL — denoises LINCS L1000 compound signatures for training
  2. GPS4Drug — predicts compound-induced transcriptomic profiles from chemical structure
  3. MolSearch — Monte Carlo tree search to design novel compounds that maximally
     reverse the disease signature

This enables screening of compounds *not* in LINCS (structure-only prediction),
and de novo design of new chemical matter guided by transcriptional reversal.

Canonical GPS path in this repo (do not confuse with this file)
---------------------------------------------------------------
**Production chemistry / Tier 4** uses `pipelines/gps_disease_screen.py`
(`run_gps_disease_screens`, LINCS-based signatures, overlap scoring). That path is
invoked from `agents/tier4_translation/chemistry_agent.py`.

**This MCP module** (`gps_server.py`) is an **optional** adapter: local subprocess
calls to Bin-Chen-Lab GPS4Drug when `GPS_REPO_PATH` points at a cloned repo with
weights. Agents do not import it by default.

Intended standalone workflow for this MCP server
-----------------------------------------------
  1. Extract disease signature (DEGs from h5ad, disease vs healthy cells)
  2. Query GPS to rank known compounds that reverse the signature
  3. Cross-reference with target prioritization output (do top compounds hit top genes?)
  4. Return ranked repurposing candidates with reversal scores + structural novelty flags

Local vs web modes
-------------------
  LOCAL  — requires cloned GitHub repo + model weights + GPU recommended
           Set GPS_REPO_PATH in .env to enable
  WEB    — GPS web portal (if available); not yet a documented REST endpoint
  STUB   — returns schema-valid stub when neither is configured

Run standalone:  python mcp_servers/gps_server.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

try:
    import fastmcp
    mcp = fastmcp.FastMCP("gps-server")
    _tool = mcp.tool()
except ImportError:
    def _tool(fn=None, **_):
        return fn if fn is not None else (lambda f: f)
    mcp = None

GPS_REPO_PATH = os.getenv("GPS_REPO_PATH", "")   # path to cloned Bin-Chen-Lab/GPS repo

# ---------------------------------------------------------------------------
# Local GPS runner
# ---------------------------------------------------------------------------

def _run_gps_local(
    up_genes: list[str],
    down_genes: list[str],
    n_compounds: int = 50,
) -> dict | None:
    """
    Run GPS4Drug locally using cloned repo.

    Requires:
      - GPS_REPO_PATH set in .env → cloned https://github.com/Bin-Chen-Lab/GPS
      - Model weights in {GPS_REPO_PATH}/weights/
      - Dependencies: torch, rdkit, pandas (in GPS conda env or current env)
    """
    if not GPS_REPO_PATH or not Path(GPS_REPO_PATH).exists():
        return None

    repo = Path(GPS_REPO_PATH)
    predict_script = repo / "GPS4Drug" / "predict.py"
    if not predict_script.exists():
        return None

    import json
    import subprocess
    import tempfile

    query = {"up_genes": up_genes, "down_genes": down_genes, "n_compounds": n_compounds}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(query, f)
        query_path = f.name

    out_path = query_path.replace(".json", "_out.json")
    try:
        subprocess.run(
            [sys.executable, str(predict_script), "--query", query_path, "--out", out_path],
            cwd=str(repo),
            timeout=300,
            check=True,
            capture_output=True,
        )
        with open(out_path) as f:
            return json.load(f)
    except Exception:
        return None
    finally:
        for p in (query_path, out_path):
            try:
                Path(p).unlink(missing_ok=True)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# MCP tools
# ---------------------------------------------------------------------------

@_tool
def score_compounds_for_disease_reversal(
    up_genes: list[str],
    down_genes: list[str],
    disease_name: str = "",
    n_compounds: int = 50,
) -> dict:
    """
    Score compounds for their ability to reverse a disease transcriptional signature.

    Uses GPS (Xing et al., Cell 2026) to predict compound-induced transcriptomic
    profiles and rank compounds by reversal score against the disease signature.

    Args:
        up_genes:     Genes upregulated in disease (to be downregulated by compound)
        down_genes:   Genes downregulated in disease (to be upregulated by compound)
        disease_name: Human-readable disease label for provenance
        n_compounds:  Number of top compounds to return

    Returns:
        {
          "disease_name": str,
          "n_up": int,
          "n_down": int,
          "compounds": [
            {
              "compound_id": str,
              "name": str,
              "smiles": str,
              "reversal_score": float,   # higher = better reversal
              "is_approved": bool,       # FDA-approved status
              "targets": [str],          # known gene targets
              "source": str,             # "gps_local" | "gps_web" | "stub"
            }
          ],
          "method": str,
          "note": str,
        }
    """
    local = _run_gps_local(up_genes, down_genes, n_compounds)
    if local is not None:
        local["source"] = "gps_local"
        return local

    # STUB — GPS not configured locally
    return {
        "disease_name": disease_name,
        "n_up":         len(up_genes),
        "n_down":       len(down_genes),
        "compounds":    [],
        "method":       "GPS4Drug (Xing et al., Cell 2026)",
        "source":       "stub",
        "note": (
            "GPS not configured. To enable: "
            "(1) git clone https://github.com/Bin-Chen-Lab/GPS "
            "(2) download model weights from Zenodo (see repo README) "
            "(3) set GPS_REPO_PATH=/path/to/GPS in .env"
        ),
        "paper":   "Xing et al. Cell 2026. doi:10.1016/j.cell.2026.02.016",
        "github":  "https://github.com/Bin-Chen-Lab/GPS",
        "docker":  "See repo README for Docker deployment",
    }


@_tool
def extract_disease_signature_from_h5ad(
    h5ad_path: str,
    disease_cell_type: str = "macrophage",
    n_top_genes: int = 200,
) -> dict:
    """
    Extract a disease transcriptional signature from a Perturb-seq / single-cell h5ad.

    Computes DEGs between pathological and healthy cell states (defined by
    state_definition.py pathological basins) for use as GPS input.

    Args:
        h5ad_path:         Path to h5ad with obs['state'] or obs['leiden'] annotations
        disease_cell_type: Cell type label (for provenance)
        n_top_genes:       Number of top up/down genes to extract

    Returns:
        {
          "up_genes": [str],     # upregulated in disease states
          "down_genes": [str],   # downregulated in disease states
          "n_pathological_cells": int,
          "n_healthy_cells": int,
          "source": str,
        }
    """
    import numpy as np

    try:
        import anndata as ad
        import scipy.sparse as sp

        adata = ad.read_h5ad(h5ad_path)

        # Prefer state labels from state_definition pipeline
        state_col = next(
            (c for c in ("state_label", "state", "pathological_state", "leiden")
             if c in adata.obs.columns),
            None,
        )
        if state_col is None:
            return {
                "up_genes": [], "down_genes": [],
                "n_pathological_cells": 0, "n_healthy_cells": 0,
                "source": "error",
                "note": f"No state column found. obs columns: {list(adata.obs.columns[:10])}",
            }

        labels = adata.obs[state_col].astype(str)
        path_mask    = labels.str.lower().str.contains("pathol|disease|inflam|cad|ibd").values
        healthy_mask = labels.str.lower().str.contains("health|normal|ctrl|control").values

        if path_mask.sum() < 10 or healthy_mask.sum() < 10:
            return {
                "up_genes": [], "down_genes": [],
                "n_pathological_cells": int(path_mask.sum()),
                "n_healthy_cells":      int(healthy_mask.sum()),
                "source": "insufficient_cells",
                "note": "Fewer than 10 cells in pathological or healthy group.",
            }

        X = adata.X
        path_idx    = np.where(path_mask)[0]
        healthy_idx = np.where(healthy_mask)[0]
        mean_path    = np.asarray(X[path_idx].mean(axis=0)).flatten()
        mean_healthy = np.asarray(X[healthy_idx].mean(axis=0)).flatten()
        log2fc = (mean_path - mean_healthy) / np.log(2)

        gene_names = list(adata.var_names)
        sorted_idx = np.argsort(log2fc)
        up_idx   = sorted_idx[-n_top_genes:][::-1]
        down_idx = sorted_idx[:n_top_genes]

        return {
            "up_genes":             [gene_names[i] for i in up_idx   if log2fc[i] > 0.3],
            "down_genes":           [gene_names[i] for i in down_idx if log2fc[i] < -0.3],
            "n_pathological_cells": int(path_mask.sum()),
            "n_healthy_cells":      int(healthy_mask.sum()),
            "cell_type":            disease_cell_type,
            "source":               h5ad_path,
        }

    except Exception as exc:
        return {
            "up_genes": [], "down_genes": [],
            "n_pathological_cells": 0, "n_healthy_cells": 0,
            "source": "error",
            "note": str(exc),
        }


@_tool
def get_gps_status() -> dict:
    """Return GPS configuration status and setup instructions."""
    repo_configured = bool(GPS_REPO_PATH and Path(GPS_REPO_PATH).exists())
    weights_present = (
        (Path(GPS_REPO_PATH) / "weights").exists()
        if repo_configured else False
    )
    return {
        "gps_repo_path":       GPS_REPO_PATH or "(not set)",
        "repo_configured":     repo_configured,
        "weights_present":     weights_present,
        "mode":                "local" if (repo_configured and weights_present) else "stub",
        "paper":               "Xing et al. Cell 2026. doi:10.1016/j.cell.2026.02.016",
        "github":              "https://github.com/Bin-Chen-Lab/GPS",
        "setup": {
            "1_clone":         "git clone https://github.com/Bin-Chen-Lab/GPS",
            "2_weights":       "See GPS repo README → Zenodo link for model weights",
            "3_env_var":       "Add GPS_REPO_PATH=/path/to/GPS to .env",
            "4_docker":        "Or use Docker: see repo README for containerized deployment",
        },
    }


if __name__ == "__main__" and mcp is not None:
    mcp.run()
