"""
lincs_server.py — MCP server for LINCS L1000 perturbation signatures.

LINCS L1000 (Subramanian et al., Cell 2017): transcriptional responses to
~20,000 chemical and genetic perturbations (shRNA KD, ORF OE, CRISPR) across
~9 cancer cell lines, measured on the L1000 platform (978 landmark genes).

This provides Tier 3 β estimates: direct perturbation data but in a cell line
that may not match the disease-relevant cell type.

Data access:
  - CLUE REST API: https://api.clue.io/api/ (free, requires CLUE_API_KEY)
  - Enrichr LINCS L1000 endpoint: no key required
  - iLINCS: https://www.ilincs.org/api/ (no key required for gene signatures)

Run standalone:  python mcp_servers/lincs_server.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import httpx

try:
    import fastmcp
    mcp = fastmcp.FastMCP("lincs-server")
    _tool = mcp.tool()
except ImportError:
    def _tool(fn=None, **_):
        return fn if fn is not None else (lambda f: f)
    mcp = None

# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

ILINCS_API   = "https://www.ilincs.org/api"
ENRICHR_API  = "https://maayanlab.cloud/Enrichr"
CLUE_API     = "https://api.clue.io/api"
CLUE_API_KEY = os.getenv("CLUE_API_KEY", "")

# LINCS L1000 cell lines and their disease-context relevance
LINCS_CELL_LINES: dict[str, dict] = {
    "A375":  {"tissue": "skin",     "disease_context": "melanoma"},
    "A549":  {"tissue": "lung",     "disease_context": "lung_adenocarcinoma"},
    "HT29":  {"tissue": "colon",    "disease_context": "colorectal_carcinoma"},
    "MCF7":  {"tissue": "breast",   "disease_context": "breast_carcinoma"},
    "PC3":   {"tissue": "prostate", "disease_context": "prostate_carcinoma"},
    "VCAP":  {"tissue": "prostate", "disease_context": "prostate_carcinoma"},
    "HA1E":  {"tissue": "kidney",   "disease_context": "transformed_kidney"},
    "HEK293T": {"tissue": "kidney", "disease_context": "transformed_embryonic"},
    "HEPG2": {"tissue": "liver",    "disease_context": "hepatocellular"},
}

# Disease → preferred LINCS cell lines
_DISEASE_LINCS_LINES: dict[str, list[str]] = {
    "CAD":  ["VCAP", "HEPG2", "A549"],
    "IBD":  ["HT29", "A549"],
    "RA":   ["A375", "A549"],
    "T2D":  ["HEPG2"],
    "AD":   ["A549"],
    "SLE":  ["A375", "A549"],
}


def _query_ilincs_signature(
    gene_symbol: str,
    perturbation_type: str = "KD",
    cell_line: str | None = None,
    top_k: int = 100,
) -> dict:
    """
    Query iLINCS for a gene's knockdown signature.
    Returns {gene, cell_line, signature: {gene_symbol: log2fc}, n_genes_measured}.
    """
    # iLINCS /GeneInfos endpoint for gene-level signature query
    params: dict[str, Any] = {
        "geneSymbol": gene_symbol,
        "pert_type":  perturbation_type,
    }
    if cell_line:
        params["cell_id"] = cell_line

    try:
        resp = httpx.get(
            f"{ILINCS_API}/SignatureMeta",
            params=params,
            timeout=15.0,
        )
        resp.raise_for_status()
        data = resp.json()
        signatures = data if isinstance(data, list) else data.get("data", [])
        if not signatures:
            return {"gene": gene_symbol, "cell_line": cell_line, "signature": {}, "n_genes_measured": 0, "source": "ilincs_not_found"}

        # Take the first (strongest) matching signature
        sig_meta = signatures[0]
        sig_id = sig_meta.get("signatureid") or sig_meta.get("pert_id")
        actual_cell = sig_meta.get("cell_id", cell_line)
        n_genes = sig_meta.get("numGenes", 0)

        # Fetch actual differential expression values
        sig_resp = httpx.get(
            f"{ILINCS_API}/Signature",
            params={"signatureid": sig_id, "pValueThreshold": 0.05},
            timeout=20.0,
        )
        sig_resp.raise_for_status()
        sig_data = sig_resp.json()
        gene_effects = sig_data if isinstance(sig_data, list) else sig_data.get("data", [])

        signature = {
            entry.get("Name", entry.get("symbol", "")): float(entry.get("Value", entry.get("log2fc", 0)))
            for entry in gene_effects[:top_k]
            if entry.get("Name") or entry.get("symbol")
        }

        return {
            "gene": gene_symbol,
            "cell_line": actual_cell,
            "signature": signature,
            "n_genes_measured": n_genes,
            "signature_id": sig_id,
            "source": "ilincs_live",
        }
    except Exception as exc:
        return {
            "gene": gene_symbol, "cell_line": cell_line, "signature": {},
            "n_genes_measured": 0, "source": "ilincs_error", "error": str(exc),
        }


def _compute_program_beta_from_signature(
    signature: dict[str, float],
    program_gene_set: list[str] | set[str],
) -> float | None:
    """
    Compute β_{gene→program} from L1000 KD signature + program gene set.

    β = mean log2FC of program genes in KD signature (coverage-weighted).
    Only returns a value if ≥ 5% of program genes are measured in the signature.
    """
    pg_set = set(program_gene_set)
    hits = {g: signature[g] for g in pg_set if g in signature}
    if not hits:
        return None
    coverage = len(hits) / max(len(pg_set), 1)
    if coverage < 0.05:
        return None
    return sum(hits.values()) / len(pg_set)


@_tool
def get_lincs_gene_signature(
    gene_symbol: str,
    perturbation_type: str = "KD",
    cell_line: str | None = None,
    disease_context: str | None = None,
    top_k: int = 100,
) -> dict:
    """
    Retrieve LINCS L1000 perturbation signature for a gene.

    Queries iLINCS REST API for the transcriptional response to gene KD/OE.
    Returns differential expression for top landmark genes.

    Args:
        gene_symbol:       Gene to query (e.g. "PCSK9")
        perturbation_type: "KD" (knockdown/shRNA) or "OE" (overexpression/ORF)
        cell_line:         Specific cell line; if None, uses disease_context
        disease_context:   Disease key for auto cell-line selection ("CAD", "IBD", etc.)
        top_k:             Max number of DE genes to return in signature

    Returns:
        {
          "gene": str,
          "cell_line": str,
          "signature": {gene_symbol: log2fc},
          "n_genes_measured": int,
          "evidence_tier": "Tier3_Provisional",
          "source": str,
        }
    """
    gene = gene_symbol.upper()

    # Select cell line from disease context if not specified
    if cell_line is None and disease_context:
        dk = disease_context.upper()
        preferred = _DISEASE_LINCS_LINES.get(dk, [])
        cell_line = preferred[0] if preferred else None

    result = _query_ilincs_signature(gene, perturbation_type, cell_line, top_k)
    result["evidence_tier"] = "Tier3_Provisional"
    result["perturbation_type"] = perturbation_type
    return result


@_tool
def compute_lincs_program_beta(
    gene_symbol: str,
    program_gene_set: list[str],
    cell_line: str | None = None,
    disease_context: str | None = None,
) -> dict:
    """
    Compute Tier 3 β_{gene→program} from LINCS L1000 KD signature.

    β = mean log2FC of program genes in gene's KD signature.
    Coverage < 5% → returns None (insufficient landmark overlap).

    Args:
        gene_symbol:      Gene being perturbed
        program_gene_set: Genes defining the biological program
        cell_line:        Cell line for the signature
        disease_context:  Disease key for auto cell-line selection

    Returns:
        {
          "gene": str,
          "program_coverage": float,     # fraction of program genes measured
          "beta": float | None,           # mean log2FC (None if < 5% coverage)
          "evidence_tier": "Tier3_Provisional",
          "cell_line": str,
          "data_source": str,
        }
    """
    sig_result = get_lincs_gene_signature(
        gene_symbol, "KD", cell_line, disease_context
    )
    signature  = sig_result.get("signature", {})
    actual_cl  = sig_result.get("cell_line", "unknown")
    pg_set     = set(program_gene_set)
    hits       = {g: signature[g] for g in pg_set if g in signature}
    coverage   = len(hits) / max(len(pg_set), 1)
    beta       = _compute_program_beta_from_signature(signature, pg_set)

    return {
        "gene":             gene_symbol,
        "program_coverage": round(coverage, 3),
        "beta":             round(beta, 4) if beta is not None else None,
        "evidence_tier":    "Tier3_Provisional",
        "cell_line":        actual_cl,
        "data_source":      f"LINCS_L1000_{actual_cl}_shRNA_{sig_result.get('source', '')}",
        "note": (
            f"Direct shRNA KD in {actual_cl} — cell line may not match disease tissue. "
            "Tier 3 evidence; not sufficient alone for clinical translation."
        ) if beta is not None else "Insufficient landmark coverage (< 5%) for β estimate.",
    }


@_tool
def list_lincs_cell_lines() -> dict:
    """List LINCS L1000 cell lines with tissue and disease context."""
    return {"cell_lines": [{"name": k, **v} for k, v in LINCS_CELL_LINES.items()]}


if __name__ == "__main__" and mcp is not None:
    mcp.run()
