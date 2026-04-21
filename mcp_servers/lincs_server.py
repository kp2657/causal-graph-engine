"""
lincs_server.py — MCP server for perturbation-based gene signatures.

Primary source: Perturb-seq CRISPR datasets (disease-matched cell lines).
  See perturbseq_server.py + data/perturbseq/{dataset_id}/signatures.json.gz

Directional fallback: Enrichr LINCS L1000 consensus gene sets (keyless, ±1 pseudo-FC).
  Enrichr LINCS library: https://maayanlab.cloud/Enrichr

Cascade:
  Perturb-seq (quantitative log2FC, CRISPR, disease-matched) →
  Enrichr LINCS L1000 (directional ±1.0, last resort)

Run standalone:  python mcp_servers/lincs_server.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import httpx

from pipelines.api_cache import api_cached


try:
    import fastmcp
    mcp = fastmcp.FastMCP("perturbation-server")
    _tool = mcp.tool()
except ImportError:
    def _tool(fn=None, **_):
        return fn if fn is not None else (lambda f: f)
    mcp = None

ENRICHR_API = "https://maayanlab.cloud/Enrichr"

# ---------------------------------------------------------------------------
# Enrichr LINCS directional fallback
# ---------------------------------------------------------------------------

def _query_enrichr_lincs(
    gene_symbol: str,
    perturbation_type: str = "KD",
    top_k: int = 100,
) -> dict:
    """
    Keyless directional fallback using Enrichr LINCS L1000 consensus gene sets.

    Returns ±1.0 pseudo-FC (direction only, no magnitude).
    UP library → genes upregulated on KD → pseudo-FC +1.0
    DN library → genes downregulated on KD → pseudo-FC −1.0
    """
    _LIBRARY_PAIRS = [
        (
            "LINCS_L1000_CRISPR_KO_Consensus_Sigs_up",
            "LINCS_L1000_CRISPR_KO_Consensus_Sigs_dn",
        ),
        (
            "LINCS_L1000_Kinase_Perturbations_up",
            "LINCS_L1000_Kinase_Perturbations_dn",
        ),
    ]
    _TERM_CANDIDATES = [gene_symbol, f"{gene_symbol} KD", f"{gene_symbol} KO", f"{gene_symbol}_KD"]

    signature: dict[str, float] = {}

    for lib_up, lib_dn in _LIBRARY_PAIRS:
        for fc_val, lib in [(+1.0, lib_up), (-1.0, lib_dn)]:
            for term in _TERM_CANDIDATES:
                try:
                    resp = httpx.get(
                        f"{ENRICHR_API}/geneSetLibrary",
                        params={"mode": "json", "libraryName": lib, "term": term},
                        timeout=10.0,
                    )
                    if resp.status_code != 200:
                        continue
                    genes = resp.json().get(term, [])
                    for g in genes[:top_k]:
                        if g not in signature:
                            signature[g] = fc_val
                    if genes:
                        break
                except Exception:
                    continue
        if signature:
            break

    return {
        "gene":             gene_symbol,
        "cell_line":        "L1000_consensus",
        "signature":        signature,
        "n_genes_measured": len(signature),
        "source":           "enrichr_lincs_directional" if signature else "enrichr_not_found",
        "note":             "Directional ±1.0 pseudo-FC from Enrichr LINCS consensus; no magnitude.",
    }


# ---------------------------------------------------------------------------
# Beta computation (shared)
# ---------------------------------------------------------------------------

def _compute_program_beta_from_signature(
    signature: dict[str, float],
    program_gene_set: list[str] | set[str],
) -> float | None:
    """
    β = mean log2FC of program genes in signature.
    None if < 5% of program genes are measured.
    """
    pg_set = set(program_gene_set)
    hits = {g: signature[g] for g in pg_set if g in signature}
    if not hits:
        return None
    coverage = len(hits) / max(len(pg_set), 1)
    if coverage < 0.05:
        return None
    return sum(hits.values()) / len(pg_set)


# ---------------------------------------------------------------------------
# MCP tools
# ---------------------------------------------------------------------------

@_tool
@api_cached(ttl_days=30)
def get_lincs_gene_signature(
    gene_symbol: str,
    perturbation_type: str = "KD",
    cell_line: str | None = None,
    disease_context: str | None = None,
    top_k: int = 100,
) -> dict:
    """
    Retrieve perturbation signature for a gene.

    Cascade:
      1. Perturb-seq (CRISPR, disease-context-matched, quantitative log2FC)
      2. Enrichr LINCS L1000 directional (±1.0 pseudo-FC, no magnitude)

    Args:
        gene_symbol:       Gene to query (e.g. "PCSK9")
        perturbation_type: "KD" (knockdown) or "OE" (overexpression)
        cell_line:         Ignored (retained for API compatibility)
        disease_context:   Disease key for Perturb-seq dataset selection
                           ("CAD" | "IBD" | "RA" | "T2D" | "AD" | "SLE")
        top_k:             Max downstream genes returned

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

    # Primary: Perturb-seq (CRISPR, disease-matched)
    try:
        from mcp_servers.perturbseq_server import get_perturbseq_signature
        ps = get_perturbseq_signature(gene, disease_context=disease_context, top_k=top_k)
        if ps.get("signature") and ps["source"] not in ("not_cached", "gene_not_found", "not_found"):
            ps["evidence_tier"] = "Tier3_Provisional"
            ps["perturbation_type"] = perturbation_type
            return ps
    except Exception:
        pass

    # Fallback: Enrichr LINCS directional
    result = _query_enrichr_lincs(gene, perturbation_type, top_k)
    result["evidence_tier"] = "Tier3_Provisional"
    result["perturbation_type"] = perturbation_type
    return result


@_tool
@api_cached(ttl_days=30)
def compute_lincs_program_beta(
    gene_symbol: str,
    program_gene_set: list[str],
    cell_line: str | None = None,
    disease_context: str | None = None,
) -> dict:
    """
    Compute Tier 3 β_{gene→program} from perturbation signature.

    Uses Perturb-seq (primary) or Enrichr directional (fallback).
    β = mean log2FC of program genes in gene's perturbation signature.
    Coverage < 5% → returns None.

    Args:
        gene_symbol:      Gene being perturbed
        program_gene_set: Genes defining the biological program
        cell_line:        Ignored (retained for API compatibility)
        disease_context:  Disease key for dataset selection

    Returns:
        {
          "gene": str,
          "program_coverage": float,
          "beta": float | None,
          "evidence_tier": "Tier3_Provisional",
          "cell_line": str,
          "data_source": str,
        }
    """
    sig_result = get_lincs_gene_signature(gene_symbol, "KD", cell_line, disease_context)
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
        "data_source":      f"Perturbation_{actual_cl}_{sig_result.get('source', '')}",
        "note": (
            f"CRISPR/KD perturbation in {actual_cl}. "
            "Tier 3 evidence; not sufficient alone for clinical translation."
        ) if beta is not None else "Insufficient landmark coverage (< 5%) for β estimate.",
    }


@_tool
def list_perturbation_data_sources() -> dict:
    """List available perturbation data sources (Perturb-seq datasets + Enrichr fallback)."""
    try:
        from mcp_servers.perturbseq_server import list_perturbseq_datasets
        ps_info = list_perturbseq_datasets()
    except Exception:
        ps_info = {"datasets": [], "n_datasets": 0}

    return {
        "sources": [
            {
                "name":        "Perturb-seq (primary)",
                "type":        "CRISPR_perturbation",
                "n_datasets":  ps_info["n_datasets"],
                "n_cached":    ps_info.get("n_cached", 0),
                "note":        "Disease-matched CRISPR KO/CRISPRi. Quantitative log2FC.",
            },
            {
                "name":        "Enrichr LINCS L1000 (fallback)",
                "type":        "consensus_gene_sets",
                "n_datasets":  2,
                "note":        "Directional ±1.0 pseudo-FC only. No magnitude. Keyless.",
                "libraries":   [
                    "LINCS_L1000_CRISPR_KO_Consensus_Sigs",
                    "LINCS_L1000_Kinase_Perturbations",
                ],
            },
        ],
        "cascade": "Perturb-seq → Enrichr",
        "datasets": ps_info.get("datasets", []),
    }


# Backwards-compat alias (perturbation_genomics_agent imports this)
list_lincs_cell_lines = list_perturbation_data_sources


if __name__ == "__main__" and mcp is not None:
    mcp.run()
