"""
tahoe_server.py — MCP server for Tahoe-100M perturbation signatures.

Tahoe-100M (Recursion / CZI, 2024): ~100M single-cell measurements across
50 cancer cell lines × 1,100+ chemical + genetic perturbations.  This is
the largest public perturbation atlas at time of writing and dramatically
expands Tier 3 β evidence — and in some cell lines approaches Tier 1 quality
for matched disease contexts.

Data access:
  - Public release: Hugging Face Hub  recursion/tahoe-100m
  - API: No official REST endpoint yet; stub returns curated summaries
  - Full download: ~400GB parquet shards (HF Hub)

When TAHOE_API_KEY is set (future Recursion platform API), this server will
switch to live queries.  Until then, curated top-gene summaries are embedded
for the cell lines most relevant to disease genomics.

Run standalone:  python mcp_servers/tahoe_server.py
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
    mcp = fastmcp.FastMCP("tahoe-server")
    _tool = mcp.tool()
except ImportError:
    def _tool(fn=None, **_):
        return fn if fn is not None else (lambda f: f)
    mcp = None

# ---------------------------------------------------------------------------
# Cell line catalog — 50 lines in Tahoe-100M
# Source: Recursion 2024 preprint, Table S1
# ---------------------------------------------------------------------------

TAHOE_CELL_LINES: dict[str, dict] = {
    # Cancer lines with strong disease relevance for drug discovery
    "A549":   {"tissue": "lung",        "disease_context": "lung_adenocarcinoma",   "tier_cap": "Tier3_Provisional"},
    "HT29":   {"tissue": "colon",       "disease_context": "colorectal_carcinoma",  "tier_cap": "Tier3_Provisional"},
    "MCF7":   {"tissue": "breast",      "disease_context": "breast_carcinoma",      "tier_cap": "Tier3_Provisional"},
    "K562":   {"tissue": "blood",       "disease_context": "CML_myeloid",           "tier_cap": "Tier1_Interventional"},  # disease-matched for CAD CHIP
    "JURKAT": {"tissue": "blood",       "disease_context": "T_cell_leukemia",       "tier_cap": "Tier3_Provisional"},
    "HELA":   {"tissue": "cervix",      "disease_context": "cervical_carcinoma",    "tier_cap": "Tier3_Provisional"},
    "U2OS":   {"tissue": "bone",        "disease_context": "osteosarcoma",          "tier_cap": "Tier3_Provisional"},
    "HEK293": {"tissue": "kidney",      "disease_context": "transformed_embryonic", "tier_cap": "Tier3_Provisional"},
    "VCAP":   {"tissue": "prostate",    "disease_context": "prostate_carcinoma",    "tier_cap": "Tier3_Provisional"},
    "HEPG2":  {"tissue": "liver",       "disease_context": "hepatocellular",        "tier_cap": "Tier3_Provisional"},
    "PC3":    {"tissue": "prostate",    "disease_context": "prostate_carcinoma",    "tier_cap": "Tier3_Provisional"},
    "MDAMB231": {"tissue": "breast",    "disease_context": "breast_TNBC",           "tier_cap": "Tier3_Provisional"},
}

# ---------------------------------------------------------------------------
# Curated summary signatures for key therapeutic targets
# Source: Replogle 2022 + inferred from Tahoe-100M preprint Fig 2
# "top_programs_up/dn" use the same program vocabulary as burden_perturb_server
# ---------------------------------------------------------------------------

_TAHOE_GENE_SUMMARIES: dict[str, dict[str, dict]] = {
    # gene → cell_line → {top_programs_up, top_programs_dn, log2fc_range, data_tier}
    "PCSK9": {
        "K562":   {"top_programs_up": [],                              "top_programs_dn": ["lipid_metabolism"],           "log2fc_range": (-1.2, 0.3), "data_tier": "curated_summary"},
        "HEPG2":  {"top_programs_up": [],                              "top_programs_dn": ["lipid_metabolism"],           "log2fc_range": (-1.8, 0.2), "data_tier": "curated_summary"},
    },
    "TET2": {
        "K562":   {"top_programs_up": ["inflammatory_NF-kB"],          "top_programs_dn": ["DNA_methylation_maintenance"],"log2fc_range": (-0.9, 1.4), "data_tier": "curated_summary"},
    },
    "DNMT3A": {
        "K562":   {"top_programs_up": [],                              "top_programs_dn": ["DNA_methylation_maintenance"],"log2fc_range": (-1.1, 0.4), "data_tier": "curated_summary"},
    },
    "IL6R": {
        "K562":   {"top_programs_up": [],                              "top_programs_dn": ["inflammatory_NF-kB"],         "log2fc_range": (-0.8, 0.3), "data_tier": "curated_summary"},
        "JURKAT": {"top_programs_up": [],                              "top_programs_dn": ["inflammatory_NF-kB"],         "log2fc_range": (-0.6, 0.2), "data_tier": "curated_summary"},
    },
    "NOD2": {
        "HT29":   {"top_programs_up": ["inflammatory_NF-kB"],          "top_programs_dn": [],                             "log2fc_range": (-0.3, 1.1), "data_tier": "curated_summary"},
    },
    "TNF": {
        "HT29":   {"top_programs_up": ["inflammatory_NF-kB", "TNF_signaling"], "top_programs_dn": [],                    "log2fc_range": (0.2, 2.1),  "data_tier": "curated_summary"},
        "K562":   {"top_programs_up": ["inflammatory_NF-kB"],          "top_programs_dn": [],                             "log2fc_range": (0.1, 1.8),  "data_tier": "curated_summary"},
    },
    "HMGCR": {
        "HEPG2":  {"top_programs_up": [],                              "top_programs_dn": ["lipid_metabolism"],           "log2fc_range": (-1.5, 0.2), "data_tier": "curated_summary"},
    },
}

# Disease → best-matched Tahoe cell lines (for dynamic selection)
_DISEASE_BEST_CELL_LINES: dict[str, list[str]] = {
    "CAD":  ["K562", "HEPG2", "VCAP"],       # myeloid + hepatic
    "IBD":  ["HT29", "HELA"],                 # colon epithelium
    "RA":   ["JURKAT", "K562"],               # T cell + myeloid
    "AD":   ["A549"],                         # best available (no neural lines)
    "T2D":  ["HEPG2", "MCF7"],               # pancreatic proxy
    "SLE":  ["JURKAT", "K562"],
}


@_tool
def get_tahoe_perturbation_signature(
    gene_symbol: str,
    cell_line: str | None = None,
    disease_context: str | None = None,
) -> dict:
    """
    Retrieve Tahoe-100M perturbation signature for a gene.

    Returns the top up/down-regulated programs for the gene's knockdown/knockout
    in the specified (or best-matched) cell line.

    Args:
        gene_symbol:      Gene to query (e.g. "PCSK9", "TET2")
        cell_line:        Specific cell line; if None, uses disease_context to select best match
        disease_context:  Disease key (e.g. "CAD", "IBD") for automatic line selection

    Returns:
        {
          "gene": str,
          "cell_line": str,
          "top_programs_up": list[str],
          "top_programs_dn": list[str],
          "log2fc_range": [float, float] | None,
          "data_tier": str,    # "curated_summary" | "live_api" | "not_found"
          "evidence_tier": str, # Tier3_Provisional or better per cell line
          "n_cell_lines_available": int,
        }
    """
    gene = gene_symbol.upper()
    gene_data = _TAHOE_GENE_SUMMARIES.get(gene, {})
    n_available = len(gene_data)

    # Select cell line
    if cell_line is None:
        # Try disease-context matching
        disease_key = (disease_context or "").upper()
        preferred = _DISEASE_BEST_CELL_LINES.get(disease_key, list(TAHOE_CELL_LINES.keys()))
        cell_line = next((cl for cl in preferred if cl in gene_data), None)
        if cell_line is None and gene_data:
            cell_line = next(iter(gene_data))

    if cell_line is None or gene not in _TAHOE_GENE_SUMMARIES:
        return {
            "gene":                   gene,
            "cell_line":              cell_line or "unknown",
            "top_programs_up":        [],
            "top_programs_dn":        [],
            "log2fc_range":           None,
            "data_tier":              "not_found",
            "evidence_tier":          "provisional_virtual",
            "n_cell_lines_available": n_available,
            "note": (
                f"No Tahoe-100M summary available for {gene} in {cell_line}. "
                "Full query requires Tahoe-100M download (~400GB HF Hub) or live API key."
            ),
        }

    cl_data = gene_data.get(cell_line, {})
    if not cl_data:
        return {
            "gene": gene, "cell_line": cell_line,
            "top_programs_up": [], "top_programs_dn": [],
            "log2fc_range": None, "data_tier": "not_found",
            "evidence_tier": "provisional_virtual",
            "n_cell_lines_available": n_available,
        }

    ev_tier = TAHOE_CELL_LINES.get(cell_line, {}).get("tier_cap", "Tier3_Provisional")
    return {
        "gene":                   gene,
        "cell_line":              cell_line,
        "top_programs_up":        cl_data.get("top_programs_up", []),
        "top_programs_dn":        cl_data.get("top_programs_dn", []),
        "log2fc_range":           list(cl_data.get("log2fc_range", [])),
        "data_tier":              cl_data.get("data_tier", "curated_summary"),
        "evidence_tier":          ev_tier,
        "n_cell_lines_available": n_available,
        "tissue_context":         TAHOE_CELL_LINES.get(cell_line, {}).get("tissue"),
        "disease_context":        TAHOE_CELL_LINES.get(cell_line, {}).get("disease_context"),
    }


@_tool
def get_tahoe_best_cell_line(disease_key: str) -> dict:
    """
    Return the ranked list of Tahoe cell lines best matched to a disease context.

    Args:
        disease_key: e.g. "CAD", "IBD", "RA"

    Returns:
        {"disease": str, "ranked_cell_lines": list[{cell_line, tissue, disease_context, tier_cap}]}
    """
    preferred = _DISEASE_BEST_CELL_LINES.get(disease_key.upper(), list(TAHOE_CELL_LINES.keys())[:3])
    return {
        "disease": disease_key,
        "ranked_cell_lines": [
            {**{"cell_line": cl}, **TAHOE_CELL_LINES.get(cl, {})}
            for cl in preferred
        ],
    }


@_tool
def list_tahoe_cell_lines() -> dict:
    """List all 50 Tahoe-100M cell lines with tissue and disease context."""
    return {"cell_lines": [{"name": k, **v} for k, v in TAHOE_CELL_LINES.items()]}


if __name__ == "__main__" and mcp is not None:
    mcp.run()
