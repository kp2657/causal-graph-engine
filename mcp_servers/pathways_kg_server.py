"""
pathways_kg_server.py — MCP server for biological pathway and knowledge graph queries.

Real implementations:
  - Reactome REST API: pathway analysis (free, no auth)
  - KEGG REST API: pathway gene sets (free for academic)
  - STRING API: protein-protein interactions (free)
  - PrimeKG: hardcoded relevant subgraph (full KG requires download)

Stubs:
  - BioPathNet: causal pathway inference (local model)
  - TxGNN: drug repurposing (see open_targets_server)

Run standalone:  python mcp_servers/pathways_kg_server.py
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
    mcp = fastmcp.FastMCP("pathways-kg-server")
    _tool = mcp.tool()
except ImportError:
    def _tool(fn=None, **_):
        return fn if fn is not None else (lambda f: f)
    mcp = None

# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

REACTOME_API  = "https://reactome.org/ContentService"
STRING_API    = "https://string-db.org/api"
KEGG_API      = "https://rest.kegg.jp"

# PrimeKG CAD-relevant subgraph (hardcoded from PrimeKG paper: Chandak 2023)
# PrimeKG integrates 20 biomedical databases into a unified KG
# Full graph: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IXA7BM
PRIMEKG_CAD_SUBGRAPH: list[dict] = [
    # Disease-gene edges (GWAS-level evidence in PrimeKG)
    {"from": "PCSK9",  "to": "CAD",         "edge_type": "disease_gene",   "source": "PrimeKG"},
    {"from": "LDLR",   "to": "CAD",         "edge_type": "disease_gene",   "source": "PrimeKG"},
    {"from": "HMGCR",  "to": "CAD",         "edge_type": "disease_gene",   "source": "PrimeKG"},
    {"from": "IL6R",   "to": "CAD",         "edge_type": "disease_gene",   "source": "PrimeKG"},
    {"from": "CXCL8",  "to": "CAD",         "edge_type": "disease_gene",   "source": "PrimeKG"},
    # Drug-target edges
    {"from": "atorvastatin", "to": "HMGCR", "edge_type": "drug_target",   "source": "PrimeKG"},
    {"from": "evolocumab",   "to": "PCSK9", "edge_type": "drug_target",   "source": "PrimeKG"},
    {"from": "tocilizumab",  "to": "IL6R",  "edge_type": "drug_target",   "source": "PrimeKG"},
    # Protein-protein interactions
    {"from": "PCSK9",  "to": "LDLR",        "edge_type": "ppi",            "source": "PrimeKG"},
    {"from": "IL6",    "to": "IL6R",        "edge_type": "ppi",            "source": "PrimeKG"},
    {"from": "DNMT3A", "to": "TET2",        "edge_type": "ppi",            "source": "PrimeKG"},
    # Pathway membership
    {"from": "PCSK9",  "to": "LDL_metabolism_pathway",  "edge_type": "pathway", "source": "Reactome"},
    {"from": "HMGCR",  "to": "cholesterol_biosynthesis", "edge_type": "pathway", "source": "Reactome"},
    {"from": "IL6R",   "to": "JAK_STAT_signaling",       "edge_type": "pathway", "source": "Reactome"},
    {"from": "TET2",   "to": "epigenetic_regulation",    "edge_type": "pathway", "source": "Reactome"},
]


# ---------------------------------------------------------------------------
# Reactome pathway tools (live)
# ---------------------------------------------------------------------------

@_tool
def get_reactome_pathways_for_gene(gene_symbol: str, species: str = "Homo sapiens") -> dict:
    """
    Return Reactome pathways containing a gene.

    Args:
        gene_symbol: Gene symbol, e.g. "PCSK9"
        species:     Species, default "Homo sapiens"
    """
    try:
        # Reactome content service: search by gene symbol
        resp = httpx.get(
            f"{REACTOME_API}/search/query",
            params={
                "query":   gene_symbol,
                "species": species,
                "types":   "Pathway",
                "cluster": True,
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])
        pathways = []
        for item in results[:20]:
            if "pathway" in item.get("typeName", "").lower():
                for entry in item.get("entries", [])[:5]:
                    pathways.append({
                        "pathway_id":   entry.get("stId"),
                        "name":         entry.get("name"),
                        "species":      entry.get("species", [species])[0] if isinstance(entry.get("species"), list) else species,
                    })
        return {
            "gene_symbol": gene_symbol,
            "n_pathways":  len(pathways),
            "pathways":    pathways,
            "source":      "Reactome ContentService",
        }
    except Exception as e:
        return {"gene_symbol": gene_symbol, "error": str(e), "pathways": []}


@_tool
def get_reactome_pathway_genes(pathway_id: str) -> dict:
    """
    Return all genes in a Reactome pathway.

    Args:
        pathway_id: Reactome stable ID, e.g. "R-HSA-191273" (cholesterol biosynthesis)
    """
    try:
        resp = httpx.get(
            f"{REACTOME_API}/data/pathway/{pathway_id}/containedEvents",
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        # Extract participant gene symbols
        genes = set()
        for event in data:
            if "input" in event:
                for inp in event.get("input", []):
                    sym = inp.get("name", "")
                    if sym:
                        genes.add(sym)
        return {
            "pathway_id": pathway_id,
            "n_genes":    len(genes),
            "genes":      sorted(genes)[:50],  # cap at 50
            "source":     "Reactome",
        }
    except Exception as e:
        return {"pathway_id": pathway_id, "error": str(e), "genes": []}


# ---------------------------------------------------------------------------
# STRING PPI tools (live)
# ---------------------------------------------------------------------------

@_tool
def get_string_interactions(genes: list[str], min_score: int = 700, organism: int = 9606) -> dict:
    """
    Query STRING for protein-protein interactions.

    Args:
        genes:     List of gene symbols
        min_score: Minimum combined score [0-1000], default 700 (high confidence)
        organism:  NCBI taxon ID, default 9606 (human)
    """
    try:
        resp = httpx.post(
            f"{STRING_API}/json/network",
            data={
                "identifiers": "\r".join(genes),
                "species":     organism,
                "required_score": min_score,
                "caller_identity": "CausalGraphEngine",
            },
            timeout=30,
        )
        resp.raise_for_status()
        interactions = resp.json()
        edges = []
        for inter in interactions:
            edges.append({
                "protein_a": inter.get("preferredName_A"),
                "protein_b": inter.get("preferredName_B"),
                "score":     inter.get("score"),
                "nscore":    inter.get("nscore"),  # neighborhood
                "fscore":    inter.get("fscore"),  # fusion
                "escore":    inter.get("escore"),  # coexpression
                "ascore":    inter.get("ascore"),  # cooccurrence
            })
        return {
            "genes":      genes,
            "min_score":  min_score,
            "n_edges":    len(edges),
            "edges":      edges,
            "source":     "STRING v12.0",
        }
    except Exception as e:
        return {"genes": genes, "error": str(e), "edges": []}


# ---------------------------------------------------------------------------
# PrimeKG subgraph (hardcoded)
# ---------------------------------------------------------------------------

@_tool
def query_primekg_subgraph(
    gene: str | None = None,
    edge_type: str | None = None,
) -> dict:
    """
    Query the PrimeKG CAD-relevant subgraph.
    Returns hardcoded edges from the pre-extracted CAD subgraph.

    Full PrimeKG download (~10GB):
    https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IXA7BM

    Args:
        gene:      Filter by gene node, e.g. "PCSK9"
        edge_type: Filter by edge type: "disease_gene" | "drug_target" | "ppi" | "pathway"
    """
    edges = PRIMEKG_CAD_SUBGRAPH

    if gene:
        gene_upper = gene.upper()
        edges = [e for e in edges if e["from"].upper() == gene_upper or e["to"].upper() == gene_upper]

    if edge_type:
        edges = [e for e in edges if e["edge_type"] == edge_type]

    return {
        "gene":        gene,
        "edge_type":   edge_type,
        "n_edges":     len(edges),
        "edges":       edges,
        "source":      "PrimeKG CAD subgraph (Chandak 2023, doi:10.1038/s41597-023-01960-3)",
        "data_tier":   "curated",
        "full_kg_url": "https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IXA7BM",
    }


@_tool
def run_bioPathNet_causal_inference(
    gene_list: list[str],
    disease: str,
) -> dict:
    """
    Run BioPathNet causal pathway inference from gene list to disease.

    STUB — requires:
      1. BioPathNet model (Yang et al. 2023, GitHub: snap-stanford/BioPathNet)
      2. Full PrimeKG input graph
      3. GPU recommended

    Returns schema-valid stub.
    """
    return {
        "gene_list":   gene_list,
        "disease":     disease,
        "causal_paths": None,
        "confidence":  None,
        "note":        "STUB — implement after PrimeKG download and BioPathNet setup",
        "paper":       "Yang 2023 doi:10.1101/2023.11.18.567645",
        "github":      "https://github.com/snap-stanford/BioPathNet",
    }


@_tool
def get_kegg_pathway_genes(pathway_id: str) -> dict:
    """
    Return genes in a KEGG pathway.

    Args:
        pathway_id: KEGG pathway ID, e.g. "hsa04975" (fat digestion and absorption)
    """
    try:
        resp = httpx.get(f"{KEGG_API}/get/{pathway_id}", timeout=30)
        resp.raise_for_status()
        text = resp.text
        # Parse GENE section from flat file
        genes = []
        in_gene_section = False
        for line in text.split("\n"):
            if line.startswith("GENE"):
                in_gene_section = True
            elif in_gene_section and line.startswith(" "):
                parts = line.strip().split()
                if len(parts) >= 2:
                    genes.append(parts[1].rstrip(";"))
            elif in_gene_section and not line.startswith(" "):
                break

        return {
            "pathway_id": pathway_id,
            "n_genes":    len(genes),
            "genes":      genes[:100],
            "source":     "KEGG REST API",
        }
    except Exception as e:
        return {"pathway_id": pathway_id, "error": str(e), "genes": []}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if mcp is None:
        raise RuntimeError("fastmcp required: pip install fastmcp")
    mcp.run()
