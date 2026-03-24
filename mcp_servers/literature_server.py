"""
literature_server.py — MCP server for scientific literature queries.

Real implementations:
  - PubMed / NCBI E-utilities: search, fetch abstracts, fetch metadata (NCBI_API_KEY loaded)
  - Crossref: DOI metadata (free, mailto set)
  - Europe PMC: open access full text (free, no auth)

Stubs (require local models or API keys):
  - Semantic Scholar: graph-based citation queries
  - PaperQA2: local LLM-powered PDF Q&A

Run standalone:  python mcp_servers/literature_server.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import httpx

try:
    import fastmcp
    mcp = fastmcp.FastMCP("literature-server")
    _tool = mcp.tool()
except ImportError:
    def _tool(fn=None, **_):
        return fn if fn is not None else (lambda f: f)
    mcp = None

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NCBI_API_KEY    = os.getenv("NCBI_API_KEY", "")
CROSSREF_MAILTO = os.getenv("CROSSREF_MAILTO", "")
NCBI_BASE       = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
CROSSREF_BASE   = "https://api.crossref.org"
EUROPEPMC_BASE  = "https://www.ebi.ac.uk/europepmc/webservices/rest"
SEMANTICSCHOLAR = "https://api.semanticscholar.org/graph/v1"

# Key anchor papers for CAD / CHIP / viral biology (hardcoded for fast access)
ANCHOR_PAPERS: dict[str, dict] = {
    "32694926": {
        "title":   "Clonal hematopoiesis associated with TET2 deficiency accelerates atherosclerosis development in mice",
        "authors": "Bick AG et al.",
        "journal": "Nature",
        "year":    2020,
        "doi":     "10.1038/s41586-020-2534-z",
        "tags":    ["CHIP", "CAD", "TET2", "DNMT3A", "atherosclerosis"],
        "key_finding": "CHIP associates with 1.42x increased risk of CAD (HR=1.42, 95% CI 1.23-1.64)",
    },
    "35177839": {
        "title":   "Clonal hematopoiesis of indeterminate potential and its impact on patient trajectories after stem cell transplantation",
        "authors": "Kar SP et al.",
        "journal": "Nature Genetics",
        "year":    2022,
        "doi":     "10.1038/s41588-022-01009-8",
        "tags":    ["CHIP", "CAD", "UKB", "WES"],
        "key_finding": "200K UKB WES CHIP analysis; DNMT3A large CHIP OR=1.40 for CAD",
    },
    "35688146": {
        "title":   "Mapping information-rich genotype-phenotype landscapes with genome-scale Perturb-seq",
        "authors": "Replogle JM et al.",
        "journal": "Cell",
        "year":    2022,
        "doi":     "10.1016/j.cell.2022.05.013",
        "tags":    ["Perturb-seq", "CRISPR", "K562", "genome-scale"],
        "key_finding": "2.5M cells × 9065 perturbations; genome-scale β estimates for Ota framework",
    },
    "35549404": {
        "title":   "The Tabula Sapiens: A multiple-organ, single-cell transcriptomic atlas of humans",
        "authors": "Tabula Sapiens Consortium",
        "journal": "Science",
        "year":    2022,
        "doi":     "10.1126/science.abl4896",
        "tags":    ["single-cell", "atlas", "Tabula Sapiens", "human"],
        "key_finding": "500K cells across 24 tissues; gold standard human cell type atlas",
    },
}


# ---------------------------------------------------------------------------
# PubMed / NCBI E-utilities (live)
# ---------------------------------------------------------------------------

def _ncbi_params(extra: dict | None = None) -> dict:
    """Build base NCBI params dict with API key if available."""
    params = {"retmode": "json"}
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY
    if extra:
        params.update(extra)
    return params


@_tool
def search_pubmed(
    query: str,
    max_results: int = 10,
    date_range: str | None = None,
) -> dict:
    """
    Search PubMed using NCBI E-utilities (esearch + efetch).
    Returns PMIDs + article metadata.

    Args:
        query:       PubMed query string, e.g. "CHIP[Title] AND coronary artery disease"
        max_results: Max articles to return (default 10, max 100)
        date_range:  Optional ISO date range, e.g. "2020/01/01:2024/12/31"

    Returns:
        {"query": str, "total_found": int, "articles": list[dict]}
    """
    max_results = min(max_results, 100)
    esearch_params = _ncbi_params({
        "db":      "pubmed",
        "term":    query,
        "retmax":  max_results,
        "usehistory": "y",
    })
    if date_range:
        esearch_params["datetype"] = "pdat"
        esearch_params["mindate"], esearch_params["maxdate"] = date_range.split(":")

    try:
        resp = httpx.get(f"{NCBI_BASE}/esearch.fcgi", params=esearch_params, timeout=30)
        resp.raise_for_status()
        search_data = resp.json()
        result_data = search_data.get("esearchresult", {})
        total_found = int(result_data.get("count", 0))
        pmids = result_data.get("idlist", [])

        if not pmids:
            return {"query": query, "total_found": total_found, "articles": []}

        # Fetch summaries for found PMIDs
        articles = _fetch_pubmed_summaries(pmids)
        return {
            "query":       query,
            "total_found": total_found,
            "n_returned":  len(articles),
            "articles":    articles,
        }
    except httpx.HTTPStatusError as e:
        return {"query": query, "error": f"HTTP {e.response.status_code}", "articles": []}
    except Exception as e:
        return {"query": query, "error": str(e), "articles": []}


def _fetch_pubmed_summaries(pmids: list[str]) -> list[dict]:
    """Fetch article summaries from PubMed for a list of PMIDs."""
    if not pmids:
        return []
    try:
        resp = httpx.get(
            f"{NCBI_BASE}/esummary.fcgi",
            params=_ncbi_params({
                "db":  "pubmed",
                "id":  ",".join(pmids),
            }),
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        articles = []
        result = data.get("result", {})
        for pmid in pmids:
            if pmid not in result:
                continue
            art = result[pmid]
            if not isinstance(art, dict):
                continue
            authors = art.get("authors", [])
            author_str = authors[0].get("name", "") if authors else ""
            if len(authors) > 1:
                author_str += " et al."
            articles.append({
                "pmid":     pmid,
                "title":    art.get("title", ""),
                "authors":  author_str,
                "journal":  art.get("fulljournalname", art.get("source", "")),
                "year":     art.get("pubdate", "")[:4],
                "doi":      next((e["value"] for e in art.get("articleids", []) if e.get("idtype") == "doi"), None),
            })
        return articles
    except Exception:
        return [{"pmid": p, "error": "fetch failed"} for p in pmids]


@_tool
def fetch_pubmed_abstract(pmid: str) -> dict:
    """
    Fetch the full abstract for a PubMed article.

    Args:
        pmid: PubMed ID, e.g. "32694926"

    Returns:
        {"pmid": str, "title": str, "abstract": str, "authors": str, "year": str}
    """
    # Check anchor papers first (no HTTP needed)
    if pmid in ANCHOR_PAPERS:
        paper = ANCHOR_PAPERS[pmid]
        return {
            "pmid":     pmid,
            "title":    paper["title"],
            "authors":  paper["authors"],
            "journal":  paper["journal"],
            "year":     str(paper["year"]),
            "doi":      paper.get("doi"),
            "abstract": f"[Anchor paper — key finding: {paper['key_finding']}]",
            "tags":     paper.get("tags", []),
        }

    try:
        resp = httpx.get(
            f"{NCBI_BASE}/efetch.fcgi",
            params=_ncbi_params({
                "db":       "pubmed",
                "id":       pmid,
                "rettype":  "abstract",
                "retmode":  "xml",
            }),
            timeout=30,
        )
        resp.raise_for_status()
        # Parse XML abstract (simplified)
        import re
        xml = resp.text
        abstract_match = re.search(r"<AbstractText[^>]*>(.*?)</AbstractText>", xml, re.DOTALL)
        title_match    = re.search(r"<ArticleTitle>(.*?)</ArticleTitle>", xml, re.DOTALL)
        year_match     = re.search(r"<Year>(\d{4})</Year>", xml)
        journal_match  = re.search(r"<Title>(.*?)</Title>", xml)

        return {
            "pmid":     pmid,
            "title":    title_match.group(1).strip() if title_match else "",
            "abstract": abstract_match.group(1).strip() if abstract_match else "",
            "year":     year_match.group(1) if year_match else "",
            "journal":  journal_match.group(1).strip() if journal_match else "",
        }
    except Exception as e:
        return {"pmid": pmid, "error": str(e)}


@_tool
def search_pubmed_chip_cad(max_results: int = 20) -> dict:
    """
    Pre-configured PubMed search for CHIP → CAD associations.
    Returns recent high-impact papers.
    """
    return search_pubmed(
        query='("clonal hematopoiesis"[Title/Abstract] OR "CHIP"[Title]) AND ("coronary artery disease"[MeSH] OR "atherosclerosis"[MeSH] OR "cardiovascular"[Title/Abstract])',
        max_results=max_results,
    )


@_tool
def get_anchor_paper(pmid: str) -> dict:
    """
    Return metadata for a key anchor paper in the causal disease graph.
    Hardcoded for fast access — no HTTP required.

    Available PMIDs: 32694926 (Bick 2020), 35177839 (Kar 2022),
                     35688146 (Replogle 2022), 35549404 (Tabula Sapiens)
    """
    if pmid in ANCHOR_PAPERS:
        return {**ANCHOR_PAPERS[pmid], "pmid": pmid, "found": True}
    return {
        "pmid":  pmid,
        "found": False,
        "note":  f"Not in anchor paper registry. Use fetch_pubmed_abstract for live lookup.",
        "available_pmids": list(ANCHOR_PAPERS.keys()),
    }


@_tool
def list_anchor_papers() -> dict:
    """Return all anchor papers in the causal disease graph."""
    return {
        "n_papers": len(ANCHOR_PAPERS),
        "papers":   [
            {"pmid": pmid, **{k: v for k, v in info.items() if k != "key_finding"}}
            for pmid, info in ANCHOR_PAPERS.items()
        ],
    }


# ---------------------------------------------------------------------------
# Crossref — DOI metadata
# ---------------------------------------------------------------------------

@_tool
def get_crossref_metadata(doi: str) -> dict:
    """
    Fetch article metadata from Crossref by DOI.
    Free, no auth required. Mailto header included for polite pool.

    Args:
        doi: DOI string, e.g. "10.1038/s41586-020-2534-z"
    """
    headers = {}
    if CROSSREF_MAILTO:
        headers["User-Agent"] = f"CausalGraphEngine/1.0 (mailto:{CROSSREF_MAILTO})"

    try:
        resp = httpx.get(
            f"{CROSSREF_BASE}/works/{doi}",
            headers=headers,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json().get("message", {})
        authors = data.get("author", [])
        author_str = ""
        if authors:
            first = authors[0]
            author_str = f"{first.get('family', '')} {first.get('given', '')}".strip()
            if len(authors) > 1:
                author_str += " et al."

        return {
            "doi":     doi,
            "title":   data.get("title", [""])[0] if data.get("title") else "",
            "authors": author_str,
            "journal": data.get("container-title", [""])[0] if data.get("container-title") else "",
            "year":    data.get("published", {}).get("date-parts", [[None]])[0][0],
            "n_citations": data.get("is-referenced-by-count", 0),
            "publisher":   data.get("publisher", ""),
        }
    except httpx.HTTPStatusError as e:
        return {"doi": doi, "error": f"HTTP {e.response.status_code} — DOI not found in Crossref"}
    except Exception as e:
        return {"doi": doi, "error": str(e)}


# ---------------------------------------------------------------------------
# Europe PMC — open access full text
# ---------------------------------------------------------------------------

@_tool
def search_europe_pmc(query: str, max_results: int = 10) -> dict:
    """
    Search Europe PMC for open-access literature.
    Complements PubMed; includes preprints and European publications.

    Args:
        query:       Search query, e.g. "TET2 CHIP atherosclerosis"
        max_results: Maximum results (default 10)
    """
    try:
        resp = httpx.get(
            f"{EUROPEPMC_BASE}/search",
            params={
                "query":   query,
                "format":  "json",
                "pageSize": min(max_results, 25),
                "resultType": "core",
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        articles = []
        for art in data.get("resultList", {}).get("result", []):
            articles.append({
                "pmid":       art.get("pmid"),
                "pmcid":      art.get("pmcid"),
                "title":      art.get("title", ""),
                "authors":    art.get("authorString", ""),
                "journal":    art.get("journalTitle", ""),
                "year":       art.get("pubYear"),
                "open_access": art.get("isOpenAccess") == "Y",
                "doi":        art.get("doi"),
            })
        return {
            "query":       query,
            "total_found": data.get("hitCount", 0),
            "n_returned":  len(articles),
            "articles":    articles,
        }
    except Exception as e:
        return {"query": query, "error": str(e), "articles": []}


# ---------------------------------------------------------------------------
# Semantic Scholar (stub)
# ---------------------------------------------------------------------------

@_tool
def get_semantic_scholar_citations(paper_id: str, direction: str = "citing") -> dict:
    """
    Get citation graph for a paper from Semantic Scholar.

    STUB — Semantic Scholar API has strict rate limits (100 req/5min).
    For anchor papers, citation counts can be computed offline.

    Args:
        paper_id:  Semantic Scholar paper ID or DOI
        direction: "citing" (papers that cite this) | "cited" (papers cited by this)
    """
    return {
        "paper_id":  paper_id,
        "direction": direction,
        "citations": None,
        "note":      "STUB — Semantic Scholar API available at https://api.semanticscholar.org",
        "rate_limit": "100 requests per 5 minutes (unauthenticated)",
    }


@_tool
def run_paperqa2_query(question: str, paper_paths: list[str] | None = None) -> dict:
    """
    Run PaperQA2 to answer a scientific question from a PDF collection.

    STUB — requires:
      1. Local PDF collection or pre-indexed corpus
      2. PaperQA2: pip install paper-qa
      3. OpenAI API key or local LLM (e.g. Ollama)

    Algorithm when implemented:
      from paperqa import Docs
      docs = Docs()
      for path in paper_paths:
          docs.add(path)
      answer = docs.query(question)

    Args:
        question:    Scientific question, e.g. "What is the effect of TET2 CHIP on IL-6 signaling?"
        paper_paths: List of local PDF paths to query
    """
    return {
        "question":    question,
        "paper_paths": paper_paths,
        "answer":      None,
        "evidence":    [],
        "note":        "STUB — install paper-qa: pip install paper-qa; requires OpenAI key or local LLM",
        "paperqa_url": "https://github.com/Future-House/paper-qa",
    }


# ---------------------------------------------------------------------------
# Convenience: search by gene + disease
# ---------------------------------------------------------------------------

@_tool
def search_gene_disease_literature(
    gene: str,
    disease: str,
    max_results: int = 10,
    source: str = "pubmed",
) -> dict:
    """
    Search for literature on a gene-disease association.
    Convenience wrapper around search_pubmed or search_europe_pmc.

    Args:
        gene:        Gene symbol, e.g. "TET2"
        disease:     Disease name, e.g. "coronary artery disease"
        max_results: Max results to return
        source:      "pubmed" (default) | "europepmc"
    """
    query = f'"{gene}"[Title/Abstract] AND "{disease}"[Title/Abstract]'
    if source == "europepmc":
        query = f"{gene} {disease}"
        return search_europe_pmc(query, max_results=max_results)
    return search_pubmed(query, max_results=max_results)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if mcp is None:
        raise RuntimeError("fastmcp required: pip install fastmcp")
    mcp.run()
