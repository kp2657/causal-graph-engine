"""
mcp_servers/reactome_server.py — Reactome pathway enrichment.

Public API:
    get_enriched_pathways(gene_list, species="9606") -> dict
        POST gene list to /AnalysisService/identifiers/
        Returns: {token, pathways: [{stId, name, pvalue, fdr, n_found, n_total}], ...}

    get_gene_pathways(gene_symbol, species="9606") -> list[dict]
        GET /ContentService/data/pathways/low/diagram/entity/{symbol}/allForms
        Returns: [{stId, name}, ...] — pathways containing this gene

    get_pathway_members_from_token(token, stid, page_size=200) -> list[str]
        GET /AnalysisService/token/{token}/pathways/{stId}/entities
        Returns gene symbols that are in BOTH the input list AND the pathway
        (used to identify which GWAS genes are in each enriched pathway)
"""
from __future__ import annotations

import time
from typing import Any

import httpx

_BASE_ANALYSIS = "https://reactome.org/AnalysisService"
_BASE_CONTENT = "https://reactome.org/ContentService"

# Small delay between requests to be polite to Reactome
_REQUEST_DELAY = 0.2

# Cache: gene_symbol → list of pathway dicts
_gene_pathway_cache: dict[str, list[dict]] = {}
# Cache: (token, stid) → list of gene symbols
_token_members_cache: dict[tuple[str, str], list[str]] = {}


def _get_client() -> httpx.Client:
    return httpx.Client(
        timeout=60.0,  # Increased to 60s for large enrichment jobs
        follow_redirects=True,
        headers={"User-Agent": "causal-graph-engine/1.0 (genomics pipeline)"},
    )


def get_enriched_pathways(
    gene_list: list[str],
    species: str = "9606",
) -> dict:
    """
    POST gene list to Reactome enrichment service.
    Handles transient 504/502 errors via robust retries.
    """
    if not gene_list:
        return {"token": None, "pathways_found": 0, "pathways": [], "error": "Empty gene list"}

    # Reactome performance degrades with > 1000 genes; prioritize by GWAS rank if needed
    # but for now we just pass the full list.
    body = "\n".join(str(g).strip() for g in gene_list[:1500] if g)

    url = f"{_BASE_ANALYSIS}/identifiers/?species={species}&pageSize=250&page=1"

    data: dict = {}
    last_error: str | None = None
    
    # Robust retry loop: 3 attempts with increasing delay
    for attempt in range(3):
        try:
            with _get_client() as client:
                resp = client.post(
                    url,
                    content=body.encode("utf-8"),
                    headers={"Content-Type": "text/plain"},
                )
                if resp.status_code in (502, 504):
                    raise httpx.HTTPStatusError(f"Server Overload {resp.status_code}", request=resp.request, response=resp)
                
                resp.raise_for_status()
                data = resp.json()
                if data.get("pathways") or data.get("summary", {}).get("token"):
                    break  # success
        except (httpx.HTTPStatusError, httpx.ReadTimeout, httpx.ConnectError) as e:
            last_error = str(e)
            # Progressive backoff: 2s, 5s
            wait = [2.0, 5.0][attempt] if attempt < 2 else 0
            if wait > 0:
                time.sleep(wait)
        except Exception as e:
            last_error = str(e)
            break

    if not data:
        return {"token": None, "pathways_found": 0, "pathways": [], "error": f"Reactome Service Failed: {last_error}"}

    time.sleep(_REQUEST_DELAY)

    token = data.get("summary", {}).get("token") or data.get("token")
    pathways_raw = data.get("pathways") or []

    pathways_out = []
    for pw in pathways_raw:
        entities = pw.get("entities", {})
        pvalue = entities.get("pValue")
        fdr = entities.get("fdr")
        n_found = entities.get("found", 0)
        n_total = entities.get("total", 0)

        # Skip if no statistics
        if pvalue is None:
            continue

        pathways_out.append({
            "stId": pw.get("stId", ""),
            "name": pw.get("name", ""),
            "pvalue": float(pvalue),
            "fdr": float(fdr) if fdr is not None else float(pvalue),
            "n_found": int(n_found),
            "n_total": int(n_total),
        })

    # Sort by FDR ascending
    pathways_out.sort(key=lambda p: p["fdr"])

    return {
        "token": token,
        "pathways_found": len(pathways_out),
        "pathways": pathways_out,
        "error": None,
    }


def get_gene_pathways(
    gene_symbol: str,
    species: str = "9606",
) -> list[dict]:
    """
    Get all Reactome pathways containing a gene.

    Uses gene SYMBOL (e.g. "CFH"), not Ensembl ID.

    Args:
        gene_symbol: Gene symbol
        species:     NCBI taxonomy ID

    Returns:
        [{stId: str, name: str}, ...]
    """
    gene_symbol = gene_symbol.strip().upper()

    if gene_symbol in _gene_pathway_cache:
        return _gene_pathway_cache[gene_symbol]

    url = (
        f"{_BASE_CONTENT}/data/pathways/low/diagram/entity/{gene_symbol}"
        f"/allForms?species={species}"
    )

    try:
        with _get_client() as client:
            resp = client.get(url)
            if resp.status_code == 404:
                _gene_pathway_cache[gene_symbol] = []
                return []
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError:
        _gene_pathway_cache[gene_symbol] = []
        return []
    except Exception:
        _gene_pathway_cache[gene_symbol] = []
        return []

    time.sleep(_REQUEST_DELAY)

    pathways = []
    if isinstance(data, list):
        for pw in data:
            stid = pw.get("stId") or pw.get("dbId") or ""
            name = pw.get("name") or pw.get("displayName") or ""
            if stid:
                pathways.append({"stId": str(stid), "name": str(name)})

    _gene_pathway_cache[gene_symbol] = pathways
    return pathways


def get_pathway_members_from_token(
    token: str,
    stid: str,
    page_size: int = 200,
) -> list[str]:
    """
    Get gene symbols found in BOTH the input gene list AND the given pathway.

    Uses the analysis token from get_enriched_pathways().

    Args:
        token:     Analysis token from Reactome enrichment result
        stid:      Pathway stable ID (e.g. "R-HSA-166658")
        page_size: Number of entities per page

    Returns:
        List of gene symbols that are members of both input list and pathway.
    """
    cache_key = (token, stid)
    if cache_key in _token_members_cache:
        return _token_members_cache[cache_key]

    if not token or not stid:
        return []

    url = (
        f"{_BASE_ANALYSIS}/token/{token}/found/all/{stid}"
        f"?resource=TOTAL&page=1&pageSize={page_size}"
    )

    try:
        with _get_client() as client:
            resp = client.get(url)
            if resp.status_code == 404:
                _token_members_cache[cache_key] = []
                return []
            resp.raise_for_status()
            data = resp.json()
    except Exception:
        # Fallback: try the older entities endpoint
        try:
            url2 = (
                f"{_BASE_ANALYSIS}/token/{token}/pathways/{stid}/entities"
                f"?resource=TOTAL&page=1&pageSize={page_size}"
            )
            with _get_client() as client:
                resp = client.get(url2)
                resp.raise_for_status()
                data = resp.json()
        except Exception:
            _token_members_cache[cache_key] = []
            return []

    time.sleep(_REQUEST_DELAY)

    genes = []
    # Handle various response shapes
    entries = data
    if isinstance(data, dict):
        entries = (
            data.get("entities", [])
            or data.get("found", [])
            or data.get("identifiers", [])
            or []
        )

    for entry in entries:
        if isinstance(entry, str):
            genes.append(entry)
        elif isinstance(entry, dict):
            # Try common fields for gene symbol
            symbol = (
                entry.get("identifier")
                or entry.get("mapsTo", [{}])[0].get("identifier") if entry.get("mapsTo") else None
                or entry.get("name")
                or entry.get("gene")
            )
            if symbol and isinstance(symbol, str):
                genes.append(symbol.strip())

    genes = [g for g in genes if g]
    _token_members_cache[cache_key] = genes
    return genes


def clear_caches() -> None:
    """Clear all in-memory caches (useful for testing)."""
    _gene_pathway_cache.clear()
    _token_members_cache.clear()
