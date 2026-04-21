"""
mcp_servers/phenoscanner_server.py — PhenoScanner pQTL lookup.

PhenoScanner indexes deCODE Ferkingstad2021, INTERVAL, and UKB-PPP pQTL data.

Public API:
    get_pqtl_for_gene(gene, p_threshold=0.01) -> dict
        Query by gene symbol. Returns top pQTL hits.

    get_pqtl_for_snp(rsid, p_threshold=0.01) -> dict
        Query by rsID (for known coding variants).

    get_best_pqtl_instrument(gene) -> dict | None
        Returns the top pQTL hit for a gene or None if not found.
"""
from __future__ import annotations

import time
from typing import Any

import httpx

_BASE_URL = "http://www.phenoscanner.medschl.cam.ac.uk/api/"
_REQUEST_DELAY = 0.5  # PhenoScanner is rate-limited
_TIMEOUT = 8.0


def _make_request(params: dict) -> dict | None:
    """Make a PhenoScanner API request with timeout and error handling."""
    try:
        with httpx.Client(timeout=_TIMEOUT, follow_redirects=True) as client:
            resp = client.get(_BASE_URL, params=params)
            resp.raise_for_status()
            return resp.json()
    except httpx.TimeoutException:
        return {"error": "PhenoScanner API timeout (8s)", "results": []}
    except httpx.HTTPStatusError as e:
        return {"error": f"HTTP {e.response.status_code}", "results": []}
    except Exception as e:
        return {"error": str(e), "results": []}


def _parse_pqtl_results(data: dict | None, p_threshold: float) -> list[dict]:
    """Extract and filter pQTL results from API response."""
    if data is None or "error" in data:
        return []

    results = data.get("results", []) or data.get("associations", []) or []
    hits = []
    for r in results:
        try:
            p_val = float(r.get("p", r.get("pvalue", 1.0)) or 1.0)
        except (TypeError, ValueError):
            continue
        if p_val > p_threshold:
            continue

        hits.append({
            "rsid": r.get("snp") or r.get("rsid", ""),
            "gene": r.get("gene", ""),
            "protein": r.get("trait") or r.get("protein", ""),
            "beta": _safe_float(r.get("beta")),
            "se": _safe_float(r.get("se")),
            "pvalue": p_val,
            "study": r.get("study") or r.get("dataset", "PhenoScanner"),
            "tissue": r.get("tissue", "blood"),
            "chr": r.get("chr", ""),
            "pos": r.get("pos", ""),
            "ref": r.get("ref", ""),
            "alt": r.get("alt", ""),
        })

    # Sort by p-value
    hits.sort(key=lambda x: x["pvalue"])
    return hits


def _safe_float(val: Any) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def get_pqtl_for_gene(
    gene: str,
    p_threshold: float = 0.01,
) -> dict:
    """
    Query PhenoScanner pQTL data by gene symbol.

    Args:
        gene:        Gene symbol (e.g. "CFH")
        p_threshold: Maximum p-value to include

    Returns:
        {
            "gene": str,
            "n_hits": int,
            "pqtl_hits": [{rsid, gene, protein, beta, se, pvalue, study, ...}],
            "error": str | None,
        }
    """
    from pipelines.api_cache import get_cache
    return get_cache().get_or_set(
        "get_pqtl_for_gene", (gene, p_threshold), {},
        lambda: _get_pqtl_for_gene_live(gene, p_threshold),
        ttl_days=30,
    )


def _get_pqtl_for_gene_live(
    gene: str,
    p_threshold: float = 0.01,
) -> dict:
    params = {
        "query": gene,
        "catalogue": "pQTL",
        "p": str(p_threshold),
        "proxies": "None",
        "r2": "0.8",
        "build": "37",
    }

    data = _make_request(params)
    time.sleep(_REQUEST_DELAY)

    error = data.get("error") if data else "No response"
    hits = _parse_pqtl_results(data, p_threshold)

    return {
        "gene": gene,
        "n_hits": len(hits),
        "pqtl_hits": hits,
        "error": error if not hits else None,
    }


def get_pqtl_for_snp(
    rsid: str,
    p_threshold: float = 0.01,
) -> dict:
    """
    Query PhenoScanner pQTL data by rsID.

    Useful for known coding variants like CFH Y402H (rs1061170),
    TREM2 R47H (rs75932628), APOE ε4 (rs429358).

    Args:
        rsid:        rsID (e.g. "rs1061170")
        p_threshold: Maximum p-value to include

    Returns:
        {
            "rsid": str,
            "n_hits": int,
            "pqtl_hits": [{rsid, gene, protein, beta, se, pvalue, study, ...}],
            "error": str | None,
        }
    """
    params = {
        "query": rsid,
        "catalogue": "pQTL",
        "p": str(p_threshold),
        "proxies": "None",
        "r2": "0.8",
        "build": "37",
    }

    data = _make_request(params)
    time.sleep(_REQUEST_DELAY)

    error = data.get("error") if data else "No response"
    hits = _parse_pqtl_results(data, p_threshold)

    return {
        "rsid": rsid,
        "n_hits": len(hits),
        "pqtl_hits": hits,
        "error": error if not hits else None,
    }


def get_best_pqtl_instrument(gene: str) -> dict | None:
    """
    Get the top pQTL instrument for a gene.

    Returns the single best (lowest p-value) pQTL hit, or None if none found.

    Args:
        gene: Gene symbol

    Returns:
        {rsid, gene, protein, beta, se, pvalue, study, tissue} or None
    """
    result = get_pqtl_for_gene(gene, p_threshold=1e-5)
    hits = result.get("pqtl_hits", [])
    if not hits:
        return None
    return hits[0]
