"""
finngen_server.py — MCP server for FinnGen R10 live REST API.

FinnGen is a Finnish biobank GWAS study with ~520,000 participants and
2,272 disease endpoints in R10. Unlike most biobanks, FinnGen provides a
live REST API for querying GWAS results without data download.

Key advantages:
  - Finnish LD structure catches distinct haplotypes (NOD2, IL23R rare variants)
  - High power for rare functional variants (founder effects)
  - Covers 2,272 endpoints including disease subtypes not in GWAS Catalog
  - Complements large-scale meta-GWAS by providing replication in a distinct
    ancestry/LD background

API base: https://r10.api.finngen.fi/api

Endpoints used:
  GET /phenos                                → all phenotypes with metadata
  GET /phenos/{phenocode}                    → phenotype metadata
  GET /variants?phenocode={p}&p_threshold={t} → top GWAS hits
  GET /variants?phenocode={p}&gene={g}       → gene-region hits

Run standalone:  python mcp_servers/finngen_server.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx

from pipelines.api_cache import api_cached

try:
    import fastmcp
    mcp = fastmcp.FastMCP("finngen-server")
    _tool = mcp.tool()
except ImportError:
    def _tool(fn=None, **_):
        return fn if fn is not None else (lambda f: f)
    mcp = None

FINNGEN_API = "https://r10.api.finngen.fi/api"
_DELAY = 0.3   # polite rate limit

# EFO → FinnGen phenocode mapping (R10 endpoint names)
EFO_TO_FINNGEN: dict[str, str] = {
    "EFO_0001645": "I9_CAD",
    "EFO_0003144": "I9_HEARTFAIL",
    "EFO_0000275": "I9_AF",
    "EFO_0000685": "M13_RHEUMA",
    "EFO_0002690": "M13_SLE",
    "EFO_0003885": "G6_MS",
    "EFO_0001359": "E4_DM1",
    "EFO_0001360": "E4_DM2",
    "EFO_0003767": "K11_IBD_STRICT",
    "EFO_0000384": "K11_CROHNS",
    "EFO_0000729": "K11_ULCER_COLI",
    "EFO_0000249": "G6_AD",
    "EFO_0000341": "J10_COPD",
    "EFO_0004611": "LDLC",    # LDL cholesterol
    "EFO_0004612": "HDLC",    # HDL cholesterol
    "EFO_0004530": "TRIG",    # Triglycerides
}


@_tool
@api_cached(ttl_days=30)
def get_finngen_phenotype_info(phenocode: str) -> dict:
    """Get FinnGen R10 phenotype metadata (n_cases, n_controls, category). Also accepts EFO ID — will auto-resolve to FinnGen phenocode via EFO_TO_FINNGEN."""
    if phenocode.startswith("EFO_"):
        phenocode = EFO_TO_FINNGEN.get(phenocode, phenocode)

    time.sleep(_DELAY)
    try:
        resp = httpx.get(f"{FINNGEN_API}/phenos/{phenocode}", timeout=20)
        if resp.status_code == 404:
            return {
                "phenocode": phenocode,
                "error": f"Phenocode '{phenocode}' not found in FinnGen R10",
                "note": "Check https://risteys.finngen.fi",
            }
        resp.raise_for_status()
        data = resp.json()
        n_cases = data.get("num_cases") or data.get("n_cases") or 0
        n_controls = data.get("num_controls") or data.get("n_controls") or 0
        return {
            "phenocode": phenocode,
            "name": data.get("name"),
            "category": data.get("category"),
            "n_cases": n_cases,
            "n_controls": n_controls,
            "n_total": n_cases + n_controls,
            "data_source": "FinnGen R10",
            "browser_url": f"https://r10.risteys.finngen.fi/phenocode/{phenocode}",
        }
    except Exception as exc:
        return {
            "phenocode": phenocode,
            "error": str(exc),
            "note": "Check https://risteys.finngen.fi",
        }


@_tool
@api_cached(ttl_days=30)
def get_finngen_top_variants(
    phenocode: str,
    p_threshold: float = 5e-8,
    n_max: int = 100,
) -> dict:
    """Retrieve genome-wide significant variants for a FinnGen phenotype. Returns GWAS hits suitable as MR instruments. Also accepts EFO ID."""
    if phenocode.startswith("EFO_"):
        phenocode = EFO_TO_FINNGEN.get(phenocode, phenocode)

    try:
        time.sleep(_DELAY)
        resp = httpx.get(
            f"{FINNGEN_API}/variants",
            params={"phenocode": phenocode},
            timeout=httpx.Timeout(connect=5.0, read=20.0, write=5.0, pool=5.0),
        )
        resp.raise_for_status()
        all_variants = resp.json()

        filtered = [
            v for v in all_variants
            if float(v.get("pval", 1.0)) <= p_threshold
        ]
        filtered.sort(key=lambda v: float(v.get("pval", 1.0)))
        filtered = filtered[:n_max]

        variants = [
            {
                "rsids": v.get("rsids"),
                "chromosome": v.get("chr") or v.get("chromosome"),
                "position": v.get("pos") or v.get("position"),
                "ref": v.get("ref"),
                "alt": v.get("alt"),
                "pval": float(v.get("pval", 1.0)),
                "beta": v.get("beta"),
                "sebeta": v.get("sebeta"),
                "maf": v.get("maf"),
                "nearest_gene": v.get("nearest_genes") or v.get("gene_most_severe"),
            }
            for v in filtered
        ]

        return {
            "phenocode": phenocode,
            "p_threshold": p_threshold,
            "n_significant": len(filtered),
            "variants": variants,
            "data_source": "FinnGen R10 REST API",
            "note": "beta in log-OR units for binary traits",
        }
    except Exception as exc:
        return {"phenocode": phenocode, "variants": [], "error": str(exc)}


@_tool
@api_cached(ttl_days=30)
def get_finngen_gene_associations(
    phenocode: str,
    gene: str,
    p_threshold: float = 1e-4,
) -> dict:
    """Retrieve FinnGen variants in the region of a specific gene. Useful for gene-level MR instrument selection. Uses a relaxed p_threshold (default 1e-4) to capture cis instruments."""
    if phenocode.startswith("EFO_"):
        phenocode = EFO_TO_FINNGEN.get(phenocode, phenocode)

    try:
        time.sleep(_DELAY)
        resp = httpx.get(
            f"{FINNGEN_API}/variants",
            params={"phenocode": phenocode, "gene": gene},
            timeout=httpx.Timeout(connect=5.0, read=20.0, write=5.0, pool=5.0),
        )
        resp.raise_for_status()
        all_variants = resp.json()

        filtered = [
            v for v in all_variants
            if float(v.get("pval", 1.0)) <= p_threshold
        ]
        filtered.sort(key=lambda v: float(v.get("pval", 1.0)))

        variants = [
            {
                "rsids": v.get("rsids"),
                "chromosome": v.get("chr") or v.get("chromosome"),
                "position": v.get("pos") or v.get("position"),
                "ref": v.get("ref"),
                "alt": v.get("alt"),
                "pval": float(v.get("pval", 1.0)),
                "beta": v.get("beta"),
                "sebeta": v.get("sebeta"),
                "maf": v.get("maf"),
                "nearest_gene": v.get("nearest_genes") or v.get("gene_most_severe"),
            }
            for v in filtered
        ]

        return {
            "phenocode": phenocode,
            "gene": gene,
            "p_threshold": p_threshold,
            "n_variants": len(filtered),
            "variants": variants,
            "data_source": "FinnGen R10 REST API",
        }
    except Exception as exc:
        return {"phenocode": phenocode, "gene": gene, "variants": [], "error": str(exc)}


@_tool
def list_finngen_phenotypes(
    search_query: str = "",
    category: str = "",
    n_max: int = 50,
) -> dict:
    """List available FinnGen R10 phenotypes. Filter by keyword or disease category. Useful for discovering relevant endpoints for a new disease."""
    try:
        time.sleep(_DELAY)
        resp = httpx.get(f"{FINNGEN_API}/phenos", timeout=httpx.Timeout(connect=5.0, read=20.0, write=5.0, pool=5.0))
        resp.raise_for_status()
        all_phenos = resp.json()

        result = all_phenos
        if search_query:
            result = [
                p for p in result
                if search_query.lower() in str(p.get("name", "")).lower()
                or search_query.lower() in str(p.get("phenocode", "")).lower()
            ]
        if category:
            result = [
                p for p in result
                if category.lower() in str(p.get("category", "")).lower()
            ]

        result = result[:n_max]
        phenotypes = [
            {
                "phenocode": p.get("phenocode"),
                "name": p.get("name"),
                "category": p.get("category"),
                "n_cases": p.get("num_cases"),
                "n_controls": p.get("num_controls"),
            }
            for p in result
        ]

        return {
            "total_phenotypes": len(all_phenos),
            "returned": len(phenotypes),
            "phenotypes": phenotypes,
            "data_source": "FinnGen R10",
        }
    except Exception as exc:
        return {"phenotypes": [], "error": str(exc)}


@api_cached(ttl_days=90)
def efo_to_finngen_phenocode(efo_id: str) -> str | None:
    """Helper: resolve EFO ID to FinnGen phenocode, or None if not mapped."""
    return EFO_TO_FINNGEN.get(efo_id)


if __name__ == "__main__":
    if mcp:
        mcp.run()
