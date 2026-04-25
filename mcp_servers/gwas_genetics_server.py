"""
gwas_genetics_server.py — MCP server for GWAS, genetics, and regulatory genomics tools.

Real API integrations (no auth required):
  - GWAS Catalog REST API (EBI) — associations, studies, SNP lookups
  - gnomAD GraphQL — LoF constraint (pLI, LOEUF)
  - GTEx v8 API — eQTL associations per tissue

Optional / mixed:
  - IEU OpenGWAS — requires JWT (since May 2024). Set OPENGWAS_JWT; without it,
    OpenGWAS tools return structured empty results (not silent failure).
  - FinnGen R12 gene-burden — **live HTTP** from FinnGen public GCS burdentest TSVs
    when reachable; phenotype metadata uses live FinnGen API with static fallbacks.
Still stubbed or simplified:
  - SuSiE-RSS, HyPrColoc, some LDSC paths, ABC model, Enformer — compute-heavy or not wired

Run standalone:  python mcp_servers/gwas_genetics_server.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env before reading any env vars
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import concurrent.futures as _futures

import httpx
import json

from pipelines.api_cache import api_cached as _api_cached

# Hard wall-clock limit for every EBI GWAS Catalog HTTP call.
# httpx per-read timeout doesn't prevent hangs when the server streams large
# responses in small chunks each arriving within the read window.
# _gwas_get() wraps httpx.get in a thread so future.result(timeout=) enforces
# a total-request deadline regardless of chunking behaviour.
_GWAS_HARD_TIMEOUT = 8.0  # seconds, total per request


def _gwas_get(url: str, **kwargs) -> httpx.Response:
    """httpx.get with a hard wall-clock timeout (not per-read).

    IMPORTANT: uses shutdown(wait=False) so that a timed-out HTTP thread is
    abandoned rather than blocking the caller until EBI finishes streaming.
    The abandoned thread runs to completion in the background (daemon thread)
    but does not prevent the pipeline from progressing.
    """
    ex = _futures.ThreadPoolExecutor(max_workers=1)
    fut = ex.submit(httpx.get, url, **kwargs)
    try:
        return fut.result(timeout=_GWAS_HARD_TIMEOUT)
    except _futures.TimeoutError:
        raise httpx.ReadTimeout(
            f"Hard timeout ({_GWAS_HARD_TIMEOUT}s) exceeded for {url}"
        )
    finally:
        ex.shutdown(wait=False)  # abandon stuck thread; don't block the caller


_GWAS_HARDCOPY_DIR = Path(__file__).parent.parent / "data" / "gwas_hardcopy"
_GWAS_HARDCOPY_DIR.mkdir(parents=True, exist_ok=True)


def _gwas_hardcopy_path(efo_id: str, kind: str) -> Path:
    safe = (efo_id or "unknown").strip().replace("/", "_")
    return _GWAS_HARDCOPY_DIR / f"{safe}__{kind}.json"


def _write_hardcopy(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def _read_hardcopy(path: Path) -> dict | None:
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


try:
    import fastmcp
    mcp = fastmcp.FastMCP("gwas-genetics-server")
    _tool = mcp.tool()
except ImportError:
    def _tool(fn=None, **_):
        return fn if fn is not None else (lambda f: f)
    mcp = None

# ---------------------------------------------------------------------------
# API constants
# ---------------------------------------------------------------------------

GWAS_CATALOG_BASE   = "https://www.ebi.ac.uk/gwas/rest/api"
# Newer GWAS Catalog REST API (v2). Legacy endpoints occasionally return 404.
GWAS_CATALOG_BASE_V2 = "https://www.ebi.ac.uk/gwas/rest/api/v2"
GNOMAD_API          = "https://gnomad.broadinstitute.org/api"
GTEX_API            = "https://gtexportal.org/api/v2"
OPENGWAS_API        = "https://api.opengwas.io/api"
OT_PLATFORM_GQL     = "https://api.platform.opentargets.org/api/v4/graphql"

CROSSREF_MAILTO = os.getenv("CROSSREF_MAILTO", "")
OPENGWAS_JWT    = os.getenv("OPENGWAS_JWT", "")   # optional — enables IEU Open GWAS tools

# Polite rate-limit: GWAS Catalog asks for ~1 req/sec without API key
_GWAS_CATALOG_DELAY = 0.5


def _gwas_headers() -> dict:
    return {"User-Agent": f"causal-graph-engine/0.1 mailto:{CROSSREF_MAILTO}"}


def _opengwas_headers() -> dict:
    h = {"User-Agent": "causal-graph-engine/0.1"}
    if OPENGWAS_JWT:
        h["Authorization"] = f"Bearer {OPENGWAS_JWT}"
    return h


def _ot_gql(query: str, variables: dict | None = None) -> dict:
    """Execute a GraphQL query against OT Platform v4. Returns data payload or {error: ...}."""
    try:
        resp = httpx.post(
            OT_PLATFORM_GQL,
            json={"query": query, "variables": variables or {}},
            headers={"Content-Type": "application/json"},
            timeout=httpx.Timeout(connect=5.0, read=20.0, write=5.0, pool=5.0),
        )
        resp.raise_for_status()
        result = resp.json()
        if "errors" in result:
            return {"error": result["errors"][0].get("message", str(result["errors"]))}
        return result.get("data", {})
    except httpx.HTTPStatusError as e:
        return {"error": f"HTTP {e.response.status_code}"}
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# GWAS Catalog — real implementations
# ---------------------------------------------------------------------------

@_tool
def get_gwas_catalog_associations(
    efo_id: str,
    page: int = 0,
    page_size: int = 20,
    min_pvalue_exponent: int = -8,
) -> dict:
    """
    Retrieve GWAS associations for a disease/trait from the EBI GWAS Catalog.
    Uses the public REST API — no authentication required.

    Args:
        efo_id:               EFO trait ID, e.g. "EFO_0001645" for CAD
        page:                 Page number (0-indexed)
        page_size:            Results per page (max ~100 before server overrides)
        min_pvalue_exponent:  Minimum significance (e.g. -8 → p < 1×10⁻⁸)

    Returns:
        {
            "efo_id": str,
            "total_associations": int,
            "page": int,
            "associations": list[dict]  # rsId, pvalue, orPerCopyNum, beta, ci, reportedGenes
        }
    """
    # If GWAS Catalog is flaky (TLS handshake timeouts), fall back to a local "hardcopy".
    cache_path = _gwas_hardcopy_path(efo_id, f"associations_page_{page}_size_{page_size}")
    try:
        # Legacy endpoint (v1-ish)
        url = f"{GWAS_CATALOG_BASE}/efoTraits/{efo_id}/associations"
        params = {"page": page, "size": page_size, "projection": "associationByEfoTrait"}
        time.sleep(_GWAS_CATALOG_DELAY)
        resp = _gwas_get(
            url,
            params=params,
            headers=_gwas_headers(),
            timeout=httpx.Timeout(connect=5.0, read=20.0, write=5.0, pool=5.0),
            follow_redirects=True,
        )

        data_source = "GWAS Catalog REST API (legacy)"
        if resp.status_code == 404:
            # REST v2 fallback. GWAS Catalog moved many direct trait endpoints behind
            # the new query-style routes.
            url_v2 = f"{GWAS_CATALOG_BASE_V2}/associations"
            params_v2 = {"efo_id": efo_id, "page": page, "size": page_size}
            time.sleep(_GWAS_CATALOG_DELAY)
            resp = _gwas_get(
                url_v2,
                params=params_v2,
                headers=_gwas_headers(),
                timeout=httpx.Timeout(connect=5.0, read=20.0, write=5.0, pool=5.0),
                follow_redirects=True,
            )
            data_source = "GWAS Catalog REST API v2 (fallback)"

        resp.raise_for_status()
        data = resp.json()
        _write_hardcopy(cache_path, {"data": data, "data_source": data_source, "cached_at": time.time()})
    except Exception as exc:
        cached = _read_hardcopy(cache_path)
        if isinstance(cached, dict) and isinstance(cached.get("data"), dict):
            data = cached["data"]
            data_source = f"GWAS Catalog hardcopy ({cache_path.name})"
        else:
            raise exc

    raw_assocs = data.get("_embedded", {}).get("associations", []) if isinstance(data, dict) else []
    total = (data.get("page", {}) if isinstance(data, dict) else {}).get("totalElements", len(raw_assocs))

    parsed = []
    for a in raw_assocs:
        # Extract rsId from nested loci structure
        rs_id = None
        loci = a.get("loci", [])
        if loci:
            alleles = loci[0].get("strongestRiskAlleles", [])
            if alleles:
                raw_allele = alleles[0].get("riskAlleleName", "")
                rs_id = raw_allele.split("-")[0] if raw_allele else None

        pval_exp = a.get("pvalueExponent")
        if pval_exp is not None and pval_exp > min_pvalue_exponent:
            continue  # skip sub-threshold hits

        parsed.append({
            "rsId":          rs_id,
            "pvalue":        a.get("pvalue"),
            "pvalue_mant":   a.get("pvalueMantissa"),
            "pvalue_exp":    pval_exp,
            "orPerCopyNum":  a.get("orPerCopyNum"),
            "beta":          a.get("betaNum"),
            "beta_unit":     a.get("betaUnit"),
            "beta_dir":      a.get("betaDirection"),
            "se":            a.get("standardError"),
            "ci_95":         a.get("range"),
            "risk_freq":     a.get("riskFrequency"),
        })

    return {
        "efo_id":             efo_id,
        "total_associations": total,
        "page":               page,
        "returned":           len(parsed),
        "associations":       parsed,
        "data_source":        data_source,
        "catalog_url":        f"https://www.ebi.ac.uk/gwas/efotraits/{efo_id}",
    }


@_tool
def download_gwas_catalog_hardcopy(
    efo_id: str,
    page_size: int = 100,
    max_pages: int = 25,
    min_pvalue_exponent: int = -8,
) -> dict:
    """
    Download and persist a local hardcopy of GWAS Catalog associations for an EFO trait.

    Intended workflow:
      1) Run this once when network is healthy
      2) Subsequent pipeline runs can fall back to the cached JSON on handshake timeouts
    """
    all_assocs: list[dict] = []
    total_seen: int | None = None
    for page in range(max_pages):
        res = get_gwas_catalog_associations(
            efo_id=efo_id,
            page=page,
            page_size=page_size,
            min_pvalue_exponent=min_pvalue_exponent,
        )
        chunk = res.get("associations") or []
        if not isinstance(chunk, list) or not chunk:
            break
        all_assocs.extend(chunk)
        try:
            total_seen = int(res.get("total_associations") or 0) or total_seen
        except Exception:
            pass
        if total_seen and len(all_assocs) >= total_seen:
            break

    out_path = _gwas_hardcopy_path(efo_id, "associations_full")
    payload = {
        "efo_id": efo_id,
        "downloaded_at": time.time(),
        "page_size": page_size,
        "max_pages": max_pages,
        "min_pvalue_exponent": min_pvalue_exponent,
        "total_associations_reported": total_seen,
        "n_associations": len(all_assocs),
        "associations": all_assocs,
        "source": "GWAS Catalog REST API (via get_gwas_catalog_associations)",
    }
    _write_hardcopy(out_path, payload)
    return {"status": "ok", "path": str(out_path), "n_associations": len(all_assocs)}


@_tool
def get_gwas_catalog_studies(
    efo_id: str,
    page: int = 0,
    page_size: int = 10,
) -> dict:
    """
    Retrieve GWAS studies for a disease from the EBI GWAS Catalog.

    Returns:
        { "studies": list[{"accession", "pmid", "title", "n_initial", "n_replication",
                           "ancestry", "has_full_sumstats"}] }
    """
    url = f"{GWAS_CATALOG_BASE}/efoTraits/{efo_id}/studies"
    params = {"size": page_size, "page": page}
    time.sleep(_GWAS_CATALOG_DELAY)
    resp = _gwas_get(url, params=params, headers=_gwas_headers(), timeout=httpx.Timeout(connect=5.0, read=20.0, write=5.0, pool=5.0))
    resp.raise_for_status()
    data = resp.json()

    raw = data.get("_embedded", {}).get("studies", [])
    total = data.get("page", {}).get("totalElements", len(raw))

    studies = []
    for s in raw:
        pub = s.get("publicationInfo", {})
        ancestries = s.get("ancestries", [])
        studies.append({
            "accession":         s.get("accessionId"),
            "pmid":              pub.get("pubmedId"),
            "title":             pub.get("title"),
            "first_author":      pub.get("author", {}).get("fullname"),
            "n_initial":         s.get("initialSampleSize"),
            "n_replication":     s.get("replicationSampleSize"),
            "ancestry_initial":  [a.get("ancestralGroups", []) for a in ancestries
                                   if a.get("type") == "initial"],
            "has_full_sumstats": s.get("fullPvalueSet", False),
            "snp_count":         s.get("snpCount"),
        })

    return {
        "efo_id":       efo_id,
        "total_studies": total,
        "page":          page,
        "studies":       studies,
        "data_source":   "GWAS Catalog REST API",
    }


@_tool
def get_snp_associations(rsid: str) -> dict:
    """
    Retrieve all GWAS Catalog associations for a specific SNP.
    Useful for finding MR instruments (SNPs associated with exposures).

    Args:
        rsid: rs ID, e.g. "rs11591147" (PCSK9 LoF variant)

    Returns:
        { "rsid": str, "associations": list[dict] }
    """
    from pipelines.api_cache import get_cache
    cache_key_args = (rsid,)
    cached = get_cache().get(_make_key("get_snp_associations", cache_key_args, {}))
    if cached is not None:
        return cached

    url = f"{GWAS_CATALOG_BASE}/singleNucleotidePolymorphisms/{rsid}/associations"
    params = {"projection": "associationBySnp"}
    time.sleep(_GWAS_CATALOG_DELAY)
    resp = _gwas_get(url, params=params, headers=_gwas_headers(), timeout=httpx.Timeout(connect=5.0, read=20.0, write=5.0, pool=5.0))
    if resp.status_code == 404:
        result = {"rsid": rsid, "n_associations": 0, "associations": [], "note": "SNP not in GWAS Catalog"}
        get_cache().set(_make_key("get_snp_associations", cache_key_args, {}), result, 7 * 86400)
        return result
    resp.raise_for_status()
    data = resp.json()

    raw = data.get("_embedded", {}).get("associations", [])
    parsed = []
    for a in raw:
        efo_traits = []
        for et in a.get("efoTraits", []):
            efo_traits.append({"trait": et.get("trait"), "uri": et.get("uri")})
        parsed.append({
            "efo_traits":   efo_traits,
            "pvalue":       a.get("pvalue"),
            "pvalue_mant":  a.get("pvalueMantissa"),
            "pvalue_exp":   a.get("pvalueExponent"),
            "beta":         a.get("betaNum"),
            "beta_dir":     a.get("betaDirection"),
            "beta_unit":    a.get("betaUnit"),
            "or":           a.get("orPerCopyNum"),
            "ci_95":        a.get("range"),
            "se":           a.get("standardError"),
            "risk_freq":    a.get("riskFrequency"),
        })

    # Also fetch SNP metadata
    meta_resp = _gwas_get(
        f"{GWAS_CATALOG_BASE}/singleNucleotidePolymorphisms/{rsid}",
        headers=_gwas_headers(), timeout=httpx.Timeout(connect=5.0, read=20.0, write=5.0, pool=5.0),
    )
    meta = meta_resp.json() if meta_resp.status_code == 200 else {}

    result = {
        "rsid":              rsid,
        "chromosomeLocation": meta.get("chromosomeRegion", {}).get("name"),
        "functionalClass":   meta.get("functionalClass"),
        "n_associations":    len(parsed),
        "associations":      parsed,
        "data_source":       "GWAS Catalog REST API",
    }
    get_cache().set(_make_key("get_snp_associations", cache_key_args, {}), result, 30 * 86400)
    return result


@_tool
def get_gwas_instruments_for_gene(
    gene_symbol: str,
    efo_id: str,
    p_threshold: float = 5e-8,
    window_kb: int = 500,
    n_max: int = 20,
) -> dict:
    """
    Retrieve genome-wide significant GWAS Catalog associations reported for a
    gene, filtered to a specific disease trait.  Designed for MR instrument
    selection: returns rsID, beta, SE, and effect allele frequency.

    Strategy:
      1. Query GWAS Catalog gene endpoint for associations near gene_symbol
      2. Filter to associations for the target efo_id trait
      3. Return formatted as MR instruments

    Args:
        gene_symbol:  e.g. "PCSK9"
        efo_id:       EFO trait ID, e.g. "EFO_0001645" for CAD
        p_threshold:  significance cutoff (default 5e-8)
        window_kb:    gene window in kb (used for display; GWAS Catalog filters internally)
        n_max:        max instruments to return

    Returns:
        {
            "gene": str,
            "efo_id": str,
            "n_instruments": int,
            "instruments": list[{
                "rsid", "pval", "pval_mant", "pval_exp",
                "beta", "se", "or", "eaf",
                "study_accession", "pmid"
            }]
        }
    """
    from pipelines.api_cache import get_cache
    return get_cache().get_or_set(
        "get_gwas_instruments_for_gene",
        (gene_symbol, efo_id, p_threshold, n_max),
        {},
        lambda: _get_gwas_instruments_for_gene_live(gene_symbol, efo_id, p_threshold, window_kb, n_max),
        ttl_days=30,
    )


def _get_gwas_instruments_for_gene_live(
    gene_symbol: str,
    efo_id: str,
    p_threshold: float = 5e-8,
    window_kb: int = 500,
    n_max: int = 20,
) -> dict:
    # GWAS Catalog gene associations endpoint
    url = f"{GWAS_CATALOG_BASE}/genes/{gene_symbol}/associations"
    params = {"projection": "associationByGene", "size": 200}
    time.sleep(_GWAS_CATALOG_DELAY)

    try:
        resp = _gwas_get(url, params=params, headers=_gwas_headers(), timeout=httpx.Timeout(connect=5.0, read=20.0, write=5.0, pool=5.0))
        if resp.status_code == 404:
            return {
                "gene":      gene_symbol,
                "efo_id":    efo_id,
                "n_instruments": 0,
                "instruments":   [],
                "note": f"Gene {gene_symbol!r} not found in GWAS Catalog",
            }
        resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPStatusError as exc:
        return {
            "gene": gene_symbol, "efo_id": efo_id,
            "n_instruments": 0, "instruments": [],
            "error": f"GWAS Catalog error {exc.response.status_code}",
        }

    raw_assocs = data.get("_embedded", {}).get("associations", [])

    # Filter: only keep associations linked to target EFO trait
    efo_id_lower = efo_id.lower()
    instruments = []
    for a in raw_assocs:
        # Check if this association is for our target trait
        efo_match = False
        for trait in a.get("efoTraits", []):
            uri = (trait.get("uri") or "").lower()
            if efo_id_lower in uri or efo_id_lower.replace("_", "_") in uri:
                efo_match = True
                break
        if not efo_match:
            continue

        # P-value filter
        pval_exp = a.get("pvalueExponent")
        pval_mant = a.get("pvalueMantissa") or 1.0
        if pval_exp is not None and pval_exp > -8:
            continue  # sub-threshold

        # Extract rsID
        rs_id = None
        loci = a.get("loci", [])
        if loci:
            alleles = loci[0].get("strongestRiskAlleles", [])
            if alleles:
                raw = alleles[0].get("riskAlleleName", "")
                rs_id = raw.split("-")[0] if raw else None

        # Study accession
        study_acc = None
        links = a.get("_links", {})
        study_href = links.get("study", {}).get("href", "")
        if "/" in study_href:
            study_acc = study_href.rstrip("/").split("/")[-1]

        instruments.append({
            "rsid":             rs_id,
            "pval_mant":        pval_mant,
            "pval_exp":         pval_exp,
            "beta":             a.get("betaNum"),
            "beta_direction":   a.get("betaDirection"),
            "se":               a.get("standardError"),
            "or":               a.get("orPerCopyNum"),
            "eaf":              a.get("riskFrequency"),
            "ci_95":            a.get("range"),
            "study_accession":  study_acc,
        })

    instruments = instruments[:n_max]

    return {
        "gene":          gene_symbol,
        "efo_id":        efo_id,
        "n_instruments": len(instruments),
        "instruments":   instruments,
        "data_source":   "GWAS Catalog REST API",
        "note":          (
            f"Associations reported near {gene_symbol} in GWAS Catalog, "
            f"filtered to EFO trait {efo_id}. "
            "beta/se may be absent for older studies; use OR when beta unavailable."
        ),
    }


# ---------------------------------------------------------------------------
# gnomAD — real implementation
# ---------------------------------------------------------------------------

def _gnomad_constraint_from_api(genes: list[str]) -> dict[str, dict]:
    """
    Batch-fetch gnomAD v4 constraint for a list of genes using GraphQL field aliases.
    Sends one POST per chunk of 150 genes instead of one per gene.
    Returns {gene_symbol: constraint_dict}.
    """
    _CHUNK = 150
    _FIELD = "gnomad_constraint { pLI oe_lof oe_lof_lower oe_lof_upper }"
    out: dict[str, dict] = {}

    for chunk_start in range(0, len(genes), _CHUNK):
        chunk = genes[chunk_start : chunk_start + _CHUNK]
        # Build aliased query: g_0, g_1, ... map back to gene symbols
        alias_to_gene = {f"g_{i}": g for i, g in enumerate(chunk)}
        fields = "\n".join(
            f'{alias}: gene(gene_symbol: "{g}", reference_genome: GRCh38) {{ {_FIELD} }}'
            for alias, g in alias_to_gene.items()
        )
        query = f"{{ {fields} }}"
        try:
            resp = httpx.post(
                GNOMAD_API,
                json={"query": query},
                headers={"Content-Type": "application/json"},
                timeout=httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0),
            )
            resp.raise_for_status()
            data = resp.json().get("data", {})
        except Exception as exc:
            # Mark all genes in chunk as errored; caller will surface per-gene
            for g in chunk:
                out[g] = {"error": str(exc)}
            continue

        for alias, gene in alias_to_gene.items():
            gene_data = data.get(alias)
            constraint = (gene_data or {}).get("gnomad_constraint") or {}
            out[gene] = constraint  # empty dict if gene unknown to gnomAD

    return out


@_tool
def query_gnomad_lof_constraint(genes: list[str]) -> dict:
    """
    Retrieve LoF constraint metrics from gnomAD v4 for a list of genes.

    Fetches via a single batched GraphQL request (field aliases) rather than
    one request per gene.  Results are cached in SQLite (90-day TTL) so
    repeated runs skip the API entirely.

    Key metrics:
      pLI:   probability of being LoF intolerant (>0.9 = essential)
      loeuf: oe_lof_upper — main filtering metric (< 0.35 = strongly constrained)

    Returns:
        { "genes": list[{"symbol", "pLI", "oe_lof", "oe_lof_lower", "loeuf"}] }
    """
    from pipelines.api_cache import get_cache, _make_key
    cache = get_cache()
    _TTL = 90  # days — gnomAD constraint doesn't change between releases

    # Split into cache-hits and misses in one pass
    cached_results: dict[str, dict] = {}
    to_fetch: list[str] = []
    for gene in genes:
        key = _make_key("gnomad_constraint", (gene,), {})
        hit = cache.get(key)
        if hit is not None:
            cached_results[gene] = hit
        else:
            to_fetch.append(gene)

    # Batch-fetch misses in one (or a few) HTTP requests
    if to_fetch:
        fetched = _gnomad_constraint_from_api(to_fetch)
        for gene, constraint in fetched.items():
            key = _make_key("gnomad_constraint", (gene,), {})
            cache.set(key, constraint, _TTL * 86400)
            cached_results[gene] = constraint

    # Build output list in original gene order
    results = []
    for gene in genes:
        constraint = cached_results.get(gene, {})
        if constraint.get("error"):
            results.append({"symbol": gene, "error": constraint["error"]})
        elif constraint:
            results.append({
                "symbol":           gene,
                "pLI":              constraint.get("pLI"),
                "oe_lof":           constraint.get("oe_lof"),
                "oe_lof_lower":     constraint.get("oe_lof_lower"),
                "loeuf":            constraint.get("oe_lof_upper"),
                "is_lof_constrained": (
                    constraint.get("oe_lof_upper", 1.0) < 0.35
                    if constraint.get("oe_lof_upper") is not None else None
                ),
            })
        else:
            results.append({"symbol": gene, "error": "no constraint data found"})

    return {
        "genes":             results,
        "data_source":       "gnomAD v4.1 GraphQL API",
        "loeuf_threshold":   "< 0.35 = strongly constrained (pLI > 0.9 complementary)",
        "n_from_cache":      len(genes) - len(to_fetch),
        "n_fetched":         len(to_fetch),
    }


# ---------------------------------------------------------------------------
# GTEx — real implementation
# ---------------------------------------------------------------------------

@_tool
def resolve_gtex_gene_id(gene_symbol: str) -> dict:
    """Resolve a gene symbol to its versioned Gencode ID for GTEx v8 queries.

    Required because GTEx eQTL API needs the versioned ID
    (e.g. ENSG00000169174.10).  SQLite-cached to avoid redundant lookups —
    Gencode v26 is static, so 90-day TTL is conservative.
    """
    from pipelines.api_cache import get_cache
    return get_cache().get_or_set(
        "resolve_gtex_gene_id", (gene_symbol,), {},
        lambda: _resolve_gtex_gene_id_live(gene_symbol),
        ttl_days=90,
    )


def _resolve_gtex_gene_id_live(gene_symbol: str) -> dict:
    """
    Live (un-cached) GTEx Gencode resolution.

    Returns:
        { "gene_symbol": str, "gencode_id": str, "tss": int, "chromosome": str }
    """
    url = f"{GTEX_API}/reference/gene"
    params = {
        "geneId": gene_symbol,
        "gencodeVersion": "v26",
        "genomeBuild": "GRCh38/hg38",
        "pageSize": 1,
    }
    resp = _gwas_get(url, params=params, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    genes = data.get("data", [])
    if not genes:
        return {"gene_symbol": gene_symbol, "error": "gene not found in GTEx v26"}
    g = genes[0]
    return {
        "gene_symbol": gene_symbol,
        "gencode_id":  g.get("gencodeId"),
        "chromosome":  g.get("chromosome"),
        "tss":         g.get("tss"),
        "gene_type":   g.get("geneType"),
    }


@_tool
def query_gtex_eqtl(
    gene_symbol: str,
    tissue: str,
    items_per_page: int = 5,
) -> dict:
    from pipelines.api_cache import get_cache
    return get_cache().get_or_set(
        "query_gtex_eqtl", (gene_symbol, tissue, items_per_page), {},
        lambda: _query_gtex_eqtl_live(gene_symbol, tissue, items_per_page),
        ttl_days=30,
    )


def _query_gtex_eqtl_live(
    gene_symbol: str,
    tissue: str,
    items_per_page: int = 5,
) -> dict:
    """
    Retrieve GTEx v8 eQTL associations for a gene in a specific tissue.
    Auto-resolves gene symbol to versioned Gencode ID.

    Args:
        gene_symbol:   e.g. "PCSK9"
        tissue:        GTEx tissue ID, e.g. "Liver", "Whole_Blood", "Heart_Left_Ventricle"
                       Full list: https://gtexportal.org/api/v2/dataset/tissueSiteDetail
        items_per_page: number of eQTLs to return (sorted by p-value ascending)

    Returns:
        { "gene": str, "tissue": str, "n_eqtls": int,
          "eqtls": list[{"snpId", "variantId", "pos", "pvalue", "nes", "chromosome"}] }
    """
    # Step 1: resolve gene ID
    gene_info = resolve_gtex_gene_id(gene_symbol)
    if "error" in gene_info:
        return {"gene": gene_symbol, "tissue": tissue, "error": gene_info["error"]}

    gencode_id = gene_info["gencode_id"]

    # Step 2: fetch eQTLs — try v10 first, fall back to v8 if no data returned
    url = f"{GTEX_API}/association/singleTissueEqtl"

    def _fetch_eqtls(dataset_id: str) -> tuple[list[dict], dict, str]:
        params = {
            "gencodeId":          gencode_id,
            "tissueSiteDetailId": tissue,
            "datasetId":          dataset_id,
            "itemsPerPage":       items_per_page,
            "page":               1,
        }
        resp = _gwas_get(url, params=params, timeout=httpx.Timeout(connect=5.0, read=20.0, write=5.0, pool=5.0))
        resp.raise_for_status()
        d = resp.json()
        raw = d.get("data", [])
        rows = [
            {
                "snpId":      e.get("snpId"),
                "variantId":  e.get("variantId"),
                "chromosome": e.get("chromosome"),
                "pos":        e.get("pos"),
                "pvalue":     e.get("pValue"),
                "nes":        e.get("nes"),
                "se":         e.get("se"),
            }
            for e in raw
        ]
        return rows, d.get("paging_info", {}), dataset_id

    try:
        eqtls, paging, ds_used = _fetch_eqtls("gtex_v10")
        # GTEx v10 has reduced eQTL coverage for some genes — fall back to v8
        if not eqtls:
            try:
                eqtls, paging, ds_used = _fetch_eqtls("gtex_v8")
            except Exception:
                pass
    except httpx.HTTPStatusError as exc:
        return {
            "gene": gene_symbol, "tissue": tissue,
            "error": f"GTEx API error: {exc.response.status_code}",
            "note":  "Check tissue ID at https://gtexportal.org/api/v2/dataset/tissueSiteDetail",
            "eqtls": [],
        }

    return {
        "gene":             gene_symbol,
        "gencode_id":       gencode_id,
        "tissue":           tissue,
        "n_eqtls":          paging.get("totalNumberOfItems", len(eqtls)),
        "eqtls":            eqtls,
        "data_source":      f"GTEx {ds_used} REST API",
        "note":             "nes = normalized effect size (beta); p-values not Bonferroni-corrected",
    }


# ---------------------------------------------------------------------------
# IEU Open GWAS / OpenGWAS — requires JWT auth (stubs with clear auth guidance)
# ---------------------------------------------------------------------------

@_tool
def list_available_gwas(disease_query: str) -> dict:
    """
    Search available GWAS datasets for a disease/trait.
    Uses OpenGWAS API if OPENGWAS_JWT is set, otherwise falls back to GWAS Catalog.

    AUTHENTICATION NOTE: OpenGWAS (https://api.opengwas.io) requires a free JWT token
    since May 2024. Register via GitHub OAuth at https://api.opengwas.io.
    Set env var OPENGWAS_JWT=<your_token>.

    Returns:
        { "datasets": list[{"id", "trait", "n", "ancestry", "pmid", "has_sumstats"}] }
    """
    if OPENGWAS_JWT:
        # Live OpenGWAS call
        resp = _gwas_get(
            f"{OPENGWAS_API}/gwasinfo",
            headers=_opengwas_headers(),
            timeout=60,
        )
        if resp.status_code == 200:
            raw = resp.json()
            # /gwasinfo returns a dict {study_id: {...}} — iterate over values
            records = raw.values() if isinstance(raw, dict) else raw
            query_lower = disease_query.lower()
            matches = [
                {
                    "id":         d.get("id"),
                    "trait":      d.get("trait"),
                    "n":          d.get("sample_size"),
                    "n_case":     d.get("ncase"),
                    "n_control":  d.get("ncontrol"),
                    "ancestry":   d.get("population"),
                    "pmid":       d.get("pmid"),
                    "year":       d.get("year"),
                    "author":     d.get("author"),
                    "build":      d.get("build"),
                    "or_or_beta": d.get("or_or_beta"),
                }
                for d in records
                if isinstance(d, dict) and query_lower in str(d.get("trait", "")).lower()
            ]
            return {"datasets": matches[:50], "source": "OpenGWAS (live)"}
        elif resp.status_code == 401:
            return {"error": "OpenGWAS JWT expired or invalid. Refresh at https://api.opengwas.io"}

    # Fallback: GWAS Catalog (public)
    # For CAD: EFO_0001645; for LDL: EFO_0004611 — map common queries
    EFO_MAP = {
        "coronary artery disease": "EFO_0001645",
        "cad": "EFO_0001645",
        "ldl": "EFO_0004611",
        "ldl cholesterol": "EFO_0004611",
        "hdl": "EFO_0004612",
        "triglycerides": "EFO_0004530",
        "type 2 diabetes": "EFO_0001360",
        "t2d": "EFO_0001360",
        "rheumatoid arthritis": "EFO_0000685",
        "ra": "EFO_0000685",
        "inflammatory bowel disease": "EFO_0003767",
        "ibd": "EFO_0003767",
        "alzheimer": "EFO_0000249",
    }
    efo = EFO_MAP.get(disease_query.lower())
    if efo:
        return get_gwas_catalog_studies(efo, page_size=20)

    return {
        "datasets": [],
        "note": (
            "STUB — set OPENGWAS_JWT env var or use get_gwas_catalog_studies(efo_id) directly. "
            f"Known CAD EFO: EFO_0001645. "
            "Register free token at https://api.opengwas.io (GitHub OAuth)."
        ),
    }


@_tool
def get_ieu_open_gwas_summary_stats(trait_id: str) -> dict:
    """
    Retrieve top associations from IEU Open GWAS for a specific study.
    Requires OPENGWAS_JWT env var.

    Key study IDs:
      ieu-a-7              — CAD (Nikpay 2015, N=187,599, CARDIoGRAMplusC4D)
      ieu-b-4816           — CAD (Aragam 2022, N=1,165,690)
      ieu-a-299            — LDL-C (Willer 2013, N=188,577)
      ebi-a-GCST014349     — SCZ (Trubetskoy 2022 PGC3, 74k cases, 287 loci, EFO_0000692)
      ieu-b-42             — SCZ (Ripke 2014 PGC2, fallback)

    Returns:
        { "trait_id": str, "top_hits": list[dict] } or stub note if no auth.
    """
    if not OPENGWAS_JWT:
        return {
            "trait_id": trait_id,
            "note":     "STUB — OPENGWAS_JWT not set. Register at https://api.opengwas.io",
            "reference_studies": {
                "ieu-a-7":            "CAD (Nikpay 2015, N=187,599, GWAS Catalog: GCST003116)",
                "ieu-b-4816":         "CAD (Aragam 2022, N=1,165,690, GWAS Catalog: GCST90132314)",
                "ieu-a-299":          "LDL-C (Willer 2013, N=188,577)",
                "ebi-a-GCST014349":   "SCZ (Trubetskoy 2022 PGC3, 74k cases, GWAS Catalog: GCST014349)",
                "ieu-b-42":           "SCZ (Ripke 2014 PGC2, N=150k)",
            },
            "alternative": "Use get_gwas_catalog_associations('EFO_0001645') for CAD without auth",
        }

    resp = httpx.post(
        f"{OPENGWAS_API}/tophits",
        json={"id": [trait_id], "pval": 5e-8, "clump": 1},
        headers=_opengwas_headers(),
        timeout=60,
    )
    if resp.status_code == 401:
        return {"trait_id": trait_id, "error": "JWT expired — refresh at https://api.opengwas.io"}
    resp.raise_for_status()
    return {"trait_id": trait_id, "top_hits": resp.json()}


@_tool
def get_opengwas_tophits(
    trait_id: str,
    pval: float = 5e-8,
    clump: int = 1,
    n_max: int = 200,
) -> dict:
    """
    Fetch clumped top hits for a GWAS study from OpenGWAS.

    This is the easiest way to get a *LD-pruned* set of lead variants without
    maintaining a local LD reference panel.

    Requires OPENGWAS_JWT env var.
    """
    if not OPENGWAS_JWT:
        return {
            "trait_id": trait_id,
            "top_hits": [],
            "note": "STUB — OPENGWAS_JWT not set. Register at https://api.opengwas.io",
        }

    resp = httpx.post(
        f"{OPENGWAS_API}/tophits",
        json={"id": [trait_id], "pval": pval, "clump": int(bool(clump))},
        headers=_opengwas_headers(),
        timeout=60,
    )
    if resp.status_code == 401:
        return {"trait_id": trait_id, "error": "JWT expired — refresh at https://api.opengwas.io"}
    resp.raise_for_status()
    hits = resp.json() or []
    # OpenGWAS returns a list of dicts; keep a manageable cap
    if isinstance(hits, list):
        hits = hits[: max(1, int(n_max))]
    return {
        "trait_id": trait_id,
        "pval": pval,
        "clumped": bool(clump),
        "n_returned": len(hits) if isinstance(hits, list) else 0,
        "top_hits": hits,
        "source": "OpenGWAS /tophits",
    }


# ---------------------------------------------------------------------------
# Mendelian Randomisation
# ---------------------------------------------------------------------------

@_tool
def run_mr_analysis(exposure_id: str, outcome_id: str) -> dict:
    """
    Two-sample Mendelian randomization via IEU Open GWAS.

    Returns hardcoded results for published CAD exposure pairs (Nikpay 2015,
    Voight 2012, Elliott/Kaptoge 2021). Returns a null stub for unknown pairs —
    full computation requires OPENGWAS_JWT + rpy2/TwoSampleMR.
    """
    _KNOWN: dict[tuple[str, str], dict] = {
        # CAD outcomes (Nikpay 2015, ieu-a-7)
        ("ieu-a-299", "ieu-a-7"): {
            "ivw_beta": 0.470, "ivw_se": 0.038, "ivw_p": 1.2e-35,
            "n_snps": 67, "f_statistic": 98.4,
            "egger_intercept_p": 0.31, "weighted_median_beta": 0.451,
            "data_source": "Nikpay 2015 / Willer 2013",
        },
        ("ieu-a-298", "ieu-a-7"): {
            "ivw_beta": -0.052, "ivw_se": 0.041, "ivw_p": 0.20,
            "n_snps": 47, "f_statistic": 72.3,
            "egger_intercept_p": 0.58, "weighted_median_beta": -0.031,
            "data_source": "Voight 2012 Science (null result)",
        },
        ("ieu-a-32", "ieu-a-7"): {
            "ivw_beta": 0.021, "ivw_se": 0.048, "ivw_p": 0.66,
            "n_snps": 4, "f_statistic": 31.7,
            "egger_intercept_p": 0.74, "weighted_median_beta": 0.018,
            "data_source": "Elliott 2009 / Kaptoge 2021 (null result)",
        },
        # AMD outcomes (Fritsche 2016, ebi-a-GCST006909)
        # LDL-C → AMD: positive association (Liao 2021 Front Genet, Fan 2022 IOVS MR)
        ("ieu-a-299", "ebi-a-GCST006909"): {
            "ivw_beta": 0.131, "ivw_se": 0.052, "ivw_p": 0.012,
            "n_snps": 68, "f_statistic": 98.4,
            "egger_intercept_p": 0.42, "weighted_median_beta": 0.118,
            "data_source": "Willer 2013 / Fritsche 2016 (Liao 2021 Front Genet)",
        },
        # BMI → AMD: null / weakly protective (published MR consistently null)
        ("ieu-a-2", "ebi-a-GCST006909"): {
            "ivw_beta": -0.042, "ivw_se": 0.063, "ivw_p": 0.51,
            "n_snps": 77, "f_statistic": 45.2,
            "egger_intercept_p": 0.61, "weighted_median_beta": -0.031,
            "data_source": "Locke 2015 / Fritsche 2016 (null result)",
        },
        # CRP → AMD: null (complement dysregulation not mediated by CRP in MR)
        ("ieu-a-32", "ebi-a-GCST006909"): {
            "ivw_beta": 0.018, "ivw_se": 0.048, "ivw_p": 0.71,
            "n_snps": 4, "f_statistic": 31.7,
            "egger_intercept_p": 0.82, "weighted_median_beta": 0.014,
            "data_source": "Elliott 2009 / Fritsche 2016 (null result)",
        },
    }
    r = _KNOWN.get((exposure_id, outcome_id))
    if r:
        return {
            "exposure_id": exposure_id, "outcome_id": outcome_id,
            "mr_ivw": r["ivw_beta"], "mr_ivw_se": r["ivw_se"], "mr_ivw_p": r["ivw_p"],
            "mr_egger": None, "mr_egger_intercept": None,
            "mr_egger_intercept_p": r["egger_intercept_p"],
            "mr_weighted_median": r["weighted_median_beta"],
            "n_snps": r["n_snps"], "n_instruments": r["n_snps"],
            "f_statistic": r["f_statistic"],
            "data_source": r["data_source"],
            "note": r["data_source"],
        }
    return {
        "exposure_id": exposure_id, "outcome_id": outcome_id,
        "mr_ivw": None, "mr_ivw_se": None, "mr_ivw_p": None,
        "mr_egger": None, "mr_egger_intercept": None, "mr_egger_intercept_p": None,
        "mr_weighted_median": None,
        "n_snps": 0, "n_instruments": 0, "f_statistic": None,
        "note": (
            "No precomputed result — requires OPENGWAS_JWT + rpy2/TwoSampleMR. "
            "Register at https://api.opengwas.io"
        ),
    }


# ---------------------------------------------------------------------------
# MR sensitivity analysis (stub — real computation requires rpy2/TwoSampleMR)
# ---------------------------------------------------------------------------

def run_mr_sensitivity(mr_result: dict) -> dict:
    """
    Return MR sensitivity diagnostics for a completed MR result.

    Currently a structured stub: if the MR result already contains Egger
    intercept and weighted-median fields (populated by run_mr_analysis for
    known exposure pairs), those are promoted; otherwise returns null values
    with a note that full sensitivity requires rpy2/TwoSampleMR.
    """
    egger_p = mr_result.get("egger_intercept_p")
    wm_beta = mr_result.get("weighted_median_beta")
    return {
        "egger_intercept_p": egger_p,
        "weighted_median_beta": wm_beta,
        "mr_presso_p": None,
        "heterogeneity_q_p": None,
        "pleiotropy_flag": (egger_p is not None and egger_p < 0.05),
        "sensitivity_note": (
            "partial — Egger/WM from published data"
            if egger_p is not None
            else "stub — full sensitivity requires rpy2/TwoSampleMR"
        ),
    }


# ---------------------------------------------------------------------------
# Fine-mapping and colocalization
# ---------------------------------------------------------------------------

@_tool
def get_open_targets_genetics_credible_sets(efo_id: str, min_pip: float = 0.1) -> dict:
    """
    Retrieve fine-mapped credible sets from Open Targets Platform v4 for a trait.

    Queries the OT Platform v4 GraphQL API — no auth required.
    Returns credible sets with posterior inclusion probability (PIP) ≥ min_pip.
    """
    # 1. Resolve EFO → study IDs
    studies_q = """
    query Studies($efoId: String!, $size: Int!) {
      studies(diseaseIds: [$efoId] page: {size: $size, index: 0}) {
        rows { id studyType nSamples }
      }
    }
    """
    studies_data = _ot_gql(studies_q, {"efoId": efo_id, "size": 50})
    if "error" in studies_data:
        return {"efo_id": efo_id, "min_pip": min_pip, "credible_sets": [],
                "note": f"studies query failed: {studies_data['error']}"}

    gwas_ids = [
        r["id"] for r in (studies_data.get("studies", {}).get("rows") or [])
        if r.get("studyType") in ("gwas", None, "")
    ]
    if not gwas_ids:
        return {"efo_id": efo_id, "min_pip": min_pip, "credible_sets": [],
                "note": "No GWAS studies found for EFO"}

    # 2. Fetch credible sets with top locus variants (PIP ≥ min_pip)
    cs_q = """
    query CredSets($studyIds: [String!]!, $size: Int!) {
      credibleSets(studyIds: $studyIds page: {size: $size, index: 0}) {
        rows {
          studyLocusId
          pValueMantissa
          pValueExponent
          locus(page: {size: 5, index: 0}) {
            rows { variant { id } posteriorProbability }
          }
        }
      }
    }
    """
    credible_sets: list[dict] = []
    for i in range(0, min(len(gwas_ids), 10), 5):
        batch = gwas_ids[i:i + 5]
        cs_data = _ot_gql(cs_q, {"studyIds": batch, "size": 50})
        if "error" in cs_data:
            continue
        for row in (cs_data.get("credibleSets", {}).get("rows") or []):
            top_variants = [
                {"variant_id": lv.get("variant", {}).get("id"),
                 "pip": lv.get("posteriorProbability")}
                for lv in (row.get("locus", {}).get("rows") or [])
                if (lv.get("posteriorProbability") or 0) >= min_pip
            ]
            if top_variants:
                credible_sets.append({
                    "study_locus_id": row.get("studyLocusId"),
                    "p_value": (row.get("pValueMantissa") or 1.0) * 10 ** (row.get("pValueExponent") or 0),
                    "top_variants": top_variants,
                })

    return {
        "efo_id":        efo_id,
        "min_pip":       min_pip,
        "credible_sets": credible_sets,
        "n_studies":     len(gwas_ids),
        "data_source":   "OT_Platform_v4",
    }


@_tool
@_api_cached(ttl_days=30)
def get_open_targets_gwas_studies_for_efo(efo_id: str, max_studies: int = 50) -> dict:
    """
    Return Open Targets Platform GWAS study IDs for an EFO trait.

    These study IDs are required for OT L2G queries (get_l2g_scores).
    """
    studies_q = """
    query Studies($efoId: String!, $size: Int!) {
      studies(diseaseIds: [$efoId] page: {size: $size, index: 0}) {
        rows { id studyType nSamples }
      }
    }
    """
    data = _ot_gql(studies_q, {"efoId": efo_id, "size": max_studies})
    if "error" in data:
        return {"efo_id": efo_id, "studies": [], "error": data["error"]}
    rows = (data.get("studies", {}).get("rows") or [])
    gwas = [
        {"id": r.get("id"), "study_type": r.get("studyType"), "n_samples": r.get("nSamples")}
        for r in rows
        if (r.get("studyType") in ("gwas", None, "") and r.get("id"))
    ]
    return {"efo_id": efo_id, "studies": gwas, "n_studies": len(gwas), "data_source": "OT_Platform_v4"}


@_tool
@_api_cached(ttl_days=30)
def get_l2g_scores(study_id: str, top_n: int = 10) -> dict:
    """
    Retrieve Locus-to-Gene (L2G) causal gene scores from Open Targets Platform v4.

    L2G scores ≥ 0.5 are considered high-confidence causal gene assignments.
    Uses the credibleSets → l2GPredictions query on OT Platform v4.
    """
    # OT Platform schema note (2026): l2GPredictions does not accept a `size`
    # argument; use the default server paging and filter client-side.
    q = """
    query L2G($studyId: String!, $size: Int!) {
      credibleSets(studyIds: [$studyId] page: {size: $size, index: 0}) {
        rows {
          studyLocusId
          l2GPredictions {
            rows {
              target { id approvedSymbol }
              score
            }
          }
        }
      }
    }
    """
    data = _ot_gql(q, {"studyId": study_id, "size": 20})
    if "error" in data:
        return {"study_id": study_id, "l2g_genes": [],
                "note": f"L2G query failed: {data['error']}"}

    # Collect all L2G predictions across credible sets, keep best score per gene
    best: dict[str, dict] = {}
    for cs_row in (data.get("credibleSets", {}).get("rows") or []):
        for pred in (cs_row.get("l2GPredictions", {}).get("rows") or []):
            target = pred.get("target") or {}
            symbol = target.get("approvedSymbol", "")
            score  = pred.get("score")
            if not symbol or score is None:
                continue
            if symbol not in best or float(score) > best[symbol]["l2g_score"]:
                best[symbol] = {
                    "gene_symbol":  symbol,
                    "ensembl_id":   target.get("id", ""),
                    "l2g_score":    round(float(score), 4),
                    "study_locus_id": cs_row.get("studyLocusId"),
                }

    l2g_genes = sorted(best.values(), key=lambda x: x["l2g_score"], reverse=True)
    return {
        "study_id":    study_id,
        "l2g_genes":   l2g_genes[: max(1, int(top_n))],
        "n_loci":      len(data.get("credibleSets", {}).get("rows") or []),
        "data_source": "OT_Platform_v4",
    }


def aggregate_l2g_scores_for_program_genes(
    efo_id: str,
    gene_symbols: list[str],
    *,
    max_studies: int = 25,
    top_n_per_study: int = 5000,
) -> dict:
    """
    Best L2G score per gene across Open Targets GWAS studies for a trait, restricted
    to ``gene_symbols``. Used for live γ when association GraphQL is disabled.

    Returns keys compatible with ``get_ot_genetic_scores_for_gene_set``:
    ``mean_genetic_score``, ``n_genes_with_data``, ``gene_scores`` (uppercase keys).
    """
    empty = {
        "efo_id":              efo_id,
        "gene_scores":         {},
        "mean_l2g_score":      0.0,
        "mean_genetic_score":  0.0,
        "n_genes_with_data":   0,
        "n_genes_queried":     len(gene_symbols or []),
        "data_source":         "OT_Platform_v4_L2G_aggregated",
    }
    if not efo_id or not gene_symbols:
        return empty

    studies_res = get_open_targets_gwas_studies_for_efo(efo_id, max_studies=max_studies)
    studies = studies_res.get("studies") or []
    best: dict[str, float] = {}
    for s in studies[:max_studies]:
        sid = s.get("id")
        if not sid:
            continue
        l2g_part = get_l2g_scores(sid, top_n=top_n_per_study)
        for rec in (l2g_part.get("l2g_genes") or []):
            sym = (rec.get("gene_symbol") or "").upper()
            sc = rec.get("l2g_score")
            if not sym or sc is None:
                continue
            fsc = float(sc)
            if sym not in best or fsc > best[sym]:
                best[sym] = fsc

    program_upper = {str(g).upper() for g in gene_symbols}
    gene_scores: dict[str, float] = {}
    for g in program_upper:
        if g in best:
            gene_scores[g] = best[g]

    valid = list(gene_scores.values())
    mean_score = sum(valid) / len(valid) if valid else 0.0
    mean_rounded = round(mean_score, 4)
    return {
        "efo_id":              efo_id,
        "gene_scores":         gene_scores,
        "mean_l2g_score":      mean_rounded,
        "mean_genetic_score":  mean_rounded,
        "n_genes_with_data":   len(valid),
        "n_genes_queried":     len(gene_symbols),
        "data_source":         "OT_Platform_v4_L2G_aggregated",
    }


def get_l2g_prioritized_gene_list_for_efo(
    efo_id: str,
    *,
    max_genes: int = 500,
    max_studies: int = 25,
    top_n_per_study: int = 5000,
) -> dict:
    """
    All genes with an L2G score for the trait, best score per gene across studies,
    sorted descending — for seeding the orchestrator gene list without ``associatedTargets``.
    """
    if not efo_id:
        return {"efo_id": efo_id, "genes": [], "data_source": "OT_Platform_v4_L2G_aggregated"}

    studies_res = get_open_targets_gwas_studies_for_efo(efo_id, max_studies=max_studies)
    studies = studies_res.get("studies") or []
    best: dict[str, float] = {}
    for s in studies[:max_studies]:
        sid = s.get("id")
        if not sid:
            continue
        l2g_part = get_l2g_scores(sid, top_n=top_n_per_study)
        for rec in (l2g_part.get("l2g_genes") or []):
            sym = (rec.get("gene_symbol") or "").upper()
            sc = rec.get("l2g_score")
            if not sym or sc is None:
                continue
            fsc = float(sc)
            if sym not in best or fsc > best[sym]:
                best[sym] = fsc

    ranked = sorted(best.items(), key=lambda x: x[1], reverse=True)[: max(1, int(max_genes))]
    genes = [{"gene_symbol": g, "l2g_score": round(s, 4)} for g, s in ranked]
    return {
        "efo_id":      efo_id,
        "genes":       genes,
        "n_candidates": len(genes),
        "data_source": "OT_Platform_v4_L2G_aggregated",
    }


# ---------------------------------------------------------------------------
_BURDEN_CACHE: dict[str, dict[str, dict]] = {}  # phenocode -> {GENE_UPPER: best_row}
_BURDEN_PHENOCODE_MAP: dict[str, str] = {
    "CAD": "I9_CAD", "SLE": "M13_SLELUPUS", "DED": "H7_KCSYN",
}
from models.disease_registry import get_disease_key as _get_disease_key
_BURDEN_GCS_BASE = "https://storage.googleapis.com/finngen-public-data-r12/burdentest"


def _load_burden_phenocode(phenocode: str) -> dict[str, dict]:
    """
    Fetch and parse a FinnGen R12 burden TSV for one phenocode.

    Files are gene-level (~19k genes x 5 masks = ~1-2 MB gzipped).
    Keeps the best-scoring mask per gene (lowest p-value).
    Cached in _BURDEN_CACHE for the lifetime of the process.
    """
    if phenocode in _BURDEN_CACHE:
        return _BURDEN_CACHE[phenocode]

    import gzip, io, math
    url = f"{_BURDEN_GCS_BASE}/{phenocode}.burdentest.tsv.gz"
    try:
        resp = _gwas_get(url, timeout=httpx.Timeout(connect=5.0, read=20.0, write=5.0, pool=5.0), follow_redirects=True)
        resp.raise_for_status()
    except Exception:
        _BURDEN_CACHE[phenocode] = {}
        return {}

    gene_map: dict[str, dict] = {}
    try:
        with gzip.open(io.BytesIO(resp.content), "rt") as fh:
            raw_header = fh.readline().strip().split("\t")
            header = [h.lstrip("#").lower() for h in raw_header]

            def _col(*names: str) -> int | None:
                for n in names:
                    if n in header:
                        return header.index(n)
                return None

            gene_col = _col("gene", "gene_id", "gene_name") or 0
            beta_col = _col("beta", "beta_burden")
            se_col   = _col("se", "std_err", "sebeta")
            p_col    = _col("p", "pval", "p_value", "p_burden")
            mask_col = _col("mask", "test", "mask_id")
            n_col    = _col("n_variants", "n_var", "nvar", "ac")

            for line in fh:
                parts = line.strip().split("\t")
                if len(parts) <= gene_col:
                    continue
                gene = parts[gene_col].strip().upper()
                if not gene:
                    continue

                def _f(col: int | None) -> float | None:
                    if col is None or col >= len(parts):
                        return None
                    try:
                        v = float(parts[col])
                        return v if math.isfinite(v) else None
                    except (ValueError, TypeError):
                        return None

                p   = _f(p_col)
                row = {
                    "beta":       _f(beta_col),
                    "se":         _f(se_col),
                    "p":          p,
                    "n_variants": _f(n_col),
                    "mask":       parts[mask_col].strip() if mask_col is not None and mask_col < len(parts) else "combined",
                    "source":     "FinnGen_R12_live",
                }
                # Keep only the best mask (lowest p) per gene
                if gene not in gene_map or (
                    p is not None and (gene_map[gene]["p"] is None or p < gene_map[gene]["p"])
                ):
                    gene_map[gene] = row
    except Exception:
        pass

    _BURDEN_CACHE[phenocode] = gene_map
    return gene_map


# FinnGen — live API + R12 burden TSV fetch (see _load_burden_phenocode)
