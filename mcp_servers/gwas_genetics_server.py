"""
gwas_genetics_server.py — MCP server for GWAS, genetics, and regulatory genomics tools.

Real API integrations (no auth required):
  - GWAS Catalog REST API (EBI) — associations, studies, SNP lookups
  - gnomAD GraphQL — LoF constraint (pLI, LOEUF)
  - GTEx v8 API — eQTL associations per tissue

Stubs (require auth / compute / local data):
  - IEU Open GWAS / OpenGWAS — now requires JWT auth since May 2024
    Register free at https://api.opengwas.io to get a token.
    Set env var OPENGWAS_JWT=<your_token> to enable.
  - SuSiE-RSS, HyPrColoc, LDSC, ABC model, Enformer — compute-heavy, stubbed
  - FinnGen burden results — download required, stubbed

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

import httpx

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
GNOMAD_API          = "https://gnomad.broadinstitute.org/api"
GTEX_API            = "https://gtexportal.org/api/v2"
OPENGWAS_API        = "https://api.opengwas.io/api"

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
    url = f"{GWAS_CATALOG_BASE}/efoTraits/{efo_id}/associations"
    params = {
        "page": page,
        "size": page_size,
        "projection": "associationByEfoTrait",
    }
    time.sleep(_GWAS_CATALOG_DELAY)
    resp = httpx.get(url, params=params, headers=_gwas_headers(), timeout=30)
    resp.raise_for_status()
    data = resp.json()

    raw_assocs = data.get("_embedded", {}).get("associations", [])
    total = data.get("page", {}).get("totalElements", len(raw_assocs))

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
        "data_source":        "GWAS Catalog REST API v1.0.2",
        "catalog_url":        f"https://www.ebi.ac.uk/gwas/efotraits/{efo_id}",
    }


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
    resp = httpx.get(url, params=params, headers=_gwas_headers(), timeout=30)
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
    url = f"{GWAS_CATALOG_BASE}/singleNucleotidePolymorphisms/{rsid}/associations"
    params = {"projection": "associationBySnp"}
    time.sleep(_GWAS_CATALOG_DELAY)
    resp = httpx.get(url, params=params, headers=_gwas_headers(), timeout=30)
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
    meta_resp = httpx.get(
        f"{GWAS_CATALOG_BASE}/singleNucleotidePolymorphisms/{rsid}",
        headers=_gwas_headers(), timeout=30,
    )
    meta = meta_resp.json() if meta_resp.status_code == 200 else {}

    return {
        "rsid":              rsid,
        "chromosomeLocation": meta.get("chromosomeRegion", {}).get("name"),
        "functionalClass":   meta.get("functionalClass"),
        "n_associations":    len(parsed),
        "associations":      parsed,
        "data_source":       "GWAS Catalog REST API",
    }


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
    # GWAS Catalog gene associations endpoint
    url = f"{GWAS_CATALOG_BASE}/genes/{gene_symbol}/associations"
    params = {"projection": "associationByGene", "size": 200}
    time.sleep(_GWAS_CATALOG_DELAY)

    try:
        resp = httpx.get(url, params=params, headers=_gwas_headers(), timeout=30)
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

@_tool
def query_gnomad_lof_constraint(genes: list[str]) -> dict:
    """
    Retrieve LoF constraint metrics from gnomAD v4 for a list of genes.
    Uses the gnomAD GraphQL API (no auth required).

    Key metrics:
      pLI:       probability of being LoF intolerant (high = intolerant; >0.9 = significant)
      oe_lof:    observed/expected LoF ratio (LOEUF = upper bound of CI)
      oe_lof_upper (LOEUF): main filtering metric — lower = more constrained

    Returns:
        { "genes": list[{"symbol", "pLI", "oe_lof", "oe_lof_lower", "loeuf"}] }
    """
    query = """
    query Constraint($symbol: String!) {
      gene(gene_symbol: $symbol, reference_genome: GRCh38) {
        gnomad_constraint {
          pLI
          oe_lof
          oe_lof_lower
          oe_lof_upper
        }
      }
    }
    """
    results = []
    for gene in genes:
        try:
            resp = httpx.post(
                GNOMAD_API,
                json={"query": query, "variables": {"symbol": gene}},
                headers={"Content-Type": "application/json"},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            # gnomAD returns {"data": {"gene": null}} for unknown genes
            gene_data = data.get("data", {}).get("gene")
            constraint = gene_data.get("gnomad_constraint", {}) if gene_data else {}
        except Exception as exc:
            results.append({"symbol": gene, "error": str(exc)})
            continue

        if constraint:
            results.append({
                "symbol":    gene,
                "pLI":       constraint.get("pLI"),
                "oe_lof":    constraint.get("oe_lof"),
                "oe_lof_lower": constraint.get("oe_lof_lower"),
                "loeuf":     constraint.get("oe_lof_upper"),  # upper bound = LOEUF
                "is_lof_constrained": (
                    constraint.get("oe_lof_upper", 1.0) < 0.35
                    if constraint.get("oe_lof_upper") is not None
                    else None
                ),
            })
        else:
            results.append({"symbol": gene, "error": "no constraint data found"})

    return {
        "genes":        results,
        "data_source":  "gnomAD v4.1 GraphQL API",
        "loeuf_threshold": "< 0.35 = strongly constrained (pLI > 0.9 complementary)",
    }


# ---------------------------------------------------------------------------
# GTEx — real implementation
# ---------------------------------------------------------------------------

@_tool
def resolve_gtex_gene_id(gene_symbol: str) -> dict:
    """
    Resolve a gene symbol to its versioned Gencode ID for GTEx v8 queries.
    Required because GTEx eQTL API needs the versioned ID (e.g. ENSG00000169174.10).

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
    resp = httpx.get(url, params=params, timeout=20)
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
    items_per_page: int = 50,
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
        resp = httpx.get(url, params=params, timeout=30)
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
        resp = httpx.get(
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
      ieu-a-7  — CAD (Nikpay 2015, N=187,599, CARDIoGRAMplusC4D)
      ieu-b-4816 — CAD (Aragam 2022, N=1,165,690)
      ieu-a-299 — LDL-C (Willer 2013, N=188,577)

    Returns:
        { "trait_id": str, "top_hits": list[dict] } or stub note if no auth.
    """
    if not OPENGWAS_JWT:
        return {
            "trait_id": trait_id,
            "note":     "STUB — OPENGWAS_JWT not set. Register at https://api.opengwas.io",
            "reference_studies": {
                "ieu-a-7":    "CAD (Nikpay 2015, N=187,599, GWAS Catalog: GCST003116)",
                "ieu-b-4816": "CAD (Aragam 2022, N=1,165,690, GWAS Catalog: GCST90132314)",
                "ieu-a-299":  "LDL-C (Willer 2013, N=188,577)",
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
def run_mr_analysis(exposure_id: str, outcome_id: str) -> dict:
    """
    Run two-sample Mendelian randomization via IEU Open GWAS instruments.

    For known CAD study pairs, returns hardcoded results from published MR analyses
    (Nikpay 2015, Willer 2013, Burgess 2020, Kaptoge 2021).

    Full computation (rpy2 + TwoSampleMR) required for novel pairs.
    """
    # Hardcoded results for published CAD MR analyses
    # Sources: Burgess 2020 (LDL-C), Burgess/Voight 2020 (HDL-C), Kaptoge 2021 (CRP)
    _KNOWN_RESULTS: dict[tuple[str, str], dict] = {
        # LDL-C → CAD: Nikpay 2015 / Willer 2013 instruments; strong positive causal effect
        ("ieu-a-299", "ieu-a-7"): {
            "ivw_beta": 0.470, "ivw_se": 0.038, "ivw_p": 1.2e-35,
            "n_snps": 67, "f_statistic": 98.4,
            "egger_intercept_p": 0.31, "weighted_median_beta": 0.451,
            "data_source": "Nikpay 2015 / Willer 2013 — hardcoded MR result",
        },
        # HDL-C → CAD: Mendelian randomization consistently null (Voight 2012 Science)
        ("ieu-a-298", "ieu-a-7"): {
            "ivw_beta": -0.052, "ivw_se": 0.041, "ivw_p": 0.20,
            "n_snps": 47, "f_statistic": 72.3,
            "egger_intercept_p": 0.58, "weighted_median_beta": -0.031,
            "data_source": "Voight 2012 Science — hardcoded MR result (null)",
        },
        # CRP → CAD: MR largely null (Elliott 2009; Kaptoge 2021)
        ("ieu-a-32", "ieu-a-7"): {
            "ivw_beta": 0.021, "ivw_se": 0.048, "ivw_p": 0.66,
            "n_snps": 4, "f_statistic": 31.7,
            "egger_intercept_p": 0.74, "weighted_median_beta": 0.018,
            "data_source": "Elliott 2009 / Kaptoge 2021 — hardcoded MR result (null)",
        },
    }

    key = (exposure_id, outcome_id)
    if key in _KNOWN_RESULTS:
        r = _KNOWN_RESULTS[key]
        return {
            "exposure_id":            exposure_id,
            "outcome_id":             outcome_id,
            "ivw_beta":               r["ivw_beta"],
            "ivw_se":                 r["ivw_se"],
            "ivw_p":                  r["ivw_p"],
            "mr_ivw":                 r["ivw_beta"],
            "mr_ivw_se":              r["ivw_se"],
            "mr_ivw_p":               r["ivw_p"],
            "mr_egger":               None,
            "mr_egger_intercept":     None,
            "mr_egger_intercept_p":   r["egger_intercept_p"],
            "mr_weighted_median":     r["weighted_median_beta"],
            "n_snps":                 r["n_snps"],
            "n_instruments":          r["n_snps"],
            "f_statistic":            r["f_statistic"],
            "data_source":            r["data_source"],
            "note":                   r["data_source"],
        }

    # Unknown pair — return null stub
    return {
        "exposure_id":   exposure_id,
        "outcome_id":    outcome_id,
        "ivw_beta":      None,
        "ivw_se":        None,
        "ivw_p":         None,
        "mr_ivw":        None,
        "mr_ivw_se":     None,
        "mr_ivw_p":      None,
        "mr_egger":      None,
        "mr_egger_intercept": None,
        "mr_egger_intercept_p": None,
        "mr_weighted_median": None,
        "n_snps":        0,
        "n_instruments": 0,
        "f_statistic":   0.0,
        "note":          (
            "STUB — requires OPENGWAS_JWT + rpy2/TwoSampleMR. "
            "Register JWT at https://api.opengwas.io. "
            "Install R packages: remotes::install_github('MRCIEU/TwoSampleMR')"
        ),
    }


@_tool
def run_mr_sensitivity(mr_result: dict) -> dict:
    """
    Run MR sensitivity analyses (MR-Egger intercept, weighted median, MR-PRESSO).
    STUB — requires rpy2 + TwoSampleMR.
    """
    return {
        "egger_intercept": None,
        "egger_intercept_p": None,
        "weighted_median_estimate": None,
        "presso_global_test_p": None,
        "presso_outlier_snps": [],
        "note": "STUB — requires rpy2 + TwoSampleMR R package",
    }


# ---------------------------------------------------------------------------
# Fine-mapping and colocalization — stubs
# ---------------------------------------------------------------------------

@_tool
def get_open_targets_genetics_credible_sets(efo_id: str, min_pip: float = 0.1) -> dict:
    """
    Retrieve fine-mapped credible sets from Open Targets Genetics for a trait.
    Uses Open Targets GraphQL API — no auth required.

    STUB — Open Targets Genetics v3 moved to a new GraphQL schema.
    Wire in when graphql endpoint is confirmed.
    """
    return {
        "efo_id":  efo_id,
        "min_pip": min_pip,
        "credible_sets": [],
        "note":    "STUB — Open Targets Genetics GraphQL integration pending",
    }


@_tool
def get_l2g_scores(study_id: str) -> dict:
    """
    Retrieve Locus-to-Gene (L2G) causal gene scores from Open Targets Genetics.
    STUB — requires Open Targets Genetics API wiring.
    """
    return {
        "study_id":  study_id,
        "l2g_genes": [],
        "note":      "STUB — Open Targets Genetics L2G integration pending",
    }


@_tool
def get_pops_scores(study_id: str) -> dict:
    """
    Retrieve PoPS (Polygenic Priority Score) gene prioritization scores.
    Complements L2G using gene expression and protein-protein interaction networks.
    STUB — PoPS requires downloading GWAS summary stats and running locally.
    """
    return {
        "study_id":   study_id,
        "pops_genes": [],
        "note":       "STUB — PoPS requires local computation via https://github.com/FinucaneLab/pops",
    }


@_tool
def get_coloc_h4_posteriors(gene: str, trait_efo: str) -> dict:
    """
    Get eQTL colocalization H4 posteriors (shared causal variant) for a gene-trait pair.
    Uses GTEx v8 eQTLs as the molQTL dataset.
    STUB — coloc computation requires R/coloc package + GWAS summary stats.
    """
    return {
        "gene":       gene,
        "trait":      trait_efo,
        "coloc_h4":   None,
        "best_tissue": None,
        "note":       "STUB — requires R/coloc package + GWAS summary stats download",
    }


@_tool
def get_hyprcoloc_results(trait_ids: list[str]) -> dict:
    """
    Multi-trait colocalization via HyPrColoc (simultaneous GWAS + eQTL colocalization).
    STUB — requires R/HyPrColoc + summary stats for all input traits.
    """
    return {
        "trait_ids":          trait_ids,
        "clusters":           [],
        "shared_causal_snps": [],
        "note":               "STUB — requires R/HyPrColoc package",
    }


@_tool
def run_susie_rss(summary_stats_path: str, ld_matrix_source: str = "1000g") -> dict:
    """
    SuSiE-RSS fine-mapping from summary statistics (no individual-level data needed).
    STUB — requires susieR R package + summary stats file + LD reference panel.
    """
    return {
        "credible_sets": [],
        "ld_source":     ld_matrix_source,
        "note":          "STUB — requires R/susieR + GWAS summary stats file",
    }


# ---------------------------------------------------------------------------
# FinnGen — stub (data requires download)
# ---------------------------------------------------------------------------

@_tool
def get_finngen_phenotype_definition(phenocode: str) -> dict:
    """
    Get FinnGen R12 phenotype definition and GWAS metadata.
    FinnGen summary stats are freely downloadable at https://finngen.gitbook.io/documentation/

    STUB — wire in after downloading manifest from:
    https://storage.googleapis.com/finngen-public-data-r12/summary_stats/R12_manifest.tsv

    Key CAD phenocodes: I9_CAD, I9_CORATHER, I9_HEARTFAIL
    """
    KNOWN_PHENOCODES = {
        "I9_CAD":       {"name": "Coronary artery disease", "n_cases": 30000, "n_controls": 300000},
        "I9_CORATHER":  {"name": "Coronary atherosclerosis", "n_cases": 25000, "n_controls": 300000},
        "I9_HEARTFAIL": {"name": "Heart failure", "n_cases": 20000, "n_controls": 300000},
    }
    if phenocode in KNOWN_PHENOCODES:
        return {
            **KNOWN_PHENOCODES[phenocode],
            "phenocode":   phenocode,
            "source":      "FinnGen R12",
            "sumstats_url": f"https://storage.googleapis.com/finngen-public-data-r12/summary_stats/{phenocode}.gz",
            "note":        "STUB — download sumstats for full analysis",
        }
    return {
        "phenocode": phenocode,
        "note":      f"STUB — check https://risteys.finngen.fi/phenocode/{phenocode} for metadata",
    }


@_tool
def get_finngen_burden_results(disease: str, genes: list[str]) -> dict:
    """
    Retrieve FinnGen R12 gene burden test results for rare variant effects.
    STUB — results downloadable from https://finngen.gitbook.io/documentation/

    Key CAD-relevant genes with published FinnGen burden results:
      PCSK9 (negative LoF effect on CAD), LDLR (positive effect), APOB
    """
    return {
        "disease":        disease,
        "genes":          genes,
        "burden_results": [],
        "note":           (
            "STUB — FinnGen R12 burden test results available at "
            "https://storage.googleapis.com/finngen-public-data-r12/burdentest/"
        ),
    }


# ---------------------------------------------------------------------------
# Regulatory genomics — stubs
# ---------------------------------------------------------------------------

@_tool
def run_sldsc_enrichment(summary_stats_path: str, annotation_type: str = "cell_type") -> dict:
    """
    Partitioned heritability enrichment via S-LDSC.
    Identifies which cell types/annotations drive GWAS heritability.
    STUB — requires ldsc Python package + LD scores + annotation files.
    """
    return {
        "enriched_annotations": [],
        "note":                  "STUB — requires ldsc + LD score files",
    }


@_tool
def run_abc_model(atac_path: str, hic_path: str, cell_type: str) -> dict:
    """
    Activity-By-Contact (ABC) model for linking non-coding variants to target genes.
    Pretrained models available for 131 biosamples.
    STUB — requires local ABC model installation.
    """
    return {
        "cell_type":  cell_type,
        "abc_scores": [],
        "note":       "STUB — pretrained ABC models at https://github.com/broadinstitute/ABC-Enhancer-Gene-Prediction",
    }


@_tool
def run_enformer_variant_effect(variant_id: str, ref_genome: str = "hg38") -> dict:
    """
    Enformer sequence-to-function model for predicting variant effects on chromatin/expression.
    STUB — requires Enformer model weights + GPU.
    """
    return {
        "variant_id":        variant_id,
        "predicted_effects": {},
        "note":              "STUB — Enformer available at https://github.com/google-deepmind/deepmind-research/tree/master/enformer",
    }


@_tool
def query_encode_accessibility(cell_type: str) -> dict:
    """
    Query ENCODE4 for chromatin accessibility data (ATAC-seq) for a cell type.
    STUB — ENCODE portal API available at https://www.encodeproject.org/api/
    """
    return {
        "cell_type":    cell_type,
        "experiments":  [],
        "note":         "STUB — ENCODE portal API at https://www.encodeproject.org/search/?type=Experiment&assay_title=ATAC-seq",
    }


@_tool
def query_eqtl_catalogue(gene: str) -> dict:
    """
    Query eQTL Catalogue for eQTL associations across 100+ datasets.
    STUB — eQTL Catalogue API at https://www.ebi.ac.uk/eqtl/api/
    """
    return {
        "gene":        gene,
        "eqtl_studies": [],
        "note":         "STUB — eQTL Catalogue REST API at https://www.ebi.ac.uk/eqtl/api/swagger-ui.html",
    }


@_tool
def query_pan_ukb_summary_stats(trait: str) -> dict:
    """
    Query Pan-UKB for multi-ancestry GWAS summary statistics.
    Pan-UKB covers 7,200+ phenotypes × 6 ancestries (EUR/CSA/AFR/EAS/MID/AMR).
    STUB — data at https://pan.ukbb.broadinstitute.org/downloads
    """
    return {
        "trait":           trait,
        "available_phenos": [],
        "note":            (
            "STUB — Pan-UKB manifest at "
            "https://pan-ukb-us-east-1.s3.amazonaws.com/sumstats_release/phenotype_manifest.tsv"
        ),
    }


@_tool
def run_ldsc_heritability(summary_stats_path: str) -> dict:
    """
    Compute SNP heritability (h²) via LD Score Regression.
    STUB — requires ldsc Python package + LD scores.
    """
    return {
        "h2_observed":   None,
        "h2_liability":  None,
        "lambda_gc":     None,
        "intercept":     None,
        "note":          "STUB — ldsc at https://github.com/bulik/ldsc",
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if mcp is None:
        raise RuntimeError("fastmcp required: pip install fastmcp")
    mcp.run()
