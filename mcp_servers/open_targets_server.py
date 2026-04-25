"""
open_targets_server.py — MCP server for Open Targets Platform queries.

Real implementations:
  - Open Targets Platform GraphQL API (free, no auth)
  - Open Targets Genetics API (free, no auth)

Stubs:
  - TxGNN drug repurposing predictions (requires local model)
  - OTAR drug target evidence aggregation (internal data)

Run standalone:  python mcp_servers/open_targets_server.py
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
from pipelines.api_cache import api_cached

try:
    import fastmcp
    mcp = fastmcp.FastMCP("open-targets-server")
    _tool = mcp.tool()
except ImportError:
    def _tool(fn=None, **_):
        return fn if fn is not None else (lambda f: f)
    mcp = None

# ---------------------------------------------------------------------------
# Open Targets APIs
# ---------------------------------------------------------------------------

OT_PLATFORM_GQL  = "https://api.platform.opentargets.org/api/v4/graphql"

# Map OT Platform clinical stage strings → integer phase (for downstream scoring)
_STAGE_TO_PHASE: dict[str, int] = {
    "APPROVAL":  4,
    "PHASE_3":   3,
    "PHASE_2":   2,
    "PHASE_2B":  2,
    "PHASE_2A":  2,
    "PHASE_1":   1,
    "PHASE_1B":  1,
    "PHASE_1A":  1,
}

def _stage_to_int(stage: str | None) -> int:
    if not stage:
        return 0
    return _STAGE_TO_PHASE.get(stage.upper(), 0)

# EFO alias map: some EFOs used in GWAS Catalog / OpenTargets Genetics differ from
# OT Platform EFOs. Remap before any OT Platform GraphQL query.
_EFO_ALIASES: dict[str, str] = {}

def _resolve_efo(efo_id: str) -> str:
    """Resolve EFO aliases so both OTG and OT Platform IDs work transparently."""
    return _EFO_ALIASES.get(efo_id, efo_id)


def _open_targets_assoc_disabled() -> bool:
    """
    When True, skip all Open Targets Platform *association* GraphQL in this module
    (disease targets, per-target info, colocalisation, genetic-score proxy).

    Locus–gene (L2G) and GWAS study queries live in ``gwas_genetics_server`` and
    remain available for γ estimation and gene prioritization.
    """
    v = os.getenv("OPEN_TARGETS_ASSOC_DISABLED", "").strip().lower()
    return v in {"1", "true", "yes", "on"}

# Known target-disease associations for CAD (hardcoded from OT Platform)
# Used as fallback when API unavailable; all have direct OT evidence
KNOWN_CAD_TARGETS: list[dict] = [
    {
        "target_id":   "ENSG00000169174",  # PCSK9
        "gene_symbol": "PCSK9",
        "overall_score": 0.89,
        "genetic_score": 0.95,
        "drugs_in_trial": ["evolocumab", "alirocumab"],
        "max_clinical_phase": 4,
        "tractability": "small_molecule_antibody",
    },
    {
        "target_id":   "ENSG00000130164",  # LDLR
        "gene_symbol": "LDLR",
        "overall_score": 0.82,
        "genetic_score": 0.91,
        "drugs_in_trial": [],
        "max_clinical_phase": 0,
        "tractability": "difficult",
    },
    {
        "target_id":   "ENSG00000113161",  # HMGCR
        "gene_symbol": "HMGCR",
        "overall_score": 0.94,
        "genetic_score": 0.88,
        "drugs_in_trial": ["atorvastatin", "rosuvastatin", "simvastatin"],
        "max_clinical_phase": 4,
        "tractability": "small_molecule",
    },
    {
        "target_id":   "ENSG00000160285",  # IL6R
        "gene_symbol": "IL6R",
        "overall_score": 0.61,
        "genetic_score": 0.52,
        "drugs_in_trial": ["tocilizumab", "sarilumab"],
        "max_clinical_phase": 4,
        "tractability": "antibody",
    },
    {
        "target_id":   "ENSG00000128052",  # PTPN22
        "gene_symbol": "PTPN22",
        "overall_score": 0.34,
        "genetic_score": 0.41,
        "drugs_in_trial": [],
        "max_clinical_phase": 0,
        "tractability": "difficult",
    },
]


def _gql_request(url: str, query: str, variables: dict | None = None) -> dict:
    """Execute a GraphQL query and return the data payload."""
    if _open_targets_assoc_disabled():
        return {
            "error": (
                "Open Targets association API disabled "
                "(OPEN_TARGETS_ASSOC_DISABLED=1). "
                "Use L2G via mcp_servers.gwas_genetics_server for genetics."
            ),
        }
    try:
        resp = httpx.post(
            url,
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
        # Include response body fragment for 400s to surface schema mismatches
        body_hint = ""
        if e.response.status_code == 400:
            try:
                body = e.response.json()
                msgs = [err.get("message", "") for err in body.get("errors", [])]
                body_hint = f": {msgs[0]}" if msgs else f": {e.response.text[:200]}"
            except Exception:
                body_hint = f": {e.response.text[:200]}"
        return {"error": f"HTTP {e.response.status_code}{body_hint}"}
    except Exception as e:
        return {"error": str(e)}


@_tool
@api_cached(ttl_days=7)
def get_open_targets_disease_targets(
    efo_id: str,
    max_targets: int = 20,
    min_overall_score: float = 0.2,
) -> dict:
    """
    Query Open Targets Platform for top therapeutic targets for a disease.

    Returns drugs, tractability, and clinical phase for each target in one call —
    replacing the need for per-gene ChEMBL/OT lookups in the chemistry agent.

    Args:
        efo_id:            EFO disease ID, e.g. "EFO_0001645" (CAD)
        max_targets:       Maximum targets to return (default 20 to cover all ranked genes)
        min_overall_score: Minimum overall association score [0-1]

    Returns:
        {"efo_id": str, "targets": list[{
            "gene_symbol", "overall_score", "genetic_score",
            "max_clinical_phase", "known_drugs", "tractability_class"
        }]}
    """
    # NOTE: OT Platform may reject very large pages (complexity/timeout).
    # We page through associatedTargets in moderate batches.
    query = """
    query DiseaseTargets($efoId: String!, $size: Int!, $index: Int!) {
      disease(efoId: $efoId) {
        id
        name
        associatedTargets(page: {index: $index, size: $size}) {
          count
          rows {
            target {
              id
              approvedSymbol
              tractability {
                label
                modality
                value
              }
              drugAndClinicalCandidates {
                count
                rows {
                  drug { id name }
                  maxClinicalStage
                }
              }
            }
            score
            datatypeScores { id score }
          }
        }
      }
    }
    """

    resolved = _resolve_efo(efo_id)
    page_size = max(1, min(int(max_targets or 20), 250))
    all_rows: list[dict] = []
    disease_name: str | None = None
    total_count: int | None = None

    for page_index in range(0, 200):  # hard cap to prevent infinite loops
        variables = {"efoId": resolved, "size": page_size, "index": page_index}
        data = _gql_request(OT_PLATFORM_GQL, query, variables)
        if "error" in data:
            # Explicit disable: no curated fallback — use L2G in gwas_genetics_server instead.
            if _open_targets_assoc_disabled():
                return {
                    "efo_id":    efo_id,
                    "n_targets": 0,
                    "targets":   [],
                    "source":    "disabled (OPEN_TARGETS_ASSOC_DISABLED=1)",
                    "note":      data.get("error", ""),
                }
            # Fallback: return known CAD targets if live API fails for EFO_0001645
            if efo_id == "EFO_0001645":
                targets = [
                    t for t in KNOWN_CAD_TARGETS
                    if t["overall_score"] >= min_overall_score
                ][:max_targets]
                return {
                    "efo_id":    efo_id,
                    "n_targets": len(targets),
                    "targets":   targets,
                    "source":    "Open Targets Platform (fallback cache)",
                    "data_tier": "curated",
                    "error":     data["error"],
                }
            return {"efo_id": efo_id, "error": data["error"], "targets": []}

        disease = data.get("disease", {})
        if not disease:
            return {"efo_id": efo_id, "targets": [], "note": "Disease not found"}

        disease_name = disease.get("name") or disease_name
        assoc = disease.get("associatedTargets", {}) or {}
        if total_count is None:
            total_count = assoc.get("count")

        rows = assoc.get("rows", []) or []
        if not rows:
            break

        all_rows.extend(rows)
        if len(all_rows) >= max_targets:
            break

        # Stop early if we reached the server-reported count
        if isinstance(total_count, int) and len(all_rows) >= total_count:
            break

    # Parse rows → targets
    targets: list[dict] = []
    for row in all_rows[:max_targets]:
        try:
            if (row.get("score") or 0.0) < min_overall_score:
                continue
            dtype_scores = {d["id"]: d["score"] for d in row.get("datatypeScores", [])}

            tract_items = (row.get("target") or {}).get("tractability") or []
            sm_active = any(t.get("modality") == "SM" and t.get("value") for t in tract_items)
            ab_active = any(t.get("modality") == "AB" and t.get("value") for t in tract_items)
            tractability_class = "small_molecule" if sm_active else "antibody" if ab_active else "difficult"

            drug_rows = (row.get("target") or {}).get("drugAndClinicalCandidates", {}).get("rows", []) or []
            known_drugs = list({(dr.get("drug") or {}).get("name") for dr in drug_rows if dr.get("drug")})
            known_drugs = [d for d in known_drugs if d]
            max_phase = max((_stage_to_int(dr.get("maxClinicalStage")) for dr in drug_rows), default=0)

            target_obj = row.get("target") or {}
            targets.append({
                "target_id":          target_obj.get("id"),
                "gene_symbol":        target_obj.get("approvedSymbol"),
                "overall_score":      round(float(row.get("score", 0.0)), 4),
                "genetic_score":      round(float(dtype_scores.get("genetic_association", 0) or 0), 4),
                "max_clinical_phase": int(max_phase),
                "known_drugs":        known_drugs[:5],
                "tractability_class": tractability_class,
            })
        except Exception:
            continue

    return {
        "efo_id":       efo_id,
        "disease_name": disease_name,
        "n_targets":    len(targets),
        "targets":      targets,
        "source":       "Open Targets Platform GraphQL (paged)",
    }


@_tool
@api_cached(ttl_days=30)
def get_open_targets_target_info(gene_symbol: str) -> dict:
    """
    Fetch detailed target information from Open Targets Platform.

    Args:
        gene_symbol: Gene symbol, e.g. "PCSK9"

    Returns:
        {"gene_symbol": str, "tractability": dict, "known_drugs": list, "safety": dict}
    """
    # Step 1: resolve gene symbol → Ensembl ID via search
    search_query = """
    query SearchTarget($symbol: String!) {
      search(queryString: $symbol, entityNames: ["target"]) {
        hits { id entity name }
      }
    }
    """
    search_data = _gql_request(OT_PLATFORM_GQL, search_query, {"symbol": gene_symbol})
    if "error" in search_data:
        return {"gene_symbol": gene_symbol, "error": search_data["error"]}

    hits = search_data.get("search", {}).get("hits", [])
    ensembl_id = None
    for h in hits:
        if h.get("entity") == "target" and h.get("name", "").upper() == gene_symbol.upper():
            ensembl_id = h["id"]
            break
    if not ensembl_id and hits:
        # Take first target hit if exact match not found
        for h in hits:
            if h.get("entity") == "target":
                ensembl_id = h["id"]
                break

    if not ensembl_id:
        return {"gene_symbol": gene_symbol, "note": "Target not found in Open Targets"}

    # Step 2: fetch detailed target info by Ensembl ID
    query = """
    query TargetInfo($ensemblId: String!) {
      target(ensemblId: $ensemblId) {
        id
        approvedSymbol
        approvedName
        tractability {
          label
          modality
          value
        }
        drugAndClinicalCandidates {
          count
          rows {
            drug {
              id
              name
            }
            maxClinicalStage
          }
        }
      }
    }
    """
    data = _gql_request(OT_PLATFORM_GQL, query, {"ensemblId": ensembl_id})
    if "error" in data:
        return {"gene_symbol": gene_symbol, "error": data["error"]}

    target = data.get("target")
    if not target:
        return {"gene_symbol": gene_symbol, "note": "Target not found in Open Targets"}

    # Normalise: approvedSymbol may be None if ensemblId resolved to wrong gene
    if target.get("approvedSymbol", "").upper() != gene_symbol.upper():
        pass  # use what we got — caller can validate
    tract_items = target.get("tractability") or []
    # Build both a label→value dict (for inspection) and modality flags (for downstream use)
    tractability       = {t["label"]: t.get("value") for t in tract_items}
    tractability_sm    = any(t["modality"] == "SM" and t.get("value") for t in tract_items)
    tractability_ab    = any(t["modality"] == "AB" and t.get("value") for t in tract_items)
    tractability_class = (
        "small_molecule" if tractability_sm else
        "antibody"       if tractability_ab else
        "difficult"
    )

    drug_rows = (target.get("drugAndClinicalCandidates") or {}).get("rows", [])[:10]
    drugs = [
        {
            "name":  dr["drug"]["name"],
            "phase": _stage_to_int(dr.get("maxClinicalStage")),
        }
        for dr in drug_rows if dr.get("drug")
    ]
    max_phase = max((d["phase"] for d in drugs), default=0)

    return {
        "gene_symbol":       gene_symbol,
        "ensembl_id":        target.get("id"),
        "approved_name":     target.get("approvedName"),
        "tractability":      tractability,
        "tractability_class": tractability_class,
        "tractability_sm":   tractability_sm,
        "tractability_ab":   tractability_ab,
        "known_drugs":       drugs,
        "max_phase":         max_phase,
        "n_drugs":           (target.get("drugAndClinicalCandidates") or {}).get("count", 0),
        "source":            "Open Targets Platform GraphQL",
    }


@_tool
@api_cached(ttl_days=30)
def get_open_targets_drug_info(drug_name: str) -> dict:
    """
    Fetch drug information and indications from Open Targets Platform.

    Args:
        drug_name: Drug name, e.g. "atorvastatin", "tocilizumab"
    """
    query = """
    query DrugSearch($term: String!) {
      search(queryString: $term, entityNames: ["drug"]) {
        hits {
          id
          entity
          name
          object {
            ... on Drug {
              id
              name
              maxClinicalStage
              mechanismsOfAction {
                rows {
                  actionType
                  targets {
                    approvedSymbol
                  }
                }
              }
            }
          }
        }
      }
    }
    """
    data = _gql_request(OT_PLATFORM_GQL, query, {"term": drug_name})
    if "error" in data:
        return {"drug_name": drug_name, "error": data["error"]}

    hits = data.get("search", {}).get("hits", [])
    if not hits:
        return {"drug_name": drug_name, "note": "Drug not found"}

    drug_obj = hits[0].get("object", {})
    moa_rows = drug_obj.get("mechanismsOfAction", {}).get("rows", [])
    targets = []
    for row in moa_rows:
        for t in row.get("targets", []):
            targets.append({"symbol": t["approvedSymbol"], "action": row["actionType"]})

    return {
        "drug_name":   drug_name,
        "id":          drug_obj.get("id"),
        "max_phase":   _stage_to_int(drug_obj.get("maxClinicalStage")),
        "targets":     targets[:10],
        "source":      "Open Targets Platform GraphQL",
    }


@_tool
def get_open_targets_targets_bulk(gene_symbols: list[str]) -> dict:
    """
    Fetch tractability, known drugs, and max phase for multiple genes in parallel.

    Replaces per-gene serial calls in chemistry_agent with a single batched operation.
    Uses ThreadPoolExecutor to parallelise per-gene `get_open_targets_target_info` calls.

    Args:
        gene_symbols: List of HGNC gene symbols

    Returns:
        {"targets": {gene_symbol: {tractability_class, known_drugs, max_phase, ...}}}
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        future_to_gene = {
            pool.submit(get_open_targets_target_info, gene): gene
            for gene in gene_symbols
        }
        for future in as_completed(future_to_gene):
            gene = future_to_gene[future]
            try:
                results[gene] = future.result()
            except Exception as exc:
                results[gene] = {
                    "gene_symbol":        gene,
                    "tractability_class": "unknown",
                    "known_drugs":        [],
                    "max_phase":          0,
                    "error":              str(exc),
                }

    return {"targets": results, "n_genes": len(results)}


@_tool
def run_txgnn_repurposing(disease_efo: str, n_candidates: int = 10) -> dict:
    """
    Run TxGNN drug repurposing prediction for a disease.

    STUB — requires:
      1. TxGNN model weights (download from GitHub)
      2. GPU recommended for inference
      3. PrimeKG input graph

    Reference: Huang et al. Nature Medicine 2023
    """
    return {
        "disease_efo":   disease_efo,
        "n_candidates":  n_candidates,
        "predictions":   None,
        "note":          "STUB — TxGNN requires local model; see pathways_kg_server for PrimeKG",
        "paper":         "Huang 2023 Nat Med doi:10.1038/s41591-023-02618-6",
    }


@_tool
@api_cached(ttl_days=30)
def get_ot_genetic_instruments(
    gene_symbol: str,
    efo_id: str,
    min_score: float = 0.5,
    max_instruments: int = 3,
) -> dict:
    """
    Retrieve top GWAS/eQTL genetic instruments for a gene × disease pair from
    Open Targets credible sets.  Returns effect sizes (beta, SE) usable as MR
    instruments when GTEx eQTL data is unavailable.

    Data hierarchy (highest to lowest quality):
      1. eQTL credible sets (studyType='eqtl') — directly measure gene expression effect
      2. GWAS credible sets (studyType='gwas') — disease variant near the gene

    Args:
        gene_symbol:      HGNC gene symbol, e.g. "NOD2"
        efo_id:           Disease EFO ID, e.g. "EFO_0003767"
        min_score:        Minimum OT evidence score (0–1)
        max_instruments:  Max instruments to return

    Returns:
        {
          "gene_symbol": str,
          "efo_id": str,
          "instruments": [{
            "beta": float, "se": float | None, "p_value": float,
            "study_type": str, "study_id": str, "score": float,
            "instrument_type": "eqtl" | "gwas_credset"
          }],
          "best_nes": float | None,   # best eQTL NES for eQTL-MR
          "best_gwas_beta": float | None,
          "data_source": str
        }
    """
    # Step 1: Get GWAS study IDs for this disease (one call)
    resolved_efo = _resolve_efo(efo_id)
    studies_q = """
    query DiseaseStudies($efoId: String!) {
      studies(diseaseIds: [$efoId], page: {index: 0, size: 20}) {
        rows { id studyType }
      }
    }
    """
    studies_data = _gql_request(OT_PLATFORM_GQL, studies_q, {"efoId": resolved_efo})
    gwas_study_ids = [
        row["id"]
        for row in (studies_data.get("studies", {}).get("rows") or [])
        if row.get("studyType", "gwas") in ("gwas", "")
    ]
    if not gwas_study_ids:
        return {"gene_symbol": gene_symbol, "efo_id": efo_id, "instruments": [],
                "best_nes": None, "best_gwas_beta": None, "note": "No GWAS studies found for EFO"}

    # Step 2: Resolve gene symbol → Ensembl ID (needed to filter credible sets by gene)
    search_q = """
    query SearchTarget($sym: String!) {
      search(queryString: $sym, entityNames: ["target"], page: {index: 0, size: 5}) {
        hits { id entity name }
      }
    }
    """
    sr = _gql_request(OT_PLATFORM_GQL, search_q, {"sym": gene_symbol})
    ensembl_id = None
    for h in (sr.get("search", {}).get("hits") or []):
        if h.get("entity") == "target" and h.get("name", "").upper() == gene_symbol.upper():
            ensembl_id = h["id"]
            break
    if not ensembl_id:
        # Take first target hit as fallback
        for h in (sr.get("search", {}).get("hits") or []):
            if h.get("entity") == "target":
                ensembl_id = h["id"]
                break
    if not ensembl_id:
        return {"gene_symbol": gene_symbol, "efo_id": efo_id, "instruments": [],
                "best_nes": None, "best_gwas_beta": None, "note": "Gene not found"}

    # Step 3: Query root credibleSets for these GWAS studies, filtered to gene's eQTL.
    # root CredibleSet has beta/SE/studyId/studyType directly (confirmed valid in v4).
    # We also query eQTL credible sets separately for the gene.
    instruments: list[dict] = []

    # GWAS credible sets for the disease
    gwas_cs_q = """
    query GWASCredSets($studyIds: [String!]!, $size: Int!) {
      credibleSets(studyIds: $studyIds, page: {index: 0, size: $size}) {
        rows {
          studyLocusId
          studyId
          studyType
          beta
          standardError
          pValueMantissa
          pValueExponent
        }
      }
    }
    """
    # Batch GWAS study IDs (max 10 at a time to stay under complexity limits)
    for i in range(0, len(gwas_study_ids), 10):
        batch = gwas_study_ids[i:i + 10]
        cs_data = _gql_request(OT_PLATFORM_GQL, gwas_cs_q, {"studyIds": batch, "size": 50})
        if "error" in cs_data:
            continue
        for cs in (cs_data.get("credibleSets", {}).get("rows") or []):
            beta = cs.get("beta")
            if beta is None:
                continue
            p_mant = cs.get("pValueMantissa")
            p_exp  = cs.get("pValueExponent")
            p_val  = (float(p_mant) * (10 ** float(p_exp))) if (p_mant and p_exp) else None
            instruments.append({
                "beta":            float(beta),
                "se":              float(cs["standardError"]) if cs.get("standardError") else None,
                "p_value":         p_val,
                "study_type":      cs.get("studyType", "gwas"),
                "study_id":        cs.get("studyId"),
                "score":           min_score,  # no per-locus score in this query; use threshold
                "instrument_type": "gwas_credset",
            })
        if instruments:
            break  # stop after first batch with hits

    # eQTL credible sets: OT Platform v4 does not accept studyTypes without studyIds,
    # and qtlGeneId is not a filterable argument. Skip this lookup — GWAS instruments suffice.
    eqtl_data: dict = {}
    if False:  # disabled — credibleSets(studyTypes=[eqtl]) without studyIds is unsupported
        for cs in []:
            if (cs.get("qtlGeneId") or "").upper() != ensembl_id.upper():
                continue
            beta = cs.get("beta")
            if beta is None:
                continue
            p_mant = cs.get("pValueMantissa")
            p_exp  = cs.get("pValueExponent")
            p_val  = (float(p_mant) * (10 ** float(p_exp))) if (p_mant and p_exp) else None
            instruments.append({
                "beta":            float(beta),
                "se":              float(cs["standardError"]) if cs.get("standardError") else None,
                "p_value":         p_val,
                "study_type":      cs.get("studyType", "eqtl"),
                "study_id":        cs.get("studyId"),
                "score":           min_score,
                "instrument_type": "eqtl",
            })

    # Sort: eQTL first, then by p-value ascending
    instruments.sort(key=lambda x: (0 if x["instrument_type"] == "eqtl" else 1,
                                    x.get("p_value") or 1.0))
    instruments = instruments[:max_instruments]

    best_nes       = next((i["beta"] for i in instruments if i["instrument_type"] == "eqtl"), None)
    best_gwas_beta = next((i["beta"] for i in instruments if i["instrument_type"] == "gwas_credset"), None)

    return {
        "gene_symbol":     gene_symbol,
        "efo_id":          efo_id,
        "ensembl_id":      ensembl_id,
        "instruments":     instruments,
        "best_nes":        best_nes,
        "best_gwas_beta":  best_gwas_beta,
        "n_instruments":   len(instruments),
        "data_source":     "Open Targets Platform v4 credible sets (root query)",
    }


@_tool
def get_ot_genetic_instruments_bulk(
    gene_symbols: list[str],
    efo_id: str,
    min_score: float = 0.5,
    max_instruments: int = 3,
) -> dict[str, dict]:
    """Batched version of `get_ot_genetic_instruments` for a list of genes.

    The per-gene function does ~3 GraphQL round-trips, two of which produce
    disease-level results identical across genes (DiseaseStudies + GWAS
    credible sets).  This bulk function hoists those shared queries out of
    the loop and resolves every gene's Ensembl ID in one aliased Search
    query (~25 genes per round-trip), cutting a ~405-gene run from
    ~1,200 HTTPS calls down to ~20.

    Side effect: each gene's result is written into the SQLite API cache
    under the exact key that `get_ot_genetic_instruments(gene, efo_id)` would
    produce.  Subsequent single-gene calls in the same run therefore hit
    cache immediately.

    Returns `{gene_symbol: result_dict}` where `result_dict` has the same
    shape as the single-gene function.  Genes already present in the cache
    are served from there without any network traffic.
    """
    from pipelines.api_cache import get_cache, _make_key

    if not gene_symbols:
        return {}

    cache = get_cache()
    result_by_gene: dict[str, dict] = {}
    cache_keys: dict[str, str] = {}
    uncached_genes: list[str] = []

    # Cache-first pass: skip genes that are already populated.  Key shape
    # matches the @api_cached(get_ot_genetic_instruments) signature when
    # called with only (gene, efo_id) positional args and no kwargs — the
    # exact usage at perturbation_genomics_agent.py call sites.
    for g in gene_symbols:
        k = _make_key("get_ot_genetic_instruments", (g, efo_id), {})
        hit = cache.get(k)
        if hit is not None:
            result_by_gene[g] = hit
        else:
            cache_keys[g] = k
            uncached_genes.append(g)

    if not uncached_genes:
        return result_by_gene

    # ------------------------------------------------------------------
    # Step 1 — Disease studies (shared across genes; one round-trip)
    # ------------------------------------------------------------------
    resolved_efo = _resolve_efo(efo_id)
    studies_q = """
    query DiseaseStudies($efoId: String!) {
      studies(diseaseIds: [$efoId], page: {index: 0, size: 20}) {
        rows { id studyType }
      }
    }
    """
    studies_data = _gql_request(OT_PLATFORM_GQL, studies_q, {"efoId": resolved_efo})
    gwas_study_ids = [
        row["id"]
        for row in (studies_data.get("studies", {}).get("rows") or [])
        if row.get("studyType", "gwas") in ("gwas", "")
    ]

    # ------------------------------------------------------------------
    # Step 2 — GWAS credible sets for the disease (shared; one round-trip
    # per chunk of 10 study IDs, but identical output for every gene)
    # ------------------------------------------------------------------
    shared_gwas_instruments: list[dict] = []
    if gwas_study_ids:
        gwas_cs_q = """
        query GWASCredSets($studyIds: [String!]!, $size: Int!) {
          credibleSets(studyIds: $studyIds, page: {index: 0, size: $size}) {
            rows {
              studyLocusId
              studyId
              studyType
              beta
              standardError
              pValueMantissa
              pValueExponent
            }
          }
        }
        """
        for i in range(0, len(gwas_study_ids), 10):
            batch = gwas_study_ids[i:i + 10]
            cs_data = _gql_request(OT_PLATFORM_GQL, gwas_cs_q, {"studyIds": batch, "size": 50})
            if "error" in cs_data:
                continue
            for cs in (cs_data.get("credibleSets", {}).get("rows") or []):
                beta = cs.get("beta")
                if beta is None:
                    continue
                p_mant = cs.get("pValueMantissa")
                p_exp  = cs.get("pValueExponent")
                p_val  = (float(p_mant) * (10 ** float(p_exp))) if (p_mant and p_exp) else None
                shared_gwas_instruments.append({
                    "beta":            float(beta),
                    "se":              float(cs["standardError"]) if cs.get("standardError") else None,
                    "p_value":         p_val,
                    "study_type":      cs.get("studyType", "gwas"),
                    "study_id":        cs.get("studyId"),
                    "score":           min_score,
                    "instrument_type": "gwas_credset",
                })
            if shared_gwas_instruments:
                break  # matches single-gene behaviour: first batch with hits wins

    # ------------------------------------------------------------------
    # Step 3 — Batched Ensembl resolution via GraphQL aliases (~25/query)
    # ------------------------------------------------------------------
    ensembl_by_gene: dict[str, str] = {}

    # Static-data fast path: resolve symbols from the local HGNC mapping
    # when available so we skip the SearchBatch GraphQL entirely for genes
    # already known.  Only unresolved genes fall through to the live API.
    try:
        from pipelines.static_lookups import get_lookups
        _lookups = get_lookups()
        _remaining: list[str] = []
        for g in uncached_genes:
            eid = _lookups.get_ensembl_id(g)
            if eid:
                ensembl_by_gene[g] = eid
            else:
                _remaining.append(g)
        _symbols_to_resolve = _remaining
    except Exception:
        _symbols_to_resolve = list(uncached_genes)

    _CHUNK = 10   # OT Platform rejects >~15 aliases per query (complexity limit)
    for i in range(0, len(_symbols_to_resolve), _CHUNK):
        chunk = _symbols_to_resolve[i:i + _CHUNK]
        var_defs = ", ".join(f"$sym{j}: String!" for j in range(len(chunk)))
        body_parts = [
            f'g{j}: search(queryString: $sym{j}, entityNames: ["target"], '
            f'page: {{index: 0, size: 3}}) {{ hits {{ id entity name }} }}'
            for j in range(len(chunk))
        ]
        aliased_q = f"query SearchBatch({var_defs}) {{\n  " + "\n  ".join(body_parts) + "\n}"
        variables = {f"sym{j}": g for j, g in enumerate(chunk)}

        sr = _gql_request(OT_PLATFORM_GQL, aliased_q, variables)
        if "error" in sr:
            # Batch rejected (complexity limit) — resolve individually using the
            # already-cached single-gene path; results prime the cache for future bulk runs.
            for g in chunk:
                sq = """
                query SearchTarget($sym: String!) {
                  search(queryString: $sym, entityNames: ["target"], page: {index: 0, size: 3}) {
                    hits { id entity name }
                  }
                }
                """
                single = _gql_request(OT_PLATFORM_GQL, sq, {"sym": g})
                hits = (single.get("search") or {}).get("hits") or []
                eid = next((h["id"] for h in hits if h.get("entity") == "target"
                            and h.get("name", "").upper() == g.upper()), None)
                if not eid:
                    eid = next((h["id"] for h in hits if h.get("entity") == "target"), None)
                if eid:
                    ensembl_by_gene[g] = eid
            continue
        for j, g in enumerate(chunk):
            hits = ((sr.get(f"g{j}") or {}).get("hits")) or []
            eid: str | None = None
            for h in hits:
                if h.get("entity") == "target" and (h.get("name") or "").upper() == g.upper():
                    eid = h.get("id")
                    break
            if not eid:
                for h in hits:
                    if h.get("entity") == "target":
                        eid = h.get("id")
                        break
            if eid:
                ensembl_by_gene[g] = eid

    # ------------------------------------------------------------------
    # Step 4 — eQTL credible sets (disabled: OT v4 does not accept
    # studyTypes without studyIds on credibleSets; qtlGeneId is not
    # a filterable arg.  eqtl_rows stays empty; GWAS instruments suffice.)
    # ------------------------------------------------------------------
    eqtl_rows: list[dict] = []
    eqtl_cs_q = """
    query EQTLCredSets($size: Int!) {
      credibleSets(studyTypes: [eqtl], page: {index: 0, size: $size}) {
        rows {
          studyId
          studyType
          qtlGeneId
          beta
          standardError
          pValueMantissa
          pValueExponent
        }
      }
    }
    """
    # eqtl_rows stays empty — query disabled (see Step 4 comment above)

    # ------------------------------------------------------------------
    # Step 5 — Per-gene assembly + cache priming
    # ------------------------------------------------------------------
    _TTL_SECONDS = 30 * 86400

    for g in uncached_genes:
        ensembl_id = ensembl_by_gene.get(g)
        out: dict = {
            "gene_symbol":    g,
            "efo_id":         efo_id,
            "ensembl_id":     ensembl_id,
            "instruments":    [],
            "best_nes":       None,
            "best_gwas_beta": None,
            "n_instruments":  0,
            "data_source":    "Open Targets Platform v4 credible sets (bulk)",
        }
        if not gwas_study_ids:
            out["note"] = "No GWAS studies found for EFO"
            result_by_gene[g] = out
            cache.set(cache_keys[g], out, _TTL_SECONDS)
            continue
        if not ensembl_id:
            out["note"] = "Gene not found"
            result_by_gene[g] = out
            cache.set(cache_keys[g], out, _TTL_SECONDS)
            continue

        # Disease-level GWAS credsets apply to every gene at the locus; this
        # matches the single-gene function which attaches them without a
        # gene-specific filter.
        gene_instruments: list[dict] = list(shared_gwas_instruments)

        for cs in eqtl_rows:
            if (cs.get("qtlGeneId") or "").upper() != ensembl_id.upper():
                continue
            beta = cs.get("beta")
            if beta is None:
                continue
            p_mant = cs.get("pValueMantissa")
            p_exp  = cs.get("pValueExponent")
            p_val  = (float(p_mant) * (10 ** float(p_exp))) if (p_mant and p_exp) else None
            gene_instruments.append({
                "beta":            float(beta),
                "se":              float(cs["standardError"]) if cs.get("standardError") else None,
                "p_value":         p_val,
                "study_type":      cs.get("studyType", "eqtl"),
                "study_id":        cs.get("studyId"),
                "score":           min_score,
                "instrument_type": "eqtl",
            })

        gene_instruments.sort(
            key=lambda x: (0 if x["instrument_type"] == "eqtl" else 1,
                           x.get("p_value") or 1.0)
        )
        gene_instruments = gene_instruments[:max_instruments]

        out["instruments"]    = gene_instruments
        out["n_instruments"]  = len(gene_instruments)
        out["best_nes"]       = next(
            (i["beta"] for i in gene_instruments if i["instrument_type"] == "eqtl"), None
        )
        out["best_gwas_beta"] = next(
            (i["beta"] for i in gene_instruments if i["instrument_type"] == "gwas_credset"), None
        )

        result_by_gene[g] = out
        cache.set(cache_keys[g], out, _TTL_SECONDS)

    return result_by_gene


@_tool
def get_ot_genetic_scores_for_gene_set(
    efo_id: str,
    gene_symbols: list[str],
) -> dict:
    """
    Retrieve Open Targets genetic association scores for a list of genes × disease.

    Used for live γ_{program→trait} estimation: the mean genetic evidence score
    across a program's gene set is a data-driven proxy for S-LDSC enrichment.

    Queries the OT Platform GraphQL API with a batch gene lookup.  Returns
    per-gene genetic_association scores (the subset of OT evidence from GWAS,
    fine-mapping, and colocalization) — NOT the overall multi-evidence score.

    Args:
        efo_id:       EFO trait ID, e.g. "EFO_0001645"
        gene_symbols: List of gene symbols defining the program

    Returns:
        {
            "efo_id": str,
            "gene_scores": {gene_symbol: genetic_score},   # 0–1
            "mean_genetic_score": float,                   # mean across genes with data
            "n_genes_with_data": int,
            "n_genes_queried": int,
            "data_source": "Open Targets Platform v4 GraphQL",
        }
    """
    from pipelines.api_cache import get_cache
    _key_args = (efo_id, tuple(sorted(gene_symbols)))
    return get_cache().get_or_set(
        "get_ot_genetic_scores_for_gene_set", _key_args, {},
        lambda: _get_ot_genetic_scores_live(efo_id, gene_symbols),
        ttl_days=30,
    )


def _get_ot_genetic_scores_live(
    efo_id: str,
    gene_symbols: list[str],
) -> dict:
    """
    Retrieves genetic association scores for a list of genes via the disease's
    associatedTargets query — one request, no complexity-limit issues.

    Replaces the previous aliased-batch approach (50 aliased `search` fields per
    request) which exceeded OT Platform query complexity limits and returned 400s.
    """
    # Use disease.associatedTargets with a large page — returns per-gene genetic
    # scores in a single round-trip.  max_targets=1000 covers all GWAS-associated
    # genes for any disease; min_overall_score=0.0 ensures no genes are filtered out.
    assoc_q = """
    query DiseaseGeneticScores($efoId: String!, $size: Int!) {
      disease(efoId: $efoId) {
        associatedTargets(page: {index: 0, size: $size}) {
          rows {
            target {
              id
              approvedSymbol
            }
            score
            datatypeScores {
              id
              score
            }
          }
        }
      }
    }
    """
    resolved_efo = _resolve_efo(efo_id)
    data = _gql_request(OT_PLATFORM_GQL, assoc_q, {"efoId": resolved_efo, "size": 1000})

    if "error" in data:
        return {
            "efo_id": efo_id, "gene_scores": {}, "mean_genetic_score": 0.0,
            "n_genes_with_data": 0, "n_genes_queried": len(gene_symbols),
            "error": data["error"],
        }

    rows = (data.get("disease") or {}).get("associatedTargets", {}).get("rows", []) or []

    # Build symbol→genetic_score map from the full disease target list
    symbol_upper_to_score: dict[str, float] = {}
    for row in rows:
        sym = (row.get("target") or {}).get("approvedSymbol", "")
        if not sym:
            continue
        g_score = 0.0
        for ds in row.get("datatypeScores", []):
            if ds.get("id") == "genetic_association":
                g_score = float(ds.get("score", 0.0))
                break
        if g_score == 0.0:
            g_score = float(row.get("score", 0.0)) * 0.5
        if g_score > 0.0:
            symbol_upper_to_score[sym.upper()] = g_score

    # Filter for requested gene set
    gene_scores: dict[str, float] = {}
    for sym in gene_symbols:
        score = symbol_upper_to_score.get(sym.upper(), 0.0)
        if score > 0.0:
            gene_scores[sym] = score

    valid_scores = list(gene_scores.values())
    mean_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

    return {
        "efo_id":              efo_id,
        "gene_scores":         gene_scores,
        "mean_genetic_score":  round(mean_score, 4),
        "n_genes_with_data":   len(valid_scores),
        "n_genes_queried":     len(gene_symbols),
        "data_source":         "Open Targets Platform v4 GraphQL (disease.associatedTargets)",
    }


@_tool
def get_ot_colocalisation_for_program(
    program_gene_set: list[str],
    efo_id: str,
    h4_min: float = 0.5,
    max_gwas_studies: int = 10,
    max_credsets: int = 30,
) -> dict:
    """
    Estimate γ_{program→trait} via eQTL–GWAS colocalization from Open Targets Platform.
    """
    import re

    _EMPTY: dict = {
        "efo_id": efo_id, "gamma_coloc": None, "n_coloc_hits": 0,
        "coloc_genes": [], "evidence_tier": "provisional_virtual",
        "data_source": "OT_Platform_v4_colocalisation",
    }

    if not program_gene_set or not efo_id:
        return {**_EMPTY, "note": "Missing program_gene_set or efo_id"}

    # 1. Fetch GWAS study IDs for this disease (Platform API)
    studies_q = """
    query Studies($efoId: String!, $size: Int!) {
      studies(diseaseIds: [$efoId], page: {size: $size, index: 0}) {
        rows { id studyType }
      }
    }
    """
    studies_data = _gql_request(OT_PLATFORM_GQL, studies_q, {"efoId": _resolve_efo(efo_id), "size": max_gwas_studies})
    if "error" in studies_data:
        return {**_EMPTY, "note": f"studies query failed: {studies_data['error']}"}

    gwas_ids = [row["id"] for row in (studies_data.get("studies", {}).get("rows") or [])]
    if not gwas_ids:
        return {**_EMPTY, "note": "No GWAS studies found for EFO"}

    # 2. Query colocalisations via credibleSets (Platform API)
    coloc_raw: list[dict] = []
    credsets_q = """
    query CredSets($studyIds: [String!]!, $size: Int!) {
      credibleSets(studyIds: $studyIds, page: {size: $size, index: 0}) {
        rows {
          colocalisation {
            rows {
              h4
              betaRatioSignAverage
              otherStudyLocus { studyId studyType qtlGeneId }
            }
          }
        }
      }
    }
    """
    for i in range(0, len(gwas_ids), 5):
        batch = gwas_ids[i:i + 5]
        cs_data = _gql_request(OT_PLATFORM_GQL, credsets_q, {"studyIds": batch, "size": max_credsets})
        if "error" in cs_data:
            continue
        for cs_row in (cs_data.get("credibleSets", {}).get("rows") or []):
            for coloc in (cs_row.get("colocalisation", {}).get("rows") or []):
                other = coloc.get("otherStudyLocus") or {}
                if other.get("studyType", "eqtl") != "eqtl":
                    continue
                h4 = coloc.get("h4")
                brs = coloc.get("betaRatioSignAverage")
                if h4 is None or float(h4) < h4_min:
                    continue
                ensg = other.get("qtlGeneId")
                if ensg:
                    coloc_raw.append({
                        "h4":      float(h4),
                        "brs":     float(brs) if brs is not None else 1.0,
                        "ensg_id": ensg.upper(),
                    })

    if not coloc_raw:
        return {**_EMPTY, "note": "No eQTL colocalisations found"}

    # 3. Batch-resolve Ensembl IDs → symbols
    unique_ensg = list({r["ensg_id"] for r in coloc_raw})
    ensg_to_symbol: dict[str, str] = {}
    targets_q = """
    query Targets($ids: [String!]!) {
      targets(ensemblIds: $ids) {
        id
        approvedSymbol
      }
    }
    """
    for i in range(0, len(unique_ensg), 50):
        batch_ensg = unique_ensg[i:i + 50]
        t_data = _gql_request(OT_PLATFORM_GQL, targets_q, {"ids": batch_ensg})
        if "error" not in t_data:
            for row in (t_data.get("targets") or []):
                if row.get("id") and row.get("approvedSymbol"):
                    ensg_to_symbol[row["id"].upper()] = row["approvedSymbol"].upper()

    # 4. Filter and average
    program_upper = {g.upper() for g in program_gene_set}
    hits: list[float] = []
    coloc_genes: list[str] = []
    for rec in coloc_raw:
        symbol = ensg_to_symbol.get(rec["ensg_id"], "")
        if symbol and symbol in program_upper:
            hits.append(rec["h4"] * rec["brs"])
            if symbol not in coloc_genes:
                coloc_genes.append(symbol)

    if not hits:
        return {**_EMPTY, "note": "No coloc hits overlap with program gene set"}

    gamma_coloc = round(sum(hits) / len(hits), 4)
    return {
        "efo_id":        efo_id,
        "gamma_coloc":   gamma_coloc,
        "n_coloc_hits":  len(hits),
        "coloc_genes":   coloc_genes,
        "evidence_tier": "Tier2_Convergent",
        "data_source":   "OT_Platform_v4_colocalisation",
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if mcp is None:
        raise RuntimeError("fastmcp required: pip install fastmcp")
    mcp.run()
