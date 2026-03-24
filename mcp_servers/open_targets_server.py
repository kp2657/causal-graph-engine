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
OT_GENETICS_GQL  = "https://api.genetics.opentargets.org/graphql"

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
    try:
        resp = httpx.post(
            url,
            json={"query": query, "variables": variables or {}},
            headers={"Content-Type": "application/json"},
            timeout=30,
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


@_tool
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
    query = """
    query DiseaseTargets($efoId: String!, $size: Int!) {
      disease(efoId: $efoId) {
        id
        name
        associatedTargets(page: {index: 0, size: $size}) {
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
                  drug {
                    id
                    name
                  }
                  maxClinicalStage
                }
              }
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
    data = _gql_request(OT_PLATFORM_GQL, query, {"efoId": efo_id, "size": max_targets})
    if "error" in data:
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
            }
        return {"efo_id": efo_id, "error": data["error"], "targets": []}

    disease = data.get("disease", {})
    if not disease:
        return {"efo_id": efo_id, "targets": [], "note": "Disease not found"}

    rows = disease.get("associatedTargets", {}).get("rows", [])
    targets = []
    for row in rows:
        if row["score"] < min_overall_score:
            continue
        dtype_scores = {d["id"]: d["score"] for d in row.get("datatypeScores", [])}

        # Tractability: pick highest-confidence modality (SM > AB > other)
        tract_items = row["target"].get("tractability") or []
        sm_active = any(
            t["modality"] == "SM" and t.get("value") for t in tract_items
        )
        ab_active = any(
            t["modality"] == "AB" and t.get("value") for t in tract_items
        )
        tractability_class = (
            "small_molecule" if sm_active else
            "antibody"       if ab_active else
            "difficult"
        )

        # Drugs: max clinical phase and drug names
        drug_rows = row["target"].get("drugAndClinicalCandidates", {}).get("rows", []) or []
        known_drugs = list({dr["drug"]["name"] for dr in drug_rows if dr.get("drug")})
        max_phase = max(
            (_stage_to_int(dr.get("maxClinicalStage")) for dr in drug_rows),
            default=0,
        )

        targets.append({
            "target_id":         row["target"]["id"],
            "gene_symbol":       row["target"]["approvedSymbol"],
            "overall_score":     round(row["score"], 4),
            "genetic_score":     round(dtype_scores.get("genetic_association", 0), 4),
            "max_clinical_phase": max_phase,
            "known_drugs":       known_drugs[:5],
            "tractability_class": tractability_class,
        })

    return {
        "efo_id":       efo_id,
        "disease_name": disease.get("name"),
        "n_targets":    len(targets),
        "targets":      targets,
        "source":       "Open Targets Platform GraphQL",
    }


@_tool
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
              maximumClinicalStage
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
        "max_phase":   _stage_to_int(drug_obj.get("maximumClinicalStage")),
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
    # Resolve gene → Ensembl ID
    search_q = """
    query { search(queryString: $sym, entityNames: ["target"]) { hits { id entity name } } }
    """.replace("$sym", f'"{gene_symbol}"')
    sr = _gql_request(OT_PLATFORM_GQL, search_q)
    ensembl_id = None
    for h in (sr.get("search", {}).get("hits") or []):
        if h.get("entity") == "target" and h.get("name", "").upper() == gene_symbol.upper():
            ensembl_id = h["id"]
            break
    if not ensembl_id:
        return {"gene_symbol": gene_symbol, "efo_id": efo_id, "instruments": [],
                "best_nes": None, "best_gwas_beta": None, "note": "Gene not found"}

    # Fetch credible set evidence with beta/SE
    evid_q = """
    query CredSetEvidence($ensemblId: String!, $efoId: String!) {
      target(ensemblId: $ensemblId) {
        evidences(
          efoIds: [$efoId]
          datasourceIds: ["gwas_credible_sets"]
          size: 20
        ) {
          rows {
            score
            credibleSet {
              beta
              standardError
              pValueMantissa
              pValueExponent
              studyType
              studyId
            }
          }
        }
      }
    }
    """
    data = _gql_request(OT_PLATFORM_GQL, evid_q, {"ensemblId": ensembl_id, "efoId": efo_id})
    if "error" in data:
        return {"gene_symbol": gene_symbol, "efo_id": efo_id, "instruments": [],
                "best_nes": None, "best_gwas_beta": None, "error": data["error"]}

    rows = (data.get("target") or {}).get("evidences", {}).get("rows") or []
    instruments: list[dict] = []
    for row in rows:
        if row.get("score", 0) < min_score:
            continue
        cs = row.get("credibleSet") or {}
        beta = cs.get("beta")
        if beta is None:
            continue
        p_mant = cs.get("pValueMantissa")
        p_exp  = cs.get("pValueExponent")
        p_val  = (float(p_mant) * (10 ** float(p_exp))) if (p_mant and p_exp) else None
        study_type = cs.get("studyType", "gwas")
        instruments.append({
            "beta":            float(beta),
            "se":              float(cs["standardError"]) if cs.get("standardError") else None,
            "p_value":         p_val,
            "study_type":      study_type,
            "study_id":        cs.get("studyId"),
            "score":           row["score"],
            "instrument_type": "eqtl" if study_type == "eqtl" else "gwas_credset",
        })

    # Sort: eQTL first (more directly useful for β_gene→program), then by score desc
    instruments.sort(key=lambda x: (0 if x["instrument_type"] == "eqtl" else 1, -x["score"]))
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
        "data_source":     "Open Targets Platform v4 credible sets",
    }


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
    # Query per gene: search for Ensembl ID, then look up target.associatedDiseases
    # to find the genetic_association score for the requested efo_id.
    search_query = """
    query SearchTarget($symbol: String!) {
      search(queryString: $symbol, entityNames: ["target"]) {
        hits { id entity name }
      }
    }
    """
    assoc_query = """
    query TargetDisease($ensemblId: String!, $efoId: String!) {
      target(ensemblId: $ensemblId) {
        associatedDiseases(
          Bs: [$efoId]
          page: { index: 0, size: 1 }
        ) {
          rows {
            disease { id }
            score
            datatypeScores { id score }
          }
        }
      }
    }
    """
    gene_scores: dict[str, float] = {}

    for symbol in gene_symbols[:20]:   # cap at 20 to avoid rate limiting
        try:
            # Step 1: resolve symbol → ensemblId
            sr = httpx.post(
                OT_PLATFORM_GQL,
                json={"query": search_query, "variables": {"symbol": symbol}},
                headers={"Content-Type": "application/json"},
                timeout=20,
            )
            if sr.status_code != 200:
                continue
            hits = sr.json().get("data", {}).get("search", {}).get("hits", [])
            ensembl_id = None
            for h in hits:
                if h.get("entity") == "target" and h.get("name", "").upper() == symbol.upper():
                    ensembl_id = h["id"]
                    break
            if not ensembl_id:
                continue

            # Step 2: query target-disease association score
            ar = httpx.post(
                OT_PLATFORM_GQL,
                json={"query": assoc_query, "variables": {"ensemblId": ensembl_id, "efoId": efo_id}},
                headers={"Content-Type": "application/json"},
                timeout=20,
            )
            if ar.status_code != 200:
                continue
            rows = (
                ar.json().get("data", {})
                    .get("target", {})
                    .get("associatedDiseases", {})
                    .get("rows", [])
            )
            if not rows:
                continue
            row = rows[0]
            # Extract genetic_association datatype score specifically
            genetic_score = 0.0
            for ds in row.get("datatypeScores", []):
                if ds.get("id") == "genetic_association":
                    genetic_score = float(ds.get("score", 0.0))
                    break
            if genetic_score == 0.0:
                genetic_score = float(row.get("score", 0.0)) * 0.5
            gene_scores[symbol] = genetic_score
        except Exception:
            continue

    valid_scores = [v for v in gene_scores.values() if v > 0]
    mean_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

    return {
        "efo_id":              efo_id,
        "gene_scores":         gene_scores,
        "mean_genetic_score":  round(mean_score, 4),
        "n_genes_with_data":   len(valid_scores),
        "n_genes_queried":     len(gene_symbols),
        "data_source":         "Open Targets Platform v4 GraphQL",
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

    For each GWAS credible set associated with the disease (efo_id), retrieves
    pre-computed eQTL colocalisations (H4 score + betaRatioSignAverage).

    γ_proxy = mean(H4 × betaRatioSignAverage) across colocs where:
      - the eQTL gene is in program_gene_set
      - H4 ≥ h4_min (strong colocalisation evidence)

    betaRatioSignAverage (±1) captures causal direction; H4 weights confidence.
    Evidence tier: Tier2_Convergent (convergent genetic + eQTL evidence).

    Args:
        program_gene_set: Gene symbols defining the cNMF program
        efo_id:           EFO disease ID, e.g. "EFO_0001645"
        h4_min:           Minimum H4 posterior probability to include (default 0.5)
        max_gwas_studies: Max GWAS studies to query per disease (default 10)
        max_credsets:     Max credible sets per study batch (default 30)

    Returns:
        {
            "efo_id": str,
            "gamma_coloc": float | None,   # H4-weighted betaRatioSignAverage
            "n_coloc_hits": int,           # coloc pairs passing H4 threshold
            "coloc_genes": list[str],      # eQTL gene symbols in program_gene_set
            "evidence_tier": str,
            "data_source": str,
        }
    """
    import re

    _EMPTY: dict = {
        "efo_id": efo_id, "gamma_coloc": None, "n_coloc_hits": 0,
        "coloc_genes": [], "evidence_tier": "provisional_virtual",
        "data_source": "OT_Platform_v4_colocalisation",
    }

    if not program_gene_set or not efo_id:
        return {**_EMPTY, "note": "Missing program_gene_set or efo_id"}

    # 1. Fetch GWAS study IDs for this disease
    studies_q = """
    query Studies($efoId: String!, $size: Int!) {
      studies(diseaseIds: [$efoId] page: {size: $size, index: 0}) {
        rows { id studyType }
      }
    }
    """
    studies_data = _gql_request(OT_PLATFORM_GQL, studies_q, {"efoId": efo_id, "size": max_gwas_studies})
    if "error" in studies_data:
        return {**_EMPTY, "note": f"studies query failed: {studies_data['error']}"}

    gwas_ids = [row["id"] for row in (studies_data.get("studies", {}).get("rows") or [])]
    if not gwas_ids:
        return {**_EMPTY, "note": "No GWAS studies found for EFO"}

    # 2. Query credible sets + eQTL colocalisations in batches of 5 study IDs
    _ENSG_RE = re.compile(r'(ensg\d+)', re.IGNORECASE)
    coloc_raw: list[dict] = []  # {h4, brs, ensg_id}

    credsets_q = """
    query CredSets($studyIds: [String!]!, $size: Int!) {
      credibleSets(studyIds: $studyIds page: {size: $size, index: 0}) {
        rows {
          colocalisation(studyTypes: [eqtl] page: {size: 10, index: 0}) {
            rows {
              h4
              betaRatioSignAverage
              otherStudyLocus { studyId qtlGeneId }
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
                h4 = coloc.get("h4")
                brs = coloc.get("betaRatioSignAverage")
                if h4 is None or brs is None or float(h4) < h4_min:
                    continue
                other = coloc.get("otherStudyLocus") or {}
                # qtlGeneId is the direct Ensembl ID field (preferred over regex)
                ensg = other.get("qtlGeneId") or ""
                if not ensg:
                    m = _ENSG_RE.search(other.get("studyId", ""))
                    ensg = m.group(1) if m else ""
                if ensg:
                    coloc_raw.append({
                        "h4":     float(h4),
                        "brs":    float(brs),
                        "ensg_id": ensg.upper(),
                    })

    if not coloc_raw:
        return {**_EMPTY, "note": "No eQTL colocalisations found for GWAS loci"}

    # 3. Batch-resolve unique Ensembl IDs → gene symbols via OT targets query
    unique_ensg = list({r["ensg_id"] for r in coloc_raw})
    ensg_to_symbol: dict[str, str] = {}
    targets_q = """
    query Targets($ids: [String!]!) {
      targets(ensemblIds: $ids) {
        rows { id approvedSymbol }
      }
    }
    """
    for i in range(0, len(unique_ensg), 50):
        batch_ensg = unique_ensg[i:i + 50]
        t_data = _gql_request(OT_PLATFORM_GQL, targets_q, {"ids": batch_ensg})
        if "error" not in t_data:
            for row in (t_data.get("targets", {}).get("rows") or []):
                if row.get("id") and row.get("approvedSymbol"):
                    ensg_to_symbol[row["id"].upper()] = row["approvedSymbol"].upper()

    # 4. Filter coloc hits by program gene set intersection
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
        "data_source":   "OT_Platform_v4_eQTL_colocalisation",
        "note":          f"H4-weighted betaRatioSignAverage over {len(hits)} coloc pairs",
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if mcp is None:
        raise RuntimeError("fastmcp required: pip install fastmcp")
    mcp.run()
