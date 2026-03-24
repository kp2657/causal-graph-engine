"""
clinical_trials_server.py — MCP server for ClinicalTrials.gov queries.

Real implementations:
  - ClinicalTrials.gov REST API v2 (free, no auth)
    Base: https://clinicaltrials.gov/api/v2/

Run standalone:  python mcp_servers/clinical_trials_server.py
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
    mcp = fastmcp.FastMCP("clinical-trials-server")
    _tool = mcp.tool()
except ImportError:
    def _tool(fn=None, **_):
        return fn if fn is not None else (lambda f: f)
    mcp = None

CT_API = "https://clinicaltrials.gov/api/v2"


def _ct_params(extra: dict | None = None) -> dict:
    params = {"format": "json"}
    if extra:
        params.update(extra)
    return params


@_tool
def search_clinical_trials(
    condition: str | None = None,
    intervention: str | None = None,
    status: str | None = "RECRUITING",
    phase: str | None = None,
    max_results: int = 10,
) -> dict:
    """
    Search ClinicalTrials.gov for trials.

    Args:
        condition:    Disease/condition, e.g. "coronary artery disease"
        intervention: Drug/intervention, e.g. "PCSK9 inhibitor"
        status:       Trial status: "RECRUITING" | "COMPLETED" | "ACTIVE_NOT_RECRUITING" | None (all)
        phase:        Phase filter: "PHASE1" | "PHASE2" | "PHASE3" | "PHASE4" | None
        max_results:  Max trials to return

    Returns:
        {"condition": str, "n_trials": int, "trials": list[dict]}
    """
    params = _ct_params({
        "pageSize": min(max_results, 100),
    })
    query_parts = []
    if condition:
        query_parts.append(f"AREA[ConditionSearch]{condition}")
    if intervention:
        query_parts.append(f"AREA[InterventionSearch]{intervention}")
    if query_parts:
        params["query.term"] = " AND ".join(query_parts)

    if status:
        params["filter.overallStatus"] = status
    if phase:
        params["filter.phase"] = phase

    try:
        resp = httpx.get(f"{CT_API}/studies", params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        studies_raw = data.get("studies", [])
        trials = []
        for study in studies_raw:
            proto = study.get("protocolSection", {})
            ident = proto.get("identificationModule", {})
            status_mod = proto.get("statusModule", {})
            design = proto.get("designModule", {})
            conds = proto.get("conditionsModule", {}).get("conditions", [])
            interventions = proto.get("armsInterventionsModule", {}).get("interventions", [])
            inames = [i.get("name", "") for i in interventions[:3]]
            trials.append({
                "nct_id":     ident.get("nctId"),
                "title":      ident.get("briefTitle", ""),
                "status":     status_mod.get("overallStatus"),
                "phase":      design.get("phases", []),
                "conditions": conds[:3],
                "interventions": inames,
                "start_date": status_mod.get("startDateStruct", {}).get("date"),
                "n_enrolled": design.get("enrollmentInfo", {}).get("count"),
            })

        return {
            "condition":  condition,
            "intervention": intervention,
            "status_filter": status,
            "n_trials":   len(trials),
            "total_count": data.get("totalCount"),
            "trials":     trials,
            "source":     "ClinicalTrials.gov API v2",
        }
    except httpx.HTTPStatusError as e:
        return {"condition": condition, "error": f"HTTP {e.response.status_code}", "trials": []}
    except Exception as e:
        return {"condition": condition, "error": str(e), "trials": []}


@_tool
def get_trial_details(nct_id: str) -> dict:
    """
    Fetch detailed information for a specific clinical trial.

    Args:
        nct_id: ClinicalTrials.gov NCT ID, e.g. "NCT01662869"
    """
    try:
        resp = httpx.get(
            f"{CT_API}/studies/{nct_id}",
            params=_ct_params(),
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        proto = data.get("protocolSection", {})
        ident = proto.get("identificationModule", {})
        status_mod = proto.get("statusModule", {})
        design = proto.get("designModule", {})
        outcome = proto.get("outcomesModule", {})
        eligib = proto.get("eligibilityModule", {})

        primary_outcomes = [
            o.get("measure", "") for o in outcome.get("primaryOutcomes", [])[:3]
        ]

        return {
            "nct_id":           nct_id,
            "title":            ident.get("briefTitle"),
            "official_title":   ident.get("officialTitle"),
            "status":           status_mod.get("overallStatus"),
            "phase":            design.get("phases", []),
            "n_enrolled":       design.get("enrollmentInfo", {}).get("count"),
            "primary_outcomes": primary_outcomes,
            "min_age":          eligib.get("minimumAge"),
            "max_age":          eligib.get("maximumAge"),
            "sex":              eligib.get("sex"),
            "start_date":       status_mod.get("startDateStruct", {}).get("date"),
            "completion_date":  status_mod.get("primaryCompletionDateStruct", {}).get("date"),
        }
    except httpx.HTTPStatusError as e:
        return {"nct_id": nct_id, "error": f"HTTP {e.response.status_code}"}
    except Exception as e:
        return {"nct_id": nct_id, "error": str(e)}


@_tool
def get_trials_for_target(gene_symbol: str, disease: str | None = None) -> dict:
    """
    Find clinical trials targeting a specific gene product.
    Useful for target validation in the Ota framework.

    Args:
        gene_symbol: Gene symbol, e.g. "PCSK9", "IL6R"
        disease:     Optional disease filter
    """
    # Map gene symbols to drug interventions used in trials
    GENE_DRUG_MAP = {
        "PCSK9":  ["evolocumab", "alirocumab", "inclisiran", "PCSK9 inhibitor"],
        "HMGCR":  ["statin", "atorvastatin", "rosuvastatin", "simvastatin"],
        "IL6R":   ["tocilizumab", "sarilumab", "IL-6"],
        "PCSK9":  ["evolocumab", "alirocumab"],
        "LDLR":   ["mipomersen", "lomitapide"],
        "DNMT3A": ["azacitidine", "decitabine"],  # CHIP-targeting demethylation agents
        "TET2":   ["azacitidine", "vitamin C"],   # TET2 restoration strategies
    }
    drugs = GENE_DRUG_MAP.get(gene_symbol.upper(), [gene_symbol])
    intervention_query = " OR ".join(drugs[:2])
    return search_clinical_trials(
        condition=disease or "cardiovascular",
        intervention=intervention_query,
        status=None,  # all statuses
        max_results=10,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if mcp is None:
        raise RuntimeError("fastmcp required: pip install fastmcp")
    mcp.run()
