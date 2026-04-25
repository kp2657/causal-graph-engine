"""
chemistry_server.py — MCP server for drug chemistry and ADMET queries.

Real implementations:
  - ChEMBL REST API (free, no auth)
  - PubChem REST API (free, no auth)

Stubs:
  - RDKit: molecular property calculation (requires rdkit package)
  - TxGemma: drug efficacy/toxicity prediction (requires local model)
  - ADMET-AI: ADMET property prediction (requires API key)

Run standalone:  python mcp_servers/chemistry_server.py
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import httpx

try:
    import fastmcp
    mcp = fastmcp.FastMCP("chemistry-server")
    _tool = mcp.tool()
except ImportError:
    def _tool(fn=None, **_):
        return fn if fn is not None else (lambda f: f)
    mcp = None

from pipelines.api_cache import api_cached

CHEMBL_API  = "https://www.ebi.ac.uk/chembl/api/data"
PUBCHEM_API = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

# GPS / ChEMBL annotation strings → HGNC (process-local cache)
_CHEMBL_LABEL_TO_GENE: dict[str, str | None] = {}
_HGNC_TOKEN = re.compile(r"^[A-Z][A-Z0-9-]{1,14}$")


def _gene_symbol_from_chembl_target(target: dict) -> str | None:
    """Best-effort HGNC-like symbol from ChEMBL target search row (human proteins)."""
    for comp in target.get("target_components") or []:
        for syn in comp.get("target_component_synonyms") or []:
            if syn.get("syn_type") != "GENE_SYMBOL":
                continue
            raw = (syn.get("component_synonym") or "").strip()
            if not raw:
                continue
            # ChEMBL stores mouse symbols as Egfr — uppercase for downstream HGNC use
            if not raw.replace("-", "").isalnum():
                continue
            return raw.upper()
    return None


@_tool
@api_cached(ttl_days=90)
def resolve_chembl_target_label_to_hgnc(label: str) -> dict:
    """
    Map a ChEMBL ``target_pref_name`` (or short free-text label) to a gene symbol.

    Uses ChEMBL ``target/search`` and reads ``target_components`` →
    ``GENE_SYMBOL`` synonyms for **Homo sapiens** **SINGLE PROTEIN** targets.

    If ``label`` already looks like an HGNC token (e.g. ``PCSK9``), it is returned
    without calling the API.

    Args:
        label: String from GPS ``putative_targets`` / similarity hits.

    Returns:
        ``{"query", "gene_symbol", "target_chembl_id", "source"}`` — gene_symbol may be null.
    """
    raw = (label or "").strip()
    if not raw:
        return {"query": label, "gene_symbol": None, "target_chembl_id": None, "source": "empty"}

    key = raw.casefold()
    if key in _CHEMBL_LABEL_TO_GENE:
        g = _CHEMBL_LABEL_TO_GENE[key]
        return {
            "query":             raw,
            "gene_symbol":       g,
            "target_chembl_id":  None,
            "source":            "cache",
        }

    u = raw.upper()
    if _HGNC_TOKEN.match(u) and raw == u:
        _CHEMBL_LABEL_TO_GENE[key] = u
        return {
            "query":             raw,
            "gene_symbol":       u,
            "target_chembl_id":  None,
            "source":            "hgnc_like_token",
        }

    try:
        q = raw[:120]
        resp = httpx.get(
            f"{CHEMBL_API}/target/search",
            params={"q": q, "format": "json", "limit": 12},
            timeout=httpx.Timeout(connect=5.0, read=20.0, write=5.0, pool=5.0),
        )
        resp.raise_for_status()
        rows = resp.json().get("targets", []) or []
    except Exception as e:
        _CHEMBL_LABEL_TO_GENE[key] = None
        return {
            "query":             raw,
            "gene_symbol":       None,
            "target_chembl_id":  None,
            "source":            f"error:{e}",
        }

    def _score_row(t: dict) -> int:
        org = (t.get("organism") or "").lower()
        tt = (t.get("target_type") or "").upper()
        s = 0
        if org == "homo sapiens":
            s += 100
        if "SINGLE PROTEIN" in tt or tt == "SINGLE PROTEIN":
            s += 50
        if float(t.get("score") or 0) > 0:
            s += int(min(float(t.get("score") or 0), 30))
        return s

    human_rows = [t for t in rows if (t.get("organism") or "") == "Homo sapiens"]
    pool = human_rows if human_rows else rows
    pool.sort(key=_score_row, reverse=True)

    for t in pool:
        sym = _gene_symbol_from_chembl_target(t)
        if sym:
            tid = t.get("target_chembl_id")
            _CHEMBL_LABEL_TO_GENE[key] = sym
            return {
                "query":             raw,
                "gene_symbol":       sym,
                "target_chembl_id":  tid,
                "source":            "ChEMBL target/search",
            }

    _CHEMBL_LABEL_TO_GENE[key] = None
    return {
        "query":             raw,
        "gene_symbol":       None,
        "target_chembl_id":  None,
        "source":            "ChEMBL target/search (no human gene symbol)",
    }


def resolve_gps_putative_target_labels_to_hgnc(
    labels: list[str],
    *,
    max_labels: int = 150,
) -> dict:
    """
    Batch-resolve GPS ``putative_targets`` strings to HGNC-style symbols.

    Returns unique sorted genes plus per-label resolution metadata (capped).
    """
    seen: set[str] = set()
    genes: list[str] = []
    mapping: list[dict[str, Any]] = []
    unresolved: list[str] = []

    for lab in (labels or [])[: max(1, int(max_labels))]:
        if not isinstance(lab, str) or not lab.strip():
            continue
        rec = resolve_chembl_target_label_to_hgnc(lab)
        sym = rec.get("gene_symbol")
        mapping.append({
            "raw":               lab.strip(),
            "gene_symbol":      sym,
            "target_chembl_id": rec.get("target_chembl_id"),
            "source":           rec.get("source"),
        })
        if sym:
            if sym not in seen:
                seen.add(sym)
                genes.append(sym)
        else:
            unresolved.append(lab.strip())

    genes.sort()
    return {
        "genes":             genes,
        "n_resolved":        len(seen),
        "n_unresolved":      len(unresolved),
        "mapping_sample":    mapping[:80],
        "unresolved_sample": unresolved[:40],
        "data_source":       "ChEMBL target/search + local cache",
    }


# ---------------------------------------------------------------------------
# ChEMBL tools (live)
# ---------------------------------------------------------------------------

@_tool
@api_cached(ttl_days=30)
def search_chembl_compound(name: str) -> dict:
    """
    Search ChEMBL for a compound by name.

    Args:
        name: Drug or compound name, e.g. "atorvastatin", "evolocumab"
    """
    try:
        resp = httpx.get(
            f"{CHEMBL_API}/molecule/search",
            params={
                "q":      name,
                "format": "json",
                "limit":  5,
            },
            timeout=httpx.Timeout(connect=5.0, read=20.0, write=5.0, pool=5.0),
        )
        resp.raise_for_status()
        data = resp.json()
        mols = data.get("molecules", [])
        results = []
        for mol in mols:
            props = mol.get("molecule_properties") or {}
            results.append({
                "chembl_id":      mol.get("molecule_chembl_id"),
                "name":           mol.get("pref_name"),
                "max_phase":      mol.get("max_phase"),
                "molecular_type": mol.get("molecule_type"),
                "mw":             props.get("mw_freebase"),
                "alogp":          props.get("alogp"),
                "hba":            props.get("hba"),
                "hbd":            props.get("hbd"),
                "psa":            props.get("psa"),
                "ro5_violations": props.get("num_ro5_violations"),
            })
        return {
            "query":     name,
            "n_results": len(results),
            "compounds": results,
            "source":    "ChEMBL REST API",
        }
    except Exception as e:
        return {"query": name, "error": str(e), "compounds": []}


@_tool
@api_cached(ttl_days=30)
def get_chembl_target_activities(target_gene: str, max_results: int = 20) -> dict:
    """
    Return bioactivity data for compounds targeting a gene from ChEMBL.

    Args:
        target_gene: Gene symbol, e.g. "HMGCR", "PCSK9"
        max_results: Maximum activities to return
    """
    try:
        # First find target ChEMBL ID
        target_resp = httpx.get(
            f"{CHEMBL_API}/target/search",
            params={"q": target_gene, "format": "json", "limit": 1},
            timeout=httpx.Timeout(connect=5.0, read=20.0, write=5.0, pool=5.0),
        )
        target_resp.raise_for_status()
        targets = target_resp.json().get("targets", [])
        if not targets:
            return {"target_gene": target_gene, "error": "Target not found", "activities": []}

        target_id = targets[0]["target_chembl_id"]

        # Fetch bioactivities
        act_resp = httpx.get(
            f"{CHEMBL_API}/activity",
            params={
                "target_chembl_id": target_id,
                "format":           "json",
                "limit":            max_results,
                "standard_type__in": "IC50,Ki,Kd,EC50",
            },
            timeout=httpx.Timeout(connect=5.0, read=20.0, write=5.0, pool=5.0),
        )
        act_resp.raise_for_status()
        activities_raw = act_resp.json().get("activities", [])
        activities = []
        for act in activities_raw:
            activities.append({
                "molecule_id":     act.get("molecule_chembl_id"),
                "molecule_name":   act.get("molecule_pref_name"),
                "activity_type":   act.get("standard_type"),
                "activity_value":  act.get("standard_value"),
                "activity_units":  act.get("standard_units"),
                "assay_type":      act.get("assay_type"),
            })

        return {
            "target_gene":  target_gene,
            "target_id":    target_id,
            "n_activities": len(activities),
            "activities":   activities,
            "source":       "ChEMBL REST API",
        }
    except Exception as e:
        return {"target_gene": target_gene, "error": str(e), "activities": []}


@_tool
def get_pubchem_compound(name_or_cid: str) -> dict:
    """
    Fetch compound information from PubChem.

    Args:
        name_or_cid: Compound name or CID, e.g. "atorvastatin" or "60823"
    """
    try:
        # Try by name first
        if name_or_cid.isdigit():
            url = f"{PUBCHEM_API}/compound/cid/{name_or_cid}/JSON"
        else:
            url = f"{PUBCHEM_API}/compound/name/{name_or_cid}/JSON"

        resp = httpx.get(url, timeout=httpx.Timeout(connect=5.0, read=20.0, write=5.0, pool=5.0))
        resp.raise_for_status()
        data = resp.json()
        compounds = data.get("PC_Compounds", [])
        if not compounds:
            return {"query": name_or_cid, "note": "Not found in PubChem"}

        cmpd = compounds[0]
        # Extract properties — PubChem raw JSON uses "Molecular Formula" (spaced),
        # "SMILES", "InChIKey", "Molecular Weight", "IUPAC Name" as urn.label values.
        # Canonical key → urn.label mapping:
        _label_map = {
            "Molecular Formula": "formula",
            "Molecular Weight":  "mw",
            "SMILES":            "smiles",
            "InChIKey":          "inchikey",
            "IUPAC Name":        "iupac_name",
        }
        props: dict[str, Any] = {}
        for prop in cmpd.get("props", []):
            urn = prop.get("urn", {})
            value = prop.get("value", {})
            label = urn.get("label", "")
            key = _label_map.get(label)
            if key and key not in props:
                props[key] = value.get("sval") or value.get("fval") or value.get("ival")

        return {
            "query":      name_or_cid,
            "cid":        cmpd.get("id", {}).get("id", {}).get("cid"),
            "formula":    props.get("formula"),
            "mw":         props.get("mw"),
            "smiles":     props.get("smiles"),
            "inchikey":   props.get("inchikey"),
            "iupac_name": props.get("iupac_name"),
            "source":     "PubChem REST API",
        }
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return {"query": name_or_cid, "note": "Compound not found in PubChem"}
        return {"query": name_or_cid, "error": f"HTTP {e.response.status_code}"}
    except Exception as e:
        return {"query": name_or_cid, "error": str(e)}


# ---------------------------------------------------------------------------
# ADMET prediction (stub — install admet-ai for real predictions)
# ---------------------------------------------------------------------------

@_tool
def run_admet_prediction(smiles_list: list[str]) -> dict:
    """
    Predict ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity) properties.

    STUB — options for implementation:
      1. ADMET-AI (free): pip install admet-ai
      2. pkCSM API (free tier): https://biosig.lab.uq.edu.au/pkcsm/
      3. SwissADME (web): http://www.swissadme.ch/

    When implemented with ADMET-AI:
      from admet_ai import ADMETModel
      model = ADMETModel()
      predictions = model.predict(smiles=smiles_list)
    """
    return {
        "smiles_list":  smiles_list,
        "predictions":  None,
        "properties":   ["hia", "bbb_penetration", "cyp2d6_substrate", "herg_blockers",
                         "oral_bioavailability", "half_life"],
        "note":         "STUB — install admet-ai: pip install admet-ai",
        "admet_ai_url": "https://github.com/swansonk14/admet_ai",
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if mcp is None:
        raise RuntimeError("fastmcp required: pip install fastmcp")
    mcp.run()
