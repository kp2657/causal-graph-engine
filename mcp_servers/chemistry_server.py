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

CHEMBL_API  = "https://www.ebi.ac.uk/chembl/api/data"
PUBCHEM_API = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"


# ---------------------------------------------------------------------------
# ChEMBL tools (live)
# ---------------------------------------------------------------------------

@_tool
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
            timeout=30,
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
            timeout=30,
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
            timeout=30,
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

        resp = httpx.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        compounds = data.get("PC_Compounds", [])
        if not compounds:
            return {"query": name_or_cid, "note": "Not found in PubChem"}

        cmpd = compounds[0]
        # Extract properties
        props = {}
        for prop in cmpd.get("props", []):
            urn = prop.get("urn", {})
            value = prop.get("value", {})
            label = urn.get("label", "")
            if label in ("MolecularFormula", "InChIKey", "SMILES", "MolecularWeight", "IUPACName"):
                val = value.get("sval") or value.get("fval") or value.get("ival")
                props[label] = val

        return {
            "query":   name_or_cid,
            "cid":     cmpd.get("id", {}).get("id", {}).get("cid"),
            "formula": props.get("MolecularFormula"),
            "mw":      props.get("MolecularWeight"),
            "smiles":  props.get("SMILES"),
            "inchikey": props.get("InChIKey"),
            "iupac_name": props.get("IUPACName"),
            "source":  "PubChem REST API",
        }
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return {"query": name_or_cid, "note": "Compound not found in PubChem"}
        return {"query": name_or_cid, "error": f"HTTP {e.response.status_code}"}
    except Exception as e:
        return {"query": name_or_cid, "error": str(e)}


# ---------------------------------------------------------------------------
# Stub tools — RDKit / TxGemma / ADMET-AI
# ---------------------------------------------------------------------------

@_tool
def compute_rdkit_properties(smiles: str) -> dict:
    """
    Compute molecular properties from SMILES using RDKit.

    STUB — requires: pip install rdkit

    Properties computed when implemented:
      - Molecular weight, LogP, TPSA
      - Lipinski Ro5 compliance
      - Number of HBA, HBD, rotatable bonds
      - QED (drug-likeness score)
    """
    return {
        "smiles":     smiles,
        "mw":         None,
        "logP":       None,
        "tpsa":       None,
        "hba":        None,
        "hbd":        None,
        "ro5_pass":   None,
        "qed":        None,
        "note":       "STUB — install RDKit: pip install rdkit",
    }


@_tool
def run_txgemma_prediction(smiles: str, task: str = "efficacy") -> dict:
    """
    Run TxGemma drug property prediction.

    STUB — requires:
      1. TxGemma model weights (Google DeepMind; download via Kaggle)
      2. Hugging Face transformers + torch

    Args:
        smiles: SMILES string for the compound
        task:   "efficacy" | "toxicity" | "admet"
    """
    return {
        "smiles":      smiles,
        "task":        task,
        "prediction":  None,
        "confidence":  None,
        "note":        "STUB — TxGemma requires local model; download from Kaggle",
        "paper":       "TxGemma: Google DeepMind drug discovery LLM",
    }


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
