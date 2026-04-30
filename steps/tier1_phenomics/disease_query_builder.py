"""
disease_query_builder.py — Tier 1: disease phenotype definition.

Builds a structured DiseaseQuery with EFO IDs, ICD-10 codes, and GWAS study IDs.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.evidence import DiseaseQuery
from models.disease_registry import get_disease_key as _get_disease_key


# EFO → ICD-10 mapping for common diseases
EFO_ICD10_MAP: dict[str, list[str]] = {
    "EFO_0001645": ["I20", "I21", "I22", "I23", "I24", "I25"],  # CAD
    "EFO_0000685": ["M05", "M06"],                               # RA
}

# Disease name → EFO ID (for common lookups)
DISEASE_EFO_MAP: dict[str, str] = {
    "coronary artery disease":            "EFO_0001645",
    "cad":                                "EFO_0001645",
    "ischemic heart disease":             "EFO_0001645",
    "rheumatoid arthritis":               "EFO_0000685",
    "ra":                                 "EFO_0000685",
    "seropositive rheumatoid arthritis":  "EFO_0000685",
}

# Disease → modifier types (which evidence types are relevant)
DISEASE_MODIFIER_MAP: dict[str, list[str]] = {
    "EFO_0001645": ["germline", "somatic_chip", "drug"],  # CAD: CHIP + germline strong
    "EFO_0000685": ["germline", "drug"],                  # RA
}


def run(disease_name: str) -> dict:
    """
    Build a structured DiseaseQuery for a disease.

    Args:
        disease_name: Human-readable disease name, e.g. "coronary artery disease"

    Returns:
        DiseaseQuery dict with EFO ID, ICD-10, modifier types, GWAS info
    """
    from mcp_servers.gwas_genetics_server import (
        get_gwas_catalog_studies,
        list_available_gwas,
    )

    # Resolve EFO ID
    efo_id = DISEASE_EFO_MAP.get(disease_name.lower())
    if efo_id is None:
        # Try live GWAS catalog lookup
        studies = get_gwas_catalog_studies(disease_name.replace(" ", "+"), page_size=1)
        efo_id = studies.get("efo_id")

    icd10 = EFO_ICD10_MAP.get(efo_id, []) if efo_id else []
    modifiers = DISEASE_MODIFIER_MAP.get(efo_id, ["germline"]) if efo_id else ["germline"]

    # Known primary OpenGWAS study IDs (fallback when live lookup fails)
    _KNOWN_GWAS_IDS: dict[str, str] = {
        "EFO_0001645": "ieu-a-7",              # CAD — CARDIoGRAMplusC4D
        "EFO_0000685": "ebi-a-GCST002318",     # RA — Okada 2014 ~100k samples
    }

    # Get GWAS catalog study count
    n_gwas = 0
    primary_gwas_id = _KNOWN_GWAS_IDS.get(efo_id) if efo_id else None
    if efo_id:
        try:
            studies_result = get_gwas_catalog_studies(efo_id, page_size=1)
            n_gwas = studies_result.get("total_studies", 0)
        except Exception:
            n_gwas = 0  # GWAS Catalog may not index all EFOs; efo_id is still valid

        # Get IEU Open GWAS study IDs (override known fallback if live lookup succeeds)
        try:
            gwas_list = list_available_gwas(disease_name)
            datasets = gwas_list.get("datasets", [])
            if datasets:
                primary_gwas_id = datasets[0].get("id")
        except Exception:
            pass  # keep the known fallback

    return {
        "disease_name":      disease_name,
        "disease_key":       _get_disease_key(disease_name),
        "efo_id":            efo_id,
        "icd10_codes":       icd10,
        "modifier_types":    modifiers,
        "primary_gwas_id":   primary_gwas_id,
        "n_gwas_studies":    n_gwas,
        "use_precomputed_only": True,
        "day_one_mode":      True,
    }
