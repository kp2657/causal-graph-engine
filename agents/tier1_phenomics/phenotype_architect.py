"""
phenotype_architect.py — Tier 1 agent: disease phenotype definition.

Builds a structured DiseaseQuery with EFO IDs, ICD-10 codes, and GWAS study IDs.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.evidence import DiseaseQuery
from mcp_servers.finngen_server import (
    get_finngen_phenotype_info,
    efo_to_finngen_phenocode,
)


# EFO → ICD-10 mapping for common diseases
EFO_ICD10_MAP: dict[str, list[str]] = {
    "EFO_0001645": ["I20", "I21", "I22", "I23", "I24", "I25"],  # CAD
    "EFO_0003144": ["I50"],                                        # Heart failure
    "EFO_0000275": ["I48"],                                        # Atrial fibrillation
    "EFO_0000685": ["M05", "M06"],                                 # RA
    "EFO_0002690": ["M32"],                                        # SLE
    "EFO_0003885": ["G35"],                                        # MS
    "EFO_0000341": ["J43", "J44"],                                 # COPD
    "EFO_0001359": ["E10"],                                        # T1D
    "EFO_0003767": ["K50", "K51", "K52"],                          # IBD (Crohn's + UC + other)
    "EFO_0001481": ["H35.3"],                                       # AMD (age-related macular degeneration)
}

# Disease name → EFO ID (for common lookups)
DISEASE_EFO_MAP: dict[str, str] = {
    "coronary artery disease": "EFO_0001645",
    "cad": "EFO_0001645",
    "heart failure": "EFO_0003144",
    "atrial fibrillation": "EFO_0000275",
    "rheumatoid arthritis": "EFO_0000685",
    "ra": "EFO_0000685",
    "systemic lupus erythematosus": "EFO_0002690",
    "sle": "EFO_0002690",
    "multiple sclerosis": "EFO_0003885",
    "ms": "EFO_0003885",
    "copd": "EFO_0000341",
    "type 1 diabetes": "EFO_0001359",
    "t1d": "EFO_0001359",
    "inflammatory bowel disease": "EFO_0003767",
    "ibd": "EFO_0003767",
    "crohn's disease": "EFO_0000384",
    "crohns disease": "EFO_0000384",
    "ulcerative colitis": "EFO_0000729",
    "uc": "EFO_0000729",
    "age-related macular degeneration": "EFO_0001481",
    "amd": "EFO_0001481",
    "macular degeneration": "EFO_0001481",
}

# Disease → modifier types (which evidence types are relevant)
DISEASE_MODIFIER_MAP: dict[str, list[str]] = {
    "EFO_0001645": ["germline", "somatic_chip", "drug"],     # CAD: CHIP + germline strong
    "EFO_0000685": ["germline", "viral", "drug"],            # RA: EBV + HLA
    "EFO_0002690": ["germline", "viral", "drug"],            # SLE: EBV + interferons
    "EFO_0003885": ["germline", "viral", "drug"],            # MS: EBV strong
    "EFO_0001359": ["germline", "viral"],                    # T1D: HLA + viral
    "EFO_0000341": ["germline", "drug"],                     # COPD: germline + drug
    "EFO_0003767": ["germline", "drug"],                     # IBD: NOD2/IL23R germline + anti-TNF drug
    "EFO_0000384": ["germline", "drug"],                     # Crohn's disease
    "EFO_0000729": ["germline", "drug"],                     # Ulcerative colitis
    "EFO_0001481": ["germline", "drug"],                     # AMD: complement germline (CFH/C3) + anti-VEGF drug
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
        "EFO_0001645": "ieu-a-7",          # CAD — CARDIoGRAMplusC4D
        "EFO_0003767": "ieu-b-30",         # IBD
        "EFO_0000685": "ieu-a-833",        # RA
        "EFO_0001360": "ieu-a-26",         # T2D
        "EFO_0001481": "ebi-a-GCST006909", # AMD — Fritsche 2016, 69k samples
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

    # FinnGen phenocode lookup via finngen_server (covers 2,272 R10 endpoints)
    finngen_phenocode = None
    finngen_n_cases = None
    finngen_n_controls = None
    if efo_id:
        resolved_code = efo_to_finngen_phenocode(efo_id)
        if resolved_code:
            try:
                fg = get_finngen_phenotype_info(resolved_code)
                if not fg.get("error"):
                    finngen_phenocode = fg.get("phenocode")
                    finngen_n_cases = fg.get("n_cases")
                    finngen_n_controls = fg.get("n_controls")
            except Exception:
                finngen_phenocode = resolved_code  # use known code even if API fails

    return {
        "disease_name":      disease_name,
        "efo_id":            efo_id,
        "icd10_codes":       icd10,
        "modifier_types":    modifiers,
        "primary_gwas_id":   primary_gwas_id,
        "n_gwas_studies":    n_gwas,
        "finngen_phenocode": finngen_phenocode,
        "finngen_n_cases":   finngen_n_cases,
        "finngen_n_controls": finngen_n_controls,
        "use_precomputed_only": True,
        "day_one_mode":      True,
    }
