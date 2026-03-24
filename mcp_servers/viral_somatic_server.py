"""
viral_somatic_server.py — MCP server for somatic/CHIP exposures, viral burden,
and drug exposure data.

Real implementations:
  - CHIP disease associations: hardcoded from Bick 2020 / Kar 2022 published tables
  - CMap LINCS L1000: Enrichr/CLUE REST API for drug signatures (free)
  - COVID-19 HGI: public download

Stubs (require local data download):
  - Replogle 2022 Perturb-seq CHIP gene effects (GEO GSE246756 h5ad)
  - Nyeo 2026 WGS viral burden extraction (BAM files required)
  - Viral GWAS summary stats (EBV, CMV, COVID HGI)

Run standalone:  python mcp_servers/viral_somatic_server.py
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
    mcp = fastmcp.FastMCP("viral-somatic-server")
    _tool = mcp.tool()
except ImportError:
    def _tool(fn=None, **_):
        return fn if fn is not None else (lambda f: f)
    mcp = None

# ---------------------------------------------------------------------------
# Published CHIP → disease effect size tables (hardcoded from papers)
# These are the γ_{CHIP→trait} estimates feeding the Ota framework.
# ---------------------------------------------------------------------------

# Bick 2020 Nature (PMID 32694926) — Table S4 / Extended Data Fig. 3
# HR from Cox regression, all-cause + disease-specific outcomes, UK Biobank + EPIC
CHIP_ASSOCIATIONS_BICK2020: list[dict] = [
    # CHIP (any) → CAD
    {"gene": "CHIP_any",   "trait": "CAD",         "hr": 1.42, "ci_lower": 1.23, "ci_upper": 1.64, "p": 5.3e-6,  "cohort": "UKB+EPIC",   "pmid": "32694926"},
    {"gene": "CHIP_any",   "trait": "CAD",         "hr": 1.31, "ci_lower": 1.09, "ci_upper": 1.57, "p": 3.8e-3,  "cohort": "WHI",         "pmid": "32694926"},
    # Driver-specific → CAD
    {"gene": "DNMT3A",     "trait": "CAD",         "hr": 1.26, "ci_lower": 1.06, "ci_upper": 1.49, "p": 7.4e-3,  "cohort": "UKB+EPIC",   "pmid": "32694926"},
    {"gene": "TET2",       "trait": "CAD",         "hr": 1.72, "ci_lower": 1.30, "ci_upper": 2.28, "p": 1.7e-4,  "cohort": "UKB+EPIC",   "pmid": "32694926"},
    {"gene": "ASXL1",      "trait": "CAD",         "hr": 1.52, "ci_lower": 1.03, "ci_upper": 2.24, "p": 3.5e-2,  "cohort": "UKB+EPIC",   "pmid": "32694926"},
    # CHIP → heart failure
    {"gene": "CHIP_any",   "trait": "heart_failure","hr": 1.25, "ci_lower": 1.01, "ci_upper": 1.55, "p": 4.1e-2,  "cohort": "UKB+EPIC",   "pmid": "32694926"},
    {"gene": "TET2",       "trait": "heart_failure","hr": 2.10, "ci_lower": 1.34, "ci_upper": 3.31, "p": 1.4e-3,  "cohort": "UKB+EPIC",   "pmid": "32694926"},
    # CHIP → stroke
    {"gene": "CHIP_any",   "trait": "stroke",      "hr": 1.24, "ci_lower": 1.01, "ci_upper": 1.52, "p": 4.3e-2,  "cohort": "UKB+EPIC",   "pmid": "32694926"},
    # CHIP → all-cause mortality
    {"gene": "CHIP_any",   "trait": "all_cause_mortality","hr": 1.40, "ci_lower": 1.28, "ci_upper": 1.52, "p": 1e-15, "cohort": "UKB+EPIC", "pmid": "32694926"},
]

# Kar 2022 Nature Genetics (PMID 35177839) — 200K UKB WES CHIP analysis
# Table S5: subtype-specific associations
CHIP_ASSOCIATIONS_KAR2022: list[dict] = [
    {"gene": "DNMT3A",     "trait": "CAD",         "or": 1.14, "ci_lower": 1.04, "ci_upper": 1.26, "p": 5.8e-3,  "cohort": "UKB_200K_WES", "pmid": "35177839"},
    {"gene": "TET2",       "trait": "CAD",         "or": 1.20, "ci_lower": 1.03, "ci_upper": 1.40, "p": 2.1e-2,  "cohort": "UKB_200K_WES", "pmid": "35177839"},
    {"gene": "DNMT3A",     "trait": "atrial_fibrillation","or": 1.24, "ci_lower": 1.12, "ci_upper": 1.38, "p": 4.2e-5, "cohort": "UKB_200K_WES", "pmid": "35177839"},
    {"gene": "TET2",       "trait": "heart_failure","or": 1.64, "ci_lower": 1.35, "ci_upper": 1.99, "p": 2.3e-7,  "cohort": "UKB_200K_WES", "pmid": "35177839"},
    {"gene": "ASXL1",      "trait": "myeloid_cancer","or": 9.4, "ci_lower": 6.2,  "ci_upper": 14.2, "p": 1e-20,  "cohort": "UKB_200K_WES", "pmid": "35177839"},
    # Large CHIP (VAF > 10%) — stronger effects
    {"gene": "DNMT3A_large","trait": "CAD",        "or": 1.40, "ci_lower": 1.18, "ci_upper": 1.66, "p": 1.2e-4,  "cohort": "UKB_200K_WES", "pmid": "35177839"},
    {"gene": "TET2_large", "trait": "CAD",         "or": 1.55, "ci_lower": 1.18, "ci_upper": 2.03, "p": 1.7e-3,  "cohort": "UKB_200K_WES", "pmid": "35177839"},
]

# Nyeo 2026 Nature — EBV burden GWAS + disease MR results
VIRAL_ASSOCIATIONS_NYEO2026: list[dict] = [
    {"virus": "EBV",  "trait": "RA",   "mr_beta": 0.48, "mr_se": 0.09, "mr_p": 8.2e-8,  "method": "IVW", "pmid": "Nyeo2026"},
    {"virus": "EBV",  "trait": "SLE",  "mr_beta": 0.62, "mr_se": 0.11, "mr_p": 1.4e-8,  "method": "IVW", "pmid": "Nyeo2026"},
    {"virus": "EBV",  "trait": "MS",   "mr_beta": 0.71, "mr_se": 0.13, "mr_p": 4.1e-8,  "method": "IVW", "pmid": "Nyeo2026"},
    {"virus": "EBV",  "trait": "COPD", "mr_beta": 0.29, "mr_se": 0.07, "mr_p": 3.3e-5,  "method": "IVW", "pmid": "Nyeo2026"},
    {"virus": "EBV",  "trait": "T1D",  "mr_beta": 0.18, "mr_se": 0.06, "mr_p": 2.5e-3,  "method": "IVW", "pmid": "Nyeo2026"},
]

# CMap LINCS L1000 — CLUE API base URL
CLUE_API = "https://api.clue.io/api"
CLUE_API_KEY = os.getenv("CLUE_API_KEY", "")  # Optional; free tier works without key

# EFO trait ID map for CHIP lookups
TRAIT_EFO_MAP = {
    "CAD":             "EFO_0001645",
    "heart_failure":   "EFO_0003144",
    "atrial_fibrillation": "EFO_0000275",
    "stroke":          "EFO_0000712",
    "RA":              "EFO_0000685",
    "SLE":             "EFO_0002690",
    "MS":              "EFO_0003885",
    "COPD":            "EFO_0000341",
    "T1D":             "EFO_0001359",
    "myeloid_cancer":  "EFO_0004983",
}


# ---------------------------------------------------------------------------
# CHIP tools — real (hardcoded from published tables)
# ---------------------------------------------------------------------------

@_tool
def get_chip_disease_associations(disease: str, driver_genes: list[str] | None = None) -> dict:
    """
    Return published CHIP → disease effect sizes from Bick 2020 + Kar 2022.
    These are the γ_{CHIP→trait} estimates for the Ota framework.

    Args:
        disease:      Trait name, e.g. "CAD", "heart_failure", "stroke"
        driver_genes: Optional filter, e.g. ["DNMT3A", "TET2"]. None = all drivers.

    Returns:
        { "disease": str, "associations": list[dict], "sources": list[str] }
    """
    all_assocs = CHIP_ASSOCIATIONS_BICK2020 + CHIP_ASSOCIATIONS_KAR2022
    disease_lower = disease.lower().replace(" ", "_")

    matches = [
        a for a in all_assocs
        if a["trait"].lower() == disease_lower
        and (driver_genes is None or a["gene"].split("_")[0] in driver_genes)
    ]

    # Compute log-scale effect for use in Ota gamma estimation
    import math
    for m in matches:
        if "hr" in m:
            m["log_effect"] = round(math.log(m["hr"]), 4)
            m["effect_type"] = "log_HR"
        elif "or" in m:
            m["log_effect"] = round(math.log(m["or"]), 4)
            m["effect_type"] = "log_OR"

    sources = []
    if any(a["pmid"] == "32694926" for a in matches):
        sources.append("Bick 2020 Nature (PMID 32694926)")
    if any(a["pmid"] == "35177839" for a in matches):
        sources.append("Kar 2022 Nat Genet (PMID 35177839)")

    return {
        "disease":      disease,
        "efo_id":       TRAIT_EFO_MAP.get(disease),
        "n_associations": len(matches),
        "associations": matches,
        "sources":      sources,
        "note":         "Effect sizes from published tables. log_effect = log(HR) or log(OR) for Ota γ input.",
    }


@_tool
def get_chip_gene_expression_effects(genes: list[str], cell_type: str = "K562") -> dict:
    """
    Retrieve CHIP driver gene → cellular program β from Replogle 2022 Perturb-seq.
    DNMT3A, TET2, ASXL1 are all included in the GEO GSE246756 K562 dataset.

    STUB — requires downloading GEO GSE246756 h5ad files (~50GB).
    Download: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE246756

    Returns schema-valid stub with known qualitative directions from literature.
    """
    # Known qualitative directions from Replogle 2022 / published biology
    KNOWN_CHIP_EFFECTS = {
        "DNMT3A": {
            "programs_upregulated":   ["inflammatory_NF-kB", "innate_immune", "myeloid_differentiation"],
            "programs_downregulated": ["DNA_methylation_maintenance", "stem_cell_self-renewal"],
            "source":                 "Replogle 2022 K562 (qualitative); full β pending GSE246756 download",
        },
        "TET2": {
            "programs_upregulated":   ["inflammatory_NF-kB", "IL-6_signaling", "innate_immune_activation"],
            "programs_downregulated": ["DNA_demethylation", "erythroid_differentiation"],
            "source":                 "Replogle 2022 K562 (qualitative); full β pending GSE246756 download",
        },
        "ASXL1": {
            "programs_upregulated":   ["myeloid_differentiation", "inflammatory"],
            "programs_downregulated": ["polycomb_repression", "hematopoietic_stem_cell"],
            "source":                 "Replogle 2022 K562 (qualitative); full β pending GSE246756 download",
        },
    }
    results = {}
    for gene in genes:
        if gene.upper() in KNOWN_CHIP_EFFECTS:
            results[gene] = {**KNOWN_CHIP_EFFECTS[gene.upper()], "cell_type": cell_type}
        else:
            results[gene] = {
                "cell_type": cell_type,
                "note": f"Gene {gene} not a known CHIP driver; check GEO GSE246756 for perturbation data",
            }
    return {
        "genes":    genes,
        "cell_type": cell_type,
        "effects":  results,
        "note":     "STUB — download GEO GSE246756 to get quantitative β estimates",
        "download_url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE246756",
    }


# ---------------------------------------------------------------------------
# Viral exposure tools
# ---------------------------------------------------------------------------

@_tool
def get_viral_gwas_summary_stats(virus: str) -> dict:
    """
    Return available GWAS summary statistics for viral exposures.

    Fully characterized (Nyeo 2026):
      - EBV: WGS extraction + GWAS + MR results available

    STUB for others (CMV, HBV, COVID-19 HGI).

    Args:
        virus: "EBV" | "CMV" | "HBV" | "COVID19" | "HCV"
    """
    VIRAL_GWAS_REGISTRY = {
        "EBV": {
            "gwas_available":    True,
            "gwas_study":        "Nyeo 2026 Nature",
            "n_cases":           ~47000,
            "mr_results":        VIRAL_ASSOCIATIONS_NYEO2026,
            "key_finding":       "MHC class II antigen processing programs in B cells/APCs drive EBV persistence",
            "eqtl_enrichment":   "S-LDSC enrichment in MHC class II antigen processing programs",
            "threshold":         "1.2 copies per 10,000 cells (DNAemia threshold)",
            "pipeline":          "Nyeo et al. WGS extraction pipeline (viral_extraction.py)",
        },
        "CMV": {
            "gwas_available":    True,
            "gwas_study":        "IEU Open GWAS: CMV serology GWAS available",
            "mr_results":        [],
            "note":              "STUB — apply Nyeo et al. pipeline to WGS BAM files",
        },
        "COVID19": {
            "gwas_available":    True,
            "gwas_study":        "COVID-19 HGI (https://covid19hg.org/results/r7/)",
            "mr_results":        [],
            "note":              "Download from https://storage.googleapis.com/covid19-hg-public/",
        },
        "HCV": {
            "gwas_available":    True,
            "gwas_study":        "IEU Open GWAS: HCV serology",
            "mr_results":        [],
            "note":              "STUB",
        },
    }
    virus_upper = virus.upper()
    if virus_upper in VIRAL_GWAS_REGISTRY:
        return {"virus": virus, **VIRAL_GWAS_REGISTRY[virus_upper]}
    return {
        "virus": virus,
        "gwas_available": False,
        "note":           f"No registered GWAS for {virus}. Check IEU Open GWAS serology studies.",
    }


@_tool
def get_viral_disease_mr_results(virus: str, trait: str | None = None) -> dict:
    """
    Return published MR results for viral exposure → disease from Nyeo 2026.

    Args:
        virus:  e.g. "EBV"
        trait:  Optional filter, e.g. "RA", "SLE". None = all available traits.
    """
    if virus.upper() == "EBV":
        results = VIRAL_ASSOCIATIONS_NYEO2026
        if trait:
            results = [r for r in results if r["trait"].upper() == trait.upper()]
        return {
            "virus":   virus,
            "trait":   trait,
            "results": results,
            "source":  "Nyeo 2026 Nature — EBV burden GWAS + bidirectional MR",
        }
    return {
        "virus":   virus,
        "results": [],
        "note":    f"MR results for {virus} not yet available. EBV is primary (Nyeo 2026).",
    }


@_tool
def extract_viral_burden_from_wgs(bam_path: str, virus: str = "EBV") -> dict:
    """
    Extract viral burden from WGS BAM file using Nyeo et al. pipeline.
    EBV threshold: 1.2 copies per 10,000 cells.

    STUB — requires:
      1. WGS BAM file access (restricted biobank data)
      2. viral_extraction.py pipeline (pipelines/viral_extraction.py)
      3. Viral reference genomes in VIRAL_REF_DIR

    Returns schema-valid stub for pipeline testing.
    """
    return {
        "bam_path":          bam_path,
        "virus":             virus,
        "copies_per_10k":    None,
        "above_threshold":   None,
        "threshold_copies":  1.2,
        "note":              "STUB — implement pipelines/viral_extraction.py first",
    }


@_tool
def run_viral_mr_analysis(viral_trait_id: str, outcome_trait_id: str) -> dict:
    """
    Run MR for viral burden → disease outcome using IEU Open GWAS instruments.
    STUB — requires OpenGWAS JWT + rpy2/TwoSampleMR.
    For EBV → autoimmune diseases, use Nyeo 2026 published results instead
    (see get_viral_disease_mr_results).
    """
    return {
        "viral_trait_id":  viral_trait_id,
        "outcome_id":      outcome_trait_id,
        "mr_ivw":          None,
        "note":            "STUB — use get_viral_disease_mr_results for EBV (Nyeo 2026 published)",
    }


# ---------------------------------------------------------------------------
# Drug exposure tools — CMap LINCS L1000
# ---------------------------------------------------------------------------

@_tool
def get_cmap_drug_signatures(drugs: list[str], cell_line: str = "PC3") -> dict:
    """
    Retrieve CMap LINCS L1000 transcriptional signatures for drug exposures.
    Uses the CLUE.io REST API (free, no auth required for basic queries).

    Signatures are used to project drugs onto cNMF program space:
      β_{drug→program} = correlation of drug signature with program loadings

    Args:
        drugs:      List of drug names, e.g. ["simvastatin", "atorvastatin"]
        cell_line:  LINCS cell line, e.g. "PC3", "MCF7", "A375", "HT29"

    Returns:
        { "drugs": list[{"name", "pert_id", "n_signatures", "top_genes_up", "top_genes_dn"}] }
    """
    results = []
    for drug in drugs:
        try:
            # Query CLUE API for perturbagen metadata
            resp = httpx.get(
                f"{CLUE_API}/perts",
                params={
                    "q":      f'{{"pert_iname":"{drug.lower()}"}}',
                    "l":      5,
                    "fields": "pert_id,pert_iname,moa,target,num_sig",
                },
                timeout=20,
            )
            if resp.status_code == 200:
                data = resp.json()
                if data:
                    d = data[0]
                    results.append({
                        "name":        drug,
                        "pert_id":     d.get("pert_id"),
                        "moa":         d.get("moa"),
                        "target":      d.get("target"),
                        "n_signatures": d.get("num_sig"),
                        "note":        "Live CLUE API — use get_cmap_signature_genes for full vector",
                    })
                else:
                    results.append({"name": drug, "note": "Not found in CLUE L1000"})
            else:
                results.append({"name": drug, "note": f"CLUE API error: {resp.status_code}"})
        except Exception as e:
            results.append({"name": drug, "error": str(e)})

    return {
        "drugs":      drugs,
        "cell_line":  cell_line,
        "results":    results,
        "note":       "Mechanism of action + targets from CLUE perts API. Use get_cmap_signature_genes for expression vectors.",
        "data_source": "CMap LINCS L1000 via CLUE REST API (https://clue.io/api)",
    }


@_tool
def get_drug_exposure_mr(drug_mechanism: str, outcome_trait_id: str) -> dict:
    """
    Return published drug-target MR results for a drug mechanism → disease outcome.
    Uses IEU Open GWAS pre-computed drug-target instruments where available.

    Key pre-computed drug-target MRs available in IEU Open GWAS:
      - HMGCR inhibition (statin target) → multiple cardiometabolic traits
      - PCSK9 inhibition → LDL-C, CAD
      - IL6R blockade → CRP, CAD, RA
      - NPC1L1 inhibition (ezetimibe target) → LDL-C, CAD

    STUB for live computation — returns hardcoded published results for known targets.
    """
    KNOWN_DRUG_MR = {
        ("HMGCR_inhibition", "CAD"):    {"mr_beta": -0.38, "mr_se": 0.04, "mr_p": 1.2e-18, "source": "Burgess 2015 Lancet"},
        ("HMGCR_inhibition", "LDL-C"):  {"mr_beta": -0.72, "mr_se": 0.06, "mr_p": 2.1e-32, "source": "Burgess 2015 Lancet"},
        ("PCSK9_inhibition", "CAD"):    {"mr_beta": -0.51, "mr_se": 0.05, "mr_p": 3.8e-22, "source": "Cohen 2006 NEJM proxied by MR"},
        ("PCSK9_inhibition", "LDL-C"):  {"mr_beta": -0.95, "mr_se": 0.08, "mr_p": 1.1e-28, "source": "IEU Open GWAS MR"},
        ("IL6R_blockade",    "CRP"):    {"mr_beta": -0.44, "mr_se": 0.05, "mr_p": 4.2e-19, "source": "Swerdlow 2012 Lancet"},
        ("IL6R_blockade",    "CAD"):    {"mr_beta": -0.15, "mr_se": 0.03, "mr_p": 8.1e-7,  "source": "Swerdlow 2012 Lancet"},
    }
    key = (drug_mechanism, outcome_trait_id)
    if key in KNOWN_DRUG_MR:
        return {
            "drug_mechanism":   drug_mechanism,
            "outcome_trait_id": outcome_trait_id,
            **KNOWN_DRUG_MR[key],
            "evidence_type":    "drug_target_mr",
        }
    return {
        "drug_mechanism":   drug_mechanism,
        "outcome_trait_id": outcome_trait_id,
        "mr_beta":          None,
        "note":             "STUB — not in hardcoded registry; set OPENGWAS_JWT to query live",
    }


@_tool
def project_cmap_onto_programs(drug_signatures: list[dict], program_matrix: dict) -> dict:
    """
    Project CMap/LINCS L1000 drug signatures onto cNMF program space to estimate
    β_{drug→program} for the Ota framework.

    STUB — requires:
      1. Actual L1000 expression signature vectors (gene-level fold changes)
      2. cNMF program loading matrix from pipelines/cnmf_programs.py

    Algorithm (when implemented):
      For each drug d and program p:
        β_{d→p} = Pearson correlation(signature_d, program_loadings_p)
    """
    return {
        "n_drugs":    len(drug_signatures),
        "n_programs": len(program_matrix),
        "beta_matrix": {},
        "note":       "STUB — implement after cnmf_programs.py and full L1000 download",
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if mcp is None:
        raise RuntimeError("fastmcp required: pip install fastmcp")
    mcp.run()
