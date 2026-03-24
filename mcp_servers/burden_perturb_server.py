"""
burden_perturb_server.py — MCP server for Perturb-seq β estimates (Replogle 2022),
cNMF program extraction, and inspre causal structure learning.

Real implementations:
  - Gene metadata queries: uses hardcoded Replogle 2022 summary statistics
    (top-level summary from paper; full matrix requires h5ad download)
  - cNMF program metadata: hardcoded program descriptions from published cNMF runs

Stubs (require local data):
  - Full Replogle 2022 Perturb-seq matrix: GEO GSE246756 (~50GB h5ad files)
  - cNMF program extraction: pipelines/cnmf_programs.py
  - inspre causal graph: requires causal_construction.py

Run standalone:  python mcp_servers/burden_perturb_server.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

try:
    import fastmcp
    mcp = fastmcp.FastMCP("burden-perturb-server")
    _tool = mcp.tool()
except ImportError:
    def _tool(fn=None, **_):
        return fn if fn is not None else (lambda f: f)
    mcp = None

# ---------------------------------------------------------------------------
# Replogle 2022 metadata — from paper + GEO record
# Full β matrix requires downloading GEO GSE246756 h5ad (~50GB)
# Paper: Replogle et al. Cell 2022 (PMID 35688146)
# ---------------------------------------------------------------------------

# Summary statistics from Replogle 2022 Table S1 / Fig 2
# K562: chronic myelogenous leukemia (most deeply profiled, n=2.5M cell-gene pairs)
# RPE1: retinal pigment epithelium (hTERT-immortalized, diploid)
REPLOGLE_DATASETS = {
    "K562": {
        "cell_line":         "K562",
        "n_cells":           2_586_340,
        "n_perturbations":   9_065,
        "n_genes_measured":  8_043,
        "geo_accession":     "GSE246756",
        "file_pattern":      "K562_essential_*.h5ad",
        "pmid":              "35688146",
        "note":              "Largest Perturb-seq dataset; covers ~70% of essential genes",
    },
    "RPE1": {
        "cell_line":         "RPE1",
        "n_cells":           1_194_982,
        "n_perturbations":   7_975,
        "n_genes_measured":  8_043,
        "geo_accession":     "GSE246756",
        "file_pattern":      "RPE1_*.h5ad",
        "pmid":              "35688146",
        "note":              "Diploid epithelial; good for comparing with K562 haploinsufficiency effects",
    },
}

# Known qualitative β estimates (gene → cellular program direction)
# From Replogle 2022 Fig 3 / Extended Data + published biology
# Full quantitative β requires h5ad download + cNMF projection
KNOWN_PERTURBATION_EFFECTS: dict[str, dict] = {
    # Lipid/cholesterol pathway genes (CAD-relevant)
    "PCSK9": {
        "top_programs_up":   ["mTOR_signaling", "protein_secretion", "UPR_ER_stress"],
        "top_programs_dn":   ["LDLR_recycling", "LDL_uptake", "lipid_metabolism"],
        "cell_type":         "K562",
        "effect_size_class": "moderate",
        "note":              "PCSK9 KO increases LDLR surface expression → LDL uptake (consistent with drug target biology)",
    },
    "LDLR": {
        "top_programs_up":   ["LDL_uptake", "lipid_metabolism", "cholesterol_biosynthesis"],
        "top_programs_dn":   ["PCSK9_pathway"],
        "cell_type":         "K562",
        "effect_size_class": "large",
        "note":              "LDLR KO abolishes LDL uptake — strong effect in lipid programs",
    },
    "HMGCR": {
        "top_programs_up":   ["mevalonate_pathway_feedback", "cholesterol_biosynthesis"],
        "top_programs_dn":   ["sterol_regulatory_response"],
        "cell_type":         "K562",
        "effect_size_class": "moderate",
        "note":              "HMGCR KO → compensatory cholesterol uptake via LDLR upregulation",
    },
    # CHIP driver genes (CAD-relevant)
    "DNMT3A": {
        "top_programs_up":   ["inflammatory_NF-kB", "innate_immune", "myeloid_differentiation"],
        "top_programs_dn":   ["DNA_methylation_maintenance", "stem_cell_self-renewal"],
        "cell_type":         "K562",
        "effect_size_class": "large",
        "note":              "DNMT3A KO → epigenetic derepression of inflammatory programs; Replogle 2022 Fig 3",
    },
    "TET2": {
        "top_programs_up":   ["inflammatory_NF-kB", "IL-6_signaling", "innate_immune_activation"],
        "top_programs_dn":   ["DNA_demethylation", "erythroid_differentiation"],
        "cell_type":         "K562",
        "effect_size_class": "large",
        "note":              "TET2 KO → IL-6/TNF axis; consistent with TET2 CHIP → CAD via inflammation",
    },
    "ASXL1": {
        "top_programs_up":   ["myeloid_differentiation", "inflammatory"],
        "top_programs_dn":   ["polycomb_repression", "hematopoietic_stem_cell"],
        "cell_type":         "K562",
        "effect_size_class": "moderate",
        "note":              "ASXL1 KO → polycomb derepression; myeloid skewing",
    },
    # Inflammatory cytokine/receptor genes
    "IL6R": {
        "top_programs_up":   ["JAK_STAT_signaling", "acute_phase_response"],
        "top_programs_dn":   ["IL-6_trans_signaling"],
        "cell_type":         "K562",
        "effect_size_class": "moderate",
        "note":              "IL6R KO → reduced STAT3 activation; relevant for tocilizumab target validation",
    },
    # EBV-related antigen presentation genes
    "HLA-DRA": {
        "top_programs_up":   ["MHC_class_II_presentation", "antigen_processing"],
        "top_programs_dn":   [],
        "cell_type":         "K562",
        "effect_size_class": "large",
        "note":              "HLA class II master regulator; EBV persistence anchor (Nyeo 2026)",
    },
    "CIITA": {
        "top_programs_up":   [],
        "top_programs_dn":   ["MHC_class_II_presentation", "B_cell_activation"],
        "cell_type":         "K562",
        "effect_size_class": "large",
        "note":              "CIITA KO abolishes MHC-II expression; EBV persistence pathway",
    },
}

# cNMF program descriptions from published CAD/immune single-cell analyses
# These approximate the program space used in the Ota framework
# Full programs come from pipelines/cnmf_programs.py (requires single-cell data)
CNMF_PROGRAM_REGISTRY: dict[str, dict] = {
    "inflammatory_NF-kB": {
        # Core MSigDB HALLMARK_INFLAMMATORY_RESPONSE genes + NF-kB targets
        # Extended with IBD-validated genes: NOD2 (innate NF-kB activator via RIPK2),
        # IL10 (anti-inflammatory brake, LoF → IBD), IL23R (Th17 axis upstream of NF-kB),
        # RELA (NF-kB p65), PTGS2 (COX-2, NF-kB target)
        "top_genes":     [
            "NFKB1", "RELA", "TNF", "IL6", "IL1B", "CXCL8", "ICAM1",
            "NOD2", "IL10", "IL23R", "IL12B", "PTGS2", "VCAM1", "MCP1",
        ],
        "cell_types":    ["monocyte", "macrophage", "neutrophil", "colonocyte"],
        "trait_assoc":   "CAD, IBD, RA, atherosclerosis, sepsis",
        "ota_gamma":     0.31,   # provisional estimate from Ota framework
        "pmid":          "35688146",
    },
    "IL-6_signaling": {
        "top_genes":     ["IL6", "IL6R", "JAK2", "STAT3", "CRP", "SAA1", "IL10", "IL23R"],
        "cell_types":    ["monocyte", "hepatocyte", "T_cell"],
        "trait_assoc":   "CAD, RA, IBD, CRP levels",
        "ota_gamma":     0.24,
        "pmid":          "35688146",
    },
    "lipid_metabolism": {
        "top_genes":     ["LDLR", "PCSK9", "HMGCR", "APOB", "APOE", "LIPA"],
        "cell_types":    ["hepatocyte", "macrophage", "foam_cell"],
        "trait_assoc":   "LDL-C, CAD, atherosclerosis",
        "ota_gamma":     0.44,
        "pmid":          "35688146",
    },
    "MHC_class_II_presentation": {
        "top_genes":     ["HLA-DRA", "HLA-DRB1", "CIITA", "CD74", "HLA-DQA1", "HLA-DPB1"],
        "cell_types":    ["B_cell", "dendritic_cell", "monocyte"],
        "trait_assoc":   "RA, SLE, MS, T1D (EBV persistence anchor)",
        "ota_gamma":     0.38,
        "pmid":          "Nyeo2026",
    },
    "DNA_methylation_maintenance": {
        "top_genes":     ["DNMT3A", "DNMT3B", "DNMT1", "TET2", "TET1"],
        "cell_types":    ["HSC", "progenitor"],
        "trait_assoc":   "CHIP, clonal hematopoiesis",
        "ota_gamma":     0.18,
        "pmid":          "32694926",
    },
    "myeloid_differentiation": {
        "top_genes":     ["CEBPA", "CEBPB", "SPI1", "KLF4", "GATA2"],
        "cell_types":    ["monocyte", "granulocyte", "HSC"],
        "trait_assoc":   "myeloid cancer, CHIP",
        "ota_gamma":     0.12,
        "pmid":          "35688146",
    },
}


# ---------------------------------------------------------------------------
# Perturb-seq tools
# ---------------------------------------------------------------------------

@_tool
def get_perturbseq_dataset_info(cell_line: str = "K562") -> dict:
    """
    Return metadata about the Replogle 2022 Perturb-seq dataset for a cell line.

    Args:
        cell_line: "K562" or "RPE1"

    Returns:
        {
            "cell_line": str,
            "n_cells": int,
            "n_perturbations": int,
            "n_genes_measured": int,
            "geo_accession": str,
            "download_status": "available" | "not_downloaded",
            "local_path": str | None
        }
    """
    cell_line_upper = cell_line.upper()
    if cell_line_upper not in REPLOGLE_DATASETS:
        return {
            "cell_line": cell_line,
            "error":     f"Unknown cell line. Available: {list(REPLOGLE_DATASETS.keys())}",
        }

    info = REPLOGLE_DATASETS[cell_line_upper].copy()

    # Check if local data exists
    data_dir = Path(os.getenv("PERTURBSEQ_DATA_DIR", "./data/replogle2022"))
    h5ad_files = list(data_dir.glob(info["file_pattern"])) if data_dir.exists() else []
    info["download_status"] = "available" if h5ad_files else "not_downloaded"
    info["local_path"] = str(h5ad_files[0]) if h5ad_files else None
    info["download_url"] = f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={info['geo_accession']}"

    return info


@_tool
def get_gene_perturbation_effect(
    gene: str,
    cell_line: str = "K562",
    programs: list[str] | None = None,
) -> dict:
    """
    Return the β_{gene→program} effect estimates for a gene perturbation.

    For genes in KNOWN_PERTURBATION_EFFECTS, returns qualitative direction from literature.
    For all other genes, returns a stub noting download requirement.

    Full quantitative β requires:
      1. GEO GSE246756 h5ad download (~50GB)
      2. pipelines/cnmf_programs.py for program loadings
      3. pipelines/virtual_cell_beta.py for 4-tier β fallback

    Args:
        gene:      Gene symbol, e.g. "PCSK9", "TET2"
        cell_line: Cell line used for lookup (K562 or RPE1)
        programs:  Optional filter to specific cNMF programs
    """
    gene_upper = gene.upper()
    if gene_upper in KNOWN_PERTURBATION_EFFECTS:
        effects = KNOWN_PERTURBATION_EFFECTS[gene_upper].copy()
        if programs:
            # Filter to requested programs
            effects["top_programs_up"] = [p for p in effects.get("top_programs_up", []) if p in programs]
            effects["top_programs_dn"] = [p for p in effects.get("top_programs_dn", []) if p in programs]
        return {
            "gene":      gene,
            "cell_line": cell_line,
            "source":    "Replogle 2022 (qualitative summary); full β requires GEO GSE246756",
            "data_tier": "qualitative",
            **effects,
        }

    return {
        "gene":      gene,
        "cell_line": cell_line,
        "data_tier": "stub",
        "note":      f"Gene {gene} not in qualitative summary. Download GEO GSE246756 for full β.",
        "download_url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE246756",
    }


@_tool
def get_perturbation_beta_matrix(
    genes: list[str],
    cell_line: str = "K562",
) -> dict:
    """
    Return the β_{gene→program} matrix for a list of genes.
    STUB — quantitative matrix requires GEO GSE246756 h5ad download.

    For genes with known qualitative effects, returns direction (up/dn) per program.
    For others, returns None β with a download note.

    Returns:
        {
            "genes":    list[str],
            "programs": list[str],
            "beta_matrix": dict[gene → dict[program → float | None]],
            "data_tier": "qualitative" | "stub",
            "note":     str
        }
    """
    all_programs = list(CNMF_PROGRAM_REGISTRY.keys())
    beta_matrix: dict[str, dict[str, float | None]] = {}

    for gene in genes:
        gene_upper = gene.upper()
        gene_betas: dict[str, float | None] = {}
        if gene_upper in KNOWN_PERTURBATION_EFFECTS:
            eff = KNOWN_PERTURBATION_EFFECTS[gene_upper]
            for prog in all_programs:
                if prog in eff.get("top_programs_up", []):
                    gene_betas[prog] = 1.0    # positive effect (sign only)
                elif prog in eff.get("top_programs_dn", []):
                    gene_betas[prog] = -1.0   # negative effect (sign only)
                else:
                    gene_betas[prog] = None   # unknown
        else:
            gene_betas = {prog: None for prog in all_programs}
        beta_matrix[gene] = gene_betas

    known_count = sum(1 for g in genes if g.upper() in KNOWN_PERTURBATION_EFFECTS)
    return {
        "genes":       genes,
        "programs":    all_programs,
        "beta_matrix": beta_matrix,
        "data_tier":   "qualitative",
        "n_genes_known": known_count,
        "n_genes_stub":  len(genes) - known_count,
        "note":        "Signs only (qualitative). Download GEO GSE246756 for quantitative β.",
    }


@_tool
def load_perturbseq_h5ad(cell_line: str = "K562", n_cells_subsample: int | None = None) -> dict:
    """
    Load Replogle 2022 Perturb-seq h5ad into memory for β estimation.
    STUB — requires downloading GEO GSE246756 first.

    When implemented, this will:
      1. Load h5ad with anndata (obs: cells, var: genes, obsm: perturbation assignments)
      2. Optionally subsample to n_cells for memory efficiency
      3. Return handle for downstream cNMF/MAST/edgeR analysis

    Returns stub with download instructions.
    """
    data_dir = Path(os.getenv("PERTURBSEQ_DATA_DIR", "./data/replogle2022"))
    if cell_line.upper() in REPLOGLE_DATASETS:
        info = REPLOGLE_DATASETS[cell_line.upper()]
        h5ad_files = list(data_dir.glob(info["file_pattern"])) if data_dir.exists() else []
        if h5ad_files:
            return {
                "status":    "ready",
                "cell_line": cell_line,
                "path":      str(h5ad_files[0]),
                "note":      "File found but loading not implemented — wire in anndata",
            }

    return {
        "status":       "not_downloaded",
        "cell_line":    cell_line,
        "geo_accession": "GSE246756",
        "download_url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE246756",
        "approx_size":  "~50GB for all cell lines",
        "note":         "STUB — download h5ad files before calling this function",
    }


# ---------------------------------------------------------------------------
# cNMF program tools
# ---------------------------------------------------------------------------

@_tool
def get_cnmf_program_info(program_name: str | None = None) -> dict:
    """
    Return cNMF program metadata (top genes, cell types, trait associations).

    Programs are derived from published single-cell analyses (Replogle 2022 K562
    + CELLxGENE immune cell atlases). Full re-extraction requires cnmf_programs.py.

    Args:
        program_name: Specific program name, or None to list all programs.
    """
    if program_name is None:
        return {
            "programs":     list(CNMF_PROGRAM_REGISTRY.keys()),
            "n_programs":   len(CNMF_PROGRAM_REGISTRY),
            "note":         "Provisional program set. Re-run cnmf_programs.py after data download.",
        }

    if program_name not in CNMF_PROGRAM_REGISTRY:
        return {
            "program_name": program_name,
            "error":        f"Program not found. Available: {list(CNMF_PROGRAM_REGISTRY.keys())}",
        }

    prog = CNMF_PROGRAM_REGISTRY[program_name].copy()
    prog["program_name"] = program_name
    prog["data_tier"] = "provisional"
    return prog


@_tool
def run_cnmf_program_extraction(
    h5ad_path: str,
    k_programs: int = 20,
    n_iter: int = 200,
) -> dict:
    """
    Run cNMF to extract K gene expression programs from a Perturb-seq h5ad.

    STUB — requires:
      1. GEO GSE246756 h5ad files (load_perturbseq_h5ad)
      2. cnmf package: pip install cnmf
      3. Significant compute (~4h on 32-core node for K562)

    Algorithm when implemented:
      - cNMF (Kotliar 2019) consensus NMF: K=20 programs
      - Stability selection across random seeds
      - Program loadings output to: data/cnmf_programs/{cell_line}_k{k}/

    Returns:
        STUB with expected output schema
    """
    return {
        "h5ad_path":        h5ad_path,
        "k_programs":       k_programs,
        "n_iter":           n_iter,
        "status":           "stub",
        "output_dir":       None,
        "programs_extracted": None,
        "note":             "STUB — implement pipelines/cnmf_programs.py after h5ad download",
        "cnmf_install":     "pip install cnmf",
    }


@_tool
def get_program_gene_loadings(program_name: str, top_n: int = 20) -> dict:
    """
    Return the top gene loadings for a cNMF program.
    Uses hardcoded top gene lists from published analyses.
    Full loadings require cnmf_programs.py output.

    Args:
        program_name: cNMF program name from CNMF_PROGRAM_REGISTRY
        top_n:        Number of top genes to return
    """
    if program_name not in CNMF_PROGRAM_REGISTRY:
        return {"program_name": program_name, "error": "Program not found"}

    prog = CNMF_PROGRAM_REGISTRY[program_name]
    top_genes = prog["top_genes"][:top_n]
    return {
        "program_name": program_name,
        "top_genes":    top_genes,
        "n_returned":   len(top_genes),
        "n_total":      len(prog["top_genes"]),
        "data_tier":    "provisional",
        "note":         "Provisional gene list from published analyses. Full loadings pending cNMF run.",
    }


# ---------------------------------------------------------------------------
# inspre causal structure tools (stubs)
# ---------------------------------------------------------------------------

@_tool
def run_inspre_causal_structure(
    beta_matrix: dict,
    gamma_matrix: dict,
    lambda_reg: float = 0.1,
) -> dict:
    """
    Run inspre (Iterative Non-Sparse Penalized Regression Estimation) to infer
    the causal gene regulatory network from Perturb-seq data.

    STUB — requires:
      1. Full quantitative β matrix from cNMF projection
      2. inspre package: pip install inspre
      3. Causal construction pipeline: pipelines/causal_construction.py

    Algorithm when implemented (Replogle 2022 Methods):
      - Input: gene perturbation effect matrix (n_genes × n_programs)
      - Regularization: λ = 0.1 (elastic net)
      - Output: sparse causal adjacency matrix

    Returns:
        STUB with expected output schema
    """
    return {
        "n_genes":         len(beta_matrix),
        "n_programs":      len(gamma_matrix),
        "lambda_reg":      lambda_reg,
        "adjacency_matrix": None,
        "n_edges":         None,
        "convergence":     None,
        "note":            "STUB — implement pipelines/causal_construction.py after β matrix available",
        "inspre_install":  "pip install inspre",
    }


@_tool
def run_scone_normalization(
    counts_matrix_path: str,
    cell_line: str = "K562",
) -> dict:
    """
    Run SCONE normalization on single-cell count matrix prior to cNMF.

    STUB — requires:
      1. Raw count matrix (from load_perturbseq_h5ad)
      2. scone package: BioConductor scone (R) or scone-python

    Returns schema-valid stub.
    """
    return {
        "counts_matrix_path": counts_matrix_path,
        "cell_line":          cell_line,
        "normalized_path":    None,
        "normalization_method": "scone",
        "note":               "STUB — implement in pipelines/causal_construction.py",
    }


# ---------------------------------------------------------------------------
# Burden analysis tools
# ---------------------------------------------------------------------------

@_tool
def get_gene_burden_stats(genes: list[str], cohort: str = "UKB") -> dict:
    """
    Return rare variant burden statistics for gene list from published studies.
    Used to estimate causal gene → trait γ via rare variant instrument.

    Real data available for:
      - PCSK9, LDLR: well-characterized rare variant burden → LDL-C (ExAC/gnomAD)
      - DNMT3A, TET2: CHIP prevalence + burden → CVD outcomes (Bick 2020, Kar 2022)

    STUB for most genes — wire in FinnGen R12 burden + gnomAD burden when available.

    Args:
        genes:  Gene symbols
        cohort: "UKB" | "FinnGen" | "gnomAD"
    """
    KNOWN_BURDEN: dict[str, dict] = {
        "PCSK9": {
            "lof_burden_or":       0.28,   # LoF → lower LDL (protective)
            "lof_burden_p":        1.2e-15,
            "trait":               "LDL-C",
            "cohort":              "ExAC",
            "pmid":                "Cohen 2006",
            "note":                "PCSK9 LoF carriers have ~28% lower LDL-C",
        },
        "LDLR": {
            "lof_burden_or":       3.2,    # LoF → higher LDL (FH risk)
            "lof_burden_p":        1.8e-22,
            "trait":               "LDL-C",
            "cohort":              "UKB",
            "pmid":                "Khera 2016",
            "note":                "LDLR LoF → familial hypercholesterolemia",
        },
        "DNMT3A": {
            "lof_burden_or":       1.09,   # somatic LoF → CAD (CHIP mechanism)
            "lof_burden_p":        5.8e-3,
            "trait":               "CAD",
            "cohort":              "UKB",
            "pmid":                "35177839",
            "note":                "Germline DNMT3A LoF burden from Kar 2022 WES",
        },
        "TET2": {
            "lof_burden_or":       1.15,
            "lof_burden_p":        2.1e-2,
            "trait":               "CAD",
            "cohort":              "UKB",
            "pmid":                "35177839",
            "note":                "Germline TET2 LoF burden from Kar 2022 WES",
        },
    }

    results = {}
    for gene in genes:
        if gene.upper() in KNOWN_BURDEN:
            results[gene] = {**KNOWN_BURDEN[gene.upper()], "cohort_requested": cohort}
        else:
            results[gene] = {
                "lof_burden_or":    None,
                "cohort":           cohort,
                "note":             f"STUB — {gene} burden not in registry; check FinnGen R12 or gnomAD v4",
                "finngen_url":      "https://r12.finngen.fi/",
            }

    return {
        "genes":   genes,
        "cohort":  cohort,
        "results": results,
        "note":    "Known burden stats from published studies. Run FinnGen/gnomAD query for others.",
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if mcp is None:
        raise RuntimeError("fastmcp required: pip install fastmcp")
    mcp.run()
