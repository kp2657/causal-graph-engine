"""
perturbseq_server.py — MCP server for Perturb-seq CRISPR perturbation signatures.

Provides disease-context-aware perturbation signatures as a higher-quality
replacement / complement to LINCS L1000 for Tier 3 β estimation.

Architecture
------------
Runtime (this file):
  - Loads pre-computed compact signatures from data/perturbseq/{dataset_id}/signatures.json.gz
  - Selects best dataset for disease via _DISEASE_DATASET_PRIORITY registry
  - Returns gene-level log2FC signatures (same interface as lincs_server)

Preprocessing (offline, one-time per dataset):
  - python -m mcp_servers.perturbseq_server preprocess <dataset_id> <h5ad_path>
  - Computes mean log2FC per perturbed gene vs NT controls
  - Saves top-200 DE genes per perturbation as compressed JSON (~5–40 MB)

Dataset coverage
----------------
  Dataset                  Cell line    Disease context       #Genes  Access
  replogle_2022_k562       K562         generic (CML)         9,866   figshare DOI 10.6084/m9.figshare.20029387
  replogle_2022_rpe1       RPE1         generic (epithelial)  2,393   same figshare
  papalexi_2021_thp1       THP-1        IBD / RA / inflammation  49   GEO GSE153056
  dixit_2016_bmdc          BMDC         IBD / RA / LPS           24   GEO GSE90063
  frangieh_2021_a375       A375         melanoma / SLE / RA     750   Broad SCP1064
  norman_2019_k562         K562         generic (CRISPRa)       105   GEO GSE133344
  natsume_2023_haec        HAEC         CAD (endothelial)     2,285   PLOS Gen doi:10.1371/journal.pgen.1010680

Run standalone:  python mcp_servers/perturbseq_server.py
"""
from __future__ import annotations

import gzip
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import numpy as np

try:
    import fastmcp
    mcp = fastmcp.FastMCP("perturbseq-server")
    _tool = mcp.tool()
except ImportError:
    def _tool(fn=None, **_):
        return fn if fn is not None else (lambda f: f)
    mcp = None

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_CACHE_DIR = Path(__file__).parent.parent / "data" / "perturbseq"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

_DATASET_REGISTRY: dict[str, dict] = {
    # ── Replogle 2022 ────────────────────────────────────────────────────────
    # Using *bulk* h5ads (rows=perturbations, already pseudobulk-aggregated).
    # Bulk files are ~0.1–0.4 GB vs 8–66 GB for single-cell versions.
    # Bulk format: obs.index = "{id}_{gene}_{guide}_{ensembl}" or "..._non-targeting_..."
    #              var.index = Ensembl IDs; var['gene_name'] = gene symbols
    #              X = normalized log expression (dense, may contain inf → handled)
    "replogle_2022_k562": {
        "name":              "Replogle 2022 genome-wide CRISPRi (K562, bulk pseudobulk)",
        "citation":          "Replogle et al., Cell 2022. doi:10.1016/j.cell.2022.05.013",
        "cell_line":         "K562",
        "tissue":            "blood_CML",
        "disease_context":   "generic",
        "perturbation_type": "CRISPRi",
        "n_genes_perturbed": 9866,
        "access": {
            "type":            "figshare_bulk_h5ad",
            "doi":             "10.6084/m9.figshare.20029387",
            "figshare_file_id": 35773217,
            "file":            "K562_gwps_normalized_bulk_01.h5ad",
            "size_gb":         0.37,
        },
        "h5ad_format":               "replogle_bulk",
        "h5ad_obs_perturbation_col": "gene",   # parsed from obs.index
        "h5ad_nt_label":             "non-targeting",
        "geo":                        "PRJNA831566",
        "note": "Largest genome-wide screen; best coverage; K562 is CML so tissue mismatch for most diseases.",
    },
    "replogle_2022_rpe1": {
        "name":              "Replogle 2022 essential-gene CRISPRi (RPE1, bulk pseudobulk)",
        "citation":          "Replogle et al., Cell 2022. doi:10.1016/j.cell.2022.05.013",
        "cell_line":         "RPE1",
        "tissue":            "retinal_epithelium",
        "disease_context":   "generic",
        "perturbation_type": "CRISPRi",
        "n_genes_perturbed": 2393,
        "access": {
            "type":            "figshare_bulk_h5ad",
            "doi":             "10.6084/m9.figshare.20029387",
            "figshare_file_id": 35775512,
            "file":            "rpe1_normalized_bulk_01.h5ad",
            "size_gb":         0.10,
        },
        "h5ad_format":               "replogle_bulk",
        "h5ad_obs_perturbation_col": "gene",
        "h5ad_nt_label":             "non-targeting",
        "geo":                        "PRJNA831566",
        "note": "Non-cancerous cell line; near-diploid. Essential gene coverage only (~2400 genes).",
    },

    # ── Papalexi 2021 ────────────────────────────────────────────────────────
    "papalexi_2021_thp1": {
        "name":              "Papalexi 2021 checkpoint screen (THP-1 monocytes)",
        "citation":          "Papalexi et al., Nature Genetics 2021. doi:10.1038/s41588-021-00778-2",
        "cell_line":         "THP-1",
        "tissue":            "monocyte",
        "disease_context":   "IBD_RA_inflammation",
        "perturbation_type": "CRISPRko",
        "n_genes_perturbed": 49,
        "access": {
            "type":    "geo_h5ad",
            "geo":     "GSE153056",
            "figshare_doi": "10.35092/yhjc.c.5303193",
        },
        "h5ad_obs_perturbation_col": "perturbation",
        "h5ad_nt_label":             "NT",
        "note": (
            "Monocytic cell line — most relevant for IBD macrophage / RA inflammatory context. "
            "Targeted screen (49 genes) around PD-L1/KEAP1/NRF2/CD58 axis."
        ),
    },

    # ── Dixit 2016 ───────────────────────────────────────────────────────────
    "dixit_2016_bmdc": {
        "name":              "Dixit 2016 TF screen in bone marrow dendritic cells (LPS stimulation)",
        "citation":          "Dixit et al., Cell 2016. doi:10.1016/j.cell.2016.11.038",
        "cell_line":         "BMDC",
        "tissue":            "dendritic_cell",
        "disease_context":   "IBD_RA_inflammation",
        "perturbation_type": "CRISPRko",
        "n_genes_perturbed": 24,
        "access": {
            "type": "geo_matrix",
            "geo":  "GSE90063",
        },
        "h5ad_obs_perturbation_col": "perturbation",
        "h5ad_nt_label":             "CTRL",
        "note": (
            "Primary-like dendritic cells under LPS stimulation — captures inflammatory "
            "response. TF-focused (24 genes). Important for IBD/RA innate immune pathway."
        ),
    },

    # ── Frangieh 2021 ────────────────────────────────────────────────────────
    "frangieh_2021_a375": {
        "name":              "Frangieh 2021 melanoma immune evasion CRISPR screen",
        "citation":          "Frangieh et al., Nature Genetics 2021. doi:10.1038/s41588-021-00779-1",
        "cell_line":         "A375",
        "tissue":            "melanoma",
        "disease_context":   "SLE_RA_cancer",
        "perturbation_type": "CRISPRko",
        "n_genes_perturbed": 750,
        "access": {
            "type":   "broad_scp",
            "scp_id": "SCP1064",
        },
        "h5ad_obs_perturbation_col": "perturbation",
        "h5ad_nt_label":             "NT",
        "note": (
            "CITE-seq (RNA + protein). Melanoma + TIL co-culture; captures IFN-gamma "
            "pathway and immune checkpoint biology. Useful for SLE/RA immune programs."
        ),
    },

    # ── Norman 2019 ──────────────────────────────────────────────────────────
    "norman_2019_k562": {
        "name":              "Norman 2019 combinatorial CRISPRa (K562)",
        "citation":          "Norman et al., Science 2019. doi:10.1126/science.aax4438",
        "cell_line":         "K562",
        "tissue":            "blood_CML",
        "disease_context":   "generic",
        "perturbation_type": "CRISPRa",
        "n_genes_perturbed": 105,
        "access": {
            "type": "geo_matrix",
            "geo":  "GSE133344",
        },
        "h5ad_obs_perturbation_col": "perturbation",
        "h5ad_nt_label":             "ctrl",
        "note": (
            "Activation screen (CRISPRa = overexpression). Useful when genes are "
            "loss-of-function disease-linked and gain-of-function β is needed."
        ),
    },

    # ── Ursu 2022 iPSC Neurons ───────────────────────────────────────────────
    "ursu_2022_ipsc_neuron": {
        "name":              "Ursu 2022 CRISPRi iPSC-derived neurons (NeurIPS 2022 benchmark)",
        "citation":          "Ursu et al., NeurIPS 2022. GEO: GSE196862.",
        "cell_line":         "iPSC_neuron",
        "tissue":            "neuron_iPSC",
        "disease_context":   "AD_SCZ_neurodegeneration",
        "perturbation_type": "CRISPRi",
        "n_genes_perturbed": 96,
        "access": {
            "type":    "zenodo_harmonized",
            "zenodo":  "7041690",
            "scperturb_id": "Ursu2022_neurips",
        },
        "h5ad_obs_perturbation_col": "perturbation",
        "h5ad_nt_label":             "control",
        "geo":                        "GSE196862",
        "note": (
            "Only human iPSC neuron Perturb-seq in the scPerturb harmonized collection. "
            "96 genes only — limited genome coverage; use with eQTL-MR to fill gaps. "
            "Cell type is directly relevant for AD/neurodegenerative disease programs."
        ),
    },

    # ── Schnitzler comprehensive CAD Perturb-seq ─────────────────────────────
    "schnitzler_cad_vascular": {
        "name":              "Schnitzler comprehensive Perturb-seq of 332 CAD risk genes in vascular cells",
        "citation":          "Schnitzler et al. GEO: GSE210681.",
        "cell_line":         "HCASMC_HAEC",
        "tissue":            "vascular_smooth_muscle_endothelium",
        "disease_context":   "CAD",
        "perturbation_type": "CRISPRi",
        "n_genes_perturbed": 332,
        "access": {
            "type": "log2fc_matrix",
            "geo":  "GSE210681",
            "log2fc_file": "GSE210681_ALL_log2fcs_dup4_s4n3.99x.txt.gz",
            "ftp_url": "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE210nnn/GSE210681/suppl/GSE210681_ALL_log2fcs_dup4_s4n3.99x.txt.gz",
            "note": (
                "Pre-computed log2FC matrix (rows=genes, cols=perturbations). "
                "Preprocess: python -m mcp_servers.perturbseq_server preprocess "
                "schnitzler_cad_vascular data/perturbseq/schnitzler_cad_vascular/GSE210681_ALL_log2fcs.txt.gz"
            ),
        },
        "h5ad_obs_perturbation_col": "perturbation",
        "h5ad_nt_label":             "non-targeting",
        "geo":                        "GSE210681",
        "note": (
            "Highest-priority CAD Perturb-seq. 332 CAD GWAS risk genes perturbed in "
            "human coronary artery smooth muscle cells (HCASMC) and/or endothelial cells — "
            "the disease-relevant vascular cell types. Signatures must be preprocessed from "
            "GEO h5ad before use (run preprocess CLI). "
            "Supersedes K562 (myeloid proxy) for all CAD vascular biology."
        ),
    },

    # ── Natsume 2023 HAEC ────────────────────────────────────────────────────
    "natsume_2023_haec": {
        "name":              "Natsume 2023 CAD GWAS loci in human aortic endothelial cells",
        "citation":          "Natsume et al., PLOS Genetics 2023. doi:10.1371/journal.pgen.1010680",
        "cell_line":         "HAEC",
        "tissue":            "aortic_endothelium",
        "disease_context":   "CAD",
        "perturbation_type": "CRISPRko_CRISPRi_CRISPRa",
        "n_genes_perturbed": 2285,
        "access": {
            "type": "geo_matrix",
            "geo":  None,   # GEO accession not confirmed; check paper supplementary
            "doi":  "10.1371/journal.pgen.1010680",
            "note": "Verify GEO accession from paper supplementary materials.",
        },
        "h5ad_obs_perturbation_col": "perturbation",
        "h5ad_nt_label":             "NT",
        "note": (
            "2,285 CAD GWAS loci × 3 perturbation modes in primary-like endothelial cells. "
            "GEO accession not yet confirmed. Use schnitzler_cad_vascular (GSE210681) as primary."
        ),
    },
}

# ---------------------------------------------------------------------------
# Disease → dataset priority order
# First available (cached) dataset in the list is used.
# ---------------------------------------------------------------------------

_DISEASE_DATASET_PRIORITY: dict[str, list[str]] = {
    # CAD: Schnitzler 332 CAD-specific vascular > Natsume HAEC > genome-wide generic
    "CAD":  ["schnitzler_cad_vascular", "natsume_2023_haec", "replogle_2022_k562"],
    # IBD: monocyte/DC (tissue match) > genome-wide
    "IBD":  ["papalexi_2021_thp1",  "dixit_2016_bmdc",    "replogle_2022_k562"],
    # RA: monocyte/melanoma-immune > genome-wide
    "RA":   ["papalexi_2021_thp1",  "frangieh_2021_a375", "replogle_2022_k562"],
    # T2D: no liver/pancreas Perturb-seq available yet; genome-wide as best proxy
    "T2D":  ["replogle_2022_k562",  "replogle_2022_rpe1"],
    # AD: iPSC neurons (cell type match, 96 genes) > RPE1 (non-cancerous) > K562
    "AD":   ["ursu_2022_ipsc_neuron", "replogle_2022_rpe1", "replogle_2022_k562"],
    # SLE: immune-context first
    "SLE":  ["papalexi_2021_thp1",  "frangieh_2021_a375", "replogle_2022_k562"],
    # AMD: RPE1 (retinal epithelial = tissue match) > K562 (generic fallback)
    "AMD":  ["replogle_2022_rpe1",  "replogle_2022_k562"],
    # Generic: largest genome-wide screen first
    "GENERIC": ["replogle_2022_k562", "replogle_2022_rpe1"],
}

# In-process signature cache: dataset_id → {gene: {downstream_gene: log2fc}}
_SIG_CACHE: dict[str, dict[str, dict[str, float]]] = {}

# ---------------------------------------------------------------------------
# Signature cache I/O
# ---------------------------------------------------------------------------

def _sig_path(dataset_id: str) -> Path:
    return _CACHE_DIR / dataset_id / "signatures.json.gz"


def _load_cached_signatures(dataset_id: str) -> dict[str, dict[str, float]] | None:
    """Load pre-computed signatures from disk. Returns None if not yet preprocessed."""
    if dataset_id in _SIG_CACHE:
        return _SIG_CACHE[dataset_id]
    p = _sig_path(dataset_id)
    if not p.exists():
        return None
    with gzip.open(p, "rt", encoding="utf-8") as f:
        data = json.load(f)
    _SIG_CACHE[dataset_id] = data
    return data


def _save_signatures(dataset_id: str, signatures: dict[str, dict[str, float]]) -> None:
    """Save pre-computed signatures to disk."""
    p = _sig_path(dataset_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(p, "wt", encoding="utf-8") as f:
        json.dump(signatures, f)
    _SIG_CACHE[dataset_id] = signatures
    log.info("Saved %d gene signatures for %s → %s", len(signatures), dataset_id, p)


# ---------------------------------------------------------------------------
# h5ad → signatures preprocessing (offline, called once per dataset)
# ---------------------------------------------------------------------------

def preprocess_h5ad(
    dataset_id: str,
    h5ad_path: str | Path,
    top_k: int = 200,
    min_abs_log2fc: float = 0.3,
) -> dict:
    """
    Preprocess a Perturb-seq h5ad into compact gene signatures.

    Computes mean log2FC per perturbed gene vs NT controls, keeps top_k
    downstream genes by |log2FC| where |log2FC| ≥ min_abs_log2fc.

    Args:
        dataset_id:      Registry key (e.g. "replogle_2022_k562")
        h5ad_path:       Path to local h5ad file
        top_k:           Max downstream genes to store per perturbation
        min_abs_log2fc:  Minimum |log2FC| threshold

    Returns:
        {"dataset_id", "n_perturbed_genes", "n_cached_genes", "sig_path"}
    """
    import scanpy as sc
    import scipy.sparse as sp

    meta = _DATASET_REGISTRY.get(dataset_id)
    if meta is None:
        return {"error": f"Unknown dataset_id: {dataset_id}"}

    pert_col = meta["h5ad_obs_perturbation_col"]
    nt_label = meta["h5ad_nt_label"]

    log.info("Loading h5ad: %s", h5ad_path)
    adata = sc.read_h5ad(str(h5ad_path))

    if pert_col not in adata.obs.columns:
        return {"error": f"Column '{pert_col}' not found in obs. Available: {list(adata.obs.columns)}"}

    # Normalize to log1p if not already (detect by value range)
    X = adata.X
    if sp.issparse(X):
        sample_max = X[:100].max()
    else:
        sample_max = X[:100].max()
    if float(sample_max) > 20:
        log.info("Counts appear raw (max=%.1f); applying log1p normalization", float(sample_max))
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        X = adata.X

    # NT control mask
    pert_labels = adata.obs[pert_col].astype(str)
    nt_mask = (pert_labels == nt_label).values
    if nt_mask.sum() < 10:
        return {"error": f"Fewer than 10 NT control cells found (label='{nt_label}')"}

    nt_idx = np.where(nt_mask)[0]
    mean_nt = np.asarray(X[nt_idx].mean(axis=0)).flatten().astype(float)

    # Compute per-perturbation mean log2FC
    unique_perts = [p for p in pert_labels.unique() if p != nt_label]
    signatures: dict[str, dict[str, float]] = {}
    gene_names = list(adata.var_names)

    for pert in unique_perts:
        pert_idx = np.where(pert_labels == pert)[0]
        if len(pert_idx) < 3:
            continue  # skip poorly sampled perturbations

        mean_pert = np.asarray(X[pert_idx].mean(axis=0)).flatten().astype(float)
        # log2FC = log2( (exp(mean_pert_log1p) ) / (exp(mean_nt_log1p)) )
        # Equivalently: (mean_pert - mean_nt) / log(2) when values are ln(x+1)
        log2fc = (mean_pert - mean_nt) / np.log(2)

        abs_lfc = np.abs(log2fc)
        # Filter by threshold + take top_k
        candidates = np.where(abs_lfc >= min_abs_log2fc)[0]
        if len(candidates) > top_k:
            candidates = candidates[np.argsort(abs_lfc[candidates])[-top_k:]]

        if len(candidates) == 0:
            continue

        sig = {gene_names[i]: round(float(log2fc[i]), 4) for i in candidates}
        signatures[pert] = sig

    _save_signatures(dataset_id, signatures)
    return {
        "dataset_id":         dataset_id,
        "n_perturbed_genes":  len(unique_perts),
        "n_cached_genes":     len(signatures),
        "sig_path":           str(_sig_path(dataset_id)),
    }


def preprocess_replogle_bulk(
    dataset_id: str,
    h5ad_path: str | Path,
    top_k: int = 200,
    min_abs_log2fc: float = 0.3,
) -> dict:
    """
    Preprocess a Replogle 2022 *bulk* pseudobulk h5ad.

    Bulk format (rows = perturbations, already aggregated):
      obs.index  = "{id}_{gene}_{guide}_{ensembl}" | "..._non-targeting_..."
      var.index  = Ensembl IDs
      var['gene_name'] = gene symbols
      X          = normalized log expression (dense, may contain inf)

    Computes log2FC per perturbed gene vs mean of NT-control rows.
    Maps Ensembl column IDs → gene symbols via var['gene_name'].
    """
    import anndata as ad

    meta = _DATASET_REGISTRY.get(dataset_id)
    if meta is None:
        return {"error": f"Unknown dataset_id: {dataset_id}"}

    log.info("Loading Replogle bulk h5ad: %s", h5ad_path)
    adata = ad.read_h5ad(str(h5ad_path))

    # Parse perturbation gene from obs.index
    # Format: "{id}_{gene}_{guide}_{ensembl}" — split from right to handle gene names with _
    def _parse_gene(idx: str) -> str:
        parts = idx.split("_")
        if len(parts) < 4:
            return idx
        # last part = ensembl ID or "non-targeting"
        # second-to-last = guide or "non-targeting"
        # everything between first part and last two = gene name
        return "_".join(parts[1:-2])

    pert_genes = [_parse_gene(i) for i in adata.obs.index]
    nt_label   = meta["h5ad_nt_label"]   # "non-targeting"
    nt_mask    = np.array([g == nt_label for g in pert_genes])

    if nt_mask.sum() < 3:
        return {"error": f"Fewer than 3 NT rows found (label='{nt_label}')"}

    # Build Ensembl → gene symbol map
    if "gene_name" in adata.var.columns:
        ensembl_to_sym = dict(zip(adata.var.index, adata.var["gene_name"]))
    else:
        ensembl_to_sym = {e: e for e in adata.var.index}
    downstream_names = [ensembl_to_sym.get(e, e) for e in adata.var.index]

    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = X.astype(float)

    # Replace inf with column max finite value (or 0)
    inf_mask = ~np.isfinite(X)
    if inf_mask.any():
        col_max = np.nanmax(np.where(np.isfinite(X), X, np.nan), axis=0)
        col_max = np.where(np.isfinite(col_max), col_max, 0.0)
        X[inf_mask] = np.take(col_max, np.where(inf_mask)[1])

    nt_idx   = np.where(nt_mask)[0]
    mean_nt  = X[nt_idx].mean(axis=0)

    # Each non-NT row is already one perturbation's aggregate
    unique_perts = sorted({g for g, is_nt in zip(pert_genes, nt_mask) if not is_nt})
    signatures: dict[str, dict[str, float]] = {}

    for pert in unique_perts:
        rows = np.where(np.array(pert_genes) == pert)[0]
        mean_pert = X[rows].mean(axis=0)
        log2fc    = (mean_pert - mean_nt) / np.log(2)

        abs_lfc    = np.abs(log2fc)
        candidates = np.where(abs_lfc >= min_abs_log2fc)[0]
        if len(candidates) > top_k:
            candidates = candidates[np.argsort(abs_lfc[candidates])[-top_k:]]
        if len(candidates) == 0:
            continue

        signatures[pert] = {downstream_names[i]: round(float(log2fc[i]), 4) for i in candidates}

    _save_signatures(dataset_id, signatures)
    log.info("%s: %d genes → %d with signatures", dataset_id, len(unique_perts), len(signatures))
    return {
        "dataset_id":        dataset_id,
        "n_perturbed_genes": len(unique_perts),
        "n_cached_genes":    len(signatures),
        "sig_path":          str(_sig_path(dataset_id)),
    }


def preprocess_log2fc_matrix(
    dataset_id: str,
    log2fc_path: str | Path,
    top_k: int = 200,
    min_abs_log2fc: float = 0.3,
) -> dict:
    """
    Preprocess a pre-computed log2FC matrix (e.g. Schnitzler GSE210681).

    Expected file format (tab-separated, gzipped):
      Row 0 (header): '' | 'genes' | PERT1 | PERT2 | ...
      Row 1+:         row_idx | 'SYMBOL:ENSEMBL' | log2fc_1 | log2fc_2 | ...

    Matrix is (downstream_genes × perturbations). This function transposes it
    to produce per-perturbation signatures. Multiple guides per gene (column
    names ending in -N for integer N) are averaged before top_k selection.
    """
    import gzip as _gz
    import re

    meta = _DATASET_REGISTRY.get(dataset_id)
    if meta is None:
        return {"error": f"Unknown dataset_id: {dataset_id}"}

    log.info("Loading log2FC matrix: %s", log2fc_path)
    path = Path(log2fc_path)
    opener = _gz.open(path, "rt") if str(path).endswith(".gz") else open(path, "r")

    with opener as fh:
        header = fh.readline().rstrip("\n").split("\t")
        pert_cols = header[2:]  # skip row_idx and 'genes' cols

        # Map column index → canonical gene name (strip trailing -N guide index)
        def _canonical(name: str) -> str:
            return re.sub(r"-\d+$", "", name)

        canonical_perts = [_canonical(p) for p in pert_cols]
        n_perts = len(pert_cols)

        # Accumulate: {canonical_gene: {downstream_gene: [log2fc values]}}
        acc: dict[str, dict[str, list[float]]] = {}

        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < n_perts + 2:
                continue
            gene_field = parts[1]  # "SYMBOL:ENSEMBL" or just "SYMBOL"
            gene_sym = gene_field.split(":")[0]
            if not gene_sym:
                continue

            for i, val_str in enumerate(parts[2:2 + n_perts]):
                try:
                    val = float(val_str)
                except ValueError:
                    continue
                if not (val == val) or abs(val) < min_abs_log2fc:  # nan check + threshold
                    continue
                cg = canonical_perts[i]
                if cg not in acc:
                    acc[cg] = {}
                if gene_sym not in acc[cg]:
                    acc[cg][gene_sym] = []
                acc[cg][gene_sym].append(val)

    log.info("Parsed %d perturbation targets; averaging multi-guide log2FC", len(acc))

    # Average multi-guide values, apply top_k
    signatures: dict[str, dict[str, float]] = {}
    for pert_gene, downstream in acc.items():
        averaged = {g: float(np.mean(vs)) for g, vs in downstream.items()}
        # Sort by |log2fc| descending, keep top_k
        top = sorted(averaged.items(), key=lambda x: abs(x[1]), reverse=True)[:top_k]
        if top:
            signatures[pert_gene] = dict(top)

    sig_path = _sig_path(dataset_id)
    sig_path.parent.mkdir(parents=True, exist_ok=True)
    with _gz.open(sig_path, "wt") as out:
        json.dump(signatures, out)

    log.info("Saved %d signatures to %s", len(signatures), sig_path)
    return {
        "dataset_id":        dataset_id,
        "n_perturbed_genes": len(signatures),
        "n_cached_genes":    len(signatures),
        "sig_path":          str(sig_path),
    }


def download_h5ad(dataset_id: str, dest_dir: str | Path | None = None) -> dict:
    """
    Download the h5ad for a dataset to dest_dir (default: data/perturbseq/{dataset_id}/).

    Supports:
      - figshare_bulk_h5ad: direct ndownloader URL from figshare_file_id
      - figshare_h5ad: same

    Returns {"path": str, "size_mb": float} on success, {"error": str} on failure.
    """
    import httpx

    meta = _DATASET_REGISTRY.get(dataset_id)
    if meta is None:
        return {"error": f"Unknown dataset_id: {dataset_id}"}

    access   = meta["access"]
    acc_type = access.get("type", "")

    dest = Path(dest_dir) if dest_dir else _CACHE_DIR / dataset_id
    dest.mkdir(parents=True, exist_ok=True)

    if acc_type in ("figshare_bulk_h5ad", "figshare_h5ad"):
        file_id  = access.get("figshare_file_id")
        filename = access.get("file", f"{dataset_id}.h5ad")
        if not file_id:
            return {"error": "figshare_file_id not set in registry"}
        url = f"https://ndownloader.figshare.com/files/{file_id}"
    else:
        return {"error": f"Automatic download not supported for access type '{acc_type}'"}

    out_path = dest / filename
    if out_path.exists():
        log.info("Already downloaded: %s", out_path)
        return {"path": str(out_path), "size_mb": out_path.stat().st_size / 1e6, "cached": True}

    log.info("Downloading %s → %s", url, out_path)
    try:
        with httpx.stream("GET", url, follow_redirects=True, timeout=600) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            done  = 0
            with open(out_path, "wb") as f:
                for chunk in r.iter_bytes(chunk_size=4 * 1024 * 1024):
                    f.write(chunk)
                    done += len(chunk)
                    if total:
                        pct = done / total * 100
                        print(f"\r  {filename}: {done/1e6:.0f}/{total/1e6:.0f} MB ({pct:.0f}%)", end="", flush=True)
            print()
    except Exception as exc:
        out_path.unlink(missing_ok=True)
        return {"error": str(exc)}

    return {"path": str(out_path), "size_mb": out_path.stat().st_size / 1e6, "cached": False}


def activate_dataset(
    dataset_id: str,
    top_k: int = 200,
    min_abs_log2fc: float = 0.3,
    delete_raw: bool = True,
) -> dict:
    """
    Download + preprocess a dataset in one step.

    Args:
        dataset_id:      Registry key
        top_k:           Max downstream genes per perturbation signature
        min_abs_log2fc:  Minimum |log2FC| threshold
        delete_raw:      Delete the downloaded h5ad after preprocessing (saves disk)

    Returns preprocessing result dict.
    """
    meta = _DATASET_REGISTRY.get(dataset_id)
    if meta is None:
        return {"error": f"Unknown dataset_id: {dataset_id}"}

    if _sig_path(dataset_id).exists():
        log.info("%s: already activated (signatures.json.gz exists)", dataset_id)
        sigs = _load_cached_signatures(dataset_id)
        return {
            "dataset_id":    dataset_id,
            "status":        "already_activated",
            "n_cached_genes": len(sigs) if sigs else 0,
            "sig_path":      str(_sig_path(dataset_id)),
        }

    dl = download_h5ad(dataset_id)
    if "error" in dl:
        return {"dataset_id": dataset_id, "error": dl["error"], "step": "download"}

    h5ad_path = dl["path"]
    h5ad_fmt  = meta.get("h5ad_format", "single_cell")

    if h5ad_fmt == "replogle_bulk":
        result = preprocess_replogle_bulk(dataset_id, h5ad_path, top_k, min_abs_log2fc)
    else:
        result = preprocess_h5ad(dataset_id, h5ad_path, top_k, min_abs_log2fc)

    if delete_raw and "error" not in result:
        Path(h5ad_path).unlink(missing_ok=True)
        log.info("Deleted raw h5ad: %s", h5ad_path)
        result["raw_deleted"] = True

    return result


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def _select_dataset(disease_context: str | None, dataset_id: str | None) -> str | None:
    """Return the first cached dataset in priority order for the disease."""
    if dataset_id:
        return dataset_id

    dk = (disease_context or "GENERIC").upper()
    priority = _DISEASE_DATASET_PRIORITY.get(dk, _DISEASE_DATASET_PRIORITY["GENERIC"])

    for ds_id in priority:
        if _sig_path(ds_id).exists():
            return ds_id

    # Nothing cached — return highest-priority dataset ID anyway (caller handles missing)
    return priority[0] if priority else None


def _compute_program_beta(
    signature: dict[str, float],
    program_gene_set: list[str] | set[str],
) -> float | None:
    """β = mean log2FC of program genes present in signature. None if < 5% coverage."""
    pg_set = set(program_gene_set)
    hits = {g: signature[g] for g in pg_set if g in signature}
    if not hits:
        return None
    coverage = len(hits) / max(len(pg_set), 1)
    if coverage < 0.05:
        return None
    return sum(hits.values()) / len(pg_set)


# ---------------------------------------------------------------------------
# MCP tools
# ---------------------------------------------------------------------------

@_tool
def get_perturbseq_signature(
    gene_symbol: str,
    disease_context: str | None = None,
    dataset_id: str | None = None,
    top_k: int = 100,
) -> dict:
    """
    Retrieve a Perturb-seq perturbation signature for a gene.

    Returns the transcriptional response (log2FC) to CRISPR perturbation of
    the gene, using the most disease-relevant **cached** dataset.

    Dataset selection is **fallback-only** (no merging of signatures across
    datasets): `_select_dataset()` walks `_DISEASE_DATASET_PRIORITY` and picks
    the first dataset whose `signatures.json.gz` cache exists. If the perturbed
    gene is absent from that cache, the implementation may try the next
    priority datasets until one contains the gene — still returning a **single**
    signature from **one** dataset.

    Priority order (must stay in sync with `_DISEASE_DATASET_PRIORITY`):
      CAD  → schnitzler_cad_vascular → natsume_2023_haec → replogle_2022_k562
      IBD  → papalexi_2021_thp1 → dixit_2016_bmdc → replogle_2022_k562
      RA   → papalexi_2021_thp1 → frangieh_2021_a375 → replogle_2022_k562
      T2D  → replogle_2022_k562 → replogle_2022_rpe1
      AD   → ursu_2022_ipsc_neuron → replogle_2022_rpe1 → replogle_2022_k562
      SLE  → papalexi_2021_thp1 → frangieh_2021_a375 → replogle_2022_k562
      AMD  → replogle_2022_rpe1 → replogle_2022_k562
      else → replogle_2022_k562 → replogle_2022_rpe1  (GENERIC)

    Args:
        gene_symbol:     Gene to query (e.g. "PCSK9")
        disease_context: Disease key — "CAD" | "IBD" | "RA" | "T2D" | "AD" | "SLE" | "AMD"
        dataset_id:      Override dataset (see list_perturbseq_datasets())
        top_k:           Max downstream genes returned in signature

    Returns:
        {
          "gene":             str,
          "dataset_id":       str,
          "cell_line":        str,
          "disease_context":  str,
          "signature":        {gene_symbol: log2fc},
          "n_genes_measured": int,
          "evidence_tier":    "Tier3_Provisional",
          "perturbation_type": str,
          "source":           str,
          "note":             str,   # if data not yet preprocessed
        }
    """
    gene = gene_symbol.upper()

    ds_id = _select_dataset(disease_context, dataset_id)
    if ds_id is None:
        return {
            "gene": gene, "signature": {}, "n_genes_measured": 0,
            "evidence_tier": "Tier3_Provisional", "source": "not_found",
            "note": "No Perturb-seq dataset configured.",
        }

    meta  = _DATASET_REGISTRY[ds_id]
    sigs  = _load_cached_signatures(ds_id)

    if sigs is None:
        return {
            "gene":            gene,
            "dataset_id":      ds_id,
            "cell_line":       meta["cell_line"],
            "disease_context": disease_context,
            "signature":       {},
            "n_genes_measured": 0,
            "evidence_tier":   "Tier3_Provisional",
            "source":          "not_cached",
            "note": (
                f"Dataset '{ds_id}' not yet preprocessed. "
                f"Download h5ad ({meta['access'].get('size_gb', '?')} GB) and run: "
                f"preprocess_h5ad('{ds_id}', '<h5ad_path>'). "
                f"Access: {meta['access']}"
            ),
        }

    # Try exact match, then case-insensitive
    sig = sigs.get(gene) or sigs.get(gene.upper()) or sigs.get(gene.lower())

    if sig is None:
        # Try the dataset's next fallback in priority list
        dk = (disease_context or "GENERIC").upper()
        priority = _DISEASE_DATASET_PRIORITY.get(dk, _DISEASE_DATASET_PRIORITY["GENERIC"])
        for fallback_id in priority:
            if fallback_id == ds_id:
                continue
            fb_sigs = _load_cached_signatures(fallback_id)
            if fb_sigs and (fb_sigs.get(gene) or fb_sigs.get(gene.upper())):
                ds_id = fallback_id
                meta  = _DATASET_REGISTRY[fallback_id]
                sigs  = fb_sigs
                sig   = fb_sigs.get(gene) or fb_sigs.get(gene.upper())
                break

    if sig is None:
        return {
            "gene":            gene,
            "dataset_id":      ds_id,
            "cell_line":       meta["cell_line"],
            "disease_context": disease_context,
            "signature":       {},
            "n_genes_measured": 0,
            "evidence_tier":   "Tier3_Provisional",
            "source":          "gene_not_found",
            "note": f"Gene '{gene}' not found in {ds_id} (perturbed gene set may not include it).",
        }

    # Trim to top_k by |log2FC|
    if len(sig) > top_k:
        sig = dict(sorted(sig.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_k])

    return {
        "gene":             gene,
        "dataset_id":       ds_id,
        "cell_line":        meta["cell_line"],
        "tissue":           meta["tissue"],
        "disease_context":  disease_context,
        "signature":        sig,
        "n_genes_measured": len(sig),
        "evidence_tier":    "Tier3_Provisional",
        "perturbation_type": meta["perturbation_type"],
        "source":           f"perturbseq_{ds_id}",
        "citation":         meta["citation"],
        "note":             meta.get("note", ""),
    }


@_tool
def compute_perturbseq_program_beta(
    gene_symbol: str,
    program_gene_set: list[str],
    disease_context: str | None = None,
    dataset_id: str | None = None,
) -> dict:
    """
    Compute Tier 3 β_{gene→program} from Perturb-seq CRISPR perturbation signature.

    β = mean log2FC of program genes in gene's CRISPR perturbation signature.
    Uses the most disease-relevant available dataset (see get_perturbseq_signature).

    Args:
        gene_symbol:      Gene being perturbed
        program_gene_set: Genes defining the biological program (NMF program or pathway)
        disease_context:  Disease key for dataset selection
        dataset_id:       Override specific dataset

    Returns:
        {
          "gene":             str,
          "program_coverage": float,   # fraction of program genes measured
          "beta":             float | None,
          "evidence_tier":    "Tier3_Provisional",
          "cell_line":        str,
          "dataset_id":       str,
          "perturbation_type": str,
          "data_source":      str,
          "note":             str,
        }
    """
    sig_result = get_perturbseq_signature(gene_symbol, disease_context, dataset_id)
    sig        = sig_result.get("signature", {})
    actual_cl  = sig_result.get("cell_line", "unknown")
    ds_id_used = sig_result.get("dataset_id", "unknown")
    pg_set     = set(program_gene_set)
    hits       = {g: sig[g] for g in pg_set if g in sig}
    coverage   = len(hits) / max(len(pg_set), 1)
    beta       = _compute_program_beta(sig, pg_set)

    source_tag = sig_result.get("source", "")
    return {
        "gene":             gene_symbol,
        "program_coverage": round(coverage, 3),
        "beta":             round(beta, 4) if beta is not None else None,
        "evidence_tier":    "Tier3_Provisional",
        "cell_line":        actual_cl,
        "dataset_id":       ds_id_used,
        "perturbation_type": sig_result.get("perturbation_type", ""),
        "data_source":      f"PerturbSeq_{ds_id_used}_{actual_cl}_{source_tag}",
        "note": (
            f"CRISPR perturbation in {actual_cl} ({sig_result.get('tissue', '')}) — "
            f"cell line matched to {disease_context or 'generic'} context. "
            "Tier 3 evidence; not sufficient alone for clinical translation."
        ) if beta is not None else (
            sig_result.get("note") or "Insufficient landmark coverage (< 5%) for β estimate."
        ),
    }


@_tool
def list_perturbseq_datasets(disease_context: str | None = None) -> dict:
    """
    List available Perturb-seq datasets with metadata and cache status.

    Args:
        disease_context: Filter to datasets in priority list for this disease.
                         If None, returns all datasets.
    """
    rows = []
    if disease_context:
        dk = disease_context.upper()
        priority = _DISEASE_DATASET_PRIORITY.get(dk, list(_DATASET_REGISTRY.keys()))
        ds_ids = priority
    else:
        ds_ids = list(_DATASET_REGISTRY.keys())

    for ds_id in ds_ids:
        meta  = _DATASET_REGISTRY[ds_id]
        cached = _sig_path(ds_id).exists()
        rows.append({
            "dataset_id":       ds_id,
            "name":             meta["name"],
            "cell_line":        meta["cell_line"],
            "tissue":           meta["tissue"],
            "disease_context":  meta["disease_context"],
            "perturbation_type": meta["perturbation_type"],
            "n_genes_perturbed": meta["n_genes_perturbed"],
            "cached":           cached,
            "access_type":      meta["access"]["type"],
            "citation":         meta["citation"],
        })

    priority_for_disease = (
        _DISEASE_DATASET_PRIORITY.get(disease_context.upper(), [])
        if disease_context else []
    )

    return {
        "disease_context":       disease_context,
        "n_datasets":            len(rows),
        "datasets":              rows,
        "priority_order":        priority_for_disease,
        "n_cached":              sum(1 for r in rows if r["cached"]),
        "preprocess_command": (
            "python -m mcp_servers.perturbseq_server preprocess <dataset_id> <h5ad_path>"
        ),
    }


@_tool
def find_upstream_regulators(
    target_genes: list[str],
    disease_context: str | None = None,
    dataset_id: str | None = None,
    min_abs_log2fc: float = 0.25,
    min_target_overlap: int = 1,
) -> dict:
    """
    Reverse-lookup: find knockouts whose perturbation downstream signature overlaps
    with target_genes.  Used to nominate upstream TF regulators from GWAS hit genes.

    For each knockout in the dataset, counts how many target_genes appear in its
    downstream signature with |log2fc| >= min_abs_log2fc.  Returns knockouts ranked
    by number of regulated targets (descending), then by sum |log2fc|.

    Args:
        target_genes:        Genes to look for in downstream signatures (e.g. GWAS hits)
        disease_context:     Disease key for dataset selection ("IBD", "CAD", ...)
        dataset_id:          Override dataset (skips disease-priority selection)
        min_abs_log2fc:      Minimum absolute log2FC to count a regulation event
        min_target_overlap:  Minimum target genes regulated to include a knockout

    Returns:
        {
          "regulators": [
            {
              "gene":                str,
              "n_targets_regulated": int,
              "regulated_targets":   list[str],
              "sum_abs_log2fc":      float,
              "dataset_id":          str,
              "evidence_tier":       "regulator_nomination",
            }, ...
          ],
          "target_genes_queried": list[str],
          "n_knockouts_tested":   int,
          "dataset_id":           str,
        }
    """
    ds_id = _select_dataset(disease_context, dataset_id)
    if ds_id is None:
        return {
            "regulators": [],
            "target_genes_queried": target_genes,
            "n_knockouts_tested": 0,
            "dataset_id": None,
        }

    sigs = _load_cached_signatures(ds_id)
    if sigs is None:
        return {
            "regulators": [],
            "target_genes_queried": target_genes,
            "n_knockouts_tested": 0,
            "dataset_id": ds_id,
        }

    targets_upper = {g.upper() for g in target_genes}
    results = []
    for ko_gene, downstream in sigs.items():
        regulated = [
            g for g, lfc in downstream.items()
            if g.upper() in targets_upper and abs(lfc) >= min_abs_log2fc
        ]
        if len(regulated) < min_target_overlap:
            continue
        results.append({
            "gene":                ko_gene,
            "n_targets_regulated": len(regulated),
            "regulated_targets":   regulated,
            "sum_abs_log2fc":      round(sum(abs(downstream[g]) for g in regulated), 4),
            "dataset_id":          ds_id,
            "evidence_tier":       "regulator_nomination",
        })

    results.sort(key=lambda r: (-r["n_targets_regulated"], -r["sum_abs_log2fc"]))

    return {
        "regulators":           results,
        "target_genes_queried": list(target_genes),
        "n_knockouts_tested":   len(sigs),
        "dataset_id":           ds_id,
    }


# ---------------------------------------------------------------------------
# CLI: preprocess a dataset from h5ad
# ---------------------------------------------------------------------------

def _cli_preprocess(args: list[str]) -> None:
    import logging as _log
    _log.basicConfig(level=_log.INFO, format="%(levelname)s %(message)s")
    if len(args) < 2:
        print("Usage: python -m mcp_servers.perturbseq_server preprocess <dataset_id> <h5ad_path>")
        print("\nAvailable dataset_ids:")
        for k, v in _DATASET_REGISTRY.items():
            print(f"  {k:30s}  {v['cell_line']:10s}  {v['n_genes_perturbed']:>6d} genes")
        sys.exit(1)
    dataset_id = args[0]
    h5ad_path  = args[1]
    meta = _DATASET_REGISTRY.get(dataset_id, {})
    # Auto-detect format from file extension or registry access type
    access_type = meta.get("access", {}).get("type", "")
    if access_type == "log2fc_matrix" or h5ad_path.endswith(".txt.gz") or h5ad_path.endswith(".txt"):
        result = preprocess_log2fc_matrix(dataset_id, h5ad_path)
    elif meta.get("h5ad_format") == "replogle_bulk":
        result = preprocess_replogle_bulk(dataset_id, h5ad_path)
    else:
        result = preprocess_h5ad(dataset_id, h5ad_path)
    print(json.dumps(result, indent=2))


def _cli_activate(args: list[str]) -> None:
    import logging as _log
    _log.basicConfig(level=_log.INFO, format="%(levelname)s %(message)s")
    if not args:
        print("Usage: python -m mcp_servers.perturbseq_server activate <dataset_id> [--keep-raw]")
        print("       python -m mcp_servers.perturbseq_server activate all [--keep-raw]")
        print("\nAvailable dataset_ids:")
        for k, v in _DATASET_REGISTRY.items():
            cached = "✓ cached" if _sig_path(k).exists() else "  not cached"
            acc = v["access"]
            size = f"  {acc.get('size_gb', '?')} GB" if acc.get("size_gb") else ""
            print(f"  {k:30s}  {v['cell_line']:10s}  {v['n_genes_perturbed']:>6d} genes  {cached}{size}")
        sys.exit(1)
    keep_raw   = "--keep-raw" in args
    dataset_id = args[0]
    targets = list(_DATASET_REGISTRY.keys()) if dataset_id == "all" else [dataset_id]
    for ds_id in targets:
        print(f"\n{'='*60}")
        print(f"Activating: {ds_id}")
        result = activate_dataset(ds_id, delete_raw=not keep_raw)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    argv = sys.argv[1:]
    if argv and argv[0] == "preprocess":
        _cli_preprocess(argv[1:])
    elif argv and argv[0] == "activate":
        _cli_activate(argv[1:])
    elif mcp is not None:
        mcp.run()
    else:
        raise RuntimeError("fastmcp required: pip install fastmcp")
