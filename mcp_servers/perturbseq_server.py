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
  natsume_2023_haec        HAEC         CAD (endothelial)     2,285   PLOS Gen doi:10.1371/journal.pgen.1010680
  schnitzler_cad_vascular  HCASMC/HAEC  CAD (vascular)          332   GEO GSE210681
  czi_2025_cd4t_perturb    CD4+ T       SLE/DED (Th17)       11,281   S3 Marson 2025

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

# Some datasets use a different directory name than their registry key.
# Maps canonical dataset_id → actual storage subdirectory under _CACHE_DIR.
_DATASET_DIR_ALIAS: dict[str, str] = {
    "Schnitzler_GSE210681": "schnitzler_cad_vascular",
}

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

    # ── CZI / Marson 2025 CD4+ T genome-scale (SLE) ──────────────────────────
    "czi_2025_cd4t_perturb": {
        "name":              "CZI/Marson 2025 genome-scale CRISPRi in primary human CD4+ T cells",
        "citation":          "Zhu*, Dann* et al., bioRxiv 2025. doi:10.1101/2025.12.23.696273",
        "cell_line":         "primary_CD4_T",
        "tissue":            "CD4_T_cell",
        "disease_context":   "SLE_RA",
        "perturbation_type": "CRISPRi",
        "n_genes_perturbed": 11327,
        "access": {
            "type": "s3_de_stats_h5ad",
            "s3_url": "https://genome-scale-tcell-perturb-seq.s3.amazonaws.com/marson2025_data/GWCD4i.DE_stats.h5ad",
            "size_gb": 15.6,
            "local_path": "data/perturbseq/czi_2025_cd4t_perturb/GWCD4i.DE_stats.h5ad",
            "note": (
                "Public S3 bucket (no auth). Download: "
                "curl -L -o data/perturbseq/czi_2025_cd4t_perturb/GWCD4i.DE_stats.h5ad "
                "https://genome-scale-tcell-perturb-seq.s3.amazonaws.com/marson2025_data/GWCD4i.DE_stats.h5ad\n"
                "Then preprocess: python -m mcp_servers.perturbseq_server preprocess "
                "czi_2025_cd4t_perturb data/perturbseq/czi_2025_cd4t_perturb/GWCD4i.DE_stats.h5ad"
            ),
        },
        "h5ad_format":               "czi_de_stats",
        "h5ad_obs_perturbation_col": "index",   # ENSG00000xxx_Condition
        "h5ad_nt_label":             "non-targeting",
        "geo":                        "GSE314342",
        "note": (
            "Best-available SLE+DED Perturb-seq β source. 11,327 genes perturbed in primary human "
            "CD4+ T cells (4 donors × 3 stimulation conditions: Rest, Stim8hr, Stim48hr). "
            "DE stats h5ad has layers['log_fc'] = (33983, 10282) matrix of log2FC per "
            "perturbation-condition vs non-targeting. Use Stim48hr condition for T cell "
            "activation programs (SLE/DED-relevant); Rest for basal regulatory network."
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
    # CAD: Schnitzler 332 CAD-specific vascular > Natsume HAEC only.
    # K562 removed: Zhu et al. (2025) Fig 6B shows K562 perturb-seq produces
    # no enrichment for lymphocyte count LoF burden signals vs primary T cells;
    # same logic applies to cardiac endothelial biology — cell-line mismatch
    # adds noise, not signal.
    "CAD":  ["schnitzler_cad_vascular", "natsume_2023_haec"],
    # RA: CZI CD4+ T Stim8hr condition (Zhu et al. 2025 Fig 7: autoimmune enrichment
    # in Stim8hr/Stim48hr clusters, not Rest; STAT3/IRF4/BATF RA circuit only
    # visible at 8hr post-stimulation). K562 removed.
    "RA":   ["czi_2025_cd4t_perturb"],
    # Generic: largest genome-wide screen first
    "GENERIC": ["replogle_2022_k562"],
}

# In-process signature cache: "{dataset_id}:{variant}" → {gene: {downstream_gene: log2fc}}
# variant="" means the default (primary condition) signatures.
_SIG_CACHE: dict[str, dict[str, dict[str, float]]] = {}

# ---------------------------------------------------------------------------
# Signature cache I/O
# ---------------------------------------------------------------------------

def _sig_path(dataset_id: str, variant: str = "") -> Path:
    """Return path to signatures file. variant="" → signatures.json.gz; else signatures_{variant}.json.gz."""
    suffix = f"_{variant}" if variant else ""
    storage_dir = _DATASET_DIR_ALIAS.get(dataset_id, dataset_id)
    return _CACHE_DIR / storage_dir / f"signatures{suffix}.json.gz"


def _npz_path(dataset_id: str) -> Path:
    """Return path to svd_loadings.npz, respecting directory aliases."""
    storage_dir = _DATASET_DIR_ALIAS.get(dataset_id, dataset_id)
    return _CACHE_DIR / storage_dir / "svd_loadings.npz"


def _cache_key(dataset_id: str, variant: str = "") -> str:
    return f"{dataset_id}:{variant}" if variant else dataset_id


def _load_cached_signatures(
    dataset_id: str, variant: str = ""
) -> dict[str, dict[str, float]] | None:
    """Load pre-computed signatures from disk. Returns None if not yet preprocessed."""
    ck = _cache_key(dataset_id, variant)
    if ck in _SIG_CACHE:
        return _SIG_CACHE[ck]
    p = _sig_path(dataset_id, variant)
    if not p.exists():
        return None
    with gzip.open(p, "rt", encoding="utf-8") as f:
        data = json.load(f)
    _SIG_CACHE[ck] = data
    return data


def _save_signatures(
    dataset_id: str, signatures: dict[str, dict[str, float]], variant: str = ""
) -> None:
    """Save pre-computed signatures to disk."""
    p = _sig_path(dataset_id, variant)
    p.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(p, "wt", encoding="utf-8") as f:
        json.dump(signatures, f)
    _SIG_CACHE[_cache_key(dataset_id, variant)] = signatures
    log.info("Saved %d gene signatures for %s (variant=%r) → %s", len(signatures), dataset_id, variant, p)


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


def preprocess_czi_de_stats(
    dataset_id: str,
    h5ad_path: str | Path,
    condition: str = "Stim48hr",
    top_k: int = 200,
    min_abs_log2fc: float = 0.1,
) -> dict:
    """
    Preprocess CZI/Marson 2025 GWCD4i.DE_stats.h5ad into per-gene signatures.

    The h5ad has:
      obs.index = "{ENSEMBL_ID}_{condition}"  (33,983 rows)
      layers["log_fc"] = (33983, 10282)  — log2FC vs non-targeting per gene
      var["gene_name"] = gene symbols

    Args:
        condition: "Rest", "Stim8hr", or "Stim48hr" (default Stim48hr for SLE T cell activation)
        top_k: top downstream genes per perturbation to keep
        min_abs_log2fc: minimum |log2FC| threshold
    """
    import gzip as _gz
    try:
        import anndata as ad
    except ImportError:
        return {"error": "anndata not installed; run: pip install anndata"}

    path = Path(h5ad_path)
    if not path.exists():
        return {"error": f"File not found: {path}"}

    log.info("Loading CZI DE stats h5ad: %s (condition=%s)", path, condition)
    adata = ad.read_h5ad(str(path))

    # Filter to requested condition
    cond_mask = adata.obs.index.str.endswith(f"_{condition}")
    if cond_mask.sum() == 0:
        available = sorted({idx.rsplit("_", 1)[-1] for idx in adata.obs.index})
        return {"error": f"Condition '{condition}' not found. Available: {available}"}
    adata = adata[cond_mask]
    log.info("Filtered to condition '%s': %d perturbations", condition, adata.n_obs)

    # Build ENSEMBL → symbol map from obs.index (format: ENSGxxxx_Condition)
    # and var gene names
    var_symbols = list(adata.var["gene_name"] if "gene_name" in adata.var.columns else adata.var_names)

    # Load log2FC matrix
    log_fc = adata.layers["log_fc"]
    if hasattr(log_fc, "toarray"):
        log_fc = log_fc.toarray()
    import numpy as np
    log_fc = np.asarray(log_fc, dtype=np.float32)

    # Build per-gene signatures
    # obs.index format: "ENSG00000121410_Stim48hr" → perturbation gene = ENSEMBL ID
    # Map ENSEMBL → symbol using HGNC if available
    try:
        from pipelines.static_lookups import get_lookups as _get_lookups
        _lu = _get_lookups()
        def _ensg_to_sym(ensg: str) -> str:
            sym = _lu.get_symbol_from_ensembl(ensg)
            return sym if sym else ensg
    except Exception:
        def _ensg_to_sym(ensg: str) -> str:
            return ensg

    signatures: dict[str, dict[str, float]] = {}
    for i, obs_id in enumerate(adata.obs.index):
        ensg = obs_id.rsplit("_", 1)[0]
        gene_sym = _ensg_to_sym(ensg)

        row = log_fc[i]
        # Pair with gene symbols
        pairs = [(sym, float(lfc)) for sym, lfc in zip(var_symbols, row)
                 if abs(float(lfc)) >= min_abs_log2fc and lfc == lfc]  # finite check

        if not pairs:
            continue
        # Top-k by |log2FC|
        pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        signatures[gene_sym] = dict(pairs[:top_k])

    # Save: Stim8hr is the primary condition → signatures.json.gz; others → signatures_{condition}.json.gz
    variant = "" if condition == "Stim8hr" else condition
    _save_signatures(dataset_id, signatures, variant=variant)
    sig_path = _sig_path(dataset_id, variant=variant)

    log.info("Saved %d signatures to %s", len(signatures), sig_path)
    return {
        "dataset_id":        dataset_id,
        "condition_used":    condition,
        "variant":           variant,
        "n_perturbed_genes": len(signatures),
        "n_cached_genes":    len(signatures),
        "sig_path":          str(sig_path),
    }


def preprocess_czi_delta_beta(
    dataset_id: str,
    h5ad_path: str | Path,
    top_k: int = 200,
    min_abs_log2fc: float = 0.1,
) -> dict:
    """
    Compute δ-β signatures for CZI/Marson 2025: δ = Stim8hr − Rest log2FC.

    δ-β isolates the TCR-stimulation-specific regulatory effect of each gene KO,
    removing constitutive housekeeping perturbation noise present in both Rest and
    Stim8hr. For RA/SLE autoimmune biology: Rest reflects basal T cell state
    irrelevant to antigen-driven inflammation; Stim8hr captures the activation
    response directly relevant to disease (Zhu et al. 2025 Fig 7A).

    Saves output to signatures_delta.json.gz.

    Args:
        dataset_id:      Registry key (must be "czi_2025_cd4t_perturb")
        h5ad_path:       Path to GWCD4i.DE_stats.h5ad
        top_k:           Max downstream genes per perturbation to keep
        min_abs_log2fc:  Minimum |δ log2FC| threshold
    """
    import gzip as _gz
    try:
        import anndata as ad
    except ImportError:
        return {"error": "anndata not installed; run: pip install anndata"}

    path = Path(h5ad_path)
    if not path.exists():
        return {"error": f"File not found: {path}"}

    log.info("Loading CZI h5ad for δ-β (Stim8hr − Rest): %s", path)
    adata = ad.read_h5ad(str(path))

    # Extract log2FC for Stim8hr and Rest
    stim_mask = adata.obs.index.str.endswith("_Stim8hr")
    rest_mask  = adata.obs.index.str.endswith("_Rest")

    if stim_mask.sum() == 0 or rest_mask.sum() == 0:
        available = sorted({idx.rsplit("_", 1)[-1] for idx in adata.obs.index})
        return {"error": f"Missing Stim8hr or Rest condition. Available: {available}"}

    log.info("Stim8hr: %d perturbations, Rest: %d perturbations", stim_mask.sum(), rest_mask.sum())

    adata_stim = adata[stim_mask]
    adata_rest  = adata[rest_mask]

    import numpy as np

    lfc_stim = adata_stim.layers["log_fc"]
    lfc_rest  = adata_rest.layers["log_fc"]
    if hasattr(lfc_stim, "toarray"):
        lfc_stim = lfc_stim.toarray()
    if hasattr(lfc_rest, "toarray"):
        lfc_rest = lfc_rest.toarray()
    lfc_stim = np.asarray(lfc_stim, dtype=np.float32)
    lfc_rest  = np.asarray(lfc_rest,  dtype=np.float32)

    var_symbols = list(adata.var["gene_name"] if "gene_name" in adata.var.columns else adata.var_names)

    try:
        from pipelines.static_lookups import get_lookups as _get_lookups
        _lu = _get_lookups()
        def _ensg_to_sym(ensg: str) -> str:
            sym = _lu.get_symbol_from_ensembl(ensg)
            return sym if sym else ensg
    except Exception:
        def _ensg_to_sym(ensg: str) -> str:
            return ensg

    # Build ENSG index for Rest rows so we can align with Stim8hr
    rest_ensg_index: dict[str, int] = {}
    for i, obs_id in enumerate(adata_rest.obs.index):
        ensg = obs_id.rsplit("_", 1)[0]
        rest_ensg_index[ensg] = i

    signatures: dict[str, dict[str, float]] = {}
    n_matched = 0

    for i, obs_id in enumerate(adata_stim.obs.index):
        ensg = obs_id.rsplit("_", 1)[0]
        gene_sym = _ensg_to_sym(ensg)

        rest_i = rest_ensg_index.get(ensg)
        if rest_i is None:
            continue  # gene not perturbed in Rest condition

        n_matched += 1
        delta = lfc_stim[i] - lfc_rest[rest_i]  # δ = Stim8hr − Rest

        pairs = [(sym, float(d)) for sym, d in zip(var_symbols, delta)
                 if abs(float(d)) >= min_abs_log2fc and d == d]
        if not pairs:
            continue
        pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        signatures[gene_sym] = dict(pairs[:top_k])

    log.info("δ-β: %d/%d genes matched Stim8hr↔Rest; %d with signatures",
             n_matched, stim_mask.sum(), len(signatures))
    _save_signatures(dataset_id, signatures, variant="delta")

    return {
        "dataset_id":        dataset_id,
        "variant":           "delta",
        "delta_formula":     "Stim8hr - Rest",
        "n_perturbed_genes": stim_mask.sum(),
        "n_matched_genes":   n_matched,
        "n_cached_genes":    len(signatures),
        "sig_path":          str(_sig_path(dataset_id, "delta")),
    }


def compute_cad_disease_regression(
    h5ad_path: str | Path,
    log2fc_matrix_path: str | Path,
    output_path: str | Path | None = None,
    top_k_genes: int = 500,
    alpha: float = 1.0,
) -> dict:
    """
    Fit ridge regression of CAD endothelial disease signature against Schnitzler perturbation matrix.

    Computes MI vs normal DEGs from CAD_cardiac_endothelial_cell.h5ad, then fits:
        signature_vector ~ Σ_r (w_r × log2fc_column_r)

    where each column r is one perturbed gene's log2FC profile in Schnitzler.
    The fitted weights w_r estimate how much gene r's perturbation contributes
    to the MI disease state — analogous to Zhu et al. (2025) Fig 4A.

    Saves per-gene regressor weights to
    data/perturbseq/schnitzler_cad_vascular/cad_disease_regression.json.

    Args:
        h5ad_path:          Path to CAD_cardiac_endothelial_cell.h5ad
        log2fc_matrix_path: Path to GSE210681_ALL_log2fcs.txt.gz
        output_path:        Override output JSON path
        top_k_genes:        Top DEGs by |log2FC| to use as signature (max 500)
        alpha:              Ridge regularisation strength (L2 penalty)
    """
    try:
        import anndata as ad
        import numpy as _np
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
    except ImportError as e:
        return {"error": f"Missing dependency: {e}. Run: pip install anndata scikit-learn"}

    h5ad_path = Path(h5ad_path)
    if not h5ad_path.exists():
        return {"error": f"h5ad not found: {h5ad_path}"}

    log.info("Loading CAD endothelial h5ad: %s", h5ad_path)
    adata = ad.read_h5ad(str(h5ad_path))

    # Compute MI vs normal differential expression
    disease_col = "disease"
    if disease_col not in adata.obs.columns:
        return {"error": f"'disease' column not found in obs. Available: {list(adata.obs.columns)}"}

    mi_mask     = adata.obs[disease_col] == "myocardial infarction"
    normal_mask = adata.obs[disease_col] == "normal"

    if mi_mask.sum() < 10 or normal_mask.sum() < 10:
        return {"error": f"Too few cells: MI={mi_mask.sum()}, normal={normal_mask.sum()}"}

    import scipy.sparse as _sp
    X_h5ad = adata.X
    if _sp.issparse(X_h5ad):
        X_h5ad = X_h5ad.toarray()
    X_h5ad = _np.asarray(X_h5ad, dtype=_np.float32)

    mean_mi     = X_h5ad[mi_mask].mean(axis=0)
    mean_normal = X_h5ad[normal_mask].mean(axis=0)
    log2fc_disease = (mean_mi - mean_normal) / _np.log(2)  # pseudo-log2FC

    # Gene symbol index from h5ad var
    if "feature_name" in adata.var.columns:
        gene_syms = list(adata.var["feature_name"])
    elif "gene_name" in adata.var.columns:
        gene_syms = list(adata.var["gene_name"])
    else:
        gene_syms = list(adata.var_names)

    # Select top DEGs as disease signature
    abs_lfc = _np.abs(log2fc_disease)
    top_idx = _np.argsort(abs_lfc)[-top_k_genes:]
    sig_genes = [gene_syms[i] for i in top_idx]
    sig_vec   = {gene_syms[i]: float(log2fc_disease[i]) for i in top_idx}
    log.info("CAD disease signature: %d genes (top %d by |log2FC|, MI vs normal)", len(sig_genes), top_k_genes)

    # Load Schnitzler matrix and build design matrix (rows=genes, cols=perturbations)
    import gzip as _gz, re
    log2fc_path = Path(log2fc_matrix_path)
    if not log2fc_path.exists():
        return {"error": f"Schnitzler matrix not found: {log2fc_path}"}

    log.info("Loading Schnitzler log2FC matrix: %s", log2fc_path)
    opener = _gz.open(log2fc_path, "rt") if str(log2fc_path).endswith(".gz") else open(log2fc_path, "r")

    row_gene_list: list[str] = []
    pert_matrix_rows: list[list[float]] = []
    pert_names: list[str] = []

    def _canonical(name: str) -> str:
        return re.sub(r"-\d+$", "", name)

    with opener as fh:
        header = fh.readline().rstrip("\n").split("\t")
        pert_cols = header[2:]
        canonical_perts = [_canonical(p) for p in pert_cols]
        unique_perts_ordered = list(dict.fromkeys(canonical_perts))  # deduplicate preserving order
        pert_names = unique_perts_ordered

        # Per-row accumulator: gene → {pert → [values]} (multiple guides per pert)
        _acc: dict[str, dict[str, list[float]]] = {}
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < len(pert_cols) + 2:
                continue
            gene_field = parts[1]
            gene_sym = gene_field.split(":")[0]
            if not gene_sym or gene_sym not in sig_genes:
                continue  # skip genes not in disease signature
            if gene_sym not in _acc:
                _acc[gene_sym] = {}
            for i, val_str in enumerate(parts[2:2 + len(pert_cols)]):
                try:
                    val = float(val_str)
                except ValueError:
                    continue
                if not _np.isfinite(val):
                    continue
                cp = canonical_perts[i]
                _acc[gene_sym].setdefault(cp, []).append(val)

    # Average multi-guide values and build (n_sig_genes × n_perts) matrix
    sig_gene_order = [g for g in sig_genes if g in _acc]
    if len(sig_gene_order) < 20:
        return {"error": f"Only {len(sig_gene_order)}/{len(sig_genes)} disease signature genes found in Schnitzler matrix"}

    log.info("Building design matrix: %d sig genes × %d perturbations", len(sig_gene_order), len(pert_names))
    X_design = _np.zeros((len(sig_gene_order), len(pert_names)), dtype=_np.float32)
    for row_i, g in enumerate(sig_gene_order):
        for col_j, pname in enumerate(pert_names):
            vals = _acc[g].get(pname, [])
            if vals:
                X_design[row_i, col_j] = float(_np.mean(vals))

    # Response vector y: disease log2FC for each sig gene
    y = _np.array([sig_vec[g] for g in sig_gene_order], dtype=_np.float32)

    # Ridge regression: perturbation columns predict disease signature
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_design)
    ridge = Ridge(alpha=alpha, fit_intercept=True)
    ridge.fit(X_scaled, y)

    # Weights: positive = KO of this gene reverses the MI signature (down-regulates disease)
    #          negative = KO amplifies the MI signature (disease-promoting gene)
    weights = {pert_names[j]: round(float(ridge.coef_[j]), 6) for j in range(len(pert_names))}
    r2 = float(ridge.score(X_scaled, y))
    log.info("Ridge R²=%.3f over %d genes × %d perturbations", r2, len(sig_gene_order), len(pert_names))

    out_path = Path(output_path) if output_path else (
        _CACHE_DIR / "schnitzler_cad_vascular" / "cad_disease_regression.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "r2":              r2,
            "alpha":           alpha,
            "n_sig_genes":     len(sig_gene_order),
            "n_perturbations": len(pert_names),
            "weights":         weights,
        }, f)
    log.info("CAD disease regression weights saved to %s", out_path)

    return {
        "r2":              round(r2, 4),
        "alpha":           alpha,
        "n_sig_genes":     len(sig_gene_order),
        "n_perturbations": len(pert_names),
        "top_positive":    sorted(weights.items(), key=lambda x: x[1], reverse=True)[:10],
        "top_negative":    sorted(weights.items(), key=lambda x: x[1])[:10],
        "output_path":     str(out_path),
    }


# ---------------------------------------------------------------------------
# RNA fingerprinting — SVD denoising + probabilistic disease matching
# ---------------------------------------------------------------------------

def preprocess_rna_fingerprints(
    dataset_id: str,
    source_variant: str = "",
    top_k: int = 200,
) -> dict:
    """
    Denoise Perturb-seq signatures via truncated SVD (RNA fingerprinting).

    Implements the denoising stage of Elorbany et al. (2025, PMC12458102):
    builds a gene × perturbation log2FC matrix, centers per-gene, applies
    rank-k truncated SVD, and reconstructs denoised per-perturbation vectors.

    Reads:  signatures{_source_variant}.json.gz
    Writes: signatures_fingerprint{_source_variant if source_variant else ''}.json.gz

    Args:
        dataset_id:     Dataset to denoise (e.g. "schnitzler_cad_vascular")
        source_variant: Variant of source signatures ("" = primary, "delta", "Stim48hr")
        top_k:          Top genes by |denoised value| to retain per perturbation
    """
    try:
        import numpy as _np
        from sklearn.utils.extmath import randomized_svd
    except ImportError as e:
        return {"error": f"Missing dependency: {e}. Run: pip install scikit-learn numpy"}

    from config.scoring_thresholds import FINGERPRINT_SVD_RANK

    sigs = _load_cached_signatures(dataset_id, source_variant)
    if not sigs:
        return {"error": f"No signatures for {dataset_id!r} variant={source_variant!r}. Run preprocess first."}

    # Build gene × perturbation matrix
    all_genes = sorted({g for fc in sigs.values() for g in fc})
    all_perts = sorted(sigs.keys())
    n_genes, n_perts = len(all_genes), len(all_perts)
    gene_idx = {g: i for i, g in enumerate(all_genes)}

    M = _np.zeros((n_genes, n_perts), dtype=_np.float32)
    for j, pert in enumerate(all_perts):
        for g, v in sigs[pert].items():
            if g in gene_idx:
                M[gene_idx[g], j] = float(v)

    # Center: subtract per-gene mean across perturbations to remove shared baseline
    M -= M.mean(axis=1, keepdims=True)

    # Truncated SVD: rank-k approximation captures perturbation-specific signal
    k = min(FINGERPRINT_SVD_RANK, n_genes - 1, n_perts - 1)
    U, S, Vt = randomized_svd(M, n_components=k, random_state=42)
    M_denoised = (U * S) @ Vt  # (n_genes × n_perts)

    # Magnitude rescaling: restore each column to its original L2 norm.
    # SVD reconstruction shrinks values toward zero (low-rank approximation),
    # which would compress β downstream. Rescaling keeps the denoised *direction*
    # (noise removed) while preserving the original *scale* (log2FC magnitude).
    raw_norms = _np.linalg.norm(M, axis=0)           # original column norms
    den_norms = _np.linalg.norm(M_denoised, axis=0)  # denoised column norms
    scale = _np.where(den_norms > 1e-8, raw_norms / den_norms, 1.0)
    M_denoised *= scale[_np.newaxis, :]

    # Extract denoised fingerprints — top_k genes by absolute denoised value
    denoised: dict[str, dict[str, float]] = {}
    for j, pert in enumerate(all_perts):
        col = M_denoised[:, j]
        top_idx = _np.argsort(_np.abs(col))[-top_k:]
        denoised[pert] = {
            all_genes[i]: round(float(col[i]), 6)
            for i in top_idx
            if abs(col[i]) > 1e-6
        }

    fp_variant = f"fingerprint{'_' + source_variant if source_variant else ''}"
    _save_signatures(dataset_id, denoised, fp_variant)

    # Save SVD perturbation loadings (Vt) for downstream nomination scoring.
    # Vt shape: (k × n_perts); each column = a perturbed gene's latent coordinates.
    # Only saved for the primary variant (source_variant="") — delta/Stim48hr share
    # the same nomination space as the primary.
    if not source_variant:
        npz_path = _npz_path(dataset_id)
        # U_scaled = U @ diag(S): gene expression loadings (n_genes × k).
        # Each column c encodes which genes define SVD component c.
        # U_scaled[:,c] is used for: (1) SVD program gene sets (top-|loading| genes),
        # (2) program_loading for eQTL Tier2 β estimates of non-perturbed GWAS genes.
        U_scaled = U * S  # (n_genes × k)
        _np.savez_compressed(
            str(npz_path),
            Vt=Vt,
            pert_names=_np.array(all_perts),
            U_scaled=U_scaled,
            gene_names=_np.array(all_genes),
            singular_values=S,
        )
        log.info("SVD loadings saved: %s (rank=%d, %d perts, %d genes)", npz_path, k, n_perts, n_genes)

    log.info(
        "RNA fingerprints saved: %s variant=%r — %d perts, %d genes, SVD rank=%d",
        dataset_id, fp_variant, n_perts, n_genes, k,
    )
    return {
        "dataset_id":       dataset_id,
        "source_variant":   source_variant,
        "fingerprint_variant": fp_variant,
        "n_perturbations":  n_perts,
        "n_genes":          n_genes,
        "svd_rank":         k,
        "sig_path":         str(_sig_path(dataset_id, fp_variant)),
        "svd_loadings_path": str(_npz_path(dataset_id)) if not source_variant else None,
    }


def compute_svd_nomination_scores(
    dataset_id: str,
    gwas_genes: list[str] | set[str],
    top_k: int | None = None,
    min_cosine: float | None = None,
) -> list[dict]:
    """
    Rank non-GWAS perturbed genes by cosine similarity to GWAS genes in SVD space.

    After fingerprint preprocessing, each perturbed gene has a k-dimensional loading
    vector in Vt (the right singular vectors of the centered gene × perturbation matrix).
    Genes that co-vary with GWAS-anchored genes across perturbations occupy similar
    positions in this latent space — they are mechanistically co-regulated and are
    stronger disease-biology nominees than raw log2FC magnitude alone would suggest.

    Args:
        dataset_id:  Dataset to use (must have svd_loadings.npz from preprocess_rna_fingerprints)
        gwas_genes:  Set of GWAS-anchored gene symbols (defines the reference centroid)
        top_k:       Max nominees to return (default: FINGERPRINT_MAX_NONGWAS_NOMINEES)
        min_cosine:  Minimum cosine score (default: FINGERPRINT_SVD_COSINE_MIN)

    Returns:
        List of {gene, cosine_score} sorted descending by score (non-GWAS only).
        Empty list if SVD loadings not yet computed.
    """
    try:
        import numpy as _np
    except ImportError:
        return []

    from config.scoring_thresholds import FINGERPRINT_SVD_COSINE_MIN

    npz_path = _npz_path(dataset_id)
    if not npz_path.exists():
        log.warning("SVD loadings not found for %s — run preprocess_rna_fingerprints first", dataset_id)
        return []

    data = _np.load(str(npz_path), allow_pickle=True)
    Vt        = data["Vt"]           # (k × n_perts)
    pert_names = list(data["pert_names"])  # perturbed gene symbols

    gwas_upper  = {g.upper() for g in gwas_genes}
    pert_upper  = [p.upper() for p in pert_names]

    # (n_perts × k): each row = one gene's latent coordinates
    V = Vt.T

    # Mean loading vector of all GWAS genes present in the perturbation set
    gwas_idx = [i for i, p in enumerate(pert_upper) if p in gwas_upper]
    if not gwas_idx:
        log.warning("No GWAS genes found in %s SVD loadings — cannot compute nomination scores", dataset_id)
        return []

    centroid     = V[gwas_idx].mean(axis=0)
    centroid_norm = _np.linalg.norm(centroid)
    if centroid_norm < 1e-8:
        return []
    centroid_hat = centroid / centroid_norm

    _min_cos = min_cosine if min_cosine is not None else FINGERPRINT_SVD_COSINE_MIN
    # top_k retained as an optional override for tests; default None = no cap
    # (the cosine threshold is the principled gate — a count cap is size-dependent)

    results: list[dict] = []
    for i, (pname, pupper) in enumerate(zip(pert_names, pert_upper)):
        if pupper in gwas_upper:
            continue  # skip GWAS genes — they're already in the candidate list
        v    = V[i]
        norm = _np.linalg.norm(v)
        if norm < 1e-8:
            continue
        cosine = float(_np.dot(centroid_hat, v / norm))
        if cosine >= _min_cos:
            results.append({"gene": pname, "cosine_score": round(cosine, 6)})

    results.sort(key=lambda x: x["cosine_score"], reverse=True)
    log.info(
        "SVD nomination: %d non-GWAS candidates (cosine≥%.2f) for %s",
        len(results), _min_cos, dataset_id,
    )
    return results[:top_k] if top_k is not None else results


def get_gwas_aligned_svd_programs(
    dataset_id: str,
    disease_key: str,
    gwas_genes: list[str] | set[str],
    de_vector: dict[str, float] | None = None,
    top_n_genes: int = 100,
) -> list[dict]:
    """
    Score and rank SVD components by joint GWAS + disease-DE alignment.

    Two orthogonal scores per component c:

      gwas_t[c] = mean(Vt[c, gwas_pert_idx]) / SEM_background
        Measures whether GWAS-perturbed genes systematically load on component c.
        Large |t| = component captures disease-relevant perturbational biology.

      de_pearson[c] = Pearson(U_scaled[:,c], de_vector)
        Measures whether the component's gene expression pattern matches the
        disease-vs-healthy DE signature from scRNA-seq.
        Large |r| = component is transcriptionally dysregulated in disease cells.

    Combined score (when de_vector provided):
      combined[c] = gwas_t[c] * de_pearson[c]
      Positive = both sources agree (GWAS genes activate a program that is
        upregulated in disease, or suppress one that is downregulated).
      Ranked by |combined| so programs with dual evidence come first.

    Without de_vector: ranked by |gwas_t| alone (GWAS-only mode).

    Args:
        dataset_id:   Perturb-seq dataset (must have svd_loadings.npz)
        disease_key:  Short disease key ("CAD", "RA")
        gwas_genes:   GWAS-anchored gene symbols
        de_vector:    {gene_symbol: log2FC(disease vs healthy)} from scRNA-seq h5ad
        top_n_genes:  Top-|U_scaled[:,c]| genes stored per program for γ estimation

    Returns:
        List of program dicts sorted by |combined_score| (or |gwas_t| if no de_vector).
        Extra fields per program: gwas_alignment, gwas_t_stat, de_pearson,
        combined_score, n_gwas_perturbed.
        Returns [] if npz missing or no GWAS genes perturbed.
    """
    try:
        import numpy as _np
    except ImportError:
        return []

    npz_path = _npz_path(dataset_id)
    if not npz_path.exists():
        log.warning("SVD loadings not found for %s", dataset_id)
        return []

    data = _np.load(str(npz_path), allow_pickle=True)
    if "U_scaled" not in data or "gene_names" not in data:
        log.warning("U_scaled missing from %s — re-run preprocess_rna_fingerprints", npz_path)
        return []

    Vt         = data["Vt"]          # (k × n_perts)
    pert_names = list(data["pert_names"])
    U_scaled   = data["U_scaled"]    # (n_genes × k)
    gene_names = list(data["gene_names"])
    k          = Vt.shape[0]

    gwas_upper = {g.upper() for g in gwas_genes}
    gwas_pert_idx = [i for i, p in enumerate(pert_names) if p.upper() in gwas_upper]
    n_gwas_perturbed = len(gwas_pert_idx)

    if n_gwas_perturbed == 0:
        log.warning("No GWAS genes found in %s perturbation set — cannot align SVD components", dataset_id)
        return []

    log.info(
        "GWAS-SVD alignment: %d/%d GWAS genes perturbed in %s",
        n_gwas_perturbed, len(gwas_upper), dataset_id,
    )

    gwas_vt = Vt[:, gwas_pert_idx]  # (k × n_gwas_perturbed)

    # Pre-build DE alignment vector aligned to gene_names order (once, outside loop)
    de_arr: "_np.ndarray | None" = None
    de_nonzero_mask: "_np.ndarray | None" = None
    if de_vector:
        de_arr = _np.array([de_vector.get(g, 0.0) for g in gene_names], dtype=_np.float32)
        de_nonzero_mask = _np.abs(de_arr) > 1e-8
        if int(de_nonzero_mask.sum()) < 20:
            log.warning("DE vector has fewer than 20 non-zero genes overlapping gene_names — DE alignment skipped")
            de_arr = None
            de_nonzero_mask = None
        else:
            log.info(
                "DE alignment: %d genes with |log2FC|>0 overlapping %s gene universe",
                int(de_nonzero_mask.sum()), dataset_id,
            )

    def _pearson(x: "_np.ndarray", y: "_np.ndarray") -> float:
        """Pearson r without scipy dependency."""
        xc = x - x.mean()
        yc = y - y.mean()
        denom = _np.sqrt((xc ** 2).sum() * (yc ** 2).sum())
        return float(_np.dot(xc, yc) / denom) if denom > 1e-12 else 0.0

    programs: list[dict] = []
    for c in range(k):
        prog_id = f"{disease_key}_SVD_C{c + 1:02d}"

        # GWAS alignment: mean loading of GWAS-perturbed genes on this component
        component_gwas = gwas_vt[c]
        gwas_alignment = float(_np.mean(component_gwas))
        background_std = float(_np.std(Vt[c]))
        sem            = background_std / _np.sqrt(n_gwas_perturbed) if n_gwas_perturbed > 1 else background_std
        gwas_t_stat    = gwas_alignment / sem if sem > 1e-12 else 0.0

        # Disease DE alignment: Pearson(U_scaled[:,c], de_vector) over shared genes
        de_pearson = 0.0
        if de_arr is not None and de_nonzero_mask is not None:
            u_col = U_scaled[de_nonzero_mask, c]
            d_col = de_arr[de_nonzero_mask]
            de_pearson = _pearson(u_col, d_col)

        # Combined score: product of both signals.
        # Same sign = both sources agree on disease direction.
        # |combined| large = strong dual evidence (GWAS + transcriptional).
        combined_score = gwas_t_stat * de_pearson if de_arr is not None else gwas_t_stat

        # Gene side: top-|U_scaled[:,c]| genes for γ estimation
        col     = U_scaled[:, c]
        top_idx = _np.argsort(_np.abs(col))[::-1][:top_n_genes]
        top_genes = [
            {"gene": gene_names[i], "weight": round(float(col[i]), 6)}
            for i in top_idx
            if abs(col[i]) > 1e-8
        ]

        programs.append({
            "program_id":       prog_id,
            "name":             prog_id,
            "gene_set":         [g["gene"] for g in top_genes],
            "top_genes":        top_genes,
            "source":           "svd_gwas_de_aligned" if de_arr is not None else "svd_gwas_aligned",
            "gwas_alignment":   round(gwas_alignment, 6),
            "gwas_t_stat":      round(gwas_t_stat, 4),
            "de_pearson":       round(de_pearson, 4),
            "combined_score":   round(combined_score, 4),
            "n_gwas_perturbed": n_gwas_perturbed,
        })

    # Sort by |combined_score| when DE available, else |gwas_t_stat|
    sort_key = "combined_score" if de_arr is not None else "gwas_t_stat"
    programs.sort(key=lambda p: abs(p[sort_key]), reverse=True)

    top3 = [(p["program_id"], p.get("combined_score", p["gwas_t_stat"])) for p in programs[:3]]
    log.info("Top GWAS+DE-aligned SVD components for %s: %s", disease_key, top3)

    return programs


def get_svd_program_gene_sets(
    dataset_id: str,
    disease_key: str,
    top_n: int = 100,
) -> dict[str, list[dict]]:
    """
    Return SVD components as program definitions — gene side (U_scaled).

    Each SVD component c becomes program "{disease_key}_SVD_C{c+1:02d}".
    The gene set = top-|U_scaled[:,c]| genes (expression space).
    These gene sets are used for γ(program→disease) GWAS enrichment and for
    the NES computation in load_replogle_betas (so β and island axes converge).

    Args:
        dataset_id:  Perturb-seq dataset (must have svd_loadings.npz with U_scaled)
        disease_key: Short disease key (e.g. "CAD", "RA") for program naming
        top_n:       Number of top genes per component (by |U_scaled[:,c]|)

    Returns:
        {program_id: [{"gene": str, "weight": float}, ...]} sorted by |weight| desc
        Empty dict if npz not found or U_scaled not saved yet.
    """
    try:
        import numpy as _np
    except ImportError:
        return {}

    npz_path = _npz_path(dataset_id)
    if not npz_path.exists():
        return {}

    data = _np.load(str(npz_path), allow_pickle=True)
    if "U_scaled" not in data or "gene_names" not in data:
        log.warning(
            "U_scaled not in %s — re-run preprocess_rna_fingerprints to regenerate", npz_path
        )
        return {}

    U_scaled   = data["U_scaled"]   # (n_genes × k)
    gene_names = list(data["gene_names"])
    k          = U_scaled.shape[1]

    result: dict[str, list[dict]] = {}
    for c in range(k):
        prog_id  = f"{disease_key}_SVD_C{c + 1:02d}"
        col      = U_scaled[:, c]
        top_idx  = _np.argsort(_np.abs(col))[::-1][:top_n]
        top_genes = [
            {"gene": gene_names[i], "weight": round(float(col[i]), 6)}
            for i in top_idx
            if abs(col[i]) > 1e-8
        ]
        result[prog_id] = top_genes

    return result


def load_svd_program_betas(
    dataset_id: str,
    disease_key: str,
) -> dict[str, dict[str, float]]:
    """
    Return direct Vt-based β(gene→program) for perturbed genes.

    β(gene→SVD_Cc) = Vt[c, pert_idx] — the perturbation's loading on SVD
    component c.  Positive = perturbation activates the component; negative =
    perturbation suppresses it.  This is the most direct mechanistic estimate:
    it uses the actual transcriptional response observed in the experiment
    rather than an NES against a gene set.

    Args:
        dataset_id:  Perturb-seq dataset (must have svd_loadings.npz)
        disease_key: Short disease key for program naming (e.g. "CAD", "RA")

    Returns:
        {gene_symbol: {prog_id: beta_float}} for all perturbed genes.
        Empty dict if npz not found.
    """
    try:
        import numpy as _np
    except ImportError:
        return {}

    npz_path = _npz_path(dataset_id)
    if not npz_path.exists():
        return {}

    data = _np.load(str(npz_path), allow_pickle=True)
    Vt         = data["Vt"]           # (k × n_perts)
    pert_names = list(data["pert_names"])
    k          = Vt.shape[0]

    result: dict[str, dict[str, float]] = {}
    for j, gene in enumerate(pert_names):
        prog_betas: dict[str, float] = {}
        for c in range(k):
            prog_id = f"{disease_key}_SVD_C{c + 1:02d}"
            prog_betas[prog_id] = round(float(Vt[c, j]), 6)
        result[gene] = prog_betas

    return result


def load_svd_vt_betas_zscored(
    dataset_id: str,
    disease_key: str,
) -> dict[str, dict]:
    """
    Return z-scored Vt-based β(gene→program) structured for estimate_beta_tier1.

    For each SVD component c, z-score Vt[c,:] across all perturbations so that
    β = (Vt[c,g] - mean_c) / std_c.  This puts the β values on a ±1-2 scale
    consistent with NES, while preserving the specificity of each perturbation's
    loading pattern (no saturation to the NES cap).

    Returns:
        {gene: {"programs": {prog_id: {"beta": z_score, "se": None, ...}}}}
        for all perturbed genes.  Empty dict if npz not found.
    """
    try:
        import numpy as _np
    except ImportError:
        return {}

    npz_path = _npz_path(dataset_id)
    if not npz_path.exists():
        return {}

    data       = _np.load(str(npz_path), allow_pickle=True)
    Vt         = data["Vt"]           # (k × n_perts)
    pert_names = list(data["pert_names"])
    k          = Vt.shape[0]

    # Z-score each component row independently
    row_mean = Vt.mean(axis=1, keepdims=True)   # (k, 1)
    row_std  = Vt.std(axis=1, keepdims=True)    # (k, 1)
    Vt_z     = (Vt - row_mean) / (_np.maximum(row_std, 1e-8))  # (k × n_perts)

    result: dict[str, dict] = {}
    for j, gene in enumerate(pert_names):
        programs: dict[str, dict] = {}
        for c in range(k):
            prog_id = f"{disease_key}_SVD_C{c + 1:02d}"
            z = round(float(Vt_z[c, j]), 6)
            programs[prog_id] = {"beta": z, "se": None}
        result[gene] = {"programs": programs}

    return result


def map_disease_to_fingerprints(
    disease_de_dict: dict[str, float],
    dataset_id: str,
    n_bootstrap: int | None = None,
    source_variant: str = "",
    min_gene_overlap: int | None = None,
) -> dict:
    """
    Match a disease DE vector to Perturb-seq fingerprints via Pearson correlation.

    Replaces CAD ridge regression (R²=1.0 overfitting) with a principled
    probabilistic approach: correlation + bootstrap SE per perturbation.

    Positive r  → KO mimics disease state (disease-promoting gene)
    Negative r  → KO reverses disease state (therapeutic target candidate)

    Args:
        disease_de_dict: {gene_symbol: log2FC} for the disease vs healthy DE
        dataset_id:      Perturb-seq dataset to match against
        n_bootstrap:     Bootstrap iterations (default: FINGERPRINT_N_BOOTSTRAP)
        source_variant:  Source variant of fingerprints ("" = primary)

    Returns dict with top therapeutic KOs (most negative r) and top disease-mimics.
    Saves full results to data/perturbseq/{dataset_id}/disease_fingerprint_match.json.
    """
    try:
        import numpy as _np
    except ImportError as e:
        return {"error": f"Missing dependency: {e}"}

    from config.scoring_thresholds import FINGERPRINT_MIN_GENE_OVERLAP, FINGERPRINT_N_BOOTSTRAP

    fp_variant = f"fingerprint{'_' + source_variant if source_variant else ''}"
    fingerprints = _load_cached_signatures(dataset_id, fp_variant)
    if not fingerprints:
        return {"error": f"No fingerprints for {dataset_id!r}. Run preprocess_rna_fingerprints first."}

    n_boot = n_bootstrap if n_bootstrap is not None else FINGERPRINT_N_BOOTSTRAP
    _min_overlap = min_gene_overlap if min_gene_overlap is not None else FINGERPRINT_MIN_GENE_OVERLAP
    disease_genes = list(disease_de_dict.keys())

    results: list[dict] = []
    for pert, fp_dict in fingerprints.items():
        shared = [g for g in disease_genes if g in fp_dict]
        if len(shared) < _min_overlap:
            continue
        d = _np.array([disease_de_dict[g] for g in shared], dtype=_np.float64)
        f = _np.array([fp_dict[g] for g in shared], dtype=_np.float64)
        if _np.std(d) < 1e-8 or _np.std(f) < 1e-8:
            continue
        r = float(_np.corrcoef(d, f)[0, 1])

        # Bootstrap SE — resample 80% of shared genes
        rng = _np.random.default_rng(42)
        n_shared = len(shared)
        n_sample = max(int(n_shared * 0.8), 2)
        boot_rs: list[float] = []
        for _ in range(n_boot):
            idx = rng.choice(n_shared, size=n_sample, replace=False)
            if _np.std(d[idx]) < 1e-8 or _np.std(f[idx]) < 1e-8:
                continue
            boot_rs.append(float(_np.corrcoef(d[idx], f[idx])[0, 1]))
        r_se = float(_np.std(boot_rs)) if boot_rs else float("nan")
        z = r / r_se if r_se > 0 and _np.isfinite(r_se) else None

        results.append({
            "gene_ko":      pert,
            "r":            round(r, 6),
            "r_se":         round(r_se, 6) if _np.isfinite(r_se) else None,
            "n_shared":     n_shared,
            "z":            round(z, 4) if z is not None else None,
        })

    results.sort(key=lambda x: x["r"], reverse=True)

    out_path = _CACHE_DIR / dataset_id / "disease_fingerprint_match.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump({"dataset_id": dataset_id, "n_matched": len(results), "results": results}, fh)

    log.info(
        "Disease fingerprint match: %d KOs matched for %s → %s",
        len(results), dataset_id, out_path,
    )
    return {
        "dataset_id":            dataset_id,
        "n_perturbations_matched": len(results),
        "top_therapeutic_kos":   sorted(results, key=lambda x: x["r"])[:10],
        "top_disease_mimics":    results[:10],
        "output_path":           str(out_path),
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
    elif acc_type == "s3_de_stats_h5ad":
        url      = access.get("s3_url")
        filename = Path(url).name if url else f"{dataset_id}.h5ad"
        if not url:
            return {"error": "s3_url not set in registry"}
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

    if h5ad_fmt == "czi_de_stats":
        result = preprocess_czi_de_stats(dataset_id, h5ad_path)
    elif h5ad_fmt == "replogle_bulk":
        result = preprocess_replogle_bulk(dataset_id, h5ad_path, top_k, min_abs_log2fc)
    else:
        result = preprocess_h5ad(dataset_id, h5ad_path, top_k, min_abs_log2fc)

    if delete_raw and "error" not in result:
        Path(h5ad_path).unlink(missing_ok=True)
        log.info("Deleted raw h5ad: %s", h5ad_path)
        result["raw_deleted"] = True

    return result


# ---------------------------------------------------------------------------
# Activation-stratified NMF: precompute per-program Stim8hr/Rest bias
# ---------------------------------------------------------------------------

def precompute_activation_biases(
    dataset_id: str,
    h5ad_path: str | Path,
) -> dict:
    """
    Extract per-gene activation bias (Stim8hr / Rest n_regulators) from CZI varm.

    The GWCD4i.DE_stats.h5ad varm contains `measured_genes_stats_Stim8hr` and
    `measured_genes_stats_Rest`, each with `n_regulators` per measured gene.
    We save {gene_symbol → {"n_reg_stim8hr": float, "n_reg_rest": float}} to
    data/perturbseq/{dataset_id}/gene_activation_biases.json.

    This is a one-time preprocessing step. The per-program bias is computed at
    runtime in load_program_activation_biases() given NMF program gene sets.

    Args:
        dataset_id:  Registry key (must be "czi_2025_cd4t_perturb")
        h5ad_path:   Path to GWCD4i.DE_stats.h5ad
    """
    try:
        import anndata as ad
    except ImportError:
        return {"error": "anndata not installed; run: pip install anndata"}

    path = Path(h5ad_path)
    if not path.exists():
        return {"error": f"File not found: {path}"}

    log.info("Loading CZI h5ad for activation bias computation: %s", path)
    adata = ad.read_h5ad(str(path))

    varm = adata.varm
    if "measured_genes_stats_Stim8hr" not in varm or "measured_genes_stats_Rest" not in varm:
        available = list(varm.keys())
        return {"error": f"varm missing activation stats. Available: {available}"}

    stim_stats = varm["measured_genes_stats_Stim8hr"]
    rest_stats  = varm["measured_genes_stats_Rest"]

    # varm entries are DataFrames or numpy structured arrays with column "n_regulators"
    import numpy as _np, pandas as _pd

    def _extract_n_reg(stats_obj) -> _np.ndarray | None:
        if isinstance(stats_obj, _pd.DataFrame):
            return stats_obj["n_regulators"].values if "n_regulators" in stats_obj.columns else None
        if hasattr(stats_obj, "dtype") and "n_regulators" in stats_obj.dtype.names:
            return stats_obj["n_regulators"]
        if hasattr(stats_obj, "__getitem__"):
            try:
                return _np.asarray(stats_obj["n_regulators"])
            except Exception:
                pass
        # Fallback: try column 0
        if hasattr(stats_obj, "shape") and len(stats_obj.shape) == 2:
            return stats_obj[:, 0]
        return None

    n_reg_stim = _extract_n_reg(stim_stats)
    n_reg_rest  = _extract_n_reg(rest_stats)

    if n_reg_stim is None or n_reg_rest is None:
        return {"error": "Could not extract n_regulators from varm stats"}

    var_symbols = list(adata.var["gene_name"] if "gene_name" in adata.var.columns else adata.var_names)

    biases: dict[str, dict] = {}
    for i, sym in enumerate(var_symbols):
        stim_val = float(n_reg_stim[i]) if i < len(n_reg_stim) else 0.0
        rest_val  = float(n_reg_rest[i])  if i < len(n_reg_rest)  else 0.0
        biases[sym] = {"n_reg_stim8hr": stim_val, "n_reg_rest": rest_val}

    out_path = _CACHE_DIR / dataset_id / "gene_activation_biases.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(biases, f)

    log.info("Saved activation biases for %d genes → %s", len(biases), out_path)
    return {
        "dataset_id":  dataset_id,
        "n_genes":     len(biases),
        "output_path": str(out_path),
    }


# In-process cache for gene activation biases
_ACTIVATION_BIAS_CACHE: dict[str, dict[str, dict]] = {}


def load_program_activation_biases(
    dataset_id: str,
    program_gene_sets: dict[str, list[str] | set[str]],
) -> dict[str, float] | None:
    """
    Compute per-program activation bias from precomputed gene-level varm data.

    activation_bias_P = mean(n_reg_Stim8hr_g for g ∈ P) / mean(n_reg_Rest_g for g ∈ P)

    Programs with activation_bias < TIMEPOINT_ACTIVATION_BIAS_MIN (1.5) are
    Rest-dominant and will be discounted in compute_ota_gamma.

    Returns None if gene_activation_biases.json has not been precomputed.
    """
    global _ACTIVATION_BIAS_CACHE

    if dataset_id not in _ACTIVATION_BIAS_CACHE:
        cache_path = _CACHE_DIR / dataset_id / "gene_activation_biases.json"
        if not cache_path.exists():
            return None
        with open(cache_path) as f:
            _ACTIVATION_BIAS_CACHE[dataset_id] = json.load(f)

    gene_biases = _ACTIVATION_BIAS_CACHE[dataset_id]

    program_biases: dict[str, float] = {}
    for prog, gene_set in program_gene_sets.items():
        stim_vals, rest_vals = [], []
        for g in gene_set:
            gb = gene_biases.get(g) or gene_biases.get(g.upper())
            if gb:
                stim_vals.append(gb["n_reg_stim8hr"])
                rest_vals.append(gb["n_reg_rest"])
        if len(stim_vals) >= 3:  # need at least 3 genes with varm data
            mean_stim = sum(stim_vals) / len(stim_vals)
            mean_rest  = sum(rest_vals)  / len(rest_vals)
            if mean_rest > 0:
                program_biases[prog] = round(mean_stim / mean_rest, 3)
            elif mean_stim > 0:
                program_biases[prog] = 99.0  # pure Stim8hr — maximally activated
            else:
                program_biases[prog] = 1.0   # no regulators in either condition

    return program_biases if program_biases else None


# ---------------------------------------------------------------------------
# Variant-aware program beta helper (for cross-timepoint concordance checks)
# ---------------------------------------------------------------------------

def get_program_beta_from_variant(
    gene_symbol: str,
    program_gene_set: list[str] | set[str],
    dataset_id: str,
    variant: str,
) -> float | None:
    """
    Compute program beta from a specific signature variant (e.g. "Stim48hr", "delta").

    Used by ota_beta_estimation to check cross-timepoint concordance without
    exposing variant logic to callers.

    Returns None if the variant signatures are not cached or gene not found.
    """
    sigs = _load_cached_signatures(dataset_id, variant=variant)
    if sigs is None:
        return None
    gene = gene_symbol.upper()
    sig = sigs.get(gene) or sigs.get(gene.lower())
    if sig is None:
        return None
    return _compute_program_beta(sig, set(program_gene_set))


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
      SLE  → czi_2025_cd4t_perturb → replogle_2022_k562
      DED  → czi_2025_cd4t_perturb → replogle_2022_k562
      else → replogle_2022_k562 (GENERIC)

    Args:
        gene_symbol:     Gene to query (e.g. "PCSK9")
        disease_context: Disease key — "CAD" | "SLE" | "DED"
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

