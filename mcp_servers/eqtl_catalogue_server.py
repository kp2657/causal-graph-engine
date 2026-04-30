"""
mcp_servers/eqtl_catalogue_server.py — eQTL Catalogue REST API v2.

Provides access to:
  1. Single-cell eQTLs (quant_method=ge, cell-type resolved):
       - OneK1K (study_label="OneK1K", Yazar 2022) — 14 PBMC cell types, 982 donors
       - BLUEPRINT (study_label="BLUEPRINT") — monocytes/neutrophils/T cells
       - Perez_2022 — lupus/SLE PBMC cell types
  2. Protein QTLs / pQTL (quant_method=aptamer):
       - Sun_2018 INTERVAL — 3,301 plasma proteins via SOMAscan (eQTL Catalogue v2)
       NOTE: UKB-PPP (Sun 2023) and deCODE (Ferkingstad 2021) are NOT yet in eQTL
       Catalogue v2. For CFH Y402H (rs1061170), Sun_2018 has low power (best p≈0.017).
  3. Bulk tissue eQTLs (quant_method=ge) — supplement to GTEx

All API calls use the public EBI eQTL Catalogue REST v2 endpoint.
No authentication required.

Public API
----------
    get_sc_eqtl(gene, cell_type, study_label)
        -> dict  {gene, cell_type, eqtls: [{rsid, beta, se, pvalue, tissue, ...}]}

    get_pqtl_instruments(gene, protein_name, study_label)
        -> dict  {gene, protein, pqtls: [{rsid, beta, se, pvalue, study, ...}]}

    resolve_gene_to_ensembl(gene_symbol)
        -> str | None   ENSG ID for the gene symbol

eQTL Catalogue API docs: https://www.ebi.ac.uk/eqtl/api/v2/docs
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx

from pipelines.api_cache import api_cached

try:
    import fastmcp
    mcp = fastmcp.FastMCP("eqtl-catalogue-server")
    _tool = mcp.tool()
except ImportError:
    def _tool(fn=None, **_):
        return fn if fn is not None else (lambda f: f)
    mcp = None

# ---------------------------------------------------------------------------
# API constants
# ---------------------------------------------------------------------------

EQTL_CATALOGUE_V2  = "https://www.ebi.ac.uk/eqtl/api/v2"
ENSEMBL_REST        = "https://rest.ensembl.org"

# polite rate limit
_DELAY = 0.3

# pQTL study labels in the eQTL Catalogue
_PQTL_STUDIES = {
    "INTERVAL_Sun2018":    "Sun2018",        # 3,622 SOMAscan proteins
    "INTERVAL_Folkersen":  "Folkersen2020",  # 90 Olink proteins
    "UKB_PPP":             "Sun2023",        # 2,923 Olink proteins, 54k participants
    "deCODE":              "Ferkingstad2021",# 4,719 SOMAscan proteins
    "ARIC":                "Suhre2017",      # 1,000 proteins, ARIC cohort
}

# sc-eQTL study labels in the eQTL Catalogue v2 (use study_label values from API)
_SCEQTL_STUDIES = {
    "OneK1K":    "OneK1K",          # 14 PBMC cell types, 982 donors (Yazar 2022)
    "Blueprint": "BLUEPRINT",       # monocytes/neutrophils/T cells
    "Perez_2022": "Perez_2022",     # SLE lupus PBMC cell types
    "Schmiedel_2018": "Schmiedel_2018",  # 7 immune cell types
}

# Disease → preferred sc-eQTL cell type keywords (matched against tissue_label in eQTL Catalogue).
# OneK1K tissue_labels: "monocyte", "CD16+ monocyte", "CD4+ T cell", "CD8+ T cell",
#   "B cell", "memory B cell", "NK cell", "dendritic cell", "plasmacytoid dendritic cell", etc.
DISEASE_SC_EQTL_CELL_TYPES: dict[str, list[str]] = {
    "CAD":  ["monocyte", "macrophage"],
    "RA":   ["CD4", "B cell", "monocyte"],
}

# Disease → relevant pQTL proteins (gene symbol → protein label in eQTL Catalogue)
# Key proteins where coding variants drive GWAS signal and cis-eQTL is absent/weak
DISEASE_KEY_PQTL_PROTEINS: dict[str, dict[str, str]] = {
    "CAD": {
        "PCSK9":  "PCSK9",
        "APOB":   "APOB",
        "LPA":    "LPA",   # Lp(a); no eQTL — driven by LPA kringle repeat number
        "LDLR":   "LDLR",
        "CRP":    "CRP",   # C-reactive protein — inflammatory marker
    },
    "RA": {
        "IL6":   "IL6",
        "TNF":   "TNF",
        "CRP":   "CRP",
    },
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get(url: str, params: dict | None = None, timeout: float = 10.0) -> dict | list:
    """GET wrapper with polite delay. Strips trailing slash to avoid 307 redirects."""
    time.sleep(_DELAY)
    url = url.rstrip("/")
    resp = httpx.get(
        url,
        params=params,
        headers={"User-Agent": "causal-graph-engine/0.1 mailto:research@example.com"},
        timeout=httpx.Timeout(connect=5.0, read=timeout, write=5.0, pool=5.0),
        follow_redirects=True,
    )
    resp.raise_for_status()
    return resp.json()


def resolve_gene_to_ensembl(gene_symbol: str) -> str | None:
    """
    Resolve gene symbol to Ensembl ENSG ID via Ensembl REST API.
    Returns None on failure (network or unknown symbol).
    """
    try:
        url = f"{ENSEMBL_REST}/lookup/symbol/homo_sapiens/{gene_symbol}"
        data = _get(url, params={"content-type": "application/json", "expand": 0})
        if isinstance(data, dict) and data.get("id"):
            return data["id"]
    except Exception:
        pass
    return None


def _get_datasets(
    quant_method: str,
    study_label: str | None = None,
    tissue_label: str | None = None,
    condition_label: str | None = None,
    size: int = 20,
) -> list[dict]:
    """List eQTL Catalogue datasets matching filters."""
    params: dict[str, Any] = {"quant_method": quant_method, "size": size, "start": 0}
    if study_label:
        params["study_label"] = study_label
    if tissue_label:
        params["tissue_label"] = tissue_label
    if condition_label:
        params["condition_label"] = condition_label
    try:
        url = f"{EQTL_CATALOGUE_V2}/datasets/"
        time.sleep(_DELAY)
        resp = httpx.get(
            url.rstrip("/"),
            params=params,
            headers={"User-Agent": "causal-graph-engine/0.1 mailto:research@example.com"},
            timeout=httpx.Timeout(connect=5.0, read=10.0, write=5.0, pool=5.0),
            follow_redirects=True,
        )
        if resp.status_code == 400 and "No results" in resp.text:
            return []
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict):
            return data.get("results", data.get("datasets", []))
        if isinstance(data, list):
            return data
    except Exception as _e:
        print(f"[WARN] eQTL Catalogue _get_datasets({quant_method!r}, study={study_label!r}, tissue={tissue_label!r}): {_e}")
    return []


def _get_associations(
    dataset_id: str,
    gene_id: str,
    p_upper: float = 5e-5,
    size: int = 100,
) -> list[dict]:
    """Get eQTL associations for a gene in a specific dataset. Client-side p-value filter."""
    params = {"gene_id": gene_id, "size": size}
    try:
        url = f"{EQTL_CATALOGUE_V2}/datasets/{dataset_id}/associations"
        time.sleep(_DELAY)
        resp = httpx.get(
            url.rstrip("/"),
            params=params,
            headers={"User-Agent": "causal-graph-engine/0.1 mailto:research@example.com"},
            timeout=httpx.Timeout(connect=5.0, read=15.0, write=5.0, pool=5.0),
            follow_redirects=True,
        )
        if resp.status_code == 400 and "No results" in resp.text:
            return []
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict):
            rows = data.get("results", data.get("associations", []))
        elif isinstance(data, list):
            rows = data
        else:
            return []
        # Client-side p-value filter (API doesn't support server-side p_upper reliably)
        return [r for r in rows if float(r.get("pvalue", 1.0)) <= p_upper]
    except Exception as _e:
        print(f"[WARN] eQTL Catalogue _get_associations(dataset={dataset_id!r}, gene={gene_id!r}): {_e}")
    return []


def _pick_top_cis_eqtl(
    associations: list[dict],
    gene_id: str,
    window_kb: int = 1000,
) -> dict | None:
    """Return the strongest cis-eQTL (by p-value) within window_kb of the gene."""
    if not associations:
        return None
    # Filter to cis (position within window) when position info available
    cis = []
    for a in associations:
        pos = a.get("position") or a.get("variant_position")
        gene_pos = a.get("gene_start") or a.get("tss_distance")
        if gene_pos is not None and abs(gene_pos) > window_kb * 1000:
            continue
        cis.append(a)
    pool = cis if cis else associations
    # Sort by p-value ascending
    pool = sorted(pool, key=lambda x: float(x.get("pvalue", 1.0)))
    return pool[0] if pool else None


# ---------------------------------------------------------------------------
# Public API — sc-eQTL
# ---------------------------------------------------------------------------

@_tool
@api_cached(ttl_days=30)
def get_sc_eqtl(
    gene: str,
    cell_type: str | None = None,
    study_label: str | None = None,
    disease: str | None = None,
) -> dict:
    """
    Retrieve single-cell eQTL data for a gene from the eQTL Catalogue.

    Cell-type-specific eQTLs capture regulatory effects that are diluted in
    bulk GTEx tissue data. Useful for immune cell programs (OneK1K) where the
    cell-type-of-interest is a minor fraction of bulk tissue.

    Args:
        gene:          Gene symbol (e.g. "CFH", "PCSK9", "IL23R")
        cell_type:     Cell type label (eQTL Catalogue condition_label)
                       e.g. "CD4_positive_alpha_beta_T_cell", "monocyte"
        study_label:   eQTL Catalogue study label (e.g. "Yazar2022" for OneK1K)
                       Default: tries Yazar2022 (OneK1K) first, then BLUEPRINT_SE
        disease:       If provided, infer preferred cell types from
                       DISEASE_SC_EQTL_CELL_TYPES (overrides cell_type)

    Returns:
        {
            "gene":      str,
            "cell_type": str | None,
            "study":     str | None,
            "eqtls":     list[{rsid, beta, se, pvalue, dataset_id, condition_label}],
            "top_eqtl":  dict | None,   # lowest p-value hit
            "n_found":   int,
            "data_source": str,
        }
    """
    # Resolve cell types to try
    cell_types_to_try: list[str | None] = []
    if disease and disease in DISEASE_SC_EQTL_CELL_TYPES:
        cell_types_to_try = DISEASE_SC_EQTL_CELL_TYPES[disease]
    if cell_type:
        cell_types_to_try = [cell_type] + [c for c in cell_types_to_try if c != cell_type]
    if not cell_types_to_try:
        cell_types_to_try = [None]  # try without cell type filter

    # Resolve gene → Ensembl ID
    gene_id = resolve_gene_to_ensembl(gene)
    if gene_id is None:
        return {"gene": gene, "cell_type": cell_type, "study": study_label,
                "eqtls": [], "top_eqtl": None, "n_found": 0,
                "error": "Could not resolve gene symbol to Ensembl ID",
                "data_source": "eQTL Catalogue v2"}

    # Studies to try (OneK1K first, then Blueprint, then all)
    studies = [study_label] if study_label else [
        _SCEQTL_STUDIES["OneK1K"],
        _SCEQTL_STUDIES["Blueprint"],
        _SCEQTL_STUDIES["Perez_2022"],
        _SCEQTL_STUDIES["Schmiedel_2018"],
        None,  # any study
    ]

    all_eqtls: list[dict] = []
    used_study = None
    used_cell_type = None

    for study in studies:
        # Get all datasets for this study (cell type is in tissue_label, not condition_label)
        all_study_datasets = _get_datasets(
            quant_method="ge",
            study_label=study,
            size=50,
        )
        # Filter by cell type keywords against tissue_label
        for ct in cell_types_to_try:
            if ct:
                matched = [
                    ds for ds in all_study_datasets
                    if ct.lower() in (ds.get("tissue_label") or "").lower()
                ]
            else:
                matched = all_study_datasets
            for ds in matched[:4]:  # limit API calls per cell type
                dataset_id = ds.get("dataset_id") or ds.get("id")
                if not dataset_id:
                    continue
                assocs = _get_associations(dataset_id, gene_id, p_upper=0.05, size=20)
                if assocs:
                    for a in assocs:
                        a.setdefault("dataset_id", dataset_id)
                        a.setdefault("condition_label", ds.get("tissue_label", ct))
                        a.setdefault("study_label", ds.get("study_label", study))
                    all_eqtls.extend(assocs)
                    used_study = study or ds.get("study_label")
                    used_cell_type = ct or ds.get("tissue_label")
                    if len(all_eqtls) >= 5:
                        break
            if all_eqtls:
                break
        if all_eqtls:
            break

    top = _pick_top_cis_eqtl(all_eqtls, gene_id) if all_eqtls else None

    return {
        "gene":        gene,
        "ensembl_id":  gene_id,
        "cell_type":   used_cell_type,
        "study":       used_study,
        "eqtls":       all_eqtls,
        "top_eqtl":    top,
        "n_found":     len(all_eqtls),
        "data_source": f"eQTL Catalogue v2 / {used_study or 'unknown study'}",
    }


# ---------------------------------------------------------------------------
# Public API — pQTL
# ---------------------------------------------------------------------------

@_tool
@api_cached(ttl_days=30)
def get_pqtl_instruments(
    gene: str,
    protein_name: str | None = None,
    study_label: str | None = None,
    disease: str | None = None,
    p_threshold: float = 1e-5,
) -> dict:
    """
    Retrieve protein QTL (pQTL) instruments for a gene from the eQTL Catalogue.

    pQTL instruments are critical for genes where the causal variant is a
    coding mutation (changes protein sequence/abundance) rather than a
    regulatory variant detected by expression QTL. Key examples:
      - CFH Y402H (rs1061170): AMD — no cis-eQTL in GTEx, but strong pQTL
        in INTERVAL/UKB-PPP (plasma CFH levels differ by ~30% per allele)
      - LPA: CAD — Lp(a) levels driven by kringle repeat number; pQTL, not eQTL
      - TREM2 R47H: AD — coding variant; pQTL in cerebrospinal fluid proteomics

    Args:
        gene:           Gene symbol (e.g. "CFH", "LPA", "TREM2")
        protein_name:   Protein name in the pQTL study (default: same as gene)
        study_label:    eQTL Catalogue study label (default: tries Sun2023 UKB-PPP,
                        then Sun2018 INTERVAL, then all pQTL studies)
        disease:        If provided, use disease-specific protein priority list
        p_threshold:    pQTL significance threshold (default 1e-5 for cis-pQTL)

    Returns:
        {
            "gene":       str,
            "protein":    str | None,
            "pqtls":      list[{rsid, beta, se, pvalue, study, ...}],
            "top_pqtl":   dict | None,
            "n_found":    int,
            "data_source": str,
        }
    """
    protein = protein_name or gene

    # Resolve gene → Ensembl ID
    gene_id = resolve_gene_to_ensembl(gene)
    # For pQTL we also try the protein name as molecular_trait_id
    molecular_trait_id = protein.upper()

    # Study priority
    studies = [study_label] if study_label else [
        _PQTL_STUDIES["UKB_PPP"],         # largest, most comprehensive
        _PQTL_STUDIES["INTERVAL_Sun2018"], # second largest
        _PQTL_STUDIES["deCODE"],
        _PQTL_STUDIES["ARIC"],
        None,  # any pQTL study
    ]

    all_pqtls: list[dict] = []
    used_study = None

    for study in studies:
        datasets = _get_datasets(
            quant_method="aptamer",  # eQTL Catalogue v2 uses 'aptamer' for plasma pQTL
            study_label=study,
            size=20,
        )
        # All aptamer datasets measure plasma proteins — take all of them
        protein_datasets = datasets if datasets else []
        if not protein_datasets:
            protein_datasets = datasets[:5]

        for ds in protein_datasets[:3]:
            dataset_id = ds.get("dataset_id") or ds.get("id")
            if not dataset_id:
                continue
            # Try by gene_id first, then by molecular_trait_id
            lookup_id = gene_id or molecular_trait_id
            if lookup_id:
                assocs = _get_associations(dataset_id, lookup_id, p_upper=p_threshold, size=20)
                if assocs:
                    for a in assocs:
                        a.setdefault("dataset_id", dataset_id)
                        a.setdefault("study_label", ds.get("study_label", study))
                        a.setdefault("protein", protein)
                    all_pqtls.extend(assocs)
                    used_study = study or ds.get("study_label")
                    if len(all_pqtls) >= 5:
                        break
        if all_pqtls:
            break

    top = _pick_top_cis_eqtl(all_pqtls, gene_id or gene) if all_pqtls else None

    return {
        "gene":        gene,
        "ensembl_id":  gene_id,
        "protein":     protein,
        "pqtls":       all_pqtls,
        "top_pqtl":    top,
        "n_found":     len(all_pqtls),
        "data_source": f"eQTL Catalogue v2 / pQTL / {used_study or 'unknown study'}",
    }


# ---------------------------------------------------------------------------
# Convenience helpers for the beta estimation pipeline
# ---------------------------------------------------------------------------

@api_cached(ttl_days=30)
def get_best_pqtl_for_gene(
    gene: str,
    disease: str | None = None,
) -> dict | None:
    """
    Return the single best (lowest p-value) cis-pQTL for a gene, or None.

    Checks disease-specific protein priority (DISEASE_KEY_PQTL_PROTEINS) first.
    Returns dict with keys: {rsid, beta, se, pvalue, study_label, dataset_id}
    """
    protein = None
    if disease and gene in DISEASE_KEY_PQTL_PROTEINS.get(disease, {}):
        protein = DISEASE_KEY_PQTL_PROTEINS[disease][gene]

    # eQTL Catalogue v2 only has Sun_2018 (n=3,301 aptamer) — low power.
    # Use relaxed threshold 0.05 for known coding variant genes; strict 1e-5 for others.
    is_known_coding_variant_gene = disease and gene in DISEASE_KEY_PQTL_PROTEINS.get(disease, {})
    threshold = 0.05 if is_known_coding_variant_gene else 1e-5

    result = get_pqtl_instruments(gene, protein_name=protein, disease=disease,
                                  p_threshold=threshold)
    return result.get("top_pqtl")


@api_cached(ttl_days=30)
def get_best_sc_eqtl_for_gene(
    gene: str,
    disease: str | None = None,
    cell_type: str | None = None,
) -> dict | None:
    """
    Return the single best (lowest p-value) sc-eQTL for a gene, or None.
    Returns dict with keys: {rsid, beta, se, pvalue, condition_label, study_label}
    """
    result = get_sc_eqtl(gene, cell_type=cell_type, disease=disease)
    return result.get("top_eqtl")


if __name__ == "__main__":
    # Quick smoke test
    print("Testing eQTL Catalogue server...")
    # pQTL test — CFH should have pQTL in INTERVAL/UKB-PPP
    r = get_pqtl_instruments("CFH", disease="AMD")
    print(f"CFH pQTL instruments: n={r['n_found']}, study={r['data_source']}")
    if r["top_pqtl"]:
        top = r["top_pqtl"]
        print(f"  Top pQTL: rsid={top.get('rsid')}, beta={top.get('beta')}, p={top.get('pvalue')}")
    # scEQTL test
    r2 = get_sc_eqtl("IL23R", disease="CAD")
    print(f"IL23R scEQTL: n={r2['n_found']}, cell_type={r2['cell_type']}")
