"""
pipelines/ldsc/eqtl_local.py — Local eQTL summary statistics.

Download the full eQTL Catalogue summary-stats file once from EBI FTP, stream
it to build a compact per-gene index, then never call the per-gene REST API
again.  A gene index is ~50 KB; the full sumstats are retained for re-indexing.

Workflow:
    1. download_dataset(disease_key)          — fetch .tsv.gz from EBI FTP
    2. build_gene_index(disease_key, genes)   — stream file → compact JSON index
    3. load_gene_index(disease_key)           — {gene_symbol → top_eqtl_dict}

Top eQTL = the row with the lowest p-value for each gene in the target set.
"""
from __future__ import annotations

import gzip
import json
import logging
import sys
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_ROOT        = Path(__file__).parent.parent.parent
_SUMSTATS_DIR = _ROOT / "data" / "eqtl" / "sumstats"
_INDEX_DIR    = _ROOT / "data" / "eqtl" / "indices"
_FTP_BASE     = "https://ftp.ebi.ac.uk/pub/databases/spot/eQTL/sumstats"

# Primary dataset per disease.
# FTP path: {_FTP_BASE}/{study_id}/{dataset_id}/{dataset_id}.all.tsv.gz
# IDs verified from https://www.ebi.ac.uk/eqtl/api/v2/datasets/
DISEASE_EQTL_DATASETS: dict[str, dict[str, str]] = {
    "CAD": {
        "study_id":     "QTS000015",   # GTEx
        "dataset_id":   "QTD000136",   # artery (coronary), n=213
        "tissue_label": "artery (coronary)",
    },
    "RA": {
        "study_id":     "QTS000038",   # OneK1K (Yazar2022)
        "dataset_id":   "QTD000612",   # CD4+ T cell naive, n=981
        "tissue_label": "CD4+ T cell",
    },
    "SLE": {
        "study_id":     "QTS000038",   # OneK1K (Yazar2022)
        "dataset_id":   "QTD000612",   # CD4+ T cell naive, n=981
        "tissue_label": "CD4+ T cell",
    },
}


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def ftp_url(disease_key: str) -> str:
    cfg = DISEASE_EQTL_DATASETS[disease_key.upper()]
    sid = cfg["study_id"]
    did = cfg["dataset_id"]
    return f"{_FTP_BASE}/{sid}/{did}/{did}.all.tsv.gz"


def sumstats_path(disease_key: str) -> Path:
    cfg = DISEASE_EQTL_DATASETS[disease_key.upper()]
    return _SUMSTATS_DIR / f"{cfg['dataset_id']}.all.tsv.gz"


def download_dataset(disease_key: str) -> Path:
    """Download full eQTL sumstats for disease_key.  No-op if already present."""
    if disease_key.upper() not in DISEASE_EQTL_DATASETS:
        raise ValueError(f"No eQTL dataset configured for {disease_key}")

    url  = ftp_url(disease_key)
    dest = sumstats_path(disease_key)

    if dest.exists() and dest.stat().st_size > 100_000:
        log.info("eQTL sumstats already present: %s", dest.name)
        return dest

    _SUMSTATS_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Downloading eQTL sumstats: %s", url)
    try:
        import urllib.request
        def _progress(count: int, block: int, total: int) -> None:
            if total > 0:
                pct = count * block * 100 // total
                sys.stdout.write(f"\r  {pct}% of {total // 1024 // 1024} MB")
                sys.stdout.flush()
        urllib.request.urlretrieve(url, str(dest), _progress)
        sys.stdout.write("\n")
    except Exception as exc:
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"eQTL download failed for {url}: {exc}") from exc

    log.info("Downloaded: %s (%.0f MB)", dest.name, dest.stat().st_size / 1e6)
    return dest


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------

def index_path(disease_key: str) -> Path:
    cfg = DISEASE_EQTL_DATASETS[disease_key.upper()]
    return _INDEX_DIR / f"{disease_key.upper()}_{cfg['dataset_id']}_top_eqtls.json"


def build_gene_index(
    disease_key: str,
    genes: list[str],
    force: bool = False,
) -> dict[str, dict]:
    """
    Stream the downloaded sumstats file and extract the top (lowest-p) eQTL
    for each gene in *genes*.

    Saves a compact JSON index keyed by gene symbol.  Re-uses existing index
    unless force=True.

    Returns {gene_symbol → {rsid, beta, se, pvalue, chromosome, position}}.
    """
    idx_file = index_path(disease_key)
    if idx_file.exists() and not force:
        log.info("Gene index already built: %s", idx_file.name)
        return load_gene_index(disease_key)

    ss_file = sumstats_path(disease_key)
    if not ss_file.exists():
        raise FileNotFoundError(
            f"eQTL sumstats not found: {ss_file}\n"
            f"Run: python -m pipelines.ldsc.eqtl_local download {disease_key}"
        )

    # Resolve gene symbols → Ensembl IDs (cached via api_cache)
    ensembl_map = _resolve_genes_to_ensembl(genes)
    if not ensembl_map:
        raise RuntimeError("Could not resolve any genes to Ensembl IDs")
    ensembl_to_symbol = {v: k for k, v in ensembl_map.items()}

    log.info(
        "Scanning eQTL sumstats for %d genes (%d resolved to Ensembl)…",
        len(genes), len(ensembl_map),
    )

    # best[ensembl_id] = row dict with lowest pvalue seen so far
    best: dict[str, dict[str, Any]] = {}
    target_ids = set(ensembl_map.values())

    # eQTL Catalogue all.tsv.gz columns (tab-separated, first row = header):
    # molecular_trait_id  chromosome  position  ref  alt  variant  ma_samples
    # maf  pvalue  beta  se  type  ac  an  r2  molecular_trait_object_id
    # gene_id  median_tpm  rsid
    opener = gzip.open if str(ss_file).endswith(".gz") else open
    col: dict[str, int] = {}
    n_rows = 0

    try:
        with opener(ss_file, "rt") as fh:
            for raw in fh:
                line = raw.rstrip("\n")
                if not col:
                    col = {c: i for i, c in enumerate(line.split("\t"))}
                    continue
                parts = line.split("\t")
                try:
                    gene_id = parts[col["gene_id"]]
                except IndexError:
                    continue
                if gene_id not in target_ids:
                    continue
                try:
                    pval = float(parts[col["pvalue"]])
                    beta = float(parts[col["beta"]])
                    se   = float(parts[col["se"]])
                    rsid = parts[col["rsid"]]
                    chrom = parts[col["chromosome"]]
                    pos   = int(parts[col["position"]])
                except (KeyError, ValueError, IndexError):
                    continue
                n_rows += 1
                if gene_id not in best or pval < best[gene_id]["pvalue"]:
                    best[gene_id] = {
                        "rsid":       rsid,
                        "beta":       beta,
                        "se":         se,
                        "pvalue":     pval,
                        "chromosome": chrom,
                        "position":   pos,
                        "gene_id":    gene_id,
                    }
    except Exception as exc:
        raise RuntimeError(f"Failed to stream eQTL sumstats: {exc}") from exc

    # Convert to symbol-keyed dict
    index: dict[str, dict] = {
        ensembl_to_symbol[eid]: row
        for eid, row in best.items()
        if eid in ensembl_to_symbol
    }

    log.info(
        "eQTL index built: %d / %d genes have eQTL signal (%d rows scanned)",
        len(index), len(genes), n_rows,
    )

    _INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with open(idx_file, "w") as fh:
        json.dump(index, fh, indent=2)
    log.info("Gene index saved: %s", idx_file)

    return index


def load_gene_index(disease_key: str) -> dict[str, dict]:
    """Load pre-built gene index.  Returns {} if not yet built."""
    idx_file = index_path(disease_key)
    if not idx_file.exists():
        return {}
    try:
        with open(idx_file) as fh:
            return json.load(fh)
    except Exception as exc:
        log.warning("Failed to load eQTL gene index for %s: %s", disease_key, exc)
        return {}


def gene_index_available(disease_key: str) -> bool:
    return index_path(disease_key.upper()).exists()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_HGNC_URL      = (
    "https://www.genenames.org/cgi-bin/download/custom"
    "?col=gd_app_sym&col=md_ensembl_id"
    "&status=Approved&hgnc_dbtag=on&order_by=gd_app_sym_sort"
    "&format=text&submit=submit"
)
_HGNC_CACHE    = _ROOT / "data" / "eqtl" / "hgnc_symbol_ensembl.json"


def _load_hgnc_mapping() -> dict[str, str]:
    """
    Return {gene_symbol → ensembl_id} from HGNC complete set.

    Downloads once from HGNC (genenames.org) — a single HTTP request returning
    ~45k approved symbols with their Ensembl IDs.  Cached locally as JSON.
    No per-gene API calls needed.
    """
    if _HGNC_CACHE.exists():
        try:
            with open(_HGNC_CACHE) as fh:
                mapping = json.load(fh)
            log.info("HGNC symbol→Ensembl loaded from cache (%d entries)", len(mapping))
            return mapping
        except Exception:
            pass

    log.info("Downloading HGNC symbol→Ensembl mapping (one-time)…")
    import urllib.request
    try:
        with urllib.request.urlopen(_HGNC_URL, timeout=30) as resp:
            text = resp.read().decode("utf-8")
    except Exception as exc:
        raise RuntimeError(f"HGNC download failed: {exc}") from exc

    mapping: dict[str, str] = {}
    for line in text.splitlines()[1:]:   # skip header
        parts = line.split("\t")
        if len(parts) >= 2 and parts[1].startswith("ENSG"):
            mapping[parts[0]] = parts[1]

    _HGNC_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(_HGNC_CACHE, "w") as fh:
        json.dump(mapping, fh)
    log.info("HGNC mapping downloaded: %d symbols → %s", len(mapping), _HGNC_CACHE.name)
    return mapping


def _resolve_genes_to_ensembl(genes: list[str]) -> dict[str, str]:
    """Return {gene_symbol → ensembl_id} from the local HGNC cache."""
    hgnc = _load_hgnc_mapping()
    mapping = {g: hgnc[g] for g in genes if g in hgnc}
    missing = len(genes) - len(mapping)
    if missing:
        log.debug("%d / %d genes not in HGNC mapping", missing, len(genes))
    log.info("Ensembl resolve: %d / %d genes mapped (HGNC)", len(mapping), len(genes))
    return mapping


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cmd     = sys.argv[1] if len(sys.argv) > 1 else "help"
    disease = sys.argv[2].upper() if len(sys.argv) > 2 else "CAD"

    if cmd == "download":
        p = download_dataset(disease)
        print(f"Downloaded: {p}")

    elif cmd == "build_index":
        # Requires the SVD gene set to know which genes to index
        import numpy as np
        from pipelines.ldsc.runner import _DISEASE_SVD_DATASET  # type: ignore[attr-defined]
        dataset_id = _DISEASE_SVD_DATASET.get(disease)
        if not dataset_id:
            print(f"No SVD dataset for {disease}"); sys.exit(1)
        npz = _ROOT / "data" / "perturbseq" / dataset_id / "svd_loadings.npz"
        gene_names = list(np.load(npz)["gene_names"])
        idx = build_gene_index(disease, gene_names, force=True)
        print(f"Index built: {len(idx)} genes with eQTL signal")

    elif cmd == "status":
        for dk in DISEASE_EQTL_DATASETS:
            ss = sumstats_path(dk)
            ix = index_path(dk)
            print(
                f"{dk}: sumstats={'OK' if ss.exists() else 'MISSING'} "
                f"({ss.stat().st_size // 1_000_000} MB)" if ss.exists() else f"{dk}: sumstats=MISSING",
            )
            print(f"    index={'OK' if ix.exists() else 'MISSING'} — {ix}")

    else:
        print("Usage:")
        print("  python -m pipelines.ldsc.eqtl_local download   CAD|RA")
        print("  python -m pipelines.ldsc.eqtl_local build_index CAD|RA")
        print("  python -m pipelines.ldsc.eqtl_local status")
