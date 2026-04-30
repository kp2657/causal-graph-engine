"""
scripts/download_wes_burden.py — Download UKB WES rare-variant gene burden summary
statistics from the GWAS Catalog (EBI FTP) for CAD and RA.

Data source: Backman et al. 2021 "Exome sequencing and analysis of 454,787 UK Biobank
participants" (Nature). UKB 450k WES gene-level burden tests deposited in the GWAS
Catalog under accessions listed below.

Burden test markers (Model column):
  M1.* = pLoF only (singletons through MAF<1%)
  M3.* = pLoF + damaging missense (various MAF thresholds)
  Numbers indicate MAF threshold: 0001=<0.001%, 001=<0.01%, 01=<0.1%, 1=<1%

Usage:
    conda run -n causal-graph python scripts/download_wes_burden.py
    conda run -n causal-graph python scripts/download_wes_burden.py --build-lookup
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import math
import os
import urllib.request
from pathlib import Path

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

_ROOT    = Path(__file__).parent.parent
_WES_DIR = _ROOT / "data" / "wes"

# GWAS Catalog accession IDs for Backman et al. 2021 gene burden results.
# One file per phenotype; each file contains all genes × all burden markers.
_STUDIES: dict[str, tuple[str, str]] = {
    # label → (GWAS Catalog accession, phenotype description)
    "CAD_I25":   ("GCST90083968", "ICD10 I25 Chronic ischemic heart disease — gene burden"),
    "RA_M05":    ("GCST90084373", "ICD10 M05 Seropositive RA — gene burden"),
    "RA_M06":    ("GCST90084378", "ICD10 M06 Other RA — gene burden"),
    "RA_SR_M06": ("GCST90081860", "Self-report + ICD10 M06 RA combined — gene burden"),
}

# Disease → primary + fallback study labels (first hit wins)
_DISEASE_STUDIES: dict[str, list[str]] = {
    "CAD": ["CAD_I25"],
    "RA":  ["RA_M05", "RA_M06", "RA_SR_M06"],
}

# Burden markers in preference order for directional signal.
# M3.01 = pLoF + damaging missense MAF<0.1% (well-powered, commonly reported).
# M1.001 = strict pLoF MAF<0.01% (cleanest causal interpretation).
_PREFERRED_MARKERS = ["M3.01", "M1.001", "M1.01", "M3.001", "M3.1", "M1.1", "M3.0001", "M1.0001"]


def _ftp_url(accession: str) -> str:
    n = int(accession.replace("GCST", ""))
    range_start = (n // 1000) * 1000 + 1
    range_end = range_start + 999
    return (
        f"https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/"
        f"GCST{range_start:08d}-GCST{range_end:08d}/{accession}/{accession}_buildGRCh38.tsv.gz"
    )


def download_all(force: bool = False) -> None:
    """Download all burden files to data/wes/."""
    _WES_DIR.mkdir(parents=True, exist_ok=True)
    for label, (acc, desc) in _STUDIES.items():
        dest = _WES_DIR / f"{acc}.tsv.gz"
        if dest.exists() and not force:
            log.info("Already present: %s (%s)", dest.name, label)
            continue
        url = _ftp_url(acc)
        log.info("Downloading %s (%s) ...", label, acc)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=300) as r, open(dest, "wb") as f:
            data = r.read()
            f.write(data)
        log.info("  → %s bytes saved to %s", f"{len(data):,}", dest)


def build_lookup(disease: str | None = None) -> None:
    """
    Parse downloaded TSV.gz files and build per-disease gene→burden lookup JSONs.

    Output: data/wes/{disease}_burden.json
    Format: {gene_symbol: {burden_beta, burden_se, burden_p, marker, n_variants, odds_ratio}}

    For each gene, selects the preferred marker (M3.01 > M1.001 > ...) from the best
    available study. burden_beta = log(odds_ratio) so sign indicates direction:
      + = LoF carriers have HIGHER disease risk (gene is risk gene)
      - = LoF carriers have LOWER disease risk (gene is protective, LoF is causal)
    """
    import pandas as pd

    diseases = list(_DISEASE_STUDIES) if disease is None else [disease.upper()]

    for dis in diseases:
        study_labels = _DISEASE_STUDIES.get(dis)
        if not study_labels:
            log.warning("No studies configured for disease %s", dis)
            continue

        # Merge all studies for this disease, taking best (lowest p) per gene
        gene_records: dict[str, dict] = {}

        for label in study_labels:
            acc, _ = _STUDIES[label]
            path = _WES_DIR / f"{acc}.tsv.gz"
            if not path.exists():
                log.warning("Missing %s — run download_all() first", path)
                continue

            log.info("Parsing %s for %s ...", path.name, dis)
            with gzip.open(path, "rt") as f:
                df = pd.read_csv(f, sep="\t", low_memory=False)

            # Name format: GENESYMBOL(ENSEMBL_ID).GENE.MARKER
            # Extract gene symbol and marker
            df["gene_symbol"] = df["Name"].str.extract(r"^([^(]+)\(")
            df["marker"]      = df["Name"].str.extract(r"\.GENE\.(.+)$")

            # Keep only preferred markers, ordered
            marker_order = {m: i for i, m in enumerate(_PREFERRED_MARKERS)}
            df["marker_rank"] = df["marker"].map(marker_order)
            df = df.dropna(subset=["marker_rank", "gene_symbol", "odds_ratio", "p_value"])
            df["marker_rank"] = df["marker_rank"].astype(int)

            # For each gene: keep the preferred marker (lowest rank), then within that lowest p
            df_sorted = df.sort_values(["gene_symbol", "marker_rank", "p_value"])
            best = df_sorted.groupby("gene_symbol").first().reset_index()

            for _, row in best.iterrows():
                gene = row["gene_symbol"]
                p    = float(row["p_value"])
                or_  = float(row["odds_ratio"])
                se   = float(row["standard_error"]) if "standard_error" in row else None
                beta = math.log(or_) if or_ > 0 else None

                existing = gene_records.get(gene)
                if existing is None or p < existing["burden_p"]:
                    gene_records[gene] = {
                        "burden_beta":   round(beta, 6) if beta is not None else None,
                        "burden_se":     round(se, 6)   if se   is not None else None,
                        "burden_p":      p,
                        "odds_ratio":    round(or_, 6),
                        "marker":        row["marker"],
                        "study":         acc,
                    }

        out_path = _WES_DIR / f"{dis}_burden.json"
        with open(out_path, "w") as f:
            json.dump(gene_records, f)
        log.info("Built lookup for %s: %d genes → %s", dis, len(gene_records), out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and build UKB WES burden lookups")
    parser.add_argument("--force",         action="store_true", help="Re-download even if files exist")
    parser.add_argument("--build-lookup",  action="store_true", help="Build gene lookup JSONs from downloaded files")
    parser.add_argument("--disease",       default=None, help="Build lookup for specific disease only (CAD, RA)")
    args = parser.parse_args()

    download_all(force=args.force)
    if args.build_lookup:
        build_lookup(args.disease)
    else:
        log.info("Files downloaded. Run with --build-lookup to build gene JSONs.")
