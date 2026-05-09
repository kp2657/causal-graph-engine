#!/usr/bin/env python3
"""
scripts/sldsc_chromatin_programs.py — Phase I (Engreitz E5)

Replace expression-loading gene sets for S-LDSC with chromatin-based program
annotations.  For each GeneticNMF program, collects ABC-predicted enhancers
linked to program genes, shrinks them to 150 bp around centre (CATLAS
ForVariantOverlap convention), optionally intersects with ATAC peaks, and
writes annotation-ready BED files for S-LDSC.

Usage
-----
python scripts/sldsc_chromatin_programs.py \
    --disease_key ra \
    --dataset_id czi_2025_cd4t_perturb \
    --min_abc_score 0.013 \
    --top_n_genes 100

# Example S-LDSC command for one program:
# python ldsc/ldsc.py \
#     --h2 {sumstats} \
#     --ref-ld-chr {baseline_ld},data/sldsc_chromatin/{disease}/{program} \
#     --w-ld-chr {weights} \
#     --out {out_prefix}
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import sys
from typing import Optional, Union

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_ROOT = pathlib.Path(__file__).parent.parent
_PERTURBSEQ = _ROOT / "data" / "perturbseq"
_ABC_DIR = _ROOT / "data" / "abc"
_ATAC_DIR = _ROOT / "data" / "atac"
_OUT_BASE = _ROOT / "data" / "sldsc_chromatin"
_MANIFEST_DIR = _ROOT / "outputs"

_ABC_DOWNLOAD_URLS: dict[str, str] = {
    "ra": (
        "https://mitra.stanford.edu/engreitz/oak/public/CATLAS_predictions/"
        "ABC_results/T_lymphocyte_2__CD4__/Predictions/"
        "EnhancerPredictionsFull_threshold0.013_self_promoter.tsv"
    ),
    "cad": (
        "https://mitra.stanford.edu/engreitz/oak/public/CATLAS_predictions/"
        "ABC_results/Endothelial_cell_1/Predictions/"
        "EnhancerPredictionsFull_threshold0.013_self_promoter.tsv"
    ),
}

_ATAC_PEAKS: dict[str, str] = {
    "ra": "calderon2019_cd4t_peaks.bed",
    "cad": None,  # no default ATAC peaks for CAD
}

_SHRINK_BP = 150


# ---------------------------------------------------------------------------
# ABC loading
# ---------------------------------------------------------------------------

def _abc_path_for_disease(disease_key: str) -> pathlib.Path:
    return _ABC_DIR / f"{disease_key.upper()}_abc_predictions.tsv"


_NASSER2021_LOCAL = _ABC_DIR / "nasser2021_all.txt.gz"
_NASSER2021_CD4T_FILTER = "CD4"

# Disease keys that should augment CATLAS with Nasser 2021 CD4+ T predictions.
# RA excluded: Nasser 2021 CD4+ T is resting-state; RA GWAS variants enrich in
# stimulation-responsive chromatin (CATLAS = stimulated). Union dilutes τ 3×.
_NASSER2021_AUGMENT_DISEASES: set[str] = set()


def _load_nasser2021_cd4t(min_abc_score: float) -> pd.DataFrame:
    """Load Nasser 2021 CD4+ T cell ABC predictions from local gzipped file."""
    import gzip as _gzip

    if not _NASSER2021_LOCAL.exists():
        log.warning("Nasser 2021 file not found: %s — skipping augmentation", _NASSER2021_LOCAL)
        return pd.DataFrame(columns=["chr", "start", "end", "name", "TargetGene", "ABC.Score"])

    rows = []
    with _gzip.open(_NASSER2021_LOCAL, "rt") as fh:
        header = fh.readline().strip().split("\t")
        col = {c: i for i, c in enumerate(header)}
        chr_i = col["chr"]
        start_i = col["start"]
        end_i = col["end"]
        name_i = col["name"]
        gene_i = col["TargetGene"]
        score_i = col["ABC.Score"]
        ct_i = col["CellType"]
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) <= ct_i:
                continue
            if _NASSER2021_CD4T_FILTER not in parts[ct_i]:
                continue
            try:
                score = float(parts[score_i])
            except ValueError:
                continue
            if score < min_abc_score:
                continue
            rows.append({
                "chr": parts[chr_i],
                "start": int(parts[start_i]),
                "end": int(parts[end_i]),
                "name": parts[name_i],
                "TargetGene": parts[gene_i],
                "ABC.Score": score,
            })

    df = pd.DataFrame(rows)
    log.info("Loaded %d Nasser 2021 CD4+ T rows (min_abc_score=%.4f)", len(df), min_abc_score)
    return df


def load_abc_predictions(disease_key: str, min_abc_score: float) -> pd.DataFrame:
    """
    Load ABC predictions for disease_key.

    For RA, unions CATLAS CD4+ T predictions with Nasser 2021 CD4+ T predictions
    to increase enhancer coverage per program.
    Required output columns: chr, start, end, TargetGene, ABC.Score.
    """
    tsv_path = _abc_path_for_disease(disease_key)
    if not tsv_path.exists():
        log.warning(
            "ABC TSV not found at %s — attempting download from %s",
            tsv_path,
            _ABC_DOWNLOAD_URLS.get(disease_key, "(no URL)"),
        )
        _download_abc(disease_key, tsv_path)

    if not tsv_path.exists():
        raise FileNotFoundError(
            f"ABC predictions not found: {tsv_path}\n"
            f"Download manually:\n  wget -O {tsv_path} {_ABC_DOWNLOAD_URLS.get(disease_key, '')}"
        )

    log.info("Reading ABC TSV: %s", tsv_path)
    df = pd.read_csv(tsv_path, sep="\t", low_memory=False)
    _validate_abc_columns(df, tsv_path)

    before = len(df)
    df = df[df["ABC.Score"] >= min_abc_score].copy()
    log.info("Filtered %d → %d rows (min_abc_score=%.4f)", before, len(df), min_abc_score)

    if disease_key.lower() in _NASSER2021_AUGMENT_DISEASES:
        nasser_df = _load_nasser2021_cd4t(min_abc_score)
        if not nasser_df.empty:
            # Ensure same columns exist in nasser_df for concat
            for col in df.columns:
                if col not in nasser_df.columns:
                    nasser_df[col] = float("nan")
            nasser_df = nasser_df.reindex(columns=df.columns)
            combined = pd.concat([df, nasser_df], ignore_index=True)
            # Deduplicate: same chr/start/end/gene, keep highest ABC.Score
            before_dedup = len(combined)
            combined = (
                combined
                .sort_values("ABC.Score", ascending=False)
                .drop_duplicates(subset=["chr", "start", "end", "TargetGene"])
                .reset_index(drop=True)
            )
            log.info(
                "Union CATLAS+Nasser2021: %d + %d → %d rows after dedup (%d unique enhancer-gene pairs)",
                before, len(nasser_df), before_dedup, len(combined),
            )
            return combined

    return df


def _validate_abc_columns(df: pd.DataFrame, path: pathlib.Path) -> None:
    required = {"chr", "start", "end", "TargetGene", "ABC.Score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"ABC TSV {path} missing columns: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )


def _download_abc(disease_key: str, dest: pathlib.Path) -> None:
    url = _ABC_DOWNLOAD_URLS.get(disease_key)
    if not url:
        return
    try:
        import urllib.request

        dest.parent.mkdir(parents=True, exist_ok=True)
        log.info("Downloading %s → %s", url, dest)
        urllib.request.urlretrieve(url, dest)
    except Exception as exc:
        log.error("Download failed: %s", exc)


# ---------------------------------------------------------------------------
# GeneticNMF loading
# ---------------------------------------------------------------------------

def load_genetic_nmf(dataset_id: str, condition: str = "") -> dict:
    """Load NMF loadings. condition="" uses bare genetic_nmf_loadings.npz (U_scaled format).
    condition="Stim48hr"/"REST" uses genetic_nmf_loadings_{cond_lower}.npz."""
    cond_lower = condition.strip().lower()
    if cond_lower:
        npz_path = _PERTURBSEQ / dataset_id / f"genetic_nmf_loadings_{cond_lower}.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"Condition-specific NMF not found: {npz_path}")
    else:
        npz_path = _PERTURBSEQ / dataset_id / "genetic_nmf_loadings.npz"
        if not npz_path.exists():
            svd_path = _PERTURBSEQ / dataset_id / "svd_loadings.npz"
            if svd_path.exists():
                log.info("genetic_nmf_loadings.npz not found — falling back to svd_loadings.npz for %s", dataset_id)
                npz_path = svd_path
            else:
                raise FileNotFoundError(f"Neither genetic_nmf_loadings.npz nor svd_loadings.npz found for {dataset_id}")
    log.info("Loading NMF loadings: %s", npz_path)
    data = np.load(npz_path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def extract_program_gene_sets(
    nmf: dict, top_n_genes: int, condition: str = "",
) -> dict[str, list[str]]:
    """
    Return {program_id → [gene_symbol, ...]} using top-N genes by |U_scaled|
    per program column.

    Program IDs:
      condition=""   → P00..P{k-1}  (legacy SVD-style, used by runner.py alias lookup)
      condition set  → C01..C{k}    (GeneticNMF condition-specific, matches RA_GeneticNMF_Stim48hr_C01)
    """
    U = nmf["U_scaled"]          # (n_genes, k)
    gene_names = nmf["gene_names"].tolist()
    n_programs = U.shape[1]
    use_c_ids = bool(condition.strip())

    programs: dict[str, list[str]] = {}
    for k in range(n_programs):
        col = U[:, k]
        idx = np.argsort(np.abs(col))[::-1][:top_n_genes]
        prog_id = f"C{k+1:02d}" if use_c_ids else f"P{k:02d}"
        programs[prog_id] = [gene_names[i] for i in idx]

    log.info("Extracted %d programs, %d genes each (max)", n_programs, top_n_genes)
    return programs


# ---------------------------------------------------------------------------
# Enhancer coordinate helpers
# ---------------------------------------------------------------------------

def _shrink_to_150bp(df: pd.DataFrame) -> pd.DataFrame:
    """Shrink enhancer windows to 150 bp centred on midpoint (CATLAS convention)."""
    midpoint = (df["start"] + df["end"]) // 2
    half = _SHRINK_BP // 2
    df = df.copy()
    df["start"] = (midpoint - half).clip(lower=0)
    df["end"] = midpoint + half
    return df


def get_enhancers_for_program(
    abc_df: pd.DataFrame,
    gene_set: list[str],
) -> pd.DataFrame:
    """Return ABC rows whose TargetGene is in gene_set."""
    return abc_df[abc_df["TargetGene"].isin(gene_set)].copy()


# ---------------------------------------------------------------------------
# GWAS-locus → gene index (for hybrid BED generation)
# ---------------------------------------------------------------------------

def build_gene_to_locus_map(dataset_id: str) -> dict[str, list[int]]:
    """
    Return {gene_symbol → [locus_indices]} from gwas_anchored_programs.npz.
    Locus index i corresponds to ra_loci/L{i:02d}.bed.
    Returns {} if the NPZ is absent (non-RA diseases or missing file).
    """
    for ds_dir in (_PERTURBSEQ).iterdir():
        if not ds_dir.is_dir():
            continue
        if dataset_id and ds_dir.name != dataset_id:
            continue
        npz = ds_dir / "gwas_anchored_programs.npz"
        if not npz.exists():
            continue
        try:
            data = np.load(npz, allow_pickle=True)
            locus_names: list[str] = list(data["locus_names"])
            locus_genes_raw = data["locus_genes_json"]
            if locus_genes_raw.ndim == 0:
                locus_genes_dict: dict = json.loads(str(locus_genes_raw.item()))
            else:
                locus_genes_dict = json.loads(str(locus_genes_raw[0]))
            gene_to_loci: dict[str, list[int]] = {}
            for i, lname in enumerate(locus_names):
                for gene in locus_genes_dict.get(lname, []):
                    gene_to_loci.setdefault(gene, []).append(i)
            log.info("Loaded GWAS locus gene map: %d loci → %d unique anchor genes", len(locus_names), len(gene_to_loci))
            return gene_to_loci
        except Exception as exc:
            log.warning("Could not parse gwas_anchored_programs.npz in %s: %s", ds_dir, exc)
    return {}


def load_locus_beds(
    locus_indices: list[int],
    loci_dir: pathlib.Path,
) -> pd.DataFrame:
    """Load and concatenate BED rows from ra_loci/L{i:02d}.bed files."""
    frames = []
    for idx in locus_indices:
        bed_file = loci_dir / f"L{idx:02d}.bed"
        if not bed_file.exists() or bed_file.stat().st_size == 0:
            continue
        try:
            df = pd.read_csv(
                bed_file, sep="\t", header=None,
                usecols=[0, 1, 2],
                names=["chr", "start", "end"],
                dtype={"chr": str, "start": int, "end": int},
            )
            frames.append(df)
        except Exception as exc:
            log.debug("Failed to read locus BED %s: %s", bed_file, exc)
    if not frames:
        return pd.DataFrame(columns=["chr", "start", "end"])
    return pd.concat(frames, ignore_index=True).drop_duplicates()


# ---------------------------------------------------------------------------
# ATAC peak intersection
# ---------------------------------------------------------------------------

def load_atac_peaks(disease_key: str) -> Optional[pd.DataFrame]:
    """
    Load ATAC peaks BED file if present.  Returns None if absent.
    """
    peak_file = _ATAC_PEAKS.get(disease_key)
    if peak_file is None:
        return None
    atac_path = _ATAC_DIR / peak_file
    if not atac_path.exists():
        log.info("ATAC peaks not found at %s — skipping ATAC filter", atac_path)
        return None
    log.info("Loading ATAC peaks: %s", atac_path)
    peaks = pd.read_csv(
        atac_path,
        sep="\t",
        header=None,
        usecols=[0, 1, 2],
        names=["chr", "start", "end"],
        low_memory=False,
    )
    return peaks


def _overlaps_any_peak(
    enh: pd.DataFrame, peaks: pd.DataFrame
) -> pd.Series:
    """
    Boolean Series — True if enhancer row overlaps at least one ATAC peak
    on the same chromosome.  Vectorised per-chromosome.
    """
    result = pd.Series(False, index=enh.index)
    for chrom, enh_chrom in enh.groupby("chr"):
        pk_chrom = peaks[peaks["chr"] == chrom]
        if pk_chrom.empty:
            continue
        pk_starts = pk_chrom["start"].values
        pk_ends = pk_chrom["end"].values
        for row_idx, row in enh_chrom.iterrows():
            overlap = (pk_starts < row["end"]) & (pk_ends > row["start"])
            if overlap.any():
                result.at[row_idx] = True
    return result


def filter_by_atac(enh: pd.DataFrame, peaks: pd.DataFrame) -> pd.DataFrame:
    mask = _overlaps_any_peak(enh, peaks)
    filtered = enh[mask].copy()
    return filtered


# ---------------------------------------------------------------------------
# BED writing
# ---------------------------------------------------------------------------

def write_program_bed(
    program_id: str,
    enh_shrunk: pd.DataFrame,
    out_dir: pathlib.Path,
) -> pathlib.Path:
    """
    Write a 4-column BED (chr, start, end, name) for S-LDSC annotation.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    bed_path = out_dir / f"{program_id}.bed"

    bed = enh_shrunk[["chr", "start", "end"]].copy()
    bed = bed.drop_duplicates()
    bed = bed.sort_values(["chr", "start"])
    bed["name"] = program_id

    bed.to_csv(bed_path, sep="\t", header=False, index=False)
    return bed_path


# ---------------------------------------------------------------------------
# Per-program stats
# ---------------------------------------------------------------------------

def compute_program_stats(
    program_id: str,
    gene_set: list[str],
    enh_raw: pd.DataFrame,
    enh_shrunk: pd.DataFrame,
    enh_atac: Optional[pd.DataFrame],
) -> dict:
    n_enh = len(enh_shrunk.drop_duplicates(subset=["chr", "start", "end"]))
    median_abc = float(enh_raw["ABC.Score"].median()) if len(enh_raw) else float("nan")
    pct_atac = float("nan")
    if enh_atac is not None and n_enh > 0:
        n_atac = len(enh_atac.drop_duplicates(subset=["chr", "start", "end"]))
        pct_atac = round(100.0 * n_atac / n_enh, 1)
    return {
        "program_id": program_id,
        "n_genes": len(gene_set),
        "n_enhancers": n_enh,
        "median_abc_score": round(median_abc, 5),
        "pct_atac_overlapping": pct_atac,
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def _build_hybrid_bed_for_program(
    gene_set: list[str],
    abc_df: pd.DataFrame,
    gene_to_locus: dict[str, list[int]],
    loci_dir: pathlib.Path,
    atac_peaks: Optional[pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, int, int]:
    """
    For each gene in gene_set:
      - If it is a GWAS anchor (has locus BEDs): use Moonen/LOCUS CRE coordinates
      - Otherwise: use CATLAS ABC enhancer predictions

    Returns (enh_raw_abc, enh_for_bed, n_locus_genes, n_abc_genes)
    where enh_raw_abc is ABC-only rows (for stats), enh_for_bed is the merged result.
    """
    locus_dir_ok = loci_dir.exists()

    locus_frames: list[pd.DataFrame] = []
    abc_gene_set: list[str] = []

    for gene in gene_set:
        locus_idxs = gene_to_locus.get(gene, []) if locus_dir_ok else []
        if locus_idxs:
            lf = load_locus_beds(locus_idxs, loci_dir)
            if not lf.empty:
                locus_frames.append(lf)
                continue
        # Fall back to ABC
        abc_gene_set.append(gene)

    n_locus_genes = len(gene_set) - len(abc_gene_set)
    n_abc_genes = len(abc_gene_set)

    # ABC rows for non-anchor genes
    enh_raw_abc = get_enhancers_for_program(abc_df, abc_gene_set) if abc_gene_set else pd.DataFrame(columns=["chr", "start", "end", "TargetGene", "ABC.Score"])

    # Shrink ABC enhancers to 150 bp
    enh_abc_shrunk = _shrink_to_150bp(enh_raw_abc) if not enh_raw_abc.empty else enh_raw_abc

    # Optionally ATAC-filter ABC enhancers (not applied to LOCUS CREs — those are already specific)
    if atac_peaks is not None and not enh_abc_shrunk.empty:
        enh_abc_filtered = filter_by_atac(enh_abc_shrunk, atac_peaks)
    else:
        enh_abc_filtered = enh_abc_shrunk

    # Merge locus BEDs and ABC BEDs
    parts: list[pd.DataFrame] = []
    for lf in locus_frames:
        parts.append(lf[["chr", "start", "end"]].copy())
    if not enh_abc_filtered.empty:
        parts.append(enh_abc_filtered[["chr", "start", "end"]].copy())

    if parts:
        enh_for_bed = pd.concat(parts, ignore_index=True).drop_duplicates()
    else:
        enh_for_bed = pd.DataFrame(columns=["chr", "start", "end"])

    return enh_raw_abc, enh_for_bed, n_locus_genes, n_abc_genes


def run(
    disease_key: str,
    dataset_id: str,
    min_abc_score: float,
    top_n_genes: int,
    condition: str = "",
) -> None:
    disease_key = disease_key.lower()
    cond_lower = condition.strip().lower()

    log.info("=== sldsc_chromatin_programs: %s / %s  condition=%r ===", disease_key, dataset_id, condition or "shared")
    log.info("min_abc_score=%.4f  top_n_genes=%d", min_abc_score, top_n_genes)

    abc_df = load_abc_predictions(disease_key, min_abc_score)
    nmf = load_genetic_nmf(dataset_id, condition=condition)
    program_gene_sets = extract_program_gene_sets(nmf, top_n_genes, condition=condition)

    atac_peaks = load_atac_peaks(disease_key)
    if atac_peaks is not None:
        log.info("ATAC peaks loaded: %d peaks", len(atac_peaks))
    else:
        log.info("No ATAC peaks — ATAC filter will be skipped (ABC enhancers used as-is)")

    # Build GWAS-anchor gene → locus-index map for hybrid BED generation.
    gene_to_locus = build_gene_to_locus_map(dataset_id)
    loci_dir = _OUT_BASE / f"{disease_key}_loci"
    if gene_to_locus and loci_dir.exists():
        log.info("Hybrid BED mode: %d GWAS anchor genes will use LOCUS CREs (ra_loci/); rest use CATLAS ABC", len(gene_to_locus))
    else:
        log.info("Pure ABC mode: no GWAS anchor gene map or loci dir — using CATLAS ABC for all genes")

    # Condition-specific NMF → separate subdirectory so SVD BEDs (ra/) are untouched.
    if cond_lower:
        out_dir = _OUT_BASE / f"{disease_key}_gnmf_{cond_lower}"
    else:
        out_dir = _OUT_BASE / disease_key
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict = {
        "disease_key": disease_key,
        "dataset_id": dataset_id,
        "min_abc_score": min_abc_score,
        "top_n_genes": top_n_genes,
        "programs": [],
    }

    log.info("%-8s  %8s  %10s  %14s  %10s  %10s", "Program", "n_genes", "n_enh", "median_abc", "n_locus", "n_abc")
    log.info("-" * 78)

    for prog_id, gene_set in sorted(program_gene_sets.items()):
        enh_raw_abc, enh_for_bed, n_locus_genes, n_abc_genes = _build_hybrid_bed_for_program(
            gene_set, abc_df, gene_to_locus, loci_dir, atac_peaks,
        )

        if enh_for_bed.empty:
            log.warning("%s: 0 enhancers after hybrid BED build (locus=%d abc=%d) — skipping BED", prog_id, n_locus_genes, n_abc_genes)
            stats = {
                "program_id": prog_id, "n_genes": len(gene_set),
                "n_enhancers": 0, "median_abc_score": float("nan"),
                "pct_atac_overlapping": float("nan"),
                "n_locus_genes": n_locus_genes, "n_abc_genes": n_abc_genes,
            }
            manifest["programs"].append(stats)
            continue

        # Use enh_raw_abc for stats only (locus rows don't have ABC.Score)
        n_enh = len(enh_for_bed.drop_duplicates(subset=["chr", "start", "end"]))
        median_abc = float(enh_raw_abc["ABC.Score"].median()) if not enh_raw_abc.empty else float("nan")

        bed_path = write_program_bed(prog_id, enh_for_bed, out_dir)
        stats = {
            "program_id": prog_id,
            "n_genes": len(gene_set),
            "n_enhancers": n_enh,
            "median_abc_score": round(median_abc, 5) if median_abc == median_abc else float("nan"),
            "pct_atac_overlapping": float("nan"),  # hybrid BEDs mix sources; not meaningful
            "n_locus_genes": n_locus_genes,
            "n_abc_genes": n_abc_genes,
            "bed_path": str(bed_path.relative_to(_ROOT)),
        }
        manifest["programs"].append(stats)

        log.info(
            "%-8s  %8d  %10d  %14s  %10d  %10d",
            prog_id, stats["n_genes"], stats["n_enhancers"],
            f"{median_abc:.5f}" if median_abc == median_abc else "n/a",
            n_locus_genes, n_abc_genes,
        )

    _MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    manifest_sfx = f"_{cond_lower}" if cond_lower else ""
    manifest_path = _MANIFEST_DIR / f"sldsc_chromatin_manifest_{disease_key}{manifest_sfx}.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    n_written = sum(1 for p in manifest["programs"] if "bed_path" in p)
    log.info("Wrote %d BED files to %s", n_written, out_dir)
    log.info("Manifest: %s", manifest_path)

    _print_sldsc_example(disease_key, manifest)


def _print_sldsc_example(disease_key: str, manifest: dict) -> None:
    programs_with_beds = [p for p in manifest["programs"] if "bed_path" in p]
    if not programs_with_beds:
        return

    sample = programs_with_beds[0]
    prog_id = sample["program_id"]
    bed_stem = str(_ROOT / sample["bed_path"]).rstrip(".bed")

    print("\n" + "=" * 72)
    print("S-LDSC commands (run from repo root, one per program):")
    print("=" * 72)
    print(f"""
# 1. Compute LD scores for the chromatin annotation:
python ldsc/make_annot.py \\
    --bed-file data/sldsc_chromatin/{disease_key}/{prog_id}.bed \\
    --bimfile {{plink_bimfile}}.bim \\
    --annot-file data/sldsc_chromatin/{disease_key}/{prog_id}.{{chrom}}.annot.gz

# 2. Run ldsc.py for partitioned h2:
python ldsc/ldsc.py \\
    --h2 data/ldsc/sumstats/{{disease_key}}.sumstats.gz \\
    --ref-ld-chr {{baseline_ld_prefix}},data/sldsc_chromatin/{disease_key}/{prog_id}. \\
    --w-ld-chr {{weights_prefix}} \\
    --overlap-annot \\
    --frqfile-chr {{frqfile_prefix}} \\
    --out data/ldsc/results/{disease_key}/{prog_id}

# Repeat for each program listed in:
#   outputs/sldsc_chromatin_manifest_{disease_key}.json
""")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build S-LDSC chromatin annotations from ABC + GeneticNMF programs."
    )
    p.add_argument(
        "--disease_key",
        required=True,
        choices=["cad", "ra"],
        help="Disease key (cad or ra)",
    )
    p.add_argument(
        "--dataset_id",
        required=True,
        help="Perturb-seq dataset ID under data/perturbseq/ (e.g. schnitzler_cad_vascular)",
    )
    p.add_argument(
        "--min_abc_score",
        type=float,
        default=0.013,
        help="Minimum ABC score threshold (default: 0.013, CATLAS default)",
    )
    p.add_argument(
        "--top_n_genes",
        type=int,
        default=100,
        help="Number of top genes per program for enhancer lookup (default: 100)",
    )
    p.add_argument(
        "--condition",
        default="",
        help="Condition suffix for condition-specific NMF (e.g. 'Stim48hr', 'REST'). "
             "Empty = use bare genetic_nmf_loadings.npz (legacy P00 naming).",
    )
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    run(
        disease_key=args.disease_key,
        dataset_id=args.dataset_id,
        min_abc_score=args.min_abc_score,
        top_n_genes=args.top_n_genes,
        condition=args.condition,
    )
