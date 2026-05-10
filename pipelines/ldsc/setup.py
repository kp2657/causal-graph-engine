"""
pipelines/ldsc/setup.py — Download GWAS sumstats, LD scores, and install ldsc.

Usage (run once before the pipeline):

    python -m pipelines.ldsc.setup download_all
    python -m pipelines.ldsc.setup install_ldsc

Data sources:
  CAD: Aragam 2022 GCST90132314 — 181,522 cases / 984,168 controls (EUR, GRCh38 harmonised)
       Aragam et al. 2022 Nat Genet — largest cardiac-endpoint-adjudicated CAD GWAS
       https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST90132001-GCST90133000/GCST90132314/harmonised/GCST90132314.h.tsv.gz

  RA:  GWAS Catalog GCST90132222 — Sakaue et al. 2021 (PMID 36333501)
       Seropositive RA, 35,871 cases / 240,149 controls (multi-cohort EUR, GRCh38 harmonised)
       https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST90132001-GCST90133000/GCST90132222/harmonised/GCST90132222.h.tsv.gz

  LD scores: Zenodo 10515792 — 1000G Phase3 EUR baselineLD v2.2 (~645 MB)
  LD weights: Zenodo 10515792 — 1000G Phase3 EUR weights HapMap3 (~12 MB)

References:
  Aragam 2022: PMID 36474045 — cardiac-endpoint adjudicated CAD GWAS
  Sakaue 2021: PMID 36333501 — seropositive RA, replaces PLAtlas Phe_714 (ICD-EHR case dilution)
  LDSC: Bulik-Sullivan et al. 2015 Nature Genetics
  Finucane et al. 2018 Nature Genetics (S-LDSC cell-type specific analysis)
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

_ROOT = Path(__file__).parent.parent.parent
_LDSC_DIR = _ROOT / "data" / "ldsc"
_LDSCORE_DIR = _LDSC_DIR / "ldscores"
_SUMSTATS_DIR = _LDSC_DIR / "sumstats"

# ldsc 2.0.1 (Python 3) is installed via pip: /opt/anaconda3/envs/causal-graph/bin/ldsc.py
# Falls back to data/ldsc/ldsc/ldsc.py if pip version not found.
def _find_ldsc_bin() -> Path:
    import shutil
    # pip install ldsc places ldsc.py in the env bin dir
    candidate = Path(sys.executable).parent / "ldsc.py"
    if candidate.exists():
        return candidate
    # Old git-clone fallback
    return _LDSC_DIR / "ldsc" / "ldsc.py"

def _find_munge_bin() -> Path:
    candidate = Path(sys.executable).parent / "munge_sumstats.py"
    if candidate.exists():
        return candidate
    return _LDSC_DIR / "ldsc" / "munge_sumstats.py"

_LDSC_BIN  = _find_ldsc_bin()
_MUNGE_BIN = _find_munge_bin()

# --------------------------------------------------------------------------- #
# GWAS sumstats configuration
# --------------------------------------------------------------------------- #

GWAS_CONFIG: dict[str, dict] = {
    "CAD": {
        # Aragam 2022 (PMID 36474045) — cardiac-endpoint adjudicated CAD GWAS
        # 181,522 cases / 984,168 controls, EUR, GRCh38 harmonised
        # Replaces PLAtlas Phe_414 (ICD-9 EHR-derived, case dilution → phenotype heterogeneity)
        "url": "https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST90132001-GCST90133000/GCST90132314/harmonised/GCST90132314.h.tsv.gz",
        "filename": "GCST90132314.h.tsv.gz",
        "description": "GCST90132314 — Aragam 2022 CAD (181,522 cases / 984,168 controls, EUR GRCh38 harmonised)",
        "snp_col": "rsid",
        "a1_col": "effect_allele",
        "a2_col": "other_allele",
        "chr_col": "chromosome",
        "pos_col": "base_pair_location",
        "beta_col": "beta",
        "se_col": "standard_error",
        "p_col": "p_value",
        "n_value": 1165690,  # 181,522 + 984,168
    },
    "RA": {
        # GWAS Catalog GCST90132222 — Sakaue et al. 2021 (PMID 36333501)
        # Seropositive RA, 35,871 cases / 240,149 controls (multi-cohort)
        # Replaces PLAtlas Phe_714 (ICD-9 EHR-derived, case dilution → τ depletion)
        "url": "https://ftp.ebi.ac.uk/pub/databases/gwas/summary_statistics/GCST90132001-GCST90133000/GCST90132222/harmonised/GCST90132222.h.tsv.gz",
        "filename": "GCST90132222.h.tsv.gz",
        "description": "GCST90132222 — Sakaue 2021 seropositive RA (35,871 cases / 240,149 controls, multi-cohort)",
        "snp_col": "rsid",
        "a1_col": "effect_allele",
        "a2_col": "other_allele",
        "chr_col": "chromosome",
        "pos_col": "base_pair_location_grch38",
        "beta_col": "beta",
        "se_col": "standard_error",
        "p_col": "p_value",
        "n_value": 276020,  # 35,871 cases + 240,149 controls
    },
}

# --------------------------------------------------------------------------- #
# LD score URLs (Zenodo 10515792)
# --------------------------------------------------------------------------- #

_LD_SCORES_URL = "https://zenodo.org/api/records/10515792/files/1000G_Phase3_baselineLD_v2.2_ldscores.tgz/content"
_LD_WEIGHTS_URL = "https://zenodo.org/api/records/10515792/files/1000G_Phase3_weights_hm3_no_MHC.tgz/content"
_HM3_SNPS_URL = "https://zenodo.org/api/records/10515792/files/hm3_no_MHC.list.txt/content"

# --------------------------------------------------------------------------- #
# ldsc software (Python 3 compatible fork)
# --------------------------------------------------------------------------- #

_LDSC_REPO = "https://github.com/bulik/ldsc.git"


def _wget(url: str, dest: Path, description: str = "") -> Path:
    """Download url to dest with progress. Skip if already exists."""
    if dest.exists() and dest.stat().st_size > 1_000:
        log.info("Already downloaded: %s", dest.name)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading %s → %s", description or url[:60], dest.name)
    try:
        import urllib.request
        def _progress(count, block, total):
            if total > 0:
                pct = count * block * 100 // total
                sys.stdout.write(f"\r  {pct}% of {total//1024//1024} MB")
                sys.stdout.flush()
        urllib.request.urlretrieve(url, str(dest), _progress)
        sys.stdout.write("\n")
    except Exception as e:
        raise RuntimeError(f"Download failed for {url}: {e}") from e
    return dest


def download_gwas_sumstats(disease_key: str) -> Path:
    """Download GWAS summary statistics for disease_key (CAD or RA)."""
    cfg = GWAS_CONFIG.get(disease_key.upper())
    if not cfg:
        raise ValueError(f"No GWAS config for {disease_key}. Available: {list(GWAS_CONFIG)}")
    _SUMSTATS_DIR.mkdir(parents=True, exist_ok=True)
    dest = _SUMSTATS_DIR / cfg["filename"]
    _wget(cfg["url"], dest, cfg["description"])
    return dest


def download_ld_scores() -> Path:
    """Download 1000G Phase3 EUR baselineLD v2.2 LD scores from Zenodo."""
    _LDSCORE_DIR.mkdir(parents=True, exist_ok=True)
    # Check if already extracted
    chr22 = _LDSCORE_DIR / "baselineLD_v2.2" / "baselineLD.22.l2.ldscore.gz"
    if chr22.exists():
        log.info("LD scores already extracted at %s", _LDSCORE_DIR / "baselineLD_v2.2")
        return _LDSCORE_DIR / "baselineLD_v2.2"

    tgz = _LDSCORE_DIR / "1000G_Phase3_baselineLD_v2.2_ldscores.tgz"
    _wget(_LD_SCORES_URL, tgz, "1000G EUR baselineLD v2.2 (645 MB)")
    log.info("Extracting LD scores...")
    with tarfile.open(tgz) as tf:
        tf.extractall(_LDSCORE_DIR, filter="data")
    return _LDSCORE_DIR


def download_ld_weights() -> Path:
    """Download 1000G EUR HapMap3 LD weights from Zenodo."""
    _LDSCORE_DIR.mkdir(parents=True, exist_ok=True)
    chr22w = next(_LDSCORE_DIR.rglob("weights.hm3_noMHC.22.l2.ldscore.gz"), None)
    if chr22w:
        log.info("LD weights already extracted")
        return chr22w.parent

    tgz = _LDSCORE_DIR / "1000G_Phase3_weights_hm3_no_MHC.tgz"
    _wget(_LD_WEIGHTS_URL, tgz, "1000G EUR HapMap3 LD weights (12 MB)")
    log.info("Extracting LD weights...")
    with tarfile.open(tgz) as tf:
        tf.extractall(_LDSCORE_DIR, filter="data")
    return _LDSCORE_DIR


def download_hapmap3_snps() -> Path:
    """Download HapMap3 SNP list for ldsc munging."""
    dest = _LDSCORE_DIR / "hm3_no_MHC.list.txt"
    _wget(_HM3_SNPS_URL, dest, "HapMap3 SNP list (12 MB)")
    return dest


def install_ldsc() -> Path:
    """Install ldsc (Python 3) via pip. Falls back to git-clone if pip fails."""
    if _LDSC_BIN.exists():
        log.info("ldsc already installed at %s", _LDSC_BIN)
        return _LDSC_BIN.parent

    log.info("Installing ldsc via pip (Python 3 compatible)...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "ldsc", "bitarray"],
            check=True, capture_output=True,
        )
        bin_path = Path(sys.executable).parent / "ldsc.py"
        if bin_path.exists():
            log.info("ldsc installed at %s", bin_path)
            return bin_path.parent
    except subprocess.CalledProcessError as e:
        log.warning("pip install ldsc failed: %s", e)

    # Fallback: clone Python 2 repo (last resort; won't work on Python 3.12)
    ldsc_dir = _LDSC_DIR / "ldsc"
    if not ldsc_dir.exists():
        log.warning("Cloning ldsc (Python 2 — may not work): %s", _LDSC_REPO)
        subprocess.run(["git", "clone", _LDSC_REPO, str(ldsc_dir)], check=True)
    return ldsc_dir


_POLYFUN_LDSCORE_DIR = _LDSC_DIR / "polyfun_ldscores"

# PolyFun UKB hg38 LD scores — hosted on AWS Open Data (free, no account needed).
# Tarball contains baselineLF_v2.2.UKB.{1..22}.l2.ldscore.gz files.
# Source: https://registry.opendata.aws/ukbb-ld/
# AWS CLI: aws s3 ls --no-sign-request s3://broad-alkesgroup-ukbb-ld/
_POLYFUN_LD_TARBALL_URL = "https://broad-alkesgroup-ukbb-ld.s3.amazonaws.com/UKBB_LD/baselineLF_v2.2.UKB.tar.gz"
_POLYFUN_LD_CHR_PREFIX  = "baselineLF_v2.2.UKB"   # files: {prefix}.{chr}.l2.ldscore.gz


def download_polyfun_ld_scores(force: bool = False) -> Path:
    """
    Download precomputed UKB-based hg38 LD scores for use with PolyFun/S-LDSC.

    Files expected: chr1.l2.ldscore.gz … chr22.l2.ldscore.gz
    Saved to: data/ldsc/polyfun_ldscores/

    IMPORTANT: As of 2026-04-25, the Alkes Group download URL is not publicly
    accessible (returns 404).  This function prints instructions for manual
    download and returns the target directory path.

    To download manually:
      1. Visit https://github.com/omerwe/polyfun and check the wiki for the
         current download link.
      2. Download all chr*.l2.ldscore.gz files.
      3. Place them in: data/ldsc/polyfun_ldscores/
      4. Re-run the pipeline — runner.py will detect the files automatically.

    Args:
        force: If True, re-download even if files already exist.

    Returns:
        Path to data/ldsc/polyfun_ldscores/ (may not contain files if URL unavailable).
    """
    _POLYFUN_LDSCORE_DIR.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded (accept both chr-prefixed and baselineLF-prefixed)
    existing = (list(_POLYFUN_LDSCORE_DIR.glob("chr*.l2.ldscore.gz")) +
                list(_POLYFUN_LDSCORE_DIR.glob(f"{_POLYFUN_LD_CHR_PREFIX}.*.l2.ldscore.gz")))
    if not force and len(existing) >= 22:
        log.info("PolyFun LD scores already present (%d files)", len(existing))
        return _POLYFUN_LDSCORE_DIR

    import tarfile, tempfile
    log.info("Downloading PolyFun hg38 LD scores tarball (~300 MB) from AWS ...")
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        _wget(_POLYFUN_LD_TARBALL_URL, tmp_path, "polyfun baselineLF_v2.2.UKB tarball")
        log.info("Extracting PolyFun LD scores → %s", _POLYFUN_LDSCORE_DIR)
        with tarfile.open(tmp_path, "r:gz") as tf:
            tf.extractall(_POLYFUN_LDSCORE_DIR, filter="data")
        log.info("PolyFun LD scores extracted successfully")
        log.info("PolyFun LD scores downloaded to %s", _POLYFUN_LDSCORE_DIR)
    except Exception as exc:
        log.warning("PolyFun tarball download failed: %s", exc)
        log.warning("Download manually: aws s3 cp --no-sign-request "
                    "s3://broad-alkesgroup-ukbb-ld/UKBB_LD/baselineLF_v2.2.UKB.tar.gz %s", tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    return _POLYFUN_LDSCORE_DIR


def download_all(diseases: list[str] | None = None) -> None:
    """Download all required files for S-LDSC (sumstats + LD scores + weights)."""
    diseases = diseases or list(GWAS_CONFIG.keys())
    for disease in diseases:
        log.info("=== Downloading GWAS sumstats for %s ===", disease)
        download_gwas_sumstats(disease)
    log.info("=== Downloading LD scores ===")
    download_ld_scores()
    log.info("=== Downloading LD weights ===")
    download_ld_weights()
    log.info("=== Downloading HapMap3 SNP list ===")
    download_hapmap3_snps()
    log.info("All downloads complete. Run: python -m pipelines.ldsc.runner run CAD && python -m pipelines.ldsc.runner run RA")


def status() -> dict:
    """Report which files are already downloaded."""
    out = {}
    for dk, cfg in GWAS_CONFIG.items():
        f = _SUMSTATS_DIR / cfg["filename"]
        out[f"{dk}_sumstats"] = f.exists() and f.stat().st_size > 1_000_000
    out["ld_scores"] = next(_LDSCORE_DIR.rglob("*.l2.ldscore.gz"), None) is not None
    out["ld_weights"] = next(_LDSCORE_DIR.rglob("weights*.l2.ldscore.gz"), None) is not None
    out["hm3_snps"] = (_LDSCORE_DIR / "hm3_no_MHC.list.txt").exists()
    out["ldsc_installed"] = _LDSC_BIN.exists()
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "download_all":
        download_all()
    elif cmd == "install_ldsc":
        install_ldsc()
    elif cmd == "status":
        s = status()
        for k, v in s.items():
            print(f"  {'✓' if v else '✗'} {k}")
    else:
        print(f"Unknown command: {cmd}. Use: download_all | install_ldsc | status")
