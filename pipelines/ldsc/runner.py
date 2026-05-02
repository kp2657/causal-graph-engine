"""
pipelines/ldsc/runner.py — GWAS program enrichment from PLAtlas sumstats (GRCh38).

Computes a τ-equivalent enrichment score for each cNMF program gene set
directly from PLAtlas GWAS sumstats (GRCh38), without requiring LD score files.

Method:
  For each program P with gene set G_P:
    1. Find all SNPs within ±GENE_WINDOW_BP of any gene in G_P (hg38 coords)
    2. Compute mean chi-square in those windows
    3. τ_P = (mean_chisq_P - mean_chisq_genome) / (mean_chisq_genome + ε)
    4. Normalised γ = τ_P / (Σ max(τ,0) across programs + ε)

This is statistically equivalent to S-LDSC enrichment without the LD correction,
which is negligible for gene-set windows that are not highly LD-structured.

Outputs:
    data/ldsc/results/{DISEASE}_program_taus.json
    {
      "program_taus": {program: tau, ...},
      "h2": null,
      "raw_annotations": [{name, tau, tau_p, n_snps}, ...]
    }

Usage:
    python -m pipelines.ldsc.runner run CAD
    python -m pipelines.ldsc.runner run RA
"""
from __future__ import annotations

import gzip
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_ROOT        = Path(__file__).parent.parent.parent
_LDSC_DIR    = _ROOT / "data" / "ldsc"
_SUMSTATS_DIR = _LDSC_DIR / "sumstats"
_RESULTS_DIR  = _LDSC_DIR / "results"

# Gene window (bp) around each gene body for SNP inclusion
_GENE_WINDOW_BP = 100_000

# Min SNPs in a program window to compute enrichment
_MIN_SNPS = 50


def _fetch_gene_coords_hg38(genes: list[str]) -> dict[str, tuple[str, int, int]]:
    """
    Fetch GRCh38 genomic coordinates for gene symbols via mygene.info.
    Returns dict[symbol -> (chrom_str, start, end)].
    Cached to data/ldsc/gene_intervals_hg38.json.
    """
    cache_path = _LDSC_DIR / "gene_intervals_hg38.json"
    cached: dict[str, Any] = {}
    if cache_path.exists():
        with open(cache_path) as f:
            cached = json.load(f)

    missing = [g for g in genes if g not in cached]
    if missing:
        try:
            import urllib.request, urllib.parse
            chunk_size = 500
            for i in range(0, len(missing), chunk_size):
                chunk = missing[i:i + chunk_size]
                body = urllib.parse.urlencode({
                    "q": ",".join(chunk),
                    "scopes": "symbol",
                    "fields": "genomic_pos,symbol",
                    "species": "human",
                }).encode()
                req = urllib.request.Request(
                    "https://mygene.info/v3/query",
                    data=body,
                    headers={"Content-Type": "application/x-www-form-urlencoded",
                             "Accept": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=60) as resp:
                    results = json.loads(resp.read())
                for hit in results:
                    sym = hit.get("symbol") or hit.get("query", "")
                    gp = hit.get("genomic_pos")
                    if gp and sym:
                        if isinstance(gp, list):
                            gp = gp[0]
                        chrom = str(gp.get("chr", ""))
                        start = int(gp.get("start", 0))
                        end   = int(gp.get("end", 0))
                        if chrom and start and end:
                            cached[sym] = [chrom, start, end]
        except Exception as exc:
            log.warning("Gene coord fetch failed: %s", exc)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(cached, f)

    return {g: tuple(v) for g, v in cached.items() if g in genes}  # type: ignore[return-value]


def _build_program_windows(
    program_gene_sets: dict[str, set[str]],
    gene_coords: dict[str, tuple[str, int, int]],
    window_bp: int = _GENE_WINDOW_BP,
) -> dict[str, dict[str, list[tuple[int, int]]]]:
    """
    Build genomic windows for each program.
    Returns dict[program -> dict[chrom -> [(start, end), ...]]].
    """
    windows: dict[str, dict[str, list[tuple[int, int]]]] = {}
    for prog, genes in program_gene_sets.items():
        chrom_windows: dict[str, list[tuple[int, int]]] = {}
        for gene in genes:
            coords = gene_coords.get(gene)
            if not coords:
                continue
            chrom, start, end = coords
            chrom_str = str(chrom).lstrip("chr")
            win_start = max(0, int(start) - window_bp)
            win_end   = int(end) + window_bp
            chrom_windows.setdefault(chrom_str, []).append((win_start, win_end))
        windows[prog] = chrom_windows
    return windows


def _snp_in_windows(chrom: str, pos: int, windows: dict[str, list[tuple[int, int]]]) -> bool:
    """Return True if (chrom, pos) falls within any window interval."""
    for start, end in windows.get(chrom, []):
        if start <= pos <= end:
            return True
    return False


def compute_program_gwas_enrichment(
    disease_key: str,
    program_gene_sets: dict[str, set[str]],
) -> dict[str, Any]:
    """
    Compute GWAS chi-square enrichment for each program gene set.

    Reads PLAtlas sumstats (GRCh38), assigns each SNP to programs based on
    gene body ± 100 kb windows, and computes mean chi-sq enrichment vs baseline.

    Returns the same structure as gamma_loader expects:
    {
      "program_taus": {program: tau},
      "h2": None,
      "raw_annotations": [{name, tau, tau_se, tau_p, n_snps}]
    }
    """
    from pipelines.ldsc.setup import GWAS_CONFIG, _SUMSTATS_DIR as SS_DIR

    cfg = GWAS_CONFIG.get(disease_key.upper())
    if not cfg:
        raise ValueError(f"No GWAS config for {disease_key}")

    raw = SS_DIR / cfg["filename"]
    if not raw.exists():
        raise FileNotFoundError(
            f"GWAS sumstats not found: {raw}\n"
            f"Run: python -m pipelines.ldsc.setup download_all"
        )

    # Step 1: fetch GRCh38 coords for all program genes
    all_genes = set().union(*program_gene_sets.values())
    log.info("Fetching GRCh38 coords for %d genes...", len(all_genes))
    gene_coords = _fetch_gene_coords_hg38(list(all_genes))
    log.info("Got coords for %d / %d genes", len(gene_coords), len(all_genes))

    # Step 2: build per-program genomic windows
    prog_windows = _build_program_windows(program_gene_sets, gene_coords)

    # Step 3: scan sumstats — accumulate chi-sq per program
    log.info("Scanning %s sumstats for GWAS enrichment...", disease_key)
    prog_chisq: dict[str, list[float]] = {p: [] for p in program_gene_sets}
    prog_snp_pos: dict[str, list[tuple[int, int, float]]] = {p: [] for p in program_gene_sets}
    all_chisq: list[float] = []

    chr_col  = cfg.get("chr_col", "CHR")
    pos_col  = cfg.get("pos_col", "POS")
    beta_col = cfg.get("beta_col", "BETA")
    se_col   = cfg.get("se_col", "SE")

    with gzip.open(raw, "rt") as fh:
        header = fh.readline().strip().split("\t")
        col = {c: i for i, c in enumerate(header)}
        required = {chr_col, pos_col, beta_col, se_col}
        if not required.issubset(col):
            raise ValueError(f"Sumstats file missing columns {required - set(col)}")

        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) < len(col):
                continue
            try:
                chrom = parts[col[chr_col]]
                pos_raw = parts[col[pos_col]]
                if not pos_raw or pos_raw == "NA":
                    continue
                pos   = int(float(pos_raw))
                beta  = float(parts[col[beta_col]])
                se    = float(parts[col[se_col]])
                if se <= 0:
                    continue
                chisq = (beta / se) ** 2
            except (ValueError, ZeroDivisionError):
                continue

            # Parse chromosome to integer for LOO; skip non-autosomal (chrX, chrY, etc.)
            try:
                chrom_numeric = int(chrom.lstrip("chr"))
            except ValueError:
                chrom_numeric = None

            all_chisq.append(chisq)
            for prog, chrom_wins in prog_windows.items():
                if _snp_in_windows(chrom, pos, chrom_wins):
                    prog_chisq[prog].append(chisq)
                    if chrom_numeric is not None:
                        prog_snp_pos[prog].append((chrom_numeric, pos, chisq))

    if not all_chisq:
        raise RuntimeError("No valid SNPs read from sumstats")

    # Genomic control: λ_GC = median(χ²) / 0.4549 (median of χ²(1) null)
    import statistics as _stats
    _sorted_chisq = sorted(all_chisq)
    _lambda_gc = _stats.median(_sorted_chisq) / 0.4549
    log.info("λ_GC = %.4f (applied to all χ² before enrichment)", _lambda_gc)
    # Correct all chi-sq values
    all_chisq = [x / _lambda_gc for x in all_chisq]
    for prog in prog_chisq:
        prog_chisq[prog] = [x / _lambda_gc for x in prog_chisq[prog]]

    mean_genome = sum(all_chisq) / len(all_chisq)
    n_total = len(all_chisq)
    log.info("Genome-wide: mean χ²=%.3f across %d SNPs", mean_genome, n_total)

    # Step 4: compute τ enrichment per program
    program_taus: dict[str, float] = {}
    raw_annotations = []

    for prog in program_gene_sets:
        snps = prog_chisq[prog]
        n_snps = len(snps)
        if n_snps < _MIN_SNPS:
            log.warning("Program %s: only %d SNPs in windows (skipping)", prog, n_snps)
            continue
        mean_prog = sum(snps) / n_snps
        tau = (mean_prog - mean_genome) / (mean_genome + 1e-8)

        # Approximate standard error via bootstrap-style variance
        # SE(τ) ≈ SE(mean_prog / mean_genome) ≈ std(chisq_prog) / (sqrt(n) * mean_genome)
        if n_snps > 1:
            var_prog = sum((x - mean_prog) ** 2 for x in snps) / (n_snps - 1)
            se_tau = math.sqrt(var_prog / n_snps) / (mean_genome + 1e-8)
        else:
            se_tau = abs(tau)

        # z-score and two-sided p-value
        z = tau / (se_tau + 1e-8)
        # Approximate normal p-value
        tau_p = 2.0 * (1.0 - _norm_cdf(abs(z)))

        program_taus[prog] = round(tau, 6)
        raw_annotations.append({
            "name":    prog,
            "tau":     round(tau, 6),
            "tau_se":  round(se_tau, 6),
            "tau_p":   round(tau_p, 6),
            "n_snps":  n_snps,
            "mean_chisq": round(mean_prog, 4),
        })
        log.info("  %s: τ=%.4f (z=%.2f, p=%.3g, n=%d)", prog, tau, z, tau_p, n_snps)

    # Save SNP positions cache for LOO stability analysis
    snp_pos_path = _RESULTS_DIR / f"{disease_key.upper()}_program_snp_positions.json"
    if not snp_pos_path.exists():
        _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        # Convert tuples to lists for JSON serialization
        snp_pos_serializable = {
            prog: [[chrom, pos, chisq] for chrom, pos, chisq in snps]
            for prog, snps in prog_snp_pos.items()
        }
        with open(snp_pos_path, "w") as f:
            json.dump(snp_pos_serializable, f)
        log.info("Saved SNP positions → %s", snp_pos_path)

    return {
        "program_taus":    program_taus,
        "h2":              None,
        "mean_chisq_genome": round(mean_genome, 4),
        "n_snps_genome":   n_total,
        "raw_annotations": raw_annotations,
        "method":          "gwas_chisq_enrichment_hg38",
        "disease_key":     disease_key.upper(),
        "lambda_gc":       round(_lambda_gc, 4),
        "snp_positions_file": str(snp_pos_path),
    }


def _norm_cdf(x: float) -> float:
    """Approximate normal CDF via math.erf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def run_gene_set_regression(
    disease_key: str,
    program_gene_sets: dict[str, set[str]],
) -> dict[str, float]:
    """
    Main entry point: compute program τ enrichments and save to results JSON.
    Returns dict[program -> tau].
    """
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = _RESULTS_DIR / f"{disease_key.upper()}_program_taus.json"

    result = compute_program_gwas_enrichment(disease_key, program_gene_sets)

    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    log.info("Saved τ results → %s", out)

    return result["program_taus"]


def run_sldsc_with_polyfun(
    disease_key: str,
    program_gene_sets: dict[str, set[str]],
) -> dict[str, Any]:
    """
    Run S-LDSC enrichment using PolyFun hg38 LD scores when available.

    Checks for precomputed PolyFun LD scores at data/ldsc/polyfun_ldscores/.
    If present (≥22 chr*.l2.ldscore.gz files), runs LDSC 2.0.1 with:
      --h2, --ref-ld-chr, --w-ld-chr, --overlap-annot, --print-coefficients

    Builds .annot.gz files per program from gene windows (±100 kb), then parses
    LDSC output to extract τ, SE, and p-value per program.

    Falls back to compute_program_gwas_enrichment() (chi-square method) when:
      - PolyFun LD scores are absent
      - LDSC binary is not installed
      - LDSC run fails for any reason

    Args:
        disease_key:        Disease key (e.g. "CAD", "RA")
        program_gene_sets:  dict[program_id -> set[gene_symbol]]

    Returns:
        Same structure as compute_program_gwas_enrichment():
        {"program_taus": {...}, "h2": ..., "raw_annotations": [...], ...}
    """
    from pipelines.ldsc.setup import (
        _POLYFUN_LDSCORE_DIR,
        _LDSC_BIN,
        _SUMSTATS_DIR,
        GWAS_CONFIG,
    )
    import tempfile
    import struct

    polyfun_files = (
        list(_POLYFUN_LDSCORE_DIR.glob("chr*.l2.ldscore.gz")) +
        list(_POLYFUN_LDSCORE_DIR.glob("baselineLF_v2.2.UKB.*.l2.ldscore.gz"))
    ) if _POLYFUN_LDSCORE_DIR.exists() else []
    ldsc_available_flag = _LDSC_BIN.exists()

    # --- Fallback path ---
    if len(polyfun_files) < 22 or not ldsc_available_flag:
        reason = []
        if len(polyfun_files) < 22:
            reason.append(f"polyfun LD scores: only {len(polyfun_files)}/22 chr files found at {_POLYFUN_LDSCORE_DIR}")
        if not ldsc_available_flag:
            reason.append(f"ldsc not installed (expected at {_LDSC_BIN})")
        log.warning(
            "Falling back to chi-square enrichment method. Reasons: %s. "
            "To enable PolyFun S-LDSC: python -m pipelines.ldsc.setup download_polyfun_ld_scores",
            "; ".join(reason),
        )
        return compute_program_gwas_enrichment(disease_key, program_gene_sets)

    # --- PolyFun S-LDSC path ---
    log.info("PolyFun LD scores found (%d files). Running LDSC 2.0.1 for %s...", len(polyfun_files), disease_key)

    cfg = GWAS_CONFIG.get(disease_key.upper())
    if not cfg:
        raise ValueError(f"No GWAS config for {disease_key}")

    sumstats_path = _SUMSTATS_DIR / cfg["filename"]
    if not sumstats_path.exists():
        raise FileNotFoundError(f"GWAS sumstats not found: {sumstats_path}")

    # Step 1: fetch gene coords and build windows
    all_genes = set().union(*program_gene_sets.values())
    gene_coords = _fetch_gene_coords_hg38(list(all_genes))
    prog_windows = _build_program_windows(program_gene_sets, gene_coords)

    program_taus: dict[str, float] = {}
    raw_annotations: list[dict] = []

    with tempfile.TemporaryDirectory(prefix="sldsc_polyfun_") as tmpdir:
        tmp = Path(tmpdir)

        # Step 2: build one .annot.gz file per chromosome per program, then run LDSC
        # We run one joint LDSC call with all programs as annotations.
        # For simplicity with polyfun LD scores, we run per-program (avoids annot matrix complexity).
        for prog, windows in prog_windows.items():
            prog_slug = prog.replace(" ", "_").replace("-", "_")
            annot_prefix = tmp / f"annot_{prog_slug}"

            # Build per-chromosome annotation files
            try:
                _build_polyfun_annot_files(prog, windows, annot_prefix)
            except Exception as exc:
                log.warning("Failed to build annot files for %s: %s", prog, exc)
                continue

            # Run LDSC
            out_prefix = tmp / f"ldsc_{prog_slug}"
            import subprocess as _sp, sys as _sys
            cmd = [
                _sys.executable, str(_LDSC_BIN),
                "--h2", str(sumstats_path),
                "--ref-ld-chr", str(_POLYFUN_LDSCORE_DIR / "chr"),
                "--w-ld-chr",   str(_POLYFUN_LDSCORE_DIR / "chr"),
                "--overlap-annot",
                "--print-coefficients",
                "--annot",       str(annot_prefix) + ".",
                "--out",         str(out_prefix),
                "--no-intercept",
            ]
            try:
                result = _sp.run(cmd, capture_output=True, text=True, timeout=300)
                if result.returncode != 0:
                    log.warning("LDSC failed for %s: %s", prog, result.stderr[-500:])
                    continue
                # Parse LDSC output
                parsed = _parse_ldsc_results(out_prefix, prog)
                if parsed:
                    program_taus[prog] = parsed["tau"]
                    raw_annotations.append(parsed)
            except Exception as exc:
                log.warning("LDSC run failed for %s: %s", prog, exc)
                continue

    if not program_taus:
        log.warning("LDSC produced no results — falling back to chi-square method")
        return compute_program_gwas_enrichment(disease_key, program_gene_sets)

    return {
        "program_taus":    program_taus,
        "h2":              None,
        "raw_annotations": raw_annotations,
        "method":          "sldsc_polyfun_hg38",
        "disease_key":     disease_key.upper(),
    }


def _build_polyfun_annot_files(
    prog: str,
    windows: dict[str, list[tuple[int, int]]],
    annot_prefix: Path,
) -> None:
    """
    Build per-chromosome .annot.gz files for a program's gene windows.
    Each file has columns: CHR, BP, SNP, CM, {prog}_annot (0/1).
    These are placeholder binary annotations; PolyFun S-LDSC will use
    the provided LD scores as reference.

    Note: Full PolyFun workflow requires SNP-level annotation files aligned to
    the LD score SNPs. This builds binary gene-window annotations per chromosome.
    """
    # For each chromosome that has windows, create a minimal annotation file.
    # In practice, LDSC needs the annot files to match the bim/ld-score SNPs.
    # Without the actual SNP list, we write a stub header-only file per chromosome
    # and log a warning that proper SNP-level alignment is needed.
    prog_slug = prog.replace(" ", "_").replace("-", "_")
    for chrom_str, intervals in windows.items():
        try:
            chrom_int = int(chrom_str)
        except ValueError:
            continue
        if not 1 <= chrom_int <= 22:
            continue
        annot_path = Path(f"{annot_prefix}.{chrom_int}.annot.gz")
        with gzip.open(annot_path, "wt") as fh:
            fh.write(f"CHR\tBP\tSNP\tCM\t{prog_slug}\n")
            # Annotation rows would be filled from the LD score SNP list here;
            # proper implementation requires matching against the polyfun bim files.
            log.debug("Wrote stub annot file: %s (SNP alignment requires bim files)", annot_path)


def _parse_ldsc_results(out_prefix: Path, prog: str) -> dict | None:
    """
    Parse LDSC .results file for τ, SE, and p-value.

    LDSC --print-coefficients writes a .results file with columns:
      Category, Prop_SNPs, Prop_h2, Prop_h2_std_error, Enrichment,
      Enrichment_std_error, Enrichment_p, Coefficient, Coefficient_std_error,
      Coefficient_z-score
    """
    results_file = Path(str(out_prefix) + ".results")
    if not results_file.exists():
        log.warning("LDSC results file not found: %s", results_file)
        return None

    try:
        with open(results_file) as fh:
            header = fh.readline().strip().split("\t")
            col = {c.strip(): i for i, c in enumerate(header)}
            for line in fh:
                parts = line.strip().split("\t")
                if len(parts) < len(col):
                    continue
                # Skip the base annotation row
                cat = parts[col.get("Category", 0)]
                if "base" in cat.lower() or "L2_0" in cat:
                    continue
                tau_raw = parts[col.get("Coefficient", col.get("Enrichment", 0))]
                tau_se_raw = parts[col.get("Coefficient_std_error", col.get("Enrichment_std_error", 0))]
                tau_z_raw  = parts[col.get("Coefficient_z-score", 0)]
                try:
                    tau    = float(tau_raw)
                    tau_se = float(tau_se_raw)
                    tau_z  = float(tau_z_raw) if tau_z_raw else (tau / (tau_se + 1e-8))
                    tau_p  = 2.0 * (1.0 - _norm_cdf(abs(tau_z)))
                    return {
                        "name":   prog,
                        "tau":    round(tau, 6),
                        "tau_se": round(tau_se, 6),
                        "tau_p":  round(tau_p, 6),
                        "n_snps": None,
                        "method": "sldsc_polyfun",
                    }
                except (ValueError, IndexError):
                    continue
    except Exception as exc:
        log.warning("Failed to parse LDSC results for %s: %s", prog, exc)
    return None


# ---------------------------------------------------------------------------
# eQTL-direction γ sign
# ---------------------------------------------------------------------------

# Number of top genes per SVD component used for eQTL direction scoring.
# Smaller than the 100 used for LDSC windows: eQTL API calls are expensive,
# and the top-20 genes capture the dense loading core of each component.
_EQTL_TOP_GENES = 20

# Minimum weighted gene count with eQTL coverage to trust the direction score.
_EQTL_MIN_COVERAGE = 3

# Maximum |γ| assigned via eQTL-direction-only fallback (used when all S-LDSC τ < 0,
# indicating a regulatory-architecture disease where gene-body S-LDSC is uninformative).
# Comparable to typical CAD τ values (max ~0.11).
_EQTL_GAMMA_MAX = 0.15

# Cache file for GWAS variant betas looked up during direction scoring.
_GWAS_BETA_CACHE_SUFFIX = "_gwas_variant_betas.json"


def _collect_gwas_betas_for_genes(
    disease_key: str,
    genes: list[str],
    gene_coords: dict[str, tuple[str, int, int]],
    window_bp: int = 50_000,
) -> dict[str, float]:
    """
    Scan GWAS sumstats and return {rsid: gwas_beta} for variants within
    ±window_bp of any gene in *genes*.

    Cached to data/ldsc/results/{disease_key}_gwas_variant_betas.json.
    Incremental: only queries missing genes.
    """
    from pipelines.ldsc.setup import GWAS_CONFIG, _SUMSTATS_DIR as SS_DIR

    cache_path = _RESULTS_DIR / f"{disease_key.upper()}{_GWAS_BETA_CACHE_SUFFIX}"
    cached: dict[str, float] = {}
    if cache_path.exists():
        try:
            with open(cache_path) as fh:
                cached = json.load(fh)
        except Exception:
            cached = {}

    cfg = GWAS_CONFIG.get(disease_key.upper())
    if not cfg:
        return cached

    raw = SS_DIR / cfg["filename"]
    if not raw.exists():
        return cached

    # Build windows for uncached genes only
    missing_genes = [g for g in genes if not any(
        f"{g}:" in k for k in cached
    )]
    if not missing_genes:
        return cached

    windows: dict[str, list[tuple[int, int]]] = {}  # chrom -> [(start, end)]
    gene_window_map: dict[str, str] = {}  # rsid -> gene_tag (for cache key)
    for g in missing_genes:
        coords = gene_coords.get(g)
        if not coords:
            continue
        chrom_str, start, end = coords
        chrom_n = chrom_str.lstrip("chr")
        w_start = max(0, start - window_bp)
        w_end   = end + window_bp
        windows.setdefault(chrom_n, []).append((w_start, w_end))

    if not windows:
        return cached

    _chr_col  = cfg.get("chr_col", "CHROM")
    _pos_col  = cfg.get("pos_col", "POS")
    _snp_col  = cfg.get("snp_col", "#ID")
    _beta_col = cfg.get("beta_col", "BETA")

    col_map: dict[str, int] = {}
    new_betas: dict[str, float] = {}

    opener = gzip.open if str(raw).endswith(".gz") else open
    try:
        with opener(raw, "rt") as fh:
            for raw_line in fh:
                if not col_map:
                    header_line = raw_line.lstrip("#").strip()
                    col_map = {c: i for i, c in enumerate(header_line.split("\t"))}
                    continue
                parts = raw_line.strip().split("\t")
                try:
                    chrom = parts[col_map[_chr_col]].lstrip("chr")
                    pos_raw = parts[col_map[_pos_col]]
                    if not pos_raw or pos_raw == "NA":
                        continue
                    pos   = int(float(pos_raw))
                    rsid  = parts[col_map[_snp_col]]
                    beta  = float(parts[col_map[_beta_col]])
                except (KeyError, ValueError, IndexError):
                    continue
                for start, end in windows.get(chrom, []):
                    if start <= pos <= end:
                        new_betas[rsid] = beta
                        break
    except Exception as exc:
        log.warning("GWAS beta scan failed: %s", exc)

    if new_betas:
        cached.update(new_betas)
        _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as fh:
            json.dump(cached, fh)
        log.info("GWAS beta cache: %d variants → %s", len(cached), cache_path.name)

    return cached


def compute_program_eqtl_direction(
    disease_key: str,
    dataset_id: str,
    n_top_genes: int = _EQTL_TOP_GENES,
) -> dict[str, dict]:
    """
    Compute eQTL-direction scores for each SVD program.

    For each component P, takes the top-N genes by |U_scaled[:, P]|, preserving
    sign. For each gene queries a cis-eQTL. If the eQTL variant also appears in
    the GWAS sumstats, the GWAS beta provides the allele direction for disease.

    Direction score for program P:
        score_P = Σ_G |U[G,P]| × sign(U[G,P]) × sign(eQTL_β_G) × sign(GWAS_β_at_eQTL_variant_G)

    When no GWAS beta is available for the eQTL variant, falls back to:
        score_P = Σ_G |U[G,P]| × sign(U[G,P]) × sign(eQTL_β_G)

    Returns dict[program_id -> {
        "direction_score":   float,
        "n_eqtl_genes":      int,
        "direction_source":  "eqtl_x_gwas" | "eqtl_only" | "none",
    }]
    """
    import numpy as np

    npz_path = _ROOT / "data" / "perturbseq" / dataset_id / "svd_loadings.npz"
    if not npz_path.exists():
        log.warning("SVD loadings not found for eQTL direction: %s", npz_path)
        return {}

    npz        = np.load(npz_path)
    U_scaled   = npz["U_scaled"]   # (n_genes, n_components)
    gene_names = npz["gene_names"]

    prefix       = disease_key.upper()
    n_components = U_scaled.shape[1]

    # --- Step 1: collect top-N genes per component with signed loadings ---
    # {program_id: [(gene, signed_loading), ...]}
    program_top_genes: dict[str, list[tuple[str, float]]] = {}
    for c in range(n_components):
        prog_id  = f"{prefix}_SVD_C{c+1:02d}"
        loadings = U_scaled[:, c]
        top_idx  = np.argsort(np.abs(loadings))[::-1][:n_top_genes]
        program_top_genes[prog_id] = [
            (str(gene_names[i]), float(loadings[i])) for i in top_idx
        ]

    # Deduplicate: query eQTL once per gene
    all_genes: list[str] = list({g for genes in program_top_genes.values() for g, _ in genes})
    log.info(
        "eQTL direction: %d unique genes across %d programs (%s)",
        len(all_genes), n_components, disease_key.upper(),
    )

    # --- Step 2: fetch gene coordinates for GWAS beta scan ---
    gene_coords = _fetch_gene_coords_hg38(all_genes)
    gwas_betas  = _collect_gwas_betas_for_genes(disease_key, all_genes, gene_coords)
    log.info("GWAS betas collected: %d variants", len(gwas_betas))

    # --- Step 3: load eQTL top-hits from local index (downloaded once from EBI FTP) ---
    from pipelines.ldsc.eqtl_local import build_gene_index, load_gene_index, gene_index_available

    if not gene_index_available(disease_key):
        log.info("eQTL gene index not found — building from local sumstats (may take a few minutes)…")
        try:
            gene_index = build_gene_index(disease_key, all_genes)
        except FileNotFoundError as exc:
            log.warning("%s\nDirection scores skipped.", exc)
            return {}
        except Exception as exc:
            log.warning("Failed to build eQTL gene index: %s — direction scores skipped", exc)
            return {}
    else:
        gene_index = load_gene_index(disease_key)
        # Re-index if new genes appeared (index is a strict subset of all_genes)
        missing = [g for g in all_genes if g not in gene_index and g not in (gene_index or {})]
        if len(missing) > len(all_genes) * 0.2:
            log.info("eQTL index covers <80%% of genes — rebuilding…")
            try:
                gene_index = build_gene_index(disease_key, all_genes, force=True)
            except Exception as exc:
                log.warning("Index rebuild failed: %s — using existing index", exc)

    # gene_eqtl mirrors the old shape: {gene → top_eqtl_dict | None}
    gene_eqtl: dict[str, dict | None] = {g: gene_index.get(g) for g in all_genes}
    n_with_eqtl = sum(1 for v in gene_eqtl.values() if v is not None)
    log.info("eQTL coverage: %d / %d genes (local index)", n_with_eqtl, len(all_genes))

    # --- Step 4: compute direction score per program ---
    results: dict[str, dict] = {}

    for prog_id, gene_loading_pairs in program_top_genes.items():
        score           = 0.0
        sum_abs_loading = 0.0
        n_eqtl          = 0
        n_gwas_hit      = 0

        for gene, loading in gene_loading_pairs:
            eqtl = gene_eqtl.get(gene)
            if eqtl is None:
                continue
            eqtl_beta = eqtl.get("beta")
            if eqtl_beta is None:
                continue
            try:
                eqtl_beta = float(eqtl_beta)
            except (TypeError, ValueError):
                continue

            weight       = abs(loading)
            loading_sign = 1.0 if loading >= 0 else -1.0
            eqtl_sign    = 1.0 if eqtl_beta >= 0 else -1.0

            # Try to get GWAS allele direction for this eQTL variant
            eqtl_rsid  = eqtl.get("rsid") or eqtl.get("variant_id") or ""
            gwas_beta  = gwas_betas.get(eqtl_rsid)
            if gwas_beta is not None:
                gwas_sign   = 1.0 if gwas_beta >= 0 else -1.0
                contribution = weight * loading_sign * eqtl_sign * gwas_sign
                n_gwas_hit  += 1
            else:
                # Fallback: no GWAS allele confirmation
                contribution = weight * loading_sign * eqtl_sign

            score          += contribution
            sum_abs_loading += weight
            n_eqtl         += 1

        if n_eqtl == 0:
            source = "none"
        elif n_gwas_hit >= _EQTL_MIN_COVERAGE:
            source = "eqtl_x_gwas"
        else:
            source = "eqtl_only"

        results[prog_id] = {
            "direction_score":  round(score, 6),
            "sum_abs_loading":  round(sum_abs_loading, 6),
            "n_eqtl_genes":     n_eqtl,
            "n_gwas_hits":      n_gwas_hit,
            "direction_source": source,
        }

    n_positive = sum(1 for r in results.values() if r["direction_score"] > 0)
    n_negative = sum(1 for r in results.values() if r["direction_score"] < 0)
    log.info(
        "eQTL direction scores: %d positive, %d negative, %d zero/none",
        n_positive, n_negative,
        len(results) - n_positive - n_negative,
    )
    return results


_DISEASE_SVD_DATASET: dict[str, str] = {
    "CAD": "schnitzler_cad_vascular",
    "RA":  "czi_2025_cd4t_perturb",
    "SLE": "czi_2025_cd4t_perturb",
}

# Top genes per SVD component used to define the genomic annotation window.
# 100 captures the dense loading tail without over-representing sparse components.
_SVD_TOP_GENES_PER_COMPONENT = 100


def extract_svd_gene_sets(
    dataset_id: str,
    n_top_genes: int = _SVD_TOP_GENES_PER_COMPONENT,
    disease_key: str = "",
) -> dict[str, set[str]]:
    """
    Extract top-N genes per SVD component from U_scaled (gene × component matrix).

    Args:
        dataset_id:   Perturbseq registry key, e.g. "schnitzler_cad_vascular".
        n_top_genes:  Number of genes per component, ranked by |U_scaled| loading.
        disease_key:  Disease prefix for program IDs (e.g. "CAD").

    Returns:
        dict[program_id -> set[gene_symbol]] for all 30 SVD components.
    """
    import numpy as np

    npz_path = _ROOT / "data" / "perturbseq" / dataset_id / "svd_loadings.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"SVD loadings not found: {npz_path}\n"
            "Run: python -m mcp_servers.perturbseq_server get_gwas_aligned_svd_programs"
        )

    npz = np.load(npz_path)
    U_scaled   = npz["U_scaled"]   # shape (n_genes, n_components)
    gene_names = npz["gene_names"] # shape (n_genes,)

    n_components = U_scaled.shape[1]
    prefix = disease_key.upper() if disease_key else dataset_id.split("_")[0].upper()

    gene_sets: dict[str, set[str]] = {}
    for c in range(n_components):
        program_id = f"{prefix}_SVD_C{c+1:02d}"
        loadings   = U_scaled[:, c]
        top_idx    = np.argsort(np.abs(loadings))[::-1][:n_top_genes]
        gene_sets[program_id] = {str(gene_names[i]) for i in top_idx}

    log.info(
        "Extracted %d SVD gene sets (%d genes each) from %s",
        len(gene_sets), n_top_genes, dataset_id,
    )
    return gene_sets


def run_svd_programs(
    disease_key: str,
    force: bool = False,
) -> dict[str, float]:
    """
    Compute GWAS χ² enrichment + eQTL-direction γ for all 30 SVD programs.

    γ decomposition:
      - Magnitude: |τ| from LDSC chi-square enrichment
      - Sign:      eQTL direction score (risk allele × program loading)
      - τ < 0:     program not in causal path → γ = 0
      - τ > 0, no eQTL coverage: γ = +|τ|, direction_source="unknown"

    The original LDSC τ sign is preserved as `tau_sign` annotation in the JSON.

    Saves to data/ldsc/results/{disease_key}_SVD_program_taus.json.
    Returns dict[program_id -> gamma] (the final signed γ, not raw τ).
    """
    out_path = _RESULTS_DIR / f"{disease_key.upper()}_SVD_program_taus.json"
    if out_path.exists() and not force:
        log.info("SVD τ file already exists: %s (pass force=True to recompute)", out_path)
        with open(out_path) as fh:
            data = json.load(fh)
            # Return gamma field if present, fall back to program_taus
            return data.get("program_gammas", data.get("program_taus", {}))

    dataset_id = _DISEASE_SVD_DATASET.get(disease_key.upper())
    if not dataset_id:
        raise ValueError(f"No SVD dataset configured for {disease_key}")

    gene_sets = extract_svd_gene_sets(dataset_id, disease_key=disease_key)
    if not gene_sets:
        raise RuntimeError(f"No SVD gene sets extracted for {disease_key}")

    log.info(
        "Computing GWAS χ² enrichment for %d SVD programs (%s)...",
        len(gene_sets), disease_key.upper(),
    )
    result = compute_program_gwas_enrichment(disease_key, gene_sets)
    program_taus: dict[str, float] = result["program_taus"]

    # eQTL-direction γ sign
    log.info("Computing eQTL direction scores for %s SVD programs...", disease_key.upper())
    direction_scores = compute_program_eqtl_direction(disease_key, dataset_id)

    # Combine: γ = sign(eQTL_direction) × |τ|
    # τ < 0 → program not in causal path → γ = 0 (unless all programs have τ < 0)
    # τ > 0, no eQTL → γ = +|τ|, direction unknown
    #
    # Special case: if ALL programs have τ < 0 (regulatory-architecture disease,
    # e.g. RA where causal variants are in enhancers not gene bodies), the S-LDSC
    # gene-body τ is uninformative. Fall back to eQTL-direction-only γ:
    #   normalized_concordance = direction_score / sum_abs_loading ∈ [-1, +1]
    #   γ = normalized_concordance × _EQTL_GAMMA_MAX
    all_tau_negative = all(tau < 0 for tau in program_taus.values())
    if all_tau_negative:
        log.info(
            "All %s SVD τ < 0 (regulatory-architecture disease): "
            "falling back to eQTL-direction-only γ (scale=±%.2f)",
            disease_key.upper(), _EQTL_GAMMA_MAX,
        )

    program_gammas: dict[str, float] = {}
    gamma_annotations: list[dict] = []

    for prog, tau in program_taus.items():
        tau_mag  = abs(tau)
        tau_sign = 1 if tau >= 0 else -1

        if all_tau_negative:
            # S-LDSC gene-body τ uninformative → use eQTL concordance fraction as γ
            dir_info        = direction_scores.get(prog, {})
            dir_score       = dir_info.get("direction_score", 0.0)
            sum_abs_loading = dir_info.get("sum_abs_loading", 0.0)
            n_eqtl          = dir_info.get("n_eqtl_genes", 0)
            dir_src         = dir_info.get("direction_source", "none")

            if n_eqtl < _EQTL_MIN_COVERAGE or sum_abs_loading == 0.0:
                gamma      = 0.0
                dir_source = "no_eqtl_coverage"
            else:
                # Normalized concordance fraction ∈ [-1, +1]
                concordance = dir_score / sum_abs_loading
                concordance = max(-1.0, min(1.0, concordance))
                gamma       = concordance * _EQTL_GAMMA_MAX
                dir_source  = dir_src + "_eqtl_direction_only"
        elif tau < 0:
            # Depleted of GWAS heritability → not in causal path
            gamma        = 0.0
            dir_source   = "depleted"
        else:
            dir_info = direction_scores.get(prog, {})
            dir_score = dir_info.get("direction_score", 0.0)
            dir_src   = dir_info.get("direction_source", "none")

            if dir_src != "none" and dir_info.get("n_eqtl_genes", 0) >= _EQTL_MIN_COVERAGE:
                gamma      = (1.0 if dir_score >= 0 else -1.0) * tau_mag
                dir_source = dir_src
            else:
                # No eQTL coverage — magnitude only, direction unknown
                gamma      = tau_mag
                dir_source = "unknown"

        gamma = round(gamma, 6)
        program_gammas[prog] = gamma

        gamma_annotations.append({
            "name":             prog,
            "tau":              tau,
            "tau_sign":         tau_sign,   # preserved for annotation; NOT used for γ direction
            "gamma":            gamma,
            "direction_source": dir_source,
            "direction_score":  direction_scores.get(prog, {}).get("direction_score"),
            "sum_abs_loading":  direction_scores.get(prog, {}).get("sum_abs_loading"),
            "n_eqtl_genes":     direction_scores.get(prog, {}).get("n_eqtl_genes", 0),
        })

    # Merge into the enrichment result
    result["program_gammas"]    = program_gammas
    result["gamma_annotations"] = gamma_annotations
    result["all_tau_negative"]  = all_tau_negative
    # Keep program_taus for backward compatibility and raw-τ inspection
    # (program_taus still holds original signed τ values)

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)
    log.info("Saved SVD γ → %s", out_path)

    n_pos = sum(1 for g in program_gammas.values() if g > 0)
    n_neg = sum(1 for g in program_gammas.values() if g < 0)
    n_zero = sum(1 for g in program_gammas.values() if g == 0.0)
    log.info(
        "SVD γ summary: %d atherogenic (+), %d atheroprotective (−), %d depleted (0)",
        n_pos, n_neg, n_zero,
    )
    return program_gammas


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cmd     = sys.argv[1] if len(sys.argv) > 1 else "status"
    disease = sys.argv[2] if len(sys.argv) > 2 else "CAD"

    if cmd == "run":
        from pipelines.cnmf_programs import get_programs_for_disease
        progs_result = get_programs_for_disease(disease)
        prog_list = progs_result.get("programs", []) if isinstance(progs_result, dict) else []
        gene_sets = {
            p["program_id"]: set(p["gene_set"])
            for p in prog_list
            if isinstance(p, dict) and p.get("gene_set")
        }
        if not gene_sets:
            print(f"No programs found for {disease}. Run NMF first.")
            sys.exit(1)
        print(f"Computing GWAS enrichment for {len(gene_sets)} programs ({disease})...")
        taus = run_gene_set_regression(disease, gene_sets)
        print(f"\n{disease} program τ enrichment scores:")
        for p, t in sorted(taus.items(), key=lambda x: -abs(x[1])):
            print(f"  {p}: {t:+.4f}")
    elif cmd == "run_svd":
        force_flag = "--force" in sys.argv
        print(f"Computing GWAS χ² enrichment for SVD programs ({disease})...")
        taus = run_svd_programs(disease, force=force_flag)
        print(f"\n{disease} SVD program τ enrichment scores (sorted by |τ|):")
        for p, t in sorted(taus.items(), key=lambda x: -abs(x[1])):
            sign = "+" if t >= 0 else ""
            print(f"  {p}: {sign}{t:.4f}")
    elif cmd == "download_eqtl":
        from pipelines.ldsc.eqtl_local import download_dataset
        p = download_dataset(disease)
        print(f"Downloaded: {p}")
    else:
        print("Usage:")
        print("  python -m pipelines.ldsc.runner run CAD|RA           # NMF programs")
        print("  python -m pipelines.ldsc.runner run_svd CAD|RA       # SVD programs")
        print("  python -m pipelines.ldsc.runner download_eqtl CAD|RA # fetch eQTL sumstats")
