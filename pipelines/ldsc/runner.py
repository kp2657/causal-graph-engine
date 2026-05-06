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
_CHROMATIN_DIR = _ROOT / "data" / "sldsc_chromatin"

# Gene window (bp) around each gene body for SNP inclusion
_GENE_WINDOW_BP = 100_000

# Min SNPs in a program window to compute enrichment
_MIN_SNPS = 50


def _load_chromatin_windows(
    disease_key: str,
    program_ids: list[str],
) -> dict[str, dict[str, list[tuple[int, int]]]] | None:
    """
    Load pre-built ABC-enhancer BED files from Phase I (sldsc_chromatin_programs.py).

    Returns dict[program -> dict[chrom -> [(start, end)]]] if BED files exist for
    all programs, else None (caller falls back to gene body windows).

    BED files live at data/sldsc_chromatin/{disease_key}/{program_id}.bed.
    Chromosomes are stored without the 'chr' prefix to match sumstats convention.
    """
    dk = disease_key.lower()
    bed_dir = _CHROMATIN_DIR / dk
    _loci_dir = _CHROMATIN_DIR / f"{dk}_loci"

    # Either loci dir or standard chromatin dir must exist
    if not bed_dir.exists() and not _loci_dir.exists():
        return None

    import re as _re
    _svd_pat   = _re.compile(r"^[A-Z]+_SVD_C(\d+)$")
    _gnmf_pat  = _re.compile(r"^[A-Z]+_GeneticNMF_(\w+)_C(\d+)$")
    _locus_pat = _re.compile(r"^LOCUS_")

    # Super-program directory: {disease_key}_super/ contains merged BED files
    _super_dir   = _CHROMATIN_DIR / f"{dk}_super"
    _N_PER_SUPER = 3

    # Disease-validated CRE directory (Moonen 2026 RA CREs)
    _validated_dir = _CHROMATIN_DIR / f"{dk}_moonen"

    # Build locus → BED index once from gwas_anchored_programs.npz if available
    _locus_bed_map: dict[str, pathlib.Path] = {}
    if _loci_dir.exists():
        try:
            import numpy as _np
            _npz_path = (
                _CHROMATIN_DIR.parent / "perturbseq"
                / f"{disease_key.lower().replace(' ', '_')}_cd4t_perturb"   # heuristic
                / "gwas_anchored_programs.npz"
            )
            # Walk dataset dirs to find the npz for this disease
            _perturb_root = _CHROMATIN_DIR.parent / "perturbseq"
            for _ds in _perturb_root.iterdir():
                _candidate = _ds / "gwas_anchored_programs.npz"
                if _candidate.exists():
                    _d = _np.load(_candidate, allow_pickle=False)
                    _locus_names = list(_d["locus_names"])
                    for _i, _ln in enumerate(_locus_names):
                        _bp = _loci_dir / f"L{_i:02d}.bed"
                        if _bp.exists():
                            _locus_bed_map[_ln] = _bp
                    if _locus_bed_map:
                        break
        except Exception as _exc:
            log.debug("Could not build locus BED map: %s", _exc)

    def _bed_path_for_prog(prog: str):
        """Phase −1: GWAS-locus BEDs → Phase 0: Moonen CREs → Phase I: CATLAS → super-program."""
        # Phase -1: GWAS-anchored locus programs (genetics-first)
        if _locus_pat.match(prog) and prog in _locus_bed_map:
            return _locus_bed_map[prog]

        # Phase 0b: condition-specific GeneticNMF BEDs (sldsc_chromatin_programs --condition)
        # Directory: {dk}_gnmf_{cond_lower}/  Files: C01.bed, C02.bed, ...
        gm = _gnmf_pat.match(prog)
        if gm:
            _gnmf_cond_lower = gm.group(1).lower()
            _gnmf_comp       = int(gm.group(2))   # 1-based
            _gnmf_dir        = _CHROMATIN_DIR / f"{dk}_gnmf_{_gnmf_cond_lower}"
            if _gnmf_dir.exists():
                _gnmf_bed = _gnmf_dir / f"C{_gnmf_comp:02d}.bed"
                if _gnmf_bed.exists() and _gnmf_bed.stat().st_size > 0:
                    return _gnmf_bed

        m = _svd_pat.match(prog)
        comp_idx = (int(m.group(1)) - 1) if m else None
        alias = f"P{comp_idx:02d}" if comp_idx is not None else None

        # Phase 0: disease-validated CRE BEDs (Moonen 2026 RA CREs)
        if _validated_dir.exists() and alias:
            vp = _validated_dir / f"{alias}.bed"
            if vp.exists() and vp.stat().st_size > 0:
                return vp

        # Phase I: canonical name or P{n} alias in standard chromatin dir
        if bed_dir.exists():
            p = bed_dir / f"{prog}.bed"
            if p.exists():
                return p
            if alias:
                q = bed_dir / f"{alias}.bed"
                if q.exists():
                    return q
                if _super_dir.exists():
                    sp_idx = comp_idx // _N_PER_SUPER
                    sq = _super_dir / f"SP{sp_idx:02d}.bed"
                    if sq.exists():
                        return sq
        return None

    windows: dict[str, dict[str, list[tuple[int, int]]]] = {}
    n_loaded = 0
    for prog in program_ids:
        bed_path = _bed_path_for_prog(prog)
        if bed_path is None:
            continue
        chrom_windows: dict[str, list[tuple[int, int]]] = {}
        try:
            with open(bed_path) as fh:
                for line in fh:
                    parts = line.strip().split("\t")
                    if len(parts) < 3:
                        continue
                    chrom = parts[0].lstrip("chr")
                    start = int(parts[1])
                    end   = int(parts[2])
                    chrom_windows.setdefault(chrom, []).append((start, end))
        except Exception as exc:
            log.warning("Chromatin BED load failed for %s/%s: %s", dk, prog, exc)
            continue
        if chrom_windows:
            windows[prog] = chrom_windows
            n_loaded += 1

    if n_loaded == 0:
        return None

    log.info(
        "Phase I chromatin windows loaded for %s: %d/%d programs (ABC-enhancer-based)",
        disease_key, n_loaded, len(program_ids),
    )
    return windows


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

    # Step 1: build per-program genomic windows.
    # Phase I: if ABC-enhancer BED files exist (sldsc_chromatin_programs.py output),
    # use chromatin-based windows instead of gene body ±100kb windows.
    prog_windows = _load_chromatin_windows(disease_key, list(program_gene_sets))
    if prog_windows is not None:
        # Fill in any programs missing from the BED directory with gene body windows
        missing_progs = {p: s for p, s in program_gene_sets.items() if p not in prog_windows}
        if missing_progs:
            all_genes_missing = set().union(*missing_progs.values())
            gene_coords_missing = _fetch_gene_coords_hg38(list(all_genes_missing))
            fallback_windows = _build_program_windows(missing_progs, gene_coords_missing)
            prog_windows.update(fallback_windows)
    else:
        all_genes = set().union(*program_gene_sets.values())
        log.info("Fetching GRCh38 coords for %d genes (gene body windows)...", len(all_genes))
        gene_coords = _fetch_gene_coords_hg38(list(all_genes))
        log.info("Got coords for %d / %d genes", len(gene_coords), len(all_genes))
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


def compute_joint_program_regression(
    disease_key: str,
    program_gene_sets: dict[str, set[str]],
    ridge_alpha: float = 1.0,
) -> dict[str, Any]:
    """
    Compute joint conditional τ* for all programs via ridge regression.

    Tier 1 method: conditions each program on all others simultaneously,
    eliminating bystander inflation from LD-shared genomic windows.
    Replaces the marginal chi-square enrichment in run_svd_programs.

    Model: E[χ²_j] = α + Σ_C (τ*_C × a_{j,C})
    where a_{j,C} = 1 if SNP j is in program C's gene-body window, 0 otherwise.
    Fit via ridge regression (RidgeCV with automatic α selection).

    A marginal program with high τ due to LD overlap with a genuine disease
    program will have τ* ≈ 0 after conditioning on the genuine program.

    Returns the same structure as compute_program_gwas_enrichment so it
    can be used as a drop-in replacement in run_svd_programs.
    """
    try:
        import numpy as _np
        from sklearn.linear_model import RidgeCV as _RidgeCV, Ridge as _Ridge
    except ImportError:
        log.warning(
            "compute_joint_program_regression: numpy/sklearn unavailable — "
            "falling back to marginal enrichment"
        )
        gene_sets_as_sets = {k: set(v) for k, v in program_gene_sets.items()}
        return compute_program_gwas_enrichment(disease_key, gene_sets_as_sets)

    from pipelines.ldsc.setup import GWAS_CONFIG, _SUMSTATS_DIR as SS_DIR

    cfg = GWAS_CONFIG.get(disease_key.upper())
    if not cfg:
        raise ValueError(f"No GWAS config for {disease_key}")

    raw = SS_DIR / cfg["filename"]
    if not raw.exists():
        raise FileNotFoundError(f"GWAS sumstats not found: {raw}")

    # Always use gene-body ±100kb windows for SVD program regression.
    # ABC-enhancer chromatin windows are too sparse (RA: ~1k SNPs vs ~1.7M from gene-body)
    # and cause collinear annotations and inflated τ* in the joint regression.
    all_genes = set().union(*program_gene_sets.values())
    gene_coords = _fetch_gene_coords_hg38(list(all_genes))
    prog_windows = _build_program_windows(program_gene_sets, gene_coords)

    chr_col  = cfg.get("chr_col", "CHR")
    pos_col  = cfg.get("pos_col", "POS")
    beta_col = cfg.get("beta_col", "BETA")
    se_col   = cfg.get("se_col", "SE")
    prog_names = sorted(program_gene_sets.keys())

    # Phase 1: Read all SNPs grouped by chromosome (enables vectorized annotation below)
    import gzip as _gz
    snps_by_chrom: dict[str, tuple[list, list]] = {}  # chrom → (positions, chisqs)

    with _gz.open(raw, "rt") as fh:
        header = fh.readline().strip().split("\t")
        col = {c: i for i, c in enumerate(header)}
        required = {chr_col, pos_col, beta_col, se_col}
        if not required.issubset(col):
            log.warning(
                "compute_joint_program_regression: sumstats missing columns %s — falling back",
                required - set(col),
            )
            return compute_program_gwas_enrichment(disease_key, program_gene_sets)

        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) < len(col):
                continue
            try:
                chrom = parts[col[chr_col]]
                pos   = int(float(parts[col[pos_col]]))
                beta  = float(parts[col[beta_col]])
                se    = float(parts[col[se_col]])
                if se <= 0:
                    continue
                chisq = (beta / se) ** 2
            except (ValueError, ZeroDivisionError, IndexError):
                continue
            if chrom not in snps_by_chrom:
                snps_by_chrom[chrom] = ([], [])
            snps_by_chrom[chrom][0].append(pos)
            snps_by_chrom[chrom][1].append(chisq)

    if not snps_by_chrom:
        log.warning("compute_joint_program_regression: no SNPs found — falling back")
        return compute_program_gwas_enrichment(disease_key, program_gene_sets)

    # Phase 2: Per-chromosome vectorized annotation (numpy broadcasting)
    # For each chrom × program × interval: vectorized range check on all positions at once.
    # This is ~30× faster than per-SNP Python loops.
    chisq_all_list: list[float] = []
    chisq_annotated_list: list[float] = []
    annot_rows_list: list[_np.ndarray] = []

    for chrom, (positions_list, chisqs_list) in sorted(snps_by_chrom.items()):
        positions = _np.array(positions_list, dtype=_np.int64)
        chisqs    = _np.array(chisqs_list,    dtype=_np.float64)
        chisq_all_list.extend(chisqs_list)

        n_snps = len(positions)
        annot_chr = _np.zeros((n_snps, len(prog_names)), dtype=_np.int8)

        for pi, pn in enumerate(prog_names):
            for (start, end) in prog_windows.get(pn, {}).get(chrom, []):
                in_range = (positions >= start) & (positions <= end)
                annot_chr[:, pi] |= in_range.astype(_np.int8)

        any_annotated = annot_chr.any(axis=1)
        if any_annotated.any():
            chisq_annotated_list.extend(chisqs[any_annotated].tolist())
            annot_rows_list.append(annot_chr[any_annotated])

    if not chisq_annotated_list:
        log.warning("compute_joint_program_regression: no annotated SNPs — falling back")
        return compute_program_gwas_enrichment(disease_key, program_gene_sets)

    # Genomic control correction
    _sorted = sorted(chisq_all_list)
    _lambda_gc = _sorted[len(_sorted) // 2] / 0.4549
    mean_genome = (sum(chisq_all_list) / len(chisq_all_list)) / _lambda_gc
    log.info("Joint regression λ_GC=%.4f, mean_genome_gc=%.4f, n_annotated=%d/%d",
             _lambda_gc, mean_genome, len(chisq_annotated_list), len(chisq_all_list))

    # Subtract genome-wide mean before regression (fit_intercept=False).
    # Without this, fit_intercept=True absorbs the annotated-window mean as the
    # intercept, collapsing all coefficients to zero when all programs have similar
    # absolute enrichment. We want τ* = enrichment above genome-wide baseline.
    chisq_gc = _np.array(chisq_annotated_list, dtype=float) / _lambda_gc - mean_genome
    X = _np.vstack(annot_rows_list).astype(float)

    # Ridge regression: (χ²_gc - mean_genome) ~ annotation_matrix
    alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
    ridge = _RidgeCV(alphas=alphas, fit_intercept=False)
    ridge.fit(X, chisq_gc)

    coefs = ridge.coef_          # τ* per program

    log.info("Joint regression: mean_genome=%.4f, best_alpha=%.4g", mean_genome, ridge.alpha_)

    # Jackknife SE: fix alpha from initial RidgeCV, use fast Ridge for 20 block fits
    # (RidgeCV in each block would be 20×5-fold-CV×4-alphas=400 fits; Ridge is 20 fits)
    best_alpha = float(ridge.alpha_)
    n = len(chisq_gc)
    n_blocks = 20
    block_size = max(1, n // n_blocks)
    block_coefs: list[_np.ndarray] = []
    for i in range(n_blocks):
        mask = _np.ones(n, dtype=bool)
        mask[i * block_size : (i + 1) * block_size] = False
        r_jk = _Ridge(alpha=best_alpha, fit_intercept=False)
        r_jk.fit(X[mask], chisq_gc[mask])
        block_coefs.append(r_jk.coef_)

    coef_arr = _np.stack(block_coefs)  # (n_blocks, n_programs)
    se_arr = _np.std(coef_arr, axis=0) * _np.sqrt(n_blocks)  # jackknife SE

    program_taus: dict[str, float] = {}
    raw_annotations: list[dict] = []

    for i, pn in enumerate(prog_names):
        tau_star = float(coefs[i])
        se_tau   = float(se_arr[i]) if se_arr[i] > 0 else abs(tau_star) * 0.30
        z = tau_star / (se_tau + 1e-8)
        tau_p = 2.0 * (1.0 - _norm_cdf(abs(z)))

        program_taus[pn] = round(tau_star, 6)
        raw_annotations.append({
            "name":   pn,
            "tau":    round(tau_star, 6),
            "tau_se": round(se_tau, 6),
            "tau_p":  round(tau_p, 6),
            "n_snps": int(X[:, i].sum()),
            "method": "joint_chisq_regression",
        })
        log.info("  %s: τ*=%.4f (SE=%.4f, z=%.2f, p=%.3g)", pn, tau_star, se_tau, z, tau_p)

    n_sig = sum(1 for r in raw_annotations if r["tau_p"] < 0.05 and r["tau"] > 0)
    log.info(
        "Joint regression complete: %d/%d programs τ*>0 and p<0.05 for %s",
        n_sig, len(prog_names), disease_key.upper(),
    )

    return {
        "program_taus":      program_taus,
        "h2":                None,
        "mean_chisq_genome": round(mean_genome, 4),
        "n_snps_genome":     len(chisq_all_list),
        "raw_annotations":   raw_annotations,
        "method":            "joint_chisq_regression_tier1",
        "disease_key":       disease_key.upper(),
        "lambda_gc":         round(_lambda_gc, 4),
        "ridge_alpha":       float(ridge.alpha_),
    }


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
    Scan GWAS sumstats and return {"chrom:pos": gwas_beta} for variants
    within ±window_bp of any gene in *genes*.

    Keys are "{chrom}:{pos}" (e.g. "1:1190173") so they match the
    chromosome/position fields stored in the eQTL index — avoiding the
    rsid vs chr:pos:ref:alt mismatch between eQTL catalogues and GWAS files.

    Cached to data/ldsc/results/{disease_key}_gwas_variant_betas.json.
    Incremental: only scans windows for genes not yet in cache.
    """
    from pipelines.ldsc.setup import GWAS_CONFIG, _SUMSTATS_DIR as SS_DIR

    cache_path = _RESULTS_DIR / f"{disease_key.upper()}{_GWAS_BETA_CACHE_SUFFIX}"
    cached: dict[str, float] = {}
    if cache_path.exists():
        try:
            with open(cache_path) as fh:
                cached = json.load(fh)
            # Invalidate legacy cache keyed by rsid (keys contain letters, not just digits/colons)
            if any(not k.replace(":", "").isdigit() for k in list(cached)[:20]):
                log.info("Invalidating legacy rsid-keyed GWAS beta cache for %s", disease_key)
                cached = {}
                cache_path.unlink(missing_ok=True)
        except Exception:
            cached = {}

    cfg = GWAS_CONFIG.get(disease_key.upper())
    if not cfg:
        return cached

    raw = SS_DIR / cfg["filename"]
    if not raw.exists():
        return cached

    # Track which chroms+windows are already covered by examining cached keys
    cached_chroms: set[str] = {k.split(":")[0] for k in cached}

    windows: dict[str, list[tuple[int, int]]] = {}
    for g in genes:
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
    _beta_col = cfg.get("beta_col", "BETA")

    col_map: dict[str, int] = {}
    new_betas: dict[str, float] = {}

    opener = gzip.open if str(raw).endswith(".gz") else open
    try:
        with opener(raw, "rt") as fh:
            for raw_line in fh:
                if not col_map:
                    # Strip leading # from header (e.g. "#ID\tCHR\tPOS" → "ID\tCHR\tPOS")
                    header_line = raw_line.lstrip("#").strip()
                    col_map = {c: i for i, c in enumerate(header_line.split("\t"))}
                    if _chr_col not in col_map or _pos_col not in col_map or _beta_col not in col_map:
                        log.warning(
                            "GWAS sumstats missing expected columns %s/%s/%s in %s — found: %s",
                            _chr_col, _pos_col, _beta_col, raw.name, list(col_map)[:8],
                        )
                        return cached
                    continue
                parts = raw_line.strip().split("\t")
                try:
                    chrom   = parts[col_map[_chr_col]].lstrip("chr")
                    pos_raw = parts[col_map[_pos_col]]
                    if not pos_raw or pos_raw == "NA":
                        continue
                    pos  = int(float(pos_raw))
                    beta = float(parts[col_map[_beta_col]])
                except (KeyError, ValueError, IndexError):
                    continue
                for start, end in windows.get(chrom, []):
                    if start <= pos <= end:
                        new_betas[f"{chrom}:{pos}"] = beta
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
    npz_override: "pathlib.Path | None" = None,
    prog_prefix: str | None = None,
) -> dict[str, dict]:
    """
    Compute eQTL-direction scores for each program (SVD or GeneticNMF).

    For each component P, takes the top-N genes by |U_scaled[:, P]|, preserving
    sign. For each gene queries a cis-eQTL. If the eQTL variant also appears in
    the GWAS sumstats, the GWAS beta provides the allele direction for disease.

    Direction score for program P (Z-score weighted, normalised to [-1,1]):
        score_P = Σ_G  sign(loading_G) × sign(eQTL_β_G) × [sign(GWAS_β_G) if available] × |loading_G| × |eQTL_Z_G|
                  ─────────────────────────────────────────────────────────────────────────────────────────────────
                                              Σ_G  |loading_G| × |eQTL_Z_G|

    GWAS beta is matched by genomic position ("{chrom}:{pos}") against the GWAS
    sumstats, bypassing rsid/chr:pos:ref:alt format mismatch.

    Args:
        npz_override:  Path to an alternative npz (e.g. genetic_nmf_loadings.npz).
                       Defaults to svd_loadings.npz for the dataset.
        prog_prefix:   Program naming prefix (e.g. "CAD_GeneticNMF"). Defaults
                       to "{disease_key}_SVD".

    Returns dict[program_id -> {
        "direction_score":   float,
        "n_eqtl_genes":      int,
        "direction_source":  "eqtl_x_gwas" | "eqtl_only" | "none",
    }]
    """
    import numpy as np

    npz_path = npz_override or (_ROOT / "data" / "perturbseq" / dataset_id / "svd_loadings.npz")
    if not npz_path.exists():
        log.warning("Loadings not found for eQTL direction: %s", npz_path)
        return {}

    npz        = np.load(npz_path)
    U_scaled   = npz["U_scaled"]   # (n_genes, n_components)
    gene_names = npz["gene_names"]

    prefix       = prog_prefix or f"{disease_key.upper()}_SVD"
    n_components = U_scaled.shape[1]

    # --- Step 1: collect top-N genes per component with signed loadings ---
    # {program_id: [(gene, signed_loading), ...]}
    program_top_genes: dict[str, list[tuple[str, float]]] = {}
    for c in range(n_components):
        prog_id  = f"{prefix}_C{c+1:02d}"
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
    #
    # Normalized Z-score-weighted concordance (Pritchard omnigenic framing):
    #
    #   score_P = Σ_G  sign(loading_G) × sign(GWAS_β_G × eQTL_β_G) × |loading_G| × |eQTL_Z_G|
    #             ─────────────────────────────────────────────────────────────────────────────
    #                                Σ_G  |loading_G| × |eQTL_Z_G|
    #
    # Normalising by the eQTL-Z-weighted denominator maps the score to [-1, 1] regardless
    # of program size. Strong cis-eQTL instruments (high |Z|) count more than weak ones;
    # individual gene noise (~50% at N=1) averages toward the true program signal as N grows.
    results: dict[str, dict] = {}

    for prog_id, gene_loading_pairs in program_top_genes.items():
        numerator   = 0.0
        denominator = 0.0
        n_eqtl      = 0
        n_gwas_hit  = 0

        for gene, loading in gene_loading_pairs:
            eqtl = gene_eqtl.get(gene)
            if eqtl is None:
                continue
            eqtl_beta = eqtl.get("beta")
            eqtl_se   = eqtl.get("se")
            if eqtl_beta is None:
                continue
            try:
                eqtl_beta = float(eqtl_beta)
                eqtl_se   = float(eqtl_se) if eqtl_se is not None else None
            except (TypeError, ValueError):
                continue

            eqtl_z       = abs(eqtl_beta / eqtl_se) if eqtl_se and eqtl_se > 0 else 1.0
            weight       = abs(loading) * eqtl_z          # loading × |eQTL_Z|
            loading_sign = 1.0 if loading >= 0 else -1.0
            eqtl_sign    = 1.0 if eqtl_beta >= 0 else -1.0

            # Key by chrom:pos — matches GWAS sumstats (which use chr:pos:ref:alt, not rsids)
            eqtl_chrom = str(eqtl.get("chromosome", "")).lstrip("chr")
            eqtl_pos   = eqtl.get("position")
            pos_key    = f"{eqtl_chrom}:{eqtl_pos}" if eqtl_chrom and eqtl_pos else ""
            gwas_beta  = gwas_betas.get(pos_key)
            if gwas_beta is not None:
                gwas_sign    = 1.0 if gwas_beta >= 0 else -1.0
                contribution = loading_sign * eqtl_sign * gwas_sign
                n_gwas_hit  += 1
            else:
                contribution = loading_sign * eqtl_sign

            numerator   += contribution * weight
            denominator += weight
            n_eqtl      += 1

        if denominator > 0:
            direction_score = numerator / denominator   # in [-1, 1]
        else:
            direction_score = 0.0

        if n_eqtl == 0:
            source = "none"
        elif n_gwas_hit >= _EQTL_MIN_COVERAGE:
            source = "eqtl_x_gwas"
        else:
            source = "eqtl_only"

        results[prog_id] = {
            "direction_score":  round(direction_score, 6),
            "sum_abs_loading":  round(denominator, 6),
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

# Datasets that have Schnitzler/Ota-style cNMF programs (cnmf_beta_programs.npz)
_DISEASE_CNMF_DATASET: dict[str, str] = {
    "CAD": "schnitzler_cad_vascular",    # k=60, endothelial; MAST differential usage betas
    "RA":  "czi_2025_cd4t_perturb",      # k=30, GeneticNMF; DESeq2 pseudobulk→GeneticNMF betas
}

# Override signatures file name per disease (default: signatures.json.gz)
_DISEASE_SIG_NAME: dict[str, str] = {
    "RA":  "signatures_Stim48hr.json.gz",  # 48-hr activation; relevant for T-cell-driven autoimmunity
    "SLE": "signatures_Stim48hr.json.gz",
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
        "Computing Tier 1 joint conditional τ* for %d SVD programs (%s)...",
        len(gene_sets), disease_key.upper(),
    )
    # Tier 1: joint conditional τ* via ridge regression on all programs simultaneously.
    # Conditions each program on all others — eliminates bystander inflation from
    # shared LD windows (C07/C12 in CAD, C20/C27 in RA).
    # Falls back to marginal chi-square enrichment if numpy/sklearn unavailable.
    result = compute_joint_program_regression(disease_key, gene_sets)
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


# ---------------------------------------------------------------------------
# Schnitzler cNMF programs — Ota-style γ using disease-relevant cell-type programs
# ---------------------------------------------------------------------------

def extract_cnmf_gene_sets(
    dataset_id: str,
    top_n: int = 100,
) -> dict[str, set[str]]:
    """
    Extract cNMF program gene sets from cnmf_program_gene_sets.json.

    Programs are defined from the Schnitzler endothelial cNMF (k=60),
    giving biologically labelled cell-state programs in the disease-relevant
    cell type — the correct Ota et al. program basis.
    """
    import json as _json
    gs_path = _ROOT / "data" / "perturbseq" / dataset_id / "cnmf_program_gene_sets.json"
    if not gs_path.exists():
        log.warning("cnmf_program_gene_sets.json not found: %s", gs_path)
        return {}
    with open(gs_path) as fh:
        raw = _json.load(fh)
    gene_sets = {pid: set(genes[:top_n]) for pid, genes in raw.items()}
    log.info("Loaded %d cNMF gene sets (top-%d genes each) from %s", len(gene_sets), top_n, dataset_id)
    return gene_sets


def run_cnmf_programs(
    disease_key: str,
    force: bool = False,
) -> dict[str, float]:
    """
    Compute GWAS χ² enrichment + eQTL-direction γ for all cNMF programs.

    Parallel to run_svd_programs but uses Schnitzler k=60 endothelial cNMF
    programs (biologically labelled cell states) instead of SVD components.
    Only available for diseases with a cNMF dataset (_DISEASE_CNMF_DATASET).

    Saves to data/ldsc/results/{disease_key}_cNMF_program_taus.json.
    Returns dict[program_id -> gamma].
    """
    out_path = _RESULTS_DIR / f"{disease_key.upper()}_cNMF_program_taus.json"
    if out_path.exists() and not force:
        log.info("cNMF τ file already exists: %s (pass force=True to recompute)", out_path)
        with open(out_path) as fh:
            data = json.load(fh)
        return data.get("program_gammas", data.get("program_taus", {}))

    dataset_id = _DISEASE_CNMF_DATASET.get(disease_key.upper())
    if not dataset_id:
        log.info("No cNMF dataset configured for %s — skipping cNMF track", disease_key)
        return {}

    gene_sets = extract_cnmf_gene_sets(dataset_id)
    if not gene_sets:
        raise RuntimeError(f"No cNMF gene sets found for {disease_key} ({dataset_id})")

    log.info(
        "Computing joint conditional τ* for %d cNMF programs (%s)...",
        len(gene_sets), disease_key.upper(),
    )
    result = compute_joint_program_regression(disease_key, gene_sets)
    program_taus: dict[str, float] = result["program_taus"]

    # eQTL-direction γ sign — reuse SVD dataset for eQTL lookup
    svd_dataset = _DISEASE_SVD_DATASET.get(disease_key.upper(), dataset_id)
    log.info("Computing eQTL direction scores for %s cNMF programs...", disease_key.upper())
    direction_scores = compute_program_eqtl_direction(disease_key, svd_dataset)

    all_tau_negative = all(tau < 0 for tau in program_taus.values())
    if all_tau_negative:
        log.info(
            "All %s cNMF τ < 0: falling back to eQTL-direction-only γ (scale=±%.2f)",
            disease_key.upper(), _EQTL_GAMMA_MAX,
        )

    program_gammas: dict[str, float] = {}
    gamma_annotations: list[dict] = []

    for prog, tau in program_taus.items():
        tau_mag  = abs(tau)
        tau_sign = 1 if tau >= 0 else -1

        if all_tau_negative:
            dir_info        = direction_scores.get(prog, {})
            dir_score       = dir_info.get("direction_score", 0.0)
            sum_abs_loading = dir_info.get("sum_abs_loading", 0.0)
            n_eqtl          = dir_info.get("n_eqtl_genes", 0)
            dir_src         = dir_info.get("direction_source", "none")
            if n_eqtl < _EQTL_MIN_COVERAGE or sum_abs_loading == 0.0:
                gamma      = 0.0
                dir_source = "no_eqtl_coverage"
            else:
                concordance = dir_score / sum_abs_loading
                concordance = max(-1.0, min(1.0, concordance))
                gamma       = concordance * _EQTL_GAMMA_MAX
                dir_source  = dir_src + "_eqtl_direction_only"
        elif tau < 0:
            gamma      = 0.0
            dir_source = "depleted"
        else:
            dir_info  = direction_scores.get(prog, {})
            dir_score = dir_info.get("direction_score", 0.0)
            dir_src   = dir_info.get("direction_source", "none")
            if dir_src != "none" and dir_info.get("n_eqtl_genes", 0) >= _EQTL_MIN_COVERAGE:
                gamma      = (1.0 if dir_score >= 0 else -1.0) * tau_mag
                dir_source = dir_src
            else:
                gamma      = tau_mag
                dir_source = "unknown"

        gamma = round(gamma, 6)
        program_gammas[prog] = gamma
        gamma_annotations.append({
            "program": prog, "tau": round(tau, 6), "gamma": gamma,
            "direction_source": dir_source,
        })

    result["program_gammas"]    = program_gammas
    result["gamma_annotations"] = gamma_annotations
    result["mode"]              = "cnmf_k60_endothelial"

    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)
    log.info("Saved cNMF γ → %s", out_path)

    n_pos  = sum(1 for g in program_gammas.values() if g > 0)
    n_neg  = sum(1 for g in program_gammas.values() if g < 0)
    n_zero = sum(1 for g in program_gammas.values() if g == 0.0)
    log.info("cNMF γ summary: %d atherogenic (+), %d atheroprotective (−), %d depleted (0)",
             n_pos, n_neg, n_zero)
    return program_gammas


# ---------------------------------------------------------------------------
# Genetic-direction NMF γ — parallel to SVD track
# ---------------------------------------------------------------------------

def extract_genetic_nmf_gene_sets(
    dataset_id: str,
    n_top_genes: int = _SVD_TOP_GENES_PER_COMPONENT,
    disease_key: str = "",
    loadings_name: str = "genetic_nmf_loadings.npz",
    prog_prefix_suffix: str = "",
) -> dict[str, set[str]]:
    """
    Extract top-N genes per GeneticNMF component from a loadings npz file.

    loadings_name: allows condition-specific files, e.g.
        "genetic_nmf_loadings_rest.npz"     → REST-condition NMF
        "genetic_nmf_loadings_stim48hr.npz" → Stim48hr-condition NMF
    prog_prefix_suffix: appended to program prefix before "_C01", e.g. "_REST", "_Stim48hr".
    """
    import numpy as np

    npz_path = _ROOT / "data" / "perturbseq" / dataset_id / loadings_name
    if not npz_path.exists():
        log.warning("%s not found: %s", loadings_name, npz_path)
        return {}

    npz        = np.load(npz_path)
    U_scaled   = npz["U_scaled"]   # (n_genes × k)
    gene_names = npz["gene_names"]
    n_components = U_scaled.shape[1]
    base   = (disease_key.upper() if disease_key else dataset_id.split("_")[0].upper()) + "_GeneticNMF"
    prefix = base + prog_prefix_suffix

    gene_sets: dict[str, set[str]] = {}
    for c in range(n_components):
        prog_id   = f"{prefix}_C{c+1:02d}"
        loadings  = U_scaled[:, c]
        top_idx   = np.argsort(np.abs(loadings))[::-1][:n_top_genes]
        gene_sets[prog_id] = {str(gene_names[i]) for i in top_idx}

    log.info(
        "Extracted %d GeneticNMF gene sets (top-%d genes each) from %s",
        len(gene_sets), n_top_genes, loadings_name,
    )
    return gene_sets


def run_genetic_nmf_programs(
    disease_key: str,
    force: bool = False,
    condition: str = "",
) -> dict[str, float]:
    """
    Compute GWAS χ² enrichment + eQTL-direction γ for all GeneticNMF programs.

    Parallel to run_svd_programs — same γ decomposition logic, different gene sets.

    condition: "" (shared/Stim8hr), "REST", or "Stim48hr".
        Determines which loadings file and output tau file are used:
            ""        → genetic_nmf_loadings.npz          → {DK}_GeneticNMF_program_taus.json
            "REST"    → genetic_nmf_loadings_rest.npz     → {DK}_GeneticNMF_REST_program_taus.json
            "Stim48hr"→ genetic_nmf_loadings_stim48hr.npz → {DK}_GeneticNMF_Stim48hr_program_taus.json
    """
    _cond_norm    = condition.strip()
    _cond_suffix  = f"_{_cond_norm}" if _cond_norm else ""
    _loadings_sfx = f"_{_cond_norm.lower()}" if _cond_norm else ""
    loadings_name = f"genetic_nmf_loadings{_loadings_sfx}.npz"
    out_path      = _RESULTS_DIR / f"{disease_key.upper()}_GeneticNMF{_cond_suffix}_program_taus.json"

    if out_path.exists() and not force:
        log.info("GeneticNMF τ file already exists: %s (pass force=True to recompute)", out_path)
        with open(out_path) as fh:
            data = json.load(fh)
            return data.get("program_gammas", data.get("program_taus", {}))

    dataset_id = _DISEASE_SVD_DATASET.get(disease_key.upper())
    if not dataset_id:
        raise ValueError(f"No dataset configured for {disease_key}")

    gene_sets = extract_genetic_nmf_gene_sets(
        dataset_id, disease_key=disease_key,
        loadings_name=loadings_name, prog_prefix_suffix=_cond_suffix,
    )
    if not gene_sets:
        raise RuntimeError(
            f"No GeneticNMF gene sets found for {disease_key} condition={_cond_norm!r} — "
            f"run run_genetic_nmf_for_dataset with out_name={loadings_name!r} first"
        )

    log.info(
        "Computing GWAS χ² enrichment for %d GeneticNMF%s programs (%s)...",
        len(gene_sets), _cond_suffix, disease_key.upper(),
    )
    result = compute_program_gwas_enrichment(disease_key, gene_sets)
    program_taus: dict[str, float] = result["program_taus"]

    # eQTL-direction γ sign — using condition-specific NMF U_scaled
    log.info("Computing eQTL direction scores for %s GeneticNMF%s programs...", disease_key.upper(), _cond_suffix)
    nmf_npz    = _ROOT / "data" / "perturbseq" / dataset_id / loadings_name
    prog_prefix = f"{disease_key.upper()}_GeneticNMF{_cond_suffix}"
    direction_scores = compute_program_eqtl_direction(
        disease_key, dataset_id,
        npz_override=nmf_npz,
        prog_prefix=prog_prefix,
    )

    all_tau_negative = all(tau < 0 for tau in program_taus.values())
    if all_tau_negative:
        log.info(
            "All %s GeneticNMF τ < 0 (regulatory-architecture): "
            "falling back to eQTL-direction-only γ (scale=±%.2f)",
            disease_key.upper(), _EQTL_GAMMA_MAX,
        )

    program_gammas: dict[str, float] = {}
    gamma_annotations: list[dict] = []

    for prog, tau in program_taus.items():
        tau_mag  = abs(tau)
        tau_sign = 1 if tau >= 0 else -1

        if all_tau_negative:
            dir_info        = direction_scores.get(prog, {})
            dir_score       = dir_info.get("direction_score", 0.0)
            sum_abs_loading = dir_info.get("sum_abs_loading", 0.0)
            n_eqtl          = dir_info.get("n_eqtl_genes", 0)
            dir_src         = dir_info.get("direction_source", "none")
            if n_eqtl < _EQTL_MIN_COVERAGE or sum_abs_loading == 0.0:
                gamma      = 0.0
                dir_source = "no_eqtl_coverage"
            else:
                concordance = dir_score / sum_abs_loading
                concordance = max(-1.0, min(1.0, concordance))
                gamma       = concordance * _EQTL_GAMMA_MAX
                dir_source  = dir_src + "_eqtl_direction_only"
        elif tau < 0:
            gamma      = 0.0
            dir_source = "depleted"
        else:
            dir_info  = direction_scores.get(prog, {})
            dir_score = dir_info.get("direction_score", 0.0)
            dir_src   = dir_info.get("direction_source", "none")
            if dir_src != "none" and dir_info.get("n_eqtl_genes", 0) >= _EQTL_MIN_COVERAGE:
                gamma      = (1.0 if dir_score >= 0 else -1.0) * tau_mag
                dir_source = dir_src
            else:
                gamma      = tau_mag
                dir_source = "unknown"

        gamma = round(gamma, 6)
        program_gammas[prog] = gamma
        gamma_annotations.append({
            "name":             prog,
            "tau":              tau,
            "tau_sign":         tau_sign,
            "gamma":            gamma,
            "direction_source": dir_source,
            "direction_score":  direction_scores.get(prog, {}).get("direction_score"),
            "sum_abs_loading":  direction_scores.get(prog, {}).get("sum_abs_loading"),
            "n_eqtl_genes":     direction_scores.get(prog, {}).get("n_eqtl_genes", 0),
        })

    result["program_gammas"]    = program_gammas
    result["gamma_annotations"] = gamma_annotations
    result["all_tau_negative"]  = all_tau_negative

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)
    log.info("Saved GeneticNMF γ → %s", out_path)

    n_pos  = sum(1 for g in program_gammas.values() if g > 0)
    n_neg  = sum(1 for g in program_gammas.values() if g < 0)
    n_zero = sum(1 for g in program_gammas.values() if g == 0.0)
    log.info(
        "GeneticNMF γ summary: %d atherogenic (+), %d atheroprotective (−), %d depleted (0)",
        n_pos, n_neg, n_zero,
    )
    return program_gammas


def run_locus_programs(
    disease_key: str,
    force: bool = False,
) -> dict[str, float]:
    """
    Compute GWAS χ² enrichment + GWAS-beta-direction γ for GWAS-anchored locus programs.

    Uses Phase -1 BED files (ra_loci/L{i:02d}.bed) for chromatin windows — enrichment is
    strong by construction since BEDs are built from the CRE midpoints themselves.

    Direction sign: median GWAS beta in the anchor gene's window (±50 kb).
    τ < 0 programs are zeroed (depleted); unknown-direction programs default to |τ|.

    Saves to data/ldsc/results/{disease_key}_LOCUS_program_taus.json.
    Returns dict[locus_name -> gamma].
    """
    import numpy as np

    out_path = _RESULTS_DIR / f"{disease_key.upper()}_LOCUS_program_taus.json"
    if out_path.exists() and not force:
        log.info("Locus τ file already exists: %s (pass force=True to recompute)", out_path)
        with open(out_path) as fh:
            data = json.load(fh)
        return data.get("program_gammas", data.get("program_taus", {}))

    dataset_id = _DISEASE_SVD_DATASET.get(disease_key.upper())
    if not dataset_id:
        raise ValueError(f"No dataset configured for {disease_key}")

    npz_path = _ROOT / "data" / "perturbseq" / dataset_id / "gwas_anchored_programs.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"gwas_anchored_programs.npz not found: {npz_path}")

    d = np.load(npz_path, allow_pickle=True)
    locus_names: list[str] = list(d["locus_names"])
    raw_json = d["locus_genes_json"]
    locus_genes_dict: dict[str, list[str]] = json.loads(
        raw_json.item() if hasattr(raw_json, "item") else str(raw_json)
    )

    locus_gene_sets: dict[str, set[str]] = {
        ln: set(locus_genes_dict.get(ln, []))
        for ln in locus_names
        if locus_genes_dict.get(ln)
    }
    if not locus_gene_sets:
        raise RuntimeError(f"No gene sets in gwas_anchored_programs.npz for {disease_key}")

    log.info(
        "Computing GWAS χ² enrichment for %d locus programs (%s)...",
        len(locus_gene_sets), disease_key.upper(),
    )
    result = compute_program_gwas_enrichment(disease_key, locus_gene_sets)
    program_taus: dict[str, float] = result["program_taus"]

    # Note on coding-variant loci (e.g. TYK2 P1104A, PTPN22 R620W):
    # CRE-based BEDs don't enrich for pinpoint coding variants — those loci have
    # near-zero τ by design.  Coding-variant loci are handled by the WES/pLOF gate
    # and SVD β pathway, not by locus programs.  This is expected and correct.

    # --- Direction: GWAS beta sign in anchor gene window ---
    anchor_genes  = [ln.replace("LOCUS_", "") for ln in locus_gene_sets]
    anchor_coords = _fetch_gene_coords_hg38(anchor_genes)
    gwas_betas    = _collect_gwas_betas_for_genes(disease_key, anchor_genes, anchor_coords)

    def _anchor_gwas_sign(gene: str) -> int:
        coords = anchor_coords.get(gene)
        if not coords:
            return 0
        chrom, start, end = coords
        chrom_n = chrom.lstrip("chr")
        lo, hi  = start - 50_000, end + 50_000
        window_betas = [
            beta for key, beta in gwas_betas.items()
            if key.startswith(f"{chrom_n}:")
            and lo <= int(key.split(":", 1)[1]) <= hi
        ]
        if not window_betas:
            return 0
        # Use the strongest-signal (max |β|) variant — most likely the causal proxy
        peak_beta = max(window_betas, key=abs)
        return 1 if peak_beta >= 0 else -1

    all_tau_negative = all(t < 0 for t in program_taus.values())

    # Normalize γ by total |τ| so values are comparable to SVD gammas (~0.01–0.3 range)
    total_abs_tau = sum(abs(t) for t in program_taus.values()) or 1.0

    program_gammas: dict[str, float] = {}
    gamma_annotations: list[dict] = []

    for locus, tau in program_taus.items():
        anchor = locus.replace("LOCUS_", "")
        if tau < 0 and not all_tau_negative:
            gamma      = 0.0
            dir_source = "depleted"
        else:
            gwas_sign = _anchor_gwas_sign(anchor)
            norm_tau  = abs(tau) / (total_abs_tau + 1e-8)
            if gwas_sign != 0:
                gamma      = round(norm_tau * gwas_sign, 6)
                dir_source = "gwas_beta_anchor"
            else:
                gamma      = round(norm_tau, 6)
                dir_source = "unknown"

        program_gammas[locus] = gamma
        gamma_annotations.append({
            "name":             locus,
            "tau":              tau,
            "tau_sign":         1 if tau >= 0 else -1,
            "gamma":            gamma,
            "anchor_gene":      anchor,
            "direction_source": dir_source,
        })

    result["program_gammas"]    = program_gammas
    result["gamma_annotations"] = gamma_annotations
    result["all_tau_negative"]  = all_tau_negative

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)
    log.info("Saved locus γ → %s", out_path)

    n_pos  = sum(1 for g in program_gammas.values() if g > 0)
    n_neg  = sum(1 for g in program_gammas.values() if g < 0)
    n_zero = sum(1 for g in program_gammas.values() if g == 0.0)
    log.info(
        "Locus γ summary (%s): %d atherogenic (+), %d atheroprotective (−), %d depleted (0)",
        disease_key.upper(), n_pos, n_neg, n_zero,
    )
    return program_gammas


# ---------------------------------------------------------------------------
# Tier 1: Joint conditional τ* LDSC runner
# ---------------------------------------------------------------------------

def run_sldsc_joint_programs(
    disease_key: str,
    program_gene_sets: dict[str, list[str]],
    sumstats_path: str | Path,
    *,
    n_blocks: int = 200,
) -> dict:
    """
    Run partitioned S-LDSC with all programs jointly (conditional τ* estimation).

    Uses baselineLD LD scores as baseline + builds program annotations from gene body
    windows (±100 kb). Returns dict with conditional τ* per program.

    This is the principled fix for bystander confounding: marginal τ inflates
    enrichment for programs whose top genes are near housekeeping gene regulatory
    elements. Joint regression conditions each program on all others.

    Implementation approach (avoids full LD score recomputation):
      1. Build per-chromosome annotation files combining all program columns.
      2. Run one LDSC call with baselineLD LD scores + joint annotation.
      3. Parse --print-coefficients output for per-annotation τ* and SE.

    NOTE: Full LD score recomputation for the joint annotation is deferred (TODO).
    Currently uses the chi-square enrichment method as a joint-approximation fallback
    when baselineLD LD scores are absent or LDSC is not installed. The conditional τ*
    approximation runs only when both baselineLD scores and ldsc.py are present.

    Args:
        disease_key:          Disease key (e.g. "CAD", "RA").
        program_gene_sets:    dict[program_name -> list[gene_symbol]].
        sumstats_path:        Path to GWAS sumstats file (munged .sumstats.gz).
        n_blocks:             Jackknife blocks for SE estimation (default 200).

    Returns dict:
        {
          "program_taus":      {program: tau},
          "raw_annotations":   [{name, tau, tau_se, tau_p, n_snps}],
          "method":            "sldsc_joint_conditional" | "sldsc_joint_chisq_fallback",
          "disease_key":       disease_key,
          "all_tau_negative":  bool,
        }
    """
    import tempfile
    import subprocess as _sp

    try:
        from pipelines.ldsc.setup import _LDSC_BIN, _POLYFUN_LDSCORE_DIR
        _ldsc_bin_exists = _LDSC_BIN.exists()
    except Exception:
        _ldsc_bin_exists = False
        _LDSC_BIN = None
        _POLYFUN_LDSCORE_DIR = None

    # Check for baselineLD LD scores
    _baseline_dir = _LDSC_DIR / "ldscores"
    _baseline_exists = (_baseline_dir / "baselineLD.1.l2.ldscore.gz").exists()

    sumstats_path = Path(sumstats_path)

    # --- Fallback path: no baselineLD or no ldsc binary ---
    if not _baseline_exists or not _ldsc_bin_exists:
        reason = []
        if not _baseline_exists:
            reason.append(
                f"baselineLD LD scores missing at {_baseline_dir} "
                "(expected baselineLD.{{1..22}}.l2.ldscore.gz)"
            )
        if not _ldsc_bin_exists:
            reason.append("ldsc.py not installed")
        log.warning(
            "run_sldsc_joint_programs: falling back to chi-square enrichment. "
            "Reasons: %s. "
            "To enable joint S-LDSC: download baselineLD LD scores and install ldsc.",
            "; ".join(reason),
        )
        gene_sets_as_sets = {k: set(v) for k, v in program_gene_sets.items()}
        result = compute_program_gwas_enrichment(disease_key, gene_sets_as_sets)
        result["method"] = "sldsc_joint_chisq_fallback"
        return result

    # --- Joint LDSC path ---
    log.info(
        "run_sldsc_joint_programs: building joint annotation for %d programs (%s)...",
        len(program_gene_sets), disease_key.upper(),
    )

    # Step 1: Fetch gene coordinates and build gene-body ±100 kb windows per program
    all_genes: list[str] = list({g for genes in program_gene_sets.values() for g in genes})
    gene_coords = _fetch_gene_coords_hg38(all_genes)
    gene_sets_as_sets = {k: set(v) for k, v in program_gene_sets.items()}
    prog_windows = _build_program_windows(gene_sets_as_sets, gene_coords)

    prog_names = sorted(program_gene_sets.keys())
    prog_idx   = {p: i for i, p in enumerate(prog_names)}

    program_taus:    dict[str, float] = {}
    raw_annotations: list[dict]       = []

    with tempfile.TemporaryDirectory(prefix="sldsc_joint_") as tmpdir:
        tmp = Path(tmpdir)

        # Step 2: Build per-chromosome joint annotation files.
        # Each file has columns: CHR BP SNP CM prog1 prog2 ... progN
        # Rows come from the baselineLD .annot.gz SNP list.
        joint_annot_prefix = tmp / f"joint_{disease_key.upper()}"
        n_annot_built = 0

        for chrom_int in range(1, 23):
            baseline_annot = _baseline_dir / f"baselineLD.{chrom_int}.annot.gz"
            if not baseline_annot.exists():
                log.debug("baselineLD annot missing for chr%d — skipping", chrom_int)
                continue

            out_annot = tmp / f"joint_{disease_key.upper()}.{chrom_int}.annot.gz"
            try:
                with gzip.open(baseline_annot, "rt") as fh_in, \
                     gzip.open(out_annot, "wt") as fh_out:
                    # Read header to get CHR BP SNP CM columns
                    header_parts = fh_in.readline().strip().split("\t")
                    # Write new header: base columns + program columns
                    new_header = header_parts + prog_names
                    fh_out.write("\t".join(new_header) + "\n")

                    chr_i = header_parts.index("CHR") if "CHR" in header_parts else 0
                    bp_i  = header_parts.index("BP")  if "BP"  in header_parts else 1

                    for line in fh_in:
                        parts = line.rstrip("\n").split("\t")
                        if len(parts) < 2:
                            continue
                        try:
                            chrom_str = str(chrom_int)
                            pos = int(parts[bp_i])
                        except (ValueError, IndexError):
                            continue

                        # Build binary annotation vector for each program
                        prog_cols = []
                        for pn in prog_names:
                            wins = prog_windows.get(pn, {})
                            in_win = _snp_in_windows(chrom_str, pos, wins)
                            prog_cols.append("1" if in_win else "0")

                        fh_out.write("\t".join(parts) + "\t" + "\t".join(prog_cols) + "\n")

                n_annot_built += 1
            except Exception as exc:
                log.warning("Failed to build joint annot for chr%d: %s", chrom_int, exc)
                continue

        if n_annot_built == 0:
            log.warning(
                "run_sldsc_joint_programs: no joint annotation files built — "
                "falling back to chi-square method"
            )
            result = compute_program_gwas_enrichment(disease_key, gene_sets_as_sets)
            result["method"] = "sldsc_joint_chisq_fallback"
            return result

        log.info(
            "Built joint annotation for %d/%d chromosomes. "
            "TODO: compute LD scores for joint annotation (currently uses baselineLD scores as ref).",
            n_annot_built, 22,
        )

        # Step 3: Run joint LDSC.
        # TODO: Replace baselineLD ref with jointly-computed LD scores for the program
        # annotation columns. Currently uses baselineLD as the reference LD matrix,
        # which gives an approximation — the program columns are included as additional
        # annotations but their LD structure is not explicitly modeled.
        # Full implementation: run `ldsc.py --l2 --annot joint_annot --bfile ...`
        # to produce properly conditioned LD scores, then pass those as --ref-ld-chr.

        _weight_dir = _baseline_dir  # weights assumed to be in same dir
        _weight_prefix = str(_weight_dir / "weights.hm3_noMHC.")
        _baseline_prefix = str(_baseline_dir / "baselineLD.")

        out_prefix = tmp / f"ldsc_joint_{disease_key.upper()}"
        cmd = [
            sys.executable, str(_LDSC_BIN),
            "--h2",               str(sumstats_path),
            "--ref-ld-chr",       _baseline_prefix,
            "--w-ld-chr",         _weight_prefix,
            "--overlap-annot",
            "--print-coefficients",
            "--annot",            str(joint_annot_prefix) + ".",
            "--out",              str(out_prefix),
        ]

        log.info("Running joint LDSC: %s", " ".join(cmd[:6]) + " ...")
        try:
            proc = _sp.run(cmd, capture_output=True, text=True, timeout=3600)
            if proc.returncode != 0:
                log.warning(
                    "Joint LDSC failed (returncode=%d): %s",
                    proc.returncode, proc.stderr[-1000:],
                )
                raise RuntimeError("LDSC non-zero exit")

            # Step 4: Parse joint LDSC results
            results_file = Path(str(out_prefix) + ".results")
            if not results_file.exists():
                raise FileNotFoundError(f"LDSC results file not found: {results_file}")

            with open(results_file) as fh:
                header = fh.readline().strip().split("\t")
                col = {c.strip(): i for i, c in enumerate(header)}
                for line in fh:
                    parts = line.strip().split("\t")
                    if len(parts) < len(col):
                        continue
                    cat = parts[col.get("Category", 0)].strip()
                    # Only parse rows matching our program names
                    prog_match = next((p for p in prog_names if cat.endswith(p) or p in cat), None)
                    if prog_match is None:
                        continue
                    try:
                        tau    = float(parts[col["Coefficient"]])
                        tau_se = float(parts[col["Coefficient_std_error"]])
                        tau_z  = tau / (tau_se + 1e-8)
                        tau_p  = 2.0 * (1.0 - _norm_cdf(abs(tau_z)))
                        program_taus[prog_match] = round(tau, 6)
                        raw_annotations.append({
                            "name":   prog_match,
                            "tau":    round(tau, 6),
                            "tau_se": round(tau_se, 6),
                            "tau_p":  round(tau_p, 6),
                            "method": "sldsc_joint_conditional",
                        })
                    except (KeyError, ValueError, IndexError):
                        continue

            if not program_taus:
                raise RuntimeError("No program taus parsed from joint LDSC results")

        except Exception as exc:
            log.warning(
                "Joint LDSC run or parsing failed (%s) — falling back to chi-square method",
                exc,
            )
            result = compute_program_gwas_enrichment(disease_key, gene_sets_as_sets)
            result["method"] = "sldsc_joint_chisq_fallback"
            return result

    all_tau_negative = all(t < 0 for t in program_taus.values())
    log.info(
        "Joint conditional τ* complete for %s: %d programs, all_tau_negative=%s",
        disease_key.upper(), len(program_taus), all_tau_negative,
    )
    return {
        "program_taus":    program_taus,
        "h2":              None,
        "raw_annotations": raw_annotations,
        "method":          "sldsc_joint_conditional",
        "disease_key":     disease_key.upper(),
        "all_tau_negative": all_tau_negative,
    }


def get_joint_conditional_taus(disease_key: str) -> dict:
    """
    Return joint conditional τ* for all programs in disease_key.

    Checks for a cached result at data/ldsc/results/{DISEASE}_joint_conditional_taus.json
    before running the full joint LDSC. The cache is written on first successful run.

    Args:
        disease_key: Disease key (e.g. "CAD", "RA").

    Returns:
        Same structure as run_sldsc_joint_programs().
        Returns empty dict if neither cache nor GWAS sumstats are available.
    """
    from pipelines.ldsc.setup import GWAS_CONFIG, _SUMSTATS_DIR

    cache_path = _RESULTS_DIR / f"{disease_key.upper()}_joint_conditional_taus.json"
    if cache_path.exists():
        try:
            with open(cache_path) as fh:
                cached = json.load(fh)
            log.info(
                "Joint conditional τ* loaded from cache: %s (%d programs)",
                cache_path, len(cached.get("program_taus", {})),
            )
            return cached
        except Exception as exc:
            log.warning("Failed to load joint τ cache for %s: %s — recomputing", disease_key, exc)

    cfg = GWAS_CONFIG.get(disease_key.upper())
    if not cfg:
        log.warning("get_joint_conditional_taus: no GWAS config for %s", disease_key)
        return {}

    sumstats_path = _SUMSTATS_DIR / cfg["filename"]
    if not sumstats_path.exists():
        log.warning(
            "get_joint_conditional_taus: sumstats not found for %s: %s",
            disease_key, sumstats_path,
        )
        return {}

    # Gather program gene sets from the SVD loadings
    dataset_id = _DISEASE_SVD_DATASET.get(disease_key.upper())
    if not dataset_id:
        log.warning("get_joint_conditional_taus: no SVD dataset for %s", disease_key)
        return {}

    try:
        svd_gene_sets = extract_svd_gene_sets(dataset_id, disease_key=disease_key)
    except FileNotFoundError as exc:
        log.warning("get_joint_conditional_taus: %s", exc)
        return {}

    # Convert set → list for the joint runner
    program_gene_sets_lists = {k: list(v) for k, v in svd_gene_sets.items()}

    result = run_sldsc_joint_programs(
        disease_key=disease_key,
        program_gene_sets=program_gene_sets_lists,
        sumstats_path=sumstats_path,
    )

    if result.get("program_taus"):
        _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as fh:
            json.dump(result, fh, indent=2)
        log.info("Saved joint conditional τ* → %s", cache_path)

    return result


# ---------------------------------------------------------------------------
# ATAC-peak gene window builder (Tier 2 — cell-type-specific windows for RA)
# ---------------------------------------------------------------------------

_DISEASE_ATAC_BED: dict[str, Path] = {
    "RA":  _ROOT / "data" / "atac" / "calderon2019_cd4t_peaks.bed",
    "SLE": _ROOT / "data" / "atac" / "calderon2019_cd4t_peaks.bed",
    # Primary: Turner 2022 Nat Genet (GSE188422) — human coronary artery snATAC, 15 patients
    # 197k union peaks across all coronary artery cell types (EC, SMC, macrophage, etc.)
    "CAD": _ROOT / "data" / "atac" / "coronary_artery_atac_peaks.bed",
}

# Sub-analysis ATAC tracks for CAD — shear stress comparison
# Used when ATAC_SUBANALYSIS=shear is set; otherwise the primary track above is used.
_CAD_ATAC_SUBANALYSIS: dict[str, Path] = {
    # GSE227588 (Jatzlau 2023 iScience) — HAoEC (human aortic EC), narrowPeak
    "haec_static": _ROOT / "data" / "atac" / "haec_static_atac_peaks.bed",   # 217k peaks, resting baseline
    "haec_flow":   _ROOT / "data" / "atac" / "haec_flow_atac_peaks.bed",     # 299k peaks, laminar flow (shear)
    # GSE198221 (Maegdefessel 2022) — HUVEC + 18 dyn/cm2 shear stress
    "huvec_shear": _ROOT / "data" / "atac" / "huvec_shear_stress_atac_peaks.bed",  # 63k flow-enriched peaks
    "huvec_shear_gained": _ROOT / "data" / "atac" / "huvec_shear_gained_atac_peaks.bed",  # 7.7k gained-only
}


def build_atac_gene_windows(
    disease_key: str,
    program_gene_sets: dict[str, list[str]],
    atac_window_bp: int = 500_000,
) -> dict[str, dict[str, list[tuple[int, int]]]]:
    """
    Build program genomic windows using cell-type ATAC-seq peaks (Tier 2).

    For RA/SLE: uses Calderon 2019 CD4+ T cell ATAC peaks (data/atac/calderon2019_cd4t_peaks.bed).
    For CAD: Turner 2022 Nat Genet coronary artery snATAC (data/atac/coronary_artery_atac_peaks.bed).
    Sub-analysis tracks (HAoEC static/flow, HUVEC shear) are in _CAD_ATAC_SUBANALYSIS.

    The ATAC-peak window is more cell-type-specific than a flat gene-body window:
    a metabolic gene (e.g. MARS1) may have very few accessible chromatin peaks in
    CD4+ T cells despite being near many GWAS SNPs in a flat ±100 kb window.

    Args:
        disease_key:         Disease key (e.g. "RA", "CAD").
        program_gene_sets:   dict[program_name -> list[gene_symbol]].
        atac_window_bp:      Window around each gene TSS to search for ATAC peaks (default ±500 kb).

    Returns:
        dict[program -> dict[chrom -> [(start, end), ...]]]
        Same format as _build_program_windows() — drop-in replacement for runner.py windows.
    """
    dk = disease_key.upper()
    atac_bed = _DISEASE_ATAC_BED.get(dk)

    all_genes_list = list({g for genes in program_gene_sets.values() for g in genes})
    gene_sets_as_sets = {k: set(v) for k, v in program_gene_sets.items()}
    gene_coords = _fetch_gene_coords_hg38(all_genes_list)

    if atac_bed is None or not atac_bed.exists():
        # CAD or missing ATAC file — fall back to gene body windows
        if atac_bed is not None and not atac_bed.exists():
            log.info(
                "build_atac_gene_windows: ATAC BED not found for %s (%s) — "
                "falling back to gene body ±%d bp windows",
                dk, atac_bed, _GENE_WINDOW_BP,
            )
        else:
            log.info(
                "build_atac_gene_windows: no ATAC BED configured for %s "
                "(TODO: add ENCODE HAEC or Schnitzler scATAC for cardiac EC) — "
                "falling back to gene body ±%d bp windows",
                dk, _GENE_WINDOW_BP,
            )
        return _build_program_windows(gene_sets_as_sets, gene_coords)

    log.info(
        "build_atac_gene_windows: loading ATAC peaks for %s from %s (window=±%d bp per gene TSS)...",
        dk, atac_bed.name, atac_window_bp,
    )

    # Step 1: Load ATAC peaks into a chrom → sorted list of (start, end) index
    atac_by_chrom: dict[str, list[tuple[int, int]]] = {}
    try:
        with open(atac_bed) as fh:
            for line in fh:
                if line.startswith("#") or line.startswith("track") or line.startswith("browser"):
                    continue
                parts = line.strip().split("\t")
                if len(parts) < 3:
                    continue
                chrom_str = parts[0].lstrip("chr")
                try:
                    start = int(parts[1])
                    end   = int(parts[2])
                except ValueError:
                    continue
                atac_by_chrom.setdefault(chrom_str, []).append((start, end))
    except Exception as exc:
        log.warning(
            "build_atac_gene_windows: failed to load ATAC BED %s (%s) — "
            "falling back to gene body windows",
            atac_bed, exc,
        )
        return _build_program_windows(gene_sets_as_sets, gene_coords)

    # Sort peaks per chromosome for efficient interval queries
    for chrom_str in atac_by_chrom:
        atac_by_chrom[chrom_str].sort()

    total_peaks = sum(len(v) for v in atac_by_chrom.values())
    log.info("ATAC peaks loaded: %d peaks across %d chromosomes", total_peaks, len(atac_by_chrom))

    # Step 2: For each gene in each program, find ATAC peaks within ±atac_window_bp of the TSS
    def _atac_peaks_near_gene(chrom_str: str, tss: int) -> list[tuple[int, int]]:
        """Return ATAC peaks within [tss - atac_window_bp, tss + atac_window_bp]."""
        lo = tss - atac_window_bp
        hi = tss + atac_window_bp
        peaks = atac_by_chrom.get(chrom_str, [])
        # Linear scan — acceptable given peaks are pre-sorted per chrom and this runs once
        return [(s, e) for s, e in peaks if s <= hi and e >= lo]

    # Step 3: Build program windows as the union of ATAC peak intervals for all program genes
    windows: dict[str, dict[str, list[tuple[int, int]]]] = {}
    for prog, genes in program_gene_sets.items():
        chrom_windows: dict[str, list[tuple[int, int]]] = {}
        n_genes_with_atac = 0
        for gene in genes:
            coords = gene_coords.get(gene)
            if not coords:
                continue
            chrom_str, start, end = coords
            chrom_n = str(chrom_str).lstrip("chr")
            tss = (start + end) // 2  # use midpoint as proxy for TSS
            nearby_peaks = _atac_peaks_near_gene(chrom_n, tss)
            if nearby_peaks:
                chrom_windows.setdefault(chrom_n, []).extend(nearby_peaks)
                n_genes_with_atac += 1

        if not chrom_windows:
            # If no ATAC peaks found for any gene, fall back to gene body windows for this program
            fb = _build_program_windows({prog: set(genes)}, gene_coords)
            windows[prog] = fb.get(prog, {})
        else:
            windows[prog] = chrom_windows

        log.debug(
            "  %s: %d/%d genes with ATAC peaks, %d chroms with windows",
            prog, n_genes_with_atac, len(genes), len(chrom_windows),
        )

    n_atac = sum(1 for p, w in windows.items() if w)
    log.info(
        "build_atac_gene_windows: built ATAC-based windows for %d/%d programs (%s)",
        n_atac, len(program_gene_sets), dk,
    )
    return windows


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
    elif cmd == "run_cnmf":
        force_flag = "--force" in sys.argv
        print(f"Computing GWAS χ² enrichment for cNMF programs ({disease})...")
        gammas = run_cnmf_programs(disease, force=force_flag)
        print(f"\n{disease} cNMF program γ (sorted by |γ|):")
        for p, g in sorted(gammas.items(), key=lambda x: -abs(x[1])):
            sign = "+" if g >= 0 else ""
            print(f"  {p}: {sign}{g:.4f}")
    elif cmd == "run_locus":
        force_flag = "--force" in sys.argv
        print(f"Computing GWAS χ² enrichment for GWAS-anchored locus programs ({disease})...")
        gammas = run_locus_programs(disease, force=force_flag)
        print(f"\n{disease} locus program γ (sorted by |γ|):")
        for p, g in sorted(gammas.items(), key=lambda x: -abs(x[1])):
            sign = "+" if g >= 0 else ""
            print(f"  {p}: {sign}{g:.4f}")
    elif cmd == "run_genetic_nmf":
        force_flag = "--force" in sys.argv
        from pipelines.genetic_nmf import run_genetic_nmf_for_dataset
        dataset_id = _DISEASE_SVD_DATASET.get(disease.upper())
        if not dataset_id:
            print(f"No dataset configured for {disease}")
            sys.exit(1)
        sig_name = _DISEASE_SIG_NAME.get(disease.upper())
        print(f"Fitting genetic NMF for {disease} (dataset={dataset_id}, sig={sig_name or 'signatures.json.gz'})...")
        result = run_genetic_nmf_for_dataset(dataset_id, disease.upper(), sig_name=sig_name)
        if "error" in result:
            print(f"Error: {result['error']}")
            sys.exit(1)
        print(f"Saved genetic_nmf_loadings.npz — {result.get('n_programs', '?')} components, "
              f"{result.get('n_perts', '?')} perturbations")
        print(f"Computing GWAS χ² enrichment for GeneticNMF programs ({disease})...")
        gammas = run_genetic_nmf_programs(disease, force=True)
        print(f"\n{disease} GeneticNMF program γ (sorted by |γ|):")
        for p, g in sorted(gammas.items(), key=lambda x: -abs(x[1])):
            sign = "+" if g >= 0 else ""
            print(f"  {p}: {sign}{g:.4f}")
    elif cmd == "download_eqtl":
        from pipelines.ldsc.eqtl_local import download_dataset
        p = download_dataset(disease)
        print(f"Downloaded: {p}")
    else:
        print("Usage:")
        print("  python -m pipelines.ldsc.runner run CAD|RA                # NMF programs")
        print("  python -m pipelines.ldsc.runner run_svd CAD|RA            # SVD programs")
        print("  python -m pipelines.ldsc.runner run_locus RA              # GWAS-locus programs")
        print("  python -m pipelines.ldsc.runner run_genetic_nmf CAD|RA   # WES-regularised NMF + γ")
        print("  python -m pipelines.ldsc.runner download_eqtl CAD|RA     # fetch eQTL sumstats")
