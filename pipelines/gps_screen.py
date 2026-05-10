"""
gps_screen.py — GPS (Gene Expression Profile Predictor from chemical Structures) integration.

Wraps the GPS4Drugs Docker container to screen compound libraries for target emulation:
given a gene's Perturb-seq knockdown signature, find compounds that produce a similar
transcriptomic shift — i.e., compounds that emulate therapeutic inhibition of that target.

GPS paper: Xing et al., Cell 2026. https://doi.org/10.1016/j.cell.2026.02.016
Repo:      https://github.com/Bin-Chen-Lab/GPS
Docker:    docker pull binchengroup/gpsimage:latest

Target emulation logic
----------------------
If gene X causally increases AMD risk (OTA gamma > 0), knocking it out is therapeutic.
The Perturb-seq KO signature captures what the cell looks like when X is absent.
We want compounds that produce a SIMILAR expression shift.

GPS_runDrugScreenRges uses the Reversed Gene Expression Score (RGES): a positive
RGES means a compound REVERSES the input signature. To find compounds that MIMIC
the KO (rather than reverse it), we negate the KO signature before input.
  input = -1 × KO_signature  →  GPS finds reversers of that  →  reversers of (-KO) = mimics of KO

GPS library options
-------------------
- "HTS"  : Enamine HTS (pre-loaded in Docker, ~2M drug-like compounds). Default.
- "ZINC" : ZINC (~250K compounds). Requires downloading ZINC_strong.npz separately.

Usage
-----
    from pipelines.gps_screen import screen_target_for_emulators

    hits = screen_target_for_emulators(
        gene="CFI",
        disease_query={"disease_name": "age-related macular degeneration", "disease_key": "amd"},
        top_n=20,
    )
    # hits: list[{"compound_id", "rges", "rank", "source_library", "note"}]
"""
from __future__ import annotations

import csv
import logging
import os
import subprocess
import tempfile
from pathlib import Path

# Prevent Numba TBB fork-safety warning when GPS subprocess is spawned from a
# non-main thread (e.g. ThreadPoolExecutor in GPS parallel screening).
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")

log = logging.getLogger(__name__)

_DOCKER_IMAGE  = "binchengroup/gpsimage:latest"
_GPS_TIMEOUT_WITH_BGRD    = 7200   # seconds; 500-gene sig ~900s
_GPS_TIMEOUT              = _GPS_TIMEOUT_WITH_BGRD  # legacy alias
_GPS_GENES     = None  # GPS's selected_genes_2198.csv — loaded lazily
_GPS_BGRD_DIR  = Path(__file__).parent.parent / "data" / "gps_bgrd"  # persistent BGRD cache
_GPS_LOGS_DIR  = Path(__file__).parent.parent / "data" / "gps_logs"  # GPS internal logfiles

# All GPS screens use exactly this many up/down genes → one shared BGRD__size500.pkl forever.
_GPS_SIG_N_UP   = 250
_GPS_SIG_N_DOWN = 250
_GPS_BGRD_KEY   = "size500"


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def screen_target_for_emulators(
    gene: str,
    disease_query: dict,
    library: str = "HTS",
    top_n: int = 20,
    ota_gamma: float | None = None,
    z_threshold: float | None = None,
    max_hits: int = 100,
) -> list[dict]:
    """
    Find compounds that emulate knockdown of `gene` using GPS drug screening.

    Args:
        gene:          Gene symbol (e.g. "CFI")
        disease_query: Pipeline disease_query dict (for disease_key + context)
        library:       "HTS" (Enamine, built-in) or "ZINC" (requires download)
        top_n:         Hard fallback cutoff when Z_RGES threshold not applicable
        ota_gamma:     OTA gamma for gene (positive = risk-increasing → inhibit;
                       negative = protective → activate; None = assume inhibit)
        z_threshold:   |Z_RGES| threshold for dynamic cutoff (None = use top_n)
        max_hits:      Hard cap on returned hits regardless of threshold

    Returns:
        List of dicts with keys: compound_id, rges, z_rges, rank, source_library, note
        Empty list if GPS unavailable or no Perturb-seq data.
    """
    if not _docker_available():
        log.warning("GPS screen skipped: Docker not available")
        return []

    sig = _get_perturb_signature(gene, disease_query)
    if not sig:
        log.info("GPS screen skipped for %s: no Perturb-seq signature", gene)
        return []

    # Minimum gene gate: BGRD__size500.pkl covers all bin sizes 2→499, so any
    # signature with ≥2 genes per direction is valid. Gate at 20 to ensure
    # the KS statistic is not degenerate (per-direction check happens in
    # _check_gps_library_overlap).
    if len(sig) < 20:
        log.info(
            "GPS emulation screen skipped for %s: only %d GPS-filtered genes "
            "(need ≥20)", gene, len(sig),
        )
        return []

    # Negate KO signature: compounds that REVERSE the negated KO mimic the KO
    # (see module docstring). For protective genes (gamma < 0), don't negate.
    direction = -1.0 if (ota_gamma is None or ota_gamma >= 0) else 1.0
    therapeutic_sig = {g: direction * lfc for g, lfc in sig.items()}

    return _run_gps_screen(gene, therapeutic_sig, library, top_n, z_threshold, max_hits)


def screen_disease_for_reversers(
    disease_sig: dict[str, float],
    label: str,
    library: str = "HTS",
    top_n: int = 20,
    z_threshold: float | None = None,
    max_hits: int = 100,
) -> list[dict]:
    """
    Find compounds that reverse a disease expression signature.

    Args:
        disease_sig:  {gene_symbol: log2FC} (positive = upregulated in disease)
        label:        Short label for output files (e.g. "amd_rpe")
        library:      "HTS" or "ZINC"
        top_n:        Hard fallback cutoff when Z_RGES threshold not applicable
        z_threshold:  |Z_RGES| threshold for dynamic cutoff (None = use top_n).
                      With a properly calibrated BGRD (≥700 perms), Z_RGES follows
                      ~N(0,1) under null → threshold 2.0 ≈ FDR 5%.
        max_hits:     Hard cap on returned hits regardless of threshold

    Returns:
        List of dicts with compound_id, rges, z_rges, rank, source_library.
    """
    if not _docker_available():
        log.warning("GPS screen skipped: Docker not available")
        return []
    if not disease_sig:
        return []
    return _run_gps_screen(label, disease_sig, library, top_n, z_threshold, max_hits)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_perturb_signature(gene: str, disease_query: dict) -> dict[str, float]:
    """Fetch Perturb-seq KO signature, restricted to GPS's selected_genes_2198 set."""
    try:
        from mcp_servers.perturbseq_server import get_perturbseq_signature
        disease_key = disease_query.get("disease_key") or disease_query.get("disease_name", "")
        # Request a large signature — GPS filters to its own 2198 gene set internally,
        # so we need to supply enough genes to get a non-empty intersection.
        result = get_perturbseq_signature(gene, disease_context=disease_key or None, top_k=5000)
        raw_sig: dict[str, float] = result.get("signature", {})
        if not raw_sig:
            return {}
        # Restrict to GPS landmark genes (selected_genes_2198.csv).
        # Using L1000 (978 genes) was wrong — GPS uses a different 2198-gene set
        # with low overlap, causing zero-overlap → min() crash in permute_rges.py.
        gps_genes = _get_gps_genes()
        if gps_genes:
            raw_sig = {g: v for g, v in raw_sig.items() if g in gps_genes}
        return raw_sig
    except Exception as exc:
        log.debug("Perturb-seq fetch failed for %s: %s", gene, exc)
        return {}


def _get_gps_genes() -> set[str] | None:
    """Return GPS's selected_genes_2198 set, or None if file not found."""
    global _GPS_GENES
    if _GPS_GENES is not None:
        return _GPS_GENES
    candidates = [
        Path(__file__).parent.parent / "data" / "annotations" / "gps_selected_genes_2198.txt",
        Path(__file__).parent.parent / "data" / "gps_selected_genes_2198.txt",
    ]
    for p in candidates:
        if p.exists():
            genes = {line.strip() for line in p.read_text().splitlines() if line.strip()}
            if genes:
                _GPS_GENES = genes
                return _GPS_GENES
    log.debug("GPS gene list not found; no signature filtering applied")
    return None  # no filter — GPS will intersect internally


def _truncate_sig(sig: dict[str, float], n_up: int = _GPS_SIG_N_UP, n_down: int = _GPS_SIG_N_DOWN) -> dict[str, float]:
    """Truncate signature to top n_up positive + top n_down negative genes by magnitude."""
    up   = sorted(((g, v) for g, v in sig.items() if v > 0), key=lambda kv: -kv[1])[:n_up]
    down = sorted(((g, v) for g, v in sig.items() if v < 0), key=lambda kv:  kv[1])[:n_down]
    return dict(up + down)


def _write_dzsig_csv(sig: dict[str, float], path: Path) -> None:
    """Write GPS disease signature CSV: GeneSymbol, Value."""
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["GeneSymbol", "Value"])
        # GPS expects genes sorted by |value| descending
        for gene, val in sorted(sig.items(), key=lambda kv: -abs(kv[1])):
            writer.writerow([gene, f"{val:.6f}"])


def _check_gps_library_overlap(
    label: str,
    sig: dict[str, float],
    min_per_direction: int = 2,
) -> bool:
    """Return True if sig has enough genes in both directions within the GPS 2198-gene library.

    GPS's Run_reversal_score.py computes rdc_transcrptm = intersection(HTS_2198_genes, sig_genes),
    then builds a BGRD only for directions present in the filtered signature.  If all filtered
    genes have the same sign, the BGRD has bins for only one direction (Up_* OR Down_*).
    Scoring then calls min(bins_up, ...) or min(bins_down, ...) on an empty list → crash.

    Skipping pre-emptively is cheaper and cleaner than catching Docker exit 1.
    """
    gps_genes = _get_gps_genes()
    if gps_genes is None:
        return True  # no filter available — let GPS try
    filtered = {g: v for g, v in sig.items() if g in gps_genes}
    if not filtered:
        log.warning(
            "GPS pre-screen %s: 0 / %d sig genes found in GPS 2198-gene library — skipping",
            label, len(sig),
        )
        return False
    n_up   = sum(1 for v in filtered.values() if v > 0)
    n_down = sum(1 for v in filtered.values() if v < 0)
    if n_up < min_per_direction or n_down < min_per_direction:
        log.warning(
            "GPS pre-screen %s: filtered sig has only %d up / %d down genes in GPS library "
            "(need ≥%d each) — skipping to avoid Run_reversal_score min() crash",
            label, n_up, n_down, min_per_direction,
        )
        return False
    return True


def _run_gps_screen(
    label: str,
    sig: dict[str, float],
    library: str,
    top_n: int,
    z_threshold: float | None = None,
    max_hits: int = 100,
) -> list[dict]:
    """Write signature, invoke Docker GPS screen, parse and return results.

    Signatures are truncated to top 250 up + 250 down GPS-represented genes so all
    screens share BGRD__size500.pkl — no per-run permutation ever needed.
    """
    # Truncate to fixed size (top n_up positive + top n_down negative GPS-represented genes).
    # All screens share BGRD__size500.pkl — no per-run permutation ever needed.
    sig = _truncate_sig(sig)

    if not _check_gps_library_overlap(label, sig):
        return []

    if len(sig) < 4:
        log.warning("GPS screen skipped for %s: only %d genes after truncation", label, len(sig))
        return []

    safe_label = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)[:40]
    bgrd_dir = _GPS_BGRD_DIR
    bgrd_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = _GPS_LOGS_DIR
    logs_dir.mkdir(parents=True, exist_ok=True)

    bgrd_key = _GPS_BGRD_KEY  # always "size500"
    bgrd_pkl = bgrd_dir / f"BGRD__{bgrd_key}.pkl"

    if not bgrd_pkl.exists():
        log.error(
            "GPS screen: BGRD__size500.pkl not found in %s — run GPS once with "
            "--RGES_bgrd_ID NONE to generate it, then cache as BGRD__size500.pkl",
            bgrd_dir,
        )
        return []

    log.info("GPS screen: %s — using cached BGRD %s (%d sig genes)", label, bgrd_key, len(sig))
    run_timeout = _GPS_TIMEOUT_WITH_BGRD
    bgrd_id_arg = ["--RGES_bgrd_ID", bgrd_key]

    with tempfile.TemporaryDirectory(prefix="gps_screen_") as tmpdir:
        tmp = Path(tmpdir)
        input_dir  = tmp / "input"
        output_dir = tmp / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        sig_filename = f"DZSIG__{safe_label}.csv"
        _write_dzsig_csv(sig, input_dir / sig_filename)

        cmd = [
            "docker", "run", "--rm",
            "--platform", "linux/amd64",  # Rosetta2 emulation on Apple Silicon
            "-v", f"{input_dir}:/app/input",
            "-v", f"{output_dir}:/app/data/reversal_score",
            "-v", f"{bgrd_dir}:/app/data/dzsig",   # persistent BGRD cache (read+write)
            "-v", f"{logs_dir}:/app/logs",          # GPS internal logfiles for debugging
            _DOCKER_IMAGE,
            "python", "code/GPS_runDrugScreenRges.py",
            "--dzSigFile", f"input/{sig_filename}",
            "--cmpdLibID", library,
            *bgrd_id_arg,
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=run_timeout,
                start_new_session=True,   # prevent fork inheriting parent thread state
                env={**os.environ, "NUMBA_THREADING_LAYER": "workqueue"},
            )
            if result.returncode != 0:
                log.warning(
                    "GPS Docker exited %d for %s:\n%s",
                    result.returncode, label, result.stderr[-2000:]
                )
                return []
            log.info("GPS Docker done: %s", label)

        except subprocess.TimeoutExpired:
            log.warning("GPS screen timed out for %s after %ds", label, run_timeout)
            return []
        except Exception as exc:
            log.warning("GPS Docker call failed for %s: %s", label, exc)
            return []

        return _parse_gps_output(output_dir, library, top_n, z_threshold, max_hits)


def _parse_gps_output(
    output_dir: Path,
    library: str,
    top_n: int,
    z_threshold: float | None = None,
    max_hits: int = 100,
) -> list[dict]:
    """
    Parse GPS reversal score CSV output.

    GPS writes one or more CSVs to the reversal_score directory.
    Expected columns (flexible): compound_id/ID/name, RGES/rges/score, p_value/pvalue.

    Cutoff logic (in priority order):
      1. If GPS output has Z_RGES column AND z_threshold is set:
           return ALL compounds with Z_RGES < -z_threshold (reversers).
           max_hits is NOT applied here — the threshold governs.
           Falls back to top_n if fewer than 3 compounds pass (under-powered BGRD).
      2. Otherwise: return top_n by |RGES| (capped at max_hits).
    """
    csvs = list(output_dir.glob("*.csv"))
    if not csvs:
        log.warning("GPS produced no output CSVs in %s", output_dir)
        return []

    # Use the largest CSV if multiple (main results file)
    target_csv = max(csvs, key=lambda p: p.stat().st_size)

    results: list[dict] = []
    has_z_rges = False
    try:
        with open(target_csv, newline="") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None:
                return []

            # Column name normalisation — GPS output varies by version
            headers = [h.lower().strip() for h in reader.fieldnames]
            id_col    = _pick_col(headers, ["compound_id", "id", "name", "drugid", "cmpd_id"])
            # GPS Run_reversal_score.py writes 'Z_RGES' (z-scored); accept all variants
            rges_col  = _pick_col(headers, ["z_rges", "rges", "score", "reversal_score", "enrichment_score"])
            pval_col  = _pick_col(headers, ["p_value", "pvalue", "p", "fdr"])

            if id_col is None or rges_col is None:
                log.warning(
                    "GPS output missing expected columns. Found: %s", reader.fieldnames
                )
                return []

            has_z_rges = rges_col == "z_rges"
            orig_headers = list(reader.fieldnames)
            id_orig   = orig_headers[headers.index(id_col)]
            rges_orig = orig_headers[headers.index(rges_col)]
            pval_orig = orig_headers[headers.index(pval_col)] if pval_col else None

            for row in reader:
                try:
                    cid  = row[id_orig].strip()
                    rges = float(row[rges_orig])
                except (KeyError, ValueError):
                    continue
                pval = None
                if pval_orig:
                    try:
                        pval = float(row[pval_orig])
                    except (ValueError, KeyError):
                        pass
                hit: dict = {
                    "compound_id":    cid,
                    "rges":           rges,
                    "p_value":        pval,
                    "source_library": library,
                    "note":           "GPS target emulation screen",
                }
                if has_z_rges:
                    hit["z_rges"] = rges
                results.append(hit)
    except Exception as exc:
        log.warning("Failed parsing GPS output %s: %s", target_csv, exc)
        return []

    # Sort by score descending (most negative = best reverser)
    results.sort(key=lambda r: r["rges"])
    for i, r in enumerate(results, 1):
        r["rank"] = i

    # Z_RGES threshold — governs when GPS output is z-scored against permuted null.
    # Primary path: data-driven step-off detection (largest gap ≥ GPS_Z_STEPOFF_MIN_RATIO
    # × median gap in signal region). Falls back to floor z_threshold if no gap found.
    if has_z_rges and z_threshold is not None:
        from config.scoring_thresholds import GPS_Z_STEPOFF_MIN_RATIO
        z_vals = [r["rges"] for r in results]
        stepoff = _find_z_stepoff(z_vals, floor_z=z_threshold, min_ratio=GPS_Z_STEPOFF_MIN_RATIO)
        if stepoff is not None:
            effective_cutoff = stepoff
            log.info(
                "GPS Z_RGES step-off detected at %.2f (floor=%.1f, ratio=%.1f): "
                "using data-driven boundary",
                effective_cutoff, z_threshold, GPS_Z_STEPOFF_MIN_RATIO,
            )
        else:
            effective_cutoff = z_threshold
            log.info(
                "GPS Z_RGES no step-off found; applying floor threshold %.1f",
                effective_cutoff,
            )
        above_threshold = [r for r in results if r["rges"] < -effective_cutoff]
        if len(above_threshold) >= 3:
            log.info(
                "GPS Z_RGES cutoff: %d/%d compounds pass Z_RGES < -%.2f",
                len(above_threshold), len(results), effective_cutoff,
            )
            return above_threshold
        else:
            log.info(
                "GPS Z_RGES cutoff: only %d compounds pass Z_RGES < -%.2f "
                "(BGRD may have too few perms); falling back to top_%d",
                len(above_threshold), effective_cutoff, top_n,
            )

    return results[:top_n]


def _find_z_stepoff(
    z_scores: list[float],
    floor_z: float = 2.0,
    min_ratio: float = 3.0,
) -> float | None:
    """
    Find the natural signal/noise boundary in a sorted Z_RGES distribution.

    Looks for the largest gap between consecutive Z values in the signal region
    (|Z| > floor_z). Accepts a gap as a step-off boundary if it is ≥ min_ratio
    times the median inter-compound gap across the whole signal region.

    Returns the absolute Z value at the gap midpoint (as a positive threshold),
    or None if no clear step-off is found.
    """
    import statistics

    # Work with absolute values; only consider the signal region
    signal = sorted([abs(z) for z in z_scores if abs(z) > floor_z], reverse=True)
    if len(signal) < 4:
        return None

    gaps = [signal[i] - signal[i + 1] for i in range(len(signal) - 1)]
    if not gaps:
        return None

    median_gap = statistics.median(gaps)
    if median_gap <= 0:
        return None

    max_gap_idx = max(range(len(gaps)), key=lambda i: gaps[i])
    max_gap = gaps[max_gap_idx]

    if max_gap < min_ratio * median_gap:
        return None

    # Threshold = midpoint of the gap (values left of gap are signal)
    threshold = (signal[max_gap_idx] + signal[max_gap_idx + 1]) / 2.0
    return threshold


def _pick_col(headers: list[str], candidates: list[str]) -> str | None:
    for c in candidates:
        if c in headers:
            return c
    return None


def _docker_available() -> bool:
    """Return True if Docker daemon is reachable and GPS image is present."""
    try:
        out = subprocess.run(
            ["docker", "images", "-q", _DOCKER_IMAGE],
            capture_output=True, text=True, timeout=10
        )
        return bool(out.stdout.strip())
    except Exception:
        return False
