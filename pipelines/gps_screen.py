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
_GPS_TIMEOUT_WITH_BGRD    = 7200   # seconds when BGRD cached; 200-gene sig ~900s, 1000-gene ~4500s
_GPS_TIMEOUT_NO_BGRD      = 21600  # seconds when BGRD must be computed (permutation: 0.5-6h)
_GPS_TIMEOUT              = _GPS_TIMEOUT_NO_BGRD  # legacy alias; internal code uses per-run value
_GPS_GENES     = None  # GPS's selected_genes_2198.csv — loaded lazily
_GPS_BGRD_DIR  = Path(__file__).parent.parent / "data" / "gps_bgrd"  # persistent BGRD cache
_GPS_LOGS_DIR  = Path(__file__).parent.parent / "data" / "gps_logs"  # GPS internal logfiles


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

    Uses a persistent BGRD cache (data/gps_bgrd/) so the expensive RGES permutation
    step (0.5–6h) is only run once per unique disease signature. Subsequent calls
    for the same label skip permutation and run only Run_reversal_score.py.
    """
    if not _check_gps_library_overlap(label, sig):
        return []

    safe_label = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)[:40]
    bgrd_dir = _GPS_BGRD_DIR
    bgrd_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = _GPS_LOGS_DIR
    logs_dir.mkdir(parents=True, exist_ok=True)

    # BGRD null distribution depends only on signature SIZE, not gene identity.
    # Round to the nearest canonical bucket so any two runs with the same size
    # share one pre-computed BGRD forever (AMD + CAD disease-state both use ~700).
    _SIZE_BUCKETS = [5, 10, 20, 50, 100, 200, 300, 500, 700, 1000]
    sig_size = len(sig)
    bgrd_size = min(_SIZE_BUCKETS, key=lambda b: abs(b - sig_size))
    bgrd_key  = f"size{bgrd_size}"

    with tempfile.TemporaryDirectory(prefix="gps_screen_") as tmpdir:
        tmp = Path(tmpdir)
        input_dir  = tmp / "input"
        output_dir = tmp / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        sig_filename = f"DZSIG__{safe_label}.csv"
        _write_dzsig_csv(sig, input_dir / sig_filename)

        # If a cached background exists, copy it into the dzsig dir and pass
        # --RGES_bgrd_ID so GPS skips the expensive permutation step.
        bgrd_pkl = bgrd_dir / f"BGRD__{bgrd_key}.pkl"
        bgrd_id_arg = ["--RGES_bgrd_ID", bgrd_key] if bgrd_pkl.exists() else []
        run_timeout = _GPS_TIMEOUT_WITH_BGRD if bgrd_pkl.exists() else _GPS_TIMEOUT_NO_BGRD
        if bgrd_pkl.exists():
            log.info("GPS screen: %s — using cached BGRD size=%d (timeout=%ds)", label, bgrd_size, run_timeout)
        else:
            log.info(
                "GPS screen: %s vs %s library (%d sig genes, BGRD size=%d) — running permutation "
                "(first time, will cache; timeout=%ds)",
                label, library, len(sig), bgrd_size, run_timeout,
            )

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

    # Z_RGES threshold — governs when GPS output is z-scored against permuted null
    if has_z_rges and z_threshold is not None:
        above_threshold = [r for r in results if r["rges"] < -z_threshold]
        if len(above_threshold) >= 3:
            log.info(
                "GPS Z_RGES cutoff: %d/%d compounds pass Z_RGES < -%.1f",
                len(above_threshold), len(results), z_threshold,
            )
            return above_threshold
        else:
            log.info(
                "GPS Z_RGES cutoff: only %d compounds pass Z_RGES < -%.1f "
                "(BGRD may have too few perms); falling back to top_%d",
                len(above_threshold), z_threshold, top_n,
            )

    return results[:top_n]


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
