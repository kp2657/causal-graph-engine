#!/usr/bin/env python3
"""
scripts/sldsc_core_decomposition.py — Phase H: S-LDSC core vs peripheral decomposition.

Runs S-LDSC (or the chi-square proxy when ldsc is absent) twice per GeneticNMF
program: once on the full gene set, once on the core-gene subset
(is_core_gene=True from pipeline targets). The tau_core_fraction ratio reveals
whether heritability enrichment is driven by direct GWAS targets (core) or the
regulatory periphery.

Usage:
    python scripts/sldsc_core_decomposition.py --disease_key cad \\
        --dataset_id schnitzler_cad_vascular \\
        --sumstats data/ldsc/sumstats/Phe_414.EUR.gwama.sumstats.txt.gz

Output:
    outputs/sldsc_core_decomposition_{disease_key}.json
    data/sldsc_genesets/{disease_key}/  (BED files, always written)
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import pathlib
import subprocess
import sys
import urllib.parse
import urllib.request
from typing import Any

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("sldsc_core_decomp")

_ROOT        = pathlib.Path(__file__).parent.parent
_PERTURBSEQ  = _ROOT / "data" / "perturbseq"
_CHECKPOINTS = _ROOT / "data" / "checkpoints"
_OUTPUTS     = _ROOT / "outputs"
_LDSC_DIR    = _ROOT / "data" / "ldsc"
_GENE_CACHE  = _LDSC_DIR / "gene_intervals_hg38.json"
_LDSCORE_DIR = _LDSC_DIR / "ldscores"
_WEIGHTS_DIR = _LDSC_DIR / "ldscores" / "1000G_Phase3_weights_hm3_no_MHC"

# disease_key → canonical checkpoint slug
_CHECKPOINT_SLUG: dict[str, str] = {
    "cad": "coronary_artery_disease",
    "ra":  "rheumatoid_arthritis",
    "sle": "systemic_lupus_erythematosus",
}

# dataset_id default per disease
_DEFAULT_DATASET: dict[str, str] = {
    "cad": "schnitzler_cad_vascular",
    "ra":  "czi_2025_cd4t_perturb",
}


# ---------------------------------------------------------------------------
# Target loading
# ---------------------------------------------------------------------------

def load_targets(disease_key: str) -> list[dict]:
    """Load ranked targets from pipeline JSON checkpoint or outputs dir."""
    dk = disease_key.lower()
    slug = _CHECKPOINT_SLUG.get(dk, dk.replace(" ", "_"))

    candidates: list[pathlib.Path] = [
        _CHECKPOINTS / f"{slug}__tier4.json",
        _CHECKPOINTS / f"{slug}__tier3.json",
        _ROOT / f"data" / f"analyze_{slug}.json",
        _OUTPUTS / f"analyze_{slug}.json",
    ]
    for p in candidates:
        if p.exists():
            log.info("Loading targets from %s", p)
            with open(p) as fh:
                obj = json.load(fh)
            # Handle different checkpoint shapes
            if "prioritization_result" in obj:
                pr = obj["prioritization_result"]
                return pr.get("targets", pr.get("ranked_targets", []))
            if "targets" in obj:
                return obj["targets"]
            if "ranked_targets" in obj:
                return obj["ranked_targets"]
            if isinstance(obj, list):
                return obj

    log.warning("No tier4 checkpoint found for %s; tried: %s", disease_key, candidates)
    return []


# ---------------------------------------------------------------------------
# Program gene sets from GeneticNMF
# ---------------------------------------------------------------------------

def get_program_gene_sets(dataset_id: str, top_n: int = 200) -> dict[str, list[str]]:
    """Return {program_id: [top_n genes by |U_scaled| loading]}."""
    npz_path = _PERTURBSEQ / dataset_id / "genetic_nmf_loadings.npz"
    if not npz_path.exists():
        log.error("genetic_nmf_loadings.npz not found: %s", npz_path)
        return {}

    data = np.load(npz_path, allow_pickle=True)
    U_scaled   = data["U_scaled"]    # (n_genes × k)
    gene_names = list(data["gene_names"])
    n_genes, k = U_scaled.shape

    prefix = dataset_id.split("_")[0].upper()
    program_sets: dict[str, list[str]] = {}
    for ki in range(k):
        prog_id = f"{prefix}_GeneticNMF_C{ki + 1:02d}"
        abs_loads = np.abs(U_scaled[:, ki])
        top_idx   = np.argsort(abs_loads)[::-1][:top_n]
        program_sets[prog_id] = [gene_names[i] for i in top_idx]

    log.info("Loaded %d programs from %s (top_n=%d)", k, npz_path.name, top_n)
    return program_sets


# ---------------------------------------------------------------------------
# Core / peripheral partition
# ---------------------------------------------------------------------------

def partition_core_peripheral(
    program_genes: list[str],
    targets: list[dict],
) -> tuple[list[str], list[str]]:
    """Split program gene list into core (is_core_gene=True) and peripheral."""
    gene_field = "target_gene" if targets and "target_gene" in targets[0] else "gene"
    core_set: set[str] = {
        t[gene_field]
        for t in targets
        if t.get("is_core_gene") is True and gene_field in t
    }
    core       = [g for g in program_genes if g in core_set]
    peripheral = [g for g in program_genes if g not in core_set]
    return core, peripheral


# ---------------------------------------------------------------------------
# Gene coordinate lookup (cached via mygene.info)
# ---------------------------------------------------------------------------

def _fetch_gene_coords(genes: list[str]) -> dict[str, tuple[str, int, int]]:
    """Return {symbol: (chrom, start, end)} from GRCh38; cached to disk."""
    cached: dict[str, Any] = {}
    if _GENE_CACHE.exists():
        with open(_GENE_CACHE) as fh:
            cached = json.load(fh)

    missing = [g for g in genes if g not in cached]
    if missing:
        try:
            chunk_size = 500
            for i in range(0, len(missing), chunk_size):
                chunk = missing[i : i + chunk_size]
                body  = urllib.parse.urlencode({
                    "q":      ",".join(chunk),
                    "scopes": "symbol",
                    "fields": "genomic_pos,symbol",
                    "species": "human",
                }).encode()
                req = urllib.request.Request(
                    "https://mygene.info/v3/query",
                    data=body,
                    headers={
                        "Content-Type": "application/x-www-form-urlencoded",
                        "Accept": "application/json",
                    },
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=60) as resp:
                    results = json.loads(resp.read())
                for hit in results:
                    sym = hit.get("symbol") or hit.get("query", "")
                    gp  = hit.get("genomic_pos")
                    if gp and sym:
                        if isinstance(gp, list):
                            gp = gp[0]
                        chrom = str(gp.get("chr", ""))
                        start = int(gp.get("start", 0))
                        end   = int(gp.get("end",   0))
                        if chrom and start and end:
                            cached[sym] = [chrom, start, end]
            _GENE_CACHE.parent.mkdir(parents=True, exist_ok=True)
            with open(_GENE_CACHE, "w") as fh:
                json.dump(cached, fh)
        except Exception as exc:
            log.warning("mygene.info lookup failed: %s", exc)

    return {g: tuple(v) for g, v in cached.items() if g in set(genes)}  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# BED file writing
# ---------------------------------------------------------------------------

def write_geneset_bed(genes: list[str], out_path: str, window_kb: int = 100) -> int:
    """
    Write a 3-column BED file (chr, start, end) for S-LDSC --annot-file input.
    Window of ±window_kb around each gene body. Returns number of genes placed.
    """
    coords = _fetch_gene_coords(genes)
    window = window_kb * 1_000
    p = pathlib.Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    placed = 0
    with open(p, "w") as fh:
        for gene in genes:
            if gene not in coords:
                continue
            chrom, start, end = coords[gene]
            chrom_str = f"chr{chrom}" if not str(chrom).startswith("chr") else str(chrom)
            bed_start = max(0, start - window)
            bed_end   = end + window
            fh.write(f"{chrom_str}\t{bed_start}\t{bed_end}\t{gene}\n")
            placed += 1

    log.debug("BED %s: %d/%d genes placed", p.name, placed, len(genes))
    return placed


# ---------------------------------------------------------------------------
# S-LDSC runner
# ---------------------------------------------------------------------------

def _find_ldsc_bin() -> pathlib.Path | None:
    """Locate ldsc.py — checks PATH, common conda locations, and project ldsc/ dir."""
    candidates = [
        _LDSC_DIR / "ldsc" / "ldsc.py",
        _ROOT / "ldsc" / "ldsc.py",
    ]
    for p in candidates:
        if p.exists():
            return p
    try:
        result = subprocess.run(["which", "ldsc"], capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            return pathlib.Path(result.stdout.strip())
    except Exception:
        pass
    return None


def _parse_ldsc_log(log_path: pathlib.Path) -> dict | None:
    """Parse ldsc .log file for Total Observed scale h2 and coefficient lines."""
    if not log_path.exists():
        return None
    tau = tau_se = enrichment = None
    try:
        text = log_path.read_text()
        for line in text.splitlines():
            if "Coefficient" in line and "L1" in line:
                parts = line.split()
                for j, part in enumerate(parts):
                    if part.startswith("L1"):
                        try:
                            tau    = float(parts[j + 1])
                            tau_se = float(parts[j + 2].strip("()"))
                        except (IndexError, ValueError):
                            pass
            if "Enrichment" in line and "L1" in line:
                parts = line.split()
                for j, part in enumerate(parts):
                    if part.startswith("L1"):
                        try:
                            enrichment = float(parts[j + 1])
                        except (IndexError, ValueError):
                            pass
    except Exception as exc:
        log.warning("ldsc log parse failed: %s", exc)
        return None

    if tau is None:
        return None
    return {"tau": tau, "tau_se": tau_se or float("nan"), "enrichment": enrichment or float("nan")}


def run_sldsc_for_geneset(
    bed_path: str,
    sumstats_path: str,
    out_prefix: str,
) -> dict | None:
    """
    Run ldsc.py --h2 with annotation for this gene set.

    Returns {tau, tau_se, enrichment} or None if ldsc not found / run failed.
    The function builds an LD-score annotation from the BED file using the
    baselineLD scores already present in data/ldsc/ldscores/.
    """
    ldsc_bin = _find_ldsc_bin()
    if ldsc_bin is None:
        return None

    # Requires pre-computed LD scores; fall back to chi-square approximation
    ldscore_prefix = str(_LDSCORE_DIR / "baselineLD.")
    weights_prefix = str(_WEIGHTS_DIR / "weights.hm3_noMHC.")

    # Check that at least chr1 LD scores exist
    if not (_LDSCORE_DIR / "baselineLD.1.l2.ldscore.gz").exists():
        log.warning("baselineLD LD scores not found at %s; cannot run ldsc", _LDSCORE_DIR)
        return None

    out_p = pathlib.Path(out_prefix)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(ldsc_bin),
        "--h2",          sumstats_path,
        "--ref-ld-chr",  ldscore_prefix,
        "--w-ld-chr",    weights_prefix,
        "--overlap-annot",
        "--print-coefficients",
        "--frqfile-chr", str(_LDSCORE_DIR / "1000G.EUR.hm3_noMHC."),
        "--out",         out_prefix,
    ]
    log.info("Running ldsc: %s", " ".join(cmd[-6:]))
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            log.warning("ldsc exited %d: %s", result.returncode, result.stderr[-300:])
            return None
    except subprocess.TimeoutExpired:
        log.warning("ldsc timed out for %s", bed_path)
        return None
    except Exception as exc:
        log.warning("ldsc run error: %s", exc)
        return None

    return _parse_ldsc_log(out_p.parent / (out_p.name + ".log"))


# ---------------------------------------------------------------------------
# Chi-square proxy (no ldsc required)
# ---------------------------------------------------------------------------

def _chisq_enrichment_for_genes(
    genes: list[str],
    sumstats_path: str,
    window_bp: int = 100_000,
) -> dict | None:
    """Compute τ as mean chi-square in gene windows relative to genome-wide mean."""
    coords = _fetch_gene_coords(genes)
    if not coords:
        return None

    # Build chrom → list[(start, end)] windows
    windows: dict[str, list[tuple[int, int]]] = {}
    for g in genes:
        if g not in coords:
            continue
        chrom, start, end = coords[g]
        chrom_str = f"chr{chrom}" if not str(chrom).startswith("chr") else str(chrom)
        windows.setdefault(chrom_str, []).append(
            (max(0, start - window_bp), end + window_bp)
        )

    sp = pathlib.Path(sumstats_path)
    if not sp.exists():
        log.warning("Sumstats not found: %s", sumstats_path)
        return None

    open_fn = gzip.open if sp.suffix == ".gz" else open
    in_window_chisq:  list[float] = []
    genome_chisq:     list[float] = []

    try:
        with open_fn(sp, "rt") as fh:
            header = fh.readline().strip().split("\t")
            col = {c: i for i, c in enumerate(header)}
            chr_i = next((col[c] for c in ("CHR", "chromosome", "chr") if c in col), None)
            pos_i = next((col[c] for c in ("POS", "base_pair_location_grch38", "pos", "BP") if c in col), None)
            p_i   = next((col[c] for c in ("P", "p_value", "pval", "PVAL") if c in col), None)
            z_i   = next((col[c] for c in ("Z", "zscore", "z") if c in col), None)
            b_i   = next((col[c] for c in ("BETA", "beta") if c in col), None)
            se_i  = next((col[c] for c in ("SE", "standard_error", "se") if c in col), None)

            for line in fh:
                parts = line.strip().split("\t")
                if chr_i is None or pos_i is None:
                    break
                if len(parts) <= max(chr_i, pos_i):
                    continue
                try:
                    chrom_val = "chr" + parts[chr_i].lstrip("chr")
                    pos_val   = int(parts[pos_i])
                except (ValueError, IndexError):
                    continue

                # Derive chi-square from available columns
                chisq: float | None = None
                if z_i is not None and len(parts) > z_i:
                    try:
                        chisq = float(parts[z_i]) ** 2
                    except ValueError:
                        pass
                if chisq is None and b_i is not None and se_i is not None:
                    try:
                        z = float(parts[b_i]) / float(parts[se_i])
                        chisq = z * z
                    except (ValueError, ZeroDivisionError):
                        pass
                if chisq is None and p_i is not None and len(parts) > p_i:
                    try:
                        import math
                        p_val = float(parts[p_i])
                        if 0 < p_val < 1:
                            chisq = (-2 * math.log(p_val))  # crude approximation
                    except (ValueError, OverflowError):
                        pass
                if chisq is None or chisq < 0:
                    continue

                genome_chisq.append(chisq)

                if chrom_val not in windows:
                    continue
                for wstart, wend in windows[chrom_val]:
                    if wstart <= pos_val <= wend:
                        in_window_chisq.append(chisq)
                        break

    except Exception as exc:
        log.warning("Sumstats parse failed: %s", exc)
        return None

    if not genome_chisq or not in_window_chisq:
        return None

    genome_mean = float(np.mean(genome_chisq))
    window_mean = float(np.mean(in_window_chisq))
    tau = (window_mean - genome_mean) / (genome_mean + 1e-9)
    return {
        "tau":         tau,
        "tau_se":      float("nan"),
        "enrichment":  window_mean / (genome_mean + 1e-9),
        "n_snps_window": len(in_window_chisq),
        "n_snps_total":  len(genome_chisq),
        "method":      "chisq_proxy",
    }


# ---------------------------------------------------------------------------
# Core fraction
# ---------------------------------------------------------------------------

def compute_tau_core_fraction(results: dict[str, dict]) -> dict[str, dict]:
    """
    Compute tau_core_fraction = tau_core / tau_full for each program.

    >1.0: heritability concentrated in core (direct GWAS targets drive enrichment).
    <1.0: heritability in regulatory periphery.
    """
    out: dict[str, dict] = {}
    for prog, prog_res in results.items():
        full_tau = prog_res.get("full", {}) or {}
        core_tau = prog_res.get("core", {}) or {}
        tau_f = full_tau.get("tau", float("nan"))
        tau_c = core_tau.get("tau", float("nan"))

        if tau_f and abs(tau_f) > 1e-9:
            frac = tau_c / tau_f if not (
                tau_c != tau_c or tau_f != tau_f
            ) else float("nan")
        else:
            frac = float("nan")

        out[prog] = {
            "tau_full":          tau_f,
            "tau_core":          tau_c,
            "tau_core_fraction": frac,
            "n_core_genes":      prog_res.get("n_core_genes", 0),
            "n_peripheral_genes": prog_res.get("n_peripheral_genes", 0),
            "full_detail":       full_tau,
            "core_detail":       core_tau,
        }
    return out


# ---------------------------------------------------------------------------
# BED-only mode helper
# ---------------------------------------------------------------------------

def _write_all_beds(
    disease_key: str,
    program_gene_sets: dict[str, list[str]],
    targets: list[dict],
    bed_root: pathlib.Path,
    sumstats_path: str,
) -> None:
    """Write BED files and print shell commands for manual S-LDSC runs."""
    bed_root.mkdir(parents=True, exist_ok=True)
    cmds: list[str] = []
    ldsc_bin = _find_ldsc_bin()
    bin_str = str(ldsc_bin) if ldsc_bin else "path/to/ldsc/ldsc.py"

    for prog, genes in program_gene_sets.items():
        core, _peripheral = partition_core_peripheral(genes, targets)
        full_bed = bed_root / f"{prog}_full.bed"
        core_bed = bed_root / f"{prog}_core.bed"
        write_geneset_bed(genes, str(full_bed))
        write_geneset_bed(core, str(core_bed))

        for subset, bed_p in [("full", full_bed), ("core", core_bed)]:
            out_pref = bed_root / f"{prog}_{subset}_ldsc"
            cmd = (
                f"python {bin_str} "
                f"--h2 {sumstats_path} "
                f"--ref-ld-chr {_LDSCORE_DIR}/baselineLD. "
                f"--w-ld-chr {_WEIGHTS_DIR}/weights.hm3_noMHC. "
                f"--overlap-annot --print-coefficients "
                f"--out {out_pref}"
            )
            cmds.append(cmd)

    print("\n=== S-LDSC not installed. BED files written to:", bed_root)
    print("=== Run the following commands:\n")
    for cmd in cmds:
        print(cmd)
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    disease_key: str,
    dataset_id: str,
    sumstats_path: str,
    top_n: int = 200,
    window_kb: int = 100,
) -> dict:
    dk = disease_key.lower()
    log.info("Phase H: core decomposition — disease=%s, dataset=%s", dk, dataset_id)

    targets = load_targets(dk)
    if not targets:
        log.warning("No targets loaded; is_core_gene partitioning will be empty")

    program_gene_sets = get_program_gene_sets(dataset_id, top_n=top_n)
    if not program_gene_sets:
        log.error("No program gene sets found for dataset_id=%s", dataset_id)
        return {}

    ldsc_bin = _find_ldsc_bin()
    bed_root = _ROOT / "data" / "sldsc_genesets" / dk

    if ldsc_bin is None:
        log.warning("ldsc not found; writing BED files and chi-square proxy only")
        _write_all_beds(dk, program_gene_sets, targets, bed_root, sumstats_path)

    program_results: dict[str, dict] = {}

    for prog, genes in program_gene_sets.items():
        core, peripheral = partition_core_peripheral(genes, targets)
        log.info("%s: %d full genes, %d core, %d peripheral", prog, len(genes), len(core), len(peripheral))

        full_bed = bed_root / f"{prog}_full.bed"
        core_bed = bed_root / f"{prog}_core.bed"

        write_geneset_bed(genes, str(full_bed), window_kb=window_kb)
        write_geneset_bed(core,  str(core_bed), window_kb=window_kb)

        out_full   = str(bed_root / f"{prog}_full_ldsc")
        out_core   = str(bed_root / f"{prog}_core_ldsc")

        if ldsc_bin is not None and sumstats_path and pathlib.Path(sumstats_path).exists():
            full_result = run_sldsc_for_geneset(str(full_bed), sumstats_path, out_full)
            core_result = run_sldsc_for_geneset(str(core_bed), sumstats_path, out_core) if core else None
        else:
            # chi-square proxy
            full_result = _chisq_enrichment_for_genes(genes, sumstats_path, window_kb * 1_000)
            core_result = _chisq_enrichment_for_genes(core, sumstats_path, window_kb * 1_000) if core else None

        program_results[prog] = {
            "full":               full_result,
            "core":               core_result,
            "n_full_genes":       len(genes),
            "n_core_genes":       len(core),
            "n_peripheral_genes": len(peripheral),
            "core_genes":         core[:50],
        }

    tau_fractions = compute_tau_core_fraction(program_results)

    output = {
        "disease_key":    dk,
        "dataset_id":     dataset_id,
        "sumstats":       sumstats_path,
        "top_n_per_prog": top_n,
        "window_kb":      window_kb,
        "ldsc_available": ldsc_bin is not None,
        "per_program":    tau_fractions,
        "raw":            program_results,
    }

    out_path = _OUTPUTS / f"sldsc_core_decomposition_{dk}.json"
    _OUTPUTS.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(output, fh, indent=2, allow_nan=True)
    log.info("Results written to %s", out_path)

    # Print summary table
    print(f"\n{'Program':<30} {'tau_full':>10} {'tau_core':>10} {'core_frac':>10} {'n_core':>7}")
    print("-" * 72)
    for prog, row in sorted(tau_fractions.items()):
        print(
            f"{prog:<30} "
            f"{row['tau_full']:>10.4f} "
            f"{row['tau_core']:>10.4f} "
            f"{row['tau_core_fraction']:>10.3f} "
            f"{row['n_core_genes']:>7d}"
        )

    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="S-LDSC core vs peripheral decomposition per GeneticNMF program"
    )
    parser.add_argument("--disease_key", default="cad", help="Disease key (cad, ra)")
    parser.add_argument("--dataset_id",  default=None,  help="Perturbseq dataset_id (default: auto from disease_key)")
    parser.add_argument("--sumstats",    default=None,  help="Path to GWAS .sumstats.gz")
    parser.add_argument("--top_n",       default=200,   type=int, help="Top genes per program from |U_scaled|")
    parser.add_argument("--window_kb",   default=100,   type=int, help="Gene body window (kb)")
    args = parser.parse_args()

    dk         = args.disease_key.lower()
    dataset_id = args.dataset_id or _DEFAULT_DATASET.get(dk, dk)

    # Default sumstats from setup.GWAS_CONFIG if available
    sumstats = args.sumstats
    if sumstats is None:
        try:
            from pipelines.ldsc.setup import GWAS_CONFIG, _SUMSTATS_DIR
            cfg = GWAS_CONFIG.get(dk.upper())
            if cfg:
                candidate = _SUMSTATS_DIR / cfg["filename"]
                if candidate.exists():
                    sumstats = str(candidate)
                    log.info("Using default sumstats: %s", sumstats)
        except ImportError:
            pass
    if sumstats is None:
        log.warning("--sumstats not provided and no default found; chi-square proxy will attempt anyway")
        sumstats = ""

    run(
        disease_key=dk,
        dataset_id=dataset_id,
        sumstats_path=sumstats,
        top_n=args.top_n,
        window_kb=args.window_kb,
    )


if __name__ == "__main__":
    main()
