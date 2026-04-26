"""
finngen_validation.py — FinnGen R10 independent holdout validation.

Downloads FinnGen R10 summary statistics (no UKB overlap), computes per-program
χ² enrichment (same procedure as runner.py), and computes Spearman rank
correlation between PLAtlas γ and FinnGen γ.

FinnGen R10 endpoints used:
  CAD:  I9_CHD  — Coronary heart disease
  RA:   M13_RHEUMA — Rheumatoid arthritis

Download URLs (public, no auth):
  https://storage.googleapis.com/finngen-public-data-r10/summary_stats/finngen_R10_{endpoint}.gz
"""
from __future__ import annotations

import csv
import gzip
import logging
import math
import urllib.request
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_ROOT = Path(__file__).parent.parent.parent
_DEFAULT_SUMSTATS_DIR = _ROOT / "data" / "ldsc" / "sumstats"

FINNGEN_R10_ENDPOINTS: dict[str, str] = {
    "CAD": "I9_CHD",
    "RA":  "M13_RHEUMA",
}

FINNGEN_BASE_URL = (
    "https://storage.googleapis.com/finngen-public-data-r10/summary_stats/"
    "finngen_R10_{endpoint}.gz"
)


def get_finngen_url(disease_key: str) -> str:
    """Return download URL for FinnGen R10 endpoint for disease_key."""
    endpoint = FINNGEN_R10_ENDPOINTS[disease_key.upper()]
    return FINNGEN_BASE_URL.format(endpoint=endpoint)


def compute_finngen_program_enrichment(
    sumstats_path: str | Path,
    program_snp_sets: dict[str, set[str]],  # program_id → set of rsids
    min_maf: float = 0.01,
) -> dict[str, float]:
    """
    Parse FinnGen sumstats (TSV.gz, columns: #chrom pos ref alt pval beta sebeta af_alt)
    and compute mean χ² per program SNP set.
    Returns dict[program_id → mean_chisq].
    Skips SNPs with MAF < min_maf or missing beta/sebeta.
    """
    sumstats_path = Path(sumstats_path)

    # Build lookup: rsid → chisq from the sumstats file
    # FinnGen does not use rsids in the primary columns — it uses chrom:pos:ref:alt.
    # We store variant keys as "{chrom}:{pos}:{ref}:{alt}" (lowercased).
    # Callers should build program_snp_sets using the same key format.
    variant_chisq: dict[str, float] = {}

    with gzip.open(sumstats_path, "rt") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        # Handle column name aliasing: FinnGen uses #chrom, pos, ref, alt, beta, sebeta, af_alt
        for row in reader:
            # Normalise column names (strip leading '#')
            norm: dict[str, str] = {k.lstrip("#"): v for k, v in row.items()}
            try:
                beta_s   = norm.get("beta", "")
                sebeta_s = norm.get("sebeta", "")
                af_s     = norm.get("af_alt", "")
                if not beta_s or not sebeta_s or not af_s:
                    continue
                beta   = float(beta_s)
                sebeta = float(sebeta_s)
                af     = float(af_s)
            except (ValueError, KeyError):
                continue

            if sebeta <= 0:
                continue

            # MAF filter
            maf = min(af, 1.0 - af)
            if maf < min_maf:
                continue

            chrom = norm.get("chrom", "")
            pos   = norm.get("pos", "")
            ref   = norm.get("ref", "").upper()
            alt   = norm.get("alt", "").upper()

            # Variant key: chrom:pos:ref:alt
            vkey = f"{chrom}:{pos}:{ref}:{alt}"
            chisq = (beta / sebeta) ** 2
            variant_chisq[vkey] = chisq

    if not variant_chisq:
        log.warning("No valid SNPs read from FinnGen sumstats: %s", sumstats_path)
        return {}

    # Compute mean chisq per program
    result: dict[str, float] = {}
    for prog, snp_set in program_snp_sets.items():
        matched = [variant_chisq[v] for v in snp_set if v in variant_chisq]
        if matched:
            result[prog] = sum(matched) / len(matched)
        else:
            result[prog] = 0.0

    return result


def spearman_rank_correlation(
    gamma_a: dict[str, float],
    gamma_b: dict[str, float],
) -> dict:
    """
    Compute Spearman rank correlation between two γ dicts (keyed by program_id).
    Only uses keys present in both dicts.
    Returns: {"rho": float, "n": int, "p_approx": float}
    p_approx uses t-approximation: t = rho * sqrt((n-2)/(1-rho²)), df=n-2
    """
    common_keys = sorted(set(gamma_a) & set(gamma_b))
    n = len(common_keys)

    if n == 0:
        return {"rho": float("nan"), "n": 0, "p_approx": float("nan")}

    if n == 1:
        return {"rho": float("nan"), "n": 1, "p_approx": float("nan")}

    vals_a = [gamma_a[k] for k in common_keys]
    vals_b = [gamma_b[k] for k in common_keys]

    # Compute ranks (average ranks for ties)
    def _rank(vals: list[float]) -> list[float]:
        indexed = sorted(enumerate(vals), key=lambda x: x[1])
        ranks = [0.0] * len(vals)
        i = 0
        while i < len(indexed):
            j = i
            # Find ties
            while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1  # 1-based average rank
            for k in range(i, j + 1):
                ranks[indexed[k][0]] = avg_rank
            i = j + 1
        return ranks

    ranks_a = _rank(vals_a)
    ranks_b = _rank(vals_b)

    # Pearson on ranks = Spearman
    mean_a = sum(ranks_a) / n
    mean_b = sum(ranks_b) / n
    cov    = sum((ranks_a[i] - mean_a) * (ranks_b[i] - mean_b) for i in range(n))
    var_a  = sum((r - mean_a) ** 2 for r in ranks_a)
    var_b  = sum((r - mean_b) ** 2 for r in ranks_b)

    denom = math.sqrt(var_a * var_b)
    if denom == 0:
        rho = 0.0
    else:
        rho = cov / denom

    # Clamp to [-1, 1] for numerical safety
    rho = max(-1.0, min(1.0, rho))

    # p-value: t-approximation
    if n <= 2:
        p_approx = 1.0
    else:
        rho2 = rho * rho
        if rho2 >= 1.0:
            p_approx = 0.0
        else:
            t_stat = rho * math.sqrt((n - 2) / (1.0 - rho2))
            df = n - 2
            if df < 30:
                # Normal approximation via erfc
                p_approx = math.erfc(abs(t_stat) / math.sqrt(2))
            else:
                # Normal CDF approximation
                p_approx = 2.0 * (0.5 * math.erfc(abs(t_stat) / math.sqrt(2)))

    return {"rho": round(rho, 6), "n": n, "p_approx": round(p_approx, 6)}


def run_finngen_holdout_validation(
    disease_key: str,
    platlas_gammas: dict[str, float],     # program → γ from PLAtlas
    program_snp_sets: dict[str, set[str]],
    sumstats_path: str | Path | None = None,
) -> dict:
    """
    Full pipeline: download if needed, enrich, correlate.
    If sumstats_path is None, uses default: data/ldsc/sumstats/finngen_R10_{endpoint}.gz
    Returns: {
        "disease_key": str,
        "endpoint": str,
        "rho": float,
        "p_approx": float,
        "n_programs": int,
        "platlas_top3": list[str],   # top 3 programs by PLAtlas γ
        "finngen_top3": list[str],   # top 3 programs by FinnGen γ
        "stable": bool,              # rho >= 0.5
    }
    If sumstats file doesn't exist: return {"disease_key": disease_key, "skipped": True, "reason": "sumstats not downloaded"}
    """
    disease_key = disease_key.upper()
    endpoint = FINNGEN_R10_ENDPOINTS.get(disease_key)
    if endpoint is None:
        return {
            "disease_key": disease_key,
            "skipped": True,
            "reason": f"no FinnGen endpoint configured for {disease_key}",
        }

    # Resolve sumstats path
    if sumstats_path is None:
        _DEFAULT_SUMSTATS_DIR.mkdir(parents=True, exist_ok=True)
        sumstats_path = _DEFAULT_SUMSTATS_DIR / f"finngen_R10_{endpoint}.gz"
    else:
        sumstats_path = Path(sumstats_path)

    if not sumstats_path.exists():
        return {
            "disease_key": disease_key,
            "skipped": True,
            "reason": "sumstats not downloaded",
        }

    # Compute FinnGen enrichment
    finngen_gammas = compute_finngen_program_enrichment(
        sumstats_path, program_snp_sets
    )

    if not finngen_gammas:
        return {
            "disease_key": disease_key,
            "skipped": True,
            "reason": "no enrichment computed from FinnGen sumstats",
        }

    # Spearman correlation
    corr = spearman_rank_correlation(platlas_gammas, finngen_gammas)

    # Top 3 programs
    platlas_top3 = sorted(platlas_gammas, key=lambda k: platlas_gammas[k], reverse=True)[:3]
    finngen_top3  = sorted(finngen_gammas, key=lambda k: finngen_gammas[k], reverse=True)[:3]

    rho = corr["rho"]
    stable = (not math.isnan(rho)) and (rho >= 0.5)

    return {
        "disease_key":   disease_key,
        "endpoint":      endpoint,
        "rho":           corr["rho"],
        "p_approx":      corr["p_approx"],
        "n_programs":    corr["n"],
        "platlas_top3":  platlas_top3,
        "finngen_top3":  finngen_top3,
        "stable":        stable,
    }
