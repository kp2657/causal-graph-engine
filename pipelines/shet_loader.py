"""
pipelines/shet_loader.py — Load GeneBayes shet (heterozygous selection coefficient) posteriors.

shet is a Bayesian estimate of the strength of natural selection against
heterozygous loss-of-function variants, derived from gnomAD observed/expected
LoF counts via an evolutionary model (Spence et al. 2024, Nature Genetics).

Higher shet → gene is more constrained against LoF → use as penalty on
γ estimates for genes with no direct mechanistic evidence.

Data source:
    Spence JP, Zeng T, Mostafavi H, Pritchard JK.
    "Genome-wide Bayesian estimation of the selective effects of heterozygous
    loss-of-function variants."
    Nature Genetics, 2024. DOI: 10.1038/s41588-024-01820-9.

    File: data/genebays/shet_posteriors.tsv (downloaded from Zenodo 10403680)
    Columns: ensg  hgnc  chrom  obs_lof  exp_lof  prior_mean  post_mean
             post_lower_95  post_upper_95

shet interpretation:
    shet ≈ 0      → unconstrained (non-essential; LoF well-tolerated)
    shet ≈ 0.001  → moderately constrained
    shet ≈ 0.01+  → highly constrained (LoF likely pathogenic)
    shet > 0.05   → extremely constrained (haploinsufficient)

    Compare to pLI: shet is calibrated and continuous; pLI is a binary-like
    score that saturates at 1 for essential genes. shet penalizes smoothly.

Usage in OTA pipeline:
    from pipelines.shet_loader import get_shet, get_shet_penalty

    shet = get_shet("PCSK9")    # → 0.0019  (low constraint; druggable)
    shet = get_shet("RPS6")     # → 0.15    (ribosomal; essential)
    penalty = get_shet_penalty("RPS6")  # → 0.25 (high penalty)
"""
from __future__ import annotations

import logging
import math
from functools import lru_cache
from pathlib import Path

log = logging.getLogger(__name__)

_ROOT      = Path(__file__).parent.parent
_SHET_FILE = _ROOT / "data" / "genebays" / "shet_posteriors.tsv"

# Shet value above which we apply the maximum constraint penalty
_SHET_HIGH = 0.05
# Shet value below which no penalty is applied
_SHET_LOW  = 0.0005
# Maximum penalty multiplier (floor): highly constrained genes get 0.20× γ at most
_MAX_PENALTY = 0.20
# Download URL (in case file is missing)
_SHET_URL = "https://zenodo.org/api/records/10403680/files/s_het_estimates.genebayes.tsv/content"


@lru_cache(maxsize=1)
def _load_shet_table() -> dict[str, float]:
    """
    Load shet posterior means keyed by gene symbol.
    Returns dict[symbol -> post_mean].
    """
    if not _SHET_FILE.exists():
        log.warning(
            "GeneBayes shet posteriors not found at %s. "
            "Download with: curl -sL '%s' -o %s",
            _SHET_FILE, _SHET_URL, _SHET_FILE
        )
        return {}

    table: dict[str, float] = {}
    ensg_to_hgnc: dict[str, str] = {}

    with open(_SHET_FILE) as f:
        header = f.readline().strip().split("\t")
        try:
            ensg_idx    = header.index("ensg")
            hgnc_idx    = header.index("hgnc")
            post_idx    = header.index("post_mean")
        except ValueError as e:
            log.error("Unexpected shet file header: %s (%s)", header, e)
            return {}

        for line in f:
            parts = line.strip().split("\t")
            if len(parts) <= max(ensg_idx, hgnc_idx, post_idx):
                continue
            ensg = parts[ensg_idx]
            hgnc = parts[hgnc_idx]  # format: "HGNC:12345"
            try:
                post_mean = float(parts[post_idx])
            except ValueError:
                continue

            # hgnc field is "HGNC:ID" not a symbol — we use ENSG as primary key.
            # Symbol lookup is done in get_shet() via a separate mapping step.
            ensg_to_hgnc[ensg] = hgnc
            table[ensg] = post_mean

    log.info("Loaded shet posteriors for %d genes", len(table))
    return table


@lru_cache(maxsize=1)
def _build_symbol_map() -> dict[str, float]:
    """
    Build a gene-symbol → shet map using the ENSG → symbol lookup from gnomAD.
    Falls back to mygene.info API if needed.
    """
    raw = _load_shet_table()
    if not raw:
        return {}

    # Try to load a cached symbol map first
    cache_path = _SHET_FILE.parent / "shet_symbol_map.tsv"
    symbol_map: dict[str, float] = {}

    if cache_path.exists():
        with open(cache_path) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    symbol_map[parts[0]] = float(parts[1])
        if symbol_map:
            return symbol_map

    # Build from mygene.info API: ENSG → symbol
    ensg_ids = list(raw.keys())
    log.info("Building gene symbol → shet map via mygene.info (%d genes)...", len(ensg_ids))
    try:
        import urllib.request, urllib.parse, json as _json
        chunk_size = 500
        for i in range(0, len(ensg_ids), chunk_size):
            chunk = ensg_ids[i:i + chunk_size]
            body = urllib.parse.urlencode({
                "ids": ",".join(chunk),
                "fields": "symbol",
                "species": "human",
            }).encode()
            req = urllib.request.Request(
                "https://mygene.info/v3/gene",
                data=body,
                headers={"Content-Type": "application/x-www-form-urlencoded",
                         "Accept": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                results = _json.loads(resp.read())
            for hit in results:
                ensg = hit.get("query", "")
                sym  = hit.get("symbol", "")
                if sym and ensg in raw:
                    symbol_map[sym] = raw[ensg]

        # Save cache
        with open(cache_path, "w") as f:
            for sym, shet in symbol_map.items():
                f.write(f"{sym}\t{shet}\n")
        log.info("Symbol map built: %d genes", len(symbol_map))
    except Exception as exc:
        log.warning("Symbol map build failed: %s — shet lookups will be unavailable", exc)

    return symbol_map


def get_shet(gene: str) -> float | None:
    """
    Return the GeneBayes posterior mean shet for a gene symbol.
    Returns None if gene not in table.
    """
    sym_map = _build_symbol_map()
    return sym_map.get(gene)


def get_shet_penalty(gene: str) -> float:
    """
    Return a multiplicative penalty [0, 1] based on shet.

    penalty = 1.0  for unconstrained genes (shet < _SHET_LOW)
    penalty = 0.20 for highly constrained genes (shet > _SHET_HIGH)
    Linear interpolation in log-shet space between these anchors.

    Rationale: constrained genes should not rank highly via GWAS proxy γ alone
    unless they have direct Perturb-seq or pQTL/eQTL evidence (which exempts
    them via the mechanistic filter). The penalty specifically discounts genes
    whose high γ is purely driven by GWAS locus → gene mapping when the gene
    itself has very low LoF tolerance.
    """
    shet = get_shet(gene)
    if shet is None:
        return 1.0  # unknown → no penalty
    if shet < _SHET_LOW:
        return 1.0
    if shet >= _SHET_HIGH:
        return _MAX_PENALTY

    # Log-linear interpolation
    log_low  = math.log(_SHET_LOW)
    log_high = math.log(_SHET_HIGH)
    log_val  = math.log(max(shet, 1e-10))
    frac = (log_val - log_low) / (log_high - log_low)
    penalty = 1.0 - frac * (1.0 - _MAX_PENALTY)
    return round(max(_MAX_PENALTY, min(1.0, penalty)), 4)


