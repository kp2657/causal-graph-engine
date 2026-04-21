"""
pipelines/upstream_recovery_benchmark.py

Phase J: Upstream regulator recovery benchmark.

Evaluates whether a gene ranking produced by the causal pipeline correctly places
known upstream regulators above terminal markers.

Two metrics:
  upstream_recovery_ratio
      median_rank(upstream_set) / median_rank(marker_set)
      < 1.0  → upstream genes rank higher than markers  ✓
      = 1.0  → indistinguishable
      > 1.0  → markers rank above upstream genes  ✗

  marker_rank_delta
      median_rank_post(marker_set) − median_rank_pre(marker_set)
      Negative → markers dropped in ranking after the intervention  ✓
      Positive → markers rose after the intervention  ✗

Rank convention: 0-based, lower index = higher priority (index 0 = top-ranked gene).

Public API:
    load_benchmark_config(path)
    compute_upstream_recovery_ratio(ranked_genes, upstream_set, marker_set)
    compute_marker_rank_delta(pre_ranked, post_ranked, marker_set)
    evaluate_ranking(ranked_genes, config, pre_ranked=None)
"""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Sequence


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_benchmark_config(path: str | Path) -> dict:
    """Load and validate a benchmark config JSON.

    Args:
        path: Path to a benchmark JSON file (e.g. ibd_upstream_regulators_v1.json).

    Returns:
        Parsed config dict.

    Raises:
        FileNotFoundError: if file does not exist.
        KeyError: if required keys are missing.
    """
    p = Path(path)
    with p.open() as f:
        cfg = json.load(f)
    # Validate required keys
    for key in ("upstream_regulators", "terminal_markers"):
        if key not in cfg:
            raise KeyError(f"Benchmark config missing required key: {key!r}")
    if not cfg["upstream_regulators"]:
        raise ValueError("upstream_regulators list is empty")
    if not cfg["terminal_markers"]:
        raise ValueError("terminal_markers list is empty")
    return cfg


# ---------------------------------------------------------------------------
# Core metric helpers
# ---------------------------------------------------------------------------

def _median_rank(genes: Sequence[str], ranked_list: Sequence[str]) -> float:
    """Compute median 0-based rank of *genes* within *ranked_list*.

    Genes absent from ranked_list are silently skipped.  Returns float('nan')
    if no gene from *genes* appears in *ranked_list*.
    """
    index = {g: i for i, g in enumerate(ranked_list)}
    ranks = [index[g] for g in genes if g in index]
    if not ranks:
        return float("nan")
    return statistics.median(ranks)


def compute_upstream_recovery_ratio(
    ranked_genes: Sequence[str],
    upstream_set: Sequence[str],
    marker_set: Sequence[str],
) -> float:
    """Compute upstream_recovery_ratio.

    = median_rank(upstream_set) / median_rank(marker_set)

    Interpretation:
        < 1.0  upstream genes rank above markers  ✓
        = 1.0  indistinguishable
        > 1.0  markers rank above upstream genes  ✗

    Returns float('nan') if either set has no members in ranked_genes, or if
    median_rank(marker_set) == 0 (markers at position 0 — degenerate).
    """
    upstream_rank = _median_rank(upstream_set, ranked_genes)
    marker_rank = _median_rank(marker_set, ranked_genes)
    if not math.isfinite(upstream_rank) or not math.isfinite(marker_rank):
        return float("nan")
    if marker_rank == 0.0:
        return float("nan")
    return upstream_rank / marker_rank


def compute_marker_rank_delta(
    pre_ranked: Sequence[str],
    post_ranked: Sequence[str],
    marker_set: Sequence[str],
) -> float:
    """Compute marker_rank_delta.

    = median_rank_post(marker_set) − median_rank_pre(marker_set)

    Interpretation:
        Negative → markers dropped in ranking (good — Phase H marker_discount worked)
        Positive → markers rose after intervention (bad)
        0        → no change

    Returns float('nan') if marker_set has no members in either ranking.
    """
    pre = _median_rank(marker_set, pre_ranked)
    post = _median_rank(marker_set, post_ranked)
    if not math.isfinite(pre) or not math.isfinite(post):
        return float("nan")
    return post - pre


# ---------------------------------------------------------------------------
# High-level evaluator
# ---------------------------------------------------------------------------

def evaluate_ranking(
    ranked_genes: Sequence[str],
    config: dict,
    pre_ranked: Sequence[str] | None = None,
) -> dict:
    """Evaluate a gene ranking against a benchmark config.

    Args:
        ranked_genes:  Ordered gene list (index 0 = top-ranked).
        config:        Loaded benchmark config (from load_benchmark_config).
        pre_ranked:    Optional pre-intervention ranking for marker_rank_delta.
                       If None, marker_rank_delta is omitted from the result.

    Returns dict with keys:
        upstream_recovery_ratio   float — target < 1.0
        upstream_median_rank      float
        marker_median_rank        float
        upstream_in_ranking       int   — count of upstream genes found
        markers_in_ranking        int   — count of markers found
        marker_rank_delta         float | None
        pass_recovery_ratio       bool  — ratio < threshold (default 1.0)
    """
    upstream_set = config["upstream_regulators"]
    marker_set = config["terminal_markers"]
    threshold = config.get("thresholds", {}).get("upstream_recovery_ratio_max", 1.0)

    upstream_rank = _median_rank(upstream_set, ranked_genes)
    marker_rank = _median_rank(marker_set, ranked_genes)
    ratio = compute_upstream_recovery_ratio(ranked_genes, upstream_set, marker_set)

    delta: float | None = None
    if pre_ranked is not None:
        delta = compute_marker_rank_delta(pre_ranked, ranked_genes, marker_set)

    upstream_in = sum(1 for g in upstream_set if g in ranked_genes)
    markers_in = sum(1 for g in marker_set if g in ranked_genes)

    pass_ratio = math.isfinite(ratio) and ratio < threshold

    return {
        "upstream_recovery_ratio": ratio,
        "upstream_median_rank": upstream_rank,
        "marker_median_rank": marker_rank,
        "upstream_in_ranking": upstream_in,
        "markers_in_ranking": markers_in,
        "marker_rank_delta": delta,
        "pass_recovery_ratio": pass_ratio,
    }
