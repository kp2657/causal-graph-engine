"""Tests for the empiric elbow (natural gap) cutoff in Tier 3.

The cutoff finds the first position past min_keep where the relative drop
between consecutive |γ| values exceeds gap_threshold (default 20%).
If no gap is found all genes are returned.
"""
from __future__ import annotations

import pytest

from agents.tier3_causal.causal_discovery_agent import _pareto_cutoff


def _rec(gene: str, gamma: float) -> dict:
    return {"gene": gene, "ota_gamma": gamma}


def _sorted(xs: list[dict]) -> list[dict]:
    return sorted(xs, key=lambda r: abs(r["ota_gamma"]), reverse=True)


def test_elbow_found_at_first_large_gap():
    # γ = [10, 9, 8, 4, 3] — gap between index 2→3: (8-4)/8 = 50% ≥ 20%
    # With min_keep=1, cut = 3 (keep first three)
    ranked = _sorted([_rec("A", 10), _rec("B", 9), _rec("C", 8),
                      _rec("D", 4), _rec("E", 3)])
    out = _pareto_cutoff(ranked, min_keep=1)
    assert [r["gene"] for r in out] == ["A", "B", "C"]


def test_elbow_respects_min_keep_floor():
    # γ = [10, 9, 8, 2, 1] — gap at i=2→3: (8-2)/8 = 75%.
    # min_keep=4 makes us skip i=0,1,2; range starts at 4 → no gap beyond that.
    # So all 5 genes are returned (no elbow found past min_keep).
    ranked = _sorted([_rec("A", 10), _rec("B", 9), _rec("C", 8),
                      _rec("D", 2), _rec("E", 1)])
    out = _pareto_cutoff(ranked, min_keep=4)
    assert len(out) == 5


def test_no_elbow_keeps_all():
    # All equal — no gap ≥ 20%, returns all genes.
    ranked = _sorted([_rec(f"G{i}", 1.0) for i in range(80)])
    out = _pareto_cutoff(ranked, min_keep=10)
    assert len(out) == 80


def test_no_elbow_small_gradual_decline():
    # γ = [1.0, 0.9, 0.81, 0.72, ...] — each drop is 10% < 20% threshold
    import math
    ranked = [_rec(f"G{i}", 1.0 * (0.9 ** i)) for i in range(50)]
    # Already sorted descending; no drop ≥ 20%, so keep all
    out = _pareto_cutoff(ranked, min_keep=10)
    assert len(out) == 50


def test_elbow_not_fired_before_min_keep():
    # γ = [10, 1] then uniform 1,1,1,... — 90% gap is at index 0→1, before min_keep=5.
    # After index 5 all values are identical (no gap ≥ 20%), so all genes returned.
    ranked = [_rec("BIG", 10.0), _rec("SMALL", 1.0)] + [_rec(f"G{i}", 1.0) for i in range(10)]
    out = _pareto_cutoff(ranked, min_keep=5)
    # Big gap at i=0 is before min_keep=5; no gap found after → keep all
    assert len(out) == len(ranked)


def test_empty_input():
    assert _pareto_cutoff([], min_keep=50) == []


def test_max_keep_cap():
    # Large list with no gap — max_keep caps the result
    ranked = _sorted([_rec(f"G{i}", 1.0) for i in range(100)])
    out = _pareto_cutoff(ranked, min_keep=10, max_keep=30)
    assert len(out) == 30


def test_all_zero_gamma_returns_all():
    # Zero-γ genes: v_curr ≤ 1e-10 so gap never fires; returns all (already
    # filtered upstream by OTA_GAMMA_MIN before pareto runs in practice)
    ranked = [_rec(f"Z{i}", 0.0) for i in range(20)]
    out = _pareto_cutoff(ranked, min_keep=5)
    assert len(out) == 20


def test_preserves_sort_order():
    ranked = _sorted([_rec("A", 3), _rec("B", 9), _rec("C", 1), _rec("D", 7)])
    # Sorted: [B=9, D=7, A=3, C=1].
    # Loop starts at min_keep=1 (i=1): D=7→A=3, gap=(7-3)/7=57% ≥ 20% → cut=2.
    out = _pareto_cutoff(ranked, min_keep=1)
    assert [r["gene"] for r in out] == ["B", "D"]
