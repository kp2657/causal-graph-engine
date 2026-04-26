"""
state_edge_bootstrap.py

Bootstrap / CI estimation for state-space (non-instrumented) target→disease edges.

These edges are derived from therapeutic redirection (TR) / state-influence signals
and therefore do not have an OTA γ in the strict genetic-instrument sense.

We approximate uncertainty by bootstrapping the composite effect computed from
state-space component scores already produced by the state-space pipeline.
"""

from __future__ import annotations

import math
import random
from typing import Any


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _percentile(sorted_vals: list[float], pct: float) -> float:
    """
    Linear-interpolated percentile for a sorted list.
    pct in [0, 100].
    """
    if not sorted_vals:
        return 0.0
    if pct <= 0:
        return float(sorted_vals[0])
    if pct >= 100:
        return float(sorted_vals[-1])
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(sorted_vals[int(k)])
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return float(d0 + d1)


def bootstrap_state_edge_effect(
    therapeutic_redirection_result: dict[str, Any] | None,
    n_bootstrap: int = 200,
    seed: int = 0,
) -> dict[str, float]:
    """
    Bootstrap the composite state-edge effect.

    The point estimate follows the same locked composite used elsewhere:
      effect = clip(|TR| + 0.3*state_influence + 0.2*transition_avg, 0, 1)

    We generate bootstrap replicates by perturbing each component with Gaussian noise.
    Noise scale is tied to stability_score: lower stability → higher uncertainty.

    Returns:
      {
        "mean": float,
        "ci_lower": float,
        "ci_upper": float,
        "cv": float,
        "confidence": float
      }
    """
    tr = therapeutic_redirection_result or {}
    if not isinstance(tr, dict) or not tr:
        return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "cv": 1.0, "confidence": 0.0}

    def _f(key: str, default: float = 0.0) -> float:
        try:
            v = float(tr.get(key, default))
            return v if math.isfinite(v) else default
        except Exception:
            return default

    tr_val = abs(_f("therapeutic_redirection", 0.0))
    si_val = _f("state_influence_score", 0.0)
    entry_s = _f("entry_score", 0.0)
    persist_s = _f("persistence_score", 0.0)
    recovery_s = _f("recovery_score", 0.0)
    boundary_s = _f("boundary_score", 0.0)
    transition_avg = 0.25 * (entry_s + persist_s + recovery_s + boundary_s)

    stability = _f("stability_score", _f("stability", 0.5))
    stability = _clamp01(stability)

    # Base noise: always some; amplified when stability is low.
    # Tuned to be conservative but not degenerate for stability≈1.
    sigma = 0.02 + (1.0 - stability) * 0.15

    rng = random.Random(seed)
    reps: list[float] = []
    for _ in range(max(10, int(n_bootstrap))):
        tr_b = max(0.0, tr_val + rng.gauss(0.0, sigma))
        si_b = _clamp01(si_val + rng.gauss(0.0, sigma))
        entry_b = _clamp01(entry_s + rng.gauss(0.0, sigma))
        persist_b = _clamp01(persist_s + rng.gauss(0.0, sigma))
        recov_b = _clamp01(recovery_s + rng.gauss(0.0, sigma))
        bound_b = _clamp01(boundary_s + rng.gauss(0.0, sigma))
        trans_avg_b = 0.25 * (entry_b + persist_b + recov_b + bound_b)

        eff = tr_b + 0.3 * si_b + 0.2 * trans_avg_b
        reps.append(_clamp01(eff))

    reps.sort()
    mean = sum(reps) / len(reps) if reps else 0.0
    ci_lower = _percentile(reps, 2.5)
    ci_upper = _percentile(reps, 97.5)

    # CV: std / |mean|
    if len(reps) >= 2:
        var = sum((x - mean) ** 2 for x in reps) / (len(reps) - 1)
        std = math.sqrt(max(var, 0.0))
    else:
        std = 0.0
    cv = std / (abs(mean) + 1e-9)

    # Convert uncertainty into a [0,1] confidence proxy.
    # Lower CV → higher confidence. Clamp so it never goes negative.
    conf = _clamp01(1.0 - min(cv, 1.0))

    # Optional risk downweights (if present)
    marker_score = _f("marker_score", 0.0)
    escape_risk = _f("escape_risk", _f("escape_risk_score", 0.0))
    if marker_score >= 0.3:
        conf *= 0.7
    if escape_risk >= 0.5:
        conf *= 0.7
    conf = _clamp01(conf)

    return {
        "mean": round(float(mean), 6),
        "ci_lower": round(float(ci_lower), 6),
        "ci_upper": round(float(ci_upper), 6),
        "cv": round(float(cv), 6),
        "confidence": round(float(conf), 6),
    }


