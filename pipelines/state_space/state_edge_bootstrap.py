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


def bootstrap_state_edge_effect_from_transitions(
    T_baseline,
    disease_axis_score: float,
    path_idxs: list[int],
    healthy_idxs: list[int],
    n_cells_per_state: list[int] | None = None,
    alpha_scale: float = 1.0,
    n_bootstrap: int = 300,
    seed: int = 0,
    resample_only_path_rows: bool = True,
) -> dict[str, float]:
    """
    Transition-edge bootstrap (Method B).

    Holds state definitions fixed and resamples each row of the transition matrix
    using a Dirichlet bootstrap:
        T_row^(b) ~ Dirichlet(alpha_row)
        alpha_row = alpha_scale * n_cells_i * T_row

    Then computes the net improvement toward healthy basins under a proxy gene
    perturbation with strength = disease_axis_score.

    Returns same schema as bootstrap_state_edge_effect.
    """
    if T_baseline is None or not path_idxs or not healthy_idxs:
        return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "cv": 1.0, "confidence": 0.0}

    # Convert T to a python list-of-lists (supports numpy arrays)
    try:
        T0 = T_baseline.tolist()
    except Exception:
        T0 = T_baseline
    if not isinstance(T0, list) or not T0 or not isinstance(T0[0], list):
        return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "cv": 1.0, "confidence": 0.0}

    n_states = len(T0)
    n_cells = list(n_cells_per_state) if n_cells_per_state else [50 for _ in range(n_states)]
    if len(n_cells) != n_states:
        n_cells = [50 for _ in range(n_states)]

    # Precompute which indices are valid
    path_i = [i for i in path_idxs if 0 <= i < n_states]
    healthy_j = [j for j in healthy_idxs if 0 <= j < n_states]
    if not path_i or not healthy_j:
        return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "cv": 1.0, "confidence": 0.0}

    rng = random.Random(seed)

    def _dirichlet_row(probs: list[float], alpha_total: float, exclude_idx: int | None = None) -> list[float]:
        # Sample gamma variates for each component; optionally force exclude_idx to 0.
        xs: list[float] = []
        for j, p in enumerate(probs):
            if exclude_idx is not None and j == exclude_idx:
                xs.append(0.0)
                continue
            a = max(alpha_total * max(p, 0.0), 1e-6)
            xs.append(rng.gammavariate(a, 1.0))
        s = sum(xs)
        if s <= 0:
            # fallback to original row (with exclude_idx zeroed)
            out = [max(float(p), 0.0) for p in probs]
            if exclude_idx is not None and 0 <= exclude_idx < len(out):
                out[exclude_idx] = 0.0
            tot = sum(out)
            return [v / tot for v in out] if tot > 0 else [0.0 for _ in out]
        return [x / s for x in xs]

    def _perturb_transition_matrix_rowwise(T_base: list[list[float]], p_loading: float) -> list[list[float]]:
        # Minimal version of perturb_transition_matrix for pure python T.
        # Mirrors the 50% per-edge cap and row renormalization.
        out = [row[:] for row in T_base]
        effect = abs(1.0) * max(float(p_loading), 0.0)  # gene_beta proxy = 1.0
        for i in path_i:
            base_row = T_base[i]
            new_row = out[i]
            deltas = [0.0 for _ in range(n_states)]
            for j in range(n_states):
                if i == j:
                    continue
                baseline_edge = float(base_row[j] or 0.0)
                if baseline_edge <= 0:
                    continue
                max_delta = 0.50 * baseline_edge
                delta_mag = min(effect * baseline_edge, max_delta)
                if j in healthy_j:
                    deltas[j] = +delta_mag
                elif j in path_i and j != i:
                    deltas[j] = -delta_mag
            for j in range(n_states):
                new_row[j] = max(float(base_row[j]) + deltas[j], 0.0)
            rs = sum(new_row)
            if rs > 0:
                out[i] = [v / rs for v in new_row]
        return out

    reps: list[float] = []
    for _ in range(max(20, int(n_bootstrap))):
        # Resample each row of T with Dirichlet; exclude diagonal to keep no-self-loop convention.
        T_b: list[list[float]] = []
        for i, row in enumerate(T0):
            # Optimization: only resample pathological source rows; other rows stay fixed.
            if resample_only_path_rows and i not in path_i:
                # Keep fixed, but enforce "no self-loop" convention.
                fixed = [max(float(v or 0.0), 0.0) for v in row]
                if 0 <= i < len(fixed):
                    fixed[i] = 0.0
                tot = sum(fixed)
                T_b.append([v / tot for v in fixed] if tot > 0 else [0.0 for _ in fixed])
                continue

            row_probs = [max(float(v or 0.0), 0.0) for v in row]
            # ensure row normalized (excluding diagonal handled after)
            tot = sum(row_probs)
            if tot > 0:
                row_probs = [v / tot for v in row_probs]
            alpha_total = alpha_scale * max(int(n_cells[i] or 1), 1)
            T_b.append(_dirichlet_row(row_probs, alpha_total=alpha_total, exclude_idx=i))

        T_pert = _perturb_transition_matrix_rowwise(T_b, p_loading=disease_axis_score)

        # Net improvement = sum_{i in path, j in healthy} (T_pert - T_base)
        delta_sum = 0.0
        for i in path_i:
            for j in healthy_j:
                delta_sum += float(T_pert[i][j]) - float(T_b[i][j])
        raw = max(delta_sum, 0.0)

        # Map to [0,1] smoothly to avoid scale dependence on basin sizes
        score = 1.0 - math.exp(-5.0 * raw)
        reps.append(_clamp01(score))

    reps.sort()
    mean = sum(reps) / len(reps) if reps else 0.0
    ci_lower = _percentile(reps, 2.5)
    ci_upper = _percentile(reps, 97.5)

    if len(reps) >= 2:
        var = sum((x - mean) ** 2 for x in reps) / (len(reps) - 1)
        std = math.sqrt(max(var, 0.0))
    else:
        std = 0.0
    cv = std / (abs(mean) + 1e-9)
    conf = _clamp01(1.0 - min(cv, 1.0))

    return {
        "mean": round(float(mean), 6),
        "ci_lower": round(float(ci_lower), 6),
        "ci_upper": round(float(ci_upper), 6),
        "cv": round(float(cv), 6),
        "confidence": round(float(conf), 6),
    }

