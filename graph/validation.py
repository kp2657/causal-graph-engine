"""
graph/validation.py — Graph-level quality validation.

Computes:
  - SID (Structural Intervention Distance) from reference anchor graph
  - SHD (Structural Hamming Distance) from reference anchor graph
  - E-value distribution for all edges
  - Anchor edge recovery rate

All checks here operate at graph level (post-write);
per-edge review lives in orchestrator/scientific_reviewer.py.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from graph.schema import ANCHOR_EDGES


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class ValidationReport(NamedTuple):
    anchor_recovery_rate: float
    n_anchors_recovered: int
    n_anchors_total: int
    missing_anchors: list[str]
    shd: int
    sid: float | None          # None if SID cannot be computed (requires DAG)
    n_edges_total: int
    n_edges_demoted: int
    n_low_evalue: int          # E-value < 2.0
    min_evalue: float | None
    mean_evalue: float | None
    passed: bool               # True if all hard checks pass
    warnings: list[str]
    errors: list[str]


# ---------------------------------------------------------------------------
# Anchor recovery
# ---------------------------------------------------------------------------

def check_anchor_recovery(graph_edges: list[dict]) -> dict:
    """
    Check how many of the 12 required anchor edges are present in the graph.

    Args:
        graph_edges: List of edge dicts from query_graph_for_disease

    Returns:
        {recovery_rate, n_recovered, n_total, missing, recovered}
    """
    predicted_pairs: set[tuple[str, str]] = set()
    for e in graph_edges:
        if not e.get("is_demoted", False):
            predicted_pairs.add((e["from_node"], e["to_node"]))

    recovered: list[str] = []
    missing:   list[str] = []

    for anchor in ANCHOR_EDGES:
        key_from = anchor.get("from_node") or anchor.get("from", "")
        key_to   = anchor.get("to_node")   or anchor.get("to", "")
        label    = f"{key_from}→{key_to}"

        # Try exact match and chip-suffix variant
        if (key_from, key_to) in predicted_pairs or (f"{key_from}_chip", key_to) in predicted_pairs:
            recovered.append(label)
        else:
            missing.append(label)

    n_total     = len(ANCHOR_EDGES)
    n_recovered = len(recovered)
    rate        = n_recovered / n_total if n_total > 0 else 0.0

    return {
        "recovery_rate": rate,
        "n_recovered":   n_recovered,
        "n_total":       n_total,
        "recovered":     recovered,
        "missing":       missing,
    }


# ---------------------------------------------------------------------------
# SHD / SID
# ---------------------------------------------------------------------------

def compute_shd(
    predicted_edges: list[dict],
    reference_edges: list[dict] | None = None,
) -> dict:
    """
    Compute Structural Hamming Distance between predicted and reference graphs.

    SHD = |missing edges| + |extra edges| (treating undirected skeleton).

    Args:
        predicted_edges: List of {from_node, to_node} dicts
        reference_edges: Defaults to ANCHOR_EDGES from schema

    Returns:
        {shd, missing_edges, extra_edges}
    """
    if reference_edges is None:
        reference_edges = ANCHOR_EDGES

    def _edge_set(edges: list[dict]) -> set[tuple[str, str]]:
        result: set[tuple[str, str]] = set()
        for e in edges:
            fn = e.get("from_node") or e.get("from", "")
            tn = e.get("to_node")   or e.get("to", "")
            result.add((fn, tn))
            # Normalise chip suffix
            result.add((fn.replace("_chip", ""), tn))
        return result

    ref_set  = _edge_set(reference_edges)
    pred_set = _edge_set(predicted_edges)

    missing_from_pred = ref_set - pred_set
    extra_in_pred     = pred_set - ref_set

    shd = len(missing_from_pred) + len(extra_in_pred)

    return {
        "shd":            shd,
        "missing_edges":  [f"{f}→{t}" for f, t in sorted(missing_from_pred)],
        "extra_edges":    [f"{f}→{t}" for f, t in sorted(extra_in_pred)],
    }


def compute_sid_approximation(
    predicted_edges: list[dict],
    reference_edges: list[dict] | None = None,
) -> float:
    """
    Approximate SID as the fraction of reference edges correctly oriented.

    True SID requires full DAG structure; this gives a conservative lower bound.

    Returns:
        Normalised [0, 1] score (1 = perfect, 0 = all wrong)
    """
    if reference_edges is None:
        reference_edges = ANCHOR_EDGES

    if not reference_edges:
        return 1.0

    pred_directed: set[tuple[str, str]] = set()
    for e in predicted_edges:
        fn = (e.get("from_node") or e.get("from", "")).replace("_chip", "")
        tn = e.get("to_node")   or e.get("to", "")
        pred_directed.add((fn, tn))

    correct = 0
    for anchor in reference_edges:
        fn = (anchor.get("from_node") or anchor.get("from", "")).replace("_chip", "")
        tn = anchor.get("to_node")    or anchor.get("to", "")
        if (fn, tn) in pred_directed:
            correct += 1

    return correct / len(reference_edges)


# ---------------------------------------------------------------------------
# E-value summary
# ---------------------------------------------------------------------------

def summarize_evalues(graph_edges: list[dict]) -> dict:
    """
    Compute E-value distribution statistics across all edges.

    Returns:
        {n_edges, n_low_evalue, min_evalue, mean_evalue, flagged}
    """
    evalues: list[float] = []
    flagged: list[str] = []

    for e in graph_edges:
        ev = e.get("e_value")
        if ev is None:
            continue
        try:
            ev_f = float(ev)
        except (TypeError, ValueError):
            continue
        evalues.append(ev_f)
        if ev_f < 2.0:
            flagged.append(f"{e.get('from_node', '?')}→{e.get('to_node', '?')} (E={ev_f:.2f})")

    return {
        "n_edges":       len(graph_edges),
        "n_with_evalue": len(evalues),
        "n_low_evalue":  len(flagged),
        "min_evalue":    min(evalues) if evalues else None,
        "mean_evalue":   sum(evalues) / len(evalues) if evalues else None,
        "flagged":       flagged,
    }


# ---------------------------------------------------------------------------
# Full validation run
# ---------------------------------------------------------------------------

def validate_graph(disease_name: str) -> ValidationReport:
    """
    Run all graph-level quality checks for a disease.

    Args:
        disease_name: Disease to validate (e.g. "coronary artery disease")

    Returns:
        ValidationReport namedtuple
    """
    from mcp_servers.graph_db_server import query_graph_for_disease

    warnings: list[str] = []
    errors:   list[str] = []

    # Load graph
    try:
        result = query_graph_for_disease(disease_name)
        graph_edges = result.get("edges", [])
    except Exception as exc:
        errors.append(f"Graph query failed: {exc}")
        return ValidationReport(
            anchor_recovery_rate=0.0, n_anchors_recovered=0,
            n_anchors_total=len(ANCHOR_EDGES), missing_anchors=[],
            shd=len(ANCHOR_EDGES), sid=None,
            n_edges_total=0, n_edges_demoted=0, n_low_evalue=0,
            min_evalue=None, mean_evalue=None,
            passed=False, warnings=[], errors=errors,
        )

    active_edges   = [e for e in graph_edges if not e.get("is_demoted", False)]
    demoted_edges  = [e for e in graph_edges if e.get("is_demoted", False)]

    # Anchor recovery
    anchor_result = check_anchor_recovery(active_edges)
    recovery_rate = anchor_result["recovery_rate"]
    if recovery_rate < 0.80:
        errors.append(
            f"Anchor recovery {recovery_rate:.0%} < 80%: missing {anchor_result['missing']}"
        )

    # SHD
    shd_result = compute_shd(active_edges)
    shd = shd_result["shd"]
    if shd > 4:
        warnings.append(f"SHD={shd} is high (> 4 edges different from reference)")

    # SID approximation
    sid = compute_sid_approximation(active_edges)
    if sid < 0.70:
        warnings.append(f"SID approximation={sid:.2f} < 0.70 — many edges incorrectly oriented")

    # E-value summary
    ev_summary = summarize_evalues(active_edges)
    n_low_ev = ev_summary["n_low_evalue"]
    if n_low_ev > 0:
        warnings.append(
            f"{n_low_ev} edge(s) with E-value < 2.0 — potential confounding: "
            f"{ev_summary['flagged'][:3]}"
        )
    min_ev = ev_summary["min_evalue"]
    if min_ev is not None and min_ev < 2.0:
        warnings.append(f"Minimum E-value = {min_ev:.2f} < 2.0")

    # Demoted edge fraction
    total = len(graph_edges)
    if total > 0 and len(demoted_edges) / total > 0.30:
        warnings.append(
            f"{len(demoted_edges)}/{total} edges are demoted — "
            "high contradiction rate; review pipeline inputs"
        )

    passed = len(errors) == 0

    return ValidationReport(
        anchor_recovery_rate  = recovery_rate,
        n_anchors_recovered   = anchor_result["n_recovered"],
        n_anchors_total       = anchor_result["n_total"],
        missing_anchors       = anchor_result["missing"],
        shd                   = shd,
        sid                   = sid,
        n_edges_total         = total,
        n_edges_demoted       = len(demoted_edges),
        n_low_evalue          = n_low_ev,
        min_evalue            = min_ev,
        mean_evalue           = ev_summary["mean_evalue"],
        passed                = passed,
        warnings              = warnings,
        errors                = errors,
    )


def validation_report_to_dict(report: ValidationReport) -> dict:
    return report._asdict()
