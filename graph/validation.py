"""
graph/validation.py — Graph-level quality validation.

Computes:
  - E-value distribution for all edges
  - Demoted edge fraction

All checks here operate at graph level (post-write);
per-edge review lives in orchestrator/scientific_reviewer.py.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class ValidationReport(NamedTuple):
    shd: int | None
    sid: float | None
    n_edges_total: int
    n_edges_demoted: int
    n_low_evalue: int          # E-value < 2.0
    min_evalue: float | None
    mean_evalue: float | None
    passed: bool
    warnings: list[str]
    errors: list[str]


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
    Run graph-level quality checks for a disease.

    Args:
        disease_name: Disease to validate (e.g. "coronary artery disease")

    Returns:
        ValidationReport namedtuple
    """
    from mcp_servers.graph_db_server import query_graph_for_disease

    warnings: list[str] = []
    errors:   list[str] = []

    try:
        result = query_graph_for_disease(disease_name)
        graph_edges = result.get("edges", [])
    except Exception as exc:
        errors.append(f"Graph query failed: {exc}")
        return ValidationReport(
            shd=None, sid=None,
            n_edges_total=0, n_edges_demoted=0, n_low_evalue=0,
            min_evalue=None, mean_evalue=None,
            passed=False, warnings=[], errors=errors,
        )

    active_edges  = [e for e in graph_edges if not e.get("is_demoted", False)]
    demoted_edges = [e for e in graph_edges if e.get("is_demoted", False)]

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
        shd=None,
        sid=None,
        n_edges_total=total,
        n_edges_demoted=len(demoted_edges),
        n_low_evalue=n_low_ev,
        min_evalue=min_ev,
        mean_evalue=ev_summary["mean_evalue"],
        passed=passed,
        warnings=warnings,
        errors=errors,
    )


def validation_report_to_dict(report: ValidationReport) -> dict:
    return report._asdict()
