"""
sensitivity_analysis.py — E-value and sensitivity analysis for causal edges.

Implements:
  1. E-value computation (VanderWeele & Ding 2017) — already in graph_db_server
  2. Batch sensitivity analysis for all edges in a disease graph
  3. Edge demotion recommendations based on E-value thresholds
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def run_batch_evalue(edges: list[dict]) -> list[dict]:
    """
    Compute E-values for a list of causal edges.

    Args:
        edges: List of edge dicts with {from_node, to_node, effect_size, se}

    Returns:
        List of edges with e_value and recommendation fields added
    """
    from mcp_servers.graph_db_server import run_evalue_check

    results = []
    for edge in edges:
        effect = edge.get("effect_size", 0.0)
        se = edge.get("se", 0.1)
        evalue_result = run_evalue_check(effect_size=effect, se=se)
        results.append({
            **edge,
            "e_value":        evalue_result.get("e_value"),
            "interpretation": evalue_result.get("interpretation"),
            "recommendation": evalue_result.get("recommendation"),
        })

    return results


def flag_low_evalue_edges(edges: list[dict], threshold: float = 2.0) -> dict:
    """
    Identify edges with E-value below threshold (potentially confounded).

    Args:
        edges:     List of causal edges with e_value fields
        threshold: E-value threshold; < 2.0 = potentially confounded

    Returns:
        {
            "flagged": list of flagged edges,
            "clear":   list of edges meeting threshold,
            "n_flagged": int,
            "n_clear": int,
        }
    """
    flagged = []
    clear = []
    for edge in edges:
        ev = edge.get("e_value")
        if ev is None or ev < threshold:
            flagged.append(edge)
        else:
            clear.append(edge)

    return {
        "flagged":   flagged,
        "clear":     clear,
        "n_flagged": len(flagged),
        "n_clear":   len(clear),
        "threshold": threshold,
    }


def generate_demotion_recommendations(flagged_edges: list[dict]) -> list[dict]:
    """
    Generate edge demotion recommendations for flagged low-E-value edges.
    Each recommendation can be passed to graph_db_server.demote_edge_tier.
    """
    recommendations = []
    for edge in flagged_edges:
        ev = edge.get("e_value")
        if ev is not None and ev < 2.0:
            tier = "Tier3_Provisional"
        else:
            tier = "Tier2_Convergent"

        recommendations.append({
            "from_node":              edge.get("from_node"),
            "to_node":                edge.get("to_node"),
            "new_tier":               tier,
            "reason":                 f"E-value = {ev}; below threshold for confounding robustness",
            "contradicting_evidence": f"E-value < 2.0 (VanderWeele & Ding 2017); requires interventional validation",
        })

    return recommendations
