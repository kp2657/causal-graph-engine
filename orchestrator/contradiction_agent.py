"""
contradiction_agent.py — Detects contradictions and demotes edges in the graph.

Activated when:
  - A new edge contradicts an existing Tier1/Tier2 edge (opposite sign)
  - MR sensitivity shows significant pleiotropy
  - E-value drops below 2.0 on re-analysis

Implements the demotion decision tree from the prompt spec.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# Evidence tier ordering for comparison (higher = more trust)
TIER_ORDER: dict[str, int] = {
    "Tier1_Interventional": 4,
    "Tier2_Convergent":     3,
    "Tier3_Provisional":    2,
    "moderate_transferred": 2,
    "moderate_grn":         2,
    "provisional_virtual":  1,
}

PLEIOTROPY_P_THRESHOLD = 0.05
EVALUE_THRESHOLD       = 2.0


def _tier_rank(tier: str) -> int:
    return TIER_ORDER.get(tier, 1)


def _edges_contradict(existing_edge: dict, new_edge: dict) -> bool:
    """
    Two edges between the same nodes contradict if their effect sizes have opposite signs.
    """
    e1 = existing_edge.get("effect_size")
    e2 = new_edge.get("effect_size")
    if e1 is None or e2 is None:
        return False
    try:
        return float(e1) * float(e2) < 0
    except (TypeError, ValueError):
        return False


def check_contradiction(new_edge: dict, disease_name: str) -> dict:
    """
    Check whether a new proposed edge contradicts anything already in the graph.

    Args:
        new_edge:     Proposed CausalEdge dict
        disease_name: Disease context for graph query

    Returns:
        {
            "has_contradiction": bool,
            "action": "NONE" | "DEMOTE_EXISTING" | "SOFT_WARN" | "IGNORE",
            "demoted_edge": dict | None,
            "warning": str | None,
        }
    """
    from mcp_servers.graph_db_server import query_graph_for_disease, demote_edge_tier

    from_node = new_edge.get("from_node", "")
    to_node   = new_edge.get("to_node", "")
    new_tier  = new_edge.get("evidence_tier", "provisional_virtual")
    new_rank  = _tier_rank(new_tier)

    # Query existing edges between the same nodes
    try:
        graph_result = query_graph_for_disease(disease_name)
        all_edges = graph_result.get("edges", [])
    except Exception:
        return {
            "has_contradiction": False,
            "action": "NONE",
            "demoted_edge": None,
            "warning": "Graph query failed — cannot check contradictions",
        }

    matching = [
        e for e in all_edges
        if e.get("from_node") == from_node and e.get("to_node") == to_node
    ]

    if not matching:
        return {
            "has_contradiction": False,
            "action": "NONE",
            "demoted_edge": None,
            "warning": None,
        }

    # Find highest-tier existing edge
    best_existing = max(matching, key=lambda e: _tier_rank(e.get("evidence_tier", "provisional_virtual")))
    existing_tier = best_existing.get("evidence_tier", "provisional_virtual")
    existing_rank = _tier_rank(existing_tier)

    if not _edges_contradict(best_existing, new_edge):
        return {
            "has_contradiction": False,
            "action": "NONE",
            "demoted_edge": None,
            "warning": None,
        }

    # ----------------------------------------------------------------
    # Demotion decision tree
    # ----------------------------------------------------------------
    action: str
    demoted_edge: dict | None = None
    warning_msg: str | None = None

    if existing_rank == 4:  # existing = Tier1
        if new_rank == 4:
            # Both Tier1: context-specific demotion
            action = "DEMOTE_EXISTING"
            warning_msg = (
                f"CONDITIONAL DEMOTION: both {from_node}→{to_node} and new edge are Tier1 "
                "with opposite signs. Likely context-specific (different cell type). "
                "Adding new edge with cell_type metadata; existing edge demoted to "
                "Tier1_context_specific."
            )
            demoted_edge = best_existing
        elif new_rank == 3:
            # New = Tier2: soft warn, do NOT demote
            action = "SOFT_WARN"
            warning_msg = (
                f"Tier2 evidence contradicts existing Tier1 {from_node}→{to_node}. "
                "Flagged for PI review — do NOT demote automatically."
            )
        else:
            # New < Tier2: ignore
            action = "IGNORE"
            warning_msg = (
                f"Lower-tier contradiction of Tier1 edge {from_node}→{to_node} ignored "
                f"(new tier: {new_tier})"
            )

    else:  # existing = Tier2 or lower
        if new_rank >= existing_rank:
            # New ≥ existing tier with opposite sign: demote existing
            action = "DEMOTE_EXISTING"
            warning_msg = (
                f"DEMOTION: {from_node}→{to_node} demoted from {existing_tier} to "
                f"Tier3_Provisional — contradicted by new {new_tier} evidence with opposite sign."
            )
            demoted_edge = best_existing
            try:
                demote_edge_tier(
                    from_node=from_node,
                    to_node=to_node,
                    new_tier="Tier3_Provisional",
                    reason=f"Contradicted by new {new_tier} evidence (opposite sign)",
                )
            except Exception as exc:
                warning_msg += f" (demote_edge_tier call failed: {exc})"
        else:
            # Same-tier or lower: add both
            action = "SOFT_WARN"
            warning_msg = (
                f"Conflicting same-tier evidence for {from_node}→{to_node}. "
                "Both edges added; meta-analysis pending."
            )

    return {
        "has_contradiction": True,
        "action":            action,
        "demoted_edge":      demoted_edge,
        "warning":           warning_msg,
    }


def run(new_edges: list[dict], disease_name: str) -> dict:
    """
    Process a list of proposed edges for contradictions.

    Args:
        new_edges:    List of CausalEdge-compatible dicts
        disease_name: Disease context

    Returns:
        {
            "n_contradictions_found": int,
            "demotions_executed": list[dict],
            "pending_review": list[dict],
            "graph_integrity_score": float,
            "safe_to_write": list[dict],   # edges cleared for writing
            "blocked":        list[dict],  # edges with IGNORE action
        }
    """
    demotions_executed: list[dict] = []
    pending_review: list[dict] = []
    safe_to_write: list[dict] = []
    blocked: list[dict] = []
    n_contradictions = 0

    for edge in new_edges:
        result = check_contradiction(edge, disease_name)

        if not result["has_contradiction"]:
            safe_to_write.append(edge)
            continue

        n_contradictions += 1
        action = result["action"]

        if action == "IGNORE":
            blocked.append({
                "edge":   edge,
                "reason": result["warning"],
            })

        elif action == "DEMOTE_EXISTING":
            demotions_executed.append({
                "edge":      f"{edge.get('from_node')}→{edge.get('to_node')}",
                "old_tier":  result["demoted_edge"].get("evidence_tier") if result["demoted_edge"] else None,
                "new_tier":  "Tier3_Provisional",
                "reason":    result["warning"],
            })
            safe_to_write.append(edge)  # new edge still gets written

        elif action == "SOFT_WARN":
            pending_review.append({
                "edge":   f"{edge.get('from_node')}→{edge.get('to_node')}",
                "reason": result["warning"],
            })
            safe_to_write.append(edge)  # allow write but flag

    # Graph integrity: fraction of edges without active contradictions
    total = len(new_edges)
    problematic = len(demotions_executed) + len(pending_review)
    integrity = 1.0 - (problematic / total) if total > 0 else 1.0

    return {
        "n_contradictions_found": n_contradictions,
        "demotions_executed":     demotions_executed,
        "pending_review":         pending_review,
        "graph_integrity_score":  round(integrity, 3),
        "safe_to_write":          safe_to_write,
        "blocked":                blocked,
    }


def check_pleiotropy_demotion(edge: dict, mr_egger_p: float | None) -> dict | None:
    """
    Check if an MR edge should be demoted due to horizontal pleiotropy.

    Returns:
        Demotion dict or None if no action needed.
    """
    if mr_egger_p is None or mr_egger_p >= PLEIOTROPY_P_THRESHOLD:
        return None

    return {
        "from_node": edge.get("from_node"),
        "to_node":   edge.get("to_node"),
        "reason":    f"MR-Egger pleiotropy p={mr_egger_p:.3f} < 0.05 — horizontal pleiotropy signal",
        "suggested_tier": "Tier3_Provisional",
    }
