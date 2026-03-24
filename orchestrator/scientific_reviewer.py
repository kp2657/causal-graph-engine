"""
scientific_reviewer.py — Peer-review gate before any edge enters the graph.

Implements the Scientific Reviewer checklist:
  - Hard-reject (IngestionError): bad data sources, NaN effects, missing PMIDs
  - Soft-warn: E-value < 2.0, missing CIs, single-study support
  - Approves with full audit trail

Called by chief_of_staff.py before dispatching edges to graph_db_server.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).parent.parent))


# -------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------

BLOCKED_SOURCES: frozenset[str] = frozenset({"memory", "unknown", "llm", "placeholder"})

Decision = Literal["APPROVE", "APPROVE_WITH_WARNING", "REJECT"]

# Effect sizes outside this range are suspicious (log-scale HR/OR)
EFFECT_SIZE_REASONABLE_ABS_MAX = 5.0


def review_edge(edge: dict) -> dict:
    """
    Review a single proposed causal edge.

    Args:
        edge: dict matching CausalEdge schema (may be Pydantic .model_dump() output)

    Returns:
        {
            "decision": "APPROVE" | "APPROVE_WITH_WARNING" | "REJECT",
            "warnings": list[str],
            "rejection_reason": str | None,
            "evidence_tier_assigned": str,
        }
    """
    warnings: list[str] = []
    rejection_reason: str | None = None

    data_source   = str(edge.get("data_source") or "").lower()
    effect_size   = edge.get("effect_size")
    evidence_tier = edge.get("evidence_tier", "provisional_virtual")
    method        = edge.get("method", "")
    mr_ivw_beta   = edge.get("mr_ivw")
    mr_egger_p    = edge.get("mr_egger_intercept")
    ci_lower      = edge.get("ci_lower")
    ci_upper      = edge.get("ci_upper")
    e_value       = edge.get("e_value")
    from_node     = edge.get("from_node", "?")
    to_node       = edge.get("to_node", "?")

    # ------------------------------------------------------------------
    # HARD REJECT checks
    # ------------------------------------------------------------------

    # 1. Blocked data source
    if any(blocked in data_source for blocked in BLOCKED_SOURCES):
        rejection_reason = (
            f"data_source '{data_source}' is not a valid scientific reference "
            "(must be PMID, DOI, or API endpoint)"
        )
        return {
            "decision":              "REJECT",
            "warnings":              [],
            "rejection_reason":      rejection_reason,
            "evidence_tier_assigned": evidence_tier,
        }

    # 2. Null/NaN/Inf effect size
    if effect_size is None:
        rejection_reason = f"{from_node}→{to_node}: effect_size is None"
        return {
            "decision":              "REJECT",
            "warnings":              [],
            "rejection_reason":      rejection_reason,
            "evidence_tier_assigned": evidence_tier,
        }
    try:
        es_float = float(effect_size)
    except (TypeError, ValueError):
        rejection_reason = f"{from_node}→{to_node}: effect_size is not numeric ({effect_size!r})"
        return {
            "decision":              "REJECT",
            "warnings":              [],
            "rejection_reason":      rejection_reason,
            "evidence_tier_assigned": evidence_tier,
        }

    if not math.isfinite(es_float):
        rejection_reason = f"{from_node}→{to_node}: effect_size is NaN or Inf"
        return {
            "decision":              "REJECT",
            "warnings":              [],
            "rejection_reason":      rejection_reason,
            "evidence_tier_assigned": evidence_tier,
        }

    # 3. Tier1 edge must have PMID/DOI
    if evidence_tier == "Tier1_Interventional":
        has_citation = (
            "10." in data_source           # DOI
            or "pmid" in data_source       # PMID
            or data_source.isdigit()       # bare PMID number
            or "pubmed" in data_source
            or "ncbi" in data_source
            or "biorxiv" in data_source
            or "pmc" in data_source
        )
        if not has_citation:
            rejection_reason = (
                f"{from_node}→{to_node}: Tier1 edge requires PMID or DOI in data_source "
                f"(got: '{data_source}')"
            )
            return {
                "decision":              "REJECT",
                "warnings":              [],
                "rejection_reason":      rejection_reason,
                "evidence_tier_assigned": evidence_tier,
            }

    # 4. Tier1 MR edge with F-statistic < 10
    f_stat = edge.get("f_statistic")
    if (
        evidence_tier == "Tier1_Interventional"
        and method == "mr"
        and f_stat is not None
        and float(f_stat) < 10
    ):
        rejection_reason = (
            f"{from_node}→{to_node}: F-statistic={f_stat} < 10 with Tier1 claim — "
            "weak instruments; demote to Tier3 or provide stronger instruments"
        )
        return {
            "decision":              "REJECT",
            "warnings":              [],
            "rejection_reason":      rejection_reason,
            "evidence_tier_assigned": evidence_tier,
        }

    # ------------------------------------------------------------------
    # SOFT WARN checks
    # ------------------------------------------------------------------

    # E-value < 2.0
    if e_value is not None and float(e_value) < 2.0:
        warnings.append(
            f"E-value={e_value:.2f} < 2.0: potential unmeasured confounding "
            "(VanderWeele & Ding 2017)"
        )

    # Missing confidence intervals
    if ci_lower is None or ci_upper is None:
        warnings.append(f"{from_node}→{to_node}: confidence interval not reported")

    # MR pleiotropy signal
    if method == "mr" and mr_egger_p is not None and float(mr_egger_p) < 0.05:
        warnings.append(
            f"MR-Egger intercept p={mr_egger_p:.3f} < 0.05 — potential horizontal pleiotropy"
        )

    # Effect size suspiciously large
    if abs(es_float) > EFFECT_SIZE_REASONABLE_ABS_MAX:
        warnings.append(
            f"effect_size={es_float:.3f} is unusually large (|β| > {EFFECT_SIZE_REASONABLE_ABS_MAX}); "
            "verify scale (should be log-HR or log-OR)"
        )

    # Tier3/virtual: note as provisional
    if evidence_tier in ("Tier3_Provisional", "provisional_virtual", "moderate_transferred", "moderate_grn"):
        warnings.append(
            f"Evidence tier '{evidence_tier}': single-source or in silico — "
            "replication required before clinical translation"
        )

    # ------------------------------------------------------------------
    # Final decision
    # ------------------------------------------------------------------
    if warnings:
        decision: Decision = "APPROVE_WITH_WARNING"
    else:
        decision = "APPROVE"

    return {
        "decision":              decision,
        "warnings":              warnings,
        "rejection_reason":      None,
        "evidence_tier_assigned": evidence_tier,
    }


def review_batch(edges: list[dict]) -> dict:
    """
    Review a list of proposed edges and return a summary.

    Args:
        edges: List of CausalEdge-compatible dicts

    Returns:
        {
            "n_approved": int,
            "n_approved_with_warning": int,
            "n_rejected": int,
            "approved_edges": list[dict],         # edges that passed
            "rejected_edges": list[dict],         # edges with rejection reason
            "all_reviews": list[dict],            # one review record per edge
        }
    """
    approved: list[dict] = []
    rejected: list[dict] = []
    all_reviews: list[dict] = []
    n_approved = 0
    n_warned = 0
    n_rejected = 0

    for edge in edges:
        review = review_edge(edge)
        all_reviews.append({"edge": edge, "review": review})

        if review["decision"] == "REJECT":
            n_rejected += 1
            rejected.append({"edge": edge, "reason": review["rejection_reason"]})
        elif review["decision"] == "APPROVE_WITH_WARNING":
            n_warned += 1
            # Attach warnings to edge copy before approving
            edge_copy = dict(edge)
            edge_copy.setdefault("review_warnings", []).extend(review["warnings"])
            approved.append(edge_copy)
        else:
            n_approved += 1
            approved.append(edge)

    return {
        "n_approved":               n_approved,
        "n_approved_with_warning":  n_warned,
        "n_rejected":               n_rejected,
        "approved_edges":           approved,
        "rejected_edges":           rejected,
        "all_reviews":              all_reviews,
    }
