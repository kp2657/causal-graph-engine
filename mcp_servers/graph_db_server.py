"""
graph_db_server.py — MCP server for the Kùzu causal graph.

All agents interact with the graph through this server.
Stubs return schema-valid mocked data; real Kùzu calls are wired in
for write_causal_edges and query_graph_for_disease.

Run standalone:  python mcp_servers/graph_db_server.py

Architecture note: tool functions are plain callables decorated with @_tool
only when fastmcp is available. Tests import them directly as module-level functions.
"""
from __future__ import annotations

import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

try:
    import fastmcp
    mcp = fastmcp.FastMCP("graph-db-server")
    _tool = mcp.tool()   # decorator
except ImportError:
    # Fallback: @_tool is a no-op — functions remain importable for unit tests
    def _tool(fn=None, **_kwargs):
        return fn if fn is not None else (lambda f: f)
    mcp = None

from graph.schema import ANCHOR_EDGES

_EXPLICIT_DB_PATH = os.getenv("GRAPH_DB_PATH")


def _slug(text: str) -> str:
    s = (text or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"


def _default_db_path_for_key(key: str) -> str:
    """
    Default per-disease DB path when GRAPH_DB_PATH is not set.

    We intentionally do NOT use a single shared graph.kuzu here because this MCP
    server can be long-lived and would otherwise lock the file and/or cause
    concurrent disease runs (AMD vs CAD) to conflict.
    """
    return str(Path("./data") / f"graph_{_slug(key)}.kuzu")


# ---------------------------------------------------------------------------
# Helper: shared DB connection (lazy, process-level singleton)
# ---------------------------------------------------------------------------

_db = None  # GraphDB | None (single DB when GRAPH_DB_PATH is explicitly set)
_db_by_path: dict[str, Any] = {}  # path -> GraphDB (per-disease DBs)

def _get_db_for_key(key: str):
    """
    Return a GraphDB connection.

    - If GRAPH_DB_PATH is set, use a single shared DB for all requests.
    - Otherwise, use a per-disease DB path derived from `key`, caching a
      GraphDB connection per path.
    """
    global _db
    if _EXPLICIT_DB_PATH:
        if _db is None:
            from graph.db import GraphDB
            _db = GraphDB(_EXPLICIT_DB_PATH)
        return _db

    path = _default_db_path_for_key(key)
    db = _db_by_path.get(path)
    if db is None:
        from graph.db import GraphDB
        db = GraphDB(path)
        _db_by_path[path] = db
    return db


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@_tool
def write_causal_edges(edges: list[dict[str, Any]], disease: str) -> dict:
    """
    Validate and write a batch of causal edges to the graph.

    Each edge dict must match the CausalEdge schema (models/evidence.py).
    Edges failing hard-block conditions are rejected and reported.

    Returns:
        {
            "written": int,
            "rejected": int,
            "disease": str,
            "errors": list[str]
        }
    """
    # `disease` is the natural key we have for sharding when GRAPH_DB_PATH is unset.
    # It is typically a disease name (e.g., "AMD") or a full disease string.
    db = _get_db_for_key(disease)
    errors: list[str] = []
    written_edges = []

    for raw in edges:
        try:
            from graph.ingestion import ingest_edge
            edge = ingest_edge(db, raw)
            written_edges.append(edge)
        except Exception as e:
            # graph.ingestion raises IngestionError for hard blocks; avoid importing it
            # at module import time by treating it as a normal Exception here.
            errors.append(str(e))

    return {
        "written": len(written_edges),
        "rejected": len(errors),
        "disease": disease,
        "errors": errors,
    }


@_tool
def query_graph_for_disease(disease_id: str) -> dict:
    """
    Retrieve all active (non-demoted) causal edges for a disease.

    Returns:
        {
            "disease_id": str,
            "edge_count": int,
            "edges": list[dict]
        }
    """
    # For reads, we only have the disease identifier. Use it as the shard key.
    db = _get_db_for_key(disease_id)
    edges = db.query_disease_edges(disease_id)
    active = [e for e in edges if not e.get("is_demoted")]
    return {
        "disease_id": disease_id,
        "edge_count": len(active),
        "edges": active,
    }


@_tool
def demote_edge_tier(
    from_node: str,
    to_node: str,
    new_tier: str,
    reason: str,
    contradicting_evidence: str,
) -> dict:
    """
    Demote a graph edge when contradicting evidence emerges.
    Called by the Contradiction Agent. Preserves history via audit log.

    Returns:
        {"success": bool, "edge": str, "reason": str}
    """
    # We don't receive a disease argument here, but `to_node` is the DiseaseTrait id.
    # Use it as the shard key when GRAPH_DB_PATH is unset.
    db = _get_db_for_key(to_node)
    full_reason = f"[{new_tier}] {reason} | Evidence: {contradicting_evidence}"
    db.demote_edge(from_node, to_node, full_reason)
    return {
        "success": True,
        "edge": f"{from_node} → {to_node}",
        "reason": full_reason,
    }


@_tool
def compute_sid_metric(predicted_edges: list[dict], reference_edges: list[dict]) -> dict:
    """
    Graph agreement metric aligned with `validate_graph` (not full causal-learn SID).

    True **Structural Intervention Distance** between DAGs requires interventional
    distributions; we instead report the same **orientation-overlap score** as
    `graph.validation.compute_sid_approximation` (fraction of reference directed
    edges recovered). This avoids reporting a misleading constant zero.

    If `reference_edges` is empty, anchor edges from `graph.schema.ANCHOR_EDGES` are used.

    Returns:
        {
          "sid": float in [0, 1],
          "metric": "orientation_overlap",
          "n_predicted": int,
          "n_reference": int,
          "note": str,
        }
    """
    from graph.schema import ANCHOR_EDGES
    from graph.validation import compute_sid_approximation

    ref = list(reference_edges) if reference_edges else list(ANCHOR_EDGES)
    sid_score = float(compute_sid_approximation(predicted_edges, ref))
    return {
        "sid": sid_score,
        "metric": "orientation_overlap",
        "n_predicted": len(predicted_edges),
        "n_reference": len(ref),
        "note": (
            "Same orientation-overlap score as validate_graph (not interventional SID "
            "from causal-learn)."
        ),
    }


@_tool
def compute_shd_metric(predicted_edges: list[dict], reference_edges: list[dict]) -> dict:
    """
    Compute Structural Hamming Distance (SHD) between predicted and reference graphs.
    SHD counts missing + extra + reversed edges.

    STUB — returns mocked SHD. Wire in causal-learn SHD when pipelines are ready.
    """
    pred_set = {(e["from_node"], e["to_node"]) for e in predicted_edges}
    ref_set  = {(e["from"], e["to"]) for e in reference_edges}
    missing = len(ref_set - pred_set)
    extra   = len(pred_set - ref_set)
    shd = missing + extra
    return {
        "shd": shd,
        "missing_edges": missing,
        "extra_edges": extra,
        "n_predicted": len(predicted_edges),
        "n_reference": len(reference_edges),
    }


@_tool
def run_anchor_edge_validation(predicted_edges: list[dict]) -> dict:
    """
    Check whether known ground-truth causal edges (anchor edges) are recovered.

    Returns:
        {
            "total_anchors": int,
            "recovered": int,
            "recovery_rate": float,
            "missing": list[str]
        }
    """
    pred_set = {(e["from_node"], e["to_node"]) for e in predicted_edges}
    recovered = []
    missing = []

    for anchor in ANCHOR_EDGES:
        key = (anchor["from"], anchor["to"])
        if key in pred_set:
            recovered.append(f"{anchor['from']} → {anchor['to']}")
        else:
            missing.append(f"{anchor['from']} → {anchor['to']}")

    total = len(ANCHOR_EDGES)
    return {
        "total_anchors": total,
        "recovered": len(recovered),
        "recovery_rate": len(recovered) / total if total else 0.0,
        "missing": missing,
    }


@_tool
def run_evalue_check(effect_size: float, se: float, sample_size: int | None = None) -> dict:
    """
    Compute sensemakr E-value for confounding robustness.
    E-value = minimum confounder association needed to explain away the effect.
    E-value < 2.0 → flag as potentially confounded.

    STUB — uses the closed-form E-value approximation (VanderWeele & Ding 2017).
    Wire in full sensemakr package when rpy2 is available.
    """
    import math
    if se <= 0:
        return {"e_value": None, "interpretation": "invalid SE", "recommendation": "check inputs"}

    # Approximate E-value from risk ratio (RR = exp(effect_size) for log-scale)
    try:
        rr = math.exp(abs(effect_size))
        # E-value formula: RR + sqrt(RR * (RR - 1))
        e_value = rr + math.sqrt(rr * (rr - 1)) if rr > 1 else 1.0
    except (OverflowError, ValueError):
        e_value = None

    if e_value is None:
        interpretation = "could not compute"
        recommendation = "check effect_size units"
    elif e_value < 2.0:
        interpretation = "potentially confounded"
        recommendation = "demote to Tier 3 pending interventional validation"
    elif e_value < 3.0:
        interpretation = "moderate confounding robustness"
        recommendation = "flag for MR validation"
    else:
        interpretation = "robust to moderate confounding"
        recommendation = "no immediate action required"

    return {
        "e_value": round(e_value, 3) if e_value else None,
        "interpretation": interpretation,
        "recommendation": recommendation,
        "note": "STUB — approximation only; use full sensemakr for publication-grade results",
    }


@_tool
def snapshot_graph_version(version_tag: str, release_notes: str) -> dict:
    """
    Create a versioned DVC snapshot of the current graph state.
    STUB — DVC integration pending.

    Returns:
        {"version_tag": str, "timestamp": str, "note": str}
    """
    return {
        "version_tag": version_tag,
        "timestamp": datetime.utcnow().isoformat(),
        "release_notes": release_notes,
        "note": "STUB — DVC snapshot integration pending",
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if mcp is None:
        raise RuntimeError("fastmcp is required to run the server. Install it: pip install fastmcp")
    mcp.run()
