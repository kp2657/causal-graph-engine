"""
Edge ingestion pipeline.
All edges pass through Pydantic validation here before hitting the DB.
The Scientific Reviewer's AUTOMATIC BLOCK conditions are enforced here structurally.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, TypedDict

from models.evidence import CausalEdge
from graph.db import GraphDB

logger = logging.getLogger(__name__)


class IngestionError(ValueError):
    """Raised when an edge fails validation and cannot be written."""


class IngestionWarning(UserWarning):
    """Raised (logged) for edges that pass but have quality flags."""


# ---------------------------------------------------------------------------
# Block conditions (mirror Scientific Reviewer AUTOMATIC BLOCK)
# ---------------------------------------------------------------------------

def _check_block_conditions(edge: CausalEdge) -> None:
    """
    Raise IngestionError for any hard-block condition.
    These mirror the Scientific Reviewer's AUTOMATIC BLOCK rules.
    """
    # Block 1: data_source must not be empty / a placeholder
    if not edge.data_source or edge.data_source.lower() in ("", "unknown", "memory", "llm"):
        raise IngestionError(
            f"Edge {edge.from_node} → {edge.to_node}: "
            "data_source is missing or derived from parametric memory. "
            "Every edge requires an explicit data provenance tag."
        )

    # Block 2: effect_size must be finite
    import math
    if not math.isfinite(edge.effect_size):
        raise IngestionError(
            f"Edge {edge.from_node} → {edge.to_node}: "
            f"effect_size is non-finite ({edge.effect_size})."
        )

    # Block 3: virtual predictions cannot be ingested without explicit provisional label
    if edge.evidence_tier == "provisional_virtual" and not edge.data_source:
        raise IngestionError(
            f"Edge {edge.from_node} → {edge.to_node}: "
            "Virtual cell prediction missing data_source. "
            "provisional_virtual edges must cite the model + version used."
        )


# ---------------------------------------------------------------------------
# Warning conditions (mirror Scientific Reviewer AUTOMATIC FLAG)
# ---------------------------------------------------------------------------

def _check_warning_conditions(edge: CausalEdge) -> list[str]:
    warnings: list[str] = []

    if edge.e_value is not None and edge.e_value < 2.0:
        warnings.append(
            f"E-value {edge.e_value:.2f} < 2.0 — edge may be explained by unmeasured confounders."
        )

    if edge.evidence_tier in ("Tier2_Convergent", "Tier1_Interventional"):
        if edge.mr_ivw is None:
            warnings.append(
                "Tier 1/2 edge has no MR validation — consider running bidirectional MR."
            )

    if edge.ci_lower is None or edge.ci_upper is None:
        warnings.append("No confidence interval provided — effect estimate precision unknown.")

    return warnings


# ---------------------------------------------------------------------------
# Main ingestion entrypoint
# ---------------------------------------------------------------------------

def ingest_edge(db: GraphDB, raw: dict[str, Any]) -> CausalEdge:
    """
    Validate and write a single causal edge to the graph.

    Args:
        db:  Open GraphDB connection.
        raw: Dict matching the CausalEdge schema fields.

    Returns:
        The validated CausalEdge model.

    Raises:
        IngestionError: if a hard-block condition is met.
        pydantic.ValidationError: if the raw dict doesn't match the schema.
    """
    # Stamp ingestion time if not provided
    raw.setdefault("created_at", datetime.now(timezone.utc))

    # 1. Pydantic validation (schema + type checking)
    edge = CausalEdge(**raw)

    # 2. Hard-block conditions
    _check_block_conditions(edge)

    # 3. Soft warnings
    warnings = _check_warning_conditions(edge)
    for w in warnings:
        logger.warning("Edge %s → %s: %s", edge.from_node, edge.to_node, w)

    # 4. Write to DB
    _write_edge(db, edge)

    logger.info(
        "Ingested edge: %s → %s [%s / %s]",
        edge.from_node, edge.to_node, edge.evidence_tier, edge.method,
    )
    return edge


class IngestionResult(TypedDict):
    written: list[CausalEdge]
    rejected: list[tuple[dict, str]]   # (raw_edge, error_message)


def ingest_edges(db: GraphDB, raw_edges: list[dict[str, Any]]) -> IngestionResult:
    """
    Batch ingest. Continues on IngestionError, collects failures explicitly.

    Returns:
        IngestionResult with .written (succeeded) and .rejected (failed with reason).
        Callers must check len(result["rejected"]) > 0 to detect partial failure.
    """
    written: list[CausalEdge] = []
    rejected: list[tuple[dict, str]] = []

    for raw in raw_edges:
        try:
            written.append(ingest_edge(db, raw))
        except Exception as exc:
            logger.error(
                "Failed to ingest edge %s → %s: %s",
                raw.get("from_node"), raw.get("to_node"), exc,
            )
            rejected.append((raw, str(exc)))

    if rejected:
        logger.warning(
            "%d / %d edges failed ingestion.",
            len(rejected), len(raw_edges),
        )

    return {"written": written, "rejected": rejected}


# ---------------------------------------------------------------------------
# Internal DB write dispatcher
# ---------------------------------------------------------------------------

def _ensure_nodes(db: GraphDB, edge: CausalEdge) -> None:
    """Upsert source and target nodes so MATCH succeeds in write_causes_trait_edge."""
    _from_type_to_node = {
        "gene":    "Gene",
        "drug":    "Drug",
        "virus":   "Virus",
        "program": "CellularProgram",
        "trait":   "DiseaseTrait",
    }
    from_node_type = _from_type_to_node.get(edge.from_type, "Gene")
    to_node_type   = _from_type_to_node.get(edge.to_type,   "DiseaseTrait")

    try:
        db.upsert_node(from_node_type, {"id": edge.from_node})
    except Exception as exc:
        # Log at DEBUG if it looks like a duplicate-key constraint (benign),
        # WARNING otherwise — a silent pass here was the root cause of the
        # n_written=0 bug (2026-03-23).
        _level = logging.DEBUG if "already exist" in str(exc).lower() else logging.WARNING
        logger.log(_level, "upsert_node(%s, %s) failed: %s", from_node_type, edge.from_node, exc)
    try:
        db.upsert_node(to_node_type, {"id": edge.to_node})
    except Exception as exc:
        _level = logging.DEBUG if "already exist" in str(exc).lower() else logging.WARNING
        logger.log(_level, "upsert_node(%s, %s) failed: %s", to_node_type, edge.to_node, exc)


def _write_edge(db: GraphDB, edge: CausalEdge) -> None:
    """Route edge to the correct DB writer based on to_type."""
    payload = edge.model_dump()
    payload["created_at"] = payload["created_at"].isoformat()

    # Ensure source and target nodes exist (MATCH requires pre-existing nodes)
    _ensure_nodes(db, edge)

    if edge.to_type == "trait":
        db.write_causes_trait_edge(payload)
    elif edge.to_type == "program":
        # RegulatesProgram — not yet implemented in db.py.
        # Logged at WARNING (not DEBUG) so callers notice edges are being dropped.
        logger.warning(
            "RegulatesProgram write not yet implemented — edge %s → %s skipped. "
            "Implement db.write_regulates_program_edge() to persist gene→program edges.",
            edge.from_node, edge.to_node,
        )
    else:
        logger.warning("Unhandled edge to_type=%s — not written to DB.", edge.to_type)
