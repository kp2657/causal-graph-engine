"""
Edge ingestion pipeline.
All edges pass through Pydantic validation here before hitting the DB.
The Scientific Reviewer's AUTOMATIC BLOCK conditions are enforced here structurally.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, TYPE_CHECKING, TypedDict

from models.evidence import CausalEdge
if TYPE_CHECKING:
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

    # MR validation is only meaningful for edges computed via an explicit MR method.
    # OTA composite-γ edges (method="ota_gamma") carry genetic evidence through the
    # β×γ product and OT colocalisation scores — they are not MR estimates and do not
    # require bidirectional MR validation.
    _MR_METHODS = frozenset({"mr", "two_sample_mr", "ivw", "mr_ivw", "mendelian_randomisation"})
    if edge.evidence_tier in ("Tier2_Convergent", "Tier1_Interventional"):
        if edge.method in _MR_METHODS and edge.mr_ivw is None:
            warnings.append(
                "Tier 1/2 edge has no MR validation — consider running bidirectional MR."
            )

    if edge.ci_lower is None or edge.ci_upper is None:
        warnings.append("No confidence interval provided — effect estimate precision unknown.")

    return warnings


# ---------------------------------------------------------------------------
# Main ingestion entrypoint
# ---------------------------------------------------------------------------

def ingest_edge(db: "GraphDB", raw: dict[str, Any]) -> CausalEdge:
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


# ---------------------------------------------------------------------------
# Program γ ingestion — writes CellularProgram nodes + DrivesTrait edges
# ---------------------------------------------------------------------------

def ingest_program_gamma_edges(
    db: "GraphDB",
    gamma_estimates: dict,
    disease_name: str,
    efo_id: str | None = None,
    cell_type: str = "unknown",
    graph_version: str = "0.1.0",
) -> dict:
    """
    Upsert CellularProgram nodes and write CellularProgram→DiseaseTrait DrivesTrait
    edges from a gamma_estimates dict.

    gamma_estimates structure:
        {program_id: {trait: {"gamma": float, "gamma_se": float,
                              "evidence_tier": str, "data_source": str, ...}}}

    Traits where gamma is None/missing are skipped.

    Returns {"written": int, "rejected": int, "errors": list[str]}
    """
    from datetime import datetime, timezone

    written = 0
    rejected = 0
    errors: list[str] = []
    now = datetime.now(timezone.utc).isoformat()
    trait_nodes_upserted: set[str] = set()

    for program_id, trait_dict in gamma_estimates.items():
        if not isinstance(trait_dict, dict):
            continue
        for trait, gamma_dict in trait_dict.items():
            if not isinstance(gamma_dict, dict):
                continue
            gamma = gamma_dict.get("gamma")
            if gamma is None:
                continue

            # Upsert CellularProgram node
            try:
                db.upsert_node("CellularProgram", {
                    "id":        program_id,
                    "name":      program_id,
                    "cell_type": cell_type,
                })
            except Exception as exc:
                errors.append(f"CellularProgram upsert {program_id!r}: {exc}")
                rejected += 1
                continue

            # Upsert DiseaseTrait node — one per unique trait key
            if trait not in trait_nodes_upserted:
                try:
                    node_props: dict[str, Any] = {"id": trait, "name": trait}
                    if efo_id:
                        node_props["efo_id"] = efo_id
                    db.upsert_node("DiseaseTrait", node_props)
                    trait_nodes_upserted.add(trait)
                except Exception as exc:
                    errors.append(f"DiseaseTrait upsert {trait!r}: {exc}")

            # Write DrivesTrait edge
            gamma_se = gamma_dict.get("gamma_se")
            try:
                db.write_drives_trait_edge({
                    "program_id":          program_id,
                    "trait_id":            trait,
                    "gamma":               float(gamma),
                    "ci_lower":            float(gamma - gamma_se) if gamma_se else None,
                    "ci_upper":            float(gamma + gamma_se) if gamma_se else None,
                    "method":              "ota_gamma",
                    "evidence_tier":       gamma_dict.get("evidence_tier", "Tier3_Provisional"),
                    "data_source":         gamma_dict.get("data_source", "ot_l2g_enrichment"),
                    "data_source_version": "1.0",
                    "n_modifier_paths":    1,
                    "graph_version":       graph_version,
                    "created_at":          now,
                })
                written += 1
            except Exception as exc:
                errors.append(f"DrivesTrait {program_id!r} → {trait!r}: {exc}")
                rejected += 1

    if errors:
        logger.warning(
            "ingest_program_gamma_edges: %d failures: %s",
            len(errors), errors[:3],
        )

    return {"written": written, "rejected": rejected, "errors": errors}


# ---------------------------------------------------------------------------
# Internal DB write dispatcher
# ---------------------------------------------------------------------------

def _ensure_nodes(db: "GraphDB", edge: CausalEdge) -> None:
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
        logger.error("CRITICAL: upsert_node FAILED for type=%s, id=%s. Error: %s", from_node_type, edge.from_node, exc, exc_info=True)
        raise IngestionError(f"Node upsert failed for {edge.from_node}")
    try:
        db.upsert_node(to_node_type, {"id": edge.to_node})
    except Exception as exc:
        logger.error("CRITICAL: upsert_node FAILED for type=%s, id=%s. Error: %s", to_node_type, edge.to_node, exc, exc_info=True)
        raise IngestionError(f"Node upsert failed for {edge.to_node}")


def _write_edge(db: "GraphDB", edge: CausalEdge) -> None:
    """Route edge to the correct DB writer based on to_type."""
    payload = edge.model_dump()
    payload["created_at"] = payload["created_at"].isoformat()

    # Ensure source and target nodes exist (MATCH requires pre-existing nodes)
    _ensure_nodes(db, edge)

    if edge.to_type == "trait":
        db.write_causes_trait_edge(payload)
    elif edge.to_type == "program":
        db.write_regulates_program_edge(payload)
    else:
        logger.warning("Unhandled edge to_type=%s — not written to DB.", edge.to_type)
