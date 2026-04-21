"""
graph/state_schema.py — State-space node and edge type definitions for the Kùzu graph.

Phase 1: Pure schema spec (Python dicts).  No Kùzu imports; no changes to graph/schema.py.
Phase 2: register_state_schema(conn) will CREATE the tables and integrate with graph/db.py.

Design notes:
- All existing CAUSES_TRAIT / program / gene edges are preserved unchanged.
- State-space objects write into NEW tables alongside, not replacing, existing tables.
- Kùzu does not support edge-on-edge natively, so PERTURBATION_SHIFTS_TRANSITION is
  denormalized: the perturbation × from_state × to_state triple is a property on the
  PERTURBATION_REDIRECTS_STATE edge, not a hyperedge.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Node table schemas  (column_name: kuzu_type)
# ---------------------------------------------------------------------------

STATE_NODE_TABLES: dict[str, dict[str, str]] = {
    "CellState": {
        "state_id":           "STRING",
        "disease":            "STRING",
        "cell_type":          "STRING",
        "resolution":         "STRING",   # coarse | intermediate | fine
        "n_cells":            "INT64",
        "pathological_score": "DOUBLE",
        "stability_score":    "DOUBLE",
        "marker_genes":       "STRING",   # JSON-encoded list[str]
        "program_labels":     "STRING",   # JSON-encoded list[str]
        "context_tags":       "STRING",   # JSON-encoded list[str]
        "evidence_sources":   "STRING",   # JSON-encoded list[str]
    },
    "StateBasin": {
        "basin_id":   "STRING",
        "disease":    "STRING",
        "basin_type": "STRING",   # healthy | pathological | escape | repair
        "state_ids":  "STRING",   # JSON-encoded list[str]
        "n_cells":    "INT64",
    },
    "Perturbation": {
        "perturbation_id":   "STRING",
        "perturbation_type": "STRING",   # KO | KD | CRISPRa | drug | cytokine | exposure
        "target_gene":       "STRING",
        "disease":           "STRING",
        "data_source":       "STRING",
    },
    "FailureMode": {
        "failure_id":        "STRING",
        "disease":           "STRING",
        "perturbation_id":   "STRING",
        "failure_mode":      "STRING",
        "evidence_strength": "DOUBLE",
        "data_source":       "STRING",
    },
}

# ---------------------------------------------------------------------------
# Edge table schemas
# { edge_name: { "from": node_table, "to": node_table, "properties": {...} } }
# ---------------------------------------------------------------------------

STATE_EDGE_TABLES: dict[str, dict] = {
    "TRANSITIONS_TO": {
        "from":       "CellState",
        "to":         "CellState",
        "properties": {
            "baseline_probability": "DOUBLE",
            "uncertainty":          "DOUBLE",
            "direction_evidence":   "STRING",   # JSON-encoded list[str] of prov markers
            "provenance":           "STRING",   # PROV_* constant from schemas.py
        },
    },
    # Denormalized: perturbation effect on a state→state transition stored here
    # (not as edge-on-edge which Kùzu does not support in v0.x)
    "PERTURBATION_REDIRECTS_STATE": {
        "from":       "Perturbation",
        "to":         "CellState",
        "properties": {
            "from_state":       "STRING",   # the origin state id
            "to_state":         "STRING",   # the destination state id (== "to" node)
            "delta_probability": "DOUBLE",
            "effect_se":        "DOUBLE",
            "durability_label": "STRING",   # transient | durable | rebound | escape | unknown
            "evidence_tier":    "STRING",
            "data_source":      "STRING",
        },
    },
    "PERTURBATION_FAILS_IN_STATE": {
        "from":       "Perturbation",
        "to":         "CellState",
        "properties": {
            "failure_mode":      "STRING",
            "evidence_strength": "DOUBLE",
            "data_source":       "STRING",
        },
    },
    "ESCAPES_TO_STATE": {
        "from":       "CellState",
        "to":         "CellState",
        "properties": {
            "trigger_perturbation": "STRING",
            "escape_probability":   "DOUBLE",
        },
    },
    "ASSOCIATED_WITH_FAILURE_MODE": {
        "from":       "Perturbation",
        "to":         "FailureMode",
        "properties": {
            "evidence_strength": "DOUBLE",
        },
    },
    "ANNOTATED_BY_PROGRAM": {
        "from":       "CellState",
        "to":         "CellState",   # placeholder — points to self, carries program annotation
        "properties": {
            "program_id":        "STRING",
            "enrichment_score":  "DOUBLE",
            "program_source":    "STRING",   # cNMF | MSigDB_Hallmark | provisional
        },
    },
    "IN_BASIN": {
        "from":       "CellState",
        "to":         "StateBasin",
        "properties": {
            "membership_score": "DOUBLE",   # [0,1]; 1.0 for hard assignments
        },
    },
}


# ---------------------------------------------------------------------------
# DDL helpers — Phase 2 activation
# ---------------------------------------------------------------------------

def get_create_node_table_cypher(table_name: str) -> str:
    """Return a Kùzu CREATE NODE TABLE statement for a state-space node type."""
    cols = STATE_NODE_TABLES[table_name]
    col_defs = ", ".join(f"{k} {v}" for k, v in cols.items())
    pk = list(cols.keys())[0]   # first column is primary key by convention
    return f"CREATE NODE TABLE IF NOT EXISTS {table_name} ({col_defs}, PRIMARY KEY ({pk}))"


def get_create_rel_table_cypher(edge_name: str) -> str:
    """Return a Kùzu CREATE REL TABLE statement for a state-space edge type."""
    spec = STATE_EDGE_TABLES[edge_name]
    props = spec.get("properties", {})
    prop_defs = (", " + ", ".join(f"{k} {v}" for k, v in props.items())) if props else ""
    return (
        f"CREATE REL TABLE IF NOT EXISTS {edge_name} "
        f"(FROM {spec['from']} TO {spec['to']}{prop_defs})"
    )


def register_state_schema(conn) -> list[str]:
    """
    Execute all CREATE NODE/REL TABLE statements against an open Kùzu connection.
    Safe to call on an already-initialised DB (uses IF NOT EXISTS).
    Returns list of executed statements for logging.

    Phase 2 activation:
        from graph.db import get_connection
        from graph.state_schema import register_state_schema
        with get_connection() as conn:
            register_state_schema(conn)
    """
    executed: list[str] = []
    for table_name in STATE_NODE_TABLES:
        cypher = get_create_node_table_cypher(table_name)
        conn.execute(cypher)
        executed.append(cypher)
    for edge_name in STATE_EDGE_TABLES:
        cypher = get_create_rel_table_cypher(edge_name)
        conn.execute(cypher)
        executed.append(cypher)
    return executed
