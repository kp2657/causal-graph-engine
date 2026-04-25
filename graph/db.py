"""
Kùzu connection and CRUD helpers.
All graph reads/writes go through this module.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import kuzu

from graph.schema import NODE_TYPES, EDGE_TYPES

_DEFAULT_DB_PATH = os.getenv("GRAPH_DB_PATH", "./data/graph.kuzu")


# ---------------------------------------------------------------------------
# Connection management
# ---------------------------------------------------------------------------

class GraphDB:
    """Thin wrapper around a Kùzu database connection."""

    def __init__(self, db_path: str = _DEFAULT_DB_PATH) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = kuzu.Database(db_path)
        self._conn = kuzu.Connection(self._db)
        self._init_schema()

    def _init_schema(self) -> None:
        """Create node and relationship tables if they don't exist."""
        # Node tables
        for node_name, defn in NODE_TYPES.items():
            cols = ", ".join(
                f"{col} {dtype}"
                for col, dtype in defn["properties"].items()
            )
            pk = defn["primary_key"]
            q = f"CREATE NODE TABLE IF NOT EXISTS {node_name} ({cols}, PRIMARY KEY ({pk}))"
            self._conn.execute(q)

        # Relationship tables — one per edge type, stored as generic CausalEdge properties
        # We use a single unified edge table per relationship name for simplicity.
        edge_ddl = {
            "RegulatesProgram": self._regulates_program_ddl(),
            "DrivesTrait":      self._drives_trait_ddl(),
            "CausesTrait":      self._causes_trait_ddl(),
        }
        for table_name, ddl in edge_ddl.items():
            self._conn.execute(ddl)

    # ------------------------------------------------------------------
    # DDL helpers
    # ------------------------------------------------------------------

    def _regulates_program_ddl(self) -> str:
        return """
        CREATE REL TABLE IF NOT EXISTS RegulatesProgram (
            FROM Gene TO CellularProgram,
            FROM Drug TO CellularProgram,
            FROM Virus TO CellularProgram,
            beta DOUBLE,
            ci_lower DOUBLE,
            ci_upper DOUBLE,
            evidence_type STRING,
            evidence_tier STRING,
            data_source STRING,
            data_source_version STRING,
            edge_method STRING,
            cell_type STRING,
            ancestry STRING,
            e_value DOUBLE,
            graph_version STRING,
            created_at STRING,
            is_demoted BOOLEAN,
            demotion_reason STRING
        )
        """

    def _drives_trait_ddl(self) -> str:
        return """
        CREATE REL TABLE IF NOT EXISTS DrivesTrait (
            FROM CellularProgram TO DiseaseTrait,
            gamma DOUBLE,
            ci_lower DOUBLE,
            ci_upper DOUBLE,
            edge_method STRING,
            evidence_tier STRING,
            data_source STRING,
            data_source_version STRING,
            n_modifier_paths INT64,
            validation_sid DOUBLE,
            validation_shd DOUBLE,
            mr_ivw DOUBLE,
            mr_egger_intercept DOUBLE,
            e_value DOUBLE,
            graph_version STRING,
            created_at STRING,
            is_demoted BOOLEAN,
            demotion_reason STRING
        )
        """

    def _causes_trait_ddl(self) -> str:
        return """
        CREATE REL TABLE IF NOT EXISTS CausesTrait (
            FROM Gene TO DiseaseTrait,
            FROM Drug TO DiseaseTrait,
            FROM Virus TO DiseaseTrait,
            effect_size DOUBLE,
            ci_lower DOUBLE,
            ci_upper DOUBLE,
            path_type STRING,
            evidence_type STRING,
            evidence_tier STRING,
            data_source STRING,
            data_source_version STRING,
            edge_method STRING,
            mr_ivw DOUBLE,
            mr_egger_intercept DOUBLE,
            mr_weighted_median DOUBLE,
            lof_burden_beta DOUBLE,
            gwas_pip DOUBLE,
            e_value DOUBLE,
            graph_version STRING,
            created_at STRING,
            is_demoted BOOLEAN,
            demotion_reason STRING
        )
        """

    # ------------------------------------------------------------------
    # Node upsert
    # ------------------------------------------------------------------

    def upsert_node(self, node_type: str, properties: dict[str, Any]) -> None:
        """Insert or merge a node by its primary key."""
        if node_type not in NODE_TYPES:
            raise ValueError(f"Unknown node type: {node_type}")

        pk = NODE_TYPES[node_type]["primary_key"]
        pk_val = properties[pk]

        # Check existence
        result = self._conn.execute(
            f"MATCH (n:{node_type} {{{pk}: $pk}}) RETURN count(n) AS cnt",
            {"pk": pk_val},
        )
        count = result.get_next()[0]

        if count == 0:
            # Serialize any list values to JSON strings
            serialized = {
                k: json.dumps(v) if isinstance(v, list) else v
                for k, v in properties.items()
            }
            # Kùzu Cypher requires {key: $key, ...} format
            kv_pairs = ", ".join(f"{k}: ${k}" for k in serialized.keys())
            self._conn.execute(
                f"CREATE (n:{node_type} {{{kv_pairs}}})",
                serialized,
            )

    # ------------------------------------------------------------------
    # Edge write
    # ------------------------------------------------------------------

    def write_causes_trait_edge(self, props: dict[str, Any]) -> None:
        """Write a Gene/Drug/Virus → DiseaseTrait causal edge."""
        from_id = props["from_node"]
        to_id = props["to_node"]
        from_type = props["from_type"].capitalize()  # Gene | Drug | Virus

        # Map our model's from_type to the correct Kùzu node type name
        _type_map = {
            "gene": "Gene",
            "drug": "Drug",
            "virus": "Virus",
            "program": "CellularProgram",
            "trait": "DiseaseTrait",
        }
        kfrom = _type_map.get(props["from_type"], from_type)

        edge_props = {k: v for k, v in props.items()
                      if k not in ("from_node", "to_node", "from_type", "to_type")}
        edge_props.setdefault("is_demoted", False)

        # Kùzu cannot bind None parameters — drop them so optional fields are simply absent
        edge_props = {k: v for k, v in edge_props.items() if v is not None}

        # Rename 'method' → 'edge_method' because 'method' is reserved in Kùzu Cypher
        if "method" in edge_props:
            edge_props["edge_method"] = edge_props.pop("method")

        # Kùzu requires inline {key: $key} on CREATE — no SET after CREATE
        kv_pairs = ", ".join(f"{k}: ${k}" for k in edge_props)
        params = {"from_id": from_id, "to_id": to_id, **edge_props}

        self._conn.execute(
            f"""
            MATCH (a:{kfrom} {{id: $from_id}}), (b:DiseaseTrait {{id: $to_id}})
            CREATE (a)-[:CausesTrait {{{kv_pairs}}}]->(b)
            """,
            params,
        )

    def write_drives_trait_edge(self, props: dict[str, Any]) -> None:
        """Write a CellularProgram → DiseaseTrait DrivesTrait edge."""
        from_id = props["program_id"]
        to_id   = props["trait_id"]

        edge_props = {k: v for k, v in props.items()
                      if k not in ("program_id", "trait_id")}
        edge_props.setdefault("is_demoted", False)
        edge_props = {k: v for k, v in edge_props.items() if v is not None}

        if "method" in edge_props:
            edge_props["edge_method"] = edge_props.pop("method")

        kv_pairs = ", ".join(f"{k}: ${k}" for k in edge_props)
        params = {"from_id": from_id, "to_id": to_id, **edge_props}

        self._conn.execute(
            f"""
            MATCH (a:CellularProgram {{id: $from_id}}), (b:DiseaseTrait {{id: $to_id}})
            CREATE (a)-[:DrivesTrait {{{kv_pairs}}}]->(b)
            """,
            params,
        )

    def write_regulates_program_edge(self, props: dict[str, Any]) -> None:
        """Write a Gene/Drug/Virus → CellularProgram regulatory edge."""
        from_id = props["from_node"]
        to_id   = props["to_node"]

        _type_map = {"gene": "Gene", "drug": "Drug", "virus": "Virus"}
        kfrom = _type_map.get(props.get("from_type", "gene"), "Gene")

        edge_props = {k: v for k, v in props.items()
                      if k not in ("from_node", "to_node", "from_type", "to_type")}
        edge_props.setdefault("is_demoted", False)
        edge_props = {k: v for k, v in edge_props.items() if v is not None}

        if "method" in edge_props:
            edge_props["edge_method"] = edge_props.pop("method")

        kv_pairs = ", ".join(f"{k}: ${k}" for k in edge_props)
        params = {"from_id": from_id, "to_id": to_id, **edge_props}

        self._conn.execute(
            f"""
            MATCH (a:{kfrom} {{id: $from_id}}), (b:CellularProgram {{id: $to_id}})
            CREATE (a)-[:RegulatesProgram {{{kv_pairs}}}]->(b)
            """,
            params,
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def query_disease_edges(self, disease_id: str) -> list[dict]:
        """Return all edges targeting a given disease node."""
        result = self._conn.execute(
            """
            MATCH (src)-[r:CausesTrait]->(d:DiseaseTrait {id: $disease_id})
            RETURN
                src.id AS from_node,
                d.id   AS to_node,
                r.effect_size,
                r.evidence_tier,
                r.evidence_type,
                r.edge_method,
                r.data_source,
                r.graph_version,
                r.is_demoted
            """,
            {"disease_id": disease_id},
        )
        rows = []
        while result.has_next():
            row = result.get_next()
            rows.append({
                "from_node":      row[0],
                "to_node":        row[1],
                "effect_size":    row[2],
                "evidence_tier":  row[3],
                "evidence_type":  row[4],
                "method":         row[5],
                "data_source":    row[6],
                "graph_version":  row[7],
                "is_demoted":     row[8],
            })
        return rows

    def demote_edge(self, from_node: str, to_node: str, reason: str) -> None:
        """Mark a CausesTrait edge as demoted (soft delete)."""
        self._conn.execute(
            """
            MATCH (a {id: $from_id})-[r:CausesTrait]->(b {id: $to_id})
            SET r.is_demoted = true, r.demotion_reason = $reason
            """,
            {"from_id": from_node, "to_id": to_node, "reason": reason},
        )

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "GraphDB":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
