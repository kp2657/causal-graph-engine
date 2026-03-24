"""
graph/export.py — RDF/Turtle + JSON-LD BioLink-compliant export.

Exports the Kùzu causal graph to:
  1. Turtle (.ttl) — standard RDF, compatible with Protégé / SPARQL
  2. JSON-LD (.jsonld) — BioLink 3.x compliant, compatible with KGX/TRAPI
  3. CSV edge list — for downstream ML pipelines

BioLink 3.x type mappings:
  Gene        → biolink:Gene
  DiseaseTrait → biolink:Disease
  Drug        → biolink:ChemicalEntity
  Virus       → biolink:InfectiousAgent
  CausesTrait → biolink:contributes_to (causal)
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# BioLink namespace mappings
# ---------------------------------------------------------------------------

PREFIXES = {
    "biolink": "https://w3id.org/biolink/vocab/",
    "ENSEMBL": "https://identifiers.org/ensembl:",
    "EFO":     "https://www.ebi.ac.uk/efo/EFO_",
    "CHEMBL":  "https://www.ebi.ac.uk/chembl/compound_report_card/",
    "NCBITaxon": "https://identifiers.org/taxonomy:",
    "cge":     "https://causal-graph-engine.org/graph/",
    "xsd":     "http://www.w3.org/2001/XMLSchema#",
    "rdf":     "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs":    "http://www.w3.org/2000/01/rdf-schema#",
}

BIOLINK_NODE_TYPES: dict[str, str] = {
    "gene":    "biolink:Gene",
    "trait":   "biolink:Disease",
    "drug":    "biolink:ChemicalEntity",
    "virus":   "biolink:InfectiousAgent",
    "program": "biolink:BiologicalProcess",
}

BIOLINK_EDGE_TYPES: dict[str, str] = {
    "Tier1_Interventional": "biolink:contributes_to",
    "Tier2_Convergent":     "biolink:contributes_to",
    "Tier3_Provisional":    "biolink:related_to",
    "moderate_transferred": "biolink:related_to",
    "moderate_grn":         "biolink:related_to",
    "provisional_virtual":  "biolink:related_to",
}


def _node_uri(node_id: str, node_type: str) -> str:
    """Map node id to a URI string."""
    node_id_clean = node_id.replace(" ", "_")
    if node_type == "trait" and "EFO_" in node_id:
        return f"EFO:{node_id.replace('EFO_', '')}"
    if node_type == "gene":
        return f"cge:gene/{node_id_clean}"
    if node_type == "drug":
        return f"cge:drug/{node_id_clean}"
    if node_type == "virus":
        return f"cge:virus/{node_id_clean}"
    return f"cge:node/{node_id_clean}"


def _edge_uri(from_node: str, to_node: str, method: str, version: str) -> str:
    return f"cge:edge/{from_node}__{to_node}__{method}__{version}".replace(" ", "_")


# ---------------------------------------------------------------------------
# Turtle export
# ---------------------------------------------------------------------------

def export_turtle(edges: list[dict], out: TextIO | None = None) -> str:
    """
    Export edges to RDF/Turtle format.

    Args:
        edges: List of CausalEdge-compatible dicts
        out:   Optional file handle to write to (also returns string)

    Returns:
        Turtle string
    """
    lines: list[str] = []

    # Prefix declarations
    for prefix, uri in PREFIXES.items():
        lines.append(f"@prefix {prefix}: <{uri}> .")
    lines.append("")

    for edge in edges:
        if edge.get("is_demoted", False):
            continue

        from_id   = edge.get("from_node", "")
        to_id     = edge.get("to_node", "")
        from_type = edge.get("from_type", "gene")
        tier      = edge.get("evidence_tier", "provisional_virtual")
        method    = edge.get("edge_method") or edge.get("method", "unknown")
        version   = edge.get("graph_version", "0.1.0")
        effect    = edge.get("effect_size")
        evalue    = edge.get("e_value")
        source    = edge.get("data_source", "")
        created   = edge.get("created_at", datetime.now(tz=timezone.utc).isoformat())

        edge_uri      = _edge_uri(from_id, to_id, method, version)
        from_uri      = _node_uri(from_id, from_type)
        to_uri        = _node_uri(to_id, "trait")
        biolink_pred  = BIOLINK_EDGE_TYPES.get(tier, "biolink:related_to")
        biolink_subj  = BIOLINK_NODE_TYPES.get(from_type, "biolink:NamedThing")

        lines.append(f"cge:{edge_uri} a biolink:Association ;")
        lines.append(f"    biolink:subject {from_uri} ;")
        lines.append(f"    biolink:predicate {biolink_pred} ;")
        lines.append(f"    biolink:object {to_uri} ;")
        lines.append(f"    biolink:subject_category {biolink_subj} ;")
        lines.append(f"    biolink:object_category biolink:Disease ;")
        lines.append(f"    cge:evidence_tier \"{tier}\" ;")
        lines.append(f"    cge:method \"{method}\" ;")
        if effect is not None:
            lines.append(f"    cge:effect_size {effect}^^xsd:double ;")
        if evalue is not None:
            lines.append(f"    cge:e_value {evalue}^^xsd:double ;")
        if source:
            lines.append(f"    dcterms:source \"{source}\" ;")
        lines.append(f"    dcterms:created \"{created}\"^^xsd:dateTime .")
        lines.append("")

    turtle_str = "\n".join(lines)
    if out is not None:
        out.write(turtle_str)
    return turtle_str


# ---------------------------------------------------------------------------
# JSON-LD export
# ---------------------------------------------------------------------------

def export_jsonld(edges: list[dict], disease_name: str = "") -> dict:
    """
    Export edges to JSON-LD BioLink 3.x format.

    Returns a JSON-LD document (as a Python dict, serialize with json.dumps).
    """
    context = {
        "@vocab":    "https://w3id.org/biolink/vocab/",
        "biolink":   "https://w3id.org/biolink/vocab/",
        "cge":       "https://causal-graph-engine.org/graph/",
        "xsd":       "http://www.w3.org/2001/XMLSchema#",
        "id":        "@id",
        "type":      "@type",
    }

    nodes_seen: dict[str, dict] = {}
    associations: list[dict] = []

    for edge in edges:
        if edge.get("is_demoted", False):
            continue

        from_id   = edge.get("from_node", "")
        to_id     = edge.get("to_node", "")
        from_type = edge.get("from_type", "gene")
        tier      = edge.get("evidence_tier", "provisional_virtual")
        method    = edge.get("edge_method") or edge.get("method", "unknown")
        version   = edge.get("graph_version", "0.1.0")
        effect    = edge.get("effect_size")
        evalue    = edge.get("e_value")
        source    = edge.get("data_source", "")

        edge_id = f"cge:edge/{from_id}__{to_id}__{method}__{version}".replace(" ", "_")
        biolink_pred = BIOLINK_EDGE_TYPES.get(tier, "biolink:related_to")
        from_biolink = BIOLINK_NODE_TYPES.get(from_type, "biolink:NamedThing")

        # Register nodes
        if from_id not in nodes_seen:
            nodes_seen[from_id] = {
                "id":   f"cge:gene/{from_id}",
                "type": from_biolink,
                "name": from_id,
            }
        if to_id not in nodes_seen:
            nodes_seen[to_id] = {
                "id":   f"cge:trait/{to_id}".replace(" ", "_"),
                "type": "biolink:Disease",
                "name": to_id,
            }

        assoc: dict = {
            "id":        edge_id,
            "type":      "biolink:Association",
            "subject":   nodes_seen[from_id]["id"],
            "predicate": biolink_pred,
            "object":    nodes_seen[to_id]["id"],
            "cge:evidence_tier": tier,
            "cge:method": method,
        }
        if effect is not None:
            assoc["cge:effect_size"] = {
                "@type":  "xsd:double",
                "@value": str(effect),
            }
        if evalue is not None:
            assoc["cge:e_value"] = {
                "@type":  "xsd:double",
                "@value": str(evalue),
            }
        if source:
            assoc["biolink:has_evidence"] = source

        associations.append(assoc)

    return {
        "@context": context,
        "@graph": list(nodes_seen.values()) + associations,
        "cge:disease": disease_name,
        "cge:generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "cge:n_associations": len(associations),
    }


# ---------------------------------------------------------------------------
# CSV edge list export
# ---------------------------------------------------------------------------

def export_csv(edges: list[dict]) -> str:
    """
    Export edges as a CSV string (for ML pipeline ingestion).

    Columns: from_node, to_node, effect_size, evidence_tier, method,
             e_value, data_source, graph_version, is_demoted
    """
    header = "from_node,to_node,effect_size,evidence_tier,method,e_value,data_source,graph_version,is_demoted"
    rows: list[str] = [header]

    for e in edges:
        row = ",".join([
            str(e.get("from_node", "")),
            str(e.get("to_node", "")),
            str(e.get("effect_size", "")),
            str(e.get("evidence_tier", "")),
            str(e.get("edge_method") or e.get("method", "")),
            str(e.get("e_value", "")),
            str(e.get("data_source", "")).replace(",", ";"),
            str(e.get("graph_version", "")),
            str(e.get("is_demoted", False)),
        ])
        rows.append(row)

    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Convenience: export disease graph to files
# ---------------------------------------------------------------------------

def export_disease_graph(
    disease_name: str,
    output_dir: str | Path = "./data/exports",
) -> dict:
    """
    Export the full causal graph for a disease to RDF, JSON-LD, and CSV.

    Args:
        disease_name: Disease to export
        output_dir:   Directory to write files to

    Returns:
        {turtle_path, jsonld_path, csv_path, n_edges}
    """
    from mcp_servers.graph_db_server import query_graph_for_disease

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    result = query_graph_for_disease(disease_name)
    edges = result.get("edges", [])

    disease_slug = disease_name.lower().replace(" ", "_")

    # Turtle
    ttl_path = output_path / f"{disease_slug}.ttl"
    with ttl_path.open("w") as f:
        export_turtle(edges, out=f)

    # JSON-LD
    jsonld_path = output_path / f"{disease_slug}.jsonld"
    jsonld_doc = export_jsonld(edges, disease_name=disease_name)
    with jsonld_path.open("w") as f:
        json.dump(jsonld_doc, f, indent=2)

    # CSV
    csv_path = output_path / f"{disease_slug}_edges.csv"
    with csv_path.open("w") as f:
        f.write(export_csv(edges))

    return {
        "turtle_path":  str(ttl_path),
        "jsonld_path":  str(jsonld_path),
        "csv_path":     str(csv_path),
        "n_edges":      len(edges),
    }
