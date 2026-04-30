"""
Anchor edge tests — validate the end-to-end slice:
  nodes upserted → edge validated → edge written → edge read back correctly

Focus: CAD-relevant anchors (PCSK9→LDL-C, TET2_chip→CAD, statin→HMGCR_pathway)
These are our first validation targets before building any agents.
"""
from __future__ import annotations

import os
import tempfile
import pytest

from graph.db import GraphDB
from graph.ingestion import ingest_edge, IngestionError
from models.evidence import CausalEdge


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_db(tmp_path):
    """Isolated Kùzu DB per test."""
    db_path = str(tmp_path / "test_graph.kuzu")
    db = GraphDB(db_path)
    yield db
    db.close()


def _seed_nodes(db: GraphDB) -> None:
    """Upsert the nodes needed for anchor edge tests."""
    db.upsert_node("Gene", {
        "id": "PCSK9",
        "symbol": "PCSK9",
        "entrez_id": "255738",
        "ensembl_id": "ENSG00000169174",
        "gnomad_pli": 0.0,
        "gnomad_loeuf": 0.52,
        "is_chip_driver": False,
        "is_viral_gene": False,
    })
    db.upsert_node("Gene", {
        "id": "TET2_chip",
        "symbol": "TET2",
        "entrez_id": "54790",
        "ensembl_id": "ENSG00000168769",
        "gnomad_pli": 0.0,
        "gnomad_loeuf": 0.21,
        "is_chip_driver": True,
        "is_viral_gene": False,
    })
    db.upsert_node("Gene", {
        "id": "DNMT3A_chip",
        "symbol": "DNMT3A",
        "entrez_id": "1788",
        "ensembl_id": "ENSG00000119772",
        "gnomad_pli": 0.92,
        "gnomad_loeuf": 0.14,
        "is_chip_driver": True,
        "is_viral_gene": False,
    })
    db.upsert_node("DiseaseTrait", {
        "id": "EFO_0001645",
        "name": "coronary artery disease",
        "efo_id": "EFO_0001645",
        "icd10_codes": '["I25", "I21"]',
        "h2_estimate": 0.08,
    })
    db.upsert_node("DiseaseTrait", {
        "id": "LDL-C",
        "name": "LDL cholesterol",
        "efo_id": "EFO_0004611",
        "icd10_codes": "[]",
        "h2_estimate": 0.52,
    })


# ---------------------------------------------------------------------------
# Test 1: schema validation — valid edge passes
# ---------------------------------------------------------------------------

def test_valid_edge_passes_validation():
    raw = {
        "from_node": "PCSK9",
        "from_type": "gene",
        "to_node": "LDL-C",
        "to_type": "trait",
        "effect_size": -0.48,
        "ci_lower": -0.55,
        "ci_upper": -0.41,
        "evidence_type": "germline",
        "evidence_tier": "Tier1_Interventional",
        "method": "mr",
        "data_source": "IEU Open GWAS: ebi-a-GCST002222",
        "data_source_version": "2024-01",
        "mr_ivw": -0.48,
        "mr_egger_intercept": 0.002,
        "e_value": 4.1,
        "ancestry": "EUR",
    }
    edge = CausalEdge(**raw)
    assert edge.from_node == "PCSK9"
    assert edge.to_node == "LDL-C"
    assert edge.evidence_tier == "Tier1_Interventional"
    assert not edge.is_confounded()


# ---------------------------------------------------------------------------
# Test 2: block condition — missing data_source is rejected
# ---------------------------------------------------------------------------

def test_missing_data_source_is_blocked(tmp_db):
    _seed_nodes(tmp_db)
    raw = {
        "from_node": "PCSK9",
        "from_type": "gene",
        "to_node": "LDL-C",
        "to_type": "trait",
        "effect_size": -0.48,
        "evidence_type": "germline",
        "evidence_tier": "Tier1_Interventional",
        "method": "mr",
        "data_source": "memory",     # <- blocked
        "data_source_version": "?",
    }
    with pytest.raises(IngestionError, match="parametric memory"):
        ingest_edge(tmp_db, raw)


# ---------------------------------------------------------------------------
# Test 3: E-value flag — low E-value triggers warning (not block)
# ---------------------------------------------------------------------------

def test_low_evalue_flagged_but_not_blocked(tmp_db):
    _seed_nodes(tmp_db)
    raw = {
        "from_node": "TET2_chip",
        "from_type": "gene",
        "to_node": "EFO_0001645",
        "to_type": "trait",
        "effect_size": 0.12,
        "ci_lower": 0.04,
        "ci_upper": 0.20,
        "evidence_type": "somatic_chip",
        "evidence_tier": "Tier2_Convergent",
        "method": "mr",
        "data_source": "Bick 2020 Nat Genet — CHIP GWAS, Table S4",
        "data_source_version": "2020-11",
        "e_value": 1.4,      # below 2.0 — should warn, not block
    }
    edge = ingest_edge(tmp_db, raw)
    assert edge.is_confounded()
    assert edge.e_value == 1.4


# ---------------------------------------------------------------------------
# Test 4: write + read roundtrip — PCSK9→LDL-C anchor edge
# ---------------------------------------------------------------------------

def test_pcsk9_ldlc_write_and_read(tmp_db):
    _seed_nodes(tmp_db)
    raw = {
        "from_node": "PCSK9",
        "from_type": "gene",
        "to_node": "LDL-C",
        "to_type": "trait",
        "effect_size": -0.48,
        "ci_lower": -0.55,
        "ci_upper": -0.41,
        "evidence_type": "germline",
        "evidence_tier": "Tier1_Interventional",
        "method": "mr",
        "data_source": "IEU Open GWAS: ebi-a-GCST002222",
        "data_source_version": "2024-01",
        "mr_ivw": -0.48,
        "mr_egger_intercept": 0.002,
        "e_value": 4.1,
    }
    ingest_edge(tmp_db, raw)
    edges = tmp_db.query_disease_edges("LDL-C")
    assert len(edges) == 1
    e = edges[0]
    assert e["from_node"] == "PCSK9"
    assert e["evidence_tier"] == "Tier1_Interventional"
    assert not e["is_demoted"]


# ---------------------------------------------------------------------------
# Test 5: write + read — CHIP (TET2→CAD) anchor edge
# ---------------------------------------------------------------------------

def test_tet2_cad_chip_anchor(tmp_db):
    _seed_nodes(tmp_db)
    raw = {
        "from_node": "TET2_chip",
        "from_type": "gene",
        "to_node": "EFO_0001645",
        "to_type": "trait",
        "effect_size": 0.18,
        "ci_lower": 0.10,
        "ci_upper": 0.27,
        "evidence_type": "somatic_chip",
        "evidence_tier": "Tier2_Convergent",
        "method": "mr",
        "data_source": "Bick 2020 Nat Genet — CHIP GWAS, Table S4",
        "data_source_version": "2020-11",
        "e_value": 3.2,
    }
    ingest_edge(tmp_db, raw)
    edges = tmp_db.query_disease_edges("EFO_0001645")
    assert any(e["from_node"] == "TET2_chip" for e in edges)


# ---------------------------------------------------------------------------
# Test 6: write + read — DNMT3A→CAD anchor edge (Bick 2020)
# ---------------------------------------------------------------------------

def test_dnmt3a_cad_anchor(tmp_db):
    _seed_nodes(tmp_db)
    raw = {
        "from_node": "DNMT3A_chip",
        "from_type": "gene",
        "to_node": "EFO_0001645",
        "to_type": "trait",
        "effect_size": 0.15,
        "ci_lower": 0.07,
        "ci_upper": 0.23,
        "evidence_type": "somatic_chip",
        "evidence_tier": "Tier2_Convergent",
        "method": "mr",
        "data_source": "Bick 2020 Nat Genet — CHIP GWAS, Table S4",
        "data_source_version": "2020-11",
        "e_value": 2.8,
    }
    ingest_edge(tmp_db, raw)
    edges = tmp_db.query_disease_edges("EFO_0001645")
    assert any(e["from_node"] == "DNMT3A_chip" for e in edges)


# ---------------------------------------------------------------------------
# Test 7: E-value tool returns sensible output
# ---------------------------------------------------------------------------

def test_evalue_tool_low():
    from mcp_servers.graph_db_server import run_evalue_check
    result = run_evalue_check(effect_size=0.05, se=0.03)
    # Small effect → low E-value → flagged
    assert result["e_value"] is not None
    if result["e_value"] < 2.0:
        assert result["interpretation"] == "potentially confounded"


def test_evalue_tool_high():
    from mcp_servers.graph_db_server import run_evalue_check
    result = run_evalue_check(effect_size=1.2, se=0.1)
    assert result["e_value"] is not None
    assert result["e_value"] >= 2.0




# ---------------------------------------------------------------------------
# Program gamma edge tests
# ---------------------------------------------------------------------------

def test_ingest_program_gamma_edges_writes_nodes_and_edges(tmp_path):
    """CellularProgram nodes and DrivesTrait edges are written from gamma_estimates."""
    db_path = str(tmp_path / "prog_test.kuzu")
    db = GraphDB(db_path)

    gamma_estimates = {
        "CAD_NMF_P01": {
            "CAD": {
                "gamma":         0.42,
                "gamma_se":      0.08,
                "evidence_tier": "Tier2_Convergent",
                "data_source":   "OT_L2G_enrichment",
            },
        },
        "CAD_NMF_P02": {
            "CAD": {
                "gamma":         0.15,
                "gamma_se":      0.05,
                "evidence_tier": "Tier3_Provisional",
                "data_source":   "gwas_ot_overlap_3_genes",
            },
            "LDL-C": {
                "gamma":         0.27,
                "gamma_se":      None,
                "evidence_tier": "Tier3_Provisional",
                "data_source":   "h5ad_deg_overlap",
            },
        },
    }

    from graph.ingestion import ingest_program_gamma_edges
    result = ingest_program_gamma_edges(
        db, gamma_estimates,
        disease_name="coronary artery disease",
        efo_id="EFO_0001645",
        cell_type="cardiac_endothelial_cell",
    )
    db.close()

    assert result["written"] == 3          # P01→CAD, P02→CAD, P02→LDL-C
    assert result["rejected"] == 0
    assert result["errors"] == []


def test_ingest_program_gamma_edges_skips_none_gamma(tmp_path):
    """Entries with gamma=None are not written."""
    db_path = str(tmp_path / "prog_none.kuzu")
    db = GraphDB(db_path)

    gamma_estimates = {
        "CAD_NMF_P03": {
            "CAD":   {"gamma": None, "evidence_tier": "Tier3_Provisional", "data_source": "x"},
            "LDL-C": {"gamma": 0.11, "gamma_se": 0.04, "evidence_tier": "Tier3_Provisional", "data_source": "x"},
        },
    }

    from graph.ingestion import ingest_program_gamma_edges
    result = ingest_program_gamma_edges(db, gamma_estimates, disease_name="coronary artery disease")
    db.close()

    assert result["written"] == 1          # only LDL-C edge written
    assert result["rejected"] == 0


def test_write_program_gamma_edges_server_wrapper(tmp_path, monkeypatch):
    """graph_db_server.write_program_gamma_edges delegates to ingest correctly."""
    import mcp_servers.graph_db_server as _srv

    db_path = str(tmp_path / "srv_prog.kuzu")
    db = GraphDB(db_path)
    monkeypatch.setattr(_srv, "_get_db_for_key", lambda _key: db)

    gamma_estimates = {
        "RA_NMF_P01": {
            "RA": {"gamma": 0.33, "gamma_se": 0.07, "evidence_tier": "Tier2_Convergent",
                   "data_source": "OT_L2G_enrichment"},
        },
    }

    result = _srv.write_program_gamma_edges(
        gamma_estimates, disease="rheumatoid arthritis",
        efo_id="EFO_0000685", cell_type="CD4_T_cell",
    )
    db.close()

    assert result["written"] == 1
    assert result["rejected"] == 0


def test_disease_trait_map_has_required_diseases():
    """DISEASE_TRAIT_MAP covers the diseases we currently run pipelines for."""
    from graph.schema import DISEASE_TRAIT_MAP
    for disease in ("CAD", "RA"):
        assert disease in DISEASE_TRAIT_MAP, f"{disease} missing from DISEASE_TRAIT_MAP"
        assert len(DISEASE_TRAIT_MAP[disease]) > 0, f"{disease} has empty trait list"
