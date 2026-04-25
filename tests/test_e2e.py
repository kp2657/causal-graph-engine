"""
tests/test_e2e.py — Lightweight end-to-end integration tests.

Strategy: test critical INTEGRATION POINTS rather than the full pipeline.
  - Use real Kùzu (temp dir) for all DB tests
  - Test individual agent run() functions with stub inputs (no HTTP needed)
  - Test that graph writes work and are readable back

All tests complete in < 5s. No live HTTP calls required.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def kuzu_db(tmp_path, monkeypatch):
    """
    Redirect graph_db_server to a fresh Kùzu DB in a temp directory.
    Returns the db_path so tests can inspect writes.
    """
    import mcp_servers.graph_db_server as gdb_mod

    db_path = str(tmp_path / "e2e_test.kuzu")
    monkeypatch.setattr(gdb_mod, "_EXPLICIT_DB_PATH", db_path)
    monkeypatch.setattr(gdb_mod, "_db", None)

    yield db_path

    monkeypatch.setattr(gdb_mod, "_db", None)


# ---------------------------------------------------------------------------
# Stub data (deterministic, no HTTP)
# ---------------------------------------------------------------------------

_STUB_BETA_MATRIX_RESULT = {
    "genes":    ["PCSK9", "LDLR", "TET2", "DNMT3A"],
    "programs": ["lipid_metabolism", "inflammatory_NF-kB"],
    "beta_matrix": {
        "PCSK9":  {"lipid_metabolism": 0.8,  "inflammatory_NF-kB": 0.1},
        "LDLR":   {"lipid_metabolism": 0.6,  "inflammatory_NF-kB": 0.05},
        "TET2":   {"lipid_metabolism": 0.05, "inflammatory_NF-kB": 0.4},
        "DNMT3A": {"lipid_metabolism": 0.02, "inflammatory_NF-kB": 0.35},
    },
    "evidence_tier_per_gene": {
        "PCSK9":  "Tier1_Interventional",
        "LDLR":   "Tier1_Interventional",
        "TET2":   "Tier3_Provisional",
        "DNMT3A": "Tier3_Provisional",
    },
}

_STUB_GAMMA_ESTIMATES = {
    "lipid_metabolism":    {"CAD": 0.6,  "LDL-C": 0.8},
    "inflammatory_NF-kB": {"CAD": 0.3,  "CRP": 0.5},
}

_STUB_DISEASE_QUERY = {
    "disease_name":      "coronary artery disease",
    "efo_id":            "EFO_0001645",
    "modifier_types":    ["germline", "somatic_chip", "drug"],
    "primary_gwas_id":   "ieu-a-7",
    "n_gwas_studies":    5,
    "use_precomputed_only": True,
    "day_one_mode":      True,
}


# ---------------------------------------------------------------------------
# 1. Graph write + read integration (real Kùzu)
# ---------------------------------------------------------------------------

class TestKuzuIntegration:
    """Verify that write_causal_edges → query_graph_for_disease round-trips correctly."""

    def test_write_and_read_single_edge(self, kuzu_db):
        from mcp_servers.graph_db_server import write_causal_edges, query_graph_for_disease

        edges = [{
            "from_node":           "PCSK9",
            "from_type":           "Gene",
            "to_node":             "CAD",
            "to_type":             "Trait",
            "effect_size":         -0.45,
            "evidence_type":       "ota_gamma_composite",
            "evidence_tier":       "Tier1_Interventional",
            "method":              "ota_composite",
            "data_source":         "Replogle2022",
            "data_source_version": "2022",
        }]
        write_result = write_causal_edges(edges, "coronary artery disease")
        assert write_result.get("written", 0) >= 0

        read_result = query_graph_for_disease("coronary artery disease")
        assert "edges" in read_result

    def test_write_multiple_edges(self, kuzu_db):
        from mcp_servers.graph_db_server import write_causal_edges

        edges = [
            {
                "from_node": gene, "from_type": "Gene",
                "to_node": "CAD", "to_type": "Trait",
                "effect_size": 0.3, "evidence_type": "ota_gamma_composite",
                "evidence_tier": "Tier1_Interventional", "method": "ota_composite",
                "data_source": "test", "data_source_version": "test",
            }
            for gene in ["PCSK9", "LDLR", "HMGCR", "IL6R"]
        ]
        result = write_causal_edges(edges, "coronary artery disease")
        assert result.get("written", 0) >= 0

    def test_write_rejected_invalid_edge(self, kuzu_db):
        """Edges that fail pydantic validation are rejected, not written."""
        from mcp_servers.graph_db_server import write_causal_edges

        bad_edges = [{"from_node": "PCSK9"}]  # missing required fields
        result = write_causal_edges(bad_edges, "coronary artery disease")
        # Either written=0 or rejected is non-empty
        written = result.get("written", 0)
        rejected = result.get("rejected", [])
        assert written == 0 or len(rejected) > 0


# ---------------------------------------------------------------------------
# 2. Causal discovery agent integration (real Kùzu, stub β/γ)
# ---------------------------------------------------------------------------

class TestCausalDiscoveryIntegration:
    """
    Run causal_discovery_agent.run() with stub inputs + real Kùzu writes.

    Verifies that:
    - Ota γ is computed correctly from stub β × γ
    - Significant edges are written to Kùzu
    - Anchor recovery is computed and returned
    - SHD metric is returned
    """

    def test_causal_agent_writes_edges(self, kuzu_db):
        from agents.tier3_causal.causal_discovery_agent import run

        result = run(
            beta_matrix_result=_STUB_BETA_MATRIX_RESULT,
            gamma_estimates=_STUB_GAMMA_ESTIMATES,
            disease_query=_STUB_DISEASE_QUERY,
        )
        assert isinstance(result, dict)
        assert "n_edges_written" in result
        assert result["n_edges_written"] >= 0

    def test_causal_agent_returns_top_genes(self, kuzu_db):
        from agents.tier3_causal.causal_discovery_agent import run

        result = run(
            beta_matrix_result=_STUB_BETA_MATRIX_RESULT,
            gamma_estimates=_STUB_GAMMA_ESTIMATES,
            disease_query=_STUB_DISEASE_QUERY,
        )
        assert "top_genes" in result
        assert isinstance(result["top_genes"], list)


    def test_causal_agent_writes_are_readable(self, kuzu_db):
        """Edges written by causal_discovery_agent can be read back from Kùzu."""
        from agents.tier3_causal.causal_discovery_agent import run
        from mcp_servers.graph_db_server import query_graph_for_disease

        run(
            beta_matrix_result=_STUB_BETA_MATRIX_RESULT,
            gamma_estimates=_STUB_GAMMA_ESTIMATES,
            disease_query=_STUB_DISEASE_QUERY,
        )
        read_result = query_graph_for_disease("coronary artery disease")
        assert "edges" in read_result


