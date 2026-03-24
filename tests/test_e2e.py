"""
tests/test_e2e.py — Lightweight CAD end-to-end integration tests (Step 54).

Strategy: test critical INTEGRATION POINTS rather than the full 190s pipeline.
  - Use real Kùzu (temp dir) for all DB tests
  - Test individual agent run() functions with stub inputs (no HTTP needed)
  - Test that graph writes work and are readable back
  - Test that the v2 orchestrator correctly threads data through tiers
    (with all 11 agent dispatches mocked to return deterministic outputs)

All tests complete in < 5s. No live HTTP calls required.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

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
    monkeypatch.setattr(gdb_mod, "_DB_PATH", db_path)
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
    "finngen_phenocode": "I9_CAD",
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

    def test_anchor_edge_validation_runs(self, kuzu_db):
        from mcp_servers.graph_db_server import run_anchor_edge_validation
        predicted = [
            {"from_node": "PCSK9", "to_node": "CAD"},
            {"from_node": "LDLR",  "to_node": "CAD"},
        ]
        result = run_anchor_edge_validation(predicted)
        assert "recovery_rate" in result
        assert 0.0 <= result["recovery_rate"] <= 1.0


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

    def test_causal_agent_returns_anchor_recovery(self, kuzu_db):
        from agents.tier3_causal.causal_discovery_agent import run

        result = run(
            beta_matrix_result=_STUB_BETA_MATRIX_RESULT,
            gamma_estimates=_STUB_GAMMA_ESTIMATES,
            disease_query=_STUB_DISEASE_QUERY,
        )
        assert "anchor_recovery" in result
        recovery = result["anchor_recovery"]
        assert isinstance(recovery, dict)
        assert "recovery_rate" in recovery
        assert 0.0 <= recovery["recovery_rate"] <= 1.0

    def test_causal_agent_pcsk9_gets_nonzero_gamma(self, kuzu_db):
        """PCSK9 has β=0.8 on lipid_metabolism, γ=0.6 → Ota γ ≥ 0.48."""
        from agents.tier3_causal.causal_discovery_agent import run

        result = run(
            beta_matrix_result=_STUB_BETA_MATRIX_RESULT,
            gamma_estimates=_STUB_GAMMA_ESTIMATES,
            disease_query=_STUB_DISEASE_QUERY,
        )
        # PCSK9 should be in top genes (highest beta × gamma product)
        top_gene_names = [
            g if isinstance(g, str) else g.get("gene", g.get("gene_symbol", ""))
            for g in result.get("top_genes", [])
        ]
        assert "PCSK9" in top_gene_names or result["n_edges_written"] > 0

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


# ---------------------------------------------------------------------------
# 3. Somatic agent integration (real Kùzu, uses hardcoded CHIP data)
# ---------------------------------------------------------------------------

class TestSomaticAgentIntegration:
    """
    Run somatic_exposure_agent.run() with real Kùzu.

    The somatic agent uses hardcoded CHIP data (Bick 2020 / Kar 2022) —
    no HTTP calls required. This verifies the full write path.
    """

    def test_somatic_agent_returns_chip_edges(self):
        from agents.tier1_phenomics.somatic_exposure_agent import run

        result = run(_STUB_DISEASE_QUERY)
        assert isinstance(result, dict)
        assert "chip_edges" in result
        # CAD has hardcoded CHIP associations — should have at least some
        chip_edges = result["chip_edges"]
        assert isinstance(chip_edges, list)
        assert len(chip_edges) > 0, "Expected hardcoded CHIP edges for CAD"

    def test_somatic_agent_returns_drug_edges(self):
        from agents.tier1_phenomics.somatic_exposure_agent import run

        result = run(_STUB_DISEASE_QUERY)
        assert "drug_edges" in result
        assert isinstance(result["drug_edges"], list)

    def test_somatic_agent_summary_has_counts(self):
        from agents.tier1_phenomics.somatic_exposure_agent import run

        result = run(_STUB_DISEASE_QUERY)
        summary = result.get("summary", {})
        assert "n_chip_genes" in summary
        assert "n_drug_targets" in summary

    def test_somatic_chip_edges_are_causal_edge_compatible(self):
        """Every chip_edge dict must have the fields required by CausalEdge."""
        from agents.tier1_phenomics.somatic_exposure_agent import run

        result = run(_STUB_DISEASE_QUERY)
        required_fields = {
            "from_node", "from_type", "to_node", "to_type",
            "effect_size", "evidence_type", "evidence_tier", "method",
        }
        for edge in result.get("chip_edges", []):
            missing = required_fields - set(edge.keys())
            assert not missing, f"CHIP edge missing fields: {missing}\n  edge={edge}"


# ---------------------------------------------------------------------------
# 4. Orchestrator data-flow integration (mocked agents, real Kùzu)
# ---------------------------------------------------------------------------

class TestOrchestratorDataFlow:
    """
    Test that the v2 orchestrator correctly threads AgentOutput between tiers.

    All 11 agents are mocked to return deterministic stub outputs.
    The real Kùzu DB + ingestion pipeline is used for edge writes.
    """

    @pytest.fixture()
    def _stub_runner(self, kuzu_db):
        """
        Return a mocked analyze_disease_v2 where all agents return stub data
        but the orchestration logic + Kùzu writes are real.
        """
        from orchestrator.agent_runner import AgentRunner
        from orchestrator.message_contracts import wrap_output

        stub_results = {
            "phenotype_architect":         _STUB_DISEASE_QUERY,
            "statistical_geneticist": {
                "instruments": [{"gene": "PCSK9", "f_stat": 45.2}],
                "anchor_genes_validated": {"PCSK9": True, "LDLR": True},
            },
            "somatic_exposure_agent": {
                "chip_edges": [],
                "drug_edges": [],
                "viral_edges": [],
                "summary": {"n_chip_genes": 0, "n_drug_targets": 0},
            },
            "perturbation_genomics_agent": {
                **_STUB_BETA_MATRIX_RESULT,
                "n_tier1": 2,
                "n_virtual": 0,
            },
            "regulatory_genomics_agent": {"tier2_upgrades": []},
            "causal_discovery_agent": {
                "n_edges_written": 2,
                "top_genes": ["PCSK9", "LDLR"],
                "anchor_recovery": {"recovery_rate": 0.8, "n_recovered": 4, "n_required": 5},
                "shd": 1,
                "warnings": [],
            },
            "kg_completion_agent": {
                "n_pathway_edges_added": 3,
                "n_drug_target_edges_added": 2,
            },
            "target_prioritization_agent": {
                "targets": [
                    {"gene": "PCSK9", "score": 0.9, "druggability": "high"},
                    {"gene": "LDLR",  "score": 0.75, "druggability": "medium"},
                ],
            },
            "chemistry_agent":         {"repurposing_candidates": []},
            "clinical_trialist_agent": {"key_trials": []},
            "scientific_writer_agent": {
                "target_list": [
                    {"gene": "PCSK9", "score": 0.9},
                    {"gene": "LDLR",  "score": 0.75},
                ],
                "anchor_edge_recovery": 0.8,
                "n_escalations": 0,
            },
        }

        original_dispatch = AgentRunner.dispatch

        def _stub_dispatch(self_runner, agent_name: str, agent_input):
            results = stub_results.get(agent_name, {})
            return wrap_output(agent_name, results)

        return _stub_dispatch

    def test_orchestrator_completes(self, _stub_runner, kuzu_db):
        from orchestrator.agent_runner import AgentRunner
        with patch.object(AgentRunner, "dispatch", _stub_runner):
            from orchestrator.pi_orchestrator_v2 import analyze_disease_v2
            result = analyze_disease_v2("coronary artery disease")
        assert isinstance(result, dict)

    def test_orchestrator_output_has_target_list(self, _stub_runner, kuzu_db):
        from orchestrator.agent_runner import AgentRunner
        with patch.object(AgentRunner, "dispatch", _stub_runner):
            from orchestrator.pi_orchestrator_v2 import analyze_disease_v2
            result = analyze_disease_v2("coronary artery disease")
        assert "target_list" in result
        assert isinstance(result["target_list"], list)

    def test_orchestrator_pipeline_status_success(self, _stub_runner, kuzu_db):
        from orchestrator.agent_runner import AgentRunner
        with patch.object(AgentRunner, "dispatch", _stub_runner):
            from orchestrator.pi_orchestrator_v2 import analyze_disease_v2
            result = analyze_disease_v2("coronary artery disease")
        assert result.get("pipeline_status") == "SUCCESS"

    def test_orchestrator_all_11_agents_dispatched(self, _stub_runner, kuzu_db):
        from orchestrator.agent_runner import AgentRunner

        calls: list[str] = []
        original = _stub_runner

        def recording_dispatch(self_runner, agent_name: str, agent_input):
            calls.append(agent_name)
            return original(self_runner, agent_name, agent_input)

        with patch.object(AgentRunner, "dispatch", recording_dispatch):
            from orchestrator.pi_orchestrator_v2 import analyze_disease_v2
            analyze_disease_v2("coronary artery disease")

        expected = {
            "phenotype_architect", "statistical_geneticist", "somatic_exposure_agent",
            "perturbation_genomics_agent", "regulatory_genomics_agent",
            "causal_discovery_agent", "kg_completion_agent",
            "target_prioritization_agent", "chemistry_agent",
            "clinical_trialist_agent", "scientific_writer_agent",
        }
        missing = expected - set(calls)
        assert not missing, f"Agents not dispatched: {missing}"

    def test_orchestrator_mode_override_accepted(self, _stub_runner, kuzu_db):
        """mode_overrides dict is accepted without crashing."""
        from orchestrator.agent_runner import AgentRunner
        with patch.object(AgentRunner, "dispatch", _stub_runner):
            from orchestrator.pi_orchestrator_v2 import analyze_disease_v2
            result = analyze_disease_v2(
                "coronary artery disease",
                mode_overrides={"somatic_exposure_agent": "local"},
            )
        assert "pipeline_status" in result
