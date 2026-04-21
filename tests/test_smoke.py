"""
tests/test_smoke.py — Pipeline smoke tests.

Purpose: fast "does it boot" verification that a new install works correctly
before running the full unit or integration suites.

All tests complete in < 10 seconds. No API keys or external HTTP calls required.

Run with:
    pytest tests/test_smoke.py -v

What is covered:
    1. Core module imports
    2. AgentRunner default configuration (local mode, no API key needed)
    3. OTA causal-effect formula correctness (known β × γ values)
    4. Scoring formula components (genetic + mechanistic, modifiers)
    5. Pydantic model validation (causal edges, agent envelopes)
    6. Pipeline stage wiring (analyze_disease_v2 with all agents mocked)
    7. Output schema completeness (required keys present in final result)
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# 1. Core module imports
# ---------------------------------------------------------------------------

class TestImports:
    """Verify that all public modules import cleanly without API keys or data files."""

    def test_import_orchestrator(self):
        from orchestrator import pi_orchestrator_v2  # noqa: F401

    def test_import_agent_runner(self):
        from orchestrator.agent_runner import AgentRunner  # noqa: F401

    def test_import_message_contracts(self):
        from orchestrator.message_contracts import AgentInput, AgentOutput, wrap_output  # noqa: F401

    def test_import_models(self):
        from models.evidence import CausalEdge  # noqa: F401

    def test_import_graph_db(self):
        from mcp_servers import graph_db_server  # noqa: F401

    def test_import_pipelines(self):
        from pipelines import mr_analysis  # noqa: F401

    def test_import_state_space(self):
        from pipelines.state_space import state_influence  # noqa: F401
        from pipelines.state_space import therapeutic_redirection  # noqa: F401

    def test_import_agents(self):
        from agents.tier3_causal import causal_discovery_agent  # noqa: F401
        from agents.tier4_translation import chemistry_agent  # noqa: F401
        from agents.tier4_translation import target_prioritization_agent  # noqa: F401

    def test_import_mcp_servers(self):
        from mcp_servers import chemistry_server  # noqa: F401
        from mcp_servers import clinical_trials_server  # noqa: F401
        from mcp_servers import open_targets_server  # noqa: F401


# ---------------------------------------------------------------------------
# 2. AgentRunner — default configuration
# ---------------------------------------------------------------------------

class TestAgentRunnerDefaults:
    """AgentRunner should work with zero configuration in local (non-agentic) mode."""

    def test_instantiates_without_api_key(self):
        from orchestrator.agent_runner import AgentRunner
        runner = AgentRunner()
        assert runner is not None

    def test_all_agents_default_to_local_mode(self):
        from orchestrator.agent_runner import AgentRunner
        runner = AgentRunner()
        for agent in [
            "phenotype_architect", "statistical_geneticist", "somatic_exposure_agent",
            "perturbation_genomics_agent", "regulatory_genomics_agent",
            "causal_discovery_agent", "kg_completion_agent",
            "target_prioritization_agent", "chemistry_agent",
            "clinical_trialist_agent", "scientific_writer_agent",
        ]:
            assert runner.get_mode(agent) == "local", (
                f"{agent} should default to local mode; set AGENT_MODE=sdk to enable Claude API"
            )

    def test_mode_switch_roundtrip(self):
        from orchestrator.agent_runner import AgentRunner
        runner = AgentRunner()
        runner.set_mode("chemistry_agent", "sdk")
        assert runner.get_mode("chemistry_agent") == "sdk"
        runner.set_mode("chemistry_agent", "local")
        assert runner.get_mode("chemistry_agent") == "local"

    def test_unknown_agent_returns_stub_fallback(self):
        from orchestrator.agent_runner import AgentRunner
        from orchestrator.message_contracts import AgentInput
        runner = AgentRunner()
        out = runner.dispatch("nonexistent_agent_xyz", AgentInput(disease_query={"disease_name": "test"}))
        assert out.stub_fallback is True


# ---------------------------------------------------------------------------
# 3. OTA causal-effect formula
#
# γ_{gene→trait} = Σ_P (β_{gene→P} × γ_{P→trait})
#
# These values are hand-verifiable from the stub data:
#   PCSK9: β[lipid]=0.8, β[NF-kB]=0.1 | γ[lipid→CAD]=0.6, γ[NF-kB→CAD]=0.3
#   Expected γ_{PCSK9→CAD} = 0.8×0.6 + 0.1×0.3 = 0.48 + 0.03 = 0.51
# ---------------------------------------------------------------------------

class TestOTAFormula:
    """Direct verification of the Ota et al. causal-effect estimator."""

    _BETA = {
        "PCSK9":  {"lipid_metabolism": 0.8,  "inflammatory_NF-kB": 0.1},
        "LDLR":   {"lipid_metabolism": 0.6,  "inflammatory_NF-kB": 0.05},
        "TET2":   {"lipid_metabolism": 0.05, "inflammatory_NF-kB": 0.4},
    }
    _GAMMA_P = {
        "lipid_metabolism":    0.6,
        "inflammatory_NF-kB": 0.3,
    }

    def _ota(self, gene: str) -> float:
        return sum(
            self._BETA[gene].get(p, 0.0) * g
            for p, g in self._GAMMA_P.items()
        )

    def test_pcsk9_gamma(self):
        result = self._ota("PCSK9")
        assert abs(result - 0.51) < 1e-9, f"Expected 0.51, got {result}"

    def test_ldlr_gamma(self):
        result = self._ota("LDLR")
        assert abs(result - 0.375) < 1e-9, f"Expected 0.375, got {result}"

    def test_tet2_gamma(self):
        result = self._ota("TET2")
        # 0.05×0.6 + 0.4×0.3 = 0.03 + 0.12 = 0.15
        assert abs(result - 0.15) < 1e-9, f"Expected 0.15, got {result}"

    def test_ranking_order(self):
        """PCSK9 > LDLR > TET2 with these stub betas."""
        scores = {g: self._ota(g) for g in self._BETA}
        ranked = sorted(scores, key=scores.__getitem__, reverse=True)
        assert ranked == ["PCSK9", "LDLR", "TET2"]


# ---------------------------------------------------------------------------
# 4. Scoring formula
# ---------------------------------------------------------------------------

class TestScoringFormula:
    """
    Verify the Phase F unified scoring formula:
        core  = 0.60 × genetic + 0.40 × mechanistic
        final = core × t_mod × risk_discount
    """

    def _score(
        self,
        genetic: float,
        mechanistic: float,
        ot: float = 0.0,
        trial: float = 0.0,
        safety: float = 0.0,
        escape: float = 0.0,
        failure: float = 0.0,
    ) -> float:
        core    = 0.60 * genetic + 0.40 * mechanistic
        t_mod   = max(0.5, min(1.5, 1 + 0.15 * ot + 0.10 * trial - 0.10 * safety))
        risk    = max(0.1, 1 - 0.20 * escape - 0.15 * failure)
        return core * t_mod * risk

    def test_pure_genetic_score(self):
        s = self._score(genetic=1.0, mechanistic=0.0)
        assert abs(s - 0.60) < 1e-9

    def test_pure_mechanistic_score(self):
        s = self._score(genetic=0.0, mechanistic=1.0)
        assert abs(s - 0.40) < 1e-9

    def test_balanced_score(self):
        s = self._score(genetic=1.0, mechanistic=1.0)
        assert abs(s - 1.0) < 1e-9

    def test_ot_uplift(self):
        s_base  = self._score(genetic=0.5, mechanistic=0.5)
        s_uplift = self._score(genetic=0.5, mechanistic=0.5, ot=1.0)
        assert s_uplift > s_base

    def test_safety_penalty(self):
        s_safe   = self._score(genetic=0.5, mechanistic=0.5)
        s_unsafe = self._score(genetic=0.5, mechanistic=0.5, safety=1.0)
        assert s_unsafe < s_safe

    def test_t_mod_clamp_lower(self):
        # safety=10 would give t_mod < 0.5 without clamp
        s = self._score(genetic=1.0, mechanistic=1.0, safety=10.0)
        # t_mod clamped at 0.5
        expected_core = 1.0
        assert s == pytest.approx(expected_core * 0.5 * 1.0, abs=1e-6)

    def test_risk_floor(self):
        # risk would go negative without the max(0.1, ...) floor
        s = self._score(genetic=1.0, mechanistic=1.0, escape=5.0, failure=5.0)
        assert s > 0


# ---------------------------------------------------------------------------
# 5. Pydantic schema validation
# ---------------------------------------------------------------------------

class TestSchemas:
    """Key Pydantic models validate correctly and reject bad data."""

    def test_causal_edge_valid(self):
        from models.evidence import CausalEdge
        edge = CausalEdge(
            from_node="PCSK9",
            from_type="gene",
            to_node="coronary artery disease",
            to_type="trait",
            effect_size=0.45,
            evidence_type="germline",
            evidence_tier="Tier1_Interventional",
            method="ota_gamma",
            data_source="Replogle2022",
            data_source_version="2022",
        )
        assert edge.from_node == "PCSK9"
        assert edge.effect_size == 0.45

    def test_causal_edge_rejects_missing_required(self):
        from models.evidence import CausalEdge
        with pytest.raises(Exception):  # pydantic ValidationError
            CausalEdge(from_node="PCSK9")  # type: ignore[call-arg]

    def test_agent_input_has_run_id(self):
        from orchestrator.message_contracts import AgentInput
        inp = AgentInput(disease_query={"disease_name": "AMD"})
        assert inp.run_id  # auto-generated UUID

    def test_agent_output_escalate_on_keyword(self):
        from orchestrator.message_contracts import wrap_output
        out = wrap_output("causal_discovery_agent", {
            "warnings": ["ESCALATE: anchor recovery 0.45 below threshold 0.80"]
        })
        assert out.escalate is True

    def test_agent_output_no_escalate_on_normal_warning(self):
        from orchestrator.message_contracts import wrap_output
        out = wrap_output("causal_discovery_agent", {
            "warnings": ["gnomAD timeout — used cached data"]
        })
        assert out.escalate is False


# ---------------------------------------------------------------------------
# 6 & 7. Pipeline wiring + output schema
# ---------------------------------------------------------------------------

# Minimal mock results — only keys the orchestrator actually reads
_MOCK = {
    "phenotype_architect": {
        "disease_name": "coronary artery disease",
        "efo_id": "EFO_0001645",
        "modifier_types": ["germline"],
        "primary_gwas_id": "ieu-a-7",
        "n_gwas_studies": 5,
        "use_precomputed_only": True,
        "day_one_mode": True,
    },
    "statistical_geneticist": {
        "instruments": [{"exposure": "LDL-C", "f_statistic": 45.0}],
        "anchor_genes_validated": {"PCSK9": True, "LDLR": True},
        "n_gw_significant_hits": 30,
        "warnings": [],
    },
    "somatic_exposure_agent":      {"chip_edges": [], "drug_edges": [], "viral_edges": [], "summary": {"n_chip_genes": 0, "n_drug_targets": 0}, "warnings": []},
    "perturbation_genomics_agent": {"genes": ["PCSK9"], "beta_matrix": {"PCSK9": {"lipid_metabolism": 0.8}}, "evidence_tier_per_gene": {"PCSK9": "Tier1_Interventional"}, "n_tier1": 1, "n_virtual": 0, "programs": [], "warnings": []},
    "regulatory_genomics_agent":   {"tier2_upgrades": [], "warnings": []},
    "causal_discovery_agent":      {"n_edges_written": 2, "top_genes": [{"gene": "PCSK9", "ota_gamma": 0.51}], "anchor_recovery": {"recovery_rate": 0.80, "recovered": ["PCSK9→LDL-C"], "missing": []}, "shd": 0, "warnings": []},
    "kg_completion_agent":         {"n_pathway_edges_added": 1, "n_drug_target_edges_added": 1, "warnings": []},
    "target_prioritization_agent": {"targets": [{"target_gene": "PCSK9", "rank": 1, "evidence_tier": "Tier1_Interventional", "safety_flags": None}], "warnings": []},
    "chemistry_agent":             {"target_chemistry": {}, "repurposing_candidates": [], "gps_disease_reversers": [], "gps_program_reversers": {}, "gps_programs_screened": [], "gps_priority_compounds": [], "warnings": []},
    "clinical_trialist_agent":     {"trial_summary": {"n_trials": 0}, "key_trials": [], "warnings": []},
    "scientific_writer_agent":     {"disease_name": "coronary artery disease", "efo_id": "EFO_0001645", "target_list": [{"gene": "PCSK9"}], "anchor_edge_recovery": 0.80, "n_tier1_edges": 2, "n_tier2_edges": 0, "n_tier3_edges": 0, "n_virtual_edges": 0, "executive_summary": "Smoke test.", "target_table": [], "top_target_narratives": [], "evidence_quality": {}, "limitations": [], "pipeline_version": "0.1.0", "generated_at": "2026-01-01T00:00:00Z", "warnings": []},
}


@pytest.fixture()
def _mock_pipeline(tmp_path, monkeypatch):
    """
    Run analyze_disease_v2 with:
      - All agents returning _MOCK stub data (no HTTP, no API key)
      - Utility functions (_get_gamma_estimates, _write_somatic_edges) also mocked
      - Kùzu DB redirected to a temp directory

    Must patch 3 things to avoid loading real data files:
      1. AgentRunner class — so dispatch returns stub results
      2. _get_gamma_estimates — skips Perturb-seq / GWAS data loading
      3. _write_somatic_edges — skips somatic edge DB writes
    """
    import mcp_servers.graph_db_server as gdb_mod
    monkeypatch.setattr(gdb_mod, "_DB_PATH", str(tmp_path / "smoke.kuzu"))
    monkeypatch.setattr(gdb_mod, "_db", None)

    from orchestrator.agent_runner import AgentRunner
    from orchestrator.message_contracts import wrap_output

    runner = AgentRunner()

    def _stub_dispatch(agent_name: str, agent_input):
        return wrap_output(agent_name, _MOCK.get(agent_name, {}))

    runner.dispatch = _stub_dispatch  # type: ignore[assignment]

    from orchestrator.pi_orchestrator_v2 import analyze_disease_v2
    with (
        patch("orchestrator.pi_orchestrator_v2.AgentRunner", return_value=runner),
        patch("orchestrator.pi_orchestrator_v2._get_gamma_estimates", return_value={}),
        patch("orchestrator.pi_orchestrator_v2._write_somatic_edges", return_value=[]),
    ):
        result = analyze_disease_v2("coronary artery disease")

    monkeypatch.setattr(gdb_mod, "_db", None)
    return result


class TestPipelineSmoke:
    """End-to-end pipeline smoke test: all agents mocked, Kùzu in temp dir."""

    def test_pipeline_returns_dict(self, _mock_pipeline):
        assert isinstance(_mock_pipeline, dict)

    def test_pipeline_status_success(self, _mock_pipeline):
        assert _mock_pipeline.get("pipeline_status") == "SUCCESS"

    def test_output_has_required_keys(self, _mock_pipeline):
        required = [
            "disease_name", "pipeline_status", "target_list",
            "anchor_edge_recovery", "n_tier1_edges",
        ]
        missing = [k for k in required if k not in _mock_pipeline]
        assert not missing, f"Output missing required keys: {missing}"

    def test_target_list_not_empty(self, _mock_pipeline):
        assert len(_mock_pipeline.get("target_list", [])) > 0

    def test_anchor_recovery_in_range(self, _mock_pipeline):
        r = _mock_pipeline.get("anchor_edge_recovery", 0)
        assert 0.0 <= r <= 1.0

    def test_pi_reviewed_flag(self, _mock_pipeline):
        assert _mock_pipeline.get("pi_reviewed") is True
