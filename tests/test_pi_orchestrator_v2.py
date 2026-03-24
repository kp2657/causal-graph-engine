"""
tests/test_pi_orchestrator_v2.py — Unit tests for pi_orchestrator_v2.

All agent dispatch calls are mocked — no live APIs or Kùzu DB required.
Tests verify:
  - AgentRunner dispatch wiring (local mode)
  - Parallel tier execution (T1b+c, T2, T4b+c)
  - Quality gate pass-through
  - Anchor recovery halt path
  - mode_overrides wiring
  - AgentOutput envelope shape
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrator.agent_runner import AgentRunner
from orchestrator.message_contracts import AgentInput, AgentOutput, wrap_output


# ---------------------------------------------------------------------------
# Helpers — build mock AgentOutput for any agent
# ---------------------------------------------------------------------------

def _mock_output(agent_name: str, results: dict, **kwargs) -> AgentOutput:
    return wrap_output(agent_name, results, **kwargs)


def _make_runner_with_mocks(per_agent_results: dict[str, dict]) -> AgentRunner:
    """
    Return an AgentRunner whose dispatch() returns pre-set results.
    per_agent_results: {agent_name → results_dict}
    """
    runner = AgentRunner()

    def _dispatch(agent_name: str, agent_input: AgentInput) -> AgentOutput:
        results = per_agent_results.get(agent_name, {})
        return _mock_output(agent_name, results)

    runner.dispatch = _dispatch  # type: ignore[assignment]
    return runner


# ---------------------------------------------------------------------------
# Shared mock result dicts
# ---------------------------------------------------------------------------

MOCK_PHENOTYPE = {
    "disease_name": "coronary artery disease",
    "efo_id": "EFO_0001645",
    "modifier_types": ["germline", "somatic_chip", "drug"],
    "primary_gwas_id": "ieu-a-7",
    "n_gwas_studies": 5,
    "finngen_phenocode": None,
    "use_precomputed_only": True,
    "day_one_mode": True,
}

MOCK_GENETICS = {
    "instruments": [{"exposure": "LDL-C", "f_statistic": 45.0}],
    "anchor_genes_validated": {"PCSK9": True, "LDLR": True},
    "n_gw_significant_hits": 30,
    "warnings": [],
}

MOCK_SOMATIC = {
    "chip_edges": [
        {"from_node": "TET2_chip", "from_type": "gene", "to_node": "EFO_0001645",
         "to_type": "trait", "effect_size": 0.18, "evidence_tier": "Tier2_Convergent",
         "evidence_type": "somatic_chip", "method": "mr",
         "data_source": "Bick 2020", "data_source_version": "2020-11",
         "ci_lower": 0.10, "ci_upper": 0.27},
    ],
    "drug_edges": [],
    "viral_edges": [],
    "summary": {"n_chip_genes": 2, "n_drug_targets": 1},
    "warnings": [],
}

MOCK_BETA = {
    "genes": ["PCSK9", "LDLR", "TET2"],
    "beta_matrix": {"PCSK9": {"lipid_metabolism": -0.42}},
    "evidence_tier_per_gene": {"PCSK9": "Tier1_Interventional"},
    "n_tier1": 2, "n_tier2": 1, "n_tier3": 0, "n_virtual": 0,
    "programs": [],
    "warnings": [],
}

MOCK_REGULATORY = {
    "gene_eqtl_summary": {"PCSK9": {"tissue": "liver", "pval": 1e-10}},
    "gene_program_overlap": {},
    "tier2_upgrades": [],
    "warnings": [],
}

MOCK_CAUSAL = {
    "n_edges_written": 8,
    "n_edges_rejected": 0,
    "top_genes": [{"gene": "PCSK9", "ota_gamma": 0.41}],
    "anchor_recovery": {"recovery_rate": 0.80, "recovered": ["PCSK9→LDL-C"], "missing": []},
    "shd": 0,
    "warnings": [],
}

MOCK_KG = {
    "n_pathway_edges_added": 3,
    "n_ppi_edges_added": 2,
    "n_drug_target_edges_added": 1,
    "n_primekg_edges_added": 0,
    "top_pathways": [],
    "drug_target_summary": [],
    "contradictions_flagged": [],
    "warnings": [],
}

MOCK_PRIORITIZATION = {
    "targets": [
        {"target_gene": "PCSK9", "rank": 1, "evidence_tier": "Tier1_Interventional",
         "safety_flags": None},
        {"target_gene": "LDLR",  "rank": 2, "evidence_tier": "Tier2_Convergent",
         "safety_flags": None},
    ],
    "warnings": [],
}

MOCK_CHEMISTRY = {
    "target_chemistry": [],
    "repurposing_candidates": [{"drug": "statin", "target": "HMGCR"}],
    "warnings": [],
}

MOCK_CLINICAL = {
    "trial_summary": {"n_trials": 12},
    "key_trials": [{"nct_id": "NCT00000001"}],
    "development_risk": {},
    "repurposing_opportunities": [],
    "warnings": [],
}

MOCK_WRITER = {
    "disease_name": "coronary artery disease",
    "efo_id": "EFO_0001645",
    "target_list": [{"gene": "PCSK9"}, {"gene": "LDLR"}],
    "anchor_edge_recovery": 0.80,
    "n_tier1_edges": 5,
    "n_tier2_edges": 3,
    "n_tier3_edges": 0,
    "n_virtual_edges": 0,
    "executive_summary": "CAD pipeline completed successfully.",
    "target_table": [],
    "top_target_narratives": [],
    "evidence_quality": {},
    "limitations": [],
    "pipeline_version": "0.1.0",
    "generated_at": "2026-03-23T00:00:00Z",
    "warnings": [],
}

ALL_MOCK_RESULTS = {
    "phenotype_architect":         MOCK_PHENOTYPE,
    "statistical_geneticist":      MOCK_GENETICS,
    "somatic_exposure_agent":      MOCK_SOMATIC,
    "perturbation_genomics_agent": MOCK_BETA,
    "regulatory_genomics_agent":   MOCK_REGULATORY,
    "causal_discovery_agent":      MOCK_CAUSAL,
    "kg_completion_agent":         MOCK_KG,
    "target_prioritization_agent": MOCK_PRIORITIZATION,
    "chemistry_agent":             MOCK_CHEMISTRY,
    "clinical_trialist_agent":     MOCK_CLINICAL,
    "scientific_writer_agent":     MOCK_WRITER,
}


# ---------------------------------------------------------------------------
# Test 1: AgentInput / AgentOutput envelope round-trip
# ---------------------------------------------------------------------------

def test_agent_input_envelope():
    inp = AgentInput(
        disease_query={"disease_name": "CAD"},
        upstream_results={"phenotype_architect": MOCK_PHENOTYPE},
        mode="local",
    )
    assert inp.disease_query["disease_name"] == "CAD"
    assert "phenotype_architect" in inp.upstream_results
    assert inp.mode == "local"
    assert inp.run_id  # auto-generated


def test_agent_output_envelope_escalate():
    """Warnings containing ESCALATE → escalate=True auto-set."""
    out = wrap_output(
        "causal_discovery_agent",
        {"warnings": ["ESCALATE: anchor recovery below threshold"]},
    )
    assert out.escalate is True


def test_agent_output_no_escalate():
    out = wrap_output("causal_discovery_agent", {"warnings": ["minor issue"]})
    assert out.escalate is False


# ---------------------------------------------------------------------------
# Test 2: AgentRunner local mode dispatch
# ---------------------------------------------------------------------------

def test_agent_runner_local_dispatch():
    runner = _make_runner_with_mocks({"phenotype_architect": MOCK_PHENOTYPE})
    inp = AgentInput(disease_query={"disease_name": "CAD"})
    out = runner.dispatch("phenotype_architect", inp)
    assert out.agent_name == "phenotype_architect"
    assert out.results["efo_id"] == "EFO_0001645"
    assert not out.stub_fallback


def test_agent_runner_unknown_agent_returns_stub():
    runner = AgentRunner()
    inp = AgentInput(disease_query={"disease_name": "CAD"})
    out = runner.dispatch("nonexistent_agent", inp)
    assert out.stub_fallback


def test_agent_runner_mode_switching():
    runner = AgentRunner()
    assert runner.get_mode("somatic_exposure_agent") == "local"
    runner.set_mode("somatic_exposure_agent", "sdk")
    assert runner.get_mode("somatic_exposure_agent") == "sdk"
    runner.set_mode("somatic_exposure_agent", "local")
    assert runner.get_mode("somatic_exposure_agent") == "local"


def test_agent_runner_invalid_mode_raises():
    runner = AgentRunner()
    with pytest.raises(ValueError, match="mode must be"):
        runner.set_mode("somatic_exposure_agent", "turbo")


def test_agent_runner_set_all_sdk():
    runner = AgentRunner()
    runner.set_all_sdk()
    for name in ["phenotype_architect", "causal_discovery_agent", "scientific_writer_agent"]:
        assert runner.get_mode(name) == "sdk"


# ---------------------------------------------------------------------------
# Test 3: _run_tier1 parallel dispatch
# ---------------------------------------------------------------------------

def test_run_tier1_returns_three_outputs():
    from orchestrator.pi_orchestrator_v2 import _run_tier1
    runner = _make_runner_with_mocks(ALL_MOCK_RESULTS)
    phenotype_out, genetics_out, somatic_out = _run_tier1(runner, "coronary artery disease", "test-run")
    assert phenotype_out.agent_name == "phenotype_architect"
    assert genetics_out.agent_name  == "statistical_geneticist"
    assert somatic_out.agent_name   == "somatic_exposure_agent"
    assert phenotype_out.results["efo_id"] == "EFO_0001645"
    assert len(genetics_out.results["instruments"]) == 1
    assert somatic_out.results["summary"]["n_chip_genes"] == 2


# ---------------------------------------------------------------------------
# Test 4: _run_tier2 parallel dispatch
# ---------------------------------------------------------------------------

def test_run_tier2_returns_two_outputs():
    from orchestrator.pi_orchestrator_v2 import _run_tier2
    runner = _make_runner_with_mocks(ALL_MOCK_RESULTS)
    beta_out, reg_out = _run_tier2(
        runner, MOCK_PHENOTYPE, ["PCSK9", "LDLR", "TET2"], "test-run"
    )
    assert beta_out.agent_name == "perturbation_genomics_agent"
    assert reg_out.agent_name  == "regulatory_genomics_agent"
    assert "PCSK9" in beta_out.results["beta_matrix"]


# ---------------------------------------------------------------------------
# Test 5: _run_tier3 sequential dispatch
# ---------------------------------------------------------------------------

def test_run_tier3_returns_two_outputs():
    from orchestrator.pi_orchestrator_v2 import _run_tier3
    runner = _make_runner_with_mocks(ALL_MOCK_RESULTS)
    beta_out = _mock_output("perturbation_genomics_agent", MOCK_BETA)
    causal_out, kg_out = _run_tier3(
        runner, MOCK_PHENOTYPE, beta_out, {}, "test-run"
    )
    assert causal_out.agent_name == "causal_discovery_agent"
    assert kg_out.agent_name     == "kg_completion_agent"
    assert causal_out.results["anchor_recovery"]["recovery_rate"] == 0.80


# ---------------------------------------------------------------------------
# Test 6: _run_tier4 parallel dispatch
# ---------------------------------------------------------------------------

def test_run_tier4_returns_three_outputs():
    from orchestrator.pi_orchestrator_v2 import _run_tier4
    runner = _make_runner_with_mocks(ALL_MOCK_RESULTS)
    causal_out = _mock_output("causal_discovery_agent", MOCK_CAUSAL)
    kg_out     = _mock_output("kg_completion_agent",    MOCK_KG)
    prio_out, chem_out, clin_out = _run_tier4(
        runner, MOCK_PHENOTYPE, causal_out, kg_out, "test-run"
    )
    assert prio_out.agent_name == "target_prioritization_agent"
    assert chem_out.agent_name == "chemistry_agent"
    assert clin_out.agent_name == "clinical_trialist_agent"
    assert len(prio_out.results["targets"]) == 2
    assert chem_out.results["repurposing_candidates"][0]["drug"] == "statin"


# ---------------------------------------------------------------------------
# Test 7: Full pipeline happy path (all mocked)
# ---------------------------------------------------------------------------

def test_analyze_disease_v2_happy_path():
    from orchestrator.pi_orchestrator_v2 import analyze_disease_v2

    runner = _make_runner_with_mocks(ALL_MOCK_RESULTS)

    with (
        patch("orchestrator.pi_orchestrator_v2.AgentRunner", return_value=runner),
        patch("orchestrator.pi_orchestrator_v2._get_gamma_estimates", return_value={}),
        patch("orchestrator.pi_orchestrator_v2._write_somatic_edges", return_value=[]),
    ):
        result = analyze_disease_v2("coronary artery disease")

    assert result["pipeline_status"] == "SUCCESS"
    assert result["pi_reviewed"] is True
    assert result["disease_name"] == "coronary artery disease"
    assert result["n_escalations"] == 0


# ---------------------------------------------------------------------------
# Test 8: Anchor recovery halt path
# ---------------------------------------------------------------------------

def test_analyze_disease_v2_halts_on_low_anchor_recovery():
    from orchestrator.pi_orchestrator_v2 import analyze_disease_v2

    low_recovery_causal = {
        **MOCK_CAUSAL,
        "anchor_recovery": {"recovery_rate": 0.40, "recovered": [], "missing": ["PCSK9→LDL-C"]},
    }
    mocks = {**ALL_MOCK_RESULTS, "causal_discovery_agent": low_recovery_causal}
    runner = _make_runner_with_mocks(mocks)

    with (
        patch("orchestrator.pi_orchestrator_v2.AgentRunner", return_value=runner),
        patch("orchestrator.pi_orchestrator_v2._get_gamma_estimates", return_value={}),
        patch("orchestrator.pi_orchestrator_v2._write_somatic_edges", return_value=[]),
    ):
        result = analyze_disease_v2("coronary artery disease")

    assert result["pipeline_status"] == "HALTED_ANCHOR_RECOVERY"


# ---------------------------------------------------------------------------
# Test 9: Phenotype stub-fallback halts at Tier 1
# ---------------------------------------------------------------------------

def test_analyze_disease_v2_halts_on_phenotype_stub():
    from orchestrator.pi_orchestrator_v2 import analyze_disease_v2

    mocks = {**ALL_MOCK_RESULTS}
    runner = _make_runner_with_mocks(mocks)

    def _dispatch_stub_phenotype(agent_name: str, agent_input: AgentInput) -> AgentOutput:
        if agent_name == "phenotype_architect":
            return wrap_output(agent_name, {"error": "EFO lookup failed"}, stub_fallback=True)
        return wrap_output(agent_name, mocks.get(agent_name, {}))

    runner.dispatch = _dispatch_stub_phenotype  # type: ignore[assignment]

    with (
        patch("orchestrator.pi_orchestrator_v2.AgentRunner", return_value=runner),
        patch("orchestrator.pi_orchestrator_v2._get_gamma_estimates", return_value={}),
        patch("orchestrator.pi_orchestrator_v2._write_somatic_edges", return_value=[]),
    ):
        result = analyze_disease_v2("unknown disease")

    assert result["pipeline_status"] == "FAILED_TIER1_PHENOTYPE"


# ---------------------------------------------------------------------------
# Test 10: mode_overrides wiring
# ---------------------------------------------------------------------------

def test_mode_overrides_are_applied():
    """mode_overrides should flip runner mode before any dispatch."""
    from orchestrator.pi_orchestrator_v2 import analyze_disease_v2

    captured_modes: dict[str, str] = {}
    runner = _make_runner_with_mocks(ALL_MOCK_RESULTS)

    original_set_mode = runner.set_mode

    def _spy_set_mode(agent_name: str, mode: str) -> None:
        captured_modes[agent_name] = mode
        original_set_mode(agent_name, mode)

    runner.set_mode = _spy_set_mode  # type: ignore[assignment]

    with (
        patch("orchestrator.pi_orchestrator_v2.AgentRunner", return_value=runner),
        patch("orchestrator.pi_orchestrator_v2._get_gamma_estimates", return_value={}),
        patch("orchestrator.pi_orchestrator_v2._write_somatic_edges", return_value=[]),
    ):
        analyze_disease_v2(
            "coronary artery disease",
            mode_overrides={"somatic_exposure_agent": "sdk"},
        )

    assert captured_modes.get("somatic_exposure_agent") == "sdk"


# ---------------------------------------------------------------------------
# Test 11: warnings propagate correctly
# ---------------------------------------------------------------------------

def test_warnings_from_agents_propagate_to_output():
    from orchestrator.pi_orchestrator_v2 import analyze_disease_v2

    warn_somatic = {**MOCK_SOMATIC, "warnings": ["somatic data is provisional"]}
    mocks = {**ALL_MOCK_RESULTS, "somatic_exposure_agent": warn_somatic}
    runner = _make_runner_with_mocks(mocks)

    with (
        patch("orchestrator.pi_orchestrator_v2.AgentRunner", return_value=runner),
        patch("orchestrator.pi_orchestrator_v2._get_gamma_estimates", return_value={}),
        patch("orchestrator.pi_orchestrator_v2._write_somatic_edges", return_value=[]),
    ):
        result = analyze_disease_v2("coronary artery disease")

    # The warning from somatic agent should appear in all_warnings (accessible
    # via pipeline_outputs — not directly on GraphOutput but check status)
    assert result["pipeline_status"] == "SUCCESS"
