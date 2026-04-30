"""
tests/test_pi_orchestrator_v2.py — Unit tests for pi_orchestrator_v2.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Shared mock result dicts
# ---------------------------------------------------------------------------

MOCK_PHENOTYPE = {
    "disease_name": "coronary artery disease",
    "efo_id": "EFO_0001645",
    "modifier_types": ["germline", "somatic_chip", "drug"],
    "primary_gwas_id": "ieu-a-7",
    "n_gwas_studies": 5,
    "use_precomputed_only": True,
    "day_one_mode": True,
}

# ---------------------------------------------------------------------------
# pi_orchestrator_v2 — direct function chain tests
# ---------------------------------------------------------------------------

def _writer_minimal_graph_output() -> dict:
    return {
        "disease_name":            "coronary artery disease",
        "efo_id":                  "EFO_0001645",
        "target_list":             [],
        "n_tier1_edges":           0,
        "n_tier2_edges":           0,
        "n_tier3_edges":           0,
        "n_virtual_edges":         0,
        "executive_summary":       "",
        "target_table":            "",
        "top_target_narratives":   [],
        "evidence_quality":        {},
        "limitations":             "",
        "warnings":                [],
        "pipeline_version":        "0.1.0",
        "generated_at":            "2026-01-01T00:00:00Z",
    }


def test_analyze_disease_v2_happy_path(tmp_path):
    """Smoke test: v2 completes with all tier run() functions mocked."""
    from orchestrator.pi_orchestrator_v2 import analyze_disease_v2

    dq = {
        "disease_name": "coronary artery disease",
        "efo_id":       "EFO_0001645",
        "stub_fallback": False,
    }
    gen = {"instruments": [], "anchor_genes_validated": {}, "warnings": []}
    beta = {
        "beta_matrix": {}, "evidence_tier_per_gene": {},
        "n_tier1": 0, "n_tier2": 0, "n_tier3": 0, "n_virtual": 0,
        "programs": [], "warnings": [],
    }
    reg = {"gene_eqtl_summary": {}, "gene_program_overlap": {}, "tier2_upgrades": [], "warnings": []}
    causal = {"n_edges_written": 0, "top_genes": [], "warnings": [], "edges_written": []}
    kg = {
        "n_pathway_edges_added": 0, "n_ppi_edges_added": 0,
        "n_drug_target_edges_added": 0, "n_primekg_edges_added": 0, "warnings": [],
    }
    prio = {
        "targets": [
            {
                "target_gene": "PCSK9", "rank": 1, "evidence_tier": "Tier1_Interventional",
                "safety_flags": [],
            },
        ],
        "warnings": [],
    }
    chem = {
        "target_chemistry": {}, "repurposing_candidates": [], "gps_disease_reversers": [],
        "gps_program_reversers": {}, "gps_programs_screened": [], "gps_priority_compounds": [],
        "warnings": [],
    }
    clin = {"trial_summary": {}, "key_trials": [], "development_risk": {}, "warnings": []}
    gamma_stub = {"lipid_metabolism": {"CAD": {"gamma": 0.1, "evidence_tier": "Tier2"}}}

    mock_chem = MagicMock(return_value=chem)

    with (
        patch("steps.tier1_phenomics.disease_query_builder.run", return_value=dq),
        patch("steps.tier1_phenomics.gwas_anchor_validator.run", return_value=gen),
        patch(
            "orchestrator.pi_orchestrator_v2._collect_gene_list",
            return_value=(["PCSK9"], {}, {"targets": [], "source": "test", "efo_id": "EFO_0001645"}),
        ),
        patch("orchestrator.pi_orchestrator_v2._get_gamma_estimates", return_value=gamma_stub),
        patch("steps.tier2_pathway.beta_matrix_builder.run", return_value=beta),
        patch("steps.tier2_pathway.eqtl_coloc_mapper.run", return_value=reg),
        patch("steps.tier3_causal.ota_gamma_calculator.run", return_value=causal),
        patch("steps.tier3_causal.drug_target_graph_enricher.run", return_value=kg),
        patch("steps.tier4_translation.target_ranker.run", return_value=dict(prio)),
        patch("steps.tier4_translation.gps_compound_screener.run", mock_chem),
        patch("steps.tier4_translation.trial_landscape_mapper.run", return_value=clin),
        patch("steps.tier5_writer.report_builder.run", return_value=_writer_minimal_graph_output()),
    ):
        result = analyze_disease_v2("coronary artery disease", _ckpt_dir=tmp_path)

    assert result["pipeline_status"] == "SUCCESS"
    assert result["pi_reviewed"] is True
    assert result["disease_name"] == "coronary artery disease"
    mock_chem.assert_called_once()
    chem_prio = mock_chem.call_args[0][0]
    assert "_gamma_estimates" in chem_prio
    assert chem_prio["_gamma_estimates"] == gamma_stub


def test_analyze_disease_v2_failed_tier1_phenotype():
    from orchestrator.pi_orchestrator_v2 import analyze_disease_v2

    stub_dq = {
        "disease_name": "unknown disease xyz",
        "stub_fallback": True,
    }

    with patch("steps.tier1_phenomics.disease_query_builder.run", return_value=stub_dq):
        result = analyze_disease_v2("unknown disease xyz")

    assert result["pipeline_status"] == "FAILED_TIER1_PHENOTYPE"
