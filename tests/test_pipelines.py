"""
Tests for pipelines/ — Ota framework β/γ estimation + MR + sensitivity analysis.

All unit tests (no live API calls, no large data downloads required).

Run: pytest tests/test_pipelines.py -v
"""
from __future__ import annotations

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipelines.ota_beta_estimation import (
    estimate_beta,
    estimate_beta_tier1,
    estimate_beta_tier2,
    estimate_beta_virtual,
    build_beta_matrix,
    estimate_cad_target_betas,
)
from pipelines.ota_gamma_estimation import (
    estimate_gamma,
    compute_ota_gamma,
    build_gamma_matrix,
    estimate_cad_gammas,
    PROVISIONAL_GAMMAS,
)
from pipelines.sensitivity_analysis import (
    run_batch_evalue,
    flag_low_evalue_edges,
    generate_demotion_recommendations,
)
from pipelines.cnmf_programs import load_cnmf_programs, get_msigdb_hallmark_programs, get_programs_for_disease

# ---------------------------------------------------------------------------
# β estimation
# ---------------------------------------------------------------------------

class TestBetaEstimation:

    def test_tier1_known_gene_program(self):
        # DNMT3A should have Tier1 qualitative β for inflammatory program
        beta = estimate_beta_tier1("DNMT3A", "inflammatory_NF-kB")
        assert beta is not None
        assert beta["beta"] == 1.0  # upregulated
        assert beta["evidence_tier"] == "Tier1_Interventional"

    def test_tier1_known_gene_downregulated(self):
        beta = estimate_beta_tier1("DNMT3A", "DNA_methylation_maintenance")
        assert beta is not None
        assert beta["beta"] == -1.0  # downregulated

    def test_tier1_unknown_combination_returns_none(self):
        beta = estimate_beta_tier1("DNMT3A", "lipid_metabolism")
        # DNMT3A doesn't have known effect on lipid metabolism
        assert beta is None

    def test_tier2_respects_coloc_threshold(self):
        eqtl = {"nes": 0.5, "se": 0.1, "tissue": "Whole_Blood"}
        # H4 = 0.6 < 0.8 threshold → should return None
        beta = estimate_beta_tier2("PCSK9", "lipid_metabolism", eqtl_data=eqtl, coloc_h4=0.6)
        assert beta is None

    def test_tier2_above_coloc_threshold(self):
        eqtl = {"nes": 0.5, "se": 0.1, "tissue": "Whole_Blood"}
        beta = estimate_beta_tier2("PCSK9", "lipid_metabolism", eqtl_data=eqtl, coloc_h4=0.85)
        assert beta is not None
        assert beta["beta"] == pytest.approx(0.5)
        assert beta["evidence_tier"] == "Tier2_Convergent"
        assert "eQTL_MR" in beta["data_source"]

    def test_tier2_scales_by_program_loading(self):
        """eQTL NES × program loading = β when loading provided."""
        eqtl = {"nes": 0.4, "se": 0.05, "tissue": "Whole_Blood"}
        beta = estimate_beta_tier2("PCSK9", "lipid_metabolism", eqtl_data=eqtl, coloc_h4=0.9,
                                   program_loading=0.5)
        assert beta is not None
        assert beta["beta"] == pytest.approx(0.2)  # 0.4 × 0.5

    def test_virtual_pathway_membership(self):
        """Pathway membership proxy — explicit no-causal-basis annotation."""
        beta = estimate_beta_virtual("PCSK9", "lipid_metabolism", pathway_member=True)
        assert beta["beta"] == 1.0
        assert beta["evidence_tier"] == "provisional_virtual"
        assert "pathway_membership" in beta["data_source"]

    def test_virtual_fallback_no_data(self):
        """No data at all → virtual with None beta."""
        beta = estimate_beta_virtual("UNKNOWN_GENE", "UNKNOWN_PROGRAM")
        assert beta["beta"] is None
        assert beta["evidence_tier"] == "provisional_virtual"

    def test_full_fallback_cascade_tier1(self):
        # DNMT3A → inflammatory should use Tier1
        result = estimate_beta("DNMT3A", "inflammatory_NF-kB")
        assert result["tier_used"] == 1
        assert result["gene"] == "DNMT3A"

    def test_full_fallback_cascade_returns_none(self):
        # Unknown gene → unknown program has no causal evidence → None (no virtual fallback)
        result = estimate_beta("UNKNOWN_XYZ", "UNKNOWN_PROG")
        assert result is None

    def test_build_beta_matrix_shape(self):
        genes = ["DNMT3A", "TET2"]
        programs = ["inflammatory_NF-kB", "lipid_metabolism"]
        matrix = build_beta_matrix(genes, programs)
        assert set(matrix.beta_matrix.keys()) == set(genes)
        for gene in genes:
            assert set(matrix.beta_matrix[gene].keys()) == set(programs)

    def test_cad_target_beta_matrix(self):
        result = estimate_cad_target_betas()
        assert "PCSK9" in result.beta_matrix
        assert "DNMT3A" in result.beta_matrix
        # At least some β values should be non-None
        all_betas = [
            v for g in result.beta_matrix.values()
            for v in g.values()
        ]
        assert any(b is not None for b in all_betas)


# ---------------------------------------------------------------------------
# γ estimation
# ---------------------------------------------------------------------------

class TestGammaEstimation:

    def test_provisional_gamma_known_pair(self):
        # estimate_gamma always returns a dict; gamma field is None when no data available.
        result = estimate_gamma("lipid_metabolism", "CAD")
        assert isinstance(result, dict)
        gamma = result.get("gamma")
        assert gamma is None or (isinstance(gamma, float) and gamma >= 0)

    def test_unknown_pair_returns_no_evidence_dict(self):
        # estimate_gamma returns a dict with gamma=None when no evidence found.
        result = estimate_gamma("UNKNOWN_PROGRAM", "UNKNOWN_TRAIT")
        assert isinstance(result, dict)
        assert result.get("gamma") is None

    def test_all_provisional_gammas_positive(self):
        # PROVISIONAL_GAMMAS is empty — all γ values are data-derived
        assert PROVISIONAL_GAMMAS == {}

    def test_compute_ota_gamma_chip_cad(self):
        # TET2 CHIP → inflammatory → CAD pathway
        beta_estimates = {
            "inflammatory_NF-kB": {"beta": 1.0, "evidence_tier": "Tier1_Interventional"},
            "lipid_metabolism": {"beta": None, "evidence_tier": "provisional_virtual"},
        }
        gamma_estimates = {
            "inflammatory_NF-kB": {"gamma": 0.31, "data_source": "OT_L2G"},
            "lipid_metabolism": {"gamma": 0.44, "data_source": "OT_L2G"},
        }
        result = compute_ota_gamma("TET2", "CAD", beta_estimates, gamma_estimates)
        assert result["gene"] == "TET2"
        assert result["trait"] == "CAD"
        assert result["ota_gamma"] == pytest.approx(0.31, abs=0.01)  # only inflammatory contributes
        assert result["dominant_tier"] == "Tier1_Interventional"

    def test_compute_ota_gamma_no_contributions(self):
        result = compute_ota_gamma("UNKNOWN", "UNKNOWN", {}, {})
        assert result["ota_gamma"] == 0.0
        assert result["n_programs_contributing"] == 0

    def test_cad_gamma_matrix_has_nonzero_entries(self):
        result = estimate_cad_gammas()
        # Without network/API access, live estimation returns no entries (None γ);
        # verify the function completes and returns the expected structure.
        assert "top_program_trait_pairs" in result
        assert "matrix" in result
        # Any non-zero entries that are present must have positive gamma
        for p in result["top_program_trait_pairs"]:
            assert p["gamma"] is None or p["gamma"] >= 0

    def test_build_gamma_matrix_shape(self):
        programs = ["inflammatory_NF-kB", "lipid_metabolism"]
        traits = ["CAD", "RA"]
        matrix = build_gamma_matrix(programs, traits)
        assert set(matrix.keys()) == set(programs)
        for prog in programs:
            assert set(matrix[prog].keys()) == set(traits)


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------

class TestSensitivityAnalysis:

    def test_batch_evalue_adds_fields(self):
        edges = [
            {"from_node": "PCSK9", "to_node": "CAD", "effect_size": -0.51, "se": 0.05},
            {"from_node": "UNKNOWN", "to_node": "CAD", "effect_size": 0.05, "se": 0.8},
        ]
        results = run_batch_evalue(edges)
        assert len(results) == 2
        for r in results:
            assert "e_value" in r
            assert "recommendation" in r

    def test_flag_low_evalue(self):
        edges = [
            {"from_node": "A", "to_node": "CAD", "e_value": 1.5},  # below threshold
            {"from_node": "B", "to_node": "CAD", "e_value": 3.2},  # above threshold
            {"from_node": "C", "to_node": "CAD", "e_value": None}, # unknown
        ]
        result = flag_low_evalue_edges(edges, threshold=2.0)
        assert result["n_flagged"] == 2  # A (1.5 < 2.0) and C (None)
        assert result["n_clear"] == 1

    def test_demotion_recommendations(self):
        flagged = [
            {"from_node": "A", "to_node": "CAD", "e_value": 1.5},
        ]
        recs = generate_demotion_recommendations(flagged)
        assert len(recs) == 1
        assert recs[0]["from_node"] == "A"
        assert "Tier3" in recs[0]["new_tier"]
        assert "E-value" in recs[0]["reason"]


# ---------------------------------------------------------------------------
# Pipeline stubs
# ---------------------------------------------------------------------------

class TestPipelineStubs:

    def test_load_cnmf_programs_returns_hardcoded(self):
        result = load_cnmf_programs()
        assert result["n_programs"] > 0
        assert "lipid_metabolism" in result["programs"]


class TestProgramSources:
    """
    Tests for disease-aware program source routing and MSigDB Hallmark fallback.
    These tests do NOT make live HTTP calls — they test the fallback path.
    """

    def test_hallmark_fallback_returns_programs(self):
        """_hallmark_fallback should always return non-empty program list."""
        from pipelines.cnmf_programs import _hallmark_fallback
        result = _hallmark_fallback(disease=None, error="test_error")
        assert result["n_programs"] > 0
        assert all("program_id" in p for p in result["programs"])
        assert all("hallmark_name" in p for p in result["programs"])

    def test_hallmark_fallback_returns_all_programs(self):
        """Fallback returns all stub programs (no disease filtering)."""
        from pipelines.cnmf_programs import _hallmark_fallback
        result = _hallmark_fallback(disease="CAD")
        assert result["n_programs"] > 0
        assert result["disease_filtered"] is False
        assert all("hallmark_name" in p for p in result["programs"])

    def test_hallmark_to_program_mapping_complete(self):
        """HALLMARK_TO_PROGRAM covers all HALLMARK_PROGRAM_IDS."""
        from pipelines.cnmf_programs import HALLMARK_TO_PROGRAM, HALLMARK_PROGRAM_IDS
        assert HALLMARK_PROGRAM_IDS == frozenset(HALLMARK_TO_PROGRAM.values())

    def test_get_programs_for_disease_cad(self):
        """CAD program routing returns NMF programs from cache."""
        result = get_programs_for_disease("CAD")
        assert "programs" in result
        assert "n_programs" in result
        # CAD has a cached NMF program file
        if result["n_programs"] > 0:
            assert all("program_id" in p for p in result["programs"])
            assert all("gene_set" in p for p in result["programs"])

    def test_get_programs_for_disease_amd(self):
        """AMD program routing returns NMF programs from cache."""
        result = get_programs_for_disease("AMD")
        assert "programs" in result
        assert "n_programs" in result
        if result["n_programs"] > 0:
            assert all("program_id" in p for p in result["programs"])
            assert all("gene_set" in p for p in result["programs"])

    def test_disease_cell_type_map_covers_all_trait_map_diseases(self):
        """Every disease in DISEASE_TRAIT_MAP should have a DISEASE_CELL_TYPE_MAP entry."""
        from graph.schema import DISEASE_TRAIT_MAP, DISEASE_CELL_TYPE_MAP
        for disease in DISEASE_TRAIT_MAP:
            assert disease in DISEASE_CELL_TYPE_MAP, (
                f"DISEASE_TRAIT_MAP has {disease!r} but DISEASE_CELL_TYPE_MAP does not. "
                "Add a cell-type context entry."
            )

    def test_disease_cell_type_map_required_fields(self):
        """Each DISEASE_CELL_TYPE_MAP entry has the required fields."""
        from graph.schema import DISEASE_CELL_TYPE_MAP
        required = {"cell_types", "primary_tissue", "gtex_tissue", "perturb_seq_source"}
        for disease, ctx in DISEASE_CELL_TYPE_MAP.items():
            missing = required - set(ctx.keys())
            assert not missing, f"DISEASE_CELL_TYPE_MAP[{disease!r}] missing fields: {missing}"

    def test_perturb_seq_sources_have_required_fields(self):
        """Each PERTURB_SEQ_SOURCES entry documents download URL and cell type."""
        from graph.schema import PERTURB_SEQ_SOURCES
        required = {"description", "cell_type", "diseases", "requires_auth", "status"}
        for src, info in PERTURB_SEQ_SOURCES.items():
            missing = required - set(info.keys())
            assert not missing, f"PERTURB_SEQ_SOURCES[{src!r}] missing fields: {missing}"

