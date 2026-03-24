"""
Tests for viral_somatic_server.py.

Split into:
  - Unit tests:        hardcoded tables, stub outputs, schema validation (always run)
  - Integration tests: real HTTP calls to CLUE API
                       marked with @pytest.mark.integration — skipped by default
                       run with: pytest -m integration

Run all:               pytest tests/test_viral_somatic_server.py -v
Run only unit:         pytest tests/test_viral_somatic_server.py -v -m "not integration"
Run only integration:  pytest tests/test_viral_somatic_server.py -v -m integration
"""
from __future__ import annotations

import math
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp_servers.viral_somatic_server import (
    get_chip_disease_associations,
    get_chip_gene_expression_effects,
    get_viral_gwas_summary_stats,
    get_viral_disease_mr_results,
    extract_viral_burden_from_wgs,
    run_viral_mr_analysis,
    get_cmap_drug_signatures,
    get_drug_exposure_mr,
    project_cmap_onto_programs,
)


# ---------------------------------------------------------------------------
# Unit tests — CHIP disease associations
# ---------------------------------------------------------------------------

class TestChipDiseaseAssociations:

    def test_cad_associations_returns_data(self):
        result = get_chip_disease_associations("CAD")
        assert result["disease"] == "CAD"
        assert result["efo_id"] == "EFO_0001645"
        assert result["n_associations"] > 0
        assert len(result["associations"]) > 0
        assert isinstance(result["sources"], list)
        assert len(result["sources"]) > 0

    def test_cad_has_bick2020_and_kar2022(self):
        result = get_chip_disease_associations("CAD")
        sources = result["sources"]
        assert any("32694926" in s for s in sources), f"Expected Bick 2020 source, got: {sources}"
        assert any("35177839" in s for s in sources), f"Expected Kar 2022 source, got: {sources}"

    def test_log_effect_computed_for_hr(self):
        result = get_chip_disease_associations("CAD")
        # Bick 2020 uses HR — check log_effect = log(HR)
        bick_assocs = [a for a in result["associations"] if a.get("pmid") == "32694926"]
        assert len(bick_assocs) > 0
        for a in bick_assocs:
            assert "log_effect" in a
            assert a["effect_type"] == "log_HR"
            expected = round(math.log(a["hr"]), 4)
            assert abs(a["log_effect"] - expected) < 1e-3

    def test_log_effect_computed_for_or(self):
        result = get_chip_disease_associations("CAD")
        # Kar 2022 uses OR
        kar_assocs = [a for a in result["associations"] if a.get("pmid") == "35177839"]
        assert len(kar_assocs) > 0
        for a in kar_assocs:
            assert "log_effect" in a
            assert a["effect_type"] == "log_OR"
            expected = round(math.log(a["or"]), 4)
            assert abs(a["log_effect"] - expected) < 1e-3

    def test_driver_gene_filter_dnmt3a(self):
        result = get_chip_disease_associations("CAD", driver_genes=["DNMT3A"])
        genes = [a["gene"] for a in result["associations"]]
        # All returned genes should be DNMT3A or DNMT3A_large (VAF filter suffix)
        for g in genes:
            assert g.startswith("DNMT3A"), f"Unexpected gene in DNMT3A filter: {g}"

    def test_driver_gene_filter_tet2(self):
        result = get_chip_disease_associations("CAD", driver_genes=["TET2"])
        assert result["n_associations"] > 0
        for a in result["associations"]:
            assert a["gene"].startswith("TET2")

    def test_tet2_has_stronger_effect_than_dnmt3a(self):
        # Published biology: TET2 CHIP has stronger CAD effect than DNMT3A (Bick 2020)
        dnmt3a = get_chip_disease_associations("CAD", driver_genes=["DNMT3A"])
        tet2 = get_chip_disease_associations("CAD", driver_genes=["TET2"])
        # TET2 HR from Bick 2020 = 1.72, DNMT3A HR = 1.26
        tet2_bick = [a for a in tet2["associations"] if a.get("pmid") == "32694926" and "hr" in a]
        dnmt3a_bick = [a for a in dnmt3a["associations"] if a.get("pmid") == "32694926" and "hr" in a]
        assert len(tet2_bick) > 0
        assert len(dnmt3a_bick) > 0
        assert max(a["hr"] for a in tet2_bick) > max(a["hr"] for a in dnmt3a_bick)

    def test_unknown_disease_returns_empty(self):
        result = get_chip_disease_associations("NONEXISTENT_DISEASE_XYZ")
        assert result["n_associations"] == 0
        assert result["associations"] == []

    def test_heart_failure_returns_data(self):
        result = get_chip_disease_associations("heart_failure")
        assert result["n_associations"] > 0

    def test_result_has_required_schema_keys(self):
        result = get_chip_disease_associations("CAD")
        required = ["disease", "efo_id", "n_associations", "associations", "sources", "note"]
        for k in required:
            assert k in result, f"Missing required key: {k}"


# ---------------------------------------------------------------------------
# Unit tests — CHIP gene expression effects (stub)
# ---------------------------------------------------------------------------

class TestChipGeneExpressionEffects:

    def test_dnmt3a_known_programs(self):
        result = get_chip_gene_expression_effects(["DNMT3A"])
        assert "DNMT3A" in result["effects"]
        effects = result["effects"]["DNMT3A"]
        assert "programs_upregulated" in effects
        assert "programs_downregulated" in effects
        assert any("inflammatory" in p or "NF-kB" in p for p in effects["programs_upregulated"])

    def test_tet2_known_programs(self):
        result = get_chip_gene_expression_effects(["TET2"])
        effects = result["effects"]["TET2"]
        assert "IL-6_signaling" in effects["programs_upregulated"] or \
               any("IL" in p for p in effects["programs_upregulated"])

    def test_unknown_gene_returns_stub_note(self):
        result = get_chip_gene_expression_effects(["NONEXISTENT_GENE"])
        assert "NONEXISTENT_GENE" in result["effects"]
        note = result["effects"]["NONEXISTENT_GENE"].get("note", "")
        assert "NONEXISTENT_GENE" in note or "not a known" in note

    def test_case_insensitive_gene_lookup(self):
        result = get_chip_gene_expression_effects(["dnmt3a"])
        # Gene names are uppercased internally
        assert "dnmt3a" in result["effects"]
        assert "programs_upregulated" in result["effects"]["dnmt3a"]

    def test_stub_note_present(self):
        result = get_chip_gene_expression_effects(["DNMT3A"])
        assert "STUB" in result["note"]
        assert "download_url" in result

    def test_multiple_genes(self):
        result = get_chip_gene_expression_effects(["DNMT3A", "TET2", "ASXL1"])
        assert len(result["genes"]) == 3
        assert all(g in result["effects"] for g in ["DNMT3A", "TET2", "ASXL1"])


# ---------------------------------------------------------------------------
# Unit tests — Viral GWAS summary stats
# ---------------------------------------------------------------------------

class TestViralGwasSummaryStats:

    def test_ebv_returns_full_data(self):
        result = get_viral_gwas_summary_stats("EBV")
        assert result["virus"] == "EBV"
        assert result["gwas_available"] is True
        assert "mr_results" in result
        assert len(result["mr_results"]) > 0

    def test_ebv_mr_results_contain_autoimmune(self):
        result = get_viral_gwas_summary_stats("EBV")
        traits = [r["trait"] for r in result["mr_results"]]
        assert "RA" in traits
        assert "SLE" in traits
        assert "MS" in traits

    def test_cmv_returns_stub(self):
        result = get_viral_gwas_summary_stats("CMV")
        assert result["gwas_available"] is True
        assert "note" in result or "STUB" in str(result)

    def test_covid19_returns_data(self):
        result = get_viral_gwas_summary_stats("COVID19")
        assert result["gwas_available"] is True

    def test_unknown_virus_returns_not_available(self):
        result = get_viral_gwas_summary_stats("UNKNOWN_VIRUS_XYZ")
        assert result["gwas_available"] is False

    def test_case_insensitive_virus(self):
        result_upper = get_viral_gwas_summary_stats("EBV")
        result_lower = get_viral_gwas_summary_stats("ebv")
        assert result_upper["gwas_available"] == result_lower["gwas_available"]


# ---------------------------------------------------------------------------
# Unit tests — Viral disease MR results
# ---------------------------------------------------------------------------

class TestViralDiseaseMrResults:

    def test_ebv_all_traits(self):
        result = get_viral_disease_mr_results("EBV")
        assert result["virus"] == "EBV"
        assert len(result["results"]) == 5  # Nyeo 2026 has 5 associations
        for r in result["results"]:
            assert "mr_beta" in r
            assert "mr_se" in r
            assert "mr_p" in r

    def test_ebv_filter_by_trait_ra(self):
        result = get_viral_disease_mr_results("EBV", trait="RA")
        assert len(result["results"]) == 1
        assert result["results"][0]["trait"] == "RA"
        # Nyeo 2026 RA: beta = 0.48
        assert abs(result["results"][0]["mr_beta"] - 0.48) < 0.01

    def test_ebv_filter_by_trait_ms(self):
        result = get_viral_disease_mr_results("EBV", trait="MS")
        assert len(result["results"]) == 1
        assert result["results"][0]["mr_beta"] > 0  # EBV → MS is positive

    def test_ebv_filter_nonexistent_trait(self):
        result = get_viral_disease_mr_results("EBV", trait="NONEXISTENT")
        assert result["results"] == []

    def test_non_ebv_returns_empty(self):
        result = get_viral_disease_mr_results("CMV")
        assert result["results"] == []
        assert "note" in result

    def test_ebv_all_mr_p_significant(self):
        # All Nyeo 2026 EBV MR results should be statistically significant
        result = get_viral_disease_mr_results("EBV")
        for r in result["results"]:
            assert r["mr_p"] < 0.01, f"Expected significant MR for {r['trait']}, got p={r['mr_p']}"


# ---------------------------------------------------------------------------
# Unit tests — Stub tools
# ---------------------------------------------------------------------------

class TestStubTools:

    def test_extract_viral_burden_stub_schema(self):
        result = extract_viral_burden_from_wgs("/path/to/file.bam", "EBV")
        assert result["bam_path"] == "/path/to/file.bam"
        assert result["virus"] == "EBV"
        assert result["copies_per_10k"] is None
        assert result["above_threshold"] is None
        assert result["threshold_copies"] == 1.2
        assert "STUB" in result["note"]

    def test_run_viral_mr_stub_schema(self):
        result = run_viral_mr_analysis("EBV_burden", "ieu-a-7")
        assert result["viral_trait_id"] == "EBV_burden"
        assert result["outcome_id"] == "ieu-a-7"
        assert result["mr_ivw"] is None
        assert "note" in result

    def test_project_cmap_stub_schema(self):
        result = project_cmap_onto_programs(
            drug_signatures=[{"drug": "simvastatin", "vector": [1.0, 2.0]}],
            program_matrix={"program1": [0.1, 0.2]},
        )
        assert result["n_drugs"] == 1
        assert result["n_programs"] == 1
        assert "beta_matrix" in result
        assert "STUB" in result["note"]

    def test_project_cmap_empty_inputs(self):
        result = project_cmap_onto_programs(drug_signatures=[], program_matrix={})
        assert result["n_drugs"] == 0
        assert result["n_programs"] == 0


# ---------------------------------------------------------------------------
# Unit tests — Drug exposure MR
# ---------------------------------------------------------------------------

class TestDrugExposureMr:

    def test_hmgcr_cad_known_result(self):
        result = get_drug_exposure_mr("HMGCR_inhibition", "CAD")
        assert result["drug_mechanism"] == "HMGCR_inhibition"
        assert result["outcome_trait_id"] == "CAD"
        assert result["mr_beta"] < 0  # Statins protective → negative beta
        assert result["mr_p"] < 1e-10
        assert result["evidence_type"] == "drug_target_mr"

    def test_pcsk9_ldl_known_result(self):
        result = get_drug_exposure_mr("PCSK9_inhibition", "LDL-C")
        assert result["mr_beta"] < 0  # PCSK9 inhibition lowers LDL
        assert "source" in result

    def test_il6r_cad_known_result(self):
        result = get_drug_exposure_mr("IL6R_blockade", "CAD")
        assert result["mr_beta"] < 0  # IL-6R blockade protective

    def test_unknown_mechanism_returns_stub(self):
        result = get_drug_exposure_mr("UNKNOWN_DRUG", "CAD")
        assert result["mr_beta"] is None
        assert "note" in result
        assert "STUB" in result["note"]

    def test_all_known_drug_mrs_have_required_fields(self):
        known = [
            ("HMGCR_inhibition", "CAD"),
            ("HMGCR_inhibition", "LDL-C"),
            ("PCSK9_inhibition", "CAD"),
            ("PCSK9_inhibition", "LDL-C"),
            ("IL6R_blockade", "CRP"),
            ("IL6R_blockade", "CAD"),
        ]
        for mechanism, outcome in known:
            result = get_drug_exposure_mr(mechanism, outcome)
            assert result["mr_beta"] is not None, f"Expected MR beta for {mechanism} → {outcome}"
            assert "source" in result, f"Expected source for {mechanism} → {outcome}"


# ---------------------------------------------------------------------------
# Integration tests — CLUE REST API
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestCmapDrugSignaturesLive:
    """Live tests for CLUE.io REST API — no auth required for basic perts queries."""

    def test_simvastatin_found_in_clue(self):
        result = get_cmap_drug_signatures(["simvastatin"])
        assert result["drugs"] == ["simvastatin"]
        assert len(result["results"]) == 1
        drug = result["results"][0]
        assert drug["name"] == "simvastatin"
        # Should have found it in CLUE
        assert "pert_id" in drug or "note" in drug

    def test_statin_has_hmgcr_target(self):
        result = get_cmap_drug_signatures(["simvastatin"])
        drug = result["results"][0]
        if drug.get("pert_id"):  # Only check if found
            target = str(drug.get("target", "")).upper()
            moa = str(drug.get("moa", "")).upper()
            assert "HMGCR" in target or "HMG" in moa or "STATIN" in moa, \
                f"Expected HMGCR target for simvastatin, got target={drug.get('target')}, moa={drug.get('moa')}"

    def test_multiple_statins(self):
        result = get_cmap_drug_signatures(["simvastatin", "atorvastatin"])
        assert len(result["results"]) == 2
        names = [r["name"] for r in result["results"]]
        assert "simvastatin" in names
        assert "atorvastatin" in names

    def test_unknown_drug_handled_gracefully(self):
        result = get_cmap_drug_signatures(["NONEXISTENT_DRUG_XYZ_12345"])
        assert len(result["results"]) == 1
        drug = result["results"][0]
        # Should return a note, not crash
        assert "note" in drug or "error" in drug

    def test_result_has_required_schema_keys(self):
        result = get_cmap_drug_signatures(["simvastatin"])
        required = ["drugs", "cell_line", "results", "note", "data_source"]
        for k in required:
            assert k in result, f"Missing key: {k}"
