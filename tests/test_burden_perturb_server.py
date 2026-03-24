"""
Tests for burden_perturb_server.py.

All tests are unit tests (no live HTTP calls needed — data is hardcoded or stub).

Run: pytest tests/test_burden_perturb_server.py -v
"""
from __future__ import annotations

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp_servers.burden_perturb_server import (
    get_perturbseq_dataset_info,
    get_gene_perturbation_effect,
    get_perturbation_beta_matrix,
    load_perturbseq_h5ad,
    get_cnmf_program_info,
    run_cnmf_program_extraction,
    get_program_gene_loadings,
    run_inspre_causal_structure,
    run_scone_normalization,
    get_gene_burden_stats,
)


# ---------------------------------------------------------------------------
# Perturb-seq dataset metadata
# ---------------------------------------------------------------------------

class TestPerturbseqDatasetInfo:

    def test_k562_metadata(self):
        result = get_perturbseq_dataset_info("K562")
        assert result["cell_line"] == "K562"
        assert result["n_cells"] > 2_000_000
        assert result["n_perturbations"] > 9_000
        assert result["geo_accession"] == "GSE246756"
        assert "download_status" in result
        assert "download_url" in result

    def test_rpe1_metadata(self):
        result = get_perturbseq_dataset_info("RPE1")
        assert result["cell_line"] == "RPE1"
        assert result["n_cells"] > 1_000_000
        assert result["geo_accession"] == "GSE246756"

    def test_case_insensitive_lookup(self):
        result_upper = get_perturbseq_dataset_info("K562")
        result_lower = get_perturbseq_dataset_info("k562")
        assert result_upper["n_cells"] == result_lower["n_cells"]

    def test_unknown_cell_line_returns_error(self):
        result = get_perturbseq_dataset_info("NONEXISTENT_LINE")
        assert "error" in result

    def test_download_status_not_downloaded_without_local_data(self):
        result = get_perturbseq_dataset_info("K562")
        # In test environment, data won't be downloaded
        assert result["download_status"] in ("not_downloaded", "available")


# ---------------------------------------------------------------------------
# Gene perturbation effects
# ---------------------------------------------------------------------------

class TestGenePerturbationEffect:

    def test_pcsk9_known_effect(self):
        result = get_gene_perturbation_effect("PCSK9")
        assert result["gene"] == "PCSK9"
        assert result["data_tier"] == "qualitative"
        assert "top_programs_up" in result
        assert "top_programs_dn" in result
        assert any("LDLR" in p or "lipid" in p for p in result["top_programs_dn"])

    def test_tet2_known_effect(self):
        result = get_gene_perturbation_effect("TET2")
        assert "inflammatory" in str(result["top_programs_up"]) or \
               "NF-kB" in str(result["top_programs_up"])

    def test_dnmt3a_known_effect(self):
        result = get_gene_perturbation_effect("DNMT3A")
        assert result["data_tier"] == "qualitative"
        assert any("inflammatory" in p for p in result["top_programs_up"])

    def test_chip_driver_inflammatory_programs(self):
        # Both TET2 and DNMT3A should upregulate inflammatory programs
        for gene in ["TET2", "DNMT3A"]:
            result = get_gene_perturbation_effect(gene)
            up_progs = " ".join(result.get("top_programs_up", []))
            assert "inflammatory" in up_progs or "NF-kB" in up_progs or "IL" in up_progs, \
                f"{gene} should have inflammatory program upregulation"

    def test_unknown_gene_returns_stub(self):
        result = get_gene_perturbation_effect("NONEXISTENT_GENE_XYZ")
        assert result["data_tier"] == "stub"
        assert "note" in result
        assert "download_url" in result

    def test_program_filter(self):
        result = get_gene_perturbation_effect(
            "DNMT3A",
            programs=["inflammatory_NF-kB"]
        )
        # Only requested programs should be in the filtered lists
        for p in result.get("top_programs_up", []):
            assert p == "inflammatory_NF-kB"
        for p in result.get("top_programs_dn", []):
            assert p == "inflammatory_NF-kB"


# ---------------------------------------------------------------------------
# Beta matrix
# ---------------------------------------------------------------------------

class TestPerturbationBetaMatrix:

    def test_single_known_gene(self):
        result = get_perturbation_beta_matrix(["PCSK9"])
        assert "PCSK9" in result["beta_matrix"]
        assert result["n_genes_known"] == 1
        assert result["n_genes_stub"] == 0

    def test_mixed_known_unknown(self):
        result = get_perturbation_beta_matrix(["PCSK9", "NONEXISTENT_GENE"])
        assert result["n_genes_known"] == 1
        assert result["n_genes_stub"] == 1

    def test_beta_values_are_sign_only(self):
        result = get_perturbation_beta_matrix(["PCSK9", "TET2"])
        for gene, betas in result["beta_matrix"].items():
            for prog, val in betas.items():
                assert val in (1.0, -1.0, None), \
                    f"Expected ±1 or None for {gene}→{prog}, got {val}"

    def test_programs_list_populated(self):
        result = get_perturbation_beta_matrix(["PCSK9"])
        assert len(result["programs"]) > 0
        # All programs should be in CNMF_PROGRAM_REGISTRY
        assert "lipid_metabolism" in result["programs"]
        assert "inflammatory_NF-kB" in result["programs"]

    def test_pcsk9_lipid_program_negative(self):
        result = get_perturbation_beta_matrix(["PCSK9"])
        betas = result["beta_matrix"]["PCSK9"]
        # PCSK9 KO should downregulate LDLR_recycling / LDL_uptake
        # but those programs may not be in CNMF_PROGRAM_REGISTRY
        # Check that lipid_metabolism is not positive
        lipid_beta = betas.get("lipid_metabolism")
        assert lipid_beta in (None, -1.0)  # PCSK9 KO inhibits lipid uptake programs or unknown

    def test_empty_gene_list(self):
        result = get_perturbation_beta_matrix([])
        assert result["genes"] == []
        assert result["beta_matrix"] == {}


# ---------------------------------------------------------------------------
# h5ad loading (stub)
# ---------------------------------------------------------------------------

class TestLoadPerturbseqH5ad:

    def test_returns_not_downloaded_without_local_data(self):
        result = load_perturbseq_h5ad("K562")
        assert result["status"] in ("not_downloaded", "ready")
        if result["status"] == "not_downloaded":
            assert "geo_accession" in result
            assert "download_url" in result

    def test_has_approx_size(self):
        result = load_perturbseq_h5ad("K562")
        if result["status"] == "not_downloaded":
            assert "approx_size" in result
            assert "50GB" in result["approx_size"]


# ---------------------------------------------------------------------------
# cNMF program tools
# ---------------------------------------------------------------------------

class TestCnmfProgramInfo:

    def test_list_all_programs(self):
        result = get_cnmf_program_info()
        assert "programs" in result
        assert result["n_programs"] > 0
        assert "inflammatory_NF-kB" in result["programs"]
        assert "lipid_metabolism" in result["programs"]

    def test_specific_program_info(self):
        result = get_cnmf_program_info("lipid_metabolism")
        assert result["program_name"] == "lipid_metabolism"
        assert "top_genes" in result
        assert "LDLR" in result["top_genes"] or "PCSK9" in result["top_genes"]
        assert "ota_gamma" in result

    def test_inflammatory_program_has_nfkb_genes(self):
        result = get_cnmf_program_info("inflammatory_NF-kB")
        assert "NFKB1" in result["top_genes"] or "TNF" in result["top_genes"]

    def test_mhc_program_has_hla_genes(self):
        result = get_cnmf_program_info("MHC_class_II_presentation")
        assert any("HLA" in g for g in result["top_genes"])

    def test_unknown_program_returns_error(self):
        result = get_cnmf_program_info("NONEXISTENT_PROGRAM_XYZ")
        assert "error" in result

    def test_all_programs_have_ota_gamma(self):
        listing = get_cnmf_program_info()
        for prog_name in listing["programs"]:
            prog = get_cnmf_program_info(prog_name)
            assert "ota_gamma" in prog, f"Program {prog_name} missing ota_gamma"
            assert 0 <= prog["ota_gamma"] <= 1, f"ota_gamma out of range for {prog_name}"


# ---------------------------------------------------------------------------
# Gene loadings
# ---------------------------------------------------------------------------

class TestProgramGeneLoadings:

    def test_lipid_metabolism_top_genes(self):
        result = get_program_gene_loadings("lipid_metabolism", top_n=5)
        assert result["program_name"] == "lipid_metabolism"
        assert len(result["top_genes"]) <= 5
        assert result["n_returned"] <= 5

    def test_unknown_program_returns_error(self):
        result = get_program_gene_loadings("NONEXISTENT_PROGRAM")
        assert "error" in result


# ---------------------------------------------------------------------------
# Stub tools — inspre + SCONE
# ---------------------------------------------------------------------------

class TestStubTools:

    def test_inspre_stub_schema(self):
        result = run_inspre_causal_structure(
            beta_matrix={"PCSK9": {"lipid_metabolism": -1.0}},
            gamma_matrix={"lipid_metabolism": 0.44},
        )
        assert result["n_genes"] == 1
        assert result["n_programs"] == 1
        assert result["adjacency_matrix"] is None
        assert "STUB" in result["note"]

    def test_cnmf_extraction_stub_schema(self):
        result = run_cnmf_program_extraction("/path/to/k562.h5ad", k_programs=20)
        assert result["k_programs"] == 20
        assert result["status"] == "stub"
        assert result["programs_extracted"] is None
        assert "STUB" in result["note"]

    def test_scone_normalization_stub_schema(self):
        result = run_scone_normalization("/path/to/counts.h5ad", "K562")
        assert result["cell_line"] == "K562"
        assert result["normalized_path"] is None
        assert "STUB" in result["note"]


# ---------------------------------------------------------------------------
# Burden statistics
# ---------------------------------------------------------------------------

class TestGeneBurdenStats:

    def test_pcsk9_lof_burden(self):
        result = get_gene_burden_stats(["PCSK9"])
        pcsk9 = result["results"]["PCSK9"]
        assert pcsk9["lof_burden_or"] is not None
        assert pcsk9["lof_burden_or"] < 1.0  # LoF is protective for LDL
        assert pcsk9["trait"] == "LDL-C"

    def test_ldlr_lof_burden(self):
        result = get_gene_burden_stats(["LDLR"])
        ldlr = result["results"]["LDLR"]
        assert ldlr["lof_burden_or"] > 1.0  # LoF increases LDL (FH)
        assert ldlr["trait"] == "LDL-C"

    def test_chip_genes_in_registry(self):
        result = get_gene_burden_stats(["DNMT3A", "TET2"])
        for gene in ["DNMT3A", "TET2"]:
            assert result["results"][gene]["lof_burden_or"] is not None
            assert result["results"][gene]["trait"] == "CAD"

    def test_unknown_gene_returns_stub(self):
        result = get_gene_burden_stats(["NONEXISTENT_GENE"])
        gene_result = result["results"]["NONEXISTENT_GENE"]
        assert gene_result["lof_burden_or"] is None
        assert "note" in gene_result

    def test_result_schema_has_required_keys(self):
        result = get_gene_burden_stats(["PCSK9"])
        required = ["genes", "cohort", "results", "note"]
        for k in required:
            assert k in result
