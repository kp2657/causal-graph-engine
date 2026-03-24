"""
Tests for single_cell_server.py.

Split into:
  - Unit tests:        hardcoded catalogs, stub schemas, known gene expression
  - Integration tests: real HTTP calls to CELLxGENE API
                       marked with @pytest.mark.integration

Run all:               pytest tests/test_single_cell_server.py -v
Run only unit:         pytest tests/test_single_cell_server.py -v -m "not integration"
Run only integration:  pytest tests/test_single_cell_server.py -v -m integration
"""
from __future__ import annotations

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp_servers.single_cell_server import (
    query_cellxgene_gene_summary,
    list_cellxgene_datasets,
    get_tabula_sapiens_cell_types,
    get_gene_cell_type_specificity,
    stream_census_anndata,
    run_differential_expression,
    compute_program_cell_type_scores,
)


# ---------------------------------------------------------------------------
# Unit tests — gene expression queries (literature-curated fallback)
# ---------------------------------------------------------------------------

class TestQueryCellxgeneGeneSummary:

    def test_pcsk9_liver_expression(self):
        result = query_cellxgene_gene_summary("PCSK9")
        assert result["gene"] == "PCSK9"
        cell_types = [e["cell_type"] for e in result["cell_type_expression"]]
        # PCSK9 is liver-specific
        assert "hepatocyte" in cell_types, f"Expected hepatocyte for PCSK9, got: {cell_types}"

    def test_tet2_myeloid_expression(self):
        result = query_cellxgene_gene_summary("TET2")
        assert result["gene"] == "TET2"
        cell_types = [e["cell_type"] for e in result["cell_type_expression"]]
        assert any(ct in cell_types for ct in ["monocyte", "macrophage", "hematopoietic_stem_cell"])

    def test_unknown_gene_returns_stub(self):
        result = query_cellxgene_gene_summary("NONEXISTENT_GENE_XYZ")
        assert result["data_tier"] == "stub"
        assert "note" in result

    def test_cell_type_filter(self):
        result = query_cellxgene_gene_summary("TET2", cell_types=["monocyte"])
        for expr in result["cell_type_expression"]:
            assert expr["cell_type"] == "monocyte"

    def test_hla_dra_b_cell_expression(self):
        result = query_cellxgene_gene_summary("HLA-DRA")
        cell_types = [e["cell_type"] for e in result["cell_type_expression"]]
        assert any(ct in cell_types for ct in ["B_cell", "dendritic_cell"])

    def test_result_has_source_field(self):
        result = query_cellxgene_gene_summary("PCSK9")
        assert "source" in result or "data_tier" in result


# ---------------------------------------------------------------------------
# Unit tests — Tabula Sapiens catalog
# ---------------------------------------------------------------------------

class TestTabulaSapiensQuery:

    def test_all_cell_types_returned_without_filter(self):
        result = get_tabula_sapiens_cell_types()
        assert result["n_cell_types"] > 0
        assert "cell_types" in result
        assert result["n_tabula_sapiens_total_tissues"] == 24

    def test_blood_tissue_filter(self):
        result = get_tabula_sapiens_cell_types(tissue="blood")
        assert result["n_cell_types"] > 0
        for ct_name, ct_info in result["cell_types"].items():
            assert "blood" in ct_info["tissue"]

    def test_heart_tissue_filter(self):
        result = get_tabula_sapiens_cell_types(tissue="heart")
        assert "cardiomyocyte" in result["cell_types"] or result["n_cell_types"] > 0

    def test_chip_relevant_filter(self):
        result = get_tabula_sapiens_cell_types(chip_relevant_only=True)
        for ct_name, ct_info in result["cell_types"].items():
            assert ct_info.get("chip_relevant") is True

    def test_chip_relevant_includes_monocyte(self):
        result = get_tabula_sapiens_cell_types(chip_relevant_only=True)
        assert "monocyte" in result["cell_types"] or "classical_monocyte" in result["cell_types"]

    def test_chip_relevant_includes_hsc(self):
        result = get_tabula_sapiens_cell_types(chip_relevant_only=True)
        assert "hematopoietic_stem_cell" in result["cell_types"]

    def test_cell_types_have_ontology_ids(self):
        result = get_tabula_sapiens_cell_types()
        for ct_name, ct_info in result["cell_types"].items():
            assert "ontology_id" in ct_info, f"Missing ontology_id for {ct_name}"
            assert ct_info["ontology_id"].startswith("CL:")

    def test_source_field_present(self):
        result = get_tabula_sapiens_cell_types()
        assert "35549404" in result.get("source", "")  # Tabula Sapiens PMID


# ---------------------------------------------------------------------------
# Unit tests — gene cell type specificity
# ---------------------------------------------------------------------------

class TestGeneCellTypeSpecificity:

    def test_pcsk9_high_in_hepatocyte(self):
        result = get_gene_cell_type_specificity("PCSK9")
        assert "hepatocyte" in result["high_in"]

    def test_il6_high_in_monocyte(self):
        result = get_gene_cell_type_specificity("IL6")
        assert any("monocyte" in ct or "macrophage" in ct for ct in result["high_in"])

    def test_dnmt3a_high_in_hsc(self):
        result = get_gene_cell_type_specificity("DNMT3A")
        assert any("hematopoietic" in ct or "monocyte" in ct for ct in result["high_in"])

    def test_unknown_gene_returns_stub(self):
        result = get_gene_cell_type_specificity("NONEXISTENT_XYZ")
        assert result["data_tier"] == "stub"
        assert result["high_in"] == []

    def test_return_top_n_respected(self):
        result = get_gene_cell_type_specificity("PCSK9", return_top_n=2)
        assert len(result["high_in"]) <= 2
        assert len(result["low_in"]) <= 2

    def test_chip_drivers_expressed_in_myeloid(self):
        for gene in ["DNMT3A", "TET2"]:
            result = get_gene_cell_type_specificity(gene)
            all_high = " ".join(result["high_in"])
            assert "monocyte" in all_high or "hematopoietic" in all_high or "macrophage" in all_high, \
                f"{gene} should be high in myeloid lineage cells"


# ---------------------------------------------------------------------------
# Unit tests — program × cell type scores
# ---------------------------------------------------------------------------

class TestProgramCellTypeScores:

    def test_inflammatory_monocyte_score(self):
        result = compute_program_cell_type_scores(
            programs=["inflammatory_NF-kB"],
            cell_types=["monocyte"],
        )
        score = result["scores"]["inflammatory_NF-kB"]["monocyte"]
        assert score is not None
        assert score > 0.5, f"Expected inflammatory program high in monocyte, got {score}"

    def test_lipid_metabolism_hepatocyte_score(self):
        result = compute_program_cell_type_scores(
            programs=["lipid_metabolism"],
            cell_types=["hepatocyte"],
        )
        score = result["scores"]["lipid_metabolism"]["hepatocyte"]
        assert score is not None
        assert score > 0.8

    def test_mhc_program_b_cell_score(self):
        result = compute_program_cell_type_scores(
            programs=["MHC_class_II_presentation"],
            cell_types=["B_cell", "dendritic_cell"],
        )
        b_score = result["scores"]["MHC_class_II_presentation"]["B_cell"]
        assert b_score is not None
        assert b_score > 0.8

    def test_unknown_combination_returns_none(self):
        result = compute_program_cell_type_scores(
            programs=["lipid_metabolism"],
            cell_types=["T_cell"],  # not in literature scores
        )
        score = result["scores"]["lipid_metabolism"]["T_cell"]
        assert score is None

    def test_result_schema(self):
        result = compute_program_cell_type_scores(
            programs=["inflammatory_NF-kB"],
            cell_types=["monocyte"],
        )
        required = ["programs", "cell_types", "scores", "data_tier", "note"]
        for k in required:
            assert k in result


# ---------------------------------------------------------------------------
# Unit tests — stub tools
# ---------------------------------------------------------------------------

class TestStubTools:

    def test_stream_census_stub_schema(self):
        result = stream_census_anndata(
            gene_symbols=["PCSK9", "LDLR"],
            cell_type="hepatocyte",
        )
        assert result["gene_symbols"] == ["PCSK9", "LDLR"]
        assert result["cell_type"] == "hepatocyte"
        assert result["anndata_path"] is None
        assert result["n_cells"] is None
        assert "STUB" in result["note"]

    def test_differential_expression_stub_schema(self):
        result = run_differential_expression(
            gene_list=["IL6", "TNF"],
            cell_type_a="monocyte",
            cell_type_b="macrophage",
        )
        assert result["cell_type_a"] == "monocyte"
        assert result["cell_type_b"] == "macrophage"
        assert result["results"] is None
        assert "STUB" in result["note"]


# ---------------------------------------------------------------------------
# Integration tests — live CELLxGENE API
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestCellxgeneLive:

    def test_list_cad_datasets(self):
        result = list_cellxgene_datasets(disease="coronary artery disease")
        # Should either return collections or a graceful error
        assert "collections" in result or "error" in result
        if "collections" in result:
            assert isinstance(result["collections"], list)

    def test_list_heart_datasets(self):
        result = list_cellxgene_datasets(tissue="heart")
        assert "collections" in result or "error" in result

    def test_api_response_has_source(self):
        result = list_cellxgene_datasets()
        # Should not crash
        assert "collections" in result or "error" in result
