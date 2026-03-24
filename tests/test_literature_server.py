"""
Tests for literature_server.py.

Split into:
  - Unit tests:        anchor papers, stub schemas, DOI parsing logic
  - Integration tests: real HTTP calls to PubMed, Crossref, Europe PMC
                       marked with @pytest.mark.integration

Run all:               pytest tests/test_literature_server.py -v
Run only unit:         pytest tests/test_literature_server.py -v -m "not integration"
Run only integration:  pytest tests/test_literature_server.py -v -m integration
"""
from __future__ import annotations

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp_servers.literature_server import (
    search_pubmed,
    fetch_pubmed_abstract,
    search_pubmed_chip_cad,
    get_anchor_paper,
    list_anchor_papers,
    get_crossref_metadata,
    search_europe_pmc,
    get_semantic_scholar_citations,
    run_paperqa2_query,
    search_gene_disease_literature,
)


# ---------------------------------------------------------------------------
# Unit tests — anchor papers
# ---------------------------------------------------------------------------

class TestAnchorPapers:

    def test_bick2020_anchor(self):
        result = get_anchor_paper("32694926")
        assert result["found"] is True
        assert result["pmid"] == "32694926"
        assert "CHIP" in result["tags"]
        assert "CAD" in result["tags"]
        assert result["year"] == 2020

    def test_kar2022_anchor(self):
        result = get_anchor_paper("35177839")
        assert result["found"] is True
        assert "DNMT3A" in result["tags"] or "CHIP" in result["tags"]

    def test_replogle2022_anchor(self):
        result = get_anchor_paper("35688146")
        assert result["found"] is True
        assert "Perturb-seq" in result["tags"]

    def test_tabula_sapiens_anchor(self):
        result = get_anchor_paper("35549404")
        assert result["found"] is True
        assert "atlas" in result["tags"] or "single-cell" in result["tags"]

    def test_unknown_pmid_returns_not_found(self):
        result = get_anchor_paper("99999999")
        assert result["found"] is False
        assert "available_pmids" in result

    def test_list_anchor_papers_returns_all_four(self):
        result = list_anchor_papers()
        assert result["n_papers"] == 4
        pmids = [p["pmid"] for p in result["papers"]]
        assert "32694926" in pmids
        assert "35177839" in pmids
        assert "35688146" in pmids
        assert "35549404" in pmids

    def test_list_anchor_papers_has_title_and_journal(self):
        result = list_anchor_papers()
        for paper in result["papers"]:
            assert "title" in paper
            assert "journal" in paper
            assert len(paper["title"]) > 0


# ---------------------------------------------------------------------------
# Unit tests — stub tools
# ---------------------------------------------------------------------------

class TestStubTools:

    def test_semantic_scholar_stub_schema(self):
        result = get_semantic_scholar_citations("32694926")
        assert result["paper_id"] == "32694926"
        assert result["citations"] is None
        assert "STUB" in result["note"]
        assert "rate_limit" in result

    def test_paperqa2_stub_schema(self):
        result = run_paperqa2_query(
            "What is the effect of TET2 CHIP on IL-6 signaling?",
            paper_paths=["/path/to/bick2020.pdf"]
        )
        assert result["answer"] is None
        assert result["evidence"] == []
        assert "STUB" in result["note"]
        assert "paperqa_url" in result

    def test_paperqa2_without_paths(self):
        result = run_paperqa2_query("What causes CHIP?")
        assert "STUB" in result["note"]


# ---------------------------------------------------------------------------
# Unit tests — fetch anchor paper abstract (no HTTP needed)
# ---------------------------------------------------------------------------

class TestFetchAbstract:

    def test_anchor_paper_abstract_no_http(self):
        # Anchor papers should return without HTTP call
        result = fetch_pubmed_abstract("32694926")
        assert result["pmid"] == "32694926"
        assert "key_finding" in result.get("abstract", "") or "abstract" in result
        assert result.get("year") == "2020" or result.get("year") == 2020

    def test_anchor_paper_has_doi(self):
        result = fetch_pubmed_abstract("32694926")
        assert result.get("doi") is not None
        assert "10.1038" in result["doi"]


# ---------------------------------------------------------------------------
# Integration tests — PubMed live
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestPubMedLive:

    def test_chip_cad_search_returns_results(self):
        result = search_pubmed(
            "clonal hematopoiesis coronary artery disease",
            max_results=5,
        )
        assert "articles" in result
        assert result["total_found"] > 0
        assert len(result["articles"]) > 0

    def test_search_returns_pmid_and_title(self):
        result = search_pubmed("PCSK9 LDL cholesterol", max_results=3)
        if result.get("total_found", 0) > 0:
            for art in result["articles"]:
                assert "pmid" in art
                assert "title" in art

    def test_bick2020_fetch_abstract(self):
        # Real NCBI fetch for Bick 2020
        result = fetch_pubmed_abstract("32694926")
        assert result["pmid"] == "32694926"
        # Either from anchor cache or live fetch
        assert "title" in result
        assert len(result.get("title", "")) > 10

    def test_chip_cad_convenience_search(self):
        result = search_pubmed_chip_cad(max_results=5)
        assert "articles" in result
        assert result["total_found"] > 5  # CHIP+CAD is a growing field

    def test_gene_disease_literature_search(self):
        result = search_gene_disease_literature("TET2", "atherosclerosis", max_results=3)
        assert "articles" in result or "error" in result


@pytest.mark.integration
class TestCrossrefLive:

    def test_bick2020_doi_metadata(self):
        result = get_crossref_metadata("10.1038/s41586-020-2534-z")
        assert result.get("doi") == "10.1038/s41586-020-2534-z"
        assert "error" not in result
        assert result.get("n_citations") is not None
        assert result["n_citations"] > 50  # Bick 2020 is highly cited

    def test_replogle2022_doi_metadata(self):
        result = get_crossref_metadata("10.1016/j.cell.2022.05.013")
        assert "error" not in result
        assert result.get("journal") is not None

    def test_invalid_doi_returns_error(self):
        result = get_crossref_metadata("10.9999/nonexistent-doi-xyz-123")
        assert "error" in result


@pytest.mark.integration
class TestEuropePMCLive:

    def test_chip_atherosclerosis_search(self):
        result = search_europe_pmc("CHIP clonal hematopoiesis atherosclerosis", max_results=5)
        assert "articles" in result
        assert result["total_found"] > 0

    def test_result_has_open_access_field(self):
        result = search_europe_pmc("TET2 CAD", max_results=3)
        if result.get("total_found", 0) > 0:
            for art in result["articles"]:
                assert "open_access" in art
                assert isinstance(art["open_access"], bool)

    def test_europepmc_gene_disease_search(self):
        result = search_gene_disease_literature(
            "DNMT3A", "coronary artery disease",
            max_results=3, source="europepmc"
        )
        assert "articles" in result or "error" in result
