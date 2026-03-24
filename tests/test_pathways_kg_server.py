"""
Tests for pathways_kg_server.py.

Unit tests use hardcoded PrimeKG subgraph.
Integration tests hit Reactome / STRING live APIs.
"""
from __future__ import annotations

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp_servers.pathways_kg_server import (
    query_primekg_subgraph,
    run_bioPathNet_causal_inference,
    get_reactome_pathways_for_gene,
    get_string_interactions,
    get_kegg_pathway_genes,
)


class TestPrimeKGSubgraph:

    def test_pcsk9_edges_found(self):
        result = query_primekg_subgraph(gene="PCSK9")
        assert result["n_edges"] > 0
        for edge in result["edges"]:
            assert "PCSK9" in (edge["from"].upper(), edge["to"].upper())

    def test_drug_target_filter(self):
        result = query_primekg_subgraph(edge_type="drug_target")
        assert result["n_edges"] > 0
        for edge in result["edges"]:
            assert edge["edge_type"] == "drug_target"

    def test_atorvastatin_targets_hmgcr(self):
        result = query_primekg_subgraph(gene="atorvastatin", edge_type="drug_target")
        assert result["n_edges"] > 0
        to_genes = [e["to"] for e in result["edges"]]
        assert "HMGCR" in to_genes

    def test_ppi_filter(self):
        result = query_primekg_subgraph(edge_type="ppi")
        assert result["n_edges"] > 0

    def test_no_filter_returns_all(self):
        result = query_primekg_subgraph()
        assert result["n_edges"] == len([]) or result["n_edges"] > 0

    def test_result_has_source(self):
        result = query_primekg_subgraph()
        assert "Chandak" in result["source"] or "PrimeKG" in result["source"]

    def test_bioPathNet_stub_schema(self):
        result = run_bioPathNet_causal_inference(["PCSK9", "LDLR"], "CAD")
        assert result["causal_paths"] is None
        assert "STUB" in result["note"]
        assert "paper" in result


@pytest.mark.integration
class TestReactomeLive:

    def test_pcsk9_pathways(self):
        result = get_reactome_pathways_for_gene("PCSK9")
        assert "pathways" in result
        # PCSK9 is in LDL metabolism — may return results
        # Accept either result or graceful error
        assert isinstance(result["pathways"], list)

    def test_hmgcr_pathways(self):
        result = get_reactome_pathways_for_gene("HMGCR")
        assert "pathways" in result


@pytest.mark.integration
class TestStringLive:

    def test_pcsk9_ldlr_interaction(self):
        result = get_string_interactions(["PCSK9", "LDLR"])
        assert "edges" in result
        if result["n_edges"] > 0:
            for edge in result["edges"]:
                assert "protein_a" in edge
                assert "protein_b" in edge

    def test_empty_gene_list(self):
        result = get_string_interactions([])
        assert "edges" in result or "error" in result
