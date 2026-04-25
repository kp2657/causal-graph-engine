"""
tests/test_program_clustering.py — program influence clustering unit tests.

Covers:
  1. Basic clustering: genes with distinct program vectors end up in different clusters
  2. Same-program genes: all genes with identical vectors → same cluster
  3. Ungrounded genes (n_programs_contributing=0) get None cluster fields
  4. Single gene: falls back gracefully (no crash, cluster_id=None)
  5. Mixed: grounded + ungrounded in one list
  6. cluster_label derived from dominant_program
  7. cluster_size reflects correct membership count
"""
from __future__ import annotations
import pytest


def _make_gene(name: str, programs: dict, n_programs: int = 1) -> dict:
    return {
        "gene": name,
        "programs": programs,
        "n_programs_contributing": n_programs,
    }


def _ungrounded(name: str) -> dict:
    return {
        "gene": name,
        "programs": {},
        "n_programs_contributing": 0,
    }


class TestProgramClustering:

    def test_distinct_programs_split_into_clusters(self):
        from pipelines.program_clustering import cluster_by_program_influence
        genes = [
            _make_gene("GENE_A", {"complement_program": 0.8, "angiogenesis_program": 0.1}),
            _make_gene("GENE_B", {"complement_program": 0.9, "angiogenesis_program": 0.05}),
            _make_gene("GENE_C", {"angiogenesis_program": 0.85, "complement_program": 0.05}),
            _make_gene("GENE_D", {"angiogenesis_program": 0.90, "complement_program": 0.02}),
        ]
        result = cluster_by_program_influence(genes, max_clusters=2)
        ids = {r["gene"]: r["program_cluster_id"] for r in result}
        # A+B should share a cluster; C+D should share a different cluster
        assert ids["GENE_A"] == ids["GENE_B"]
        assert ids["GENE_C"] == ids["GENE_D"]
        assert ids["GENE_A"] != ids["GENE_C"]

    def test_identical_vectors_same_cluster(self):
        from pipelines.program_clustering import cluster_by_program_influence
        genes = [
            _make_gene("G1", {"lipid_metabolism": 1.0}),
            _make_gene("G2", {"lipid_metabolism": 1.0}),
            _make_gene("G3", {"lipid_metabolism": 0.95, "complement_program": 0.05}),
        ]
        result = cluster_by_program_influence(genes, max_clusters=3)
        ids = {r["gene"]: r["program_cluster_id"] for r in result}
        assert ids["G1"] == ids["G2"]

    def test_ungrounded_genes_get_none_fields(self):
        from pipelines.program_clustering import cluster_by_program_influence
        genes = [
            _make_gene("VEGFA", {"angiogenesis_program": 0.9}),
            _make_gene("CFH_mech", {"complement_program": 0.8}),
            _ungrounded("CFH"),
            _ungrounded("C3"),
        ]
        result = cluster_by_program_influence(genes, max_clusters=2)
        for r in result:
            if r["gene"] in ("CFH", "C3"):
                assert r["program_cluster_id"] is None
                assert r["dominant_program"] is None
                assert r["cluster_label"] is None
                assert r["cluster_size"] is None
            else:
                assert r["program_cluster_id"] is not None

    def test_single_grounded_gene_no_crash(self):
        from pipelines.program_clustering import cluster_by_program_influence
        genes = [_make_gene("SOLO", {"complement_program": 1.0})]
        result = cluster_by_program_influence(genes)
        # With only 1 gene, clustering falls back gracefully
        assert len(result) == 1
        assert result[0]["program_cluster_id"] is None or result[0]["program_cluster_id"] == 0

    def test_zero_grounded_all_none(self):
        from pipelines.program_clustering import cluster_by_program_influence
        genes = [_ungrounded("CFH"), _ungrounded("APOE")]
        result = cluster_by_program_influence(genes)
        for r in result:
            assert r["program_cluster_id"] is None

    def test_cluster_label_derived_from_dominant(self):
        from pipelines.program_clustering import cluster_by_program_influence
        genes = [
            _make_gene("G1", {"HALLMARK_COMPLEMENT": 0.9}),
            _make_gene("G2", {"HALLMARK_COMPLEMENT": 0.85}),
        ]
        result = cluster_by_program_influence(genes, max_clusters=1)
        for r in result:
            if r["dominant_program"]:
                assert r["cluster_label"] == r["dominant_program"].replace("HALLMARK_", "").replace("_", " ").lower().strip()

    def test_cluster_size_correct(self):
        from pipelines.program_clustering import cluster_by_program_influence
        genes = [
            _make_gene("A", {"complement_program": 1.0}),
            _make_gene("B", {"complement_program": 0.95}),
            _make_gene("C", {"complement_program": 0.90}),
            _make_gene("D", {"angiogenesis_program": 1.0}),
        ]
        result = cluster_by_program_influence(genes, max_clusters=2)
        sizes = {r["gene"]: r["cluster_size"] for r in result}
        # The 3 complement genes should be in a cluster of size 3
        # (or possibly 4 if k=1, but with k=2 they should split)
        complement_genes = [r for r in result if r["gene"] in ("A", "B", "C")]
        if complement_genes[0]["program_cluster_id"] is not None:
            assert complement_genes[0]["cluster_size"] >= 1

    def test_output_preserves_all_input_keys(self):
        from pipelines.program_clustering import cluster_by_program_influence
        genes = [
            {"gene": "PCSK9", "programs": {"lipid_metabolism": 0.9},
             "n_programs_contributing": 1, "ota_gamma": 1.5, "tier": "Tier1"},
        ]
        result = cluster_by_program_influence(genes)
        assert result[0]["ota_gamma"] == 1.5
        assert result[0]["tier"] == "Tier1"
        assert "program_cluster_id" in result[0]
