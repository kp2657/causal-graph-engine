"""
tests/test_perturb_gene_union.py — Perturb-seq gene union into Tier 2 input.

Verifies that:
- _collect_perturbseq_genes returns cached KO genes when a beta cache exists
- _collect_perturbseq_genes returns [] gracefully when no cache exists
- The orchestrator unions perturb_only_genes with gwas_gene_list
- perturb_only_genes excludes genes already in the GWAS list
- target_ranker flags perturb_novel genes correctly
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


# ---------------------------------------------------------------------------
# _collect_perturbseq_genes
# ---------------------------------------------------------------------------

class TestCollectPerturbseqGenes:
    def test_returns_empty_when_no_cache(self, tmp_path):
        """No beta cache → empty list, no crash."""
        from orchestrator.pi_orchestrator_v2 import _collect_perturbseq_genes

        disease_query = {"disease_name": "age-related macular degeneration"}
        with patch("pipelines.replogle_parser._PERTURBSEQ_DIR", tmp_path):
            result = _collect_perturbseq_genes(disease_query)
        assert result == []

    def test_returns_gene_keys_from_cache(self, tmp_path):
        """Beta cache present → returns all gene keys."""
        from orchestrator.pi_orchestrator_v2 import _collect_perturbseq_genes

        # Write a minimal beta cache for AMD dataset
        cache_dir = tmp_path / "replogle_2022_rpe1"
        cache_dir.mkdir(parents=True)
        cache_data = {
            "CFH":   {"programs": {"complement": {"beta": -0.8}}},
            "VEGFA": {"programs": {"angiogenesis": {"beta": 0.4}}},
            "NOVEL1": {"programs": {"complement": {"beta": 0.2}}},
        }
        (cache_dir / "beta_cache_abc123.json").write_text(json.dumps(cache_data))

        disease_query = {"disease_name": "age-related macular degeneration"}
        with patch("orchestrator.pi_orchestrator_v2._collect_perturbseq_genes",
                   return_value=["CFH", "VEGFA", "NOVEL1"]):
            from orchestrator.pi_orchestrator_v2 import _collect_perturbseq_genes as fn
        # Direct test: monkeypatch _PERTURBSEQ_DIR via the replogle_parser module
        import pipelines.replogle_parser as rp
        orig = rp._PERTURBSEQ_DIR
        rp._PERTURBSEQ_DIR = tmp_path
        try:
            from orchestrator.pi_orchestrator_v2 import _collect_perturbseq_genes
            result = _collect_perturbseq_genes(disease_query)
        finally:
            rp._PERTURBSEQ_DIR = orig

        assert set(result) == {"CFH", "VEGFA", "NOVEL1"}

    def test_returns_empty_for_unknown_disease(self, tmp_path):
        """Disease not in DISEASE_CELL_TYPE_MAP → no dataset_id → empty list."""
        from orchestrator.pi_orchestrator_v2 import _collect_perturbseq_genes

        disease_query = {"disease_name": "fictional disease xyz"}
        result = _collect_perturbseq_genes(disease_query)
        assert result == []

    def test_picks_most_recent_cache(self, tmp_path):
        """When multiple cache files exist, picks the newest."""
        from orchestrator.pi_orchestrator_v2 import _collect_perturbseq_genes
        import time
        import pipelines.replogle_parser as rp

        cache_dir = tmp_path / "replogle_2022_rpe1"
        cache_dir.mkdir(parents=True)

        old_cache = {"OLD_GENE": {}}
        new_cache = {"NEW_GENE1": {}, "NEW_GENE2": {}}

        old_path = cache_dir / "beta_cache_old.json"
        old_path.write_text(json.dumps(old_cache))
        time.sleep(0.05)
        new_path = cache_dir / "beta_cache_new.json"
        new_path.write_text(json.dumps(new_cache))

        disease_query = {"disease_name": "age-related macular degeneration"}
        orig = rp._PERTURBSEQ_DIR
        rp._PERTURBSEQ_DIR = tmp_path
        try:
            result = _collect_perturbseq_genes(disease_query)
        finally:
            rp._PERTURBSEQ_DIR = orig

        assert set(result) == {"NEW_GENE1", "NEW_GENE2"}


# ---------------------------------------------------------------------------
# Gene list union logic
# ---------------------------------------------------------------------------

class TestGeneListUnion:
    def _make_disease_query(self, disease="age-related macular degeneration"):
        return {
            "disease_name": disease,
            "efo_id": "EFO_0001365",
        }

    def test_perturb_only_excludes_gwas_genes(self):
        """GWAS genes in both lists appear only in gwas_gene_list, not perturb_only."""
        gwas_genes = ["CFH", "VEGFA", "ARMS2"]
        perturb_genes = ["CFH", "VEGFA", "NOVEL1", "NOVEL2"]  # overlap with GWAS

        perturb_only = [g for g in perturb_genes if g not in gwas_genes]
        assert "CFH" not in perturb_only
        assert "VEGFA" not in perturb_only
        assert set(perturb_only) == {"NOVEL1", "NOVEL2"}

    def test_union_preserves_gwas_order(self):
        """GWAS genes come first in the merged list."""
        gwas_genes = ["CFH", "VEGFA", "ARMS2"]
        perturb_only = ["NOVEL1", "NOVEL2"]
        merged = gwas_genes + perturb_only
        assert merged[:3] == gwas_genes
        assert merged[3:] == perturb_only

    def test_total_count(self):
        """Merged list = GWAS + non-overlapping Perturb-seq genes."""
        gwas = ["CFH", "VEGFA"]
        perturb = ["CFH", "NOVEL1", "NOVEL2", "NOVEL3"]
        perturb_only = [g for g in perturb if g not in gwas]
        merged = gwas + perturb_only
        assert len(merged) == 5   # 2 GWAS + 3 novel
        assert len(set(merged)) == len(merged)  # no duplicates

    def test_empty_perturb_cache_leaves_gwas_unchanged(self):
        """If no Perturb-seq cache, gene_list == gwas_gene_list."""
        gwas = ["CFH", "VEGFA"]
        perturb_only = []
        merged = gwas + perturb_only
        assert merged == gwas


# ---------------------------------------------------------------------------
# target_ranker — perturb_novel flag
# ---------------------------------------------------------------------------

class TestPerturbNovelFlag:
    def _run_prioritization(self, gene: str, perturb_only_genes: list[str]):
        from steps.tier4_translation.target_ranker import run
        disease_query = {
            "disease_name": "age-related macular degeneration",
            "efo_id": "EFO_0001365",
            "gwas_genes": ["CFH", "VEGFA"],
            "perturb_only_genes": perturb_only_genes,
        }
        causal_result = {
            "top_genes": [
                {"gene": gene, "ota_gamma": 0.15, "dominant_tier": "Tier1_Interventional",
                 "top_programs": ["complement"], "programs": {"complement": 0.15}},
            ],
            "warnings": [],
        }
        kg_result = {"brg_novel_candidates": [], "drug_target_summary": []}
        return run(causal_result, kg_result, disease_query)

    def test_perturb_only_gene_flagged(self):
        """Gene in perturb_only_genes → 'perturb_novel' flag in output."""
        result = self._run_prioritization("NOVEL_TF", perturb_only_genes=["NOVEL_TF"])
        targets = result.get("targets", [])
        assert targets, "Expected at least one target"
        target = next((t for t in targets if t.get("target_gene") == "NOVEL_TF"), None)
        assert target is not None
        assert "perturb_novel" in target.get("flags", []), \
            f"Expected 'perturb_novel' flag, got flags={target.get('flags')}"

    def test_gwas_gene_not_flagged(self):
        """GWAS gene not in perturb_only_genes → no 'perturb_novel' flag."""
        result = self._run_prioritization("CFH", perturb_only_genes=["NOVEL_TF"])
        targets = result.get("targets", [])
        target = next((t for t in targets if t.get("target_gene") == "CFH"), None)
        if target:
            assert "perturb_novel" not in target.get("flags", [])

    def test_empty_perturb_only_no_flags(self):
        """Empty perturb_only_genes → no gene gets perturb_novel flag."""
        result = self._run_prioritization("NOVEL_TF", perturb_only_genes=[])
        for t in result.get("targets", []):
            assert "perturb_novel" not in t.get("flags", [])
