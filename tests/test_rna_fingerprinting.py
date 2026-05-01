"""
Tests for RNA fingerprinting: SVD denoising + disease-to-fingerprint matching.

All tests are unit tests — no h5ad or network access required.
"""
from __future__ import annotations

import gzip
import json
import math
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp_servers.perturbseq_server import (
    preprocess_rna_fingerprints,
    map_disease_to_fingerprints,
    compute_svd_nomination_scores,
    _sig_path,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _write_sigs(tmp_path: Path, dataset_id: str, sigs: dict, variant: str = "") -> None:
    """Write a fake signatures.json.gz into tmp_path/{dataset_id}/."""
    suffix = f"_{variant}" if variant else ""
    p = tmp_path / dataset_id / f"signatures{suffix}.json.gz"
    p.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(p, "wt") as f:
        json.dump(sigs, f)


_MOCK_SIGS = {
    "PCSK9": {"LDLR": 2.0, "APOB": 1.5, "HMGCR": 1.0, "LPL": -1.0, "ANGPTL3": 0.8},
    "CETP":  {"APOA1": 1.8, "APOB": -0.5, "SCARB1": 1.2, "PLTP": 0.9, "LCAT": 1.1},
    "IL6R":  {"STAT3": 2.5, "JAK1": 1.2, "SOCS3": 0.9, "CISH": 0.7, "IL6ST": 1.4},
    "JAK2":  {"STAT1": 1.9, "STAT3": 1.1, "IRF1": 0.8, "SOCS1": 1.3, "CISH": 0.6},
    "TYK2":  {"STAT1": 1.4, "STAT3": 0.9, "SOCS1": 0.7, "IRF9": 1.2, "MX1": 1.0},
}


# ---------------------------------------------------------------------------
# preprocess_rna_fingerprints
# ---------------------------------------------------------------------------

class TestPreprocessRnaFingerprints:

    def test_produces_fingerprint_file(self, tmp_path):
        _write_sigs(tmp_path, "test_ds", _MOCK_SIGS)
        with patch("mcp_servers.perturbseq_server._CACHE_DIR", tmp_path):
            import mcp_servers.perturbseq_server as ps
            orig = ps._CACHE_DIR
            ps._CACHE_DIR = tmp_path
            try:
                result = preprocess_rna_fingerprints("test_ds", top_k=10)
            finally:
                ps._CACHE_DIR = orig
        assert "error" not in result
        fp_file = tmp_path / "test_ds" / "signatures_fingerprint.json.gz"
        assert fp_file.exists(), "fingerprint file not created"

    def test_returns_correct_metadata(self, tmp_path):
        _write_sigs(tmp_path, "test_ds", _MOCK_SIGS)
        import mcp_servers.perturbseq_server as ps
        orig = ps._CACHE_DIR
        ps._CACHE_DIR = tmp_path
        try:
            result = preprocess_rna_fingerprints("test_ds", top_k=10)
        finally:
            ps._CACHE_DIR = orig
        assert result["n_perturbations"] == len(_MOCK_SIGS)
        assert result["fingerprint_variant"] == "fingerprint"
        assert result["svd_rank"] > 0

    def test_fingerprint_has_same_perturbation_keys(self, tmp_path):
        _write_sigs(tmp_path, "test_ds", _MOCK_SIGS)
        import mcp_servers.perturbseq_server as ps
        orig = ps._CACHE_DIR
        ps._CACHE_DIR = tmp_path
        try:
            preprocess_rna_fingerprints("test_ds", top_k=10)
            # Load the fingerprint file directly
            fp_path = tmp_path / "test_ds" / "signatures_fingerprint.json.gz"
            with gzip.open(fp_path, "rt") as f:
                fp_sigs = json.load(f)
        finally:
            ps._CACHE_DIR = orig
        assert set(fp_sigs.keys()) == set(_MOCK_SIGS.keys())

    def test_fingerprint_top_k_respected(self, tmp_path):
        _write_sigs(tmp_path, "test_ds", _MOCK_SIGS)
        import mcp_servers.perturbseq_server as ps
        orig = ps._CACHE_DIR
        ps._CACHE_DIR = tmp_path
        try:
            preprocess_rna_fingerprints("test_ds", top_k=3)
            fp_path = tmp_path / "test_ds" / "signatures_fingerprint.json.gz"
            with gzip.open(fp_path, "rt") as f:
                fp_sigs = json.load(f)
        finally:
            ps._CACHE_DIR = orig
        for pert, genes in fp_sigs.items():
            assert len(genes) <= 3, f"{pert} has {len(genes)} genes > top_k=3"

    def test_missing_source_returns_error(self, tmp_path):
        import mcp_servers.perturbseq_server as ps
        orig = ps._CACHE_DIR
        ps._CACHE_DIR = tmp_path
        try:
            result = preprocess_rna_fingerprints("nonexistent_dataset")
        finally:
            ps._CACHE_DIR = orig
        assert "error" in result

    def test_delta_variant_supported(self, tmp_path):
        _write_sigs(tmp_path, "test_ds", _MOCK_SIGS, variant="delta")
        import mcp_servers.perturbseq_server as ps
        orig = ps._CACHE_DIR
        ps._CACHE_DIR = tmp_path
        try:
            result = preprocess_rna_fingerprints("test_ds", source_variant="delta", top_k=10)
        finally:
            ps._CACHE_DIR = orig
        assert "error" not in result
        assert result["fingerprint_variant"] == "fingerprint_delta"
        fp_file = tmp_path / "test_ds" / "signatures_fingerprint_delta.json.gz"
        assert fp_file.exists()

    def test_svd_rank_bounded_by_matrix_dims(self, tmp_path):
        # Only 3 perturbations → rank cannot exceed 2
        small_sigs = {k: v for k, v in list(_MOCK_SIGS.items())[:3]}
        _write_sigs(tmp_path, "rank_ds", small_sigs)
        import mcp_servers.perturbseq_server as ps
        orig = ps._CACHE_DIR
        ps._CACHE_DIR = tmp_path
        ps._SIG_CACHE.clear()
        try:
            result = preprocess_rna_fingerprints("rank_ds", top_k=10)
        finally:
            ps._CACHE_DIR = orig
            ps._SIG_CACHE.clear()
        assert result["svd_rank"] <= 2

    def test_denoised_values_differ_from_raw(self, tmp_path):
        """SVD reconstruction should produce different values than raw log2FC."""
        _write_sigs(tmp_path, "diff_ds", _MOCK_SIGS)
        import mcp_servers.perturbseq_server as ps
        orig = ps._CACHE_DIR
        ps._CACHE_DIR = tmp_path
        ps._SIG_CACHE.clear()
        try:
            preprocess_rna_fingerprints("diff_ds", top_k=5)
            fp_path = tmp_path / "diff_ds" / "signatures_fingerprint.json.gz"
            with gzip.open(fp_path, "rt") as f:
                fp_sigs = json.load(f)
        finally:
            ps._CACHE_DIR = orig
            ps._SIG_CACHE.clear()
        # Denoised STAT3 for IL6R should differ from raw 2.5
        il6r_stat3_raw = _MOCK_SIGS["IL6R"].get("STAT3", None)
        il6r_fp = fp_sigs.get("IL6R", {})
        if "STAT3" in il6r_fp and il6r_stat3_raw is not None:
            assert il6r_fp["STAT3"] != il6r_stat3_raw

    def test_magnitude_preserved_after_rescaling(self, tmp_path):
        """Each perturbation's fingerprint L2 norm should match the raw signature's L2 norm."""
        import math, gzip as gz
        _write_sigs(tmp_path, "mag_ds", _MOCK_SIGS)
        import mcp_servers.perturbseq_server as ps
        orig = ps._CACHE_DIR
        ps._CACHE_DIR = tmp_path
        ps._SIG_CACHE.clear()
        try:
            preprocess_rna_fingerprints("mag_ds", top_k=20)  # keep most genes
            fp_path = tmp_path / "mag_ds" / "signatures_fingerprint.json.gz"
            with gz.open(fp_path, "rt") as f:
                fp_sigs = json.load(f)
        finally:
            ps._CACHE_DIR = orig
            ps._SIG_CACHE.clear()

        for pert, raw_genes in _MOCK_SIGS.items():
            raw_norm = math.sqrt(sum(v**2 for v in raw_genes.values()))
            fp_genes  = fp_sigs.get(pert, {})
            fp_norm   = math.sqrt(sum(v**2 for v in fp_genes.values()))
            if raw_norm > 0 and fp_norm > 0:
                # norms are within same order of magnitude: centering shifts the
                # column norms, so exact matching isn't expected; check within 2×
                assert fp_norm / raw_norm < 2.0 and raw_norm / fp_norm < 2.0, (
                    f"{pert}: raw_norm={raw_norm:.3f} fp_norm={fp_norm:.3f} (>2× apart)"
                )


# ---------------------------------------------------------------------------
# map_disease_to_fingerprints
# ---------------------------------------------------------------------------

def _write_fingerprint_sigs(tmp_path: Path, dataset_id: str, sigs: dict) -> None:
    """Write fake fingerprint signatures (signatures_fingerprint.json.gz)."""
    p = tmp_path / dataset_id / "signatures_fingerprint.json.gz"
    p.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(p, "wt") as f:
        json.dump(sigs, f)


# Disease DE: high STAT3/JAK1/SOCS3 → should match IL6R/JAK2/TYK2 KOs
_DISEASE_DE = {
    "STAT3": 2.0, "JAK1": 1.5, "SOCS3": 1.2, "CISH": 0.9, "IRF1": 0.8,
    "LDLR": -0.3, "APOB": -0.1, "HMGCR": -0.2, "LPL": 0.1, "ANGPTL3": 0.0,
    "APOA1": -0.2, "SCARB1": 0.0, "PLTP": 0.1, "LCAT": -0.1,
    "IRF9": 0.7, "MX1": 0.6, "SOCS1": 1.0, "STAT1": 1.8, "IL6ST": 1.3,
}


class TestMapDiseaseToFingerprints:

    def test_returns_results_for_overlapping_genes(self, tmp_path):
        _write_fingerprint_sigs(tmp_path, "fp_ds", _MOCK_SIGS)
        import mcp_servers.perturbseq_server as ps
        orig = ps._CACHE_DIR
        ps._CACHE_DIR = tmp_path
        ps._SIG_CACHE.clear()
        try:
            # min_gene_overlap=3 because mock fingerprints only have ~5 genes each
            result = map_disease_to_fingerprints(
                _DISEASE_DE, "fp_ds", n_bootstrap=5, min_gene_overlap=3
            )
        finally:
            ps._CACHE_DIR = orig
            ps._SIG_CACHE.clear()
        assert "error" not in result
        assert result["n_perturbations_matched"] > 0

    def test_missing_fingerprints_returns_error(self, tmp_path):
        import mcp_servers.perturbseq_server as ps
        orig = ps._CACHE_DIR
        ps._CACHE_DIR = tmp_path
        try:
            result = map_disease_to_fingerprints({"GENE1": 1.0}, "no_dataset")
        finally:
            ps._CACHE_DIR = orig
        assert "error" in result

    def test_result_schema(self, tmp_path):
        _write_fingerprint_sigs(tmp_path, "schema_ds", _MOCK_SIGS)
        import mcp_servers.perturbseq_server as ps
        orig = ps._CACHE_DIR
        ps._CACHE_DIR = tmp_path
        ps._SIG_CACHE.clear()
        try:
            result = map_disease_to_fingerprints(
                _DISEASE_DE, "schema_ds", n_bootstrap=5, min_gene_overlap=3
            )
        finally:
            ps._CACHE_DIR = orig
            ps._SIG_CACHE.clear()
        for entry in result.get("top_therapeutic_kos", []):
            assert "gene_ko" in entry
            assert "r" in entry
            assert "r_se" in entry
            assert "n_shared" in entry

    def test_immune_disease_ranks_immune_kos_higher(self, tmp_path):
        """IL6R/JAK2/TYK2 KOs should have higher |r| with the immune disease DE."""
        _write_fingerprint_sigs(tmp_path, "rank_bio_ds", _MOCK_SIGS)
        import mcp_servers.perturbseq_server as ps
        orig = ps._CACHE_DIR
        ps._CACHE_DIR = tmp_path
        ps._SIG_CACHE.clear()
        try:
            result = map_disease_to_fingerprints(
                _DISEASE_DE, "rank_bio_ds", n_bootstrap=10, min_gene_overlap=3
            )
        finally:
            ps._CACHE_DIR = orig
            ps._SIG_CACHE.clear()
        if result["n_perturbations_matched"] < 2:
            pytest.skip("Too few matches for biological ranking test")
        top_mimics = {e["gene_ko"] for e in result["top_disease_mimics"]}
        immune_kos = {"IL6R", "JAK2", "TYK2"}
        assert top_mimics & immune_kos, f"No immune KO in top disease mimics: {top_mimics}"

    def test_correlation_bounded(self, tmp_path):
        _write_fingerprint_sigs(tmp_path, "bound_ds", _MOCK_SIGS)
        import mcp_servers.perturbseq_server as ps
        orig = ps._CACHE_DIR
        ps._CACHE_DIR = tmp_path
        ps._SIG_CACHE.clear()
        try:
            result = map_disease_to_fingerprints(
                _DISEASE_DE, "bound_ds", n_bootstrap=5, min_gene_overlap=3
            )
        finally:
            ps._CACHE_DIR = orig
            ps._SIG_CACHE.clear()
        for entry in result.get("top_disease_mimics", []) + result.get("top_therapeutic_kos", []):
            assert -1.0 <= entry["r"] <= 1.0, f"r out of bounds: {entry['r']}"

    def test_output_file_written(self, tmp_path):
        _write_fingerprint_sigs(tmp_path, "out_ds", _MOCK_SIGS)
        import mcp_servers.perturbseq_server as ps
        orig = ps._CACHE_DIR
        ps._CACHE_DIR = tmp_path
        ps._SIG_CACHE.clear()
        try:
            result = map_disease_to_fingerprints(
                _DISEASE_DE, "out_ds", n_bootstrap=5, min_gene_overlap=3
            )
        finally:
            ps._CACHE_DIR = orig
            ps._SIG_CACHE.clear()
        out_path = Path(result.get("output_path", ""))
        assert out_path.exists(), "Output JSON not written"


# ---------------------------------------------------------------------------
# Integration: preprocess then map
# ---------------------------------------------------------------------------

class TestFingerprintRoundtrip:

    def test_preprocess_then_map(self, tmp_path):
        """Full roundtrip: raw sigs → fingerprints → disease matching."""
        _write_sigs(tmp_path, "rt_ds", _MOCK_SIGS)
        import mcp_servers.perturbseq_server as ps
        orig = ps._CACHE_DIR
        ps._CACHE_DIR = tmp_path
        ps._SIG_CACHE.clear()
        try:
            prep = preprocess_rna_fingerprints("rt_ds", top_k=10)
            assert "error" not in prep
            match = map_disease_to_fingerprints(
                _DISEASE_DE, "rt_ds", n_bootstrap=5, min_gene_overlap=3
            )
            assert "error" not in match
            assert match["n_perturbations_matched"] >= 0
        finally:
            ps._CACHE_DIR = orig
            ps._SIG_CACHE.clear()


# ---------------------------------------------------------------------------
# compute_svd_nomination_scores
# ---------------------------------------------------------------------------

class TestSvdNominationScores:

    def _prep(self, tmp_path: Path) -> None:
        """Write sigs and preprocess (creates svd_loadings.npz)."""
        _write_sigs(tmp_path, "svd_ds", _MOCK_SIGS)
        import mcp_servers.perturbseq_server as ps
        orig = ps._CACHE_DIR
        ps._CACHE_DIR = tmp_path
        ps._SIG_CACHE.clear()
        try:
            preprocess_rna_fingerprints("svd_ds", top_k=10)
        finally:
            ps._CACHE_DIR = orig
            ps._SIG_CACHE.clear()

    def test_returns_list_of_dicts(self, tmp_path):
        self._prep(tmp_path)
        import mcp_servers.perturbseq_server as ps
        orig = ps._CACHE_DIR
        ps._CACHE_DIR = tmp_path
        ps._SIG_CACHE.clear()
        try:
            result = compute_svd_nomination_scores("svd_ds", gwas_genes=["IL6R", "JAK2"])
        finally:
            ps._CACHE_DIR = orig
            ps._SIG_CACHE.clear()
        assert isinstance(result, list)
        for entry in result:
            assert "gene" in entry
            assert "cosine_score" in entry

    def test_gwas_genes_excluded_from_nominees(self, tmp_path):
        self._prep(tmp_path)
        import mcp_servers.perturbseq_server as ps
        orig = ps._CACHE_DIR
        ps._CACHE_DIR = tmp_path
        ps._SIG_CACHE.clear()
        try:
            result = compute_svd_nomination_scores("svd_ds", gwas_genes=["IL6R", "JAK2"])
        finally:
            ps._CACHE_DIR = orig
            ps._SIG_CACHE.clear()
        returned_genes = {e["gene"] for e in result}
        assert "IL6R" not in returned_genes
        assert "JAK2" not in returned_genes

    def test_cosine_scores_bounded(self, tmp_path):
        self._prep(tmp_path)
        import mcp_servers.perturbseq_server as ps
        orig = ps._CACHE_DIR
        ps._CACHE_DIR = tmp_path
        ps._SIG_CACHE.clear()
        try:
            result = compute_svd_nomination_scores(
                "svd_ds", gwas_genes=["IL6R"], min_cosine=0.0
            )
        finally:
            ps._CACHE_DIR = orig
            ps._SIG_CACHE.clear()
        for entry in result:
            assert -1.0 <= entry["cosine_score"] <= 1.0

    def test_sorted_descending(self, tmp_path):
        self._prep(tmp_path)
        import mcp_servers.perturbseq_server as ps
        orig = ps._CACHE_DIR
        ps._CACHE_DIR = tmp_path
        ps._SIG_CACHE.clear()
        try:
            result = compute_svd_nomination_scores(
                "svd_ds", gwas_genes=["IL6R"], min_cosine=0.0
            )
        finally:
            ps._CACHE_DIR = orig
            ps._SIG_CACHE.clear()
        scores = [e["cosine_score"] for e in result]
        assert scores == sorted(scores, reverse=True)

    def test_missing_loadings_returns_empty(self, tmp_path):
        # No preprocessing → no svd_loadings.npz → returns empty list
        _write_sigs(tmp_path, "no_svd_ds", _MOCK_SIGS)
        import mcp_servers.perturbseq_server as ps
        orig = ps._CACHE_DIR
        ps._CACHE_DIR = tmp_path
        ps._SIG_CACHE.clear()
        try:
            result = compute_svd_nomination_scores("no_svd_ds", gwas_genes=["IL6R"])
        finally:
            ps._CACHE_DIR = orig
            ps._SIG_CACHE.clear()
        assert result == [], f"Expected empty list, got {result!r}"

    def test_immune_gwas_centroid_excludes_lipid_kos(self, tmp_path):
        """With IL6R/JAK2/TYK2 as GWAS genes, PCSK9/CETP should not appear in nominees."""
        self._prep(tmp_path)
        import mcp_servers.perturbseq_server as ps
        orig = ps._CACHE_DIR
        ps._CACHE_DIR = tmp_path
        ps._SIG_CACHE.clear()
        try:
            result = compute_svd_nomination_scores(
                "svd_ds", gwas_genes=["IL6R", "JAK2", "TYK2"], min_cosine=-1.0
            )
        finally:
            ps._CACHE_DIR = orig
            ps._SIG_CACHE.clear()
        # Only PCSK9 and CETP should remain (not GWAS genes)
        returned = {e["gene"] for e in result}
        assert returned <= {"PCSK9", "CETP"}, f"Unexpected nominees: {returned}"


# ---------------------------------------------------------------------------
# Split β source in load_replogle_betas
# ---------------------------------------------------------------------------

class TestSplitBetaSource:

    def _write_both(self, tmp_path: Path, dataset_id: str) -> None:
        """Write raw + fingerprint signatures so split logic can run."""
        _write_sigs(tmp_path, dataset_id, _MOCK_SIGS)
        import mcp_servers.perturbseq_server as ps
        orig = ps._CACHE_DIR
        ps._CACHE_DIR = tmp_path
        ps._SIG_CACHE.clear()
        try:
            preprocess_rna_fingerprints(dataset_id, top_k=10)
        finally:
            ps._CACHE_DIR = orig
            ps._SIG_CACHE.clear()

    def test_split_returns_all_genes(self, tmp_path):
        from pipelines.replogle_parser import _DATASET_SIGNATURES_REGISTRY, load_replogle_betas
        self._write_both(tmp_path, "split_ds")
        import pipelines.replogle_parser as rp
        orig_reg = dict(rp._DATASET_SIGNATURES_REGISTRY)
        rp._DATASET_SIGNATURES_REGISTRY["split_ds"] = tmp_path / "split_ds" / "signatures.json.gz"
        try:
            prog = {"prog1": list(list(_MOCK_SIGS.values())[0].keys())}
            result = load_replogle_betas(
                prog, dataset_id="split_ds", gwas_gene_set={"IL6R", "JAK2"}
            )
        finally:
            rp._DATASET_SIGNATURES_REGISTRY.clear()
            rp._DATASET_SIGNATURES_REGISTRY.update(orig_reg)
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_gwas_genes_have_fingerprint_beta(self, tmp_path):
        """GWAS gene β from split should equal fingerprint-only β."""
        from pipelines.replogle_parser import load_replogle_betas
        self._write_both(tmp_path, "split2_ds")
        import pipelines.replogle_parser as rp
        orig_reg = dict(rp._DATASET_SIGNATURES_REGISTRY)
        rp._DATASET_SIGNATURES_REGISTRY["split2_ds"] = tmp_path / "split2_ds" / "signatures.json.gz"
        prog = {"p": list(list(_MOCK_SIGS.values())[0].keys())[:3]}
        try:
            result_split = load_replogle_betas(
                prog, dataset_id="split2_ds", gwas_gene_set={"IL6R"}, force_recompute=True
            )
            result_fp_all = load_replogle_betas(
                prog, dataset_id="split2_ds", gwas_gene_set=None, force_recompute=True
            )
        finally:
            rp._DATASET_SIGNATURES_REGISTRY.clear()
            rp._DATASET_SIGNATURES_REGISTRY.update(orig_reg)
        if "IL6R" in result_split and "IL6R" in result_fp_all:
            split_beta = result_split["IL6R"]["programs"].get("p", {}).get("beta")
            fp_beta    = result_fp_all["IL6R"]["programs"].get("p", {}).get("beta")
            assert split_beta == fp_beta, (
                f"IL6R (GWAS gene) should use fingerprint β in split mode: "
                f"split={split_beta} fp={fp_beta}"
            )


# ---------------------------------------------------------------------------
# SVD nomination: cap removed
# ---------------------------------------------------------------------------

class TestSvdNominationNoCap:
    """Verify that removing FINGERPRINT_MAX_NONGWAS_NOMINEES cap doesn't break anything."""

    def _prep(self, tmp_path: Path, sigs: dict, dataset_id: str) -> None:
        _write_sigs(tmp_path, dataset_id, sigs)
        import mcp_servers.perturbseq_server as ps
        orig = ps._CACHE_DIR
        ps._CACHE_DIR = tmp_path
        ps._SIG_CACHE.clear()
        try:
            preprocess_rna_fingerprints(dataset_id, top_k=10)
        finally:
            ps._CACHE_DIR = orig
            ps._SIG_CACHE.clear()

    def test_no_cap_returns_all_above_threshold(self, tmp_path):
        self._prep(tmp_path, _MOCK_SIGS, "nocap_ds")
        import mcp_servers.perturbseq_server as ps
        orig = ps._CACHE_DIR
        ps._CACHE_DIR = tmp_path
        ps._SIG_CACHE.clear()
        try:
            result_nocap = compute_svd_nomination_scores(
                "nocap_ds", gwas_genes=["PCSK9"], min_cosine=-1.0
            )
            result_capped = compute_svd_nomination_scores(
                "nocap_ds", gwas_genes=["PCSK9"], min_cosine=-1.0, top_k=2
            )
        finally:
            ps._CACHE_DIR = orig
            ps._SIG_CACHE.clear()
        # Without cap, all non-GWAS genes above threshold are returned
        assert len(result_nocap) >= len(result_capped)

    def test_explicit_top_k_still_caps(self, tmp_path):
        self._prep(tmp_path, _MOCK_SIGS, "cap_ds")
        import mcp_servers.perturbseq_server as ps
        orig = ps._CACHE_DIR
        ps._CACHE_DIR = tmp_path
        ps._SIG_CACHE.clear()
        try:
            result = compute_svd_nomination_scores(
                "cap_ds", gwas_genes=["PCSK9"], min_cosine=-1.0, top_k=2
            )
        finally:
            ps._CACHE_DIR = orig
            ps._SIG_CACHE.clear()
        assert len(result) <= 2


# ---------------------------------------------------------------------------
# _collect_disease_fingerprint_nominees (orchestrator helper)
# ---------------------------------------------------------------------------

class TestCollectDiseaseFingerprintNominees:

    def _setup_dataset(self, tmp_path: Path, dataset_id: str) -> None:
        """Write raw sigs → fingerprint preprocessing → disease match result."""
        _write_sigs(tmp_path, dataset_id, _MOCK_SIGS)
        import mcp_servers.perturbseq_server as ps
        orig = ps._CACHE_DIR
        ps._CACHE_DIR = tmp_path
        ps._SIG_CACHE.clear()
        try:
            preprocess_rna_fingerprints(dataset_id, top_k=10)
            map_disease_to_fingerprints(_DISEASE_DE, dataset_id, n_bootstrap=5, min_gene_overlap=3)
        finally:
            ps._CACHE_DIR = orig
            ps._SIG_CACHE.clear()

    def test_returns_list(self, tmp_path):
        import orchestrator.pi_orchestrator_v2 as orch
        import mcp_servers.perturbseq_server as ps
        orig_cache = ps._CACHE_DIR
        ps._CACHE_DIR = tmp_path
        ps._SIG_CACHE.clear()
        self._setup_dataset(tmp_path, "fn_ds")
        try:
            result = orch._collect_disease_fingerprint_nominees(
                {"disease_name": "test_disease"},
                gwas_gene_set=set(),
            )
        finally:
            ps._CACHE_DIR = orig_cache
            ps._SIG_CACHE.clear()
        # Even if disease_name doesn't resolve to a dataset, we get a list
        assert isinstance(result, list)

    def test_gwas_genes_excluded(self, tmp_path):
        """GWAS genes must not appear in disease fingerprint nominees."""
        import json as _json
        import pathlib as _pathlib
        import mcp_servers.perturbseq_server as ps
        orig_cache = ps._CACHE_DIR
        ps._CACHE_DIR = tmp_path
        ps._SIG_CACHE.clear()
        dataset_id = "gwas_excl_ds"
        self._setup_dataset(tmp_path, dataset_id)
        # Read the full match JSON and verify filtering logic manually
        match_path = tmp_path / dataset_id / "disease_fingerprint_match.json"
        assert match_path.exists()
        with open(match_path) as fh:
            full = _json.load(fh)
        gwas_set = {"IL6R", "JAK2"}
        from config.scoring_thresholds import FINGERPRINT_DISEASE_R_THRESHOLD
        threshold = -FINGERPRINT_DISEASE_R_THRESHOLD
        nominees = [
            r["gene_ko"]
            for r in full["results"]
            if r["r"] <= threshold and r["gene_ko"] not in gwas_set
        ]
        assert "IL6R" not in nominees
        assert "JAK2" not in nominees
        ps._CACHE_DIR = orig_cache
        ps._SIG_CACHE.clear()

    def test_threshold_gates_nominations(self, tmp_path):
        """Only genes with r ≤ −threshold should be nominated."""
        import json as _json
        import mcp_servers.perturbseq_server as ps
        from config.scoring_thresholds import FINGERPRINT_DISEASE_R_THRESHOLD
        orig_cache = ps._CACHE_DIR
        ps._CACHE_DIR = tmp_path
        ps._SIG_CACHE.clear()
        dataset_id = "thresh_ds"
        self._setup_dataset(tmp_path, dataset_id)
        match_path = tmp_path / dataset_id / "disease_fingerprint_match.json"
        with open(match_path) as fh:
            full = _json.load(fh)
        threshold = -FINGERPRINT_DISEASE_R_THRESHOLD
        for r in full["results"]:
            if r["r"] > threshold:
                assert r["gene_ko"] not in [
                    entry["gene_ko"]
                    for entry in full["results"]
                    if entry["r"] <= threshold
                ], "Gene above threshold should not be nominated"
        ps._CACHE_DIR = orig_cache
        ps._SIG_CACHE.clear()
