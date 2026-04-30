"""
Tests for perturbseq_server.py.

Unit tests mock signature loading. Integration tests require preprocessed datasets on disk.
"""
from __future__ import annotations

import gzip
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp_servers.perturbseq_server import (
    _DATASET_REGISTRY,
    _DISEASE_DATASET_PRIORITY,
    _compute_program_beta,
    _select_dataset,
    _sig_path,
    compute_perturbseq_program_beta,
    get_perturbseq_signature,
    list_perturbseq_datasets,
    preprocess_h5ad,
)

# ---------------------------------------------------------------------------
# Minimal mock signature fixture
# ---------------------------------------------------------------------------

_MOCK_SIG = {
    "PCSK9": {"LDLR": 1.5, "HMGCR": 0.8, "APOB": 0.6, "LPL": -0.4},
    "IL6R":  {"STAT3": 1.2, "JAK1": 0.9, "SOCS3": 0.7},
}


def _patch_load(dataset_id):
    return _MOCK_SIG if dataset_id == "replogle_2022_k562" else None


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestRegistry:

    def test_all_required_fields_present(self):
        required = {"name", "citation", "cell_line", "tissue", "disease_context",
                    "perturbation_type", "n_genes_perturbed", "access",
                    "h5ad_obs_perturbation_col", "h5ad_nt_label"}
        for ds_id, meta in _DATASET_REGISTRY.items():
            missing = required - set(meta.keys())
            assert not missing, f"{ds_id} missing fields: {missing}"

    def test_disease_priorities_reference_valid_datasets(self):
        for disease, priority in _DISEASE_DATASET_PRIORITY.items():
            for ds_id in priority:
                assert ds_id in _DATASET_REGISTRY, \
                    f"{disease} priority references unknown dataset: {ds_id}"

    def test_active_diseases_covered(self):
        for disease in ("CAD", "RA", "GENERIC"):
            assert disease in _DISEASE_DATASET_PRIORITY

    def test_cad_vascular_first(self):
        # Schnitzler 2023 (332 CAD-specific vascular genes) is highest priority
        cad_priority = _DISEASE_DATASET_PRIORITY["CAD"]
        assert cad_priority[0] == "schnitzler_cad_vascular"

    def test_generic_has_replogle_k562(self):
        assert "replogle_2022_k562" in _DISEASE_DATASET_PRIORITY["GENERIC"]

    def test_n_genes_positive(self):
        for ds_id, meta in _DATASET_REGISTRY.items():
            assert meta["n_genes_perturbed"] > 0, f"{ds_id} has n_genes_perturbed=0"


# ---------------------------------------------------------------------------
# Dataset selection
# ---------------------------------------------------------------------------

class TestSelectDataset:

    def test_explicit_dataset_id_overrides_disease(self):
        result = _select_dataset("CAD", "replogle_2022_k562")
        assert result == "replogle_2022_k562"

    def test_unknown_disease_falls_back_to_generic(self):
        result = _select_dataset("UNKNOWN_DISEASE", None)
        # Should return first in GENERIC priority
        assert result == _DISEASE_DATASET_PRIORITY["GENERIC"][0]

    def test_none_disease_uses_generic(self):
        result = _select_dataset(None, None)
        assert result is not None

    def test_cached_dataset_selected_preferentially(self, tmp_path):
        # Create a fake cached signature for replogle (not natsume_2023_rpe)
        fake_cache = tmp_path / "replogle_2022_k562" / "signatures.json.gz"
        fake_cache.parent.mkdir(parents=True)
        with gzip.open(fake_cache, "wt") as f:
            json.dump({}, f)

        with patch("mcp_servers.perturbseq_server._CACHE_DIR", tmp_path):
            from mcp_servers import perturbseq_server as ps
            orig_cache_dir = ps._CACHE_DIR
            ps._CACHE_DIR = tmp_path
            try:
                result = _select_dataset("GENERIC", None)
                # replogle is cached → should be selected
                assert result == "replogle_2022_k562"
            finally:
                ps._CACHE_DIR = orig_cache_dir


# ---------------------------------------------------------------------------
# Beta computation
# ---------------------------------------------------------------------------

class TestComputeProgramBeta:

    def test_basic_beta(self):
        sig = {"A": 1.0, "B": 2.0, "C": -1.0}
        beta = _compute_program_beta(sig, {"A", "B"})
        assert beta == pytest.approx((1.0 + 2.0) / 2)  # mean over program (size=2)

    def test_beta_uses_program_size_denominator(self):
        """β = sum(hits) / len(pg_set), not sum(hits) / len(hits)."""
        sig = {"A": 2.0}
        # program has 2 genes, only 1 in signature
        beta = _compute_program_beta(sig, {"A", "MISSING"})
        assert beta == pytest.approx(2.0 / 2)

    def test_coverage_below_5pct_returns_none(self):
        # 1 hit out of 25 = 4% < 5%
        sig = {"A": 1.0}
        program = {f"G{i}" for i in range(25)}
        program.add("A")
        result = _compute_program_beta(sig, program)
        assert result is None

    def test_empty_signature_returns_none(self):
        result = _compute_program_beta({}, {"A", "B"})
        assert result is None

    def test_empty_program_returns_none(self):
        result = _compute_program_beta({"A": 1.0}, set())
        assert result is None

    def test_negative_beta(self):
        sig = {"A": -2.0, "B": -1.0}
        beta = _compute_program_beta(sig, {"A", "B"})
        assert beta < 0


# ---------------------------------------------------------------------------
# get_perturbseq_signature
# ---------------------------------------------------------------------------

class TestGetPerturbseqSignature:

    @patch("mcp_servers.perturbseq_server._load_cached_signatures", side_effect=_patch_load)
    def test_returns_signature_for_known_gene(self, mock_load):
        result = get_perturbseq_signature("PCSK9", dataset_id="replogle_2022_k562")
        assert result["n_genes_measured"] > 0
        assert "LDLR" in result["signature"]

    @patch("mcp_servers.perturbseq_server._load_cached_signatures", side_effect=_patch_load)
    def test_evidence_tier_present(self, mock_load):
        result = get_perturbseq_signature("PCSK9", dataset_id="replogle_2022_k562")
        assert result["evidence_tier"] == "Tier3_Provisional"

    @patch("mcp_servers.perturbseq_server._load_cached_signatures", side_effect=_patch_load)
    def test_gene_not_found_returns_empty_signature(self, mock_load):
        result = get_perturbseq_signature("UNKNOWN_GENE", dataset_id="replogle_2022_k562")
        assert result["signature"] == {}
        assert result["n_genes_measured"] == 0
        assert "not_found" in result["source"]

    @patch("mcp_servers.perturbseq_server._load_cached_signatures", return_value=None)
    def test_not_cached_returns_not_cached_source(self, mock_load):
        result = get_perturbseq_signature("PCSK9", dataset_id="replogle_2022_k562")
        assert result["source"] == "not_cached"
        assert "preprocess" in result["note"].lower()

    @patch("mcp_servers.perturbseq_server._load_cached_signatures", side_effect=_patch_load)
    def test_top_k_respected(self, mock_load):
        result = get_perturbseq_signature("PCSK9", dataset_id="replogle_2022_k562", top_k=2)
        assert result["n_genes_measured"] <= 2

    @patch("mcp_servers.perturbseq_server._load_cached_signatures", side_effect=_patch_load)
    def test_returns_dataset_id_and_cell_line(self, mock_load):
        result = get_perturbseq_signature("PCSK9", dataset_id="replogle_2022_k562")
        assert result["dataset_id"] == "replogle_2022_k562"
        assert result["cell_line"] == "K562"

    @patch("mcp_servers.perturbseq_server._load_cached_signatures", side_effect=_patch_load)
    def test_case_insensitive_gene_lookup(self, mock_load):
        result = get_perturbseq_signature("pcsk9", dataset_id="replogle_2022_k562")
        assert result["n_genes_measured"] > 0


# ---------------------------------------------------------------------------
# compute_perturbseq_program_beta
# ---------------------------------------------------------------------------

class TestComputePerturbseqProgramBeta:

    @patch("mcp_servers.perturbseq_server._load_cached_signatures", side_effect=_patch_load)
    def test_beta_computed_for_known_gene(self, mock_load):
        result = compute_perturbseq_program_beta(
            "PCSK9", ["LDLR", "HMGCR", "APOB"], dataset_id="replogle_2022_k562"
        )
        assert result["beta"] is not None
        assert result["program_coverage"] > 0

    @patch("mcp_servers.perturbseq_server._load_cached_signatures", side_effect=_patch_load)
    def test_schema_fields_present(self, mock_load):
        result = compute_perturbseq_program_beta(
            "PCSK9", ["LDLR", "HMGCR"], dataset_id="replogle_2022_k562"
        )
        for field in ("gene", "program_coverage", "beta", "evidence_tier",
                      "cell_line", "dataset_id", "perturbation_type", "data_source", "note"):
            assert field in result, f"Missing field: {field}"

    @patch("mcp_servers.perturbseq_server._load_cached_signatures", side_effect=_patch_load)
    def test_beta_is_none_for_insufficient_coverage(self, mock_load):
        # 25-gene program, only LDLR overlaps (4% < 5%)
        program = [f"GENE{i}" for i in range(25)] + ["LDLR"]
        result = compute_perturbseq_program_beta(
            "PCSK9", program, dataset_id="replogle_2022_k562"
        )
        assert result["beta"] is None

    @patch("mcp_servers.perturbseq_server._load_cached_signatures", side_effect=_patch_load)
    def test_coverage_between_0_and_1(self, mock_load):
        result = compute_perturbseq_program_beta(
            "PCSK9", ["LDLR", "HMGCR", "FAKE1", "FAKE2"], dataset_id="replogle_2022_k562"
        )
        assert 0.0 <= result["program_coverage"] <= 1.0


# ---------------------------------------------------------------------------
# list_perturbseq_datasets
# ---------------------------------------------------------------------------

class TestListPerturbseqDatasets:

    def test_returns_all_datasets_when_no_filter(self):
        result = list_perturbseq_datasets()
        assert result["n_datasets"] == len(_DATASET_REGISTRY)

    def test_disease_filter_returns_priority_subset(self):
        result = list_perturbseq_datasets("CAD")
        ids = [r["dataset_id"] for r in result["datasets"]]
        assert "schnitzler_cad_vascular" in ids
        assert "natsume_2023_haec" in ids

    def test_cached_field_is_bool(self):
        result = list_perturbseq_datasets()
        for row in result["datasets"]:
            assert isinstance(row["cached"], bool)

    def test_preprocess_command_in_result(self):
        result = list_perturbseq_datasets()
        assert "preprocess" in result["preprocess_command"]


# ---------------------------------------------------------------------------
# preprocess_h5ad (unit test with synthetic h5ad)
# ---------------------------------------------------------------------------

class TestPreprocessH5ad:

    def test_unknown_dataset_id_returns_error(self, tmp_path):
        with tempfile.NamedTemporaryFile(suffix=".h5ad") as f:
            result = preprocess_h5ad("nonexistent_dataset", f.name)
        assert "error" in result

    def test_synthetic_h5ad_produces_signatures(self, tmp_path):
        """Create a minimal synthetic h5ad and verify preprocessing runs."""
        import anndata as ad
        import numpy as np
        import pandas as pd
        import scipy.sparse as sp

        # 100 cells: 30 NT controls + 20 cells × 3 perturbations (PCSK9, IL6R, HMGCR) + 10 extra NT
        n_nt = 40
        n_pert = 20
        genes = [f"GENE{i}" for i in range(50)]
        rng = np.random.default_rng(42)

        # NT cells: random expression
        X_nt = rng.random((n_nt, 50)).astype(np.float32)
        obs_nt = pd.DataFrame({"gene": ["non-targeting"] * n_nt})

        # Perturbed cells: add signal for first 5 genes
        perts = ["PCSK9", "IL6R", "HMGCR"]
        X_perts, obs_perts = [], []
        for p in perts:
            X_p = rng.random((n_pert, 50)).astype(np.float32)
            X_p[:, :5] += 2.0  # strong signal in first 5 genes
            X_perts.append(X_p)
            obs_perts.append(pd.DataFrame({"gene": [p] * n_pert}))

        X_all = np.vstack([X_nt] + X_perts)
        obs_all = pd.concat([obs_nt] + obs_perts, ignore_index=True)

        adata = ad.AnnData(
            X=sp.csr_matrix(X_all),
            obs=obs_all,
            var=pd.DataFrame(index=genes),
        )

        h5ad_path = tmp_path / "test.h5ad"
        adata.write_h5ad(str(h5ad_path))

        # Temporarily redirect cache dir
        import mcp_servers.perturbseq_server as ps
        orig = ps._CACHE_DIR
        ps._CACHE_DIR = tmp_path
        try:
            result = preprocess_h5ad(
                "replogle_2022_k562", str(h5ad_path),
                top_k=10, min_abs_log2fc=0.1,
            )
        finally:
            ps._CACHE_DIR = orig

        assert "error" not in result
        assert result["n_cached_genes"] == 3   # PCSK9, IL6R, HMGCR
        assert result["n_perturbed_genes"] == 3
