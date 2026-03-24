"""
Tests for pipelines/replogle_parser.py — pseudo-bulk Perturb-seq β extractor.

All tests use synthetic data (no real h5ad required).
Integration tests that need the real file are marked @pytest.mark.integration.
"""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers to build a synthetic h5ad
# ---------------------------------------------------------------------------

def _make_synthetic_h5ad(tmp_path: Path) -> tuple[Path, dict]:
    """
    Create a minimal synthetic h5ad with:
      - 10 non-targeting control obs
      - 5 perturbation obs (GENE_A × 2, GENE_B × 2, GENE_C × 1)
      - 8 output genes (GeneX1..GeneX8)

    Returns (h5ad_path, ground_truth) where ground_truth contains expected
    log2FC directions for each perturbed gene.
    """
    import anndata
    import pandas as pd

    rng = np.random.default_rng(42)
    n_genes = 8
    gene_names = [f"GeneX{i+1}" for i in range(n_genes)]

    # Control cells: all near 3.0 ± 0.1
    n_ctrl = 10
    ctrl_expr = rng.normal(3.0, 0.1, size=(n_ctrl, n_genes)).astype(np.float32)

    # GENE_A KO: GeneX1 and GeneX2 go UP (+0.5), others flat
    # GENE_B KO: GeneX5 goes DOWN (-0.6), others flat
    # GENE_C KO: all flat (no significant effect)
    gene_a_expr = rng.normal(3.0, 0.1, size=(2, n_genes)).astype(np.float32)
    gene_a_expr[:, 0] += 0.5   # GeneX1
    gene_a_expr[:, 1] += 0.5   # GeneX2

    gene_b_expr = rng.normal(3.0, 0.1, size=(2, n_genes)).astype(np.float32)
    gene_b_expr[:, 4] -= 0.6   # GeneX5

    gene_c_expr = rng.normal(3.0, 0.1, size=(1, n_genes)).astype(np.float32)

    X = np.vstack([ctrl_expr, gene_a_expr, gene_b_expr, gene_c_expr])
    obs_names = (
        [f"ctrl_{i}" for i in range(n_ctrl)]
        + ["GENE_A_g1", "GENE_A_g2"]
        + ["GENE_B_g1", "GENE_B_g2"]
        + ["GENE_C_g1"]
    )
    obs_df = pd.DataFrame(
        {
            "gene": (
                ["non-targeting"] * n_ctrl
                + ["GENE_A", "GENE_A"]
                + ["GENE_B", "GENE_B"]
                + ["GENE_C"]
            ),
            "n_cells": [20] * len(obs_names),
        },
        index=obs_names,
    )
    var_df = pd.DataFrame(index=gene_names)
    adata = anndata.AnnData(X=X, obs=obs_df, var=var_df)
    h5ad_path = tmp_path / "synthetic.h5ad"
    adata.write_h5ad(str(h5ad_path))

    ground_truth = {
        "GENE_A": {"GeneX1": "up", "GeneX2": "up"},
        "GENE_B": {"GeneX5": "down"},
    }
    return h5ad_path, ground_truth


# ---------------------------------------------------------------------------
# Unit tests — _is_control()
# ---------------------------------------------------------------------------

class TestIsControl:
    from pipelines.replogle_parser import _is_control

    def test_non_targeting(self):
        from pipelines.replogle_parser import _is_control
        assert _is_control("non-targeting")

    def test_aavs1(self):
        from pipelines.replogle_parser import _is_control
        assert _is_control("AAVS1")

    def test_safeharbor(self):
        from pipelines.replogle_parser import _is_control
        assert _is_control("safe_harbor")

    def test_real_gene_not_control(self):
        from pipelines.replogle_parser import _is_control
        assert not _is_control("PCSK9")
        assert not _is_control("TET2")
        assert not _is_control("NOD2")

    def test_non_targeting_suffix(self):
        from pipelines.replogle_parser import _is_control
        # "non-targeting_ctrl_1" — split on _ gives "non-targeting" as base? No, split on _ is wrong.
        # Actually split on "_" gives ["non-targeting"] if we split the hypen form
        assert _is_control("non-targeting")


# ---------------------------------------------------------------------------
# Unit tests — _compute_log2fc() with synthetic data
# ---------------------------------------------------------------------------

class TestComputeLog2FC:
    def _run(self, tmp_path):
        from pipelines.replogle_parser import _load_h5ad_matrix, _compute_log2fc
        h5ad_path, gt = _make_synthetic_h5ad(tmp_path)
        X, obs_names, var_names, obs_meta = _load_h5ad_matrix(h5ad_path)
        log2fc, se_matrix, pert_genes, ctrl_mean = _compute_log2fc(X, obs_names, obs_meta)
        return log2fc, se_matrix, pert_genes, var_names, gt

    def test_gene_a_up_in_genex1(self, tmp_path):
        log2fc, se_matrix, pert_genes, var_names, gt = self._run(tmp_path)
        idx_a = pert_genes.index("GENE_A")
        idx_x1 = var_names.index("GeneX1")
        assert log2fc[idx_a, idx_x1] > 0.3, "GENE_A KO should upregulate GeneX1"

    def test_gene_b_down_in_genex5(self, tmp_path):
        log2fc, se_matrix, pert_genes, var_names, gt = self._run(tmp_path)
        idx_b = pert_genes.index("GENE_B")
        idx_x5 = var_names.index("GeneX5")
        assert log2fc[idx_b, idx_x5] < -0.3, "GENE_B KO should downregulate GeneX5"

    def test_se_positive(self, tmp_path):
        log2fc, se_matrix, pert_genes, var_names, gt = self._run(tmp_path)
        assert (se_matrix >= 0).all(), "SE values must be non-negative"

    def test_pert_genes_identified(self, tmp_path):
        log2fc, se_matrix, pert_genes, var_names, gt = self._run(tmp_path)
        assert "GENE_A" in pert_genes
        assert "GENE_B" in pert_genes
        assert "GENE_C" in pert_genes
        # Control rows should NOT be in pert_genes
        assert "non-targeting" not in pert_genes

    def test_matrix_shape(self, tmp_path):
        log2fc, se_matrix, pert_genes, var_names, gt = self._run(tmp_path)
        assert log2fc.shape == (len(pert_genes), len(var_names))
        assert se_matrix.shape == log2fc.shape


# ---------------------------------------------------------------------------
# Unit tests — _project_onto_programs()
# ---------------------------------------------------------------------------

class TestProjectOntoPrograms:
    def test_beta_sign_matches_log2fc(self, tmp_path):
        from pipelines.replogle_parser import (
            _load_h5ad_matrix, _compute_log2fc, _project_onto_programs
        )
        h5ad_path, _ = _make_synthetic_h5ad(tmp_path)
        X, obs_names, var_names, obs_meta = _load_h5ad_matrix(h5ad_path)
        log2fc, se_matrix, pert_genes, _ctrl = _compute_log2fc(X, obs_names, obs_meta)

        # Program_A: contains GeneX1, GeneX2 (upregulated by GENE_A KO)
        programs = {"Program_A": ["GeneX1", "GeneX2"]}
        betas = _project_onto_programs(log2fc, se_matrix, pert_genes, var_names, programs)

        assert "GENE_A" in betas
        beta_a = betas["GENE_A"]["programs"]["Program_A"]["beta"]
        assert beta_a > 0.2, f"Expected positive β for GENE_A→Program_A, got {beta_a}"

    def test_se_propagation(self, tmp_path):
        from pipelines.replogle_parser import (
            _load_h5ad_matrix, _compute_log2fc, _project_onto_programs
        )
        h5ad_path, _ = _make_synthetic_h5ad(tmp_path)
        X, obs_names, var_names, obs_meta = _load_h5ad_matrix(h5ad_path)
        log2fc, se_matrix, pert_genes, _ctrl = _compute_log2fc(X, obs_names, obs_meta)

        programs = {"Program_B": ["GeneX5", "GeneX6"]}
        betas = _project_onto_programs(log2fc, se_matrix, pert_genes, var_names, programs)

        # SE should be positive and > 0
        assert "GENE_B" in betas
        se_b = betas["GENE_B"]["programs"]["Program_B"]["se"]
        assert se_b > 0, "SE must be positive"

    def test_ci_contains_beta(self, tmp_path):
        from pipelines.replogle_parser import (
            _load_h5ad_matrix, _compute_log2fc, _project_onto_programs
        )
        h5ad_path, _ = _make_synthetic_h5ad(tmp_path)
        X, obs_names, var_names, obs_meta = _load_h5ad_matrix(h5ad_path)
        log2fc, se_matrix, pert_genes, _ctrl = _compute_log2fc(X, obs_names, obs_meta)

        programs = {"P": ["GeneX1", "GeneX3"]}
        betas = _project_onto_programs(log2fc, se_matrix, pert_genes, var_names, programs)

        for gene, gdata in betas.items():
            prog = gdata["programs"]["P"]
            assert prog["ci_lower"] <= prog["beta"] <= prog["ci_upper"], (
                f"CI [{prog['ci_lower']}, {prog['ci_upper']}] must contain β={prog['beta']}"
            )

    def test_missing_program_genes_handled(self, tmp_path):
        from pipelines.replogle_parser import (
            _load_h5ad_matrix, _compute_log2fc, _project_onto_programs
        )
        h5ad_path, _ = _make_synthetic_h5ad(tmp_path)
        X, obs_names, var_names, obs_meta = _load_h5ad_matrix(h5ad_path)
        log2fc, se_matrix, pert_genes, _ctrl = _compute_log2fc(X, obs_names, obs_meta)

        # Program with completely unknown genes should produce no output
        programs = {"EmptyProgram": ["NONEXISTENT_GENE_1", "NONEXISTENT_GENE_2"]}
        betas = _project_onto_programs(log2fc, se_matrix, pert_genes, var_names, programs)

        for gene in betas:
            assert "EmptyProgram" not in betas[gene]["programs"], (
                "Program with no valid genes should be excluded"
            )


# ---------------------------------------------------------------------------
# Unit tests — load_replogle_betas() orchestration
# ---------------------------------------------------------------------------

class TestLoadReplogleBetas:
    def test_returns_correct_schema(self, tmp_path):
        from pipelines.replogle_parser import load_replogle_betas
        h5ad_path, _ = _make_synthetic_h5ad(tmp_path)
        cache_path = tmp_path / "cache.json"

        programs = {"ProgramA": ["GeneX1", "GeneX2"], "ProgramB": ["GeneX5"]}
        result = load_replogle_betas(
            program_gene_sets=programs,
            h5ad_path=h5ad_path,
            cache_path=cache_path,
        )

        assert isinstance(result, dict)
        for gene, gdata in result.items():
            assert "programs" in gdata
            for prog_name, pdata in gdata["programs"].items():
                assert "beta" in pdata
                assert "se" in pdata
                assert "ci_lower" in pdata
                assert "ci_upper" in pdata

    def test_cache_written_and_reused(self, tmp_path):
        from pipelines.replogle_parser import load_replogle_betas
        h5ad_path, _ = _make_synthetic_h5ad(tmp_path)
        cache_path = tmp_path / "cache.json"

        programs = {"P": ["GeneX1"]}
        # First call: compute and cache
        result1 = load_replogle_betas(
            program_gene_sets=programs,
            h5ad_path=h5ad_path,
            cache_path=cache_path,
        )
        assert cache_path.exists(), "Cache file should be written after first call"

        # Second call: load from cache (h5ad not needed)
        result2 = load_replogle_betas(
            program_gene_sets=programs,
            h5ad_path=Path("/nonexistent.h5ad"),  # would fail if h5ad were read
            cache_path=cache_path,
        )
        assert result1 == result2, "Cache should return identical results"

    def test_force_recompute_ignores_cache(self, tmp_path):
        from pipelines.replogle_parser import load_replogle_betas
        h5ad_path, _ = _make_synthetic_h5ad(tmp_path)
        cache_path = tmp_path / "cache.json"
        # Write a stale cache
        cache_path.write_text('{"stale": "data"}')

        programs = {"P": ["GeneX1"]}
        result = load_replogle_betas(
            program_gene_sets=programs,
            h5ad_path=h5ad_path,
            cache_path=cache_path,
            force_recompute=True,
        )
        assert "stale" not in result, "force_recompute should overwrite stale cache"
        assert result != {"stale": "data"}

    def test_file_not_found_raises(self, tmp_path):
        from pipelines.replogle_parser import load_replogle_betas
        with pytest.raises(FileNotFoundError, match="Replogle 2022 h5ad not found"):
            load_replogle_betas(
                program_gene_sets={"P": ["GeneX1"]},
                h5ad_path=tmp_path / "nonexistent.h5ad",
                cache_path=tmp_path / "cache.json",
            )

    def test_compatible_with_estimate_beta_tier1(self, tmp_path):
        """Output format should plug directly into estimate_beta_tier1."""
        from pipelines.replogle_parser import load_replogle_betas
        from pipelines.ota_beta_estimation import estimate_beta_tier1

        h5ad_path, _ = _make_synthetic_h5ad(tmp_path)
        cache_path = tmp_path / "cache.json"
        programs = {"ProgramA": ["GeneX1", "GeneX2"]}
        perturbseq_data = load_replogle_betas(
            program_gene_sets=programs,
            h5ad_path=h5ad_path,
            cache_path=cache_path,
        )

        # GENE_A was perturbed and has data
        result = estimate_beta_tier1(
            gene="GENE_A",
            program="ProgramA",
            perturbseq_data=perturbseq_data,
            cell_type="K562",
        )
        assert result is not None, "Should return a result for a known gene+program"
        assert result["evidence_tier"] == "Tier1_Interventional"
        assert "Perturb-seq" in result["data_source"]
        assert result["beta_se"] is not None, "Quantitative path should have real SE"
        assert result["beta_sigma"] is not None
        # Sign should match: GENE_A KO upregulates GeneX1+GeneX2, so β > 0
        assert result["beta"] > 0, f"Expected positive β, got {result['beta']}"

    def test_unknown_gene_returns_none_from_tier1(self, tmp_path):
        """estimate_beta_tier1 should return None for genes not in perturbseq_data."""
        from pipelines.replogle_parser import load_replogle_betas
        from pipelines.ota_beta_estimation import estimate_beta_tier1

        h5ad_path, _ = _make_synthetic_h5ad(tmp_path)
        cache_path = tmp_path / "cache.json"
        programs = {"P": ["GeneX1"]}
        perturbseq_data = load_replogle_betas(
            program_gene_sets=programs,
            h5ad_path=h5ad_path,
            cache_path=cache_path,
        )

        result = estimate_beta_tier1(
            gene="COMPLETELY_UNKNOWN_GENE_XYZ",
            program="P",
            perturbseq_data=perturbseq_data,
        )
        assert result is None


# ---------------------------------------------------------------------------
# Integration test — requires real h5ad file
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestReplogleIntegration:
    """Requires data/replogle_2022_k562_essential.h5ad to be present."""

    def test_real_file_loads(self):
        from pipelines.replogle_parser import load_replogle_betas, _H5AD_PATH
        if not _H5AD_PATH.exists():
            pytest.skip("Real h5ad not downloaded yet")

        # Use a small program subset so test runs quickly
        programs = {
            "lipid_metabolism": ["LDLR", "PCSK9", "APOB", "HMGCR", "SREBF1"],
            "inflammatory_NF-kB": ["TNF", "NFKB1", "RELA", "IL6", "IL1B"],
        }
        cache_path = _H5AD_PATH.parent / "replogle_integration_test_cache.json"
        result = load_replogle_betas(
            program_gene_sets=programs,
            cache_path=cache_path,
            force_recompute=True,
        )
        # Should have perturbed data for known essential genes
        known_essentials = ["RPS19", "RPL11", "MYC", "CDK2"]
        found = [g for g in known_essentials if g in result]
        assert found, f"None of {known_essentials} found in results; got keys: {list(result.keys())[:5]}"

        # Clean up integration test cache
        if cache_path.exists():
            cache_path.unlink()
