"""
Tests for Phase C:
  pipelines/state_space/therapeutic_redirection.py
  trajectory_scoring.py deprecation stub
"""
from __future__ import annotations

import math

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def simple_T():
    """3-state row-normalised T matrix: path(0) → path(1) | healthy(2)."""
    T = np.array([
        [0.0, 0.6, 0.4],   # path state 0: 60% → path1, 40% → healthy
        [0.3, 0.0, 0.7],   # path state 1: 30% → path0, 70% → healthy
        [0.1, 0.1, 0.0],   # healthy: small recirculation
    ], dtype=float)
    row_sums = T.sum(axis=1, keepdims=True)
    return T / np.where(row_sums == 0, 1.0, row_sums)


@pytest.fixture()
def transition_result_3state(simple_T):
    return {
        "transition_matrix":    simple_T,
        "state_labels":         ["path0", "path1", "healthy"],
        "pathologic_basin_ids": ["path0", "path1"],
        "healthy_basin_ids":    ["healthy"],
        "basin_assignments":    {"path0": "pathological", "path1": "pathological", "healthy": "healthy"},
    }


@pytest.fixture()
def program_loadings_tnf():
    from models.latent_mediator import ProgramLoading
    return [
        ProgramLoading(gene="TNF", program_id="IBD_macro_P00", cell_type="macrophage",
                       disease="IBD", nmf_loading=0.6, transition_de_signal=0.8,
                       p_loading=0.7 * 0.6 + 0.3 * 0.8),
    ]


@pytest.fixture()
def conditional_betas_tnf():
    from models.latent_mediator import ConditionalBeta
    return [
        ConditionalBeta(gene="TNF", program_id="IBD_macro_P00", cell_type="macrophage",
                        disease="IBD", beta=0.7, context_verified=True,
                        evidence_tier="Tier2_TrajectoryInferred"),
    ]


@pytest.fixture()
def conditional_gammas_ibd():
    from models.latent_mediator import ConditionalGamma
    return {
        "IBD_macro_P00": ConditionalGamma(
            program_id="IBD_macro_P00", trait="IBD", disease="IBD",
            gamma_gwas=0.39, gamma_transition=0.6, alpha=0.55, gamma_mixed=0.4845,
        ),
    }


# ---------------------------------------------------------------------------
# perturb_transition_matrix
# ---------------------------------------------------------------------------

class TestPerturbTransitionMatrix:
    def test_increases_healthy_flow(self, simple_T):
        from pipelines.state_space.therapeutic_redirection import perturb_transition_matrix
        T_pert = perturb_transition_matrix(simple_T, gene_beta=0.8, p_loading=0.7,
                                           path_idxs=[0, 1], healthy_idxs=[2])
        # Rows of path states should have MORE probability to healthy
        assert T_pert[0, 2] > simple_T[0, 2]
        assert T_pert[1, 2] > simple_T[1, 2]

    def test_row_normalised(self, simple_T):
        from pipelines.state_space.therapeutic_redirection import perturb_transition_matrix
        T_pert = perturb_transition_matrix(simple_T, gene_beta=0.9, p_loading=0.9,
                                           path_idxs=[0, 1], healthy_idxs=[2])
        row_sums = T_pert.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-9)

    def test_50_percent_cap(self, simple_T):
        from pipelines.state_space.therapeutic_redirection import perturb_transition_matrix
        # Very large beta should still be capped
        T_pert = perturb_transition_matrix(simple_T, gene_beta=100.0, p_loading=100.0,
                                           path_idxs=[0], healthy_idxs=[2])
        # T[0,1] should not drop below 50% of baseline
        # (cap: reduction ≤ 0.5 × T[0,1])
        min_allowed = simple_T[0, 1] * 0.5
        assert T_pert[0, 1] >= min_allowed - 1e-9

    def test_nan_beta_no_change(self, simple_T):
        from pipelines.state_space.therapeutic_redirection import perturb_transition_matrix
        T_pert = perturb_transition_matrix(simple_T, gene_beta=float("nan"), p_loading=0.5,
                                           path_idxs=[0], healthy_idxs=[2])
        np.testing.assert_array_equal(T_pert, simple_T)

    def test_tiny_p_loading_no_change(self, simple_T):
        from pipelines.state_space.therapeutic_redirection import perturb_transition_matrix
        T_pert = perturb_transition_matrix(simple_T, gene_beta=0.9, p_loading=1e-6,
                                           path_idxs=[0], healthy_idxs=[2])
        np.testing.assert_array_equal(T_pert, simple_T)

    def test_healthy_row_unchanged(self, simple_T):
        from pipelines.state_space.therapeutic_redirection import perturb_transition_matrix
        T_pert = perturb_transition_matrix(simple_T, gene_beta=0.8, p_loading=0.7,
                                           path_idxs=[0, 1], healthy_idxs=[2])
        # Healthy state row should not be modified (not a path source)
        np.testing.assert_array_almost_equal(T_pert[2], simple_T[2])

    def test_out_of_range_idx_safe(self, simple_T):
        from pipelines.state_space.therapeutic_redirection import perturb_transition_matrix
        # path_idxs contains index beyond matrix — should not crash
        T_pert = perturb_transition_matrix(simple_T, gene_beta=0.5, p_loading=0.5,
                                           path_idxs=[0, 99], healthy_idxs=[2])
        assert T_pert.shape == simple_T.shape


# ---------------------------------------------------------------------------
# compute_net_trajectory_improvement
# ---------------------------------------------------------------------------

class TestNetTrajectoryImprovement:
    def test_positive_for_beneficial_perturbation(self, simple_T):
        from pipelines.state_space.therapeutic_redirection import (
            perturb_transition_matrix, compute_net_trajectory_improvement
        )
        T_pert = perturb_transition_matrix(simple_T, gene_beta=0.8, p_loading=0.7,
                                           path_idxs=[0, 1], healthy_idxs=[2])
        net = compute_net_trajectory_improvement(simple_T, T_pert, [0, 1], [2])
        assert net > 0

    def test_zero_for_no_change(self, simple_T):
        from pipelines.state_space.therapeutic_redirection import compute_net_trajectory_improvement
        net = compute_net_trajectory_improvement(simple_T, simple_T.copy(), [0, 1], [2])
        assert net == 0.0

    def test_clipped_to_zero_not_negative(self, simple_T):
        from pipelines.state_space.therapeutic_redirection import compute_net_trajectory_improvement
        # Worsening T (manually decrease healthy flow)
        T_worse = simple_T.copy()
        T_worse[0, 2] = 0.1
        T_worse[0, 1] = 0.9
        net = compute_net_trajectory_improvement(simple_T, T_worse, [0, 1], [2])
        assert net == 0.0   # clipped

    def test_empty_basins_returns_zero(self, simple_T):
        from pipelines.state_space.therapeutic_redirection import compute_net_trajectory_improvement
        assert compute_net_trajectory_improvement(simple_T, simple_T, [], [2]) == 0.0
        assert compute_net_trajectory_improvement(simple_T, simple_T, [0], []) == 0.0


# ---------------------------------------------------------------------------
# compute_therapeutic_redirection_per_celltype
# ---------------------------------------------------------------------------

class TestRedirectionPerCelltype:
    def test_positive_score_for_good_target(
        self, transition_result_3state, program_loadings_tnf,
        conditional_betas_tnf, conditional_gammas_ibd
    ):
        from pipelines.state_space.therapeutic_redirection import (
            compute_therapeutic_redirection_per_celltype
        )
        result = compute_therapeutic_redirection_per_celltype(
            gene="TNF", disease="IBD", cell_type="macrophage",
            program_loadings=program_loadings_tnf,
            conditional_betas=conditional_betas_tnf,
            conditional_gammas=conditional_gammas_ibd,
            transition_result=transition_result_3state,
        )
        assert result["redirection"] > 0
        assert result["n_programs"] == 1
        assert len(result["provenance"]) == 1

    def test_zero_score_when_no_transition_matrix(
        self, program_loadings_tnf, conditional_betas_tnf, conditional_gammas_ibd
    ):
        from pipelines.state_space.therapeutic_redirection import (
            compute_therapeutic_redirection_per_celltype
        )
        result = compute_therapeutic_redirection_per_celltype(
            gene="TNF", disease="IBD", cell_type="macrophage",
            program_loadings=program_loadings_tnf,
            conditional_betas=conditional_betas_tnf,
            conditional_gammas=conditional_gammas_ibd,
            transition_result={},  # empty
        )
        assert result["redirection"] == 0.0

    def test_nan_beta_excluded(
        self, transition_result_3state, program_loadings_tnf, conditional_gammas_ibd
    ):
        from pipelines.state_space.therapeutic_redirection import (
            compute_therapeutic_redirection_per_celltype
        )
        from models.latent_mediator import ConditionalBeta
        nan_betas = [ConditionalBeta(gene="TNF", program_id="IBD_macro_P00",
                                      cell_type="macrophage", disease="IBD",
                                      beta=float("nan"))]
        result = compute_therapeutic_redirection_per_celltype(
            gene="TNF", disease="IBD", cell_type="macrophage",
            program_loadings=program_loadings_tnf,
            conditional_betas=nan_betas,
            conditional_gammas=conditional_gammas_ibd,
            transition_result=transition_result_3state,
        )
        assert result["redirection"] == 0.0
        assert result["n_programs"] == 0

    def test_pooled_fraction_computed(
        self, transition_result_3state, program_loadings_tnf,
        conditional_betas_tnf, conditional_gammas_ibd
    ):
        from pipelines.state_space.therapeutic_redirection import (
            compute_therapeutic_redirection_per_celltype
        )
        from models.latent_mediator import ConditionalBeta
        pooled_betas = [ConditionalBeta(gene="TNF", program_id="IBD_macro_P00",
                                         cell_type="macrophage", disease="IBD",
                                         beta=0.7, pooled_fallback=True)]
        result = compute_therapeutic_redirection_per_celltype(
            gene="TNF", disease="IBD", cell_type="macrophage",
            program_loadings=program_loadings_tnf,
            conditional_betas=pooled_betas,
            conditional_gammas=conditional_gammas_ibd,
            transition_result=transition_result_3state,
        )
        assert result["pooled_fraction"] == 1.0


# ---------------------------------------------------------------------------
# compute_therapeutic_redirection (multi-celltype summation)
# ---------------------------------------------------------------------------

class TestComputeTherapeuticRedirection:
    def test_equal_weight_average(self):
        from pipelines.state_space.therapeutic_redirection import compute_therapeutic_redirection
        data = {
            "macrophage": {"redirection": 0.4, "n_programs": 1,
                           "pooled_fraction": 0.0, "evidence_tiers": ["T2"], "provenance": ["p"]},
            "enterocyte": {"redirection": 0.2, "n_programs": 1,
                           "pooled_fraction": 0.0, "evidence_tiers": ["T3"], "provenance": ["q"]},
        }
        result = compute_therapeutic_redirection("TNF", "IBD", data)
        # Equal weights → (0.4 + 0.2) / 2 = 0.3
        assert abs(result.therapeutic_redirection - 0.3) < 1e-9

    def test_custom_weights(self):
        from pipelines.state_space.therapeutic_redirection import compute_therapeutic_redirection
        data = {
            "macrophage": {"redirection": 1.0, "n_programs": 1, "pooled_fraction": 0.0,
                           "evidence_tiers": [], "provenance": []},
            "enterocyte": {"redirection": 0.0, "n_programs": 0, "pooled_fraction": 0.0,
                           "evidence_tiers": [], "provenance": []},
        }
        result = compute_therapeutic_redirection("TNF", "IBD", data,
                                                  cell_type_weights={"macrophage": 2.0, "enterocyte": 1.0})
        # macrophage weight = 2/3, enterocyte = 1/3 → 1.0×(2/3) + 0.0×(1/3) ≈ 0.667
        assert abs(result.therapeutic_redirection - 2/3) < 1e-9

    def test_context_confidence_warning_above_threshold(self):
        from pipelines.state_space.therapeutic_redirection import compute_therapeutic_redirection
        data = {
            "macrophage": {"redirection": 0.5, "n_programs": 2,
                           "pooled_fraction": 0.8, "evidence_tiers": [], "provenance": []},
        }
        result = compute_therapeutic_redirection("TNF", "IBD", data)
        assert result.context_confidence_warning is True

    def test_no_warning_below_threshold(self):
        from pipelines.state_space.therapeutic_redirection import compute_therapeutic_redirection
        data = {
            "macrophage": {"redirection": 0.5, "n_programs": 2,
                           "pooled_fraction": 0.3, "evidence_tiers": [], "provenance": []},
        }
        result = compute_therapeutic_redirection("TNF", "IBD", data)
        assert result.context_confidence_warning is False

    def test_returns_therapeutic_redirection_result(self):
        from pipelines.state_space.therapeutic_redirection import compute_therapeutic_redirection
        from models.latent_mediator import TherapeuticRedirectionResult
        data = {"m": {"redirection": 0.0, "n_programs": 0, "pooled_fraction": 0.0,
                      "evidence_tiers": [], "provenance": []}}
        result = compute_therapeutic_redirection("X", "IBD", data)
        assert isinstance(result, TherapeuticRedirectionResult)
        assert result.gene == "X"

    def test_n_cell_types_contributing(self):
        from pipelines.state_space.therapeutic_redirection import compute_therapeutic_redirection
        data = {
            "m": {"redirection": 0.5, "n_programs": 1, "pooled_fraction": 0.0,
                  "evidence_tiers": [], "provenance": []},
            "e": {"redirection": 0.0, "n_programs": 0, "pooled_fraction": 0.0,
                  "evidence_tiers": [], "provenance": []},
        }
        result = compute_therapeutic_redirection("TNF", "IBD", data)
        assert result.n_cell_types_contributing == 1


# ---------------------------------------------------------------------------
# trajectory_scoring deprecation
# ---------------------------------------------------------------------------

class TestTrajectoryScoreDeprecated:
    def test_score_gene_warns(self):
        from pipelines.state_space.trajectory_scoring import score_gene
        import pytest, warnings
        # Minimal valid inputs (empty state_result → score returns zero-ish)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            score_gene("TNF", {}, {"coarse": [], "intermediate": []}, disease="IBD")
        assert any(issubclass(warning.category, DeprecationWarning) for warning in w)

    def test_score_all_genes_warns(self):
        from pipelines.state_space.trajectory_scoring import score_all_genes
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            score_all_genes(["TNF"], {}, {"intermediate": []}, disease="IBD")
        assert any(issubclass(warning.category, DeprecationWarning) for warning in w)
