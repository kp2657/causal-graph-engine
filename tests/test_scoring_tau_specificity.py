"""
tests/test_phase_r_tau_specificity.py — Unit tests for disease-state τ specificity.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from pipelines.state_space.tau_specificity import (
    compute_disease_tau,
    classify_specificity,
    TauResult,
    _tau_from_means,
    _MIN_MEAN_EXPR,
)
from models.evidence import TauSpecificityResult


# ---------------------------------------------------------------------------
# Helpers — build minimal AnnData-like objects without anndata dependency
# ---------------------------------------------------------------------------

class _FakeAnnData:
    """Minimal AnnData substitute for testing without the full library."""
    def __init__(self, X, var_names, obs):
        self.X = X
        self.var_names = var_names

        class _Obs:
            def __init__(self, d):
                self._d = d
                self.columns = list(d.keys())
            def __getitem__(self, key):
                return self._d[key]
            def __contains__(self, key):
                return key in self._d

        self.obs = _Obs(obs)
        self.n_obs, self.n_vars = X.shape


def _make_adata(n_cells_per_group: dict[str, int], gene_means: dict[str, list[float]],
                genes: list[str], noise: float = 0.1) -> _FakeAnnData:
    """
    Build a fake AnnData where each group has `n_cells` cells.

    gene_means: {gene: [mean_in_group0, mean_in_group1, ...]}
    groups ordered by sorted(n_cells_per_group.keys())
    """
    rng = np.random.default_rng(42)
    group_names = sorted(n_cells_per_group.keys())
    all_cells_X = []
    all_labels = []
    for grp in group_names:
        n = n_cells_per_group[grp]
        for g in genes:
            pass  # just building rows
        row_means = np.array([gene_means.get(g, [0.0] * len(group_names))[group_names.index(grp)]
                               for g in genes])
        # Generate n cells with Poisson-like variation around mean
        cells = rng.poisson(lam=np.maximum(row_means, 0) * 10, size=(n, len(genes))) / 10.0
        cells = cells + rng.normal(0, noise, size=cells.shape)
        cells = np.clip(cells, 0, None)
        all_cells_X.append(cells)
        all_labels.extend([grp] * n)
    X = np.vstack(all_cells_X).astype(np.float32)
    obs = {"disease": np.array(all_labels)}
    return _FakeAnnData(X, genes, obs)


# ---------------------------------------------------------------------------
# _tau_from_means
# ---------------------------------------------------------------------------

class TestTauFromMeans:
    def test_single_group_returns_zero(self):
        assert _tau_from_means(np.array([1.0])) == 0.0

    def test_perfectly_specific(self):
        # [1, 0, 0] — all expression in group 0
        tau = _tau_from_means(np.array([1.0, 0.0, 0.0]))
        assert abs(tau - 1.0) < 1e-9

    def test_perfectly_ubiquitous(self):
        # [1, 1, 1] — equal in all groups
        tau = _tau_from_means(np.array([1.0, 1.0, 1.0]))
        assert abs(tau) < 1e-9

    def test_partial_specificity(self):
        tau = _tau_from_means(np.array([1.0, 0.5, 0.0]))
        assert 0.0 < tau < 1.0

    def test_all_zero_returns_zero(self):
        assert _tau_from_means(np.array([0.0, 0.0, 0.0])) == 0.0

    def test_two_groups_one_expressed(self):
        tau = _tau_from_means(np.array([1.0, 0.0]))
        assert abs(tau - 1.0) < 1e-9

    def test_two_groups_equal(self):
        tau = _tau_from_means(np.array([0.5, 0.5]))
        assert abs(tau) < 1e-9


# ---------------------------------------------------------------------------
# classify_specificity
# ---------------------------------------------------------------------------

class TestClassifySpecificity:
    def test_disease_specific(self):
        assert classify_specificity(0.75, 2.0) == "disease_specific"

    def test_normal_specific(self):
        assert classify_specificity(0.70, -1.5) == "normal_specific"

    def test_ubiquitous(self):
        assert classify_specificity(0.15, 0.0) == "ubiquitous"

    def test_moderately_specific_mid_tau(self):
        assert classify_specificity(0.45, 0.5) == "moderately_specific"

    def test_high_tau_no_enrichment(self):
        # τ ≥ 0.6 but |log2fc| < 1
        assert classify_specificity(0.65, 0.5) == "moderately_specific"

    def test_boundary_tau(self):
        # Exactly at 0.30 boundary
        assert classify_specificity(0.30, 0.0) == "moderately_specific"

    def test_boundary_high_tau(self):
        # Exactly at 0.60 boundary with disease enrichment
        assert classify_specificity(0.60, 1.5) == "disease_specific"


# ---------------------------------------------------------------------------
# compute_disease_tau — basic correctness
# ---------------------------------------------------------------------------

class TestComputeDiseaseTau:
    def _make_simple_adata(self):
        """Three genes: disease-specific, normal-specific, ubiquitous.

        Groups sorted alphabetically: ["disease", "normal"]
        So means list is [disease_mean, normal_mean].
        """
        genes = ["DISEASE_GENE", "NORMAL_GENE", "UBIQ_GENE"]
        means = {
            "DISEASE_GENE": [2.0, 0.0],   # [disease_mean, normal_mean]
            "NORMAL_GENE":  [0.0, 2.0],   # [disease_mean, normal_mean]
            "UBIQ_GENE":    [1.0, 1.0],
        }
        return _make_adata(
            n_cells_per_group={"normal": 200, "disease": 200},
            gene_means=means,
            genes=genes,
            noise=0.05,
        )

    def test_returns_all_genes(self):
        adata = self._make_simple_adata()
        results = compute_disease_tau(adata, ["DISEASE_GENE", "NORMAL_GENE", "UBIQ_GENE"])
        assert set(results.keys()) == {"DISEASE_GENE", "NORMAL_GENE", "UBIQ_GENE"}

    def test_disease_specific_has_high_tau(self):
        adata = self._make_simple_adata()
        results = compute_disease_tau(adata)
        tau_disease = results["DISEASE_GENE"].tau_disease
        tau_ubiq = results["UBIQ_GENE"].tau_disease
        assert tau_disease > tau_ubiq, f"disease_gene τ={tau_disease} should > ubiq τ={tau_ubiq}"

    def test_ubiquitous_has_low_tau(self):
        adata = self._make_simple_adata()
        results = compute_disease_tau(adata)
        tau_ubiq = results["UBIQ_GENE"].tau_disease
        assert tau_ubiq < 0.2, f"ubiquitous gene should have τ < 0.2, got {tau_ubiq}"

    def test_disease_gene_positive_log2fc(self):
        adata = self._make_simple_adata()
        results = compute_disease_tau(adata)
        assert results["DISEASE_GENE"].disease_log2fc > 0

    def test_normal_gene_negative_log2fc(self):
        adata = self._make_simple_adata()
        results = compute_disease_tau(adata)
        assert results["NORMAL_GENE"].disease_log2fc < 0

    def test_disease_specific_classification(self):
        adata = self._make_simple_adata()
        results = compute_disease_tau(adata)
        cls = results["DISEASE_GENE"].specificity_class
        assert cls in ("disease_specific", "moderately_specific"), f"got {cls}"

    def test_none_gene_list_scores_all(self):
        adata = self._make_simple_adata()
        results = compute_disease_tau(adata, gene_list=None)
        assert len(results) == 3

    def test_missing_gene_omitted(self):
        adata = self._make_simple_adata()
        results = compute_disease_tau(adata, gene_list=["DISEASE_GENE", "NONEXISTENT"])
        assert "DISEASE_GENE" in results
        assert "NONEXISTENT" not in results

    def test_no_disease_col_returns_neutral(self):
        genes = ["G1", "G2"]
        X = np.ones((10, 2), dtype=np.float32)
        obs = {"cell_type": np.array(["mac"] * 10)}
        adata = _FakeAnnData(X, genes, obs)
        results = compute_disease_tau(adata, genes, disease_col="disease")
        for g in genes:
            assert results[g].tau_disease == 0.5
            assert results[g].specificity_class == "unknown"

    def test_single_group_returns_neutral(self):
        genes = ["G1"]
        X = np.ones((20, 1), dtype=np.float32)
        obs = {"disease": np.array(["normal"] * 20)}
        adata = _FakeAnnData(X, genes, obs)
        results = compute_disease_tau(adata, genes)
        assert results["G1"].tau_disease == 0.5

    def test_n_groups_field(self):
        adata = self._make_simple_adata()
        results = compute_disease_tau(adata)
        for r in results.values():
            assert r.n_groups == 2

    def test_group_means_populated(self):
        adata = self._make_simple_adata()
        results = compute_disease_tau(adata)
        gm = results["DISEASE_GENE"].group_means
        assert "normal" in gm
        assert "disease" in gm
        assert gm["disease"] > gm["normal"]

    def test_lowly_expressed_class(self):
        genes = ["SILENT"]
        X = np.zeros((100, 1), dtype=np.float32) + 0.001  # near zero
        obs = {"disease": np.array(["normal"] * 50 + ["ibd"] * 50)}
        adata = _FakeAnnData(X, genes, obs)
        results = compute_disease_tau(adata, genes)
        assert results["SILENT"].specificity_class == "lowly_expressed"

    def test_pct_disease_and_pct_normal(self):
        adata = self._make_simple_adata()
        results = compute_disease_tau(adata)
        r = results["DISEASE_GENE"]
        assert 0.0 <= r.pct_disease <= 1.0
        assert 0.0 <= r.pct_normal <= 1.0

    def test_multi_disease_groups(self):
        """τ with 3 groups: normal, disease, subtype."""
        genes = ["G1"]
        X = np.array(
            [[0.0]] * 100 +   # normal
            [[2.0]] * 100 +   # disease
            [[1.8]] * 100,    # geographic_atrophy
            dtype=np.float32
        )
        obs = {"disease": np.array(["normal"] * 100 + ["AD"] * 100 + ["late-onset alzheimer's disease"] * 100)}
        adata = _FakeAnnData(X, genes, obs)
        results = compute_disease_tau(adata, genes)
        r = results["G1"]
        assert r.n_groups == 3
        # G1 is mostly silent in normal, high in disease groups → should be moderately/disease specific
        assert r.tau_disease > 0.3


# ---------------------------------------------------------------------------
# TauSpecificityResult model
# ---------------------------------------------------------------------------

class TestTauSpecificityResult:
    def test_default_neutral(self):
        r = TauSpecificityResult(gene="NOD2")
        assert r.tau_disease == 0.5
        assert r.specificity_class == "unknown"

    def test_full_construction(self):
        r = TauSpecificityResult(
            gene="STAT1",
            disease="AD",
            tau_disease=0.78,
            disease_log2fc=1.5,
            pct_disease=0.65,
            pct_normal=0.12,
            mean_disease=0.8,
            mean_normal=0.1,
            n_groups=3,
            specificity_class="disease_specific",
            group_means={"normal": 0.1, "AD": 0.8, "late-onset alzheimer's disease": 0.75},
        )
        assert r.tau_disease == 0.78
        assert r.specificity_class == "disease_specific"

    def test_model_dump_round_trip(self):
        r = TauSpecificityResult(gene="NOD2", tau_disease=0.65, disease_log2fc=1.2)
        d = r.model_dump()
        r2 = TauSpecificityResult.model_validate(d)
        assert r2.tau_disease == r.tau_disease


# ---------------------------------------------------------------------------
# LatentMediatorState.assessment with tau_disease_specificity
# ---------------------------------------------------------------------------

class TestFinalScoreWithTau:
    def _make_tr(self, tau=0.5, **kwargs) -> "TherapeuticRedirectionResult":
        from models.latent_mediator import TherapeuticRedirectionResult
        defaults = dict(
            gene="TEST",
            disease="AD",
            therapeutic_redirection=0.3,
            durability_score=0.5,
            escape_risk=0.0,
            failure_risk=0.0,
            ot_combined=0.0,
            trial_bonus=0.0,
            safety_penalty=0.0,
            state_influence_score=0.0,
            directionality=1,
            genetic_grounding=0.4,
            tau_disease_specificity=tau,
        )
        defaults.update(kwargs)
        return TherapeuticRedirectionResult(**defaults)

    def test_neutral_tau_no_change(self):
        """τ=0.5 → moderate specificity; τ=0.4 → same threshold."""
        tr_neutral = self._make_tr(tau=0.5)
        tr_at_40   = self._make_tr(tau=0.4)
        a_neutral = tr_neutral.assessment["disease_specificity"]
        a_40      = tr_at_40.assessment["disease_specificity"]
        assert a_neutral["tau"] == pytest.approx(0.5)
        assert a_40["tau"] == pytest.approx(0.4)

    def test_disease_specific_higher_score(self):
        tr_specific   = self._make_tr(tau=0.9)
        tr_ubiquitous = self._make_tr(tau=0.1)
        assert tr_specific.assessment["disease_specificity"]["tau"] > \
               tr_ubiquitous.assessment["disease_specificity"]["tau"]
        assert tr_specific.assessment["disease_specificity"]["verdict"] == "disease-specific"
        assert tr_ubiquitous.assessment["disease_specificity"]["verdict"] == "broadly expressed"

    def test_tau_bonus_upper_bound(self):
        """τ=1.0 → disease-specific verdict."""
        tr_high = self._make_tr(tau=1.0)
        assert tr_high.assessment["disease_specificity"]["verdict"] == "disease-specific"

    def test_tau_bonus_lower_bound(self):
        """τ=0.0 → broadly expressed verdict."""
        tr_low = self._make_tr(tau=0.0)
        assert tr_low.assessment["disease_specificity"]["verdict"] == "broadly expressed"

    def test_t_mod_still_bounded(self):
        """Assessment dict is well-formed for extreme inputs."""
        tr = self._make_tr(tau=1.0, ot_combined=1.0, trial_bonus=1.0, safety_penalty=0.0)
        a = tr.assessment
        assert "causal" in a and "mechanistic" in a and "safety" in a

        tr2 = self._make_tr(tau=0.0, ot_combined=0.0, trial_bonus=0.0, safety_penalty=2.0)
        a2 = tr2.assessment
        assert a2["safety"]["verdict"] == "high risk"

    def test_default_tau_is_neutral(self):
        """TherapeuticRedirectionResult default tau=0.5 → moderate specificity."""
        from models.latent_mediator import TherapeuticRedirectionResult
        tr = TherapeuticRedirectionResult(
            gene="X", disease="AD",
            therapeutic_redirection=0.2,
            durability_score=0.4,
            escape_risk=0.0, failure_risk=0.0,
            ot_combined=0.0, trial_bonus=0.0, safety_penalty=0.0,
            genetic_grounding=0.3,
        )
        assert tr.tau_disease_specificity == 0.5
        a = tr.assessment
        assert a["causal"]["ota_gamma"] == pytest.approx(0.3)
        assert a["disease_specificity"]["verdict"] == "moderate specificity"
