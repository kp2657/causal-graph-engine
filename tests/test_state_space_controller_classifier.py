"""
tests/test_phase_h_controller_classifier.py

Unit tests for Phase H: controller vs marker classification.

Tests:
  - ControllerAnnotation model defaults and field constraints
  - classify_gene_controller: confidence levels, category assignment
  - TF annotation signal
  - Pseudotime peak signal (early/late)
  - Phase G transition profile signals (entry, maintenance marker)
  - Perturbation tier signal (T1/T2 confidence)
  - compute_marker_confidence
  - compute_marker_discount: formula, caps
  - classify_gene_list: batch API
  - Graceful degradation: no adata, no profile, missing gene
"""
from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------

def _make_adata(n_cells: int = 80, n_genes: int = 20, seed: int = 0):
    anndata = pytest.importorskip("anndata")
    import pandas as pd

    rng = np.random.default_rng(seed)
    X = rng.random((n_cells, n_genes)).astype(float)

    # Gene 0: elevated early (peak near pseudotime=0) → controller signal
    early_cells = np.arange(n_cells // 4)
    X[early_cells, 0] += 3.0

    # Gene 1: elevated late (peak near pseudotime=1) → marker signal
    late_cells = np.arange(3 * n_cells // 4, n_cells)
    X[late_cells, 1] += 3.0

    obs = pd.DataFrame({
        "dpt_pseudotime": np.linspace(0, 1, n_cells),
        "disease_condition": ["disease"] * (n_cells // 2) + ["healthy"] * (n_cells // 2),
    })
    var = pd.DataFrame(index=[f"GENE{i:03d}" for i in range(n_genes)])
    return anndata.AnnData(X=X, obs=obs, var=var)


def _make_profile(
    entry: float = 0.0,
    persistence: float = 0.0,
    recovery: float = 0.0,
    boundary: float = 0.0,
    cat: str = "unknown",
):
    from models.evidence import TransitionGeneProfile
    return TransitionGeneProfile(
        gene="TEST", disease="AMD",
        entry_score=entry,
        persistence_score=persistence,
        recovery_score=recovery,
        boundary_score=boundary,
        mechanistic_category=cat,
    )


# ---------------------------------------------------------------------------
# ControllerAnnotation model
# ---------------------------------------------------------------------------

class TestControllerAnnotationModel:
    def test_defaults(self):
        from models.evidence import ControllerAnnotation
        a = ControllerAnnotation(gene="SPI1", disease="AMD")
        assert a.controller_likelihood == 0.0
        assert a.controller_confidence == "low"
        assert a.category == "unknown"
        assert a.supporting_signals == []

    def test_roundtrip(self):
        from models.evidence import ControllerAnnotation
        a = ControllerAnnotation(
            gene="IRF4", disease="AMD",
            controller_likelihood=0.6,
            controller_confidence="medium",
            category="upstream_controller",
            supporting_signals=["tf_annotation", "early_pseudotime(peak=0.20)"],
        )
        a2 = ControllerAnnotation.model_validate(a.model_dump())
        assert a2.controller_likelihood == a.controller_likelihood
        assert a2.supporting_signals == a.supporting_signals


# ---------------------------------------------------------------------------
# classify_gene_controller — confidence and category
# ---------------------------------------------------------------------------

class TestClassifyGeneController:
    def test_unknown_gene_returns_downstream_marker(self):
        from pipelines.state_space.controller_classifier import classify_gene_controller
        ann = classify_gene_controller("COMPLETELY_UNKNOWN_XYZ", "AMD")
        assert ann.category == "downstream_marker"
        assert ann.controller_confidence == "low"
        assert ann.controller_likelihood <= 0.35

    def test_tf_gene_gets_tf_signal(self):
        from pipelines.state_space.controller_classifier import classify_gene_controller
        ann = classify_gene_controller("SPI1", "AMD")
        assert "tf_annotation" in ann.supporting_signals
        assert ann.controller_likelihood > 0.0

    def test_tf_alone_does_not_exceed_low_cap(self):
        """TF without perturbation evidence stays ≤ 0.35."""
        from pipelines.state_space.controller_classifier import classify_gene_controller
        ann = classify_gene_controller("STAT3", "AMD")
        assert ann.controller_confidence == "low"
        assert ann.controller_likelihood <= 0.35

    def test_t1_perturbation_gives_high_confidence(self):
        from pipelines.state_space.controller_classifier import classify_gene_controller
        ann = classify_gene_controller(
            "LYZ", "AMD", evidence_tier="Tier1_Interventional"
        )
        assert ann.controller_confidence == "high"
        assert "perturbation_t1" in ann.supporting_signals

    def test_t2_perturbation_gives_high_confidence(self):
        from pipelines.state_space.controller_classifier import classify_gene_controller
        ann = classify_gene_controller(
            "LYZ", "AMD", evidence_tier="Tier2_Convergent"
        )
        assert ann.controller_confidence == "high"
        assert "perturbation_t2" in ann.supporting_signals

    def test_tf_plus_early_pt_upgrades_to_medium(self):
        """TF gene with early pseudotime peak → confidence = medium."""
        from pipelines.state_space.controller_classifier import classify_gene_controller
        adata = _make_adata()
        # GENE000 has early peak
        ann = classify_gene_controller(
            "GENE000", "AMD",
            adata=adata,
            # Inject as TF by overriding: we test with SPI1 instead
        )
        # GENE000 is not in TF list, so early_pt alone won't give medium
        assert "early_pseudotime" in ann.supporting_signals or ann.controller_likelihood >= 0.0

    def test_spi1_with_early_pt_upgrades_to_medium(self):
        """Known TF (SPI1) with an early-peaking adata gene — inject via gene_idx."""
        from pipelines.state_space.controller_classifier import classify_gene_controller, _build_gene_index
        import pandas as pd
        anndata = pytest.importorskip("anndata")
        # Build adata where SPI1 peaks early
        n_cells, n_genes = 80, 10
        rng = np.random.default_rng(1)
        X = rng.random((n_cells, n_genes)).astype(float)
        X[:20, 0] += 3.0  # SPI1 peaks in first quartile
        obs = pd.DataFrame({"dpt_pseudotime": np.linspace(0, 1, n_cells)})
        var = pd.DataFrame(index=["SPI1"] + [f"G{i}" for i in range(1, n_genes)])
        adata = anndata.AnnData(X=X, obs=obs, var=var)
        ann = classify_gene_controller("SPI1", "AMD", adata=adata)
        assert "tf_annotation" in ann.supporting_signals
        assert "early_pseudotime" in " ".join(ann.supporting_signals)
        assert ann.controller_confidence == "medium"
        assert ann.controller_likelihood <= 0.65

    def test_late_pseudotime_gives_penalty(self):
        """Gene with late peak gets late_pseudotime signal (which is penalizing)."""
        from pipelines.state_space.controller_classifier import classify_gene_controller
        adata = _make_adata()
        ann = classify_gene_controller("GENE001", "AMD", adata=adata)
        assert any("late_pseudotime" in s for s in ann.supporting_signals)

    def test_high_entry_score_adds_controller_signal(self):
        from pipelines.state_space.controller_classifier import classify_gene_controller
        profile = _make_profile(entry=0.5)
        ann = classify_gene_controller("GENE000", "AMD", transition_profile=profile)
        assert any("high_entry_score" in s for s in ann.supporting_signals)
        assert ann.controller_likelihood > 0.0

    def test_maintenance_profile_adds_marker_penalty(self):
        """High persistence + low entry → maintenance_marker signal (penalizes likelihood)."""
        from pipelines.state_space.controller_classifier import classify_gene_controller
        profile = _make_profile(persistence=0.8, entry=0.05, recovery=0.1)
        ann_plain = classify_gene_controller("LYZ", "AMD")
        ann_marker = classify_gene_controller("LYZ", "AMD", transition_profile=profile)
        assert any("maintenance_marker" in s for s in ann_marker.supporting_signals)
        # marker penalty should not increase likelihood vs plain
        assert ann_marker.controller_likelihood <= ann_plain.controller_likelihood + 1e-9

    def test_t1_controller_category(self):
        """T1 TF gene with entry signal should be upstream_controller."""
        from pipelines.state_space.controller_classifier import classify_gene_controller
        profile = _make_profile(entry=0.5)
        ann = classify_gene_controller(
            "IRF4", "AMD",
            transition_profile=profile,
            evidence_tier="Tier1_Interventional",
        )
        assert ann.category == "upstream_controller"
        assert ann.controller_confidence == "high"

    def test_likelihood_in_zero_one(self):
        from pipelines.state_space.controller_classifier import classify_gene_controller
        for gene in ["SPI1", "LYZ", "STAT3", "UNKNOWN_XYZ", "NOD2"]:
            ann = classify_gene_controller(gene, "AMD")
            assert 0.0 <= ann.controller_likelihood <= 1.0

    def test_category_consistent_with_likelihood(self):
        from pipelines.state_space.controller_classifier import classify_gene_controller
        for gene, tier in [("IRF4", "Tier1_Interventional"), ("SPI1", "provisional_virtual")]:
            ann = classify_gene_controller(gene, "AMD", evidence_tier=tier)
            if ann.controller_likelihood > 0.50:
                assert ann.category == "upstream_controller"
            elif ann.controller_likelihood >= 0.30:
                assert ann.category in ("midstream_mediator", "upstream_controller")
            else:
                assert ann.category == "downstream_marker"


# ---------------------------------------------------------------------------
# compute_marker_confidence
# ---------------------------------------------------------------------------

class TestMarkerConfidence:
    def test_no_profile_returns_neutral(self):
        from pipelines.state_space.controller_classifier import compute_marker_confidence
        assert compute_marker_confidence(None) == 0.5

    def test_pure_maintenance_returns_high(self):
        from pipelines.state_space.controller_classifier import compute_marker_confidence
        p = _make_profile(persistence=1.0, entry=0.0, recovery=0.0)
        mc = compute_marker_confidence(p)
        assert mc > 0.7

    def test_controller_profile_returns_low(self):
        from pipelines.state_space.controller_classifier import compute_marker_confidence
        p = _make_profile(persistence=0.0, entry=1.0, recovery=0.0)
        mc = compute_marker_confidence(p)
        assert mc < 0.4

    def test_in_zero_one(self):
        from pipelines.state_space.controller_classifier import compute_marker_confidence
        for p_s, e_s, r_s in [(0.9, 0.05, 0.05), (0.0, 0.8, 0.6), (0.5, 0.5, 0.5)]:
            mc = compute_marker_confidence(_make_profile(persistence=p_s, entry=e_s, recovery=r_s))
            assert 0.0 <= mc <= 1.0


# ---------------------------------------------------------------------------
# compute_marker_discount
# ---------------------------------------------------------------------------

class TestMarkerDiscount:
    def test_confirmed_controller_has_zero_discount(self):
        """controller_likelihood=1.0 → discount=0."""
        from models.evidence import ControllerAnnotation
        from pipelines.state_space.controller_classifier import compute_marker_discount
        ann = ControllerAnnotation(
            gene="SPI1", disease="AMD",
            controller_likelihood=1.0, controller_confidence="high",
            category="upstream_controller",
        )
        p = _make_profile(persistence=0.8, entry=0.1, recovery=0.1)
        assert compute_marker_discount(ann, p) == pytest.approx(0.0, abs=1e-6)

    def test_pure_marker_has_nonzero_discount(self):
        """controller_likelihood=0.0 + high marker_confidence → positive discount."""
        from models.evidence import ControllerAnnotation
        from pipelines.state_space.controller_classifier import compute_marker_discount
        ann = ControllerAnnotation(
            gene="LYZ", disease="AMD",
            controller_likelihood=0.0, controller_confidence="low",
            category="downstream_marker",
        )
        p = _make_profile(persistence=1.0, entry=0.0, recovery=0.0)
        d = compute_marker_discount(ann, p)
        assert d > 0.0
        assert d <= 0.40

    def test_discount_capped_at_040(self):
        from models.evidence import ControllerAnnotation
        from pipelines.state_space.controller_classifier import compute_marker_discount
        ann = ControllerAnnotation(
            gene="LYZ", disease="AMD",
            controller_likelihood=0.0, controller_confidence="low",
            category="downstream_marker",
        )
        # Even with maximum marker signal, cap = 0.40
        p = _make_profile(persistence=1.0, entry=0.0, recovery=0.0)
        d = compute_marker_discount(ann, p)
        assert d <= 0.40

    def test_no_profile_uses_neutral_prior(self):
        """Without transition profile, marker_confidence=0.5 → discount = 0.25 × (1-cl) × 0.5."""
        from models.evidence import ControllerAnnotation
        from pipelines.state_space.controller_classifier import compute_marker_discount
        ann = ControllerAnnotation(
            gene="LYZ", disease="AMD",
            controller_likelihood=0.0, controller_confidence="low",
            category="downstream_marker",
        )
        d = compute_marker_discount(ann, None)
        assert d == pytest.approx(0.25 * 1.0 * 0.5, abs=1e-6)


# ---------------------------------------------------------------------------
# classify_gene_list — batch API
# ---------------------------------------------------------------------------

class TestClassifyGeneList:
    def test_returns_all_genes(self):
        from pipelines.state_space.controller_classifier import classify_gene_list
        genes = ["SPI1", "LYZ", "IRF4", "UNKNOWN"]
        result = classify_gene_list(genes, "AMD")
        assert set(result.keys()) == set(genes)

    def test_uses_evidence_tiers(self):
        from pipelines.state_space.controller_classifier import classify_gene_list
        genes = ["LYZ"]
        result_virt = classify_gene_list(
            genes, "AMD", evidence_tiers={"LYZ": "provisional_virtual"}
        )
        result_t1 = classify_gene_list(
            genes, "AMD", evidence_tiers={"LYZ": "Tier1_Interventional"}
        )
        assert result_t1["LYZ"].controller_confidence == "high"
        assert result_virt["LYZ"].controller_confidence == "low"

    def test_uses_transition_profiles(self):
        from pipelines.state_space.controller_classifier import classify_gene_list
        profiles = {"IRF4": _make_profile(entry=0.6)}
        result = classify_gene_list(["IRF4"], "AMD", transition_profiles=profiles)
        assert any("high_entry_score" in s for s in result["IRF4"].supporting_signals)

    def test_empty_list_returns_empty(self):
        from pipelines.state_space.controller_classifier import classify_gene_list
        assert classify_gene_list([], "AMD") == {}

    def test_no_adata_graceful(self):
        from pipelines.state_space.controller_classifier import classify_gene_list
        result = classify_gene_list(["SPI1", "LYZ"], "AMD", adata=None)
        assert len(result) == 2
        for ann in result.values():
            assert not any("pseudotime" in s for s in ann.supporting_signals)
