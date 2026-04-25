"""
tests/test_phase_i_disagreement_profile.py

Unit tests for Phase I: structured DisagreementProfile.

Tests:
  - DisagreementProfile model defaults and roundtrip
  - 5 dimension scores: genetics, expression, perturbation, cell_type, cross_context
  - 6 label assignments with strict rules:
      discordant, context_dependent, likely_upstream_controller,
      likely_marker, likely_non_transportable, supported, unknown
  - build_disagreement_profile: end-to-end with various input combos
  - Graceful degradation: no betas, no gamma, no profile
"""
from __future__ import annotations

import math
import pytest


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------

def _cb(
    beta: float,
    tier: str = "provisional_virtual",
    context_verified: bool = False,
    cell_type: str = "macrophage",
    program_id: str = "P0",
    gene: str = "TEST",
    disease: str = "AMD",
):
    from models.latent_mediator import ConditionalBeta
    return ConditionalBeta(
        gene=gene, program_id=program_id, cell_type=cell_type,
        disease=disease, beta=beta, evidence_tier=tier,
        context_verified=context_verified,
        pooled_fallback=not context_verified,
        data_source="test",
    )


def _profile(
    entry: float = 0.0,
    persistence: float = 0.0,
    recovery: float = 0.0,
    boundary: float = 0.0,
    gene: str = "TEST",
    disease: str = "AMD",
):
    from models.evidence import TransitionGeneProfile
    return TransitionGeneProfile(
        gene=gene, disease=disease,
        entry_score=entry, persistence_score=persistence,
        recovery_score=recovery, boundary_score=boundary,
    )


def _ctrl_ann(
    cl: float = 0.0,
    confidence: str = "low",
    category: str = "downstream_marker",
    signals: list | None = None,
    gene: str = "TEST",
    disease: str = "AMD",
):
    from models.evidence import ControllerAnnotation
    return ControllerAnnotation(
        gene=gene, disease=disease,
        controller_likelihood=cl,
        controller_confidence=confidence,
        category=category,
        supporting_signals=signals or [],
    )


# ---------------------------------------------------------------------------
# DisagreementProfile model
# ---------------------------------------------------------------------------

class TestDisagreementProfileModel:
    def test_defaults(self):
        from models.evidence import DisagreementProfile
        p = DisagreementProfile(gene="SPI1", disease="AMD")
        assert p.genetics_support == 0.0
        assert p.mechanistic_label == "unknown"
        assert p.label_confidence == 0.0
        assert p.supporting_evidence == []

    def test_roundtrip(self):
        from models.evidence import DisagreementProfile
        p = DisagreementProfile(
            gene="IRF4", disease="AMD",
            genetics_support=0.8, expression_coupling=0.7,
            perturbation_support=0.6, cell_type_specificity=0.5,
            cross_context_consistency=0.9,
            mechanistic_label="likely_upstream_controller",
            label_confidence=0.75,
            supporting_evidence=["gamma=0.5", "tier=Tier1_Interventional"],
        )
        p2 = DisagreementProfile.model_validate(p.model_dump())
        assert p2.mechanistic_label == p.mechanistic_label
        assert p2.genetics_support == p.genetics_support


# ---------------------------------------------------------------------------
# Dimension functions
# ---------------------------------------------------------------------------

class TestDimensions:
    def test_genetics_t1_high_gamma(self):
        from pipelines.evidence_disagreement import _dim_genetics
        score, sup, con = _dim_genetics(0.6, "Tier1_Interventional")
        assert score > 0.8
        assert not con  # no contradicting evidence

    def test_genetics_virtual_no_gamma(self):
        from pipelines.evidence_disagreement import _dim_genetics
        score, sup, con = _dim_genetics(None, "provisional_virtual")
        assert score < 0.2
        assert "virtual_tier" in con

    def test_expression_with_profile(self):
        from pipelines.evidence_disagreement import _dim_expression
        p = _profile(persistence=0.8)
        score, sup, con = _dim_expression(p, das=0.0)
        assert score >= 0.8
        assert any("persistence" in s for s in sup)

    def test_expression_das_fallback(self):
        from pipelines.evidence_disagreement import _dim_expression
        score, sup, con = _dim_expression(None, das=0.6)
        assert abs(score - 0.6) < 1e-6

    def test_perturbation_t1_consistent(self):
        from pipelines.evidence_disagreement import _dim_perturbation
        betas = [
            _cb(0.4, "Tier1_Interventional"),
            _cb(0.5, "Tier1_Interventional"),
            _cb(0.3, "Tier2_Convergent"),
        ]
        score, sup, con = _dim_perturbation(betas)
        assert score > 0.5
        assert "n_t1t2=3" in sup

    def test_perturbation_no_t1t2(self):
        from pipelines.evidence_disagreement import _dim_perturbation
        betas = [_cb(0.4, "provisional_virtual")]
        score, sup, con = _dim_perturbation(betas)
        assert score < 0.1
        assert "no_t1t2_perturbation" in con

    def test_cross_context_clean(self):
        from pipelines.evidence_disagreement import _dim_cross_context
        score, sup, con = _dim_cross_context([])
        assert score == 1.0
        assert "no_disagreement" in sup

    def test_cross_context_penalised_by_block(self):
        from pipelines.evidence_disagreement import _dim_cross_context
        from models.latent_mediator import EvidenceDisagreementRecord
        rec = EvidenceDisagreementRecord(
            gene="X", disease="AMD", rule="cross_context_sign_flip",
            value_a=0.3, value_b=-0.3, severity="block", explanation="flip"
        )
        score, sup, con = _dim_cross_context([rec])
        assert score == pytest.approx(0.5, abs=1e-6)
        assert "block:cross_context_sign_flip" in con

    def test_cell_type_neutral_when_insufficient(self):
        from pipelines.evidence_disagreement import _dim_cell_type_specificity
        score, sup, con = _dim_cell_type_specificity([])
        assert score == 0.5
        assert any("neutral_prior" in s for s in sup)


# ---------------------------------------------------------------------------
# build_disagreement_profile — labels
# ---------------------------------------------------------------------------

class TestDisagreementLabels:
    def test_discordant_genetics_vs_perturbation(self):
        """sign(gamma) ≠ sign(mean T1 beta) → discordant."""
        from pipelines.evidence_disagreement import build_disagreement_profile
        betas = [_cb(0.4, "Tier1_Interventional"), _cb(0.5, "Tier1_Interventional")]
        p = build_disagreement_profile(
            "TEST", "AMD", betas, mr_gamma=-0.4,
            evidence_tier="Tier2_Convergent"
        )
        assert p.mechanistic_label == "discordant"
        assert p.label_confidence >= 0.6

    def test_context_dependent_when_rule4_fired(self):
        """cross_context_sign_flip rule → context_dependent."""
        from pipelines.evidence_disagreement import build_disagreement_profile
        # Two context-verified betas with opposite signs
        betas = [
            _cb(0.5, "Tier1_Interventional", context_verified=True, cell_type="macrophage"),
            _cb(-0.5, "Tier1_Interventional", context_verified=True, cell_type="T_cell"),
        ]
        p = build_disagreement_profile("TEST", "AMD", betas, mr_gamma=0.3,
                                        evidence_tier="Tier1_Interventional")
        assert p.mechanistic_label == "context_dependent"
        assert p.label_confidence >= 0.8

    def test_likely_upstream_controller_from_t1(self):
        """T1/T2 perturbation evidence → likely_upstream_controller."""
        from pipelines.evidence_disagreement import build_disagreement_profile
        betas = [_cb(0.4, "Tier1_Interventional")]
        p = build_disagreement_profile(
            "IRF4", "AMD", betas, mr_gamma=0.5,
            evidence_tier="Tier1_Interventional"
        )
        assert p.mechanistic_label == "likely_upstream_controller"
        assert p.label_confidence >= 0.7

    def test_likely_upstream_controller_from_profile(self):
        """No T1/T2, but TF + early pseudotime + entry>0.2 → likely_upstream_controller."""
        from pipelines.evidence_disagreement import build_disagreement_profile
        betas = [_cb(0.2, "provisional_virtual")]
        tp = _profile(entry=0.5)
        ctrl = _ctrl_ann(
            cl=0.25, signals=["tf_annotation", "early_pseudotime(peak=0.20)"]
        )
        p = build_disagreement_profile(
            "SPI1", "AMD", betas,
            transition_profile=tp,
            controller_annotation=ctrl,
        )
        assert p.mechanistic_label == "likely_upstream_controller"
        assert p.label_confidence == pytest.approx(0.55, abs=0.05)

    def test_likely_marker(self):
        """persistence>0.5 + late pseudotime + virtual tier + low cl → likely_marker."""
        from pipelines.evidence_disagreement import build_disagreement_profile
        betas = [_cb(0.05, "provisional_virtual")]
        tp = _profile(persistence=0.8, entry=0.05)
        ctrl = _ctrl_ann(
            cl=0.0,
            signals=["maintenance_marker(persist=0.80,entry=0.05)", "late_pseudotime(peak=0.85)"]
        )
        p = build_disagreement_profile(
            "LYZ", "AMD", betas,
            transition_profile=tp,
            controller_annotation=ctrl,
        )
        assert p.mechanistic_label == "likely_marker"
        assert p.label_confidence >= 0.7

    def test_likely_non_transportable(self):
        """expression_coupling>0.5 + cross_context<0.3 → likely_non_transportable."""
        from pipelines.evidence_disagreement import build_disagreement_profile
        from models.latent_mediator import EvidenceDisagreementRecord
        # High expression coupling via profile, + block disagreement record
        betas = [_cb(0.4, "provisional_virtual")]
        tp = _profile(persistence=0.7)
        # Inject a mock block record by providing betas that trigger rule4
        # Easiest: provide context-verified flip
        betas_flip = [
            _cb(0.6, "Tier1_Interventional", context_verified=True, cell_type="macrophage"),
            _cb(-0.6, "Tier1_Interventional", context_verified=True, cell_type="T_cell"),
        ]
        # This triggers context_dependent — not the right label. For non-transportable, we
        # need expression high and cross_context low WITHOUT Rule 4.
        # Simulate by using build_disagreement_profile with a flag record only
        # (flag = -0.25, starting from 1.0 → 0.75, not enough to go below 0.3)
        # Instead: provide 3 flag records
        # We can't inject records directly, so test this via the helper function
        from pipelines.evidence_disagreement import _dim_cross_context, _assign_label
        from models.latent_mediator import EvidenceDisagreementRecord
        # Build profile scores that match non-transportable conditions
        profile_scores = {
            "genetics_support": 0.5,
            "expression_coupling": 0.6,
            "perturbation_support": 0.1,
            "cell_type_specificity": 0.5,
            "cross_context_consistency": 0.25,
        }
        label, conf = _assign_label(
            profile_scores,
            mr_gamma=0.2,  # same sign as betas below
            conditional_betas=[_cb(0.1, "provisional_virtual")],
            transition_profile=_profile(persistence=0.6),
            controller_annotation=None,
            has_rule4=False,
        )
        assert label == "likely_non_transportable"

    def test_supported(self):
        """≥4 dimensions above threshold → supported (no T1/T2 betas to avoid upstream_controller)."""
        from pipelines.evidence_disagreement import build_disagreement_profile
        # Tier3 betas: contribute to cell_type dimension but don't trigger likely_upstream_controller
        betas = [
            _cb(0.4, "Tier3_Provisional"),
            _cb(0.45, "Tier3_Provisional"),
        ]
        tp = _profile(entry=0.6, persistence=0.3, recovery=0.2)
        p = build_disagreement_profile(
            "IRF4", "AMD", betas,
            mr_gamma=0.55,
            evidence_tier="Tier3_Provisional",
            transition_profile=tp,
        )
        assert p.mechanistic_label == "supported"
        assert p.label_confidence > 0.0

    def test_unknown_when_no_signal(self):
        """Virtual tier, no betas, no profile → unknown."""
        from pipelines.evidence_disagreement import build_disagreement_profile
        p = build_disagreement_profile("UNCHARACTERISED_XYZ", "AMD", [])
        assert p.mechanistic_label == "unknown"
        assert p.label_confidence == 0.0


# ---------------------------------------------------------------------------
# build_disagreement_profile — end-to-end properties
# ---------------------------------------------------------------------------

class TestBuildDisagreementProfileE2E:
    def test_scores_in_zero_one(self):
        from pipelines.evidence_disagreement import build_disagreement_profile
        betas = [_cb(0.3, "Tier2_Convergent"), _cb(0.2, "Tier1_Interventional")]
        p = build_disagreement_profile(
            "SPI1", "AMD", betas,
            mr_gamma=0.4,
            evidence_tier="Tier2_Convergent",
            transition_profile=_profile(entry=0.4, persistence=0.2),
        )
        for field in ("genetics_support", "expression_coupling", "perturbation_support",
                      "cell_type_specificity", "cross_context_consistency"):
            v = getattr(p, field)
            assert 0.0 <= v <= 1.0, f"{field}={v} out of range"

    def test_supporting_evidence_populated(self):
        from pipelines.evidence_disagreement import build_disagreement_profile
        betas = [_cb(0.5, "Tier1_Interventional")]
        p = build_disagreement_profile("IRF4", "AMD", betas, mr_gamma=0.5,
                                        evidence_tier="Tier1_Interventional")
        assert len(p.supporting_evidence) > 0

    def test_no_betas_graceful(self):
        from pipelines.evidence_disagreement import build_disagreement_profile
        p = build_disagreement_profile("GENE_X", "AMD", [], mr_gamma=None)
        assert isinstance(p.mechanistic_label, str)
        assert 0.0 <= p.label_confidence <= 1.0

    def test_label_confidence_in_zero_one(self):
        from pipelines.evidence_disagreement import build_disagreement_profile
        for mr in [None, 0.0, 0.5, -0.3]:
            betas = [_cb(0.4, "Tier1_Interventional")] if mr is not None else []
            p = build_disagreement_profile("TEST", "AMD", betas, mr_gamma=mr)
            assert 0.0 <= p.label_confidence <= 1.0

    def test_uses_controller_annotation_dict(self):
        """controller_annotation passed as model_dump() dict (as in pipeline)."""
        from pipelines.evidence_disagreement import build_disagreement_profile
        betas = [_cb(0.2, "provisional_virtual")]
        tp = _profile(entry=0.5)
        ctrl_dict = {
            "gene": "SPI1", "disease": "AMD",
            "controller_likelihood": 0.25,
            "controller_confidence": "low",
            "category": "downstream_marker",
            "supporting_signals": ["tf_annotation", "early_pseudotime(peak=0.15)"],
        }
        p = build_disagreement_profile(
            "SPI1", "AMD", betas,
            transition_profile=tp,
            controller_annotation=ctrl_dict,
        )
        assert p.mechanistic_label == "likely_upstream_controller"
