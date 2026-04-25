"""
Tests for Phase D: pipelines/evidence_disagreement.py
"""
from __future__ import annotations

import pytest
from models.latent_mediator import ConditionalBeta, EvidenceDisagreementRecord


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _cb(gene="CFH", program_id="p", cell_type="retinal_pigment_epithelial", disease="AMD",
        beta=0.6, context_verified=True, evidence_tier="Tier1_Interventional",
        pooled_fallback=False):
    return ConditionalBeta(gene=gene, program_id=program_id, cell_type=cell_type,
                           disease=disease, beta=beta, context_verified=context_verified,
                           evidence_tier=evidence_tier, pooled_fallback=pooled_fallback)


# ---------------------------------------------------------------------------
# Rule 1: genetics_vs_perturbation
# ---------------------------------------------------------------------------

class TestGeneticsVsPerturbation:
    def test_sign_conflict_detected(self):
        from pipelines.evidence_disagreement import check_genetics_vs_perturbation
        cb = _cb(beta=0.6, evidence_tier="Tier1_Interventional")
        r = check_genetics_vs_perturbation("CFH", "AMD", -0.4, [cb])
        assert r is not None
        assert r.rule == "genetics_vs_perturbation"
        assert r.value_a == pytest.approx(-0.4)
        assert r.value_b == pytest.approx(0.6)

    def test_no_conflict_same_sign(self):
        from pipelines.evidence_disagreement import check_genetics_vs_perturbation
        cb = _cb(beta=0.6, evidence_tier="Tier1_Interventional")
        assert check_genetics_vs_perturbation("CFH", "AMD", 0.4, [cb]) is None

    def test_nan_mr_gamma_returns_none(self):
        from pipelines.evidence_disagreement import check_genetics_vs_perturbation
        cb = _cb(beta=0.6)
        assert check_genetics_vs_perturbation("CFH", "AMD", float("nan"), [cb]) is None

    def test_zero_mr_gamma_returns_none(self):
        from pipelines.evidence_disagreement import check_genetics_vs_perturbation
        assert check_genetics_vs_perturbation("CFH", "AMD", 0.0, [_cb()]) is None

    def test_no_t1t2_betas_returns_none(self):
        from pipelines.evidence_disagreement import check_genetics_vs_perturbation
        # Only Tier3 beta — not counted
        cb = _cb(beta=0.6, evidence_tier="Tier3_TrajectoryProxy")
        assert check_genetics_vs_perturbation("CFH", "AMD", -0.4, [cb]) is None

    def test_severity_flag_with_multiple_betas(self):
        from pipelines.evidence_disagreement import check_genetics_vs_perturbation
        betas = [_cb(beta=0.5), _cb(cell_type="choroidal_endothelial", beta=0.4)]
        r = check_genetics_vs_perturbation("CFH", "AMD", -0.3, betas)
        assert r.severity == "flag"

    def test_severity_warning_with_single_beta(self):
        from pipelines.evidence_disagreement import check_genetics_vs_perturbation
        r = check_genetics_vs_perturbation("CFH", "AMD", -0.3, [_cb(beta=0.5)])
        assert r.severity == "warning"

    def test_program_id_filter(self):
        from pipelines.evidence_disagreement import check_genetics_vs_perturbation
        cb_match = _cb(program_id="p1", beta=0.5)
        cb_other = _cb(program_id="p2", beta=-0.8)   # would flip mean without filter
        r = check_genetics_vs_perturbation("CFH", "AMD", -0.3,
                                            [cb_match, cb_other], program_id="p1")
        assert r is not None   # only p1 beta used; still conflict

    def test_nan_beta_excluded(self):
        from pipelines.evidence_disagreement import check_genetics_vs_perturbation
        import math
        cb_nan = _cb(beta=float("nan"))
        assert check_genetics_vs_perturbation("CFH", "AMD", -0.4, [cb_nan]) is None


# ---------------------------------------------------------------------------
# Rule 2: perturbation_vs_chemical
# ---------------------------------------------------------------------------

class TestPerturbationVsChemical:
    def test_sign_conflict(self):
        from pipelines.evidence_disagreement import check_perturbation_vs_chemical
        r = check_perturbation_vs_chemical("CFH", "AMD", 0.6, -0.4)
        assert r is not None
        assert r.rule == "perturbation_vs_chemical"
        assert r.severity == "flag"

    def test_no_conflict_same_sign(self):
        from pipelines.evidence_disagreement import check_perturbation_vs_chemical
        assert check_perturbation_vs_chemical("CFH", "AMD", 0.6, 0.4) is None

    def test_zero_betas_returns_none(self):
        from pipelines.evidence_disagreement import check_perturbation_vs_chemical
        assert check_perturbation_vs_chemical("CFH", "AMD", 0.0, 0.4) is None
        assert check_perturbation_vs_chemical("CFH", "AMD", 0.4, 0.0) is None

    def test_nan_betas_returns_none(self):
        from pipelines.evidence_disagreement import check_perturbation_vs_chemical
        assert check_perturbation_vs_chemical("CFH", "AMD", float("nan"), 0.4) is None

    def test_fields_populated(self):
        from pipelines.evidence_disagreement import check_perturbation_vs_chemical
        r = check_perturbation_vs_chemical("IL6", "CAD", -0.5, 0.3,
                                            program_id="prog1", cell_type="monocyte")
        assert r.gene == "IL6"
        assert r.cell_type_a == "monocyte"
        assert "prog1" in r.explanation


# ---------------------------------------------------------------------------
# Rule 3: bulk_vs_singlecell
# ---------------------------------------------------------------------------

class TestBulkVsSinglecell:
    def test_strong_bulk_near_zero_sc(self):
        from pipelines.evidence_disagreement import check_bulk_vs_singlecell
        cb_tiny = _cb(beta=0.02, evidence_tier="Tier1_Interventional")
        r = check_bulk_vs_singlecell("CFH", "AMD", 0.45, [cb_tiny])
        assert r is not None
        assert r.rule == "bulk_vs_singlecell"
        assert r.severity == "flag"

    def test_weak_lincs_not_triggered(self):
        from pipelines.evidence_disagreement import check_bulk_vs_singlecell
        cb_tiny = _cb(beta=0.02)
        # LINCS beta below threshold (0.30) → no flag
        assert check_bulk_vs_singlecell("CFH", "AMD", 0.20, [cb_tiny]) is None

    def test_strong_sc_not_triggered(self):
        from pipelines.evidence_disagreement import check_bulk_vs_singlecell
        # LINCS strong but sc also strong → no discordance
        cb_strong = _cb(beta=0.5, evidence_tier="Tier1_Interventional")
        assert check_bulk_vs_singlecell("CFH", "AMD", 0.45, [cb_strong]) is None

    def test_no_t1t2_sc_betas_returns_none(self):
        from pipelines.evidence_disagreement import check_bulk_vs_singlecell
        cb_t3 = _cb(beta=0.01, evidence_tier="Tier3_TrajectoryProxy")
        assert check_bulk_vs_singlecell("CFH", "AMD", 0.45, [cb_t3]) is None

    def test_values_populated(self):
        from pipelines.evidence_disagreement import check_bulk_vs_singlecell
        cb_tiny = _cb(beta=0.01, evidence_tier="Tier2_Convergent")
        r = check_bulk_vs_singlecell("CFH", "AMD", 0.50, [cb_tiny])
        assert r.value_a == pytest.approx(0.50)
        assert r.value_b == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# Rule 4: cross_context_sign_flip
# ---------------------------------------------------------------------------

class TestCrossContextSignFlip:
    def test_sign_flip_detected(self):
        from pipelines.evidence_disagreement import check_cross_context_sign_flip
        cb_mac = _cb(cell_type="retinal_pigment_epithelial", beta=0.6, context_verified=True)
        cb_ent = _cb(cell_type="choroidal_endothelial", beta=-0.5, context_verified=True)
        r = check_cross_context_sign_flip("CFH", "AMD", [cb_mac, cb_ent])
        assert r is not None
        assert r.rule == "cross_context_sign_flip"
        assert r.severity == "block"
        assert {r.cell_type_a, r.cell_type_b} == {"retinal_pigment_epithelial", "choroidal_endothelial"}

    def test_same_sign_no_flip(self):
        from pipelines.evidence_disagreement import check_cross_context_sign_flip
        cb_mac = _cb(cell_type="retinal_pigment_epithelial", beta=0.6, context_verified=True)
        cb_ent = _cb(cell_type="choroidal_endothelial", beta=0.4, context_verified=True)
        assert check_cross_context_sign_flip("CFH", "AMD", [cb_mac, cb_ent]) is None

    def test_only_one_context_verified(self):
        from pipelines.evidence_disagreement import check_cross_context_sign_flip
        cb_mac = _cb(cell_type="retinal_pigment_epithelial", beta=0.6, context_verified=True)
        cb_ent = _cb(cell_type="choroidal_endothelial", beta=-0.5, context_verified=False)
        assert check_cross_context_sign_flip("CFH", "AMD", [cb_mac, cb_ent]) is None

    def test_nan_betas_excluded(self):
        from pipelines.evidence_disagreement import check_cross_context_sign_flip
        import math
        cb_mac = _cb(cell_type="retinal_pigment_epithelial", beta=0.6, context_verified=True)
        cb_nan = _cb(cell_type="choroidal_endothelial", beta=float("nan"), context_verified=True)
        assert check_cross_context_sign_flip("CFH", "AMD", [cb_mac, cb_nan]) is None

    def test_program_id_filter(self):
        from pipelines.evidence_disagreement import check_cross_context_sign_flip
        cb_mac = _cb(program_id="p1", cell_type="retinal_pigment_epithelial", beta=0.6, context_verified=True)
        # Same cell type, different program — deduplicated, so only one per cell type
        cb_ent = _cb(program_id="p2", cell_type="choroidal_endothelial", beta=-0.5, context_verified=True)
        r = check_cross_context_sign_flip("CFH", "AMD", [cb_mac, cb_ent], program_id="p1")
        # p2 excluded → only one cell type left → no flip
        assert r is None

    def test_three_cell_types_finds_first_conflict(self):
        from pipelines.evidence_disagreement import check_cross_context_sign_flip
        cb1 = _cb(cell_type="retinal_pigment_epithelial",  beta=0.6,  context_verified=True)
        cb2 = _cb(cell_type="choroidal_endothelial",  beta=0.4,  context_verified=True)
        cb3 = _cb(cell_type="smooth_muscle", beta=-0.3, context_verified=True)
        r = check_cross_context_sign_flip("CFH", "AMD", [cb1, cb2, cb3])
        assert r is not None
        assert r.severity == "block"


# ---------------------------------------------------------------------------
# run_all_disagreement_checks
# ---------------------------------------------------------------------------

class TestRunAllDisagreementChecks:
    def test_returns_list(self):
        from pipelines.evidence_disagreement import run_all_disagreement_checks
        records = run_all_disagreement_checks("CFH", "AMD", [])
        assert isinstance(records, list)

    def test_empty_when_no_conflict(self):
        from pipelines.evidence_disagreement import run_all_disagreement_checks
        cb = _cb(beta=0.6, context_verified=False)
        records = run_all_disagreement_checks("CFH", "AMD", [cb], mr_gamma=0.5)
        assert records == []

    def test_rule4_fires_independently(self):
        from pipelines.evidence_disagreement import run_all_disagreement_checks
        cb_mac = _cb(cell_type="retinal_pigment_epithelial", beta=0.6, context_verified=True)
        cb_ent = _cb(cell_type="choroidal_endothelial", beta=-0.5, context_verified=True)
        records = run_all_disagreement_checks("CFH", "AMD", [cb_mac, cb_ent])
        rules = {r.rule for r in records}
        assert "cross_context_sign_flip" in rules

    def test_multiple_rules_fire(self):
        from pipelines.evidence_disagreement import run_all_disagreement_checks
        cb_mac = _cb(cell_type="retinal_pigment_epithelial", beta=0.6, context_verified=True)
        cb_ent = _cb(cell_type="choroidal_endothelial", beta=-0.5, context_verified=True)
        # MR says negative → rule 1 fires; cross context → rule 4 fires
        records = run_all_disagreement_checks(
            "CFH", "AMD", [cb_mac, cb_ent], mr_gamma=-0.4
        )
        rules = {r.rule for r in records}
        assert "genetics_vs_perturbation" in rules
        assert "cross_context_sign_flip" in rules

    def test_lincs_triggers_rules_2_and_3(self):
        from pipelines.evidence_disagreement import run_all_disagreement_checks
        cb_t1 = _cb(beta=0.6, context_verified=True, evidence_tier="Tier1_Interventional")
        cb_tiny = _cb(cell_type="choroidal_endothelial", beta=0.01,
                      context_verified=True, evidence_tier="Tier1_Interventional")
        records = run_all_disagreement_checks(
            "CFH", "AMD", [cb_t1, cb_tiny],
            lincs_betas_by_program={"p": -0.45},   # sign flip vs mean T1 (0.305)
        )
        rules = {r.rule for r in records}
        # rule 2: perturb mean=0.305 vs lincs=-0.45 → conflict
        assert "perturbation_vs_chemical" in rules
        # rule 3: lincs 0.45>0.3, mean sc abs=0.305 (not near-zero) → NOT triggered
        assert "bulk_vs_singlecell" not in rules

    def test_all_records_are_correct_type(self):
        from pipelines.evidence_disagreement import run_all_disagreement_checks
        cb_mac = _cb(cell_type="retinal_pigment_epithelial", beta=0.6, context_verified=True)
        cb_ent = _cb(cell_type="choroidal_endothelial", beta=-0.5, context_verified=True)
        records = run_all_disagreement_checks("CFH", "AMD", [cb_mac, cb_ent], mr_gamma=-0.3)
        for r in records:
            assert isinstance(r, EvidenceDisagreementRecord)
            assert r.gene == "CFH"
            assert r.disease == "AMD"
