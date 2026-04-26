"""
Tests for gps_transcriptional_convergence.py

8 tests covering: convergence scoring, formula verification, disease-reverser
handling, annotation, z_rges threshold filtering, summarize counts, empty inputs.
"""
from __future__ import annotations

import pytest
from pipelines.gps_transcriptional_convergence import (
    compute_gps_genetic_convergence,
    annotate_reversers_with_convergence,
    summarize_convergence,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_program_reverser(compound_id: str, prog_id: str, z_rges: float) -> dict:
    return {"compound_id": compound_id, "rges": -0.5, "z_rges": z_rges}


def _make_disease_reverser(compound_id: str) -> dict:
    return {"compound_id": compound_id, "rges": -0.5, "z_rges": 3.0}


def _make_anchor(gene: str, top_programs: dict) -> dict:
    return {"target_gene": gene, "top_programs": top_programs}


# ---------------------------------------------------------------------------
# Test 1: shared programs yield a positive convergence score
# ---------------------------------------------------------------------------

def test_convergence_score_shared_programs():
    """Compound on programs [P1, P2], anchor drives [P1, P2] → positive score."""
    program_reversers = {
        "P1": [_make_program_reverser("CMP1", "P1", z_rges=2.0)],
        "P2": [_make_program_reverser("CMP1", "P2", z_rges=3.0)],
    }
    anchors = [_make_anchor("GENE_A", {"P1": 0.5, "P2": 0.8})]

    result = compute_gps_genetic_convergence(
        disease_reversers=[],
        program_reversers=program_reversers,
        genetic_anchors=anchors,
        min_z_rges=1.5,
        min_convergence_score=0.0,
    )

    assert "CMP1" in result
    convergent = result["CMP1"]
    assert len(convergent) == 1
    assert convergent[0]["gene"] == "GENE_A"
    assert convergent[0]["convergence_score"] > 0
    assert convergent[0]["n_shared"] == 2
    assert set(convergent[0]["shared_programs"]) == {"P1", "P2"}


# ---------------------------------------------------------------------------
# Test 2: no program overlap → score = 0, not convergent
# ---------------------------------------------------------------------------

def test_convergence_no_overlap():
    """Compound on [P1], anchor drives [P2] → no convergent annotation."""
    program_reversers = {
        "P1": [_make_program_reverser("CMP1", "P1", z_rges=2.0)],
    }
    anchors = [_make_anchor("GENE_A", {"P2": 0.9})]

    result = compute_gps_genetic_convergence(
        disease_reversers=[],
        program_reversers=program_reversers,
        genetic_anchors=anchors,
        min_z_rges=1.5,
        min_convergence_score=0.0,
    )

    # CMP1 has P1 but GENE_A has P2 only → no shared programs → not in result
    assert "CMP1" not in result


# ---------------------------------------------------------------------------
# Test 3: manually verify the convergence score formula
# ---------------------------------------------------------------------------

def test_convergence_score_formula():
    """
    Convergence score = sum(z × w) / n_shared
    CMP1 on P1 (z=2.0), P2 (z=4.0); GENE_A on P1 (w=0.5), P2 (w=1.0)
    score = (2.0*0.5 + 4.0*1.0) / 2 = (1.0 + 4.0) / 2 = 2.5
    """
    program_reversers = {
        "P1": [_make_program_reverser("CMP1", "P1", z_rges=2.0)],
        "P2": [_make_program_reverser("CMP1", "P2", z_rges=4.0)],
    }
    anchors = [_make_anchor("GENE_A", {"P1": 0.5, "P2": 1.0})]

    result = compute_gps_genetic_convergence(
        disease_reversers=[],
        program_reversers=program_reversers,
        genetic_anchors=anchors,
        min_z_rges=1.5,
        min_convergence_score=0.0,
    )

    assert "CMP1" in result
    score = result["CMP1"][0]["convergence_score"]
    expected = (2.0 * 0.5 + 4.0 * 1.0) / 2
    assert abs(score - expected) < 1e-3


# ---------------------------------------------------------------------------
# Test 4: disease reverser (no program specificity) still gets convergence
# ---------------------------------------------------------------------------

def test_disease_reverser_gets_convergence():
    """
    Disease reversers have no per-program z_rges. They should still receive
    convergence annotation using uniform weight = 1.0 across all anchor programs.
    """
    disease_reversers = [_make_disease_reverser("CMP_D")]
    anchors = [_make_anchor("GENE_B", {"P1": 0.7, "P2": 0.3})]

    result = compute_gps_genetic_convergence(
        disease_reversers=disease_reversers,
        program_reversers={},
        genetic_anchors=anchors,
        min_z_rges=1.5,
        min_convergence_score=0.0,
    )

    assert "CMP_D" in result
    assert len(result["CMP_D"]) == 1
    assert result["CMP_D"][0]["gene"] == "GENE_B"
    # Both P1 and P2 should be shared (disease reverser gets all anchor programs)
    assert result["CMP_D"][0]["n_shared"] == 2


# ---------------------------------------------------------------------------
# Test 5: annotate_reversers_with_convergence adds the field to each reverser
# ---------------------------------------------------------------------------

def test_annotate_adds_field():
    """After annotation, reverser dicts have 'convergent_genetic_targets_hypothesis' key."""
    disease_reversers = [_make_disease_reverser("CMP_A")]
    program_reversers = {
        "P1": [_make_program_reverser("CMP_B", "P1", z_rges=2.5)],
    }
    anchors = [_make_anchor("GENE_C", {"P1": 0.6})]

    dr_out, pr_out = annotate_reversers_with_convergence(
        disease_reversers=disease_reversers,
        program_reversers=program_reversers,
        genetic_anchors=anchors,
    )

    # Disease reverser should have the key
    assert "annotation" in dr_out[0]
    assert "convergent_genetic_targets_hypothesis" in dr_out[0]["annotation"]

    # Program reverser should have the key
    assert "annotation" in pr_out["P1"][0]
    assert "convergent_genetic_targets_hypothesis" in pr_out["P1"][0]["annotation"]


# ---------------------------------------------------------------------------
# Test 6: min_z_rges threshold filters out low-z compounds
# ---------------------------------------------------------------------------

def test_min_z_threshold():
    """Compound with |z_rges| < min_z_rges should not count toward convergence."""
    program_reversers = {
        "P1": [_make_program_reverser("CMP_LOW", "P1", z_rges=1.0)],  # below 1.5 threshold
    }
    anchors = [_make_anchor("GENE_D", {"P1": 0.9})]

    result = compute_gps_genetic_convergence(
        disease_reversers=[],
        program_reversers=program_reversers,
        genetic_anchors=anchors,
        min_z_rges=1.5,
        min_convergence_score=0.0,
    )

    # CMP_LOW has |z_rges|=1.0 < 1.5 → excluded from compound_programs → not in result
    assert "CMP_LOW" not in result


# ---------------------------------------------------------------------------
# Test 7: summarize_convergence returns correct counts
# ---------------------------------------------------------------------------

def test_summarize_counts():
    """2 convergent disease reversers, 3 not → n_disease_convergent=2."""
    def _hit_with_convergence(cid: str, genes: list[str]) -> dict:
        targets = [{"gene": g, "convergence_score": 0.5, "shared_programs": ["P1"], "n_shared": 1}
                   for g in genes]
        return {"compound_id": cid, "annotation": {"convergent_genetic_targets_hypothesis": targets}}

    def _hit_no_convergence(cid: str) -> dict:
        return {"compound_id": cid, "annotation": {"convergent_genetic_targets_hypothesis": []}}

    disease_reversers = [
        _hit_with_convergence("CMP1", ["GENE_A"]),
        _hit_with_convergence("CMP2", ["GENE_B"]),
        _hit_no_convergence("CMP3"),
        _hit_no_convergence("CMP4"),
        _hit_no_convergence("CMP5"),
    ]

    summary = summarize_convergence(disease_reversers=disease_reversers, program_reversers={})

    assert summary["n_disease_reversers"] == 5
    assert summary["n_disease_convergent"] == 2
    assert summary["n_program_reversers"] == 0
    assert summary["n_program_convergent"] == 0
    assert len(summary["convergent_pairs"]) == 2


# ---------------------------------------------------------------------------
# Test 8: empty inputs → no crash, returns empty dict
# ---------------------------------------------------------------------------

def test_empty_inputs():
    """Empty reversers and anchors should not crash and return empty dict."""
    result = compute_gps_genetic_convergence(
        disease_reversers=[],
        program_reversers={},
        genetic_anchors=[],
    )
    assert result == {}

    dr_out, pr_out = annotate_reversers_with_convergence(
        disease_reversers=[],
        program_reversers={},
        genetic_anchors=[],
    )
    assert dr_out == []
    assert pr_out == {}

    dr_out2, pr_out2 = annotate_reversers_with_convergence(
        disease_reversers=[],
        program_reversers={},
        genetic_anchors=None,
    )
    assert dr_out2 == []
    assert pr_out2 == {}

    summary = summarize_convergence(disease_reversers=[], program_reversers={})
    assert summary["n_disease_reversers"] == 0
    assert summary["n_disease_convergent"] == 0
    assert summary["convergent_pairs"] == []
