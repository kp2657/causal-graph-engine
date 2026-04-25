"""Tests for the `inherited_genetic_evidence` writer helper and the
master-regulator narrative prefix.

Surfaces the OT program γ of programs that a mechanistic-only gene
controls, so a target with `ot_genetic_score < 0.05` can still be
defended via the heritability of its downstream programs.
"""
from __future__ import annotations

import pytest

from agents.tier5_writer.scientific_writer_agent import (
    _inherited_genetic_evidence,
    _lookup_program_gamma,
    _causal_narrative,
)


# ---------------------------------------------------------------------------
# _lookup_program_gamma — shape drift handling
# ---------------------------------------------------------------------------

def test_lookup_gamma_scalar():
    assert _lookup_program_gamma({"P": 0.42}, "P", "AMD") == pytest.approx(0.42)


def test_lookup_gamma_trait_keyed():
    g = {"P": {"AMD": 0.7, "CAD": 0.1}}
    assert _lookup_program_gamma(g, "P", "AMD") == pytest.approx(0.7)


def test_lookup_gamma_dict_with_gamma_key():
    assert _lookup_program_gamma({"P": {"gamma": 0.33}}, "P", "AMD") == pytest.approx(0.33)


def test_lookup_gamma_missing_returns_none():
    assert _lookup_program_gamma({"OTHER": 0.2}, "P", "AMD") is None


# ---------------------------------------------------------------------------
# _inherited_genetic_evidence
# ---------------------------------------------------------------------------

def test_inherited_returns_none_when_gene_has_direct_gwas():
    t = {
        "target_gene": "CFH",
        "ot_genetic_score": 0.8,
        "top_programs": {"COMPLEMENT": 0.5},
    }
    assert _inherited_genetic_evidence(t, {"COMPLEMENT": 0.4}, "AMD") is None


def test_inherited_returns_ranked_list_for_mechanistic_only():
    t = {
        "target_gene": "ATP6V1B2",
        "ot_genetic_score": 0.0,
        "top_programs": {"HYPOXIA": 0.21, "GLYCOLYSIS": 0.06, "TRANSLATION": 0.13},
    }
    gammas = {"HYPOXIA": {"CAD": 0.85}, "GLYCOLYSIS": {"CAD": 0.12}, "TRANSLATION": {"CAD": 0.4}}
    out = _inherited_genetic_evidence(t, gammas, "CAD", top_n=3)
    assert isinstance(out, list) and len(out) == 3
    progs = [e["program"] for e in out]
    # Sorted by |contribution| desc — HYPOXIA(0.21) > TRANSLATION(0.13) > GLYCOLYSIS(0.06)
    assert progs == ["HYPOXIA", "TRANSLATION", "GLYCOLYSIS"]
    assert out[0]["program_gamma"] == pytest.approx(0.85)


def test_inherited_empty_list_when_no_programs():
    t = {"target_gene": "G", "ot_genetic_score": 0.0, "top_programs": {}}
    assert _inherited_genetic_evidence(t, {}, "AMD") == []


def test_inherited_handles_list_top_programs_as_empty():
    # Legacy shape: top_programs may be a list (e.g. older checkpoints).
    t = {"target_gene": "G", "ot_genetic_score": 0.0, "top_programs": ["P1", "P2"]}
    assert _inherited_genetic_evidence(t, {}, "AMD") == []


# ---------------------------------------------------------------------------
# _causal_narrative — master-regulator prefix
# ---------------------------------------------------------------------------

def test_narrative_emits_master_regulator_prefix_when_bridge_is_strong():
    target = {
        "target_gene": "ATP6V1B2",
        "ota_gamma": 0.08,
        "evidence_tier": "Tier3_Provisional",
        "ot_genetic_score": 0.0,
        "top_programs": {"HYPOXIA": 2.1},
        "key_evidence": ["Perturb-seq"],
        "known_drugs": [],
        "max_phase": 0,
    }
    gammas = {"HYPOXIA": {"CAD": 0.85}}
    out = _causal_narrative(
        target, chemistry={}, trials={},
        gamma_per_program_per_trait=gammas, trait="CAD",
    )
    assert "master regulator of HYPOXIA" in out
    assert "program γ=0.85" in out
    assert "no direct GWAS hit" in out


def test_narrative_suppresses_prefix_for_direct_gwas_gene():
    target = {
        "target_gene": "CFH",
        "ota_gamma": 0.5,
        "evidence_tier": "Tier2_Convergent",
        "ot_genetic_score": 0.9,
        "top_programs": {"COMPLEMENT": 0.4},
        "key_evidence": ["GWAS L2G"],
        "known_drugs": [],
        "max_phase": 2,
    }
    gammas = {"COMPLEMENT": {"AMD": 0.9}}
    out = _causal_narrative(
        target, chemistry={}, trials={},
        gamma_per_program_per_trait=gammas, trait="AMD",
    )
    assert "master regulator" not in out


def test_narrative_suppresses_prefix_when_program_gamma_below_threshold():
    target = {
        "target_gene": "X",
        "ota_gamma": 0.05,
        "evidence_tier": "Tier3_Provisional",
        "ot_genetic_score": 0.0,
        "top_programs": {"P": 0.3},
        "key_evidence": [],
        "known_drugs": [],
        "max_phase": 0,
    }
    gammas = {"P": {"AMD": 0.1}}  # below 0.3 threshold
    out = _causal_narrative(
        target, chemistry={}, trials={},
        gamma_per_program_per_trait=gammas, trait="AMD",
    )
    assert "master regulator" not in out
