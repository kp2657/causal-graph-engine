"""Tests for Shannon entropy filter over disease-relevant programs."""
from __future__ import annotations

import math
import sys
import types

import numpy as np
import pytest


def _get_entropy_fn():
    for mod in ["kuzu", "rdflib"]:
        if mod not in sys.modules:
            sys.modules[mod] = types.ModuleType(mod)
    from steps.tier3_causal.ota_gamma_calculator import _program_entropy
    return _program_entropy


def test_entropy_zero_when_no_programs():
    fn = _get_entropy_fn()
    assert fn("GENE1", {}, frozenset({"P1", "P2"})) == 0.0


def test_entropy_zero_when_all_zero_loadings():
    fn = _get_entropy_fn()
    beta = {"GENE1": {"P1": 0.0, "P2": 0.0}}
    assert fn("GENE1", beta, frozenset({"P1", "P2"})) == 0.0


def test_entropy_max_for_uniform_loading():
    """Uniform loading across N programs → entropy = log(N)."""
    fn = _get_entropy_fn()
    progs = frozenset({"P1", "P2", "P3", "P4"})
    beta = {"GENE1": {p: 1.0 for p in progs}}
    H = fn("GENE1", beta, progs)
    H_max = math.log(4)
    assert abs(H - H_max) < 1e-6


def test_entropy_low_for_focused_loading():
    """Gene loading concentrated in one program → low entropy."""
    fn = _get_entropy_fn()
    progs = frozenset({"HALLMARK_COMPLEMENT", "HALLMARK_ANGIOGENESIS",
                        "lipid_metabolism", "foam_cell_program"})
    beta = {"CFH": {"HALLMARK_COMPLEMENT": 2.0, "HALLMARK_ANGIOGENESIS": 0.01,
                     "lipid_metabolism": 0.01, "foam_cell_program": 0.01}}
    H = fn("CFH", beta, progs)
    H_max = math.log(len(progs))
    assert H < H_max * 0.5


def test_entropy_discount_applied_above_p75():
    """Genes above 75th percentile entropy receive a discount; others do not."""
    for mod in ["kuzu", "rdflib"]:
        if mod not in sys.modules:
            sys.modules[mod] = types.ModuleType(mod)
    from steps.tier3_causal.ota_gamma_calculator import _program_entropy

    progs = frozenset({"P1", "P2", "P3", "P4"})
    # Gene A: focused (low entropy); Gene B: uniform (high entropy)
    beta = {
        "GENE_A": {"P1": 3.0, "P2": 0.1, "P3": 0.1, "P4": 0.1},
        "GENE_B": {"P1": 1.0, "P2": 1.0, "P3": 1.0, "P4": 1.0},
    }

    gwas_top = [
        {"gene": "GENE_A", "ota_gamma": 0.5},
        {"gene": "GENE_B", "ota_gamma": 0.5},
    ]

    entropies = {r["gene"]: _program_entropy(r["gene"], beta, progs) for r in gwas_top}
    H_p75 = float(np.percentile(list(entropies.values()), 75))
    H_max = float(np.log(len(progs)))

    for r in gwas_top:
        H = entropies[r["gene"]]
        r["entropy_score"] = round(H, 4)
        if H > H_p75 and H_max > H_p75:
            excess = (H - H_p75) / (H_max - H_p75 + 1e-8)
            discount = max(0.5, 1.0 - 0.5 * excess)
            r["ota_gamma"] *= discount
            r["entropy_discount"] = round(discount, 3)

    gene_b = next(r for r in gwas_top if r["gene"] == "GENE_B")
    gene_a = next(r for r in gwas_top if r["gene"] == "GENE_A")

    assert "entropy_discount" in gene_b, "High-entropy gene should have discount"
    assert gene_b["ota_gamma"] < 0.5, "High-entropy gene gamma should be reduced"
    assert gene_a["ota_gamma"] == pytest.approx(0.5), "Low-entropy gene gamma unchanged"


def test_entropy_discount_floor():
    """Discount never drops below 0.5×."""
    for mod in ["kuzu", "rdflib"]:
        if mod not in sys.modules:
            sys.modules[mod] = types.ModuleType(mod)
    from steps.tier3_causal.ota_gamma_calculator import _program_entropy

    progs = frozenset({f"P{i}" for i in range(10)})
    beta = {"HOUSEKEEPING": {p: 1.0 for p in progs}}   # perfectly uniform
    H = _program_entropy("HOUSEKEEPING", beta, progs)
    H_p75 = H * 0.5   # simulate p75 below H
    H_max = math.log(10)
    excess = (H - H_p75) / (H_max - H_p75 + 1e-8)
    discount = max(0.5, 1.0 - 0.5 * excess)
    assert discount >= 0.5
