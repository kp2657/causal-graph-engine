"""Tests for mechanistic necessity filter in causal_discovery_agent."""
from __future__ import annotations

import sys
import types


def _get_filter_fn():
    for mod in ["kuzu", "rdflib"]:
        if mod not in sys.modules:
            sys.modules[mod] = types.ModuleType(mod)
    from agents.tier3_causal.causal_discovery_agent import _mechanistic_necessity_filter
    return _mechanistic_necessity_filter


def test_gene_with_beta_above_threshold_kept():
    fn = _get_filter_fn()
    progs = frozenset({"HALLMARK_COMPLEMENT"})
    beta = {"CFH": {"HALLMARK_COMPLEMENT": 0.8}}
    genes = [{"gene": "CFH", "ota_gamma": 0.5, "dominant_tier": "Tier3_Provisional"}]
    result = fn(genes, beta, progs)
    assert any(r["gene"] == "CFH" for r in result)


def test_gene_with_beta_zero_and_no_programs_removed():
    fn = _get_filter_fn()
    progs = frozenset({"HALLMARK_COMPLEMENT"})
    beta = {"BYSTANDER": {"HALLMARK_COMPLEMENT": 0.0}}
    genes = [{"gene": "BYSTANDER", "ota_gamma": 0.2, "dominant_tier": "Tier3_Provisional"}]
    result = fn(genes, beta, progs, min_keep=0)
    assert not any(r["gene"] == "BYSTANDER" for r in result)


def test_tier2_exempt():
    fn = _get_filter_fn()
    progs = frozenset({"lipid_metabolism"})
    beta = {"PCSK9": {"lipid_metabolism": 0.0}}    # β≈0
    genes = [{"gene": "PCSK9", "ota_gamma": 0.6,
              "dominant_tier": "Tier3_Provisional", "dominant_tier": "Tier2_Convergent"}]
    result = fn(genes, beta, progs, min_keep=0)
    assert any(r["gene"] == "PCSK9" for r in result)


def test_min_keep_floor():
    """If fewer than min_keep genes pass, return the original list capped at min_keep."""
    fn = _get_filter_fn()
    progs = frozenset({"P1"})
    beta = {f"GENE{i}": {"P1": 0.0} for i in range(10)}
    genes = [{"gene": f"GENE{i}", "ota_gamma": float(i) * 0.01,
              "dominant_tier": "Tier3_Provisional", "dominant_tier": "Tier3_Provisional"}
             for i in range(10)]
    result = fn(genes, beta, progs, min_keep=5)
    assert len(result) == 5   # floor applied


def test_top_programs_dict_counts_as_mechanistic():
    fn = _get_filter_fn()
    progs = frozenset({"P1"})
    beta = {"GENE_X": {"P1": 0.0}}
    genes = [{"gene": "GENE_X", "ota_gamma": 0.3,
              "dominant_tier": "Tier3_Provisional",
              "dominant_tier": "Tier3_Provisional",
              "top_programs": {"P1": 0.4}}]   # non-empty top_programs → kept
    result = fn(genes, beta, progs, min_keep=0)
    assert any(r["gene"] == "GENE_X" for r in result)


def test_call_site_guards_empty_programs():
    """The agent call site wraps _mechanistic_necessity_filter in `if _disease_programs`.
    Verify DISEASE_PROGRAMS is always non-empty for supported diseases."""
    from models.disease_registry import DISEASE_PROGRAMS
    for key in ("RA", "CAD"):
        assert len(DISEASE_PROGRAMS[key]) > 0, f"DISEASE_PROGRAMS[{key}] should not be empty"
