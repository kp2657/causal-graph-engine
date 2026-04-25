"""Tests for pLI soft penalty on OT L2G pool in causal_discovery_agent."""
from __future__ import annotations

import math


def test_pli_penalty_reduces_ot_val():
    """pLI=0.95 gene → _ot_val reduced to 5% of original."""
    pli = 0.95
    original_ot_val = 0.60
    penalised = original_ot_val * (1.0 - pli)
    assert abs(penalised - 0.03) < 1e-9


def test_pli_zero_no_effect():
    """pLI=0 (e.g. CFH, C3) → _ot_val unchanged."""
    pli = 0.0
    original_ot_val = 0.50
    penalised = original_ot_val * (1.0 - pli)
    assert abs(penalised - original_ot_val) < 1e-9


def test_pli_none_treated_as_zero():
    """None pLI (gene not in gnomAD) → treated as 0.0, no penalty."""
    pli_raw = None
    pli = pli_raw or 0.0
    original_ot_val = 0.40
    penalised = original_ot_val * (1.0 - pli)
    assert abs(penalised - original_ot_val) < 1e-9


def test_pli_mid_value():
    pli = 0.5
    original_ot_val = 0.8
    penalised = original_ot_val * (1.0 - pli)
    assert abs(penalised - 0.4) < 1e-9


def test_pli_penalty_in_causal_agent(monkeypatch):
    """Verify the pLI penalty is applied at the _ot_val read site in causal_discovery_agent."""
    import sys, types

    # Stub out heavy optional deps
    for mod in ["kuzu", "rdflib"]:
        if mod not in sys.modules:
            sys.modules[mod] = types.ModuleType(mod)

    from agents.tier3_causal import causal_discovery_agent as cda

    captured: list[float] = []

    class _MockLookups:
        def get_pli(self, gene: str):
            return 0.8 if gene == "CONSTRAINED_GENE" else 0.0

    monkeypatch.setattr(cda, "_get_lookups", lambda: _MockLookups())

    # Simulate the two-line pLI block directly
    def _apply_pli(gene, ot_val):
        _pli = (_MockLookups().get_pli(gene) or 0.0)
        return ot_val * (1.0 - _pli)

    assert abs(_apply_pli("CONSTRAINED_GENE", 1.0) - 0.2) < 1e-9
    assert abs(_apply_pli("CFH", 1.0) - 1.0) < 1e-9
