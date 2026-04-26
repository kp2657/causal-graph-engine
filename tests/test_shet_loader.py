"""Tests for pipelines/shet_loader.py — GeneBayes shet penalty."""
from __future__ import annotations

import math
import types
import sys


def test_shet_penalty_unconstrained():
    from pipelines.shet_loader import get_shet_penalty
    # Monkeypatch get_shet to return a very low shet
    import pipelines.shet_loader as sl
    orig = sl.get_shet
    sl.get_shet = lambda gene: 0.0001  # below _SHET_LOW (0.0005)
    try:
        assert get_shet_penalty("ANY_GENE") == 1.0
    finally:
        sl.get_shet = orig


def test_shet_penalty_highly_constrained():
    from pipelines.shet_loader import get_shet_penalty
    import pipelines.shet_loader as sl
    orig = sl.get_shet
    sl.get_shet = lambda gene: 0.1  # above _SHET_HIGH (0.05)
    try:
        p = get_shet_penalty("CONSTRAINED")
        assert p == 0.20
    finally:
        sl.get_shet = orig


def test_shet_penalty_intermediate():
    from pipelines.shet_loader import get_shet_penalty
    import pipelines.shet_loader as sl
    orig = sl.get_shet
    # Mid-range shet → penalty between 0.20 and 1.0
    sl.get_shet = lambda gene: 0.005  # between 0.0005 and 0.05
    try:
        p = get_shet_penalty("MID_GENE")
        assert 0.20 < p < 1.0
    finally:
        sl.get_shet = orig


def test_shet_penalty_unknown_gene():
    from pipelines.shet_loader import get_shet_penalty
    import pipelines.shet_loader as sl
    orig = sl.get_shet
    sl.get_shet = lambda gene: None
    try:
        assert get_shet_penalty("UNKNOWN_GENE") == 1.0
    finally:
        sl.get_shet = orig


def test_shet_penalty_floor():
    """Penalty never goes below _MAX_PENALTY (0.20)."""
    from pipelines.shet_loader import get_shet_penalty
    import pipelines.shet_loader as sl
    orig = sl.get_shet
    sl.get_shet = lambda gene: 999.0  # extreme shet
    try:
        p = get_shet_penalty("SUPER_CONSTRAINED")
        assert p >= 0.20
    finally:
        sl.get_shet = orig


def test_shet_penalty_log_linear_monotonic():
    """Penalty should decrease monotonically as shet increases."""
    from pipelines.shet_loader import get_shet_penalty
    import pipelines.shet_loader as sl
    orig = sl.get_shet
    shet_vals = [0.001, 0.005, 0.01, 0.03, 0.05]
    penalties = []
    for sv in shet_vals:
        sl.get_shet = lambda gene, v=sv: v
        penalties.append(get_shet_penalty("G"))
    sl.get_shet = orig
    # Each penalty should be <= the previous
    for i in range(1, len(penalties)):
        assert penalties[i] <= penalties[i - 1], (
            f"Penalty not monotonic at index {i}: {penalties}"
        )


def test_shet_agent_integration(monkeypatch):
    """shet penalty is applied to ota_gamma for constrained genes in causal_discovery_agent."""
    for mod in ("kuzu", "rdflib"):
        if mod not in sys.modules:
            sys.modules[mod] = types.ModuleType(mod)

    import pipelines.shet_loader as sl
    # constrained gene → max penalty (0.20)
    monkeypatch.setattr(sl, "get_shet", lambda gene: 0.1 if gene == "CONSTRAINED" else 0.0001)

    # Simulate the shet penalty block from causal_discovery_agent
    records = [
        {"gene": "CONSTRAINED", "ota_gamma": 0.5, "dominant_tier": "Tier3_Provisional"},
        {"gene": "CFH", "ota_gamma": 0.5, "dominant_tier": "Tier2_Convergent"},
        {"gene": "SAFE", "ota_gamma": 0.5, "dominant_tier": "Tier3_Provisional"},
    ]
    from pipelines.shet_loader import get_shet_penalty
    for r in records:
        if (r.get("dominant_tier") or "").startswith("Tier2"):
            continue
        sp = get_shet_penalty(r["gene"])
        if sp < 1.0:
            r["ota_gamma"] = round(r["ota_gamma"] * sp, 6)
            r["shet_penalty"] = sp

    # CONSTRAINED: shet=0.1 → max penalty 0.20 → gamma 0.5 * 0.20 = 0.10
    assert abs(records[0]["ota_gamma"] - 0.10) < 1e-5
    assert records[0].get("shet_penalty") == 0.20
    # CFH: Tier2 → exempt
    assert abs(records[1]["ota_gamma"] - 0.5) < 1e-5
    assert "shet_penalty" not in records[1]
    # SAFE: shet=0.0001 → no penalty
    assert abs(records[2]["ota_gamma"] - 0.5) < 1e-5
