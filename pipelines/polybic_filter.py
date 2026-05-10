"""
polybic_filter.py — PolyBIC edge selection.

Filters causal edge candidates by penalising complexity relative to evidence tier.
Penalty = (k/2) * log(n) where k is tier-dependent complexity cost.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List


def polybic_selection(
    edges: List[Dict[str, Any]],
    n_samples: int,
) -> List[Dict[str, Any]]:
    """
    PolyBIC filter: retain edges where log|γ| − (k/2)·log(n) > −10.

    k is smaller for well-evidenced tiers, so high-evidence edges survive
    even with modest γ.
    """
    _K: Dict[str, float] = {
        "Tier1_Interventional":    1.0,
        "Tier2_Convergent":        2.0,
        "Tier2_eQTL_direction":    2.0,
        "Tier2_PerturbNominated":  2.0,
        "Tier3_Provisional":       4.0,
        "provisional_virtual":    10.0,
    }
    selected = []
    for edge in edges:
        gamma = edge.get("refined_gamma")
        if gamma is None or (isinstance(gamma, float) and math.isnan(gamma)):
            gamma = edge.get("ota_gamma", 0.0)
        if gamma is None or (isinstance(gamma, float) and math.isnan(gamma)):
            gamma = 0.0
        tier  = edge.get("evidence_tier") or edge.get("dominant_tier") or "provisional_virtual"
        k     = _K.get(tier, 5.0)
        score = math.log(abs(gamma) + 1e-10) - (k / 2.0) * math.log(n_samples)
        if score > -10:
            edge["polybic_score"] = round(score, 4)
            selected.append(edge)
    return selected
