"""
disease_registry.py — Canonical disease name/key/EFO mapping for the causal graph engine.

Single source of truth. All agents, servers, and pipelines should import from here
instead of defining their own local dicts.

Usage:
    from models.disease_registry import get_disease_key, get_efo_id, DISEASE_SHORT_KEY

    key = get_disease_key("coronary artery disease")  # → "CAD"
    efo = get_efo_id("CAD")                           # → "EFO_0001645"
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Canonical display name / alias → short key
# ---------------------------------------------------------------------------
DISEASE_SHORT_KEY: dict[str, str] = {
    # CAD
    "coronary artery disease":            "CAD",
    "cad":                                "CAD",
    "ischemic heart disease":             "CAD",
    "atherosclerosis":                    "CAD",
    # RA
    "rheumatoid arthritis":               "RA",
    "ra":                                 "RA",
    "seropositive rheumatoid arthritis":  "RA",
}

# ---------------------------------------------------------------------------
# Short key → EFO identifier
# ---------------------------------------------------------------------------
DISEASE_EFO: dict[str, str] = {
    "CAD":  "EFO_0001645",
    "RA":   "EFO_0000685",
}

# ---------------------------------------------------------------------------
# Short key → canonical display name (title case)
# ---------------------------------------------------------------------------
DISEASE_DISPLAY_NAME: dict[str, str] = {
    "CAD":  "Coronary Artery Disease",
    "RA":   "Rheumatoid Arthritis",
}

# ---------------------------------------------------------------------------
# Short key → file-safe slug (for JSON output paths etc.)
# ---------------------------------------------------------------------------
DISEASE_SLUG: dict[str, str] = {
    "CAD":  "coronary_artery_disease",
    "RA":   "rheumatoid_arthritis",
}

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def get_disease_key(name: str) -> str | None:
    """Return canonical short key for a disease name (case-insensitive)."""
    return DISEASE_SHORT_KEY.get(name.lower().strip())


def get_efo_id(key: str) -> str | None:
    """Return EFO ID for a disease short key."""
    return DISEASE_EFO.get(key.upper().strip())


def get_display_name(key: str) -> str | None:
    """Return human-readable display name for a disease short key."""
    return DISEASE_DISPLAY_NAME.get(key.upper().strip())


def get_slug(key: str) -> str | None:
    """Return file-safe slug for a disease short key."""
    return DISEASE_SLUG.get(key.upper().strip())


# ---------------------------------------------------------------------------
# Disease-relevant programs — used for entropy filtering and mechanistic
# necessity classification in causal_discovery_agent and target_prioritization_agent.
# ---------------------------------------------------------------------------
DISEASE_PROGRAMS: dict[str, frozenset] = {
    "CAD": frozenset({
        "HALLMARK_INFLAMMATORY_RESPONSE",
        "HALLMARK_IL6_JAK_STAT3_SIGNALING",
        "HALLMARK_TNFA_SIGNALING_VIA_NFKB",
        "lipid_metabolism",
        "foam_cell_program",
        "plaque_inflammation",
        "HALLMARK_ANGIOGENESIS",
        "HALLMARK_OXIDATIVE_PHOSPHORYLATION",
    }),
    "RA": frozenset({
        "HALLMARK_INFLAMMATORY_RESPONSE",
        "HALLMARK_IL6_JAK_STAT3_SIGNALING",
        "HALLMARK_TNFA_SIGNALING_VIA_NFKB",
        "HALLMARK_INTERFERON_GAMMA_RESPONSE",
        "T_cell_activation",
        "B_cell_activation",
    }),
}


def resolve(name_or_key: str) -> dict | None:
    """Resolve any disease name or key to a full info dict."""
    key = get_disease_key(name_or_key) or DISEASE_EFO.get(name_or_key.upper().strip()) and name_or_key.upper().strip()
    if key is None:
        k = name_or_key.upper().strip()
        if k in DISEASE_EFO:
            key = k
    if key is None:
        return None
    return {
        "key":          key,
        "efo_id":       DISEASE_EFO.get(key),
        "display_name": DISEASE_DISPLAY_NAME.get(key),
        "slug":         DISEASE_SLUG.get(key),
    }
