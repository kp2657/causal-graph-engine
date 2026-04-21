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
# Keys are lowercased; values are the canonical short identifier used
# throughout the pipeline (e.g. in disease_query["disease_key"]).
# ---------------------------------------------------------------------------
DISEASE_SHORT_KEY: dict[str, str] = {
    # CAD
    "coronary artery disease":            "CAD",
    "cad":                                "CAD",
    "ischemic heart disease":             "CAD",
    "atherosclerosis":                    "CAD",
    # IBD
    "inflammatory bowel disease":         "IBD",
    "ibd":                                "IBD",
    "crohn's disease":                    "IBD",
    "crohns disease":                     "IBD",
    "ulcerative colitis":                 "IBD",
    "uc":                                 "IBD",
    # RA
    "rheumatoid arthritis":               "RA",
    "ra":                                 "RA",
    # SLE
    "systemic lupus erythematosus":       "SLE",
    "sle":                                "SLE",
    "lupus":                              "SLE",
    # AD
    "alzheimer's disease":                "AD",
    "alzheimer disease":                  "AD",
    "ad":                                 "AD",
    # T2D
    "type 2 diabetes":                    "T2D",
    "type 2 diabetes mellitus":           "T2D",
    "t2d":                                "T2D",
    # T1D
    "type 1 diabetes":                    "T1D",
    "type 1 diabetes mellitus":           "T1D",
    "t1d":                                "T1D",
    # AMD
    "age-related macular degeneration":   "AMD",
    "age related macular degeneration":   "AMD",
    "amd":                                "AMD",
    "macular degeneration":               "AMD",
    # MS
    "multiple sclerosis":                 "MS",
    "ms":                                 "MS",
    # POAG
    "primary open-angle glaucoma":        "POAG",
    "glaucoma":                           "POAG",
    "poag":                               "POAG",
}

# ---------------------------------------------------------------------------
# Short key → EFO identifier
# EFO IDs sourced from EMBL-EBI Experimental Factor Ontology.
# ---------------------------------------------------------------------------
DISEASE_EFO: dict[str, str] = {
    "CAD":  "EFO_0001645",
    "IBD":  "EFO_0003767",
    "RA":   "EFO_0000685",
    "SLE":  "EFO_0002690",
    "AD":   "EFO_0000249",
    "T2D":  "EFO_0001360",
    "T1D":  "EFO_0001359",
    "AMD":  "EFO_0001481",
    "MS":   "EFO_0003885",
    "POAG": "EFO_0004190",
}

# ---------------------------------------------------------------------------
# Short key → canonical display name (title case)
# ---------------------------------------------------------------------------
DISEASE_DISPLAY_NAME: dict[str, str] = {
    "CAD":  "Coronary Artery Disease",
    "IBD":  "Inflammatory Bowel Disease",
    "RA":   "Rheumatoid Arthritis",
    "SLE":  "Systemic Lupus Erythematosus",
    "AD":   "Alzheimer's Disease",
    "T2D":  "Type 2 Diabetes",
    "T1D":  "Type 1 Diabetes",
    "AMD":  "Age-Related Macular Degeneration",
    "MS":   "Multiple Sclerosis",
    "POAG": "Primary Open-Angle Glaucoma",
}

# ---------------------------------------------------------------------------
# Short key → file-safe slug (for JSON output paths etc.)
# ---------------------------------------------------------------------------
DISEASE_SLUG: dict[str, str] = {
    "CAD":  "coronary_artery_disease",
    "IBD":  "inflammatory_bowel_disease",
    "RA":   "rheumatoid_arthritis",
    "SLE":  "systemic_lupus_erythematosus",
    "AD":   "alzheimers_disease",
    "T2D":  "type_2_diabetes",
    "T1D":  "type_1_diabetes",
    "AMD":  "age_related_macular_degeneration",
    "MS":   "multiple_sclerosis",
    "POAG": "primary_open_angle_glaucoma",
}

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def get_disease_key(name: str) -> str | None:
    """Return canonical short key for a disease name (case-insensitive).

    Args:
        name: Any spelling of the disease name (display, alias, short key).

    Returns:
        Short key (e.g. "CAD") or None if unrecognised.
    """
    return DISEASE_SHORT_KEY.get(name.lower().strip())


def get_efo_id(key: str) -> str | None:
    """Return EFO ID for a disease short key.

    Args:
        key: Short key (e.g. "CAD", "AMD") — case-insensitive.

    Returns:
        EFO string (e.g. "EFO_0001645") or None.
    """
    return DISEASE_EFO.get(key.upper().strip())


def get_display_name(key: str) -> str | None:
    """Return human-readable display name for a disease short key."""
    return DISEASE_DISPLAY_NAME.get(key.upper().strip())


def get_slug(key: str) -> str | None:
    """Return file-safe slug for a disease short key."""
    return DISEASE_SLUG.get(key.upper().strip())


def resolve(name_or_key: str) -> dict | None:
    """Resolve any disease name or key to a full info dict.

    Returns:
        {"key", "efo_id", "display_name", "slug"} or None if unrecognised.
    """
    key = get_disease_key(name_or_key) or DISEASE_EFO.get(name_or_key.upper().strip()) and name_or_key.upper().strip()
    if key is None:
        # Try treating the input as a short key directly
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
