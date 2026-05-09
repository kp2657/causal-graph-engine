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


# ---------------------------------------------------------------------------
# Disease-relevant programs — used for entropy filtering and mechanistic
# necessity classification in ota_gamma_calculator and target_ranker.
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


# ---------------------------------------------------------------------------
# High-confidence GWAS anchor genes per disease.
# Source: PAV-supported gene-disease associations (protein-altering variants)
# with L2G ≥ 0.5 from Tsepilov et al. 2026 (Human Pleiotropic Map) ST7.
# These are used as hard-coded seeds for OT L2G anchor bypass when the
# pipeline has no Perturb-seq β for a GWAS-anchored gene.
# ---------------------------------------------------------------------------
DISEASE_GWAS_ANCHORS: dict[str, frozenset] = {
    "CAD": frozenset({
        "PCSK9", "LDLR", "APOB", "HMGCR", "LPA", "CETP", "SORT1",
        "TRIB1", "PPARG", "NOS3",
    }),
    "RA": frozenset({
        # PAV-supported, L2G ≥ 0.5, from ST7 (Tsepilov 2026) filtered to EFO_0000685
        "IFNGR2",   # L2G 0.951, eQTL+pQTL coloc, GCST90132222
        "AHNAK2",   # L2G 0.941, GCST90013534
        "DNASE1L3", # L2G 0.944, GCST007278
        "FCRL3",    # L2G 0.923, eQTL+pQTL coloc, GCST90013534
        "FCGR2A",   # L2G 0.910, eQTL+pQTL coloc, GCST90132222
        "DCLRE1B",  # L2G 0.914, eQTL coloc, GCST90302863
        "WDFY4",    # L2G 0.904, GCST90013534
        "NCF2",     # L2G 0.949, GCST007278
        "AIRE",     # L2G 0.861, GCST90132222
        "SIRPG",    # L2G 0.811, eQTL+pQTL coloc, GCST90302863
        "SWAP70",   # L2G 0.856, eQTL+pQTL coloc, GCST011389
        "PADI4",    # L2G 0.691, eQTL coloc, RA-classic citrullination
        "IRAK1",    # L2G 0.646, eQTL coloc, IL-1/TLR signaling
        "PLD4",     # L2G 0.710, eQTL coloc, GCST011389
        # Clinically validated targets (ST5, L2G ≥ 0.5, Phase ≥ 3)
        "TYK2",     # L2G 0.934, Phase 4 (deucravacitinib)
        "IL6R",     # L2G 0.940, Phase 4 (tocilizumab/sarilumab)
        "TRAF3IP2", # L2G 0.969, Phase 3
        "CD40",     # L2G 0.948, Phase 1
        "IL12B",    # L2G 0.875, Phase 2
    }),
}


# ---------------------------------------------------------------------------
# GPS force-emulate genes — KD/KO targets validated in disease-relevant cells
# but absent from genome-scale Perturb-seq atlases.  Always attempted for GPS
# emulation regardless of dominant_tier; GPS skips gracefully if no signature.
# Source: Schnitzler 2024 Fig. 2a KD targets (GSE210681 HAEC Perturb-seq).
# ---------------------------------------------------------------------------
GPS_FORCE_GENES: dict[str, list[str]] = {
    "CAD": ["CCM2", "PLPP3"],
    "RA":  [],
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
