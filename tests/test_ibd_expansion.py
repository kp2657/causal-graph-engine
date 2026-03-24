"""
test_ibd_expansion.py — Tests for IBD disease expansion (Step 55).

Validates:
  - EFO/ICD-10/modifier config for IBD in phenotype_architect
  - IBD provisional γ values in PROVISIONAL_GAMMAS
  - IBD anchors in ANCHOR_EDGES and REQUIRED_ANCHORS_BY_DISEASE
  - DISEASE_TRAIT_MAP expanded for IBD
  - Disease name resolver handles IBD aliases
  - Regression guard: IBD required anchors ⊆ ANCHOR_EDGES
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# 1. Phenotype architect — EFO/ICD-10/modifier config
# ---------------------------------------------------------------------------

def test_ibd_efo_lookup():
    from agents.tier1_phenomics.phenotype_architect import DISEASE_EFO_MAP
    assert DISEASE_EFO_MAP["inflammatory bowel disease"] == "EFO_0003767"
    assert DISEASE_EFO_MAP["ibd"] == "EFO_0003767"


def test_ibd_icd10_codes():
    from agents.tier1_phenomics.phenotype_architect import EFO_ICD10_MAP
    icd10 = EFO_ICD10_MAP["EFO_0003767"]
    assert "K50" in icd10   # Crohn's disease
    assert "K51" in icd10   # Ulcerative colitis


def test_ibd_modifier_map():
    from agents.tier1_phenomics.phenotype_architect import DISEASE_MODIFIER_MAP
    modifiers = DISEASE_MODIFIER_MAP["EFO_0003767"]
    assert "germline" in modifiers
    assert "drug" in modifiers


def test_crohns_uc_efo_entries():
    from agents.tier1_phenomics.phenotype_architect import DISEASE_EFO_MAP
    assert "crohn's disease" in DISEASE_EFO_MAP
    assert "ulcerative colitis" in DISEASE_EFO_MAP
    assert "uc" in DISEASE_EFO_MAP


# ---------------------------------------------------------------------------
# 2. Provisional γ values for IBD
# ---------------------------------------------------------------------------

def test_ibd_provisional_gammas_exist():
    from pipelines.ota_gamma_estimation import PROVISIONAL_GAMMAS
    ibd_keys = [k for k in PROVISIONAL_GAMMAS if k[1] == "IBD"]
    assert len(ibd_keys) >= 3, f"Expected ≥3 IBD γ entries, got {len(ibd_keys)}: {ibd_keys}"


def test_ibd_nfkb_gamma_tier():
    from pipelines.ota_gamma_estimation import PROVISIONAL_GAMMAS
    entry = PROVISIONAL_GAMMAS[("inflammatory_NF-kB", "IBD")]
    assert entry["gamma"] > 0
    assert entry["evidence_tier"] == "Tier2_Convergent"


def test_ibd_tnf_signaling_gamma():
    from pipelines.ota_gamma_estimation import PROVISIONAL_GAMMAS
    entry = PROVISIONAL_GAMMAS[("TNF_signaling", "IBD")]
    assert entry["gamma"] > 0
    assert entry["evidence_tier"] == "Tier2_Convergent"


def test_estimate_gamma_returns_ibd_value():
    from pipelines.ota_gamma_estimation import estimate_gamma
    result = estimate_gamma("inflammatory_NF-kB", "IBD")
    assert result["gamma"] > 0
    assert result["evidence_tier"] in ("Tier2_Convergent", "Tier3_Provisional")


def test_estimate_gamma_returns_zero_for_unknown_ibd_program():
    from pipelines.ota_gamma_estimation import estimate_gamma
    result = estimate_gamma("nonexistent_program_xyz", "IBD")
    assert result["gamma"] == 0.0
    assert result["evidence_tier"] == "provisional_virtual"


# ---------------------------------------------------------------------------
# 3. Schema — anchor edges and required anchors
# ---------------------------------------------------------------------------

def test_ibd_anchor_edges_exist():
    from graph.schema import ANCHOR_EDGES
    ibd_anchors = [a for a in ANCHOR_EDGES if a["to"] == "IBD"]
    genes = {a["from"] for a in ibd_anchors}
    assert "NOD2" in genes,  "NOD2→IBD anchor missing from ANCHOR_EDGES"
    assert "IL23R" in genes, "IL23R→IBD anchor missing from ANCHOR_EDGES"
    assert "TNF" in genes,   "TNF→IBD anchor missing from ANCHOR_EDGES"


def test_ibd_required_anchors_defined():
    from graph.schema import REQUIRED_ANCHORS_BY_DISEASE
    assert "IBD" in REQUIRED_ANCHORS_BY_DISEASE
    anchors = REQUIRED_ANCHORS_BY_DISEASE["IBD"]
    assert len(anchors) >= 3
    anchor_genes = {a[0] for a in anchors}
    assert "NOD2" in anchor_genes
    assert "IL23R" in anchor_genes
    assert "TNF" in anchor_genes


def test_ibd_required_anchors_subset_of_anchor_edges():
    """Regression guard: IBD required anchors must exist in ANCHOR_EDGES."""
    from graph.schema import ANCHOR_EDGES, REQUIRED_ANCHORS_BY_DISEASE
    global_set = {(a["from"], a["to"]) for a in ANCHOR_EDGES}
    for from_node, to_node in REQUIRED_ANCHORS_BY_DISEASE["IBD"]:
        assert (from_node, to_node) in global_set, (
            f"IBD required anchor ({from_node!r}, {to_node!r}) not in ANCHOR_EDGES"
        )


# ---------------------------------------------------------------------------
# 4. DISEASE_TRAIT_MAP — expanded IBD traits
# ---------------------------------------------------------------------------

def test_ibd_disease_trait_map_expanded():
    from graph.schema import DISEASE_TRAIT_MAP
    traits = DISEASE_TRAIT_MAP["IBD"]
    assert "IBD" in traits
    assert "Crohn_disease" in traits
    assert "UC" in traits
    assert "CRP" in traits


# ---------------------------------------------------------------------------
# 5. Disease name resolver — IBD aliases
# ---------------------------------------------------------------------------

def test_ibd_name_resolver_full():
    from graph.schema import _DISEASE_SHORT_NAMES_FOR_ANCHORS
    assert _DISEASE_SHORT_NAMES_FOR_ANCHORS["inflammatory bowel disease"] == "IBD"
    assert _DISEASE_SHORT_NAMES_FOR_ANCHORS["ibd"] == "IBD"


def test_ibd_name_resolver_crohns_alias():
    from graph.schema import _DISEASE_SHORT_NAMES_FOR_ANCHORS
    # Crohn's maps to IBD (uses same anchor set)
    assert _DISEASE_SHORT_NAMES_FOR_ANCHORS.get("crohn's disease") == "IBD"
    assert _DISEASE_SHORT_NAMES_FOR_ANCHORS.get("ulcerative colitis") == "IBD"


# ---------------------------------------------------------------------------
# 6. Compute Ota γ for an IBD gene (unit test, no DB)
# ---------------------------------------------------------------------------

def test_compute_ota_gamma_ibd():
    from pipelines.ota_gamma_estimation import compute_ota_gamma

    # Stub β: NOD2 upregulates innate_immune_sensing + inflammatory_NF-kB
    beta_estimates = {
        "innate_immune_sensing":  {"beta": 0.7,  "evidence_tier": "Tier3_Provisional"},
        "inflammatory_NF-kB":    {"beta": 0.5,  "evidence_tier": "Tier2_Convergent"},
        "TNF_signaling":         {"beta": 0.3,  "evidence_tier": "provisional_virtual"},
    }
    # Real γ from PROVISIONAL_GAMMAS
    gamma_estimates = {
        "innate_immune_sensing": {"gamma": 0.33, "evidence_tier": "Tier2_Convergent"},
        "inflammatory_NF-kB":   {"gamma": 0.39, "evidence_tier": "Tier2_Convergent"},
        "TNF_signaling":        {"gamma": 0.45, "evidence_tier": "Tier2_Convergent"},
    }

    result = compute_ota_gamma("NOD2", "IBD", beta_estimates, gamma_estimates)
    assert result["ota_gamma"] > 0, "NOD2→IBD Ota γ should be positive"
    assert result["n_programs_contributing"] >= 2
    assert result["dominant_tier"] in ("Tier2_Convergent", "Tier3_Provisional")
