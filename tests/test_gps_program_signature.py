"""
tests/test_phase_gps_program_sig.py

Tests for _build_program_signature bidirectional centering fix.

Core invariant: the signature must always contain both positive and negative
weights when h5ad DEG data is available, regardless of whether all absolute
log2FC values share the same sign (the CAD all-down-in-endothelial case).
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipelines.gps_disease_screen import _build_program_signature


# ---------------------------------------------------------------------------
# Mock NMF loadings used across tests
# ---------------------------------------------------------------------------

MOCK_NMF_INFO = {
    "top_genes": [
        {"gene": "GENEА", "weight": 1.0},
        {"gene": "GENEB", "weight": 0.8},
        {"gene": "GENEC", "weight": 0.6},
        {"gene": "GENED", "weight": 0.4},
    ],
    "gene_loadings": {},
}


def _patch_loadings(info=MOCK_NMF_INFO):
    return patch(
        "mcp_servers.burden_perturb_server.get_program_gene_loadings",
        return_value=info,
    )


# ---------------------------------------------------------------------------
# 1. All-down case (the CAD NMF bug): bidirectional split guaranteed
# ---------------------------------------------------------------------------

def test_all_down_genes_still_bidirectional():
    """
    When every program gene has negative log2FC (all-down in disease),
    centering on program mean must still produce both + and - weights.
    """
    h5ad = {"GENEА": -0.5, "GENEB": -1.0, "GENEC": -2.0, "GENED": -0.3}
    with _patch_loadings():
        sig = _build_program_signature("PROG1", direction=1.0, h5ad_deg=h5ad)

    assert sig, "signature must not be empty"
    pos = {g: w for g, w in sig.items() if w > 0}
    neg = {g: w for g, w in sig.items() if w < 0}
    assert pos, "must have at least one positive weight"
    assert neg, "must have at least one negative weight"


# ---------------------------------------------------------------------------
# 2. All-up case: same guarantee
# ---------------------------------------------------------------------------

def test_all_up_genes_still_bidirectional():
    h5ad = {"GENEА": 0.3, "GENEB": 0.8, "GENEC": 1.5, "GENED": 0.5}
    with _patch_loadings():
        sig = _build_program_signature("PROG1", direction=1.0, h5ad_deg=h5ad)

    pos = {g: w for g, w in sig.items() if w > 0}
    neg = {g: w for g, w in sig.items() if w < 0}
    assert pos, "must have positive weights even when all genes up"
    assert neg, "must have negative weights even when all genes up"


# ---------------------------------------------------------------------------
# 3. Risk vs protective: direction flips the whole signature
# ---------------------------------------------------------------------------

def test_direction_flips_signature():
    h5ad = {"GENEА": -0.5, "GENEB": -1.0, "GENEC": -2.0, "GENED": -0.3}
    with _patch_loadings():
        sig_risk      = _build_program_signature("PROG1", direction=+1.0, h5ad_deg=h5ad)
        sig_protective = _build_program_signature("PROG1", direction=-1.0, h5ad_deg=h5ad)

    for gene in sig_risk:
        if gene in sig_protective:
            assert sig_risk[gene] == pytest.approx(-sig_protective[gene]), (
                f"{gene}: risk and protective weights should be negatives of each other"
            )


# ---------------------------------------------------------------------------
# 4. Gene above mean gets + weight for risk program (direction > 0)
# ---------------------------------------------------------------------------

def test_above_mean_gene_gets_positive_for_risk():
    # mean_lfc = (-0.5 + -1.0 + -2.0 + -0.3) / 4 = -0.95
    # GENEА (-0.5) is above mean → relative = +0.45 → direction(+1) × nmf(1.0) × 0.45 > 0
    h5ad = {"GENEА": -0.5, "GENEB": -1.0, "GENEC": -2.0, "GENED": -0.3}
    with _patch_loadings():
        sig = _build_program_signature("PROG1", direction=1.0, h5ad_deg=h5ad)
    assert sig["GENEА"] > 0, "gene above program mean should get + weight for risk program"
    assert sig["GENEC"] < 0, "gene below program mean should get - weight for risk program"


# ---------------------------------------------------------------------------
# 5. Genes absent from h5ad are excluded
# ---------------------------------------------------------------------------

def test_genes_absent_from_h5ad_excluded():
    h5ad = {"GENEА": -0.5, "GENEB": -1.0}  # GENEC and GENED absent
    with _patch_loadings():
        sig = _build_program_signature("PROG1", direction=1.0, h5ad_deg=h5ad)
    assert "GENEC" not in sig
    assert "GENED" not in sig


# ---------------------------------------------------------------------------
# 6. No h5ad data → unsigned NMF fallback (all same sign, direction applied)
# ---------------------------------------------------------------------------

def test_no_h5ad_falls_back_to_nmf():
    with _patch_loadings():
        sig = _build_program_signature("PROG1", direction=1.0, h5ad_deg=None)
    assert all(w > 0 for w in sig.values()), "NMF fallback: all weights positive for risk program"

    with _patch_loadings():
        sig_prot = _build_program_signature("PROG1", direction=-1.0, h5ad_deg=None)
    assert all(w < 0 for w in sig_prot.values()), "NMF fallback: all weights negative for protective"


# ---------------------------------------------------------------------------
# 7. No h5ad coverage for any program gene → NMF fallback
# ---------------------------------------------------------------------------

def test_no_h5ad_coverage_for_program_genes_falls_back():
    h5ad = {"UNRELATED_GENE": 1.0}  # no overlap with NMF genes
    with _patch_loadings():
        sig = _build_program_signature("PROG1", direction=1.0, h5ad_deg=h5ad)
    assert all(w > 0 for w in sig.values()), "should fall back to unsigned NMF when no h5ad overlap"


# ---------------------------------------------------------------------------
# 8. Empty NMF info → empty signature
# ---------------------------------------------------------------------------

def test_empty_nmf_returns_empty():
    with _patch_loadings({"top_genes": [], "gene_loadings": {}}):
        sig = _build_program_signature("PROG1", direction=1.0, h5ad_deg={"A": 1.0})
    assert sig == {}
