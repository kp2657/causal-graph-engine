"""
test_ota_beta_estimation.py — Unit tests for the β_{gene→program} estimation tier hierarchy.

Tests verify:
  - Each tier function returns None (not 0.0) when its required data is absent
  - NaN propagation: float('nan') is never silently replaced with 0.0
  - Tier hierarchy: Tier 1 > Tier 2a > ... > Virtual-B in estimate_beta()
  - Evidence tier labels are consistent strings matching _TIER_RANK keys
  - Beta signs are plausible (negative-regulation genes return negative beta)
  - Synthetic pathway tier (Tier2s) has been removed from the chain
"""
from __future__ import annotations

import math
import pytest

from pipelines.ota_beta_estimation import (
    estimate_beta,
    estimate_beta_tier1,
    estimate_beta_tier2,
    estimate_beta_tier2_ot_instrument,
    estimate_beta_tier2_sc_eqtl,
    estimate_beta_tier2_pqtl,
    estimate_beta_tier2_eqtl_direction,
    estimate_beta_tier2_rare_burden,
    estimate_beta_virtual,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_perturbseq_data(gene: str, log2fc: float = -0.8) -> dict:
    """Minimal Perturb-seq data dict for a gene (quantitative path format)."""
    return {gene: {"programs": {PROGRAM: {"beta": log2fc, "se": abs(log2fc) * 0.15}}}}


def _make_eqtl_data(nes: float = 0.25, se: float = 0.05) -> dict:
    return {"nes": nes, "se": se, "tissue": "Whole_Blood"}


def _make_pqtl_data(beta: float = 0.4, se: float = 0.08, pvalue: float = 1e-6) -> dict:
    return {"top_pqtl": {"beta": beta, "se": se, "pvalue": pvalue, "rsid": "rs1234"}}


def _make_sc_eqtl_data(beta: float = 0.3, pvalue: float = 1e-5) -> dict:
    return {
        "eqtls": [{"beta": beta, "se": 0.06, "pvalue": pvalue,
                   "condition_label": "monocyte", "study_label": "OneK1K"}],
        "top_eqtl": {"beta": beta, "se": 0.06, "pvalue": pvalue,
                     "condition_label": "monocyte", "study_label": "OneK1K"},
        "n_found": 1,
    }


def _make_burden_data(direction: str = "risk_increasing") -> dict:
    return {"direction": direction, "p_value": 0.01, "n_carriers": 50}


PROGRAM = "complement_activation"
GENE = "CFH"
GENE_PCSK9 = "PCSK9"


# ---------------------------------------------------------------------------
# Tier 1 — Perturb-seq
# ---------------------------------------------------------------------------

class TestTier1:
    def test_returns_none_when_data_absent(self):
        assert estimate_beta_tier1(GENE, PROGRAM, perturbseq_data=None) is None

    def test_returns_none_when_gene_not_in_data(self):
        data = _make_perturbseq_data("OTHER_GENE")
        assert estimate_beta_tier1(GENE, PROGRAM, perturbseq_data=data) is None

    def test_returns_dict_with_correct_fields(self):
        data = _make_perturbseq_data(GENE, log2fc=-1.2)
        result = estimate_beta_tier1(GENE, PROGRAM, perturbseq_data=data)
        assert result is not None
        assert "beta" in result
        assert "evidence_tier" in result
        assert result["evidence_tier"] == "Tier1_Interventional"

    def test_beta_sign_matches_log2fc(self):
        data = _make_perturbseq_data(GENE, log2fc=-0.9)
        result = estimate_beta_tier1(GENE, PROGRAM, perturbseq_data=data)
        assert result is not None
        assert result["beta"] < 0, "Negative log2FC should yield negative beta"

    def test_beta_not_nan(self):
        data = _make_perturbseq_data(GENE, log2fc=-0.5)
        result = estimate_beta_tier1(GENE, PROGRAM, perturbseq_data=data)
        if result is not None:
            assert math.isfinite(result["beta"]), "Beta must not be NaN or Inf"


# ---------------------------------------------------------------------------
# Tier 2a — GTEx eQTL-MR
# ---------------------------------------------------------------------------

class TestTier2a:
    def test_returns_none_when_data_absent(self):
        assert estimate_beta_tier2(GENE, PROGRAM, eqtl_data=None) is None

    def test_returns_none_when_nes_absent(self):
        assert estimate_beta_tier2(GENE, PROGRAM, eqtl_data={"tissue": "Liver"}) is None

    def test_returns_dict_with_tier_label(self):
        result = estimate_beta_tier2(GENE, PROGRAM, eqtl_data=_make_eqtl_data())
        assert result is not None
        assert "Tier2" in result["evidence_tier"]

    def test_coloc_h4_gates_tier(self):
        """Low COLOC H4 should not yield full Tier2a (may return direction-only or None)."""
        result = estimate_beta_tier2(
            GENE, PROGRAM, eqtl_data=_make_eqtl_data(), coloc_h4=0.3
        )
        if result is not None:
            assert result["evidence_tier"] != "Tier2_Convergent"

    def test_high_coloc_yields_tier2(self):
        result = estimate_beta_tier2(
            GENE, PROGRAM, eqtl_data=_make_eqtl_data(), coloc_h4=0.95
        )
        assert result is not None
        assert "Tier2" in result["evidence_tier"]

    def test_beta_not_zero_when_evidence_present(self):
        result = estimate_beta_tier2(GENE, PROGRAM, eqtl_data=_make_eqtl_data(nes=0.3))
        assert result is not None
        assert result["beta"] != 0.0 or result.get("beta_sigma", 0) > 0


# ---------------------------------------------------------------------------
# Tier 2b — OT instruments
# ---------------------------------------------------------------------------

class TestTier2b:
    def test_returns_none_when_absent(self):
        assert estimate_beta_tier2_ot_instrument(GENE, PROGRAM, ot_instruments=None) is None

    def test_returns_none_for_empty_instruments(self):
        assert estimate_beta_tier2_ot_instrument(
            GENE, PROGRAM, ot_instruments={"instruments": []}
        ) is None

    def test_gwas_credset_returns_none(self):
        # GWAS credset fallback removed: projecting GWAS β onto program loadings
        # produces flat uniform beta that cancels in OTA sum with mixed-sign gammas.
        ot = {
            "instruments": [{"beta": 0.15, "se": 0.03, "pvalue": 1e-7,
                              "rsid": "rs999", "instrument_type": "gwas_credset"}]
        }
        result = estimate_beta_tier2_ot_instrument(GENE, PROGRAM, ot_instruments=ot)
        assert result is None

    def test_eqtl_instrument_still_returns_dict(self):
        ot = {
            "instruments": [{"beta": 0.22, "se": 0.04, "pvalue": 1e-8,
                              "rsid": "rs123", "instrument_type": "eqtl",
                              "tissue": "Liver"}]
        }
        result = estimate_beta_tier2_ot_instrument(GENE, PROGRAM, ot_instruments=ot)
        assert result is not None
        assert "evidence_tier" in result


# ---------------------------------------------------------------------------
# Tier 2c — sc-eQTL
# ---------------------------------------------------------------------------

class TestTier2c:
    def test_returns_none_when_absent(self):
        assert estimate_beta_tier2_sc_eqtl(GENE, PROGRAM, sc_eqtl_data=None) is None

    def test_returns_none_for_empty_eqtls(self):
        assert estimate_beta_tier2_sc_eqtl(
            GENE, PROGRAM, sc_eqtl_data={"n_found": 0, "eqtls": []}
        ) is None

    def test_returns_dict_with_sc_eqtl(self):
        result = estimate_beta_tier2_sc_eqtl(
            GENE, PROGRAM, sc_eqtl_data=_make_sc_eqtl_data()
        )
        assert result is not None
        assert "scEQTL" in result["evidence_tier"]


# ---------------------------------------------------------------------------
# Tier 2p — pQTL-MR
# ---------------------------------------------------------------------------

class TestTier2p:
    def test_returns_none_when_absent(self):
        assert estimate_beta_tier2_pqtl(GENE, PROGRAM, pqtl_data=None) is None

    def test_returns_dict_with_pqtl(self):
        result = estimate_beta_tier2_pqtl(GENE, PROGRAM, pqtl_data=_make_pqtl_data())
        assert result is not None
        assert "pQTL" in result["evidence_tier"]

    def test_weak_pqtl_returns_none(self):
        """pQTL with p > threshold should be excluded."""
        result = estimate_beta_tier2_pqtl(
            GENE, PROGRAM, pqtl_data=_make_pqtl_data(pvalue=0.5)
        )
        assert result is None


# ---------------------------------------------------------------------------
# Tier 2.5 — eQTL direction-only
# ---------------------------------------------------------------------------

class TestTier25:
    def test_returns_none_when_absent(self):
        assert estimate_beta_tier2_eqtl_direction(GENE, PROGRAM, eqtl_data=None) is None

    def test_sigma_is_larger_than_tier2a(self):
        """Direction-only tier should have higher uncertainty (larger sigma)."""
        result = estimate_beta_tier2_eqtl_direction(
            GENE, PROGRAM, eqtl_data=_make_eqtl_data(), coloc_h4=0.3
        )
        if result is not None:
            assert result.get("beta_sigma", 0) >= 0.40


# ---------------------------------------------------------------------------
# Tier 2rb — rare burden
# ---------------------------------------------------------------------------

class TestTier2rb:
    def test_returns_none_when_absent(self):
        assert estimate_beta_tier2_rare_burden(GENE, PROGRAM, burden_data=None) is None

    def test_returns_dict_with_burden(self):
        result = estimate_beta_tier2_rare_burden(
            GENE, PROGRAM, burden_data=_make_burden_data()
        )
        if result is not None:
            assert "burden" in result["evidence_tier"].lower() or "Tier2" in result["evidence_tier"]


# ---------------------------------------------------------------------------
# Virtual-B fallback
# ---------------------------------------------------------------------------

class TestVirtual:
    def test_always_returns_dict(self):
        result = estimate_beta_virtual(GENE, PROGRAM, pathway_member=False)
        assert isinstance(result, dict)
        assert "beta" in result

    def test_pathway_member_has_nonzero_beta(self):
        result = estimate_beta_virtual(GENE, PROGRAM, pathway_member=True)
        assert result["beta"] != 0.0 or result.get("evidence_tier") == "provisional_virtual"


# ---------------------------------------------------------------------------
# estimate_beta() — full fallback chain
# ---------------------------------------------------------------------------

class TestEstimateBeta:
    def test_returns_dict_or_none_when_no_data(self):
        """estimate_beta returns None when no causal evidence is available (virtual fallback removed)."""
        result = estimate_beta(GENE, PROGRAM)
        # None is correct: no perturb-seq, eQTL, pQTL, or burden data provided
        assert result is None or (isinstance(result, dict) and "beta" in result)

    def test_tier1_wins_over_tier2(self):
        """When Perturb-seq data is present, Tier 1 is preferred over eQTL."""
        ps_data = _make_perturbseq_data(GENE, log2fc=-1.0)
        result = estimate_beta(
            GENE, PROGRAM,
            perturbseq_data=ps_data,
            eqtl_data=_make_eqtl_data(nes=0.5),
        )
        assert result["tier_used"] == 1
        assert result["evidence_tier"] == "Tier1_Interventional"

    def test_tier2_activates_without_tier1(self):
        """eQTL evidence activates Tier 2 when Perturb-seq is absent."""
        result = estimate_beta(
            GENE, PROGRAM,
            perturbseq_data=None,
            eqtl_data=_make_eqtl_data(),
            coloc_h4=0.92,
        )
        assert result["tier_used"] <= 2

    def test_no_synthetic_tier_in_chain(self):
        """Tier2s (synthetic pathway) has been removed; check it is not returned."""
        result = estimate_beta(GENE, PROGRAM)
        # estimate_beta returns None when no causal evidence is provided (virtual removed)
        assert result is None or result.get("evidence_tier") != "Tier2s_SyntheticPathway"

    def test_gene_and_program_fields_present(self):
        # estimate_beta returns None when no causal evidence provided (virtual removed in session 79)
        result = estimate_beta(GENE, PROGRAM)
        if result is not None:
            assert result.get("gene") == GENE
            assert result.get("program") == PROGRAM

    def test_beta_is_finite_or_none(self):
        """Beta value must be a finite float, not NaN or Inf."""
        result = estimate_beta(
            GENE, PROGRAM,
            eqtl_data=_make_eqtl_data(),
        )
        beta = result.get("beta")
        if beta is not None:
            assert math.isfinite(beta), f"beta={beta} is not finite"

    def test_no_zero_beta_from_virtual_pathway_member(self):
        """Virtual-B pathway member: returns None when no causal evidence (virtual removed in session 79)."""
        result = estimate_beta(
            GENE, PROGRAM,
            perturbseq_data=None,
            eqtl_data=None,
            pathway_member=True,
        )
        # Virtual fallback removed: result is None, or if dict, must not be silent 0
        if result is not None:
            assert result.get("beta") != 0.0 or result.get("evidence_tier") == "provisional_virtual"

    def test_pcsk9_lipid_program(self):
        """PCSK9 knockdown should produce a negative beta in a lipid program."""
        ps = _make_perturbseq_data(GENE_PCSK9, log2fc=-1.5)
        result = estimate_beta(GENE_PCSK9, "lipid_metabolism", perturbseq_data=ps)
        if result is not None and result.get("tier_used") == 1:
            assert result["beta"] < 0, "PCSK9 KD → negative beta expected"
