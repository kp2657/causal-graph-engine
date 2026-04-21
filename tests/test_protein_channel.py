"""
tests/test_protein_channel.py — Tier2pt ProteinChannel + Tier2s program_gamma fix.

Covers:
  1. estimate_beta_tier2pt() — fires with GWAS + pQTL
  2. estimate_beta_tier2pt() — fires with GWAS only (no pQTL)
  3. estimate_beta_tier2pt() — silent for non-GWAS genes
  4. estimate_beta_tier2pt() — silent for weak GWAS (p > 1e-5)
  5. estimate_beta_tier2pt() — correct beta priority (pQTL > GWAS > OT)
  6. compute_ota_gamma — program_gamma fallback (Tier2s/Tier2pt fix)
  7. compute_ota_gamma — fallback does not fire when gamma_estimates has entry
  8. estimate_beta() chain — Tier2pt fires after Tier2s misses, before Tier3
  9. CFH AMD scenario — full chain, protein channel gives meaningful OTA
 10. Mechanism confidence threshold — no fire when confidence < 0.15
"""
from __future__ import annotations
import math
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gwas_inst(pvalue: float = 1e-20, beta: float = 0.4) -> dict:
    return {
        "instruments": [
            {
                "study_type":      "gwas",
                "instrument_type": "gwas_credset",
                "p_value":         pvalue,
                "beta":            beta,
                "score":           0.8,
            }
        ]
    }


def _pqtl(pvalue: float = 0.019, beta: float = 0.45) -> dict:
    return {"pvalue": pvalue, "beta": beta, "source": "Sun_2018"}


# ---------------------------------------------------------------------------
# 1. Fires with GWAS + pQTL
# ---------------------------------------------------------------------------
def test_tier2pt_fires_gwas_pqtl():
    from pipelines.ota_beta_estimation import estimate_beta_tier2pt

    result = estimate_beta_tier2pt(
        gene="CFH",
        program="ribosome_biogenesis",
        ot_instruments=_gwas_inst(1e-50, 0.6),
        pqtl_data=_pqtl(0.019, 0.45),
        ot_score=0.82,
    )
    assert result is not None
    assert result["evidence_tier"] == "Tier2pt_ProteinChannel"
    assert result["beta"] == pytest.approx(0.45, abs=0.01)   # pQTL beta wins
    assert 0 < result["program_gamma"] <= 0.85
    assert result["beta_sigma"] == pytest.approx(0.35, abs=0.01)


# ---------------------------------------------------------------------------
# 2. Fires with GWAS only (no pQTL) — lower mechanism confidence
# ---------------------------------------------------------------------------
def test_tier2pt_fires_gwas_only():
    from pipelines.ota_beta_estimation import estimate_beta_tier2pt

    result = estimate_beta_tier2pt(
        gene="HMCN1",
        program="cell_cycle",
        ot_instruments=_gwas_inst(1e-12, 0.3),
        pqtl_data=None,
        ot_score=0.55,
    )
    assert result is not None
    assert result["evidence_tier"] == "Tier2pt_ProteinChannel"
    assert result["beta"] == pytest.approx(0.3, abs=0.01)  # GWAS beta
    # mechanism_conf = 0.55 * 0.5 = 0.275; must be above threshold
    assert result["program_gamma"] > 0


# ---------------------------------------------------------------------------
# 3. Silent for non-GWAS gene (no OT instruments)
# ---------------------------------------------------------------------------
def test_tier2pt_silent_no_instruments():
    from pipelines.ota_beta_estimation import estimate_beta_tier2pt

    result = estimate_beta_tier2pt(
        gene="RPL19",
        program="ribosome",
        ot_instruments=None,
        pqtl_data=None,
        ot_score=0.0,
    )
    assert result is None


def test_tier2pt_fires_on_high_l2g_without_credsets():
    """L2G fallback: no GWAS credsets but ot_score >= 0.5 fires the channel.

    Rescues complement genes (e.g. CFH L2G=0.86) whose loci are too complex
    for clean credible-set attribution.
    """
    from pipelines.ota_beta_estimation import estimate_beta_tier2pt

    result = estimate_beta_tier2pt(
        gene="CFH",
        program="__protein_channel__",
        ot_instruments={"instruments": []},   # empty credset list
        pqtl_data=None,
        ot_score=0.86,
    )
    assert result is not None
    assert result["evidence_tier"] == "Tier2pt_ProteinChannel"
    assert result["program_gamma"] > 0
    # beta falls back to OT score when no pQTL/GWAS beta
    assert abs(result["beta"]) > 0


def test_tier2pt_silent_low_l2g_without_credsets():
    """L2G fallback does NOT fire for low-confidence genes."""
    from pipelines.ota_beta_estimation import estimate_beta_tier2pt

    result = estimate_beta_tier2pt(
        gene="RANDOM",
        program="prog",
        ot_instruments={"instruments": []},
        pqtl_data=None,
        ot_score=0.3,   # below 0.5 threshold
    )
    assert result is None


# ---------------------------------------------------------------------------
# 4. Silent for weak GWAS (p > 1e-5)
# ---------------------------------------------------------------------------
def test_tier2pt_silent_weak_gwas():
    from pipelines.ota_beta_estimation import estimate_beta_tier2pt

    result = estimate_beta_tier2pt(
        gene="GENE1",
        program="prog",
        ot_instruments=_gwas_inst(1e-4, 0.2),  # just below threshold
        pqtl_data=_pqtl(0.01, 0.3),
        ot_score=0.7,
    )
    assert result is None


# ---------------------------------------------------------------------------
# 5. Beta priority: pQTL > GWAS beta > OT score
# ---------------------------------------------------------------------------
def test_tier2pt_beta_priority_pqtl_over_gwas():
    from pipelines.ota_beta_estimation import estimate_beta_tier2pt

    result = estimate_beta_tier2pt(
        gene="CFH",
        program="prog",
        ot_instruments=_gwas_inst(1e-30, beta=0.9),  # large GWAS beta
        pqtl_data=_pqtl(0.005, beta=-0.3),           # negative pQTL wins
        ot_score=0.8,
    )
    assert result is not None
    assert result["beta"] == pytest.approx(-0.3, abs=0.01)


def test_tier2pt_beta_falls_back_to_ot_score():
    from pipelines.ota_beta_estimation import estimate_beta_tier2pt

    inst = {"instruments": [{"study_type": "gwas", "instrument_type": "gwas_credset",
                              "p_value": 1e-20, "score": 0.7}]}  # no beta field
    result = estimate_beta_tier2pt(
        gene="C3",
        program="prog",
        ot_instruments=inst,
        pqtl_data=None,
        ot_score=0.65,
    )
    assert result is not None
    assert result["beta"] == pytest.approx(0.65, abs=0.02)


# ---------------------------------------------------------------------------
# 6. compute_ota_gamma uses embedded program_gamma (Tier2s/Tier2pt fix)
# ---------------------------------------------------------------------------
def test_ota_gamma_uses_embedded_program_gamma():
    """
    When gamma_estimates has no entry for a program but beta_info has
    program_gamma, compute_ota_gamma must use it as fallback.
    """
    from pipelines.ota_gamma_estimation import compute_ota_gamma

    beta_estimates = {
        "ribosome_biogenesis": {
            "beta": 0.45,
            "evidence_tier": "Tier2pt_ProteinChannel",
            "program_gamma": 0.70,
        }
    }
    gamma_estimates = {}  # no entry for ribosome_biogenesis

    result = compute_ota_gamma("CFH", "AMD", beta_estimates, gamma_estimates)
    assert result["ota_gamma"] == pytest.approx(0.45 * 0.70, abs=0.001)
    assert result["n_programs_contributing"] == 1


# ---------------------------------------------------------------------------
# 7. Falls back to gamma_estimates entry when present (no double-counting)
# ---------------------------------------------------------------------------
def test_ota_gamma_prefers_gamma_estimates_when_present():
    from pipelines.ota_gamma_estimation import compute_ota_gamma

    beta_estimates = {
        "cell_cycle": {
            "beta": 0.5,
            "evidence_tier": "Tier1_Interventional",
            "program_gamma": 0.9,  # this should be IGNORED
        }
    }
    gamma_estimates = {
        "cell_cycle": {"gamma": 0.3, "evidence_tier": "Tier1_Interventional"}
    }

    result = compute_ota_gamma("GENE", "AMD", beta_estimates, gamma_estimates)
    # Should use gamma_estimates value (0.3), not embedded program_gamma (0.9)
    assert result["ota_gamma"] == pytest.approx(0.5 * 0.3, abs=0.001)


# ---------------------------------------------------------------------------
# 8. estimate_beta_tier2pt called directly — returns correct structure
# ---------------------------------------------------------------------------
def test_tier2pt_direct_call_structure():
    """estimate_beta_tier2pt returns correct fields for use as virtual program."""
    from pipelines.ota_beta_estimation import estimate_beta_tier2pt

    result = estimate_beta_tier2pt(
        gene="NOVEL_PQTL_GENE",
        program="__protein_channel__",
        ot_instruments=_gwas_inst(1e-15, 0.3),
        pqtl_data=_pqtl(0.03, 0.25),
        ot_score=0.6,
    )
    assert result is not None
    assert result["evidence_tier"] == "Tier2pt_ProteinChannel"
    assert result["program_gamma"] > 0
    assert "beta" in result
    assert "beta_sigma" in result


# ---------------------------------------------------------------------------
# 9. CFH AMD scenario — protein channel virtual slot gives meaningful OTA
# ---------------------------------------------------------------------------
def test_cfh_amd_protein_channel_ota():
    """
    Simulate the agent adding __protein_channel__ to beta_matrix for CFH.
    compute_ota_gamma must use the embedded program_gamma.
    """
    from pipelines.ota_beta_estimation import estimate_beta_tier2pt
    from pipelines.ota_gamma_estimation import compute_ota_gamma

    # CFH: strong GWAS (p=1e-50), pQTL p=0.019, OT score=0.82
    pt_result = estimate_beta_tier2pt(
        gene="CFH",
        program="__protein_channel__",
        ot_instruments=_gwas_inst(1e-50, 0.6),
        pqtl_data=_pqtl(0.019, 0.45),
        ot_score=0.82,
    )
    assert pt_result is not None
    assert pt_result["program_gamma"] > 0.3   # meaningful gamma

    # Simulate beta_matrix after agent loop (existing programs near-zero + channel)
    beta_estimates = {
        "ribosome_biogenesis": {
            "beta": 0.0, "evidence_tier": "provisional_virtual", "program_gamma": None
        },
        "__protein_channel__": {
            "beta":          pt_result["beta"],
            "evidence_tier": pt_result["evidence_tier"],
            "program_gamma": pt_result["program_gamma"],
        },
    }
    # gamma_estimates for normal program is near-zero (AMD doesn't enrich ribosome)
    gamma_estimates = {
        "ribosome_biogenesis": {"gamma": 0.01, "evidence_tier": "Tier3_Provisional"}
    }
    ota = compute_ota_gamma("CFH", "AMD", beta_estimates, gamma_estimates)
    # OTA should be substantially above Run9 value of 0.013
    assert ota["ota_gamma"] > 0.10, f"Expected >0.10, got {ota['ota_gamma']}"


# ---------------------------------------------------------------------------
# 10. Mechanism confidence threshold — no fire when confidence < 0.15
# ---------------------------------------------------------------------------
def test_tier2pt_no_fire_low_confidence():
    from pipelines.ota_beta_estimation import estimate_beta_tier2pt

    # ot_score=0.2 → mechanism_conf = 0.2*0.5 = 0.10 < 0.15
    result = estimate_beta_tier2pt(
        gene="LOWCONF",
        program="prog",
        ot_instruments=_gwas_inst(1e-8, 0.2),
        pqtl_data=None,
        ot_score=0.20,
    )
    assert result is None


# ---------------------------------------------------------------------------
# 11. Tier2s program_gamma also stored correctly in beta_matrix entry
# ---------------------------------------------------------------------------
def test_tier2s_program_gamma_survives_beta_matrix_storage():
    """
    estimate_beta_tier2_synthetic returns program_gamma; the agent stores it.
    Simulate the storage step and verify OTA can use it.
    """
    from pipelines.ota_gamma_estimation import compute_ota_gamma

    # Simulate what perturbation_genomics_agent stores after Tier2s fires
    beta_estimates = {
        "ribosome_biogenesis": {
            "beta":          0.79,
            "evidence_tier": "Tier2s_SyntheticPathway",
            "beta_sigma":    0.45,
            "program_gamma": 0.599,  # from complement synthetic program
        }
    }
    gamma_estimates = {}  # no Perturb-seq gamma for ribosome_biogenesis in AMD

    ota = compute_ota_gamma("CFH", "AMD", beta_estimates, gamma_estimates)
    expected = 0.79 * 0.599
    assert ota["ota_gamma"] == pytest.approx(expected, abs=0.01)
    assert ota["ota_gamma"] > 0.40   # substantially above old value of 0.013
