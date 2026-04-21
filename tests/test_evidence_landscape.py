"""
tests/test_evidence_landscape.py — Unit tests for evidence landscape builder.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from pipelines.evidence_landscape import (
    build_evidence_landscape,
    summarize_landscape,
    _evidence_class,
    _genetic_track,
    _perturbseq_track,
    _functional_track,
    _translational_track,
    _literature_track,
    _counterfactual_track,
    _adversarial_track,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_target(**kwargs) -> dict:
    defaults = {
        "target_gene": "TEST1",
        "rank": 1,
        "target_score": 0.5,
        "ota_gamma": 0.4,
        "ota_gamma_raw": 0.5,
        "ota_gamma_sigma": 0.1,
        "ota_gamma_ci_lower": 0.2,
        "ota_gamma_ci_upper": 0.8,
        "scone_confidence": 0.9,
        "scone_flags": [],
        "evidence_tier": "Tier2_Convergent",
        "ot_score": 0.7,
        "max_phase": 2,
        "known_drugs": ["DRUG_A"],
        "pli": 1e-10,
        "flags": ["genetic_anchor"],
        "top_programs": ["inflam_prog"],
        "key_evidence": ["GWAS hit"],
    }
    defaults.update(kwargs)
    return defaults


def _make_pipeline_outputs(targets: list[dict], **kwargs) -> dict:
    return {
        "graph_output": {"target_list": targets},
        "literature_result": kwargs.get("literature_result", {}),
        "red_team_result": kwargs.get("red_team_result", {}),
        "chemistry_result": kwargs.get("chemistry_result", {}),
        "trials_result": kwargs.get("trials_result", {}),
        "upstream_regulator_evidence": kwargs.get("upstream_regulator_evidence", {}),
    }


# ---------------------------------------------------------------------------
# _evidence_class
# ---------------------------------------------------------------------------

class TestEvidenceClass:
    def test_genetic_anchor(self):
        t = _make_target(evidence_tier="Tier2_Convergent", ota_gamma=0.4)
        assert _evidence_class(t, set()) == "genetic_anchor"

    def test_convergent_with_perturb(self):
        t = _make_target(target_gene="SPI1", evidence_tier="Tier2_Convergent", ota_gamma=0.4)
        assert _evidence_class(t, {"SPI1"}) == "convergent"

    def test_perturb_seq_regulator_no_genetic(self):
        t = _make_target(target_gene="SPI1", evidence_tier="state_nominated", ota_gamma=0.0)
        assert _evidence_class(t, {"SPI1"}) == "perturb_seq_regulator"

    def test_state_nominated(self):
        t = _make_target(evidence_tier="state_nominated", ota_gamma=0.0)
        assert _evidence_class(t, set()) == "state_nominated"

    def test_gwas_provisional(self):
        t = _make_target(evidence_tier="Tier3_Provisional", ota_gamma=0.2)
        assert _evidence_class(t, set()) == "gwas_provisional"

    def test_tier1_genetic_anchor(self):
        t = _make_target(evidence_tier="Tier1_Interventional", ota_gamma=0.6)
        assert _evidence_class(t, set()) == "genetic_anchor"

    def test_zero_gamma_tier2_is_state_nominated(self):
        # Tier2 with ota_gamma=0 shouldn't be classified as genetic_anchor
        t = _make_target(evidence_tier="Tier2_Convergent", ota_gamma=0.0)
        assert _evidence_class(t, set()) == "state_nominated"


# ---------------------------------------------------------------------------
# _genetic_track
# ---------------------------------------------------------------------------

class TestGeneticTrack:
    def test_ci_packed(self):
        t = _make_target()
        gt = _genetic_track(t)
        assert gt["ota_gamma_ci"] == [0.2, 0.8]

    def test_no_ci_when_null(self):
        t = _make_target(ota_gamma_ci_lower=None, ota_gamma_ci_upper=None)
        gt = _genetic_track(t)
        assert gt["ota_gamma_ci"] is None

    def test_all_fields_present(self):
        t = _make_target()
        gt = _genetic_track(t)
        for k in ("ota_gamma", "ota_gamma_raw", "ota_gamma_sigma", "scone_confidence",
                  "scone_flags", "evidence_tier"):
            assert k in gt


# ---------------------------------------------------------------------------
# _perturbseq_track
# ---------------------------------------------------------------------------

class TestPerturbseqTrack:
    def test_is_upstream_regulator(self):
        t = _make_target(target_gene="SPI1", top_programs=["prog1", "prog2"])
        pt = _perturbseq_track("SPI1", {"SPI1", "SMAD4"}, t)
        assert pt["is_upstream_regulator"] is True
        assert pt["n_programs"] == 2

    def test_not_upstream_regulator(self):
        t = _make_target(target_gene="NOD2", top_programs=[])
        pt = _perturbseq_track("NOD2", {"SPI1"}, t)
        assert pt["is_upstream_regulator"] is False
        assert pt["n_programs"] == 0


# ---------------------------------------------------------------------------
# _functional_track
# ---------------------------------------------------------------------------

class TestFunctionalTrack:
    def test_flags_parsed(self):
        t = _make_target(flags=["bimodal_expression", "escape_risk"])
        ft = _functional_track(t)
        assert ft["bimodal_expression"] is True
        assert ft["escape_risk"] is True
        assert ft["chip_mechanism"] is False

    def test_empty_flags(self):
        t = _make_target(flags=[])
        ft = _functional_track(t)
        assert ft["bimodal_expression"] is False
        assert ft["escape_risk"] is False


# ---------------------------------------------------------------------------
# _translational_track
# ---------------------------------------------------------------------------

class TestTranslationalTrack:
    def test_basic_fields(self):
        t = _make_target(max_phase=4, known_drugs=["DRUG_A"], ot_score=0.87)
        chem = {"tractability": "small_molecule", "best_ic50_nM": 50.0}
        tt = _translational_track(t, chem, {})
        assert tt["max_phase"] == 4
        assert tt["known_drugs"] == ["DRUG_A"]
        assert tt["tractability"] == "small_molecule"
        assert tt["best_ic50_nM"] == 50.0
        assert tt["ot_score"] == 0.87

    def test_empty_chem(self):
        t = _make_target()
        tt = _translational_track(t, {}, {})
        assert tt["tractability"] is None
        assert tt["best_ic50_nM"] is None


# ---------------------------------------------------------------------------
# _literature_track
# ---------------------------------------------------------------------------

class TestLiteratureTrack:
    def test_gene_found(self):
        lit = {
            "NOD2": {
                "n_papers_found": 655,
                "n_supporting": 10,
                "n_contradicting": 0,
                "literature_confidence": "SUPPORTED",
                "recency_score": 1.0,
                "key_citations": [
                    {"pmid": "123", "title": "A study", "year": "2025"}
                ],
                "search_query": '"NOD2" AND "IBD"',
            }
        }
        track = _literature_track("NOD2", lit)
        assert track["n_papers_found"] == 655
        assert track["literature_confidence"] == "SUPPORTED"
        assert len(track["key_citations"]) == 1
        assert track["key_citations"][0]["pmid"] == "123"

    def test_gene_not_found_returns_zeros(self):
        track = _literature_track("UNKNOWN_GENE", {})
        assert track["n_papers_found"] == 0
        assert track["literature_confidence"] is None

    def test_citation_title_truncated(self):
        long_title = "A" * 200
        lit = {
            "GENE1": {
                "n_papers_found": 1,
                "key_citations": [{"pmid": "1", "title": long_title, "year": "2025"}],
            }
        }
        track = _literature_track("GENE1", lit)
        assert len(track["key_citations"][0]["title"]) <= 100

    def test_max_3_citations(self):
        citations = [{"pmid": str(i), "title": f"paper {i}", "year": "2025"} for i in range(10)]
        lit = {"GENE1": {"n_papers_found": 10, "key_citations": citations}}
        track = _literature_track("GENE1", lit)
        assert len(track["key_citations"]) == 3


# ---------------------------------------------------------------------------
# _counterfactual_track
# ---------------------------------------------------------------------------

class TestCounterfactualTrack:
    def test_none_when_no_assessment(self):
        assert _counterfactual_track(None) is None

    def test_none_when_no_cf_field(self):
        assert _counterfactual_track({"red_team_verdict": "PROCEED"}) is None

    def test_compacted_fields(self):
        rt = {
            "counterfactual": {
                "primary_trait": "IBD",
                "inhibition_50pct": {
                    "baseline_gamma": 0.5,
                    "perturbed_gamma": 0.25,
                    "percent_change": -50.0,
                    "ci_lower": 0.17,
                    "ci_upper": 0.34,
                    "uncertainty_note": "Bootstrap CI",
                    "dominant_program": "prog1",
                    "interpretation": "50% inhibition...",
                },
                "knockout": {
                    "baseline_gamma": 0.5,
                    "perturbed_gamma": 0.0,
                    "percent_change": -100.0,
                    "ci_lower": 0.0,
                    "ci_upper": 0.0,
                    "uncertainty_note": "KO",
                    "dominant_program": None,
                    "interpretation": "Knockout...",
                },
            }
        }
        cf = _counterfactual_track(rt)
        assert cf["primary_trait"] == "IBD"
        assert cf["inhibition_50pct"]["percent_change"] == -50.0
        assert cf["inhibition_50pct"]["dominant_program"] == "prog1"
        assert cf["knockout"]["percent_change"] == -100.0


# ---------------------------------------------------------------------------
# _adversarial_track
# ---------------------------------------------------------------------------

class TestAdversarialTrack:
    def test_none_when_no_assessment(self):
        assert _adversarial_track(None) is None

    def test_fields(self):
        rt = {
            "red_team_verdict": "PROCEED",
            "confidence_level": "HIGH",
            "rank_stability": "STABLE",
            "rank_stability_rationale": "Tier2 evidence",
            "counterargument": "Some caveat",
            "evidence_vulnerability": "Wide CI",
            "literature_flag": None,
        }
        adv = _adversarial_track(rt)
        assert adv["verdict"] == "PROCEED"
        assert adv["confidence_level"] == "HIGH"
        assert adv["rank_stability"] == "STABLE"
        assert adv["counterargument"] == "Some caveat"


# ---------------------------------------------------------------------------
# build_evidence_landscape
# ---------------------------------------------------------------------------

class TestBuildEvidenceLandscape:
    def test_one_gene_profile(self):
        t = _make_target(target_gene="NOD2", evidence_tier="Tier2_Convergent", ota_gamma=0.4)
        po = _make_pipeline_outputs([t])
        profiles = build_evidence_landscape(po)
        assert len(profiles) == 1
        p = profiles[0]
        assert p["gene"] == "NOD2"
        assert p["evidence_class"] == "genetic_anchor"
        assert "genetic_evidence" in p
        assert "perturbseq_evidence" in p
        assert "functional_evidence" in p
        assert "translational_evidence" in p
        assert "literature_evidence" in p
        assert "counterfactual_evidence" in p
        assert "adversarial_assessment" in p

    def test_convergent_when_perturb_regulator(self):
        t = _make_target(target_gene="SPI1", evidence_tier="Tier2_Convergent", ota_gamma=0.3)
        po = _make_pipeline_outputs([t], upstream_regulator_evidence={"SPI1": {"source": "papalexi"}})
        profiles = build_evidence_landscape(po)
        assert profiles[0]["evidence_class"] == "convergent"
        assert profiles[0]["perturbseq_evidence"]["is_upstream_regulator"] is True

    def test_state_nominated_class(self):
        t = _make_target(target_gene="RNASE1", evidence_tier="state_nominated", ota_gamma=0.0)
        po = _make_pipeline_outputs([t])
        profiles = build_evidence_landscape(po)
        assert profiles[0]["evidence_class"] == "state_nominated"

    def test_literature_evidence_merged(self):
        t = _make_target(target_gene="NOD2")
        lit_ev = {
            "NOD2": {
                "n_papers_found": 100,
                "n_supporting": 5,
                "n_contradicting": 0,
                "literature_confidence": "SUPPORTED",
                "recency_score": 0.9,
                "key_citations": [],
            }
        }
        po = _make_pipeline_outputs([t], literature_result={"literature_evidence": lit_ev})
        profiles = build_evidence_landscape(po)
        assert profiles[0]["literature_evidence"]["n_papers_found"] == 100
        assert profiles[0]["literature_evidence"]["literature_confidence"] == "SUPPORTED"

    def test_tau_from_prioritization_result(self):
        """tau_disease fields should be read from prioritization_result TR data, not stripped target_list."""
        t = _make_target(target_gene="STAT1")
        po = {
            "graph_output": {"target_list": [t]},
            "prioritization_result": {
                "targets": [{
                    **t,
                    "therapeutic_redirection_result": {
                        "tau_disease_specificity": 0.72,
                        "disease_log2fc": 1.5,
                        "tau_specificity_class": "disease_specific",
                    }
                }]
            },
        }
        profiles = build_evidence_landscape(po)
        fe = profiles[0]["functional_evidence"]
        assert fe["tau_disease"] == 0.72
        assert fe["disease_log2fc"] == 1.5
        assert fe["tau_specificity_class"] == "disease_specific"

    def test_red_team_merged(self):
        t = _make_target(target_gene="NOD2")
        rt_result = {
            "red_team_assessments": [{
                "target_gene": "NOD2",
                "red_team_verdict": "PROCEED",
                "confidence_level": "HIGH",
                "rank_stability": "STABLE",
                "rank_stability_rationale": "Tier2",
                "counterargument": "no major caveat",
                "evidence_vulnerability": "Wide CI",
                "literature_flag": None,
                "counterfactual": None,
            }]
        }
        po = _make_pipeline_outputs([t], red_team_result=rt_result)
        profiles = build_evidence_landscape(po)
        adv = profiles[0]["adversarial_assessment"]
        assert adv["verdict"] == "PROCEED"
        assert adv["confidence_level"] == "HIGH"

    def test_empty_targets_returns_empty_list(self):
        po = _make_pipeline_outputs([])
        profiles = build_evidence_landscape(po)
        assert profiles == []

    def test_rank_preserved_as_internal(self):
        targets = [
            _make_target(target_gene="A", rank=1),
            _make_target(target_gene="B", rank=2),
        ]
        po = _make_pipeline_outputs(targets)
        profiles = build_evidence_landscape(po)
        assert profiles[0]["_rank"] == 1
        assert profiles[1]["_rank"] == 2

    def test_missing_graph_output_returns_empty(self):
        profiles = build_evidence_landscape({})
        assert profiles == []

    def test_regulator_nomination_evidence_key_fallback(self):
        """upstream_regulator_evidence should also read regulator_nomination_evidence key"""
        t = _make_target(target_gene="SPI1", evidence_tier="Tier2_Convergent", ota_gamma=0.3)
        po = {
            "graph_output": {"target_list": [t]},
            "regulator_nomination_evidence": {"SPI1": {"source": "papalexi"}},
        }
        profiles = build_evidence_landscape(po)
        assert profiles[0]["perturbseq_evidence"]["is_upstream_regulator"] is True

    def test_regulator_evidence_regulators_list_shape(self):
        """Handle {'regulators': [{gene: 'SPI1', ...}]} shape from perturb-seq server."""
        t = _make_target(target_gene="SPI1", evidence_tier="Tier2_Convergent", ota_gamma=0.3)
        po = {
            "graph_output": {"target_list": [t]},
            "upstream_regulator_evidence": {
                "regulators": [{"gene": "SPI1", "n_targets_regulated": 4, "dataset_id": "papalexi_2021_thp1"}]
            },
        }
        profiles = build_evidence_landscape(po)
        assert profiles[0]["perturbseq_evidence"]["is_upstream_regulator"] is True
        assert profiles[0]["evidence_class"] == "convergent"

    def test_perturb_seq_regulator_no_genetic_from_list_shape(self):
        """state_nominated gene that is a Perturb-seq regulator → perturb_seq_regulator class."""
        t = _make_target(target_gene="SMAD4", evidence_tier="state_nominated", ota_gamma=0.0)
        po = {
            "graph_output": {"target_list": [t]},
            "upstream_regulator_evidence": {
                "regulators": [{"gene": "SMAD4", "n_targets_regulated": 3}]
            },
        }
        profiles = build_evidence_landscape(po)
        assert profiles[0]["evidence_class"] == "perturb_seq_regulator"


# ---------------------------------------------------------------------------
# summarize_landscape
# ---------------------------------------------------------------------------

class TestSummarizeLandscape:
    def _make_profile(self, **kwargs) -> dict:
        defaults = {
            "gene": "X",
            "evidence_class": "genetic_anchor",
            "_rank": 1,
            "_target_score": 0.5,
            "genetic_evidence": {"ota_gamma": 0.4},
            "perturbseq_evidence": {"is_upstream_regulator": False},
            "functional_evidence": {},
            "translational_evidence": {},
            "literature_evidence": {"literature_confidence": "SUPPORTED"},
            "counterfactual_evidence": {"primary_trait": "IBD"},
            "adversarial_assessment": {"verdict": "PROCEED"},
        }
        defaults.update(kwargs)
        return defaults

    def test_basic_counts(self):
        profiles = [
            self._make_profile(gene="A", evidence_class="genetic_anchor"),
            self._make_profile(gene="B", evidence_class="convergent"),
            self._make_profile(gene="C", evidence_class="state_nominated",
                               genetic_evidence={"ota_gamma": 0.0},
                               perturbseq_evidence={"is_upstream_regulator": False},
                               literature_evidence={"literature_confidence": "NOVEL"},
                               counterfactual_evidence=None,
                               adversarial_assessment={"verdict": "DEPRIORITIZE"}),
        ]
        s = summarize_landscape(profiles)
        assert s["n_genes"] == 3
        assert s["by_class"]["genetic_anchor"] == 1
        assert s["by_class"]["convergent"] == 1
        assert s["by_class"]["state_nominated"] == 1
        assert s["n_with_genetic_instrument"] == 2  # A and B have gamma != 0
        assert s["n_with_literature_support"] == 2  # A and B are SUPPORTED
        assert s["n_with_counterfactual"] == 2
        assert s["n_proceed"] == 2
        assert s["n_deprioritize"] == 1
        assert s["n_caution"] == 0

    def test_empty(self):
        s = summarize_landscape([])
        assert s["n_genes"] == 0
        assert s["by_class"] == {}
        assert s["n_proceed"] == 0
