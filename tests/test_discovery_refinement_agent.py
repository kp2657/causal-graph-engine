"""
tests/test_discovery_refinement_agent.py — Unit tests for discovery_refinement_agent.

All tests are unit (no live API calls). Tests cover heuristic local mode classification
into opportunity buckets.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.discovery.discovery_refinement_agent import run, _local_run, OPPORTUNITY_CLASSES


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

DISEASE_QUERY = {
    "disease_name": "inflammatory bowel disease",
    "efo_id":       "EFO:0003767",
}


def _make_profile(
    gene: str,
    evidence_class: str = "state_nominated",
    ota_gamma: float = 0.0,
    tau_disease: float | None = None,
    tau_specificity_class: str | None = None,
    tau_log2fc: float | None = None,
    lit_confidence: str = "MODERATE",
    rt_verdict: str = "PROCEED",
    max_phase: int = 0,
    known_drugs: list | None = None,
    is_upstream_regulator: bool = False,
    n_programs: int = 0,
) -> dict:
    return {
        "gene": gene,
        "evidence_class": evidence_class,
        "genetic_evidence": {"ota_gamma": ota_gamma},
        "functional_evidence": {
            "tau_disease": tau_disease,
            "tau_specificity_class": tau_specificity_class,
            "tau_log2fc": tau_log2fc,
        },
        "literature_evidence": {"literature_confidence": lit_confidence},
        "adversarial_assessment": {"verdict": rt_verdict},
        "translational_evidence": {"max_phase": max_phase, "known_drugs": known_drugs or []},
        "perturbseq_evidence": {"is_upstream_regulator": is_upstream_regulator, "n_programs": n_programs},
    }


def _make_pipeline_outputs(profiles: list[dict], upstream_regs: list[str] | None = None) -> dict:
    return {
        "disease_name": "inflammatory bowel disease",
        "efo_id":       "EFO:0003767",
        "evidence_landscape": {
            "profiles": profiles,
        },
        "upstream_regulator_evidence": {
            "regulators": [{"gene": g} for g in (upstream_regs or [])],
        },
    }


# ---------------------------------------------------------------------------
# OPPORTUNITY_CLASSES coverage
# ---------------------------------------------------------------------------

class TestOpportunityClassesDefinition:
    def test_all_classes_defined(self):
        required = {
            "convergent_needs_drug",
            "gwas_provisional_upgradable",
            "druggable_state_nominated",
            "novel_unexplored",
            "upstream_chokepoint",
            "cross_disease_overlap",
            "escaped_ranking",
        }
        assert required.issubset(set(OPPORTUNITY_CLASSES.keys()))


# ---------------------------------------------------------------------------
# run() entry point
# ---------------------------------------------------------------------------

class TestRunEntryPoint:
    def test_returns_dict(self):
        result = run({}, DISEASE_QUERY)
        assert isinstance(result, dict)

    def test_required_keys_present(self):
        result = run({}, DISEASE_QUERY)
        for key in ("novel_high_value_targets", "upgraded_evidence", "evidence_gaps",
                    "chokepoint_regulators", "analysis_summary", "mode", "n_queries_run"):
            assert key in result, f"Missing key: {key}"

    def test_mode_is_local_heuristic_without_api(self):
        result = run({}, DISEASE_QUERY)
        assert result["mode"] == "local_heuristic"

    def test_no_queries_in_local_mode(self):
        result = run({}, DISEASE_QUERY)
        assert result["n_queries_run"] == 0

    def test_empty_input_produces_empty_lists(self):
        result = run({}, DISEASE_QUERY)
        assert result["novel_high_value_targets"] == []
        assert result["chokepoint_regulators"] == []

    def test_disease_name_propagated(self):
        result = run({}, DISEASE_QUERY)
        assert result["disease_name"] == "inflammatory bowel disease"

    def test_efo_id_propagated(self):
        result = run({}, DISEASE_QUERY)
        assert result["efo_id"] == "EFO:0003767"


# ---------------------------------------------------------------------------
# Convergent gene — no drug
# ---------------------------------------------------------------------------

class TestConvergentNeedsDrug:
    def test_convergent_no_drug_flagged(self):
        profile = _make_profile("IL23R", evidence_class="convergent", ota_gamma=0.42)
        po = _make_pipeline_outputs([profile])
        result = _local_run({**po, **DISEASE_QUERY})
        genes = [t["gene"] for t in result["novel_high_value_targets"]]
        assert "IL23R" in genes

    def test_convergent_no_drug_opportunity_class(self):
        profile = _make_profile("IL23R", evidence_class="convergent", ota_gamma=0.42)
        po = _make_pipeline_outputs([profile])
        result = _local_run({**po, **DISEASE_QUERY})
        entry = next(t for t in result["novel_high_value_targets"] if t["gene"] == "IL23R")
        assert entry["opportunity_class"] == "convergent_needs_drug"

    def test_convergent_no_drug_urgency_high(self):
        profile = _make_profile("IL23R", evidence_class="convergent", ota_gamma=0.42)
        po = _make_pipeline_outputs([profile])
        result = _local_run({**po, **DISEASE_QUERY})
        entry = next(t for t in result["novel_high_value_targets"] if t["gene"] == "IL23R")
        assert entry["urgency"] == "high"

    def test_convergent_with_drug_not_flagged(self):
        profile = _make_profile("TNF", evidence_class="convergent", ota_gamma=0.55,
                                max_phase=4, known_drugs=["adalimumab"])
        po = _make_pipeline_outputs([profile])
        result = _local_run({**po, **DISEASE_QUERY})
        genes = [t["gene"] for t in result["novel_high_value_targets"]]
        assert "TNF" not in genes


# ---------------------------------------------------------------------------
# Genetic anchor — no drug
# ---------------------------------------------------------------------------

class TestGeneticAnchorNeedsDrug:
    def test_genetic_anchor_no_drug_flagged(self):
        profile = _make_profile("NOD2", evidence_class="genetic_anchor", ota_gamma=0.35)
        po = _make_pipeline_outputs([profile])
        result = _local_run({**po, **DISEASE_QUERY})
        genes = [t["gene"] for t in result["novel_high_value_targets"]]
        assert "NOD2" in genes

    def test_genetic_anchor_no_drug_opportunity_class(self):
        profile = _make_profile("NOD2", evidence_class="genetic_anchor", ota_gamma=0.35)
        po = _make_pipeline_outputs([profile])
        result = _local_run({**po, **DISEASE_QUERY})
        entry = next(t for t in result["novel_high_value_targets"] if t["gene"] == "NOD2")
        assert entry["opportunity_class"] == "convergent_needs_drug"


# ---------------------------------------------------------------------------
# GWAS provisional — evidence gap flagged
# ---------------------------------------------------------------------------

class TestGwasProvisional:
    def test_gwas_provisional_added_to_evidence_gaps(self):
        profile = _make_profile("PTPN22", evidence_class="gwas_provisional", ota_gamma=0.12)
        po = _make_pipeline_outputs([profile])
        result = _local_run({**po, **DISEASE_QUERY})
        gap_genes = [g["gene"] for g in result["evidence_gaps"]]
        assert "PTPN22" in gap_genes

    def test_gwas_provisional_gap_type(self):
        profile = _make_profile("PTPN22", evidence_class="gwas_provisional", ota_gamma=0.12)
        po = _make_pipeline_outputs([profile])
        result = _local_run({**po, **DISEASE_QUERY})
        gap = next(g for g in result["evidence_gaps"] if g["gene"] == "PTPN22")
        assert gap["gap_type"] == "missing_eqtl_instrument"

    def test_gwas_provisional_potential_upgrade(self):
        profile = _make_profile("PTPN22", evidence_class="gwas_provisional")
        po = _make_pipeline_outputs([profile])
        result = _local_run({**po, **DISEASE_QUERY})
        gap = next(g for g in result["evidence_gaps"] if g["gene"] == "PTPN22")
        assert gap["potential_upgrade"] == "genetic_anchor"


# ---------------------------------------------------------------------------
# State-nominated high-τ
# ---------------------------------------------------------------------------

class TestStateNominatedHighTau:
    def test_high_tau_state_nominated_flagged(self):
        profile = _make_profile(
            "TREM2", evidence_class="state_nominated",
            tau_disease=0.82, tau_specificity_class="disease_specific", tau_log2fc=2.1
        )
        po = _make_pipeline_outputs([profile])
        result = _local_run({**po, **DISEASE_QUERY})
        genes = [t["gene"] for t in result["novel_high_value_targets"]]
        assert "TREM2" in genes

    def test_high_tau_state_nominated_opportunity_class(self):
        profile = _make_profile(
            "TREM2", evidence_class="state_nominated",
            tau_disease=0.82, tau_specificity_class="disease_specific"
        )
        po = _make_pipeline_outputs([profile])
        result = _local_run({**po, **DISEASE_QUERY})
        entry = next(t for t in result["novel_high_value_targets"] if t["gene"] == "TREM2")
        assert entry["opportunity_class"] == "druggable_state_nominated"

    def test_low_tau_state_nominated_not_flagged(self):
        profile = _make_profile(
            "GAPDH", evidence_class="state_nominated",
            tau_disease=0.2, tau_specificity_class="ubiquitous"
        )
        po = _make_pipeline_outputs([profile])
        result = _local_run({**po, **DISEASE_QUERY})
        genes = [t["gene"] for t in result["novel_high_value_targets"]]
        assert "GAPDH" not in genes

    def test_high_tau_with_drug_urgency_high(self):
        profile = _make_profile(
            "TREM2", evidence_class="state_nominated",
            tau_disease=0.82, tau_specificity_class="disease_specific",
            max_phase=2, known_drugs=["AL002C"]
        )
        po = _make_pipeline_outputs([profile])
        result = _local_run({**po, **DISEASE_QUERY})
        entry = next(t for t in result["novel_high_value_targets"] if t["gene"] == "TREM2")
        assert entry["urgency"] == "high"

    def test_high_tau_without_drug_urgency_medium(self):
        profile = _make_profile(
            "TREM2", evidence_class="state_nominated",
            tau_disease=0.82, tau_specificity_class="disease_specific"
        )
        po = _make_pipeline_outputs([profile])
        result = _local_run({**po, **DISEASE_QUERY})
        entry = next(t for t in result["novel_high_value_targets"] if t["gene"] == "TREM2")
        assert entry["urgency"] == "medium"

    def test_null_tau_not_flagged(self):
        profile = _make_profile("MYH9", evidence_class="state_nominated", tau_disease=None)
        po = _make_pipeline_outputs([profile])
        result = _local_run({**po, **DISEASE_QUERY})
        genes = [t["gene"] for t in result["novel_high_value_targets"]]
        assert "MYH9" not in genes


# ---------------------------------------------------------------------------
# Novel — no prior literature
# ---------------------------------------------------------------------------

class TestNovelUnexplored:
    def test_novel_state_nominated_flagged(self):
        profile = _make_profile(
            "CFAP46", evidence_class="state_nominated",
            lit_confidence="NOVEL"
        )
        po = _make_pipeline_outputs([profile])
        result = _local_run({**po, **DISEASE_QUERY})
        genes = [t["gene"] for t in result["novel_high_value_targets"]]
        assert "CFAP46" in genes

    def test_novel_state_nominated_opportunity_class(self):
        profile = _make_profile(
            "CFAP46", evidence_class="state_nominated",
            lit_confidence="NOVEL"
        )
        po = _make_pipeline_outputs([profile])
        result = _local_run({**po, **DISEASE_QUERY})
        entry = next(t for t in result["novel_high_value_targets"] if t["gene"] == "CFAP46")
        assert entry["opportunity_class"] == "novel_unexplored"

    def test_novel_convergent_not_double_flagged_as_novel(self):
        # Convergent genes should NOT get novel_unexplored (they're already flagged as convergent_needs_drug)
        profile = _make_profile("IL23R", evidence_class="convergent", ota_gamma=0.42,
                                lit_confidence="NOVEL")
        po = _make_pipeline_outputs([profile])
        result = _local_run({**po, **DISEASE_QUERY})
        # convergent is excluded from novel_unexplored by design
        novel_entries = [t for t in result["novel_high_value_targets"]
                         if t["gene"] == "IL23R" and t["opportunity_class"] == "novel_unexplored"]
        assert novel_entries == []


# ---------------------------------------------------------------------------
# Upstream chokepoints
# ---------------------------------------------------------------------------

class TestChokepoints:
    def test_upstream_regulator_added_to_chokepoints(self):
        profile = _make_profile(
            "IRF4", evidence_class="perturb_seq_regulator",
            is_upstream_regulator=True, n_programs=3
        )
        po = _make_pipeline_outputs([profile])
        result = _local_run({**po, **DISEASE_QUERY})
        chokepoint_genes = [c["gene"] for c in result["chokepoint_regulators"]]
        assert "IRF4" in chokepoint_genes

    def test_low_program_count_not_chokepoint(self):
        profile = _make_profile(
            "RUNX1", evidence_class="state_nominated",
            is_upstream_regulator=True, n_programs=1
        )
        po = _make_pipeline_outputs([profile])
        result = _local_run({**po, **DISEASE_QUERY})
        chokepoint_genes = [c["gene"] for c in result["chokepoint_regulators"]]
        assert "RUNX1" not in chokepoint_genes

    def test_chokepoints_sorted_by_n_programs_desc(self):
        profiles = [
            _make_profile("GENE_A", is_upstream_regulator=True, n_programs=2),
            _make_profile("GENE_B", is_upstream_regulator=True, n_programs=5),
            _make_profile("GENE_C", is_upstream_regulator=True, n_programs=3),
        ]
        po = _make_pipeline_outputs(profiles)
        result = _local_run({**po, **DISEASE_QUERY})
        n_programs = [c["n_programs"] for c in result["chokepoint_regulators"]]
        assert n_programs == sorted(n_programs, reverse=True)

    def test_chokepoint_max_5(self):
        profiles = [
            _make_profile(f"GENE_{i}", is_upstream_regulator=True, n_programs=i + 2)
            for i in range(8)
        ]
        po = _make_pipeline_outputs(profiles)
        result = _local_run({**po, **DISEASE_QUERY})
        assert len(result["chokepoint_regulators"]) <= 5


# ---------------------------------------------------------------------------
# DEPRIORITIZE filtering
# ---------------------------------------------------------------------------

class TestDeprioritize:
    def test_deprioritized_gene_excluded(self):
        profile = _make_profile(
            "VEGFA", evidence_class="convergent", ota_gamma=0.9,
            rt_verdict="DEPRIORITIZE"
        )
        po = _make_pipeline_outputs([profile])
        result = _local_run({**po, **DISEASE_QUERY})
        genes = [t["gene"] for t in result["novel_high_value_targets"]]
        assert "VEGFA" not in genes

    def test_deprioritized_gene_not_in_chokepoints(self):
        profile = _make_profile(
            "VEGFA", evidence_class="convergent",
            is_upstream_regulator=True, n_programs=4,
            rt_verdict="DEPRIORITIZE"
        )
        po = _make_pipeline_outputs([profile])
        result = _local_run({**po, **DISEASE_QUERY})
        chokepoint_genes = [c["gene"] for c in result["chokepoint_regulators"]]
        assert "VEGFA" not in chokepoint_genes


# ---------------------------------------------------------------------------
# Escaped ranking gap
# ---------------------------------------------------------------------------

class TestEscapedRanking:
    def test_perturb_regulator_state_nominated_gap_flagged(self):
        profile = _make_profile("STAT3", evidence_class="state_nominated", ota_gamma=0.0)
        po = _make_pipeline_outputs([profile], upstream_regs=["STAT3"])
        result = _local_run({**po, **DISEASE_QUERY})
        gap_genes = [g["gene"] for g in result["evidence_gaps"]]
        assert "STAT3" in gap_genes

    def test_escaped_ranking_gap_type(self):
        profile = _make_profile("STAT3", evidence_class="state_nominated", ota_gamma=0.0)
        po = _make_pipeline_outputs([profile], upstream_regs=["STAT3"])
        result = _local_run({**po, **DISEASE_QUERY})
        gap = next(g for g in result["evidence_gaps"] if g["gene"] == "STAT3")
        assert gap["gap_type"] == "escaped_ranking_perturb_regulator"


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

class TestDeduplication:
    def test_gene_not_duplicated_in_novel_high_value(self):
        # A gene with convergent evidence AND NOVEL lit could trigger two paths
        profile = _make_profile(
            "IL23R", evidence_class="convergent", ota_gamma=0.42,
            lit_confidence="NOVEL"
        )
        po = _make_pipeline_outputs([profile])
        result = _local_run({**po, **DISEASE_QUERY})
        genes = [t["gene"] for t in result["novel_high_value_targets"]]
        assert genes.count("IL23R") == 1


# ---------------------------------------------------------------------------
# Output caps
# ---------------------------------------------------------------------------

class TestOutputCaps:
    def test_novel_high_value_capped_at_10(self):
        profiles = [
            _make_profile(f"GENE_{i}", evidence_class="convergent", ota_gamma=0.3 + i * 0.01)
            for i in range(15)
        ]
        po = _make_pipeline_outputs(profiles)
        result = _local_run({**po, **DISEASE_QUERY})
        assert len(result["novel_high_value_targets"]) <= 10

    def test_evidence_gaps_capped_at_10(self):
        profiles = [
            _make_profile(f"GENE_{i}", evidence_class="gwas_provisional")
            for i in range(15)
        ]
        po = _make_pipeline_outputs(profiles)
        result = _local_run({**po, **DISEASE_QUERY})
        assert len(result["evidence_gaps"]) <= 10


# ---------------------------------------------------------------------------
# Analysis summary
# ---------------------------------------------------------------------------

class TestAnalysisSummary:
    def test_summary_is_string(self):
        result = run({}, DISEASE_QUERY)
        assert isinstance(result["analysis_summary"], str)

    def test_summary_contains_disease_name(self):
        profile = _make_profile("IL23R", evidence_class="convergent", ota_gamma=0.42)
        po = _make_pipeline_outputs([profile])
        result = _local_run({**po, **DISEASE_QUERY})
        assert "inflammatory bowel disease" in result["analysis_summary"]

    def test_summary_mentions_urgency_counts(self):
        profile = _make_profile("IL23R", evidence_class="convergent", ota_gamma=0.42)
        po = _make_pipeline_outputs([profile])
        result = _local_run({**po, **DISEASE_QUERY})
        # Summary should mention high/medium urgency counts
        assert "high-urgency" in result["analysis_summary"] or "high urgency" in result["analysis_summary"].lower()
