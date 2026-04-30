"""
Tests for Phase 1 agentic infrastructure:
    agent_config, agent_contracts, run_journal
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from orchestrator.agentic.agent_config import (
    AGENTIC_MODEL,
    ALL_AGENTS,
    estimate_cost,
    is_agentic,
)
from orchestrator.agentic.agent_contracts import (
    AgentRunInput,
    AgentTokenUsage,
    AugmentationRecommendation,
    CSOOutput,
    ConvergentEvidenceOutput,
    DiscoveryRefinementOutput,
    EvidenceItem,
    HumanPause,
    LibraryGap,
    MethodsChoice,
    PathReasoning,
    RedelegationRecord,
    ReviewerOutput,
    StatisticalGeneticistOutput,
    TokenUsage,
    VirginTarget,
)
from orchestrator.agentic.run_journal import RunJournal


# ---------------------------------------------------------------------------
# agent_config
# ---------------------------------------------------------------------------

def test_default_mode_is_local():
    assert is_agentic() is False


def test_model_constant():
    assert AGENTIC_MODEL == "claude-sonnet-4-6"


def test_estimate_cost_zero():
    assert estimate_cost(0, 0) == 0.0


def test_estimate_cost_proportional():
    cost_1m_input = estimate_cost(1_000_000, 0)
    cost_1m_output = estimate_cost(0, 1_000_000)
    assert cost_1m_output > cost_1m_input  # output tokens cost more


def test_all_agents_nonempty():
    assert len(ALL_AGENTS) == 9


# ---------------------------------------------------------------------------
# TokenUsage addition
# ---------------------------------------------------------------------------

def test_token_usage_add():
    t1 = TokenUsage(input_tokens=100, output_tokens=50, cost_usd=estimate_cost(100, 50))
    t2 = TokenUsage(input_tokens=200, output_tokens=100, cost_usd=estimate_cost(200, 100))
    t3 = t1 + t2
    assert t3.input_tokens == 300
    assert t3.output_tokens == 150
    assert t3.cost_usd == pytest.approx(estimate_cost(300, 150))


# ---------------------------------------------------------------------------
# EvidenceItem and VirginTarget tier logic
# ---------------------------------------------------------------------------

def _make_evidence(gene: str, tier: str, confidence: str = "HIGH") -> EvidenceItem:
    return EvidenceItem(
        gene=gene, modality_name="test", tier=tier,  # type: ignore[arg-type]
        source="test", confidence=confidence, description="test",  # type: ignore[arg-type]
    )


def test_convergent_evidence_requires_tier_a_or_b():
    # Tier A + Tier C — qualifies (has strong leg + multi-tier)
    vt = VirginTarget(
        gene="G1",
        evidence_items=[_make_evidence("G1", "A"), _make_evidence("G1", "C")],
        priority="HIGH",
    )
    assert vt.has_convergent_evidence is True


def test_convergent_evidence_tier_b_plus_c():
    # Tier B + Tier C — qualifies
    vt = VirginTarget(
        gene="G1",
        evidence_items=[_make_evidence("G1", "B"), _make_evidence("G1", "C")],
        priority="HIGH",
    )
    assert vt.has_convergent_evidence is True


def test_gps_plus_perturb_seq_is_not_convergent():
    # Both Tier C — GPS reversal + Perturb-seq β = replication, not convergence
    vt = VirginTarget(
        gene="G2",
        evidence_items=[_make_evidence("G2", "C"), _make_evidence("G2", "C")],
        priority="LOW",
    )
    assert vt.has_convergent_evidence is False


def test_single_tier_a_not_convergent():
    # Only one leg, even if Tier A
    vt = VirginTarget(
        gene="G3",
        evidence_items=[_make_evidence("G3", "A")],
        priority="LOW",
    )
    assert vt.has_convergent_evidence is False


def test_low_confidence_legs_dont_count():
    # LOW confidence items shouldn't contribute
    vt = VirginTarget(
        gene="G4",
        evidence_items=[
            _make_evidence("G4", "A", confidence="LOW"),
            _make_evidence("G4", "C", confidence="LOW"),
        ],
        priority="LOW",
    )
    assert vt.has_convergent_evidence is False


# ---------------------------------------------------------------------------
# Pydantic model round-trips
# ---------------------------------------------------------------------------

def test_statistical_geneticist_output_round_trip():
    out = StatisticalGeneticistOutput(
        agent_name="statistical_geneticist",
        run_id="test_run",
        validated_anchors=["PCSK9", "IL6R"],
        library_gaps=[LibraryGap(gene="LDLR", known_causal_reason="Mendelian", suggested_modalities=["LOF"])],
    )
    data = out.model_dump()
    assert data["validated_anchors"] == ["PCSK9", "IL6R"]
    assert data["library_gaps"][0]["gene"] == "LDLR"


def test_reviewer_output_defaults():
    out = ReviewerOutput(agent_name="scientific_reviewer", run_id="r1")
    assert out.verdict == "APPROVE"
    assert out.hard_rejected_targets == []


def test_cso_output_modes():
    for mode in ("briefing", "conflict_analysis", "exec_summary"):
        out = CSOOutput(agent_name="chief_of_staff", run_id="r1", mode=mode)  # type: ignore[arg-type]
        assert out.mode == mode


# ---------------------------------------------------------------------------
# RunJournal
# ---------------------------------------------------------------------------

def test_run_journal_creates_output_dir():
    with tempfile.TemporaryDirectory() as tmp:
        j = RunJournal(disease_key="CAD", run_id="test001", base_dir=tmp)
        assert j.run_dir.exists()


def test_run_journal_close_writes_files():
    with tempfile.TemporaryDirectory() as tmp:
        j = RunJournal(disease_key="RA", run_id="test002", base_dir=tmp)

        j.log_path_reasoning(PathReasoning(agent="statistical_geneticist", statement="Most likely path is PubMed search"))
        j.log_methods_choice(MethodsChoice(agent="cso", choice="pause", rationale="low anchor coverage", alternatives_considered=["continue"]))
        j.log_virgin_target(VirginTarget(
            gene="NOVEL1",
            evidence_items=[_make_evidence("NOVEL1", "A"), _make_evidence("NOVEL1", "B")],
            priority="HIGH",
            gate1_confirmed=True,
            gate2_confirmed=True,
        ))
        j.log_library_gap(LibraryGap(gene="CTLA4", known_causal_reason="Mendelian RA gene", suggested_modalities=["animal_model"]))
        j.log_augmentation_recommendation(AugmentationRecommendation(
            agent="convergent_evidence_agent", gene="NOVEL1", accession="GSE123456",
            source_db="GEO", description="CRISPRa screen in CD4+ T cells",
        ))
        j.log_token_usage(AgentTokenUsage(
            agent_name="statistical_geneticist",
            usage=TokenUsage(input_tokens=1000, output_tokens=500, cost_usd=estimate_cost(1000, 500)),
        ))
        run_dir = j.close()

        journal_path = run_dir / "journal.json"
        assert journal_path.exists()
        data = json.loads(journal_path.read_text())
        assert data["disease_key"] == "RA"
        assert data["virgin_targets"][0]["gene"] == "NOVEL1"
        assert data["library_gaps"][0]["gene"] == "CTLA4"
        assert data["augmentation_recommendations"][0]["accession"] == "GSE123456"

        token_path = run_dir / "token_usage.json"
        assert token_path.exists()
        token_data = json.loads(token_path.read_text())
        assert token_data["total"]["input_tokens"] == 1000
        assert token_data["total"]["output_tokens"] == 500


def test_run_journal_empty_close():
    with tempfile.TemporaryDirectory() as tmp:
        j = RunJournal(disease_key="CAD", run_id="empty", base_dir=tmp)
        run_dir = j.close()
        data = json.loads((run_dir / "journal.json").read_text())
        assert data["virgin_targets"] == []
        assert data["library_gaps"] == []
