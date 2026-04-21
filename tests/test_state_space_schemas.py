"""
tests/test_state_space_schemas.py

Unit tests for Phase 1 schema additions:
  - StateEvidenceTier Literal values
  - Five new pydantic models in models.evidence
  - graph/state_schema.py structure
  - pipelines/state_space/schemas.py constants

No sc-RNA data or heavy dependencies required — pure Python + pydantic.
"""
import pytest
from typing import get_args


class TestStateEvidenceTier:
    def test_has_three_tiers(self):
        from models.evidence import StateEvidenceTier
        tiers = get_args(StateEvidenceTier)
        assert len(tiers) == 3

    def test_tier1_direct(self):
        from models.evidence import StateEvidenceTier
        assert "Tier1_TrajectoryDirect" in get_args(StateEvidenceTier)

    def test_tier2_inferred(self):
        from models.evidence import StateEvidenceTier
        assert "Tier2_TrajectoryInferred" in get_args(StateEvidenceTier)

    def test_tier3_proxy(self):
        from models.evidence import StateEvidenceTier
        assert "Tier3_TrajectoryProxy" in get_args(StateEvidenceTier)

    def test_original_evidence_tier_unchanged(self):
        """Existing EvidenceTier must not have state-space tiers injected."""
        from models.evidence import EvidenceTier
        original_tiers = get_args(EvidenceTier)
        assert "Tier1_TrajectoryDirect" not in original_tiers
        assert "provisional_virtual" in original_tiers   # sentinel from original


class TestCellStateModel:
    def test_minimal_construction(self):
        from models.evidence import CellState
        s = CellState(state_id="IBD_inter_0", disease="IBD", cell_type="macrophage",
                      resolution="intermediate", n_cells=42)
        assert s.state_id == "IBD_inter_0"
        assert s.n_cells == 42

    def test_optional_fields_default_empty(self):
        from models.evidence import CellState
        s = CellState(state_id="x", disease="IBD", cell_type="T_cell",
                      resolution="coarse", n_cells=10)
        assert s.marker_genes == []
        assert s.program_labels == []
        assert s.context_tags == []
        assert s.centroid is None
        assert s.stability_score is None
        assert s.pathological_score is None

    def test_all_fields(self):
        from models.evidence import CellState
        s = CellState(
            state_id="IBD_fine_3",
            disease="IBD",
            cell_type="macrophage",
            resolution="fine",
            n_cells=120,
            centroid=[0.1, 0.2, -0.3],
            marker_genes=["NOD2", "IL23R"],
            program_labels=["HALLMARK_TNF_SIGNALING"],
            context_tags=["pathological"],
            stability_score=0.82,
            pathological_score=0.71,
            evidence_sources=["EFO_0003767"],
        )
        assert s.pathological_score == pytest.approx(0.71)
        assert "NOD2" in s.marker_genes


class TestStateTransitionModel:
    def test_minimal(self):
        from models.evidence import StateTransition
        t = StateTransition(
            from_state="IBD_inter_0",
            to_state="IBD_inter_1",
            disease="IBD",
            baseline_probability=0.35,
        )
        assert t.baseline_probability == pytest.approx(0.35)
        assert t.uncertainty is None

    def test_direction_evidence_list(self):
        from models.evidence import StateTransition
        t = StateTransition(
            from_state="A", to_state="B", disease="IBD",
            baseline_probability=0.2,
            direction_evidence=["pseudotime_inferred"],
        )
        assert "pseudotime_inferred" in t.direction_evidence


class TestPerturbationTransitionEffectModel:
    def test_construction(self):
        from models.evidence import PerturbationTransitionEffect
        e = PerturbationTransitionEffect(
            perturbation_id="NOD2_KO",
            perturbation_type="KO",
            disease="IBD",
            cell_type="macrophage",
            from_state="IBD_inter_3",
            to_state="IBD_inter_1",
            delta_probability=0.18,
            evidence_tier="Tier2_TrajectoryInferred",
            data_source="Papalexi2021_PBMC",
        )
        assert e.delta_probability == pytest.approx(0.18)
        assert e.durability_label is None

    def test_invalid_tier_rejected(self):
        from models.evidence import PerturbationTransitionEffect
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            PerturbationTransitionEffect(
                perturbation_id="X", perturbation_type="KO", disease="IBD",
                cell_type="mac", from_state="A", to_state="B",
                delta_probability=0.1,
                evidence_tier="Tier1_Interventional",   # classic tier — invalid here
                data_source="test",
            )


class TestFailureRecordModel:
    def test_construction(self):
        from models.evidence import FailureRecord
        r = FailureRecord(
            failure_id="anti_TNF_non_responder_IBD",
            disease="IBD",
            perturbation_id="TNF_blockade",
            perturbation_type="drug",
            failure_mode="non_responder",
            evidence_strength=0.75,
            data_source="ClinicalTrials.gov",
        )
        assert r.failure_mode == "non_responder"
        assert r.evidence_strength == pytest.approx(0.75)

    def test_explanation_candidates_default_empty(self):
        from models.evidence import FailureRecord
        r = FailureRecord(
            failure_id="x", disease="IBD", perturbation_id="y",
            perturbation_type="KO", failure_mode="escape",
            evidence_strength=0.5, data_source="test",
        )
        assert r.explanation_candidates == []


class TestTrajectoryRedirectionScoreModel:
    def test_construction(self):
        from models.evidence import TrajectoryRedirectionScore
        sc = TrajectoryRedirectionScore(
            entity_id="NOD2",
            entity_type="gene",
            disease="IBD",
            expected_pathology_reduction=0.45,
            durable_redirection_score=0.60,
            escape_risk_score=0.15,
            non_response_risk_score=0.30,
            negative_memory_penalty=0.10,
        )
        assert sc.entity_type == "gene"
        assert sc.provenance == []


class TestStateSpaceSchemasConstants:
    def test_default_resolutions_keys(self):
        from pipelines.state_space.schemas import DEFAULT_RESOLUTIONS
        assert set(DEFAULT_RESOLUTIONS.keys()) == {"coarse", "intermediate", "fine"}

    def test_coarse_lt_intermediate_lt_fine(self):
        from pipelines.state_space.schemas import DEFAULT_RESOLUTIONS
        assert DEFAULT_RESOLUTIONS["coarse"] < DEFAULT_RESOLUTIONS["intermediate"]
        assert DEFAULT_RESOLUTIONS["intermediate"] < DEFAULT_RESOLUTIONS["fine"]

    def test_provenance_constants_are_strings(self):
        from pipelines.state_space import schemas
        for attr in ("PROV_PSEUDOTIME_INFERRED", "PROV_PCA_DIFFUSION", "PROV_KNN_FLOW"):
            assert isinstance(getattr(schemas, attr), str)

    def test_failure_modes_tuple(self):
        from pipelines.state_space.schemas import FAILURE_MODES
        assert "no_effect" in FAILURE_MODES
        assert "escape" in FAILURE_MODES
        assert "non_responder" in FAILURE_MODES


class TestStateGraphSchema:
    def test_node_tables_defined(self):
        from graph.state_schema import STATE_NODE_TABLES
        assert "CellState" in STATE_NODE_TABLES
        assert "StateBasin" in STATE_NODE_TABLES
        assert "Perturbation" in STATE_NODE_TABLES
        assert "FailureMode" in STATE_NODE_TABLES

    def test_edge_tables_defined(self):
        from graph.state_schema import STATE_EDGE_TABLES
        assert "TRANSITIONS_TO" in STATE_EDGE_TABLES
        assert "PERTURBATION_REDIRECTS_STATE" in STATE_EDGE_TABLES
        assert "PERTURBATION_FAILS_IN_STATE" in STATE_EDGE_TABLES
        assert "ESCAPES_TO_STATE" in STATE_EDGE_TABLES

    def test_transitions_to_connects_cell_states(self):
        from graph.state_schema import STATE_EDGE_TABLES
        spec = STATE_EDGE_TABLES["TRANSITIONS_TO"]
        assert spec["from"] == "CellState"
        assert spec["to"] == "CellState"

    def test_cell_state_pk_is_state_id(self):
        from graph.state_schema import STATE_NODE_TABLES, get_create_node_table_cypher
        cypher = get_create_node_table_cypher("CellState")
        assert "state_id" in cypher
        assert "PRIMARY KEY" in cypher

    def test_ddl_helpers_produce_valid_strings(self):
        from graph.state_schema import get_create_node_table_cypher, get_create_rel_table_cypher
        node_ddl = get_create_node_table_cypher("Perturbation")
        rel_ddl  = get_create_rel_table_cypher("TRANSITIONS_TO")
        assert "CREATE NODE TABLE" in node_ddl
        assert "CREATE REL TABLE" in rel_ddl
        assert "IF NOT EXISTS" in node_ddl
        assert "IF NOT EXISTS" in rel_ddl
