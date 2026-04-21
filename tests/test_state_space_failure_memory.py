"""
tests/test_state_space_failure_memory.py

Unit tests for pipelines/state_space/failure_memory.py.
All tests use curated seed data only (no network calls).
"""
import pytest
from models.evidence import FailureRecord


# ---------------------------------------------------------------------------
# build_failure_records (seed only)
# ---------------------------------------------------------------------------

class TestBuildFailureRecords:
    def test_ibd_returns_nonempty_list(self):
        from pipelines.state_space.failure_memory import build_failure_records
        records = build_failure_records("IBD", include_ct=False)
        assert isinstance(records, list)
        assert len(records) > 0

    def test_all_records_are_failure_record_instances(self):
        from pipelines.state_space.failure_memory import build_failure_records
        records = build_failure_records("IBD", include_ct=False)
        for r in records:
            assert isinstance(r, FailureRecord)

    def test_cad_returns_nonempty_list(self):
        from pipelines.state_space.failure_memory import build_failure_records
        records = build_failure_records("CAD", include_ct=False)
        assert len(records) > 0

    def test_unknown_disease_returns_empty(self):
        from pipelines.state_space.failure_memory import build_failure_records
        records = build_failure_records("UNKNOWN_XYZ", include_ct=False)
        assert records == []

    def test_disease_key_case_insensitive(self):
        from pipelines.state_space.failure_memory import build_failure_records
        lower = build_failure_records("ibd", include_ct=False)
        upper = build_failure_records("IBD", include_ct=False)
        assert len(lower) == len(upper)

    def test_failure_modes_are_valid_vocab(self):
        from pipelines.state_space.failure_memory import build_failure_records
        from pipelines.state_space.schemas import FAILURE_MODES
        records = build_failure_records("IBD", include_ct=False)
        for r in records:
            assert r.failure_mode in FAILURE_MODES, \
                f"Unknown failure_mode: {r.failure_mode}"

    def test_evidence_strength_in_zero_one(self):
        from pipelines.state_space.failure_memory import build_failure_records
        for disease in ("IBD", "CAD"):
            for r in build_failure_records(disease, include_ct=False):
                assert 0.0 <= r.evidence_strength <= 1.0

    def test_failure_ids_are_unique(self):
        from pipelines.state_space.failure_memory import build_failure_records
        records = build_failure_records("IBD", include_ct=False)
        ids = [r.failure_id for r in records]
        assert len(ids) == len(set(ids)), "Duplicate failure_ids"

    def test_disease_field_matches_input(self):
        from pipelines.state_space.failure_memory import build_failure_records
        for r in build_failure_records("IBD", include_ct=False):
            assert r.disease == "IBD"

    def test_ibd_contains_anti_tnf_failure(self):
        from pipelines.state_space.failure_memory import build_failure_records
        records = build_failure_records("IBD", include_ct=False)
        tnf_records = [r for r in records if "anti-TNF" in r.perturbation_id]
        assert len(tnf_records) >= 1

    def test_ibd_contains_escape_mode(self):
        from pipelines.state_space.failure_memory import build_failure_records
        records = build_failure_records("IBD", include_ct=False)
        modes = {r.failure_mode for r in records}
        assert "escape" in modes


# ---------------------------------------------------------------------------
# get_failure_modes_for_perturbation
# ---------------------------------------------------------------------------

class TestGetFailureModes:
    def test_returns_modes_for_known_perturbation(self):
        from pipelines.state_space.failure_memory import (
            build_failure_records, get_failure_modes_for_perturbation,
        )
        records = build_failure_records("IBD", include_ct=False)
        modes = get_failure_modes_for_perturbation("anti-TNF", records)
        assert isinstance(modes, list)
        assert len(modes) >= 1
        assert "escape" in modes

    def test_returns_empty_for_unknown_perturbation(self):
        from pipelines.state_space.failure_memory import (
            build_failure_records, get_failure_modes_for_perturbation,
        )
        records = build_failure_records("IBD", include_ct=False)
        modes = get_failure_modes_for_perturbation("completely_unknown_drug_xyz", records)
        assert modes == []


# ---------------------------------------------------------------------------
# failure_penalty_score
# ---------------------------------------------------------------------------

class TestFailurePenaltyScore:
    def test_zero_for_unknown_perturbation(self):
        from pipelines.state_space.failure_memory import (
            build_failure_records, failure_penalty_score,
        )
        records = build_failure_records("IBD", include_ct=False)
        score = failure_penalty_score("nonexistent_drug_xyz", records)
        assert score == 0.0

    def test_positive_for_known_bad_actor(self):
        from pipelines.state_space.failure_memory import (
            build_failure_records, failure_penalty_score,
        )
        records = build_failure_records("IBD", include_ct=False)
        score = failure_penalty_score("anti-TNF", records)
        assert score > 0.0

    def test_score_capped_at_one(self):
        from pipelines.state_space.failure_memory import (
            build_failure_records, failure_penalty_score,
        )
        records = build_failure_records("IBD", include_ct=False)
        for r in records:
            score = failure_penalty_score(r.perturbation_id, records)
            assert score <= 1.0

    def test_escape_mode_gets_higher_penalty_than_toxicity(self):
        from pipelines.state_space.failure_memory import (
            build_failure_records, failure_penalty_score,
        )
        ibd = build_failure_records("IBD", include_ct=False)
        cad = build_failure_records("CAD", include_ct=False)
        all_records = ibd + cad
        # anti-TNF (escape mode, high evidence) vs HDAC-inhibitor (toxicity, lower evidence)
        tnf_score = failure_penalty_score("anti-TNF", ibd)
        hdac_score = failure_penalty_score("HDAC-inhibitor", cad)
        assert tnf_score > hdac_score
