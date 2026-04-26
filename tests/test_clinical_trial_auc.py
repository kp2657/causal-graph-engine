"""
tests/test_clinical_trial_auc.py — Unit tests for clinical trial AUC validation.

All tests use synthetic data and mock the OT network call.
"""
from __future__ import annotations

import sys
import urllib.error
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipelines.validation.clinical_trial_auc import (
    compute_auroc,
    query_ot_clinical_trials,
    run_clinical_trial_auc,
    _map_trial_status,
)


class TestComputeAuroc:
    def test_auroc_perfect(self):
        """Perfect classifier: high scores → success, low scores → failure."""
        scores = [1.0, 0.9, 0.1, 0.0]
        labels = [1, 1, 0, 0]
        auroc = compute_auroc(scores, labels)
        assert abs(auroc - 1.0) < 1e-9

    def test_auroc_random(self):
        """Random classifier (all same score) should yield AUROC = 0.5."""
        scores = [0.5, 0.5, 0.5, 0.5]
        labels = [1, 0, 1, 0]
        auroc = compute_auroc(scores, labels)
        assert abs(auroc - 0.5) < 1e-9

    def test_auroc_degenerate_all_success(self):
        """All labels = 1 → degenerate case, returns 0.5."""
        scores = [0.9, 0.7, 0.5]
        labels = [1, 1, 1]
        auroc = compute_auroc(scores, labels)
        assert auroc == 0.5

    def test_compute_auroc_known_value(self):
        """
        Manual calculation:
        scores=[0.8, 0.6, 0.4, 0.2], labels=[1, 0, 1, 0]
        pos_scores=[0.8, 0.4], neg_scores=[0.6, 0.2]
        Pairs: (0.8>0.6)✓, (0.8>0.2)✓, (0.4<0.6)✗, (0.4>0.2)✓
        U = 3, total pairs = 4, AUROC = 3/4 = 0.75
        """
        scores = [0.8, 0.6, 0.4, 0.2]
        labels = [1, 0, 1, 0]
        auroc = compute_auroc(scores, labels)
        assert abs(auroc - 0.75) < 0.01


class TestRunClinicalTrialAuc:
    def test_run_auc_insufficient_labels(self):
        """Fewer than 5 labeled genes → returns skipped=True."""
        # Mock query to return empty (no labeled genes)
        with patch(
            "pipelines.validation.clinical_trial_auc.query_ot_clinical_trials",
            return_value=[],
        ):
            scored_genes = [
                {"gene": f"GENE{i}", "ota_gamma": float(i) / 10}
                for i in range(10)
            ]
            result = run_clinical_trial_auc(
                scored_genes=scored_genes,
                disease_efo="EFO_0001645",
                disease_key="CAD",
            )
        assert result.get("skipped") is True
        assert result["reason"] == "insufficient_labels"
        assert result["disease_key"] == "CAD"


class TestQueryOtTrials:
    def test_query_ot_trials_network_error(self):
        """Network error from urllib → returns empty list, does not raise."""
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("timeout")):
            result = query_ot_clinical_trials(
                gene_symbol="PCSK9",
                disease_efo="EFO_0001645",
            )
        assert result == []


class TestPhaseMapping:
    def test_phase_mapping_approved_success(self):
        """Phase 3 + 'Approved' status → 'success'."""
        outcome = _map_trial_status(phase=3, status="Approved")
        assert outcome == "success"

    def test_phase_mapping_withdrawn_failure(self):
        """'Withdrawn' status (any phase) → 'failure'."""
        outcome = _map_trial_status(phase=2, status="Withdrawn")
        assert outcome == "failure"

    def test_phase_mapping_phase4_no_status(self):
        """Phase 4 with no status → 'success' (approved drug)."""
        outcome = _map_trial_status(phase=4, status=None)
        assert outcome == "success"

    def test_phase_mapping_phase2_completed_unknown(self):
        """Phase 2 + no explicit failure/success status → 'unknown'."""
        outcome = _map_trial_status(phase=2, status="ongoing")
        assert outcome == "unknown"
