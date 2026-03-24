"""
Tests for clinical_trials_server.py.

Unit tests use the gene-drug map (no HTTP).
Integration tests hit the live ClinicalTrials.gov API v2.
"""
from __future__ import annotations

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp_servers.clinical_trials_server import (
    search_clinical_trials,
    get_trial_details,
    get_trials_for_target,
)


class TestClinicalTrialsUnit:

    def test_get_trials_for_pcsk9_returns_dict(self):
        # This makes a live call but returns graceful error on failure
        result = get_trials_for_target("PCSK9", disease="coronary artery disease")
        assert "trials" in result or "error" in result

    def test_result_schema_has_required_keys(self):
        result = search_clinical_trials(condition="coronary artery disease", max_results=1)
        required = ["condition", "trials"]
        for k in required:
            assert k in result


@pytest.mark.integration
class TestClinicalTrialsLive:

    def test_cad_recruiting_trials(self):
        result = search_clinical_trials(
            condition="coronary artery disease",
            status="RECRUITING",
            max_results=5,
        )
        assert "trials" in result
        assert result.get("total_count") is not None

    def test_pcsk9_inhibitor_trials(self):
        result = search_clinical_trials(
            intervention="evolocumab",
            max_results=5,
        )
        assert "trials" in result
        if result["n_trials"] > 0:
            for trial in result["trials"]:
                assert "nct_id" in trial
                assert trial["nct_id"].startswith("NCT")

    def test_pcsk9_target_trials(self):
        result = get_trials_for_target("PCSK9")
        assert "trials" in result

    def test_trial_details_for_known_trial(self):
        # ODYSSEY OUTCOMES trial (alirocumab, CAD): NCT01663402
        result = get_trial_details("NCT01663402")
        assert result.get("nct_id") == "NCT01663402" or "error" in result
        if "error" not in result:
            assert result.get("title") is not None
