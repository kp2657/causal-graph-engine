"""
Tests for open_targets_server.py.

Unit tests use the hardcoded CAD target registry.
Integration tests hit the live Open Targets GraphQL API.

Run unit:        pytest tests/test_open_targets_server.py -v -m "not integration"
Run integration: pytest tests/test_open_targets_server.py -v -m integration
"""
from __future__ import annotations

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp_servers.open_targets_server import (
    get_open_targets_disease_targets,
    get_open_targets_target_info,
    get_open_targets_drug_info,
    run_txgnn_repurposing,
)


class TestOpenTargetsUnit:

    def test_cad_targets_uses_cache(self):
        result = get_open_targets_disease_targets("EFO_0001645")
        assert result["efo_id"] == "EFO_0001645"
        assert result["n_targets"] > 0
        assert "targets" in result

    def test_cad_includes_pcsk9(self):
        result = get_open_targets_disease_targets("EFO_0001645")
        symbols = [t["gene_symbol"] for t in result["targets"]]
        assert "PCSK9" in symbols

    def test_cad_includes_hmgcr(self):
        result = get_open_targets_disease_targets("EFO_0001645")
        symbols = [t["gene_symbol"] for t in result["targets"]]
        assert "HMGCR" in symbols

    def test_min_score_filter(self):
        result = get_open_targets_disease_targets("EFO_0001645", min_overall_score=0.8)
        for t in result["targets"]:
            assert t["overall_score"] >= 0.8

    def test_max_targets_limit(self):
        result = get_open_targets_disease_targets("EFO_0001645", max_targets=2)
        assert result["n_targets"] <= 2

    def test_txgnn_stub_schema(self):
        result = run_txgnn_repurposing("EFO_0001645")
        assert result["predictions"] is None
        assert "STUB" in result["note"]
        assert "paper" in result


@pytest.mark.integration
class TestOpenTargetsLive:

    def test_cad_live_query(self):
        # Test non-cached disease
        result = get_open_targets_disease_targets("EFO_0000685")  # RA
        assert "targets" in result or "error" in result

    def test_pcsk9_target_info_live(self):
        result = get_open_targets_target_info("PCSK9")
        assert "gene_symbol" in result
        if "error" not in result:
            assert result.get("n_drugs") is not None

    def test_atorvastatin_drug_info_live(self):
        result = get_open_targets_drug_info("atorvastatin")
        assert "drug_name" in result
        if "error" not in result:
            assert "targets" in result
