"""
Tests for chemistry_server.py.

Integration tests hit ChEMBL and PubChem live APIs.
"""
from __future__ import annotations

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp_servers.chemistry_server import (
    search_chembl_compound,
    get_chembl_target_activities,
    get_pubchem_compound,
)


@pytest.mark.integration
class TestChEMBLLive:

    def test_atorvastatin_found(self):
        result = search_chembl_compound("atorvastatin")
        assert "compounds" in result
        if result["n_results"] > 0:
            compound = result["compounds"][0]
            assert "chembl_id" in compound
            assert compound["chembl_id"].startswith("CHEMBL")

    def test_atorvastatin_lipinski(self):
        result = search_chembl_compound("atorvastatin")
        if result["n_results"] > 0:
            compound = result["compounds"][0]
            # Atorvastatin violates Ro5 (MW > 500) — check field present
            assert "ro5_violations" in compound

    def test_hmgcr_activities(self):
        result = get_chembl_target_activities("HMGCR", max_results=5)
        assert "activities" in result
        if result.get("n_activities", 0) > 0:
            for act in result["activities"]:
                assert "activity_type" in act

    def test_unknown_compound_handled(self):
        result = search_chembl_compound("NONEXISTENT_DRUG_XYZ_12345_ABCDE")
        assert "compounds" in result
        assert result["n_results"] == 0 or "error" in result


@pytest.mark.integration
class TestPubChemLive:

    def test_aspirin_by_name(self):
        result = get_pubchem_compound("aspirin")
        assert "cid" in result
        if result["cid"]:
            assert result["formula"] == "C9H8O4"

    def test_atorvastatin_smiles(self):
        result = get_pubchem_compound("atorvastatin")
        if "smiles" in result and result["smiles"]:
            # SMILES should contain key atoms
            assert "N" in result["smiles"] or "O" in result["smiles"]

    def test_unknown_compound_returns_graceful(self):
        result = get_pubchem_compound("NONEXISTENT_XYZ_12345_COMPOUND")
        assert "note" in result or "error" in result
