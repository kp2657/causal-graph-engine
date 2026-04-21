"""Unit tests for ChEMBL label → HGNC resolution (GPS downstream)."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_resolve_gps_putative_target_labels_to_hgnc_batch():
    from mcp_servers import chemistry_server as cs

    cs._CHEMBL_LABEL_TO_GENE.clear()

    def _fake(label: str) -> dict:
        m = {
            "EGFR": {"gene_symbol": "EGFR", "target_chembl_id": "CHEMBL203", "source": "mock"},
            "weird label": {"gene_symbol": None, "target_chembl_id": None, "source": "mock"},
        }
        return {"query": label, **m.get(label, {"gene_symbol": "MAPK14", "target_chembl_id": "CHEMBL4282", "source": "mock"})}

    with patch.object(cs, "resolve_chembl_target_label_to_hgnc", side_effect=_fake):
        out = cs.resolve_gps_putative_target_labels_to_hgnc(
            ["EGFR", "weird label", "Mitogen-activated protein kinase 14"],
            max_labels=50,
        )

    assert "EGFR" in out["genes"]
    assert "MAPK14" in out["genes"]
    assert out["n_resolved"] == 2
    assert out["n_unresolved"] >= 1


def test_resolve_chembl_hgnc_like_token_cached():
    from mcp_servers import chemistry_server as cs

    cs._CHEMBL_LABEL_TO_GENE.clear()
    r = cs.resolve_chembl_target_label_to_hgnc("PCSK9")
    assert r["gene_symbol"] == "PCSK9"
    assert r["source"] == "hgnc_like_token"
    r2 = cs.resolve_chembl_target_label_to_hgnc("PCSK9")
    assert r2["source"] == "cache"
