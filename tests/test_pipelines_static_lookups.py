"""Tests for `pipelines.static_lookups` and the MCP server wiring that
falls back to the live API when the static file is missing.

Each test uses a temp directory with tiny fixture TSVs so the behaviour
can be exercised deterministically without network access.
"""
from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from pipelines.static_lookups import StaticLookups, get_lookups, reset_lookups


# ---------------------------------------------------------------------------
# Fixture files
# ---------------------------------------------------------------------------

_GNOMAD_TSV = textwrap.dedent("""\
    gene\tgene_id\tlof.pLI\tlof.oe_ci.upper\tmis.z_score\tlof.obs\tlof.exp\tmane_select
    CFH\tENSG00000000971\t0.05\t1.12\t-0.4\t8\t14.5\ttrue
    PCSK9\tENSG00000169174\t0.00\t2.02\t1.1\t22\t18.0\ttrue
    HMCN1\tENSG00000143341\t1.00\t0.21\t3.2\t1\t25.0\ttrue
    FOO\tENSG00000009999\tNA\tNA\tNA\tNA\tNA\ttrue
""")

_HGNC_TSV = textwrap.dedent("""\
    hgnc_id\tsymbol\tensembl_gene_id\talias_symbol\tprev_symbol
    HGNC:4883\tCFH\tENSG00000000971\t\tHF1
    HGNC:8621\tPCSK9\tENSG00000169174\tNARC1|FH3\t
    HGNC:19188\tHMCN1\tENSG00000143341\tFIBL-6\t
""")

_REACTOME_TSV = textwrap.dedent("""\
    ENSG00000000971\tR-HSA-166658\thttps://reactome.org/x\tInitial triggering of complement\tIEA\tHomo sapiens
    ENSG00000000971\tR-HSA-977606\thttps://reactome.org/y\tRegulation of Complement cascade\tIEA\tHomo sapiens
    ENSG00000169174\tR-HSA-8963899\thttps://reactome.org/z\tPlasma lipoprotein remodeling\tIEA\tHomo sapiens
    ENSG00000999999\tR-HSA-111\thttps://reactome.org/w\tMouse only\tIEA\tMus musculus
""")


@pytest.fixture()
def static_dir(tmp_path: Path) -> Path:
    (tmp_path / "gnomad_constraint.tsv").write_text(_GNOMAD_TSV)
    (tmp_path / "hgnc_complete.tsv").write_text(_HGNC_TSV)
    (tmp_path / "reactome_ensembl2pathways.tsv").write_text(_REACTOME_TSV)
    return tmp_path


@pytest.fixture()
def lookups(static_dir: Path) -> StaticLookups:
    reset_lookups()
    sl = StaticLookups(static_dir)
    yield sl
    reset_lookups()


# ---------------------------------------------------------------------------
# Per-dataset queries
# ---------------------------------------------------------------------------

def test_gnomad_lookup_returns_drop_in_dict(lookups: StaticLookups):
    rec = lookups.get_gnomad_constraint("CFH")
    assert rec is not None
    assert rec["gene"] == "CFH"
    assert rec["gene_id"] == "ENSG00000000971"
    assert rec["pli"] == pytest.approx(0.05)
    assert rec["loeuf"] == pytest.approx(1.12)
    assert rec["data_source"].startswith("gnomAD")


def test_gnomad_lookup_handles_NA(lookups: StaticLookups):
    rec = lookups.get_gnomad_constraint("FOO")
    assert rec is not None
    assert rec["pli"] is None
    assert rec["loeuf"] is None


def test_gnomad_lookup_unknown_gene_returns_none(lookups: StaticLookups):
    assert lookups.get_gnomad_constraint("GENE_NOT_IN_FIXTURE") is None


def test_gnomad_resolves_via_alias(lookups: StaticLookups):
    # NARC1 is a PCSK9 alias per HGNC fixture; constraint lookup should
    # resolve it back to PCSK9's record.
    rec = lookups.get_gnomad_constraint("NARC1")
    assert rec is not None
    assert rec["gene"] == "PCSK9"


def test_hgnc_symbol_to_ensembl(lookups: StaticLookups):
    assert lookups.get_ensembl_id("CFH") == "ENSG00000000971"
    assert lookups.get_ensembl_id("PCSK9") == "ENSG00000169174"


def test_hgnc_alias_resolution(lookups: StaticLookups):
    assert lookups.get_ensembl_id("NARC1") == "ENSG00000169174"  # PCSK9 alias
    assert lookups.get_ensembl_id("HF1") == "ENSG00000000971"    # CFH prev
    assert lookups.get_ensembl_id("FH3") == "ENSG00000169174"    # PCSK9 alias


def test_hgnc_unknown_symbol_returns_none(lookups: StaticLookups):
    assert lookups.get_ensembl_id("NOT_A_GENE") is None


def test_reactome_lookup_by_symbol(lookups: StaticLookups):
    out = lookups.get_reactome_pathways("CFH")
    assert isinstance(out, list)
    assert len(out) == 2
    assert all(p["species"] == "Homo sapiens" for p in out)
    assert {p["pathway_id"] for p in out} == {"R-HSA-166658", "R-HSA-977606"}


def test_reactome_lookup_by_ensembl_id(lookups: StaticLookups):
    out = lookups.get_reactome_pathways("ENSG00000169174")
    assert out == [{"pathway_id": "R-HSA-8963899",
                    "name": "Plasma lipoprotein remodeling",
                    "species": "Homo sapiens"}]


def test_reactome_excludes_non_human(lookups: StaticLookups):
    # The fixture has a mouse row; no symbol maps to it, but a raw Ensembl
    # ID lookup should still return [] because the loader filters species.
    out = lookups.get_reactome_pathways("ENSG00000999999")
    assert out == []


def test_reactome_symbol_not_in_hgnc_returns_empty_list(lookups: StaticLookups):
    out = lookups.get_reactome_pathways("UNKNOWN_SYMBOL")
    assert out == []


# ---------------------------------------------------------------------------
# Graceful degradation when files are absent
# ---------------------------------------------------------------------------

def test_all_queries_return_none_when_static_dir_empty(tmp_path: Path):
    sl = StaticLookups(tmp_path)
    assert sl.get_gnomad_constraint("CFH") is None
    assert sl.get_ensembl_id("CFH") is None
    assert sl.get_reactome_pathways("CFH") is None


def test_status_reports_load_state(static_dir: Path):
    sl = StaticLookups(static_dir)
    s = sl.status()
    assert s["gnomad_loaded"] is True
    assert s["hgnc_loaded"] is True
    assert s["reactome_loaded"] is True
    assert s["gnomad_n_genes"] >= 3
    assert s["hgnc_n_symbols"] >= 3


def test_status_partial_load_when_one_file_missing(tmp_path: Path):
    (tmp_path / "hgnc_complete.tsv").write_text(_HGNC_TSV)
    sl = StaticLookups(tmp_path)
    s = sl.status()
    assert s["hgnc_loaded"] is True
    assert s["gnomad_loaded"] is False
    assert s["reactome_loaded"] is False


# ---------------------------------------------------------------------------
# MCP-server wiring: static-first with graceful live fallback
# ---------------------------------------------------------------------------

def test_ukb_wes_get_gnomad_constraint_uses_static_when_available(static_dir, monkeypatch):
    """`get_gnomad_constraint` should serve from the static file without
    touching the live gnomAD GraphQL endpoint when the gene is present."""
    monkeypatch.setenv("STATIC_DATA_DIR", str(static_dir))
    reset_lookups()
    import mcp_servers.ukb_wes_server as ukb

    call_counter = {"n": 0}

    def _fail_if_called(query, variables):
        call_counter["n"] += 1
        raise AssertionError("live gnomAD GraphQL should not be invoked")

    with patch.object(ukb, "_gnomad_gql", _fail_if_called):
        # Call the bare cached function to bypass pre-existing SQLite cache
        # entries: use the decorator's wrapped fn with a disease-specific gene.
        from pipelines.api_cache import get_cache, _make_key
        cache = get_cache()
        key = _make_key("get_gnomad_constraint", ("PCSK9",), {})
        cache.invalidate("get_gnomad_constraint", ("PCSK9",), {})
        result = ukb.get_gnomad_constraint("PCSK9")
    assert call_counter["n"] == 0
    assert result["gene"] == "PCSK9"
    assert result["gene_id"] == "ENSG00000169174"
    reset_lookups()


def test_pathways_kg_get_reactome_uses_static_when_available(static_dir, monkeypatch):
    monkeypatch.setenv("STATIC_DATA_DIR", str(static_dir))
    reset_lookups()
    import mcp_servers.pathways_kg_server as pks
    from pipelines.api_cache import get_cache
    get_cache().invalidate("get_reactome_pathways_for_gene", ("CFH",), {})
    get_cache().invalidate(
        "get_reactome_pathways_for_gene", ("CFH",), {"species": "Homo sapiens"},
    )

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("live Reactome ContentService should not be invoked")

    with patch.object(pks.httpx, "get", _fail_if_called):
        result = pks.get_reactome_pathways_for_gene("CFH")
    assert result["n_pathways"] == 2
    assert result["source"].startswith("Reactome Ensembl2Reactome")
    reset_lookups()


def test_pathways_kg_falls_back_to_live_when_gene_not_in_static(static_dir, monkeypatch):
    monkeypatch.setenv("STATIC_DATA_DIR", str(static_dir))
    reset_lookups()
    import mcp_servers.pathways_kg_server as pks
    from pipelines.api_cache import get_cache
    get_cache().invalidate("get_reactome_pathways_for_gene", ("NOVEL_GENE",), {})

    class _FakeResp:
        status_code = 200
        def raise_for_status(self): pass
        def json(self):
            return {"results": [
                {"typeName": "Pathway", "entries": [
                    {"stId": "R-HSA-LIVE", "name": "Live path", "species": ["Homo sapiens"]},
                ]},
            ]}

    with patch.object(pks.httpx, "get", lambda *a, **k: _FakeResp()):
        result = pks.get_reactome_pathways_for_gene("NOVEL_GENE")
    assert result["n_pathways"] == 1
    assert result["pathways"][0]["pathway_id"] == "R-HSA-LIVE"
    reset_lookups()


def test_get_lookups_singleton_respects_env_var(static_dir, monkeypatch):
    monkeypatch.setenv("STATIC_DATA_DIR", str(static_dir))
    reset_lookups()
    sl = get_lookups()
    assert sl.get_ensembl_id("CFH") == "ENSG00000000971"
    reset_lookups()
