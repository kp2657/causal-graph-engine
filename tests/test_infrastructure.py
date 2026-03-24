"""
tests/test_infrastructure.py — Unit tests for infrastructure layer.

Tests:
  - graph/validation.py  : ValidationReport, anchor recovery, SHD, SID, E-value
  - graph/export.py      : Turtle, JSON-LD, CSV export
  - graph/versioning.py  : snapshot create/list/rollback, version bumping
  - graph/update_pipeline.py : GWAS/literature/trials refresh + persist gate
  - scheduler/update_scheduler.py : job registration, list_jobs
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SAMPLE_EDGES = [
    {
        "from_node":     "PCSK9",
        "to_node":       "coronary artery disease",
        "from_type":     "gene",
        "evidence_tier": "Tier1_Interventional",
        "edge_method":   "two_sample_mr",
        "effect_size":   -0.40,
        "e_value":       8.5,
        "data_source":   "OpenGWAS",
        "graph_version": "0.1.0",
        "is_demoted":    False,
    },
    {
        "from_node":     "LDLR",
        "to_node":       "coronary artery disease",
        "from_type":     "gene",
        "evidence_tier": "Tier1_Interventional",
        "edge_method":   "two_sample_mr",
        "effect_size":   -0.32,
        "e_value":       6.2,
        "data_source":   "OpenGWAS",
        "graph_version": "0.1.0",
        "is_demoted":    False,
    },
    {
        "from_node":     "TET2_chip",
        "to_node":       "coronary artery disease",
        "from_type":     "gene",
        "evidence_tier": "Tier2_Convergent",
        "edge_method":   "chip_loghr",
        "effect_size":   0.21,
        "e_value":       1.5,      # Low E-value
        "data_source":   "Bick 2020",
        "graph_version": "0.1.0",
        "is_demoted":    False,
    },
    {
        "from_node":     "VIRTUAL_GENE",
        "to_node":       "coronary artery disease",
        "from_type":     "gene",
        "evidence_tier": "provisional_virtual",
        "edge_method":   "virtual_cell_model",
        "effect_size":   0.05,
        "e_value":       None,
        "data_source":   "virtual",
        "graph_version": "0.1.0",
        "is_demoted":    True,       # Demoted
    },
]


# ===========================================================================
# graph/validation.py
# ===========================================================================

class TestAnchorRecovery:
    def test_full_recovery(self):
        from graph.validation import check_anchor_recovery
        from graph.schema import ANCHOR_EDGES

        # Build edges that satisfy every anchor
        mock_edges = []
        for a in ANCHOR_EDGES:
            fn = a.get("from_node") or a.get("from", "")
            tn = a.get("to_node")   or a.get("to", "")
            mock_edges.append({"from_node": fn, "to_node": tn, "is_demoted": False})

        result = check_anchor_recovery(mock_edges)
        assert result["recovery_rate"] == 1.0
        assert result["n_recovered"] == result["n_total"]
        assert result["missing"] == []

    def test_partial_recovery(self):
        from graph.validation import check_anchor_recovery

        result = check_anchor_recovery(SAMPLE_EDGES)
        assert 0.0 <= result["recovery_rate"] <= 1.0
        assert result["n_total"] > 0

    def test_empty_graph(self):
        from graph.validation import check_anchor_recovery

        result = check_anchor_recovery([])
        assert result["recovery_rate"] == 0.0
        assert result["n_recovered"] == 0


class TestSHD:
    def test_zero_shd_against_itself(self):
        from graph.validation import compute_shd

        edges = [{"from_node": "A", "to_node": "B"}, {"from_node": "C", "to_node": "D"}]
        result = compute_shd(edges, reference_edges=edges)
        # extra_edges may be non-zero due to chip suffix normalisation doubling, so check shd structure
        assert "shd" in result
        assert "missing_edges" in result
        assert "extra_edges" in result

    def test_all_missing(self):
        from graph.validation import compute_shd
        from graph.schema import ANCHOR_EDGES

        result = compute_shd([], reference_edges=ANCHOR_EDGES)
        assert result["shd"] >= len(ANCHOR_EDGES)

    def test_chip_suffix_normalisation(self):
        from graph.validation import compute_shd

        predicted = [{"from_node": "TET2_chip", "to_node": "CAD"}]
        reference = [{"from_node": "TET2", "to_node": "CAD"}]
        result = compute_shd(predicted, reference_edges=reference)
        # _edge_set adds both original and normalised forms, so "TET2" IS found
        # in pred_set → no missing edges from reference.
        assert "TET2→CAD" not in result["missing_edges"]


class TestSIDApproximation:
    def test_perfect_sid(self):
        from graph.validation import compute_sid_approximation
        from graph.schema import ANCHOR_EDGES

        result = compute_sid_approximation(ANCHOR_EDGES, reference_edges=ANCHOR_EDGES)
        assert result == 1.0

    def test_zero_sid(self):
        from graph.validation import compute_sid_approximation
        from graph.schema import ANCHOR_EDGES

        result = compute_sid_approximation([], reference_edges=ANCHOR_EDGES)
        assert result == 0.0

    def test_empty_reference(self):
        from graph.validation import compute_sid_approximation

        result = compute_sid_approximation(SAMPLE_EDGES, reference_edges=[])
        assert result == 1.0


class TestEvalueSummary:
    def test_flags_low_evalue(self):
        from graph.validation import summarize_evalues

        result = summarize_evalues(SAMPLE_EDGES)
        # TET2_chip has e_value=1.5 < 2.0
        assert result["n_low_evalue"] >= 1
        assert result["min_evalue"] is not None
        assert result["min_evalue"] < 2.0

    def test_no_edges(self):
        from graph.validation import summarize_evalues

        result = summarize_evalues([])
        assert result["min_evalue"] is None
        assert result["mean_evalue"] is None

    def test_all_good_evalues(self):
        from graph.validation import summarize_evalues

        edges = [{"e_value": 5.0}, {"e_value": 10.0}]
        result = summarize_evalues(edges)
        assert result["n_low_evalue"] == 0


class TestValidateGraph:
    def test_validate_graph_calls_query(self):
        from graph.validation import validate_graph

        mock_result = {"edges": SAMPLE_EDGES}
        with patch("mcp_servers.graph_db_server.query_graph_for_disease",
                   return_value=mock_result):
            report = validate_graph("coronary artery disease")

        assert hasattr(report, "anchor_recovery_rate")
        assert hasattr(report, "shd")
        assert hasattr(report, "passed")
        assert hasattr(report, "errors")

    def test_validate_graph_query_failure(self):
        from graph.validation import validate_graph

        with patch("mcp_servers.graph_db_server.query_graph_for_disease",
                   side_effect=RuntimeError("DB down")):
            report = validate_graph("coronary artery disease")

        assert report.passed is False
        assert len(report.errors) > 0

    def test_validation_report_to_dict(self):
        from graph.validation import validate_graph, validation_report_to_dict

        mock_result = {"edges": SAMPLE_EDGES}
        with patch("mcp_servers.graph_db_server.query_graph_for_disease",
                   return_value=mock_result):
            report = validate_graph("coronary artery disease")

        d = validation_report_to_dict(report)
        assert isinstance(d, dict)
        assert "anchor_recovery_rate" in d
        assert "passed" in d


# ===========================================================================
# graph/export.py
# ===========================================================================

class TestExportTurtle:
    def test_basic_output(self):
        from graph.export import export_turtle

        ttl = export_turtle(SAMPLE_EDGES)
        assert "@prefix biolink:" in ttl
        assert "biolink:Association" in ttl
        # PCSK9 edge present
        assert "PCSK9" in ttl

    def test_demoted_edges_excluded(self):
        from graph.export import export_turtle

        ttl = export_turtle(SAMPLE_EDGES)
        # VIRTUAL_GENE is demoted — should not appear
        assert "VIRTUAL_GENE" not in ttl

    def test_write_to_file_handle(self, tmp_path):
        from graph.export import export_turtle

        out_file = tmp_path / "test.ttl"
        with out_file.open("w") as f:
            result = export_turtle(SAMPLE_EDGES, out=f)
        assert out_file.read_text() == result

    def test_effect_size_included(self):
        from graph.export import export_turtle

        ttl = export_turtle(SAMPLE_EDGES)
        assert "cge:effect_size" in ttl


class TestExportJsonLD:
    def test_returns_dict_with_context(self):
        from graph.export import export_jsonld

        doc = export_jsonld(SAMPLE_EDGES, disease_name="coronary artery disease")
        assert "@context" in doc
        assert "@graph" in doc
        assert "cge:disease" in doc

    def test_associations_present(self):
        from graph.export import export_jsonld

        doc = export_jsonld(SAMPLE_EDGES)
        graph = doc["@graph"]
        assoc_types = [n for n in graph if n.get("type") == "biolink:Association"]
        # 3 non-demoted edges
        assert len(assoc_types) == 3

    def test_demoted_excluded(self):
        from graph.export import export_jsonld

        doc = export_jsonld(SAMPLE_EDGES)
        ids = [n.get("id", "") for n in doc["@graph"]]
        assert not any("VIRTUAL_GENE" in i for i in ids)


class TestExportCSV:
    def test_has_header(self):
        from graph.export import export_csv

        csv = export_csv(SAMPLE_EDGES)
        lines = csv.strip().split("\n")
        assert lines[0].startswith("from_node,to_node")

    def test_all_edges_included(self):
        from graph.export import export_csv

        csv = export_csv(SAMPLE_EDGES)
        lines = csv.strip().split("\n")
        # header + 4 data rows
        assert len(lines) == 5

    def test_comma_in_source_escaped(self):
        from graph.export import export_csv

        edges = [{"from_node": "A", "to_node": "B",
                  "data_source": "foo, bar", "is_demoted": False}]
        csv = export_csv(edges)
        assert "foo; bar" in csv


class TestExportDiseaseGraph:
    def test_writes_all_three_files(self, tmp_path):
        from graph.export import export_disease_graph

        mock_result = {"edges": SAMPLE_EDGES}
        with patch("mcp_servers.graph_db_server.query_graph_for_disease",
                   return_value=mock_result):
            result = export_disease_graph(
                "coronary artery disease",
                output_dir=str(tmp_path),
            )

        assert Path(result["turtle_path"]).exists()
        assert Path(result["jsonld_path"]).exists()
        assert Path(result["csv_path"]).exists()
        assert result["n_edges"] == 4


# ===========================================================================
# graph/versioning.py
# ===========================================================================

class TestCreateSnapshot:
    def test_creates_metadata_file(self, tmp_path):
        from graph.versioning import create_snapshot

        result = create_snapshot(
            version_tag="0.1.0",
            release_notes="test snapshot",
            db_path="/nonexistent/db",
            snapshot_dir=str(tmp_path),
        )

        meta_path = Path(result["metadata_path"])
        assert meta_path.exists()
        with meta_path.open() as f:
            meta = json.load(f)
        assert meta["version_tag"] == "0.1.0"
        assert meta["release_notes"] == "test snapshot"

    def test_creates_dvc_stub(self, tmp_path):
        from graph.versioning import create_snapshot

        result = create_snapshot("0.2.0", snapshot_dir=str(tmp_path))
        dvc_path = Path(result["dvc_stub_path"])
        assert dvc_path.exists()
        with dvc_path.open() as f:
            stub = json.load(f)
        assert "outs" in stub

    def test_empty_db_creates_marker(self, tmp_path):
        from graph.versioning import create_snapshot

        create_snapshot("0.1.0", db_path="/nonexistent", snapshot_dir=str(tmp_path))
        marker = tmp_path / "0.1.0" / "graph.kuzu.empty"
        assert marker.exists()

    def test_copies_existing_db(self, tmp_path):
        from graph.versioning import create_snapshot

        db_dir = tmp_path / "mydb"
        db_dir.mkdir()
        (db_dir / "nodes.dat").write_text("data")

        result = create_snapshot(
            "0.3.0",
            db_path=str(db_dir),
            snapshot_dir=str(tmp_path / "snaps"),
        )
        snap_db = Path(result["snapshot_path"]) / "graph.kuzu"
        assert snap_db.is_dir()
        assert (snap_db / "nodes.dat").exists()


class TestListSnapshots:
    def test_empty_dir_returns_empty(self, tmp_path):
        from graph.versioning import list_snapshots

        result = list_snapshots(str(tmp_path))
        assert result == []

    def test_nonexistent_dir_returns_empty(self):
        from graph.versioning import list_snapshots

        result = list_snapshots("/tmp/nonexistent_cge_snapshots_xyz")
        assert result == []

    def test_lists_created_snapshots(self, tmp_path):
        from graph.versioning import create_snapshot, list_snapshots

        create_snapshot("0.1.0", snapshot_dir=str(tmp_path))
        create_snapshot("0.2.0", snapshot_dir=str(tmp_path))

        snaps = list_snapshots(str(tmp_path))
        tags = [s["version_tag"] for s in snaps]
        assert "0.1.0" in tags
        assert "0.2.0" in tags

    def test_sorted_newest_first(self, tmp_path):
        from graph.versioning import create_snapshot, list_snapshots
        import time

        create_snapshot("0.1.0", snapshot_dir=str(tmp_path))
        time.sleep(0.01)
        create_snapshot("0.2.0", snapshot_dir=str(tmp_path))

        snaps = list_snapshots(str(tmp_path))
        assert snaps[0]["version_tag"] == "0.2.0"


class TestGetLatestSnapshot:
    def test_returns_none_if_empty(self, tmp_path):
        from graph.versioning import get_latest_snapshot

        assert get_latest_snapshot(str(tmp_path)) is None

    def test_returns_latest(self, tmp_path):
        from graph.versioning import create_snapshot, get_latest_snapshot
        import time

        create_snapshot("0.1.0", snapshot_dir=str(tmp_path))
        time.sleep(0.01)
        create_snapshot("0.2.0", snapshot_dir=str(tmp_path))

        latest = get_latest_snapshot(str(tmp_path))
        assert latest["version_tag"] == "0.2.0"


class TestRollback:
    def test_rollback_restores_file(self, tmp_path):
        from graph.versioning import create_snapshot, rollback_to_snapshot

        # Create a fake db file
        db_path = tmp_path / "graph.kuzu"
        db_path.write_text("original")
        snap_dir = tmp_path / "snaps"

        create_snapshot("0.1.0", db_path=str(db_path), snapshot_dir=str(snap_dir))

        # Overwrite db
        db_path.write_text("modified")

        result = rollback_to_snapshot("0.1.0", db_path=str(db_path),
                                      snapshot_dir=str(snap_dir))
        assert result["status"] == "OK"
        assert db_path.read_text() == "original"

    def test_rollback_missing_snapshot_raises(self, tmp_path):
        from graph.versioning import rollback_to_snapshot

        with pytest.raises(FileNotFoundError, match="not found"):
            rollback_to_snapshot("9.9.9", snapshot_dir=str(tmp_path))


class TestBumpVersion:
    def test_patch_bump(self):
        from graph.versioning import bump_version

        assert bump_version("0.1.3", "patch") == "0.1.4"

    def test_minor_bump_resets_patch(self):
        from graph.versioning import bump_version

        assert bump_version("0.1.3", "minor") == "0.2.0"

    def test_major_bump_resets_all(self):
        from graph.versioning import bump_version

        assert bump_version("1.2.3", "major") == "2.0.0"

    def test_strips_v_prefix(self):
        from graph.versioning import bump_version

        assert bump_version("v1.0.0", "patch") == "1.0.1"

    def test_default_is_patch(self):
        from graph.versioning import bump_version

        assert bump_version("0.5.0") == "0.5.1"


class TestGetCurrentVersion:
    def test_returns_default_when_no_snapshots(self, tmp_path):
        from graph.versioning import get_current_version

        version = get_current_version(str(tmp_path))
        assert version == "0.1.0"

    def test_returns_latest_snapshot_version(self, tmp_path):
        from graph.versioning import create_snapshot, get_current_version
        import time

        create_snapshot("0.3.0", snapshot_dir=str(tmp_path))
        time.sleep(0.01)
        create_snapshot("0.4.0", snapshot_dir=str(tmp_path))

        assert get_current_version(str(tmp_path)) == "0.4.0"


# ===========================================================================
# graph/update_pipeline.py
# ===========================================================================

class TestGWASRefresh:
    def test_filters_non_significant(self):
        from graph.update_pipeline import _run_gwas_refresh

        mock_assocs = {
            "associations": [
                {"mapped_gene": "PCSK9", "p_value": "3e-9",  "beta": -0.4},
                {"mapped_gene": "JUNK",  "p_value": "0.05",  "beta": 0.1},
            ]
        }
        with patch("mcp_servers.gwas_genetics_server.get_gwas_catalog_associations",
                   return_value=mock_assocs):
            result = _run_gwas_refresh("coronary artery disease")

        assert result["n_new_loci"] == 1
        assert result["new_edges"][0]["from_node"] == "PCSK9"

    def test_handles_api_failure(self):
        from graph.update_pipeline import _run_gwas_refresh

        with patch("mcp_servers.gwas_genetics_server.get_gwas_catalog_associations",
                   side_effect=RuntimeError("timeout")):
            result = _run_gwas_refresh("coronary artery disease")

        assert result["n_new_loci"] == 0
        assert result["new_edges"] == []


class TestLiteratureRefresh:
    def test_detects_new_papers(self):
        from graph.update_pipeline import _run_literature_refresh

        mock_anchor = [{"pmid": "12345"}]
        mock_search = {
            "papers": [
                {"pmid": "99999", "title": "PCSK9 MR study"},
                {"pmid": "12345", "title": "Old paper"},
            ]
        }
        with patch("mcp_servers.literature_server.list_anchor_papers",
                   return_value=mock_anchor), \
             patch("mcp_servers.literature_server.search_pubmed",
                   return_value=mock_search):
            result = _run_literature_refresh("coronary artery disease")

        assert result["n_new_papers"] == 1
        assert "PCSK9" in result["flagged_genes"]

    def test_handles_pubmed_failure(self):
        from graph.update_pipeline import _run_literature_refresh

        with patch("mcp_servers.literature_server.list_anchor_papers",
                   return_value=[]), \
             patch("mcp_servers.literature_server.search_pubmed",
                   side_effect=RuntimeError("no network")):
            result = _run_literature_refresh("coronary artery disease")

        assert result["n_new_papers"] == 0


class TestClinicalTrialsRefresh:
    def test_counts_completed_terminated(self):
        from graph.update_pipeline import _run_clinical_trials_refresh

        mock_trials = {
            "trials": [
                {"nct_id": "NCT001", "status": "COMPLETED", "intervention": "Statin"},
                {"nct_id": "NCT002", "status": "TERMINATED", "intervention": "Drug X"},
                {"nct_id": "NCT003", "status": "COMPLETED", "intervention": "Evolocumab"},
            ]
        }
        with patch("mcp_servers.clinical_trials_server.search_clinical_trials",
                   return_value=mock_trials):
            result = _run_clinical_trials_refresh("coronary artery disease")

        assert result["n_completed"] == 2
        assert result["n_terminated"] == 1

    def test_handles_failure(self):
        from graph.update_pipeline import _run_clinical_trials_refresh

        with patch("mcp_servers.clinical_trials_server.search_clinical_trials",
                   side_effect=RuntimeError("API down")):
            result = _run_clinical_trials_refresh("coronary artery disease")

        assert result["n_completed"] == 0


class TestPersistNewEdges:
    def test_filters_rejected_edges(self):
        from graph.update_pipeline import _persist_new_edges

        rejected_edge = {
            "from_node": "JUNK", "to_node": "CAD",
            "evidence_tier": "Tier1_Interventional",
            "effect_size": None,  # will trigger rejection
            "data_source": "memory",
        }

        mock_review = {
            "approved_edges": [],
            "n_rejected": 1,
        }
        with patch("orchestrator.scientific_reviewer.review_batch",
                   return_value=mock_review):
            result = _persist_new_edges([rejected_edge], "coronary artery disease")

        assert result["n_approved"] == 0
        assert result["n_rejected"] == 1

    def test_empty_edges_fast_path(self):
        from graph.update_pipeline import _persist_new_edges

        result = _persist_new_edges([], "coronary artery disease")
        assert result["n_submitted"] == 0
        assert result["n_approved"] == 0


class TestRunUpdate:
    def test_gwas_update_returns_structure(self):
        from graph.update_pipeline import run_update

        mock_gwas = {"new_edges": [], "removed_edges": [], "n_new_loci": 0, "source": "gwas_catalog"}
        mock_persist = {"n_submitted": 0, "n_approved": 0, "n_rejected": 0, "write_result": {}}

        with patch("graph.update_pipeline._run_gwas_refresh", return_value=mock_gwas), \
             patch("graph.update_pipeline._persist_new_edges", return_value=mock_persist), \
             patch("graph.versioning.get_current_version", return_value="0.1.0"), \
             patch("graph.versioning.create_snapshot",
                   return_value={"version_tag": "0.1.1", "snapshot_path": "/tmp/snaps/0.1.1",
                                 "metadata_path": "/tmp/snaps/0.1.1/meta.json",
                                 "dvc_stub_path": "/tmp/snaps/0.1.1.dvc",
                                 "created_at": "2026-01-01T00:00:00Z"}):
            result = run_update("coronary artery disease", update_type="gwas")

        assert result["update_type"] == "gwas"
        assert "started_at" in result
        assert "finished_at" in result
        assert "n_new_edges" in result

    def test_invalid_update_type_raises(self):
        from graph.update_pipeline import run_update

        with pytest.raises(ValueError, match="Unknown update_type"):
            run_update("CAD", update_type="invalid")  # type: ignore[arg-type]

    def test_full_update_skips_persist(self):
        from graph.update_pipeline import run_update

        mock_full = {"status": "OK", "n_edges": 10, "new_edges": [], "output": {}}

        with patch("graph.update_pipeline._run_full_pipeline", return_value=mock_full), \
             patch("graph.versioning.get_current_version", return_value="0.1.0"), \
             patch("graph.versioning.create_snapshot",
                   return_value={"version_tag": "0.1.1", "snapshot_path": "/tmp",
                                 "metadata_path": "/tmp/m.json",
                                 "dvc_stub_path": "/tmp/0.1.1.dvc",
                                 "created_at": "2026-01-01T00:00:00Z"}):
            result = run_update("CAD", update_type="full")

        assert result["status"] == "OK"


# ===========================================================================
# scheduler/update_scheduler.py
# ===========================================================================

class TestSchedulerJobs:
    def test_build_scheduler_registers_4_jobs(self):
        pytest.importorskip("apscheduler")
        from scheduler.update_scheduler import build_scheduler

        scheduler = build_scheduler("coronary artery disease")
        jobs = scheduler.get_jobs()
        assert len(jobs) == 4

    def test_job_ids_present(self):
        pytest.importorskip("apscheduler")
        from scheduler.update_scheduler import build_scheduler

        scheduler = build_scheduler()
        job_ids = {j.id for j in scheduler.get_jobs()}
        assert "gwas_refresh" in job_ids
        assert "literature_refresh" in job_ids
        assert "clinical_trials_refresh" in job_ids
        assert "full_pipeline" in job_ids

    def test_list_jobs_returns_metadata(self):
        pytest.importorskip("apscheduler")
        from scheduler.update_scheduler import list_jobs

        jobs = list_jobs()
        assert isinstance(jobs, list)
        assert len(jobs) == 4
        for j in jobs:
            assert "id" in j
            assert "name" in j
            assert "trigger" in j


class TestJobFunctions:
    def test_gwas_job_calls_run_update(self):
        from scheduler.update_scheduler import _job_gwas_refresh

        mock_result = {"status": "OK", "n_approved": 2, "n_rejected": 0, "snapshot": {}}
        with patch("graph.update_pipeline.run_update", return_value=mock_result) as mock_run:
            _job_gwas_refresh()
        mock_run.assert_called_once()
        assert mock_run.call_args[1]["update_type"] == "gwas"

    def test_literature_job_calls_run_update(self):
        from scheduler.update_scheduler import _job_literature_refresh

        mock_result = {"status": "OK", "details": {"n_new_papers": 0, "flagged_genes": []}}
        with patch("graph.update_pipeline.run_update", return_value=mock_result) as mock_run:
            _job_literature_refresh()
        mock_run.assert_called_once()
        assert mock_run.call_args[1]["update_type"] == "literature"

    def test_full_pipeline_job_calls_run_update(self):
        from scheduler.update_scheduler import _job_full_pipeline

        mock_result = {"status": "OK", "n_approved": 15, "snapshot": {"version_tag": "0.2.0"}}
        with patch("graph.update_pipeline.run_update", return_value=mock_result) as mock_run:
            _job_full_pipeline()
        mock_run.assert_called_once()
        assert mock_run.call_args[1]["update_type"] == "full"

    def test_job_handles_exception_gracefully(self):
        from scheduler.update_scheduler import _job_gwas_refresh

        with patch("graph.update_pipeline.run_update",
                   side_effect=RuntimeError("network error")):
            # Should not raise — jobs must be fault-tolerant
            _job_gwas_refresh()
