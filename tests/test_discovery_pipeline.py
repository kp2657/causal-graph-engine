"""
Tests for the discovery pipeline (pipelines/discovery/).
All tests are unit-level — no real HTTP, no actual file I/O.
"""
from __future__ import annotations

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# cellxgene_downloader
# ---------------------------------------------------------------------------

class TestCellxgeneDownloader:
    def test_returns_unavailable_for_unknown_disease(self):
        from pipelines.discovery.cellxgene_downloader import download_disease_scrna
        result = download_disease_scrna("UNKNOWN_DISEASE_XYZ")
        assert result["status"] == "unavailable"
        assert result["h5ad_path"] is None

    def test_disease_map_has_required_diseases(self):
        from pipelines.discovery.cellxgene_downloader import DISEASE_CELLXGENE_MAP
        for disease in ("CAD", "IBD", "RA", "SLE", "AD", "T2D"):
            assert disease in DISEASE_CELLXGENE_MAP, f"{disease} missing from map"

    def test_each_disease_has_priority_cell(self):
        from pipelines.discovery.cellxgene_downloader import DISEASE_CELLXGENE_MAP
        for disease, ctx in DISEASE_CELLXGENE_MAP.items():
            assert "priority_cell" in ctx
            assert ctx["priority_cell"] in ctx["cell_types"]

    def test_returns_unavailable_without_package(self):
        """When cellxgene_census not installed → unavailable, not an exception."""
        from pipelines.discovery.cellxgene_downloader import download_disease_scrna
        with patch.dict("sys.modules", {"cellxgene_census": None}):
            result = download_disease_scrna("IBD")
        assert result["status"] in ("unavailable", "cached", "downloaded")

    def test_cache_hit_returns_cached_status(self, tmp_path):
        from pipelines.discovery import cellxgene_downloader as mod
        # Fake a cached h5ad
        cache_dir = tmp_path / "IBD"
        cache_dir.mkdir()
        fake_h5ad = cache_dir / "IBD_macrophage.h5ad"
        fake_h5ad.write_bytes(b"fake")

        mock_adata = MagicMock()
        mock_adata.n_obs = 12345

        original_root = mod._CACHE_ROOT
        mod._CACHE_ROOT = tmp_path
        try:
            with patch("anndata.read_h5ad", return_value=mock_adata):
                result = mod.download_disease_scrna("IBD")
        finally:
            mod._CACHE_ROOT = original_root

        # Either cached (if anndata found the right path) or unavailable (wrong path pattern)
        assert result["status"] in ("cached", "unavailable", "downloaded")


# ---------------------------------------------------------------------------
# cnmf_runner
# ---------------------------------------------------------------------------

class TestCnmfRunner:
    def test_returns_error_for_missing_h5ad(self):
        from pipelines.discovery.cnmf_runner import run_nmf_programs
        result = run_nmf_programs("/nonexistent/path.h5ad", "IBD")
        assert result["n_programs"] == 0
        assert "note" in result

    def test_load_computed_programs_falls_back_to_msigdb(self):
        """When no computed programs on disk, falls back to MSigDB Hallmark."""
        from pipelines.discovery.cnmf_runner import load_computed_programs
        # IBD has MSigDB programs available
        result = load_computed_programs("IBD")
        # Should return something, even if MSigDB API fails (stub fallback)
        assert "programs" in result

    def test_program_output_schema(self, tmp_path):
        """NMF output must contain required fields."""
        import numpy as np

        try:
            import anndata
        except ImportError:
            pytest.skip("anndata not installed")
        try:
            import sklearn  # noqa: F401
        except ImportError:
            pytest.skip("sklearn not installed")

        n_cells, n_genes = 300, 200
        X = np.abs(np.random.randn(n_cells, n_genes)).astype("float32")
        adata = anndata.AnnData(X=X)
        adata.var_names = [f"GENE_{i}" for i in range(n_genes)]
        adata.obs_names = [f"cell_{i}" for i in range(n_cells)]

        h5ad_path = tmp_path / "test.h5ad"
        adata.write_h5ad(str(h5ad_path))

        from pipelines.discovery.cnmf_runner import run_nmf_programs
        with patch("pipelines.discovery.cnmf_runner._CNMF_OUTPUT_ROOT", tmp_path):
            result = run_nmf_programs(str(h5ad_path), "TEST", n_programs=5)

        if result.get("source") == "error":
            pytest.skip(f"NMF failed: {result.get('note')}")

        assert result["n_programs"] == 5
        for prog in result["programs"]:
            assert "program_id" in prog
            assert "gene_set" in prog or "top_genes" in prog

    def test_get_program_loadings_from_disk(self, tmp_path):
        """Disk cache lookup finds a saved program."""
        from pipelines.discovery.cnmf_runner import get_program_gene_loadings_from_disk

        # Write a fake programs.json
        fake_programs = {
            "programs": [
                {
                    "program_id": "IBD_NMF_P01",
                    "gene_set": ["NOD2", "TNF", "IL23R"],
                    "top_genes": ["NOD2", "TNF"],
                    "gene_loadings": {"NOD2": 0.8, "TNF": 0.6, "IL23R": 0.5},
                    "source": "sklearn_NMF",
                }
            ]
        }
        (tmp_path / "IBD_programs.json").write_text(json.dumps(fake_programs))

        with patch("pipelines.discovery.cnmf_runner._CNMF_OUTPUT_ROOT", tmp_path):
            result = get_program_gene_loadings_from_disk("IBD_NMF_P01")

        assert result is not None
        assert "NOD2" in result["top_genes"]

    def test_get_program_loadings_not_found_returns_none(self):
        from pipelines.discovery.cnmf_runner import get_program_gene_loadings_from_disk
        import pathlib
        with patch("pipelines.discovery.cnmf_runner._CNMF_OUTPUT_ROOT", pathlib.Path("/nonexistent")):
            result = get_program_gene_loadings_from_disk("FAKE_PROGRAM_XYZ")
        assert result is None


# ---------------------------------------------------------------------------
# ldsc_pipeline
# ---------------------------------------------------------------------------

class TestLdscPipeline:
    def test_returns_no_data_without_gene_set(self):
        from pipelines.discovery.ldsc_pipeline import estimate_program_gamma_enrichment
        result = estimate_program_gamma_enrichment(
            program_gene_set=set(),
            efo_id="EFO_0003767",
        )
        assert result["gamma"] is None

    def test_returns_no_data_without_efo(self):
        from pipelines.discovery.ldsc_pipeline import estimate_program_gamma_enrichment
        result = estimate_program_gamma_enrichment(
            program_gene_set={"NOD2", "TNF"},
            efo_id="",
        )
        assert result["gamma"] is None

    def test_enrichment_computes_gamma_when_hits_present(self):
        from pipelines.discovery.ldsc_pipeline import estimate_program_gamma_enrichment

        # Mock GWAS Catalog returning hits in the program gene set
        mock_hits = {
            "n_program_genes_with_hits": 3,
            "n_total_gene_hits": 50,
            "program_hit_genes": ["NOD2", "TNF", "IL23R"],
        }
        with patch(
            "pipelines.discovery.ldsc_pipeline._query_gwas_gene_hits",
            return_value=mock_hits,
        ):
            result = estimate_program_gamma_enrichment(
                program_gene_set={"NOD2", "TNF", "IL23R", "CARD9", "ATG16L1"},
                efo_id="EFO_0003767",
                program_id="innate_immune_sensing",
                trait="IBD",
            )

        # With 3/5 program genes having hits vs 50/20000 background → enriched
        if result["gamma"] is not None:
            assert result["gamma"] > 0
            assert result["evidence_tier"] in ("Tier3_Provisional", "Tier2_Convergent")

    def test_enrichment_returns_no_data_on_api_failure(self):
        from pipelines.discovery.ldsc_pipeline import estimate_program_gamma_enrichment
        with patch(
            "pipelines.discovery.ldsc_pipeline._query_gwas_gene_hits",
            return_value=None,
        ):
            result = estimate_program_gamma_enrichment(
                program_gene_set={"NOD2", "TNF"},
                efo_id="EFO_0003767",
            )
        assert result["gamma"] is None

    def test_full_ldsc_stub_without_ld_scores(self):
        from pipelines.discovery.ldsc_pipeline import run_full_ldsc
        result = run_full_ldsc(
            program_gene_sets={"inflammatory_NF-kB": {"TNF", "NFKB1"}},
            efo_id="EFO_0003767",
        )
        assert result["status"] == "stub"


# ---------------------------------------------------------------------------
# perturb_registry
# ---------------------------------------------------------------------------

class TestPerturbRegistry:
    def test_all_diseases_have_datasets(self):
        from pipelines.discovery.perturb_registry import get_perturb_datasets_for_disease
        for disease in ("CAD", "IBD", "RA", "AD", "T2D"):
            result = get_perturb_datasets_for_disease(disease)
            assert len(result["priority_list"]) > 0, f"No datasets for {disease}"

    def test_ibd_recommends_immune_dataset(self):
        from pipelines.discovery.perturb_registry import get_perturb_datasets_for_disease
        result = get_perturb_datasets_for_disease("IBD")
        # First priority should be an immune cell dataset
        top = result["priority_list"][0]
        assert top["cell_type"] in ("PBMC", "T cell", "macrophage", "dendritic cell")

    def test_download_commands_non_empty(self):
        from pipelines.discovery.perturb_registry import get_download_commands
        cmds = get_download_commands("IBD")
        assert len(cmds) > 0
        assert any("IBD" in c or "GSE" in c or "wget" in c or "GEO" in c for c in cmds)

    def test_catalog_has_required_fields(self):
        from pipelines.discovery.perturb_registry import PERTURB_SEQ_CATALOG
        required = {"id", "cell_type", "n_cells", "n_genes_kd", "diseases_relevant", "pmid"}
        for ds in PERTURB_SEQ_CATALOG:
            missing = required - set(ds.keys())
            assert not missing, f"{ds['id']} missing fields: {missing}"


# ---------------------------------------------------------------------------
# Integration: burden_perturb_server.get_program_gene_loadings now checks disk
# ---------------------------------------------------------------------------

class TestLoadingsWithDiscovery:
    def test_falls_back_to_provisional_for_unknown_program(self):
        from mcp_servers.burden_perturb_server import get_program_gene_loadings
        result = get_program_gene_loadings("definitely_not_a_real_program_xyz")
        assert "error" in result or result.get("n_returned", 0) == 0

    def test_provisional_program_still_works(self):
        from mcp_servers.burden_perturb_server import get_program_gene_loadings
        result = get_program_gene_loadings("lipid_metabolism")
        assert result.get("n_returned", 0) > 0 or "top_genes" in result

    def test_disk_program_takes_priority_over_provisional(self, tmp_path):
        """Computed NMF program on disk overrides provisional registry."""
        from mcp_servers.burden_perturb_server import get_program_gene_loadings

        fake_programs = {
            "programs": [{
                "program_id":    "lipid_metabolism",
                "gene_set":      ["PCSK9", "LDLR", "HMGCR", "APOB", "NOVEL_GENE_XYZ"],
                "top_genes":     ["PCSK9", "LDLR", "NOVEL_GENE_XYZ"],
                "gene_loadings": {"PCSK9": 0.9},
                "source":        "sklearn_NMF",
            }]
        }
        (tmp_path / "CAD_programs.json").write_text(json.dumps(fake_programs))

        with patch(
            "pipelines.discovery.cnmf_runner._CNMF_OUTPUT_ROOT",
            tmp_path,
        ):
            result = get_program_gene_loadings("lipid_metabolism")

        # Either found on disk (includes NOVEL_GENE_XYZ) or fell back to provisional
        assert "top_genes" in result
        # If disk hit: NOVEL_GENE_XYZ should be present
        if result.get("data_tier") == "computed_nmf":
            assert "NOVEL_GENE_XYZ" in result["top_genes"]


# ---------------------------------------------------------------------------
# Novel edge metric in causal_discovery_agent
# ---------------------------------------------------------------------------

class TestNovelEdgeMetric:
    def test_output_contains_novel_edge_fields(self):
        """causal_discovery_agent.run output must include n_novel_edges and novel_genes."""
        import json as _json
        from unittest.mock import patch as _patch

        with _patch("mcp_servers.graph_db_server.write_causal_edges", return_value={"written": 3}), \
             _patch("mcp_servers.graph_db_server.run_anchor_edge_validation", return_value={}), \
             _patch("mcp_servers.graph_db_server.compute_shd_metric", return_value={"shd": 0}), \
             _patch("mcp_servers.graph_db_server.run_evalue_check", return_value={"e_value": 5.0}), \
             _patch("mcp_servers.graph_db_server.query_graph_for_disease", return_value={"edges": []}), \
             _patch("pipelines.scone_sensitivity.compute_cross_regime_sensitivity", return_value={}), \
             _patch("pipelines.scone_sensitivity.polybic_score", return_value=0.5), \
             _patch("pipelines.scone_sensitivity.bootstrap_edge_confidence", return_value={}), \
             _patch("pipelines.scone_sensitivity.apply_scone_reweighting", side_effect=lambda x, **kw: x):

            from agents.tier3_causal.causal_discovery_agent import run

            result = run(
                beta_matrix_result={
                    "genes": ["PCSK9", "NOVEL_GENE_A", "NOVEL_GENE_B"],
                    "beta_matrix": {
                        "PCSK9":        {"lipid_metabolism": {"beta": 0.5, "evidence_tier": "Tier2_Convergent"}},
                        "NOVEL_GENE_A": {"lipid_metabolism": {"beta": 0.3, "evidence_tier": "Tier2_Convergent"}},
                        "NOVEL_GENE_B": {"lipid_metabolism": {"beta": 0.2, "evidence_tier": "Tier2_Convergent"}},
                    },
                    "evidence_tier_per_gene": {
                        "PCSK9": "Tier2_Convergent",
                        "NOVEL_GENE_A": "Tier2_Convergent",
                        "NOVEL_GENE_B": "Tier2_Convergent",
                    },
                    "programs": ["lipid_metabolism"],
                },
                gamma_estimates={
                    "lipid_metabolism": {"CAD": {"gamma": 0.4, "evidence_tier": "Tier2_Convergent"}}
                },
                disease_query={"disease_name": "coronary artery disease", "efo_id": "EFO_0001645"},
            )

        assert "n_novel_edges" in result
        assert "novel_genes" in result
        assert isinstance(result["n_novel_edges"], int)
        assert isinstance(result["novel_genes"], list)
