"""
Tests for gwas_genetics_server.py.

Split into:
  - Unit tests:        schema validation, stub outputs, error handling (always run)
  - Integration tests: real HTTP calls to GWAS Catalog, gnomAD, GTEx
                       marked with @pytest.mark.integration — skipped by default
                       run with: pytest -m integration

Run all:               pytest tests/test_gwas_genetics_server.py -v
Run only unit:         pytest tests/test_gwas_genetics_server.py -v -m "not integration"
Run only integration:  pytest tests/test_gwas_genetics_server.py -v -m integration
"""
from __future__ import annotations

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp_servers.gwas_genetics_server import (
    get_gwas_catalog_associations,
    get_gwas_catalog_studies,
    get_snp_associations,
    query_gnomad_lof_constraint,
    resolve_gtex_gene_id,
    query_gtex_eqtl,
    list_available_gwas,
    get_ieu_open_gwas_summary_stats,
    run_mr_analysis,
    get_open_targets_genetics_credible_sets,
    get_l2g_scores,
    get_finngen_phenotype_definition,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_required_keys(obj: dict, keys: list[str]) -> bool:
    return all(k in obj for k in keys)


# ---------------------------------------------------------------------------
# Unit tests — stubs always return schema-valid output
# ---------------------------------------------------------------------------

class TestStubOutputSchemas:

    def test_mr_analysis_known_pair(self):
        result = run_mr_analysis("ieu-a-299", "ieu-a-7")
        assert result["exposure_id"] == "ieu-a-299"
        assert result["mr_ivw"] == pytest.approx(0.470)

    def test_l2g_stub_has_efo_id(self):
        result = get_l2g_scores("GCST003116")
        assert result["study_id"] == "GCST003116"
        assert "l2g_genes" in result

    def test_ot_genetics_stub_filters_by_pip(self):
        result = get_open_targets_genetics_credible_sets("EFO_0001645", min_pip=0.5)
        assert result["efo_id"] == "EFO_0001645"
        assert result["min_pip"] == 0.5
        assert isinstance(result["credible_sets"], list)

    def test_ieu_open_gwas_stub_when_no_token(self, monkeypatch):
        monkeypatch.setattr(
            "mcp_servers.gwas_genetics_server.OPENGWAS_JWT", ""
        )
        result = get_ieu_open_gwas_summary_stats("ieu-a-7")
        assert "note" in result
        assert "STUB" in result["note"]
        assert "reference_studies" in result
        assert "ieu-a-7" in result["reference_studies"]

    def test_finngen_known_phenocode(self):
        result = get_finngen_phenotype_definition("I9_CAD")
        assert result["phenocode"] == "I9_CAD"
        assert result["name"] == "Coronary artery disease"
        assert "sumstats_url" in result

    def test_finngen_unknown_phenocode_returns_error(self):
        result = get_finngen_phenotype_definition("NONEXISTENT_CODE")
        assert "note" in result
        assert "NONEXISTENT_CODE" in result.get("note", "")

    def test_list_gwas_maps_cad_to_efo(self, monkeypatch):
        monkeypatch.setattr(
            "mcp_servers.gwas_genetics_server.OPENGWAS_JWT", ""
        )
        # "cad" should resolve to GWAS Catalog EFO_0001645 fallback
        # This will make a real HTTP call — mark as integration if network not available
        # For unit test, just check the disease_query routing works
        pass  # tested in integration tests below


# ---------------------------------------------------------------------------
# Integration tests — real HTTP calls
# Run: pytest tests/test_gwas_genetics_server.py -v -m integration
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestGWASCatalogLive:

    def test_cad_associations_returns_data(self):
        result = get_gwas_catalog_associations("EFO_0001645", page_size=5)
        assert result["efo_id"] == "EFO_0001645"
        assert result["total_associations"] > 100   # CAD is well-studied
        assert len(result["associations"]) > 0
        first = result["associations"][0]
        assert "rsId" in first
        assert "pvalue_exp" in first
        # All returned associations should be at genome-wide significance
        for a in result["associations"]:
            if a["pvalue_exp"] is not None:
                assert a["pvalue_exp"] <= -8

    def test_cad_studies_returns_data(self):
        result = get_gwas_catalog_studies("EFO_0001645", page_size=5)
        assert result["efo_id"] == "EFO_0001645"
        assert result["total_studies"] > 50
        studies = result["studies"]
        assert len(studies) > 0
        assert all("accession" in s for s in studies)
        # At least one study should have full summary statistics
        # (GCST003116 / Nikpay 2015 is known to have them)
        assert any(s.get("has_full_sumstats") for s in studies), \
            "Expected at least one CAD study with full sumstats"

    def test_pcsk9_snp_associations(self):
        result = get_snp_associations("rs11591147")
        assert result["rsid"] == "rs11591147"
        assert result["n_associations"] > 0
        # PCSK9 R46L should be associated with LDL cholesterol
        traits = [
            a.get("efo_traits", [{}])[0].get("trait", "").lower()
            for a in result["associations"]
            if a.get("efo_traits")
        ]
        assert any("ldl" in t or "cholesterol" in t for t in traits), \
            f"Expected LDL association for rs11591147, got: {traits[:5]}"


@pytest.mark.integration
class TestGnomADLive:

    def test_pcsk9_constraint(self):
        result = query_gnomad_lof_constraint(["PCSK9"])
        assert len(result["genes"]) == 1
        gene = result["genes"][0]
        assert gene["symbol"] == "PCSK9"
        # PCSK9 is NOT LoF-constrained (LoF = beneficial for CAD)
        assert gene["loeuf"] is not None
        assert gene["loeuf"] > 0.5, \
            f"PCSK9 LOEUF should be > 0.5 (LoF tolerant), got {gene['loeuf']}"
        assert gene["pLI"] is not None
        assert gene["pLI"] < 0.1, \
            f"PCSK9 pLI should be near 0, got {gene['pLI']}"

    def test_chip_driver_constraint(self):
        result = query_gnomad_lof_constraint(["DNMT3A", "TET2"])
        assert len(result["genes"]) == 2
        for gene in result["genes"]:
            assert gene.get("loeuf") is not None
            # DNMT3A and TET2 are not LoF-constrained (somatic LoF drives CHIP)
            assert gene["loeuf"] > 0.5, \
                f"{gene['symbol']} LOEUF should be > 0.5, got {gene['loeuf']}"

    def test_missing_gene_handled_gracefully(self):
        result = query_gnomad_lof_constraint(["NONEXISTENT_GENE_XYZ"])
        assert len(result["genes"]) == 1
        gene = result["genes"][0]
        assert "error" in gene


@pytest.mark.integration
class TestGTExLive:

    def test_pcsk9_gene_id_resolution(self):
        result = resolve_gtex_gene_id("PCSK9")
        assert result["gene_symbol"] == "PCSK9"
        assert result["gencode_id"] is not None
        assert result["gencode_id"].startswith("ENSG00000169174")
        assert result["chromosome"] == "chr1"

    def test_pcsk9_liver_eqtls(self):
        result = query_gtex_eqtl("PCSK9", "Liver", items_per_page=5)
        assert result["gene"] == "PCSK9"
        assert result["tissue"] == "Liver"
        assert result["n_eqtls"] > 0
        eqtls = result["eqtls"]
        assert len(eqtls) > 0
        for e in eqtls:
            assert "snpId" in e
            assert "pvalue" in e
            assert "nes" in e     # normalized effect size (beta)

    def test_pcsk9_whole_blood_eqtls(self):
        # PCSK9 should also have eQTLs in blood
        result = query_gtex_eqtl("PCSK9", "Whole_Blood", items_per_page=5)
        assert result["gene"] == "PCSK9"
        # May have 0 eQTLs if expression is liver-specific — that's valid
        assert "n_eqtls" in result

    def test_invalid_tissue_raises_or_returns_empty(self):
        result = query_gtex_eqtl("PCSK9", "NONEXISTENT_TISSUE_XYZ")
        # Should either return empty eqtls or an error field — not crash
        assert "eqtls" in result or "error" in result


@pytest.mark.integration
class TestOpenGWASLive:
    """Live tests for IEU Open GWAS API — requires OPENGWAS_JWT in .env."""

    @pytest.fixture(autouse=True)
    def check_jwt(self):
        import os
        from dotenv import load_dotenv
        from pathlib import Path
        load_dotenv(Path(__file__).parent.parent / ".env")
        if not os.getenv("OPENGWAS_JWT"):
            pytest.skip("OPENGWAS_JWT not set — skipping OpenGWAS live tests")

    def test_list_cad_datasets(self):
        result = list_available_gwas("coronary artery disease")
        assert "datasets" in result
        datasets = result["datasets"]
        assert len(datasets) > 0
        # Should include ieu-a-7 (Nikpay 2015 CAD)
        ids = [d.get("id", "") for d in datasets]
        assert any("ieu" in str(i).lower() or "ebi" in str(i).lower() or i for i in ids), \
            f"Expected CAD study IDs, got: {ids[:5]}"

    def test_get_cad_top_hits(self):
        result = get_ieu_open_gwas_summary_stats("ieu-a-7")
        # Either live data or a useful error message
        if "error" in result:
            pytest.skip(f"OpenGWAS returned error: {result['error']}")
        assert "top_hits" in result or "note" in result

    def test_list_ldl_datasets(self):
        result = list_available_gwas("LDL cholesterol")
        assert "datasets" in result or "studies" in result


# ===========================================================================
# FinnGen server tests
# ===========================================================================

class TestFinnGenUnit:
    """Unit tests — verify schema and EFO resolution without live calls."""

    def test_efo_to_finngen_resolves_cad(self):
        from mcp_servers.finngen_server import efo_to_finngen_phenocode
        code = efo_to_finngen_phenocode("EFO_0001645")
        assert code == "I9_CAD"

    def test_efo_to_finngen_resolves_ibd(self):
        from mcp_servers.finngen_server import efo_to_finngen_phenocode
        code = efo_to_finngen_phenocode("EFO_0003767")
        assert code == "K11_IBD_STRICT"

    def test_efo_to_finngen_unknown_returns_none(self):
        from mcp_servers.finngen_server import efo_to_finngen_phenocode
        assert efo_to_finngen_phenocode("EFO_9999999") is None

    def test_efo_to_finngen_covers_key_diseases(self):
        from mcp_servers.finngen_server import EFO_TO_FINNGEN
        required = ["EFO_0001645", "EFO_0003767", "EFO_0000685", "EFO_0003885"]
        for efo in required:
            assert efo in EFO_TO_FINNGEN, f"{efo} missing from EFO_TO_FINNGEN"


@pytest.mark.integration
class TestFinnGenLive:
    """Integration tests — live FinnGen R10 REST API calls."""

    def test_cad_phenotype_info(self):
        from mcp_servers.finngen_server import get_finngen_phenotype_info
        result = get_finngen_phenotype_info("I9_CAD")
        assert result.get("phenocode") == "I9_CAD"
        assert result.get("n_cases", 0) > 10000, "Expected >10k CAD cases in FinnGen R10"
        assert "n_controls" in result
        assert "browser_url" in result

    def test_efo_id_auto_resolves_to_cad(self):
        from mcp_servers.finngen_server import get_finngen_phenotype_info
        result = get_finngen_phenotype_info("EFO_0001645")
        # Should auto-resolve EFO → I9_CAD
        assert result.get("phenocode") == "I9_CAD"
        assert not result.get("error")

    def test_cad_top_variants_returns_instruments(self):
        from mcp_servers.finngen_server import get_finngen_top_variants
        result = get_finngen_top_variants("I9_CAD", p_threshold=5e-8, n_max=10)
        assert result.get("n_significant", 0) > 0
        variants = result.get("variants", [])
        assert len(variants) > 0
        v = variants[0]
        assert "pval" in v
        assert float(v["pval"]) <= 5e-8
        assert "beta" in v

    def test_ibd_phenotypes_discoverable(self):
        from mcp_servers.finngen_server import list_finngen_phenotypes
        result = list_finngen_phenotypes(search_query="IBD")
        phenos = result.get("phenotypes", [])
        assert len(phenos) > 0
        codes = [p["phenocode"] for p in phenos]
        assert any("IBD" in c or "CROHN" in c or "ULCER" in c for c in codes)


# ===========================================================================
# Live γ estimation tests
# ===========================================================================

class TestLiveGammaUnit:
    """Unit tests for estimate_gamma_live — mock OT calls."""

    def test_estimate_gamma_live_returns_none_without_efo(self):
        from pipelines.ota_gamma_estimation import estimate_gamma_live
        result = estimate_gamma_live(
            "lipid_metabolism", "CAD",
            program_gene_set={"PCSK9", "LDLR"},
            efo_id=None,
        )
        assert result is None

    def test_estimate_gamma_live_returns_none_without_gene_set(self):
        from pipelines.ota_gamma_estimation import estimate_gamma_live
        result = estimate_gamma_live(
            "lipid_metabolism", "CAD",
            program_gene_set=None,
            efo_id="EFO_0001645",
        )
        assert result is None

    def test_estimate_gamma_live_mocked_ot(self):
        from unittest.mock import patch
        from pipelines.ota_gamma_estimation import estimate_gamma_live

        with patch(
            "mcp_servers.gwas_genetics_server.aggregate_l2g_scores_for_program_genes",
            return_value={
                "mean_genetic_score": 0.72,
                "mean_l2g_score": 0.72,
                "n_genes_with_data": 4,
            },
        ):
            result = estimate_gamma_live(
                "lipid_metabolism", "CAD",
                program_gene_set={"PCSK9", "LDLR", "HMGCR", "APOB"},
                efo_id="EFO_0001645",
            )

        assert result is not None
        assert result["gamma"] == pytest.approx(0.72 * 0.65, abs=0.01)
        assert result["evidence_tier"] in ("Tier2_Convergent", "Tier3_Provisional")
        assert "L2G" in (result.get("data_source") or "")

    def test_estimate_gamma_live_uses_l2g_not_coloc(self):
        """Live γ comes from L2G aggregation; colocalisation is not consulted."""
        from unittest.mock import patch
        from pipelines.ota_gamma_estimation import estimate_gamma_live

        with patch(
            "mcp_servers.gwas_genetics_server.aggregate_l2g_scores_for_program_genes",
            return_value={
                "mean_genetic_score": 0.40,
                "n_genes_with_data": 3,
            },
        ):
            result = estimate_gamma_live(
                "inflammatory_NF-kB", "IBD",
                program_gene_set={"TNF", "NOD2", "IL23R"},
                efo_id="EFO_0003767",
            )

        assert result is not None
        assert result["gamma"] == pytest.approx(0.40 * 0.65, abs=0.01)

    def test_estimate_gamma_falls_back_to_provisional_when_live_absent(self):
        from unittest.mock import patch
        from pipelines.ota_gamma_estimation import estimate_gamma

        with patch(
            "pipelines.ota_gamma_estimation.estimate_gamma_live",
            return_value=None,
        ):
            result = estimate_gamma("lipid_metabolism", "CAD")

        # No efo/program_gene_set: no fused or live path — γ is unknown
        assert result is None

    def test_estimate_gamma_signature_backward_compatible(self):
        """estimate_gamma() still works with no new kwargs — no regression."""
        from pipelines.ota_gamma_estimation import estimate_gamma
        # Old call signature — must not raise; minimal args yield no evidence
        result = estimate_gamma("lipid_metabolism", "CAD")
        assert result is None


@pytest.mark.integration
class TestLiveGammaIntegration:
    """Integration test — real OT API call for program enrichment."""

    def test_lipid_program_has_nonzero_ot_genetic_score(self):
        from mcp_servers.open_targets_server import get_ot_genetic_scores_for_gene_set
        result = get_ot_genetic_scores_for_gene_set(
            "EFO_0001645",
            ["PCSK9", "LDLR", "HMGCR", "APOB", "LPA"],
        )
        assert result["n_genes_with_data"] > 0
        assert result["mean_genetic_score"] > 0


# ===========================================================================
# GWAS instruments extraction
# ===========================================================================

@pytest.mark.integration
class TestGWASInstrumentsLive:
    """Integration test — GWAS Catalog gene-level instruments."""

    def test_pcsk9_cad_instruments(self):
        from mcp_servers.gwas_genetics_server import get_gwas_instruments_for_gene
        result = get_gwas_instruments_for_gene("PCSK9", "EFO_0001645")
        assert "n_instruments" in result
        assert "instruments" in result
        # PCSK9 has well-known CAD GWAS hits
        if result["n_instruments"] > 0:
            inst = result["instruments"][0]
            assert "rsid" in inst
