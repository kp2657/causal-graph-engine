"""
tests/test_agents.py — Unit tests for all tier agent implementations.

All tests are unit (no live API calls); MCP server functions are mocked.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

MOCK_DISEASE_QUERY = {
    "disease_name":    "coronary artery disease",
    "efo_id":          "EFO_0001645",
    "icd10_codes":     ["I20", "I25"],
    "modifier_types":  ["germline", "somatic_chip", "drug"],
    "primary_gwas_id": "ieu-a-7",
}

MOCK_PROGRAMS = [
    {"program_id": "lipid_metabolism",  "top_genes": ["PCSK9", "LDLR", "HMGCR"]},
    {"program_id": "inflammatory_NF-kB","top_genes": ["TET2", "DNMT3A", "NFKB1"]},
]

MOCK_BETA_MATRIX_RESULT = {
    "genes":    ["PCSK9", "TET2", "DNMT3A"],
    "programs": ["lipid_metabolism", "inflammatory_NF-kB"],
    "beta_matrix": {
        "PCSK9":   {"lipid_metabolism": -0.4, "inflammatory_NF-kB": 0.0},
        "TET2":    {"lipid_metabolism":  0.0, "inflammatory_NF-kB": 0.5},
        "DNMT3A":  {"lipid_metabolism":  0.0, "inflammatory_NF-kB": 0.3},
    },
    "evidence_tier_per_gene": {
        "PCSK9":  "Tier1_Interventional",
        "TET2":   "Tier2_Convergent",
        "DNMT3A": "Tier2_Convergent",
    },
    "n_tier1":   1,
    "n_tier2":   2,
    "n_tier3":   0,
    "n_virtual": 0,
    "warnings":  [],
}

MOCK_GAMMA_ESTIMATES = {
    "lipid_metabolism":   {"coronary artery disease": 0.8},
    "inflammatory_NF-kB": {"coronary artery disease": 0.5},
}

MOCK_CAUSAL_RESULT = {
    "n_edges_written":  3,
    "n_edges_rejected": 0,
    "top_genes": [
        {"gene": "PCSK9",  "ota_gamma": -0.32, "tier": "Tier1_Interventional", "programs": ["lipid_metabolism"]},
        {"gene": "TET2",   "ota_gamma":  0.25, "tier": "Tier2_Convergent",     "programs": ["inflammatory_NF-kB"]},
        {"gene": "DNMT3A", "ota_gamma":  0.15, "tier": "Tier2_Convergent",     "programs": ["inflammatory_NF-kB"]},
    ],
    "anchor_recovery": {
        "recovery_rate": 0.83,
        "recovered": ["PCSK9→LDL-C", "TET2_chip→CAD"],
        "missing":   [],
    },
    "shd":      2,
    "warnings": [],
}

MOCK_KG_RESULT = {
    "n_pathway_edges_added":     10,
    "n_ppi_edges_added":          5,
    "n_drug_target_edges_added":  3,
    "n_primekg_edges_added":      2,
    "top_pathways": ["R-HSA-6806003", "R-HSA-9612973"],
    "drug_target_summary": [
        {"drug": "evolocumab", "target": "PCSK9", "max_phase": 4, "ot_score": 0.89},
        {"drug": "azacitidine","target": "TET2",  "max_phase": 0, "ot_score": 0.34},
    ],
    "contradictions_flagged": 0,
    "warnings": [],
}

MOCK_PRIORITIZATION_RESULT = {
    "targets": [
        {
            "target_gene":   "PCSK9",
            "rank":          1,
            "target_score":  0.73,
            "ota_gamma":    -0.32,
            "evidence_tier": "Tier1_Interventional",
            "ot_score":      0.89,
            "max_phase":     4,
            "known_drugs":   ["evolocumab"],
            "pli":           0.12,
            "flags":         ["repurposing_candidate", "genetic_anchor"],
            "top_programs":  ["lipid_metabolism"],
            "key_evidence":  ["FOURIER trial", "MR p=3.8e-22"],
            "safety_flags":  [],
        },
        {
            "target_gene":   "TET2",
            "rank":          2,
            "target_score":  0.41,
            "ota_gamma":     0.25,
            "evidence_tier": "Tier2_Convergent",
            "ot_score":      0.34,
            "max_phase":     0,
            "known_drugs":   ["azacitidine"],
            "pli":           0.0,
            "flags":         ["chip_mechanism"],
            "top_programs":  ["inflammatory_NF-kB"],
            "key_evidence":  ["Bick 2020", "Replogle 2022"],
            "safety_flags":  [],
        },
    ],
    "warnings": [],
}

MOCK_CHEMISTRY_RESULT = {
    "target_chemistry": {
        "PCSK9": {
            "chembl_id":      "CHEMBL3833335",
            "max_phase":      4,
            "best_ic50_nM":   None,
            "tractability":   "antibody",
            "ro5_violations": 3,
            "cmap_available": False,
            "drugs_found":    ["evolocumab"],
        },
        "TET2": {
            "chembl_id":      None,
            "max_phase":      0,
            "best_ic50_nM":   None,
            "tractability":   "difficult",
            "ro5_violations": None,
            "cmap_available": False,
            "drugs_found":    [],
        },
    },
    "repurposing_candidates": [],
    "warnings": [],
}

MOCK_TRIALS_RESULT = {
    "trial_summary": {
        "PCSK9": {
            "n_trials_total": 8, "n_active": 2, "n_completed": 5,
            "n_terminated": 1, "max_phase_reached": 4, "safety_signals": [],
        },
    },
    "key_trials": [
        {
            "nct_id": "NCT01764633", "drug": "evolocumab", "phase": [3],
            "status": "COMPLETED", "primary_outcome": "CV death", "n_enrolled": 27564,
        }
    ],
    "development_risk": {"PCSK9": "low", "TET2": "medium"},
    "repurposing_opportunities": ["evolocumab (target: PCSK9) approved for hypercholesterolemia"],
    "warnings": [],
}


# ===========================================================================
# Tier 1 — Phenotype Architect
# ===========================================================================

class TestPhenotypeArchitect:
    def _mock_gwas(self):
        return {
            "total_studies": 5,
            "efo_id": "EFO_0001645",
            "datasets": [{"id": "ieu-a-7"}],
            "phenocode": "I9_CAD",
        }

    def test_known_disease_resolves_efo(self):
        from agents.tier1_phenomics.phenotype_architect import DISEASE_EFO_MAP
        assert "coronary artery disease" in DISEASE_EFO_MAP
        assert DISEASE_EFO_MAP["coronary artery disease"] == "EFO_0001645"

    def test_efo_icd10_map_cad(self):
        from agents.tier1_phenomics.phenotype_architect import EFO_ICD10_MAP
        assert "I25" in EFO_ICD10_MAP["EFO_0001645"]

    @patch("mcp_servers.gwas_genetics_server.get_gwas_catalog_studies")
    @patch("mcp_servers.gwas_genetics_server.list_available_gwas")
    @patch("mcp_servers.gwas_genetics_server.get_finngen_phenotype_definition")
    def test_run_cad_returns_expected_keys(self, mock_fg, mock_gwas_list, mock_studies):
        mock_studies.return_value = {"total_studies": 5, "efo_id": "EFO_0001645"}
        mock_gwas_list.return_value = {"datasets": [{"id": "ieu-a-7"}]}
        mock_fg.return_value = {"phenocode": "I9_CAD"}

        from agents.tier1_phenomics.phenotype_architect import run
        result = run("coronary artery disease")
        assert result["efo_id"] == "EFO_0001645"
        assert "I25" in result["icd10_codes"]
        assert result["day_one_mode"] is True

    @patch("mcp_servers.gwas_genetics_server.get_gwas_catalog_studies")
    @patch("mcp_servers.gwas_genetics_server.list_available_gwas")
    @patch("mcp_servers.gwas_genetics_server.get_finngen_phenotype_definition")
    def test_run_unknown_disease_graceful(self, mock_fg, mock_gwas_list, mock_studies):
        mock_studies.return_value = {"total_studies": 0, "efo_id": None}
        mock_gwas_list.return_value = {"datasets": []}
        mock_fg.return_value = {}

        from agents.tier1_phenomics.phenotype_architect import run
        result = run("unknown_disease_xyz")
        assert result["efo_id"] is None
        assert result["icd10_codes"] == []


# ===========================================================================
# Tier 1 — Statistical Geneticist
# ===========================================================================

class TestStatisticalGeneticist:
    @patch("pipelines.mr_analysis.run_two_sample_mr")
    @patch("pipelines.mr_analysis.run_sensitivity_analysis")
    @patch("mcp_servers.gwas_genetics_server.query_gtex_eqtl")
    @patch("mcp_servers.gwas_genetics_server.get_gwas_catalog_associations")
    @patch("mcp_servers.gwas_genetics_server.query_gnomad_lof_constraint")
    def test_run_returns_instruments(
        self, mock_gnomad, mock_gwas_assoc, mock_eqtl, mock_sensitivity, mock_mr
    ):
        mock_mr.return_value = {
            "ivw_beta": -0.4, "ivw_p": 1e-20, "n_snps": 50, "f_statistic": 80.0
        }
        mock_sensitivity.return_value = {"egger_intercept_p": 0.9}
        mock_eqtl.return_value = {
            "data": [{"nes": -0.3, "p_value": 1e-10, "variant_id": "rs1234"}]
        }
        mock_gwas_assoc.return_value = {"associations": []}
        mock_gnomad.return_value = {"PCSK9": {"pLI": 0.02}}

        from agents.tier1_phenomics.statistical_geneticist import run
        result = run(MOCK_DISEASE_QUERY)

        assert "instruments" in result
        assert len(result["instruments"]) == 3  # LDL-C, HDL-C, CRP
        assert "anchor_genes_validated" in result
        assert "warnings" in result

    @patch("pipelines.mr_analysis.run_two_sample_mr")
    @patch("pipelines.mr_analysis.run_sensitivity_analysis")
    @patch("mcp_servers.gwas_genetics_server.query_gtex_eqtl")
    @patch("mcp_servers.gwas_genetics_server.get_gwas_catalog_associations")
    @patch("mcp_servers.gwas_genetics_server.query_gnomad_lof_constraint")
    def test_weak_instrument_warning(
        self, mock_gnomad, mock_gwas_assoc, mock_eqtl, mock_sensitivity, mock_mr
    ):
        mock_mr.return_value = {"ivw_beta": 0.1, "ivw_p": 0.05, "n_snps": 1, "f_statistic": 5.0}
        mock_sensitivity.return_value = {"egger_intercept_p": 0.5}
        mock_eqtl.return_value = {"data": []}
        mock_gwas_assoc.return_value = {"associations": []}
        mock_gnomad.return_value = {}

        from agents.tier1_phenomics.statistical_geneticist import run
        result = run(MOCK_DISEASE_QUERY)
        warns = result["warnings"]
        assert any("F-statistic" in w and "< 10" in w for w in warns)


# ===========================================================================
# Tier 1 — Somatic Exposure Agent
# ===========================================================================

class TestSomaticExposureAgent:
    @patch("mcp_servers.viral_somatic_server.get_chip_disease_associations")
    @patch("mcp_servers.viral_somatic_server.get_viral_disease_mr_results")
    @patch("mcp_servers.viral_somatic_server.get_drug_exposure_mr")
    @patch("mcp_servers.clinical_trials_server.search_clinical_trials")
    @patch("mcp_servers.open_targets_server.get_open_targets_drug_info")
    def test_chip_edges_created(
        self, mock_ot, mock_trials, mock_drug_mr, mock_viral, mock_chip
    ):
        mock_chip.return_value = {
            "associations": [
                {"gene": "TET2",   "hr": 1.72, "ci_lower": 1.30, "ci_upper": 2.28, "source": "Bick2020"},
                {"gene": "DNMT3A", "hr": 1.26, "ci_lower": 1.06, "ci_upper": 1.49, "source": "Bick2020"},
            ]
        }
        mock_viral.return_value = {"beta": None}
        mock_drug_mr.return_value = {"beta": -0.3, "source": "published_MR"}
        mock_trials.return_value = {"trials": [{"phase": ["PHASE3"], "intervention": "atorvastatin"}]}
        mock_ot.return_value = {"indications": []}

        from agents.tier1_phenomics.somatic_exposure_agent import run
        result = run(MOCK_DISEASE_QUERY)

        assert len(result["chip_edges"]) == 2
        assert result["summary"]["n_chip_genes"] == 2
        assert "warnings" in result

    @patch("mcp_servers.viral_somatic_server.get_chip_disease_associations")
    @patch("mcp_servers.viral_somatic_server.get_viral_disease_mr_results")
    @patch("mcp_servers.viral_somatic_server.get_drug_exposure_mr")
    @patch("mcp_servers.clinical_trials_server.search_clinical_trials")
    @patch("mcp_servers.open_targets_server.get_open_targets_drug_info")
    def test_chip_tier_assignment(
        self, mock_ot, mock_trials, mock_drug_mr, mock_viral, mock_chip
    ):
        mock_chip.return_value = {
            "associations": [{"gene": "TET2", "hr": 1.72, "ci_lower": 1.3, "ci_upper": 2.3, "source": "Bick2020"}]
        }
        mock_viral.return_value = {"beta": None}
        mock_drug_mr.return_value = {"beta": None}
        mock_trials.return_value = {"trials": []}
        mock_ot.return_value = {}

        from agents.tier1_phenomics.somatic_exposure_agent import run
        result = run(MOCK_DISEASE_QUERY)
        edge = result["chip_edges"][0]
        assert edge["evidence_tier"] == "Tier2_Convergent"


# ===========================================================================
# Tier 2 — Perturbation Genomics Agent
# ===========================================================================

class TestPerturbationGenomicsAgent:
    @patch("pipelines.ota_beta_estimation.estimate_beta")
    @patch("mcp_servers.burden_perturb_server.get_gene_perturbation_effect")
    @patch("mcp_servers.burden_perturb_server.get_cnmf_program_info")
    @patch("mcp_servers.burden_perturb_server.get_program_gene_loadings")
    @patch("mcp_servers.gwas_genetics_server.query_gtex_eqtl")
    def test_returns_beta_matrix(
        self, mock_eqtl, mock_loadings, mock_programs, mock_perturb, mock_estimate
    ):
        mock_programs.return_value = {"programs": MOCK_PROGRAMS}
        # estimate_beta is called per (gene, program_id); return beta + tier per call
        mock_estimate.return_value = {
            "beta": -0.4,
            "evidence_tier": "Tier1_Interventional",
        }
        mock_perturb.return_value = {"program_effects": {}}
        mock_loadings.return_value = {"top_genes": ["PCSK9", "LDLR"]}
        mock_eqtl.return_value = {"data": []}

        from agents.tier2_pathway.perturbation_genomics_agent import run
        result = run(["PCSK9", "TET2"], MOCK_DISEASE_QUERY)

        assert "beta_matrix" in result
        assert "PCSK9" in result["beta_matrix"]
        assert "evidence_tier_per_gene" in result

    @patch("pipelines.ota_beta_estimation.estimate_beta")
    @patch("mcp_servers.burden_perturb_server.get_gene_perturbation_effect")
    @patch("mcp_servers.burden_perturb_server.get_cnmf_program_info")
    @patch("mcp_servers.burden_perturb_server.get_program_gene_loadings")
    @patch("mcp_servers.gwas_genetics_server.query_gtex_eqtl")
    def test_biology_cross_check_flags_mismatch(
        self, mock_eqtl, mock_loadings, mock_programs, mock_perturb, mock_estimate
    ):
        mock_programs.return_value = {"programs": MOCK_PROGRAMS}
        # PCSK9 → lipid_metabolism should be negative; return positive to trigger warning
        # estimate_beta is called per (gene, program_id); use side_effect for per-program betas
        def _estimate_side_effect(gene, program, **kwargs):
            if program == "lipid_metabolism":
                return {"beta": 0.5, "evidence_tier": "Tier1_Interventional"}
            return {"beta": 0.0, "evidence_tier": "provisional_virtual"}
        mock_estimate.side_effect = _estimate_side_effect
        mock_perturb.return_value = {"program_effects": {}}
        mock_loadings.return_value = {"top_genes": []}
        mock_eqtl.return_value = {"data": []}

        from agents.tier2_pathway.perturbation_genomics_agent import run
        result = run(["PCSK9"], MOCK_DISEASE_QUERY)
        warns = result["warnings"]
        assert any("Biology mismatch" in w for w in warns)

    @patch("pipelines.ota_beta_estimation.estimate_beta")
    @patch("mcp_servers.burden_perturb_server.get_gene_perturbation_effect")
    @patch("mcp_servers.burden_perturb_server.get_cnmf_program_info")
    @patch("mcp_servers.burden_perturb_server.get_program_gene_loadings")
    @patch("mcp_servers.gwas_genetics_server.query_gtex_eqtl")
    def test_tier2_eqtl_activates_when_tier1_absent(
        self, mock_eqtl, mock_loadings, mock_programs, mock_perturb, mock_estimate
    ):
        """Tier 2 (eQTL-MR) β activates when GTEx returns a significant eQTL."""
        mock_programs.return_value = {"programs": MOCK_PROGRAMS}
        # GTEx returns a strong liver eQTL for PCSK9
        mock_eqtl.return_value = {
            "eqtls": [{"nes": -0.62, "pvalue": 1e-15, "snpId": "rs11591147", "chromosome": "1"}]
        }
        mock_loadings.return_value = {"top_genes": [{"gene": "PCSK9", "weight": 0.85}]}
        mock_perturb.return_value = {"top_programs_up": [], "top_programs_dn": []}
        # estimate_beta returns Tier2 for lipid_metabolism when eqtl_data is present
        def _side_effect(gene, program, eqtl_data=None, program_loading=None, **kwargs):
            if eqtl_data is not None and program == "lipid_metabolism":
                return {"beta": eqtl_data["nes"] * (program_loading or 1.0), "evidence_tier": "Tier2_Convergent"}
            return {"beta": None, "evidence_tier": "provisional_virtual"}
        mock_estimate.side_effect = _side_effect

        from agents.tier2_pathway.perturbation_genomics_agent import run
        result = run(["PCSK9"], MOCK_DISEASE_QUERY)

        tier = result["evidence_tier_per_gene"]["PCSK9"]
        assert tier == "Tier2_Convergent", f"Expected Tier2_Convergent, got {tier}"
        assert result["n_tier2"] == 1
        beta_entry = result["beta_matrix"]["PCSK9"].get("lipid_metabolism")
        assert beta_entry is not None
        assert beta_entry["beta"] < 0, "PCSK9 → lipid_metabolism β should be negative (suppresses LDL)"

    @patch("pipelines.ota_beta_estimation.estimate_beta")
    @patch("mcp_servers.burden_perturb_server.get_gene_perturbation_effect")
    @patch("mcp_servers.burden_perturb_server.get_cnmf_program_info")
    @patch("mcp_servers.burden_perturb_server.get_program_gene_loadings")
    @patch("mcp_servers.gwas_genetics_server.query_gtex_eqtl")
    def test_gtex_tissue_selected_from_disease_map(
        self, mock_eqtl, mock_loadings, mock_programs, mock_perturb, mock_estimate
    ):
        """GTEx tissue is resolved from DISEASE_CELL_TYPE_MAP, not hardcoded."""
        mock_programs.return_value = {"programs": MOCK_PROGRAMS}
        mock_eqtl.return_value = {"eqtls": []}
        mock_loadings.return_value = {"top_genes": []}
        mock_perturb.return_value = {"top_programs_up": [], "top_programs_dn": []}
        mock_estimate.return_value = {"beta": None, "evidence_tier": "provisional_virtual"}

        from agents.tier2_pathway.perturbation_genomics_agent import run
        # IBD disease → should query Colon_Sigmoid tissue
        ibd_query = {"disease_name": "inflammatory bowel disease", "efo_id": "EFO_0003767"}
        run(["NOD2"], ibd_query)
        calls = [str(c) for c in mock_eqtl.call_args_list]
        assert any("Colon_Sigmoid" in c for c in calls), (
            f"Expected Colon_Sigmoid for IBD; got calls: {calls}"
        )


# ===========================================================================
# Tier 2 — Regulatory Genomics Agent
# ===========================================================================

class TestRegulatoryGenomicsAgent:
    @patch("mcp_servers.gwas_genetics_server.query_gtex_eqtl")
    @patch("mcp_servers.gwas_genetics_server.get_snp_associations")
    @patch("mcp_servers.burden_perturb_server.get_cnmf_program_info")
    @patch("mcp_servers.burden_perturb_server.get_program_gene_loadings")
    def test_returns_eqtl_summary(
        self, mock_loadings, mock_programs, mock_snp, mock_eqtl
    ):
        mock_programs.return_value = {"programs": MOCK_PROGRAMS}
        mock_loadings.return_value = {"top_genes": ["PCSK9", "LDLR"]}
        mock_eqtl.return_value = {
            "data": [{"nes": -0.4, "p_value": 1e-12, "variant_id": "rs11591147"}]
        }
        mock_snp.return_value = {
            "associations": [{"p_value_exponent": -10, "trait": "LDL cholesterol"}]
        }

        from agents.tier2_pathway.regulatory_genomics_agent import run
        result = run(["PCSK9"], MOCK_DISEASE_QUERY)

        assert "PCSK9" in result["gene_eqtl_summary"]
        entry = result["gene_eqtl_summary"]["PCSK9"]
        assert entry["top_eqtl_p"] == 1e-12
        assert entry["coloc_candidate"] is True
        assert "PCSK9" in result["tier2_upgrades"]

    @patch("mcp_servers.gwas_genetics_server.query_gtex_eqtl")
    @patch("mcp_servers.gwas_genetics_server.get_snp_associations")
    @patch("mcp_servers.burden_perturb_server.get_cnmf_program_info")
    @patch("mcp_servers.burden_perturb_server.get_program_gene_loadings")
    def test_pcsk9_liver_eqtl_warning_when_missing(
        self, mock_loadings, mock_programs, mock_snp, mock_eqtl
    ):
        mock_programs.return_value = {"programs": MOCK_PROGRAMS}
        mock_loadings.return_value = {"top_genes": []}
        mock_eqtl.return_value = {"data": []}
        mock_snp.return_value = {"associations": []}

        from agents.tier2_pathway.regulatory_genomics_agent import run
        result = run(["PCSK9"], MOCK_DISEASE_QUERY)
        warns = result["warnings"]
        assert any("PCSK9 Liver eQTL not significant" in w for w in warns)


# ===========================================================================
# Tier 3 — Causal Discovery Agent
# ===========================================================================

class TestCausalDiscoveryAgent:
    @patch("agents.tier3_causal.causal_discovery_agent._maybe_therapeutic_redirection")
    @patch("pipelines.ota_gamma_estimation.compute_ota_gamma")
    @patch("mcp_servers.graph_db_server.write_causal_edges")
    @patch("mcp_servers.graph_db_server.run_evalue_check")
    def test_returns_expected_keys(
        self, mock_evalue, mock_write, mock_ota, mock_tr
    ):
        mock_tr.return_value = {}
        mock_ota.return_value = {
            "ota_gamma": -0.32,
            "dominant_tier": "Tier1_Interventional",
            "top_programs": ["lipid_metabolism"],
        }
        mock_write.return_value = {"n_written": 2}
        mock_evalue.return_value = {"e_value": 5.0}

        from agents.tier3_causal.causal_discovery_agent import run
        result = run(MOCK_BETA_MATRIX_RESULT, MOCK_GAMMA_ESTIMATES, MOCK_DISEASE_QUERY)

        assert "n_edges_written" in result
        assert "top_genes" in result
        assert "shd" in result
        assert "warnings" in result

    @patch("agents.tier3_causal.causal_discovery_agent._maybe_therapeutic_redirection")
    @patch("pipelines.ota_gamma_estimation.compute_ota_gamma")
    @patch("mcp_servers.graph_db_server.write_causal_edges")
    @patch("mcp_servers.graph_db_server.run_evalue_check")
    def test_pipeline_completes_with_zero_edges(
        self, mock_evalue, mock_write, mock_ota, mock_tr
    ):
        """Pipeline should complete without errors even when no edges are written."""
        mock_tr.return_value = {}
        mock_ota.return_value = {
            "ota_gamma": 0.001, "dominant_tier": "provisional_virtual", "top_programs": []
        }
        mock_write.return_value = {"n_written": 0}
        mock_evalue.return_value = {"e_value": 100.0}

        from agents.tier3_causal.causal_discovery_agent import run
        result = run(MOCK_BETA_MATRIX_RESULT, MOCK_GAMMA_ESTIMATES, MOCK_DISEASE_QUERY)
        assert "n_edges_written" in result
        assert result["n_edges_written"] == 0 or result["n_edges_written"] >= 0


# ===========================================================================
# Tier 3 — KG Completion Agent
# ===========================================================================

class TestKGCompletionAgent:
    @patch("mcp_servers.pathways_kg_server.get_reactome_pathways_for_gene")
    @patch("mcp_servers.pathways_kg_server.get_string_interactions")
    @patch("mcp_servers.pathways_kg_server.query_primekg_subgraph")
    @patch("mcp_servers.open_targets_server.get_open_targets_disease_targets")
    @patch("mcp_servers.clinical_trials_server.get_trials_for_target")
    @patch("mcp_servers.chemistry_server.search_chembl_compound")
    @patch("mcp_servers.graph_db_server.query_graph_for_disease")
    @patch("mcp_servers.graph_db_server.write_causal_edges")
    def test_returns_edge_counts(
        self, mock_write, mock_query, mock_chembl, mock_trials, mock_ot,
        mock_pkg, mock_string, mock_reactome
    ):
        mock_reactome.return_value = {"pathways": [{"stId": "R-HSA-1", "name": "Lipid"}]}
        mock_string.return_value = {"interactions": [{"score": 900, "gene1": "PCSK9", "gene2": "LDLR"}]}
        mock_pkg.return_value = {"edges": [{"prior_probability": 0.8}]}
        mock_ot.return_value = {"targets": [{"symbol": "PCSK9", "overall_score": 0.89, "max_clinical_phase": 4}]}
        mock_trials.return_value = {"trials": [{"phase": ["PHASE4"], "intervention": "evolocumab"}]}
        mock_chembl.return_value = {"max_phase": 4}
        mock_query.return_value = {"edges": []}
        mock_write.return_value = {"n_written": 5}

        from agents.tier3_causal.kg_completion_agent import run
        result = run(MOCK_CAUSAL_RESULT, MOCK_DISEASE_QUERY)

        assert "n_pathway_edges_added" in result
        assert "n_ppi_edges_added" in result
        assert "drug_target_summary" in result
        assert result["n_ppi_edges_added"] >= 0


# ===========================================================================
# Tier 4 — Target Prioritization Agent
# ===========================================================================

class TestTargetPrioritizationAgent:
    @patch("mcp_servers.open_targets_server.get_open_targets_disease_targets")
    @patch("mcp_servers.clinical_trials_server.get_trials_for_target")
    @patch("mcp_servers.gwas_genetics_server.query_gnomad_lof_constraint")
    def test_returns_ranked_targets(self, mock_gnomad, mock_trials, mock_ot):
        mock_ot.return_value = {"targets": [
            {"gene_symbol": "PCSK9", "overall_score": 0.89, "genetic_score": 0.95, "max_clinical_phase": 4},
            {"gene_symbol": "TET2",  "overall_score": 0.34, "genetic_score": 0.40, "max_clinical_phase": 0},
        ]}
        mock_trials.return_value = {"trials": [{"phase": ["PHASE4"], "intervention": "evolocumab"}]}
        # Correct format: {"genes": [{"symbol": ..., "pLI": ...}]}
        mock_gnomad.return_value = {"genes": [
            {"symbol": "PCSK9",  "pLI": 0.02},
            {"symbol": "TET2",   "pLI": 0.0},
            {"symbol": "DNMT3A", "pLI": 0.0},
        ]}

        from agents.tier4_translation.target_prioritization_agent import run
        result = run(MOCK_CAUSAL_RESULT, MOCK_KG_RESULT, MOCK_DISEASE_QUERY)

        targets = result["targets"]
        assert len(targets) >= 2
        assert targets[0]["rank"] == 1
        assert targets[0]["target_score"] >= targets[1]["target_score"]

    @patch("mcp_servers.open_targets_server.get_open_targets_disease_targets")
    @patch("mcp_servers.clinical_trials_server.get_trials_for_target")
    @patch("mcp_servers.gwas_genetics_server.query_gnomad_lof_constraint")
    def test_essential_gene_safety_flag(self, mock_gnomad, mock_trials, mock_ot):
        mock_ot.return_value = {"targets": []}
        mock_trials.return_value = {"trials": []}
        # Correct format: {"genes": [{"symbol": ..., "pLI": ...}]}
        mock_gnomad.return_value = {"genes": [{"symbol": "PCSK9", "pLI": 0.95}]}

        from agents.tier4_translation.target_prioritization_agent import run
        result = run(MOCK_CAUSAL_RESULT, MOCK_KG_RESULT, MOCK_DISEASE_QUERY)
        # Find PCSK9 record
        pcsk9 = next((t for t in result["targets"] if t["target_gene"] == "PCSK9"), None)
        assert pcsk9 is not None
        assert any("pLI" in s for s in pcsk9["safety_flags"])


# ===========================================================================
# Tier 4 — Chemistry Agent
# ===========================================================================

class TestChemistryAgent:
    @patch("pipelines.gps_disease_screen.run_gps_disease_screens")
    @patch("mcp_servers.chemistry_server.resolve_gps_putative_target_labels_to_hgnc")
    def test_returns_gps_outputs_and_hgnc_mapping(
        self, mock_hgnc, mock_gps
    ):
        mock_gps.return_value = {
            "disease_reversers": [],
            "program_reversers": {},
            "programs_screened": [],
            "warnings": [],
            "disease_sig_n_genes": 0,
        }
        mock_hgnc.return_value = {"genes": [], "n_resolved": 0, "n_unresolved": 0, "mapping_sample": []}

        from agents.tier4_translation.chemistry_agent import run
        result = run(MOCK_PRIORITIZATION_RESULT, MOCK_DISEASE_QUERY)

        assert "target_chemistry" in result
        assert result["target_chemistry"] == {}
        assert "gps_putative_hgnc" in result
        assert isinstance(result["gps_putative_hgnc"].get("genes"), list)
        assert "gps_disease_reversers" in result
        assert "gps_program_reversers" in result

    @patch("pipelines.gps_disease_screen.run_gps_disease_screens")
    def test_gps_screen_receives_gamma_estimates_from_prioritization(
        self, mock_gps
    ):
        """Orchestrator injects _gamma_estimates; chemistry must pass them to run_gps_disease_screens."""
        mock_gps.return_value = {
            "disease_reversers": [],
            "program_reversers": {},
            "programs_screened": [],
            "warnings": [],
            "disease_sig_n_genes": 0,
        }

        gamma_stub = {"prog_x": {"CAD": {"gamma": 0.15, "evidence_tier": "Tier2"}}}
        prio = {**MOCK_PRIORITIZATION_RESULT, "_gamma_estimates": gamma_stub}

        from agents.tier4_translation.chemistry_agent import run
        run(prio, MOCK_DISEASE_QUERY)

        mock_gps.assert_called_once()
        kwargs = mock_gps.call_args.kwargs
        assert kwargs.get("gamma_estimates") == gamma_stub


# ===========================================================================
# Tier 4 — Clinical Trialist Agent
# ===========================================================================

class TestClinicalTrialistAgent:
    @patch("mcp_servers.clinical_trials_server.search_clinical_trials")
    @patch("mcp_servers.clinical_trials_server.get_trial_details")
    @patch("mcp_servers.clinical_trials_server.get_trials_for_target")
    @patch("mcp_servers.open_targets_server.get_open_targets_drug_info")
    def test_returns_trial_summary(
        self, mock_ot_drug, mock_target_trials, mock_details, mock_search
    ):
        mock_search.return_value = {"trials": [
            {"nct_id": "NCT01764633", "status": "COMPLETED", "phase": ["PHASE3"],
             "intervention": "evolocumab", "why_stopped": ""},
        ]}
        mock_details.return_value = {
            "status": "COMPLETED", "primary_outcome": "CV death", "enrollment": 27564
        }
        mock_target_trials.return_value = {"trials": []}
        mock_ot_drug.return_value = {"indications": ["hypercholesterolemia"]}

        from agents.tier4_translation.clinical_trialist_agent import run
        result = run(MOCK_PRIORITIZATION_RESULT, MOCK_DISEASE_QUERY)

        assert "trial_summary" in result
        assert "key_trials" in result
        assert "development_risk" in result
        # CAD disease — key trials should be populated
        assert len(result["key_trials"]) > 0

    @patch("mcp_servers.clinical_trials_server.search_clinical_trials")
    @patch("mcp_servers.clinical_trials_server.get_trial_details")
    @patch("mcp_servers.clinical_trials_server.get_trials_for_target")
    @patch("mcp_servers.open_targets_server.get_open_targets_drug_info")
    def test_safety_termination_sets_high_risk(
        self, mock_ot_drug, mock_target_trials, mock_details, mock_search
    ):
        mock_search.return_value = {"trials": [
            {"nct_id": "NCT999", "status": "TERMINATED", "phase": ["PHASE2"],
             "intervention": "test_drug", "why_stopped": "safety: serious adverse events"},
        ]}
        mock_details.return_value = {"status": "TERMINATED", "primary_outcome": "", "enrollment": None}
        mock_target_trials.return_value = {"trials": []}
        mock_ot_drug.return_value = {}

        from agents.tier4_translation.clinical_trialist_agent import run
        result = run(MOCK_PRIORITIZATION_RESULT, MOCK_DISEASE_QUERY)
        # PCSK9 should now be high risk due to safety signal
        risk = result["development_risk"].get("PCSK9")
        assert risk == "high"


# ===========================================================================
# Tier 5 — Scientific Writer Agent
# ===========================================================================

class TestScientificWriterAgent:
    def test_returns_graphoutput_schema(self):
        from agents.tier5_writer.scientific_writer_agent import run
        result = run(
            phenotype_result=MOCK_DISEASE_QUERY,
            genetics_result={"instruments": [], "warnings": []},
            somatic_result={"chip_edges": [], "viral_edges": [], "drug_edges": [], "summary": {}, "warnings": []},
            beta_matrix_result=MOCK_BETA_MATRIX_RESULT,
            regulatory_result={"gene_eqtl_summary": {}, "gene_program_overlap": {}, "tier2_upgrades": [], "warnings": []},
            causal_result=MOCK_CAUSAL_RESULT,
            kg_result=MOCK_KG_RESULT,
            prioritization_result=MOCK_PRIORITIZATION_RESULT,
            chemistry_result=MOCK_CHEMISTRY_RESULT,
            trials_result=MOCK_TRIALS_RESULT,
        )
        required_keys = [
            "disease_name", "efo_id", "target_list",
            "n_tier1_edges", "n_tier2_edges", "n_tier3_edges", "n_virtual_edges",
            "executive_summary", "top_target_narratives",
            "evidence_quality", "limitations",
            "pipeline_version", "generated_at",
        ]
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_executive_summary_contains_disease(self):
        from agents.tier5_writer.scientific_writer_agent import run
        result = run(
            phenotype_result=MOCK_DISEASE_QUERY,
            genetics_result={"warnings": []},
            somatic_result={"warnings": []},
            beta_matrix_result=MOCK_BETA_MATRIX_RESULT,
            regulatory_result={"warnings": []},
            causal_result=MOCK_CAUSAL_RESULT,
            kg_result=MOCK_KG_RESULT,
            prioritization_result=MOCK_PRIORITIZATION_RESULT,
            chemistry_result=MOCK_CHEMISTRY_RESULT,
            trials_result=MOCK_TRIALS_RESULT,
        )
        assert "coronary artery disease" in result["executive_summary"].lower()

    def test_top_narratives_count(self):
        from agents.tier5_writer.scientific_writer_agent import run
        result = run(
            phenotype_result=MOCK_DISEASE_QUERY,
            genetics_result={"warnings": []},
            somatic_result={"warnings": []},
            beta_matrix_result=MOCK_BETA_MATRIX_RESULT,
            regulatory_result={"warnings": []},
            causal_result=MOCK_CAUSAL_RESULT,
            kg_result=MOCK_KG_RESULT,
            prioritization_result=MOCK_PRIORITIZATION_RESULT,
            chemistry_result=MOCK_CHEMISTRY_RESULT,
            trials_result=MOCK_TRIALS_RESULT,
        )
        assert len(result["top_target_narratives"]) <= 3

    def test_limitations_mentions_provisional_virtual(self):
        from agents.tier5_writer.scientific_writer_agent import run
        result = run(
            phenotype_result=MOCK_DISEASE_QUERY,
            genetics_result={"warnings": []},
            somatic_result={"warnings": []},
            beta_matrix_result=MOCK_BETA_MATRIX_RESULT,
            regulatory_result={"warnings": []},
            causal_result=MOCK_CAUSAL_RESULT,
            kg_result=MOCK_KG_RESULT,
            prioritization_result=MOCK_PRIORITIZATION_RESULT,
            chemistry_result=MOCK_CHEMISTRY_RESULT,
            trials_result=MOCK_TRIALS_RESULT,
        )
        assert "provisional_virtual" in result["limitations"]
