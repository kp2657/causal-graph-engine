"""
Tests for Phase A modules:
  models/latent_mediator.py
  pipelines/state_space/program_labeler.py
  pipelines/state_space/program_loading.py
  pipelines/state_space/transition_graph.compute_transition_gene_weights
  pipelines/state_space/latent_model.build_multi_celltype_latent_space / merge_multi_celltype_adata
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import scipy.sparse as sp


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def minimal_nmf_result():
    return {
        "programs": [
            {
                "program_id": "P00",
                "top_genes": ["TNF", "IL6", "NFKB1", "CCL2", "IFNG", "IL1B"],
                "gene_loadings": {"TNF": 0.9, "IL6": 0.7, "NFKB1": 0.5},
                "n_cells_expressing": 1200,
            },
            {
                "program_id": "P01",
                "top_genes": ["TGFB1", "COL1A1", "FN1", "ACTA2", "VIM", "TIMP1"],
                "gene_loadings": {"TGFB1": 0.8, "COL1A1": 0.6, "FN1": 0.3},
            },
            {
                "program_id": "P02",
                "top_genes": ["UNKN1", "UNKN2", "UNKN3"],  # no type match
                "gene_loadings": {"UNKN1": 0.5},
            },
        ],
        "source": "sklearn_NMF",
        "disease": "IBD",
    }


@pytest.fixture()
def transition_weights():
    return {"TNF": 0.9, "IL6": 0.6, "NFKB1": 0.4, "TGFB1": 0.7, "COL1A1": 0.8, "FN1": 0.3}


@pytest.fixture()
def synthetic_adata():
    """60-cell, 10-gene AnnData with 2 clearly separated states."""
    import anndata
    import pandas as pd

    n_cells, n_genes = 60, 10
    rng = np.random.default_rng(42)
    X = rng.random((n_cells, n_genes)).astype(np.float32)
    X[:30, :5] += 1.5  # disease cells higher for genes 0-4

    obs = pd.DataFrame(
        {"leiden_coarse": ["s0"] * 30 + ["s1"] * 30},
        index=[str(i) for i in range(n_cells)],
    )
    var = pd.DataFrame(index=[f"GENE{i}" for i in range(n_genes)])
    return anndata.AnnData(X=X, obs=obs, var=var)


# ---------------------------------------------------------------------------
# models/latent_mediator.py
# ---------------------------------------------------------------------------

class TestLatentProgram:
    def test_basic_creation(self):
        from models.latent_mediator import LatentProgram
        lp = LatentProgram(
            program_id="IBD_macro_P00",
            disease="IBD",
            cell_type="macrophage",
            program_index=0,
        )
        assert lp.program_type == "unknown"
        assert lp.top_genes == []

    def test_program_index_negative_raises(self):
        from models.latent_mediator import LatentProgram
        with pytest.raises(Exception):
            LatentProgram(
                program_id="x", disease="IBD", cell_type="macro", program_index=-1
            )


class TestConditionalBeta:
    def test_pooled_flag(self):
        from models.latent_mediator import ConditionalBeta
        cb = ConditionalBeta(
            gene="TNF", program_id="IBD_macro_P00",
            cell_type="macrophage", disease="IBD",
            beta=0.5, pooled_fallback=True,
        )
        assert cb.pooled_fallback is True
        assert not cb.context_verified

    def test_nan_beta_allowed(self):
        from models.latent_mediator import ConditionalBeta
        cb = ConditionalBeta(
            gene="TNF", program_id="p", cell_type="m", disease="IBD",
            beta=float("nan"),
        )
        assert math.isnan(cb.beta)


class TestConditionalGamma:
    def test_alpha_bounds(self):
        from models.latent_mediator import ConditionalGamma
        with pytest.raises(Exception):
            ConditionalGamma(
                program_id="p", trait="CAD", disease="CAD",
                gamma_gwas=0.3, gamma_transition=0.5,
                alpha=1.5, gamma_mixed=0.4,
            )

    def test_mixed_gamma_stored(self):
        from models.latent_mediator import ConditionalGamma
        cg = ConditionalGamma(
            program_id="p", trait="CAD", disease="CAD",
            gamma_gwas=0.4, gamma_transition=0.6,
            alpha=0.5, gamma_mixed=0.5,
        )
        assert cg.gamma_mixed == 0.5


class TestProgramLoading:
    def test_valid_creation(self):
        from models.latent_mediator import ProgramLoading
        pl = ProgramLoading(
            gene="TNF", program_id="IBD_macro_P00",
            cell_type="macrophage", disease="IBD",
            nmf_loading=0.8, transition_de_signal=0.6,
            p_loading=0.7 * 0.8 + 0.3 * 0.6,
        )
        assert abs(pl.p_loading - 0.74) < 1e-6

    def test_nan_rejected(self):
        from models.latent_mediator import ProgramLoading
        with pytest.raises(Exception):
            ProgramLoading(
                gene="TNF", program_id="p", cell_type="m", disease="IBD",
                nmf_loading=float("nan"), transition_de_signal=0.5, p_loading=0.5,
            )


class TestTherapeuticRedirectionResult:
    def test_final_score_formula(self):
        """Phase F unified formula: core × t_mod × risk_discount."""
        from models.latent_mediator import TherapeuticRedirectionResult
        thr = TherapeuticRedirectionResult(
            gene="TNF", disease="IBD",
            therapeutic_redirection=0.5,
            durability_score=0.2,
            escape_risk=0.1,
            failure_risk=0.05,
            ot_combined=0.8,
            trial_bonus=0.5,
            safety_penalty=0.2,
            genetic_grounding=0.49,   # 0.49/0.7 = 0.7 → genetic_component=0.7
            state_influence_score=0.0,
        )
        # genetic_component = min(0.49/0.7, 1.0) = 0.7
        # mech_raw = min(|0.5| + 0.0*0.3, 1.0) = 0.5  [no transition scores]
        # core = 0.60*0.7 + 0.40*0.5 = 0.62
        # tau_bonus = clamp(0.15*(0.5-0.4), -0.05, 0.10) = 0.015  [Phase R; default tau=0.5]
        # t_mod = clamp(1 + 0.15*0.8 + 0.10*0.5 - 0.10*0.2 + 0.015, 0.5, 1.5) = 1.165
        # risk_discount = max(0.1, 1 - 0.20*0.1 - 0.15*0.05) = 0.9725
        # expected = 0.62 * 1.165 * 0.9725
        expected = 0.62 * 1.165 * 0.9725
        assert abs(thr.final_score - expected) < 1e-6
        assert 0 < thr.final_score <= 1.5

    def test_context_confidence_warning(self):
        from models.latent_mediator import TherapeuticRedirectionResult
        thr = TherapeuticRedirectionResult(
            gene="IL6", disease="IBD",
            therapeutic_redirection=0.3,
            pooled_fraction=0.6,
            context_confidence_warning=True,
        )
        assert thr.context_confidence_warning is True


class TestEvidenceDisagreementRecord:
    def test_all_rules_valid(self):
        from models.latent_mediator import EvidenceDisagreementRecord, DisagreementRule
        rules: list[DisagreementRule] = [
            "genetics_vs_perturbation",
            "perturbation_vs_chemical",
            "bulk_vs_singlecell",
            "cross_context_sign_flip",
        ]
        for rule in rules:
            rec = EvidenceDisagreementRecord(gene="TNF", disease="IBD", rule=rule)
            assert rec.rule == rule

    def test_severity_default(self):
        from models.latent_mediator import EvidenceDisagreementRecord
        rec = EvidenceDisagreementRecord(gene="TNF", disease="IBD", rule="bulk_vs_singlecell")
        assert rec.severity == "warning"


# ---------------------------------------------------------------------------
# program_labeler.py
# ---------------------------------------------------------------------------

class TestProgramLabeler:
    def test_inflammatory_label(self, minimal_nmf_result):
        from pipelines.state_space.program_labeler import label_programs
        programs = label_programs(minimal_nmf_result, "IBD", "macrophage")
        assert len(programs) == 3
        p0 = programs[0]
        assert p0.program_type == "inflammatory"
        assert any("TNFA" in h or "INFLAMMATORY" in h for h in p0.hallmark_annotations)

    def test_fibrotic_label(self, minimal_nmf_result):
        from pipelines.state_space.program_labeler import label_programs
        programs = label_programs(minimal_nmf_result, "IBD", "macrophage")
        assert programs[1].program_type == "fibrotic"

    def test_unknown_label(self, minimal_nmf_result):
        from pipelines.state_space.program_labeler import label_programs
        programs = label_programs(minimal_nmf_result, "IBD", "macrophage")
        assert programs[2].program_type == "unknown"

    def test_program_id_includes_disease_celltype(self, minimal_nmf_result):
        from pipelines.state_space.program_labeler import label_programs
        programs = label_programs(minimal_nmf_result, "IBD", "macrophage")
        for p in programs:
            assert "IBD" in p.program_id
            assert "macrophage" in p.program_id

    def test_gwas_enrichment_injected(self, minimal_nmf_result):
        from pipelines.state_space.program_labeler import label_programs
        programs = label_programs(
            minimal_nmf_result, "IBD", "macrophage",
            gwas_enrichment={"IBD_macrophage_P00": 4.5},
        )
        assert programs[0].gwas_enrichment_score == 4.5
        assert programs[1].gwas_enrichment_score is None

    def test_multi_celltype(self, minimal_nmf_result):
        from pipelines.state_space.program_labeler import label_programs_multi_celltype
        results = label_programs_multi_celltype(
            {"macrophage": minimal_nmf_result, "enterocyte": minimal_nmf_result},
            "IBD",
        )
        assert "macrophage" in results
        assert "enterocyte" in results
        assert len(results["macrophage"]) == 3

    def test_empty_programs(self):
        from pipelines.state_space.program_labeler import label_programs
        programs = label_programs({"programs": [], "source": "NMF"}, "IBD", "macro")
        assert programs == []


# ---------------------------------------------------------------------------
# program_loading.py
# ---------------------------------------------------------------------------

class TestProgramLoading:
    def test_p_loading_formula(self, minimal_nmf_result, transition_weights):
        from pipelines.state_space.program_loading import compute_program_loading
        loadings = compute_program_loading(minimal_nmf_result, transition_weights, "IBD", "macrophage")
        assert len(loadings) > 0
        tnf_load = next(pl for pl in loadings if pl.gene == "TNF")
        # nmf_loading is L1-normalised: 0.9/(0.9+0.7+0.5) = 0.9/2.1 ≈ 0.4286
        expected_nmf = 0.9 / (0.9 + 0.7 + 0.5)
        expected_p = 0.7 * expected_nmf + 0.3 * 0.9   # tw["TNF"] = 0.9
        assert abs(tnf_load.nmf_loading - expected_nmf) < 1e-5
        assert abs(tnf_load.p_loading - expected_p) < 1e-5

    def test_gene_absent_in_tw_gets_de_zero(self, minimal_nmf_result):
        from pipelines.state_space.program_loading import compute_program_loading
        loadings = compute_program_loading(minimal_nmf_result, {}, "IBD", "macrophage")
        for pl in loadings:
            assert pl.transition_de_signal == 0.0

    def test_sorted_by_program_then_p_loading(self, minimal_nmf_result, transition_weights):
        from pipelines.state_space.program_loading import compute_program_loading
        loadings = compute_program_loading(minimal_nmf_result, transition_weights, "IBD", "macrophage")
        prev_prog = None
        prev_val = float("inf")
        for pl in loadings:
            if pl.program_id != prev_prog:
                prev_prog = pl.program_id
                prev_val = float("inf")
            assert pl.p_loading <= prev_val + 1e-9
            prev_val = pl.p_loading

    def test_multi_celltype(self, minimal_nmf_result, transition_weights):
        from pipelines.state_space.program_loading import compute_program_loading_multi_celltype
        results = compute_program_loading_multi_celltype(
            {"macrophage": minimal_nmf_result, "enterocyte": minimal_nmf_result},
            {"macrophage": transition_weights, "enterocyte": transition_weights},
            "IBD",
        )
        assert "macrophage" in results
        assert "enterocyte" in results
        assert all(pl.cell_type == "macrophage" for pl in results["macrophage"])
        assert all(pl.cell_type == "enterocyte" for pl in results["enterocyte"])

    def test_empty_gene_loadings_skipped(self):
        from pipelines.state_space.program_loading import compute_program_loading
        nmf = {"programs": [{"program_id": "P00", "top_genes": [], "gene_loadings": {}}]}
        loadings = compute_program_loading(nmf, {}, "IBD", "macro")
        assert loadings == []


# ---------------------------------------------------------------------------
# compute_transition_gene_weights
# ---------------------------------------------------------------------------

class TestTransitionGeneWeights:
    def test_basic(self, synthetic_adata):
        from pipelines.state_space.transition_graph import compute_transition_gene_weights
        tr = {"pathologic_basin_ids": ["s0"], "healthy_basin_ids": ["s1"]}
        weights = compute_transition_gene_weights(synthetic_adata, tr)
        # Genes 0-4 should have high weight; genes 5-9 lower
        assert len(weights) > 0
        all_vals = list(weights.values())
        assert max(all_vals) <= 1.0
        assert min(all_vals) >= 0.0
        # At least one disease-enriched gene (GENE0–GENE4) should be in top 5
        top_genes = set(list(weights.keys())[:5])
        assert any(g in top_genes for g in [f"GENE{i}" for i in range(5)])

    def test_empty_basin_ids(self, synthetic_adata):
        from pipelines.state_space.transition_graph import compute_transition_gene_weights
        weights = compute_transition_gene_weights(synthetic_adata, {})
        assert weights == {}

    def test_too_few_cells(self):
        from pipelines.state_space.transition_graph import compute_transition_gene_weights
        import anndata, pandas as pd
        X = np.ones((5, 4), dtype=np.float32)
        obs = pd.DataFrame({"leiden_coarse": ["s0"] * 3 + ["s1"] * 2}, index=list("abcde"))
        var = pd.DataFrame(index=["G0", "G1", "G2", "G3"])
        adata = anndata.AnnData(X=X, obs=obs, var=var)
        tr = {"pathologic_basin_ids": ["s0"], "healthy_basin_ids": ["s1"]}
        weights = compute_transition_gene_weights(adata, tr)
        assert weights == {}  # < 10 cells per group

    def test_returns_top_n(self, synthetic_adata):
        from pipelines.state_space.transition_graph import compute_transition_gene_weights
        tr = {"pathologic_basin_ids": ["s0"], "healthy_basin_ids": ["s1"]}
        weights = compute_transition_gene_weights(synthetic_adata, tr, n_top_genes=3)
        assert len(weights) == 3


# ---------------------------------------------------------------------------
# build_multi_celltype_latent_space + merge_multi_celltype_adata
# ---------------------------------------------------------------------------

class TestMultiCelltypeLatent:
    @pytest.fixture()
    def two_cell_adata_map(self, tmp_path):
        """Create two tiny h5ad files (different cell types).
        Dense data ensures no cells are filtered during preprocessing."""
        import anndata, pandas as pd
        adatas = {}
        for ct, prefix in [("macrophage", "M"), ("enterocyte", "E")]:
            rng = np.random.default_rng(1)
            # Dense matrix: all cells have expression in all genes (no filtering risk)
            X = rng.random((30, 8)).astype(np.float32) + 0.5
            obs = pd.DataFrame(
                {"cell_type": [ct] * 30},
                index=[f"{prefix}{i}" for i in range(30)],
            )
            var = pd.DataFrame(index=[f"G{i}" for i in range(8)])
            adata = anndata.AnnData(X=X, obs=obs, var=var)
            p = tmp_path / f"{ct}.h5ad"
            adata.write_h5ad(str(p))
            adatas[ct] = str(p)
        return adatas

    def test_separate_latent_spaces(self, two_cell_adata_map):
        from pipelines.state_space.latent_model import build_multi_celltype_latent_space
        results = build_multi_celltype_latent_space(
            "IBD", two_cell_adata_map
        )
        assert "macrophage" in results
        assert "enterocyte" in results
        assert results["macrophage"].get("error") is None
        assert results["enterocyte"].get("error") is None
        # Independent latent matrices
        lm_mac = results["macrophage"]["latent_matrix"]
        lm_ent = results["enterocyte"]["latent_matrix"]
        assert lm_mac is not None
        assert lm_ent is not None
        # They should differ (different data)
        assert lm_mac.shape[0] == 30
        assert lm_ent.shape[0] == 30

    def test_merge_adds_cell_type_source(self, two_cell_adata_map):
        from pipelines.state_space.latent_model import (
            build_multi_celltype_latent_space,
            merge_multi_celltype_adata,
        )
        results = build_multi_celltype_latent_space("IBD", two_cell_adata_map)
        merged = merge_multi_celltype_adata(results, "IBD")
        assert merged.get("error") is None
        adata = merged["adata"]
        assert "cell_type_source" in adata.obs.columns
        assert set(adata.obs["cell_type_source"].unique()) == {"macrophage", "enterocyte"}
        assert merged["n_cells_total"] == 60

    def test_merge_with_one_failed_result(self, two_cell_adata_map):
        from pipelines.state_space.latent_model import (
            build_multi_celltype_latent_space,
            merge_multi_celltype_adata,
        )
        results = build_multi_celltype_latent_space("IBD", two_cell_adata_map)
        # Simulate failure for one cell type
        results["enterocyte"] = {"error": "simulated failure", "adata": None}
        merged = merge_multi_celltype_adata(results, "IBD")
        assert merged.get("error") is None  # partial merge succeeds
        assert "macrophage" in merged["cell_types"]
        assert "enterocyte" not in merged["cell_types"]
        assert len(merged["warnings"]) > 0

    def test_merge_all_failed(self):
        from pipelines.state_space.latent_model import merge_multi_celltype_adata
        results = {
            "macrophage": {"error": "fail", "adata": None},
            "enterocyte": {"error": "fail", "adata": None},
        }
        merged = merge_multi_celltype_adata(results, "IBD")
        assert "error" in merged
        assert merged["n_cells_total"] == 0
