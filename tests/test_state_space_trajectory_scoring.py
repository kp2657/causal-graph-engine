"""
tests/test_state_space_trajectory_scoring.py

Unit tests for pipelines/state_space/trajectory_scoring.py.
Uses synthetic transition_result and state_result fixtures.
"""
import numpy as np
import pytest

anndata = pytest.importorskip("anndata")


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------

def _make_synthetic_pipeline_result(n_cells=120, n_genes=60, seed=42):
    """Build real transition_result + state_result using synthetic data."""
    import pandas as pd
    from pipelines.state_space.latent_model import build_disease_latent_space
    from pipelines.state_space.state_definition import define_cell_states
    from pipelines.state_space.transition_graph import infer_state_transition_graph

    rng = np.random.default_rng(seed)
    X = rng.negative_binomial(5, 0.5, size=(n_cells, n_genes)).astype(float)
    n_disease = n_cells // 2
    obs = pd.DataFrame({
        "disease_condition": ["disease"] * n_disease + ["control"] * (n_cells - n_disease),
        "donor_id":  [f"D{i % 5}" for i in range(n_cells)],
        "cell_type": ["macrophage"] * (n_cells // 2) + ["T_cell"] * (n_cells // 2),
    })
    var = pd.DataFrame(index=[f"GENE{i:03d}" for i in range(n_genes)])
    adata = anndata.AnnData(X=X, obs=obs, var=var)

    latent = build_disease_latent_space("IBD", [], _adata_override=adata)
    if latent.get("error"):
        pytest.skip(f"latent build failed: {latent['error']}")
    states = define_cell_states(latent, "IBD")
    trans = infer_state_transition_graph(latent, states, "IBD")
    return states, trans


# ---------------------------------------------------------------------------
# score_gene
# ---------------------------------------------------------------------------

class TestScoreGene:
    def test_returns_trajectory_redirection_score(self):
        from pipelines.state_space.trajectory_scoring import score_gene
        from models.evidence import TrajectoryRedirectionScore
        states, trans = _make_synthetic_pipeline_result()
        score = score_gene("GENE000", trans, states, disease="IBD")
        assert isinstance(score, TrajectoryRedirectionScore)

    def test_entity_id_and_type(self):
        from pipelines.state_space.trajectory_scoring import score_gene
        states, trans = _make_synthetic_pipeline_result()
        score = score_gene("GENE001", trans, states, disease="IBD")
        assert score.entity_id == "GENE001"
        assert score.entity_type == "gene"

    def test_disease_propagated(self):
        from pipelines.state_space.trajectory_scoring import score_gene
        states, trans = _make_synthetic_pipeline_result()
        score = score_gene("GENE002", trans, states, disease="IBD")
        assert score.disease == "IBD"

    def test_all_scores_in_zero_one(self):
        from pipelines.state_space.trajectory_scoring import score_gene
        states, trans = _make_synthetic_pipeline_result()
        score = score_gene("GENE003", trans, states, disease="IBD")
        assert 0.0 <= score.expected_pathology_reduction <= 1.0
        assert 0.0 <= score.durable_redirection_score <= 1.0
        assert 0.0 <= score.escape_risk_score <= 1.0
        assert 0.0 <= score.non_response_risk_score <= 1.0
        assert 0.0 <= score.negative_memory_penalty <= 1.0

    def test_negative_memory_zero_without_records(self):
        from pipelines.state_space.trajectory_scoring import score_gene
        states, trans = _make_synthetic_pipeline_result()
        score = score_gene("GENE004", trans, states, failure_records=None)
        assert score.negative_memory_penalty == 0.0

    def test_negative_memory_nonzero_with_records(self):
        from pipelines.state_space.trajectory_scoring import score_gene
        from pipelines.state_space.failure_memory import build_failure_records
        states, trans = _make_synthetic_pipeline_result()
        # Use a real failure record for a perturbation that matches gene name pattern
        records = build_failure_records("IBD", include_ct=False)
        # Use "anti-TNF" gene name to match curated failure
        score = score_gene("anti-TNF", trans, states, failure_records=records, disease="IBD")
        assert score.negative_memory_penalty > 0.0

    def test_unknown_gene_returns_valid_score(self):
        from pipelines.state_space.trajectory_scoring import score_gene
        states, trans = _make_synthetic_pipeline_result()
        score = score_gene("TOTALLY_UNKNOWN_GENE_XYZ", trans, states, disease="IBD")
        assert isinstance(score.expected_pathology_reduction, float)


# ---------------------------------------------------------------------------
# score_all_genes
# ---------------------------------------------------------------------------

class TestScoreAllGenes:
    def test_returns_list_same_length(self):
        from pipelines.state_space.trajectory_scoring import score_all_genes
        states, trans = _make_synthetic_pipeline_result()
        genes = [f"GENE{i:03d}" for i in range(5)]
        scores = score_all_genes(genes, trans, states, disease="IBD")
        assert len(scores) == 5

    def test_sorted_descending_by_reduction(self):
        from pipelines.state_space.trajectory_scoring import score_all_genes
        states, trans = _make_synthetic_pipeline_result()
        genes = [f"GENE{i:03d}" for i in range(10)]
        scores = score_all_genes(genes, trans, states, disease="IBD")
        reductions = [s.expected_pathology_reduction for s in scores]
        assert reductions == sorted(reductions, reverse=True)

    def test_empty_gene_list_returns_empty(self):
        from pipelines.state_space.trajectory_scoring import score_all_genes
        states, trans = _make_synthetic_pipeline_result()
        scores = score_all_genes([], trans, states, disease="IBD")
        assert scores == []
