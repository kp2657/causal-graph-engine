"""
tests/test_state_space_transition_graph.py

Unit tests for pipelines/state_space/transition_graph.py.

Tests the pseudotime + kNN transition matrix, basin assignment,
StateTransition object construction, and the public API contract.
All tests use synthetic data via _NumpyPCABackend.
"""
import numpy as np
import pytest

anndata = pytest.importorskip("anndata")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_full_pipeline_result(
    n_cells: int = 200,
    n_genes: int = 60,
    n_disease: int = 110,
    resolution: str = "intermediate",
    seed: int = 0,
):
    """Return (latent_result, state_result) ready for transition inference."""
    import pandas as pd
    from pipelines.state_space.latent_model import build_disease_latent_space
    from pipelines.state_space.state_definition import define_cell_states

    rng = np.random.default_rng(seed)
    X = rng.negative_binomial(5, 0.5, size=(n_cells, n_genes)).astype(float)
    obs = pd.DataFrame({
        "disease_condition": ["disease"] * n_disease + ["control"] * (n_cells - n_disease),
        "donor_id":  [f"D{i % 4}" for i in range(n_cells)],
        "cell_type": ["macrophage"] * n_cells,
    })
    var = pd.DataFrame(index=[f"GENE{i:03d}" for i in range(n_genes)])
    adata = anndata.AnnData(X=X, obs=obs, var=var)

    latent = build_disease_latent_space("IBD", [], _adata_override=adata)
    if latent.get("error"):
        return None, None

    states = define_cell_states(latent, "IBD")
    return latent, states


# ---------------------------------------------------------------------------
# Unit tests for internal helpers
# ---------------------------------------------------------------------------

class TestBuildPseudotimeTransitionMatrix:
    def test_row_normalised(self):
        from pipelines.state_space.transition_graph import (
            _build_pseudotime_transition_matrix,
        )
        from scipy.sparse import csr_matrix

        n_cells  = 20
        n_states = 3
        rng = np.random.default_rng(1)
        assignments = np.array([str(i % n_states) for i in range(n_cells)])
        pseudotime  = np.linspace(0, 1, n_cells)
        conn = csr_matrix(
            (np.ones(n_cells * 4), (
                np.repeat(np.arange(n_cells), 4),
                (np.arange(n_cells)[:, None] + np.array([1, 2, 3, 4])).clip(0, n_cells - 1).ravel(),
            )),
            shape=(n_cells, n_cells),
        )
        T = _build_pseudotime_transition_matrix(
            assignments, pseudotime, conn, [str(i) for i in range(n_states)]
        )
        row_sums = T.sum(axis=1)
        for s in row_sums:
            assert s == pytest.approx(1.0) or s == pytest.approx(0.0)

    def test_no_diagonal(self):
        from pipelines.state_space.transition_graph import (
            _build_pseudotime_transition_matrix,
        )
        from scipy.sparse import csr_matrix

        n_cells = 30
        n_states = 4
        assignments = np.array([str(i % n_states) for i in range(n_cells)])
        pseudotime  = np.linspace(0, 1, n_cells)
        rows  = np.repeat(np.arange(n_cells), 3)
        cols  = np.clip(rows + np.tile([1, 2, 3], n_cells), 0, n_cells - 1)
        conn  = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_cells, n_cells))
        T = _build_pseudotime_transition_matrix(
            assignments, pseudotime, conn, [str(i) for i in range(n_states)]
        )
        assert np.diag(T).sum() == pytest.approx(0.0)


class TestAssignBasins:
    def test_high_pathological_score_gives_pathological_basin(self):
        from models.evidence import CellState
        from pipelines.state_space.transition_graph import _assign_basins
        states = [
            CellState(state_id="s0", disease="IBD", cell_type="mac",
                      resolution="intermediate", n_cells=50,
                      pathological_score=0.85, stability_score=0.7),
        ]
        basin_map = _assign_basins(states)
        assert basin_map["s0"] == "pathological"

    def test_low_pathological_score_gives_healthy_basin(self):
        from models.evidence import CellState
        from pipelines.state_space.transition_graph import _assign_basins
        states = [
            CellState(state_id="s1", disease="IBD", cell_type="mac",
                      resolution="intermediate", n_cells=50,
                      pathological_score=0.10, stability_score=0.8),
        ]
        basin_map = _assign_basins(states)
        assert basin_map["s1"] == "healthy"

    def test_unstable_pathological_state_is_escape(self):
        from models.evidence import CellState
        from pipelines.state_space.transition_graph import _assign_basins
        from pipelines.state_space.schemas import ESCAPE_STABILITY_THRESHOLD
        states = [
            CellState(state_id="s2", disease="IBD", cell_type="mac",
                      resolution="intermediate", n_cells=20,
                      pathological_score=0.75,
                      stability_score=ESCAPE_STABILITY_THRESHOLD - 0.05),
        ]
        basin_map = _assign_basins(states)
        assert basin_map["s2"] == "escape"

    def test_none_pathological_score_gives_mixed(self):
        from models.evidence import CellState
        from pipelines.state_space.transition_graph import _assign_basins
        states = [
            CellState(state_id="s3", disease="IBD", cell_type="mac",
                      resolution="intermediate", n_cells=30,
                      pathological_score=None),
        ]
        basin_map = _assign_basins(states)
        assert basin_map["s3"] == "mixed"


# ---------------------------------------------------------------------------
# Full pipeline: infer_state_transition_graph
# ---------------------------------------------------------------------------

class TestInferStateTransitionGraph:
    def setup_method(self):
        self.latent, self.states = _make_full_pipeline_result()
        if self.latent is None:
            pytest.skip("latent build failed")

    def test_output_has_required_keys(self):
        from pipelines.state_space.transition_graph import infer_state_transition_graph
        result = infer_state_transition_graph(self.latent, self.states, "IBD")
        required = [
            "transitions", "transition_matrix", "state_labels",
            "basin_assignments", "healthy_basin_ids", "pathologic_basin_ids",
            "escape_basin_ids", "state_pseudotime", "confidence_summary",
            "disease", "resolution",
        ]
        for key in required:
            assert key in result, f"Missing key: {key}"

    def test_transitions_are_state_transition_objects(self):
        from pipelines.state_space.transition_graph import infer_state_transition_graph
        from models.evidence import StateTransition
        result = infer_state_transition_graph(self.latent, self.states, "IBD")
        for t in result["transitions"]:
            assert isinstance(t, StateTransition)

    def test_transitions_carry_pseudotime_provenance(self):
        from pipelines.state_space.transition_graph import infer_state_transition_graph
        from pipelines.state_space.schemas import PROV_PSEUDOTIME_INFERRED
        result = infer_state_transition_graph(self.latent, self.states, "IBD")
        for t in result["transitions"]:
            assert PROV_PSEUDOTIME_INFERRED in t.direction_evidence

    def test_basin_assignments_covers_all_states(self):
        from pipelines.state_space.transition_graph import infer_state_transition_graph
        result = infer_state_transition_graph(self.latent, self.states, "IBD")
        basin_ids  = set(result["basin_assignments"].keys())
        state_ids  = set(result["state_labels"])
        assert basin_ids == state_ids

    def test_healthy_and_pathologic_non_overlapping(self):
        from pipelines.state_space.transition_graph import infer_state_transition_graph
        result = infer_state_transition_graph(self.latent, self.states, "IBD")
        healthy  = set(result["healthy_basin_ids"])
        patholog = set(result["pathologic_basin_ids"])
        assert len(healthy & patholog) == 0

    def test_transition_matrix_square(self):
        from pipelines.state_space.transition_graph import infer_state_transition_graph
        result = infer_state_transition_graph(self.latent, self.states, "IBD")
        T = result["transition_matrix"]
        n = len(result["state_labels"])
        assert T.shape == (n, n)

    def test_confidence_summary_has_provenance(self):
        from pipelines.state_space.transition_graph import infer_state_transition_graph
        from pipelines.state_space.schemas import PROV_PSEUDOTIME_INFERRED
        result = infer_state_transition_graph(self.latent, self.states, "IBD")
        cs = result["confidence_summary"]
        assert cs["provenance"] == PROV_PSEUDOTIME_INFERRED

    def test_unsupported_mode_raises(self):
        from pipelines.state_space.transition_graph import infer_state_transition_graph
        with pytest.raises(NotImplementedError):
            infer_state_transition_graph(
                self.latent, self.states, "IBD", mode="velocity"
            )

    def test_error_latent_raises(self):
        from pipelines.state_space.transition_graph import infer_state_transition_graph
        with pytest.raises(ValueError, match="error"):
            infer_state_transition_graph(
                {"error": "missing h5ad", "adata": None}, {}, "IBD"
            )


class TestGetBasinSummary:
    def test_returns_summary_dict(self):
        from pipelines.state_space.transition_graph import (
            infer_state_transition_graph, get_basin_summary,
        )
        latent, states = _make_full_pipeline_result(seed=7)
        if latent is None:
            pytest.skip("latent build failed")
        result = infer_state_transition_graph(latent, states, "IBD")
        summary = get_basin_summary(result)
        assert "n_states" in summary
        assert "n_transitions" in summary
        assert "healthy_basins" in summary
        assert "pathologic_basins" in summary
