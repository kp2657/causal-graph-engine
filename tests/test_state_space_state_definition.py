"""
tests/test_state_space_state_definition.py

Unit tests for pipelines/state_space/state_definition.py.

Uses synthetic latent_result (built from _NumpyPCABackend on a synthetic AnnData).
Skips if anndata is absent.
"""
import numpy as np
import pytest

anndata = pytest.importorskip("anndata")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_latent_result(
    n_cells: int = 150,
    n_genes: int = 60,
    n_disease: int | None = None,
    seed: int = 42,
) -> dict:
    """Build a real latent_result using _NumpyPCABackend on synthetic data."""
    import pandas as pd
    from pipelines.state_space.latent_model import build_disease_latent_space

    if n_disease is None:
        n_disease = n_cells // 2
    n_disease = min(n_disease, n_cells)

    rng = np.random.default_rng(seed)
    X = rng.negative_binomial(5, 0.5, size=(n_cells, n_genes)).astype(float)
    obs = pd.DataFrame({
        "disease_condition": ["disease"] * n_disease + ["control"] * (n_cells - n_disease),
        "donor_id":  [f"D{i % 5}" for i in range(n_cells)],
        "cell_type": ["macrophage"] * (n_cells // 2) + ["T_cell"] * (n_cells // 2),
    })
    var = pd.DataFrame(index=[f"GENE{i:03d}" for i in range(n_genes)])
    adata = anndata.AnnData(X=X, obs=obs, var=var)

    return build_disease_latent_space("IBD", [], _adata_override=adata)


# ---------------------------------------------------------------------------
# define_cell_states
# ---------------------------------------------------------------------------

class TestDefineCellStates:
    def test_returns_all_three_resolutions(self):
        from pipelines.state_space.state_definition import define_cell_states
        latent = _make_synthetic_latent_result()
        if latent.get("error"):
            pytest.skip(f"latent build failed: {latent['error']}")
        result = define_cell_states(latent, "IBD")
        assert "coarse" in result
        assert "intermediate" in result
        assert "fine" in result

    def test_each_resolution_is_list_of_cell_states(self):
        from pipelines.state_space.state_definition import define_cell_states
        from models.evidence import CellState
        latent = _make_synthetic_latent_result()
        if latent.get("error"):
            pytest.skip(f"latent build failed: {latent['error']}")
        result = define_cell_states(latent, "IBD")
        for res_name, states in result.items():
            assert isinstance(states, list), f"{res_name} not a list"
            for s in states:
                assert isinstance(s, CellState), f"expected CellState, got {type(s)}"

    def test_coarse_has_fewer_states_than_fine(self):
        from pipelines.state_space.state_definition import define_cell_states
        latent = _make_synthetic_latent_result()
        if latent.get("error"):
            pytest.skip(f"latent build failed: {latent['error']}")
        result = define_cell_states(latent, "IBD")
        assert len(result["coarse"]) <= len(result["fine"])

    def test_n_cells_sums_to_total(self):
        from pipelines.state_space.state_definition import define_cell_states
        latent = _make_synthetic_latent_result(n_cells=100)
        if latent.get("error"):
            pytest.skip(f"latent build failed: {latent['error']}")
        result = define_cell_states(latent, "IBD")
        # Each resolution should partition all cells
        for res_name, states in result.items():
            total = sum(s.n_cells for s in states)
            assert total == 100, f"{res_name}: n_cells sum {total} != 100"

    def test_state_ids_contain_disease_and_resolution(self):
        from pipelines.state_space.state_definition import define_cell_states
        latent = _make_synthetic_latent_result()
        if latent.get("error"):
            pytest.skip(f"latent build failed: {latent['error']}")
        result = define_cell_states(latent, "IBD")
        for res_name, states in result.items():
            for s in states:
                assert "IBD" in s.state_id
                assert res_name in s.state_id

    def test_pathological_score_in_zero_one(self):
        from pipelines.state_space.state_definition import define_cell_states
        latent = _make_synthetic_latent_result()
        if latent.get("error"):
            pytest.skip(f"latent build failed: {latent['error']}")
        result = define_cell_states(latent, "IBD")
        for states in result.values():
            for s in states:
                if s.pathological_score is not None:
                    assert 0.0 <= s.pathological_score <= 1.0

    def test_stability_score_in_zero_one(self):
        from pipelines.state_space.state_definition import define_cell_states
        latent = _make_synthetic_latent_result()
        if latent.get("error"):
            pytest.skip(f"latent build failed: {latent['error']}")
        result = define_cell_states(latent, "IBD")
        for states in result.values():
            for s in states:
                if s.stability_score is not None:
                    assert 0.0 <= s.stability_score <= 1.0 + 1e-9

    def test_program_labels_overlay(self):
        from pipelines.state_space.state_definition import define_cell_states
        latent = _make_synthetic_latent_result(n_cells=120, n_genes=60)
        if latent.get("error"):
            pytest.skip(f"latent build failed: {latent['error']}")
        # Inject a fake program whose genes overlap heavily with the data
        prog_labels = {"NF_kB_SIGNALING": [f"GENE{i:03d}" for i in range(15)]}
        result = define_cell_states(latent, "IBD", program_labels_source=prog_labels)
        # At least one state should pick up this program
        all_prog_labels = [
            pl for states in result.values()
            for s in states for pl in s.program_labels
        ]
        assert "NF_kB_SIGNALING" in all_prog_labels

    def test_error_on_bad_latent_result(self):
        from pipelines.state_space.state_definition import define_cell_states
        with pytest.raises(ValueError):
            define_cell_states({"error": "missing h5ad"}, "IBD")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

class TestGetPathologicalAndHealthyStates:
    def test_pathological_states_above_threshold(self):
        from pipelines.state_space.state_definition import (
            define_cell_states, get_pathological_states,
        )
        from pipelines.state_space.schemas import PATHOLOGICAL_ENRICHMENT_THRESHOLD
        latent = _make_synthetic_latent_result(n_cells=200, n_disease=170)
        if latent.get("error"):
            pytest.skip(f"latent build failed: {latent['error']}")
        result = define_cell_states(latent, "IBD")
        path_states = get_pathological_states(result, "intermediate")
        for s in path_states:
            assert s.pathological_score >= PATHOLOGICAL_ENRICHMENT_THRESHOLD

    def test_healthy_states_below_threshold(self):
        from pipelines.state_space.state_definition import (
            define_cell_states, get_healthy_states,
        )
        from pipelines.state_space.schemas import HEALTHY_ENRICHMENT_THRESHOLD
        latent = _make_synthetic_latent_result(n_cells=200, n_disease=30)
        if latent.get("error"):
            pytest.skip(f"latent build failed: {latent['error']}")
        result = define_cell_states(latent, "IBD")
        health_states = get_healthy_states(result, "intermediate")
        for s in health_states:
            assert s.pathological_score <= (1.0 - HEALTHY_ENRICHMENT_THRESHOLD) + 1e-9


# ---------------------------------------------------------------------------
# State-definition caching
# ---------------------------------------------------------------------------

class TestStateDefinitionCaching:
    def _make_latent_result_with_file_anchor(self, tmp_path):
        """Build a latent_result with a real file anchor for cache path derivation."""
        import pandas as pd
        import anndata as ad
        from pipelines.state_space.latent_model import build_disease_latent_space

        anchor = tmp_path / "latent_cache_IBD_macrophage_pca_diffusion.h5ad"
        ad.AnnData(np.zeros((3, 3))).write_h5ad(str(anchor))

        rng = np.random.default_rng(99)
        X = rng.random((80, 20))
        obs = pd.DataFrame({
            "disease_condition": ["disease"] * 40 + ["control"] * 40,
        })
        adata = ad.AnnData(X=X, obs=obs)
        result = build_disease_latent_space("IBD", [], _adata_override=adata)
        result["provenance"]["cache_file"] = str(anchor)
        return result

    def test_cache_hit_returns_identical_states(self, tmp_path):
        from pipelines.state_space.state_definition import define_cell_states
        latent = self._make_latent_result_with_file_anchor(tmp_path)
        if latent.get("error"):
            pytest.skip(f"latent build failed: {latent['error']}")

        res1 = define_cell_states(latent, "IBD", use_cache=True)
        res2 = define_cell_states(latent, "IBD", use_cache=True)

        for resolution in res1:
            ids1 = [s.state_id for s in res1[resolution]]
            ids2 = [s.state_id for s in res2[resolution]]
            assert ids1 == ids2, f"State IDs differ at {resolution} on cache hit"

    def test_cache_disabled_does_not_write_files(self, tmp_path):
        from pipelines.state_space.state_definition import define_cell_states, _state_cache_paths
        from pipelines.state_space.schemas import DEFAULT_RESOLUTIONS
        latent = self._make_latent_result_with_file_anchor(tmp_path)
        if latent.get("error"):
            pytest.skip(f"latent build failed: {latent['error']}")

        define_cell_states(latent, "IBD", use_cache=False)
        state_path, meta_path = _state_cache_paths(latent, DEFAULT_RESOLUTIONS)
        assert not state_path.exists(), "Cache file should not be written when use_cache=False"
