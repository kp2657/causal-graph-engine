"""
tests/test_phase_g_transition_scoring.py

Unit tests for Phase G: transition-aware gene scoring.

Tests:
  - TransitionGeneProfile model fields and defaults
  - compute_transition_gene_scores output shape and value ranges
  - Entry / persistence / recovery cell identification from T_baseline
  - Boundary kNN mask (sparse matrix-vector product)
  - Boundary pseudotime mask (temporal inflection)
  - Direction signs
  - Category assignment
  - Backward-compat compute_gene_state_influence wrapper
  - Graceful degradation (no state col, no T, missing adata)
"""
from __future__ import annotations

import numpy as np
import pytest

anndata = pytest.importorskip("anndata")


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------

def _make_transition_result(
    n_path: int = 3,
    n_healthy: int = 3,
    disease: str = "IBD",
    resolution: str = "intermediate",
    rng_seed: int = 0,
) -> dict:
    """
    Minimal transition_result with a non-degenerate T_baseline.

    Row layout: [path_0, path_1, path_2, healthy_3, healthy_4, healthy_5]

    Designed so that:
      - path_0: low healthy outflow → persistence cluster
      - path_1: high healthy outflow → recovery cluster
      - path_2: mixed
      - healthy_3/4/5: high outbound flow toward path_0 → entry clusters
    """
    n_states = n_path + n_healthy
    prefix = f"{disease}_{resolution}_"
    state_labels = [f"{prefix}{i}" for i in range(n_states)]
    path_ids  = state_labels[:n_path]
    health_ids = state_labels[n_path:]

    T = np.zeros((n_states, n_states))

    # Pathological clusters — varying healthy outflow
    T[0, 1]         = 0.90  # path_0 → path_1  (mostly stays pathological → persistence)
    T[0, n_path]    = 0.10  # path_0 → healthy_3 (low exit)

    T[1, n_path]    = 0.80  # path_1 → healthy_3 (high exit → recovery)
    T[1, n_path+1]  = 0.20  # path_1 → healthy_4

    T[2, 0]         = 0.40  # path_2 → path_0
    T[2, n_path]    = 0.40  # path_2 → healthy_3
    T[2, n_path+1]  = 0.20  # path_2 → healthy_4

    # Healthy clusters — high outbound flow to path_0 (entry signal)
    for hi in range(n_path, n_states):
        T[hi, 0] = 0.80     # healthy → path_0 (entry)
        T[hi, hi-1 if hi > 0 else hi] = 0.20  # residual

    # Row-normalise
    row_sums = T.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    T = T / row_sums

    return {
        "disease":              disease,
        "resolution":           resolution,
        "transition_matrix":    T,
        "state_labels":         state_labels,
        "pathologic_basin_ids": path_ids,
        "healthy_basin_ids":    health_ids,
        "escape_basin_ids":     [],
        "state_pseudotime":     {s: float(i) / n_states for i, s in enumerate(state_labels)},
    }


def _make_adata(
    n_path_cells: int = 60,
    n_healthy_cells: int = 60,
    n_genes: int = 30,
    n_path_states: int = 3,
    n_healthy_states: int = 3,
    disease: str = "IBD",
    resolution: str = "intermediate",
    seed: int = 42,
) -> any:
    """
    Synthetic AnnData with:
    - disease_condition obs column
    - state_intermediate obs column (raw cluster labels 0..n_states-1)
    - dpt_pseudotime (0..1)
    - kNN connectivities (random sparse graph)
    - X: expression with a gene elevated in path cells (gene 0)
           and a gene elevated in healthy cells (gene 1)
    """
    import pandas as pd
    import scipy.sparse

    rng = np.random.default_rng(seed)
    n_cells = n_path_cells + n_healthy_cells
    n_states = n_path_states + n_healthy_states

    X = rng.random((n_cells, n_genes)).astype(float)
    # Gene 0: elevated in ALL path cells (disease-axis signal, not category-specific)
    X[:n_path_cells, 0] += 2.0
    # Gene 1: elevated in healthy cells
    X[n_path_cells:, 1] += 2.0
    # Gene 2: specifically elevated in path cluster 0 (the persistence cluster)
    #  state label "0" → cells 0, 3, 6, ... (every n_path_states-th path cell)
    persistence_cell_idx = [i for i in range(n_path_cells) if i % n_path_states == 0]
    X[persistence_cell_idx, 2] += 4.0
    # Gene 3: specifically elevated in path cluster 1 (the recovery cluster)
    recovery_cell_idx = [i for i in range(n_path_cells) if i % n_path_states == 1]
    X[recovery_cell_idx, 3] += 4.0

    # Assign cells to states (even distribution)
    path_labels = [str(i % n_path_states) for i in range(n_path_cells)]
    health_labels = [str(n_path_states + (i % n_healthy_states)) for i in range(n_healthy_cells)]
    all_labels = path_labels + health_labels

    pseudotime = np.linspace(0, 1, n_cells)  # monotone proxy

    obs = pd.DataFrame({
        "disease_condition": ["disease"] * n_path_cells + ["healthy"] * n_healthy_cells,
        f"state_{resolution}": all_labels,
        "dpt_pseudotime": pseudotime,
    })
    var = pd.DataFrame(index=[f"GENE{i:03d}" for i in range(n_genes)])

    # Random sparse kNN connectivity (symmetric)
    k = 5
    rows, cols = [], []
    for i in range(n_cells):
        nbrs = rng.choice([j for j in range(n_cells) if j != i], size=k, replace=False)
        for nb in nbrs:
            rows.append(i); cols.append(nb)
            rows.append(nb); cols.append(i)
    conn = scipy.sparse.csr_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(n_cells, n_cells)
    )
    conn.data[:] = 1.0

    adata = anndata.AnnData(X=X, obs=obs, var=var)
    adata.obsp["connectivities"] = conn
    adata.obsm["X_pca"] = rng.random((n_cells, 5))
    adata.obsm["X_diffmap"] = rng.random((n_cells, 3))
    return adata


# ---------------------------------------------------------------------------
# TransitionGeneProfile model
# ---------------------------------------------------------------------------

class TestTransitionGeneProfileModel:
    def test_defaults_are_zero(self):
        from models.evidence import TransitionGeneProfile
        p = TransitionGeneProfile(gene="NOD2", disease="IBD")
        assert p.entry_score == 0.0
        assert p.persistence_score == 0.0
        assert p.recovery_score == 0.0
        assert p.boundary_score == 0.0
        assert p.disease_axis_score == 0.0
        assert p.mechanistic_category == "unknown"

    def test_directions_are_int(self):
        from models.evidence import TransitionGeneProfile
        p = TransitionGeneProfile(
            gene="LYZ", disease="IBD",
            entry_direction=1, persistence_direction=-1, recovery_direction=0,
        )
        assert isinstance(p.entry_direction, int)
        assert p.entry_direction == 1
        assert p.persistence_direction == -1

    def test_boundary_score_is_max(self):
        from models.evidence import TransitionGeneProfile
        p = TransitionGeneProfile(
            gene="X", disease="IBD",
            boundary_knn_score=0.3,
            boundary_pseudotime_score=0.6,
            boundary_score=0.6,
        )
        assert p.boundary_score == max(p.boundary_knn_score, p.boundary_pseudotime_score)

    def test_model_roundtrip_json(self):
        from models.evidence import TransitionGeneProfile
        p = TransitionGeneProfile(
            gene="S100A9", disease="IBD",
            entry_score=0.5, persistence_score=0.8,
            entry_direction=1, mechanistic_category="maintenance",
        )
        p2 = TransitionGeneProfile.model_validate(p.model_dump())
        assert p2.gene == p.gene
        assert p2.persistence_score == p.persistence_score
        assert p2.mechanistic_category == p.mechanistic_category


# ---------------------------------------------------------------------------
# compute_transition_gene_scores
# ---------------------------------------------------------------------------

class TestComputeTransitionGeneScores:
    def test_returns_dict_keyed_by_gene(self):
        from pipelines.state_space.transition_scoring import compute_transition_gene_scores
        adata  = _make_adata()
        trans  = _make_transition_result()
        result = compute_transition_gene_scores(adata, trans, gene_list=["GENE000", "GENE001"])
        assert set(result.keys()) == {"GENE000", "GENE001"}

    def test_all_scores_in_zero_one(self):
        from pipelines.state_space.transition_scoring import compute_transition_gene_scores
        adata  = _make_adata()
        trans  = _make_transition_result()
        result = compute_transition_gene_scores(adata, trans)
        for profile in result.values():
            assert 0.0 <= profile.entry_score       <= 1.0
            assert 0.0 <= profile.persistence_score <= 1.0
            assert 0.0 <= profile.recovery_score    <= 1.0
            assert 0.0 <= profile.boundary_score    <= 1.0
            assert 0.0 <= profile.disease_axis_score <= 1.0

    def test_directions_are_valid(self):
        from pipelines.state_space.transition_scoring import compute_transition_gene_scores
        adata  = _make_adata()
        trans  = _make_transition_result()
        result = compute_transition_gene_scores(adata, trans)
        for profile in result.values():
            assert profile.entry_direction       in (-1, 0, 1)
            assert profile.persistence_direction in (-1, 0, 1)
            assert profile.recovery_direction    in (-1, 0, 1)
            assert profile.boundary_direction    in (-1, 0, 1)

    def test_persistence_cluster_gene_has_positive_persistence_direction(self):
        """GENE002 is elevated specifically in path cluster 0 (the low-exit/persistence
        cluster per T_baseline). persistence_direction should be +1."""
        from pipelines.state_space.transition_scoring import compute_transition_gene_scores
        adata  = _make_adata(seed=0)
        trans  = _make_transition_result()
        result = compute_transition_gene_scores(adata, trans, gene_list=["GENE002"])
        # persistence cells = cluster 0 (low healthy outflow)
        # recovery cells    = cluster 1 (high healthy outflow)
        # GENE002 is elevated in cluster 0 → persistence_direction = +1
        assert result["GENE002"].persistence_direction == 1

    def test_recovery_cluster_gene_has_positive_recovery_direction(self):
        """GENE003 is elevated specifically in path cluster 1 (the high-exit/recovery
        cluster per T_baseline). recovery_direction should be +1."""
        from pipelines.state_space.transition_scoring import compute_transition_gene_scores
        adata  = _make_adata(seed=0)
        trans  = _make_transition_result()
        result = compute_transition_gene_scores(adata, trans, gene_list=["GENE003"])
        assert result["GENE003"].recovery_direction == 1

    def test_missing_gene_returns_zero_profile(self):
        from pipelines.state_space.transition_scoring import compute_transition_gene_scores
        adata  = _make_adata()
        trans  = _make_transition_result()
        result = compute_transition_gene_scores(adata, trans, gene_list=["NONEXISTENT_XYZ"])
        p = result["NONEXISTENT_XYZ"]
        assert p.entry_score == 0.0
        assert p.persistence_score == 0.0
        assert p.mechanistic_category == "unknown"

    def test_none_adata_returns_empty(self):
        from pipelines.state_space.transition_scoring import compute_transition_gene_scores
        result = compute_transition_gene_scores(None, {})
        assert result == {}

    def test_no_basins_returns_zero_profiles(self):
        from pipelines.state_space.transition_scoring import compute_transition_gene_scores
        adata = _make_adata()
        trans = {"disease": "IBD", "resolution": "intermediate",
                 "pathologic_basin_ids": [], "healthy_basin_ids": []}
        result = compute_transition_gene_scores(adata, trans, gene_list=["GENE000"])
        assert result["GENE000"].entry_score == 0.0

    def test_boundary_score_is_max_of_knn_and_pt(self):
        from pipelines.state_space.transition_scoring import compute_transition_gene_scores
        adata  = _make_adata()
        trans  = _make_transition_result()
        result = compute_transition_gene_scores(adata, trans)
        for p in result.values():
            assert abs(p.boundary_score - max(p.boundary_knn_score, p.boundary_pseudotime_score)) < 1e-9

    def test_cell_counts_are_positive(self):
        from pipelines.state_space.transition_scoring import compute_transition_gene_scores
        adata  = _make_adata()
        trans  = _make_transition_result()
        result = compute_transition_gene_scores(adata, trans, gene_list=["GENE000"])
        p = result["GENE000"]
        assert p.n_entry_cells > 0
        assert p.n_persistence_cells > 0
        assert p.n_recovery_cells > 0

    def test_category_assigned(self):
        from pipelines.state_space.transition_scoring import compute_transition_gene_scores
        adata  = _make_adata()
        trans  = _make_transition_result()
        result = compute_transition_gene_scores(adata, trans)
        valid_cats = {"trigger", "maintenance", "recovery", "mixed", "unknown"}
        for p in result.values():
            assert p.mechanistic_category in valid_cats


# ---------------------------------------------------------------------------
# Category assignment
# ---------------------------------------------------------------------------

class TestAssignCategory:
    def test_trigger_when_entry_dominant(self):
        from pipelines.state_space.transition_scoring import _assign_category
        assert _assign_category(entry=0.5, persistence=0.1, recovery=0.1, boundary=0.1) == "trigger"

    def test_maintenance_when_persistence_dominant(self):
        from pipelines.state_space.transition_scoring import _assign_category
        assert _assign_category(entry=0.1, persistence=0.6, recovery=0.1, boundary=0.1) == "maintenance"

    def test_recovery_when_recovery_dominant(self):
        from pipelines.state_space.transition_scoring import _assign_category
        assert _assign_category(entry=0.1, persistence=0.1, recovery=0.5, boundary=0.1) == "recovery"

    def test_mixed_when_all_below_threshold(self):
        from pipelines.state_space.transition_scoring import _assign_category
        assert _assign_category(entry=0.05, persistence=0.05, recovery=0.05, boundary=0.05) == "mixed"


# ---------------------------------------------------------------------------
# Backward-compat wrapper
# ---------------------------------------------------------------------------

class TestComputeGeneStateInfluence:
    def test_legacy_keys_present(self):
        from pipelines.state_space.state_influence import compute_gene_state_influence
        adata  = _make_adata()
        trans  = _make_transition_result()
        result = compute_gene_state_influence(adata, trans, gene_list=["GENE000"])
        d = result["GENE000"]
        assert "disease_axis_score" in d
        assert "directionality"     in d
        assert "entry_score"        in d
        assert "persistence_score"  in d
        assert "recovery_score"     in d
        assert "mechanistic_category" in d

    def test_legacy_das_is_float(self):
        from pipelines.state_space.state_influence import compute_gene_state_influence
        adata  = _make_adata()
        trans  = _make_transition_result()
        result = compute_gene_state_influence(adata, trans, gene_list=["GENE000"])
        assert isinstance(result["GENE000"]["disease_axis_score"], float)

    def test_new_function_returns_profiles(self):
        from pipelines.state_space.state_influence import compute_gene_transition_profiles
        from models.evidence import TransitionGeneProfile
        adata  = _make_adata()
        trans  = _make_transition_result()
        result = compute_gene_transition_profiles(adata, trans, gene_list=["GENE000"])
        assert isinstance(result["GENE000"], TransitionGeneProfile)
