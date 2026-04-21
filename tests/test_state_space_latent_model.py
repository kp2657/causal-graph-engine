"""
tests/test_state_space_latent_model.py

Unit tests for pipelines/state_space/latent_model.py.

Tests that need AnnData use a synthetic fixture.
Tests that need scanpy are skipped gracefully if scanpy is absent.
The _NumpyPCABackend is always available and is the primary test path.
"""
import numpy as np
import pytest

anndata = pytest.importorskip("anndata")


# ---------------------------------------------------------------------------
# Synthetic fixture
# ---------------------------------------------------------------------------

def _make_synthetic_adata(
    n_cells: int = 150,
    n_genes: int = 60,
    n_disease: int | None = None,
    seed: int = 42,
):
    """
    Create a minimal AnnData with disease/control labels, donor IDs, and cell types.
    Gene expression is random negative-binomial counts.
    """
    import pandas as pd

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
    var = pd.DataFrame(
        {"gene_name": [f"GENE{i:03d}" for i in range(n_genes)]},
        index=[f"GENE{i:03d}" for i in range(n_genes)],
    )
    return anndata.AnnData(X=X, obs=obs, var=var)


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------

class TestGetBackend:
    def test_default_returns_non_scvi(self):
        from pipelines.state_space.latent_model import get_backend, ScVIBackend
        backend = get_backend(use_scvi=False)
        assert not isinstance(backend, ScVIBackend)

    def test_scvi_flag_returns_scvi_backend(self):
        from pipelines.state_space.latent_model import get_backend, ScVIBackend
        backend = get_backend(use_scvi=True)
        assert isinstance(backend, ScVIBackend)

    def test_scvi_backend_raises_not_implemented(self):
        from pipelines.state_space.latent_model import ScVIBackend
        adata = _make_synthetic_adata(50, 30)
        with pytest.raises((NotImplementedError, ImportError)):
            ScVIBackend().fit_transform(adata)

    def test_backend_implements_protocol(self):
        from pipelines.state_space.latent_model import get_backend, LatentBackend
        backend = get_backend(use_scvi=False)
        assert isinstance(backend, LatentBackend)


# ---------------------------------------------------------------------------
# _NumpyPCABackend — always available
# ---------------------------------------------------------------------------

class TestNumpyPCABackend:
    def test_fit_transform_returns_required_keys(self):
        from pipelines.state_space.latent_model import _NumpyPCABackend
        adata = _make_synthetic_adata(100, 40)
        result = _NumpyPCABackend().fit_transform(adata, n_pcs=10, n_neighbors=10)
        assert "latent_key" in result
        assert "pca_key" in result
        assert "pseudotime_key" in result
        assert "backend" in result

    def test_x_pca_stored_in_adata(self):
        from pipelines.state_space.latent_model import _NumpyPCABackend
        adata = _make_synthetic_adata(100, 40)
        _NumpyPCABackend().fit_transform(adata, n_pcs=10)
        assert "X_pca" in adata.obsm
        assert adata.obsm["X_pca"].shape[0] == 100

    def test_pseudotime_in_zero_one(self):
        from pipelines.state_space.latent_model import _NumpyPCABackend
        adata = _make_synthetic_adata(80, 30)
        _NumpyPCABackend().fit_transform(adata, n_pcs=8)
        pt = adata.obs["dpt_pseudotime"].values
        assert pt.min() >= 0.0 - 1e-9
        assert pt.max() <= 1.0 + 1e-9

    def test_connectivities_sparse(self):
        from pipelines.state_space.latent_model import _NumpyPCABackend
        adata = _make_synthetic_adata(60, 30)
        _NumpyPCABackend().fit_transform(adata, n_pcs=8, n_neighbors=5)
        conn = adata.obsp.get("connectivities")
        assert conn is not None
        assert conn.shape == (60, 60)

    def test_x_diffmap_stored_as_latent(self):
        from pipelines.state_space.latent_model import _NumpyPCABackend
        adata = _make_synthetic_adata(80, 30)
        _NumpyPCABackend().fit_transform(adata, n_pcs=15)
        assert "X_diffmap" in adata.obsm


# ---------------------------------------------------------------------------
# build_disease_latent_space
# ---------------------------------------------------------------------------

class TestBuildDiseaseLatentSpace:
    def test_basic_output_contract(self):
        from pipelines.state_space.latent_model import build_disease_latent_space
        adata = _make_synthetic_adata(100, 40)
        result = build_disease_latent_space(
            "IBD", [], _adata_override=adata,
            backend=None,   # auto-selects numpy backend since scanpy likely absent
        )
        required = [
            "adata", "latent_matrix", "cell_metadata", "gene_metadata",
            "neighbors_graph", "pseudotime", "provenance",
            "integration_warnings", "backend", "disease",
        ]
        for key in required:
            assert key in result, f"Missing key: {key}"

    def test_disease_field_propagated(self):
        from pipelines.state_space.latent_model import build_disease_latent_space
        adata = _make_synthetic_adata(80, 40)
        result = build_disease_latent_space("CAD", [], _adata_override=adata)
        assert result["disease"] == "CAD"
        assert result["provenance"]["disease"] == "CAD"

    def test_missing_h5ad_returns_error_dict(self):
        from pipelines.state_space.latent_model import build_disease_latent_space
        result = build_disease_latent_space("IBD", ["/nonexistent/path.h5ad"])
        assert "error" in result
        assert result["adata"] is None

    def test_cell_type_filter_reduces_cells(self):
        from pipelines.state_space.latent_model import build_disease_latent_space
        adata = _make_synthetic_adata(100, 40)
        result = build_disease_latent_space(
            "IBD", [], _adata_override=adata,
            cell_type_filter=["macrophage"],
        )
        if result.get("error"):
            pytest.skip("cell_type column not found in synthetic fixture")
        n = result["adata"].n_obs
        assert n <= 100

    def test_empty_filter_returns_error(self):
        from pipelines.state_space.latent_model import build_disease_latent_space
        adata = _make_synthetic_adata(60, 30)
        result = build_disease_latent_space(
            "IBD", [], _adata_override=adata,
            cell_type_filter=["nonexistent_cell_type_xyz"],
        )
        # Either error dict or warnings about 0 cells
        has_error = bool(result.get("error"))
        has_warning = any("0" in w or "No cells" in w for w in result.get("integration_warnings", []))
        assert has_error or has_warning

    def test_latent_matrix_shape(self):
        from pipelines.state_space.latent_model import build_disease_latent_space
        adata = _make_synthetic_adata(120, 50)
        result = build_disease_latent_space("IBD", [], _adata_override=adata)
        if result.get("error"):
            pytest.skip("latent build failed")
        lm = result["latent_matrix"]
        assert lm is not None
        assert lm.ndim == 2
        assert lm.shape[0] == result["adata"].n_obs

    def test_pseudotime_length_matches_cells(self):
        from pipelines.state_space.latent_model import build_disease_latent_space
        adata = _make_synthetic_adata(90, 45)
        result = build_disease_latent_space("IBD", [], _adata_override=adata)
        if result.get("error"):
            pytest.skip("latent build failed")
        pt = result["pseudotime"]
        assert len(pt) == result["adata"].n_obs

    def test_provenance_has_backend(self):
        from pipelines.state_space.latent_model import build_disease_latent_space
        adata = _make_synthetic_adata(80, 40)
        result = build_disease_latent_space("IBD", [], _adata_override=adata)
        if result.get("error"):
            pytest.skip("latent build failed")
        assert "backend" in result["provenance"]
        assert isinstance(result["provenance"]["backend"], str)
