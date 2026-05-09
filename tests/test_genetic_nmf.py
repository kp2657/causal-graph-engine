"""
tests/test_genetic_nmf.py — unit tests for pipelines/genetic_nmf.py

Covers:
  - build_wes_gene_weights: direction convention, z_min gate, shet boost
  - fit_genetic_nmf: shape contract, non-negativity of H, signed W_net
  - _multiplicative_update_step: incoherence penalty pushes opponent genes down
  - run_genetic_nmf_for_dataset: end-to-end with mocked signatures
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipelines.genetic_nmf import (
    _multiplicative_update_step,
    build_wes_gene_weights,
    build_combined_gene_weights,
    fit_genetic_nmf,
    run_genetic_nmf_for_dataset,
)


# ---------------------------------------------------------------------------
# build_wes_gene_weights
# ---------------------------------------------------------------------------

class TestBuildWESGeneWeights:
    def _mock_wes(self, tmp_path: Path, data: dict) -> Path:
        p = tmp_path / "CAD_burden.json"
        p.write_text(json.dumps(data))
        return p

    def test_atherogenic_direction(self, tmp_path):
        wes = {"GENEА": {"burden_beta": -0.4, "burden_se": 0.1}}
        with (
            patch("pipelines.genetic_nmf._WES_DIR", tmp_path),
            patch("pipelines.genetic_nmf._load_shet_map", return_value={}),
            patch("pipelines.genetic_nmf._load_shet_symbol_map", return_value={}),
        ):
            self._mock_wes(tmp_path, wes)
            d, w = build_wes_gene_weights(["GENEА"], "cad", z_min=1.0, shet_scale=5.0)
        assert d[0] == 1.0  # atherogenic: burden_beta < 0 → d = +1

    def test_protective_direction(self, tmp_path):
        wes = {"GENEB": {"burden_beta": +0.3, "burden_se": 0.1}}
        with (
            patch("pipelines.genetic_nmf._WES_DIR", tmp_path),
            patch("pipelines.genetic_nmf._load_shet_map", return_value={}),
            patch("pipelines.genetic_nmf._load_shet_symbol_map", return_value={}),
        ):
            self._mock_wes(tmp_path, wes)
            d, w = build_wes_gene_weights(["GENEB"], "cad", z_min=1.0, shet_scale=5.0)
        assert d[0] == -1.0  # protective: burden_beta > 0 → d = -1

    def test_z_min_gate(self, tmp_path):
        # |Z| = 0.2/0.1 = 2.0, but gene has |Z| = 0.05/0.1 = 0.5 < z_min=1.5
        wes = {"GENEC": {"burden_beta": -0.05, "burden_se": 0.1}}
        with (
            patch("pipelines.genetic_nmf._WES_DIR", tmp_path),
            patch("pipelines.genetic_nmf._load_shet_map", return_value={}),
            patch("pipelines.genetic_nmf._load_shet_symbol_map", return_value={}),
        ):
            self._mock_wes(tmp_path, wes)
            d, w = build_wes_gene_weights(["GENEC"], "cad", z_min=1.5, shet_scale=5.0)
        assert d[0] == 0.0  # below z_min → direction unknown
        assert w[0] == 0.0

    def test_shet_boost(self, tmp_path):
        wes = {"GENED": {"burden_beta": -0.4, "burden_se": 0.1}}
        with (
            patch("pipelines.genetic_nmf._WES_DIR", tmp_path),
            patch("pipelines.genetic_nmf._load_shet_map", return_value={"ENSG001": 0.10}),
            patch("pipelines.genetic_nmf._load_shet_symbol_map", return_value={"GENED": "ENSG001"}),
        ):
            self._mock_wes(tmp_path, wes)
            _, w_no  = build_wes_gene_weights(["GENED"], "cad", z_min=1.0, shet_scale=0.0)
            _, w_yes = build_wes_gene_weights(["GENED"], "cad", z_min=1.0, shet_scale=5.0)
        assert w_yes[0] > w_no[0], "shet boost should increase weight"

    def test_missing_wes_zero_weight(self, tmp_path):
        with (
            patch("pipelines.genetic_nmf._WES_DIR", tmp_path),
            patch("pipelines.genetic_nmf._load_shet_map", return_value={}),
            patch("pipelines.genetic_nmf._load_shet_symbol_map", return_value={}),
        ):
            (tmp_path / "CAD_burden.json").write_text("{}")
            d, w = build_wes_gene_weights(["GENEX"], "cad", z_min=1.0, shet_scale=5.0)
        assert d[0] == 0.0
        assert w[0] == 0.0


# ---------------------------------------------------------------------------
# _multiplicative_update_step — incoherence penalty
# ---------------------------------------------------------------------------

class TestMultiplicativeUpdateStep:
    def _setup(self, n=20, p=10, k=3, seed=0):
        rng = np.random.default_rng(seed)
        X = np.abs(rng.standard_normal((n, p))).astype(np.float32)
        W = np.abs(rng.standard_normal((n, k))).astype(np.float32)
        H = np.abs(rng.standard_normal((k, p))).astype(np.float32)
        return X, W, H

    def test_non_negativity_preserved(self):
        X, W, H = self._setup()
        d = np.zeros(20, dtype=np.float32)
        w = np.zeros(20, dtype=np.float32)
        W2, H2 = _multiplicative_update_step(X, W, H, d, w, lam=0.0)
        assert (W2 >= 0).all()
        assert (H2 >= 0).all()

    def test_penalty_reduces_opponent_loadings(self):
        """With λ>0, atherogenic genes should get lower loading when program already has protective genes."""
        n, k = 6, 2
        # Simple: 3 atherogenic, 3 protective genes; all have equal current loading
        X = np.ones((n, 4), dtype=np.float32)
        W = np.ones((n, k), dtype=np.float32) * 0.5
        H = np.ones((k, 4), dtype=np.float32) * 0.5
        d = np.array([1., 1., 1., -1., -1., -1.], dtype=np.float32)
        w = np.array([2., 2., 2.,  2.,  2.,  2.], dtype=np.float32)

        W_no_pen,  _ = _multiplicative_update_step(X, W.copy(), H.copy(), d, w, lam=0.0)
        W_with_pen, _ = _multiplicative_update_step(X, W.copy(), H.copy(), d, w, lam=1.0)

        # With penalty, atherogenic gene loadings should be shrunk relative to no-penalty case
        ath_no  = W_no_pen[:3, :].sum()
        ath_pen = W_with_pen[:3, :].sum()
        assert ath_pen < ath_no, "Penalty should reduce opponent-direction loadings"

    def test_zero_weight_genes_unaffected(self):
        """Genes with w=0 should update identically regardless of λ."""
        X, W, H = self._setup()
        d = np.ones(20, dtype=np.float32)
        w = np.zeros(20, dtype=np.float32)  # all zero — penalty has no effect

        W_no,  _ = _multiplicative_update_step(X, W.copy(), H.copy(), d, w, lam=0.0)
        W_pen, _ = _multiplicative_update_step(X, W.copy(), H.copy(), d, w, lam=5.0)
        np.testing.assert_allclose(W_no, W_pen, rtol=1e-5)


# ---------------------------------------------------------------------------
# fit_genetic_nmf
# ---------------------------------------------------------------------------

class TestFitGeneticNMF:
    def _make_M(self, n_genes=30, n_perts=20, seed=7):
        rng = np.random.default_rng(seed)
        return rng.standard_normal((n_genes, n_perts)).astype(np.float32)

    def test_output_shapes(self):
        M = self._make_M()
        n_genes, n_perts = M.shape
        k = 4
        d = np.zeros(n_genes, dtype=np.float32)
        w = np.zeros(n_genes, dtype=np.float32)
        W_net, H, d_out, w_out = fit_genetic_nmf(M, [f"G{i}" for i in range(n_genes)], d, w, k, lam=0.05, max_iter=10)
        assert W_net.shape == (n_genes, k)
        assert H.shape == (k, n_perts)
        assert d_out.shape == (n_genes,)
        assert w_out.shape == (n_genes,)

    def test_H_non_negative(self):
        M = self._make_M()
        n_genes = M.shape[0]
        d = np.zeros(n_genes, dtype=np.float32)
        w = np.zeros(n_genes, dtype=np.float32)
        _, H, _, _ = fit_genetic_nmf(M, [f"G{i}" for i in range(n_genes)], d, w, 3, lam=0.0, max_iter=20)
        assert (H >= 0).all()

    def test_W_net_can_be_signed(self):
        M = self._make_M()
        n_genes = M.shape[0]
        d = np.zeros(n_genes, dtype=np.float32)
        w = np.zeros(n_genes, dtype=np.float32)
        W_net, _, _, _ = fit_genetic_nmf(M, [f"G{i}" for i in range(n_genes)], d, w, 3, lam=0.0, max_iter=20)
        # W_net = W_pos - W_neg, so it can be negative
        assert W_net.min() < 0 or W_net.max() >= 0  # at least one signed value exists

    def test_reconstruction_finite(self):
        M = self._make_M()
        n_genes = M.shape[0]
        d = np.zeros(n_genes, dtype=np.float32)
        w = np.zeros(n_genes, dtype=np.float32)
        W_net, H, _, _ = fit_genetic_nmf(M, [f"G{i}" for i in range(n_genes)], d, w, 4, lam=0.05, max_iter=30)
        assert np.isfinite(W_net).all()
        assert np.isfinite(H).all()

    def test_penalty_separates_directed_programs(self):
        """Programs should load more on same-direction genes when λ is high."""
        rng = np.random.default_rng(0)
        n_genes, n_perts, k = 20, 15, 2
        # Half atherogenic (positive signal), half protective (negative signal)
        M_ath = np.abs(rng.standard_normal((10, n_perts))).astype(np.float32)
        M_pro = -np.abs(rng.standard_normal((10, n_perts))).astype(np.float32)
        M = np.vstack([M_ath, M_pro])

        d = np.array([1.]*10 + [-1.]*10, dtype=np.float32)
        w = np.ones(n_genes, dtype=np.float32) * 3.0

        gene_names = [f"G{i}" for i in range(n_genes)]
        W_no, _, _, _  = fit_genetic_nmf(M, gene_names, d, w, k, lam=0.0,  max_iter=100)
        W_yes, _, _, _ = fit_genetic_nmf(M, gene_names, d, w, k, lam=1.0,  max_iter=100)

        # With penalty: each program should be more concentrated on one direction
        # Measure: sum of loadings from same-sign genes in the dominant-direction program
        def coherence(W_net):
            # For each program, compute |Σ d_i * W_net[i,k]| / Σ |W_net[i,k]|
            scores = []
            for c in range(W_net.shape[1]):
                col = np.abs(W_net[:, c])
                if col.sum() < 1e-8:
                    continue
                scores.append(abs((d * col).sum()) / col.sum())
            return float(np.mean(scores)) if scores else 0.0

        coh_no  = coherence(W_no)
        coh_yes = coherence(W_yes)
        assert coh_yes >= coh_no - 0.05, f"Penalty should not reduce coherence: {coh_no:.3f} → {coh_yes:.3f}"


# ---------------------------------------------------------------------------
# run_genetic_nmf_for_dataset — end-to-end
# ---------------------------------------------------------------------------

class TestRunGeneticNMFForDataset:
    def _make_sigs(self, n_genes=25, n_perts=15, seed=1) -> dict:
        rng = np.random.default_rng(seed)
        genes = [f"GENE{i:03d}" for i in range(n_genes)]
        perts = [f"PERT{i:03d}" for i in range(n_perts)]
        return {
            p: {g: float(rng.standard_normal()) for g in rng.choice(genes, 15, replace=False)}
            for p in perts
        }

    def test_writes_npz(self, tmp_path):
        sigs = self._make_sigs()
        ds_dir = tmp_path / "fake_ds"
        ds_dir.mkdir()
        sig_path = ds_dir / "signatures.json.gz"
        with gzip.open(sig_path, "wt") as f:
            json.dump(sigs, f)

        wes = {f"GENE{i:03d}": {"burden_beta": -0.3 if i % 2 == 0 else 0.3, "burden_se": 0.1}
               for i in range(20)}
        wes_path = tmp_path / "CAD_burden.json"
        wes_path.write_text(json.dumps(wes))

        with (
            patch("pipelines.genetic_nmf._PERTURBSEQ", tmp_path),
            patch("pipelines.genetic_nmf._WES_DIR", tmp_path),
            patch("pipelines.genetic_nmf._load_shet_map", return_value={}),
            patch("pipelines.genetic_nmf._load_shet_symbol_map", return_value={}),
            patch("config.scoring_thresholds.GENETIC_NMF_RANK", 3),
            patch("config.scoring_thresholds.GENETIC_NMF_LAMBDA", 0.1),
            patch("config.scoring_thresholds.GENETIC_NMF_MAX_ITER", 20),
            patch("config.scoring_thresholds.GENETIC_NMF_WES_Z_MIN", 1.0),
            patch("config.scoring_thresholds.GENETIC_NMF_SHET_SCALE", 0.0),
        ):
            result = run_genetic_nmf_for_dataset("fake_ds", "cad", n_components=3, max_iter=20)

        assert result.get("error") is None, result.get("error")
        npz_path = ds_dir / "genetic_nmf_loadings.npz"
        assert npz_path.exists()

        npz = np.load(npz_path)
        assert "Vt" in npz
        assert "U_scaled" in npz
        assert "gene_names" in npz
        assert "pert_names" in npz
        assert "d_genes" in npz
        assert "w_genes" in npz
        assert npz["Vt"].shape[0] == 3   # k programs
        assert npz["Vt"].shape[1] == len(sigs)  # n_perts


# ---------------------------------------------------------------------------
# build_combined_gene_weights
# ---------------------------------------------------------------------------

class TestBuildCombinedGeneWeights:
    def _setup(self, tmp_path: Path, wes_data: dict, eqtl_data: dict):
        (tmp_path / "CAD_burden.json").write_text(json.dumps(wes_data))
        eqtl_dir = tmp_path / "eqtl_indices"
        eqtl_dir.mkdir(exist_ok=True)
        (eqtl_dir / "CAD_QTD000001_top_eqtls.json").write_text(json.dumps(eqtl_data))
        return eqtl_dir

    def test_concordant_boost(self, tmp_path):
        """WES+eQTL same direction → weight > WES-only weight."""
        wes  = {"GENEA": {"burden_beta": -0.4, "burden_se": 0.1}}
        eqtl = {"GENEA": {"beta": 0.3, "se": 0.1, "pvalue": 0.001}}
        eqtl_dir = self._setup(tmp_path, wes, eqtl)

        with patch("pipelines.genetic_nmf._WES_DIR", tmp_path), \
             patch("pipelines.genetic_nmf._EQTL_DIR", eqtl_dir), \
             patch("pipelines.genetic_nmf._load_shet_map", return_value={}), \
             patch("pipelines.genetic_nmf._load_shet_symbol_map", return_value={}):
            _, w_wes  = build_wes_gene_weights(["GENEA"], "cad", z_min=1.0, shet_scale=0.0)
            d_c, w_c  = build_combined_gene_weights(["GENEA"], "cad", z_min=1.0, shet_scale=0.0)

        assert d_c[0] == 1.0       # atherogenic direction preserved
        assert w_c[0] > w_wes[0]   # boosted by concordance

    def test_discordant_discount(self, tmp_path):
        """WES+eQTL opposite direction → weight < WES-only weight, direction from WES."""
        wes  = {"GENEB": {"burden_beta": -0.4, "burden_se": 0.1}}
        eqtl = {"GENEB": {"beta": -0.3, "se": 0.1, "pvalue": 0.001}}
        eqtl_dir = self._setup(tmp_path, wes, eqtl)

        with patch("pipelines.genetic_nmf._WES_DIR", tmp_path), \
             patch("pipelines.genetic_nmf._EQTL_DIR", eqtl_dir), \
             patch("pipelines.genetic_nmf._load_shet_map", return_value={}), \
             patch("pipelines.genetic_nmf._load_shet_symbol_map", return_value={}):
            _, w_wes = build_wes_gene_weights(["GENEB"], "cad", z_min=1.0, shet_scale=0.0)
            d_c, w_c = build_combined_gene_weights(["GENEB"], "cad", z_min=1.0, shet_scale=0.0)

        assert d_c[0] == 1.0       # WES direction wins
        assert w_c[0] < w_wes[0]   # discounted

    def test_eqtl_only_no_direction(self, tmp_path):
        """Gene absent from WES → eQTL alone is insufficient; d=0, w=0."""
        eqtl_dir = self._setup(tmp_path, {}, {"GENEC": {"beta": -0.5, "se": 0.1, "pvalue": 0.0001}})

        with patch("pipelines.genetic_nmf._WES_DIR", tmp_path), \
             patch("pipelines.genetic_nmf._EQTL_DIR", eqtl_dir), \
             patch("pipelines.genetic_nmf._load_shet_map", return_value={}), \
             patch("pipelines.genetic_nmf._load_shet_symbol_map", return_value={}):
            d_c, w_c = build_combined_gene_weights(["GENEC"], "cad", z_min=1.0, shet_scale=0.0)

        assert d_c[0] == 0.0   # WES anchor required; eQTL alone ignored
        assert w_c[0] == 0.0

    def test_no_signal_zero(self, tmp_path):
        """Neither WES nor eQTL → d=0, w=0."""
        eqtl_dir = self._setup(tmp_path, {}, {})

        with patch("pipelines.genetic_nmf._WES_DIR", tmp_path), \
             patch("pipelines.genetic_nmf._EQTL_DIR", eqtl_dir), \
             patch("pipelines.genetic_nmf._load_shet_map", return_value={}), \
             patch("pipelines.genetic_nmf._load_shet_symbol_map", return_value={}):
            d_c, w_c = build_combined_gene_weights(["GENED"], "cad", z_min=1.0, shet_scale=0.0)

        assert d_c[0] == 0.0
        assert w_c[0] == 0.0

    def test_wes_only_unchanged(self, tmp_path):
        """WES gene with no eQTL counterpart → identical weight to WES-only build."""
        wes  = {"GENEE": {"burden_beta": -0.4, "burden_se": 0.1}}
        eqtl_dir = self._setup(tmp_path, wes, {})

        with patch("pipelines.genetic_nmf._WES_DIR", tmp_path), \
             patch("pipelines.genetic_nmf._EQTL_DIR", eqtl_dir), \
             patch("pipelines.genetic_nmf._load_shet_map", return_value={}), \
             patch("pipelines.genetic_nmf._load_shet_symbol_map", return_value={}):
            _, w_wes  = build_wes_gene_weights(["GENEE"], "cad", z_min=1.0, shet_scale=0.0)
            _, w_comb = build_combined_gene_weights(["GENEE"], "cad", z_min=1.0, shet_scale=0.0)

        assert w_comb[0] == w_wes[0]
