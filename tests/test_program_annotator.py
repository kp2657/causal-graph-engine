"""
tests/test_program_annotator.py — unit tests for pipelines/program_annotator.py

Covers:
  - _hypergeometric_p: boundary cases
  - _bh_correct: monotonicity + length invariant
  - _run_ora: overlap threshold, FDR gating, returns sorted by p
  - _cell_state_label: Resting/Activated/Mixed branches
  - _genetic_direction_score: sign convention, missing data
  - _wes_direction_stats: concordance + depleted-program branch
  - annotate_programs: end-to-end with mocked filesystem + dependencies
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipelines.program_annotator import (
    _bh_correct,
    _cell_state_label,
    _genetic_direction_score,
    _hypergeometric_p,
    _make_mechanistic_narrative,
    _run_ora,
    _wes_direction_stats,
    annotate_programs,
    load_program_annotations,
)


# ---------------------------------------------------------------------------
# _hypergeometric_p
# ---------------------------------------------------------------------------

class TestHypergeometricP:
    def test_perfect_overlap_small(self):
        # k=n=N=2, M=10 → p should be small but >0
        p = _hypergeometric_p(k=2, M=10, n=2, N=2)
        assert 0 < p < 0.5

    def test_no_overlap(self):
        # k=0 is below min_overlap; but p=1.0 when k=0
        p = _hypergeometric_p(k=0, M=100, n=10, N=10)
        assert p == 1.0

    def test_full_overlap_large(self):
        # k=10=n=N, M=10 → deterministic full overlap, p=1.0
        p = _hypergeometric_p(k=10, M=10, n=10, N=10)
        assert math.isclose(p, 1.0)

    def test_returns_value_leq_1(self):
        for M, n, N, k in [(100, 20, 15, 3), (500, 50, 50, 10), (200, 10, 30, 4)]:
            p = _hypergeometric_p(k, M, n, N)
            assert 0.0 <= p <= 1.0, f"p={p} out of range for M={M},n={n},N={N},k={k}"

    def test_p_decreases_as_k_increases(self):
        # Higher overlap → lower (more significant) p-value
        p3 = _hypergeometric_p(k=3, M=200, n=30, N=30)
        p6 = _hypergeometric_p(k=6, M=200, n=30, N=30)
        assert p6 < p3


# ---------------------------------------------------------------------------
# _bh_correct
# ---------------------------------------------------------------------------

class TestBHCorrect:
    def test_empty(self):
        assert _bh_correct([]) == []

    def test_single(self):
        result = _bh_correct([0.04])
        assert len(result) == 1
        assert result[0] == pytest.approx(0.04, abs=1e-9)

    def test_monotonic_non_decreasing(self):
        pvals = [0.001, 0.01, 0.05, 0.1, 0.5]
        adj = _bh_correct(pvals)
        for i in range(len(adj) - 1):
            assert adj[i] <= adj[i + 1] + 1e-9, f"Not monotone at {i}: {adj}"

    def test_all_equal(self):
        adj = _bh_correct([0.05, 0.05, 0.05])
        # All equal → BH unchanged = 0.05
        for v in adj:
            assert v == pytest.approx(0.05, abs=1e-9)

    def test_capped_at_one(self):
        adj = _bh_correct([0.9, 0.95, 0.99])
        for v in adj:
            assert v <= 1.0

    def test_preserves_length(self):
        pvals = [0.001, 0.05, 0.1, 0.2, 0.4]
        assert len(_bh_correct(pvals)) == len(pvals)


# ---------------------------------------------------------------------------
# _run_ora
# ---------------------------------------------------------------------------

class TestRunORA:
    def _make_gene_sets(self):
        return {
            "HALLMARK_INFLAMMATORY_RESPONSE": frozenset(
                [f"GENE{i}" for i in range(40)]
            ),
            "HALLMARK_APOPTOSIS": frozenset(
                [f"GENE{i}" for i in range(20, 60)]
            ),
            "HALLMARK_TINY": frozenset(["GENEА", "GENEB"]),  # only 2 genes
        }

    def test_below_min_overlap_excluded(self):
        gene_sets = self._make_gene_sets()
        query = {"GENEА", "GENEB"}  # overlap=2 with HALLMARK_TINY but <3
        results = _run_ora(query, gene_sets, background_size=500, min_overlap=3)
        terms = {r["term"] for r in results}
        assert "HALLMARK_TINY" not in terms

    def test_hits_sorted_by_p(self):
        gene_sets = self._make_gene_sets()
        # 10 genes from INFLAMMATORY (overlap=10), 5 from APOPTOSIS (overlap=5)
        query = {f"GENE{i}" for i in range(10)} | {f"GENE{i}" for i in range(20, 25)}
        results = _run_ora(query, gene_sets, background_size=500, min_overlap=3)
        if len(results) >= 2:
            assert results[0]["p"] <= results[1]["p"]

    def test_empty_query_returns_empty(self):
        gene_sets = self._make_gene_sets()
        results = _run_ora(set(), gene_sets, background_size=500)
        assert results == []

    def test_result_has_required_keys(self):
        gene_sets = self._make_gene_sets()
        query = {f"GENE{i}" for i in range(10)}
        results = _run_ora(query, gene_sets, background_size=500, min_overlap=3)
        for r in results:
            for key in ("term", "overlap", "p", "fdr", "genes"):
                assert key in r, f"Missing key {key!r} in result"

    def test_fdr_filter(self):
        # Use tiny background so p-values are large → FDR will be high → all excluded
        gene_sets = {"TERM_A": frozenset([f"GENE{i}" for i in range(5)])}
        query = {f"GENE{i}" for i in range(3)}
        results = _run_ora(query, gene_sets, background_size=5, min_overlap=3, max_fdr=0.001)
        # If FDR > 0.001, should be filtered
        for r in results:
            assert r["fdr"] <= 0.001


# ---------------------------------------------------------------------------
# _cell_state_label
# ---------------------------------------------------------------------------

class TestCellStateLabel:
    def _threshold(self):
        from config.scoring_thresholds import TIMEPOINT_ACTIVATION_BIAS_MIN
        return TIMEPOINT_ACTIVATION_BIAS_MIN

    def test_no_biases_returns_mixed(self):
        assert _cell_state_label("CAD_SVD_C01", None) == "Mixed"
        assert _cell_state_label("CAD_SVD_C01", {}) == "Mixed"

    def test_missing_program_returns_mixed(self):
        assert _cell_state_label("CAD_SVD_C01", {"CAD_SVD_C02": 2.0}) == "Mixed"

    def test_high_bias_returns_activated(self):
        thr = self._threshold()
        biases = {"CAD_SVD_C01": thr + 0.1}
        assert _cell_state_label("CAD_SVD_C01", biases) == "Activated"

    def test_low_bias_returns_resting(self):
        thr = self._threshold()
        biases = {"CAD_SVD_C01": 1.0 / (thr + 0.1)}
        assert _cell_state_label("CAD_SVD_C01", biases) == "Resting"

    def test_mid_bias_returns_mixed(self):
        biases = {"CAD_SVD_C01": 1.0}  # ratio = 1.0 → neither extreme
        assert _cell_state_label("CAD_SVD_C01", biases) == "Mixed"


# ---------------------------------------------------------------------------
# _genetic_direction_score
# ---------------------------------------------------------------------------

class TestGeneticDirectionScore:
    def test_all_concordant_positive(self):
        # eQTL_β > 0 AND program_β > 0 → risk allele activates program → score = +1
        top_genes  = ["A", "B", "C"]
        top_betas  = {"A": 0.5, "B": 0.3, "C": 0.4}
        eqtl_betas = {"A": 0.2, "B": 0.1, "C": 0.3}
        gwas_prox  = {"A", "B", "C"}
        score, n = _genetic_direction_score(top_genes, top_betas, eqtl_betas, gwas_prox)
        assert math.isclose(score, 1.0)
        assert n == 3

    def test_all_discordant(self):
        # eQTL_β > 0, program_β < 0 → score = -1
        top_genes  = ["A", "B"]
        top_betas  = {"A": -0.5, "B": -0.3}
        eqtl_betas = {"A": 0.2, "B": 0.1}
        gwas_prox  = {"A", "B"}
        score, n = _genetic_direction_score(top_genes, top_betas, eqtl_betas, gwas_prox)
        assert math.isclose(score, -1.0)
        assert n == 2

    def test_mixed_signs_zero(self):
        top_genes  = ["A", "B"]
        top_betas  = {"A": 0.5, "B": -0.3}
        eqtl_betas = {"A": 0.2, "B": 0.1}
        gwas_prox  = {"A", "B"}
        score, n = _genetic_direction_score(top_genes, top_betas, eqtl_betas, gwas_prox)
        assert math.isclose(score, 0.0)

    def test_no_gwas_proximal_returns_zero(self):
        score, n = _genetic_direction_score(
            ["A", "B"], {"A": 0.5, "B": -0.3}, {"A": 0.2}, set()
        )
        assert score == 0.0
        assert n == 0

    def test_missing_eqtl_skipped(self):
        # Only "B" has eQTL → only it contributes
        top_genes  = ["A", "B"]
        top_betas  = {"A": 0.5, "B": 0.3}
        eqtl_betas = {"B": 0.1}  # A missing
        gwas_prox  = {"A", "B"}
        score, n = _genetic_direction_score(top_genes, top_betas, eqtl_betas, gwas_prox)
        assert n == 1


# ---------------------------------------------------------------------------
# _wes_direction_stats
# ---------------------------------------------------------------------------

class TestWESDirectionStats:
    def _make_burden(self, genes_betas: dict) -> dict:
        return {
            g: {"burden_beta": b, "burden_se": 0.1, "burden_p": 0.01}
            for g, b in genes_betas.items()
        }

    def test_atherogenic_concordant(self):
        # dir=+1 (atherogenic). burden_beta<0 → LoF reduces disease → gene is atherogenic → concordant
        burden = self._make_burden({"GENEA": -0.2, "GENEB": -0.3})
        score, n_hits = _wes_direction_stats(["GENEA", "GENEB"], 1, burden)
        assert math.isclose(score, 1.0)
        assert n_hits == 2

    def test_protective_concordant(self):
        # dir=-1 (protective). burden_beta>0 → LoF increases disease → gene is protective → concordant
        burden = self._make_burden({"GENEA": 0.2, "GENEB": 0.4})
        score, n_hits = _wes_direction_stats(["GENEA", "GENEB"], -1, burden)
        assert math.isclose(score, 1.0)

    def test_discordant(self):
        # dir=+1 but burden_beta>0 (LoF increases disease → gene is protective → wrong direction)
        burden = self._make_burden({"GENEA": 0.2, "GENEB": 0.3})
        score, n_hits = _wes_direction_stats(["GENEA", "GENEB"], 1, burden)
        assert math.isclose(score, 0.0)

    def test_depleted_program_returns_none(self):
        burden = self._make_burden({"GENEA": -0.2})
        score, n_hits = _wes_direction_stats(["GENEA"], 0, burden)
        assert score is None

    def test_no_wes_data_returns_none(self):
        score, n_hits = _wes_direction_stats(["GENEA", "GENEB"], 1, {})
        assert score is None
        assert n_hits == 0

    def test_n_wes_hits_counts_nominal(self):
        burden = {
            "GENEA": {"burden_beta": -0.5, "burden_se": 0.1, "burden_p": 0.001},  # sig
            "GENEB": {"burden_beta": -0.1, "burden_se": 0.1, "burden_p": 0.8},    # not sig
        }
        score, n_hits = _wes_direction_stats(["GENEA", "GENEB"], 1, burden)
        assert n_hits == 1


# ---------------------------------------------------------------------------
# _make_mechanistic_narrative
# ---------------------------------------------------------------------------

_DEFAULT_FUNC_TERMS = [{"term": "HALLMARK_INFLAMMATORY_RESPONSE", "fdr": 0.05, "p": 0.001}]

def _make_ann(
    direction="atherogenic",
    direction_int=1,
    cell_state="Activated",
    gamma=0.18,
    func_terms=_DEFAULT_FUNC_TERMS,
    gen_dir=0.6,
    n_eqtl=5,
    wes_score=0.7,
    n_wes=3,
    gwas_anchors=None,
    top_genes=None,
):
    return {
        "direction":               direction,
        "direction_int":           direction_int,
        "cell_state":              cell_state,
        "gamma":                   gamma,
        "top_genes":               top_genes or ["CCM2", "KRIT1", "PLPP3"],
        "functional_terms":        func_terms,
        "genetic_direction_score": gen_dir,
        "n_eqtl_genes":            n_eqtl,
        "wes_direction_score":     wes_score,
        "n_wes_hits":              n_wes,
        "gwas_anchor_overlap":     gwas_anchors or ["CCM2"],
    }


class TestMechanisticNarrative:
    def test_returns_nonempty_string(self):
        ann = _make_ann()
        result = _make_mechanistic_narrative(ann)
        assert isinstance(result, str)
        assert len(result) > 50

    def test_depleted_program_short(self):
        ann = _make_ann(direction="depleted", direction_int=0, gamma=0.005)
        result = _make_mechanistic_narrative(ann)
        assert "depleted" in result.lower()
        assert "no directional" in result.lower()

    def test_atherogenic_activate_language(self):
        ann = _make_ann(direction="atherogenic", direction_int=1, gen_dir=0.7, n_eqtl=6)
        result = _make_mechanistic_narrative(ann)
        assert "activate" in result

    def test_protective_suppress_language(self):
        ann = _make_ann(direction="protective", direction_int=-1, gen_dir=-0.6, n_eqtl=4)
        result = _make_mechanistic_narrative(ann)
        assert "suppress" in result

    def test_wes_concordant_language(self):
        ann = _make_ann(wes_score=0.75)
        result = _make_mechanistic_narrative(ann)
        assert "concordant" in result

    def test_wes_discordant_language(self):
        ann = _make_ann(wes_score=0.30)
        result = _make_mechanistic_narrative(ann)
        assert "discordant" in result

    def test_wes_none_language(self):
        ann = _make_ann(wes_score=None, n_wes=0)
        result = _make_mechanistic_narrative(ann)
        assert "unavailable" in result

    def test_no_func_terms(self):
        ann = _make_ann(func_terms=[])
        result = _make_mechanistic_narrative(ann)
        assert "no significant" in result.lower()

    def test_no_eqtl_data(self):
        ann = _make_ann(gen_dir=0.0, n_eqtl=0)
        result = _make_mechanistic_narrative(ann)
        assert "no cis-eqtl" in result.lower()

    def test_gamma_appears_in_narrative(self):
        ann = _make_ann(gamma=0.215)
        result = _make_mechanistic_narrative(ann)
        assert "0.215" in result or "+0.215" in result

    def test_gwas_anchor_genes_appear(self):
        ann = _make_ann(gwas_anchors=["CCM2", "KRIT1"])
        result = _make_mechanistic_narrative(ann)
        assert "CCM2" in result

    def test_end_to_end_has_narrative_field(self, tmp_path):
        """annotate_programs output includes mechanistic_narrative for every program."""
        import json as _json
        import numpy as _np
        from unittest.mock import patch as _patch

        ds_dir = tmp_path / "fake_dataset"
        ds_dir.mkdir()
        rng = _np.random.default_rng(0)
        Vt = rng.standard_normal((2, 15)).astype(_np.float32)
        pert_names = _np.array([f"GENE{i:03d}" for i in range(15)])
        _np.savez(ds_dir / "svd_loadings.npz", Vt=Vt, pert_names=pert_names)
        msigdb_path = tmp_path / "msigdb_hallmark_2024.json"
        msigdb_path.write_text(_json.dumps({"HALLMARK_T": [f"GENE{i:03d}" for i in range(8)]}))
        mock_gammas = {"CAD_SVD_C01": {"gamma": 0.15}, "CAD_SVD_C02": {"gamma": -0.12}}

        with (
            _patch("pipelines.program_annotator._ROOT", tmp_path),
            _patch("pipelines.program_annotator._PERTURBSEQ", tmp_path),
            _patch("pipelines.program_annotator._MSIGDB_PATH", msigdb_path),
            _patch("graph.schema.DISEASE_CELL_TYPE_MAP", {"CAD": {"scperturb_dataset": "fake_dataset"}}),
            _patch("graph.schema._DISEASE_SHORT_NAMES_FOR_ANCHORS", {"cad": "CAD"}),
            _patch("pipelines.ldsc.gamma_loader.get_genetic_nmf_program_gammas", return_value=mock_gammas),
            _patch("config.scoring_thresholds.SLDSC_GAMMA_FLOOR", 0.05),
        ):
            result = annotate_programs("cad")

        for prog, ann in result.items():
            assert "mechanistic_narrative" in ann, f"{prog} missing mechanistic_narrative"
            assert isinstance(ann["mechanistic_narrative"], str)
            assert len(ann["mechanistic_narrative"]) > 30


# ---------------------------------------------------------------------------
# annotate_programs — end-to-end with mocked dependencies
# ---------------------------------------------------------------------------

class TestAnnotatePrograms:
    """
    Uses a temporary directory to avoid touching real data.
    Mocks: DISEASE_CELL_TYPE_MAP, _DISEASE_SHORT_NAMES_FOR_ANCHORS,
           get_svd_program_gammas, SLDSC_GAMMA_FLOOR, MSigDB JSON.
    """

    N_PROGS = 3
    N_PERTS = 20

    def _setup_tmpdir(self, tmp_path: Path) -> Path:
        ds_dir = tmp_path / "fake_dataset"
        ds_dir.mkdir()

        # Synthetic Vt: (3 × 20)
        rng = np.random.default_rng(42)
        Vt = rng.standard_normal((self.N_PROGS, self.N_PERTS)).astype(np.float32)
        pert_names = np.array([f"GENE{i:03d}" for i in range(self.N_PERTS)])
        np.savez(ds_dir / "svd_loadings.npz", Vt=Vt, pert_names=pert_names)

        # MSigDB with one term overlapping GENE000–GENE009
        msigdb = {
            "HALLMARK_TEST_TERM": [f"GENE{i:03d}" for i in range(10)],
            "HALLMARK_OTHER": [f"GENE{i:03d}" for i in range(10, 25)],
        }
        msigdb_path = tmp_path / "msigdb_hallmark_2024.json"
        msigdb_path.write_text(json.dumps(msigdb))

        return ds_dir, msigdb_path

    def test_annotate_programs_writes_json(self, tmp_path):
        ds_dir, msigdb_path = self._setup_tmpdir(tmp_path)

        mock_gammas = {
            f"CAD_SVD_C{c+1:02d}": {"gamma": 0.1 * (c - 1)}
            for c in range(self.N_PROGS)
        }

        with (
            patch("pipelines.program_annotator._ROOT", tmp_path),
            patch("pipelines.program_annotator._PERTURBSEQ", tmp_path),
            patch("pipelines.program_annotator._MSIGDB_PATH", msigdb_path),
            patch(
                "graph.schema.DISEASE_CELL_TYPE_MAP",
                {"CAD": {"scperturb_dataset": "fake_dataset"}},
            ),
            patch(
                "graph.schema._DISEASE_SHORT_NAMES_FOR_ANCHORS",
                {"cad": "CAD"},
            ),
            patch(
                "pipelines.ldsc.gamma_loader.get_genetic_nmf_program_gammas",
                return_value=mock_gammas,
            ),
            patch("config.scoring_thresholds.SLDSC_GAMMA_FLOOR", 0.05),
        ):
            result = annotate_programs("cad")

        assert len(result) == self.N_PROGS
        out_path = ds_dir / "program_annotations.json"
        assert out_path.exists()
        on_disk = json.loads(out_path.read_text())
        assert set(on_disk.keys()) == set(result.keys())

    def test_annotation_fields_present(self, tmp_path):
        ds_dir, msigdb_path = self._setup_tmpdir(tmp_path)
        mock_gammas = {
            f"CAD_SVD_C{c+1:02d}": {"gamma": 0.15}
            for c in range(self.N_PROGS)
        }

        with (
            patch("pipelines.program_annotator._ROOT", tmp_path),
            patch("pipelines.program_annotator._PERTURBSEQ", tmp_path),
            patch("pipelines.program_annotator._MSIGDB_PATH", msigdb_path),
            patch(
                "graph.schema.DISEASE_CELL_TYPE_MAP",
                {"CAD": {"scperturb_dataset": "fake_dataset"}},
            ),
            patch(
                "graph.schema._DISEASE_SHORT_NAMES_FOR_ANCHORS",
                {"cad": "CAD"},
            ),
            patch(
                "pipelines.ldsc.gamma_loader.get_genetic_nmf_program_gammas",
                return_value=mock_gammas,
            ),
            patch("config.scoring_thresholds.SLDSC_GAMMA_FLOOR", 0.05),
        ):
            result = annotate_programs("cad")

        required = {
            "program", "gamma", "direction", "direction_int",
            "gwas_t", "cell_state", "top_genes", "top_betas",
            "functional_terms", "n_background",
            # Phase 2
            "gwas_anchor_enrichment_p", "gwas_anchor_overlap",
            "genetic_direction_score", "n_eqtl_genes",
            "wes_direction_score", "n_wes_hits",
        }
        for prog, ann in result.items():
            missing = required - set(ann.keys())
            assert not missing, f"{prog} missing fields: {missing}"

    def test_direction_values(self, tmp_path):
        ds_dir, msigdb_path = self._setup_tmpdir(tmp_path)
        # γ = |τ*| (unsigned): all above-floor programs are disease_relevant
        mock_gammas = {
            "CAD_SVD_C01": {"gamma": 0.20},   # disease_relevant
            "CAD_SVD_C02": {"gamma": 0.15},   # disease_relevant (no longer "protective")
            "CAD_SVD_C03": {"gamma": 0.001},  # depleted (below floor)
        }

        with (
            patch("pipelines.program_annotator._ROOT", tmp_path),
            patch("pipelines.program_annotator._PERTURBSEQ", tmp_path),
            patch("pipelines.program_annotator._MSIGDB_PATH", msigdb_path),
            patch(
                "graph.schema.DISEASE_CELL_TYPE_MAP",
                {"CAD": {"scperturb_dataset": "fake_dataset"}},
            ),
            patch(
                "graph.schema._DISEASE_SHORT_NAMES_FOR_ANCHORS",
                {"cad": "CAD"},
            ),
            patch(
                "pipelines.ldsc.gamma_loader.get_genetic_nmf_program_gammas",
                return_value=mock_gammas,
            ),
            patch("config.scoring_thresholds.SLDSC_GAMMA_FLOOR", 0.05),
        ):
            result = annotate_programs("cad")

        assert result["CAD_SVD_C01"]["direction"] == "disease_relevant"
        assert result["CAD_SVD_C01"]["direction_int"] == 1
        assert result["CAD_SVD_C02"]["direction"] == "disease_relevant"
        assert result["CAD_SVD_C02"]["direction_int"] == 1
        assert result["CAD_SVD_C03"]["direction"] == "depleted"
        assert result["CAD_SVD_C03"]["direction_int"] == 0

    def test_load_cached(self, tmp_path):
        ds_dir, msigdb_path = self._setup_tmpdir(tmp_path)
        # Pre-write a cached result
        cached = {"CAD_SVD_C01": {"program": "CAD_SVD_C01", "gamma": 0.1}}
        (ds_dir / "program_annotations.json").write_text(json.dumps(cached))

        with (
            patch("pipelines.program_annotator._PERTURBSEQ", tmp_path),
            patch(
                "graph.schema.DISEASE_CELL_TYPE_MAP",
                {"CAD": {"scperturb_dataset": "fake_dataset"}},
            ),
            patch(
                "graph.schema._DISEASE_SHORT_NAMES_FOR_ANCHORS",
                {"cad": "CAD"},
            ),
        ):
            result = load_program_annotations("cad")

        assert result == cached
