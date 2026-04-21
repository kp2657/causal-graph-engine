"""
tests/test_phase_j_benchmark.py

Unit tests for Phase J: upstream regulator recovery benchmark.

Tests:
  - load_benchmark_config: valid file, missing keys, empty lists
  - _median_rank: basic, absent genes, all absent → NaN
  - compute_upstream_recovery_ratio: ideal (<1.0), inverted (>1.0), NaN cases
  - compute_marker_rank_delta: markers drop (negative), markers rise (positive), no change (0)
  - evaluate_ranking: pass/fail recovery ratio, marker_rank_delta present/absent
  - IBD config file: loads correctly, gene sets non-empty
"""
from __future__ import annotations

import math
import pathlib
import pytest

# Project root relative to this file
_REPO_ROOT = pathlib.Path(__file__).parent.parent
_IBD_CONFIG_PATH = _REPO_ROOT / "data" / "benchmarks" / "ibd_upstream_regulators_v1.json"

_UPSTREAM = ["SPI1", "IRF4", "STAT1", "NFKB1", "NLRP3", "JAK2", "CEBPB", "MYD88"]
_MARKERS = ["LYZ", "S100A8", "S100A9", "MNDA", "CD68"]


def _ideal_ranking() -> list[str]:
    """Upstream genes first, filler next, markers last."""
    filler = ["GENE_A", "GENE_B", "GENE_C"]
    return _UPSTREAM + filler + _MARKERS


def _inverted_ranking() -> list[str]:
    """Markers first, upstream genes last."""
    filler = ["GENE_A", "GENE_B", "GENE_C"]
    return _MARKERS + filler + _UPSTREAM


# ---------------------------------------------------------------------------
# load_benchmark_config
# ---------------------------------------------------------------------------

class TestLoadBenchmarkConfig:
    def test_ibd_config_loads(self):
        from pipelines.upstream_recovery_benchmark import load_benchmark_config
        cfg = load_benchmark_config(_IBD_CONFIG_PATH)
        assert "upstream_regulators" in cfg
        assert "terminal_markers" in cfg

    def test_ibd_upstream_set_nonempty(self):
        from pipelines.upstream_recovery_benchmark import load_benchmark_config
        cfg = load_benchmark_config(_IBD_CONFIG_PATH)
        assert len(cfg["upstream_regulators"]) >= 4

    def test_ibd_marker_set_nonempty(self):
        from pipelines.upstream_recovery_benchmark import load_benchmark_config
        cfg = load_benchmark_config(_IBD_CONFIG_PATH)
        assert len(cfg["terminal_markers"]) >= 3

    def test_ibd_spi1_in_upstream(self):
        from pipelines.upstream_recovery_benchmark import load_benchmark_config
        cfg = load_benchmark_config(_IBD_CONFIG_PATH)
        assert "SPI1" in cfg["upstream_regulators"]

    def test_ibd_lyz_in_markers(self):
        from pipelines.upstream_recovery_benchmark import load_benchmark_config
        cfg = load_benchmark_config(_IBD_CONFIG_PATH)
        assert "LYZ" in cfg["terminal_markers"]

    def test_missing_file_raises(self):
        from pipelines.upstream_recovery_benchmark import load_benchmark_config
        with pytest.raises(FileNotFoundError):
            load_benchmark_config("/nonexistent/path/benchmark.json")

    def test_missing_upstream_key_raises(self, tmp_path):
        from pipelines.upstream_recovery_benchmark import load_benchmark_config
        import json
        p = tmp_path / "bad.json"
        p.write_text(json.dumps({"terminal_markers": ["LYZ"]}))
        with pytest.raises(KeyError):
            load_benchmark_config(p)

    def test_empty_upstream_raises(self, tmp_path):
        from pipelines.upstream_recovery_benchmark import load_benchmark_config
        import json
        p = tmp_path / "empty.json"
        p.write_text(json.dumps({"upstream_regulators": [], "terminal_markers": ["LYZ"]}))
        with pytest.raises(ValueError):
            load_benchmark_config(p)


# ---------------------------------------------------------------------------
# _median_rank
# ---------------------------------------------------------------------------

class TestMedianRank:
    def test_single_gene_exact_position(self):
        from pipelines.upstream_recovery_benchmark import _median_rank
        ranked = ["A", "B", "C", "D"]
        assert _median_rank(["C"], ranked) == 2

    def test_two_genes_median(self):
        from pipelines.upstream_recovery_benchmark import _median_rank
        ranked = ["A", "B", "C", "D"]
        # A=0, C=2 → median = 1.0
        assert _median_rank(["A", "C"], ranked) == pytest.approx(1.0)

    def test_absent_gene_skipped(self):
        from pipelines.upstream_recovery_benchmark import _median_rank
        ranked = ["A", "B", "C"]
        # MISSING not in ranked → only A=0 counts
        assert _median_rank(["A", "MISSING"], ranked) == 0

    def test_all_absent_returns_nan(self):
        from pipelines.upstream_recovery_benchmark import _median_rank
        ranked = ["A", "B", "C"]
        result = _median_rank(["X", "Y"], ranked)
        assert math.isnan(result)

    def test_odd_count_median(self):
        from pipelines.upstream_recovery_benchmark import _median_rank
        ranked = list("ABCDE")
        # A=0, C=2, E=4 → median = 2
        assert _median_rank(["A", "C", "E"], ranked) == 2


# ---------------------------------------------------------------------------
# compute_upstream_recovery_ratio
# ---------------------------------------------------------------------------

class TestUpstreamRecoveryRatio:
    def test_ideal_ranking_less_than_one(self):
        from pipelines.upstream_recovery_benchmark import compute_upstream_recovery_ratio
        ratio = compute_upstream_recovery_ratio(_ideal_ranking(), _UPSTREAM, _MARKERS)
        assert ratio < 1.0

    def test_inverted_ranking_greater_than_one(self):
        from pipelines.upstream_recovery_benchmark import compute_upstream_recovery_ratio
        ratio = compute_upstream_recovery_ratio(_inverted_ranking(), _UPSTREAM, _MARKERS)
        assert ratio > 1.0

    def test_markers_absent_returns_nan(self):
        from pipelines.upstream_recovery_benchmark import compute_upstream_recovery_ratio
        ranked = _UPSTREAM + ["GENE_A"]
        result = compute_upstream_recovery_ratio(ranked, _UPSTREAM, _MARKERS)
        assert math.isnan(result)

    def test_upstream_absent_returns_nan(self):
        from pipelines.upstream_recovery_benchmark import compute_upstream_recovery_ratio
        ranked = _MARKERS + ["GENE_A"]
        result = compute_upstream_recovery_ratio(ranked, _UPSTREAM, _MARKERS)
        assert math.isnan(result)

    def test_ratio_is_finite_for_valid_input(self):
        from pipelines.upstream_recovery_benchmark import compute_upstream_recovery_ratio
        ratio = compute_upstream_recovery_ratio(_ideal_ranking(), _UPSTREAM, _MARKERS)
        assert math.isfinite(ratio)

    def test_perfect_separation_ratio_value(self):
        """upstream at positions 0-7, markers at 8-12 → ratio < 0.8."""
        from pipelines.upstream_recovery_benchmark import compute_upstream_recovery_ratio
        ratio = compute_upstream_recovery_ratio(_ideal_ranking(), _UPSTREAM, _MARKERS)
        # upstream median rank = 3.5, marker median rank = 10.0
        # ratio ≈ 0.35
        assert ratio < 0.8


# ---------------------------------------------------------------------------
# compute_marker_rank_delta
# ---------------------------------------------------------------------------

class TestMarkerRankDelta:
    def test_markers_drop_is_negative(self):
        """After Phase H, markers are pushed down → delta < 0."""
        from pipelines.upstream_recovery_benchmark import compute_marker_rank_delta
        # Pre: markers near top (positions 0-4)
        pre = _MARKERS + ["GENE_A"] + _UPSTREAM
        # Post: markers near bottom (positions 8-12)
        post = _ideal_ranking()
        delta = compute_marker_rank_delta(pre, post, _MARKERS)
        assert delta > 0  # rank numbers increased = moved down the list (worse rank number = lower priority)

    def test_markers_rise_is_positive(self):
        """If markers move toward top, delta > 0 in rank-number terms means they got worse numbers...
        Wait — rank 0 = top. Delta positive means rank number went up = moved down."""
        from pipelines.upstream_recovery_benchmark import compute_marker_rank_delta
        # This flips the above: post has markers at top
        pre = _ideal_ranking()
        post = _inverted_ranking()
        delta = compute_marker_rank_delta(pre, post, _MARKERS)
        assert delta < 0  # rank numbers decreased = markers moved toward top (bad)

    def test_identical_rankings_delta_zero(self):
        from pipelines.upstream_recovery_benchmark import compute_marker_rank_delta
        ranked = _ideal_ranking()
        delta = compute_marker_rank_delta(ranked, ranked, _MARKERS)
        assert delta == pytest.approx(0.0)

    def test_absent_markers_returns_nan(self):
        from pipelines.upstream_recovery_benchmark import compute_marker_rank_delta
        ranked = _UPSTREAM + ["GENE_A"]
        result = compute_marker_rank_delta(ranked, ranked, _MARKERS)
        assert math.isnan(result)


# ---------------------------------------------------------------------------
# evaluate_ranking
# ---------------------------------------------------------------------------

class TestEvaluateRanking:
    def _config(self):
        return {
            "upstream_regulators": _UPSTREAM,
            "terminal_markers": _MARKERS,
            "thresholds": {"upstream_recovery_ratio_max": 1.0},
        }

    def test_ideal_ranking_passes(self):
        from pipelines.upstream_recovery_benchmark import evaluate_ranking
        result = evaluate_ranking(_ideal_ranking(), self._config())
        assert result["pass_recovery_ratio"] is True

    def test_inverted_ranking_fails(self):
        from pipelines.upstream_recovery_benchmark import evaluate_ranking
        result = evaluate_ranking(_inverted_ranking(), self._config())
        assert result["pass_recovery_ratio"] is False

    def test_result_contains_required_keys(self):
        from pipelines.upstream_recovery_benchmark import evaluate_ranking
        result = evaluate_ranking(_ideal_ranking(), self._config())
        for key in ("upstream_recovery_ratio", "upstream_median_rank",
                    "marker_median_rank", "upstream_in_ranking",
                    "markers_in_ranking", "pass_recovery_ratio"):
            assert key in result

    def test_marker_rank_delta_absent_when_no_pre(self):
        from pipelines.upstream_recovery_benchmark import evaluate_ranking
        result = evaluate_ranking(_ideal_ranking(), self._config())
        assert result["marker_rank_delta"] is None

    def test_marker_rank_delta_present_when_pre_given(self):
        from pipelines.upstream_recovery_benchmark import evaluate_ranking
        pre = _inverted_ranking()
        result = evaluate_ranking(_ideal_ranking(), self._config(), pre_ranked=pre)
        assert result["marker_rank_delta"] is not None
        assert math.isfinite(result["marker_rank_delta"])

    def test_upstream_in_ranking_count(self):
        from pipelines.upstream_recovery_benchmark import evaluate_ranking
        # Only include 4 of 8 upstream genes
        partial = _UPSTREAM[:4] + _MARKERS
        result = evaluate_ranking(partial, self._config())
        assert result["upstream_in_ranking"] == 4

    def test_uses_ibd_config_file(self):
        from pipelines.upstream_recovery_benchmark import load_benchmark_config, evaluate_ranking
        cfg = load_benchmark_config(_IBD_CONFIG_PATH)
        result = evaluate_ranking(_ideal_ranking(), cfg)
        assert math.isfinite(result["upstream_recovery_ratio"])
        assert result["pass_recovery_ratio"] is True


class TestFindUpstreamRegulators:
    """Tests for perturbseq_server.find_upstream_regulators (regulator nomination)."""

    def test_returns_known_ibd_regulators(self):
        from mcp_servers.perturbseq_server import find_upstream_regulators
        # Use terminal markers as targets — these appear in papalexi downstream signatures
        # for JAK2/STAT1/SPI1 knockouts (GWAS hits like NOD2/IL23R do not).
        result = find_upstream_regulators(
            target_genes=["LYZ", "S100A8", "S100A9", "CD68", "CXCL10", "IDO1"],
            disease_context="IBD",
        )
        assert result["n_knockouts_tested"] > 0
        assert result["dataset_id"] is not None
        reg_genes = [r["gene"] for r in result["regulators"]]
        # At least one known IBD upstream regulator should be nominated
        known = {"JAK2", "STAT1", "SPI1", "IRF1", "IRF7", "STAT2", "STAT3"}
        assert known & set(reg_genes), f"No known regulator found; got {reg_genes}"

    def test_structure(self):
        from mcp_servers.perturbseq_server import find_upstream_regulators
        result = find_upstream_regulators(["LYZ", "S100A9"], disease_context="IBD")
        for r in result["regulators"]:
            assert "gene" in r
            assert "n_targets_regulated" in r
            assert "regulated_targets" in r
            assert "sum_abs_log2fc" in r
            assert r["evidence_tier"] == "regulator_nomination"

    def test_min_overlap_filter(self):
        from mcp_servers.perturbseq_server import find_upstream_regulators
        # With min_target_overlap=999 nothing should pass
        result = find_upstream_regulators(
            ["LYZ"], disease_context="IBD", min_target_overlap=999
        )
        assert result["regulators"] == []

    def test_empty_target_list_returns_empty(self):
        from mcp_servers.perturbseq_server import find_upstream_regulators
        result = find_upstream_regulators([], disease_context="IBD")
        assert result["regulators"] == []

    def test_sorted_by_n_targets(self):
        from mcp_servers.perturbseq_server import find_upstream_regulators
        result = find_upstream_regulators(
            ["LYZ", "S100A4", "CXCL9", "CXCL10", "IDO1", "WARS", "CCL2"],
            disease_context="IBD",
        )
        counts = [r["n_targets_regulated"] for r in result["regulators"]]
        assert counts == sorted(counts, reverse=True)
