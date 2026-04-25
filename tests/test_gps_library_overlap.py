"""Tests for GPS library overlap pre-screen (prevents Run_reversal_score min() crash)."""
import pytest
from unittest.mock import patch
from pipelines.gps_screen import _check_gps_library_overlap


_GPS_GENES = {"GENE_A", "GENE_B", "GENE_C", "GENE_D", "GENE_E", "GENE_F"}


def _with_gps_genes(sig, **kwargs):
    with patch("pipelines.gps_screen._get_gps_genes", return_value=_GPS_GENES):
        return _check_gps_library_overlap("test_label", sig, **kwargs)


class TestCheckGpsLibraryOverlap:
    def test_bidirectional_passes(self):
        sig = {"GENE_A": 1.2, "GENE_B": 0.8, "GENE_C": -0.9, "GENE_D": -0.5}
        assert _with_gps_genes(sig) is True

    def test_all_positive_fails(self):
        # BGRD would only have Up_* bins → min(bins_down) crashes during scoring
        sig = {"GENE_A": 1.2, "GENE_B": 0.8, "GENE_C": 0.3}
        assert _with_gps_genes(sig) is False

    def test_all_negative_fails(self):
        sig = {"GENE_A": -1.2, "GENE_B": -0.8, "GENE_C": -0.3}
        assert _with_gps_genes(sig) is False

    def test_zero_overlap_fails(self):
        sig = {"NO_MATCH_1": 1.0, "NO_MATCH_2": -1.0}
        assert _with_gps_genes(sig) is False

    def test_below_min_per_direction_fails(self):
        # Only 1 down gene (< min_per_direction=2)
        sig = {"GENE_A": 1.5, "GENE_B": 1.2, "GENE_C": -0.3}
        assert _with_gps_genes(sig, min_per_direction=2) is False

    def test_exactly_min_per_direction_passes(self):
        sig = {"GENE_A": 1.5, "GENE_B": 1.2, "GENE_C": -0.3, "GENE_D": -0.9}
        assert _with_gps_genes(sig, min_per_direction=2) is True

    def test_no_gps_gene_list_always_passes(self):
        # If GPS gene list unavailable, don't block the run
        sig = {"GENE_X": 1.0}
        with patch("pipelines.gps_screen._get_gps_genes", return_value=None):
            assert _check_gps_library_overlap("test_label", sig) is True

    def test_non_library_genes_excluded_from_direction_count(self):
        # GENE_Z not in library; only GENE_A (up) + GENE_C (down) survive
        sig = {"GENE_A": 1.0, "GENE_Z": -2.0, "GENE_C": -0.5}
        # up=1, down=1 — passes min_per_direction=1
        assert _with_gps_genes(sig, min_per_direction=1) is True
        # fails min_per_direction=2
        assert _with_gps_genes(sig, min_per_direction=2) is False
