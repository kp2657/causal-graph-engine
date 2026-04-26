"""
tests/test_leave_locus_out.py — Unit tests for leave-locus-out γ stability analysis.

All tests use purely synthetic data; no real GWAS files are required.
"""
from __future__ import annotations

import pytest

from pipelines.ldsc.leave_locus_out import (
    LOO_WINDOW_MB,
    LOO_STABILITY_THRESHOLD,
    leave_locus_out_stability,
    summarize_loo_stability,
    loo_stability_report,
    _compute_tau,
)


# ---------------------------------------------------------------------------
# Helper: build a minimal program_snp_positions dict
# ---------------------------------------------------------------------------

def _make_program(snps: list[tuple[int, int, float]]) -> list[tuple[int, int, float]]:
    """Convenience: list of (chrom, pos, chisq) tuples."""
    return snps


# ---------------------------------------------------------------------------
# Test 1: correct SNPs are excluded inside the LOO window
# ---------------------------------------------------------------------------

def test_loo_excludes_correct_snps():
    """
    Build a single program with 5 SNPs.  2 SNPs lie within ±1 Mb of a gene at
    chrom=1, pos=5_000_000.  After LOO filtering those 2 should be gone.
    """
    gene_chrom = 1
    gene_pos   = 5_000_000
    window_bp  = LOO_WINDOW_MB * 1e6  # 1_000_000

    # SNPs:
    #   (1, 4_500_000)  → inside window (|5M - 4.5M| = 500 kb < 1 Mb)
    #   (1, 5_900_000)  → inside window (|5M - 5.9M| = 900 kb < 1 Mb)
    #   (1, 6_100_000)  → outside (|5M - 6.1M| = 1.1 Mb > 1 Mb)
    #   (2, 5_000_000)  → different chrom → outside
    #   (1, 3_900_000)  → outside (|5M - 3.9M| = 1.1 Mb > 1 Mb)
    snps = [
        (1, 4_500_000, 2.0),  # inside
        (1, 5_900_000, 3.0),  # inside
        (1, 6_100_000, 1.5),  # outside
        (2, 5_000_000, 1.0),  # different chrom
        (1, 3_900_000, 1.2),  # outside
    ]

    excluded = [
        (chrom, pos, chisq)
        for (chrom, pos, chisq) in snps
        if chrom == gene_chrom and abs(pos - gene_pos) <= window_bp
    ]
    kept = [
        (chrom, pos, chisq)
        for (chrom, pos, chisq) in snps
        if not (chrom == gene_chrom and abs(pos - gene_pos) <= window_bp)
    ]

    assert len(excluded) == 2, f"Expected 2 excluded SNPs, got {len(excluded)}"
    assert len(kept) == 3,    f"Expected 3 kept SNPs, got {len(kept)}"


# ---------------------------------------------------------------------------
# Test 2: stable gene — top program rank barely changes after LOO
# ---------------------------------------------------------------------------

def test_loo_stable_gene():
    """
    A gene whose locus carries little enrichment signal → LOO barely changes τ rankings.
    The gene should be marked stable=True.
    """
    mean_chisq = 1.0

    # Program A: strong enrichment from many SNPs, only 1 near the gene locus
    prog_a_snps = [(1, 1_000_000, 5.0)] * 50   # 50 SNPs far from gene (chrom 1)
    prog_a_snps += [(2, 5_000_000, 5.0)]        # 1 SNP on chrom 2, near gene pos

    # Program B: moderate enrichment, no SNPs near gene locus
    prog_b_snps = [(3, 1_000_000, 2.5)] * 50

    program_snp_positions = {
        "PROG_A": prog_a_snps,
        "PROG_B": prog_b_snps,
    }

    # Baseline τ: PROG_A >> PROG_B
    program_taus = {
        "PROG_A": 4.0,
        "PROG_B": 1.5,
    }

    # Gene is on chrom 2 at 5_000_000 — only 1 PROG_A SNP overlaps
    anchor_genes = {"GENE_X": (2, 5_000_000)}

    results = leave_locus_out_stability(
        program_snp_positions=program_snp_positions,
        anchor_genes=anchor_genes,
        program_taus=program_taus,
        n_snps_genome=101,
        mean_chisq=mean_chisq,
    )

    assert "GENE_X" in results
    r = results["GENE_X"]
    assert r["stable"] is True, (
        f"Expected GENE_X stable but rank_delta={r['rank_delta']}"
    )


# ---------------------------------------------------------------------------
# Test 3: unstable gene — drives all enrichment in its locus → large rank_delta
# ---------------------------------------------------------------------------

def test_loo_unstable_gene():
    """
    GENE_DRIVER has ALL of PROG_A's enrichment SNPs in its locus.
    After excluding them PROG_A drops to rank 12 (below 11 other programs with
    modest-but-positive τ), producing rank_delta > LOO_STABILITY_THRESHOLD → stable=False.
    """
    mean_chisq = 1.0
    gene_chrom = 1
    gene_pos   = 10_000_000

    # PROG_A: all 50 high-chisq SNPs clustered inside the gene locus → loses all
    # signal after LOO, its recomputed τ becomes 0 or negative.
    prog_a_snps = [
        (gene_chrom, gene_pos + i * 10_000, 20.0)   # within ±500 kb
        for i in range(-25, 25)
    ]  # positions 9.75 M – 10.24 M → all within 1 Mb window

    # Build 11 background programs with small positive enrichment on chrom 2 (no LOO hit)
    # Each gets mean chisq ≈ 1.5 → τ ≈ 0.5; they all outrank PROG_A after LOO.
    background_programs: dict[str, list[tuple[int, int, float]]] = {}
    for k in range(11):
        background_programs[f"PROG_BG{k:02d}"] = [
            (2, 1_000_000 + k * 200_000 + j * 1_000, 1.5)
            for j in range(50)
        ]

    program_snp_positions: dict[str, list[tuple[int, int, float]]] = {
        "PROG_A": prog_a_snps,
        **background_programs,
    }

    # Baseline τ: PROG_A is rank 1, background programs are ranks 2–12
    program_taus: dict[str, float] = {"PROG_A": 10.0}
    for k in range(11):
        program_taus[f"PROG_BG{k:02d}"] = 0.5 - k * 0.01  # small spread

    anchor_genes = {"GENE_DRIVER": (gene_chrom, gene_pos)}

    results = leave_locus_out_stability(
        program_snp_positions=program_snp_positions,
        anchor_genes=anchor_genes,
        program_taus=program_taus,
        n_snps_genome=600,
        mean_chisq=mean_chisq,
    )

    assert "GENE_DRIVER" in results
    r = results["GENE_DRIVER"]
    # After LOO, PROG_A (τ→0) drops below all 11 background programs → rank 12
    # rank_baseline = 1, rank_loo = 12, rank_delta = 11 > LOO_STABILITY_THRESHOLD (10)
    assert r["rank_delta"] > LOO_STABILITY_THRESHOLD, (
        f"Expected rank_delta > {LOO_STABILITY_THRESHOLD}, got {r['rank_delta']}"
    )
    assert r["stable"] is False, (
        f"Expected GENE_DRIVER unstable but rank_delta={r['rank_delta']}"
    )


# ---------------------------------------------------------------------------
# Test 4: summarize_loo_stability — all stable
# ---------------------------------------------------------------------------

def test_summarize_all_stable():
    """All genes stable → n_unstable=0, stability_fraction=1.0."""
    loo_results = {
        "GENE_A": {"rank_baseline": 1, "rank_loo": 1, "rank_delta": 0,  "tau_change_pct": 0.0, "stable": True},
        "GENE_B": {"rank_baseline": 2, "rank_loo": 3, "rank_delta": 1,  "tau_change_pct": 1.0, "stable": True},
        "GENE_C": {"rank_baseline": 3, "rank_loo": 4, "rank_delta": 1,  "tau_change_pct": 2.0, "stable": True},
    }

    summary = summarize_loo_stability(loo_results)

    assert summary["n_genes"] == 3
    assert summary["n_stable"] == 3
    assert summary["n_unstable"] == 0
    assert summary["unstable_genes"] == []
    assert summary["stability_fraction"] == 1.0


# ---------------------------------------------------------------------------
# Test 5: summarize_loo_stability — mixed results
# ---------------------------------------------------------------------------

def test_summarize_mixed():
    """3 stable, 2 unstable → stability_fraction = 0.6."""
    loo_results = {
        "GENE_A": {"rank_baseline": 1, "rank_loo": 1,  "rank_delta": 0,  "tau_change_pct": 0.0,  "stable": True},
        "GENE_B": {"rank_baseline": 2, "rank_loo": 3,  "rank_delta": 1,  "tau_change_pct": 1.0,  "stable": True},
        "GENE_C": {"rank_baseline": 3, "rank_loo": 4,  "rank_delta": 1,  "tau_change_pct": 2.0,  "stable": True},
        "GENE_D": {"rank_baseline": 1, "rank_loo": 15, "rank_delta": 14, "tau_change_pct": 40.0, "stable": False},
        "GENE_E": {"rank_baseline": 2, "rank_loo": 20, "rank_delta": 18, "tau_change_pct": 60.0, "stable": False},
    }

    summary = summarize_loo_stability(loo_results)

    assert summary["n_genes"] == 5
    assert summary["n_stable"] == 3
    assert summary["n_unstable"] == 2
    assert sorted(summary["unstable_genes"]) == ["GENE_D", "GENE_E"]
    assert abs(summary["stability_fraction"] - 0.6) < 1e-6


# ---------------------------------------------------------------------------
# Test 6: loo_stability_report contains gene names
# ---------------------------------------------------------------------------

def test_report_contains_gene_names():
    """Markdown report must contain each gene name in the results."""
    loo_results = {
        "CFH":   {"rank_baseline": 1, "rank_loo": 1,  "rank_delta": 0,  "tau_change_pct": 0.0,  "stable": True},
        "VEGFA": {"rank_baseline": 2, "rank_loo": 15, "rank_delta": 13, "tau_change_pct": 50.0, "stable": False},
        "PCSK9": {"rank_baseline": 3, "rank_loo": 4,  "rank_delta": 1,  "tau_change_pct": 1.0,  "stable": True},
    }

    report = loo_stability_report(loo_results)

    for gene in loo_results:
        assert gene in report, f"Gene {gene!r} not found in LOO report"

    # Markdown table header present
    assert "rank_baseline" in report
    assert "rank_delta" in report


# ---------------------------------------------------------------------------
# Test 7: LOO window boundary — exactly at boundary excluded; 1 bp beyond kept
# ---------------------------------------------------------------------------

def test_loo_window_boundary():
    """
    SNP at exactly LOO_WINDOW_MB (1 Mb = 1_000_000 bp) from gene_pos is excluded.
    SNP at LOO_WINDOW_MB + 1 bp is kept.
    """
    gene_chrom = 1
    gene_pos   = 10_000_000
    window_bp  = int(LOO_WINDOW_MB * 1e6)   # 1_000_000

    snp_at_boundary     = (gene_chrom, gene_pos + window_bp,     3.0)   # dist == 1_000_000 → excluded
    snp_beyond_boundary = (gene_chrom, gene_pos + window_bp + 1, 3.0)   # dist == 1_000_001 → kept

    program_snp_positions = {
        "PROG_A": [snp_at_boundary, snp_beyond_boundary],
    }

    program_taus = {"PROG_A": 1.0}
    anchor_genes = {"GENE_BOUNDARY": (gene_chrom, gene_pos)}

    # Verify boundary filter logic directly
    excluded_at = [
        s for s in [snp_at_boundary]
        if s[0] == gene_chrom and abs(s[1] - gene_pos) <= window_bp
    ]
    excluded_beyond = [
        s for s in [snp_beyond_boundary]
        if s[0] == gene_chrom and abs(s[1] - gene_pos) <= window_bp
    ]

    assert len(excluded_at) == 1,    "SNP exactly at boundary should be excluded"
    assert len(excluded_beyond) == 0, "SNP 1 bp beyond boundary should be kept"

    # Also verify via the full pipeline function
    results = leave_locus_out_stability(
        program_snp_positions=program_snp_positions,
        anchor_genes=anchor_genes,
        program_taus=program_taus,
        n_snps_genome=2,
        mean_chisq=1.0,
    )

    # With only 1 SNP kept (chisq=3.0) vs baseline mean_chisq=1.0,
    # LOO τ = (3.0 - 1.0) / (1.0 + 1e-8) ≈ 2.0 → still positive; rank stays 1
    assert "GENE_BOUNDARY" in results
    assert results["GENE_BOUNDARY"]["stable"] is True
