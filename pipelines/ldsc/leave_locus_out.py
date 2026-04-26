"""
leave_locus_out.py — Leave-locus-out γ stability analysis.

For each anchor gene, exclude SNPs within ±LOO_WINDOW_MB of its genomic position
from the program χ² vector, recompute τ enrichment, and record rank change.

This tests whether the OTA γ ranking is robust to removing the signal from
each gene's own locus (circularity guard).
"""
from __future__ import annotations

import math
from typing import Any

# Exclusion window around each gene's position (±1 Mb)
LOO_WINDOW_MB: float = 1.0

# Import stability threshold from central config; fall back to local definition
# if config is unavailable in standalone test contexts.
try:
    from config.scoring_thresholds import LOO_STABILITY_THRESHOLD
except ImportError:
    LOO_STABILITY_THRESHOLD: int = 10  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_tau(
    snp_chisq_values: list[float],
    mean_chisq_genome: float,
) -> float | None:
    """
    Compute τ enrichment = (mean_prog_chisq - mean_genome_chisq) / (mean_genome_chisq + ε).
    Returns None when the filtered SNP list is empty.
    """
    if not snp_chisq_values:
        return None
    mean_prog = sum(snp_chisq_values) / len(snp_chisq_values)
    return (mean_prog - mean_chisq_genome) / (mean_chisq_genome + 1e-8)


def _rank_programs_by_tau(
    program_taus: dict[str, float],
) -> dict[str, int]:
    """
    Rank programs from highest τ (rank 1) to lowest.
    Ties broken by program name for determinism.
    """
    sorted_progs = sorted(
        program_taus.keys(),
        key=lambda p: (-program_taus[p], p),
    )
    return {prog: idx + 1 for idx, prog in enumerate(sorted_progs)}


def _gene_top_program(
    gene: str,
    program_snp_positions: dict[str, list[tuple[int, int, float]]],
    gene_chrom: int,
    gene_pos: int,
    mean_chisq_genome: float,
) -> str | None:
    """
    Find the program in which `gene`'s locus contributes the most SNPs,
    used as the 'gene's top-contributing program' for rank tracking.

    Returns the program id whose window contains the most SNPs near `gene_pos`,
    or the program with highest baseline τ if the locus-contribution heuristic
    yields a tie / no match.
    """
    window_bp = LOO_WINDOW_MB * 1e6
    best_prog: str | None = None
    best_count: int = -1
    for prog, snps in program_snp_positions.items():
        count = sum(
            1 for (chrom, pos, _) in snps
            if chrom == gene_chrom and abs(pos - gene_pos) <= window_bp
        )
        if count > best_count or (count == best_count and best_prog is None):
            best_count = count
            best_prog = prog
    return best_prog


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def leave_locus_out_stability(
    program_snp_positions: dict[str, list[tuple[int, int, float]]],
    anchor_genes: dict[str, tuple[int, int]],   # gene → (chrom, pos_bp)
    program_taus: dict[str, float],              # baseline τ per program
    n_snps_genome: int,
    mean_chisq: float,
) -> dict[str, dict]:
    """
    For each anchor gene, exclude its locus and recompute program τ enrichments.

    Args:
        program_snp_positions: dict[program_id → list of (chrom, pos, chisq)]
        anchor_genes:          dict[gene_symbol → (chrom_int, pos_bp)]
        program_taus:          baseline τ per program (pre-computed)
        n_snps_genome:         total number of genome-wide SNPs (informational)
        mean_chisq:            genome-wide mean χ² (used as background for τ recomputation)

    Returns:
        dict[gene → {
            "rank_baseline": int,
            "rank_loo":      int,
            "rank_delta":    int,           # abs(rank_loo - rank_baseline)
            "tau_change_pct": float,        # % change in max τ across programs
            "stable":        bool,          # rank_delta <= LOO_STABILITY_THRESHOLD
        }]
    """
    window_bp = LOO_WINDOW_MB * 1e6

    # Baseline ranks (all programs, all SNPs)
    baseline_ranks = _rank_programs_by_tau(program_taus)

    results: dict[str, dict] = {}

    for gene, (gene_chrom, gene_pos) in anchor_genes.items():
        # Determine which program is the gene's primary contributor (for rank tracking)
        top_prog = _gene_top_program(
            gene,
            program_snp_positions,
            gene_chrom,
            gene_pos,
            mean_chisq,
        )

        # Fallback: use the program with the highest baseline τ
        if top_prog is None and program_taus:
            top_prog = max(program_taus, key=lambda p: program_taus[p])

        # Build LOO τ: for each program, exclude SNPs within ±window of gene locus
        loo_taus: dict[str, float] = {}
        for prog, snps in program_snp_positions.items():
            filtered = [
                chisq for (chrom, pos, chisq) in snps
                if not (chrom == gene_chrom and abs(pos - gene_pos) <= window_bp)
            ]
            tau_loo = _compute_tau(filtered, mean_chisq)
            # When all SNPs for this program are inside the locus, the filtered list
            # is empty.  Assign τ=0 (no enrichment signal outside locus) rather than
            # falling back to the baseline — that would hide the instability.
            if tau_loo is None:
                tau_loo = 0.0
            loo_taus[prog] = tau_loo

        loo_ranks = _rank_programs_by_tau(loo_taus)

        # Rank of the gene's top program in baseline vs LOO
        if top_prog is not None:
            rank_baseline = baseline_ranks.get(top_prog, len(program_taus))
            rank_loo = loo_ranks.get(top_prog, len(loo_taus))
        else:
            rank_baseline = 1
            rank_loo = 1

        rank_delta = abs(rank_loo - rank_baseline)

        # τ change %: compare max τ baseline vs max τ LOO (sensitivity of peak signal)
        max_tau_baseline = max(program_taus.values()) if program_taus else 0.0
        max_tau_loo = max(loo_taus.values()) if loo_taus else 0.0
        if abs(max_tau_baseline) > 1e-10:
            tau_change_pct = abs(max_tau_loo - max_tau_baseline) / abs(max_tau_baseline) * 100.0
        else:
            tau_change_pct = 0.0

        results[gene] = {
            "rank_baseline":  rank_baseline,
            "rank_loo":       rank_loo,
            "rank_delta":     rank_delta,
            "tau_change_pct": round(tau_change_pct, 2),
            "stable":         rank_delta <= LOO_STABILITY_THRESHOLD,
        }

    return results


def summarize_loo_stability(loo_results: dict[str, dict]) -> dict:
    """
    Summarize stability across all genes.

    Returns:
        {
            "n_genes":            int,
            "n_stable":           int,
            "n_unstable":         int,
            "unstable_genes":     list[str],
            "median_rank_delta":  float,
            "stability_fraction": float,
        }
    """
    n_genes = len(loo_results)
    if n_genes == 0:
        return {
            "n_genes":            0,
            "n_stable":           0,
            "n_unstable":         0,
            "unstable_genes":     [],
            "median_rank_delta":  0.0,
            "stability_fraction": 1.0,
        }

    stable_genes = [g for g, r in loo_results.items() if r["stable"]]
    unstable_genes = [g for g, r in loo_results.items() if not r["stable"]]

    rank_deltas = sorted(r["rank_delta"] for r in loo_results.values())
    mid = len(rank_deltas) // 2
    if len(rank_deltas) % 2 == 1:
        median_delta = float(rank_deltas[mid])
    else:
        median_delta = (rank_deltas[mid - 1] + rank_deltas[mid]) / 2.0

    return {
        "n_genes":            n_genes,
        "n_stable":           len(stable_genes),
        "n_unstable":         len(unstable_genes),
        "unstable_genes":     sorted(unstable_genes),
        "median_rank_delta":  round(median_delta, 2),
        "stability_fraction": round(len(stable_genes) / n_genes, 4),
    }


def loo_stability_report(loo_results: dict[str, dict]) -> str:
    """
    Return a human-readable markdown string summarizing leave-locus-out results.

    Format: table of gene | rank_baseline | rank_loo | rank_delta | stable
    """
    if not loo_results:
        return "## Leave-Locus-Out γ Stability Report\n\n_No anchor genes analysed._\n"

    summary = summarize_loo_stability(loo_results)

    lines: list[str] = [
        "## Leave-Locus-Out γ Stability Report",
        "",
        f"- **Genes analysed:** {summary['n_genes']}",
        f"- **Stable (rank Δ ≤ {LOO_STABILITY_THRESHOLD}):** {summary['n_stable']} "
        f"({summary['stability_fraction'] * 100:.1f}%)",
        f"- **Unstable:** {summary['n_unstable']}",
        f"- **Median rank Δ:** {summary['median_rank_delta']}",
        "",
        "| Gene | rank_baseline | rank_loo | rank_delta | stable |",
        "|------|--------------|----------|------------|--------|",
    ]

    for gene in sorted(loo_results.keys()):
        r = loo_results[gene]
        stable_str = "yes" if r["stable"] else "NO"
        lines.append(
            f"| {gene} | {r['rank_baseline']} | {r['rank_loo']} "
            f"| {r['rank_delta']} | {stable_str} |"
        )

    lines.append("")
    if summary["unstable_genes"]:
        lines.append(
            f"**Unstable genes:** {', '.join(summary['unstable_genes'])} — "
            "ranks shift substantially when their own locus is excluded; "
            "may reflect circularity in γ estimation."
        )
    else:
        lines.append(
            "All anchor genes show stable γ ranking under leave-locus-out "
            f"(rank Δ ≤ {LOO_STABILITY_THRESHOLD} for all)."
        )
    lines.append("")

    return "\n".join(lines)
