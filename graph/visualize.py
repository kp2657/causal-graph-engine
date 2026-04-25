"""
graph/visualize.py — Publication-ready figures from GraphOutput dict.

Three figures produced per disease run:
  1. target_rankings.png  — Horizontal bar chart: genes ranked by |Ota γ|, coloured by tier
  2. causal_network.png   — 3-layer network: Genes → Programs → Traits
  3. evidence_summary.png — Evidence quality dashboard: tier breakdown + anchor recovery gauge

All figures use a consistent palette and are suitable for papers/reports.

Usage:
    from graph.visualize import generate_report_figures
    paths = generate_report_figures(result, output_dir="./data/figures")
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Colour palette (tier → hex)
# ---------------------------------------------------------------------------

TIER_COLORS: dict[str, str] = {
    "Tier1_Interventional": "#1a6faf",   # deep blue — strongest evidence
    "Tier2_Convergent":     "#2ca02c",   # green — MR convergent
    "Tier3_Provisional":    "#ff7f0e",   # orange — provisional
    "provisional_virtual":  "#9b9b9b",   # grey — in silico only
}

TIER_LABELS: dict[str, str] = {
    "Tier1_Interventional": "Tier 1 — Interventional",
    "Tier2_Convergent":     "Tier 2 — Convergent MR",
    "Tier3_Provisional":    "Tier 3 — Provisional",
    "provisional_virtual":  "Virtual (in silico)",
}

_FIGURE_DPI = 150
_FONT_FAMILY = "DejaVu Sans"


# ---------------------------------------------------------------------------
# 1. Target Rankings
# ---------------------------------------------------------------------------

def plot_target_rankings(result: dict, output_path: str | Path) -> Path:
    """
    Horizontal bar chart: genes ranked by |Ota γ|, coloured by evidence tier.

    Positive γ bars extend right (risk-increasing), negative left (protective).
    A vertical dashed line at x=0 separates directions.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    targets = result.get("target_list", result.get("targets", []))
    if not targets:
        return _empty_figure("No targets to plot", output_path)

    # Sort by |gamma| descending, take top 12
    targets = sorted(targets, key=lambda t: abs(t.get("ota_gamma", 0.0)), reverse=True)[:12]

    genes   = [t.get("target_gene", "?") for t in targets]
    gammas  = [t.get("ota_gamma", 0.0)   for t in targets]
    tiers   = [t.get("evidence_tier", "provisional_virtual") for t in targets]
    colors  = [TIER_COLORS.get(ti, "#9b9b9b") for ti in tiers]

    fig, ax = plt.subplots(figsize=(8, max(4, len(genes) * 0.45)))

    y_pos = range(len(genes))
    bars = ax.barh(y_pos, gammas, color=colors, edgecolor="white", linewidth=0.5, height=0.7)

    # Add γ value labels
    for bar, g in zip(bars, gammas):
        x_label = g + (0.015 if g >= 0 else -0.015)
        ha = "left" if g >= 0 else "right"
        ax.text(x_label, bar.get_y() + bar.get_height() / 2,
                f"{g:+.3f}", va="center", ha=ha, fontsize=8, color="#333333")

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(genes, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color="#444444", linewidth=0.8, linestyle="--")

    # Axis labels and title
    disease = result.get("disease_name", "Disease").title()
    ax.set_xlabel("Composite Ota γ  (gene → trait causal effect size)", fontsize=9)
    ax.set_title(f"Drug Target Rankings — {disease}", fontsize=11, fontweight="bold", pad=10)

    # Colour legend
    seen_tiers = list(dict.fromkeys(tiers))  # preserve order, unique
    legend_patches = [
        mpatches.Patch(color=TIER_COLORS.get(ti, "#9b9b9b"),
                       label=TIER_LABELS.get(ti, ti))
        for ti in seen_tiers
    ]
    ax.legend(handles=legend_patches, fontsize=8, loc="lower right",
              framealpha=0.9, edgecolor="#cccccc")

    # Annotations (no anchor recovery — removed)

    _style_ax(ax)
    fig.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=_FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 2. Causal Network Graph
# ---------------------------------------------------------------------------

def plot_causal_network(result: dict, output_path: str | Path) -> Path:
    """
    3-layer network: Genes → (implied Programs) → Traits.

    Nodes are sized by effect magnitude. Edges are coloured by tier.
    Protective edges (γ < 0) are drawn dashed.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import networkx as nx

    targets = result.get("target_list", result.get("targets", []))
    disease = result.get("disease_name", "disease").title()

    if not targets:
        return _empty_figure("No targets to plot network", output_path)

    # Limit to top 8 for readability
    targets = sorted(targets, key=lambda t: abs(t.get("ota_gamma", 0.0)), reverse=True)[:8]

    G = nx.DiGraph()

    # Layer x-positions
    X_GENE, X_PROG, X_TRAIT = 0.0, 1.8, 3.6

    # Collect unique traits from DISEASE_TRAIT_MAP
    try:
        from graph.schema import DISEASE_TRAIT_MAP, _DISEASE_SHORT_NAMES_FOR_ANCHORS
        d_name = result.get("disease_name", "coronary artery disease")
        short  = _DISEASE_SHORT_NAMES_FOR_ANCHORS.get(d_name.lower(), "CAD")
        traits = DISEASE_TRAIT_MAP.get(short, [short])[:3]  # top 3 traits
    except Exception:
        traits = ["CAD"]

    # Add trait nodes
    for ti, trait in enumerate(traits):
        G.add_node(trait, layer="trait", x=X_TRAIT,
                   y=ti * 2.0 - (len(traits) - 1))

    gene_positions: dict[str, tuple[float, float]] = {}
    edge_attrs: list[dict] = []

    n = len(targets)
    for i, t in enumerate(targets):
        gene  = t.get("target_gene", "?")
        gamma = t.get("ota_gamma", 0.0)
        tier  = t.get("evidence_tier", "provisional_virtual")
        progs = t.get("top_programs", [])

        y_gene = (i - (n - 1) / 2) * 1.5
        G.add_node(gene, layer="gene", x=X_GENE, y=y_gene)
        gene_positions[gene] = (X_GENE, y_gene)

        # Program intermediary (use first program if available, else infer from flags)
        if progs:
            prog_name = progs[0]
        elif "chip_mechanism" in t.get("flags", []):
            prog_name = "inflammatory\nNF-kB"
        elif abs(gamma) > 0.3:
            prog_name = "lipid\nmetabolism"
        else:
            prog_name = "mixed\nprograms"

        prog_id = f"P:{gene}"
        y_prog = y_gene
        G.add_node(prog_id, layer="program", x=X_PROG, y=y_prog, label=prog_name)

        # Gene → Program edge
        G.add_edge(gene, prog_id, gamma=abs(gamma), tier=tier, sign=math.copysign(1, gamma))
        # Program → Trait edge (connect to most relevant trait)
        target_trait = traits[0]
        G.add_edge(prog_id, target_trait, gamma=abs(gamma), tier=tier, sign=math.copysign(1, gamma))

    # Build position dict
    pos: dict[str, tuple[float, float]] = {}
    for node, data in G.nodes(data=True):
        pos[node] = (data["x"], data["y"])

    fig, ax = plt.subplots(figsize=(12, max(6, n * 1.1)))

    # Draw nodes by layer
    gene_nodes  = [n for n, d in G.nodes(data=True) if d.get("layer") == "gene"]
    prog_nodes  = [n for n, d in G.nodes(data=True) if d.get("layer") == "program"]
    trait_nodes = [n for n, d in G.nodes(data=True) if d.get("layer") == "trait"]

    # Node sizes by |gamma| or fixed
    gene_gammas = {t.get("target_gene"): abs(t.get("ota_gamma", 0.0)) for t in targets}
    gene_sizes  = [max(600, gene_gammas.get(g, 0.1) * 3000) for g in gene_nodes]

    nx.draw_networkx_nodes(G, pos, nodelist=gene_nodes,
                           node_color="#1a6faf", node_size=gene_sizes,
                           alpha=0.9, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=prog_nodes,
                           node_color="#e8f4fd", node_size=800,
                           edgecolors="#1a6faf", linewidths=1.2, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=trait_nodes,
                           node_color="#d62728", node_size=1400,
                           alpha=0.85, ax=ax)

    # Edge colours by tier, style by sign
    for u, v, data in G.edges(data=True):
        tier  = data.get("tier", "provisional_virtual")
        sign  = data.get("sign", 1.0)
        gamma = data.get("gamma", 0.1)
        color = TIER_COLORS.get(tier, "#9b9b9b")
        style = "dashed" if sign < 0 else "solid"
        width = max(0.8, gamma * 3.5)
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)],
                               edge_color=color, style=style,
                               width=width, alpha=0.75,
                               arrows=True, arrowsize=12,
                               connectionstyle="arc3,rad=0.05",
                               ax=ax)

    # Gene labels
    gene_label_dict = {g: g for g in gene_nodes}
    nx.draw_networkx_labels(G, pos, labels=gene_label_dict,
                            font_size=8, font_color="white",
                            font_weight="bold", ax=ax)

    # Program labels (use custom label if set)
    prog_label_dict = {n: G.nodes[n].get("label", n.replace("P:", ""))
                       for n in prog_nodes}
    nx.draw_networkx_labels(G, pos, labels=prog_label_dict,
                            font_size=6.5, font_color="#1a6faf", ax=ax)

    # Trait labels
    trait_label_dict = {t: t for t in trait_nodes}
    nx.draw_networkx_labels(G, pos, labels=trait_label_dict,
                            font_size=8.5, font_color="white",
                            font_weight="bold", ax=ax)

    # Layer headers
    y_top = max(d["y"] for _, d in G.nodes(data=True)) + 1.5
    for x, label in [(X_GENE, "Genes"), (X_PROG, "Programs"), (X_TRAIT, "Traits")]:
        ax.text(x, y_top, label, ha="center", va="center",
                fontsize=10, fontweight="bold", color="#333333",
                bbox=dict(boxstyle="round,pad=0.3", fc="#f5f5f5", ec="#cccccc", lw=1))

    # Legend
    legend_patches = [
        mpatches.Patch(color=c, label=TIER_LABELS.get(k, k))
        for k, c in TIER_COLORS.items()
    ]
    legend_patches += [
        plt.Line2D([0], [0], color="#555", lw=1.5, linestyle="solid",  label="Risk-increasing (γ > 0)"),
        plt.Line2D([0], [0], color="#555", lw=1.5, linestyle="dashed", label="Protective (γ < 0)"),
    ]
    ax.legend(handles=legend_patches, fontsize=7.5, loc="lower left",
              framealpha=0.95, edgecolor="#cccccc", ncol=2)

    ax.set_title(f"Causal Network — {disease}", fontsize=12, fontweight="bold", pad=12)
    ax.axis("off")
    fig.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=_FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 3. Evidence Summary Dashboard
# ---------------------------------------------------------------------------

def plot_evidence_summary(result: dict, output_path: str | Path) -> Path:
    """
    2-panel figure:
      Left  — Donut chart: edge count by evidence tier
      Right — Anchor recovery gauge + SHD/pipeline metrics
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    eq       = result.get("evidence_quality", {})
    disease  = result.get("disease_name", "Disease").title()

    t1 = eq.get("n_tier1_edges",   result.get("n_tier1_edges", 0))
    t2 = eq.get("n_tier2_edges",   result.get("n_tier2_edges", 0))
    t3 = eq.get("n_tier3_edges",   result.get("n_tier3_edges", 0))
    tv = eq.get("n_virtual_edges", result.get("n_virtual_edges", 0))
    duration = result.get("pipeline_duration_s")
    n_targets = len(result.get("target_list", []))

    fig, (ax_donut, ax_metrics) = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(f"Evidence Quality Dashboard — {disease}",
                 fontsize=12, fontweight="bold", y=1.01)

    # ---- Left: Donut ----
    tier_counts = [t1, t2, t3, tv]
    tier_names  = ["Tier 1\nInterventional", "Tier 2\nConvergent MR",
                   "Tier 3\nProvisional", "Virtual\n(in silico)"]
    tier_cols   = [TIER_COLORS["Tier1_Interventional"],
                   TIER_COLORS["Tier2_Convergent"],
                   TIER_COLORS["Tier3_Provisional"],
                   TIER_COLORS["provisional_virtual"]]

    # Only plot non-zero slices
    nonzero = [(c, n, col) for c, n, col in zip(tier_counts, tier_names, tier_cols) if c > 0]
    if nonzero:
        counts, names, cols = zip(*nonzero)
        wedges, texts, autotexts = ax_donut.pie(
            counts, labels=names, colors=cols,
            autopct="%1.0f%%", startangle=90,
            wedgeprops=dict(width=0.5, edgecolor="white", linewidth=1.5),
            pctdistance=0.75, textprops=dict(fontsize=8.5),
        )
        for at in autotexts:
            at.set_fontsize(8)
            at.set_color("white")
            at.set_fontweight("bold")
        total = sum(counts)
        ax_donut.text(0, 0, f"{total}\nedges", ha="center", va="center",
                      fontsize=11, fontweight="bold", color="#333333")
    else:
        ax_donut.text(0, 0, "No edges\nin graph", ha="center", va="center",
                      fontsize=10, color="#888888")
        ax_donut.axis("off")

    ax_donut.set_title("Edge Evidence Tiers", fontsize=10, pad=8)

    # ---- Right: Metrics table ----
    ax_metrics.set_xlim(0, 1)
    ax_metrics.set_ylim(0, 1)
    ax_metrics.axis("off")

    metrics = [
        ("Tier 1 (interventional)", str(t1)),
        ("Tier 2 (convergent MR)",  str(t2)),
        ("Tier 3 (provisional)",    str(t3)),
        ("Virtual (in silico)",     str(tv)),
        ("Targets ranked",          str(n_targets)),
        ("Pipeline duration",       f"{duration:.0f}s" if duration else "N/A"),
    ]
    y_start = 0.75
    for label, val in metrics:
        ax_metrics.text(0.10, y_start, label, fontsize=9, color="#444444", va="center")
        ax_metrics.text(0.90, y_start, val,   fontsize=9, color="#222222",
                        va="center", ha="right", fontweight="bold")
        ax_metrics.axhline(y_start - 0.025, xmin=0.08, xmax=0.92,
                           color="#eeeeee", linewidth=0.5)
        y_start -= 0.10

    ax_metrics.set_title("Pipeline Metrics", fontsize=10, pad=8)

    fig.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=_FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Combined entry point
# ---------------------------------------------------------------------------

def generate_report_figures(
    result: dict,
    output_dir: str | Path = "./data/figures",
) -> dict[str, Path]:
    """
    Generate all three figures for a pipeline result.

    Args:
        result:     GraphOutput dict from analyze_disease / analyze_disease_v2
        output_dir: Directory to write PNG files into

    Returns:
        Dict mapping figure name → Path
    """
    slug = result.get("disease_name", "disease").lower().replace(" ", "_")
    out  = Path(output_dir)

    paths: dict[str, Path] = {}
    errors: list[str] = []

    for name, fn, suffix in [
        ("target_rankings",  plot_target_rankings,  "target_rankings"),
        ("causal_network",   plot_causal_network,   "causal_network"),
        ("evidence_summary", plot_evidence_summary, "evidence_summary"),
    ]:
        try:
            p = fn(result, out / f"{slug}_{suffix}.png")
            paths[name] = p
        except Exception as exc:
            errors.append(f"{name}: {exc}")

    if errors:
        import warnings
        warnings.warn(f"Figure generation errors: {errors}")

    return paths


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty_figure(message: str, output_path: str | Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.text(0.5, 0.5, message, ha="center", va="center",
            fontsize=12, color="#888888", transform=ax.transAxes)
    ax.axis("off")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=_FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def _style_ax(ax: Any) -> None:
    """Apply consistent minimal styling to a matplotlib Axes."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(axis="x", linestyle="--", linewidth=0.4, alpha=0.5, color="#cccccc")
    ax.set_axisbelow(True)
