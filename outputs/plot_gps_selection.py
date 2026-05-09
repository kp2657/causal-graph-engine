"""
outputs/plot_gps_selection.py — GPS gene + program selection rationale figure.

Six-panel figure (3 cols × 2 rows):
  Row 1 (CAD): [A] program τ* bars  [B] β heatmap (benchmark+novel × GPS programs)  [C] OTA γ ranking
  Row 2 (RA):  [D] program τ* bars  [E] β heatmap (benchmark+novel × GPS programs)  [F] OTA γ ranking
"""
from __future__ import annotations
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm

# ── palette ───────────────────────────────────────────────────────────────────
C_SELECTED  = "#e05c2a"   # orange-red — selected programs
C_NOVEL     = "#2a7db5"   # blue — novel nominated genes
C_BENCHMARK = "#2e8b57"   # green — GWAS-validated benchmark genes
C_GREY      = "#bbbbbb"
C_FAINT     = "#e0e0e0"

# ── constants ─────────────────────────────────────────────────────────────────
CAD_PROGRAMS   = ["P14", "P43", "P26"]
RA_PROGRAMS    = ["C25", "C01", "C12"]

CAD_BENCHMARKS = ["PLPP3", "NOS3", "COL4A1", "COL4A2", "EXOC3L2", "CALCRL", "LOX"]
CAD_NOVEL      = ["ROCK1", "PLEKHA1", "GIT1", "NPR2", "ELOVL2"]

RA_BENCHMARKS  = ["TYK2", "JAK2", "CTLA4", "IL12RB2", "ICOS", "TRAF3IP2",
                   "CD226", "IL2RA", "PTPN22", "REL", "TRAF1"]
RA_NOVEL       = ["NUGGC", "CRTAM", "MACC1"]

# OTA γ from tier4 checkpoint
CAD_GAMMAS = {
    "PLEKHA1": -0.869, "CCM2": -0.650, "GIT1": -0.623, "NPR2": -0.569,
    "ELOVL2": -0.587, "LOX": +0.462, "ROCK1": -0.447, "NOS3": +0.297,
    "PLPP3": +0.277, "COL4A1": -0.241, "EXOC3L2": -0.217, "CALCRL": -0.214,
    "COL4A2": -0.038,
}
CAD_RANKS = {
    "PLEKHA1": 64, "CCM2": 208, "GIT1": 239, "ELOVL2": 285, "NPR2": 313,
    "LOX": 522, "ROCK1": 551, "NOS3": 945, "PLPP3": 1021, "COL4A1": 1124,
    "EXOC3L2": 1187, "CALCRL": 1198, "COL4A2": 1406,
}

RA_GAMMAS = {
    "IL12RB2": 0.691, "ICOS": 0.661, "NUGGC": 0.477, "CRTAM": 0.398,
    "CD226": 0.381, "TRAF1": 0.354, "TYK2": 0.284, "TRAF3IP2": 0.261,
    "CTLA4": 0.251, "JAK2": 0.244, "MACC1": 0.236, "PTPN22": 0.226,
    "REL": 0.196, "IL2RA": 0.169,
}
RA_RANKS = {
    "IL12RB2": 3, "ICOS": 4, "NUGGC": 8, "CRTAM": 17, "CD226": 22,
    "TRAF1": 32, "TYK2": 83, "TRAF3IP2": 120, "CTLA4": 141, "JAK2": 159,
    "MACC1": 175, "PTPN22": 194, "REL": 248, "IL2RA": 306,
}


# ── data loading ──────────────────────────────────────────────────────────────
def _load_cad_taus() -> dict[str, float]:
    with open("data/ldsc/results/CAD_cNMF_program_taus.json") as f:
        d = json.load(f)
    return {k: float(v) for k, v in d["program_taus"].items()}


def _load_ra_taus() -> dict[str, float]:
    with open("data/ldsc/results/RA_GeneticNMF_Stim48hr_program_taus.json") as f:
        d = json.load(f)
    return {
        k.replace("RA_GeneticNMF_Stim48hr_", ""): float(v)
        for k, v in d["program_taus"].items()
    }


def _load_cad_betas() -> tuple[np.ndarray, list[str], list[str]]:
    npz = np.load("data/perturbseq/schnitzler_cad_vascular/cnmf_mast_betas.npz",
                  allow_pickle=True)
    return npz["beta"], [str(g) for g in npz["ko_genes"]], [str(p) for p in npz["program_ids"]]


def _load_ra_betas() -> tuple[np.ndarray, list[str], list[str]]:
    npz = np.load("data/perturbseq/czi_2025_cd4t_perturb/genetic_nmf_de_betas.npz",
                  allow_pickle=True)
    return npz["beta"], [str(g) for g in npz["ko_genes"]], [str(p) for p in npz["program_ids"]]


# ── panel A / D: program τ* bar chart ────────────────────────────────────────
def _tau_bar(ax: plt.Axes, taus: dict[str, float], selected: list[str], title: str) -> None:
    items = sorted(taus.items(), key=lambda x: x[1], reverse=True)
    programs, vals = zip(*items)
    max_pos = max((v for v in vals if v > 0), default=1.0)
    norm_vals = [v / max_pos for v in vals]

    colors = [C_SELECTED if p in selected else (C_GREY if v > 0 else C_FAINT)
              for p, v in zip(programs, norm_vals)]
    ax.barh(range(len(programs)), norm_vals, color=colors, height=0.7, linewidth=0)

    for i, (p, v) in enumerate(zip(programs, norm_vals)):
        if p in selected:
            ax.text(max(v, 0) + 0.02, i, f"  {p}", va="center", ha="left",
                    fontsize=8, fontweight="bold", color=C_SELECTED)

    ax.set_yticks([])
    ax.set_xlabel("τ* / max(τ*>0)", fontsize=8)
    ax.set_xlim(-0.15, 1.45)
    ax.axvline(0, color="black", lw=0.5)
    ax.set_title(title, fontsize=9, fontweight="bold")

    n_pos = sum(1 for v in vals if v > 0)
    ax.set_ylabel(f"{len(programs)} programs total\n({n_pos} with τ*>0)", fontsize=7)

    sel_patch = mpatches.Patch(color=C_SELECTED, label=f"GPS selected ({len(selected)})")
    pos_patch = mpatches.Patch(color=C_GREY,     label="τ*>0 (heritable)")
    neg_patch = mpatches.Patch(color=C_FAINT,    label="τ*≤0 (excluded)")
    ax.legend(handles=[sel_patch, pos_patch, neg_patch], fontsize=6.5,
              loc="lower right", framealpha=0.8)


# ── panel B / E: β heatmap ───────────────────────────────────────────────────
def _beta_heatmap(ax: plt.Axes, beta: np.ndarray, ko_genes: list[str],
                  prog_ids: list[str], genes_show: list[str], progs_show: list[str],
                  novel_genes: list[str], title: str) -> None:
    g_ok   = [g for g in genes_show if g in ko_genes]
    g_idx  = [ko_genes.index(g) for g in g_ok]
    p_ok   = [p for p in progs_show if p in prog_ids]
    p_idx  = [prog_ids.index(p) for p in p_ok]

    mat = np.clip(beta[np.ix_(g_idx, p_idx)], -3, 3)

    norm = TwoSlopeNorm(vmin=-3, vcenter=0, vmax=3)
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", norm=norm)

    ax.set_xticks(range(len(p_ok)))
    ax.set_xticklabels(p_ok, fontsize=8.5, fontweight="bold")
    ax.set_yticks(range(len(g_ok)))
    ax.set_yticklabels(g_ok, fontsize=7)

    for tick, gene in zip(ax.get_yticklabels(), g_ok):
        if gene in novel_genes:
            tick.set_color(C_NOVEL)
            tick.set_fontweight("bold")
        else:
            tick.set_color(C_BENCHMARK)

    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.03, label="β (z-score, capped ±3)")
    ax.set_title(title, fontsize=9, fontweight="bold")

    # separator
    n_bench = len([g for g in g_ok if g not in novel_genes])
    if 0 < n_bench < len(g_ok):
        ax.axhline(n_bench - 0.5, color="white", lw=2)

    bench_patch = mpatches.Patch(color=C_BENCHMARK, label="GWAS benchmark")
    novel_patch = mpatches.Patch(color=C_NOVEL,     label="Novel (GPS force)")
    ax.legend(handles=[bench_patch, novel_patch], fontsize=6.5,
              loc="upper right", framealpha=0.85)


# ── panel C: CAD OTA γ ranking ────────────────────────────────────────────────
def _gamma_rank_cad(ax: plt.Axes) -> None:
    # merge benchmark + novel, sorted by |gamma|
    all_genes = {**{g: CAD_GAMMAS[g] for g in CAD_BENCHMARKS if g in CAD_GAMMAS},
                 **{g: CAD_GAMMAS[g] for g in CAD_NOVEL if g in CAD_GAMMAS}}
    # sort by gamma (most negative first = strongest atherogenic suppressor KD)
    order = sorted(all_genes.items(), key=lambda x: x[1])
    genes, gammas = zip(*order)

    colors = [C_NOVEL if g in CAD_NOVEL else C_BENCHMARK for g in genes]
    bars = ax.barh(range(len(genes)), gammas, color=colors, height=0.65, linewidth=0)

    ax.set_yticks(range(len(genes)))
    ax.set_yticklabels(
        [f"{g}  (#{CAD_RANKS.get(g,'?')})" for g in genes],
        fontsize=7.5,
    )
    for tick, gene in zip(ax.get_yticklabels(), genes):
        tick.set_color(C_NOVEL if gene in CAD_NOVEL else C_BENCHMARK)
        if gene in CAD_NOVEL:
            tick.set_fontweight("bold")

    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("OTA γ  (γ<0: KD↓risk; γ>0: KD↑risk)", fontsize=8)
    ax.set_title("C  CAD: OTA γ ranking\nbenchmarks + novel GPS genes", fontsize=9, fontweight="bold")

    bench_patch = mpatches.Patch(color=C_BENCHMARK, label="Schnitzler benchmark")
    novel_patch = mpatches.Patch(color=C_NOVEL,     label="Novel (Criterion A: cluster)")
    ax.legend(handles=[bench_patch, novel_patch], fontsize=6.5, framealpha=0.85)

    # annotate direction labels
    ax.text(-0.05, len(genes) - 0.5, "← protective KD", ha="right", va="top",
            fontsize=7, color="#555555", style="italic")
    ax.text(+0.05, len(genes) - 0.5, "atherogenic KD →", ha="left", va="top",
            fontsize=7, color="#555555", style="italic")


# ── panel F: RA OTA γ ranking ─────────────────────────────────────────────────
def _gamma_rank_ra(ax: plt.Axes) -> None:
    all_genes = {**{g: RA_GAMMAS[g] for g in RA_BENCHMARKS if g in RA_GAMMAS},
                 **{g: RA_GAMMAS[g] for g in RA_NOVEL if g in RA_GAMMAS}}
    # sort by gamma descending (highest = most heritable program amplification on KO)
    order = sorted(all_genes.items(), key=lambda x: -x[1])
    genes, gammas = zip(*order)

    colors = [C_NOVEL if g in RA_NOVEL else C_BENCHMARK for g in genes]
    ax.barh(range(len(genes)), gammas, color=colors, height=0.65, linewidth=0)

    ax.set_yticks(range(len(genes)))
    ax.set_yticklabels(
        [f"{g}  (#{RA_RANKS.get(g,'?')})" for g in genes],
        fontsize=7.5,
    )
    ax.invert_yaxis()
    for tick, gene in zip(ax.get_yticklabels(), genes):
        tick.set_color(C_NOVEL if gene in RA_NOVEL else C_BENCHMARK)
        if gene in RA_NOVEL:
            tick.set_fontweight("bold")

    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("OTA γ  (all γ>0: GeneticNMF τ* all positive)", fontsize=8)
    ax.set_title("F  RA: OTA γ ranking\nbenchmarks + novel GPS genes", fontsize=9, fontweight="bold")

    bench_patch = mpatches.Patch(color=C_BENCHMARK, label="CD4+ T benchmark")
    novel_patch = mpatches.Patch(color=C_NOVEL,     label="Novel (Criterion B: cosine sim)")
    ax.legend(handles=[bench_patch, novel_patch], fontsize=6.5, framealpha=0.85)


# ── annotation box ────────────────────────────────────────────────────────────
def _box(fig: plt.Figure, ax: plt.Axes, text: str) -> None:
    ax.annotate(
        text,
        xy=(0.5, -0.26), xycoords="axes fraction",
        ha="center", va="top", fontsize=6.5,
        bbox=dict(boxstyle="round,pad=0.35", fc="#f7f7f7", ec="#cccccc", lw=0.8),
    )


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    cad_taus                     = _load_cad_taus()
    ra_taus                      = _load_ra_taus()
    cad_beta, cad_genes, cad_progs = _load_cad_betas()
    ra_beta,  ra_genes,  ra_progs  = _load_ra_betas()

    fig, axes = plt.subplots(
        2, 3,
        figsize=(16, 11),
        gridspec_kw={"wspace": 0.42, "hspace": 0.52},
    )
    fig.suptitle(
        "GPS gene and program selection rationale\n"
        "CAD (cardiac endothelial · Schnitzler 2024 · cNMF k=60)   ·   "
        "RA (CD4⁺ T · CZI 2025 · GeneticNMF Stim48hr k=30)",
        fontsize=11, fontweight="bold", y=0.99,
    )

    # ── Row 1: CAD ────────────────────────────────────────────────────────────
    _tau_bar(axes[0, 0], cad_taus, CAD_PROGRAMS,
             "A  CAD program heritability (S-LDSC τ*)\n31 programs with τ*>0")

    cad_genes_show = [g for g in CAD_BENCHMARKS + CAD_NOVEL if g in cad_genes]
    _beta_heatmap(axes[0, 1], cad_beta, cad_genes, cad_progs,
                  cad_genes_show, CAD_PROGRAMS, CAD_NOVEL,
                  "B  CAD β loadings on selected programs\nbenchmarks (green) + novel (blue)")

    _gamma_rank_cad(axes[0, 2])

    # ── Row 2: RA ─────────────────────────────────────────────────────────────
    _tau_bar(axes[1, 0], ra_taus, RA_PROGRAMS,
             "D  RA program heritability (S-LDSC τ*)\n8 programs with τ*>0")

    ra_genes_show = [g for g in RA_BENCHMARKS + RA_NOVEL if g in ra_genes]
    _beta_heatmap(axes[1, 1], ra_beta, ra_genes, ra_progs,
                  ra_genes_show, RA_PROGRAMS, RA_NOVEL,
                  "E  RA β loadings on selected programs\nbenchmarks (green) + novel (blue)")

    _gamma_rank_ra(axes[1, 2])

    # ── annotation boxes ─────────────────────────────────────────────────────
    _box(fig, axes[0, 0],
         "Criteria: (1) τ* rank  (2) axis independence  (3) benchmark gene coverage\n"
         "Selected: P14 (τ*=0.092), P43 (0.083), P26 (0.075)")
    _box(fig, axes[0, 1],
         "β = Schnitzler CRISPRi MAST z-scores projected onto cNMF Vt\n"
         "Red = KD ↓ program loading; Blue = KD ↑ program loading")
    _box(fig, axes[0, 2],
         "Novel genes (all γ<0, KD↓risk, no GWAS): ROCK1+PLEKHA1 (COL4A1 cluster) · GIT1+NPR2 (COL4A2/FN1) · ELOVL2 (LOX cluster)\n"
         "Selected as inhibitable ECM-remodelling proteins; Criterion A: τ*-weighted cluster co-membership")
    _box(fig, axes[1, 0],
         "Criteria: (1) τ* rank  (2) axis independence  (3) benchmark gene coverage\n"
         "Selected: C25 (τ*_norm=1.000), C01 (0.543), C12 (0.195)")
    _box(fig, axes[1, 1],
         "β = CZI CD4⁺ T GeneticNMF Stim48hr DESeq2 z-scores (11,281 KOs × 30 programs)\n"
         "Red = KD ↓ program loading; Blue = KD ↑ program loading")
    _box(fig, axes[1, 2],
         "Novel genes: no GWAS signal, no eQTL at GWAS locus, no WES burden\n"
         "Criterion B: cosine sim of heritable-program β vector to mean benchmark vector")

    out = "outputs/gps_selection_rationale.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
