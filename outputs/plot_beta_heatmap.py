"""
β-program heatmap for CAD/RA Perturb-seq runs.

What this plot shows
--------------------
Rows  = gene KOs ranked by |ota_gamma| (top N, default 300)
Cols  = cNMF/GeneticNMF programs with τ* > 0 (heritable programs only)
Color = MAST β: red = KO activates program; blue = KO suppresses it

Both rows and columns are hierarchically clustered by Pearson correlation of
the β matrix.  The bar above the heatmap shows τ* (S-LDSC heritability
enrichment) for each program in its clustered position.

How to interpret it
-------------------
1. Column clusters = groups of programs that tend to be co-regulated by the
   same gene KOs.  A tight cluster of high-τ programs means any gene that
   hits one hits all — these are the dominant heritability modules.

2. Row clusters = gene groups with similar β profiles.  Genes in the same
   row cluster drive (or suppress) the same set of programs.  This is the
   ground truth for "which genes belong together" and should inform how we
   assign genes to panels in the long-island plot.

3. Benchmark genes (labeled) should appear in row clusters with coherent β
   sign on the high-τ programs (left-most columns after clustering have the
   highest τ*, since we sort by τ* within each cluster).  If a benchmark
   gene is isolated from other high-β genes, that is a signal it drives a
   program idiosyncratically.

4. Winner-takes-all caveat: most genes have β spread across several programs.
   A gene assigned to panel P14 by max |β×τ*| may also have meaningful β on
   P19, P26, etc.  The heatmap makes this visible.  Use it to decide whether
   a threshold-based multi-panel assignment (gene appears in all panels where
   |β×τ*| > k) would be more informative than strict winner-takes-all.

Saved to
--------
  outputs/beta_heatmap_{disease_key}.png

Usage
-----
    python -m outputs.plot_beta_heatmap --disease cad
    python -m outputs.plot_beta_heatmap --disease ra

    # or programmatically:
    from outputs.plot_beta_heatmap import plot_beta_heatmap
    plot_beta_heatmap("cad")
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib

import numpy as np

# ── disease config ─────────────────────────────────────────────────────────────
_DISEASE_CFG: dict[str, dict] = {
    "cad": {
        "beta_npz":   "data/perturbseq/schnitzler_cad_vascular/cnmf_mast_betas.npz",
        "tau_json":   "data/ldsc/results/CAD_cNMF_program_taus.json",
        "tau_key":    "program_taus",
        "checkpoint": "data/checkpoints/coronary_artery_disease__tier4.json",
        "title":      "CAD — Schnitzler cNMF MAST β",
    },
    "ra": {
        "beta_npz":     "data/perturbseq/czi_2025_cd4t_perturb/genetic_nmf_de_betas.npz",
        "tau_json":     "data/ldsc/results/RA_GeneticNMF_Stim48hr_program_taus.json",
        "tau_key":      "program_taus",
        "tau_prefix":   "RA_GeneticNMF_Stim48hr_",
        "min_tau_frac": 0.0,
        "checkpoint":   "data/checkpoints/rheumatoid_arthritis__tier4.json",
        "title":        "RA — CZI GeneticNMF β (Stim48hr)",
    },
}

# Benchmark gene annotations loaded from disease_registry
def _load_bench(disease_key: str) -> tuple[set[str], set[str]]:
    try:
        import sys as _sys
        _root = pathlib.Path(__file__).parent.parent
        if str(_root) not in _sys.path:
            _sys.path.insert(0, str(_root))
        from config.drug_target_registry import get_validated_targets
        vt = get_validated_targets(disease_key.upper())
        up   = {t["gene"] for t in vt if t.get("expected_sign") == -1}
        down = {t["gene"] for t in vt if t.get("expected_sign") == +1}
        return up, down
    except Exception:
        return set(), set()


def plot_beta_heatmap(
    disease_key: str,
    top_n: int = 950,
    min_tau_frac: float | None = None,
    cluster_by: str = "ota",
    out_path: str | None = None,
    gene_subset: list[str] | None = None,
) -> str:
    """
    Generate and save the β-program cluster heatmap.

    Parameters
    ----------
    disease_key  : "cad" or "ra"
    top_n        : number of top-|ota_gamma| genes to show (benchmark genes always included).
                   Ignored when gene_subset is provided.
    min_tau_frac : minimum τ* as a fraction of the max τ* across all programs.
                   Programs below this threshold are excluded before clustering — they
                   have negligible heritability enrichment and add noise to the row
                   clustering without contributing signal to program assignment.
                   None = use per-disease config default (CAD: 0.20, RA: 0.0 to
                   include all 8 GeneticNMF programs with τ*>0).
    cluster_by   : "ota"  — cluster rows and cols by β×τ* (OTA contribution vector).
                            Genetically correct: programs with low τ* have low weight,
                            so only heritability-enriched programs drive the clustering.
                            This groups genes by their shared genetic mechanism.
                 : "beta" — cluster by raw β (legacy; treats all programs equally).
    out_path     : override output path; defaults to outputs/beta_heatmap_{disease_key}.png
    gene_subset  : if provided, restrict rows to exactly these genes (skips top_n selection).
                   Useful for zooming in on a specific cluster.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
    from scipy.spatial.distance import pdist

    dk  = disease_key.lower()
    cfg = _DISEASE_CFG[dk]
    if min_tau_frac is None:
        min_tau_frac = cfg.get("min_tau_frac", 0.20)

    # ── Load β matrix ──────────────────────────────────────────────────────────
    npz      = np.load(cfg["beta_npz"], allow_pickle=True)
    beta_full = npz["beta"]                       # (n_ko, n_prog)
    ko_genes  = list(npz.get("ko_genes", npz.get("gene_names", [])))
    prog_ids  = list(npz.get("program_ids", npz.get("program_names", [])))

    # ── Load OTA gammas + taus ─────────────────────────────────────────────────
    ckpt     = json.loads(pathlib.Path(cfg["checkpoint"]).read_text())
    targets  = ckpt["prioritization_result"]["targets"]
    ota_gamma = {t["target_gene"]: t.get("ota_gamma", 0.0) for t in targets}

    tau_raw = json.loads(pathlib.Path(cfg["tau_json"]).read_text())
    tau_map: dict[str, float] = tau_raw.get(cfg["tau_key"], {})
    # Strip disease-specific prefix so bare program IDs (e.g. "C01") match tau keys
    tau_prefix = cfg.get("tau_prefix", "")
    if tau_prefix:
        tau_map = {
            (k[len(tau_prefix):] if k.startswith(tau_prefix) else k): v
            for k, v in tau_map.items()
        }

    bench_up, bench_down = _load_bench(dk)
    bench_all = bench_up | bench_down

    # ── Select genes: explicit subset OR top_n by |ota_gamma| + always benchmark ─
    if gene_subset is not None:
        gene_list = [g for g in gene_subset if g in ko_genes]
    else:
        ranked    = sorted(ota_gamma.items(), key=lambda x: -abs(x[1]))
        top_genes = [g for g, _ in ranked[:top_n]]
        for g in bench_all:
            if g in ota_gamma and g not in top_genes:
                top_genes.append(g)
        gene_list = [g for g in top_genes if g in ko_genes]

    # ── Select programs: τ* ≥ min_tau_frac × max(τ*) ─────────────────────────
    # Low-τ programs add noise to clustering without contributing OTA signal.
    # We filter to programs whose heritability enrichment is at least min_tau_frac
    # of the strongest program, keeping only the meaningful heritability modules.
    all_taus   = [tau_map.get(p, 0.0) for p in prog_ids]
    tau_max    = max(all_taus) if all_taus else 0.0
    tau_thresh = tau_max * min_tau_frac
    heritable_progs = [p for p in prog_ids if tau_map.get(p, 0.0) >= tau_thresh]
    if not heritable_progs:
        print(f"  No heritable programs found for {dk} — check tau_json path")
        return ""
    print(f"  Programs with τ* ≥ {min_tau_frac:.0%} of max ({tau_thresh:.4f}): "
          f"{len(heritable_progs)} / {len(prog_ids)}")

    prog_idx  = [prog_ids.index(p) for p in heritable_progs]
    gene_ridx = [ko_genes.index(g) for g in gene_list]
    beta      = beta_full[np.ix_(gene_ridx, prog_idx)]   # (n_genes, n_heritable_progs)

    print(f"  Heatmap: {beta.shape[0]} genes × {beta.shape[1]} heritable programs")

    # ── Normalize β before clustering ─────────────────────────────────────────
    # Row-wise z-score: each gene's β vector is centred and scaled to unit std
    # across programs before computing distances.  This clusters by *pattern*
    # (which programs does this KO activate vs suppress relative to its own
    # mean effect) rather than by magnitude.  Without normalisation, a gene
    # with globally large β dominates its cluster regardless of pattern shape.
    row_mean = beta.mean(axis=1, keepdims=True)
    row_std  = beta.std(axis=1, keepdims=True)
    row_std[row_std == 0] = 1.0                   # avoid div-by-zero for flat rows
    beta_norm = (beta - row_mean) / row_std       # z-scored across programs

    # Drop genes that are flat across all selected programs — their row z-score is
    # identically 0, producing NaN correlation distances that crash linkage.
    _nonflat = beta.std(axis=1) > 1e-8
    if not _nonflat.all():
        n_dropped = int((~_nonflat).sum())
        print(f"  Dropped {n_dropped} flat-row genes (zero variance across selected programs)")
        beta      = beta[_nonflat]
        beta_norm = beta_norm[_nonflat]
        gene_list = [g for g, k in zip(gene_list, _nonflat) if k]

    tau_vec = np.array([tau_map.get(p, 0.0) for p in heritable_progs])   # (n_progs,)
    if cluster_by == "ota":
        beta_clust_mat = beta_norm * tau_vec[np.newaxis, :]
        print(f"  Clustering by z-scored β × τ*")
    else:
        beta_clust_mat = beta_norm
        print(f"  Clustering by z-scored β (pattern, not magnitude)")

    col_link  = linkage(pdist(beta_clust_mat.T, "correlation"), method="average")
    row_link  = linkage(pdist(beta_clust_mat,   "correlation"), method="average")
    col_order = leaves_list(col_link)
    row_order = leaves_list(row_link)

    beta_c      = beta[np.ix_(row_order, col_order)]
    beta_norm_c = beta_norm[np.ix_(row_order, col_order)]   # z-scored for display
    genes_c = [gene_list[i] for i in row_order]
    progs_c = [heritable_progs[j] for j in col_order]
    tau_c   = np.array([tau_map.get(p, 0.0) for p in progs_c])

    n_genes = len(genes_c)
    n_progs = len(progs_c)

    # ── Figure layout ──────────────────────────────────────────────────────────
    fig_h = max(10, n_genes * 0.038)   # ~36px per row at 160dpi keeps labels readable
    fig   = plt.figure(figsize=(14, fig_h))
    gs    = fig.add_gridspec(
        3, 3,
        height_ratios=[0.06, 0.10, 1.0],
        width_ratios=[0.22, 0.03, 1.0],
        hspace=0.02, wspace=0.02,
    )

    # Row dendrogram
    ax_rdend = fig.add_subplot(gs[2, 0])
    dendrogram(row_link, ax=ax_rdend, orientation="left", color_threshold=0,
               above_threshold_color="#888", link_color_func=lambda k: "#888",
               no_labels=True)
    ax_rdend.set_axis_off()

    # Row color strip (benchmark annotation)
    # imshow row 0 = top → invert y so strip matches heatmap row order
    ax_rstrip = fig.add_subplot(gs[2, 1])
    for i, g in enumerate(genes_c):
        if g in bench_up:
            ax_rstrip.barh(i, 1, color="#D32F2F", height=1.0)
        elif g in bench_down:
            ax_rstrip.barh(i, 1, color="#1565C0", height=1.0)
    ax_rstrip.set_xlim(0, 1)
    ax_rstrip.set_ylim(n_genes - 0.5, -0.5)   # inverted: row 0 at top
    ax_rstrip.set_axis_off()

    # τ* bar (above heatmap)
    ax_tau = fig.add_subplot(gs[0, 2])
    ax_tau.bar(np.arange(n_progs), tau_c, color="#1565C0", width=0.8)
    ax_tau.set_xlim(-0.5, n_progs - 0.5)
    if tau_c.max() > 0:
        ax_tau.set_yticks([0, tau_c.max()])
        ax_tau.set_yticklabels(["0", f"{tau_c.max():.3f}"], fontsize=6)
    ax_tau.set_ylabel("τ*", fontsize=7, rotation=0, labelpad=10)
    ax_tau.set_xticks([])
    ax_tau.spines[["top", "right", "bottom"]].set_visible(False)

    # Column dendrogram
    ax_cdend = fig.add_subplot(gs[1, 2])
    dendrogram(col_link, ax=ax_cdend, orientation="top", color_threshold=0,
               above_threshold_color="#888", link_color_func=lambda k: "#888",
               no_labels=True)
    ax_cdend.set_axis_off()

    # Heatmap — display row-z-scored β so pattern is visible regardless of raw scale.
    # vmax capped at 3 z-scores; extreme outliers (self-locus effects) don't wash out structure.
    ax_h = fig.add_subplot(gs[2, 2])
    vmax = min(np.percentile(np.abs(beta_norm_c), 97), 3.0)
    im = ax_h.imshow(beta_norm_c, aspect="auto", cmap="RdBu_r",
                     vmin=-vmax, vmax=vmax, interpolation="none")

    ax_h.set_xticks(np.arange(n_progs))
    ax_h.set_xticklabels(
        [f"{p}  τ={tau_map.get(p, 0.0):.3f}" for p in progs_c],
        fontsize=5.5, rotation=90,
    )

    # Benchmark gene labels: placed as text annotations on the right edge of the
    # heatmap so they sit at the exact pixel row of the gene, not as ytick labels
    # (which can shift when matplotlib auto-spaces ticks).
    ax_h.set_yticks([])
    for i, g in enumerate(genes_c):
        if g in bench_all:
            color = "#D32F2F" if g in bench_up else "#1565C0"
            ax_h.annotate(
                g,
                xy=(1.01, i),
                xycoords=("axes fraction", "data"),
                fontsize=7, fontweight="bold", color=color,
                va="center", ha="left", annotation_clip=False,
            )

    ax_h.set_ylabel(f"top-{n_genes} OTA gene KOs by |γ|  (clustered by β-profile)", fontsize=8)
    prog_label = (
        "GeneticNMF programs (τ* > 0)  ·  OTA β source  ·  clustered by β-correlation"
        if dk == "ra"
        else "cNMF programs (τ* > 0)  ·  clustered by β-correlation"
    )
    ax_h.set_xlabel(prog_label, fontsize=8)

    cbar = fig.colorbar(im, ax=ax_h, fraction=0.012, pad=0.01)
    cbar.set_label("β (row-z-scored, KO → program)", fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    patches = [
        mpatches.Patch(color="#D32F2F", label="KD increases risk (atheroprotective)"),
        mpatches.Patch(color="#1565C0", label="KD reduces risk (atherogenic driver)"),
    ]
    ax_h.legend(handles=patches, loc="upper left",
                bbox_to_anchor=(-0.3, 1.08), fontsize=7, frameon=True)

    n_row_clusters = _estimate_clusters(row_link, n_genes)
    n_col_clusters = _estimate_clusters(col_link, n_progs)
    title_note = ""
    fig.suptitle(
        f"{cfg['title']}{title_note}\n"
        f"{n_genes} gene KOs × {n_progs} heritable programs  ·  "
        f"~{n_row_clusters} gene clusters, ~{n_col_clusters} program clusters",
        fontsize=9, y=1.01,
    )

    dest = out_path or f"outputs/beta_heatmap_{dk}.png"
    plt.savefig(dest, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Beta heatmap saved → {dest}")
    return dest


def _estimate_clusters(link: np.ndarray, n: int, frac: float = 0.3) -> int:
    """Rough cluster count at 30% of max linkage height."""
    from scipy.cluster.hierarchy import fcluster
    threshold = link[:, 2].max() * frac
    return int(len(set(fcluster(link, t=threshold, criterion="distance"))))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="β-program cluster heatmap")
    ap.add_argument("--disease",       choices=["cad", "ra"], required=True)
    ap.add_argument("--top-n",         type=int,   default=950,  help="top N genes by |ota_gamma|")
    ap.add_argument("--min-tau-frac",  type=float, default=0.20,  help="min τ* as fraction of max (default 0.20)")
    ap.add_argument("--cluster-by",    default="ota", choices=["ota", "beta"],
                    help="cluster by β×τ* (ota, default) or raw β (beta)")
    ap.add_argument("--out",           type=str,   default=None)
    args = ap.parse_args()
    plot_beta_heatmap(args.disease, top_n=args.top_n,
                      min_tau_frac=args.min_tau_frac,
                      cluster_by=args.cluster_by,
                      out_path=args.out)
