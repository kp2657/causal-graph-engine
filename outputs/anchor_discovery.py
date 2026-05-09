"""
De novo anchor discovery for CAD and RA Perturb-seq atlases.

An "anchor" is a GWAS-library gene that sits at the centre of a coherent
β-cluster.  Knowing the anchor tells you which GWAS locus drives the cluster's
biological programme.

Algorithm
---------
1. Restrict to top-N genes by |OTA γ| (same subset used in beta_heatmap).
   Force-include all GWAS-library genes so none are dropped.
2. Cluster their z-scored β vectors (same linkage as beta_heatmap).
3. For each cluster compute:
     gwas_frac  = fraction of genes from the Schnitzler/CZI GWAS library
                  (genes WITHOUT the perturb_novel flag — selected because they
                  overlap a disease GWAS locus)
     mean_gamma = mean |OTA γ| across cluster genes
     silhouette = mean silhouette score (cluster tightness in z-β space)
     anchor_score = gwas_frac × mean_gamma × max(silhouette, 0)
4. Within each cluster identify:
     centroid_gene  = closest to cluster mean in z-β space (most representative)
     top_gwas_gene  = GWAS-library gene with highest |OTA γ| (the de novo anchor)

Usage
-----
    python -m outputs.anchor_discovery --disease cad
    python -m outputs.anchor_discovery --disease ra --top-n 500

    from outputs.anchor_discovery import rank_denovo_anchors, get_anchor_cluster
    clusters = rank_denovo_anchors("cad")          # → list of cluster dicts
    genes    = get_anchor_cluster("cad", "COL4A2") # → 71-gene list for heatmap
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).parent.parent

_DISEASE_CFG: dict[str, dict] = {
    "cad": {
        "beta_npz":   "data/perturbseq/schnitzler_cad_vascular/cnmf_mast_betas.npz",
        "tau_json":   "data/ldsc/results/CAD_cNMF_program_taus.json",
        "tau_key":    "program_taus",
        "checkpoint": "data/checkpoints/coronary_artery_disease__tier4.json",
    },
    "ra": {
        # GeneticNMF β matrix — same source as RA OTA (DESeq2 z-scores projected onto Vt).
        # Program IDs are bare "C01"…"C30"; tau_prefix strips the JSON key prefix.
        # min_tau_frac=0.0: use all 8 programs with τ*>0 (taus drop steeply from C25=1.0
        # to C28=0.005; a 20% floor would keep only C25+C01).
        "beta_npz":     "data/perturbseq/czi_2025_cd4t_perturb/genetic_nmf_de_betas.npz",
        "tau_json":     "data/ldsc/results/RA_GeneticNMF_Stim48hr_program_taus.json",
        "tau_key":      "program_taus",
        "tau_prefix":   "RA_GeneticNMF_Stim48hr_",
        "min_tau_frac": 0.0,
        "checkpoint":   "data/checkpoints/rheumatoid_arthritis__tier4.json",
    },
}


def _load(disease_key: str):
    dk  = disease_key.lower()
    cfg = _DISEASE_CFG[dk]

    npz       = np.load(ROOT / cfg["beta_npz"], allow_pickle=True)
    beta_full = npz["beta"]
    ko_genes  = list(npz.get("ko_genes", npz.get("gene_names", [])))
    prog_ids  = list(npz.get("program_ids", npz.get("program_names", [])))

    tau_raw  = json.loads((ROOT / cfg["tau_json"]).read_text())
    tau_map: dict[str, float] = tau_raw.get(cfg["tau_key"], {})
    tau_prefix = cfg.get("tau_prefix", "")
    if tau_prefix:
        tau_map = {
            (k[len(tau_prefix):] if k.startswith(tau_prefix) else k): v
            for k, v in tau_map.items()
        }

    ckpt    = json.loads((ROOT / cfg["checkpoint"]).read_text())
    targets = ckpt["prioritization_result"]["targets"]
    ota_map = {t["target_gene"]: t.get("ota_gamma", 0.0) for t in targets}

    # GWAS-library genes: those WITHOUT the perturb_novel flag.
    # These were selected by the study's authors because they overlap a disease GWAS locus.
    gwas_lib: set[str] = {
        t["target_gene"]
        for t in targets
        if "perturb_novel" not in t.get("flags", [])
    }

    return beta_full, ko_genes, prog_ids, tau_map, ota_map, gwas_lib


def _build_clustering(
    beta_full, ko_genes, prog_ids, tau_map, ota_map, gwas_lib,
    top_n: int, min_tau_frac: float,
):
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import pdist

    # Heritable programs: those with τ* ≥ min_tau_frac × max(τ*), matched by key.
    tau_max    = max(tau_map.values()) if tau_map else 0.0
    tau_thresh = tau_max * min_tau_frac
    heritable  = [p for p in prog_ids if tau_map.get(p, 0.0) >= tau_thresh]
    if not heritable:
        heritable = list(prog_ids)
        print(f"  No τ* keys matched prog_ids — clustering on all {len(prog_ids)} programs "
              f"(expected for RA: LOCUS τ* don't map to GeneticNMF C01-C30)")
    else:
        print(f"  Programs with τ* ≥ {min_tau_frac:.0%} of max: "
              f"{len(heritable)} / {len(prog_ids)}")
    prog_idx_l = [prog_ids.index(p) for p in heritable]

    ranked    = sorted(ota_map.items(), key=lambda x: -abs(x[1]))
    top_genes = [g for g, _ in ranked[:top_n]]
    # Force all GWAS-library genes in so they are always clustered
    for g in gwas_lib:
        if g not in top_genes and g in ota_map:
            top_genes.append(g)
    gene_list = [g for g in top_genes if g in ko_genes]
    gene_ridx = [ko_genes.index(g) for g in gene_list]

    beta = beta_full[np.ix_(gene_ridx, prog_idx_l)]
    row_mean = beta.mean(axis=1, keepdims=True)
    row_std  = beta.std(axis=1, keepdims=True)
    # Genes with zero variance across all programs cluster by their flat profile —
    # replace zero std with 1 to keep them as all-zero z-score rows (they will
    # cluster together as a "no effect" group, away from the interesting clusters).
    row_std[row_std == 0] = 1.0
    beta_z = (beta - row_mean) / row_std

    # Drop genes with non-finite z-scores or all-zero z-score vectors.
    # All-zero z-score rows arise when β = 0 across all selected programs —
    # their Pearson correlation with any other vector is undefined (NaN in pdist).
    valid = np.isfinite(beta_z).all(axis=1) & (beta_z != 0).any(axis=1)
    if not valid.all():
        n_dropped = int((~valid).sum())
        print(f"  Dropping {n_dropped} genes with flat/non-finite β across selected programs")
        beta_z    = beta_z[valid]
        gene_list = [gene_list[i] for i, v in enumerate(valid) if v]

    # Weight clustering by τ* so genes that share loading on heritable programs
    # (not just any program) end up in the same cluster.
    tau_vec    = np.array([tau_map.get(p, 0.0) for p in heritable])
    tau_norm   = tau_vec / tau_vec.max() if tau_vec.max() > 0 else tau_vec
    beta_clust = beta_z * tau_norm[np.newaxis, :]
    row_link   = linkage(pdist(beta_clust, "correlation"), method="average")
    return gene_list, beta_z, beta_clust, row_link, heritable


def rank_denovo_anchors(
    disease_key: str,
    top_n: int = 950,
    k: int = 40,
    min_tau_frac: float | None = None,
    min_n_genes: int = 5,
) -> list[dict]:
    """
    Score each dendrogram cluster by genetic grounding and return ranked results.

    Parameters
    ----------
    disease_key  : "cad" or "ra"
    top_n        : genes to include (top by |OTA γ|); all GWAS-library genes forced in
    k            : number of flat clusters to cut from the dendrogram
    min_tau_frac : heritable program filter — defaults to per-disease config value
                   (CAD: 0.20, RA: 0.0 to include all 8 GeneticNMF τ*>0 programs)
    min_n_genes  : minimum cluster size to include in results; filters singleton /
                   tiny clusters that score artificially high on silhouette

    Returns
    -------
    List of cluster dicts sorted by anchor_score descending.  Each dict has:
      cluster_id, n_genes, n_gwas, anchor_score, gwas_frac, mean_gamma,
      mean_sil, centroid_gene, top_gwas_gene, gwas_genes, genes
    """
    from scipy.cluster.hierarchy import fcluster
    from sklearn.metrics import silhouette_samples

    dk = disease_key.lower()
    if min_tau_frac is None:
        min_tau_frac = _DISEASE_CFG[dk].get("min_tau_frac", 0.20)

    beta_full, ko_genes, prog_ids, tau_map, ota_map, gwas_lib = _load(disease_key)
    gene_list, beta_z, beta_clust, row_link, _ = _build_clustering(
        beta_full, ko_genes, prog_ids, tau_map, ota_map, gwas_lib,
        top_n=top_n, min_tau_frac=min_tau_frac,
    )

    labels   = fcluster(row_link, t=k, criterion="maxclust")
    sil_vals = silhouette_samples(beta_clust, labels, metric="correlation")

    clusters: dict[int, list[int]] = {}
    for i, cid in enumerate(labels):
        clusters.setdefault(cid, []).append(i)

    results = []
    for cid, idxs in clusters.items():
        genes_in   = [gene_list[i] for i in idxs]
        gammas     = [abs(ota_map.get(g, 0.0)) for g in genes_in]
        gwas_in    = [g for g in genes_in if g in gwas_lib]
        gwas_frac  = len(gwas_in) / len(genes_in)
        mean_gamma = float(np.mean(gammas))
        mean_sil   = float(np.mean([sil_vals[i] for i in idxs]))

        sub_beta = beta_z[idxs]
        centroid  = sub_beta.mean(axis=0)
        dists     = np.linalg.norm(sub_beta - centroid, axis=1)
        centroid_gene = genes_in[int(np.argmin(dists))]

        # De novo anchor: GWAS-library gene with highest |OTA γ| in cluster
        top_gwas_gene = max(
            gwas_in,
            key=lambda g: abs(ota_map.get(g, 0.0)),
            default=centroid_gene,
        )

        anchor_score = gwas_frac * mean_gamma * max(mean_sil, 0.0)
        direction    = annotate_cluster_direction(genes_in, disease_key)

        results.append({
            "cluster_id":    int(cid),
            "n_genes":       int(len(genes_in)),
            "n_gwas":        int(len(gwas_in)),
            "anchor_score":  round(anchor_score, 5),
            "gwas_frac":     round(gwas_frac, 3),
            "mean_gamma":    round(mean_gamma, 4),
            "mean_sil":      round(mean_sil, 4),
            "centroid_gene": centroid_gene,
            "top_gwas_gene": top_gwas_gene,
            "gwas_genes":    sorted(gwas_in),
            "genes":         sorted(genes_in),
            **direction,
        })

    results = [r for r in results if r["n_genes"] >= min_n_genes]
    results.sort(key=lambda x: -x["anchor_score"])
    return results


def _benchmark_direction_map(disease_key: str) -> dict[str, int]:
    """
    Return {gene: expected_sign} for all benchmark genes for this disease.

    expected_sign (Schnitzler KD phenotype convention):
      CAD: -1 = KD reduces risk (atherogenic driver / blue in Schnitzler Fig 2c)
           +1 = KD increases risk (atheroprotective / red in Schnitzler Fig 2c)
      RA:  -1 = gene drives disease, KD is protective (JAK2/TYK2 inhibitors)
           +1 = gene is protective, KD amplifies disease (CTLA4)
    """
    from config.drug_target_registry import VALIDATED_DRUG_TARGETS
    dk = disease_key.upper()
    return {
        b["gene"]: b["expected_sign"]
        for b in VALIDATED_DRUG_TARGETS.get(dk, [])
        if "expected_sign" in b
    }


def annotate_cluster_direction(cluster_genes: list[str], disease_key: str) -> dict:
    """
    Annotate a cluster with its KD direction based on benchmark evidence.

    Sign convention (both diseases): expected_sign=-1 means KD reduces risk (therapeutic).
      CAD: -1 = KD reduces risk (atherogenic driver, Schnitzler blue)
           +1 = KD increases risk (atheroprotective, Schnitzler red)
      RA:  -1 = gene drives disease, KD protective (JAK2/TYK2 inhibitors)
           +1 = gene protective, KD amplifies disease (CTLA4)

    Returns a dict with:
      kd_direction   : "kd_reduces_risk" | "kd_increases_risk" | "mixed" | "novel"
      n_kd_reduces   : count of benchmark genes whose KD is therapeutic
      n_kd_increases : count of benchmark genes whose KD worsens disease
      benchmark_hits : list of (gene, expected_sign) in cluster
    """
    bmap = _benchmark_direction_map(disease_key)
    hits = [(g, bmap[g]) for g in cluster_genes if g in bmap]
    # expected_sign=-1 = KD reduces risk (therapeutic) for both CAD and RA
    therapeutic_sign = -1
    n_reduces   = sum(1 for _, s in hits if s == therapeutic_sign)
    n_increases = sum(1 for _, s in hits if s != therapeutic_sign and s != 0)
    if n_reduces == 0 and n_increases == 0:
        direction = "novel"
    elif n_reduces > 0 and n_increases == 0:
        direction = "kd_reduces_risk"
    elif n_increases > 0 and n_reduces == 0:
        direction = "kd_increases_risk"
    else:
        direction = "mixed"
    return {
        "kd_direction":   direction,
        "n_kd_reduces":   n_reduces,
        "n_kd_increases": n_increases,
        "benchmark_hits": hits,
    }


def get_anchor_cluster(
    disease_key: str,
    anchor_gene: str,
    top_n: int = 950,
    k: int = 40,
    min_tau_frac: float | None = None,
) -> list[str]:
    """
    Return the list of genes in the same dendrogram cluster as anchor_gene.

    Useful for driving `plot_beta_heatmap(..., gene_subset=genes)` to zoom
    into the cluster around a gene of interest.
    """
    from scipy.cluster.hierarchy import fcluster

    dk = disease_key.lower()
    if min_tau_frac is None:
        min_tau_frac = _DISEASE_CFG[dk].get("min_tau_frac", 0.20)

    beta_full, ko_genes, prog_ids, tau_map, ota_map, gwas_lib = _load(disease_key)

    # Force anchor gene into the clustering gene list
    extra = {anchor_gene} if anchor_gene in ota_map else set()
    gwas_lib_aug = gwas_lib | extra

    gene_list, beta_z, beta_clust, row_link, _ = _build_clustering(
        beta_full, ko_genes, prog_ids, tau_map, ota_map, gwas_lib_aug,
        top_n=top_n, min_tau_frac=min_tau_frac,
    )

    if anchor_gene not in gene_list:
        raise ValueError(
            f"{anchor_gene} not in the clustering gene list for {disease_key}. "
            "Check that it has an OTA γ in the checkpoint."
        )

    labels = fcluster(row_link, t=k, criterion="maxclust")
    anchor_local = gene_list.index(anchor_gene)
    anchor_cid   = labels[anchor_local]
    return [gene_list[i] for i, c in enumerate(labels) if c == anchor_cid]


def plot_anchor_cluster_landscape(
    disease_key: str,
    anchor_gene: str,
    top_n: int = 950,
    k: int = 40,
    min_tau_frac: float | None = None,
    min_beta_abs: float = 0.05,
    out_path: str | None = None,
) -> str:
    """
    program_landscape-style scatter for all genes in anchor_gene's β-cluster.

    One subplot per heritable program.  x = fingerprint r (KO transcriptional
    similarity to disease DEG), y = |β × τ*| (program-specific OTA contribution).
    All cluster genes labeled; anchor gene marked with a star.  No colorbar.

    Parameters
    ----------
    disease_key  : "cad" or "ra"
    anchor_gene  : seed gene — its cluster is extracted via get_anchor_cluster()
    min_beta_abs : gene appears in a program panel only if |β| ≥ this value
    out_path     : defaults to outputs/{anchor_gene.lower()}_cluster_landscape_{disease_key}.png
    """
    import math as _math
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.lines import Line2D

    dk = disease_key.lower()
    if min_tau_frac is None:
        min_tau_frac = _DISEASE_CFG[dk].get("min_tau_frac", 0.20)

    cluster_genes = get_anchor_cluster(disease_key, anchor_gene, top_n=top_n, k=k,
                                       min_tau_frac=min_tau_frac)
    print(f"  {anchor_gene} cluster: {len(cluster_genes)} genes")

    bench_map = _benchmark_direction_map(disease_key)  # {gene: expected_sign}
    bench_reduces  = {g for g, s in bench_map.items() if s == -1}  # KD reduces risk
    bench_increases = {g for g, s in bench_map.items() if s == +1}  # KD increases risk

    beta_full, ko_genes, prog_ids, tau_map, ota_map, gwas_lib = _load(disease_key)

    # Fingerprint r — loaded from disease fingerprint file
    cfg = _DISEASE_CFG[dk]
    fp_path = pathlib.Path(ROOT / cfg["beta_npz"]).parent / "disease_fingerprint_match.json"
    fp_r: dict[str, float] = {}
    if fp_path.exists():
        fp_raw = json.loads(fp_path.read_text())
        for entry in fp_raw.get("results", []):
            fp_r[entry["gene_ko"]] = float(entry["r"])

    tau_max    = max(tau_map.values()) if tau_map else 0.0
    tau_thresh = tau_max * min_tau_frac
    heritable  = [p for p in prog_ids if tau_map.get(p, 0.0) >= tau_thresh]
    if not heritable:
        heritable = list(prog_ids)
    heritable_sorted = sorted(heritable, key=lambda p: -tau_map.get(p, 0.0))
    prog_idx = {p: prog_ids.index(p) for p in heritable}

    cg_ridx = {g: ko_genes.index(g) for g in cluster_genes if g in ko_genes}

    bench_in_cluster = {g for g in (bench_reduces | bench_increases) if g in cg_ridx}

    # Build per-program members: (gene, |β×τ*|, fingerprint_r)
    # Benchmark genes bypass min_beta_abs so they always appear in their programs.
    per_prog: dict[str, list] = {p: [] for p in heritable}
    for gene, ridx in cg_ridx.items():
        r = fp_r.get(gene, float("nan"))
        threshold = 0.0 if gene in bench_in_cluster else min_beta_abs
        for p in heritable:
            b = float(beta_full[ridx, prog_idx[p]])
            if abs(b) < threshold:
                continue
            per_prog[p].append((gene, abs(b * tau_map.get(p, 0.0)), r))

    show_progs = [p for p in heritable_sorted if per_prog[p]]
    print(f"  Programs with ≥1 cluster gene: {len(show_progs)}")

    y_max = max(
        (bt for p in show_progs for _, bt, _ in per_prog[p]),
        default=1.0,
    ) * 1.08

    n_panels = 1 + len(show_progs)
    n_cols   = 4
    n_rows   = _math.ceil(n_panels / n_cols)
    panel_sz = 4.0

    fig = plt.figure(figsize=(n_cols * panel_sz, n_rows * panel_sz))
    gs  = GridSpec(n_rows, n_cols, figure=fig, hspace=0.55, wspace=0.38)

    ax_k = fig.add_subplot(gs[0, 0])
    ax_k.axis("off")
    ax_k.set_title("How to read", fontsize=9, fontweight="bold", pad=4)
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2C7BB6",
               markeredgecolor="none", markersize=7, label="Cluster gene"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#E67E22",
               markeredgecolor="#7D3C00", markersize=11, label=f"{anchor_gene} (anchor)"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#D32F2F",
               markeredgecolor="#7B1818", markersize=7, label="Benchmark: KD increases risk"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#1565C0",
               markeredgecolor="#0D3B6E", markersize=7, label="Benchmark: KD reduces risk"),
    ]
    ax_k.legend(handles=handles, loc="upper left", fontsize=7.0,
                frameon=False, handletextpad=0.6, labelspacing=0.8)
    ax_k.text(0.03, 0.42, (
        "x  = fingerprint r (KO vs disease DEG)\n"
        "     negative → KO reverses disease RNA\n"
        "     positive → KO mimics disease RNA\n\n"
        "y  = |β × τ*| (program-specific OTA contrib)\n"
        "     β = Perturb-seq KO loading on program\n"
        "     τ* = S-LDSC heritability enrichment\n\n"
        "Gene shown if |β| ≥ 0.05 in that program."
    ), transform=ax_k.transAxes, fontsize=6.5, va="top",
       family="monospace", color="#2C2C2C", linespacing=1.5)

    for pi, prog in enumerate(show_progs):
        slot = pi + 1
        row_p, col_p = divmod(slot, n_cols)
        ax = fig.add_subplot(gs[row_p, col_p])

        members = per_prog[prog]
        ax.axvspan(-1.05, 0, alpha=0.04, color="#27AE60", zorder=0)
        ax.axvspan(0, 1.05, alpha=0.04, color="#E74C3C", zorder=0)

        for gene, bt, r in members:
            r_plot = 0.0 if _math.isnan(r) else r
            is_anchor = gene == anchor_gene
            is_reduces  = gene in bench_reduces
            is_increases = gene in bench_increases
            is_bench = is_reduces or is_increases

            if is_anchor:
                fc = "#E67E22"; ec = "#7D3C00"; sz = 80; mk = "*"; zo = 9
            elif is_reduces:
                fc = "#1565C0"; ec = "#0D3B6E"; sz = 36; mk = "D"; zo = 7
            elif is_increases:
                fc = "#D32F2F"; ec = "#7B1818"; sz = 36; mk = "D"; zo = 7
            else:
                fc = "#2C7BB6"; ec = "none";    sz = 18; mk = "o"; zo = 4

            ax.scatter(r_plot, bt, c=fc, s=sz, marker=mk,
                       edgecolors=ec, linewidths=0.8 if is_bench or is_anchor else 0,
                       alpha=0.95 if is_bench or is_anchor else 0.65, zorder=zo)
            ax.annotate(
                gene, (r_plot, bt), xytext=(2, 2), textcoords="offset points",
                ha="left", va="bottom",
                fontsize=5.5 if is_anchor or is_bench else 4.5,
                color="#7D3C00" if is_anchor else ("#D32F2F" if is_increases else "#1565C0" if is_reduces else "#2C3E50"),
                fontweight="bold" if is_anchor or is_bench else "normal",
                zorder=10,
            )

        ax.axvline(0, color="#999", lw=0.8, ls="--", zorder=2)
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(bottom=0, top=y_max)
        ax.set_xlabel("KO phenotype reversal  (fingerprint r)", fontsize=5.5)
        ax.set_ylabel("|β × τ*|", fontsize=5.5)
        ax.set_facecolor("white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=5)
        tau_val = tau_map.get(prog, float("nan"))
        tau_str = f"τ*={tau_val:.3f}" if not _math.isnan(tau_val) else ""
        ax.set_title(
            f"{prog}  ·  {tau_str}\n[{len(members)} cluster genes]",
            fontsize=6.5, fontweight="bold", pad=3,
        )

    fig.suptitle(
        f"{anchor_gene} cluster — {disease_key.upper()} program landscape\n"
        f"{len(cluster_genes)} genes  ·  genes with |β| ≥ {min_beta_abs} shown per program",
        fontsize=9, fontweight="bold", y=1.01,
    )

    dest = out_path or f"outputs/{anchor_gene.lower()}_cluster_landscape_{disease_key}.png"
    plt.savefig(dest, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {dest}")
    return dest


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="De novo anchor discovery")
    ap.add_argument("--disease",      choices=["cad", "ra"], required=True)
    ap.add_argument("--top-n",        type=int,   default=950)
    ap.add_argument("--k",            type=int,   default=40)
    ap.add_argument("--min-tau-frac", type=float, default=0.20)
    ap.add_argument("--min-n",        type=int,   default=5, help="min cluster size")
    ap.add_argument("--top",          type=int,   default=15, help="clusters to print")
    args = ap.parse_args()

    print(f"\n── De novo anchor ranking  ({args.disease.upper()}, k={args.k}) ──")
    ranked = rank_denovo_anchors(
        args.disease, top_n=args.top_n, k=args.k,
        min_tau_frac=args.min_tau_frac, min_n_genes=args.min_n,
    )
    _DIR_SHORT = {
        "kd_reduces_risk":   "↓risk",
        "kd_increases_risk": "↑risk",
        "mixed":             "mixed",
        "novel":             "novel",
    }
    print(f"{'#':>3}  {'cl':>3}  {'n':>4}  {'gwas':>4}  {'score':>7}  "
          f"{'gwas_f':>6}  {'γ':>6}  {'sil':>5}  {'dir':>7}  anchor")
    print("─" * 82)
    for i, r in enumerate(ranked[: args.top]):
        dshort = _DIR_SHORT.get(r.get("kd_direction", "novel"), "?")
        print(
            f"{i+1:3d}  {r['cluster_id']:3d}  {r['n_genes']:4d}  "
            f"{r['n_gwas']:4d}  {r['anchor_score']:7.4f}  "
            f"{r['gwas_frac']:6.2f}  {r['mean_gamma']:6.3f}  "
            f"{r['mean_sil']:5.3f}  {dshort:>7}  {r['top_gwas_gene']}"
        )
