"""
Long-island Manhattan plot — transcriptional fingerprint landscape.

Modelled on Grabski/Satija RNA fingerprinting (bioRxiv 2025).

X-axis: ALL gene KOs in the Perturb-seq library, ordered by hierarchical
        clustering of their 30-dim SVD fingerprint vectors, weighted by each
        dimension's correlation with disease fingerprint_r.  Genes that modify
        disease via the same transcriptional axes cluster together.

Y-axis: fingerprint r — Pearson r of each gene's KO profile vs the disease
        DEG vector.  Negative = KO reverses disease state (therapeutic).
        Positive = KO mimics disease.

Visual structure:
  - Background dots colored by k-means cluster; blue = therapeutic island,
    red = disease-amplifying island, grey = neutral
  - Cluster regions shaded at low opacity to make islands pop
  - Cluster boundaries marked; top OTA gene per island annotated
  - OTA targets overlaid as colored circles (by NMF program), Tier1 larger
  - GPS compounds shown as ranked bar chart (no false x-axis placement)

Usage:
    python outputs/plot_long_island.py --disease ra
    python outputs/plot_long_island.py --disease cad
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import math

import numpy as np

_DISEASE_CONFIG = {
    "ra": {
        "label":      "Rheumatoid arthritis",
        "slug":       "rheumatoid_arthritis",
        "dataset_id": "czi_2025_cd4t_perturb",
        "known":      {"IL6R","DHODH","TYK2","JAK1","JAK2","CTLA4","TNF","IL2RA","STAT3"},
    },
    "cad": {
        "label":      "Coronary artery disease",
        "slug":       "coronary_artery_disease",
        "dataset_id": "schnitzler_cad_vascular",
        "known":      {"LDLR","PCSK9","APOB","HMGCR","IL6R","LPA","CETP","SORT1"},
    },
}

_PROG_PALETTE = [
    "#1565C0","#2E7D32","#6A1B9A","#E65100","#B71C1C",
    "#00695C","#4527A0","#F57F17","#37474F","#AD1457",
]

# ─── loaders ──────────────────────────────────────────────────────────────────

def _load_ck(slug: str) -> dict:
    p = pathlib.Path(f"data/checkpoints/{slug}__tier4.json")
    if not p.exists():
        sys.exit(f"Tier-4 checkpoint not found: {p}")
    with open(p) as f:
        return json.load(f)


def _load_t3(slug: str) -> dict:
    p = pathlib.Path(f"data/checkpoints/{slug}__tier3.json")
    return json.load(open(p)) if p.exists() else {}


def _load_svd(dataset_id: str) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(f"data/perturbseq/{dataset_id}/svd_loadings.npz")
    return d["Vt"].T, d["pert_names"]   # (n_genes, 30), (n_genes,)


def _load_fp_r(dataset_id: str) -> dict[str, float]:
    p = pathlib.Path(f"data/perturbseq/{dataset_id}/disease_fingerprint_match.json")
    if not p.exists():
        return {}
    with open(p) as f:
        d = json.load(f)
    return {e["gene_ko"]: e["r"] for e in d["results"]}


def _prog_gamma(t3: dict, disease_key: str) -> dict[str, float]:
    ge = t3.get("gamma_estimates", {})
    trait_keys = {"ra": ["RA", "rheumatoid arthritis"],
                  "cad": ["CAD", "coronary artery disease"]}.get(disease_key, [])
    out: dict[str, float] = {}
    for prog, data in ge.items():
        for tk in trait_keys:
            v = data.get(tk)
            if v and isinstance(v, dict) and v.get("gamma") is not None:
                out[prog] = v["gamma"]
                break
    return out


# ─── disease-weighted clustering ──────────────────────────────────────────────

def _disease_weighted_fingerprints(fp: np.ndarray, all_r: np.ndarray) -> np.ndarray:
    """
    Scale each SVD dimension by |Pearson r(dimension loadings, fingerprint_r)|.
    Emphasises axes that co-vary with disease outcome; suppresses noise dimensions.
    """
    fp_w = fp.copy()
    valid = ~np.isnan(all_r)
    if valid.sum() < 4:
        return fp_w
    dim_weights = np.zeros(fp.shape[1])
    for k in range(fp.shape[1]):
        x = fp[valid, k]; y = all_r[valid]
        xm, ym = x - x.mean(), y - y.mean()
        denom = np.sqrt((xm**2).sum() * (ym**2).sum())
        dim_weights[k] = abs(xm @ ym / denom) if denom > 1e-12 else 0.0
    w_sum = dim_weights.sum()
    if w_sum > 1e-12:
        dim_weights = dim_weights / w_sum * fp.shape[1]
    return fp_w * dim_weights[np.newaxis, :]


def _cluster_order(fp_w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Ward linkage on cosine distance of disease-weighted fingerprints.
    Returns (order, linkage_Z).
    """
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import pdist
    norms = np.linalg.norm(fp_w, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    fp_norm = fp_w / norms
    dists = pdist(fp_norm, metric="cosine")
    Z = linkage(dists, method="ward")
    return leaves_list(Z), Z


def _assign_clusters(Z: np.ndarray, n_clusters: int) -> np.ndarray:
    from scipy.cluster.hierarchy import fcluster
    return fcluster(Z, n_clusters, criterion="maxclust")


# ─── island colour map ────────────────────────────────────────────────────────

def _island_color(mean_r: float, alpha: float = 1.0) -> str:
    """
    Map mean cluster r to a colour:
      mean_r <= -0.15  → blue  (strong therapeutic)
      mean_r >= +0.15  → red   (strong disease-amplifying)
      in-between        → grey
    Returns hex string.
    """
    import matplotlib.colors as mcolors
    if mean_r <= -0.15:
        t = min(1.0, abs(mean_r) / 0.4)
        r, g, b = 0.13 + (1-t)*0.60, 0.47 + (1-t)*0.30, 0.71 + (1-t)*0.20
    elif mean_r >= 0.15:
        t = min(1.0, mean_r / 0.4)
        r, g, b = 0.80 + t*0.10, 0.20 - t*0.10, 0.20 - t*0.10
    else:
        r, g, b = 0.75, 0.75, 0.75
    return mcolors.to_hex((r, g, b))


# ─── main ─────────────────────────────────────────────────────────────────────

def plot_long_island(disease_key: str, out_path: str | None = None,
                     n_islands: int = 12) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    from matplotlib.gridspec import GridSpec
    import matplotlib.colors as mcolors

    cfg = _DISEASE_CONFIG[disease_key]
    ck  = _load_ck(cfg["slug"])
    t3  = _load_t3(cfg["slug"])

    targets     = ck["prioritization_result"]["targets"]
    pheno       = ck.get("phenotype_result", {})
    chem        = ck.get("chemistry_result", {})
    known       = cfg["known"]
    prog_gammas = _prog_gamma(t3, disease_key)

    def _dom_prog(t: dict) -> str | None:
        return (t.get("program_drivers") or {}).get("top_program")

    all_progs = sorted(
        {p for t in targets if (p := _dom_prog(t))},
        key=lambda p: -prog_gammas.get(p, 0.0),
    )
    prog_color = {p: _PROG_PALETTE[i % len(_PROG_PALETTE)] for i, p in enumerate(all_progs)}
    ota_genes: dict[str, dict] = {t["target_gene"]: t for t in targets}

    disease_rev    = chem.get("gps_disease_reversers", []) or []
    prog_rev_dict  = chem.get("gps_program_reversers",  {}) or {}
    screened_progs = [p for p in all_progs if prog_rev_dict.get(p)]

    # ── Load SVD fingerprints and cluster ─────────────────────────────────────
    print(f"Loading SVD fingerprints for {cfg['dataset_id']}…")
    fp_mat, fp_names = _load_svd(cfg["dataset_id"])
    fp_r_all         = _load_fp_r(cfg["dataset_id"])

    n_genes = len(fp_names)
    all_r   = np.array([fp_r_all.get(fp_names[i], float("nan")) for i in range(n_genes)])

    print(f"  {n_genes} KOs × {fp_mat.shape[1]} SVD dims")
    print(f"  Building disease-weighted fingerprints…")
    fp_w            = _disease_weighted_fingerprints(fp_mat, all_r)
    order, link_Z   = _cluster_order(fp_w)
    cluster_labels  = _assign_clusters(link_Z, n_islands)

    x_pos = np.empty(n_genes, dtype=int)
    x_pos[order] = np.arange(len(order))
    name_to_xi = {name: x_pos[i] for i, name in enumerate(fp_names)}

    all_xi   = np.array([x_pos[i] for i in range(n_genes)])
    is_ota   = np.array([fp_names[i] in ota_genes for i in range(n_genes)])
    is_known = np.array([fp_names[i] in known      for i in range(n_genes)])

    # ── Per-cluster metadata ──────────────────────────────────────────────────
    # After ordering, cluster_labels[order[j]] tells us which island x-position j belongs to.
    # Build: xi → cluster, cluster → {x_range, mean_r, top_ota_gene}
    xi_to_cluster = np.empty(n_genes, dtype=int)
    for i in range(n_genes):
        xi_to_cluster[x_pos[i]] = cluster_labels[i]

    cluster_xi: dict[int, list[int]] = {}
    cluster_r:  dict[int, list[float]] = {}
    cluster_ota: dict[int, list[tuple[str, float]]] = {}
    for i in range(n_genes):
        c  = cluster_labels[i]
        xi = x_pos[i]
        r  = all_r[i]
        cluster_xi.setdefault(c, []).append(xi)
        if not math.isnan(r):
            cluster_r.setdefault(c, []).append(r)
        if is_ota[i] and not math.isnan(r):
            cluster_ota.setdefault(c, []).append((fp_names[i], r))

    cluster_meta: dict[int, dict] = {}
    for c in cluster_xi:
        xs       = cluster_xi[c]
        mean_r   = float(np.mean(cluster_r.get(c, [0.0])))
        x_min, x_max = min(xs), max(xs)
        # Top OTA gene by |r|
        ota_sorted = sorted(cluster_ota.get(c, []), key=lambda t: t[1])
        top_ota    = ota_sorted[0][0] if ota_sorted else None
        cluster_meta[c] = {
            "x_min":   x_min, "x_max": x_max,
            "x_mid":   (x_min + x_max) / 2,
            "mean_r":  mean_r,
            "n":       len(xs),
            "top_ota": top_ota,
            "color":   _island_color(mean_r),
        }

    # Per-gene dot colour for background: island colour at low saturation
    bg_colors = []
    for i in range(n_genes):
        if is_ota[i]:
            bg_colors.append(None)
            continue
        c = cluster_labels[i]
        base = cluster_meta[c]["color"]
        # Desaturate and lighten by blending toward #EEEEEE
        import matplotlib.colors as mc
        rgb = np.array(mc.to_rgb(base))
        grey = np.array([0.92, 0.92, 0.92])
        blended = 0.40 * rgb + 0.60 * grey
        bg_colors.append(tuple(blended))

    # ── Layout ────────────────────────────────────────────────────────────────
    has_gps = bool(disease_rev or any(prog_rev_dict.values()))
    fig = plt.figure(figsize=(20, 10 if has_gps else 6.5))
    gs  = GridSpec(
        2 if has_gps else 1, 1, figure=fig,
        hspace=0.38,
        height_ratios=[4, 2] if has_gps else [1],
    )
    ax_main = fig.add_subplot(gs[0])

    # ── Island shading (behind everything) ───────────────────────────────────
    for c, meta in cluster_meta.items():
        col = meta["color"]
        import matplotlib.colors as mc
        rgb = np.array(mc.to_rgb(col))
        grey = np.array([0.97, 0.97, 0.97])
        shade = tuple(0.18 * rgb + 0.82 * grey)
        ax_main.axvspan(meta["x_min"] - 0.5, meta["x_max"] + 0.5,
                        color=shade, alpha=1.0, zorder=0, linewidth=0)
        # Cluster boundary hairline
        ax_main.axvline(meta["x_max"] + 0.5, color="#CCCCCC", lw=0.4,
                        ls="-", zorder=1, alpha=0.7)

    # ── Background library dots (coloured by island) ──────────────────────────
    bg_mask = ~is_ota & ~np.isnan(all_r)
    bg_c_arr = [bg_colors[i] for i in range(n_genes) if bg_mask[i]]
    ax_main.scatter(
        all_xi[bg_mask], all_r[bg_mask],
        c=bg_c_arr, s=4, alpha=0.55, linewidths=0, zorder=2, rasterized=True,
    )

    # ── OTA target genes (coloured by NMF program) ───────────────────────────
    for i in range(n_genes):
        if not is_ota[i] or math.isnan(all_r[i]):
            continue
        gene = fp_names[i]
        t    = ota_genes[gene]
        prog = _dom_prog(t)
        col  = prog_color.get(prog, "#9E9E9E")
        tier = t.get("dominant_tier", "")
        size = 70 if tier == "Tier1_Interventional" else 26
        ec   = "#B71C1C" if gene in known else "white"
        ew   = 1.8 if gene in known else 0.5
        ax_main.scatter(all_xi[i], all_r[i], c=col, s=size,
                        edgecolors=ec, linewidths=ew, alpha=0.93, zorder=4)

    # ── Gene labels: known + top 15% OTA by |r| ──────────────────────────────
    ota_r_vals = sorted(
        [(all_r[i], i) for i in range(n_genes) if is_ota[i] and not math.isnan(all_r[i])],
        key=lambda x: abs(x[0])
    )
    thresh_idx = max(1, int(len(ota_r_vals) * 0.85))
    r_threshold_label = abs(ota_r_vals[thresh_idx][0]) if ota_r_vals else 0.0

    labeled: set[str] = set()
    for i in range(n_genes):
        if not is_ota[i] or math.isnan(all_r[i]):
            continue
        gene = fp_names[i]
        if gene in known or abs(all_r[i]) >= r_threshold_label:
            if gene not in labeled:
                col = "#B71C1C" if gene in known else "#222222"
                fw  = "bold"   if gene in known else "normal"
                va  = "bottom" if all_r[i] >= 0 else "top"
                dy  = 0.020 if all_r[i] >= 0 else -0.020
                ax_main.annotate(gene, (all_xi[i], all_r[i] + dy),
                                 ha="center", va=va,
                                 fontsize=5.5, color=col, fontweight=fw, zorder=6)
                labeled.add(gene)

    # ── Island labels: annotate top OTA gene at cluster midpoint ─────────────
    # Only annotate clusters with clear therapeutic or disease signal (|mean_r| > 0.10)
    for c, meta in cluster_meta.items():
        if abs(meta["mean_r"]) < 0.10 or meta["top_ota"] is None:
            continue
        if meta["top_ota"] in labeled:
            continue
        y_label = -0.88 if meta["mean_r"] < 0 else 0.88
        va      = "bottom" if meta["mean_r"] < 0 else "top"
        col_lbl = "#1A237E" if meta["mean_r"] < 0 else "#B71C1C"
        ax_main.text(
            meta["x_mid"], y_label,
            f"{meta['top_ota']}\n(r={meta['mean_r']:+.2f})",
            ha="center", va=va, fontsize=5, color=col_lbl,
            fontweight="bold", zorder=6,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=col_lbl,
                      alpha=0.75, linewidth=0.6),
        )

    # ── Axes ─────────────────────────────────────────────────────────────────
    ax_main.axhline(0, color="#888888", lw=0.8, ls="--", zorder=3)
    ax_main.set_ylabel("Fingerprint r\n(KO profile vs disease DEG)", fontsize=9)
    ax_main.set_ylim(-1.05, 1.05)
    ax_main.set_xlim(-80, n_genes + 80)
    ax_main.set_facecolor("white")
    ax_main.spines["top"].set_visible(False)
    ax_main.spines["right"].set_visible(False)
    ax_main.set_xticks([])

    ax_main.text(100, -1.01, "← KO reverses disease (therapeutic)",
                 fontsize=7, color="#1A237E", va="bottom", fontweight="bold")
    ax_main.text(100,  0.99, "KO amplifies disease →",
                 fontsize=7, color="#B71C1C", va="top", fontweight="bold")

    n_ota_with_r = sum(1 for i in range(n_genes) if is_ota[i] and not math.isnan(all_r[i]))
    ax_main.set_title(
        f"{cfg['label']}  —  long island plot\n"
        f"{n_genes:,} KOs · x = disease-weighted fingerprint clustering "
        f"({n_islands} islands) · {sum(is_ota)} OTA targets ({n_ota_with_r} with fingerprint r)",
        fontsize=10, fontweight="bold",
    )

    # ── Legend ────────────────────────────────────────────────────────────────
    prog_h = [mpatches.Patch(color=prog_color[p],
                              label=f"{p.split('_')[-1]} γ={prog_gammas.get(p,0):+.2f}")
              for p in all_progs]
    misc_h = [
        mpatches.Patch(color=_island_color(-0.25), alpha=0.6,
                       label="Therapeutic island (blue)"),
        mpatches.Patch(color=_island_color(+0.25), alpha=0.6,
                       label="Disease island (red)"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#AAAAAA",
               markersize=5, label="Perturb-seq library (island-coloured)"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#1565C0",
               markersize=8, label="OTA target (large=Tier1)"),
        mpatches.Patch(facecolor="none", edgecolor="#B71C1C", linewidth=1.8,
                       label="Validated drug target"),
    ]
    ax_main.legend(handles=prog_h + misc_h, loc="upper right",
                   fontsize=6, ncol=3, framealpha=0.92)

    # ── GPS compound panel ────────────────────────────────────────────────────
    if has_gps:
        ax_gps = fig.add_subplot(gs[1])

        gps_rows: list[tuple[str, float, str, str]] = []
        for c in disease_rev:
            cid = c.get("compound_id", "?")
            z   = c.get("z_rges", c.get("rges", 0.0))
            gps_rows.append((cid[:16], z, "#FFC107", "disease reversal"))

        for prog, hits in prog_rev_dict.items():
            pcol  = prog_color.get(prog, "#555555")
            pname = prog.split("_")[-1]
            for c in (hits or []):
                cid = c.get("compound_id", "?")
                z   = c.get("z_rges", c.get("rges", 0.0))
                gps_rows.append((cid[:16], z, pcol, pname))

        seen_cids: dict[str, tuple] = {}
        for row in gps_rows:
            existing = seen_cids.get(row[0])
            if existing is None or row[1] < existing[1]:
                seen_cids[row[0]] = row
        gps_rows = sorted(seen_cids.values(), key=lambda r: r[1])

        labels_gps = [r[0] for r in gps_rows]
        zvals      = [r[1] for r in gps_rows]
        colors_gps = [r[2] for r in gps_rows]

        y_pos = range(len(gps_rows))
        ax_gps.barh(list(y_pos), [abs(z) for z in zvals],
                    color=colors_gps, edgecolor="white", linewidth=0.4, alpha=0.88)
        ax_gps.set_yticks(list(y_pos))
        ax_gps.set_yticklabels(labels_gps, fontsize=5.5)
        ax_gps.invert_yaxis()
        ax_gps.set_xlabel("|Z_RGES|  (target-emulation reversal score)", fontsize=8)
        ax_gps.set_title(
            "GPS compound reversers  (ranked by reversal strength)\n"
            "Note: compounds lack measured transcriptional profiles — "
            "x-axis placement in gene landscape is undefined",
            fontsize=8, color="#555555",
        )
        ax_gps.spines["top"].set_visible(False)
        ax_gps.spines["right"].set_visible(False)
        ax_gps.set_facecolor("#FAFAFA")
        ax_gps.axvline(2.0, color="#888888", lw=0.6, ls="--", alpha=0.5)

        cat_labels = {"disease reversal": "#FFC107"}
        for prog in screened_progs:
            cat_labels[prog.split("_")[-1]] = prog_color.get(prog, "#555555")
        cat_h = [mpatches.Patch(color=c, label=lbl) for lbl, c in cat_labels.items()]
        ax_gps.legend(handles=cat_h, fontsize=6, loc="lower right")

    plt.savefig(out_path or f"outputs/long_island_{disease_key}.png",
                dpi=180, bbox_inches="tight")
    print(f"Saved → {out_path or f'outputs/long_island_{disease_key}.png'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--disease",   choices=["ra", "cad"], required=True)
    ap.add_argument("--out",       type=str, default=None)
    ap.add_argument("--n_islands", type=int, default=12)
    args = ap.parse_args()
    plot_long_island(args.disease, out_path=args.out, n_islands=args.n_islands)
