"""
Long-island multi-panel plot — transcriptional fingerprint landscape.

Panel 0  — Disease state: all KOs, y = fingerprint_r (KO vs disease DEG).
Panel 1…N — One per NMF program with ≥MIN_PROG_TARGETS OTA genes:
             background grey, program OTA genes highlighted,
             GPS program reversers placed at convergence-target x-position.

X-axis (shared): all Perturb-seq KOs ordered by disease-weighted hierarchical
clustering of 30-dim SVD fingerprints.  Genes that modify the disease via the
same transcriptional axes cluster together (islands).

GPS compound placement in program panels:
  x = score-weighted mean x-position of the compound's convergence target genes
  y = mean fingerprint_r of those targets
  This is now valid: convergence targets are gene-specific (fixed score bug).

Usage:
    python outputs/plot_long_island.py --disease ra
    python outputs/plot_long_island.py --disease cad
    python outputs/plot_long_island.py --disease ra --n_islands 14
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import math
from collections import defaultdict

import numpy as np

# Allow running as `python outputs/plot_long_island.py` from project root
_PROJECT_ROOT = pathlib.Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

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

MIN_PROG_TARGETS = 10   # minimum OTA genes with fingerprint_r to render a program panel
MIN_PROG_GAMMA   = 0.10  # minimum |γ_{program→disease}| to render a program panel


# ─── loaders ──────────────────────────────────────────────────────────────────

def _load_ck(slug: str) -> dict:
    p = pathlib.Path(f"data/checkpoints/{slug}__tier4.json")
    if not p.exists():
        sys.exit(f"Tier-4 checkpoint not found: {p}")
    return json.load(open(p))


def _load_t3(slug: str) -> dict:
    p = pathlib.Path(f"data/checkpoints/{slug}__tier3.json")
    return json.load(open(p)) if p.exists() else {}


def _load_svd(dataset_id: str) -> tuple[np.ndarray, list[str]]:
    d = np.load(f"data/perturbseq/{dataset_id}/svd_loadings.npz")
    return d["Vt"].T, list(d["pert_names"])


def _load_fp_r(dataset_id: str) -> dict[str, float]:
    p = pathlib.Path(f"data/perturbseq/{dataset_id}/disease_fingerprint_match.json")
    if not p.exists():
        return {}
    d = json.load(open(p))
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

def _build_layout(fp: np.ndarray, all_r: np.ndarray,
                  n_islands: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (x_pos, xi_to_cluster, link_Z).
    x_pos[i] = x-axis position of gene i.
    """
    from scipy.cluster.hierarchy import linkage, leaves_list, fcluster
    from scipy.spatial.distance import pdist

    # Weight SVD dims by correlation with fingerprint_r
    fp_w = fp.copy()
    valid = ~np.isnan(all_r)
    if valid.sum() >= 4:
        dw = np.zeros(fp.shape[1])
        for k in range(fp.shape[1]):
            x = fp[valid, k]; y = all_r[valid]
            xm, ym = x - x.mean(), y - y.mean()
            den = np.sqrt((xm**2).sum() * (ym**2).sum())
            dw[k] = abs(xm @ ym / den) if den > 1e-12 else 0.0
        ws = dw.sum()
        if ws > 1e-12:
            dw = dw / ws * fp.shape[1]
        fp_w *= dw[np.newaxis, :]

    norms = np.linalg.norm(fp_w, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    fp_norm = fp_w / norms

    dists  = pdist(fp_norm, metric="cosine")
    Z      = linkage(dists, method="ward")
    order  = leaves_list(Z)
    labels = fcluster(Z, n_islands, criterion="maxclust")

    x_pos = np.empty(len(order), dtype=int)
    x_pos[order] = np.arange(len(order))

    # xi_to_cluster[xi] = cluster label
    xi_to_cluster = np.empty(len(order), dtype=int)
    for i in range(len(order)):
        xi_to_cluster[x_pos[i]] = labels[i]

    return x_pos, xi_to_cluster, Z, labels


# ─── cluster metadata ─────────────────────────────────────────────────────────

def _island_color(mean_r: float) -> str:
    import matplotlib.colors as mc
    if mean_r <= -0.15:
        t = min(1.0, abs(mean_r) / 0.4)
        rgb = (0.13 + (1-t)*0.60, 0.47 + (1-t)*0.30, 0.71 + (1-t)*0.20)
    elif mean_r >= 0.15:
        t = min(1.0, mean_r / 0.4)
        rgb = (0.80 + t*0.10, 0.20 - t*0.10, 0.20 - t*0.10)
    else:
        rgb = (0.75, 0.75, 0.75)
    return mc.to_hex(rgb)


def _build_cluster_meta(n_genes, fp_names, labels, x_pos, all_r, ota_genes):
    import numpy as np
    cluster_xi:  dict[int, list[int]]           = defaultdict(list)
    cluster_r:   dict[int, list[float]]         = defaultdict(list)
    cluster_ota: dict[int, list[tuple[str,float]]] = defaultdict(list)

    for i in range(n_genes):
        c = labels[i]; xi = x_pos[i]; r = all_r[i]
        cluster_xi[c].append(xi)
        if not math.isnan(r):
            cluster_r[c].append(r)
        if fp_names[i] in ota_genes and not math.isnan(r):
            cluster_ota[c].append((fp_names[i], r))

    meta: dict[int, dict] = {}
    for c in cluster_xi:
        xs     = cluster_xi[c]
        mean_r = float(np.mean(cluster_r.get(c, [0.0])))
        top_ota = sorted(cluster_ota.get(c, []), key=lambda t: t[1])
        meta[c] = {
            "x_min":   min(xs), "x_max": max(xs),
            "x_mid":   (min(xs) + max(xs)) / 2,
            "mean_r":  mean_r,
            "top_ota": top_ota[0][0] if top_ota else None,
            "color":   _island_color(mean_r),
        }
    return meta


# ─── helpers for a single panel ───────────────────────────────────────────────

def _draw_islands(ax, cluster_meta):
    """Shade cluster regions and draw boundary hairlines."""
    import matplotlib.colors as mc, numpy as np
    for c, meta in cluster_meta.items():
        rgb  = np.array(mc.to_rgb(meta["color"]))
        grey = np.array([0.97, 0.97, 0.97])
        shade = tuple(0.18 * rgb + 0.82 * grey)
        ax.axvspan(meta["x_min"] - 0.5, meta["x_max"] + 0.5,
                   color=shade, alpha=1.0, zorder=0, linewidth=0)
        ax.axvline(meta["x_max"] + 0.5, color="#CCCCCC", lw=0.4,
                   ls="-", zorder=1, alpha=0.6)


def _draw_background(ax, all_xi, all_r, bg_mask, bg_colors):
    ax.scatter(all_xi[bg_mask], all_r[bg_mask],
               c=[bg_colors[i] for i in range(len(all_xi)) if bg_mask[i]],
               s=3, alpha=0.45, linewidths=0, zorder=2, rasterized=True)


def _make_bg_colors(n_genes, labels, cluster_meta, is_ota):
    import matplotlib.colors as mc, numpy as np
    bg_colors = []
    for i in range(n_genes):
        if is_ota[i]:
            bg_colors.append(None)
            continue
        c   = labels[i]
        rgb = np.array(mc.to_rgb(cluster_meta[c]["color"]))
        grey = np.array([0.92, 0.92, 0.92])
        bg_colors.append(tuple(0.35 * rgb + 0.65 * grey))
    return bg_colors


def _draw_island_labels(ax, cluster_meta, labeled_genes, n_genes,
                        y_bot=-0.88, y_top=0.88):
    """Annotate therapeutic / disease-amplifying cluster midpoints."""
    for c, meta in cluster_meta.items():
        if abs(meta["mean_r"]) < 0.10 or meta["top_ota"] is None:
            continue
        if meta["top_ota"] in labeled_genes:
            continue
        y_lbl  = y_bot if meta["mean_r"] < 0 else y_top
        va     = "bottom" if meta["mean_r"] < 0 else "top"
        col_lb = "#1A237E" if meta["mean_r"] < 0 else "#B71C1C"
        ax.text(meta["x_mid"], y_lbl,
                f"{meta['top_ota']}\n(r={meta['mean_r']:+.2f})",
                ha="center", va=va, fontsize=4.8, color=col_lb,
                fontweight="bold", zorder=6,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec=col_lb,
                          alpha=0.80, linewidth=0.5))


def _style_ax(ax, n_genes, ylabel, title="", yticks=True):
    ax.axhline(0, color="#888888", lw=0.7, ls="--", zorder=3)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlim(-80, n_genes + 80)
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticks([])
    if title:
        ax.set_title(title, fontsize=8, fontweight="bold", pad=3)


# ─── main ─────────────────────────────────────────────────────────────────────

def plot_long_island(disease_key: str, out_path: str | None = None,
                     n_islands: int = 12) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    from matplotlib.gridspec import GridSpec
    from scipy.cluster.hierarchy import fcluster

    cfg = _DISEASE_CONFIG[disease_key]
    ck  = _load_ck(cfg["slug"])
    t3  = _load_t3(cfg["slug"])

    targets     = ck["prioritization_result"]["targets"]
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

    disease_rev   = chem.get("gps_disease_reversers", []) or []
    prog_rev_dict = chem.get("gps_program_reversers", {}) or {}

    # ── Load SVD + fingerprint r ──────────────────────────────────────────────
    print(f"Loading SVD fingerprints for {cfg['dataset_id']}…")
    fp_mat, fp_names = _load_svd(cfg["dataset_id"])
    fp_r_all         = _load_fp_r(cfg["dataset_id"])
    name_to_idx      = {n: i for i, n in enumerate(fp_names)}

    n_genes = len(fp_names)
    all_r   = np.array([fp_r_all.get(fp_names[i], float("nan")) for i in range(n_genes)])

    # ── Build layout (cluster order, labels) ──────────────────────────────────
    print("  Building disease-weighted clustering…")
    x_pos, xi_to_cluster, link_Z, cluster_labels = _build_layout(fp_mat, all_r, n_islands)
    name_to_xi  = {name: x_pos[i] for i, name in enumerate(fp_names)}
    all_xi      = np.array([x_pos[i] for i in range(n_genes)])
    is_ota      = np.array([fp_names[i] in ota_genes for i in range(n_genes)])
    is_known    = np.array([fp_names[i] in known for i in range(n_genes)])

    cluster_meta = _build_cluster_meta(n_genes, fp_names, cluster_labels, x_pos, all_r, ota_genes)
    bg_colors    = _make_bg_colors(n_genes, cluster_labels, cluster_meta, is_ota)
    bg_mask      = ~is_ota & ~np.isnan(all_r)

    # Build prog → list[(gene, xi, r)] lookup for fast compound placement
    prog_gene_coords: dict[str, list[tuple[str, float, float]]] = {}
    for t in targets:
        gene = t["target_gene"]
        if gene not in name_to_xi or gene not in fp_r_all:
            continue
        tp = t.get("top_programs") or {}
        for prog, beta in tp.items():
            if abs(beta) >= 0.05:
                prog_gene_coords.setdefault(prog, []).append(
                    (gene, float(name_to_xi[gene]), fp_r_all[gene])
                )

    def _gps_x_y_from_vector(
        program_vector: dict[str, float],
    ) -> tuple[float | None, float | None]:
        """
        Place a compound in island space using its program reversal vector.

        x = |z_rges_P|-weighted mean x-position of OTA genes in reversed programs.
            Genes weighted by |z_rges_P| × |β_{gene→P}|.
        y = fingerprint_r of those genes, weighted by the same joint weight.
            Compounds reversing therapeutic programs land in the therapeutic zone.

        Returns (None, None) when no OTA genes with fingerprint_r overlap the
        compound's active programs.
        """
        gene_w: dict[str, float] = {}
        gene_r: dict[str, float] = {}

        for prog, z in program_vector.items():
            az = abs(z)
            if az < 1.0:
                continue
            for gene, xi, r in prog_gene_coords.get(prog, []):
                # Look up |β_{gene→P}| from OTA target record
                tp   = ota_genes.get(gene, {}).get("top_programs") or {}
                beta = abs(float(tp.get(prog, 0.0)))
                w    = az * max(beta, 0.05)   # floor so genes with β just below threshold still count
                gene_w[gene] = gene_w.get(gene, 0.0) + w
                gene_r[gene] = r

        if not gene_w:
            return None, None

        total_w = sum(gene_w.values())
        x = sum(name_to_xi[g] * w for g, w in gene_w.items() if g in name_to_xi) / total_w
        y = sum(gene_r[g] * w for g, w in gene_w.items()) / total_w
        return x, float(y)

    def _gps_x_y(compound_id: str) -> tuple[float | None, float | None]:
        """Resolve compound position from its stored program_vector."""
        # Search disease reversers then program reversers for the program_vector
        for hit in disease_rev:
            if hit.get("compound_id") == compound_id:
                pv = hit.get("program_vector") or {}
                if pv:
                    return _gps_x_y_from_vector(pv)
        for hits in prog_rev_dict.values():
            for hit in hits:
                if hit.get("compound_id") == compound_id:
                    pv = hit.get("program_vector") or {}
                    if pv:
                        return _gps_x_y_from_vector(pv)
        return None, None

    # ── Determine program panels (≥MIN_PROG_TARGETS OTA genes with fp_r) ──────
    prog_ota_with_r: dict[str, list[tuple[str, float]]] = {}
    for t in targets:
        prog = _dom_prog(t)
        gene = t["target_gene"]
        if prog and gene in fp_r_all:
            prog_ota_with_r.setdefault(prog, []).append((gene, fp_r_all[gene]))

    show_progs = [p for p in all_progs
                  if len(prog_ota_with_r.get(p, [])) >= MIN_PROG_TARGETS
                  and abs(prog_gammas.get(p, 0.0)) >= MIN_PROG_GAMMA]

    # ── Layout ────────────────────────────────────────────────────────────────
    n_panels = 1 + len(show_progs)
    # Disease panel taller; program panels shorter
    h_ratios = [3.5] + [2.0] * len(show_progs)
    fig_h    = 4.5 + 2.5 * len(show_progs)
    fig = plt.figure(figsize=(22, fig_h))
    gs  = GridSpec(n_panels, 1, figure=fig, hspace=0.45,
                   height_ratios=h_ratios)

    # ═════════════════════════════════════════════════════════════════════════
    # Panel 0: Disease state — full KO landscape
    # ═════════════════════════════════════════════════════════════════════════
    ax0 = fig.add_subplot(gs[0])
    _draw_islands(ax0, cluster_meta)
    _draw_background(ax0, all_xi, all_r, bg_mask, bg_colors)

    # OTA targets
    for i in range(n_genes):
        if not is_ota[i] or math.isnan(all_r[i]):
            continue
        t    = ota_genes[fp_names[i]]
        prog = _dom_prog(t)
        col  = prog_color.get(prog, "#9E9E9E")
        size = 65 if t.get("dominant_tier", "") == "Tier1_Interventional" else 24
        ec   = "#B71C1C" if is_known[i] else "white"
        ew   = 1.8       if is_known[i] else 0.4
        ax0.scatter(all_xi[i], all_r[i], c=col, s=size,
                    edgecolors=ec, linewidths=ew, alpha=0.93, zorder=4)

    # Labels: known + top-15% OTA by |r|
    ota_rv = sorted([(abs(all_r[i]), i) for i in range(n_genes)
                     if is_ota[i] and not math.isnan(all_r[i])])
    thresh = ota_rv[max(0, int(len(ota_rv)*0.85))][0] if ota_rv else 0.0
    labeled: set[str] = set()
    for i in range(n_genes):
        if not is_ota[i] or math.isnan(all_r[i]):
            continue
        gene = fp_names[i]
        if is_known[i] or abs(all_r[i]) >= thresh:
            col = "#B71C1C" if is_known[i] else "#222222"
            fw  = "bold"   if is_known[i] else "normal"
            va  = "bottom" if all_r[i] >= 0 else "top"
            dy  = 0.020    if all_r[i] >= 0 else -0.020
            ax0.annotate(gene, (all_xi[i], all_r[i]+dy),
                         ha="center", va=va, fontsize=5.5,
                         color=col, fontweight=fw, zorder=6)
            labeled.add(gene)

    _draw_island_labels(ax0, cluster_meta, labeled, n_genes)

    ax0.text(100, -1.01, "← KO reverses disease (therapeutic)",
             fontsize=7, color="#1A237E", va="bottom", fontweight="bold")
    ax0.text(100,  0.99, "KO amplifies disease →",
             fontsize=7, color="#B71C1C", va="top", fontweight="bold")

    n_with_r = sum(1 for i in range(n_genes) if is_ota[i] and not math.isnan(all_r[i]))
    _style_ax(ax0, n_genes,
              ylabel="Fingerprint r\n(KO vs disease DEG)",
              title=(f"{cfg['label']}  ·  disease state\n"
                     f"{n_genes:,} KOs · {n_islands} islands · "
                     f"{sum(is_ota)} OTA targets ({n_with_r} with fingerprint r)"))

    # Legend for panel 0
    prog_h = [mpatches.Patch(color=prog_color[p],
                              label=f"{p.split('_')[-1]} γ={prog_gammas.get(p,0):+.2f}")
              for p in all_progs]
    misc_h = [
        mpatches.Patch(color=_island_color(-0.25), alpha=0.6, label="Therapeutic island"),
        mpatches.Patch(color=_island_color(+0.25), alpha=0.6, label="Disease island"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#AAAAAA",
               markersize=5, label="Perturb-seq library"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#1565C0",
               markersize=8, label="OTA target (large=Tier1)"),
        mpatches.Patch(facecolor="none", edgecolor="#B71C1C", linewidth=1.8,
                       label="Validated target"),
        Line2D([0],[0], marker="*", color="w", markerfacecolor="#FFC107",
               markersize=11, label="GPS reverser (program panel)"),
    ]
    ax0.legend(handles=prog_h + misc_h, loc="upper right",
               fontsize=5.8, ncol=3, framealpha=0.92)

    # ═════════════════════════════════════════════════════════════════════════
    # Panels 1…N: One per NMF program
    # ═════════════════════════════════════════════════════════════════════════
    for pi, prog in enumerate(show_progs):
        ax = fig.add_subplot(gs[1 + pi], sharex=ax0)
        pcol  = prog_color.get(prog, "#555555")
        pname = prog.split("_")[-1]
        gamma = prog_gammas.get(prog, float("nan"))

        _draw_islands(ax, cluster_meta)

        # Background: all KOs at their landscape positions but faint grey
        ax.scatter(all_xi[bg_mask], all_r[bg_mask],
                   c="#DDDDDD", s=2, alpha=0.30, linewidths=0,
                   zorder=2, rasterized=True)

        # Program OTA genes
        prog_genes = prog_ota_with_r.get(prog, [])
        prog_gene_set = {g for g, _ in prog_genes}
        labeled_prog: set[str] = set()
        rv_prog = sorted([(abs(r), g) for g, r in prog_genes])
        thresh_prog = rv_prog[max(0, int(len(rv_prog)*0.80))][0] if rv_prog else 0.0

        for gene, r in prog_genes:
            if gene not in name_to_xi:
                continue
            xi = name_to_xi[gene]
            t  = ota_genes.get(gene, {})
            size = 55 if t.get("dominant_tier", "") == "Tier1_Interventional" else 22
            ec   = "#B71C1C" if gene in known else "white"
            ew   = 1.6       if gene in known else 0.4
            ax.scatter(xi, r, c=pcol, s=size,
                       edgecolors=ec, linewidths=ew, alpha=0.92, zorder=4)
            if gene in known or abs(r) >= thresh_prog:
                va = "bottom" if r >= 0 else "top"
                dy = 0.018 if r >= 0 else -0.018
                ax.annotate(gene, (xi, r+dy), ha="center", va=va,
                            fontsize=5, color="#222222", zorder=6)
                labeled_prog.add(gene)

        # GPS program reversers: place using program_vector directly
        gps_hits = prog_rev_dict.get(prog, []) or []
        seen_cids: set[str] = set()
        placed: list[tuple[float, float, float]] = []  # (x, y, z_on_this_prog)
        for c in gps_hits:
            cid = c.get("compound_id", "")
            if cid in seen_cids:
                continue
            seen_cids.add(cid)
            pv = c.get("program_vector") or {}
            z  = pv.get(prog, c.get("z_rges", c.get("rges", 0.0)))
            gx, gy = _gps_x_y_from_vector(pv) if pv else (None, None)
            if gx is not None and gy is not None:
                placed.append((gx, gy, float(z)))

        if placed:
            gxs = np.array([p[0] for p in placed])
            gys = np.array([p[1] for p in placed])
            gzs = np.array([p[2] for p in placed])
            sz  = np.clip(np.abs(gzs) * 20, 40, 200)
            ax.scatter(gxs, gys, c="#FFC107", s=sz, marker="*",
                       edgecolors="#E65100", linewidths=0.8,
                       alpha=0.95, zorder=7,
                       label=f"{len(placed)} GPS reversers")
            # Label the strongest GPS hit
            best = int(np.argmin(gzs))
            ax.annotate(f"z={gzs[best]:.1f}", (gxs[best], gys[best]+0.04),
                        ha="center", va="bottom", fontsize=4.8,
                        color="#E65100", fontweight="bold", zorder=8)

        gamma_str = f"γ={gamma:+.2f}" if not math.isnan(gamma) else ""
        n_placed  = len(placed)
        gps_str   = f" · {n_placed}/{len(gps_hits)} GPS hits placed" if gps_hits else ""
        _style_ax(ax, n_genes,
                  ylabel=f"Fingerprint r\n(KO vs disease DEG)",
                  title=f"{prog}  {gamma_str}  ·  {len(prog_genes)} OTA targets{gps_str}")

        if placed:
            ax.legend(fontsize=6, loc="upper right", framealpha=0.85)

    # ── X-axis label on bottom panel ──────────────────────────────────────────
    last_ax = ax if show_progs else ax0
    last_ax.set_xlabel(
        "Gene KOs  ·  x = disease-weighted fingerprint clustering  "
        "(← therapeutic islands   |   disease islands →)",
        fontsize=8,
    )

    plt.savefig(out_path or f"outputs/long_island_{disease_key}.png",
                dpi=160, bbox_inches="tight")
    print(f"Saved → {out_path or f'outputs/long_island_{disease_key}.png'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--disease",   choices=["ra", "cad"], required=True)
    ap.add_argument("--out",       type=str, default=None)
    ap.add_argument("--n_islands", type=int, default=12)
    args = ap.parse_args()
    plot_long_island(args.disease, out_path=args.out, n_islands=args.n_islands)
