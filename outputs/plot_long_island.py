"""
Long-island Manhattan plot — transcriptional fingerprint landscape.

Modelled on Grabski/Satija RNA fingerprinting (bioRxiv 2025).

X-axis: ALL gene KOs in the Perturb-seq library, ordered by hierarchical
        clustering of their 30-dim SVD fingerprint vectors.  Genes with
        similar transcriptional perturbation profiles cluster together.
        The x-ordering is entirely data-driven.

Y-axis: fingerprint r — Pearson r of each gene's KO profile vs the disease
        DEG vector.  Negative = KO reverses disease state (therapeutic).
        Positive = KO mimics disease.

Highlighted points: OTA targets (our credible set) — colored by NMF program,
        sized by tier.  Background library genes shown as grey.

GPS annotation strip (below x-axis):
        GPS disease-state reversers placed at approximate x-position based on
        the mean fingerprint position of their matched OTA targets (RGES proxy).
        Separate strip for each GPS program screen.

Usage:
    python outputs/plot_long_island.py --disease ra
    python outputs/plot_long_island.py --disease cad
"""
from __future__ import annotations

import argparse
import json
import gzip
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
    """Returns (fingerprints, names): fingerprints shape (n_genes, n_components)."""
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


# ─── hierarchical clustering x-order ─────────────────────────────────────────

def _cluster_order(fingerprints: np.ndarray, fp_r: np.ndarray | None = None) -> np.ndarray:
    """
    Return indices ordering genes by hierarchical clustering of their SVD fingerprint
    vectors, weighted by each dimension's correlation with disease fingerprint_r.

    When fp_r is provided (array of per-gene Pearson r vs disease DEG):
      - Each SVD dimension k is scaled by |corr(fingerprints[:, k], fp_r)|
      - Dimensions with high correlation to disease outcome dominate the clustering
      - Result: genes cluster by SHARED DISEASE-MODIFICATION MECHANISM, not global txn noise
    When fp_r is None: falls back to uniform weighting (original behaviour).
    """
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import pdist

    fp = fingerprints.copy()

    if fp_r is not None and len(fp_r) == fp.shape[0]:
        # Build mask for genes with valid r values
        valid = ~np.isnan(fp_r)
        dim_weights = np.zeros(fp.shape[1])
        for k in range(fp.shape[1]):
            if valid.sum() >= 4:
                # Pearson r between SVD component k loadings and fingerprint_r
                x = fp[valid, k]
                y = fp_r[valid]
                xm, ym = x - x.mean(), y - y.mean()
                denom = np.sqrt((xm**2).sum() * (ym**2).sum())
                dim_weights[k] = abs(xm @ ym / denom) if denom > 1e-12 else 0.0
        # Normalise weights to sum to n_dims so overall scale is preserved
        w_sum = dim_weights.sum()
        if w_sum > 1e-12:
            dim_weights = dim_weights / w_sum * fp.shape[1]
        fp = fp * dim_weights[np.newaxis, :]

    norms = np.linalg.norm(fp, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    fp_norm = fp / norms

    dists = pdist(fp_norm, metric="cosine")
    Z     = linkage(dists, method="ward")
    return leaves_list(Z)


# ─── main ─────────────────────────────────────────────────────────────────────

def plot_long_island(disease_key: str, out_path: str | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    from matplotlib.gridspec import GridSpec

    cfg = _DISEASE_CONFIG[disease_key]
    ck  = _load_ck(cfg["slug"])
    t3  = _load_t3(cfg["slug"])

    targets     = ck["prioritization_result"]["targets"]
    pheno       = ck.get("phenotype_result", {})
    chem        = ck.get("chemistry_result", {})
    gwas_set    = set(pheno.get("gwas_genes", []))
    known       = cfg["known"]
    prog_gammas = _prog_gamma(t3, disease_key)

    # OTA target lookup
    def _dom_prog(t: dict) -> str | None:
        return (t.get("program_drivers") or {}).get("top_program")

    all_progs = sorted(
        {p for t in targets if (p := _dom_prog(t))},
        key=lambda p: -prog_gammas.get(p, 0.0),
    )
    prog_color = {p: _PROG_PALETTE[i % len(_PROG_PALETTE)] for i, p in enumerate(all_progs)}

    ota_genes: dict[str, dict] = {t["target_gene"]: t for t in targets}

    # GPS data
    disease_rev   = chem.get("gps_disease_reversers", []) or []
    prog_rev_dict = chem.get("gps_program_reversers",  {}) or {}
    screened_progs = [p for p in all_progs if prog_rev_dict.get(p)]

    # ── Load full fingerprint landscape ───────────────────────────────────────
    print(f"Loading SVD fingerprints for {cfg['dataset_id']}…")
    fp_mat, fp_names = _load_svd(cfg["dataset_id"])
    fp_r_all         = _load_fp_r(cfg["dataset_id"])

    n_genes = len(fp_names)
    all_r   = np.array([fp_r_all.get(fp_names[i], float("nan")) for i in range(n_genes)])

    print(f"  {n_genes} KOs × {fp_mat.shape[1]} SVD dims")
    print(f"  Clustering by disease-weighted fingerprint similarity…")
    order   = _cluster_order(fp_mat, fp_r=all_r)
    x_pos   = np.empty(n_genes, dtype=int)
    x_pos[order] = np.arange(len(order))   # gene index → x position

    name_to_xi = {name: x_pos[i] for i, name in enumerate(fp_names)}

    # Build position arrays
    all_xi   = np.array([x_pos[i] for i in range(n_genes)])
    is_ota   = np.array([fp_names[i] in ota_genes for i in range(n_genes)])
    is_known = np.array([fp_names[i] in known      for i in range(n_genes)])

    # ── Layout: main gene landscape + GPS compound panel ─────────────────────
    has_gps = bool(disease_rev or any(prog_rev_dict.values()))
    fig = plt.figure(figsize=(18, 9 if has_gps else 6))
    gs  = GridSpec(
        2 if has_gps else 1, 1, figure=fig,
        hspace=0.35,
        height_ratios=[4, 2] if has_gps else [1],
    )

    ax_main = fig.add_subplot(gs[0])

    # ── Main panel: full fingerprint landscape ────────────────────────────────
    # Background library genes (grey, small)
    bg_mask = ~is_ota & ~np.isnan(all_r)
    ax_main.scatter(
        all_xi[bg_mask], all_r[bg_mask],
        c="#CCCCCC", s=3, alpha=0.40, linewidths=0, zorder=1, rasterized=True,
    )

    # OTA target genes — colored by program
    for i in range(n_genes):
        if not is_ota[i] or math.isnan(all_r[i]):
            continue
        gene = fp_names[i]
        t    = ota_genes[gene]
        prog = _dom_prog(t)
        col  = prog_color.get(prog, "#9E9E9E")
        tier = t.get("dominant_tier", "")
        size = 60 if tier == "Tier1_Interventional" else 22
        ec   = "#B71C1C" if gene in known else "white"
        ew   = 1.6 if gene in known else 0.4
        ax_main.scatter(all_xi[i], all_r[i], c=col, s=size,
                        edgecolors=ec, linewidths=ew, alpha=0.92, zorder=3)

    # Label known targets + most extreme OTA genes
    ota_r_vals = sorted([all_r[i] for i in range(n_genes)
                         if is_ota[i] and not math.isnan(all_r[i])], key=abs)
    r_threshold_label = ota_r_vals[-max(1, int(len(ota_r_vals) * 0.15))] if ota_r_vals else 0.0
    labeled: set[str] = set()
    for i in range(n_genes):
        if not is_ota[i] or math.isnan(all_r[i]):
            continue
        gene = fp_names[i]
        if gene in known or abs(all_r[i]) >= abs(r_threshold_label):
            if gene not in labeled:
                col = "#B71C1C" if gene in known else "#222222"
                fw  = "bold"   if gene in known else "normal"
                va  = "bottom" if all_r[i] >= 0 else "top"
                dy  = 0.018 if all_r[i] >= 0 else -0.018
                ax_main.annotate(gene, (all_xi[i], all_r[i] + dy),
                                 ha="center", va=va,
                                 fontsize=5.5, color=col, fontweight=fw, zorder=5)
                labeled.add(gene)

    ax_main.axhline(0, color="#888888", lw=0.7, ls="--", zorder=2)
    ax_main.set_ylabel("Fingerprint r\n(KO vs disease DEG)", fontsize=9)
    ax_main.set_ylim(-1.0, 1.0)
    ax_main.set_xlim(-50, n_genes + 50)
    ax_main.set_facecolor("#FAFAFA")
    ax_main.spines["top"].set_visible(False)
    ax_main.spines["right"].set_visible(False)
    ax_main.set_xticks([])

    ax_main.axhspan(-1.0, 0, color="#E8F5E9", alpha=0.3, zorder=0)
    ax_main.axhspan( 0, 1.0, color="#FFEBEE", alpha=0.2, zorder=0)
    ax_main.text(60, -0.95, "← KO reverses disease (therapeutic)",
                 fontsize=7, color="#2E7D32", va="bottom")
    ax_main.text(60,  0.90, "KO amplifies disease →",
                 fontsize=7, color="#C62828", va="top")

    ax_main.set_title(
        f"{cfg['label']}  —  long island plot\n"
        f"{n_genes} KOs · x = disease-weighted fingerprint clustering · "
        f"{sum(is_ota)} OTA targets highlighted",
        fontsize=10, fontweight="bold",
    )

    prog_h = [mpatches.Patch(color=prog_color[p],
                              label=f"{p.split('_')[-1]} γ={prog_gammas.get(p,0):+.2f}")
              for p in all_progs]
    misc_h = [
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#CCCCCC",
               markersize=5, label="Full Perturb-seq library"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#1565C0",
               markersize=7, label="OTA target (large=Tier1)"),
        mpatches.Patch(facecolor="none", edgecolor="#B71C1C", linewidth=1.6,
                       label="Validated target"),
    ]
    ax_main.legend(handles=prog_h + misc_h, loc="upper right",
                   fontsize=6, ncol=3, framealpha=0.9)

    # ── GPS compound panel ────────────────────────────────────────────────────
    # GPS compounds do NOT have transcriptional profiles in the Perturb-seq space
    # (novel Enamine Z-IDs, not in L1000/CMap). x-position in the gene landscape
    # is undefined. Instead: horizontal bar chart ranked by |z_rges|, coloured by
    # which screen produced the hit (disease-state vs program-specific).
    if has_gps:
        ax_gps = fig.add_subplot(gs[1])

        # Build compound table: (label, z_rges, colour, category)
        gps_rows: list[tuple[str, float, str, str]] = []
        for c in disease_rev:
            cid = c.get("compound_id", "?")
            z   = c.get("z_rges", c.get("rges", 0.0))
            gps_rows.append((cid[:14], z, "#FFC107", "disease reversal"))

        for prog, hits in prog_rev_dict.items():
            pcol  = prog_color.get(prog, "#555555")
            pname = prog.split("_")[-1]
            for c in (hits or []):
                cid = c.get("compound_id", "?")
                z   = c.get("z_rges", c.get("rges", 0.0))
                gps_rows.append((cid[:14], z, pcol, pname))

        # Deduplicate: keep most-negative z per compound
        seen_cids: dict[str, tuple] = {}
        for row in gps_rows:
            existing = seen_cids.get(row[0])
            if existing is None or row[1] < existing[1]:
                seen_cids[row[0]] = row
        gps_rows = sorted(seen_cids.values(), key=lambda r: r[1])  # most negative first

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

        # Category legend
        cat_labels = {"disease reversal": "#FFC107"}
        for prog in screened_progs:
            cat_labels[prog.split("_")[-1]] = prog_color.get(prog, "#555555")
        cat_h = [mpatches.Patch(color=c, label=lbl) for lbl, c in cat_labels.items()]
        ax_gps.legend(handles=cat_h, fontsize=6, loc="lower right")

    plt.tight_layout()
    if out_path is None:
        out_path = f"outputs/long_island_{disease_key}.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--disease", choices=["ra", "cad"], required=True)
    ap.add_argument("--out",     type=str, default=None)
    args = ap.parse_args()
    plot_long_island(args.disease, out_path=args.out)
