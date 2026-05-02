"""
Long-island multi-panel plot — transcriptional fingerprint landscape.

Panel 0  — Disease state: all KOs, y = fingerprint_r (KO vs disease DEG).
Panel 1…N — One per SVD program with ≥MIN_PROG_TARGETS OTA genes:
             background grey, program OTA genes highlighted,
             GPS program reversers placed at convergence-target x-position.

X-axis (shared): all Perturb-seq KOs ordered by hierarchical clustering
restricted to the TOP_GWAS_COMPONENTS SVD dimensions ranked by |γ_{P→disease}|
(OT L2G GWAS signal).  Genes that perturb the same GWAS-relevant transcriptional
axes cluster together (islands).

Coloring: each OTA target is coloured by the GWAS-aligned SVD component where
its z-scored Vt loading is most extreme (|z| > GWAS_PROG_Z_THRESHOLD).  Genes
below threshold get grey ("no dominant GWAS program").  This is computed fresh
from the SVD matrix — independent of the checkpoint's top_program assignment.

GPS compound placement in program panels:
  x = score-weighted mean x-position of the compound's convergence target genes
  y = mean fingerprint_r of those targets

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
MIN_PROG_GAMMA   = 0.02  # minimum |γ_{program→disease}| to render a program panel
                         # (LDSC τ-normalised values peak at ~0.10; OT L2G values at 0.1–0.5)

TOP_GWAS_COMPONENTS  = 15   # SVD components (ranked by gwas_t) used for x-axis clustering
GWAS_PROG_Z_THRESHOLD = 1.0 # |z-score| on a GWAS-aligned component to assign membership
NMF_N_COMPONENTS     = 20   # NMF programs to extract in --mode nmf


# ─── loaders ──────────────────────────────────────────────────────────────────

def _load_ck(slug: str) -> dict:
    p = pathlib.Path(f"data/checkpoints/{slug}__tier4.json")
    if not p.exists():
        sys.exit(f"Tier-4 checkpoint not found: {p}")
    return json.load(open(p))


def _load_t3(slug: str) -> dict:
    p = pathlib.Path(f"data/checkpoints/{slug}__tier3.json")
    return json.load(open(p)) if p.exists() else {}


def _load_gwas_aligned_programs(dataset_id: str, top_k: int) -> list[dict]:
    """Load pre-saved gwas_t-ranked program list (written by perturbseq_server)."""
    p = pathlib.Path(f"data/perturbseq/{dataset_id}/gwas_aligned_programs.json")
    if not p.exists():
        return []
    entries = json.load(open(p))
    # Sort by |gwas_t_stat| — most strongly GWAS-aligned first
    entries.sort(key=lambda e: abs(e["gwas_t_stat"]), reverse=True)
    return entries[:top_k]


def _load_svd(dataset_id: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Returns (fp_mat, Vt, pert_names).
    fp_mat = Vt.T  shape (n_perts × k) — used for clustering.
    Vt             shape (k × n_perts)  — used for z-score thresholding.
    """
    d = np.load(f"data/perturbseq/{dataset_id}/svd_loadings.npz")
    Vt = d["Vt"]           # (k × n_perts)
    return Vt.T, Vt, list(d["pert_names"])


def _svd_component_index(prog_name: str) -> int | None:
    """'RA_SVD_C07' → 6  (0-indexed).  Returns None on parse failure."""
    try:
        return int(prog_name.rsplit("C", 1)[-1]) - 1
    except (ValueError, IndexError):
        return None


def _gwas_prog_assignment(
    fp_mat: np.ndarray,
    pert_names: list[str],
    gwas_col_indices: list[int],
    gwas_prog_names: list[str],
) -> tuple[dict[str, str | None], dict[str, float]]:
    """
    Assign each Perturb-seq gene to its dominant GWAS-aligned SVD component.

    For each gene, z-score its loading on every GWAS-aligned component
    (z = (Vt[c,g] - mean_c) / std_c across all perturbations).
    Assign to the component with the largest |z| if |z| > GWAS_PROG_Z_THRESHOLD,
    else assign None ("no dominant GWAS program").

    Returns:
        gene_prog  — {gene_name: program_id | None}
        gene_z     — {gene_name: z-score on its assigned component} (for sorting)
    """
    if not gwas_col_indices:
        empty = {g: None for g in pert_names}
        return empty, {g: 0.0 for g in pert_names}

    gwas_fp  = fp_mat[:, gwas_col_indices]           # (n_perts × top_k)
    col_mean = gwas_fp.mean(axis=0)
    col_std  = gwas_fp.std(axis=0)
    col_std[col_std < 1e-12] = 1.0
    gwas_z   = (gwas_fp - col_mean) / col_std        # (n_perts × top_k)

    abs_z  = np.abs(gwas_z)
    best_c = abs_z.argmax(axis=1)                    # (n_perts,)
    best_z = abs_z[np.arange(len(pert_names)), best_c]
    # Signed z on best component — used for within-program sort direction
    signed_z = gwas_z[np.arange(len(pert_names)), best_c]

    gene_prog: dict[str, str | None] = {}
    gene_z:    dict[str, float]      = {}
    for i, gene in enumerate(pert_names):
        if best_z[i] >= GWAS_PROG_Z_THRESHOLD:
            gene_prog[gene] = gwas_prog_names[best_c[i]]
            gene_z[gene]    = float(signed_z[i])
        else:
            gene_prog[gene] = None
            gene_z[gene]    = 0.0
    return gene_prog, gene_z


def _run_nmf_assignment(
    Vt: np.ndarray,
    pert_names: list[str],
    disease_key: str,
    gwas_anchor_genes: set[str],
    n_components: int = NMF_N_COMPONENTS,
) -> tuple[dict[str, str | None], list[str], dict[str, float]]:
    """
    Run NMF on |Vt.T| (n_perts × k) to get sparse, parts-based program assignments.

    Unlike SVD (dense/orthogonal, every gene loads on all components), NMF
    produces sparse non-negative W where each perturbation loads primarily on
    one or few components — argmax is a meaningful "dominant program."

    Ranks programs by mean loading of GWAS anchor genes: the component where
    GWAS-perturbed genes cluster most strongly = the most disease-relevant axis.

    Returns:
        gene_prog      — {gene: 'RA_NMF_C01' | None}
        prog_order     — top-k program names sorted by GWAS relevance (desc)
        prog_gwas_score — {prog: mean_gwas_loading}  (used as proxy for γ_{P→disease})
    """
    from sklearn.decomposition import NMF

    X = np.abs(Vt.T)  # (n_perts × k) — abs makes non-negative; captures loading magnitude
    model = NMF(n_components=n_components, init="nndsvda", random_state=42, max_iter=500)
    W = model.fit_transform(X)  # (n_perts × n_components)

    assignments = W.argmax(axis=1)  # argmax is meaningful with sparse NMF

    prefix = disease_key.upper()
    comp_names = [f"{prefix}_NMF_C{c+1:02d}" for c in range(n_components)]

    gene_prog: dict[str, str | None] = {
        gene: comp_names[assignments[i]] for i, gene in enumerate(pert_names)
    }

    gwas_idx = [i for i, g in enumerate(pert_names) if g in gwas_anchor_genes]
    if gwas_idx:
        gwas_scores = W[gwas_idx, :].mean(axis=0)
    else:
        gwas_scores = W.mean(axis=0)

    prog_gwas_score = {comp_names[c]: float(gwas_scores[c]) for c in range(n_components)}
    prog_order = sorted(comp_names, key=lambda p: prog_gwas_score[p], reverse=True)
    return gene_prog, prog_order, prog_gwas_score


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

    # Override SVD program gammas with signed chi-square τ (CAD only; RA has no
    # sign variance so falls back to checkpoint values).
    try:
        import sys as _sys
        import pathlib as _pl
        _root = _pl.Path(__file__).parent.parent
        if str(_root) not in _sys.path:
            _sys.path.insert(0, str(_root))
        from pipelines.ldsc.gamma_loader import get_svd_program_gammas
        svd_gammas = get_svd_program_gammas(disease_key.upper())
        for prog, est in svd_gammas.items():
            if est.get("gamma") is not None:
                out[prog] = est["gamma"]
    except Exception:
        pass  # non-fatal; old checkpoint gammas remain

    return out


# ─── disease-weighted clustering ──────────────────────────────────────────────

def _build_layout(
    fp_mat: np.ndarray,
    all_r: np.ndarray,
    n_islands: int,
    gene_prog: dict[str, str | None],
    gene_z: dict[str, float],
    pert_names: list[str],
    gwas_progs_sorted: list[str],
    gene_gamma: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build x-axis ordering: GWAS-program blocks, within each block sorted by
    OTA gamma ascending (most protective/therapeutic leftmost).

    x = OTA γ (causal, genetics-based)
    y = fingerprint_r (transcriptional, RNA-based)

    Two independent axes: a gene bottom-left within its block has both
    transcriptional similarity to disease reversal AND genetic causal evidence
    for protection — the strongest therapeutic candidates.

    Genes with no OTA gamma (library-only) sort to x=0 within their block.
    Unassigned genes (grey, no dominant GWAS axis) appended at right, also
    sorted by OTA gamma.

    Returns:
        x_pos          — (n_perts,) int  position of each gene on x-axis
        cluster_labels — (n_perts,) int  cluster id (1…n_programs+1)
    """
    n = len(pert_names)
    prog_to_label = {p: i + 1 for i, p in enumerate(gwas_progs_sorted)}
    unassigned_label = len(gwas_progs_sorted) + 1

    def _sort_val(i: int) -> float:
        g = pert_names[i]
        if g in gene_gamma:
            return gene_gamma[g]           # OTA target: position by OTA γ
        # Library gene: fall back to fingerprint_r so background cloud
        # spreads horizontally rather than stacking at x=0.
        return all_r[i] if not math.isnan(all_r[i]) else 0.0

    ordered: list[int] = []

    for prog in gwas_progs_sorted:
        members = [
            (i, _sort_val(i))
            for i in range(n)
            if gene_prog.get(pert_names[i]) == prog
        ]
        members.sort(key=lambda t: t[1])   # ascending: protective left, disease right
        ordered.extend(idx for idx, _ in members)

    unassigned = [
        (i, _sort_val(i))
        for i in range(n)
        if gene_prog.get(pert_names[i]) is None
    ]
    unassigned.sort(key=lambda t: t[1])
    ordered.extend(idx for idx, _ in unassigned)

    x_pos = np.empty(n, dtype=int)
    for xi, gene_idx in enumerate(ordered):
        x_pos[gene_idx] = xi

    cluster_labels = np.array([
        prog_to_label.get(gene_prog.get(pert_names[i]), unassigned_label)
        for i in range(n)
    ], dtype=int)

    return x_pos, cluster_labels


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
                     n_islands: int = 12, mode: str = "svd") -> None:
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

    ota_genes: dict[str, dict] = {t["target_gene"]: t for t in targets}

    disease_rev   = chem.get("gps_disease_reversers", []) or []
    prog_rev_dict = chem.get("gps_program_reversers", {}) or {}

    # ── Load SVD + fingerprint r ──────────────────────────────────────────────
    print(f"Loading SVD fingerprints for {cfg['dataset_id']}…")
    fp_mat, Vt, fp_names = _load_svd(cfg["dataset_id"])
    fp_r_all              = _load_fp_r(cfg["dataset_id"])

    n_genes = len(fp_names)
    all_r   = np.array([fp_r_all.get(fp_names[i], float("nan")) for i in range(n_genes)])

    # ── Program selection: SVD (gwas_t threshold) or NMF (argmax) ────────────
    if mode == "nmf":
        print(f"  Running NMF (n={NMF_N_COMPONENTS}) on |Vt.T| for sparse program assignment…")
        gwas_anchor_genes = {t["target_gene"] for t in targets
                             if t.get("dominant_tier", "").startswith("Tier1")}
        gene_gwas_prog, nmf_prog_order, nmf_gwas_score = _run_nmf_assignment(
            Vt, fp_names, disease_key, gwas_anchor_genes, NMF_N_COMPONENTS
        )
        gwas_progs_sorted = nmf_prog_order[:TOP_GWAS_COMPONENTS]
        print(f"  Top NMF programs (by GWAS anchor loading): "
              f"{', '.join(p.split('_')[-1] for p in gwas_progs_sorted)}")
        # Use NMF gwas_score as proxy for γ_{program→disease} in panel filter + title
        prog_gammas_display = nmf_gwas_score
    else:
        # SVD mode: load gwas_t-ranked programs, z-threshold assignment
        gwas_ranked = _load_gwas_aligned_programs(cfg["dataset_id"], TOP_GWAS_COMPONENTS)
        if not gwas_ranked:
            gwas_ranked = [
                {"program_id": p, "gwas_t_stat": prog_gammas.get(p, 0.0)}
                for p in sorted(prog_gammas, key=lambda p: abs(prog_gammas[p]), reverse=True)
                if "_SVD_" in p
            ][:TOP_GWAS_COMPONENTS]

        gwas_progs_sorted = [e["program_id"] for e in gwas_ranked]

        gwas_col_indices: list[int] = []
        for p in gwas_progs_sorted:
            idx = _svd_component_index(p)
            if idx is not None and 0 <= idx < Vt.shape[0]:
                gwas_col_indices.append(idx)

        print(f"  GWAS-aligned components ({len(gwas_progs_sorted)}, by gwas_t): "
              f"{', '.join(p.split('_')[-1] for p in gwas_progs_sorted)}")

        gene_gwas_prog, gene_z = _gwas_prog_assignment(
            fp_mat, fp_names, gwas_col_indices, gwas_progs_sorted
        )
        prog_gammas_display = prog_gammas

    # Build color palette from selected programs
    all_progs  = gwas_progs_sorted
    prog_color = {p: _PROG_PALETTE[i % len(_PROG_PALETTE)] for i, p in enumerate(all_progs)}

    # ── Build gene_gamma lookup (OTA γ per gene, for x-axis within blocks) ───
    # x = OTA γ (causal, genetics-based); y = fingerprint_r (transcriptional)
    gene_gamma: dict[str, float] = {
        t["target_gene"]: float(t["ota_gamma"])
        for t in targets
        if t.get("ota_gamma") is not None and not math.isnan(float(t["ota_gamma"]))
    }

    # ── Build layout: program-sorted x-axis, within-block by OTA γ ───────────
    print("  Building program-sorted x-axis layout (x=OTA γ within blocks)…")
    # gene_z only exists in SVD mode; NMF uses argmax so no signed z needed
    _gene_z = gene_z if mode == "svd" else {g: 0.0 for g in fp_names}
    x_pos, cluster_labels = _build_layout(
        fp_mat, all_r, n_islands,
        gene_prog=gene_gwas_prog, gene_z=_gene_z,
        pert_names=fp_names, gwas_progs_sorted=gwas_progs_sorted,
        gene_gamma=gene_gamma,
    )
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

    # ── Determine which programs get a subpanel ───────────────────────────────
    # A gene appears in program P's panel if its OTA contribution |β(gene→P)×γ(P)|
    # clears a minimum threshold. This is more informative than winner-takes-all
    # (which hides high-γ programs like C15 with moderate β) and avoids the all-same-
    # pattern of full top_programs (where every gene appears in every panel).
    PROG_CONTRIB_MIN = 0.03   # minimum |β × γ| for program panel membership
    prog_ota_with_gamma: dict[str, list[tuple[str, float, float]]] = {}
    gwas_prog_set = set(all_progs)
    for t in targets:
        gene = t["target_gene"]
        gamma = gene_gamma.get(gene)
        r = fp_r_all.get(gene)
        if gamma is None or r is None:
            continue
        tp = t.get("top_programs") or {}
        assigned = False
        for prog, b in tp.items():
            if prog not in gwas_prog_set:
                continue
            prog_g = prog_gammas.get(prog, 0.0)
            if abs(b * prog_g) >= PROG_CONTRIB_MIN:
                prog_ota_with_gamma.setdefault(prog, []).append((gene, gamma, r))
                assigned = True
        if not assigned:
            # Fallback: dominant z-score assignment so gene still appears somewhere
            dom = gene_gwas_prog.get(gene)
            if dom:
                prog_ota_with_gamma.setdefault(dom, []).append((gene, gamma, r))

    if mode == "nmf":
        # NMF: top-8 programs by GWAS loading with enough OTA targets (no γ threshold)
        show_progs = [p for p in all_progs
                      if len(prog_ota_with_gamma.get(p, [])) >= MIN_PROG_TARGETS]
    else:
        show_progs = [p for p in all_progs
                      if len(prog_ota_with_gamma.get(p, [])) >= MIN_PROG_TARGETS
                      and abs(prog_gammas.get(p, 0.0)) >= MIN_PROG_GAMMA]

    # ── Layout: square grid of per-program panels only ───────────────────────
    n_cols   = min(4, len(show_progs)) if show_progs else 1
    n_rows   = math.ceil(len(show_progs) / n_cols) if show_progs else 1
    panel_sz = 4.5  # inches per panel (square)
    fig = plt.figure(figsize=(n_cols * panel_sz, n_rows * panel_sz))
    gs  = GridSpec(n_rows, n_cols, figure=fig, hspace=0.45, wspace=0.35)

    # Shared x-range: symmetric around 0, driven by global max |γ| across all panels
    all_gammas_shown = [
        gm
        for prog in show_progs
        for _, gm, _ in prog_ota_with_gamma.get(prog, [])
    ]
    x_abs_max = max((abs(g) for g in all_gammas_shown), default=1.0)
    x_abs_max = math.ceil(x_abs_max * 10) / 10 * 1.1   # 10% headroom, rounded up
    shared_xlim = (-x_abs_max, x_abs_max)

    # ═════════════════════════════════════════════════════════════════════════
    # Per-program panels: x = OTA γ (actual value), y = fingerprint_r
    # ═════════════════════════════════════════════════════════════════════════
    for pi, prog in enumerate(show_progs):
        row, col = divmod(pi, n_cols)
        ax = fig.add_subplot(gs[row, col])
        pcol  = prog_color.get(prog, "#555555")
        gamma_prog = prog_gammas_display.get(prog, float("nan"))

        genes_in_prog = prog_ota_with_gamma.get(prog, [])  # (gene, gamma, r)

        # OTA targets: x = actual OTA gamma, y = fingerprint_r
        # Density-aware size; radial alpha (distance from origin → vivid, center → faint).
        n_dots = len(genes_in_prog)
        scale    = max(0.25, 1.0 - 0.006 * n_dots)   # 1.0 at n=0, ~0.25 at n=125
        sz_t1    = max(12, int(55 * scale))            # Tier1: 55→12
        sz_other = max(5,  int(22 * scale))            # others: 22→5

        gamma_norm = x_abs_max if x_abs_max > 0 else 1.0

        rv = sorted([(abs(r), gene) for gene, _, r in genes_in_prog])
        label_set  = {g for _, g in rv[max(0, int(len(rv) * 0.80)):]} | known

        for gene, gm, r in genes_in_prog:
            is_k      = gene in known
            highlight = gene in label_set
            size  = sz_t1 if ota_genes.get(gene, {}).get("dominant_tier","") == "Tier1_Interventional" else sz_other
            ec    = "#B71C1C" if is_k else "none"
            ew    = 1.4       if is_k else 0.0
            if highlight:
                alpha = 0.95
            else:
                dist  = math.sqrt((gm / gamma_norm) ** 2 + r ** 2) / math.sqrt(2)
                alpha = max(0.06, min(0.90, 0.10 + 0.80 * dist))
            ax.scatter(gm, r, c=pcol, s=size,
                       edgecolors=ec, linewidths=ew, alpha=alpha, zorder=4)

        for gene, gm, r in genes_in_prog:
            if gene in label_set:
                va = "bottom" if r >= 0 else "top"
                dy = 0.018 if r >= 0 else -0.018
                ax.annotate(gene, (gm, r + dy), ha="center", va=va,
                            fontsize=5, color="#B71C1C" if gene in known else "#222222",
                            fontweight="bold" if gene in known else "normal", zorder=6)

        ax.axvline(0, color="#888888", lw=0.8, ls="--", zorder=3)
        ax.axhline(0, color="#888888", lw=0.8, ls="--", zorder=3)
        ax.set_xlim(*shared_xlim)
        ax.set_ylim(-1.05, 1.05)
        ax.set_ylabel("Fingerprint r\n(KO vs disease DEG)", fontsize=7)
        ax.set_xlabel("OTA γ  (← protective   |   disease-amplifying →)", fontsize=7)
        ax.set_facecolor("white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if mode == "nmf":
            score_str = f"gwas_load={gamma_prog:.3f}" if not math.isnan(gamma_prog) else ""
            ax.set_title(
                f"{prog}  ·  {score_str}  ·  {len(genes_in_prog)} OTA targets\n"
                f"(x = SVD-based OTA γ,  NMF used for grouping only)",
                fontsize=7, fontweight="bold", pad=3,
            )
        else:
            gwas_t_val = gwas_ranked[all_progs.index(prog)]["gwas_t_stat"] if prog in all_progs else float("nan")
            gamma_str  = f"γ={gamma_prog:+.2f}" if not math.isnan(gamma_prog) else ""
            ax.set_title(
                f"{prog}  ·  gwas_t={gwas_t_val:+.1f}  {gamma_str}  ·  {len(genes_in_prog)} OTA targets",
                fontsize=8, fontweight="bold", pad=3,
            )

    default_path = f"outputs/long_island_{disease_key}_{mode}.png"
    plt.savefig(out_path or default_path, dpi=160, bbox_inches="tight")
    print(f"Saved → {out_path or default_path}")


def plot_explainer(out_path: str = "outputs/long_island_explainer.png") -> None:
    """
    Mock quadrant diagram explaining the x/y axes of the per-program panels.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np

    fig, ax = plt.subplots(figsize=(5, 5))

    # Quadrant shading
    ax.fill_between([-3, 0], [0, 0], [1, 1],  color="#D6E4F0", alpha=0.5, zorder=0)  # top-left
    ax.fill_between([0,  3], [0, 0], [1, 1],  color="#FADBD8", alpha=0.5, zorder=0)  # top-right
    ax.fill_between([-3, 0], [-1, -1], [0, 0], color="#D5F5E3", alpha=0.6, zorder=0)  # bottom-left ★
    ax.fill_between([0,  3], [-1, -1], [0, 0], color="#FEF9E7", alpha=0.5, zorder=0)  # bottom-right

    # Quadrant labels
    kw = dict(fontsize=8, ha="center", va="center", style="italic")
    ax.text(-1.5,  0.6, "Transcriptionally\nprotective\nbut no genetic\ncausal evidence", color="#1A5276", **kw)
    ax.text( 1.5,  0.6, "Transcriptionally\ndisease-amplifying\n& genetically\nharmful", color="#922B21", **kw)
    ax.text(-1.5, -0.6, "★ Best candidates\nGenetically protective\n& transcriptionally\nreverses disease", color="#1E8449",
            fontsize=8, ha="center", va="center", fontweight="bold")
    ax.text( 1.5, -0.6, "Transcriptionally\nreverses disease\nbut genetic evidence\nfor harm — flag", color="#7D6608", **kw)

    # Axes
    ax.axvline(0, color="#555555", lw=1.2, zorder=3)
    ax.axhline(0, color="#555555", lw=1.2, zorder=3)
    ax.set_xlim(-3, 3); ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("OTA γ  (causal / genetics-based)\n← KO protects from disease   |   KO drives disease →",
                  fontsize=8)
    ax.set_ylabel("Fingerprint r  (transcriptional / RNA-based)\n← KO reverses disease DEG   |   KO mimics disease DEG →",
                  fontsize=8)
    ax.set_title("Per-GWAS-program panel — axis guide\n(one panel per GWAS-aligned SVD component)",
                 fontsize=9, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--disease",   choices=["ra", "cad"], required=False)
    ap.add_argument("--out",       type=str, default=None)
    ap.add_argument("--n_islands", type=int, default=12)
    ap.add_argument("--mode",      choices=["svd", "nmf"], default="svd",
                    help="Program assignment method: svd (z-threshold) or nmf (argmax)")
    ap.add_argument("--explainer", action="store_true",
                    help="Generate the axis explainer mock figure instead")
    args = ap.parse_args()
    if args.explainer:
        plot_explainer(out_path=args.out or "outputs/long_island_explainer.png")
    else:
        plot_long_island(args.disease, out_path=args.out, n_islands=args.n_islands,
                         mode=args.mode)
