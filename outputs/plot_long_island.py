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


def _load_genetic_nmf(dataset_id: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load GeneticNMF output. Returns (fp_mat, Vt, pert_names).
    fp_mat = Vt.T  shape (n_perts × k) — used for clustering (same role as SVD fp_mat).
    Vt             shape (k × n_perts)  — perturbation loadings per program.
    """
    p = pathlib.Path(f"data/perturbseq/{dataset_id}/genetic_nmf_loadings.npz")
    if not p.exists():
        sys.exit(f"genetic_nmf_loadings.npz not found: {p}\n"
                 f"Run: python -m pipelines.genetic_nmf {dataset_id}")
    d = np.load(p)
    Vt = d["Vt"]           # (k × n_perts)
    return Vt.T, Vt, list(d["pert_names"])


def _genetic_nmf_assignment(
    Vt: np.ndarray,
    pert_names: list[str],
    prog_names: list[str],
    z_threshold: float = 1.0,
) -> tuple[dict[str, str | None], dict[str, float]]:
    """
    Assign each perturbation gene to its dominant GeneticNMF program.

    For each gene g: z-score its column in Vt across all programs,
    assign to the program with the largest |z| if |z| > z_threshold, else None.
    Argmax on the raw NMF loadings also works (non-negative) but z-scoring
    makes the threshold comparable to SVD mode.
    """
    col      = Vt.T                              # (n_perts × k)
    col_mean = col.mean(axis=0)
    col_std  = col.std(axis=0)
    col_std[col_std < 1e-12] = 1.0
    col_z    = (col - col_mean) / col_std        # (n_perts × k)

    abs_z    = np.abs(col_z)
    best_k   = abs_z.argmax(axis=1)             # (n_perts,)
    best_z   = abs_z[np.arange(len(pert_names)), best_k]
    signed_z = col_z[np.arange(len(pert_names)), best_k]

    gene_prog: dict[str, str | None] = {}
    gene_z:    dict[str, float]      = {}
    for i, gene in enumerate(pert_names):
        if best_z[i] >= z_threshold:
            gene_prog[gene] = prog_names[best_k[i]]
            gene_z[gene]    = float(signed_z[i])
        else:
            gene_prog[gene] = None
            gene_z[gene]    = 0.0
    return gene_prog, gene_z


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

    # Override with S-LDSC signed gammas (SVD and GeneticNMF) when available.
    try:
        import sys as _sys
        import pathlib as _pl
        _root = _pl.Path(__file__).parent.parent
        if str(_root) not in _sys.path:
            _sys.path.insert(0, str(_root))
        from pipelines.ldsc.gamma_loader import get_svd_program_gammas, get_genetic_nmf_program_gammas, get_locus_program_gammas
        nmf_gammas = get_genetic_nmf_program_gammas(disease_key.upper())
        for prog, est in nmf_gammas.items():
            if est.get("gamma") is not None:
                out[prog] = est["gamma"]
        svd_gammas = get_svd_program_gammas(disease_key.upper())
        for prog, est in svd_gammas.items():
            if est.get("gamma") is not None:
                out[prog] = est["gamma"]
        locus_gammas = get_locus_program_gammas(disease_key.upper())
        for prog, est in locus_gammas.items():
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

    # ── Program selection: SVD / NMF (legacy) / GeneticNMF (pre-computed) ────
    if mode == "genetic_nmf":
        print(f"Loading GeneticNMF programs for {cfg['dataset_id']}…")
        fp_mat, Vt, fp_names = _load_genetic_nmf(cfg["dataset_id"])
        n_genes = len(fp_names)
        all_r   = np.array([fp_r_all.get(fp_names[i], float("nan")) for i in range(n_genes)])

        n_progs    = Vt.shape[0]
        prefix     = disease_key.upper()
        prog_names = [f"{prefix}_GeneticNMF_C{c+1:02d}" for c in range(n_progs)]

        gene_gwas_prog, gene_z = _genetic_nmf_assignment(
            Vt, fp_names, prog_names, z_threshold=GWAS_PROG_Z_THRESHOLD,
        )

        # Rank programs by |γ| from S-LDSC; fall back to mean Vt loading
        nmf_gammas = {p: prog_gammas.get(p, 0.0) for p in prog_names}
        gwas_progs_sorted = sorted(
            prog_names,
            key=lambda p: abs(nmf_gammas.get(p, 0.0)),
            reverse=True,
        )[:TOP_GWAS_COMPONENTS]
        prog_gammas_display = nmf_gammas
        print(f"  Top GeneticNMF programs (by |γ|): "
              f"{', '.join(p.split('_C')[-1] for p in gwas_progs_sorted)}")

    elif mode == "nmf":
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
    # gene_z exists in SVD and genetic_nmf modes; legacy NMF uses argmax only
    _gene_z = gene_z if mode in ("svd", "genetic_nmf") else {g: 0.0 for g in fp_names}
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

    if mode == "genetic_nmf":
        # GeneticNMF: use U_scaled (response-gene expression loadings) to assign OTA
        # targets to programs.  Vt is distorted by large enhancer perturbations (~10-40×
        # higher than coding genes) that monopolise C16/C08; U_scaled is not affected.
        # 1153/1293 OTA targets are also response genes → U_scaled covers >89%.
        # Fallback: raw Vt argmax for the ~11% without U_scaled coverage.
        npz_path = pathlib.Path(f"data/perturbseq/{cfg['dataset_id']}/genetic_nmf_loadings.npz")
        _nmf_d   = np.load(npz_path)
        U_scaled_full = _nmf_d["U_scaled"]          # (n_response_genes × k)
        response_names = list(_nmf_d["gene_names"])  # response gene order
        resp_to_idx = {g: i for i, g in enumerate(response_names)}
        pert_to_idx = {name: i for i, name in enumerate(fp_names)}

        for t in targets:
            gene  = t["target_gene"]
            gamma = gene_gamma.get(gene)
            r     = fp_r_all.get(gene)
            if gamma is None or r is None:
                continue
            if gene in resp_to_idx:
                # Primary: assign by gene's expression loading on each NMF program
                u_vec  = U_scaled_full[resp_to_idx[gene], :]   # (k,)
                best_k = int(u_vec.argmax())
            else:
                # Fallback: raw Vt argmax (not column-normalised; few genes)
                j = pert_to_idx.get(gene)
                if j is None:
                    continue
                best_k = int(Vt[:, j].argmax())
            best_prog = (
                [p for p in gwas_prog_set if p.endswith(f"_C{best_k+1:02d}")]
                or [None]
            )[0]
            if best_prog:
                prog_ota_with_gamma.setdefault(best_prog, []).append((gene, gamma, r))
    else:
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
                # Fallback: dominant z-score assignment
                dom = gene_gwas_prog.get(gene)
                if dom:
                    prog_ota_with_gamma.setdefault(dom, []).append((gene, gamma, r))

    if mode == "genetic_nmf":
        # No γ floor; lower target threshold (programs like C16/C08 have few OTA targets
        # because GeneticNMF separates enhancer-driven vs coding-gene programs)
        GENETIC_NMF_MIN_TARGETS = 2
        show_progs = [p for p in all_progs
                      if len(prog_ota_with_gamma.get(p, [])) >= GENETIC_NMF_MIN_TARGETS]
    elif mode == "nmf":
        # No γ floor — show any program with enough OTA targets
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

        if mode == "genetic_nmf":
            gamma_str = f"γ={gamma_prog:+.4f}" if not math.isnan(gamma_prog) else ""
            ax.set_title(
                f"{prog}  ·  {gamma_str}  ·  {len(genes_in_prog)} OTA targets",
                fontsize=8, fontweight="bold", pad=3,
            )
        elif mode == "nmf":
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


def plot_gnmf_programs(
    disease_key: str,
    condition: str = "Stim48hr",
    out_path: str | None = None,
) -> None:
    """
    Per-program scatter plots (GeneticNMF for RA; cNMF MAST betas for CAD).

    x-axis: OTA γ (condition-specific for RA — Stim48hr or REST; always ota_gamma for CAD)
      Derived from Perturb-seq × GWAS heritability (S-LDSC).
      γ < 0: KO protects from disease.  γ > 0: KO amplifies disease.

    y-axis: fingerprint_r (disease transcriptional proximity)
      Pearson r of KO signature vs disease DEG.
      r < 0: KO reverses disease transcriptome.  r > 0: KO mimics disease transcriptome.

    Four quadrants (shaded):
      Q2 (γ<0, r<0) — ★ Long-island / therapeutic: genetic + transcriptional convergence
      Q1 (γ>0, r>0) — Convergent disease-promoting
      Q3 (γ<0, r>0) — Genetic protection, transcriptional paradox
      Q4 (γ>0, r<0) — Transcriptional reverser but genetic disease-amplifier

    Beta source (auto-detected):
      RA  — condition-specific GeneticNMF DE betas (Stim48hr or REST)
      CAD — Schnitzler MAST cNMF betas (no condition split; condition arg ignored)

    Program assignment: gene → program with highest |β(gene→P)| × |γ(P)| in that condition.
    Programs titled with top expression genes from program annotations.
    Known clinical drug targets highlighted with diamond markers.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    import math as _math

    cfg = _DISEASE_CONFIG.get(disease_key)
    if cfg is None:
        print(f"plot_gnmf_programs: unknown disease key {disease_key!r}")
        return

    ck = _load_ck(cfg["slug"])
    targets = ck["prioritization_result"]["targets"]

    # ── Fingerprint r (y-axis) ────────────────────────────────────────────────
    fp_r_all = _load_fp_r(cfg["dataset_id"])

    # ── Beta source and program gammas — auto-detect by disease ──────────────
    dataset_dir = pathlib.Path(f"data/perturbseq/{cfg['dataset_id']}")
    mast_path   = dataset_dir / "cnmf_mast_betas.npz"

    if mast_path.exists():
        # CAD: Schnitzler MAST cNMF betas; condition arg irrelevant
        beta_path  = mast_path
        prog_prefix = f"{disease_key.upper()}_cNMF_"
        gamma_field = "ota_gamma"
        try:
            from pipelines.ldsc.gamma_loader import get_cnmf_program_gammas
            _raw = get_cnmf_program_gammas(disease_key.upper())
            prog_gammas: dict[str, float] = {p: v["gamma"] for p, v in _raw.items()
                                             if v.get("gamma") is not None}
        except Exception:
            prog_gammas = {}
        cond_label = "Cardiac endothelial cells"

        if not beta_path.exists():
            print(f"plot_gnmf_programs: {beta_path} not found — skipping")
            return
        _bd    = np.load(str(beta_path), allow_pickle=True)
        beta_m = _bd["beta"]
        ko_genes_list = [str(g) for g in _bd["ko_genes"]]
        prog_ids_list = [str(p) for p in _bd["program_ids"]]
        ko_to_row: dict[str, int] = {g: i for i, g in enumerate(ko_genes_list)}
    else:
        # RA: condition-specific GeneticNMF loadings (Vt.T = pert × program beta matrix)
        _cond_norm    = condition.strip() if condition.strip() else "Stim48hr"
        _is_rest      = _cond_norm.upper() == "REST"
        _is_combined  = "_" in _cond_norm and _cond_norm.upper() not in ("REST", "STIM48HR")
        _cond_lower   = _cond_norm.lower()
        loadings_path = dataset_dir / f"genetic_nmf_loadings_{_cond_lower}.npz"
        prog_prefix   = f"{disease_key.upper()}_GeneticNMF_{_cond_norm}_"
        gamma_field   = "ota_gamma_rest" if _is_rest else "ota_gamma"
        try:
            from pipelines.ldsc.gamma_loader import get_genetic_nmf_program_gammas
            _raw = get_genetic_nmf_program_gammas(disease_key.upper(), condition=_cond_norm)
            prog_gammas = {p: v["gamma"] for p, v in _raw.items()
                           if v.get("gamma") is not None}
        except Exception:
            prog_gammas = {}
        if _is_rest:
            cond_label = "Resting T cells"
        elif _is_combined:
            cond_label = "Resting + Activated T cells"
        else:
            cond_label = "Activated T cells (48 hr)"

        if not loadings_path.exists():
            print(f"plot_gnmf_programs: {loadings_path} not found — skipping")
            return
        _ld = np.load(str(loadings_path), allow_pickle=True)
        _Vt = _ld["Vt"]           # (k × n_perts)
        beta_m = _Vt.T.copy()     # (n_perts × k)
        k = beta_m.shape[1]
        prog_ids_list = [f"C{c+1:02d}" for c in range(k)]
        _pert_raw = [str(g) for g in _ld["pert_names"]]

        # Build ko_to_row — supports plain gene name AND gene_COND suffixed entries.
        # For combined (REST_Stim48hr): Stim48hr row wins for base-name lookup.
        ko_to_row = {}
        for i, g in enumerate(_pert_raw):
            ko_to_row[g] = i
            _base = g[:-len("_REST")] if g.endswith("_REST") else (
                    g[:-len("_Stim48hr")] if g.endswith("_Stim48hr") else g)
            if _base != g:
                if _is_combined:
                    if _base not in ko_to_row or g.endswith("_Stim48hr"):
                        ko_to_row[_base] = i
                elif _base not in ko_to_row:
                    ko_to_row[_base] = i

    # Full program IDs matching gamma keys
    full_progs = [f"{prog_prefix}{pid}" for pid in prog_ids_list]

    # ── Program annotations (biological gene names) ───────────────────────────
    ann_path = pathlib.Path(f"data/perturbseq/{cfg['dataset_id']}/program_annotations.json")
    prog_ann: dict[str, dict] = {}
    if ann_path.exists():
        import json as _json
        prog_ann = _json.load(open(ann_path))

    def _prog_label(prog: str) -> str:
        ann = prog_ann.get(prog, {})
        # Cap at 3 genes; strip long enhancer tokens for CAD
        raw_top = ann.get("top_genes", [])
        top = [g for g in raw_top if not g.startswith("Enhancer")][:3]
        # Extract short ID: 'RA_GeneticNMF_C01' → 'C01'; 'CAD_cNMF_P03' → 'P03'
        short_id = prog.rsplit("_", 1)[-1]
        genes_str = (" · ".join(top) + "  ") if top else ""
        g = prog_gammas.get(prog, float("nan"))
        g_str = f"h={g:+.2f}" if not _math.isnan(g) else ""
        # R² = mean squared fingerprint correlation across panel members
        # High R² → KO transcriptional response converges with disease DEG signature
        _r_vals = [r for _, _, r in prog_members.get(prog, []) if not _math.isnan(r)]
        r2_str = f"  R²={sum(v**2 for v in _r_vals)/len(_r_vals):.2f}" if _r_vals else ""
        return f"{short_id}  {genes_str}{g_str}{r2_str}"

    # ── Validated markers: clinical drug targets vs. research benchmark genes ──
    # status=="Research" → Schnitzler benchmark genes (star, gold); labeled but not drug diamonds.
    # All other statuses → clinical drug targets (diamond, green=inhibitor / red=agonist).
    try:
        from config.drug_target_registry import get_validated_targets
        _val = get_validated_targets(disease_key.upper())
        drug_target_set: set[str] = {v["gene"] for v in _val if v.get("status") != "Research"}
        drug_target_sign: dict[str, int] = {v["gene"]: v["expected_sign"]
                                             for v in _val if v.get("status") != "Research"}
        benchmark_set: set[str] = {v["gene"] for v in _val if v.get("status") == "Research"}
    except Exception:
        drug_target_set = set()
        drug_target_sign = {}
        benchmark_set = set()

    # ── Assign each OTA target to its most-contributing GeneticNMF program ───
    # Contribution = |β(gene→P)| × |γ(P)| — highest wins; gene appears in ONE panel.
    CONTRIB_FLOOR = 0.001   # ignore near-zero contributions
    prog_members: dict[str, list[tuple[str, float, float]]] = {}

    for t in targets:
        gene = t["target_gene"]
        gm_raw = t.get(gamma_field)
        if gm_raw is None or _math.isnan(float(gm_raw)):
            continue
        gm  = float(gm_raw)
        r   = fp_r_all.get(gene)
        if r is None:
            continue
        row = ko_to_row.get(gene)
        if row is None:
            continue

        # Compute β×γ contribution per program for this gene
        best_prog: str | None = None
        best_contrib = CONTRIB_FLOOR
        for c, fp in enumerate(full_progs):
            pg = prog_gammas.get(fp, 0.0)
            if pg == 0.0:
                continue
            contrib = abs(float(beta_m[row, c])) * abs(pg)
            if contrib > best_contrib:
                best_contrib = contrib
                best_prog = fp

        if best_prog is not None:
            prog_members.setdefault(best_prog, []).append((gene, gm, r))

    # ── Select programs to render ─────────────────────────────────────────────
    MIN_MEMBERS = 20

    def _prog_sort_key(p: str) -> tuple[float, float]:
        # Primary: h² × (Q2 drug targets + 0.5) — heritability-weighted therapeutic convergence
        # Squaring h penalises low-heritability programs even when they have many Q2 hits
        h = abs(prog_gammas.get(p, 0.0))
        members = prog_members.get(p, [])
        n_q2_drugs = sum(
            1 for g, gm, r in members
            if g in drug_target_set and gm < 0 and r < 0
        )
        # Secondary tiebreaker: h × program size (total genetic liability mass)
        return (h * h * (n_q2_drugs + 0.5), h * len(members))

    # Programs that contain at least one validated benchmark gene are always shown,
    # even if they have fewer than MIN_MEMBERS OTA targets.
    _benchmark_progs = {
        p for p in full_progs
        if any(g in (drug_target_set | benchmark_set) for g, _, _ in prog_members.get(p, []))
        and abs(prog_gammas.get(p, 0.0)) > 0.0
    }
    show_progs = sorted(
        [p for p in full_progs
         if (len(prog_members.get(p, [])) >= MIN_MEMBERS
             or p in _benchmark_progs)
         and abs(prog_gammas.get(p, 0.0)) > 0.0],
        key=_prog_sort_key,
        reverse=True,
    )

    if not show_progs:
        print(f"plot_gnmf_programs: no programs with ≥{MIN_MEMBERS} OTA targets for {condition}")
        return

    # ── Layout — first panel is the quadrant key; program panels follow ───────
    n_panels = len(show_progs) + 1   # +1 for key panel
    n_cols   = min(4, n_panels)
    n_rows   = _math.ceil(n_panels / n_cols)
    panel_sz = 5.2
    fig = plt.figure(figsize=(n_cols * panel_sz, n_rows * panel_sz))
    gs  = GridSpec(n_rows, n_cols, figure=fig, hspace=0.50, wspace=0.35)

    all_gm_vals = [abs(gm) for members in prog_members.values() for _, gm, _ in members]
    if all_gm_vals:
        # Use 97th percentile for regular genes; always include special gene positions
        all_gm_vals.sort()
        p97 = all_gm_vals[int(len(all_gm_vals) * 0.97)]
        special_max = max(
            (abs(gm) for members in prog_members.values()
             for g, gm, _ in members if g in (drug_target_set | benchmark_set)),
            default=0.0,
        )
        x_abs_max = max(p97, special_max, 1.0) * 1.15
        x_abs_max = min(x_abs_max, 14.0)   # hard cap; wider to accommodate validated anchors
    else:
        x_abs_max = 1.0
    shared_xlim = (-x_abs_max, x_abs_max)

    # Quadrant shading colors
    Q_THERAPEUTIC = "#D5F5E3"   # γ<0, r<0 — therapeutic
    Q_DISEASE_AMP = "#FDEDEC"   # γ>0, r>0 — convergent disease
    Q_GENETIC_ODD = "#EBF5FB"   # γ<0, r>0 — paradox
    Q_TRANSCRIP   = "#FDFEFE"   # γ>0, r<0 — transcriptional reverser

    def _shade_quadrants(ax_: "plt.Axes", xlim: tuple[float, float]) -> None:
        ax_.fill_between([xlim[0], 0], [0, 0], [1.1, 1.1],
                         color=Q_GENETIC_ODD, alpha=0.55, zorder=0)
        ax_.fill_between([0, xlim[1]], [0, 0], [1.1, 1.1],
                         color=Q_DISEASE_AMP, alpha=0.55, zorder=0)
        ax_.fill_between([xlim[0], 0], [-1.1, -1.1], [0, 0],
                         color=Q_THERAPEUTIC, alpha=0.60, zorder=0)
        ax_.fill_between([0, xlim[1]], [-1.1, -1.1], [0, 0],
                         color=Q_TRANSCRIP, alpha=0.40, zorder=0)

    # ── Panel 0 — Quadrant key ────────────────────────────────────────────────
    ax_key = fig.add_subplot(gs[0, 0])
    _shade_quadrants(ax_key, shared_xlim)
    ax_key.axvline(0, color="#555", lw=1.2, ls="--")
    ax_key.axhline(0, color="#555", lw=1.2, ls="--")

    # Axis spine annotations
    _arrow_kw = dict(fontsize=7.5, color="#444", ha="center")
    ax_key.text(0, -1.13, "← KO is protective (GWAS)       KO is harmful →",
                transform=ax_key.get_xaxis_transform(), **_arrow_kw)
    ax_key.text(-x_abs_max * 1.02, 0, "KO mimics disease ↑\n\nKO reverses disease ↓",
                fontsize=6.5, color="#444", ha="right", va="center", rotation=90)

    _kq = dict(ha="center", va="center", style="italic")
    # Top-left: Q3
    ax_key.text(-x_abs_max * 0.52, 0.62,
                "KO is protective (genetics)\nbut mimics disease RNA",
                color="#1A5276", fontsize=7.5, **_kq)
    # Top-right: Q1
    ax_key.text(+x_abs_max * 0.52, 0.62,
                "KO amplifies disease:\nboth genetics and RNA converge",
                color="#922B21", fontsize=7.5, **_kq)
    # Bottom-left: Q2 — therapeutic
    ax_key.text(-x_abs_max * 0.52, -0.62,
                "★  Therapeutic target\nKO is protective (genetics)\nand reverses disease RNA",
                color="#1E8449", fontsize=8, fontweight="bold", **_kq)
    # Bottom-right: Q4
    ax_key.text(+x_abs_max * 0.52, -0.62,
                "KO reverses disease RNA\nbut harmful by genetics",
                color="#7D6608", fontsize=7.5, **_kq)

    ax_key.set_xlim(*shared_xlim)
    ax_key.set_ylim(-1.05, 1.05)
    ax_key.set_xlabel("OTA causal score  (GWAS × Perturb-seq)", fontsize=8)
    ax_key.set_ylabel("Fingerprint r  (KO vs disease DEG)", fontsize=8)
    ax_key.set_facecolor("white")
    ax_key.spines["top"].set_visible(False)
    ax_key.spines["right"].set_visible(False)
    ax_key.set_title("How to read these plots", fontsize=8, fontweight="bold")

    for pi, prog in enumerate(show_progs):
        row_idx, col_idx = divmod(pi + 1, n_cols)   # +1 offset for key panel
        ax = fig.add_subplot(gs[row_idx, col_idx])

        members = prog_members[prog]
        n_dots  = len(members)
        scale   = max(0.3, 1.0 - 0.005 * n_dots)
        sz_base = max(14, int(50 * scale))
        sz_drug = max(35, int(110 * scale))   # drug/benchmark markers; smaller cap for dense RA panels

        # Quadrant background shading (no text labels — key panel handles them)
        _shade_quadrants(ax, shared_xlim)

        # Reference lines
        ax.axvline(0, color="#777777", lw=0.9, ls="--", zorder=3)
        ax.axhline(0, color="#777777", lw=0.9, ls="--", zorder=3)

        # CAD: benchmark stars (Schnitzler) only; no drug diamonds.
        # RA:  clinical drug targets → diamond; benchmark_set → star.
        is_mast = mast_path.exists()   # True for CAD
        special_genes = drug_target_set | benchmark_set
        bench_members = [(g, gm, r) for g, gm, r in members if g in benchmark_set]
        if is_mast:
            # CAD: no drug diamonds; benchmarks shown as stars
            drug_members  = []
            reg_members   = [(g, gm, r) for g, gm, r in members if g not in benchmark_set]
        else:
            drug_members  = [(g, gm, r) for g, gm, r in members if g in drug_target_set]
            reg_members   = [(g, gm, r) for g, gm, r in members if g not in special_genes]
        always_labeled = benchmark_set | (drug_target_set if not is_mast else set())

        # Top genes by |βγ| contribution to this program — always labeled
        _prog_col = full_progs.index(prog) if prog in full_progs else -1
        _pg = abs(prog_gammas.get(prog, 0.0))
        def _contrib(gene: str) -> float:
            if _prog_col < 0 or _pg == 0.0:
                return 0.0
            row_ = ko_to_row.get(gene)
            return abs(float(beta_m[row_, _prog_col])) * _pg if row_ is not None else 0.0

        sorted_by_contrib = sorted(reg_members, key=lambda x: _contrib(x[0]), reverse=True)
        N_LABEL = min(10, max(4, len(reg_members) // 12))
        top_contrib_genes = {g for g, _, _ in sorted_by_contrib[:N_LABEL]}
        # Always label genes in the protective+disease-fingerprint quadrant (gamma<0, r<0)
        # — these are the most actionable targets: KO amplifies disease AND mimics disease state.
        bottom_left_genes = {g for g, gm, r in members if gm < 0 and r < 0}
        label_genes = top_contrib_genes | bottom_left_genes

        # ── Scatter with non-linear radial alpha ───────────────────────────────
        # Alpha scales as dist^2.0 from origin — dots near (0,0) are dim,
        # periphery genes (high |OTA score| or |r|) remain saturated.
        # Floor raised so no dot is fully invisible.
        for gene, gm, r in reg_members:
            dist  = _math.sqrt((gm / x_abs_max) ** 2 + r ** 2) / _math.sqrt(2)
            alpha = max(0.65, min(0.92, dist ** 1.1))
            ax.scatter(gm, r, c="#2C7BB6", s=sz_base, marker="o",
                       edgecolors="none", linewidths=0, alpha=alpha, zorder=4)

        # Benchmark stars (Schnitzler research anchors) — RA only
        for gene, gm, r in bench_members:
            ax.scatter(gm, r, c="#E67E22", s=sz_drug * 0.9, marker="*",
                       edgecolors="#7D3C00", linewidths=0.8, alpha=0.95, zorder=7)

        # Clinical drug target diamonds — RA only
        for gene, gm, r in drug_members:
            expected  = drug_target_sign.get(gene, 0)
            dot_color = "#27AE60" if expected == -1 else "#E74C3C"
            ax.scatter(gm, r, c=dot_color, s=sz_drug, marker="D",
                       edgecolors="#1A1A1A", linewidths=1.5, alpha=0.95, zorder=7)

        # ── Labels ────────────────────────────────────────────────────────────
        all_label_genes = label_genes | always_labeled
        for gene, gm, r in members:
            if gene not in all_label_genes:
                continue
            is_drug  = gene in drug_target_set and not is_mast
            is_bench = gene in benchmark_set and not is_mast
            va = "bottom" if r >= 0 else "top"
            dy = 0.028 if r >= 0 else -0.028
            ax.annotate(
                gene, (gm, r + dy), ha="center", va=va,
                fontsize=6.5 if (is_drug or is_bench or gene in always_labeled) else 5.0,
                color=(
                    "#7D3C00" if is_bench else
                    ("#1E8449" if drug_target_sign.get(gene, 0) == -1 else "#922B21") if is_drug else
                    "#2C3E50" if gene in always_labeled else
                    "#1A1A1A"
                ),
                fontweight="bold" if (is_drug or is_bench or gene in always_labeled) else "normal",
                zorder=8,
            )

        ax.set_xlim(*shared_xlim)
        ax.set_ylim(-1.05, 1.05)
        ax.set_ylabel("Fingerprint r  (KO vs disease DEG)", fontsize=6.5)
        ax.set_xlabel(f"OTA causal score ({cond_label})  ← protective | disease-amplifying →", fontsize=6.5)
        ax.set_facecolor("white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=6)

        ax.set_title(_prog_label(prog) + f"  [{n_dots}]",
                     fontsize=7, fontweight="bold", pad=4, wrap=True)

    # Legend
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2C7BB6",
               markersize=6, label="OTA target"),
    ]
    if benchmark_set:
        legend_elems.append(
            Line2D([0], [0], marker="*", color="w", markerfacecolor="#E67E22",
                   markeredgecolor="#7D3C00", markersize=10, label="Schnitzler et al. benchmark gene")
        )
    if not mast_path.exists():
        legend_elems += [
            Line2D([0], [0], marker="D", color="w", markerfacecolor="#27AE60",
                   markeredgecolor="#1A1A1A", markersize=7, label="Drug target (inhibitor, γ<0)"),
            Line2D([0], [0], marker="D", color="w", markerfacecolor="#E74C3C",
                   markeredgecolor="#1A1A1A", markersize=7, label="Drug target (agonist, γ>0)"),
        ]
    fig.legend(handles=legend_elems, loc="lower center", ncol=len(legend_elems),
               fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.01))

    prog_type = "cNMF" if mast_path.exists() else "GeneticNMF"
    fig.suptitle(
        f"{cfg['label']} — {prog_type} programs  ·  {cond_label}\n"
        f"x = OTA causal score (GWAS × Perturb-seq)   y = fingerprint r (KO vs disease DEG)"
        f"   |   Panel title: h = program heritability enrichment weight",
        fontsize=8.5, fontweight="bold", y=1.01,
    )

    if mast_path.exists():
        default = f"outputs/program_landscape_{disease_key}.png"
    else:
        suffix = "stim48hr" if condition == "Stim48hr" else condition.lower()
        default = f"outputs/program_landscape_{disease_key}_{suffix}.png"
    plt.savefig(out_path or default, dpi=160, bbox_inches="tight")
    print(f"Saved → {out_path or default}")

    # ── Benchmark check ────────────────────────────────────────────────────────
    _all_val = {v["gene"]: v for v in (list(drug_target_sign.items()) and [])}  # reset
    try:
        from config.drug_target_registry import get_validated_targets
        _all_val = {v["gene"]: v for v in get_validated_targets(disease_key.upper())}
    except Exception:
        pass
    if _all_val and targets:
        _t_map = {t["target_gene"]: t for t in targets}
        _pass = _fail = _missing = 0
        rows = []
        for gene, meta in sorted(_all_val.items()):
            t = _t_map.get(gene)
            exp = meta.get("expected_sign", 0)
            if t is None:
                _missing += 1
                rows.append(f"  {'MISS':5s}  {gene:<12s}  exp={'±' if exp==0 else ('+' if exp>0 else '-')}  not in targets")
            else:
                gm = t.get(gamma_field, float("nan"))
                if _math.isnan(float(gm)):
                    _missing += 1
                    rows.append(f"  {'NaN':5s}  {gene:<12s}  exp={'±' if exp==0 else ('+' if exp>0 else '-')}  ota_gamma=NaN  rank={t.get('rank','?')}")
                else:
                    actual_sign = 1 if float(gm) > 0 else -1
                    ok = (exp == 0) or (actual_sign == exp)
                    if ok:
                        _pass += 1
                    else:
                        _fail += 1
                    flag = "✓" if ok else "✗"
                    rows.append(
                        f"  {flag:5s}  {gene:<12s}  exp={'±' if exp==0 else ('+' if exp>0 else '-')}  "
                        f"γ={float(gm):+.3f}  rank={t.get('rank','?')}"
                    )
        total = _pass + _fail + _missing
        print(f"\nBenchmark ({disease_key.upper()}, {condition or 'default'})  "
              f"{_pass}/{total} pass  {_fail} wrong-sign  {_missing} missing")
        for row in rows:
            print(row)


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
    ap.add_argument("--explainer", action="store_true",
                    help="Generate the axis explainer mock figure instead")
    args = ap.parse_args()
    if args.explainer:
        plot_explainer(out_path=args.out or "outputs/long_island_explainer.png")
    elif args.disease == "cad":
        plot_gnmf_programs("cad", out_path=args.out)
    elif args.disease == "ra":
        plot_gnmf_programs("ra", condition="Stim48hr",
                           out_path=args.out or "outputs/program_landscape_ra_stim48hr.png")
        plot_gnmf_programs("ra", condition="REST",
                           out_path="outputs/program_landscape_ra_rest.png")
        plot_gnmf_programs("ra", condition="REST_Stim48hr",
                           out_path="outputs/program_landscape_ra_rest_stim48hr.png")
    else:
        for _d in ["cad", "ra"]:
            if _d == "ra":
                plot_gnmf_programs(_d, condition="Stim48hr",
                                   out_path="outputs/program_landscape_ra_stim48hr.png")
                plot_gnmf_programs(_d, condition="REST",
                                   out_path="outputs/program_landscape_ra_rest.png")
                plot_gnmf_programs(_d, condition="REST_Stim48hr",
                                   out_path="outputs/program_landscape_ra_rest_stim48hr.png")
            else:
                plot_gnmf_programs(_d)
