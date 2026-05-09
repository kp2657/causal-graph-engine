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
    },
    "cad": {
        "label":      "Coronary artery disease",
        "slug":       "coronary_artery_disease",
        "dataset_id": "schnitzler_cad_vascular",
    },
}

# Benchmark gene colors
_BENCH_RISK_UP_COLOR   = "#D32F2F"   # KD increases risk (atheroprotective gene) — red
_BENCH_RISK_DOWN_COLOR = "#1565C0"   # KD reduces risk (atherogenic driver) — blue


def _load_benchmark_sets(disease_key: str) -> tuple[set[str], set[str]]:
    """Return (risk_up, risk_down) benchmark gene sets from drug_target_registry.
    risk_up   = KD increases disease risk (expected_sign=-1, atheroprotective)
    risk_down = KD reduces disease risk  (expected_sign=+1, atherogenic driver)
    """
    try:
        from config.drug_target_registry import get_validated_targets
        targets = get_validated_targets(disease_key.upper())
        risk_up   = {t["gene"] for t in targets if t.get("expected_sign") == -1}
        risk_down = {t["gene"] for t in targets if t.get("expected_sign") == +1}
        return risk_up, risk_down
    except Exception:
        return set(), set()

_PROG_PALETTE = [
    "#1565C0","#2E7D32","#6A1B9A","#E65100","#B71C1C",
    "#00695C","#4527A0","#F57F17","#37474F","#AD1457",
]

# Biological identity markers — used to label GeneticNMF programs by cell-state
_BIO_MARKERS: list[tuple[set, str]] = [
    ({"CCR6", "RORC", "IL17F", "IL17A", "IL23R", "IL6", "RORA"},  "Th17"),
    ({"IL4", "IL5", "IL10", "GATA3", "IL13", "IL25"},              "Th2/reg"),
    ({"FOXP3", "IKZF2", "TNFRSF18", "CTLA4"},                      "Treg"),
    ({"IFNG", "TBX21", "CXCR3", "TNF"},                            "Th1"),
    ({"GZMA", "GZMB", "PRF1", "FASL", "GNLY"},                     "Cytotoxic"),
    ({"NR4A3", "NR4A1", "FOS", "JUN", "CCL4", "CCL3"},             "Early activation"),
    ({"CCR7", "LEF1", "TCF7", "SELL"},                              "Naive/memory"),
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

    # Override with S-LDSC |τ*| gammas (GeneticNMF + LOCUS) when available.
    try:
        import sys as _sys
        import pathlib as _pl
        _root = _pl.Path(__file__).parent.parent
        if str(_root) not in _sys.path:
            _sys.path.insert(0, str(_root))
        from pipelines.ldsc.gamma_loader import get_genetic_nmf_program_gammas, get_locus_program_gammas
        nmf_gammas = get_genetic_nmf_program_gammas(disease_key.upper())
        for prog, est in nmf_gammas.items():
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
    prog_gammas = _prog_gamma(t3, disease_key)

    bench_risk_up, bench_risk_down = _load_benchmark_sets(disease_key)
    known = bench_risk_up | bench_risk_down   # all benchmark genes (for label inclusion)

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

    elif mode == "cnmf":
        # cNMF mode: panels are the actual Schnitzler cNMF programs (P01–P60).
        # Winner-takes-all uses |β_{gene→P} × τ*_{P→trait}| — exactly the OTA product.
        # top_programs keys in the checkpoint are bare "P01".."P60" matching tau keys.
        import json as _json
        _tau_path = pathlib.Path("data/ldsc/results/CAD_cNMF_program_taus.json")
        _tau_data = _json.loads(_tau_path.read_text())
        _tau_map: dict[str, float] = _tau_data.get("program_taus", {})
        # Update prog_gammas with cNMF taus (bare "P14" keys)
        prog_gammas.update(_tau_map)

        # All 60 cNMF programs; sort by |τ| descending; take top K with τ > 0
        prog_names = sorted(_tau_map.keys(), key=lambda p: int(p[1:]))  # P01..P60 order
        gwas_progs_sorted = sorted(
            [p for p in prog_names if _tau_map.get(p, 0.0) > 0],
            key=lambda p: -abs(_tau_map.get(p, 0.0)),
        )[:TOP_GWAS_COMPONENTS]

        # Build gene_gwas_prog (dominant program by |β×τ|) from top_programs in checkpoint
        gene_gwas_prog = {}
        gene_z: dict[str, float] = {}
        for t in targets:
            gene = t["target_gene"]
            tp = t.get("top_programs") or {}
            best_p, best_c = None, 0.0
            for p, b in tp.items():
                c = abs(b * _tau_map.get(p, 0.0))
                if c > best_c:
                    best_c, best_p = c, p
            if best_p and best_p in gwas_progs_sorted:
                gene_gwas_prog[gene] = best_p
                gene_z[gene] = best_c

        prog_gammas_display = _tau_map
        print(f"  Top cNMF programs (by |τ*|): "
              f"{', '.join(gwas_progs_sorted[:10])}")

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
    _gene_z = gene_z if mode in ("svd", "genetic_nmf", "cnmf") else {g: 0.0 for g in fp_names}
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

    # ── Assign each gene to exactly one program (winner-takes-all by |β×γ|) ───
    # Each gene appears in the panel of the program it drives most strongly.
    # This makes panels non-overlapping: every OTA gene appears exactly once.
    prog_ota_with_gamma: dict[str, list[tuple[str, float, float]]] = {}
    gwas_prog_set = set(all_progs)

    if mode == "genetic_nmf":
        npz_path = pathlib.Path(f"data/perturbseq/{cfg['dataset_id']}/genetic_nmf_loadings.npz")
        _nmf_d   = np.load(npz_path)
        U_scaled_full = _nmf_d["U_scaled"]
        response_names = list(_nmf_d["gene_names"])
        resp_to_idx = {g: i for i, g in enumerate(response_names)}
        pert_to_idx = {name: i for i, name in enumerate(fp_names)}

        for t in targets:
            gene  = t["target_gene"]
            gamma = gene_gamma.get(gene)
            r     = fp_r_all.get(gene)
            if gamma is None or r is None:
                continue
            if gene in resp_to_idx:
                u_vec  = U_scaled_full[resp_to_idx[gene], :]
                best_k = int(u_vec.argmax())
            else:
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
            # Find the program in gwas_prog_set with the highest |β×γ| contribution
            best_prog, best_contrib = None, 0.0
            for prog, b in tp.items():
                if prog not in gwas_prog_set:
                    continue
                contrib = abs(b * prog_gammas.get(prog, 0.0))
                if contrib > best_contrib:
                    best_contrib, best_prog = contrib, prog
            if best_prog is None:
                # Fallback: dominant z-score assignment
                best_prog = gene_gwas_prog.get(gene)
            if best_prog:
                prog_ota_with_gamma.setdefault(best_prog, []).append((gene, gamma, r))

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
            highlight = gene in label_set
            is_bench  = gene in bench_risk_up or gene in bench_risk_down
            size  = sz_t1 if (ota_genes.get(gene, {}).get("dominant_tier","") == "Tier1_Interventional" or is_bench) else sz_other
            if highlight:
                alpha = 0.95
            else:
                dist  = math.sqrt((gm / gamma_norm) ** 2 + r ** 2) / math.sqrt(2)
                alpha = max(0.06, min(0.90, 0.10 + 0.80 * dist))
            ax.scatter(gm, r, c=pcol, s=size,
                       edgecolors="none", linewidths=0, alpha=alpha, zorder=4)

        for gene, gm, r in genes_in_prog:
            if gene in label_set:
                va = "bottom" if r >= 0 else "top"
                dy = 0.022 if r >= 0 else -0.022
                if gene in bench_risk_up:
                    # Red up-arrow then gene name
                    ax.annotate("▲", (gm, r + dy), ha="center", va=va,
                                fontsize=6, color=_BENCH_RISK_UP_COLOR,
                                fontweight="bold", zorder=7)
                    dy2 = dy + (0.06 if r >= 0 else -0.06)
                    ax.annotate(gene, (gm, r + dy2), ha="center", va=va,
                                fontsize=5.5, color=_BENCH_RISK_UP_COLOR,
                                fontweight="bold", zorder=7)
                elif gene in bench_risk_down:
                    # Black down-arrow then gene name
                    ax.annotate("▼", (gm, r + dy), ha="center", va=va,
                                fontsize=6, color="#222222",
                                fontweight="bold", zorder=7)
                    dy2 = dy + (0.06 if r >= 0 else -0.06)
                    ax.annotate(gene, (gm, r + dy2), ha="center", va=va,
                                fontsize=5.5, color="#222222",
                                fontweight="bold", zorder=7)
                else:
                    ax.annotate(gene, (gm, r + dy), ha="center", va=va,
                                fontsize=5, color="#222222",
                                fontweight="normal", zorder=6)

        ax.axvline(0, color="#888888", lw=0.8, ls="--", zorder=3)
        ax.axhline(0, color="#888888", lw=0.8, ls="--", zorder=3)
        ax.set_xlim(*shared_xlim)
        ax.set_ylim(-1.05, 1.05)
        ax.set_ylabel("Fingerprint r\n(KO vs disease DEG)", fontsize=7)
        ax.set_xlabel("OTA γ  (β direction: ← suppresses programs  |  activates programs →)", fontsize=7)
        ax.set_facecolor("white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Benchmark legend (only on first panel)
        if pi == 0 and (bench_risk_up or bench_risk_down):
            import matplotlib.lines as mlines
            _leg_handles = []
            if bench_risk_up:
                _leg_handles.append(mlines.Line2D([], [], marker="^", color="w",
                    markerfacecolor=_BENCH_RISK_UP_COLOR, markersize=7,
                    label="▲ KD increases risk (atheroprotective)"))
            if bench_risk_down:
                _leg_handles.append(mlines.Line2D([], [], marker="v", color="w",
                    markerfacecolor="#222222", markersize=7,
                    label="▼ KD reduces risk (atherogenic driver)"))
            ax.legend(handles=_leg_handles, fontsize=5.5, loc="upper left",
                      framealpha=0.8, borderpad=0.5)

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
        elif mode == "cnmf":
            tau_val = prog_gammas.get(prog, float("nan"))
            tau_str = f"τ*={tau_val:+.3f}" if not math.isnan(tau_val) else ""
            ax.set_title(
                f"{prog}  ·  {tau_str}  ·  {len(genes_in_prog)} OTA targets",
                fontsize=8, fontweight="bold", pad=3,
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
    _U_scaled   = None   # gene-space NMF loadings — set in RA branch only
    _gene_names: list[str] = []

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
            prog_gamma_annots: dict[str, dict] = {p: v for p, v in _raw.items()}
        except Exception:
            prog_gammas = {}
            prog_gamma_annots = {}
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
            prog_gamma_annots = {p: v for p, v in _raw.items()}
        except Exception:
            prog_gammas = {}
            prog_gamma_annots = {}
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
        # Gene-space loadings for biological labelling (genes × k)
        _U_scaled   = _ld["U_scaled"] if "U_scaled" in _ld else None
        _gene_names = [str(g) for g in _ld["gene_names"]] if "gene_names" in _ld else []

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

    # ── Program annotations: top genes from U_scaled gene-space loadings ────────
    # U_scaled[:,c] = gene weights for program c. Top genes identify cell-state biology.
    _prog_top_genes: dict[str, list[str]] = {}
    if _U_scaled is not None and _gene_names:
        for _ci, _fp in enumerate(full_progs):
            _idx = np.argsort(_U_scaled[:, _ci])[::-1][:5]
            _prog_top_genes[_fp] = [_gene_names[i] for i in _idx
                                    if not _gene_names[i].startswith("Enhancer")][:4]

    def _bio_label(top_genes: list[str]) -> str:
        top_set = set(top_genes)
        for markers, label in _BIO_MARKERS:
            if top_set & markers:
                return label
        return ""

    def _prog_label(prog: str) -> str:
        top = _prog_top_genes.get(prog, [])[:3]
        bio = _bio_label(_prog_top_genes.get(prog, []))
        bio_str = f" ({bio})" if bio else ""
        # Extract short ID: 'RA_GeneticNMF_C01' → 'C01'; 'CAD_cNMF_P03' → 'P03'
        short_id = prog.rsplit("_", 1)[-1]
        genes_str = (" · ".join(top) + "  ") if top else ""
        # In β-only mode: show effective OTA γ (mean of member genes) since GeneticNMF
        # LDSC τ falls back to LOCUS; the causal signal still exists, just not per-program.
        g = _display_gammas.get(prog, float("nan"))
        _gann = prog_gamma_annots.get(prog, {})
        _ph2 = _gann.get("prop_h2")
        _enr = _gann.get("enrichment")
        if _beta_only_mode:
            g_str = f"γ̄={g:+.2f}" if not _math.isnan(g) else ""
        elif _ph2 is not None and _enr is not None:
            g_str = f"h²={_ph2*100:.1f}%  E={_enr:.1f}×"
        elif not _math.isnan(g):
            g_str = f"h={g:+.2f}"
        else:
            g_str = ""
        # R² = Pearson r² between OTA causal γ (x) and fingerprint_r (y) across panel genes
        # High R² → genes with higher genetic causal score also have stronger transcriptional
        # convergence with the disease DEG signature (genetic and txn evidence aligned)
        _pairs = [(gm, r) for _, gm, r in prog_members.get(prog, [])
                  if not _math.isnan(gm) and not _math.isnan(r)]
        return f"{short_id}{bio_str}  {genes_str}{g_str}"

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

    # ── Build program membership: inclusive β-threshold ───────────────────────
    # Each gene appears in every program where |β| ≥ BETA_THRESHOLD (not winner-takes-all).
    # y-value stored = program-specific contribution = β_prog × γ_prog (signed).
    # This reveals underweighting: genes with tiny β on high-γ programs show up near y=0.
    # β-only fallback (no S-LDSC γ available): use winner-takes-all by max |β|.
    _beta_only_mode = not prog_gammas
    BETA_THRESHOLD = 0.10   # minimum |β| to include gene in a program panel

    prog_members: dict[str, list[tuple[str, float, float]]] = {}

    for t in targets:
        gene  = t["target_gene"]
        r     = fp_r_all.get(gene)
        if r is None:
            continue
        row = ko_to_row.get(gene)
        if row is None:
            continue

        if _beta_only_mode:
            # β-only fallback: winner-takes-all by largest |β|
            best_prog: str | None = None
            best_b = BETA_THRESHOLD
            for c, fp in enumerate(full_progs):
                b = abs(float(beta_m[row, c]))
                if b > best_b:
                    best_b = b
                    best_prog = fp
            if best_prog is not None:
                prog_members.setdefault(best_prog, []).append((gene, best_b, r))
        else:
            # Inclusive: add gene to every program where |β| ≥ threshold AND γ > 0
            for c, fp in enumerate(full_progs):
                pg = prog_gammas.get(fp, 0.0)
                if pg == 0.0:
                    continue
                b = float(beta_m[row, c])
                if abs(b) < BETA_THRESHOLD:
                    continue
                contrib = b * pg   # signed program-specific OTA contribution
                prog_members.setdefault(fp, []).append((gene, contrib, r))

    if not _beta_only_mode:
        _display_gammas = prog_gammas
    else:
        _display_gammas = {}
        for p, _pm in prog_members.items():
            _vals = [gm for _, gm, _ in _pm if not _math.isnan(gm)]
            if _vals:
                _display_gammas[p] = sum(_vals) / len(_vals)

    # ── Select programs to render ─────────────────────────────────────────────
    MIN_MEMBERS = 10   # lower threshold: inclusive assignment fills programs with more genes

    def _prog_sort_key(p: str) -> float:
        return abs(_display_gammas.get(p, 0.0))

    # Programs containing any validated target are always shown (even if < MIN_MEMBERS).
    _validated_progs = {
        p for p in full_progs
        if any(g in (drug_target_set | benchmark_set) for g, _, _ in prog_members.get(p, []))
        and abs(_display_gammas.get(p, 0.0)) > 0.0
    }
    H_MIN = 0.20   # only render programs where |h| = |τ*| ≥ this threshold
    show_progs = sorted(
        [p for p in full_progs
         if (len(prog_members.get(p, [])) >= MIN_MEMBERS or p in _validated_progs)
         and abs(_display_gammas.get(p, 0.0)) >= H_MIN],
        key=_prog_sort_key,
        reverse=True,
    )

    if not show_progs:
        print(f"plot_gnmf_programs: no programs with ≥{MIN_MEMBERS} OTA targets for {condition}")
        return

    # ── Split targets by evidence class ──────────────────────────────────────
    # "grounded": has direct Perturb-seq β (any tier except provisional_virtual)
    # "provisional": GWAS-grounded only, no Perturb-seq KO β (provisional_virtual)
    _prov_genes: set[str] = {
        t["target_gene"] for t in targets
        if t.get("dominant_tier") == "provisional_virtual"
    }

    def _split_members(members: list) -> tuple[list, list]:
        """Return (grounded, provisional) sublists."""
        grounded = [(g, gm, r) for g, gm, r in members if g not in _prov_genes]
        prov     = [(g, gm, r) for g, gm, r in members if g in _prov_genes]
        return grounded, prov

    is_mast = mast_path.exists()   # True for CAD
    special_genes  = drug_target_set | benchmark_set
    always_labeled = benchmark_set | (drug_target_set if not is_mast else set())
    prog_type = "cNMF" if is_mast else "GeneticNMF"

    _layout_note = (
        "  [β-only mode; γ from LOCUS genetics]" if _beta_only_mode else
        "  |  Panel title: h = heritability enrichment weight"
    )
    if is_mast:
        _file_suffix = disease_key
    else:
        _cond_slug = "stim48hr" if condition == "Stim48hr" else condition.lower()
        _file_suffix = f"{disease_key}_{_cond_slug}"

    # ── Per-program scatter: x = fingerprint r, y = OTA γ (γ > 0 only) ───────
    def _draw_subpanel(ax, members, prog, dot_color,
                       title_extra="", _y_max_shared=None):
        """Scatter: x = KO reversal (fingerprint r), y = OTA γ. Positive γ only."""
        pos_m = [(g, gm, r) for g, gm, r in members if gm > 0]
        if not pos_m:
            ax.set_xlim(-1.05, 1.05)
            ax.set_ylim(bottom=0, top=_y_max_shared)
            ax.set_xlabel("KO phenotype reversal  (fingerprint r)", fontsize=6)
            ax.set_ylabel("β × γ_prog", fontsize=6)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(labelsize=5.5)
            ax.set_title(f"{_prog_label(prog)}{title_extra}\n[no contrib>0]",
                         fontsize=6.5, fontweight="bold", pad=3)
            return

        n = len(pos_m)
        scale   = max(0.3, 1.0 - 0.004 * n)
        sz_base = max(8, int(35 * scale))
        sz_hi   = max(28, int(80 * scale))

        bench_m = [(g, gm, r) for g, gm, r in pos_m if g in benchmark_set]
        drug_m  = [(g, gm, r) for g, gm, r in pos_m
                   if g in drug_target_set and not is_mast]
        reg_m   = [(g, gm, r) for g, gm, r in pos_m
                   if g not in benchmark_set and not (g in drug_target_set and not is_mast)]

        top_by_gm = sorted(pos_m, key=lambda t: t[1], reverse=True)
        N_LBL = min(10, max(3, n // 8))
        label_genes_ = {g for g, _, _ in top_by_gm[:N_LBL]} | always_labeled

        CONTRIB_DIM = 0.15   # β×γ below this → dim dot (weak program driver)
        for gene, gm, r in reg_m:
            a = 0.55 if gm >= CONTRIB_DIM else 0.12
            ax.scatter(r, gm, c=dot_color, s=sz_base, marker="o",
                       edgecolors="none", alpha=a, zorder=4)
        for gene, gm, r in bench_m:
            a = 0.95 if gm >= CONTRIB_DIM else 0.30
            ax.scatter(r, gm, c="#E67E22", s=sz_hi * 0.9, marker="*",
                       edgecolors="#7D3C00", linewidths=0.8, alpha=a, zorder=7)
        for gene, gm, r in drug_m:
            dc = "#27AE60" if drug_target_sign.get(gene, 0) == -1 else "#E74C3C"
            a = 0.95 if gm >= CONTRIB_DIM else 0.30
            ax.scatter(r, gm, c=dc, s=sz_hi, marker="D",
                       edgecolors="#1A1A1A", linewidths=1.0, alpha=a, zorder=7)

        for gene, gm, r in pos_m:
            if gene not in label_genes_:
                continue
            is_bench_ = gene in benchmark_set
            is_drug_  = gene in drug_target_set and not is_mast
            ax.annotate(
                gene, (r, gm), xytext=(3, 3), textcoords="offset points",
                ha="left", va="bottom",
                fontsize=5.5 if (is_drug_ or is_bench_ or gene in always_labeled) else 4.5,
                color=(
                    "#7D3C00" if is_bench_ else
                    ("#1E8449" if drug_target_sign.get(gene, 0) == -1 else "#922B21") if is_drug_ else
                    "#2C3E50"
                ),
                fontweight="bold" if (is_drug_ or is_bench_ or gene in always_labeled) else "normal",
                zorder=8,
            )

        ax.axvline(0, color="#999", lw=0.8, ls="--", zorder=2)
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(bottom=0, top=_y_max_shared)
        ax.set_xlabel("KO phenotype reversal  (fingerprint r)", fontsize=6)
        ax.set_ylabel("Program contribution  β × γ_prog", fontsize=6)
        ax.set_facecolor("white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=5.5)
        ax.set_title(
            f"{_prog_label(prog)}{title_extra}\n[n={n} with γ>0]",
            fontsize=6.5, fontweight="bold", pad=3,
        )

    # ── Helper: build + save one figure for one evidence class ────────────────
    from matplotlib.lines import Line2D as _L2D

    def _make_key_ax(ax_k, dot_color, evidence_label):
        ax_k.axis("off")
        ax_k.set_facecolor("white")
        ax_k.set_title("How to read", fontsize=9, fontweight="bold", pad=4)

        _legend_lines = [
            ("o", dot_color, "none",    8,  evidence_label),
            ("*", "#E67E22", "#7D3C00", 12, "Schnitzler benchmark gene"),
            ("D", "#27AE60", "#1A1A1A", 9,  "Approved drug (inhibitor)"),
            ("D", "#E74C3C", "#1A1A1A", 9,  "Approved drug (agonist)"),
        ]
        _handles = [
            _L2D([0], [0], marker=mk, color="w", markerfacecolor=fc,
                 markeredgecolor=ec, markersize=ms, label=lbl)
            for mk, fc, ec, ms, lbl in _legend_lines
        ]
        ax_k.legend(handles=_handles, loc="upper left", fontsize=7.5,
                    frameon=False, handletextpad=0.6, labelspacing=0.8)

        _desc = (
            "x  = fingerprint r (KO vs disease DEG)\n"
            "     negative → KO reverses disease RNA\n"
            "     positive → KO mimics disease RNA\n\n"
            "y  = β × γ_prog  (program-specific, >0 only)\n"
            "     β = Perturb-seq KO loading on program\n"
            "     γ = S-LDSC heritability enrichment\n"
            "     low y → gene barely drives this program\n\n"
            "Each program shown independently — a gene\n"
            "can appear in multiple programs."
        )
        ax_k.text(0.03, 0.38, _desc, transform=ax_k.transAxes,
                  fontsize=7, va="top", family="monospace",
                  color="#2C2C2C", linespacing=1.5)

    def _build_and_save(per_prog_m, dot_color, evidence_label,
                        title_tag, out_file, title_extra_fn=lambda p: ""):
        # Shared y-axis: max positive γ across all programs in this figure
        _y_max = max(
            (gm for prog in show_progs
             for _, gm, _ in per_prog_m.get(prog, []) if gm > 0),
            default=1.0,
        ) * 1.08

        n_panels = 1 + len(show_progs)
        n_cols_f = min(5, n_panels)
        n_rows_f = _math.ceil(n_panels / n_cols_f)
        panel_sz = 4.0
        fig_f = plt.figure(figsize=(n_cols_f * panel_sz, n_rows_f * panel_sz))
        gs_f  = GridSpec(n_rows_f, n_cols_f, figure=fig_f,
                         hspace=0.55, wspace=0.38)

        ax_k = fig_f.add_subplot(gs_f[0, 0])
        _make_key_ax(ax_k, dot_color, evidence_label)

        for pi, prog in enumerate(show_progs):
            slot = pi + 1
            row_f, col_f = divmod(slot, n_cols_f)
            ax_p = fig_f.add_subplot(gs_f[row_f, col_f])
            _draw_subpanel(ax_p, per_prog_m.get(prog, []), prog,
                           dot_color, title_extra=title_extra_fn(prog),
                           _y_max_shared=_y_max)

        fig_f.suptitle(
            f"{cfg['label']} — {prog_type} programs  ·  {cond_label}  ·  {title_tag}\n"
            f"OTA γ = Σ_P β(gene→program) × |τ*|(program→trait){_layout_note}",
            fontsize=8.5, fontweight="bold", y=1.01,
        )
        plt.savefig(out_file, dpi=160, bbox_inches="tight")
        plt.close(fig_f)
        print(f"Saved → {out_file}")

    # Build grounded members dict (per program, only non-provisional genes)
    grounded_pm = {prog: [t for t in prog_members[prog] if t[0] not in _prov_genes]
                   for prog in show_progs}
    # Build provisional members dict (per program, only provisional_virtual genes)
    prov_pm     = {prog: [t for t in prog_members[prog] if t[0] in _prov_genes]
                   for prog in show_progs}

    _build_and_save(
        grounded_pm, "#2C7BB6", "Perturb-seq + genetics grounded",
        "Perturb-seq + genetics",
        out_path or f"outputs/program_landscape_{_file_suffix}.png",
    )
    if any(prov_pm.values()):
        _build_and_save(
            prov_pm, "#8E44AD", "Genetics only  (provisional_virtual, no KO β)",
            "Genetics only  ⚠ GWAS / L2G, no Perturb-seq KO",
            f"outputs/program_landscape_{_file_suffix}_provisional.png",
            title_extra_fn=lambda p: "  ⚠ GWAS only",
        )
    else:
        print(f"Skipping provisional plot — no provisional_virtual genes for {disease_key}")

    # ── Benchmark check (rank-based) ───────────────────────────────────────────
    # With unsigned |τ*| gammas all OTA values are ≥ 0, so sign comparison is
    # meaningless. Pass criterion: rank ≤ top-15% of all targets (rank_threshold).
    _all_val = {v["gene"]: v for v in (list(drug_target_sign.items()) and [])}  # reset
    try:
        from config.drug_target_registry import get_validated_targets
        _all_val = {v["gene"]: v for v in get_validated_targets(disease_key.upper())}
    except Exception:
        pass
    if _all_val and targets:
        _t_map = {t["target_gene"]: t for t in targets}
        _n_total_targets = len(targets)
        _rank_threshold = max(50, int(_n_total_targets * 0.15))
        _pass = _miss = 0
        rows = []
        for gene, meta in sorted(_all_val.items()):
            t = _t_map.get(gene)
            exp_lbl = "±" if meta.get("expected_sign", 0) == 0 else ("+" if meta.get("expected_sign", 0) > 0 else "-")
            if t is None:
                _miss += 1
                rows.append(f"  {'MISS':6s}  {gene:<12s}  exp={exp_lbl}  not in targets")
                continue
            rank = t.get("rank", 9999)
            gm_raw = t.get(gamma_field)
            gm = float("nan") if gm_raw is None else float(gm_raw)
            if _math.isnan(gm):
                _miss += 1
                rows.append(f"  {'NaN':6s}  {gene:<12s}  exp={exp_lbl}  ota_gamma=NaN  rank={rank}")
                continue
            ok = rank <= _rank_threshold
            flag = "✓" if ok else "✗"
            if ok:
                _pass += 1
            rows.append(
                f"  {flag:6s}  {gene:<12s}  exp={exp_lbl}  γ={gm:+.3f}  rank={rank}"
            )
        total = len(_all_val)
        print(f"\nBenchmark ({disease_key.upper()}, {condition or 'default'})  "
              f"{_pass}/{total} in top-{_rank_threshold}  {_miss} missing")
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


def plot_program_comparison(
    disease_key: str = "ra",
    condition: str = "Stim48hr",
    prog_a: str | None = None,   # override auto-detected Th17 program
    prog_b: str | None = None,   # override auto-detected Th2 program
    out_path: str | None = None,
) -> None:
    """
    Head-to-head scatter: program A (Th17) vs program B (Th2/reg) contribution
    for every validated drug target.

    x = β × γ_B  (e.g. Th2/reg contribution)
    y = β × γ_A  (e.g. Th17 contribution)
    color = fingerprint r (KO phenotype reversal)
    Diagonal = equal contribution to both programs.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    import numpy as np
    import pathlib

    from pipelines.ldsc.gamma_loader import get_genetic_nmf_program_gammas
    from config.drug_target_registry import get_validated_targets

    # ── Load data ─────────────────────────────────────────────────────────────
    _DISEASE_CFG = {
        "ra":  {"dataset_id": "czi_2025_cd4t_perturb", "label": "Rheumatoid arthritis"},
        "cad": {"dataset_id": "czi_2025_cardiac_endothelial", "label": "Coronary artery disease"},
    }
    cfg = _DISEASE_CFG[disease_key.lower()]
    dataset_dir = pathlib.Path(f"data/perturbseq/{cfg['dataset_id']}")

    _cond_lower = condition.lower()
    loadings_path = dataset_dir / f"genetic_nmf_loadings_{_cond_lower}.npz"
    if not loadings_path.exists():
        print(f"plot_program_comparison: {loadings_path} not found")
        return

    _ld       = np.load(str(loadings_path), allow_pickle=True)
    beta_m    = _ld["Vt"].T.copy()     # n_perts × k
    k         = beta_m.shape[1]
    _pert_raw = [str(g) for g in _ld["pert_names"]]
    U_scaled  = _ld["U_scaled"] if "U_scaled" in _ld else None
    gene_names = [str(g) for g in _ld["gene_names"]] if "gene_names" in _ld else []

    ko_to_row: dict[str, int] = {}
    for i, g in enumerate(_pert_raw):
        ko_to_row[g] = i
        base = g[:-len("_Stim48hr")] if g.endswith("_Stim48hr") else (
               g[:-len("_REST")]     if g.endswith("_REST") else g)
        if base != g:
            ko_to_row.setdefault(base, i)

    _raw_gammas  = get_genetic_nmf_program_gammas(disease_key.upper(), condition=condition)
    prog_gammas  = {p: v["gamma"] for p, v in _raw_gammas.items() if v.get("gamma") is not None}
    full_progs   = [f"{disease_key.upper()}_GeneticNMF_{condition}_C{c+1:02d}" for c in range(k)]

    fp_r_all = _load_fp_r(cfg["dataset_id"])

    all_val       = get_validated_targets(disease_key.upper())
    drug_tgt_sign = {v["gene"]: v["expected_sign"] for v in all_val if v.get("status") != "Research"}

    # ── Auto-detect programs by top-gene bio label ────────────────────────────
    def _top_genes_prog(ci: int, n: int = 6) -> list[str]:
        if U_scaled is None or not gene_names:
            return []
        idx = np.argsort(U_scaled[:, ci])[::-1][:n]
        return [gene_names[i] for i in idx]

    def _bio_label_prog(ci: int) -> str:
        top = set(_top_genes_prog(ci))
        for markers, label in _BIO_MARKERS:
            if top & markers:
                return label
        return ""

    if prog_a is None or prog_b is None:
        _bio_map: dict[str, list[int]] = {}
        for ci in range(k):
            lbl = _bio_label_prog(ci)
            if lbl:
                _bio_map.setdefault(lbl, []).append(ci)
        # Pick highest-γ program per label
        def _best_ci(label: str) -> int | None:
            cands = _bio_map.get(label, [])
            if not cands:
                return None
            return max(cands, key=lambda ci: abs(prog_gammas.get(full_progs[ci], 0.0)))

        ci_a = _best_ci("Th17") if prog_a is None else \
               next((i for i, fp in enumerate(full_progs) if fp.endswith(prog_a)), None)
        ci_b = _best_ci("Th2/reg") if prog_b is None else \
               next((i for i, fp in enumerate(full_progs) if fp.endswith(prog_b)), None)

        if ci_a is None or ci_b is None:
            print(f"plot_program_comparison: could not auto-detect Th17/Th2 programs. "
                  f"bio_map={list(_bio_map)}")
            return

        fp_a = full_progs[ci_a]
        fp_b = full_progs[ci_b]
    else:
        fp_a = next((fp for fp in full_progs if fp.endswith(prog_a)), None)
        fp_b = next((fp for fp in full_progs if fp.endswith(prog_b)), None)
        ci_a = full_progs.index(fp_a) if fp_a else None
        ci_b = full_progs.index(fp_b) if fp_b else None
        if ci_a is None or ci_b is None:
            print(f"plot_program_comparison: prog_a={prog_a} or prog_b={prog_b} not found")
            return

    ga = prog_gammas.get(fp_a, 0.0)
    gb = prog_gammas.get(fp_b, 0.0)
    lbl_a = f"{fp_a.rsplit('_',1)[-1]} {_bio_label_prog(ci_a) or ''}  (γ={ga:.2f})"
    lbl_b = f"{fp_b.rsplit('_',1)[-1]} {_bio_label_prog(ci_b) or ''}  (γ={gb:.2f})"
    top_a = " · ".join(_top_genes_prog(ci_a, 4)[:3])
    top_b = " · ".join(_top_genes_prog(ci_b, 4)[:3])

    # ── Collect per-gene contributions ────────────────────────────────────────
    genes, contrib_a, contrib_b, fp_r, signs = [], [], [], [], []
    for gene, exp_sign in sorted(drug_tgt_sign.items()):
        row = ko_to_row.get(gene)
        if row is None:
            continue
        ca = float(beta_m[row, ci_a]) * ga
        cb = float(beta_m[row, ci_b]) * gb
        r  = fp_r_all.get(gene, float("nan"))
        genes.append(gene)
        contrib_a.append(ca)
        contrib_b.append(cb)
        fp_r.append(r)
        signs.append(exp_sign)

    if not genes:
        print("plot_program_comparison: no drug targets with KO data")
        return

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6.5, 6.5))

    # Diagonal = equal contribution
    _mx = max(max(contrib_a + [0.05]), max(contrib_b + [0.05])) * 1.15
    ax.plot([0, _mx], [0, _mx], ls="--", lw=0.9, color="#AAAAAA", zorder=1,
            label="equal contribution")

    # Color by fingerprint_r (diverging: blue=reversal, red=mimics disease)
    _r_vals = [v for v in fp_r if not (v != v)]  # filter NaN
    _r_min  = min(_r_vals) if _r_vals else -1.0
    _r_max  = max(_r_vals) if _r_vals else  1.0
    _norm   = mcolors.TwoSlopeNorm(vmin=_r_min, vcenter=0.0, vmax=_r_max)
    _cmap   = cm.RdBu_r

    import math
    for gene, ca, cb, r, sign in zip(genes, contrib_a, contrib_b, fp_r, signs):
        is_nan_r = math.isnan(r)
        dot_c    = _cmap(_norm(r)) if not is_nan_r else "#AAAAAA"
        mk = "D" if sign != 0 else "o"
        ec = "#1A1A1A"
        ax.scatter(cb, ca, c=[dot_c], s=90, marker=mk,
                   edgecolors=ec, linewidths=1.0, zorder=4)
        ax.annotate(gene, (cb, ca), xytext=(5, 3), textcoords="offset points",
                    fontsize=8, fontweight="bold",
                    color=("#1E8449" if sign == -1 else "#922B21" if sign == 1 else "#333"))

    # Colorbar
    sm = cm.ScalarMappable(cmap=_cmap, norm=_norm)
    sm.set_array([])
    cb_ = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cb_.set_label("Fingerprint r  (KO vs disease DEG)\n← reverses disease   |   mimics disease →",
                  fontsize=7.5)

    # Quadrant annotations
    ax.text(_mx * 0.02, _mx * 0.88, f"Th17 dominant\n({lbl_a.split('(')[0].strip()})",
            fontsize=7.5, color="#1A5276", style="italic", va="top")
    ax.text(_mx * 0.55, _mx * 0.12, f"Th2/reg dominant\n({lbl_b.split('(')[0].strip()})",
            fontsize=7.5, color="#784212", style="italic")

    ax.set_xlim(left=0, right=_mx)
    ax.set_ylim(bottom=0, top=_mx)
    ax.set_xlabel(f"β × γ  on  {lbl_b}\n({top_b})", fontsize=8.5)
    ax.set_ylabel(f"β × γ  on  {lbl_a}\n({top_a})", fontsize=8.5)
    ax.set_title(
        f"{cfg['label']} — Th17 vs Th2/reg program contribution\n"
        f"Approved drug targets  ·  {condition}\n"
        f"Color = fingerprint r  |  ◆ = directional drug target",
        fontsize=9, fontweight="bold",
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    default = f"outputs/program_comparison_{disease_key}_{_cond_lower}.png"
    plt.tight_layout()
    plt.savefig(out_path or default, dpi=160, bbox_inches="tight")
    print(f"Saved → {out_path or default}")
    plt.close(fig)


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
