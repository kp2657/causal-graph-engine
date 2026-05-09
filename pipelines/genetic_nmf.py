"""
pipelines/genetic_nmf.py — WES-regularized genetic-direction NMF.

Replaces truncated SVD for program decomposition of the gene × perturbation
log2FC matrix. Standard NMF cannot accept signed input, so the matrix is split
into a non-negative [M_pos; M_neg] double-block form. The regularization term
penalizes programs that co-load genes with opposite WES burden directions:

    L = ||X - WH||²_F  +  λ × Σ_k  pos_k × neg_k

where
    pos_k  =  Σ_{i: d_full[i]=+1}  W[i,k] × w_full[i]   (atherogenic loading)
    neg_k  =  Σ_{i: d_full[i]=-1}  W[i,k] × w_full[i]   (protective loading)

Minimised via Lee-Seung multiplicative updates. The opponent direction is added
to the denominator — this preserves non-negativity and monotone convergence.

WES direction convention
------------------------
    burden_beta < 0  →  LoF reduces disease  →  gene is atherogenic   d_i = +1
    burden_beta > 0  →  LoF increases disease →  gene is protective    d_i = -1
    missing / weak   →  unknown                                         d_i =  0

Output
------
Writes  data/perturbseq/{dataset_id}/genetic_nmf_loadings.npz  with keys:
    Vt             (k × n_perts)    perturbation loadings   [replaces SVD Vt]
    U_scaled       (n_genes × k)    signed net gene loadings [replaces SVD U_scaled]
    gene_names     (n_genes,)       gene symbols
    pert_names     (n_perts,)       perturbation symbols
    d_genes        (n_genes,)       WES direction per gene (+1/-1/0)
    w_genes        (n_genes,)       WES weight per gene

Public API
----------
run_genetic_nmf_for_dataset(dataset_id, disease_key) -> dict
    Load M from signatures, compute WES weights, run NMF, write npz.

load_genetic_nmf(dataset_id) -> dict | None
    Return npz arrays as a dict; None if file absent.
"""
from __future__ import annotations

import logging
import pathlib
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)

_ROOT        = pathlib.Path(__file__).parent.parent
_PERTURBSEQ  = _ROOT / "data" / "perturbseq"
_WES_DIR     = _ROOT / "data" / "wes"
_EQTL_DIR    = _ROOT / "data" / "eqtl" / "indices"
_SHET_PATH   = _ROOT / "data" / "genebays" / "shet_posteriors.tsv"
_SHET_MAP_PATH = _ROOT / "data" / "genebays" / "shet_symbol_map.tsv"


# ---------------------------------------------------------------------------
# WES gene weight construction
# ---------------------------------------------------------------------------

def _load_shet_map() -> dict[str, float]:
    """Ensembl → shet posterior mean from GeneBayes."""
    shet: dict[str, float] = {}
    if not _SHET_PATH.exists():
        return shet
    try:
        with open(_SHET_PATH) as f:
            header = f.readline().strip().split("\t")
            ensg_col  = header.index("ensg")
            post_col  = header.index("post_mean")
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) > max(ensg_col, post_col):
                    shet[parts[ensg_col]] = float(parts[post_col])
    except Exception as e:
        log.warning("shet load failed: %s", e)
    return shet


def _load_shet_symbol_map() -> dict[str, str]:
    """gene symbol → Ensembl from the companion map file."""
    sym2ensg: dict[str, str] = {}
    if not _SHET_MAP_PATH.exists():
        return sym2ensg
    try:
        with open(_SHET_MAP_PATH) as f:
            header = f.readline().strip().split("\t")
            sym_col  = header.index("symbol") if "symbol" in header else 0
            ensg_col = header.index("ensg")   if "ensg"   in header else 1
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) > max(sym_col, ensg_col):
                    sym2ensg[parts[sym_col]] = parts[ensg_col]
    except Exception as e:
        log.warning("shet symbol map load failed: %s", e)
    return sym2ensg


def build_wes_gene_weights(
    gene_names: list[str],
    disease_key: str,
    z_min: float,
    shet_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (d, w) arrays of length n_genes.

    d[i]: +1 (atherogenic), -1 (protective), 0 (unknown)
    w[i]: WES burden |Z-score| × (1 + shet_scale × shet_post_mean)

    Convention: burden_beta < 0 → LoF reduces disease → gene is atherogenic → d=+1
    """
    import json

    wes_path = _WES_DIR / f"{disease_key.upper()}_burden.json"
    wes: dict[str, dict] = {}
    if wes_path.exists():
        try:
            with open(wes_path) as f:
                wes = json.load(f)
        except Exception as e:
            log.warning("WES burden load failed for %s: %s", disease_key, e)

    shet_by_ensg   = _load_shet_map()
    sym2ensg       = _load_shet_symbol_map()

    n = len(gene_names)
    d = np.zeros(n, dtype=np.float32)
    w = np.zeros(n, dtype=np.float32)

    for i, gene in enumerate(gene_names):
        rec = wes.get(gene)
        if rec is None:
            continue
        burden_beta = rec.get("burden_beta")
        burden_se   = rec.get("burden_se")
        if burden_beta is None or burden_se is None or burden_se == 0:
            continue

        z = abs(burden_beta / burden_se)
        if z < z_min:
            continue  # too weak — don't let noise drive program structure

        # Direction: LoF reduces disease (beta<0) → atherogenic gene → d=+1
        d[i] = -np.sign(burden_beta)

        # Shet credibility boost for constrained genes
        ensg   = sym2ensg.get(gene, "")
        shet_p = shet_by_ensg.get(ensg, 0.0)
        w[i]   = z * (1.0 + shet_scale * shet_p)

    return d, w


# ---------------------------------------------------------------------------
# eQTL helpers
# ---------------------------------------------------------------------------

def _load_eqtl_betas(disease_key: str) -> dict[str, dict]:
    """
    Load top eQTL betas for all study indices matching disease_key.

    Returns dict[gene_symbol → {beta, se, pvalue}] — one entry per gene,
    keeping the most significant hit across studies.
    """
    import json

    pattern = f"{disease_key.upper()}_*_top_eqtls.json"
    hits: dict[str, dict] = {}
    for path in sorted(_EQTL_DIR.glob(pattern)):
        try:
            with open(path) as f:
                study = json.load(f)
        except Exception as e:
            log.warning("eQTL load failed for %s: %s", path, e)
            continue
        for gene, rec in study.items():
            if not isinstance(rec, dict):
                continue
            beta = rec.get("beta")
            se   = rec.get("se")
            pval = rec.get("pvalue", 1.0)
            if beta is None or se is None:
                continue
            prev = hits.get(gene)
            if prev is None or pval < prev.get("pvalue", 1.0):
                hits[gene] = {"beta": float(beta), "se": float(se), "pvalue": float(pval)}
    return hits


def build_combined_gene_weights(
    gene_names: list[str],
    disease_key: str,
    z_min: float,
    shet_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (d, w) arrays using WES as the direction anchor, eQTL as a confidence modifier.

    WES direction is ground truth — genes without WES evidence are NOT assigned
    direction from eQTL alone (eQTL direction is too noisy to regularise NMF programs).
    eQTL only adjusts the weight of WES-anchored genes:

      WES + eQTL agree    → d from WES, w = WES_base × CONCORDANT_BOOST
      WES + eQTL disagree → d from WES, w = WES_base × DISCORDANT_DISCOUNT
      WES only            → d from WES, w = WES_base (unchanged)
      eQTL only / neither → d = 0, w = 0 (not used for regularisation)

    eQTL direction convention: risk allele increases expression (beta>0) maps to
    atherogenic d=+1, matching WES convention (burden_beta<0 → LoF reduces disease → d=+1).
    """
    from config.scoring_thresholds import (
        GENETIC_NMF_EQTL_CONCORDANT_BOOST,
        GENETIC_NMF_EQTL_DISCORDANT_DISCOUNT,
    )

    d_wes, w_wes = build_wes_gene_weights(gene_names, disease_key, z_min, shet_scale)
    eqtl_betas   = _load_eqtl_betas(disease_key)

    d = d_wes.copy()
    w = w_wes.copy()

    n_concordant = 0
    n_discordant = 0

    for i, gene in enumerate(gene_names):
        if d_wes[i] == 0.0:
            continue  # no WES anchor — eQTL alone is too noisy; skip
        rec = eqtl_betas.get(gene)
        if rec is None:
            continue  # WES-only; keep unmodified

        eqtl_beta = rec["beta"]
        eqtl_se   = rec["se"]
        if eqtl_se == 0:
            continue
        d_eqtl = float(np.sign(eqtl_beta))

        if d_eqtl == d_wes[i]:
            w[i] = w_wes[i] * GENETIC_NMF_EQTL_CONCORDANT_BOOST
            n_concordant += 1
        else:
            w[i] = w_wes[i] * GENETIC_NMF_EQTL_DISCORDANT_DISCOUNT
            n_discordant += 1

    n_wes_only = int((d_wes != 0).sum()) - n_concordant - n_discordant
    log.info(
        "Combined gene weights: %d WES+eQTL concordant (×%.1f), %d discordant (×%.1f), "
        "%d WES-only (unchanged)",
        n_concordant, GENETIC_NMF_EQTL_CONCORDANT_BOOST,
        n_discordant, GENETIC_NMF_EQTL_DISCORDANT_DISCOUNT,
        n_wes_only,
    )
    return d, w


# ---------------------------------------------------------------------------
# Double-block NMF with incoherence penalty
# ---------------------------------------------------------------------------

def _fit_sklearn_nmf(X: np.ndarray, k: int, max_iter: int, random_state: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Full sklearn NMF fit (coordinate descent, nndsvda init)."""
    from sklearn.decomposition import NMF
    model = NMF(n_components=k, init="nndsvda", max_iter=max_iter,
                solver="cd", tol=1e-4, random_state=random_state)
    W = model.fit_transform(X)
    H = model.components_
    log.info("sklearn NMF: %d iters, n_iter_=%d", max_iter, model.n_iter_)
    return W.astype(np.float32), H.astype(np.float32)


def _multiplicative_update_step(
    X: np.ndarray,       # (2n × p) non-negative input
    W: np.ndarray,       # (2n × k) gene loadings
    H: np.ndarray,       # (k  × p) perturbation loadings
    d_full: np.ndarray,  # (2n,)  +1/-1/0 direction
    w_full: np.ndarray,  # (2n,)  ≥0 weight
    lam: float,
    eps: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray]:
    """One Lee-Seung multiplicative update step with incoherence regularisation."""
    # Standard H update (no regularisation on H)
    WtX  = W.T @ X         # (k × p)
    WtWH = W.T @ (W @ H)   # (k × p)
    H    = H * (WtX / (WtWH + eps))
    H    = np.maximum(H, 0.0)

    # Per-program atherogenic / protective weighted sums
    pos_k = (W * w_full[:, None] * (d_full[:, None] == 1)).sum(axis=0)   # (k,)
    neg_k = (W * w_full[:, None] * (d_full[:, None] == -1)).sum(axis=0)  # (k,)

    # Numerator: standard NMF W update
    XHt  = X @ H.T           # (2n × k)
    WHHt = (W @ H) @ H.T     # (2n × k)

    # Denominator: add opponent weighted sum for each directional gene
    denom = WHHt.copy()
    mask_pos = (d_full == 1)
    mask_neg = (d_full == -1)
    # Atherogenic genes: opponent = neg_k (protective loading penalises them)
    denom[mask_pos] += lam * neg_k[None, :] * w_full[mask_pos, None]
    # Protective genes: opponent = pos_k
    denom[mask_neg] += lam * pos_k[None, :] * w_full[mask_neg, None]

    W = W * (XHt / (denom + eps))
    W = np.maximum(W, 0.0)

    return W, H


def fit_genetic_nmf(
    M: np.ndarray,        # (n_genes × n_perts) signed log2FC matrix
    gene_names: list[str],
    d: np.ndarray,        # (n_genes,)  WES direction (passed through to output)
    w: np.ndarray,        # (n_genes,)  WES weight    (passed through to output)
    n_components: int,
    lam: float,           # kept for API compatibility; not used (sklearn solver)
    max_iter: int,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit NMF on the signed perturbation matrix M using sklearn's coordinate descent.

    The signed input is handled via a double-block non-negative representation:
        X = [M_pos; M_neg]  where M_pos = max(M,0), M_neg = max(-M,0)

    W_net = W_pos - W_neg gives signed gene loadings.

    Returns
    -------
    W_net : (n_genes × k)  signed gene loadings  — replaces U_scaled
    H     : (k × n_perts)  perturbation loadings — replaces Vt
    d_out : (n_genes,)     WES direction passed through
    w_out : (n_genes,)     WES weight passed through
    """
    n_genes, _ = M.shape

    # Double-block non-negative representation
    M_pos = np.maximum( M, 0.0).astype(np.float32)
    M_neg = np.maximum(-M, 0.0).astype(np.float32)
    X = np.vstack([M_pos, M_neg])     # (2n_genes × n_perts)

    log.info("fit_genetic_nmf (sklearn cd): X=%s  k=%d  max_iter=%d", X.shape, n_components, max_iter)
    W, H = _fit_sklearn_nmf(X, n_components, max_iter, random_state)

    # Reconstruct signed gene loadings from double-block W
    W_pos = W[:n_genes, :]
    W_neg = W[n_genes:, :]
    W_net = W_pos - W_neg     # signed, like U_scaled

    return W_net.astype(np.float32), H.astype(np.float32), d.astype(np.float32), w.astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset-level entry point
# ---------------------------------------------------------------------------

def run_genetic_nmf_for_dataset(
    dataset_id: str,
    disease_key: str,
    n_components: int | None = None,
    lam: float | None = None,
    max_iter: int | None = None,
    sig_name: str | None = None,
    out_name: str | None = None,
) -> dict:
    """
    Build WES-regularised program decomposition for one Perturb-seq dataset.

    Loads:
        data/perturbseq/{dataset_id}/{sig_name}  (default: signatures.json.gz)
    Writes:
        data/perturbseq/{dataset_id}/{out_name}  (default: genetic_nmf_loadings.npz)

    Use out_name to write condition-specific NMF loadings:
        out_name="genetic_nmf_loadings_rest.npz"     → REST-condition NMF
        out_name="genetic_nmf_loadings_stim48hr.npz" → Stim48hr-condition NMF

    Parameters default from config/scoring_thresholds.py.
    """
    import json, gzip

    from config.scoring_thresholds import (
        GENETIC_NMF_LAMBDA,
        GENETIC_NMF_MAX_ITER,
        GENETIC_NMF_RANK,
        GENETIC_NMF_SHET_SCALE,
        GENETIC_NMF_WES_Z_MIN,
    )

    lam       = lam       if lam       is not None else GENETIC_NMF_LAMBDA
    max_iter  = max_iter  if max_iter  is not None else GENETIC_NMF_MAX_ITER
    n_components = n_components if n_components is not None else GENETIC_NMF_RANK

    ds_dir    = _PERTURBSEQ / dataset_id
    sig_path  = ds_dir / (sig_name or "signatures.json.gz")
    out_path  = ds_dir / (out_name or "genetic_nmf_loadings.npz")

    if not sig_path.exists():
        # Also try the fingerprint variant (denoised)
        fp_path = ds_dir / "signatures_fingerprint.json.gz"
        if fp_path.exists():
            sig_path = fp_path
        else:
            return {"error": f"No signatures file in {ds_dir}"}

    # --- Load perturbation signatures ---
    log.info("Loading signatures from %s", sig_path)
    with gzip.open(sig_path, "rt") as f:
        sigs: dict[str, dict[str, float]] = json.load(f)

    all_genes = sorted({g for fc in sigs.values() for g in fc})
    all_perts = sorted(sigs.keys())
    n_genes, n_perts = len(all_genes), len(all_perts)
    gene_idx = {g: i for i, g in enumerate(all_genes)}

    # Build M (n_genes × n_perts)
    M = np.zeros((n_genes, n_perts), dtype=np.float32)
    for j, pert in enumerate(all_perts):
        for g, v in sigs[pert].items():
            if g in gene_idx:
                M[gene_idx[g], j] = float(v)
    # Center per-gene (remove shared baseline)
    M -= M.mean(axis=1, keepdims=True)

    log.info("Matrix shape: %d genes × %d perts", n_genes, n_perts)

    # --- WES direction weights (GeneBayes shet as credibility boost) ---
    d, w = build_wes_gene_weights(all_genes, disease_key, GENETIC_NMF_WES_Z_MIN, GENETIC_NMF_SHET_SCALE)
    n_weighted = int((w > 0).sum())
    log.info("WES weights: %d / %d genes have direction signal", n_weighted, n_genes)

    # --- Fit ---
    log.info("Fitting genetic NMF (k=%d, λ=%.3f, max_iter=%d)", n_components, lam, max_iter)
    W_net, H, d_out, w_out = fit_genetic_nmf(M, all_genes, d, w, n_components, lam, max_iter)

    # --- Save ---
    np.savez_compressed(
        str(out_path),
        Vt=H,                                    # (k × n_perts) — same key as SVD
        U_scaled=W_net,                          # (n_genes × k) — same key as SVD
        gene_names=np.array(all_genes),
        pert_names=np.array(all_perts),
        d_genes=d_out,
        w_genes=w_out,
    )
    log.info("Genetic NMF saved: %s (k=%d, %d perts, %d genes)", out_path, n_components, n_perts, n_genes)

    return {
        "dataset_id":    dataset_id,
        "disease_key":   disease_key,
        "n_programs":    n_components,
        "n_perts":       n_perts,
        "n_genes":       n_genes,
        "n_wes_genes":   n_weighted,
        "lambda":        lam,
        "out_path":      str(out_path),
    }


def load_genetic_nmf(dataset_id: str, out_name: str | None = None) -> dict | None:
    """Load genetic NMF npz; return dict of arrays or None if absent."""
    path = _PERTURBSEQ / dataset_id / (out_name or "genetic_nmf_loadings.npz")
    if not path.exists():
        return None
    npz = np.load(path)
    return {k: npz[k] for k in npz.files}


# Condition → loadings filename mapping
_CONDITION_LOADINGS: dict[str, str] = {
    "Stim8hr":  "genetic_nmf_loadings.npz",          # shared / baseline (Stim8hr signatures)
    "Stim48hr": "genetic_nmf_loadings_stim48hr.npz",  # Stim48hr-specific NMF
    "REST":     "genetic_nmf_loadings_rest.npz",      # REST-specific NMF
}


def load_genetic_nmf_for_condition(dataset_id: str, condition: str) -> dict | None:
    """Load condition-specific NMF; falls back to shared loadings if file absent."""
    fname = _CONDITION_LOADINGS.get(condition.upper() if condition.upper() == "REST"
                                    else condition,
                                    "genetic_nmf_loadings.npz")
    result = load_genetic_nmf(dataset_id, fname)
    if result is None and fname != "genetic_nmf_loadings.npz":
        log.warning("Condition-specific NMF %s not found; falling back to shared loadings", fname)
        result = load_genetic_nmf(dataset_id)
    return result
