"""
pipelines/state_space/tau_specificity.py — Disease-state τ specificity index.

Computes Yanai 2005 τ specificity from scRNA-seq expression across disease groups
within a single cell-type population (e.g., IBD vs normal macrophages).

This is orthogonal to the cross-tissue GTEx τ in single_cell_server.py, which asks
"is this gene gut-specific across tissues?"  Disease-state τ asks "within gut macrophages,
is this gene specifically expressed in IBD cells vs normal cells?"

Public API
----------
    compute_disease_tau(adata, gene_list, disease_col, normal_label)
        -> dict[str, TauResult]

    classify_specificity(tau, log2fc)
        -> str   "disease_specific" | "normal_specific" | "ubiquitous" |
                 "moderately_specific" | "lowly_expressed"
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


# Minimum mean expression to consider a gene expressed in a group
_MIN_MEAN_EXPR = 0.05
# Minimum fraction of cells expressing a gene (count > 0) for it to be considered
_MIN_FRACTION   = 0.02


@dataclass
class TauResult:
    """Per-gene disease-state τ specificity result."""
    gene: str

    # τ across disease groups (Yanai 2005): 0 = ubiquitous, 1 = single-group specific
    tau_disease: float = 0.5

    # log2FC: disease_mean / normal_mean (pseudocount added)
    disease_log2fc: float = 0.0

    # Fraction of disease cells (non-normal) expressing this gene (count > 0)
    pct_disease: float = 0.0

    # Fraction of normal cells expressing this gene
    pct_normal: float = 0.0

    # Mean expression in disease cells (log-normalised)
    mean_disease: float = 0.0

    # Mean expression in normal cells (log-normalised)
    mean_normal: float = 0.0

    # Number of disease groups (n in τ formula)
    n_groups: int = 0

    # Specificity class
    specificity_class: str = "unknown"

    # Group breakdown: {disease_label: mean_expression}
    group_means: dict[str, float] = field(default_factory=dict)


def classify_specificity(tau: float, log2fc: float) -> str:
    """
    Classify a gene's disease-state specificity.

    Returns one of:
        disease_specific    τ ≥ 0.6 AND log2fc > 1.0
        normal_specific     τ ≥ 0.6 AND log2fc < -1.0
        moderately_specific τ in [0.3, 0.6)
        ubiquitous          τ < 0.3
        lowly_expressed     (set externally when both group means < _MIN_MEAN_EXPR)
    """
    if tau >= 0.60:
        if log2fc > 1.0:
            return "disease_specific"
        if log2fc < -1.0:
            return "normal_specific"
        return "moderately_specific"
    if tau < 0.30:
        return "ubiquitous"
    return "moderately_specific"


def _tau_from_means(group_means: np.ndarray) -> float:
    """
    Yanai 2005 τ formula from a vector of group mean expressions.

    τ = (n − Σ(x_i / x_max)) / (n − 1)
    """
    n = len(group_means)
    if n <= 1:
        return 0.0
    x_max = group_means.max()
    if x_max < 1e-9:
        return 0.0
    return float(np.clip((n - (group_means / x_max).sum()) / (n - 1), 0.0, 1.0))


def compute_disease_tau(
    adata: Any,
    gene_list: list[str] | None = None,
    disease_col: str = "disease",
    normal_label: str = "normal",
) -> dict[str, TauResult]:
    """
    Compute disease-state τ specificity for each gene in gene_list.

    Groups cells by `disease_col` values.  Computes per-group mean expression,
    then applies Yanai 2005 τ across groups.  Also reports log2FC(disease/normal).

    Args:
        adata:        AnnData with obs[disease_col] and normalized expression in .X.
        gene_list:    Genes to score.  None → all genes in adata.var_names.
        disease_col:  obs column with disease group labels.
        normal_label: Label that identifies healthy/normal cells.

    Returns:
        {gene: TauResult}. Genes absent from adata.var_names are omitted.
    """
    import scipy.sparse as sp

    if disease_col not in adata.obs.columns:
        # No disease column available — return neutral τ for all genes
        genes = gene_list if gene_list is not None else list(adata.var_names)
        return {g: TauResult(gene=g, tau_disease=0.5, specificity_class="unknown") for g in genes}

    # Resolve gene indices
    var_index = {g: i for i, g in enumerate(adata.var_names)}
    if gene_list is not None:
        target_genes = [g for g in gene_list if g in var_index]
    else:
        target_genes = list(adata.var_names)

    if not target_genes:
        return {}

    gene_indices = np.array([var_index[g] for g in target_genes], dtype=np.int64)

    # Get expression matrix subset (n_cells × n_target_genes)
    X = adata.X
    if sp.issparse(X):
        X_sub = np.asarray(X[:, gene_indices].todense(), dtype=np.float32)
    else:
        X_sub = np.asarray(X[:, gene_indices], dtype=np.float32)

    # Disease group labels — handle both pandas Series and numpy arrays
    _raw_labels = adata.obs[disease_col]
    if hasattr(_raw_labels, "values"):
        labels = _raw_labels.astype(str).values
    else:
        labels = np.asarray(_raw_labels, dtype=str)
    unique_labels = sorted(set(labels))

    # Group cells: normal vs all disease groups
    # Build mask per group
    group_masks: dict[str, np.ndarray] = {}
    for lab in unique_labels:
        mask = labels == lab
        if mask.sum() >= 10:  # require at least 10 cells per group
            group_masks[lab] = mask

    if len(group_masks) < 2:
        # Only one group — τ is undefined; return neutral
        return {g: TauResult(gene=g, tau_disease=0.5, specificity_class="unknown")
                for g in target_genes}

    group_names = sorted(group_masks.keys())
    n_groups = len(group_names)

    # Compute per-group mean expression: shape (n_groups, n_target_genes)
    mean_matrix = np.zeros((n_groups, len(target_genes)), dtype=np.float64)
    pct_matrix  = np.zeros((n_groups, len(target_genes)), dtype=np.float64)
    for i, lab in enumerate(group_names):
        mask = group_masks[lab]
        cells = X_sub[mask]  # shape (n_cells_in_group, n_genes)
        mean_matrix[i] = cells.mean(axis=0)
        pct_matrix[i]  = (cells > 0).mean(axis=0)

    # Vectorized τ: for each gene (column), compute τ over group means
    # Yanai 2005: τ = (n − Σ(x_i/x_max)) / (n − 1)
    x_max = mean_matrix.max(axis=0)  # shape (n_genes,)
    # Avoid division by zero
    x_max_safe = np.where(x_max > 1e-9, x_max, 1.0)
    tau_vec = np.clip(
        (n_groups - (mean_matrix / x_max_safe).sum(axis=0)) / (n_groups - 1),
        0.0, 1.0
    )
    # Where x_max ~ 0, all groups are essentially silent → ubiquitous (τ=0 is correct)
    tau_vec = np.where(x_max < 1e-9, 0.0, tau_vec)

    # Normal vs disease means (aggregate non-normal groups)
    normal_idx = group_names.index(normal_label) if normal_label in group_names else None
    if normal_idx is not None:
        normal_mean = mean_matrix[normal_idx]
        disease_rows = [i for i, lab in enumerate(group_names) if lab != normal_label]
        if disease_rows:
            disease_mean = mean_matrix[disease_rows].mean(axis=0)
            disease_pct  = pct_matrix[disease_rows].mean(axis=0)
            normal_pct   = pct_matrix[normal_idx]
        else:
            disease_mean = normal_mean
            disease_pct  = pct_matrix[normal_idx]
            normal_pct   = pct_matrix[normal_idx]
    else:
        # No normal group found — use group 0 as reference
        normal_mean  = mean_matrix[0]
        normal_pct   = pct_matrix[0]
        disease_mean = mean_matrix[1:].mean(axis=0) if n_groups > 1 else mean_matrix[0]
        disease_pct  = pct_matrix[1:].mean(axis=0) if n_groups > 1 else pct_matrix[0]

    # log2FC: disease_mean / normal_mean (pseudocount 0.1 to avoid log(0))
    log2fc_vec = np.log2((disease_mean + 0.1) / (normal_mean + 0.1))

    # Assemble results
    results: dict[str, TauResult] = {}
    for j, gene in enumerate(target_genes):
        tau_val   = float(tau_vec[j])
        log2fc    = float(log2fc_vec[j])
        d_mean    = float(disease_mean[j])
        n_mean    = float(normal_mean[j])
        d_pct     = float(disease_pct[j])
        n_pct     = float(normal_pct[j])

        # Lowly expressed if both groups have mean < threshold
        if d_mean < _MIN_MEAN_EXPR and n_mean < _MIN_MEAN_EXPR:
            cls = "lowly_expressed"
        else:
            cls = classify_specificity(tau_val, log2fc)

        group_means_dict = {
            lab: float(mean_matrix[i, j])
            for i, lab in enumerate(group_names)
        }

        results[gene] = TauResult(
            gene=gene,
            tau_disease=tau_val,
            disease_log2fc=log2fc,
            pct_disease=d_pct,
            pct_normal=n_pct,
            mean_disease=d_mean,
            mean_normal=n_mean,
            n_groups=n_groups,
            specificity_class=cls,
            group_means=group_means_dict,
        )

    return results
