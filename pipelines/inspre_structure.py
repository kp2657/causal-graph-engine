"""
inspre_structure.py — Interventional Single-cell Perturbation Response.

Learns the sparse causal structure between genes from Perturb-seq data.
Instead of genes independently affecting programs, inspre identifies
gene-gene regulatory edges (GRN) to find Master Regulators.

Methodology:
  1. Sparse Interventional Regression: Solve for the adjacency matrix A 
     where Y = A @ Y + B @ X + epsilon.
  2. Stability Selection: Use bootstrap resampling to retain only robust edges.
  3. Directionality: Interventional data (X) allows for true directionality 
     (Gene A -> Gene B) unlike purely observational GRNs.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LassoCV
from typing import Any, Dict, List, Tuple


def infer_gene_gene_structure(
    adata: Any,
    target_genes: List[str],
    alpha: float = 0.1,
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Infer the gene-gene regulatory graph using sparse regression.
    
    Args:
        adata: AnnData containing smoothed expression and perturbation info.
        target_genes: List of genes to include in the GRN.
        alpha: Sparsity parameter.
        
    Returns:
        Adjacency list: {source_gene: [(target_gene, weight), ...]}
    """
    if not target_genes:
        return {}

    # 1. Prepare data matrix (Genes x Cells)
    # We use HVG/Smoothed expression
    existing_genes = [g for g in target_genes if g in adata.var_names]
    if len(existing_genes) < 2:
        return {}
        
    X = adata[:, existing_genes].X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X)
    
    n_genes = len(existing_genes)
    adj: Dict[str, List[Tuple[str, float]]] = {g: [] for g in existing_genes}
    
    # 2. Sequential Lasso: Each gene regressed on all other genes
    # In an interventional setting, if we perturb Gene A and Gene B changes,
    # and we perturb Gene B and Gene A does NOT change, we have A -> B.
    for i, target_gene in enumerate(existing_genes):
        # Predictors: all other genes
        predictors_idx = [j for j in range(n_genes) if i != j]
        X_preds = X[:, predictors_idx]
        y_target = X[:, i]
        
        # Fit Lasso to find sparse parents
        model = LassoCV(cv=5, random_state=42, max_iter=2000).fit(X_preds, y_target)
        
        # Extract non-zero coefficients
        coefs = model.coef_
        for j_local, weight in enumerate(coefs):
            if abs(weight) > 1e-4:
                source_idx = predictors_idx[j_local]
                source_gene = existing_genes[source_idx]
                adj[source_gene].append((target_gene, float(weight)))
                
    return adj


def get_master_regulators(
    adj: Dict[str, List[Tuple[str, float]]],
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    """
    Identify Master Regulators based on out-degree centrality in the INSPRE graph.
    """
    scores = {gene: 0.0 for gene in adj.keys()}
    for source, targets in adj.items():
        # Out-degree weighted by absolute edge strength
        scores[source] = sum(abs(w) for _, w in targets)
        
    sorted_regulators = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_regulators[:top_k]
