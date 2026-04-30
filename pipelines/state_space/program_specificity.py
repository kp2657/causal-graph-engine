"""
pipelines/state_space/program_specificity.py

Compute disease-specific program activity weights w_P ∈ [0, 1].

The OTA formula Σ_P (β_gene→P × γ_P→disease) gives pleiotropic transcriptional
regulators inflated γ because they have large β across all programs, some of which
coincidentally carry GWAS signal.

Fix: weight each program's contribution by how disease-specific its activity is in
patient tissue:

    γ_refined = Σ_P (β_gene→P × γ_P→disease × w_P)

where w_P = how much more active program P is in disease cells vs. healthy controls.

A gene whose β is spread uniformly across generic programs gets low w_P weights.
A gene whose β is concentrated in disease-upregulated programs retains a large γ.

Important: this does NOT modify ota_gamma (backward compat). The result is surfaced
as ota_gamma_specificity_weighted alongside the raw ota_gamma.

Public API
----------
    compute_program_disease_weights(adata, programs, disease_col, disease_label)
        -> dict[str, float]   {program_id → w_P ∈ [0, 1]}

    compute_beta_program_concentration(beta_estimates, program_weights)
        -> float   ∈ [0, 1]  fraction of β-mass in disease-specific programs
"""
from __future__ import annotations

import math
from typing import Any


# Minimum cells required in each group (disease/healthy) to compute a reliable LFC.
_MIN_CELLS_PER_GROUP = 10

# Log2FC threshold above which w_P = 1.0 (saturates at 2-fold).
_MAX_LOG2FC = 2.0

# CELLxGENE MONDO-specific disease labels per disease key.
_CAD_DISEASE_LABELS = {
    "coronary artery disease",
    "coronary heart disease",
    "ischemic heart disease",
    "myocardial infarction",
    "atherosclerosis",
    "atherosclerosis of coronary artery",
    "acute myocardial infarction",
    "chronic ischemic heart disease",
}
_DISEASE_LABEL_MAP: dict[str, set[str]] = {
    "CAD": _CAD_DISEASE_LABELS,
}
_HEALTHY_LABELS = {"normal", "healthy", "control", "none"}


def compute_program_disease_weights(
    adata: Any,
    programs: list[dict],
    disease_col: str = "disease",
    disease_label: str | None = None,
    disease_key: str | None = None,
) -> dict[str, float]:
    """
    Compute disease-specificity weight w_P for each program.

    For each program P with a gene_set (or gene_loadings), compute:
      activity(cell, P) = mean( log-normalised expression of top genes in P )
      LFC_P = log2( mean_AMD(activity_P) + ε ) - log2( mean_healthy(activity_P) + ε )
      w_P   = clamp( LFC_P / MAX_LOG2FC, 0, 1 )

    Programs with ≥2-fold upregulation in disease cells → w_P = 1.0.
    Programs unchanged or downregulated             → w_P = 0.0.

    Args:
        adata:          AnnData with obs[disease_col] and log-normalised X.
                        
                        both disease patients and healthy controls).
        programs:       List of program dicts with keys:
                          "program_id": str
                          "gene_set":   list[str]  (or)
                          "gene_loadings": dict[str, float]
        disease_col:    Column in adata.obs containing disease labels.
        disease_label:  Explicit disease obs label override (takes priority).
        disease_key:    Short disease key (e.g. "AMD", "CAD", "IBD") used to
                        look up the appropriate label set when disease_label is None.

    Returns:
        {program_id: w_P}  — uniform 1.0 for all programs if fallback triggered.
    """
    try:
        import numpy as np

        if adata is None or not programs:
            return {}

        obs = adata.obs
        if disease_col not in obs.columns:
            return {}

        disease_vals = obs[disease_col].astype(str).str.lower()

        # Detect disease vs. healthy cell indices
        if disease_label:
            disease_mask = disease_vals == disease_label.lower()
        elif disease_key and disease_key.upper() in _DISEASE_LABEL_MAP:
            disease_mask = disease_vals.isin(_DISEASE_LABEL_MAP[disease_key.upper()])
        else:
            # No disease key — treat all non-healthy cells as disease
            disease_mask = ~disease_vals.isin(_HEALTHY_LABELS)
        disease_idx_mask = disease_mask

        healthy_mask = disease_vals.isin(_HEALTHY_LABELS)

        # Batch effect mitigation: prefer healthy cells from the same dataset(s) as disease
        # cells to avoid depth/batch LFC inflation.
        _BATCH_COLS = ("dataset_id", "batch", "study_id", "assay_ontology_term_id")
        _batch_col = next((c for c in _BATCH_COLS if c in obs.columns), None)
        if _batch_col is not None:
            _disease_datasets = set(obs[_batch_col][disease_idx_mask].unique())
            _matched_healthy = healthy_mask & obs[_batch_col].isin(_disease_datasets)
            if int(_matched_healthy.sum()) >= _MIN_CELLS_PER_GROUP:
                healthy_mask = _matched_healthy  # use dataset-matched subset

        n_disease = int(disease_idx_mask.sum())
        n_healthy = int(healthy_mask.sum())

        if n_disease < _MIN_CELLS_PER_GROUP or n_healthy < _MIN_CELLS_PER_GROUP:
            # Not enough contrast — uniform weights (no downweighting)
            return {p["program_id"]: 1.0 for p in programs if p.get("program_id")}

        disease_idx = np.where(disease_idx_mask.values)[0]
        healthy_idx = np.where(healthy_mask.values)[0]

        # Build gene name → column index lookup (var may use Ensembl IDs or symbols)
        var_names = list(adata.var_names)
        # Also try feature_name if var_names are Ensembl IDs
        symbol_to_col: dict[str, int] = {g: i for i, g in enumerate(var_names)}
        if "feature_name" in adata.var.columns:
            for i, sym in enumerate(adata.var["feature_name"]):
                if sym and sym not in symbol_to_col:
                    symbol_to_col[sym] = i

        X = adata.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)

        # Library-size normalisation (counts per 10k) + log1p.
        # Required before AMD/healthy comparison — otherwise depth differences between
        # single-dataset AMD cells and multi-dataset normal cells dominate the LFC signal.
        row_sums = X.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        X = np.log1p(X / row_sums * 1e4)

        weights: dict[str, float] = {}
        for prog in programs:
            pid = prog.get("program_id")
            if not pid:
                continue

            # Resolve gene set to column indices
            gene_loadings = prog.get("gene_loadings", {})
            gene_set = prog.get("gene_set") or list(gene_loadings.keys())
            col_idxs = [symbol_to_col[g] for g in gene_set if g in symbol_to_col]

            if len(col_idxs) < 3:
                weights[pid] = 1.0  # not enough overlap → don't penalise
                continue

            # Per-cell program activity = weighted mean of gene expression
            if gene_loadings:
                loading_vals = np.array([
                    gene_loadings.get(g, 1.0) for g in gene_set if g in symbol_to_col
                ], dtype=np.float32)
                loading_vals /= (loading_vals.sum() + 1e-8)
                prog_X = X[:, col_idxs] @ loading_vals   # shape: (n_cells,)
            else:
                prog_X = X[:, col_idxs].mean(axis=1)     # shape: (n_cells,)

            mean_disease = float(prog_X[disease_idx].mean())
            mean_healthy = float(prog_X[healthy_idx].mean())

            # mean_disease / mean_healthy are in log1p(CP10K) space.
            # LFC in log2 = (mean_disease - mean_healthy) / ln(2)
            lfc = (mean_disease - mean_healthy) / math.log(2)

            # w_P ∈ [0, 1]: clamp positive LFC to [0, MAX_LOG2FC], then normalise
            w = max(0.0, min(lfc, _MAX_LOG2FC)) / _MAX_LOG2FC
            weights[pid] = round(w, 4)

        return weights

    except Exception:
        # Non-fatal — return empty dict (caller uses default weight=1.0)
        return {}


def compute_pathway_coherence_score(
    beta_estimates: dict[str, Any],
    program_disease_weights: dict[str, float],
    program_labels: dict[str, dict] | None = None,
    min_weight_threshold: float = 0.3,
) -> dict:
    """
    Measure how well a gene's causal β-mass aligns with disease-relevant programs.

    A gene whose β is concentrated in programs that are (a) upregulated in disease
    tissue AND (b) annotated as disease-relevant gets a high coherence score.
    A gene like JAZF1 whose β is spread across generic programs gets a low score.

    Two components:
      specificity_alignment  = beta_program_concentration (already computed)
      label_coherence        = fraction of β-mass in programs with disease label

    Args:
        beta_estimates:          {program_id → {beta: float, ...} | float}
        program_disease_weights: {program_id → w_P ∈ [0,1]} from compute_program_disease_weights
        program_labels:          {program_id → {"disease_relevant": bool, "label": str, ...}}
                                 from label_programs() in program_labeler.py
        min_weight_threshold:    Program weight below this → "generic" (not disease-relevant)

    Returns:
        {
            "pathway_coherence_score": float ∈ [0, 1],
            "n_disease_programs":      int,  # programs with w_P >= threshold
            "n_labeled_programs":      int,  # programs with disease_relevant label
            "dominant_program":        str | None,
        }
    """
    if not beta_estimates:
        return {"pathway_coherence_score": 0.0, "n_disease_programs": 0,
                "n_labeled_programs": 0, "dominant_program": None}

    total_mass      = 0.0
    specific_mass   = 0.0  # β-mass in w_P >= threshold programs
    labeled_mass    = 0.0  # β-mass in disease-labeled programs
    best_prog       = None
    best_abs_b      = 0.0
    n_disease_progs = 0
    n_labeled_progs = 0

    for prog, beta_info in beta_estimates.items():
        if beta_info is None:
            continue
        b = beta_info.get("beta") if isinstance(beta_info, dict) else beta_info
        if b is None:
            continue
        try:
            b = float(b)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(b):
            continue

        abs_b = abs(b)
        total_mass += abs_b

        w = program_disease_weights.get(prog, 0.0)
        if w >= min_weight_threshold:
            specific_mass += abs_b
            n_disease_progs += 1

        if program_labels:
            label_info = program_labels.get(prog, {})
            if label_info.get("disease_relevant") or label_info.get("relevance_score", 0) > 0.5:
                labeled_mass += abs_b
                n_labeled_progs += 1

        if abs_b > best_abs_b:
            best_abs_b = abs_b
            best_prog  = prog

    if total_mass < 1e-10:
        return {"pathway_coherence_score": 0.0, "n_disease_programs": 0,
                "n_labeled_programs": 0, "dominant_program": None}

    # Combine specificity and label signals; weight labels lower (optional data)
    specificity_score = specific_mass / total_mass
    if total_mass > 0 and (specific_mass > 0 or labeled_mass > 0):
        label_score = labeled_mass / total_mass if program_labels else specificity_score
        coherence = 0.7 * specificity_score + 0.3 * label_score
    else:
        coherence = specificity_score

    return {
        "pathway_coherence_score": round(coherence, 4),
        "n_disease_programs":      n_disease_progs,
        "n_labeled_programs":      n_labeled_progs,
        "dominant_program":        best_prog,
    }


def compute_beta_program_concentration(
    beta_estimates: dict[str, Any],
    program_weights: dict[str, float],
) -> float:
    """
    Fraction of a gene's total β-mass that falls in AMD-specific programs.

    beta_program_concentration ∈ [0, 1]:
      - 0.0: all β-mass in generic (non-AMD) programs
      - 1.0: all β-mass in AMD-upregulated programs

    This is a pleiotropy diagnostic: a high-pleiotropy gene like JAZF1 that
    perturbs many programs uniformly will have low concentration even if a few
    of those programs happen to have AMD γ. A truly AMD-specific regulator
    will have high concentration.

    Args:
        beta_estimates:   {program_id → {beta: float, ...} | float}
        program_weights:  {program_id → w_P}  from compute_program_disease_weights

    Returns:
        Concentration ∈ [0, 1], or 0.0 if inputs are empty / all-zero.
    """
    if not beta_estimates or not program_weights:
        return 0.0

    total_mass  = 0.0
    amd_mass    = 0.0

    for prog, beta_info in beta_estimates.items():
        if beta_info is None:
            continue
        if isinstance(beta_info, dict):
            b = beta_info.get("beta")
        else:
            b = beta_info
        if b is None:
            continue
        try:
            b = float(b)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(b):
            continue

        abs_b = abs(b)
        total_mass += abs_b
        amd_mass   += abs_b * program_weights.get(prog, 1.0)

    if total_mass < 1e-10:
        return 0.0
    return round(amd_mass / total_mass, 4)
