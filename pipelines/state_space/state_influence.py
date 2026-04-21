"""
pipelines/state_space/state_influence.py

Gene → disease-state influence scoring.

Phase G: this module is now a thin wrapper around transition_scoring.py.
The primary output is a TransitionGeneProfile per gene.

For backward compatibility, the public API also returns a flat dict per gene
with the legacy keys (disease_axis_score, directionality) so callers that
have not been updated to consume TransitionGeneProfile can still function.
Those legacy keys are annotation-only — they are NOT used in any composite
formula.

Public API
----------
    compute_gene_state_influence(adata, transition_result, gene_list)
        -> dict[str, dict]   (legacy flat dict — backward compat)

    compute_gene_transition_profiles(adata, transition_result, gene_list)
        -> dict[str, TransitionGeneProfile]   (Phase G primary output)

    nominate_state_genes(adata, transition_result, top_k, exclude)
        -> list[tuple[str, dict]]   (Phase K: state-space gene nomination)
"""
from __future__ import annotations

from typing import Any

from models.evidence import TransitionGeneProfile
from pipelines.state_space.transition_scoring import compute_transition_gene_scores


def compute_gene_transition_profiles(
    adata: Any,
    transition_result: dict,
    gene_list: list[str] | None = None,
) -> dict[str, TransitionGeneProfile]:
    """
    Compute full transition-landscape profiles for each gene (Phase G).

    Args:
        adata:             Preprocessed AnnData with state obs columns and pseudotime.
        transition_result: Output of infer_state_transition_graph().
        gene_list:         Genes to score.  None → all genes in adata.

    Returns:
        {gene: TransitionGeneProfile}
    """
    return compute_transition_gene_scores(adata, transition_result, gene_list)


def compute_gene_state_influence(
    adata: Any,
    transition_result: dict,
    gene_list: list[str] | None = None,
) -> dict[str, dict]:
    """
    Backward-compatible flat dict output.

    Returns the same keys as the pre-Phase G implementation so callers that
    read disease_axis_score / directionality / pathological_mean / healthy_mean
    continue to work without modification.

    NOTE: disease_axis_score here is populated from TransitionGeneProfile.disease_axis_score
    (the legacy DAS computation is preserved inside transition_scoring._compute_das).
    It is annotation only and does not feed any composite formula.

    New callers should use compute_gene_transition_profiles() directly.
    """
    profiles = compute_transition_gene_scores(adata, transition_result, gene_list)

    result: dict[str, dict] = {}
    for gene, profile in profiles.items():
        result[gene] = {
            # Legacy keys (annotation only from Phase G onward)
            "disease_axis_score": profile.disease_axis_score,
            "directionality":     profile.entry_direction or profile.persistence_direction,

            # Phase G transition scores (new — callers can consume these)
            "entry_score":               profile.entry_score,
            "persistence_score":         profile.persistence_score,
            "recovery_score":            profile.recovery_score,
            "boundary_knn_score":        profile.boundary_knn_score,
            "boundary_pseudotime_score": profile.boundary_pseudotime_score,
            "boundary_score":            profile.boundary_score,
            "entry_direction":           profile.entry_direction,
            "persistence_direction":     profile.persistence_direction,
            "recovery_direction":        profile.recovery_direction,
            "boundary_direction":        profile.boundary_direction,
            "mechanistic_category":      profile.mechanistic_category,

            # Support metadata
            "pathological_mean": 0.0,  # no longer computed separately; use disease_axis_score
            "healthy_mean":      0.0,
        }

    return result


def nominate_state_genes(
    adata: Any,
    transition_result: dict,
    top_k: int = 25,
    exclude: set[str] | None = None,
) -> list[tuple[str, dict]]:
    """
    Score all HVGs in adata and return the top-k by combined transition score.

    Phase K: used to supplement GWAS-instrument genes with state-space candidates
    that are not genetic instruments but have strong transition-landscape evidence.

    Combined score = entry_score + persistence_score + recovery_score + 0.5 × boundary_score

    Args:
        adata:             AnnData with state obs columns and pseudotime.
        transition_result: Output of infer_state_transition_graph().
        top_k:             Maximum number of genes to return.
        exclude:           Gene names to exclude (e.g. already-ranked GWAS genes).

    Returns:
        List of (gene, profile_dict) sorted by combined score descending.
        profile_dict is TransitionGeneProfile.model_dump().
    """
    exclude = exclude or set()
    # Filter out mitochondrial-encoded genes (MT-) and ribo genes — not nuclear drug targets
    all_genes = [
        g for g in list(adata.var_names)
        if g not in exclude
        and not g.startswith("MT-")
        and not g.startswith("MTRNR")
        and not (g.startswith("RPS") or g.startswith("RPL"))  # ribosomal proteins
    ]
    if not all_genes:
        return []

    profiles = compute_transition_gene_scores(adata, transition_result, all_genes)

    scored: list[tuple[str, float, dict]] = []
    for gene, profile in profiles.items():
        combined = (
            profile.entry_score
            + profile.persistence_score
            + profile.recovery_score
            + profile.boundary_score * 0.5
        )
        scored.append((gene, combined, profile.model_dump()))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [(g, p) for g, _, p in scored[:top_k]]
