"""
pipelines/state_space/controller_classifier.py

Controller vs marker classification — Phase H.

Classifies each gene as upstream_controller | midstream_mediator | downstream_marker
based on a prioritised signal stack:

  1. Perturbation evidence tier  — T1/T2 perturbation confirms; virtual = silent
  2. TF/signaling annotation     — static curated lookup (data/annotations/tf_signaling_genes.json)
  3. Pseudotime peak position     — early peak → controller candidate; late peak → marker
  4. Phase G transition profile  — entry_score enrichment adds controller signal;
                                   high persistence + low entry → marker signal
  5. STRING network degree        — hub genes get a small bonus (graceful fallback)

Confidence levels
-----------------
  high    T1/T2 perturbation evidence present  → controller_likelihood up to 1.0
  medium  TF annotation + early pseudotime     → controller_likelihood capped at 0.65
  low     annotation only                      → controller_likelihood capped at 0.35
          (annotation nominates; perturbation confirms)

Category thresholds
-------------------
  upstream_controller  controller_likelihood > 0.50
  midstream_mediator   0.30 <= controller_likelihood <= 0.50
  downstream_marker    controller_likelihood < 0.30

Marker discount (consumed by target_prioritization_agent.py)
------------------------------------------------------------
  marker_confidence   = signal that gene is a terminal/maintenance marker
                        derived from transition profile (high persistence, low entry/recovery)
  marker_discount     = clip(0.25 × (1 − controller_likelihood) × marker_confidence, 0, 0.40)
  final               = core × t_mod × risk_discount × (1 − marker_discount)

Public API
----------
  classify_gene_controller(gene, disease, transition_profile, evidence_tier, adata, ...) -> ControllerAnnotation
  compute_marker_confidence(transition_profile) -> float
  compute_marker_discount(controller_annotation, transition_profile) -> float
  classify_gene_list(genes, disease, profiles, tiers, adata, ...) -> dict[str, ControllerAnnotation]
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from models.evidence import ControllerAnnotation, TransitionGeneProfile

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EARLY_PT_THRESHOLD    = 0.40   # fractional pseudotime; below = early = controller signal
_LATE_PT_THRESHOLD     = 0.70   # above = late = marker signal
_ENTRY_CONTROLLER_MIN  = 0.30   # entry_score above this adds controller signal
_PERSISTENCE_MARKER_MIN = 0.50  # persistence_score above this (with low entry) adds marker signal
_STRING_HUB_DEGREE     = 20     # minimum STRING degree to count as network hub

# Signal contribution weights (sum > 1 before capping — intentional)
_W_PERTURBATION_T1  = 0.45
_W_PERTURBATION_T2  = 0.35
_W_TF_ANNOTATION    = 0.25
_W_EARLY_PSEUDOTIME = 0.20
_W_ENTRY_SCORE      = 0.15
_W_NETWORK_HUB      = 0.10
_W_LATE_PT_PENALTY  = -0.15
_W_MARKER_PENALTY   = -0.10

# Confidence ceilings
_CAP_LOW    = 0.35
_CAP_MEDIUM = 0.65
_CAP_HIGH   = 1.00

# Category thresholds
_CONTROLLER_THRESHOLD = 0.50
_MEDIATOR_THRESHOLD   = 0.30


# ---------------------------------------------------------------------------
# TF/signaling lookup — loaded once, cached
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_tf_signaling_genes() -> frozenset[str]:
    """Load curated TF + signaling gene set from data/annotations/."""
    ann_path = Path(__file__).parent.parent.parent / "data" / "annotations" / "tf_signaling_genes.json"
    try:
        with open(ann_path) as f:
            data = json.load(f)
        genes: list[str] = (
            data.get("transcription_factors", []) +
            data.get("signaling_molecules", [])
        )
        return frozenset(g.upper() for g in genes)
    except Exception:
        return frozenset()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pseudotime_peak(
    gene: str,
    adata: Any,
    gene_idx: dict[str, int],
) -> float | None:
    """
    Fractional pseudotime position of peak mean expression for gene.
    Returns value in [0, 1]; lower = earlier = more likely controller.
    Returns None if pseudotime or gene not available.
    """
    if adata is None:
        return None
    pt_series = adata.obs.get("dpt_pseudotime")
    if pt_series is None:
        return None
    idx = gene_idx.get(gene)
    if idx is None:
        return None

    try:
        pt_arr = np.asarray(pt_series, dtype=float)
        X = adata.X
        if hasattr(X, "toarray"):
            expr = np.asarray(X[:, idx]).ravel()
        else:
            expr = np.asarray(X[:, idx], dtype=float).ravel()

        n_bins = 10
        bin_edges  = np.percentile(pt_arr, np.linspace(0, 100, n_bins + 1))
        bin_assign = np.digitize(pt_arr, bin_edges[1:-1])  # 0-indexed 0..n_bins-1
        bin_means  = np.array([
            expr[bin_assign == b].mean() if (bin_assign == b).sum() > 0 else 0.0
            for b in range(n_bins)
        ])
        peak_bin = int(np.argmax(bin_means))
        # Map bin → fractional pseudotime (midpoint of bin)
        return (peak_bin + 0.5) / n_bins
    except Exception:
        return None


def _build_gene_index(adata: Any) -> dict[str, int]:
    """Gene symbol → column index in adata."""
    if adata is None:
        return {}
    idx: dict[str, int] = {g: i for i, g in enumerate(adata.var_names)}
    if "feature_name" in adata.var.columns:
        for i, sym in enumerate(adata.var["feature_name"]):
            if sym and str(sym) not in idx:
                idx[str(sym)] = i
    return idx


def _string_degree(gene: str) -> int | None:
    """
    Query STRING for network degree.  Graceful fallback — returns None on any failure.
    Only called once per gene; caller caches result if needed.
    """
    try:
        from mcp_servers.string_server import get_string_interactions  # type: ignore
        result = get_string_interactions(gene, min_score=400)
        return len(result.get("interactions", []))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Marker confidence
# ---------------------------------------------------------------------------

def compute_marker_confidence(
    transition_profile: TransitionGeneProfile | None,
) -> float:
    """
    Probability [0,1] that a gene is a terminal/maintenance marker rather than a controller.

    High when: high persistence_score, low entry_score, low recovery_score.
    Returns 0.5 (neutral prior) when no profile available.
    """
    if transition_profile is None:
        return 0.5
    mc = (
        0.50 * transition_profile.persistence_score +
        0.25 * (1.0 - transition_profile.entry_score) +
        0.25 * (1.0 - transition_profile.recovery_score)
    )
    return float(np.clip(mc, 0.0, 1.0))


def compute_marker_discount(
    controller_annotation: ControllerAnnotation,
    transition_profile: TransitionGeneProfile | None = None,
) -> float:
    """
    marker_discount = clip(0.25 × (1 − controller_likelihood) × marker_confidence, 0, 0.40)

    Applied in target_prioritization_agent.py:
        final = core × t_mod × risk_discount × (1 − marker_discount)
    """
    cl = controller_annotation.controller_likelihood
    mc = compute_marker_confidence(transition_profile)
    return float(np.clip(0.25 * (1.0 - cl) * mc, 0.0, 0.40))


# ---------------------------------------------------------------------------
# Core classification
# ---------------------------------------------------------------------------

def classify_gene_controller(
    gene: str,
    disease: str,
    transition_profile: TransitionGeneProfile | None = None,
    evidence_tier: str = "provisional_virtual",
    adata: Any = None,
    gene_idx: dict[str, int] | None = None,
    use_string: bool = False,
) -> ControllerAnnotation:
    """
    Classify a single gene as controller / mediator / marker.

    Args:
        gene:               Gene symbol.
        disease:            Short disease key (e.g. "IBD").
        transition_profile: Phase G TransitionGeneProfile (optional but recommended).
        evidence_tier:      Evidence tier from Tier 3 OTA γ computation.
        adata:              AnnData for pseudotime peak computation (optional).
        gene_idx:           Pre-built {gene: col_idx} for adata (optional; built if needed).
        use_string:         If True, query STRING for network degree (slow; off by default).

    Returns:
        ControllerAnnotation
    """
    tf_genes = _load_tf_signaling_genes()
    gene_upper = gene.upper()
    signals: list[str] = []
    raw_score = 0.0
    confidence = "low"

    # ----- Signal 1: Perturbation evidence tier --------------------------------
    if evidence_tier in ("Tier1_Interventional",):
        raw_score += _W_PERTURBATION_T1
        confidence = "high"
        signals.append("perturbation_t1")
    elif evidence_tier in ("Tier2_Convergent", "moderate_transferred", "moderate_grn"):
        raw_score += _W_PERTURBATION_T2
        confidence = "high"
        signals.append("perturbation_t2")
    # provisional_virtual / Tier3: no perturbation signal, confidence stays low

    # ----- Signal 2: TF/signaling annotation -----------------------------------
    is_tf = gene_upper in tf_genes
    if is_tf:
        raw_score += _W_TF_ANNOTATION
        signals.append("tf_annotation")

    # ----- Signal 3: Pseudotime peak position ----------------------------------
    if adata is not None:
        if gene_idx is None:
            gene_idx = _build_gene_index(adata)
        pt_peak = _pseudotime_peak(gene, adata, gene_idx)
    else:
        pt_peak = None

    is_early_pt = pt_peak is not None and pt_peak < _EARLY_PT_THRESHOLD
    is_late_pt  = pt_peak is not None and pt_peak > _LATE_PT_THRESHOLD

    if is_early_pt:
        raw_score += _W_EARLY_PSEUDOTIME
        signals.append(f"early_pseudotime(peak={pt_peak:.2f})")
    elif is_late_pt:
        raw_score += _W_LATE_PT_PENALTY  # negative
        signals.append(f"late_pseudotime(peak={pt_peak:.2f})")

    # ----- Signal 4: Phase G transition profile --------------------------------
    if transition_profile is not None:
        if transition_profile.entry_score > _ENTRY_CONTROLLER_MIN:
            raw_score += _W_ENTRY_SCORE
            signals.append(f"high_entry_score({transition_profile.entry_score:.2f})")
        if (transition_profile.persistence_score > _PERSISTENCE_MARKER_MIN
                and transition_profile.entry_score < 0.20):
            raw_score += _W_MARKER_PENALTY  # negative
            signals.append(
                f"maintenance_marker(persist={transition_profile.persistence_score:.2f},"
                f"entry={transition_profile.entry_score:.2f})"
            )

    # ----- Signal 5: STRING network hub ----------------------------------------
    if use_string:
        degree = _string_degree(gene)
        if degree is not None and degree >= _STRING_HUB_DEGREE:
            raw_score += _W_NETWORK_HUB
            signals.append(f"network_hub(degree={degree})")

    # ----- Apply confidence ceiling --------------------------------------------
    raw_score = max(0.0, raw_score)

    # Upgrade confidence to medium if TF + early pseudotime (no perturbation)
    if confidence == "low" and is_tf and is_early_pt:
        confidence = "medium"

    if confidence == "high":
        controller_likelihood = min(raw_score, _CAP_HIGH)
    elif confidence == "medium":
        controller_likelihood = min(raw_score, _CAP_MEDIUM)
    else:
        controller_likelihood = min(raw_score, _CAP_LOW)

    # ----- Category assignment --------------------------------------------------
    if controller_likelihood > _CONTROLLER_THRESHOLD:
        category = "upstream_controller"
    elif controller_likelihood >= _MEDIATOR_THRESHOLD:
        category = "midstream_mediator"
    else:
        category = "downstream_marker"

    return ControllerAnnotation(
        gene=gene,
        disease=disease,
        controller_likelihood=round(controller_likelihood, 4),
        controller_confidence=confidence,
        category=category,
        supporting_signals=signals,
    )


# ---------------------------------------------------------------------------
# Batch API
# ---------------------------------------------------------------------------

def classify_gene_list(
    genes: list[str],
    disease: str,
    transition_profiles: dict[str, TransitionGeneProfile] | None = None,
    evidence_tiers: dict[str, str] | None = None,
    adata: Any = None,
    use_string: bool = False,
) -> dict[str, ControllerAnnotation]:
    """
    Classify a list of genes, sharing the gene_idx computation across all genes.

    Args:
        genes:               List of gene symbols to classify.
        disease:             Short disease key.
        transition_profiles: {gene: TransitionGeneProfile} from Phase G (optional).
        evidence_tiers:      {gene: evidence_tier} from Tier 3 (optional).
        adata:               AnnData for pseudotime peak computation (optional).
        use_string:          If True, query STRING per gene (slow; off by default).

    Returns:
        {gene: ControllerAnnotation}
    """
    gene_idx = _build_gene_index(adata) if adata is not None else {}
    tiers = evidence_tiers or {}
    profiles = transition_profiles or {}

    result: dict[str, ControllerAnnotation] = {}
    for gene in genes:
        result[gene] = classify_gene_controller(
            gene=gene,
            disease=disease,
            transition_profile=profiles.get(gene),
            evidence_tier=tiers.get(gene, "provisional_virtual"),
            adata=adata,
            gene_idx=gene_idx,
            use_string=use_string,
        )
    return result
