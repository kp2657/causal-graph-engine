"""
pipelines/state_space/therapeutic_redirection.py

Compute therapeutic_redirection — the primary component of the new OTA score.

Formula (locked):
  therapeutic_redirection = Σ_c w(c) × Σ_P P_loading(gene,P,c)
                            × Σ_T [T_perturbed(c) - T_baseline(c)]
                            × gamma_conditional(P,d|c)

  Σ_T sums over all (i→j) state transitions originating from pathological basins.
  [T_perturbed - T_baseline] is the signed change in flow toward healthy attractors.

T_perturbed construction (locked):
  For pathological source states, the gene perturbation effect is:
    beta × P_loading drives T[i→pathological] down / T[i→healthy] up.
  Each edge change is capped at 50% of T_baseline[i,j].
  Rows are renormalised after applying all deltas.

Public API:
    perturb_transition_matrix(T_baseline, gene_beta, p_loading, path_idxs, healthy_idxs)
    compute_net_trajectory_improvement(T_baseline, T_perturbed, path_idxs, healthy_idxs)
    compute_therapeutic_redirection_per_celltype(gene, disease, cell_type, ...)
    compute_therapeutic_redirection(gene, disease, per_celltype_results)
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from models.latent_mediator import (
    ProgramLoading,
    ConditionalBeta,
    ConditionalGamma,
    TherapeuticRedirectionResult,
)
from pipelines.state_space.conditional_beta import compute_pooled_fraction

_T_PERTURB_CAP = 0.50          # max fraction of baseline that can be redirected per edge
_POOLED_WARNING_THRESHOLD = 0.50   # >50% pooled betas → context_confidence_warning
_MIN_P_LOADING = 1e-4          # ignore trivially small program loadings
_MIN_STATE_INFLUENCE = 0.01    # minimum disease_axis_score to enter state-direct TR path


# ---------------------------------------------------------------------------
# T_perturbed construction
# ---------------------------------------------------------------------------

def perturb_transition_matrix(
    T_baseline: np.ndarray,
    gene_beta: float,
    p_loading: float,
    path_idxs: list[int],
    healthy_idxs: list[int],
) -> np.ndarray:
    """
    Apply a gene perturbation effect to the baseline transition matrix.

    For each pathological source state i:
      delta[i, j in healthy]      = +min(|beta|×p_loading×T[i,j], 0.5×T[i,j])
      delta[i, j in pathological] = -min(|beta|×p_loading×T[i,j], 0.5×T[i,j])

    Sign convention: positive beta means gene drives pathological programme;
    a loss-of-function perturbation (KO) with positive beta → improves transitions.

    Returns T_perturbed (same shape as T_baseline), row-renormalised.

    Args:
        T_baseline:  (n_states × n_states) row-normalised float array.
        gene_beta:   ConditionalBeta.beta for gene→program.  NaN → no change.
        p_loading:   ProgramLoading.p_loading for gene→program.
        path_idxs:   Row indices of pathological states.
        healthy_idxs: Column indices of healthy states.
    """
    if not math.isfinite(gene_beta) or p_loading < _MIN_P_LOADING:
        return T_baseline.copy()

    T = T_baseline.copy()
    n = T.shape[0]
    effect = abs(gene_beta) * p_loading

    healthy_set = set(healthy_idxs)
    path_set = set(path_idxs)

    for i in path_idxs:
        if i >= n:
            continue
        deltas = np.zeros(n, dtype=float)

        for j in range(n):
            if i == j:
                continue
            baseline_edge = T_baseline[i, j]
            if baseline_edge <= 0:
                continue
            max_delta = _T_PERTURB_CAP * baseline_edge
            delta_mag = min(effect * baseline_edge, max_delta)

            if j in healthy_set:
                deltas[j] = +delta_mag   # increase outflow to healthy
            elif j in path_set and j != i:
                deltas[j] = -delta_mag   # decrease self-reinforcing path flow

        T[i] = np.maximum(T_baseline[i] + deltas, 0.0)

    # Reduce inflow: healthy→pathological edges (prevents cells entering disease state).
    # This fires when T[path, path]=0 and all outflow is already healthy (e.g. AMD),
    # making the outflow perturbation above a renormalization no-op.
    for i in healthy_idxs:
        if i >= n:
            continue
        deltas = np.zeros(n, dtype=float)
        for j in path_idxs:
            if j >= n:
                continue
            baseline_edge = T_baseline[i, j]
            if baseline_edge <= 0:
                continue
            max_delta = _T_PERTURB_CAP * baseline_edge
            delta_mag = min(effect * baseline_edge, max_delta)
            deltas[j] = -delta_mag  # reduce transition into pathological state
        T[i] = np.maximum(T[i] + deltas, 0.0)

    # Row-renormalise
    row_sums = T.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums <= 0, 1.0, row_sums)
    T = T / row_sums
    return T


def compute_net_trajectory_improvement(
    T_baseline: np.ndarray,
    T_perturbed: np.ndarray,
    path_idxs: list[int],
    healthy_idxs: list[int],
) -> float:
    """
    Net improvement in transition flow between pathological and healthy states.

    Two complementary signals are summed:
      1. Outflow increase: Σ_{i∈path, j∈healthy} [T_pert[i,j] - T_base[i,j]]
         (more outflow from disease state toward health)
      2. Inflow reduction: Σ_{i∈healthy, j∈path} [T_base[i,j] - T_pert[i,j]]
         (fewer cells entering the disease state from healthy states)

    Signal 2 fires when T[path, path]=0 and all outflow is already to healthy states
    (e.g. AMD Müller cells), where signal 1 is a renormalization no-op.

    Returns a non-negative float. Clipped to [0, +∞].
    """
    if not path_idxs or not healthy_idxs:
        return 0.0
    n = T_baseline.shape[0]
    path_i   = [i for i in path_idxs  if i < n]
    healthy_j = [j for j in healthy_idxs if j < n]
    if not path_i or not healthy_j:
        return 0.0

    # Signal 1: outflow improvement from pathological states
    delta_out = T_perturbed[np.ix_(path_i, healthy_j)] - T_baseline[np.ix_(path_i, healthy_j)]
    outflow_improvement = float(max(delta_out.sum(), 0.0))

    # Signal 2: inflow reduction into pathological states from healthy states
    delta_in = T_baseline[np.ix_(healthy_j, path_i)] - T_perturbed[np.ix_(healthy_j, path_i)]
    inflow_reduction = float(max(delta_in.sum(), 0.0))

    return outflow_improvement + inflow_reduction


# ---------------------------------------------------------------------------
# Genetic-track redirection — GWAS genes with no Perturb-seq coverage
# ---------------------------------------------------------------------------

def compute_genetic_tr(
    gamma_ota: float,
    T_baseline: "np.ndarray | None",
    path_idxs: list[int],
    healthy_idxs: list[int],
) -> float:
    """
    Genetic-evidence TR for GWAS genes with no Perturb-seq knockout data.

    Uses a full-KO proxy (gene_beta=1.0, p_loading=1.0) so the transition
    matrix perturbation is derived purely from the state-space geometry, not
    from transcriptomic perturbation data.  Grounded solely in |γ_ota|.

    Formula: net_trajectory_improvement × |gamma_ota|

    Deliberately does NOT use disease_axis_score — that quantity is estimated
    from Perturb-seq knockouts and is undefined for GWAS-only genes.  Mixing
    it with γ would double-count genetics (DAS proxy = f(γ) × γ).
    """
    if T_baseline is None or not path_idxs or not healthy_idxs:
        return 0.0
    if not math.isfinite(gamma_ota) or abs(gamma_ota) < 1e-8:
        return 0.0

    T_perturbed = perturb_transition_matrix(
        T_baseline=T_baseline,
        gene_beta=1.0,
        p_loading=1.0,
        path_idxs=path_idxs,
        healthy_idxs=healthy_idxs,
    )
    net_improvement = compute_net_trajectory_improvement(T_baseline, T_perturbed, path_idxs, healthy_idxs)
    return net_improvement * abs(gamma_ota)


# ---------------------------------------------------------------------------
# State-direct redirection (Phase F — no NMF membership required)
# ---------------------------------------------------------------------------

def compute_state_direct_redirection(
    gene_beta: float,
    disease_axis_score: float,
    directionality: int,
    gamma_ota: float,
    T_baseline: "np.ndarray | None",
    path_idxs: list[int],
    healthy_idxs: list[int],
) -> float:
    """
    Directional redirection score derived from a gene's continuous disease-state
    influence, without requiring NMF program membership.

    This enables TR to fire for known GWAS anchor genes (NOD2, IL10, TNF, IL23R)
    that are causal but do not appear in NMF top-N gene sets.

    Mechanics:
      1. Use ``disease_axis_score`` as the influence weight (replaces p_loading).
      2. Perturb the transition matrix via the existing ``perturb_transition_matrix``
         helper — same cap (50%) and row-renormalisation.
      3. Ground in genetics: multiply by ``|gamma_ota|`` so disease relevance is
         anchored in genetic evidence, not just expression patterns.

    Args:
        gene_beta:           Perturbational β from beta_matrix (may be pooled/virtual).
                             NaN or 0 → uses 1.0 as full-KO proxy.
        disease_axis_score:  Continuous [0,1] influence from compute_gene_state_influence.
        directionality:      +1 = gene higher in pathological states (KO helps);
                             -1 = gene higher in healthy states (activation helps).
                              0 = no signal.
        gamma_ota:           OTA γ_{gene→disease} for genetic grounding.
        T_baseline:          (n_states × n_states) baseline transition matrix.
        path_idxs:           Row indices of pathological states.
        healthy_idxs:        Column indices of healthy states.

    Returns:
        Directional redirection score ≥ 0  (positive = toward healthier states).
        Returns 0.0 if inputs are insufficient.
    """
    if T_baseline is None or not path_idxs or not healthy_idxs:
        return 0.0
    if disease_axis_score < _MIN_STATE_INFLUENCE:
        return 0.0
    if not math.isfinite(gamma_ota):
        gamma_ota = 0.0

    # Use 1.0 as proxy when no real beta is available (full-KO, magnitude bounded
    # by disease_axis_score and gamma_ota below).
    effective_beta = gene_beta if math.isfinite(gene_beta) and abs(gene_beta) > 1e-8 else 1.0

    T_perturbed = perturb_transition_matrix(
        T_baseline   = T_baseline,
        gene_beta    = effective_beta,
        p_loading    = disease_axis_score,
        path_idxs    = path_idxs,
        healthy_idxs = healthy_idxs,
    )

    net_improvement = compute_net_trajectory_improvement(
        T_baseline, T_perturbed, path_idxs, healthy_idxs
    )

    # Genetic grounding: γ_ota anchors disease relevance
    return net_improvement * disease_axis_score * abs(gamma_ota)


def compute_stability_score(
    T_baseline: np.ndarray,
    gene_beta: float,
    p_loading: float,
    path_idxs: list[int],
    healthy_idxs: list[int],
    n_iterations: int = 50,
    noise_level: float = 0.05,
) -> float:
    """
    Monte Carlo sensitivity analysis for transition-based redirection.

    Adds Gaussian noise (sigma=noise_level) to non-zero baseline edges,
    re-normalises, and re-computes TR.

    Stability = 1 - CV (Coefficient of Variation) of TR across iterations.
    Saturated at [0, 1].
    """
    if not math.isfinite(gene_beta) or p_loading < _MIN_P_LOADING:
        return 1.0

    tr_samples = []
    T_base = T_baseline.copy()
    
    for _ in range(n_iterations):
        # Add noise to non-zero edges
        noise = np.random.normal(0, noise_level, T_base.shape)
        T_noisy = np.maximum(T_base + noise * (T_base > 0), 0.0)
        
        # Row-renormalise
        row_sums = T_noisy.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums <= 0, 1.0, row_sums)
        T_noisy = T_noisy / row_sums
        
        T_perturbed = perturb_transition_matrix(
            T_noisy, gene_beta, p_loading, path_idxs, healthy_idxs
        )
        improvement = compute_net_trajectory_improvement(
            T_noisy, T_perturbed, path_idxs, healthy_idxs
        )
        tr_samples.append(improvement)
    
    mean_tr = np.mean(tr_samples)
    if mean_tr <= 1e-8:
        return 0.0
    
    std_tr = np.std(tr_samples)
    cv = std_tr / mean_tr
    
    # Stability: 1.0 is perfect, 0.0 is high variance (CV >= 1.0)
    return float(np.clip(1.0 - cv, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Per-cell-type contribution
# ---------------------------------------------------------------------------

def compute_therapeutic_redirection_per_celltype(
    gene: str,
    disease: str,
    cell_type: str,
    program_loadings: list[ProgramLoading],
    conditional_betas: list[ConditionalBeta],
    conditional_gammas: dict[str, ConditionalGamma],
    transition_result: dict[str, Any],
    # Phase F: continuous state-influence path (fires even without NMF membership)
    state_influence: dict | None = None,
    gene_gamma_ota: float | None = None,
) -> dict:
    """
    Compute therapeutic_redirection contribution for one cell type.

    Args:
        gene:               Gene symbol.
        disease:            Short disease key.
        cell_type:          Cell type label.
        program_loadings:   ProgramLoading list for this (gene, cell_type).
        conditional_betas:  ConditionalBeta list for this (gene, cell_type).
        conditional_gammas: {program_id: ConditionalGamma} for this cell type.
        transition_result:  infer_state_transition_graph output.

    Returns dict:
        redirection          float — contribution to therapeutic_redirection
        stability            float — Monte Carlo stability score [0,1]
        n_programs           int
        pooled_fraction      float
        evidence_tiers       list[str]
        provenance           list[str]
    """
    T_baseline: np.ndarray | None = transition_result.get("transition_matrix")
    state_labels: list = transition_result.get("state_labels", [])
    path_ids: list[str] = transition_result.get("pathologic_basin_ids", [])
    healthy_ids: list[str] = transition_result.get("healthy_basin_ids", [])

    if T_baseline is None or len(state_labels) == 0 or not path_ids or not healthy_ids:
        return {
            "redirection": 0.0, "stability": 1.0, "n_programs": 0,
            "pooled_fraction": 0.0, "evidence_tiers": [], "provenance": [],
            "note": "No transition matrix or empty basins",
        }

    label_to_idx = {str(s): i for i, s in enumerate(state_labels)}
    path_idxs = [label_to_idx[s] for s in path_ids if s in label_to_idx]
    healthy_idxs = [label_to_idx[s] for s in healthy_ids if s in label_to_idx]

    # Index betas by program_id
    beta_by_prog: dict[str, ConditionalBeta] = {cb.program_id: cb for cb in conditional_betas}

    redirection_total = 0.0
    stability_samples = []
    n_programs = 0
    tiers: list[str] = []
    prov: list[str] = []

    for pl in program_loadings:
        if pl.gene != gene or pl.cell_type != cell_type:
            continue
        if pl.p_loading < _MIN_P_LOADING:
            continue

        cb = beta_by_prog.get(pl.program_id)
        if cb is None or not math.isfinite(cb.beta):
            continue

        cg = conditional_gammas.get(pl.program_id)
        if cg is None or not math.isfinite(cg.gamma_mixed):
            continue

        # Build T_perturbed for this (gene, program) pair
        T_perturbed = perturb_transition_matrix(
            T_baseline=T_baseline,
            gene_beta=cb.beta,
            p_loading=pl.p_loading,
            path_idxs=path_idxs,
            healthy_idxs=healthy_idxs,
        )

        net_improvement = compute_net_trajectory_improvement(
            T_baseline, T_perturbed, path_idxs, healthy_idxs
        )

        # Phase Z7: Stability for this program contribution
        stab = compute_stability_score(
            T_baseline, cb.beta, pl.p_loading, path_idxs, healthy_idxs, n_iterations=20
        )
        stability_samples.append(stab)

        # Contribution: P_loading × Σ_T[T_perturbed - T_baseline] × gamma_conditional
        contribution = pl.p_loading * net_improvement * cg.gamma_mixed
        redirection_total += contribution
        n_programs += 1
        tiers.append(cb.evidence_tier)
        prov.append(f"{pl.program_id}(β={cb.beta:.3f},γ={cg.gamma_mixed:.3f},Δ={net_improvement:.4f},stab={stab:.2f})")

    pooled_frac = compute_pooled_fraction(
        [cb for cb in conditional_betas if math.isfinite(cb.beta)]
    )

    # --- Phase G: state-direct path + transition decomposition ---------------
    state_direct = 0.0
    state_direct_stability = 1.0
    state_influence_score_out = 0.0
    directionality_out = 0
    # Phase G transition scores (from TransitionGeneProfile passed as state_influence dict)
    entry_score_out = 0.0
    persistence_score_out = 0.0
    recovery_score_out = 0.0
    boundary_score_out = 0.0
    mechanistic_category_out = "unknown"
    if state_influence is not None and gene_gamma_ota is not None:
        si = state_influence if isinstance(state_influence, dict) and "disease_axis_score" in state_influence \
            else (state_influence.get(gene, {}) if isinstance(state_influence, dict) else {})
        das = float(si.get("disease_axis_score", 0.0))
        directionality_out = int(si.get("directionality", 0))
        state_influence_score_out = das
        # Extract Phase G transition scores
        entry_score_out       = float(si.get("entry_score", 0.0))
        persistence_score_out = float(si.get("persistence_score", 0.0))
        recovery_score_out    = float(si.get("recovery_score", 0.0))
        boundary_score_out    = float(si.get("boundary_score", 0.0))
        mechanistic_category_out = str(si.get("mechanistic_category", "unknown"))
        if das >= _MIN_STATE_INFLUENCE and T_baseline is not None:
            # Use a virtual beta proxy; effect is bounded by das and gamma_ota
            beta_for_state = float("nan")  # triggers 1.0 proxy inside compute_state_direct_redirection
            state_direct = compute_state_direct_redirection(
                gene_beta=beta_for_state,
                disease_axis_score=das,
                directionality=directionality_out,
                gamma_ota=gene_gamma_ota,
                T_baseline=T_baseline,
                path_idxs=path_idxs,
                healthy_idxs=healthy_idxs,
            )
            # Stability for state-direct path
            state_direct_stability = compute_stability_score(
                T_baseline, 1.0, das, path_idxs, healthy_idxs, n_iterations=20
            )
            if state_direct > 0:
                prov.append(
                    f"state_direct(das={das:.3f},dir={directionality_out:+d},"
                    f"γ={gene_gamma_ota:.3f},Δ={state_direct:.4f},stab={state_direct_stability:.2f})"
                )

    # Combined stability: mean of all active paths
    if state_direct > 0:
        stability_samples.append(state_direct_stability)
    
    mean_stability = float(np.mean(stability_samples)) if stability_samples else 1.0

    return {
        "redirection":           redirection_total + state_direct,
        "stability":             mean_stability,
        "nmf_redirection":       redirection_total,
        "state_direct":          state_direct,
        "state_influence_score": state_influence_score_out,
        "directionality":        directionality_out,
        # Phase G
        "entry_score":           entry_score_out,
        "persistence_score":     persistence_score_out,
        "recovery_score":        recovery_score_out,
        "boundary_score":        boundary_score_out,
        "mechanistic_category":  mechanistic_category_out,
        "n_programs":            n_programs,
        "pooled_fraction":       pooled_frac,
        "evidence_tiers":        tiers,
        "provenance":            prov,
    }


# ---------------------------------------------------------------------------
# Multi-cell-type summation
# ---------------------------------------------------------------------------

def compute_therapeutic_redirection(
    gene: str,
    disease: str,
    per_celltype_data: dict[str, dict],
    cell_type_weights: dict[str, float] | None = None,
    genetic_grounding: float = 0.0,
) -> TherapeuticRedirectionResult:
    """
    Sum therapeutic redirection across all cell types.

    therapeutic_redirection = Σ_c w(c) × celltype_contribution(c)

    Cell-type weights default to equal (1/n_cell_types).

    Args:
        gene:               Gene symbol.
        disease:            Short disease key.
        per_celltype_data:  {cell_type: result from compute_therapeutic_redirection_per_celltype}.
        cell_type_weights:  Optional {cell_type: weight}.  Normalised internally.
        genetic_grounding:  OTA γ_{gene→disease} for final_score genetic component.

    Returns:
        TherapeuticRedirectionResult with therapeutic_redirection filled.
        Other score components (durability, escape_risk, etc.) are left at 0.0
        for Phase C — they will be filled by Phase D / failure_memory integration.
    """
    cell_types = [ct for ct, r in per_celltype_data.items() if r.get("redirection", 0) > 0]
    if not cell_types:
        cell_types = list(per_celltype_data.keys())

    # Build normalised weights
    if cell_type_weights:
        raw_w = {ct: cell_type_weights.get(ct, 1.0) for ct in per_celltype_data}
    else:
        raw_w = {ct: 1.0 for ct in per_celltype_data}
    total_w = sum(raw_w.values()) or 1.0
    norm_w = {ct: w / total_w for ct, w in raw_w.items()}

    redirection_total = 0.0
    stability_sum = 0.0
    all_tiers: list[str] = []
    all_prov: list[str] = []
    n_programs_total = 0
    n_ct_contributing = 0
    pooled_fracs: list[float] = []
    # Phase F: aggregate state-influence (DAS) — annotation only from Phase G
    si_weighted_sum = 0.0
    dir_votes: list[int] = []
    # Phase G: aggregate transition decomposition scores (weighted mean)
    entry_weighted_sum = 0.0
    persist_weighted_sum = 0.0
    recovery_weighted_sum = 0.0
    boundary_weighted_sum = 0.0
    cat_votes: list[str] = []

    for ct, result in per_celltype_data.items():
        w = norm_w.get(ct, 0.0)
        contrib = result.get("redirection", 0.0)
        if contrib > 0:
            n_ct_contributing += 1
        redirection_total += w * contrib
        stability_sum += w * result.get("stability", 1.0)
        n_programs_total += result.get("n_programs", 0)
        all_tiers.extend(result.get("evidence_tiers", []))
        all_prov.extend([f"[{ct}] {p}" for p in result.get("provenance", [])])
        pf = result.get("pooled_fraction", 0.0)
        if result.get("n_programs", 0) > 0:
            pooled_fracs.append(pf)
        si = result.get("state_influence_score", 0.0)
        si_weighted_sum += w * si
        d = result.get("directionality", 0)
        if d != 0:
            dir_votes.append(d)
        # Phase G
        entry_weighted_sum   += w * result.get("entry_score", 0.0)
        persist_weighted_sum += w * result.get("persistence_score", 0.0)
        recovery_weighted_sum += w * result.get("recovery_score", 0.0)
        boundary_weighted_sum += w * result.get("boundary_score", 0.0)
        cat = result.get("mechanistic_category", "unknown")
        if cat != "unknown":
            cat_votes.append(cat)

    mean_pooled = float(np.mean(pooled_fracs)) if pooled_fracs else 0.0
    context_warning = mean_pooled > _POOLED_WARNING_THRESHOLD

    # Consensus directionality: majority vote (ties → 0)
    if dir_votes:
        from collections import Counter
        c = Counter(dir_votes)
        consensus_dir = c.most_common(1)[0][0]
    else:
        consensus_dir = 0

    # Consensus mechanistic category: plurality vote
    if cat_votes:
        from collections import Counter
        consensus_cat = Counter(cat_votes).most_common(1)[0][0]
    else:
        consensus_cat = "unknown"

    return TherapeuticRedirectionResult(
        gene=gene,
        disease=disease,
        therapeutic_redirection=redirection_total,
        stability_score=stability_sum,
        n_programs_contributing=n_programs_total,
        n_cell_types_contributing=n_ct_contributing,
        pooled_fraction=mean_pooled,
        context_confidence_warning=context_warning,
        evidence_tiers_used=sorted(set(all_tiers)),
        provenance=all_prov,
        state_influence_score=si_weighted_sum,
        directionality=consensus_dir,
        genetic_grounding=genetic_grounding,
        # Phase G
        entry_score=entry_weighted_sum,
        persistence_score=persist_weighted_sum,
        recovery_score=recovery_weighted_sum,
        boundary_score=boundary_weighted_sum,
        mechanistic_category=consensus_cat,
    )
