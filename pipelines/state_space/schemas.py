"""
State-space pipeline type contracts and shared constants.

Re-exports core pydantic models from models.evidence.
Defines resolution parameters, provenance markers, and scoring thresholds
used consistently across latent_model, state_definition, and transition_graph.
"""
from __future__ import annotations

from models.evidence import (  # noqa: F401  (re-exported for callers)
    CellState,
    FailureRecord,
    PerturbationTransitionEffect,
    StateEvidenceTier,
    StateTransition,
    TrajectoryRedirectionScore,
)

# ---------------------------------------------------------------------------
# Leiden / KMeans resolution parameters for multi-resolution state discovery
# ---------------------------------------------------------------------------

DEFAULT_RESOLUTIONS: dict[str, float] = {
    "coarse":       0.20,   # broad disease basins (~3–6 states typical)
    "intermediate": 0.50,   # transition neighborhoods (~8–15 states)
    "fine":         1.00,   # local sub-states (~20–40 states)
}

# ---------------------------------------------------------------------------
# Provenance markers — attached to StateTransition.direction_evidence
# and LatentResult.provenance to make inference method explicit and queryable
# ---------------------------------------------------------------------------

PROV_PSEUDOTIME_INFERRED    = "pseudotime_inferred"
PROV_VELOCITY_INFERRED      = "rna_velocity_inferred"
PROV_PERTURBATION_DIRECT    = "perturbation_direct"
PROV_KNN_FLOW               = "knn_flow_heuristic"
PROV_OPTIMAL_TRANSPORT      = "optimal_transport"
PROV_NUMPY_PCA              = "numpy_pca_backend"      # test / no-scanpy path
PROV_PCA_DIFFUSION          = "pca_diffusion_backend"  # scanpy PCA + diffmap
PROV_SCVI                   = "scvi_backend"           # Phase 2

# ---------------------------------------------------------------------------
# Basin classification thresholds
# ---------------------------------------------------------------------------

PATHOLOGICAL_ENRICHMENT_THRESHOLD: float = 0.60   # fraction disease cells → pathological
HEALTHY_ENRICHMENT_THRESHOLD:      float = 0.60   # fraction control cells → healthy
ESCAPE_STABILITY_THRESHOLD:        float = 0.40   # kNN purity below this → candidate escape

# ---------------------------------------------------------------------------
# Trajectory scoring weights — DEPRECATED (Phase C refactor)
# W_TRAJ_* constants removed. New scoring uses TherapeuticRedirectionResult.final_score.
# These aliases kept only for backward compatibility with any external callers;
# do not use in new code.
# ---------------------------------------------------------------------------

W_TRAJECTORY_REDIRECTION: float = 0.35   # deprecated — do not use
W_DURABILITY:             float = 0.25   # deprecated — do not use
W_ESCAPE_PENALTY:         float = 0.20   # deprecated — do not use
W_NEGATIVE_MEMORY:        float = 0.20   # deprecated — do not use

# ---------------------------------------------------------------------------
# Failure mode vocabulary (mirrors FailureRecord.failure_mode field)
# ---------------------------------------------------------------------------

FAILURE_MODES: tuple[str, ...] = (
    "no_effect",
    "transient_only",
    "non_responder",
    "escape",
    "discordant_genetic_support",
    "toxicity_limit",
    "disease_context_mismatch",
    "cell_type_mismatch",
    "donor_specific_resistance",
)
