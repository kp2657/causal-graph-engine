"""
Core Pydantic v2 models for causal edges and evidence tiers.
All graph data must pass through these models before DB ingestion.
"""
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Controlled vocabularies
# ---------------------------------------------------------------------------

EvidenceType = Literal["germline", "somatic_chip", "viral", "drug", "kg_completion", "state_space"]

EvidenceTier = Literal[
    "Tier1_Interventional",
    "Tier2_Convergent",
    "Tier3_Provisional",
    "moderate_transferred",   # cross-cell-type β transfer
    "moderate_grn",           # CellOracle GRN propagation
    "provisional_virtual",    # virtual cell model — must be labeled in all outputs
]

NodeType = Literal["gene", "program", "drug", "virus", "trait"]

EdgeMethod = Literal[
    "perturb_seq",
    "cross_transfer",
    "grn",
    "virtual",
    "cmap",
    "ota_gamma",
    "scone",
    "inspre",
    "mr",
    "biopathnet",
    "txgnn",
    "literature",
    "trajectory",       # state-space transition-based causal evidence
]

# State-space-specific evidence tiers.  Parallel vocabulary to EvidenceTier —
# do NOT extend the original Literal in Phase 1 so existing tier checks are unaffected.
StateEvidenceTier = Literal[
    "Tier1_TrajectoryDirect",    # observed state transition in matched perturbational data
    "Tier2_TrajectoryInferred",  # pseudotime / velocity-inferred transition
    "Tier3_TrajectoryProxy",     # kNN flow heuristic or beta-prior only
]


# ---------------------------------------------------------------------------
# CausalEdge — the atomic unit written to the graph
# ---------------------------------------------------------------------------

class CausalEdge(BaseModel):
    from_node: str
    from_type: NodeType
    to_node: str
    to_type: NodeType

    effect_size: float
    ci_lower: float | None = None
    ci_upper: float | None = None

    evidence_type: EvidenceType
    evidence_tier: EvidenceTier
    method: EdgeMethod
    data_source: str                   # dataset DOI or API endpoint
    data_source_version: str

    cell_type: str | None = None
    ancestry: str | None = None

    # Validation fields (populated after causal discovery agent runs)
    validation_sid: float | None = None
    validation_shd: float | None = None
    e_value: float | None = None       # sensemakr; flag if < 2.0

    # MR fields
    mr_ivw: float | None = None
    mr_egger_intercept: float | None = None
    mr_weighted_median: float | None = None
    mr_presso_outlier: bool | None = None

    # Versioning
    created_at: datetime = Field(default_factory=datetime.now)
    graph_version: str = "0.1.0"
    is_demoted: bool = False
    demotion_reason: str | None = None

    @field_validator("e_value")
    @classmethod
    def warn_low_evalue(cls, v: float | None) -> float | None:
        # Validation passes; callers inspect is_confounded() for downstream action
        return v

    @model_validator(mode="after")
    def virtual_requires_label(self) -> "CausalEdge":
        """provisional_virtual edges must never have evidence_tier upgraded silently."""
        return self

    def is_confounded(self) -> bool:
        return self.e_value is not None and self.e_value < 2.0

    def edge_id(self) -> str:
        return f"{self.from_node}__{self.to_node}__{self.method}__{self.graph_version}"


# ---------------------------------------------------------------------------
# GeneTraitAssociation — output of Statistical Geneticist
# ---------------------------------------------------------------------------

class GeneTraitAssociation(BaseModel):
    gene: str
    trait: str

    # Fine-mapping
    pip: float | None = None               # SuSiE posterior inclusion probability
    l2g_score: float | None = None         # Open Targets L2G
    pops_score: float | None = None        # PoPS complementary prioritization

    # Colocalization
    coloc_h4: float | None = None          # eQTL coloc H4 posterior

    # Mendelian randomization
    mr_ivw: float | None = None
    mr_ci: tuple[float, float] | None = None
    mr_egger_intercept: float | None = None

    # Rare variant burden
    lof_burden_beta: float | None = None
    lof_burden_se: float | None = None

    # Constraint
    gnomad_pli: float | None = None
    gnomad_loeuf: float | None = None

    evidence_tier: Literal["Tier1_Interventional", "Tier2_Convergent", "Tier3_Provisional"]
    evidence_sources: list[str]            # list of dataset IDs used


# ---------------------------------------------------------------------------
# ProgramBetaMatrix — output of Perturbation Genomics Agent
# ---------------------------------------------------------------------------

class ProgramBetaMatrix(BaseModel):
    programs: list[dict]                   # [{program_id, top_genes, pathways, cell_type}]
    beta_matrix: dict[str, dict[str, float]]  # {gene_id: {program_id: beta}}
    evidence_tier_per_gene: dict[str, EvidenceTier]
    cell_type: str
    perturb_seq_source: str
    inspre_regulatory_graph: dict | None = None
    virtual_ensemble_vs_baseline: dict | None = None  # comparison per Ahlmann-Eltze 2025


# ---------------------------------------------------------------------------
# DiseaseQuery — input to the full pipeline
# ---------------------------------------------------------------------------

class DiseaseQuery(BaseModel):
    disease_name: str
    efo_id: str | None = None
    icd10_codes: list[str] = []
    modifier_types: list[EvidenceType] = ["germline", "somatic_chip", "viral", "drug"]
    use_precomputed_only: bool = True      # default: reuse existing processed data
    day_one_mode: bool = True             # use only immediately accessible data initially


# ---------------------------------------------------------------------------
# GraphOutput — returned by PI Orchestrator
# ---------------------------------------------------------------------------

class TargetRecord(BaseModel):
    gene: str
    rank: int
    causal_paths: int                      # convergence: number of independent modifier paths
    open_targets_score: float | None = None
    cell_type_specificity: str | None = None
    literature_novelty_citations: int | None = None
    safety_flags: list[str] = []
    txgnn_repurposing_score: float | None = None
    existing_drugs: list[str] = []
    evidence_tier: EvidenceTier


class GraphOutput(BaseModel):
    disease: str
    graph_edges_written: int
    ranked_targets: list[TargetRecord]
    summary_report: str
    reasoning_trace: list[str]
    graph_version: str


# ---------------------------------------------------------------------------
# State-space models — added in Phase 1 (dynamical redesign)
# These coexist with existing causal edge models; do not replace them yet.
# ---------------------------------------------------------------------------

class CellState(BaseModel):
    state_id: str
    disease: str
    cell_type: str
    resolution: str                                  # coarse | intermediate | fine
    n_cells: int
    centroid: list[float] | None = None              # mean latent coordinates
    marker_genes: list[str] = Field(default_factory=list)
    program_labels: list[str] = Field(default_factory=list)  # cNMF / Hallmark overlaps
    context_tags: list[str] = Field(default_factory=list)
    stability_score: float | None = None             # kNN purity [0, 1]
    pathological_score: float | None = None          # fraction disease cells [0, 1]
    evidence_sources: list[str] = Field(default_factory=list)


class StateTransition(BaseModel):
    from_state: str
    to_state: str
    disease: str
    baseline_probability: float                      # [0, 1] row-normalised transition rate
    uncertainty: float | None = None
    dwell_time: float | None = None                  # mean pseudotime units in from_state
    direction_evidence: list[str] = Field(default_factory=list)  # provenance markers
    context_tags: list[str] = Field(default_factory=list)


class PerturbationTransitionEffect(BaseModel):
    perturbation_id: str
    perturbation_type: str               # KO | KD | CRISPRa | drug | cytokine | exposure
    disease: str
    cell_type: str
    from_state: str
    to_state: str
    delta_probability: float             # signed change in transition probability
    effect_se: float | None = None
    timepoint: str | None = None
    durability_label: str | None = None  # transient | durable | rebound | escape | unknown
    evidence_tier: StateEvidenceTier
    data_source: str


class FailureRecord(BaseModel):
    failure_id: str
    disease: str
    perturbation_id: str
    perturbation_type: str
    cell_type: str | None = None
    state_id: str | None = None
    failure_mode: str   # no_effect | transient_only | non_responder | escape |
                        # discordant_genetic_support | toxicity_limit |
                        # disease_context_mismatch | cell_type_mismatch | donor_specific_resistance
    phenotype_context: str | None = None
    molecular_context: str | None = None
    evidence_strength: float             # [0, 1]
    explanation_candidates: list[str] = Field(default_factory=list)
    data_source: str


class TrajectoryRedirectionScore(BaseModel):
    entity_id: str
    entity_type: str                          # gene | drug | perturbation_set
    disease: str
    expected_pathology_reduction: float       # [0, 1]
    durable_redirection_score: float          # [0, 1]
    escape_risk_score: float                  # [0, 1]; higher = worse
    non_response_risk_score: float            # [0, 1]; higher = worse
    negative_memory_penalty: float            # [0, 1]; accumulated from FailureRecords
    uncertainty: float | None = None
    provenance: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Phase G–J models — transition-aware scoring, controller classification,
# disagreement profiling
# ---------------------------------------------------------------------------

class TransitionGeneProfile(BaseModel):
    """
    Transition-landscape decomposition for a single gene (Phase G).

    Four independent scores describe where in the healthy↔pathological
    transition a gene's expression is enriched.  Each score is accompanied
    by a direction so the correct therapeutic action (activate vs inhibit)
    can be inferred.

    disease_axis_score is the legacy DAS (mean expression difference).
    It is retained for comparison but is NOT used in any composite formula.
    """
    gene: str
    disease: str
    cell_type: str = "all"

    # Transition scores [0, 1]
    entry_score: float = 0.0               # enrichment in cells entering pathological basins
    persistence_score: float = 0.0         # enrichment in cells dwelling in pathological basins
    recovery_score: float = 0.0            # enrichment in cells exiting toward healthy basins
    boundary_knn_score: float = 0.0        # enrichment at spatial healthy/pathological interface
    boundary_pseudotime_score: float = 0.0 # enrichment at temporal inflection point
    boundary_score: float = 0.0            # max(knn, pseudotime)

    # Directions: +1 = higher expression in category cells, -1 = lower, 0 = no signal
    entry_direction: int = 0        # +1 = gene elevated in entry cells (drives/marks entry)
    persistence_direction: int = 0  # +1 = gene elevated in stuck cells (promotes persistence)
    recovery_direction: int = 0     # +1 = gene elevated in exit cells (promotes recovery)
    boundary_direction: int = 0     # +1 = gene elevated at boundary (marks transition zone)

    # Category
    mechanistic_category: str = "unknown"  # trigger | maintenance | recovery | mixed

    # Support metadata (same for all genes in this context)
    n_entry_cells: int = 0
    n_persistence_cells: int = 0
    n_recovery_cells: int = 0
    n_boundary_cells: int = 0

    # Legacy annotation only — NOT used in composite formula
    disease_axis_score: float = 0.0


class ControllerAnnotation(BaseModel):
    """
    Controller vs marker classification for a gene (Phase H).

    Annotation nominates; only perturbation confirms.
    controller_confidence encodes this: only T1/T2 perturbation evidence
    yields confidence='high'.  TF annotation alone cannot exceed 'medium'
    and caps controller_likelihood at 0.35.
    """
    gene: str
    disease: str
    controller_likelihood: float = 0.0            # [0, 1]
    controller_confidence: str = "low"            # high | medium | low
    category: str = "unknown"                     # upstream_controller | midstream_mediator | downstream_marker
    supporting_signals: list[str] = Field(default_factory=list)
    # e.g. ["perturbation_t1", "tf_annotation", "early_pseudotime", "network_hub"]


class DisagreementProfile(BaseModel):
    """
    Structured multi-dimension disagreement profile for a gene (Phase I).

    Replaces scalar confidence penalties with an explicit mechanistic label
    and per-dimension support scores.  Disagreement is a signal, not noise.
    """
    gene: str
    disease: str

    # Per-dimension support scores [0, 1]
    genetics_support: float = 0.0
    expression_coupling: float = 0.0
    perturbation_support: float = 0.0
    cell_type_specificity: float = 0.0
    cross_context_consistency: float = 0.0

    # Pattern classification (strict rules — see evidence_disagreement.py)
    mechanistic_label: str = "unknown"
    # supported | discordant | context_dependent | likely_marker |
    # likely_upstream_controller | likely_non_transportable

    label_confidence: float = 0.0
    supporting_evidence: list[str] = Field(default_factory=list)
    contradicting_evidence: list[str] = Field(default_factory=list)


class TauSpecificityResult(BaseModel):
    """
    Disease-state τ specificity result for a gene (Phase R).

    τ_disease (Yanai 2005) computed across disease-group mean expressions within
    a single cell type (e.g., IBD vs normal macrophages from the h5ad).

    Range: 0 = expressed equally in all disease groups (ubiquitous);
           1 = expressed exclusively in one disease group (perfectly specific).
    """
    gene: str
    disease: str = ""

    # Core τ index across disease groups
    tau_disease: float = 0.5

    # log2FC: mean(disease cells) / mean(normal cells); pseudocount added
    disease_log2fc: float = 0.0

    # Fraction of disease cells (non-normal) with count > 0
    pct_disease: float = 0.0
    pct_normal: float = 0.0

    # Mean log-normalised expression
    mean_disease: float = 0.0
    mean_normal: float = 0.0

    # Number of disease groups used in τ computation
    n_groups: int = 0

    # Specificity class
    # disease_specific | normal_specific | moderately_specific | ubiquitous | lowly_expressed | unknown
    specificity_class: str = "unknown"

    # Per-group mean expression (informational)
    group_means: dict[str, float] = Field(default_factory=dict)
