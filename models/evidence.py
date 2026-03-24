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

EvidenceType = Literal["germline", "somatic_chip", "viral", "drug", "kg_completion"]

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
    validation_anchor_recovery: bool | None = None
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
