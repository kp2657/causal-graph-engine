"""
agent_runner.py — Claude Agent SDK runner for subagent dispatch.

Supports two modes:
  - "local":  call the existing agent run() function directly (current behavior)
  - "sdk":    call Claude via the Anthropic API with tool_use for structured output

The mode can be set per-agent, enabling gradual migration from local → sdk
without breaking the working pipeline.

Typical usage (chief_of_staff):
    runner = AgentRunner()

    # Current: local mode (no API call, just wraps existing run())
    output = runner.dispatch("somatic_exposure_agent", input_env)

    # Future: SDK mode (real Claude subagent)
    runner.set_mode("somatic_exposure_agent", "sdk")
    output = runner.dispatch("somatic_exposure_agent", input_env)
"""
from __future__ import annotations

import importlib
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestrator.message_contracts import AgentInput, AgentOutput, wrap_output

# ---------------------------------------------------------------------------
# Agent module registry — maps agent_name → importable module path
# ---------------------------------------------------------------------------

_AGENT_MODULES: dict[str, str] = {
    "phenotype_architect":         "agents.tier1_phenomics.phenotype_architect",
    "statistical_geneticist":      "agents.tier1_phenomics.statistical_geneticist",
    "somatic_exposure_agent":      "agents.tier1_phenomics.somatic_exposure_agent",
    "perturbation_genomics_agent": "agents.tier2_pathway.perturbation_genomics_agent",
    "regulatory_genomics_agent":   "agents.tier2_pathway.regulatory_genomics_agent",
    "causal_discovery_agent":      "agents.tier3_causal.causal_discovery_agent",
    "kg_completion_agent":         "agents.tier3_causal.kg_completion_agent",
    "target_prioritization_agent": "agents.tier4_translation.target_prioritization_agent",
    "chemistry_agent":             "agents.tier4_translation.chemistry_agent",
    "clinical_trialist_agent":     "agents.tier4_translation.clinical_trialist_agent",
    "scientific_writer_agent":     "agents.tier5_writer.scientific_writer_agent",
    "scientific_reviewer_agent":   "agents.tier5_writer.scientific_reviewer_agent",
    "literature_validation_agent": "agents.tier5_writer.literature_validation_agent",
    "chief_of_staff_agent":           "agents.cso.chief_of_staff_agent",
    "red_team_agent":                 "agents.tier5_writer.red_team_agent",
    "discovery_refinement_agent":     "agents.discovery.discovery_refinement_agent",
}

# ---------------------------------------------------------------------------
# Per-agent system prompt paths
# ---------------------------------------------------------------------------

_PROMPT_PATHS: dict[str, str] = {
    "phenotype_architect":         "agents/tier1_phenomics/prompts/phenotype_architect.md",
    "statistical_geneticist":      "agents/tier1_phenomics/prompts/statistical_geneticist.md",
    "somatic_exposure_agent":      "agents/tier1_phenomics/prompts/somatic_exposure_agent.md",
    "perturbation_genomics_agent": "agents/tier2_pathway/prompts/perturbation_genomics_agent.md",
    "regulatory_genomics_agent":   "agents/tier2_pathway/prompts/regulatory_genomics_agent.md",
    "causal_discovery_agent":      "agents/tier3_causal/prompts/causal_discovery_agent.md",
    "kg_completion_agent":         "agents/tier3_causal/prompts/kg_completion_agent.md",
    "target_prioritization_agent": "agents/tier4_translation/prompts/target_prioritization_agent.md",
    "chemistry_agent":             "agents/tier4_translation/prompts/chemistry_agent.md",
    "clinical_trialist_agent":     "agents/tier4_translation/prompts/clinical_trialist_agent.md",
    "scientific_writer_agent":          "agents/tier5_writer/prompts/scientific_writer_agent.md",
    "literature_validation_agent":      "agents/tier5_writer/prompts/literature_validation_agent.md",
    "chief_of_staff_agent":             "agents/cso/prompts/chief_of_staff_agent.md",
    "red_team_agent":                   "agents/tier5_writer/prompts/red_team_agent.md",
    "discovery_refinement_agent":       "agents/discovery/prompts/discovery_refinement_agent.md",
}

# ---------------------------------------------------------------------------
# Per-agent MCP tool category assignments (ToolUniverse categories)
# Used when building the tool list for SDK mode.
# ---------------------------------------------------------------------------

AGENT_TOOL_CATEGORIES: dict[str, list[str]] = {
    "phenotype_architect": [
        "opentarget", "disease_target_score", "orphanet", "gnomad",
    ],
    "statistical_geneticist": [
        "gwas", "gnomad", "gtex_v2", "ensembl",
    ],
    "somatic_exposure_agent": [
        "cbioportal", "civic", "epigenomics",
    ],
    "perturbation_genomics_agent": [
        "hpa", "software_single_cell", "ensembl",
    ],
    "regulatory_genomics_agent": [
        "reactome", "intact", "proteins_api", "ensembl",
    ],
    "causal_discovery_agent": [
        "opentarget", "disease_target_score", "gnomad", "gwas",
    ],
    "kg_completion_agent": [
        "openalex", "opentarget", "reactome", "intact",
    ],
    "target_prioritization_agent": [
        "opentarget", "disease_target_score", "clinical_trials", "gnomad",
    ],
    "chemistry_agent": [
        "chembl", "pubchem", "admetai",
    ],
    "clinical_trialist_agent": [
        "clinical_trials", "ada_aha_nccn", "guidelines",
    ],
    "scientific_writer_agent": [
        "openalex",
    ],
}

# ---------------------------------------------------------------------------
# Tool schemas for causal_discovery_agent SDK mode
# ---------------------------------------------------------------------------

_CAUSAL_DISCOVERY_TOOLS: list[dict] = [
    {
        "name": "compute_ota_gammas",
        "description": (
            "Run the full Ota composite γ computation + SCONE sensitivity reweighting "
            "for all genes. Returns gene_gamma_records (one per gene×trait), "
            "anchor_gene_set (validated disease anchors), required_anchors, and warnings. "
            "Call this first to get all scored edge candidates."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "beta_matrix_result": {
                    "type": "object",
                    "description": "Output of perturbation_genomics_agent: {genes, beta_matrix, evidence_tier_per_gene, programs}",
                },
                "gamma_estimates": {
                    "type": "object",
                    "description": "Program→trait γ matrix: {program: {trait: float}}",
                },
                "disease_query": {
                    "type": "object",
                    "description": "Disease context dict with disease_name, efo_id, etc.",
                },
            },
            "required": ["beta_matrix_result", "gamma_estimates", "disease_query"],
        },
    },
    {
        "name": "write_causal_edges",
        "description": (
            "Write a list of selected causal edges to the Kùzu graph database. "
            "Each edge must have from_node, from_type, to_node, to_type, effect_size, "
            "evidence_type, evidence_tier, method, data_source fields. "
            "Call after selecting which gene_gamma_records pass your inclusion criteria."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "edges": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "List of CausalEdge dicts to write.",
                },
                "disease_name": {
                    "type": "string",
                    "description": "Disease name, e.g. 'coronary artery disease'",
                },
            },
            "required": ["edges", "disease_name"],
        },
    },
    {
        "name": "check_anchor_recovery",
        "description": (
            "Check what fraction of required disease anchor edges are present in written_edges. "
            "Also queries the DB for previously-written edges (somatic/CHIP from Tier 1). "
            "Returns recovery_rate, recovered list, missing list, required_anchors. "
            "Minimum acceptable recovery_rate is 0.80."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "written_edges": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "CausalEdge dicts written so far (from_node, to_node required).",
                },
                "disease_query": {
                    "type": "object",
                    "description": "Disease context dict.",
                },
            },
            "required": ["written_edges", "disease_query"],
        },
    },
    {
        "name": "compute_shd",
        "description": (
            "Compute Structural Hamming Distance between the predicted causal graph "
            "and the reference anchor graph for this disease. "
            "Returns shd, extra_edges (false positives), missing_edges (false negatives)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "predicted_edges": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "List of {from_node, to_node} dicts.",
                },
                "disease_query": {
                    "type": "object",
                    "description": "Disease context dict.",
                },
            },
            "required": ["predicted_edges", "disease_query"],
        },
    },
]


# ---------------------------------------------------------------------------
# Execution tools — available to all autonomous SDK agents
# ---------------------------------------------------------------------------

_EXECUTION_TOOLS: list[dict] = [
    {
        "name": "run_python",
        "description": (
            "Execute Python code in the project virtualenv and return stdout/stderr. "
            "Use this to run any analysis, call any MCP server function, inspect data files, "
            "debug failures, or try alternative approaches. "
            "The code runs with the project root as working directory — all imports work. "
            "Print JSON to stdout to return structured results. "
            "If a tool or API call fails, investigate why using this tool and try alternatives "
            "before giving up."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python source code to execute",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Time limit in seconds (default 60, max 120)",
                },
            },
            "required": ["code"],
        },
    },
    {
        "name": "read_project_file",
        "description": (
            "Read a file within the project directory. "
            "Use to inspect data files, cached results, benchmark configs, logs, "
            "or any other file that may inform your analysis. "
            "Path is relative to project root, e.g. 'data/benchmarks/ibd_upstream_regulators_v1.json'"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "relative_path": {
                    "type": "string",
                    "description": "File path relative to project root",
                },
            },
            "required": ["relative_path"],
        },
    },
    {
        "name": "list_project_files",
        "description": (
            "Glob for files within the project directory. "
            "Use to discover what data files, caches, or configs exist before reading them. "
            "Pattern is relative to project root, e.g. 'data/cellxgene/**/*.h5ad'"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern relative to project root (supports **)",
                },
            },
            "required": ["pattern"],
        },
    },
]

# Agents that receive execution tools in SDK mode.
# These are agents where autonomous investigation meaningfully improves output quality.
_AUTONOMOUS_AGENTS: frozenset[str] = frozenset({
    "statistical_geneticist",
    "perturbation_genomics_agent",
    "causal_discovery_agent",
    "chemistry_agent",
    "clinical_trialist_agent",
    "regulatory_genomics_agent",
    "literature_validation_agent",
    "discovery_refinement_agent",
})


# ---------------------------------------------------------------------------
# Tool schemas for chemistry_agent SDK mode
# ---------------------------------------------------------------------------

_CHEMISTRY_TOOLS: list[dict] = [
    {
        "name": "get_chembl_target_activities",
        "description": (
            "Fetch IC50/Ki activity data for a gene target from ChEMBL. "
            "Returns a list of activities with standard_value (nM), standard_type, "
            "molecule_chembl_id, and target_chembl_id. "
            "Use to check tractability and find potent compounds."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "target_gene": {
                    "type": "string",
                    "description": "HGNC gene symbol, e.g. 'PCSK9'",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of activity records to return (default 20)",
                },
            },
            "required": ["target_gene"],
        },
    },
    {
        "name": "get_open_targets_targets_bulk",
        "description": (
            "Batch fetch tractability class, max clinical phase, and known drugs "
            "for a list of gene symbols from Open Targets. "
            "More efficient than individual calls. Use first to triage all targets."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "gene_symbols": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of HGNC gene symbols",
                },
            },
            "required": ["gene_symbols"],
        },
    },
    {
        "name": "search_chembl_compound",
        "description": (
            "Search ChEMBL for a compound by name. Returns chembl_id, smiles, "
            "molecular weight, and max clinical phase. "
            "Use to look up a specific drug or compound."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Drug or compound name, e.g. 'atorvastatin'",
                },
            },
            "required": ["name"],
        },
    },
    {
        "name": "get_pubchem_compound",
        "description": (
            "Fetch compound info from PubChem by name or CID. "
            "Returns molecular formula, weight, InChI, canonical SMILES. "
            "Use for SMILES to pass to ADMET prediction."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name_or_cid": {
                    "type": "string",
                    "description": "Compound name or PubChem CID",
                },
            },
            "required": ["name_or_cid"],
        },
    },
    {
        "name": "run_admet_prediction",
        "description": (
            "Predict ADMET properties for a list of SMILES strings. "
            "Returns solubility, permeability, hERG liability, metabolic stability. "
            "Use to flag compounds with poor drug-like properties."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "smiles_list": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of canonical SMILES strings",
                },
            },
            "required": ["smiles_list"],
        },
    },
]


# ---------------------------------------------------------------------------
# Tool schemas for clinical_trialist_agent SDK mode
# ---------------------------------------------------------------------------

_CLINICAL_TRIALS_TOOLS: list[dict] = [
    {
        "name": "search_clinical_trials",
        "description": (
            "Search ClinicalTrials.gov for trials by condition and/or intervention. "
            "Returns list of trials with nct_id, status, phase, why_stopped. "
            "Use for each target gene and its known drugs."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "condition": {
                    "type": "string",
                    "description": "Disease/condition name",
                },
                "intervention": {
                    "type": "string",
                    "description": "Drug name or gene target",
                },
                "status": {
                    "type": "string",
                    "description": "Trial status filter (RECRUITING, COMPLETED, etc.) or null for all",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Max results to return",
                },
            },
            "required": ["condition"],
        },
    },
    {
        "name": "get_trial_details",
        "description": (
            "Fetch full details for a specific trial by NCT ID. "
            "Returns status, primary_outcome, enrollment, why_stopped. "
            "Use to investigate terminated trials for safety signals."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "nct_id": {
                    "type": "string",
                    "description": "ClinicalTrials.gov NCT identifier, e.g. 'NCT01764633'",
                },
            },
            "required": ["nct_id"],
        },
    },
    {
        "name": "get_trials_for_target",
        "description": (
            "Find all trials targeting a specific gene, optionally filtered by disease. "
            "Returns trial list with phase and status. "
            "Use as a quick sweep before per-drug searches."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "gene_symbol": {
                    "type": "string",
                    "description": "HGNC gene symbol",
                },
                "disease": {
                    "type": "string",
                    "description": "Disease name filter (optional)",
                },
            },
            "required": ["gene_symbol"],
        },
    },
    {
        "name": "get_open_targets_drug_info",
        "description": (
            "Fetch drug indications and approval status from Open Targets. "
            "Returns indications list (other diseases this drug is approved for). "
            "Use to identify repurposing opportunities."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "drug_name": {
                    "type": "string",
                    "description": "Drug name, e.g. 'tocilizumab'",
                },
            },
            "required": ["drug_name"],
        },
    },
]


_DISCOVERY_REFINEMENT_TOOLS: list[dict] = [
    # ---- GWAS / Genetics ----
    {
        "name": "get_gwas_instruments_for_gene",
        "description": (
            "Find GWAS genetic instruments (SNPs) near a gene for a specific disease. "
            "Returns rsid, beta, pvalue, se for each instrument. "
            "Use to check if a state-nominated or Perturb-seq gene has any GWAS support."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "gene_symbol": {"type": "string", "description": "HGNC gene symbol"},
                "efo_id":      {"type": "string", "description": "EFO disease ID, e.g. 'EFO_0003767'"},
                "p_threshold": {"type": "number",  "description": "P-value threshold (default 5e-8)"},
                "window_kb":   {"type": "integer", "description": "Window around gene in kb (default 500)"},
            },
            "required": ["gene_symbol", "efo_id"],
        },
    },
    {
        "name": "query_gtex_eqtl",
        "description": (
            "Query GTEx for cis-eQTLs for a gene in a specified tissue. "
            "Returns snpId, pvalue, nes (normalized effect size). "
            "Use to find eQTL instruments for genes lacking GWAS hits."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "gene_symbol": {"type": "string", "description": "HGNC gene symbol"},
                "tissue":      {"type": "string", "description": "GTEx tissue ID, e.g. 'Colon_Transverse'"},
            },
            "required": ["gene_symbol", "tissue"],
        },
    },
    {
        "name": "query_gnomad_lof_constraint",
        "description": (
            "Get LoF constraint metrics (pLI, LOEUF) for a list of genes from gnomAD. "
            "pLI > 0.9 = essential gene — on-target toxicity risk. "
            "Use to flag safety concerns for top candidates."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "genes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of HGNC gene symbols",
                },
            },
            "required": ["genes"],
        },
    },
    {
        "name": "get_coloc_h4_posteriors",
        "description": (
            "Check colocalization H4 posterior probability between a gene's eQTL "
            "and a GWAS trait. H4 > 0.5 = shared causal variant (strong evidence). "
            "Use to confirm genetic instruments or detect cross-disease overlap."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "gene":       {"type": "string", "description": "HGNC gene symbol"},
                "trait_efo":  {"type": "string", "description": "EFO disease ID"},
            },
            "required": ["gene", "trait_efo"],
        },
    },
    {
        "name": "get_open_targets_genetics_credible_sets",
        "description": (
            "Get credible sets (fine-mapped GWAS loci) for a disease from OT Genetics. "
            "Returns lead variant, posterior inclusion probabilities, and mapped genes. "
            "Use to check if a functional candidate gene is in a GWAS credible set."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "efo_id":   {"type": "string", "description": "EFO disease ID"},
                "min_pip":  {"type": "number",  "description": "Min posterior inclusion probability (default 0.1)"},
            },
            "required": ["efo_id"],
        },
    },
    {
        "name": "get_l2g_scores",
        "description": (
            "Get locus-to-gene (L2G) scores for a GWAS study from OT Genetics. "
            "Returns per-locus gene rankings. Combine with credible sets to identify "
            "likely causal gene at each GWAS locus."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "study_id": {"type": "string", "description": "OT Genetics study ID"},
                "top_n":    {"type": "integer", "description": "Top N genes to return (default 10)"},
            },
            "required": ["study_id"],
        },
    },
    # ---- Open Targets ----
    {
        "name": "get_open_targets_target_info",
        "description": (
            "Get tractability class, approved name, and known drugs for a single gene. "
            "tractability_class: 'small_molecule' | 'antibody' | 'other_modality' | 'unknown'. "
            "Use to assess druggability for state-nominated or novel targets."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "gene_symbol": {"type": "string", "description": "HGNC gene symbol"},
            },
            "required": ["gene_symbol"],
        },
    },
    {
        "name": "get_open_targets_targets_bulk",
        "description": (
            "Batch fetch tractability, max phase, and known drugs for multiple genes. "
            "More efficient than individual calls. Use for initial druggability triage."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "gene_symbols": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of HGNC gene symbols",
                },
            },
            "required": ["gene_symbols"],
        },
    },
    {
        "name": "get_ot_genetic_scores_for_gene_set",
        "description": (
            "Get Open Targets genetic association scores for a list of genes in a disease. "
            "Returns per-gene genetic_score (0–1). Use to rank genes by OT evidence strength."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "efo_id":       {"type": "string", "description": "EFO disease ID"},
                "gene_symbols": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["efo_id", "gene_symbols"],
        },
    },
    {
        "name": "get_open_targets_disease_targets",
        "description": (
            "Get all Open Targets disease-gene associations for a disease, ranked by overall score. "
            "Use to find genes the pipeline may have missed that OT considers high-confidence."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "efo_id":          {"type": "string", "description": "EFO disease ID"},
                "max_targets":     {"type": "integer", "description": "Max targets to return (default 20)"},
                "min_overall_score": {"type": "number", "description": "Min score filter (default 0.2)"},
            },
            "required": ["efo_id"],
        },
    },
    # ---- Chemistry ----
    {
        "name": "get_chembl_target_activities",
        "description": (
            "Fetch IC50/Ki activity data for a target gene from ChEMBL. "
            "Use to check if small-molecule tool compounds exist for tractability assessment."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "target_gene": {"type": "string", "description": "HGNC gene symbol"},
                "max_results": {"type": "integer", "description": "Max activity records (default 20)"},
            },
            "required": ["target_gene"],
        },
    },
    {
        "name": "search_chembl_compound",
        "description": (
            "Search ChEMBL for a compound by name. Returns chembl_id, max_phase, MW, SMILES. "
            "Use to look up a specific drug compound by name."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Drug or compound name"},
            },
            "required": ["name"],
        },
    },
    # ---- Perturb-seq ----
    {
        "name": "find_upstream_regulators",
        "description": (
            "Find upstream transcriptional regulators of a gene set using Perturb-seq data. "
            "Returns regulators with beta (perturbation effect on downstream gene set). "
            "Use to identify druggable TFs/signaling nodes that control disease programs."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "downstream_genes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of downstream target gene symbols",
                },
                "disease_context": {
                    "type": "string",
                    "description": "Disease context: 'IBD', 'CAD', 'RA', 'T2D', 'AD', 'SLE'",
                },
                "max_regulators": {
                    "type": "integer",
                    "description": "Max regulators to return (default 20)",
                },
            },
            "required": ["downstream_genes"],
        },
    },
    # ---- Literature ----
    {
        "name": "search_gene_disease_literature",
        "description": (
            "Search PubMed for papers linking a gene to a disease. "
            "Use for novel targets (lit_confidence=NOVEL) to find mechanism papers."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "gene":        {"type": "string", "description": "HGNC gene symbol"},
                "disease":     {"type": "string", "description": "Disease name"},
                "max_results": {"type": "integer", "description": "Max papers (default 10)"},
            },
            "required": ["gene", "disease"],
        },
    },
    {
        "name": "search_pubmed",
        "description": (
            "Free-text PubMed search. Use to explore mechanism papers for novel genes "
            "using pathway/cell-type keywords rather than disease name."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query":       {"type": "string", "description": "PubMed query string"},
                "max_results": {"type": "integer", "description": "Max results (default 10)"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "fetch_pubmed_abstract",
        "description": "Fetch abstract text and metadata for a PubMed article by PMID.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pmid": {"type": "string", "description": "PubMed ID"},
            },
            "required": ["pmid"],
        },
    },
    # ---- Clinical ----
    {
        "name": "get_trials_for_target",
        "description": (
            "Find clinical trials targeting a specific gene. "
            "Use to check if any existing drugs in trials could be repurposed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "gene_symbol": {"type": "string", "description": "HGNC gene symbol"},
                "disease":     {"type": "string", "description": "Disease filter (optional)"},
            },
            "required": ["gene_symbol"],
        },
    },
]


_LITERATURE_TOOLS: list[dict] = [
    {
        "name": "search_gene_disease_literature",
        "description": (
            "Search PubMed for papers linking a specific gene to a disease. "
            "Returns articles with pmid, title, authors, year, journal. "
            "Use for each top target gene to find supporting/contradicting evidence."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "gene":        {"type": "string", "description": "HGNC gene symbol, e.g. 'NOD2'"},
                "disease":     {"type": "string", "description": "Disease name, e.g. 'inflammatory bowel disease'"},
                "max_results": {"type": "integer", "description": "Max papers to return (default 10)"},
                "source":      {"type": "string", "description": "'pubmed' (default) or 'europepmc'"},
            },
            "required": ["gene", "disease"],
        },
    },
    {
        "name": "fetch_pubmed_abstract",
        "description": (
            "Fetch the full abstract and metadata for a PubMed article by PMID. "
            "Use to read the abstract and classify supporting vs contradicting evidence."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pmid": {"type": "string", "description": "PubMed ID, e.g. '32694926'"},
            },
            "required": ["pmid"],
        },
    },
    {
        "name": "search_pubmed",
        "description": (
            "Search PubMed with a free-text query. Use when gene + disease search returns no results — "
            "try broader queries like gene name with pathway keywords."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query":       {"type": "string", "description": "PubMed query string"},
                "max_results": {"type": "integer", "description": "Max results (default 10)"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_europe_pmc",
        "description": (
            "Search Europe PMC for open-access literature. "
            "Use as fallback when PubMed returns sparse results."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query":       {"type": "string", "description": "Search query"},
                "max_results": {"type": "integer", "description": "Max results (default 10)"},
            },
            "required": ["query"],
        },
    },
]


# Structured output tool — every SDK agent returns its result via this tool
_RETURN_RESULT_TOOL = {
    "name": "return_result",
    "description": (
        "Return the final structured result for this agent. "
        "Call this once when you have completed your analysis."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "result": {
                "type": "object",
                "description": "The agent's complete output as a JSON object.",
            },
            "warnings": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Any warnings or quality flags from this agent.",
            },
            "edges_written": {
                "type": "integer",
                "description": "Number of causal edges written to the graph DB.",
                "default": 0,
            },
        },
        "required": ["result"],
    },
}


# ---------------------------------------------------------------------------
# AgentRunner
# ---------------------------------------------------------------------------

class AgentRunner:
    """
    Dispatch agent calls in local or SDK mode.

    Local mode:  import and call existing agent run() functions.
    SDK mode:    call Claude via Anthropic API with tool_use.

    Agents default to local mode. Flip individual agents to SDK mode via
    set_mode() to enable gradual migration.
    """

    DEFAULT_MODEL = "claude-opus-4-6"
    MAX_RETRIES = 2

    def __init__(
        self,
        model: str | None = None,
        project_root: str | Path | None = None,
    ) -> None:
        self._model = model or self.DEFAULT_MODEL
        self._project_root = Path(project_root or Path(__file__).parent.parent)
        self._modes: dict[str, str] = {}         # agent_name → "local" | "sdk"
        self._model_overrides: dict[str, str] = {}  # agent_name → model string
        self._client: Any = None               # lazy-init Anthropic client
        self._total_input_tokens:  int = 0     # cumulative across all SDK calls
        self._total_output_tokens: int = 0

    # ------------------------------------------------------------------
    # Mode management
    # ------------------------------------------------------------------

    def set_model(self, agent_name: str, model: str) -> None:
        """Override the model for a specific agent (e.g. haiku for cheap screening agents)."""
        self._model_overrides[agent_name] = model

    def get_token_usage(self) -> dict:
        """Return cumulative token usage and estimated cost across all SDK calls."""
        # Pricing: claude-sonnet-4-6 — $3/M input, $15/M output (as of 2026-04)
        cost_input  = self._total_input_tokens  / 1_000_000 * 3.00
        cost_output = self._total_output_tokens / 1_000_000 * 15.00
        return {
            "input_tokens":  self._total_input_tokens,
            "output_tokens": self._total_output_tokens,
            "total_tokens":  self._total_input_tokens + self._total_output_tokens,
            "estimated_cost_usd": round(cost_input + cost_output, 4),
        }

    def set_mode(self, agent_name: str, mode: str) -> None:
        """Set an agent to 'local' or 'sdk' mode."""
        if mode not in ("local", "sdk"):
            raise ValueError(f"mode must be 'local' or 'sdk', got {mode!r}")
        self._modes[agent_name] = mode

    def set_all_sdk(self) -> None:
        """Switch every agent to SDK mode (full multiagent deployment)."""
        for name in _AGENT_MODULES:
            self._modes[name] = "sdk"

    def get_mode(self, agent_name: str) -> str:
        return self._modes.get(agent_name, "local")

    # ------------------------------------------------------------------
    # Primary dispatch
    # ------------------------------------------------------------------

    def dispatch(self, agent_name: str, agent_input: AgentInput) -> AgentOutput:
        """
        Run an agent and return a typed AgentOutput envelope.

        Args:
            agent_name:   One of the registered agent names.
            agent_input:  Typed AgentInput envelope.

        Returns:
            AgentOutput with results, warnings, edges_written, etc.
        """
        mode = self.get_mode(agent_name)
        if mode == "sdk":
            return self._dispatch_sdk(agent_name, agent_input)
        return self._dispatch_local(agent_name, agent_input)

    # ------------------------------------------------------------------
    # Local dispatch (wraps existing run() functions)
    # ------------------------------------------------------------------

    def _dispatch_local(self, agent_name: str, agent_input: AgentInput) -> AgentOutput:
        """Call the agent's existing run() function with retry."""
        module_path = _AGENT_MODULES.get(agent_name)
        if not module_path:
            # Unknown agent name can't pass the AgentName Literal validator —
            # construct the envelope directly, bypassing field validation.
            return AgentOutput.model_construct(
                tier="tier1",
                agent_name=agent_name,
                results={"error": f"Unknown agent: {agent_name}"},
                edges_written=0,
                warnings=[f"Unknown agent: {agent_name}"],
                escalate=False,
                escalation_reason=None,
                duration_s=None,
                stub_fallback=True,
            )

        last_exc: Exception | None = None
        for attempt in range(self.MAX_RETRIES):
            try:
                t0 = time.time()
                raw = self._call_local(agent_name, module_path, agent_input)
                duration = time.time() - t0
                edges = raw.get("n_edges_written", 0) + raw.get("edges_written", 0)
                return wrap_output(
                    agent_name, raw,
                    edges_written=edges,
                    duration_s=round(duration, 2),
                )
            except Exception as exc:
                last_exc = exc
                if attempt < self.MAX_RETRIES - 1:
                    print(f"[RETRY] {agent_name}: {exc}")

        return wrap_output(
            agent_name,
            {"error": str(last_exc), "warnings": [f"{agent_name} failed: {last_exc}"]},
            stub_fallback=True,
        )

    def _call_local(self, agent_name: str, module_path: str, inp: AgentInput) -> dict:
        """Import and call the agent's run() function with the right arguments."""
        mod = importlib.import_module(module_path)
        run_fn = mod.run
        dq = inp.disease_query
        up = inp.upstream_results

        # Match each agent's existing run() signature
        if agent_name == "phenotype_architect":
            return run_fn(dq.get("disease_name", ""))

        if agent_name == "statistical_geneticist":
            return run_fn(dq)

        if agent_name == "somatic_exposure_agent":
            return run_fn(dq)

        if agent_name == "perturbation_genomics_agent":
            gene_list = up.get("_gene_list", [])
            return run_fn(gene_list, dq)

        if agent_name == "regulatory_genomics_agent":
            gene_list = up.get("_gene_list", [])
            return run_fn(gene_list, dq)

        if agent_name == "causal_discovery_agent":
            beta_result = up.get("perturbation_genomics_agent", {})
            gamma_estimates = up.get("_gamma_estimates", {})
            return run_fn(beta_result, gamma_estimates, dq)

        if agent_name == "kg_completion_agent":
            causal_result = up.get("causal_discovery_agent", {})
            return run_fn(causal_result, dq)

        if agent_name == "target_prioritization_agent":
            causal_result = up.get("causal_discovery_agent", {})
            kg_result = up.get("kg_completion_agent", {})
            return run_fn(causal_result, kg_result, dq)

        if agent_name == "chemistry_agent":
            prioritization_result = up.get("target_prioritization_agent", {})
            return run_fn(prioritization_result, dq)

        if agent_name == "clinical_trialist_agent":
            prioritization_result = up.get("target_prioritization_agent", {})
            return run_fn(prioritization_result, dq)

        if agent_name == "scientific_writer_agent":
            writer_keys = [
                "phenotype_result", "genetics_result", "somatic_result",
                "beta_matrix_result", "regulatory_result", "causal_result",
                "kg_result", "prioritization_result", "chemistry_result",
                "trials_result",
            ]
            return run_fn(**{k: up.get(k, {}) for k in writer_keys})

        if agent_name == "scientific_reviewer_agent":
            # Reviewer receives the full pipeline_outputs dict + disease_query
            return run_fn(pipeline_outputs=up, disease_query=dq)

        if agent_name == "literature_validation_agent":
            prioritization_result = up.get("prioritization_result", {})
            return run_fn(prioritization_result, dq)

        if agent_name == "chief_of_staff_agent":
            return run_fn(dq, up)

        if agent_name == "discovery_refinement_agent":
            return run_fn(up, dq)

        if agent_name == "red_team_agent":
            prioritization_result = up.get("prioritization_result", {})
            literature_result = up.get("literature_result", {})
            beta_matrix_result = up.get("beta_matrix_result", {})
            gamma_estimates = up.get("_gamma_estimates", {})
            return run_fn(
                prioritization_result, literature_result, dq,
                beta_matrix_result=beta_matrix_result,
                gamma_estimates=gamma_estimates,
            )

        raise ValueError(f"No local dispatch mapping for agent: {agent_name}")

    # ------------------------------------------------------------------
    # SDK dispatch (Claude Agent SDK via Anthropic API)
    # ------------------------------------------------------------------

    def _dispatch_sdk(self, agent_name: str, agent_input: AgentInput) -> AgentOutput:
        """
        Call Claude as a subagent using the Anthropic messages API.

        The agent receives its system prompt + input JSON, calls ToolUniverse
        and local MCP tools, then returns structured output via return_result().
        """
        system_prompt = self._load_system_prompt(agent_name)
        tools = self._build_tool_list(agent_name)
        user_message = json.dumps(agent_input.model_dump(), indent=2, default=str)

        messages = [{"role": "user", "content": user_message}]

        last_exc: Exception | None = None
        for attempt in range(self.MAX_RETRIES):
            try:
                client = self._get_client()
                t0 = time.time()
                raw = self._agentic_loop(
                    client, system_prompt, messages, tools, agent_name,
                    max_turns=self._get_max_turns(agent_name),
                )
                duration = time.time() - t0
                edges = raw.get("edges_written", 0)
                in_tok  = raw.pop("_input_tokens",  0)
                out_tok = raw.pop("_output_tokens", 0)
                self._total_input_tokens  += in_tok
                self._total_output_tokens += out_tok
                if in_tok or out_tok:
                    print(f"[TOKENS  ] {agent_name:<30s} in={in_tok:,}  out={out_tok:,}")
                return wrap_output(
                    agent_name, raw.get("result", raw),
                    edges_written=edges,
                    duration_s=round(duration, 2),
                )
            except Exception as exc:
                last_exc = exc
                if attempt < self.MAX_RETRIES - 1:
                    print(f"[SDK_RETRY] {agent_name}: {exc}")

        return wrap_output(
            agent_name,
            {"error": str(last_exc), "warnings": [f"{agent_name} SDK call failed: {last_exc}"]},
            stub_fallback=True,
        )

    def _agentic_loop(
        self,
        client: Any,
        system_prompt: str,
        messages: list[dict],
        tools: list[dict],
        agent_name: str,
        max_turns: int = 20,
    ) -> dict:
        """
        Run the agent loop: model calls tools until return_result is invoked.

        Each tool call is executed and the result is fed back. The loop ends
        when the model calls return_result (structured output) or max_turns is
        reached.
        """
        current_messages = list(messages)
        total_input_tokens = 0
        total_output_tokens = 0

        model = self._model_overrides.get(agent_name, self._model)
        for _ in range(max_turns):
            response = client.messages.create(
                model=model,
                max_tokens=8192,
                system=system_prompt,
                tools=tools,
                messages=current_messages,
            )

            # Accumulate token usage (guard against MagicMock in tests)
            if hasattr(response, "usage") and response.usage:
                _in  = getattr(response.usage, "input_tokens",  0)
                _out = getattr(response.usage, "output_tokens", 0)
                if isinstance(_in,  int): total_input_tokens  += _in
                if isinstance(_out, int): total_output_tokens += _out

            # Append assistant turn
            current_messages.append({
                "role": "assistant",
                "content": response.content,
            })

            # Check stop reason
            if response.stop_reason == "end_turn":
                # No tool call — extract text and wrap as result
                text = " ".join(
                    block.text for block in response.content
                    if hasattr(block, "text")
                )
                return {
                    "result": {"summary": text}, "warnings": [],
                    "_input_tokens": total_input_tokens,
                    "_output_tokens": total_output_tokens,
                }

            if response.stop_reason != "tool_use":
                break

            # Process tool calls
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue

                if block.name == "return_result":
                    # Agent is done — return its structured output
                    result = dict(block.input)
                    result["_input_tokens"]  = total_input_tokens
                    result["_output_tokens"] = total_output_tokens
                    return result

                # Execute other tool calls (ToolUniverse / local MCPs)
                tool_result = self._execute_tool(block.name, block.input, agent_name)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(tool_result, default=str),
                })

            if tool_results:
                current_messages.append({"role": "user", "content": tool_results})

        return {
            "result": {}, "warnings": [f"{agent_name}: max_turns={max_turns} reached"],
            "_input_tokens": total_input_tokens,
            "_output_tokens": total_output_tokens,
        }

    def _execute_tool(self, tool_name: str, tool_input: dict, agent_name: str) -> dict:
        """
        Execute a tool call made by the Claude subagent.

        Routes to local MCP server functions. ToolUniverse tools are handled
        by the MCP server process and don't need manual routing here.
        """
        # Local MCP server routing
        local_routes = self._get_local_tool_routes()
        if tool_name in local_routes:
            try:
                return local_routes[tool_name](**tool_input)
            except Exception as exc:
                return {"error": str(exc), "tool": tool_name}

        return {"error": f"Unknown tool: {tool_name}", "tool": tool_name}

    def _get_local_tool_routes(self) -> dict[str, Any]:
        """Return a map of local MCP tool names to callable functions."""
        routes: dict[str, Any] = {}
        try:
            from mcp_servers import graph_db_server as gdb
            routes.update({
                "write_causal_edges":         gdb.write_causal_edges,
                "query_graph_for_disease":    gdb.query_graph_for_disease,
                "run_anchor_edge_validation": gdb.run_anchor_edge_validation,
                "compute_shd_metric":         gdb.compute_shd_metric,
                "run_evalue_check":           gdb.run_evalue_check,
            })
        except ImportError:
            pass
        try:
            from mcp_servers import gwas_genetics_server as gwas
            routes.update({
                "get_gwas_catalog_associations":     gwas.get_gwas_catalog_associations,
                "query_gnomad_lof_constraint":       gwas.query_gnomad_lof_constraint,
                "query_gtex_eqtl":                   gwas.query_gtex_eqtl,
                "get_gwas_instruments_for_gene":     gwas.get_gwas_instruments_for_gene,
                "get_coloc_h4_posteriors":            gwas.get_coloc_h4_posteriors,
                "get_open_targets_genetics_credible_sets": gwas.get_open_targets_genetics_credible_sets,
                "get_l2g_scores":                    gwas.get_l2g_scores,
            })
        except ImportError:
            pass
        try:
            from mcp_servers import viral_somatic_server as vs
            routes.update({
                "get_chip_disease_associations": vs.get_chip_disease_associations,
                "get_drug_exposure_mr":          vs.get_drug_exposure_mr,
            })
        except ImportError:
            pass
        # causal_discovery_agent SDK tools
        try:
            from agents.tier3_causal.sdk_tools import (
                compute_ota_gammas,
                check_anchor_recovery,
                compute_shd,
            )
            routes.update({
                "compute_ota_gammas":    compute_ota_gammas,
                "check_anchor_recovery": check_anchor_recovery,
                "compute_shd":           compute_shd,
            })
        except ImportError:
            pass
        # chemistry_agent tools
        try:
            from mcp_servers import chemistry_server as chem
            routes.update({
                "get_chembl_target_activities": chem.get_chembl_target_activities,
                "search_chembl_compound":       chem.search_chembl_compound,
                "get_pubchem_compound":         chem.get_pubchem_compound,
                "run_admet_prediction":         chem.run_admet_prediction,
            })
        except ImportError:
            pass
        # clinical_trialist_agent + chemistry_agent shared OT tools
        try:
            from mcp_servers import open_targets_server as ot
            routes.update({
                "get_open_targets_targets_bulk":      ot.get_open_targets_targets_bulk,
                "get_open_targets_drug_info":         ot.get_open_targets_drug_info,
                "get_open_targets_target_info":       ot.get_open_targets_target_info,
                "get_ot_genetic_scores_for_gene_set": ot.get_ot_genetic_scores_for_gene_set,
                "get_open_targets_disease_targets":   ot.get_open_targets_disease_targets,
            })
        except ImportError:
            pass
        # clinical_trialist_agent tools
        try:
            from mcp_servers import clinical_trials_server as ct
            routes.update({
                "search_clinical_trials": ct.search_clinical_trials,
                "get_trial_details":      ct.get_trial_details,
                "get_trials_for_target":  ct.get_trials_for_target,
            })
        except ImportError:
            pass
        # discovery_refinement_agent — perturbseq upstream regulators
        try:
            from mcp_servers import perturbseq_server as ps
            routes.update({
                "find_upstream_regulators":    ps.find_upstream_regulators,
                "get_perturbseq_signature":    ps.get_perturbseq_signature,
            })
        except ImportError:
            pass
        # literature_validation_agent tools
        try:
            from mcp_servers.literature_server import (
                search_gene_disease_literature,
                fetch_pubmed_abstract,
                search_pubmed,
                search_europe_pmc,
            )
            routes.update({
                "search_gene_disease_literature": search_gene_disease_literature,
                "fetch_pubmed_abstract":          fetch_pubmed_abstract,
                "search_pubmed":                  search_pubmed,
                "search_europe_pmc":              search_europe_pmc,
            })
        except ImportError:
            pass
        # execution tools (autonomous agents)
        try:
            from orchestrator.execution_tools import (
                run_python,
                read_project_file,
                list_project_files,
            )
            routes.update({
                "run_python":          run_python,
                "read_project_file":   read_project_file,
                "list_project_files":  list_project_files,
            })
        except ImportError:
            pass
        return routes

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_client(self) -> Any:
        if self._client is None:
            # Load .env so ANTHROPIC_API_KEY is available when running via CLI
            try:
                from dotenv import load_dotenv
                load_dotenv(self._project_root / ".env")
            except ImportError:
                pass  # dotenv optional — rely on environment
            try:
                import anthropic
                self._client = anthropic.Anthropic()
            except ImportError as exc:
                raise RuntimeError(
                    "anthropic package required for SDK mode: pip install anthropic"
                ) from exc
            except Exception as exc:
                # Covers AuthenticationError (missing/bad ANTHROPIC_API_KEY) and
                # any other client-init failure — let the retry loop handle it.
                raise RuntimeError(
                    f"Anthropic client init failed (check ANTHROPIC_API_KEY): {exc}"
                ) from exc
        return self._client

    def _load_system_prompt(self, agent_name: str) -> str:
        prompt_path = self._project_root / _PROMPT_PATHS.get(agent_name, "")
        if prompt_path.exists():
            return prompt_path.read_text()
        return (
            f"You are the {agent_name} agent in a causal genomics pipeline. "
            "Analyze the input data and return structured results using the return_result tool."
        )

    def _build_tool_list(self, agent_name: str) -> list[dict]:
        """
        Build the tool list for an SDK agent call.

        Includes return_result (required) plus:
          - per-agent domain tool schemas
          - execution tools (_EXECUTION_TOOLS) for autonomous agents
        """
        tools = [_RETURN_RESULT_TOOL]
        if agent_name == "causal_discovery_agent":
            tools.extend(_CAUSAL_DISCOVERY_TOOLS)
        elif agent_name == "chemistry_agent":
            tools.extend(_CHEMISTRY_TOOLS)
        elif agent_name == "clinical_trialist_agent":
            tools.extend(_CLINICAL_TRIALS_TOOLS)
        elif agent_name == "literature_validation_agent":
            tools.extend(_LITERATURE_TOOLS)
        elif agent_name == "discovery_refinement_agent":
            tools.extend(_DISCOVERY_REFINEMENT_TOOLS)
        # Pure-reasoning agents — only return_result, no external tools
        elif agent_name in ("chief_of_staff_agent", "red_team_agent"):
            return tools
        # Execution tools for all autonomous agents
        if agent_name in _AUTONOMOUS_AGENTS:
            tools.extend(_EXECUTION_TOOLS)
        return tools

    def _get_max_turns(self, agent_name: str) -> int:
        """Autonomous agents get more turns to investigate and self-correct."""
        if agent_name in _AUTONOMOUS_AGENTS:
            return 40
        # Pure-reasoning agents complete in 1–3 turns
        if agent_name in ("chief_of_staff_agent", "red_team_agent"):
            return 3
        return 20
