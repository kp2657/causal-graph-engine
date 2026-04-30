"""
_base.py — Shared utilities for all agentic agents.

Provides:
  - Shared system prompt preamble builder
  - Structured output block parser (JSON between <output> tags)
  - Common tool schemas and function wrappers
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


# ---------------------------------------------------------------------------
# Shared prompt components
# ---------------------------------------------------------------------------

_SHARED_HOW_YOU_WORK = """\
## How you work
You receive deterministic pipeline outputs as your starting point. Your job is not to
re-execute what the pipeline already computed. Your job is to find what it missed,
challenge what it got wrong, and surface what it cannot see.

Before calling any tool, state in one sentence:
  "The most likely path to a useful result here is [X] because [Y]."
Then pursue that path. Stop when you have sufficient evidence to make the call.
Do not exhaust a list just because the list exists.

## Evidence confidence levels
- HIGH: ≥2 independent sources (different modalities). State your stopping reason.
- MEDIUM: 1 strong source. Note what would upgrade to HIGH.
- LOW: inference or extrapolation. Flag explicitly.

## Evidence tier hierarchy
- Tier A (genetic/clinical): LOF burden, GWAS colocalization, Mendelian disease gene, clinical outcome
- Tier B (orthogonal perturbation): CRISPRa, bulk RNA-seq KO different system, animal KO phenotype, patient iPSC
- Tier C (correlational/same ecosystem): GPS reversal, co-essentiality, additional Perturb-seq
Convergent evidence requires legs from different tiers. GPS + Perturb-seq = both Tier C = not convergent.

## Escalation criterion
Genuine scientific ambiguity that changes the entire run conclusion — escalate to the
Chief of Staff with one sentence. Do not escalate uncertainty you can resolve with one more tool call.\
"""


def build_system_prompt(persona_block: str, authority_block: str, output_schema: str) -> str:
    return "\n\n".join([
        f"## Who you are\n{persona_block}",
        _SHARED_HOW_YOU_WORK,
        f"## Decision authority\n{authority_block}",
        f"## Output format\nEnd your response with a structured block:\n{output_schema}",
    ])


# ---------------------------------------------------------------------------
# Structured output parser
# ---------------------------------------------------------------------------

def parse_output_block(text: str, tag: str = "output") -> dict:
    """Parse JSON from <tag>...</tag> block. Returns {} on failure."""
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(1).strip())
    except json.JSONDecodeError:
        return {}


# ---------------------------------------------------------------------------
# Tool schema builders
# ---------------------------------------------------------------------------

def _tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "name": name,
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }


# Tool schemas shared across agents
TOOL_SEARCH_PUBMED = _tool(
    "search_pubmed",
    "Search PubMed for biomedical literature. Returns titles, PMIDs, and abstracts.",
    {"query": {"type": "string"}, "max_results": {"type": "integer", "default": 10}},
    ["query"],
)

TOOL_SEARCH_EUROPE_PMC = _tool(
    "search_europe_pmc",
    "Search Europe PMC for literature. Complementary source to PubMed.",
    {"query": {"type": "string"}, "max_results": {"type": "integer", "default": 10}},
    ["query"],
)

TOOL_GET_GENE_BURDEN_STATS = _tool(
    "get_gene_burden_stats",
    "Get LOF burden statistics for genes from UK Biobank WES or gnomAD.",
    {"genes": {"type": "array", "items": {"type": "string"}}, "cohort": {"type": "string", "default": "UKB"}},
    ["genes"],
)

TOOL_GET_OT_TARGET_INFO = _tool(
    "get_open_targets_target_info",
    "Get Open Targets target info: max_phase, known drugs, genetic associations, tractability.",
    {"gene_symbol": {"type": "string"}},
    ["gene_symbol"],
)

TOOL_GET_TRIALS_FOR_TARGET = _tool(
    "get_trials_for_target",
    "Get ClinicalTrials.gov entries for a gene target and optional disease.",
    {"gene_symbol": {"type": "string"}, "disease": {"type": "string"}},
    ["gene_symbol"],
)

TOOL_GET_TRIAL_DETAILS = _tool(
    "get_trial_details",
    "Get detailed information for a specific ClinicalTrials.gov NCT ID.",
    {"nct_id": {"type": "string"}},
    ["nct_id"],
)

TOOL_SEARCH_CLINICAL_TRIALS = _tool(
    "search_clinical_trials",
    "Search ClinicalTrials.gov for trials matching a query.",
    {"query": {"type": "string"}, "status": {"type": "string"}, "max_results": {"type": "integer", "default": 10}},
    ["query"],
)

TOOL_GET_CHEMBL_ACTIVITIES = _tool(
    "get_chembl_target_activities",
    "Get ChEMBL bioactivity data for a gene target. Returns compounds with IC50/Ki values.",
    {"target_gene": {"type": "string"}, "max_results": {"type": "integer", "default": 20}},
    ["target_gene"],
)

TOOL_SEARCH_CHEMBL_COMPOUND = _tool(
    "search_chembl_compound",
    "Search ChEMBL for a compound by name. Returns SMILES, molecular properties, max phase.",
    {"name": {"type": "string"}},
    ["name"],
)

TOOL_RUN_ADMET = _tool(
    "run_admet_prediction",
    "Run ADMET property predictions for a list of SMILES strings.",
    {"smiles_list": {"type": "array", "items": {"type": "string"}}},
    ["smiles_list"],
)

TOOL_GET_OT_DISEASE_TARGETS = _tool(
    "get_open_targets_disease_targets",
    "Get Open Targets L2G-scored genetic targets for a disease. Returns anchor gene list.",
    {"disease_name": {"type": "string"}, "max_targets": {"type": "integer", "default": 50}},
    ["disease_name"],
)


# ---------------------------------------------------------------------------
# Tool function wrappers — import lazily to avoid circular imports at module load
# ---------------------------------------------------------------------------

def _get_tool_functions(names: list[str]) -> dict[str, Any]:
    fns: dict[str, Any] = {}
    if any(n in names for n in ("search_pubmed", "search_europe_pmc", "search_gene_disease_literature")):
        from mcp_servers.literature_server import search_pubmed, search_europe_pmc
        fns["search_pubmed"] = search_pubmed
        fns["search_europe_pmc"] = search_europe_pmc
    if "get_gene_burden_stats" in names:
        from mcp_servers.burden_perturb_server import get_gene_burden_stats
        fns["get_gene_burden_stats"] = get_gene_burden_stats
    if "get_open_targets_target_info" in names:
        from mcp_servers.open_targets_server import get_open_targets_target_info
        fns["get_open_targets_target_info"] = get_open_targets_target_info
    if "get_open_targets_disease_targets" in names:
        from mcp_servers.open_targets_server import get_open_targets_disease_targets
        fns["get_open_targets_disease_targets"] = get_open_targets_disease_targets
    if any(n in names for n in ("get_trials_for_target", "get_trial_details", "search_clinical_trials")):
        from mcp_servers.clinical_trials_server import (
            get_trials_for_target, get_trial_details, search_clinical_trials,
        )
        fns["get_trials_for_target"] = get_trials_for_target
        fns["get_trial_details"] = get_trial_details
        fns["search_clinical_trials"] = search_clinical_trials
    if any(n in names for n in ("get_chembl_target_activities", "search_chembl_compound", "run_admet_prediction")):
        from mcp_servers.chemistry_server import (
            get_chembl_target_activities, search_chembl_compound, run_admet_prediction,
        )
        fns["get_chembl_target_activities"] = get_chembl_target_activities
        fns["search_chembl_compound"] = search_chembl_compound
        fns["run_admet_prediction"] = run_admet_prediction
    return fns
