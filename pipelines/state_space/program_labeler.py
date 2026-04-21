"""
pipelines/state_space/program_labeler.py

Convert raw cNMF/NMF program dicts (from cnmf_runner.run_nmf_programs) into
typed LatentProgram objects with biological annotations.

Phase A v1 annotation strategy:
  1. Rule-based program type assignment from top gene overlap with known gene sets.
  2. MSigDB Hallmark annotations stored as annotation-only (not causal mediators).
  3. GWAS enrichment score injected from external enrichment result if supplied.

Public API:
    label_programs(nmf_result, disease, cell_type, gwas_enrichment)
    label_programs_multi_celltype(nmf_results_by_ct, disease, gwas_enrichment_by_ct)
"""
from __future__ import annotations

from models.latent_mediator import LatentProgram, ProgramType

# ---------------------------------------------------------------------------
# Keyword → ProgramType gene signatures (rule-based, curated)
# ---------------------------------------------------------------------------

_TYPE_GENE_SIGNATURES: dict[ProgramType, frozenset[str]] = {
    "inflammatory": frozenset({
        "TNF", "IL1B", "IL6", "IL8", "CXCL10", "CXCL9", "CXCL8",
        "CCL2", "CCL3", "CCL5", "IFNG", "IL18", "IL23A", "IL12B",
        "NFKB1", "NFKBIA", "TNFRSF1A", "STAT1", "IRF1",
    }),
    "fibrotic": frozenset({
        "TGFB1", "TGFB2", "COL1A1", "COL1A2", "COL3A1", "FN1",
        "ACTA2", "VIM", "POSTN", "TIMP1", "MMP2", "MMP9", "CTGF",
        "LOXL2", "SNAI1", "SNAI2", "ZEB1",
    }),
    "metabolic": frozenset({
        "PPARG", "PPARA", "FASN", "ACACA", "SCD", "HMGCR",
        "LDLR", "ABCA1", "ABCG1", "ADIPOQ", "FABP4", "CPT1A",
        "HADHA", "ACSL1", "PLIN2",
    }),
    "stress_response": frozenset({
        "HSPA1A", "HSPA1B", "HSP90AA1", "HSPB1", "DDIT3",
        "ATF3", "ATF4", "ATF6", "XBP1", "EIF2AK3", "PERK",
        "CHOP", "GRP78", "HMOX1", "NQO1", "SQSTM1",
    }),
    "proliferative": frozenset({
        "MKI67", "TOP2A", "CDK1", "CCNB1", "CCNA2", "PCNA",
        "MCM2", "MCM6", "E2F1", "MYBL2", "BUB1", "PLK1",
        "AURKB", "CENPF", "HIST1H4C",
    }),
    "immunoregulatory": frozenset({
        "FOXP3", "CTLA4", "PDCD1", "LAG3", "TIGIT", "HAVCR2",
        "IL10", "TGFB3", "IDO1", "CD274", "PDCD1LG2",
        "ENTPD1", "CD39", "IL2RA",
    }),
    "angiogenic": frozenset({
        "VEGFA", "VEGFB", "VEGFC", "KDR", "FLT1", "ANGPT1",
        "ANGPT2", "TEK", "PECAM1", "CDH5", "HIF1A", "EPAS1",
        "NOTCH1", "DLL4", "PDGFB",
    }),
}

# MSigDB Hallmark set → keyword markers (used for annotation-only labelling)
_HALLMARK_KEYWORDS: dict[str, list[str]] = {
    "HALLMARK_TNFA_SIGNALING_VIA_NFKB": ["TNF", "NFKB1", "NFKBIA", "IL1B", "IL6"],
    "HALLMARK_INFLAMMATORY_RESPONSE":    ["IL1B", "IL6", "CXCL10", "CCL2", "IFNG"],
    "HALLMARK_TGF_BETA_SIGNALING":       ["TGFB1", "TGFB2", "SMAD3", "SMAD2"],
    "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION": ["VIM", "FN1", "ACTA2", "COL1A1"],
    "HALLMARK_HYPOXIA":                  ["HIF1A", "EPAS1", "VEGFA", "LDHA"],
    "HALLMARK_MYC_TARGETS_V1":           ["MKI67", "TOP2A", "CDK4", "PCNA"],
    "HALLMARK_INTERFERON_GAMMA_RESPONSE": ["STAT1", "IRF1", "CXCL10", "CXCL9", "GBP1"],
    "HALLMARK_OXIDATIVE_PHOSPHORYLATION": ["UQCRFS1", "COX6A1", "ATP5F1A", "NDUFS1"],
    "HALLMARK_FATTY_ACID_METABOLISM":    ["FASN", "ACACA", "ACSL1", "CPT1A", "SCD"],
    "HALLMARK_UNFOLDED_PROTEIN_RESPONSE": ["ATF6", "XBP1", "DDIT3", "HSPA5"],
}

_MIN_OVERLAP = 2   # minimum gene overlap to assign a type / hallmark annotation


def _assign_program_type(top_genes: list[str]) -> ProgramType:
    """Return the ProgramType with the highest gene set overlap."""
    gene_set = frozenset(g.upper() for g in top_genes)
    best_type: ProgramType = "unknown"
    best_overlap = 0
    for ptype, signature in _TYPE_GENE_SIGNATURES.items():
        overlap = len(gene_set & signature)
        if overlap > best_overlap:
            best_overlap = overlap
            best_type = ptype
    if best_overlap < _MIN_OVERLAP:
        return "unknown"
    return best_type


def _assign_hallmark_annotations(top_genes: list[str]) -> list[str]:
    """Return Hallmark set names that overlap sufficiently with top_genes."""
    gene_set = frozenset(g.upper() for g in top_genes)
    annotations: list[str] = []
    for hallmark, markers in _HALLMARK_KEYWORDS.items():
        overlap = len(gene_set & frozenset(m.upper() for m in markers))
        if overlap >= _MIN_OVERLAP:
            annotations.append(hallmark)
    return annotations


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def label_programs(
    nmf_result: dict,
    disease: str,
    cell_type: str,
    gwas_enrichment: dict[str, float] | None = None,
) -> list[LatentProgram]:
    """
    Convert a single nmf_result dict (from run_nmf_programs) into LatentProgram objects.

    Args:
        nmf_result:       Output dict from cnmf_runner.run_nmf_programs.
        disease:          Short disease key (IBD, CAD, …).
        cell_type:        Cell type this h5ad was built from.
        gwas_enrichment:  Optional {program_id: -log10(p)} from GWAS enrichment.

    Returns:
        List of LatentProgram objects, one per NMF component.
    """
    programs = nmf_result.get("programs", [])
    labeled: list[LatentProgram] = []

    for i, prog in enumerate(programs):
        raw_id    = prog.get("program_id", f"P{i:02d}")
        top_genes = prog.get("top_genes", [])
        n_cells   = prog.get("n_cells_expressing", None)

        program_id = f"{disease}_{cell_type.replace(' ', '_')}_{raw_id}"

        labeled.append(LatentProgram(
            program_id=program_id,
            disease=disease,
            cell_type=cell_type,
            program_index=i,
            top_genes=top_genes,
            program_type=_assign_program_type(top_genes),
            hallmark_annotations=_assign_hallmark_annotations(top_genes),
            gwas_enrichment_score=(
                gwas_enrichment.get(program_id)
                if gwas_enrichment else None
            ),
            n_cells_expressing=n_cells,
            data_source=nmf_result.get("source", "NMF"),
        ))

    return labeled


def label_programs_multi_celltype(
    nmf_results_by_ct: dict[str, dict],
    disease: str,
    gwas_enrichment_by_ct: dict[str, dict[str, float]] | None = None,
) -> dict[str, list[LatentProgram]]:
    """
    Label programs for multiple cell types.

    Args:
        nmf_results_by_ct:      {cell_type: nmf_result}
        disease:                Short disease key.
        gwas_enrichment_by_ct:  Optional {cell_type: {program_id: score}}.

    Returns:
        {cell_type: list[LatentProgram]}
    """
    result: dict[str, list[LatentProgram]] = {}
    for cell_type, nmf_result in nmf_results_by_ct.items():
        gwas_enrich = (gwas_enrichment_by_ct or {}).get(cell_type)
        result[cell_type] = label_programs(
            nmf_result=nmf_result,
            disease=disease,
            cell_type=cell_type,
            gwas_enrichment=gwas_enrich,
        )
    return result
