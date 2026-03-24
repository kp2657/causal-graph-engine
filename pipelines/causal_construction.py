"""
causal_construction.py — Causal graph construction pipeline.

Orchestrates:
  1. SCONE normalization of single-cell counts
  2. cNMF program extraction
  3. inspre causal structure learning (gene-gene + gene-program)
  4. BioPathNet pathway context
  5. Edge writing to Kùzu graph via graph_db_server

STUB — requires:
  - GEO GSE246756 h5ad download (cnmf_programs.py)
  - inspre package: pip install inspre
  - Full β matrix (ota_beta_estimation.py)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def run_causal_construction_pipeline(
    disease: str = "CAD",
    genes: list[str] | None = None,
    use_provisional_betas: bool = True,
) -> dict:
    """
    Run the full causal graph construction pipeline for a disease.

    Steps:
      1. Get β matrix (from Replogle 2022 or provisional)
      2. Get γ matrix (from GWAS S-LDSC + provisional)
      3. Compute Ota composite γ for each gene → trait
      4. Write significant edges to Kùzu graph

    Args:
        disease:               Disease name, e.g. "CAD"
        genes:                 Optional gene list; defaults to CAD targets
        use_provisional_betas: Use provisional β if Perturb-seq not downloaded

    Returns:
        {
            "n_edges_written": int,
            "n_edges_rejected": int,
            "top_targets": list of top-ranked gene→disease edges,
        }
    """
    from pipelines.ota_beta_estimation import estimate_cad_target_betas
    from pipelines.ota_gamma_estimation import estimate_cad_gammas, compute_ota_gamma
    from mcp_servers.burden_perturb_server import get_cnmf_program_info

    if genes is None:
        genes = ["PCSK9", "LDLR", "HMGCR", "DNMT3A", "TET2", "IL6R"]

    programs_info = get_cnmf_program_info()
    programs = programs_info["programs"]

    # Build β matrix
    beta_matrix_obj = estimate_cad_target_betas()
    # Build γ matrix
    gamma_result = estimate_cad_gammas()
    gamma_matrix = gamma_result["matrix"]

    # Compute Ota γ for each gene → trait
    trait = disease
    ota_results = []
    for gene in genes:
        beta_estimates = {}
        for prog in programs:
            beta_val = beta_matrix_obj.beta_matrix.get(gene, {}).get(prog)
            beta_estimates[prog] = {
                "beta": beta_val,
                "evidence_tier": "Tier1_Interventional" if beta_val is not None else "provisional_virtual",
            }

        gamma_estimates = {prog: gamma_matrix.get(prog, {}).get(trait, {}) for prog in programs}

        ota = compute_ota_gamma(gene, trait, beta_estimates, gamma_estimates)
        ota_results.append(ota)

    # Sort by absolute Ota γ
    ota_results.sort(key=lambda x: abs(x["ota_gamma"]), reverse=True)

    # Write to graph (non-virtual edges with |γ| > 0.01)
    edges_to_write = []
    for ota in ota_results:
        if abs(ota["ota_gamma"]) > 0.01 and ota["dominant_tier"] != "provisional_virtual":
            edges_to_write.append({
                "from_node":   ota["gene"],
                "to_node":     trait,
                "effect_size": ota["ota_gamma"],
                "evidence_tier": ota["dominant_tier"],
                "data_source": "ota_causal_construction",
                "se":          None,
                "ci_lower":    None,
                "ci_upper":    None,
            })

    # Write to graph
    written = 0
    rejected = 0
    errors = []
    if edges_to_write:
        from mcp_servers.graph_db_server import write_causal_edges
        result = write_causal_edges(edges_to_write, disease)
        written = result.get("written", 0)
        rejected = result.get("rejected", 0)
        errors = result.get("errors", [])

    return {
        "disease":          disease,
        "n_genes":          len(genes),
        "n_edges_written":  written,
        "n_edges_rejected": rejected,
        "errors":           errors[:5],
        "top_targets":      ota_results[:5],
        "note":             "Provisional construction. Re-run after quantitative β + GWAS data available.",
    }
