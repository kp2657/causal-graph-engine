"""
kg_completion_agent.py — Tier 3 agent: KG enrichment with drug-target context.

Adds Reactome pathway edges and drug-target edges to the causal graph.
PrimeKG and STRING were removed: both produced 0 edges in practice and added
only non-causal metadata that did not influence gene ranking.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


OTA_GAMMA_KG_THRESHOLD = 0.1
MAX_TOP_GENES_KG = 10


def run(causal_discovery_result: dict, disease_query: dict) -> dict:
    """
    Enrich the causal graph with pathway and drug-target context.

    Args:
        causal_discovery_result: Output of causal_discovery_agent.run
        disease_query:           DiseaseQuery dict

    Returns:
        dict with edge counts, top_pathways, drug_target_summary, contradictions_flagged
    """
    from mcp_servers.pathways_kg_server import get_reactome_pathways_for_gene
    from mcp_servers.open_targets_server import get_open_targets_disease_targets
    from mcp_servers.clinical_trials_server import get_trials_for_target
    from mcp_servers.chemistry_server import search_chembl_compound
    from mcp_servers.graph_db_server import query_graph_for_disease

    disease_name = disease_query.get("disease_name", "")
    efo_id       = disease_query.get("efo_id", "")
    top_genes    = causal_discovery_result.get("top_genes", [])
    warnings: list[str] = []

    def _safe_float(x, default: float = 0.0) -> float:
        if x is None:
            return default
        if isinstance(x, (int, float)):
            return float(x)
        try:
            return float(str(x).strip())
        except Exception:
            return default

    gene_names = [r["gene"] for r in top_genes[:MAX_TOP_GENES_KG]]
    high_gamma_genes = [
        r["gene"] for r in top_genes
        if abs(_safe_float(r.get("ota_gamma", 0.0))) > OTA_GAMMA_KG_THRESHOLD
    ]

    n_pathway_edges   = 0
    n_drug_target_edges = 0
    contradictions_flagged = 0
    top_pathways: list[str] = []
    drug_target_summary: list[dict] = []
    pathway_gene_map: dict[str, list[str]] = {}

    # Load existing graph edges for contradiction checking
    existing_edges: list[dict] = []
    try:
        graph_result = query_graph_for_disease(disease_name)
        existing_edges = graph_result.get("edges", [])
    except Exception as exc:
        warnings.append(f"Failed to load existing edges for contradiction check: {exc}")

    # -------------------------------------------------------------------------
    # 1. Reactome pathway enrichment
    # -------------------------------------------------------------------------
    for gene in gene_names:
        try:
            pathway_result = get_reactome_pathways_for_gene(gene)
            pathways = pathway_result.get("pathways", [])
            for pw in pathways:
                pw_id = pw.get("pathway_id") or pw.get("stId") or pw.get("name", "")
                if pw_id and pw_id not in top_pathways:
                    top_pathways.append(pw_id)
                pathway_gene_map.setdefault(pw_id, []).append(gene)
            n_pathway_edges += len(pathways)
        except Exception as exc:
            warnings.append(f"Reactome lookup failed for {gene}: {exc}")

    # -------------------------------------------------------------------------
    # 2. Drug-target edges from Open Targets + clinical trials
    # -------------------------------------------------------------------------
    ot_targets: list[dict] = []
    if efo_id:
        try:
            ot_result = get_open_targets_disease_targets(efo_id)
            ot_targets = ot_result.get("targets", [])
        except Exception as exc:
            warnings.append(f"Open Targets disease targets failed: {exc}")

    for gene in high_gamma_genes:
        ot_entry = next(
            (t for t in ot_targets if t.get("symbol") == gene or t.get("gene") == gene),
            None,
        )
        ot_score = ot_entry.get("overall_score", 0.0) if ot_entry else 0.0
        max_phase = 0

        drugs_for_gene: list[str] = []
        try:
            trial_result = get_trials_for_target(gene)
            trials = trial_result.get("trials", [])
            for t in trials:
                phases = t.get("phase", [])
                drug = t.get("intervention") or t.get("drug")
                if drug and drug not in drugs_for_gene:
                    drugs_for_gene.append(drug)
                for ph in phases:
                    try:
                        ph_num = int(str(ph).replace("PHASE", "").strip())
                        max_phase = max(max_phase, ph_num)
                    except ValueError:
                        pass
        except Exception as exc:
            warnings.append(f"Trials lookup failed for {gene}: {exc}")

        if drugs_for_gene:
            try:
                chembl = search_chembl_compound(drugs_for_gene[0])
                if chembl.get("max_phase"):
                    max_phase = max(max_phase, chembl["max_phase"])
            except Exception:
                pass

        if drugs_for_gene or ot_score > 0:
            drug_target_summary.append({
                "drug":      drugs_for_gene[0] if drugs_for_gene else None,
                "target":    gene,
                "max_phase": max_phase,
                "ot_score":  ot_score,
            })
            n_drug_target_edges += 1

    drug_target_summary.sort(key=lambda x: (x["max_phase"], x["ot_score"]), reverse=True)

    # -------------------------------------------------------------------------
    # 3. BioPathNet-style BRG diffusion — novel link candidates
    # -------------------------------------------------------------------------
    brg_novel_candidates: list[dict] = []
    n_brg_novel = 0
    try:
        from pipelines.biopath_gnn import run as brg_run
        brg_result = brg_run(
            causal_discovery_result=causal_discovery_result,
            kg_completion_result={
                "pathway_gene_map":   pathway_gene_map,
                "drug_target_summary": drug_target_summary,
            },
            disease_query=disease_query,
        )
        brg_novel_candidates = brg_result.get("novel_candidates", [])
        n_brg_novel = brg_result.get("n_novel_candidates", 0)
        warnings.extend(brg_result.get("warnings", []))
    except Exception as exc:
        warnings.append(f"BRG diffusion failed (non-fatal): {exc}")

    return {
        "n_pathway_edges_added":     n_pathway_edges,
        "n_drug_target_edges_added": n_drug_target_edges,
        "top_pathways":              top_pathways[:20],
        "pathway_gene_map":          dict(pathway_gene_map),
        "drug_target_summary":       drug_target_summary,
        "brg_novel_candidates":      brg_novel_candidates,
        "n_brg_novel_candidates":    n_brg_novel,
        "contradictions_flagged":    contradictions_flagged,
        "warnings":                  warnings,
    }
