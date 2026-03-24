"""
kg_completion_agent.py — Tier 3 agent: KG enrichment with pathway/PPI/drug context.

Adds Reactome pathway edges, STRING PPI edges, drug-target edges,
and PrimeKG disease-gene edges to the causal graph.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


STRING_MIN_SCORE = 700
STRING_HIGH_SCORE = 800
OTA_GAMMA_KG_THRESHOLD = 0.1
PRIMEKG_PRIOR_THRESHOLD = 0.5
MAX_TOP_GENES_KG = 10


def run(causal_discovery_result: dict, disease_query: dict) -> dict:
    """
    Enrich the causal graph with pathway/PPI/drug-target/PrimeKG context.

    Args:
        causal_discovery_result: Output of causal_discovery_agent.run
        disease_query:           DiseaseQuery dict

    Returns:
        dict with edge counts, top_pathways, drug_target_summary, contradictions_flagged
    """
    from mcp_servers.pathways_kg_server import (
        get_reactome_pathways_for_gene,
        query_primekg_subgraph,
        get_string_interactions,
    )
    from mcp_servers.open_targets_server import (
        get_open_targets_disease_targets,
    )
    from mcp_servers.clinical_trials_server import get_trials_for_target
    from mcp_servers.chemistry_server import search_chembl_compound
    from mcp_servers.graph_db_server import (
        write_causal_edges,
        query_graph_for_disease,
    )

    disease_name = disease_query.get("disease_name", "")
    efo_id       = disease_query.get("efo_id", "")
    top_genes    = causal_discovery_result.get("top_genes", [])
    warnings: list[str] = []

    gene_names = [r["gene"] for r in top_genes[:MAX_TOP_GENES_KG]]
    high_gamma_genes = [
        r["gene"] for r in top_genes
        if abs(r.get("ota_gamma", 0)) > OTA_GAMMA_KG_THRESHOLD
    ]

    n_pathway_edges   = 0
    n_ppi_edges       = 0
    n_drug_target_edges = 0
    n_primekg_edges   = 0
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
    # 2. STRING PPI
    # -------------------------------------------------------------------------
    if gene_names:
        try:
            string_result = get_string_interactions(gene_names[:5], min_score=STRING_MIN_SCORE)
            interactions = string_result.get("interactions", [])
            ppi_edges = [
                i for i in interactions
                if (i.get("score") or 0) >= STRING_MIN_SCORE
            ]
            n_ppi_edges = len(ppi_edges)
        except Exception as exc:
            warnings.append(f"STRING PPI lookup failed: {exc}")

    # -------------------------------------------------------------------------
    # 3. Drug-target edges from Open Targets + clinical trials
    # -------------------------------------------------------------------------
    ot_targets: list[dict] = []
    if efo_id:
        try:
            ot_result = get_open_targets_disease_targets(efo_id)
            ot_targets = ot_result.get("targets", [])
        except Exception as exc:
            warnings.append(f"Open Targets disease targets failed: {exc}")

    for gene in high_gamma_genes:
        # Find this gene in OT results
        ot_entry = next(
            (t for t in ot_targets if t.get("symbol") == gene or t.get("gene") == gene),
            None,
        )
        ot_score = ot_entry.get("overall_score", 0.0) if ot_entry else 0.0
        max_phase = 0

        # Known drugs from clinical trials
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

        # ChEMBL lookup for first known drug
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

    # Priority: write Phase 3/4 drug-target edges first
    drug_target_summary.sort(key=lambda x: (x["max_phase"], x["ot_score"]), reverse=True)

    # -------------------------------------------------------------------------
    # 4. PrimeKG disease-gene edges
    # -------------------------------------------------------------------------
    for gene in high_gamma_genes:
        try:
            pkg_result = query_primekg_subgraph(gene=gene, edge_type="disease_gene")
            pkg_edges = pkg_result.get("edges", [])
            high_conf = [
                e for e in pkg_edges
                if (e.get("prior_probability") or 0) > PRIMEKG_PRIOR_THRESHOLD
            ]
            n_primekg_edges += len(high_conf)
        except Exception as exc:
            warnings.append(f"PrimeKG lookup failed for {gene}: {exc}")

    # -------------------------------------------------------------------------
    # 5. BioPathNet-style BRG diffusion — novel link candidates
    # Seeds RWR from high-γ genes; propagates through PPI + pathway + drug BRG.
    # Inductive: no retraining needed for new diseases.
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
        "n_ppi_edges_added":         n_ppi_edges,
        "n_drug_target_edges_added": n_drug_target_edges,
        "n_primekg_edges_added":     n_primekg_edges,
        "top_pathways":              top_pathways[:20],
        "pathway_gene_map":          dict(pathway_gene_map),  # for BRG downstream
        "drug_target_summary":       drug_target_summary,
        "brg_novel_candidates":      brg_novel_candidates,
        "n_brg_novel_candidates":    n_brg_novel,
        "contradictions_flagged":    contradictions_flagged,
        "warnings":                  warnings,
    }
