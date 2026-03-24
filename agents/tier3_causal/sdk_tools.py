"""
sdk_tools.py — SDK-callable tool wrappers for causal_discovery_agent.

These functions are exposed as tools when causal_discovery_agent runs in SDK mode.
Each wraps a computation step so Claude can call them sequentially, adding reasoning
between calls, rather than executing a fixed script.

Tool design principle: each tool does one well-defined computation and returns
enough context for Claude to make the next decision.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def compute_ota_gammas(
    beta_matrix_result: dict,
    gamma_estimates: dict,
    disease_query: dict,
) -> dict:
    """
    Run the full Ota γ computation + SCONE reweighting for all genes.

    Returns gene_gamma_records (one per gene×trait), anchor_gene_set,
    required_anchors for this disease, and any computation warnings.

    Claude should examine these records and decide which edges to include.
    """
    from pipelines.ota_gamma_estimation import compute_ota_gamma, compute_ota_gamma_with_uncertainty
    from pipelines.scone_sensitivity import (
        compute_cross_regime_sensitivity,
        polybic_score,
        bootstrap_edge_confidence,
        apply_scone_reweighting,
    )
    from graph.schema import (
        ANCHOR_EDGES as _ALL_ANCHORS,
        REQUIRED_ANCHORS_BY_DISEASE,
        _DISEASE_SHORT_NAMES_FOR_ANCHORS,
    )

    disease_name = disease_query.get("disease_name", "")
    gene_list    = beta_matrix_result.get("genes", [])
    beta_matrix  = beta_matrix_result.get("beta_matrix", {})
    tier_per_gene = beta_matrix_result.get("evidence_tier_per_gene", {})
    warnings: list[str] = []

    short = _DISEASE_SHORT_NAMES_FOR_ANCHORS.get(disease_name.lower(), "CAD")
    required_anchors = REQUIRED_ANCHORS_BY_DISEASE.get(short, REQUIRED_ANCHORS_BY_DISEASE["CAD"])

    anchor_gene_set: set[str] = {
        e["from"] for e in _ALL_ANCHORS
        if e.get("to", "").upper() in (short, disease_name.upper().replace(" ", "_"))
        and not e["from"].endswith("_chip")
        and not e["from"].endswith("_exposure")
        and not e["from"].endswith("_program")
    }

    # Collect traits
    traits: set[str] = set()
    for prog_gammas in gamma_estimates.values():
        traits.update(prog_gammas.keys())
    if not traits:
        traits = {disease_name}

    # Compute Ota γ for each gene × trait
    gene_gamma: dict[str, dict] = {}
    for gene in gene_list:
        gene_beta = beta_matrix.get(gene, {})
        for trait in traits:
            try:
                trait_gammas: dict[str, dict] = {}
                for prog, prog_gammas in gamma_estimates.items():
                    if isinstance(prog_gammas, dict):
                        g_val = prog_gammas.get(trait, prog_gammas.get("gamma", 0.0))
                    else:
                        g_val = 0.0
                    if isinstance(g_val, (int, float)):
                        trait_gammas[prog] = {"gamma": float(g_val), "evidence_tier": "Tier3_Provisional"}
                    elif isinstance(g_val, dict):
                        trait_gammas[prog] = g_val

                ota_result = compute_ota_gamma_with_uncertainty(
                    gene=gene, trait=trait,
                    beta_estimates=gene_beta,
                    gamma_estimates=trait_gammas,
                )
                ota_gamma    = ota_result.get("ota_gamma", 0.0) or 0.0
                dominant_tier = ota_result.get("dominant_tier", "provisional_virtual")

                key = f"{gene}__{trait}"
                gene_gamma[key] = {
                    "gene":               gene,
                    "trait":              trait,
                    "ota_gamma":          ota_gamma,
                    "dominant_tier":      dominant_tier,
                    "top_programs":       ota_result.get("top_programs", []),
                    "tier":               tier_per_gene.get(gene, dominant_tier),
                    "ota_gamma_sigma":    ota_result.get("ota_gamma_sigma", 0.0),
                    "ota_gamma_ci_lower": ota_result.get("ota_gamma_ci_lower"),
                    "ota_gamma_ci_upper": ota_result.get("ota_gamma_ci_upper"),
                    "is_anchor":          gene in anchor_gene_set,
                }
            except Exception as exc:
                warnings.append(f"Ota γ failed for {gene} → {trait}: {exc}")

    # SCONE reweighting
    try:
        sensitivity_matrix = compute_cross_regime_sensitivity(
            beta_matrix=beta_matrix,
            gamma_matrix=gamma_estimates,
            evidence_tier_per_gene=tier_per_gene,
        )
        bic_scores: dict[str, float] = {}
        for key, rec in gene_gamma.items():
            tier = rec.get("tier") or rec.get("dominant_tier", "Tier3_Provisional")
            if tier == "provisional_virtual" and rec.get("gene") in anchor_gene_set:
                tier = "Tier3_Provisional"
            bic_scores[key] = polybic_score(ota_gamma=rec["ota_gamma"], evidence_tier=tier)

        bootstrap_conf: dict[str, dict[str, float]] = {}
        for gene in gene_list:
            try:
                bootstrap_conf[gene] = bootstrap_edge_confidence(
                    gene=gene,
                    beta_matrix_row=beta_matrix.get(gene, {}),
                    gamma_matrix=gamma_estimates,
                    ota_gamma_fn=compute_ota_gamma,
                    n_bootstrap=30,
                )
            except Exception:
                pass

        gene_gamma = apply_scone_reweighting(
            gene_gamma_records=gene_gamma,
            sensitivity_matrix=sensitivity_matrix,
            bic_scores=bic_scores,
            bootstrap_confidence=bootstrap_conf,
            anchor_gene_set=anchor_gene_set,
        )
        warnings.append(
            f"SCONE: reweighted {len(gene_gamma)} edge candidates; "
            f"bootstrap computed for {len(bootstrap_conf)}/{len(gene_list)} genes"
        )
    except Exception as exc:
        warnings.append(f"SCONE reweighting failed (non-fatal): {exc}")

    return {
        "gene_gamma_records": list(gene_gamma.values()),
        "anchor_gene_set":    sorted(anchor_gene_set),
        "required_anchors":   [{"gene": g, "trait": t} for g, t in required_anchors],
        "n_genes":            len(gene_list),
        "n_traits":           len(traits),
        "warnings":           warnings,
    }


def check_anchor_recovery(
    written_edges: list[dict],
    disease_query: dict,
) -> dict:
    """
    Check what fraction of required disease anchor edges are present in written_edges.

    written_edges should be CausalEdge dicts with from_node / to_node fields,
    or plain (gene, trait) dicts. Also queries the Kùzu DB for previously-written
    edges so somatic/CHIP edges from Tier 1 count toward anchor recovery.

    Returns recovery_rate, recovered list, missing list, and required_anchors.
    """
    from graph.schema import REQUIRED_ANCHORS_BY_DISEASE, _DISEASE_SHORT_NAMES_FOR_ANCHORS
    from mcp_servers.graph_db_server import query_graph_for_disease

    disease_name = disease_query.get("disease_name", "")
    short = _DISEASE_SHORT_NAMES_FOR_ANCHORS.get(disease_name.lower(), "CAD")
    required_anchors = REQUIRED_ANCHORS_BY_DISEASE.get(short, REQUIRED_ANCHORS_BY_DISEASE["CAD"])

    # Normalise written_edges to (from_node, to_node) pairs
    predicted_set: set[tuple[str, str]] = set()
    for e in written_edges:
        fn = e.get("from_node") or e.get("from", "") or e.get("gene", "")
        tn = e.get("to_node")   or e.get("to", "")   or e.get("trait", "")
        if fn and tn:
            predicted_set.add((fn, tn))

    # Also include edges already in the DB (e.g. somatic from Tier 1)
    _SHORT_MAP: dict[str, str] = {
        "coronary artery disease": "CAD", "ischemic heart disease": "CAD",
        "myocardial infarction": "CAD", "rheumatoid arthritis": "RA",
    }
    disease_ids = {disease_name}
    ds = _SHORT_MAP.get(disease_name.lower())
    if ds:
        disease_ids.add(ds)

    try:
        for did in disease_ids:
            existing = query_graph_for_disease(did)
            for e in existing.get("edges", []):
                fn = e.get("from_node") or e.get("from", "")
                tn = e.get("to_node")   or e.get("to", "")
                if fn and tn:
                    predicted_set.add((fn, tn))
    except Exception:
        pass

    recovered = [
        f"{g}→{t}" for g, t in required_anchors
        if (g, t) in predicted_set or (f"{g}_chip", t) in predicted_set
    ]
    missing = [
        f"{g}→{t}" for g, t in required_anchors
        if (g, t) not in predicted_set and (f"{g}_chip", t) not in predicted_set
    ]
    recovery_rate = len(recovered) / len(required_anchors) if required_anchors else 1.0

    return {
        "recovery_rate":    recovery_rate,
        "recovered":        recovered,
        "missing":          missing,
        "required_anchors": [{"gene": g, "trait": t} for g, t in required_anchors],
        "n_required":       len(required_anchors),
        "n_recovered":      len(recovered),
    }


def compute_shd(
    predicted_edges: list[dict],
    disease_query: dict,
) -> dict:
    """
    Compute Structural Hamming Distance between predicted edges and reference anchors.

    predicted_edges: list of {from_node, to_node} dicts.
    Returns shd, extra_edges (in predicted but not reference), missing_edges (in reference but not predicted).
    """
    from graph.schema import REQUIRED_ANCHORS_BY_DISEASE, _DISEASE_SHORT_NAMES_FOR_ANCHORS
    from mcp_servers.graph_db_server import compute_shd_metric

    disease_name = disease_query.get("disease_name", "")
    short = _DISEASE_SHORT_NAMES_FOR_ANCHORS.get(disease_name.lower(), "CAD")
    required_anchors = REQUIRED_ANCHORS_BY_DISEASE.get(short, REQUIRED_ANCHORS_BY_DISEASE["CAD"])

    try:
        result = compute_shd_metric(
            predicted_edges=predicted_edges,
            reference_edges=[{"from": g, "to": t} for g, t in required_anchors],
        )
        return {
            "shd":           result.get("shd", 0),
            "extra_edges":   result.get("extra_edges", []),
            "missing_edges": result.get("missing_edges", []),
        }
    except Exception as exc:
        return {"shd": -1, "extra_edges": [], "missing_edges": [], "error": str(exc)}
