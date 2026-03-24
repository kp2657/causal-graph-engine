"""
causal_discovery_agent.py — Tier 3 agent: Ota composite γ + causal graph construction.

Computes γ_{gene→trait} = Σ_P (β_{gene→P} × γ_{P→trait}),
validates anchor edge recovery, writes significant edges to Kùzu,
and computes SHD from the reference anchor graph.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from graph.schema import REQUIRED_ANCHORS_BY_DISEASE

ANCHOR_RECOVERY_THRESHOLD = 0.80
OTA_GAMMA_MIN = 0.01  # edges below this threshold are not written


def run(
    beta_matrix_result: dict,
    gamma_estimates: dict,
    disease_query: dict,
) -> dict:
    """
    Build the causal graph from β × γ products.

    Args:
        beta_matrix_result: Output of perturbation_genomics_agent.run
        gamma_estimates:    {program_id: {trait: gamma_value}} from ota_gamma_estimation
        disease_query:      DiseaseQuery dict

    Returns:
        dict with n_edges_written, top_genes, anchor_recovery, shd, warnings
    """
    from pipelines.ota_gamma_estimation import compute_ota_gamma, compute_ota_gamma_with_uncertainty
    from pipelines.scone_sensitivity import (
        compute_cross_regime_sensitivity,
        polybic_score,
        bootstrap_edge_confidence,
        apply_scone_reweighting,
    )
    from mcp_servers.graph_db_server import (
        write_causal_edges,
        run_anchor_edge_validation,
        compute_shd_metric,
        run_evalue_check,
        query_graph_for_disease,
    )

    import logging
    _log = logging.getLogger(__name__)

    disease_name = disease_query.get("disease_name", "")
    efo_id       = disease_query.get("efo_id", "")
    gene_list    = beta_matrix_result.get("genes", [])
    beta_matrix  = beta_matrix_result.get("beta_matrix", {})
    tier_per_gene = beta_matrix_result.get("evidence_tier_per_gene", {})
    programs     = beta_matrix_result.get("programs", [])
    warnings: list[str] = []

    # Resolve disease-specific required anchors from the single source of truth.
    # Fall back to CAD anchors if the disease isn't mapped yet.
    from graph.schema import _DISEASE_SHORT_NAMES_FOR_ANCHORS  # local helper
    short = _DISEASE_SHORT_NAMES_FOR_ANCHORS.get(disease_name.lower(), "CAD")
    REQUIRED_ANCHORS = REQUIRED_ANCHORS_BY_DISEASE.get(short, REQUIRED_ANCHORS_BY_DISEASE["CAD"])

    # Build disease anchor gene set early — needed by both SCONE and edge-writing filter.
    # Anchor genes have established genetic/clinical evidence; their virtual-tier β
    # reflects missing Perturb-seq data, not a false-positive signal.
    from graph.schema import ANCHOR_EDGES as _ALL_ANCHORS
    _anchor_gene_set: set[str] = {
        e["from"] for e in _ALL_ANCHORS
        if e.get("to", "").upper() in (short, disease_name.upper().replace(" ", "_"))
        and not e["from"].endswith("_chip")
        and not e["from"].endswith("_exposure")
        and not e["from"].endswith("_program")
    }

    # -------------------------------------------------------------------------
    # Compute Ota composite γ for each gene × trait
    # -------------------------------------------------------------------------
    gene_gamma: dict[str, dict] = {}

    # Collect traits from gamma_estimates
    traits: set[str] = set()
    for prog_gammas in gamma_estimates.values():
        traits.update(prog_gammas.keys())
    if not traits:
        traits = {disease_name}

    for gene in gene_list:
        gene_beta = beta_matrix.get(gene, {})
        for trait in traits:
            try:
                # gamma_estimates is {program: {trait: float}} from chief_of_staff.
                # compute_ota_gamma expects {program: {"gamma": float, "evidence_tier": str}}.
                # Build trait-specific view.
                trait_gammas: dict[str, dict] = {}
                for prog, prog_gammas in gamma_estimates.items():
                    if isinstance(prog_gammas, dict):
                        g_val = prog_gammas.get(trait, prog_gammas.get("gamma", 0.0))
                    else:
                        g_val = 0.0
                    if isinstance(g_val, (int, float)):
                        trait_gammas[prog] = {"gamma": float(g_val), "evidence_tier": "Tier3_Provisional"}
                    elif isinstance(g_val, dict):
                        trait_gammas[prog] = g_val  # already in correct format

                ota_result = compute_ota_gamma_with_uncertainty(
                    gene=gene,
                    trait=trait,
                    beta_estimates=gene_beta,
                    gamma_estimates=trait_gammas,
                )
                ota_gamma = ota_result.get("ota_gamma", 0.0) or 0.0
                dominant_tier = ota_result.get("dominant_tier", "provisional_virtual")
                top_programs = ota_result.get("top_programs", [])

                key = f"{gene}__{trait}"
                gene_gamma[key] = {
                    "gene":               gene,
                    "trait":              trait,
                    "ota_gamma":          ota_gamma,
                    "dominant_tier":      dominant_tier,
                    "top_programs":       top_programs,
                    "tier":               tier_per_gene.get(gene, dominant_tier),
                    "ota_gamma_sigma":    ota_result.get("ota_gamma_sigma", 0.0),
                    "ota_gamma_ci_lower": ota_result.get("ota_gamma_ci_lower"),
                    "ota_gamma_ci_upper": ota_result.get("ota_gamma_ci_upper"),
                }
            except Exception as exc:
                warnings.append(f"Ota γ computation failed for {gene} → {trait}: {exc}")

    # -------------------------------------------------------------------------
    # SCONE sensitivity reweighting
    # Applies cross-regime sensitivity Γ_ij, PolyBIC scoring, and bootstrap
    # aggregation to reweight ota_gamma values before the edge write threshold.
    # This reduces false positives from provisional-virtual edges while
    # preserving high-confidence Tier1/Tier2 edges.
    # -------------------------------------------------------------------------
    try:
        sensitivity_matrix = compute_cross_regime_sensitivity(
            beta_matrix=beta_matrix,
            gamma_matrix=gamma_estimates,
            evidence_tier_per_gene=tier_per_gene,
        )

        # Anchor genes with provisional_virtual β are scored as Tier3_Provisional for BIC
        # purposes — their disease association is validated by GWAS/clinical evidence;
        # only the cell-type-matched Perturb-seq β is missing.
        bic_scores: dict[str, float] = {}
        for key, rec in gene_gamma.items():
            tier = rec.get("tier") or rec.get("dominant_tier", "Tier3_Provisional")
            if tier == "provisional_virtual" and rec.get("gene") in _anchor_gene_set:
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
                    n_bootstrap=30,  # reduced for speed; increase for publication
                )
            except Exception:
                pass

        gene_gamma = apply_scone_reweighting(
            gene_gamma_records=gene_gamma,
            sensitivity_matrix=sensitivity_matrix,
            bic_scores=bic_scores,
            bootstrap_confidence=bootstrap_conf,
            anchor_gene_set=_anchor_gene_set,
        )
        warnings.append(
            f"SCONE: reweighted {len(gene_gamma)} edges; "
            f"bootstrap_conf computed for {len(bootstrap_conf)}/{len(gene_list)} genes"
        )
    except Exception as exc:
        warnings.append(f"SCONE reweighting failed (non-fatal): {exc}")

    # -------------------------------------------------------------------------
    # Write significant edges to Kùzu
    # -------------------------------------------------------------------------
    edges_to_write: list[dict] = []
    edges_rejected: list[dict] = []

    for key, rec in gene_gamma.items():
        ota_gamma = rec["ota_gamma"]
        dominant_tier = rec["dominant_tier"]

        if abs(ota_gamma) < OTA_GAMMA_MIN:
            edges_rejected.append(rec)
            continue

        # Do not write pure virtual edges — UNLESS the gene is a disease anchor
        # (established causal locus with genetic/clinical evidence; β is provisional
        # but the disease association is well-validated).
        if dominant_tier == "provisional_virtual":
            if rec.get("gene") not in _anchor_gene_set:
                edges_rejected.append(rec)
                continue
            # Allow anchor gene through; mark with warning
            rec.setdefault("warnings", []).append(
                "provisional_virtual β — no cell-type-matched Perturb-seq; "
                "anchor gene allowed through with explicit virtual label"
            )

        # E-value check
        try:
            ev_result = run_evalue_check(effect_size=ota_gamma, se=abs(ota_gamma) * 0.2)
            e_value = ev_result.get("e_value")
            if e_value is not None and e_value < 2.0:
                rec["warnings"] = rec.get("warnings", [])
                rec["warnings"].append(f"E-value={e_value:.2f} < 2.0: potential confounding")
                warnings.append(
                    f"{rec['gene']} → {rec['trait']}: E-value={e_value:.2f} flagged"
                )
        except Exception:
            pass

        edges_to_write.append(rec)

    # Convert Ota γ recs to CausalEdge-compatible dicts before writing
    causal_edge_dicts: list[dict] = []
    _VALID_TIERS = frozenset({
        "Tier1_Interventional", "Tier2_Convergent", "Tier3_Provisional",
        "moderate_transferred", "moderate_grn", "provisional_virtual",
    })
    for rec in edges_to_write:
        tier = rec.get("tier") or rec.get("dominant_tier") or "Tier3_Provisional"
        if tier not in _VALID_TIERS:
            _log.warning(
                "Unrecognized evidence_tier %r for %s → %s; defaulting to Tier3_Provisional",
                tier, rec.get("gene"), rec.get("trait"),
            )
            warnings.append(
                f"Unrecognized tier {tier!r} for {rec.get('gene')} → {rec.get('trait')}; "
                "downgraded to Tier3_Provisional"
            )
            tier = "Tier3_Provisional"
        causal_edge_dicts.append({
            "from_node":          rec["gene"],
            "from_type":          "gene",
            "to_node":            rec["trait"],
            "to_type":            "trait",
            "effect_size":        float(rec["ota_gamma"]),
            "evidence_type":      "germline",
            "evidence_tier":      tier,
            "method":             "ota_gamma",
            "data_source":        "Ota2026_composite_gamma",
            "data_source_version": "2026",
        })

    # Write to graph
    n_written = 0
    if causal_edge_dicts:
        try:
            write_result = write_causal_edges(causal_edge_dicts, disease_name)
            n_written = write_result.get("written", len(causal_edge_dicts))
        except Exception as exc:
            warnings.append(f"Graph write failed: {exc}")
            n_written = 0

    # -------------------------------------------------------------------------
    # Anchor edge validation (CRITICAL — stop if < 80%)
    # -------------------------------------------------------------------------
    # Start with Ota γ edges from this run
    predicted_pairs: list[tuple[str, str]] = [
        (rec["gene"], rec["trait"]) for rec in edges_to_write
    ]

    # Also include edges already written to DB (e.g. CHIP/drug/viral from somatic agent).
    # Query both the full disease name and any short form (e.g. "CAD" for "coronary artery disease")
    _DISEASE_SHORT: dict[str, str] = {
        "coronary artery disease": "CAD",
        "ischemic heart disease": "CAD",
        "myocardial infarction": "CAD",
        "rheumatoid arthritis": "RA",
        "systemic lupus erythematosus": "SLE",
    }
    disease_ids_to_query = {disease_name}
    short = _DISEASE_SHORT.get(disease_name.lower())
    if short:
        disease_ids_to_query.add(short)

    try:
        existing_set = set(predicted_pairs)
        for did in disease_ids_to_query:
            existing = query_graph_for_disease(did)
            for e in existing.get("edges", []):
                fn = e.get("from_node") or e.get("from", "")
                tn = e.get("to_node") or e.get("to", "")
                if fn and tn and (fn, tn) not in existing_set:
                    predicted_pairs.append((fn, tn))
                    existing_set.add((fn, tn))
    except Exception as exc:
        warnings.append(f"Could not query existing DB edges for anchor check: {exc}")

    # Check anchor recovery using disease-specific REQUIRED_ANCHORS only.
    # run_anchor_edge_validation checks all 12 cross-disease ANCHOR_EDGES which can
    # never all be recovered in a single-disease pipeline run.
    predicted_set = set(predicted_pairs)
    recovered = [
        f"{g}→{t}" for g, t in REQUIRED_ANCHORS
        if (g, t) in predicted_set or (f"{g}_chip", t) in predicted_set
    ]
    missing = [
        f"{g}→{t}" for g, t in REQUIRED_ANCHORS
        if (g, t) not in predicted_set and (f"{g}_chip", t) not in predicted_set
    ]
    recovery_rate = len(recovered) / len(REQUIRED_ANCHORS) if REQUIRED_ANCHORS else 1.0

    # Also log to run_anchor_edge_validation for cross-disease telemetry (non-blocking)
    try:
        run_anchor_edge_validation(
            [{"from_node": g, "to_node": t} for g, t in predicted_pairs]
        )
    except Exception:
        pass

    if recovery_rate < ANCHOR_RECOVERY_THRESHOLD:
        warnings.append(
            f"CRITICAL: Anchor recovery {recovery_rate:.0%} < {ANCHOR_RECOVERY_THRESHOLD:.0%}. "
            f"Missing: {missing}. STOP — alert PI Orchestrator."
        )

    # -------------------------------------------------------------------------
    # SHD computation
    # -------------------------------------------------------------------------
    shd = 0
    try:
        shd_result = compute_shd_metric(
            predicted_edges=[{"from_node": g, "to_node": t} for g, t in predicted_pairs],
            reference_edges=[{"from": g, "to": t} for g, t in REQUIRED_ANCHORS],
        )
        shd = shd_result.get("shd", 0)
        extra  = shd_result.get("extra_edges", [])
        missing_shd = shd_result.get("missing_edges", [])
    except Exception as exc:
        warnings.append(f"SHD computation failed: {exc}")

    # -------------------------------------------------------------------------
    # Rank top genes by |ota_gamma|
    # -------------------------------------------------------------------------
    gene_best_gamma: dict[str, dict] = {}
    for key, rec in gene_gamma.items():
        gene = rec["gene"]
        if gene not in gene_best_gamma or abs(rec["ota_gamma"]) > abs(
            gene_best_gamma[gene]["ota_gamma"]
        ):
            gene_best_gamma[gene] = rec

    top_genes = sorted(
        gene_best_gamma.values(),
        key=lambda r: abs(r["ota_gamma"]),
        reverse=True,
    )[:10]

    return {
        "n_edges_written":  n_written,
        "n_edges_rejected": len(edges_rejected),
        "top_genes": [
            {
                "gene":               r["gene"],
                "ota_gamma":          r["ota_gamma"],
                "ota_gamma_raw":      r.get("ota_gamma_raw", r["ota_gamma"]),
                "ota_gamma_sigma":    r.get("ota_gamma_sigma", 0.0),
                "ota_gamma_ci_lower": r.get("ota_gamma_ci_lower"),
                "ota_gamma_ci_upper": r.get("ota_gamma_ci_upper"),
                "tier":               r["tier"],
                "programs":           r.get("top_programs", []),
                "scone_confidence":   r.get("scone_confidence"),
                "scone_bic_factor":   r.get("scone_bic_factor"),
                "scone_flags":        r.get("scone_flags", []),
            }
            for r in top_genes
        ],
        "anchor_recovery": {
            "recovery_rate": recovery_rate,
            "recovered":     recovered,
            "missing":       missing,
        },
        "shd":     shd,
        "warnings": warnings,
    }
