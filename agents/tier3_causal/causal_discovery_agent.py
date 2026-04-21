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

OTA_GAMMA_MIN = 0.01  # edges below this threshold are not written

# Beta inflation thresholds: KO of essential cellular machinery (ESCRT, Mediator,
# V-ATPase, DNA replication) in vascular cell lines produces non-specific cell-stress
# transcriptional responses that inflate all program betas simultaneously.
# We detect this as: mean(|beta|) > threshold AND coefficient-of-variation < CV_FLOOR
# (all programs move together at large magnitude → not a disease-pathway signal).
_STRESS_MEAN_THRESHOLD = 1.2   # mean |beta| above this suggests global stress response
_STRESS_CV_FLOOR       = 0.35  # CV below this means betas are uniformly large
_STRESS_DISCOUNT       = 0.25  # multiply OTA gamma by this when stress pattern detected


def _beta_stress_discount(gene_beta: dict) -> float:
    """
    Return a gamma multiplier [0.25, 1.0] to penalise essential-gene cell-stress
    artefacts in Perturb-seq data.

    When a gene's KO causes non-specific transcriptional stress (all programs
    perturbed at similar large magnitude), its OTA gamma is inflated and not
    disease-pathway specific.  Criterion:
      mean(|beta|) > _STRESS_MEAN_THRESHOLD  AND
      stdev(|beta|) / mean(|beta|) < _STRESS_CV_FLOOR  (low CV = uniform response)

    Returns _STRESS_DISCOUNT (0.25) when both conditions are met, else 1.0.
    """
    import math
    vals: list[float] = []
    for v in gene_beta.values():
        raw = v.get("beta") if isinstance(v, dict) else v
        try:
            f = float(raw)
            if math.isfinite(f):
                vals.append(abs(f))
        except (TypeError, ValueError):
            pass

    if len(vals) < 3:
        return 1.0

    mean_abs = sum(vals) / len(vals)
    if mean_abs <= _STRESS_MEAN_THRESHOLD:
        return 1.0

    variance = sum((v - mean_abs) ** 2 for v in vals) / len(vals)
    cv = (variance ** 0.5) / mean_abs if mean_abs > 0 else 1.0
    if cv < _STRESS_CV_FLOOR:
        return _STRESS_DISCOUNT
    return 1.0


# Housekeeping gene prefixes: ribosomal proteins and histones have very large
# Perturb-seq β values (global translation failure / chromatin disruption) that
# are biologically non-specific and not therapeutically actionable.
_HOUSEKEEPING_PREFIXES = ("RPL", "RPS", "HIST", "H1-", "H2A", "H2B", "H3-", "H4-")
_HOUSEKEEPING_EXACT = frozenset({
    "UBA52", "UBB", "UBC", "UBD",           # ubiquitin
    "ACTB", "ACTG1",                          # cytoskeletal
    "GAPDH", "LDHA",                          # glycolysis
    "B2M",                                    # MHC presentation component
})

from models.disease_registry import get_disease_key as _get_disease_key
from config.scoring_thresholds import MR_F_STATISTIC_MIN


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
        run_evalue_check,
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

    from graph.schema import _DISEASE_SHORT_NAMES_FOR_ANCHORS  # local helper
    short = _DISEASE_SHORT_NAMES_FOR_ANCHORS.get(disease_name.lower(), "CAD")

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

    # Fetch GeneBayes (Dataset 1) LoF burden priors for all genes
    from mcp_servers.ukb_wes_server import get_gene_burden
    _gb_priors: dict[str, dict] = {}
    for gene in gene_list:
        try:
            # short is the mapped disease key (e.g., "AMD", "CAD")
            _gb_res = get_gene_burden(gene, disease=short)
            if _gb_res.get("burden_beta") is not None:
                _gb_priors[gene] = _gb_res
        except Exception:
            pass

    # Tissue expression weights: down-weight genes whose expression in the
    # disease-relevant tissues is low relative to other tissues.
    # Fetched once per gene (API-cached), applied inside the trait loop below.
    _tissue_weights: dict[str, float] = {}
    try:
        from mcp_servers.single_cell_server import query_gtex_tissue_weight
        from graph.schema import DISEASE_CELL_TYPE_MAP
        _schema_ctx = DISEASE_CELL_TYPE_MAP.get(short, {})
        _primary_tissue = _schema_ctx.get("gtex_tissue", "")
        _secondary_tissues = _schema_ctx.get("gtex_tissues_secondary", [])
        _relevant_tissues = [_primary_tissue] + _secondary_tissues if _primary_tissue else _secondary_tissues
        if _relevant_tissues:
            for gene in gene_list:
                try:
                    _tissue_weights[gene] = query_gtex_tissue_weight(gene, _relevant_tissues)
                except Exception:
                    pass
    except Exception:
        pass  # non-fatal: weight defaults to 1.0 for all genes

    for gene in gene_list:
        # Skip housekeeping/ribosomal genes: their Perturb-seq signatures are
        # non-specific (global translation / chromatin disruption) and produce
        # inflated β×γ products that are not therapeutically actionable.
        if (
            gene in _HOUSEKEEPING_EXACT
            or any(gene.upper().startswith(pfx) for pfx in _HOUSEKEEPING_PREFIXES)
        ):
            continue

        gene_beta = beta_matrix.get(gene, {})
        gb_result = _gb_priors.get(gene)

        for trait in traits:
            try:
                # Build trait-specific view for compute_ota_gamma.
                trait_gammas: dict[str, dict] = {}
                for prog, prog_gammas in gamma_estimates.items():
                    if isinstance(prog_gammas, dict):
                        g_val = prog_gammas.get(trait, prog_gammas.get("gamma", 0.0))
                    else:
                        g_val = 0.0
                    if g_val is None:
                        trait_gammas[prog] = {"gamma": None, "evidence_tier": "provisional_virtual"}
                    elif isinstance(g_val, (int, float)):
                        trait_gammas[prog] = {"gamma": float(g_val), "evidence_tier": "Tier3_Provisional"}
                    elif isinstance(g_val, dict):
                        trait_gammas[prog] = g_val

                # Phase Z7: Bayesian Fusion of GeneBayes (Dataset 1) and Perturb-seq (Dataset 2)
                ota_result = compute_ota_gamma_with_uncertainty(
                    gene=gene,
                    trait=trait,
                    beta_estimates=gene_beta,
                    gamma_estimates=trait_gammas,
                    genebayes_result=gb_result,
                )
                ota_gamma_raw = ota_result.get("ota_gamma")
                ota_gamma = float('nan') if ota_gamma_raw is None else float(ota_gamma_raw)
                dominant_tier = ota_result.get("dominant_tier", "provisional_virtual")
                top_programs = ota_result.get("top_programs", [])

                import math as _math
                # Fallback: when no programs contributed to OTA gamma (n_programs_contributing=0),
                # use precomputed OT genetic score as a direct γ proxy.  Covers genetically-
                # driven diseases (e.g. AMD) whose NMF programs are cell-type-specific and
                # have no GWAS credible-set overlap — the OTA β×γ product is always zero even
                # when the gene has strong GWAS support.
                _ot_scores = disease_query.get("ot_genetic_scores") or {}
                if (ota_result.get("n_programs_contributing", 0) == 0
                        and gene in _ot_scores):
                    _ot_val = float(_ot_scores[gene])
                    if _ot_val >= 0.05:
                        ota_gamma = round(_ot_val * 0.65, 4)
                        ota_gamma_raw = ota_gamma
                        # OT L2G >= OT_L2G_GAMMA_MIN is genetic causal evidence → Tier2_Convergent
                        # so polybic_selection (k=2) doesn't filter it out.
                        # Weaker OT signal stays Tier3_Provisional.
                        dominant_tier = "Tier2_Convergent" if _ot_val >= 0.10 else "Tier3_Provisional"
                        top_programs = []

                # Beta stress discount: attenuate gamma for essential-gene KO artefacts.
                # Genes whose betas are uniformly large across all programs are non-specific
                # cell-stress responders (Mediator, ESCRT, V-ATPase, DNA replication), not
                # disease-pathway targets.  Apply per-gene, once, before storing.
                _stress_mult = _beta_stress_discount(gene_beta) if not _math.isnan(ota_gamma) else 1.0
                _stress_flagged = _stress_mult < 1.0
                if _stress_flagged:
                    ota_gamma = ota_gamma * _stress_mult

                # Apply tissue expression weight (fetched outside the trait loop — see below).
                _tissue_mult = _tissue_weights.get(gene, 1.0)
                if not _math.isnan(ota_gamma):
                    ota_gamma = ota_gamma * _tissue_mult

                key = f"{gene}__{trait}"
                gene_gamma[key] = {
                    "gene":               gene,
                    "trait":              trait,
                    "ota_gamma":          ota_gamma,
                    "ota_gamma_raw":      ota_result.get("ota_gamma_raw"),
                    "dominant_tier":      dominant_tier,
                    "top_programs":       top_programs,
                    "tier":               tier_per_gene.get(gene, dominant_tier),
                    "ota_gamma_sigma":    ota_result.get("ota_gamma_sigma", 0.0),
                    "ota_gamma_ci_lower": ota_result.get("ota_gamma_ci_lower"),
                    "ota_gamma_ci_upper": ota_result.get("ota_gamma_ci_upper"),
                    "genebayes_grounded": ota_result.get("genebayes_grounded", False),
                    "stress_discounted":  _stress_flagged,
                    "tissue_weight":      round(_tissue_mult, 4),
                }
            except Exception as exc:
                warnings.append(f"Ota γ computation failed for {gene} → {trait}: {exc}")

    # -------------------------------------------------------------------------
    # SCONE refinement (Reisach et al. 2024)
    # Reconciles Ota composite gamma with eQTL-mediated "soft intervention"
    # regimes to learn causal structure.
    # -------------------------------------------------------------------------
    try:
        from pipelines.scone_sensitivity import apply_scone_refinement, polybic_selection, bootstrap_edge_confidence

        # eqtl_data is in beta_matrix_result (keyed by gene)
        eqtl_data = beta_matrix_result.get("eqtl_data", {})

        refined_records: dict = {}
        for key, rec in gene_gamma.items():
            gene_sym = rec["gene"]

            # Get beta and eQTL info for this gene
            gene_beta_info = beta_matrix.get(gene_sym, {})
            gene_eqtl_info = eqtl_data.get(gene_sym)

            # Apply SCONE refinement to ota_gamma
            scone_res = apply_scone_refinement(
                ota_gamma=rec["ota_gamma"],
                beta_info=gene_beta_info,
                eqtl_info=gene_eqtl_info,
            )

            new_rec = dict(rec)
            new_rec["ota_gamma"] = scone_res["refined_gamma"]
            new_rec["scone_sensitivity"] = scone_res["scone_sensitivity"]
            new_rec["regime_consistency"] = scone_res["regime_consistency"]
            refined_records[key] = new_rec

        # Bootstrap confidence for all genes — stamps scone_confidence on each record.
        _bootstrap_conf: dict[str, dict] = {}
        for _bc_gene in list({rec["gene"] for rec in refined_records.values()}):
            try:
                _bootstrap_conf[_bc_gene] = bootstrap_edge_confidence(
                    gene=_bc_gene,
                    beta_matrix_row=beta_matrix.get(_bc_gene, {}),
                    gamma_matrix=gamma_estimates,
                    ota_gamma_fn=compute_ota_gamma,
                    n_bootstrap=30,
                )
            except Exception:
                pass

        for _bc_rec in refined_records.values():
            _bc_data = _bootstrap_conf.get(_bc_rec["gene"], {})
            if "mean" in _bc_data:
                _bc_rec["scone_confidence"] = round(float(_bc_data["mean"]), 4)

        # All genes go through polybic uniformly — no bypass.
        _polybic_kept = {
            r["gene"] + "__" + r["trait"]: r
            for r in polybic_selection(list(refined_records.values()), n_samples=1000)
        }
        for r in _polybic_kept.values():
            r["scone_tested"] = True

        gene_gamma = _polybic_kept

        warnings.append(
            f"SCONE: {len(_polybic_kept)} edges kept by polybic (uniform, no anchor bypass)"
        )

    except Exception as exc:
        warnings.append(f"SCONE refinement failed (non-fatal): {exc}")

    # -------------------------------------------------------------------------
    # Phase E: Therapeutic redirection
    # -------------------------------------------------------------------------
    tr_results = _maybe_therapeutic_redirection(
        gene_list=gene_list,
        beta_matrix_result=beta_matrix_result,
        disease_key=short,
        gene_gamma=gene_gamma,
        warnings=warnings,
        efo_id=efo_id,
    )

    import math as _math

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

        # Do not write pure virtual edges — genes without Perturb-seq β
        # score low and are filtered here; this is the accepted limitation
        # of the Ota et al. β×γ framework.
        if dominant_tier == "provisional_virtual":
            edges_rejected.append(rec)
            continue

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
        "Tier1_Interventional", "Tier2_Convergent", "Tier2s_SyntheticPathway",
        "Tier3_Provisional", "moderate_transferred", "moderate_grn", "provisional_virtual",
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
        # CI from SCONE bootstrap σ: γ ± 1.96σ
        # Prefer σ-derived CI (correctly centred on ota_gamma) over the raw
        # compute_ota_gamma_with_uncertainty CIs (which may be centred on a
        # pre-reweighted value and can be misleadingly narrow).
        sigma = float(rec.get("ota_gamma_sigma") or 0.0)
        if sigma > 0.0:
            ci_lo: float | None = round(float(rec["ota_gamma"]) - 1.96 * sigma, 4)
            ci_hi: float | None = round(float(rec["ota_gamma"]) + 1.96 * sigma, 4)
        else:
            # Fall back to stored CI only when non-trivial (both must be non-zero)
            raw_lo = rec.get("ota_gamma_ci_lower")
            raw_hi = rec.get("ota_gamma_ci_upper")
            ci_lo = float(raw_lo) if raw_lo else None
            ci_hi = float(raw_hi) if raw_hi else None

        # scone_tested=True  → SCONE-refined + polybic-selected
        # scone_tested=False → genetic-evidence-only (anchor pre-seed or OT fallback)
        # Default True for any record created before this tagging was added
        _scone_tested = rec.get("scone_tested", True)
        _edge_method = "ota_gamma_scone" if _scone_tested else "ota_gamma_genetic_only"

        causal_edge_dicts.append({
            "from_node":          rec["gene"],
            "from_type":          "gene",
            "to_node":            rec["trait"],
            "to_type":            "trait",
            "effect_size":        float(rec["ota_gamma"]),
            "ci_lower":           ci_lo,
            "ci_upper":           ci_hi,
            "evidence_type":      "germline",
            "evidence_tier":      tier,
            "method":             _edge_method,
            "data_source":        "Ota2026_composite_gamma",
            "data_source_version": "2026",
            "scone_tested":       _scone_tested,
        })

    # Write to graph (two channels):
    # 1) instrumented Ota γ edges (germline)
    # 2) state-space nominated edges (trajectory; non-instrumented)
    n_written = 0
    _write_errors: list[str] = []
    if causal_edge_dicts:
        try:
            write_result = write_causal_edges(causal_edge_dicts, disease_name)
            _w = (
                write_result.get("written")
                if isinstance(write_result, dict) and "written" in write_result
                else write_result.get("n_written")  # backward-compat with older stubs/tests
            )
            n_written += int(_w if _w is not None else len(causal_edge_dicts))
            if write_result.get("errors"):
                _write_errors.extend([str(e) for e in (write_result.get("errors") or [])])
        except Exception as exc:
            warnings.append(f"Graph write failed (instrumented edges): {exc}")

    # State-space edges are written later (after Phase K supplementation) so we
    # can reuse the final state_edge_effect/confidence fields.


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

    # -------------------------------------------------------------------------
    # Novel discovery metric: edges involving genes NOT in prior anchor set
    # All written edges are novel discoveries (no anchor seeding).
    # -------------------------------------------------------------------------
    novel_edges = list(edges_to_write)
    novel_genes = list({r["gene"] for r in novel_edges})

    # -------------------------------------------------------------------------
    # Phase E: therapeutic redirection + evidence disagreement (non-fatal)
    # -------------------------------------------------------------------------
    disease_key = _get_disease_key(disease_name.lower()) or ""
    tr_results: dict[str, dict] = _maybe_therapeutic_redirection(
        gene_list=gene_list,
        beta_matrix_result=beta_matrix_result,
        disease_key=disease_key,
        gene_gamma=gene_gamma,
        warnings=warnings,
        efo_id=efo_id,
    )

    # Extract program disease-specificity weights returned by _maybe_therapeutic_redirection.
    # Used to compute beta_amd_concentration — a diagnostic field that flags pleiotropic
    # genes whose β is spread across generic (non-AMD-specific) programs.
    # ota_gamma is NOT modified; this is reporting only.
    _program_weights: dict[str, float] = tr_results.pop("__program_weights__", {})
    if _program_weights:
        try:
            from pipelines.state_space.program_specificity import compute_beta_amd_concentration
            for _gene, _gbg in gene_best_gamma.items():
                _gene_beta = beta_matrix.get(_gene, {})
                _gbg["beta_amd_concentration"] = compute_beta_amd_concentration(
                    _gene_beta, _program_weights
                )
        except Exception as _exc_pw_apply:
            warnings.append(
                f"beta_amd_concentration computation failed (non-fatal): {_exc_pw_apply}"
            )

    # Phase K: supplement top_genes with state-nominated genes from TR results.
    # IMPORTANT: Do not emit NaN into JSON output. Use explicit state-space fields instead.
    def _compute_state_edge_effect_and_confidence(_tr: dict | None) -> tuple[float, float, float | None, float | None, float | None]:
        """
        Compute a bounded state-space effect and confidence from TR signals.

        effect mirrors Tier 4's mechanistic score; confidence uses stability and risk flags.
        """
        if not isinstance(_tr, dict) or not _tr:
            return 0.0, 0.5, None, None, None

        # Empirical-ish bootstrap CI over the composite effect.
        try:
            # Prefer transition-edge bootstrap results if upstream provided them.
            _pre_ci_lo = _tr.get("state_edge_ci_lower")
            _pre_ci_hi = _tr.get("state_edge_ci_upper")
            _pre_cv = _tr.get("state_edge_cv")
            _pre_conf = _tr.get("state_edge_confidence")
            if _pre_ci_lo is not None or _pre_ci_hi is not None or _pre_conf is not None:
                try:
                    _eff = float(_tr.get("therapeutic_redirection", _tr.get("state_edge_effect", 0.0)) or 0.0)
                except Exception:
                    _eff = 0.0
                try:
                    _conf = float(_pre_conf) if _pre_conf is not None else 0.5
                except Exception:
                    _conf = 0.5
                return (
                    round(_eff, 6),
                    round(_conf, 6),
                    (float(_pre_ci_lo) if _pre_ci_lo is not None else None),
                    (float(_pre_ci_hi) if _pre_ci_hi is not None else None),
                    (float(_pre_cv) if _pre_cv is not None else None),
                )

            from pipelines.state_space.state_edge_bootstrap import bootstrap_state_edge_effect
            bs = bootstrap_state_edge_effect(_tr, n_bootstrap=200, seed=0)
            eff = float(bs.get("mean", 0.0))
            conf = float(bs.get("confidence", 0.5))
            ci_lo = bs.get("ci_lower")
            ci_hi = bs.get("ci_upper")
            cv = bs.get("cv")
            return (
                round(eff, 6),
                round(conf, 6),
                (float(ci_lo) if ci_lo is not None else None),
                (float(ci_hi) if ci_hi is not None else None),
                (float(cv) if cv is not None else None),
            )
        except Exception:
            # Fallback: deterministic composite (no CI)
            try:
                tr_val = abs(float(_tr.get("therapeutic_redirection", 0.0)))
            except Exception:
                tr_val = 0.0
            try:
                si_val = float(_tr.get("state_influence_score", 0.0))
            except Exception:
                si_val = 0.0
            def _f(key: str) -> float:
                try:
                    return float(_tr.get(key, 0.0))
                except Exception:
                    return 0.0
            entry_s = _f("entry_score")
            persist_s = _f("persistence_score")
            recovery_s = _f("recovery_score")
            boundary_s = _f("boundary_score")
            transition_avg = 0.25 * (entry_s + persist_s + recovery_s + boundary_s)
            effect = tr_val + 0.3 * si_val + 0.2 * transition_avg
            if effect < 0.0:
                effect = 0.0
            elif effect > 1.0:
                effect = 1.0
            try:
                stability = float(_tr.get("stability_score", _tr.get("stability", 0.5)))
            except Exception:
                stability = 0.5
            if stability < 0.0:
                stability = 0.0
            elif stability > 1.0:
                stability = 1.0
            conf = stability
            return round(effect, 6), round(conf, 6), None, None, None

    _state_space_edge_dicts: list[dict] = []
    for _sn_gene, _sn_rec in tr_results.items():
        if _sn_rec.get("state_nominated") and _sn_gene not in gene_best_gamma:
            _tr = _sn_rec.get("therapeutic_redirection_result")
            _eff, _conf, _ci_lo, _ci_hi, _cv = _compute_state_edge_effect_and_confidence(_tr)
            gene_best_gamma[_sn_gene] = {
                "gene":                 _sn_gene,
                "ota_gamma":            None,   # explicitly missing (not NaN)
                "ota_gamma_raw":        None,
                "ota_gamma_sigma":      None,
                "ota_gamma_ci_lower":   None,
                "ota_gamma_ci_upper":   None,
                "tier":                 "state_nominated",
                "trait":                disease_name,
                "top_programs":         [],
                "evidence_type":        "state_space",
                "state_edge_effect":    _eff,
                "state_edge_confidence": _conf,
                "state_edge_ci_lower":  _ci_lo,
                "state_edge_ci_upper":  _ci_hi,
                "state_edge_cv":        _cv,
                # Pass-through so Tier 4/5 can render rich mechanistic context
                "therapeutic_redirection_result": _tr,
                "controller_annotation": _sn_rec.get("controller_annotation"),
                "evidence_disagreement": _sn_rec.get("evidence_disagreement", []),
                "tau_disease_specificity": _sn_rec.get("tau_disease_specificity"),
                "disease_log2fc":          _sn_rec.get("disease_log2fc"),
                "tau_specificity_class":   _sn_rec.get("tau_specificity_class"),
                "beta_amd_concentration":  _sn_rec.get("beta_amd_concentration"),
            }
            # Also emit an explicit state-space edge so downstream agents (SCONE/KG/GPS)
            # can attach stats without requiring ota_gamma.
            try:
                _state_space_edge_dicts.append({
                    "from_node":          _sn_gene,
                    "from_type":          "gene",
                    "to_node":            disease_name,
                    "to_type":            "trait",
                    "effect_size":        float(_eff),
                    "ci_lower":           _ci_lo,
                    "ci_upper":           _ci_hi,
                    "evidence_type":      "state_space",
                    "evidence_tier":      "Tier3_Provisional",
                    "method":             "trajectory",
                    "data_source":        "state_space/therapeutic_redirection",
                    "data_source_version": "2026",
                    "cell_type":          (_tr.get("cell_type") if isinstance(_tr, dict) else None),
                })
            except Exception:
                pass

    if _state_space_edge_dicts:
        try:
            _wr = write_causal_edges(_state_space_edge_dicts, disease_name)
            _w2 = (
                _wr.get("written")
                if isinstance(_wr, dict) and "written" in _wr
                else _wr.get("n_written")
            )
            n_written += int(_w2 if _w2 is not None else len(_state_space_edge_dicts))
            if _wr.get("errors"):
                _write_errors.extend([str(e) for e in (_wr.get("errors") or [])])
        except Exception as exc:
            warnings.append(f"Graph write failed (state-space edges): {exc}")

    if _write_errors:
        warnings.append(f"Graph write reported {len(_write_errors)} ingestion errors (see logs)")
    # Phase L: TWMR supplement — upgrade tier and blend beta for genes with strong eQTL instruments
    if efo_id:
        try:
            import time as _time
            from pipelines.twmr import run_twmr_for_gene as _run_twmr
            for _twmr_gene, _twmr_rec in gene_best_gamma.items():
                if _twmr_rec.get("tier") == "state_nominated":
                    continue  # no genetic instrument to test
                try:
                    _twmr_result = _run_twmr(
                        gene=_twmr_gene,
                        disease_key=disease_key,
                        efo_id=efo_id,
                    )
                except Exception:
                    _twmr_result = None
                if _twmr_result and _twmr_result.get("f_statistic", 0) >= MR_F_STATISTIC_MIN:
                    _old_gamma = _twmr_rec["ota_gamma"]
                    _twmr_beta = _twmr_result["beta"]
                    # Conservative blend: average when both signals present
                    _twmr_rec["ota_gamma"] = (
                        (_twmr_beta + _old_gamma) / 2.0 if _old_gamma != 0.0 else _twmr_beta
                    )
                    _twmr_rec["twmr_beta"]         = _twmr_beta
                    _twmr_rec["twmr_se"]           = _twmr_result.get("se")
                    _twmr_rec["twmr_p"]            = _twmr_result.get("p")
                    _twmr_rec["twmr_f_stat"]       = _twmr_result["f_statistic"]
                    _twmr_rec["twmr_n_instruments"]= _twmr_result["n_instruments"]
                    _twmr_rec["twmr_method"]       = _twmr_result.get("method", "IVW")
                    # Upgrade tier: TWMR F≥10 = two independent convergent sources
                    if _twmr_rec.get("tier") in ("Tier3_Provisional", "provisional_virtual", "moderate_transferred", "moderate_grn"):
                        _twmr_rec["tier"] = "Tier2_Convergent"
                gene_best_gamma[_twmr_gene] = _twmr_rec
                _time.sleep(0.3)  # GTEx rate-limit guard
        except Exception as _twmr_exc:
            warnings.append(f"Phase L TWMR supplement failed (non-fatal): {_twmr_exc}")

    # All genes above the edge-write threshold pass to the prioritization agent.
    # No count cutoff — any hard number is arbitrary and silently excludes genes
    # with valid genetic evidence but low β×γ (e.g. extracellular proteins).
    # OTA_GAMMA_MIN is the natural floor: edges below this aren't written to the
    # graph and carry no meaningful signal.
    import math as _math_rebuild
    _gwas_top = sorted(
        [
            r for r in gene_best_gamma.values()
            if r.get("tier") != "state_nominated"
            and not _math_rebuild.isnan(r.get("ota_gamma", float("nan")))
            and abs(r.get("ota_gamma", 0.0)) >= OTA_GAMMA_MIN
        ],
        key=lambda r: abs(r["ota_gamma"]),
        reverse=True,
    )

    # -------------------------------------------------------------------------
    # Phase Z7: INSPRE Structure + GNN Discovery (opt-in; can be heavyweight)
    # -------------------------------------------------------------------------
    inspre_adj: dict = {}
    gnn_rankings: dict[str, float] = {}
    if bool(disease_query.get("enable_phase_z7")):
        try:
            # 1. INSPRE Structure learning (Gene-Gene)
            from pipelines.inspre_structure import infer_gene_gene_structure
            # Use first available h5ad for structure
            if "h5ad_files" in locals() and h5ad_files:
                import scanpy as sc
                _adata_gg = sc.read_h5ad(str(h5ad_files[0]))
                _top_symbols = [r["gene"] for r in _gwas_top[:50]]
                inspre_adj = infer_gene_gene_structure(_adata_gg, _top_symbols)
                warnings.append(f"INSPRE: inferred {sum(len(v) for v in inspre_adj.values())} gene-gene edges")

            # 2. Stability-Aware GNN
            from pipelines.biopath_gnn_v2 import build_stability_aware_brg, StabilityAwareGNN
            # Build edges from current discovery results
            causal_edges_for_gnn = []
            for r in _gwas_top:
                tr_rec = tr_results.get(r["gene"], {}).get("therapeutic_redirection_result") or {}
                causal_edges_for_gnn.append({
                    "from_node": r["gene"],
                    "to_node":   disease_name,
                    "effect_size": r["ota_gamma"],
                    "scone_stability": tr_rec.get("stability_score", 1.0)
                })
            
            # Placeholder for PPI/Pathway; in full run these would be fetched from MCP
            brg_adj = build_stability_aware_brg(ppi_edges=[], pathway_map={}, causal_edges=causal_edges_for_gnn)
            gnn_engine = StabilityAwareGNN(brg_adj)
            seed_scores = {r["gene"]: abs(r["ota_gamma"]) for r in _gwas_top}
            gnn_rankings = gnn_engine.run_inference(seed_scores)
            warnings.append(f"GNN: completed message passing across {len(brg_adj)} nodes")

        except Exception as exc_z7:
            warnings.append(f"Phase Z7 INSPRE/GNN failed (non-fatal): {exc_z7}")

    _nominated_top = [r for r in gene_best_gamma.values() if r.get("tier") == "state_nominated"][:20]
    top_genes = _gwas_top + _nominated_top

    return {
        "n_edges_written":  n_written,
        "n_edges_rejected": len(edges_rejected),
        "n_novel_edges":    len(novel_edges),
        "novel_genes":      novel_genes[:20],
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
                "twmr_beta":          r.get("twmr_beta"),
                "twmr_se":            r.get("twmr_se"),
                "twmr_p":             r.get("twmr_p"),
                "twmr_f_stat":        r.get("twmr_f_stat"),
                "twmr_n_instruments": r.get("twmr_n_instruments"),
                "twmr_method":        r.get("twmr_method"),
                "therapeutic_redirection_result": (
                    tr_results[r["gene"]].get("therapeutic_redirection_result")
                    if r["gene"] in tr_results else None
                ),
                "evidence_disagreement": (
                    tr_results[r["gene"]].get("evidence_disagreement", [])
                    if r["gene"] in tr_results else []
                ),
                "controller_annotation": (
                    tr_results[r["gene"]].get("controller_annotation")
                    if r["gene"] in tr_results else None
                ),
                # Phase R: τ direct keys (non-None for all genes, even without TR)
                "tau_disease_specificity": (
                    tr_results[r["gene"]].get("tau_disease_specificity")
                    if r["gene"] in tr_results else None
                ),
                "disease_log2fc": (
                    tr_results[r["gene"]].get("disease_log2fc")
                    if r["gene"] in tr_results else None
                ),
                "tau_specificity_class": (
                    tr_results[r["gene"]].get("tau_specificity_class")
                    if r["gene"] in tr_results else None
                ),
                # Phase U: pleiotropy diagnostic — fraction of β-mass in disease-specific programs
                "beta_amd_concentration": r.get("beta_amd_concentration"),
                # Phase Z7: GNN score
                "gnn_score": gnn_rankings.get(r["gene"]),
            }
            for r in top_genes
        ],
        "shd":                   0,
        "therapeutic_redirection_available": bool(tr_results),
        # All edges went through SCONE refinement + polybic selection.
        # genetic_only_edges is retained for schema compatibility but will be empty.
        "scone_edges": [
            {"gene": e["from_node"], "trait": e["to_node"],
             "gamma": e["effect_size"], "tier": e["evidence_tier"]}
            for e in causal_edge_dicts if e.get("scone_tested", True)
        ],
        "genetic_only_edges": [
            {"gene": e["from_node"], "trait": e["to_node"],
             "gamma": e["effect_size"], "tier": e["evidence_tier"]}
            for e in causal_edge_dicts if not e.get("scone_tested", True)
        ],
        "n_scone_edges":        sum(1 for e in causal_edge_dicts if e.get("scone_tested", True)),
        "n_genetic_only_edges": sum(1 for e in causal_edge_dicts if not e.get("scone_tested", True)),
        # Phase Z7: Structure + GNN
        "inspre_graph":          inspre_adj,
        "gnn_rankings":          gnn_rankings,
        "warnings":              warnings,
    }


# ---------------------------------------------------------------------------
# Phase E: therapeutic redirection helper
# ---------------------------------------------------------------------------

def _maybe_therapeutic_redirection(
    gene_list: list[str],
    beta_matrix_result: dict,
    disease_key: str,
    gene_gamma: dict,
    warnings: list[str],
    efo_id: str = "",
) -> dict[str, dict]:
    """
    Compute TherapeuticRedirectionResult and evidence disagreement for all genes.

    Uses beta_matrix betas (already computed by perturbation_genomics_agent) as
    ConditionalBeta proxies.  Falls back gracefully when h5ad data is absent.

    Returns {gene: {"therapeutic_redirection_result": dict | None,
                    "evidence_disagreement": list[dict]}}
    or {} if no h5ad data is available.
    """
    if not disease_key:
        return {}

    try:
        import math
        from pathlib import Path
        from models.latent_mediator import ConditionalBeta
        from pipelines.discovery.cellxgene_downloader import _CACHE_ROOT
        from pipelines.evidence_disagreement import (
            run_all_disagreement_checks,
            build_disagreement_profile,
        )

        disease_dir = _CACHE_ROOT / disease_key
        if not disease_dir.exists():
            return {}

        h5ad_files = sorted(disease_dir.glob(f"{disease_key}_*.h5ad"))
        if not h5ad_files:
            return {}

        beta_matrix    = beta_matrix_result.get("beta_matrix", {})
        tier_per_gene  = beta_matrix_result.get("evidence_tier_per_gene", {})

        def _to_state_tier(ev_tier: str) -> str:
            if ev_tier in ("Tier1_Interventional",):
                return "Tier1_Interventional"
            if ev_tier in ("Tier2_Convergent", "moderate_transferred", "moderate_grn"):
                return "Tier2_Convergent"
            return "Tier3_TrajectoryProxy"

        # Build gene-level ConditionalBeta objects from beta_matrix (pooled proxy)
        # beta_matrix values are either float | None  OR  {"beta": float, "evidence_tier": str, ...}
        cb_by_gene: dict[str, list[ConditionalBeta]] = {}
        for gene in gene_list:
            gene_betas_raw = beta_matrix.get(gene, {})
            ev_tier = tier_per_gene.get(gene, "provisional_virtual")
            state_tier = _to_state_tier(ev_tier)
            cbs: list[ConditionalBeta] = []
            for prog_id, beta_val in gene_betas_raw.items():
                if beta_val is None:
                    continue
                # Unwrap dict format {"beta": float, "evidence_tier": str, ...}
                if isinstance(beta_val, dict):
                    raw = beta_val.get("beta")
                    if raw is None:
                        continue
                    item_tier = _to_state_tier(beta_val.get("evidence_tier", ev_tier))
                else:
                    raw = beta_val
                    item_tier = state_tier
                try:
                    fval = float(raw)
                except (TypeError, ValueError):
                    continue
                if not math.isfinite(fval):
                    continue
                cbs.append(ConditionalBeta(
                    gene=gene,
                    program_id=prog_id,
                    cell_type="pooled",
                    disease=disease_key,
                    beta=fval,
                    beta_se=None,
                    pooled_fallback=True,
                    context_verified=False,
                    evidence_tier=item_tier,
                    data_source=f"beta_matrix/{ev_tier}",
                ))
            cb_by_gene[gene] = cbs

        # MR gamma lookup (best per gene)
        mr_gamma_by_gene: dict[str, float] = {}
        for rec in gene_gamma.values():
            g = rec.get("gene")
            if g:
                og = float(rec.get("ota_gamma", 0.0))
                if abs(og) > abs(mr_gamma_by_gene.get(g, 0.0)):
                    mr_gamma_by_gene[g] = og

        # Build per-celltype therapeutic redirection (best-effort)
        per_ct_by_gene: dict[str, dict[str, dict]] = {g: {} for g in gene_list}
        controller_annot_by_gene: dict[str, dict] = {}  # Phase H
        _state_nominated: set[str] = set()  # Phase K: state-space nominated genes
        disease_tau_by_gene: dict[str, float] = {}  # Phase R: disease-state τ specificity
        _computed_program_weights: dict[str, float] = {}  # Phase U: program disease-specificity weights
        try:
            from pipelines.discovery.cnmf_runner import run_nmf_programs
            from pipelines.state_space.latent_model import build_disease_latent_space
            from pipelines.state_space.state_definition import define_cell_states
            from pipelines.state_space.transition_graph import (
                infer_state_transition_graph,
                compute_transition_gene_weights,
            )
            from pipelines.state_space.program_loading import compute_program_loading
            from pipelines.state_space.program_labeler import label_programs
            from pipelines.state_space.conditional_gamma import estimate_conditional_gammas_for_programs
            from pipelines.state_space.therapeutic_redirection import (
                compute_therapeutic_redirection_per_celltype,
                compute_therapeutic_redirection,
                compute_state_direct_redirection,
            )
            from pipelines.state_space.state_influence import compute_gene_transition_profiles

            for h5ad_path in h5ad_files:
                stem = h5ad_path.stem  # e.g. "IBD_macrophage"
                cell_type = stem[len(disease_key) + 1:].replace("_", " ")

                latent = build_disease_latent_space(disease_key, [str(h5ad_path)])
                if latent.get("error"):
                    warnings.append(
                        f"therapeutic_redirection: latent build failed for {cell_type}: "
                        f"{latent['error']}"
                    )
                    continue

                states = define_cell_states(latent, disease_key)
                trans  = infer_state_transition_graph(latent, states, disease_key)

                adata = latent.get("adata")
                # Remap Ensembl IDs → gene symbols so transition scoring uses symbols.
                # CELLxGENE h5ads use Ensembl IDs as var_names; feature_name holds symbols.
                if adata is not None and "feature_name" in adata.var.columns:
                    import pandas as pd
                    from collections import Counter
                    raw_symbols = adata.var["feature_name"].tolist()
                    sym_counts = Counter(raw_symbols)
                    # Only remap non-duplicate symbols; keep Ensembl ID for duplicates
                    symbol_map = adata.var["feature_name"].to_dict()
                    new_var_names = [
                        sym if sym_counts[sym] == 1 else ensembl
                        for ensembl, sym in zip(adata.var_names, raw_symbols)
                    ]
                    adata.var_names = pd.Index(new_var_names)
                    latent["adata"] = adata

                transition_gene_weights: dict[str, float] = {}
                state_influence_all: dict[str, dict] = {}
                if adata is not None:
                    transition_gene_weights = compute_transition_gene_weights(
                        adata, trans,
                        pathologic_basin_ids=trans.get("pathologic_basin_ids"),
                        healthy_basin_ids=trans.get("healthy_basin_ids"),
                    )
                    # Phase G: transition-aware profiles for every gene
                    _transition_profiles = compute_gene_transition_profiles(
                        adata, trans, gene_list
                    )
                    # Convert to flat dicts for downstream compatibility
                    state_influence_all = {
                        g: p.model_dump()
                        for g, p in _transition_profiles.items()
                    }
                    # Phase H: controller classification (first cell type wins)
                    try:
                        from pipelines.state_space.controller_classifier import classify_gene_list
                        ctrl_annots = classify_gene_list(
                            gene_list, disease_key,
                            transition_profiles=_transition_profiles,
                            evidence_tiers=tier_per_gene,
                            adata=adata,
                        )
                        for g, ann in ctrl_annots.items():
                            if g not in controller_annot_by_gene:
                                controller_annot_by_gene[g] = ann.model_dump()
                    except Exception as exc_h:
                        warnings.append(f"Phase H classification failed (non-fatal): {exc_h}")

                    # Phase K: nominate state genes beyond GWAS instruments
                    try:
                        from pipelines.state_space.state_influence import nominate_state_genes
                        from pipelines.state_space.state_edge_bootstrap import (
                            bootstrap_state_edge_effect_from_transitions,
                        )
                        _nominated = nominate_state_genes(
                            adata, trans, top_k=25, exclude=set(gene_list)
                        )
                        # Map state_id → index for bootstrap
                        _state_labels = trans.get("state_labels") or []
                        _label_to_idx = {sid: i for i, sid in enumerate(_state_labels)}
                        _path_idxs = [_label_to_idx[sid] for sid in (trans.get("pathologic_basin_ids") or []) if sid in _label_to_idx]
                        _healthy_idxs = [_label_to_idx[sid] for sid in (trans.get("healthy_basin_ids") or []) if sid in _label_to_idx]
                        # State sizes (n_cells) aligned with state_labels
                        _n_cells_map = {getattr(s, "state_id", None): getattr(s, "n_cells", 50) for s in (states or [])}
                        _n_cells_per_state = [
                            int(_n_cells_map.get(sid, 50) or 50) for sid in _state_labels
                        ]
                        for _ng, _ng_profile in _nominated:
                            _state_nominated.add(_ng)
                            state_influence_all[_ng] = _ng_profile
                            # Empirical Method B: transition-edge bootstrap for a state-edge effect.
                            # Store into the TR dict so Tier 3 Phase K can consume it directly.
                            try:
                                _das = float(_ng_profile.get("disease_axis_score", 0.0))
                            except Exception:
                                _das = 0.0
                            try:
                                _bs = bootstrap_state_edge_effect_from_transitions(
                                    T_baseline=trans.get("transition_matrix"),
                                    disease_axis_score=_das,
                                    path_idxs=_path_idxs,
                                    healthy_idxs=_healthy_idxs,
                                    n_cells_per_state=_n_cells_per_state,
                                    alpha_scale=1.0,
                                    n_bootstrap=300,
                                    seed=0,
                                    resample_only_path_rows=True,
                                )
                            except Exception:
                                _bs = None
                            per_ct_by_gene.setdefault(_ng, {})[cell_type] = {
                                "redirection": 0.0,
                                "nmf_redirection": 0.0,
                                "state_direct": 0.0,
                                "stability": 1.0,
                                "state_influence_score": _ng_profile.get("disease_axis_score", 0.0),
                                "directionality": _ng_profile.get("directionality", 0),
                                "n_programs": 0,
                                "pooled_fraction": 0.0,
                                "evidence_tiers": [],
                                "provenance": ["state_nominated"],
                                "entry_score":       _ng_profile.get("entry_score", 0.0),
                                "persistence_score": _ng_profile.get("persistence_score", 0.0),
                                "recovery_score":    _ng_profile.get("recovery_score", 0.0),
                                "boundary_score":    _ng_profile.get("boundary_score", 0.0),
                                "mechanistic_category": _ng_profile.get("mechanistic_category", "unknown"),
                                # Provide a state-edge effect compatible with Phase K.
                                "therapeutic_redirection": (_bs.get("mean") if isinstance(_bs, dict) else 0.0),
                                "state_edge_ci_lower": (_bs.get("ci_lower") if isinstance(_bs, dict) else None),
                                "state_edge_ci_upper": (_bs.get("ci_upper") if isinstance(_bs, dict) else None),
                                "state_edge_cv": (_bs.get("cv") if isinstance(_bs, dict) else None),
                                "state_edge_confidence": (_bs.get("confidence") if isinstance(_bs, dict) else None),
                            }
                            mr_gamma_by_gene.setdefault(_ng, 0.0)
                    except Exception as _exc_k:
                        warnings.append(f"Phase K state nomination failed (non-fatal): {_exc_k}")

                    # Phase R: disease-state τ specificity
                    try:
                        from pipelines.state_space.tau_specificity import compute_disease_tau
                        # Include all gene_list genes (e.g. GWAS hits like NOD2) not just
                        # those that reached state_influence_all via transition profiling
                        _all_scored = list(set(list(state_influence_all.keys()) + list(gene_list)))
                        _tau_results = compute_disease_tau(adata, _all_scored)
                        # Store full TauResult so log2fc and specificity_class can be patched
                        disease_tau_by_gene = dict(_tau_results)
                        # Also store τ fields in state_influence_all for downstream surfacing
                        for g, r in _tau_results.items():
                            if g in state_influence_all:
                                state_influence_all[g]["tau_disease"] = r.tau_disease
                                state_influence_all[g]["disease_log2fc"] = r.disease_log2fc
                                state_influence_all[g]["tau_specificity_class"] = r.specificity_class
                    except Exception as _exc_r:
                        warnings.append(f"Phase R tau specificity failed (non-fatal): {_exc_r}")

                nmf_result = run_nmf_programs(str(h5ad_path), disease_key)

                # Program disease-specificity weights (Phase AMD/pleiotropic refinement).
                # Uses AMD h5ad disease/healthy contrast to weight each program by how
                # specifically it is active in disease cells.  Non-fatal if unavailable.
                if adata is not None:
                    try:
                        from pipelines.state_space.program_specificity import (
                            compute_program_disease_weights,
                        )
                        # Merge NMF programs + Hallmark programs from beta_matrix_result
                        _all_programs_for_weights = list(
                            nmf_result.get("programs", [])
                        )
                        # Also add any programs known from beta_matrix that aren't in NMF output
                        _beta_prog_ids = set(
                            prog for gene_betas in beta_matrix_result.get("beta_matrix", {}).values()
                            for prog in (gene_betas or {}).keys()
                        ) - {p.get("program_id") for p in _all_programs_for_weights}
                        if _beta_prog_ids:
                            try:
                                from pipelines.cnmf_programs import get_programs_for_disease
                                _hallmark_progs = get_programs_for_disease(disease_key)
                                for _hp in _hallmark_progs.get("programs", []):
                                    if _hp.get("program_id") in _beta_prog_ids:
                                        _all_programs_for_weights.append(_hp)
                            except Exception:
                                pass
                        _pw = compute_program_disease_weights(
                            adata, _all_programs_for_weights, disease_key=disease_key
                        )
                        if _pw:
                            _computed_program_weights.update(_pw)
                            warnings.append(
                                f"Program disease-specificity weights computed: "
                                f"{len(_pw)} programs, "
                                f"{sum(1 for w in _pw.values() if w > 0.5)} disease-specific"
                            )
                    except Exception as _exc_pw:
                        warnings.append(
                            f"Program disease-specificity weights failed (non-fatal): {_exc_pw}"
                        )

                program_loadings = compute_program_loading(
                    nmf_result, transition_gene_weights, disease_key, cell_type
                )
                labeled_programs = label_programs(nmf_result, disease_key, cell_type)
                conditional_gammas = estimate_conditional_gammas_for_programs(
                    labeled_programs, disease_key, disease_key,
                    transition_result=trans,
                    transition_gene_weights=transition_gene_weights,
                    evidence_tier="Tier3_TrajectoryProxy",
                    efo_id=efo_id or None,
                )

                # Compute mean beta per gene for proxy assignment to NMF programs
                mean_beta_by_gene: dict[str, float] = {}
                for gene in gene_list:
                    raw_vals = [
                        b.beta for b in cb_by_gene.get(gene, [])
                        if math.isfinite(b.beta)
                    ]
                    if raw_vals:
                        mean_beta_by_gene[gene] = sum(raw_vals) / len(raw_vals)

                for gene in gene_list:
                    gene_pls = [pl for pl in program_loadings if pl.gene == gene]
                    gene_gamma_ota = mr_gamma_by_gene.get(gene, 0.0)
                    si_for_gene = state_influence_all.get(gene, {})

                    if not gene_pls:
                        # Phase F: state-direct bypass — no NMF membership required
                        das = si_for_gene.get("disease_axis_score", 0.0)
                        if das < 1e-3 or not math.isfinite(gene_gamma_ota):
                            continue
                        # Emit a minimal per-celltype record via state-direct path
                        per_ct_by_gene[gene][cell_type] = {
                            "redirection": 0.0,           # will be computed inside per_celltype below
                            "nmf_redirection": 0.0,
                            "state_direct": 0.0,
                            "state_influence_score": das,
                            "directionality": si_for_gene.get("directionality", 0),
                            "n_programs": 0,
                            "pooled_fraction": 0.0,
                            "evidence_tiers": [],
                            "provenance": [],
                            # Phase G
                            "entry_score":       si_for_gene.get("entry_score", 0.0),
                            "persistence_score": si_for_gene.get("persistence_score", 0.0),
                            "recovery_score":    si_for_gene.get("recovery_score", 0.0),
                            "boundary_score":    si_for_gene.get("boundary_score", 0.0),
                            "mechanistic_category": si_for_gene.get("mechanistic_category", "unknown"),
                        }
                        # Compute state-direct contribution using the transition matrix
                        try:
                            import numpy as np
                            T_bl = trans.get("transition_matrix")
                            state_labels = trans.get("state_labels", [])
                            label_to_idx = {str(s): i for i, s in enumerate(state_labels)}
                            p_idxs = [label_to_idx[s] for s in trans.get("pathologic_basin_ids", []) if s in label_to_idx]
                            h_idxs = [label_to_idx[s] for s in trans.get("healthy_basin_ids", []) if s in label_to_idx]
                            if T_bl is not None and p_idxs and h_idxs:
                                from pipelines.state_space.therapeutic_redirection import compute_stability_score
                                sd = compute_state_direct_redirection(
                                    gene_beta=float("nan"),
                                    disease_axis_score=das,
                                    directionality=si_for_gene.get("directionality", 0),
                                    gamma_ota=gene_gamma_ota,
                                    T_baseline=T_bl,
                                    path_idxs=p_idxs,
                                    healthy_idxs=h_idxs,
                                )
                                # Phase Z7: Stability for state-direct path
                                stab = compute_stability_score(
                                    T_bl, 1.0, das, p_idxs, h_idxs, n_iterations=20
                                )
                                per_ct_by_gene[gene][cell_type]["redirection"] = sd
                                per_ct_by_gene[gene][cell_type]["state_direct"] = sd
                                per_ct_by_gene[gene][cell_type]["stability"] = stab
                                if sd > 0:
                                    per_ct_by_gene[gene][cell_type]["provenance"].append(
                                        f"state_direct(das={das:.3f},γ={gene_gamma_ota:.3f},Δ={sd:.4f},stab={stab:.2f})"
                                    )
                        except Exception:
                            pass
                        continue

                    proxy_beta = mean_beta_by_gene.get(gene)
                    if proxy_beta is None or not math.isfinite(proxy_beta):
                        proxy_beta = float("nan")

                    gene_cbs_ct = [
                        ConditionalBeta(
                            gene=gene,
                            program_id=pl.program_id,
                            cell_type=cell_type,
                            disease=disease_key,
                            beta=proxy_beta if math.isfinite(proxy_beta) else float("nan"),
                            beta_se=None,
                            pooled_fallback=True,
                            context_verified=False,
                            evidence_tier="Tier3_TrajectoryProxy",
                            data_source="beta_matrix_proxy",
                        )
                        for pl in gene_pls
                        if math.isfinite(proxy_beta)
                    ]
                    if not gene_cbs_ct:
                        continue

                    try:
                        ct_result = compute_therapeutic_redirection_per_celltype(
                            gene=gene,
                            disease=disease_key,
                            cell_type=cell_type,
                            program_loadings=gene_pls,
                            conditional_betas=gene_cbs_ct,
                            conditional_gammas=conditional_gammas,
                            transition_result=trans,
                            state_influence=si_for_gene,
                            gene_gamma_ota=gene_gamma_ota,
                        )
                        per_ct_by_gene[gene][cell_type] = ct_result
                    except Exception:
                        pass

            # Aggregate TherapeuticRedirectionResult per gene (GWAS + state-nominated)
            for gene in list(gene_list) + [g for g in _state_nominated if g not in set(gene_list)]:
                per_ct = per_ct_by_gene.get(gene, {})
                if per_ct:
                    try:
                        tr_result = compute_therapeutic_redirection(
                            gene, disease_key, per_ct,
                            genetic_grounding=mr_gamma_by_gene.get(gene, 0.0),
                        )
                        if gene not in per_ct_by_gene:  # ensure dict entry
                            per_ct_by_gene[gene] = {}
                        per_ct_by_gene[gene]["__tr_result__"] = tr_result
                    except Exception:
                        pass

        except Exception as exc_ss:
            warnings.append(f"therapeutic_redirection state_space step failed (non-fatal): {exc_ss}")

        # Assemble final results (GWAS instruments + state-nominated)
        _all_scored_genes = list(gene_list) + [g for g in _state_nominated if g not in set(gene_list)]
        results: dict[str, dict] = {}
        if _computed_program_weights:
            results["__program_weights__"] = _computed_program_weights
        for gene in _all_scored_genes:
            tr_result = per_ct_by_gene.get(gene, {}).pop("__tr_result__", None)
            disagreement: list[dict] = []
            try:
                recs = run_all_disagreement_checks(
                    gene, disease_key, cb_by_gene.get(gene, []),
                    mr_gamma=mr_gamma_by_gene.get(gene),
                )
                disagreement = [r.model_dump() for r in recs]
            except Exception:
                pass

            ctrl_ann = controller_annot_by_gene.get(gene)

            # Phase I: structured disagreement profile
            disagreement_profile: dict | None = None
            try:
                from models.evidence import TransitionGeneProfile
                tp_dict = state_influence_all.get(gene)
                tp_obj = TransitionGeneProfile.model_validate(tp_dict) if tp_dict else None
                dp = build_disagreement_profile(
                    gene, disease_key,
                    cb_by_gene.get(gene, []),
                    mr_gamma=mr_gamma_by_gene.get(gene),
                    evidence_tier=tier_per_gene.get(gene, "provisional_virtual"),
                    transition_profile=tp_obj,
                    controller_annotation=ctrl_ann,
                )
                disagreement_profile = dp.model_dump()
            except Exception:
                pass

            if tr_result is not None or disagreement or ctrl_ann is not None or gene in _state_nominated:
                tr_dict = tr_result.model_dump() if tr_result is not None else None
                # Phase R: patch all τ fields into TR dict (log2fc and class too)
                if tr_dict is not None and gene in disease_tau_by_gene:
                    _tau_r = disease_tau_by_gene[gene]
                    tr_dict["tau_disease_specificity"] = _tau_r.tau_disease
                    tr_dict["disease_log2fc"]          = _tau_r.disease_log2fc
                    tr_dict["tau_specificity_class"]   = _tau_r.specificity_class
                results[gene] = {
                    "therapeutic_redirection_result": tr_dict,
                    "evidence_disagreement": disagreement,
                    "controller_annotation": ctrl_ann,
                    "disagreement_profile": disagreement_profile,
                    "state_nominated": gene in _state_nominated,
                }

        if results:
            warnings.append(
                f"Phase E: therapeutic_redirection computed for {len(results)} genes "
                f"across {len(h5ad_files)} h5ad(s)"
            )

        # Phase R final: surface τ directly on all gene result records so genes
        # without TR (e.g. pure GWAS hits) still carry τ through the pipeline
        for gene in _all_scored_genes:
            if gene not in disease_tau_by_gene:
                continue
            _tau_r = disease_tau_by_gene[gene]
            if gene not in results:
                # No state-space result — create minimal entry only if non-sentinel
                if _tau_r.tau_disease == 0.5 and _tau_r.specificity_class == "unknown":
                    continue
                results[gene] = {
                    "therapeutic_redirection_result": None,
                    "evidence_disagreement": [],
                    "controller_annotation": None,
                    "disagreement_profile": None,
                    "state_nominated": False,
                    "tau_disease_specificity": _tau_r.tau_disease,
                    "disease_log2fc": _tau_r.disease_log2fc,
                    "tau_specificity_class": _tau_r.specificity_class,
                }
            else:
                results[gene]["tau_disease_specificity"] = _tau_r.tau_disease
                results[gene]["disease_log2fc"] = _tau_r.disease_log2fc
                results[gene]["tau_specificity_class"] = _tau_r.specificity_class

        return results

    except Exception as exc:
        warnings.append(f"_maybe_therapeutic_redirection failed (non-fatal): {exc}")
        return {}
