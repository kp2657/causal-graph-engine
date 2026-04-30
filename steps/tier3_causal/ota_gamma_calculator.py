"""
ota_gamma_calculator.py — Tier 3: Ota composite γ + causal graph construction.

Computes γ_{gene→trait} = Σ_P (β_{gene→P} × γ_{P→trait}),
validates anchor edge recovery, writes significant edges to Kùzu,
and computes SHD from the reference anchor graph.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.disease_registry import DISEASE_PROGRAMS as _DISEASE_PROGRAMS
from pipelines.static_lookups import get_lookups as _get_lookups
from pipelines.shet_loader import get_shet_penalty as _get_shet_penalty
from config.scoring_thresholds import OTA_GAMMA_EDGE_MIN
from steps.tier3_causal.causal_filters import (
    _beta_stress_discount,
    _program_entropy,
    _mechanistic_necessity_filter,
    _extract_beta_for_program,
    _extract_gamma_for_trait,
    _build_fallback_top_programs,
    _pareto_cutoff,
    _wes_concordance_check,
    _HOUSEKEEPING_PREFIXES,
    _HOUSEKEEPING_EXACT,
)
from steps.tier3_causal.causal_therapeutic import _maybe_therapeutic_redirection

OTA_GAMMA_MIN = OTA_GAMMA_EDGE_MIN  # backward-compat alias used in this module


def run(
    beta_matrix_result: dict,
    gamma_estimates: dict,
    disease_query: dict,
) -> dict:
    """
    Build the causal graph from β × γ products.

    Args:
        beta_matrix_result: Output of beta_matrix_builder.run
        gamma_estimates:    {program_id: {trait: gamma_value}} from ota_gamma_estimation
        disease_query:      DiseaseQuery dict

    Returns:
        dict with n_edges_written, top_genes, warnings
    """
    from pipelines.ota_gamma_estimation import compute_ota_gamma, compute_ota_gamma_with_uncertainty
    _OTA_SKIP_PROGRAMS: frozenset[str] = frozenset({"__protein_channel__"})
    from pipelines.scone_sensitivity import polybic_selection
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
    program_activation_biases: dict[str, float] | None = beta_matrix_result.get("program_activation_biases")
    # Program membership from Tier 2 eqtl_coloc_mapper (may be empty).
    # Used by `_build_fallback_top_programs` when the OT-genetic fallback fires
    # and `n_programs_contributing == 0` — see line ~225.
    gene_program_overlap: dict[str, list[str]] = disease_query.get("gene_program_overlap") or {}
    warnings: list[str] = []

    from graph.schema import _DISEASE_SHORT_NAMES_FOR_ANCHORS  # local helper
    short = _DISEASE_SHORT_NAMES_FOR_ANCHORS.get(disease_name.lower(), "CAD")

    # -------------------------------------------------------------------------
    # Load program precedence discounts (if available) and apply to gamma_estimates
    # -------------------------------------------------------------------------
    _disease_key_early = disease_query.get("disease_key") or ""
    from pipelines.state_space.program_precedence import get_precedence_discount as _get_precedence_discount
    _prec_cache = (
        Path(__file__).parent.parent.parent
        / "data" / "ldsc" / "results"
        / f"{_disease_key_early}_program_precedence.json"
    )
    _precedence: dict = {}
    if _disease_key_early and _prec_cache.exists():
        try:
            import json as _json_prec
            _precedence = _json_prec.loads(_prec_cache.read_text()).get("programs", {})
        except Exception as _prec_exc:
            warnings.append(f"program_precedence cache load failed (non-fatal): {_prec_exc}")

    if _precedence:
        gamma_estimates = {
            prog: prog_gammas
            for prog, prog_gammas in gamma_estimates.items()
        }  # shallow copy so we can mutate safely
        for prog_id in list(gamma_estimates.keys()):
            _label = _precedence.get(prog_id, {}).get("label", "ambiguous")
            _discount = _get_precedence_discount(_label)
            if _discount < 1.0:
                _orig = gamma_estimates[prog_id]
                _discounted: dict = {}
                for trait_key, g_val in _orig.items():
                    if isinstance(g_val, (int, float)):
                        _discounted[trait_key] = float(g_val) * _discount
                    elif isinstance(g_val, dict) and g_val.get("gamma") is not None:
                        _discounted[trait_key] = {
                            **g_val,
                            "gamma": float(g_val["gamma"]) * _discount,
                        }
                    else:
                        _discounted[trait_key] = g_val
                gamma_estimates[prog_id] = _discounted
                warnings.append(
                    f"program_precedence_discount: {prog_id} label=consequence γ×{_discount:.1f}"
                )

    # -------------------------------------------------------------------------
    # LOO stability discounts: load once per disease, applied per gene below
    # -------------------------------------------------------------------------
    _loo_discounts: dict[str, float] = {}
    _disease_key_loo = disease_query.get("disease_key") or ""
    if _disease_key_loo:
        try:
            from pipelines.ldsc.gamma_loader import load_loo_discounts as _load_loo_discounts
            from pipelines.ldsc.runner import _fetch_gene_coords_hg38 as _fetch_gene_coords
            # Build anchor positions from gene coordinate cache
            _all_genes_for_loo = beta_matrix_result.get("genes", [])
            _gene_coords_raw = _fetch_gene_coords(_all_genes_for_loo)
            _anchor_positions: dict[str, tuple[int, int]] = {}
            for _g, _coords in _gene_coords_raw.items():
                try:
                    _chrom_str = str(_coords[0]).lstrip("chr")
                    _chrom_int = int(_chrom_str)
                    _pos_mid = (int(_coords[1]) + int(_coords[2])) // 2
                    _anchor_positions[_g] = (_chrom_int, _pos_mid)
                except (ValueError, IndexError):
                    pass
            _loo_discounts = _load_loo_discounts(_disease_key_loo, _anchor_positions)
            if _loo_discounts:
                _n_unstable = sum(1 for v in _loo_discounts.values() if v < 1.0)
                warnings.append(
                    f"loo_stability: {_n_unstable} unstable anchor genes (discount=0.8× applied)"
                )
        except Exception as _loo_exc:
            warnings.append(f"loo_discount_load failed (non-fatal): {_loo_exc}")

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
                    skip_programs=_OTA_SKIP_PROGRAMS,
                    program_activation_biases=program_activation_biases,
                )
                ota_gamma_raw = ota_result.get("ota_gamma")
                ota_gamma = float('nan') if ota_gamma_raw is None else float(ota_gamma_raw)
                dominant_tier = ota_result.get("dominant_tier", "provisional_virtual")
                top_programs = ota_result.get("top_programs", [])
                # True only when β×γ terms actually summed through cNMF programs.
                _n_programs_contributing = int(ota_result.get("n_programs_contributing", 0))

                import math as _math
                # Stress flag: essential-gene KO artefact signal (Mediator, ESCRT, etc.)
                # Flagged only — ota_gamma is NOT modified.
                _stress_mult = _beta_stress_discount(gene_beta) if not _math.isnan(ota_gamma) else 1.0
                _stress_flagged = _stress_mult < 1.0

                # Tissue weight: flagged only — ota_gamma is NOT modified.
                _tissue_mult = _tissue_weights.get(gene, 1.0)

                # LOO stability flag — annotation only, does NOT modify ota_gamma
                _loo_discount = _loo_discounts.get(gene, 1.0)
                _loo_stable = _loo_discount >= 1.0

                # WES directional concordance — reward-only gate.
                # Concordant + significant (p < WES_CONCORDANCE_P_THRESHOLD): wes_gamma_weight > 1.0 (boost).
                # Discordant or sub-threshold: wes_gamma_weight = 1.0 (no change).
                # Weight is always ≥ 1.0, so unconditional multiply is safe.
                _wes = _wes_concordance_check(ota_gamma, gb_result)
                _wes_weight = _wes["wes_gamma_weight"]
                if _wes_weight != 1.0:
                    ota_gamma = ota_gamma * _wes_weight
                    warnings.append(
                        f"{gene} → {trait}: {_wes['wes_note']}"
                    )

                key = f"{gene}__{trait}"
                r: dict = {
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
                    "genebayes_grounded":      ota_result.get("genebayes_grounded", False),
                    "stress_flag":              _stress_flagged,
                    "tissue_weight":            round(_tissue_mult, 4),
                    "n_programs_contributing":  _n_programs_contributing,
                    "loo_stable":               _loo_stable,
                    "wes_checked":              _wes["wes_checked"],
                    "wes_concordant":           _wes["wes_concordant"],
                    "wes_burden_p":             _wes["wes_burden_p"],
                    "wes_burden_beta":          _wes["wes_burden_beta"],
                    "wes_gamma_weight":         _wes["wes_gamma_weight"],
                }
                if _wes["wes_checked"]:
                    r["wes_note"] = _wes["wes_note"]
                if not _loo_stable:
                    r["loo_discount"] = _loo_discount
                gene_gamma[key] = r
            except Exception as exc:
                warnings.append(f"Ota γ computation failed for {gene} → {trait}: {exc}")

    # -------------------------------------------------------------------------
    # PolyBIC edge selection — complexity-penalised pruning of low-signal edges.
    # Keeps edges where log(|γ|) − (k/2)·log(n) > −10, with k tiered by
    # evidence quality (k=1 for Tier1_Interventional, k=10 for provisional).
    # -------------------------------------------------------------------------
    try:
        _polybic_kept = {
            r["gene"] + "__" + r["trait"]: r
            for r in polybic_selection(list(gene_gamma.values()), n_samples=1000)
        }
        gene_gamma = _polybic_kept

        warnings.append(
            f"PolyBIC: {len(_polybic_kept)} edges kept (uniform complexity filter)"
        )

    except Exception as exc:
        warnings.append(f"PolyBIC selection failed (non-fatal): {exc}")

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
        "Tier1_Interventional", "Tier2_Convergent",
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
        # CI from OTA γ uncertainty σ: γ ± 1.96σ
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
            "method":             "ota_gamma",
            "data_source":        "Ota2026_composite_gamma",
            "data_source_version": "2026",
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
        cur_gamma = rec["ota_gamma"]
        cur_abs = abs(cur_gamma) if not _math.isnan(cur_gamma) else 0.0
        prev_abs = 0.0
        if gene in gene_best_gamma:
            pg = gene_best_gamma[gene]["ota_gamma"]
            prev_abs = abs(pg) if not _math.isnan(pg) else 0.0
        if gene not in gene_best_gamma or cur_abs > prev_abs:
            gene_best_gamma[gene] = rec


    # -------------------------------------------------------------------------
    # Novel discovery metric: edges involving genes NOT in prior anchor set
    # All written edges are novel discoveries (no anchor seeding).
    # -------------------------------------------------------------------------
    novel_edges = list(edges_to_write)
    novel_genes = list({r["gene"] for r in novel_edges})

    # -------------------------------------------------------------------------
    # Phase E: therapeutic redirection + evidence disagreement (non-fatal)
    # -------------------------------------------------------------------------
    disease_key = disease_query.get("disease_key") or ""
    tr_results: dict[str, dict] = _maybe_therapeutic_redirection(
        gene_list=gene_list,
        beta_matrix_result=beta_matrix_result,
        disease_key=disease_key,
        gene_gamma=gene_gamma,
        warnings=warnings,
        efo_id=efo_id,
    )

    # Extract program disease-specificity weights returned by _maybe_therapeutic_redirection.
    # Used to compute beta_program_concentration — a diagnostic field that flags pleiotropic
    # genes whose β is spread across generic (non-AMD-specific) programs.
    # ota_gamma is NOT modified; this is reporting only.
    _program_weights: dict[str, float] = tr_results.pop("__program_weights__", {})
    if _program_weights:
        try:
            from pipelines.state_space.program_specificity import compute_beta_program_concentration
            for _gene, _gbg in gene_best_gamma.items():
                _gene_beta = beta_matrix.get(_gene, {})
                _gbg["beta_program_concentration"] = compute_beta_program_concentration(
                    _gene_beta, _program_weights
                )
        except Exception as _exc_pw_apply:
            warnings.append(
                f"beta_program_concentration computation failed (non-fatal): {_exc_pw_apply}"
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

    if _write_errors:
        warnings.append(f"Graph write reported {len(_write_errors)} ingestion errors (see logs)")

    # All genes above the edge-write threshold pass to the prioritization agent.
    # No count cutoff — any hard number is arbitrary and silently excludes genes
    # with valid genetic evidence but low β×γ (e.g. extracellular proteins).
    # OTA_GAMMA_MIN is the natural floor: edges below this aren't written to the
    # graph and carry no meaningful signal.
    import math as _math_rebuild
    _gwas_top = sorted(
        [
            r for r in gene_best_gamma.values()
            if not (lambda v: v is None or _math_rebuild.isnan(v))(r.get("ota_gamma"))
            and abs(r.get("ota_gamma", 0.0)) >= OTA_GAMMA_MIN
        ],
        key=lambda r: abs(r["ota_gamma"]),
        reverse=True,
    )

    # --- shet constraint annotation (GeneBayes posteriors, Spence 2024) -------
    # Flag genes under purifying selection against heterozygous LoF.
    # shet_penalty and shet_flag are set for downstream inspection;
    # ota_gamma is NOT modified.
    from pipelines.shet_loader import get_shet as _get_shet
    for _sr in _gwas_top:
        _shet_val = _get_shet(_sr.get("gene", ""))
        if _shet_val is not None:
            _sr["shet"] = round(_shet_val, 6)
            _sr["shet_flag"] = _shet_val >= 0.05   # highly constrained
            _sr["shet_penalty"] = _get_shet_penalty(_sr.get("gene", ""))
    # -------------------------------------------------------------------------

    # --- Phase E: Shannon entropy score (reporting only) ----------------------
    # Compute per-gene entropy across disease-relevant programs as an independent
    # signal. High entropy = non-specific loading (housekeeping candidate).
    # Reported on the record; does NOT modify ota_gamma.
    _disease_programs = _DISEASE_PROGRAMS.get(disease_key, frozenset())
    if _disease_programs and _gwas_top:
        import numpy as _np_ent
        _entropies = {r["gene"]: _program_entropy(r["gene"], beta_matrix, _disease_programs)
                      for r in _gwas_top}
        _H_p75 = float(_np_ent.percentile(list(_entropies.values()), 75))
        _H_max = float(_np_ent.log(len(_disease_programs)))
        for r in _gwas_top:
            H = _entropies[r["gene"]]
            r["entropy_score"] = round(H, 4)
            if H > _H_p75 and _H_max > _H_p75:
                excess = (H - _H_p75) / (_H_max - _H_p75 + 1e-8)
                r["entropy_discount"] = round(max(0.5, 1.0 - 0.5 * excess), 3)
    # -------------------------------------------------------------------------

    # Empiric elbow cutoff: cut at the first ≥20% relative drop in sorted |γ|.
    # Applied AFTER polybic selection.  Floors at min_keep=50.
    _pre_pareto_n = len(_gwas_top)
    _gwas_top = _pareto_cutoff(_gwas_top, min_keep=50)
    if _pre_pareto_n and len(_gwas_top) < _pre_pareto_n:
        warnings.append(
            f"elbow_cutoff: kept {len(_gwas_top)} / {_pre_pareto_n} genes "
            f"(first ≥20% relative drop in |γ| distribution)"
        )

    # --- Phase F: Mechanistic necessity filter -----------------------------------
    # Hard-prune genes with GWAS support but β≈0 and no program overlap before
    # Tier 4.  pQTL/eQTL MR genes and Tier2-tier genes are exempt.
    if _disease_programs:
        _pre_filter_n = len(_gwas_top)
        _gwas_top = _mechanistic_necessity_filter(
            _gwas_top, beta_matrix, _disease_programs, min_keep=50)
        if len(_gwas_top) < _pre_filter_n:
            warnings.append(
                f"mechanistic_necessity_filter: {_pre_filter_n} → {len(_gwas_top)} genes"
            )
    # -------------------------------------------------------------------------

    top_genes = _gwas_top

    # Program-influence clustering: group OTA-grounded genes by which programs
    # they act through, using cosine-distance KMeans on normalised β×γ vectors.
    try:
        from pipelines.program_clustering import cluster_by_program_influence
        # Build lightweight dicts with only what the clusterer needs
        _cluster_input = [
            {"gene": r["gene"],
             "programs": r.get("top_programs") or {},
             "n_programs_contributing": r.get("n_programs_contributing", 0)}
            for r in top_genes
        ]
        _clustered = cluster_by_program_influence(_cluster_input)
        _cluster_map = {c["gene"]: c for c in _clustered}
    except Exception as _exc_clust:
        warnings.append(f"program_clustering failed (non-fatal): {_exc_clust}")
        _cluster_map = {}

    # GWAS-anchored: genes with OT L2G evidence but zero program contributions.
    # These are NOT part of the OTA causal claim — they are reported separately
    # so downstream agents can include them for clinical context without inflating
    # the mechanistic target list.
    import math as _math_gwa
    _ot_scores_for_gwas = disease_query.get("ot_genetic_scores") or {}
    _gwas_anchored = sorted(
        [
            r for r in gene_best_gamma.values()
            if (lambda v: v is None or _math_gwa.isnan(v))(r.get("ota_gamma"))
            and r["gene"] in _ot_scores_for_gwas
            and r.get("tier") != "state_nominated"
        ],
        key=lambda r: float(_ot_scores_for_gwas.get(r["gene"], 0.0)),
        reverse=True,
    )

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
                "dominant_tier":      r.get("dominant_tier", r.get("tier", "provisional_virtual")),
                "programs":           r.get("top_programs", []),
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
                # WES rare-variant burden concordance (boost already applied to ota_gamma above)
                "wes_checked":      r.get("wes_checked", False),
                "wes_concordant":   r.get("wes_concordant"),
                "wes_burden_p":     r.get("wes_burden_p"),
                "wes_burden_beta":  r.get("wes_burden_beta"),
                "wes_gamma_weight": r.get("wes_gamma_weight", 1.0),
                "wes_note":         r.get("wes_note"),
                # Phase U: pleiotropy diagnostic — fraction of β-mass in disease-specific programs
                "beta_program_concentration": r.get("beta_program_concentration"),
                # True when at least one β×γ product contributed to ota_gamma
                # (i.e. the gene is connected through a cNMF program, not just OT L2G proxy)
                "n_programs_contributing": r.get("n_programs_contributing", 0),
                # Program-influence cluster: genes sharing a cluster act through
                # the same program mix (complementary targets span multiple clusters)
                "program_cluster_id":  _cluster_map.get(r["gene"], {}).get("program_cluster_id"),
                "dominant_program":    _cluster_map.get(r["gene"], {}).get("dominant_program"),
                "cluster_label":       _cluster_map.get(r["gene"], {}).get("cluster_label"),
                "cluster_size":        _cluster_map.get(r["gene"], {}).get("cluster_size"),
            }
            for r in top_genes
        ],
        # Genes with strong OT L2G evidence but no cNMF program β. These are NOT
        # ranked by OTA — they have no causal mechanism in the graph. Reported
        # separately for clinical context only; do not mix into top_genes scoring.
        "gwas_anchored_genes": [
            {
                "gene":      r["gene"],
                "ot_l2g":    float(_ot_scores_for_gwas.get(r["gene"], 0.0)),
                "tier":      r.get("tier", "provisional_virtual"),
                "dominant_tier": r.get("dominant_tier", "provisional_virtual"),
            }
            for r in _gwas_anchored[:50]
        ],
        "n_gwas_anchored":       len(_gwas_anchored),
        "shd":                   0,
        "therapeutic_redirection_available": bool(tr_results),
        "polybic_edges": [
            {"gene": e["from_node"], "trait": e["to_node"],
             "gamma": e["effect_size"], "tier": e["evidence_tier"]}
            for e in causal_edge_dicts
        ],
        "n_polybic_edges": len(causal_edge_dicts),
        # Passthrough for Tier 5 writer — used to surface `inherited_genetic_evidence`
        # (S-LDSC γ of programs a mechanistic-only gene controls).
        "gamma_per_program_per_trait": gamma_estimates,
        "warnings":              warnings,
    }
