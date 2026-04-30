"""
causal_therapeutic.py — Therapeutic redirection helper for Tier 3 causal discovery.

Extracted from ota_gamma_calculator.py to keep the main agent focused on
the OTA γ computation and graph construction.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

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

    Uses beta_matrix betas (already computed by beta_matrix_builder) as
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
                        das = si_for_gene.get("disease_axis_score", 0.0)
                        _has_perturb = das >= 1e-3 and math.isfinite(gene_gamma_ota)
                        _has_genetic = (
                            not _has_perturb
                            and math.isfinite(gene_gamma_ota)
                            and abs(gene_gamma_ota) >= 0.05
                        )
                        if not _has_perturb and not _has_genetic:
                            continue
                        _tr_track = "perturb_seq" if _has_perturb else "genetic"
                        per_ct_by_gene[gene][cell_type] = {
                            "redirection": 0.0,
                            "nmf_redirection": 0.0,
                            "state_direct": 0.0,
                            "state_influence_score": das if _has_perturb else 0.0,
                            "directionality": si_for_gene.get("directionality", 0),
                            "n_programs": 0,
                            "pooled_fraction": 0.0,
                            "evidence_tiers": [],
                            "provenance": [],
                            "tr_track": _tr_track,
                            "entry_score":       si_for_gene.get("entry_score", 0.0),
                            "persistence_score": si_for_gene.get("persistence_score", 0.0),
                            "recovery_score":    si_for_gene.get("recovery_score", 0.0),
                            "boundary_score":    si_for_gene.get("boundary_score", 0.0),
                            "mechanistic_category": si_for_gene.get("mechanistic_category", "unknown"),
                        }
                        try:
                            T_bl = trans.get("transition_matrix")
                            state_labels = trans.get("state_labels", [])
                            label_to_idx = {str(s): i for i, s in enumerate(state_labels)}
                            p_idxs = [label_to_idx[s] for s in trans.get("pathologic_basin_ids", []) if s in label_to_idx]
                            h_idxs = [label_to_idx[s] for s in trans.get("healthy_basin_ids", []) if s in label_to_idx]
                            if T_bl is not None and p_idxs and h_idxs:
                                from pipelines.state_space.therapeutic_redirection import (
                                    compute_stability_score, compute_genetic_tr,
                                )
                                if _has_perturb:
                                    sd = compute_state_direct_redirection(
                                        gene_beta=float("nan"),
                                        disease_axis_score=das,
                                        directionality=si_for_gene.get("directionality", 0),
                                        gamma_ota=gene_gamma_ota,
                                        T_baseline=T_bl,
                                        path_idxs=p_idxs,
                                        healthy_idxs=h_idxs,
                                    )
                                    stab = compute_stability_score(
                                        T_bl, 1.0, das, p_idxs, h_idxs, n_iterations=20
                                    )
                                    prov_tag = f"state_direct(das={das:.3f},γ={gene_gamma_ota:.3f},Δ={sd:.4f},stab={stab:.2f})"
                                else:
                                    sd = compute_genetic_tr(
                                        gamma_ota=gene_gamma_ota,
                                        T_baseline=T_bl,
                                        path_idxs=p_idxs,
                                        healthy_idxs=h_idxs,
                                    )
                                    stab = 1.0
                                    prov_tag = f"genetic_tr(γ={gene_gamma_ota:.3f},Δ={sd:.4f})"
                                per_ct_by_gene[gene][cell_type]["redirection"] = sd
                                per_ct_by_gene[gene][cell_type]["state_direct"] = sd
                                per_ct_by_gene[gene][cell_type]["stability"] = stab
                                if sd > 0:
                                    per_ct_by_gene[gene][cell_type]["provenance"].append(prov_tag)
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
                        # Has NMF loadings but no Perturb-seq beta.
                        # Two-track: use Perturb-seq DAS if available, else genetic-track TR.
                        das_fb = si_for_gene.get("disease_axis_score", 0.0)
                        _fb_perturb = das_fb >= 1e-3 and math.isfinite(gene_gamma_ota)
                        _fb_genetic = (
                            not _fb_perturb
                            and math.isfinite(gene_gamma_ota)
                            and abs(gene_gamma_ota) >= 0.05
                        )
                        if not _fb_perturb and not _fb_genetic:
                            continue
                        _fb_track = "perturb_seq" if _fb_perturb else "genetic"
                        per_ct_by_gene[gene][cell_type] = {
                            "redirection": 0.0, "nmf_redirection": 0.0,
                            "state_direct": 0.0,
                            "state_influence_score": das_fb if _fb_perturb else 0.0,
                            "directionality": si_for_gene.get("directionality", 0),
                            "n_programs": 0, "pooled_fraction": 0.0,
                            "evidence_tiers": [], "provenance": [],
                            "tr_track": _fb_track,
                            "entry_score":       si_for_gene.get("entry_score", 0.0),
                            "persistence_score": si_for_gene.get("persistence_score", 0.0),
                            "recovery_score":    si_for_gene.get("recovery_score", 0.0),
                            "boundary_score":    si_for_gene.get("boundary_score", 0.0),
                            "mechanistic_category": si_for_gene.get("mechanistic_category", "unknown"),
                        }
                        try:
                            T_bl_fb = trans.get("transition_matrix")
                            sl_fb = trans.get("state_labels", [])
                            li_fb = {str(s): i for i, s in enumerate(sl_fb)}
                            p_fb = [li_fb[s] for s in trans.get("pathologic_basin_ids", []) if s in li_fb]
                            h_fb = [li_fb[s] for s in trans.get("healthy_basin_ids", []) if s in li_fb]
                            if T_bl_fb is not None and p_fb and h_fb:
                                from pipelines.state_space.therapeutic_redirection import (
                                    compute_stability_score, compute_genetic_tr,
                                )
                                if _fb_perturb:
                                    sd_fb = compute_state_direct_redirection(
                                        gene_beta=float("nan"),
                                        disease_axis_score=das_fb,
                                        directionality=si_for_gene.get("directionality", 0),
                                        gamma_ota=gene_gamma_ota,
                                        T_baseline=T_bl_fb,
                                        path_idxs=p_fb,
                                        healthy_idxs=h_fb,
                                    )
                                    stab_fb = compute_stability_score(
                                        T_bl_fb, 1.0, das_fb, p_fb, h_fb, n_iterations=20
                                    )
                                    prov_fb = f"state_direct(das={das_fb:.3f},γ={gene_gamma_ota:.3f},Δ={sd_fb:.4f},stab={stab_fb:.2f})"
                                else:
                                    sd_fb = compute_genetic_tr(
                                        gamma_ota=gene_gamma_ota,
                                        T_baseline=T_bl_fb,
                                        path_idxs=p_fb,
                                        healthy_idxs=h_fb,
                                    )
                                    stab_fb = 1.0
                                    prov_fb = f"genetic_tr(γ={gene_gamma_ota:.3f},Δ={sd_fb:.4f})"
                                per_ct_by_gene[gene][cell_type]["redirection"] = sd_fb
                                per_ct_by_gene[gene][cell_type]["state_direct"] = sd_fb
                                per_ct_by_gene[gene][cell_type]["stability"] = stab_fb
                                if sd_fb > 0:
                                    per_ct_by_gene[gene][cell_type]["provenance"].append(prov_fb)
                        except Exception:
                            pass
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
            for gene in gene_list:
                per_ct = per_ct_by_gene.get(gene, {})
                if per_ct:
                    # Capture tr_track from per-cell-type dicts before aggregation
                    _ct_tracks = [
                        v.get("tr_track")
                        for v in per_ct.values()
                        if isinstance(v, dict) and "tr_track" in v
                    ]
                    _gene_tr_track = (
                        "genetic" if any(t == "genetic" for t in _ct_tracks) else
                        "perturb_seq" if _ct_tracks else None
                    )
                    try:
                        tr_result = compute_therapeutic_redirection(
                            gene, disease_key, per_ct,
                            genetic_grounding=mr_gamma_by_gene.get(gene, 0.0),
                        )
                        if gene not in per_ct_by_gene:
                            per_ct_by_gene[gene] = {}
                        per_ct_by_gene[gene]["__tr_result__"] = tr_result
                        per_ct_by_gene[gene]["__tr_track__"] = _gene_tr_track
                    except Exception:
                        pass

        except Exception as exc_ss:
            warnings.append(f"therapeutic_redirection state_space step failed (non-fatal): {exc_ss}")

        # Assemble final results (GWAS instruments + state-nominated)
        _all_scored_genes = list(gene_list)
        results: dict[str, dict] = {}
        if _computed_program_weights:
            results["__program_weights__"] = _computed_program_weights
        for gene in _all_scored_genes:
            _gene_data  = per_ct_by_gene.get(gene, {})
            tr_result   = _gene_data.pop("__tr_result__", None)
            _tr_track   = _gene_data.pop("__tr_track__", None)
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

            if tr_result is not None or disagreement or ctrl_ann is not None:
                tr_dict = tr_result.model_dump() if tr_result is not None else None
                if tr_dict is not None and _tr_track is not None:
                    tr_dict["tr_track"] = _tr_track
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
