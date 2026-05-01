"""
gps_transcriptional_convergence.py

Hypothesis-generation annotation linking GPS reversers to genetic anchor genes via
shared NMF program overlap.

NOTE: This is NOT causal evidence. GPS screens use CMAP profiles from cancer cell lines
(MCF7, A549, etc.) which do not match the disease cell types used for target nomination
(endothelial cells for CAD, CD4+ T cells for RA). Convergence scores indicate compounds
worth investigating in the correct cell type — not validation of genetic targets.
Output field: 'convergent_genetic_targets_hypothesis_hypothesis' (annotation only; never used in scoring).

Convergence score = Σ_P (|reverser_z_rges_on_P| × |anchor_weight_on_P|) / n_shared_programs
Protein-coding filter: excludes lncRNA/pseudogene artifacts (AL*, AC*, LINC* prefixes).
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def compute_gps_genetic_convergence(
    disease_reversers: list[dict],
    program_reversers: dict[str, list[dict]],  # program_id → [{compound_id, z_rges, ...}]
    genetic_anchors: list[dict],               # [{target_gene, top_programs: {prog: weight}}]
    min_z_rges: float = 1.5,
    min_convergence_score: float = 0.1,
) -> dict[str, list[dict]]:
    """
    For each compound (disease + program reversers), find convergent genetic anchors.

    Returns: dict[compound_id → list of {
        "gene": str,
        "convergence_score": float,
        "shared_programs": list[str],
        "n_shared": int,
    }] sorted by convergence_score descending.
    """
    # ------------------------------------------------------------------
    # Build compound_programs: compound_id → {program_id: |z_rges|}
    # Only program reversers get entries here — disease reversers reversed the
    # full signature and cannot be attributed to a specific program, so they
    # receive no convergence hypothesis (avoids degenerate uniform-weight scores).
    # ------------------------------------------------------------------
    compound_programs: dict[str, dict[str, float]] = {}

    # Program reversers: compound appears in a specific program with known z_rges
    for prog_id, hits in program_reversers.items():
        for hit in hits:
            cid = hit.get("compound_id", "")
            if not cid:
                continue
            z = abs(hit.get("z_rges") or hit.get("rges") or 0.0)
            if z < min_z_rges:
                continue
            if cid not in compound_programs:
                compound_programs[cid] = {}
            existing = compound_programs[cid].get(prog_id, 0.0)
            compound_programs[cid][prog_id] = max(existing, z)

    # Disease reversers: phenotypic hits — no per-program z_rges available.
    # Uniform program weights are degenerate (identical for every compound).
    # These compounds are annotated as "no_convergence_hypothesis" elsewhere.

    # ------------------------------------------------------------------
    # Build anchor_programs: gene → {program_id: |weight|}
    # ------------------------------------------------------------------
    anchor_programs: dict[str, dict[str, float]] = {}
    for anchor in genetic_anchors:
        gene = anchor.get("target_gene", "")
        if not gene:
            continue
        tp = anchor.get("top_programs") or {}
        if not isinstance(tp, dict):
            continue
        anchor_programs[gene] = {prog: abs(float(w)) for prog, w in tp.items() if w is not None}

    # ------------------------------------------------------------------
    # Compute convergence scores for compound × anchor pairs.
    # score = Σ_P (z_rges_P × anchor_weight_P)  [unnormalized sum, not mean]
    # Dividing by n_shared suppressed multi-program evidence; summing rewards it.
    # Threshold replaced by top_n_per_compound to avoid scale-sensitivity.
    # ------------------------------------------------------------------
    top_n_per_compound = 30
    result: dict[str, list[dict]] = {}

    for cid, comp_progs in compound_programs.items():
        convergent: list[dict] = []

        for gene, anc_progs in anchor_programs.items():
            # Exclude lncRNA / pseudogene artifacts from convergence annotation
            if gene.startswith(("AL", "AC", "LINC", "SNHG", "MIR")):
                continue
            shared = set(comp_progs.keys()) & set(anc_progs.keys())
            if not shared:
                continue

            n_shared = len(shared)
            score_sum = sum(comp_progs[p] * anc_progs[p] for p in shared)

            if score_sum > 0:
                convergent.append({
                    "gene":              gene,
                    "convergence_score": round(score_sum, 4),
                    "shared_programs":   sorted(shared),
                    "n_shared":          n_shared,
                })

        if convergent:
            convergent.sort(key=lambda x: -x["convergence_score"])
            result[cid] = convergent[:top_n_per_compound]

    log.debug(
        "GPS transcriptional convergence: %d/%d compounds have convergent anchors",
        len(result), len(compound_programs),
    )
    return result


def annotate_reversers_with_convergence(
    disease_reversers: list[dict],
    program_reversers: dict[str, list[dict]],
    genetic_anchors: list[dict] | None,
    **kwargs,
) -> tuple[list[dict], dict[str, list[dict]]]:
    """
    In-place annotate reversers with convergence info.
    Adds 'convergent_genetic_targets_hypothesis' key to each reverser's annotation dict.
    Returns updated (disease_reversers, program_reversers).

    If genetic_anchors is None or empty, returns inputs unchanged.
    """
    if not genetic_anchors:
        return disease_reversers, program_reversers

    convergence_map = compute_gps_genetic_convergence(
        disease_reversers=disease_reversers,
        program_reversers=program_reversers,
        genetic_anchors=genetic_anchors,
        **kwargs,
    )

    # Annotate disease reversers
    for hit in disease_reversers:
        cid = hit.get("compound_id", "")
        convergent_targets = convergence_map.get(cid, [])
        ann = hit.get("annotation")
        if isinstance(ann, dict):
            ann["convergent_genetic_targets_hypothesis"] = convergent_targets
        else:
            hit["annotation"] = {"convergent_genetic_targets_hypothesis": convergent_targets}

    # Annotate program reversers
    for prog_id, hits in program_reversers.items():
        for hit in hits:
            cid = hit.get("compound_id", "")
            convergent_targets = convergence_map.get(cid, [])
            ann = hit.get("annotation")
            if isinstance(ann, dict):
                ann["convergent_genetic_targets_hypothesis"] = convergent_targets
            else:
                hit["annotation"] = {"convergent_genetic_targets_hypothesis": convergent_targets}

    return disease_reversers, program_reversers


def summarize_convergence(
    disease_reversers: list[dict],
    program_reversers: dict[str, list[dict]],
) -> dict:
    """
    Count how many compounds have at least one convergent genetic target.
    Returns: {
        "n_disease_reversers": int,
        "n_disease_convergent": int,
        "n_program_reversers": int,
        "n_program_convergent": int,
        "convergent_pairs": list[{compound_id, gene, score, programs}],
    }
    """
    convergent_pairs: list[dict] = []

    # Disease reversers
    n_disease_convergent = 0
    for hit in disease_reversers:
        ann = hit.get("annotation") or {}
        targets = ann.get("convergent_genetic_targets_hypothesis") or []
        if targets:
            n_disease_convergent += 1
            cid = hit.get("compound_id", "")
            for t in targets:
                convergent_pairs.append({
                    "compound_id": cid,
                    "gene":        t["gene"],
                    "score":       t["convergence_score"],
                    "programs":    t["shared_programs"],
                })

    # Program reversers — count unique compound_ids with convergence
    seen_prog_compounds: set[str] = set()
    prog_convergent_ids: set[str] = set()
    for prog_id, hits in program_reversers.items():
        for hit in hits:
            cid = hit.get("compound_id", "")
            seen_prog_compounds.add(cid)
            ann = hit.get("annotation") or {}
            targets = ann.get("convergent_genetic_targets_hypothesis") or []
            if targets:
                prog_convergent_ids.add(cid)
                for t in targets:
                    # Avoid duplicate pairs already added from disease_reversers
                    pair = {
                        "compound_id": cid,
                        "gene":        t["gene"],
                        "score":       t["convergence_score"],
                        "programs":    t["shared_programs"],
                    }
                    if pair not in convergent_pairs:
                        convergent_pairs.append(pair)

    convergent_pairs.sort(key=lambda x: -x["score"])

    return {
        "n_disease_reversers":  len(disease_reversers),
        "n_disease_convergent": n_disease_convergent,
        "n_program_reversers":  len(seen_prog_compounds),
        "n_program_convergent": len(prog_convergent_ids),
        "convergent_pairs":     convergent_pairs,
    }
