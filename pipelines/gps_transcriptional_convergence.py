"""
gps_transcriptional_convergence.py

Links GPS compound reversers to genetic anchor genes via program-space dot product.

Convergence score = Σ_P (|z_rges_P| × |β_{gene→P}|)

where z_rges_P comes from the compound's program_vector (built by gps_disease_screen.py
across all program reversal screens) and β_{gene→P} comes from the gene's top_programs
field (raw Perturb-seq β footprint, |β| ≥ 0.05).

This is the GPS analogue of the OTA formula: γ_{gene→trait} = Σ_P (β_{gene→P} × γ_{P→trait}).
A compound converges with a gene when both have strong signal in the same programs.

NOTE: This is NOT causal evidence. GPS screens use CMAP profiles from cancer cell lines
(MCF7, A549, etc.) which do not match the disease cell types used for target nomination
(endothelial cells for CAD, CD4+ T cells for RA). Convergence scores indicate compounds
worth investigating in the correct cell type — not validation of genetic targets.
Output field: 'convergent_genetic_targets_hypothesis' (annotation only; never used in scoring).
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def compute_gps_genetic_convergence(
    disease_reversers: list[dict],
    program_reversers: dict[str, list[dict]],
    genetic_anchors: list[dict],
    min_z_rges: float = 1.5,
    min_convergence_score: float = 0.0,
) -> dict[str, list[dict]]:
    """
    For each compound, score convergence with every genetic anchor gene.

    Convergence score = Σ_P (|z_rges_P| × |β_{gene→P}|)
      z_rges_P  — compound's reversal strength on program P (from program_vector)
      β_{gene→P} — gene's raw Perturb-seq loading on program P (from top_programs)

    Compounds without a program_vector (disease-state-only reversers) receive no
    hypothesis: they reversed the full disease signature but cannot be attributed to
    specific programs, so the dot product is undefined.

    Returns: {compound_id → list[{gene, convergence_score, shared_programs, n_shared}]}
             sorted by convergence_score descending, capped at top_n_per_compound.
    """
    # Build compound_program_vectors from program_vector field on hits.
    # Only program reversers have meaningful vectors; disease-state reversers
    # may have an empty program_vector if they never appeared in a program screen.
    compound_vectors: dict[str, dict[str, float]] = {}

    for hits in program_reversers.values():
        for hit in hits:
            cid = hit.get("compound_id", "")
            pv  = hit.get("program_vector") or {}
            if cid and pv:
                # Merge: if compound appeared in multiple program screens, union vectors
                existing = compound_vectors.get(cid, {})
                for prog, z in pv.items():
                    if prog not in existing or abs(z) > abs(existing[prog]):
                        existing[prog] = z
                compound_vectors[cid] = existing

    # Also include disease reversers that have a program_vector (compounds that
    # appeared in both disease-state and program screens)
    for hit in disease_reversers:
        cid = hit.get("compound_id", "")
        pv  = hit.get("program_vector") or {}
        if cid and pv and cid not in compound_vectors:
            compound_vectors[cid] = {k: v for k, v in pv.items()}

    # Filter by min_z_rges: only keep programs where |z| is meaningful
    filtered_vectors: dict[str, dict[str, float]] = {}
    for cid, pv in compound_vectors.items():
        fv = {p: z for p, z in pv.items() if abs(z) >= min_z_rges}
        if fv:
            filtered_vectors[cid] = fv

    if not filtered_vectors:
        return {}

    # Build gene β-footprints from anchor top_programs (raw β, set in Session 95)
    anchor_betas: dict[str, dict[str, float]] = {}
    for anchor in genetic_anchors:
        gene = anchor.get("target_gene", "")
        if not gene:
            continue
        # Exclude non-coding artifacts
        if gene.startswith(("AL", "AC", "LINC", "SNHG", "MIR")):
            continue
        tp = anchor.get("top_programs") or {}
        if not isinstance(tp, dict):
            continue
        betas = {prog: abs(float(w)) for prog, w in tp.items()
                 if w is not None and abs(float(w)) >= 0.05}
        if betas:
            anchor_betas[gene] = betas

    if not anchor_betas:
        return {}

    top_n_per_compound = 30
    result: dict[str, list[dict]] = {}

    for cid, pv in filtered_vectors.items():
        convergent: list[dict] = []

        for gene, gene_betas in anchor_betas.items():
            shared = set(pv.keys()) & set(gene_betas.keys())
            if not shared:
                continue

            # Dot product: |z_rges_P| × |β_{gene→P}|
            score = sum(abs(pv[p]) * gene_betas[p] for p in shared)

            if score > min_convergence_score:
                convergent.append({
                    "gene":              gene,
                    "convergence_score": round(score, 4),
                    "shared_programs":   sorted(shared),
                    "n_shared":          len(shared),
                })

        if convergent:
            convergent.sort(key=lambda x: -x["convergence_score"])
            result[cid] = convergent[:top_n_per_compound]

    log.debug(
        "GPS convergence (dot product): %d/%d compounds have convergent anchors",
        len(result), len(filtered_vectors),
    )
    return result


def annotate_reversers_with_convergence(
    disease_reversers: list[dict],
    program_reversers: dict[str, list[dict]],
    genetic_anchors: list[dict] | None,
    **kwargs,
) -> tuple[list[dict], dict[str, list[dict]]]:
    """
    Annotate reversers with convergence hypothesis.
    Adds 'convergent_genetic_targets_hypothesis' to each reverser's annotation dict.

    Disease reversers without a program_vector get an empty list — they reversed
    the full disease signature but lack program-level attribution.
    """
    if not genetic_anchors:
        return disease_reversers, program_reversers

    convergence_map = compute_gps_genetic_convergence(
        disease_reversers=disease_reversers,
        program_reversers=program_reversers,
        genetic_anchors=genetic_anchors,
        **kwargs,
    )

    def _annotate(hit: dict) -> None:
        cid = hit.get("compound_id", "")
        targets = convergence_map.get(cid, [])
        ann = hit.get("annotation")
        if isinstance(ann, dict):
            ann["convergent_genetic_targets_hypothesis"] = targets
        else:
            hit["annotation"] = {"convergent_genetic_targets_hypothesis": targets}

    for hit in disease_reversers:
        _annotate(hit)

    for hits in program_reversers.values():
        for hit in hits:
            _annotate(hit)

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

    seen_prog_compounds: set[str] = set()
    prog_convergent_ids: set[str] = set()
    for hits in program_reversers.values():
        for hit in hits:
            cid = hit.get("compound_id", "")
            seen_prog_compounds.add(cid)
            ann = hit.get("annotation") or {}
            targets = ann.get("convergent_genetic_targets_hypothesis") or []
            if targets:
                prog_convergent_ids.add(cid)
                for t in targets:
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
