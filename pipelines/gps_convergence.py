"""
pipelines/gps_convergence.py — GPS × OTA convergence analysis.

Cross-references GPS compound reversers against the OTA-ranked target list
to identify two high-value classes:

  converged:        compound's predicted target(s) are in the OTA top-gene list
                    → pharmacological + genetic + mechanistic triangulation

  novel_mechanism:  compound reverses the disease state but has no known targets
                    → phenotypic hit, uncharted causal path, probe for novel biology

Compounds in neither class (known target but not in OTA list) are still
returned as off_target reversers for completeness.

Public API
----------
compute_gps_convergence(
    targets:          list of Tier 4 target records (must have "target_gene" key)
    gps_result:       dict from run_gps_disease_screen (disease_reversers, program_reversers)
    top_n:            how many top OTA targets to use as convergence set (default 200)
) -> dict with keys:
    converged_compounds       list[dict]   — compound + target gene + convergence score
    novel_mechanism_compounds list[dict]   — putative_targets empty / all unknown
    off_target_compounds      list[dict]   — known target, not in OTA list
    convergence_by_program    dict[str, list[dict]]  — per-program breakdown
    n_converged               int
    n_novel_mechanism         int
    warnings                  list[str]
"""
from __future__ import annotations

from typing import Any


def compute_gps_convergence(
    targets: list[dict],
    gps_result: dict[str, Any],
    top_n: int = 200,
) -> dict[str, Any]:
    warnings: list[str] = []

    # Build the OTA convergence set: top_n genes by causal_gamma rank
    ota_set: set[str] = {
        r.get("target_gene", r.get("gene", ""))
        for r in targets[:top_n]
    }
    ota_rank: dict[str, int] = {
        r.get("target_gene", r.get("gene", "")): r.get("rank", i + 1)
        for i, r in enumerate(targets[:top_n])
    }
    ota_gamma: dict[str, float] = {
        r.get("target_gene", r.get("gene", "")): abs(r.get("causal_gamma") or r.get("ota_gamma") or 0.0)
        for r in targets[:top_n]
    }

    # Collect all GPS reversers (disease-level + program-level)
    disease_reversers: list[dict] = list(gps_result.get("disease_reversers") or [])
    program_reversers: dict[str, list[dict]] = dict(gps_result.get("program_reversers") or {})

    def _classify(compound: dict, program_id: str | None = None) -> dict:
        """Classify one GPS hit and return an enriched record."""
        targets_list: list[str] = list(compound.get("putative_targets") or [])
        converged_targets = [t for t in targets_list if t in ota_set]
        has_known_targets = bool(targets_list)
        is_novel = not has_known_targets

        # Convergence score: fraction of predicted targets that are in OTA list,
        # weighted by the OTA rank of the best converged target.
        if converged_targets:
            best_rank  = min(ota_rank.get(t, top_n + 1) for t in converged_targets)
            best_gamma = max(ota_gamma.get(t, 0.0) for t in converged_targets)
            conv_score = round(
                len(converged_targets) / max(len(targets_list), 1)
                * (1.0 - (best_rank - 1) / top_n),
                4,
            )
        else:
            best_rank  = None
            best_gamma = 0.0
            conv_score = 0.0

        return {
            **compound,
            "program_id":          program_id,
            "converged_targets":   converged_targets,
            "is_novel_mechanism":  is_novel,
            "convergence_score":   conv_score,
            "best_ota_rank":       best_rank,
            "best_ota_gamma":      round(best_gamma, 4) if best_gamma else None,
            "evidence_class": (
                "converged"        if converged_targets else
                "novel_mechanism"  if is_novel         else
                "off_target"
            ),
        }

    # Classify all reversers
    all_classified: list[dict] = []

    for hit in disease_reversers:
        all_classified.append(_classify(hit, program_id=None))

    convergence_by_program: dict[str, list[dict]] = {}
    for prog_id, hits in program_reversers.items():
        prog_classified = [_classify(h, program_id=prog_id) for h in hits]
        convergence_by_program[prog_id] = prog_classified
        all_classified.extend(prog_classified)

    # Deduplicate by compound_id, keeping the record with highest convergence_score
    seen: dict[str, dict] = {}
    for rec in all_classified:
        cid = rec.get("compound_id") or rec.get("smiles") or id(rec)
        existing = seen.get(str(cid))
        if existing is None or rec["convergence_score"] > existing["convergence_score"]:
            seen[str(cid)] = rec
    deduped = list(seen.values())

    converged          = sorted([r for r in deduped if r["evidence_class"] == "converged"],
                                key=lambda r: -r["convergence_score"])
    novel_mechanism    = [r for r in deduped if r["evidence_class"] == "novel_mechanism"]
    off_target         = [r for r in deduped if r["evidence_class"] == "off_target"]

    if not deduped:
        warnings.append("GPS convergence: no GPS reversers found — run GPS screen first")
    else:
        warnings.append(
            f"GPS convergence: {len(converged)} converged, "
            f"{len(novel_mechanism)} novel-mechanism, "
            f"{len(off_target)} off-target (top-{top_n} OTA set, "
            f"{len(disease_reversers)} disease + "
            f"{sum(len(v) for v in program_reversers.values())} program reversers)"
        )

    return {
        "converged_compounds":        converged,
        "novel_mechanism_compounds":  novel_mechanism,
        "off_target_compounds":       off_target,
        "convergence_by_program":     convergence_by_program,
        "n_converged":                len(converged),
        "n_novel_mechanism":          len(novel_mechanism),
        "top_n_ota_set":              top_n,
        "warnings":                   warnings,
    }


def annotate_targets_with_gps(
    targets: list[dict],
    convergence_result: dict[str, Any],
) -> None:
    """Back-fill each target record's evidence_summary with its converged compounds.

    Modifies target records in-place.
    """
    # Index converged compounds by target gene
    by_gene: dict[str, list[dict]] = {}
    for comp in convergence_result.get("converged_compounds", []):
        for gene in comp.get("converged_targets", []):
            by_gene.setdefault(gene, []).append(comp)

    # Index novel-mechanism compounds (not tied to a specific gene, attach to top targets)
    novel = convergence_result.get("novel_mechanism_compounds", [])

    for rec in targets:
        gene = rec.get("target_gene", "")
        summary = rec.get("evidence_summary")
        if summary is None:
            continue
        summary["gps_converged_compounds"] = [
            {
                "compound_id":      c.get("compound_id"),
                "smiles":           c.get("smiles"),
                "convergence_score": c.get("convergence_score"),
                "program_id":       c.get("program_id"),
                "z_score":          c.get("z_score") or c.get("log2fc"),
            }
            for c in by_gene.get(gene, [])
        ]

    # Attach novel-mechanism compounds to the top-ranked targets (rank ≤ 20)
    top_20 = sorted(
        [r for r in targets if r.get("rank", 999) <= 20],
        key=lambda r: r.get("rank", 999),
    )
    for rec in top_20:
        summary = rec.get("evidence_summary")
        if summary is None:
            continue
        summary["gps_novel_mechanism_compounds"] = [
            {
                "compound_id": c.get("compound_id"),
                "smiles":      c.get("smiles"),
                "z_score":     c.get("z_score") or c.get("log2fc"),
            }
            for c in novel[:10]
        ]
