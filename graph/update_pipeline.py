"""
graph/update_pipeline.py — Incremental graph update pipeline.

Orchestrates periodic refreshes of the causal graph:
  1. GWAS refresh — new loci from GWAS Catalog + OpenGWAS
  2. Literature scan — new PubMed abstracts for anchor genes
  3. Clinical trial update — status changes for tracked drugs
  4. Full re-run — complete Ota pipeline (quarterly)

Each update type produces a diff (new_edges, demoted_edges) that is
written to the graph via mcp_servers.graph_db_server.
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

UpdateType = Literal["gwas", "literature", "clinical_trials", "full"]


# ---------------------------------------------------------------------------
# Per-update-type runners
# ---------------------------------------------------------------------------

def _run_gwas_refresh(disease_name: str) -> dict:
    """
    Fetch new GWAS associations and check for novel loci.

    Returns:
        {new_edges, removed_edges, n_new_loci, source}
    """
    from mcp_servers.gwas_genetics_server import (
        get_gwas_catalog_associations,
        get_gwas_catalog_studies,
    )

    logger.info("[GWAS_REFRESH] Fetching GWAS Catalog for %s", disease_name)

    try:
        assoc_result = get_gwas_catalog_associations(disease_name, max_results=200)
        associations = assoc_result.get("associations", [])
    except Exception as exc:
        logger.warning("[GWAS_REFRESH] GWAS Catalog fetch failed: %s", exc)
        associations = []

    # Convert genome-wide significant hits to provisional edges
    new_edges: list[dict] = []
    for assoc in associations:
        pval = assoc.get("p_value")
        gene  = assoc.get("mapped_gene") or assoc.get("gene")
        if not gene:
            continue
        try:
            if float(pval) > 5e-8:
                continue
        except (TypeError, ValueError):
            continue

        new_edges.append({
            "from_node":     gene.split(", ")[0].strip(),
            "to_node":       disease_name,
            "from_type":     "gene",
            "evidence_tier": "Tier3_Provisional",
            "edge_method":   "gwas_catalog_refresh",
            "effect_size":   assoc.get("beta") or assoc.get("or_or_beta"),
            "e_value":       None,
            "data_source":   "GWAS Catalog",
            "graph_version": "refresh",
            "is_demoted":    False,
        })

    return {
        "new_edges":    new_edges,
        "removed_edges": [],
        "n_new_loci":   len(new_edges),
        "source":       "gwas_catalog",
    }


def _run_literature_refresh(disease_name: str) -> dict:
    """
    Scan PubMed for new publications on anchor genes.

    Returns:
        {new_pmids, n_new_papers, flagged_genes}
    """
    from mcp_servers.literature_server import search_pubmed, list_anchor_papers

    logger.info("[LIT_REFRESH] Scanning PubMed for %s", disease_name)

    anchor_papers = list_anchor_papers()
    existing_pmids = {p.get("pmid") for p in anchor_papers}

    try:
        result = search_pubmed(
            f"{disease_name} causal genetic Mendelian randomization",
            max_results=50,
        )
        papers = result.get("papers", [])
    except Exception as exc:
        logger.warning("[LIT_REFRESH] PubMed search failed: %s", exc)
        papers = []

    new_pmids = [
        p.get("pmid") for p in papers
        if p.get("pmid") and p.get("pmid") not in existing_pmids
    ]

    # Flag genes mentioned in new high-impact papers
    flagged_genes: list[str] = []
    for p in papers:
        title = (p.get("title") or "").upper()
        for gene in ["PCSK9", "LDLR", "HMGCR", "TET2", "DNMT3A", "IL6R", "CHIP"]:
            if gene in title and p.get("pmid") not in existing_pmids:
                if gene not in flagged_genes:
                    flagged_genes.append(gene)

    return {
        "new_pmids":     new_pmids,
        "n_new_papers":  len(new_pmids),
        "flagged_genes": flagged_genes,
        "new_edges":     [],  # literature refresh does not directly add edges
    }


def _run_clinical_trials_refresh(disease_name: str) -> dict:
    """
    Check for status changes in tracked clinical trials.

    Returns:
        {updated_trials, n_completed, n_terminated}
    """
    from mcp_servers.clinical_trials_server import search_clinical_trials

    logger.info("[TRIALS_REFRESH] Checking trial status for %s", disease_name)

    try:
        result = search_clinical_trials(
            condition=disease_name,
            status=["COMPLETED", "TERMINATED", "ACTIVE_NOT_RECRUITING"],
            max_results=100,
        )
        trials = result.get("trials", [])
    except Exception as exc:
        logger.warning("[TRIALS_REFRESH] ClinicalTrials fetch failed: %s", exc)
        trials = []

    n_completed   = sum(1 for t in trials if t.get("status") == "COMPLETED")
    n_terminated  = sum(1 for t in trials if t.get("status") == "TERMINATED")

    updated_trials = [
        {
            "nct_id": t.get("nct_id"),
            "status": t.get("status"),
            "drug":   t.get("intervention"),
        }
        for t in trials
    ]

    return {
        "updated_trials": updated_trials,
        "n_completed":    n_completed,
        "n_terminated":   n_terminated,
        "new_edges":      [],
    }


def _run_full_pipeline(disease_name: str) -> dict:
    """
    Trigger a full Ota pipeline re-run via the PI orchestrator.

    Returns the full analyze_disease output.
    """
    from orchestrator.pi_orchestrator import analyze_disease

    logger.info("[FULL_PIPELINE] Starting full re-run for %s", disease_name)

    try:
        result = analyze_disease(disease_name)
        return {
            "status":    "OK",
            "n_edges":   result.get("n_edges", 0),
            "new_edges": result.get("edges", []),
            "output":    result,
        }
    except ValueError as exc:
        # Anchor recovery gate failure
        logger.error("[FULL_PIPELINE] Halted: %s", exc)
        return {
            "status":    "HALTED",
            "reason":    str(exc),
            "new_edges": [],
        }
    except Exception as exc:
        logger.error("[FULL_PIPELINE] Unexpected error: %s", exc)
        return {
            "status":    "ERROR",
            "reason":    str(exc),
            "new_edges": [],
        }


# ---------------------------------------------------------------------------
# Write new edges to graph (with scientific review gate)
# ---------------------------------------------------------------------------

def _persist_new_edges(new_edges: list[dict], disease_name: str) -> dict:
    """
    Run new edges through the scientific reviewer, then write approved ones.

    Returns:
        {n_submitted, n_approved, n_rejected, write_result}
    """
    if not new_edges:
        return {
            "n_submitted": 0, "n_approved": 0,
            "n_rejected": 0, "write_result": {},
        }

    from orchestrator.scientific_reviewer import review_batch
    from mcp_servers.graph_db_server import write_causal_edges

    review = review_batch(new_edges)
    approved = review.get("approved_edges", [])

    write_result: dict = {}
    if approved:
        try:
            write_result = write_causal_edges(approved, disease_name)
        except Exception as exc:
            logger.error("[PERSIST] Write failed: %s", exc)
            write_result = {"error": str(exc)}

    return {
        "n_submitted": len(new_edges),
        "n_approved":  len(approved),
        "n_rejected":  review.get("n_rejected", 0),
        "write_result": write_result,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_update(
    disease_name: str,
    update_type: UpdateType = "gwas",
    auto_snapshot: bool = True,
) -> dict:
    """
    Run an incremental graph update.

    Args:
        disease_name: Disease to update (e.g. "coronary artery disease")
        update_type:  "gwas" | "literature" | "clinical_trials" | "full"
        auto_snapshot: If True, create a versioned snapshot after successful update

    Returns:
        {update_type, disease_name, started_at, finished_at,
         n_new_edges, n_approved, n_rejected, snapshot, status}
    """
    started_at = datetime.now(tz=timezone.utc).isoformat()
    logger.info("[UPDATE] Starting %s update for %s", update_type, disease_name)

    # Dispatch
    dispatch = {
        "gwas":             _run_gwas_refresh,
        "literature":       _run_literature_refresh,
        "clinical_trials":  _run_clinical_trials_refresh,
        "full":             _run_full_pipeline,
    }
    runner = dispatch.get(update_type)
    if runner is None:
        raise ValueError(f"Unknown update_type: {update_type!r}")

    try:
        update_result = runner(disease_name)
        status = update_result.get("status", "OK")
    except Exception as exc:
        logger.error("[UPDATE] Runner error: %s", exc)
        return {
            "update_type":  update_type,
            "disease_name": disease_name,
            "started_at":   started_at,
            "finished_at":  datetime.now(tz=timezone.utc).isoformat(),
            "status":       "ERROR",
            "error":        str(exc),
        }

    # Persist new edges (full pipeline already writes its own edges)
    new_edges = update_result.get("new_edges", [])
    if update_type != "full" and new_edges:
        persist = _persist_new_edges(new_edges, disease_name)
    else:
        persist = {
            "n_submitted": len(new_edges),
            "n_approved":  len(new_edges),
            "n_rejected":  0,
        }

    # Auto-snapshot
    snapshot_info: dict = {}
    if auto_snapshot and status not in ("HALTED", "ERROR"):
        try:
            from graph.versioning import get_current_version, bump_version, create_snapshot
            current = get_current_version()
            new_ver = bump_version(current, bump="patch")
            snapshot_info = create_snapshot(
                version_tag=new_ver,
                release_notes=f"Auto-snapshot after {update_type} update: {disease_name}",
            )
            logger.info("[UPDATE] Snapshot created: %s", new_ver)
        except Exception as exc:
            logger.warning("[UPDATE] Snapshot failed: %s", exc)
            snapshot_info = {"error": str(exc)}

    finished_at = datetime.now(tz=timezone.utc).isoformat()
    logger.info(
        "[UPDATE] Finished %s update. approved=%d rejected=%d",
        update_type, persist.get("n_approved", 0), persist.get("n_rejected", 0),
    )

    return {
        "update_type":  update_type,
        "disease_name": disease_name,
        "started_at":   started_at,
        "finished_at":  finished_at,
        "status":       status if status != "OK" else "OK",
        "n_new_edges":  persist.get("n_submitted", 0),
        "n_approved":   persist.get("n_approved", 0),
        "n_rejected":   persist.get("n_rejected", 0),
        "snapshot":     snapshot_info,
        "details":      update_result,
    }
