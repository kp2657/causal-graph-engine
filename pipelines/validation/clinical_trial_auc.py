"""
clinical_trial_auc.py — Clinical trial AUC validation.

Queries OpenTargets for Phase 2+ clinical trial outcomes (approved/failed/withdrawn)
per (gene, disease) pair. Computes AUROC of pipeline OTA gamma scores.

Success = drug approved or Phase 3 completed.
Failure = Phase 2/3 withdrawn or failed.
"""
from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any

log = logging.getLogger(__name__)

OT_CLINICAL_URL = "https://api.platform.opentargets.org/api/v4/graphql"

PHASE_SUCCESS_MIN = 3  # Phase 3+ = success signal

# GraphQL query for known drugs per target/disease
_KNOWN_DRUGS_QUERY = """
query KnownDrugs($ensemblId: String!, $diseaseId: String!) {
  target(ensemblId: $ensemblId) {
    knownDrugs(disease: $diseaseId) {
      rows {
        drug { name }
        phase
        status
      }
    }
  }
}
"""

# Status strings that indicate trial failure/withdrawal
_FAILURE_STATUSES = {
    "withdrawn", "terminated", "failed",
    "no longer pursued", "suspended",
}

# Status strings that indicate approval/success
_SUCCESS_STATUSES = {
    "approved", "completed", "marketed",
    "registered", "authorised",
}


def _map_trial_status(phase: int, status: str | None) -> str:
    """
    Map (phase, status) to "success", "failure", or "unknown".
    phase >= 3 + approved/completed → "success"
    withdrawn/terminated/failed (any phase) → "failure"
    otherwise → "unknown"
    """
    status_lower = (status or "").lower().strip()

    if status_lower in _FAILURE_STATUSES:
        return "failure"

    if phase >= PHASE_SUCCESS_MIN and status_lower in _SUCCESS_STATUSES:
        return "success"

    # Phase 4 with no explicit failure → success
    if phase >= 4 and status_lower not in _FAILURE_STATUSES:
        return "success"

    return "unknown"


def query_ot_clinical_trials(
    gene_symbol: str,
    disease_efo: str,
    timeout_s: float = 15.0,
    ensembl_id: str | None = None,
) -> list[dict]:
    """
    Query OT GraphQL for known drug–target–disease associations with clinical phase info.
    Returns list of {
        "gene": str,
        "drug_name": str,
        "max_phase": int,
        "status": "success" | "failure" | "unknown",
    }

    Use the OT /graphql endpoint with knownDrugs query.
    Map: phase >= 3 + status approved/completed → "success"
         status withdrawn/terminated/failed → "failure"
         otherwise → "unknown"

    Return empty list on network error (don't raise).
    """
    # We need an Ensembl ID for the OT query; without it we cannot proceed
    target_id = ensembl_id or gene_symbol  # callers may pass Ensembl ID as gene_symbol
    if not target_id:
        return []

    payload = json.dumps({
        "query": _KNOWN_DRUGS_QUERY,
        "variables": {
            "ensemblId": target_id,
            "diseaseId": disease_efo,
        },
    }).encode("utf-8")

    req = urllib.request.Request(
        OT_CLINICAL_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read()
            data = json.loads(raw)
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, Exception) as exc:
        log.warning("OT clinical trials query failed for %s/%s: %s", gene_symbol, disease_efo, exc)
        return []

    rows: list[dict] = []
    try:
        target_data = data.get("data", {}).get("target") or {}
        known_drugs = target_data.get("knownDrugs") or {}
        for row in known_drugs.get("rows", []):
            drug_name = (row.get("drug") or {}).get("name", "unknown")
            phase     = int(row.get("phase") or 0)
            status    = row.get("status")
            outcome   = _map_trial_status(phase, status)
            rows.append({
                "gene":      gene_symbol,
                "drug_name": drug_name,
                "max_phase": phase,
                "status":    outcome,
            })
    except (KeyError, TypeError, ValueError) as exc:
        log.warning("Failed to parse OT response for %s: %s", gene_symbol, exc)

    return rows


def compute_auroc(
    scores: list[float],
    labels: list[int],  # 1=success, 0=failure
) -> float:
    """
    Compute AUROC using the Mann-Whitney U statistic (no sklearn needed).
    Returns 0.5 if all labels are same class (degenerate case).
    """
    if not scores or not labels or len(scores) != len(labels):
        return 0.5

    pos_scores = [s for s, l in zip(scores, labels) if l == 1]
    neg_scores = [s for s, l in zip(scores, labels) if l == 0]

    n_pos = len(pos_scores)
    n_neg = len(neg_scores)

    if n_pos == 0 or n_neg == 0:
        return 0.5  # degenerate case

    # Mann-Whitney U: count pairs where positive score > negative score
    u = 0
    for p in pos_scores:
        for n in neg_scores:
            if p > n:
                u += 1
            elif p == n:
                u += 0.5

    auroc = u / (n_pos * n_neg)
    return auroc


def run_clinical_trial_auc(
    scored_genes: list[dict],   # list of {"gene": str, "ota_gamma": float, ...}
    disease_efo: str,
    disease_key: str,
    gene_to_ensembl: dict[str, str] | None = None,
) -> dict:
    """
    Full validation pipeline:
    1. For each gene in scored_genes, query OT clinical trials
    2. Label each gene: 1=success, 0=failure, skip=unknown
    3. Compute AUROC of ota_gamma scores vs labels

    Returns: {
        "disease_key": str,
        "n_genes_queried": int,
        "n_success": int,
        "n_failure": int,
        "auroc": float,
        "above_random": bool,   # auroc > 0.55
    }
    If < 5 labeled genes: return {"disease_key": disease_key, "skipped": True, "reason": "insufficient_labels"}
    """
    gene_to_ensembl = gene_to_ensembl or {}

    scores: list[float] = []
    labels: list[int]   = []
    n_success = 0
    n_failure = 0
    n_queried = 0

    for entry in scored_genes:
        gene       = entry.get("gene", "")
        ota_gamma  = entry.get("ota_gamma", 0.0)
        ensembl_id = gene_to_ensembl.get(gene)

        n_queried += 1
        trials = query_ot_clinical_trials(
            gene_symbol=gene,
            disease_efo=disease_efo,
            ensembl_id=ensembl_id,
        )

        # Determine overall label for this gene: take the best outcome
        gene_label: int | None = None
        for trial in trials:
            if trial["status"] == "success":
                gene_label = 1
                break  # success wins
            elif trial["status"] == "failure":
                gene_label = 0  # keep looking for success

        if gene_label is None:
            continue  # unknown — skip

        if gene_label == 1:
            n_success += 1
        else:
            n_failure += 1

        scores.append(float(ota_gamma))
        labels.append(gene_label)

    n_labeled = len(labels)
    if n_labeled < 5:
        return {
            "disease_key": disease_key,
            "skipped": True,
            "reason": "insufficient_labels",
        }

    auroc = compute_auroc(scores, labels)

    return {
        "disease_key":    disease_key,
        "n_genes_queried": n_queried,
        "n_success":      n_success,
        "n_failure":      n_failure,
        "auroc":          round(auroc, 4),
        "above_random":   auroc > 0.55,
    }
