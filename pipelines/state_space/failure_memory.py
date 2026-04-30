"""
pipelines/state_space/failure_memory.py

Build FailureRecord objects from:
  1. ClinicalTrials.gov API v2 — terminated/suspended trials for a disease
  2. Curated seed lists — known IBD / CAD therapeutic failures with annotated failure modes

FailureRecord captures *why* a perturbation failed in context, not just that it failed.
This context feeds `trajectory_scoring.py` to penalise targets with known escape/memory issues.

Failure mode vocabulary (defined in schemas.py FAILURE_MODES):
  no_effect              — intervention produced no measurable change
  transient_only         — initial response but no durable effect
  non_responder          — subpopulation did not respond (≥30% non-response rate)
  escape                 — alternative state activated after initial response
  discordant_genetic_support — genetic evidence contradicts drug direction
  toxicity_limit         — dose-limiting toxicity before efficacy achieved
  disease_context_mismatch — efficacy seen in different disease subtype only
  cell_type_mismatch     — target not expressed in relevant cell type in vivo
  donor_specific_resistance — responder/non-responder split by donor genetics
"""
from __future__ import annotations

from models.evidence import FailureRecord
from pipelines.state_space.schemas import FAILURE_MODES


_CAD_SEED_FAILURES: list[dict] = [
    {
        "perturbation_id":  "CETP-inhibitor",  # Torcetrapib, Dalcetrapib, Evacetrapib
        "failure_mode":     "no_effect",
        "evidence_strength": 0.95,
        "data_source":      "curated_cad_seed",
        "notes":            "Multiple phase 3 failures despite HDL-C raising; off-target effects and no plaque benefit",
    },
    {
        "perturbation_id":  "HDAC-inhibitor",
        "failure_mode":     "toxicity_limit",
        "evidence_strength": 0.7,
        "data_source":      "curated_cad_seed",
        "notes":            "Systemic HDAC inhibition limits cardiac dose; macrophage off-target toxicity",
    },
    {
        "perturbation_id":  "anti-IL1B",     # Canakinumab in CAD (CANTOS)
        "failure_mode":     "non_responder",
        "evidence_strength": 0.75,
        "data_source":      "curated_cad_seed",
        "notes":            "CANTOS positive but only in hsCRP-high subgroup; broad population non-response",
    },
    {
        "perturbation_id":  "anti-Lp(a)",
        "failure_mode":     "disease_context_mismatch",
        "evidence_strength": 0.5,
        "data_source":      "curated_cad_seed",
        "notes":            "Phase 3 ongoing; genetic evidence strong but phenotype may be Lp(a)-high subtype only",
    },
]

_SEED_FAILURES_BY_DISEASE: dict[str, list[dict]] = {
    "CAD": _CAD_SEED_FAILURES,
}


# ---------------------------------------------------------------------------
# ClinicalTrials.gov helper
# ---------------------------------------------------------------------------

def _fetch_ct_failures(disease_query: str, max_results: int = 50) -> list[dict]:
    """
    Query ClinicalTrials.gov API v2 for terminated/suspended trials.
    Returns minimal dicts: {nct_id, official_title, status, why_stopped, interventions}
    Falls back to empty list on network error (non-blocking).
    """
    try:
        import urllib.request
        import urllib.parse
        import json

        base = "https://clinicaltrials.gov/api/v2/studies"
        params = {
            "query.cond": disease_query,
            "filter.overallStatus": "TERMINATED,SUSPENDED",
            "fields": "NCTId,OfficialTitle,OverallStatus,WhyStopped,InterventionName",
            "pageSize": str(max_results),
            "format": "json",
        }
        url = base + "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())

        studies = data.get("studies", [])
        results = []
        for s in studies:
            proto = s.get("protocolSection", {})
            ident = proto.get("identificationModule", {})
            status = proto.get("statusModule", {})
            arms = proto.get("armsInterventionsModule", {})
            interventions = [
                i.get("name", "") for i in arms.get("interventions", [])
            ]
            results.append({
                "nct_id":         ident.get("nctId", ""),
                "title":          ident.get("officialTitle", ""),
                "status":         status.get("overallStatus", ""),
                "why_stopped":    status.get("whyStopped", ""),
                "interventions":  interventions,
            })
        return results
    except Exception:
        return []


def _infer_failure_mode(why_stopped: str, interventions: list[str]) -> str:
    """
    Heuristically map ClinicalTrials.gov whyStopped text to FAILURE_MODES vocabulary.
    """
    text = why_stopped.lower()
    if any(t in text for t in ("efficacy", "futility", "no effect", "no benefit")):
        return "no_effect"
    if any(t in text for t in ("toxic", "adverse", "safety", "harm")):
        return "toxicity_limit"
    if any(t in text for t in ("sponsor", "business", "funding", "financial")):
        return "no_effect"   # treat business decisions as no_effect (conservative)
    return "no_effect"       # default: unknown → conservative


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_failure_records(
    disease: str,
    include_ct: bool = True,
    max_ct_results: int = 30,
) -> list[FailureRecord]:
    """
    Build FailureRecord list for a disease.

    Combines:
      1. Curated seed failures (always included)
      2. ClinicalTrials.gov terminated/suspended trials (optional, requires network)

    Args:
        disease:         Short disease key, e.g. "IBD", "CAD"
        include_ct:      Whether to query ClinicalTrials.gov API (default True)
        max_ct_results:  Max trials to fetch from ClinicalTrials.gov

    Returns:
        list[FailureRecord]
    """
    records: list[FailureRecord] = []
    disease_upper = disease.upper()

    # --- Seed failures ---
    for i, seed in enumerate(_SEED_FAILURES_BY_DISEASE.get(disease_upper, [])):
        fid = f"{disease_upper}_seed_{i:03d}_{seed['perturbation_id'].replace(' ', '_')}"
        records.append(FailureRecord(
            failure_id        = fid,
            disease           = disease_upper,
            perturbation_id   = seed["perturbation_id"],
            perturbation_type = seed.get("perturbation_type", "drug"),
            failure_mode      = seed["failure_mode"],
            evidence_strength = seed["evidence_strength"],
            data_source       = seed["data_source"],
        ))

    # --- ClinicalTrials.gov ---
    if include_ct:
        disease_query_map = {
            "CAD": "Coronary Artery Disease",
            "RA":  "Rheumatoid Arthritis",
            "T2D": "Type 2 Diabetes",
        }
        ct_query = disease_query_map.get(disease_upper, disease)
        ct_trials = _fetch_ct_failures(ct_query, max_results=max_ct_results)

        for trial in ct_trials:
            if not trial.get("interventions"):
                continue
            for intervention in trial["interventions"]:
                if not intervention:
                    continue
                fmode = _infer_failure_mode(
                    trial.get("why_stopped", ""), trial["interventions"]
                )
                fid = f"{disease_upper}_ct_{trial['nct_id']}_{intervention[:20].replace(' ', '_')}"
                records.append(FailureRecord(
                    failure_id        = fid,
                    disease           = disease_upper,
                    perturbation_id   = intervention,
                    perturbation_type = "drug",
                    failure_mode      = fmode,
                    evidence_strength = 0.5,   # ClinicalTrials.gov: moderate evidence (terminated ≠ proven failure)
                    data_source       = f"ClinicalTrials.gov:{trial['nct_id']}",
                ))

    return records


def get_failure_modes_for_perturbation(
    perturbation_id: str,
    records: list[FailureRecord],
) -> list[str]:
    """Return failure modes observed for a given perturbation across all records."""
    return list({
        r.failure_mode for r in records
        if perturbation_id.lower() in r.perturbation_id.lower()
    })


def failure_penalty_score(
    perturbation_id: str,
    records: list[FailureRecord],
) -> float:
    """
    Aggregate failure penalty for a perturbation: 0.0 (no failures) → 1.0 (strong multi-failure).

    Weighted sum of evidence_strength × mode_weight, capped at 1.0.
    escape and non_responder carry highest weights (most relevant to state-space model).
    """
    MODE_WEIGHTS: dict[str, float] = {
        "escape":                      1.0,
        "non_responder":               0.9,
        "transient_only":              0.7,
        "negative_memory":             0.8,
        "no_effect":                   0.6,
        "disease_context_mismatch":    0.5,
        "cell_type_mismatch":          0.5,
        "donor_specific_resistance":   0.4,
        "toxicity_limit":              0.3,
        "discordant_genetic_support":  0.7,
    }
    matched = [
        r for r in records
        if perturbation_id.lower() in r.perturbation_id.lower()
    ]
    if not matched:
        return 0.0
    score = sum(
        r.evidence_strength * MODE_WEIGHTS.get(r.failure_mode, 0.5)
        for r in matched
    )
    return min(1.0, score)
