"""
scheduler/update_scheduler.py — APScheduler-based graph update scheduler.

Schedule:
  - Weekly  (Monday 02:00 UTC): GWAS refresh
  - Monthly (1st, 03:00 UTC):   Literature scan + clinical trials
  - Quarterly (1st Jan/Apr/Jul/Oct, 04:00 UTC): Full Ota pipeline re-run

Requires: apscheduler >= 3.10
  conda run -n causal-graph pip install apscheduler
"""
from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

_DISEASE_NAME = os.getenv("TARGET_DISEASE", "coronary artery disease")

# ---------------------------------------------------------------------------
# Job functions — imported lazily to allow offline import of the module
# ---------------------------------------------------------------------------

def _job_gwas_refresh() -> None:
    """Weekly: refresh GWAS loci."""
    from graph.update_pipeline import run_update

    logger.info("[SCHEDULER] Running weekly GWAS refresh for %s", _DISEASE_NAME)
    try:
        result = run_update(_DISEASE_NAME, update_type="gwas", auto_snapshot=True)
        logger.info(
            "[SCHEDULER] GWAS refresh done. approved=%d rejected=%d snapshot=%s",
            result.get("n_approved", 0),
            result.get("n_rejected", 0),
            result.get("snapshot", {}).get("version_tag", "—"),
        )
    except Exception as exc:
        logger.error("[SCHEDULER] GWAS refresh failed: %s", exc)


def _job_literature_refresh() -> None:
    """Monthly: PubMed scan for new papers."""
    from graph.update_pipeline import run_update

    logger.info("[SCHEDULER] Running monthly literature refresh for %s", _DISEASE_NAME)
    try:
        result = run_update(_DISEASE_NAME, update_type="literature", auto_snapshot=False)
        logger.info(
            "[SCHEDULER] Literature refresh done. new_papers=%d flagged=%s",
            result.get("details", {}).get("n_new_papers", 0),
            result.get("details", {}).get("flagged_genes", []),
        )
    except Exception as exc:
        logger.error("[SCHEDULER] Literature refresh failed: %s", exc)


def _job_clinical_trials_refresh() -> None:
    """Monthly: clinical trial status update."""
    from graph.update_pipeline import run_update

    logger.info("[SCHEDULER] Running monthly clinical trials refresh for %s", _DISEASE_NAME)
    try:
        result = run_update(_DISEASE_NAME, update_type="clinical_trials", auto_snapshot=False)
        logger.info(
            "[SCHEDULER] Trials refresh done. completed=%d terminated=%d",
            result.get("details", {}).get("n_completed", 0),
            result.get("details", {}).get("n_terminated", 0),
        )
    except Exception as exc:
        logger.error("[SCHEDULER] Clinical trials refresh failed: %s", exc)


def _job_full_pipeline() -> None:
    """Quarterly: full Ota pipeline re-run."""
    from graph.update_pipeline import run_update

    logger.info("[SCHEDULER] Running quarterly full pipeline for %s", _DISEASE_NAME)
    try:
        result = run_update(_DISEASE_NAME, update_type="full", auto_snapshot=True)
        logger.info(
            "[SCHEDULER] Full pipeline done. status=%s n_edges=%d snapshot=%s",
            result.get("status"),
            result.get("n_approved", 0),
            result.get("snapshot", {}).get("version_tag", "—"),
        )
    except Exception as exc:
        logger.error("[SCHEDULER] Full pipeline failed: %s", exc)


# ---------------------------------------------------------------------------
# Scheduler factory
# ---------------------------------------------------------------------------

def build_scheduler(disease_name: str | None = None):
    """
    Build and return a configured APScheduler BlockingScheduler.

    Args:
        disease_name: Override TARGET_DISEASE env var

    Returns:
        apscheduler.schedulers.blocking.BlockingScheduler (not yet started)
    """
    global _DISEASE_NAME
    if disease_name:
        _DISEASE_NAME = disease_name

    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
        from apscheduler.triggers.cron import CronTrigger
    except ImportError as exc:
        raise ImportError(
            "apscheduler not installed. Run: pip install apscheduler>=3.10"
        ) from exc

    scheduler = BlockingScheduler(timezone="UTC")

    # Weekly: Monday 02:00 UTC — GWAS refresh
    scheduler.add_job(
        _job_gwas_refresh,
        trigger=CronTrigger(day_of_week="mon", hour=2, minute=0),
        id="gwas_refresh",
        name="Weekly GWAS Refresh",
        misfire_grace_time=3600,  # 1 hour grace
        coalesce=True,
    )

    # Monthly: 1st of month 03:00 UTC — literature + clinical trials
    scheduler.add_job(
        _job_literature_refresh,
        trigger=CronTrigger(day=1, hour=3, minute=0),
        id="literature_refresh",
        name="Monthly Literature Refresh",
        misfire_grace_time=3600,
        coalesce=True,
    )
    scheduler.add_job(
        _job_clinical_trials_refresh,
        trigger=CronTrigger(day=1, hour=3, minute=30),
        id="clinical_trials_refresh",
        name="Monthly Clinical Trials Refresh",
        misfire_grace_time=3600,
        coalesce=True,
    )

    # Quarterly: 1st of Jan/Apr/Jul/Oct 04:00 UTC — full pipeline
    scheduler.add_job(
        _job_full_pipeline,
        trigger=CronTrigger(month="1,4,7,10", day=1, hour=4, minute=0),
        id="full_pipeline",
        name="Quarterly Full Pipeline",
        misfire_grace_time=7200,  # 2 hour grace
        coalesce=True,
    )

    logger.info(
        "[SCHEDULER] Configured 4 jobs for disease=%s", _DISEASE_NAME
    )
    return scheduler


def list_jobs(scheduler=None) -> list[dict]:
    """
    Return a list of scheduled job metadata.

    Args:
        scheduler: Optional existing scheduler; builds a fresh one if None

    Returns:
        List of {id, name, next_run_time, trigger}
    """
    if scheduler is None:
        scheduler = build_scheduler()

    return [
        {
            "id":            job.id,
            "name":          job.name,
            "next_run_time": job.next_run_time.isoformat() if job.next_run_time else None,
            "trigger":       str(job.trigger),
        }
        for job in scheduler.get_jobs()
    ]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def start(disease_name: str | None = None) -> None:
    """
    Start the blocking scheduler (blocks until Ctrl-C).

    Args:
        disease_name: Override TARGET_DISEASE env var
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )
    scheduler = build_scheduler(disease_name)
    logger.info("[SCHEDULER] Starting. Press Ctrl-C to exit.")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("[SCHEDULER] Stopped.")
