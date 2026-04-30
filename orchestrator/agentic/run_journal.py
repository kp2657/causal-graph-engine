"""
run_journal.py — Structured run journal for the agentic pipeline.

Produces per-run output directory:
    outputs/runs/{disease}_{date}_{run_id}/
        journal.json        structured log
        token_usage.json    per-agent token counts + cost estimate

Usage:
    journal = RunJournal(disease_key="CAD", run_id="abc123")
    journal.log_path_reasoning("statistical_geneticist", "Most likely path is...")
    journal.log_virgin_target(target)
    journal.close()
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from orchestrator.agentic.agent_contracts import (
    AgentTokenUsage,
    AugmentationRecommendation,
    HumanPause,
    LibraryGap,
    MethodsChoice,
    PathReasoning,
    RedelegationRecord,
    TokenUsage,
    VirginTarget,
)


class RunJournal:
    def __init__(self, disease_key: str, run_id: str, base_dir: str = "outputs/runs") -> None:
        self.disease_key = disease_key
        self.run_id = run_id
        self._date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self._run_dir = Path(base_dir) / f"{disease_key}_{self._date}_{run_id}"
        self._run_dir.mkdir(parents=True, exist_ok=True)

        self._methods_choices: list[dict] = []
        self._path_reasoning: list[dict] = []
        self._human_pauses: list[dict] = []
        self._virgin_targets: list[dict] = []
        self._library_gaps: list[dict] = []
        self._augmentation_recommendations: list[dict] = []
        self._reviewer_verdict: dict = {}
        self._redelegation_log: list[dict] = []
        self._token_usage: list[dict] = []

    # ------------------------------------------------------------------
    # Logging methods
    # ------------------------------------------------------------------

    def log_methods_choice(self, choice: MethodsChoice) -> None:
        self._methods_choices.append(choice.model_dump())

    def log_path_reasoning(self, reasoning: PathReasoning) -> None:
        self._path_reasoning.append(reasoning.model_dump())

    def log_pause(self, pause: HumanPause) -> None:
        self._human_pauses.append(pause.model_dump())

    def log_virgin_target(self, target: VirginTarget) -> None:
        self._virgin_targets.append(target.model_dump())

    def log_library_gap(self, gap: LibraryGap) -> None:
        self._library_gaps.append(gap.model_dump())

    def log_augmentation_recommendation(self, rec: AugmentationRecommendation) -> None:
        self._augmentation_recommendations.append(rec.model_dump())

    def log_reviewer_verdict(
        self,
        verdict: Literal["APPROVE", "REVISE", "HARD_REJECT"],
        redelegation_instructions: list[RedelegationRecord] | None = None,
    ) -> None:
        self._reviewer_verdict = {
            "verdict": verdict,
            "redelegation_instructions": [r.model_dump() for r in (redelegation_instructions or [])],
        }

    def log_redelegation(self, record: RedelegationRecord) -> None:
        self._redelegation_log.append(record.model_dump())

    def log_token_usage(self, usage: AgentTokenUsage) -> None:
        self._token_usage.append(usage.model_dump())

    # ------------------------------------------------------------------
    # Finalisation
    # ------------------------------------------------------------------

    def close(self) -> Path:
        journal = {
            "disease_key": self.disease_key,
            "run_id": self.run_id,
            "date": self._date,
            "methods_choices": self._methods_choices,
            "path_reasoning": self._path_reasoning,
            "human_in_loop_pauses": self._human_pauses,
            "virgin_targets": self._virgin_targets,
            "library_gaps": self._library_gaps,
            "augmentation_recommendations": self._augmentation_recommendations,
            "reviewer_verdict": self._reviewer_verdict,
            "redelegation_log": self._redelegation_log,
        }

        journal_path = self._run_dir / "journal.json"
        journal_path.write_text(json.dumps(journal, indent=2))

        token_summary = self._build_token_summary()
        token_path = self._run_dir / "token_usage.json"
        token_path.write_text(json.dumps(token_summary, indent=2))

        return self._run_dir

    def _build_token_summary(self) -> dict:
        per_agent: dict[str, dict] = {}
        for entry in self._token_usage:
            name = entry["agent_name"]
            usage = entry["usage"]
            if name in per_agent:
                per_agent[name]["input_tokens"] += usage["input_tokens"]
                per_agent[name]["output_tokens"] += usage["output_tokens"]
                per_agent[name]["cost_usd"] += usage["cost_usd"]
            else:
                per_agent[name] = dict(usage)
        total_input = sum(v["input_tokens"] for v in per_agent.values())
        total_output = sum(v["output_tokens"] for v in per_agent.values())
        total_cost = sum(v["cost_usd"] for v in per_agent.values())
        return {
            "per_agent": per_agent,
            "total": {
                "input_tokens": total_input,
                "output_tokens": total_output,
                "cost_usd": round(total_cost, 4),
            },
        }

    @property
    def run_dir(self) -> Path:
        return self._run_dir
