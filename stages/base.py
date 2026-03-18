from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from utils.common import AttemptRecord, GlobalConfig, OutputPaths, StageResult, save_json, stage_failure_report


@dataclass
class StageContext:
    config: GlobalConfig
    device: torch.device
    outputs: OutputPaths
    history: dict[str, StageResult] = field(default_factory=dict)
    shared: dict[str, Any] = field(default_factory=dict)


class BaseStage:
    name: str = "base"
    max_attempts: int = 5
    score_direction: str = "min"

    def attempt_settings(self) -> list[dict[str, Any]]:
        return [{} for _ in range(self.max_attempts)]

    def execute_attempt(
        self,
        context: StageContext,
        attempt: int,
        params: dict[str, Any],
        stage_dir: Path,
    ) -> tuple[bool, float, dict[str, Any], str]:
        raise NotImplementedError

    def _is_better(self, score: float, incumbent: float | None) -> bool:
        if incumbent is None:
            return True
        if self.score_direction == "max":
            return score > incumbent
        return score < incumbent

    def run(self, context: StageContext) -> StageResult:
        stage_dir = context.outputs.stage_dir(self.name)
        attempt_records: list[AttemptRecord] = []
        best_score: float | None = None
        best_attempt = 1
        best_metrics: dict[str, Any] = {}
        stage_success = False
        failure_payload: dict[str, Any] | None = None

        for attempt_number, params in enumerate(self.attempt_settings(), start=1):
            success, score, metrics, notes = self.execute_attempt(context, attempt_number, params, stage_dir)
            attempt_records.append(
                AttemptRecord(
                    attempt=attempt_number,
                    success=success,
                    score=score,
                    metrics=metrics,
                    notes=notes,
                )
            )
            if self._is_better(score, best_score):
                best_score = score
                best_attempt = attempt_number
                best_metrics = metrics
            if success:
                stage_success = True
                break

        if not stage_success:
            failure_payload = stage_failure_report(
                issue=f"{self.name} did not meet its success criterion after {self.max_attempts} attempts.",
                cause="The current configuration failed to cross the stage threshold despite automatic retries.",
                evidence={"best_metrics": best_metrics, "attempts": [record.metrics for record in attempt_records]},
                suggested_fix="Use the logged plots and metrics to retune feature extraction or training hyperparameters.",
                requires_next_model=False,
            )

        result = StageResult(
            name=self.name,
            success=stage_success,
            best_attempt=best_attempt,
            best_score=float(best_score if best_score is not None else 0.0),
            best_metrics=best_metrics,
            attempts=attempt_records,
            failure_report=failure_payload,
        )
        save_json(context.outputs.logs / f"{self.name}.json", result.to_dict())
        context.history[self.name] = result
        return result
