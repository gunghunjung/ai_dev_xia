"""Enhancement Experiment Manager — lifecycle tracking for training experiments."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, List, Optional

from .enhancement_training_manager import EnhancementTrainConfig, EnhancementTrainingManager


@dataclass
class EnhancementExperiment:
    experiment_id: str
    job_config: EnhancementTrainConfig
    status: str = "pending"         # pending / running / completed / failed
    started_at: str = ""
    finished_at: str = ""
    epoch_metrics: List[dict] = field(default_factory=list)
    best_psnr: float = 0.0
    output_model_path: str = ""
    failure_reason: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["job_config"] = asdict(self.job_config)
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "EnhancementExperiment":
        cfg_data = data.pop("job_config")
        cfg = EnhancementTrainConfig(**cfg_data)
        return cls(job_config=cfg, **data)


class EnhancementExperimentManager:
    """Tracks the full lifecycle of enhancement training experiments."""

    def __init__(self) -> None:
        self._experiments: Dict[str, EnhancementExperiment] = {}
        self._trainer = EnhancementTrainingManager()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def create(self, job_config: EnhancementTrainConfig) -> EnhancementExperiment:
        exp = EnhancementExperiment(
            experiment_id=str(uuid.uuid4()),
            job_config=job_config,
        )
        self._experiments[exp.experiment_id] = exp
        return exp

    def start(self, experiment_id: str) -> None:
        exp = self._get_or_raise(experiment_id)
        exp.status = "running"
        exp.started_at = datetime.now().isoformat()
        exp.epoch_metrics.clear()

        def _epoch_cb(epoch: int, loss: float, val_psnr: float) -> None:
            exp.epoch_metrics.append(
                {"epoch": epoch, "loss": round(loss, 6), "val_psnr": round(val_psnr, 4)}
            )
            if val_psnr > exp.best_psnr:
                exp.best_psnr = val_psnr

        self._trainer.run_training(exp.job_config, epoch_callback=_epoch_cb)

    def finish(
        self, experiment_id: str, output_path: str, best_psnr: float
    ) -> None:
        exp = self._get_or_raise(experiment_id)
        exp.status = "completed"
        exp.finished_at = datetime.now().isoformat()
        exp.output_model_path = output_path
        if best_psnr > exp.best_psnr:
            exp.best_psnr = best_psnr

    def fail(self, experiment_id: str, reason: str) -> None:
        exp = self._get_or_raise(experiment_id)
        exp.status = "failed"
        exp.finished_at = datetime.now().isoformat()
        exp.failure_reason = reason

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get(self, experiment_id: str) -> Optional[EnhancementExperiment]:
        return self._experiments.get(experiment_id)

    def list_all(self) -> List[EnhancementExperiment]:
        return list(self._experiments.values())

    def best_experiment(self, task_type: str) -> Optional[EnhancementExperiment]:
        """Return completed experiment with highest best_psnr for the given task_type."""
        candidates = [
            exp for exp in self._experiments.values()
            if exp.status == "completed"
            and exp.job_config.task_type == task_type
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda e: e.best_psnr)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_all(self, path: str) -> None:
        data = {eid: exp.to_dict() for eid, exp in self._experiments.items()}
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)

    def load_all(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        self._experiments = {
            eid: EnhancementExperiment.from_dict(v) for eid, v in data.items()
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_or_raise(self, experiment_id: str) -> EnhancementExperiment:
        exp = self._experiments.get(experiment_id)
        if exp is None:
            raise KeyError(f"Experiment '{experiment_id}' not found.")
        return exp
