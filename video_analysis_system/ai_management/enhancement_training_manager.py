"""Enhancement Training Manager — runs simulated training jobs in background threads."""

from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional


@dataclass
class EnhancementTrainConfig:
    job_name: str
    dataset_id: str
    task_type: str            # super_resolution / sharpening / denoising / deblurring
    scale_factor: int = 2
    epochs: int = 50
    batch_size: int = 16
    learning_rate: float = 1e-4
    output_dir: str = "outputs/enhancement"
    checkpoint_interval: int = 10


class EnhancementTrainingManager:
    """Manages a single enhancement training job in a background thread."""

    def __init__(self) -> None:
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._is_training = False
        self._last_output_path: str = ""

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_training(self) -> bool:
        return self._is_training

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_training(
        self,
        config: EnhancementTrainConfig,
        epoch_callback: Optional[Callable[[int, float, float], None]] = None,
    ) -> str:
        """Start training in a background thread; returns the output directory path."""
        if self._is_training:
            raise RuntimeError("Training is already in progress.")

        self._stop_event.clear()
        self._is_training = True

        output_dir = Path(config.output_dir) / config.job_name
        output_dir.mkdir(parents=True, exist_ok=True)
        self._last_output_path = str(output_dir)

        self._thread = threading.Thread(
            target=self._train_loop,
            args=(config, output_dir, epoch_callback),
            daemon=True,
        )
        self._thread.start()
        return str(output_dir)

    def stop(self) -> None:
        """Signal the training loop to stop gracefully."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._is_training = False

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _train_loop(
        self,
        config: EnhancementTrainConfig,
        output_dir: Path,
        epoch_callback: Optional[Callable[[int, float, float], None]],
    ) -> None:
        try:
            base_loss = 0.5
            for epoch in range(1, config.epochs + 1):
                if self._stop_event.is_set():
                    break

                # Simulate exponential loss decay with noise
                import random
                decay = math.exp(-3.0 * epoch / config.epochs)
                loss = base_loss * decay + random.uniform(0.001, 0.01)
                val_psnr = 25.0 + 10.0 * (1.0 - decay) + random.uniform(-0.5, 0.5)

                time.sleep(0.02)  # simulate iteration time

                if epoch_callback is not None:
                    try:
                        epoch_callback(epoch, loss, val_psnr)
                    except Exception:
                        pass

                if epoch % config.checkpoint_interval == 0 or epoch == config.epochs:
                    self._save_checkpoint(config, output_dir, epoch, loss, val_psnr)
        finally:
            self._is_training = False

    def _save_checkpoint(
        self,
        config: EnhancementTrainConfig,
        output_dir: Path,
        epoch: int,
        loss: float,
        val_psnr: float,
    ) -> None:
        ckpt = {
            "job_name": config.job_name,
            "task_type": config.task_type,
            "dataset_id": config.dataset_id,
            "epoch": epoch,
            "loss": round(loss, 6),
            "val_psnr": round(val_psnr, 4),
            "saved_at": datetime.now().isoformat(),
        }
        ckpt_path = output_dir / f"checkpoint_epoch{epoch:04d}.json"
        with open(ckpt_path, "w", encoding="utf-8") as fh:
            json.dump(ckpt, fh, indent=2)
