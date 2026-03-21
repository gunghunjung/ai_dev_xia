"""
logging_system/logger.py — Per-frame and per-ROI structured logging.

Writes to:
  - Python logging (console / rotating file)
  - In-memory frame log (exportable to CSV/JSON)
"""

from __future__ import annotations

import csv
import json
import logging
import logging.handlers
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import LoggingConfig
from core.data_models import FrameContext
from utils.helpers import ensure_dir


# ---------------------------------------------------------------------------
# Module-level Python logger setup
# ---------------------------------------------------------------------------

def setup_logging(log_dir: str, level: int = logging.INFO) -> None:
    """Configure root Python logger to write to console and a rotating file."""
    ensure_dir(log_dir)
    root = logging.getLogger()
    root.setLevel(level)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # Rotating file handler (10 MB × 5 backups)
    fh = logging.handlers.RotatingFileHandler(
        Path(log_dir) / "system.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    fh.setFormatter(fmt)
    root.addHandler(fh)


# ---------------------------------------------------------------------------
# FrameLogger — structured per-frame record accumulation
# ---------------------------------------------------------------------------

class FrameLogger:
    """
    Collects lightweight per-frame summaries and exports them as CSV or JSON.

    Only one row per frame is stored (keeping memory predictable).
    """

    def __init__(self, cfg: LoggingConfig):
        self._cfg = cfg
        self._log_dir = ensure_dir(cfg.log_dir)
        self._records: List[Dict[str, Any]] = []
        self._log_every = cfg.log_every_n_frames
        self._logger = logging.getLogger(self.__class__.__name__)

    def log_frame(self, ctx: FrameContext) -> None:
        """Called after each processed frame."""
        if ctx.frame_index % self._log_every != 0:
            return

        row: Dict[str, Any] = {
            "frame_index":    ctx.frame_index,
            "timestamp":      ctx.timestamp,
            "system_state":   ctx.system_state.value,
            "state_conf":     round(ctx.state_confidence, 4),
            "inference_ms":   round(ctx.inference_time_ms, 2),
            "is_event":       ctx.is_event,
            "event_type":     ctx.event_type,
        }

        # Per-ROI summary (flattened)
        for roi in ctx.rois:
            prefix = f"roi_{roi.roi_id}"
            row[f"{prefix}_state"] = roi.roi_state.value
            row[f"{prefix}_mean_intensity"] = round(
                roi.features.get("mean_intensity", 0.0), 2
            )
            row[f"{prefix}_mean_diff"] = round(
                roi.features.get("mean_abs_diff", 0.0), 2
            )
            if roi.detection:
                row[f"{prefix}_ai_label"] = roi.detection.label
                row[f"{prefix}_ai_conf"]  = round(roi.detection.confidence, 4)

        # Triggered rule names
        row["triggered_rules"] = "|".join(r.rule_name for r in ctx.triggered_rules)

        self._records.append(row)

    def export_csv(self, filename: Optional[str] = None) -> Path:
        if not self._records:
            self._logger.warning("No records to export")
            return self._log_dir / "empty.csv"

        ts = time.strftime("%Y%m%d_%H%M%S")
        path = self._log_dir / (filename or f"frame_log_{ts}.csv")
        fieldnames = list(self._records[0].keys())

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self._records)

        self._logger.info("CSV exported: %s (%d rows)", path, len(self._records))
        return path

    def export_json(self, filename: Optional[str] = None) -> Path:
        if not self._records:
            self._logger.warning("No records to export")
            return self._log_dir / "empty.json"

        ts = time.strftime("%Y%m%d_%H%M%S")
        path = self._log_dir / (filename or f"frame_log_{ts}.json")

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._records, f, indent=2, ensure_ascii=False)

        self._logger.info("JSON exported: %s (%d frames)", path, len(self._records))
        return path

    def export_all(self) -> None:
        if self._cfg.export_csv:
            self.export_csv()
        if self._cfg.export_json:
            self.export_json()

    def clear(self) -> None:
        self._records.clear()

    @property
    def record_count(self) -> int:
        return len(self._records)
