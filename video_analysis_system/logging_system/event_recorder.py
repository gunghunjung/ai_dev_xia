"""
logging_system/event_recorder.py — Captures abnormal event clips and metadata.

When an event is detected the recorder:
  1. Saves a JPEG snapshot of the event frame.
  2. Saves a short video clip using pre/post-event frames from the ring buffer.
  3. Appends an EventRecord to the events JSON log.

Pre-event frames come from an internal ring buffer that continuously stores
raw frames, independent of TemporalBuffer (which stores FrameContext objects).
"""

from __future__ import annotations

import collections
import json
import logging
import time
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

from config import LoggingConfig
from core.data_models import EventRecord, FrameContext
from utils.helpers import ensure_dir

logger = logging.getLogger(__name__)


class EventRecorder:
    """
    Maintains a rolling raw-frame buffer for pre-event capture.
    On event, writes snapshot + clip + JSON entry.
    """

    def __init__(self, cfg: LoggingConfig, fps: float = 30.0):
        self._cfg = cfg
        self._fps = fps
        self._event_dir = ensure_dir(Path(cfg.log_dir) / "events")
        self._events_json = self._event_dir / "events.json"

        # Rolling buffer of (frame_index, raw_frame) tuples
        pre_cap = max(cfg.pre_event_frames, 1)
        self._pre_buffer: Deque[Tuple[int, np.ndarray]] = collections.deque(maxlen=pre_cap)

        # Post-event capture queue
        self._post_frames: List[Tuple[int, np.ndarray]] = []
        self._post_remaining: int = 0
        self._active_event: Optional[EventRecord] = None

        self._event_records: List[EventRecord] = []
        self._event_counter: int = 0

    # ── Feed raw frames ───────────────────────────────────────────────────

    def feed_frame(self, ctx: FrameContext) -> None:
        """
        Must be called every frame (before or after event detection).
        Stores raw frame in pre-buffer; collects post-event frames when active.
        """
        raw = ctx.raw_frame
        if raw is None:
            return

        # Always feed pre-buffer
        self._pre_buffer.append((ctx.frame_index, raw.copy()))

        # Collect post-event frames
        if self._post_remaining > 0 and self._active_event is not None:
            self._post_frames.append((ctx.frame_index, raw.copy()))
            self._post_remaining -= 1
            if self._post_remaining == 0:
                self._finalise_event_clip()

    # ── Trigger an event ──────────────────────────────────────────────────

    def trigger_event(self, ctx: FrameContext) -> Optional[EventRecord]:
        """
        Call when FrameContext.is_event is True.
        Saves snapshot immediately; schedules post-event clip collection.
        Returns EventRecord.
        """
        self._event_counter += 1
        event_id = f"evt_{ctx.frame_index:08d}_{self._event_counter:04d}"
        ts = time.strftime("%Y%m%d_%H%M%S")

        # Save snapshot
        snapshot_path = ""
        if self._cfg.save_event_frames and ctx.raw_frame is not None:
            snap_file = self._event_dir / f"{event_id}_snapshot.jpg"
            cv2.imwrite(str(snap_file), ctx.raw_frame)
            snapshot_path = str(snap_file)

        # Build EventRecord
        roi_ids = [r.roi_id for r in ctx.rois if r.roi_state.value != "NORMAL"]
        roi_id = roi_ids[0] if roi_ids else ""
        abnorm_types = [r.abnormality_type.value for r in ctx.triggered_rules if r.abnormality_type]
        abnorm_str = "|".join(set(abnorm_types))

        record = EventRecord(
            event_id=event_id,
            frame_index=ctx.frame_index,
            timestamp=ctx.timestamp,
            event_type=ctx.event_type,
            severity=ctx.event_severity,
            system_state=ctx.system_state.value,
            roi_id=roi_id,
            abnormality_type=abnorm_str,
            message=" | ".join(r.message for r in ctx.triggered_rules[:3]),
            snapshot_path=snapshot_path,
        )

        # Schedule post-event clip
        if self._cfg.save_event_frames:
            self._active_event = record
            self._post_frames = []
            self._post_remaining = self._cfg.post_event_frames

        self._event_records.append(record)
        self._append_event_json(record)
        logger.info("Event recorded: %s type=%s frame=%d", event_id, ctx.event_type, ctx.frame_index)
        return record

    # ── JSON event log ─────────────────────────────────────────────────────

    def _append_event_json(self, record: EventRecord) -> None:
        existing: List[dict] = []
        if self._events_json.exists():
            try:
                with open(self._events_json, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            except json.JSONDecodeError:
                existing = []
        existing.append(record.to_dict())
        with open(self._events_json, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)

    # ── Clip writing ──────────────────────────────────────────────────────

    def _finalise_event_clip(self) -> None:
        if self._active_event is None:
            return

        pre_frames = [f for _, f in list(self._pre_buffer)]
        post_frames = [f for _, f in self._post_frames]
        all_frames = pre_frames + post_frames

        if not all_frames:
            self._active_event = None
            return

        clip_path = self._event_dir / f"{self._active_event.event_id}_clip.avi"
        h, w = all_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(str(clip_path), fourcc, self._fps, (w, h))
        for frame in all_frames:
            writer.write(frame)
        writer.release()

        logger.info("Event clip saved: %s (%d frames)", clip_path, len(all_frames))
        self._active_event = None
        self._post_frames = []

    # ── Summary ───────────────────────────────────────────────────────────

    @property
    def event_count(self) -> int:
        return len(self._event_records)

    def get_all_events(self) -> List[EventRecord]:
        return list(self._event_records)
