"""
gui/workers.py — Background engine thread and result dataclasses.

The EngineWorker runs in a daemon thread.
It calls EngineCore.process_one_frame() in a loop and posts FrameResult
objects to a thread-safe queue.  The Tkinter main thread polls that queue
with .after() and updates widgets.

No Tkinter calls are ever made from the worker thread.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from config import SystemConfig
from core.data_models import EventRecord, FrameContext, RuleResult
from core.states import SystemState
from input.frame_source import create_frame_source

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FrameResult — lightweight snapshot sent to the GUI thread each frame
# ---------------------------------------------------------------------------

@dataclass
class FrameResult:
    """Everything the GUI needs to refresh its widgets.

    display_frame is a CLEAN BGR frame (no OpenCV annotations baked in).
    All overlays are drawn by VideoCanvas as Tkinter items so they do not
    zoom or drift when the viewport is panned / zoomed.
    """
    frame_index: int = 0
    timestamp: float = 0.0
    display_frame: Optional[np.ndarray] = None      # clean BGR, no overlays
    system_state: SystemState = SystemState.UNKNOWN
    state_confidence: float = 0.0
    fps_actual: float = 0.0
    triggered_rules: List[RuleResult] = field(default_factory=list)
    roi_summaries: List[dict] = field(default_factory=list)  # lightweight dicts
    new_event: Optional[EventRecord] = None
    is_event: bool = False
    event_type: str = ""
    inference_ms: float = 0.0
    pipeline_ms: float = 0.0
    warning: str = ""


def _build_roi_summary(ctx: FrameContext) -> List[dict]:
    out = []
    for roi in ctx.rois:
        d = {
            "roi_id":    roi.roi_id,
            "state":     roi.roi_state.value,
            "mean_int":  round(roi.features.get("mean_intensity", 0.0), 1),
            "diff":      round(roi.features.get("mean_abs_diff",  0.0), 1),
            "ai_label":  roi.detection.label      if roi.detection else "—",
            "ai_conf":   round(roi.detection.confidence, 3) if roi.detection else 0.0,
            "stuck":     bool(roi.temporal_summary and roi.temporal_summary.is_stuck),
            "drift":     bool(roi.temporal_summary and roi.temporal_summary.is_drifting),
            "oscillate": bool(roi.temporal_summary and roi.temporal_summary.is_oscillating),
        }
        out.append(d)
    return out


# ---------------------------------------------------------------------------
# EngineWorker
# ---------------------------------------------------------------------------

class EngineWorker:
    """
    Manages the EngineCore lifecycle in a background thread.

    Public API (called from Tkinter main thread):
      start()  / stop()  / is_running
      result_queue  — consume FrameResult objects with .get_nowait()
    """

    def __init__(self, cfg: SystemConfig):
        self._cfg = cfg
        self.result_queue: queue.Queue[FrameResult] = queue.Queue(maxsize=8)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._engine = None
        self._error: Optional[str] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._error = None
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="engine-worker"
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._engine:
            self._engine.stop()

    def join(self, timeout: float = 3.0) -> None:
        if self._thread:
            self._thread.join(timeout=timeout)

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def last_error(self) -> Optional[str]:
        return self._error

    # ── Internal thread body ──────────────────────────────────────────────

    def _run(self) -> None:
        from ai.decision_engine import DecisionEngine
        from ai.inference_engine import InferenceEngine
        from ai.model_manager import ModelManager
        from core.data_models import FrameContext
        from core.engine_core import EngineCore
        from logging_system.event_recorder import EventRecorder
        from logging_system.logger import FrameLogger, setup_logging
        from processing.feature_extractor import (
            FrameLevelFeatureExtractor, StandardFeatureExtractor,
        )
        from processing.preprocessor import Preprocessor
        from processing.roi_manager import ROIManager
        from processing.temporal_buffer import TemporalBuffer
        from processing.tracker import TrackerRegistry
        from visualization.overlay_drawer import OverlayDrawer
        import time

        try:
            setup_logging(self._cfg.logging.log_dir)
            logger.info("EngineWorker: initialising …")

            source = create_frame_source(
                source_type=self._cfg.video.source_type,
                source_path=self._cfg.video.source_path,
                camera_index=self._cfg.video.camera_index,
                fps=self._cfg.video.target_fps,
                loop=self._cfg.video.loop,
            )
            if not source.open():
                raise RuntimeError(
                    f"Cannot open source: "
                    f"{self._cfg.video.source_path or self._cfg.video.camera_index}"
                )

            preprocessor   = Preprocessor.from_config(self._cfg.preprocess)
            roi_manager     = ROIManager.from_config(self._cfg.rois)
            tracker_reg     = TrackerRegistry(tracker_type="none")
            feat_ext        = StandardFeatureExtractor()
            frame_feat_ext  = FrameLevelFeatureExtractor()
            temporal_buf    = TemporalBuffer(self._cfg.temporal.window_size)
            model_mgr       = ModelManager(self._cfg.ai)
            model_mgr.load_default()
            inf_engine      = InferenceEngine(model_mgr, self._cfg.ai)
            dec_engine      = DecisionEngine(self._cfg.decision)
            frame_logger    = FrameLogger(self._cfg.logging)
            event_rec       = EventRecorder(self._cfg.logging, fps=source.fps)

            from utils.helpers import timer

            frame_index = 0
            prev_crops: dict = {}
            fps_timer = time.perf_counter()
            fps_count = 0
            fps_actual = 0.0

            logger.info("EngineWorker: loop started")

            while not self._stop_event.is_set():
                ok, raw = source.read_frame()
                if not ok:
                    logger.info("EngineWorker: source exhausted")
                    break

                t_pipe: dict = {}
                ctx = FrameContext(frame_index=frame_index, timestamp=time.time())
                ctx.raw_frame = raw

                with timer("preprocess",   t_pipe): ctx.processed_frame = preprocessor.process(raw)
                with timer("frame_feats",  t_pipe): ctx.frame_features   = frame_feat_ext.extract(ctx.processed_frame)
                with timer("roi_extract",  t_pipe): ctx.rois             = roi_manager.extract(ctx.processed_frame, prev_crops)
                with timer("features",     t_pipe):
                    for roi in ctx.rois:
                        roi.features = feat_ext.extract(roi.cropped, roi.cropped_prev)

                temporal_buf.push(ctx)

                with timer("temporal", t_pipe):
                    cfg_d = self._cfg.decision
                    if len(temporal_buf) >= self._cfg.temporal.min_frames_for_decision:
                        sums = temporal_buf.compute_all_summaries(
                            stuck_variance_thresh=cfg_d.stuck_variance_threshold,
                            stuck_min_frames=cfg_d.stuck_min_frames,
                            drift_threshold=cfg_d.drift_threshold,
                            oscillation_threshold=cfg_d.oscillation_threshold,
                            oscillation_min_cycles=cfg_d.oscillation_min_cycles,
                            sudden_change_threshold=cfg_d.sudden_change_threshold,
                        )
                        for roi in ctx.rois:
                            roi.temporal_summary = sums.get(roi.roi_id)

                with timer("inference", t_pipe): inf_engine.run(ctx, mode="roi")
                with timer("decision",  t_pipe): dec_engine.decide(ctx)

                ctx.pipeline_times_ms = t_pipe
                event_rec.feed_frame(ctx)
                new_event = None
                if ctx.is_event:
                    new_event = event_rec.trigger_event(ctx)
                frame_logger.log_frame(ctx)

                # ── Clean display frame (NO annotations baked in) ────────────
                # All overlays (ROI boxes, state banner, frame info, event marker)
                # are drawn by VideoCanvas as Tkinter canvas items AFTER the
                # viewport crop, so they never zoom or drift with the viewport.
                import cv2
                display = (ctx.processed_frame if ctx.processed_frame is not None else ctx.raw_frame).copy()
                if display.ndim == 2:
                    display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)

                # FPS calc
                fps_count += 1
                now = time.perf_counter()
                if now - fps_timer >= 1.0:
                    fps_actual = fps_count / (now - fps_timer)
                    fps_count = 0
                    fps_timer = now

                result = FrameResult(
                    frame_index=frame_index,
                    timestamp=ctx.timestamp,
                    display_frame=display,
                    system_state=ctx.system_state,
                    state_confidence=ctx.state_confidence,
                    fps_actual=fps_actual,
                    triggered_rules=list(ctx.triggered_rules),
                    roi_summaries=_build_roi_summary(ctx),
                    new_event=new_event,
                    is_event=ctx.is_event,
                    event_type=ctx.event_type if ctx.is_event else "",
                    inference_ms=ctx.inference_time_ms,
                    pipeline_ms=sum(t_pipe.values()),
                )

                try:
                    self.result_queue.put_nowait(result)
                except queue.Full:
                    pass  # GUI is slow; drop frame to keep latency low

                prev_crops = roi_manager.get_prev_crops(ctx.rois)
                frame_index += 1

                if self._cfg.max_frames and frame_index >= self._cfg.max_frames:
                    break

            source.release()
            frame_logger.export_all()
            logger.info("EngineWorker: finished  frames=%d  events=%d",
                        frame_index, event_rec.event_count)

        except Exception as exc:
            logger.exception("EngineWorker crashed: %s", exc)
            self._error = str(exc)
            # Push a sentinel so GUI can show the error
            try:
                self.result_queue.put_nowait(FrameResult(warning=str(exc)))
            except queue.Full:
                pass
