"""
core/engine_core.py — Main pipeline orchestrator.

EngineCore wires all modules together and drives the frame loop.

Single-thread execution order per frame:
  FrameSource → Preprocessor → ROIManager → Tracker → FeatureExtractor
  → TemporalBuffer → InferenceEngine → DecisionEngine
  → VisualizationEngine → Logger / EventRecorder

Multithreaded mode (optional):
  capture_thread  → capture_queue
  process_thread  ← capture_queue → display_queue
  render_thread   ← display_queue
"""

from __future__ import annotations

import logging
import threading
import time
from queue import Empty, Queue
from typing import Optional

import cv2

from ai.decision_engine import DecisionEngine
from ai.inference_engine import InferenceEngine
from ai.model_manager import ModelManager
from config import SystemConfig
from core.data_models import FrameContext
from core.states import SystemState
from input.frame_source import create_frame_source
from logging_system.event_recorder import EventRecorder
from logging_system.logger import FrameLogger, setup_logging
from processing.feature_extractor import (
    FrameLevelFeatureExtractor,
    StandardFeatureExtractor,
)
from processing.preprocessor import Preprocessor
from processing.roi_manager import ROIManager
from processing.temporal_buffer import TemporalBuffer
from processing.tracker import TrackerRegistry
from utils.helpers import timer
from visualization.renderer import FrameRenderer

logger = logging.getLogger(__name__)


class EngineCore:
    """
    Top-level orchestrator. Initialises all subsystems and runs the pipeline.

    Usage::
        engine = EngineCore(cfg)
        engine.initialize()
        engine.run()
        engine.shutdown()
    """

    def __init__(self, cfg: SystemConfig):
        self._cfg = cfg
        self._running = False
        self._stop_event = threading.Event()

        # Subsystem references (populated in initialize())
        self._source = None
        self._preprocessor: Optional[Preprocessor] = None
        self._roi_manager: Optional[ROIManager] = None
        self._tracker_registry: Optional[TrackerRegistry] = None
        self._feat_extractor: Optional[StandardFeatureExtractor] = None
        self._frame_feat_extractor: Optional[FrameLevelFeatureExtractor] = None
        self._temporal_buffer: Optional[TemporalBuffer] = None
        self._model_manager: Optional[ModelManager] = None
        self._inference_engine: Optional[InferenceEngine] = None
        self._decision_engine: Optional[DecisionEngine] = None
        self._renderer: Optional[FrameRenderer] = None
        self._frame_logger: Optional[FrameLogger] = None
        self._event_recorder: Optional[EventRecorder] = None

        # State
        self._frame_index: int = 0
        self._prev_crops: dict = {}

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def initialize(self) -> None:
        """Build all subsystems. Must be called before run()."""
        setup_logging(self._cfg.logging.log_dir)
        logger.info("EngineCore initializing …")

        cfg = self._cfg

        # Frame source
        self._source = create_frame_source(
            source_type=cfg.video.source_type,
            source_path=cfg.video.source_path,
            camera_index=cfg.video.camera_index,
            fps=cfg.video.target_fps,
            loop=cfg.video.loop,
        )
        if not self._source.open():
            raise RuntimeError(f"Cannot open frame source: {cfg.video.source_path or cfg.video.camera_index}")
        logger.info("Frame source opened  fps=%.1f  frames=%d", self._source.fps, self._source.frame_count)

        # Preprocessor
        self._preprocessor = Preprocessor.from_config(cfg.preprocess)

        # ROI manager
        self._roi_manager = ROIManager.from_config(cfg.rois)

        # Tracker
        self._tracker_registry = TrackerRegistry(tracker_type="none")

        # Feature extractors
        self._feat_extractor = StandardFeatureExtractor()
        self._frame_feat_extractor = FrameLevelFeatureExtractor()

        # Temporal buffer
        self._temporal_buffer = TemporalBuffer(window_size=cfg.temporal.window_size)

        # AI subsystem
        self._model_manager = ModelManager(cfg.ai)
        self._model_manager.load_default()
        self._inference_engine = InferenceEngine(self._model_manager, cfg.ai)
        self._decision_engine = DecisionEngine(cfg.decision)

        # Visualization
        self._renderer = FrameRenderer(cfg.visualization)

        # Logging / event recording
        self._frame_logger = FrameLogger(cfg.logging)
        self._event_recorder = EventRecorder(cfg.logging, fps=self._source.fps)

        logger.info("EngineCore ready.")

    def run(self) -> None:
        """
        Run the main processing loop until the source is exhausted,
        max_frames is reached, the user presses 'q', or stop() is called.
        """
        if self._cfg.multithreaded:
            self._run_multithreaded()
        else:
            self._run_single_thread()

    def stop(self) -> None:
        """Signal the main loop to stop gracefully."""
        self._stop_event.set()

    def shutdown(self) -> None:
        """Release all resources and export logs."""
        logger.info("Shutting down …")
        self._stop_event.set()

        if self._source is not None:
            self._source.release()
        if self._renderer is not None:
            self._renderer.destroy()
        if self._frame_logger is not None:
            self._frame_logger.export_all()

        logger.info(
            "Shutdown complete.  Frames processed: %d  Events recorded: %d",
            self._frame_index,
            self._event_recorder.event_count if self._event_recorder else 0,
        )

    # ── Single-thread loop ────────────────────────────────────────────────

    def _run_single_thread(self) -> None:
        logger.info("Starting single-thread loop …")
        max_frames = self._cfg.max_frames

        while not self._stop_event.is_set():
            ok, raw_frame = self._source.read_frame()
            if not ok:
                logger.info("Source exhausted at frame %d", self._frame_index)
                break

            ctx = self.process_one_frame(raw_frame)

            # Render and display
            display_frame = self._renderer.render(ctx)
            key = self._renderer.show(display_frame)
            if key & 0xFF == ord("q"):
                logger.info("User pressed 'q', stopping")
                break

            # Max frame limit
            if max_frames is not None and self._frame_index >= max_frames:
                logger.info("max_frames=%d reached", max_frames)
                break

    def process_one_frame(self, raw_frame) -> FrameContext:
        """
        Run the full pipeline for one raw frame.
        Returns the populated FrameContext.
        """
        t_pipeline: dict = {}
        ctx = FrameContext(frame_index=self._frame_index, timestamp=time.time())
        ctx.raw_frame = raw_frame

        # 1. Preprocess
        with timer("preprocess", t_pipeline):
            ctx.processed_frame = self._preprocessor.process(raw_frame)

        # 2. Extract frame-level features
        with timer("frame_features", t_pipeline):
            ctx.frame_features = self._frame_feat_extractor.extract(ctx.processed_frame)

        # 3. ROI extraction
        with timer("roi_extract", t_pipeline):
            ctx.rois = self._roi_manager.extract(
                ctx.processed_frame,
                prev_crops=self._prev_crops,
            )

        # 4. (Optional) Tracker update
        with timer("tracking", t_pipeline):
            for roi in ctx.rois:
                ok, new_bbox = self._tracker_registry.update_roi(roi.roi_id, ctx.processed_frame)
                if ok and new_bbox:
                    roi.bbox = new_bbox
                    roi.tracking_active = True

        # 5. Feature extraction per ROI
        with timer("features", t_pipeline):
            for roi in ctx.rois:
                roi.features = self._feat_extractor.extract(roi.cropped, roi.cropped_prev)

        # 6. Push to temporal buffer
        self._temporal_buffer.push(ctx)

        # 7. Compute temporal summaries
        with timer("temporal", t_pipeline):
            if len(self._temporal_buffer) >= self._cfg.temporal.min_frames_for_decision:
                tcfg = self._cfg.decision
                summaries = self._temporal_buffer.compute_all_summaries(
                    stuck_variance_thresh=tcfg.stuck_variance_threshold,
                    stuck_min_frames=tcfg.stuck_min_frames,
                    drift_threshold=tcfg.drift_threshold,
                    oscillation_threshold=tcfg.oscillation_threshold,
                    oscillation_min_cycles=tcfg.oscillation_min_cycles,
                    sudden_change_threshold=tcfg.sudden_change_threshold,
                )
                for roi in ctx.rois:
                    roi.temporal_summary = summaries.get(roi.roi_id)

        # 8. AI Inference
        with timer("inference", t_pipeline):
            self._inference_engine.run(ctx, mode="roi")

        # 9. Decision engine
        with timer("decision", t_pipeline):
            self._decision_engine.decide(ctx)

        ctx.pipeline_times_ms = t_pipeline

        # 10. Event recording
        self._event_recorder.feed_frame(ctx)
        if ctx.is_event:
            self._event_recorder.trigger_event(ctx)

        # 11. Frame logging
        self._frame_logger.log_frame(ctx)

        # Bookkeeping
        self._prev_crops = self._roi_manager.get_prev_crops(ctx.rois)
        self._frame_index += 1

        return ctx

    # ── Multithreaded loop ────────────────────────────────────────────────

    def _run_multithreaded(self) -> None:
        """
        Three-thread design:
          capture_thread  → capture_queue (raw frames)
          process_thread  ← capture_queue → display_queue (FrameContext)
          render loop     ← display_queue (main thread)
        """
        QUEUE_MAXSIZE = 4
        capture_queue: Queue = Queue(maxsize=QUEUE_MAXSIZE)
        display_queue: Queue = Queue(maxsize=QUEUE_MAXSIZE)

        def capture_thread():
            max_frames = self._cfg.max_frames
            count = 0
            while not self._stop_event.is_set():
                ok, frame = self._source.read_frame()
                if not ok:
                    break
                capture_queue.put(frame)
                count += 1
                if max_frames and count >= max_frames:
                    break
            capture_queue.put(None)  # sentinel

        def process_thread():
            while not self._stop_event.is_set():
                try:
                    frame = capture_queue.get(timeout=1.0)
                except Empty:
                    continue
                if frame is None:
                    display_queue.put(None)
                    break
                ctx = self.process_one_frame(frame)
                display_queue.put(ctx)

        t1 = threading.Thread(target=capture_thread, daemon=True, name="capture")
        t2 = threading.Thread(target=process_thread, daemon=True, name="process")
        t1.start()
        t2.start()

        logger.info("Multithreaded loop started")

        # Main thread: render
        while not self._stop_event.is_set():
            try:
                ctx = display_queue.get(timeout=1.0)
            except Empty:
                continue
            if ctx is None:
                break
            display_frame = self._renderer.render(ctx)
            key = self._renderer.show(display_frame)
            if key & 0xFF == ord("q"):
                self._stop_event.set()
                break

        t1.join(timeout=3.0)
        t2.join(timeout=3.0)
