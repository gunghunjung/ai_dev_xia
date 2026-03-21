"""
ai/inference_engine.py — Runs the AI model and maps output to DetectionResult.

InferenceEngine is intentionally thin: it bridges ModelManager ↔ FrameContext.
It does NOT make final decisions (that is DecisionEngine's responsibility).
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import cv2
import numpy as np

from ai.model_manager import ModelManager
from config import AIConfig
from core.data_models import DetectionResult, FrameContext, ROIData

logger = logging.getLogger(__name__)


class InferenceEngine:
    """
    Runs AI inference on ROI crops (or the full frame) and attaches
    DetectionResult objects back to FrameContext.

    Decision mode:
      - "roi"   : run inference on each ROI crop independently
      - "frame" : run inference on the whole processed frame
      - "both"  : run frame-level inference AND per-ROI inference
    """

    def __init__(self, model_manager: ModelManager, cfg: AIConfig):
        self._mm = model_manager
        self._cfg = cfg
        self._class_names = cfg.class_names

    def run(self, ctx: FrameContext, mode: str = "roi") -> None:
        """
        Populate DetectionResult fields inside *ctx*.
        This method mutates ctx in-place.
        """
        model = self._mm.default
        if not model.is_loaded:
            ctx.add_warning("InferenceEngine: model not loaded, skipping inference")
            return

        t0 = time.perf_counter()

        if mode in ("frame", "both") and ctx.processed_frame is not None:
            ctx.frame_detection = self._infer_frame(model, ctx.processed_frame)

        if mode in ("roi", "both"):
            for roi in ctx.rois:
                if roi.cropped is not None:
                    roi.detection = self._infer_roi(model, roi)

        ctx.inference_time_ms = (time.perf_counter() - t0) * 1000.0

    # ── Internal helpers ──────────────────────────────────────────────────

    def _infer_frame(self, model, frame: np.ndarray) -> DetectionResult:
        try:
            scores = model.predict(frame)
            return self._scores_to_result(scores)
        except Exception as exc:
            logger.error("Frame inference failed: %s", exc)
            return DetectionResult(label="error", confidence=0.0)

    def _infer_roi(self, model, roi: ROIData) -> DetectionResult:
        try:
            scores = model.predict(roi.cropped)
            result = self._scores_to_result(scores)
            result.bbox = roi.bbox
            return result
        except Exception as exc:
            logger.error("ROI inference failed for %s: %s", roi.roi_id, exc)
            return DetectionResult(label="error", confidence=0.0)

    def _scores_to_result(self, raw_scores: np.ndarray) -> DetectionResult:
        """Map raw model output to a DetectionResult."""
        if raw_scores.ndim > 1:
            raw_scores = raw_scores.flatten()

        # Softmax normalisation (in case model does not output probabilities)
        scores = self._softmax(raw_scores)

        class_id = int(np.argmax(scores))
        confidence = float(scores[class_id])
        label = (
            self._class_names[class_id]
            if class_id < len(self._class_names)
            else str(class_id)
        )
        return DetectionResult(
            class_id=class_id,
            label=label,
            confidence=confidence,
            raw_scores=scores,
        )

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        x = x - x.max()
        e = np.exp(x)
        return e / (e.sum() + 1e-9)
