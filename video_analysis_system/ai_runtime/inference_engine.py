"""
ai_runtime/inference_engine.py — 런타임 추론 엔진

ai/inference_engine.py 를 기반으로 RuntimeModelManager 와 연동하고
추론 통계(지연 시간 이력, 처리량) 를 추가로 수집한다.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Deque, Optional

import cv2
import numpy as np

from ai_runtime.model_manager import RuntimeModelManager
from config import AIConfig
from core.data_models import DetectionResult, FrameContext, ROIData

logger = logging.getLogger(__name__)


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


class RuntimeInferenceEngine:
    """
    RuntimeModelManager 를 사용하는 추론 엔진.

    동작 방식은 ai/InferenceEngine 과 동일하지만:
      - RuntimeModelManager.default 에서 활성 모델을 동적으로 가져옴
      - 최근 N 프레임의 추론 지연 시간을 추적
      - 처리량(frames/sec) 계산 지원
    """

    LATENCY_WINDOW = 60   # 이동 평균 윈도우 크기

    def __init__(self, model_manager: RuntimeModelManager, cfg: AIConfig):
        self._mm = model_manager
        self._cfg = cfg
        self._class_names = cfg.class_names
        self._latencies: Deque[float] = deque(maxlen=self.LATENCY_WINDOW)

    # ── 추론 실행 ─────────────────────────────────────────────────────────

    def run(self, ctx: FrameContext, mode: str = "roi") -> None:
        """FrameContext 에 DetectionResult 를 채운다 (in-place)."""
        model = self._mm.default
        if not model.is_loaded:
            ctx.add_warning("RuntimeInferenceEngine: 모델 미로드, 추론 생략")
            return

        t0 = time.perf_counter()

        if mode in ("frame", "both") and ctx.processed_frame is not None:
            ctx.frame_detection = self._infer_frame(model, ctx.processed_frame)

        if mode in ("roi", "both"):
            for roi in ctx.rois:
                if roi.cropped is not None:
                    roi.detection = self._infer_roi(model, roi)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        ctx.inference_time_ms = elapsed_ms
        self._latencies.append(elapsed_ms)

    # ── 내부 헬퍼 ────────────────────────────────────────────────────────

    def _infer_frame(self, model, frame: np.ndarray) -> DetectionResult:
        try:
            raw = model.predict(frame)
            scores = _softmax(raw.astype(np.float32))
            class_id = int(np.argmax(scores))
            label = (self._class_names[class_id]
                     if class_id < len(self._class_names) else str(class_id))
            return DetectionResult(
                class_id=class_id,
                label=label,
                confidence=float(scores[class_id]),
                scores=scores,
                source_model_name=self._mm.active_name,
            )
        except Exception as exc:
            logger.error("프레임 추론 오류: %s", exc)
            return DetectionResult(label="error", confidence=0.0)

    def _infer_roi(self, model, roi: ROIData) -> DetectionResult:
        try:
            raw = model.predict(roi.cropped)
            scores = _softmax(raw.astype(np.float32))
            class_id = int(np.argmax(scores))
            label = (self._class_names[class_id]
                     if class_id < len(self._class_names) else str(class_id))
            return DetectionResult(
                class_id=class_id,
                label=label,
                confidence=float(scores[class_id]),
                scores=scores,
                bbox=roi.original_rect,
                source_model_name=self._mm.active_name,
            )
        except Exception as exc:
            logger.error("ROI '%s' 추론 오류: %s", roi.roi_id, exc)
            return DetectionResult(label="error", confidence=0.0)

    # ── 통계 ─────────────────────────────────────────────────────────────

    @property
    def avg_latency_ms(self) -> float:
        """최근 N 프레임의 평균 추론 지연 시간 (ms)."""
        return float(np.mean(self._latencies)) if self._latencies else 0.0

    @property
    def p95_latency_ms(self) -> float:
        """최근 N 프레임의 95th 백분위 추론 지연 시간 (ms)."""
        if not self._latencies:
            return 0.0
        return float(np.percentile(list(self._latencies), 95))

    def get_stats(self) -> dict:
        return {
            "avg_ms":   self.avg_latency_ms,
            "p95_ms":   self.p95_latency_ms,
            "samples":  len(self._latencies),
            "model":    self._mm.active_name,
        }
