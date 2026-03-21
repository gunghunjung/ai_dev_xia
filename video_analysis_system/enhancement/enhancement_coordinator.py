"""
enhancement/enhancement_coordinator.py

EnhancementCoordinator integrates the enhancement pipeline into the analysis pipeline.
It decides WHEN to enhance (PRE_PREPROCESS / POST_PREPROCESS / ROI_AFTER_EXTRACT),
WHAT to enhance (full frame or ROI), and WHERE the enhanced output goes
(display / analysis / both).

Holds:
  - model_manager:   EnhancementModelManager
  - preprocessor:    EnhancementPreprocessor
  - postprocessor:   EnhancementPostprocessor
  - cfg:             EnhancementConfig
  - _cache:          OrderedDict  (cache_key -> EnhancementResult, LRU up to
                                   cfg.max_cache_size)

Public API
──────────
  __init__(cfg: EnhancementConfig)
  process_frame(ctx: FrameContext) -> FrameContext
  preview_enhance(image: np.ndarray, task: EnhancementType = None) -> np.ndarray
  update_config(new_cfg: EnhancementConfig) -> None
  clear_cache() -> None
  get_stats() -> dict
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections import OrderedDict
from typing import Dict, List, Optional

import cv2
import numpy as np

from enhancement.enhancement_config import (
    EnhancementConfig,
    EnhancementMode,
    EnhancementScope,
    EnhancementType,
)
from enhancement.enhancement_model_manager import EnhancementModelManager
from enhancement.enhancement_preprocessor import EnhancementPreprocessor
from enhancement.enhancement_postprocessor import EnhancementPostprocessor
from enhancement.data_model_extensions import EnhancedFrameInfo, EnhancementResult

# FrameContext / ROIData are imported lazily to avoid circular dependency issues
# when the coordinator module is imported before the core package is fully loaded.
try:
    from core.data_models import FrameContext, ROIData  # type: ignore
except ImportError:  # pragma: no cover
    FrameContext = object  # type: ignore
    ROIData = object  # type: ignore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EnhancementCoordinator
# ---------------------------------------------------------------------------

class EnhancementCoordinator:
    """
    Orchestrates the full image-enhancement subsystem.

    Lifecycle
    ─────────
    1.  Instantiate with an EnhancementConfig.
    2.  Call process_frame() once per frame from EngineCore / workers.py.
    3.  Optionally call preview_enhance() from the GUI preview panel.
    4.  Call update_config() to hot-swap settings without restarting.
    """

    def __init__(self, cfg: EnhancementConfig) -> None:
        self.cfg = cfg
        self.model_manager = EnhancementModelManager(cfg)
        self.preprocessor   = EnhancementPreprocessor(cfg)
        self.postprocessor  = EnhancementPostprocessor(cfg)

        # LRU cache:  key -> EnhancementResult
        self._cache: OrderedDict[str, EnhancementResult] = OrderedDict()

        # Statistics
        self._cache_hits:   int   = 0
        self._cache_misses: int   = 0
        self._total_frames: int   = 0
        self._total_ms:     float = 0.0

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def process_frame(self, ctx: "FrameContext") -> "FrameContext":
        """
        Main entry point called from EngineCore / workers.py.

        Reads self.cfg to decide scope and mode:

        EnhancementScope.FULL_FRAME
            Enhance ctx.processed_frame (falls back to ctx.raw_frame if
            processed_frame is None).  Stores the result in
            ctx.enhanced_frame.  When mode is DUAL or ANALYSIS_ONLY,
            also replaces ctx.processed_frame with the enhanced image.

        EnhancementScope.ROI_ONLY / SELECTED_ROI
            For each qualifying ROI in ctx.rois:
              - Enhance roi.cropped.
              - Store the result in roi.enhanced_crop.
              - When mode is DUAL or ANALYSIS_ONLY, also replace
                roi.cropped with roi.enhanced_crop.

        A lightweight EnhancedFrameInfo summary is placed in
        ctx.debug_info["enhancement"] so downstream components can
        inspect what happened.

        Returns the mutated ctx.
        """
        if not self.cfg.enabled:
            return ctx

        frame_start = time.perf_counter()
        self._total_frames += 1

        active_types = self.cfg.active_types()
        if not active_types:
            return ctx

        debug_info = EnhancedFrameInfo(
            enabled=True,
            scope=self.cfg.scope.value,
            mode=self.cfg.mode.value,
        )

        try:
            if self.cfg.scope == EnhancementScope.FULL_FRAME:
                ctx = self._process_full_frame(ctx, active_types, debug_info)

            elif self.cfg.scope in (EnhancementScope.ROI_ONLY,
                                    EnhancementScope.SELECTED_ROI):
                ctx = self._process_rois(ctx, active_types, debug_info)

        except Exception as exc:  # pragma: no cover
            logger.error("EnhancementCoordinator.process_frame error: %s", exc, exc_info=True)
            ctx.warnings.append(f"enhancement error: {exc}")

        elapsed_ms = (time.perf_counter() - frame_start) * 1000.0
        debug_info.total_ms = elapsed_ms
        self._total_ms += elapsed_ms

        # Store summary on the context
        if not hasattr(ctx, "debug_info"):
            ctx.debug_info = {}
        ctx.debug_info["enhancement"] = debug_info

        return ctx

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------

    def preview_enhance(
        self,
        image: np.ndarray,
        task: Optional[EnhancementType] = None,
    ) -> np.ndarray:
        """
        Single-frame on-demand enhancement for the GUI preview panel.

        Uses all active_types() when task is None.
        Returns a BGR uint8 image.
        """
        if not self.cfg.preview_enabled:
            return image

        tasks = [task] if task is not None else self.cfg.active_types()
        if not tasks:
            return image

        result_image = image.copy()
        for t in tasks:
            try:
                result = self._enhance_image(result_image, t)
                if result.success and result.enhanced_image is not None:
                    result_image = result.enhanced_image
            except Exception as exc:  # pragma: no cover
                logger.warning("preview_enhance task=%s error: %s", t, exc)

        return result_image

    # ------------------------------------------------------------------
    # Config management
    # ------------------------------------------------------------------

    def update_config(self, new_cfg: EnhancementConfig) -> None:
        """Hot-swap configuration without restarting."""
        self.cfg = new_cfg
        self.model_manager._cfg = new_cfg
        self.preprocessor._cfg  = new_cfg
        self.postprocessor._cfg = new_cfg
        self.clear_cache()
        logger.debug("EnhancementCoordinator config updated")

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    def clear_cache(self) -> None:
        """Evict all cached results."""
        self._cache.clear()
        logger.debug("Enhancement cache cleared")

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """
        Returns a summary dict:
            cache_hits, cache_misses, total_frames, avg_ms_per_frame,
            active_models (dict of task -> model_name).
        """
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0.0
        avg_ms   = self._total_ms / self._total_frames if self._total_frames > 0 else 0.0

        active_models: Dict[str, str] = {}
        for t in EnhancementType:
            meta = self.model_manager.get_active(t)
            if meta is not None:
                active_models[t.value] = meta.model_name

        return {
            "cache_hits":        self._cache_hits,
            "cache_misses":      self._cache_misses,
            "cache_hit_rate":    round(hit_rate, 4),
            "total_frames":      self._total_frames,
            "avg_ms_per_frame":  round(avg_ms, 3),
            "active_models":     active_models,
        }

    # ------------------------------------------------------------------
    # Private — full-frame path
    # ------------------------------------------------------------------

    def _process_full_frame(
        self,
        ctx: "FrameContext",
        active_types: List[EnhancementType],
        debug_info: EnhancedFrameInfo,
    ) -> "FrameContext":
        """Enhance the full processed (or raw) frame."""
        source = ctx.processed_frame if ctx.processed_frame is not None else ctx.raw_frame
        if source is None:
            debug_info.warnings.append("no source frame available for full-frame enhancement")
            return ctx

        enhanced = source.copy()
        for task in active_types:
            result = self._enhance_image(enhanced, task)
            debug_info.results.append(result)
            if result.applied_model_name not in debug_info.models_used:
                debug_info.models_used.append(result.applied_model_name)
            if result.success and result.enhanced_image is not None:
                enhanced = result.enhanced_image

        # Store on the context via the attribute if it exists, otherwise via
        # debug_info only (ctx.enhanced_frame is an optional extension field).
        if hasattr(ctx, "enhanced_frame"):
            ctx.enhanced_frame = enhanced
        else:
            # Attach dynamically as a plain attribute — the coordinator
            # spec requires this field even if data_models.py does not define it.
            object.__setattr__(ctx, "enhanced_frame", enhanced)

        # Propagate to analysis path when required
        if self.cfg.mode in (EnhancementMode.DUAL, EnhancementMode.ANALYSIS_ONLY):
            ctx.processed_frame = enhanced

        return ctx

    # ------------------------------------------------------------------
    # Private — ROI path
    # ------------------------------------------------------------------

    def _process_rois(
        self,
        ctx: "FrameContext",
        active_types: List[EnhancementType],
        debug_info: EnhancedFrameInfo,
    ) -> "FrameContext":
        """Enhance individual ROI crops."""
        # Determine the ID of the selected ROI (if SELECTED_ROI mode)
        selected_roi_id: Optional[str] = None
        if self.cfg.scope == EnhancementScope.SELECTED_ROI:
            for roi in ctx.rois:
                if getattr(roi, "selected", False):
                    selected_roi_id = roi.roi_id
                    break

        for roi in ctx.rois:
            if not self._should_process_roi(roi, selected_roi_id):
                continue

            if roi.cropped is None:
                continue

            enhanced_crop = roi.cropped.copy()
            for task in active_types:
                result = self._enhance_image(enhanced_crop, task, roi_id=roi.roi_id)
                debug_info.results.append(result)
                if result.applied_model_name not in debug_info.models_used:
                    debug_info.models_used.append(result.applied_model_name)
                if result.success and result.enhanced_image is not None:
                    enhanced_crop = result.enhanced_image

            # Store the enhanced crop on the ROI via attribute (optional field)
            if hasattr(roi, "enhanced_crop"):
                roi.enhanced_crop = enhanced_crop
            else:
                object.__setattr__(roi, "enhanced_crop", enhanced_crop)

            # Propagate to analysis path when required
            if self.cfg.mode in (EnhancementMode.DUAL, EnhancementMode.ANALYSIS_ONLY):
                roi.cropped = enhanced_crop

        return ctx

    # ------------------------------------------------------------------
    # Private — single-image enhancement with caching
    # ------------------------------------------------------------------

    def _enhance_image(
        self,
        image: np.ndarray,
        task: EnhancementType,
        roi_id: Optional[str] = None,
    ) -> EnhancementResult:
        """
        Run one enhancement task on a single image.

        Checks the LRU cache first (when cfg.cache_results is True).
        Falls back to the registered OpenCV routines when no AI model is
        loaded for the requested task.
        """
        cache_key = self._cache_key(image, task) if self.cfg.cache_results else None

        # Cache lookup
        if cache_key is not None and cache_key in self._cache:
            self._cache_hits += 1
            cached = self._cache[cache_key]
            # Move to end (most-recently-used)
            self._cache.move_to_end(cache_key)
            result = EnhancementResult(
                mode=self.cfg.mode.value,
                applied_model_name=cached.applied_model_name,
                applied_model_version=cached.applied_model_version,
                enhancement_type=task.value,
                input_shape=image.shape,
                output_shape=cached.output_shape,
                scale_factor=self.cfg.scale_factor,
                processing_time_ms=0.0,
                quality_notes="cache hit",
                enhanced_image=cached.enhanced_image.copy() if cached.enhanced_image is not None else None,
                roi_id=roi_id,
                success=cached.success,
            )
            return result

        self._cache_misses += 1

        # Run the enhancement
        t0 = time.perf_counter()
        result = self._run_enhancement(image, task, roi_id)
        result.processing_time_ms = (time.perf_counter() - t0) * 1000.0

        # Cache insert
        if cache_key is not None and result.success:
            self._cache[cache_key] = result
            self._cache.move_to_end(cache_key)
            # Evict oldest when over capacity
            while len(self._cache) > self.cfg.max_cache_size:
                self._cache.popitem(last=False)

        return result

    def _run_enhancement(
        self,
        image: np.ndarray,
        task: EnhancementType,
        roi_id: Optional[str],
    ) -> EnhancementResult:
        """
        Execute enhancement for a single task using the active model.

        Tries the active model registered in model_manager.  When the
        model is OpenCV-based (or no AI model is available), delegates to
        the built-in CV2 fallback routines.
        """
        input_shape = tuple(image.shape)
        result = EnhancementResult(
            mode=self.cfg.mode.value,
            enhancement_type=task.value,
            input_shape=input_shape,
            scale_factor=self.cfg.scale_factor,
            roi_id=roi_id,
        )

        meta = self.model_manager.get_active(task)
        if meta is None:
            # No model registered for this task — use direct CV2 fallback
            enhanced = self._cv2_fallback(image, task)
            result.applied_model_name    = "cv2_fallback"
            result.applied_model_version = "1.0"
            result.enhanced_image        = enhanced
            result.output_shape          = tuple(enhanced.shape)
            result.success               = True
            result.quality_notes         = "no model registered; used cv2 fallback"
            return result

        result.applied_model_name    = meta.model_name
        result.applied_model_version = meta.version

        if meta.framework == "opencv":
            enhanced = self._cv2_fallback(image, task)
            result.enhanced_image = enhanced
            result.output_shape   = tuple(enhanced.shape)
            result.success        = True
            result.quality_notes  = meta.quality_note
            return result

        # AI backend (onnx / torch)
        session = self.model_manager.get_session(meta.model_name)
        if session is None:
            # Model not loaded yet — fall back to CV2
            logger.warning(
                "Model '%s' not loaded; falling back to cv2 for task %s",
                meta.model_name, task,
            )
            enhanced = self._cv2_fallback(image, task)
            result.applied_model_name = f"{meta.model_name}(fallback)"
            result.enhanced_image     = enhanced
            result.output_shape       = tuple(enhanced.shape)
            result.success            = True
            result.quality_notes      = "model not loaded; cv2 fallback"
            return result

        try:
            prepared = self.preprocessor.prepare(image, meta)
            raw_output = self._run_ai_session(session, prepared, meta)
            enhanced = self.postprocessor.postprocess(
                raw_output,
                original_shape=(input_shape[0], input_shape[1]),
                task=task,
                meta=meta,
            )
            result.enhanced_image = enhanced
            result.output_shape   = tuple(enhanced.shape)
            result.success        = True
            result.quality_notes  = meta.quality_note
        except Exception as exc:  # pragma: no cover
            logger.error(
                "AI enhancement failed for task %s model %s: %s",
                task, meta.model_name, exc, exc_info=True,
            )
            enhanced = self._cv2_fallback(image, task)
            result.enhanced_image = enhanced
            result.output_shape   = tuple(enhanced.shape)
            result.success        = True
            result.quality_notes  = f"ai failed ({exc}); cv2 fallback used"

        return result

    # ------------------------------------------------------------------
    # Private — AI session runner
    # ------------------------------------------------------------------

    @staticmethod
    def _run_ai_session(session: object, prepared: np.ndarray, meta) -> np.ndarray:
        """
        Dispatch prepared input to the appropriate AI runtime.

        Supports onnxruntime InferenceSession and torch nn.Module.
        Returns a numpy array (may be float or uint8).
        """
        framework = meta.framework

        if framework == "onnx":
            import onnxruntime as ort  # type: ignore  # noqa: F401
            # ONNX models typically expect (N, C, H, W) float32
            inp = prepared
            if inp.ndim == 3:
                # (H, W, C) → (1, C, H, W)
                inp = np.transpose(inp, (2, 0, 1))[np.newaxis, ...]
            inp = inp.astype(np.float32)
            input_name = session.get_inputs()[0].name
            output = session.run(None, {input_name: inp})[0]
            return output

        if framework == "torch":
            import torch  # type: ignore
            inp = prepared
            if inp.ndim == 3:
                inp = np.transpose(inp, (2, 0, 1))[np.newaxis, ...]
            tensor = torch.from_numpy(inp.astype(np.float32))
            with torch.no_grad():
                out = session(tensor)
            return out.cpu().numpy()

        raise ValueError(f"Unsupported framework: {framework}")

    # ------------------------------------------------------------------
    # Private — OpenCV fallback algorithms
    # ------------------------------------------------------------------

    @staticmethod
    def _cv2_fallback(image: np.ndarray, task: EnhancementType) -> np.ndarray:
        """
        Lightweight OpenCV implementations used when no AI model is
        available or loaded.

        All routines accept and return BGR uint8 images.
        """
        # Ensure uint8 BGR input
        img = image
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        if task == EnhancementType.SHARPENING:
            # Unsharp mask
            blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=3)
            sharpened = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
            return sharpened

        if task == EnhancementType.DENOISING:
            # Non-local means denoising
            try:
                denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
            except cv2.error:
                # fastNlMeansDenoising requires uint8 C3
                denoised = cv2.bilateralFilter(img, 9, 75, 75)
            return denoised

        if task == EnhancementType.DEBLURRING:
            # Simple Laplacian-based sharpening as a deblur approximation
            kernel = np.array([[0, -1, 0],
                                [-1, 5, -1],
                                [0, -1, 0]], dtype=np.float32)
            deblurred = cv2.filter2D(img, -1, kernel)
            return deblurred

        if task == EnhancementType.SUPER_RESOLUTION:
            # Bicubic upscale — real SR models are handled in the AI path
            scale = max(1, int(2))  # default 2×; cfg.scale_factor used above
            h, w = img.shape[:2]
            upscaled = cv2.resize(img, (w * scale, h * scale),
                                  interpolation=cv2.INTER_CUBIC)
            return upscaled

        if task == EnhancementType.COMBINED:
            # Chain denoising → sharpening
            step1 = EnhancementCoordinator._cv2_fallback(img, EnhancementType.DENOISING)
            step2 = EnhancementCoordinator._cv2_fallback(step1, EnhancementType.SHARPENING)
            return step2

        return img.copy()

    # ------------------------------------------------------------------
    # Private — helpers
    # ------------------------------------------------------------------

    def _should_process_roi(
        self,
        roi: "ROIData",
        selected_roi_id: Optional[str],
    ) -> bool:
        """
        Return True if the given ROI should be enhanced in this pass.

        - FULL_FRAME scope:   always False (handled separately).
        - ROI_ONLY scope:     True for every visible ROI.
        - SELECTED_ROI scope: True only for the ROI whose id matches
                              selected_roi_id.
        """
        if self.cfg.scope == EnhancementScope.FULL_FRAME:
            return False

        if not getattr(roi, "visible", True):
            return False

        if self.cfg.scope == EnhancementScope.SELECTED_ROI:
            return roi.roi_id == selected_roi_id

        # ROI_ONLY: all visible ROIs
        return True

    @staticmethod
    def _cache_key(image: np.ndarray, task: EnhancementType) -> str:
        """
        Compute a deterministic cache key for (image, task).

        Uses MD5 of the raw image bytes concatenated with the task value
        string.  Fast enough for typical frame sizes; not
        cryptographically sensitive.
        """
        h = hashlib.md5(image.tobytes() + task.value.encode())
        return h.hexdigest()
