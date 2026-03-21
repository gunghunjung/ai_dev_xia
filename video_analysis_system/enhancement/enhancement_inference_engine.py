"""
enhancement/enhancement_inference_engine.py — Unified inference entry point.

Routes each EnhancementType through the correct backend (OpenCV / ONNX / Torch),
wraps pre/postprocessing, and returns a structured EnhancementResult.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np

from enhancement.enhancement_config import EnhancementConfig, EnhancementType
from enhancement.enhancement_model_manager import EnhancementModelManager
from enhancement.enhancement_preprocessor import EnhancementPreprocessor
from enhancement.enhancement_postprocessor import EnhancementPostprocessor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class EnhancementResult:
    """Carries the result of a single enhancement operation."""

    mode: str = ""
    applied_model_name: str = ""
    applied_model_version: str = ""
    enhancement_type: str = ""
    input_shape: Tuple = ()
    output_shape: Tuple = ()
    scale_factor: int = 1
    processing_time_ms: float = 0.0
    quality_notes: str = ""
    enhanced_image: Optional[np.ndarray] = None
    roi_id: Optional[str] = None
    success: bool = False


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class EnhancementInferenceEngine:
    """
    Orchestrates preprocessing → inference → postprocessing for one or
    all active enhancement tasks.
    """

    def __init__(
        self,
        model_manager: EnhancementModelManager,
        cfg: EnhancementConfig,
    ) -> None:
        self._mm = model_manager
        self._cfg = cfg
        self._pre = EnhancementPreprocessor(cfg)
        self._post = EnhancementPostprocessor(cfg)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enhance(
        self,
        image: np.ndarray,
        task: EnhancementType,
        roi_rect: Optional[Tuple[int, int, int, int]] = None,
    ) -> EnhancementResult:
        """
        Apply a single enhancement task to an image (or ROI crop within it).

        Args:
            image:    BGR uint8 input frame or crop.
            task:     Which enhancement to apply.
            roi_rect: If given, extract this (x, y, w, h) sub-region,
                      enhance it, and paste it back into the full image.

        Returns:
            EnhancementResult with enhanced_image set on success.
        """
        t0 = time.perf_counter()
        result = EnhancementResult(
            enhancement_type=task.value,
            mode=self._cfg.mode.value,
            input_shape=tuple(image.shape),
        )

        meta = self._mm.get_active(task)
        if meta is None:
            logger.warning("No active model for task '%s'", task)
            result.enhanced_image = image.copy()
            result.output_shape = tuple(image.shape)
            return result

        result.applied_model_name = meta.model_name
        result.applied_model_version = meta.version
        result.scale_factor = meta.scale_factor
        result.quality_notes = meta.quality_note

        # Crop ROI if requested
        base_frame = image
        work_image = image
        if roi_rect is not None:
            x, y, w, h = roi_rect
            work_image = image[y : y + h, x : x + w]

        try:
            prepared = self._pre.prepare(work_image, meta)
            orig_h, orig_w = work_image.shape[:2]

            if meta.framework == "opencv":
                raw_out = self._run_opencv(prepared, task)
            elif meta.framework == "onnx":
                session = self._mm.get_session(meta.model_name)
                if session is None:
                    logger.warning("ONNX session not loaded for '%s'; falling back to opencv", meta.model_name)
                    raw_out = self._run_opencv(prepared, task)
                else:
                    raw_out = self._run_onnx(prepared, session, meta)
            elif meta.framework == "torch":
                model_obj = self._mm.get_session(meta.model_name)
                if model_obj is None:
                    logger.warning("Torch model not loaded for '%s'; falling back to opencv", meta.model_name)
                    raw_out = self._run_opencv(prepared, task)
                else:
                    raw_out = self._run_torch(prepared, model_obj, meta)
            else:
                logger.warning("Unknown framework '%s'; passing image through", meta.framework)
                raw_out = prepared

            enhanced = self._post.postprocess(raw_out, (orig_h, orig_w), task, meta)

            # Paste back if ROI mode
            if roi_rect is not None:
                enhanced = self._post.merge_roi_back(base_frame, enhanced, roi_rect)

            result.enhanced_image = enhanced
            result.output_shape = tuple(enhanced.shape)
            result.success = True

        except Exception as exc:
            logger.error("enhance() failed for task '%s': %s", task, exc, exc_info=True)
            result.enhanced_image = image.copy()
            result.output_shape = tuple(image.shape)

        result.processing_time_ms = (time.perf_counter() - t0) * 1000.0
        return result

    def enhance_frame(self, frame: np.ndarray) -> EnhancementResult:
        """
        Apply all active enhancement types in pipeline order.

        Each task's output is chained as the input to the next task.
        Returns the final EnhancementResult after the complete chain.
        """
        active_types = self._cfg.active_types()
        if not active_types:
            return EnhancementResult(
                enhanced_image=frame.copy(),
                input_shape=tuple(frame.shape),
                output_shape=tuple(frame.shape),
                success=True,
            )

        current = frame
        last_result = EnhancementResult()
        total_ms = 0.0

        for task in active_types:
            last_result = self.enhance(current, task)
            total_ms += last_result.processing_time_ms
            if last_result.success and last_result.enhanced_image is not None:
                current = last_result.enhanced_image

        last_result.processing_time_ms = total_ms
        last_result.enhanced_image = current
        last_result.output_shape = tuple(current.shape)
        return last_result

    def enhance_roi_crop(
        self,
        crop: np.ndarray,
        roi_id: str,
        task: EnhancementType,
    ) -> EnhancementResult:
        """
        Enhance a pre-extracted ROI crop directly.

        Args:
            crop:   BGR uint8 ROI crop image.
            roi_id: Identifier for the ROI (stored in result).
            task:   Enhancement task to apply.

        Returns:
            EnhancementResult with roi_id populated.
        """
        result = self.enhance(crop, task)
        result.roi_id = roi_id
        return result

    # ------------------------------------------------------------------
    # Backend runners
    # ------------------------------------------------------------------

    def _run_opencv(self, image: np.ndarray, task: EnhancementType) -> np.ndarray:
        """
        Run OpenCV-based enhancement.

        SHARPENING : unsharp mask (alpha=1.5, beta=-0.5, 3x3 Gaussian)
        DENOISING  : fastNlMeansDenoisingColored (h=7, hColor=7)
        DEBLURRING : simple subtract-blur approximation
        Others     : pass through unchanged
        """
        if task == EnhancementType.SHARPENING:
            blurred = cv2.GaussianBlur(image, (3, 3), sigmaX=0)
            sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
            return sharpened

        if task == EnhancementType.DENOISING:
            denoised = cv2.fastNlMeansDenoisingColored(
                image,
                None,
                h=7,
                hColor=7,
                templateWindowSize=7,
                searchWindowSize=21,
            )
            return denoised

        if task == EnhancementType.DEBLURRING:
            # Simple Wiener-like approximation:
            # enhanced = original - 0.3 * blurred, then clip
            blurred = cv2.GaussianBlur(image, (5, 5), sigmaX=1.0)
            img_f = image.astype(np.float32)
            blur_f = blurred.astype(np.float32)
            deblurred_f = img_f - 0.3 * blur_f
            deblurred = np.clip(deblurred_f, 0, 255).astype(np.uint8)
            return deblurred

        # SUPER_RESOLUTION or COMBINED with no AI model: bicubic upscale + downscale
        if task == EnhancementType.SUPER_RESOLUTION:
            h, w = image.shape[:2]
            up = cv2.resize(
                image,
                (w * self._cfg.scale_factor, h * self._cfg.scale_factor),
                interpolation=cv2.INTER_CUBIC,
            )
            return up

        return image.copy()

    def _run_onnx(
        self,
        image: np.ndarray,
        session: object,
        meta,
    ) -> np.ndarray:
        """Run inference through an ONNX InferenceSession."""
        try:
            import onnxruntime as ort  # type: ignore

            # Build input tensor: (1, C, H, W) float32
            tensor = image.astype(np.float32)
            if tensor.ndim == 3:
                tensor = np.transpose(tensor, (2, 0, 1))   # HWC → CHW
            tensor = np.expand_dims(tensor, axis=0)         # → (1, C, H, W)

            input_name = session.get_inputs()[0].name
            outputs = session.run(None, {input_name: tensor})
            out = outputs[0]   # (1, C, H, W) or (1, H, W, C)
            return out

        except Exception as exc:
            logger.error("ONNX inference failed: %s; falling back to opencv", exc)
            return self._run_opencv(image, meta.task_type)

    def _run_torch(
        self,
        image: np.ndarray,
        model: object,
        meta,
    ) -> np.ndarray:
        """Run inference through a PyTorch nn.Module."""
        try:
            import torch  # type: ignore

            device = next(model.parameters()).device  # type: ignore

            # Build input tensor: (1, C, H, W) float32
            tensor = image.astype(np.float32)
            if tensor.ndim == 3:
                tensor = np.transpose(tensor, (2, 0, 1))
            tensor_t = torch.from_numpy(tensor).unsqueeze(0).to(device)

            with torch.no_grad():
                output_t = model(tensor_t)  # type: ignore

            out = output_t.squeeze(0).cpu().numpy()   # (C, H, W)
            return out

        except Exception as exc:
            logger.error("Torch inference failed: %s; falling back to opencv", exc)
            return self._run_opencv(image, meta.task_type)
