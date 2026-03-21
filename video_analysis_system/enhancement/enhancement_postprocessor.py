"""
enhancement/enhancement_postprocessor.py — Output normalisation and ROI paste-back.

Converts raw model output (float arrays or already-uint8) back to a
display-ready BGR uint8 frame, and optionally pastes an enhanced ROI
crop back into the original full frame.
"""

from __future__ import annotations

import logging
from typing import Tuple

import cv2
import numpy as np

from enhancement.enhancement_config import EnhancementConfig, EnhancementType
from enhancement.enhancement_model_manager import EnhancementModelMeta

logger = logging.getLogger(__name__)


class EnhancementPostprocessor:
    """
    Converts model output arrays to final BGR uint8 images.

    Handles:
    - float [0, 1] → uint8 clipping and conversion
    - spatial resize back to the expected display size
    - RGB → BGR channel swap for AI model outputs
    - ROI crop paste-back into a full frame
    """

    def __init__(self, cfg: EnhancementConfig) -> None:
        self._cfg = cfg

    # ------------------------------------------------------------------
    # Primary postprocessing
    # ------------------------------------------------------------------

    def postprocess(
        self,
        output: np.ndarray,
        original_shape: Tuple[int, int],
        task: EnhancementType,
        meta: EnhancementModelMeta,
    ) -> np.ndarray:
        """
        Convert raw model output to a final BGR uint8 image.

        Args:
            output:         Raw output array from the model.
            original_shape: (orig_h, orig_w) of the input image before preprocessing.
            task:           The enhancement task that was applied.
            meta:           Model metadata used for framework detection.

        Returns:
            BGR uint8 ndarray with shape (orig_h, orig_w, 3).
        """
        img = output.copy()

        # --- Float → uint8 ---
        if img.dtype in (np.float32, np.float64):
            img = np.clip(img, 0.0, 1.0)
            img = (img * 255).astype(np.uint8)
        elif img.dtype != np.uint8:
            img = img.astype(np.uint8)

        # --- Squeeze batch or channel dims if needed ---
        # (1, H, W, C) or (1, C, H, W) → (H, W, C)
        if img.ndim == 4 and img.shape[0] == 1:
            img = img[0]
        # (C, H, W) channel-first → (H, W, C)
        if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[2] not in (1, 3):
            img = np.transpose(img, (1, 2, 0))

        # --- Grayscale → BGR ---
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.ndim == 3 and img.shape[2] == 1:
            img = cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR)

        # --- Resize back to expected output size ---
        orig_h, orig_w = original_shape
        target_h = orig_h * meta.scale_factor
        target_w = orig_w * meta.scale_factor

        current_h, current_w = img.shape[:2]
        if current_h != target_h or current_w != target_w:
            img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        # --- RGB → BGR for AI model outputs ---
        if meta.framework in ("onnx", "torch"):
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        return img

    # ------------------------------------------------------------------
    # ROI paste-back
    # ------------------------------------------------------------------

    def merge_roi_back(
        self,
        base_frame: np.ndarray,
        enhanced_crop: np.ndarray,
        roi_rect: Tuple[int, int, int, int],
    ) -> np.ndarray:
        """
        Paste an enhanced ROI crop back into a copy of the base frame.

        Args:
            base_frame:     Full-frame BGR uint8 image (unmodified original).
            enhanced_crop:  Enhanced BGR uint8 crop to paste in.
            roi_rect:       (x, y, w, h) position in base_frame coordinates.

        Returns:
            A new BGR uint8 image with the enhanced crop blended in.
        """
        result = base_frame.copy()
        x, y, w, h = roi_rect

        # Clamp roi_rect to frame boundaries
        frame_h, frame_w = base_frame.shape[:2]
        x = max(0, min(x, frame_w - 1))
        y = max(0, min(y, frame_h - 1))
        w = max(1, min(w, frame_w - x))
        h = max(1, min(h, frame_h - y))

        # Resize enhanced crop to exactly match roi_rect (w, h)
        crop_h, crop_w = enhanced_crop.shape[:2]
        if crop_h != h or crop_w != w:
            enhanced_crop = cv2.resize(enhanced_crop, (w, h), interpolation=cv2.INTER_LINEAR)

        # Paste
        result[y : y + h, x : x + w] = enhanced_crop
        return result
