"""
enhancement/enhancement_preprocessor.py — Input preparation for enhancement models.

Handles resizing, channel conversion, padding, and normalization so that
each backend (OpenCV / ONNX / Torch) receives the format it expects.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import cv2
import numpy as np

from enhancement.enhancement_config import EnhancementConfig
from enhancement.enhancement_model_manager import EnhancementModelMeta

logger = logging.getLogger(__name__)


class EnhancementPreprocessor:
    """
    Prepares images for enhancement model inference.

    Rules:
    - OpenCV models  → keep BGR uint8, optionally resize to input_size
    - AI (onnx/torch) models → resize if input_size specified,
      pad to multiple of 32, convert to float32 [0, 1]
    """

    def __init__(self, cfg: EnhancementConfig) -> None:
        self._cfg = cfg

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prepare(
        self,
        image: np.ndarray,
        target_meta: EnhancementModelMeta,
    ) -> np.ndarray:
        """
        Prepare a single image for the given model.

        Returns:
            uint8 BGR for opencv framework.
            float32 RGB [0, 1] array for onnx/torch frameworks.
        """
        img = self._ensure_bgr_uint8(image)

        if target_meta.input_size is not None:
            w, h = target_meta.input_size
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

        img = self._normalize(img, target_meta.framework)
        return img

    def prepare_batch(
        self,
        images: List[np.ndarray],
        target_meta: EnhancementModelMeta,
    ) -> np.ndarray:
        """
        Prepare a list of images into a single batch array of shape (N, H, W, C).

        All images are prepared individually then stacked.
        """
        prepared = [self.prepare(img, target_meta) for img in images]

        # Ensure all images have the same spatial shape by padding to max dims
        max_h = max(p.shape[0] for p in prepared)
        max_w = max(p.shape[1] for p in prepared)

        padded = []
        for p in prepared:
            h_pad = max_h - p.shape[0]
            w_pad = max_w - p.shape[1]
            if h_pad > 0 or w_pad > 0:
                p = np.pad(
                    p,
                    ((0, h_pad), (0, w_pad), (0, 0)),
                    mode="reflect",
                )
            padded.append(p)

        batch = np.stack(padded, axis=0)   # (N, H, W, C)
        return batch

    # ------------------------------------------------------------------
    # Channel helpers
    # ------------------------------------------------------------------

    def _ensure_bgr_uint8(self, image: np.ndarray) -> np.ndarray:
        """Convert any incoming image to BGR uint8."""
        # Handle float images [0, 1]
        if image.dtype in (np.float32, np.float64):
            image = np.clip(image, 0.0, 1.0)
            image = (image * 255).astype(np.uint8)

        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        # Grayscale → BGR
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        return image

    def _to_rgb(self, image: np.ndarray) -> np.ndarray:
        """Convert a BGR uint8 image to RGB uint8."""
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _pad_to_multiple(
        self,
        image: np.ndarray,
        multiple: int = 32,
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Pad image height and width to the nearest multiple of `multiple`.

        Returns:
            padded image and (orig_h, orig_w) for later unpadding.
        """
        orig_h, orig_w = image.shape[:2]
        pad_h = (multiple - orig_h % multiple) % multiple
        pad_w = (multiple - orig_w % multiple) % multiple

        if pad_h == 0 and pad_w == 0:
            return image, (orig_h, orig_w)

        padded = np.pad(
            image,
            ((0, pad_h), (0, pad_w), (0, 0)) if image.ndim == 3 else ((0, pad_h), (0, pad_w)),
            mode="reflect",
        )
        return padded, (orig_h, orig_w)

    def _normalize(self, image: np.ndarray, framework: str) -> np.ndarray:
        """
        Normalize image according to the target framework.

        - opencv  → returns uint8 BGR unchanged
        - onnx/torch → converts to RGB float32 [0, 1], pads to multiple of 32
        """
        if framework == "opencv":
            return image

        # AI backends: convert BGR → RGB, float32 [0, 1], pad
        rgb = self._to_rgb(image)
        padded, _ = self._pad_to_multiple(rgb, multiple=32)
        normalized = padded.astype(np.float32) / 255.0
        return normalized
