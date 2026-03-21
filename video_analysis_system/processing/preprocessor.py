"""
processing/preprocessor.py — Pluggable frame preprocessing pipeline.

Steps are registered in order and applied sequentially.
Each step implements IPreprocessStep.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from config import PreprocessConfig
from core.interfaces import IPreprocessStep


# ---------------------------------------------------------------------------
# Concrete preprocessing steps
# ---------------------------------------------------------------------------

class ResizeStep(IPreprocessStep):
    def __init__(self, size: Tuple[int, int]):
        self._size = size  # (width, height)

    def apply(self, frame: np.ndarray) -> np.ndarray:
        return cv2.resize(frame, self._size, interpolation=cv2.INTER_LINEAR)


class GrayscaleStep(IPreprocessStep):
    def apply(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # keep 3-ch for pipeline
        return frame


class NormalizeStep(IPreprocessStep):
    """Scale pixel values to [0, 1] as float32, then back to uint8 range."""
    def apply(self, frame: np.ndarray) -> np.ndarray:
        f = frame.astype(np.float32) / 255.0
        return (f * 255).astype(np.uint8)


class DenoiseStep(IPreprocessStep):
    def __init__(self, ksize: int = 5):
        self._ksize = ksize if ksize % 2 == 1 else ksize + 1

    def apply(self, frame: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(frame, (self._ksize, self._ksize), 0)


class EqualizeHistStep(IPreprocessStep):
    """Histogram equalisation applied per channel (for BGR input)."""
    def apply(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            return cv2.equalizeHist(frame)
        channels = cv2.split(frame)
        eq = [cv2.equalizeHist(c) for c in channels]
        return cv2.merge(eq)


class MaskStep(IPreprocessStep):
    """Apply a binary mask image to zero out unwanted regions."""
    def __init__(self, mask: np.ndarray):
        self._mask = mask

    def apply(self, frame: np.ndarray) -> np.ndarray:
        mask = self._mask
        if mask.shape[:2] != frame.shape[:2]:
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
        if frame.ndim == 3 and mask.ndim == 2:
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        return cv2.bitwise_and(frame, mask)


class CLAHEStep(IPreprocessStep):
    """CLAHE contrast enhancement — useful for thermal / IR frames."""
    def __init__(self, clip_limit: float = 2.0, tile_size: Tuple[int, int] = (8, 8)):
        self._clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)

    def apply(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 3:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = self._clahe.apply(l)
            return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        return self._clahe.apply(frame)


# ---------------------------------------------------------------------------
# Preprocessor — manages and runs the step chain
# ---------------------------------------------------------------------------

class Preprocessor:
    """
    Applies an ordered sequence of IPreprocessStep objects to each frame.
    Steps can be added programmatically or built from PreprocessConfig.
    """

    def __init__(self):
        self._steps: List[IPreprocessStep] = []

    def add_step(self, step: IPreprocessStep) -> "Preprocessor":
        self._steps.append(step)
        return self

    def clear_steps(self) -> None:
        self._steps = []

    def process(self, frame: np.ndarray) -> np.ndarray:
        result = frame.copy()
        for step in self._steps:
            result = step.apply(result)
        return result

    @staticmethod
    def from_config(cfg: PreprocessConfig, mask_image: Optional[np.ndarray] = None) -> "Preprocessor":
        """Build a Preprocessor from PreprocessConfig."""
        pp = Preprocessor()
        if cfg.resize:
            pp.add_step(ResizeStep(cfg.resize))
        if cfg.denoise:
            pp.add_step(DenoiseStep())
        if cfg.equalize_hist:
            pp.add_step(EqualizeHistStep())
        if cfg.grayscale:
            pp.add_step(GrayscaleStep())
        if cfg.normalize:
            pp.add_step(NormalizeStep())
        if cfg.apply_mask and mask_image is not None:
            pp.add_step(MaskStep(mask_image))
        return pp
