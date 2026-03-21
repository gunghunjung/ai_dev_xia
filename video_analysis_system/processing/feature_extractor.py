"""
processing/feature_extractor.py — Extract numeric features from an image region.

Features are returned as a plain dict[str, float] so they are
JSON-serialisable and easy to log.
"""

from __future__ import annotations

from typing import Dict, Optional

import cv2
import numpy as np

from core.interfaces import IFeatureExtractor
from utils.helpers import compute_edge_density, compute_histogram


# ---------------------------------------------------------------------------
# StandardFeatureExtractor — practical default set
# ---------------------------------------------------------------------------

class StandardFeatureExtractor(IFeatureExtractor):
    """
    Extracts the following features per ROI crop:

    Intensity
      mean_intensity, min_intensity, max_intensity, std_intensity

    Temporal delta (requires prev_region)
      mean_abs_diff, max_abs_diff

    Structure
      edge_density
      contour_count, largest_contour_area

    Histogram (16 bins → hist_0 … hist_15)
    """

    def extract(
        self,
        region: np.ndarray,
        prev_region: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        if region is None or region.size == 0:
            return {}

        features: Dict[str, float] = {}

        # ── Gray conversion ──────────────────────────────────────────────
        if region.ndim == 3:
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            gray = region.astype(np.float32)

        # ── Intensity statistics ─────────────────────────────────────────
        features["mean_intensity"] = float(gray.mean())
        features["min_intensity"]  = float(gray.min())
        features["max_intensity"]  = float(gray.max())
        features["std_intensity"]  = float(gray.std())
        features["intensity_range"] = features["max_intensity"] - features["min_intensity"]

        # ── Temporal delta ───────────────────────────────────────────────
        if prev_region is not None and prev_region.size > 0:
            if prev_region.ndim == 3:
                prev_gray = cv2.cvtColor(prev_region, cv2.COLOR_BGR2GRAY).astype(np.float32)
            else:
                prev_gray = prev_region.astype(np.float32)
            # Resize prev to match current if sizes differ
            if prev_gray.shape != gray.shape:
                prev_gray = cv2.resize(prev_gray, (gray.shape[1], gray.shape[0]))
            diff = np.abs(gray - prev_gray)
            features["mean_abs_diff"] = float(diff.mean())
            features["max_abs_diff"]  = float(diff.max())
            features["std_abs_diff"]  = float(diff.std())
        else:
            features["mean_abs_diff"] = 0.0
            features["max_abs_diff"]  = 0.0
            features["std_abs_diff"]  = 0.0

        # ── Edge density ─────────────────────────────────────────────────
        features["edge_density"] = compute_edge_density(region)

        # ── Contour features ─────────────────────────────────────────────
        _, binarised = cv2.threshold(
            gray.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        contours, _ = cv2.findContours(
            binarised, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        features["contour_count"] = float(len(contours))
        features["largest_contour_area"] = float(
            max((cv2.contourArea(c) for c in contours), default=0.0)
        )

        # ── Histogram (16 bins) ──────────────────────────────────────────
        hist = compute_histogram(region, bins=16)
        for i, val in enumerate(hist):
            features[f"hist_{i:02d}"] = float(val)

        return features


# ---------------------------------------------------------------------------
# MinimalFeatureExtractor — lightweight fallback / fast path
# ---------------------------------------------------------------------------

class MinimalFeatureExtractor(IFeatureExtractor):
    """Only extracts intensity stats and temporal diff. Fastest option."""

    def extract(
        self,
        region: np.ndarray,
        prev_region: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        if region is None or region.size == 0:
            return {}
        if region.ndim == 3:
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            gray = region.astype(np.float32)

        feats = {
            "mean_intensity": float(gray.mean()),
            "std_intensity":  float(gray.std()),
            "mean_abs_diff":  0.0,
        }

        if prev_region is not None and prev_region.size > 0:
            if prev_region.ndim == 3:
                prev_gray = cv2.cvtColor(prev_region, cv2.COLOR_BGR2GRAY).astype(np.float32)
            else:
                prev_gray = prev_region.astype(np.float32)
            if prev_gray.shape != gray.shape:
                prev_gray = cv2.resize(prev_gray, (gray.shape[1], gray.shape[0]))
            feats["mean_abs_diff"] = float(np.abs(gray - prev_gray).mean())

        return feats


# ---------------------------------------------------------------------------
# FrameLevelFeatureExtractor — for whole-frame features
# ---------------------------------------------------------------------------

class FrameLevelFeatureExtractor:
    """Extracts frame-wide features stored in FrameContext.frame_features."""

    def extract(self, frame: np.ndarray) -> Dict[str, float]:
        if frame is None or frame.size == 0:
            return {}
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        else:
            gray = frame.astype(np.float32)
        return {
            "frame_mean":  float(gray.mean()),
            "frame_std":   float(gray.std()),
            "frame_min":   float(gray.min()),
            "frame_max":   float(gray.max()),
        }
