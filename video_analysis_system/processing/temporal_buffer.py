"""
processing/temporal_buffer.py — Sliding-window frame history and temporal analysis.

TemporalBuffer stores a deque of recent FrameContext objects and exposes
helpers to extract per-ROI temporal features (for trend, stuck, drift, etc.).
"""

from __future__ import annotations

import collections
from typing import Dict, List, Optional

import numpy as np

from config import TemporalConfig
from core.data_models import FrameContext, TemporalSummary
from utils.helpers import count_zero_crossings, linear_regression_slope


class TemporalBuffer:
    """
    Maintains a fixed-length deque of FrameContext objects.

    Usage::
        buf = TemporalBuffer(window_size=60)
        buf.push(ctx)
        summaries = buf.compute_summaries(cfg.decision)
    """

    def __init__(self, window_size: int = 60):
        self._window_size = window_size
        self._buffer: collections.deque[FrameContext] = collections.deque(
            maxlen=window_size
        )

    # ── Basic operations ──────────────────────────────────────────────────

    def push(self, ctx: FrameContext) -> None:
        self._buffer.append(ctx)

    def get_recent(self, n: Optional[int] = None) -> List[FrameContext]:
        """Return the last *n* contexts (or all if n is None)."""
        frames = list(self._buffer)
        if n is not None:
            frames = frames[-n:]
        return frames

    def __len__(self) -> int:
        return len(self._buffer)

    @property
    def is_warm(self) -> bool:
        return len(self._buffer) >= self._window_size

    # ── Per-ROI feature series ────────────────────────────────────────────

    def get_feature_series(self, roi_id: str, feature_name: str) -> List[float]:
        """Return a time-ordered list of one feature value across the buffer."""
        series = []
        for ctx in self._buffer:
            roi = ctx.get_roi(roi_id)
            if roi is not None and feature_name in roi.features:
                series.append(roi.features[feature_name])
        return series

    def get_all_roi_ids(self) -> List[str]:
        """Collect all ROI IDs seen in the buffer."""
        ids = set()
        for ctx in self._buffer:
            for roi in ctx.rois:
                ids.add(roi.roi_id)
        return list(ids)

    # ── Temporal summary computation ──────────────────────────────────────

    def compute_summary(
        self,
        roi_id: str,
        stuck_variance_thresh: float = 1.5,
        stuck_min_frames: int = 25,
        drift_threshold: float = 25.0,
        oscillation_threshold: float = 15.0,
        oscillation_min_cycles: int = 3,
        sudden_change_threshold: float = 40.0,
    ) -> TemporalSummary:
        """Compute TemporalSummary for a single ROI over the current buffer."""
        summary = TemporalSummary(roi_id=roi_id, window_size=len(self._buffer))

        mean_series = self.get_feature_series(roi_id, "mean_intensity")
        std_series  = self.get_feature_series(roi_id, "std_intensity")
        diff_series = self.get_feature_series(roi_id, "mean_abs_diff")

        summary.mean_intensity_history = mean_series
        summary.variance_history = std_series
        summary.delta_history = diff_series

        if not mean_series:
            return summary

        arr = np.array(mean_series, dtype=np.float64)
        summary.recent_mean = float(arr.mean())
        summary.recent_std  = float(arr.std())
        summary.trend_slope = linear_regression_slope(mean_series)

        if diff_series:
            summary.max_sudden_change = float(max(diff_series))

        # ── Stuck detection ──────────────────────────────────────────────
        if (
            len(mean_series) >= stuck_min_frames
            and summary.recent_std < stuck_variance_thresh
        ):
            summary.is_stuck = True

        # ── Drift detection ──────────────────────────────────────────────
        if len(mean_series) >= 2:
            first_mean = float(np.mean(arr[: max(1, len(arr) // 4)]))
            last_mean  = float(np.mean(arr[-max(1, len(arr) // 4):]))
            if abs(last_mean - first_mean) > drift_threshold:
                summary.is_drifting = True

        # ── Oscillation detection ─────────────────────────────────────────
        if len(mean_series) >= oscillation_min_cycles * 2:
            crossings = count_zero_crossings(mean_series)
            if (
                crossings >= oscillation_min_cycles
                and summary.recent_std > oscillation_threshold
            ):
                summary.oscillation_amplitude = summary.recent_std
                summary.is_oscillating = True

        # ── Sudden change detection ───────────────────────────────────────
        if diff_series and max(diff_series) > sudden_change_threshold:
            summary.has_sudden_change = True

        return summary

    def compute_all_summaries(self, **kwargs) -> Dict[str, TemporalSummary]:
        """Compute summaries for every ROI seen in the buffer."""
        return {
            roi_id: self.compute_summary(roi_id, **kwargs)
            for roi_id in self.get_all_roi_ids()
        }

    # ── Consecutive state counting ────────────────────────────────────────

    def count_consecutive_system_state(self, target_state: str, from_end: bool = True) -> int:
        """Count how many recent frames share the same system state string."""
        frames = list(self._buffer)
        if from_end:
            frames = list(reversed(frames))
        count = 0
        for ctx in frames:
            if ctx.system_state.value == target_state:
                count += 1
            else:
                break
        return count
