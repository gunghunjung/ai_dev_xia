"""
processing/tracker.py — ROI tracking abstractions and implementations.

Tracking is optional. When enabled it updates ROI bounding boxes between frames
so that moving targets stay centred in the analysis window.

Implementations:
  - NullTracker          : no-op (fixed ROI, default)
  - CentroidTracker      : simple centroid / template matching approach
  - OpenCVTrackerAdapter : wraps OpenCV built-in trackers (KCF, CSRT, etc.)
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from core.interfaces import ITracker


# ---------------------------------------------------------------------------
# NullTracker — use when ROI is fixed
# ---------------------------------------------------------------------------

class NullTracker(ITracker):
    """Placeholder tracker that never updates the bounding box."""

    def __init__(self, bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)):
        self._bbox = bbox

    def init(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> None:
        self._bbox = bbox

    def update(self, frame: np.ndarray) -> Tuple[bool, Tuple[int, int, int, int]]:
        return True, self._bbox

    def reset(self) -> None:
        pass


# ---------------------------------------------------------------------------
# CentroidTracker — template matching-based
# ---------------------------------------------------------------------------

class CentroidTracker(ITracker):
    """
    Lightweight tracker using cv2.matchTemplate.

    Strategy:
      1. Store a template from the initial bbox.
      2. Each frame: search a small neighbourhood around last known position.
      3. If match score exceeds threshold, update centroid.
      4. If score drops below threshold, report tracking lost.
    """

    def __init__(
        self,
        search_expand: int = 30,
        match_threshold: float = 0.6,
        update_template_every: int = 10,
    ):
        self._template: Optional[np.ndarray] = None
        self._bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
        self._search_expand = search_expand
        self._match_threshold = match_threshold
        self._update_every = update_template_every
        self._frame_count = 0

    def init(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> None:
        self._bbox = bbox
        self._template = self._crop_template(frame, bbox)
        self._frame_count = 0

    def update(self, frame: np.ndarray) -> Tuple[bool, Tuple[int, int, int, int]]:
        if self._template is None:
            return False, self._bbox

        x, y, w, h = self._bbox
        fh, fw = frame.shape[:2]
        exp = self._search_expand

        # Search region (clamped to frame bounds)
        sx = max(0, x - exp)
        sy = max(0, y - exp)
        ex = min(fw, x + w + exp)
        ey = min(fh, y + h + exp)
        search_region = frame[sy:ey, sx:ex]

        gray_region = self._to_gray(search_region)
        gray_tmpl = self._to_gray(self._template)

        if gray_region.shape[0] < gray_tmpl.shape[0] or gray_region.shape[1] < gray_tmpl.shape[1]:
            return False, self._bbox

        result = cv2.matchTemplate(gray_region, gray_tmpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val < self._match_threshold:
            return False, self._bbox   # tracking lost

        # Convert back to frame coords
        new_x = sx + max_loc[0]
        new_y = sy + max_loc[1]
        self._bbox = (new_x, new_y, w, h)

        # Periodically refresh the template
        self._frame_count += 1
        if self._frame_count % self._update_every == 0:
            self._template = self._crop_template(frame, self._bbox)

        return True, self._bbox

    def reset(self) -> None:
        self._template = None
        self._frame_count = 0

    @staticmethod
    def _to_gray(img: np.ndarray) -> np.ndarray:
        if img.ndim == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    @staticmethod
    def _crop_template(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        x, y, w, h = bbox
        fh, fw = frame.shape[:2]
        x1, y1 = max(x, 0), max(y, 0)
        x2, y2 = min(x + w, fw), min(y + h, fh)
        return frame[y1:y2, x1:x2].copy()


# ---------------------------------------------------------------------------
# OpenCVTrackerAdapter — wraps cv2 built-in trackers
# ---------------------------------------------------------------------------

class OpenCVTrackerAdapter(ITracker):
    """
    Wraps any OpenCV tracker that follows the Tracker API
    (KCF, CSRT, MIL, …).

    Pass the tracker name as a string, e.g. ``tracker_type="CSRT"``.
    """

    _FACTORY = {
        "KCF":  lambda: cv2.TrackerKCF_create(),
        "CSRT": lambda: cv2.TrackerCSRT_create(),
        "MIL":  lambda: cv2.TrackerMIL_create(),
    }

    def __init__(self, tracker_type: str = "CSRT"):
        if tracker_type not in self._FACTORY:
            raise ValueError(f"Unknown tracker type: {tracker_type}. Choose from {list(self._FACTORY)}")
        self._tracker_type = tracker_type
        self._tracker = self._FACTORY[tracker_type]()
        self._bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)

    def init(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> None:
        self._bbox = bbox
        self._tracker = self._FACTORY[self._tracker_type]()
        self._tracker.init(frame, bbox)

    def update(self, frame: np.ndarray) -> Tuple[bool, Tuple[int, int, int, int]]:
        ok, bbox = self._tracker.update(frame)
        if ok:
            self._bbox = tuple(int(v) for v in bbox)  # type: ignore[assignment]
        return bool(ok), self._bbox

    def reset(self) -> None:
        self._tracker = self._FACTORY[self._tracker_type]()


# ---------------------------------------------------------------------------
# Per-ROI tracker registry
# ---------------------------------------------------------------------------

class TrackerRegistry:
    """
    Manages one ITracker instance per roi_id.
    Used by EngineCore to optionally update ROI bboxes each frame.
    """

    def __init__(self, tracker_type: str = "none"):
        self._tracker_type = tracker_type
        self._trackers: dict[str, ITracker] = {}

    def init_roi(
        self, roi_id: str, frame: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> None:
        tracker = self._build_tracker()
        tracker.init(frame, bbox)
        self._trackers[roi_id] = tracker

    def update_roi(
        self, roi_id: str, frame: np.ndarray
    ) -> Tuple[bool, Optional[Tuple[int, int, int, int]]]:
        tracker = self._trackers.get(roi_id)
        if tracker is None:
            return False, None
        return tracker.update(frame)

    def reset_all(self) -> None:
        for t in self._trackers.values():
            t.reset()
        self._trackers.clear()

    def _build_tracker(self) -> ITracker:
        if self._tracker_type == "none":
            return NullTracker()
        elif self._tracker_type == "centroid":
            return CentroidTracker()
        elif self._tracker_type in OpenCVTrackerAdapter._FACTORY:
            return OpenCVTrackerAdapter(self._tracker_type)
        else:
            return NullTracker()
