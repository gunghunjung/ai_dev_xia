"""
processing/roi_manager.py — Manages one or more Regions of Interest.

Responsibilities:
  - store ROI definitions (id, bbox)
  - extract crop images from processed frame
  - support add / update / remove operations
  - extensible to dynamic/tracked ROIs
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from config import ROIConfig
from core.data_models import ROIData
from utils.helpers import crop_roi


class ROIManager:
    """
    Holds a registry of named ROIs and produces ROIData objects each frame.

    Usage::
        mgr = ROIManager.from_config(cfg.rois)
        roi_data_list = mgr.extract(processed_frame)
    """

    def __init__(self):
        # roi_id → (bbox, label, enabled)
        self._rois: Dict[str, dict] = {}

    # ── Registration API ──────────────────────────────────────────────────

    def add_roi(
        self,
        roi_id: str,
        bbox: Tuple[int, int, int, int],
        label: str = "",
        enabled: bool = True,
    ) -> None:
        self._rois[roi_id] = {"bbox": bbox, "label": label, "enabled": enabled}

    def update_roi(
        self,
        roi_id: str,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        enabled: Optional[bool] = None,
    ) -> None:
        if roi_id not in self._rois:
            raise KeyError(f"ROI '{roi_id}' not registered")
        if bbox is not None:
            self._rois[roi_id]["bbox"] = bbox
        if enabled is not None:
            self._rois[roi_id]["enabled"] = enabled

    def remove_roi(self, roi_id: str) -> None:
        self._rois.pop(roi_id, None)

    def list_roi_ids(self) -> List[str]:
        return list(self._rois.keys())

    def get_bbox(self, roi_id: str) -> Optional[Tuple[int, int, int, int]]:
        entry = self._rois.get(roi_id)
        return entry["bbox"] if entry else None

    # ── Frame extraction ──────────────────────────────────────────────────

    def extract(
        self,
        frame: np.ndarray,
        prev_crops: Optional[Dict[str, np.ndarray]] = None,
    ) -> List[ROIData]:
        """
        For each enabled ROI, crop the region from *frame* and return
        a list of ROIData objects (crops attached, features empty).
        """
        results: List[ROIData] = []
        for roi_id, info in self._rois.items():
            if not info["enabled"]:
                continue
            bbox = info["bbox"]
            crop = crop_roi(frame, bbox)
            prev = prev_crops.get(roi_id) if prev_crops else None
            roi_data = ROIData(
                roi_id=roi_id,
                bbox=bbox,
                cropped=crop,
                cropped_prev=prev,
            )
            results.append(roi_data)
        return results

    def get_prev_crops(self, roi_data_list: List[ROIData]) -> Dict[str, np.ndarray]:
        """Build a prev_crops dict from the *current* frame's ROIData for next frame."""
        return {r.roi_id: r.cropped for r in roi_data_list if r.cropped is not None}

    # ── Factory ───────────────────────────────────────────────────────────

    @staticmethod
    def from_config(roi_configs: List[ROIConfig]) -> "ROIManager":
        mgr = ROIManager()
        for rc in roi_configs:
            mgr.add_roi(rc.roi_id, rc.bbox, rc.label, rc.enabled)
        return mgr

    # ── Interactive ROI selection (OpenCV window helper) ──────────────────

    @staticmethod
    def select_roi_interactive(
        frame: np.ndarray,
        roi_id: str = "roi_0",
        window_name: str = "Select ROI — press SPACE/ENTER to confirm, C to cancel",
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Open an OpenCV window to let the user draw an ROI rectangle.
        Returns (x, y, w, h) or None if cancelled.
        """
        bbox = cv2.selectROI(window_name, frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow(window_name)
        x, y, w, h = bbox
        if w > 0 and h > 0:
            return (x, y, w, h)
        return None
