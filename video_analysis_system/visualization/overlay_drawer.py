"""
visualization/overlay_drawer.py — All OpenCV drawing operations.

Kept stateless: every method takes a frame + data and returns a modified copy.
This makes it easy to unit-test and swap out the rendering backend later.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from core.data_models import FrameContext, ROIData
from core.states import ROI_STATE_COLORS, STATE_COLORS, ROIState, SystemState


# Overlay palette
_WHITE  = (255, 255, 255)
_BLACK  = (0, 0, 0)
_GRAY   = (160, 160, 160)
_YELLOW = (0, 220, 220)

_FONT      = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SM   = 0.45
_FONT_MD   = 0.55
_FONT_LG   = 0.85
_THICK_TH  = 1
_THICK_BOX = 2


class OverlayDrawer:
    """
    Stateless helper that draws all visual overlays onto a BGR frame.
    Call methods in any order; each returns an annotated copy.
    """

    # ── ROI boxes ─────────────────────────────────────────────────────────

    @staticmethod
    def draw_roi_box(
        frame: np.ndarray,
        roi: ROIData,
        show_label: bool = True,
        show_confidence: bool = True,
    ) -> np.ndarray:
        color = ROI_STATE_COLORS.get(roi.roi_state, _GRAY)
        x, y, w, h = roi.bbox

        # Rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, _THICK_BOX)

        # ROI id + state label
        if show_label:
            label = f"{roi.roi_id} [{roi.roi_state.value}]"
            label_y = max(y - 6, 12)
            cv2.putText(frame, label, (x, label_y), _FONT, _FONT_SM, _BLACK, _THICK_TH + 1)
            cv2.putText(frame, label, (x, label_y), _FONT, _FONT_SM, color, _THICK_TH)

        # Confidence
        if show_confidence and roi.detection is not None:
            conf_text = f"{roi.detection.label} {roi.detection.confidence:.2f}"
            cv2.putText(
                frame, conf_text, (x, y + h + 14), _FONT, _FONT_SM, _BLACK, _THICK_TH + 1
            )
            cv2.putText(
                frame, conf_text, (x, y + h + 14), _FONT, _FONT_SM, color, _THICK_TH
            )

        return frame

    @staticmethod
    def draw_all_rois(
        frame: np.ndarray,
        rois: List[ROIData],
        show_label: bool = True,
        show_confidence: bool = True,
    ) -> np.ndarray:
        for roi in rois:
            frame = OverlayDrawer.draw_roi_box(frame, roi, show_label, show_confidence)
        return frame

    # ── State banner ──────────────────────────────────────────────────────

    @staticmethod
    def draw_state_banner(
        frame: np.ndarray,
        state: SystemState,
        confidence: float = 0.0,
    ) -> np.ndarray:
        color = STATE_COLORS.get(state, _GRAY)
        h, w = frame.shape[:2]

        # Semi-transparent filled bar at top
        banner_h = 36
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), color, -1)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

        text = f"STATE: {state.value}  (conf {confidence:.2f})"
        cv2.putText(frame, text, (8, 24), _FONT, _FONT_LG, _BLACK, 3)
        cv2.putText(frame, text, (8, 24), _FONT, _FONT_LG, _WHITE, 1)
        return frame

    # ── Frame info strip ──────────────────────────────────────────────────

    @staticmethod
    def draw_frame_info(
        frame: np.ndarray,
        frame_index: int,
        timestamp: float,
        inference_ms: float = 0.0,
    ) -> np.ndarray:
        h, w = frame.shape[:2]
        info = f"Frame {frame_index}  |  {timestamp:.3f}s  |  inf {inference_ms:.1f}ms"
        y_pos = h - 8
        cv2.putText(frame, info, (6, y_pos), _FONT, _FONT_SM, _BLACK, 2)
        cv2.putText(frame, info, (6, y_pos), _FONT, _FONT_SM, _GRAY,  1)
        return frame

    # ── Event marker ─────────────────────────────────────────────────────

    @staticmethod
    def draw_event_marker(frame: np.ndarray, event_type: str = "") -> np.ndarray:
        h, w = frame.shape[:2]
        # Red border flash
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 4)
        cv2.putText(frame, f"EVENT: {event_type}", (10, h // 2), _FONT, _FONT_LG, (0, 0, 0), 4)
        cv2.putText(frame, f"EVENT: {event_type}", (10, h // 2), _FONT, _FONT_LG, (0, 0, 255), 2)
        return frame

    # ── Rule triggers ─────────────────────────────────────────────────────

    @staticmethod
    def draw_triggered_rules(frame: np.ndarray, ctx: FrameContext) -> np.ndarray:
        h = frame.shape[0]
        y_base = h - 30
        for i, rule in enumerate(ctx.triggered_rules[:5]):  # max 5 shown
            y = y_base - i * 16
            if y < 40:
                break
            text = f"! {rule.rule_name}: {rule.message[:50]}"
            cv2.putText(frame, text, (6, y), _FONT, 0.38, _BLACK, 2)
            cv2.putText(frame, text, (6, y), _FONT, 0.38, _YELLOW, 1)
        return frame

    # ── Crosshair at ROI centre ───────────────────────────────────────────

    @staticmethod
    def draw_roi_centre(frame: np.ndarray, roi: ROIData, size: int = 8) -> np.ndarray:
        cx, cy = roi.center
        color = ROI_STATE_COLORS.get(roi.roi_state, _GRAY)
        cv2.line(frame, (cx - size, cy), (cx + size, cy), color, 1)
        cv2.line(frame, (cx, cy - size), (cx, cy + size), color, 1)
        return frame

    # ── PIP (minimap) overlay ─────────────────────────────────────────────

    @staticmethod
    def draw_pip_overlay(
        frame: np.ndarray,
        viewport_norm: Tuple[float, float, float, float],
        pip_w: int = 180,
        pip_h: int = 100,
        margin: int = 10,
    ) -> np.ndarray:
        """
        Draw a Picture-in-Picture minimap in the top-right corner of *frame*.

        This is used for the headless (OpenCV window) rendering path.
        The Tkinter GUI path renders the PIP directly on the canvas instead.

        Args:
            frame:          BGR frame to annotate (modified in-place copy returned).
            viewport_norm:  (nx, ny, nw, nh) — current viewport as normalised
                            fractions of the full frame (from ViewportManager.get_pip_data()).
            pip_w / pip_h:  PIP thumbnail dimensions in pixels.
            margin:         Distance from frame edge in pixels.
        """
        fh, fw = frame.shape[:2]
        # PIP bounding box (top-right corner)
        px = fw - pip_w - margin
        py = margin

        # Dark background
        overlay = frame.copy()
        cv2.rectangle(overlay, (px - 1, py - 1), (px + pip_w + 1, py + pip_h + 1),
                      (13, 13, 26), -1)
        cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)

        # Thumbnail: scale full frame down to pip_w × pip_h
        thumb = cv2.resize(frame, (pip_w, pip_h), interpolation=cv2.INTER_AREA)
        frame[py: py + pip_h, px: px + pip_w] = thumb

        # Border
        cv2.rectangle(frame, (px - 1, py - 1), (px + pip_w + 1, py + pip_h + 1),
                      (88, 91, 112), 1)

        # Viewport indicator rectangle (yellow)
        nx, ny, nw, nh = viewport_norm
        vx1 = px + int(nx * pip_w)
        vy1 = py + int(ny * pip_h)
        vx2 = px + int((nx + nw) * pip_w)
        vy2 = py + int((ny + nh) * pip_h)
        vx1, vx2 = max(px, vx1), min(px + pip_w, vx2)
        vy1, vy2 = max(py, vy1), min(py + pip_h, vy2)
        cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), (0, 220, 220), 2)

        # Label
        cv2.putText(frame, "MINIMAP", (px + 3, py + pip_h - 3),
                    _FONT, 0.30, (88, 91, 112), 1)

        return frame
