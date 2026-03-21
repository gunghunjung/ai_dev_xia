"""
visualization/renderer.py — Composes all overlay layers into the final display frame.

FrameRenderer is the single entry point for visualization.
It delegates drawing to OverlayDrawer and DebugVisualizer.

Viewport support (headless / OpenCV mode)
──────────────────────────────────────────
When a ViewportManager is attached via set_viewport(), FrameRenderer will:
  1. Render all annotations at source resolution (correct coordinate space).
  2. Crop the annotated frame to the current viewport region.
  3. Resize the crop to the target display size.
  4. Draw the PIP minimap overlay on top.

The Tkinter GUI path does this cropping in VideoCanvas instead.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from config import VisualizationConfig
from core.data_models import FrameContext
from core.interfaces import IRenderer
from visualization.debug_visualizer import DebugVisualizer
from visualization.overlay_drawer import OverlayDrawer


class FrameRenderer(IRenderer):
    """
    Assembles a displayable BGR frame from a FrameContext.

    Rendering order (drawn bottom to top in z-order):
      1. Processed frame (base image)
      2. ROI boxes + labels + confidence
      3. State banner (top strip)
      4. Frame info (bottom strip)
      5. Event marker (red border, only when is_event)
      6. Triggered rules list
      7. [Optional] Debug panel appended as side column
      8. [Optional] Viewport crop + resize  (headless viewport mode)
      9. [Optional] PIP minimap overlay     (headless viewport mode)
    """

    def __init__(self, cfg: VisualizationConfig) -> None:
        self._cfg = cfg
        self._viewport = None    # Optional[ViewportManager]

    # ── Optional viewport attachment ─────────────────────────────────────

    def set_viewport(self, viewport) -> None:
        """
        Attach a ViewportManager for headless (OpenCV window) viewport support.

        In the Tkinter GUI path this is NOT needed — VideoCanvas handles it.

        Args:
            viewport: core.viewport_manager.ViewportManager instance, or None.
        """
        self._viewport = viewport

    # ── Main render path ─────────────────────────────────────────────────

    def render(self, ctx: FrameContext) -> np.ndarray:
        """Build and return the complete annotated display frame."""
        base = ctx.processed_frame if ctx.processed_frame is not None else ctx.raw_frame
        if base is None:
            return self._blank_frame()

        frame = base.copy()

        # Ensure BGR
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        cfg = self._cfg

        # ── Annotations at source resolution ─────────────────────────────

        if cfg.show_roi_boxes and ctx.rois:
            frame = OverlayDrawer.draw_all_rois(
                frame, ctx.rois,
                show_label=cfg.show_labels,
                show_confidence=cfg.show_confidence,
            )
            for roi in ctx.rois:
                OverlayDrawer.draw_roi_centre(frame, roi)

        if cfg.show_state_banner:
            frame = OverlayDrawer.draw_state_banner(
                frame, ctx.system_state, ctx.state_confidence
            )

        if cfg.show_frame_info:
            frame = OverlayDrawer.draw_frame_info(
                frame, ctx.frame_index, ctx.timestamp, ctx.inference_time_ms
            )

        if ctx.is_event:
            frame = OverlayDrawer.draw_event_marker(frame, ctx.event_type)

        if cfg.show_debug_overlay and ctx.triggered_rules:
            frame = OverlayDrawer.draw_triggered_rules(frame, ctx)

        if cfg.show_debug_overlay:
            panel = DebugVisualizer.build_panel(ctx, width=300)
            fh = frame.shape[0]
            if panel.shape[0] != fh:
                panel = cv2.resize(panel, (panel.shape[1], fh))
            frame = np.hstack([frame, panel])

        # ── Viewport crop (headless mode only) ───────────────────────────

        if self._viewport is not None:
            vp = self._viewport
            vp_norm = vp.get_pip_data()
            vx, vy, vw, vh = vp.viewport_rect_int
            if vw > 0 and vh > 0:
                crop = frame[vy: vy + vh, vx: vx + vw]
                dw, dh = vp.canvas_size
                frame = cv2.resize(crop, (dw, dh), interpolation=cv2.INTER_LINEAR)
            # Draw PIP on cropped frame
            frame = OverlayDrawer.draw_pip_overlay(frame, vp_norm)

        elif cfg.scale_display != 1.0 and cfg.scale_display > 0:
            new_w = int(frame.shape[1] * cfg.scale_display)
            new_h = int(frame.shape[0] * cfg.scale_display)
            frame = cv2.resize(frame, (new_w, new_h))

        return frame

    # ── OpenCV window helpers ─────────────────────────────────────────────

    def show(self, frame: np.ndarray) -> int:
        """Display in an OpenCV window. Returns cv2.waitKey() value."""
        cv2.imshow(self._cfg.window_name, frame)
        return cv2.waitKey(self._cfg.wait_key_ms)

    def destroy(self) -> None:
        cv2.destroyWindow(self._cfg.window_name)

    # ── Utilities ────────────────────────────────────────────────────────

    @staticmethod
    def _blank_frame(h: int = 480, w: int = 640) -> np.ndarray:
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(
            frame, "No Frame", (w // 2 - 60, h // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (128, 128, 128), 2,
        )
        return frame
