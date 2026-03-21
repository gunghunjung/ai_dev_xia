"""
core/viewport_manager.py — Professional viewport management: zoom, pan, fullscreen.

══════════════════════════════════════════════════════════════════════════════
Mathematical Model
══════════════════════════════════════════════════════════════════════════════

At zoom level Z the viewport covers a rectangular window inside the source frame:

    vp_w  = src_w / Z        (source pixels visible horizontally)
    vp_h  = src_h / Z        (source pixels visible vertically)

The viewport centre is tracked as normalised coords (cx, cy) ∈ [0, 1]:

    vp_x  = cx · src_w − vp_w / 2    clamped to [0, src_w − vp_w]
    vp_y  = cy · src_h − vp_h / 2    clamped to [0, src_h − vp_h]

Stretching the cropped viewport to fill the canvas:

    sx = canvas_w / vp_w      (display pixels per source pixel, horizontal)
    sy = canvas_h / vp_h      (display pixels per source pixel, vertical)

    NOTE: sx ≈ sy when source and canvas share the same aspect ratio.
          When they differ a small stretch occurs (same as the existing
          non-aspect-ratio-preserving rendering mode).

Coordinate transforms:

    original → display:   dx = (ox − vp_x) · sx
                          dy = (oy − vp_y) · sy

    display → original:   ox = dx / sx + vp_x
                          oy = dy / sy + vp_y

Pan (image-dragging convention):

    The user grabs the image and drags it, so a drag of (+dx_d, +dy_d)
    reveals the opposite (left / up) portion of the source frame.

        Δcx = −dx_d / (sx · src_w)
        Δcy = −dy_d / (sy · src_h)

Zoom (centre-based):

    Zoom changes vp_w / vp_h while keeping the viewport centre fixed.
    _clamp_center() re-constrains after every zoom or source-size change.

ROI Fullscreen:

    focus_roi(rect) sets zoom and centre so the ROI fills the canvas
    (plus a configurable padding fraction).  exit_focus() restores the
    previous zoom and centre.

PIP (Picture-in-Picture minimap):

    get_pip_data() returns the viewport rect as normalised [0, 1] fractions
    so the GUI can draw a coloured rectangle over a thumbnail of the full frame.

══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


# Type aliases
Rect  = Tuple[float, float, float, float]   # (x, y, w, h)  float
IRect = Tuple[int,   int,   int,   int]     # (x, y, w, h)  int


# ---------------------------------------------------------------------------
# ViewportState  — snapshot for undo / restore
# ---------------------------------------------------------------------------

@dataclass
class ViewportState:
    """Immutable snapshot of ViewportManager state (used for undo / restore)."""
    zoom: float = 1.0
    center_x: float = 0.5
    center_y: float = 0.5


# ---------------------------------------------------------------------------
# ViewportManager
# ---------------------------------------------------------------------------

class ViewportManager:
    """
    Manages zoom level, pan offset and viewport-aware coordinate transforms.

    Design goals
    ────────────
    • All ROI / annotation coordinates are ALWAYS in original source-pixel space.
    • This class computes the transform from source → canvas on the fly.
    • Zoom is centred on the current viewport centre (stable, predictable).
    • Pan is bounded: the viewport never reveals areas outside the source frame.
    • ROI fullscreen: temporarily set viewport to a specific ROI rect.
    • PIP data: get_pip_data() for the minimap overlay.

    Thread safety: designed to be used only from the GUI (main) thread.
    """

    #: Zoom limits
    MIN_ZOOM: float = 0.5
    MAX_ZOOM: float = 20.0

    #: ROI fullscreen padding (fraction of ROI size added around the ROI)
    ROI_PAD: float = 0.15

    def __init__(self) -> None:
        # Current state
        self._zoom: float       = 1.0
        self._center_x: float   = 0.5   # normalised [0, 1]
        self._center_y: float   = 0.5

        # Source frame size (updated when first frame arrives)
        self._src_w: int = 1
        self._src_h: int = 1

        # Canvas (display) size (updated on <Configure>)
        self._canvas_w: int = 1
        self._canvas_h: int = 1

        # ROI fullscreen state
        self._roi_fullscreen: bool = False
        self._saved_state: Optional[ViewportState] = None

    # ── Size updates ──────────────────────────────────────────────────────

    def update_source(self, src_w: int, src_h: int) -> None:
        """Call whenever the source frame dimensions change."""
        if src_w < 1 or src_h < 1:
            return
        changed = (src_w != self._src_w or src_h != self._src_h)
        self._src_w = src_w
        self._src_h = src_h
        if changed:
            self._clamp_center()

    def update_canvas(self, canvas_w: int, canvas_h: int) -> None:
        """Call whenever the canvas (display area) is resized."""
        self._canvas_w = max(canvas_w, 1)
        self._canvas_h = max(canvas_h, 1)

    # ── Zoom ──────────────────────────────────────────────────────────────

    def zoom_in(self, factor: float = 1.25) -> None:
        """Zoom in, keeping the viewport centre fixed."""
        new_zoom = min(self._zoom * factor, self.MAX_ZOOM)
        self._zoom = new_zoom
        self._clamp_center()

    def zoom_out(self, factor: float = 1.25) -> None:
        """Zoom out, keeping the viewport centre fixed."""
        new_zoom = max(self._zoom / factor, self.MIN_ZOOM)
        self._zoom = new_zoom
        self._clamp_center()

    def zoom_reset(self) -> None:
        """Reset to 1× zoom, centred on the full frame."""
        self._zoom = 1.0
        self._center_x = 0.5
        self._center_y = 0.5
        self._roi_fullscreen = False
        self._saved_state = None

    # ── Pan ───────────────────────────────────────────────────────────────

    def pan(self, dx_display: float, dy_display: float) -> None:
        """
        Pan the viewport by (dx_display, dy_display) canvas pixels.

        Positive dx_display → image moves right → viewport shifts left
        (image-dragging / "Google Maps" convention).
        """
        sx, sy = self._display_scale()
        self._center_x -= dx_display / (sx * self._src_w)
        self._center_y -= dy_display / (sy * self._src_h)
        self._clamp_center()

    # ── ROI Fullscreen ────────────────────────────────────────────────────

    def focus_roi(
        self,
        roi_rect: Tuple[int, int, int, int],
        padding: float = ROI_PAD,
    ) -> None:
        """
        Set the viewport so that *roi_rect* fills the canvas with padding.

        The previous state is saved and can be restored with exit_focus().

        Args:
            roi_rect: (x, y, w, h) in original source pixel coordinates.
            padding:  Fraction of ROI size to add around the ROI on each side.
        """
        if not self._roi_fullscreen:
            self._saved_state = ViewportState(
                zoom=self._zoom,
                center_x=self._center_x,
                center_y=self._center_y,
            )

        rx, ry, rw, rh = roi_rect
        if rw < 1 or rh < 1:
            return

        # Centre the viewport on the ROI
        self._center_x = (rx + rw / 2.0) / self._src_w
        self._center_y = (ry + rh / 2.0) / self._src_h

        # Determine the zoom that makes the padded ROI fill the canvas.
        # We want:  (rw * (1 + 2·pad)) / (src_w / Z)  = 1  →  Z = src_w / (rw*(1+2p))
        padded_w = rw * (1.0 + 2.0 * padding)
        padded_h = rh * (1.0 + 2.0 * padding)
        zoom_x = self._src_w / max(padded_w, 1.0)
        zoom_y = self._src_h / max(padded_h, 1.0)

        # Use the larger zoom that still keeps the full ROI visible.
        # Also factor in canvas vs source aspect ratio so the ROI actually
        # fills the canvas nicely without over-cropping.
        canvas_ar = self._canvas_w / max(self._canvas_h, 1)
        roi_ar    = rw / max(rh, 1)
        if canvas_ar > roi_ar:          # canvas is wider → height is limiting
            new_zoom = zoom_y
        else:
            new_zoom = zoom_x

        self._zoom = max(self.MIN_ZOOM, min(new_zoom, self.MAX_ZOOM))
        self._clamp_center()
        self._roi_fullscreen = True

    def exit_focus(self) -> None:
        """Restore the viewport state that existed before focus_roi()."""
        if self._saved_state is not None:
            self._zoom     = self._saved_state.zoom
            self._center_x = self._saved_state.center_x
            self._center_y = self._saved_state.center_y
            self._clamp_center()
            self._saved_state = None
        else:
            self.zoom_reset()
        self._roi_fullscreen = False

    def toggle_focus(self, roi_rect: Tuple[int, int, int, int]) -> None:
        """Toggle between ROI fullscreen and normal view."""
        if self._roi_fullscreen:
            self.exit_focus()
        else:
            self.focus_roi(roi_rect)

    # ── Coordinate transforms ─────────────────────────────────────────────

    def original_to_display(self, ox: float, oy: float) -> Tuple[float, float]:
        """Map a point from original source pixel coords to canvas display coords."""
        vp_x, vp_y = self._viewport_topleft()
        sx, sy = self._display_scale()
        return (ox - vp_x) * sx, (oy - vp_y) * sy

    def display_to_original(self, dx: float, dy: float) -> Tuple[float, float]:
        """Map a point from canvas display coords to original source pixel coords."""
        vp_x, vp_y = self._viewport_topleft()
        sx, sy = self._display_scale()
        return dx / sx + vp_x, dy / sy + vp_y

    def original_to_display_rect(
        self, x: float, y: float, w: float, h: float
    ) -> Tuple[float, float, float, float]:
        """Map a rect from original source pixel coords to canvas display coords."""
        dx1, dy1 = self.original_to_display(x, y)
        dx2, dy2 = self.original_to_display(x + w, y + h)
        return dx1, dy1, dx2 - dx1, dy2 - dy1

    def display_to_original_rect(
        self, dx: float, dy: float, dw: float, dh: float
    ) -> Tuple[float, float, float, float]:
        """Map a rect from canvas display coords to original source pixel coords."""
        ox1, oy1 = self.display_to_original(dx, dy)
        ox2, oy2 = self.display_to_original(dx + dw, dy + dh)
        return ox1, oy1, ox2 - ox1, oy2 - oy1

    def clamp_to_source(
        self, x: float, y: float, w: float, h: float
    ) -> IRect:
        """Clamp a rect (in source pixel coords) to lie within the source frame."""
        x1 = max(0.0, min(x, self._src_w - 1))
        y1 = max(0.0, min(y, self._src_h - 1))
        x2 = max(x1 + 1, min(x + w, self._src_w))
        y2 = max(y1 + 1, min(y + h, self._src_h))
        return (round(x1), round(y1), round(x2 - x1), round(y2 - y1))

    def is_point_in_canvas(self, dx: float, dy: float) -> bool:
        """Return True if the display point is within the canvas area."""
        return 0 <= dx <= self._canvas_w and 0 <= dy <= self._canvas_h

    # ── Frame cropping ────────────────────────────────────────────────────

    def crop_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Crop *frame* to the current viewport region and resize to canvas size.

        Returns None if the frame is empty or the viewport is degenerate.
        """
        import cv2  # lazy import — optional if cv2 not present at import time

        if frame is None or frame.size == 0:
            return None

        vx, vy, vw, vh = self.viewport_rect_int
        if vw < 1 or vh < 1:
            return None

        crop = frame[vy : vy + vh, vx : vx + vw]
        if crop.size == 0:
            return None

        # Resize to canvas size (stretch, same as existing rendering behaviour)
        return cv2.resize(
            crop,
            (self._canvas_w, self._canvas_h),
            interpolation=cv2.INTER_LINEAR,
        )

    # ── PIP helper ────────────────────────────────────────────────────────

    def get_pip_data(self) -> Tuple[float, float, float, float]:
        """
        Return the current viewport as normalised [0, 1] fractions
        of the source frame, suitable for drawing a rectangle on a minimap.

        Returns:
            (norm_x, norm_y, norm_w, norm_h)  — all in [0, 1]
        """
        vp_x, vp_y, vp_w, vp_h = self.viewport_rect
        sw, sh = max(self._src_w, 1), max(self._src_h, 1)
        return (
            vp_x / sw,
            vp_y / sh,
            vp_w / sw,
            vp_h / sh,
        )

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def zoom(self) -> float:
        return self._zoom

    @property
    def is_zoomed(self) -> bool:
        return self._zoom > 1.001

    @property
    def is_roi_fullscreen(self) -> bool:
        return self._roi_fullscreen

    @property
    def source_size(self) -> Tuple[int, int]:
        return (self._src_w, self._src_h)

    @property
    def canvas_size(self) -> Tuple[int, int]:
        return (self._canvas_w, self._canvas_h)

    @property
    def viewport_rect(self) -> Rect:
        """Current viewport as (x, y, w, h) in source pixel coords (float)."""
        vp_w = self._src_w / self._zoom
        vp_h = self._src_h / self._zoom
        vp_x, vp_y = self._viewport_topleft()
        return (vp_x, vp_y, vp_w, vp_h)

    @property
    def viewport_rect_int(self) -> IRect:
        """Current viewport as (x, y, w, h) for cv2 array slicing (int, clamped)."""
        vx, vy, vw, vh = self.viewport_rect
        x1 = max(0, round(vx))
        y1 = max(0, round(vy))
        x2 = min(self._src_w, round(vx + vw))
        y2 = min(self._src_h, round(vy + vh))
        return (x1, y1, max(1, x2 - x1), max(1, y2 - y1))

    # ── Internal helpers ──────────────────────────────────────────────────

    def _display_scale(self) -> Tuple[float, float]:
        """
        Returns (sx, sy): canvas pixels per source pixel for current viewport.
        """
        vp_w = self._src_w / max(self._zoom, 1e-9)
        vp_h = self._src_h / max(self._zoom, 1e-9)
        sx = self._canvas_w / max(vp_w, 1.0)
        sy = self._canvas_h / max(vp_h, 1.0)
        return sx, sy

    def _viewport_topleft(self) -> Tuple[float, float]:
        """
        Compute the viewport top-left corner (source pixel coords, clamped).
        """
        vp_w = self._src_w / max(self._zoom, 1e-9)
        vp_h = self._src_h / max(self._zoom, 1e-9)
        vp_x = self._center_x * self._src_w - vp_w / 2.0
        vp_y = self._center_y * self._src_h - vp_h / 2.0
        # Clamp so the viewport stays entirely within the source frame
        vp_x = max(0.0, min(vp_x, self._src_w - vp_w))
        vp_y = max(0.0, min(vp_y, self._src_h - vp_h))
        return vp_x, vp_y

    def _clamp_center(self) -> None:
        """Prevent the centre from drifting beyond the valid range for current zoom."""
        vp_w = self._src_w / max(self._zoom, 1e-9)
        vp_h = self._src_h / max(self._zoom, 1e-9)
        half_w_norm = (vp_w / 2.0) / max(self._src_w, 1)
        half_h_norm = (vp_h / 2.0) / max(self._src_h, 1)
        self._center_x = max(half_w_norm, min(self._center_x, 1.0 - half_w_norm))
        self._center_y = max(half_h_norm, min(self._center_y, 1.0 - half_h_norm))

    def __repr__(self) -> str:
        vx, vy, vw, vh = self.viewport_rect
        return (
            f"ViewportManager("
            f"zoom={self._zoom:.2f}x "
            f"src={self._src_w}×{self._src_h} "
            f"canvas={self._canvas_w}×{self._canvas_h} "
            f"vp=({vx:.0f},{vy:.0f},{vw:.0f}×{vh:.0f}) "
            f"roi_fs={self._roi_fullscreen})"
        )
