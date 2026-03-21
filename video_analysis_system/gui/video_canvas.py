"""
gui/video_canvas.py — Tkinter Canvas: viewport-aware video display + ROI editing.

Architecture
────────────
┌──────────────────────────────────────────────────────────────┐
│  Single Source of Truth: original source-pixel coordinates   │
│                                                              │
│  • ROI stored in original_rect (source pixels)               │
│  • Display pos = ViewportManager.original_to_display_rect()  │
│  • Mouse drag → ViewportManager.display_to_original_rect()   │
│  • <Configure> → ViewportManager.update_canvas()             │
│  • New frame   → ViewportManager.update_source() + crop      │
└──────────────────────────────────────────────────────────────┘

Interaction modes
─────────────────
ROI-draw mode OFF  (default):
  • Left drag  → pan viewport
  • Scroll     → zoom in / out (centre-based)

ROI-draw mode ON  (enabled via enable_roi_drawing()):
  • Left drag  → draw ROI rectangle
  • Scroll     → zoom in / out (still active)

PIP minimap (top-right corner):
  • Always visible when a video frame is present
  • Shows the full annotated frame scaled down
  • Yellow rectangle indicates current viewport region
  • Clicking the PIP teleports the viewport to that position

ROI fullscreen (keyboard shortcut "F"):
  • focus_roi(roi_id)  → zoom into the selected ROI
  • exit_focus()       → restore previous viewport
"""

from __future__ import annotations

import tkinter as tk
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from core.viewport_manager import ViewportManager
from core.states import SystemState


# ---------------------------------------------------------------------------
# Fixed-overlay colour maps (Tk hex colours, not BGR)
# ---------------------------------------------------------------------------

_STATE_BG: dict = {
    SystemState.NORMAL:        "#1a3a1a",
    SystemState.WARNING:       "#3a2a00",
    SystemState.ABNORMAL:      "#3a0a0a",
    SystemState.TRACKING_LOST: "#2a2a3a",
    SystemState.UNKNOWN:       "#1e1e2e",
}
_STATE_FG: dict = {
    SystemState.NORMAL:        "#a6e3a1",
    SystemState.WARNING:       "#f9e2af",
    SystemState.ABNORMAL:      "#f38ba8",
    SystemState.TRACKING_LOST: "#89b4fa",
    SystemState.UNKNOWN:       "#6c7086",
}
_STATE_LABEL: dict = {
    SystemState.NORMAL:        "■ 정상",
    SystemState.WARNING:       "▲ 경고",
    SystemState.ABNORMAL:      "✖ 이상",
    SystemState.TRACKING_LOST: "? 추적 손실",
    SystemState.UNKNOWN:       "— 불명",
}


# ---------------------------------------------------------------------------
# PIL helper
# ---------------------------------------------------------------------------

def _bgr_to_photo(frame: np.ndarray, w: int, h: int):
    """BGR numpy array → Tkinter PhotoImage, resized to (w × h)."""
    import cv2
    from PIL import Image, ImageTk
    if frame.shape[1] != w or frame.shape[0] != h:
        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return ImageTk.PhotoImage(image=Image.fromarray(rgb))


def _bgr_to_photo_exact(frame: np.ndarray):
    """BGR numpy array → Tkinter PhotoImage at frame's native resolution."""
    import cv2
    from PIL import Image, ImageTk
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return ImageTk.PhotoImage(image=Image.fromarray(rgb))


# ---------------------------------------------------------------------------
# ROI colour palette
# ---------------------------------------------------------------------------

_ROI_COLORS = [
    "#00e676", "#ff5252", "#40c4ff", "#ffab40",
    "#ea80fc", "#b2ff59", "#ff6d00", "#64ffda",
]


# ---------------------------------------------------------------------------
# PIP (minimap) constants
# ---------------------------------------------------------------------------

_PIP_W         = 180        # minimap width in canvas pixels
_PIP_H         = 100        # minimap height in canvas pixels
_PIP_MARGIN    = 10         # distance from canvas edge
_PIP_BG_COLOR  = "#0d0d1a"  # dark background
_PIP_BOX_COLOR = "#f9e2af"  # viewport indicator colour (warm yellow)
_PIP_BORDER    = "#585b70"  # outer border colour


# ---------------------------------------------------------------------------
# VideoCanvas
# ---------------------------------------------------------------------------

class VideoCanvas(tk.Canvas):
    """
    Viewport-aware video canvas with zoom, pan, PIP minimap and ROI editing.

    Public API
    ──────────
    update_frame(frame, src_w, src_h)     — display a new BGR frame
    clear_frame()                          — reset to placeholder state
    enable_roi_drawing(callback)           — enter ROI-draw mode
    disable_roi_drawing()                  — exit ROI-draw mode
    add_roi_overlay(roi_id, original_rect, color)
    remove_roi_overlay(roi_id)
    update_roi_overlay(roi_id, original_rect)
    clear_roi_overlays()
    focus_roi(roi_id)                      — zoom into ROI (ROI fullscreen)
    exit_focus()                           — restore normal viewport
    zoom_in() / zoom_out() / zoom_reset()
    pan(dx, dy)                            — programmatic pan (display pixels)
    """

    def __init__(self, master, width: int = 820, height: int = 560, **kwargs):
        super().__init__(
            master,
            width=width,
            height=height,
            bg="#1a1a2e",
            highlightthickness=0,
            **kwargs,
        )
        # Canvas size (avoid collision with tk.Canvas._w internal attribute)
        self._canvas_w: int = width
        self._canvas_h: int = height

        # ── ViewportManager ───────────────────────────────────────────────
        self._vp = ViewportManager()
        self._vp.update_canvas(width, height)

        # Source frame dimensions
        self._src_w: int = 1
        self._src_h: int = 1

        # ── Frame storage ─────────────────────────────────────────────────
        # Full annotated frame (at source resolution) for PIP thumbnail
        self._latest_frame: Optional[np.ndarray] = None
        # Tkinter photo references (GC prevention)
        self._main_photo: Optional[object] = None
        self._pip_photo:  Optional[object] = None

        # ── ROI state ─────────────────────────────────────────────────────
        # roi_id → original_rect (x, y, w, h) in source pixel coords
        self._roi_original: Dict[str, Tuple[int, int, int, int]] = {}
        # roi_id → (rect_canvas_id, label_canvas_id)
        self._roi_items:    Dict[str, Tuple[int, int]] = {}
        self._roi_colors:   Dict[str, str] = {}
        self._roi_counter:  int = 0

        # Currently selected ROI id (for focus / fullscreen)
        self._selected_roi_id: Optional[str] = None

        # ROI draw mode
        self._roi_draw_enabled: bool = False
        self._drawing:           bool = False
        self._draw_start: Optional[Tuple[int, int]] = None
        self._drag_rect_id: Optional[int] = None
        self._on_roi_drawn: Optional[Callable] = None   # (roi_id, original_rect)

        # ── Pan state ─────────────────────────────────────────────────────
        self._panning:    bool = False
        self._pan_start:  Optional[Tuple[int, int]] = None

        # ── Fixed-position annotation overlay ────────────────────────────
        # These Tkinter canvas items sit at fixed canvas positions and are
        # NEVER affected by viewport zoom / pan.
        self._ann_state_bg:   Optional[int] = None   # state banner background
        self._ann_state_txt:  Optional[int] = None   # state text
        self._ann_conf_txt:   Optional[int] = None   # confidence text
        self._ann_info_txt:   Optional[int] = None   # frame / fps / inference info
        self._ann_event_rect: Optional[int] = None   # red border on event
        self._ann_rules_txt:  Optional[int] = None   # triggered rules list
        self._ann_visible:    bool = False

        # ── PIP canvas items ──────────────────────────────────────────────
        self._pip_visible: bool = False
        self._pip_img_id:    Optional[int] = None
        self._pip_bg_id:     Optional[int] = None
        self._pip_border_id: Optional[int] = None
        self._pip_vp_id:     Optional[int] = None
        self._pip_label_id:  Optional[int] = None

        # ── Placeholder text ──────────────────────────────────────────────
        self._placeholder_id: Optional[int] = self.create_text(
            width // 2, height // 2,
            text="소스 없음  —  ▶ 시작 버튼을 누르세요",
            fill="#555577",
            font=("맑은 고딕", 14),
        )

        # ── Event bindings ────────────────────────────────────────────────
        self.bind("<Configure>",        self._on_configure)
        self.bind("<ButtonPress-1>",    self._on_lbutton_press)
        self.bind("<B1-Motion>",        self._on_lbutton_drag)
        self.bind("<ButtonRelease-1>",  self._on_lbutton_release)
        self.bind("<ButtonPress-3>",    self._on_rbutton_press)

        # Mouse wheel (platform-specific)
        self.bind("<MouseWheel>",       self._on_mousewheel)     # Windows / macOS
        self.bind("<Button-4>",         self._on_scroll_up)      # Linux scroll up
        self.bind("<Button-5>",         self._on_scroll_down)    # Linux scroll down

        # Set initial cursor
        self._update_cursor()

    # ─────────────────────────────────────────────────────────────────────
    # Viewport / coordinate API (backward-compatible facade)
    # ─────────────────────────────────────────────────────────────────────

    @property
    def viewport(self) -> ViewportManager:
        """Direct access to the ViewportManager."""
        return self._vp

    @property
    def zoom_factor(self) -> float:
        return self._vp.zoom

    @property
    def source_size(self) -> Tuple[int, int]:
        return (self._src_w, self._src_h)

    @property
    def canvas_size(self) -> Tuple[int, int]:
        return (self._canvas_w, self._canvas_h)

    def zoom_in(self, factor: float = 1.25) -> None:
        self._vp.zoom_in(factor)
        self._redraw_all_roi_overlays()
        self._update_pip()

    def zoom_out(self, factor: float = 1.25) -> None:
        self._vp.zoom_out(factor)
        self._redraw_all_roi_overlays()
        self._update_pip()

    def zoom_reset(self) -> None:
        self._vp.zoom_reset()
        self._redraw_all_roi_overlays()
        self._update_pip()

    def pan(self, dx: float, dy: float) -> None:
        """Programmatic pan by (dx, dy) display pixels."""
        self._vp.pan(dx, dy)
        self._redraw_all_roi_overlays()
        self._update_pip()

    # ─────────────────────────────────────────────────────────────────────
    # ROI fullscreen toggle
    # ─────────────────────────────────────────────────────────────────────

    def focus_roi(self, roi_id: str) -> bool:
        """
        Zoom the viewport into the given ROI (fullscreen mode).

        Returns True on success, False if the roi_id is not found.
        """
        rect = self._roi_original.get(roi_id)
        if rect is None:
            return False
        self._selected_roi_id = roi_id
        self._vp.focus_roi(rect)
        self._redraw_all_roi_overlays()
        self._update_pip()
        return True

    def exit_focus(self) -> None:
        """Exit ROI fullscreen and restore the previous viewport."""
        self._vp.exit_focus()
        self._redraw_all_roi_overlays()
        self._update_pip()

    def toggle_focus(self, roi_id: Optional[str] = None) -> None:
        """
        Toggle ROI fullscreen for *roi_id* (or the currently selected ROI).
        If already in fullscreen mode, exit regardless of roi_id.
        """
        if self._vp.is_roi_fullscreen:
            self.exit_focus()
        else:
            target = roi_id or self._selected_roi_id
            if target:
                self.focus_roi(target)

    @property
    def selected_roi_id(self) -> Optional[str]:
        return self._selected_roi_id

    @selected_roi_id.setter
    def selected_roi_id(self, roi_id: Optional[str]) -> None:
        self._selected_roi_id = roi_id
        self._highlight_selected_roi()

    # ─────────────────────────────────────────────────────────────────────
    # Frame update
    # ─────────────────────────────────────────────────────────────────────

    def update_frame(
        self,
        frame: np.ndarray,
        src_w: Optional[int] = None,
        src_h: Optional[int] = None,
    ) -> None:
        """
        Display a BGR numpy frame on the canvas.

        The frame is stored at full resolution; the viewport manager crops and
        scales it to the current canvas size. ROI overlays are re-positioned
        using the viewport coordinate transform.

        Args:
            frame:  BGR numpy array (H×W×3) — the full annotated frame
            src_w:  original source width  (inferred from frame.shape if None)
            src_h:  original source height (inferred from frame.shape if None)
        """
        # Remove placeholder on first frame
        if self._placeholder_id is not None:
            self.delete(self._placeholder_id)
            self._placeholder_id = None

        # Update source dimensions
        new_src_w = src_w if src_w is not None else frame.shape[1]
        new_src_h = src_h if src_h is not None else frame.shape[0]
        if new_src_w != self._src_w or new_src_h != self._src_h:
            self._src_w = new_src_w
            self._src_h = new_src_h
            self._vp.update_source(new_src_w, new_src_h)

        # Store full frame for PIP
        self._latest_frame = frame

        # Crop viewport from the full frame and scale to canvas
        cropped = self._vp.crop_frame(frame)
        if cropped is None:
            # Fallback: show full frame
            cropped = frame

        # Render main image
        photo = _bgr_to_photo(cropped, self._canvas_w, self._canvas_h)
        self._main_photo = photo
        self.create_image(0, 0, anchor="nw", image=photo)

        # Ensure ROI overlays and PIP are on top
        self._raise_roi_items()
        self._update_pip()

    def clear_frame(self) -> None:
        """Reset canvas to placeholder state."""
        self.delete("all")
        self._main_photo = None
        self._pip_photo  = None
        self._latest_frame = None
        self._roi_items.clear()
        self._pip_img_id    = None
        self._pip_bg_id     = None
        self._pip_border_id = None
        self._pip_vp_id     = None
        self._pip_label_id  = None
        self._drag_rect_id  = None
        self._pip_visible   = False
        # Annotation item references cleared (deleted by "all" above)
        self._ann_state_bg   = None
        self._ann_state_txt  = None
        self._ann_conf_txt   = None
        self._ann_info_txt   = None
        self._ann_event_rect = None
        self._ann_rules_txt  = None
        self._ann_visible    = False
        self._placeholder_id = self.create_text(
            self._canvas_w // 2, self._canvas_h // 2,
            text="소스 없음  —  ▶ 시작 버튼을 누르세요",
            fill="#555577",
            font=("맑은 고딕", 14),
        )

    # ─────────────────────────────────────────────────────────────────────
    # Fixed-position annotation overlay
    # ─────────────────────────────────────────────────────────────────────

    def update_annotations(
        self,
        state: "SystemState",
        confidence: float,
        frame_index: int,
        timestamp: float,
        fps: float,
        inference_ms: float,
        is_event: bool = False,
        event_type: str = "",
        triggered_rules: Optional[List] = None,
    ) -> None:
        """
        Draw (or update) fixed-position annotation overlays on the canvas.

        These are rendered as Tkinter canvas items at absolute canvas
        coordinates — they are NEVER scaled, cropped, or moved when the
        viewport zooms or pans.

        Call this every frame from main_window._handle_result().
        """
        cw, ch = self._canvas_w, self._canvas_h
        bg_color  = _STATE_BG.get(state,  "#1e1e2e")
        txt_color = _STATE_FG.get(state,  "#6c7086")
        label     = _STATE_LABEL.get(state, "—")
        banner_h  = 28

        # ── State banner (top strip) ──────────────────────────────────────
        if self._ann_state_bg is None:
            self._ann_state_bg = self.create_rectangle(
                0, 0, cw, banner_h, fill=bg_color, outline="", width=0,
            )
        else:
            self.coords(self._ann_state_bg, 0, 0, cw, banner_h)
            self.itemconfig(self._ann_state_bg, fill=bg_color)

        state_text = f"{label}  conf {confidence:.2f}"
        if self._ann_state_txt is None:
            self._ann_state_txt = self.create_text(
                8, banner_h // 2,
                text=state_text, anchor="w",
                fill=txt_color, font=("맑은 고딕", 10, "bold"),
            )
        else:
            self.coords(self._ann_state_txt, 8, banner_h // 2)
            self.itemconfig(self._ann_state_txt, text=state_text, fill=txt_color)

        # ── Frame info (bottom strip) ─────────────────────────────────────
        info_text = (
            f"Frame {frame_index:,}   {timestamp:.2f}s   "
            f"FPS {fps:.1f}   inf {inference_ms:.1f}ms"
        )
        if self._ann_info_txt is None:
            self._ann_info_txt = self.create_text(
                6, ch - 6,
                text=info_text, anchor="sw",
                fill="#585b70", font=("Consolas", 8),
            )
        else:
            self.coords(self._ann_info_txt, 6, ch - 6)
            self.itemconfig(self._ann_info_txt, text=info_text)

        # ── Triggered rules (bottom-right, above info) ────────────────────
        rules = triggered_rules or []
        rules_text = "\n".join(
            f"! {r.rule_name}: {r.message[:45]}"
            for r in rules[:4]
        )
        if self._ann_rules_txt is None:
            self._ann_rules_txt = self.create_text(
                cw - 6, ch - 18,
                text=rules_text, anchor="se",
                fill="#f9e2af", font=("Consolas", 8),
                justify="right",
            )
        else:
            self.coords(self._ann_rules_txt, cw - 6, ch - 18)
            self.itemconfig(self._ann_rules_txt, text=rules_text)

        # ── Event border (full canvas red rectangle) ──────────────────────
        if is_event:
            if self._ann_event_rect is None:
                self._ann_event_rect = self.create_rectangle(
                    2, 2, cw - 2, ch - 2,
                    outline="#f38ba8", width=4,
                )
            else:
                self.coords(self._ann_event_rect, 2, 2, cw - 2, ch - 2)
                self.itemconfig(self._ann_event_rect, state="normal")
            # Event label
            event_label = f"⚡ 이벤트: {event_type}"
            if self._ann_conf_txt is None:
                self._ann_conf_txt = self.create_text(
                    cw // 2, ch // 2,
                    text=event_label, anchor="center",
                    fill="#f38ba8", font=("맑은 고딕", 14, "bold"),
                )
            else:
                self.coords(self._ann_conf_txt, cw // 2, ch // 2)
                self.itemconfig(self._ann_conf_txt,
                                text=event_label, state="normal")
        else:
            if self._ann_event_rect is not None:
                self.itemconfig(self._ann_event_rect, state="hidden")
            if self._ann_conf_txt is not None:
                self.itemconfig(self._ann_conf_txt, state="hidden")

        # Raise all annotation items above the video frame and ROI overlays
        for item in (
            self._ann_state_bg, self._ann_state_txt,
            self._ann_info_txt, self._ann_rules_txt,
            self._ann_event_rect, self._ann_conf_txt,
        ):
            if item:
                self.tag_raise(item)

        self._ann_visible = True

    # ─────────────────────────────────────────────────────────────────────
    # ROI overlay management
    # ─────────────────────────────────────────────────────────────────────

    def add_roi_overlay(
        self,
        roi_id: str,
        original_rect: Tuple[int, int, int, int],
        color: Optional[str] = None,
    ) -> None:
        """
        Show a ROI on the canvas.

        Args:
            roi_id:        ROI identifier string
            original_rect: (x, y, w, h) in source pixel coords
            color:         Tk colour string; cycled from palette if None
        """
        self.remove_roi_overlay(roi_id)
        if color is None:
            color = _ROI_COLORS[self._roi_counter % len(_ROI_COLORS)]
        self._roi_colors[roi_id]   = color
        self._roi_original[roi_id] = tuple(original_rect)  # type: ignore[assignment]
        rect_id, label_id = self._create_roi_canvas_items(roi_id, original_rect, color)
        self._roi_items[roi_id] = (rect_id, label_id)
        self._roi_counter += 1

    def remove_roi_overlay(self, roi_id: str) -> None:
        if roi_id in self._roi_items:
            for item_id in self._roi_items.pop(roi_id):
                self.delete(item_id)
        self._roi_original.pop(roi_id, None)
        self._roi_colors.pop(roi_id, None)

    def clear_roi_overlays(self) -> None:
        for roi_id in list(self._roi_items):
            self.remove_roi_overlay(roi_id)

    def update_roi_overlay(
        self,
        roi_id: str,
        original_rect: Tuple[int, int, int, int],
    ) -> None:
        """Update a ROI's source-pixel coords and refresh its canvas items."""
        color = self._roi_colors.get(roi_id, _ROI_COLORS[0])
        self.remove_roi_overlay(roi_id)
        self._roi_colors[roi_id] = color
        self.add_roi_overlay(roi_id, original_rect, color)

    def get_roi_original_rect(
        self, roi_id: str
    ) -> Optional[Tuple[int, int, int, int]]:
        return self._roi_original.get(roi_id)

    def list_roi_ids(self) -> List[str]:
        return list(self._roi_original.keys())

    # ─────────────────────────────────────────────────────────────────────
    # ROI draw mode
    # ─────────────────────────────────────────────────────────────────────

    def enable_roi_drawing(self, callback: Optional[Callable] = None) -> None:
        """
        Enter ROI-draw mode.  Left-drag draws a new ROI rectangle.

        Args:
            callback: called with (roi_id: str, original_rect: tuple) after draw
        """
        self._roi_draw_enabled = True
        self._on_roi_drawn = callback
        self._update_cursor()

    def disable_roi_drawing(self) -> None:
        """Exit ROI-draw mode.  Left-drag becomes pan."""
        self._roi_draw_enabled = False
        self._update_cursor()
        if self._drag_rect_id:
            self.delete(self._drag_rect_id)
            self._drag_rect_id = None
        self._drawing   = False
        self._draw_start = None

    # ─────────────────────────────────────────────────────────────────────
    # Layout change
    # ─────────────────────────────────────────────────────────────────────

    def _on_configure(self, event: tk.Event) -> None:
        """Canvas resized: update viewport dimensions and redraw."""
        self._canvas_w = event.width
        self._canvas_h = event.height
        self._vp.update_canvas(event.width, event.height)
        self._redraw_all_roi_overlays()
        self._update_pip()
        # Reposition bottom-edge annotation items (their y depends on canvas height)
        if self._ann_info_txt:
            self.coords(self._ann_info_txt, 6, self._canvas_h - 6)
        if self._ann_rules_txt:
            self.coords(self._ann_rules_txt, self._canvas_w - 6, self._canvas_h - 18)
        if self._ann_state_bg:
            self.coords(self._ann_state_bg, 0, 0, self._canvas_w, 28)
        if self._ann_event_rect:
            self.coords(self._ann_event_rect,
                        2, 2, self._canvas_w - 2, self._canvas_h - 2)
        if self._ann_conf_txt:
            self.coords(self._ann_conf_txt, self._canvas_w // 2, self._canvas_h // 2)

    # ─────────────────────────────────────────────────────────────────────
    # Mouse: left button (draw ROI  OR  pan)
    # ─────────────────────────────────────────────────────────────────────

    def _on_lbutton_press(self, event: tk.Event) -> None:
        # Check if click is on PIP → teleport viewport
        if self._is_on_pip(event.x, event.y):
            self._pip_click_teleport(event.x, event.y)
            return

        # Check if click is on a ROI box → select it
        clicked_roi = self._roi_id_at(event.x, event.y)
        if clicked_roi:
            self.selected_roi_id = clicked_roi

        if self._roi_draw_enabled:
            self._start_drawing(event)
        else:
            self._start_pan(event)

    def _on_lbutton_drag(self, event: tk.Event) -> None:
        if self._roi_draw_enabled and self._drawing:
            self._update_drawing(event)
        elif self._panning:
            self._update_pan(event)

    def _on_lbutton_release(self, event: tk.Event) -> None:
        if self._roi_draw_enabled and self._drawing:
            self._finish_drawing(event)
        elif self._panning:
            self._finish_pan(event)

    # ─────────────────────────────────────────────────────────────────────
    # Mouse: right button (deselect / context)
    # ─────────────────────────────────────────────────────────────────────

    def _on_rbutton_press(self, _event: tk.Event) -> None:
        self.selected_roi_id = None

    # ─────────────────────────────────────────────────────────────────────
    # Mouse: wheel (zoom)
    # ─────────────────────────────────────────────────────────────────────

    def _on_mousewheel(self, event: tk.Event) -> None:
        """Windows / macOS wheel: event.delta > 0 → zoom in."""
        if event.delta > 0:
            self._vp.zoom_in(1.15)
        else:
            self._vp.zoom_out(1.15)
        self._after_zoom()

    def _on_scroll_up(self, _event: tk.Event) -> None:
        """Linux Button-4 → zoom in."""
        self._vp.zoom_in(1.15)
        self._after_zoom()

    def _on_scroll_down(self, _event: tk.Event) -> None:
        """Linux Button-5 → zoom out."""
        self._vp.zoom_out(1.15)
        self._after_zoom()

    def _after_zoom(self) -> None:
        self._redraw_all_roi_overlays()
        self._update_pip()
        # Notify parent via virtual event so status bar can update zoom %
        self.event_generate("<<ZoomChanged>>")

    # ─────────────────────────────────────────────────────────────────────
    # Draw-ROI implementation
    # ─────────────────────────────────────────────────────────────────────

    def _start_drawing(self, event: tk.Event) -> None:
        self._drawing = True
        self._draw_start = (event.x, event.y)
        if self._drag_rect_id:
            self.delete(self._drag_rect_id)
        self._drag_rect_id = self.create_rectangle(
            event.x, event.y, event.x, event.y,
            outline="#ffffff", width=2, dash=(4, 2),
        )

    def _update_drawing(self, event: tk.Event) -> None:
        if self._draw_start and self._drag_rect_id:
            x0, y0 = self._draw_start
            self.coords(self._drag_rect_id, x0, y0, event.x, event.y)

    def _finish_drawing(self, event: tk.Event) -> None:
        self._drawing = False
        if self._draw_start is None:
            return

        x0, y0 = self._draw_start
        x1, y1 = event.x, event.y
        self._draw_start = None

        lx, rx = (x0, x1) if x0 < x1 else (x1, x0)
        ty, by = (y0, y1) if y0 < y1 else (y1, y0)

        if self._drag_rect_id:
            self.delete(self._drag_rect_id)
            self._drag_rect_id = None

        if (rx - lx) < 8 or (by - ty) < 8:
            return

        # Convert display → original source pixel coords
        ox, oy, ow, oh = self._vp.display_to_original_rect(lx, ty, rx - lx, by - ty)
        ox, oy, ow, oh = self._vp.clamp_to_source(ox, oy, ow, oh)

        if ow < 4 or oh < 4:
            return

        if self._on_roi_drawn:
            roi_id = f"roi_{self._roi_counter}"
            self._on_roi_drawn(roi_id, (ox, oy, ow, oh))

    # ─────────────────────────────────────────────────────────────────────
    # Pan implementation
    # ─────────────────────────────────────────────────────────────────────

    def _start_pan(self, event: tk.Event) -> None:
        if not self._vp.is_zoomed:
            return      # Nothing to pan at 1× zoom
        self._panning  = True
        self._pan_start = (event.x, event.y)
        self.config(cursor="fleur")

    def _update_pan(self, event: tk.Event) -> None:
        if not self._panning or self._pan_start is None:
            return
        dx = event.x - self._pan_start[0]
        dy = event.y - self._pan_start[1]
        self._pan_start = (event.x, event.y)
        self._vp.pan(dx, dy)
        self._redraw_all_roi_overlays()
        self._update_pip()

    def _finish_pan(self, _event: tk.Event) -> None:
        self._panning   = False
        self._pan_start = None
        self._update_cursor()

    # ─────────────────────────────────────────────────────────────────────
    # PIP (minimap) rendering
    # ─────────────────────────────────────────────────────────────────────

    def _pip_x(self) -> int:
        """Canvas X coordinate of PIP top-left corner."""
        return self._canvas_w - _PIP_W - _PIP_MARGIN

    def _pip_y(self) -> int:
        """Canvas Y coordinate of PIP top-left corner."""
        return _PIP_MARGIN

    def _update_pip(self) -> None:
        """Redraw the PIP minimap overlay."""
        if self._latest_frame is None:
            self._hide_pip()
            return

        px = self._pip_x()
        py = self._pip_y()

        # ── Background & border ──────────────────────────────────────────
        if self._pip_bg_id is None:
            self._pip_bg_id = self.create_rectangle(
                px - 1, py - 1, px + _PIP_W + 1, py + _PIP_H + 1,
                fill=_PIP_BG_COLOR, outline=_PIP_BORDER, width=1,
            )
        else:
            self.coords(
                self._pip_bg_id,
                px - 1, py - 1, px + _PIP_W + 1, py + _PIP_H + 1,
            )

        # ── Thumbnail image ──────────────────────────────────────────────
        try:
            pip_photo = _bgr_to_photo(self._latest_frame, _PIP_W, _PIP_H)
            self._pip_photo = pip_photo   # GC prevention
            if self._pip_img_id is None:
                self._pip_img_id = self.create_image(px, py, anchor="nw", image=pip_photo)
            else:
                self.coords(self._pip_img_id, px, py)
                self.itemconfig(self._pip_img_id, image=pip_photo)
        except Exception:
            pass

        # ── Viewport indicator rectangle ─────────────────────────────────
        nx, ny, nw, nh = self._vp.get_pip_data()
        vx1 = px + nx * _PIP_W
        vy1 = py + ny * _PIP_H
        vx2 = px + (nx + nw) * _PIP_W
        vy2 = py + (ny + nh) * _PIP_H

        # Clamp to PIP area
        vx1, vx2 = max(px, vx1), min(px + _PIP_W, vx2)
        vy1, vy2 = max(py, vy1), min(py + _PIP_H, vy2)

        if self._pip_vp_id is None:
            self._pip_vp_id = self.create_rectangle(
                vx1, vy1, vx2, vy2,
                outline=_PIP_BOX_COLOR, width=2,
            )
        else:
            self.coords(self._pip_vp_id, vx1, vy1, vx2, vy2)

        # ── "MINIMAP" label ──────────────────────────────────────────────
        if self._pip_label_id is None:
            self._pip_label_id = self.create_text(
                px + 4, py + _PIP_H - 2,
                text="MINIMAP",
                anchor="sw",
                fill="#585b70",
                font=("Consolas", 7),
            )
        else:
            self.coords(self._pip_label_id, px + 4, py + _PIP_H - 2)

        # Raise PIP items above the video frame
        for item in (
            self._pip_bg_id, self._pip_img_id,
            self._pip_vp_id, self._pip_label_id,
        ):
            if item:
                self.tag_raise(item)

        self._pip_visible = True

    def _hide_pip(self) -> None:
        for attr in ("_pip_bg_id", "_pip_img_id", "_pip_vp_id", "_pip_label_id"):
            item_id = getattr(self, attr)
            if item_id:
                self.delete(item_id)
                setattr(self, attr, None)
        self._pip_visible = False

    def _is_on_pip(self, x: int, y: int) -> bool:
        """Return True if (x, y) is inside the PIP area."""
        if not self._pip_visible:
            return False
        px, py = self._pip_x(), self._pip_y()
        return px <= x <= px + _PIP_W and py <= y <= py + _PIP_H

    def _pip_click_teleport(self, x: int, y: int) -> None:
        """Click on PIP → move viewport centre to that location."""
        px, py = self._pip_x(), self._pip_y()
        # Normalised position within PIP
        nx = (x - px) / _PIP_W
        ny = (y - py) / _PIP_H
        nx = max(0.0, min(1.0, nx))
        ny = max(0.0, min(1.0, ny))
        # Update viewport centre (normalised coords)
        self._vp._center_x = nx     # type: ignore[attr-defined]
        self._vp._center_y = ny     # type: ignore[attr-defined]
        self._vp._clamp_center()    # type: ignore[attr-defined]
        self._redraw_all_roi_overlays()
        self._update_pip()
        self.event_generate("<<ViewportChanged>>")

    # ─────────────────────────────────────────────────────────────────────
    # ROI canvas item helpers
    # ─────────────────────────────────────────────────────────────────────

    def _create_roi_canvas_items(
        self,
        roi_id: str,
        original_rect: Tuple[int, int, int, int],
        color: str,
    ) -> Tuple[int, int]:
        """
        Create canvas rectangle + label for a ROI.

        Coordinates are computed via the viewport transform so the overlay
        is always positioned correctly regardless of zoom/pan.
        """
        x, y, w, h = original_rect
        dx, dy, dw, dh = self._vp.original_to_display_rect(x, y, w, h)
        dx, dy, dw, dh = round(dx), round(dy), round(dw), round(dh)

        is_selected = (roi_id == self._selected_roi_id)
        line_w = 3 if is_selected else 2
        dash   = () if is_selected else (6, 3)

        rect_id = self.create_rectangle(
            dx, dy, dx + dw, dy + dh,
            outline=color, width=line_w, dash=dash,
        )
        label_id = self.create_text(
            dx + 4, dy + 4,
            text=roi_id,
            anchor="nw",
            fill=color,
            font=("Consolas", 9),
        )
        return rect_id, label_id

    def _redraw_all_roi_overlays(self) -> None:
        """Delete and recreate all ROI canvas items using current viewport."""
        for rect_id, label_id in self._roi_items.values():
            self.delete(rect_id)
            self.delete(label_id)
        self._roi_items.clear()

        for roi_id, original_rect in self._roi_original.items():
            color = self._roi_colors.get(roi_id, _ROI_COLORS[0])
            rect_id, label_id = self._create_roi_canvas_items(
                roi_id, original_rect, color
            )
            self._roi_items[roi_id] = (rect_id, label_id)

    def _raise_roi_items(self) -> None:
        """Ensure all ROI overlay items are rendered above the video frame."""
        for rect_id, label_id in self._roi_items.values():
            self.tag_raise(rect_id)
            self.tag_raise(label_id)
        if self._drag_rect_id:
            self.tag_raise(self._drag_rect_id)

    def _highlight_selected_roi(self) -> None:
        """Redraw the selected ROI with thicker border to highlight it."""
        self._redraw_all_roi_overlays()

    def _roi_id_at(self, x: int, y: int) -> Optional[str]:
        """
        Return the roi_id whose display rect contains (x, y), or None.
        Iterates in reverse insertion order so the topmost item wins.
        """
        for roi_id in reversed(list(self._roi_original)):
            ox, oy, ow, oh = self._roi_original[roi_id]
            dx, dy, dw, dh = self._vp.original_to_display_rect(ox, oy, ow, oh)
            if dx <= x <= dx + dw and dy <= y <= dy + dh:
                return roi_id
        return None

    # ─────────────────────────────────────────────────────────────────────
    # Cursor management
    # ─────────────────────────────────────────────────────────────────────

    def _update_cursor(self) -> None:
        if self._roi_draw_enabled:
            self.config(cursor="crosshair")
        elif self._vp.is_zoomed:
            self.config(cursor="hand2")
        else:
            self.config(cursor="arrow")
