"""
visualization/debug_visualizer.py — Optional debug panels.

Produces side-panels with feature trend plots, temporal history bars,
and ROI crop thumbnails. All rendered with OpenCV (no matplotlib dependency
for the main loop — matplotlib is optional and only used for standalone plots).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from core.data_models import FrameContext, ROIData, TemporalSummary


_FONT  = cv2.FONT_HERSHEY_SIMPLEX
_FSIZE = 0.38
_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)
_GREEN = (0, 200, 0)
_RED   = (0, 0, 220)
_GRAY  = (140, 140, 140)


class DebugVisualizer:
    """
    Builds a debug side-panel image that can be placed next to the main frame.

    Usage::
        panel = DebugVisualizer.build_panel(ctx, width=320)
        combined = np.hstack([main_frame, panel])
    """

    @staticmethod
    def build_panel(ctx: FrameContext, width: int = 300) -> np.ndarray:
        """Return a (H x width) debug panel aligned to ctx's frame height."""
        target_h = ctx.raw_frame.shape[0] if ctx.raw_frame is not None else 480
        rows: List[np.ndarray] = []

        # Header
        rows.append(DebugVisualizer._text_row("=== DEBUG PANEL ===", width, _WHITE))
        rows.append(DebugVisualizer._text_row(
            f"Frame {ctx.frame_index}  inf={ctx.inference_time_ms:.1f}ms", width, _GRAY
        ))

        # Per-ROI info
        for roi in ctx.rois:
            rows.append(DebugVisualizer._roi_summary_rows(roi, width))

        # Pipeline timings
        if ctx.pipeline_times_ms:
            rows.append(DebugVisualizer._text_row("--- Timings ---", width, _GRAY))
            for k, v in ctx.pipeline_times_ms.items():
                rows.append(DebugVisualizer._text_row(f"  {k}: {v:.1f}ms", width, _WHITE))

        # Warnings
        for w_msg in ctx.warnings:
            rows.append(DebugVisualizer._text_row(f"WARN: {w_msg}", width, (0, 200, 255)))

        # Stack rows
        if rows:
            panel = np.vstack(rows)
        else:
            panel = np.zeros((target_h, width, 3), dtype=np.uint8)

        # Pad / crop to target height
        ph = panel.shape[0]
        if ph < target_h:
            pad = np.zeros((target_h - ph, width, 3), dtype=np.uint8)
            panel = np.vstack([panel, pad])
        else:
            panel = panel[:target_h]

        # Divider line
        panel[:, 0] = (80, 80, 80)
        return panel

    # ── Feature trend bar chart ───────────────────────────────────────────

    @staticmethod
    def draw_trend_bar(
        history: List[float],
        width: int = 300,
        height: int = 60,
        color: Tuple[int, int, int] = _GREEN,
        label: str = "",
    ) -> np.ndarray:
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        if not history:
            return canvas

        n = len(history)
        lo, hi = min(history), max(history)
        span = hi - lo if hi != lo else 1.0

        bar_w = max(1, width // n)
        for i, val in enumerate(history):
            norm = (val - lo) / span
            bar_h = int(norm * (height - 8))
            x = i * bar_w
            cv2.rectangle(canvas, (x, height - bar_h), (x + bar_w - 1, height), color, -1)

        if label:
            cv2.putText(canvas, label, (2, 12), _FONT, _FSIZE, _WHITE, 1)

        return canvas

    # ── Internal helpers ──────────────────────────────────────────────────

    @staticmethod
    def _text_row(text: str, width: int, color: Tuple[int, int, int]) -> np.ndarray:
        row = np.zeros((16, width, 3), dtype=np.uint8)
        cv2.putText(row, text[:60], (2, 12), _FONT, _FSIZE, color, 1)
        return row

    @staticmethod
    def _roi_summary_rows(roi: ROIData, width: int) -> np.ndarray:
        rows = [
            DebugVisualizer._text_row(f"ROI: {roi.roi_id} → {roi.roi_state.value}", width, _WHITE),
        ]
        for key in ("mean_intensity", "std_intensity", "mean_abs_diff", "edge_density"):
            val = roi.features.get(key)
            if val is not None:
                rows.append(DebugVisualizer._text_row(f"  {key}: {val:.3f}", width, _GRAY))

        # Temporal flags
        ts = roi.temporal_summary
        if ts:
            flags = []
            if ts.is_stuck:       flags.append("STUCK")
            if ts.is_drifting:    flags.append("DRIFT")
            if ts.is_oscillating: flags.append("OSCIL")
            if ts.has_sudden_change: flags.append("SUDDEN")
            if flags:
                rows.append(DebugVisualizer._text_row(f"  FLAGS: {' '.join(flags)}", width, _RED))

            # Mini trend bar
            if ts.mean_intensity_history:
                bar = DebugVisualizer.draw_trend_bar(
                    ts.mean_intensity_history[-width:], width=width, height=30, label="mean_int"
                )
                rows.append(bar)

        return np.vstack(rows)


# ---------------------------------------------------------------------------
# Optional matplotlib-based plots (called from external scripts, not the loop)
# ---------------------------------------------------------------------------

def plot_feature_history(
    history: List[float],
    title: str = "Feature History",
    save_path: Optional[str] = None,
) -> None:
    """
    Render a matplotlib line plot for offline analysis.
    Only call from scripts, not from the real-time pipeline.
    """
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(history, linewidth=1.2)
        ax.set_title(title)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)
        if save_path:
            fig.savefig(save_path, dpi=100, bbox_inches="tight")
        else:
            plt.show()
        plt.close(fig)
    except ImportError:
        print("matplotlib not installed — cannot plot feature history")
