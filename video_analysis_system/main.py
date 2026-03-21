"""
main.py — Entry point for the Video Analysis System.

Default: launches the Tkinter GUI.

Usage examples:

  # GUI (default)
  python main.py

  # CLI / headless mode (no GUI window, OpenCV display only)
  python main.py --no-gui --source file --path sample.mp4 --debug

  # Live camera via GUI
  python main.py

Press 'q' in the OpenCV window (--no-gui mode) to exit.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path when run directly
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    AIConfig,
    DecisionConfig,
    LoggingConfig,
    PreprocessConfig,
    ROIConfig,
    SystemConfig,
    TemporalConfig,
    VideoConfig,
    VisualizationConfig,
)
from core.engine_core import EngineCore


# ---------------------------------------------------------------------------
# Config builder — edit this section to customise defaults
# ---------------------------------------------------------------------------

def build_default_config(args: argparse.Namespace) -> SystemConfig:
    """Construct a SystemConfig from command-line args + sensible defaults."""

    cfg = SystemConfig()

    # Video source
    cfg.video = VideoConfig(
        source_type=args.source,
        source_path=args.path if args.path else "",
        camera_index=args.camera,
        target_fps=args.fps,
        loop=args.loop,
    )

    # ROIs — define your regions here or override via --roi x,y,w,h
    if args.roi:
        for i, roi_str in enumerate(args.roi):
            parts = list(map(int, roi_str.split(",")))
            if len(parts) == 4:
                cfg.rois.append(ROIConfig(
                    roi_id=f"roi_{i}",
                    bbox=tuple(parts),      # type: ignore[arg-type]
                    label=f"ROI {i}",
                ))
    else:
        # Default demo ROI (top-left quadrant)
        cfg.rois = [
            ROIConfig(roi_id="roi_0", bbox=(50, 50, 200, 150), label="Zone A"),
            ROIConfig(roi_id="roi_1", bbox=(300, 100, 180, 120), label="Zone B"),
        ]

    # Preprocessing
    cfg.preprocess = PreprocessConfig(
        resize=None,        # e.g. (640, 480) to normalise input size
        denoise=False,
        grayscale=False,
    )

    # Temporal analysis
    cfg.temporal = TemporalConfig(window_size=60, min_frames_for_decision=10)

    # AI
    cfg.ai = AIConfig(
        model_type="placeholder",    # swap to "onnx" or "pytorch" for real models
        model_path=None,
        confidence_threshold=0.5,
        class_names=["normal", "abnormal"],
    )

    # Decision thresholds
    cfg.decision = DecisionConfig(
        abnormal_score_threshold=0.70,
        warning_score_threshold=0.40,
        consecutive_abnormal_frames=5,
        consecutive_normal_frames=10,
        sudden_change_threshold=40.0,
        stuck_variance_threshold=1.5,
        stuck_min_frames=20,
        drift_threshold=25.0,
    )

    # Visualisation
    cfg.visualization = VisualizationConfig(
        window_name="Video Analysis System",
        show_roi_boxes=True,
        show_labels=True,
        show_confidence=True,
        show_state_banner=True,
        show_frame_info=True,
        show_debug_overlay=args.debug,
        wait_key_ms=1,
    )

    # Logging
    cfg.logging = LoggingConfig(
        log_dir=args.log_dir,
        save_event_frames=True,
        pre_event_frames=30,
        post_event_frames=30,
        export_csv=True,
        export_json=True,
    )

    cfg.max_frames = args.max_frames
    cfg.multithreaded = args.threaded

    return cfg


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Modular Video Analysis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--no-gui", action="store_true", dest="no_gui",
                        help="Run in headless CLI mode (OpenCV window only)")
    parser.add_argument("--source", choices=["file", "camera", "images"],
                        default="file", help="Frame source type")
    parser.add_argument("--path", default="", help="Path to video file or image directory")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index")
    parser.add_argument("--fps", type=float, default=30.0, help="Target FPS")
    parser.add_argument("--loop", action="store_true", help="Loop video file")
    parser.add_argument("--roi", nargs="*", metavar="x,y,w,h",
                        help="One or more ROI bounding boxes e.g. 50,50,200,150")
    parser.add_argument("--debug", action="store_true", help="Show debug overlay panel")
    parser.add_argument("--threaded", action="store_true", help="Use multithreaded pipeline")
    parser.add_argument("--max-frames", type=int, default=None, dest="max_frames",
                        help="Stop after N frames")
    parser.add_argument("--log-dir", default="logs", dest="log_dir",
                        help="Output directory for logs and events")
    parser.add_argument("--advanced", action="store_true",
                        help="GUI 고급 모드 (AI 관리 탭 포함 MainWindow)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if not args.no_gui:
        # ── GUI mode (default) ────────────────────────────────────────────
        # --advanced 플래그 지정 시 AI 관리 탭이 포함된 MainWindow 실행
        if getattr(args, "advanced", False):
            from gui.main_window import MainWindow
            MainWindow().mainloop()
        else:
            from gui.app import App
            App().mainloop()
        return

    # ── Headless CLI mode ─────────────────────────────────────────────────
    cfg = build_default_config(args)
    engine = EngineCore(cfg)
    try:
        engine.initialize()
        engine.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as exc:
        print(f"Fatal error: {exc}", file=sys.stderr)
        raise
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
