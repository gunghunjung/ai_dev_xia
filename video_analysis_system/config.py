"""
config.py — System-wide configuration dataclasses.
All tunable parameters live here. Modules read config at init time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class VideoConfig:
    """Frame acquisition settings."""
    source_type: str = "file"          # "file" | "camera" | "image_sequence"
    source_path: str = ""              # path to file/directory
    camera_index: int = 0
    target_fps: float = 30.0
    loop: bool = False                 # loop video file


@dataclass
class ROIConfig:
    """Definition of a single Region of Interest."""
    roi_id: str = "roi_0"
    bbox: Tuple[int, int, int, int] = (0, 0, 100, 100)   # (x, y, w, h)
    enabled: bool = True
    label: str = ""


@dataclass
class PreprocessConfig:
    """Frame preprocessing pipeline flags."""
    resize: Optional[Tuple[int, int]] = None   # (width, height) or None
    grayscale: bool = False
    normalize: bool = False            # 0-1 float normalisation
    denoise: bool = False              # Gaussian blur
    equalize_hist: bool = False
    apply_mask: bool = False
    mask_path: str = ""


@dataclass
class TemporalConfig:
    """Sliding-window temporal analysis settings."""
    window_size: int = 60             # frames kept in buffer
    min_frames_for_decision: int = 10  # warm-up frames before decisions


@dataclass
class AIConfig:
    """AI model backend settings."""
    model_type: str = "placeholder"   # "onnx" | "pytorch" | "placeholder"
    model_path: Optional[str] = None
    input_size: Tuple[int, int] = (224, 224)
    confidence_threshold: float = 0.5
    device: str = "cpu"
    class_names: List[str] = field(default_factory=lambda: ["normal", "abnormal"])


@dataclass
class DecisionConfig:
    """Rule-based decision / thresholding settings."""
    abnormal_score_threshold: float = 0.70
    warning_score_threshold: float = 0.40
    consecutive_abnormal_frames: int = 5   # N frames above threshold → ABNORMAL
    consecutive_normal_frames: int = 10    # N frames below threshold → NORMAL
    sudden_change_threshold: float = 40.0  # intensity delta
    stuck_variance_threshold: float = 1.5  # very low variance → STUCK
    stuck_min_frames: int = 25
    drift_threshold: float = 25.0         # gradual mean shift
    oscillation_threshold: float = 15.0
    oscillation_min_cycles: int = 3


@dataclass
class VisualizationConfig:
    """Rendering and overlay toggles."""
    window_name: str = "Video Analysis System"
    show_roi_boxes: bool = True
    show_labels: bool = True
    show_confidence: bool = True
    show_state_banner: bool = True
    show_frame_info: bool = True
    show_debug_overlay: bool = False
    show_feature_trend: bool = False
    scale_display: float = 1.0         # display scale factor
    wait_key_ms: int = 1               # cv2.waitKey delay


@dataclass
class LoggingConfig:
    """Logging and event recording settings."""
    log_dir: str = "logs"
    save_event_frames: bool = True
    pre_event_frames: int = 30
    post_event_frames: int = 30
    export_csv: bool = True
    export_json: bool = True
    log_every_n_frames: int = 1        # 1 = every frame


@dataclass
class SystemConfig:
    """Root config — passed to EngineCore at startup."""
    video: VideoConfig = field(default_factory=VideoConfig)
    rois: List[ROIConfig] = field(default_factory=list)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    ai: AIConfig = field(default_factory=AIConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    multithreaded: bool = False
    max_frames: Optional[int] = None   # None = unlimited


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def default_config_with_file(video_path: str, rois: Optional[List[ROIConfig]] = None) -> SystemConfig:
    """Convenience factory for file-based analysis."""
    cfg = SystemConfig()
    cfg.video.source_type = "file"
    cfg.video.source_path = video_path
    if rois:
        cfg.rois = rois
    return cfg


def default_config_with_camera(camera_index: int = 0, rois: Optional[List[ROIConfig]] = None) -> SystemConfig:
    """Convenience factory for live camera analysis."""
    cfg = SystemConfig()
    cfg.video.source_type = "camera"
    cfg.video.camera_index = camera_index
    if rois:
        cfg.rois = rois
    return cfg
