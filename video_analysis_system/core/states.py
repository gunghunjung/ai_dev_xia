"""
core/states.py — All state enumerations used throughout the system.
"""

from __future__ import annotations
from enum import Enum


class SystemState(Enum):
    """Top-level system state emitted by DecisionEngine."""
    INITIALIZING = "INITIALIZING"
    NORMAL       = "NORMAL"
    WARNING      = "WARNING"
    ABNORMAL     = "ABNORMAL"
    TRACKING_LOST = "TRACKING_LOST"
    UNKNOWN      = "UNKNOWN"


class ROIState(Enum):
    """Per-ROI state computed from temporal and AI analysis."""
    NORMAL        = "NORMAL"
    WARNING       = "WARNING"
    ABNORMAL      = "ABNORMAL"
    STUCK         = "STUCK"
    DRIFTING      = "DRIFTING"
    OSCILLATING   = "OSCILLATING"
    SUDDEN_CHANGE = "SUDDEN_CHANGE"
    UNKNOWN       = "UNKNOWN"


class AbnormalityType(Enum):
    """Categorised reason for an abnormal determination."""
    STUCK            = "stuck"
    DRIFT            = "drift"
    SUDDEN_CHANGE    = "sudden_change"
    OSCILLATION      = "oscillation"
    DELAYED_RESPONSE = "delayed_response"
    MISSING_MOTION   = "missing_motion"
    AI_DETECTION     = "ai_detection"


# Colour map for visualisation (BGR)
STATE_COLORS: dict[SystemState, tuple[int, int, int]] = {
    SystemState.NORMAL:        (0, 200, 0),
    SystemState.WARNING:       (0, 200, 255),
    SystemState.ABNORMAL:      (0, 0, 220),
    SystemState.TRACKING_LOST: (128, 0, 128),
    SystemState.UNKNOWN:       (180, 180, 180),
    SystemState.INITIALIZING:  (200, 200, 0),
}

ROI_STATE_COLORS: dict[ROIState, tuple[int, int, int]] = {
    ROIState.NORMAL:        (0, 200, 0),
    ROIState.WARNING:       (0, 200, 255),
    ROIState.ABNORMAL:      (0, 0, 220),
    ROIState.STUCK:         (255, 100, 0),
    ROIState.DRIFTING:      (0, 165, 255),
    ROIState.OSCILLATING:   (128, 0, 255),
    ROIState.SUDDEN_CHANGE: (0, 0, 255),
    ROIState.UNKNOWN:       (180, 180, 180),
}
