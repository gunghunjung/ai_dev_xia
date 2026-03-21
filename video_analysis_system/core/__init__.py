"""core — shared data models, interfaces, coordinate transforms, and viewport management."""

from core.data_models import (
    DetectionResult,
    EventRecord,
    FrameContext,
    ROIData,
    RuleResult,
    TemporalSummary,
    TrackingInfo,
)
from core.states import ROIState, SystemState
from core.coordinate_transform import CoordinateTransform, CoordinateTransformManager
from core.viewport_manager import ViewportManager, ViewportState

__all__ = [
    # data models
    "DetectionResult", "EventRecord", "FrameContext",
    "ROIData", "RuleResult", "TemporalSummary", "TrackingInfo",
    # states
    "ROIState", "SystemState",
    # coordinate / viewport
    "CoordinateTransform", "CoordinateTransformManager",
    "ViewportManager", "ViewportState",
]
