"""
core/interfaces.py — Abstract base classes that define module contracts.

All pluggable components must inherit from the appropriate ABC here.
This keeps modules loosely coupled while enforcing clear interfaces.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Frame Source
# ---------------------------------------------------------------------------

class IFrameSource(ABC):
    """Contract for any frame acquisition backend."""

    @abstractmethod
    def open(self) -> bool:
        """Open the source. Returns True on success."""

    @abstractmethod
    def read_frame(self) -> tuple[bool, Optional[np.ndarray]]:
        """Read the next frame. Returns (success, frame)."""

    @abstractmethod
    def release(self) -> None:
        """Release underlying resources."""

    @property
    @abstractmethod
    def fps(self) -> float:
        """Reported FPS of the source."""

    @property
    @abstractmethod
    def frame_count(self) -> int:
        """Total frame count (-1 if unknown)."""

    @property
    @abstractmethod
    def frame_size(self) -> tuple[int, int]:
        """(width, height) of frames."""


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------

class IPreprocessStep(ABC):
    """A single pluggable preprocessing step."""

    @abstractmethod
    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Transform frame and return result."""


# ---------------------------------------------------------------------------
# Feature Extractor
# ---------------------------------------------------------------------------

class IFeatureExtractor(ABC):
    """Extracts a feature dictionary from an image region."""

    @abstractmethod
    def extract(self, region: np.ndarray, prev_region: Optional[np.ndarray] = None) -> dict:
        """Return feature dict from the supplied image region."""


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class ITracker(ABC):
    """Tracks an object region across frames."""

    @abstractmethod
    def init(self, frame: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
        """Initialize tracker with first frame and bounding box."""

    @abstractmethod
    def update(self, frame: np.ndarray) -> tuple[bool, tuple[int, int, int, int]]:
        """Update tracker. Returns (success, new_bbox)."""

    @abstractmethod
    def reset(self) -> None:
        """Reset tracker state."""


# ---------------------------------------------------------------------------
# AI Model
# ---------------------------------------------------------------------------

class IModel(ABC):
    """AI model backend contract."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model weights from path."""

    @abstractmethod
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """Run forward pass. Returns raw output array."""

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """True when model is ready for inference."""


# ---------------------------------------------------------------------------
# Decision Rule
# ---------------------------------------------------------------------------

class IDecisionRule(ABC):
    """A single pluggable validation / decision rule."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable rule name."""

    @abstractmethod
    def evaluate(self, context: "FrameContext") -> Optional["RuleResult"]:  # type: ignore[name-defined]
        """Evaluate rule against the current FrameContext. Return result or None."""


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

class IRenderer(ABC):
    """Visualisation output contract."""

    @abstractmethod
    def render(self, context: "FrameContext") -> np.ndarray:  # type: ignore[name-defined]
        """Render a FrameContext to a displayable image."""

    @abstractmethod
    def show(self, frame: np.ndarray) -> int:
        """Display the rendered frame. Returns cv2.waitKey value."""
