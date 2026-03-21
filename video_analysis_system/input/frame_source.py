"""
input/frame_source.py — Concrete FrameSource implementations.

Supported sources:
  - VideoFileSource    : read from a .mp4 / .avi / ... file
  - CameraSource       : read from a live USB/IP camera
  - ImageSequenceSource: read from a sorted directory of images
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from core.interfaces import IFrameSource


# ---------------------------------------------------------------------------
# VideoFileSource
# ---------------------------------------------------------------------------

class VideoFileSource(IFrameSource):
    """Reads frames from a video file via OpenCV VideoCapture."""

    def __init__(self, path: str, loop: bool = False):
        self._path = path
        self._loop = loop
        self._cap: Optional[cv2.VideoCapture] = None
        self._fps: float = 30.0
        self._frame_count: int = -1
        self._frame_size: tuple[int, int] = (0, 0)

    def open(self) -> bool:
        self._cap = cv2.VideoCapture(self._path)
        if not self._cap.isOpened():
            return False
        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._frame_size = (w, h)
        return True

    def read_frame(self) -> tuple[bool, Optional[np.ndarray]]:
        if self._cap is None:
            return False, None
        ok, frame = self._cap.read()
        if not ok:
            if self._loop:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self._cap.read()
            if not ok:
                return False, None
        return True, frame

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def frame_size(self) -> tuple[int, int]:
        return self._frame_size


# ---------------------------------------------------------------------------
# CameraSource
# ---------------------------------------------------------------------------

class CameraSource(IFrameSource):
    """Reads frames from a live camera device."""

    def __init__(self, index: int = 0, target_fps: float = 30.0):
        self._index = index
        self._target_fps = target_fps
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_size: tuple[int, int] = (0, 0)

    def open(self) -> bool:
        self._cap = cv2.VideoCapture(self._index)
        if not self._cap.isOpened():
            return False
        self._cap.set(cv2.CAP_PROP_FPS, self._target_fps)
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._frame_size = (w, h)
        return True

    def read_frame(self) -> tuple[bool, Optional[np.ndarray]]:
        if self._cap is None:
            return False, None
        return self._cap.read()

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    @property
    def fps(self) -> float:
        return self._target_fps

    @property
    def frame_count(self) -> int:
        return -1  # live source has no total count

    @property
    def frame_size(self) -> tuple[int, int]:
        return self._frame_size


# ---------------------------------------------------------------------------
# ImageSequenceSource
# ---------------------------------------------------------------------------

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


class ImageSequenceSource(IFrameSource):
    """Reads frames from a directory of sorted image files."""

    def __init__(self, directory: str, fps: float = 10.0, loop: bool = False):
        self._directory = Path(directory)
        self._fps = fps
        self._loop = loop
        self._paths: list[Path] = []
        self._index: int = 0
        self._frame_size: tuple[int, int] = (0, 0)

    def open(self) -> bool:
        if not self._directory.is_dir():
            return False
        self._paths = sorted(
            p for p in self._directory.iterdir()
            if p.suffix.lower() in _IMAGE_EXTS
        )
        if not self._paths:
            return False
        # Peek first image to get size
        sample = cv2.imread(str(self._paths[0]))
        if sample is None:
            return False
        h, w = sample.shape[:2]
        self._frame_size = (w, h)
        self._index = 0
        return True

    def read_frame(self) -> tuple[bool, Optional[np.ndarray]]:
        if self._index >= len(self._paths):
            if self._loop:
                self._index = 0
            else:
                return False, None
        frame = cv2.imread(str(self._paths[self._index]))
        self._index += 1
        if frame is None:
            return False, None
        return True, frame

    def release(self) -> None:
        self._paths = []
        self._index = 0

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_count(self) -> int:
        return len(self._paths)

    @property
    def frame_size(self) -> tuple[int, int]:
        return self._frame_size


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_frame_source(
    source_type: str,
    source_path: str = "",
    camera_index: int = 0,
    fps: float = 30.0,
    loop: bool = False,
) -> IFrameSource:
    """Return the correct IFrameSource implementation based on source_type."""
    if source_type == "file":
        return VideoFileSource(source_path, loop=loop)
    elif source_type == "camera":
        return CameraSource(camera_index, target_fps=fps)
    elif source_type == "image_sequence":
        return ImageSequenceSource(source_path, fps=fps, loop=loop)
    else:
        raise ValueError(f"Unknown source_type: {source_type!r}")
