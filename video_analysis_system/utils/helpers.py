"""
utils/helpers.py — Shared utility functions used across modules.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

@contextmanager
def timer(label: str, out_dict: Optional[dict] = None) -> Generator[None, None, None]:
    """Context manager that measures elapsed ms and optionally stores in dict."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if out_dict is not None:
            out_dict[label] = elapsed_ms


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

def crop_roi(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> Optional[np.ndarray]:
    """Safely crop (x, y, w, h) from frame. Returns None if out of bounds."""
    x, y, w, h = bbox
    fh, fw = frame.shape[:2]
    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + w, fw), min(y + h, fh)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2].copy()


def safe_resize(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize frame to (width, height)."""
    return cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR)


def ensure_bgr(frame: np.ndarray) -> np.ndarray:
    """Convert grayscale to BGR if needed."""
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return frame


def ensure_uint8(frame: np.ndarray) -> np.ndarray:
    """Clamp and convert float frame to uint8."""
    if frame.dtype != np.uint8:
        frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
    return frame


def compute_histogram(region: np.ndarray, bins: int = 16) -> np.ndarray:
    """Return normalised grayscale histogram as 1-D float32 array."""
    if region.ndim == 3:
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    else:
        gray = region
    hist = cv2.calcHist([gray], [0], None, [bins], [0, 256])
    hist = hist.flatten().astype(np.float32)
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def compute_edge_density(region: np.ndarray) -> float:
    """Fraction of pixels with strong edges (Canny)."""
    if region.ndim == 3:
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    else:
        gray = region.copy()
    edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
    total = edges.size
    return float(edges.sum() / 255) / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# File/path utilities
# ---------------------------------------------------------------------------

def ensure_dir(path: str | Path) -> Path:
    """Create directory (and parents) if it doesn't exist. Return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def unique_filename(base_dir: str | Path, stem: str, suffix: str) -> Path:
    """Return a path that doesn't collide with existing files."""
    base = Path(base_dir)
    candidate = base / f"{stem}{suffix}"
    counter = 1
    while candidate.exists():
        candidate = base / f"{stem}_{counter}{suffix}"
        counter += 1
    return candidate


# ---------------------------------------------------------------------------
# Maths
# ---------------------------------------------------------------------------

def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b != 0 else default


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def linear_regression_slope(values: list[float]) -> float:
    """Simple least-squares slope over equally-spaced x."""
    n = len(values)
    if n < 2:
        return 0.0
    x = np.arange(n, dtype=np.float64)
    y = np.array(values, dtype=np.float64)
    mx, my = x.mean(), y.mean()
    denom = ((x - mx) ** 2).sum()
    if denom == 0:
        return 0.0
    return float(((x - mx) * (y - my)).sum() / denom)


def count_zero_crossings(values: list[float]) -> int:
    """Count sign changes in a list of floats (for oscillation detection)."""
    if len(values) < 2:
        return 0
    crossings = 0
    mean = sum(values) / len(values)
    centered = [v - mean for v in values]
    for i in range(1, len(centered)):
        if centered[i - 1] * centered[i] < 0:
            crossings += 1
    return crossings
