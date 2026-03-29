"""Guidance base class."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseGuidance(ABC):
    """
    All guidance laws implement compute_command().
    Returns a commanded acceleration vector (m/s²).
    Gravity compensation is NOT applied here — the physics engine handles gravity.
    """

    @abstractmethod
    def compute_command(
        self,
        int_pos: np.ndarray,
        int_vel: np.ndarray,
        tgt_pos: np.ndarray,
        tgt_vel: np.ndarray,
    ) -> np.ndarray:
        """Return commanded acceleration (m/s²) for the interceptor."""

    @staticmethod
    def _safe_norm(v: np.ndarray, eps: float = 1e-9) -> float:
        return float(np.linalg.norm(v)) + eps
