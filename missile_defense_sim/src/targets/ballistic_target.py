"""
Ballistic Target
================
Re-entry vehicle / ballistic missile in terminal phase.
No thrust — purely governed by gravity and aerodynamic drag.
"""
from __future__ import annotations

import numpy as np

from src.targets.base_target import BaseTarget


class BallisticTarget(BaseTarget):
    """
    Starts at (launch_range, 0, launch_altitude), aimed at the battery (origin).
    After launch it is fully ballistic: zero thrust.
    """

    def __init__(self, cfg: dict, azimuth_deg: float = 0.0, rng: np.random.Generator | None = None) -> None:
        if rng is None:
            rng = np.random.default_rng()

        t_cfg = cfg["targets"]["ballistic"]
        launch_range: float = float(t_cfg["launch_range"])
        launch_alt: float = float(t_cfg["launch_altitude"])
        speed: float = float(t_cfg["speed"])

        # Vary parameters slightly for Monte Carlo realism
        launch_range *= rng.uniform(0.85, 1.15)
        launch_alt   *= rng.uniform(0.80, 1.20)
        speed        *= rng.uniform(0.90, 1.10)

        az = np.radians(azimuth_deg)
        init_pos = np.array([
            launch_range * np.cos(az),
            launch_range * np.sin(az),
            launch_alt,
        ])
        # Aim directly at the battery (origin)
        direction = -init_pos / np.linalg.norm(init_pos)
        init_vel = speed * direction

        super().__init__(init_pos, init_vel, cfg, "ballistic")

    def _compute_thrust(self, dt: float) -> np.ndarray:
        """No thrust — ballistic."""
        return np.zeros(3)
