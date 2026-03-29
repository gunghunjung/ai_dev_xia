"""
Cruise Target
=============
Low-altitude, constant-speed cruise missile.
Uses proportional control to maintain cruise altitude and desired speed.
Flies directly toward the battery (origin).
"""
from __future__ import annotations

import numpy as np

from src.targets.base_target import BaseTarget


class CruiseTarget(BaseTarget):
    """
    Flies at constant altitude and speed toward the origin.
    A simple altitude-hold and speed-hold feedback loop supplies the thrust.
    """

    def __init__(self, cfg: dict, azimuth_deg: float = 0.0, rng: np.random.Generator | None = None) -> None:
        if rng is None:
            rng = np.random.default_rng()

        t_cfg = cfg["targets"]["cruise"]
        launch_range: float = float(t_cfg["launch_range"])
        self._target_altitude: float = float(t_cfg["altitude"])
        self._target_speed: float = float(t_cfg["speed"])

        launch_range *= rng.uniform(0.85, 1.15)
        az = np.radians(azimuth_deg)

        init_pos = np.array([
            launch_range * np.cos(az),
            launch_range * np.sin(az),
            self._target_altitude,
        ])
        # Initial velocity: horizontal, toward origin at cruise speed
        horiz = np.array([-np.cos(az), -np.sin(az), 0.0])
        init_vel = self._target_speed * horiz

        # Altitude-hold and speed-hold gains
        self._kp_alt: float = 2.0    # proportional gain for altitude
        self._kd_alt: float = 1.0    # derivative (velocity z) gain
        self._kp_spd: float = 1.5    # speed error gain

        super().__init__(init_pos, init_vel, cfg, "cruise")

    def _compute_thrust(self, dt: float) -> np.ndarray:
        # ── Altitude hold ─────────────────────────────────────────────────
        alt_error = self._target_altitude - self.position[2]
        az_thrust = self._kp_alt * alt_error - self._kd_alt * self.velocity[2]
        # Counteract gravity
        az_thrust += 9.81

        # ── Speed hold (horizontal component) ─────────────────────────────
        horiz_vel = self.velocity[:2]
        horiz_speed = np.linalg.norm(horiz_vel)
        speed_error = self._target_speed - horiz_speed
        if horiz_speed > 1.0:
            thrust_horiz = self._kp_spd * speed_error * (horiz_vel / horiz_speed)
        else:
            # Aim toward origin if nearly stationary
            to_origin = -self.position[:2]
            dist = np.linalg.norm(to_origin)
            if dist > 1.0:
                thrust_horiz = self._kp_spd * self._target_speed * (to_origin / dist)
            else:
                thrust_horiz = np.zeros(2)

        return np.array([thrust_horiz[0], thrust_horiz[1], az_thrust])
