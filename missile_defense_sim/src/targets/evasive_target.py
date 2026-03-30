"""
Evasive Target
==============
Like a cruise missile but periodically fires random lateral thrust pulses
to evade interceptors.
"""
from __future__ import annotations

import numpy as np

from src.targets.base_target import BaseTarget


class EvasiveTarget(BaseTarget):
    """
    Cruise-style flight with periodic evasion maneuvers.
    Maneuver: random lateral impulse for a short burst, then resume straight flight.
    """

    def __init__(self, cfg: dict, azimuth_deg: float = 0.0, rng: np.random.Generator | None = None) -> None:
        if rng is None:
            rng = np.random.default_rng()
        self._rng = rng

        t_cfg = cfg["targets"]["evasive"]
        launch_range: float = float(t_cfg["launch_range"])
        self._target_altitude: float = float(t_cfg["altitude"])
        self._target_speed: float = float(t_cfg["speed"])
        self._evasion_interval: float = float(t_cfg["evasion_interval"])
        self._evasion_magnitude: float = float(t_cfg["evasion_magnitude"])

        launch_range *= rng.uniform(0.85, 1.15)
        az = np.radians(azimuth_deg)

        init_pos = np.array([
            launch_range * np.cos(az),
            launch_range * np.sin(az),
            self._target_altitude,
        ])
        horiz = np.array([-np.cos(az), -np.sin(az), 0.0])
        init_vel = self._target_speed * horiz

        self._kp_alt: float = 2.0
        self._kd_alt: float = 1.0
        self._kp_spd: float = 1.5

        self._time_since_last_evasion: float = 0.0
        self._evasion_thrust: np.ndarray = np.zeros(3)
        self._evasion_duration: float = 0.0
        self._evasion_time_remaining: float = 0.0

        super().__init__(init_pos, init_vel, cfg, "evasive")

    def _compute_thrust(self, dt: float) -> np.ndarray:
        # ── Altitude hold ─────────────────────────────────────────────────
        alt_error = self._target_altitude - self.position[2]
        az_thrust = 2.0 * alt_error - 1.0 * self.velocity[2] + 9.81

        # ── Speed hold ────────────────────────────────────────────────────
        horiz_vel = self.velocity[:2]
        horiz_speed = np.linalg.norm(horiz_vel)
        speed_error = self._target_speed - horiz_speed
        if horiz_speed > 1.0:
            thrust_horiz = 1.5 * speed_error * (horiz_vel / horiz_speed)
        else:
            to_origin = -self.position[:2]
            dist = np.linalg.norm(to_origin)
            if dist > 1.0:
                thrust_horiz = 1.5 * self._target_speed * (to_origin / dist)
            else:
                thrust_horiz = np.zeros(2)

        # ── Evasion maneuver ──────────────────────────────────────────────
        self._time_since_last_evasion += dt
        if self._evasion_time_remaining > 0:
            self._evasion_time_remaining -= dt
            evasion = self._evasion_thrust
        else:
            evasion = np.zeros(3)
            if self._time_since_last_evasion >= self._evasion_interval:
                self._trigger_evasion()

        return np.array([
            thrust_horiz[0] + evasion[0],
            thrust_horiz[1] + evasion[1],
            az_thrust + evasion[2],
        ])

    def _trigger_evasion(self) -> None:
        """Pick a random lateral evasion direction and magnitude."""
        self._time_since_last_evasion = 0.0
        # Lateral = perpendicular to current horizontal velocity in XY plane
        horiz_vel = self.velocity[:2]
        if np.linalg.norm(horiz_vel) > 1.0:
            fwd = horiz_vel / np.linalg.norm(horiz_vel)
            lateral = np.array([-fwd[1], fwd[0]])  # 90° rotation
        else:
            angle = self._rng.uniform(0, 2 * np.pi)
            lateral = np.array([np.cos(angle), np.sin(angle)])

        sign = self._rng.choice([-1.0, 1.0])
        magnitude = self._evasion_magnitude * self._rng.uniform(0.7, 1.3)
        self._evasion_thrust = np.array([
            sign * magnitude * lateral[0],
            sign * magnitude * lateral[1],
            0.0,
        ])
        self._evasion_duration = self._rng.uniform(0.5, 1.5)
        self._evasion_time_remaining = self._evasion_duration
