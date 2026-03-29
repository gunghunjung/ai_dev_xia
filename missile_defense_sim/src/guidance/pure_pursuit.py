"""
Pure Pursuit Guidance
=====================
The interceptor continuously steers its velocity vector toward the current
target position.  Simpler than PN but less energy-efficient against crossing
targets; useful as a baseline for guidance comparison.

Implementation
--------------
1. Compute heading error angle (angle between current velocity and LOS vector).
2. Compute required angular rate to close the error at gain * error_rad/s.
3. Lateral acceleration = v_missile * angular_rate  (perpendicular to velocity, toward target).
4. Cap lateral acceleration to max_lateral to avoid saturating G-limit.
5. Add explicit gravity compensation so control authority is preserved.
"""
from __future__ import annotations

import numpy as np

from src.guidance.base_guidance import BaseGuidance


class PurePursuit(BaseGuidance):

    def __init__(
        self,
        gain: float = 3.0,
        max_lateral_accel: float = 200.0,
    ) -> None:
        """
        Parameters
        ----------
        gain             : heading-rate gain (rad/s per radian of error)
        max_lateral_accel: cap on the lateral steering acceleration (m/s²)
                           keeps the G-budget available for gravity comp
        """
        self.gain = float(gain)
        self.max_lateral = float(max_lateral_accel)
        self._G_UP = np.array([0.0, 0.0, 9.81])   # gravity compensation vector

    def compute_command(
        self,
        int_pos: np.ndarray,
        int_vel: np.ndarray,
        tgt_pos: np.ndarray,
        tgt_vel: np.ndarray,
    ) -> np.ndarray:
        R = tgt_pos - int_pos
        R_norm = np.linalg.norm(R)
        if R_norm < 1.0:
            return self._G_UP.copy()

        desired_dir = R / R_norm                 # unit LOS vector

        v_norm = np.linalg.norm(int_vel)
        if v_norm < 1.0:
            # Stationary: boost directly toward target
            return self.max_lateral * desired_dir + self._G_UP

        current_dir = int_vel / v_norm

        # Axis of rotation (cross product → axis ⊥ both dirs)
        axis = np.cross(current_dir, desired_dir)
        sin_angle = np.linalg.norm(axis)

        if sin_angle < 1e-9:
            # Already aligned — only gravity comp needed
            return self._G_UP.copy()

        axis_hat = axis / sin_angle

        # Required angular rate (proportional to heading error)
        omega = self.gain * sin_angle  # rad/s

        # Lateral acceleration vector (perpendicular to velocity, toward target)
        # a_lat = v * omega in the direction perpendicular to v toward desired
        a_lat_dir = np.cross(axis_hat, current_dir)   # unit vec ⊥ v, toward target
        a_lat_mag = min(v_norm * omega, self.max_lateral)
        a_lateral = a_lat_mag * a_lat_dir

        # Gravity compensation (keeps interceptor from falling)
        return a_lateral + self._G_UP
