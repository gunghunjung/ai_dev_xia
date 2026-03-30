"""
6-DOF Kalman Filter for target tracking.
State vector: [x, y, z, vx, vy, vz]
Measurement:  [x, y, z]  (position only)
Model:        constant-velocity with zero-mean process acceleration noise.
"""
from __future__ import annotations

import numpy as np


class KalmanFilter6D:
    """
    Standard linear Kalman Filter for 3-D position+velocity estimation.

    Parameters
    ----------
    dt                : nominal update interval (s)
    process_noise     : scalar Q — tuning parameter for maneuver uncertainty
    measurement_noise : scalar R — 1-sigma position measurement noise (m)
    """

    def __init__(
        self,
        dt: float,
        process_noise: float = 20.0,
        measurement_noise: float = 50.0,
    ) -> None:
        self.dt = dt
        n, m = 6, 3

        # ── State transition matrix (constant-velocity model) ─────────────
        self.F = np.eye(n)
        self.F[0, 3] = self.F[1, 4] = self.F[2, 5] = dt

        # ── Measurement matrix (observe position only) ────────────────────
        self.H = np.zeros((m, n))
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = 1.0

        # ── Process noise covariance (Singer model approximation) ─────────
        dt2 = dt ** 2
        dt3 = dt ** 3
        q = process_noise
        Qpos = (dt3 / 3.0) * q
        Qcross = (dt2 / 2.0) * q
        Qvel = dt * q
        self.Q = np.array([
            [Qpos, 0, 0, Qcross, 0, 0],
            [0, Qpos, 0, 0, Qcross, 0],
            [0, 0, Qpos, 0, 0, Qcross],
            [Qcross, 0, 0, Qvel, 0, 0],
            [0, Qcross, 0, 0, Qvel, 0],
            [0, 0, Qcross, 0, 0, Qvel],
        ])

        # ── Measurement noise covariance ──────────────────────────────────
        r = measurement_noise ** 2
        self.R = np.eye(m) * r

        # ── State and covariance ──────────────────────────────────────────
        self.x = np.zeros(n)
        self.P = np.eye(n) * 1e6
        self.initialized = False

    # ── API ────────────────────────────────────────────────────────────────
    def initialize(self, position: np.ndarray, velocity: np.ndarray | None = None) -> None:
        self.x[:3] = position
        self.x[3:] = velocity if velocity is not None else np.zeros(3)
        self.P = np.eye(6) * 1e4
        self.initialized = True

    def predict(self, dt: float | None = None) -> None:
        """Time-update (propagate state forward by dt)."""
        if dt is not None and dt != self.dt:
            # Rebuild F for new dt
            F = np.eye(6)
            F[0, 3] = F[1, 4] = F[2, 5] = dt
        else:
            F = self.F
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q

    def update(self, measurement: np.ndarray) -> None:
        """Measurement-update with a position observation."""
        y = measurement - self.H @ self.x          # innovation
        S = self.H @ self.P @ self.H.T + self.R    # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        self.x = self.x + K @ y
        I_KH = np.eye(6) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T  # Joseph form

    def step(self, measurement: np.ndarray, dt: float | None = None) -> np.ndarray:
        """Convenience: predict then update, return estimated state."""
        self.predict(dt)
        self.update(measurement)
        return self.x.copy()

    @property
    def position(self) -> np.ndarray:
        return self.x[:3].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.x[3:].copy()

    @property
    def covariance_position(self) -> np.ndarray:
        return self.P[:3, :3].copy()
