"""
Physics Engine
==============
3D rigid-body motion with gravity, quadratic drag, and G-limit clamping.
All units: SI (m, s, kg, N, rad).
"""
from __future__ import annotations

import numpy as np
from typing import Optional, Tuple

# ── Constants ────────────────────────────────────────────────────────────────
GRAVITY = np.array([0.0, 0.0, -9.81], dtype=np.float64)   # m/s²
G_SCALAR = 9.81                                             # m/s²


class PhysicsEngine:
    """
    Stateless helper.  Each entity (target / interceptor) owns its own
    position / velocity arrays and calls integrate() each tick.
    """

    def __init__(self, enable_gravity: bool = True, enable_drag: bool = True):
        self.enable_gravity = enable_gravity
        self.enable_drag = enable_drag

    # ── Core integrator ──────────────────────────────────────────────────────
    def integrate(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        cmd_accel: np.ndarray,
        dt: float,
        *,
        drag_coeff: float = 0.0,
        mass: float = 1.0,
        max_speed: Optional[float] = None,
        max_cmd_accel: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        RK4 integration of equations of motion.

        Parameters
        ----------
        position    : current position  [x, y, z] m
        velocity    : current velocity  [vx, vy, vz] m/s
        cmd_accel   : commanded (thrust / guidance) acceleration m/s²
        dt          : time step s
        drag_coeff  : k in F_drag = -k * |v| * v  (N·s²/m² — divided by mass inside)
        mass        : vehicle mass kg
        max_speed   : speed cap m/s (hard clamp after integration)
        max_cmd_accel: magnitude cap on cmd_accel (G-limit pre-clamp) m/s²
        """
        # ── Clamp commanded acceleration (structural G-limit) ────────────────
        accel_cmd = cmd_accel.copy()
        if max_cmd_accel is not None:
            mag = np.linalg.norm(accel_cmd)
            if mag > max_cmd_accel and mag > 1e-12:
                accel_cmd *= max_cmd_accel / mag

        # ── RK4 ─────────────────────────────────────────────────────────────
        def derivs(v: np.ndarray) -> np.ndarray:
            a = accel_cmd.copy()
            if self.enable_gravity:
                a = a + GRAVITY
            if self.enable_drag and drag_coeff > 0.0:
                speed = np.linalg.norm(v)
                if speed > 1e-12:
                    a = a - (drag_coeff / mass) * speed * v
            return a

        k1v = velocity
        k1a = derivs(velocity)

        k2v = velocity + 0.5 * dt * k1a
        k2a = derivs(k2v)

        k3v = velocity + 0.5 * dt * k2a
        k3a = derivs(k3v)

        k4v = velocity + dt * k3a
        k4a = derivs(k4v)

        new_pos = position + (dt / 6.0) * (k1v + 2 * k2v + 2 * k3v + k4v)
        new_vel = velocity + (dt / 6.0) * (k1a + 2 * k2a + 2 * k3a + k4a)

        # ── Speed cap ────────────────────────────────────────────────────────
        if max_speed is not None:
            speed = np.linalg.norm(new_vel)
            if speed > max_speed and speed > 1e-12:
                new_vel *= max_speed / speed

        return new_pos, new_vel

    # ── Utility ──────────────────────────────────────────────────────────────
    @staticmethod
    def miss_distance(p1: np.ndarray, p2: np.ndarray) -> float:
        return float(np.linalg.norm(p1 - p2))

    @staticmethod
    def time_to_closest_approach(
        pos_i: np.ndarray, vel_i: np.ndarray,
        pos_t: np.ndarray, vel_t: np.ndarray,
    ) -> float:
        """
        Linear estimate of time until minimum separation.
        Used for intercept scheduling and threat assessment.
        """
        dr = pos_t - pos_i
        dv = vel_t - vel_i
        dv2 = np.dot(dv, dv)
        if dv2 < 1e-9:
            return 0.0
        t = -np.dot(dr, dv) / dv2
        return max(float(t), 0.0)

    @staticmethod
    def closing_speed(
        pos_i: np.ndarray, vel_i: np.ndarray,
        pos_t: np.ndarray, vel_t: np.ndarray,
    ) -> float:
        """Positive value = closing, negative = opening."""
        r = pos_t - pos_i
        r_norm = np.linalg.norm(r)
        if r_norm < 1e-9:
            return 0.0
        r_hat = r / r_norm
        dv = vel_t - vel_i
        return float(-np.dot(dv, r_hat))
