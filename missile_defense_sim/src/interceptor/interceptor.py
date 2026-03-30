"""
Interceptor
===========
Surface-to-air missile model.
- 3-D motion under physics engine
- Pluggable guidance law (PN or Pure Pursuit)
- G-limit enforcement
- Fuel / flight-time limit
- Hit detection (kill radius)
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import numpy as np

from src.physics.engine import PhysicsEngine
from src.guidance.base_guidance import BaseGuidance


_interceptor_counter = 0


def _next_interceptor_id() -> int:
    global _interceptor_counter
    _interceptor_counter += 1
    return _interceptor_counter


class InterceptorStatus(Enum):
    READY = auto()
    AIRBORNE = auto()
    INTERCEPT = auto()    # successful kill
    MISS = auto()         # past closest approach, no kill
    SELF_DESTRUCT = auto()


@dataclass
class InterceptorState:
    position: np.ndarray
    velocity: np.ndarray
    status: InterceptorStatus
    target_id: Optional[int]


class Interceptor:
    """
    One interceptor missile.

    Parameters
    ----------
    cfg      : full simulation config dict
    guidance : guidance law instance
    launcher_pos : launch site position (m)
    """

    def __init__(
        self,
        cfg: dict,
        guidance: BaseGuidance,
        launcher_pos: np.ndarray | None = None,
    ) -> None:
        self.interceptor_id: int = _next_interceptor_id()
        self.guidance = guidance

        i_cfg = cfg["interceptor"]
        self.max_speed: float = float(i_cfg["max_speed"])
        self.max_accel: float = float(i_cfg["max_accel"])
        self.g_limit: float = float(i_cfg["g_limit"])
        self.kill_radius: float = float(i_cfg["kill_radius"])
        self.max_range: float = float(i_cfg["max_range"])
        self.max_flight_time: float = float(i_cfg["max_flight_time"])
        self.drag_coeff: float = float(i_cfg["drag_coeff"])
        self.mass: float = float(i_cfg["mass"])

        self.launcher_pos = launcher_pos if launcher_pos is not None else np.zeros(3)
        self.position: np.ndarray = self.launcher_pos.copy().astype(np.float64)
        self.velocity: np.ndarray = np.zeros(3, dtype=np.float64)

        self.status: InterceptorStatus = InterceptorStatus.READY
        self.target_id: Optional[int] = None

        self._physics = PhysicsEngine(enable_gravity=True, enable_drag=True)
        self._flight_time: float = 0.0
        self._prev_miss_dist: float = float("inf")
        self._trajectory: list[np.ndarray] = []
        self._closest_approach: float = float("inf")
        self._launch_time: float = 0.0

    # ── Launch ─────────────────────────────────────────────────────────────

    def launch(
        self,
        target_id: int,
        initial_velocity: np.ndarray,
        launch_time: float = 0.0,
    ) -> None:
        self.target_id = target_id
        self.velocity = initial_velocity.copy().astype(np.float64)
        self.position = self.launcher_pos.copy().astype(np.float64)
        self.status = InterceptorStatus.AIRBORNE
        self._launch_time = launch_time
        self._flight_time = 0.0
        self._prev_miss_dist = float("inf")
        self._closest_approach = float("inf")
        self._trajectory = [self.position.copy()]

    # ── Step ───────────────────────────────────────────────────────────────

    def update(
        self,
        dt: float,
        tgt_pos: Optional[np.ndarray],
        tgt_vel: Optional[np.ndarray],
    ) -> Optional[int]:
        """
        Advance interceptor by dt.
        Returns target_id if intercept achieved, else None.
        """
        if self.status != InterceptorStatus.AIRBORNE:
            return None

        self._flight_time += dt

        # ── Terminal check — time / range limit ────────────────────────────
        if self._flight_time > self.max_flight_time:
            self.status = InterceptorStatus.SELF_DESTRUCT
            return None
        if np.linalg.norm(self.position - self.launcher_pos) > self.max_range:
            self.status = InterceptorStatus.SELF_DESTRUCT
            return None
        if self.position[2] < 0.0:
            self.status = InterceptorStatus.MISS
            return None

        # ── Guidance ───────────────────────────────────────────────────────
        if tgt_pos is not None and tgt_vel is not None:
            cmd = self.guidance.compute_command(
                self.position, self.velocity, tgt_pos, tgt_vel
            )
        else:
            # No track — coast straight
            cmd = np.zeros(3)

        max_cmd = self.g_limit * 9.81
        self.position, self.velocity = self._physics.integrate(
            self.position, self.velocity, cmd, dt,
            drag_coeff=self.drag_coeff,
            mass=self.mass,
            max_speed=self.max_speed,
            max_cmd_accel=max_cmd,
        )
        self._trajectory.append(self.position.copy())

        # ── Hit detection ──────────────────────────────────────────────────
        if tgt_pos is not None:
            dist = PhysicsEngine.miss_distance(self.position, tgt_pos)
            self._closest_approach = min(self._closest_approach, dist)

            if dist <= self.kill_radius:
                self.status = InterceptorStatus.INTERCEPT
                return self.target_id

            # Past-closest-approach check (dist diverging for > 2 s)
            if dist > self._prev_miss_dist + 50.0 and self._flight_time > 3.0:
                self.status = InterceptorStatus.MISS
                return None
            self._prev_miss_dist = min(self._prev_miss_dist, dist)

        return None

    # ── Properties ─────────────────────────────────────────────────────────

    @property
    def active(self) -> bool:
        return self.status == InterceptorStatus.AIRBORNE

    @property
    def trajectory(self) -> list:
        return self._trajectory

    @property
    def closest_approach(self) -> float:
        return self._closest_approach

    @property
    def flight_time(self) -> float:
        return self._flight_time

    def __repr__(self) -> str:
        return (
            f"Interceptor(id={self.interceptor_id}, "
            f"tgt={self.target_id}, "
            f"status={self.status.name}, "
            f"t={self._flight_time:.1f}s)"
        )
