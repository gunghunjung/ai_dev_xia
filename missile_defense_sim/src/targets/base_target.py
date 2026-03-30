"""
Base Target
===========
Abstract base class for all threat models.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any

import numpy as np

from src.physics.engine import PhysicsEngine


_target_counter = 0


def _next_target_id() -> int:
    global _target_counter
    _target_counter += 1
    return _target_counter


@dataclass
class TargetState:
    position: np.ndarray
    velocity: np.ndarray
    alive: bool = True
    intercepted: bool = False
    hit_ground: bool = False


class BaseTarget(ABC):
    """
    Common interface for all target types.
    Subclasses override _compute_thrust() to implement specific motion models.
    """

    def __init__(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        cfg: dict,
        target_type: str,
    ) -> None:
        self.target_id: int = _next_target_id()
        self.target_type: str = target_type
        self.position: np.ndarray = position.copy().astype(np.float64)
        self.velocity: np.ndarray = velocity.copy().astype(np.float64)
        self.alive: bool = True
        self.intercepted: bool = False
        self.hit_ground: bool = False

        t_cfg = cfg["targets"][target_type]
        self.drag_coeff: float = float(t_cfg.get("drag_coeff", 0.1))
        self.mass: float = float(t_cfg.get("mass", 500.0))

        self._physics = PhysicsEngine(enable_gravity=True, enable_drag=True)
        self._trajectory: list[np.ndarray] = [position.copy()]

    # ── Public API ─────────────────────────────────────────────────────────

    def update(self, dt: float) -> None:
        if not self.alive:
            return
        thrust = self._compute_thrust(dt)
        self.position, self.velocity = self._physics.integrate(
            self.position, self.velocity, thrust, dt,
            drag_coeff=self.drag_coeff,
            mass=self.mass,
        )
        self._trajectory.append(self.position.copy())
        if self.position[2] <= 0.0:
            self.position[2] = 0.0
            self.alive = False
            self.hit_ground = True

    @abstractmethod
    def _compute_thrust(self, dt: float) -> np.ndarray:
        """Return thrust acceleration vector (m/s²) for this time step."""

    def mark_intercepted(self) -> None:
        self.alive = False
        self.intercepted = True

    @property
    def trajectory(self) -> list:
        return self._trajectory

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(id={self.target_id}, "
            f"pos={self.position.round(1)}, "
            f"spd={np.linalg.norm(self.velocity):.0f} m/s, "
            f"alive={self.alive})"
        )
