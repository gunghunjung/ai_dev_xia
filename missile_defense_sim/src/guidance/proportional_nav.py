"""
Proportional Navigation (LOS-rate formulation)
================================================
Standard PN law used in practice:

    a_cmd = N · V_c · λ̇_⊥

where:
  N     = navigation constant (3–5 typical)
  V_c   = closing speed (positive when interceptor and target approach)
  λ̇_⊥  = LOS angular rate vector (component perpendicular to LOS)

Derivation
----------
  r     = r_T - r_I                   (LOS vector)
  v_r   = v_T - v_I                   (relative velocity, target w.r.t. interceptor)
  r̂     = r / |r|                     (LOS unit vector)
  V_c   = -dot(v_r, r̂)               (positive → closing)
  v_⊥   = v_r - dot(v_r, r̂)·r̂       (v_r component perp to LOS)
  λ̇_⊥  = v_⊥ / |r|                  (angular rate, rad/s, as a 3-D vector)

  a_cmd = N · V_c · λ̇_⊥

The command is always perpendicular to the LOS vector, which is the
physically correct intercept acceleration direction.  This formulation does
NOT depend on missile speed, making it robust across all flight phases.

Optional gravity compensation
------------------------------
When intercept altitude changes significantly, a constant upward bias
  a_bias = (N/2) · g_vertical
can reduce miss distance against strongly curved trajectories (ballistic targets).
Disabled by default.
"""
from __future__ import annotations

import numpy as np

from src.guidance.base_guidance import BaseGuidance


class ProportionalNavigation(BaseGuidance):

    def __init__(self, N: float = 4.0, augmented: bool = False) -> None:
        """
        Parameters
        ----------
        N         : navigation constant (3–5 typical)
        augmented : if True, add (N/2)·g upward bias for ballistic targets
        """
        self.N = float(N)
        self.augmented = augmented
        self._G = np.array([0.0, 0.0, 9.81])   # upward unit, for bias only

    def compute_command(
        self,
        int_pos: np.ndarray,
        int_vel: np.ndarray,
        tgt_pos: np.ndarray,
        tgt_vel: np.ndarray,
    ) -> np.ndarray:
        r = tgt_pos - int_pos           # LOS vector
        v_r = tgt_vel - int_vel         # relative velocity (tgt w.r.t. int)
        r_norm = np.linalg.norm(r)

        if r_norm < 1.0:
            return np.zeros(3)

        r_hat = r / r_norm

        # Closing speed (positive = closing)
        V_c = float(-np.dot(v_r, r_hat))

        if V_c <= 0.0:
            # Opening — zero command (will be caught by miss detection)
            return np.zeros(3)

        # Perpendicular component of relative velocity (=> LOS rate direction)
        v_perp = v_r - np.dot(v_r, r_hat) * r_hat   # component ⊥ to LOS
        los_rate_vec = v_perp / r_norm               # rad/s vector

        # PN command
        a_cmd = self.N * V_c * los_rate_vec

        if self.augmented:
            # Partial gravity compensation for target on curved trajectory
            a_cmd = a_cmd + (self.N / 2.0) * self._G

        return a_cmd
