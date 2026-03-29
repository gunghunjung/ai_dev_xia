"""
Engagement Controller
=====================
Decides WHEN and WHICH targets to engage and assigns interceptors.

Threat assessment
-----------------
Score = w_tti  / TTI_normalized
      + w_rng  * (1 - range / range_max)
      + w_spd  * closing_speed / speed_max_ref

Higher score = higher priority.

Assignment methods
------------------
- greedy  : sort targets by threat, assign one interceptor each
- hungarian : uses scipy.optimize.linear_sum_assignment on cost matrix
              (cost = 1 - threat score for existing assignments)
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from src.radar.radar import Track
from src.interceptor.interceptor import Interceptor, InterceptorStatus
from src.guidance.base_guidance import BaseGuidance
from src.guidance.proportional_nav import ProportionalNavigation
from src.guidance.pure_pursuit import PurePursuit
from src.physics.engine import PhysicsEngine


class EngagementController:

    def __init__(self, cfg: dict) -> None:
        e_cfg = cfg["engagement"]
        g_cfg = cfg["guidance"]
        i_cfg = cfg["interceptor"]

        self._max_interceptors: int = int(e_cfg["max_interceptors_in_air"])
        self._salvo: int = int(e_cfg["interceptors_per_target"])
        self._method: str = e_cfg["assignment_method"]
        self._min_intercept_alt: float = float(e_cfg["min_intercept_altitude"])
        self._tti_threshold: float = float(e_cfg["tti_threshold"])

        lz = e_cfg["launch_zone"]
        self._lz_range_min: float = float(lz["range_min"])
        self._lz_range_max: float = float(lz["range_max"])
        self._lz_elev_min: float = math.radians(lz["elevation_min_deg"])
        self._lz_elev_max: float = math.radians(lz["elevation_max_deg"])

        self._w_tti: float = float(e_cfg["threat_weight_tti"])
        self._w_rng: float = float(e_cfg["threat_weight_range"])
        self._w_spd: float = float(e_cfg["threat_weight_speed"])

        # Guidance factory
        guidance_type = g_cfg["type"].upper()
        self._guidance_type = guidance_type
        self._nav_constant = float(g_cfg["navigation_constant"])

        self._kill_radius = float(i_cfg["kill_radius"])
        self._max_speed = float(i_cfg["max_speed"])
        self._range_max = float(cfg["radar"]["range_max"])

        # Battery launcher position (origin by default)
        self._launcher_pos = np.zeros(3)

        # Track which target IDs have been engaged
        self._engaged: Dict[int, List[int]] = {}       # target_id → [interceptor_ids]
        self._interceptors: List[Interceptor] = []

    # ── Public API ─────────────────────────────────────────────────────────

    def set_launcher_pos(self, pos: np.ndarray) -> None:
        self._launcher_pos = pos.copy()

    @property
    def interceptors(self) -> List[Interceptor]:
        return self._interceptors

    def step(self, tracks: List[Track], sim_time: float, cfg: dict) -> List[Interceptor]:
        """
        Evaluate engagement decisions and return any newly launched interceptors.
        """
        # ── Prune finished interceptors ────────────────────────────────────
        finished_ids: Set[int] = {
            i.interceptor_id for i in self._interceptors if not i.active
        }
        for i in self._interceptors:
            if i.interceptor_id in finished_ids:
                tgt_id = i.target_id
                if tgt_id in self._engaged:
                    try:
                        self._engaged[tgt_id].remove(i.interceptor_id)
                    except ValueError:
                        pass
        self._interceptors = [i for i in self._interceptors if i.active]

        airborne_count = len(self._interceptors)
        new_launches: List[Interceptor] = []

        if airborne_count >= self._max_interceptors:
            return new_launches

        # ── Score and prioritise tracks ────────────────────────────────────
        scored = []
        for track in tracks:
            if not self._in_launch_zone(track):
                continue
            score = self._threat_score(track)
            if score > 0:
                scored.append((score, track))

        scored.sort(key=lambda x: x[0], reverse=True)

        # ── Assign ────────────────────────────────────────────────────────
        if self._method == "hungarian":
            assignments = self._hungarian_assign(scored)
        else:
            assignments = self._greedy_assign(scored)

        for track in assignments:
            already = self._engaged.get(track.target_id, [])
            slots_needed = self._salvo - len(already)
            available = self._max_interceptors - airborne_count - len(new_launches)
            n_launch = min(slots_needed, available)
            if n_launch <= 0:
                continue
            for _ in range(n_launch):
                interceptor = self._create_interceptor(track, sim_time, cfg)
                self._interceptors.append(interceptor)
                new_launches.append(interceptor)
                self._engaged.setdefault(track.target_id, []).append(
                    interceptor.interceptor_id
                )

        return new_launches

    def notify_intercept(self, target_id: int) -> None:
        """Called when a target is confirmed destroyed."""
        self._engaged.pop(target_id, None)

    # ── Internals ──────────────────────────────────────────────────────────

    def _in_launch_zone(self, track: Track) -> bool:
        pos = track.estimated_position
        rng = np.linalg.norm(pos - self._launcher_pos)
        if rng < self._lz_range_min or rng > self._lz_range_max:
            return False
        if rng < 1e-6:
            return False
        elev = math.asin(np.clip(pos[2] / rng, -1.0, 1.0))
        if not (self._lz_elev_min <= elev <= self._lz_elev_max):
            return False
        if pos[2] < self._min_intercept_alt:
            return False
        return True

    def _threat_score(self, track: Track) -> float:
        pos = track.estimated_position
        vel = track.estimated_velocity
        rng = float(np.linalg.norm(pos - self._launcher_pos))
        if rng < 1e-6:
            return 0.0

        closing = PhysicsEngine.closing_speed(
            self._launcher_pos, np.zeros(3), pos, vel
        )
        if closing <= 0.0:
            return 0.0  # opening — not a threat

        tti = rng / max(closing, 1.0)
        if tti > self._tti_threshold:
            return 0.0

        tti_norm = 1.0 / max(tti, 1.0)
        rng_norm = 1.0 - rng / self._range_max
        spd_norm = closing / (self._max_speed + 1e-9)

        return (
            self._w_tti * tti_norm
            + self._w_rng * max(rng_norm, 0.0)
            + self._w_spd * min(spd_norm, 1.0)
        )

    def _greedy_assign(
        self, scored: List[Tuple[float, Track]]
    ) -> List[Track]:
        result = []
        for _, track in scored:
            already = len(self._engaged.get(track.target_id, []))
            if already < self._salvo:
                result.append(track)
        return result

    def _hungarian_assign(
        self, scored: List[Tuple[float, Track]]
    ) -> List[Track]:
        """
        Use scipy linear_sum_assignment on a cost matrix.
        Rows = interceptor slots available, Cols = scored tracks.
        """
        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError:
            return self._greedy_assign(scored)

        if not scored:
            return []

        airborne = len(self._interceptors)
        slots = self._max_interceptors - airborne
        if slots <= 0:
            return []

        n_tracks = len(scored)
        n_rows = min(slots, n_tracks)
        cost = np.zeros((n_rows, n_tracks))
        for r in range(n_rows):
            for c, (score, _) in enumerate(scored):
                cost[r, c] = 1.0 - score   # minimise cost = maximise score

        row_ind, col_ind = linear_sum_assignment(cost)
        assigned: Set[int] = set()
        result = []
        for r, c in zip(row_ind, col_ind):
            track = scored[c][1]
            if track.target_id not in assigned:
                already = len(self._engaged.get(track.target_id, []))
                if already < self._salvo:
                    result.append(track)
                    assigned.add(track.target_id)
        return result

    def _create_interceptor(
        self, track: Track, launch_time: float, cfg: dict
    ) -> Interceptor:
        guidance = self._make_guidance()
        interceptor = Interceptor(cfg, guidance, self._launcher_pos.copy())

        tgt_pos = track.estimated_position
        tgt_vel = track.estimated_velocity
        launch_speed = 0.88 * interceptor.max_speed
        G = 9.81

        # ── Estimate intercept time (kinematic lead, no gravity) ───────────
        dp = tgt_pos - self._launcher_pos
        dv = tgt_vel
        a_q = np.dot(dv, dv) - launch_speed ** 2
        b_q = 2.0 * np.dot(dp, dv)
        c_q = np.dot(dp, dp)

        t_i = None
        if abs(a_q) > 1e-6:
            disc = b_q * b_q - 4.0 * a_q * c_q
            if disc >= 0:
                sqrt_disc = np.sqrt(max(disc, 0.0))
                for sign in [-1.0, 1.0]:
                    t_cand = (-b_q + sign * sqrt_disc) / (2.0 * a_q)
                    if t_cand > 0.5:
                        t_i = t_cand
                        break

        if t_i is None or t_i > interceptor.max_flight_time:
            t_i = np.linalg.norm(dp) / (launch_speed + 1e-9)

        t_i = min(float(t_i), interceptor.max_flight_time * 0.7)

        # ── Predicted target position at intercept time ────────────────────
        pred_pos = tgt_pos + tgt_vel * t_i

        # ── Compute launch velocity with GRAVITY COMPENSATION ─────────────
        # Ballistic trajectory: z(t) = v_z0*t - 0.5*g*t²  →  v_z0 = (Δz + 0.5*g*t²)/t
        delta = pred_pos - self._launcher_pos
        v_req = np.array([
            delta[0] / t_i,
            delta[1] / t_i,
            (delta[2] + 0.5 * G * t_i ** 2) / t_i,
        ])

        # Clamp to launch speed
        spd = np.linalg.norm(v_req)
        if spd > 1e-6:
            v_req = v_req * (launch_speed / spd)
        else:
            v_req = np.array([0.0, 0.0, launch_speed])

        # Safety: minimum upward component to avoid immediate ground impact
        if v_req[2] < 10.0:
            v_req[2] = 10.0
            v_req = v_req * (launch_speed / np.linalg.norm(v_req))

        interceptor.launch(track.target_id, v_req, launch_time)
        return interceptor

    def _make_guidance(self) -> BaseGuidance:
        if self._guidance_type == "PN":
            return ProportionalNavigation(N=self._nav_constant, augmented=False)
        elif self._guidance_type == "PURSUIT":
            return PurePursuit(gain=10.0)
        else:
            return ProportionalNavigation(N=self._nav_constant, augmented=False)
