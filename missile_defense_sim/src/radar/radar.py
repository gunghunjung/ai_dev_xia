"""
Radar System
============
Detection, measurement noise injection, and Kalman-filter-based multi-target tracking.
Supports optional ECM (Electronic Counter-Measures) degradation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from src.radar.kalman_filter import KalmanFilter6D


@dataclass
class Track:
    """Estimated state of a tracked target."""
    track_id: int
    target_id: int
    estimated_position: np.ndarray
    estimated_velocity: np.ndarray
    last_update_time: float
    age: float = 0.0          # seconds since first detection
    lost: bool = False        # True if not updated recently
    kf: Optional[KalmanFilter6D] = field(default=None, repr=False)


class Radar:
    """
    Single-site phased-array radar with:
    - Conical field-of-view (elevation band)
    - Range-limited detection
    - Gaussian measurement noise
    - Per-target Kalman Filter tracking
    - ECM / jamming mode
    """

    def __init__(self, cfg: dict, physics_dt: float) -> None:
        r = cfg["radar"]
        self.range_max: float = r["range_max"]
        self.elev_min: float = math.radians(r["fov_elevation_min"])
        self.elev_max: float = math.radians(r["fov_elevation_max"])
        self.noise_std_pos: float = r["noise_std_position"]
        self.noise_std_vel: float = r["noise_std_velocity"]
        self.update_interval: float = r["update_interval"]
        self.ecm_enabled: bool = r["ecm_enabled"]
        self.ecm_noise_mult: float = r["ecm_noise_multiplier"]

        kf_cfg = r["kalman"]
        self._kf_process_noise: float = kf_cfg["process_noise"]
        self._kf_meas_noise: float = kf_cfg["measurement_noise"]
        self._physics_dt = physics_dt

        self._tracks: Dict[int, Track] = {}
        self._next_track_id: int = 0
        self._target_to_track: Dict[int, int] = {}   # target_id → track_id
        self._last_update: float = -999.0

    # ── Public API ─────────────────────────────────────────────────────────

    def step(self, targets: list, sim_time: float) -> None:
        """
        Called every simulation tick.  Only runs the full detect+KF update
        at self.update_interval cadence.  Otherwise just propagates the KF.
        """
        if sim_time - self._last_update >= self.update_interval:
            self._detect_and_update(targets, sim_time)
            self._last_update = sim_time
        else:
            dt = sim_time - self._last_update if sim_time > self._last_update else self._physics_dt
            # KF time-propagation only
            for track in self._tracks.values():
                if not track.lost and track.kf and track.kf.initialized:
                    track.kf.predict(dt)
                    track.estimated_position = track.kf.position
                    track.estimated_velocity = track.kf.velocity

    def get_tracks(self) -> List[Track]:
        return [t for t in self._tracks.values() if not t.lost]

    def get_track(self, target_id: int) -> Optional[Track]:
        tid = self._target_to_track.get(target_id)
        if tid is None:
            return None
        track = self._tracks.get(tid)
        if track is None or track.lost:
            return None
        return track

    def remove_track(self, target_id: int) -> None:
        tid = self._target_to_track.pop(target_id, None)
        if tid is not None:
            self._tracks.pop(tid, None)

    # ── Internal ───────────────────────────────────────────────────────────

    def _in_fov(self, pos: np.ndarray) -> bool:
        r = np.linalg.norm(pos)
        if r < 1e-6 or r > self.range_max:
            return False
        if pos[2] < 0.0:
            return False
        elev = math.asin(np.clip(pos[2] / r, -1.0, 1.0))
        return self.elev_min <= elev <= self.elev_max

    def _measure(self, true_pos: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        noise_mult = self.ecm_noise_mult if self.ecm_enabled else 1.0
        sigma = self.noise_std_pos * noise_mult
        noise = rng.normal(0.0, sigma, size=3)
        return true_pos + noise

    def _detect_and_update(self, targets: list, sim_time: float) -> None:
        rng = np.random.default_rng()
        dt = sim_time - self._last_update if self._last_update > 0 else self.update_interval

        # Mark all existing tracks as stale; refresh below
        stale_ids = set(self._target_to_track.keys())

        for target in targets:
            if not target.alive:
                self.remove_track(target.target_id)
                stale_ids.discard(target.target_id)
                continue

            if not self._in_fov(target.position):
                # Lost track
                if target.target_id in self._target_to_track:
                    tid = self._target_to_track[target.target_id]
                    self._tracks[tid].lost = True
                stale_ids.discard(target.target_id)
                continue

            stale_ids.discard(target.target_id)
            measured_pos = self._measure(target.position, rng)

            if target.target_id in self._target_to_track:
                # Update existing track
                tid = self._target_to_track[target.target_id]
                track = self._tracks[tid]
                track.lost = False
                track.age += dt
                track.last_update_time = sim_time
                if track.kf and track.kf.initialized:
                    track.kf.step(measured_pos, dt)
                    track.estimated_position = track.kf.position
                    track.estimated_velocity = track.kf.velocity
            else:
                # New track
                kf = KalmanFilter6D(
                    dt=self.update_interval,
                    process_noise=self._kf_process_noise,
                    measurement_noise=self._kf_meas_noise,
                )
                kf.initialize(measured_pos, target.velocity.copy())
                new_tid = self._next_track_id
                self._next_track_id += 1
                track = Track(
                    track_id=new_tid,
                    target_id=target.target_id,
                    estimated_position=measured_pos.copy(),
                    estimated_velocity=target.velocity.copy(),
                    last_update_time=sim_time,
                    kf=kf,
                )
                self._tracks[new_tid] = track
                self._target_to_track[target.target_id] = new_tid

        # Any remaining stale ids → lost
        for tid_target in stale_ids:
            tid = self._target_to_track.get(tid_target)
            if tid is not None and tid in self._tracks:
                self._tracks[tid].lost = True
