"""
Simulation Controller
=====================
Owns the main time-loop and wires together every subsystem:

    Radar ↔ Targets ↔ EngagementController ↔ Interceptors ↔ PhysicsEngine

Single trial flow
-----------------
1. tick targets (physics + target-specific logic)
2. tick radar   (detect / KF-update)
3. engagement   (threat evaluation → launch decisions)
4. tick interceptors (guidance + physics)
5. hit detection
6. log state
7. termination check

Monte Carlo
-----------
Runs N independent trials with fresh randomised initial conditions,
aggregates results via MonteCarloAnalyzer.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.physics.engine import PhysicsEngine
from src.radar.radar import Radar
from src.targets import create_random_targets, create_target
from src.targets.base_target import BaseTarget
from src.interceptor.interceptor import Interceptor, InterceptorStatus
from src.engagement.controller import EngagementController
from src.logger.analyzer import (
    SimLogger,
    MonteCarloAnalyzer,
    InterceptEvent,
    TrialResult,
)

# Reset module-level counters between trials so IDs stay meaningful
import src.targets.base_target as _bt_mod
import src.interceptor.interceptor as _int_mod


def _reset_counters() -> None:
    _bt_mod._target_counter = 0
    _int_mod._interceptor_counter = 0


class SimulationController:
    """
    Runs one or many simulation trials.

    Parameters
    ----------
    cfg     : parsed YAML config dict
    verbose : print progress to stdout
    """

    def __init__(self, cfg: dict, verbose: bool = True) -> None:
        self.cfg = cfg
        self.verbose = verbose
        self._physics = PhysicsEngine()

    # ── Public entry-points ────────────────────────────────────────────────

    def run_single(
        self,
        n_targets: Optional[int] = None,
        duration: Optional[float] = None,
        guidance_type: Optional[str] = None,
        rng_seed: Optional[int] = None,
        save_csv: Optional[str] = None,
    ) -> Tuple[TrialResult, SimLogger]:
        """
        Run one scenario and return (TrialResult, SimLogger).
        The SimLogger contains full trajectory data for visualisation.
        """
        cfg = self._override_cfg(n_targets, duration, guidance_type)
        rng = np.random.default_rng(rng_seed)
        result, logger = self._run_trial(cfg, trial_idx=0, rng=rng, record=True)

        if save_csv:
            logger.save_csv(save_csv)
            if self.verbose:
                print(f"[Sim] Trajectory data → {save_csv}")

        return result, logger

    def run_monte_carlo(
        self,
        n_targets: Optional[int] = None,
        duration: Optional[float] = None,
        guidance_type: Optional[str] = None,
        save_csv: Optional[str] = None,
    ) -> MonteCarloAnalyzer:
        """Run N_TRIALS independent trials and return aggregated statistics."""
        cfg = self._override_cfg(n_targets, duration, guidance_type)
        n_trials: int = cfg["simulation"]["n_trials"]
        base_seed: int = cfg["simulation"].get("random_seed", 42)

        analyzer = MonteCarloAnalyzer()
        t0 = time.perf_counter()

        for i in range(n_trials):
            rng = np.random.default_rng(base_seed + i)
            result, _ = self._run_trial(cfg, trial_idx=i, rng=rng, record=False)
            analyzer.add(result)
            if self.verbose and (i + 1) % max(1, n_trials // 10) == 0:
                pct = 100 * (i + 1) / n_trials
                elapsed = time.perf_counter() - t0
                print(f"  [{pct:5.1f}%]  trial {i+1}/{n_trials}  "
                      f"elapsed {elapsed:.1f}s")

        if self.verbose:
            analyzer.print_summary()

        if save_csv:
            analyzer.save_csv(save_csv)

        return analyzer

    # ── Core trial loop ────────────────────────────────────────────────────

    def _run_trial(
        self,
        cfg: dict,
        trial_idx: int,
        rng: np.random.Generator,
        record: bool,
    ) -> Tuple[TrialResult, SimLogger]:
        _reset_counters()

        dt: float = cfg["simulation"]["dt"]
        max_dur: float = cfg["simulation"]["max_duration"]
        n_targets: int = cfg["targets"]["n_targets"]

        # ── Instantiate subsystems ─────────────────────────────────────────
        targets: List[BaseTarget] = create_random_targets(n_targets, cfg, rng)
        radar = Radar(cfg, physics_dt=dt)
        engagement = EngagementController(cfg)

        if cfg["guidance"]["type"].upper() != cfg["guidance"]["type"]:
            # normalise
            cfg["guidance"]["type"] = cfg["guidance"]["type"].upper()

        logger = SimLogger()

        # ── State tracking ─────────────────────────────────────────────────
        all_interceptors: List[Interceptor] = []       # every launched interceptor
        intercepted_target_ids: set = set()
        intercept_events: List[InterceptEvent] = []

        # Map target_id → target object for O(1) lookup
        tgt_map: Dict[int, BaseTarget] = {t.target_id: t for t in targets}

        t = 0.0
        step = 0

        while t <= max_dur:
            alive_targets = [tg for tg in targets if tg.alive]
            active_interceptors = [i for i in engagement.interceptors if i.active]

            # ── 1. Targets ─────────────────────────────────────────────────
            for tgt in alive_targets:
                tgt.update(dt)

            # ── 2. Radar ───────────────────────────────────────────────────
            radar.step(alive_targets, t)

            # ── 3. Engagement decisions ────────────────────────────────────
            tracks = radar.get_tracks()
            # Filter tracks for already-dead targets
            tracks = [tr for tr in tracks if tr.target_id in tgt_map
                      and tgt_map[tr.target_id].alive]

            new_interceptors = engagement.step(tracks, t, cfg)
            all_interceptors.extend(new_interceptors)

            # ── 4. Interceptors ────────────────────────────────────────────
            for itr in list(engagement.interceptors):
                if not itr.active:
                    continue
                track = radar.get_track(itr.target_id)
                tgt_obj = tgt_map.get(itr.target_id) if itr.target_id else None

                if track and tgt_obj and tgt_obj.alive:
                    tgt_pos = track.estimated_position
                    tgt_vel = track.estimated_velocity
                else:
                    tgt_pos = None
                    tgt_vel = None

                killed_id = itr.update(dt, tgt_pos, tgt_vel)

                # ── 5. Hit detection ───────────────────────────────────────
                if killed_id is not None:
                    tgt_obj = tgt_map.get(killed_id)
                    if tgt_obj and tgt_obj.alive:
                        tgt_obj.mark_intercepted()
                        intercepted_target_ids.add(killed_id)
                        radar.remove_track(killed_id)
                        engagement.notify_intercept(killed_id)

                        event = InterceptEvent(
                            sim_time=t,
                            interceptor_id=itr.interceptor_id,
                            target_id=killed_id,
                            target_type=tgt_obj.target_type,
                            intercept_position=itr.position.copy(),
                            miss_distance=itr.closest_approach,
                            flight_time=itr.flight_time,
                            success=True,
                        )
                        intercept_events.append(event)
                        logger.log_event(event)
                        if self.verbose and record:
                            print(
                                f"  [t={t:.1f}s] INTERCEPT  "
                                f"interceptor={itr.interceptor_id}  "
                                f"target={killed_id} ({tgt_obj.target_type})  "
                                f"miss_dist={itr.closest_approach:.1f}m  "
                                f"alt={itr.position[2]:.0f}m"
                            )

            # ── 6. Log ─────────────────────────────────────────────────────
            if record and step % 4 == 0:   # log every 4 steps to reduce data size
                logger.log_step(t, targets, list(engagement.interceptors) + [
                    i for i in all_interceptors if not i.active
                ])

            # ── 7. Termination ─────────────────────────────────────────────
            still_alive = [tg for tg in targets if tg.alive]
            if not still_alive:
                if self.verbose and record:
                    print(f"  [t={t:.1f}s] All targets neutralised.")
                break

            t += dt
            step += 1

        # ── Build result ───────────────────────────────────────────────────
        n_intercepts = len(intercepted_target_ids)
        n_misses = n_targets - n_intercepts
        interceptors_used = len(all_interceptors)

        result = TrialResult(
            trial_index=trial_idx,
            n_targets=n_targets,
            n_intercepts=n_intercepts,
            n_misses=n_misses,
            intercept_events=intercept_events,
            interceptors_expended=interceptors_used,
            duration=t,
        )

        if self.verbose and record:
            print(
                f"\n[Trial {trial_idx}]  "
                f"Intercepts: {n_intercepts}/{n_targets}  "
                f"({result.success_rate*100:.0f}%)  "
                f"Interceptors fired: {interceptors_used}  "
                f"Duration: {t:.1f}s"
            )

        return result, logger

    # ── Config helpers ─────────────────────────────────────────────────────

    def _override_cfg(
        self,
        n_targets: Optional[int],
        duration: Optional[float],
        guidance_type: Optional[str],
    ) -> dict:
        """Return a shallow-copy of cfg with CLI overrides applied."""
        import copy
        cfg = copy.deepcopy(self.cfg)
        if n_targets is not None:
            cfg["targets"]["n_targets"] = n_targets
        if duration is not None:
            cfg["simulation"]["max_duration"] = duration
        if guidance_type is not None:
            cfg["guidance"]["type"] = guidance_type.upper()
        return cfg
