"""
Logger & Analyzer
=================
Records simulation state each tick.
Computes post-run statistics.
Exports to CSV / pandas DataFrame.
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False


@dataclass
class InterceptEvent:
    sim_time: float
    interceptor_id: int
    target_id: int
    target_type: str
    intercept_position: np.ndarray
    miss_distance: float
    flight_time: float
    success: bool


@dataclass
class TrialResult:
    trial_index: int
    n_targets: int
    n_intercepts: int
    n_misses: int
    intercept_events: List[InterceptEvent] = field(default_factory=list)
    interceptors_expended: int = 0
    duration: float = 0.0

    @property
    def success_rate(self) -> float:
        return self.n_intercepts / max(self.n_targets, 1)

    @property
    def avg_miss_distance(self) -> float:
        hits = [e.miss_distance for e in self.intercept_events if e.success]
        return float(np.mean(hits)) if hits else float("nan")

    @property
    def avg_flight_time(self) -> float:
        hits = [e.flight_time for e in self.intercept_events if e.success]
        return float(np.mean(hits)) if hits else float("nan")


class SimLogger:
    """Records raw step-by-step state for one trial."""

    def __init__(self) -> None:
        self._records: List[Dict] = []
        self._events: List[InterceptEvent] = []

    def log_step(
        self,
        t: float,
        targets: list,
        interceptors: list,
    ) -> None:
        for tgt in targets:
            self._records.append({
                "time": t,
                "entity": "target",
                "id": tgt.target_id,
                "type": tgt.target_type,
                "x": tgt.position[0],
                "y": tgt.position[1],
                "z": tgt.position[2],
                "vx": tgt.velocity[0],
                "vy": tgt.velocity[1],
                "vz": tgt.velocity[2],
                "alive": tgt.alive,
            })
        for itr in interceptors:
            self._records.append({
                "time": t,
                "entity": "interceptor",
                "id": itr.interceptor_id,
                "type": "interceptor",
                "x": itr.position[0],
                "y": itr.position[1],
                "z": itr.position[2],
                "vx": itr.velocity[0],
                "vy": itr.velocity[1],
                "vz": itr.velocity[2],
                "alive": itr.active,
            })

    def log_event(self, event: InterceptEvent) -> None:
        self._events.append(event)

    def to_dataframe(self):
        if not _HAS_PANDAS:
            raise ImportError("pandas not installed")
        return pd.DataFrame(self._records)

    def events_to_dataframe(self):
        if not _HAS_PANDAS:
            raise ImportError("pandas not installed")
        rows = []
        for e in self._events:
            rows.append({
                "sim_time": e.sim_time,
                "interceptor_id": e.interceptor_id,
                "target_id": e.target_id,
                "target_type": e.target_type,
                "ix": e.intercept_position[0],
                "iy": e.intercept_position[1],
                "iz": e.intercept_position[2],
                "miss_distance_m": e.miss_distance,
                "flight_time_s": e.flight_time,
                "success": e.success,
            })
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def save_csv(self, path: str) -> None:
        if not self._records:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        keys = list(self._records[0].keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self._records)

    @property
    def events(self) -> List[InterceptEvent]:
        return self._events


class MonteCarloAnalyzer:
    """Aggregates results from multiple TrialResults."""

    def __init__(self) -> None:
        self._trials: List[TrialResult] = []

    def add(self, result: TrialResult) -> None:
        self._trials.append(result)

    def summary(self) -> Dict:
        if not self._trials:
            return {}
        n = len(self._trials)
        success_rates = [t.success_rate for t in self._trials]
        miss_dists = [t.avg_miss_distance for t in self._trials if not np.isnan(t.avg_miss_distance)]
        flight_times = [t.avg_flight_time for t in self._trials if not np.isnan(t.avg_flight_time)]
        interceptors = [t.interceptors_expended for t in self._trials]

        return {
            "n_trials": n,
            "success_rate_mean": float(np.mean(success_rates)),
            "success_rate_std": float(np.std(success_rates)),
            "success_rate_min": float(np.min(success_rates)),
            "success_rate_max": float(np.max(success_rates)),
            "avg_miss_distance_m": float(np.nanmean(miss_dists)) if miss_dists else float("nan"),
            "avg_intercept_time_s": float(np.nanmean(flight_times)) if flight_times else float("nan"),
            "avg_interceptors_per_trial": float(np.mean(interceptors)),
        }

    def to_dataframe(self):
        if not _HAS_PANDAS:
            raise ImportError("pandas not installed")
        rows = []
        for t in self._trials:
            rows.append({
                "trial": t.trial_index,
                "n_targets": t.n_targets,
                "n_intercepts": t.n_intercepts,
                "n_misses": t.n_misses,
                "success_rate": t.success_rate,
                "avg_miss_dist_m": t.avg_miss_distance,
                "avg_flight_time_s": t.avg_flight_time,
                "interceptors_expended": t.interceptors_expended,
                "duration_s": t.duration,
            })
        return pd.DataFrame(rows)

    def save_csv(self, path: str) -> None:
        if not _HAS_PANDAS:
            raise ImportError("pandas not installed")
        df = self.to_dataframe()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        df.to_csv(path, index=False)
        print(f"[Analyzer] Saved Monte Carlo results → {path}")

    def print_summary(self) -> None:
        s = self.summary()
        print("\n" + "=" * 50)
        print("  Monte Carlo Summary")
        print("=" * 50)
        for k, v in s.items():
            if isinstance(v, float):
                print(f"  {k:<35s}: {v:.4f}")
            else:
                print(f"  {k:<35s}: {v}")
        print("=" * 50)
