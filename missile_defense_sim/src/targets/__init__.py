"""Target factory."""
from __future__ import annotations

import numpy as np

from src.targets.ballistic_target import BallisticTarget
from src.targets.cruise_target import CruiseTarget
from src.targets.evasive_target import EvasiveTarget

_TYPE_MAP = {
    "ballistic": BallisticTarget,
    "cruise": CruiseTarget,
    "evasive": EvasiveTarget,
}


def create_target(
    target_type: str,
    cfg: dict,
    azimuth_deg: float = 0.0,
    rng: np.random.Generator | None = None,
):
    """Instantiate a target by type string."""
    cls = _TYPE_MAP.get(target_type)
    if cls is None:
        raise ValueError(f"Unknown target type '{target_type}'. Choose from {list(_TYPE_MAP)}")
    return cls(cfg, azimuth_deg=azimuth_deg, rng=rng)


def create_random_targets(n: int, cfg: dict, rng: np.random.Generator | None = None):
    """Create n targets with random types and azimuths."""
    if rng is None:
        rng = np.random.default_rng()
    dist = cfg["targets"]["types_distribution"]
    types = list(dist.keys())
    weights = np.array([dist[t] for t in types], dtype=float)
    weights /= weights.sum()
    chosen = rng.choice(types, size=n, p=weights)
    azimuths = rng.uniform(0, 360, size=n)
    return [create_target(t, cfg, az, rng) for t, az in zip(chosen, azimuths)]
