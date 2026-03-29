"""
Visualizer
==========
matplotlib-based 3-D and 2-D trajectory plots.

Features
--------
- 3D trajectory view (mpl_toolkits.mplot3d)
- Top-down (X-Y) and Range-Altitude (R-Z) 2D views
- Intercept / miss markers
- Battery (origin) marker
- Radar range circle overlay
- Monte Carlo aggregate success-rate bar chart
- Guidance comparison: PN vs Pure Pursuit side-by-side

All functions are stateless — pass in a SimLogger (or raw data)
and they return the Figure.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)

from src.logger.analyzer import SimLogger, InterceptEvent, MonteCarloAnalyzer

# ── Colour palette ─────────────────────────────────────────────────────────
_COLOURS = {
    "ballistic": "#e74c3c",
    "cruise":    "#e67e22",
    "evasive":   "#9b59b6",
    "interceptor": "#2980b9",
    "intercept_ok": "#27ae60",
    "intercept_fail": "#c0392b",
    "battery": "#f39c12",
    "radar_ring": "#bdc3c7",
}


# ── Helper ─────────────────────────────────────────────────────────────────

def _entity_records(logger: SimLogger, entity: str, eid: int) -> Dict[str, list]:
    """Extract time-series lists for one entity from logger records."""
    xs, ys, zs, ts = [], [], [], []
    for rec in logger._records:
        if rec["entity"] == entity and rec["id"] == eid:
            ts.append(rec["time"])
            xs.append(rec["x"])
            ys.append(rec["y"])
            zs.append(rec["z"])
    return {"t": ts, "x": xs, "y": ys, "z": zs}


def _unique_ids(logger: SimLogger, entity: str) -> List[int]:
    seen = []
    for rec in logger._records:
        if rec["entity"] == entity and rec["id"] not in seen:
            seen.append(rec["id"])
    return seen


def _type_for_id(logger: SimLogger, target_id: int) -> str:
    for rec in logger._records:
        if rec["entity"] == "target" and rec["id"] == target_id:
            return rec.get("type", "unknown")
    return "unknown"


# ── 3-D trajectory plot ────────────────────────────────────────────────────

def plot_3d(
    logger: SimLogger,
    title: str = "3-D Engagement Trajectory",
    radar_range: float = 80_000.0,
    show_radar_sphere: bool = False,
    figsize: tuple = (12, 9),
) -> plt.Figure:
    """Full 3-D view of all trajectories."""
    fig = plt.figure(figsize=figsize)
    ax: Axes3D = fig.add_subplot(111, projection="3d")

    # ── Targets ───────────────────────────────────────────────────────────
    for tid in _unique_ids(logger, "target"):
        d = _entity_records(logger, "target", tid)
        if not d["x"]:
            continue
        ttype = _type_for_id(logger, tid)
        colour = _COLOURS.get(ttype, "gray")
        ax.plot(d["x"], d["y"], d["z"], color=colour, linewidth=1.5,
                label=f"{ttype} #{tid}" if tid <= 6 else None)
        # Start marker
        ax.scatter(d["x"][0], d["y"][0], d["z"][0],
                   marker="^", s=60, color=colour, zorder=5)
        # End marker
        ax.scatter(d["x"][-1], d["y"][-1], d["z"][-1],
                   marker="x", s=60, color=colour, zorder=5)

    # ── Interceptors ──────────────────────────────────────────────────────
    for iid in _unique_ids(logger, "interceptor"):
        d = _entity_records(logger, "interceptor", iid)
        if not d["x"]:
            continue
        ax.plot(d["x"], d["y"], d["z"],
                color=_COLOURS["interceptor"], linewidth=1.0,
                linestyle="--", alpha=0.7)

    # ── Intercept events ──────────────────────────────────────────────────
    for ev in logger.events:
        colour = _COLOURS["intercept_ok"] if ev.success else _COLOURS["intercept_fail"]
        ax.scatter(*ev.intercept_position, s=120, marker="*",
                   color=colour, zorder=10,
                   label="Intercept" if ev == logger.events[0] else None)

    # ── Battery ───────────────────────────────────────────────────────────
    ax.scatter(0, 0, 0, s=200, marker="D",
               color=_COLOURS["battery"], zorder=10, label="Battery")

    ax.set_xlabel("X (m) — East")
    ax.set_ylabel("Y (m) — North")
    ax.set_zlabel("Z (m) — Altitude")
    ax.set_title(title)
    _dedupe_legend(ax)
    plt.tight_layout()
    return fig


# ── 2-D composite plot ─────────────────────────────────────────────────────

def plot_2d(
    logger: SimLogger,
    title: str = "Engagement — 2D Views",
    radar_range: float = 80_000.0,
    figsize: tuple = (14, 6),
) -> plt.Figure:
    """Top-down (X-Y) and Range-Altitude (R-Z) side by side."""
    fig, (ax_top, ax_alt) = plt.subplots(1, 2, figsize=figsize)

    # ── Radar range ring (top-down) ───────────────────────────────────────
    circle = plt.Circle((0, 0), radar_range,
                         fill=False, linestyle=":",
                         color=_COLOURS["radar_ring"], linewidth=1, alpha=0.5)
    ax_top.add_patch(circle)

    for tid in _unique_ids(logger, "target"):
        d = _entity_records(logger, "target", tid)
        if not d["x"]:
            continue
        ttype = _type_for_id(logger, tid)
        colour = _COLOURS.get(ttype, "gray")
        ax_top.plot(d["x"], d["y"], color=colour, linewidth=1.5)
        ax_top.scatter(d["x"][0], d["y"][0], marker="^", s=50, color=colour)

        # Range-altitude view
        rng = [np.sqrt(x**2 + y**2) for x, y in zip(d["x"], d["y"])]
        ax_alt.plot(rng, d["z"], color=colour, linewidth=1.5,
                    label=f"{ttype} #{tid}" if tid <= 6 else None)

    for iid in _unique_ids(logger, "interceptor"):
        d = _entity_records(logger, "interceptor", iid)
        if not d["x"]:
            continue
        ax_top.plot(d["x"], d["y"],
                    color=_COLOURS["interceptor"], linewidth=0.8,
                    linestyle="--", alpha=0.6)
        rng = [np.sqrt(x**2 + y**2) for x, y in zip(d["x"], d["y"])]
        ax_alt.plot(rng, d["z"],
                    color=_COLOURS["interceptor"], linewidth=0.8,
                    linestyle="--", alpha=0.6)

    for ev in logger.events:
        col = _COLOURS["intercept_ok"] if ev.success else _COLOURS["intercept_fail"]
        p = ev.intercept_position
        ax_top.scatter(p[0], p[1], s=100, marker="*", color=col, zorder=9)
        ax_alt.scatter(np.sqrt(p[0]**2 + p[1]**2), p[2],
                       s=100, marker="*", color=col, zorder=9)

    # Battery
    ax_top.scatter(0, 0, s=150, marker="D",
                   color=_COLOURS["battery"], zorder=10, label="Battery")
    ax_alt.scatter(0, 0, s=150, marker="D",
                   color=_COLOURS["battery"], zorder=10)

    ax_top.set_xlabel("X (m) — East")
    ax_top.set_ylabel("Y (m) — North")
    ax_top.set_title("Top-Down View")
    ax_top.set_aspect("equal", "box")
    ax_top.grid(True, alpha=0.3)

    ax_alt.set_xlabel("Range (m)")
    ax_alt.set_ylabel("Altitude (m)")
    ax_alt.set_title("Range–Altitude View")
    ax_alt.grid(True, alpha=0.3)

    _add_legend_patches(ax_alt)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


# ── Timeline plot ──────────────────────────────────────────────────────────

def plot_timeline(
    logger: SimLogger,
    figsize: tuple = (12, 5),
) -> plt.Figure:
    """Speed vs time and altitude vs time for all entities."""
    fig, (ax_spd, ax_alt) = plt.subplots(1, 2, figsize=figsize)

    def speeds(d):
        return [np.sqrt(vx**2 + vy**2 + vz**2)
                for vx, vy, vz in zip(
                    [r["vx"] for r in logger._records
                     if r["entity"] == "target" and r["id"] == d],
                    [r["vy"] for r in logger._records
                     if r["entity"] == "target" and r["id"] == d],
                    [r["vz"] for r in logger._records
                     if r["entity"] == "target" and r["id"] == d],
                )]

    for tid in _unique_ids(logger, "target"):
        d = _entity_records(logger, "target", tid)
        if not d["t"]:
            continue
        ttype = _type_for_id(logger, tid)
        colour = _COLOURS.get(ttype, "gray")
        spd = [np.sqrt(vx**2+vy**2+vz**2) for vx, vy, vz in zip(
            [r["vx"] for r in logger._records if r["entity"] == "target" and r["id"] == tid],
            [r["vy"] for r in logger._records if r["entity"] == "target" and r["id"] == tid],
            [r["vz"] for r in logger._records if r["entity"] == "target" and r["id"] == tid],
        )]
        if len(spd) == len(d["t"]):
            ax_spd.plot(d["t"], spd, color=colour, linewidth=1.5,
                        label=f"{ttype} #{tid}")
        ax_alt.plot(d["t"], d["z"], color=colour, linewidth=1.5)

    for iid in _unique_ids(logger, "interceptor"):
        d = _entity_records(logger, "interceptor", iid)
        if not d["t"]:
            continue
        spd = [np.sqrt(vx**2+vy**2+vz**2) for vx, vy, vz in zip(
            [r["vx"] for r in logger._records if r["entity"] == "interceptor" and r["id"] == iid],
            [r["vy"] for r in logger._records if r["entity"] == "interceptor" and r["id"] == iid],
            [r["vz"] for r in logger._records if r["entity"] == "interceptor" and r["id"] == iid],
        )]
        if len(spd) == len(d["t"]):
            ax_spd.plot(d["t"], spd, color=_COLOURS["interceptor"],
                        linewidth=0.8, linestyle="--", alpha=0.7)
        ax_alt.plot(d["t"], d["z"],
                    color=_COLOURS["interceptor"], linewidth=0.8,
                    linestyle="--", alpha=0.7)

    # Event markers
    for ev in logger.events:
        col = _COLOURS["intercept_ok"] if ev.success else _COLOURS["intercept_fail"]
        ax_alt.axvline(ev.sim_time, color=col, linewidth=1.0, linestyle=":", alpha=0.8)

    ax_spd.set_xlabel("Time (s)")
    ax_spd.set_ylabel("Speed (m/s)")
    ax_spd.set_title("Speed vs Time")
    ax_spd.legend(fontsize=8)
    ax_spd.grid(True, alpha=0.3)

    ax_alt.set_xlabel("Time (s)")
    ax_alt.set_ylabel("Altitude (m)")
    ax_alt.set_title("Altitude vs Time")
    ax_alt.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ── Monte Carlo summary plot ───────────────────────────────────────────────

def plot_monte_carlo(
    analyzer: MonteCarloAnalyzer,
    title: str = "Monte Carlo Analysis",
    figsize: tuple = (14, 5),
) -> plt.Figure:
    """Bar charts + distributions from Monte Carlo results."""
    try:
        df = analyzer.to_dataframe()
    except ImportError:
        print("[Visualizer] pandas required for MC plot.")
        return plt.figure()

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Success-rate histogram
    axes[0].hist(df["success_rate"] * 100, bins=20,
                 color="#3498db", edgecolor="white", alpha=0.85)
    axes[0].set_xlabel("Success Rate (%)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Intercept Success Rate Distribution")
    axes[0].axvline(df["success_rate"].mean() * 100,
                    color="red", linestyle="--", label=f"Mean {df['success_rate'].mean()*100:.1f}%")
    axes[0].legend()

    # Miss-distance distribution
    valid_md = df["avg_miss_dist_m"].dropna()
    if len(valid_md) > 0:
        axes[1].hist(valid_md, bins=20,
                     color="#2ecc71", edgecolor="white", alpha=0.85)
        axes[1].set_xlabel("Avg Miss Distance (m)")
        axes[1].set_ylabel("Frequency")
        axes[1].set_title("Miss Distance Distribution")
        axes[1].axvline(valid_md.mean(), color="red", linestyle="--",
                        label=f"Mean {valid_md.mean():.1f}m")
        axes[1].legend()

    # Interceptors expended
    axes[2].hist(df["interceptors_expended"], bins=range(0, int(df["interceptors_expended"].max()) + 2),
                 color="#e74c3c", edgecolor="white", alpha=0.85, align="left")
    axes[2].set_xlabel("Interceptors Fired")
    axes[2].set_ylabel("Frequency")
    axes[2].set_title("Interceptors Expended")
    mean_exp = df["interceptors_expended"].mean()
    axes[2].axvline(mean_exp, color="navy", linestyle="--",
                    label=f"Mean {mean_exp:.1f}")
    axes[2].legend()

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


# ── Guidance comparison ────────────────────────────────────────────────────

def plot_guidance_comparison(
    logger_pn: SimLogger,
    logger_pursuit: SimLogger,
    figsize: tuple = (14, 6),
) -> plt.Figure:
    """Side-by-side Range–Altitude plot for PN vs Pure Pursuit."""
    fig, (ax_pn, ax_pp) = plt.subplots(1, 2, figsize=figsize, sharey=True)

    def _draw(ax, logger, label):
        for tid in _unique_ids(logger, "target"):
            d = _entity_records(logger, "target", tid)
            if not d["x"]:
                continue
            ttype = _type_for_id(logger, tid)
            colour = _COLOURS.get(ttype, "gray")
            rng = [np.sqrt(x**2+y**2) for x, y in zip(d["x"], d["y"])]
            ax.plot(rng, d["z"], color=colour, linewidth=1.5)

        for iid in _unique_ids(logger, "interceptor"):
            d = _entity_records(logger, "interceptor", iid)
            if not d["x"]:
                continue
            rng = [np.sqrt(x**2+y**2) for x, y in zip(d["x"], d["y"])]
            ax.plot(rng, d["z"], color=_COLOURS["interceptor"],
                    linewidth=0.9, linestyle="--", alpha=0.7)

        for ev in logger.events:
            col = _COLOURS["intercept_ok"] if ev.success else _COLOURS["intercept_fail"]
            p = ev.intercept_position
            ax.scatter(np.sqrt(p[0]**2+p[1]**2), p[2],
                       s=120, marker="*", color=col, zorder=9)

        n_ok = sum(1 for ev in logger.events if ev.success)
        n_tot = len({ev.target_id for ev in logger.events}) or 1
        ax.set_title(f"{label}  ({n_ok}/{n_tot} intercepts)", fontweight="bold")
        ax.set_xlabel("Range (m)")
        ax.set_ylabel("Altitude (m)")
        ax.grid(True, alpha=0.3)

    _draw(ax_pn, logger_pn, "Proportional Navigation")
    _draw(ax_pp, logger_pursuit, "Pure Pursuit")
    _add_legend_patches(ax_pn)
    fig.suptitle("Guidance Law Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


# ── Utility ────────────────────────────────────────────────────────────────

def _dedupe_legend(ax) -> None:
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        ax.legend(by_label.values(), by_label.keys(), fontsize=8, loc="best")


def _add_legend_patches(ax) -> None:
    patches = [
        mpatches.Patch(color=_COLOURS["ballistic"], label="Ballistic target"),
        mpatches.Patch(color=_COLOURS["cruise"],    label="Cruise target"),
        mpatches.Patch(color=_COLOURS["evasive"],   label="Evasive target"),
        mpatches.Patch(color=_COLOURS["interceptor"], label="Interceptor"),
        mpatches.Patch(color=_COLOURS["intercept_ok"], label="Intercept ✓"),
        mpatches.Patch(color=_COLOURS["intercept_fail"], label="Miss ✗"),
        mpatches.Patch(color=_COLOURS["battery"], label="Battery"),
    ]
    ax.legend(handles=patches, fontsize=7, loc="best")


def show_all() -> None:
    plt.show()


def save_figure(fig: plt.Figure, path: str, dpi: int = 150) -> None:
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"[Visualizer] Saved → {path}")
