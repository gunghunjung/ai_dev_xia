"""
main.py — CLI entry point
=========================

Usage examples
--------------
Single scenario, 3 targets, PN guidance:
    python main.py --targets 3 --duration 120 --mode single --guidance PN

Monte Carlo, 5 targets, 200 trials:
    python main.py --targets 5 --mode montecarlo --trials 200

Guidance comparison (PN vs Pure Pursuit):
    python main.py --targets 2 --mode compare

Save trajectory CSV and plots:
    python main.py --targets 3 --save-csv results/traj.csv --save-plots results/
"""
from __future__ import annotations

import argparse
import os
import sys

import yaml

# ── Bootstrap sys.path so `src` is importable from any cwd ────────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ── Config loader ──────────────────────────────────────────────────────────

def load_config(path: str | None = None) -> dict:
    if path is None:
        path = os.path.join(_ROOT, "config", "default_config.yaml")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── CLI ────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="missile_defense_sim",
        description="Surface-to-Air Missile Defence Simulation",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--targets",  "-t", type=int,   default=None,
                   help="Number of incoming targets (overrides config)")
    p.add_argument("--duration", "-d", type=float, default=None,
                   help="Simulation duration in seconds (overrides config)")
    p.add_argument("--mode",     "-m", type=str,   default="single",
                   choices=["single", "montecarlo", "compare"],
                   help="single  — one scenario\n"
                        "montecarlo — N randomised trials\n"
                        "compare — PN vs Pure Pursuit side-by-side")
    p.add_argument("--guidance", "-g", type=str,   default=None,
                   choices=["PN", "pursuit"],
                   help="Guidance law: PN (Proportional Navigation) or pursuit")
    p.add_argument("--trials",   type=int,   default=None,
                   help="Number of Monte Carlo trials (overrides config)")
    p.add_argument("--config",   type=str,   default=None,
                   help="Path to custom YAML config file")
    p.add_argument("--save-csv", type=str,   default=None,
                   help="Save trajectory / stats CSV to this path")
    p.add_argument("--save-plots", type=str, default=None,
                   help="Save all plots to this directory")
    p.add_argument("--no-plot",  action="store_true",
                   help="Suppress interactive plot display")
    p.add_argument("--seed",     type=int,   default=None,
                   help="Random seed for reproducibility")
    p.add_argument("--ecm",      action="store_true",
                   help="Enable ECM (Electronic Counter-Measures) noise")
    p.add_argument("--verbose",  "-v", action="store_true", default=True)
    p.add_argument("--quiet",    "-q", action="store_true",
                   help="Suppress detailed output")
    return p


# ── Mode handlers ──────────────────────────────────────────────────────────

def run_single(args, cfg) -> None:
    from src.simulation.sim_controller import SimulationController
    from src.visualization.visualizer import (
        plot_3d, plot_2d, plot_timeline, save_figure, show_all
    )

    verbose = not args.quiet
    sim = SimulationController(cfg, verbose=verbose)

    if verbose:
        print("\n" + "=" * 60)
        print("  Surface-to-Air Missile Defence Simulation")
        print("  Mode: Single Scenario")
        print("=" * 60)

    result, logger = sim.run_single(
        n_targets=args.targets,
        duration=args.duration,
        guidance_type=args.guidance,
        rng_seed=args.seed,
        save_csv=args.save_csv,
    )

    # ── Print results table ────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print(f"  Targets         : {result.n_targets}")
    print(f"  Intercepts      : {result.n_intercepts}  ({result.success_rate*100:.0f}%)")
    print(f"  Misses          : {result.n_misses}")
    print(f"  Interceptors    : {result.interceptors_expended}")
    print(f"  Duration        : {result.duration:.1f} s")
    if result.intercept_events:
        avg_md = result.avg_miss_distance
        avg_ft = result.avg_flight_time
        if avg_md == avg_md:   # not NaN
            print(f"  Avg miss dist   : {avg_md:.1f} m")
            print(f"  Avg flight time : {avg_ft:.1f} s")
    print("─" * 60)

    if not args.no_plot and len(logger._records) > 0:
        r_cfg = cfg["radar"]
        fig3d  = plot_3d(logger, radar_range=r_cfg["range_max"])
        fig2d  = plot_2d(logger, radar_range=r_cfg["range_max"])
        fig_tl = plot_timeline(logger)

        if args.save_plots:
            os.makedirs(args.save_plots, exist_ok=True)
            save_figure(fig3d,  os.path.join(args.save_plots, "trajectory_3d.png"))
            save_figure(fig2d,  os.path.join(args.save_plots, "trajectory_2d.png"))
            save_figure(fig_tl, os.path.join(args.save_plots, "timeline.png"))

        show_all()


def run_montecarlo(args, cfg) -> None:
    from src.simulation.sim_controller import SimulationController
    from src.visualization.visualizer import plot_monte_carlo, save_figure, show_all

    if args.trials is not None:
        cfg["simulation"]["n_trials"] = args.trials
    if args.seed is not None:
        cfg["simulation"]["random_seed"] = args.seed

    verbose = not args.quiet
    sim = SimulationController(cfg, verbose=verbose)

    if verbose:
        n_trials = cfg["simulation"]["n_trials"]
        n_tgt    = args.targets or cfg["targets"]["n_targets"]
        print("\n" + "=" * 60)
        print("  Surface-to-Air Missile Defence Simulation")
        print(f"  Mode: Monte Carlo  ({n_trials} trials × {n_tgt} targets)")
        print("=" * 60)

    analyzer = sim.run_monte_carlo(
        n_targets=args.targets,
        duration=args.duration,
        guidance_type=args.guidance,
        save_csv=args.save_csv,
    )

    if not args.no_plot:
        guidance_label = (args.guidance or cfg["guidance"]["type"]).upper()
        fig = plot_monte_carlo(
            analyzer,
            title=f"Monte Carlo — {cfg['simulation']['n_trials']} Trials  |  "
                  f"Guidance: {guidance_label}  |  "
                  f"Targets: {args.targets or cfg['targets']['n_targets']}",
        )
        if args.save_plots:
            os.makedirs(args.save_plots, exist_ok=True)
            save_figure(fig, os.path.join(args.save_plots, "monte_carlo.png"))
        show_all()


def run_compare(args, cfg) -> None:
    """
    Run two scenarios with identical seed — one with PN, one with Pure Pursuit —
    and render a side-by-side plot.
    """
    from src.simulation.sim_controller import SimulationController
    from src.visualization.visualizer import (
        plot_guidance_comparison, save_figure, show_all
    )

    seed = args.seed if args.seed is not None else 99
    verbose = not args.quiet

    if verbose:
        print("\n" + "=" * 60)
        print("  Guidance Comparison: PN  vs  Pure Pursuit")
        print("=" * 60)

    sim = SimulationController(cfg, verbose=verbose)

    print("\n[1/2] Running Proportional Navigation...")
    _, logger_pn = sim.run_single(
        n_targets=args.targets, duration=args.duration,
        guidance_type="PN", rng_seed=seed,
    )

    print("\n[2/2] Running Pure Pursuit...")
    _, logger_pursuit = sim.run_single(
        n_targets=args.targets, duration=args.duration,
        guidance_type="PURSUIT", rng_seed=seed,
    )

    if not args.no_plot:
        fig = plot_guidance_comparison(logger_pn, logger_pursuit)
        if args.save_plots:
            os.makedirs(args.save_plots, exist_ok=True)
            save_figure(fig, os.path.join(args.save_plots, "guidance_comparison.png"))
        show_all()


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.ecm:
        cfg["radar"]["ecm_enabled"] = True
        if not args.quiet:
            print("[Config] ECM mode enabled — radar noise multiplied by "
                  f"{cfg['radar']['ecm_noise_multiplier']}×")

    try:
        import matplotlib
        if args.no_plot:
            matplotlib.use("Agg")
        else:
            matplotlib.use("TkAgg" if sys.platform == "win32" else "TkAgg")
    except Exception:
        pass

    mode = args.mode.lower()
    if mode == "single":
        run_single(args, cfg)
    elif mode == "montecarlo":
        run_montecarlo(args, cfg)
    elif mode == "compare":
        run_compare(args, cfg)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
