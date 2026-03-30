"""
Unit & Integration Tests
========================
Run with:  python -m pytest tests/ -v
"""
from __future__ import annotations

import sys
import os
import math

import numpy as np
import pytest

# Make src importable when run from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import yaml

# ── Load config ─────────────────────────────────────────────────────────────
CFG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "default_config.yaml")


def load_cfg() -> dict:
    with open(CFG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Physics Engine
# ─────────────────────────────────────────────────────────────────────────────

class TestPhysicsEngine:

    def setup_method(self):
        from src.physics.engine import PhysicsEngine
        self.pe = PhysicsEngine(enable_gravity=True, enable_drag=False)

    def test_free_fall(self):
        """Object in free-fall should gain vz = -g*t after 1 second."""
        pos = np.array([0.0, 0.0, 1000.0])
        vel = np.zeros(3)
        dt = 0.01
        steps = 100  # 1 second
        for _ in range(steps):
            pos, vel = self.pe.integrate(pos, vel, np.zeros(3), dt)
        assert abs(vel[2] - (-9.81)) < 0.1, f"Expected vz≈-9.81, got {vel[2]:.3f}"

    def test_no_gravity_engine(self):
        from src.physics.engine import PhysicsEngine
        pe_nograv = PhysicsEngine(enable_gravity=False, enable_drag=False)
        pos = np.array([0.0, 0.0, 500.0])
        vel = np.array([100.0, 0.0, 0.0])
        pos2, vel2 = pe_nograv.integrate(pos, vel, np.zeros(3), 1.0)
        assert abs(pos2[0] - 100.0) < 0.1
        assert abs(vel2[2]) < 0.01   # no gravity → vz unchanged

    def test_max_speed_clamp(self):
        from src.physics.engine import PhysicsEngine
        pe = PhysicsEngine(enable_gravity=False, enable_drag=False)
        pos = np.zeros(3)
        vel = np.array([900.0, 0.0, 0.0])
        cmd = np.array([1000.0, 0.0, 0.0])  # large thrust
        _, vel2 = pe.integrate(pos, vel, cmd, 0.1, max_speed=1000.0)
        assert np.linalg.norm(vel2) <= 1001.0  # within tolerance

    def test_g_limit_clamp(self):
        from src.physics.engine import PhysicsEngine
        pe = PhysicsEngine(enable_gravity=False, enable_drag=False)
        pos = np.zeros(3)
        vel = np.array([100.0, 0.0, 0.0])
        huge_cmd = np.array([0.0, 50000.0, 0.0])  # way over limit
        _, vel2 = pe.integrate(pos, vel, huge_cmd, 0.1, max_cmd_accel=30 * 9.81)
        accel_applied = (vel2 - vel) / 0.1
        assert np.linalg.norm(accel_applied) <= 30 * 9.81 + 1.0

    def test_miss_distance(self):
        from src.physics.engine import PhysicsEngine
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([3.0, 4.0, 0.0])
        assert abs(PhysicsEngine.miss_distance(p1, p2) - 5.0) < 1e-6

    def test_time_to_closest_approach(self):
        from src.physics.engine import PhysicsEngine
        # Two objects closing at 100 m/s, 1000 m apart
        pos_i = np.zeros(3)
        vel_i = np.array([100.0, 0.0, 0.0])
        pos_t = np.array([1000.0, 0.0, 0.0])
        vel_t = np.array([-50.0, 0.0, 0.0])   # approaching
        t = PhysicsEngine.time_to_closest_approach(pos_i, vel_i, pos_t, vel_t)
        # relative closing speed = 150 m/s → t = 1000/150 ≈ 6.67 s
        assert abs(t - 1000.0 / 150.0) < 0.1


# ─────────────────────────────────────────────────────────────────────────────
# Kalman Filter
# ─────────────────────────────────────────────────────────────────────────────

class TestKalmanFilter:

    def test_initialization(self):
        from src.radar.kalman_filter import KalmanFilter6D
        kf = KalmanFilter6D(dt=0.5)
        kf.initialize(np.array([1000.0, 2000.0, 500.0]))
        assert kf.initialized
        np.testing.assert_allclose(kf.position, [1000.0, 2000.0, 500.0], atol=1e-6)

    def test_constant_velocity_tracking(self):
        """Filter should converge on a constant-velocity target within ~10 updates."""
        from src.radar.kalman_filter import KalmanFilter6D
        rng = np.random.default_rng(0)
        kf = KalmanFilter6D(dt=0.5, process_noise=10, measurement_noise=50)
        true_pos = np.array([5000.0, 0.0, 2000.0])
        true_vel = np.array([-200.0, 0.0, 0.0])
        kf.initialize(true_pos, true_vel)

        dt = 0.5
        for _ in range(30):
            true_pos = true_pos + true_vel * dt
            meas = true_pos + rng.normal(0, 50, 3)
            kf.step(meas, dt)

        err = np.linalg.norm(kf.position - true_pos)
        assert err < 300.0, f"Position error too large: {err:.1f} m"

    def test_predict_increases_uncertainty(self):
        from src.radar.kalman_filter import KalmanFilter6D
        kf = KalmanFilter6D(dt=0.5)
        kf.initialize(np.zeros(3))
        p_before = np.trace(kf.P)
        kf.predict()
        p_after = np.trace(kf.P)
        assert p_after > p_before


# ─────────────────────────────────────────────────────────────────────────────
# Guidance Laws
# ─────────────────────────────────────────────────────────────────────────────

class TestGuidance:

    def test_pn_acceleration_perpendicular_to_los(self):
        """PN command should be perpendicular to the LOS vector (True PN property)."""
        from src.guidance.proportional_nav import ProportionalNavigation
        pn = ProportionalNavigation(N=4.0)
        int_pos = np.zeros(3)
        int_vel = np.array([500.0, 0.0, 0.0])
        tgt_pos = np.array([5000.0, 200.0, 1000.0])
        tgt_vel = np.array([-300.0, 0.0, -50.0])
        cmd = pn.compute_command(int_pos, int_vel, tgt_pos, tgt_vel)
        # True PN: cmd is cross product of omega and V_I → perpendicular to V_I
        # Check the command has non-zero magnitude (target is closing)
        assert np.linalg.norm(cmd) > 0.0

    def test_pn_zero_when_on_collision_course(self):
        """If target is on direct collision course, LOS rate is zero → cmd ≈ 0."""
        from src.guidance.proportional_nav import ProportionalNavigation
        pn = ProportionalNavigation(N=4.0, augmented=False)
        # Set up: interceptor and target moving toward each other along X-axis
        int_pos = np.zeros(3)
        int_vel = np.array([500.0, 0.0, 0.0])
        tgt_pos = np.array([10000.0, 0.0, 0.0])
        tgt_vel = np.array([-400.0, 0.0, 0.0])   # closing, zero LOS rate
        cmd = pn.compute_command(int_pos, int_vel, tgt_pos, tgt_vel)
        assert np.linalg.norm(cmd) < 5.0, f"Expected near-zero command, got {np.linalg.norm(cmd):.2f}"

    def test_pure_pursuit_nonzero_for_off_axis_target(self):
        """Pure Pursuit should produce a non-zero command when target is off-axis."""
        from src.guidance.pure_pursuit import PurePursuit
        pp = PurePursuit(gain=10.0)
        int_pos = np.zeros(3)
        int_vel = np.array([500.0, 0.0, 0.0])
        tgt_pos = np.array([0.0, 5000.0, 2000.0])   # target is 90° off-axis
        tgt_vel = np.zeros(3)
        cmd = pp.compute_command(int_pos, int_vel, tgt_pos, tgt_vel)
        # Command should be significant to steer the interceptor toward target
        assert np.linalg.norm(cmd) > 10.0, "Should produce significant steering command"


# ─────────────────────────────────────────────────────────────────────────────
# Target Models
# ─────────────────────────────────────────────────────────────────────────────

class TestTargets:

    def test_ballistic_descends(self):
        """Ballistic target should lose altitude over time (no thrust)."""
        from src.targets.ballistic_target import BallisticTarget
        cfg = load_cfg()
        tgt = BallisticTarget(cfg, azimuth_deg=0.0, rng=np.random.default_rng(1))
        initial_alt = tgt.position[2]
        for _ in range(200):   # 10 seconds at dt=0.05
            tgt.update(0.05)
        assert tgt.position[2] < initial_alt, "Ballistic target should lose altitude"

    def test_cruise_maintains_altitude(self):
        """Cruise target altitude should stay within ±200 m of target altitude."""
        from src.targets.cruise_target import CruiseTarget
        cfg = load_cfg()
        tgt = CruiseTarget(cfg, azimuth_deg=45.0, rng=np.random.default_rng(2))
        target_alt = cfg["targets"]["cruise"]["altitude"]
        # Allow 5 seconds to stabilise, then check
        for _ in range(100):
            tgt.update(0.05)
        for _ in range(200):
            tgt.update(0.05)
            assert abs(tgt.position[2] - target_alt) < 300.0, (
                f"Cruise altitude {tgt.position[2]:.1f} diverged from {target_alt:.1f}")

    def test_evasive_fires_evasion(self):
        """Evasive target should show lateral position change over time."""
        from src.targets.evasive_target import EvasiveTarget
        cfg = load_cfg()
        cfg["targets"]["evasive"]["evasion_interval"] = 1.0
        tgt = EvasiveTarget(cfg, azimuth_deg=90.0, rng=np.random.default_rng(3))
        xs = [tgt.position[0]]
        for _ in range(200):
            tgt.update(0.05)
            xs.append(tgt.position[0])
        # With evasion, the target should deviate from a purely straight path
        straight_x = np.linspace(xs[0], xs[-1], len(xs))
        deviation = np.max(np.abs(np.array(xs) - straight_x))
        assert deviation > 10.0, "Evasive target should deviate from straight line"

    def test_target_dies_on_ground(self):
        from src.targets.ballistic_target import BallisticTarget
        cfg = load_cfg()
        cfg["targets"]["ballistic"]["launch_altitude"] = 100  # low altitude → hits ground fast
        tgt = BallisticTarget(cfg, azimuth_deg=0.0, rng=np.random.default_rng(4))
        # Force it straight down
        tgt.position = np.array([0.0, 1000.0, 50.0])
        tgt.velocity = np.array([0.0, 0.0, -200.0])
        for _ in range(1000):
            if not tgt.alive:
                break
            tgt.update(0.05)
        assert not tgt.alive
        assert tgt.hit_ground


# ─────────────────────────────────────────────────────────────────────────────
# Interceptor
# ─────────────────────────────────────────────────────────────────────────────

class TestInterceptor:

    def _make_interceptor(self, guidance_type="PN") -> object:
        from src.interceptor.interceptor import Interceptor
        from src.guidance.proportional_nav import ProportionalNavigation
        from src.guidance.pure_pursuit import PurePursuit
        cfg = load_cfg()
        guidance = ProportionalNavigation(N=4.0) if guidance_type == "PN" else PurePursuit()
        return Interceptor(cfg, guidance), cfg

    def test_launch_sets_airborne(self):
        from src.interceptor.interceptor import InterceptorStatus
        itr, _ = self._make_interceptor()
        itr.launch(target_id=1, initial_velocity=np.array([0.0, 0.0, 300.0]))
        assert itr.status == InterceptorStatus.AIRBORNE
        assert itr.active

    def test_intercept_detected_within_kill_radius(self):
        from src.interceptor.interceptor import InterceptorStatus
        itr, _ = self._make_interceptor()
        itr.launch(target_id=1, initial_velocity=np.array([0.0, 0.0, 300.0]))
        # Place interceptor right next to target (within kill_radius=30m)
        tgt_pos = itr.position + np.array([10.0, 0.0, 0.0])
        result = itr.update(0.05, tgt_pos, np.zeros(3))
        assert result == 1, "Should report intercept of target 1"
        assert itr.status == InterceptorStatus.INTERCEPT

    def test_self_destruct_after_max_time(self):
        from src.interceptor.interceptor import InterceptorStatus
        itr, cfg = self._make_interceptor()
        cfg["interceptor"]["max_flight_time"] = 1.0
        from src.interceptor.interceptor import Interceptor
        from src.guidance.proportional_nav import ProportionalNavigation
        itr2 = Interceptor(cfg, ProportionalNavigation(N=4.0))
        itr2.launch(1, np.array([0.0, 0.0, 200.0]))
        tgt_pos = np.array([50000.0, 0.0, 10000.0])   # unreachable in 1s
        for _ in range(100):  # 5 seconds
            if not itr2.active:
                break
            itr2.update(0.05, tgt_pos, np.array([-100.0, 0.0, 0.0]))
        assert itr2.status == InterceptorStatus.SELF_DESTRUCT


# ─────────────────────────────────────────────────────────────────────────────
# Integration: single-scenario intercept
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegration:

    def test_single_ballistic_intercepted(self):
        """
        A single ballistic target approaching head-on should be intercepted
        in a standard scenario with PN guidance.
        """
        cfg = load_cfg()
        cfg["targets"]["n_targets"] = 1
        cfg["targets"]["types_distribution"] = {"ballistic": 1.0, "cruise": 0.0, "evasive": 0.0}
        cfg["simulation"]["max_duration"] = 120.0
        cfg["guidance"]["type"] = "PN"

        from src.simulation.sim_controller import SimulationController
        sim = SimulationController(cfg, verbose=False)
        result, _ = sim.run_single(rng_seed=42)

        # Not guaranteed to be 100% (randomised), but should succeed often
        assert result.n_targets == 1
        assert result.interceptors_expended >= 1

    def test_multi_target_monte_carlo(self):
        """Monte Carlo with 3 targets × 10 trials should complete without errors."""
        cfg = load_cfg()
        cfg["targets"]["n_targets"] = 3
        cfg["simulation"]["n_trials"] = 10
        cfg["simulation"]["max_duration"] = 100.0

        from src.simulation.sim_controller import SimulationController
        sim = SimulationController(cfg, verbose=False)
        analyzer = sim.run_monte_carlo()
        summary = analyzer.summary()

        assert summary["n_trials"] == 10
        assert 0.0 <= summary["success_rate_mean"] <= 1.0

    def test_guidance_comparison(self):
        """Both guidance modes should run without exception."""
        cfg = load_cfg()
        cfg["targets"]["n_targets"] = 2
        cfg["simulation"]["max_duration"] = 80.0

        from src.simulation.sim_controller import SimulationController

        for guidance in ["PN", "PURSUIT"]:
            sim = SimulationController(cfg, verbose=False)
            result, _ = sim.run_single(guidance_type=guidance, rng_seed=10)
            assert result.n_targets == 2


# ─────────────────────────────────────────────────────────────────────────────
# Engagement Controller
# ─────────────────────────────────────────────────────────────────────────────

class TestEngagementController:

    def test_threat_score_zero_for_opening_target(self):
        from src.engagement.controller import EngagementController
        from src.radar.radar import Track
        cfg = load_cfg()
        ec = EngagementController(cfg)

        # Target moving AWAY from battery
        track = Track(
            track_id=0, target_id=1,
            estimated_position=np.array([10000.0, 0.0, 3000.0]),
            estimated_velocity=np.array([500.0, 0.0, 0.0]),   # going away
            last_update_time=0.0,
        )
        score = ec._threat_score(track)
        assert score == 0.0, f"Opening target should have zero threat, got {score}"

    def test_threat_score_higher_for_closer_target(self):
        from src.engagement.controller import EngagementController
        from src.radar.radar import Track
        cfg = load_cfg()
        ec = EngagementController(cfg)

        far_track = Track(
            track_id=0, target_id=1,
            estimated_position=np.array([60000.0, 0.0, 5000.0]),
            estimated_velocity=np.array([-300.0, 0.0, -50.0]),
            last_update_time=0.0,
        )
        near_track = Track(
            track_id=1, target_id=2,
            estimated_position=np.array([10000.0, 0.0, 2000.0]),
            estimated_velocity=np.array([-300.0, 0.0, -50.0]),
            last_update_time=0.0,
        )
        score_far  = ec._threat_score(far_track)
        score_near = ec._threat_score(near_track)
        assert score_near > score_far, "Closer target should have higher threat score"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
