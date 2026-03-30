import numpy as np
from matplotlib.figure import Figure

from ui.base_module import ModulePage, apply_dark_axes
from engine.mechanics import simulate_projectile


class ProjectileModule(ModulePage):
    TIMER_INTERVAL = 40

    def __init__(self, parent=None):
        self._traj_x = None
        self._traj_y = None
        self._anim_idx = 0
        super().__init__("포물선 운동", "#ff6b6b", parent)

    def setup_module(self):
        self.add_ctrl_label("발사 파라미터")
        self.sl_v0 = self.create_slider("초기속도 v₀ (m/s)", 1.0, 100.0, 30.0, decimals=1)
        self.sl_theta = self.create_slider("발사각 θ (°)", 0.0, 90.0, 45.0, decimals=1)
        self.sl_k = self.create_slider("공기저항 k", 0.0, 0.5, 0.0, decimals=3)

        self.sl_v0.connect(self._on_param_changed)
        self.sl_theta.connect(self._on_param_changed)
        self.sl_k.connect(self._on_param_changed)

        self.add_ctrl_stretch()
        self._simulate()

    def _simulate(self):
        v0 = self.sl_v0.value()
        theta = self.sl_theta.value()
        k = self.sl_k.value()
        xs, ys = simulate_projectile(v0, theta, k)
        self._traj_x = xs
        self._traj_y = ys
        self._anim_idx = 0

    def _on_param_changed(self, *args):
        self._simulate()
        self.update_plot()

    def update_plot(self):
        self.ax.cla()
        apply_dark_axes(self.ax)

        if self._traj_x is None or len(self._traj_x) == 0:
            self.canvas.draw_idle()
            return

        xs, ys = self._traj_x, self._traj_y
        idx = min(self._anim_idx, len(xs) - 1)

        # 전체 궤적 (희미하게)
        self.ax.plot(xs, ys, color='#ff6b6b', linewidth=1.5, alpha=0.4, label='전체 궤적')

        # 현재까지의 궤적
        self.ax.plot(xs[:idx + 1], ys[:idx + 1], color='#ff6b6b', linewidth=2, label='현재 궤적')

        # 현재 위치 (공)
        self.ax.scatter([xs[idx]], [ys[idx]], color='#ffd700', s=120, zorder=5, label='공')

        # 지면
        max_x = max(xs) * 1.05
        self.ax.axhline(y=0, color='#2a2a45', linewidth=1.5)
        self.ax.fill_between([0, max_x], [-0.5, -0.5], [0, 0], color='#1a1a30', alpha=0.5)

        self.ax.set_xlim(-1, max_x)
        y_max = max(ys) * 1.15 + 1
        self.ax.set_ylim(-1, y_max)
        self.ax.set_xlabel("x (m)")
        self.ax.set_ylabel("y (m)")
        self.ax.set_title("포물선 운동 시뮬레이션")
        self.ax.legend(facecolor='#1e1e2e', edgecolor='#2a2a45', labelcolor='#e0e0ff', fontsize=9)
        self.ax.grid(True, color='#1a1a30', linewidth=0.5, alpha=0.8)
        self.canvas.draw_idle()

        # 현재 속도 추정 (간단히 위치 차이)
        if idx > 0:
            dt = 0.01
            vx = (xs[idx] - xs[idx - 1]) / dt if idx > 0 else 0
            vy = (ys[idx] - ys[idx - 1]) / dt if idx > 0 else 0
            speed = np.sqrt(vx ** 2 + vy ** 2)
        else:
            speed = self.sl_v0.value()

        max_h = float(np.max(ys))
        range_x = float(xs[-1])

        self.set_info([
            ("초기속도", f"{self.sl_v0.value():.1f} m/s"),
            ("발사각", f"{self.sl_theta.value():.1f}°"),
            ("공기저항", f"{self.sl_k.value():.3f}"),
            ("현재 x", f"{xs[idx]:.2f} m"),
            ("현재 y", f"{ys[idx]:.2f} m"),
            ("최대 높이", f"{max_h:.2f} m"),
            ("사거리", f"{range_x:.2f} m"),
        ])

    def on_animate(self, frame: int):
        if self._traj_x is None:
            return
        step = max(1, len(self._traj_x) // 80)
        self._anim_idx = min(self._anim_idx + step, len(self._traj_x) - 1)
        self.update_plot()
        if self._anim_idx >= len(self._traj_x) - 1:
            self._toggle_animation()

    def on_reset(self):
        self._anim_idx = 0
