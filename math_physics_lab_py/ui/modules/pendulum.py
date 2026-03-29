import numpy as np
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from ui.base_module import ModulePage, apply_dark_axes
from engine.mechanics import simulate_pendulum


class PendulumModule(ModulePage):
    TIMER_INTERVAL = 40

    def __init__(self, parent=None):
        self._thetas = None
        self._omegas = None
        self._anim_idx = 0
        super().__init__("진자 운동", "#a8edea", parent)

    def _create_figure(self):
        """2개 subplot: 진자 애니메이션 + 위상 공간"""
        fig = Figure(facecolor='#0a0a0f')
        gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1],
                      left=0.08, right=0.97, top=0.92, bottom=0.10, wspace=0.35)
        ax_pend = fig.add_subplot(gs[0])
        ax_phase = fig.add_subplot(gs[1])
        apply_dark_axes(ax_pend)
        apply_dark_axes(ax_phase)
        self.ax_phase = ax_phase
        return fig, ax_pend  # self.ax = ax_pend

    def setup_module(self):
        self.add_ctrl_label("진자 파라미터")
        self.sl_L = self.create_slider("줄 길이 L (m)", 0.1, 3.0, 1.0, decimals=2)
        self.sl_theta0 = self.create_slider("초기각도 θ₀ (°)", 5.0, 170.0, 45.0, decimals=1)
        self.sl_b = self.create_slider("감쇠 계수 b", 0.0, 2.0, 0.1, decimals=2)

        self.sl_L.connect(self._on_param_changed)
        self.sl_theta0.connect(self._on_param_changed)
        self.sl_b.connect(self._on_param_changed)

        self.add_ctrl_stretch()
        self._simulate()

    def _simulate(self):
        L = self.sl_L.value()
        theta0 = self.sl_theta0.value()
        b = self.sl_b.value()
        self._thetas, self._omegas = simulate_pendulum(L, theta0, b)
        self._anim_idx = 0

    def _on_param_changed(self, *args):
        self._simulate()
        self.update_plot()

    def update_plot(self):
        if self._thetas is None:
            return

        idx = min(self._anim_idx, len(self._thetas) - 1)
        theta = self._thetas[idx]
        omega = self._omegas[idx]
        L = self.sl_L.value()

        # --- 진자 서브플롯 ---
        self.ax.cla()
        apply_dark_axes(self.ax)

        # 피벗
        px, py = 0.0, 0.0
        # 추 위치
        bx = L * np.sin(theta)
        by = -L * np.cos(theta)

        # 실
        self.ax.plot([px, bx], [py, by], color='#a8edea', linewidth=2.5)
        # 피벗 점
        self.ax.scatter([px], [py], color='#888aaa', s=60, zorder=5)
        # 추 (원)
        circle = plt_circle(bx, by, 0.06 * L, '#a8edea')
        self.ax.add_patch(circle)

        # 궤적 (trailing) — LineCollection으로 한 번에 그리기
        trail_len = 40
        start = max(0, idx - trail_len)
        trail_thetas = self._thetas[start:idx + 1]
        trail_x = L * np.sin(trail_thetas)
        trail_y = -L * np.cos(trail_thetas)
        if len(trail_x) > 1:
            from matplotlib.collections import LineCollection
            points = np.array([trail_x, trail_y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            n = len(segments)
            colors = [(0.66, 0.93, 0.92, float(a))
                      for a in np.linspace(0.05, 0.75, n)]
            lc = LineCollection(segments, colors=colors, linewidths=1.5)
            self.ax.add_collection(lc)

        self.ax.set_xlim(-L * 1.3, L * 1.3)
        self.ax.set_ylim(-L * 1.4, L * 0.4)
        self.ax.set_aspect('equal')
        self.ax.set_title("진자 운동")
        self.ax.grid(True, color='#1a1a30', linewidth=0.5, alpha=0.5)

        # --- 위상 공간 ---
        self.ax_phase.cla()
        apply_dark_axes(self.ax_phase)
        self.ax_phase.plot(
            np.degrees(self._thetas), self._omegas,
            color='#6c63ff', linewidth=1.0, alpha=0.5
        )
        self.ax_phase.scatter(
            [np.degrees(theta)], [omega],
            color='#ffd700', s=60, zorder=5
        )
        self.ax_phase.set_xlabel("θ (°)")
        self.ax_phase.set_ylabel("ω (rad/s)")
        self.ax_phase.set_title("위상 공간")
        self.ax_phase.grid(True, color='#1a1a30', linewidth=0.5, alpha=0.5)
        self.canvas.draw_idle()

        # 소각도 주기 근사
        g = 9.81
        T_approx = 2 * np.pi * np.sqrt(L / g)
        self.set_info([
            ("줄 길이", f"{L:.2f} m"),
            ("초기각도", f"{self.sl_theta0.value():.1f}°"),
            ("감쇠 계수", f"{self.sl_b.value():.2f}"),
            ("현재 각도 θ", f"{np.degrees(theta):.2f}°"),
            ("현재 각속도 ω", f"{omega:.3f} rad/s"),
            ("소각도 근사 주기", f"{T_approx:.3f} s"),
        ])

    def on_animate(self, frame: int):
        if self._thetas is None:
            return
        self._anim_idx = (self._anim_idx + 2) % len(self._thetas)
        self.update_plot()

    def on_reset(self):
        self._anim_idx = 0


def plt_circle(cx, cy, r, color):
    """matplotlib Circle 패치 반환"""
    import matplotlib.patches as mpatches
    return mpatches.Circle((cx, cy), r, color=color, zorder=5)
