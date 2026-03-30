import numpy as np
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from ui.base_module import ModulePage, apply_dark_axes
from engine.mechanics import simulate_spring


class SpringMassModule(ModulePage):
    TIMER_INTERVAL = 40

    def __init__(self, parent=None):
        self._xs = None
        self._vs = None
        self._anim_idx = 0
        super().__init__("용수철 진동", "#ffecd2", parent)

    def _create_figure(self):
        fig = Figure(facecolor='#0a0a0f')
        gs = GridSpec(2, 1, figure=fig, height_ratios=[1.5, 1],
                      left=0.10, right=0.97, top=0.93, bottom=0.08, hspace=0.45)
        ax_spring = fig.add_subplot(gs[0])
        ax_xt = fig.add_subplot(gs[1])
        apply_dark_axes(ax_spring)
        apply_dark_axes(ax_xt)
        self.ax_xt = ax_xt
        return fig, ax_spring

    def setup_module(self):
        self.add_ctrl_label("용수철 파라미터")
        self.sl_k = self.create_slider("스프링 상수 k", 0.5, 20.0, 5.0, decimals=1)
        self.sl_m = self.create_slider("질량 m (kg)", 0.1, 5.0, 1.0, decimals=2)
        self.sl_b = self.create_slider("감쇠 계수 b", 0.0, 4.0, 0.2, decimals=2)
        self.sl_x0 = self.create_slider("초기 변위 x₀ (m)", 0.1, 3.0, 1.0, decimals=2)

        self.sl_k.connect(self._on_param_changed)
        self.sl_m.connect(self._on_param_changed)
        self.sl_b.connect(self._on_param_changed)
        self.sl_x0.connect(self._on_param_changed)

        self.add_ctrl_stretch()
        self._simulate()

    def _simulate(self):
        k = self.sl_k.value()
        m = self.sl_m.value()
        b = self.sl_b.value()
        x0 = self.sl_x0.value()
        self._xs, self._vs = simulate_spring(k, m, b, x0)
        self._anim_idx = 0

    def _on_param_changed(self, *args):
        self._simulate()
        self.update_plot()

    def _draw_spring(self, ax, x_mass, color='#ffecd2'):
        """간단한 용수철 그리기 (지그재그)"""
        wall_x = -2.5
        n_coils = 8
        coil_xs = np.linspace(wall_x, x_mass - 0.3, n_coils * 4)
        coil_ys = np.zeros_like(coil_xs)
        for i in range(len(coil_xs)):
            t = i / (len(coil_xs) - 1) * n_coils * 2 * np.pi
            if i % 4 == 1:
                coil_ys[i] = 0.2
            elif i % 4 == 3:
                coil_ys[i] = -0.2
        ax.plot(coil_xs, coil_ys, color=color, linewidth=2.0, zorder=3)
        # 벽
        ax.axvline(x=wall_x, color='#888aaa', linewidth=3, ymin=0.3, ymax=0.7)

    def update_plot(self):
        if self._xs is None:
            return

        idx = min(self._anim_idx, len(self._xs) - 1)
        x_cur = self._xs[idx]

        # --- 용수철 애니메이션 서브플롯 ---
        self.ax.cla()
        apply_dark_axes(self.ax)

        self._draw_spring(self.ax, x_cur)

        # 질량 블록
        block_w, block_h = 0.5, 0.4
        import matplotlib.patches as mpatches
        rect = mpatches.FancyBboxPatch(
            (x_cur - block_w / 2, -block_h / 2), block_w, block_h,
            boxstyle="round,pad=0.02",
            facecolor='#ffecd2', edgecolor='#f7971e', linewidth=2, zorder=4
        )
        self.ax.add_patch(rect)

        # 바닥
        self.ax.axhline(y=-0.5, color='#2a2a45', linewidth=1)
        # 평형 위치
        self.ax.axvline(x=0, color='#43e97b', linewidth=0.8, linestyle='--', alpha=0.5, label='평형 위치')

        self.ax.set_xlim(-3.5, 4.0)
        self.ax.set_ylim(-1.0, 1.0)
        self.ax.set_xlabel("x (m)")
        self.ax.set_title("용수철-질량 시스템")
        self.ax.legend(facecolor='#1e1e2e', edgecolor='#2a2a45', labelcolor='#e0e0ff', fontsize=9)
        self.ax.set_yticks([])

        # --- x-t 그래프 ---
        self.ax_xt.cla()
        apply_dark_axes(self.ax_xt)

        dt = 0.02
        ts = np.arange(len(self._xs)) * dt
        display_len = min(len(self._xs), 500)
        start = max(0, idx - display_len)
        self.ax_xt.plot(ts[start:idx + 1], self._xs[start:idx + 1],
                        color='#ffecd2', linewidth=1.5)
        self.ax_xt.scatter([ts[idx]], [self._xs[idx]], color='#ffd700', s=50, zorder=5)
        self.ax_xt.axhline(y=0, color='#2a2a45', linewidth=0.8, linestyle='--')
        self.ax_xt.set_xlabel("t (s)")
        self.ax_xt.set_ylabel("x (m)")
        self.ax_xt.set_title("변위-시간 그래프")
        self.ax_xt.grid(True, color='#1a1a30', linewidth=0.5, alpha=0.8)
        self.canvas.draw_idle()

        # 정보
        k = self.sl_k.value()
        m = self.sl_m.value()
        b = self.sl_b.value()
        omega0 = np.sqrt(k / m)
        zeta = b / (2 * np.sqrt(k * m))
        if zeta < 1:
            damp_type = "부족감쇠 (진동)"
        elif abs(zeta - 1) < 0.01:
            damp_type = "임계감쇠"
        else:
            damp_type = "과감쇠"

        self.set_info([
            ("스프링 상수 k", f"{k:.1f} N/m"),
            ("질량 m", f"{m:.2f} kg"),
            ("감쇠 계수 b", f"{b:.2f}"),
            ("고유 각진동수 ω₀", f"{omega0:.3f} rad/s"),
            ("감쇠비 ζ", f"{zeta:.3f}"),
            ("감쇠 종류", damp_type),
            ("현재 변위 x", f"{x_cur:.3f} m"),
        ])

    def on_animate(self, frame: int):
        if self._xs is None:
            return
        self._anim_idx = (self._anim_idx + 2) % len(self._xs)
        self.update_plot()

    def on_reset(self):
        self._anim_idx = 0
