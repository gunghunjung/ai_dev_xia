import numpy as np
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from ui.base_module import ModulePage, apply_dark_axes
from engine.waves import wave


class WaveInterferenceModule(ModulePage):
    TIMER_INTERVAL = 50

    def __init__(self, parent=None):
        self._time = 0.0
        super().__init__("파동 간섭", "#c471ed", parent)

    def _create_figure(self):
        fig = Figure(facecolor='#0a0a0f')
        gs = GridSpec(3, 1, figure=fig, hspace=0.55,
                      top=0.93, bottom=0.08, left=0.10, right=0.97)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        ax3 = fig.add_subplot(gs[2])
        for ax in [ax1, ax2, ax3]:
            apply_dark_axes(ax)
        self.ax2 = ax2
        self.ax3 = ax3
        return fig, ax1

    def setup_module(self):
        self.add_ctrl_label("파동 1")
        self.sl_f1 = self.create_slider("주파수 f₁ (Hz)", 0.1, 5.0, 1.0, decimals=2)
        self.sl_a1 = self.create_slider("진폭 A₁", 0.1, 2.0, 1.0, decimals=2)
        self.sl_f1.connect(self._on_param_changed)
        self.sl_a1.connect(self._on_param_changed)

        self.add_separator()
        self.add_ctrl_label("파동 2")
        self.sl_f2 = self.create_slider("주파수 f₂ (Hz)", 0.1, 5.0, 1.2, decimals=2)
        self.sl_a2 = self.create_slider("진폭 A₂", 0.1, 2.0, 1.0, decimals=2)
        self.sl_f2.connect(self._on_param_changed)
        self.sl_a2.connect(self._on_param_changed)

        self.add_ctrl_stretch()

    def _on_param_changed(self, *args):
        self.update_plot()

    def update_plot(self):
        f1 = self.sl_f1.value()
        f2 = self.sl_f2.value()
        a1 = self.sl_a1.value()
        a2 = self.sl_a2.value()
        t = self._time

        x = np.linspace(0, 10, 500)
        y1 = wave(x, t, f1, a1)
        y2 = wave(x, t, f2, a2)
        y_sum = y1 + y2

        # 파동 1
        self.ax.cla()
        apply_dark_axes(self.ax)
        self.ax.plot(x, y1, color='#6c63ff', linewidth=1.8, label=f'파동 1 (f={f1:.2f}Hz, A={a1:.2f})')
        self.ax.set_ylim(-2.2, 2.2)
        self.ax.set_ylabel("진폭")
        self.ax.set_title("파동 1")
        self.ax.legend(facecolor='#1e1e2e', edgecolor='#2a2a45', labelcolor='#e0e0ff', fontsize=8, loc='upper right')
        self.ax.axhline(0, color='#2a2a45', linewidth=0.8)
        self.ax.grid(True, color='#1a1a30', linewidth=0.5, alpha=0.6)

        # 파동 2
        self.ax2.cla()
        apply_dark_axes(self.ax2)
        self.ax2.plot(x, y2, color='#43e97b', linewidth=1.8, label=f'파동 2 (f={f2:.2f}Hz, A={a2:.2f})')
        self.ax2.set_ylim(-2.2, 2.2)
        self.ax2.set_ylabel("진폭")
        self.ax2.set_title("파동 2")
        self.ax2.legend(facecolor='#1e1e2e', edgecolor='#2a2a45', labelcolor='#e0e0ff', fontsize=8, loc='upper right')
        self.ax2.axhline(0, color='#2a2a45', linewidth=0.8)
        self.ax2.grid(True, color='#1a1a30', linewidth=0.5, alpha=0.6)

        # 합성파
        self.ax3.cla()
        apply_dark_axes(self.ax3)
        max_amp = a1 + a2
        self.ax3.plot(x, y_sum, color='#c471ed', linewidth=2.0, label='합성파 (맥놀이)')
        # 맥놀이 포락선
        df = abs(f1 - f2)
        if df > 0.01:
            env_freq = df / 2
            env = (a1 + a2) * np.cos(2 * np.pi * env_freq * (x - t))
            self.ax3.plot(x, env, color='#ffd700', linewidth=0.8, linestyle='--', alpha=0.6, label='포락선')
            self.ax3.plot(x, -env, color='#ffd700', linewidth=0.8, linestyle='--', alpha=0.6)
        self.ax3.set_ylim(-(max_amp + 0.3), max_amp + 0.3)
        self.ax3.set_xlabel("x")
        self.ax3.set_ylabel("진폭")
        self.ax3.set_title("합성파 (맥놀이)")
        self.ax3.legend(facecolor='#1e1e2e', edgecolor='#2a2a45', labelcolor='#e0e0ff', fontsize=8, loc='upper right')
        self.ax3.axhline(0, color='#2a2a45', linewidth=0.8)
        self.ax3.grid(True, color='#1a1a30', linewidth=0.5, alpha=0.6)

        self.canvas.draw_idle()

        # 맥놀이 정보
        beat_freq = abs(f1 - f2)
        beat_period = 1.0 / beat_freq if beat_freq > 1e-6 else float('inf')
        self.set_info([
            ("파동 1 주파수 f₁", f"{f1:.2f} Hz"),
            ("파동 1 진폭 A₁", f"{a1:.2f}"),
            ("파동 2 주파수 f₂", f"{f2:.2f} Hz"),
            ("파동 2 진폭 A₂", f"{a2:.2f}"),
            ("맥놀이 주파수", f"{beat_freq:.3f} Hz"),
            ("맥놀이 주기", f"{beat_period:.3f} s" if beat_period != float('inf') else "∞"),
            ("최대 합성 진폭", f"{a1 + a2:.2f}"),
        ])

    def on_animate(self, frame: int):
        self._time += 0.05
        self.update_plot()

    def on_reset(self):
        self._time = 0.0
