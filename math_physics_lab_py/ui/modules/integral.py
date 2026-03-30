import numpy as np
from PyQt6.QtWidgets import QComboBox
from matplotlib.figure import Figure

from ui.base_module import ModulePage, apply_dark_axes
from engine.calculus import riemann_sum, exact_integral

FUNCTIONS = {
    "x²": (lambda x: x ** 2, "x²"),
    "sin(x)": (lambda x: np.sin(x), "sin(x)"),
    "cos(x)": (lambda x: np.cos(x), "cos(x)"),
    "eˣ": (lambda x: np.exp(np.clip(x, -10, 10)), "eˣ"),
}


class IntegralModule(ModulePage):
    def __init__(self, parent=None):
        super().__init__("적분", "#43e97b", parent)

    def setup_module(self):
        self.add_ctrl_label("함수 선택")
        self.combo = QComboBox()
        for name in FUNCTIONS:
            self.combo.addItem(name)
        self.combo.currentTextChanged.connect(self._on_changed)
        self.ctrl_layout.addWidget(self.combo)

        self.add_separator()
        self.add_ctrl_label("적분 범위")
        self.sl_a = self.create_slider("하한 a", -5.0, 4.9, 0.0, decimals=1)
        self.sl_b = self.create_slider("상한 b", -4.9, 5.0, 2.0, decimals=1)
        self.sl_a.connect(self._on_changed)
        self.sl_b.connect(self._on_changed)

        self.add_separator()
        self.add_ctrl_label("리만 분할")
        self.sl_n = self.create_slider("구간 수 n", 2, 200, 10, decimals=0)
        self.sl_n.connect(self._on_changed)

        self.add_ctrl_stretch()

        self.play_btn.setVisible(False)
        self.stop_btn.setVisible(False)

    def _on_changed(self, *args):
        self.update_plot()

    def update_plot(self):
        name = self.combo.currentText()
        func, label = FUNCTIONS[name]
        a = self.sl_a.value()
        b = self.sl_b.value()
        n = max(2, int(self.sl_n.value()))

        if a >= b:
            b = a + 0.1

        self.ax.cla()
        apply_dark_axes(self.ax)

        x_plot = np.linspace(a - 0.5, b + 0.5, 500)
        y_plot = func(x_plot)

        # 함수 곡선
        self.ax.plot(x_plot, y_plot, color='#43e97b', linewidth=2, label=f'f(x) = {label}', zorder=3)

        # 리만 합 사각형
        dx = (b - a) / n
        xs_rect = np.linspace(a, b - dx, n)
        for xi in xs_rect:
            yi = float(func(np.array([xi]))[0]) if hasattr(func(np.array([xi])), '__len__') else float(func(xi))
            rect_x = [xi, xi, xi + dx, xi + dx, xi]
            rect_y = [0, yi, yi, 0, 0]
            self.ax.fill(rect_x, rect_y, alpha=0.3, color='#43e97b')
            self.ax.plot(rect_x, rect_y, color='#43e97b', linewidth=0.5, alpha=0.6)

        # 실제 적분 영역 채우기
        x_fill = np.linspace(a, b, 300)
        y_fill = func(x_fill)
        self.ax.fill_between(x_fill, 0, y_fill, alpha=0.15, color='#a8edea', label='실제 면적')

        self.ax.axhline(y=0, color='#2a2a45', linewidth=1)
        self.ax.axvline(x=a, color='#ff6b6b', linewidth=1, linestyle='--', alpha=0.7, label=f'a={a:.1f}')
        self.ax.axvline(x=b, color='#ffd700', linewidth=1, linestyle='--', alpha=0.7, label=f'b={b:.1f}')

        self.ax.set_xlabel("x")
        self.ax.set_ylabel("f(x)")
        self.ax.set_title(f"적분 시각화: ∫ {label} dx, n={n}")
        self.ax.legend(facecolor='#1e1e2e', edgecolor='#2a2a45', labelcolor='#e0e0ff', fontsize=9)
        self.ax.grid(True, color='#1a1a30', linewidth=0.5, alpha=0.8)
        self.canvas.draw_idle()

        # 계산값
        riemann = riemann_sum(func, a, b, n)
        exact = exact_integral(func, a, b)
        error = abs(riemann - exact)

        self.set_info([
            ("함수", f"f(x) = {label}"),
            ("적분 범위", f"[{a:.1f}, {b:.1f}]"),
            ("구간 수 n", f"{n}"),
            ("리만 합", f"{riemann:.6f}"),
            ("실제 적분값", f"{exact:.6f}"),
            ("오차", f"{error:.6f}"),
        ])
