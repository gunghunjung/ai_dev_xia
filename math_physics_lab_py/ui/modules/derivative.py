import numpy as np
from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QWidget
from PyQt6.QtGui import QFont
from matplotlib.figure import Figure

from ui.base_module import ModulePage, apply_dark_axes
from engine.calculus import numerical_derivative

# 함수 정의
FUNCTIONS = {
    "x²": (lambda x: x ** 2, "x²"),
    "x³": (lambda x: x ** 3, "x³"),
    "sin(x)": (lambda x: np.sin(x), "sin(x)"),
    "cos(x)": (lambda x: np.cos(x), "cos(x)"),
    "eˣ": (lambda x: np.exp(x), "eˣ"),
    "ln(x)": (lambda x: np.log(np.where(x > 0, x, 1e-10)), "ln(x)"),
}


class DerivativeModule(ModulePage):
    def __init__(self, parent=None):
        super().__init__("미분", "#6c63ff", parent)

    def setup_module(self):
        self.add_ctrl_label("함수 선택")
        self.combo = QComboBox()
        for name in FUNCTIONS:
            self.combo.addItem(name)
        self.combo.setCurrentText("x²")
        self.combo.currentTextChanged.connect(self._on_param_changed)
        self.ctrl_layout.addWidget(self.combo)

        self.add_separator()
        self.add_ctrl_label("x 위치")
        self.sl_x = self.create_slider("x", -5.0, 5.0, 1.0, decimals=2)
        self.sl_x.connect(self._on_param_changed)

        self.add_ctrl_stretch()

        # 재생/정지 버튼 숨기기 (이 모듈은 정적)
        self.play_btn.setVisible(False)
        self.stop_btn.setVisible(False)

    def _get_func(self):
        name = self.combo.currentText()
        return FUNCTIONS[name]

    def _on_param_changed(self, *args):
        self.update_plot()

    def update_plot(self):
        func, label = self._get_func()
        x_pos = self.sl_x.value()

        self.ax.cla()
        apply_dark_axes(self.ax)

        # x 범위 결정
        if label == "ln(x)":
            x_min, x_max = 0.01, 10.0
            x_pos = max(x_pos, 0.1)
        else:
            x_min, x_max = -5.0, 5.0

        x = np.linspace(x_min, x_max, 500)
        y = func(x)

        # 함수 곡선
        self.ax.plot(x, y, color='#6c63ff', linewidth=2, label=f'f(x) = {label}', zorder=2)

        # 현재 x 위치에서의 값과 미분
        y_pos = float(func(np.array([x_pos]))[0]) if hasattr(func(np.array([x_pos])), '__len__') else float(func(x_pos))
        dy = numerical_derivative(lambda xi: float(func(np.array([xi]))[0]) if hasattr(func(np.array([xi])), '__len__') else float(func(xi)), x_pos)

        # 접선 그리기
        tangent_x = np.linspace(x_pos - 2, x_pos + 2, 100)
        tangent_y = y_pos + dy * (tangent_x - x_pos)

        # y 범위 제한
        y_lim_min = np.nanmin(y[np.isfinite(y)]) if np.any(np.isfinite(y)) else -10
        y_lim_max = np.nanmax(y[np.isfinite(y)]) if np.any(np.isfinite(y)) else 10
        margin = (y_lim_max - y_lim_min) * 0.2 + 1
        y_lo = y_lim_min - margin
        y_hi = y_lim_max + margin

        # 접선 클리핑
        mask = (tangent_y >= y_lo) & (tangent_y <= y_hi)
        if np.any(mask):
            self.ax.plot(tangent_x[mask], tangent_y[mask], color='#ff6b6b', linewidth=1.5,
                         linestyle='--', label="접선", zorder=3)

        # 점
        self.ax.scatter([x_pos], [y_pos], color='#ffd700', s=80, zorder=5, label=f'(x={x_pos:.2f})')

        # 수직선
        self.ax.axvline(x=x_pos, color='#ffd700', linewidth=0.8, alpha=0.4, linestyle=':')

        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_lo, y_hi)
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("f(x)")
        self.ax.set_title(f"미분 시각화: f(x) = {label}")
        self.ax.legend(facecolor='#1e1e2e', edgecolor='#2a2a45', labelcolor='#e0e0ff', fontsize=9)
        self.ax.grid(True, color='#1a1a30', linewidth=0.5, alpha=0.8)
        self.canvas.draw_idle()

        # 우측 정보 갱신
        self.set_info([
            ("함수", f"f(x) = {label}"),
            ("x 위치", f"{x_pos:.4f}"),
            ("f(x) 값", f"{y_pos:.4f}"),
            ("f'(x) 값", f"{dy:.4f}"),
            ("접선 기울기", f"{dy:.4f}"),
        ])
