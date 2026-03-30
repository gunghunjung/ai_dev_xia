import numpy as np
from PyQt6.QtWidgets import (
    QDoubleSpinBox, QGridLayout, QHBoxLayout, QLabel,
    QPushButton, QWidget, QVBoxLayout
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from matplotlib.figure import Figure

from ui.base_module import ModulePage, apply_dark_axes

PRESETS = {
    "단위행렬": np.array([[1, 0], [0, 1]], dtype=float),
    "90° 회전": np.array([[0, -1], [1, 0]], dtype=float),
    "2배 확대": np.array([[2, 0], [0, 2]], dtype=float),
    "수평 전단": np.array([[1, 1], [0, 1]], dtype=float),
    "수평 반사": np.array([[1, 0], [0, -1]], dtype=float),
}


class LinearTransformModule(ModulePage):
    def __init__(self, parent=None):
        super().__init__("선형변환", "#f7971e", parent)

    def setup_module(self):
        self.add_ctrl_label("변환 행렬")

        mat_widget = QWidget()
        mat_widget.setStyleSheet("background: transparent;")
        mat_grid = QGridLayout(mat_widget)
        mat_grid.setContentsMargins(0, 4, 0, 4)
        mat_grid.setSpacing(6)

        # 행렬 레이블
        lbl_bracket_l = QLabel("[")
        lbl_bracket_l.setFont(QFont("Segoe UI", 28))
        lbl_bracket_l.setStyleSheet("color: #6c63ff;")
        lbl_bracket_r = QLabel("]")
        lbl_bracket_r.setFont(QFont("Segoe UI", 28))
        lbl_bracket_r.setStyleSheet("color: #6c63ff;")

        mat_grid.addWidget(lbl_bracket_l, 0, 0, 2, 1, Qt.AlignmentFlag.AlignCenter)

        self.spin_a = self._make_spin(1.0)
        self.spin_b = self._make_spin(0.0)
        self.spin_c = self._make_spin(0.0)
        self.spin_d = self._make_spin(1.0)
        mat_grid.addWidget(self.spin_a, 0, 1)
        mat_grid.addWidget(self.spin_b, 0, 2)
        mat_grid.addWidget(self.spin_c, 1, 1)
        mat_grid.addWidget(self.spin_d, 1, 2)
        mat_grid.addWidget(lbl_bracket_r, 0, 3, 2, 1, Qt.AlignmentFlag.AlignCenter)

        self.ctrl_layout.addWidget(mat_widget)

        self.add_separator()
        self.add_ctrl_label("프리셋")

        for name, mat in PRESETS.items():
            btn = QPushButton(name)
            btn.setStyleSheet(
                "QPushButton { font-size: 11px; padding: 5px 8px; }"
            )
            btn.clicked.connect(lambda checked, m=mat: self._apply_preset(m))
            self.ctrl_layout.addWidget(btn)

        self.add_ctrl_stretch()

        self.play_btn.setVisible(False)
        self.stop_btn.setVisible(False)

    def _make_spin(self, val):
        sp = QDoubleSpinBox()
        sp.setRange(-10.0, 10.0)
        sp.setSingleStep(0.1)
        sp.setValue(val)
        sp.setDecimals(2)
        sp.setFixedWidth(70)
        sp.valueChanged.connect(self._on_changed)
        return sp

    def _apply_preset(self, mat):
        self.spin_a.setValue(mat[0, 0])
        self.spin_b.setValue(mat[0, 1])
        self.spin_c.setValue(mat[1, 0])
        self.spin_d.setValue(mat[1, 1])

    def _on_changed(self, *args):
        self.update_plot()

    def _get_matrix(self):
        return np.array([
            [self.spin_a.value(), self.spin_b.value()],
            [self.spin_c.value(), self.spin_d.value()]
        ])

    def update_plot(self):
        M = self._get_matrix()
        det = np.linalg.det(M)

        self.ax.cla()
        apply_dark_axes(self.ax)

        # 격자 생성 (-3 ~ 3)
        grid_range = range(-3, 4)
        gray = '#444466'
        purple = '#6c63ff'
        orange = '#f7971e'

        # 변환 전 격자 (회색)
        for i in grid_range:
            # 수평선
            p1 = np.array([[-3, i], [3, i]], dtype=float).T
            self.ax.plot(p1[0], p1[1], color=gray, linewidth=0.5, alpha=0.5)
            # 수직선
            p2 = np.array([[i, -3], [i, 3]], dtype=float).T
            self.ax.plot(p2[0], p2[1], color=gray, linewidth=0.5, alpha=0.5)

        # 변환 후 격자 (컬러)
        for i in grid_range:
            ph1 = M @ np.array([[-3, i], [3, i]], dtype=float).T
            self.ax.plot(ph1[0], ph1[1], color=purple, linewidth=0.7, alpha=0.6)
            pv1 = M @ np.array([[i, -3], [i, 3]], dtype=float).T
            self.ax.plot(pv1[0], pv1[1], color=purple, linewidth=0.7, alpha=0.6)

        # 기저 벡터 (변환 전: 회색, 변환 후: 컬러)
        origin = np.array([0, 0])
        e1 = np.array([1, 0])
        e2 = np.array([0, 1])
        Me1 = M @ e1
        Me2 = M @ e2

        self.ax.annotate('', xy=e1, xytext=origin,
                         arrowprops=dict(arrowstyle='->', color='#888aaa', lw=1.5))
        self.ax.annotate('', xy=e2, xytext=origin,
                         arrowprops=dict(arrowstyle='->', color='#888aaa', lw=1.5))
        self.ax.annotate('', xy=Me1, xytext=origin,
                         arrowprops=dict(arrowstyle='->', color=orange, lw=2.5))
        self.ax.annotate('', xy=Me2, xytext=origin,
                         arrowprops=dict(arrowstyle='->', color='#43e97b', lw=2.5))

        lim = max(5, np.max(np.abs(M)) * 3 + 1)
        self.ax.set_xlim(-lim, lim)
        self.ax.set_ylim(-lim, lim)
        self.ax.set_aspect('equal', adjustable='datalim')
        self.ax.axhline(0, color='#2a2a45', linewidth=0.8)
        self.ax.axvline(0, color='#2a2a45', linewidth=0.8)
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_title("선형변환 시각화")
        self.ax.grid(False)
        self.canvas.draw_idle()

        # 정보
        try:
            inv_possible = abs(det) > 1e-10
            inv_text = "가능" if inv_possible else "불가능 (특이행렬)"
            if inv_possible:
                inv_m = np.linalg.inv(M)
                inv_str = f"[[{inv_m[0,0]:.2f},{inv_m[0,1]:.2f}],[{inv_m[1,0]:.2f},{inv_m[1,1]:.2f}]]"
            else:
                inv_str = "없음"
        except Exception:
            inv_text = "계산 오류"
            inv_str = "-"

        self.set_info([
            ("행렬", f"[[{M[0,0]:.2f},{M[0,1]:.2f}],[{M[1,0]:.2f},{M[1,1]:.2f}]]"),
            ("행렬식 det(M)", f"{det:.4f}"),
            ("역행렬 가능", inv_text),
            ("역행렬", inv_str),
        ])
