import numpy as np
from PyQt6.QtWidgets import QPushButton, QLabel
from PyQt6.QtCore import Qt
from matplotlib.figure import Figure

from ui.base_module import ModulePage, apply_dark_axes
from engine.waves import electric_field_2d


GRID_SIZE = 20
GRID_RANGE = 5.0


class ElectricFieldModule(ModulePage):
    TIMER_INTERVAL = 50

    def __init__(self, parent=None):
        self._charges = []  # list of [x, y, q]
        self._drag_idx = None
        self._drag_start = None
        super().__init__("전기장", "#ffd700", parent)

    def setup_module(self):
        self.add_ctrl_label("조작 방법")
        hint = QLabel(
            "• 좌클릭: 양전하(+) 추가\n"
            "• 우클릭: 음전하(-) 추가\n"
            "• 드래그: 전하 이동\n"
            "• 더블클릭: 전하 삭제"
        )
        hint.setStyleSheet("color: #888aaa; font-size: 11px; line-height: 1.5;")
        hint.setWordWrap(True)
        self.ctrl_layout.addWidget(hint)

        self.add_separator()
        clear_btn = QPushButton("전하 모두 지우기")
        clear_btn.clicked.connect(self._clear_charges)
        self.ctrl_layout.addWidget(clear_btn)

        self.add_ctrl_stretch()

        # 재생/정지 버튼 숨기기 (이 모듈은 정적)
        self.play_btn.setVisible(False)
        self.stop_btn.setVisible(False)

        # matplotlib 이벤트 연결
        self.canvas.mpl_connect('button_press_event', self._on_mouse_press)
        self.canvas.mpl_connect('button_release_event', self._on_mouse_release)
        self.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)

    def _clear_charges(self):
        self._charges.clear()
        self._drag_idx = None
        self.update_plot()

    def _find_nearby_charge(self, x, y, radius=0.4):
        for i, (cx, cy, q) in enumerate(self._charges):
            if np.sqrt((x - cx) ** 2 + (y - cy) ** 2) < radius:
                return i
        return None

    def _on_mouse_press(self, event):
        if event.inaxes != self.ax:
            return
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return

        if event.dblclick:
            # 더블클릭: 전하 삭제
            idx = self._find_nearby_charge(x, y)
            if idx is not None:
                self._charges.pop(idx)
                self.update_plot()
            return

        idx = self._find_nearby_charge(x, y)
        if idx is not None:
            # 드래그 시작
            self._drag_idx = idx
        else:
            # 새 전하 추가
            if event.button == 1:
                q = 1.0  # 양전하
            else:
                q = -1.0  # 음전하
            self._charges.append([x, y, q])
            self.update_plot()

    def _on_mouse_release(self, event):
        self._drag_idx = None

    def _on_mouse_move(self, event):
        if self._drag_idx is None:
            return
        if event.inaxes != self.ax:
            return
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
        self._charges[self._drag_idx][0] = x
        self._charges[self._drag_idx][1] = y
        self.update_plot()

    def update_plot(self):
        self.ax.cla()
        apply_dark_axes(self.ax)

        R = GRID_RANGE
        N = GRID_SIZE
        xg = np.linspace(-R, R, N)
        yg = np.linspace(-R, R, N)
        X, Y = np.meshgrid(xg, yg)

        if self._charges:
            Ex, Ey = electric_field_2d(self._charges, X, Y)

            # 전기장 크기 로그 스케일 정규화
            E_mag = np.sqrt(Ex ** 2 + Ey ** 2)
            E_mag_safe = np.where(E_mag == 0, 1e-10, E_mag)
            Ex_norm = Ex / E_mag_safe
            Ey_norm = Ey / E_mag_safe

            log_mag = np.log1p(E_mag / (E_mag.max() + 1e-10) * 10)

            self.ax.quiver(
                X, Y, Ex_norm * log_mag, Ey_norm * log_mag,
                E_mag, cmap='plasma', alpha=0.8,
                scale=None, scale_units='xy', angles='xy',
                width=0.003
            )

            # 전하 표시
            for i, (cx, cy, q) in enumerate(self._charges):
                color = '#ff4444' if q > 0 else '#4444ff'
                label = '+' if q > 0 else '-'
                self.ax.scatter([cx], [cy], color=color, s=200, zorder=10,
                                edgecolors='white', linewidths=1.5)
                self.ax.text(cx, cy, label, ha='center', va='center',
                             color='white', fontsize=12, fontweight='bold', zorder=11)
        else:
            # 안내 텍스트
            self.ax.text(0, 0, "캔버스를 클릭하여\n전하를 추가하세요",
                         ha='center', va='center', color='#5a5a8a', fontsize=12)

        self.ax.set_xlim(-R, R)
        self.ax.set_ylim(-R, R)
        self.ax.set_aspect('equal')
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_title("전기장 시각화")
        self.ax.grid(True, color='#1a1a30', linewidth=0.5, alpha=0.4)
        self.canvas.draw_idle()

        # 전하 목록 정보
        if self._charges:
            info_lines = [("전하 수", f"{len(self._charges)}")]
            for i, (cx, cy, q) in enumerate(self._charges):
                sign = "양(+)" if q > 0 else "음(-)"
                info_lines.append((f"전하{i + 1}", f"{sign} ({cx:.1f}, {cy:.1f})"))
        else:
            info_lines = [("안내", "클릭으로 전하 추가")]
        self.set_info(info_lines)
