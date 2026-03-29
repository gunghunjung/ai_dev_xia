from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QStackedWidget, QLabel, QPushButton, QFrame
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from ui.home_page import HomePage


DARK_QSS = """
QMainWindow, QWidget {
    background-color: #0a0a0f;
    color: #e0e0ff;
    font-family: 'Segoe UI', '맑은 고딕', sans-serif;
}
QLabel { color: #e0e0ff; }
QSlider::groove:horizontal {
    height: 4px; background: #2a2a45; border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #6c63ff; width: 16px; height: 16px;
    margin: -6px 0; border-radius: 8px;
}
QSlider::sub-page:horizontal { background: #6c63ff; border-radius: 2px; }
QPushButton {
    background: #1e1e2e; border: 1px solid #2a2a45;
    color: #e0e0ff; padding: 8px 16px; border-radius: 8px;
}
QPushButton:hover { border-color: #6c63ff; color: #a89fff; }
QPushButton:pressed { background: #252538; }
QComboBox {
    background: #1e1e2e; border: 1px solid #2a2a45;
    color: #e0e0ff; padding: 6px 12px; border-radius: 6px;
}
QComboBox::drop-down { border: none; }
QComboBox QAbstractItemView {
    background: #1e1e2e; border: 1px solid #2a2a45; color: #e0e0ff;
    selection-background-color: #252538;
}
QDoubleSpinBox {
    background: #1e1e2e; border: 1px solid #2a2a45;
    color: #e0e0ff; padding: 4px 8px; border-radius: 6px;
}
QScrollArea { border: none; }
QScrollBar:vertical {
    background: #0a0a0f; width: 8px; margin: 0;
}
QScrollBar::handle:vertical {
    background: #2a2a45; border-radius: 4px; min-height: 20px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar:horizontal {
    background: #0a0a0f; height: 8px; margin: 0;
}
QScrollBar::handle:horizontal {
    background: #2a2a45; border-radius: 4px; min-width: 20px;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
"""


class NavBar(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(56)
        self.setStyleSheet(
            "QFrame { background: #0e0e1a; border-bottom: 1px solid #1a1a30; }"
        )
        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 0, 20, 0)

        # 뒤로가기 버튼
        self.back_btn = QPushButton("← 홈으로")
        self.back_btn.setStyleSheet(
            "QPushButton { background: transparent; border: 1px solid #2a2a45; "
            "color: #888aaa; padding: 6px 14px; border-radius: 6px; font-size: 13px; }"
            "QPushButton:hover { border-color: #6c63ff; color: #a89fff; }"
        )
        self.back_btn.setFixedWidth(110)
        layout.addWidget(self.back_btn)

        # 로고
        logo = QLabel("수학·물리 학습")
        logo.setFont(QFont("Segoe UI", 15, QFont.Weight.Bold))
        logo.setStyleSheet("color: #6c63ff; background: transparent;")
        logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(logo, stretch=1)

        # 오른쪽 여백용 더미
        dummy = QWidget()
        dummy.setFixedWidth(110)
        dummy.setStyleSheet("background: transparent;")
        layout.addWidget(dummy)

    def show_back(self, visible: bool):
        self.back_btn.setVisible(visible)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("수학·물리 인터랙티브 학습")
        self.resize(1280, 800)
        self.setMinimumSize(900, 600)

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # NavBar
        self.navbar = NavBar()
        self.navbar.back_btn.clicked.connect(self._go_home)
        root_layout.addWidget(self.navbar)

        # Stacked widget
        self.stack = QStackedWidget()
        root_layout.addWidget(self.stack, stretch=1)

        # Home
        self.home_page = HomePage()
        self.home_page.module_selected.connect(self._open_module)
        self.stack.addWidget(self.home_page)

        self._module_cache: dict = {}
        self._go_home()

    # ------------------------------------------------------------------
    def _go_home(self):
        self.stack.setCurrentWidget(self.home_page)
        self.navbar.show_back(False)

    def _open_module(self, module_id: str):
        if module_id not in self._module_cache:
            widget = self._create_module(module_id)
            if widget is None:
                return
            self._module_cache[module_id] = widget
            self.stack.addWidget(widget)
        self.stack.setCurrentWidget(self._module_cache[module_id])
        self.navbar.show_back(True)

    def _create_module(self, module_id: str):
        try:
            if module_id == "derivative":
                from ui.modules.derivative import DerivativeModule
                return DerivativeModule()
            elif module_id == "integral":
                from ui.modules.integral import IntegralModule
                return IntegralModule()
            elif module_id == "linear_transform":
                from ui.modules.linear_transform import LinearTransformModule
                return LinearTransformModule()
            elif module_id == "projectile":
                from ui.modules.projectile import ProjectileModule
                return ProjectileModule()
            elif module_id == "pendulum":
                from ui.modules.pendulum import PendulumModule
                return PendulumModule()
            elif module_id == "spring_mass":
                from ui.modules.spring_mass import SpringMassModule
                return SpringMassModule()
            elif module_id == "electric_field":
                from ui.modules.electric_field import ElectricFieldModule
                return ElectricFieldModule()
            elif module_id == "wave_interference":
                from ui.modules.wave_interference import WaveInterferenceModule
                return WaveInterferenceModule()
        except Exception as e:
            print(f"[MainWindow] 모듈 로드 오류 ({module_id}): {e}")
            import traceback
            traceback.print_exc()
        return None
