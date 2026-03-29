from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QGridLayout, QPushButton,
    QLabel, QSizePolicy, QScrollArea
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont


MODULE_INFO = [
    {
        "id": "derivative",
        "icon": "f'(x)",
        "title": "미분",
        "subtitle": "Derivative",
        "desc": "함수의 순간 변화율과\n접선을 시각화합니다",
        "color": "#6c63ff",
    },
    {
        "id": "integral",
        "icon": "∫",
        "title": "적분",
        "subtitle": "Integral",
        "desc": "리만 합과 정적분의\n넓이를 시각화합니다",
        "color": "#43e97b",
    },
    {
        "id": "linear_transform",
        "icon": "[T]",
        "title": "선형변환",
        "subtitle": "Linear Transform",
        "desc": "2×2 행렬 변환으로\n공간이 변형되는 과정",
        "color": "#f7971e",
    },
    {
        "id": "projectile",
        "icon": "🎯",
        "title": "포물선 운동",
        "subtitle": "Projectile Motion",
        "desc": "초기속도와 발사각에 따른\n포물선 궤적 시뮬레이션",
        "color": "#ff6b6b",
    },
    {
        "id": "pendulum",
        "icon": "🔄",
        "title": "진자 운동",
        "subtitle": "Pendulum",
        "desc": "단진자의 진동과\n위상 공간을 관찰합니다",
        "color": "#a8edea",
    },
    {
        "id": "spring_mass",
        "icon": "~",
        "title": "용수철 진동",
        "subtitle": "Spring-Mass System",
        "desc": "감쇠 진동과 공명\n현상을 시뮬레이션합니다",
        "color": "#ffecd2",
    },
    {
        "id": "electric_field",
        "icon": "⚡",
        "title": "전기장",
        "subtitle": "Electric Field",
        "desc": "전하를 배치하고\n전기장 벡터를 시각화합니다",
        "color": "#ffd700",
    },
    {
        "id": "wave_interference",
        "icon": "≈",
        "title": "파동 간섭",
        "subtitle": "Wave Interference",
        "desc": "두 파동의 중첩과\n맥놀이 현상을 관찰합니다",
        "color": "#c471ed",
    },
]


class ModuleCard(QPushButton):
    def __init__(self, info: dict, parent=None):
        super().__init__(parent)
        self.info = info
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumHeight(160)
        self._color = info["color"]
        self._build()
        self._apply_style(False)

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(4)

        icon_lbl = QLabel(self.info["icon"])
        icon_lbl.setFont(QFont("Segoe UI Emoji", 28))
        icon_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon_lbl.setStyleSheet(f"color: {self._color}; background: transparent;")
        layout.addWidget(icon_lbl)

        title_lbl = QLabel(self.info["title"])
        title_lbl.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        title_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_lbl.setStyleSheet(f"color: #e0e0ff; background: transparent;")
        layout.addWidget(title_lbl)

        sub_lbl = QLabel(self.info["subtitle"])
        sub_lbl.setFont(QFont("Segoe UI", 9))
        sub_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sub_lbl.setStyleSheet("color: #5a5a8a; background: transparent;")
        layout.addWidget(sub_lbl)

        desc_lbl = QLabel(self.info["desc"])
        desc_lbl.setFont(QFont("Segoe UI", 10))
        desc_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc_lbl.setStyleSheet("color: #888aaa; background: transparent;")
        desc_lbl.setWordWrap(True)
        layout.addWidget(desc_lbl)

        # 위젯들이 버튼 이벤트를 받지 않도록
        for w in [icon_lbl, title_lbl, sub_lbl, desc_lbl]:
            w.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    def _apply_style(self, hovered: bool):
        border_color = self._color if hovered else "#2a2a45"
        bg = "#252538" if hovered else "#1e1e2e"
        self.setStyleSheet(
            f"QPushButton {{"
            f"  background: {bg};"
            f"  border: 2px solid {border_color};"
            f"  border-radius: 14px;"
            f"  color: #e0e0ff;"
            f"}}"
        )

    def enterEvent(self, event):
        self._apply_style(True)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._apply_style(False)
        super().leaveEvent(event)


class HomePage(QWidget):
    module_selected = pyqtSignal(str)  # module id

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(32, 24, 32, 24)
        outer.setSpacing(16)

        # 헤더
        header = QLabel("학습 모듈 선택")
        header.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))
        header.setStyleSheet("color: #e0e0ff;")
        outer.addWidget(header)

        sub = QLabel("수학과 물리학의 핵심 개념을 인터랙티브하게 탐구해 보세요")
        sub.setStyleSheet("color: #5a5a8a; font-size: 13px;")
        outer.addWidget(sub)

        # 스크롤 영역
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        scroll_content = QWidget()
        scroll_content.setStyleSheet("background: transparent;")

        grid = QGridLayout(scroll_content)
        grid.setContentsMargins(0, 8, 0, 8)
        grid.setSpacing(16)

        for i, info in enumerate(MODULE_INFO):
            row, col = divmod(i, 4)
            card = ModuleCard(info)
            card.clicked.connect(lambda checked, mid=info["id"]: self.module_selected.emit(mid))
            grid.addWidget(card, row, col)

        for col in range(4):
            grid.setColumnStretch(col, 1)

        scroll.setWidget(scroll_content)
        outer.addWidget(scroll, stretch=1)
