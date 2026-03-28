"""
HelpWindow — 모달리스 단축키 도움말 창

- F1 로 토글 (메인 창 재생 방해 없음)
- 섹션별 카드 레이아웃
- 항상 메인 창 위에 표시
"""

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QColor
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QScrollArea,
    QWidget, QLabel, QFrame, QGridLayout,
)


# ── 단축키 데이터 ─────────────────────────────────────────────────────────
SECTIONS = [
    ("재생 제어", [
        ("Space",           "재생 / 일시정지"),
        ("S",               "정지"),
        (".",               "프레임 한 칸 전진 (일시정지 중)"),
    ]),
    ("탐색 (이동)", [
        ("←  /  →",         "±5초"),
        ("Shift + ←  /  →", "±1초  (미세 조정)"),
        ("Alt + ←  /  →",   "±10초"),
        ("Ctrl + ←  /  →",  "±1분"),
        ("1 ~ 9",           "10% ~ 90% 위치 점프"),
    ]),
    ("볼륨", [
        ("↑  /  ↓",         "볼륨 +5 / -5"),
        ("마우스 휠",        "볼륨 조절 (영상·컨트롤 바 모두)"),
        ("M",               "음소거 토글"),
    ]),
    ("재생 속도", [
        ("]",               "속도 올리기"),
        ("[",               "속도 내리기"),
        ("\\",              "속도 1.0x 리셋"),
    ]),
    ("자막", [
        ("V",               "자막 온 / 오프"),
        ("J",               "자막 딜레이  +200ms"),
        ("H",               "자막 딜레이  -200ms"),
        ("드래그 앤 드롭",   "자막 파일(.srt/.smi 등) 바로 로드"),
    ]),
    ("화면", [
        ("F  /  F11",       "전체화면 토글"),
        ("Esc",             "전체화면 해제"),
        ("T",               "항상 위 (Always on Top)"),
        ("더블클릭",         "전체화면 토글"),
    ]),
    ("파일 & 기타", [
        ("Ctrl + O",        "파일 열기"),
        ("우클릭",           "컨텍스트 메뉴"),
        ("F1",              "이 도움말 토글"),
        ("Ctrl + Q",        "프로그램 종료"),
    ]),
]


class HelpWindow(QDialog):
    """모달리스 도움말 창."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent, Qt.WindowType.Tool)   # Tool: 항상 부모 위
        self.setWindowTitle("PyPlayer — 단축키 도움말")
        self.setMinimumWidth(480)
        self.setMaximumWidth(560)
        self.resize(500, 620)
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        # 모달리스 — 메인 창 입력 차단 없음
        self.setModal(False)

        self._build_ui()
        self._apply_style()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── 헤더 ─────────────────────────────────────────────────────────
        header = QWidget()
        header.setObjectName("help_header")
        header.setFixedHeight(52)
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(18, 0, 18, 0)

        title = QLabel("⌨  단축키 도움말")
        title.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        title.setStyleSheet("color: #e0e0e0;")

        hint = QLabel("F1  토글")
        hint.setFont(QFont("Segoe UI", 10))
        hint.setStyleSheet("color: #607080;")

        h_layout.addWidget(title)
        h_layout.addStretch()
        h_layout.addWidget(hint)

        root.addWidget(header)

        # ── 구분선 ───────────────────────────────────────────────────────
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #0f3460;")
        root.addWidget(sep)

        # ── 스크롤 영역 ───────────────────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(14, 10, 14, 14)
        content_layout.setSpacing(10)

        for section_title, rows in SECTIONS:
            content_layout.addWidget(self._make_card(section_title, rows))

        content_layout.addStretch()
        scroll.setWidget(content)
        root.addWidget(scroll)

    def _make_card(self, title: str, rows: list[tuple[str, str]]) -> QWidget:
        """섹션 카드 위젯을 생성한다."""
        card = QFrame()
        card.setObjectName("help_card")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(14, 10, 14, 12)
        card_layout.setSpacing(6)

        # 섹션 제목
        lbl_title = QLabel(title)
        lbl_title.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        lbl_title.setStyleSheet("color: #53c0f0;")
        card_layout.addWidget(lbl_title)

        # 구분선
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color: #1e2a3a;")
        card_layout.addWidget(line)

        # 키/설명 그리드
        grid = QGridLayout()
        grid.setColumnMinimumWidth(0, 170)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(5)

        for i, (key, desc) in enumerate(rows):
            key_lbl = QLabel(key)
            key_lbl.setFont(QFont("Consolas", 10, QFont.Weight.Bold))
            key_lbl.setStyleSheet(
                "color: #e0c060;"
                "background: #0f1e2e;"
                "border: 1px solid #1e3050;"
                "border-radius: 4px;"
                "padding: 2px 6px;"
            )
            key_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

            desc_lbl = QLabel(desc)
            desc_lbl.setFont(QFont("Segoe UI", 10))
            desc_lbl.setStyleSheet("color: #c0c8d8;")

            grid.addWidget(key_lbl,  i, 0)
            grid.addWidget(desc_lbl, i, 1)

        card_layout.addLayout(grid)
        return card

    def _apply_style(self) -> None:
        self.setStyleSheet("""
            QDialog {
                background-color: #1a1a2e;
            }
            QWidget#help_header {
                background-color: #16213e;
            }
            QFrame#help_card {
                background-color: #16213e;
                border: 1px solid #0f3460;
                border-radius: 8px;
            }
            QScrollArea {
                background-color: #1a1a2e;
            }
            QScrollBar:vertical {
                background: #1a1a2e;
                width: 6px;
                border-radius: 3px;
            }
            QScrollBar::handle:vertical {
                background: #0f3460;
                border-radius: 3px;
                min-height: 20px;
            }
        """)

    def toggle(self) -> None:
        """F1 토글 — 보이면 숨기고, 숨겨져 있으면 표시한다."""
        if self.isVisible():
            self.hide()
        else:
            self.show()
            self.raise_()
            self.activateWindow()
