"""TitleBarWidget — 프레임리스 창 전용 커스텀 타이틀바 오버레이."""
from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel, QPushButton


class TitleBarWidget(QWidget):
    """창 상단 타이틀바 (레이아웃 내 실제 공간 차지).

    - 드래그: 창 이동 (windowed 모드)
    - 더블클릭: 최대화 토글
    - 버튼: 최소화 / 최대화 / 닫기
    """

    mouse_activity = pyqtSignal()   # 마우스 움직임 → 숨김 타이머 억제용

    _BTN_BASE = (
        "QPushButton {{ background:{bg}; color:#ccc; border:none;"
        " border-radius:3px; font-size:12px; font-weight:bold; }}"
        "QPushButton:hover {{ background:{hv}; color:#fff; }}"
        "QPushButton:pressed {{ background:{pr}; }}"
    )

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("title_bar")
        self.setFixedHeight(30)
        self.setMouseTracking(True)
        self._drag_pos = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 0, 4, 0)
        layout.setSpacing(2)

        self._lbl = QLabel("PyPlayer")
        self._lbl.setObjectName("title_label")
        self._lbl.setStyleSheet(
            "color:#b0b8cc; font-size:12px; background:transparent;"
        )
        layout.addWidget(self._lbl)
        layout.addStretch()

        self._btn_min   = self._make_btn("─",  "#3a3a5c", "#555580", "#666698")
        self._btn_max   = self._make_btn("□",  "#3a3a5c", "#555580", "#666698")
        self._btn_close = self._make_btn("✕",  "#5a2525", "#c0392b", "#e74c3c")

        for btn in (self._btn_min, self._btn_max, self._btn_close):
            layout.addWidget(btn)

        self._btn_min.clicked.connect(lambda: self.window().showMinimized())
        self._btn_max.clicked.connect(self._toggle_max)
        self._btn_close.clicked.connect(self.window().close)

    # ── 헬퍼 ───────────────────────────────────────────────────────────────

    def _make_btn(self, text: str, bg: str, hv: str, pr: str) -> QPushButton:
        btn = QPushButton(text, self)
        btn.setFixedSize(30, 22)
        btn.setCursor(Qt.CursorShape.ArrowCursor)
        btn.setStyleSheet(self._BTN_BASE.format(bg=bg, hv=hv, pr=pr))
        return btn

    def set_title(self, title: str) -> None:
        self._lbl.setText(title)

    def _toggle_max(self) -> None:
        win = self.window()
        if win.isMaximized():
            win.showNormal()
        else:
            win.showMaximized()

    # ── 드래그 이동 ────────────────────────────────────────────────────────

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = (
                event.globalPosition().toPoint()
                - self.window().frameGeometry().topLeft()
            )
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        self.mouse_activity.emit()
        if self._drag_pos is not None and (
            event.buttons() & Qt.MouseButton.LeftButton
        ):
            self.window().move(
                event.globalPosition().toPoint() - self._drag_pos
            )
        super().mouseMoveEvent(event)

    def enterEvent(self, event) -> None:
        self.mouse_activity.emit()
        super().enterEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        self._drag_pos = None
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._toggle_max()
        super().mouseDoubleClickEvent(event)
