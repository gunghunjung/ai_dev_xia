"""
VideoWidget v2 — VLC 렌더링 프레임

[v2 Evolution]
- 마우스 휠 → 볼륨 조절 신호 추가 (비디오 영역에서 휠 가능)
- 우클릭 컨텍스트 메뉴 지원
- 더블클릭/싱글클릭 분리 안정화
"""

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtWidgets import QFrame, QSizePolicy


class VideoWidget(QFrame):
    """VLC 비디오 출력 대상 위젯."""

    clicked         = pyqtSignal()
    double_clicked  = pyqtSignal()
    mouse_moved     = pyqtSignal()
    wheel_up        = pyqtSignal()   # 볼륨 올리기
    wheel_down      = pyqtSignal()   # 볼륨 내리기
    right_clicked   = pyqtSignal()   # 우클릭

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("video_frame")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(320, 180)
        self.setMouseTracking(True)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)

        # 싱글클릭 판정 타이머 (더블클릭과 분리)
        self._click_timer = QTimer(self)
        self._click_timer.setSingleShot(True)
        self._click_timer.setInterval(250)
        self._click_timer.timeout.connect(self._emit_click)
        self._pending_click = False

    def get_hwnd(self) -> int:
        return int(self.winId())

    # ── 마우스 이벤트 ──────────────────────────────────────────────────────

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._pending_click = True
            self._click_timer.start()
        elif event.button() == Qt.MouseButton.RightButton:
            self.right_clicked.emit()
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._click_timer.stop()
            self._pending_click = False
            self.double_clicked.emit()
        super().mouseDoubleClickEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        self.mouse_moved.emit()
        super().mouseMoveEvent(event)

    def wheelEvent(self, event) -> None:
        """[BUG FIX] 비디오 영역 마우스 휠로 볼륨 조절."""
        if event.angleDelta().y() > 0:
            self.wheel_up.emit()
        else:
            self.wheel_down.emit()
        event.accept()

    def _emit_click(self) -> None:
        if self._pending_click:
            self._pending_click = False
            self.clicked.emit()
