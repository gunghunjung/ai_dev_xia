"""
OSD (On-Screen Display) — 팟플레이어 스타일 화면 알림

볼륨 변경, 시간 이동, 속도, 자막 딜레이를 반투명 오버레이로 표시.
2초 후 자동으로 페이드 아웃.
"""

from PyQt6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, pyqtProperty
from PyQt6.QtGui import QColor, QPainter, QFont
from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout


class OsdWidget(QWidget):
    """비디오 위젯 위에 띄우는 반투명 알림 오버레이."""

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)

        self._opacity = 0.0

        # 레이아웃
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 12, 18, 12)

        self._label = QLabel("")
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = QFont("Segoe UI", 15, QFont.Weight.Bold)
        self._label.setFont(font)
        self._label.setStyleSheet("color: #ffffff;")
        layout.addWidget(self._label)

        self.adjustSize()

        # 자동 숨김 타이머
        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.setInterval(1800)
        self._hide_timer.timeout.connect(self._start_fade_out)

        # 페이드 아웃 애니메이션
        self._fade_timer = QTimer(self)
        self._fade_timer.setInterval(30)
        self._fade_out_step = 0.0
        self._fade_timer.timeout.connect(self._fade_step)

        self.hide()

    # ── 공개 API ───────────────────────────────────────────────────────────

    def show_message(self, text: str) -> None:
        """메시지를 표시하고 2초 후 자동으로 숨긴다."""
        self._fade_timer.stop()
        self._label.setText(text)
        self._opacity = 0.88
        self.adjustSize()
        self._reposition()
        self.show()
        self.raise_()
        self.update()
        self._hide_timer.start()

    def show_volume(self, vol: int, muted: bool = False) -> None:
        if muted:
            bar = "[ 음소거 ]"
        else:
            filled = int(vol / 150 * 14)
            bar = "[" + "=" * filled + " " * (14 - filled) + "]"
            bar = f"{bar}  {vol}%"
        self.show_message(f"볼륨  {bar}")

    def show_seek(self, delta_ms: int) -> None:
        sign = "+" if delta_ms >= 0 else ""
        sec  = delta_ms / 1000
        self.show_message(f"{'앞으로' if delta_ms > 0 else '뒤로'}  {sign}{sec:.0f}초")

    def show_speed(self, rate: float) -> None:
        self.show_message(f"재생 속도  {rate:.2f}x")

    def show_subtitle_delay(self, delay_ms: int) -> None:
        sign = "+" if delay_ms >= 0 else ""
        self.show_message(f"자막 딜레이  {sign}{delay_ms}ms")

    def show_subtitle(self, text: str) -> None:
        self.show_message(f"자막  {text}")

    # ── 내부 ──────────────────────────────────────────────────────────────

    def _reposition(self) -> None:
        """부모 위젯 좌하단에 위치."""
        if self.parent():
            pw = self.parent()
            x  = 20
            y  = pw.height() - self.height() - 80
            self.move(x, y)

    def _start_fade_out(self) -> None:
        self._fade_out_step = 0.06
        self._fade_timer.start()

    def _fade_step(self) -> None:
        self._opacity = max(0.0, self._opacity - self._fade_out_step)
        self.update()
        if self._opacity <= 0:
            self._fade_timer.stop()
            self.hide()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # 반투명 검정 배경
        bg = QColor(0, 0, 0, int(200 * self._opacity))
        painter.setBrush(bg)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(self.rect(), 10, 10)

        # 텍스트 페이드
        painter.setOpacity(self._opacity)
        super().paintEvent(event)
