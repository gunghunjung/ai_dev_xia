"""
ControlsWidget v3 — 오버레이용 컨트롤 바

[v3 변경]
- mouse_activity 신호 추가: 마우스가 컨트롤 바 위에 있는 동안 auto-hide 억제
- 배경 그라디언트: 아래가 진하고 위로 갈수록 투명 (비디오 위에 자연스럽게 올라가도록)
"""

from PyQt6.QtCore import Qt, QPoint, pyqtSignal
from PyQt6.QtGui import QPainter, QLinearGradient, QColor
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout,
    QSlider, QPushButton, QLabel, QSizePolicy, QToolTip,
)


def _fmt_ms(ms: int) -> str:
    if ms < 0:
        ms = 0
    s = ms // 1000
    h, rem = divmod(s, 3600)
    m, s   = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


class TimelineSlider(QSlider):
    def __init__(self, parent=None) -> None:
        super().__init__(Qt.Orientation.Horizontal, parent)
        self._duration_ms = 0
        self.setMouseTracking(True)

    def set_duration(self, ms: int) -> None:
        self._duration_ms = ms

    def _pos_from_x(self, x: float) -> int:
        """클릭/드래그 X 좌표 → 슬라이더 값 계산."""
        ratio = max(0.0, min(1.0, x / max(1, self.width())))
        return int(ratio * self.maximum())

    def mousePressEvent(self, event) -> None:
        """클릭한 위치로 즉시 점프 — 기본 PageStep 동작 대신."""
        if event.button() == Qt.MouseButton.LeftButton:
            # 핸들 위치를 클릭 위치로 먼저 이동시킨 뒤 super() 호출
            # → Qt 가 '핸들을 드래그 중'으로 인식해서 sliderPressed 정상 발생
            self.setValue(self._pos_from_x(event.position().x()))
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._duration_ms > 0 and self.width() > 0:
            ratio  = max(0.0, min(1.0, event.position().x() / self.width()))
            pos_ms = int(ratio * self._duration_ms)
            QToolTip.showText(event.globalPosition().toPoint(), _fmt_ms(pos_ms), self)
        super().mouseMoveEvent(event)


class ControlsWidget(QWidget):
    play_pause_clicked = pyqtSignal()
    stop_clicked       = pyqtSignal()
    seek_begin         = pyqtSignal()
    seek_end           = pyqtSignal(float)
    volume_changed     = pyqtSignal(int)
    mute_clicked       = pyqtSignal()
    skip_forward       = pyqtSignal()
    skip_backward      = pyqtSignal()
    mouse_activity     = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("controls_bar")
        self.setMouseTracking(True)
        # [키보드 Fix] 컨트롤 바 자체는 포커스를 받지 않음
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._duration_ms = 0
        self._is_dragging = False
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 8, 10, 8)
        root.setSpacing(4)

        # 타임라인
        self.timeline = TimelineSlider(self)
        self.timeline.setObjectName("timeline")
        self.timeline.setRange(0, 10000)
        self.timeline.setValue(0)
        self.timeline.sliderPressed.connect(self._on_seek_start)
        self.timeline.sliderReleased.connect(self._on_seek_end)
        root.addWidget(self.timeline)

        # 버튼 행
        row = QHBoxLayout()
        row.setSpacing(4)

        self.btn_backward = QPushButton("⏮")
        self.btn_backward.setToolTip("10초 뒤로 (←)")
        self.btn_backward.clicked.connect(self.skip_backward)

        self.btn_play = QPushButton("▶")
        self.btn_play.setObjectName("btn_play")
        self.btn_play.setToolTip("재생 / 일시정지 (Space)")
        self.btn_play.clicked.connect(self.play_pause_clicked)

        self.btn_forward = QPushButton("⏭")
        self.btn_forward.setToolTip("10초 앞으로 (→)")
        self.btn_forward.clicked.connect(self.skip_forward)

        self.btn_stop = QPushButton("⏹")
        self.btn_stop.setToolTip("정지 (S)")
        self.btn_stop.clicked.connect(self.stop_clicked)

        self.lbl_time = QLabel("00:00 / 00:00")
        self.lbl_time.setObjectName("time_label")
        self.lbl_time.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.lbl_speed = QLabel("1.00x")
        self.lbl_speed.setObjectName("title_label")
        self.lbl_speed.setFixedWidth(46)
        self.lbl_speed.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_speed.setToolTip("재생 속도 ([ ] 조절)")

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        self.btn_mute = QPushButton("🔊")
        self.btn_mute.setObjectName("btn_mute")
        self.btn_mute.setToolTip("음소거 (M)")
        self.btn_mute.clicked.connect(self.mute_clicked)

        self.vol_slider = QSlider(Qt.Orientation.Horizontal)
        self.vol_slider.setObjectName("volume")
        self.vol_slider.setRange(0, 150)
        self.vol_slider.blockSignals(True)
        self.vol_slider.setValue(100)
        self.vol_slider.blockSignals(False)
        self.vol_slider.setToolTip("볼륨 (↑↓ / 마우스 휠)")
        self.vol_slider.valueChanged.connect(self.volume_changed)

        # [키보드 Fix] 모든 자식 위젯 NoFocus 설정
        # → 슬라이더/버튼이 ←→↑↓ 를 가로채지 않고 MainWindow.keyPressEvent 로 전달
        _nf = Qt.FocusPolicy.NoFocus
        self.timeline.setFocusPolicy(_nf)
        for w in (self.btn_backward, self.btn_play, self.btn_forward,
                  self.btn_stop, self.lbl_time, self.lbl_speed, spacer,
                  self.btn_mute, self.vol_slider):
            w.setFocusPolicy(_nf)
            row.addWidget(w)

        root.addLayout(row)

    # ── 상태 업데이트 ─────────────────────────────────────────────────────

    def set_position(self, pos: float) -> None:
        if not self._is_dragging:
            self.timeline.setValue(int(pos * 10000))

    def set_time(self, ms: int, duration_ms: int) -> None:
        if not self._is_dragging:
            self.lbl_time.setText(f"{_fmt_ms(ms)} / {_fmt_ms(duration_ms)}")

    def set_duration(self, ms: int) -> None:
        self._duration_ms = ms
        self.timeline.set_duration(ms)

    def set_playing(self, playing: bool) -> None:
        self.btn_play.setText("⏸" if playing else "▶")

    def set_muted(self, muted: bool) -> None:
        self.btn_mute.setText("🔇" if muted else "🔊")

    def set_volume(self, vol: int) -> None:
        self.vol_slider.blockSignals(True)
        self.vol_slider.setValue(vol)
        self.vol_slider.blockSignals(False)

    def set_speed(self, rate: float) -> None:
        self.lbl_speed.setText(f"{rate:.2f}x")

    # ── 타임라인 ──────────────────────────────────────────────────────────

    def _on_seek_start(self) -> None:
        self._is_dragging = True
        self.seek_begin.emit()

    def _on_seek_end(self) -> None:
        self._is_dragging = False
        self.seek_end.emit(self.timeline.value() / 10000.0)

    # ── 마우스 ────────────────────────────────────────────────────────────

    def enterEvent(self, event) -> None:
        """마우스가 컨트롤 바 진입 → auto-hide 억제 신호."""
        self.mouse_activity.emit()
        super().enterEvent(event)

    def mouseMoveEvent(self, event) -> None:
        self.mouse_activity.emit()
        super().mouseMoveEvent(event)

    def wheelEvent(self, event) -> None:
        step = 5 if event.angleDelta().y() > 0 else -5
        self.vol_slider.setValue(max(0, min(150, self.vol_slider.value() + step)))
        event.accept()

    # ── 그라디언트 배경 (오버레이 전용) ──────────────────────────────────

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        grad = QLinearGradient(0, 0, 0, self.height())
        grad.setColorAt(0.0, QColor(0, 0, 0, 0))      # 위: 완전 투명
        grad.setColorAt(1.0, QColor(0, 0, 0, 210))     # 아래: 진한 검정
        painter.fillRect(self.rect(), grad)
        super().paintEvent(event)
