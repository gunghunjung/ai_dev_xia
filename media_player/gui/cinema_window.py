"""
CinemaWindow — 순수 영상 출력 전용 플레이어 모드

목표: "사용자는 플레이어의 존재를 인지하지 못하고
       오직 영상만 화면 전체에 표시되는 상태"

[1] OS 타이틀바·테두리·버튼 완전 제거 (FramelessWindowHint)
[2] 실행 즉시 전체화면, 멀티모니터 선택 가능
[3] 비디오 렌더링 100% — padding/margin 0, 순수 검정 배경
[4] UI 요소 전무 — 컨트롤/메뉴/OSD 없음 (자막 독립 허용)
[5] 마우스 커서 자동 숨김, 이동해도 UI 미표시
[6] 키 입력 최소화: ESC=종료, Space=재생일시정지, ↑↓=볼륨
[7] always-on-top / kiosk / loop / autorun 옵션
[8] VLC HW 디코딩 + 화면비율 모드 (fit / fill / stretch)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QKeyEvent
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget

from gui.video_widget  import VideoWidget
from core.media_engine import MediaEngine


# ── 설정 데이터클래스 ──────────────────────────────────────────────────────

@dataclass
class CinemaConfig:
    monitor_index:   int        = 0        # 출력 모니터 (0=기본)
    always_on_top:   bool       = False    # 항상 위
    loop:            bool       = False    # 반복 재생
    kiosk:           bool       = False    # 키오스크 (ESC 차단)
    aspect:          str        = "fit"    # fit | fill | stretch
    show_subtitles:  bool       = True     # 자막 표시
    cursor_hide_ms:  int        = 3000     # 커서 자동 숨김 지연 (ms)
    playlist:        List[str]  = field(default_factory=list)
    autorun:         bool       = False    # 목록 자동 재생


# ── CinemaWindow ───────────────────────────────────────────────────────────

class CinemaWindow(QMainWindow):
    """UI 요소가 완전히 제거된 순수 영상 출력 창.

    사용법::

        cfg = CinemaConfig(monitor_index=0, loop=True, aspect="fill")
        cw  = CinemaWindow(engine, cfg)
        cw.play_file("/path/to/video.mp4")
        cw.exiting.connect(restore_main_window)
    """

    exiting = pyqtSignal()   # 창이 닫힐 때 MainWindow 복원 요청

    # 커서 숨김 후 내부적으로 사용하는 blank cursor
    _BLANK = Qt.CursorShape.BlankCursor

    def __init__(
        self,
        engine: MediaEngine,
        config: CinemaConfig | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._engine        = engine
        self._cfg           = config or CinemaConfig()
        self._playlist_idx  = 0
        self._cursor_shown  = True

        self._setup_window_flags()
        self._build_ui()
        self._setup_cursor_timer()
        self._connect_engine_signals()

        # 자막 표시 상태 동기화
        if not self._cfg.show_subtitles:
            # VLC 자막 비활성화
            try:
                self._engine._player.video_set_spu(-1)
            except Exception:
                pass

        # 오토런: 목록이 있으면 첫 항목 자동 재생
        if self._cfg.autorun and self._cfg.playlist:
            QTimer.singleShot(200, lambda: self._engine.load(
                self._cfg.playlist[0]
            ))

    # ── 윈도우 플래그 ─────────────────────────────────────────────────────

    def _setup_window_flags(self) -> None:
        flags = (
            Qt.WindowType.Window
            | Qt.WindowType.FramelessWindowHint
        )
        if self._cfg.always_on_top:
            flags |= Qt.WindowType.WindowStaysOnTopHint
        self.setWindowFlags(flags)
        # Qt 내부 크롬 완전 제거
        self.menuBar().hide()
        self.statusBar().hide()
        self.setContentsMargins(0, 0, 0, 0)

    # ── UI 구성 ───────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        # 순수 검정 배경
        self.setStyleSheet("QMainWindow, QWidget { background:#000; border:none; }")

        central = QWidget(self)
        central.setStyleSheet("background:#000;")
        central.setContentsMargins(0, 0, 0, 0)
        central.setMouseTracking(True)
        self.setCentralWidget(central)

        self._video = VideoWidget(central)
        self._video.setStyleSheet("background:#000;")
        self._video.setMouseTracking(True)

        # 마우스 이벤트 → 커서 타이머 리셋 (UI 표시 없음)
        self._video.mouse_moved.connect(self._on_mouse_activity)

        # HWND 바인딩 (위젯이 실제로 그려진 뒤 실행)
        QTimer.singleShot(0, self._bind_hwnd)

        # 모니터 선택 + 전체화면
        self._apply_monitor()

    def _bind_hwnd(self) -> None:
        """VLC 렌더링을 이 창의 VideoWidget에 연결."""
        try:
            self._engine.set_hwnd(self._video.get_hwnd())
            self._apply_aspect()
        except Exception:
            pass

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        cw = self.centralWidget()
        if cw:
            self._video.setGeometry(cw.rect())

    # ── 모니터 선택 + 전체화면 ───────────────────────────────────────────

    def _apply_monitor(self) -> None:
        screens = QApplication.screens()
        idx     = min(self._cfg.monitor_index, len(screens) - 1)
        screen  = screens[idx]
        # 대상 모니터 좌상단으로 이동 후 전체화면
        self.move(screen.geometry().topLeft())
        self.showFullScreen()

    # ── 화면비율 (VLC) ────────────────────────────────────────────────────

    def _apply_aspect(self) -> None:
        """
        fit     : VLC 기본 (레터박스/필러박스 유지)
        fill    : 화면을 가득 채우도록 크롭
        stretch : 가로세로 비율 무시, 창 전체에 꽉 채움
        """
        player = self._engine._player
        mode   = self._cfg.aspect

        try:
            if mode == "fit":
                player.video_set_aspect_ratio(None)
                player.video_set_crop_geometry(None)

            elif mode == "fill":
                # 현재 화면 비율로 크롭
                geo   = self._screen_geometry()
                ratio = f"{geo.width()}:{geo.height()}"
                player.video_set_crop_geometry(ratio)
                player.video_set_aspect_ratio(None)

            elif mode == "stretch":
                # 화면 비율을 영상 비율로 강제
                geo   = self._screen_geometry()
                ratio = f"{geo.width()}:{geo.height()}"
                player.video_set_aspect_ratio(ratio)
                player.video_set_crop_geometry(None)

        except Exception:
            pass

    def set_aspect(self, mode: str) -> None:
        """런타임 화면비율 변경 (fit | fill | stretch)."""
        self._cfg.aspect = mode
        self._apply_aspect()

    def _screen_geometry(self):
        screens = QApplication.screens()
        idx     = min(self._cfg.monitor_index, len(screens) - 1)
        return screens[idx].geometry()

    # ── 커서 자동 숨김 ────────────────────────────────────────────────────

    def _setup_cursor_timer(self) -> None:
        self._cursor_timer = QTimer(self)
        self._cursor_timer.setSingleShot(True)
        self._cursor_timer.setInterval(self._cfg.cursor_hide_ms)
        self._cursor_timer.timeout.connect(self._hide_cursor)

    def _hide_cursor(self) -> None:
        self._cursor_shown = False
        self.setCursor(self._BLANK)

    def _show_cursor_temporarily(self) -> None:
        """커서를 잠깐 복원하고 숨김 타이머를 재시작.
        UI(컨트롤/오버레이)는 절대 표시하지 않음."""
        if not self._cursor_shown:
            self._cursor_shown = True
            self.unsetCursor()
        self._cursor_timer.start()

    def _on_mouse_activity(self) -> None:
        self._show_cursor_temporarily()

    def mouseMoveEvent(self, event) -> None:
        self._show_cursor_temporarily()
        super().mouseMoveEvent(event)

    # ── 엔진 신호 ─────────────────────────────────────────────────────────

    def _connect_engine_signals(self) -> None:
        self._engine.state_changed.connect(self._on_state_changed)

    def _on_state_changed(self, state: str) -> None:
        if state != "ended":
            return

        playlist = self._cfg.playlist

        if self._cfg.loop and not playlist:
            # 단일 파일 루프
            self._engine.seek(0.0)
            self._engine.play()

        elif playlist:
            if self._cfg.loop or self._playlist_idx < len(playlist) - 1:
                self._play_next()

    # ── 재생 목록 ─────────────────────────────────────────────────────────

    def _play_next(self) -> None:
        pl = self._cfg.playlist
        if not pl:
            return
        self._playlist_idx = (self._playlist_idx + 1) % len(pl)
        self._engine.load(pl[self._playlist_idx])

    def play_file(self, path: str) -> None:
        """외부에서 특정 파일 재생 지시."""
        if self._cfg.playlist:
            try:
                self._playlist_idx = self._cfg.playlist.index(path)
            except ValueError:
                pass
        self._engine.load(path)

    # ── 키보드 ────────────────────────────────────────────────────────────

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key  = event.key()
        mods = event.modifiers()
        Ctrl = Qt.KeyboardModifier.ControlModifier

        if key == Qt.Key.Key_Escape:
            if not self._cfg.kiosk:
                self.close()

        elif key == Qt.Key.Key_Space:
            self._engine.toggle_play_pause()

        elif key == Qt.Key.Key_Up:
            self._engine.set_volume(min(150, self._engine.get_volume() + 5))

        elif key == Qt.Key.Key_Down:
            self._engine.set_volume(max(0, self._engine.get_volume() - 5))

        elif key == Qt.Key.Key_Left:
            delta = -60_000 if mods & Ctrl else -10_000
            self._engine.seek_relative(delta)

        elif key == Qt.Key.Key_Right:
            delta = 60_000 if mods & Ctrl else 10_000
            self._engine.seek_relative(delta)

        elif key == Qt.Key.Key_M:
            self._engine.toggle_mute()

        elif key == Qt.Key.Key_L:
            # 루프 토글
            self._cfg.loop = not self._cfg.loop

        elif key == Qt.Key.Key_1:
            self.set_aspect("fit")
        elif key == Qt.Key.Key_2:
            self.set_aspect("fill")
        elif key == Qt.Key.Key_3:
            self.set_aspect("stretch")

        else:
            super().keyPressEvent(event)

    # ── 키오스크: Alt+F4 차단 ─────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        if self._cfg.kiosk:
            event.ignore()
            return
        self._cursor_timer.stop()
        self.unsetCursor()
        # 엔진 신호 연결 해제 (MainWindow가 다시 연결할 것)
        try:
            self._engine.state_changed.disconnect(self._on_state_changed)
        except Exception:
            pass
        self.exiting.emit()
        event.accept()
