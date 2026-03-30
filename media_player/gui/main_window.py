"""
MainWindow v5

[Fix 4] 마우스 이탈 시 제목표시줄까지 영상이 덮도록 수정
        FramelessWindowHint 로 OS 타이틀바 제거
        → 커스텀 TitleBarWidget 오버레이 (마우스 진입 시 페이드인, 이탈 시 페이드아웃)
        → 마우스 이탈: 영상이 창 전체(타이틀바 영역 포함) 완전 점유
        Windows 리사이즈: nativeEvent WM_NCHITTEST 로 OS 엣지 리사이즈 복원
        최대화 오버플로우: changeEvent 에서 setContentsMargins 로 보정

[Fix 2] 잔상 제거: geometry 슬라이드 → opacity 페이드로 교체
        VLC DirectX 렌더링 위에서 위젯을 밀어내면 HWND가 오염됨
        → 위젯 위치는 고정, 투명도만 변경 (HWND 렌더 영역 불변)

[Fix 3] 전체 패널 숨김: titleBar + statusBar + controls 동시 숨김
        숨길 때: controls/titleBar 페이드 아웃 → statusBar 즉시 숨김
        보일 때: statusBar 즉시 표시 → controls/titleBar 페이드 인
        → 영상이 창 전체를 꽉 채움 (OS 타이틀바 영역 포함)
"""

from __future__ import annotations

import ctypes
from pathlib import Path

from PyQt6.QtCore import (
    Qt, QTimer, QPropertyAnimation, QEasingCurve, QEvent, QObject,
)
from PyQt6.QtGui import QAction, QDragEnterEvent, QDropEvent, QKeyEvent, QCursor
from PyQt6.QtWidgets import (
    QApplication, QFileDialog, QMainWindow,
    QMenu, QMessageBox, QStatusBar, QVBoxLayout, QWidget,
)

from gui.video_widget    import VideoWidget
from gui.controls_widget import ControlsWidget
from gui.title_bar       import TitleBarWidget
from gui.theme           import get_stylesheet
from gui.osd_widget      import OsdWidget
from gui.help_window     import HelpWindow
from core.media_engine   import MediaEngine, SUBTITLE_EXT


SUPPORTED_EXT = {
    ".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm",
    ".m4v", ".ts", ".m2ts", ".mpg", ".mpeg", ".3gp", ".ogv",
    ".mp3", ".flac", ".aac", ".ogg", ".wav", ".wma", ".m4a",
}

_HIDE_DELAY   = 2500   # 마우스 정지 후 숨김 대기 (ms)
_FADE_SHOW_MS = 160    # 페이드 인 시간
_FADE_HIDE_MS = 260    # 페이드 아웃 시간


class MainWindow(QMainWindow):

    APP_NAME = "경훈이 플레이어"
    VERSION  = "3.0"

    def __init__(self) -> None:
        super().__init__()
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint)
        self.setWindowTitle(self.APP_NAME)
        self.setMinimumSize(640, 420)
        self.setAcceptDrops(True)
        self.setMouseTracking(True)

        try:
            self._engine = MediaEngine(self)
        except RuntimeError as e:
            QMessageBox.critical(self, "초기화 오류", str(e))
            raise SystemExit(1)

        self._is_fullscreen = False
        self._always_on_top = False
        self._current_file  = ""
        self._is_paused     = False
        self._ui_shown      = False   # 현재 UI 패널 표시 여부

        # 비디오 영역 드래그로 창 이동
        self._drag_start_cursor = None
        self._drag_start_window = None
        self._is_dragging       = False

        self._help_window = HelpWindow(self)

        # [키보드 Fix] 앱 전체 키 이벤트를 MainWindow 가 우선 수신
        QApplication.instance().installEventFilter(self)

        self._build_ui()
        self._build_slide_anim()
        self._build_menu()
        self._connect_signals()
        self.setStyleSheet(get_stylesheet())
        # _build_menu() 가 menuBar 를 자동 표시시키므로 여기서 다시 숨김
        self.menuBar().hide()
        self._status.hide()

        # auto-hide 타이머
        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.setInterval(_HIDE_DELAY)
        self._hide_timer.timeout.connect(self._on_hide_timer)

        # VLC 가 마우스 이벤트를 가로채므로 커서 위치를 직접 폴링
        self._last_cursor = QCursor.pos()
        self._mouse_poll = QTimer(self)
        self._mouse_poll.setInterval(50)   # 50ms = 20fps 감지
        self._mouse_poll.timeout.connect(self._poll_cursor)
        self._mouse_poll.start()

    # ── UI 빌드 ───────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        central = QWidget(self)
        central.setMouseTracking(True)
        self.setCentralWidget(central)

        # ── 레이아웃: TitleBar / Video / Controls 수직 배치 ──────────────
        # UI 표시 시 실제 공간을 차지하고, 숨김 시 Video가 그 공간을 채움
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # 타이틀바 (상단) — 초기 높이 0 (숨김 상태)
        self._title_bar = TitleBarWidget(central)
        self._title_bar.setMouseTracking(True)
        self._title_bar.setMinimumHeight(0)
        self._title_bar.setMaximumHeight(0)
        layout.addWidget(self._title_bar)

        # 비디오 (중앙) — 레이아웃에서 남은 공간 전부 차지
        self._video = VideoWidget(central)
        self._video.setMouseTracking(True)
        layout.addWidget(self._video, 1)

        # 컨트롤 (하단) — 초기 높이 0 (숨김 상태)
        self._controls = ControlsWidget(central)
        self._controls.setMouseTracking(True)
        self._controls.setMinimumHeight(0)
        self._controls.setMaximumHeight(0)
        layout.addWidget(self._controls)

        # OSD
        self._osd = OsdWidget(self._video)

        # 상태바 (숨김 시작)
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage(
            f"{self.APP_NAME} v{self.VERSION}  —  파일을 드래그하거나 Ctrl+O로 열어주세요."
        )
        # HWND
        QTimer.singleShot(0, lambda: self._engine.set_hwnd(self._video.get_hwnd()))

    def _build_slide_anim(self) -> None:
        """UI 슬라이드 애니메이션 — maximumHeight 변경으로 Video 영역 실시간 확장/축소."""
        # 자연 높이 저장 (레이아웃이 확정된 뒤이므로 sizeHint 신뢰 가능)
        self._tb_h   = self._title_bar.sizeHint().height() or 30
        self._ctrl_h = self._controls.sizeHint().height()

        # TitleBar 슬라이드
        self._tb_anim = QPropertyAnimation(self._title_bar, b"maximumHeight")
        self._tb_anim.setEasingCurve(QEasingCurve.Type.InOutCubic)

        # Controls 슬라이드
        self._ctrl_anim = QPropertyAnimation(self._controls, b"maximumHeight")
        self._ctrl_anim.setEasingCurve(QEasingCurve.Type.InOutCubic)
        self._ctrl_anim.finished.connect(self._on_slide_finished)

    # ── 메뉴 빌드 ─────────────────────────────────────────────────────────

    def _build_menu(self) -> None:
        mb = self.menuBar()

        m_file = mb.addMenu("파일(&F)")
        self._act(m_file, "파일 열기(&O)…", "Ctrl+O", self._open_file_dialog)
        m_file.addSeparator()
        self._act(m_file, "종료(&Q)", "Ctrl+Q", self.close)

        m_play = mb.addMenu("재생(&P)")
        self._act(m_play, "재생 / 일시정지", "Space", self._engine.toggle_play_pause)
        self._act(m_play, "정지", "S", self._engine.stop)
        m_play.addSeparator()

        m_speed = m_play.addMenu("재생 속도")
        for rate in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0]:
            self._act(m_speed,
                      f"{'◆ ' if rate == 1.0 else ''}{rate:.2f}x", None,
                      lambda r=rate: self._set_speed(r))
        m_speed.addSeparator()
        self._act(m_speed, "속도 올리기", "]",  self._speed_up)
        self._act(m_speed, "속도 내리기", "[",  self._speed_down)
        self._act(m_speed, "속도 초기화", "\\", self._speed_reset)

        m_play.addSeparator()
        self._act(m_play, "전체화면 토글", "F", self._toggle_fullscreen)
        self._act(m_play, "항상 위 토글",  "T", self._toggle_always_on_top)

        m_sub = mb.addMenu("자막(&S)")
        self._act(m_sub, "자막 파일 열기…", None, self._open_subtitle_dialog)
        m_sub.addSeparator()
        self._act(m_sub, "자막 온/오프", "V", self._engine.toggle_subtitle)
        m_sub.addSeparator()
        self._act(m_sub, "딜레이 +200ms", "J", lambda: self._sub_delay(200))
        self._act(m_sub, "딜레이 -200ms", "H", lambda: self._sub_delay(-200))
        self._act(m_sub, "딜레이 초기화", None, self._sub_delay_reset)

        m_help = mb.addMenu("도움말(&H)")
        self._act(m_help, "단축키 도움말 (F1)", "F1", self._help_window.toggle)
        m_help.addSeparator()
        self._act(m_help, "PyPlayer 정보", None, self._show_about)

    def _act(self, menu: QMenu, label: str, shortcut: str | None, slot) -> QAction:
        a = QAction(label, self)
        if shortcut:
            a.setShortcut(shortcut)
        a.triggered.connect(slot)
        menu.addAction(a)
        return a

    def _connect_signals(self) -> None:
        e = self._engine
        e.position_changed.connect(self._controls.set_position)
        e.time_changed.connect(self._on_time_changed)
        e.duration_changed.connect(self._controls.set_duration)
        e.state_changed.connect(self._on_state_changed)
        e.media_loaded.connect(self._on_media_loaded)
        e.volume_changed.connect(self._controls.set_volume)
        e.speed_changed.connect(self._on_speed_changed)
        e.subtitle_changed.connect(self._on_subtitle_changed)
        e.error_occurred.connect(self._on_error)

        c = self._controls
        c.play_pause_clicked.connect(e.toggle_play_pause)
        c.stop_clicked.connect(e.stop)
        c.seek_begin.connect(e.begin_seek)
        c.seek_end.connect(e.end_seek)
        c.volume_changed.connect(self._on_volume_ctrl_changed)
        c.mute_clicked.connect(self._on_mute_toggle)
        c.skip_forward.connect(lambda: self._seek_with_osd(10_000))
        c.skip_backward.connect(lambda: self._seek_with_osd(-10_000))
        c.mouse_activity.connect(self._on_ctrl_hover)
        self._title_bar.mouse_activity.connect(self._on_ctrl_hover)

        v = self._video
        v.clicked.connect(self._on_video_clicked)
        v.double_clicked.connect(self._toggle_fullscreen)
        v.mouse_moved.connect(self._on_mouse_activity)
        v.wheel_up.connect(lambda: self._volume_step(+5))
        v.wheel_down.connect(lambda: self._volume_step(-5))
        v.right_clicked.connect(self._show_context_menu)

    # ── UI 패널 표시/숨김 ────────────────────────────────────────────────

    def _show_all(self) -> None:
        """TitleBar + Controls 슬라이드 다운 → Video 영역 축소."""
        if self._ui_shown:
            if not self._is_paused:
                self._hide_timer.start()
            return

        self._ui_shown = True

        if not self._is_fullscreen:
            self._status.show()

        # TitleBar 슬라이드 다운 (0 → 자연 높이)
        self._tb_anim.stop()
        self._tb_anim.setDuration(_FADE_SHOW_MS)
        self._tb_anim.setStartValue(self._title_bar.maximumHeight())
        self._tb_anim.setEndValue(self._tb_h)
        self._tb_anim.start()

        # Controls 슬라이드 업 (0 → 자연 높이)
        self._ctrl_anim.stop()
        self._ctrl_anim.setDuration(_FADE_SHOW_MS)
        self._ctrl_anim.setStartValue(self._controls.maximumHeight())
        self._ctrl_anim.setEndValue(self._ctrl_h)
        self._ctrl_anim.start()

        if not self._is_paused:
            self._hide_timer.start()

    def _hide_all(self) -> None:
        """TitleBar + Controls 슬라이드 업 → Video 영역 확장."""
        if not self._ui_shown:
            return
        if self._is_paused:
            return

        self._ui_shown = False
        self._hide_timer.stop()

        # TitleBar 슬라이드 업 (자연 높이 → 0)
        self._tb_anim.stop()
        self._tb_anim.setDuration(_FADE_HIDE_MS)
        self._tb_anim.setStartValue(self._title_bar.maximumHeight())
        self._tb_anim.setEndValue(0)
        self._tb_anim.start()

        # Controls 슬라이드 다운 (자연 높이 → 0)
        self._ctrl_anim.stop()
        self._ctrl_anim.setDuration(_FADE_HIDE_MS)
        self._ctrl_anim.setStartValue(self._controls.maximumHeight())
        self._ctrl_anim.setEndValue(0)
        self._ctrl_anim.start()

        self._ctrl_anim.finished.connect(self._hide_chrome_once)

    def _on_slide_finished(self) -> None:
        """슬라이드 애니메이션 완료 후 영상 영역 강제 갱신 (잔상 제거)."""
        if not self._ui_shown:
            # UI가 숨겨진 직후 → video 영역 전체 다시 그리기
            self._video.update()
            cw = self.centralWidget()
            if cw:
                cw.update()

    def _hide_chrome_once(self) -> None:
        """페이드 아웃 완료 시 1회만 실행."""
        try:
            self._ctrl_anim.finished.disconnect(self._hide_chrome_once)
        except RuntimeError:
            pass
        if not self._ui_shown:   # 그 사이 다시 표시 요청이 없었을 때만 숨김
            self._status.hide()

    # ── auto-hide 로직 ────────────────────────────────────────────────────

    @staticmethod
    def _lbutton_down() -> bool:
        """Win32 GetAsyncKeyState — VLC HWND 가 포커스를 가져도 좌클릭 상태 반환."""
        return bool(ctypes.windll.user32.GetAsyncKeyState(0x01) & 0x8000)

    def _poll_cursor(self) -> None:
        """VLC 가 마우스 이벤트를 가로채므로 커서 위치를 직접 폴링해 UI 표시 + 창 드래그."""
        pos = QCursor.pos()
        btn = self._lbutton_down()

        # ── 드래그 감지 ────────────────────────────────────────────────────
        if btn:
            if self._drag_start_cursor is None:
                # 버튼이 방금 눌림: 비디오 영역 위인지 확인
                from PyQt6.QtCore import QRect
                video_tl = self._video.mapToGlobal(self._video.rect().topLeft())
                video_br = self._video.mapToGlobal(self._video.rect().bottomRight())
                if QRect(video_tl, video_br).contains(pos):
                    self._drag_start_cursor = pos
                    self._drag_start_window = self.pos()
            else:
                delta = pos - self._drag_start_cursor
                if not self._is_dragging and (abs(delta.x()) + abs(delta.y()) > 6):
                    self._is_dragging = True
                if self._is_dragging:
                    self.move(self._drag_start_window + delta)
        else:
            # 버튼 떼짐
            if self._drag_start_cursor is not None:
                self._drag_start_cursor = None
                self._drag_start_window = None
                # 클릭 타이머(250ms)가 먼저 발화하도록 300ms 뒤 초기화
                if self._is_dragging:
                    QTimer.singleShot(300, self._clear_drag)

        # ── 커서 이동 → UI 표시 / 창 이탈 감지 ──────────────────────────
        if pos != self._last_cursor:
            was_inside = self.geometry().contains(self._last_cursor)
            is_inside  = self.geometry().contains(pos)
            self._last_cursor = pos
            if is_inside:
                self._on_mouse_activity()
            elif was_inside:
                # 방금 창 밖으로 나감 (좌/우/상/하 공통)
                self._hide_timer.stop()
                self._hide_all()

    def _clear_drag(self) -> None:
        self._is_dragging = False

    def _on_video_clicked(self) -> None:
        """드래그가 아닐 때만 재생/일시정지."""
        if not self._is_dragging:
            self._engine.toggle_play_pause()

    def _on_mouse_activity(self) -> None:
        """비디오 영역 마우스 이동."""
        self._show_all()

    def _on_ctrl_hover(self) -> None:
        """컨트롤 바 위 마우스 → 숨김 타이머 정지."""
        if not self._ui_shown:
            self._show_all()
        self._hide_timer.stop()

    def _on_hide_timer(self) -> None:
        # 마우스가 컨트롤 바 위에 있으면 타이머 재시작
        if self._ui_shown:
            ctrl_global = self._controls.mapToGlobal(
                self._controls.rect().topLeft()
            )
            from PyQt6.QtCore import QRect
            if QRect(ctrl_global, self._controls.size()).contains(QCursor.pos()):
                self._hide_timer.start()
                return
        self._hide_all()

    # ── 마우스 이벤트 (MainWindow 레벨) ──────────────────────────────────

    def mouseMoveEvent(self, event) -> None:
        self._on_mouse_activity()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event) -> None:
        """창 밖으로 나가면 즉시 숨김."""
        self._hide_timer.stop()
        self._hide_all()
        super().leaveEvent(event)

    # ── 엔진 상태 슬롯 ───────────────────────────────────────────────────

    def _on_time_changed(self, ms: int) -> None:
        self._controls.set_time(ms, self._engine.get_duration_ms())

    def _on_state_changed(self, state: str) -> None:
        self._is_paused = (state == "paused")
        self._controls.set_playing(state == "playing")

        if self._is_paused:
            # 일시정지: 패널 고정 표시
            self._hide_timer.stop()
            self._show_all()
        elif state == "playing":
            # 재생 재개: hide 타이머 시작
            if self._ui_shown:
                self._hide_timer.start()

        label = {
            "playing":   "재생 중",   "paused":    "일시정지",
            "stopped":   "정지",      "ended":     "재생 완료",
            "opening":   "열기 중...", "buffering": "버퍼링...",
            "error":     "오류 발생",
        }.get(state, state)
        name = Path(self._current_file).name if self._current_file else ""
        self._status.showMessage(f"{label}  {name}")

    def _on_media_loaded(self, filename: str) -> None:
        title = f"{filename}  —  {self.APP_NAME}"
        self.setWindowTitle(title)
        self._title_bar.set_title(title)
        self._status.showMessage(f"로드 완료: {filename}")
        self._engine.set_volume(self._controls.vol_slider.value())
        self._engine.play()

    def _on_speed_changed(self, rate: float) -> None:
        self._controls.set_speed(rate)
        self._osd.show_speed(rate)

    def _on_subtitle_changed(self, name: str) -> None:
        self._osd.show_subtitle(name if name else "OFF")
        if name:
            self._status.showMessage(f"자막 로드: {name}")

    def _on_error(self, msg: str) -> None:
        QMessageBox.warning(self, "재생 오류", msg)
        self._status.showMessage(f"오류: {msg[:80]}")

    def _on_mute_toggle(self) -> None:
        self._engine.toggle_mute()
        muted = self._engine.is_muted()
        self._controls.set_muted(muted)
        self._osd.show_volume(self._engine.get_volume(), muted)

    def _on_volume_ctrl_changed(self, vol: int) -> None:
        self._engine.set_volume(vol)
        self._osd.show_volume(vol, self._engine.is_muted())

    # ── 볼륨 / 탐색 ──────────────────────────────────────────────────────

    def _volume_step(self, delta: int) -> None:
        cur = self._controls.vol_slider.value()
        new_vol = max(0, min(150, cur + delta))
        self._controls.vol_slider.setValue(new_vol)
        self._engine.set_volume(new_vol)
        self._osd.show_volume(new_vol, self._engine.is_muted())

    def _seek_with_osd(self, delta_ms: int) -> None:
        self._engine.seek_relative(delta_ms)
        self._osd.show_seek(delta_ms)

    # ── 속도 ──────────────────────────────────────────────────────────────

    def _set_speed(self, rate: float) -> None:
        self._engine.set_speed(rate)

    def _speed_up(self) -> None:
        self._engine.speed_up()

    def _speed_down(self) -> None:
        self._engine.speed_down()

    def _speed_reset(self) -> None:
        self._engine.reset_speed()

    # ── 자막 ──────────────────────────────────────────────────────────────

    def _sub_delay(self, delta_ms: int) -> None:
        self._engine.set_subtitle_delay(delta_ms)
        self._osd.show_subtitle_delay(self._engine.get_subtitle_delay_ms())

    def _sub_delay_reset(self) -> None:
        self._engine.reset_subtitle_delay()
        self._osd.show_subtitle_delay(0)

    def _open_subtitle_dialog(self) -> None:
        ext_str = " ".join(f"*{e}" for e in sorted(SUBTITLE_EXT))
        path, _ = QFileDialog.getOpenFileName(
            self, "자막 파일 열기", "",
            f"자막 파일 ({ext_str});;모든 파일 (*.*)"
        )
        if path:
            ok = self._engine.load_subtitle(path)
            if not ok:
                QMessageBox.warning(self, "자막 오류",
                                    "재생 중에만 자막을 로드할 수 있습니다.")

    # ── 파일 열기 ─────────────────────────────────────────────────────────

    def _open_file_dialog(self) -> None:
        ext_str = " ".join(f"*{e}" for e in sorted(SUPPORTED_EXT))
        path, _ = QFileDialog.getOpenFileName(
            self, "파일 열기", "",
            f"미디어 파일 ({ext_str});;모든 파일 (*.*)"
        )
        if path:
            self._load_file(path)

    def _load_file(self, path: str) -> None:
        self._current_file = path
        self._engine.load(path)

    # ── 전체화면 ──────────────────────────────────────────────────────────

    def _toggle_fullscreen(self) -> None:
        if self._is_fullscreen:
            # showNormal() 은 FramelessWindowHint 를 날리므로 재적용
            self.setWindowFlag(Qt.WindowType.FramelessWindowHint)
            self.showNormal()
            self._is_fullscreen = False
        else:
            self.showFullScreen()
            self._is_fullscreen = True
            self._status.hide()

    # ── 창 크기 프리셋 ────────────────────────────────────────────────────

    # 16:9 기준, 1=120px, 2=240px, 3=360px, 4=480px 가로
    _WIN_SIZES = {1: (120, 68), 2: (240, 135), 3: (360, 203), 4: (480, 270)}

    def _set_window_size(self, n: int) -> None:
        w, h = self._WIN_SIZES[n]
        if self._is_fullscreen:
            self.setWindowFlag(Qt.WindowType.FramelessWindowHint)
            self.showNormal()
            self._is_fullscreen = False
        self.resize(w, h)
        screen = QApplication.primaryScreen().geometry()
        self.move(
            (screen.width()  - w) // 2,
            (screen.height() - h) // 2,
        )
        self._osd.show_message(f"{w} × {h}")

    # ── 항상 위 ───────────────────────────────────────────────────────────

    def _toggle_always_on_top(self) -> None:
        self._always_on_top = not self._always_on_top
        flags = self.windowFlags()
        if self._always_on_top:
            flags |= Qt.WindowType.WindowStaysOnTopHint
        else:
            flags &= ~Qt.WindowType.WindowStaysOnTopHint
        self.setWindowFlags(flags)
        self.show()
        state = "ON" if self._always_on_top else "OFF"
        self._osd.show_message(f"항상 위  {state}")

    # ── 우클릭 컨텍스트 메뉴 ─────────────────────────────────────────────

    def _show_context_menu(self) -> None:
        menu = QMenu(self)
        menu.setStyleSheet(self.styleSheet())

        # 파일 열기 맨 위
        menu.addAction("파일 열기… (Ctrl+O)", self._open_file_dialog)
        menu.addSeparator()
        menu.addAction("재생 / 일시정지", self._engine.toggle_play_pause)
        menu.addAction("정지", self._engine.stop)
        menu.addSeparator()

        m_speed = menu.addMenu("재생 속도")
        for rate in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
            m_speed.addAction(
                f"{'▶ ' if abs(self._engine.get_speed()-rate) < 0.01 else ''}{rate:.2f}x",
                lambda r=rate: self._set_speed(r)
            )

        menu.addSeparator()
        menu.addAction("자막 파일 열기…", self._open_subtitle_dialog)
        menu.addAction("자막 온/오프 (V)", self._engine.toggle_subtitle)
        menu.addSeparator()
        menu.addAction("전체화면 (F)", self._toggle_fullscreen)
        menu.addAction("항상 위 (T)", self._toggle_always_on_top)
        menu.exec(QCursor.pos())

    # ── 드래그 앤 드롭 ────────────────────────────────────────────────────

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            if any(
                Path(u.toLocalFile()).suffix.lower() in SUPPORTED_EXT | SUBTITLE_EXT
                for u in event.mimeData().urls()
            ):
                event.acceptProposedAction()
                return
        event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        for u in event.mimeData().urls():
            fp  = u.toLocalFile()
            ext = Path(fp).suffix.lower()
            if ext in SUBTITLE_EXT:
                self._engine.load_subtitle(fp)
                return
            if ext in SUPPORTED_EXT:
                self._load_file(fp)
                return

    # ── 키보드 이벤트 필터 (앱 레벨) ─────────────────────────────────────

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        """
        자식 위젯(슬라이더·버튼)이 포커스를 가져도
        KeyPress 이벤트를 MainWindow 가 먼저 처리한다.
        단, 다이얼로그·다른 창은 제외.
        """
        if event.type() == QEvent.Type.KeyPress:
            # 이 창의 자식 위젯에서 발생한 키 이벤트만 가로챔
            if isinstance(obj, QWidget) and self.isAncestorOf(obj):
                self.keyPressEvent(event)   # type: ignore[arg-type]
                return True
        return False

    # ── 키보드 단축키 ─────────────────────────────────────────────────────

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key  = event.key()
        mods = event.modifiers()
        Ctrl  = Qt.KeyboardModifier.ControlModifier
        Shift = Qt.KeyboardModifier.ShiftModifier
        Alt   = Qt.KeyboardModifier.AltModifier

        if key == Qt.Key.Key_Left:
            if   mods & Ctrl:  self._seek_with_osd(-60_000)
            elif mods & Shift: self._seek_with_osd(-1_000)
            elif mods & Alt:   self._seek_with_osd(-10_000)
            else:              self._seek_with_osd(-5_000)
        elif key == Qt.Key.Key_Right:
            if   mods & Ctrl:  self._seek_with_osd(60_000)
            elif mods & Shift: self._seek_with_osd(1_000)
            elif mods & Alt:   self._seek_with_osd(10_000)
            else:              self._seek_with_osd(5_000)
        elif key == Qt.Key.Key_Up:
            self._volume_step(+5)
        elif key == Qt.Key.Key_Down:
            self._volume_step(-5)
        elif key == Qt.Key.Key_Space:
            self._engine.toggle_play_pause()
        elif key == Qt.Key.Key_S and not mods:
            self._engine.stop()
        elif key == Qt.Key.Key_Period:
            self._engine.next_frame()
        elif key == Qt.Key.Key_BracketRight:
            self._speed_up()
        elif key == Qt.Key.Key_BracketLeft:
            self._speed_down()
        elif key == Qt.Key.Key_Backslash:
            self._speed_reset()
        elif key == Qt.Key.Key_J:
            self._sub_delay(200)
        elif key == Qt.Key.Key_H:
            self._sub_delay(-200)
        elif key == Qt.Key.Key_V:
            self._engine.toggle_subtitle()
            self._osd.show_message("자막 토글")
        elif key == Qt.Key.Key_F1:
            self._help_window.toggle()
        elif key in (Qt.Key.Key_F, Qt.Key.Key_F11,
                     Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._toggle_fullscreen()
        elif key == Qt.Key.Key_T:
            self._toggle_always_on_top()
        elif key == Qt.Key.Key_Escape and self._is_fullscreen:
            self._toggle_fullscreen()
        elif key == Qt.Key.Key_M:
            self._on_mute_toggle()
        elif Qt.Key.Key_1 <= key <= Qt.Key.Key_4 and not mods:
            self._set_window_size(key - Qt.Key.Key_0)
        elif Qt.Key.Key_5 <= key <= Qt.Key.Key_9 and not mods:
            pct = (key - Qt.Key.Key_0) * 0.1
            self._engine.seek(pct)
            self._osd.show_message(f"위치 이동  {int(pct * 100)}%")
        else:
            super().keyPressEvent(event)

    # ── 정보 ──────────────────────────────────────────────────────────────

    def _show_about(self) -> None:
        QMessageBox.about(
            self, f"{self.APP_NAME} 정보",
            f"<h3>{self.APP_NAME} v{self.VERSION}</h3>"
            "<p>Python + PyQt6 + libVLC 기반 고성능 미디어 플레이어</p>"
        )

    # ── 종료 ──────────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        # Qt 타이머 전부 정지
        self._mouse_poll.stop()
        self._hide_timer.stop()
        # 창을 즉시 숨겨 VLC 정리 중 멈춘 것처럼 보이지 않게 함
        self.hide()
        # cleanup 내부에서 HWND 분리 → stop → 150ms 대기 → release 순으로 처리
        self._engine.cleanup()
        event.accept()
