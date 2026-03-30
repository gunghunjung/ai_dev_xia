"""
Launcher Server GUI
- 폴더/포트/토큰 설정 저장
- 서버 시작/정지
- 실시간 로그 뷰
"""
from __future__ import annotations

import asyncio
import logging
import sys
import threading
from pathlib import Path

# ── 서버 패키지 경로 설정 ──────────────────────────────────────────────────
_SERVER_DIR = Path(__file__).resolve().parent
if str(_SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVER_DIR))

from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QTextCursor
from PyQt6.QtWidgets import (
    QApplication, QFileDialog, QFrame,
    QHBoxLayout, QLabel, QLineEdit,
    QMainWindow, QPushButton, QScrollArea,
    QSizePolicy, QSpinBox, QStatusBar,
    QTextEdit, QVBoxLayout, QWidget,
)

from config import ServerConfig, load_config, save_config


# ── 로그 캡처 핸들러 ───────────────────────────────────────────────────────

class _QtLogHandler(logging.Handler):
    """Python logging → Qt signal."""
    def __init__(self, signal: pyqtSignal) -> None:
        super().__init__()
        self._signal = signal
        self.setFormatter(logging.Formatter("[%(name)s] %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._signal.emit(self.format(record))
        except Exception:
            pass


class _PrintCapture:
    """sys.stdout/stderr → Qt signal."""
    def __init__(self, signal: pyqtSignal, original) -> None:
        self._signal = signal
        self._original = original

    def write(self, text: str) -> None:
        text = text.rstrip("\n")
        if text:
            try:
                self._signal.emit(text)
            except Exception:
                pass
        if self._original:
            self._original.write(text + "\n")

    def flush(self) -> None:
        if self._original:
            self._original.flush()


# ── 서버 실행 스레드 ───────────────────────────────────────────────────────

class _ServerThread(QThread):
    """uvicorn 을 백그라운드 스레드에서 실행."""
    log_line  = pyqtSignal(str)
    started   = pyqtSignal()
    stopped   = pyqtSignal()
    error_msg = pyqtSignal(str)

    def __init__(self, cfg: ServerConfig) -> None:
        super().__init__()
        self._cfg = cfg
        self._uvicorn_server = None
        self._loop: asyncio.AbstractEventLoop | None = None

    # -- 내부 -----------------------------------------------------------

    def run(self) -> None:
        try:
            import uvicorn
            import main as server_main

            # config 주입
            server_main._config = self._cfg

            # 로그 핸들러 부착
            handler = _QtLogHandler(self.log_line)
            for name in ("uvicorn", "uvicorn.access", "uvicorn.error",
                         "fastapi", "watchdog"):
                lg = logging.getLogger(name)
                lg.addHandler(handler)
                lg.setLevel(logging.INFO)

            # stdout 캡처 (scanner 의 print() 등)
            orig_stdout = sys.stdout
            sys.stdout = _PrintCapture(self.log_line, orig_stdout)

            app = server_main.create_app()
            ucfg = uvicorn.Config(
                app,
                host=self._cfg.host,
                port=self._cfg.port,
                log_config=None,   # 자체 핸들러 사용
                log_level="info",
            )
            self._uvicorn_server = uvicorn.Server(ucfg)

            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

            # startup_complete 에 연결해서 started 시그널 발생
            orig_startup = self._uvicorn_server.startup

            async def _patched_startup(sockets=None):
                await orig_startup(sockets)
                self.started.emit()

            self._uvicorn_server.startup = _patched_startup

            self._loop.run_until_complete(self._uvicorn_server.serve())
        except Exception as exc:
            self.error_msg.emit(str(exc))
        finally:
            try:
                sys.stdout = orig_stdout
            except Exception:
                pass
            self.stopped.emit()

    def stop(self) -> None:
        if self._uvicorn_server:
            self._uvicorn_server.should_exit = True
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)


# ── 메인 GUI ──────────────────────────────────────────────────────────────

class ServerWindow(QMainWindow):
    """경훈이 런처 서버 GUI."""

    def __init__(self) -> None:
        super().__init__()
        self._cfg = load_config()
        self._thread: _ServerThread | None = None
        self._running = False

        self.setWindowTitle("경훈이 런처 서버")
        self.setMinimumSize(680, 560)
        self.resize(760, 640)

        self._build_ui()
        self._apply_stylesheet()
        self._load_fields()

    # ── UI 구성 ────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # 상단 타이틀바
        root.addWidget(self._build_topbar())
        # 설정 패널
        root.addWidget(self._build_settings())
        # 구분선
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #2a2a2a;")
        root.addWidget(sep)
        # 로그 영역
        root.addWidget(self._build_log_area(), stretch=1)

        # 상태바
        self._statusbar = QStatusBar()
        self.setStatusBar(self._statusbar)
        self._statusbar.showMessage("서버 정지 상태")

    def _build_topbar(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("TopBar")
        bar.setFixedHeight(64)
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(20, 0, 20, 0)
        lay.setSpacing(14)

        # 제목
        title = QLabel("경훈이 런처 서버")
        title.setStyleSheet("color:#e50914; font-size:18px; font-weight:bold;")
        lay.addWidget(title)
        lay.addStretch()

        # 상태 도트
        self._dot = QLabel("●")
        self._dot.setStyleSheet("color:#555; font-size:18px;")
        lay.addWidget(self._dot)

        # URL 표시
        self._url_label = QLabel("정지 중")
        self._url_label.setStyleSheet("color:#aaa; font-size:12px;")
        lay.addWidget(self._url_label)

        # 복사 버튼
        self._copy_url_btn = QPushButton("URL 복사")
        self._copy_url_btn.setFixedSize(80, 30)
        self._copy_url_btn.setEnabled(False)
        self._copy_url_btn.clicked.connect(self._copy_url)
        lay.addWidget(self._copy_url_btn)

        # 시작/정지 버튼
        self._toggle_btn = QPushButton("▶  서버 시작")
        self._toggle_btn.setFixedSize(120, 36)
        self._toggle_btn.setStyleSheet(
            "QPushButton{background:#e50914;color:#fff;font-weight:bold;"
            "border-radius:6px;font-size:13px;}"
            "QPushButton:hover{background:#ff2222;}"
            "QPushButton:disabled{background:#444;color:#888;}"
        )
        self._toggle_btn.clicked.connect(self._on_toggle)
        lay.addWidget(self._toggle_btn)

        return bar

    def _build_settings(self) -> QWidget:
        box = QFrame()
        box.setObjectName("SettingsBox")
        box.setStyleSheet("QFrame#SettingsBox{background:#1a1a1a; border-bottom:1px solid #2a2a2a;}")
        lay = QVBoxLayout(box)
        lay.setContentsMargins(20, 14, 20, 14)
        lay.setSpacing(10)

        # 제목
        lbl = QLabel("서버 설정")
        lbl.setStyleSheet("color:#bbb; font-size:11px; font-weight:bold;")
        lay.addWidget(lbl)

        grid_w = QWidget()
        grid = QHBoxLayout(grid_w)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(16)

        # ── 루트 폴더 ────────────────────────────────────────────────
        col1 = QVBoxLayout()
        col1.setSpacing(4)
        col1.addWidget(self._field_label("앱 루트 폴더"))
        folder_row = QHBoxLayout()
        folder_row.setSpacing(6)
        self._root_edit = QLineEdit()
        self._root_edit.setPlaceholderText("예: C:/Programs")
        self._root_edit.setMinimumWidth(260)
        folder_row.addWidget(self._root_edit)
        browse_btn = QPushButton("폴더 선택")
        browse_btn.setFixedWidth(80)
        browse_btn.setStyleSheet(
            "QPushButton{background:#333;color:#ddd;border-radius:4px;padding:4px 8px;}"
            "QPushButton:hover{background:#444;}"
        )
        browse_btn.clicked.connect(self._browse_folder)
        folder_row.addWidget(browse_btn)
        col1.addLayout(folder_row)
        grid.addLayout(col1)

        # ── 포트 ─────────────────────────────────────────────────────
        col2 = QVBoxLayout()
        col2.setSpacing(4)
        col2.addWidget(self._field_label("포트"))
        self._port_spin = QSpinBox()
        self._port_spin.setRange(1024, 65535)
        self._port_spin.setValue(8765)
        self._port_spin.setFixedWidth(90)
        self._port_spin.setStyleSheet(
            "QSpinBox{background:#252525;color:#eee;border:1px solid #3a3a3a;"
            "border-radius:4px;padding:4px;}"
        )
        col2.addWidget(self._port_spin)
        grid.addLayout(col2)

        # ── 토큰 ─────────────────────────────────────────────────────
        col3 = QVBoxLayout()
        col3.setSpacing(4)
        col3.addWidget(self._field_label("인증 토큰"))
        token_row = QHBoxLayout()
        token_row.setSpacing(6)
        self._token_edit = QLineEdit()
        self._token_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self._token_edit.setMinimumWidth(160)
        token_row.addWidget(self._token_edit)
        show_btn = QPushButton("표시")
        show_btn.setCheckable(True)
        show_btn.setFixedWidth(48)
        show_btn.setStyleSheet(
            "QPushButton{background:#333;color:#ddd;border-radius:4px;padding:4px 6px;}"
            "QPushButton:checked{background:#555;}"
        )
        show_btn.toggled.connect(
            lambda on: self._token_edit.setEchoMode(
                QLineEdit.EchoMode.Normal if on else QLineEdit.EchoMode.Password
            )
        )
        token_row.addWidget(show_btn)
        col3.addLayout(token_row)
        grid.addLayout(col3)

        grid.addStretch()
        lay.addWidget(grid_w)

        return box

    def _build_log_area(self) -> QWidget:
        box = QWidget()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(16, 10, 16, 10)
        lay.setSpacing(6)

        # 로그 헤더
        hdr = QHBoxLayout()
        lbl = QLabel("서버 로그")
        lbl.setStyleSheet("color:#888; font-size:11px; font-weight:bold;")
        hdr.addWidget(lbl)
        hdr.addStretch()
        clear_btn = QPushButton("지우기")
        clear_btn.setFixedSize(60, 22)
        clear_btn.setStyleSheet(
            "QPushButton{background:#2a2a2a;color:#888;border-radius:3px;font-size:10px;}"
            "QPushButton:hover{background:#333;color:#ccc;}"
        )
        clear_btn.clicked.connect(lambda: self._log_view.clear())
        hdr.addWidget(clear_btn)
        lay.addLayout(hdr)

        # 로그 뷰
        self._log_view = QTextEdit()
        self._log_view.setReadOnly(True)
        self._log_view.setFont(QFont("Consolas", 9))
        self._log_view.setStyleSheet(
            "QTextEdit{background:#0d0d0d;color:#cccccc;"
            "border:1px solid #2a2a2a;border-radius:4px;"
            "selection-background-color:#333;}"
        )
        lay.addWidget(self._log_view)
        return box

    @staticmethod
    def _field_label(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet("color:#999; font-size:10px;")
        return lbl

    # ── 동작 ───────────────────────────────────────────────────────────

    def _load_fields(self) -> None:
        self._root_edit.setText(self._cfg.root_folder)
        self._port_spin.setValue(self._cfg.port)
        self._token_edit.setText(self._cfg.token)

    def _collect_fields(self) -> None:
        self._cfg.root_folder = self._root_edit.text().strip()
        self._cfg.port = self._port_spin.value()
        self._cfg.token = self._token_edit.text().strip() or "launcher-token"
        save_config(self._cfg)

    def _browse_folder(self) -> None:
        d = QFileDialog.getExistingDirectory(
            self, "앱 루트 폴더 선택",
            self._root_edit.text() or str(Path.home()),
        )
        if d:
            self._root_edit.setText(d)

    def _copy_url(self) -> None:
        url = f"http://localhost:{self._cfg.port}"
        QApplication.clipboard().setText(url)
        self._statusbar.showMessage("URL 복사됨", 2000)

    def _on_toggle(self) -> None:
        if self._running:
            self._stop_server()
        else:
            self._start_server()

    def _start_server(self) -> None:
        self._collect_fields()

        if not self._cfg.root_folder:
            self._append_log("⚠ 앱 루트 폴더를 먼저 지정하세요.", color="#ffaa44")
            return

        self._set_controls_enabled(False)
        self._append_log(
            f"서버 시작 중... (포트 {self._cfg.port})", color="#aaddff"
        )

        self._thread = _ServerThread(self._cfg)
        self._thread.log_line.connect(self._append_log)
        self._thread.started.connect(self._on_server_started)
        self._thread.stopped.connect(self._on_server_stopped)
        self._thread.error_msg.connect(lambda m: self._append_log(f"ERROR: {m}", "#ff6666"))
        self._thread.start()

    def _stop_server(self) -> None:
        if self._thread:
            self._append_log("서버 정지 중...", color="#ffaa44")
            self._toggle_btn.setEnabled(False)
            self._thread.stop()

    def _on_server_started(self) -> None:
        self._running = True
        url = f"http://localhost:{self._cfg.port}"
        self._dot.setStyleSheet("color:#4cff8a; font-size:18px;")
        self._url_label.setText(url)
        self._toggle_btn.setText("■  서버 정지")
        self._toggle_btn.setStyleSheet(
            "QPushButton{background:#333;color:#fff;font-weight:bold;"
            "border-radius:6px;font-size:13px;border:1px solid #555;}"
            "QPushButton:hover{background:#444;}"
        )
        self._toggle_btn.setEnabled(True)
        self._copy_url_btn.setEnabled(True)
        self._append_log(f"✓ 서버 실행 중: {url}", color="#4cff8a")
        self._append_log(f"  토큰: {self._cfg.token}", color="#888888")
        self._statusbar.showMessage(f"실행 중: {url}")

    def _on_server_stopped(self) -> None:
        self._running = False
        self._dot.setStyleSheet("color:#555; font-size:18px;")
        self._url_label.setText("정지 중")
        self._toggle_btn.setText("▶  서버 시작")
        self._toggle_btn.setStyleSheet(
            "QPushButton{background:#e50914;color:#fff;font-weight:bold;"
            "border-radius:6px;font-size:13px;}"
            "QPushButton:hover{background:#ff2222;}"
            "QPushButton:disabled{background:#444;color:#888;}"
        )
        self._copy_url_btn.setEnabled(False)
        self._set_controls_enabled(True)
        self._append_log("서버가 정지됐습니다.", color="#ffaa44")
        self._statusbar.showMessage("서버 정지 상태")
        self._thread = None

    def _set_controls_enabled(self, enabled: bool) -> None:
        self._root_edit.setEnabled(enabled)
        self._port_spin.setEnabled(enabled)
        self._token_edit.setEnabled(enabled)
        self._toggle_btn.setEnabled(True)

    def _append_log(self, text: str, color: str = "#cccccc") -> None:
        cursor = self._log_view.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self._log_view.setTextCursor(cursor)
        self._log_view.setTextColor(QColor(color))
        self._log_view.insertPlainText(text + "\n")
        # 자동 스크롤
        self._log_view.verticalScrollBar().setValue(
            self._log_view.verticalScrollBar().maximum()
        )

    # ── 종료 ───────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        if self._running and self._thread:
            self._thread.stop()
            self._thread.wait(3000)
        event.accept()

    # ── 스타일시트 ─────────────────────────────────────────────────────

    def _apply_stylesheet(self) -> None:
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #141414;
                color: #eeeeee;
                font-family: 'Malgun Gothic', 'Segoe UI', sans-serif;
            }
            QFrame#TopBar {
                background-color: #1a1a1a;
                border-bottom: 1px solid #2a2a2a;
            }
            QLineEdit {
                background: #252525;
                color: #eee;
                border: 1px solid #3a3a3a;
                border-radius: 4px;
                padding: 5px 8px;
            }
            QLineEdit:focus {
                border: 1px solid #e50914;
            }
            QStatusBar {
                background: #111;
                color: #888;
                font-size: 10px;
            }
            QScrollBar:vertical {
                background: #1a1a1a;
                width: 8px;
            }
            QScrollBar::handle:vertical {
                background: #3a3a3a;
                border-radius: 4px;
            }
        """)


# ── 진입점 ────────────────────────────────────────────────────────────────

def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("경훈이 런처 서버")
    app.setStyle("Fusion")
    app.setFont(QFont("Malgun Gothic", 10))

    window = ServerWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
