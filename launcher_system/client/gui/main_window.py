from __future__ import annotations

import asyncio
import json
import time
from typing import Optional

from PyQt6.QtCore import (
    Qt,
    QRunnable,
    QThreadPool,
    QThread,
    pyqtSignal,
    pyqtSlot,
    QTimer,
)
from PyQt6.QtGui import QPixmap, QFont
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from config import ClientConfig, save_config
from core.api_client import ApiClient
from core.app_manager import AppManager
from core.downloader import Downloader
from gui.app_card import AppCard
from gui.connect_dialog import ConnectDialog
from gui.download_panel import DownloadPanel


# ─── WebSocket listener ───────────────────────────────────────────────────────

class _WsListener(QThread):
    """Background QThread that connects to ws://server/ws and emits on update."""
    apps_updated = pyqtSignal(list)   # list[dict]
    disconnected = pyqtSignal()

    def __init__(self, ws_url: str, token: str) -> None:
        super().__init__()
        self._ws_url = ws_url
        self._token = token
        self._running = True

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        try:
            import websockets
            import websockets.sync.client as ws_sync
        except ImportError:
            print("[ws] websockets not installed, skipping live updates.")
            return

        while self._running:
            try:
                extra_headers = {"Authorization": f"Bearer {self._token}"}
                with ws_sync.connect(
                    self._ws_url,
                    additional_headers=extra_headers,
                    open_timeout=5,
                ) as ws:
                    print(f"[ws] Connected to {self._ws_url}")
                    while self._running:
                        try:
                            msg = ws.recv(timeout=35)
                            data = json.loads(msg)
                            if data.get("event") == "update":
                                self.apps_updated.emit(data.get("apps", []))
                        except TimeoutError:
                            try:
                                ws.send("ping")
                            except Exception:
                                break
                        except Exception:
                            break
            except Exception as e:
                if self._running:
                    print(f"[ws] Connection error: {e}. Retrying in 5s...")
                    self.disconnected.emit()
                    QThread.sleep(5)
            if not self._running:
                break
        print("[ws] Listener stopped.")


# ─── Icon loader runnable ─────────────────────────────────────────────────────

class _TitleImageLoader(QRunnable):
    """QRunnable: title.jpg 우선 → 없으면 icon 폴백, 백그라운드에서 로드."""

    def __init__(self, api_client: ApiClient, app_id: str, has_title: bool, callback) -> None:
        super().__init__()
        self._api_client = api_client
        self._app_id = app_id
        self._has_title = has_title
        self._callback = callback
        self.setAutoDelete(True)

    def run(self) -> None:
        data: Optional[bytes] = None
        if self._has_title:
            data = self._api_client.get_title_image(self._app_id)
        if not data:
            data = self._api_client.get_icon(self._app_id)
        if data:
            pixmap = QPixmap()
            pixmap.loadFromData(data)
            if not pixmap.isNull():
                self._callback(self._app_id, pixmap)


# ─── Main Window ──────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    """Main application window — Netflix-style program launcher."""

    _title_ready = pyqtSignal(str, QPixmap)  # app_id, pixmap (thread-safe delivery)

    def __init__(self, config: ClientConfig) -> None:
        super().__init__()
        self._config = config
        self._api_client: Optional[ApiClient] = None
        self._app_manager = AppManager(config.apps_dir)
        self._apps_data: list[dict] = []
        self._cards: dict[str, AppCard] = {}           # app_id → AppCard
        self._downloaders: dict[str, Downloader] = {}  # app_id → Downloader
        self._ws_listener: Optional[_WsListener] = None
        self._thread_pool = QThreadPool.globalInstance()

        self._build_ui()
        self._title_ready.connect(self._apply_title_image)

        # Refresh time tracking
        self._last_refresh: float = 0.0
        self._refresh_timer = QTimer(self)
        self._refresh_timer.timeout.connect(self._update_status_bar_time)
        self._refresh_timer.start(30_000)  # update "last refresh" text every 30s

    # ── UI construction ────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.setWindowTitle("경훈이 런처")
        self.setMinimumSize(960, 640)
        self.resize(1200, 760)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ── Top bar ────────────────────────────────────────────────
        top_bar = QFrame()
        top_bar.setObjectName("TopBar")
        top_bar.setFixedHeight(56)
        top_bar_layout = QHBoxLayout(top_bar)
        top_bar_layout.setContentsMargins(20, 0, 20, 0)
        top_bar_layout.setSpacing(12)

        logo = QLabel("경훈이 런처")
        logo.setObjectName("LogoLabel")
        font = logo.font()
        font.setBold(True)
        font.setPointSize(16)
        logo.setFont(font)
        top_bar_layout.addWidget(logo)

        top_bar_layout.addStretch()

        # Status indicator
        self._status_dot = QLabel("●")
        self._status_dot.setFixedWidth(16)
        self._status_dot.setStyleSheet("color: #666666; font-size: 14px;")
        top_bar_layout.addWidget(self._status_dot)

        self._server_url_label = QLabel("미연결")
        self._server_url_label.setObjectName("StatusLabel")
        self._server_url_label.setMaximumWidth(260)
        top_bar_layout.addWidget(self._server_url_label)

        settings_btn = QPushButton("서버 설정")
        settings_btn.setObjectName("SecondaryButton")
        settings_btn.setFixedSize(80, 30)
        settings_btn.clicked.connect(self._on_settings_clicked)
        top_bar_layout.addWidget(settings_btn)

        refresh_btn = QPushButton("새로고침")
        refresh_btn.setObjectName("SecondaryButton")
        refresh_btn.setFixedSize(80, 30)
        refresh_btn.clicked.connect(self._on_refresh_clicked)
        top_bar_layout.addWidget(refresh_btn)

        main_layout.addWidget(top_bar)

        # ── Scroll area with app grid ───────────────────────────────
        self._scroll_area = QScrollArea()
        self._scroll_area.setWidgetResizable(True)
        self._scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        self._grid_container = QWidget()
        self._grid_container.setStyleSheet("background-color: #141414;")
        self._grid_layout = QGridLayout(self._grid_container)
        self._grid_layout.setContentsMargins(20, 20, 20, 20)
        self._grid_layout.setSpacing(16)
        self._grid_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

        # Empty state label
        self._empty_label = QLabel("앱 목록이 없습니다.\n서버에 연결하거나 서버에 앱을 추가하세요.")
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setStyleSheet("color: #555555; font-size: 14px;")
        self._grid_layout.addWidget(
            self._empty_label, 0, 0, 1, 4, Qt.AlignmentFlag.AlignCenter
        )

        self._scroll_area.setWidget(self._grid_container)
        main_layout.addWidget(self._scroll_area, stretch=1)

        # ── Download panel ─────────────────────────────────────────
        self._download_panel = DownloadPanel()
        main_layout.addWidget(self._download_panel)

        # ── Status bar ─────────────────────────────────────────────
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("서버에 연결되지 않음")

    # ── Server connection ──────────────────────────────────────────────────

    def connect_to_server(self, url: str, token: str) -> None:
        """Initialize ApiClient, load apps, start WebSocket listener."""
        # Stop existing WS listener
        self._stop_ws_listener()

        self._api_client = ApiClient(url, token)
        self._config.server_url = url
        self._config.token = token
        self._config.last_connected = True
        save_config(self._config)

        # Update top bar
        short_url = url.replace("http://", "").replace("https://", "")
        if len(short_url) > 30:
            short_url = short_url[:27] + "..."
        self._server_url_label.setText(short_url)
        self._status_dot.setStyleSheet("color: #4cff8a; font-size: 14px;")

        self._status_bar.showMessage(f"연결됨: {url}")

        # Load apps
        self.refresh_apps()

        # Start WS listener
        ws_url = url.replace("http://", "ws://").replace("https://", "wss://") + "/ws"
        self._ws_listener = _WsListener(ws_url, token)
        self._ws_listener.apps_updated.connect(self._on_ws_apps_updated)
        self._ws_listener.disconnected.connect(self._on_ws_disconnected)
        self._ws_listener.start()

    def refresh_apps(self) -> None:
        """Reload app list from server and rebuild cards."""
        if self._api_client is None:
            return
        try:
            self._apps_data = self._api_client.get_apps()
            self._last_refresh = time.time()
            self._rebuild_cards()
            self._load_icons_async()
            self._update_status_bar_time()
        except Exception as e:
            self._status_bar.showMessage(f"앱 목록 로드 실패: {e}")
            QMessageBox.warning(self, "오류", f"앱 목록을 불러오는 데 실패했습니다:\n{e}")

    def _rebuild_cards(self) -> None:
        """Clear grid and recreate all AppCards from current app data."""
        # Remove old cards from layout
        for card in self._cards.values():
            self._grid_layout.removeWidget(card)
            card.deleteLater()
        self._cards.clear()

        if not self._apps_data:
            self._empty_label.setVisible(True)
            return
        self._empty_label.setVisible(False)

        cols = 4
        for idx, app_meta in enumerate(self._apps_data):
            app_id = app_meta.get("id", "")
            installed = self._app_manager.is_installed(app_id)
            installed_version = self._app_manager.get_installed_version(app_id) if installed else None

            update_available = False
            if installed and installed_version and self._api_client:
                try:
                    result = self._api_client.version_check(app_id, installed_version)
                    update_available = result.get("update_available", False)
                except Exception:
                    pass

            card = AppCard()
            card.set_app_data(app_meta, installed, installed_version, update_available)
            card.download_clicked.connect(self._on_download_clicked)
            card.launch_clicked.connect(self._on_launch_clicked)
            card.update_clicked.connect(self._on_update_clicked)
            card.uninstall_clicked.connect(self._on_uninstall_clicked)

            row, col = divmod(idx, cols)
            self._grid_layout.addWidget(card, row, col)
            self._cards[app_id] = card

    def _load_icons_async(self) -> None:
        """title.jpg → icon 순으로 백그라운드에서 이미지 로드."""
        if self._api_client is None:
            return
        for app_meta in self._apps_data:
            app_id = app_meta.get("id", "")
            has_title = bool(app_meta.get("title_image"))
            has_icon  = bool(app_meta.get("icon"))
            if not has_title and not has_icon:
                continue
            loader = _TitleImageLoader(
                self._api_client,
                app_id,
                has_title,
                self._on_title_loaded_from_thread,
            )
            self._thread_pool.start(loader)

    def _on_title_loaded_from_thread(self, app_id: str, pixmap: QPixmap) -> None:
        """worker 스레드에서 호출 — 메인 스레드로 시그널 전달."""
        self._title_ready.emit(app_id, pixmap)

    @pyqtSlot(str, QPixmap)
    def _apply_title_image(self, app_id: str, pixmap: QPixmap) -> None:
        """메인 스레드에서 카드에 이미지 적용."""
        if app_id in self._cards:
            self._cards[app_id].set_title_image(pixmap)

    # ── Card actions ───────────────────────────────────────────────────────

    def _on_download_clicked(self, app_id: str) -> None:
        """Start downloading an app."""
        if app_id in self._downloaders:
            return  # Already downloading
        self._start_download(app_id)

    def _on_launch_clicked(self, app_id: str) -> None:
        """Launch an installed app."""
        app_meta = self._get_app_meta(app_id)
        if app_meta is None:
            return
        entry = app_meta.get("entry", "")
        if not entry:
            QMessageBox.warning(
                self, "실행 오류", f"앱 '{app_id}'의 실행 파일 정보가 없습니다."
            )
            return
        ok = self._app_manager.launch(app_id, entry)
        if not ok:
            QMessageBox.warning(
                self, "실행 오류",
                f"앱을 실행하는 데 실패했습니다.\n경로: {self._app_manager.get_entry_path(app_id, entry)}"
            )

    def _on_update_clicked(self, app_id: str) -> None:
        """Re-download an app to update it."""
        if app_id in self._downloaders:
            return
        self._start_download(app_id)

    def _on_uninstall_clicked(self, app_id: str) -> None:
        """Ask for confirmation, then uninstall the app."""
        app_meta = self._get_app_meta(app_id)
        name = app_meta.get("name", app_id) if app_meta else app_id
        reply = QMessageBox.question(
            self,
            "앱 삭제",
            f"'{name}'을(를) 삭제하시겠습니까?\n이 작업은 되돌릴 수 없습니다.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            try:
                self._app_manager.uninstall(app_id)
            except OSError as e:
                QMessageBox.critical(self, "삭제 오류", f"삭제 중 오류 발생:\n{e}")
                return
            # Refresh this card
            self._refresh_card(app_id)
            self._status_bar.showMessage(f"'{name}' 삭제 완료")

    def _start_download(self, app_id: str) -> None:
        """Create and start a Downloader thread for app_id."""
        if self._api_client is None:
            return
        app_meta = self._get_app_meta(app_id)
        if app_meta is None:
            return
        app_name = app_meta.get("name", app_id)

        downloader = Downloader(self._api_client, app_id, str(self._app_manager.apps_dir))
        downloader.progress.connect(lambda pct, aid=app_id: self._on_download_progress(aid, pct))
        downloader.finished.connect(self._on_download_finished)
        downloader.error.connect(self._on_download_error)
        self._downloaders[app_id] = downloader

        # Show progress in card and download panel
        if app_id in self._cards:
            self._cards[app_id].set_downloading(0)
        self._download_panel.add_download(app_id, app_name)

        downloader.start()

    def _on_download_progress(self, app_id: str, pct: int) -> None:
        if app_id in self._cards:
            self._cards[app_id].set_downloading(pct)
        self._download_panel.update_progress(app_id, pct)

    @pyqtSlot(str, str)
    def _on_download_finished(self, app_id: str, local_path: str) -> None:
        """Save version file, update card, remove from download panel."""
        app_meta = self._get_app_meta(app_id)
        if app_meta:
            version = app_meta.get("version", "1.0.0")
            self._app_manager.save_version(app_id, version)

        if app_id in self._downloaders:
            del self._downloaders[app_id]

        if app_id in self._cards:
            self._cards[app_id].set_downloading(-1)

        self._download_panel.remove_download(app_id)
        self._refresh_card(app_id)
        self._status_bar.showMessage(
            f"'{app_meta.get('name', app_id) if app_meta else app_id}' 다운로드 완료"
        )

    @pyqtSlot(str, str)
    def _on_download_error(self, app_id: str, message: str) -> None:
        """Show error, reset card state."""
        if app_id in self._downloaders:
            del self._downloaders[app_id]

        if app_id in self._cards:
            self._cards[app_id].set_downloading(-1)

        self._download_panel.remove_download(app_id)
        app_meta = self._get_app_meta(app_id)
        name = app_meta.get("name", app_id) if app_meta else app_id
        QMessageBox.critical(
            self, "다운로드 오류", f"'{name}' 다운로드 실패:\n{message}"
        )

    # ── WebSocket events ───────────────────────────────────────────────────

    @pyqtSlot(list)
    def _on_ws_apps_updated(self, apps: list) -> None:
        """Handle live update from server — refresh app list."""
        self._apps_data = apps
        self._last_refresh = time.time()
        self._rebuild_cards()
        self._load_icons_async()
        self._update_status_bar_time()

    @pyqtSlot()
    def _on_ws_disconnected(self) -> None:
        self._status_dot.setStyleSheet("color: #ffaa44; font-size: 14px;")

    # ── Toolbar button handlers ────────────────────────────────────────────

    def _on_settings_clicked(self) -> None:
        """Show server connection dialog."""
        dlg = ConnectDialog(
            self,
            initial_url=self._config.server_url,
            initial_token=self._config.token,
        )
        if dlg.exec():
            result = dlg.get_result()
            if result:
                url, token = result
                self.connect_to_server(url, token)

    def _on_refresh_clicked(self) -> None:
        if self._api_client is None:
            self._on_settings_clicked()
        else:
            self.refresh_apps()

    # ── Helpers ────────────────────────────────────────────────────────────

    def _get_app_meta(self, app_id: str) -> Optional[dict]:
        for meta in self._apps_data:
            if meta.get("id") == app_id:
                return meta
        return None

    def _refresh_card(self, app_id: str) -> None:
        """Refresh a single card's install state."""
        if app_id not in self._cards:
            return
        card = self._cards[app_id]
        app_meta = self._get_app_meta(app_id)
        if app_meta is None:
            return
        installed = self._app_manager.is_installed(app_id)
        installed_version = self._app_manager.get_installed_version(app_id) if installed else None
        update_available = False
        if installed and installed_version and self._api_client:
            try:
                result = self._api_client.version_check(app_id, installed_version)
                update_available = result.get("update_available", False)
            except Exception:
                pass
        card.set_app_data(app_meta, installed, installed_version, update_available)

    def _update_status_bar_time(self) -> None:
        if self._last_refresh > 0:
            elapsed = int(time.time() - self._last_refresh)
            if elapsed < 60:
                time_str = f"{elapsed}초 전"
            elif elapsed < 3600:
                time_str = f"{elapsed // 60}분 전"
            else:
                time_str = f"{elapsed // 3600}시간 전"
            url = self._config.server_url
            short_url = url.replace("http://", "").replace("https://", "")
            self._status_bar.showMessage(
                f"연결됨: {short_url}  |  마지막 갱신: {time_str}"
            )

    def _stop_ws_listener(self) -> None:
        if self._ws_listener is not None:
            self._ws_listener.stop()
            self._ws_listener.quit()
            self._ws_listener.wait(2000)
            self._ws_listener = None

    # ── Window close ──────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._stop_ws_listener()
        # Stop all active downloaders
        for downloader in list(self._downloaders.values()):
            downloader.cancel()
            downloader.quit()
            downloader.wait(1000)
        self._downloaders.clear()
        save_config(self._config)
        super().closeEvent(event)
