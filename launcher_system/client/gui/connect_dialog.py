from __future__ import annotations

from typing import Optional, Tuple

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from core.api_client import ApiClient


class _TestConnectionThread(QThread):
    """Background thread to test server connection without blocking UI."""
    result = pyqtSignal(bool, str)  # success, message

    def __init__(self, url: str, token: str) -> None:
        super().__init__()
        self._url = url
        self._token = token

    def run(self) -> None:
        try:
            client = ApiClient(self._url, self._token)
            ok = client.test_connection()
            if ok:
                self.result.emit(True, "연결 성공!")
            else:
                self.result.emit(False, "연결 실패: 서버 응답 오류 (401 또는 기타)")
        except Exception as e:
            self.result.emit(False, f"연결 실패: {e}")


class ConnectDialog(QDialog):
    """Dialog to enter server URL and auth token, with connection test."""

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        initial_url: str = "",
        initial_token: str = "",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("서버 연결")
        self.setMinimumWidth(420)
        self.setModal(True)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)

        self._url: str = ""
        self._token: str = ""
        self._test_thread: Optional[_TestConnectionThread] = None

        self._build_ui(initial_url, initial_token)

    def _build_ui(self, initial_url: str, initial_token: str) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        # Title
        title = QLabel("서버 연결 설정")
        title.setObjectName("ConnectTitle")
        layout.addWidget(title)

        subtitle = QLabel("런처 서버의 주소와 인증 토큰을 입력하세요.")
        subtitle.setObjectName("ConnectSubtitle")
        layout.addWidget(subtitle)

        # URL field
        url_label = QLabel("서버 URL")
        url_label.setObjectName("FieldLabel")
        layout.addWidget(url_label)

        self._url_edit = QLineEdit()
        self._url_edit.setPlaceholderText("예: http://192.168.0.1:8765")
        self._url_edit.setText(initial_url)
        layout.addWidget(self._url_edit)

        # Token field
        token_label = QLabel("인증 토큰")
        token_label.setObjectName("FieldLabel")
        layout.addWidget(token_label)

        self._token_edit = QLineEdit()
        self._token_edit.setPlaceholderText("예: launcher-token")
        self._token_edit.setText(initial_token)
        self._token_edit.setEchoMode(QLineEdit.EchoMode.Password)
        layout.addWidget(self._token_edit)

        # Test connection button + status
        test_row = QHBoxLayout()
        test_row.setContentsMargins(0, 0, 0, 0)

        self._test_btn = QPushButton("연결 테스트")
        self._test_btn.setObjectName("SecondaryButton")
        self._test_btn.setFixedWidth(110)
        self._test_btn.clicked.connect(self._on_test_clicked)
        test_row.addWidget(self._test_btn)

        self._status_label = QLabel("")
        self._status_label.setObjectName("ConnectStatus")
        test_row.addWidget(self._status_label)
        test_row.addStretch()
        layout.addLayout(test_row)

        # Buttons
        btn_box = QHBoxLayout()
        btn_box.setContentsMargins(0, 8, 0, 0)
        btn_box.addStretch()

        self._cancel_btn = QPushButton("취소")
        self._cancel_btn.setObjectName("SecondaryButton")
        self._cancel_btn.setFixedWidth(80)
        self._cancel_btn.clicked.connect(self.reject)
        btn_box.addWidget(self._cancel_btn)

        self._connect_btn = QPushButton("연결")
        self._connect_btn.setObjectName("PrimaryButton")
        self._connect_btn.setFixedWidth(80)
        self._connect_btn.clicked.connect(self._on_connect_clicked)
        btn_box.addWidget(self._connect_btn)

        layout.addLayout(btn_box)

    def _on_test_clicked(self) -> None:
        url = self._url_edit.text().strip()
        token = self._token_edit.text().strip()
        if not url:
            self._set_status("URL을 입력하세요.", error=True)
            return
        if not token:
            self._set_status("토큰을 입력하세요.", error=True)
            return

        self._test_btn.setEnabled(False)
        self._connect_btn.setEnabled(False)
        self._set_status("연결 중...", error=False, pending=True)

        self._test_thread = _TestConnectionThread(url, token)
        self._test_thread.result.connect(self._on_test_result)
        self._test_thread.start()

    def _on_test_result(self, success: bool, message: str) -> None:
        self._test_btn.setEnabled(True)
        self._connect_btn.setEnabled(True)
        self._set_status(message, error=not success)

    def _set_status(
        self, text: str, error: bool = False, pending: bool = False
    ) -> None:
        if pending:
            self._status_label.setStyleSheet("color: #b3b3b3;")
        elif error:
            self._status_label.setStyleSheet("color: #ff6666;")
        else:
            self._status_label.setStyleSheet("color: #4cff8a;")
        self._status_label.setText(text)

    def _on_connect_clicked(self) -> None:
        url = self._url_edit.text().strip()
        token = self._token_edit.text().strip()
        if not url:
            self._set_status("URL을 입력하세요.", error=True)
            return
        if not token:
            self._set_status("토큰을 입력하세요.", error=True)
            return
        self._url = url
        self._token = token
        self.accept()

    def get_result(self) -> Optional[Tuple[str, str]]:
        """Returns (url, token) if accepted, else None."""
        if self.result() == QDialog.DialogCode.Accepted:
            return self._url, self._token
        return None
