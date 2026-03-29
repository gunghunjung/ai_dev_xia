from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


class _DownloadRow(QWidget):
    """Single row in the download panel showing progress for one app."""

    def __init__(self, app_id: str, app_name: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._app_id = app_id

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)

        self._name_label = QLabel(app_name)
        self._name_label.setFixedWidth(160)
        self._name_label.setStyleSheet("color: #ffffff; font-size: 12px;")
        layout.addWidget(self._name_label)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setFixedHeight(8)
        self._progress_bar.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        layout.addWidget(self._progress_bar)

        self._pct_label = QLabel("0%")
        self._pct_label.setFixedWidth(36)
        self._pct_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._pct_label.setStyleSheet("color: #b3b3b3; font-size: 11px;")
        layout.addWidget(self._pct_label)

        # Cancel button (placeholder — actual cancellation wired by parent if needed)
        self._cancel_btn = QPushButton("✕")
        self._cancel_btn.setObjectName("IconButton")
        self._cancel_btn.setFixedSize(24, 24)
        self._cancel_btn.setToolTip("다운로드 취소")
        layout.addWidget(self._cancel_btn)

    def set_progress(self, value: int) -> None:
        clamped = max(0, min(100, value))
        self._progress_bar.setValue(clamped)
        self._pct_label.setText(f"{clamped}%")

    @property
    def cancel_button(self) -> QPushButton:
        return self._cancel_btn


class DownloadPanel(QWidget):
    """
    Collapsible panel showing active downloads.
    Hidden automatically when no downloads are active.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("DownloadPanel")
        self._rows: dict[str, _DownloadRow] = {}

        self._build_ui()
        self.setVisible(False)

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Header bar
        header = QFrame()
        header.setObjectName("DownloadPanel")
        header.setFixedHeight(30)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(12, 0, 12, 0)

        title = QLabel("다운로드 중")
        title.setObjectName("DownloadPanelTitle")
        header_layout.addWidget(title)
        header_layout.addStretch()
        outer.addWidget(header)

        # Scroll area for rows
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._scroll.setMaximumHeight(120)
        self._scroll.setStyleSheet("border: none; background: transparent;")

        self._rows_widget = QWidget()
        self._rows_widget.setStyleSheet("background: transparent;")
        self._rows_layout = QVBoxLayout(self._rows_widget)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(2)
        self._rows_layout.addStretch()

        self._scroll.setWidget(self._rows_widget)
        outer.addWidget(self._scroll)

    def add_download(self, app_id: str, app_name: str) -> None:
        """Add a new download row."""
        if app_id in self._rows:
            self._rows[app_id].set_progress(0)
            return
        row = _DownloadRow(app_id, app_name)
        self._rows[app_id] = row
        # Insert before the stretch at the end
        count = self._rows_layout.count()
        self._rows_layout.insertWidget(count - 1, row)
        self._update_visibility()

    def update_progress(self, app_id: str, progress: int) -> None:
        """Update progress bar for a download."""
        if app_id in self._rows:
            self._rows[app_id].set_progress(progress)

    def remove_download(self, app_id: str) -> None:
        """Remove a completed or cancelled download row."""
        if app_id in self._rows:
            row = self._rows.pop(app_id)
            self._rows_layout.removeWidget(row)
            row.deleteLater()
        self._update_visibility()

    def _update_visibility(self) -> None:
        """Show panel only when there are active downloads."""
        self.setVisible(bool(self._rows))
