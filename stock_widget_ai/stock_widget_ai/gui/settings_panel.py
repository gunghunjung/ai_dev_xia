"""
Settings Panel
"""
from __future__ import annotations
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QGroupBox,
    QCheckBox, QComboBox, QPushButton, QFileDialog, QLineEdit,
)
from PyQt6.QtCore import pyqtSignal
from ..state_schema import AppState


class SettingsPanel(QWidget):
    reset_requested = pyqtSignal()

    def __init__(self, state: AppState, parent=None) -> None:
        super().__init__(parent)
        self._state = state
        self._build()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        grp = QGroupBox("일반 설정")
        form = QFormLayout()

        self._dark_chk = QCheckBox("Dark Mode")
        self._dark_chk.setChecked(self._state.chart.dark_mode)

        self._log_cb = QComboBox()
        self._log_cb.addItems(["DEBUG", "INFO", "WARNING", "ERROR"])
        self._log_cb.setCurrentText(self._state.log_level)

        self._out_edit = QLineEdit(self._state.output_dir)
        browse_btn = QPushButton("찾아보기")
        browse_btn.clicked.connect(self._browse_output)

        form.addRow("Dark Mode",       self._dark_chk)
        form.addRow("Log Level",       self._log_cb)
        form.addRow("출력 디렉토리",    self._out_edit)
        form.addRow("",                browse_btn)

        reset_btn = QPushButton("⚠️  설정 초기화")
        reset_btn.clicked.connect(self.reset_requested.emit)
        form.addRow("", reset_btn)

        grp.setLayout(form)
        layout.addWidget(grp)
        layout.addStretch()

    def sync_to_state(self, state: AppState) -> None:
        state.chart.dark_mode = self._dark_chk.isChecked()
        state.log_level       = self._log_cb.currentText()
        state.output_dir      = self._out_edit.text()

    def _browse_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "출력 디렉토리 선택")
        if path:
            self._out_edit.setText(path)
