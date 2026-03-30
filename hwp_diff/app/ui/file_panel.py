"""File panel for selecting documents and running comparison."""
import os
from pathlib import Path
from typing import Optional, Callable

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QLineEdit, QFileDialog, QProgressBar,
    QSizePolicy, QFrame,
)
from PySide6.QtCore import Qt, Signal, QUrl
from PySide6.QtGui import QDragEnterEvent, QDragMoveEvent, QDropEvent

_DEFAULT_TEMPLATE = r"F:\template2.xlsx"

SUPPORTED_FILTERS = (
    "지원 문서 파일 (*.hwpx *.hwp *.docx *.txt);;"
    "HWPX 파일 (*.hwpx);;"
    "HWP 파일 (*.hwp);;"
    "Word 파일 (*.docx);;"
    "텍스트 파일 (*.txt);;"
    "모든 파일 (*.*)"
)


SUPPORTED_EXTS = {".hwpx", ".hwp", ".docx", ".txt"}


def _is_valid_drop(event) -> bool:
    """드롭 이벤트가 지원 파일인지 확인."""
    if not event.mimeData().hasUrls():
        return False
    urls = event.mimeData().urls()
    if not urls:
        return False
    path = urls[0].toLocalFile()
    if not path:
        return False
    ext = Path(path).suffix.lower()
    # 확장자 제한 없이 모든 파일 허용 (HWP 등 확장자 다양)
    return True


class FileDropLabel(QLabel):
    """파일 끌어다 놓기를 지원하는 레이블."""

    file_dropped = Signal(str)

    _STYLE_IDLE = (
        "QLabel { border: 2px dashed #AAAAAA; border-radius: 5px; "
        "background: #F8F8F8; color: #888; padding: 8px; "
        "font-size: 10px; min-height: 48px; }"
    )
    _STYLE_HOVER = (
        "QLabel { border: 2px dashed #1F4E79; border-radius: 5px; "
        "background: #D6E8FF; color: #1F4E79; padding: 8px; "
        "font-size: 10px; min-height: 48px; font-weight: bold; }"
    )

    def __init__(self, placeholder: str, parent=None):
        super().__init__(placeholder, parent)
        self.setAcceptDrops(True)
        self.setStyleSheet(self._STYLE_IDLE)
        self.setAlignment(Qt.AlignCenter)
        self.setWordWrap(True)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if _is_valid_drop(event):
            self.setStyleSheet(self._STYLE_HOVER)
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:
        # dragMoveEvent 를 명시적으로 수락해야 dropEvent 가 호출됨
        if _is_valid_drop(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event) -> None:
        self.setStyleSheet(self._STYLE_IDLE)
        event.accept()

    def dropEvent(self, event: QDropEvent) -> None:
        self.setStyleSheet(self._STYLE_IDLE)
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            if path:
                self.file_dropped.emit(path)
                event.acceptProposedAction()
                return
        event.ignore()


class FilePanel(QWidget):
    """
    파일 선택 패널. 드래그앤드롭 + 찾기 버튼 지원.

    Signals:
        compare_requested(old_path, new_path): 비교 실행 버튼 클릭
        export_requested(): 엑셀 저장 버튼 클릭
        options_requested(): 옵션 버튼 클릭
    """

    compare_requested = Signal(str, str)
    export_requested = Signal()
    options_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._old_path: str = ""
        self._new_path: str = ""
        # 패널 자체도 드롭 허용 (QGroupBox가 이벤트를 가로채는 경우 대비)
        self.setAcceptDrops(True)
        self._init_ui()

    # ── 패널 전체를 드롭 영역으로 사용 ──────────────────────────────────
    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if _is_valid_drop(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:
        if _is_valid_drop(event):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:
        """패널 어디에 드롭해도 처리 — Y 좌표 기준으로 기준/비교 문서 판단."""
        urls = event.mimeData().urls()
        if not urls:
            event.ignore()
            return
        path = urls[0].toLocalFile()
        if not path:
            event.ignore()
            return

        # 드롭 위치가 패널 상반부이면 기준문서, 하반부이면 비교문서
        mid_y = self.height() // 2
        pos_y = event.position().y() if hasattr(event, "position") else event.pos().y()
        if pos_y < mid_y:
            self._set_old_path(path)
        else:
            self._set_new_path(path)
        event.acceptProposedAction()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        # --- Title ---
        title = QLabel("문서 변경점 비교기")
        title.setStyleSheet(
            "font-size: 15px; font-weight: bold; color: #1F4E79; padding: 4px 0;"
        )
        layout.addWidget(title)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #CCCCCC;")
        layout.addWidget(sep)

        # --- Old file section ---
        old_group = QGroupBox("기준문서 (원본)")
        old_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        old_layout = QVBoxLayout(old_group)

        self.drop_old = FileDropLabel("여기에 파일을 끌어다 놓으세요\n(.hwpx / .hwp / .docx / .txt)")
        self.drop_old.file_dropped.connect(self._set_old_path)
        old_layout.addWidget(self.drop_old)

        self.edit_old = QLineEdit()
        self.edit_old.setPlaceholderText("파일 경로...")
        self.edit_old.setReadOnly(True)
        self.edit_old.setStyleSheet(
            "QLineEdit { background: #F0F0F0; border: 1px solid #CCC; "
            "border-radius: 3px; padding: 3px; font-size: 10px; }"
        )

        btn_browse_old = QPushButton("찾기")
        btn_browse_old.setFixedWidth(60)
        btn_browse_old.clicked.connect(self._browse_old)

        file_row_old = QHBoxLayout()
        file_row_old.addWidget(self.edit_old)
        file_row_old.addWidget(btn_browse_old)
        old_layout.addLayout(file_row_old)
        layout.addWidget(old_group)

        # --- New file section ---
        new_group = QGroupBox("비교문서 (수정본)")
        new_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        new_layout = QVBoxLayout(new_group)

        self.drop_new = FileDropLabel("여기에 파일을 끌어다 놓으세요\n(.hwpx / .hwp / .docx / .txt)")
        self.drop_new.file_dropped.connect(self._set_new_path)
        new_layout.addWidget(self.drop_new)

        self.edit_new = QLineEdit()
        self.edit_new.setPlaceholderText("파일 경로...")
        self.edit_new.setReadOnly(True)
        self.edit_new.setStyleSheet(
            "QLineEdit { background: #F0F0F0; border: 1px solid #CCC; "
            "border-radius: 3px; padding: 3px; font-size: 10px; }"
        )

        btn_browse_new = QPushButton("찾기")
        btn_browse_new.setFixedWidth(60)
        btn_browse_new.clicked.connect(self._browse_new)

        file_row_new = QHBoxLayout()
        file_row_new.addWidget(self.edit_new)
        file_row_new.addWidget(btn_browse_new)
        new_layout.addLayout(file_row_new)
        layout.addWidget(new_group)

        # --- Document info section ---
        info_group = QGroupBox("문서 정보 (엑셀 출력용)")
        info_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        info_layout = QVBoxLayout(info_group)
        info_layout.setSpacing(4)

        _lbl_w = 55  # label fixed width

        # 품명
        row_product = QHBoxLayout()
        lbl_product = QLabel("품  명:")
        lbl_product.setFixedWidth(_lbl_w)
        self.edit_product = QLineEdit()
        self.edit_product.setPlaceholderText("품명 입력...")
        row_product.addWidget(lbl_product)
        row_product.addWidget(self.edit_product)
        info_layout.addLayout(row_product)

        # 규격서번호/도번
        row_spec = QHBoxLayout()
        lbl_spec = QLabel("도  번:")
        lbl_spec.setFixedWidth(_lbl_w)
        self.edit_spec = QLineEdit()
        self.edit_spec.setPlaceholderText("규격서번호 또는 도번...")
        row_spec.addWidget(lbl_spec)
        row_spec.addWidget(self.edit_spec)
        info_layout.addLayout(row_spec)

        # 양식 파일 (template)
        row_tpl = QHBoxLayout()
        lbl_tpl = QLabel("양  식:")
        lbl_tpl.setFixedWidth(_lbl_w)
        self.edit_template = QLineEdit()
        self.edit_template.setPlaceholderText("QAR 양식 파일 (.xlsx)...")
        self.edit_template.setReadOnly(True)
        self.edit_template.setStyleSheet(
            "QLineEdit { background: #F0F0F0; border: 1px solid #CCC; "
            "border-radius: 3px; padding: 2px; font-size: 9px; }"
        )
        # set default template path if it exists
        if Path(_DEFAULT_TEMPLATE).exists():
            self.edit_template.setText(_DEFAULT_TEMPLATE)
        btn_browse_tpl = QPushButton("찾기")
        btn_browse_tpl.setFixedWidth(45)
        btn_browse_tpl.clicked.connect(self._browse_template)
        row_tpl.addWidget(lbl_tpl)
        row_tpl.addWidget(self.edit_template)
        row_tpl.addWidget(btn_browse_tpl)
        info_layout.addLayout(row_tpl)

        layout.addWidget(info_group)

        # --- Buttons ---
        self.btn_compare = QPushButton("비교 실행")
        self.btn_compare.setMinimumHeight(40)
        self.btn_compare.setStyleSheet(
            "QPushButton { background: #1F4E79; color: white; font-size: 13px; "
            "font-weight: bold; border-radius: 5px; padding: 6px; }"
            "QPushButton:hover { background: #2E6096; }"
            "QPushButton:disabled { background: #AAAAAA; }"
        )
        self.btn_compare.clicked.connect(self._on_compare)
        self.btn_compare.setEnabled(False)
        layout.addWidget(self.btn_compare)

        action_row = QHBoxLayout()

        self.btn_export = QPushButton("엑셀 저장")
        self.btn_export.setMinimumHeight(32)
        self.btn_export.setStyleSheet(
            "QPushButton { background: #375623; color: white; font-size: 11px; "
            "border-radius: 4px; padding: 4px 8px; }"
            "QPushButton:hover { background: #4E7A30; }"
            "QPushButton:disabled { background: #AAAAAA; }"
        )
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self.export_requested.emit)

        self.btn_options = QPushButton("옵션")
        self.btn_options.setMinimumHeight(32)
        self.btn_options.setStyleSheet(
            "QPushButton { background: #555555; color: white; font-size: 11px; "
            "border-radius: 4px; padding: 4px 8px; }"
            "QPushButton:hover { background: #777777; }"
        )
        self.btn_options.clicked.connect(self.options_requested.emit)

        action_row.addWidget(self.btn_export)
        action_row.addWidget(self.btn_options)
        layout.addLayout(action_row)

        # --- Progress bar ---
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet(
            "QProgressBar { border: 1px solid #CCC; border-radius: 3px; "
            "text-align: center; height: 16px; }"
            "QProgressBar::chunk { background: #1F4E79; }"
        )
        layout.addWidget(self.progress_bar)

        # --- Status label ---
        self.lbl_status = QLabel("")
        self.lbl_status.setStyleSheet("color: #444; font-size: 10px;")
        self.lbl_status.setWordWrap(True)
        layout.addWidget(self.lbl_status)

        layout.addStretch()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_progress(self, pct: int, message: str) -> None:
        """Update progress bar and status message."""
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(pct)
        self.lbl_status.setText(message)
        if pct >= 100:
            self.progress_bar.setVisible(False)

    def set_export_enabled(self, enabled: bool) -> None:
        self.btn_export.setEnabled(enabled)

    def set_compare_enabled(self, enabled: bool) -> None:
        self.btn_compare.setEnabled(enabled)

    def get_paths(self):
        """Return (old_path, new_path) tuple."""
        return self._old_path, self._new_path

    def get_product_name(self) -> str:
        """Return product name entered by user."""
        return self.edit_product.text().strip()

    def get_spec_number(self) -> str:
        """Return spec/drawing number entered by user."""
        return self.edit_spec.text().strip()

    def get_template_path(self) -> str:
        """Return selected QAR template file path."""
        return self.edit_template.text().strip()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _browse_template(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "QAR 양식 파일 선택", "", "Excel 파일 (*.xlsx);;모든 파일 (*.*)"
        )
        if path:
            self.edit_template.setText(path)

    def _browse_old(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "기준문서 선택", "", SUPPORTED_FILTERS
        )
        if path:
            self._set_old_path(path)

    def _browse_new(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "비교문서 선택", "", SUPPORTED_FILTERS
        )
        if path:
            self._set_new_path(path)

    def _set_old_path(self, path: str) -> None:
        self._old_path = path
        name = Path(path).name
        self.edit_old.setText(path)
        self.drop_old.setText(f"✓ {name}")
        self.drop_old.setStyleSheet(
            "QLabel { border: 2px solid #1F4E79; border-radius: 5px; "
            "background: #EEF4FF; color: #1F4E79; padding: 5px; "
            "font-size: 10px; min-height: 36px; font-weight: bold; }"
        )
        self._update_compare_btn()

    def _set_new_path(self, path: str) -> None:
        self._new_path = path
        name = Path(path).name
        self.edit_new.setText(path)
        self.drop_new.setText(f"✓ {name}")
        self.drop_new.setStyleSheet(
            "QLabel { border: 2px solid #375623; border-radius: 5px; "
            "background: #EEF8EE; color: #375623; padding: 5px; "
            "font-size: 10px; min-height: 36px; font-weight: bold; }"
        )
        self._update_compare_btn()

    def _update_compare_btn(self) -> None:
        self.btn_compare.setEnabled(bool(self._old_path and self._new_path))

    def _on_compare(self) -> None:
        if self._old_path and self._new_path:
            self.compare_requested.emit(self._old_path, self._new_path)
