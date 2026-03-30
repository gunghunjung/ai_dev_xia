"""Main application window."""
import os
import json
from pathlib import Path
from typing import Optional, List

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QSplitter, QStatusBar, QMenuBar, QToolBar,
    QFileDialog, QMessageBox, QApplication, QVBoxLayout, QLabel,
)
from PySide6.QtCore import Qt, QThread, Signal, QObject, QSettings, QSize
from PySide6.QtGui import QAction, QKeySequence, QIcon, QFont

from app.ui.file_panel import FilePanel
from app.ui.compare_view import CompareView
from app.ui.change_list import ChangeListWidget, ChangeListDialog
from app.ui.options_dialog import OptionsDialog
from app.core.controller import CompareController
from app.models.change_record import CompareResult, ChangeRecord
from app.models.document import DocumentStructure
from app.utils.logger import get_logger

logger = get_logger("ui.main_window")

MAX_RECENT_FILES = 8


class CompareWorker(QObject):
    """Worker thread for running document comparison."""

    progress = Signal(int, str)
    finished = Signal(object, object, object)  # old_doc, new_doc, result
    error = Signal(str)

    def __init__(
        self,
        old_path: str,
        new_path: str,
        options: dict,
    ):
        super().__init__()
        self.old_path = old_path
        self.new_path = new_path
        self.options = options

    def run(self) -> None:
        try:
            controller = CompareController()
            controller.set_progress_callback(self.progress.emit)
            result = controller.run_compare_from_paths(
                self.old_path, self.new_path, self.options
            )
            old_doc, new_doc = controller.get_loaded_documents()
            self.finished.emit(old_doc, new_doc, result)
        except Exception as e:
            logger.exception("Compare failed: %s", e)
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    """
    Main application window with:
    - Menu bar
    - Toolbar
    - 3-panel layout (file/options | compare view | change list)
    - Status bar
    - Recent files
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("문서 변경점 비교기")
        self.setMinimumSize(1200, 700)
        # Windows에서 탐색기 드래그앤드롭이 동작하려면
        # 최상위 QMainWindow에 반드시 setAcceptDrops(True) 필요
        # (Qt가 HWND를 OLE 드롭 대상으로 등록함)
        self.setAcceptDrops(True)

        self._controller = CompareController()
        self._options: dict = {
            "ignore_whitespace": False,
            "ignore_case": False,
            "ignore_newline": False,
            "include_format_changes": True,
            "detect_moves": True,
            "similarity_threshold": 0.6,
            "table_sensitivity": 0.5,
            "important_keywords": [],
        }
        self._last_result: Optional[CompareResult] = None
        self._old_doc: Optional[DocumentStructure] = None
        self._new_doc: Optional[DocumentStructure] = None
        self._worker_thread: Optional[QThread] = None

        self._settings = QSettings("HwpDiff", "DocumentDiff")
        self._recent_files: List[tuple] = []  # List of (old_path, new_path)

        self._init_ui()
        self._init_menus()
        self._init_toolbar()
        self._load_settings()

    def _init_ui(self):
        """Initialize central widget and panels."""
        # 모달리스 변경목록 다이얼로그 (비교 완료 전까지 숨김)
        self.change_list_dialog = ChangeListDialog(self)
        self.change_list_dialog.change_selected.connect(self._on_change_selected)

        # Central splitter: file panel | compare view (2-panel, 목록은 별도 창)
        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.setHandleWidth(3)

        # Left: file panel
        self.file_panel = FilePanel()
        self.file_panel.setMinimumWidth(240)
        self.file_panel.setMaximumWidth(320)
        self.file_panel.compare_requested.connect(self._on_compare_requested)
        self.file_panel.export_requested.connect(self._on_export_requested)
        self.file_panel.options_requested.connect(self._on_options_requested)

        # Center: compare view (전체 너비 사용)
        self.compare_view = CompareView()

        main_splitter.addWidget(self.file_panel)
        main_splitter.addWidget(self.compare_view)
        main_splitter.setSizes([270, 900])

        self.setCentralWidget(main_splitter)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.lbl_status = QLabel("준비")
        self.status_bar.addWidget(self.lbl_status)

    def _init_menus(self):
        """Create menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("파일(&F)")

        self.action_open_old = QAction("기준문서 열기...", self)
        self.action_open_old.setShortcut(QKeySequence("Ctrl+O"))
        self.action_open_old.triggered.connect(self._on_open_old)
        file_menu.addAction(self.action_open_old)

        self.action_open_new = QAction("비교문서 열기...", self)
        self.action_open_new.setShortcut(QKeySequence("Ctrl+Shift+O"))
        self.action_open_new.triggered.connect(self._on_open_new)
        file_menu.addAction(self.action_open_new)

        file_menu.addSeparator()

        self.action_compare = QAction("비교 실행(&C)", self)
        self.action_compare.setShortcut(QKeySequence("F5"))
        self.action_compare.setEnabled(False)
        self.action_compare.triggered.connect(self._on_compare_action)
        file_menu.addAction(self.action_compare)

        file_menu.addSeparator()

        self.action_export = QAction("Excel로 내보내기(&E)...", self)
        self.action_export.setShortcut(QKeySequence("Ctrl+S"))
        self.action_export.setEnabled(False)
        self.action_export.triggered.connect(self._on_export_requested)
        file_menu.addAction(self.action_export)

        file_menu.addSeparator()

        # Recent files submenu
        self.recent_menu = file_menu.addMenu("최근 비교 파일")
        self._update_recent_menu()

        file_menu.addSeparator()

        action_quit = QAction("종료(&Q)", self)
        action_quit.setShortcut(QKeySequence("Alt+F4"))
        action_quit.triggered.connect(self.close)
        file_menu.addAction(action_quit)

        # View menu
        view_menu = menubar.addMenu("보기(&V)")

        action_zoom_in = QAction("확대", self)
        action_zoom_in.setShortcut(QKeySequence("Ctrl+="))
        action_zoom_in.triggered.connect(self._zoom_in)
        view_menu.addAction(action_zoom_in)

        action_zoom_out = QAction("축소", self)
        action_zoom_out.setShortcut(QKeySequence("Ctrl+-"))
        action_zoom_out.triggered.connect(self._zoom_out)
        view_menu.addAction(action_zoom_out)

        view_menu.addSeparator()

        action_options = QAction("비교 옵션...", self)
        action_options.triggered.connect(self._on_options_requested)
        view_menu.addAction(action_options)

        # Help menu
        help_menu = menubar.addMenu("도움말(&H)")

        action_about = QAction("정보(&A)", self)
        action_about.triggered.connect(self._on_about)
        help_menu.addAction(action_about)

    def _init_toolbar(self):
        """Create main toolbar."""
        toolbar = self.addToolBar("주요 도구")
        toolbar.setObjectName("main_toolbar")   # saveState() 경고 방지
        toolbar.setIconSize(QSize(24, 24))
        toolbar.setMovable(False)

        btn_style = (
            "QToolButton { padding: 4px 10px; margin: 2px; "
            "border-radius: 3px; font-size: 11px; }"
        )
        toolbar.setStyleSheet(btn_style)

        toolbar.addAction(self.action_compare)
        toolbar.addSeparator()
        toolbar.addAction(self.action_export)
        toolbar.addSeparator()

        # 변경목록 버튼 — 비교 완료 후 활성화
        self.action_show_changes = QAction("변경목록 보기", self)
        self.action_show_changes.setEnabled(False)
        self.action_show_changes.triggered.connect(self._on_show_change_list)
        toolbar.addAction(self.action_show_changes)
        toolbar.addSeparator()

        toolbar.addAction(QAction("옵션", self, triggered=self._on_options_requested))

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_open_old(self) -> None:
        from app.ui.file_panel import SUPPORTED_FILTERS
        path, _ = QFileDialog.getOpenFileName(self, "기준문서 선택", "", SUPPORTED_FILTERS)
        if path:
            self.file_panel._set_old_path(path)

    def _on_open_new(self) -> None:
        from app.ui.file_panel import SUPPORTED_FILTERS
        path, _ = QFileDialog.getOpenFileName(self, "비교문서 선택", "", SUPPORTED_FILTERS)
        if path:
            self.file_panel._set_new_path(path)

    def _on_compare_action(self) -> None:
        old_path, new_path = self.file_panel.get_paths()
        if old_path and new_path:
            self._start_compare(old_path, new_path)

    def _on_compare_requested(self, old_path: str, new_path: str) -> None:
        self._start_compare(old_path, new_path)

    def _start_compare(self, old_path: str, new_path: str) -> None:
        """Launch comparison in a background thread."""
        if self._worker_thread and self._worker_thread.isRunning():
            QMessageBox.warning(self, "진행 중", "이미 비교 작업이 진행 중입니다.")
            return

        # Disable buttons
        self.file_panel.set_compare_enabled(False)
        self.file_panel.set_export_enabled(False)
        self.action_compare.setEnabled(False)
        self.action_export.setEnabled(False)
        self.compare_view.clear()
        self.change_list_dialog.clear()
        self.action_show_changes.setEnabled(False)

        self.lbl_status.setText("비교 중...")
        self.file_panel.set_progress(0, "준비 중...")

        # Create worker
        self._worker = CompareWorker(old_path, new_path, self._options)
        self._worker_thread = QThread()
        self._worker.moveToThread(self._worker_thread)
        self._worker_thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_compare_finished)
        self._worker.error.connect(self._on_compare_error)
        self._worker.finished.connect(self._worker_thread.quit)
        self._worker.error.connect(self._worker_thread.quit)
        self._worker_thread.start()

        # Update window title
        old_name = Path(old_path).name
        new_name = Path(new_path).name
        self.setWindowTitle(f"문서 변경점 비교기 - [{old_name}] vs [{new_name}]")

    def _on_progress(self, pct: int, message: str) -> None:
        self.file_panel.set_progress(pct, message)
        self.lbl_status.setText(message)

    def _on_compare_finished(
        self,
        old_doc: DocumentStructure,
        new_doc: DocumentStructure,
        result: CompareResult,
    ) -> None:
        """Handle successful comparison completion."""
        self._old_doc = old_doc
        self._new_doc = new_doc
        self._last_result = result

        # Update UI
        old_path, new_path = self.file_panel.get_paths()
        old_name = Path(old_path).name if old_path else "기준문서"
        new_name = Path(new_path).name if new_path else "비교문서"

        self.compare_view.set_documents(old_doc, new_doc, result.changes, old_name, new_name)

        # 모달리스 변경목록 다이얼로그에 데이터 전달
        self.change_list_dialog.set_changes(result.changes)

        # Re-enable buttons
        self.file_panel.set_compare_enabled(True)
        self.file_panel.set_export_enabled(True)
        self.action_compare.setEnabled(True)
        self.action_export.setEnabled(True)
        self.action_show_changes.setEnabled(True)   # 변경목록 버튼 활성화

        stats = result.get_summary_stats()
        total_changes = len(result.changes)
        similarity = result.overall_similarity
        self.lbl_status.setText(
            f"비교 완료 | 총 {total_changes}건 변경 | "
            f"전체 유사도 {similarity:.1%} | "
            f"추가: {result.added_count} 삭제: {result.deleted_count} "
            f"수정: {result.modified_count}"
        )
        self.file_panel.set_progress(100, f"완료: {total_changes}건 변경 발견")

        # Add to recent files
        if old_path and new_path:
            self._add_recent(old_path, new_path)

        logger.info("Compare UI update complete: %d changes", total_changes)

    def _on_compare_error(self, error_msg: str) -> None:
        """Handle comparison error."""
        self.file_panel.set_compare_enabled(True)
        self.action_compare.setEnabled(True)
        self.lbl_status.setText(f"오류: {error_msg}")
        self.file_panel.set_progress(0, "")

        QMessageBox.critical(
            self,
            "비교 오류",
            f"문서 비교 중 오류가 발생했습니다:\n\n{error_msg}",
        )

    def _on_export_requested(self) -> None:
        """Export results to Excel."""
        if not self._last_result:
            QMessageBox.information(self, "알림", "내보낼 비교 결과가 없습니다. 먼저 비교를 실행하세요.")
            return

        old_path, new_path = self.file_panel.get_paths()
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"세부항목내역서_{timestamp}.xlsx"

        output_path, _ = QFileDialog.getSaveFileName(
            self, "Excel 파일 저장", default_name,
            "Excel 파일 (*.xlsx);;모든 파일 (*.*)"
        )
        if not output_path:
            return

        old_name = Path(old_path).name if old_path else "기준문서"
        new_name = Path(new_path).name if new_path else "비교문서"

        success = self._controller.export_excel(
            self._last_result,
            output_path,
            old_name,
            new_name,
            product_name=self.file_panel.get_product_name(),
            spec_number=self.file_panel.get_spec_number(),
            template_path=self.file_panel.get_template_path(),
        )
        if success:
            QMessageBox.information(
                self, "내보내기 완료",
                f"Excel 파일이 저장되었습니다:\n{output_path}"
            )
            self.lbl_status.setText(f"Excel 저장: {output_path}")
        else:
            QMessageBox.critical(
                self, "저장 실패",
                "Excel 파일 저장에 실패했습니다. 로그를 확인하세요."
            )

    def _on_options_requested(self) -> None:
        """Show options dialog."""
        dialog = OptionsDialog(self._options, parent=self)
        if dialog.exec():
            self._options = dialog.get_options()
            logger.info("Options updated: %s", self._options)

    def _on_show_change_list(self) -> None:
        """변경목록 다이얼로그 표시."""
        self.change_list_dialog.show()
        self.change_list_dialog.raise_()
        self.change_list_dialog.activateWindow()

    def _on_change_selected(self, change: ChangeRecord) -> None:
        """변경목록에서 항목 선택 시 뷰어 연동."""
        self.compare_view.highlight_change(change)
        # 메인 창을 앞으로
        self.raise_()
        self.activateWindow()

    def _on_about(self) -> None:
        QMessageBox.about(
            self,
            "문서 변경점 비교기 정보",
            "<h3>문서 변경점 비교기 v1.0.0</h3>"
            "<p>HWP/HWPX/DOCX/TXT 문서를 비교하여 변경사항을 분석하고 Excel로 내보냅니다.</p>"
            "<p>지원 형식: HWPX, HWP (변환필요), DOCX, TXT</p>"
        )

    # ------------------------------------------------------------------
    # Zoom
    # ------------------------------------------------------------------

    def _zoom_in(self) -> None:
        for view in (self.compare_view.text_old, self.compare_view.text_new):
            font = view.font()
            font.setPointSize(min(24, font.pointSize() + 1))
            view.setFont(font)

    def _zoom_out(self) -> None:
        for view in (self.compare_view.text_old, self.compare_view.text_new):
            font = view.font()
            font.setPointSize(max(6, font.pointSize() - 1))
            view.setFont(font)

    # ------------------------------------------------------------------
    # Recent files
    # ------------------------------------------------------------------

    def _add_recent(self, old_path: str, new_path: str) -> None:
        entry = (old_path, new_path)
        if entry in self._recent_files:
            self._recent_files.remove(entry)
        self._recent_files.insert(0, entry)
        self._recent_files = self._recent_files[:MAX_RECENT_FILES]
        self._update_recent_menu()
        self._save_settings()

    def _update_recent_menu(self) -> None:
        self.recent_menu.clear()
        if not self._recent_files:
            action = QAction("(없음)", self)
            action.setEnabled(False)
            self.recent_menu.addAction(action)
            return

        for old_path, new_path in self._recent_files:
            old_name = Path(old_path).name
            new_name = Path(new_path).name
            label = f"{old_name} vs {new_name}"
            action = QAction(label, self)
            action.setData((old_path, new_path))
            action.triggered.connect(
                lambda checked, op=old_path, np=new_path:
                    self._load_recent(op, np)
            )
            self.recent_menu.addAction(action)

    def _load_recent(self, old_path: str, new_path: str) -> None:
        if not Path(old_path).exists():
            QMessageBox.warning(self, "파일 없음", f"파일을 찾을 수 없습니다:\n{old_path}")
            return
        if not Path(new_path).exists():
            QMessageBox.warning(self, "파일 없음", f"파일을 찾을 수 없습니다:\n{new_path}")
            return
        self.file_panel._set_old_path(old_path)
        self.file_panel._set_new_path(new_path)

    # ------------------------------------------------------------------
    # Settings persistence
    # ------------------------------------------------------------------

    def _save_settings(self) -> None:
        try:
            self._settings.setValue("geometry", self.saveGeometry())
            self._settings.setValue("windowState", self.saveState())
            recent_data = json.dumps(self._recent_files)
            self._settings.setValue("recentFiles", recent_data)
            self._settings.setValue("options", json.dumps(self._options))
        except Exception as e:
            logger.debug("Save settings error: %s", e)

    def _load_settings(self) -> None:
        try:
            geom = self._settings.value("geometry")
            if geom:
                self.restoreGeometry(geom)
            state = self._settings.value("windowState")
            if state:
                self.restoreState(state)
            recent_data = self._settings.value("recentFiles", "[]")
            self._recent_files = json.loads(recent_data)
            self._update_recent_menu()

            opts_data = self._settings.value("options", "")
            if opts_data:
                loaded_opts = json.loads(opts_data)
                self._options.update(loaded_opts)
        except Exception as e:
            logger.debug("Load settings error: %s", e)

    def closeEvent(self, event) -> None:
        self._save_settings()
        super().closeEvent(event)
