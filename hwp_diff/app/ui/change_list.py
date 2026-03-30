"""Change list widget - shows all detected changes with filtering and search."""
from typing import List, Optional, Callable

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTreeWidget, QTreeWidgetItem,
    QLabel, QLineEdit, QComboBox, QPushButton, QHeaderView,
    QDialog, QSizePolicy,
)
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QColor, QFont, QBrush

from app.models.change_record import ChangeRecord, ChangeType, Importance


_CHANGE_TYPE_COLORS = {
    ChangeType.ADD: QColor("#DDEEFF"),
    ChangeType.DELETE: QColor("#FFDDDD"),
    ChangeType.MODIFY: QColor("#FFFACD"),
    ChangeType.MOVE: QColor("#F0E6FF"),
    ChangeType.FORMAT: QColor("#E8F5E9"),
}

_CHANGE_TYPE_LABELS = {
    ChangeType.ADD: "추가",
    ChangeType.DELETE: "삭제",
    ChangeType.MODIFY: "수정",
    ChangeType.MOVE: "이동",
    ChangeType.FORMAT: "서식",
}


class ChangeListWidget(QWidget):
    """
    Widget displaying all change records in a tree/list view.

    Signals:
        change_selected(ChangeRecord): Emitted when a change is clicked.
    """

    change_selected = Signal(object)  # ChangeRecord

    def __init__(self, parent=None):
        super().__init__(parent)
        self._all_changes: List[ChangeRecord] = []
        self._filtered_changes: List[ChangeRecord] = []
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # --- Header ---
        header_layout = QHBoxLayout()
        self.lbl_title = QLabel("변경사항 목록")
        self.lbl_title.setStyleSheet("font-weight: bold; font-size: 13px;")
        self.lbl_count = QLabel("0건")
        self.lbl_count.setStyleSheet("color: #666; font-size: 11px;")
        header_layout.addWidget(self.lbl_title)
        header_layout.addStretch()
        header_layout.addWidget(self.lbl_count)
        layout.addLayout(header_layout)

        # --- Filter toolbar ---
        filter_layout = QHBoxLayout()

        self.combo_filter = QComboBox()
        self.combo_filter.addItems([
            "전체", "추가", "삭제", "수정", "이동", "서식변경", "표변경", "제목변경"
        ])
        self.combo_filter.currentTextChanged.connect(self._apply_filters)
        self.combo_filter.setFixedWidth(90)

        self.edit_search = QLineEdit()
        self.edit_search.setPlaceholderText("검색...")
        self.edit_search.textChanged.connect(self._apply_filters)
        self.edit_search.setClearButtonEnabled(True)

        filter_layout.addWidget(QLabel("유형:"))
        filter_layout.addWidget(self.combo_filter)
        filter_layout.addWidget(self.edit_search)
        layout.addLayout(filter_layout)

        # --- Tree widget ---
        self.tree = QTreeWidget()
        self.tree.setColumnCount(4)
        self.tree.setHeaderLabels(["#", "유형", "위치", "요약"])
        self.tree.header().setSectionResizeMode(0, QHeaderView.Fixed)
        self.tree.header().setSectionResizeMode(1, QHeaderView.Fixed)
        self.tree.header().setSectionResizeMode(2, QHeaderView.Interactive)
        self.tree.header().setSectionResizeMode(3, QHeaderView.Stretch)
        self.tree.setColumnWidth(0, 45)
        self.tree.setColumnWidth(1, 60)
        self.tree.setColumnWidth(2, 120)
        self.tree.setAlternatingRowColors(True)
        self.tree.setRootIsDecorated(False)
        self.tree.setSelectionMode(QTreeWidget.SingleSelection)
        self.tree.itemClicked.connect(self._on_item_clicked)
        self.tree.itemDoubleClicked.connect(self._on_item_double_clicked)

        # Header styling
        self.tree.setStyleSheet("""
            QTreeWidget::item { padding: 3px; }
            QHeaderView::section {
                background-color: #1F4E79;
                color: white;
                font-weight: bold;
                padding: 4px;
                border: 1px solid #2E6096;
            }
        """)
        layout.addWidget(self.tree)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_changes(self, changes: List[ChangeRecord]) -> None:
        """Load a list of ChangeRecord objects."""
        self._all_changes = list(changes)
        self._apply_filters()

    def clear(self) -> None:
        """Clear all changes."""
        self._all_changes = []
        self._filtered_changes = []
        self.tree.clear()
        self.lbl_count.setText("0건")

    def select_change(self, change_id: str) -> None:
        """Select a change by ID in the list."""
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            if item and item.data(0, Qt.UserRole + 1) == change_id:
                self.tree.setCurrentItem(item)
                self.tree.scrollToItem(item)
                break

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def _apply_filters(self) -> None:
        """Apply type filter and search filter."""
        filter_text = self.combo_filter.currentText()
        search_text = self.edit_search.text().strip().lower()

        filtered = []
        for change in self._all_changes:
            # Type filter
            if filter_text != "전체":
                if filter_text == "표변경":
                    if change.object_type.value not in ("표", "셀"):
                        continue
                elif filter_text == "제목변경":
                    if change.object_type.value != "제목":
                        continue
                elif change.change_type.value != filter_text:
                    continue

            # Search filter
            if search_text:
                searchable = " ".join([
                    change.location_info or "",
                    change.summary or "",
                    change.old_content or "",
                    change.new_content or "",
                    change.section_title or "",
                ]).lower()
                if search_text not in searchable:
                    continue

            filtered.append(change)

        self._filtered_changes = filtered
        self._populate_tree(filtered)

    def _populate_tree(self, changes: List[ChangeRecord]) -> None:
        """Populate tree widget with filtered changes."""
        self.tree.clear()

        for seq, change in enumerate(changes, 1):
            item = QTreeWidgetItem()
            item.setText(0, str(seq))
            item.setText(1, change.change_type.value)
            item.setText(2, _truncate(change.location_info, 20))
            item.setText(3, _truncate(change.summary, 50))

            # Store change reference
            item.setData(0, Qt.UserRole, change)
            item.setData(0, Qt.UserRole + 1, change.change_id)

            # Color coding
            color = _CHANGE_TYPE_COLORS.get(change.change_type, QColor("#FFFFFF"))
            for col in range(4):
                item.setBackground(col, QBrush(color))

            # Bold for high importance
            if change.importance == Importance.HIGH:
                font = QFont()
                font.setBold(True)
                for col in range(4):
                    item.setFont(col, font)

            # Tooltip with full content
            item.setToolTip(3, (
                f"[{change.change_id}] {change.change_type.value}\n"
                f"위치: {change.location_info}\n"
                f"기준: {change.old_content[:100] if change.old_content else '(없음)'}\n"
                f"비교: {change.new_content[:100] if change.new_content else '(없음)'}"
            ))

            self.tree.addTopLevelItem(item)

        total = len(self._all_changes)
        shown = len(changes)
        if shown == total:
            self.lbl_count.setText(f"{total}건")
        else:
            self.lbl_count.setText(f"{shown}/{total}건")

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def _on_item_clicked(self, item: QTreeWidgetItem, column: int) -> None:
        change = item.data(0, Qt.UserRole)
        if change:
            self.change_selected.emit(change)

    def _on_item_double_clicked(self, item: QTreeWidgetItem, column: int) -> None:
        self._on_item_clicked(item, column)


def _truncate(text: str, max_len: int) -> str:
    if not text:
        return ""
    text = text.replace("\n", " ").strip()
    if len(text) > max_len:
        return text[:max_len] + "…"
    return text


# ══════════════════════════════════════════════════════════════════════
# 모달리스 변경사항 목록 다이얼로그
# ══════════════════════════════════════════════════════════════════════

class ChangeListDialog(QDialog):
    """
    변경사항 목록을 별도 창으로 표시하는 모달리스 다이얼로그.
    - 비교 완료 후 버튼으로 열림
    - 창을 닫아도 메인 창은 그대로
    - 항목 클릭 시 메인 뷰어와 연동
    """

    change_selected = Signal(object)   # ChangeRecord → 메인 뷰어에 전달

    def __init__(self, parent=None):
        super().__init__(parent)
        # 모달리스: 열려 있어도 메인 창 조작 가능
        self.setWindowModality(Qt.NonModal)
        self.setWindowTitle("변경사항 목록")
        self.resize(820, 560)
        self.setMinimumSize(QSize(600, 400))

        # 창 닫기 버튼만 있으면 됨 (최소화/최대화 포함)
        self.setWindowFlags(
            Qt.Window |
            Qt.WindowMinimizeButtonHint |
            Qt.WindowMaximizeButtonHint |
            Qt.WindowCloseButtonHint
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        # 내부에 ChangeListWidget 그대로 사용
        self.list_widget = ChangeListWidget(self)
        self.list_widget.change_selected.connect(self.change_selected.emit)
        layout.addWidget(self.list_widget)

    # ── Public API ────────────────────────────────────────────────────

    def set_changes(self, changes: List[ChangeRecord]) -> None:
        self.list_widget.set_changes(changes)
        total = len(changes)
        self.setWindowTitle(f"변경사항 목록  ({total}건)")

    def clear(self) -> None:
        self.list_widget.clear()
        self.setWindowTitle("변경사항 목록")

    def select_change(self, change_id: str) -> None:
        """메인 뷰어에서 항목 선택 시 목록에서도 해당 항목 강조."""
        self.list_widget.select_change(change_id)

    # ── 닫기 버튼: 창을 숨기기만 (destroy 하지 않음) ──────────────────
    def closeEvent(self, event):
        self.hide()
        event.ignore()   # 실제 소멸은 막음 → 재사용 가능
