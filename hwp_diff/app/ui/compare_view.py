"""Side-by-side document comparison view with synchronized scrolling."""
from typing import List, Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QLabel,
    QPushButton, QSplitter, QScrollBar,
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QTextCursor, QFont

from app.models.change_record import ChangeRecord, ChangeType
from app.models.document import DocumentStructure
from app.diff_engine.text_differ import highlight_diff_html


class CompareView(QWidget):
    """
    Side-by-side document comparison view.

    - Left panel: original document
    - Right panel: new document
    - Synchronized scrolling
    - HTML highlighting for changes
    - Navigation buttons for changes
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._changes: List[ChangeRecord] = []
        self._current_change_idx = -1
        self._syncing_scroll = False
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # --- Navigation bar ---
        nav_layout = QHBoxLayout()
        nav_layout.setContentsMargins(4, 2, 4, 2)

        self.lbl_nav = QLabel("변경사항: 0 / 0")
        self.lbl_nav.setStyleSheet("font-size: 11px; color: #444;")

        self.btn_prev = QPushButton("◀ 이전")
        self.btn_prev.setFixedWidth(80)
        self.btn_prev.setEnabled(False)
        self.btn_prev.clicked.connect(self._go_prev)

        self.btn_next = QPushButton("다음 ▶")
        self.btn_next.setFixedWidth(80)
        self.btn_next.setEnabled(False)
        self.btn_next.clicked.connect(self._go_next)

        nav_layout.addWidget(self.lbl_nav)
        nav_layout.addStretch()
        nav_layout.addWidget(self.btn_prev)
        nav_layout.addWidget(self.btn_next)
        layout.addLayout(nav_layout)

        # --- 헤더: 변경 전 / 변경 후 파일명만, 얇게 ---
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(4, 2, 4, 2)
        header_layout.setSpacing(4)

        _hdr_style = (
            "font-size: 10px; font-weight: bold; color: #333; "
            "background: #E8E8E8; border: 1px solid #CCCCCC; "
            "padding: 2px 8px;"
        )

        self.lbl_old_header = QLabel("변경 전")
        self.lbl_old_header.setStyleSheet(_hdr_style)
        self.lbl_old_header.setAlignment(Qt.AlignCenter)
        self.lbl_old_header.setFixedHeight(22)

        self.lbl_new_header = QLabel("변경 후")
        self.lbl_new_header.setStyleSheet(_hdr_style)
        self.lbl_new_header.setAlignment(Qt.AlignCenter)
        self.lbl_new_header.setFixedHeight(22)

        header_layout.addWidget(self.lbl_old_header)
        header_layout.addWidget(self.lbl_new_header)
        layout.addLayout(header_layout)

        # --- Splitter with two text views ---
        splitter = QSplitter(Qt.Horizontal)

        self.text_old = QTextEdit()
        self.text_old.setReadOnly(True)
        self.text_old.setFont(QFont("맑은 고딕", 10))
        self.text_old.setAcceptRichText(True)
        self.text_old.setStyleSheet(
            "QTextEdit { background: #FAFAFA; border: 1px solid #CCC; }"
        )

        self.text_new = QTextEdit()
        self.text_new.setReadOnly(True)
        self.text_new.setFont(QFont("맑은 고딕", 10))
        self.text_new.setAcceptRichText(True)
        self.text_new.setStyleSheet(
            "QTextEdit { background: #FAFAFA; border: 1px solid #CCC; }"
        )

        splitter.addWidget(self.text_old)
        splitter.addWidget(self.text_new)
        splitter.setSizes([1, 1])
        layout.addWidget(splitter)

        # Sync scrolling
        self.text_old.verticalScrollBar().valueChanged.connect(
            self._sync_scroll_from_old
        )
        self.text_new.verticalScrollBar().valueChanged.connect(
            self._sync_scroll_from_new
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_documents(
        self,
        old_doc: DocumentStructure,
        new_doc: DocumentStructure,
        changes: List[ChangeRecord],
        old_name: str = "기준문서",
        new_name: str = "비교문서",
    ) -> None:
        """
        Render both documents with change highlights.

        Args:
            old_doc: Original document structure
            new_doc: New document structure
            changes: List of ChangeRecord from comparison
            old_name: Display name for original document
            new_name: Display name for new document
        """
        self._changes = changes
        self._current_change_idx = -1

        self.lbl_old_header.setText(f"변경 전  │  {old_name}" if old_name else "변경 전")
        self.lbl_new_header.setText(f"변경 후  │  {new_name}" if new_name else "변경 후")

        old_html = self._build_document_html(old_doc, changes, side="old")
        new_html = self._build_document_html(new_doc, changes, side="new")

        self.text_old.setHtml(old_html)
        self.text_new.setHtml(new_html)

        n = len(changes)
        self.lbl_nav.setText(f"변경사항: 0 / {n}")
        self.btn_prev.setEnabled(n > 0)
        self.btn_next.setEnabled(n > 0)

    def highlight_change(self, change: ChangeRecord) -> None:
        """
        Scroll both views to show a specific change.

        Args:
            change: ChangeRecord to navigate to
        """
        # Find index in our change list
        for idx, c in enumerate(self._changes):
            if c.change_id == change.change_id:
                self._current_change_idx = idx
                break

        self._update_nav_label()

        # Scroll to the anchor for this change
        anchor = f"change_{change.change_id}"
        self.text_old.scrollToAnchor(anchor)
        self.text_new.scrollToAnchor(anchor)

    def clear(self) -> None:
        """Clear both views."""
        self.text_old.clear()
        self.text_new.clear()
        self._changes = []
        self._current_change_idx = -1
        self.lbl_nav.setText("변경사항: 0 / 0")
        self.btn_prev.setEnabled(False)
        self.btn_next.setEnabled(False)

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _go_prev(self) -> None:
        if not self._changes:
            return
        n = len(self._changes)
        if self._current_change_idx <= 0:
            self._current_change_idx = n - 1
        else:
            self._current_change_idx -= 1
        self._navigate_to_current()

    def _go_next(self) -> None:
        if not self._changes:
            return
        n = len(self._changes)
        if self._current_change_idx >= n - 1:
            self._current_change_idx = 0
        else:
            self._current_change_idx += 1
        self._navigate_to_current()

    def _navigate_to_current(self) -> None:
        if 0 <= self._current_change_idx < len(self._changes):
            change = self._changes[self._current_change_idx]
            anchor = f"change_{change.change_id}"
            self.text_old.scrollToAnchor(anchor)
            self.text_new.scrollToAnchor(anchor)
            self._update_nav_label()

    def _update_nav_label(self) -> None:
        n = len(self._changes)
        current = self._current_change_idx + 1 if self._current_change_idx >= 0 else 0
        self.lbl_nav.setText(f"변경사항: {current} / {n}")

    # ------------------------------------------------------------------
    # Scroll synchronization
    # ------------------------------------------------------------------

    def _sync_scroll_from_old(self, value: int) -> None:
        if self._syncing_scroll:
            return
        self._syncing_scroll = True
        old_sb = self.text_old.verticalScrollBar()
        new_sb = self.text_new.verticalScrollBar()
        if old_sb.maximum() > 0 and new_sb.maximum() > 0:
            ratio = value / old_sb.maximum()
            new_sb.setValue(int(ratio * new_sb.maximum()))
        self._syncing_scroll = False

    def _sync_scroll_from_new(self, value: int) -> None:
        if self._syncing_scroll:
            return
        self._syncing_scroll = True
        old_sb = self.text_old.verticalScrollBar()
        new_sb = self.text_new.verticalScrollBar()
        if old_sb.maximum() > 0 and new_sb.maximum() > 0:
            ratio = value / new_sb.maximum()
            old_sb.setValue(int(ratio * old_sb.maximum()))
        self._syncing_scroll = False

    # ------------------------------------------------------------------
    # HTML generation
    # ------------------------------------------------------------------

    def _build_document_html(
        self,
        doc: DocumentStructure,
        changes: List[ChangeRecord],
        side: str,
    ) -> str:
        """Build HTML representation of document with change highlights."""
        # Build a map: block_id -> change record
        block_to_change = {}
        for change in changes:
            if side == "old" and change.old_block_id:
                block_to_change[change.old_block_id] = change
            elif side == "new" and change.new_block_id:
                block_to_change[change.new_block_id] = change

        parts = [
            '<html><body style="font-family: 맑은 고딕, Arial; font-size: 10pt; '
            'margin: 10px; line-height: 1.6;">'
        ]

        for block in doc.blocks:
            change = block_to_change.get(block.block_id)

            block_style = self._get_block_style(block, change, side)
            anchor_id = f"change_{change.change_id}" if change else ""

            tag = "h3" if block.block_type == "heading" else "p"
            if block.block_type == "table":
                tag = "div"

            anchor_attr = f' id="{anchor_id}"' if anchor_id else ""

            # 변경 유형 뱃지
            badge = ""
            if change:
                ct = change.change_type
                if ct == ChangeType.DELETE and side == "old":
                    badge = '<span style="font-size:8pt;background:#CC0000;color:white;padding:1px 4px;border-radius:2px;margin-right:4px;">삭제</span>'
                elif ct == ChangeType.ADD and side == "new":
                    badge = '<span style="font-size:8pt;background:#00AA00;color:white;padding:1px 4px;border-radius:2px;margin-right:4px;">추가</span>'
                elif ct == ChangeType.MODIFY:
                    badge = '<span style="font-size:8pt;background:#BBAA00;color:white;padding:1px 4px;border-radius:2px;margin-right:4px;">수정</span>'
                elif ct == ChangeType.MOVE:
                    badge = '<span style="font-size:8pt;background:#7700CC;color:white;padding:1px 4px;border-radius:2px;margin-right:4px;">이동</span>'

            if change:
                content_html = self._render_content_with_diff(block, change, side)
            elif block.block_type == "table" and block.table_data:
                content_html = self._render_table_html(block)
            else:
                content_html = _escape_html(block.content)

            parts.append(
                f'<{tag}{anchor_attr} style="{block_style}">'
                f'{badge}{content_html}'
                f'</{tag}>'
            )

        # 블록 ID 없는 추가/삭제 항목 (하단에 표시)
        if side == "new":
            for change in changes:
                if change.change_type == ChangeType.ADD and not change.new_block_id:
                    parts.append(
                        f'<p id="change_{change.change_id}" '
                        f'style="background:#D6FFD6; border-left: 4px solid #00AA00; '
                        f'color:#005500; padding: 4px 8px; margin: 4px 0;">'
                        f'<span style="font-size:8pt;background:#00AA00;color:white;'
                        f'padding:1px 4px;border-radius:2px;margin-right:4px;">추가</span>'
                        f'{_escape_html(change.new_content)}</p>'
                    )
        elif side == "old":
            for change in changes:
                if change.change_type == ChangeType.DELETE and not change.old_block_id:
                    parts.append(
                        f'<p id="change_{change.change_id}" '
                        f'style="background:#FFD6D6; border-left: 4px solid #CC0000; '
                        f'color:#880000; padding: 4px 8px; margin: 4px 0; '
                        f'text-decoration: line-through;">'
                        f'<span style="font-size:8pt;background:#CC0000;color:white;'
                        f'padding:1px 4px;border-radius:2px;margin-right:4px;">삭제</span>'
                        f'{_escape_html(change.old_content)}</p>'
                    )

        parts.append("</body></html>")
        return "".join(parts)

    def _get_block_style(
        self, block, change: Optional[ChangeRecord], side: str = ""
    ) -> str:
        """Get CSS style string for a block."""
        base = "margin: 3px 0; padding: 3px 8px; border-radius: 2px;"

        if block.block_type == "heading":
            level = block.level or 1
            sizes = {1: "15pt", 2: "13pt", 3: "11pt", 4: "10pt"}
            size = sizes.get(level, "10pt")
            base += f" font-size: {size}; font-weight: bold;"

        if not change:
            return base

        ct = change.change_type
        if ct == ChangeType.DELETE:
            # 좌측(변경 전): 삭제는 빨강 배경 + 취소선
            return (base + " background: #FFD6D6; border-left: 4px solid #CC0000;"
                    " text-decoration: line-through; color: #880000;")
        elif ct == ChangeType.ADD:
            # 우측(변경 후): 추가는 초록 배경
            return (base + " background: #D6FFD6; border-left: 4px solid #00AA00;"
                    " color: #005500;")
        elif ct == ChangeType.MODIFY:
            return base + " background: #FFFACD; border-left: 4px solid #CCAA00;"
        elif ct == ChangeType.MOVE:
            return base + " background: #EEE0FF; border-left: 4px solid #7700CC;"
        elif ct == ChangeType.FORMAT:
            return base + " background: #E8F5E9; border-left: 4px solid #00AA44;"
        return base

    def _render_content_with_diff(
        self, block, change: ChangeRecord, side: str
    ) -> str:
        """Render block content with inline diff highlights."""
        if change.change_type in (ChangeType.DELETE, ChangeType.ADD, ChangeType.MOVE):
            content = block.content if block.content else ""
            return _escape_html(content)

        old_html, new_html = highlight_diff_html(
            change.old_content, change.new_content
        )

        if side == "old":
            return old_html if old_html else _escape_html(change.old_content)
        else:
            return new_html if new_html else _escape_html(change.new_content)

    def _render_table_html(self, block) -> str:
        """Render a table as HTML."""
        if not block.table_data:
            return _escape_html(block.content)

        parts = ['<table border="1" cellpadding="3" cellspacing="0" '
                 'style="border-collapse: collapse; width: 100%; font-size: 9pt;">']
        for r_idx, row in enumerate(block.table_data.cells):
            parts.append('<tr>')
            for cell in row:
                tag = "th" if r_idx == 0 else "td"
                bg = ' style="background: #E8EEF5; font-weight: bold;"' if r_idx == 0 else ""
                parts.append(f'<{tag}{bg}>{_escape_html(cell.content)}</{tag}>')
            parts.append('</tr>')
        parts.append('</table>')
        return "".join(parts)


def _escape_html(text: str) -> str:
    if not text:
        return ""
    return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br>"))
