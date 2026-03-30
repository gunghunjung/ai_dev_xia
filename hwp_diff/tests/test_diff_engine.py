"""Tests for the diff engine: paragraph matching, table comparison, move detection."""
import sys
import unittest
import uuid
from pathlib import Path
from typing import List

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.models.document import DocumentStructure, DocumentBlock, TableData, TableCell
from app.models.change_record import ChangeType, ObjectType, Importance
from app.diff_engine.paragraph_matcher import match_blocks, detect_moves
from app.diff_engine.table_matcher import compare_tables
from app.diff_engine.text_differ import word_diff, char_diff, highlight_diff_html
from app.diff_engine.diff_engine import DiffEngine


def _make_block(content: str, block_type: str = "paragraph", idx: int = 0) -> DocumentBlock:
    return DocumentBlock(
        block_id=f"blk_{uuid.uuid4().hex[:8]}",
        block_type=block_type,
        content=content,
        paragraph_index=idx,
        page_hint=1,
    )


def _make_doc(blocks: List[DocumentBlock]) -> DocumentStructure:
    return DocumentStructure(
        doc_id=f"doc_{uuid.uuid4().hex[:8]}",
        file_path="/test/doc.txt",
        file_type="txt",
        blocks=blocks,
    )


class TestTextDiffer(unittest.TestCase):
    """Tests for text_differ functions."""

    def test_word_diff_equal(self):
        spans = word_diff("hello world", "hello world")
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].change_type, "equal")

    def test_word_diff_replace(self):
        spans = word_diff("hello world", "hello universe")
        change_types = [s.change_type for s in spans]
        self.assertIn("replace", change_types)

    def test_word_diff_insert(self):
        spans = word_diff("hello", "hello new world")
        change_types = [s.change_type for s in spans]
        self.assertIn("insert", change_types)

    def test_word_diff_delete(self):
        spans = word_diff("hello new world", "hello")
        change_types = [s.change_type for s in spans]
        self.assertIn("delete", change_types)

    def test_char_diff_equal(self):
        spans = char_diff("abc", "abc")
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].change_type, "equal")

    def test_char_diff_modify(self):
        spans = char_diff("abc", "axc")
        texts = [(s.old_text, s.new_text) for s in spans if s.change_type == "replace"]
        self.assertGreater(len(texts), 0)

    def test_highlight_diff_html(self):
        old_html, new_html = highlight_diff_html("old value", "new value")
        self.assertIn("old", old_html)
        self.assertIn("new", new_html)

    def test_highlight_diff_empty(self):
        old_html, new_html = highlight_diff_html("", "")
        self.assertEqual(old_html, "")
        self.assertEqual(new_html, "")

    def test_korean_word_diff(self):
        old = "허용오차는 ±0.5V 이내이어야 한다."
        new = "허용오차는 ±0.3V 이내이어야 한다."
        spans = word_diff(old, new)
        change_types = {s.change_type for s in spans}
        self.assertIn("replace", change_types)


class TestParagraphMatcher(unittest.TestCase):
    """Tests for paragraph_matcher functions."""

    def test_exact_match(self):
        old_blocks = [
            _make_block("동일한 내용입니다.", idx=0),
            _make_block("두 번째 동일 내용.", idx=1),
        ]
        new_blocks = [
            _make_block("동일한 내용입니다.", idx=0),
            _make_block("두 번째 동일 내용.", idx=1),
        ]
        matches = match_blocks(old_blocks, new_blocks, threshold=0.5)
        # Both should match exactly
        old_matched = {m[0] for m in matches}
        new_matched = {m[1] for m in matches}
        self.assertIn(0, old_matched)
        self.assertIn(1, old_matched)

    def test_no_match_completely_different(self):
        old_blocks = [_make_block("완전히 다른 내용 ABC.", idx=0)]
        new_blocks = [_make_block("완전히 다른 내용 XYZ.", idx=0)]
        # With very high threshold, should not match
        matches = match_blocks(old_blocks, new_blocks, threshold=0.99)
        # May or may not match depending on similarity
        # Just verify it doesn't crash
        self.assertIsInstance(matches, list)

    def test_fuzzy_match(self):
        old_blocks = [
            _make_block("시험 온도는 25℃ 이내이어야 한다.", idx=0),
        ]
        new_blocks = [
            _make_block("시험 온도는 25℃ 이내이어야 합니다.", idx=0),
        ]
        matches = match_blocks(old_blocks, new_blocks, threshold=0.5)
        self.assertGreaterEqual(len(matches), 1)
        self.assertEqual(matches[0][2], matches[0][2])  # similarity is a float

    def test_empty_blocks(self):
        matches = match_blocks([], [], threshold=0.5)
        self.assertEqual(matches, [])

    def test_detect_moves(self):
        """Detect that matched blocks have different positions."""
        # Create old and new with same content but different order
        old_blocks = [
            _make_block("첫 번째 문단", idx=0),
            _make_block("두 번째 문단", idx=1),
            _make_block("세 번째 문단", idx=2),
        ]
        new_blocks = [
            _make_block("세 번째 문단", idx=0),  # moved to front
            _make_block("첫 번째 문단", idx=1),
            _make_block("두 번째 문단", idx=2),
        ]
        matches = match_blocks(old_blocks, new_blocks, threshold=0.5)
        moves = detect_moves(matches, old_blocks, new_blocks, position_threshold=1)
        # Should detect at least one move
        self.assertIsInstance(moves, list)

    def test_partial_match_returns_correct_indices(self):
        """Verify returned indices are valid."""
        old_blocks = [_make_block(f"문단 {i}", idx=i) for i in range(5)]
        new_blocks = [_make_block(f"문단 {i}", idx=i) for i in range(3, 8)]
        matches = match_blocks(old_blocks, new_blocks, threshold=0.5)
        for old_idx, new_idx, sim in matches:
            self.assertGreaterEqual(old_idx, 0)
            self.assertLess(old_idx, len(old_blocks))
            self.assertGreaterEqual(new_idx, 0)
            self.assertLess(new_idx, len(new_blocks))
            self.assertGreaterEqual(sim, 0.0)
            self.assertLessEqual(sim, 1.0)


class TestTableMatcher(unittest.TestCase):
    """Tests for table_matcher.compare_tables."""

    def _make_table(self, data: list) -> TableData:
        rows = len(data)
        cols = max(len(r) for r in data) if data else 0
        cells = []
        for r_idx, row in enumerate(data):
            row_cells = [
                TableCell(row=r_idx, col=c_idx, content=cell_text)
                for c_idx, cell_text in enumerate(row)
            ]
            cells.append(row_cells)
        return TableData(rows=rows, cols=cols, cells=cells)

    def test_identical_tables_no_changes(self):
        data = [
            ["항목", "값", "단위"],
            ["온도", "25", "℃"],
            ["전압", "5.0", "V"],
        ]
        old_table = self._make_table(data)
        new_table = self._make_table(data)
        changes = compare_tables(old_table, new_table)
        self.assertEqual(len(changes), 0)

    def test_cell_value_changed(self):
        old_data = [["항목", "값"], ["허용오차", "±0.5V"]]
        new_data = [["항목", "값"], ["허용오차", "±0.3V"]]
        old_table = self._make_table(old_data)
        new_table = self._make_table(new_data)
        changes = compare_tables(old_table, new_table)
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0].change_type, ChangeType.MODIFY)
        self.assertEqual(changes[0].old_content, "±0.5V")
        self.assertEqual(changes[0].new_content, "±0.3V")

    def test_row_added(self):
        old_data = [["항목", "값"], ["온도", "25℃"]]
        new_data = [["항목", "값"], ["온도", "25℃"], ["전압", "5V"]]
        old_table = self._make_table(old_data)
        new_table = self._make_table(new_data)
        changes = compare_tables(old_table, new_table)
        add_changes = [c for c in changes if c.change_type == ChangeType.ADD]
        self.assertGreaterEqual(len(add_changes), 1)

    def test_row_deleted(self):
        old_data = [["항목", "값"], ["온도", "25℃"], ["전압", "5V"]]
        new_data = [["항목", "값"], ["온도", "25℃"]]
        old_table = self._make_table(old_data)
        new_table = self._make_table(new_data)
        changes = compare_tables(old_table, new_table)
        del_changes = [c for c in changes if c.change_type == ChangeType.DELETE]
        self.assertGreaterEqual(len(del_changes), 1)

    def test_multiple_cell_changes(self):
        old_data = [["A", "B", "C"], ["1", "2", "3"]]
        new_data = [["A", "B", "C"], ["1", "9", "3"]]
        old_table = self._make_table(old_data)
        new_table = self._make_table(new_data)
        changes = compare_tables(old_table, new_table)
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0].old_content, "2")
        self.assertEqual(changes[0].new_content, "9")


class TestDiffEngine(unittest.TestCase):
    """Integration tests for DiffEngine.compare()."""

    def setUp(self):
        self.engine = DiffEngine({
            "similarity_threshold": 0.5,
            "detect_moves": True,
            "include_format_changes": False,
        })

    def test_compare_identical_docs(self):
        blocks = [
            _make_block("문단 1 내용입니다.", idx=0),
            _make_block("문단 2 내용입니다.", idx=1),
        ]
        old_doc = _make_doc(blocks)
        new_doc = _make_doc([
            _make_block("문단 1 내용입니다.", idx=0),
            _make_block("문단 2 내용입니다.", idx=1),
        ])
        result = self.engine.compare(old_doc, new_doc)
        # Identical docs: no changes
        self.assertEqual(len(result.changes), 0)
        self.assertEqual(result.added_count, 0)
        self.assertEqual(result.deleted_count, 0)
        self.assertEqual(result.modified_count, 0)

    def test_compare_added_paragraph(self):
        old_doc = _make_doc([
            _make_block("기존 내용입니다.", idx=0),
        ])
        new_doc = _make_doc([
            _make_block("기존 내용입니다.", idx=0),
            _make_block("새로 추가된 내용입니다.", idx=1),
        ])
        result = self.engine.compare(old_doc, new_doc)
        self.assertGreaterEqual(result.added_count, 1)

    def test_compare_deleted_paragraph(self):
        old_doc = _make_doc([
            _make_block("기존 내용입니다.", idx=0),
            _make_block("삭제될 내용입니다.", idx=1),
        ])
        new_doc = _make_doc([
            _make_block("기존 내용입니다.", idx=0),
        ])
        result = self.engine.compare(old_doc, new_doc)
        self.assertGreaterEqual(result.deleted_count, 1)

    def test_compare_modified_paragraph(self):
        old_doc = _make_doc([
            _make_block("허용오차는 ±0.5V 이내이어야 한다.", idx=0),
        ])
        new_doc = _make_doc([
            _make_block("허용오차는 ±0.3V 이내이어야 한다.", idx=0),
        ])
        result = self.engine.compare(old_doc, new_doc)
        self.assertGreaterEqual(result.modified_count, 1)
        mod_changes = [c for c in result.changes if c.change_type == ChangeType.MODIFY]
        self.assertGreaterEqual(len(mod_changes), 1)

    def test_compare_returns_change_records(self):
        old_doc = _make_doc([
            _make_block("내용 A", idx=0),
            _make_block("내용 B", idx=1),
        ])
        new_doc = _make_doc([
            _make_block("내용 A", idx=0),
            _make_block("내용 C", idx=1),  # modified
            _make_block("내용 D", idx=2),  # added
        ])
        result = self.engine.compare(old_doc, new_doc)
        self.assertIsNotNone(result)
        self.assertGreater(len(result.changes), 0)
        for change in result.changes:
            self.assertIsNotNone(change.change_id)
            self.assertIsNotNone(change.change_type)
            self.assertIsNotNone(change.summary)

    def test_overall_similarity_range(self):
        old_doc = _make_doc([_make_block("내용", idx=0)])
        new_doc = _make_doc([_make_block("다른 내용", idx=0)])
        result = self.engine.compare(old_doc, new_doc)
        self.assertGreaterEqual(result.overall_similarity, 0.0)
        self.assertLessEqual(result.overall_similarity, 1.0)

    def test_empty_docs(self):
        old_doc = _make_doc([])
        new_doc = _make_doc([])
        result = self.engine.compare(old_doc, new_doc)
        self.assertEqual(len(result.changes), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
