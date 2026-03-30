"""Tests for document parsers."""
import os
import sys
import tempfile
import zipfile
import unittest
from pathlib import Path

# Ensure project root is in path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.parsers.txt_parser import TXTParser
from app.parsers.hwpx_parser import HWPXParser
from app.models.document import DocumentStructure, DocumentBlock


class TestTXTParser(unittest.TestCase):
    """Tests for TXTParser."""

    def setUp(self):
        self.parser = TXTParser()

    def _write_temp(self, content: str, suffix: str = ".txt") -> str:
        fd, path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    def tearDown(self):
        pass  # Temp files are cleaned in individual tests

    def test_parse_simple_text(self):
        """Basic paragraph parsing."""
        content = "첫 번째 문단입니다.\n\n두 번째 문단입니다."
        path = self._write_temp(content)
        try:
            doc = self.parser.parse(path)
            self.assertIsInstance(doc, DocumentStructure)
            self.assertEqual(doc.file_type, "txt")
            texts = [b.content for b in doc.blocks]
            self.assertIn("첫 번째 문단입니다.", texts)
            self.assertIn("두 번째 문단입니다.", texts)
        finally:
            os.unlink(path)

    def test_parse_heading_numeric(self):
        """Detect numeric headings like '1. 개요'."""
        content = "1. 개요\n\n본문 내용입니다.\n\n2. 적용범위\n\n두 번째 절입니다."
        path = self._write_temp(content)
        try:
            doc = self.parser.parse(path)
            headings = doc.get_heading_blocks()
            heading_texts = [b.content for b in headings]
            # At least some headings should be detected
            self.assertTrue(len(heading_texts) >= 1, f"Expected headings, got: {heading_texts}")
        finally:
            os.unlink(path)

    def test_parse_korean_chapter_heading(self):
        """Detect Korean chapter headings like '제1장'."""
        content = "제1장 개요\n\n본문 내용.\n\n제2장 적용범위\n\n내용."
        path = self._write_temp(content)
        try:
            doc = self.parser.parse(path)
            headings = doc.get_heading_blocks()
            self.assertGreaterEqual(len(headings), 1)
            heading_contents = [h.content for h in headings]
            self.assertTrue(
                any("제1장" in c or "제2장" in c for c in heading_contents),
                f"Korean chapter headings not detected: {heading_contents}"
            )
        finally:
            os.unlink(path)

    def test_parse_table_pipe_delimited(self):
        """Detect pipe-delimited table."""
        content = (
            "표 제목\n\n"
            "| 항목 | 기준값 | 허용오차 |\n"
            "|------|--------|----------|\n"
            "| 온도 | 25℃   | ±2℃    |\n"
            "| 전압 | 5V     | ±0.5V  |\n\n"
            "이후 내용."
        )
        path = self._write_temp(content)
        try:
            doc = self.parser.parse(path)
            table_blocks = doc.get_table_blocks()
            self.assertGreaterEqual(len(table_blocks), 1, "No table blocks detected")
            tbl = table_blocks[0]
            self.assertIsNotNone(tbl.table_data)
            self.assertGreaterEqual(tbl.table_data.rows, 2)
        finally:
            os.unlink(path)

    def test_parse_empty_file(self):
        """Handles empty file gracefully."""
        path = self._write_temp("")
        try:
            doc = self.parser.parse(path)
            self.assertIsInstance(doc, DocumentStructure)
            self.assertEqual(len(doc.blocks), 0)
        finally:
            os.unlink(path)

    def test_parse_file_not_found(self):
        """Raises FileNotFoundError for missing file."""
        with self.assertRaises(FileNotFoundError):
            self.parser.parse("/nonexistent/path/file.txt")

    def test_parse_numbering_detection(self):
        """Detect list items or headings with numbering prefix."""
        # Lines like "1. 항목" may be classified as headings or list_items depending on context
        # Both are valid - the key thing is that numbering is extracted
        content = "(1) 첫 번째 항목\n(2) 두 번째 항목\n(3) 세 번째 항목"
        path = self._write_temp(content)
        try:
            doc = self.parser.parse(path)
            numbered_blocks = [b for b in doc.blocks if b.numbering]
            self.assertGreaterEqual(len(numbered_blocks), 1,
                                    f"Expected blocks with numbering, got: {[b.block_type for b in doc.blocks]}")
        finally:
            os.unlink(path)

    def test_parse_korean_content(self):
        """Handles Korean UTF-8 content correctly."""
        content = (
            "시험절차서\n\n"
            "1. 개요\n\n"
            "본 절차서는 전기/전자 장치의 성능 시험을 위한 표준 절차를 기술합니다.\n\n"
            "2. 적용범위\n\n"
            "본 절차서는 모든 전기 장치에 적용됩니다."
        )
        path = self._write_temp(content)
        try:
            doc = self.parser.parse(path)
            self.assertGreater(len(doc.blocks), 0)
            all_content = " ".join(b.content for b in doc.blocks)
            self.assertIn("시험절차서", all_content)
            self.assertIn("적용범위", all_content)
        finally:
            os.unlink(path)


class TestHWPXParser(unittest.TestCase):
    """Tests for HWPXParser using synthetic HWPX files."""

    def setUp(self):
        self.parser = HWPXParser()

    def _create_minimal_hwpx(self, paragraphs: list, tables: list = None) -> str:
        """Create a minimal HWPX zip file for testing."""
        section_xml = self._build_section_xml(paragraphs, tables or [])

        fd, path = tempfile.mkstemp(suffix=".hwpx")
        os.close(fd)

        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("Contents/section0.xml", section_xml)
            zf.writestr("mimetype", "application/hwp+zip")

        return path

    def _build_section_xml(self, paragraphs: list, tables: list) -> str:
        """Build a minimal section XML."""
        NS = "http://www.hancom.co.kr/hwpml/2012/paragraph"
        NS_T = "http://www.hancom.co.kr/hwpml/2012/table"

        parts = [
            f'<?xml version="1.0" encoding="UTF-8"?>',
            f'<hp:sec xmlns:hp="{NS}" xmlns:ht="{NS_T}">',
        ]

        for para_text in paragraphs:
            parts.append(
                f'<hp:p>'
                f'<hp:run><hp:t>{_xml_escape(para_text)}</hp:t></hp:run>'
                f'</hp:p>'
            )

        for table_data in tables:
            # table_data: list of list of str (rows x cols)
            parts.append('<ht:tbl>')
            for row in table_data:
                parts.append('<ht:tr>')
                for cell_text in row:
                    parts.append(
                        f'<ht:td><hp:p><hp:run>'
                        f'<hp:t>{_xml_escape(cell_text)}</hp:t>'
                        f'</hp:run></hp:p></ht:td>'
                    )
                parts.append('</ht:tr>')
            parts.append('</ht:tbl>')

        parts.append('</hp:sec>')
        return "\n".join(parts)

    def test_parse_basic_paragraphs(self):
        """Parse basic paragraphs from HWPX."""
        path = self._create_minimal_hwpx([
            "첫 번째 문단입니다.",
            "두 번째 문단입니다.",
            "세 번째 문단입니다.",
        ])
        try:
            doc = self.parser.parse(path)
            self.assertEqual(doc.file_type, "hwpx")
            texts = [b.content for b in doc.blocks]
            self.assertIn("첫 번째 문단입니다.", texts)
            self.assertIn("두 번째 문단입니다.", texts)
        finally:
            os.unlink(path)

    def test_parse_with_table(self):
        """Parse HWPX with embedded table."""
        table_data = [
            ["항목", "기준값", "허용오차"],
            ["온도", "25", "±2"],
            ["전압", "5", "±0.5"],
        ]
        path = self._create_minimal_hwpx(
            ["표 테스트"],
            tables=[table_data]
        )
        try:
            doc = self.parser.parse(path)
            table_blocks = doc.get_table_blocks()
            self.assertGreaterEqual(len(table_blocks), 1)
            tbl = table_blocks[0]
            self.assertIsNotNone(tbl.table_data)
            self.assertEqual(tbl.table_data.rows, 3)
            self.assertEqual(tbl.table_data.cols, 3)
        finally:
            os.unlink(path)

    def test_parse_invalid_zip(self):
        """Handles invalid zip/hwpx gracefully."""
        fd, path = tempfile.mkstemp(suffix=".hwpx")
        os.close(fd)
        with open(path, "wb") as f:
            f.write(b"not a zip file content here")
        try:
            doc = self.parser.parse(path)
            self.assertIsInstance(doc, DocumentStructure)
            self.assertLess(doc.parse_confidence, 0.5)
        finally:
            os.unlink(path)

    def test_parse_empty_hwpx(self):
        """Handles HWPX with no sections."""
        fd, path = tempfile.mkstemp(suffix=".hwpx")
        os.close(fd)
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("mimetype", "application/hwp+zip")
        try:
            doc = self.parser.parse(path)
            self.assertIsInstance(doc, DocumentStructure)
        finally:
            os.unlink(path)


def _xml_escape(text: str) -> str:
    return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
