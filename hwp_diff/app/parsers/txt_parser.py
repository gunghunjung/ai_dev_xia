"""Parser for plain text files."""
import re
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

from app.models.document import DocumentStructure, DocumentBlock, TextSpan
from app.parsers.base_parser import BaseParser
from app.utils.logger import get_logger
from app.utils.text_utils import detect_numbering

logger = get_logger("parsers.txt")

# Heading detection patterns (ordered by priority)
_HEADING_PATTERNS: List[Tuple[re.Pattern, int]] = [
    # Underline-style headings (======= or -------)
    # Detected by looking at next line
    # Pattern: ALL CAPS line
    (re.compile(r"^[A-Z가-힣\s\d]{3,}$"), 1),
    # Numeric heading like "1.", "1.1.", "제1장", "제1절"
    (re.compile(r"^제\d+[장절조항]"), 1),
    (re.compile(r"^\d+\.\s+\S"), 2),
    (re.compile(r"^\d+\.\d+\.\s+\S"), 3),
    (re.compile(r"^\d+\.\d+\.\d+\.\s+\S"), 4),
]

_HEADING_KR_PREFIXES = re.compile(
    r"^(제\d+[장절조항]\s*|[일이삼사오육칠팔구십]+\s*[장절])\s*"
)


def _detect_heading_level(line: str, next_line: str = "") -> int:
    """
    Detect heading level of a text line.
    Returns 0 if not a heading, 1-4 if heading.
    """
    stripped = line.strip()
    if not stripped:
        return 0

    # Underline heading
    if next_line and re.match(r"^[=\-]{3,}\s*$", next_line.strip()):
        return 1 if next_line.strip().startswith("=") else 2

    # Korean chapter/section headings
    if _HEADING_KR_PREFIXES.match(stripped):
        m = _HEADING_KR_PREFIXES.match(stripped)
        prefix = m.group(0).strip()
        if "장" in prefix:
            return 1
        if "절" in prefix:
            return 2
        if "조" in prefix:
            return 3
        if "항" in prefix:
            return 4
        return 1

    # Numeric dotted headings - only treat as heading if there are sub-levels (e.g. 1.1, 1.1.1)
    # or if the text is short and looks like a section title (no sentence ending)
    m = re.match(r"^(\d+(?:\.\d+)+)\.\s+\S", stripped)
    if m:
        dots = m.group(1).count(".")
        return min(dots + 1, 4)
    # Single digit like "1. Title" - only heading if text is short (<=40 chars, no period at end)
    m = re.match(r"^(\d+)\.\s+(.+)", stripped)
    if m:
        body = m.group(2).strip()
        if len(body) <= 40 and not body.endswith(("다.", "요.", "임.", "함.")):
            return 2

    # Short ALL-CAPS lines (3-60 chars, no lowercase)
    if (3 <= len(stripped) <= 60
            and not re.search(r"[a-z]", stripped)
            and re.search(r"[A-Z가-힣]", stripped)
            and not stripped.endswith(".")
    ):
        return 1

    return 0


class TXTParser(BaseParser):
    """Parses plain text files line by line."""

    def parse(self, file_path: str) -> DocumentStructure:
        p = self._check_file(file_path)
        doc_id = self._new_doc_id(file_path)
        logger.info("Parsing TXT: %s", file_path)

        doc = DocumentStructure(
            doc_id=doc_id,
            file_path=str(p),
            file_type="txt",
            parse_confidence=0.8,
            page_confidence=0.2,
            table_confidence=0.5,
        )

        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.error("Cannot read file %s: %s", file_path, e)
            return doc

        lines = text.splitlines()
        para_idx = 0
        current_para_lines: List[str] = []
        i = 0

        def flush_paragraph(lines_buf: List[str], idx: int) -> Optional[DocumentBlock]:
            """Flush accumulated lines into a paragraph block."""
            content = " ".join(l.strip() for l in lines_buf if l.strip())
            if not content:
                return None
            numbering = detect_numbering(content)
            block_id = f"p_{idx}_{uuid.uuid4().hex[:6]}"
            page_hint = idx // 40 + 1
            return DocumentBlock(
                block_id=block_id,
                block_type="list_item" if numbering else "paragraph",
                content=content,
                spans=[TextSpan(text=content)],
                paragraph_index=idx,
                page_hint=page_hint,
                numbering=numbering,
            )

        while i < len(lines):
            line = lines[i]
            next_line = lines[i + 1] if i + 1 < len(lines) else ""

            # Check for table-like lines (pipe-delimited)
            if "|" in line and line.count("|") >= 2:
                # Flush current paragraph
                if current_para_lines:
                    blk = flush_paragraph(current_para_lines, para_idx)
                    if blk:
                        doc.blocks.append(blk)
                        para_idx += 1
                    current_para_lines = []

                # Collect table rows
                table_lines = []
                while i < len(lines) and "|" in lines[i] and lines[i].count("|") >= 2:
                    table_lines.append(lines[i])
                    i += 1
                tbl_block = self._parse_table_lines(table_lines, para_idx)
                if tbl_block:
                    doc.blocks.append(tbl_block)
                    para_idx += 1
                continue

            # Detect heading
            heading_level = _detect_heading_level(line.strip(), next_line.strip())

            # Skip underline lines
            if re.match(r"^[=\-]{3,}\s*$", line.strip()):
                i += 1
                continue

            if heading_level > 0 and line.strip():
                # Flush current paragraph
                if current_para_lines:
                    blk = flush_paragraph(current_para_lines, para_idx)
                    if blk:
                        doc.blocks.append(blk)
                        para_idx += 1
                    current_para_lines = []

                content = line.strip()
                block_id = f"h_{para_idx}_{uuid.uuid4().hex[:6]}"
                page_hint = para_idx // 40 + 1
                heading_block = DocumentBlock(
                    block_id=block_id,
                    block_type="heading",
                    content=content,
                    spans=[TextSpan(text=content)],
                    level=heading_level,
                    paragraph_index=para_idx,
                    page_hint=page_hint,
                    numbering=detect_numbering(content),
                )
                doc.blocks.append(heading_block)
                para_idx += 1
                i += 1
                continue

            # Empty line = paragraph break
            if not line.strip():
                if current_para_lines:
                    blk = flush_paragraph(current_para_lines, para_idx)
                    if blk:
                        doc.blocks.append(blk)
                        para_idx += 1
                    current_para_lines = []
                i += 1
                continue

            current_para_lines.append(line)
            i += 1

        # Flush remaining
        if current_para_lines:
            blk = flush_paragraph(current_para_lines, para_idx)
            if blk:
                doc.blocks.append(blk)
                para_idx += 1

        doc.total_pages = max(1, para_idx // 40 + 1)
        doc.title = self._extract_title(doc.blocks)
        logger.info("TXT parsed: %d blocks", len(doc.blocks))
        return doc

    def _parse_table_lines(
        self, lines: List[str], para_idx: int
    ) -> Optional[DocumentBlock]:
        """Parse pipe-delimited lines into a table block."""
        from app.models.document import TableData, TableCell

        rows_data: List[List[TableCell]] = []
        max_cols = 0

        for r_idx, line in enumerate(lines):
            # Skip separator lines (|---|---|)
            if re.match(r"^\s*\|[-:\s|]+\|\s*$", line):
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            row_cells = [
                TableCell(row=r_idx, col=c_idx, content=cell_text)
                for c_idx, cell_text in enumerate(cells)
            ]
            if row_cells:
                rows_data.append(row_cells)
                max_cols = max(max_cols, len(row_cells))

        if not rows_data:
            return None

        table_data = TableData(rows=len(rows_data), cols=max_cols, cells=rows_data)
        flat_text = "\n".join(
            " | ".join(cell.content for cell in row) for row in rows_data
        )
        block_id = f"tbl_{para_idx}_{uuid.uuid4().hex[:6]}"
        return DocumentBlock(
            block_id=block_id,
            block_type="table",
            content=flat_text,
            paragraph_index=para_idx,
            page_hint=para_idx // 40 + 1,
            table_data=table_data,
            table_id=f"tbl_{para_idx}",
        )

    def _extract_title(self, blocks: List[DocumentBlock]) -> str:
        for block in blocks:
            if block.block_type == "heading" and block.level == 1:
                return block.content
        for block in blocks:
            if block.content.strip():
                return block.content[:80]
        return ""
