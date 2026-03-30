"""Parser for HWPX format (zip archive containing XML files)."""
import zipfile
import uuid
import re
from pathlib import Path
from typing import List, Optional, Tuple
import xml.etree.ElementTree as ET

from app.models.document import (
    DocumentStructure, DocumentBlock, TextSpan, TableData, TableCell
)
from app.parsers.base_parser import BaseParser
from app.utils.logger import get_logger
from app.utils.text_utils import detect_numbering

logger = get_logger("parsers.hwpx")

# HWPX XML namespaces – versions vary so we use a flexible approach
_NS_MAP = {
    "hp": "http://www.hancom.co.kr/hwpml/2012/paragraph",
    "hc": "http://www.hancom.co.kr/hwpml/2012/core",
    "hs": "http://www.hancom.co.kr/hwpml/2012/section",
    "ht": "http://www.hancom.co.kr/hwpml/2012/table",
    "hh": "http://www.hancom.co.kr/hwpml/2012/head",
    "hf": "http://www.hancom.co.kr/hwpml/2012/frameset",
}

# Register namespaces for pretty print (not strictly needed for parsing)
for prefix, uri in _NS_MAP.items():
    ET.register_namespace(prefix, uri)


def _local(tag: str) -> str:
    """Return the local part of a Clark notation tag {ns}local."""
    if tag.startswith("{"):
        return tag.split("}", 1)[1]
    return tag


def _find_all_by_local(element: ET.Element, local_name: str) -> List[ET.Element]:
    """Find all descendant elements matching a local tag name (namespace-agnostic)."""
    results = []
    for child in element.iter():
        if _local(child.tag) == local_name:
            results.append(child)
    return results


def _find_children_by_local(element: ET.Element, local_name: str) -> List[ET.Element]:
    """Find direct children with a given local tag name."""
    return [c for c in element if _local(c.tag) == local_name]


class HWPXParser(BaseParser):
    """Parses HWPX files (zip + XML format)."""

    def parse(self, file_path: str) -> DocumentStructure:
        p = self._check_file(file_path)
        doc_id = self._new_doc_id(file_path)
        logger.info("Parsing HWPX: %s", file_path)

        doc = DocumentStructure(
            doc_id=doc_id,
            file_path=str(p),
            file_type="hwpx",
            parse_confidence=0.9,
            page_confidence=0.4,
            table_confidence=0.9,
        )

        try:
            with zipfile.ZipFile(str(p), "r") as zf:
                # Collect section XML files
                section_files = self._find_section_files(zf)
                if not section_files:
                    logger.warning("No section files found in HWPX: %s", file_path)
                    doc.parse_confidence = 0.3

                # Parse header for style info
                style_map = self._parse_styles(zf)

                para_idx = 0
                table_idx = 0
                for sec_name in sorted(section_files):
                    try:
                        with zf.open(sec_name) as sec_f:
                            tree = ET.parse(sec_f)
                            root = tree.getroot()
                    except Exception as e:
                        logger.warning("Failed to parse section %s: %s", sec_name, e)
                        continue

                    # Top-level children of section are paragraphs/tables
                    for elem in root.iter():
                        local = _local(elem.tag)
                        if local == "p":
                            block = self._parse_paragraph(elem, para_idx, style_map)
                            if block is not None:
                                doc.blocks.append(block)
                                para_idx += 1
                        elif local in ("tbl", "table"):
                            block = self._parse_table(elem, table_idx, para_idx)
                            if block is not None:
                                doc.blocks.append(block)
                                table_idx += 1
                                para_idx += 1

            # Deduplicate consecutive identical blocks (artifact of iter)
            doc.blocks = self._deduplicate_blocks(doc.blocks)
            doc.total_pages = max(1, para_idx // 40 + 1)
            doc.title = self._extract_title(doc.blocks)
            logger.info(
                "HWPX parsed: %d blocks, %d pages est.",
                len(doc.blocks), doc.total_pages
            )

        except zipfile.BadZipFile:
            logger.error("Not a valid zip/HWPX file: %s", file_path)
            doc.parse_confidence = 0.0
        except Exception as e:
            logger.exception("Unexpected error parsing HWPX %s: %s", file_path, e)
            doc.parse_confidence = 0.1

        return doc

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_section_files(self, zf: zipfile.ZipFile) -> List[str]:
        """Find all section XML files inside the HWPX archive."""
        candidates = []
        for name in zf.namelist():
            lower = name.lower()
            # Common patterns: Contents/section0.xml, Contents/Section1.xml
            if re.search(r"contents/section\d*\.xml$", lower):
                candidates.append(name)
            elif re.search(r"section\d*\.xml$", lower) and "header" not in lower:
                candidates.append(name)
        return candidates

    def _parse_styles(self, zf: zipfile.ZipFile) -> dict:
        """Parse style definitions from header.xml or styles.xml."""
        style_map = {}  # style_id -> {"name": ..., "level": ...}
        header_candidates = [
            n for n in zf.namelist()
            if "header" in n.lower() and n.endswith(".xml")
        ]
        for hname in header_candidates:
            try:
                with zf.open(hname) as hf:
                    tree = ET.parse(hf)
                    root = tree.getroot()
                for style_elem in _find_all_by_local(root, "style"):
                    sid = style_elem.get("id") or style_elem.get("styleId") or ""
                    sname = style_elem.get("name") or style_elem.get("localName") or ""
                    level = self._heading_level_from_name(sname)
                    style_map[sid] = {"name": sname, "level": level}
            except Exception as e:
                logger.debug("Could not parse header %s: %s", hname, e)
        return style_map

    def _heading_level_from_name(self, style_name: str) -> int:
        """Determine heading level (1-6) from a style name, 0 if not heading."""
        sn = style_name.lower()
        # Explicit numeric level
        m = re.search(r"(?:제목|heading|head)\s*(\d)", sn)
        if m:
            return int(m.group(1))
        if "제목" in sn or "heading" in sn:
            return 1
        if "소제목" in sn:
            return 2
        return 0

    def _parse_paragraph(
        self,
        p_elem: ET.Element,
        para_idx: int,
        style_map: dict,
    ) -> Optional[DocumentBlock]:
        """Parse a <hp:p> element into a DocumentBlock."""
        # Extract style reference
        style_id = ""
        para_pr = None
        for child in p_elem:
            cl = _local(child.tag)
            if cl in ("pPr", "parapr", "paraPr"):
                para_pr = child
                break

        if para_pr is not None:
            style_id = para_pr.get("styleIDRef") or para_pr.get("styleId") or ""

        style_info = style_map.get(style_id, {})
        style_name = style_info.get("name", "")
        heading_level = style_info.get("level", 0)

        # If no style info, try detecting heading from attributes
        if heading_level == 0:
            heading_level = self._heading_level_from_name(style_name)

        # Extract all run texts and spans
        spans: List[TextSpan] = []
        full_text_parts = []

        for run in _find_all_by_local(p_elem, "run"):
            run_text = self._extract_run_text(run)
            if run_text:
                span = self._build_text_span(run, run_text)
                spans.append(span)
                full_text_parts.append(run_text)

        # Also get raw <t> text not inside explicit run elements
        for t_elem in _find_all_by_local(p_elem, "t"):
            parent = None
            # Check if this t is directly under p (not inside a run we already handled)
            t_text = (t_elem.text or "").strip()
            if t_text and t_text not in "".join(full_text_parts):
                full_text_parts.append(t_elem.text or "")

        content = "".join(full_text_parts).strip()

        if not content:
            return None

        # Determine block type
        if heading_level > 0:
            block_type = "heading"
        else:
            # Check numbering pattern
            numbering = detect_numbering(content)
            if numbering:
                block_type = "list_item"
            else:
                block_type = "paragraph"

        numbering = detect_numbering(content)
        block_id = f"p_{para_idx}_{uuid.uuid4().hex[:6]}"
        page_hint = para_idx // 40 + 1

        return DocumentBlock(
            block_id=block_id,
            block_type=block_type,
            content=content,
            spans=spans,
            level=heading_level,
            paragraph_index=para_idx,
            page_hint=page_hint,
            numbering=numbering,
            style_name=style_name,
        )

    def _extract_run_text(self, run_elem: ET.Element) -> str:
        """Extract concatenated text from a run element."""
        parts = []
        for t in _find_all_by_local(run_elem, "t"):
            parts.append(t.text or "")
        return "".join(parts)

    def _build_text_span(self, run_elem: ET.Element, text: str) -> TextSpan:
        """Build a TextSpan from a run element's character properties."""
        span = TextSpan(text=text)
        # Find character properties
        for child in run_elem:
            cl = _local(child.tag)
            if cl in ("charPr", "charpr", "rPr"):
                span.bold = child.get("bold", "false").lower() in ("true", "1")
                span.italic = child.get("italic", "false").lower() in ("true", "1")
                span.underline = child.get("underline", "false").lower() not in ("false", "0", "none", "")
                try:
                    span.font_size = float(child.get("size", "10"))
                except (ValueError, TypeError):
                    span.font_size = 10.0
                span.font_name = child.get("fontType") or child.get("font", "")
                break
        return span

    def _parse_table(
        self,
        tbl_elem: ET.Element,
        table_idx: int,
        para_idx: int,
    ) -> Optional[DocumentBlock]:
        """Parse a table element into a DocumentBlock with TableData."""
        table_id = f"tbl_{table_idx}"
        rows_data: List[List[TableCell]] = []
        row_elems = _find_all_by_local(tbl_elem, "tr")

        max_cols = 0
        for r_idx, row_elem in enumerate(row_elems):
            row_cells = []
            cell_elems = _find_all_by_local(row_elem, "td")
            if not cell_elems:
                cell_elems = _find_all_by_local(row_elem, "cell")
            for c_idx, cell_elem in enumerate(cell_elems):
                # Extract cell text
                cell_text_parts = []
                for t in _find_all_by_local(cell_elem, "t"):
                    cell_text_parts.append(t.text or "")
                cell_text = " ".join(cell_text_parts).strip()

                row_span = int(cell_elem.get("rowSpan", cell_elem.get("rowspan", 1)))
                col_span = int(cell_elem.get("colSpan", cell_elem.get("colspan", 1)))

                cell = TableCell(
                    row=r_idx,
                    col=c_idx,
                    row_span=row_span,
                    col_span=col_span,
                    content=cell_text,
                )
                row_cells.append(cell)
            if row_cells:
                rows_data.append(row_cells)
                max_cols = max(max_cols, len(row_cells))

        if not rows_data:
            return None

        table_data = TableData(
            rows=len(rows_data),
            cols=max_cols,
            cells=rows_data,
        )

        # Build a flat text representation
        flat_text = self._table_to_flat_text(table_data)
        block_id = f"tbl_{table_idx}_{uuid.uuid4().hex[:6]}"
        page_hint = para_idx // 40 + 1

        return DocumentBlock(
            block_id=block_id,
            block_type="table",
            content=flat_text,
            paragraph_index=para_idx,
            page_hint=page_hint,
            table_data=table_data,
            table_id=table_id,
        )

    def _table_to_flat_text(self, table_data: TableData) -> str:
        """Convert table to pipe-delimited text representation."""
        lines = []
        for row in table_data.cells:
            line = " | ".join(cell.content for cell in row)
            lines.append(line)
        return "\n".join(lines)

    def _deduplicate_blocks(self, blocks: List[DocumentBlock]) -> List[DocumentBlock]:
        """
        Remove duplicate consecutive blocks that arise from double-iteration
        in nested element traversal.
        """
        if not blocks:
            return blocks
        result = []
        seen_ids = set()
        for block in blocks:
            if block.block_id not in seen_ids:
                seen_ids.add(block.block_id)
                result.append(block)
        return result

    def _extract_title(self, blocks: List[DocumentBlock]) -> str:
        """Extract document title from first heading block."""
        for block in blocks:
            if block.block_type == "heading" and block.level == 1:
                return block.content
        # Fallback: first non-empty block
        for block in blocks:
            if block.content.strip():
                return block.content[:80]
        return ""
