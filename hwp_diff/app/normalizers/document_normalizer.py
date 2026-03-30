"""Document normalizer - ensures consistent DocumentStructure state."""
import re
import uuid
from typing import List, Optional

from app.models.document import DocumentStructure, DocumentBlock
from app.utils.logger import get_logger
from app.utils.text_utils import detect_numbering

logger = get_logger("normalizers")


class DocumentNormalizer:
    """
    Normalizes a DocumentStructure after parsing.

    Ensures:
    - All block_ids are unique
    - page_hint is estimated if not set
    - section_title is propagated to child blocks
    - numbering is extracted if not set
    - content is stripped of excess whitespace
    - paragraph_index is sequential
    """

    def normalize(
        self,
        doc: DocumentStructure,
        options: Optional[dict] = None,
    ) -> DocumentStructure:
        """
        Normalize the document structure in-place and return it.

        Args:
            doc: DocumentStructure to normalize
            options: Dict with keys:
                - ignore_whitespace (bool)
                - ignore_case (bool)
                - ignore_newline (bool)
        """
        if options is None:
            options = {}

        logger.debug("Normalizing document: %s (%d blocks)", doc.doc_id, len(doc.blocks))

        # 1. Ensure unique block IDs
        self._ensure_unique_ids(doc.blocks)

        # 2. Normalize content strings
        ignore_whitespace = options.get("ignore_whitespace", False)
        ignore_case = options.get("ignore_case", False)
        ignore_newline = options.get("ignore_newline", False)
        self._normalize_content(doc.blocks, ignore_whitespace, ignore_case, ignore_newline)

        # 3. Set sequential paragraph_index
        self._set_paragraph_indices(doc.blocks)

        # 4. Estimate page_hint
        self._estimate_page_hints(doc.blocks)

        # 5. Extract numbering if missing
        self._extract_numbering(doc.blocks)

        # 6. Propagate section titles
        self._propagate_section_titles(doc.blocks)

        # 7. Update document-level metadata
        if not doc.title:
            doc.title = self._find_title(doc.blocks)

        if doc.total_pages == 0 and doc.blocks:
            max_page = max((b.page_hint for b in doc.blocks), default=1)
            doc.total_pages = max(1, max_page)

        logger.debug("Normalization complete: %d blocks", len(doc.blocks))
        return doc

    # ------------------------------------------------------------------
    # Internal steps
    # ------------------------------------------------------------------

    def _ensure_unique_ids(self, blocks: List[DocumentBlock]) -> None:
        """Make sure all block_ids are unique."""
        seen = set()
        for block in blocks:
            if not block.block_id or block.block_id in seen:
                block.block_id = f"blk_{uuid.uuid4().hex[:10]}"
            seen.add(block.block_id)

    def _normalize_content(
        self,
        blocks: List[DocumentBlock],
        ignore_whitespace: bool,
        ignore_case: bool,
        ignore_newline: bool,
    ) -> None:
        """Normalize text content in each block."""
        for block in blocks:
            content = block.content or ""

            # Always strip leading/trailing whitespace
            content = content.strip()

            if ignore_newline:
                content = content.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")

            if ignore_whitespace:
                content = re.sub(r"\s+", " ", content).strip()

            if ignore_case:
                content = content.lower()

            block.content = content

            # Normalize table cell content too
            if block.table_data:
                for row in block.table_data.cells:
                    for cell in row:
                        cell_content = cell.content or ""
                        cell_content = cell_content.strip()
                        if ignore_whitespace:
                            cell_content = re.sub(r"\s+", " ", cell_content).strip()
                        if ignore_case:
                            cell_content = cell_content.lower()
                        cell.content = cell_content

    def _set_paragraph_indices(self, blocks: List[DocumentBlock]) -> None:
        """Set sequential paragraph_index values."""
        for idx, block in enumerate(blocks):
            block.paragraph_index = idx

    def _estimate_page_hints(self, blocks: List[DocumentBlock]) -> None:
        """
        블록 page_hint 추정.
        - 파서가 다양한 페이지 값을 제공하면 신뢰 (HWP COM, DOCX 등)
        - 모두 동일한 값(1)이거나 0이면 paragraph_index 기반 재추정
        규칙: 한 페이지당 평균 25~35줄, 여기서는 30 문단 = 1페이지 기준
        """
        if not blocks:
            return

        # 파서가 제공한 page_hint 분포 확인
        existing = [b.page_hint for b in blocks if b.page_hint > 0]
        all_same = len(set(existing)) <= 1  # 전부 같은 값(예: 모두 1)

        # 신뢰할 수 없는 경우: 값이 없거나 전부 1로 동일
        needs_reestimate = (not existing) or all_same

        if needs_reestimate:
            # paragraph_index 기반 재추정 (30 문단 ≈ 1페이지)
            for block in blocks:
                block.page_hint = block.paragraph_index // 30 + 1
        else:
            # 파서 값 신뢰 — 0인 것만 보정
            for block in blocks:
                if block.page_hint <= 0:
                    block.page_hint = block.paragraph_index // 30 + 1

    def _extract_numbering(self, blocks: List[DocumentBlock]) -> None:
        """Extract numbering prefix from content if not already set."""
        for block in blocks:
            if not block.numbering and block.content:
                block.numbering = detect_numbering(block.content)

    def _propagate_section_titles(self, blocks: List[DocumentBlock]) -> None:
        """
        Propagate section_title from heading blocks to subsequent non-heading blocks.
        Also builds a breadcrumb path for nested headings.
        """
        section_stack: List[str] = []  # stack of (level, title)
        level_titles: dict = {}  # level -> current title

        for block in blocks:
            if block.block_type == "heading":
                level = block.level or 1
                # Pop deeper levels
                keys_to_remove = [k for k in level_titles if k >= level]
                for k in keys_to_remove:
                    del level_titles[k]
                level_titles[level] = block.content
                # Build breadcrumb
                breadcrumb = " > ".join(
                    level_titles[k] for k in sorted(level_titles.keys())
                )
                block.section_title = breadcrumb
            else:
                # Propagate current section title
                if not block.section_title and level_titles:
                    deepest_level = max(level_titles.keys())
                    block.section_title = level_titles[deepest_level]

    def _find_title(self, blocks: List[DocumentBlock]) -> str:
        """Find document title from first heading."""
        for block in blocks:
            if block.block_type == "heading":
                return block.content
        for block in blocks:
            if block.content.strip():
                return block.content[:80]
        return ""
