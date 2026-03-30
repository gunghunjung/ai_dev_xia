"""
Main diff engine orchestrator.
Coordinates paragraph matching, table comparison, and change record generation.
"""
import uuid
from typing import List, Set, Tuple, Callable, Optional

from app.models.document import DocumentStructure, DocumentBlock
from app.models.change_record import (
    ChangeRecord, ChangeType, ObjectType, Importance, CompareResult, DiffSpan
)
from app.diff_engine.paragraph_matcher import match_blocks, detect_moves
from app.diff_engine.table_matcher import compare_tables
from app.diff_engine.text_differ import word_diff, char_diff
from app.utils.logger import get_logger
from app.utils.text_utils import (
    compute_similarity, estimate_importance, generate_summary
)

logger = get_logger("diff_engine")

_DEFAULT_OPTIONS = {
    "ignore_whitespace": False,
    "ignore_case": False,
    "ignore_newline": False,
    "include_format_changes": True,
    "detect_moves": True,
    "similarity_threshold": 0.6,
    "table_sensitivity": 0.5,
    "important_keywords": [],
}


class DiffEngine:
    """
    Main orchestrator for document comparison.

    Usage:
        engine = DiffEngine(options)
        result = engine.compare(old_doc, new_doc)
    """

    def __init__(self, options: Optional[dict] = None):
        self.options = {**_DEFAULT_OPTIONS, **(options or {})}
        self._change_counter = 0
        self._progress_callback: Optional[Callable[[int, str], None]] = None

    def set_progress_callback(self, cb: Callable[[int, str], None]) -> None:
        """Set a callback for progress reporting: cb(percent, message)."""
        self._progress_callback = cb

    def _progress(self, pct: int, msg: str) -> None:
        if self._progress_callback:
            self._progress_callback(pct, msg)
        logger.debug("[%d%%] %s", pct, msg)

    def _next_change_id(self) -> str:
        self._change_counter += 1
        return f"CHG-{self._change_counter:04d}"

    def compare(
        self,
        old_doc: DocumentStructure,
        new_doc: DocumentStructure,
    ) -> CompareResult:
        """
        Compare two DocumentStructure objects.

        Steps:
        1. Separate text blocks and table blocks
        2. Match text blocks using paragraph_matcher
        3. Compare matched pairs -> MODIFY records
        4. Unmatched old -> DELETE records
        5. Unmatched new -> ADD records
        6. Detect moves among matched
        7. Compare tables
        8. Build CompareResult with all ChangeRecords
        9. Compute statistics

        Returns:
            CompareResult with all detected changes
        """
        self._change_counter = 0
        result = CompareResult(
            total_old_blocks=len(old_doc.blocks),
            total_new_blocks=len(new_doc.blocks),
        )

        threshold = float(self.options.get("similarity_threshold", 0.6))
        use_tfidf = True

        self._progress(5, "텍스트 블록 분리 중...")

        # Separate text and table blocks
        old_text_blocks = old_doc.get_text_blocks()
        new_text_blocks = new_doc.get_text_blocks()
        old_table_blocks = old_doc.get_table_blocks()
        new_table_blocks = new_doc.get_table_blocks()

        logger.info(
            "Comparing: old=%d text + %d tables, new=%d text + %d tables",
            len(old_text_blocks), len(old_table_blocks),
            len(new_text_blocks), len(new_table_blocks),
        )

        self._progress(15, "문단 매칭 중...")

        # Match text blocks
        matches = match_blocks(
            old_text_blocks, new_text_blocks,
            threshold=threshold,
            use_tfidf=use_tfidf,
        )

        matched_old_indices: Set[int] = {m[0] for m in matches}
        matched_new_indices: Set[int] = {m[1] for m in matches}

        self._progress(35, "변경 사항 분석 중...")

        # Process matches -> MODIFY records
        move_set: Set[Tuple[int, int]] = set()
        if self.options.get("detect_moves", True):
            move_pairs = detect_moves(matches, old_text_blocks, new_text_blocks)
            move_set = {(m[0], m[1]) for m in move_pairs}

        for old_idx, new_idx, sim in matches:
            old_block = old_text_blocks[old_idx]
            new_block = new_text_blocks[new_idx]

            is_move = (old_idx, new_idx) in move_set

            if sim >= 0.999 and not is_move:
                # Exactly equal - no record needed
                continue

            if is_move and sim >= 0.9:
                # Moved without significant content change
                record = self._build_move_record(old_block, new_block, sim)
                result.changes.append(record)
                result.moved_count += 1
                continue

            if is_move:
                # Moved AND modified
                record = self._build_modify_record(old_block, new_block, sim)
                record.note = "이동 및 수정됨"
                result.changes.append(record)
                result.modified_count += 1
                result.moved_count += 1
                continue

            # Format-only change detection
            if (sim >= 0.98
                    and self.options.get("include_format_changes", True)
                    and self._has_format_change(old_block, new_block)):
                record = self._build_format_record(old_block, new_block)
                result.changes.append(record)
                result.format_count += 1
                continue

            # Content modification
            record = self._build_modify_record(old_block, new_block, sim)
            result.changes.append(record)
            result.modified_count += 1

        self._progress(55, "삭제된 내용 처리 중...")

        # Unmatched old blocks -> DELETE
        for old_idx, old_block in enumerate(old_text_blocks):
            if old_idx not in matched_old_indices and old_block.content.strip():
                record = self._build_delete_record(old_block)
                result.changes.append(record)
                result.deleted_count += 1

        self._progress(65, "추가된 내용 처리 중...")

        # Unmatched new blocks -> ADD
        for new_idx, new_block in enumerate(new_text_blocks):
            if new_idx not in matched_new_indices and new_block.content.strip():
                record = self._build_add_record(new_block)
                result.changes.append(record)
                result.added_count += 1

        self._progress(75, "표 비교 중...")

        # Compare tables
        table_changes = self._compare_all_tables(
            old_table_blocks, new_table_blocks, threshold
        )
        result.changes.extend(table_changes)
        result.table_change_count = len(table_changes)

        self._progress(90, "통계 계산 중...")

        # Count heading changes
        result.heading_change_count = sum(
            1 for c in result.changes
            if c.object_type == ObjectType.HEADING
        )

        # Sort changes by location (page, paragraph_index)
        result.changes.sort(key=lambda c: (c.page_hint, c.location_info))

        # Compute statistics
        self._compute_statistics(result, old_doc, new_doc)

        self._progress(100, "비교 완료")
        logger.info(
            "Compare complete: %d changes (add=%d, del=%d, mod=%d, move=%d, fmt=%d, tbl=%d)",
            len(result.changes),
            result.added_count, result.deleted_count, result.modified_count,
            result.moved_count, result.format_count, result.table_change_count,
        )
        return result

    # ------------------------------------------------------------------
    # Record builders
    # ------------------------------------------------------------------

    def _block_location(self, block: DocumentBlock) -> str:
        """Build a human-readable location string for a block."""
        parts = []
        if block.section_title:
            parts.append(block.section_title[:30])
        parts.append(f"문단 {block.paragraph_index + 1}")
        return " > ".join(parts)

    def _page_hint_str(self, block: DocumentBlock) -> str:
        return f"{block.page_hint}페이지 (추정)" if block.page_hint else "페이지 미상"

    def _object_type_from_block(self, block: DocumentBlock) -> ObjectType:
        bt = block.block_type
        if bt == "heading":
            return ObjectType.HEADING
        if bt == "table":
            return ObjectType.TABLE
        if bt == "list_item":
            return ObjectType.LIST_ITEM
        if bt == "caption":
            return ObjectType.CAPTION
        return ObjectType.PARAGRAPH

    def _build_modify_record(
        self, old_block: DocumentBlock, new_block: DocumentBlock, sim: float
    ) -> ChangeRecord:
        loc = self._block_location(new_block)
        obj_type = self._object_type_from_block(new_block)
        importance = estimate_importance(old_block.content, new_block.content)

        # Choose diff granularity
        use_char = len(old_block.content) + len(new_block.content) < 300
        diff_spans = (
            char_diff(old_block.content, new_block.content)
            if use_char
            else word_diff(old_block.content, new_block.content)
        )

        summary = generate_summary(
            ChangeType.MODIFY.value, obj_type.value,
            loc, old_block.content, new_block.content
        )
        return ChangeRecord(
            change_id=self._next_change_id(),
            change_type=ChangeType.MODIFY,
            object_type=obj_type,
            location_info=loc,
            page_hint=self._page_hint_str(new_block),
            old_content=old_block.content,
            new_content=new_block.content,
            summary=summary,
            similarity=sim,
            importance=importance,
            review_needed=(importance == Importance.HIGH),
            diff_spans=diff_spans,
            section_title=new_block.section_title,
            old_block_id=old_block.block_id,
            new_block_id=new_block.block_id,
        )

    def _build_add_record(self, new_block: DocumentBlock) -> ChangeRecord:
        loc = self._block_location(new_block)
        obj_type = self._object_type_from_block(new_block)
        importance = Importance.HIGH if new_block.block_type == "heading" else Importance.MEDIUM
        summary = generate_summary(
            ChangeType.ADD.value, obj_type.value, loc, "", new_block.content
        )
        return ChangeRecord(
            change_id=self._next_change_id(),
            change_type=ChangeType.ADD,
            object_type=obj_type,
            location_info=loc,
            page_hint=self._page_hint_str(new_block),
            old_content="",
            new_content=new_block.content,
            summary=summary,
            similarity=0.0,
            importance=importance,
            review_needed=True,
            section_title=new_block.section_title,
            new_block_id=new_block.block_id,
        )

    def _build_delete_record(self, old_block: DocumentBlock) -> ChangeRecord:
        loc = self._block_location(old_block)
        obj_type = self._object_type_from_block(old_block)
        importance = Importance.HIGH if old_block.block_type == "heading" else Importance.MEDIUM
        summary = generate_summary(
            ChangeType.DELETE.value, obj_type.value, loc, old_block.content, ""
        )
        return ChangeRecord(
            change_id=self._next_change_id(),
            change_type=ChangeType.DELETE,
            object_type=obj_type,
            location_info=loc,
            page_hint=self._page_hint_str(old_block),
            old_content=old_block.content,
            new_content="",
            summary=summary,
            similarity=0.0,
            importance=importance,
            review_needed=True,
            section_title=old_block.section_title,
            old_block_id=old_block.block_id,
        )

    def _build_move_record(
        self, old_block: DocumentBlock, new_block: DocumentBlock, sim: float
    ) -> ChangeRecord:
        loc = self._block_location(new_block)
        obj_type = self._object_type_from_block(new_block)
        summary = generate_summary(
            ChangeType.MOVE.value, obj_type.value, loc,
            old_block.content, new_block.content
        )
        return ChangeRecord(
            change_id=self._next_change_id(),
            change_type=ChangeType.MOVE,
            object_type=obj_type,
            location_info=loc,
            page_hint=self._page_hint_str(new_block),
            old_content=old_block.content,
            new_content=new_block.content,
            summary=summary,
            similarity=sim,
            importance=Importance.LOW,
            review_needed=False,
            section_title=new_block.section_title,
            old_block_id=old_block.block_id,
            new_block_id=new_block.block_id,
        )

    def _build_format_record(
        self, old_block: DocumentBlock, new_block: DocumentBlock
    ) -> ChangeRecord:
        loc = self._block_location(new_block)
        obj_type = self._object_type_from_block(new_block)
        summary = generate_summary(
            ChangeType.FORMAT.value, obj_type.value, loc,
            old_block.content, new_block.content
        )
        return ChangeRecord(
            change_id=self._next_change_id(),
            change_type=ChangeType.FORMAT,
            object_type=obj_type,
            location_info=loc,
            page_hint=self._page_hint_str(new_block),
            old_content=old_block.content,
            new_content=new_block.content,
            summary=summary,
            similarity=1.0,
            importance=Importance.LOW,
            review_needed=False,
            section_title=new_block.section_title,
            old_block_id=old_block.block_id,
            new_block_id=new_block.block_id,
        )

    # ------------------------------------------------------------------
    # Table comparison
    # ------------------------------------------------------------------

    def _compare_all_tables(
        self,
        old_table_blocks: List[DocumentBlock],
        new_table_blocks: List[DocumentBlock],
        threshold: float,
    ) -> List[ChangeRecord]:
        """Match and compare all table blocks."""
        changes: List[ChangeRecord] = []

        # Simple sequential matching for tables
        matched_new: Set[int] = set()

        for o_idx, old_tbl_block in enumerate(old_table_blocks):
            if not old_tbl_block.table_data:
                continue

            best_sim = 0.0
            best_n_idx = -1

            for n_idx, new_tbl_block in enumerate(new_table_blocks):
                if n_idx in matched_new or not new_tbl_block.table_data:
                    continue
                sim = compute_similarity(old_tbl_block.content, new_tbl_block.content)
                if sim > best_sim:
                    best_sim = sim
                    best_n_idx = n_idx

            if best_n_idx >= 0 and best_sim >= threshold * 0.5:
                new_tbl_block = new_table_blocks[best_n_idx]
                matched_new.add(best_n_idx)

                loc = self._block_location(old_tbl_block)
                page_hint = self._page_hint_str(old_tbl_block)
                table_changes = compare_tables(
                    old_tbl_block.table_data,
                    new_tbl_block.table_data,
                    table_id=old_tbl_block.table_id or f"tbl_{o_idx}",
                    location_info=loc,
                    section_title=old_tbl_block.section_title,
                    page_hint=page_hint,
                )
                # Re-assign IDs to avoid collision
                for rec in table_changes:
                    rec.change_id = self._next_change_id()
                changes.extend(table_changes)
            else:
                # Deleted table
                record = ChangeRecord(
                    change_id=self._next_change_id(),
                    change_type=ChangeType.DELETE,
                    object_type=ObjectType.TABLE,
                    location_info=self._block_location(old_tbl_block),
                    page_hint=self._page_hint_str(old_tbl_block),
                    old_content=old_tbl_block.content[:200],
                    new_content="",
                    summary=f"표 삭제: {old_tbl_block.table_id}",
                    similarity=0.0,
                    importance=Importance.HIGH,
                    review_needed=True,
                    table_id=old_tbl_block.table_id,
                )
                changes.append(record)

        # Added tables
        for n_idx, new_tbl_block in enumerate(new_table_blocks):
            if n_idx not in matched_new and new_tbl_block.table_data:
                record = ChangeRecord(
                    change_id=self._next_change_id(),
                    change_type=ChangeType.ADD,
                    object_type=ObjectType.TABLE,
                    location_info=self._block_location(new_tbl_block),
                    page_hint=self._page_hint_str(new_tbl_block),
                    old_content="",
                    new_content=new_tbl_block.content[:200],
                    summary=f"표 추가: {new_tbl_block.table_id}",
                    similarity=0.0,
                    importance=Importance.HIGH,
                    review_needed=True,
                    table_id=new_tbl_block.table_id,
                )
                changes.append(record)

        return changes

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def _compute_statistics(
        self,
        result: CompareResult,
        old_doc: DocumentStructure,
        new_doc: DocumentStructure,
    ) -> None:
        """Compute overall_similarity and revision_rate."""
        old_texts = [b.content for b in old_doc.get_text_blocks() if b.content.strip()]
        new_texts = [b.content for b in new_doc.get_text_blocks() if b.content.strip()]

        if not old_texts or not new_texts:
            result.overall_similarity = 0.0
            result.revision_rate = 1.0
            return

        # Overall similarity: average of matched pairs' similarities
        modify_sims = [
            c.similarity for c in result.changes
            if c.change_type in (ChangeType.MODIFY, ChangeType.MOVE)
               and c.object_type != ObjectType.TABLE
        ]

        unchanged = max(0, len(old_texts) - result.deleted_count - result.modified_count)
        total_old = len(old_texts)

        if total_old > 0:
            if modify_sims:
                avg_modify_sim = sum(modify_sims) / len(modify_sims)
            else:
                avg_modify_sim = 1.0
            weight_unchanged = unchanged / total_old
            weight_modified = len(modify_sims) / total_old
            weight_deleted = result.deleted_count / total_old

            result.overall_similarity = (
                weight_unchanged * 1.0
                + weight_modified * avg_modify_sim
                + weight_deleted * 0.0
            )
            result.overall_similarity = max(0.0, min(1.0, result.overall_similarity))
            result.revision_rate = 1.0 - result.overall_similarity
        else:
            result.overall_similarity = 0.0
            result.revision_rate = 1.0

    def _has_format_change(
        self, old_block: DocumentBlock, new_block: DocumentBlock
    ) -> bool:
        """Check if blocks differ only in formatting (spans), not content."""
        if old_block.content != new_block.content:
            return False
        # Compare style names
        if old_block.style_name != new_block.style_name:
            return True
        # Compare span properties
        if len(old_block.spans) != len(new_block.spans):
            return True
        for old_span, new_span in zip(old_block.spans, new_block.spans):
            if (old_span.bold != new_span.bold
                    or old_span.italic != new_span.italic
                    or old_span.font_size != new_span.font_size):
                return True
        return False
