"""Table comparison engine - compares two TableData structures cell by cell."""
import uuid
from typing import List, Optional

from app.models.document import TableData, TableCell
from app.models.change_record import ChangeRecord, ChangeType, ObjectType, Importance, DiffSpan
from app.utils.logger import get_logger
from app.utils.text_utils import compute_similarity, estimate_importance, generate_summary

logger = get_logger("diff_engine.table_matcher")


def compare_tables(
    old_table: TableData,
    new_table: TableData,
    table_id: str = "",
    location_info: str = "",
    section_title: str = "",
    page_hint: str = "",
) -> List[ChangeRecord]:
    """
    Compare two tables cell by cell and return change records.

    Detects:
    - Cell content modification
    - Row additions/deletions (if row counts differ)
    - Column additions/deletions (if column counts differ)

    Args:
        old_table: Original TableData
        new_table: New TableData
        table_id: Identifier for the table
        location_info: Human-readable location string
        section_title: Section containing the table
        page_hint: Page hint string

    Returns:
        List of ChangeRecord for each detected change.
    """
    changes: List[ChangeRecord] = []
    change_counter = [0]  # Use list for closure mutation

    def next_id() -> str:
        change_counter[0] += 1
        return f"TC-{table_id}-{change_counter[0]:04d}"

    old_grid = old_table.get_text_grid()
    new_grid = new_table.get_text_grid()

    old_rows = len(old_grid)
    new_rows = len(new_grid)
    old_cols = old_table.cols
    new_cols = new_table.cols

    # Determine comparison bounds
    common_rows = min(old_rows, new_rows)
    common_cols = min(old_cols, new_cols)

    # Compare cells in the common region
    for r in range(common_rows):
        old_row = old_grid[r] if r < old_rows else []
        new_row = new_grid[r] if r < new_rows else []

        for c in range(common_cols):
            old_cell_text = old_row[c] if c < len(old_row) else ""
            new_cell_text = new_row[c] if c < len(new_row) else ""

            if old_cell_text == new_cell_text:
                continue

            sim = compute_similarity(old_cell_text, new_cell_text)
            importance = estimate_importance(old_cell_text, new_cell_text)
            loc = f"{location_info} [{r+1}행 {c+1}열]" if location_info else f"[{r+1}행 {c+1}열]"
            summary = generate_summary(
                ChangeType.MODIFY.value, ObjectType.CELL.value,
                loc, old_cell_text, new_cell_text
            )

            # Build diff spans
            from app.diff_engine.text_differ import char_diff
            diff_spans = char_diff(old_cell_text, new_cell_text)

            record = ChangeRecord(
                change_id=next_id(),
                change_type=ChangeType.MODIFY,
                object_type=ObjectType.CELL,
                location_info=loc,
                page_hint=page_hint,
                old_content=old_cell_text,
                new_content=new_cell_text,
                summary=summary,
                similarity=sim,
                importance=importance,
                review_needed=(importance == Importance.HIGH),
                diff_spans=diff_spans,
                section_title=section_title,
                table_id=table_id,
                row=r,
                col=c,
            )
            changes.append(record)

    # Added rows (new_rows > old_rows)
    for r in range(common_rows, new_rows):
        new_row = new_grid[r] if r < new_rows else []
        row_text = " | ".join(new_row)
        if not row_text.strip():
            continue

        loc = f"{location_info} [{r+1}행 추가]" if location_info else f"[{r+1}행 추가]"
        summary = generate_summary(
            ChangeType.ADD.value, ObjectType.TABLE.value,
            loc, "", row_text
        )
        record = ChangeRecord(
            change_id=next_id(),
            change_type=ChangeType.ADD,
            object_type=ObjectType.TABLE,
            location_info=loc,
            page_hint=page_hint,
            old_content="",
            new_content=row_text,
            summary=summary,
            similarity=0.0,
            importance=Importance.MEDIUM,
            review_needed=True,
            section_title=section_title,
            table_id=table_id,
            row=r,
            col=-1,
        )
        changes.append(record)

    # Deleted rows (old_rows > new_rows)
    for r in range(common_rows, old_rows):
        old_row = old_grid[r] if r < old_rows else []
        row_text = " | ".join(old_row)
        if not row_text.strip():
            continue

        loc = f"{location_info} [{r+1}행 삭제]" if location_info else f"[{r+1}행 삭제]"
        summary = generate_summary(
            ChangeType.DELETE.value, ObjectType.TABLE.value,
            loc, row_text, ""
        )
        record = ChangeRecord(
            change_id=next_id(),
            change_type=ChangeType.DELETE,
            object_type=ObjectType.TABLE,
            location_info=loc,
            page_hint=page_hint,
            old_content=row_text,
            new_content="",
            summary=summary,
            similarity=0.0,
            importance=Importance.MEDIUM,
            review_needed=True,
            section_title=section_title,
            table_id=table_id,
            row=r,
            col=-1,
        )
        changes.append(record)

    # Added columns (new_cols > old_cols)
    for c in range(common_cols, new_cols):
        col_texts = []
        for r in range(new_rows):
            row = new_grid[r] if r < new_rows else []
            col_texts.append(row[c] if c < len(row) else "")
        col_text = " | ".join(ct for ct in col_texts if ct)
        if not col_text.strip():
            continue

        loc = f"{location_info} [{c+1}열 추가]" if location_info else f"[{c+1}열 추가]"
        summary = f"{loc}: 새 열 추가"
        record = ChangeRecord(
            change_id=next_id(),
            change_type=ChangeType.ADD,
            object_type=ObjectType.TABLE,
            location_info=loc,
            page_hint=page_hint,
            old_content="",
            new_content=col_text,
            summary=summary,
            similarity=0.0,
            importance=Importance.MEDIUM,
            review_needed=True,
            section_title=section_title,
            table_id=table_id,
            row=-1,
            col=c,
        )
        changes.append(record)

    # Deleted columns (old_cols > new_cols)
    for c in range(common_cols, old_cols):
        col_texts = []
        for r in range(old_rows):
            row = old_grid[r] if r < old_rows else []
            col_texts.append(row[c] if c < len(row) else "")
        col_text = " | ".join(ct for ct in col_texts if ct)
        if not col_text.strip():
            continue

        loc = f"{location_info} [{c+1}열 삭제]" if location_info else f"[{c+1}열 삭제]"
        summary = f"{loc}: 열 삭제"
        record = ChangeRecord(
            change_id=next_id(),
            change_type=ChangeType.DELETE,
            object_type=ObjectType.TABLE,
            location_info=loc,
            page_hint=page_hint,
            old_content=col_text,
            new_content="",
            summary=summary,
            similarity=0.0,
            importance=Importance.MEDIUM,
            review_needed=True,
            section_title=section_title,
            table_id=table_id,
            row=-1,
            col=c,
        )
        changes.append(record)

    logger.debug("Table comparison found %d changes", len(changes))
    return changes
