from .diff_engine import DiffEngine
from .text_differ import word_diff, char_diff, highlight_diff_html
from .paragraph_matcher import match_blocks, detect_moves
from .table_matcher import compare_tables

__all__ = [
    "DiffEngine",
    "word_diff", "char_diff", "highlight_diff_html",
    "match_blocks", "detect_moves",
    "compare_tables",
]
