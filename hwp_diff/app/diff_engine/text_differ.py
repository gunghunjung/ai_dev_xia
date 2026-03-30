"""Text diffing utilities: word-level and character-level diffs with HTML highlighting."""
import re
import difflib
from typing import List, Tuple

from app.models.change_record import DiffSpan


def _tokenize_words(text: str) -> List[str]:
    """Tokenize text into word tokens, preserving separators."""
    # Split on word boundaries but keep the separators
    return re.findall(r"\S+|\s+", text) if text else []


def word_diff(old: str, new: str) -> List[DiffSpan]:
    """
    Compute word-level diff between old and new text.

    Returns:
        List of DiffSpan objects representing the diff operations.
    """
    if old == new:
        return [DiffSpan(start=0, end=len(old), change_type="equal",
                         old_text=old, new_text=new)]

    old_words = _tokenize_words(old)
    new_words = _tokenize_words(new)

    matcher = difflib.SequenceMatcher(None, old_words, new_words, autojunk=False)
    spans: List[DiffSpan] = []

    old_pos = 0
    new_pos = 0

    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        old_chunk = "".join(old_words[i1:i2])
        new_chunk = "".join(new_words[j1:j2])

        if op == "equal":
            spans.append(DiffSpan(
                start=old_pos,
                end=old_pos + len(old_chunk),
                change_type="equal",
                old_text=old_chunk,
                new_text=new_chunk,
            ))
            old_pos += len(old_chunk)
            new_pos += len(new_chunk)

        elif op == "replace":
            spans.append(DiffSpan(
                start=old_pos,
                end=old_pos + len(old_chunk),
                change_type="replace",
                old_text=old_chunk,
                new_text=new_chunk,
            ))
            old_pos += len(old_chunk)
            new_pos += len(new_chunk)

        elif op == "delete":
            spans.append(DiffSpan(
                start=old_pos,
                end=old_pos + len(old_chunk),
                change_type="delete",
                old_text=old_chunk,
                new_text="",
            ))
            old_pos += len(old_chunk)

        elif op == "insert":
            spans.append(DiffSpan(
                start=old_pos,
                end=old_pos,
                change_type="insert",
                old_text="",
                new_text=new_chunk,
            ))
            new_pos += len(new_chunk)

    return spans


def char_diff(old: str, new: str) -> List[DiffSpan]:
    """
    Compute character-level diff for short texts.
    Best suited for texts under ~200 characters.

    Returns:
        List of DiffSpan objects.
    """
    if old == new:
        return [DiffSpan(start=0, end=len(old), change_type="equal",
                         old_text=old, new_text=new)]

    matcher = difflib.SequenceMatcher(None, old, new, autojunk=False)
    spans: List[DiffSpan] = []
    old_pos = 0
    new_pos = 0

    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        old_chunk = old[i1:i2]
        new_chunk = new[j1:j2]

        if op == "equal":
            spans.append(DiffSpan(
                start=old_pos, end=old_pos + len(old_chunk),
                change_type="equal",
                old_text=old_chunk, new_text=new_chunk,
            ))
            old_pos += len(old_chunk)
            new_pos += len(new_chunk)

        elif op == "replace":
            spans.append(DiffSpan(
                start=old_pos, end=old_pos + len(old_chunk),
                change_type="replace",
                old_text=old_chunk, new_text=new_chunk,
            ))
            old_pos += len(old_chunk)
            new_pos += len(new_chunk)

        elif op == "delete":
            spans.append(DiffSpan(
                start=old_pos, end=old_pos + len(old_chunk),
                change_type="delete",
                old_text=old_chunk, new_text="",
            ))
            old_pos += len(old_chunk)

        elif op == "insert":
            spans.append(DiffSpan(
                start=old_pos, end=old_pos,
                change_type="insert",
                old_text="", new_text=new_chunk,
            ))
            new_pos += len(new_chunk)

    return spans


def _escape_html(text: str) -> str:
    """Escape HTML special characters."""
    return (text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;"))


def highlight_diff_html(old: str, new: str) -> Tuple[str, str]:
    """
    Return two HTML strings with highlighted diff marks.

    - Deleted text in old: <mark class="del">text</mark>
    - Inserted text in new: <mark class="ins">text</mark>
    - Replaced text: <mark class="del">old</mark> / <mark class="ins">new</mark>

    Returns:
        (old_html, new_html) tuple
    """
    if not old and not new:
        return "", ""

    # Use word diff for longer texts, char diff for short
    use_char = len(old) + len(new) < 200
    spans = char_diff(old, new) if use_char else word_diff(old, new)

    old_parts = []
    new_parts = []

    for span in spans:
        if span.change_type == "equal":
            escaped = _escape_html(span.old_text)
            old_parts.append(escaped)
            new_parts.append(_escape_html(span.new_text))

        elif span.change_type == "delete":
            old_parts.append(
                f'<mark style="background:#ffaaaa;text-decoration:line-through">'
                f'{_escape_html(span.old_text)}</mark>'
            )

        elif span.change_type == "insert":
            new_parts.append(
                f'<mark style="background:#aaffaa">'
                f'{_escape_html(span.new_text)}</mark>'
            )

        elif span.change_type == "replace":
            old_parts.append(
                f'<mark style="background:#ffaaaa;text-decoration:line-through">'
                f'{_escape_html(span.old_text)}</mark>'
            )
            new_parts.append(
                f'<mark style="background:#aaffaa">'
                f'{_escape_html(span.new_text)}</mark>'
            )

    return "".join(old_parts), "".join(new_parts)
