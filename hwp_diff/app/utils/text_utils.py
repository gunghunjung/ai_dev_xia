"""Text utility functions for document comparison."""
import re
import difflib
from typing import List, Optional, Dict, Any

from app.models.change_record import Importance


# ---------------------------------------------------------------------------
# Normalize text
# ---------------------------------------------------------------------------

def normalize_text(
    text: str,
    ignore_whitespace: bool = False,
    ignore_case: bool = False,
    ignore_newline: bool = False,
    strip_special: bool = False,
) -> str:
    """
    Normalize text according to specified options.

    Args:
        text: Input text string
        ignore_whitespace: Collapse multiple spaces into one and strip
        ignore_case: Convert to lowercase
        ignore_newline: Replace newlines with spaces
        strip_special: Remove special characters (keep Korean, alphanumeric, spaces)

    Returns:
        Normalized text string
    """
    if not text:
        return ""

    if ignore_newline:
        text = text.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")

    if ignore_whitespace:
        text = re.sub(r"\s+", " ", text).strip()
    else:
        # At minimum, strip leading/trailing whitespace
        text = text.strip()

    if ignore_case:
        text = text.lower()

    if strip_special:
        # Keep Korean (가-힣), alphanumeric, spaces, dots, commas
        text = re.sub(r"[^\uAC00-\uD7A3\u1100-\u11FF\u3130-\u318F\w\s.,·]", "", text)

    return text


# ---------------------------------------------------------------------------
# Similarity computation
# ---------------------------------------------------------------------------

def compute_similarity(text1: str, text2: str) -> float:
    """
    Compute text similarity using difflib SequenceMatcher.

    Returns:
        float in [0.0, 1.0]
    """
    if not text1 and not text2:
        return 1.0
    if not text1 or not text2:
        return 0.0
    return difflib.SequenceMatcher(None, text1, text2).ratio()


def compute_tfidf_similarity(texts1: List[str], texts2: List[str]) -> List[float]:
    """
    Compute pairwise TF-IDF cosine similarity between corresponding pairs.
    Falls back to SequenceMatcher if sklearn unavailable.

    Args:
        texts1: List of reference texts
        texts2: List of comparison texts (same length as texts1)

    Returns:
        List of similarity scores [0.0, 1.0]
    """
    if len(texts1) != len(texts2):
        raise ValueError("texts1 and texts2 must have the same length")

    if not texts1:
        return []

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        all_texts = texts1 + texts2
        # Handle empty strings
        all_texts_safe = [t if t.strip() else " " for t in all_texts]

        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            min_df=1,
        )
        try:
            tfidf_matrix = vectorizer.fit_transform(all_texts_safe)
        except ValueError:
            # Fallback if vocabulary is empty
            return [compute_similarity(t1, t2) for t1, t2 in zip(texts1, texts2)]

        n = len(texts1)
        matrix1 = tfidf_matrix[:n]
        matrix2 = tfidf_matrix[n:]

        similarities = []
        for i in range(n):
            sim = cosine_similarity(matrix1[i], matrix2[i])[0][0]
            similarities.append(float(sim))
        return similarities

    except ImportError:
        # Fallback to SequenceMatcher
        return [compute_similarity(t1, t2) for t1, t2 in zip(texts1, texts2)]


# ---------------------------------------------------------------------------
# Numbering detection
# ---------------------------------------------------------------------------

_NUMBERING_PATTERNS = [
    # Dotted numeric like 1., 1.1, 1.1.1
    (re.compile(r"^(\d+(?:\.\d+)*\.?)\s"), "numeric_dotted"),
    # Parenthesized like (1), (가)
    (re.compile(r"^(\(\d+\)|\([가-힣]\))\s"), "parenthesized"),
    # Korean ordinals: 가., 나., 다., 라.
    (re.compile(r"^([가-힣]\.)\s"), "korean_alpha"),
    # Roman numerals: I., II., III.
    (re.compile(r"^((?:I{1,3}|IV|VI{0,3}|IX|X{1,3})\.)\s", re.IGNORECASE), "roman"),
    # Letter + dot: a., b.
    (re.compile(r"^([a-zA-Z]\.)\s"), "alpha"),
    # Circled numbers (unicode): ①②③...
    (re.compile(r"^([\u2460-\u2473])\s?"), "circled"),
    # Em dash or bullet: -, ·, •
    (re.compile(r"^([-·•])\s"), "bullet"),
]


def detect_numbering(text: str) -> str:
    """
    Detect and return the numbering prefix of a text string.

    Returns:
        The numbering prefix string, or "" if none detected.
    """
    if not text:
        return ""
    text = text.strip()
    for pattern, _ in _NUMBERING_PATTERNS:
        m = pattern.match(text)
        if m:
            return m.group(1)
    return ""


# ---------------------------------------------------------------------------
# Importance estimation
# ---------------------------------------------------------------------------

_NUMBER_PATTERN = re.compile(r"\d+(?:[.,]\d+)?")
_UNIT_PATTERN = re.compile(r"[℃°%Ω㎜㎝㎞㎡㎥㎏mg㎎mL㎖kPaMPaGPa]|[Vv]olt|[Aa]mp|[Ww]att|[Hh]z")
_KEYWORD_HIGH = [
    "합격기준", "불합격", "허용오차", "임계값", "한계", "최대", "최소",
    "규격", "기준", "판정", "오차", "±", "이상", "이하", "초과", "미만",
    "approval", "reject", "criteria", "limit", "threshold",
]


def estimate_importance(old_text: str, new_text: str) -> Importance:
    """
    Estimate the importance of a change based on content analysis.

    Rules:
    - HIGH: numeric/unit values changed, criteria keywords present
    - MEDIUM: semantic content changed (non-trivial word changes)
    - LOW: whitespace/punctuation only differences

    Returns:
        Importance enum value
    """
    if not old_text and not new_text:
        return Importance.LOW
    if not old_text or not new_text:
        return Importance.HIGH  # Complete add/delete of substantive content

    # Check for numeric changes
    old_nums = set(_NUMBER_PATTERN.findall(old_text))
    new_nums = set(_NUMBER_PATTERN.findall(new_text))
    if old_nums != new_nums:
        return Importance.HIGH

    # Check for unit changes
    if _UNIT_PATTERN.search(old_text) or _UNIT_PATTERN.search(new_text):
        old_units = set(_UNIT_PATTERN.findall(old_text))
        new_units = set(_UNIT_PATTERN.findall(new_text))
        if old_units != new_units:
            return Importance.HIGH

    # Check for high-importance keywords
    combined = (old_text + new_text).lower()
    for kw in _KEYWORD_HIGH:
        if kw.lower() in combined:
            return Importance.HIGH

    # Check if only whitespace/punctuation changed
    old_stripped = re.sub(r"[\s\W]", "", old_text)
    new_stripped = re.sub(r"[\s\W]", "", new_text)
    if old_stripped == new_stripped:
        return Importance.LOW

    # Otherwise medium
    return Importance.MEDIUM


# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------

_CHANGE_TYPE_KR = {
    "추가": "추가",
    "삭제": "삭제",
    "수정": "수정",
    "이동": "이동",
    "서식변경": "서식 변경",
}


def generate_summary(
    change_type: str,
    object_type: str,
    location: str,
    old_text: str,
    new_text: str,
    max_preview: int = 40,
) -> str:
    """
    Generate a human-readable Korean summary of a change.

    Args:
        change_type: ChangeType value string (e.g., "수정")
        object_type: ObjectType value string (e.g., "문단")
        location: Location string (e.g., "3장 2절 문단 5")
        old_text: Original text content
        new_text: New text content
        max_preview: Max characters for text preview

    Returns:
        Korean summary string
    """
    ct = _CHANGE_TYPE_KR.get(change_type, change_type)
    loc = location if location else "불명확한 위치"

    def _preview(text: str) -> str:
        text = text.strip().replace("\n", " ")
        if len(text) > max_preview:
            return text[:max_preview] + "..."
        return text

    if change_type == "추가":
        preview = _preview(new_text)
        return f"{loc}에 {object_type} {ct}: \"{preview}\""
    elif change_type == "삭제":
        preview = _preview(old_text)
        return f"{loc}에서 {object_type} {ct}: \"{preview}\""
    elif change_type == "수정":
        old_p = _preview(old_text)
        new_p = _preview(new_text)
        # Highlight specific numbers if changed
        old_nums = _NUMBER_PATTERN.findall(old_text)
        new_nums = _NUMBER_PATTERN.findall(new_text)
        if old_nums != new_nums and old_nums and new_nums:
            num_changes = []
            for on, nn in zip(old_nums[:3], new_nums[:3]):
                if on != nn:
                    num_changes.append(f"{on} → {nn}")
            if num_changes:
                return f"{loc} {object_type} {ct}: {', '.join(num_changes)} (수치 변경)"
        return f"{loc} {object_type} {ct}: \"{old_p}\" → \"{new_p}\""
    elif change_type == "이동":
        preview = _preview(new_text or old_text)
        return f"{object_type} 이동: \"{preview}\""
    elif change_type == "서식변경":
        return f"{loc} {object_type} 서식 변경"
    else:
        return f"{loc} {object_type} {ct}"
