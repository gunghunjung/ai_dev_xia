"""
Paragraph matching engine.

Two-pass approach:
1. Exact/near-exact matches (similarity >= 0.9)
2. Fuzzy matches using TF-IDF similarity matrix
"""
from typing import List, Tuple, Set, Optional
import difflib

from app.models.document import DocumentBlock
from app.utils.logger import get_logger
from app.utils.text_utils import compute_similarity, compute_tfidf_similarity

logger = get_logger("diff_engine.paragraph_matcher")


def match_blocks(
    old_blocks: List[DocumentBlock],
    new_blocks: List[DocumentBlock],
    threshold: float = 0.6,
    use_tfidf: bool = True,
) -> List[Tuple[int, int, float]]:
    """
    Match old blocks to new blocks by content similarity.

    Args:
        old_blocks: List of DocumentBlock from original document
        new_blocks: List of DocumentBlock from new document
        threshold: Minimum similarity to consider a match (0.0 - 1.0)
        use_tfidf: Whether to use TF-IDF similarity (falls back to SequenceMatcher)

    Returns:
        List of (old_idx, new_idx, similarity) tuples, sorted by similarity descending.
        Each index appears at most once.
    """
    if not old_blocks or not new_blocks:
        return []

    old_texts = [b.content for b in old_blocks]
    new_texts = [b.content for b in new_blocks]

    logger.debug(
        "Matching %d old blocks against %d new blocks (threshold=%.2f)",
        len(old_blocks), len(new_blocks), threshold
    )

    # Pass 1: Exact matches
    matches: List[Tuple[int, int, float]] = []
    matched_old: Set[int] = set()
    matched_new: Set[int] = set()

    for o_idx, old_text in enumerate(old_texts):
        if not old_text.strip():
            continue
        for n_idx, new_text in enumerate(new_texts):
            if n_idx in matched_new or not new_text.strip():
                continue
            if old_text == new_text:
                matches.append((o_idx, n_idx, 1.0))
                matched_old.add(o_idx)
                matched_new.add(n_idx)
                break

    # Pass 2: Near-exact (>= 0.9) using SequenceMatcher
    for o_idx, old_text in enumerate(old_texts):
        if o_idx in matched_old or not old_text.strip():
            continue
        best_sim = 0.0
        best_n_idx = -1
        for n_idx, new_text in enumerate(new_texts):
            if n_idx in matched_new or not new_text.strip():
                continue
            sim = compute_similarity(old_text, new_text)
            if sim >= 0.9 and sim > best_sim:
                best_sim = sim
                best_n_idx = n_idx
        if best_n_idx >= 0:
            matches.append((o_idx, best_n_idx, best_sim))
            matched_old.add(o_idx)
            matched_new.add(best_n_idx)

    # Pass 3: Fuzzy matching for remaining unmatched blocks
    remaining_old = [i for i in range(len(old_blocks)) if i not in matched_old and old_texts[i].strip()]
    remaining_new = [i for i in range(len(new_blocks)) if i not in matched_new and new_texts[i].strip()]

    if remaining_old and remaining_new:
        fuzzy_matches = _fuzzy_match(
            remaining_old, remaining_new,
            old_texts, new_texts,
            threshold, use_tfidf
        )
        for (o_idx, n_idx, sim) in fuzzy_matches:
            if o_idx not in matched_old and n_idx not in matched_new:
                matches.append((o_idx, n_idx, sim))
                matched_old.add(o_idx)
                matched_new.add(n_idx)

    # Sort by old_idx for deterministic output
    matches.sort(key=lambda x: x[0])
    logger.debug("Matched %d block pairs", len(matches))
    return matches


def _fuzzy_match(
    old_indices: List[int],
    new_indices: List[int],
    old_texts: List[str],
    new_texts: List[str],
    threshold: float,
    use_tfidf: bool,
) -> List[Tuple[int, int, float]]:
    """
    Fuzzy matching using similarity matrix.
    Greedy: picks highest similarity pairs first.
    """
    # Build texts for this sub-problem
    sub_old = [old_texts[i] for i in old_indices]
    sub_new = [new_texts[j] for j in new_indices]

    # Compute similarity matrix
    sim_matrix = _build_similarity_matrix(sub_old, sub_new, use_tfidf)

    # Greedy matching: pick best pairs
    candidate_matches: List[Tuple[int, int, float]] = []
    for so_idx, row in enumerate(sim_matrix):
        for sn_idx, sim in enumerate(row):
            if sim >= threshold:
                candidate_matches.append((so_idx, sn_idx, sim))

    # Sort by similarity descending
    candidate_matches.sort(key=lambda x: -x[2])

    matched_old_sub: Set[int] = set()
    matched_new_sub: Set[int] = set()
    result: List[Tuple[int, int, float]] = []

    for so_idx, sn_idx, sim in candidate_matches:
        if so_idx in matched_old_sub or sn_idx in matched_new_sub:
            continue
        # Map back to original indices
        o_idx = old_indices[so_idx]
        n_idx = new_indices[sn_idx]
        result.append((o_idx, n_idx, sim))
        matched_old_sub.add(so_idx)
        matched_new_sub.add(sn_idx)

    return result


def _build_similarity_matrix(
    old_texts: List[str],
    new_texts: List[str],
    use_tfidf: bool,
) -> List[List[float]]:
    """
    Build an N x M similarity matrix between old and new text lists.
    Uses TF-IDF if available, otherwise SequenceMatcher.
    """
    n = len(old_texts)
    m = len(new_texts)
    matrix = [[0.0] * m for _ in range(n)]

    if use_tfidf and n > 0 and m > 0:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np

            all_texts = old_texts + new_texts
            all_texts_safe = [t if t.strip() else " " for t in all_texts]

            vectorizer = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(2, 4),
                min_df=1,
            )
            try:
                tfidf = vectorizer.fit_transform(all_texts_safe)
                old_mat = tfidf[:n]
                new_mat = tfidf[n:]
                sim_mat = cosine_similarity(old_mat, new_mat)
                for i in range(n):
                    for j in range(m):
                        matrix[i][j] = float(sim_mat[i, j])
                return matrix
            except ValueError:
                pass  # Fall through to SequenceMatcher
        except ImportError:
            pass

    # Fallback: SequenceMatcher for each pair
    for i, old_t in enumerate(old_texts):
        for j, new_t in enumerate(new_texts):
            matrix[i][j] = compute_similarity(old_t, new_t)

    return matrix


def detect_moves(
    matches: List[Tuple[int, int, float]],
    old_blocks: List[DocumentBlock],
    new_blocks: List[DocumentBlock],
    position_threshold: int = 5,
) -> List[Tuple[int, int, float]]:
    """
    Detect moved paragraphs from a list of matches.

    A paragraph is considered "moved" if:
    - It was matched (not add/delete)
    - Its relative position changed significantly

    Args:
        matches: List of (old_idx, new_idx, similarity) from match_blocks
        old_blocks: Original document blocks
        new_blocks: New document blocks
        position_threshold: Minimum position delta to consider a move

    Returns:
        List of (old_idx, new_idx, similarity) that are classified as moves.
    """
    if not matches:
        return []

    n_old = len(old_blocks)
    n_new = len(new_blocks)
    if n_old == 0 or n_new == 0:
        return []

    moves: List[Tuple[int, int, float]] = []

    for old_idx, new_idx, sim in matches:
        # Normalize positions to [0, 1]
        old_rel = old_idx / max(1, n_old - 1)
        new_rel = new_idx / max(1, n_new - 1)
        pos_delta = abs(old_rel - new_rel)

        # Also check absolute position difference
        abs_delta = abs(old_idx - new_idx)

        if abs_delta >= position_threshold and pos_delta > 0.1:
            moves.append((old_idx, new_idx, sim))

    logger.debug("Detected %d moved blocks", len(moves))
    return moves
