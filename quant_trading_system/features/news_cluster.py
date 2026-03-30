# features/news_cluster.py — 뉴스 중복 제거 및 이벤트 클러스터링
"""
2단계 중복 제거 전략:
  1단계: URL/해시 기반 완전 중복 → news_db가 INSERT OR IGNORE 처리
  2단계: 의미 중복 (같은 이슈 반복 보도) → 이 모듈이 처리

클러스터링 알고리즘:
  - TF-IDF 벡터화 (제목 + 주요 키워드)
  - 코사인 유사도 행렬 계산
  - 탐욕적 클러스터 병합 (Connected Components)
  - 클러스터당 대표 기사 선정 (중요도 × 시간 최신성)

외부 의존성:
  - scikit-learn (TfidfVectorizer) — 있으면 사용, 없으면 단순 자카드 유사도 fallback
"""
from __future__ import annotations

import logging
import math
import re
import uuid
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("quant.news_cluster")

# ── 상수 ──────────────────────────────────────────────────────────────────────

_DEFAULT_THRESHOLD = 0.75   # 코사인 유사도 임계값 (이 이상 = 같은 이벤트)
_MAX_CLUSTER_SIZE  = 200    # 하나의 클러스터 최대 멤버 수 (이상치 방지)


# ── 한국어/영어 토크나이저 ────────────────────────────────────────────────────

_KO_STOPWORDS = {
    "및", "등", "의", "를", "이", "가", "은", "는", "에", "에서",
    "으로", "로", "한", "하고", "했다", "하는", "것", "이다", "하며",
    "대한", "그", "이번", "올해", "지난", "앞서", "한편", "또한",
    "관련", "위해", "때문", "위한", "통해", "따라", "있다", "된다",
}

_EN_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "in", "on", "at", "to", "for", "of", "and", "or", "but",
    "that", "this", "it", "as", "by", "with", "from", "has",
    "have", "had", "will", "would", "could", "said", "says",
}


def _tokenize(text: str) -> List[str]:
    """한국어/영어 혼용 텍스트 단순 토크나이징"""
    text = text.lower()
    # 숫자+단위 보존, 나머지 특수문자 제거
    text = re.sub(r"[^\w\s%$€¥₩\.]", " ", text)
    tokens = text.split()
    result = []
    for t in tokens:
        if len(t) < 2:
            continue
        if t in _KO_STOPWORDS or t in _EN_STOPWORDS:
            continue
        result.append(t)
    return result


# ── TF-IDF (sklearn fallback → 자체 구현) ────────────────────────────────────

class _SimpleTFIDF:
    """sklearn 없을 때 사용하는 경량 TF-IDF"""

    def __init__(self, max_features: int = 500):
        self.max_features = max_features
        self._vocab: Dict[str, int] = {}
        self._idf: Dict[str, float] = {}

    def fit_transform(self, texts: List[str]):
        tokens_list = [_tokenize(t) for t in texts]
        n = len(texts)

        # DF 계산
        df: Dict[str, int] = defaultdict(int)
        for tokens in tokens_list:
            for t in set(tokens):
                df[t] += 1

        # IDF 계산 (smoothed)
        all_tokens_sorted = sorted(df.items(), key=lambda x: -x[1])
        vocab_tokens = [t for t, _ in all_tokens_sorted[:self.max_features]]
        self._vocab = {t: i for i, t in enumerate(vocab_tokens)}
        for t in vocab_tokens:
            self._idf[t] = math.log((n + 1) / (df[t] + 1)) + 1.0

        # TF-IDF 행렬 (희소 표현: list of dicts)
        vectors = []
        for tokens in tokens_list:
            tf: Dict[str, float] = defaultdict(float)
            for t in tokens:
                if t in self._vocab:
                    tf[t] += 1.0
            n_tok = max(len(tokens), 1)
            vec: Dict[int, float] = {}
            for t, count in tf.items():
                if t in self._vocab:
                    idx = self._vocab[t]
                    vec[idx] = (count / n_tok) * self._idf[t]
            vectors.append(vec)
        return vectors

    @staticmethod
    def cosine_similarity(v1: Dict[int, float], v2: Dict[int, float]) -> float:
        """희소 벡터 코사인 유사도"""
        dot = sum(v1[k] * v2[k] for k in v1 if k in v2)
        n1  = math.sqrt(sum(x * x for x in v1.values()))
        n2  = math.sqrt(sum(x * x for x in v2.values()))
        if n1 < 1e-9 or n2 < 1e-9:
            return 0.0
        return dot / (n1 * n2)


def _try_sklearn_tfidf(texts: List[str], max_features: int = 500):
    """sklearn TfidfVectorizer 시도, 실패 시 None"""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(
            max_features=max_features,
            analyzer="word",
            token_pattern=r"(?u)\b\w\w+\b",
            sublinear_tf=True,
            ngram_range=(1, 2),
        )
        mat = vec.fit_transform(texts)
        return mat  # scipy sparse matrix
    except Exception:
        return None


def _cosine_sparse(mat, i: int, j: int) -> float:
    """scipy sparse matrix 행 i, j 간 코사인 유사도"""
    try:
        from sklearn.metrics.pairwise import cosine_similarity as sk_cos
        import numpy as np
        ri = mat[i]
        rj = mat[j]
        return float(sk_cos(ri, rj)[0, 0])
    except Exception:
        return 0.0


# ── 클러스터러 ────────────────────────────────────────────────────────────────

class NewsClusterer:
    """
    뉴스 이벤트 클러스터링.

    cluster(events) → Dict[cluster_id, [event_dict, ...]]
      - 같은 이슈의 반복 보도 → 하나의 클러스터
      - 클러스터 내 첫 번째 항목이 대표 기사 (가장 높은 confidence × importance)
    """

    def __init__(
        self,
        similarity_threshold: float = _DEFAULT_THRESHOLD,
        max_features: int = 500,
        max_per_window: int = 500,   # 한 번에 처리할 최대 이벤트 수
    ):
        self.threshold     = similarity_threshold
        self.max_features  = max_features
        self.max_per_window = max_per_window

    def cluster(self, events: List[Dict]) -> Dict[str, List[Dict]]:
        """
        이벤트 목록 클러스터링.

        Args:
            events: news_events 딕셔너리 목록 (event_id, title, keywords, ... 포함)

        Returns:
            {cluster_id: [event_dict, ...]}  (대표 기사가 첫 번째)
        """
        if not events:
            return {}

        events = events[:self.max_per_window]
        texts  = [self._event_text(e) for e in events]
        n      = len(texts)

        # ── 벡터화 ──────────────────────────────────────────────────────
        sklearn_mat = _try_sklearn_tfidf(texts, self.max_features)
        use_sklearn = sklearn_mat is not None

        if not use_sklearn:
            tfidf   = _SimpleTFIDF(self.max_features)
            vectors = tfidf.fit_transform(texts)

        # ── 유사도 행렬 기반 그래프 구성 ─────────────────────────────────
        # 메모리 절약: 유사도 임계값 이상인 쌍만 저장
        edges: Dict[int, List[int]] = defaultdict(list)

        for i in range(n):
            for j in range(i + 1, n):
                # 카테고리가 완전히 다르면 빠른 스킵
                if not self._categories_overlap(events[i], events[j]):
                    continue

                if use_sklearn:
                    sim = _cosine_sparse(sklearn_mat, i, j)
                else:
                    sim = _SimpleTFIDF.cosine_similarity(vectors[i], vectors[j])

                if sim >= self.threshold:
                    edges[i].append(j)
                    edges[j].append(i)

        # ── Connected Components → 클러스터 ─────────────────────────────
        visited  = [False] * n
        clusters: Dict[str, List[Dict]] = {}

        for start in range(n):
            if visited[start]:
                continue

            # BFS
            component = []
            queue     = [start]
            visited[start] = True
            while queue:
                node = queue.pop(0)
                component.append(node)
                for neighbor in edges[node]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
                if len(component) >= _MAX_CLUSTER_SIZE:
                    break

            # 클러스터 ID 생성 및 대표 기사 선정
            cluster_id = str(uuid.uuid4())[:8]
            members    = [events[idx] for idx in component]
            members    = self._sort_by_importance(members)
            clusters[cluster_id] = members

        logger.debug(f"클러스터링: {n}건 → {len(clusters)}개 클러스터")
        return clusters

    def get_representative_events(self, events: List[Dict]) -> List[Dict]:
        """클러스터별 대표 이벤트만 반환 (중복 제거 결과)"""
        clusters = self.cluster(events)
        reps     = []
        for cluster_id, members in clusters.items():
            rep = members[0]
            rep["cluster_id"]     = cluster_id
            rep["repeat_count"]   = len(members)
            rep["is_representative"] = 1
            reps.append(rep)
        return reps

    # ── 내부 메서드 ───────────────────────────────────────────────────────────

    @staticmethod
    def _event_text(evt: Dict) -> str:
        """이벤트 → 벡터화용 텍스트"""
        parts = [evt.get("title", "")]
        kws   = evt.get("keywords", [])
        if isinstance(kws, list):
            parts.extend(kws[:8])
        cats = evt.get("categories", [])
        if isinstance(cats, list):
            parts.extend(cats)
        return " ".join(str(p) for p in parts if p)

    @staticmethod
    def _categories_overlap(e1: Dict, e2: Dict) -> bool:
        """두 이벤트가 공통 카테고리를 가지는지 확인 (빠른 사전 필터)"""
        c1 = set(e1.get("categories", []) or [])
        c2 = set(e2.get("categories", []) or [])
        # 카테고리 정보가 없으면 true (필터 건너뜀)
        if not c1 or not c2:
            return True
        # primary_category가 같은 경우도 overlap
        if e1.get("primary_category") == e2.get("primary_category"):
            return True
        return bool(c1 & c2)

    @staticmethod
    def _sort_by_importance(members: List[Dict]) -> List[Dict]:
        """
        중요도 × 신뢰도 기준 정렬 → 가장 중요한 기사가 첫 번째.
        같은 점수면 최신 기사 우선.
        """
        def score(e: Dict) -> float:
            imp = e.get("importance", 0.5)
            con = e.get("confidence", 0.5)
            # abs(computed_score)도 고려
            sc  = abs(e.get("computed_score", 0.0))
            return imp * con + sc * 0.3

        return sorted(members, key=score, reverse=True)


# ── 자카드 유사도 폴백 ─────────────────────────────────────────────────────────

def jaccard_similarity(text1: str, text2: str) -> float:
    """단순 자카드 유사도 (torch/sklearn 미설치 환경용)"""
    s1 = set(_tokenize(text1))
    s2 = set(_tokenize(text2))
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)
