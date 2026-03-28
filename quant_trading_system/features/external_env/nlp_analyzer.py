# features/external_env/nlp_analyzer.py — NLP 분석 엔진
"""
뉴스 텍스트 NLP 분석:
  1. 감성 점수  (-1 ~ +1)
  2. 중요도     (0 ~ 1)
  3. 키워드 TF-IDF

우선순위:
  [1] FinBERT (transformers 설치 시) → 가장 정확
  [2] VADER (vaderSentiment 설치 시) → 중간 정확도
  [3] 규칙 기반 (항상 동작)          → fallback
"""
from __future__ import annotations
import re
import logging
import math
from typing import Optional

logger = logging.getLogger("quant.external_env.nlp")

# ──────────────────────────────────────────────────────────────────────
# 선택적 임포트
# ──────────────────────────────────────────────────────────────────────

_VADER_AVAILABLE = False
_FINBERT_AVAILABLE = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as _VADER
    _VADER_AVAILABLE = True
except ImportError:
    pass

try:
    # transformers는 import 시점에 바로 로드하지 않고 lazy 체크만
    import importlib
    _transformers_spec = importlib.util.find_spec("transformers")
    _FINBERT_AVAILABLE = _transformers_spec is not None
except Exception:
    _FINBERT_AVAILABLE = False
_hf_pipeline = None  # 실제 사용 시 지연 로드


# ──────────────────────────────────────────────────────────────────────
# 규칙 기반 감성 사전
# ──────────────────────────────────────────────────────────────────────

_POS_STRONG = ["급등", "폭등", "사상최고", "beat", "surpass", "record", "rally",
               "호실적", "최대", "어닝서프라이즈", "상회", "acquisition win"]
_POS_MILD   = ["상승", "증가", "성장", "개선", "긍정", "호재", "회복",
               "rise", "gain", "grow", "positive", "upbeat", "recovery"]
_NEG_STRONG = ["급락", "폭락", "패닉", "파산", "default", "crash", "collapse",
               "위기", "공황", "miss", "실망", "어닝쇼크", "하회"]
_NEG_MILD   = ["하락", "감소", "악화", "부정", "악재", "우려",
               "fall", "decline", "drop", "negative", "concern", "risk"]

# 중요도 부스터 (이 단어가 있으면 중요도 +)
_IMPORTANCE_BOOST = [
    "fomc", "연준", "금리", "기준금리", "cpi", "gdp", "전쟁", "war",
    "실적", "earnings", "ipo", "합병", "인수", "규제", "제재",
    "반도체", "ai", "ChatGPT", "삼성", "애플", "nvidia",
]


# ──────────────────────────────────────────────────────────────────────
# NLP 분석기 클래스
# ──────────────────────────────────────────────────────────────────────

class NLPAnalyzer:
    """
    텍스트 → (sentiment_score, importance, keywords) 변환.

    lazy-load: FinBERT/VADER는 처음 사용 시에만 초기화.
    """

    def __init__(self, prefer_finbert: bool = False):
        self._prefer_finbert = prefer_finbert
        self._vader: Optional[object] = None
        self._finbert = None
        self._finbert_loaded = False

        if _VADER_AVAILABLE:
            try:
                self._vader = _VADER()
                logger.info("NLPAnalyzer: VADER 초기화 완료")
            except Exception as e:
                logger.warning(f"NLPAnalyzer: VADER 초기화 실패: {e}")

        mode = "FinBERT" if prefer_finbert and _FINBERT_AVAILABLE \
               else ("VADER" if _VADER_AVAILABLE else "규칙기반")
        logger.info(f"NLPAnalyzer 모드: {mode}")

    # ──────────────────────────────────────────────────────────────────
    # 공개 API
    # ──────────────────────────────────────────────────────────────────

    def analyze(self, title: str, summary: str = "") -> dict:
        """
        Returns:
            {
                "sentiment_score": float (-1~+1),
                "importance":      float (0~1),
                "keywords":        list[str],
                "confidence":      float (0~1),
                "method":          str
            }
        """
        text = title + " " + summary

        # 감성 점수
        if self._prefer_finbert and _FINBERT_AVAILABLE:
            sentiment, conf, method = self._finbert_sentiment(text)
        elif self._vader is not None:
            sentiment, conf, method = self._vader_sentiment(text)
        else:
            sentiment, conf, method = self._rule_sentiment(text)

        importance = self._compute_importance(title, summary)
        keywords   = self._extract_keywords_tfidf(title, summary)

        return {
            "sentiment_score": round(sentiment, 4),
            "importance":      round(importance, 4),
            "keywords":        keywords,
            "confidence":      round(conf, 4),
            "method":          method,
        }

    def batch_analyze(self, items: list[tuple[str, str]]) -> list[dict]:
        """(title, summary) 리스트 배치 처리"""
        return [self.analyze(t, s) for t, s in items]

    # ──────────────────────────────────────────────────────────────────
    # 감성 분석 메서드
    # ──────────────────────────────────────────────────────────────────

    def _finbert_sentiment(self, text: str) -> tuple[float, float, str]:
        """FinBERT 감성 (lazy load)"""
        if not self._finbert_loaded:
            try:
                global _hf_pipeline
                if _hf_pipeline is None:
                    from transformers import pipeline as _hf_pipeline
                self._finbert = _hf_pipeline(
                    "text-classification",
                    model="ProsusAI/finbert",
                    truncation=True, max_length=512,
                )
                self._finbert_loaded = True
                logger.info("FinBERT 로드 완료")
            except Exception as e:
                logger.warning(f"FinBERT 로드 실패: {e} → VADER 사용")
                self._finbert_loaded = True  # 재시도 방지
                self._finbert = None

        if self._finbert is None:
            if self._vader:
                return self._vader_sentiment(text)
            return self._rule_sentiment(text)

        try:
            truncated = text[:512]
            result = self._finbert(truncated)[0]
            label  = result["label"].lower()   # positive / negative / neutral
            score  = float(result["score"])
            if label == "positive":
                return score, score, "FinBERT"
            elif label == "negative":
                return -score, score, "FinBERT"
            else:
                return 0.0, score, "FinBERT"
        except Exception as e:
            logger.debug(f"FinBERT 추론 실패: {e}")
            return self._rule_sentiment(text)

    def _vader_sentiment(self, text: str) -> tuple[float, float, str]:
        """VADER 감성 (영어 텍스트에 최적)"""
        try:
            scores = self._vader.polarity_scores(text)
            compound = float(scores["compound"])    # -1 ~ +1
            conf = abs(compound) * 0.8 + 0.2       # 신뢰도 보정
            return compound, min(conf, 0.95), "VADER"
        except Exception:
            return self._rule_sentiment(text)

    def _rule_sentiment(self, text: str) -> tuple[float, float, str]:
        """규칙 기반 감성 점수 (항상 동작)"""
        t = text.lower()
        ps = sum(2 for w in _POS_STRONG if w in t) + sum(1 for w in _POS_MILD if w in t)
        ns = sum(2 for w in _NEG_STRONG if w in t) + sum(1 for w in _NEG_MILD if w in t)

        total = ps + ns
        if total == 0:
            return 0.0, 0.3, "Rule"

        score = (ps - ns) / total
        conf  = min(total * 0.08 + 0.3, 0.8)
        return round(score, 4), round(conf, 4), "Rule"

    # ──────────────────────────────────────────────────────────────────
    # 중요도 계산
    # ──────────────────────────────────────────────────────────────────

    def _compute_importance(self, title: str, summary: str) -> float:
        """
        중요도 = f(키워드 부스터, 문장 길이, 특수 마커)
        0 ~ 1 범위
        """
        text = (title + " " + summary).lower()
        boost = sum(0.08 for kw in _IMPORTANCE_BOOST if kw in text)

        # 제목 길이 (짧고 임팩트 있는 제목 = 중요)
        title_len_score = min(len(title) / 80, 1.0) * 0.15

        # 숫자 포함 (구체적 수치 = 중요도 높음)
        has_numbers = 0.1 if re.search(r'\d+\.?\d*[%억조만]?', title) else 0.0

        # 느낌표/긴급성 표현
        urgency = 0.1 if any(w in title.lower() for w in
                             ["긴급", "속보", "breaking", "urgent", "!"]) else 0.0

        total = 0.3 + boost + title_len_score + has_numbers + urgency
        return round(min(total, 1.0), 4)

    # ──────────────────────────────────────────────────────────────────
    # 키워드 추출 (TF-IDF 근사)
    # ──────────────────────────────────────────────────────────────────

    def _extract_keywords_tfidf(self, title: str, summary: str,
                                 top_k: int = 8) -> list[str]:
        """
        단순 TF-IDF 근사:
        - 제목 단어에 가중치 2× (title boost)
        - 한글 2글자+ / 영어 3글자+ 단어만
        - 불용어 제거
        """
        stopwords = {
            "있다", "하다", "이다", "없다", "되다", "관련", "대한", "따른",
            "위한", "통해", "에서", "으로", "에게", "까지", "부터",
            "the", "and", "for", "with", "that", "this", "from",
            "has", "was", "are", "will", "but", "its", "said",
        }

        def tokenize(text: str) -> list[str]:
            return re.findall(r"[가-힣]{2,}|[a-zA-Z]{3,}", text.lower())

        # TF 계산 (제목 2배 가중)
        tf: dict[str, float] = {}
        for w in tokenize(title):
            if w not in stopwords:
                tf[w] = tf.get(w, 0) + 2.0
        for w in tokenize(summary):
            if w not in stopwords:
                tf[w] = tf.get(w, 0) + 1.0

        # IDF 근사: 중요 뉴스 키워드는 IDF 부스트
        high_idf = set(w.lower() for w in _IMPORTANCE_BOOST)
        scored = {w: score * (1.5 if w in high_idf else 1.0)
                  for w, score in tf.items()}

        ranked = sorted(scored, key=lambda w: scored[w], reverse=True)
        return ranked[:top_k]


# 싱글톤 (모듈 레벨 캐시)
_default_analyzer: Optional[NLPAnalyzer] = None


def get_analyzer(prefer_finbert: bool = False) -> NLPAnalyzer:
    """기본 분석기 인스턴스 반환 (싱글톤)"""
    global _default_analyzer
    if _default_analyzer is None:
        _default_analyzer = NLPAnalyzer(prefer_finbert=prefer_finbert)
    return _default_analyzer
