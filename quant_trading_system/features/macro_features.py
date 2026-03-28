# features/macro_features.py — 외부환경 정형 피처 생성기
"""
뉴스/이벤트 → 모델 입력 정형 피처 변환 파이프라인

설계 원칙:
  - 텍스트를 그대로 모델에 입력하지 않음 → 반드시 수치 피처로 변환
  - VADER 없어도 rule-based로 동작 (선택적 고급 감성 분석)
  - 32차원 고정 출력 벡터 (MacroEncoder 입력 차원과 일치)

피처 구조 (MACRO_FEATURE_DIM = 32):
  [0-2]   감성 점수:  positive, negative, uncertainty                  (3)
  [3]     전체 감성:  -1(최악) ~ +1(최고)                               (1)
  [4-12]  이벤트 플래그 (0/1): FOMC, RATE, CPI, WAR, POLICY,           (9)
                               EARNINGS, SUPPLY, FX, COMMODITY
  [13-21] 이벤트 강도 (0~1):  위와 동일 카테고리                          (9)
  [22-24] 래그 피처:  1일/3일/7일 이벤트 밀도 변화율                      (3)
  [25-28] 섹터 관련도: 금융, 에너지, 반도체/기술, 헬스케어                 (4)
  [29-31] 시장 상태:  공포↔탐욕 지수, 총 이벤트 수(정규화), 정보 신선도    (3)
"""
from __future__ import annotations

import datetime
import logging
import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("quant.macro")

# ──────────────────────────────────────────────────────────────────────────────
# 상수
# ──────────────────────────────────────────────────────────────────────────────

MACRO_FEATURE_DIM: int = 32   # MacroEncoder 입력 차원과 반드시 일치

# 이벤트 타입 인덱스 (피처 벡터 [4~12], [13~21])
_EVT_IDX = {
    "FOMC":      0,
    "RATE":      1,
    "CPI":       2,
    "WAR":       3,
    "POLICY":    4,
    "EARNINGS":  5,
    "SUPPLY":    6,
    "FX":        7,
    "COMMODITY": 8,
}

# 섹터 인덱스 (피처 벡터 [25~28])
_SEC_IDX = {
    "FINANCE":   0,
    "ENERGY":    1,
    "TECH":      2,
    "HEALTH":    3,
}


# ──────────────────────────────────────────────────────────────────────────────
# 열거형 / 데이터 클래스
# ──────────────────────────────────────────────────────────────────────────────

class MacroEventType(str, Enum):
    FOMC       = "FOMC"        # 연준 금리 결정
    RATE       = "RATE"        # 금리 변동 일반
    CPI        = "CPI"         # 물가 지표
    WAR        = "WAR"         # 전쟁/분쟁
    POLICY     = "POLICY"      # 정책/규제
    EARNINGS   = "EARNINGS"    # 기업 실적
    SUPPLY     = "SUPPLY"      # 공급망
    FX         = "FX"          # 환율
    COMMODITY  = "COMMODITY"   # 원자재
    UNKNOWN    = "UNKNOWN"     # 미분류


class SentimentLabel(str, Enum):
    POSITIVE    = "positive"    # 호재
    NEGATIVE    = "negative"    # 악재
    UNCERTAINTY = "uncertainty" # 불확실
    NEUTRAL     = "neutral"     # 중립


@dataclass
class MacroEvent:
    """단일 이벤트 데이터클래스"""
    title:      str
    body:       str
    source:     str
    timestamp:  datetime.datetime
    event_type: MacroEventType  = MacroEventType.UNKNOWN
    sentiment:  SentimentLabel  = SentimentLabel.NEUTRAL
    intensity:  float           = 0.0    # 강도 [0, 1]
    relevance:  float           = 0.5    # 종목/시장 관련도 [0, 1]
    sectors:    List[str]       = field(default_factory=list)
    tags:       List[str]       = field(default_factory=list)
    url:        str             = ""

    @property
    def age_hours(self) -> float:
        """이벤트 발생 후 경과 시간(시간)"""
        now = datetime.datetime.now()
        if self.timestamp.tzinfo is not None:
            import pytz
            now = datetime.datetime.now(pytz.UTC)
        delta = now - self.timestamp
        return max(0.0, delta.total_seconds() / 3600.0)

    @property
    def freshness(self) -> float:
        """신선도 점수 [0,1] — 24시간 지수 감소"""
        return float(np.exp(-self.age_hours / 24.0))


# ──────────────────────────────────────────────────────────────────────────────
# 키워드 사전
# ──────────────────────────────────────────────────────────────────────────────

_KW_POSITIVE = [
    # 한국어
    "상승", "급등", "호재", "개선", "돌파", "강세", "상향", "매수", "성장",
    "증가", "확대", "회복", "개선", "기대", "완화", "반등", "사상최고", "최고치",
    "흑자", "수혜", "기회", "협력", "계약", "수주",
    # 영어
    "rise", "surge", "rally", "beat", "positive", "growth", "gain",
    "record high", "upgrade", "outperform", "strong", "bullish",
    "recovery", "expansion", "profit", "boost", "deal", "win",
]

_KW_NEGATIVE = [
    # 한국어
    "하락", "급락", "악재", "폭락", "위험", "리스크", "하향", "매도", "감소",
    "축소", "손실", "적자", "우려", "위기", "침체", "파산", "분쟁", "전쟁",
    "제재", "금지", "규제", "충격", "쇼크", "불안", "공포", "폭등(물가)",
    # 영어
    "fall", "drop", "crash", "miss", "negative", "decline", "loss",
    "risk", "downgrade", "underperform", "weak", "bearish", "recession",
    "default", "sanction", "ban", "shock", "fear", "concern", "cut",
    "layoff", "war", "conflict", "tension",
]

_KW_UNCERTAINTY = [
    # 한국어
    "불확실", "변동성", "관망", "불투명", "미지수", "혼조",
    "대기", "보류", "검토", "지켜봐야", "예측불허",
    # 영어
    "uncertainty", "volatile", "wait", "unclear", "mixed",
    "pending", "review", "unknown", "unpredictable", "depends",
]

_KW_EVENT: Dict[str, List[str]] = {
    "FOMC":      ["fomc", "federal reserve", "연준", "연방준비", "fed meeting",
                  "powell", "파월", "연방공개시장위원회"],
    "RATE":      ["금리", "interest rate", "rate hike", "rate cut", "기준금리",
                  "금리인상", "금리인하", "boe", "ecb", "한국은행", "기준"],
    "CPI":       ["cpi", "소비자물가", "inflation", "인플레", "물가", "ppi",
                  "deflation", "디플레", "물가지수"],
    "WAR":       ["전쟁", "분쟁", "war", "conflict", "military", "attack",
                  "invasion", "missile", "핵", "nuclear", "러시아", "ukraine",
                  "중동", "이스라엘", "하마스", "팔레스타인", "taiwan", "대만"],
    "POLICY":    ["정책", "policy", "규제", "regulation", "법안", "입법",
                  "세금", "tax", "tariff", "관세", "subsidy", "보조금",
                  "대통령", "정부", "국회", "제재"],
    "EARNINGS":  ["실적", "earnings", "revenue", "profit", "eps", "어닝",
                  "순이익", "매출", "분기", "연간실적", "guidance"],
    "SUPPLY":    ["공급망", "supply chain", "반도체", "semiconductor", "칩",
                  "chip", "부품", "재고", "logistics", "물류", "항만"],
    "FX":        ["환율", "exchange rate", "달러", "dollar", "usd", "eur",
                  "원달러", "엔화", "위안", "sterling", "currency"],
    "COMMODITY": ["원자재", "oil", "유가", "crude", "gold", "금값", "은",
                  "구리", "copper", "철강", "steel", "곡물", "corn",
                  "wheat", "밀", "natural gas", "천연가스", "배터리"],
}

_KW_SECTOR: Dict[str, List[str]] = {
    "FINANCE": ["은행", "증권", "보험", "금융", "bank", "finance", "credit",
                "금리", "대출", "loan", "펀드", "fund"],
    "ENERGY":  ["유가", "oil", "에너지", "energy", "정유", "refinery",
                "가스", "gas", "발전", "태양광", "solar", "풍력"],
    "TECH":    ["반도체", "semiconductor", "it", "기술", "tech", "ai", "인공지능",
                "software", "소프트웨어", "chip", "칩", "samsung", "삼성",
                "sk hynix", "하이닉스", "nvidia", "tsmc"],
    "HEALTH":  ["제약", "바이오", "pharma", "biotech", "healthcare", "의료",
                "임상", "clinical", "fda", "신약", "백신", "vaccine"],
}


# ──────────────────────────────────────────────────────────────────────────────
# 감성 점수 계산기
# ──────────────────────────────────────────────────────────────────────────────

class SentimentScorer:
    """
    키워드 기반 감성 점수 계산기

    우선순위:
    1. VADER (설치되어 있으면 사용 — 더 정확)
    2. Rule-based 키워드 매칭 (항상 사용 가능)
    """

    def __init__(self):
        self._vader = None
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self._vader = SentimentIntensityAnalyzer()
            logger.debug("VADER 감성 분석기 활성화")
        except ImportError:
            pass

    def score(self, text: str) -> Tuple[float, float, float, float]:
        """
        Returns: (positive, negative, uncertainty, overall)
          - positive:    [0, 1] 호재 강도
          - negative:    [0, 1] 악재 강도
          - uncertainty: [0, 1] 불확실성
          - overall:     [-1, 1] 종합 감성 (VADER compound 또는 pos-neg)
        """
        if not text:
            return 0.0, 0.0, 0.0, 0.0

        txt_lower = text.lower()

        # ── Rule-based ────────────────────────────────────────────────────
        pos_hits = sum(1 for kw in _KW_POSITIVE    if kw in txt_lower)
        neg_hits = sum(1 for kw in _KW_NEGATIVE    if kw in txt_lower)
        unc_hits = sum(1 for kw in _KW_UNCERTAINTY if kw in txt_lower)

        total = max(pos_hits + neg_hits + unc_hits, 1)
        pos_rb = min(pos_hits / total, 1.0)
        neg_rb = min(neg_hits / total, 1.0)
        unc_rb = min(unc_hits / max(total, 3), 1.0)

        # ── VADER (영어 텍스트에서 더 정확) ──────────────────────────────
        if self._vader:
            scores = self._vader.polarity_scores(text)
            pos  = max(pos_rb, scores["pos"])
            neg  = max(neg_rb, scores["neg"])
            unc  = unc_rb
            over = scores["compound"]
        else:
            pos  = pos_rb
            neg  = neg_rb
            unc  = unc_rb
            over = float(np.clip(pos - neg, -1.0, 1.0))

        return float(pos), float(neg), float(unc), float(over)


# ──────────────────────────────────────────────────────────────────────────────
# 이벤트 분류기
# ──────────────────────────────────────────────────────────────────────────────

class EventClassifier:
    """
    텍스트 → 이벤트 타입 + 섹터 + 강도 분류

    강도 계산:
    - 감성 절댓값 + 관련 키워드 수 + 단어 강조 패턴(급/shock/surge 등) 반영
    """

    _INTENSITY_AMPLIFIERS = [
        "급", "폭", "사상최고", "사상최저", "최악", "충격", "쇼크", "붕괴",
        "surge", "crash", "record", "shock", "plunge", "soar", "extreme",
        "emergency", "crisis", "최고치", "최저치", "급등", "급락",
    ]

    def __init__(self):
        self._scorer = SentimentScorer()

    def classify(self, title: str, body: str = "") -> MacroEvent:
        """단일 이벤트 분류 및 스코어링"""
        text = (title + " " + body).strip()
        txt_lower = text.lower()

        # 감성
        pos, neg, unc, overall = self._scorer.score(text)
        if overall >= 0.1:
            sentiment = SentimentLabel.POSITIVE
        elif overall <= -0.1:
            sentiment = SentimentLabel.NEGATIVE
        elif unc > 0.15:
            sentiment = SentimentLabel.UNCERTAINTY
        else:
            sentiment = SentimentLabel.NEUTRAL

        # 이벤트 타입 (첫 매칭 사용, 여러 개면 강도 높은 것)
        event_type = MacroEventType.UNKNOWN
        best_hits  = 0
        for etype, keywords in _KW_EVENT.items():
            hits = sum(1 for kw in keywords if kw in txt_lower)
            if hits > best_hits:
                best_hits  = hits
                event_type = MacroEventType(etype)

        # 태그 (복수 이벤트 타입 허용)
        tags = []
        for etype, keywords in _KW_EVENT.items():
            if any(kw in txt_lower for kw in keywords):
                tags.append(etype)

        # 섹터 관련도
        sectors = []
        for sec, keywords in _KW_SECTOR.items():
            if any(kw in txt_lower for kw in keywords):
                sectors.append(sec)

        # 강도 계산
        amp_hits  = sum(1 for a in self._INTENSITY_AMPLIFIERS if a in txt_lower)
        intensity = float(np.clip(
            abs(overall) * 0.5 + min(amp_hits * 0.15, 0.4) + min(best_hits * 0.05, 0.1),
            0.0, 1.0
        ))

        # 관련도: 태그가 많을수록 관련도 높음
        relevance = float(np.clip(len(tags) * 0.15 + len(sectors) * 0.10 + 0.3, 0.0, 1.0))

        return MacroEvent(
            title      = title,
            body       = body[:500],  # 최대 500자
            source     = "",
            timestamp  = datetime.datetime.now(),
            event_type = event_type,
            sentiment  = sentiment,
            intensity  = intensity,
            relevance  = relevance,
            sectors    = sectors,
            tags       = tags,
        )


# ──────────────────────────────────────────────────────────────────────────────
# 래그(시간 지연) 피처 계산기
# ──────────────────────────────────────────────────────────────────────────────

class LagFeatureComputer:
    """
    이벤트 이력에서 시간 지연 피처 생성

    래그 피처:
    - 1일/3일/7일 이벤트 밀도 변화율 (최근 밀도 / 이전 밀도)
    - 가격 반응 시차 반영 모델링
    """

    def compute(self,
                events: List[MacroEvent],
                reference_time: Optional[datetime.datetime] = None) -> Tuple[float, float, float]:
        """
        Returns: (lag_1d, lag_3d, lag_7d)
          - lag_Xd: X일 전 대비 이벤트 밀도 변화 (-1 ~ +1)
        """
        if not events:
            return 0.0, 0.0, 0.0

        now = reference_time or datetime.datetime.now()

        def density(hours_from: float, hours_to: float) -> float:
            n = sum(1 for e in events
                    if hours_from <= e.age_hours < hours_to)
            return n / max(hours_to - hours_from, 1.0) * 24  # events/day

        # 1일 래그: 최근 24h 밀도 vs 1~2일전 밀도
        d_now_1   = density(0,  24)
        d_prev_1  = density(24, 48)
        lag_1 = float(np.tanh((d_now_1 - d_prev_1) / max(d_prev_1, 0.1)))

        # 3일 래그
        d_now_3  = density(0,  72)
        d_prev_3 = density(72, 144)
        lag_3 = float(np.tanh((d_now_3 - d_prev_3) / max(d_prev_3, 0.1)))

        # 7일 래그
        d_now_7  = density(0,   168)
        d_prev_7 = density(168, 336)
        lag_7 = float(np.tanh((d_now_7 - d_prev_7) / max(d_prev_7, 0.1)))

        return lag_1, lag_3, lag_7


# ──────────────────────────────────────────────────────────────────────────────
# 메인 피처 빌더
# ──────────────────────────────────────────────────────────────────────────────

class MacroFeatureBuilder:
    """
    이벤트 목록 → 32차원 정형 피처 벡터

    사용법:
        builder = MacroFeatureBuilder()
        feat = builder.build(events)   # np.ndarray, shape=(32,)

    모델 입력:
        feat[np.newaxis, :]  → (1, 32) — 단일 추론용
    """

    def __init__(self):
        self._lag = LagFeatureComputer()

    def build(
        self,
        events: List[MacroEvent],
        symbol: str = "",
        reference_time: Optional[datetime.datetime] = None,
    ) -> np.ndarray:
        """
        Args:
            events:    최근 이벤트 목록 (신선한 것부터)
            symbol:    종목 코드 (섹터 관련도 가중치 조정용, 현재는 미사용)
        Returns:
            feat: float32 배열 (MACRO_FEATURE_DIM,)
        """
        feat = np.zeros(MACRO_FEATURE_DIM, dtype=np.float32)

        if not events:
            return feat

        # ── [0-3] 감성 집계 ───────────────────────────────────────────
        fresh_events = [e for e in events if e.age_hours < 72]  # 3일 이내
        if fresh_events:
            w = np.array([e.freshness * e.relevance for e in fresh_events])
            w_sum = w.sum() + 1e-10

            pos_arr = np.array([1.0 if e.sentiment == SentimentLabel.POSITIVE else 0.0
                                for e in fresh_events])
            neg_arr = np.array([1.0 if e.sentiment == SentimentLabel.NEGATIVE else 0.0
                                for e in fresh_events])
            unc_arr = np.array([1.0 if e.sentiment == SentimentLabel.UNCERTAINTY else 0.0
                                for e in fresh_events])

            feat[0] = float((pos_arr * w).sum() / w_sum)   # positive
            feat[1] = float((neg_arr * w).sum() / w_sum)   # negative
            feat[2] = float((unc_arr * w).sum() / w_sum)   # uncertainty
            feat[3] = float(np.clip(feat[0] - feat[1], -1, 1))  # overall

        # ── [4-12] 이벤트 플래그, [13-21] 이벤트 강도 ───────────────
        for evt in fresh_events:
            if evt.event_type != MacroEventType.UNKNOWN:
                idx = _EVT_IDX.get(evt.event_type.value, -1)
                if idx >= 0:
                    feat[4  + idx] = 1.0                     # 플래그
                    feat[13 + idx] = max(feat[13 + idx],
                                         evt.intensity * evt.freshness)  # 강도 (최대값)
            # 복수 태그도 반영
            for tag in evt.tags:
                tidx = _EVT_IDX.get(tag, -1)
                if tidx >= 0 and feat[4 + tidx] == 0:
                    feat[4  + tidx] = 0.5  # 주 이벤트가 아닌 연관 태그
                    feat[13 + tidx] = max(feat[13 + tidx],
                                          evt.intensity * 0.5 * evt.freshness)

        # ── [22-24] 래그 피처 ─────────────────────────────────────────
        lag1, lag3, lag7 = self._lag.compute(events, reference_time)
        feat[22] = float(np.clip(lag1, -1, 1))
        feat[23] = float(np.clip(lag3, -1, 1))
        feat[24] = float(np.clip(lag7, -1, 1))

        # ── [25-28] 섹터 관련도 ───────────────────────────────────────
        sec_scores: Dict[str, float] = {s: 0.0 for s in _SEC_IDX}
        for evt in fresh_events:
            for sec in evt.sectors:
                sec_scores[sec] = max(sec_scores.get(sec, 0.0),
                                       evt.intensity * evt.freshness)
        for sec, sidx in _SEC_IDX.items():
            feat[25 + sidx] = float(sec_scores.get(sec, 0.0))

        # ── [29] 공포↔탐욕 지수 (−1: 극도 공포, +1: 극도 탐욕) ──────
        n_pos = sum(1 for e in fresh_events if e.sentiment == SentimentLabel.POSITIVE)
        n_neg = sum(1 for e in fresh_events if e.sentiment == SentimentLabel.NEGATIVE)
        fear_greed = (n_pos - n_neg) / max(len(fresh_events), 1)
        feat[29] = float(np.clip(fear_greed, -1, 1))

        # ── [30] 총 이벤트 수 (정규화, 최대 50개 기준) ───────────────
        feat[30] = float(np.clip(len(fresh_events) / 50.0, 0, 1))

        # ── [31] 정보 신선도 (평균 freshness) ─────────────────────────
        if fresh_events:
            feat[31] = float(np.mean([e.freshness for e in fresh_events]))

        # NaN/Inf 방어
        feat = np.nan_to_num(feat, nan=0.0, posinf=1.0, neginf=-1.0)
        return feat

    def build_batch(
        self,
        events_per_sample: List[List[MacroEvent]],
        symbols: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        배치 처리: N개 샘플 × 각 샘플별 이벤트 목록 → (N, 32)

        Args:
            events_per_sample: [events_for_sample_0, events_for_sample_1, ...]
        """
        N = len(events_per_sample)
        result = np.zeros((N, MACRO_FEATURE_DIM), dtype=np.float32)
        for i, evts in enumerate(events_per_sample):
            sym = (symbols[i] if symbols and i < len(symbols) else "")
            result[i] = self.build(evts, symbol=sym)
        return result


# ──────────────────────────────────────────────────────────────────────────────
# 이벤트 기반 실패 유형 분류
# ──────────────────────────────────────────────────────────────────────────────

class MacroFailureType(str, Enum):
    """외생변수 관련 예측 실패 유형"""
    EVENT_MISSED     = "EVENT_MISSED"      # 주요 이벤트 미반영
    EVENT_OVERREACT  = "EVENT_OVERREACT"   # 이벤트 과잉 반응
    EVENT_LAG        = "EVENT_LAG"         # 이벤트 시간 지연 반응 미반영


def classify_macro_failure(
    prediction_error: float,    # 예측오차 (예측-실제)
    events_at_time:  List[MacroEvent],
    high_impact_threshold: float = 0.5,  # 강도 기준
) -> Optional[MacroFailureType]:
    """
    이벤트 존재 여부와 예측 오차를 기반으로 실패 유형 분류.

    Returns: 해당 실패 유형 or None
    """
    if not events_at_time:
        return None

    high_impact = [e for e in events_at_time if e.intensity >= high_impact_threshold]
    if not high_impact:
        return None

    abs_error = abs(prediction_error)
    if abs_error > 0.05:  # 5% 이상 오차
        # 강한 이벤트가 있었는데 크게 빗나갔다 → 미반영
        return MacroFailureType.EVENT_MISSED

    # 오차 방향과 이벤트 감성이 반대 → 과잉 반응
    neg_events = [e for e in high_impact if e.sentiment == SentimentLabel.NEGATIVE]
    if neg_events and prediction_error < -0.02:  # 악재인데 과도하게 하락 예측
        return MacroFailureType.EVENT_OVERREACT

    # 래그 이벤트 (1~3일 전 강한 이벤트) → 지연 반응 누락
    lag_events = [e for e in high_impact if 24 <= e.age_hours <= 72]
    if lag_events and abs_error > 0.02:
        return MacroFailureType.EVENT_LAG

    return None
