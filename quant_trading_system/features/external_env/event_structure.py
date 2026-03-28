# features/external_env/event_structure.py — 구조화 이벤트 정의
"""
모든 외부환경 뉴스/이벤트를 표준 구조체로 변환한다.

StructuredEvent는 파이프라인 전체의 공통 언어(lingua franca)다.
뉴스 fetcher → categorizer → nlp → feature_engineer 모두 이 구조를 사용.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Optional


# ──────────────────────────────────────────────────────────────────────
# 12개 카테고리 열거형
# ──────────────────────────────────────────────────────────────────────

class EventCategory(Enum):
    MACRO           = "Macro"           # 거시경제 (GDP, 고용, 인플레이션)
    MONETARY_POLICY = "MonetaryPolicy"  # 통화정책 (금리, 양적완화)
    GEOPOLITICS     = "Geopolitics"     # 전쟁·외교·정치 리스크
    INDUSTRY        = "Industry"        # 산업/섹터 동향
    CORPORATE       = "Corporate"       # 기업 이벤트 (실적, M&A, CEO)
    GOVERNMENT      = "Government"      # 정책·규제·세금
    FLOW            = "Flow"            # 수급 (기관/외국인 매매, ETF)
    MARKET_EVENT    = "MarketEvent"     # 시장 이벤트 (서킷브레이커, IPO)
    TECHNOLOGY      = "Technology"      # 기술 트렌드 (AI, 반도체, 특허)
    COMMODITY       = "Commodity"       # 원자재·환율 (유가, 달러, 금)
    FINANCIAL_MKT   = "FinancialMkt"    # 금융시장 (채권, 파생상품)
    SENTIMENT       = "Sentiment"       # 시장 심리 (공포·탐욕, 투자심리)


# 카테고리별 기본 가중치 (모델 feature 계산 시 곱함)
CATEGORY_WEIGHTS: dict[EventCategory, float] = {
    EventCategory.MACRO:           1.5,
    EventCategory.MONETARY_POLICY: 1.5,
    EventCategory.GEOPOLITICS:     1.4,
    EventCategory.INDUSTRY:        1.2,
    EventCategory.CORPORATE:       1.2,
    EventCategory.GOVERNMENT:      1.1,
    EventCategory.FLOW:            1.3,
    EventCategory.MARKET_EVENT:    1.1,
    EventCategory.TECHNOLOGY:      1.0,
    EventCategory.COMMODITY:       1.2,
    EventCategory.FINANCIAL_MKT:   1.1,
    EventCategory.SENTIMENT:       0.8,
}


class ImpactDirection(Enum):
    BULLISH  = +1
    BEARISH  = -1
    NEUTRAL  =  0


class EventDuration(Enum):
    SHORT = "short"   # ~3일
    MID   = "mid"     # ~2주
    LONG  = "long"    # ~3개월+


# ──────────────────────────────────────────────────────────────────────
# 구조화 이벤트 데이터클래스
# ──────────────────────────────────────────────────────────────────────

@dataclass
class StructuredEvent:
    """
    뉴스 1건을 완전 수치화한 구조체.

    모든 필드는 모델 입력 또는 시각화에 직접 사용된다.
    """
    # ─── 식별 ─────────────────────────────────────────────────────────
    event_id:    str = ""
    source_url:  str = ""
    title:       str = ""
    summary:     str = ""
    timestamp:   datetime = field(default_factory=datetime.now)

    # ─── 분류 ─────────────────────────────────────────────────────────
    categories:    list[EventCategory] = field(default_factory=list)
    primary_cat:   Optional[EventCategory] = None
    event_type:    str = ""           # 세부 유형 (FOMC/RATE/WAR/EARNINGS 등)

    # ─── 영향 수치 ────────────────────────────────────────────────────
    impact_direction: ImpactDirection = ImpactDirection.NEUTRAL
    impact_strength:  float = 0.0    # 0~1 (강도)
    confidence:       float = 0.5    # 0~1 (분류 신뢰도)

    # ─── 적용 범위 ────────────────────────────────────────────────────
    target_market:  str = "KR"       # KR / US / GLOBAL
    target_sectors: list[str] = field(default_factory=list)
    target_tickers: list[str] = field(default_factory=list)
    duration:       EventDuration = EventDuration.SHORT

    # ─── NLP 결과 ────────────────────────────────────────────────────
    keywords:        list[str] = field(default_factory=list)
    sentiment_score: float = 0.0    # -1 ~ +1
    importance:      float = 0.5    # 0~1

    # ─── 파생 Feature ────────────────────────────────────────────────
    external_score:  float = 0.0    # = direction × strength × confidence × weight

    # ──────────────────────────────────────────────────────────────────

    def compute_score(self) -> float:
        """
        외부환경 점수 계산.
        external_score = direction × strength × confidence × category_weight
        """
        weight = max(
            (CATEGORY_WEIGHTS.get(c, 1.0) for c in self.categories),
            default=1.0,
        )
        score = (self.impact_direction.value
                 * self.impact_strength
                 * self.confidence
                 * weight)
        self.external_score = float(score)
        return self.external_score

    def to_dict(self) -> dict:
        return {
            "event_id":        self.event_id,
            "title":           self.title,
            "timestamp":       self.timestamp.isoformat(),
            "categories":      [c.value for c in self.categories],
            "primary_cat":     self.primary_cat.value if self.primary_cat else None,
            "event_type":      self.event_type,
            "impact_direction": self.impact_direction.value,
            "impact_strength": self.impact_strength,
            "confidence":      self.confidence,
            "target_market":   self.target_market,
            "target_sectors":  self.target_sectors,
            "duration":        self.duration.value,
            "keywords":        self.keywords,
            "sentiment_score": self.sentiment_score,
            "importance":      self.importance,
            "external_score":  self.external_score,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "StructuredEvent":
        evt = cls()
        evt.event_id        = d.get("event_id", "")
        evt.title           = d.get("title", "")
        try:
            evt.timestamp   = datetime.fromisoformat(d.get("timestamp", ""))
        except Exception:
            evt.timestamp   = datetime.now()
        evt.categories      = [EventCategory(v) for v in d.get("categories", [])
                                if v in {e.value for e in EventCategory}]
        pc = d.get("primary_cat")
        evt.primary_cat     = EventCategory(pc) if pc else None
        evt.event_type      = d.get("event_type", "")
        evt.impact_direction = ImpactDirection(d.get("impact_direction", 0))
        evt.impact_strength  = float(d.get("impact_strength", 0))
        evt.confidence       = float(d.get("confidence", 0.5))
        evt.target_sectors   = d.get("target_sectors", [])
        evt.duration         = EventDuration(d.get("duration", "short"))
        evt.keywords         = d.get("keywords", [])
        evt.sentiment_score  = float(d.get("sentiment_score", 0))
        evt.importance       = float(d.get("importance", 0.5))
        evt.external_score   = float(d.get("external_score", 0))
        return evt
