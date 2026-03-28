# features/external_env/categorizer.py — 12카테고리 자동 분류기
"""
뉴스 텍스트를 12개 카테고리로 자동 분류한다.
다중 카테고리 매핑 지원 (한 뉴스가 여러 카테고리에 속할 수 있음).

분류 방식:
  1차: 키워드 규칙 기반 (빠름, 항상 동작)
  2차: 선택적 ML 보정 (transformers 설치 시 활성화)
"""
from __future__ import annotations
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from .event_structure import (
    EventCategory, ImpactDirection, EventDuration, StructuredEvent,
)

# ──────────────────────────────────────────────────────────────────────
# 카테고리별 키워드 사전 (한국어 + 영어)
# ──────────────────────────────────────────────────────────────────────

_CAT_KEYWORDS: dict[EventCategory, list[str]] = {
    EventCategory.MACRO: [
        "gdp", "경제성장", "성장률", "소비자물가", "cpi", "pce", "인플레",
        "실업", "고용", "취업", "경기", "침체", "recession", "inflation",
        "deflation", "소매판매", "무역수지", "경상수지", "pmi", "ism",
    ],
    EventCategory.MONETARY_POLICY: [
        "금리", "기준금리", "금리인상", "금리인하", "연준", "fed", "fomc",
        "연방준비", "한국은행", "boe", "ecb", "boj", "통화정책", "양적완화",
        "qe", "qt", "테이퍼링", "피벗", "pivot", "rate hike", "rate cut",
        "interest rate", "hawkish", "dovish", "매파", "비둘기",
    ],
    EventCategory.GEOPOLITICS: [
        "전쟁", "war", "conflict", "침공", "군사", "지정학", "geopolit",
        "제재", "sanction", "북한", "중국", "대만", "러시아", "우크라이나",
        "iran", "이란", "핵", "nuclear", "외교", "summit", "무역전쟁",
        "trade war", "tariff", "관세", "긴장", "tension",
    ],
    EventCategory.INDUSTRY: [
        "반도체", "semiconductor", "chip", "배터리", "전기차", "ev",
        "바이오", "제약", "pharma", "헬스케어", "healthcare", "it",
        "소프트웨어", "하드웨어", "통신", "telecom", "에너지", "정유",
        "철강", "화학", "섬유", "게임", "엔터", "금융", "은행", "보험",
        "sector", "industry", "산업",
    ],
    EventCategory.CORPORATE: [
        "실적", "earnings", "revenue", "영업이익", "당기순이익", "eps",
        "가이던스", "guidance", "배당", "dividend", "자사주", "buyback",
        "합병", "인수", "m&a", "merger", "acquisition", "분할", "상장",
        "ipo", "ceo", "경영진", "대표이사", "공시", "disclosure",
    ],
    EventCategory.GOVERNMENT: [
        "정부", "government", "규제", "regulation", "법", "법안", "policy",
        "정책", "세금", "tax", "공정위", "sec", "감독", "공정거래",
        "지원", "subsidy", "보조금", "인허가", "승인", "approval",
    ],
    EventCategory.FLOW: [
        "외국인", "기관", "수급", "매수", "매도", "순매수", "순매도",
        "etf", "펀드", "fund", "flow", "포지션", "position",
        "공매도", "short selling", "대차", "옵션만기", "선물",
    ],
    EventCategory.MARKET_EVENT: [
        "서킷브레이커", "circuit breaker", "vix", "변동성", "volatility",
        "ipo", "상장", "시장", "마감", "개장", "holiday", "거래정지",
        "서스펜션", "suspension", "리밸런싱", "rebalancing", "msci",
        "ftse", "지수편입", "편출",
    ],
    EventCategory.TECHNOLOGY: [
        "ai", "인공지능", "머신러닝", "딥러닝", "chatgpt", "gpt",
        "반도체", "chip", "gpu", "nvidia", "특허", "patent",
        "혁신", "innovation", "5g", "6g", "자율주행", "autonomous",
        "로봇", "robot", "양자컴퓨팅", "quantum", "블록체인", "blockchain",
    ],
    EventCategory.COMMODITY: [
        "유가", "원유", "oil", "wti", "brent", "가스", "gas",
        "금", "gold", "은", "silver", "구리", "copper", "철광석",
        "달러", "dollar", "환율", "fx", "원달러", "위안", "yuan",
        "엔", "yen", "유로", "euro", "원자재", "commodity",
    ],
    EventCategory.FINANCIAL_MKT: [
        "채권", "bond", "국채", "treasury", "금리", "yield", "스프레드",
        "spread", "신용", "credit", "cds", "파생", "derivative",
        "선물", "futures", "옵션", "option", "레버리지", "leverage",
        "리스크", "risk", "hedge", "헤지",
    ],
    EventCategory.SENTIMENT: [
        "공포", "fear", "탐욕", "greed", "심리", "sentiment",
        "투자심리", "과열", "침체", "낙관", "비관", "optimism", "pessimism",
        "공황", "panic", "급락", "급등", "폭락", "폭등",
        "과매도", "과매수", "oversold", "overbought",
    ],
}

# ──────────────────────────────────────────────────────────────────────
# 이벤트 유형 세부 분류 사전
# ──────────────────────────────────────────────────────────────────────

_EVENT_TYPE_PATTERNS: list[tuple[str, str]] = [
    (r"fomc|연방준비위|기준금리.*결정|금리.*인상|금리.*인하", "RATE_DECISION"),
    (r"cpi|소비자물가|pce|인플레이션", "CPI"),
    (r"gdp|국내총생산|경제성장률", "GDP"),
    (r"고용|실업|비농업|nfp|payroll", "EMPLOYMENT"),
    (r"전쟁|침공|군사|폭격", "WAR"),
    (r"제재|sanction", "SANCTION"),
    (r"실적|earnings|revenue|영업이익", "EARNINGS"),
    (r"합병|인수|m&a|merger|acquisition", "MA"),
    (r"ipo|상장|공모", "IPO"),
    (r"배당|dividend", "DIVIDEND"),
    (r"규제|regulation|법안|policy", "REGULATION"),
    (r"유가|oil|wti|brent", "OIL"),
    (r"달러|환율|원달러|dollar.*index|dxy", "FX"),
    (r"공매도|short|vix|변동성", "VOLATILITY"),
    (r"ai|인공지능|chatgpt", "AI_TECH"),
    (r"반도체|chip|semiconductor|nvidia", "SEMICONDUCTOR"),
]

# ──────────────────────────────────────────────────────────────────────
# 방향/강도 키워드 사전
# ──────────────────────────────────────────────────────────────────────

_BULLISH_STRONG  = ["급등", "폭등", "사상최고", "강세", "호황", "beat", "예상상회",
                    "record high", "surge", "soar", "rally", "boom"]
_BULLISH_MILD    = ["상승", "올라", "긍정", "호재", "개선", "증가", "성장",
                    "rise", "gain", "positive", "improve", "grow"]
_BEARISH_STRONG  = ["급락", "폭락", "사상최저", "패닉", "위기", "crash", "collapse",
                    "plunge", "tumble", "plummet", "crisis", "default"]
_BEARISH_MILD    = ["하락", "내려", "부정", "악재", "악화", "감소", "침체",
                    "fall", "drop", "decline", "negative", "worsen"]
_UNCERTAINTY     = ["불확실", "변동성", "우려", "위험", "리스크", "concern",
                    "worry", "uncertainty", "risk", "volatile"]

# 지속기간 키워드
_DURATION_LONG   = ["장기", "구조적", "트렌드", "long-term", "structural", "decade"]
_DURATION_MID    = ["중기", "분기", "quarter", "mid-term", "weeks", "month"]


# ──────────────────────────────────────────────────────────────────────
# Categorizer 클래스
# ──────────────────────────────────────────────────────────────────────

class NewsEventCategorizer:
    """
    뉴스 텍스트 → StructuredEvent 변환기.

    classify(title, summary) → StructuredEvent
    """

    def __init__(self):
        # 컴파일된 패턴 캐싱
        self._type_patterns = [
            (re.compile(pat, re.IGNORECASE), etype)
            for pat, etype in _EVENT_TYPE_PATTERNS
        ]

    def classify(self, title: str, summary: str = "",
                 url: str = "", ts=None) -> StructuredEvent:
        """뉴스 텍스트 하나를 StructuredEvent로 변환"""
        from datetime import datetime
        import hashlib

        text = (title + " " + summary).lower()

        evt = StructuredEvent()
        evt.title       = title
        evt.summary     = summary
        evt.source_url  = url
        evt.timestamp   = ts or datetime.now()
        evt.event_id    = hashlib.md5(
            (title + evt.timestamp.isoformat()).encode()).hexdigest()[:12]

        # 1. 카테고리 분류
        evt.categories   = self._classify_categories(text)
        evt.primary_cat  = evt.categories[0] if evt.categories else None

        # 2. 이벤트 유형
        evt.event_type   = self._classify_event_type(text)

        # 3. 방향 + 강도
        direction, strength = self._classify_impact(text)
        evt.impact_direction = direction
        evt.impact_strength  = strength

        # 4. 신뢰도 (카테고리 수 + 이벤트 유형 존재 여부로 휴리스틱)
        n_cats = len(evt.categories)
        has_type = 1.0 if evt.event_type else 0.0
        evt.confidence = min(0.5 + 0.1 * n_cats + 0.2 * has_type, 0.95)

        # 5. 지속기간
        evt.duration = self._classify_duration(text)

        # 6. 섹터 관련성
        evt.target_sectors = self._classify_sectors(text)

        # 7. 키워드 (상위 6개)
        evt.keywords = self._extract_keywords(title, summary)

        # 8. 점수 계산
        evt.compute_score()

        return evt

    # ──────────────────────────────────────────────────────────────────

    def _classify_categories(self, text: str) -> list[EventCategory]:
        """텍스트에서 해당하는 모든 카테고리 반환 (점수 내림차순)"""
        scores: dict[EventCategory, int] = {}
        for cat, keywords in _CAT_KEYWORDS.items():
            hit = sum(1 for kw in keywords if kw in text)
            if hit:
                scores[cat] = hit
        # 점수 높은 순, 최대 4개
        ranked = sorted(scores, key=lambda c: scores[c], reverse=True)
        return ranked[:4]

    def _classify_event_type(self, text: str) -> str:
        for pattern, etype in self._type_patterns:
            if pattern.search(text):
                return etype
        return "GENERAL"

    def _classify_impact(self, text: str) -> tuple[ImpactDirection, float]:
        bs  = sum(1 for w in _BULLISH_STRONG if w in text)
        bm  = sum(1 for w in _BULLISH_MILD   if w in text)
        brs = sum(1 for w in _BEARISH_STRONG  if w in text)
        brm = sum(1 for w in _BEARISH_MILD    if w in text)
        unc = sum(1 for w in _UNCERTAINTY     if w in text)

        bull_score = bs * 2 + bm
        bear_score = brs * 2 + brm

        if bull_score == 0 and bear_score == 0:
            return ImpactDirection.NEUTRAL, 0.1 + unc * 0.05

        total = bull_score + bear_score
        if bull_score > bear_score:
            strength = min((bull_score / total) * 0.8 + bs * 0.1, 1.0)
            return ImpactDirection.BULLISH, round(strength, 3)
        elif bear_score > bull_score:
            strength = min((bear_score / total) * 0.8 + brs * 0.1, 1.0)
            return ImpactDirection.BEARISH, round(strength, 3)
        else:
            return ImpactDirection.NEUTRAL, 0.2

    def _classify_duration(self, text: str) -> EventDuration:
        if any(w in text for w in _DURATION_LONG):
            return EventDuration.LONG
        if any(w in text for w in _DURATION_MID):
            return EventDuration.MID
        return EventDuration.SHORT

    def _classify_sectors(self, text: str) -> list[str]:
        sector_map = {
            "TECH":     ["반도체", "it", "software", "하드웨어", "통신", "ai"],
            "FINANCE":  ["금융", "은행", "보험", "증권", "카드"],
            "ENERGY":   ["에너지", "정유", "석유", "gas", "원유"],
            "HEALTH":   ["바이오", "제약", "헬스", "healthcare", "pharma"],
            "CONSUMER": ["소비", "retail", "consumer", "음식", "패션"],
            "INDUSTRY": ["철강", "화학", "조선", "건설", "자동차"],
        }
        result = []
        for sector, kws in sector_map.items():
            if any(k in text for k in kws):
                result.append(sector)
        return result

    def _extract_keywords(self, title: str, summary: str) -> list[str]:
        """제목+요약에서 중요 키워드 추출 (상위 6개)"""
        import re as re_mod
        combined = title + " " + summary
        # 숫자/특수문자 제거, 단어 분리
        words = re_mod.findall(r"[가-힣]{2,}|[a-zA-Z]{3,}", combined)
        # 불용어
        stopwords = {"있다", "하다", "이다", "없다", "되다", "관련", "대한",
                     "따르", "위한", "the", "and", "for", "with", "that",
                     "this", "from", "has", "was", "are", "will"}
        seen = set()
        result = []
        for w in words:
            lw = w.lower()
            if lw not in stopwords and lw not in seen:
                seen.add(lw)
                result.append(w)
            if len(result) >= 6:
                break
        return result
