# features/external_env/feature_engineer.py — 수치화 + 모델 입력 변환
"""
StructuredEvent 리스트 → 32차원 모델 입력 피처 벡터

기존 macro_features.py (32D) 와 동일한 차원을 유지하여
기존 학습된 모델과의 하위 호환성을 보장한다.

확장 모드 (extended=True) 시 64차원 벡터 출력.
"""
from __future__ import annotations
import math
from datetime import datetime
from typing import Optional

import numpy as np

from .event_structure import (
    EventCategory, ImpactDirection, CATEGORY_WEIGHTS, StructuredEvent,
)
from .accumulator import EventAccumulator, ACCUMULATOR_FEATURE_DIM

# 표준 피처 차원 (기존 모델과 호환)
EXTERNAL_FEATURE_DIM    = 32
EXTENDED_FEATURE_DIM    = 64


class ExternalEnvFeatureEngineer:
    """
    이벤트 스트림 → 모델 입력 피처 벡터 변환기.

    사용법:
        eng = ExternalEnvFeatureEngineer()
        eng.ingest(structured_events)
        feat32 = eng.get_features()      # 32D (기존 모델 호환)
        feat64 = eng.get_features(extended=True)  # 64D (확장 모델)
    """

    def __init__(self, cache_file: Optional[str] = None):
        self._accumulator = EventAccumulator(
            max_events=500, cache_file=cache_file
        )
        # 카테고리 가중치 (자기교정으로 업데이트 가능)
        self._cat_weights: dict[EventCategory, float] = dict(CATEGORY_WEIGHTS)

    # ──────────────────────────────────────────────────────────────────
    # 공개 API
    # ──────────────────────────────────────────────────────────────────

    def ingest(self, events: list[StructuredEvent]):
        """이벤트 목록을 누산기에 추가"""
        self._accumulator.push_many(events)

    def ingest_one(self, event: StructuredEvent):
        self._accumulator.push(event)

    def get_features(self, as_of: Optional[datetime] = None,
                     extended: bool = False) -> np.ndarray:
        """
        피처 벡터 반환.
        extended=False → 32D  (기존 MacroEncoder 호환)
        extended=True  → 64D  (새 확장 모델용)
        """
        now  = as_of or datetime.now()
        acc  = self._accumulator.compute_features(now)
        cats = self._accumulator.compute_category_scores(now)

        vec32 = self._build_32d(acc, cats, now)

        if not extended:
            return np.array(vec32, dtype=np.float32)

        vec32_ext = self._build_32d_extended(acc, cats, now)
        return np.array(vec32 + vec32_ext, dtype=np.float32)

    def get_recent_events(self, hours: float = 24) -> list[StructuredEvent]:
        return self._accumulator.get_recent(hours)

    def get_category_scores(self) -> dict[EventCategory, float]:
        return self._accumulator.compute_category_scores()

    def update_category_weight(self, cat: EventCategory, weight: float):
        """자기교정: 카테고리 가중치 업데이트"""
        self._cat_weights[cat] = max(0.1, min(weight, 3.0))

    # ──────────────────────────────────────────────────────────────────
    # 32차원 피처 구성 (기존 모델 호환 레이아웃)
    # ──────────────────────────────────────────────────────────────────

    def _build_32d(self, acc: list[float], cats: dict[EventCategory, float],
                   now: datetime) -> list[float]:
        """
        32차원 배치:
          [0]     전체 외부환경 점수 (weighted sum)
          [1]     호재 강도
          [2]     악재 강도
          [3]     불확실성 지수
          [4-7]   통화정책/거시/지정학/기업 이벤트 점수 (4개 핵심 카테고리)
          [8-11]  산업/정부/수급/시장이벤트 점수
          [12-15] 기술/원자재/금융시장/센티먼트 점수
          [16-19] 1d/3d/7d/14d 이벤트 밀도
          [20-23] 1d/3d/7d/14d 평균 중요도
          [24]    센티먼트 트렌드 (최근 3일 변화)
          [25]    최강 이벤트 점수
          [26]    이벤트 다양성
          [27]    신뢰도 평균
          [28-31] 예비 (0.0)
        """
        v = [0.0] * 32

        # [0-3] 기본 지표
        v[0] = acc[0]   # 전체 점수
        v[1] = acc[1]   # 호재
        v[2] = acc[2]   # 악재
        v[3] = acc[3]   # 불확실성

        # [4-15] 12개 카테고리 점수 (CATEGORY_WEIGHTS 적용)
        cat_order = list(EventCategory)
        for i, cat in enumerate(cat_order):
            raw = cats.get(cat, 0.0)
            weighted = raw * self._cat_weights.get(cat, 1.0)
            v[4 + i] = _clip(weighted, -2.0, 2.0)

        # [16-27] accumulator 나머지 피처
        for i in range(16, 28):
            v[i] = acc[i]

        return v

    def _build_32d_extended(self, acc: list[float],
                             cats: dict[EventCategory, float],
                             now: datetime) -> list[float]:
        """
        확장 32차원 (총 64D의 뒷쪽 32개):
          [0-7]   섹터별 점수 (TECH/FINANCE/ENERGY/HEALTH/CONSUMER/INDUSTRY + 2예비)
          [8-11]  이벤트 지속기간 분포 (단기/중기/장기 비율 + 활성비율)
          [12-15] 방향별 이벤트 수 (호재수/악재수/중립수/총수) 정규화
          [16-19] 핵심 이벤트 유형 플래그 (RATE/WAR/EARNINGS/CPI)
          [20-23] 시나리오 위험 지수 (극단 이벤트 탐지)
          [24-31] 예비
        """
        v = [0.0] * 32

        events = self._accumulator.get_recent(hours=72)

        # 섹터 점수
        sector_idx = {"TECH": 0, "FINANCE": 1, "ENERGY": 2,
                      "HEALTH": 3, "CONSUMER": 4, "INDUSTRY": 5}
        for evt in events:
            dt = max((now - evt.timestamp).total_seconds() / 86400.0, 0)
            decay = math.exp(-0.3 * dt)
            for sec in evt.target_sectors:
                idx = sector_idx.get(sec)
                if idx is not None:
                    v[idx] += evt.external_score * decay
        for i in range(6):
            v[i] = _clip(v[i], -2.0, 2.0)

        # 지속기간 분포
        from .event_structure import EventDuration
        n = max(len(events), 1)
        v[8]  = sum(1 for e in events if e.duration == EventDuration.SHORT) / n
        v[9]  = sum(1 for e in events if e.duration == EventDuration.MID)   / n
        v[10] = sum(1 for e in events if e.duration == EventDuration.LONG)  / n
        v[11] = min(n / 20.0, 1.0)  # 활성 이벤트 비율

        # 방향별 카운트
        n_bull = sum(1 for e in events if e.impact_direction == ImpactDirection.BULLISH)
        n_bear = sum(1 for e in events if e.impact_direction == ImpactDirection.BEARISH)
        n_neu  = sum(1 for e in events if e.impact_direction == ImpactDirection.NEUTRAL)
        v[12]  = n_bull / n
        v[13]  = n_bear / n
        v[14]  = n_neu  / n
        v[15]  = min(n  / 30.0, 1.0)

        # 핵심 이벤트 플래그 (최근 24h)
        recent24 = self._accumulator.get_recent(24)
        types = {e.event_type for e in recent24}
        v[16] = 1.0 if "RATE_DECISION" in types else 0.0
        v[17] = 1.0 if "WAR"   in types or "SANCTION" in types else 0.0
        v[18] = 1.0 if "EARNINGS" in types else 0.0
        v[19] = 1.0 if "CPI"   in types or "GDP" in types else 0.0

        # 극단 이벤트 위험 지수
        extreme = [e for e in events if abs(e.external_score) > 0.6]
        v[20] = min(len(extreme) / 5.0, 1.0)
        v[21] = max((e.external_score for e in extreme), default=0.0)
        v[22] = min((e.external_score for e in extreme), default=0.0)
        v[23] = (sum(e.importance for e in extreme) / len(extreme)
                 if extreme else 0.0)

        return v

    # ──────────────────────────────────────────────────────────────────
    # 시나리오 분석
    # ──────────────────────────────────────────────────────────────────

    def simulate_scenario(self, hypothetical_events: list[StructuredEvent],
                          base_return_pct: float) -> dict:
        """
        가상 이벤트를 추가했을 때 예측 수익률 변화 시뮬레이션.
        base_return_pct: 현재 예측 기본 수익률 (%)
        """
        extra_score = sum(e.compute_score() for e in hypothetical_events)
        # 간단한 선형 영향 모델
        impact = extra_score * 2.0   # +1점 → +2% 이동
        new_return = base_return_pct + impact
        direction = "UP" if new_return > 0 else "DOWN"
        return {
            "base_return":    round(base_return_pct, 4),
            "event_impact":   round(impact, 4),
            "simulated_return": round(new_return, 4),
            "direction":      direction,
            "events_added":   len(hypothetical_events),
        }


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))
