# features/external_env/accumulator.py — 시계열 누적 + 시간 감쇠
"""
이벤트 스트림 → 시계열 외부환경 피처 벡터

weighted_score(t) = Σ event_score_i × exp(-λ × Δt_i)

λ 값 (감쇠 속도):
  단기 이벤트: λ = 0.5 (반감기 ≈ 1.4일)
  중기 이벤트: λ = 0.1 (반감기 ≈ 7일)
  장기 이벤트: λ = 0.02 (반감기 ≈ 35일)
"""
from __future__ import annotations
import math
import json
import os
from datetime import datetime, timedelta
from typing import Optional
import logging

from .event_structure import (
    StructuredEvent, EventCategory, EventDuration, CATEGORY_WEIGHTS,
)

logger = logging.getLogger("quant.external_env.accumulator")

# 지속기간별 감쇠 계수 (λ)
_LAMBDA: dict[EventDuration, float] = {
    EventDuration.SHORT: 0.5,
    EventDuration.MID:   0.1,
    EventDuration.LONG:  0.02,
}

# 피처 벡터 차원 수
ACCUMULATOR_FEATURE_DIM = 28


class EventAccumulator:
    """
    이벤트 큐를 관리하며 현재 시점의 피처 벡터를 계산한다.

    피처 벡터 구성 (28차원):
      [0]     전체 가중 점수 (scalar)
      [1]     호재 강도 누적
      [2]     악재 강도 누적
      [3]     불확실성 지수
      [4-15]  카테고리별 누적 점수 (12개)
      [16-19] 최근 N일 이벤트 밀도 (1d/3d/7d/14d)
      [20-23] 최근 N일 평균 중요도
      [24]    최강 단일 이벤트 점수
      [25]    이벤트 다양성 (활성 카테고리 수 / 12)
      [26]    센티먼트 트렌드 (최근 3일 vs 이전 3일 변화)
      [27]    시간 가중 신뢰도 평균
    """

    def __init__(self, max_events: int = 500, cache_file: Optional[str] = None):
        self._events: list[StructuredEvent] = []
        self._max_events = max_events
        self._cache_file = cache_file
        if cache_file:
            self._load_cache()

    # ──────────────────────────────────────────────────────────────────
    # 이벤트 추가
    # ──────────────────────────────────────────────────────────────────

    def push(self, event: StructuredEvent):
        """이벤트 추가 (중복 제거 + 크기 제한)"""
        # 중복 제거 (동일 event_id)
        if any(e.event_id == event.event_id for e in self._events):
            return
        self._events.append(event)
        # 오래된 이벤트 제거 (30일 초과)
        cutoff = datetime.now() - timedelta(days=30)
        self._events = [e for e in self._events if e.timestamp >= cutoff]
        # 크기 제한
        if len(self._events) > self._max_events:
            self._events = sorted(self._events,
                                  key=lambda e: e.timestamp)[-self._max_events:]
        if self._cache_file:
            self._save_cache()

    def push_many(self, events: list[StructuredEvent]):
        for e in events:
            self.push(e)

    # ──────────────────────────────────────────────────────────────────
    # 피처 벡터 계산
    # ──────────────────────────────────────────────────────────────────

    def compute_features(self, as_of: Optional[datetime] = None) -> list[float]:
        """
        현재 시점(as_of) 기준 28차원 피처 벡터 반환.
        이벤트가 없으면 모두 0인 벡터 반환.
        """
        now = as_of or datetime.now()

        if not self._events:
            return [0.0] * ACCUMULATOR_FEATURE_DIM

        vec = [0.0] * ACCUMULATOR_FEATURE_DIM

        # 카테고리 인덱스 매핑 (0~11)
        cat_idx = {cat: i for i, cat in enumerate(EventCategory)}

        total_score = 0.0
        bull_sum    = 0.0
        bear_sum    = 0.0
        uncert_sum  = 0.0
        cat_scores  = [0.0] * 12
        conf_sum    = 0.0
        weight_sum  = 0.0
        max_score   = 0.0

        # 시간 구간 이벤트 카운트
        windows = [1, 3, 7, 14]
        density  = [0] * 4
        imp_sums = [0.0] * 4

        for evt in self._events:
            dt_hours = (now - evt.timestamp).total_seconds() / 3600.0
            if dt_hours < 0:
                continue
            dt_days = dt_hours / 24.0

            lam = _LAMBDA[evt.duration]
            decay = math.exp(-lam * dt_days)

            raw_score = evt.external_score
            w_score   = raw_score * decay

            total_score += w_score
            if raw_score > 0:
                bull_sum += abs(w_score)
            elif raw_score < 0:
                bear_sum += abs(w_score)
            else:
                uncert_sum += evt.impact_strength * decay

            # 카테고리별 누적
            for cat in evt.categories:
                idx = cat_idx.get(cat)
                if idx is not None:
                    cat_scores[idx] += w_score

            # 신뢰도 가중 평균
            conf_sum   += evt.confidence * decay
            weight_sum += decay

            # 최강 이벤트
            if abs(raw_score) > abs(max_score):
                max_score = raw_score

            # 시간 구간 밀도
            for wi, days in enumerate(windows):
                if dt_days <= days:
                    density[wi]  += 1
                    imp_sums[wi] += evt.importance

        # 벡터 조립
        vec[0] = _clip(total_score, -3.0, 3.0)
        vec[1] = _clip(bull_sum,    0.0,  3.0)
        vec[2] = _clip(bear_sum,    0.0,  3.0)
        vec[3] = _clip(uncert_sum,  0.0,  1.0)

        for i, s in enumerate(cat_scores):
            vec[4 + i] = _clip(s, -2.0, 2.0)

        for wi in range(4):
            vec[16 + wi] = min(density[wi] / 10.0, 1.0)
            vec[20 + wi] = (imp_sums[wi] / density[wi]
                            if density[wi] > 0 else 0.0)

        vec[24] = _clip(max_score, -1.0, 1.0)

        # 다양성: 활성 카테고리 수
        active_cats = sum(1 for s in cat_scores if abs(s) > 0.01)
        vec[25] = active_cats / 12.0

        # 센티먼트 트렌드 (최근 3일 vs 이전 3~6일)
        vec[26] = self._compute_sentiment_trend(now)

        # 신뢰도 평균
        vec[27] = (conf_sum / weight_sum) if weight_sum > 0 else 0.0

        return [round(v, 5) for v in vec]

    def compute_category_scores(self, as_of: Optional[datetime] = None
                                 ) -> dict[EventCategory, float]:
        """카테고리별 누적 점수 반환 (UI 시각화용)"""
        now    = as_of or datetime.now()
        scores: dict[EventCategory, float] = {c: 0.0 for c in EventCategory}
        for evt in self._events:
            dt_days = (now - evt.timestamp).total_seconds() / 86400.0
            if dt_days < 0:
                continue
            decay = math.exp(-_LAMBDA[evt.duration] * dt_days)
            for cat in evt.categories:
                if cat in scores:
                    scores[cat] += evt.external_score * decay
        return {k: round(v, 4) for k, v in scores.items()}

    def get_recent(self, hours: float = 24) -> list[StructuredEvent]:
        """최근 N시간 이벤트 반환"""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [e for e in self._events if e.timestamp >= cutoff]

    def get_all(self) -> list[StructuredEvent]:
        return list(self._events)

    def clear(self):
        self._events.clear()

    # ──────────────────────────────────────────────────────────────────
    # 내부 유틸
    # ──────────────────────────────────────────────────────────────────

    def _compute_sentiment_trend(self, now: datetime) -> float:
        recent  = sum(e.sentiment_score for e in self._events
                      if (now - e.timestamp).days <= 3)
        older   = sum(e.sentiment_score for e in self._events
                      if 3 < (now - e.timestamp).days <= 6)
        n_r = sum(1 for e in self._events if (now - e.timestamp).days <= 3)
        n_o = sum(1 for e in self._events if 3 < (now - e.timestamp).days <= 6)
        avg_r = recent / n_r if n_r else 0.0
        avg_o = older  / n_o if n_o else 0.0
        return _clip(avg_r - avg_o, -1.0, 1.0)

    def _save_cache(self):
        try:
            os.makedirs(os.path.dirname(self._cache_file), exist_ok=True)
            data = [e.to_dict() for e in self._events[-200:]]  # 최근 200개만
            with open(self._cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
        except Exception as e:
            logger.debug(f"accumulator 캐시 저장 실패: {e}")

    def _load_cache(self):
        try:
            if os.path.exists(self._cache_file):
                with open(self._cache_file, encoding="utf-8") as f:
                    data = json.load(f)
                self._events = [StructuredEvent.from_dict(d) for d in data]
                logger.info(f"accumulator 캐시 로드: {len(self._events)}건")
        except Exception as e:
            logger.debug(f"accumulator 캐시 로드 실패: {e}")
            self._events = []


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))
