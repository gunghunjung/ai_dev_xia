# features/news_features.py — 멀티윈도우 뉴스 특징 벡터 생성기
"""
뉴스 이벤트 → 모델 입력용 수치 벡터 변환

출력 차원: NEWS_FEATURE_DIM = 40

벡터 구조:
  [0-11]  : 12개 EventCategory별 시간감쇠 가중 점수 (1d 기준)
  [12-17] : 6개 시간 윈도우별 종합 점수 (1h/4h/1d/3d/5d/20d)
  [18-23] : 6개 시간 윈도우별 이벤트 밀도 (정규화된 이벤트 수)
  [24-29] : 6개 시간 윈도우별 감성 평균
  [30]    : 종목 직접 관련 뉴스 점수
  [31]    : 섹터 관련 뉴스 점수
  [32]    : 시장 전체 리스크 점수 (지정학 + 거시)
  [33]    : 정책 리스크 점수
  [34]    : 충격 이벤트 플래그 (0/1)
  [35]    : 반복보도 강도 (같은 이슈 보도 수 정규화)
  [36]    : 정보 신선도 (마지막 이벤트 이후 시간 역수)
  [37]    : 감성 변화 속도 (최근 - 과거 감성)
  [38]    : 뉴스 볼륨 z-score (오늘 vs 평균)
  [39]    : 누적 충격 점수 (시간감쇠 합산)

시간 감쇠:
  weighted_score(t) = Σ event_score × exp(-λ × hours_elapsed)
  λ = 0.1 (1일 기준 약 90% 감쇠)

미래 누수 방지:
  reference_time 파라미터를 예측 기준 시점으로 설정 →
  reference_time 이후 이벤트는 모두 제외
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("quant.news_features")

# ── 상수 ──────────────────────────────────────────────────────────────────────

NEWS_FEATURE_DIM = 40

# 6개 시간 윈도우 (이름, 시간 수)
_WINDOWS: List[Tuple[str, float]] = [
    ("1h",  1.0),
    ("4h",  4.0),
    ("1d",  24.0),
    ("3d",  72.0),
    ("5d",  120.0),
    ("20d", 480.0),
]

# 시간 감쇠 계수 λ
# exp(-λ × 1h) ≈ 0.905 (1시간 후 90.5%)
# exp(-λ × 24h) ≈ 0.09  (하루 후 9%)
_DECAY_LAMBDA = 0.1

# 12개 EventCategory 순서 (event_structure.py 와 동일)
_CATEGORIES = [
    "Macro",
    "MonetaryPolicy",
    "Geopolitics",
    "Industry",
    "Corporate",
    "Government",
    "Flow",
    "MarketEvent",
    "Technology",
    "Commodity",
    "FinancialMkt",
    "Sentiment",
]

# 카테고리별 리스크 가중치 (지정학·거시·통화정책 = 높음)
_RISK_WEIGHTS: Dict[str, float] = {
    "Macro":          1.5,
    "MonetaryPolicy": 1.5,
    "Geopolitics":    1.4,
    "Flow":           1.3,
    "Industry":       1.2,
    "Corporate":      1.2,
    "Commodity":      1.2,
    "Government":     1.1,
    "MarketEvent":    1.1,
    "FinancialMkt":   1.1,
    "Technology":     1.0,
    "Sentiment":      0.8,
}

# 충격 이벤트: impact_strength >= 이 임계값이면 shock_flag = 1
_SHOCK_THRESHOLD = 0.75

# 이벤트 밀도 정규화 기준 (하루 평균 이벤트 수)
_DENSITY_NORM_DAILY = 50.0


# ── 시간 유틸 ─────────────────────────────────────────────────────────────────

def _to_naive(dt: datetime) -> datetime:
    return dt.replace(tzinfo=None) if dt.tzinfo else dt


def _hours_diff(ref: datetime, dt: datetime) -> float:
    """ref - dt (dt가 과거면 양수)"""
    return (_to_naive(ref) - _to_naive(dt)).total_seconds() / 3600.0


def _decay(hours: float) -> float:
    """시간 감쇠 가중치 exp(-λ × hours), hours >= 0"""
    return math.exp(-_DECAY_LAMBDA * max(hours, 0.0))


# ── 핵심 생성기 ───────────────────────────────────────────────────────────────

class NewsFeatureGenerator:
    """
    뉴스 이벤트 → NEWS_FEATURE_DIM=40 차원 특징 벡터 생성.

    사용법:
        gen = NewsFeatureGenerator(db=get_news_db())

        # 예측 기준 시점 t에서 종목 005930.KS 의 뉴스 피처 생성
        feat = gen.build_features(
            symbol        = "005930.KS",
            reference_time = datetime(2025, 3, 15, 9, 0),
            sector        = "Technology",
        )  # shape: (40,) float32

        # 학습 데이터셋 일괄 생성 (날짜 범위)
        feat_matrix = gen.build_dataset(
            symbols       = ["005930.KS", "000660.KS"],
            dates         = pd.date_range("2024-01-01", "2025-03-15", freq="B"),
        )  # shape: (N_days × N_symbols, 40)
    """

    def __init__(
        self,
        db=None,
        decay_lambda: float = _DECAY_LAMBDA,
        shock_threshold: float = _SHOCK_THRESHOLD,
    ):
        self.db              = db
        self.decay_lambda    = decay_lambda
        self.shock_threshold = shock_threshold

        # 밀도 z-score 계산용 슬라이딩 통계 (일간 이벤트 수)
        self._density_history: List[float] = []

    # ── 공개 API ─────────────────────────────────────────────────────────────

    def build_features(
        self,
        symbol:          Optional[str]      = None,
        sector:          Optional[str]      = None,
        reference_time:  Optional[datetime] = None,
        events:          Optional[List[Dict]] = None,
        use_cache:        bool = True,
    ) -> np.ndarray:
        """
        단일 시점 뉴스 특징 벡터 (40D) 생성.

        Args:
            symbol:         종목 코드 (None이면 시장 전체 기준)
            sector:         섹터명 (종목 섹터 필터용)
            reference_time: 예측 기준 시점 (미래 누수 방지 기준)
            events:         직접 제공할 이벤트 목록 (None이면 DB 조회)
            use_cache:      특징 캐시 사용 여부

        Returns:
            np.ndarray shape (40,) float32
        """
        ref = reference_time or datetime.now()
        ref = _to_naive(ref)

        # ── 캐시 확인 ────────────────────────────────────────────────────
        if use_cache and self.db and symbol:
            date_str = ref.strftime("%Y-%m-%d")
            cached   = self.db.get_cached_features(symbol, date_str, "all")
            if cached is not None:
                return np.array(cached, dtype=np.float32)

        # ── 이벤트 로드 ───────────────────────────────────────────────────
        if events is None:
            events = self._load_events(symbol, sector, ref)

        # ── 특징 계산 ─────────────────────────────────────────────────────
        feat = self._compute(events, ref, symbol, sector)

        # ── 캐시 저장 ─────────────────────────────────────────────────────
        if use_cache and self.db and symbol:
            date_str = ref.strftime("%Y-%m-%d")
            self.db.cache_features(symbol, date_str, "all",
                                   feat.tolist(), len(events))

        return feat

    def build_features_multiwindow(
        self,
        symbol:         Optional[str] = None,
        sector:         Optional[str] = None,
        reference_time: Optional[datetime] = None,
        events:         Optional[List[Dict]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        각 시간 윈도우별 특징 벡터 반환.

        Returns:
            {"1h": np.ndarray(40), "4h": ..., "1d": ..., ...}
        """
        ref    = _to_naive(reference_time or datetime.now())
        events = events or self._load_events(symbol, sector, ref)

        result = {}
        for window_name, window_hours in _WINDOWS:
            cutoff   = ref - timedelta(hours=window_hours)
            filtered = [e for e in events
                        if _to_naive(_parse_dt(e.get("published_at", ""))) >= cutoff]
            result[window_name] = self._compute(filtered, ref, symbol, sector)

        return result

    def build_dataset(
        self,
        symbols:    List[str],
        dates,                        # pd.DatetimeIndex 또는 list[datetime]
        sector_map: Dict[str, str] = None,
    ) -> Dict[str, np.ndarray]:
        """
        학습용 데이터셋 일괄 생성.

        Returns:
            {symbol: np.ndarray shape (N_dates, 40)}
        """
        sector_map = sector_map or {}
        result     = {}

        for symbol in symbols:
            sector    = sector_map.get(symbol)
            rows: List[np.ndarray] = []

            for dt in dates:
                ref = dt if isinstance(dt, datetime) else datetime(dt.year, dt.month, dt.day, 9, 0)
                try:
                    feat = self.build_features(
                        symbol=symbol, sector=sector,
                        reference_time=ref, use_cache=True,
                    )
                except Exception as e:
                    logger.debug(f"feature 생성 실패 ({symbol}, {ref}): {e}")
                    feat = np.zeros(NEWS_FEATURE_DIM, dtype=np.float32)
                rows.append(feat)

            result[symbol] = np.stack(rows, axis=0) if rows else np.zeros((0, NEWS_FEATURE_DIM), dtype=np.float32)
            logger.debug(f"dataset 생성: {symbol} → {result[symbol].shape}")

        return result

    # ── 이벤트 로드 ───────────────────────────────────────────────────────────

    def _load_events(
        self,
        symbol:  Optional[str],
        sector:  Optional[str],
        ref:     datetime,
    ) -> List[Dict]:
        """DB에서 최대 20일치 이벤트 로드 (미래 누수 방지)"""
        if self.db is None:
            return []
        since = ref - timedelta(days=20)
        try:
            return self.db.get_events(
                since_dt=since, until_dt=ref,
                symbol=symbol, sector=sector,
                only_representative=True,
                limit=1000,
            )
        except Exception as e:
            logger.warning(f"이벤트 로드 실패: {e}")
            return []

    # ── 핵심 계산 ─────────────────────────────────────────────────────────────

    def _compute(
        self,
        events:  List[Dict],
        ref:     datetime,
        symbol:  Optional[str],
        sector:  Optional[str],
    ) -> np.ndarray:
        """40D 특징 벡터 계산"""
        feat = np.zeros(NEWS_FEATURE_DIM, dtype=np.float32)

        if not events:
            return feat

        # 이벤트별 기본 값 추출 및 시간 감쇠 가중치 계산
        enriched = []
        for e in events:
            dt    = _parse_dt(e.get("published_at", ""))
            hours = _hours_diff(ref, dt)
            if hours < 0:  # 미래 이벤트 제외 (누수 방지)
                continue
            w = _decay(hours)
            enriched.append({
                **e,
                "_hours": hours,
                "_decay": w,
                "_score": float(e.get("computed_score", 0.0)),
                "_dir":   int(e.get("impact_direction", 0)),
                "_str":   float(e.get("impact_strength", 0.0)),
                "_conf":  float(e.get("confidence", 0.0)),
                "_sent":  float(e.get("sentiment_score", 0.0)),
                "_cats":  e.get("categories", []) or [],
                "_repeat": int(e.get("repeat_count", 1)),
            })

        if not enriched:
            return feat

        # ────────────────────────────────────────────────────────────────
        # [0-11] 카테고리별 시간감쇠 가중 점수 (1d 기준)
        # ────────────────────────────────────────────────────────────────
        one_day_events = [e for e in enriched if e["_hours"] <= 24.0]
        for e in one_day_events:
            w  = e["_decay"]
            sc = e["_dir"] * e["_str"] * e["_conf"]
            rw = _RISK_WEIGHTS.get
            for cat in e["_cats"]:
                if cat in _CATEGORIES:
                    idx = _CATEGORIES.index(cat)
                    feat[idx] += float(w * sc * rw(cat, 1.0))

        # ────────────────────────────────────────────────────────────────
        # [12-17] 윈도우별 종합 점수
        # [18-23] 윈도우별 이벤트 밀도
        # [24-29] 윈도우별 감성 평균
        # ────────────────────────────────────────────────────────────────
        for wi, (wname, whours) in enumerate(_WINDOWS):
            w_events = [e for e in enriched if e["_hours"] <= whours]
            if not w_events:
                continue

            # 종합 점수: 방향 × 강도 × 신뢰도 × 시간감쇠
            total_score = sum(
                e["_dir"] * e["_str"] * e["_conf"] * e["_decay"]
                for e in w_events
            )
            feat[12 + wi] = float(np.clip(total_score, -5.0, 5.0))

            # 이벤트 밀도 (정규화)
            norm_factor  = max(1.0, whours / 24.0) * _DENSITY_NORM_DAILY
            feat[18 + wi] = float(min(1.0, len(w_events) / norm_factor))

            # 감성 평균 (가중)
            sent_num = sum(e["_sent"] * e["_decay"] for e in w_events)
            sent_den = sum(e["_decay"] for e in w_events)
            feat[24 + wi] = float(sent_num / max(sent_den, 1e-9))

        # ────────────────────────────────────────────────────────────────
        # [30] 종목 직접 관련 뉴스 점수
        # ────────────────────────────────────────────────────────────────
        if symbol:
            stock_events = [e for e in enriched
                            if symbol in (e.get("target_stocks") or [])]
            if stock_events:
                sc = sum(e["_dir"] * e["_str"] * e["_decay"] for e in stock_events)
                feat[30] = float(np.clip(sc, -3.0, 3.0))

        # ────────────────────────────────────────────────────────────────
        # [31] 섹터 관련 뉴스 점수
        # ────────────────────────────────────────────────────────────────
        if sector:
            sector_events = [e for e in enriched
                             if sector in (e.get("target_sectors") or [])]
            if sector_events:
                sc = sum(e["_dir"] * e["_str"] * e["_decay"] for e in sector_events)
                feat[31] = float(np.clip(sc, -3.0, 3.0))

        # ────────────────────────────────────────────────────────────────
        # [32] 시장 리스크 (지정학 + 거시)
        # ────────────────────────────────────────────────────────────────
        risk_cats = {"Geopolitics", "Macro", "MonetaryPolicy"}
        risk_events = [e for e in enriched
                       if any(c in risk_cats for c in e["_cats"])
                       and e["_hours"] <= 24.0]
        if risk_events:
            risk_sc = sum(
                -e["_dir"] * e["_str"] * e["_conf"] * e["_decay"]
                for e in risk_events
            )
            feat[32] = float(np.clip(risk_sc, 0.0, 3.0))

        # ────────────────────────────────────────────────────────────────
        # [33] 정책 리스크
        # ────────────────────────────────────────────────────────────────
        policy_events = [e for e in enriched
                         if any(c in {"Government", "MonetaryPolicy"}
                                for c in e["_cats"])
                         and e["_hours"] <= 24.0]
        if policy_events:
            feat[33] = float(np.clip(
                sum(-e["_dir"] * e["_str"] * e["_decay"] for e in policy_events),
                0.0, 3.0
            ))

        # ────────────────────────────────────────────────────────────────
        # [34] 충격 이벤트 플래그
        # ────────────────────────────────────────────────────────────────
        shock = any(
            e["_str"] >= self.shock_threshold and e["_hours"] <= 24.0
            for e in enriched
        )
        feat[34] = 1.0 if shock else 0.0

        # ────────────────────────────────────────────────────────────────
        # [35] 반복보도 강도 (대표 이벤트의 repeat_count 합산, 정규화)
        # ────────────────────────────────────────────────────────────────
        repeat_total = sum(e["_repeat"] for e in enriched if e["_hours"] <= 24.0)
        feat[35]     = float(min(1.0, repeat_total / 100.0))

        # ────────────────────────────────────────────────────────────────
        # [36] 정보 신선도 (마지막 이벤트 이후 경과 시간, 역수)
        # ────────────────────────────────────────────────────────────────
        min_hours = min(e["_hours"] for e in enriched)
        feat[36]  = float(1.0 / (1.0 + min_hours))

        # ────────────────────────────────────────────────────────────────
        # [37] 감성 변화 속도 (최근 4h 감성 - 최근 24h 감성)
        # ────────────────────────────────────────────────────────────────
        recent = [e for e in enriched if e["_hours"] <= 4.0]
        older  = [e for e in enriched if 4.0 < e["_hours"] <= 24.0]
        s_recent = (sum(e["_sent"] for e in recent) / max(len(recent), 1)
                    if recent else 0.0)
        s_older  = (sum(e["_sent"] for e in older) / max(len(older), 1)
                    if older else 0.0)
        feat[37]  = float(np.clip(s_recent - s_older, -2.0, 2.0))

        # ────────────────────────────────────────────────────────────────
        # [38] 뉴스 볼륨 z-score (오늘 이벤트 수 vs 최근 이력 평균)
        # ────────────────────────────────────────────────────────────────
        today_count = len([e for e in enriched if e["_hours"] <= 24.0])
        self._density_history.append(float(today_count))
        if len(self._density_history) > 60:
            self._density_history = self._density_history[-60:]

        if len(self._density_history) >= 5:
            mu  = float(np.mean(self._density_history))
            std = float(np.std(self._density_history)) + 1e-6
            feat[38] = float(np.clip((today_count - mu) / std, -3.0, 3.0))

        # ────────────────────────────────────────────────────────────────
        # [39] 누적 충격 점수 (20일치, 시간감쇠 합산)
        # ────────────────────────────────────────────────────────────────
        cum_impact = sum(e["_dir"] * e["_str"] * e["_decay"] for e in enriched)
        feat[39]   = float(np.clip(cum_impact, -10.0, 10.0))

        return feat


# ── 미래 누수 검사 ─────────────────────────────────────────────────────────────

def check_leakage(features_by_date: Dict[str, np.ndarray],
                  events_by_date:   Dict[str, List[Dict]]) -> List[str]:
    """
    학습 데이터셋의 미래 정보 누수 검사.

    Returns:
        누수 의심 날짜 목록 (비어 있으면 정상)
    """
    leaks = []
    for date_str, feat in features_by_date.items():
        ref_dt   = datetime.strptime(date_str, "%Y-%m-%d")
        evts     = events_by_date.get(date_str, [])
        for e in evts:
            evt_dt = _parse_dt(e.get("published_at", ""))
            if _to_naive(evt_dt) > _to_naive(ref_dt) + timedelta(hours=1):
                leaks.append(f"{date_str}: {e.get('title','')[:50]} ({evt_dt})")
    return leaks


# ── 유틸 ──────────────────────────────────────────────────────────────────────

def _parse_dt(s: str) -> datetime:
    """다양한 ISO 포맷 → datetime"""
    if not s:
        return datetime(2000, 1, 1)
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d"):
        try:
            return datetime.strptime(s[:26].rstrip("Z"), fmt)
        except ValueError:
            continue
    return datetime(2000, 1, 1)


# ── 싱글턴 ────────────────────────────────────────────────────────────────────

_generator_instance: Optional[NewsFeatureGenerator] = None


def get_news_feature_generator(db=None) -> NewsFeatureGenerator:
    global _generator_instance
    if _generator_instance is None:
        from data.news_db import get_news_db
        _generator_instance = NewsFeatureGenerator(db=db or get_news_db())
    return _generator_instance
