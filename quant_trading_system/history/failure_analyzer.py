"""
history/failure_analyzer.py
============================
자기교정 실패 분석 엔진.

핵심 철학:
  - 예측은 가설, 실제 결과가 진실이다
  - 성공보다 실패 패턴 해부가 더 중요하다
  - 반복되는 실패 유형을 식별해 모델 개선 방향을 제시한다
"""
from __future__ import annotations

import math
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import Any, Optional

from history.schema import PredictionRecord
from history.logger import PredictionLogger

# ── 실패 유형 분류 ─────────────────────────────────────────────────────────────
FAILURE_TYPES = {
    "MISSED_CRASH":       "급락 미감지",        # 상승/중립 예측 → 5%+ 실제 하락
    "MISSED_RALLY":       "급등 미감지",        # 하락/중립 예측 → 5%+ 실제 상승
    "OVER_OPTIMISTIC":    "과도한 낙관",        # 예측 수익률이 실제보다 5%p+ 높음
    "OVER_PESSIMISTIC":   "과도한 비관",        # 예측 수익률이 실제보다 5%p+ 낮음
    "SIDEWAYS_MISREAD":   "횡보 추세 오판",     # 상승/하락 예측 → ±1% 이내 횡보
    "DIRECTION_HIT":      "방향 적중",          # 방향 일치
    "WITHIN_TOLERANCE":   "허용 오차 이내",     # 방향 불일치지만 오차 2% 미만
}

CRASH_THRESHOLD  = -0.05   # -5% 이하 → 급락
RALLY_THRESHOLD  =  0.05   # +5% 이상 → 급등
SIDEWAYS_BAND    =  0.01   # ±1% 이내 → 횡보
OPTIMISM_DELTA   =  0.05   # 5%p 이상 초과 예측 → 낙관 편향
TOLERANCE_ERROR  =  0.02   # 2% 이하 오차 → 허용


@dataclass
class FailureSummary:
    total_predictions: int = 0
    verified_count: int = 0
    hit_count: int = 0
    hit_rate: float = 0.0
    mae: float = 0.0          # Mean Absolute Error (%)
    mse: float = 0.0          # Mean Squared Error
    rmse: float = 0.0         # Root MSE
    mape: float = 0.0         # Mean Absolute Percentage Error
    directional_accuracy: float = 0.0
    failure_type_counts: dict = field(default_factory=dict)
    by_symbol: dict = field(default_factory=dict)
    by_horizon: dict = field(default_factory=dict)
    by_confidence: dict = field(default_factory=dict)
    top_failures: list = field(default_factory=list)
    suggestions: list = field(default_factory=list)


class FailureAnalyzer:
    def __init__(self, pl: PredictionLogger):
        self._pl = pl

    def classify_failure_type(self, rec: PredictionRecord) -> str:
        """단일 검증된 예측의 실패 유형을 분류."""
        if not rec.verified or rec.actual_return_pct is None:
            return "UNVERIFIED"

        actual    = rec.actual_return_pct / 100.0   # convert pct to ratio
        predicted = rec.predicted_return_pct / 100.0

        if rec.hit:
            err = abs(actual - predicted)
            if err < TOLERANCE_ERROR:
                return "WITHIN_TOLERANCE"
            return "DIRECTION_HIT"

        # Direction was wrong
        if actual <= CRASH_THRESHOLD and rec.predicted_direction in ("UP", "NEUTRAL"):
            return "MISSED_CRASH"
        if actual >= RALLY_THRESHOLD and rec.predicted_direction in ("DOWN", "NEUTRAL"):
            return "MISSED_RALLY"

        pred_err = predicted - actual  # positive = over-optimistic
        if pred_err > OPTIMISM_DELTA:
            return "OVER_OPTIMISTIC"
        if pred_err < -OPTIMISM_DELTA:
            return "OVER_PESSIMISTIC"
        if abs(actual) < SIDEWAYS_BAND:
            return "SIDEWAYS_MISREAD"

        return "DIRECTION_HIT"  # fallback (shouldn't reach here if hit=False)

    def analyze(self, top_n: int = 20) -> FailureSummary:
        """전체 이력 분석. FailureSummary 반환."""
        all_recs = self._pl.load_all()
        verified = [r for r in all_recs if r.verified and r.actual_return_pct is not None]

        summary = FailureSummary(
            total_predictions=len(all_recs),
            verified_count=len(verified),
        )

        if not verified:
            summary.suggestions = [
                "검증된 예측이 없습니다. '예측이력 > 검증 실행'을 먼저 실행하세요."
            ]
            return summary

        # 적중 통계
        hits = [r for r in verified if r.hit]
        summary.hit_count = len(hits)
        summary.hit_rate  = len(hits) / len(verified)

        # 오차 통계
        errors     = [abs(r.actual_return_pct - r.predicted_return_pct) for r in verified]
        sq_errors  = [e ** 2 for e in errors]
        pct_errors = [
            abs((r.actual_return_pct - r.predicted_return_pct) / (abs(r.predicted_return_pct) + 1e-8)) * 100
            for r in verified
        ]

        summary.mae  = sum(errors) / len(errors)
        summary.mse  = sum(sq_errors) / len(sq_errors)
        summary.rmse = math.sqrt(summary.mse)
        summary.mape = sum(pct_errors) / len(pct_errors)

        # 방향 정확도 (verified hit 기준)
        dir_hits = sum(1 for r in verified if r.hit)
        summary.directional_accuracy = dir_hits / len(verified)

        # 실패 유형 분류
        type_counter = Counter()
        enriched: list[tuple[PredictionRecord, str, float]] = []
        for r in verified:
            ft = self.classify_failure_type(r)
            type_counter[ft] += 1
            enriched.append((r, ft, abs(r.actual_return_pct - r.predicted_return_pct)))

        summary.failure_type_counts = dict(type_counter)

        # 종목별 통계
        by_sym: dict[str, Any] = defaultdict(
            lambda: {"total": 0, "hits": 0, "total_error": 0.0}
        )
        for r in verified:
            s = by_sym[r.symbol]
            s["total"] += 1
            if r.hit:
                s["hits"] += 1
            s["total_error"] += abs(r.actual_return_pct - r.predicted_return_pct)

        summary.by_symbol = {
            sym: {
                "hit_rate": d["hits"] / d["total"],
                "mae":      d["total_error"] / d["total"],
                "count":    d["total"],
            }
            for sym, d in by_sym.items()
        }

        # 기간별 통계
        by_horizon: dict[int, Any] = defaultdict(
            lambda: {"total": 0, "hits": 0, "total_error": 0.0}
        )
        for r in verified:
            h = r.horizon_days
            by_horizon[h]["total"] += 1
            if r.hit:
                by_horizon[h]["hits"] += 1
            by_horizon[h]["total_error"] += abs(r.actual_return_pct - r.predicted_return_pct)

        summary.by_horizon = {
            h: {
                "hit_rate": d["hits"] / d["total"],
                "mae":      d["total_error"] / d["total"],
                "count":    d["total"],
            }
            for h, d in by_horizon.items()
        }

        # 신뢰도별
        by_conf: dict[str, Any] = defaultdict(lambda: {"total": 0, "hits": 0})
        for r in verified:
            by_conf[r.confidence]["total"] += 1
            if r.hit:
                by_conf[r.confidence]["hits"] += 1

        summary.by_confidence = {
            c: {
                "hit_rate": d["hits"] / d["total"],
                "count":    d["total"],
            }
            for c, d in by_conf.items()
        }

        # Top N 실패 (가장 큰 오차 우선)
        failures_only = [
            (r, ft, err) for r, ft, err in enriched
            if ft not in ("DIRECTION_HIT", "WITHIN_TOLERANCE")
        ]
        failures_only.sort(key=lambda x: x[2], reverse=True)

        summary.top_failures = [
            {
                "symbol":               r.symbol,
                "name":                 getattr(r, "name", r.symbol),
                "timestamp":            r.timestamp[:10],
                "horizon_days":         r.horizon_days,
                "predicted_direction":  r.predicted_direction,
                "actual_return_pct":    r.actual_return_pct,
                "predicted_return_pct": r.predicted_return_pct,
                "abs_error_pct":        err,
                "failure_type":         ft,
                "failure_type_kr":      FAILURE_TYPES.get(ft, ft),
                "confidence":           r.confidence,
                "action":               r.action,
                "id":                   r.id,
            }
            for r, ft, err in failures_only[:top_n]
        ]

        # 개선 제안 생성
        summary.suggestions = self._generate_suggestions(summary)

        return summary

    # ──────────────────────────────────────────────────────────────────────────
    # 내부 헬퍼
    # ──────────────────────────────────────────────────────────────────────────

    def _generate_suggestions(self, s: FailureSummary) -> list[str]:
        """실패 패턴 기반 개선 제안."""
        suggestions: list[str] = []
        counts     = s.failure_type_counts
        total_fail = sum(
            v for k, v in counts.items()
            if k not in ("DIRECTION_HIT", "WITHIN_TOLERANCE")
        )

        if total_fail == 0:
            suggestions.append("심각한 실패 패턴이 발견되지 않았습니다.")
            return suggestions

        # 급락 미감지
        missed_crash = counts.get("MISSED_CRASH", 0)
        if missed_crash > 0:
            pct = missed_crash / total_fail * 100
            if pct >= 20:
                suggestions.append(
                    f"[급락 미감지 {pct:.0f}%] 하방 리스크 피처 강화 필요: "
                    "거래량 급증, 시가 갭다운, VIX 급등 피처를 추가하세요."
                )

        # 과도한 낙관
        over_opt = counts.get("OVER_OPTIMISTIC", 0)
        if over_opt > 0:
            pct = over_opt / total_fail * 100
            if pct >= 20:
                suggestions.append(
                    f"[과도한 낙관 {pct:.0f}%] 수익률 예측 상한을 낮추거나 "
                    "정규화를 강화하세요. mu 클리핑 임계값을 줄이는 것도 고려하세요."
                )

        # 횡보 오판
        sideways = counts.get("SIDEWAYS_MISREAD", 0)
        if sideways > 0:
            pct = sideways / total_fail * 100
            if pct >= 25:
                suggestions.append(
                    f"[횡보 오판 {pct:.0f}%] NEUTRAL 임계값을 넓히세요. "
                    "mu_min_signal을 현재보다 높게 설정하면 횡보 오판이 줄어듭니다."
                )

        # 신뢰도별 괴리
        conf_data = s.by_confidence
        high_hit = conf_data.get("HIGH", {}).get("hit_rate", None)
        low_hit  = conf_data.get("LOW",  {}).get("hit_rate", None)
        if high_hit is not None and high_hit < 0.55:
            suggestions.append(
                f"[HIGH 신뢰도 적중률 낮음: {high_hit:.0%}] "
                "SNR 임계값(_SNR_HIGH)을 높이거나, "
                "prob_buy_thresh를 0.65 이상으로 올리세요."
            )

        # 전체 적중률
        if s.hit_rate < 0.50:
            suggestions.append(
                f"[전체 적중률 {s.hit_rate:.0%} < 50%] "
                "랜덤보다 못한 예측입니다. "
                "피처 엔지니어링 재검토 및 모델 재학습이 필요합니다."
            )
        elif s.hit_rate < 0.55:
            suggestions.append(
                f"[적중률 {s.hit_rate:.0%}] 랜덤 수준에 가깝습니다. "
                "더 많은 학습 데이터 또는 앙상블 모델을 고려하세요."
            )

        # MAE 기준
        if s.mae > 10.0:
            suggestions.append(
                f"[MAE {s.mae:.1f}%p 과대] 예측 스케일이 실제와 크게 다릅니다. "
                "GBM 스케일링 팩터(time_scale)를 검토하세요."
            )

        # 급등 미감지
        missed_rally = counts.get("MISSED_RALLY", 0)
        if missed_rally > 0 and missed_rally >= missed_crash:
            suggestions.append(
                "[급등 미감지] 하락 편향이 있습니다. "
                "훈련 레이블 분포를 확인하고 상승 샘플 비중을 높이세요."
            )

        if not suggestions:
            suggestions.append(
                "주요 개선 포인트가 발견되지 않았습니다. 데이터를 더 수집하세요."
            )

        return suggestions
