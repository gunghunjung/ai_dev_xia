"""
RegimeEnsemble — 시장 국면 적응형 동적 앙상블
────────────────────────────────────────────────
국면별로 모델 가중치를 동적으로 조정:
  BULL_QUIET    → Transformer, TFT, PatchTST 강조
  BULL_VOLATILE → 앙상블 분산 최소화 (균등 + Uncertainty 필터)
  BEAR_QUIET    → LSTM/GRU + GBM 균등
  BEAR_VOLATILE → GBM, TCN 강조 (비선형 패턴 포착)
  RANGING       → LightGBM, XGBoost 강조 (평균 회귀 포착)
"""
from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from ..data.regime_detector import Regime, RegimeDetector
from .ensemble_predictor import EnsemblePredictor
from ..logger_config import get_logger

log = get_logger("ensemble.regime")

# 국면별 선호 모델 키워드 + 가중치 배수
_REGIME_PREF: Dict[Regime, Dict[str, float]] = {
    Regime.BULL_QUIET: {
        "transformer": 1.8, "tft": 1.8, "patchtst": 1.6,
        "lstm": 1.2, "gru": 1.2,
        "xgboost": 0.7, "lightgbm": 0.7, "tcn": 1.0,
    },
    Regime.BULL_VOLATILE: {
        # 고변동 → 모든 모델 균등 + 앙상블 효과 극대화
        "transformer": 1.0, "tft": 1.0, "patchtst": 1.0,
        "lstm": 1.0, "gru": 1.0, "xgboost": 1.0, "lightgbm": 1.0, "tcn": 1.0,
    },
    Regime.BEAR_QUIET: {
        "lstm": 1.4, "gru": 1.4, "xgboost": 1.3, "lightgbm": 1.3,
        "transformer": 0.9, "tft": 0.9, "tcn": 1.1,
    },
    Regime.BEAR_VOLATILE: {
        "xgboost": 1.6, "lightgbm": 1.6, "tcn": 1.4,
        "lstm": 1.0, "gru": 1.0,
        "transformer": 0.7, "tft": 0.7, "patchtst": 0.7,
    },
    Regime.RANGING: {
        "lightgbm": 1.8, "xgboost": 1.8, "tcn": 1.2,
        "lstm": 0.8, "gru": 0.8,
        "transformer": 0.7, "tft": 0.7,
    },
}


class RegimeEnsemble:
    """
    Parameters
    ----------
    detector     : RegimeDetector 인스턴스
    base_ensemble: EnsemblePredictor 인스턴스
    blend_factor : 국면 가중치 혼합 비율 (0=무시, 1=완전 적용)
    """

    def __init__(
        self,
        detector:      Optional[RegimeDetector] = None,
        base_ensemble: Optional[EnsemblePredictor] = None,
        blend_factor:  float = 0.7,
    ) -> None:
        self._detector  = detector or RegimeDetector()
        self._ensemble  = base_ensemble or EnsemblePredictor()
        self._blend     = blend_factor
        self._last_regime: Optional[Regime] = None
        self._regime_history: List[Regime] = []

    @property
    def ensemble(self) -> EnsemblePredictor:
        return self._ensemble

    # ── 학습/모델 위임 ────────────────────────────────────────────────────
    def add_model(self, name: str, model: Any, val_score: float = 1.0) -> None:
        self._ensemble.add_model(name, model, val_score)

    # ── 예측 ──────────────────────────────────────────────────────────────
    def predict(
        self,
        X: np.ndarray,
        price_df=None,   # regime 감지용 원시 DataFrame
    ) -> Dict[str, Any]:
        """
        국면을 감지하고 동적 가중치 적용 후 예측.
        price_df 없으면 국면 감지 스킵, base 앙상블 사용.
        """
        regime_weights = None

        if price_df is not None and len(price_df) >= 20:
            regime, label, confidence = self._detector.current_regime(price_df)
            self._last_regime = regime
            self._regime_history.append(regime)
            if len(self._regime_history) > 100:
                self._regime_history.pop(0)

            # 국면별 가중치 계산
            regime_weights = self._calc_regime_weights(regime, confidence)
            log.info(f"국면: {label} (신뢰도={confidence:.2f})")

        result = self._ensemble.predict(X, regime_weights=regime_weights)
        result["regime"]            = self._last_regime
        result["regime_label"]      = self._regime_label(self._last_regime)
        result["regime_weights"]    = regime_weights or {}
        return result

    def drift_check(self, price_df) -> Dict[str, Any]:
        """분포 변화(Drift) 감지 — 가중치 신뢰도에 반영"""
        return self._detector.detect_drift(price_df)

    # ── 내부 ──────────────────────────────────────────────────────────────
    def _calc_regime_weights(
        self, regime: Regime, confidence: float
    ) -> Dict[str, float]:
        """
        국면 선호도 × 밸런스 가중치 혼합.
        신뢰도 낮으면 균등 가중치 쪽으로 blend.
        """
        model_names = self._ensemble.model_names
        if not model_names:
            return {}

        # 베이스 (성능 기반)
        base_w = self._ensemble.compute_weights()

        # 국면 배수
        pref = _REGIME_PREF.get(regime, {})
        regime_w: Dict[str, float] = {}
        for name in model_names:
            multiplier = 1.0
            for key, mult in pref.items():
                if key in name.lower():
                    multiplier = mult
                    break
            regime_w[name] = base_w.get(name, 1.0 / len(model_names)) * multiplier

        # 정규화
        total = sum(regime_w.values())
        if total > 0:
            regime_w = {n: v / total for n, v in regime_w.items()}

        # 신뢰도 기반 블렌딩
        effective_blend = self._blend * confidence
        final_w: Dict[str, float] = {}
        for name in model_names:
            bw = base_w.get(name, 1.0 / len(model_names))
            rw = regime_w.get(name, 1.0 / len(model_names))
            final_w[name] = (1 - effective_blend) * bw + effective_blend * rw

        # 재정규화
        total = sum(final_w.values())
        return {n: v / total for n, v in final_w.items()}

    @staticmethod
    def _regime_label(regime: Optional[Regime]) -> str:
        if regime is None:
            return "국면 미감지"
        from ..data.regime_detector import REGIME_LABELS
        return REGIME_LABELS.get(regime, str(regime))

    def regime_distribution(self) -> Dict[str, float]:
        """최근 국면 분포 통계"""
        if not self._regime_history:
            return {}
        total = len(self._regime_history)
        counts: Dict[int, int] = {}
        for r in self._regime_history:
            counts[int(r)] = counts.get(int(r), 0) + 1
        from ..data.regime_detector import REGIME_LABELS
        return {
            REGIME_LABELS.get(Regime(k), str(k)): v / total
            for k, v in counts.items()
        }
