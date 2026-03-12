"""
ForecastEngine — 메인 예측 오케스트레이터
──────────────────────────────────────────
학습된 앙상블 모델을 이용해 다음 항목을 계산:

  [예측 출력]
  - 향후 1/3/5/20일 수익률 예측
  - 방향성 확률 P(상승) / P(하락)
  - 목표가 도달 확률 (target_pct 기준)
  - 손절선 이탈 확률 (stop_loss_pct 기준)
  - 95%/80% 신뢰구간
  - Conformal Prediction 보정 구간
  - 변동성 예측 (향후 5일 RV)

  [시장 판단]
  - 종합 신호: 매우 약세 / 약세 / 중립 / 강세 / 매우 강세

  [리스크 경고]
  - 예측 불확실성 높음
  - 분포 변화(Drift) 감지
  - 모델 간 의견 불일치

⚠ 이 예측은 과거 패턴 기반 확률 추정이며 미래를 보장하지 않습니다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from ..logger_config import get_logger
from ..ensemble.regime_ensemble import RegimeEnsemble

log = get_logger("prediction.engine")

# 종합 신호 임계값
SIGNAL_THRESHOLDS = [
    ("매우 강세 📈📈", 0.65),
    ("강세 📈",       0.55),
    ("중립 ↔",       0.45),
    ("약세 📉",       0.35),
    ("매우 약세 📉📉", 0.0),
]


class ForecastEngine:
    """
    Parameters
    ----------
    ensemble      : RegimeEnsemble (학습된 모델 포함)
    horizons      : 예측 시계 목록 (일)
    seq_len       : 입력 시퀀스 길이
    target_pct    : 목표가 기준 수익률 (예: 0.05 = 5%)
    stop_loss_pct : 손절 기준 수익률 (예: -0.03 = -3%)
    mc_samples    : MC Dropout 샘플 수
    calibration_data: Conformal calibration용 residuals
    """

    def __init__(
        self,
        ensemble:          RegimeEnsemble,
        horizons:          List[int] = [1, 3, 5, 20],
        seq_len:           int = 60,
        target_pct:        float = 0.05,
        stop_loss_pct:     float = -0.03,
        mc_samples:        int = 30,
        calibration_data:  Optional[np.ndarray] = None,
    ) -> None:
        self._ensemble   = ensemble
        self._horizons   = horizons
        self._seq_len    = seq_len
        self._target_pct = target_pct
        self._stop_pct   = stop_loss_pct
        self._mc_n       = mc_samples
        self._calib      = calibration_data   # residuals for conformal

    # ── 메인 예측 ─────────────────────────────────────────────────────────
    def forecast(
        self,
        X_seq:     np.ndarray,    # (seq_len, n_features) 또는 (B, seq_len, n_features)
        price_df:  Optional[pd.DataFrame] = None,
        current_price: float = 0.0,
    ) -> Dict[str, Any]:
        """
        종합 예측 결과 반환.

        Returns
        -------
        dict with all forecast outputs + risk warnings + final signal
        """
        # 배치 차원 추가
        if X_seq.ndim == 2:
            X_batch = X_seq[np.newaxis]
        else:
            X_batch = X_seq

        # 앙상블 예측
        ens_result = self._ensemble.predict(X_batch, price_df=price_df)

        point       = float(ens_result["point"].mean())
        lower_90    = float(ens_result["lower_90"].mean())
        upper_90    = float(ens_result["upper_90"].mean())
        lower_50    = float(ens_result["lower_50"].mean())
        upper_50    = float(ens_result["upper_50"].mean())
        prob_up     = float(ens_result["prob_up"].mean())
        prob_down   = float(ens_result["prob_down"].mean())
        uncertainty = float(ens_result["uncertainty"].mean())
        regime      = ens_result.get("regime")
        regime_label= ens_result.get("regime_label", "N/A")

        # 컨포멀 보정 구간
        conf_lower, conf_upper = self._conformal_interval(point, alpha=0.1)

        # 목표가 / 손절가 확률
        target_prob    = self._prob_threshold(lower_90, upper_90, self._target_pct)
        stop_loss_prob = self._prob_threshold(lower_90, upper_90, self._stop_pct, above=False)

        # 다중 horizon 예측
        multi_horizon = self._multi_horizon_forecast(ens_result)

        # 변동성 예측 (개별 예측 분산)
        pred_vol = self._forecast_volatility(ens_result)

        # Drift 체크
        drift_info: Dict[str, Any] = {}
        if price_df is not None and len(price_df) > 60:
            drift_info = self._ensemble.drift_check(price_df)

        # 종합 신호
        signal = self._compute_signal(prob_up, uncertainty, drift_info)

        # 리스크 경고
        warnings = self._build_warnings(
            uncertainty, drift_info, ens_result.get("individual_preds", {}),
            prob_up, target_prob, stop_loss_prob,
        )

        # 가격 기반 목표가/손절가 (절대값)
        target_price    = current_price * (1 + self._target_pct) if current_price else None
        stop_loss_price = current_price * (1 + self._stop_pct)   if current_price else None

        result = {
            # 핵심 예측
            "point_return":    point,
            "prob_up":         prob_up,
            "prob_down":       prob_down,
            "uncertainty":     uncertainty,

            # 신뢰구간
            "ci_90_lower":     lower_90,
            "ci_90_upper":     upper_90,
            "ci_50_lower":     lower_50,
            "ci_50_upper":     upper_50,
            "conformal_lower": conf_lower,
            "conformal_upper": conf_upper,

            # 목표/손절
            "target_pct":        self._target_pct,
            "stop_loss_pct":     self._stop_pct,
            "target_prob":       target_prob,
            "stop_loss_prob":    stop_loss_prob,
            "target_price":      target_price,
            "stop_loss_price":   stop_loss_price,

            # 다중 시계
            "multi_horizon":     multi_horizon,

            # 변동성
            "forecast_volatility": pred_vol,

            # 국면
            "regime":            regime,
            "regime_label":      regime_label,

            # 모델 정보
            "individual_preds":  ens_result.get("individual_preds", {}),
            "weights_used":      ens_result.get("weights_used", {}),

            # 신호 + 경고
            "signal":            signal,
            "warnings":          warnings,
            "drift":             drift_info,

            # 법적 고지 (필수)
            "disclaimer": (
                "⚠ 이 예측은 과거 패턴 기반 확률 추정이며 미래를 보장하지 않습니다. "
                "투자 결정 전 전문가 상담을 권장합니다."
            ),
        }

        log.info(
            f"예측 완료: point={point:.4f}, P(↑)={prob_up:.2%}, "
            f"signal={signal}, regime={regime_label}"
        )
        return result

    # ── 다중 시계 예측 ────────────────────────────────────────────────────
    def _multi_horizon_forecast(
        self, ens_result: Dict[str, Any]
    ) -> Dict[int, Dict[str, float]]:
        """
        단순 선형 외삽 + 불확실성 증폭으로 다중 시계 근사.
        (각 horizon별 별도 모델 학습이 이상적이지만, 단일 모델에서 근사)
        """
        base_point = float(ens_result["point"].mean())
        base_unc   = float(ens_result["uncertainty"].mean())
        result = {}
        for h in self._horizons:
            # horizon이 길수록 불확실성 증가 (sqrt(h) 법칙)
            scale = np.sqrt(h)
            result[h] = {
                "point":    base_point * h,         # 단순 누적 근사
                "lower_90": base_point * h - 1.645 * base_unc * scale,
                "upper_90": base_point * h + 1.645 * base_unc * scale,
                "prob_up":  self._clamp(0.5 + (base_point / (base_unc * scale + 1e-9)) * 0.1),
            }
        return result

    # ── Conformal 보정 ────────────────────────────────────────────────────
    def _conformal_interval(
        self, point: float, alpha: float = 0.1
    ) -> Tuple[float, float]:
        """
        Split Conformal Prediction.
        calibration_data (캘리브레이션 잔차) 기반 보정.
        데이터 없으면 2σ 근사.
        """
        if self._calib is not None and len(self._calib) >= 10:
            q = np.quantile(np.abs(self._calib), 1 - alpha)
        else:
            # 보정 데이터 없으면 99th percentile 근사 (넓게)
            q = abs(point) * 2.0 + 0.01

        return float(point - q), float(point + q)

    # ── 확률 계산 ─────────────────────────────────────────────────────────
    @staticmethod
    def _prob_threshold(
        lower: float, upper: float, threshold: float, above: bool = True
    ) -> float:
        """
        정규분포 근사로 수익률이 threshold를 초과/미만할 확률 계산.
        """
        if upper == lower:
            return float(threshold < 0)
        # 95% CI → σ 역산
        sigma = (upper - lower) / (2 * 1.645)
        mu    = (upper + lower) / 2
        if sigma <= 0:
            return 0.0
        # P(X > threshold) 또는 P(X < threshold)
        from math import erf, sqrt
        z = (threshold - mu) / (sigma * sqrt(2))
        cdf = 0.5 * (1 + erf(z))
        return float(1 - cdf if above else cdf)

    # ── 변동성 예측 ───────────────────────────────────────────────────────
    @staticmethod
    def _forecast_volatility(ens_result: Dict[str, Any]) -> float:
        """모델 간 분산 → 예측 변동성 추정 (annualized)"""
        preds = list(ens_result.get("individual_preds", {}).values())
        if len(preds) < 2:
            return float("nan")
        return float(np.std([p.mean() for p in preds]) * np.sqrt(252))

    # ── 종합 신호 ─────────────────────────────────────────────────────────
    @staticmethod
    def _compute_signal(
        prob_up: float, uncertainty: float, drift: Dict[str, Any]
    ) -> str:
        """
        prob_up 기반 신호, 불확실성/드리프트가 높으면 한 단계 하향.
        """
        effective_prob = prob_up
        # 불확실성이 높으면 (>0.02) 중립 방향으로 압축
        if uncertainty > 0.02:
            effective_prob = 0.5 + (effective_prob - 0.5) * 0.5
        # Drift 감지 시 추가 압축
        if drift.get("is_drifted"):
            effective_prob = 0.5 + (effective_prob - 0.5) * 0.3

        for label, threshold in SIGNAL_THRESHOLDS:
            if effective_prob >= threshold:
                return label
        return "매우 약세 📉📉"

    # ── 리스크 경고 ───────────────────────────────────────────────────────
    @staticmethod
    def _build_warnings(
        uncertainty: float,
        drift: Dict[str, Any],
        individual: Dict[str, Any],
        prob_up: float,
        target_prob: float,
        stop_prob: float,
    ) -> List[str]:
        warnings = []

        if uncertainty > 0.015:
            warnings.append(
                f"⚠ 예측 불확실성 높음 (σ={uncertainty:.3f}) — 관망 고려"
            )
        if drift.get("is_drifted"):
            ks  = drift.get("ks_stat", 0)
            psi = drift.get("psi", 0)
            warnings.append(
                f"⚠ 분포 변화(Drift) 감지 (KS={ks:.3f}, PSI={psi:.3f}) "
                "— 최근 학습 데이터가 현재와 다른 특성을 보입니다"
            )
        if individual and len(individual) >= 2:
            preds = [float(p.mean()) for p in individual.values()]
            disagreement = np.std(preds) / (np.mean(np.abs(preds)) + 1e-9)
            if disagreement > 0.5:
                warnings.append(
                    f"⚠ 모델 간 의견 불일치 (분산={disagreement:.2f}) "
                    "— 예측 신뢰도 낮음"
                )
        if stop_prob > 0.3:
            warnings.append(
                f"⚠ 손절선 이탈 확률 {stop_prob:.1%} — 리스크 관리 필요"
            )
        if 0.45 < prob_up < 0.55:
            warnings.append("ℹ 방향성 확률이 중립에 가까움 — 강한 신호 없음")

        return warnings

    @staticmethod
    def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return float(max(lo, min(hi, x)))
