"""
UncertaintyEstimator — 예측 불확실성 정량화
─────────────────────────────────────────────
방법:
  1. MC Dropout       : Dropout을 추론 시에도 활성화, N회 샘플링
  2. Conformal        : 캘리브레이션 잔차 기반 분포 무가정 구간
  3. Quantile Ensemble: 앙상블 분위수
  4. GARCH-like Vol   : 잔차 기반 변동성 추정
"""
from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from ..logger_config import get_logger

log = get_logger("prediction.uncertainty")


class UncertaintyEstimator:
    """
    Parameters
    ----------
    method     : "mc_dropout" | "conformal" | "ensemble" | "all"
    n_samples  : MC Dropout 샘플 수
    alpha      : Conformal prediction 오류율 (1-alpha = 신뢰수준)
    """

    def __init__(
        self,
        method:    str = "all",
        n_samples: int = 50,
        alpha:     float = 0.1,
    ) -> None:
        self._method   = method
        self._n        = n_samples
        self._alpha    = alpha
        self._calib_residuals: Optional[np.ndarray] = None

    # ── 캘리브레이션 ──────────────────────────────────────────────────────
    def calibrate(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> "UncertaintyEstimator":
        """
        검증셋 잔차로 Conformal Prediction 보정.
        ⚠ 반드시 학습에 사용하지 않은 데이터로 보정.
        """
        residuals = np.abs(y_true - y_pred)
        self._calib_residuals = residuals
        log.info(
            f"Conformal calibration: n={len(residuals)}, "
            f"q{100*(1-self._alpha):.0f}={np.quantile(residuals, 1-self._alpha):.4f}"
        )
        return self

    # ── MC Dropout ────────────────────────────────────────────────────────
    def mc_dropout(
        self,
        model: Any,
        X: np.ndarray,
        n_samples: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """
        MC Dropout으로 예측 분포 추정.
        모델의 Dropout 레이어를 추론 시에도 활성화.

        Returns
        -------
        dict: mean, std, q10, q50, q90, samples
        """
        n = n_samples or self._n
        try:
            import torch
            if not isinstance(model, torch.nn.Module):
                raise TypeError("PyTorch 모델 필요")

            model.train()   # Dropout 활성화
            preds = []
            xt = torch.tensor(X, dtype=torch.float32)
            with torch.no_grad():
                for _ in range(n):
                    out = model(xt)
                    preds.append(out.cpu().numpy())
            model.eval()

            preds = np.stack(preds, axis=0)  # (n_samples, B, ...)
            mean  = preds.mean(axis=0)
            std   = preds.std(axis=0)
            return {
                "mean":    mean,
                "std":     std,
                "q10":     np.quantile(preds, 0.10, axis=0),
                "q50":     np.quantile(preds, 0.50, axis=0),
                "q90":     np.quantile(preds, 0.90, axis=0),
                "samples": preds,
            }
        except Exception as e:
            log.warning(f"MC Dropout 실패: {e}")
            dummy = np.zeros(len(X))
            return {"mean": dummy, "std": dummy + 1e-6,
                    "q10": dummy, "q50": dummy, "q90": dummy, "samples": dummy[np.newaxis]}

    # ── Conformal Prediction ───────────────────────────────────────────────
    def conformal_interval(
        self,
        y_pred: np.ndarray,
        alpha:  Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split Conformal Prediction interval.
        requires calibrate() to be called first.

        Returns (lower, upper) arrays.
        """
        a = alpha or self._alpha
        if self._calib_residuals is not None and len(self._calib_residuals) >= 5:
            # Conformal quantile (coverage guarantee: 1-alpha)
            n    = len(self._calib_residuals)
            q_level = np.ceil((n + 1) * (1 - a)) / n
            q_level = min(q_level, 1.0)
            margin = float(np.quantile(self._calib_residuals, q_level))
        else:
            # Fallback: 2σ 근사
            margin = float(np.std(y_pred)) * 2.0 + 1e-4

        return y_pred - margin, y_pred + margin

    # ── GARCH-like 변동성 ─────────────────────────────────────────────────
    @staticmethod
    def garch_vol_estimate(
        residuals: np.ndarray,
        omega: float = 1e-6,
        alpha_g: float = 0.1,
        beta_g: float = 0.85,
        horizon: int = 5,
    ) -> np.ndarray:
        """
        GARCH(1,1) 변동성 예측.
        잔차 배열로부터 향후 horizon 스텝 변동성 예측.
        """
        r = np.asarray(residuals, dtype=float)
        h = np.zeros(len(r) + horizon)
        h[0] = np.var(r[:10]) if len(r) >= 10 else 0.01

        for t in range(1, len(r)):
            h[t] = omega + alpha_g * r[t-1]**2 + beta_g * h[t-1]

        # 향후 예측
        last_h = h[len(r) - 1]
        last_e = r[-1] ** 2 if len(r) > 0 else 0.0
        long_run = omega / max(1 - alpha_g - beta_g, 1e-9)
        for t in range(len(r), len(r) + horizon):
            last_h = omega + alpha_g * last_e + beta_g * last_h
            last_e = last_h   # E[e²] = h 로 근사
            h[t] = last_h

        return np.sqrt(h[-horizon:]) * np.sqrt(252)   # 연율화

    # ── 불확실성 점수 (0~1) ───────────────────────────────────────────────
    def uncertainty_score(
        self,
        std_pred:  float,
        hist_vol:  float,
        drift_psi: float = 0.0,
    ) -> float:
        """
        예측 불확실성을 0~1 점수로 변환.
        높을수록 = 더 불확실.
        """
        # 정규화: 예측 표준편차 / 과거 변동성
        rel_unc = min(std_pred / (hist_vol + 1e-9), 3.0) / 3.0
        # PSI 기반 드리프트 패널티
        drift_penalty = min(drift_psi / 0.5, 1.0) * 0.3
        return float(min(rel_unc * 0.7 + drift_penalty, 1.0))

    # ── 보정 품질 검사 ────────────────────────────────────────────────────
    def calibration_score(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_bins: int = 10,
    ) -> Dict[str, float]:
        """
        예측값의 캘리브레이션 품질.
        이상적: Expected Calibration Error (ECE) → 0
        """
        abs_err   = np.abs(y_true - y_pred)
        std_pred  = np.std(y_pred) + 1e-9
        z_scores  = abs_err / std_pred

        # Reliability (실제 커버리지)
        coverages = {}
        for sigma in [1, 2, 3]:
            covered = np.mean(z_scores <= sigma)
            expected = {1: 0.683, 2: 0.954, 3: 0.997}[sigma]
            coverages[f"coverage_{sigma}sigma"] = float(covered)
            coverages[f"expected_{sigma}sigma"] = expected

        ece = np.mean(
            [abs(coverages[f"coverage_{s}sigma"] - coverages[f"expected_{s}sigma"])
             for s in [1, 2, 3]]
        )
        coverages["ece"] = float(ece)
        return coverages
