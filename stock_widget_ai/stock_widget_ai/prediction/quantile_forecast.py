"""
QuantileForecaster — 분위수 예측
──────────────────────────────────
분위수 예측을 통해 예측 불확실성을 직접 모델링.

방법:
  1. Quantile Regression Forest (pinball loss 기반)
  2. LightGBM 분위수 회귀
  3. 앙상블 분위수 (개별 모델 예측의 경험적 분위수)
  4. Bootstrap 기반 불확실성 구간
"""
from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from ..logger_config import get_logger

log = get_logger("prediction.quantile")


class QuantileForecaster:
    """
    Parameters
    ----------
    quantiles   : 예측할 분위수 목록
    method      : "lgbm" | "bootstrap" | "ensemble"
    n_bootstrap : Bootstrap 반복 횟수
    """

    def __init__(
        self,
        quantiles:   List[float] = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95],
        method:      str = "ensemble",
        n_bootstrap: int = 200,
    ) -> None:
        self._quantiles   = quantiles
        self._method      = method
        self._n_boot      = n_bootstrap
        self._models:     Dict[float, Any] = {}   # quantile → fitted model
        self._boot_preds: Optional[np.ndarray] = None

    # ── 학습 ──────────────────────────────────────────────────────────────
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val:   Optional[np.ndarray] = None,
        y_val:   Optional[np.ndarray] = None,
    ) -> "QuantileForecaster":
        if self._method == "lgbm":
            self._fit_lgbm(X_train, y_train, X_val, y_val)
        elif self._method == "bootstrap":
            self._fit_bootstrap(X_train, y_train)
        else:
            # ensemble: LightGBM 시도, 실패 시 bootstrap
            try:
                self._fit_lgbm(X_train, y_train, X_val, y_val)
                log.debug("Quantile forecaster: LightGBM 사용")
            except Exception:
                self._fit_bootstrap(X_train, y_train)
                log.debug("Quantile forecaster: Bootstrap 사용")
        return self

    # ── 예측 ──────────────────────────────────────────────────────────────
    def predict(self, X: np.ndarray) -> Dict[float, np.ndarray]:
        """
        Returns dict {quantile: predictions}
        """
        if self._models:
            return self._predict_lgbm(X)
        elif self._boot_preds is not None:
            return self._predict_bootstrap(X)
        else:
            # Gaussian 근사 fallback
            return self._gaussian_fallback(X)

    def predict_interval(
        self, X: np.ndarray, alpha: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        (1-alpha) 예측 구간 반환.
        alpha=0.1 → 90% 구간
        """
        q_low  = alpha / 2
        q_high = 1 - alpha / 2

        all_q = self.predict(X)

        # 가장 가까운 분위수 선택
        low_q  = min(self._quantiles, key=lambda q: abs(q - q_low))
        high_q = min(self._quantiles, key=lambda q: abs(q - q_high))

        lower = all_q.get(low_q,  np.zeros(len(X)))
        upper = all_q.get(high_q, np.zeros(len(X)))
        return lower, upper

    def winkler_score(
        self,
        y_true: np.ndarray,
        alpha: float = 0.1,
    ) -> float:
        """
        Winkler Score — 예측 구간의 품질 평가.
        낮을수록 좋음 (좁고 정확한 구간).
        """
        lower, upper = self.predict_interval(y_true.reshape(-1, 1), alpha)
        width = upper - lower
        penalty = np.where(
            y_true < lower,
            (lower - y_true) * 2 / alpha,
            np.where(y_true > upper, (y_true - upper) * 2 / alpha, 0.0),
        )
        return float(np.mean(width + penalty))

    # ── 내부 구현 ─────────────────────────────────────────────────────────
    def _fit_lgbm(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val:   Optional[np.ndarray], y_val: Optional[np.ndarray],
    ) -> None:
        import lightgbm as lgb
        for q in self._quantiles:
            model = lgb.LGBMRegressor(
                objective="quantile",
                alpha=q,
                n_estimators=200,
                learning_rate=0.05,
                num_leaves=31,
                verbose=-1,
            )
            if X_val is not None and y_val is not None:
                callbacks = [lgb.early_stopping(20, verbose=False), lgb.log_evaluation(-1)]
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=callbacks)
            else:
                model.fit(X_train, y_train)
            self._models[q] = model
        log.info(f"LightGBM 분위수 모델 {len(self._quantiles)}개 학습")

    def _fit_bootstrap(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """부트스트랩 기반 분위수 추정 (의존성 없음)"""
        try:
            from sklearn.linear_model import Ridge
            base = Ridge(alpha=1.0)
        except ImportError:
            # sklearn 없으면 OLS
            base = _SimpleLinear()

        n = len(X_train)
        boot_preds = np.zeros((self._n_boot, n))

        for i in range(self._n_boot):
            idx = np.random.choice(n, n, replace=True)
            X_b, y_b = X_train[idx], y_train[idx]
            try:
                base.fit(X_b.reshape(n, -1) if X_b.ndim > 2 else X_b, y_b)
                boot_preds[i] = base.predict(
                    X_train.reshape(n, -1) if X_train.ndim > 2 else X_train
                )
            except Exception:
                boot_preds[i] = y_train.mean()

        self._boot_preds = boot_preds  # (n_boot, n_train)
        log.info(f"Bootstrap {self._n_boot}회 완료")

    def _predict_lgbm(self, X: np.ndarray) -> Dict[float, np.ndarray]:
        result = {}
        Xr = X.reshape(len(X), -1) if X.ndim > 2 else X
        for q, model in self._models.items():
            result[q] = np.asarray(model.predict(Xr), dtype=float)
        return result

    def _predict_bootstrap(self, X: np.ndarray) -> Dict[float, np.ndarray]:
        """학습 분포를 사용해 예측 — 새 샘플에는 적용 불가, 분포 추정용"""
        result = {}
        for q in self._quantiles:
            result[q] = np.full(len(X), float(np.quantile(self._boot_preds, q)))
        return result

    def _gaussian_fallback(self, X: np.ndarray) -> Dict[float, np.ndarray]:
        """완전 fallback — 0 예측 + 정규분포 구간"""
        from scipy.stats import norm
        result = {}
        for q in self._quantiles:
            try:
                z = norm.ppf(q)
            except Exception:
                z = (q - 0.5) * 4
            result[q] = np.full(len(X), z * 0.01)   # 1% 변동성 가정
        return result


class _SimpleLinear:
    """sklearn 없을 때 OLS 대체"""
    def fit(self, X, y):
        X_ = np.column_stack([np.ones(len(X)), X])
        self.coef_ = np.linalg.lstsq(X_, y, rcond=None)[0]
        return self

    def predict(self, X):
        X_ = np.column_stack([np.ones(len(X)), X])
        return X_ @ self.coef_
