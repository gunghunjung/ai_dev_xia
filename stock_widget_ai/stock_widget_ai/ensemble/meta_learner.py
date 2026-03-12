"""
MetaLearner — 스태킹 앙상블 메타 레이어
─────────────────────────────────────────
Level-1 모델들의 예측값을 피처로 사용해
Level-2 (메타) 모델을 학습.

메타 모델 옵션:
  - ridge     : Ridge 회귀 (기본, 과적합 방지)
  - lasso     : Lasso 회귀 (희소 앙상블)
  - lightgbm  : LightGBM (비선형 메타)
  - simple    : 단순 가중 평균 (검증 기반)

⚠ 메타 학습에는 Out-Of-Fold (OOF) 예측을 사용해 누수 방지.
"""
from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from ..logger_config import get_logger

log = get_logger("ensemble.meta")


class MetaLearner:
    """
    Parameters
    ----------
    meta_model_type : "ridge" | "lasso" | "lightgbm" | "simple"
    alpha           : Ridge/Lasso 정규화 강도
    n_folds         : OOF 교차검증 폴드 수
    random_seed     : 재현성 시드
    """

    def __init__(
        self,
        meta_model_type: str = "ridge",
        alpha:           float = 1.0,
        n_folds:         int = 5,
        random_seed:     int = 42,
    ) -> None:
        self._mtype  = meta_model_type
        self._alpha  = alpha
        self._folds  = n_folds
        self._seed   = random_seed
        self._meta:  Optional[Any] = None
        self._model_names: List[str] = []
        self._weights: Dict[str, float] = {}   # simple 방식 가중치

    # ── 학습 ──────────────────────────────────────────────────────────────
    def fit(
        self,
        oof_preds: Dict[str, np.ndarray],   # {model_name: oof_predictions}
        y_true:    np.ndarray,
    ) -> "MetaLearner":
        """
        Out-Of-Fold 예측을 이용해 메타 모델 학습.

        Parameters
        ----------
        oof_preds : 각 Level-1 모델의 OOF 예측 (같은 인덱스)
        y_true    : 실제 라벨
        """
        self._model_names = list(oof_preds.keys())
        X_meta = np.column_stack([oof_preds[n] for n in self._model_names])

        if self._mtype == "simple":
            self._fit_simple(oof_preds, y_true)
        elif self._mtype == "ridge":
            self._meta = self._fit_ridge(X_meta, y_true)
        elif self._mtype == "lasso":
            self._meta = self._fit_lasso(X_meta, y_true)
        elif self._mtype == "lightgbm":
            self._meta = self._fit_lgbm(X_meta, y_true)
        else:
            raise ValueError(f"알 수 없는 메타 모델: {self._mtype}")

        log.info(f"MetaLearner 학습 완료: {self._mtype}, 모델={self._model_names}")
        return self

    # ── 추론 ──────────────────────────────────────────────────────────────
    def predict(self, preds: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Level-1 모델 예측값 → 메타 모델 통합 예측
        """
        names = [n for n in self._model_names if n in preds]
        if not names:
            raise ValueError("등록된 모델 예측 없음")

        if self._mtype == "simple":
            total_w = sum(self._weights.get(n, 0) for n in names)
            if total_w == 0:
                return np.mean([preds[n] for n in names], axis=0)
            return sum(preds[n] * self._weights.get(n, 0) for n in names) / total_w

        X_meta = np.column_stack([preds.get(n, np.zeros_like(list(preds.values())[0]))
                                   for n in self._model_names])
        return np.asarray(self._meta.predict(X_meta), dtype=float)

    # ── OOF 생성 헬퍼 ────────────────────────────────────────────────────
    @staticmethod
    def generate_oof(
        model_factory,    # callable: () → fitted_model 반환
        X: np.ndarray,
        y: np.ndarray,
        n_folds: int = 5,
    ) -> np.ndarray:
        """
        Time-Series CV 방식 OOF 예측 생성.
        미래 데이터 누수 방지를 위해 항상 시간 순서 분할 사용.

        Parameters
        ----------
        model_factory : 새 모델 인스턴스를 반환하는 callable
        Returns : OOF predictions (N,)
        """
        N = len(X)
        oof = np.zeros(N)
        fold_size = N // (n_folds + 1)

        for fold in range(n_folds):
            train_end = fold_size * (fold + 1)
            val_start = train_end
            val_end   = min(val_start + fold_size, N)

            if val_end <= val_start:
                break

            X_tr, y_tr = X[:train_end], y[:train_end]
            X_val       = X[val_start:val_end]

            model = model_factory()
            try:
                model.fit(X_tr, y_tr, X_val, y[val_start:val_end])
                preds = model.predict(X_val)
            except Exception:
                try:
                    model.fit(X_tr, y_tr)
                    preds = model.predict(X_val)
                except Exception as e:
                    log.warning(f"OOF fold {fold} 실패: {e}")
                    preds = np.zeros(val_end - val_start)

            oof[val_start:val_end] = preds

        return oof

    # ── 내부 구현 ─────────────────────────────────────────────────────────
    def _fit_simple(
        self, oof_preds: Dict[str, np.ndarray], y_true: np.ndarray
    ) -> None:
        """검증 RMSE 역수 기반 가중치"""
        from ..ml.metrics import regression_metrics
        for name, pred in oof_preds.items():
            metrics = regression_metrics(y_true, pred)
            rmse = metrics.get("rmse", 1.0)
            self._weights[name] = 1.0 / (rmse + 1e-9)
        total = sum(self._weights.values())
        self._weights = {n: v / total for n, v in self._weights.items()}

    def _fit_ridge(self, X: np.ndarray, y: np.ndarray) -> Any:
        try:
            from sklearn.linear_model import Ridge
            model = Ridge(alpha=self._alpha, random_state=self._seed)
            model.fit(X, y)
            return model
        except ImportError:
            log.warning("sklearn 없음 — simple 방식으로 fallback")
            self._mtype = "simple"
            return None

    def _fit_lasso(self, X: np.ndarray, y: np.ndarray) -> Any:
        try:
            from sklearn.linear_model import Lasso
            model = Lasso(alpha=self._alpha, random_state=self._seed, max_iter=5000)
            model.fit(X, y)
            return model
        except ImportError:
            log.warning("sklearn 없음 — simple 방식으로 fallback")
            self._mtype = "simple"
            return None

    def _fit_lgbm(self, X: np.ndarray, y: np.ndarray) -> Any:
        try:
            from lightgbm import LGBMRegressor
            model = LGBMRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=3,
                num_leaves=15,
                random_state=self._seed,
                verbose=-1,
            )
            model.fit(X, y)
            return model
        except ImportError:
            log.warning("lightgbm 없음 — Ridge fallback")
            self._mtype = "ridge"
            return self._fit_ridge(X, y)

    def model_weights_as_dict(self) -> Dict[str, float]:
        """Ridge/Lasso 의 경우 계수를 가중치로 변환"""
        if self._mtype == "simple":
            return self._weights.copy()
        if self._meta is None or not hasattr(self._meta, "coef_"):
            return {}
        coefs = self._meta.coef_
        pos   = np.clip(coefs, 0, None)   # 음수 제거 (보수적)
        total = pos.sum()
        if total == 0:
            return {n: 1.0 / len(self._model_names) for n in self._model_names}
        return {n: float(pos[i] / total) for i, n in enumerate(self._model_names)}
