"""
LightGBMModel — LightGBM 래퍼
──────────────────────────────
XGBoostModel 과 동일 인터페이스, 더 빠른 학습 속도.
분위수 회귀(objective=quantile) 직접 지원.
"""
from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Any
from ...logger_config import get_logger

log = get_logger("ml.gbm.lightgbm")


class LightGBMModel:

    name = "lightgbm"

    def __init__(
        self,
        task: str = "regression",
        n_estimators: int = 300,
        max_depth: int = -1,
        num_leaves: int = 63,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        early_stopping_rounds: int = 30,
        random_seed: int = 42,
    ) -> None:
        self._task   = task
        self._params = {
            "n_estimators":      n_estimators,
            "max_depth":         max_depth,
            "num_leaves":        num_leaves,
            "learning_rate":     learning_rate,
            "subsample":         subsample,
            "colsample_bytree":  colsample_bytree,
            "reg_alpha":         reg_alpha,
            "reg_lambda":        reg_lambda,
            "random_state":      random_seed,
            "n_jobs":            -1,
            "verbose":           -1,
            "early_stopping_rounds": early_stopping_rounds,
        }
        self._model:  Optional[Any] = None
        self._feature_names: List[str] = []
        self._importances: Dict[str, float] = {}
        self._quantile_models: Dict[float, Any] = {}  # quantile별 별도 모델

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val:   np.ndarray,
        y_val:   np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, List[float]]:
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("lightgbm 패키지가 설치되지 않았습니다: pip install lightgbm")

        self._feature_names = feature_names or [f"f{i}" for i in range(X_train.shape[1])]
        params = {k: v for k, v in self._params.items() if k != "early_stopping_rounds"}

        if self._task == "classification":
            from lightgbm import LGBMClassifier
            self._model = LGBMClassifier(objective="binary", metric="auc", **params)
        else:
            from lightgbm import LGBMRegressor
            self._model = LGBMRegressor(objective="regression", metric="rmse", **params)

        callbacks = [
            lgb.early_stopping(self._params["early_stopping_rounds"], verbose=False),
            lgb.log_evaluation(period=-1),
        ]
        self._model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=callbacks,
            feature_name=self._feature_names,
        )

        scores = self._model.feature_importances_
        self._importances = dict(zip(self._feature_names, [float(s) for s in scores]))
        log.info(f"LightGBM 학습 완료: best_iteration={self._model.best_iteration_}")
        return {}

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("모델이 학습되지 않았습니다")
        return np.asarray(self._model.predict(X), dtype=float)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._task != "classification":
            raise RuntimeError("분류 모델에서만 사용 가능")
        return np.asarray(self._model.predict_proba(X)[:, 1], dtype=float)

    def predict_quantile(
        self, X: np.ndarray, quantiles: List[float]
    ) -> Dict[float, np.ndarray]:
        """LightGBM 분위수 회귀 (objective=quantile) — 분위수별 별도 모델"""
        try:
            import lightgbm as lgb
        except ImportError:
            return {}

        result = {}
        for q in quantiles:
            if q not in self._quantile_models:
                log.warning(f"분위수 {q} 모델 미학습, base prediction 반환")
                result[q] = self.predict(X)
            else:
                result[q] = np.asarray(self._quantile_models[q].predict(X))
        return result

    def fit_quantile_models(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray,   y_val: np.ndarray,
        quantiles: List[float] = [0.1, 0.25, 0.5, 0.75, 0.9],
    ) -> None:
        """분위수별 별도 모델 학습 (선택적)"""
        try:
            import lightgbm as lgb
        except ImportError:
            return
        for q in quantiles:
            model = lgb.LGBMRegressor(
                objective="quantile",
                alpha=q,
                n_estimators=self._params.get("n_estimators", 200),
                learning_rate=self._params.get("learning_rate", 0.05),
                num_leaves=self._params.get("num_leaves", 63),
                verbose=-1,
            )
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                      callbacks=[lgb.log_evaluation(period=-1)])
            self._quantile_models[q] = model
        log.info(f"분위수 모델 {len(quantiles)}개 학습 완료")

    def feature_importance(self) -> Dict[str, float]:
        return dict(sorted(self._importances.items(), key=lambda x: -x[1]))

    def shap_values(self, X: np.ndarray) -> Optional[np.ndarray]:
        try:
            import shap
            explainer = shap.TreeExplainer(self._model)
            return explainer.shap_values(X)
        except Exception as e:
            log.warning(f"SHAP 계산 실패: {e}")
            return None

    @staticmethod
    def optuna_space(trial) -> Dict[str, Any]:
        return {
            "n_estimators":  trial.suggest_int("n_estimators", 100, 800),
            "num_leaves":    trial.suggest_int("num_leaves", 20, 200),
            "max_depth":     trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample":     trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":     trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":    trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        }

    def get_params(self) -> Dict[str, Any]:
        return self._params.copy()
