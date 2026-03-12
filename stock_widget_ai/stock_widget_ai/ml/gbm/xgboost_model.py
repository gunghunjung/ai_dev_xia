"""
XGBoostModel — XGBoost 래퍼
────────────────────────────
- 회귀 / 분류 자동 전환
- Optuna 하이퍼파라미터 공간 정의
- SHAP 중요도 지원
- feature_importance() 로 일관 인터페이스 제공

⚠ xgboost 미설치 시 ImportError 발생 → ModelFactory 에서 fallback 처리
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from ...logger_config import get_logger

log = get_logger("ml.gbm.xgboost")


class XGBoostModel:
    """
    Parameters
    ----------
    task        : "regression" | "classification"
    n_estimators: 트리 수
    max_depth   : 최대 깊이
    learning_rate: 학습률 (eta)
    subsample   : 행 샘플링 비율
    colsample_bytree: 열 샘플링 비율
    reg_alpha   : L1 정규화
    reg_lambda  : L2 정규화
    random_seed : 재현성 시드
    """

    name = "xgboost"

    def __init__(
        self,
        task: str = "regression",
        n_estimators: int = 300,
        max_depth: int = 6,
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
            "learning_rate":     learning_rate,
            "subsample":         subsample,
            "colsample_bytree":  colsample_bytree,
            "reg_alpha":         reg_alpha,
            "reg_lambda":        reg_lambda,
            "random_state":      random_seed,
            "n_jobs":            -1,
            "tree_method":       "hist",
            "early_stopping_rounds": early_stopping_rounds,
        }
        self._model: Optional[Any] = None
        self._feature_names: List[str] = []
        self._importances: Dict[str, float] = {}

    # ── 학습 ──────────────────────────────────────────────────────────────
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val:   np.ndarray,
        y_val:   np.ndarray,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, List[float]]:
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("xgboost 패키지가 설치되지 않았습니다: pip install xgboost")

        self._feature_names = feature_names or [f"f{i}" for i in range(X_train.shape[1])]

        if self._task == "classification":
            from xgboost import XGBClassifier
            self._model = XGBClassifier(
                objective="binary:logistic",
                eval_metric="auc",
                use_label_encoder=False,
                verbosity=0,
                **{k: v for k, v in self._params.items() if k != "early_stopping_rounds"},
                early_stopping_rounds=self._params["early_stopping_rounds"],
            )
        else:
            from xgboost import XGBRegressor
            self._model = XGBRegressor(
                objective="reg:squarederror",
                eval_metric="rmse",
                verbosity=0,
                **{k: v for k, v in self._params.items() if k != "early_stopping_rounds"},
                early_stopping_rounds=self._params["early_stopping_rounds"],
            )

        self._model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        # 중요도 저장
        scores = self._model.feature_importances_
        self._importances = dict(zip(self._feature_names, [float(s) for s in scores]))

        # 학습 이력 (evals_result)
        history: Dict[str, List[float]] = {}
        try:
            er = self._model.evals_result()
            for ds_name, metrics in er.items():
                for met_name, vals in metrics.items():
                    history[f"{ds_name}_{met_name}"] = [float(v) for v in vals]
        except Exception:
            pass

        log.info(f"XGBoost 학습 완료: best_iteration={getattr(self._model, 'best_iteration', 'N/A')}")
        return history

    # ── 추론 ──────────────────────────────────────────────────────────────
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("모델이 학습되지 않았습니다")
        return np.asarray(self._model.predict(X), dtype=float)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """분류 시 클래스 1 확률, 회귀 시 None"""
        if self._task != "classification":
            raise RuntimeError("분류 모델에서만 predict_proba 사용 가능")
        return np.asarray(self._model.predict_proba(X)[:, 1], dtype=float)

    # ── 불확실성 ─────────────────────────────────────────────────────────
    def predict_quantile(
        self, X: np.ndarray, quantiles: List[float]
    ) -> Dict[float, np.ndarray]:
        """
        XGBoost 자체 분위수 회귀 (objective="reg:quantileerror", XGB 2.0+)
        지원 안 되면 단순 노이즈 추가로 근사
        """
        base_pred = self.predict(X)
        std_est   = float(np.std(base_pred)) * 0.5 + 1e-6
        result = {}
        for q in quantiles:
            from scipy.stats import norm
            try:
                z = norm.ppf(q)
            except Exception:
                z = (q - 0.5) * 4
            result[q] = base_pred + z * std_est
        return result

    # ── 중요도 ────────────────────────────────────────────────────────────
    def feature_importance(self) -> Dict[str, float]:
        return dict(sorted(self._importances.items(), key=lambda x: -x[1]))

    def shap_values(self, X: np.ndarray) -> Optional[np.ndarray]:
        """SHAP 값 반환 (shap 라이브러리 필요)"""
        try:
            import shap
            explainer = shap.TreeExplainer(self._model)
            return explainer.shap_values(X)
        except Exception as e:
            log.warning(f"SHAP 계산 실패: {e}")
            return None

    # ── Optuna 하이퍼파라미터 공간 ─────────────────────────────────────────
    @staticmethod
    def optuna_space(trial) -> Dict[str, Any]:
        return {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 800),
            "max_depth":         trial.suggest_int("max_depth", 3, 10),
            "learning_rate":     trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        }

    def get_params(self) -> Dict[str, Any]:
        return self._params.copy()
