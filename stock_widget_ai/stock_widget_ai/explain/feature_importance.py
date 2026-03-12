"""
FeatureImportanceAnalyzer — 피처 중요도 종합 분석
──────────────────────────────────────────────────
다양한 방법으로 피처 중요도를 계산하고 통합:
  1. 모델 내장 중요도 (XGBoost, LightGBM 등)
  2. Permutation Importance (모델 무관)
  3. SHAP 기반 중요도 (별도 계산)
  4. 상관 기반 중요도
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from ..logger_config import get_logger

log = get_logger("explain.importance")


class FeatureImportanceAnalyzer:
    """
    Parameters
    ----------
    feature_names : 피처 이름 목록
    """

    def __init__(self, feature_names: List[str]) -> None:
        self._names = feature_names
        self._results: Dict[str, pd.DataFrame] = {}   # method → importance df

    # ── 모델 내장 중요도 ─────────────────────────────────────────────────
    def from_model(
        self, model: Any, method_name: str = "model"
    ) -> pd.DataFrame:
        """XGBoost / LightGBM / RandomForest 내장 중요도"""
        if hasattr(model, "feature_importances_"):
            scores = model.feature_importances_
        elif hasattr(model, "feature_importance"):
            scores = model.feature_importance()
            if isinstance(scores, dict):
                scores = [scores.get(n, 0.0) for n in self._names]
        else:
            log.warning("모델에 feature_importances_ 없음")
            return pd.DataFrame()

        scores = np.asarray(scores, dtype=float)
        scores = scores / (scores.sum() + 1e-9)

        df = self._make_df(scores, method_name)
        self._results[method_name] = df
        return df

    # ── Permutation Importance ────────────────────────────────────────────
    def permutation(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        n_repeats: int = 10,
        metric: str = "rmse",
    ) -> pd.DataFrame:
        """
        피처 값을 무작위로 섞었을 때 성능 저하 → 중요도.
        모델 인터페이스에 독립적.
        """
        from ..ml.metrics import regression_metrics

        baseline = self._score(model, X, y, metric)
        n_feat = X.shape[-1]
        importances = np.zeros(n_feat)

        Xf = X.reshape(len(X), -1)   # flatten if needed

        for feat_i in range(n_feat):
            drops = []
            for _ in range(n_repeats):
                X_perm = Xf.copy()
                X_perm[:, feat_i] = np.random.permutation(X_perm[:, feat_i])
                perm_score = self._score(model, X_perm, y, metric)
                drops.append(perm_score - baseline)
            importances[feat_i] = max(np.mean(drops), 0)

        total = importances.sum() + 1e-9
        importances /= total

        df = self._make_df(importances[:len(self._names)], "permutation")
        self._results["permutation"] = df
        log.info(f"Permutation importance 완료: top={df.iloc[0]['feature'] if not df.empty else 'N/A'}")
        return df

    # ── 상관 기반 중요도 ─────────────────────────────────────────────────
    def correlation_based(
        self, X: np.ndarray, y: np.ndarray
    ) -> pd.DataFrame:
        """라벨과 각 피처의 |Pearson r| 기반 중요도"""
        Xf = X.reshape(len(X), -1) if X.ndim > 2 else X
        n_feat = min(Xf.shape[1], len(self._names))
        scores = np.zeros(n_feat)
        for i in range(n_feat):
            try:
                r = float(np.corrcoef(Xf[:, i], y)[0, 1])
                scores[i] = abs(r) if not np.isnan(r) else 0.0
            except Exception:
                pass
        total = scores.sum() + 1e-9
        df = self._make_df(scores / total, "correlation")
        self._results["correlation"] = df
        return df

    # ── 종합 중요도 ───────────────────────────────────────────────────────
    def aggregate(
        self,
        weights: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        """
        여러 방법의 중요도를 가중 평균으로 통합.
        """
        if not self._results:
            return pd.DataFrame()

        default_w = 1.0 / len(self._results)
        w = weights or {k: default_w for k in self._results}

        names = self._names
        agg = np.zeros(len(names))

        for method, df in self._results.items():
            wt = w.get(method, default_w)
            feat_imp = dict(zip(df["feature"], df["importance"]))
            for i, n in enumerate(names):
                agg[i] += wt * feat_imp.get(n, 0.0)

        total = agg.sum() + 1e-9
        df = self._make_df(agg / total, "aggregate")
        return df

    def top_k(self, k: int = 10, method: str = "aggregate") -> List[str]:
        """상위 k개 피처 이름"""
        if method == "aggregate":
            df = self.aggregate()
        else:
            df = self._results.get(method, pd.DataFrame())
        if df.empty:
            return self._names[:k]
        return df.head(k)["feature"].tolist()

    # ── 내부 ──────────────────────────────────────────────────────────────
    def _make_df(self, scores: np.ndarray, method: str) -> pd.DataFrame:
        n = min(len(scores), len(self._names))
        df = pd.DataFrame({
            "feature":    self._names[:n],
            "importance": [float(s) for s in scores[:n]],
            "method":     method,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        df["rank"] = range(1, len(df) + 1)
        return df

    @staticmethod
    def _score(model: Any, X: np.ndarray, y: np.ndarray, metric: str) -> float:
        from ..ml.metrics import regression_metrics
        try:
            if hasattr(model, "predict"):
                preds = model.predict(X)
            else:
                return 0.0
            m = regression_metrics(y, np.asarray(preds, dtype=float).ravel())
            return m.get(metric, m.get("rmse", 0.0))
        except Exception:
            return 0.0
