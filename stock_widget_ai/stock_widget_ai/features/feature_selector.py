"""
FeatureSelector — 중요도 기반 피처 선택 + 누수 탐지
────────────────────────────────────────────────────
- 상관관계 기반 중복 제거
- 분산 필터 (저분산 피처 제거)
- RandomForest 기반 중요도 선택
- 미래정보 누수 탐지 (라벨과 높은 상관 = 누수 의심)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from ..logger_config import get_logger

log = get_logger("features.selector")


class FeatureSelector:
    """
    Parameters
    ----------
    var_threshold   : 분산이 이 값 미만이면 제거
    corr_threshold  : 피처 간 상관 > 임계값이면 하나 제거
    top_k           : 중요도 상위 k개만 유지 (None이면 전체)
    leakage_corr    : 라벨과 이 이상 상관 시 누수 경고
    """

    def __init__(
        self,
        var_threshold: float = 1e-5,
        corr_threshold: float = 0.95,
        top_k: Optional[int] = None,
        leakage_corr: float = 0.8,
    ) -> None:
        self._var_thr  = var_threshold
        self._corr_thr = corr_threshold
        self._top_k    = top_k
        self._leak_cor = leakage_corr
        self._selected: List[str] = []
        self._importances: Dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FeatureSelector":
        """피처 선택 기준 학습"""
        cols = list(X.columns)

        # 1) 누수 탐지
        self._check_leakage(X, y)

        # 2) 분산 필터
        cols = self._filter_low_variance(X, cols)

        # 3) 상관관계 중복 제거
        cols = self._filter_correlated(X[cols])

        # 4) 중요도 기반 선택
        cols, importances = self._importance_filter(X[cols], y, cols)
        self._selected = cols
        self._importances = importances

        log.info(f"피처 선택: {len(X.columns)}개 → {len(cols)}개")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """선택된 피처만 반환"""
        available = [c for c in self._selected if c in X.columns]
        return X[available]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    @property
    def selected_features(self) -> List[str]:
        return self._selected.copy()

    @property
    def importances(self) -> Dict[str, float]:
        return dict(sorted(self._importances.items(), key=lambda x: -x[1]))

    # ── 내부 메서드 ───────────────────────────────────────────────────────

    def _check_leakage(self, X: pd.DataFrame, y: pd.Series) -> None:
        """라벨과 높은 상관을 가진 피처 경고 (미래정보 누수 의심)"""
        suspect = []
        for col in X.columns:
            try:
                c = abs(float(X[col].corr(y)))
                if c > self._leak_cor:
                    suspect.append((col, c))
            except Exception:
                pass
        if suspect:
            for col, c in suspect:
                log.warning(
                    f"⚠ 누수 의심: '{col}' — 라벨 상관={c:.3f} "
                    f"(> {self._leak_cor}). 미래정보 참조 여부 확인 필요!"
                )

    def _filter_low_variance(self, X: pd.DataFrame, cols: List[str]) -> List[str]:
        kept = [c for c in cols if X[c].var() >= self._var_thr]
        removed = len(cols) - len(kept)
        if removed:
            log.debug(f"저분산 피처 {removed}개 제거")
        return kept

    def _filter_correlated(self, X: pd.DataFrame) -> List[str]:
        """높은 상관을 가진 피처 쌍에서 하나 제거"""
        corr = X.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = set()
        for col in upper.columns:
            if any(upper[col] > self._corr_thr):
                to_drop.add(col)
        kept = [c for c in X.columns if c not in to_drop]
        if to_drop:
            log.debug(f"고상관 피처 {len(to_drop)}개 제거")
        return kept

    def _importance_filter(
        self, X: pd.DataFrame, y: pd.Series, cols: List[str]
    ) -> Tuple[List[str], Dict[str, float]]:
        """RandomForest 또는 분산으로 중요도 계산"""
        importances: Dict[str, float] = {}
        try:
            from sklearn.ensemble import RandomForestRegressor
            rf = RandomForestRegressor(
                n_estimators=50, max_depth=5, random_state=42, n_jobs=-1
            )
            Xv = X.fillna(0).values
            yv = y.fillna(0).values
            rf.fit(Xv, yv)
            for col, imp in zip(cols, rf.feature_importances_):
                importances[col] = float(imp)
        except Exception:
            # sklearn 없거나 실패 시 분산으로 대체
            for col in cols:
                importances[col] = float(X[col].var())
            total = sum(importances.values()) or 1.0
            importances = {k: v / total for k, v in importances.items()}

        if self._top_k and len(cols) > self._top_k:
            sorted_cols = sorted(importances, key=lambda c: -importances[c])
            cols = sorted_cols[: self._top_k]
            log.info(f"중요도 Top-{self._top_k} 피처 유지")

        return cols, importances

    def importance_df(self) -> pd.DataFrame:
        """중요도 DataFrame 반환 (GUI 표 출력용)"""
        df = pd.DataFrame(
            {"feature": list(self._importances.keys()),
             "importance": list(self._importances.values())}
        ).sort_values("importance", ascending=False).reset_index(drop=True)
        df["rank"] = range(1, len(df) + 1)
        return df
