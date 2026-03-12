"""
데이터 전처리 — 결측 처리, train/val/test 분할, 정규화
sklearn 없이 동작하는 자체 RobustScaler 사용
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple, Dict
from ..logger_config import get_logger

log = get_logger("data.preprocessing")


class _RobustScaler:
    """중앙값 + IQR 기반 스케일러 (sklearn 미설치 환경 대응)"""
    def __init__(self):
        self.center_ = 0.0
        self.scale_  = 1.0

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        v = X.ravel()
        self.center_ = float(np.median(v))
        q75, q25 = np.percentile(v, [75, 25])
        self.scale_ = max(float(q75 - q25), 1e-9)
        return ((v - self.center_) / self.scale_).reshape(-1, 1)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((X.ravel() - self.center_) / self.scale_).reshape(-1, 1)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return (X.ravel() * self.scale_ + self.center_).reshape(-1, 1)


class Preprocessor:
    def __init__(self) -> None:
        self._scalers: Dict[str, _RobustScaler] = {}

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """결측 처리 + 이상값 클리핑"""
        df = df.copy()
        df = df.ffill().bfill()
        for col in df.select_dtypes(include=np.number).columns:
            mu, sigma = df[col].mean(), df[col].std()
            if sigma > 0:
                df[col] = df[col].clip(mu - 5 * sigma, mu + 5 * sigma)
        nan_ratio = df.isna().mean()
        bad_cols = nan_ratio[nan_ratio > 0.3].index.tolist()
        if bad_cols:
            log.warning(f"NaN 비율 >30%로 드롭: {bad_cols}")
            df = df.drop(columns=bad_cols)
        df = df.dropna()
        return df

    def split(
        self,
        df: pd.DataFrame,
        val_ratio: float = 0.15,
        test_ratio: float = 0.10,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        n = len(df)
        test_n = max(1, int(n * test_ratio))
        val_n  = max(1, int(n * val_ratio))
        train_end = n - val_n - test_n
        train = df.iloc[:train_end]
        val   = df.iloc[train_end: train_end + val_n]
        test  = df.iloc[train_end + val_n:]
        log.info(f"split: train={len(train)}, val={len(val)}, test={len(test)}")
        return train, val, test

    def fit_transform(self, df: pd.DataFrame, cols: list) -> pd.DataFrame:
        df = df.copy()
        for col in cols:
            if col not in df.columns:
                continue
            scaler = _RobustScaler()
            df[col] = scaler.fit_transform(df[[col]].values).ravel()
            self._scalers[col] = scaler
        return df

    def transform(self, df: pd.DataFrame, cols: list) -> pd.DataFrame:
        df = df.copy()
        for col in cols:
            if col in self._scalers and col in df.columns:
                df[col] = self._scalers[col].transform(df[[col]].values).ravel()
        return df

    def inverse_transform_col(self, col: str, values: np.ndarray) -> np.ndarray:
        if col in self._scalers:
            return self._scalers[col].inverse_transform(values.reshape(-1, 1)).ravel()
        return values
