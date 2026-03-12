"""
Feature Engineering — 가격/수익률 + 기술지표 + 라벨 생성
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from .technical_indicators import add_all_ta
from ..logger_config import get_logger

log = get_logger("features.engineering")

FEATURE_GROUPS = {
    "price":  ["ret", "log_ret", "ret5", "ret20", "mom10", "mom20",
               "rv5", "rv20", "z5", "z20"],
    "ta":     ["sma5","sma20","sma60","ema20","bb_width","bb_pct",
               "rsi14","rsi7","macd","macd_signal","macd_hist",
               "atr14","obv","cci20","adx14","stoch_k","stoch_d"],
    "market": ["bm_return", "rel_strength"],
}


class FeatureEngineer:
    def __init__(self, groups: Optional[List[str]] = None) -> None:
        self._groups = groups or list(FEATURE_GROUPS.keys())
        self._feature_cols: List[str] = []

    def build(
        self,
        df: pd.DataFrame,
        target_type: str = "return",
        horizon: int = 5,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Returns
        -------
        X : feature DataFrame
        y : target Series
        """
        df = df.copy()

        # 기술지표 추가
        df = add_all_ta(df)

        # 가격/수익률 피처
        if "price" in self._groups:
            df = self._add_price_features(df)

        # 시장 비교 피처
        if "market" in self._groups:
            df = self._add_market_features(df)

        # 라벨
        y = self._make_label(df, target_type, horizon)

        # 피처 컬럼 결정
        feat_cols: List[str] = []
        for g in self._groups:
            for col in FEATURE_GROUPS.get(g, []):
                if col in df.columns:
                    feat_cols.append(col)

        # 중복 제거 + 존재 확인
        feat_cols = list(dict.fromkeys(feat_cols))
        feat_cols = [c for c in feat_cols if c in df.columns]
        self._feature_cols = feat_cols

        X = df[feat_cols]

        # 라벨과 피처를 같은 인덱스로 정렬 후 NaN 제거
        combined = pd.concat([X, y], axis=1).dropna()
        X = combined.iloc[:, :-1]
        y = combined.iloc[:, -1]

        log.info(f"features: {len(feat_cols)}개, samples: {len(X)}")
        return X, y

    def feature_cols(self) -> List[str]:
        return self._feature_cols.copy()

    # ── 내부 메서드 ──────────────────────────────────────────────────
    @staticmethod
    def _add_price_features(df: pd.DataFrame) -> pd.DataFrame:
        c = df["close"]
        df["ret"]    = c.pct_change()
        df["log_ret"]= np.log(c / c.shift(1))
        df["ret5"]   = c.pct_change(5)
        df["ret20"]  = c.pct_change(20)
        df["mom10"]  = c / c.shift(10) - 1
        df["mom20"]  = c / c.shift(20) - 1
        df["rv5"]    = df["ret"].rolling(5).std()
        df["rv20"]   = df["ret"].rolling(20).std()
        df["z5"]     = (c - c.rolling(5).mean()) / (c.rolling(5).std() + 1e-9)
        df["z20"]    = (c - c.rolling(20).mean()) / (c.rolling(20).std() + 1e-9)
        return df

    @staticmethod
    def _add_market_features(df: pd.DataFrame) -> pd.DataFrame:
        if "bm_return" not in df.columns:
            df["bm_return"] = 0.0
        ret = df.get("ret", df["close"].pct_change())
        bm  = df["bm_return"]
        df["rel_strength"] = ret - bm
        return df

    @staticmethod
    def _make_label(
        df: pd.DataFrame, target_type: str, horizon: int
    ) -> pd.Series:
        c = df["close"]
        if target_type == "close":
            return c.shift(-horizon).rename("label")
        elif target_type == "return":
            return c.pct_change(horizon).shift(-horizon).rename("label")
        elif target_type == "direction":
            ret = c.pct_change(horizon).shift(-horizon)
            # ⚠️ (NaN > 0) → False → 0.0 변환 방지:
            # shift(-horizon)으로 인한 미래 구간 NaN은 float()로 캐스팅 시
            # False(=0)로 변환되어 "하락" 레이블이 되고 dropna()에도 걸리지 않음.
            # → NaN을 명시적으로 보존해서 dropna()가 미래 구간을 제거하도록 한다.
            label = (ret > 0).astype(float)
            label[ret.isna()] = np.nan
            return label.rename("label")
        else:
            raise ValueError(f"Unknown target_type: {target_type}")
