# signals/generator.py — 확률적 신호 생성
from __future__ import annotations
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger("quant.signal")


class SignalGenerator:
    """
    모델 출력 → 트레이딩 신호 변환

    방법:
    1. 횡단면 랭킹(Cross-sectional ranking):
       모든 종목의 기댓값을 랭킹하여 상위 K% → 롱, 하위 K% → 숏
    2. 임계값(Threshold):
       개별 종목의 신뢰 구간이 threshold를 초과하면 신호 발생
    3. 확률 가중(Probability weighted):
       μ/σ 비율(Sharpe-like)로 신호 강도 결정
    """

    def __init__(
        self,
        method: str = "ranking",
        long_pct: float = 0.3,        # 상위 30% → 롱
        short_pct: float = 0.3,       # 하위 30% → 숏 (없으면 현금)
        threshold: float = 0.005,     # 기댓값 임계값 (0.5%)
        min_confidence: float = 0.3,  # 최소 신뢰도 (μ/σ)
    ):
        self.method = method
        self.long_pct = long_pct
        self.short_pct = short_pct
        self.threshold = threshold
        self.min_confidence = min_confidence

    def generate(
        self,
        mu: np.ndarray,           # 기댓값 배열 (N,)
        sigma: np.ndarray,        # 불확실성 배열 (N,)
        symbols: List[str],
    ) -> pd.DataFrame:
        """
        신호 생성
        Returns:
            DataFrame with columns: [symbol, mu, sigma, confidence, signal, weight]
            signal: 1=롱, -1=숏, 0=중립
            weight: 포지션 가중치 (절댓값 합 = 1)
        """
        n = len(symbols)
        if len(mu) != n or len(sigma) != n:
            raise ValueError(
                f"mu({len(mu)}), sigma({len(sigma)}), symbols({n}) 길이 불일치"
            )
        if n == 0:
            return pd.DataFrame(
                columns=["symbol", "mu", "sigma", "confidence", "signal", "weight"]
            )

        # 신뢰도 점수: μ/σ (Sharpe-like)
        confidence = mu / (sigma + 1e-10)

        records = []
        for i, sym in enumerate(symbols):
            records.append({
                "symbol": sym,
                "mu": float(mu[i]),
                "sigma": float(sigma[i]),
                "confidence": float(confidence[i]),
            })
        df = pd.DataFrame(records)

        if self.method == "ranking":
            df = self._ranking_signal(df)
        elif self.method == "threshold":
            df = self._threshold_signal(df)
        elif self.method == "prob_weight":
            df = self._prob_weight_signal(df)
        else:
            df = self._ranking_signal(df)

        logger.debug(f"신호 생성: {df['signal'].value_counts().to_dict()}")
        return df

    def generate_from_dict(
        self,
        preds: Dict[str, Tuple[float, float]]  # symbol → (mu, sigma)
    ) -> pd.DataFrame:
        """딕셔너리 형태로 신호 생성"""
        symbols = list(preds.keys())
        mu = np.array([preds[s][0] for s in symbols])
        sigma = np.array([preds[s][1] for s in symbols])
        return self.generate(mu, sigma, symbols)

    # ──────────────────────────────────────────────────
    # 신호 방법
    # ──────────────────────────────────────────────────

    def _ranking_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        """횡단면 랭킹 기반 신호"""
        n = len(df)
        df = df.sort_values("mu", ascending=False).reset_index(drop=True)

        n_long = max(1, int(n * self.long_pct))
        n_short = max(0, int(n * self.short_pct))

        signals = np.zeros(n, dtype=int)
        weights = np.zeros(n, dtype=float)

        # 상위 → 롱
        for i in range(n_long):
            if df.loc[i, "confidence"] >= self.min_confidence:
                signals[i] = 1

        # 하위 → 숏 (매도 전략 포함 시)
        for i in range(n - n_short, n):
            if df.loc[i, "confidence"] <= -self.min_confidence:
                signals[i] = -1

        df["signal"] = signals

        # 가중치: 신뢰도 비례 (절댓값 합 = 1로 정규화)
        df["weight"] = np.where(
            signals != 0,
            np.abs(df["confidence"].values) * signals,
            0.0
        )
        total = np.abs(df["weight"]).sum()
        if total > 1e-10:
            df["weight"] = df["weight"] / total

        return df

    def _threshold_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        """임계값 기반 신호"""
        df = df.copy()
        signals = np.zeros(len(df), dtype=int)
        weights = np.zeros(len(df), dtype=float)

        for i in range(len(df)):
            mu = df.loc[i, "mu"]
            conf = df.loc[i, "confidence"]
            if mu > self.threshold and conf > self.min_confidence:
                signals[i] = 1
                weights[i] = conf
            elif mu < -self.threshold and conf < -self.min_confidence:
                signals[i] = -1
                weights[i] = -conf

        df["signal"] = signals
        total = weights.sum()
        df["weight"] = weights / max(total, 1e-10)
        return df

    def _prob_weight_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        """확률 가중 신호 (μ/σ 비례 연속 가중치)"""
        df = df.copy()
        conf = df["confidence"].values

        # 소프트맥스 형태 가중치
        pos_conf = np.clip(conf, 0, None)
        total = pos_conf.sum()

        if total > 1e-10:
            weights = pos_conf / total
            signals = np.where(weights > 1e-4, 1, 0)
        else:
            weights = np.zeros(len(df))
            signals = np.zeros(len(df), dtype=int)

        df["signal"] = signals
        df["weight"] = weights
        return df
