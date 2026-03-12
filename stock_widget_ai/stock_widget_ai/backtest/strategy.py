"""
Trading Strategies — 예측 기반 매수/매도 시그널 생성
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict


class PredictionStrategy:
    """
    예측 수익률(return) 또는 방향(direction) 기반 시그널 생성
    """
    def __init__(
        self,
        buy_threshold: float = 0.01,
        sell_threshold: float = -0.01,
        target_type: str = "return",
        confidence_min: float = 0.0,
        use_confidence: bool = False,
    ) -> None:
        self.buy_th  = buy_threshold
        self.sell_th = sell_threshold
        self.target_type = target_type
        self.conf_min    = confidence_min
        self.use_conf    = use_confidence

    def generate_signals(
        self,
        pred: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> pd.Series:
        """
        Returns
        -------
        signal: pd.Series — 1=buy, -1=sell, 0=hold
        """
        conf = np.zeros_like(pred)
        if self.use_conf:
            width = np.abs(upper - lower) + 1e-9
            conf  = 1.0 / width   # 좁을수록 신뢰도 높음 (정규화 없이 상대적 사용)

        sig = np.zeros(len(pred), dtype=int)
        for i, p in enumerate(pred):
            if self.use_conf and conf[i] < self.conf_min:
                sig[i] = 0
                continue
            if self.target_type == "direction":
                sig[i] = 1 if p > 0.5 else (-1 if p < 0.5 else 0)
            else:
                sig[i] = 1 if p >= self.buy_th else (-1 if p <= self.sell_th else 0)
        return pd.Series(sig, name="signal")
