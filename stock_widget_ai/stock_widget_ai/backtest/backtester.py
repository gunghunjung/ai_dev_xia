"""
Backtester — 시그널 기반 포지션 시뮬레이션
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, Optional
from .strategy import PredictionStrategy
from .portfolio import PortfolioState
from .performance import compute_performance
from ..logger_config import get_logger

log = get_logger("backtest.backtester")


class Backtester:
    def __init__(
        self,
        initial_cash: float = 10_000_000,
        fee: float = 0.00015,
        slippage: float = 0.0005,
        position_size: float = 1.0,
        long_only: bool = True,
    ) -> None:
        self.initial_cash = initial_cash
        self.fee          = fee
        self.slippage     = slippage
        self.position_size = position_size
        self.long_only    = long_only

    def run(
        self,
        price_series: pd.Series,
        pred:  np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        strategy: PredictionStrategy,
    ) -> Dict:
        """
        price_series: 실제 종가 (DatetimeIndex)
        pred/lower/upper: 예측값 array (같은 길이)
        Returns backtest result dict
        """
        n = min(len(price_series), len(pred))
        prices = price_series.values[-n:]
        dates  = [str(d.date()) for d in price_series.index[-n:]]

        signals = strategy.generate_signals(pred[:n], lower[:n], upper[:n])

        port = PortfolioState(cash=self.initial_cash)
        port.record_equity(dates[0], prices[0])

        for i in range(n):
            sig   = int(signals.iloc[i])
            price = float(prices[i]) * (1 + self.slippage)

            if sig == 1 and port.shares == 0:
                port.open_long(dates[i], price, self.position_size, self.fee)
            elif sig == -1 and port.shares > 0:
                port.close_long(dates[i], price, self.fee)

            port.record_equity(dates[i], float(prices[i]))

        # 마지막에 열린 포지션 청산
        if port.shares > 0:
            port.close_long(dates[-1], float(prices[-1]), self.fee)

        perf = compute_performance(port.equity_curve, port.trades)
        log.info(f"backtest 완료: trades={perf['n_trades']}, sharpe={perf['sharpe']:.2f}")

        return {
            "performance":  perf,
            "equity_curve": port.equity_curve,
            "dates":        dates,
            "trades":       [
                {"entry": t.entry_date, "exit": t.exit_date,
                 "entry_price": t.entry_price, "exit_price": t.exit_price,
                 "shares": t.shares, "pnl": t.pnl}
                for t in port.trades
            ],
        }
