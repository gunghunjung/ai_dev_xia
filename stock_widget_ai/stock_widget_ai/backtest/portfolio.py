"""
Portfolio — 포지션 관리 및 PnL 계산
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List


@dataclass
class Trade:
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    shares: float
    pnl: float


@dataclass
class PortfolioState:
    cash: float
    shares: float = 0.0
    entry_price: float = 0.0
    entry_date: str = ""
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    dates: List[str] = field(default_factory=list)

    def record_equity(self, date: str, price: float) -> None:
        equity = self.cash + self.shares * price
        self.equity_curve.append(equity)
        self.dates.append(date)

    def open_long(self, date: str, price: float, position_size: float, fee: float) -> None:
        invest = self.cash * position_size
        shares = invest / (price * (1 + fee))
        cost   = shares * price * (1 + fee)
        if cost > self.cash:
            cost   = self.cash
            shares = cost / (price * (1 + fee))
        self.cash        -= cost
        self.shares      += shares
        self.entry_price  = price
        self.entry_date   = date

    def close_long(self, date: str, price: float, fee: float) -> None:
        if self.shares <= 0:
            return
        proceeds = self.shares * price * (1 - fee)
        pnl = proceeds - (self.shares * self.entry_price)
        self.trades.append(Trade(
            self.entry_date, date,
            self.entry_price, price,
            self.shares, pnl,
        ))
        self.cash   += proceeds
        self.shares  = 0.0
