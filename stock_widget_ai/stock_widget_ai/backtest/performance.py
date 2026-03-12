"""
Performance Metrics — CAGR, Sharpe, MDD, Win Rate, ...
"""
from __future__ import annotations
import numpy as np
from typing import Dict, List


def compute_performance(equity: List[float], trades) -> Dict[str, float]:
    eq  = np.array(equity, dtype=float)
    if len(eq) < 2 or eq[0] <= 0:
        return _empty()

    total_ret = eq[-1] / eq[0] - 1
    n_years   = len(eq) / 252
    cagr      = (eq[-1] / eq[0]) ** (1 / max(n_years, 1e-3)) - 1

    daily_ret = np.diff(eq) / (eq[:-1] + 1e-9)
    sharpe    = 0.0
    if daily_ret.std() > 0:
        sharpe = float(daily_ret.mean() / daily_ret.std() * np.sqrt(252))

    sortino = 0.0
    neg_std = daily_ret[daily_ret < 0].std()
    if neg_std > 0:
        sortino = float(daily_ret.mean() / neg_std * np.sqrt(252))

    mdd = _max_drawdown(eq)

    n_trades = len(trades)
    win_rate = 0.0
    if n_trades > 0:
        win_rate = float(sum(1 for t in trades if t.pnl > 0) / n_trades)

    avg_hold = 0.0

    return {
        "total_return": float(total_ret),
        "cagr":          float(cagr),
        "sharpe":        float(sharpe),
        "sortino":       float(sortino),
        "max_drawdown":  float(mdd),
        "win_rate":      float(win_rate),
        "n_trades":      int(n_trades),
    }


def _max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd   = (equity - peak) / (peak + 1e-9)
    return float(dd.min())


def _empty() -> Dict[str, float]:
    return {k: 0.0 for k in
            ["total_return","cagr","sharpe","sortino","max_drawdown","win_rate","n_trades"]}
