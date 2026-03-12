"""
기술 지표 계산 — ta 라이브러리 기반
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from ..logger_config import get_logger

log = get_logger("features.ta")


def add_all_ta(df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV DataFrame에 기술지표 전부 추가"""
    df = df.copy()
    c = df["close"]
    h = df["high"]
    l = df["low"]
    v = df["volume"]

    # ── 이동평균 ────────────────────────────────────────────────────
    for w in [5, 10, 20, 60]:
        df[f"sma{w}"]  = c.rolling(w).mean()
        df[f"ema{w}"]  = c.ewm(span=w, adjust=False).mean()

    # ── 볼린저 밴드 ────────────────────────────────────────────────
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    df["bb_upper"] = sma20 + 2 * std20
    df["bb_lower"] = sma20 - 2 * std20
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (sma20 + 1e-9)
    df["bb_pct"]   = (c - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-9)

    # ── RSI ──────────────────────────────────────────────────────
    df["rsi14"] = _rsi(c, 14)
    df["rsi7"]  = _rsi(c, 7)

    # ── MACD ─────────────────────────────────────────────────────
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["macd"]        = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]

    # ── ATR ──────────────────────────────────────────────────────
    df["atr14"] = _atr(h, l, c, 14)

    # ── OBV ──────────────────────────────────────────────────────
    df["obv"] = _obv(c, v)

    # ── CCI ──────────────────────────────────────────────────────
    tp = (h + l + c) / 3
    df["cci20"] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std() + 1e-9)

    # ── ADX ──────────────────────────────────────────────────────
    df["adx14"] = _adx(h, l, c, 14)

    # ── Stochastic ────────────────────────────────────────────────
    lo14 = l.rolling(14).min()
    hi14 = h.rolling(14).max()
    df["stoch_k"] = 100 * (c - lo14) / (hi14 - lo14 + 1e-9)
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    # ── VWAP 근사 (누적) ─────────────────────────────────────────
    df["vwap"] = (tp * v).cumsum() / (v.cumsum() + 1e-9)

    return df


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs    = gain / (loss + 1e-9)
    return 100 - 100 / (1 + rs)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    plus_dm  = (high.diff()).clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    mask     = plus_dm < minus_dm
    plus_dm[mask]  = 0
    mask2    = minus_dm <= plus_dm
    minus_dm[mask2] = 0
    atr      = _atr(high, low, close, period)
    plus_di  = 100 * plus_dm.ewm(span=period, adjust=False).mean() / (atr + 1e-9)
    minus_di = 100 * minus_dm.ewm(span=period, adjust=False).mean() / (atr + 1e-9)
    dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    return dx.ewm(span=period, adjust=False).mean()
