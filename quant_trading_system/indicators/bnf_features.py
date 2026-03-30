# indicators/bnf_features.py
# BNF 매수 타이밍 탐지기 — 피처 계산 엔진
# ─────────────────────────────────────────────────────────────────────────────
# 입력 : OHLCV DataFrame (Open/High/Low/Close/Volume 컬럼, DatetimeIndex)
# 출력 : 원본 df + 100+ 파생 피처 (데이터 누수 없는 과거 참조만 사용)
# ─────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger("quant.bnf.features")

# ─── 필수 컬럼 ────────────────────────────────────────────────────────────────
_REQ_COLS = {"Open", "High", "Low", "Close", "Volume"}


# ─── 최신 행 스냅샷 (신호 엔진이 사용하는 구조체) ─────────────────────────────
@dataclass
class BNFFeatureRow:
    """compute_bnf_features() 결과에서 최신(마지막) 행을 추출한 구조체"""
    # --- 이동평균 ---
    close:    float = np.nan
    ma5:      float = np.nan
    ma20:     float = np.nan
    ma25:     float = np.nan
    ma60:     float = np.nan
    ma120:    float = np.nan
    ma20_slope: float = np.nan
    ma60_slope: float = np.nan

    # --- 괴리율 ---
    dev_ma20_pct:   float = np.nan
    dev_ma25_pct:   float = np.nan
    dev_vwap20_pct: float = np.nan
    zscore_ma25:    float = np.nan

    # --- 위치 ---
    pos_in_range20: float = np.nan
    pos_in_range60: float = np.nan
    gain_from_low20: float = np.nan
    near_range_bottom: float = np.nan
    near_range_top:    float = np.nan
    bb_pct:            float = np.nan

    # --- 거래량 ---
    volume:         float = np.nan
    vol_ma5:        float = np.nan
    vol_ma20:       float = np.nan
    vol_ratio5:     float = np.nan
    vol_ratio20:    float = np.nan
    vol_bias:       float = np.nan    # 상승/하락일 거래량 비율
    amount:         float = np.nan
    amount_ma20:    float = np.nan

    # --- ATR / 변동성 ---
    atr14:       float = np.nan
    atr5:        float = np.nan
    atr_pct:     float = np.nan

    # --- 모멘텀 ---
    ret1:  float = np.nan
    ret3:  float = np.nan
    ret5:  float = np.nan
    ret10: float = np.nan
    ret20: float = np.nan
    streak: int  = 0     # 양(상승) or 음(하락) 연속 봉 수

    # --- 캔들 패턴 ---
    body_ratio:          float = np.nan
    lower_shadow_ratio:  float = np.nan
    shadow_ratio:        float = np.nan
    is_bullish:          float = 0.0
    big_bear_candle:     float = 0.0
    recovery_after_bear: float = 0.0
    consec_down:         int   = 0
    higher_low:          float = 0.0
    breaks_prev_high5:   float = 0.0

    # --- 고점/저점 ---
    swing_high:  float = np.nan
    swing_low:   float = np.nan
    high20:      float = np.nan
    low20:       float = np.nan
    high60:      float = np.nan
    low60:       float = np.nan

    # --- 메타 ---
    valid:  bool = True   # 유효성 플래그 (데이터 부족 / NaN 과다 시 False)
    n_rows: int  = 0


# ─────────────────────────────────────────────────────────────────────────────
# 보조 함수
# ─────────────────────────────────────────────────────────────────────────────

def _safe_pct(a: pd.Series, b: pd.Series, scale: float = 100.0) -> pd.Series:
    """(a - b) / b × scale, b=0 방지"""
    return (a - b) / (b.replace(0, np.nan)).fillna(np.nan) * scale


def _calc_streak(ret: pd.Series) -> pd.Series:
    """연속 상승(+N) / 하락(-N) 캔들 수 계산"""
    values = []
    streak = 0
    for r in ret:
        if pd.isna(r):
            streak = 0
        elif r > 0:
            streak = max(0, streak) + 1
        elif r < 0:
            streak = min(0, streak) - 1
        else:
            streak = 0
        values.append(streak)
    return pd.Series(values, index=ret.index, dtype=float)


def _calc_consec_down(ret: pd.Series) -> pd.Series:
    """연속 하락 봉 수 (0 = 하락 없음)"""
    values = []
    count = 0
    for r in ret:
        if pd.isna(r):
            count = 0
        elif r < 0:
            count += 1
        else:
            count = 0
        values.append(count)
    return pd.Series(values, index=ret.index, dtype=int)


# ─────────────────────────────────────────────────────────────────────────────
# 메인 피처 계산 함수
# ─────────────────────────────────────────────────────────────────────────────

def compute_bnf_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    BNF 탐지에 필요한 모든 피처를 계산한다.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame. 컬럼: Open, High, Low, Close, Volume
        DatetimeIndex 권장. 최소 60개 행.

    Returns
    -------
    pd.DataFrame
        원본 컬럼 + 파생 피처 컬럼 (누수 없는 과거 참조만 사용).
    """
    # ── 검증 ─────────────────────────────────────────────────────────────────
    missing = _REQ_COLS - set(df.columns)
    if missing:
        raise ValueError(f"compute_bnf_features: 필수 컬럼 없음 → {missing}")

    f = df.copy()
    c  = f["Close"]
    h  = f["High"]
    lo = f["Low"]
    o  = f["Open"]
    v  = f["Volume"]

    # ── A. 이동평균 & 기울기 ──────────────────────────────────────────────────
    for n in [5, 20, 25, 60, 120]:
        f[f"ma{n}"] = c.rolling(n, min_periods=n).mean()

    # 기울기: 최근 N봉 평균의 변화율
    f["ma20_slope"] = f["ma20"].diff(3) / (f["ma20"].shift(3).replace(0, np.nan))
    f["ma60_slope"] = f["ma60"].diff(5) / (f["ma60"].shift(5).replace(0, np.nan))

    # MA 정렬 여부 (5>20>60 = 1, 역배열 = -1, 기타 = 0)
    f["ma_aligned_bull"] = (
        (f["ma5"] > f["ma20"]) & (f["ma20"] > f["ma60"])
    ).astype(float)
    f["ma_aligned_bear"] = (
        (f["ma5"] < f["ma20"]) & (f["ma20"] < f["ma60"])
    ).astype(float)

    # ── B. 괴리율 / 이격도 ───────────────────────────────────────────────────
    f["dev_ma20_pct"] = _safe_pct(c, f["ma20"])
    f["dev_ma25_pct"] = _safe_pct(c, f["ma25"])

    # Z-score: (close - ma25) / rolling_std
    _ma25_std = c.rolling(25, min_periods=10).std()
    f["zscore_ma25"] = (c - f["ma25"]) / (_ma25_std.replace(0, np.nan))

    # VWAP (일봉: HLC3 × Volume 기반 20일 누적)
    f["hlc3"]    = (h + lo + c) / 3
    _vw_sum      = (f["hlc3"] * v).rolling(20, min_periods=5).sum()
    _v_sum       = v.rolling(20, min_periods=5).sum()
    f["vwap20"]  = _vw_sum / (_v_sum.replace(0, np.nan))
    f["dev_vwap20_pct"] = _safe_pct(c, f["vwap20"])

    # ── C. 볼린저 밴드 / 박스권 ──────────────────────────────────────────────
    _bb_std       = c.rolling(20, min_periods=10).std()
    f["bb_upper"] = f["ma20"] + 2 * _bb_std
    f["bb_lower"] = f["ma20"] - 2 * _bb_std
    _bb_range     = (f["bb_upper"] - f["bb_lower"]).replace(0, np.nan)
    f["bb_pct"]   = (c - f["bb_lower"]) / _bb_range   # 0=하단, 1=상단

    # ── D. 스윙 고저 / 레인지 ────────────────────────────────────────────────
    for n in [10, 20, 60]:
        f[f"high{n}"] = h.rolling(n, min_periods=n).max()
        f[f"low{n}"]  = lo.rolling(n, min_periods=n).min()

    f["swing_high"] = f["high10"]
    f["swing_low"]  = f["low10"]

    _range20 = (f["high20"] - f["low20"]).replace(0, np.nan)
    _range60 = (f["high60"] - f["low60"]).replace(0, np.nan)
    f["pos_in_range20"] = (c - f["low20"]) / _range20
    f["pos_in_range60"] = (c - f["low60"]) / _range60
    f["gain_from_low20"] = _safe_pct(c, f["low20"])

    f["near_range_bottom"] = (f["pos_in_range60"] < 0.15).astype(float)
    f["near_range_top"]    = (f["pos_in_range60"] > 0.85).astype(float)

    # ── E. ATR ───────────────────────────────────────────────────────────────
    _tr = pd.concat([
        h - lo,
        (h - c.shift(1)).abs(),
        (lo - c.shift(1)).abs(),
    ], axis=1).max(axis=1)
    f["atr14"]   = _tr.rolling(14, min_periods=5).mean()
    f["atr5"]    = _tr.rolling(5,  min_periods=3).mean()
    f["atr_pct"] = f["atr14"] / c.replace(0, np.nan) * 100

    # ── F. 모멘텀 / 수익률 ───────────────────────────────────────────────────
    for n in [1, 3, 5, 10, 20]:
        f[f"ret{n}"] = c.pct_change(n) * 100

    f["streak"]    = _calc_streak(c.pct_change())
    f["consec_down"] = _calc_consec_down(c.pct_change())

    # ── G. 거래량 ────────────────────────────────────────────────────────────
    f["vol_ma5"]  = v.rolling(5,  min_periods=3).mean()
    f["vol_ma20"] = v.rolling(20, min_periods=10).mean()
    f["vol_ratio5"]  = v / f["vol_ma5"].replace(0, np.nan)
    f["vol_ratio20"] = v / f["vol_ma20"].replace(0, np.nan)

    # 상승/하락일 거래량 분리
    _ret = c.pct_change()
    _is_up = (_ret > 0).astype(float)
    _up_v  = v * _is_up
    _dn_v  = v * (1 - _is_up)
    _up_v_ma = _up_v.rolling(10, min_periods=5).mean()
    _dn_v_ma = _dn_v.rolling(10, min_periods=5).mean()
    f["vol_bias"] = _up_v_ma / (_dn_v_ma.replace(0, np.nan))

    f["amount"]      = c * v
    f["amount_ma5"]  = f["amount"].rolling(5,  min_periods=3).mean()
    f["amount_ma20"] = f["amount"].rolling(20, min_periods=10).mean()

    # ── H. 캔들 패턴 ─────────────────────────────────────────────────────────
    _body_top    = pd.concat([o, c], axis=1).max(axis=1)
    _body_bottom = pd.concat([o, c], axis=1).min(axis=1)
    _body        = (_body_top - _body_bottom).clip(lower=0)
    _hl_range    = (h - lo).clip(lower=1e-10)
    _upper_sh    = h - _body_top
    _lower_sh    = _body_bottom - lo

    f["body_ratio"]         = _body   / _hl_range
    f["lower_shadow_ratio"] = _lower_sh / _hl_range
    f["shadow_ratio"]       = (_upper_sh + _lower_sh) / _hl_range
    f["is_bullish"]         = (c > o).astype(float)

    # 장대음봉: 몸통 > ATR × 배수 + 음봉
    f["big_bear_candle"] = (
        (c < o) & (_body > f["atr14"] * 1.5)
    ).astype(float)

    # 장대음봉 다음 날 반등
    f["recovery_after_bear"] = (
        (c > o) & (f["big_bear_candle"].shift(1).fillna(0) > 0)
    ).astype(float)

    # Higher Low: 현재 저점 > 20봉 전 저점
    _prev_lo = lo.shift(1).rolling(20, min_periods=10).min()
    f["higher_low"] = (lo > _prev_lo).astype(float)

    # 전일 고가 돌파
    _prev_high5 = h.shift(1).rolling(5, min_periods=3).max()
    f["breaks_prev_high5"] = (c > _prev_high5).astype(float)

    # 하락 둔화: 5일 수익률의 절대값 감소
    _abs_ret5 = f["ret5"].abs()
    f["decline_slowing"] = (
        (_abs_ret5 < _abs_ret5.shift(3)) & (f["ret5"] < 0)
    ).astype(float)

    # 첫 양봉 전환 (연속 N봉 하락 후 첫 양봉)
    f["first_up_after_streak"] = (
        (f["streak"] > 0) & (f["streak"].shift(1).fillna(0) <= 0)
    ).astype(float)

    return f


# ─────────────────────────────────────────────────────────────────────────────
# 마지막 행 → BNFFeatureRow 변환
# ─────────────────────────────────────────────────────────────────────────────

def extract_latest_row(feat_df: pd.DataFrame) -> BNFFeatureRow:
    """
    compute_bnf_features() 결과의 마지막 유효 행을 BNFFeatureRow 로 변환.
    """
    if feat_df is None or len(feat_df) == 0:
        return BNFFeatureRow(valid=False, n_rows=0)

    row = feat_df.iloc[-1]
    n   = len(feat_df)

    def _g(col, default=np.nan):
        v = row.get(col, default)
        if isinstance(v, (pd.Series, pd.DataFrame)):
            return default
        return float(v) if pd.notna(v) else default

    def _gi(col, default=0):
        v = row.get(col, default)
        if isinstance(v, (pd.Series, pd.DataFrame)):
            return default
        try:
            return int(v) if pd.notna(v) else default
        except Exception:
            return default

    # NaN 과다 체크 (핵심 컬럼 NaN 이면 valid=False)
    core_nan = any(pd.isna(row.get(col)) for col in
                   ["Close", "ma20", "atr14", "vol_ma20"])
    valid = (not core_nan) and (n >= 20)

    return BNFFeatureRow(
        close    = _g("Close"),
        ma5      = _g("ma5"),
        ma20     = _g("ma20"),
        ma25     = _g("ma25"),
        ma60     = _g("ma60"),
        ma120    = _g("ma120"),
        ma20_slope = _g("ma20_slope"),
        ma60_slope = _g("ma60_slope"),

        dev_ma20_pct   = _g("dev_ma20_pct"),
        dev_ma25_pct   = _g("dev_ma25_pct"),
        dev_vwap20_pct = _g("dev_vwap20_pct"),
        zscore_ma25    = _g("zscore_ma25"),

        pos_in_range20    = _g("pos_in_range20"),
        pos_in_range60    = _g("pos_in_range60"),
        gain_from_low20   = _g("gain_from_low20"),
        near_range_bottom = _g("near_range_bottom"),
        near_range_top    = _g("near_range_top"),
        bb_pct            = _g("bb_pct"),

        volume       = _g("Volume"),
        vol_ma5      = _g("vol_ma5"),
        vol_ma20     = _g("vol_ma20"),
        vol_ratio5   = _g("vol_ratio5"),
        vol_ratio20  = _g("vol_ratio20"),
        vol_bias     = _g("vol_bias"),
        amount       = _g("amount"),
        amount_ma20  = _g("amount_ma20"),

        atr14    = _g("atr14"),
        atr5     = _g("atr5"),
        atr_pct  = _g("atr_pct"),

        ret1  = _g("ret1"),
        ret3  = _g("ret3"),
        ret5  = _g("ret5"),
        ret10 = _g("ret10"),
        ret20 = _g("ret20"),
        streak = _gi("streak"),

        body_ratio          = _g("body_ratio"),
        lower_shadow_ratio  = _g("lower_shadow_ratio"),
        shadow_ratio        = _g("shadow_ratio"),
        is_bullish          = _g("is_bullish"),
        big_bear_candle     = _g("big_bear_candle"),
        recovery_after_bear = _g("recovery_after_bear"),
        consec_down         = _gi("consec_down"),
        higher_low          = _g("higher_low"),
        breaks_prev_high5   = _g("breaks_prev_high5"),

        swing_high = _g("swing_high"),
        swing_low  = _g("swing_low"),
        high20 = _g("high20"),
        low20  = _g("low20"),
        high60 = _g("high60"),
        low60  = _g("low60"),

        valid  = valid,
        n_rows = n,
    )
