"""
DataProvider — OHLCV 데이터 어댑터
────────────────────────────────────────────────────────────────
• 기존 AppController.df (pandas DataFrame) 우선 사용
• 열 이름 자동 정규화 (대소문자, 한글 alias 처리)
• CSV 로더 어댑터 (독립 실행용)
• 다운샘플링: OHLCV 집계 (O=first, H=max, L=min, C=last, V=sum)
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# 열 이름 alias 매핑
_COL_ALIASES: dict[str, list[str]] = {
    "open":   ["open", "Open", "시가", "o"],
    "high":   ["high", "High", "고가", "h"],
    "low":    ["low",  "Low",  "저가", "l"],
    "close":  ["close", "Close", "종가", "c", "adj close", "adj_close"],
    "volume": ["volume", "Volume", "거래량", "vol", "v"],
}
_REQUIRED = list(_COL_ALIASES.keys())


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """열 이름을 소문자 표준 이름으로 정규화."""
    col_map: dict[str, str] = {}
    lower_map = {c.lower(): c for c in df.columns}
    for std, aliases in _COL_ALIASES.items():
        for alias in aliases:
            if alias in df.columns:
                col_map[alias] = std
                break
            if alias.lower() in lower_map:
                col_map[lower_map[alias.lower()]] = std
                break
    return df.rename(columns=col_map)


class DataProvider:
    """
    OHLCV 데이터 소스 추상화.
    load_from_df() 또는 load_from_csv() 로 데이터 주입 후
    get_range() / get_downsampled() 로 슬라이싱.
    """

    MAX_RENDER: int = 2_000   # 화면에 직접 렌더링할 최대 캔들 수

    def __init__(self) -> None:
        self._raw: Optional[pd.DataFrame] = None
        self._symbol: str = ""

    # ── 데이터 주입 ────────────────────────────────────────────
    def load_from_df(self, df: pd.DataFrame, symbol: str = "") -> None:
        """기존 AppController.df에서 로드."""
        df = _normalize_columns(df.copy())
        missing = [c for c in _REQUIRED if c not in df.columns]
        if missing:
            raise ValueError(f"DataProvider: 필수 열 없음 — {missing}")
        self._raw = df[_REQUIRED].copy()
        self._raw.index = pd.to_datetime(self._raw.index)
        self._symbol = symbol

    def load_from_csv(self, path: str, symbol: str = "") -> None:
        """CSV 파일에서 로드 (어댑터)."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"CSV not found: {path}")
        df = pd.read_csv(p, index_col=0, parse_dates=True)
        self.load_from_df(df, symbol or p.stem)

    # ── 데이터 접근 ────────────────────────────────────────────
    @property
    def symbol(self) -> str:
        return self._symbol

    @property
    def raw(self) -> Optional[pd.DataFrame]:
        return self._raw

    @property
    def n(self) -> int:
        return len(self._raw) if self._raw is not None else 0

    def is_empty(self) -> bool:
        return self._raw is None or len(self._raw) == 0

    def get_range(self, start: int, end: int) -> pd.DataFrame:
        """원본 슬라이스 [start, end)."""
        if self._raw is None:
            return pd.DataFrame(columns=_REQUIRED)
        s = max(0, int(start))
        e = min(self.n, int(end))
        return self._raw.iloc[s:e]

    def get_downsampled(self, start: int, end: int, max_pts: int) -> pd.DataFrame:
        """
        [start, end) 구간을 최대 max_pts 개 캔들로 집계 반환.
        이미 max_pts 이하면 원본 반환.
        """
        data = self.get_range(start, end)
        n = len(data)
        if n == 0 or n <= max_pts:
            return data

        k = math.ceil(n / max_pts)
        opens  = data["open"].to_numpy()
        highs  = data["high"].to_numpy()
        lows   = data["low"].to_numpy()
        closes = data["close"].to_numpy()
        vols   = data["volume"].to_numpy()
        dates  = data.index

        rows: list[dict] = []
        idxs: list = []
        for i in range(0, n, k):
            j = min(i + k, n)
            rows.append({
                "open":   opens[i],
                "high":   highs[i:j].max(),
                "low":    lows[i:j].min(),
                "close":  closes[j - 1],
                "volume": vols[i:j].sum(),
            })
            idxs.append(dates[i])

        return pd.DataFrame(rows, index=pd.DatetimeIndex(idxs))
