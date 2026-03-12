"""
CalendarFeatureEngineer — 캘린더 기반 특성 생성
────────────────────────────────────────────────
- 요일/월/분기 효과
- 월말/월초 효과
- 옵션 만기일 (매월 세 번째 금요일)
- 결산/실적발표 시즌 (1Q: 3~4월, 2Q: 7~8월, 3Q: 10~11월, 4Q: 1~2월)
- 공휴일 근접도 (yfinance 캘린더 없으면 단순 스킵)
- 배당락일 (데이터 없으면 스킵)

⚠ 모든 특성은 미래정보를 사용하지 않는 캘린더 속성만 포함.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List
from ..logger_config import get_logger

log = get_logger("features.calendar")


class CalendarFeatureEngineer:
    """
    Parameters
    ----------
    include_day_of_week  : 요일 효과 포함 여부
    include_month_effect : 월별 효과 포함 여부
    include_option_expiry: 옵션 만기일 효과 포함
    include_earnings_season: 실적 시즌 효과
    """

    def __init__(
        self,
        include_day_of_week:    bool = True,
        include_month_effect:   bool = True,
        include_option_expiry:  bool = True,
        include_earnings_season: bool = True,
    ) -> None:
        self._dow = include_day_of_week
        self._mon = include_month_effect
        self._opt = include_option_expiry
        self._ear = include_earnings_season

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        df 의 DatetimeIndex 를 이용해 캘린더 특성 추가.
        원본 수정 없이 새 df 반환.
        """
        df = df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            log.warning("DatetimeIndex가 아님 — 캘린더 특성 스킵")
            return df

        idx = df.index

        if self._dow:
            df = self._add_dow(df, idx)
        if self._mon:
            df = self._add_month(df, idx)
        if self._opt:
            df = self._add_option_expiry(df, idx)
        if self._ear:
            df = self._add_earnings_season(df, idx)

        df = self._add_misc(df, idx)
        log.debug(f"캘린더 특성 {len(self.feature_names())}개 추가")
        return df

    def feature_names(self) -> List[str]:
        names = []
        if self._dow: names += ["dow_sin", "dow_cos", "is_monday", "is_friday"]
        if self._mon: names += ["month_sin", "month_cos", "is_month_end", "is_month_start",
                                 "is_quarter_end", "is_quarter_start"]
        if self._opt: names += ["option_expiry", "days_to_option_expiry"]
        if self._ear: names += ["earnings_season", "earnings_season_strength"]
        names += ["week_of_year_sin", "week_of_year_cos", "trading_days_in_month"]
        return names

    # ── 요일 ──────────────────────────────────────────────────────────────
    @staticmethod
    def _add_dow(df: pd.DataFrame, idx: pd.DatetimeIndex) -> pd.DataFrame:
        dow = idx.dayofweek  # 0=Monday, 4=Friday
        df["dow_sin"]   = np.sin(2 * np.pi * dow / 5)
        df["dow_cos"]   = np.cos(2 * np.pi * dow / 5)
        df["is_monday"] = (dow == 0).astype(float)
        df["is_friday"] = (dow == 4).astype(float)
        return df

    # ── 월/분기 ───────────────────────────────────────────────────────────
    @staticmethod
    def _add_month(df: pd.DataFrame, idx: pd.DatetimeIndex) -> pd.DataFrame:
        month = idx.month
        df["month_sin"]        = np.sin(2 * np.pi * month / 12)
        df["month_cos"]        = np.cos(2 * np.pi * month / 12)
        df["is_month_end"]     = idx.is_month_end.astype(float)
        df["is_month_start"]   = idx.is_month_start.astype(float)
        df["is_quarter_end"]   = idx.is_quarter_end.astype(float)
        df["is_quarter_start"] = idx.is_quarter_start.astype(float)
        return df

    # ── 옵션 만기 ─────────────────────────────────────────────────────────
    @staticmethod
    def _add_option_expiry(df: pd.DataFrame, idx: pd.DatetimeIndex) -> pd.DataFrame:
        """미국 기준: 매월 세 번째 금요일이 옵션 만기일"""
        expiries = set()
        for year in range(idx.year.min(), idx.year.max() + 1):
            for month in range(1, 13):
                fridays = [
                    d for d in pd.date_range(f"{year}-{month:02d}-01", periods=31, freq="D")
                    if d.month == month and d.dayofweek == 4
                ]
                if len(fridays) >= 3:
                    expiries.add(fridays[2].date())

        is_expiry = pd.Series(
            [d.date() in expiries for d in idx], index=idx, dtype=float
        )
        df["option_expiry"] = is_expiry

        # 다음 만기까지 남은 거래일 (최대 30)
        days_to = []
        sorted_exp = sorted(expiries)
        for d in idx:
            future = [e for e in sorted_exp if e > d.date()]
            if future:
                delta = (pd.Timestamp(future[0]) - d).days
                days_to.append(min(delta, 30) / 30.0)
            else:
                days_to.append(0.0)
        df["days_to_option_expiry"] = days_to

        return df

    # ── 실적 시즌 ─────────────────────────────────────────────────────────
    @staticmethod
    def _add_earnings_season(df: pd.DataFrame, idx: pd.DatetimeIndex) -> pd.DataFrame:
        """
        미국 실적 시즌 (분기별 1~2개월 후 발표):
          1Q: 4~5월  2Q: 7~8월  3Q: 10~11월  4Q: 1~2월
        """
        month = idx.month
        in_season = np.isin(month, [1, 2, 4, 5, 7, 8, 10, 11])
        # 핵심 월 (발표 집중): 1,4,7,10
        core_month = np.isin(month, [1, 4, 7, 10])

        df["earnings_season"]          = in_season.astype(float)
        df["earnings_season_strength"] = core_month.astype(float)
        return df

    # ── 기타 ──────────────────────────────────────────────────────────────
    @staticmethod
    def _add_misc(df: pd.DataFrame, idx: pd.DatetimeIndex) -> pd.DataFrame:
        # 연중 주차 (순환 인코딩)
        woy = idx.isocalendar().week.astype(float).values
        df["week_of_year_sin"] = np.sin(2 * np.pi * woy / 52)
        df["week_of_year_cos"] = np.cos(2 * np.pi * woy / 52)

        # 월별 거래일 수 (정규화)
        month_trading = pd.Series(idx).groupby(pd.Series(idx).dt.to_period("M")).transform("count")
        df["trading_days_in_month"] = (month_trading.values / 23.0).clip(0, 1)

        return df
