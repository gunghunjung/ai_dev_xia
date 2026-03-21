# data/loader.py — 시장 데이터 수집 레이어 (yfinance / pykrx)
from __future__ import annotations
import logging
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from .cache import CacheManager

logger = logging.getLogger("quant.data")


class DataLoader:
    """
    멀티-소스 OHLCV 데이터 로더
    - yfinance: 국내/해외 모든 심볼 지원 (.KS, .KQ 포함)
    - 캐시 우선 사용 → 없으면 다운로드 → 캐시 저장
    - 미래 데이터 누수 방지: 미래 날짜 행 자동 제거
    """

    def __init__(self, cache_dir: str = "cache", ttl_hours: int = 23):
        self.cache = CacheManager(cache_dir, ttl_hours)

    # ──────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────

    def load(self, symbol: str, period: str = "5y",
             interval: str = "1d") -> Optional[pd.DataFrame]:
        """단일 심볼 OHLCV 로드 (캐시 우선)"""
        cached = self.cache.load(symbol, period, interval)
        if cached is not None:
            return cached

        df = self._download(symbol, period, interval)
        if df is not None and not df.empty:
            self.cache.save(df, symbol, period, interval)
        return df

    def load_multi(self, symbols: List[str], period: str = "5y",
                   interval: str = "1d") -> Dict[str, pd.DataFrame]:
        """복수 심볼 로드. 실패한 심볼은 건너뜀"""
        result = {}
        for sym in symbols:
            try:
                df = self.load(sym, period, interval)
                if df is not None and len(df) >= 50:
                    result[sym] = df
                    logger.info(f"[데이터] {sym}: {len(df)}행 로드 완료")
                else:
                    logger.warning(f"[데이터] {sym}: 데이터 부족 (건너뜀)")
            except Exception as e:
                logger.error(f"[데이터] {sym} 로드 실패: {e}")
        return result

    def load_aligned(self, symbols: List[str], period: str = "5y",
                     interval: str = "1d") -> Dict[str, pd.DataFrame]:
        """
        복수 심볼을 공통 날짜 인덱스로 정렬
        - 모든 심볼의 공통 거래일만 포함
        - Forward-fill (최대 5일) 후 NaN 행 제거
        """
        raw = self.load_multi(symbols, period, interval)
        if not raw:
            return {}

        # 공통 인덱스 구축
        closes = {s: df["Close"] for s, df in raw.items()}
        price_df = pd.DataFrame(closes).sort_index()
        price_df = price_df.ffill(limit=5).dropna(how="all")

        # 각 심볼을 공통 인덱스로 reindex
        aligned = {}
        for sym, df in raw.items():
            df_aligned = df.reindex(price_df.index).ffill(limit=5)
            aligned[sym] = df_aligned.dropna(subset=["Close"])
        return aligned

    # ──────────────────────────────────────────────────
    # Download implementations
    # ──────────────────────────────────────────────────

    def _download(self, symbol: str, period: str,
                  interval: str) -> Optional[pd.DataFrame]:
        """yfinance로 다운로드"""
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval, auto_adjust=True)
            if df.empty:
                logger.warning(f"{symbol}: yfinance 데이터 없음")
                return None
            df = self._clean(df, symbol)
            return df
        except Exception as e:
            logger.error(f"{symbol} yfinance 다운로드 실패: {e}")
            return None

    def _clean(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """데이터 정제"""
        # 컬럼명 표준화
        col_map = {}
        for c in df.columns:
            cl = c.lower().strip()
            if "open" in cl:
                col_map[c] = "Open"
            elif "high" in cl:
                col_map[c] = "High"
            elif "low" in cl:
                col_map[c] = "Low"
            elif "close" in cl and "adj" not in cl:
                col_map[c] = "Close"
            elif "volume" in cl:
                col_map[c] = "Volume"
        df = df.rename(columns=col_map)

        # 필수 컬럼만 유지
        needed = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        df = df[needed].copy()

        # 인덱스 정규화
        if hasattr(df.index, "tz") and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df.index = pd.to_datetime(df.index).normalize()
        df.index.name = "Date"

        # 미래 데이터 누수 방지
        today = pd.Timestamp.now().normalize()
        df = df[df.index <= today]

        # 이상치 제거
        df = df[df["Close"] > 0]
        df = df.dropna(subset=["Close"])

        # 거래량 0인 날 (비거래일) 제거
        if "Volume" in df.columns:
            df = df[df["Volume"] > 0]

        df = df.sort_index()
        return df

    def get_benchmark(self, symbol: str = "^KS11",
                      period: str = "5y") -> Optional[pd.Series]:
        """벤치마크 수익률 시리즈 반환"""
        df = self.load(symbol, period)
        if df is None:
            return None
        return df["Close"].pct_change().dropna()
