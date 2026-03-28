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

    def load_cached_only(self, symbol: str,
                         period: str = "5y",
                         interval: str = "1d") -> Optional[pd.DataFrame]:
        """
        다운로드 없이 캐시에서만 로드.
        캐시 없으면 None 반환. 신선도 검사용.
        """
        return self.cache.load(symbol, period, interval)

    def is_stale(self, symbol: str, period: str = "5y",
                 interval: str = "1d", max_age_days: int = 2) -> bool:
        """
        종목 데이터가 max_age_days일 이상 오래됐는지 확인.
        True = 갱신 필요
        """
        import datetime
        df = self.load_cached_only(symbol, period, interval)
        if df is None or df.empty:
            return True
        last_date = df.index[-1]
        if hasattr(last_date, "date"):
            last_date = last_date.date()
        delta = (datetime.date.today() - last_date).days
        return delta >= max_age_days

    def refresh_if_stale(self, symbol: str, period: str = "5y",
                          interval: str = "1d",
                          max_age_days: int = 2) -> Optional[pd.DataFrame]:
        """
        오래된 데이터면 재다운로드, 최신이면 캐시 반환.
        일일 자동갱신 루틴에서 사용.
        """
        if self.is_stale(symbol, period, interval, max_age_days):
            logger.info(f"[{symbol}] 데이터 stale → 재다운로드")
            df = self._download(symbol, period, interval)
            if df is not None and not df.empty:
                self.cache.save(df, symbol, period, interval)
                return df
            # 다운로드 실패 시 이전 캐시라도 반환
            return self.load_cached_only(symbol, period, interval)
        return self.load_cached_only(symbol, period, interval)

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

    # ──────────────────────────────────────────────────
    # 실시간 시세 (yfinance fast_info, ~15분 지연)
    # ──────────────────────────────────────────────────

    def get_realtime_price(self, symbol: str) -> Optional[dict]:
        """
        yfinance fast_info 로 현재가 조회 (약 15분 지연).
        반환 dict: symbol, price, prev_close, change, change_pct,
                   high, low, volume, currency
        """
        try:
            import yfinance as yf
            fi = yf.Ticker(symbol).fast_info

            def _safe(key, default=None):
                try:
                    v = fi[key]
                    return float(v) if v is not None else default
                except Exception:
                    return default

            price      = _safe("last_price")
            prev_close = _safe("previous_close")

            if price is None:
                return None

            change     = (price - prev_close) if prev_close else 0.0
            change_pct = (change / prev_close * 100) if prev_close else 0.0

            volume = _safe("regular_market_volume")
            if volume is None:
                volume = _safe("three_month_average_volume")

            currency = ""
            try:
                currency = fi["currency"] or ""
            except Exception:
                pass

            return {
                "symbol":     symbol,
                "price":      price,
                "prev_close": prev_close,
                "change":     change,
                "change_pct": change_pct,
                "high":       _safe("day_high"),
                "low":        _safe("day_low"),
                "volume":     volume,
                "currency":   currency,
            }
        except Exception as e:
            logger.debug(f"{symbol} 실시간 시세 조회 실패: {e}")
            return None

    def get_realtime_prices(
        self,
        symbols: List[str],
        max_workers: int = 8,
    ) -> Dict[str, dict]:
        """
        복수 심볼 실시간 현재가 일괄 조회 (병렬 ThreadPoolExecutor).

        순차 조회 대비 ~N배 빠름 (네트워크 I/O 병렬화).
        max_workers 기본값 8 — yfinance 서버 부하 방지를 위해 16 이하 권장.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        if not symbols:
            return {}

        result: Dict[str, dict] = {}
        with ThreadPoolExecutor(max_workers=min(max_workers, len(symbols))) as ex:
            future_to_sym = {ex.submit(self.get_realtime_price, sym): sym
                             for sym in symbols}
            for fut in as_completed(future_to_sym):
                sym  = future_to_sym[fut]
                try:
                    data = fut.result()
                    if data:
                        result[sym] = data
                except Exception as e:
                    logger.debug(f"{sym} 병렬 실시간 조회 실패: {e}")
        return result
