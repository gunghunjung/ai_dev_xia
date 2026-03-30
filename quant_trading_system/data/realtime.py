# data/realtime.py — 실시간 시세 조회
"""
RealtimeFetcher: yfinance fast_info를 이용해 등록 종목의 현재가를 빠르게 조회합니다.

- fast_info 사용 → 1회 HTTP 요청으로 현재가/전일종가/거래량 취득
- 네트워크 오류 시 마지막 성공값 캐싱 반환 (stale 표시)
- 한국 시장 시간 외에도 마지막 체결가 반환
"""
from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Dict, Optional

logger = logging.getLogger("quant.data.realtime")


class PriceSnapshot:
    """단일 종목 시세 스냅샷."""
    __slots__ = ("symbol", "price", "prev_close", "change", "change_pct",
                 "volume", "fetched_at", "stale")

    def __init__(
        self,
        symbol: str,
        price: float,
        prev_close: float,
        volume: int = 0,
        stale: bool = False,
    ):
        self.symbol     = symbol
        self.price      = price
        self.prev_close = prev_close if prev_close else price
        self.change     = price - self.prev_close
        self.change_pct = self.change / self.prev_close if self.prev_close else 0.0
        self.volume     = volume
        self.fetched_at = datetime.now()
        self.stale      = stale  # 네트워크 오류 시 True

    @property
    def fetched_str(self) -> str:
        suffix = " ⚠" if self.stale else ""
        return self.fetched_at.strftime("%H:%M:%S") + suffix


class RealtimeFetcher:
    """
    종목별 실시간 시세 조회기.

    Parameters
    ----------
    cache_sec : 같은 종목을 이 초(sec) 이내에 재요청하면 캐시 반환 (기본 10초)
    """

    def __init__(self, cache_sec: int = 10):
        self._cache: Dict[str, PriceSnapshot] = {}
        self._cache_sec = cache_sec

    # ─────────────────────────────────────────────
    # 퍼블릭 API
    # ─────────────────────────────────────────────

    def fetch(self, symbol: str) -> Optional[PriceSnapshot]:
        """
        단일 종목 현재가 조회.
        캐시 유효 시 캐시 반환, 아니면 yfinance fast_info 사용.
        """
        cached = self._cache.get(symbol)
        if cached and not cached.stale:
            age = (datetime.now() - cached.fetched_at).total_seconds()
            if age < self._cache_sec:
                return cached

        snap = self._fetch_one(symbol)
        if snap is not None:
            self._cache[symbol] = snap
        elif cached is not None:
            # 이전 값을 stale로 표시해서 반환
            stale = PriceSnapshot(
                symbol, cached.price, cached.prev_close,
                cached.volume, stale=True,
            )
            stale.fetched_at = cached.fetched_at
            return stale
        return snap

    def fetch_all(self, symbols: list) -> Dict[str, Optional[PriceSnapshot]]:
        """복수 종목 현재가 일괄 조회."""
        result = {}
        for sym in symbols:
            try:
                result[sym] = self.fetch(sym)
            except Exception as exc:
                logger.debug(f"[{sym}] fetch 실패: {exc}")
                result[sym] = None
        return result

    def clear_cache(self):
        self._cache.clear()

    # ─────────────────────────────────────────────
    # 내부 구현
    # ─────────────────────────────────────────────

    def _fetch_one(self, symbol: str) -> Optional[PriceSnapshot]:
        """yfinance fast_info로 현재가 조회."""
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            fi = ticker.fast_info

            price      = float(fi.get("last_price") or fi.get("lastPrice") or 0)
            prev_close = float(fi.get("previous_close") or fi.get("previousClose") or 0)
            volume     = int(fi.get("three_month_average_volume") or 0)

            # fast_info 값이 없으면 최근 2일 히스토리로 fallback
            if price == 0:
                price, prev_close, volume = self._fetch_fallback(ticker)

            if price == 0:
                return None

            return PriceSnapshot(symbol, price, prev_close, volume)

        except Exception as exc:
            logger.debug(f"[{symbol}] _fetch_one 오류: {exc}")
            return None

    @staticmethod
    def _fetch_fallback(ticker) -> tuple:
        """fast_info 실패 시 최근 히스토리에서 현재가 추출."""
        try:
            hist = ticker.history(period="5d", interval="1d", auto_adjust=True)
            if hist.empty:
                return 0, 0, 0
            price      = float(hist["Close"].iloc[-1])
            prev_close = float(hist["Close"].iloc[-2]) if len(hist) > 1 else price
            volume     = int(hist["Volume"].iloc[-1]) if "Volume" in hist.columns else 0
            return price, prev_close, volume
        except Exception:
            return 0, 0, 0
