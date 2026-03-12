"""
데이터 수집 — yfinance 기반 OHLCV + 벤치마크
"""
from __future__ import annotations
import os
import pandas as pd
import numpy as np
from typing import Optional, Tuple
from ..logger_config import get_logger

log = get_logger("data.loader")


class DataLoader:
    """yfinance 기반 주식 데이터 수집기"""

    PERIODS = ["1mo", "3mo", "6mo", "1y", "2y", "3y", "5y", "10y", "max"]
    INTERVALS = ["1d", "1wk", "1mo"]

    def __init__(self, cache_dir: str = "data") -> None:
        self._cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def load(
        self,
        symbol: str,
        period: str = "5y",
        interval: str = "1d",
        use_cache: bool = True,
        force_reload: bool = False,
    ) -> pd.DataFrame:
        """OHLCV DataFrame 반환 (인덱스: DatetimeIndex)"""
        parquet_path = self._cache_path(symbol, period, interval, ext="parquet")
        csv_path     = self._cache_path(symbol, period, interval, ext="csv")

        if use_cache and not force_reload:
            # 1순위: parquet (빠름), 2순위: CSV (pyarrow/fastparquet 없을 때 fallback)
            for path, reader in [
                (parquet_path, lambda p: pd.read_parquet(p)),
                (csv_path,     lambda p: pd.read_csv(p, index_col=0, parse_dates=True)),
            ]:
                if os.path.exists(path):
                    try:
                        df = reader(path)
                        log.info(f"캐시 로드: {symbol} ({len(df)}행)  [{os.path.basename(path)}]")
                        return df
                    except Exception as e:
                        log.warning(f"캐시 읽기 실패 ({os.path.basename(path)}): {e}")

        df = self._download(symbol, period, interval)
        if df is not None and not df.empty and use_cache:
            # parquet 저장 시도 → 실패 시 CSV로 저장
            saved = False
            try:
                df.to_parquet(parquet_path)
                saved = True
            except Exception:
                pass
            if not saved:
                try:
                    df.to_csv(csv_path)
                    log.info(f"캐시 저장 (CSV fallback): {os.path.basename(csv_path)}")
                except Exception as e:
                    log.warning(f"캐시 저장 실패: {e}")
        return df if df is not None else pd.DataFrame()

    def load_with_benchmark(
        self,
        symbol: str,
        benchmark: str = "^GSPC",
        period: str = "5y",
        interval: str = "1d",
        use_cache: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df = self.load(symbol, period, interval, use_cache)
        bm = self.load(benchmark, period, interval, use_cache)
        return df, bm

    def _download(
        self, symbol: str, period: str, interval: str
    ) -> Optional[pd.DataFrame]:
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            raw = ticker.history(period=period, interval=interval, auto_adjust=True)
            if raw is None or raw.empty:
                log.warning(f"yfinance: {symbol} 데이터 없음")
                return None
            df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
            df.columns = ["open", "high", "low", "close", "volume"]
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df = df.dropna(subset=["close"]).sort_index()
            log.info(f"다운로드: {symbol} {df.index[0].date()}~{df.index[-1].date()} ({len(df)}행)")
            return df
        except Exception as e:
            log.error(f"다운로드 실패 [{symbol}]: {e}")
            return None

    def _cache_path(self, symbol: str, period: str, interval: str, ext: str = "parquet") -> str:
        safe = symbol.replace("^", "IDX_").replace("/", "_")
        return os.path.join(self._cache_dir, f"{safe}_{period}_{interval}.{ext}")

    @staticmethod
    def generate_synthetic(n: int = 500, seed: int = 42) -> pd.DataFrame:
        """테스트용 합성 데이터"""
        rng = np.random.default_rng(seed)
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        close = 100 * np.cumprod(1 + rng.normal(0, 0.01, n))
        noise = rng.uniform(0.99, 1.01, n)
        return pd.DataFrame({
            "open":   close * rng.uniform(0.99, 1.01, n),
            "high":   close * rng.uniform(1.00, 1.02, n),
            "low":    close * rng.uniform(0.98, 1.00, n),
            "close":  close,
            "volume": rng.integers(1_000_000, 10_000_000, n).astype(float),
        }, index=dates)
