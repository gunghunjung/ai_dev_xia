"""
MarketDataService — 종목 + 벤치마크를 합쳐 학습용 DataFrame 제공
"""
from __future__ import annotations
import pandas as pd
from typing import Optional, Tuple
from .data_loader import DataLoader
from .preprocessing import Preprocessor
from ..logger_config import get_logger

log = get_logger("data.service")


class MarketDataService:
    def __init__(self, cache_dir: str = "data") -> None:
        self._loader = DataLoader(cache_dir)
        self._prep   = Preprocessor()
        self._raw_df: Optional[pd.DataFrame] = None
        self._raw_bm: Optional[pd.DataFrame] = None

    def fetch(
        self,
        symbol: str,
        benchmark: str = "^GSPC",
        period: str = "5y",
        interval: str = "1d",
        force_reload: bool = False,
    ) -> pd.DataFrame:
        """종목 + 벤치마크 OHLCV를 결합한 DataFrame 반환"""
        self._raw_df, self._raw_bm = self._loader.load_with_benchmark(
            symbol, benchmark, period, interval, use_cache=not force_reload
        )

        if self._raw_df is None or self._raw_df.empty:
            log.warning(f"{symbol}: 데이터 없음 → 합성 데이터 사용")
            self._raw_df = DataLoader.generate_synthetic()

        df = self._raw_df.copy()

        # 벤치마크 수익률 병합
        if self._raw_bm is not None and not self._raw_bm.empty:
            bm_ret = self._raw_bm["close"].pct_change().rename("bm_return")
            df = df.join(bm_ret, how="left")

        df = self._prep.clean(df)
        log.info(f"fetch 완료: {symbol} {len(df)}행")
        return df

    @property
    def raw(self) -> Optional[pd.DataFrame]:
        return self._raw_df

    @property
    def preprocessor(self) -> Preprocessor:
        return self._prep
