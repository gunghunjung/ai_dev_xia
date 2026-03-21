# data/cache.py — Parquet 기반 데이터 캐시 관리
from __future__ import annotations
import os
import hashlib
import time
import logging
from typing import Optional
import pandas as pd

logger = logging.getLogger("quant.cache")


class CacheManager:
    """
    시장 데이터 Parquet 캐시
    - TTL(Time-To-Live) 기반 캐시 유효성 검사
    - 심볼/기간/주기 조합으로 캐시 키 생성
    """

    def __init__(self, cache_dir: str = "cache", ttl_hours: int = 23):
        self.cache_dir = cache_dir
        self.ttl_seconds = ttl_hours * 3600
        os.makedirs(cache_dir, exist_ok=True)

    def _key(self, symbol: str, period: str, interval: str) -> str:
        raw = f"{symbol}_{period}_{interval}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _path(self, key: str) -> str:
        return os.path.join(self.cache_dir, f"{key}.parquet")

    def load(self, symbol: str, period: str, interval: str) -> Optional[pd.DataFrame]:
        """캐시에서 데이터 로드. 없거나 만료됐으면 None 반환"""
        path = self._path(self._key(symbol, period, interval))
        if not os.path.exists(path):
            return None
        age = time.time() - os.path.getmtime(path)
        if age > self.ttl_seconds:
            logger.debug(f"캐시 만료: {symbol} ({age/3600:.1f}시간 경과)")
            return None
        try:
            df = pd.read_parquet(path)
            logger.debug(f"캐시 로드: {symbol} ({len(df)}행)")
            return df
        except Exception as e:
            logger.warning(f"캐시 읽기 실패 {symbol}: {e}")
            return None

    def save(self, df: pd.DataFrame, symbol: str, period: str, interval: str) -> None:
        """데이터프레임을 캐시에 저장"""
        path = self._path(self._key(symbol, period, interval))
        try:
            df.to_parquet(path, engine="pyarrow")
            logger.debug(f"캐시 저장: {symbol} ({len(df)}행)")
        except Exception as e:
            logger.warning(f"캐시 저장 실패 {symbol}: {e}")

    def invalidate(self, symbol: str, period: str, interval: str) -> None:
        """특정 캐시 삭제"""
        path = self._path(self._key(symbol, period, interval))
        if os.path.exists(path):
            os.remove(path)

    def clear_all(self) -> int:
        """전체 캐시 삭제. 삭제된 파일 수 반환"""
        count = 0
        for fn in os.listdir(self.cache_dir):
            if fn.endswith(".parquet"):
                os.remove(os.path.join(self.cache_dir, fn))
                count += 1
        logger.info(f"캐시 전체 삭제: {count}개 파일")
        return count
