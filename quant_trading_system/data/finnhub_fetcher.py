# data/finnhub_fetcher.py — Finnhub API 뉴스 수집기
"""
Finnhub API 기반 금융 뉴스 수집기

무료 티어 제한:
  - 60 API calls/min
  - 일반 시장 뉴스: /api/v1/news?category=general
  - 종목별 뉴스:   /api/v1/company-news?symbol={sym}&from=&to=

API 키:
  - https://finnhub.io/register 에서 무료 발급
  - settings.json → news.finnhub_api_key 에 저장

응답 형식:
  [
    {"id": ..., "headline": "...", "summary": "...", "source": "...",
     "url": "...", "datetime": 1234567890, "category": "general", ...}
  ]
"""
from __future__ import annotations

import datetime
import json
import logging
import time
from typing import List, Optional
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

logger = logging.getLogger("quant.news.finnhub")

_BASE_URL     = "https://finnhub.io/api/v1"
_TIMEOUT_SEC  = 8
_MAX_ARTICLES = 50        # 요청당 최대 기사 수
_MIN_CALL_GAP = 1.1       # 60 calls/min → 1초 간격 (안전 마진)

# Finnhub 카테고리
_CATEGORIES = ["general", "forex", "crypto", "merger"]


class FinnhubFetcher:
    """
    Finnhub API 뉴스 수집기

    사용법:
        fetcher = FinnhubFetcher(api_key="your_key")
        items = fetcher.fetch_market_news()              # 일반 시장 뉴스
        items = fetcher.fetch_symbol_news("AAPL")        # 종목별 뉴스
        items = fetcher.fetch_symbol_news("005930", days_back=7)
    """

    def __init__(self, api_key: str = ""):
        self.api_key     = api_key.strip()
        self._last_call  = 0.0
        self._call_count = 0

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    # ── 공개 API ──────────────────────────────────────────────────────────────

    def fetch_market_news(self, category: str = "general",
                          min_id: int = 0) -> List:
        """
        일반 시장 뉴스 수집.
        category: 'general' | 'forex' | 'crypto' | 'merger'
        Returns: List[NewsItem]
        """
        if not self.is_configured:
            logger.debug("[Finnhub] API 키 없음 — 건너뜀")
            return []

        url = (f"{_BASE_URL}/news"
               f"?category={category}"
               f"&minId={min_id}"
               f"&token={self.api_key}")
        try:
            data = self._get(url)
        except Exception as e:
            logger.warning(f"[Finnhub] 시장뉴스 수집 실패: {e}")
            return []

        if not isinstance(data, list):
            return []

        items = [_to_news_item(d, source_tag="Finnhub")
                 for d in data[:_MAX_ARTICLES] if d.get("headline")]
        logger.info(f"[Finnhub] 시장뉴스 {len(items)}건 수집 (category={category})")
        return items

    def fetch_symbol_news(self, symbol: str, days_back: int = 7) -> List:
        """
        종목별 뉴스 수집.
        symbol: 'AAPL', 'TSLA' 등 (한국 종목은 Finnhub 미지원 — 빈 목록 반환)
        Returns: List[NewsItem]
        """
        if not self.is_configured:
            return []

        # Finnhub은 한국 종목코드(숫자.KS) 미지원 → 건너뜀
        if symbol.endswith(".KS") or symbol.endswith(".KQ"):
            return []

        today = datetime.date.today()
        from_dt = (today - datetime.timedelta(days=days_back)).strftime("%Y-%m-%d")
        to_dt   = today.strftime("%Y-%m-%d")

        url = (f"{_BASE_URL}/company-news"
               f"?symbol={symbol}"
               f"&from={from_dt}"
               f"&to={to_dt}"
               f"&token={self.api_key}")
        try:
            data = self._get(url)
        except Exception as e:
            logger.warning(f"[Finnhub] {symbol} 뉴스 수집 실패: {e}")
            return []

        if not isinstance(data, list):
            return []

        items = [_to_news_item(d, source_tag=f"Finnhub/{symbol}")
                 for d in data[:_MAX_ARTICLES] if d.get("headline")]
        logger.info(f"[Finnhub] {symbol} 뉴스 {len(items)}건")
        return items

    def fetch_all_market_news(self) -> List:
        """모든 카테고리 시장 뉴스 수집 (general + forex)"""
        all_items = []
        seen: set = set()
        for cat in ("general", "forex"):
            try:
                for item in self.fetch_market_news(category=cat):
                    key = item.url or item.title
                    if key and key not in seen:
                        seen.add(key)
                        all_items.append(item)
            except Exception:
                pass
        return all_items

    # ── 내부 메서드 ───────────────────────────────────────────────────────────

    def _get(self, url: str) -> object:
        """HTTP GET + 레이트 리밋"""
        elapsed = time.time() - self._last_call
        if elapsed < _MIN_CALL_GAP:
            time.sleep(_MIN_CALL_GAP - elapsed)

        req = Request(url, headers={
            "User-Agent": "QuantTradingSystem/1.0",
            "Accept": "application/json",
        })
        try:
            with urlopen(req, timeout=_TIMEOUT_SEC) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            self._last_call = time.time()
            self._call_count += 1
            return json.loads(raw)
        except HTTPError as e:
            if e.code == 429:
                logger.warning("[Finnhub] 레이트 리밋 초과 — 5초 대기")
                time.sleep(5)
                raise
            elif e.code == 401:
                logger.error("[Finnhub] API 키 인증 실패 (401)")
                raise
            raise
        except URLError as e:
            logger.debug(f"[Finnhub] 네트워크 오류: {e}")
            raise


# ── 변환 유틸 ──────────────────────────────────────────────────────────────────

def _to_news_item(d: dict, source_tag: str = "Finnhub"):
    """Finnhub API 응답 딕셔너리 → NewsItem"""
    from data.news_fetcher import NewsItem

    headline = d.get("headline", "")
    summary  = d.get("summary", "")[:500]
    source   = d.get("source", source_tag)
    url      = d.get("url", "")

    # unix timestamp → ISO string
    ts = d.get("datetime", 0)
    if isinstance(ts, (int, float)) and ts > 0:
        try:
            pub = datetime.datetime.fromtimestamp(ts).isoformat()
        except (OSError, OverflowError):
            pub = datetime.datetime.now().isoformat()
    else:
        pub = datetime.datetime.now().isoformat()

    return NewsItem(
        title=headline,
        body=summary,
        source=source_tag,
        url=url,
        published=pub,
        lang="en",
        weight=0.9,
    )


# ── 싱글턴 ────────────────────────────────────────────────────────────────────

_instance: Optional[FinnhubFetcher] = None


def get_finnhub_fetcher(api_key: str = "") -> FinnhubFetcher:
    global _instance
    if _instance is None or (_instance.api_key != api_key and api_key):
        _instance = FinnhubFetcher(api_key=api_key)
    return _instance
