# data/google_news_fetcher.py — Google News RSS 키워드 기반 뉴스 수집기
"""
Google News RSS 피드 (API 키 불필요)

특징:
  - 키워드 기반 검색 (한국어/영어 동시 지원)
  - 종목 코드 → 회사명 매핑 후 검색
  - URL 패턴: https://news.google.com/rss/search?q={keyword}&hl=ko&gl=KR&ceid=KR:ko
  - NewsItem 형식으로 반환 (news_fetcher.py 호환)
  - 레이트 리밋: 키워드당 30초 간격 권장
"""
from __future__ import annotations

import datetime
import logging
import re
import time
import urllib.parse
from typing import List, Optional
from urllib.request import Request, urlopen

logger = logging.getLogger("quant.news.google")

# ── 기본 검색 키워드 (시장 전반) ──────────────────────────────────────────────

_DEFAULT_KEYWORDS_KO = [
    "주식시장",
    "코스피",
    "금리",
    "환율",
    "반도체",
    "경제",
]

_DEFAULT_KEYWORDS_EN = [
    "stock market",
    "interest rate",
    "inflation",
    "federal reserve",
    "earnings",
]

# 종목코드 → 회사명 (검색용)
_SYMBOL_TO_COMPANY: dict = {
    "005930.KS": "삼성전자",
    "000660.KS": "SK하이닉스",
    "035420.KS": "NAVER",
    "051910.KS": "LG화학",
    "006400.KS": "삼성SDI",
    "035720.KS": "카카오",
    "000270.KS": "기아",
    "005380.KS": "현대차",
    "373220.KS": "LG에너지솔루션",
    "207940.KS": "삼성바이오로직스",
    # 미국 종목
    "AAPL":  "Apple",
    "MSFT":  "Microsoft",
    "NVDA":  "NVIDIA",
    "TSLA":  "Tesla",
    "AMZN":  "Amazon",
    "GOOGL": "Google",
    "META":  "Meta",
}

_MAX_ITEMS_PER_KEYWORD = 15
_REQUEST_TIMEOUT = 8
_MIN_INTERVAL_SEC = 1.0   # 키워드 간 최소 간격


class GoogleNewsFetcher:
    """
    Google News RSS 기반 키워드 뉴스 수집기

    사용법:
        fetcher = GoogleNewsFetcher()
        items = fetcher.fetch_market_news()        # 시장 전반
        items = fetcher.fetch_symbol_news("005930.KS")  # 종목별
    """

    def __init__(self, lang: str = "ko", region: str = "KR",
                 max_per_keyword: int = _MAX_ITEMS_PER_KEYWORD):
        self.lang            = lang
        self.region          = region
        self.max_per_keyword = max_per_keyword
        self._last_request   = 0.0

    # ── 공개 API ──────────────────────────────────────────────────────────────

    def fetch_market_news(self, keywords: List[str] = None) -> List:
        """
        시장 전반 뉴스 수집.
        keywords 미지정 시 _DEFAULT_KEYWORDS_KO + _DEFAULT_KEYWORDS_EN 사용.
        Returns: List[NewsItem]
        """
        from data.news_fetcher import NewsItem
        kws = keywords or (_DEFAULT_KEYWORDS_KO if self.lang == "ko"
                           else _DEFAULT_KEYWORDS_EN)
        all_items: List[NewsItem] = []
        seen_urls: set = set()

        for kw in kws:
            try:
                items = self._fetch_keyword(kw)
                for item in items:
                    key = item.url or item.title
                    if key and key not in seen_urls:
                        seen_urls.add(key)
                        all_items.append(item)
            except Exception as e:
                logger.debug(f"[GoogleNews] 키워드 '{kw}' 수집 실패: {e}")

        logger.info(f"[GoogleNews] 시장뉴스 {len(all_items)}건 수집")
        return all_items

    def fetch_symbol_news(self, symbol: str,
                          extra_keywords: List[str] = None) -> List:
        """
        종목별 뉴스 수집.
        symbol: '005930.KS' 또는 'AAPL' 형식
        Returns: List[NewsItem]
        """
        from data.news_fetcher import NewsItem
        company = _SYMBOL_TO_COMPANY.get(symbol)
        keywords = []
        if company:
            keywords.append(company)
        # 종목코드 자체도 검색 (영문 티커 대상)
        ticker = symbol.split(".")[0]
        if len(ticker) <= 6 and ticker.isalpha():
            keywords.append(ticker)
        if extra_keywords:
            keywords.extend(extra_keywords)
        if not keywords:
            keywords = [ticker]

        all_items: List[NewsItem] = []
        seen_urls: set = set()
        for kw in keywords:
            try:
                items = self._fetch_keyword(kw)
                for item in items:
                    key = item.url or item.title
                    if key and key not in seen_urls:
                        seen_urls.add(key)
                        all_items.append(item)
            except Exception as e:
                logger.debug(f"[GoogleNews] '{kw}' 수집 실패: {e}")

        logger.info(f"[GoogleNews] {symbol} 관련 뉴스 {len(all_items)}건")
        return all_items

    def fetch_batch(self, keywords: List[str]) -> List:
        """복수 키워드 일괄 수집 — 중복 제거"""
        from data.news_fetcher import NewsItem
        all_items: List[NewsItem] = []
        seen: set = set()
        for kw in keywords:
            try:
                for item in self._fetch_keyword(kw):
                    key = item.url or item.title
                    if key and key not in seen:
                        seen.add(key)
                        all_items.append(item)
            except Exception as e:
                logger.debug(f"[GoogleNews] '{kw}' 실패: {e}")
        return all_items

    # ── 내부 메서드 ───────────────────────────────────────────────────────────

    def _fetch_keyword(self, keyword: str) -> List:
        """단일 키워드 Google News RSS 요청"""
        # 레이트 리밋
        elapsed = time.time() - self._last_request
        if elapsed < _MIN_INTERVAL_SEC:
            time.sleep(_MIN_INTERVAL_SEC - elapsed)

        encoded = urllib.parse.quote(keyword)
        if self.lang == "ko":
            url = (f"https://news.google.com/rss/search"
                   f"?q={encoded}&hl=ko&gl=KR&ceid=KR:ko")
        else:
            url = (f"https://news.google.com/rss/search"
                   f"?q={encoded}&hl=en-US&gl=US&ceid=US:en")

        req = Request(url, headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 Chrome/120.0"
            ),
            "Accept": "application/rss+xml, application/xml, text/xml",
        })

        with urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:
            raw = resp.read().decode("utf-8", errors="replace")

        self._last_request = time.time()
        return _parse_google_rss(raw, keyword, self.lang)


# ── RSS 파서 ──────────────────────────────────────────────────────────────────

def _parse_google_rss(xml_text: str, keyword: str, lang: str) -> List:
    """Google News RSS XML → List[NewsItem]"""
    from data.news_fetcher import NewsItem

    items: List[NewsItem] = []
    item_blocks = re.findall(r"<item>(.*?)</item>", xml_text, re.DOTALL)

    for block in item_blocks[:_MAX_ITEMS_PER_KEYWORD]:
        def _tag(t: str) -> str:
            m = re.search(rf"<{t}[^>]*>(.*?)</{t}>", block, re.DOTALL)
            if m:
                txt = m.group(1).strip()
                txt = re.sub(r"<!\[CDATA\[(.*?)\]\]>", r"\1", txt, flags=re.DOTALL)
                txt = re.sub(r"<[^>]+>", " ", txt)
                txt = (txt.replace("&amp;", "&").replace("&lt;", "<")
                           .replace("&gt;", ">").replace("&quot;", '"')
                           .replace("&#39;", "'").replace("&nbsp;", " "))
                return txt.strip()
            return ""

        title = _tag("title")
        if not title:
            continue

        # Google News 제목에서 "- 언론사명" 뒷부분 제거
        title = re.sub(r"\s+-\s+\S+$", "", title).strip()

        source_name = _tag("source") or f"Google News ({keyword})"
        url = _tag("link") or _tag("guid") or ""
        pub = (_tag("pubDate") or _tag("published")
               or datetime.datetime.now().isoformat())
        body = _tag("description") or ""

        items.append(NewsItem(
            title=title,
            body=body[:500],
            source=f"Google News: {keyword}",
            url=url,
            published=pub,
            lang=lang,
            weight=0.85,
        ))

    return items


# ── 싱글턴 ────────────────────────────────────────────────────────────────────

_instance: Optional[GoogleNewsFetcher] = None


def get_google_news_fetcher(lang: str = "ko") -> GoogleNewsFetcher:
    global _instance
    if _instance is None:
        _instance = GoogleNewsFetcher(lang=lang)
    return _instance
