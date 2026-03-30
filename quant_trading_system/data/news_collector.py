# data/news_collector.py — 멀티소스 병렬 뉴스 수집기 (확장판)
"""
확장 수집 전략:
  1차 (제목 스캔): 모든 RSS 소스 병렬 수집 → news_raw 저장 + 중복 제거
  2차 (본문 분석): importance >= 0.7 기사만 추가 요청 → 본문 저장
  3차 (이벤트화): 제목/요약 → StructuredEvent → news_events 저장
  Backfill: GDELT API 를 통한 월/연 단위 히스토리 수집 (무료, 키 불필요)

지원 소스 (60+개, 12 카테고리):
  Macro / MonetaryPolicy / Geopolitics / Industry / Corporate /
  Government / Flow / MarketEvent / Technology / Commodity /
  FinancialMkt / Sentiment
  글로벌: Reuters, CNBC, MarketWatch, Yahoo Finance, FT, WSJ 등
  한국:   연합뉴스, 한국경제, 매일경제, 조선비즈, 이데일리 등 다수

설계:
  - ThreadPoolExecutor 병렬 수집
  - 재시도 3회 (지수 백오프)
  - INSERT OR IGNORE — 증분 업데이트 (신규 기사만 저장)
  - GDELT API 백필 — 월/년 단위 히스토리 (날짜 범위 지원)
"""
from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen

from data.news_db import NewsDB, get_news_db

logger = logging.getLogger("quant.collector")

# ── RSS 소스 카탈로그 (12 카테고리, 60+ 소스) ──────────────────────────────

_RSS_CATALOG: List[Dict] = [

    # ════════════════════════════════════════════════════════
    # 1. MACRO  거시경제 — GDP·고용·CPI·경기지표
    # ════════════════════════════════════════════════════════
    {"name": "CNBC Economy",
     "url": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258",
     "lang": "en", "weight": 0.9, "category": "Macro"},
    {"name": "Reuters Business",
     "url": "https://feeds.reuters.com/reuters/businessNews",
     "lang": "en", "weight": 1.0, "category": "Macro"},
    {"name": "Reuters Top News",
     "url": "https://feeds.reuters.com/reuters/topNews",
     "lang": "en", "weight": 0.9, "category": "Macro"},
    {"name": "MarketWatch Headlines",
     "url": "https://feeds.marketwatch.com/marketwatch/realtimeheadlines/",
     "lang": "en", "weight": 0.9, "category": "Macro"},
    {"name": "Yahoo Finance",
     "url": "https://finance.yahoo.com/news/rssindex",
     "lang": "en", "weight": 1.0, "category": "Macro"},
    {"name": "AP Business",
     "url": "https://feeds.apnews.com/rss/business",
     "lang": "en", "weight": 0.8, "category": "Macro"},
    {"name": "연합뉴스 경제",
     "url": "https://www.yonhapnewstv.co.kr/category/news/economy/feed/",
     "lang": "ko", "weight": 1.0, "category": "Macro"},
    {"name": "한국경제 경제",
     "url": "https://www.hankyung.com/feed/economy",
     "lang": "ko", "weight": 0.9, "category": "Macro"},
    {"name": "매일경제",
     "url": "https://www.mk.co.kr/rss/30000001/",
     "lang": "ko", "weight": 0.9, "category": "Macro"},
    {"name": "서울경제",
     "url": "https://www.sedaily.com/Rss/01",
     "lang": "ko", "weight": 0.8, "category": "Macro"},

    # ════════════════════════════════════════════════════════
    # 2. MonetaryPolicy  통화정책 — 금리·연준·중앙은행
    # ════════════════════════════════════════════════════════
    {"name": "FedReserve Press",
     "url": "https://www.federalreserve.gov/feeds/press_all.xml",
     "lang": "en", "weight": 1.3, "category": "MonetaryPolicy"},
    {"name": "FedReserve Speeches",
     "url": "https://www.federalreserve.gov/feeds/speeches.xml",
     "lang": "en", "weight": 1.2, "category": "MonetaryPolicy"},
    {"name": "ECB Press",
     "url": "https://www.ecb.europa.eu/rss/press.html",
     "lang": "en", "weight": 1.1, "category": "MonetaryPolicy"},
    {"name": "BIS News",
     "url": "https://www.bis.org/rss/home.rss",
     "lang": "en", "weight": 1.0, "category": "MonetaryPolicy"},
    {"name": "BOE News",
     "url": "https://www.bankofengland.co.uk/rss/news",
     "lang": "en", "weight": 0.9, "category": "MonetaryPolicy"},
    {"name": "Treasury Releases",
     "url": "https://home.treasury.gov/system/files/rss/press-releases-feed.xml",
     "lang": "en", "weight": 1.0, "category": "MonetaryPolicy"},
    {"name": "CNBC FedWatch",
     "url": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664",
     "lang": "en", "weight": 0.9, "category": "MonetaryPolicy"},
    {"name": "뉴시스 금융",
     "url": "https://newsis.com/RSS/finance.xml",
     "lang": "ko", "weight": 0.8, "category": "MonetaryPolicy"},

    # ════════════════════════════════════════════════════════
    # 3. Geopolitics  지정학 — 전쟁·분쟁·제재·외교
    # ════════════════════════════════════════════════════════
    {"name": "Reuters World",
     "url": "https://feeds.reuters.com/Reuters/worldNews",
     "lang": "en", "weight": 1.0, "category": "Geopolitics"},
    {"name": "AP World News",
     "url": "https://feeds.apnews.com/rss/world-news",
     "lang": "en", "weight": 0.9, "category": "Geopolitics"},
    {"name": "BBC World",
     "url": "https://feeds.bbci.co.uk/news/world/rss.xml",
     "lang": "en", "weight": 0.9, "category": "Geopolitics"},
    {"name": "Al Jazeera",
     "url": "https://www.aljazeera.com/xml/rss/all.xml",
     "lang": "en", "weight": 0.8, "category": "Geopolitics"},
    {"name": "NPR World",
     "url": "https://feeds.npr.org/1004/rss.xml",
     "lang": "en", "weight": 0.7, "category": "Geopolitics"},
    {"name": "연합뉴스 국제",
     "url": "https://www.yonhapnewstv.co.kr/category/news/international/feed/",
     "lang": "ko", "weight": 0.9, "category": "Geopolitics"},

    # ════════════════════════════════════════════════════════
    # 4. Industry  산업 — 반도체·자동차·에너지·화학
    # ════════════════════════════════════════════════════════
    {"name": "IEEE Spectrum",
     "url": "https://spectrum.ieee.org/feeds/feed.rss",
     "lang": "en", "weight": 0.8, "category": "Industry"},
    {"name": "Semiconductor Industry",
     "url": "https://www.semiconductors.org/feed/",
     "lang": "en", "weight": 1.0, "category": "Industry"},
    {"name": "EE Times",
     "url": "https://www.eetimes.com/rss/",
     "lang": "en", "weight": 0.8, "category": "Industry"},
    {"name": "Wards Auto",
     "url": "https://www.wardsauto.com/rss/all",
     "lang": "en", "weight": 0.7, "category": "Industry"},
    {"name": "Chemical & Engineering News",
     "url": "https://cen.acs.org/rss/latest.xml",
     "lang": "en", "weight": 0.7, "category": "Industry"},
    {"name": "전자신문",
     "url": "https://rss.etnews.com/Section901",
     "lang": "ko", "weight": 0.9, "category": "Industry"},
    {"name": "한국경제 산업",
     "url": "https://www.hankyung.com/feed/industry",
     "lang": "ko", "weight": 0.9, "category": "Industry"},

    # ════════════════════════════════════════════════════════
    # 5. Corporate  기업 — 실적·M&A·공시·배당
    # ════════════════════════════════════════════════════════
    {"name": "Seeking Alpha Market Currents",
     "url": "https://seekingalpha.com/market_currents.xml",
     "lang": "en", "weight": 0.8, "category": "Corporate"},
    {"name": "Investing.com Stocks",
     "url": "https://www.investing.com/rss/stock_stock_picks.rss",
     "lang": "en", "weight": 0.8, "category": "Corporate"},
    {"name": "CNBC Earnings",
     "url": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=15839069",
     "lang": "en", "weight": 0.9, "category": "Corporate"},
    {"name": "MarketWatch Earnings",
     "url": "https://feeds.marketwatch.com/marketwatch/topstories/",
     "lang": "en", "weight": 0.8, "category": "Corporate"},
    {"name": "조선비즈",
     "url": "https://biz.chosun.com/site/data/rss/rss.xml",
     "lang": "ko", "weight": 0.9, "category": "Corporate"},
    {"name": "이데일리 증권",
     "url": "https://www.edaily.co.kr/rss/01400000000000",
     "lang": "ko", "weight": 0.8, "category": "Corporate"},
    {"name": "머니투데이",
     "url": "https://news.mt.co.kr/mtview.php?type=rss",
     "lang": "ko", "weight": 0.8, "category": "Corporate"},
    {"name": "한국경제 증권",
     "url": "https://www.hankyung.com/feed/finance",
     "lang": "ko", "weight": 0.9, "category": "Corporate"},

    # ════════════════════════════════════════════════════════
    # 6. Government  정부/규제 — 정책·법안·규제·공정위
    # ════════════════════════════════════════════════════════
    {"name": "SEC Press Releases",
     "url": "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=&dateb=&owner=include&count=40&search_text=&output=atom",
     "lang": "en", "weight": 1.1, "category": "Government"},
    {"name": "FTC News",
     "url": "https://www.ftc.gov/news-events/news/rss.xml",
     "lang": "en", "weight": 1.0, "category": "Government"},
    {"name": "EU Commission",
     "url": "https://ec.europa.eu/commission/presscorner/api/documents?documentType=IP&lang=en&pageSize=20&pageNumber=1&sortField=publicationDate&sortOrder=DESC&output=rss",
     "lang": "en", "weight": 0.8, "category": "Government"},
    {"name": "Reuters Politics",
     "url": "https://feeds.reuters.com/Reuters/PoliticsNews",
     "lang": "en", "weight": 0.8, "category": "Government"},
    {"name": "한국경제 정책",
     "url": "https://www.hankyung.com/feed/politics",
     "lang": "ko", "weight": 0.9, "category": "Government"},
    {"name": "매일경제 정책",
     "url": "https://www.mk.co.kr/rss/50400012/",
     "lang": "ko", "weight": 0.8, "category": "Government"},

    # ════════════════════════════════════════════════════════
    # 7. Flow  수급 — 외국인/기관 매매·ETF·펀드플로우
    # ════════════════════════════════════════════════════════
    {"name": "ETF.com",
     "url": "https://www.etf.com/sections/news-and-markets/rss.xml",
     "lang": "en", "weight": 0.8, "category": "Flow"},
    {"name": "Seeking Alpha ETF",
     "url": "https://seekingalpha.com/tag/etf-portfolio-strategy.xml",
     "lang": "en", "weight": 0.7, "category": "Flow"},
    {"name": "Investing.com Global Markets",
     "url": "https://www.investing.com/rss/news.rss",
     "lang": "en", "weight": 0.8, "category": "Flow"},
    {"name": "이데일리 투자",
     "url": "https://www.edaily.co.kr/rss/01500000000000",
     "lang": "ko", "weight": 0.8, "category": "Flow"},

    # ════════════════════════════════════════════════════════
    # 8. MarketEvent  시장이벤트 — IPO·지수재편·옵션만기
    # ════════════════════════════════════════════════════════
    {"name": "IPO Monitor",
     "url": "https://www.iposcoop.com/rss/",
     "lang": "en", "weight": 0.8, "category": "MarketEvent"},
    {"name": "Yahoo Finance Earnings",
     "url": "https://finance.yahoo.com/rss/2.0/headline?s=%5EGSPC&region=US&lang=en-US",
     "lang": "en", "weight": 0.8, "category": "MarketEvent"},
    {"name": "CNBC Markets",
     "url": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=15839069",
     "lang": "en", "weight": 0.8, "category": "MarketEvent"},
    {"name": "한국경제 시장",
     "url": "https://www.hankyung.com/feed/market",
     "lang": "ko", "weight": 0.8, "category": "MarketEvent"},

    # ════════════════════════════════════════════════════════
    # 9. Technology  기술 — AI·반도체·플랫폼·바이오
    # ════════════════════════════════════════════════════════
    {"name": "TechCrunch",
     "url": "https://techcrunch.com/feed/",
     "lang": "en", "weight": 0.9, "category": "Technology"},
    {"name": "Ars Technica",
     "url": "https://feeds.arstechnica.com/arstechnica/index",
     "lang": "en", "weight": 0.8, "category": "Technology"},
    {"name": "The Verge",
     "url": "https://www.theverge.com/rss/index.xml",
     "lang": "en", "weight": 0.8, "category": "Technology"},
    {"name": "MIT Technology Review",
     "url": "https://www.technologyreview.com/feed/",
     "lang": "en", "weight": 0.9, "category": "Technology"},
    {"name": "Wired",
     "url": "https://www.wired.com/feed/rss",
     "lang": "en", "weight": 0.7, "category": "Technology"},
    {"name": "전자신문 IT",
     "url": "https://rss.etnews.com/Section902",
     "lang": "ko", "weight": 0.9, "category": "Technology"},
    {"name": "블로터",
     "url": "https://www.bloter.net/feed",
     "lang": "ko", "weight": 0.7, "category": "Technology"},

    # ════════════════════════════════════════════════════════
    # 10. Commodity  원자재 — 원유·금·구리·농산물
    # ════════════════════════════════════════════════════════
    {"name": "Oil Price News",
     "url": "https://oilprice.com/rss/main",
     "lang": "en", "weight": 1.0, "category": "Commodity"},
    {"name": "Reuters Commodities",
     "url": "https://feeds.reuters.com/reuters/commoditiesNews",
     "lang": "en", "weight": 1.0, "category": "Commodity"},
    {"name": "Kitco Gold",
     "url": "https://www.kitco.com/rss/",
     "lang": "en", "weight": 0.9, "category": "Commodity"},
    {"name": "Seeking Alpha Commodities",
     "url": "https://seekingalpha.com/tag/commodities.xml",
     "lang": "en", "weight": 0.7, "category": "Commodity"},
    {"name": "Mining.com",
     "url": "https://www.mining.com/feed/",
     "lang": "en", "weight": 0.7, "category": "Commodity"},
    {"name": "매일경제 원자재",
     "url": "https://www.mk.co.kr/rss/50300009/",
     "lang": "ko", "weight": 0.8, "category": "Commodity"},

    # ════════════════════════════════════════════════════════
    # 11. FinancialMkt  금융시장 — 채권·외환·파생
    # ════════════════════════════════════════════════════════
    {"name": "Investing.com Forex",
     "url": "https://www.investing.com/rss/forex.rss",
     "lang": "en", "weight": 0.8, "category": "FinancialMkt"},
    {"name": "Investing.com Bonds",
     "url": "https://www.investing.com/rss/bond.rss",
     "lang": "en", "weight": 0.8, "category": "FinancialMkt"},
    {"name": "Reuters Finance",
     "url": "https://feeds.reuters.com/reuters/financialsNews",
     "lang": "en", "weight": 0.9, "category": "FinancialMkt"},
    {"name": "MarketWatch Economy",
     "url": "https://feeds.marketwatch.com/marketwatch/economy-politics/",
     "lang": "en", "weight": 0.8, "category": "FinancialMkt"},
    {"name": "FX Street",
     "url": "https://www.fxstreet.com/rss/news",
     "lang": "en", "weight": 0.7, "category": "FinancialMkt"},
    {"name": "한국경제 외환",
     "url": "https://www.hankyung.com/feed/international",
     "lang": "ko", "weight": 0.8, "category": "FinancialMkt"},

    # ════════════════════════════════════════════════════════
    # 12. Sentiment  심리 — 투자심리·공포탐욕·섹터센티먼트
    # ════════════════════════════════════════════════════════
    {"name": "Zero Hedge",
     "url": "https://feeds.feedburner.com/zerohedge/feed",
     "lang": "en", "weight": 0.6, "category": "Sentiment"},
    {"name": "Seeking Alpha Market Outlook",
     "url": "https://seekingalpha.com/market-outlook.xml",
     "lang": "en", "weight": 0.8, "category": "Sentiment"},
    {"name": "CNBC Investing",
     "url": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114",
     "lang": "en", "weight": 0.8, "category": "Sentiment"},
    {"name": "MarketWatch Opinion",
     "url": "https://feeds.marketwatch.com/marketwatch/marketpulse/",
     "lang": "en", "weight": 0.7, "category": "Sentiment"},
    {"name": "머니투데이 시황",
     "url": "https://news.mt.co.kr/mtview.php?type=rss&sec=market",
     "lang": "ko", "weight": 0.8, "category": "Sentiment"},
]

# ── GDELT 백필 설정 ───────────────────────────────────────────────────────────
# GDELT Global Knowledge Graph (무료, API 키 불필요)
# https://blog.gdeltproject.org/gdelt-2-0-our-global-world-in-realtime/
_GDELT_DOC_URL = (
    "https://api.gdeltproject.org/api/v2/doc/doc"
    "?query={query}&mode=artlist"
    "&startdatetime={start}&enddatetime={end}"
    "&maxrecords=250&format=json"
)

# 카테고리별 GDELT 검색 키워드
_GDELT_KEYWORDS: Dict[str, str] = {
    "Macro":          "economy GDP inflation employment",
    "MonetaryPolicy": "Federal Reserve interest rate central bank",
    "Geopolitics":    "geopolitics war sanctions conflict",
    "Industry":       "semiconductor manufacturing supply chain",
    "Corporate":      "earnings profit revenue merger acquisition",
    "Government":     "regulation policy government legislation",
    "Flow":           "investment fund ETF capital flow",
    "MarketEvent":    "IPO stock market index rebalancing",
    "Technology":     "artificial intelligence technology innovation",
    "Commodity":      "oil gold copper commodity energy",
    "FinancialMkt":   "bond forex currency derivative",
    "Sentiment":      "investor sentiment fear greed market mood",
}

# 노이즈 필터 패턴 (정규식)
_NOISE_PATTERNS = [
    r"^\d+개.*공모$",
    r"무료.*세미나",
    r"광고|스폰서|협찬",
    r"오늘의.*운세",
    r"추천종목\s*\d+선",
    r"(오늘|내일).*주가 전망\s*\[",
    r"급등주|작전주|리딩방",
]

_MAX_PER_SOURCE   = 50      # 소스당 최대 기사
_RETRY_COUNT      = 3       # 재시도 횟수
_RETRY_DELAY_BASE = 1.5     # 지수 백오프 기반 (초)
_REQUEST_TIMEOUT  = 10      # HTTP 타임아웃 (초)
_PARALLEL_WORKERS = 8       # 병렬 수집 스레드 수


# ── 데이터 클래스 ─────────────────────────────────────────────────────────────

@dataclass
class RawArticle:
    """수집된 원본 기사"""
    title:     str
    body:      str          # 요약 500자 이내
    source:    str
    url:       str
    published: str          # ISO 문자열
    lang:      str
    weight:    float
    category:  str = ""

    @property
    def published_dt(self) -> datetime.datetime:
        for fmt in ("%a, %d %b %Y %H:%M:%S %z",
                    "%Y-%m-%dT%H:%M:%S%z",
                    "%Y-%m-%dT%H:%M:%S",
                    "%Y-%m-%d %H:%M:%S",
                    "%Y-%m-%d"):
            try:
                s = self.published[:25]
                return datetime.datetime.strptime(s, fmt)
            except ValueError:
                continue
        return datetime.datetime.now()

    def to_db_dict(self) -> dict:
        return {
            "title":     self.title,
            "body":      self.body[:500],
            "source":    self.source,
            "url":       self.url,
            "published": self.published,
            "lang":      self.lang,
            "weight":    self.weight,
        }


# ── RSS 파서 ──────────────────────────────────────────────────────────────────

def _parse_rss(xml_text: str, source_name: str,
               lang: str, weight: float, category: str) -> List[RawArticle]:
    """정규표현식 기반 범용 RSS/Atom 파서"""
    articles: List[RawArticle] = []

    # Atom <entry> 또는 RSS <item> 지원
    item_blocks = re.findall(r"<(?:item|entry)>(.*?)</(?:item|entry)>",
                             xml_text, re.DOTALL)

    for block in item_blocks[:_MAX_PER_SOURCE]:

        def _tag(t: str) -> str:
            m = re.search(rf"<{t}[^>]*>(.*?)</{t}>", block, re.DOTALL)
            if m:
                txt = m.group(1)
                txt = re.sub(r"<!\[CDATA\[(.*?)\]\]>", r"\1", txt, flags=re.DOTALL)
                txt = re.sub(r"<[^>]+>", " ", txt)
                txt = (txt.replace("&amp;", "&").replace("&lt;", "<")
                           .replace("&gt;", ">").replace("&quot;", '"')
                           .replace("&#39;", "'").replace("&nbsp;", " "))
                return " ".join(txt.split())
            return ""

        title = _tag("title")
        if not title or len(title) < 5:
            continue
        if any(re.search(p, title, re.IGNORECASE) for p in _NOISE_PATTERNS):
            continue

        body  = _tag("description") or _tag("summary") or _tag("content:encoded") or ""
        url   = _tag("link") or _tag("guid") or ""
        pub   = (_tag("pubDate") or _tag("published") or
                 _tag("dc:date") or _tag("updated") or
                 datetime.datetime.now().isoformat())

        articles.append(RawArticle(
            title=title[:300], body=body[:500],
            source=source_name, url=url[:500],
            published=pub, lang=lang, weight=weight, category=category,
        ))

    return articles


# ── 단일 소스 수집 ────────────────────────────────────────────────────────────

def _fetch_one_source(src: Dict, retries: int = _RETRY_COUNT) -> Tuple[str, List[RawArticle]]:
    """단일 소스 RSS 수집 (재시도 포함). 실패 시 빈 목록 반환."""
    name = src["name"]
    url  = src["url"]

    for attempt in range(retries):
        try:
            req  = Request(url, headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                "Accept": "application/rss+xml, application/xml, text/xml, */*",
                "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8",
            })
            with urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            articles = _parse_rss(raw, name, src["lang"], src["weight"], src.get("category",""))
            logger.debug(f"[{name}] {len(articles)}건 수집")
            return name, articles

        except (URLError, HTTPError, TimeoutError) as e:
            if attempt < retries - 1:
                delay = _RETRY_DELAY_BASE ** attempt
                logger.debug(f"[{name}] 재시도 {attempt+1}/{retries}: {e} → {delay:.1f}s 대기")
                time.sleep(delay)
            else:
                logger.debug(f"[{name}] 수집 실패 (최종): {e}")
                return name, []
        except Exception as e:
            logger.debug(f"[{name}] 예외: {e}")
            return name, []

    return name, []


# ── 메인 수집기 ───────────────────────────────────────────────────────────────

class NewsCollector:
    """
    멀티소스 병렬 뉴스 수집기

    사용법:
        collector = NewsCollector(db=get_news_db())

        # 1회 수집 (자동 DB 저장)
        stats = collector.collect_once(progress_cb=print)

        # 자동 반복 수집 시작
        collector.start_continuous(interval_minutes=15)

        # 백필 (과거 히스토리 수집 — RSS 지원 범위 내)
        collector.backfill(days=7)
    """

    def __init__(
        self,
        db: Optional[NewsDB] = None,
        sources: Optional[List[Dict]] = None,
        max_workers: int = _PARALLEL_WORKERS,
    ):
        self.db          = db or get_news_db()
        self.sources     = sources or _RSS_CATALOG
        self.max_workers = max_workers
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_stats: Dict = {}

    # ── 공개 API ─────────────────────────────────────────────────────────────

    def collect_once(
        self,
        progress_cb: Optional[Callable[[str], None]] = None,
        auto_process: bool = True,
    ) -> Dict:
        """
        모든 소스 병렬 수집 → DB 저장 → (선택) 이벤트 변환.

        Returns:
            수집 통계 딕셔너리
        """
        t0 = time.time()
        all_articles: List[RawArticle] = []
        source_counts: Dict[str, int] = {}
        failed_sources: List[str] = []

        _log = progress_cb or logger.info

        _log(f"[수집 시작] {len(self.sources)}개 소스 병렬 수집...")

        with ThreadPoolExecutor(max_workers=self.max_workers) as exe:
            futures = {exe.submit(_fetch_one_source, src): src
                       for src in self.sources}
            for future in as_completed(futures):
                src = futures[future]
                try:
                    name, articles = future.result()
                    if articles:
                        all_articles.extend(articles)
                        source_counts[name] = len(articles)
                    else:
                        failed_sources.append(src["name"])
                except Exception as e:
                    failed_sources.append(src["name"])
                    logger.debug(f"수집 future 예외 ({src['name']}): {e}")

        _log(f"[1차 완료] {len(all_articles)}건 수집 "
             f"({len(source_counts)}/{len(self.sources)} 소스 성공)")

        # ── DB 저장 (중복 자동 제거) ─────────────────────────────────────
        if all_articles:
            dicts = [a.to_db_dict() for a in all_articles]
            inserted, skipped = self.db.insert_raw(dicts)
            _log(f"[DB 저장] 신규 {inserted}건 / 중복 {skipped}건 스킵")
        else:
            inserted, skipped = 0, 0

        # ── 이벤트 변환 (선택) ────────────────────────────────────────────
        processed = 0
        if auto_process and all_articles:
            processed = self._process_to_events(all_articles, progress_cb=_log)

        elapsed = time.time() - t0
        stats = {
            "total_fetched":  len(all_articles),
            "inserted":       inserted,
            "skipped":        skipped,
            "events_created": processed,
            "sources_ok":     len(source_counts),
            "sources_fail":   len(failed_sources),
            "elapsed_sec":    round(elapsed, 2),
            "source_counts":  source_counts,
            "failed_sources": failed_sources,
        }
        self._last_stats = stats
        return stats

    def backfill(
        self,
        days: int = 7,
        progress_cb: Optional[Callable[[str], None]] = None,
    ) -> Dict:
        """RSS 범위 내 단기 백필 (최대 3일). 장기 백필은 backfill_range() 사용."""
        _log = progress_cb or logger.info
        _log(f"[RSS 백필] {days}일치 수집 시작...")
        stats = self.collect_once(progress_cb=progress_cb, auto_process=True)
        stats["backfill_days"] = days
        return stats

    def backfill_range(
        self,
        start_dt: datetime.datetime,
        end_dt: Optional[datetime.datetime] = None,
        granularity: str = "month",
        progress_cb: Optional[Callable[[str], None]] = None,
    ) -> Dict:
        """
        GDELT API를 이용한 월/년 단위 히스토리 백필.

        Args:
            start_dt:    수집 시작 날짜
            end_dt:      수집 종료 날짜 (기본: 오늘)
            granularity: "day" | "month" | "year"
            progress_cb: 진행 상황 콜백 (text)

        Returns:
            통계 딕셔너리 {total, inserted, skipped, periods, errors}
        """
        _log = progress_cb or logger.info
        end_dt = end_dt or datetime.datetime.now()

        periods = _build_periods(start_dt, end_dt, granularity)
        total_inserted = 0
        total_skipped  = 0
        total_errors   = 0

        _log(f"[GDELT 백필] {start_dt.strftime('%Y-%m')} ~ "
             f"{end_dt.strftime('%Y-%m')} ({len(periods)}개 구간, {granularity} 단위)")

        for i, (p_start, p_end) in enumerate(periods):
            if self._stop_event.is_set():
                _log("[백필 중단] 사용자 요청")
                break

            p_start_s = p_start.strftime("%Y-%m-%d")
            p_end_s   = p_end.strftime("%Y-%m-%d")
            pct       = int((i + 1) / len(periods) * 100)

            # ── 이미 수집된 기간 스킵 ──────────────────────────────────
            if self.db.is_backfilled(p_start_s, p_end_s, "*"):
                _log(f"[{pct:3d}%] {p_start_s} ~ {p_end_s} "
                     f"[이미 수집됨 — 스킵]")
                continue

            _log(f"[{pct:3d}%] {p_start_s} ~ {p_end_s} 수집 중 "
                 f"(총 {total_inserted}건 저장)")

            cats_to_fetch = list(_GDELT_KEYWORDS.keys())
            period_ins = 0
            period_skip = 0

            with ThreadPoolExecutor(max_workers=4) as exe:
                futs = {
                    exe.submit(self._fetch_gdelt, cat, p_start, p_end): cat
                    for cat in cats_to_fetch
                }
                # 구간당 최대 60초 — 초과 시 남은 future 포기
                try:
                    for fut in as_completed(futs, timeout=60):
                        if self._stop_event.is_set():
                            break
                        cat = futs[fut]
                        try:
                            articles = fut.result(timeout=2)
                            if articles:
                                dicts = [a.to_db_dict() for a in articles]
                                ins, skip = self.db.insert_raw(dicts)
                                period_ins  += ins
                                period_skip += skip
                        except Exception as e:
                            total_errors += 1
                            logger.debug(f"GDELT [{cat}] 오류: {e}")
                except Exception:
                    # TimeoutError or other: 해당 구간 포기, 다음 구간 진행
                    total_errors += 1
                    _log(f"  └ ⚠ 구간 타임아웃 — 부분 저장 후 계속")

            total_inserted += period_ins
            total_skipped  += period_skip
            _log(f"  └ 신규 {period_ins}건 / 중복(URL) {period_skip}건")

            # ── 백필 이력 기록 ──────────────────────────────────────────
            self.db.mark_backfilled(
                p_start_s, p_end_s, "*",
                inserted=period_ins, skipped=period_skip,
                source="gdelt",
            )

            # GDELT rate-limit 방지 (1초 대기)
            time.sleep(1.0)

        _log(f"[백필 완료] 총 {total_inserted}건 저장 / {total_skipped}건 중복")
        return {
            "total_inserted": total_inserted,
            "total_skipped":  total_skipped,
            "periods":        len(periods),
            "errors":         total_errors,
        }

    def backfill_months(
        self,
        months_back: int = 12,
        progress_cb: Optional[Callable[[str], None]] = None,
    ) -> Dict:
        """최근 N개월 치 GDELT 백필."""
        end   = datetime.datetime.now()
        # N개월 전 (month 연산)
        year  = end.year - (months_back // 12)
        month = end.month - (months_back % 12)
        if month <= 0:
            year  -= 1
            month += 12
        start = end.replace(year=year, month=month, day=1,
                             hour=0, minute=0, second=0, microsecond=0)
        return self.backfill_range(start, end, granularity="month",
                                   progress_cb=progress_cb)

    def backfill_years(
        self,
        years_back: int = 3,
        progress_cb: Optional[Callable[[str], None]] = None,
    ) -> Dict:
        """최근 N년 치 GDELT 백필 (분기 단위로 수집)."""
        end   = datetime.datetime.now()
        start = end.replace(year=end.year - years_back, month=1, day=1,
                             hour=0, minute=0, second=0, microsecond=0)
        return self.backfill_range(start, end, granularity="month",
                                   progress_cb=progress_cb)

    # ── GDELT 수집 ────────────────────────────────────────────────────────────

    def _fetch_gdelt(
        self,
        category: str,
        start_dt: datetime.datetime,
        end_dt: datetime.datetime,
    ) -> List[RawArticle]:
        """GDELT Document API 로 특정 카테고리·기간 기사 수집."""
        keyword = _GDELT_KEYWORDS.get(category, category.lower())
        url = _GDELT_DOC_URL.format(
            query=keyword.replace(" ", "%20"),
            start=start_dt.strftime("%Y%m%d%H%M%S"),
            end=end_dt.strftime("%Y%m%d%H%M%S"),
        )

        try:
            req = Request(url, headers={
                "User-Agent": "Mozilla/5.0 (compatible; QuantBot/1.0)",
                "Accept": "application/json",
            })
            with urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="replace"))
        except Exception as e:
            logger.debug(f"GDELT [{category}] 수집 실패: {e}")
            return []

        articles: List[RawArticle] = []
        for art in (data.get("articles") or []):
            title = art.get("title", "").strip()
            if not title or len(title) < 5:
                continue
            articles.append(RawArticle(
                title=title[:300],
                body=(art.get("seendate", "") + " " + title)[:500],
                source=art.get("domain", "gdelt"),
                url=art.get("url", "")[:500],
                published=art.get("seendate",
                                  datetime.datetime.now().isoformat()),
                lang=art.get("language", "en")[:2],
                weight=_GDELT_KEYWORDS.get(category, {}) and 0.7 or 0.7,
                category=category,
            ))

        return articles

    def start_continuous(
        self,
        interval_minutes: int = 15,
        progress_cb: Optional[Callable[[str], None]] = None,
    ) -> None:
        """백그라운드 연속 수집 시작"""
        if self._thread and self._thread.is_alive():
            logger.warning("이미 수집 중입니다")
            return

        self._stop_event.clear()

        def _loop():
            while not self._stop_event.is_set():
                try:
                    self.collect_once(progress_cb=progress_cb)
                except Exception as e:
                    logger.error(f"연속 수집 오류: {e}")
                if self._stop_event.wait(interval_minutes * 60):
                    break

        self._thread = threading.Thread(target=_loop, daemon=True,
                                        name="news-collector")
        self._thread.start()
        logger.info(f"뉴스 연속 수집 시작 (간격: {interval_minutes}분)")

    def stop_continuous(self) -> None:
        """연속 수집 중지"""
        self._stop_event.set()
        logger.info("뉴스 연속 수집 중지 요청")

    def get_last_stats(self) -> Dict:
        return self._last_stats.copy()

    # ── 이벤트 변환 ───────────────────────────────────────────────────────────

    def _process_to_events(
        self,
        articles: List[RawArticle],
        progress_cb: Optional[Callable[[str], None]] = None,
    ) -> int:
        """
        수집 기사 → StructuredEvent 변환 → DB 저장.
        2단계 분석:
          1단계: 제목 전체 → 빠른 분류 (NewsEventCategorizer)
          2단계: importance>=0.7 기사 → NLP 심화 분석
        """
        try:
            from features.external_env.categorizer import NewsEventCategorizer
            from features.external_env.nlp_analyzer import get_analyzer
        except ImportError as e:
            logger.warning(f"external_env 미설치, 이벤트 변환 건너뜀: {e}")
            return 0

        categorizer = NewsEventCategorizer()
        nlp_analyzer = get_analyzer()

        evt_dicts: List[Dict] = []
        important_count = 0

        for article in articles:
            try:
                # ── 1단계: 제목 기반 빠른 분류 ────────────────────────
                ts = article.published_dt
                evt = categorizer.classify(
                    title=article.title,
                    summary=article.body,
                    url=article.url,
                    ts=ts,
                )

                # ── 2단계: 중요 기사 NLP 심화 분석 ────────────────────
                nlp_result = nlp_analyzer.analyze(article.title, article.body)
                importance = nlp_result.get("importance", 0.0)

                evt.sentiment_score = nlp_result.get("sentiment_score", 0.0)
                evt.importance      = importance
                evt.confidence      = max(evt.confidence,
                                          nlp_result.get("confidence", 0.0))
                if nlp_result.get("keywords"):
                    evt.keywords = nlp_result["keywords"][:8]
                evt.compute_score()

                if importance >= 0.7:
                    important_count += 1

                # ── DB 저장용 딕셔너리 변환 ───────────────────────────
                raw_id = _make_article_id(article)
                self.db.update_raw_importance(raw_id, importance)

                d = _event_to_dict(evt, raw_id)
                evt_dicts.append(d)

            except Exception as e:
                logger.debug(f"이벤트 변환 실패 ({article.title[:40]}): {e}")

        if evt_dicts:
            stored = self.db.insert_events_bulk(evt_dicts)
            if progress_cb:
                progress_cb(f"[이벤트 저장] {stored}건 "
                            f"(중요 기사 {important_count}건 심화 분석)")
            return stored

        return 0

    # ── 클러스터링 ────────────────────────────────────────────────────────────

    def deduplicate_events(
        self,
        since_dt: Optional[datetime.datetime] = None,
        similarity_threshold: float = 0.75,
    ) -> int:
        """
        최근 이벤트 클러스터링 → 대표 기사 갱신.
        features/news_cluster.py 의 NewsClusterer 사용.
        """
        try:
            from features.news_cluster import NewsClusterer
        except ImportError:
            logger.warning("news_cluster 미설치, 클러스터링 건너뜀")
            return 0

        since = since_dt or (datetime.datetime.now() - datetime.timedelta(days=3))
        events = self.db.get_events(since_dt=since, only_representative=False)
        if not events:
            return 0

        clusterer = NewsClusterer(similarity_threshold=similarity_threshold)
        clusters  = clusterer.cluster(events)
        updated   = 0

        with self.db._tx() as conn:
            for cluster_id, members in clusters.items():
                rep_id = members[0]["event_id"]  # 첫 번째가 대표
                repeat = len(members)

                # 언론사 다양성 계산
                raw_ids = [m.get("raw_id", "") for m in members]
                sources = set()
                for rid in raw_ids:
                    r = conn.execute(
                        "SELECT source FROM news_raw WHERE id=?", (rid,)
                    ).fetchone()
                    if r:
                        sources.add(r["source"])
                diversity = min(1.0, len(sources) / max(5, len(members)))

                for m in members:
                    is_rep = 1 if m["event_id"] == rep_id else 0
                    conn.execute("""
                        UPDATE news_events
                        SET cluster_id=?, is_representative=?,
                            repeat_count=?, source_diversity=?
                        WHERE event_id=?
                    """, (cluster_id, is_rep, repeat, diversity, m["event_id"]))
                    updated += 1

        logger.info(f"클러스터링 완료: {len(clusters)}개 클러스터, {updated}건 갱신")
        return updated


# ── 유틸 ──────────────────────────────────────────────────────────────────────

def _build_periods(
    start_dt: datetime.datetime,
    end_dt: datetime.datetime,
    granularity: str,
) -> List[Tuple[datetime.datetime, datetime.datetime]]:
    """날짜 범위를 granularity 단위로 분할."""
    periods = []
    cur = start_dt.replace(hour=0, minute=0, second=0, microsecond=0)

    if granularity == "day":
        delta = datetime.timedelta(days=1)
        while cur < end_dt:
            nxt = min(cur + delta, end_dt)
            periods.append((cur, nxt))
            cur = nxt
    elif granularity == "month":
        while cur < end_dt:
            # 다음 달 1일
            if cur.month == 12:
                nxt = cur.replace(year=cur.year + 1, month=1, day=1)
            else:
                nxt = cur.replace(month=cur.month + 1, day=1)
            nxt = min(nxt, end_dt)
            periods.append((cur, nxt))
            cur = nxt
    elif granularity == "year":
        while cur < end_dt:
            nxt = min(cur.replace(year=cur.year + 1, month=1, day=1), end_dt)
            periods.append((cur, nxt))
            cur = nxt
    else:
        periods = [(start_dt, end_dt)]

    return periods


def _make_article_id(article: RawArticle) -> str:
    text = article.url or (article.title + article.published)
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]


def _event_to_dict(evt, raw_id: str) -> Dict:
    """StructuredEvent → DB 저장용 딕셔너리"""
    import hashlib
    # raw_id 기반 고정 event_id — 랜덤 UUID 사용 시 같은 기사가 매번 새 레코드 생성됨
    event_id = hashlib.md5(raw_id.encode()).hexdigest()[:12] if raw_id else hashlib.md5(
        getattr(evt, "title", "").encode()).hexdigest()[:12]

    categories = []
    if hasattr(evt, "categories"):
        categories = [c.value if hasattr(c, "value") else str(c)
                      for c in (evt.categories or [])]
    primary_cat = ""
    if hasattr(evt, "primary_category") and evt.primary_category:
        primary_cat = (evt.primary_category.value
                       if hasattr(evt.primary_category, "value")
                       else str(evt.primary_category))

    impact_dir = 0
    if hasattr(evt, "impact_direction"):
        v = evt.impact_direction
        impact_dir = v.value if hasattr(v, "value") else int(v)

    return {
        "event_id":          event_id,
        "raw_id":            raw_id,
        "published_at":      evt.timestamp.isoformat() if hasattr(evt, "timestamp") and evt.timestamp else datetime.datetime.now().isoformat(),
        "title":             getattr(evt, "title", ""),
        "categories":        categories,
        "primary_category":  primary_cat,
        "keywords":          getattr(evt, "keywords", []),
        "sentiment_score":   float(getattr(evt, "sentiment_score", 0.0)),
        "impact_direction":  impact_dir,
        "impact_strength":   float(getattr(evt, "impact_strength", 0.0)),
        "confidence":        float(getattr(evt, "confidence", 0.0)),
        "duration":          getattr(evt, "duration", "short"),
        "target_scope":      getattr(evt, "target_scope", "market"),
        "target_markets":    getattr(evt, "target_markets", []),
        "target_sectors":    getattr(evt, "target_sectors", []),
        "target_stocks":     getattr(evt, "target_stocks", []),
        "event_type":        getattr(evt, "event_type", ""),
        "computed_score":    float(getattr(evt, "score", 0.0)),
    }


# ── 싱글턴 ────────────────────────────────────────────────────────────────────

_collector_instance: Optional[NewsCollector] = None
_collector_lock = threading.Lock()


def get_collector(db: Optional[NewsDB] = None) -> NewsCollector:
    global _collector_instance
    with _collector_lock:
        if _collector_instance is None:
            _collector_instance = NewsCollector(db=db or get_news_db())
        return _collector_instance
