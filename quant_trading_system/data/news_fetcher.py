# data/news_fetcher.py — 뉴스/이벤트 데이터 수집기
"""
RSS 피드 기반 실시간 뉴스 수집 & 이벤트 분류기

특징:
  - 무료 RSS 피드만 사용 (API 키 불필요)
  - JSON 캐시 (TTL: 30분) — 요청 과다 방지
  - 글로벌 + 한국 금융 뉴스 동시 수집
  - 오프라인 / 네트워크 오류 시 마지막 캐시 자동 사용
  - 노이즈 필터: 광고성/단순 가격 기사 제거

RSS 소스:
  글로벌: Yahoo Finance, MarketWatch, Reuters Business
  한국:   연합뉴스 경제, 네이버 금융 뉴스
"""
from __future__ import annotations

import datetime
import json
import logging
import os
import re
import threading
import time
from dataclasses import asdict, dataclass
from typing import Callable, Dict, List, Optional
from urllib.error import URLError
from urllib.request import Request, urlopen

logger = logging.getLogger("quant.news")

# ──────────────────────────────────────────────────────────────────────────────
# RSS 소스 목록
# ──────────────────────────────────────────────────────────────────────────────

_RSS_SOURCES: List[Dict[str, str]] = [
    # 글로벌
    {
        "name":   "Yahoo Finance",
        "url":    "https://finance.yahoo.com/news/rssindex",
        "lang":   "en",
        "weight": 1.0,
    },
    {
        "name":   "MarketWatch",
        "url":    "https://feeds.marketwatch.com/marketwatch/realtimeheadlines/",
        "lang":   "en",
        "weight": 0.9,
    },
    {
        "name":   "Reuters Business",
        "url":    "https://feeds.reuters.com/reuters/businessNews",
        "lang":   "en",
        "weight": 1.0,
    },
    {
        "name":   "Investing.com 글로벌",
        "url":    "https://www.investing.com/rss/news.rss",
        "lang":   "en",
        "weight": 0.8,
    },
    # 한국
    {
        "name":   "연합뉴스 경제",
        "url":    "https://www.yonhapnewstv.co.kr/category/news/economy/feed/",
        "lang":   "ko",
        "weight": 1.0,
    },
    {
        "name":   "한국경제",
        "url":    "https://www.hankyung.com/feed/economy",
        "lang":   "ko",
        "weight": 0.9,
    },
    {
        "name":   "매일경제",
        "url":    "https://www.mk.co.kr/rss/30000001/",
        "lang":   "ko",
        "weight": 0.9,
    },
]

# 캐시 TTL
_CACHE_TTL_MINUTES = 1          # 1분마다 갱신
_MAX_ITEMS_PER_SOURCE = 30
_MEMORY_MAX_ITEMS = 3000        # _accumulated 메모리 상한 (날짜 범위 지원을 위해 크게)
_TOTAL_MAX_ITEMS = 500          # _cached_items 표시 상한 (UI 성능)
_DB_LOAD_PER_DAY = 150          # DB에서 일당 로드 기사 수 추정치

# 노이즈 필터: 이 단어만 포함된 제목은 제거
_NOISE_PATTERNS = [
    r"^\d+개.*공모$",               # 공모주 광고
    r"무료.*세미나",
    r"광고|스폰서|협찬",
    r"오늘의.*운세",
    r"추천종목",
]


# ──────────────────────────────────────────────────────────────────────────────
# 데이터 클래스
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class NewsItem:
    """단일 뉴스 아이템"""
    title:     str
    body:      str
    source:    str
    url:       str
    published: str    # ISO format string
    lang:      str
    weight:    float = 1.0

    @property
    def published_dt(self) -> datetime.datetime:
        s = (self.published or "").strip()
        if not s:
            return datetime.datetime.now()
        # 1순위: fromisoformat — DB 저장 형식 대부분 커버
        #   "2026-02-20T15:30:00", "2026-02-20T15:30:00.123456",
        #   "2026-02-20T15:30:00+00:00" 등
        try:
            return datetime.datetime.fromisoformat(s)
        except ValueError:
            pass
        # 2순위: RSS RFC-2822 (타임존 포함/미포함)
        for fmt in (
            "%a, %d %b %Y %H:%M:%S %z",   # "Mon, 20 Feb 2026 12:00:00 +0000"
            "%a, %d %b %Y %H:%M:%S",       # "Mon, 20 Feb 2026 12:00:00"
            "%d %b %Y %H:%M:%S %z",
            "%d %b %Y %H:%M:%S",
        ):
            try:
                return datetime.datetime.strptime(s, fmt)
            except ValueError:
                continue
        # 3순위: 날짜만
        try:
            return datetime.datetime.strptime(s[:10], "%Y-%m-%d")
        except ValueError:
            pass
        return datetime.datetime.now()


# ──────────────────────────────────────────────────────────────────────────────
# RSS 파서
# ──────────────────────────────────────────────────────────────────────────────

def _parse_rss(xml_text: str, source_name: str,
               lang: str, weight: float) -> List[NewsItem]:
    """단순 정규표현식 기반 RSS 파서 (외부 라이브러리 불필요)"""
    items: List[NewsItem] = []

    # <item>...</item> 블록 추출
    item_blocks = re.findall(r"<item>(.*?)</item>", xml_text, re.DOTALL)
    for block in item_blocks[:_MAX_ITEMS_PER_SOURCE]:
        def _tag(t: str) -> str:
            m = re.search(rf"<{t}[^>]*>(.*?)</{t}>", block, re.DOTALL)
            if m:
                txt = m.group(1).strip()
                # CDATA 제거
                txt = re.sub(r"<!\[CDATA\[(.*?)\]\]>", r"\1", txt, flags=re.DOTALL)
                # HTML 태그 제거
                txt = re.sub(r"<[^>]+>", " ", txt)
                # &amp; 등 HTML 엔티티 변환
                txt = txt.replace("&amp;", "&").replace("&lt;", "<") \
                         .replace("&gt;", ">").replace("&quot;", '"') \
                         .replace("&#39;", "'").replace("&nbsp;", " ")
                return txt.strip()
            return ""

        title = _tag("title")
        if not title:
            continue

        # 노이즈 필터
        if any(re.search(p, title, re.IGNORECASE) for p in _NOISE_PATTERNS):
            continue

        body  = _tag("description") or _tag("summary") or ""
        url   = _tag("link") or _tag("guid") or ""
        pub   = _tag("pubDate") or _tag("published") or _tag("dc:date") or \
                datetime.datetime.now().isoformat()

        items.append(NewsItem(
            title=title, body=body[:500], source=source_name,
            url=url, published=pub, lang=lang, weight=weight,
        ))

    return items


# ──────────────────────────────────────────────────────────────────────────────
# 메인 뉴스 수집기
# ──────────────────────────────────────────────────────────────────────────────

class NewsFetcher:
    """
    멀티소스 RSS 뉴스 수집기

    사용법:
        fetcher = NewsFetcher(cache_dir="cache")
        items   = fetcher.fetch_all()            # 전체 소스 수집
        events  = fetcher.fetch_as_events()      # MacroEvent 목록으로 변환
        fetcher.start_auto_refresh(callback=fn)  # 자동 갱신 (30분마다)
    """

    def __init__(self, cache_dir: str = "cache", ttl_minutes: int = _CACHE_TTL_MINUTES,
                 days_back: int = 7, db_path: str = None):
        self.cache_dir    = cache_dir
        self.ttl_minutes  = ttl_minutes
        self.days_back    = days_back          # 조회 기간 (일)
        self._cache_path  = os.path.join(cache_dir, "news_cache.json")
        self._db_path     = db_path            # None → news_db 기본 경로 사용
        self._lock        = threading.Lock()
        # URL 기준 누적 딕셔너리 (중복 방지) — pruning 금지, 전체 보관
        self._accumulated: Dict[str, NewsItem] = {}
        self._cached_items: List[NewsItem] = []
        self._last_fetch: Optional[datetime.datetime] = None
        self._refresh_thread: Optional[threading.Thread] = None
        self._stop_event  = threading.Event()
        os.makedirs(cache_dir, exist_ok=True)

        # 시작 시 캐시 로드 (JSON 단기 캐시 → SQLite DB 장기 캐시 순)
        self._load_disk_cache()
        # DB 캐시는 비동기 로드 — UI 표시는 pipeline.get_structured_events()가 담당
        threading.Thread(target=self._load_db_cache, daemon=True,
                         name="fetcher-db-cache").start()

    # ──────────────────────────────────────────────────────────────────────────
    # 공개 API
    # ──────────────────────────────────────────────────────────────────────────

    def set_days_back(self, days: int) -> None:
        """뉴스 조회 기간 변경 (즉시 적용 — DB 재쿼리 포함)"""
        old_days = self.days_back
        self.days_back = max(1, int(days))
        logger.debug(f"[날짜변경] days_back: {old_days}→{self.days_back}일")
        # 항상 DB 재쿼리 (기간 증가/감소 무관 — DB에 해당 기간 데이터가 있어야 반영됨)
        self._load_db_cache()
        with self._lock:
            self._cached_items = self._filter_by_date(
                list(self._accumulated.values()))
        logger.debug(f"[날짜변경] 필터 후 {len(self._cached_items)}건 표시 "
                     f"(accumulated {len(self._accumulated)}건)")

    def fetch_all(self, force_refresh: bool = False) -> List[NewsItem]:
        """
        전체 소스에서 뉴스 수집 — 누적 방식 (URL 기반 중복 제거).
        TTL 이내면 캐시 반환, 만료됐으면 갱신.
        """
        if not force_refresh and self._is_cache_valid():
            return self._cached_items

        new_items: List[NewsItem] = []
        for src in _RSS_SOURCES:
            try:
                items = self._fetch_source(src)
                new_items.extend(items)
            except Exception as e:
                logger.debug(f"[{src['name']}] 뉴스 수집 실패: {e}")

        if new_items:
            # 누적 딕셔너리에 병합 (URL 기반 중복 제거) — pruning 금지
            with self._lock:
                added = 0
                for item in new_items:
                    key = item.url or item.title   # URL 없으면 제목으로 키
                    if key and key not in self._accumulated:
                        self._accumulated[key] = item
                        added += 1

                # 전체 정렬 후 메모리 상한 적용 (_MEMORY_MAX_ITEMS — 날짜 범위 지원)
                merged_all = sorted(self._accumulated.values(),
                                    key=lambda x: self._to_naive(x.published_dt),
                                    reverse=True)
                # 오버플로 시 오래된 항목 제거 (메모리 상한: _MEMORY_MAX_ITEMS)
                if len(merged_all) > _MEMORY_MAX_ITEMS:
                    merged_all = merged_all[:_MEMORY_MAX_ITEMS]
                    self._accumulated = {
                        (item.url or item.title): item for item in merged_all
                    }
                # _cached_items는 days_back 기준 필터 + 표시 상한(_TOTAL_MAX_ITEMS)
                self._cached_items = self._filter_by_date(merged_all)
                self._last_fetch   = datetime.datetime.now()

            # SQLite DB에 신규 기사 영구 저장
            self._save_to_db(new_items)
            self._save_disk_cache(self._cached_items)
            logger.info(
                f"뉴스 수집 완료: 신규 {added}건 추가 / "
                f"메모리 {len(self._accumulated)}건 / "
                f"표시({self.days_back}일) {len(self._cached_items)}건"
            )
        elif self._cached_items:
            logger.warning("뉴스 수집 실패 — 이전 누적 캐시 사용")

        return self._cached_items

    @staticmethod
    def _to_naive(dt: datetime.datetime) -> datetime.datetime:
        """timezone-aware datetime을 naive로 변환"""
        if dt.tzinfo is not None:
            return dt.replace(tzinfo=None)
        return dt

    def _filter_by_date(self, items: List[NewsItem]) -> List[NewsItem]:
        """days_back 기준으로 항목 필터링"""
        cutoff = datetime.datetime.now() - datetime.timedelta(days=self.days_back)
        cutoff_naive = cutoff.replace(tzinfo=None)
        filtered = [i for i in items
                    if self._to_naive(i.published_dt) >= cutoff_naive]
        filtered.sort(key=lambda x: self._to_naive(x.published_dt), reverse=True)
        return filtered[:_TOTAL_MAX_ITEMS]

    def fetch_as_events(self, force_refresh: bool = False):
        """
        NewsItem 목록을 MacroEvent 목록으로 변환.
        Returns: List[MacroEvent]
        """
        from features.macro_features import EventClassifier
        items = self.fetch_all(force_refresh=force_refresh)
        classifier = EventClassifier()
        events = []
        for item in items:
            evt = classifier.classify(item.title, item.body)
            evt.source    = item.source
            evt.url       = item.url
            evt.timestamp = item.published_dt
            events.append(evt)
        return events

    def fetch_as_structured_events(self, force_refresh: bool = False):
        """
        NewsItem 목록을 StructuredEvent 목록으로 변환 (외부환경 분석 파이프라인용).
        NLPAnalyzer 감성 분석 + NewsEventCategorizer 12카테고리 분류.
        Returns: List[StructuredEvent]
        """
        try:
            from features.external_env.categorizer import NewsEventCategorizer
            from features.external_env.nlp_analyzer import get_analyzer
        except ImportError as e:
            logger.warning(f"external_env 모듈 없음, MacroEvent로 대체: {e}")
            return []

        items = self.fetch_all(force_refresh=force_refresh)
        categorizer = NewsEventCategorizer()
        nlp = get_analyzer()
        structured: list = []
        for item in items:
            try:
                evt = categorizer.classify(
                    title=item.title,
                    summary=item.body,
                    url=item.url,
                    ts=item.published_dt,
                )
                # NLP 분석 결과 덮어쓰기 (categorizer의 기본값보다 NLP가 정확)
                nlp_result = nlp.analyze(item.title, item.body)
                evt.sentiment_score = nlp_result["sentiment_score"]
                evt.importance      = nlp_result["importance"]
                evt.confidence      = max(evt.confidence,
                                          nlp_result["confidence"])
                if nlp_result["keywords"]:
                    evt.keywords = nlp_result["keywords"][:6]
                # compute_score 재실행 (nlp 점수 반영)
                evt.compute_score()
                structured.append(evt)
            except Exception as e:
                logger.debug(f"StructuredEvent 변환 실패: {e}")

        # Layer 2 저장은 NewsPipeline.classify_pending()이 담당 (raw_id 없이 저장하면 중복/오류)
        return structured

    def start_auto_refresh(
        self,
        callback: Optional[Callable] = None,
        interval_minutes: int = None,
    ) -> None:
        """
        백그라운드 RSS 수집 자동 실행 (수집 전용 — AI 분류는 NewsPipeline 담당).
        callback: 수집 완료 시 호출. 인자 없음 (UI 갱신 트리거용).
        """
        if self._refresh_thread and self._refresh_thread.is_alive():
            return

        interval = (interval_minutes or self.ttl_minutes) * 60
        self._stop_event.clear()

        def _loop():
            while not self._stop_event.is_set():
                try:
                    self.fetch_all(force_refresh=True)   # 수집 + DB 저장만
                    if callback:
                        callback()                       # 인자 없이 시그널만
                except Exception as e:
                    logger.debug(f"자동 수집 오류: {e}")
                self._stop_event.wait(interval)

        self._refresh_thread = threading.Thread(target=_loop, daemon=True,
                                                 name="news-auto-collect")
        self._refresh_thread.start()
        logger.info(f"뉴스 자동 수집 시작 (간격: {interval_minutes or self.ttl_minutes}분)")

    def stop_auto_refresh(self) -> None:
        self._stop_event.set()

    # ──────────────────────────────────────────────────────────────────────────
    # 내부 메서드
    # ──────────────────────────────────────────────────────────────────────────

    def _fetch_source(self, src: Dict[str, str]) -> List[NewsItem]:
        """단일 RSS 소스 요청"""
        url  = src["url"]
        req  = Request(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 Chrome/120.0"
                ),
                "Accept": "application/rss+xml, application/xml, text/xml",
            },
        )
        with urlopen(req, timeout=8) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        return _parse_rss(raw, src["name"], src["lang"], src["weight"])

    def _is_cache_valid(self) -> bool:
        if not self._cached_items or self._last_fetch is None:
            return False
        elapsed = (datetime.datetime.now() - self._last_fetch).total_seconds() / 60
        return elapsed < self.ttl_minutes

    def _save_disk_cache(self, items: List[NewsItem]) -> None:
        try:
            data = {
                "fetched_at": datetime.datetime.now().isoformat(),
                "items": [asdict(i) for i in items],
            }
            with open(self._cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.debug(f"캐시 저장 실패: {e}")

    def _load_disk_cache(self) -> None:
        if not os.path.exists(self._cache_path):
            return
        try:
            with open(self._cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            items = [NewsItem(**i) for i in data.get("items", [])]
            fetched_at = datetime.datetime.fromisoformat(data["fetched_at"])
            with self._lock:
                # 누적 딕셔너리에 복원
                for item in items:
                    key = item.url or item.title
                    if key:
                        self._accumulated[key] = item
                self._cached_items = self._filter_by_date(items)
                self._last_fetch   = fetched_at
            logger.debug(f"JSON캐시 로드: {len(items)}건 → {self.days_back}일 필터 후 {len(self._cached_items)}건")
        except Exception as e:
            logger.debug(f"JSON캐시 로드 실패: {e}")

    def _load_db_cache(self) -> None:
        """SQLite DB에서 히스토리 로드 — 날짜 범위 전체를 커버하는 분할 쿼리"""
        try:
            from data.news_db import get_news_db
            db = get_news_db(self._db_path)
            now = datetime.datetime.now()
            load_days = max(self.days_back, 1)   # 상한 없음 — 백필 전체 범위 지원
            all_rows: list = []

            if load_days <= 7:
                # 7일 이하: 단순 최신순 쿼리
                since = now - datetime.timedelta(days=load_days)
                all_rows = db.get_raw(since_dt=since, limit=1000)
            else:
                # 7일 초과: 구간을 7일씩 나눠 쿼리 — DESC LIMIT 으로 오래된 기사가
                # 최신 기사에 밀려 잘리는 문제 방지
                segment_days = 7
                d = load_days
                while d > 0:
                    seg_until = now - datetime.timedelta(days=d - segment_days)
                    seg_since = now - datetime.timedelta(days=d)
                    if seg_until > now:
                        seg_until = now
                    rows = db.get_raw(
                        since_dt=seg_since,
                        until_dt=seg_until,
                        limit=500,
                    )
                    all_rows.extend(rows)
                    d -= segment_days

            if not all_rows:
                logger.debug(f"DB캐시: {load_days}일 범위에 저장된 기사 없음")
                return

            added = 0
            with self._lock:
                for r in all_rows:
                    item = NewsItem(
                        title=r["title"],
                        body=r.get("summary", ""),
                        source=r["source"],
                        url=r.get("url", ""),
                        published=r["published_at"],
                        lang=r.get("lang", "ko"),
                        weight=float(r.get("weight", 1.0)),
                    )
                    key = item.url or item.title
                    if key and key not in self._accumulated:
                        self._accumulated[key] = item
                        added += 1
                self._cached_items = self._filter_by_date(
                    list(self._accumulated.values()))
            logger.debug(
                f"DB캐시 로드: {added}건 신규 / "
                f"누적 {len(self._accumulated)}건 / "
                f"표시({self.days_back}일) {len(self._cached_items)}건"
            )
        except Exception as e:
            logger.debug(f"DB캐시 로드 실패: {e}")

    def _save_to_db(self, items: List[NewsItem]) -> None:
        """신규 기사를 SQLite DB에 영구 저장 (Layer 1: news_raw)"""
        if not items:
            return
        try:
            from data.news_db import get_news_db
            db = get_news_db(self._db_path)
            inserted, skipped = db.insert_raw(items)
            logger.debug(f"DB저장: {inserted}건 신규, {skipped}건 중복/스킵")
        except Exception as e:
            logger.debug(f"DB저장 실패: {e}")

    def _save_structured_to_db(self, events: list) -> int:
        """StructuredEvent 목록을 news_events DB에 저장 (Layer 2: AI 분류 결과)"""
        if not events:
            return 0
        try:
            import datetime as _dt
            from data.news_db import get_news_db
            db = get_news_db(self._db_path)
            dicts = []
            for evt in events:
                try:
                    ts = getattr(evt, "timestamp", None)
                    if ts is None:
                        ts = _dt.datetime.now()
                    if hasattr(ts, "tzinfo") and ts.tzinfo is not None:
                        ts = ts.replace(tzinfo=None)
                    cats = [c.value for c in (getattr(evt, "categories", []) or [])]
                    pcat = getattr(evt, "primary_cat", None)
                    impact_dir = getattr(evt, "impact_direction", None)
                    dur = getattr(evt, "duration", None)
                    dicts.append({
                        "event_id":         getattr(evt, "event_id", ""),
                        "raw_id":           "",
                        "published_at":     ts.isoformat(),
                        "title":            getattr(evt, "title", ""),
                        "categories":       cats,
                        "primary_category": pcat.value if pcat else "",
                        "keywords":         getattr(evt, "keywords", []),
                        "sentiment_score":  float(getattr(evt, "sentiment_score", 0.0)),
                        "impact_direction": getattr(impact_dir, "value", 0),
                        "impact_strength":  float(getattr(evt, "impact_strength", 0.0)),
                        "confidence":       float(getattr(evt, "confidence", 0.0)),
                        "duration":         dur.value if dur and hasattr(dur, "value") else "short",
                        "target_scope":     getattr(evt, "target_market", "market"),
                        "target_sectors":   getattr(evt, "target_sectors", []),
                        "target_stocks":    getattr(evt, "target_tickers", []),
                        "event_type":       getattr(evt, "event_type", ""),
                        "computed_score":   float(getattr(evt, "external_score", 0.0)),
                    })
                except Exception:
                    pass
            count = db.insert_events_bulk(dicts)
            logger.debug(f"StructuredEvent DB저장(Layer 2): {count}건 / 시도 {len(dicts)}건")
            return count
        except Exception as e:
            logger.debug(f"StructuredEvent DB저장 실패: {e}")
            return 0


# ──────────────────────────────────────────────────────────────────────────────
# 목업 데이터 (오프라인 / 개발용)
# ──────────────────────────────────────────────────────────────────────────────

_MOCK_EVENTS_KO = [
    ("연준, 금리 동결 결정… '인플레 여전히 우려'",
     "연방준비제도(Fed)가 이번 회의에서 기준금리를 동결했다. 파월 의장은 인플레가 여전히 목표치를 상회한다고 밝혔다.",
     "Yahoo Finance", "FOMC/RATE"),
    ("中 제조업 PMI 예상치 상회… 글로벌 경기 회복 기대감",
     "중국의 3월 제조업 PMI가 51.3을 기록, 시장 예상을 웃돌며 글로벌 공급망 안정 기대감 확산.",
     "Reuters Business", "SUPPLY/COMMODITY"),
    ("국제유가 급등… 중동 긴장 고조",
     "WTI 원유 선물이 배럴당 5% 급등. 이스라엘-하마스 분쟁 격화로 공급 차질 우려.",
     "MarketWatch", "WAR/COMMODITY"),
    ("삼성전자, 1분기 영업이익 시장 예상 대폭 상회",
     "삼성전자가 1분기 영업이익 6조원을 발표, 컨센서스 4.5조원을 크게 웃돌며 반도체 업황 회복 신호.",
     "한국경제", "EARNINGS/SUPPLY"),
    ("원달러 환율 1390원 돌파… 달러 강세 지속",
     "달러 인덱스 강세에 원달러 환율이 1390원을 돌파하며 수출주 수혜 기대.",
     "연합뉴스 경제", "FX"),
    ("한국은행, 기준금리 3.50% 동결 결정",
     "한국은행 금통위가 기준금리를 3.50%로 동결. 국내 물가 안정세 확인 후 인하 검토 예정.",
     "매일경제", "RATE/POLICY"),
    ("CPI 예상 상회… 금리인하 기대 후퇴",
     "미국 3월 CPI가 전년 대비 3.5% 상승, 예상치 3.4%를 상회하며 금리인하 기대감 후퇴.",
     "Yahoo Finance", "CPI/RATE"),
    ("반도체 공급망 재편… TSMC 일본 공장 가동",
     "TSMC가 일본 구마모토 1공장을 공식 가동, 글로벌 반도체 공급망 다변화 가속.",
     "Reuters Business", "SUPPLY/TECH"),
]


def get_mock_events():
    """네트워크 없이 테스트용 목업 이벤트 반환 (MacroEvent)"""
    from features.macro_features import EventClassifier
    classifier = EventClassifier()
    events = []
    base = datetime.datetime.now()
    for i, (title, body, src, _) in enumerate(_MOCK_EVENTS_KO):
        evt = classifier.classify(title, body)
        evt.source    = src
        evt.timestamp = base - datetime.timedelta(hours=i * 3.5)
        events.append(evt)
    return events


def get_mock_structured_events():
    """네트워크 없이 테스트용 목업 StructuredEvent 반환"""
    try:
        from features.external_env.categorizer import NewsEventCategorizer
        from features.external_env.nlp_analyzer import get_analyzer
    except ImportError:
        return []
    categorizer = NewsEventCategorizer()
    nlp = get_analyzer()
    events = []
    base = datetime.datetime.now()
    for i, (title, body, src, _) in enumerate(_MOCK_EVENTS_KO):
        try:
            ts = base - datetime.timedelta(hours=i * 3.5)
            evt = categorizer.classify(title, body, ts=ts)
            nlp_result = nlp.analyze(title, body)
            evt.sentiment_score = nlp_result["sentiment_score"]
            evt.importance      = nlp_result["importance"]
            evt.confidence      = max(evt.confidence, nlp_result["confidence"])
            if nlp_result["keywords"]:
                evt.keywords = nlp_result["keywords"][:6]
            evt.compute_score()
            events.append(evt)
        except Exception:
            pass
    return events
