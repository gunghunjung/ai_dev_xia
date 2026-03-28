# data/news_service.py — 뉴스 통합 수집 서비스 레이어
"""
뉴스 소스 통합 오케스트레이션

지원 소스:
  - rss      : NewsFetcher (Yahoo/Reuters/MarketWatch/연합뉴스 등)
  - google   : GoogleNewsFetcher (키워드 기반 RSS, API 키 불필요)
  - finnhub  : FinnhubFetcher (종목별/시장 뉴스, API 키 필요)
  - collector: news_collector.py 60+ RSS 소스

기능:
  - collect(sources, symbols, days_back) — 소스별 수집 후 DB 저장
  - get_news(since, until, sources, symbols) — DB 쿼리
  - get_source_counts() — 소스별 기사 수
  - 증분 수집: last_collected 타임스탬프 추적 → 중복 요청 최소화
"""
from __future__ import annotations

import datetime
import logging
import os
import threading
from typing import Callable, Dict, List, Optional, Set

logger = logging.getLogger("quant.news.service")

# 지원 소스 이름
SOURCE_RSS       = "rss"
SOURCE_GOOGLE    = "google"
SOURCE_FINNHUB   = "finnhub"
SOURCE_COLLECTOR = "collector"

ALL_SOURCES = [SOURCE_RSS, SOURCE_GOOGLE, SOURCE_FINNHUB, SOURCE_COLLECTOR]
DEFAULT_SOURCES = [SOURCE_RSS, SOURCE_GOOGLE]   # API 키 불필요 소스만


class NewsService:
    """
    통합 뉴스 수집 서비스

    사용법:
        svc = NewsService(settings)
        stats = svc.collect(sources=["rss","google"], days_back=7)
        items = svc.get_news(since_dt=..., limit=200)
        counts = svc.get_source_counts(since_dt=...)
    """

    def __init__(self, settings=None, cache_dir: str = "cache",
                 db_path: str = None):
        self.settings  = settings
        self.cache_dir = cache_dir
        self._db_path  = db_path or (
            os.path.join("outputs", "news.db")
            if settings is None
            else getattr(settings.news, "db_path", "outputs/news.db")
        )
        self._last_collected: Dict[str, datetime.datetime] = {}
        self._lock = threading.Lock()

    # ── 수집 ──────────────────────────────────────────────────────────────────

    def collect(
        self,
        sources: List[str] = None,
        symbols: List[str] = None,
        days_back: int = 7,
        progress_cb: Optional[Callable[[str], None]] = None,
        force: bool = False,
    ) -> Dict:
        """
        지정된 소스에서 뉴스를 수집해 DB에 저장.

        Args:
            sources:     수집할 소스 목록 (기본: DEFAULT_SOURCES)
            symbols:     종목별 수집 대상 (Finnhub/Google 전용)
            days_back:   수집 기간 (일)
            progress_cb: 진행 메시지 콜백
            force:       증분 체크 무시 — 항상 전체 수집

        Returns:
            {source: {inserted, skipped, error}, "total_inserted": N}
        """
        active_sources = sources or DEFAULT_SOURCES
        stats: Dict[str, Dict] = {}
        total_inserted = 0

        def _progress(msg: str):
            if progress_cb:
                try:
                    progress_cb(msg)
                except Exception:
                    pass

        for src in active_sources:
            _progress(f"[{src}] 수집 중...")
            try:
                result = self._collect_source(
                    src, symbols=symbols, days_back=days_back, force=force
                )
                stats[src] = result
                total_inserted += result.get("inserted", 0)
                _progress(f"[{src}] 완료 — 신규 {result.get('inserted', 0)}건")
            except Exception as e:
                logger.warning(f"[NewsService] {src} 수집 오류: {e}")
                stats[src] = {"error": str(e), "inserted": 0, "skipped": 0}

        stats["total_inserted"] = total_inserted
        logger.info(f"[NewsService] 수집 완료: 총 {total_inserted}건 신규")
        return stats

    def collect_async(
        self,
        sources: List[str] = None,
        symbols: List[str] = None,
        days_back: int = 7,
        progress_cb: Optional[Callable[[str], None]] = None,
        done_cb: Optional[Callable[[Dict], None]] = None,
    ) -> threading.Thread:
        """백그라운드 수집 — 완료 시 done_cb(stats) 호출"""
        def _run():
            result = self.collect(sources=sources, symbols=symbols,
                                  days_back=days_back, progress_cb=progress_cb)
            if done_cb:
                try:
                    done_cb(result)
                except Exception:
                    pass

        t = threading.Thread(target=_run, daemon=True, name="news-service-collect")
        t.start()
        return t

    # ── 조회 ──────────────────────────────────────────────────────────────────

    def get_news(
        self,
        since_dt: Optional[datetime.datetime] = None,
        until_dt: Optional[datetime.datetime] = None,
        sources: Optional[List[str]] = None,
        symbols: Optional[List[str]] = None,
        limit: int = 500,
    ) -> List[Dict]:
        """
        DB에서 뉴스 조회.

        sources: DB source 컬럼 필터 (예: ["Google News: 주식시장", "Finnhub"])
        """
        from data.news_db import get_news_db
        db = get_news_db(self._db_path)

        # source 필터: 소스 이름 → DB source 필드 패턴 변환
        db_sources = None
        if sources:
            db_sources = _sources_to_db_patterns(sources)

        rows = db.get_raw(
            since_dt=since_dt,
            until_dt=until_dt,
            sources=db_sources,
            limit=limit,
        )
        return rows

    def get_source_counts(
        self,
        since_dt: Optional[datetime.datetime] = None,
    ) -> Dict[str, int]:
        """
        소스별 기사 수 반환.
        Returns: {"Google News: 주식시장": 42, "Finnhub": 18, ...}
        """
        from data.news_db import get_news_db
        db = get_news_db(self._db_path)
        since = since_dt or (datetime.datetime.now() - datetime.timedelta(days=30))
        rows = db.get_raw(since_dt=since, limit=5000)
        counts: Dict[str, int] = {}
        for r in rows:
            src = r.get("source", "unknown")
            counts[src] = counts.get(src, 0) + 1
        return counts

    def get_source_summary(
        self,
        since_dt: Optional[datetime.datetime] = None,
    ) -> Dict[str, int]:
        """
        논리 소스(rss/google/finnhub/collector)별 집계.
        Returns: {"rss": 120, "google": 45, "finnhub": 30, "collector": 200}
        """
        raw_counts = self.get_source_counts(since_dt=since_dt)
        summary: Dict[str, int] = {s: 0 for s in ALL_SOURCES}
        for src_name, cnt in raw_counts.items():
            logical = _infer_logical_source(src_name)
            summary[logical] = summary.get(logical, 0) + cnt
        return summary

    # ── 소스별 수집 구현 ──────────────────────────────────────────────────────

    def _collect_source(
        self,
        source: str,
        symbols: List[str] = None,
        days_back: int = 7,
        force: bool = False,
    ) -> Dict:
        if source == SOURCE_RSS:
            return self._collect_rss(days_back=days_back)
        elif source == SOURCE_GOOGLE:
            return self._collect_google(symbols=symbols, days_back=days_back)
        elif source == SOURCE_FINNHUB:
            return self._collect_finnhub(symbols=symbols, days_back=days_back)
        elif source == SOURCE_COLLECTOR:
            return self._collect_via_collector(days_back=days_back)
        else:
            logger.warning(f"[NewsService] 알 수 없는 소스: {source}")
            return {"inserted": 0, "skipped": 0}

    def _collect_rss(self, days_back: int = 7) -> Dict:
        """기본 RSS 수집기 (NewsFetcher)"""
        from data.news_fetcher import NewsFetcher
        from data.news_db import get_news_db
        fetcher = NewsFetcher(
            cache_dir=self.cache_dir,
            days_back=days_back,
            db_path=self._db_path,
        )
        items = fetcher.fetch_all(force_refresh=True)
        db = get_news_db(self._db_path)
        inserted, skipped = db.insert_raw(items)
        return {"inserted": inserted, "skipped": skipped}

    def _collect_google(self, symbols: List[str] = None,
                        days_back: int = 7) -> Dict:
        """Google News RSS 수집"""
        from data.google_news_fetcher import GoogleNewsFetcher
        from data.news_db import get_news_db

        fetcher = GoogleNewsFetcher(lang="ko")
        all_items = []

        # 시장 전반 뉴스
        all_items.extend(fetcher.fetch_market_news())

        # 영문 시장 뉴스
        en_fetcher = GoogleNewsFetcher(lang="en", region="US")
        all_items.extend(en_fetcher.fetch_market_news())

        # 종목별 뉴스
        if symbols:
            for sym in symbols[:5]:   # 최대 5개 종목 (레이트리밋 고려)
                try:
                    all_items.extend(fetcher.fetch_symbol_news(sym))
                except Exception as e:
                    logger.debug(f"[Google] {sym} 수집 실패: {e}")

        db = get_news_db(self._db_path)
        inserted, skipped = db.insert_raw(all_items)
        return {"inserted": inserted, "skipped": skipped}

    def _collect_finnhub(self, symbols: List[str] = None,
                         days_back: int = 7) -> Dict:
        """Finnhub API 수집"""
        api_key = ""
        if self.settings:
            api_key = getattr(self.settings.news, "finnhub_api_key", "")
        if not api_key:
            return {"inserted": 0, "skipped": 0, "note": "API 키 없음"}

        from data.finnhub_fetcher import FinnhubFetcher
        from data.news_db import get_news_db

        fetcher = FinnhubFetcher(api_key=api_key)
        all_items = []

        # 시장 전반
        all_items.extend(fetcher.fetch_all_market_news())

        # 종목별 (영문 티커만)
        if symbols:
            for sym in symbols:
                if not (sym.endswith(".KS") or sym.endswith(".KQ")):
                    try:
                        all_items.extend(fetcher.fetch_symbol_news(sym, days_back))
                    except Exception as e:
                        logger.debug(f"[Finnhub] {sym} 수집 실패: {e}")

        db = get_news_db(self._db_path)
        inserted, skipped = db.insert_raw(all_items)
        return {"inserted": inserted, "skipped": skipped}

    def _collect_via_collector(self, days_back: int = 7) -> Dict:
        """news_collector.py 60+ RSS 소스 수집"""
        try:
            from data.news_collector import get_collector
            collector = get_collector()
            stats = collector.collect_once(auto_process=False)
            return {
                "inserted": stats.get("inserted", 0),
                "skipped": stats.get("skipped", 0),
                "sources_ok": stats.get("sources_ok", 0),
            }
        except Exception as e:
            logger.warning(f"[NewsService] collector 수집 실패: {e}")
            return {"inserted": 0, "skipped": 0, "error": str(e)}


# ── 유틸 ──────────────────────────────────────────────────────────────────────

def _sources_to_db_patterns(logical_sources: List[str]) -> List[str]:
    """논리 소스 이름 → DB source 컬럼 패턴 목록"""
    patterns = []
    mapping = {
        SOURCE_RSS:       ["Yahoo Finance", "MarketWatch", "Reuters", "연합뉴스", "한국경제", "매일경제", "Investing"],
        SOURCE_GOOGLE:    ["Google News"],
        SOURCE_FINNHUB:   ["Finnhub"],
        SOURCE_COLLECTOR: ["Collector", "GDELT"],
    }
    for src in logical_sources:
        patterns.extend(mapping.get(src, [src]))
    return patterns


def _infer_logical_source(db_source: str) -> str:
    """DB source 문자열 → 논리 소스 이름"""
    sl = db_source.lower()
    if "google" in sl:
        return SOURCE_GOOGLE
    if "finnhub" in sl:
        return SOURCE_FINNHUB
    if any(x in sl for x in ("collector", "gdelt")):
        return SOURCE_COLLECTOR
    return SOURCE_RSS


# ── 싱글턴 ────────────────────────────────────────────────────────────────────

_service_instance: Optional[NewsService] = None
_service_lock = threading.Lock()


def get_news_service(settings=None, cache_dir: str = "cache",
                     db_path: str = None) -> NewsService:
    """싱글턴 NewsService 인스턴스 반환"""
    global _service_instance
    with _service_lock:
        if _service_instance is None:
            _service_instance = NewsService(
                settings=settings, cache_dir=cache_dir, db_path=db_path
            )
        elif settings is not None:
            # settings 갱신
            _service_instance.settings = settings
        return _service_instance
