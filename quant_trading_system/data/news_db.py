# data/news_db.py — 뉴스 3계층 저장소 (Raw / StructuredEvent / FeatureCache)
"""
SQLite 기반 3단계 뉴스 데이터 저장 시스템

계층 구조:
  Layer 1 — news_raw       : 수집된 원본 기사 (제목/요약/메타)
  Layer 2 — news_events    : AI 분류된 구조화 이벤트 (카테고리/감성/영향도)
  Layer 3 — news_feat_cache: 모델 입력용 특징 벡터 캐시 (심볼×날짜×윈도우)

설계 원칙:
  - 전체 본문 무조건 저장 금지 → 제목 + 요약(500자) + 구조화 특징 저장
  - 중요 기사(importance >= 0.7)만 본문 추가 저장
  - 중복 기사: URL 해시 기반 1차 중복 제거, 임베딩 클러스터 기반 2차 정리
  - 오래된 raw 데이터: 30일 이상 자동 압축(ZLIB)
  - 인덱싱: (published_at, source, primary_category) 복합 인덱스
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import threading
import zlib
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("quant.news_db")


# ── 날짜 정규화 ────────────────────────────────────────────────────────────────

def _normalize_pub_date(s: str) -> str:
    """날짜 문자열을 ISO8601 naive 형식으로 정규화.
    RFC-2822 (RSS pubDate), ISO+TZ 등 → 'YYYY-MM-DDTHH:MM:SS'
    파싱 실패 시 원본 문자열 반환.
    """
    if not s:
        return s
    # 이미 ISO 형식이면 빠르게 처리
    try:
        dt = datetime.fromisoformat(s)
        return dt.replace(tzinfo=None).isoformat()
    except ValueError:
        pass
    # RFC-2822 포맷 (RSS pubDate)
    for fmt in (
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y %H:%M:%S",
        "%d %b %Y %H:%M:%S %z",
        "%d %b %Y %H:%M:%S",
    ):
        try:
            dt = datetime.strptime(s.strip(), fmt)
            return dt.replace(tzinfo=None).isoformat()
        except ValueError:
            continue
    # 날짜만
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d").isoformat()
    except ValueError:
        pass
    return s  # 파싱 불가 → 원본 유지


# ── 상수 ──────────────────────────────────────────────────────────────────────
_SCHEMA_VERSION    = 3
_COMPRESS_DAYS     = 30     # 이 일수 이상 된 raw body 압축
_ARCHIVE_DAYS      = 365    # 이 일수 이상 된 raw 기사 아카이브 가능
_MAX_BODY_CHARS    = 500    # 일반 기사 본문 최대 저장 길이
_IMPORTANT_THRESH  = 0.7    # 중요 기사 본문 전체 저장 임계값


# ── DDL ───────────────────────────────────────────────────────────────────────

_DDL = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous  = NORMAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY
);

-- Layer 1: 원본 기사
CREATE TABLE IF NOT EXISTS news_raw (
    id              TEXT PRIMARY KEY,          -- SHA256(url or title+published)
    title           TEXT NOT NULL,
    summary         TEXT DEFAULT '',           -- 500자 이내 요약
    body_full       TEXT DEFAULT '',           -- 중요 기사만 전체 본문 (ZLIB 가능)
    body_compressed INTEGER DEFAULT 0,         -- 1=ZLIB 압축됨
    source          TEXT NOT NULL,
    url             TEXT DEFAULT '',
    lang            TEXT DEFAULT 'ko',
    published_at    TEXT NOT NULL,             -- ISO8601
    fetched_at      TEXT NOT NULL,
    weight          REAL DEFAULT 1.0,
    importance      REAL DEFAULT 0.0,          -- 0~1, AI 분류 후 업데이트
    cluster_id      TEXT DEFAULT '',           -- 중복 클러스터 ID
    is_representative INTEGER DEFAULT 1,       -- 클러스터 대표 기사 여부
    tags            TEXT DEFAULT '[]'          -- JSON 배열 (종목/섹터/이벤트)
);

CREATE INDEX IF NOT EXISTS idx_raw_published
    ON news_raw(published_at DESC);
CREATE INDEX IF NOT EXISTS idx_raw_source
    ON news_raw(source, published_at DESC);
CREATE INDEX IF NOT EXISTS idx_raw_cluster
    ON news_raw(cluster_id);

-- Layer 2: 구조화 이벤트
CREATE TABLE IF NOT EXISTS news_events (
    event_id            TEXT PRIMARY KEY,      -- UUID or hash
    raw_id              TEXT REFERENCES news_raw(id),
    published_at        TEXT NOT NULL,
    title               TEXT NOT NULL,
    categories          TEXT DEFAULT '[]',     -- JSON [EventCategory value strings]
    primary_category    TEXT DEFAULT '',
    keywords            TEXT DEFAULT '[]',     -- JSON
    sentiment_score     REAL DEFAULT 0.0,      -- -1 ~ +1
    impact_direction    INTEGER DEFAULT 0,     -- -1/0/+1
    impact_strength     REAL DEFAULT 0.0,      -- 0 ~ 1
    confidence          REAL DEFAULT 0.0,
    duration            TEXT DEFAULT 'short',  -- short/mid/long
    target_scope        TEXT DEFAULT 'market', -- market/sector/stock
    target_markets      TEXT DEFAULT '[]',     -- JSON
    target_sectors      TEXT DEFAULT '[]',     -- JSON
    target_stocks       TEXT DEFAULT '[]',     -- JSON
    event_type          TEXT DEFAULT '',
    cluster_id          TEXT DEFAULT '',
    is_representative   INTEGER DEFAULT 1,
    repeat_count        INTEGER DEFAULT 1,     -- 같은 이벤트 반복 보도 수
    computed_score      REAL DEFAULT 0.0,      -- impact_direction × impact_strength × confidence
    source_diversity    REAL DEFAULT 0.0       -- 클러스터 내 언론사 다양성 (0~1)
);

CREATE INDEX IF NOT EXISTS idx_evt_published
    ON news_events(published_at DESC);
CREATE INDEX IF NOT EXISTS idx_evt_category
    ON news_events(primary_category, published_at DESC);
CREATE INDEX IF NOT EXISTS idx_evt_stocks
    ON news_events(target_stocks);
CREATE INDEX IF NOT EXISTS idx_evt_score
    ON news_events(computed_score, published_at DESC);
CREATE INDEX IF NOT EXISTS idx_evt_raw_id
    ON news_events(raw_id);

-- Layer 3: 모델 입력용 특징 벡터 캐시
CREATE TABLE IF NOT EXISTS news_feat_cache (
    cache_key       TEXT PRIMARY KEY,          -- "symbol|date|window"
    symbol          TEXT NOT NULL,
    date            TEXT NOT NULL,             -- YYYY-MM-DD
    window          TEXT NOT NULL,             -- 1h/4h/1d/3d/5d/20d
    features        TEXT NOT NULL,             -- JSON float array (40D)
    computed_at     TEXT NOT NULL,
    event_count     INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_feat_symbol_date
    ON news_feat_cache(symbol, date DESC);

-- 수집 통계
CREATE TABLE IF NOT EXISTS collection_stats (
    date            TEXT PRIMARY KEY,          -- YYYY-MM-DD
    total_fetched   INTEGER DEFAULT 0,
    total_stored    INTEGER DEFAULT 0,
    duplicates      INTEGER DEFAULT 0,
    filtered        INTEGER DEFAULT 0,
    sources         TEXT DEFAULT '{}'          -- JSON {source_name: count}
);

-- 백필 이력 (이미 수집된 기간 추적)
CREATE TABLE IF NOT EXISTS backfill_log (
    id              TEXT PRIMARY KEY,          -- "{start_date}_{end_date}_{category}"
    start_date      TEXT NOT NULL,             -- YYYY-MM-DD
    end_date        TEXT NOT NULL,             -- YYYY-MM-DD
    category        TEXT NOT NULL DEFAULT '*', -- 카테고리 or '*' (전체)
    inserted        INTEGER DEFAULT 0,
    skipped         INTEGER DEFAULT 0,
    collected_at    TEXT NOT NULL,             -- ISO8601
    source          TEXT DEFAULT 'gdelt'
);

CREATE INDEX IF NOT EXISTS idx_backfill_range
    ON backfill_log(start_date, end_date);
"""


# ── DB 관리자 ─────────────────────────────────────────────────────────────────

class NewsDB:
    """
    뉴스 SQLite 데이터베이스 관리자

    사용법:
        db = NewsDB("outputs/news.db")
        db.insert_raw(items)
        events = db.get_events(since_dt, until_dt)
        feats  = db.get_cached_features("005930.KS", "2025-01-15", "1d")
    """

    def __init__(self, db_path: str = "outputs/news.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
        self._local = threading.local()  # 스레드별 커넥션
        self._init_db()

    # ── 커넥션 관리 ──────────────────────────────────────────────────────────

    def _conn(self) -> sqlite3.Connection:
        """스레드 로컬 커넥션 반환 (스레드 안전)"""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return self._local.conn

    @contextmanager
    def _tx(self):
        """트랜잭션 컨텍스트 매니저"""
        conn = self._conn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def _init_db(self) -> None:
        """스키마 초기화 및 버전 확인"""
        with self._tx() as conn:
            conn.executescript(_DDL)
            row = conn.execute("SELECT version FROM schema_version").fetchone()
            if row is None:
                conn.execute("INSERT INTO schema_version VALUES (?)", (_SCHEMA_VERSION,))
            elif row["version"] < _SCHEMA_VERSION:
                # 마이그레이션 (현재는 재생성 방식)
                conn.execute("UPDATE schema_version SET version=?", (_SCHEMA_VERSION,))
        logger.debug(f"NewsDB 초기화: {self.db_path}")

    # ── Layer 1: Raw 기사 ──────────────────────────────────────────────────────

    def insert_raw(self, items: list, update_stats: bool = True) -> Tuple[int, int]:
        """
        NewsItem 또는 딕셔너리 목록을 news_raw에 삽입.
        중복(같은 ID)은 무시(INSERT OR IGNORE).

        Returns:
            (inserted_count, skipped_count)
        """
        inserted = 0
        skipped  = 0
        today    = datetime.now().strftime("%Y-%m-%d")
        now_iso  = datetime.now().isoformat()

        # 기존 ID 셋을 미리 로드해 메모리에서 중복 체크 (SELECT-per-row 제거)
        existing_ids: set = set()
        with self._tx() as conn:
            for item in items:
                if hasattr(item, "__dict__"):
                    title     = getattr(item, "title", "")
                    summary   = getattr(item, "body",  "")[:_MAX_BODY_CHARS]
                    source    = getattr(item, "source", "")
                    url       = getattr(item, "url", "")
                    lang      = getattr(item, "lang", "ko")
                    weight    = getattr(item, "weight", 1.0)
                    published = getattr(item, "published", now_iso)
                else:
                    title     = item.get("title", "")
                    summary   = item.get("body", item.get("summary", ""))[:_MAX_BODY_CHARS]
                    source    = item.get("source", "")
                    url       = item.get("url", "")
                    lang      = item.get("lang", "ko")
                    weight    = item.get("weight", 1.0)
                    published = item.get("published", now_iso)

                if not title:
                    skipped += 1
                    continue

                published = _normalize_pub_date(published)
                raw_id    = _make_id(url or (title + published))

                if raw_id in existing_ids:
                    skipped += 1
                    continue

                try:
                    conn.execute("""
                        INSERT OR IGNORE INTO news_raw
                          (id, title, summary, source, url, lang,
                           published_at, fetched_at, weight)
                        VALUES (?,?,?,?,?,?,?,?,?)
                    """, (raw_id, title, summary, source, url, lang,
                          published, now_iso, weight))
                    if conn.execute("SELECT changes()").fetchone()[0] > 0:
                        inserted += 1
                        existing_ids.add(raw_id)
                    else:
                        skipped += 1
                        existing_ids.add(raw_id)
                except Exception as e:
                    logger.debug(f"raw 삽입 실패: {e}")
                    skipped += 1

            if update_stats:
                conn.execute("""
                    INSERT INTO collection_stats(date, total_fetched, total_stored, duplicates)
                    VALUES(?,?,?,?)
                    ON CONFLICT(date) DO UPDATE SET
                        total_fetched = total_fetched + excluded.total_fetched,
                        total_stored  = total_stored  + excluded.total_stored,
                        duplicates    = duplicates    + excluded.duplicates
                """, (today, len(items), inserted, skipped))

        logger.debug(f"raw 삽입: {inserted}건 신규, {skipped}건 중복/스킵")
        return inserted, skipped

    def update_raw_importance(self, raw_id: str, importance: float,
                              tags: list = None) -> None:
        """AI 분류 후 중요도 및 태그 업데이트"""
        with self._tx() as conn:
            conn.execute("""
                UPDATE news_raw SET importance=?, tags=?
                WHERE id=?
            """, (importance, json.dumps(tags or [], ensure_ascii=False), raw_id))

    def compress_old_bodies(self, days_threshold: int = _COMPRESS_DAYS) -> int:
        """오래된 기사 본문 ZLIB 압축 (저장 공간 절약)"""
        cutoff = (datetime.now() - timedelta(days=days_threshold)).isoformat()
        conn   = self._conn()
        rows   = conn.execute("""
            SELECT id, body_full FROM news_raw
            WHERE published_at < ? AND body_compressed=0
              AND length(body_full) > 100
        """, (cutoff,)).fetchall()

        compressed = 0
        with self._tx() as conn:
            for row in rows:
                try:
                    data = zlib.compress(row["body_full"].encode("utf-8"), level=6)
                    conn.execute(
                        "UPDATE news_raw SET body_full=?, body_compressed=1 WHERE id=?",
                        (data, row["id"])
                    )
                    compressed += 1
                except Exception:
                    pass

        logger.info(f"본문 압축: {compressed}건")
        return compressed

    def get_raw(
        self,
        since_dt:  Optional[datetime] = None,
        until_dt:  Optional[datetime] = None,
        sources:   Optional[List[str]] = None,
        limit:     int = 1000,
        only_representative: bool = False,
    ) -> List[Dict]:
        """raw 기사 조회"""
        clauses = ["1=1"]
        params  = []

        if since_dt:
            clauses.append("published_at >= ?")
            params.append(since_dt.isoformat())
        if until_dt:
            clauses.append("published_at <= ?")
            params.append(until_dt.isoformat())
        if sources:
            placeholders = ",".join("?" * len(sources))
            clauses.append(f"source IN ({placeholders})")
            params.extend(sources)
        if only_representative:
            clauses.append("is_representative=1")

        where = " AND ".join(clauses)
        rows  = self._conn().execute(f"""
            SELECT * FROM news_raw WHERE {where}
            ORDER BY published_at DESC LIMIT ?
        """, params + [limit]).fetchall()

        return [dict(r) for r in rows]

    # ── Layer 2: 구조화 이벤트 ────────────────────────────────────────────────

    def insert_event(self, evt_dict: Dict) -> bool:
        """StructuredEvent 딕셔너리를 news_events에 삽입"""
        try:
            with self._tx() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO news_events
                      (event_id, raw_id, published_at, title,
                       categories, primary_category, keywords,
                       sentiment_score, impact_direction, impact_strength,
                       confidence, duration, target_scope,
                       target_markets, target_sectors, target_stocks,
                       event_type, cluster_id, is_representative,
                       repeat_count, computed_score, source_diversity)
                    VALUES
                      (?,?,?,?, ?,?,?, ?,?,?, ?,?,?, ?,?,?, ?,?,?, ?,?,?)
                """, (
                    evt_dict.get("event_id", _make_id(evt_dict.get("title",""))),
                    evt_dict.get("raw_id", ""),
                    evt_dict.get("published_at", datetime.now().isoformat()),
                    evt_dict.get("title", ""),
                    json.dumps(evt_dict.get("categories", []),  ensure_ascii=False),
                    evt_dict.get("primary_category", ""),
                    json.dumps(evt_dict.get("keywords", []),    ensure_ascii=False),
                    float(evt_dict.get("sentiment_score", 0.0)),
                    int(evt_dict.get("impact_direction", 0)),
                    float(evt_dict.get("impact_strength", 0.0)),
                    float(evt_dict.get("confidence", 0.0)),
                    evt_dict.get("duration", "short"),
                    evt_dict.get("target_scope", "market"),
                    json.dumps(evt_dict.get("target_markets", []), ensure_ascii=False),
                    json.dumps(evt_dict.get("target_sectors", []), ensure_ascii=False),
                    json.dumps(evt_dict.get("target_stocks", []),  ensure_ascii=False),
                    evt_dict.get("event_type", ""),
                    evt_dict.get("cluster_id", ""),
                    int(evt_dict.get("is_representative", 1)),
                    int(evt_dict.get("repeat_count", 1)),
                    float(evt_dict.get("computed_score", 0.0)),
                    float(evt_dict.get("source_diversity", 0.0)),
                ))
            return True
        except Exception as e:
            logger.debug(f"이벤트 삽입 실패: {e}")
            return False

    def insert_events_bulk(self, evt_dicts: List[Dict]) -> int:
        """이벤트 대량 삽입 — 단일 트랜잭션 (개별 트랜잭션 대비 10~50배 빠름)."""
        if not evt_dicts:
            return 0
        count = 0
        _sql = """
            INSERT OR REPLACE INTO news_events
              (event_id, raw_id, published_at, title,
               categories, primary_category, keywords,
               sentiment_score, impact_direction, impact_strength,
               confidence, duration, target_scope,
               target_markets, target_sectors, target_stocks,
               event_type, cluster_id, is_representative,
               repeat_count, computed_score, source_diversity)
            VALUES (?,?,?,?, ?,?,?, ?,?,?, ?,?,?, ?,?,?, ?,?,?, ?,?,?)
        """
        with self._tx() as conn:
            for d in evt_dicts:
                try:
                    conn.execute(_sql, (
                        d.get("event_id", _make_id(d.get("title", ""))),
                        d.get("raw_id", ""),
                        d.get("published_at", datetime.now().isoformat()),
                        d.get("title", ""),
                        json.dumps(d.get("categories", []),     ensure_ascii=False),
                        d.get("primary_category", ""),
                        json.dumps(d.get("keywords", []),       ensure_ascii=False),
                        float(d.get("sentiment_score", 0.0)),
                        int(d.get("impact_direction", 0)),
                        float(d.get("impact_strength", 0.0)),
                        float(d.get("confidence", 0.0)),
                        d.get("duration", "short"),
                        d.get("target_scope", "market"),
                        json.dumps(d.get("target_markets", []), ensure_ascii=False),
                        json.dumps(d.get("target_sectors", []), ensure_ascii=False),
                        json.dumps(d.get("target_stocks", []),  ensure_ascii=False),
                        d.get("event_type", ""),
                        d.get("cluster_id", ""),
                        int(d.get("is_representative", 1)),
                        int(d.get("repeat_count", 1)),
                        float(d.get("computed_score", 0.0)),
                        float(d.get("source_diversity", 0.0)),
                    ))
                    count += 1
                except Exception as e:
                    logger.debug(f"이벤트 삽입 실패: {e}")
        return count

    def get_events(
        self,
        since_dt:  Optional[datetime] = None,
        until_dt:  Optional[datetime] = None,
        symbol:    Optional[str] = None,
        sector:    Optional[str] = None,
        categories: Optional[List[str]] = None,
        min_score: float = -10.0,
        only_representative: bool = True,
        limit:     int = 500,
    ) -> List[Dict]:
        """구조화 이벤트 조회 (예측 feature 생성용)"""
        clauses = ["1=1"]
        params  = []

        if since_dt:
            clauses.append("published_at >= ?")
            params.append(since_dt.isoformat())
        if until_dt:
            clauses.append("published_at <= ?")
            params.append(until_dt.isoformat())
        if symbol:
            clauses.append("(target_scope='market' OR target_stocks LIKE ?)")
            params.append(f'%"{symbol}"%')
        if sector:
            clauses.append("target_sectors LIKE ?")
            params.append(f'%"{sector}"%')
        if categories:
            # OR 조건으로 카테고리 필터
            cat_clauses = " OR ".join(["categories LIKE ?"] * len(categories))
            clauses.append(f"({cat_clauses})")
            params.extend([f'%"{c}"%' for c in categories])
        if only_representative:
            clauses.append("is_representative=1")
        if min_score > -10.0:
            clauses.append("abs(computed_score) >= ?")
            params.append(abs(min_score))

        where = " AND ".join(clauses)
        rows  = self._conn().execute(f"""
            SELECT * FROM news_events WHERE {where}
            ORDER BY published_at DESC LIMIT ?
        """, params + [limit]).fetchall()

        result = []
        for r in rows:
            d = dict(r)
            for json_col in ("categories", "keywords", "target_markets",
                             "target_sectors", "target_stocks"):
                try:
                    d[json_col] = json.loads(d[json_col])
                except Exception:
                    d[json_col] = []
            result.append(d)
        return result

    def get_event_stats(self, date: str) -> Dict:
        """특정 날짜의 이벤트 통계"""
        row = self._conn().execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN impact_direction=1  THEN 1 ELSE 0 END) as bullish,
                SUM(CASE WHEN impact_direction=-1 THEN 1 ELSE 0 END) as bearish,
                AVG(sentiment_score) as avg_sentiment,
                AVG(impact_strength) as avg_strength,
                SUM(repeat_count) as total_reports
            FROM news_events
            WHERE published_at LIKE ?
        """, (f"{date}%",)).fetchone()
        return dict(row) if row else {}

    # ── Layer 3: 특징 벡터 캐시 ───────────────────────────────────────────────

    def cache_features(self, symbol: str, date: str, window: str,
                       features: List[float], event_count: int = 0) -> None:
        """모델 입력용 특징 벡터 캐싱"""
        key = f"{symbol}|{date}|{window}"
        with self._tx() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO news_feat_cache
                  (cache_key, symbol, date, window, features, computed_at, event_count)
                VALUES (?,?,?,?,?,?,?)
            """, (key, symbol, date, window,
                  json.dumps(features), datetime.now().isoformat(), event_count))

    def get_cached_features(self, symbol: str, date: str,
                            window: str, max_age_hours: int = 6) -> Optional[List[float]]:
        """캐시된 특징 벡터 조회 (만료 시 None)"""
        key     = f"{symbol}|{date}|{window}"
        cutoff  = (datetime.now() - timedelta(hours=max_age_hours)).isoformat()
        row     = self._conn().execute("""
            SELECT features FROM news_feat_cache
            WHERE cache_key=? AND computed_at >= ?
        """, (key, cutoff)).fetchone()
        if row:
            return json.loads(row["features"])
        return None

    def invalidate_feature_cache(self, symbol: str = None, date: str = None) -> int:
        """특징 캐시 무효화"""
        if symbol and date:
            r = self._conn().execute(
                "DELETE FROM news_feat_cache WHERE symbol=? AND date=?",
                (symbol, date))
        elif symbol:
            r = self._conn().execute(
                "DELETE FROM news_feat_cache WHERE symbol=?", (symbol,))
        else:
            r = self._conn().execute("DELETE FROM news_feat_cache")
        self._conn().commit()
        return r.rowcount

    # ── 통계 / 유지보수 ────────────────────────────────────────────────────────

    def get_summary(self) -> Dict:
        """DB 전체 현황 요약"""
        conn = self._conn()
        raw_count   = conn.execute("SELECT COUNT(*) FROM news_raw").fetchone()[0]
        evt_count   = conn.execute("SELECT COUNT(*) FROM news_events").fetchone()[0]
        feat_count  = conn.execute("SELECT COUNT(*) FROM news_feat_cache").fetchone()[0]
        oldest_raw  = conn.execute(
            "SELECT MIN(published_at) FROM news_raw").fetchone()[0] or ""
        newest_raw  = conn.execute(
            "SELECT MAX(published_at) FROM news_raw").fetchone()[0] or ""
        return {
            "raw_articles":    raw_count,
            "structured_events": evt_count,
            "feature_caches":  feat_count,
            "oldest_article":  oldest_raw[:10],
            "newest_article":  newest_raw[:10],
            "db_size_mb":      round(os.path.getsize(self.db_path) / 1_048_576, 2)
                               if os.path.exists(self.db_path) else 0,
        }

    # ── 백필 이력 ─────────────────────────────────────────────────────────────

    def is_backfilled(self, start_date: str, end_date: str,
                      category: str = "*") -> bool:
        """해당 기간+카테고리가 이미 수집됐는지 확인"""
        row = self._conn().execute("""
            SELECT 1 FROM backfill_log
            WHERE start_date=? AND end_date=? AND category=?
        """, (start_date, end_date, category)).fetchone()
        return row is not None

    def mark_backfilled(self, start_date: str, end_date: str,
                        category: str = "*",
                        inserted: int = 0, skipped: int = 0,
                        source: str = "gdelt") -> None:
        """백필 완료 기록"""
        log_id = f"{start_date}_{end_date}_{category}"
        with self._tx() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO backfill_log
                  (id, start_date, end_date, category, inserted, skipped,
                   collected_at, source)
                VALUES (?,?,?,?,?,?,?,?)
            """, (log_id, start_date, end_date, category,
                  inserted, skipped, datetime.now().isoformat(), source))

    def get_backfill_log(self, limit: int = 100) -> List[Dict]:
        """백필 이력 조회 (최신순)"""
        rows = self._conn().execute("""
            SELECT * FROM backfill_log
            ORDER BY collected_at DESC LIMIT ?
        """, (limit,)).fetchall()
        return [dict(r) for r in rows]

    def get_backfill_coverage(self) -> Dict:
        """수집된 기간 범위 요약"""
        conn = self._conn()
        total_periods = conn.execute(
            "SELECT COUNT(*) FROM backfill_log WHERE category='*' OR category='ALL'"
        ).fetchone()[0]
        earliest = conn.execute(
            "SELECT MIN(start_date) FROM backfill_log"
        ).fetchone()[0] or "없음"
        latest_end = conn.execute(
            "SELECT MAX(end_date) FROM backfill_log"
        ).fetchone()[0] or "없음"
        cat_counts = conn.execute("""
            SELECT category, COUNT(*) as cnt, SUM(inserted) as total_ins
            FROM backfill_log
            GROUP BY category
            ORDER BY cnt DESC
        """).fetchall()
        return {
            "total_periods": total_periods,
            "earliest": earliest,
            "latest_end": latest_end,
            "categories": [dict(r) for r in cat_counts],
        }

    def get_unclassified_raw(
        self,
        since_dt: Optional[datetime] = None,
        until_dt: Optional[datetime] = None,
        limit: int = 300,
    ) -> List[Dict]:
        """news_events에 아직 분류되지 않은 news_raw 기사 반환 (LEFT JOIN 방식)"""
        clauses = ["e.event_id IS NULL"]
        params: list = []
        if since_dt:
            clauses.append("r.published_at >= ?")
            params.append(since_dt.isoformat())
        if until_dt:
            clauses.append("r.published_at <= ?")
            params.append(until_dt.isoformat())
        where = " AND ".join(clauses)
        rows = self._conn().execute(f"""
            SELECT r.* FROM news_raw r
            LEFT JOIN news_events e ON e.raw_id = r.id
            WHERE {where}
            ORDER BY r.published_at DESC LIMIT ?
        """, params + [limit]).fetchall()
        return [dict(r) for r in rows]

    def count_unclassified_raw(
        self,
        since_dt: Optional[datetime] = None,
        until_dt: Optional[datetime] = None,
    ) -> int:
        """미분류 news_raw 기사 수 반환"""
        clauses = ["e.event_id IS NULL"]
        params: list = []
        if since_dt:
            clauses.append("r.published_at >= ?")
            params.append(since_dt.isoformat())
        if until_dt:
            clauses.append("r.published_at <= ?")
            params.append(until_dt.isoformat())
        where = " AND ".join(clauses)
        row = self._conn().execute(f"""
            SELECT COUNT(*) FROM news_raw r
            LEFT JOIN news_events e ON e.raw_id = r.id
            WHERE {where}
        """, params).fetchone()
        return row[0] if row else 0

    def purge_orphan_events(self) -> int:
        """잘못된 news_events 레코드 정리:
        1) raw_id가 비어있는 레코드
        2) raw_id가 news_raw에 없는 고아 레코드
        3) 같은 raw_id에 여러 event가 있는 중복 — MD5(raw_id) 포맷 우선 보존
        """
        total = 0
        conn = self._conn()

        with self._tx() as conn:
            # 1) raw_id 없는 레코드
            r = conn.execute("DELETE FROM news_events WHERE raw_id = '' OR raw_id IS NULL")
            total += r.rowcount

            # 2) 고아 레코드 (raw_id가 news_raw에 없음)
            r = conn.execute("""
                DELETE FROM news_events
                WHERE NOT EXISTS (
                    SELECT 1 FROM news_raw WHERE news_raw.id = news_events.raw_id
                )
            """)
            total += r.rowcount

        # 3) 같은 raw_id 중복 — MD5(raw_id)[:12] 포맷인 것을 우선 보존
        #    SQL에서 MD5를 계산할 수 없으므로 Python에서 처리
        dup_rows = conn.execute("""
            SELECT raw_id, COUNT(*) as cnt
            FROM news_events
            GROUP BY raw_id
            HAVING cnt > 1
        """).fetchall()

        if dup_rows:
            delete_ids: list = []
            for row in dup_rows:
                raw_id = row["raw_id"] if isinstance(row, sqlite3.Row) else row[0]
                correct_eid = hashlib.md5(raw_id.encode()).hexdigest()[:12]
                evts = conn.execute(
                    "SELECT event_id FROM news_events WHERE raw_id = ?", (raw_id,)
                ).fetchall()
                existing = [e[0] for e in evts]
                if correct_eid in existing:
                    # 정규 event_id 보존, 나머지 삭제
                    delete_ids.extend(e for e in existing if e != correct_eid)
                else:
                    # 정규 event_id 없음: 첫 번째만 보존, 나머지 삭제
                    delete_ids.extend(existing[1:])

            if delete_ids:
                with self._tx() as conn2:
                    for eid in delete_ids:
                        conn2.execute("DELETE FROM news_events WHERE event_id = ?", (eid,))
                total += len(delete_ids)

        if total > 0:
            logger.info(f"news_events 정리: {total}건 삭제")
        return total

    def title_exists_on_date(self, title: str, date_str: str) -> bool:
        """같은 날짜에 같은 제목의 기사가 이미 있는지 확인 (2차 중복 제거용)"""
        title_hash = hashlib.sha256(title.lower().strip().encode()).hexdigest()[:8]
        row = self._conn().execute("""
            SELECT 1 FROM news_raw
            WHERE published_at LIKE ? AND id LIKE ?
            LIMIT 1
        """, (f"{date_str}%", f"%")).fetchone()
        # title 정확 매칭
        row2 = self._conn().execute("""
            SELECT 1 FROM news_raw
            WHERE published_at LIKE ? AND title=?
            LIMIT 1
        """, (f"{date_str}%", title)).fetchone()
        return row2 is not None

    def vacuum(self) -> None:
        """DB 최적화 (VACUUM)"""
        self._conn().execute("VACUUM")
        logger.info("NewsDB VACUUM 완료")

    def delete_old_raws(self, keep_days: int = _ARCHIVE_DAYS) -> int:
        """오래된 원본 기사 삭제 (구조화 이벤트는 보존)"""
        cutoff = (datetime.now() - timedelta(days=keep_days)).isoformat()
        with self._tx() as conn:
            r = conn.execute(
                "DELETE FROM news_raw WHERE published_at < ?", (cutoff,))
        logger.info(f"오래된 raw 삭제: {r.rowcount}건 (기준: {keep_days}일)")
        return r.rowcount

    def close(self) -> None:
        """커넥션 종료"""
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None


# ── 유틸 ──────────────────────────────────────────────────────────────────────

def _make_id(text: str) -> str:
    """텍스트 → SHA256 8자리 ID"""
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]


# ── 싱글턴 접근 ───────────────────────────────────────────────────────────────

_db_instance: Optional[NewsDB] = None
_db_lock = threading.Lock()


def get_news_db(db_path: str = None) -> NewsDB:
    """
    싱글턴 NewsDB 인스턴스 반환.
    db_path를 처음 호출 시에만 지정하면 이후 호출에서 재사용.
    """
    global _db_instance
    with _db_lock:
        if _db_instance is None:
            path = db_path or os.path.join("outputs", "news.db")
            _db_instance = NewsDB(path)
        return _db_instance
