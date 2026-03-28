# data/news_pipeline.py — 통합 뉴스 AI 분류 파이프라인
"""
뉴스 수집과 AI 분류를 통합하는 단일 파이프라인.

설계 원칙:
  - 수집(RSS/GDELT) → news_raw  : 소스별 수집기 담당
  - 분류(AI 12카테고리) → news_events : 이 파이프라인 담당
  - 표시 → news_events에서만 읽음

흐름:
  RSS 수집기  ─┐
  GDELT 백필 ─┤→  news_raw  →  [NewsPipeline.classify_pending()]  →  news_events  →  UI
  기타 소스   ─┘

사용법:
    pipeline = get_pipeline()
    pipeline.start_auto(interval_min=5, on_done=callback)
    pipeline.classify_pending(batch=200)   # 수동 트리거
    stats = pipeline.get_stats()
"""
from __future__ import annotations

import datetime
import hashlib
import logging
import threading
from typing import Callable, Dict, List, Optional

logger = logging.getLogger("quant.pipeline")


class NewsPipeline:
    """
    news_raw 미분류 기사 → AI 12카테고리 분류 → news_events 저장.
    백그라운드 자동 실행 또는 수동 트리거 모두 지원.
    """

    def __init__(self, db_path: Optional[str] = None):
        from data.news_db import get_news_db
        self.db = get_news_db(db_path)
        self._stop   = threading.Event()
        self._lock   = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stats: Dict = {
            "classified":  0,
            "pending":     0,
            "last_run":    None,
            "running":     False,
        }
        # 시작 시 raw_id가 없는 고아 이벤트 정리 (이전 버전 충돌 레코드 제거)
        try:
            self.db.purge_orphan_events()
        except Exception:
            pass

    # ──────────────────────────────────────────────────────────────────────────
    # 핵심 — 분류 실행
    # ──────────────────────────────────────────────────────────────────────────

    def classify_pending(
        self,
        batch:       int = 300,
        days_back:   Optional[int] = None,
        progress_cb: Optional[Callable[[str], None]] = None,
    ) -> Dict:
        """
        news_raw의 미분류 기사를 AI 12카테고리로 분류 후 news_events에 저장.

        Args:
            batch:       한 번에 처리할 최대 기사 수
            days_back:   None이면 전체, 숫자면 최근 N일만
            progress_cb: 진행 메시지 콜백

        Returns:
            {"classified": int, "total": int, "pending": int}
        """
        _log = progress_cb or (lambda m: logger.info(m))

        def _diag(msg: str) -> None:
            """진단 로그: 항상 logger.info(콘솔/파일) + 있으면 progress_cb(UI)에도 출력."""
            logger.info(msg)
            if progress_cb:
                try:
                    progress_cb(msg)
                except Exception:
                    pass

        with self._lock:
            if self._stats.get("running"):
                _log("⚠ 이미 분류 중 — 중복 실행 스킵")
                return self._stats
            self._stats["running"] = True

        try:
            conn = self.db._conn()
            since = (datetime.datetime.now() - datetime.timedelta(days=days_back)
                     if days_back else None)

            # ── 시작 전 DB 상태 전수 출력 ──────────────────────────────────
            raw_total_before   = conn.execute("SELECT COUNT(*) FROM news_raw").fetchone()[0]
            evt_total_before   = conn.execute("SELECT COUNT(*) FROM news_events").fetchone()[0]
            evt_with_rawid     = conn.execute("SELECT COUNT(*) FROM news_events WHERE raw_id != '' AND raw_id IS NOT NULL").fetchone()[0]
            evt_no_rawid       = conn.execute("SELECT COUNT(*) FROM news_events WHERE raw_id = '' OR raw_id IS NULL").fetchone()[0]
            pending_before     = self.db.count_unclassified_raw()
            _diag(f"[진단] 시작 전 ─────────────────────────────")
            _diag(f"[진단]   news_raw 총계       : {raw_total_before:,}")
            _diag(f"[진단]   news_events 총계     : {evt_total_before:,}")
            _diag(f"[진단]   events(raw_id 있음) : {evt_with_rawid:,}")
            _diag(f"[진단]   events(raw_id 없음) : {evt_no_rawid:,}")
            _diag(f"[진단]   미분류(LEFT JOIN)    : {pending_before:,}")
            _diag(f"[진단]   batch 파라미터       : {batch}")
            _diag(f"[진단] ─────────────────────────────────────")

            rows = self.db.get_unclassified_raw(since_dt=since, limit=batch)

            if not rows:
                _diag(f"[진단] 미분류 없음 → 종료")
                self._stats.update({"classified": 0, "total": 0,
                                    "pending": 0, "last_run": datetime.datetime.now()})
                return self._stats

            _diag(f"[진단] get_unclassified_raw 반환: {len(rows)}건")
            if rows:
                _diag(f"[진단]   첫번째 raw_id: {rows[0].get('id','(없음)')}")
                _diag(f"[진단]   마지막 raw_id: {rows[-1].get('id','(없음)')}")

            # 카테고리 분류기 로드
            try:
                from features.external_env.categorizer import NewsEventCategorizer
                categorizer = NewsEventCategorizer()
            except ImportError as e:
                _log(f"⚠ categorizer 없음: {e}")
                return self._stats

            classified = 0
            empty_id_count = 0
            dicts: List[Dict] = []
            now = datetime.datetime.now()

            for i, r in enumerate(rows):
                if self._stop.is_set():
                    _log(f"[중단] {i}건 처리 후 중단")
                    break

                if progress_cb and i > 0 and i % 50 == 0:
                    _log(f"  분류 중... {i}/{len(rows)}건")

                try:
                    raw_id = r.get("id", "")
                    if not raw_id:
                        empty_id_count += 1
                        continue  # raw_id 없는 건 건너뜀

                    pub_str = r.get("published_at", "")
                    try:
                        ts = datetime.datetime.fromisoformat(pub_str)
                    except Exception:
                        ts = now
                    if hasattr(ts, "tzinfo") and ts.tzinfo is not None:
                        ts = ts.replace(tzinfo=None)

                    evt = categorizer.classify(
                        title=r.get("title", ""),
                        summary=r.get("summary", ""),
                        url=r.get("url", ""),
                        ts=ts,
                    )
                    evt.compute_score()

                    pcat       = getattr(evt, "primary_cat", None)
                    impact_dir = getattr(evt, "impact_direction", None)
                    dur        = getattr(evt, "duration", None)
                    cats       = [c.value for c in (getattr(evt, "categories", []) or [])]

                    event_id = hashlib.md5(raw_id.encode()).hexdigest()[:12]

                    dicts.append({
                        "event_id":         event_id,
                        "raw_id":           raw_id,
                        "published_at":     ts.isoformat(),
                        "title":            r.get("title", ""),
                        "categories":       cats,
                        "primary_category": pcat.value if pcat else "",
                        "keywords":         getattr(evt, "keywords", []),
                        "sentiment_score":  float(getattr(evt, "sentiment_score", 0.0)),
                        "impact_direction": getattr(impact_dir, "value", 0),
                        "impact_strength":  float(getattr(evt, "impact_strength", 0.0)),
                        "confidence":       float(getattr(evt, "confidence", 0.0)),
                        "duration":         (dur.value if dur and hasattr(dur, "value")
                                             else "short"),
                        "target_scope":     getattr(evt, "target_market", "market"),
                        "target_markets":   [],
                        "target_sectors":   getattr(evt, "target_sectors", []),
                        "target_stocks":    getattr(evt, "target_tickers", []),
                        "event_type":       getattr(evt, "event_type", ""),
                        "cluster_id":       "",
                        "is_representative": 1,
                        "repeat_count":     1,
                        "computed_score":   float(getattr(evt, "external_score", 0.0)),
                        "source_diversity": 0.0,
                    })
                    classified += 1

                except Exception as e:
                    logger.debug(f"분류 실패 ({r.get('title','')[:30]}): {e}")

            _diag(f"[진단] 분류 루프 결과: {classified}건 성공 / {empty_id_count}건 raw_id 없어 스킵")

            # DB 저장
            saved = 0
            if dicts:
                saved = self.db.insert_events_bulk(dicts)
            _diag(f"[진단] insert_events_bulk: {len(dicts)}건 시도 → {saved}건 저장")

            # ── 저장 후 DB 상태 전수 출력 ──────────────────────────────────
            raw_total_after    = conn.execute("SELECT COUNT(*) FROM news_raw").fetchone()[0]
            evt_total_after    = conn.execute("SELECT COUNT(*) FROM news_events").fetchone()[0]
            evt_with_rawid_a   = conn.execute("SELECT COUNT(*) FROM news_events WHERE raw_id != '' AND raw_id IS NOT NULL").fetchone()[0]
            pending_after      = self.db.count_unclassified_raw()
            classified_ok      = raw_total_after - pending_after

            _diag(f"[진단] 저장 후 ─────────────────────────────")
            _diag(f"[진단]   news_raw 총계       : {raw_total_after:,}")
            _diag(f"[진단]   news_events 총계     : {evt_total_after:,}  (이전: {evt_total_before:,}, 증가: {evt_total_after - evt_total_before:+,})")
            _diag(f"[진단]   events(raw_id 있음) : {evt_with_rawid_a:,}")
            _diag(f"[진단]   미분류(LEFT JOIN)    : {pending_after:,}  (이전: {pending_before:,}, 감소: {pending_before - pending_after:+,})")
            _diag(f"[진단]   분류완료(계산값)      : {classified_ok:,}")
            if evt_total_after > raw_total_after:
                _diag(f"[진단] ⚠ news_events({evt_total_after:,}) > news_raw({raw_total_after:,}) — 중복/고아 레코드 존재!")
            _diag(f"[진단] ─────────────────────────────────────")

            self._stats.update({
                "classified":    saved,
                "pending":       pending_after,
                "classified_ok": classified_ok,
                "raw_total":     raw_total_after,
                "last_run":      datetime.datetime.now(),
            })
            return self._stats

        finally:
            with self._lock:
                self._stats["running"] = False

    # ──────────────────────────────────────────────────────────────────────────
    # 자동 실행
    # ──────────────────────────────────────────────────────────────────────────

    def start_auto(
        self,
        interval_min: int = 5,
        on_done: Optional[Callable[[Dict], None]] = None,
    ) -> None:
        """
        백그라운드에서 N분마다 classify_pending() 자동 실행.
        on_done: 분류 완료 시 stats dict를 인자로 호출되는 콜백.
        """
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()

        def _loop():
            # 시작 즉시 1회 실행
            try:
                stats = self.classify_pending(batch=200)
                if on_done:
                    on_done(stats)
            except Exception as e:
                logger.debug(f"파이프라인 초기 실행 오류: {e}")

            while not self._stop.wait(interval_min * 60):
                try:
                    stats = self.classify_pending(batch=200)
                    if on_done:
                        on_done(stats)
                except Exception as e:
                    logger.debug(f"파이프라인 자동 실행 오류: {e}")

        self._thread = threading.Thread(target=_loop, daemon=True,
                                        name="news-pipeline-auto")
        self._thread.start()
        logger.info(f"뉴스 AI 분류 파이프라인 시작 (간격: {interval_min}분)")

    def stop(self) -> None:
        self._stop.set()

    # ──────────────────────────────────────────────────────────────────────────
    # 통계 / 조회
    # ──────────────────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict:
        """현재 파이프라인 통계 반환"""
        return dict(self._stats)

    def get_db_summary(self) -> Dict:
        """DB 전체 현황 요약"""
        return self.db.get_summary()

    def get_structured_events(
        self,
        days: int = 7,
        limit: int = 2000,
    ) -> list:
        """
        news_events DB에서 StructuredEvent 목록 반환.
        UI 표시용 — 분류가 완료된 데이터만.
        """
        try:
            from features.external_env.event_structure import StructuredEvent
            import datetime as _dt

            now   = _dt.datetime.now()
            since = now - _dt.timedelta(days=days)

            # 단일 쿼리로 조회 — since_dt 기준으로 최신순 limit건
            # (이전의 7일 분할 521회 쿼리 방식 제거 → 즉시 반환)
            rows: list = self.db.get_events(
                since_dt=since, only_representative=False, limit=limit
            )

            events: list = []
            seen: set = set()
            for row in rows:
                try:
                    eid = row.get("event_id", "")
                    if eid in seen:
                        continue
                    seen.add(eid)
                    d_map = {
                        "event_id":         eid,
                        "title":            row.get("title", ""),
                        "timestamp":        row.get("published_at", ""),
                        "categories":       row.get("categories", []),
                        "primary_cat":      row.get("primary_category", ""),
                        "event_type":       row.get("event_type", ""),
                        "impact_direction": row.get("impact_direction", 0),
                        "impact_strength":  float(row.get("impact_strength", 0.0)),
                        "confidence":       float(row.get("confidence", 0.5)),
                        "target_sectors":   row.get("target_sectors", []),
                        "duration":         row.get("duration", "short"),
                        "keywords":         row.get("keywords", []),
                        "sentiment_score":  float(row.get("sentiment_score", 0.0)),
                        "importance":       0.5,
                        "external_score":   float(row.get("computed_score", 0.0)),
                    }
                    evt = StructuredEvent.from_dict(d_map)
                    events.append(evt)
                except Exception:
                    pass

            return events

        except Exception as e:
            logger.debug(f"get_structured_events 실패: {e}")
            return []


# ── 싱글턴 ────────────────────────────────────────────────────────────────────

_pipeline_instance: Optional[NewsPipeline] = None
_pipeline_lock = threading.Lock()


def get_pipeline(db_path: Optional[str] = None) -> NewsPipeline:
    """싱글턴 NewsPipeline 인스턴스 반환"""
    global _pipeline_instance
    with _pipeline_lock:
        if _pipeline_instance is None:
            _pipeline_instance = NewsPipeline(db_path)
        return _pipeline_instance
