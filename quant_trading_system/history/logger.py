"""
history/logger.py
=================
Append-only prediction history logger.

Storage layout (under history_dir/):
    predictions.jsonl          ← every PredictionRecord, one JSON per line
    predictions_YYYY-MM.jsonl  ← monthly shard (same format, for fast filtering)

Design guarantees
-----------------
* Append-only: records are never silently overwritten or deleted.
* Thread-safe: a single lock guards all file I/O.
* Crash-safe: each record is flushed + fsync'd before returning.
* Idempotent IDs: writing the same id twice is rejected (returns False).
"""
from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

from history.schema import PredictionRecord

logger = logging.getLogger("quant.history.logger")

_MAIN_FILE    = "predictions.jsonl"


class PredictionLogger:
    """
    Append-only file-based prediction history store.

    Usage
    -----
    pl = PredictionLogger("outputs/history")
    rec = PredictionRecord(symbol="005930", ...)
    pl.append(rec)

    all_recs = pl.load_all()
    recent   = pl.load_since("2026-01-01")
    """

    def __init__(self, history_dir: str) -> None:
        self._dir = Path(history_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._main_path = self._dir / _MAIN_FILE
        self._lock = threading.Lock()
        # In-memory id set: prevents duplicate writes in the same session
        self._seen_ids: set[str] = self._load_ids()

    # ──────────────────────────────────────────────────────────────────────────
    # 쓰기
    # ──────────────────────────────────────────────────────────────────────────

    def append(self, record: PredictionRecord) -> bool:
        """
        PredictionRecord 를 추가 (append-only).

        Returns:
            True  — 기록 성공
            False — 중복 id (이미 존재하는 예측)
        """
        with self._lock:
            if record.id in self._seen_ids:
                logger.debug("Duplicate prediction id skipped: %s", record.id)
                return False

            data = json.dumps(record.to_dict(), ensure_ascii=False)
            # 월별 샤드 경로 계산 (ISO timestamp 앞 7자: "YYYY-MM")
            shard_path = self._shard_path(record.timestamp)

            # 메인 파일에 쓰기
            with self._main_path.open("a", encoding="utf-8") as f:
                f.write(data + "\n")
                f.flush()
                os.fsync(f.fileno())

            # 월별 샤드에도 쓰기 (선택적 — 빠른 필터링용)
            with shard_path.open("a", encoding="utf-8") as f:
                f.write(data + "\n")
                f.flush()
                os.fsync(f.fileno())

            self._seen_ids.add(record.id)

        logger.info("Prediction logged: %s %s %s",
                    record.symbol, record.action, record.timestamp[:10])
        return True

    def update_verification(self, record_id: str, updates: dict[str, Any]) -> bool:
        """
        기존 예측의 검증 필드(verified, actual_return_pct, hit, verification_date)를
        *새 레코드를 추가*하는 방식으로 갱신한다 (append-only 원칙 유지).

        검증 업데이트를 별도 레코드로 쓰지 않고, 원본 레코드를 수정한 버전을
        파일 끝에 덧붙인다. load_all() 은 중복 id 중 마지막 항목을 사용한다.
        """
        with self._lock:
            # 원본 로드
            original = self._find_by_id(record_id)
            if original is None:
                logger.warning("update_verification: id not found %s", record_id)
                return False

            # 허용 필드만 반영
            d = original.to_dict()
            for k in ("verified", "actual_return_pct", "hit", "verification_date"):
                if k in updates:
                    d[k] = updates[k]

            updated_record = PredictionRecord.from_dict(d)
            data = json.dumps(updated_record.to_dict(), ensure_ascii=False)

            with self._main_path.open("a", encoding="utf-8") as f:
                f.write(data + "\n")
                f.flush()
                os.fsync(f.fileno())

            shard_path = self._shard_path(original.timestamp)
            with shard_path.open("a", encoding="utf-8") as f:
                f.write(data + "\n")
                f.flush()
                os.fsync(f.fileno())

        logger.info("Verification updated for prediction %s", record_id)
        return True

    # ──────────────────────────────────────────────────────────────────────────
    # 읽기
    # ──────────────────────────────────────────────────────────────────────────

    def load_all(self) -> list[PredictionRecord]:
        """
        전체 예측 이력을 반환한다.
        중복 id 가 있으면 *마지막* 항목(최신 업데이트)을 사용한다.
        최신 순으로 정렬한다.
        """
        with self._lock:
            raw = self._read_jsonl(self._main_path)

        dedup: dict[str, dict] = {}
        for row in raw:
            rec_id = row.get("id", "")
            if rec_id:
                dedup[rec_id] = row

        records = [PredictionRecord.from_dict(d) for d in dedup.values()]
        records.sort(key=lambda r: r.timestamp, reverse=True)
        return records

    def load_since(self, date_str: str) -> list[PredictionRecord]:
        """
        date_str (ISO 날짜, 예: "2026-01-15") 이후의 예측만 반환.
        월별 샤드를 이용해 탐색 범위를 줄인다.
        """
        all_recs = self.load_all()
        return [r for r in all_recs if r.timestamp[:10] >= date_str]

    def load_by_symbol(self, symbol: str) -> list[PredictionRecord]:
        """특정 종목의 예측 이력만 반환."""
        return [r for r in self.load_all() if r.symbol == symbol]

    def load_unverified(self) -> list[PredictionRecord]:
        """아직 검증되지 않은 예측만 반환."""
        return [r for r in self.load_all() if not r.verified]

    def load_page(self, page: int = 0, page_size: int = 50,
                  symbol: str = "",
                  action: str = "",
                  verified_only: bool = False,
                  unverified_only: bool = False,
                  date_from: str = "",
                  date_to: str = "") -> tuple[list[PredictionRecord], int]:
        """
        필터 + 페이지네이션 로드.

        Returns:
            (page_records, total_count)
        """
        recs = self.load_all()

        if symbol:
            recs = [r for r in recs if r.symbol == symbol]
        if action:
            recs = [r for r in recs if r.action == action]
        if verified_only:
            recs = [r for r in recs if r.verified]
        if unverified_only:
            recs = [r for r in recs if not r.verified]
        if date_from:
            recs = [r for r in recs if r.timestamp[:10] >= date_from]
        if date_to:
            recs = [r for r in recs if r.timestamp[:10] <= date_to]

        total = len(recs)
        start = page * page_size
        return recs[start:start + page_size], total

    # ──────────────────────────────────────────────────────────────────────────
    # 통계
    # ──────────────────────────────────────────────────────────────────────────

    def summary_stats(self) -> dict[str, Any]:
        """
        전체 이력에 대한 요약 통계를 반환한다.
        (적중률, 평균 예측 수익률, 평균 실제 수익률, MAPE 등)
        """
        recs = self.load_all()
        verified = [r for r in recs if r.verified and r.hit is not None]

        if not verified:
            return {
                "total": len(recs),
                "verified": 0,
                "hit_rate": None,
                "avg_predicted_return": None,
                "avg_actual_return": None,
                "mape": None,
                "buy_precision": None,
                "sell_precision": None,
            }

        hits      = [r for r in verified if r.hit]
        hit_rate  = len(hits) / len(verified)

        pred_rets = [r.predicted_return_pct for r in verified]
        act_rets  = [r.actual_return_pct for r in verified
                     if r.actual_return_pct is not None]

        avg_pred = sum(pred_rets) / len(pred_rets) if pred_rets else None

        avg_act  = sum(act_rets) / len(act_rets)   if act_rets  else None

        # MAPE
        mape_vals = [
            abs((r.actual_return_pct - r.predicted_return_pct)
                / (abs(r.predicted_return_pct) + 1e-8)) * 100
            for r in verified
            if r.actual_return_pct is not None
        ]
        mape = sum(mape_vals) / len(mape_vals) if mape_vals else None

        # BUY precision
        buy_v = [r for r in verified if r.action == "BUY"]
        buy_p = (len([r for r in buy_v if r.hit]) / len(buy_v)
                 if buy_v else None)

        # SELL precision
        sell_v = [r for r in verified if r.action == "SELL"]
        sell_p = (len([r for r in sell_v if r.hit]) / len(sell_v)
                  if sell_v else None)

        return {
            "total": len(recs),
            "verified": len(verified),
            "hit_rate": round(hit_rate, 4),
            "avg_predicted_return": round(avg_pred, 4) if avg_pred is not None else None,
            "avg_actual_return":    round(avg_act,  4) if avg_act  is not None else None,
            "mape": round(mape, 2) if mape is not None else None,
            "buy_precision":  round(buy_p,  4) if buy_p  is not None else None,
            "sell_precision": round(sell_p, 4) if sell_p is not None else None,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # 내부 헬퍼
    # ──────────────────────────────────────────────────────────────────────────

    def _shard_path(self, timestamp: str) -> Path:
        """예: "2026-03-22T..." → predictions_2026-03.jsonl"""
        month = timestamp[:7] if len(timestamp) >= 7 else "unknown"
        return self._dir / f"predictions_{month}.jsonl"

    def _load_ids(self) -> set[str]:
        """시작 시 기존 파일에서 id 집합을 복원한다."""
        ids: set[str] = set()
        if not self._main_path.exists():
            return ids
        for row in self._read_jsonl(self._main_path):
            rec_id = row.get("id", "")
            if rec_id:
                ids.add(rec_id)
        return ids

    def _find_by_id(self, record_id: str) -> PredictionRecord | None:
        """메인 파일에서 id 에 해당하는 *마지막* 레코드를 반환."""
        found: dict | None = None
        for row in self._read_jsonl(self._main_path):
            if row.get("id") == record_id:
                found = row
        return PredictionRecord.from_dict(found) if found else None

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        if not path.exists():
            return rows
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except OSError:
            pass
        return rows
