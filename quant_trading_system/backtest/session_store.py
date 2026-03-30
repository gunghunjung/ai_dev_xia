"""backtest/session_store.py — Thread-safe persistence layer for BacktestSession objects."""
from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

from history.schema import BacktestSession

logger = logging.getLogger("quant.backtest.session_store")

_INDEX_FILE = "sessions.jsonl"


def _session_summary(session: BacktestSession) -> dict[str, Any]:
    """Return the lightweight summary dict written to sessions.jsonl."""
    return {
        "id":               session.session_id,
        "timestamp":        session.timestamp,
        "symbols":          session.symbols,
        "total_return_pct": session.total_return_pct,
        "sharpe":           session.sharpe,
        "cagr":             session.cagr,
        "max_drawdown":     session.max_drawdown,
        "num_trades":       session.n_trades,
        "win_rate":         session.win_rate,
        "notes":            session.notes,
    }


class BacktestSessionStore:
    """
    BacktestSession 영속성 저장소.

    구조:
        store_dir/
            sessions.jsonl          ← 요약 인덱스 (한 줄 = 한 세션 JSON)
            {session_id}.json       ← 전체 세션 데이터
    """

    def __init__(self, store_dir: str) -> None:
        self._dir = Path(store_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._dir / _INDEX_FILE
        self._lock = threading.Lock()

        # 인덱스 파일이 없으면 빈 파일 생성
        if not self._index_path.exists():
            self._index_path.touch()

    # ──────────────────────────────────────────────────────────────────────────
    # 저장
    # ──────────────────────────────────────────────────────────────────────────

    def save(self, session: BacktestSession) -> str:
        """
        세션을 {session_id}.json 에 저장하고 인덱스에 요약을 추가한다.

        Returns:
            session_id (str)
        """
        with self._lock:
            session_path = self._dir / f"{session.session_id}.json"

            # 전체 세션 저장 (schema의 to_dict() 사용)
            with session_path.open("w", encoding="utf-8") as f:
                json.dump(session.to_dict(), f, ensure_ascii=False, indent=2)

            # 인덱스에 요약 추가
            summary = _session_summary(session)
            with self._index_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(summary, ensure_ascii=False) + "\n")

        logger.info("Session saved: %s", session.session_id)
        return session.session_id

    # ──────────────────────────────────────────────────────────────────────────
    # 로드
    # ──────────────────────────────────────────────────────────────────────────

    def load(self, session_id: str) -> BacktestSession | None:
        """
        session_id에 해당하는 전체 BacktestSession 을 반환한다.

        Returns:
            BacktestSession or None (파일 없음 / 파싱 실패)
        """
        with self._lock:
            session_path = self._dir / f"{session_id}.json"
            if not session_path.exists():
                logger.warning("Session file not found: %s", session_id)
                return None
            try:
                with session_path.open("r", encoding="utf-8") as f:
                    data: dict[str, Any] = json.load(f)
                return BacktestSession.from_dict(data)
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                logger.error("Failed to load session %s: %s", session_id, exc)
                return None

    # ──────────────────────────────────────────────────────────────────────────
    # 목록
    # ──────────────────────────────────────────────────────────────────────────

    def list_sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        """
        인덱스에서 세션 요약 목록을 반환한다 (최신순).

        중복 session_id 가 있을 경우 가장 마지막(최신) 항목을 사용한다.
        삭제된 세션은 실제 파일 존재 여부로 걸러낸다.

        Returns:
            list of summary dicts, most recent first, up to `limit` items.
        """
        with self._lock:
            summaries = self._read_index()

        # 중복 제거: 같은 id 라면 나중에 기록된 것 우선 (OrderedDict 대신 plain dict)
        seen: dict[str, dict[str, Any]] = {}
        for s in summaries:
            seen[s.get("id", "")] = s

        # 파일이 실제로 존재하는 항목만 (삭제된 세션 제외)
        valid = [
            v for v in seen.values()
            if (self._dir / f"{v.get('id', '')}.json").exists()
        ]

        # timestamp 내림차순 정렬 후 limit 적용
        valid.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return valid[:limit]

    # ──────────────────────────────────────────────────────────────────────────
    # 삭제
    # ──────────────────────────────────────────────────────────────────────────

    def delete(self, session_id: str) -> bool:
        """
        세션 파일을 삭제하고 인덱스에서도 제거한다.

        Returns:
            True if the session existed and was deleted, False otherwise.
        """
        with self._lock:
            session_path = self._dir / f"{session_id}.json"
            if not session_path.exists():
                return False

            try:
                session_path.unlink()
            except OSError as exc:
                logger.error("Failed to delete session file %s: %s", session_id, exc)
                return False

            # 인덱스에서 해당 session_id 줄 제거
            self._remove_from_index(session_id)

        logger.info("Session deleted: %s", session_id)
        return True

    # ──────────────────────────────────────────────────────────────────────────
    # CSV 내보내기
    # ──────────────────────────────────────────────────────────────────────────

    def export_csv(self, session_id: str, out_path: str) -> bool:
        """
        equity_curve / equity_dates 와 trades 를 두 개의 CSV 파일로 내보낸다.

        생성 파일:
            {out_path}_equity.csv   ← date, portfolio_value
            {out_path}_trades.csv   ← 거래 내역

        Returns:
            True on success, False if session not found.
        """
        import csv

        session = self.load(session_id)
        if session is None:
            logger.warning("export_csv: session not found %s", session_id)
            return False

        try:
            # equity curve — schema stores dates and values as parallel lists
            equity_path = f"{out_path}_equity.csv"
            dates  = session.equity_dates
            values = session.equity_curve
            with open(equity_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["date", "portfolio_value"])
                for d, v in zip(dates, values):
                    writer.writerow([d, v])

            # trades
            trades_path = f"{out_path}_trades.csv"
            with open(trades_path, "w", newline="", encoding="utf-8") as f:
                if session.trades:
                    fieldnames = list(session.trades[0].keys())
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(session.trades)
                else:
                    f.write("no trades\n")

            logger.info(
                "Exported session %s → %s, %s",
                session_id, equity_path, trades_path,
            )
            return True

        except OSError as exc:
            logger.error("export_csv failed for %s: %s", session_id, exc)
            return False

    # ──────────────────────────────────────────────────────────────────────────
    # 내부 헬퍼
    # ──────────────────────────────────────────────────────────────────────────

    def _read_index(self) -> list[dict[str, Any]]:
        """sessions.jsonl 을 읽어 list[dict] 로 반환."""
        summaries: list[dict[str, Any]] = []
        if not self._index_path.exists():
            return summaries
        with self._index_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    summaries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return summaries

    def _remove_from_index(self, session_id: str) -> None:
        """인덱스에서 특정 session_id 를 가진 모든 줄을 제거한 뒤 다시 쓴다."""
        summaries = self._read_index()
        kept = [s for s in summaries if s.get("id") != session_id]
        with self._index_path.open("w", encoding="utf-8") as f:
            for s in kept:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
