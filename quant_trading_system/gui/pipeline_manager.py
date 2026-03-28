# gui/pipeline_manager.py — 단계별 파이프라인 상태 관리
from __future__ import annotations
import json, os
from datetime import datetime
import logging

logger = logging.getLogger("quant.gui.pipeline")

# (key, 표시명, 탭 인덱스, 아이콘)
STAGE_DEFS: list[tuple[str, str, int, str]] = [
    ("data",     "데이터",    0, "📊"),
    ("train",    "학습",      1, "🧠"),
    ("backtest", "백테스트",  2, "📈"),
    ("predict",  "예측",      3, "🔮"),
]

# 단계 완료 시 하위 단계를 stale 처리
_INVALIDATES: dict[str, list[str]] = {
    "data":     ["train", "backtest", "predict"],
    "train":    ["backtest", "predict"],
    "backtest": [],
    "predict":  [],
}

# 상태별 색상 (bg, fg)
STATUS_STYLE: dict[str, tuple[str, str]] = {
    "pending": ("#2a2a3e", "#6c7086"),   # 회색 — 미실행
    "fresh":   ("#1a3a22", "#a6e3a1"),   # 녹색 — 최신
    "stale":   ("#3a2800", "#f9e2af"),   # 주황 — 갱신 필요
    "running": ("#0f1a3a", "#89b4fa"),   # 파랑 — 실행 중
}
STATUS_ICON: dict[str, str] = {
    "pending": "○",
    "fresh":   "✅",
    "stale":   "⚠",
    "running": "⚡",
}


class PipelineManager:
    """
    4단계 파이프라인 상태 추적:
      데이터 → 학습 → 백테스트 → 예측

    사용법:
        mgr = PipelineManager("outputs/pipeline.json", on_update=refresh_ui)
        mgr.mark_running("data")   # 작업 시작 시
        mgr.mark_done("data")      # 작업 완료 시 (하위 단계 자동 stale)
    """

    def __init__(self, status_file: str, on_update=None):
        self._file = status_file
        self._on_update = on_update
        self._data: dict = self._load()

    # ──────────────────────────────────────────────
    # 공개 API
    # ──────────────────────────────────────────────

    def mark_running(self, stage: str):
        """작업 시작 — 해당 단계를 '실행 중' 표시"""
        self._set(stage, "running")

    def mark_done(self, stage: str):
        """작업 완료 — 해당 단계 fresh, 하위 단계 stale"""
        self._set(stage, "fresh", save=False)
        for downstream in _INVALIDATES.get(stage, []):
            cur = self._data.get(downstream, {}).get("status", "pending")
            if cur in ("fresh",):       # fresh → stale (pending은 그대로)
                self._set(downstream, "stale", save=False)
        self._save()
        self._notify()

    def get_status(self, stage: str) -> str:
        return self._data.get(stage, {}).get("status", "pending")

    def get_ts(self, stage: str) -> str:
        ts = self._data.get(stage, {}).get("ts", "")
        if not ts:
            return "미실행"
        try:
            return datetime.fromisoformat(ts).strftime("%m/%d %H:%M")
        except Exception:
            return ts

    def get_stale_stages(self) -> list[str]:
        """stale 상태인 단계 표시명 목록"""
        return [label for key, label, _, _ in STAGE_DEFS
                if self.get_status(key) == "stale"]

    def any_stale(self) -> bool:
        return bool(self.get_stale_stages())

    # ──────────────────────────────────────────────
    # 내부
    # ──────────────────────────────────────────────

    def _set(self, stage: str, status: str, save: bool = True):
        entry = self._data.setdefault(stage, {})
        entry["status"] = status
        if status in ("fresh", "running"):
            entry["ts"] = datetime.now().isoformat()
        if save:
            self._save()
            self._notify()

    def _notify(self):
        if self._on_update:
            try:
                self._on_update()
            except Exception as e:
                logger.debug(f"pipeline on_update error: {e}")

    def _load(self) -> dict:
        try:
            if os.path.exists(self._file):
                with open(self._file, encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

    def _save(self):
        try:
            os.makedirs(os.path.dirname(self._file), exist_ok=True)
            with open(self._file, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"파이프라인 저장 실패: {e}")
