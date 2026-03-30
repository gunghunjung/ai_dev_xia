"""
portfolio/manager.py
====================
포트폴리오 저장·불러오기·버전 관리.

구조:
    portfolios_dir/
        index.json              ← 모든 포트폴리오 메타 인덱스 (이름·날짜·id)
        {portfolio_id}/
            v{N}.json           ← 버전별 스냅샷 (append-only)
            latest.json         ← 최신 버전 심볼릭 복사본

기능:
    create / load / save / rename / duplicate / delete
    list_all / version_history / restore_version
    export_json / import_json
    autosave (덮어쓰기 방지 — 항상 새 버전 생성)
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("quant.portfolio.manager")

_INDEX_FILE = "index.json"
_LATEST     = "latest.json"


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _new_id() -> str:
    return str(uuid.uuid4())[:8]


# ─────────────────────────────────────────────────────────────────────────────
# Schema
# ─────────────────────────────────────────────────────────────────────────────

class Portfolio:
    """
    포트폴리오 데이터 컨테이너.

    Attributes
    ----------
    id : str            8자리 UUID prefix
    name : str          사용자 지정 이름
    version : int       버전 번호 (1부터 시작, 저장 시 자동 증가)
    created_at : str    최초 생성 ISO timestamp
    updated_at : str    최근 저장 ISO timestamp
    symbols : list[str] 편입 종목 (티커 목록)
    weights : dict      {ticker: weight_float}  (합 = 1.0)
    strategy : str      전략 이름 (예: "risk_parity")
    notes : str         자유 메모
    tags : list[str]    사용자 태그
    risk_profile : str  "aggressive" | "moderate" | "conservative"
    prediction_settings : dict  예측 탭 설정 스냅샷
    scenario_settings : dict    시나리오 설정 스냅샷
    preferred_indicators : list[str]  차트 기본 인디케이터
    """

    def __init__(
        self,
        name: str = "새 포트폴리오",
        symbols: list[str] | None = None,
        weights: dict[str, float] | None = None,
        strategy: str = "equal_weight",
        notes: str = "",
        tags: list[str] | None = None,
        risk_profile: str = "moderate",
        prediction_settings: dict | None = None,
        scenario_settings: dict | None = None,
        preferred_indicators: list[str] | None = None,
        portfolio_id: str = "",
        version: int = 1,
        created_at: str = "",
        updated_at: str = "",
    ) -> None:
        self.id          = portfolio_id or _new_id()
        self.name        = name
        self.version     = version
        self.created_at  = created_at or _now_iso()
        self.updated_at  = updated_at or _now_iso()
        self.symbols     = symbols or []
        self.weights     = weights or {}
        self.strategy    = strategy
        self.notes       = notes
        self.tags        = tags or []
        self.risk_profile = risk_profile
        self.prediction_settings  = prediction_settings or {}
        self.scenario_settings    = scenario_settings or {}
        self.preferred_indicators = preferred_indicators or ["MA20", "BB", "Volume"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id":                   self.id,
            "name":                 self.name,
            "version":              self.version,
            "created_at":           self.created_at,
            "updated_at":           self.updated_at,
            "symbols":              self.symbols,
            "weights":              self.weights,
            "strategy":             self.strategy,
            "notes":                self.notes,
            "tags":                 self.tags,
            "risk_profile":         self.risk_profile,
            "prediction_settings":  self.prediction_settings,
            "scenario_settings":    self.scenario_settings,
            "preferred_indicators": self.preferred_indicators,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Portfolio":
        return cls(
            name                 = data.get("name", "포트폴리오"),
            symbols              = list(data.get("symbols", [])),
            weights              = dict(data.get("weights", {})),
            strategy             = data.get("strategy", "equal_weight"),
            notes                = data.get("notes", ""),
            tags                 = list(data.get("tags", [])),
            risk_profile         = data.get("risk_profile", "moderate"),
            prediction_settings  = dict(data.get("prediction_settings", {})),
            scenario_settings    = dict(data.get("scenario_settings", {})),
            preferred_indicators = list(data.get("preferred_indicators",
                                                  ["MA20", "BB", "Volume"])),
            portfolio_id         = data.get("id", _new_id()),
            version              = int(data.get("version", 1)),
            created_at           = data.get("created_at", _now_iso()),
            updated_at           = data.get("updated_at", _now_iso()),
        )

    def meta_dict(self) -> dict[str, Any]:
        """인덱스에 저장하는 경량 메타만 반환."""
        return {
            "id":         self.id,
            "name":       self.name,
            "version":    self.version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "n_symbols":  len(self.symbols),
            "strategy":   self.strategy,
            "tags":       self.tags,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Manager
# ─────────────────────────────────────────────────────────────────────────────

class PortfolioManager:
    """
    포트폴리오 영속성 관리자.

    Usage
    -----
    pm = PortfolioManager("outputs/portfolios")

    # 새 포트폴리오 생성
    p = pm.create("내 첫 포트폴리오", symbols=["005930", "000660"])

    # 저장 (버전 자동 증가)
    pm.save(p)

    # 불러오기
    p2 = pm.load(p.id)

    # 목록
    for meta in pm.list_all():
        print(meta["name"], meta["version"])
    """

    def __init__(self, portfolios_dir: str) -> None:
        self._root = Path(portfolios_dir)
        self._root.mkdir(parents=True, exist_ok=True)
        self._index_path = self._root / _INDEX_FILE
        self._lock = threading.Lock()

        if not self._index_path.exists():
            self._write_index([])

    # ──────────────────────────────────────────────────────────────────────────
    # CRUD
    # ──────────────────────────────────────────────────────────────────────────

    def create(self, name: str = "새 포트폴리오",
               symbols: list[str] | None = None,
               strategy: str = "equal_weight") -> Portfolio:
        """새 포트폴리오를 만들고 즉시 저장한다."""
        p = Portfolio(name=name, symbols=symbols or [], strategy=strategy)
        self.save(p)
        logger.info("Portfolio created: %s (%s)", p.name, p.id)
        return p

    def save(self, portfolio: Portfolio) -> int:
        """
        포트폴리오를 새 버전으로 저장한다.

        반환값: 저장된 버전 번호.
        """
        with self._lock:
            p_dir = self._root / portfolio.id
            p_dir.mkdir(exist_ok=True)

            # 다음 버전 번호 결정
            existing = sorted(p_dir.glob("v*.json"))
            next_ver = len(existing) + 1
            portfolio.version    = next_ver
            portfolio.updated_at = _now_iso()

            # 버전 스냅샷 저장
            ver_path = p_dir / f"v{next_ver}.json"
            with ver_path.open("w", encoding="utf-8") as f:
                json.dump(portfolio.to_dict(), f, ensure_ascii=False, indent=2)

            # latest.json 갱신
            latest_path = p_dir / _LATEST
            shutil.copy2(ver_path, latest_path)

            # 인덱스 갱신
            self._upsert_index(portfolio.meta_dict())

        logger.info("Portfolio saved: %s v%d", portfolio.name, portfolio.version)
        return next_ver

    def load(self, portfolio_id: str, version: int = -1) -> Portfolio | None:
        """
        포트폴리오를 불러온다.

        Parameters
        ----------
        portfolio_id : str
        version : int
            -1 (기본) → latest.json 사용.
            N          → v{N}.json 사용.
        """
        with self._lock:
            p_dir = self._root / portfolio_id
            if not p_dir.exists():
                logger.warning("Portfolio dir not found: %s", portfolio_id)
                return None

            if version == -1:
                path = p_dir / _LATEST
            else:
                path = p_dir / f"v{version}.json"

            if not path.exists():
                logger.warning("Portfolio file not found: %s", path)
                return None

            try:
                with path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                return Portfolio.from_dict(data)
            except (json.JSONDecodeError, KeyError) as exc:
                logger.error("Failed to load portfolio %s: %s", portfolio_id, exc)
                return None

    def delete(self, portfolio_id: str) -> bool:
        """포트폴리오 디렉터리를 삭제하고 인덱스에서 제거한다."""
        with self._lock:
            p_dir = self._root / portfolio_id
            if not p_dir.exists():
                return False
            try:
                shutil.rmtree(p_dir)
            except OSError as exc:
                logger.error("Failed to delete portfolio %s: %s", portfolio_id, exc)
                return False
            self._remove_from_index(portfolio_id)
        logger.info("Portfolio deleted: %s", portfolio_id)
        return True

    def rename(self, portfolio_id: str, new_name: str) -> bool:
        """이름 변경 (새 버전 저장)."""
        p = self.load(portfolio_id)
        if p is None:
            return False
        p.name = new_name
        self.save(p)
        return True

    def duplicate(self, portfolio_id: str,
                  new_name: str = "") -> Portfolio | None:
        """복제본 생성 (새 id 부여, 버전 1부터 시작)."""
        original = self.load(portfolio_id)
        if original is None:
            return None
        copy = Portfolio(
            name       = new_name or f"{original.name} (복사)",
            symbols    = list(original.symbols),
            weights    = dict(original.weights),
            strategy   = original.strategy,
            notes      = original.notes,
            tags       = list(original.tags),
            risk_profile         = original.risk_profile,
            prediction_settings  = dict(original.prediction_settings),
            scenario_settings    = dict(original.scenario_settings),
            preferred_indicators = list(original.preferred_indicators),
        )
        self.save(copy)
        logger.info("Portfolio duplicated: %s → %s", portfolio_id, copy.id)
        return copy

    # ──────────────────────────────────────────────────────────────────────────
    # 목록 / 이력
    # ──────────────────────────────────────────────────────────────────────────

    def list_all(self) -> list[dict[str, Any]]:
        """전체 포트폴리오 메타 목록 (최신순)."""
        index = self._read_index()
        # 실제 디렉터리 존재 여부 확인
        valid = [m for m in index
                 if (self._root / m.get("id", "")).exists()]
        valid.sort(key=lambda m: m.get("updated_at", ""), reverse=True)
        return valid

    def version_history(self, portfolio_id: str) -> list[dict[str, Any]]:
        """버전 이력 목록 반환 (버전 1부터 최신까지)."""
        p_dir = self._root / portfolio_id
        if not p_dir.exists():
            return []
        versions = []
        for vf in sorted(p_dir.glob("v*.json")):
            try:
                with vf.open("r", encoding="utf-8") as f:
                    d = json.load(f)
                versions.append({
                    "version":    d.get("version"),
                    "updated_at": d.get("updated_at"),
                    "n_symbols":  len(d.get("symbols", [])),
                    "file":       str(vf),
                })
            except Exception:
                continue
        return versions

    def restore_version(self, portfolio_id: str, version: int) -> Portfolio | None:
        """특정 버전을 불러와 최신 버전으로 저장한다 (롤백)."""
        p = self.load(portfolio_id, version=version)
        if p is None:
            return None
        p.notes = (f"[v{version}에서 복원, {_now_iso()[:10]}]\n" + p.notes)
        self.save(p)
        logger.info("Portfolio %s restored to v%d (new version=%d)",
                    portfolio_id, version, p.version)
        return p

    # ──────────────────────────────────────────────────────────────────────────
    # 내보내기 / 가져오기
    # ──────────────────────────────────────────────────────────────────────────

    def export_json(self, portfolio_id: str, out_path: str) -> bool:
        p = self.load(portfolio_id)
        if p is None:
            return False
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(p.to_dict(), f, ensure_ascii=False, indent=2)
            logger.info("Portfolio exported: %s → %s", portfolio_id, out_path)
            return True
        except OSError as exc:
            logger.error("export_json failed: %s", exc)
            return False

    def import_json(self, json_path: str) -> Portfolio | None:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # 새 id 부여 (충돌 방지)
            data["id"] = _new_id()
            data["version"] = 1
            data["created_at"] = _now_iso()
            data["updated_at"] = _now_iso()
            p = Portfolio.from_dict(data)
            self.save(p)
            logger.info("Portfolio imported from %s → %s", json_path, p.id)
            return p
        except (OSError, json.JSONDecodeError, KeyError) as exc:
            logger.error("import_json failed: %s", exc)
            return None

    # ──────────────────────────────────────────────────────────────────────────
    # 인덱스 헬퍼
    # ──────────────────────────────────────────────────────────────────────────

    def _read_index(self) -> list[dict[str, Any]]:
        try:
            with self._index_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return []

    def _write_index(self, items: list[dict[str, Any]]) -> None:
        with self._index_path.open("w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)

    def _upsert_index(self, meta: dict[str, Any]) -> None:
        items = self._read_index()
        # 기존 항목 교체 또는 추가
        replaced = False
        for i, item in enumerate(items):
            if item.get("id") == meta["id"]:
                items[i] = meta
                replaced = True
                break
        if not replaced:
            items.append(meta)
        self._write_index(items)

    def _remove_from_index(self, portfolio_id: str) -> None:
        items = self._read_index()
        items = [m for m in items if m.get("id") != portfolio_id]
        self._write_index(items)
