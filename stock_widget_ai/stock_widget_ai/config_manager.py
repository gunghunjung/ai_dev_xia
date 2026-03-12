"""
ConfigManager — settings.json 자동 저장/복원
"""
from __future__ import annotations
import json
import os
import shutil
from datetime import datetime
from typing import Any, Optional

from .state_schema import AppState, SCHEMA_VERSION
from .logger_config import get_logger

log = get_logger("config")

_DEFAULT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "settings.json",
)


class ConfigManager:
    def __init__(self, path: str = _DEFAULT_PATH) -> None:
        self._path = path
        self._state: AppState = AppState()
        self.load()

    # ── 공개 API ───────────────────────────────────────────────────
    @property
    def state(self) -> AppState:
        return self._state

    def get(self, key: str, default: Any = None) -> Any:
        """점(.) 구분 경로 지원: "model.epochs" """
        parts = key.split(".")
        obj: Any = self._state
        for p in parts:
            if hasattr(obj, p):
                obj = getattr(obj, p)
            else:
                return default
        return obj

    def set(self, key: str, value: Any) -> None:
        parts = key.split(".")
        obj: Any = self._state
        for p in parts[:-1]:
            if hasattr(obj, p):
                obj = getattr(obj, p)
            else:
                return
        if hasattr(obj, parts[-1]):
            setattr(obj, parts[-1], value)

    def load(self) -> None:
        if not os.path.exists(self._path):
            log.info(f"settings.json 없음 → 기본값 사용: {self._path}")
            self._state = AppState()
            self.safe_save()
            return
        try:
            with open(self._path, encoding="utf-8") as f:
                raw = json.load(f)
            self.validate_state(raw)
            self._state = AppState.from_dict(raw)
            log.info(f"설정 로드: {self._path}")
        except Exception as e:
            log.warning(f"설정 파일 손상 — 백업 후 기본값 복구: {e}")
            self.backup_if_corrupted()
            self._state = AppState()
            self.safe_save()

    def safe_save(self) -> None:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self._path)), exist_ok=True)
            tmp = self._path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._state.to_dict(), f, ensure_ascii=False, indent=2)
            os.replace(tmp, self._path)
            log.debug("설정 저장 완료")
        except Exception as e:
            log.error(f"설정 저장 실패: {e}")

    def validate_state(self, raw: dict) -> None:
        if not isinstance(raw, dict):
            raise ValueError("JSON root must be dict")
        ver = raw.get("version", 1)
        if ver > SCHEMA_VERSION:
            log.warning(f"settings.json version={ver} > schema={SCHEMA_VERSION}")

    def backup_if_corrupted(self) -> None:
        if os.path.exists(self._path):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            bk = self._path + f".bak_{ts}"
            shutil.copy2(self._path, bk)
            log.info(f"백업 생성: {bk}")

    # ── PyQt6 창 상태 동기화 ───────────────────────────────────────
    def update_from_window(self, window: Any) -> None:
        """
        Qt의 saveGeometry()를 base64로 직렬화해 저장.
        - 최대화 상태에서도 복원 크기를 정확히 보존
        - 멀티모니터, DPI 스케일링 모두 처리
        - fallback으로 width/height/x/y도 같이 저장
        """
        try:
            import base64
            raw_bytes = bytes(window.saveGeometry())            # QByteArray → bytes
            self._state.window.geometry_b64 = base64.b64encode(raw_bytes).decode("ascii")
            # fallback 값도 갱신 (JSON에서 직접 읽을 때 참고용)
            self._state.window.maximized = window.isMaximized()
            geo = window.normalGeometry() if window.isMaximized() else window.geometry()
            self._state.window.x      = geo.x()
            self._state.window.y      = geo.y()
            self._state.window.width  = geo.width()
            self._state.window.height = geo.height()
            log.debug(f"창 상태 저장: {geo.width()}×{geo.height()} @ ({geo.x()},{geo.y()}) "
                      f"maximized={self._state.window.maximized}")
        except Exception as e:
            log.debug(f"update_from_window 오류: {e}")

    def apply_to_window(self, window: Any) -> None:
        """
        저장된 geometry_b64 → restoreGeometry() 로 완벽 복원.
        b64 없으면 fallback으로 move()+resize() 사용.
        """
        try:
            from PyQt6.QtCore import QByteArray, QRect
            ws = self._state.window

            if ws.geometry_b64:
                import base64
                ba = QByteArray(base64.b64decode(ws.geometry_b64))
                ok = window.restoreGeometry(ba)
                log.debug(f"restoreGeometry: {'성공' if ok else '실패(fallback)'}")
                if ok:
                    return   # 성공 시 여기서 종료

            # fallback: 저장된 수치로 직접 설정
            screen = window.screen()
            if screen:
                avail = screen.availableGeometry()
                x = max(0, min(ws.x, avail.width()  - 100))
                y = max(0, min(ws.y, avail.height() - 100))
                w = max(800,  min(ws.width,  avail.width()))
                h = max(500,  min(ws.height, avail.height()))
            else:
                x, y, w, h = ws.x, ws.y, ws.width, ws.height

            window.move(x, y)
            window.resize(w, h)
            if ws.maximized:
                window.showMaximized()
            log.debug(f"fallback 복원: {w}×{h} @ ({x},{y})")
        except Exception as e:
            log.debug(f"apply_to_window 오류: {e}")
