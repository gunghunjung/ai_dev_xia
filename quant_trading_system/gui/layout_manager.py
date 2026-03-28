# gui/layout_manager.py — PanedWindow sash 위치 자동 저장/복원
from __future__ import annotations
import json
import os
import tkinter as tk
import logging

logger = logging.getLogger("quant.gui.layout")


class LayoutManager:
    """
    모든 PanedWindow의 sash(분할선) 위치를 자동으로 저장하고 복원합니다.

    사용법:
        mgr = LayoutManager("outputs/layout.json")
        # UI 빌드 완료 후:
        root.after(400, mgr.attach(root))
    """

    def __init__(self, layout_file: str):
        self._file = layout_file
        self._data: dict = self._load()
        self._panes: dict[str, tk.PanedWindow] = {}
        self._pending_save: str | None = None   # after() 핸들 (디바운스)
        self._root: tk.Widget | None = None

    # ──────────────────────────────────────────────────────
    # 공개 API
    # ──────────────────────────────────────────────────────

    def attach(self, root: tk.Widget):
        """위젯 트리를 순회하여 PanedWindow를 등록하고 저장된 위치 복원"""
        self._root = root
        self._panes.clear()
        self._discover(root)
        logger.info(f"LayoutManager: {len(self._panes)}개 PanedWindow 등록")
        self._restore_all()

    def save_now(self):
        """즉시 저장 (종료 시 호출)"""
        self._save_all()

    # ──────────────────────────────────────────────────────
    # 내부 로직
    # ──────────────────────────────────────────────────────

    def _discover(self, widget: tk.Widget):
        """재귀적으로 위젯 트리를 탐색하여 PanedWindow 등록"""
        if isinstance(widget, tk.PanedWindow):
            path = str(widget)
            self._panes[path] = widget
            # 마우스 버튼을 놓을 때 저장 트리거 (드래그 종료)
            widget.bind("<ButtonRelease-1>",
                        lambda e, p=path: self._schedule_save(widget),
                        add="+")
        try:
            for child in widget.winfo_children():
                self._discover(child)
        except Exception:
            pass

    def _schedule_save(self, widget: tk.Widget):
        """300ms 디바운스 저장 (연속 드래그 중 과도한 I/O 방지)"""
        if self._pending_save:
            try:
                widget.after_cancel(self._pending_save)
            except Exception:
                pass
        self._pending_save = widget.after(300, self._save_all)

    def _save_all(self):
        """모든 등록된 PanedWindow의 sash 좌표를 JSON에 저장"""
        for path, pane in list(self._panes.items()):
            try:
                n_sashes = len(pane.panes()) - 1
                if n_sashes > 0:
                    coords = [list(pane.sash_coord(i)) for i in range(n_sashes)]
                    self._data[path] = coords
            except Exception:
                pass
        self._write()

    def _restore_all(self):
        """저장된 sash 좌표를 각 PanedWindow에 복원"""
        for path, pane in self._panes.items():
            if path not in self._data:
                continue
            saved = self._data[path]
            def _do_restore(p=pane, s=saved):
                for i, coord in enumerate(s):
                    try:
                        x, y = int(coord[0]), int(coord[1])
                        p.sash_place(i, x, y)
                    except Exception:
                        pass
            # 짧은 지연 후 복원 (창 렌더링 완료 대기)
            pane.after(50, _do_restore)

    def _load(self) -> dict:
        try:
            if os.path.exists(self._file):
                with open(self._file, encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

    def _write(self):
        try:
            os.makedirs(os.path.dirname(self._file), exist_ok=True)
            with open(self._file, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2, ensure_ascii=False)
            logger.debug("레이아웃 저장")
        except Exception as e:
            logger.error(f"레이아웃 저장 실패: {e}")
