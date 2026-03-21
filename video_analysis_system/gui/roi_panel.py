"""
gui/roi_panel.py — 독립형 ROI 관리 패널

panels.py 의 ROIPanel 과 동일한 역할이지만:
  - 직접 입력 좌표를 원본 픽셀 기준으로 처리
  - ROI 선택/이름변경/삭제를 완전히 지원
  - CoordinateTransform 인식: 입력 필드 옆에 단위 표시

외부 인터페이스:
  panel.add_roi(roi_id, original_rect)       — 코드에서 ROI 추가
  panel.get_roi_configs() → list[dict]        — 현재 ROI 목록 조회
  panel.clear()                               — 전체 삭제
  panel.on_rois_changed = callback(list)      — 변경 알림 콜백
  panel.draw_mode_callback = callback(bool)   — 마우스 그리기 토글 콜백
"""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, simpledialog, ttk
from typing import Callable, Dict, List, Optional, Tuple


_KO = ("맑은 고딕", 9)
_KO_BOLD = ("맑은 고딕", 9, "bold")


class ROIPanel(ttk.Frame):
    """
    독립형 ROI 관리 패널.

    ROI 는 항상 원본 프레임 픽셀 좌표로 저장된다.
    화면 표시 스케일 변환은 VideoCanvas 와 CoordinateTransform 이 담당한다.
    """

    def __init__(self, master, on_rois_changed: Optional[Callable] = None, **kwargs):
        super().__init__(master, padding=8, **kwargs)

        self.on_rois_changed: Optional[Callable] = on_rois_changed
        self.draw_mode_callback: Optional[Callable] = None

        # roi_id → {"roi_id", "bbox": (x,y,w,h), "label": str}
        self._rois: Dict[str, dict] = {}

        self._build_ui()

    # ── UI 구성 ───────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        # 타이틀
        ttk.Label(self, text="□  ROI 관리", font=("맑은 고딕", 10, "bold")).pack(
            anchor="w", pady=(0, 6))

        # 마우스 그리기 버튼
        draw_frame = ttk.Frame(self)
        draw_frame.pack(fill="x", pady=(0, 4))
        self._draw_var = tk.BooleanVar(value=False)
        self._draw_btn = ttk.Checkbutton(
            draw_frame,
            text="✏  마우스로 ROI 그리기",
            variable=self._draw_var,
            command=self._on_draw_toggle,
        )
        self._draw_btn.pack(side="left")

        # ROI 목록 트리뷰
        list_frame = ttk.LabelFrame(self, text="ROI 목록", padding=4)
        list_frame.pack(fill="both", expand=True, pady=(0, 6))

        cols = ("roi_id", "bbox", "label")
        self._tree = ttk.Treeview(list_frame, columns=cols, show="headings",
                                  selectmode="browse", height=8)
        self._tree.heading("roi_id", text="ID")
        self._tree.heading("bbox",   text="원본 좌표 (x,y,w,h)")
        self._tree.heading("label",  text="이름")
        self._tree.column("roi_id", width=70,  anchor="center")
        self._tree.column("bbox",   width=160, anchor="center")
        self._tree.column("label",  width=90,  anchor="w")

        vsb = ttk.Scrollbar(list_frame, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=vsb.set)
        self._tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        self._tree.bind("<Double-1>", self._on_rename)

        # 버튼 행
        btn_row = ttk.Frame(self)
        btn_row.pack(fill="x", pady=(0, 6))
        ttk.Button(btn_row, text="이름 변경",  command=self._on_rename).pack(side="left", padx=(0, 3))
        ttk.Button(btn_row, text="삭제",       command=self._on_delete).pack(side="left", padx=(0, 3))
        ttk.Button(btn_row, text="전체 삭제",  command=self._on_clear).pack(side="left")

        # 직접 입력
        manual = ttk.LabelFrame(self, text="좌표 직접 입력  (원본 픽셀)", padding=6)
        manual.pack(fill="x")

        row1 = ttk.Frame(manual)
        row1.pack(fill="x", pady=(0, 3))
        for lbl, w in [("x:", 5), ("y:", 5), ("w:", 5), ("h:", 5)]:
            ttk.Label(row1, text=lbl).pack(side="left")
            v = tk.IntVar(value=0)
            ttk.Entry(row1, textvariable=v, width=w).pack(side="left", padx=(0, 6))
            setattr(self, f"_manual_{lbl[0]}", v)

        row2 = ttk.Frame(manual)
        row2.pack(fill="x")
        self._manual_label = tk.StringVar()
        ttk.Label(row2, text="이름:").pack(side="left")
        ttk.Entry(row2, textvariable=self._manual_label, width=12).pack(side="left", padx=(2, 8))
        ttk.Button(row2, text="추가", command=self._on_manual_add,
                   style="Accent.TButton").pack(side="left")

    # ── 공용 API ─────────────────────────────────────────────────────────

    def add_roi(self, roi_id: str,
                original_rect: Tuple[int, int, int, int],
                label: str = "") -> None:
        """원본 픽셀 좌표 기준 ROI 를 추가한다."""
        entry = {"roi_id": roi_id, "bbox": original_rect, "label": label or roi_id}
        self._rois[roi_id] = entry
        self._tree.insert("", "end", iid=roi_id,
                          values=(roi_id,
                                  f"{original_rect[0]},{original_rect[1]},{original_rect[2]},{original_rect[3]}",
                                  entry["label"]))
        self._notify()

    def remove_roi(self, roi_id: str) -> None:
        if roi_id in self._rois:
            del self._rois[roi_id]
            if self._tree.exists(roi_id):
                self._tree.delete(roi_id)
            self._notify()

    def clear(self) -> None:
        self._rois.clear()
        for item in self._tree.get_children():
            self._tree.delete(item)
        self._notify()

    def get_roi_configs(self) -> List[dict]:
        return list(self._rois.values())

    # ── 내부 핸들러 ──────────────────────────────────────────────────────

    def _notify(self) -> None:
        if self.on_rois_changed:
            self.on_rois_changed(list(self._rois.values()))

    def _on_draw_toggle(self) -> None:
        if self.draw_mode_callback:
            self.draw_mode_callback(self._draw_var.get())

    def _selected_id(self) -> Optional[str]:
        sel = self._tree.selection()
        return sel[0] if sel else None

    def _on_rename(self, _event=None) -> None:
        roi_id = self._selected_id()
        if not roi_id:
            messagebox.showinfo("선택 필요", "이름을 바꿀 ROI 를 먼저 선택하세요.")
            return
        new_name = simpledialog.askstring(
            "이름 변경",
            f"'{roi_id}' 의 새 이름:",
            initialvalue=self._rois[roi_id]["label"],
            parent=self,
        )
        if new_name:
            self._rois[roi_id]["label"] = new_name
            self._tree.set(roi_id, "label", new_name)
            self._notify()

    def _on_delete(self) -> None:
        roi_id = self._selected_id()
        if not roi_id:
            messagebox.showinfo("선택 필요", "삭제할 ROI 를 먼저 선택하세요.")
            return
        self.remove_roi(roi_id)

    def _on_clear(self) -> None:
        if not self._rois:
            return
        if messagebox.askyesno("전체 삭제", "모든 ROI 를 삭제하시겠습니까?"):
            self.clear()

    def _on_manual_add(self) -> None:
        x = self._manual_x.get()
        y = self._manual_y.get()
        w = self._manual_w.get()
        h = self._manual_h.get()
        if w <= 0 or h <= 0:
            messagebox.showwarning("잘못된 좌표", "너비(w)와 높이(h) 는 0보다 커야 합니다.")
            return
        label = self._manual_label.get().strip()
        roi_id = label if label else f"roi_{len(self._rois)}"
        self.add_roi(roi_id, (x, y, w, h), label)
