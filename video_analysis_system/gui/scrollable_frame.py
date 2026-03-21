"""
gui/scrollable_frame.py — 재사용 가능한 마우스 휠 지원 스크롤 컨테이너

═══════════════════════════════════════════════════════════════
설계 전략
═══════════════════════════════════════════════════════════════
문제: Tkinter Frame 에는 스크롤 기능이 없다.
      Canvas + Frame 조합으로 가짜 스크롤을 구현해야 한다.

문제: 마우스 휠 이벤트는 포커스를 가진 위젯으로만 전달된다.
      중첩 프레임에서 Inner Frame 이 이벤트를 가로채면 Outer 가 스크롤 안 됨.

해결:
  1. ScrollableFrame 생성 시 _bind_mousewheel() 로 <Enter>/<Leave> 훅.
  2. 마우스가 위젯 위에 있을 때만 해당 ScrollableFrame 이 휠 이벤트를 처리.
  3. 전역 bind_all 대신 위젯별 bind 를 사용해 충돌 방지.
  4. Windows(<MouseWheel>), Linux(<Button-4/5>), Mac(<MouseWheel>) 모두 처리.
═══════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import platform
import tkinter as tk
from tkinter import ttk
from typing import Optional


_PLATFORM = platform.system()   # "Windows" | "Linux" | "Darwin"


def _scroll_delta(event: tk.Event) -> int:
    """플랫폼별 마우스 휠 스크롤 방향·크기 → 정수 반환."""
    if _PLATFORM == "Windows":
        return -1 if event.delta > 0 else 1
    elif _PLATFORM == "Darwin":
        return -1 if event.delta > 0 else 1
    else:   # Linux
        return -1 if event.num == 4 else 1


# ---------------------------------------------------------------------------
# ScrollableFrame
# ---------------------------------------------------------------------------

class ScrollableFrame(ttk.Frame):
    """
    세로 스크롤을 지원하는 컨테이너 위젯.

    self.inner 에 자식 위젯을 붙이면 된다::

        sf = ScrollableFrame(parent)
        sf.pack(fill="both", expand=True)
        ttk.Label(sf.inner, text="Hello").pack()

    마우스가 위젯 위에 있을 때만 스크롤 이벤트를 처리하므로
    중첩 사용 시에도 의도치 않은 스크롤이 발생하지 않는다.
    """

    def __init__(self, master, autohide_scrollbar: bool = True, **kwargs):
        super().__init__(master, **kwargs)

        # ── 내부 구조: Canvas + 수직 스크롤바 ────────────────────────────
        self._canvas = tk.Canvas(self, bg="#1e1e2e",
                                 highlightthickness=0, bd=0)
        self._vsb = ttk.Scrollbar(self, orient="vertical",
                                  command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=self._on_scroll_set)
        self._autohide = autohide_scrollbar
        self._sb_visible = False

        self._vsb.pack(side="right", fill="y")
        self._canvas.pack(side="left", fill="both", expand=True)

        # ── 내부 프레임 (실제 자식이 놓이는 곳) ──────────────────────────
        self.inner = ttk.Frame(self._canvas)
        self._window_id = self._canvas.create_window(
            (0, 0), window=self.inner, anchor="nw"
        )

        # ── 이벤트 바인딩 ─────────────────────────────────────────────────
        self.inner.bind("<Configure>", self._on_inner_configure)
        self._canvas.bind("<Configure>", self._on_canvas_configure)
        self._bind_mousewheel()

    # ── 스크롤바 자동 숨김 ─────────────────────────────────────────────

    def _on_scroll_set(self, lo: str, hi: str) -> None:
        self._vsb.set(lo, hi)
        if self._autohide:
            if float(lo) <= 0.0 and float(hi) >= 1.0:
                if self._sb_visible:
                    self._vsb.pack_forget()
                    self._sb_visible = False
            else:
                if not self._sb_visible:
                    self._vsb.pack(side="right", fill="y")
                    self._sb_visible = True

    # ── 크기 변경 ─────────────────────────────────────────────────────

    def _on_inner_configure(self, _event: tk.Event) -> None:
        """내부 프레임 크기가 바뀌면 스크롤 영역을 갱신한다."""
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _on_canvas_configure(self, event: tk.Event) -> None:
        """캔버스 너비에 맞게 내부 프레임을 늘린다."""
        self._canvas.itemconfig(self._window_id, width=event.width)

    # ── 마우스 휠 ─────────────────────────────────────────────────────

    def _bind_mousewheel(self) -> None:
        """마우스가 이 위젯 위에 있을 때만 스크롤 처리."""
        widgets = [self._canvas, self.inner, self]
        for w in widgets:
            w.bind("<Enter>", self._on_enter)
            w.bind("<Leave>", self._on_leave)

    def _on_enter(self, _event: tk.Event) -> None:
        if _PLATFORM in ("Windows", "Darwin"):
            self._canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        else:
            self._canvas.bind_all("<Button-4>", self._on_mousewheel)
            self._canvas.bind_all("<Button-5>", self._on_mousewheel)

    def _on_leave(self, _event: tk.Event) -> None:
        if _PLATFORM in ("Windows", "Darwin"):
            self._canvas.unbind_all("<MouseWheel>")
        else:
            self._canvas.unbind_all("<Button-4>")
            self._canvas.unbind_all("<Button-5>")

    def _on_mousewheel(self, event: tk.Event) -> None:
        delta = _scroll_delta(event)
        self._canvas.yview_scroll(delta, "units")

    # ── 공용 API ──────────────────────────────────────────────────────

    def scroll_to_top(self) -> None:
        self._canvas.yview_moveto(0.0)

    def scroll_to_bottom(self) -> None:
        self._canvas.yview_moveto(1.0)

    def scroll_to_widget(self, widget: tk.Widget) -> None:
        """특정 자식 위젯이 보이도록 스크롤한다."""
        self._canvas.update_idletasks()
        y = widget.winfo_y()
        total = self.inner.winfo_height()
        if total > 0:
            self._canvas.yview_moveto(y / total)


# ---------------------------------------------------------------------------
# HScrollableFrame  (가로 스크롤 변형)
# ---------------------------------------------------------------------------

class HScrollableFrame(ttk.Frame):
    """가로 스크롤을 지원하는 컨테이너 (타임라인, 표 등에 사용)."""

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)

        self._canvas = tk.Canvas(self, bg="#1e1e2e",
                                 highlightthickness=0, bd=0)
        self._hsb = ttk.Scrollbar(self, orient="horizontal",
                                  command=self._canvas.xview)
        self._canvas.configure(xscrollcommand=self._hsb.set)
        self._hsb.pack(side="bottom", fill="x")
        self._canvas.pack(fill="both", expand=True)

        self.inner = ttk.Frame(self._canvas)
        self._window_id = self._canvas.create_window(
            (0, 0), window=self.inner, anchor="nw"
        )
        self.inner.bind("<Configure>", self._on_inner_configure)
        self._canvas.bind("<Configure>", self._on_canvas_configure)

        self._canvas.bind("<Enter>", self._on_enter)
        self._canvas.bind("<Leave>", self._on_leave)

    def _on_inner_configure(self, _: tk.Event) -> None:
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _on_canvas_configure(self, event: tk.Event) -> None:
        self._canvas.itemconfig(self._window_id, height=event.height)

    def _on_enter(self, _: tk.Event) -> None:
        if _PLATFORM == "Windows":
            self._canvas.bind_all("<Shift-MouseWheel>", self._on_mousewheel)
        else:
            self._canvas.bind_all("<Button-6>", self._on_mousewheel)
            self._canvas.bind_all("<Button-7>", self._on_mousewheel)

    def _on_leave(self, _: tk.Event) -> None:
        if _PLATFORM == "Windows":
            self._canvas.unbind_all("<Shift-MouseWheel>")
        else:
            self._canvas.unbind_all("<Button-6>")
            self._canvas.unbind_all("<Button-7>")

    def _on_mousewheel(self, event: tk.Event) -> None:
        delta = _scroll_delta(event)
        self._canvas.xview_scroll(delta, "units")
