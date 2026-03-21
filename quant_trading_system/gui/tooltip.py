# gui/tooltip.py — 재사용 가능한 툴팁 시스템
"""
마우스를 올리면 설명 팝업이 나타나는 툴팁 헬퍼.

사용 예:
    from gui.tooltip import Tooltip, add_tooltip

    # 단독 사용
    Tooltip(some_widget, "이 값은 학습 데이터의 기간입니다.\n초보자 추천: 2년")

    # 헬퍼 함수 사용
    add_tooltip(entry_widget, "학습률: 모델이 학습할 때 한 번에\n얼마나 크게 가중치를 바꿀지 결정합니다.")
"""
from __future__ import annotations
import tkinter as tk


class Tooltip:
    """
    위젯에 마우스 오버 시 나타나는 다크-테마 툴팁.

    Parameters
    ----------
    widget   : 툴팁을 달 Tkinter 위젯
    text     : 표시할 설명 (줄바꿈 허용)
    delay_ms : 툴팁이 나타날 때까지 대기 시간 (ms)
    max_width: 텍스트 자동 줄바꿈 폭 (px)
    """

    _BG      = "#2a2a3e"
    _FG      = "#cdd6f4"
    _BORDER  = "#585b70"
    _FONT    = ("맑은 고딕", 9)
    _PAD     = 8

    def __init__(
        self,
        widget: tk.Widget,
        text: str,
        delay_ms: int = 600,
        max_width: int = 280,
    ):
        self.widget   = widget
        self.text     = text
        self.delay_ms = delay_ms
        self.max_width = max_width

        self._tip_win: tk.Toplevel | None = None
        self._after_id: str | None = None

        widget.bind("<Enter>",   self._on_enter,  add="+")
        widget.bind("<Leave>",   self._on_leave,  add="+")
        widget.bind("<Destroy>", self._on_destroy, add="+")

    # ── 이벤트 핸들러 ──────────────────────────────────────────────────────

    def _on_enter(self, _event=None):
        self._cancel()
        self._after_id = self.widget.after(self.delay_ms, self._show)

    def _on_leave(self, _event=None):
        self._cancel()
        self._hide()

    def _on_destroy(self, _event=None):
        self._cancel()
        self._hide()

    # ── 표시 / 숨김 ────────────────────────────────────────────────────────

    def _show(self):
        if self._tip_win or not self.text:
            return

        # 위젯의 화면 좌표 계산
        try:
            x = self.widget.winfo_rootx() + 20
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        except Exception:
            return

        self._tip_win = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)   # 제목 표시줄 제거
        tw.wm_geometry(f"+{x}+{y}")
        tw.configure(bg=self._BORDER)

        inner = tk.Frame(tw, bg=self._BG, padx=self._PAD, pady=self._PAD // 2)
        inner.pack(padx=1, pady=1)

        lbl = tk.Label(
            inner,
            text=self.text,
            bg=self._BG,
            fg=self._FG,
            font=self._FONT,
            justify="left",
            wraplength=self.max_width,
        )
        lbl.pack()

        # 화면 경계 보정 (화면 오른쪽 밖으로 나가지 않게)
        tw.update_idletasks()
        sw = tw.winfo_screenwidth()
        sh = tw.winfo_screenheight()
        tw_w = tw.winfo_width()
        tw_h = tw.winfo_height()

        if x + tw_w > sw - 10:
            x = sw - tw_w - 10
        if y + tw_h > sh - 10:
            y = self.widget.winfo_rooty() - tw_h - 4

        tw.wm_geometry(f"+{x}+{y}")

    def _hide(self):
        if self._tip_win:
            try:
                self._tip_win.destroy()
            except Exception:
                pass
            self._tip_win = None

    def _cancel(self):
        if self._after_id:
            try:
                self.widget.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None

    def update_text(self, new_text: str):
        """실행 중 툴팁 텍스트 변경"""
        self.text = new_text


# ── 편의 함수 ──────────────────────────────────────────────────────────────

def add_tooltip(widget: tk.Widget, text: str, **kwargs) -> Tooltip:
    """
    위젯에 툴팁 한 줄로 추가.

    Returns the Tooltip instance (if you need to update text later).
    """
    return Tooltip(widget, text, **kwargs)


def add_help_label(
    parent: tk.Widget,
    text: str,
    fg: str = "#9399b2",
    font: tuple = ("맑은 고딕", 8),
    padx: int = 4,
    pady: int = 0,
) -> tk.Label:
    """
    위젯 아래 작은 회색 설명 텍스트를 추가.

    사용 예:
        add_help_label(frame, "초보자 추천값: 20  |  너무 작으면 결과가 불안정합니다")
    """
    lbl = tk.Label(
        parent,
        text=text,
        fg=fg,
        bg=parent.cget("bg") if hasattr(parent, "cget") else "#1e1e2e",
        font=font,
        anchor="w",
        justify="left",
    )
    lbl.pack(anchor="w", padx=padx, pady=pady)
    return lbl
