"""
gui/failure_panel.py
====================
예측 실패 분석 패널 (자기교정 시스템)

철학: 성공 자랑보다 실패 해부가 더 중요하다.
     예측 실패 패턴을 시각화하고 개선 방향을 제시한다.
"""
from __future__ import annotations

import os
import sys
import threading
import tkinter as tk
from tkinter import ttk
from typing import Any

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from config import AppSettings
from history.logger import PredictionLogger
from history.failure_analyzer import FailureAnalyzer, FailureSummary, FAILURE_TYPES

# ── 색상 팔레트 ────────────────────────────────────────────────────────────────
BG        = "#181825"
PANEL_BG  = "#1e1e2e"
FG        = "#cdd6f4"
ACCENT    = "#cba6f7"
GREEN     = "#a6e3a1"
RED       = "#f38ba8"
YELLOW    = "#f9e2af"
ORANGE    = "#fab387"
CYAN      = "#89dceb"
TEAL      = "#94e2d5"
DIM       = "#6c7086"
SURFACE   = "#313244"
OVERLAY   = "#45475a"

_CHART_COLORS = {
    "MISSED_CRASH":     "#f38ba8",
    "MISSED_RALLY":     "#a6e3a1",
    "OVER_OPTIMISTIC":  "#fab387",
    "OVER_PESSIMISTIC": "#89dceb",
    "SIDEWAYS_MISREAD": "#f9e2af",
    "DIRECTION_HIT":    "#a6e3a1",
    "WITHIN_TOLERANCE": "#94e2d5",
}

_DIR_COLORS = {
    "UP":      GREEN,
    "DOWN":    RED,
    "NEUTRAL": YELLOW,
}

_CONF_COLORS = {
    "HIGH":   GREEN,
    "MEDIUM": YELLOW,
    "LOW":    RED,
}


class FailurePanel:
    """예측 실패 분석 패널."""

    def __init__(self, parent: tk.Widget, settings: AppSettings) -> None:
        self.settings = settings
        self.frame    = ttk.Frame(parent)

        self._history_dir = os.path.join(BASE_DIR, "outputs", "history")
        self._logger      = PredictionLogger(self._history_dir)
        self._analyzer    = FailureAnalyzer(self._logger)
        self._summary: FailureSummary | None = None
        self._running = False

        self._build()

    # ─────────────────────────────────────────────────────────────────────────
    # 공개 API
    # ─────────────────────────────────────────────────────────────────────────

    def refresh(self) -> None:
        """외부에서 호출 — 분석을 백그라운드로 실행."""
        if self._running:
            return
        self._running = True
        self._set_status("분석 중...", YELLOW)
        self._btn_refresh.config(state="disabled")
        threading.Thread(target=self._bg_analyze, daemon=True).start()

    # ─────────────────────────────────────────────────────────────────────────
    # UI 구성
    # ─────────────────────────────────────────────────────────────────────────

    def _build(self) -> None:
        self.frame.configure(style="TFrame")

        # ── 배너 ──────────────────────────────────────────────────────────────
        banner = tk.Frame(self.frame, bg="#1a1a2e", pady=7)
        banner.pack(fill="x", padx=6, pady=(6, 2))

        tk.Label(
            banner,
            text="🔬  실패 분석  —  예측 실패 패턴 해부 및 자기교정 방향 제시",
            bg="#1a1a2e", fg=ACCENT,
            font=("맑은 고딕", 10, "bold"),
        ).pack(side="left", padx=12)

        self._lbl_status = tk.Label(
            banner,
            text="  (분석 실행 버튼을 눌러 분석을 시작하세요)",
            bg="#1a1a2e", fg=DIM,
            font=("맑은 고딕", 8),
        )
        self._lbl_status.pack(side="left")

        # ── 스크롤 가능한 메인 영역 ───────────────────────────────────────────
        outer = tk.Frame(self.frame, bg=BG)
        outer.pack(fill="both", expand=True, padx=6, pady=(0, 6))

        canvas = tk.Canvas(outer, bg=BG, highlightthickness=0)
        vscroll = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vscroll.set)

        vscroll.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        self._scroll_frame = tk.Frame(canvas, bg=BG)
        self._scroll_win_id = canvas.create_window(
            (0, 0), window=self._scroll_frame, anchor="nw"
        )

        def _on_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _on_canvas_resize(event):
            canvas.itemconfig(self._scroll_win_id, width=event.width)

        self._scroll_frame.bind("<Configure>", _on_configure)
        canvas.bind("<Configure>", _on_canvas_resize)

        # 마우스 휠 스크롤
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind("<MouseWheel>", _on_mousewheel)

        self._canvas_ref = canvas

        # ── 섹션 1: 요약 대시보드 ─────────────────────────────────────────────
        self._build_summary_section(self._scroll_frame)

        # ── 섹션 2: 실패 유형 분포 차트 ──────────────────────────────────────
        self._build_chart_section(self._scroll_frame)

        # ── 섹션 3: Top N 실패 사례 ───────────────────────────────────────────
        self._build_failures_section(self._scroll_frame)

        # ── 섹션 4: 개선 제안 ─────────────────────────────────────────────────
        self._build_suggestions_section(self._scroll_frame)

        # ── 섹션 5: 종목별 성능 매트릭스 ─────────────────────────────────────
        self._build_symbol_matrix_section(self._scroll_frame)

        # ── 섹션 6: 선택된 실패 사례 상세 ────────────────────────────────────
        self._build_detail_section(self._scroll_frame)

    # ─────────────────────────────────────────────────────────────────────────
    # 섹션 빌더
    # ─────────────────────────────────────────────────────────────────────────

    def _section_header(self, parent: tk.Widget, title: str) -> tk.Frame:
        hdr = tk.Frame(parent, bg=PANEL_BG, pady=4)
        hdr.pack(fill="x", padx=4, pady=(8, 2))
        tk.Label(
            hdr, text=title,
            bg=PANEL_BG, fg=ACCENT,
            font=("맑은 고딕", 9, "bold"),
        ).pack(side="left", padx=10)
        tk.Frame(hdr, bg=OVERLAY, height=1).pack(side="left", fill="x", expand=True, padx=(0, 10))
        return hdr

    def _build_summary_section(self, parent: tk.Widget) -> None:
        self._section_header(parent, "📊  요약 대시보드")

        card = tk.Frame(parent, bg=PANEL_BG, pady=10)
        card.pack(fill="x", padx=4, pady=2)

        kpi_row = tk.Frame(card, bg=PANEL_BG)
        kpi_row.pack(fill="x", padx=10, pady=4)

        # KPI 위젯 딕셔너리
        self._kpi_vars: dict[str, tk.StringVar] = {}
        kpi_defs = [
            ("total",     "전체 예측",   FG),
            ("verified",  "검증 완료",   CYAN),
            ("hits",      "적중 수",      GREEN),
            ("hit_rate",  "적중률",       GREEN),
            ("mae",       "MAE",          YELLOW),
            ("rmse",      "RMSE",         ORANGE),
            ("mape",      "MAPE",         CYAN),
        ]

        for key, label, color in kpi_defs:
            var = tk.StringVar(value="—")
            self._kpi_vars[key] = var
            box = tk.Frame(kpi_row, bg=SURFACE, padx=10, pady=6)
            box.pack(side="left", padx=4, fill="y")
            tk.Label(box, text=label, bg=SURFACE, fg=DIM,
                     font=("맑은 고딕", 7)).pack()
            self._kpi_labels = getattr(self, "_kpi_labels", {})
            lbl = tk.Label(box, textvariable=var, bg=SURFACE, fg=color,
                           font=("맑은 고딕", 11, "bold"))
            lbl.pack()
            self._kpi_labels[key] = lbl

        # 분석 실행 버튼
        btn_frame = tk.Frame(card, bg=PANEL_BG)
        btn_frame.pack(fill="x", padx=10, pady=(4, 2))

        self._btn_refresh = tk.Button(
            btn_frame,
            text="🔄 분석 실행",
            bg=ACCENT, fg="#11111b",
            font=("맑은 고딕", 9, "bold"),
            relief="flat", padx=16, pady=4,
            cursor="hand2",
            command=self.refresh,
        )
        self._btn_refresh.pack(side="left")

    def _build_chart_section(self, parent: tk.Widget) -> None:
        self._section_header(parent, "📊  실패 유형 분포")

        chart_frame = tk.Frame(parent, bg=PANEL_BG, pady=4)
        chart_frame.pack(fill="x", padx=4, pady=2)

        self._chart_canvas = tk.Canvas(
            chart_frame, bg=PANEL_BG, height=170,
            highlightthickness=0,
        )
        self._chart_canvas.pack(fill="x", padx=8, pady=4)

        self._chart_canvas.create_text(
            10, 80,
            text="분석 실행 후 차트가 표시됩니다",
            fill=DIM, anchor="w",
            font=("맑은 고딕", 9),
            tags="placeholder",
        )

    def _build_failures_section(self, parent: tk.Widget) -> None:
        self._section_header(parent, "❌  Top 실패 사례  (오차 큰 순)")

        frame = tk.Frame(parent, bg=PANEL_BG)
        frame.pack(fill="x", padx=4, pady=2)

        cols = ("rank", "symbol", "date", "horizon", "dir",
                "pred_ret", "actual_ret", "error", "fail_type", "conf")
        col_labels = ("순위", "종목", "예측일", "기간(일)",
                      "예측방향", "예측수익률", "실제수익률",
                      "오차(%)", "실패유형", "신뢰도")
        col_widths = (40, 70, 85, 55, 65, 75, 75, 60, 100, 60)

        style = ttk.Style()
        style.configure(
            "Failure.Treeview",
            background=PANEL_BG, foreground=FG,
            fieldbackground=PANEL_BG,
            rowheight=22,
        )
        style.configure(
            "Failure.Treeview.Heading",
            background=SURFACE, foreground=ACCENT,
        )
        style.map("Failure.Treeview", background=[("selected", OVERLAY)])

        tree_frame = tk.Frame(frame, bg=PANEL_BG)
        tree_frame.pack(fill="x", padx=8, pady=4)

        self._fail_tree = ttk.Treeview(
            tree_frame, columns=cols, show="headings",
            height=10, style="Failure.Treeview",
        )
        for col, label, width in zip(cols, col_labels, col_widths):
            self._fail_tree.heading(
                col, text=label,
                command=lambda c=col: self._sort_fail_tree(c),
            )
            self._fail_tree.column(col, width=width, anchor="center", stretch=False)

        hsb = ttk.Scrollbar(tree_frame, orient="horizontal",
                             command=self._fail_tree.xview)
        vsb = ttk.Scrollbar(tree_frame, orient="vertical",
                             command=self._fail_tree.yview)
        self._fail_tree.configure(xscrollcommand=hsb.set, yscrollcommand=vsb.set)

        vsb.pack(side="right", fill="y")
        self._fail_tree.pack(side="left", fill="both", expand=True)
        hsb.pack(fill="x")

        self._fail_tree.bind("<<TreeviewSelect>>", self._on_row_select)
        self._fail_sort_col = "error"
        self._fail_sort_rev = True

    def _build_suggestions_section(self, parent: tk.Widget) -> None:
        self._section_header(parent, "💡  자기교정 제안")

        frame = tk.Frame(parent, bg=PANEL_BG)
        frame.pack(fill="x", padx=4, pady=2)

        self._suggestion_text = tk.Text(
            frame, bg=SURFACE, fg=FG,
            font=("맑은 고딕", 9),
            height=6, wrap="word",
            relief="flat", padx=10, pady=6,
            state="disabled",
        )
        self._suggestion_text.pack(fill="x", padx=8, pady=4)

        # 색상 태그 정의
        self._suggestion_text.tag_configure("red",    foreground=RED)
        self._suggestion_text.tag_configure("orange", foreground=ORANGE)
        self._suggestion_text.tag_configure("yellow", foreground=YELLOW)
        self._suggestion_text.tag_configure("green",  foreground=GREEN)
        self._suggestion_text.tag_configure("cyan",   foreground=CYAN)
        self._suggestion_text.tag_configure("dim",    foreground=DIM)
        self._suggestion_text.tag_configure("normal", foreground=FG)

    def _build_symbol_matrix_section(self, parent: tk.Widget) -> None:
        self._section_header(parent, "📈  종목별 성능 매트릭스")

        frame = tk.Frame(parent, bg=PANEL_BG)
        frame.pack(fill="x", padx=4, pady=2)

        sym_cols = ("symbol", "count", "hit_rate", "mae")
        sym_labels = ("종목", "예측횟수", "적중률", "MAE(%p)")
        sym_widths = (90, 70, 80, 80)

        style = ttk.Style()
        style.configure(
            "Symbol.Treeview",
            background=PANEL_BG, foreground=FG,
            fieldbackground=PANEL_BG, rowheight=20,
        )
        style.configure(
            "Symbol.Treeview.Heading",
            background=SURFACE, foreground=CYAN,
        )
        style.map("Symbol.Treeview", background=[("selected", OVERLAY)])

        tree_frame = tk.Frame(frame, bg=PANEL_BG)
        tree_frame.pack(fill="x", padx=8, pady=4)

        self._sym_tree = ttk.Treeview(
            tree_frame, columns=sym_cols, show="headings",
            height=6, style="Symbol.Treeview",
        )
        for col, label, width in zip(sym_cols, sym_labels, sym_widths):
            self._sym_tree.heading(col, text=label)
            self._sym_tree.column(col, width=width, anchor="center", stretch=False)

        sym_vsb = ttk.Scrollbar(tree_frame, orient="vertical",
                                 command=self._sym_tree.yview)
        self._sym_tree.configure(yscrollcommand=sym_vsb.set)
        sym_vsb.pack(side="right", fill="y")
        self._sym_tree.pack(side="left", fill="both", expand=True)

        # 기간별 섹션 (같은 카드 안)
        self._section_header(parent, "⏱  기간별 성능 매트릭스")
        frame2 = tk.Frame(parent, bg=PANEL_BG)
        frame2.pack(fill="x", padx=4, pady=2)

        hor_cols = ("horizon", "count", "hit_rate", "mae")
        hor_labels = ("기간(일)", "예측횟수", "적중률", "MAE(%p)")
        hor_widths = (70, 70, 80, 80)

        style.configure(
            "Horizon.Treeview",
            background=PANEL_BG, foreground=FG,
            fieldbackground=PANEL_BG, rowheight=20,
        )
        style.configure(
            "Horizon.Treeview.Heading",
            background=SURFACE, foreground=TEAL,
        )
        style.map("Horizon.Treeview", background=[("selected", OVERLAY)])

        tree_frame2 = tk.Frame(frame2, bg=PANEL_BG)
        tree_frame2.pack(fill="x", padx=8, pady=4)

        self._hor_tree = ttk.Treeview(
            tree_frame2, columns=hor_cols, show="headings",
            height=4, style="Horizon.Treeview",
        )
        for col, label, width in zip(hor_cols, hor_labels, hor_widths):
            self._hor_tree.heading(col, text=label)
            self._hor_tree.column(col, width=width, anchor="center", stretch=False)

        hor_vsb = ttk.Scrollbar(tree_frame2, orient="vertical",
                                  command=self._hor_tree.yview)
        self._hor_tree.configure(yscrollcommand=hor_vsb.set)
        hor_vsb.pack(side="right", fill="y")
        self._hor_tree.pack(side="left", fill="both", expand=True)

    def _build_detail_section(self, parent: tk.Widget) -> None:
        self._section_header(parent, "🔍  선택된 실패 사례 상세")

        frame = tk.Frame(parent, bg=PANEL_BG)
        frame.pack(fill="x", padx=4, pady=(2, 10))

        self._detail_text = tk.Text(
            frame, bg=SURFACE, fg=FG,
            font=("맑은 고딕", 9),
            height=5, wrap="word",
            relief="flat", padx=10, pady=6,
            state="disabled",
        )
        self._detail_text.pack(fill="x", padx=8, pady=4)
        self._detail_text.tag_configure("label",  foreground=DIM)
        self._detail_text.tag_configure("value",  foreground=FG)
        self._detail_text.tag_configure("red",    foreground=RED)
        self._detail_text.tag_configure("green",  foreground=GREEN)
        self._detail_text.tag_configure("yellow", foreground=YELLOW)
        self._detail_text.tag_configure("accent", foreground=ACCENT)

        # 초기 안내
        self._set_detail_text("실패 사례 목록에서 행을 선택하면 상세 정보가 표시됩니다.")

    # ─────────────────────────────────────────────────────────────────────────
    # 백그라운드 분석
    # ─────────────────────────────────────────────────────────────────────────

    def _bg_analyze(self) -> None:
        try:
            summary = self._analyzer.analyze(top_n=30)
            self.frame.after(0, lambda: self._on_analysis_done(summary))
        except Exception as exc:
            self.frame.after(0, lambda: self._on_analysis_error(str(exc)))

    def _on_analysis_done(self, summary: FailureSummary) -> None:
        self._summary = summary
        self._running = False
        self._btn_refresh.config(state="normal")
        self._set_status(
            f"  마지막 분석: 검증 {summary.verified_count}건 / 전체 {summary.total_predictions}건",
            DIM,
        )
        self._update_kpi(summary)
        self._draw_failure_chart(summary.failure_type_counts)
        self._populate_top_failures(summary.top_failures)
        self._show_suggestions(summary.suggestions)
        self._populate_symbol_matrix(summary.by_symbol)
        self._populate_horizon_matrix(summary.by_horizon)

    def _on_analysis_error(self, msg: str) -> None:
        self._running = False
        self._btn_refresh.config(state="normal")
        self._set_status(f"  오류: {msg}", RED)

    # ─────────────────────────────────────────────────────────────────────────
    # UI 업데이트 메서드
    # ─────────────────────────────────────────────────────────────────────────

    def _set_status(self, text: str, color: str = DIM) -> None:
        self._lbl_status.config(text=text, fg=color)

    def _update_kpi(self, s: FailureSummary) -> None:
        self._kpi_vars["total"].set(str(s.total_predictions))
        self._kpi_vars["verified"].set(str(s.verified_count))
        self._kpi_vars["hits"].set(str(s.hit_count))

        hit_pct = f"{s.hit_rate:.1%}"
        self._kpi_vars["hit_rate"].set(hit_pct)

        # 색상 변경
        if s.hit_rate >= 0.60:
            hr_color = GREEN
        elif s.hit_rate >= 0.50:
            hr_color = YELLOW
        else:
            hr_color = RED
        if hasattr(self, "_kpi_labels"):
            self._kpi_labels["hit_rate"].config(fg=hr_color)
            self._kpi_labels["hits"].config(fg=hr_color)

        self._kpi_vars["mae"].set(f"{s.mae:.2f}%")
        self._kpi_vars["rmse"].set(f"{s.rmse:.2f}%")
        self._kpi_vars["mape"].set(f"{s.mape:.1f}%")

    def _draw_failure_chart(self, data: dict) -> None:
        """수평 막대 차트를 Canvas에 그린다."""
        c = self._chart_canvas
        c.delete("all")

        if not data:
            c.create_text(10, 80, text="데이터 없음", fill=DIM,
                          anchor="w", font=("맑은 고딕", 9))
            return

        # 정렬: 개수 큰 순
        items = sorted(data.items(), key=lambda x: x[1], reverse=True)
        max_val = max(v for _, v in items) if items else 1
        if max_val == 0:
            max_val = 1

        canvas_w = c.winfo_width() or 600
        left_margin  = 130
        right_margin = 60
        bar_area_w   = max(canvas_w - left_margin - right_margin, 100)
        bar_height   = 18
        bar_gap      = 8
        top_pad      = 10

        c.configure(height=len(items) * (bar_height + bar_gap) + top_pad * 2)

        for i, (ftype, count) in enumerate(items):
            y_top  = top_pad + i * (bar_height + bar_gap)
            y_mid  = y_top + bar_height // 2
            y_bot  = y_top + bar_height
            bar_w  = int(bar_area_w * count / max_val)

            color = _CHART_COLORS.get(ftype, OVERLAY)
            label = FAILURE_TYPES.get(ftype, ftype)

            # 레이블 (왼쪽)
            c.create_text(
                left_margin - 6, y_mid,
                text=label, fill=FG,
                anchor="e", font=("맑은 고딕", 8),
            )

            # 막대
            if bar_w > 0:
                c.create_rectangle(
                    left_margin, y_top,
                    left_margin + bar_w, y_bot,
                    fill=color, outline="",
                )

            # 배경 트랙
            c.create_rectangle(
                left_margin, y_top,
                left_margin + bar_area_w, y_bot,
                fill="", outline=OVERLAY,
            )

            # 값 텍스트
            c.create_text(
                left_margin + bar_w + 6, y_mid,
                text=str(count), fill=color,
                anchor="w", font=("맑은 고딕", 8, "bold"),
            )

    def _populate_top_failures(self, failures: list[dict]) -> None:
        """Top N 실패 사례를 Treeview에 채운다."""
        for item in self._fail_tree.get_children():
            self._fail_tree.delete(item)

        if not failures:
            return

        for rank, f in enumerate(failures, start=1):
            dir_val  = f["predicted_direction"]
            conf_val = f["confidence"]
            err_pct  = f["abs_error_pct"]

            # 실패 유형에 따른 행 태그
            ft = f["failure_type"]
            tag = "crash" if ft == "MISSED_CRASH" else \
                  "rally" if ft == "MISSED_RALLY" else \
                  "opt"   if ft == "OVER_OPTIMISTIC" else \
                  "pes"   if ft == "OVER_PESSIMISTIC" else \
                  "side"  if ft == "SIDEWAYS_MISREAD" else "normal"

            self._fail_tree.insert(
                "", "end",
                iid=f["id"],
                values=(
                    rank,
                    f["symbol"],
                    f["timestamp"],
                    f["horizon_days"],
                    dir_val,
                    f"{f['predicted_return_pct']:+.2f}%",
                    f"{f['actual_return_pct']:+.2f}%",
                    f"{err_pct:.2f}",
                    f["failure_type_kr"],
                    conf_val,
                ),
                tags=(tag,),
            )

        # 태그 색상
        self._fail_tree.tag_configure("crash",  foreground=RED)
        self._fail_tree.tag_configure("rally",  foreground=GREEN)
        self._fail_tree.tag_configure("opt",    foreground=ORANGE)
        self._fail_tree.tag_configure("pes",    foreground=CYAN)
        self._fail_tree.tag_configure("side",   foreground=YELLOW)
        self._fail_tree.tag_configure("normal", foreground=FG)

    def _populate_symbol_matrix(self, by_symbol: dict) -> None:
        """종목별 매트릭스 Treeview 채우기."""
        for item in self._sym_tree.get_children():
            self._sym_tree.delete(item)

        if not by_symbol:
            return

        sorted_syms = sorted(
            by_symbol.items(),
            key=lambda x: x[1]["hit_rate"],
            reverse=True,
        )

        for sym, d in sorted_syms:
            hr = d["hit_rate"]
            tag = "good" if hr >= 0.60 else "mid" if hr >= 0.50 else "bad"
            self._sym_tree.insert(
                "", "end",
                values=(
                    sym,
                    d["count"],
                    f"{hr:.1%}",
                    f"{d['mae']:.2f}",
                ),
                tags=(tag,),
            )

        self._sym_tree.tag_configure("good", foreground=GREEN)
        self._sym_tree.tag_configure("mid",  foreground=YELLOW)
        self._sym_tree.tag_configure("bad",  foreground=RED)

    def _populate_horizon_matrix(self, by_horizon: dict) -> None:
        """기간별 매트릭스 Treeview 채우기."""
        for item in self._hor_tree.get_children():
            self._hor_tree.delete(item)

        if not by_horizon:
            return

        sorted_hor = sorted(by_horizon.items(), key=lambda x: x[0])

        for h, d in sorted_hor:
            hr  = d["hit_rate"]
            tag = "good" if hr >= 0.60 else "mid" if hr >= 0.50 else "bad"
            self._hor_tree.insert(
                "", "end",
                values=(
                    f"{h}일",
                    d["count"],
                    f"{hr:.1%}",
                    f"{d['mae']:.2f}",
                ),
                tags=(tag,),
            )

        self._hor_tree.tag_configure("good", foreground=GREEN)
        self._hor_tree.tag_configure("mid",  foreground=YELLOW)
        self._hor_tree.tag_configure("bad",  foreground=RED)

    def _show_suggestions(self, suggestions: list[str]) -> None:
        """개선 제안을 색상 구분하여 텍스트 위젯에 표시."""
        txt = self._suggestion_text
        txt.config(state="normal")
        txt.delete("1.0", "end")

        for sug in suggestions:
            # 첫 글자로 색상 결정 (이모지 없이 접두사로 판단)
            if "[급락" in sug or "[전체 적중률" in sug:
                tag = "red"
            elif "[과도한 낙관" in sug or "[HIGH 신뢰도" in sug:
                tag = "orange"
            elif "[횡보" in sug or "[적중률" in sug:
                tag = "yellow"
            elif "발견되지 않았습니다" in sug or "포인트가 없습니다" in sug:
                tag = "green"
            elif "검증된 예측이 없습니다" in sug:
                tag = "dim"
            elif "[MAE" in sug:
                tag = "yellow"
            elif "[급등" in sug:
                tag = "cyan"
            else:
                tag = "normal"

            txt.insert("end", "• " + sug + "\n", tag)

        txt.config(state="disabled")

    def _on_row_select(self, event=None) -> None:
        """실패 사례 행 선택 시 상세 패널에 표시."""
        selection = self._fail_tree.selection()
        if not selection:
            return

        iid   = selection[0]   # row iid == prediction id
        items = self._fail_tree.item(iid, "values")
        if not items:
            return

        # items: rank, symbol, date, horizon, dir, pred_ret, actual_ret, error, fail_type, conf
        rank, symbol, date, horizon, direction, pred_ret, actual_ret, error, fail_type_kr, conf = items

        txt = self._detail_text
        txt.config(state="normal")
        txt.delete("1.0", "end")

        def _row(label: str, value: str, val_tag: str = "value") -> None:
            txt.insert("end", f"  {label:<18}", "label")
            txt.insert("end", f"{value}\n", val_tag)

        txt.insert("end", f"  ═══════════════ 실패 사례 #{rank} 상세 ═══════════════\n", "accent")

        _row("종목 코드:", symbol)
        _row("예측일:", date)
        _row("예측 기간:", f"{horizon}일")

        dir_tag = "green" if direction == "UP" else "red" if direction == "DOWN" else "yellow"
        _row("예측 방향:", direction, dir_tag)
        _row("예측 수익률:", pred_ret)

        # 실제 수익률 색상
        try:
            actual_val = float(actual_ret.replace("%", "").replace("+", ""))
            actual_tag = "green" if actual_val > 0 else "red" if actual_val < 0 else "yellow"
        except ValueError:
            actual_tag = "value"
        _row("실제 수익률:", actual_ret, actual_tag)
        _row("절대 오차:", f"{error}%p")
        _row("실패 유형:", fail_type_kr, "red")

        conf_tag = "green" if conf == "HIGH" else "yellow" if conf == "MEDIUM" else "red"
        _row("신뢰도:", conf, conf_tag)

        txt.config(state="disabled")

    # ─────────────────────────────────────────────────────────────────────────
    # 정렬
    # ─────────────────────────────────────────────────────────────────────────

    def _sort_fail_tree(self, col: str) -> None:
        """Treeview 헤더 클릭 시 정렬."""
        if self._fail_sort_col == col:
            self._fail_sort_rev = not self._fail_sort_rev
        else:
            self._fail_sort_col = col
            self._fail_sort_rev = True

        data = [
            (self._fail_tree.set(child, col), child)
            for child in self._fail_tree.get_children("")
        ]

        def _sort_key(x: tuple) -> Any:
            val = x[0]
            try:
                return float(val.replace("%", "").replace("+", ""))
            except (ValueError, AttributeError):
                return val

        data.sort(key=_sort_key, reverse=self._fail_sort_rev)

        for idx, (_, child) in enumerate(data):
            self._fail_tree.move(child, "", idx)

    # ─────────────────────────────────────────────────────────────────────────
    # 내부 헬퍼
    # ─────────────────────────────────────────────────────────────────────────

    def _set_detail_text(self, msg: str) -> None:
        self._detail_text.config(state="normal")
        self._detail_text.delete("1.0", "end")
        self._detail_text.insert("end", f"  {msg}", "label")
        self._detail_text.config(state="disabled")
