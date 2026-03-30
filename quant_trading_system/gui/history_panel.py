"""
gui/history_panel.py
====================
예측 이력 패널 — 예측 기록 조회, 검증 현황, 정확도 통계를 한 화면에 표시.

탭 구성:
    ① 예측 이력   — 전체 예측 목록 (필터/정렬 가능)
    ② 정확도 분석 — 적중률·MAPE·액션별 성과
    ③ 검증 실행   — 미검증 예측 일괄 검증
"""
from __future__ import annotations

import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from typing import Any

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from config import AppSettings
from data import DataLoader
from data.korean_stocks import get_name
from history.logger import PredictionLogger
from history.schema import PredictionRecord
from history.verifier import TruthVerifier
from gui.tooltip import add_tooltip


class HistoryPanel:
    """예측 이력 조회·검증 패널."""

    def __init__(self, parent: tk.Widget, settings: AppSettings) -> None:
        self.settings = settings
        self.frame    = ttk.Frame(parent)

        self._history_dir = os.path.join(BASE_DIR, "outputs", "history")
        self._logger      = PredictionLogger(self._history_dir)
        self._records:    list[PredictionRecord] = []
        self._sort_col    = "timestamp"
        self._sort_rev    = True
        self._filter_sym  = ""
        self._filter_act  = ""

        self._build()
        # 비동기 로드
        threading.Thread(target=self._async_reload, daemon=True).start()

    # ─────────────────────────────────────────────────────────────────────────
    # 공개 API
    # ─────────────────────────────────────────────────────────────────────────

    def refresh(self) -> None:
        """외부에서 호출 — 이력을 다시 로드."""
        threading.Thread(target=self._async_reload, daemon=True).start()

    # ─────────────────────────────────────────────────────────────────────────
    # UI 구성
    # ─────────────────────────────────────────────────────────────────────────

    def _build(self) -> None:
        # 배너
        banner = tk.Frame(self.frame, bg="#1a1a2e", pady=7)
        banner.pack(fill="x", padx=6, pady=(6, 2))
        tk.Label(
            banner,
            text="📋  예측 이력  —  과거 예측 기록 조회 및 실제 결과 검증",
            bg="#1a1a2e", fg="#cba6f7",
            font=("맑은 고딕", 10, "bold"),
        ).pack(side="left", padx=12)
        tk.Label(
            banner,
            text="  (예측 탭에서 실행한 예측이 모두 여기에 자동 저장됩니다)",
            bg="#1a1a2e", fg="#9399b2",
            font=("맑은 고딕", 8),
        ).pack(side="left")

        nb = ttk.Notebook(self.frame)
        nb.pack(fill="both", expand=True, padx=6, pady=4)

        # ── 탭 ① 예측 이력 ──────────────────────────────────────────────────
        tab1 = ttk.Frame(nb)
        nb.add(tab1, text="  📋 예측 이력  ")
        self._build_list_tab(tab1)

        # ── 탭 ② 정확도 분석 ─────────────────────────────────────────────────
        tab2 = ttk.Frame(nb)
        nb.add(tab2, text="  📊 정확도 분석  ")
        self._build_stats_tab(tab2)

        # ── 탭 ③ 검증 실행 ──────────────────────────────────────────────────
        tab3 = ttk.Frame(nb)
        nb.add(tab3, text="  ✅ 검증 실행  ")
        self._build_verify_tab(tab3)

    # ─── 탭 ①: 예측 이력 목록 ───────────────────────────────────────────────

    def _build_list_tab(self, parent: ttk.Frame) -> None:
        # 필터 바
        fbar = tk.Frame(parent, bg="#1e1e2e", pady=4)
        fbar.pack(fill="x", padx=4, pady=(4, 0))

        tk.Label(fbar, text="종목:", bg="#1e1e2e", fg="#9399b2",
                 font=("맑은 고딕", 9)).pack(side="left", padx=(8, 2))
        self._sym_filter_var = tk.StringVar()
        sym_entry = ttk.Entry(fbar, textvariable=self._sym_filter_var, width=12)
        sym_entry.pack(side="left", padx=(0, 10))
        add_tooltip(sym_entry, "종목 코드 일부 입력 (예: 005930)")

        tk.Label(fbar, text="액션:", bg="#1e1e2e", fg="#9399b2",
                 font=("맑은 고딕", 9)).pack(side="left", padx=(0, 2))
        self._act_filter_var = tk.StringVar(value="전체")
        act_combo = ttk.Combobox(fbar, textvariable=self._act_filter_var,
                                 values=["전체", "BUY", "SELL", "HOLD", "WATCH"],
                                 width=8, state="readonly")
        act_combo.pack(side="left", padx=(0, 10))

        self._verified_only_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(fbar, text="검증 완료만",
                        variable=self._verified_only_var).pack(side="left", padx=4)

        ttk.Button(fbar, text="🔍 필터 적용",
                   command=self._apply_filter).pack(side="left", padx=4)
        ttk.Button(fbar, text="↺ 새로고침",
                   command=self.refresh).pack(side="left", padx=4)

        self._count_var = tk.StringVar(value="총 0건")
        tk.Label(fbar, textvariable=self._count_var,
                 bg="#1e1e2e", fg="#89b4fa",
                 font=("맑은 고딕", 9)).pack(side="right", padx=10)

        # Treeview
        cols = ("날짜", "종목", "방향", "확률", "예측수익률",
                "신뢰도", "액션", "현재가", "검증", "실제수익률", "적중")
        self._tree = ttk.Treeview(parent, columns=cols, show="headings", height=20)

        col_widths = {
            "날짜": 115, "종목": 140, "방향": 60, "확률": 60,
            "예측수익률": 75, "신뢰도": 60, "액션": 60,
            "현재가": 75, "검증": 50, "실제수익률": 75, "적중": 50,
        }
        for c in cols:
            self._tree.heading(c, text=c,
                               command=lambda _c=c: self._sort_by(_c))
            self._tree.column(c, width=col_widths.get(c, 80), anchor="center")
        self._tree.column("날짜", anchor="w")
        self._tree.column("종목", anchor="w")

        # 색상 태그
        self._tree.tag_configure("buy",      foreground="#a6e3a1")
        self._tree.tag_configure("sell",     foreground="#f38ba8")
        self._tree.tag_configure("watch",    foreground="#f9e2af")
        self._tree.tag_configure("hold",     foreground="#9399b2")
        self._tree.tag_configure("hit",      foreground="#a6e3a1")
        self._tree.tag_configure("miss",     foreground="#f38ba8")
        self._tree.tag_configure("unverified", foreground="#585b70")

        vsb = ttk.Scrollbar(parent, orient="vertical",   command=self._tree.yview)
        hsb = ttk.Scrollbar(parent, orient="horizontal", command=self._tree.xview)
        self._tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        self._tree.pack(side="left", fill="both", expand=True, padx=(4, 0), pady=4)
        vsb.pack(side="right", fill="y", pady=4)

        # 선택 상세 패널
        detail_fr = ttk.LabelFrame(parent, text="선택 항목 상세", padding=6)
        detail_fr.pack(fill="x", padx=4, pady=(0, 4))
        self._detail_var = tk.StringVar(value="목록에서 항목을 선택하세요.")
        tk.Label(detail_fr, textvariable=self._detail_var,
                 bg="#1e1e2e", fg="#cdd6f4",
                 font=("맑은 고딕", 9), justify="left",
                 wraplength=800).pack(anchor="w")

        self._tree.bind("<<TreeviewSelect>>", self._on_select)

    # ─── 탭 ②: 정확도 분석 ──────────────────────────────────────────────────

    def _build_stats_tab(self, parent: ttk.Frame) -> None:
        btn_fr = ttk.Frame(parent)
        btn_fr.pack(fill="x", padx=8, pady=(8, 4))
        ttk.Button(btn_fr, text="📊 통계 계산",
                   command=self._calc_stats).pack(side="left")
        add_tooltip(btn_fr.winfo_children()[0],
                    "검증 완료된 예측 전체의 적중률, MAPE, 액션별 성과를 계산합니다.")

        self._stats_text = scrolledtext.ScrolledText(
            parent, bg="#181825", fg="#cdd6f4",
            font=("Consolas", 10), relief="flat",
            state="disabled", height=30,
        )
        self._stats_text.pack(fill="both", expand=True, padx=8, pady=4)

    # ─── 탭 ③: 검증 실행 ────────────────────────────────────────────────────

    def _build_verify_tab(self, parent: ttk.Frame) -> None:
        info = ttk.LabelFrame(parent, text="검증 안내", padding=8)
        info.pack(fill="x", padx=8, pady=(8, 4))
        tk.Label(
            info,
            text=(
                "🔍 검증이란?\n"
                "  예측 당시 기록된 '예측 방향'과 '예측 수익률'을\n"
                "  실제 이후 주가와 비교하여 적중 여부를 판정합니다.\n\n"
                "  ✅ 조건: 예측 horizon_days 가 지난 예측만 검증 가능합니다.\n"
                "  ⚠️  최신 예측은 아직 검증할 수 없습니다."
            ),
            bg="#1e1e2e", fg="#a6adc8",
            font=("맑은 고딕", 9), justify="left",
        ).pack(anchor="w")

        ctrl_fr = ttk.Frame(parent)
        ctrl_fr.pack(fill="x", padx=8, pady=4)
        self._verify_btn = ttk.Button(
            ctrl_fr, text="✅ 미검증 예측 일괄 검증",
            command=self._run_verify,
            style="Accent.TButton",
        )
        self._verify_btn.pack(side="left", padx=(0, 8))
        add_tooltip(self._verify_btn,
                    "만료된 미검증 예측을 실제 시장 데이터와 비교하여 결과를 기록합니다.")

        self._verify_progress = ttk.Progressbar(ctrl_fr, mode="indeterminate",
                                                length=200)
        self._verify_progress.pack(side="left")

        self._verify_status_var = tk.StringVar(value="대기 중")
        ttk.Label(parent, textvariable=self._verify_status_var,
                  foreground="#89b4fa",
                  font=("맑은 고딕", 9)).pack(anchor="w", padx=8)

        self._verify_log = scrolledtext.ScrolledText(
            parent, bg="#181825", fg="#a6adc8",
            font=("Consolas", 9), relief="flat",
            state="disabled", height=20,
        )
        self._verify_log.pack(fill="both", expand=True, padx=8, pady=4)

    # ─────────────────────────────────────────────────────────────────────────
    # 데이터 로드 / 필터
    # ─────────────────────────────────────────────────────────────────────────

    def _async_reload(self) -> None:
        recs = self._logger.load_all()
        self.frame.after(0, lambda r=recs: self._set_records(r))

    def _set_records(self, recs: list[PredictionRecord]) -> None:
        self._records = recs
        self._apply_filter()

    def _apply_filter(self) -> None:
        sym_f = self._sym_filter_var.get().strip().upper()
        act_f = self._act_filter_var.get()
        if act_f == "전체":
            act_f = ""
        verified_only = self._verified_only_var.get()

        filtered = self._records
        if sym_f:
            filtered = [r for r in filtered if sym_f in r.symbol.upper()]
        if act_f:
            filtered = [r for r in filtered if r.action == act_f]
        if verified_only:
            filtered = [r for r in filtered if r.verified]

        # 정렬
        key_map = {
            "날짜":      "timestamp",
            "종목":      "symbol",
            "방향":      "predicted_direction",
            "확률":      "prob_up",
            "예측수익률": "predicted_return_pct",
            "신뢰도":    "confidence",
            "액션":      "action",
        }
        sort_attr = key_map.get(self._sort_col, "timestamp")
        try:
            filtered.sort(key=lambda r: getattr(r, sort_attr, ""),
                          reverse=self._sort_rev)
        except Exception:
            pass

        self._populate_tree(filtered)
        self._count_var.set(f"총 {len(filtered)}건")

    def _populate_tree(self, recs: list[PredictionRecord]) -> None:
        self._tree.delete(*self._tree.get_children())
        for r in recs:
            name = get_name(r.symbol)
            sym_disp = f"{name} ({r.symbol.split('.')[0]})" if name and name != r.symbol else r.symbol

            prob_str = f"{r.prob_up*100:.1f}%"
            ret_str  = f"{r.predicted_return_pct:+.2f}%"
            price_str = f"{r.price_at_prediction:,.0f}"

            if r.verified:
                verified_str = "✅"
                act_ret_str  = (f"{r.actual_return_pct:+.2f}%"
                                if r.actual_return_pct is not None else "—")
                hit_str      = ("✅" if r.hit else "❌") if r.hit is not None else "—"
                tag_v        = "hit" if r.hit else "miss"
            else:
                verified_str = "—"
                act_ret_str  = "—"
                hit_str      = "—"
                tag_v        = "unverified"

            # action colour tag
            act_tags = {"BUY": "buy", "SELL": "sell",
                        "WATCH": "watch", "HOLD": "hold"}
            tag_a = act_tags.get(r.action, "hold")

            values = (
                r.timestamp[:19].replace("T", " "),
                sym_disp,
                r.predicted_direction,
                prob_str,
                ret_str,
                r.confidence,
                r.action,
                price_str,
                verified_str,
                act_ret_str,
                hit_str,
            )
            self._tree.insert("", "end", iid=r.id,
                              values=values, tags=(tag_a, tag_v))

    # ─────────────────────────────────────────────────────────────────────────
    # 선택 상세
    # ─────────────────────────────────────────────────────────────────────────

    def _on_select(self, _event: tk.Event) -> None:
        sel = self._tree.selection()
        if not sel:
            return
        rec_id = sel[0]
        rec = next((r for r in self._records if r.id == rec_id), None)
        if rec is None:
            return

        name = get_name(rec.symbol)
        lines = [
            f"ID        : {rec.id}",
            f"종목      : {name} ({rec.symbol})",
            f"예측 날짜 : {rec.timestamp[:19].replace('T', ' ')} UTC",
            f"방향      : {rec.predicted_direction}  |  확률: {rec.prob_up*100:.1f}%",
            f"예측수익률: {rec.predicted_return_pct:+.2f}%  (horizon={rec.horizon_days}거래일)",
            f"신뢰도    : {rec.confidence}  |  액션: {rec.action}",
            f"예측가    : {rec.price_at_prediction:,.0f}원",
            f"모델 버전 : {rec.model_version}  |  d_model={rec.d_model}  seg={rec.segment_length}",
        ]
        if rec.verified:
            lines += [
                "─" * 60,
                f"[검증 결과]  날짜: {(rec.verification_date or '')[:10]}",
                f"실제수익률 : {rec.actual_return_pct:+.2f}%",
                f"적중       : {'✅ 맞음' if rec.hit else '❌ 틀림'}",
                f"오차       : {abs((rec.actual_return_pct or 0) - rec.predicted_return_pct):.2f}%p",
            ]
        else:
            lines.append("─" * 60 + "\n[아직 검증되지 않음]")

        self._detail_var.set("\n".join(lines))

    # ─────────────────────────────────────────────────────────────────────────
    # 정렬
    # ─────────────────────────────────────────────────────────────────────────

    def _sort_by(self, col: str) -> None:
        if self._sort_col == col:
            self._sort_rev = not self._sort_rev
        else:
            self._sort_col = col
            self._sort_rev = True
        self._apply_filter()

    # ─────────────────────────────────────────────────────────────────────────
    # 정확도 통계
    # ─────────────────────────────────────────────────────────────────────────

    def _calc_stats(self) -> None:
        stats = self._logger.summary_stats()
        recs  = self._logger.load_all()
        verified = [r for r in recs if r.verified]

        lines = [
            "═" * 52,
            "  예측 정확도 분석 리포트",
            "═" * 52,
            f"  전체 예측 수  : {stats['total']}건",
            f"  검증 완료     : {stats['verified']}건",
            "",
        ]

        if stats["verified"] == 0:
            lines += [
                "  ⚠️  검증된 예측이 없습니다.",
                "  '검증 실행' 탭에서 먼저 검증을 실행하세요.",
            ]
        else:
            hr    = stats.get("hit_rate")
            mape  = stats.get("mape")
            ap    = stats.get("avg_predicted_return")
            aa    = stats.get("avg_actual_return")
            buy_p = stats.get("buy_precision")
            sel_p = stats.get("sell_precision")

            lines += [
                f"  적중률 (방향) : {hr*100:.1f}%" if hr is not None else "  적중률        : —",
                f"  MAPE          : {mape:.2f}%" if mape is not None else "  MAPE           : —",
                f"  평균 예측수익률: {ap:+.2f}%" if ap is not None else "  평균 예측수익률: —",
                f"  평균 실제수익률: {aa:+.2f}%" if aa is not None else "  평균 실제수익률: —",
                "",
                "  ── 액션별 정밀도 ──",
                f"  BUY  정밀도: {buy_p*100:.1f}%" if buy_p is not None else "  BUY  정밀도: —",
                f"  SELL 정밀도: {sel_p*100:.1f}%" if sel_p is not None else "  SELL 정밀도: —",
                "",
            ]

            # Per-symbol breakdown
            by_sym: dict[str, list[PredictionRecord]] = {}
            for r in verified:
                by_sym.setdefault(r.symbol, []).append(r)

            lines.append("  ── 종목별 적중률 ──")
            for sym, sym_recs in sorted(by_sym.items()):
                hits    = [r for r in sym_recs if r.hit]
                sym_hr  = len(hits) / len(sym_recs)
                name    = get_name(sym)
                disp    = f"{name} ({sym.split('.')[0]})" if name and name != sym else sym
                lines.append(f"  {disp:<28} {len(hits)}/{len(sym_recs)} = {sym_hr*100:.0f}%")

        lines.append("")
        lines.append("═" * 52)

        self._stats_text.configure(state="normal")
        self._stats_text.delete("1.0", "end")
        self._stats_text.insert("end", "\n".join(lines))
        self._stats_text.configure(state="disabled")

    # ─────────────────────────────────────────────────────────────────────────
    # 검증 실행
    # ─────────────────────────────────────────────────────────────────────────

    def _run_verify(self) -> None:
        self._verify_btn.configure(state="disabled")
        self._verify_progress.start(12)
        self._verify_log_write("검증 시작...\n")
        self._verify_status_var.set("검증 중...")

        def _do():
            try:
                cache_dir = os.path.join(BASE_DIR, self.settings.data.cache_dir)
                loader    = DataLoader(cache_dir, self.settings.data.cache_ttl_hours)
                verifier  = TruthVerifier(self._logger, loader)
                results   = verifier.verify_all(
                    period   = self.settings.data.period,
                    interval = self.settings.data.interval,
                )
                self.frame.after(0, lambda r=results: self._on_verify_done(r))
            except Exception as e:
                import traceback
                msg = traceback.format_exc()
                self.frame.after(0, lambda m=msg: self._on_verify_error(m))

        threading.Thread(target=_do, daemon=True).start()

    def _on_verify_done(self, results: list) -> None:
        self._verify_progress.stop()
        self._verify_btn.configure(state="normal")

        if not results:
            msg = "검증할 만료된 미검증 예측이 없습니다.\n(아직 horizon_days가 지나지 않은 예측은 검증할 수 없습니다.)"
            self._verify_log_write(msg + "\n")
            self._verify_status_var.set(msg)
            return

        self._verify_log_write(f"✅ 검증 완료: {len(results)}건\n\n")
        hits = [r for r in results if r.hit]
        self._verify_log_write(f"  적중: {len(hits)}건  |  미적중: {len(results)-len(hits)}건\n")
        self._verify_log_write(f"  적중률: {len(hits)/len(results)*100:.1f}%\n\n")

        for vr in results:
            name = get_name(vr.symbol)
            disp = f"{name} ({vr.symbol.split('.')[0]})" if name and name != vr.symbol else vr.symbol
            self._verify_log_write(
                f"  [{disp}]  실제 {vr.actual_return_pct:+.2f}%  "
                f"예측 {vr.predicted_return_pct:+.2f}%  "
                f"{'✅' if vr.hit else '❌'}\n"
            )

        self._verify_status_var.set(f"✅ {len(results)}건 검증 완료 (적중률 {len(hits)/len(results)*100:.0f}%)")
        # 이력 탭 갱신
        self.refresh()

    def _on_verify_error(self, msg: str) -> None:
        self._verify_progress.stop()
        self._verify_btn.configure(state="normal")
        self._verify_log_write(f"\n❌ 오류:\n{msg}\n")
        self._verify_status_var.set("검증 중 오류 발생")

    def _verify_log_write(self, text: str) -> None:
        self._verify_log.configure(state="normal")
        self._verify_log.insert("end", text)
        self._verify_log.see("end")
        self._verify_log.configure(state="disabled")
