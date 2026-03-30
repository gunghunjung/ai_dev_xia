# gui/prediction_panel.py — 미래 예측 패널
"""
프로덕션 등급 미래 주가 예측 패널.

기능:
  ① 예측 기간 / 고신뢰도 필터 / 가중 방법 설정
  ② "미래 예측 실행" 버튼 → InferencePredictor 스레드 실행
  ③ 결과 테이블: 종목 / 현재가 / 방향 / 상승확률 / 예상수익 / 신뢰도 / 추천
  ④ 포트폴리오 제안 패널 (좌측 하단)
  ⑤ 선택 종목 AI 설명 패널 (우측 하단)
  ⑥ 확률 막대 미니 차트 (Tkinter Canvas)
  ⑦ 예측 이력 탭
"""
from __future__ import annotations

import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext
import logging

import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from config import AppSettings
from data import DataLoader
from data.korean_stocks import get_name
from gui.tooltip import add_tooltip
from gui.theme import BG as _BG, PANEL_BG as _PANEL_BG, FG as _FG, DIM as _DIM
from gui.theme import ACCENT as _ACCENT, GREEN as _GREEN, RED as _RED
from gui.theme import YELLOW as _YELLOW, PURPLE as _PURPLE

logger = logging.getLogger("quant.gui.predict")


class PredictionPanel:
    """
    미래 예측 패널 — InferencePredictor를 GUI로 감쌉니다.

    사용 흐름:
      1. 학습 탭에서 모델 학습 → 모델 파일 저장
      2. 예측 탭으로 이동 → 기간 선택 → "미래 예측 실행"
      3. 결과 테이블 확인 → 종목 클릭 → 상세 설명 확인
      4. 포트폴리오 제안 확인 → 투자 판단 참고
    """

    # 통합 예측 horizon 목록 (고정)
    _MULTI_HORIZONS = [1, 3, 5, 20]
    _HORIZON_LABELS = ["1일", "3일", "5일", "20일"]

    def __init__(self, parent: ttk.Notebook, settings: AppSettings, on_complete=None):
        self.settings     = settings
        self._on_complete = on_complete   # 예측 완료 콜백
        self._stop_event  = threading.Event()
        self._thread: threading.Thread | None = None
        self._predictions = []      # 최신 예측 결과 목록
        self._predictor   = None    # InferencePredictor (lazy init)
        self._name_cache: dict[str, str] = {}   # ticker → 종목명 캐시

        self.frame = ttk.Frame(parent)
        self._build()

    # ══════════════════════════════════════════════════════════════════════════
    # UI 구성
    # ══════════════════════════════════════════════════════════════════════════

    def notify_data_updated(self):
        """데이터 탭에서 새 데이터 다운로드 완료 시 메인 윈도우가 호출"""
        if hasattr(self, "_data_update_bar"):
            self._data_update_bar.configure(height=30)

    def dismiss_data_update_notice(self):
        if hasattr(self, "_data_update_bar"):
            self._data_update_bar.configure(height=0)

    def _build(self):
        # 데이터 업데이트 알림 배너 (초기 숨김)
        self._data_update_bar = tk.Frame(self.frame, bg="#2a2a00", height=0)
        self._data_update_bar.pack(fill="x")
        self._data_update_bar.pack_propagate(False)
        _bar_inner = tk.Frame(self._data_update_bar, bg="#2a2a00")
        _bar_inner.pack(fill="both", expand=True, padx=10)
        tk.Label(
            _bar_inner,
            text="🔔  데이터가 업데이트되었습니다  —  새 데이터 반영을 위해 모델 재학습 후 예측을 다시 실행하세요",
            bg="#2a2a00", fg="#f9e2af",
            font=("맑은 고딕", 9, "bold"), anchor="w",
        ).pack(side="left", fill="y")
        ttk.Button(_bar_inner, text="✕ 닫기",
                   command=self.dismiss_data_update_notice).pack(side="right")

        # 안내 배너
        banner = tk.Frame(self.frame, bg="#1a1a2e", pady=7)
        banner.pack(fill="x", padx=6, pady=(6, 2))
        tk.Label(
            banner,
            text="🔮  미래 예측  —  학습된 AI 모델로 향후 주가 방향을 예측합니다",
            bg="#1a1a2e", fg=_PURPLE,
            font=("맑은 고딕", 10, "bold"),
        ).pack(side="left", padx=12)
        tk.Label(
            banner,
            text="  (학습 탭에서 먼저 모델을 학습해야 합니다)",
            bg="#1a1a2e", fg=_DIM,
            font=("맑은 고딕", 8),
        ).pack(side="left")

        # 대상 종목 표시 띠
        sym_bar = tk.Frame(self.frame, bg="#12122a", height=26)
        sym_bar.pack(fill="x", padx=6, pady=(0, 2))
        sym_bar.pack_propagate(False)
        tk.Label(sym_bar, text="🎯", bg="#12122a", fg=_ACCENT,
                 font=("맑은 고딕", 9)).pack(side="left", padx=(10, 4))
        self._target_var = tk.StringVar(value="")
        tk.Label(sym_bar, textvariable=self._target_var,
                 bg="#12122a", fg="#cba6f7",
                 font=("맑은 고딕", 9, "bold"), anchor="w").pack(side="left", fill="x", expand=True)
        self._update_target_symbols_label()

        # 제어 바
        self._build_control_bar()

        # 메인 컨텐츠 (Paned)
        pane = tk.PanedWindow(self.frame, orient="vertical",
                              bg=_BG, sashwidth=4)
        pane.pack(fill="both", expand=True, padx=6, pady=2)

        top_fr  = ttk.Frame(pane)
        bot_fr  = ttk.Frame(pane)
        pane.add(top_fr, minsize=200)
        pane.add(bot_fr, minsize=180)

        self._build_result_table(top_fr)
        self._build_bottom_panels(bot_fr)

    def _build_control_bar(self):
        ctrl = tk.Frame(self.frame, bg=_PANEL_BG, pady=6)
        ctrl.pack(fill="x", padx=6, pady=(0, 2))

        # ── 통합 예측 모드 표시 배지 ───────────────────────────────────
        mode_lbl = tk.Label(
            ctrl,
            text="  🔮 통합 예측  1일 · 3일 · 5일 · 20일 동시 출력  ",
            bg="#1a1a40", fg="#cba6f7",
            font=("맑은 고딕", 9, "bold"),
            relief="flat", padx=8, pady=2,
        )
        mode_lbl.pack(side="left", padx=(12, 8))

        # ── 뉴스 AI 피처 상태 배지 ─────────────────────────────────────
        news_enabled = getattr(getattr(self.settings, "news", None),
                               "use_news_in_model", False)
        badge_text = "뉴스 AI: ON" if news_enabled else "뉴스 AI: OFF"
        badge_bg   = "#1a3020" if news_enabled else "#2a1a1a"
        badge_fg   = "#a6e3a1" if news_enabled else "#6c7086"
        self._news_badge = tk.Label(
            ctrl,
            text=f"  {'●' if news_enabled else '○'}  {badge_text}  ",
            bg=badge_bg, fg=badge_fg,
            font=("맑은 고딕", 8, "bold"),
            relief="flat", padx=6, pady=2,
        )
        self._news_badge.pack(side="left", padx=(0, 12))

        # ── 가중 방법 ──────────────────────────────────────────────────
        tk.Label(ctrl, text="포트폴리오 비중:", bg=_PANEL_BG, fg=_FG,
                 font=("맑은 고딕", 9)).pack(side="left", padx=(0, 4))

        self._method_var = tk.StringVar(value="prob_weight")
        method_combo = ttk.Combobox(
            ctrl, textvariable=self._method_var,
            values=["prob_weight", "return", "equal"],
            state="readonly", width=14,
        )
        method_combo.pack(side="left", padx=(0, 12))
        add_tooltip(method_combo,
                    "포트폴리오 비중을 어떻게 계산할지 선택합니다.\n\n"
                    "prob_weight : 상승 확률에 비례 (추천)\n"
                    "return      : 예상 수익률에 비례\n"
                    "equal       : 모든 종목 동일 비중")

        # ── 상위 N 선택 ────────────────────────────────────────────────
        tk.Label(ctrl, text="상위:", bg=_PANEL_BG, fg=_FG,
                 font=("맑은 고딕", 9)).pack(side="left", padx=(0, 4))

        self._topn_var = tk.StringVar(value="5")
        ttk.Combobox(
            ctrl, textvariable=self._topn_var,
            values=["3", "5", "7", "10"],
            state="readonly", width=4,
        ).pack(side="left", padx=(0, 4))

        tk.Label(ctrl, text="종목", bg=_PANEL_BG, fg=_FG,
                 font=("맑은 고딕", 9)).pack(side="left", padx=(0, 12))

        # ── 고신뢰도 필터 ──────────────────────────────────────────────
        self._high_conf_var = tk.BooleanVar(value=False)
        hc_cb = ttk.Checkbutton(
            ctrl, text="고신뢰도만 표시",
            variable=self._high_conf_var,
        )
        hc_cb.pack(side="left", padx=(0, 12))
        add_tooltip(hc_cb,
                    "체크 시 신뢰도가 HIGH인 종목만 결과 테이블에 표시합니다.\n"
                    "신뢰도가 낮은 예측은 무시됩니다.")

        # ── 데이터 새로고침 ────────────────────────────────────────────
        refresh_btn = ttk.Button(ctrl, text="⟳ 데이터 새로고침",
                                 command=self._refresh_data, width=16)
        refresh_btn.pack(side="right", padx=(4, 12))
        add_tooltip(refresh_btn,
                    "최신 주가 데이터를 다시 다운로드한 후 예측합니다.\n"
                    "캐시를 무시하고 인터넷에서 새로 받아옵니다.")

        # ── 실행 / 중단 버튼 ──────────────────────────────────────────
        self._run_btn = ttk.Button(
            ctrl, text="🔮 미래 예측 실행",
            command=self._run_prediction,
            style="Accent.TButton", width=18,
        )
        self._run_btn.pack(side="right", padx=(0, 4))
        add_tooltip(self._run_btn,
                    "모든 종목에 대해 AI 미래 예측을 실행합니다.\n\n"
                    "⚠️ 학습 탭에서 최소 1개 종목의 모델을 학습한 후 실행하세요.\n"
                    "모델이 없는 종목은 자동으로 건너뜁니다.")

        self._stop_btn = ttk.Button(
            ctrl, text="⏹ 중단",
            command=lambda: self._stop_event.set(),
            style="Danger.TButton", state="disabled",
        )
        self._stop_btn.pack(side="right", padx=(0, 4))

        # ── 진행바 ────────────────────────────────────────────────────
        prog_fr = tk.Frame(self.frame, bg=_BG)
        prog_fr.pack(fill="x", padx=6, pady=(0, 2))
        self._progress = ttk.Progressbar(prog_fr, mode="determinate")
        self._progress.pack(side="left", fill="x", expand=True)
        self._status_var = tk.StringVar(
            value="준비  —  '🔮 미래 예측 실행'을 눌러 시작하세요"
        )
        ttk.Label(prog_fr, textvariable=self._status_var,
                  font=("맑은 고딕", 9), width=40).pack(side="left", padx=8)

    def _build_result_table(self, parent: ttk.Frame):
        """예측 결과 Treeview 테이블"""
        hdr = tk.Frame(parent, bg=_BG)
        hdr.pack(fill="x", pady=(4, 2))
        tk.Label(hdr, text="📋 예측 결과 — 클릭하면 상세 설명이 표시됩니다",
                 bg=_BG, fg=_ACCENT,
                 font=("맑은 고딕", 9, "bold")).pack(side="left", padx=8)
        self._count_var = tk.StringVar(value="")
        tk.Label(hdr, textvariable=self._count_var,
                 bg=_BG, fg=_DIM,
                 font=("맑은 고딕", 9)).pack(side="left")

        # 멀티-호라이즌 통합 리포트 컬럼
        cols = ("종목", "현재가", "1일후", "3일후", "5일후", "20일후", "신뢰도(5d)", "추천(5d)")
        self._tree = ttk.Treeview(parent, columns=cols, show="headings", height=9)

        col_w = {
            "종목":    140, "현재가": 80,
            "1일후": 90, "3일후": 90, "5일후": 90, "20일후": 90,
            "신뢰도(5d)": 80, "추천(5d)": 80,
        }
        for c in cols:
            self._tree.heading(c, text=c, command=lambda col=c: self._sort_by(col))
            self._tree.column(c, width=col_w.get(c, 90), anchor="center")

        # 색상 태그
        self._tree.tag_configure("UP_HIGH",   foreground=_GREEN)
        self._tree.tag_configure("UP_MED",    foreground="#74c7ec")
        self._tree.tag_configure("DOWN_HIGH", foreground=_RED)
        self._tree.tag_configure("DOWN_MED",  foreground="#fab387")
        self._tree.tag_configure("NEUTRAL",   foreground=_DIM)
        self._tree.tag_configure("ERROR",     foreground="#6c7086")
        self._tree.tag_configure("BUY",       foreground=_GREEN,  font=("맑은 고딕", 9, "bold"))
        self._tree.tag_configure("SELL",      foreground=_RED,    font=("맑은 고딕", 9, "bold"))
        self._tree.tag_configure("WATCH",     foreground=_YELLOW)
        self._tree.tag_configure("HOLD",      foreground=_DIM)

        vsb = ttk.Scrollbar(parent, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=vsb.set)
        self._tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        self._tree.bind("<<TreeviewSelect>>", self._on_row_select)
        # 키보드 ↑↓ 이동 시 포커스 → 선택 동기화 후 상세 표시
        def _tree_key_nav(e):
            def _sync():
                focused = self._tree.focus()
                if focused:
                    self._tree.selection_set(focused)
                    self._on_row_select(None)
            self.frame.after(30, _sync)

        for _key in ("<Up>", "<Down>", "<Prior>", "<Next>", "<Home>", "<End>"):
            self._tree.bind(_key, _tree_key_nav)

        # 정렬 상태
        self._sort_col = "예상수익률"
        self._sort_rev = True

    def _build_bottom_panels(self, parent: ttk.Frame):
        """하단 패널: 포트폴리오 제안 + 종목 상세 + 확률 차트"""
        pane = tk.PanedWindow(parent, orient="horizontal",
                              bg=_BG, sashwidth=4)
        pane.pack(fill="both", expand=True)

        # ── 왼쪽: 포트폴리오 제안 ──────────────────────────────────
        port_fr = ttk.LabelFrame(pane, text="📊 포트폴리오 제안", padding=6)
        pane.add(port_fr, minsize=320)

        self._port_text = scrolledtext.ScrolledText(
            port_fr, height=9, bg=_PANEL_BG, fg=_FG,
            font=("맑은 고딕", 9), relief="flat", state="disabled",
            wrap="word",
        )
        self._port_text.pack(fill="both", expand=True)

        # ── 가운데: 선택 종목 AI 설명 ─────────────────────────────
        detail_fr = ttk.LabelFrame(pane, text="🔎 선택 종목 상세", padding=6)
        pane.add(detail_fr, minsize=320)

        self._detail_text = scrolledtext.ScrolledText(
            detail_fr, height=9, bg=_PANEL_BG, fg=_FG,
            font=("맑은 고딕", 9), relief="flat", state="disabled",
            wrap="word",
        )
        self._detail_text.pack(fill="both", expand=True)

        # ── 오른쪽: 확률 막대 차트 ───────────────────────────────
        chart_fr = ttk.LabelFrame(pane, text="📈 상승 확률 분포", padding=4)
        pane.add(chart_fr, minsize=260)

        self._chart_canvas = tk.Canvas(
            chart_fr, bg=_PANEL_BG, highlightthickness=0,
        )
        self._chart_canvas.pack(fill="both", expand=True)

    # ══════════════════════════════════════════════════════════════════════════
    # 예측 실행
    # ══════════════════════════════════════════════════════════════════════════

    def _run_prediction(self):
        if self._thread and self._thread.is_alive():
            return
        if not self.settings.data.symbols:
            self._set_status("❌ 등록된 종목이 없습니다 — 데이터 탭에서 종목을 추가하세요")
            return

        self._stop_event.clear()
        self._run_btn.config(state="disabled")
        self._stop_btn.config(state="normal")
        self._progress["value"] = 0
        self._clear_results()

        self._thread = threading.Thread(
            target=self._prediction_thread, daemon=True
        )
        self._thread.start()

    def _prediction_thread(self):
        """백그라운드 예측 스레드."""
        try:
            from inference import InferencePredictor

            symbols   = self.settings.data.symbols
            cache_dir = os.path.join(BASE_DIR, self.settings.data.cache_dir)
            model_dir = os.path.join(BASE_DIR, self.settings.model_dir)

            # ── InferencePredictor 초기화 ──────────────────────────────
            predictor = InferencePredictor(self.settings, model_dir, cache_dir)

            # ── CUDA 감지 ──────────────────────────────────────────────
            device = "cpu"
            try:
                import torch
                if torch.cuda.is_available():
                    device = "cuda"
            except ImportError:
                pass

            self._set_status("데이터 로드 중...")

            # ── 데이터 로드 ────────────────────────────────────────────
            loader   = DataLoader(cache_dir, self.settings.data.cache_ttl_hours)
            data_dict = {}
            for sym in symbols:
                df = loader.load(sym, self.settings.data.period,
                                 self.settings.data.interval)
                if df is not None and len(df) >= 10:
                    data_dict[sym] = df

            if not data_dict:
                self._set_status("❌ 로드 가능한 데이터가 없습니다. 데이터 탭에서 다운로드하세요.")
                return

            # ── 외부환경 벡터 시도 로드 ───────────────────────────────
            macro_vec = None
            try:
                from features.macro_features import MacroFeatureBuilder
                from data.news_fetcher import NewsFetcher
                fetcher = NewsFetcher(cache_dir=cache_dir)
                events  = fetcher.fetch(max_items=60)
                builder = MacroFeatureBuilder()
                macro_vec = builder.build(events)
            except Exception:
                pass

            # ── 통합 멀티-호라이즌 예측 실행 ──────────────────────────
            high_conf = self._high_conf_var.get()

            def _prog(pct, msg):
                if not self._stop_event.is_set():
                    self.frame.after(0, lambda p=pct, m=msg:
                                     self._update_progress(p, m))

            results = predictor.predict_all_multi_horizon(
                symbols        = symbols,
                data_dict      = data_dict,
                device         = device,
                macro_vec      = macro_vec,
                high_conf_only = high_conf,
                progress_cb    = _prog,
            )

            if self._stop_event.is_set():
                self._set_status("중단됨")
                return

            # ── 포트폴리오 구성 (5d primary 기준) ─────────────────────
            top_n  = int(self._topn_var.get())
            method = self._method_var.get()
            # predict_all_multi_horizon → primary dict 리스트로 변환
            compat_results = []
            for r in results:
                if r.get("error") is None:
                    h5 = r.get("horizons", {}).get("5", {})
                    compat_results.append({
                        "symbol":           r["symbol"],
                        "current_price":    r["current_price"],
                        "predicted_return": h5.get("return_pct", 0.0) / 100.0,
                        "prob_up":          h5.get("prob_up", 0.5),
                        "direction":        h5.get("direction", "NEUTRAL"),
                        "confidence":       h5.get("confidence", "LOW"),
                        "action":           h5.get("action", "HOLD"),
                        "error":            None,
                    })
                else:
                    compat_results.append({"symbol": r["symbol"],
                                           "error": r["error"]})

            portfolio = predictor.build_portfolio_from_predictions(
                compat_results,
                method         = method,
                top_n          = top_n,
                min_confidence = "MEDIUM" if high_conf else "LOW",
                data_dict      = data_dict,
            )

            # UI 업데이트 (메인 스레드에서)
            self.frame.after(0, lambda r=results, p=portfolio:
                             self._show_results(r, p))

        except Exception as exc:
            import traceback
            msg = f"❌ 예측 오류:\n{traceback.format_exc()}"
            logger.error(msg)
            self.frame.after(0, lambda m=msg: self._set_status(m[:80]))
        finally:
            self.frame.after(0, self._on_done)

    def _refresh_data(self):
        """캐시를 무시하고 데이터를 새로 받아온 뒤 예측을 실행합니다."""
        try:
            cache_dir = os.path.join(BASE_DIR, self.settings.data.cache_dir)
            loader    = DataLoader(cache_dir, ttl_hours=0)   # TTL=0 → 강제 갱신
            for sym in self.settings.data.symbols:
                loader.load(sym, self.settings.data.period,
                            self.settings.data.interval)
        except Exception as e:
            logger.warning(f"데이터 새로고침 실패: {e}")
        self._run_prediction()

    # ══════════════════════════════════════════════════════════════════════════
    # 결과 표시
    # ══════════════════════════════════════════════════════════════════════════

    def _show_results(
        self,
        results:   list,
        portfolio: dict,
    ):
        self._predictions = results
        self._tree.delete(*self._tree.get_children())

        valid_cnt = sum(1 for r in results if r.get("error") is None)
        model_cnt = sum(1 for r in results
                        if r.get("error") is None
                        and r.get("horizons", {}).get("5", {}).get("action") != "HOLD")
        self._count_var.set(
            f"  총 {len(results)}개 종목  |  유효 예측 {valid_cnt}개  |  "
            f"매매신호(5d기준) {model_cnt}개"
        )

        for pred in results:
            self._insert_row(pred)

        # 예측 이력 자동 저장 (백그라운드)
        threading.Thread(
            target=self._log_predictions_to_history,
            args=(results,),
            daemon=True,
        ).start()

        # 포트폴리오 표시
        self._update_text_widget(self._port_text, portfolio.get("summary", ""))

        # 확률 차트 그리기
        self.frame.after(100, lambda: self._draw_prob_chart(results))

        # 5d primary 기준 집계
        def _action5(r):
            return r.get("horizons", {}).get("5", {}).get("action", "HOLD")
        n_buy  = sum(1 for r in results if _action5(r) == "BUY")
        n_sell = sum(1 for r in results if _action5(r) == "SELL")
        n_hold = sum(1 for r in results if _action5(r) == "HOLD")
        self._set_status(
            f"✅ 통합 예측 완료 (1d·3d·5d·20d)  |  "
            f"BUY {n_buy}  |  SELL {n_sell}  |  HOLD {n_hold}"
        )
        if self._on_complete:
            self.frame.after(0, self._on_complete)

    def _insert_row(self, pred: dict):
        """Treeview에 멀티-호라이즌 통합 예측 결과 행 삽입."""
        sym      = pred["symbol"]
        sym_disp = self._sym_label(sym)
        error    = pred.get("error")

        if error:
            self._tree.insert("", "end", iid=sym, tags=("ERROR",),
                              values=(sym_disp, "—", "오류", "—", "—", "—", "—", "—"))
            return

        price = pred.get("current_price", 0.0)
        price_str = f"{price:,.0f}원" if price >= 1000 else f"{price:.2f}"

        horizons = pred.get("horizons", {})

        def _h_cell(h_key: str) -> str:
            """horizon 셀 문자열: '📈 +2.31%' 형식."""
            h = horizons.get(h_key, {})
            d = h.get("direction", "NEUTRAL")
            r = h.get("return_pct", 0.0)
            icon = {"UP": "📈", "DOWN": "📉", "NEUTRAL": "↔"}.get(d, "↔")
            return f"{icon} {r:+.1f}%"

        h1  = _h_cell("1")
        h3  = _h_cell("3")
        h5  = _h_cell("5")
        h20 = _h_cell("20")

        # 5d 대표값 기준 신뢰도·추천
        h5_data = horizons.get("5", {})
        conf    = h5_data.get("confidence", "LOW")
        action  = h5_data.get("action",     "HOLD")

        conf_map   = {"HIGH": "HIGH ✅", "MEDIUM": "MED ⚡", "LOW": "LOW ⚠️"}
        action_map = {"BUY": "🟢 BUY", "SELL": "🔴 SELL",
                      "WATCH": "🟡 WATCH", "HOLD": "⚪ HOLD"}
        conf_str   = conf_map.get(conf, conf)
        action_str = action_map.get(action, action)

        # 행 색상
        if action == "BUY" and conf == "HIGH":
            tag = "UP_HIGH"
        elif action == "BUY":
            tag = "UP_MED"
        elif action == "SELL" and conf == "HIGH":
            tag = "DOWN_HIGH"
        elif action == "SELL":
            tag = "DOWN_MED"
        elif action == "WATCH":
            tag = "WATCH"
        else:
            tag = "NEUTRAL"

        self._tree.insert("", "end", iid=sym, tags=(tag,),
                          values=(sym_disp, price_str, h1, h3, h5, h20,
                                  conf_str, action_str))

    def _on_row_select(self, event):
        """행 선택 → 상세 설명 표시."""
        sel = self._tree.selection()
        if not sel:
            return
        sym = sel[0]
        pred = next((p for p in self._predictions if p["symbol"] == sym), None)
        if not pred:
            return

        # 상세 텍스트 — 멀티-호라이즌 통합 리포트
        lines = []
        name = self._get_name(sym)
        if pred.get("error"):
            lines.append(f"❌ {name} ({sym})")
            lines.append(f"오류: {pred['error']}")
        else:
            price = pred.get("current_price", 0.0)
            lines.append(f"══════ {name} ({sym}) 통합 예측 리포트 ══════")
            lines.append(f"현재가: {price:,.0f}원")
            as_of = pred.get("as_of_date")
            if as_of is not None:
                try:
                    lines.append(f"예측 기준일: {as_of.date()}")
                except Exception:
                    pass
            lines.append("")

            # 호라이즌별 상세
            lines.append("─── Multi-Horizon 예측 ───")
            lines.append(f"{'기간':<8} {'방향':<8} {'P(↑)':<8} {'수익률':<10} {'신뢰도':<8} {'추천'}")
            lines.append("─" * 58)
            horizons = pred.get("horizons", {})
            h_labels = {"1": "1거래일", "3": "3거래일", "5": "5거래일", "20": "20거래일"}
            for hk in ["1", "3", "5", "20"]:
                h = horizons.get(hk, {})
                d    = h.get("direction", "NEUTRAL")
                pu   = h.get("prob_up", 0.5)
                ret  = h.get("return_pct", 0.0)
                conf = h.get("confidence", "LOW")
                act  = h.get("action", "HOLD")
                icon = {"UP": "📈", "DOWN": "📉", "NEUTRAL": "↔"}.get(d, "↔")
                lbl  = h_labels.get(hk, hk)
                lines.append(
                    f"{lbl:<8} {icon+d:<8} {pu:.0%}    {ret:+.2f}%      "
                    f"{conf:<8} {act}"
                )

            lines.append("")
            # 외부환경 반영
            ms = pred.get("macro_sentiment", 0.0)
            mt = pred.get("macro_event_tags", "")
            if ms != 0.0 or mt:
                lines.append("─── 외부환경 반영 ───")
                mood = "🔵 호재" if ms > 0.1 else ("🔴 악재" if ms < -0.1 else "⚪ 중립")
                lines.append(f"시장 분위기: {mood}  (감성 점수: {ms:+.2f})")
                if mt:
                    lines.append(f"감지된 이벤트: {mt}")
                lines.append("")

            # 기준 정보
            bl = pred.get("baseline_lookahead", "?")
            lines.append(f"모델 기준 lookahead: {bl}일 → GBM 스케일링 적용")

        self._update_text_widget(self._detail_text, "\n".join(lines))

    # ══════════════════════════════════════════════════════════════════════════
    # 확률 막대 차트
    # ══════════════════════════════════════════════════════════════════════════

    def _draw_prob_chart(self, results: list):
        """유효 종목의 상승 확률을 수평 막대 차트로 그립니다."""
        canvas = self._chart_canvas
        canvas.update_idletasks()
        W = canvas.winfo_width()
        H = canvas.winfo_height()
        canvas.delete("all")

        if W < 50 or H < 50:
            return

        valid = [r for r in results if r.get("error") is None]
        if not valid:
            canvas.create_text(W // 2, H // 2, text="예측 결과 없음",
                               fill=_DIM, font=("맑은 고딕", 9))
            return

        # 최대 15개 표시
        valid = valid[:15]
        n     = len(valid)

        pad_l = 90
        pad_r = 10
        pad_t = 10
        bar_h = max(int((H - pad_t * 2) / n) - 4, 10)
        bar_w = W - pad_l - pad_r

        # 50% 기준선
        mid_x = pad_l + int(bar_w * 0.5)
        canvas.create_line(mid_x, pad_t, mid_x, H - pad_t,
                           fill="#45475a", dash=(3, 3))
        canvas.create_text(mid_x, H - pad_t + 2, text="50%",
                           fill=_DIM, font=("맑은 고딕", 7), anchor="n")

        for i, pred in enumerate(valid):
            y_top = pad_t + i * (bar_h + 4)
            y_bot = y_top + bar_h

            h5     = pred.get("horizons", {}).get("5", {})
            prob   = h5.get("prob_up", pred.get("prob_up", 0.5))
            action = h5.get("action", pred.get("action", "HOLD"))

            # 막대 색상
            color_map = {
                "BUY":   _GREEN,
                "SELL":  _RED,
                "WATCH": _YELLOW,
                "HOLD":  "#585b70",
            }
            clr = color_map.get(action, "#585b70")

            bar_end = pad_l + int(bar_w * prob)

            # 배경 막대 (0~100%)
            canvas.create_rectangle(pad_l, y_top, pad_l + bar_w, y_bot,
                                     fill="#2a2a3e", outline="")
            # 확률 막대
            canvas.create_rectangle(mid_x if prob >= 0.5 else bar_end,
                                     y_top,
                                     bar_end if prob >= 0.5 else mid_x,
                                     y_bot,
                                     fill=clr, outline="")

            # 종목 라벨 — 종목명 우선 표시
            raw_sym = pred["symbol"]
            sym_name = self._get_name(raw_sym)
            # 이름이 너무 길면 6글자로 자름
            if len(sym_name) > 6:
                sym_name = sym_name[:5] + "…"
            canvas.create_text(pad_l - 4, (y_top + y_bot) // 2,
                                text=sym_name, fill=_FG,
                                font=("맑은 고딕", 7), anchor="e")

            # 확률 텍스트
            canvas.create_text(bar_end + 4, (y_top + y_bot) // 2,
                                text=f"{prob:.0%}", fill=_FG,
                                font=("맑은 고딕", 7), anchor="w")

        # 제목
        canvas.create_text(W // 2, 4, text="상승 확률",
                           fill=_ACCENT, font=("맑은 고딕", 8, "bold"), anchor="n")

    # ══════════════════════════════════════════════════════════════════════════
    # 테이블 정렬
    # ══════════════════════════════════════════════════════════════════════════

    def _sort_by(self, col: str):
        """컬럼 헤더 클릭 → 정렬."""
        if self._sort_col == col:
            self._sort_rev = not self._sort_rev
        else:
            self._sort_col = col
            self._sort_rev = True

        def _h_ret(r, hk):
            return r.get("horizons", {}).get(hk, {}).get("return_pct", 0.0)

        col_key_fn = {
            "1일후":     lambda r: _h_ret(r, "1"),
            "3일후":     lambda r: _h_ret(r, "3"),
            "5일후":     lambda r: _h_ret(r, "5"),
            "20일후":    lambda r: _h_ret(r, "20"),
            "신뢰도(5d)": lambda r: {"HIGH": 3, "MEDIUM": 2, "LOW": 1}.get(
                r.get("horizons", {}).get("5", {}).get("confidence", "LOW"), 0),
            "종목":      lambda r: r.get("symbol", ""),
        }
        fn = col_key_fn.get(col)
        if not fn or not self._predictions:
            return

        valid   = [p for p in self._predictions if p.get("error") is None]
        invalid = [p for p in self._predictions if p.get("error") is not None]
        valid.sort(key=fn, reverse=self._sort_rev)

        self._tree.delete(*self._tree.get_children())
        for pred in valid + invalid:
            self._insert_row(pred)

    # ══════════════════════════════════════════════════════════════════════════
    # 헬퍼 / 상태 관리
    # ══════════════════════════════════════════════════════════════════════════

    def _update_progress(self, pct: float, msg: str):
        self._progress["value"] = pct * 100
        self._set_status(msg)

    def _set_status(self, msg: str):
        self._status_var.set(msg[:100])

    def _clear_results(self):
        self._tree.delete(*self._tree.get_children())
        self._count_var.set("")
        self._update_text_widget(self._port_text, "")
        self._update_text_widget(self._detail_text, "")
        self._chart_canvas.delete("all")

    def _update_text_widget(self, widget: scrolledtext.ScrolledText, text: str):
        widget.config(state="normal")
        widget.delete("1.0", "end")
        widget.insert("end", text)
        widget.config(state="disabled")

    def _on_done(self):
        self._run_btn.config(state="normal")
        self._stop_btn.config(state="disabled")

    # ─────────────────────────────────────────────
    # 예측 이력 자동 저장
    # ─────────────────────────────────────────────

    def _log_predictions_to_history(self, results: list) -> None:
        """멀티-호라이즌 예측 결과를 PredictionLogger에 append-only로 기록."""
        try:
            from history.logger import PredictionLogger
            from history.schema import PredictionRecord

            hist_dir = os.path.join(BASE_DIR, "outputs", "history")
            pl = PredictionLogger(hist_dir)

            saved = 0
            for pred in results:
                if pred.get("error") is not None:
                    continue
                sym = pred.get("symbol", "")
                if not sym:
                    continue
                try:
                    horizons = pred.get("horizons", {})
                    h5 = horizons.get("5", {})
                    rec = PredictionRecord(
                        symbol               = sym,
                        name                 = self._get_name(sym),
                        model_version        = pred.get("model_version", ""),
                        # primary (5d) 대표값
                        predicted_direction  = h5.get("direction", "NEUTRAL"),
                        prob_up              = float(h5.get("prob_up", 0.5)),
                        predicted_return_pct = float(h5.get("return_pct", 0.0)),
                        confidence           = h5.get("confidence", "LOW"),
                        action               = h5.get("action", "HOLD"),
                        price_at_prediction  = float(pred.get("current_price", 0.0)),
                        # 멀티-호라이즌 전체 저장
                        horizons             = {
                            k: {
                                "direction":  v.get("direction", "NEUTRAL"),
                                "prob_up":    round(float(v.get("prob_up", 0.5)), 4),
                                "return_pct": round(float(v.get("return_pct", 0.0)), 4),
                                "confidence": v.get("confidence", "LOW"),
                                "action":     v.get("action", "HOLD"),
                            }
                            for k, v in horizons.items()
                        },
                        horizon_days         = 5,
                        segment_length       = int(self.settings.roi.segment_length),
                        lookahead            = int(self.settings.roi.lookahead),
                        d_model              = int(self.settings.model.d_model),
                        macro_sentiment      = float(pred.get("macro_sentiment", 0.0)),
                        macro_event_tags     = pred.get("macro_event_tags", ""),
                    )
                    pl.append(rec)
                    saved += 1
                except Exception as _e:
                    logger.debug("prediction log skip %s: %s", sym, _e)

            if saved:
                logger.info("Prediction history: %d records saved (multi-horizon)", saved)
        except Exception as e:
            logger.warning("Failed to log predictions to history: %s", e)

    def refresh_symbols(self):
        """데이터 탭에서 종목 변경 시 호출 — 대상 종목 표시 업데이트."""
        self._update_target_symbols_label()

    # ── 종목명 캐시 ──────────────────────────────────────────────────────────

    def _get_name(self, sym: str) -> str:
        """종목코드 → 종목명 (캐시 적용)."""
        if sym not in self._name_cache:
            self._name_cache[sym] = get_name(sym)
        return self._name_cache[sym]

    def _sym_label(self, sym: str) -> str:
        """테이블에 표시할 '종목명 (코드)' 문자열."""
        name = self._get_name(sym)
        code = sym.split(".")[0]          # "005930.KS" → "005930"
        if name and name != sym:
            return f"{name} ({code})"
        return sym                         # fallback: 코드만

    def _update_target_symbols_label(self):
        """대상 종목 라벨을 현재 settings 기준으로 갱신."""
        try:
            syms = self.settings.data.symbols
            if not syms:
                self._target_var.set("대상 종목: 없음  (데이터 탭에서 추가하세요)")
                return
            names = [f"{self._get_name(s)}({s.split('.')[0]})" for s in syms]
            MAX = 6
            shown = names[:MAX]
            rest  = len(names) - MAX
            text  = "  |  ".join(shown) + (f"  +{rest}개" if rest > 0 else "")
            self._target_var.set(f"대상 ({len(syms)}개): " + text)
        except Exception:
            pass
