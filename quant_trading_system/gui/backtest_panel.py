# gui/backtest_panel.py — 백테스트 패널 (Walk-Forward, 데이터 누수 없음)
from __future__ import annotations
import os, sys, threading, tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import logging
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from config import AppSettings
from data import DataLoader
from data.korean_stocks import get_name
from features import ROIDetector, CVFeatureExtractor, TSFeatureExtractor
from models.hybrid import HybridModel
from models.trainer import ModelTrainer
from models.store import ModelStore
from signals import SignalGenerator
from portfolio import PortfolioConstructor
from backtest import BacktestEngine
from risk import RiskManager
from evaluation import PerformanceEvaluator
from gui.tooltip import add_tooltip
from gui.ui_meta import METRIC_DESCRIPTIONS
from backtest.session_store import BacktestSessionStore
from history.schema import BacktestSession

logger = logging.getLogger("quant.gui.backtest")


class BacktestPanel:
    def __init__(self, parent, settings: AppSettings, on_complete=None):
        self.settings = settings
        self._complete_cb = on_complete   # 외부 완료 콜백 (main_window용)
        # ↑ 주의: 이름을 _on_complete 로 하면 아래 def _on_complete(self) 메서드를
        #   인스턴스 속성이 가려(shadow)버려 버튼 재활성화가 안 되는 버그 발생.
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._pending_equity: "pd.Series | None" = None   # 캔버스 미배치 시 재시도용

        self.frame = ttk.Frame(parent)
        self._build()

    # ─────────────────────────────────────────────
    # UI 구성
    # ─────────────────────────────────────────────

    def _build(self):
        # ── 안내 배너 ────────────────────────────────
        banner = tk.Frame(self.frame, bg="#1a1a2e", pady=7)
        banner.pack(fill="x", padx=6, pady=(6, 2))
        tk.Label(
            banner,
            text="📈  3단계: 백테스트  —  전략의 과거 성과를 Walk-Forward로 검증하세요",
            bg="#1a1a2e", fg="#a6e3a1",
            font=("맑은 고딕", 10, "bold"),
        ).pack(side="left", padx=12)
        tk.Label(
            banner,
            text="  (데이터 누수 없음: 미래 데이터를 학습에 절대 사용하지 않습니다)",
            bg="#1a1a2e", fg="#9399b2",
            font=("맑은 고딕", 8),
        ).pack(side="left")

        # ── 대상 종목 표시 띠 ─────────────────────────
        sym_bar = tk.Frame(self.frame, bg="#0f1a2a", height=26)
        sym_bar.pack(fill="x", padx=6, pady=(0, 2))
        sym_bar.pack_propagate(False)
        tk.Label(sym_bar, text="🎯 백테스트 대상:", bg="#0f1a2a", fg="#89b4fa",
                 font=("맑은 고딕", 9, "bold")).pack(side="left", padx=(10, 6))
        self._sym_bar_var = tk.StringVar(value="")
        tk.Label(sym_bar, textvariable=self._sym_bar_var,
                 bg="#0f1a2a", fg="#cdd6f4",
                 font=("맑은 고딕", 9), anchor="w").pack(side="left", fill="x", expand=True)
        self._refresh_sym_bar()

        pane = tk.PanedWindow(self.frame, orient="horizontal",
                              bg="#1e1e2e", sashwidth=4)
        pane.pack(fill="both", expand=True, padx=6, pady=2)

        left  = ttk.LabelFrame(pane, text="백테스트 설정", padding=10)
        right = ttk.Frame(pane)
        pane.add(left,  minsize=220)
        pane.add(right, minsize=520)

        self._build_left(left)
        self._build_right(right)

    # ─────────────────────────────────────────────
    # 공개 API
    # ─────────────────────────────────────────────

    def refresh_symbols(self):
        """데이터 탭 종목 변경 시 main_window에서 호출 → 심볼 바 갱신"""
        self._refresh_sym_bar()

    def _refresh_sym_bar(self):
        """대상 종목 띠 업데이트 (종목명 + 코드)"""
        if not hasattr(self, "_name_cache"):
            self._name_cache: dict[str, str] = {}
        syms = self.settings.data.symbols
        if not syms:
            self._sym_bar_var.set("(등록된 종목 없음 — 데이터 탭에서 종목을 추가하세요)")
            return
        parts = []
        for s in syms:
            if s not in self._name_cache:
                self._name_cache[s] = get_name(s)
            name = self._name_cache[s]
            code = s.split(".")[0]
            parts.append(f"{name}({code})" if name and name != s else s)
        MAX = 6
        shown = parts[:MAX]
        rest  = len(parts) - MAX
        text  = "  |  ".join(shown) + (f"  +{rest}개" if rest > 0 else "")
        self._sym_bar_var.set(f"총 {len(syms)}개:  " + text)

    def _build_left(self, parent):
        row = 0

        # 툴팁 있는 레이블+입력 헬퍼
        def add(lbl, key, default, tooltip=""):
            nonlocal row
            label_w = ttk.Label(parent, text=lbl + ":")
            label_w.grid(row=row, column=0, sticky="w", padx=4, pady=3)
            var = tk.StringVar(value=str(default))
            self.param_vars[key] = var
            entry_w = ttk.Entry(parent, textvariable=var, width=14)
            entry_w.grid(row=row, column=1, sticky="w", padx=4)
            if tooltip:
                add_tooltip(label_w, tooltip)
                add_tooltip(entry_w, tooltip)
            row += 1

        def add_combo(lbl, key, default, choices, tooltip=""):
            nonlocal row
            label_w = ttk.Label(parent, text=lbl + ":")
            label_w.grid(row=row, column=0, sticky="w", padx=4, pady=3)
            var = tk.StringVar(value=default)
            self.param_vars[key] = var
            combo_w = ttk.Combobox(parent, textvariable=var, values=choices,
                                   width=14, state="readonly")
            combo_w.grid(row=row, column=1, sticky="w", padx=4)
            if tooltip:
                add_tooltip(label_w, tooltip)
                add_tooltip(combo_w, tooltip)
            row += 1

        self.param_vars: dict[str, tk.StringVar] = {}

        b = self.settings.backtest

        # ── 섹션: 자본 / 비용 ──────────────────────
        tk.Label(parent, text="▸ 자본 및 거래 비용",
                 bg="#1e1e2e", fg="#89b4fa",
                 font=("맑은 고딕", 9, "bold")).grid(
            row=row, column=0, columnspan=2, sticky="w", padx=4, pady=(4, 2))
        row += 1

        add("초기 자본 (원)", "capital", str(int(b.initial_capital)),
            "백테스트를 시작할 때의 가상 자본금입니다.\n실제 투자금이 아닌 시뮬레이션 금액입니다.\n\n기본: 1억원 (100,000,000)")

        add("거래 비용 (%)", "tx_cost", str(round(b.transaction_cost * 100, 4)),
            "주식을 사고팔 때 발생하는 수수료+세금 비율입니다.\n\n"
            "한국 주식 추천: 0.15%\n"
            "  (수수료 0.015% + 증권거래세 약 0.1~0.2%)\n\n"
            "이 값을 실제보다 낮게 설정하면 백테스트 결과가 과장됩니다.")

        add("슬리피지 (%)", "slippage", str(round(b.slippage * 100, 4)),
            "주문 가격과 실제 체결 가격의 차이입니다.\n\n"
            "대형주 추천: 0.05%\n"
            "소형주/유동성 낮은 종목: 0.1~0.3%\n\n"
            "이 값도 너무 낮게 설정하면 결과가 과대평가됩니다.")

        add("체결 지연 (일)", "delay", str(int(b.execution_delay)),
            "신호가 나온 다음 며칠 뒤에 실제 매매하는지 설정합니다.\n\n"
            "0 = 당일 체결 (비현실적)\n"
            "1 = 다음날 체결 (추천, 현실적)\n\n"
            "실제 투자에서는 신호 당일 즉시 체결이 어렵습니다.")

        ttk.Separator(parent, orient="horizontal").grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=6)
        row += 1

        # ── 섹션: Walk-Forward 기간 ──────────────
        tk.Label(parent, text="▸ Walk-Forward 기간 설정",
                 bg="#1e1e2e", fg="#89b4fa",
                 font=("맑은 고딕", 9, "bold")).grid(
            row=row, column=0, columnspan=2, sticky="w", padx=4, pady=(2, 2))
        row += 1

        tk.Label(parent,
                 text="학습→테스트 사이클을 슬라이딩하며 공정하게 검증합니다",
                 bg="#1e1e2e", fg="#585b70",
                 font=("맑은 고딕", 8), justify="left").grid(
            row=row, column=0, columnspan=2, sticky="w", padx=4)
        row += 1

        add("학습 윈도우 (거래일)", "train_days", str(int(b.wf_train_days)),
            "전략을 '가르치는' 기간입니다.\n\n"
            "504일 ≈ 2년치 거래일 (추천)\n"
            "252일 ≈ 1년\n\n"
            "너무 짧으면 학습 데이터가 부족하고,\n"
            "너무 길면 오래된 패턴에 과도하게 의존합니다.")

        add("테스트 윈도우 (거래일)", "test_days", str(int(b.wf_test_days)),
            "학습 후 '실제 미래처럼' 테스트하는 기간입니다.\n"
            "이 구간은 학습에 전혀 사용하지 않습니다.\n\n"
            "126일 ≈ 반년 (추천)\n"
            "63일  ≈ 분기 (빠른 검증)")

        add("슬라이딩 간격 (거래일)", "step_days", str(int(b.wf_step_days)),
            "각 학습→테스트 사이클을 얼마씩 앞으로 이동하며 반복할지입니다.\n\n"
            "63일 ≈ 3개월 (추천)\n"
            "간격이 짧을수록 더 많은 검증 구간이 생깁니다.")

        ttk.Separator(parent, orient="horizontal").grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=6)
        row += 1

        # ── 섹션: 전략 설정 ─────────────────────
        tk.Label(parent, text="▸ 전략 설정",
                 bg="#1e1e2e", fg="#89b4fa",
                 font=("맑은 고딕", 9, "bold")).grid(
            row=row, column=0, columnspan=2, sticky="w", padx=4, pady=(2, 2))
        row += 1

        add_combo("포트폴리오 방법", "method", self.settings.portfolio.method,
                  ["risk_parity", "mean_variance", "vol_scaling", "equal_weight"],
                  "각 종목에 얼마씩 투자할지 결정하는 방법입니다.\n\n"
                  "equal_weight  (균등): 모든 종목에 같은 비중 → 초보자 추천\n"
                  "risk_parity   (위험 균등): 위험도에 따라 비중 조절 → 안정적\n"
                  "mean_variance (최적화): 수익 최대·위험 최소 계산 → 고급\n"
                  "vol_scaling   (변동성 조정): 변동성 낮은 종목에 더 투자")

        add_combo("리밸런싱 주기", "rebal", self.settings.portfolio.rebalance_freq,
                  ["daily", "weekly", "monthly"],
                  "포트폴리오 비중을 목표치로 다시 맞추는 주기입니다.\n\n"
                  "daily   (매일): 거래 비용이 누적될 수 있음\n"
                  "weekly  (매주): 균형 잡힌 선택 (추천)\n"
                  "monthly (매월): 장기 투자, 비용 절약")

        add_combo("신호 방법", "signal", "ranking",
                  ["ranking", "threshold", "prob_weight"],
                  "여러 종목의 예측 신호를 어떻게 순위 매길지 결정합니다.\n\n"
                  "ranking     : 예측 수익률 순위 기반 (추천)\n"
                  "threshold   : 일정 수준 이상 신호만 선택\n"
                  "prob_weight : 예측 확률에 비례한 가중치")

        add_combo("신호 소스", "sig_src", "momentum",
                  ["momentum", "model"],
                  "매매 신호를 어디서 가져올지 선택합니다.\n\n"
                  "momentum (모멘텀): AI 없이 과거 수익률 기반 신호 → 바로 사용 가능\n"
                  "model    (AI 모델): 학습 탭에서 훈련한 모델의 예측 신호\n\n"
                  "⚠️ 'model' 선택 시 먼저 학습 탭에서 모델을 학습해야 합니다!")

        ttk.Separator(parent, orient="horizontal").grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=8)
        row += 1

        # 도움말 메모
        help_fr = tk.Frame(parent, bg="#181825", padx=6, pady=6)
        help_fr.grid(row=row, column=0, columnspan=2, sticky="ew", padx=4, pady=(0, 8))
        tk.Label(help_fr,
                 text="💡 처음 사용하신다면:\n"
                      "  신호 소스 = 모멘텀 을 선택하고 바로 실행해 보세요!\n"
                      "  AI 모델 없이도 백테스트를 실행할 수 있습니다.",
                 bg="#181825", fg="#a6adc8",
                 font=("맑은 고딕", 8), justify="left").pack(anchor="w")
        row += 1

        btn_fr = ttk.Frame(parent)
        btn_fr.grid(row=row, column=0, columnspan=2, pady=(4, 0))
        self.run_btn = ttk.Button(btn_fr, text="▶ 백테스트 실행",
                                  command=self._run_backtest,
                                  style="Accent.TButton")
        self.run_btn.pack(side="left", padx=(0, 4))
        add_tooltip(self.run_btn,
                    "설정한 파라미터로 Walk-Forward 백테스트를 시작합니다.\n"
                    "완료까지 수십 초~수 분이 소요될 수 있습니다.")
        self.stop_btn = ttk.Button(btn_fr, text="⏹ 중단",
                                   command=lambda: self._stop_event.set(),
                                   style="Danger.TButton", state="disabled")
        self.stop_btn.pack(side="left")
        row += 1

        self.progress = ttk.Progressbar(parent, mode="determinate")
        self.progress.grid(row=row, column=0, columnspan=2,
                           sticky="ew", padx=4, pady=(8, 0))
        row += 1
        self.status_var = tk.StringVar(value="준비 — '▶ 백테스트 실행'을 누르세요")
        ttk.Label(parent, textvariable=self.status_var,
                  font=("맑은 고딕", 9)).grid(
            row=row, column=0, columnspan=2, sticky="w", padx=4)

    def _build_right(self, parent):
        nb = ttk.Notebook(parent)
        nb.pack(fill="both", expand=True)

        # 탭 1: 성과 요약
        sum_fr = ttk.Frame(nb)
        nb.add(sum_fr, text="  📊 성과 요약  ")
        self._build_summary(sum_fr)

        # 탭 2: 자산 곡선
        chart_fr = ttk.Frame(nb)
        nb.add(chart_fr, text="  📈 자산 곡선  ")
        self._build_equity_chart(chart_fr)

        # 탭 3: WF 상세
        wf_fr = ttk.Frame(nb)
        nb.add(wf_fr, text="  🔄 Walk-Forward  ")
        self._build_wf_tab(wf_fr)

        # 탭 4: 로그
        log_fr = ttk.Frame(nb)
        nb.add(log_fr, text="  📝 로그  ")
        self.log_box = scrolledtext.ScrolledText(
            log_fr, bg="#181825", fg="#a6adc8",
            font=("Consolas", 9), relief="flat", state="disabled",
        )
        self.log_box.pack(fill="both", expand=True, padx=4, pady=4)

    def _build_summary(self, parent):
        items = [
            ("총 수익률",   "total_return",       "pct"),
            ("연간 CAGR",   "cagr",               "pct"),
            ("연간 변동성", "annual_volatility",   "pct"),
            ("최대낙폭",    "max_drawdown",        "pct"),
            ("Sharpe",      "sharpe_ratio",        "num"),
            ("Sortino",     "sortino_ratio",       "num"),
            ("Calmar",      "calmar_ratio",        "num"),
            ("승률",        "win_rate",            "pct"),
            ("손익비",      "payoff_ratio",        "num"),
            ("VaR 95%",     "var_95",              "pct"),
            ("CVaR 95%",    "cvar_95",             "pct"),
            ("알파",        "alpha",               "pct"),
            ("베타",        "beta",                "num"),
            ("정보비율",    "information_ratio",   "num"),
            ("Walk-Forward 윈도우 수", "n_windows", "int"),
        ]
        self.metric_vars: dict = {}
        self.metric_labels: dict = {}

        # ── 수치 그리드 ────────────────────────────
        fr = ttk.Frame(parent)
        fr.pack(fill="x", padx=10, pady=(10, 4))

        for i, (lbl, key, fmt) in enumerate(items):
            col = i % 2
            r   = i // 2
            meta_d = METRIC_DESCRIPTIONS.get(key, {})
            tooltip_text = ""
            if meta_d:
                parts = [meta_d.get("simple", "")]
                if meta_d.get("good"):
                    parts.append(f"✅ {meta_d['good']}")
                if meta_d.get("warn"):
                    parts.append(f"⚠️ {meta_d['warn']}")
                tooltip_text = "\n".join(p for p in parts if p)

            name_lbl = tk.Label(fr, text=lbl + ":", bg="#1e1e2e", fg="#9399b2",
                                font=("맑은 고딕", 10), anchor="e", width=18)
            name_lbl.grid(row=r, column=col*2, sticky="e", padx=(8, 4), pady=3)

            var = tk.StringVar(value="—")
            self.metric_vars[key] = (var, fmt)
            val_lbl = tk.Label(fr, textvariable=var, bg="#1e1e2e", fg="#cdd6f4",
                               font=("맑은 고딕", 10, "bold"), anchor="w", width=14)
            val_lbl.grid(row=r, column=col*2+1, sticky="w", padx=(0, 16), pady=3)
            self.metric_labels[key] = val_lbl

            if tooltip_text:
                add_tooltip(name_lbl, tooltip_text)
                add_tooltip(val_lbl,  tooltip_text)

        # 과적합 경고 레이블
        n_rows = len(items) // 2 + 1
        self.overfit_var = tk.StringVar(value="")
        tk.Label(fr, textvariable=self.overfit_var, bg="#1e1e2e",
                 fg="#f38ba8", font=("맑은 고딕", 11, "bold")).grid(
            row=n_rows, column=0, columnspan=4,
            pady=(12, 0), sticky="w", padx=8)

        ttk.Separator(parent, orient="horizontal").pack(fill="x", padx=8, pady=4)

        # ── 초보자 친화적 해석 패널 ────────────────
        interp_fr = ttk.LabelFrame(parent, text="📖 결과 해석 가이드", padding=8)
        interp_fr.pack(fill="x", padx=8, pady=(0, 8))

        self.interp_var = tk.StringVar(
            value="백테스트를 실행하면 결과 해석이 여기에 표시됩니다.")
        self.interp_lbl = tk.Label(
            interp_fr,
            textvariable=self.interp_var,
            bg="#1e1e2e", fg="#a6adc8",
            font=("맑은 고딕", 9),
            justify="left",
            wraplength=680,
            anchor="nw",
        )
        self.interp_lbl.pack(fill="x", anchor="nw")

    def _build_equity_chart(self, parent):
        self.equity_canvas = tk.Canvas(parent, bg="#181825",
                                       highlightthickness=0)
        self.equity_canvas.pack(fill="both", expand=True, padx=4, pady=4)
        # 탭이 숨겨진 채로 백테스트가 완료될 경우 캔버스가 1×1 → 재시도용 저장소
        self._pending_equity: "pd.Series | None" = None
        self.equity_canvas.bind("<Configure>", self._on_equity_canvas_configure)

    def _build_wf_tab(self, parent):
        cols = ("윈도우", "학습시작", "학습종료", "테스트시작", "테스트종료",
                "CAGR", "MDD", "Sharpe", "Sortino", "승률")
        self.wf_tree = ttk.Treeview(parent, columns=cols,
                                    show="headings", height=22)
        for c in cols:
            self.wf_tree.heading(c, text=c)
            self.wf_tree.column(c, width=90, anchor="center")
        self.wf_tree.tag_configure("pos", foreground="#a6e3a1")
        self.wf_tree.tag_configure("neg", foreground="#f38ba8")

        vsb = ttk.Scrollbar(parent, orient="vertical", command=self.wf_tree.yview)
        self.wf_tree.configure(yscrollcommand=vsb.set)
        self.wf_tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        # 키보드 이동 시 포커스/선택 유지
        for key in ("<Up>", "<Down>", "<Prior>", "<Next>"):
            self.wf_tree.bind(key, self._on_tree_keypress)

    # ─────────────────────────────────────────────
    # 키보드 이동 핸들러
    # ─────────────────────────────────────────────

    def _on_tree_keypress(self, event):
        """Treeview 키보드 ↑↓ 이동 시 focused 항목을 선택으로 동기화"""
        tree = event.widget
        def _sync():
            focused = tree.focus()
            if focused:
                tree.selection_set(focused)
        self.frame.after(30, _sync)

    # ─────────────────────────────────────────────
    # 백테스트 실행
    # ─────────────────────────────────────────────

    def _run_backtest(self):
        if self._thread and self._thread.is_alive():
            messagebox.showinfo("알림", "이미 실행 중입니다.")
            return
        if not self.settings.data.symbols:
            messagebox.showwarning("경고",
                "등록된 종목이 없습니다.\n데이터 탭에서 종목을 추가하고 다운로드하세요.")
            return

        self._stop_event.clear()
        self.run_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.progress["value"] = 0
        self.wf_tree.delete(*self.wf_tree.get_children())
        for key, (var, _) in self.metric_vars.items():
            var.set("—")
        self.overfit_var.set("")

        self._thread = threading.Thread(target=self._backtest_thread, daemon=True)
        self._thread.start()

    # ─────────────────────────────────────────────
    # 백테스트 스레드 (누수 없음)
    # ─────────────────────────────────────────────

    def _backtest_thread(self):
        try:
            params   = self._get_params()
            symbols  = self.settings.data.symbols
            sig_src  = self.param_vars["sig_src"].get()

            # ── 1. 데이터 로드 ──────────────────────
            self._set_status("데이터 로드 중...")
            cache_dir = os.path.join(BASE_DIR, self.settings.data.cache_dir)
            loader    = DataLoader(cache_dir)
            store     = ModelStore(os.path.join(BASE_DIR, self.settings.model_dir))

            raw_data: dict[str, pd.DataFrame] = {}
            for sym in symbols:
                df = loader.load(sym, self.settings.data.period,
                                 self.settings.data.interval)
                if df is not None and len(df) >= 60:
                    raw_data[sym] = df
                    self._log(f"로드: {sym} ({get_name(sym)}) — {len(df)}행")
                else:
                    self._log(f"건너뜀: {sym} ({get_name(sym)}) — 데이터 부족")

            if not raw_data:
                self._log("❌ 사용 가능한 데이터가 없습니다. 데이터 탭에서 다운로드하세요.")
                return

            available_syms = list(raw_data.keys())
            names_str = "  |  ".join(
                f"{get_name(s)} ({s.split('.')[0]})" for s in available_syms
            )
            self._log(f"\n사용 종목 {len(available_syms)}개:\n  {names_str}")

            # ── 2. 종가 행렬 구성 ───────────────────
            closes = {s: raw_data[s]["Close"] for s in available_syms}
            price_df = (pd.DataFrame(closes)
                        .sort_index()
                        .ffill(limit=5)
                        .dropna(how="all"))

            if len(price_df) < params["train_days"] + params["test_days"]:
                self._log(f"❌ 데이터 기간 부족: {len(price_df)}일 "
                          f"(필요: {params['train_days']+params['test_days']}일 이상)\n"
                          "  기간을 줄이거나 더 오래된 데이터를 다운로드하세요.")
                return

            self._log(f"가격 행렬: {len(price_df)}일 × {len(available_syms)}종목")

            # ── 3. 전략 함수 생성기 정의 ────────────
            # strategy_fn(train_data: DataFrame) → weights_fn
            # ⚠️ 미래 참조 금지: train_data만 사용

            roi_det = ROIDetector(
                segment_length=self.settings.roi.segment_length,
                lookahead=self.settings.roi.lookahead,
            )
            cv_ext  = CVFeatureExtractor(image_size=self.settings.model.image_size)
            ts_ext  = TSFeatureExtractor(n_features=32)
            sig_gen = SignalGenerator(method=self.param_vars["signal"].get())
            port_ctor = PortfolioConstructor(
                method=self.param_vars["method"].get(),
                max_weight=self.settings.portfolio.max_weight,
                target_vol=self.settings.portfolio.target_volatility,
            )
            risk_mgr = RiskManager(
                vol_target=self.settings.risk.vol_target,
                max_drawdown_limit=self.settings.risk.max_drawdown_limit,
                kill_switch_sharpe=self.settings.risk.kill_switch_sharpe,
            )

            def strategy_fn(train_data: pd.DataFrame):
                """
                학습 윈도우 데이터만으로 신호·비중 계산
                → 고정 비중 반환 (테스트 기간 동안 동일 전략 적용)
                ⚠️  train_data 외부 데이터 절대 참조 금지
                """
                # 학습 기간 수익률 (비중 계산용)
                train_ret = train_data.pct_change().dropna()

                preds: dict[str, tuple[float, float]] = {}
                for sym in available_syms:
                    if sym not in train_data.columns:
                        continue
                    col_ret = train_ret[sym].dropna() if sym in train_ret.columns else pd.Series(dtype=float)

                    # ── 신호 소스: 모델 or 모멘텀 ──────
                    if sig_src == "model" and store.has_model(sym):
                        try:
                            sym_df = raw_data[sym]
                            # 학습 윈도우 내 데이터만 사용
                            sym_train = sym_df[sym_df.index.isin(train_data.index)]
                            if len(sym_train) < 60:
                                raise ValueError("학습 데이터 부족")
                            segs, _, _ = roi_det.extract_segments(sym_train)
                            if len(segs) == 0:
                                raise ValueError("ROI 없음")
                            use_n = min(10, len(segs))
                            imgs  = cv_ext.transform(segs[-use_n:])
                            ts_f  = ts_ext.transform(segs[-use_n:])

                            # 저장된 아키텍처 설정 읽기 (UI 설정과 불일치 방지)
                            import torch as _torch
                            _ckpt_path = os.path.join(store._sym_dir(sym), "latest.pt")
                            _cfg = {}
                            if os.path.exists(_ckpt_path):
                                try:
                                    _peeked = _torch.load(_ckpt_path, map_location="cpu",
                                                          weights_only=False)
                                    _cfg = _peeked.get("config", {})
                                except Exception:
                                    pass

                            _ts_mode   = _cfg.get("ts_mode", "scalar")
                            _seg_len   = int(segs.shape[1])
                            _ts_indim  = (int(segs.shape[2])      # 5 (segment mode)
                                          if _ts_mode == "segment"
                                          else _cfg.get("ts_input_dim", 32))
                            mdl = HybridModel(
                                img_in_channels=imgs.shape[1],
                                ts_input_dim=_ts_indim,
                                cnn_out_dim=_cfg.get("cnn_out_dim", 128),
                                d_model=_cfg.get("d_model", self.settings.model.d_model),
                                nhead=_cfg.get("nhead", self.settings.model.nhead),
                                num_encoder_layers=_cfg.get("num_encoder_layers",
                                                            self.settings.model.num_encoder_layers),
                                dim_feedforward=_cfg.get("dim_feedforward",
                                                         self.settings.model.dim_feedforward),
                                dropout=_cfg.get("dropout", self.settings.model.dropout),
                                max_seq_len=_seg_len + 10,
                            )
                            import torch as _t2
                            _bt_device = "cuda" if _t2.cuda.is_available() else "cpu"
                            ckpt = store.load(mdl, sym, device=_bt_device)
                            if ckpt:
                                trainer = ModelTrainer(mdl, device=_bt_device)
                                # ts_mode에 따라 올바른 입력 전달
                                _ts_input = (segs[-use_n:].astype(np.float32)
                                             if _ts_mode == "segment" else ts_f)
                                mu_arr, sigma_arr = trainer.predict(imgs, _ts_input)
                                preds[sym] = (float(mu_arr.mean()),
                                              float(sigma_arr.mean()))
                                continue
                        except Exception as e:
                            logger.debug(f"{sym} 모델 예측 실패: {e}")

                    # ── Fallback: 모멘텀 신호 ────────────
                    # 학습 윈도우 내 수익률/변동성만 사용
                    if len(col_ret) > 0:
                        mu_est  = col_ret.mean() * 252
                        vol_est = col_ret.std() * np.sqrt(252) if len(col_ret) > 1 else 0.2
                        vol_est = max(vol_est, 0.01)
                    else:
                        mu_est, vol_est = 0.0, 0.2
                    preds[sym] = (mu_est, vol_est)

                if not preds:
                    # 빈 전략 (현금 100%)
                    return lambda date, hist: {}

                signal_df = sig_gen.generate_from_dict(preds)

                # ⚠️  포트폴리오 구성 시 train_ret만 사용 (테스트 기간 참조 금지)
                weights = port_ctor.construct(signal_df, train_ret)

                # 리스크 조정 (변동성 타겟팅 + 킬스위치)
                port_ret = train_ret[list(weights.keys())].mean(axis=1) if weights else pd.Series(dtype=float)
                weights = risk_mgr.adjust_weights(weights, port_ret, 1.0)
                if not weights:
                    # 킬스위치 발동 → 현금 보유
                    return lambda date, hist: {}

                # 고정 비중 반환 클로저
                _fixed_weights = dict(weights)
                def weights_fn(date, hist_data):
                    return _fixed_weights

                return weights_fn

            # ── 4. Walk-Forward 백테스트 ────────────
            self._log(f"\nWalk-Forward 시작 "
                      f"(학습 {params['train_days']}일 / "
                      f"테스트 {params['test_days']}일 / "
                      f"스텝 {params['step_days']}일)")

            engine = BacktestEngine(
                initial_capital=params["capital"],
                transaction_cost=params["tx_cost"] / 100,
                slippage=params["slippage"] / 100,
                execution_delay=params["delay"],
                rebalance_freq=self.param_vars["rebal"].get(),
            )

            def progress_cb(pct: float, msg: str):
                self.frame.after(0, lambda p=pct, m=msg: self._update_progress(p, m))

            result = engine.run_walk_forward(
                price_df=price_df,
                strategy_fn=strategy_fn,
                train_days=params["train_days"],
                test_days=params["test_days"],
                step_days=params["step_days"],
                progress_cb=progress_cb,
            )

            # ── 5. 결과 표시 ────────────────────────
            if result.equity_curve.empty:
                self._log("❌ 백테스트 결과 없음 — 데이터 기간이 너무 짧거나 종목이 부족합니다.")
                return

            n_wins = len(result.walk_forward_results)
            self._log(f"\nWalk-Forward 완료: {n_wins}개 윈도우")
            evaluator = PerformanceEvaluator()
            report    = evaluator.format_report(result.metrics)
            self._log(report)

            self.frame.after(0, lambda: self._show_results(result))
            if self._complete_cb:
                self.frame.after(0, self._complete_cb)

            # ── 6. 세션 자동 저장 (감사 추적) ──────────────────────────────────
            try:
                self._auto_save_session(result, params, symbols)
            except Exception as _se:
                self._log(f"⚠️  세션 저장 실패 (결과에는 영향 없음): {_se}")

        except Exception as e:
            import traceback
            self._log(f"\n❌ 백테스트 오류:\n{traceback.format_exc()}")
        finally:
            self.frame.after(0, self._on_complete)

    # ─────────────────────────────────────────────
    # 결과 표시
    # ─────────────────────────────────────────────

    def _show_results(self, result):
        m = result.metrics

        # 성과 지표
        for key, (var, fmt) in self.metric_vars.items():
            v = m.get(key)
            if v is None:
                var.set("—")
            elif fmt == "pct":
                clr = "#a6e3a1" if (isinstance(v, float) and v >= 0) else "#f38ba8"
                var.set(f"{v:.2%}")
                if key in self.metric_labels:
                    self.metric_labels[key].config(fg=clr)
            elif fmt == "int":
                var.set(str(int(v)))
            else:
                var.set(f"{v:.4f}")

        # 과적합 경고
        self.overfit_var.set(
            "⚠️  과적합 경고: OOS 성과 크게 저하됨"
            if m.get("overfit_warning") else ""
        )

        # 초보자 친화적 결과 해석 생성
        self._update_interpretation(m)

        # 자산 곡선
        self._draw_equity_curve(result.equity_curve)

        # WF 상세
        self.wf_tree.delete(*self.wf_tree.get_children())
        for i, wf in enumerate(result.walk_forward_results, 1):
            sharpe = wf.get("sharpe_ratio", 0) or 0
            tag = "pos" if sharpe >= 0 else "neg"
            self.wf_tree.insert("", "end", tags=(tag,), values=(
                i,
                wf.get("train_start", "—"),
                wf.get("train_end",   "—"),
                wf.get("test_start",  "—"),
                wf.get("test_end",    "—"),
                f"{wf.get('cagr', 0) or 0:.2%}",
                f"{wf.get('max_drawdown', 0) or 0:.2%}",
                f"{wf.get('sharpe_ratio', 0) or 0:.3f}",
                f"{wf.get('sortino_ratio', 0) or 0:.3f}",
                f"{wf.get('win_rate', 0) or 0:.1%}",
            ))

    # ─────────────────────────────────────────────
    # 백테스트 세션 자동 저장 (감사 추적)
    # ─────────────────────────────────────────────

    def _auto_save_session(self, result, params: dict, symbols: list) -> None:
        """BacktestResult → BacktestSession → BacktestSessionStore 자동 저장."""
        store_dir = os.path.join(BASE_DIR, "outputs", "backtest_sessions")
        store = BacktestSessionStore(store_dir)
        m = result.metrics

        eq = result.equity_curve
        trades_list = []
        for t in result.trades:
            try:
                trades_list.append({
                    "date":       str(t.date.date()) if hasattr(t.date, "date") else str(t.date),
                    "symbol":     t.symbol,
                    "action":     t.action,
                    "shares":     float(t.shares),
                    "price":      float(t.price),
                    "commission": float(t.commission),
                    "pnl_pct":    float(getattr(t, "pnl_pct", 0.0)),
                })
            except Exception:
                continue

        session = BacktestSession(
            strategy_config  = {
                "signal_method": self.param_vars.get("signal", tk.StringVar()).get(),
                "signal_source": self.param_vars.get("sig_src", tk.StringVar()).get(),
                "method":        self.param_vars.get("method", tk.StringVar()).get(),
                "rebal":         self.param_vars.get("rebal",  tk.StringVar()).get(),
            },
            capital          = params["capital"],
            transaction_cost = params["tx_cost"] / 100,
            slippage         = params["slippage"] / 100,
            execution_delay  = int(params["delay"]),
            train_days       = int(params["train_days"]),
            test_days        = int(params["test_days"]),
            step_days        = int(params["step_days"]),
            symbols          = symbols,
            total_return_pct = float(m.get("total_return", 0) or 0) * 100,
            cagr             = float(m.get("cagr", 0) or 0) * 100,
            max_drawdown     = float(m.get("max_drawdown", 0) or 0) * 100,
            sharpe           = float(m.get("sharpe_ratio", 0) or 0),
            sortino          = float(m.get("sortino_ratio", 0) or 0),
            win_rate         = float(m.get("win_rate", 0) or 0),
            n_windows        = int(m.get("n_windows", 0) or 0),
            n_trades         = len(trades_list),
            equity_curve     = [float(v) for v in eq.values],
            equity_dates     = [str(d.date()) for d in eq.index] if hasattr(eq.index[0], "date") else [str(d) for d in eq.index],
            wf_results       = result.walk_forward_results or [],
            trades           = trades_list,
        )
        sid = store.save(session)
        self._log(f"✅ 백테스트 세션 저장 완료 → {sid[:8]}...  (outputs/backtest_sessions/)")

    def _update_interpretation(self, m: dict):
        """백테스트 결과를 초보자 언어로 해석하여 interp_lbl에 표시"""
        lines = []

        cagr      = m.get("cagr", 0) or 0
        sharpe    = m.get("sharpe_ratio", 0) or 0
        mdd       = m.get("max_drawdown", 0) or 0
        win_rate  = m.get("win_rate", 0) or 0
        n_wins    = m.get("n_windows", 0) or 0
        total_ret = m.get("total_return", 0) or 0
        overfit   = m.get("overfit_warning", False)

        # ① 전체 수익 요약
        ret_emoji = "📈" if total_ret >= 0 else "📉"
        lines.append(
            f"{ret_emoji} 이 전략은 테스트 기간 동안 총 {total_ret:.1%} 수익을 기록했습니다."
        )

        # ② CAGR 대 시장 비교
        if cagr >= 0.15:
            lines.append(f"  연평균 {cagr:.1%} 수익 — 코스피 평균(약 8~10%)을 크게 웃돌았습니다! 🎯")
        elif cagr >= 0.08:
            lines.append(f"  연평균 {cagr:.1%} 수익 — 코스피 평균 수준의 성과를 냈습니다.")
        elif cagr >= 0:
            lines.append(f"  연평균 {cagr:.1%} 수익 — 플러스이지만 시장 평균보다 다소 낮습니다.")
        else:
            lines.append(f"  연평균 {cagr:.1%} 손실 ⚠️ — 이 설정에서는 수익을 내지 못했습니다.")

        # ③ 위험 평가
        mdd_abs = abs(mdd)
        if mdd_abs <= 0.10:
            lines.append(f"  최대 낙폭 {mdd:.1%} — 매우 안정적인 전략입니다. ✅")
        elif mdd_abs <= 0.20:
            lines.append(f"  최대 낙폭 {mdd:.1%} — 적당한 수준의 손실이 있었습니다.")
        elif mdd_abs <= 0.35:
            lines.append(f"  최대 낙폭 {mdd:.1%} ⚠️ — 꽤 큰 손실 구간이 있었습니다. 위험 관리가 필요합니다.")
        else:
            lines.append(f"  최대 낙폭 {mdd:.1%} 🚨 — 매우 큰 낙폭입니다. 실제 투자 시 매우 주의하세요!")

        # ④ 샤프 지수 해석
        if sharpe >= 2.0:
            lines.append(f"  샤프 지수 {sharpe:.2f} — 위험 대비 수익이 훌륭합니다! 🌟")
        elif sharpe >= 1.0:
            lines.append(f"  샤프 지수 {sharpe:.2f} — 위험 대비 수익이 양호합니다.")
        elif sharpe >= 0.5:
            lines.append(f"  샤프 지수 {sharpe:.2f} — 보통 수준의 효율입니다.")
        else:
            lines.append(f"  샤프 지수 {sharpe:.2f} ⚠️ — 위험 대비 수익이 낮습니다.")

        # ⑤ 승률
        if win_rate >= 0.55:
            lines.append(f"  승률 {win_rate:.0%} — 거래의 절반 이상이 수익으로 마감되었습니다.")
        elif win_rate >= 0.45:
            lines.append(f"  승률 {win_rate:.0%} — 수익/손실 거래가 비슷한 비율이었습니다.")
        else:
            lines.append(f"  승률 {win_rate:.0%} ⚠️ — 손실 거래가 더 많았습니다 (손익비 확인 권장).")

        # ⑥ 신뢰도
        if n_wins >= 5:
            lines.append(f"  검증 구간 {int(n_wins)}개 — 결과의 신뢰도가 충분합니다.")
        elif n_wins >= 3:
            lines.append(f"  검증 구간 {int(n_wins)}개 — 어느 정도 신뢰할 수 있습니다.")
        else:
            lines.append(f"  검증 구간 {int(n_wins)}개 ⚠️ — 구간이 너무 적어 결과 신뢰도가 낮습니다. 데이터 기간을 늘려보세요.")

        # ⑦ 과적합 경고
        if overfit:
            lines.append(
                "\n⚠️ 과적합 경고: 학습 구간에서는 좋았지만 실제 테스트 구간에서 성과가 크게 떨어졌습니다."
                "\n   파라미터를 단순하게 줄이거나 데이터를 늘려보세요."
            )

        # ⑧ 종합 평가
        score = 0
        if cagr >= 0.10: score += 2
        elif cagr >= 0.05: score += 1
        if sharpe >= 1.0: score += 2
        elif sharpe >= 0.5: score += 1
        if mdd_abs <= 0.20: score += 2
        elif mdd_abs <= 0.35: score += 1
        if not overfit: score += 1

        grades = {
            (7, 8): "⭐⭐⭐ 매우 우수",
            (5, 6): "⭐⭐ 양호",
            (3, 4): "⭐ 보통",
            (0, 2): "개선 필요",
        }
        grade_str = next(
            (v for (lo, hi), v in grades.items() if lo <= score <= hi),
            "—"
        )
        lines.append(f"\n종합 평가: {grade_str}  (점수: {score}/8)")
        lines.append("  ※ 백테스트 결과는 과거 성과이며 미래를 보장하지 않습니다.")

        self.interp_var.set("\n".join(lines))

    def _on_equity_canvas_configure(self, event=None):
        """탭 전환·리사이즈 시 pending 자산곡선을 다시 그린다."""
        if self._pending_equity is not None and len(self._pending_equity) >= 2:
            self._draw_equity_curve(self._pending_equity)

    def _draw_equity_curve(self, equity: pd.Series, _retry: int = 0):
        """자산 곡선 그리기. 캔버스 크기가 아직 확정 안 됐으면 최대 10회 재시도."""
        # pending 에 저장해 두면 탭 전환 시에도 재그릴 수 있음
        self._pending_equity = equity

        canvas = self.equity_canvas
        canvas.update_idletasks()
        W, H = canvas.winfo_width(), canvas.winfo_height()
        canvas.delete("all")

        if W < 50 or H < 50:
            # 캔버스가 아직 배치되지 않은 경우 → 잠시 후 재시도 (최대 10회, 2초)
            if _retry < 10:
                self.frame.after(200, lambda: self._draw_equity_curve(equity, _retry + 1))
            return

        if len(equity) < 2:
            return

        pad = 55
        vals = equity.values
        vmin, vmax = vals.min(), vals.max()
        vrange = max(vmax - vmin, 1)
        n = len(vals)

        def xc(i): return pad + (W - 2*pad) * i / (n-1)
        def yc(v): return H - pad - (H - 2*pad) * (v - vmin) / vrange

        # 격자 + Y축
        for frac in [0, 0.25, 0.5, 0.75, 1.0]:
            val = vmin + vrange * frac
            y   = yc(val)
            canvas.create_line(pad, y, W-pad, y, fill="#313244", dash=(2, 4))
            canvas.create_text(pad-4, y, text=f"{val/1e6:.1f}M",
                               fill="#9399b2", font=("Consolas", 8), anchor="e")

        # 손익분기선
        base_y = yc(equity.iloc[0])
        canvas.create_line(pad, base_y, W-pad, base_y,
                           fill="#45475a", dash=(4, 2), width=1)

        # 드로우다운 음영 (Tkinter는 8자리 hex 색상 미지원 → stipple로 반투명 효과)
        equity_norm = (equity / equity.cummax() - 1)
        for i in range(n):
            if equity_norm.iloc[i] < -0.05:
                x = xc(i)
                canvas.create_line(x, base_y, x, yc(vals[i]),
                                   fill="#f38ba8", width=1,
                                   stipple="gray25")

        # 자산 곡선 라인
        pts = []
        for i in range(n):
            pts.extend([xc(i), yc(vals[i])])
        if len(pts) >= 4:
            canvas.create_line(*pts, fill="#a6e3a1", width=2.5, smooth=True)

        # X축 날짜
        dates = equity.index
        for frac in [0, 0.25, 0.5, 0.75, 1.0]:
            i = min(int(frac * (n-1)), n-1)
            canvas.create_text(xc(i), H-pad+14,
                               text=str(dates[i].date()),
                               fill="#9399b2", font=("Consolas", 8))

        # 제목
        total_ret = vals[-1] / vals[0] - 1
        color = "#a6e3a1" if total_ret >= 0 else "#f38ba8"
        canvas.create_text(W//2, 16,
                           text=f"초기 {vals[0]/1e6:.1f}M → 최종 {vals[-1]/1e6:.1f}M  "
                                f"({total_ret:+.1%})",
                           fill=color, font=("맑은 고딕", 10, "bold"))

    # ─────────────────────────────────────────────
    # 유틸
    # ─────────────────────────────────────────────

    def _get_params(self) -> dict:
        defaults = {
            "capital":    100_000_000.0,
            "tx_cost":    0.15,
            "slippage":   0.05,
            "delay":      1,
            "train_days": 504,
            "test_days":  126,
            "step_days":  63,
        }
        result = {}
        for k, d in defaults.items():
            try:
                result[k] = type(d)(self.param_vars[k].get())
            except Exception:
                result[k] = d
        return result

    def _update_progress(self, pct: float, msg: str):
        self.progress["value"] = min(pct, 100)
        self.status_var.set(msg)

    def _on_complete(self):
        self.run_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self._set_status("완료")

    def _set_status(self, msg: str):
        self.frame.after(0, lambda: self.status_var.set(msg))

    def _log(self, msg: str):
        def _do():
            self.log_box.config(state="normal")
            self.log_box.insert("end", msg + "\n")
            self.log_box.see("end")
            self.log_box.config(state="disabled")
        self.frame.after(0, _do)
