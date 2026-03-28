# gui/bnf_panel.py
# BNF 매수 타이밍 탐지기 — 전체 UI 패널
# ─────────────────────────────────────────────────────────────────────────────
# 탭 구성:
#   탭1: 신호 스캔  — 종목 일괄 스캔, 시그널 테이블
#   탭2: 상세 분석  — 단일 종목 BNF Score 상세 + 근거 설명
#   탭3: 백테스트   — 신호 품질 리포트, 민감도 분석
#   탭4: 설정       — 가중치·임계값·파라미터 조정
# ─────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import logging
import math
import os
import sys
import threading
import traceback
from datetime import datetime
from typing import Dict, List, Optional

import tkinter as tk
from tkinter import ttk, messagebox

logger = logging.getLogger("quant.bnf.panel")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

# ── 안전한 선택적 임포트 ─────────────────────────────────────────────────────
try:
    import pandas as pd
    import numpy as np
    _PANDAS_OK = True
except ImportError:
    _PANDAS_OK = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    _MPL_OK = True
except ImportError:
    _MPL_OK = False

try:
    import yfinance as yf
    _YF_OK = True
except ImportError:
    _YF_OK = False

try:
    from config.bnf_config import BNFConfig, load_bnf_config, save_bnf_config
    from indicators.bnf_features import compute_bnf_features
    from strategies.bnf_signal_engine import (
        score_bnf_buy_signal, scan_bnf_signals, explain_bnf_signal,
        BNFSignalResult,
    )
    from backtest.bnf_backtester import (
        backtest_bnf_signals, batch_backtest, sensitivity_analysis,
        BNFBacktestResult,
    )
    _BNF_OK = True
except ImportError as e:
    logger.error(f"BNF 모듈 임포트 실패: {e}")
    _BNF_OK = False

# ── 한글 종목명 조회 (data.korean_stocks.get_name) ───────────────────────────
try:
    from data.korean_stocks import get_name as _ks_get_name
    _KS_OK = True
except ImportError:
    _KS_OK = False
    def _ks_get_name(ticker: str) -> str:   # fallback: ticker 그대로
        return ticker

# ── 테마 색상 (기존 앱 다크 테마와 동일) ────────────────────────────────────
_C = {
    "bg":       "#1e1e2e",
    "panel":    "#181825",
    "fg":       "#cdd6f4",
    "accent":   "#89b4fa",
    "dim":      "#9399b2",
    "green":    "#a6e3a1",
    "yellow":   "#f9e2af",
    "red":      "#f38ba8",
    "sky":      "#89dceb",
    "border":   "#585b70",
    "select":   "#313244",
    "header":   "#2a2a3e",
    "purple":   "#cba6f7",
}

# 신호 레이블 → 색
_LABEL_COLOR = {
    "강한 매수 후보": _C["green"],
    "매수 후보":      _C["yellow"],
    "관찰 필요":      _C["sky"],
    "관심 없음":      _C["dim"],
}


# ─────────────────────────────────────────────────────────────────────────────
# 헬퍼
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(v, fmt=".1f", fallback="-"):
    try:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return fallback
        return format(float(v), fmt)
    except Exception:
        return fallback


def _label_widget(parent, text, fg=None, bg=None, font=None, **kw):
    kw.setdefault("bg", bg or _C["bg"])
    kw.setdefault("fg", fg or _C["fg"])
    kw.setdefault("font", font or ("맑은 고딕", 9))
    kw.setdefault("anchor", "w")
    return tk.Label(parent, text=text, **kw)


# ─────────────────────────────────────────────────────────────────────────────
# BNF 패널 메인 클래스
# ─────────────────────────────────────────────────────────────────────────────

class BNFPanel:
    """
    BNF 매수 타이밍 탐지기 메인 패널.
    MainWindow.notebook 에 탭으로 추가됨.
    """

    def __init__(self, parent: ttk.Notebook, settings):
        self.settings = settings
        self.frame    = ttk.Frame(parent)
        self._cfg: BNFConfig = load_bnf_config() if _BNF_OK else None

        # 상태
        self._scan_thread: Optional[threading.Thread] = None
        self._bt_thread:   Optional[threading.Thread] = None
        self._stop_event   = threading.Event()
        self._last_signals: List[BNFSignalResult] = []
        self._symbol_dfs:   Dict[str, "pd.DataFrame"] = {}

        # 한글 종목명 캐시 & 표시라벨→ticker 역매핑
        self._name_cache:          Dict[str, str] = {}   # "005930.KS" → "삼성전자"
        self._label_to_ticker:     Dict[str, str] = {}   # "삼성전자 (005930)" → "005930.KS"

        # 차트 레퍼런스
        self._chart_canvas = None
        self._chart_fig    = None

        if not _PANDAS_OK or not _BNF_OK:
            self._build_error_ui()
        else:
            self._build_ui()

    # ──────────────────────────────────────────────────────────────────────────
    # 오류 화면
    # ──────────────────────────────────────────────────────────────────────────

    def _build_error_ui(self):
        msg = (
            "BNF 탐지기를 사용하려면 아래 패키지가 필요합니다:\n\n"
            "  pip install pandas numpy yfinance matplotlib\n\n"
            "설치 후 프로그램을 재시작하세요."
        )
        tk.Label(
            self.frame, text=msg,
            bg=_C["bg"], fg=_C["yellow"],
            font=("맑은 고딕", 11), justify="left",
        ).pack(expand=True)

    # ──────────────────────────────────────────────────────────────────────────
    # 메인 UI 빌드
    # ──────────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        self.frame.configure(style="TFrame")

        # ── 상단 타이틀 바 ────────────────────────────────────────────────────
        hdr = tk.Frame(self.frame, bg=_C["panel"], height=36)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        tk.Label(
            hdr,
            text="  🎯 BNF 매수 타이밍 탐지기",
            bg=_C["panel"], fg=_C["purple"],
            font=("맑은 고딕", 11, "bold"),
        ).pack(side="left", padx=8)

        tk.Label(
            hdr,
            text="규칙 기반 × 점수 기반 × 멀티팩터 진입 탐지 시스템",
            bg=_C["panel"], fg=_C["dim"],
            font=("맑은 고딕", 8),
        ).pack(side="left")

        self._status_var = tk.StringVar(value="대기 중")
        tk.Label(
            hdr,
            textvariable=self._status_var,
            bg=_C["panel"], fg=_C["sky"],
            font=("맑은 고딕", 8),
        ).pack(side="right", padx=12)

        # ── 내부 탭 ───────────────────────────────────────────────────────────
        nb = ttk.Notebook(self.frame)
        nb.pack(fill="both", expand=True, padx=4, pady=4)

        def _safe_add(text, builder):
            fr = ttk.Frame(nb)
            nb.add(fr, text=text)
            try:
                builder(fr)
            except Exception as e:
                logger.error(f"BNF 탭 '{text}' 빌드 실패: {e}", exc_info=True)
                tk.Label(
                    fr, text=f"⚠ 탭 빌드 실패\n{e}",
                    bg=_C["bg"], fg=_C["red"],
                    font=("맑은 고딕", 9),
                ).pack(pady=20)

        _safe_add("📡 신호 스캔",   self._build_tab_scan)
        _safe_add("🔍 상세 분석",   self._build_tab_detail)
        _safe_add("📊 백테스트",    self._build_tab_backtest)
        _safe_add("⚙ 설정",        self._build_tab_settings)

    # ═══════════════════════════════════════════════════════════════════════════
    # 탭 1: 신호 스캔
    # ═══════════════════════════════════════════════════════════════════════════

    def _build_tab_scan(self, parent):
        parent.configure(style="TFrame")

        # ── 컨트롤 바 ────────────────────────────────────────────────────────
        ctrl = tk.Frame(parent, bg=_C["panel"], height=40)
        ctrl.pack(fill="x", padx=4, pady=(4, 2))
        ctrl.pack_propagate(False)

        tk.Label(ctrl, text="최소점수:", bg=_C["panel"], fg=_C["dim"],
                 font=("맑은 고딕", 9)).pack(side="left", padx=(8, 2))
        self._min_score_var = tk.IntVar(value=30)
        ttk.Spinbox(
            ctrl, from_=0, to=100, increment=5,
            textvariable=self._min_score_var, width=5
        ).pack(side="left", padx=(0, 12))

        tk.Label(ctrl, text="데이터기간:", bg=_C["panel"], fg=_C["dim"],
                 font=("맑은 고딕", 9)).pack(side="left", padx=(0, 2))
        self._period_var = tk.StringVar(value="1y")
        ttk.Combobox(
            ctrl, textvariable=self._period_var,
            values=["6mo", "1y", "2y", "3y"], width=5, state="readonly"
        ).pack(side="left", padx=(0, 12))

        self._mkt_filter_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            ctrl, text="시장필터", variable=self._mkt_filter_var
        ).pack(side="left", padx=(0, 8))

        self._scan_btn = ttk.Button(
            ctrl, text="▶ 스캔 시작", style="Accent.TButton",
            command=self._on_scan_start,
        )
        self._scan_btn.pack(side="left", padx=4)

        self._stop_scan_btn = ttk.Button(
            ctrl, text="■ 중단", command=self._on_scan_stop, state="disabled"
        )
        self._stop_scan_btn.pack(side="left", padx=4)

        self._scan_prog = ttk.Progressbar(ctrl, length=200, mode="determinate")
        self._scan_prog.pack(side="left", padx=8)

        self._scan_info_var = tk.StringVar(value="")
        tk.Label(ctrl, textvariable=self._scan_info_var,
                 bg=_C["panel"], fg=_C["sky"],
                 font=("맑은 고딕", 8)).pack(side="left", padx=4)

        # ── 시그널 테이블 ────────────────────────────────────────────────────
        cols = [
            ("종목명",    150),
            ("현재가",    80),
            ("BNF점수",   70),
            ("판정",      90),
            ("손절가",    80),
            ("1차목표",   80),
            ("R:R",       55),
            ("5일수익",   70),
            ("핵심근거",  300),
        ]
        tree_fr = tk.Frame(parent, bg=_C["bg"])
        tree_fr.pack(fill="both", expand=True, padx=4, pady=2)

        tree_scroll_y = ttk.Scrollbar(tree_fr)
        tree_scroll_y.pack(side="right", fill="y")
        tree_scroll_x = ttk.Scrollbar(tree_fr, orient="horizontal")
        tree_scroll_x.pack(side="bottom", fill="x")

        self._scan_tree = ttk.Treeview(
            tree_fr,
            columns=[c[0] for c in cols],
            show="headings",
            yscrollcommand=tree_scroll_y.set,
            xscrollcommand=tree_scroll_x.set,
            height=20,
        )
        tree_scroll_y.config(command=self._scan_tree.yview)
        tree_scroll_x.config(command=self._scan_tree.xview)

        for col, width in cols:
            self._scan_tree.heading(col, text=col)
            self._scan_tree.column(col, width=width, minwidth=40, anchor="center")
        self._scan_tree.column("종목명",   anchor="w")
        self._scan_tree.column("핵심근거", anchor="w")

        self._scan_tree.pack(fill="both", expand=True)
        self._scan_tree.bind("<<TreeviewSelect>>", self._on_scan_select)

        # 태그 색상
        for label, color in _LABEL_COLOR.items():
            self._scan_tree.tag_configure(label, foreground=color)

    # ═══════════════════════════════════════════════════════════════════════════
    # 탭 2: 상세 분석
    # ═══════════════════════════════════════════════════════════════════════════

    def _build_tab_detail(self, parent):
        parent.configure(style="TFrame")

        # ── 상단 컨트롤 ──────────────────────────────────────────────────────
        ctrl = tk.Frame(parent, bg=_C["panel"])
        ctrl.pack(fill="x", padx=4, pady=(4, 2))

        tk.Label(ctrl, text="종목코드:", bg=_C["panel"], fg=_C["dim"],
                 font=("맑은 고딕", 9)).pack(side="left", padx=(8, 2))

        self._detail_sym_var = tk.StringVar()
        _det_labels = self._build_sym_labels()
        self._detail_sym_cb = ttk.Combobox(
            ctrl, textvariable=self._detail_sym_var,
            values=_det_labels, width=22, state="readonly"
        )
        if _det_labels:
            self._detail_sym_cb.current(0)
        self._detail_sym_cb.pack(side="left", padx=(0, 8))

        ttk.Button(
            ctrl, text="🔍 분석", command=self._on_detail_analyze,
        ).pack(side="left", padx=4)

        # ── PanedWindow: 좌=점수, 우=차트 ───────────────────────────────────
        paned = tk.PanedWindow(parent, orient="horizontal",
                               bg=_C["bg"], sashwidth=5)
        paned.pack(fill="both", expand=True, padx=4, pady=2)

        # 좌측: 점수 상세
        score_fr = tk.Frame(paned, bg=_C["bg"], width=360)
        paned.add(score_fr, minsize=300)
        self._build_score_panel(score_fr)

        # 우측: 차트
        chart_fr = tk.Frame(paned, bg=_C["bg"])
        paned.add(chart_fr, minsize=400)
        self._build_chart_panel(chart_fr)

    def _build_score_panel(self, parent):
        """좌측 점수 상세 패널"""
        parent.configure(bg=_C["bg"])

        # 판정 표시
        self._det_label_var = tk.StringVar(value="—")
        self._det_label_lbl = tk.Label(
            parent,
            textvariable=self._det_label_var,
            bg=_C["bg"], fg=_C["dim"],
            font=("맑은 고딕", 16, "bold"),
            anchor="center",
        )
        self._det_label_lbl.pack(fill="x", pady=(8, 2))

        self._det_score_var = tk.StringVar(value="BNF Score: — / 100")
        tk.Label(
            parent,
            textvariable=self._det_score_var,
            bg=_C["bg"], fg=_C["accent"],
            font=("맑은 고딕", 12, "bold"),
            anchor="center",
        ).pack(fill="x", pady=(0, 6))

        # 점수 바 캔버스
        self._score_canvas = tk.Canvas(
            parent, bg=_C["bg"], height=160,
            highlightthickness=0,
        )
        self._score_canvas.pack(fill="x", padx=12)
        self._draw_score_bars(None)

        ttk.Separator(parent).pack(fill="x", padx=8, pady=4)

        # 리스크 정보
        risk_fr = tk.LabelFrame(
            parent, text="리스크 / 손절",
            bg=_C["bg"], fg=_C["accent"],
            font=("맑은 고딕", 9, "bold"),
        )
        risk_fr.pack(fill="x", padx=8, pady=2)

        self._risk_vars = {}
        for key, label in [
            ("stop",    "손절가"),
            ("stop_p",  "손절폭"),
            ("t1",      "1차목표"),
            ("t2",      "2차목표"),
            ("rr",      "R:R"),
        ]:
            row = tk.Frame(risk_fr, bg=_C["bg"])
            row.pack(fill="x", padx=4, pady=1)
            tk.Label(row, text=f"{label}:", bg=_C["bg"], fg=_C["dim"],
                     font=("맑은 고딕", 9), width=8, anchor="e").pack(side="left")
            var = tk.StringVar(value="—")
            self._risk_vars[key] = var
            tk.Label(row, textvariable=var, bg=_C["bg"], fg=_C["fg"],
                     font=("맑은 고딕", 9, "bold"), anchor="w").pack(side="left", padx=4)

        ttk.Separator(parent).pack(fill="x", padx=8, pady=4)

        # 근거 설명
        tk.Label(
            parent, text="핵심 근거", bg=_C["bg"], fg=_C["accent"],
            font=("맑은 고딕", 9, "bold"), anchor="w",
        ).pack(fill="x", padx=10)

        reason_fr = tk.Frame(parent, bg=_C["panel"])
        reason_fr.pack(fill="both", expand=True, padx=8, pady=4)

        self._reason_text = tk.Text(
            reason_fr,
            bg=_C["panel"], fg=_C["fg"],
            font=("맑은 고딕", 8),
            wrap="word", state="disabled",
            height=10, borderwidth=0,
        )
        rsb = ttk.Scrollbar(reason_fr, command=self._reason_text.yview)
        self._reason_text.configure(yscrollcommand=rsb.set)
        rsb.pack(side="right", fill="y")
        self._reason_text.pack(fill="both", expand=True)

    def _draw_score_bars(self, sig: Optional["BNFSignalResult"]):
        """각 하위 점수 막대 그래프 그리기"""
        c = self._score_canvas
        c.delete("all")
        w = c.winfo_width() or 340
        h = 160

        items = [
            ("추세",   sig.trend_score     if sig else 0, _C["accent"]),
            ("괴리",   sig.deviation_score if sig else 0, _C["purple"]),
            ("거래량", sig.volume_score    if sig else 0, _C["sky"]),
            ("반등",   sig.reversal_score  if sig else 0, _C["green"]),
            ("시장",   sig.market_score    if sig else 0, _C["yellow"]),
            ("리스크", sig.risk_score      if sig else 0, _C["red"]),
        ]
        n = len(items)
        bar_h = 16
        gap   = (h - n * bar_h) / (n + 1)
        label_w = 52
        bar_max  = w - label_w - 50

        for i, (name, score, color) in enumerate(items):
            y = gap * (i + 1) + i * bar_h
            # 레이블
            c.create_text(label_w - 4, y + bar_h / 2,
                          text=name, fill=_C["dim"],
                          font=("맑은 고딕", 8), anchor="e")
            # 배경 바
            c.create_rectangle(label_w, y, label_w + bar_max, y + bar_h,
                                fill=_C["panel"], outline="")
            # 점수 바
            score = max(0, min(100, score or 0))
            bar_w = bar_max * score / 100
            c.create_rectangle(label_w, y, label_w + bar_w, y + bar_h,
                                fill=color, outline="")
            # 수치
            c.create_text(label_w + bar_max + 4, y + bar_h / 2,
                          text=f"{score:.0f}", fill=_C["fg"],
                          font=("맑은 고딕", 8), anchor="w")

    def _build_chart_panel(self, parent):
        """우측 차트 패널"""
        if not _MPL_OK:
            tk.Label(
                parent,
                text="matplotlib 미설치 — 차트 비활성\npip install matplotlib",
                bg=_C["bg"], fg=_C["dim"],
                font=("맑은 고딕", 9),
            ).pack(expand=True)
            return

        self._chart_fig, axes = plt.subplots(
            2, 1, figsize=(7, 5),
            gridspec_kw={"height_ratios": [3, 1]},
            facecolor=_C["bg"],
        )
        self._chart_ax_price  = axes[0]
        self._chart_ax_volume = axes[1]

        for ax in axes:
            ax.set_facecolor(_C["panel"])
            ax.tick_params(colors=_C["dim"], labelsize=7)
            for spine in ax.spines.values():
                spine.set_color(_C["border"])

        self._chart_fig.tight_layout(pad=1.2)

        self._chart_canvas = FigureCanvasTkAgg(self._chart_fig, master=parent)
        self._chart_canvas.get_tk_widget().pack(fill="both", expand=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # 탭 3: 백테스트
    # ═══════════════════════════════════════════════════════════════════════════

    def _build_tab_backtest(self, parent):
        parent.configure(style="TFrame")

        # ── 컨트롤 ───────────────────────────────────────────────────────────
        ctrl = tk.Frame(parent, bg=_C["panel"])
        ctrl.pack(fill="x", padx=4, pady=(4, 2))

        tk.Label(ctrl, text="종목:", bg=_C["panel"], fg=_C["dim"],
                 font=("맑은 고딕", 9)).pack(side="left", padx=(8, 2))
        self._bt_sym_var = tk.StringVar()
        _bt_labels = self._build_sym_labels()
        self._bt_sym_cb = ttk.Combobox(
            ctrl, textvariable=self._bt_sym_var,
            values=["(전체 종목)"] + _bt_labels, width=22, state="readonly"
        )
        self._bt_sym_cb.current(0)
        self._bt_sym_cb.pack(side="left", padx=(0, 8))

        tk.Label(ctrl, text="기간:", bg=_C["panel"], fg=_C["dim"],
                 font=("맑은 고딕", 9)).pack(side="left", padx=(0, 2))
        self._bt_period_var = tk.StringVar(value="2y")
        ttk.Combobox(
            ctrl, textvariable=self._bt_period_var,
            values=["1y", "2y", "3y", "5y"], width=5, state="readonly"
        ).pack(side="left", padx=(0, 8))

        tk.Label(ctrl, text="최소점수:", bg=_C["panel"], fg=_C["dim"],
                 font=("맑은 고딕", 9)).pack(side="left", padx=(0, 2))
        self._bt_min_score_var = tk.IntVar(value=50)
        ttk.Spinbox(
            ctrl, from_=0, to=100, increment=5,
            textvariable=self._bt_min_score_var, width=5
        ).pack(side="left", padx=(0, 8))

        self._bt_btn = ttk.Button(
            ctrl, text="▶ 백테스트 실행", style="Accent.TButton",
            command=self._on_bt_start,
        )
        self._bt_btn.pack(side="left", padx=4)

        self._bt_prog = ttk.Progressbar(ctrl, length=150, mode="indeterminate")
        self._bt_prog.pack(side="left", padx=8)

        # ── 결과 노트북 ──────────────────────────────────────────────────────
        bt_nb = ttk.Notebook(parent)
        bt_nb.pack(fill="both", expand=True, padx=4, pady=2)

        # 요약 탭
        self._bt_summary_fr = ttk.Frame(bt_nb)
        bt_nb.add(self._bt_summary_fr, text="요약")
        self._build_bt_summary_panel(self._bt_summary_fr)

        # 트레이드 목록 탭
        self._bt_trades_fr = ttk.Frame(bt_nb)
        bt_nb.add(self._bt_trades_fr, text="신호 목록")
        self._build_bt_trades_panel(self._bt_trades_fr)

        # 민감도 분석 탭
        self._bt_sens_fr = ttk.Frame(bt_nb)
        bt_nb.add(self._bt_sens_fr, text="민감도 분석")
        self._build_bt_sensitivity_panel(self._bt_sens_fr)

    def _build_bt_summary_panel(self, parent):
        self._bt_summary_text = tk.Text(
            parent,
            bg=_C["panel"], fg=_C["fg"],
            font=("맑은 고딕", 9),
            wrap="word", state="disabled",
            borderwidth=0,
        )
        sb = ttk.Scrollbar(parent, command=self._bt_summary_text.yview)
        self._bt_summary_text.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self._bt_summary_text.pack(fill="both", expand=True)

    def _build_bt_trades_panel(self, parent):
        cols = [
            ("신호날짜", 90),  ("종목", 80), ("진입가", 80),
            ("점수", 60), ("판정", 90),
            ("1일", 60), ("3일", 60), ("5일", 60), ("10일", 60),
            ("MAE", 60), ("MFE", 60), ("손절히트", 70), ("국면", 70),
        ]
        fr = tk.Frame(parent, bg=_C["bg"])
        fr.pack(fill="both", expand=True)

        sb_y = ttk.Scrollbar(fr)
        sb_y.pack(side="right", fill="y")
        sb_x = ttk.Scrollbar(fr, orient="horizontal")
        sb_x.pack(side="bottom", fill="x")

        self._bt_trade_tree = ttk.Treeview(
            fr, columns=[c[0] for c in cols], show="headings",
            yscrollcommand=sb_y.set, xscrollcommand=sb_x.set, height=18,
        )
        sb_y.config(command=self._bt_trade_tree.yview)
        sb_x.config(command=self._bt_trade_tree.xview)

        for col, w in cols:
            self._bt_trade_tree.heading(col, text=col)
            self._bt_trade_tree.column(col, width=w, minwidth=40, anchor="center")

        self._bt_trade_tree.pack(fill="both", expand=True)

    def _build_bt_sensitivity_panel(self, parent):
        self._bt_sens_text = tk.Text(
            parent,
            bg=_C["panel"], fg=_C["fg"],
            font=("맑은 고딕", 9),
            wrap="none", state="disabled",
            borderwidth=0,
        )
        sb = ttk.Scrollbar(parent, command=self._bt_sens_text.yview)
        self._bt_sens_text.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self._bt_sens_text.pack(fill="both", expand=True)

    # ═══════════════════════════════════════════════════════════════════════════
    # 탭 4: 설정
    # ═══════════════════════════════════════════════════════════════════════════

    def _build_tab_settings(self, parent):
        parent.configure(style="TFrame")

        canvas = tk.Canvas(parent, bg=_C["bg"], highlightthickness=0)
        vsb    = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas.pack(fill="both", expand=True)

        inner = tk.Frame(canvas, bg=_C["bg"])
        win_id = canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>",
                    lambda e: canvas.itemconfig(win_id, width=e.width))

        self._cfg_vars: Dict[str, tk.Variable] = {}

        def _section(title):
            fr = tk.LabelFrame(
                inner, text=title,
                bg=_C["bg"], fg=_C["accent"],
                font=("맑은 고딕", 9, "bold"),
            )
            fr.pack(fill="x", padx=10, pady=4)
            return fr

        def _slider_row(parent, key, label, from_, to, resolution=1, default=None):
            row = tk.Frame(parent, bg=_C["bg"])
            row.pack(fill="x", padx=8, pady=2)
            tk.Label(row, text=label, bg=_C["bg"], fg=_C["fg"],
                     font=("맑은 고딕", 9), width=22, anchor="w").pack(side="left")
            var = tk.DoubleVar(value=default or 0)
            self._cfg_vars[key] = var
            val_lbl = tk.Label(row, textvariable=var, bg=_C["bg"], fg=_C["sky"],
                               font=("맑은 고딕", 9), width=6)
            val_lbl.pack(side="right")
            sl = ttk.Scale(row, from_=from_, to=to, variable=var, orient="horizontal",
                           command=lambda v, vl=val_lbl, vr=var:
                               vl.config(text=f"{float(v):.{2 if resolution < 1 else 0}f}"))
            sl.pack(side="left", fill="x", expand=True, padx=4)

        # ── 가중치 설정 ───────────────────────────────────────────────────────
        if self._cfg:
            w = self._cfg.weights
            wfr = _section("가중치 (합계 = 1.0)")
            for key, label, default in [
                ("w_trend",     "추세/위치 가중치",    w.trend),
                ("w_deviation", "괴리율 가중치",       w.deviation),
                ("w_volume",    "거래량/수급 가중치",  w.volume),
                ("w_reversal",  "반등 확인 가중치",    w.reversal),
                ("w_market",    "시장 필터 가중치",    w.market),
                ("w_risk",      "리스크 가중치",       w.risk),
            ]:
                _slider_row(wfr, key, label, 0.0, 0.5, 0.05, default)

            # ── 판정 임계값 ───────────────────────────────────────────────────
            t = self._cfg.thresholds
            tfr = _section("판정 임계값")
            for key, label, default in [
                ("thr_observe",   "관찰 필요 하한",    t.observe),
                ("thr_candidate", "매수 후보 하한",    t.candidate),
                ("thr_strong",    "강한 매수 후보 하한", t.strong),
            ]:
                _slider_row(tfr, key, label, 0, 100, 5, default)

            # ── 괴리율 ────────────────────────────────────────────────────────
            d = self._cfg.deviation
            dfr = _section("괴리율 기준")
            for key, label, default, lo, hi in [
                ("dev_max_up",  "MA25 최대 상승 이격(%)", d.max_dev_ma25_pct,  0, 30),
                ("dev_min_dn",  "MA25 최대 하락 이격(%)", abs(d.min_dev_ma25_pct), 0, 40),
                ("z_overbought","Z-score 과매수 경계",    d.zscore_overbought,  1, 4),
            ]:
                _slider_row(dfr, key, label, lo, hi, 0.5, default)

            # ── 리스크 ────────────────────────────────────────────────────────
            r = self._cfg.risk
            rfr = _section("리스크 / 손절 기준")
            for key, label, default, lo, hi in [
                ("min_rr",       "최소 R:R",        r.min_rr,       0.5, 5.0),
                ("max_stop",     "최대 손절폭(%)",   r.max_stop_pct, 2, 20),
                ("atr_mult",     "ATR 손절 배수",    r.atr_multiplier, 0.5, 5.0),
                ("t1_atr",       "1차목표 ATR 배수", r.target1_atr_mult, 1, 10),
                ("t2_atr",       "2차목표 ATR 배수", r.target2_atr_mult, 2, 15),
            ]:
                _slider_row(rfr, key, label, lo, hi, 0.5, default)

        # ── 저장 버튼 ─────────────────────────────────────────────────────────
        btn_fr = tk.Frame(inner, bg=_C["bg"])
        btn_fr.pack(pady=8)
        ttk.Button(
            btn_fr, text="💾 설정 저장",
            style="Accent.TButton",
            command=self._on_save_settings,
        ).pack(side="left", padx=6)
        ttk.Button(
            btn_fr, text="↺ 기본값 복원",
            command=self._on_reset_settings,
        ).pack(side="left", padx=6)

    # ──────────────────────────────────────────────────────────────────────────
    # 이벤트 핸들러: 스캔
    # ──────────────────────────────────────────────────────────────────────────

    def _on_scan_start(self):
        if self._scan_thread and self._scan_thread.is_alive():
            return

        symbols = getattr(self.settings.data, "symbols", [])
        if not symbols:
            messagebox.showwarning("BNF 탐지기", "분석할 종목이 없습니다.\n데이터 탭에서 종목을 추가하세요.")
            return

        self._stop_event.clear()
        self._scan_btn.config(state="disabled")
        self._stop_scan_btn.config(state="normal")
        self._scan_prog["value"] = 0
        self._scan_prog["maximum"] = len(symbols)

        # 테이블 초기화
        for item in self._scan_tree.get_children():
            self._scan_tree.delete(item)

        self._status_var.set("스캔 중...")
        self._scan_thread = threading.Thread(
            target=self._scan_worker,
            args=(symbols,),
            daemon=True,
        )
        self._scan_thread.start()

    def _on_scan_stop(self):
        self._stop_event.set()
        self._status_var.set("중단 요청...")

    def _scan_worker(self, symbols: List[str]):
        cfg = self._cfg or BNFConfig()
        cfg.market_filter.enabled = self._mkt_filter_var.get()
        min_score = float(self._min_score_var.get())
        period    = self._period_var.get()

        # 시장 지수 데이터
        market_df = None
        if cfg.market_filter.enabled and _YF_OK:
            try:
                market_df = yf.download(
                    cfg.market_filter.market_ticker,
                    period=period, auto_adjust=True, progress=False
                )
                if isinstance(market_df.columns, pd.MultiIndex):
                    market_df.columns = market_df.columns.get_level_values(0)
            except Exception as e:
                logger.debug(f"시장지수 다운로드 실패: {e}")

        self._symbol_dfs = {}
        results = []

        for i, sym in enumerate(symbols):
            if self._stop_event.is_set():
                break

            self.frame.after(0, lambda v=i: self._scan_prog.config(value=v))
            self.frame.after(0, lambda s=sym: self._scan_info_var.set(f"분석 중: {s}"))

            df = self._fetch_ohlcv(sym, period)
            if df is None or len(df) < cfg.min_data_days:
                continue

            self._symbol_dfs[sym] = df

            try:
                sig = score_bnf_buy_signal(df, sym, market_df, cfg)
                if sig.valid and sig.bnf_score >= min_score:
                    results.append(sig)
                    self.frame.after(0, lambda r=sig: self._append_scan_row(r))
            except Exception as e:
                logger.warning(f"[{sym}] 스캔 채점 실패: {e}")

        self._last_signals = sorted(results, key=lambda x: x.bnf_score, reverse=True)

        self.frame.after(0, self._scan_finished)

    def _fetch_ohlcv(self, symbol: str, period: str = "1y") -> Optional["pd.DataFrame"]:
        """yfinance로 OHLCV 다운로드 (캐시 우선)"""
        # 기존 DataLoader 캐시 사용 시도
        try:
            from data.loader import DataLoader
            loader = DataLoader(self.settings)
            df = loader.load(symbol)
            if df is not None and len(df) >= 60:
                return df
        except Exception:
            pass

        # 직접 다운로드
        if not _YF_OK:
            return None
        try:
            df = yf.download(symbol, period=period, auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if df.empty or len(df) < 20:
                return None
            return df
        except Exception as e:
            logger.debug(f"[{symbol}] OHLCV 다운로드 실패: {e}")
            return None

    def _append_scan_row(self, sig: BNFSignalResult):
        """스캔 테이블에 행 추가 (메인 스레드에서 호출)"""
        short_reason = " | ".join(
            r.split("] ")[-1] for r in sig.reasons[:3] if r
        )
        stop_str  = f"{sig.stop_price:,.0f}" if not math.isnan(sig.stop_price)             else "-"
        t1_str    = f"{sig.target1:,.0f}"    if not math.isnan(sig.target1)                else "-"
        rr_str    = f"{sig.reward_risk_ratio:.1f}" if not math.isnan(sig.reward_risk_ratio) else "-"
        close_str = f"{sig.close:,.0f}"      if not math.isnan(sig.close)                  else "-"

        # 종목명 표시: "삼성전자 (005930)" 형태
        name_label = self._sym_label(sig.symbol)

        # iid = ticker 코드 (선택 시 역조회에 사용)
        iid = sig.symbol
        # 중복 방지 (같은 종목이 두 번 들어올 경우)
        if self._scan_tree.exists(iid):
            self._scan_tree.delete(iid)

        self._scan_tree.insert(
            "", "end",
            iid=iid,
            values=(
                name_label,
                close_str,
                f"{sig.bnf_score:.0f}",
                sig.label,
                stop_str,
                t1_str,
                rr_str,
                "—",   # 5일 수익 (백테스트 미실행)
                short_reason,
            ),
            tags=(sig.label,),
        )

    def _scan_finished(self):
        n = len(self._last_signals)
        self._scan_prog["value"] = self._scan_prog["maximum"]
        self._scan_btn.config(state="normal")
        self._stop_scan_btn.config(state="disabled")
        self._scan_info_var.set("")
        self._status_var.set(
            f"스캔 완료 — {n}개 신호 발견 ({datetime.now().strftime('%H:%M:%S')})"
        )

    def _on_scan_select(self, event):
        """테이블에서 종목 선택 시 상세 분석 탭 combobox 동기화"""
        sel = self._scan_tree.selection()
        if not sel:
            return
        ticker = sel[0]   # iid = ticker (예: "005930.KS")
        # combobox에서 해당 종목의 라벨을 찾아 선택
        label = self._sym_label(ticker)
        vals  = list(self._detail_sym_cb["values"])
        if label in vals:
            self._detail_sym_cb.current(vals.index(label))
        else:
            self._detail_sym_var.set(label)

    # ──────────────────────────────────────────────────────────────────────────
    # 이벤트 핸들러: 상세 분석
    # ──────────────────────────────────────────────────────────────────────────

    def _on_detail_analyze(self):
        label = self._detail_sym_var.get().strip()
        if not label:
            messagebox.showwarning("BNF", "종목을 선택하세요.")
            return

        # "삼성전자 (005930)" → "005930.KS"
        sym = self._ticker_from_label(label)
        name = self._get_name(sym)
        display = f"{name} ({sym.split('.')[0]})" if name != sym else sym

        self._status_var.set(f"{display} 분석 중...")

        def _worker():
            df = self._symbol_dfs.get(sym) or self._fetch_ohlcv(sym)
            if df is None or len(df) < 20:
                self.frame.after(0, lambda: messagebox.showerror(
                    "BNF", f"[{display}] 데이터 없음 또는 부족합니다."
                ))
                return
            try:
                sig = score_bnf_buy_signal(df, sym, cfg=self._cfg)
                self.frame.after(0, lambda: self._update_detail_ui(sig, df))
            except Exception as e:
                err = traceback.format_exc()
                self.frame.after(0, lambda: messagebox.showerror("BNF 오류", err[:400]))

        threading.Thread(target=_worker, daemon=True).start()

    def _update_detail_ui(self, sig: BNFSignalResult, df: "pd.DataFrame"):
        """상세 분석 결과를 UI에 반영"""
        # 판정 레이블
        self._det_label_var.set(sig.label)
        self._det_label_lbl.config(fg=sig.label_color)
        self._det_score_var.set(f"BNF Score: {sig.bnf_score:.0f} / 100")

        # 점수 바 재그리기
        self._score_canvas.after(50, lambda: self._draw_score_bars(sig))

        # 리스크 정보
        def _rfmt(v, suffix=""):
            return (f"{v:,.0f}{suffix}" if not math.isnan(v) else "—") if isinstance(v, float) else "—"

        self._risk_vars["stop"].set(_rfmt(sig.stop_price, "원"))
        self._risk_vars["stop_p"].set(f"{sig.stop_pct:.1f}%" if not math.isnan(sig.stop_pct) else "—")
        self._risk_vars["t1"].set(_rfmt(sig.target1, "원"))
        self._risk_vars["t2"].set(_rfmt(sig.target2, "원"))
        self._risk_vars["rr"].set(
            f"{sig.reward_risk_ratio:.2f}" if not math.isnan(sig.reward_risk_ratio) else "—"
        )

        # 근거 설명
        explanation = explain_bnf_signal(sig)
        self._reason_text.config(state="normal")
        self._reason_text.delete("1.0", "end")
        self._reason_text.insert("1.0", explanation)
        self._reason_text.config(state="disabled")

        # 차트 업데이트
        if _MPL_OK and self._chart_canvas:
            self._update_chart(df, sig)

        self._status_var.set(
            f"{sig.symbol} 분석 완료 — {sig.label} ({sig.bnf_score:.0f}점)"
        )

    def _update_chart(self, df: "pd.DataFrame", sig: BNFSignalResult):
        """캔들 차트 + BNF 오버레이"""
        try:
            feat = compute_bnf_features(df)
            ax_p = self._chart_ax_price
            ax_v = self._chart_ax_volume

            ax_p.clear()
            ax_v.clear()
            for ax in (ax_p, ax_v):
                ax.set_facecolor(_C["panel"])
                ax.tick_params(colors=_C["dim"], labelsize=7)

            # 최근 80봉만 표시
            show = feat.tail(80)
            idx  = range(len(show))

            # 캔들
            for i, (_, row) in enumerate(show.iterrows()):
                o, h, lo_v, c = row["Open"], row["High"], row["Low"], row["Close"]
                color = _C["green"] if c >= o else _C["red"]
                ax_p.plot([i, i], [lo_v, h], color=color, linewidth=0.8)
                ax_p.add_patch(plt.Rectangle(
                    (i - 0.3, min(o, c)), 0.6, abs(c - o),
                    color=color, alpha=0.85,
                ))

            # MA 라인
            for n, color, alpha in [(20, _C["accent"], 0.9),
                                     (25, _C["purple"], 0.7),
                                     (60, _C["yellow"], 0.6)]:
                col = f"ma{n}"
                if col in show.columns:
                    vals = show[col].values
                    ax_p.plot(idx, vals, color=color, linewidth=1.0,
                              alpha=alpha, label=f"MA{n}")

            # 손절가 / 목표가 라인
            if not math.isnan(sig.stop_price):
                ax_p.axhline(sig.stop_price, color=_C["red"],
                             linestyle="--", linewidth=1.0, alpha=0.8,
                             label=f"손절 {sig.stop_price:,.0f}")
            if not math.isnan(sig.target1):
                ax_p.axhline(sig.target1, color=_C["green"],
                             linestyle="--", linewidth=1.0, alpha=0.8,
                             label=f"목표 {sig.target1:,.0f}")

            # 신호 아이콘 (마지막 봉)
            last_i   = len(show) - 1
            last_cls = show["Close"].iloc[-1]
            marker = "★" if sig.label == "강한 매수 후보" else "▲"
            color  = sig.label_color
            ax_p.annotate(
                marker,
                xy=(last_i, last_cls),
                xytext=(last_i, last_cls * 0.985),
                fontsize=12, color=color, ha="center",
            )

            ax_p.legend(fontsize=6, loc="upper left",
                        facecolor=_C["panel"], labelcolor=_C["fg"], framealpha=0.7)
            _chart_name = self._sym_label(sig.symbol)
            ax_p.set_title(
                f"{_chart_name}  BNF:{sig.bnf_score:.0f}  [{sig.label}]",
                color=_C["fg"], fontsize=8, pad=4,
            )

            # 거래량
            vol_ma20 = show["vol_ma20"].values if "vol_ma20" in show.columns else None
            for i, (_, row) in enumerate(show.iterrows()):
                c = row["Close"]
                o = row["Open"]
                color = _C["green"] if c >= o else _C["red"]
                ax_v.bar(i, row["Volume"], color=color, alpha=0.6, width=0.8)
            if vol_ma20 is not None:
                ax_v.plot(idx, vol_ma20, color=_C["accent"],
                          linewidth=0.8, alpha=0.7)
            ax_v.set_ylabel("거래량", color=_C["dim"], fontsize=6)

            self._chart_fig.tight_layout(pad=1.0)
            self._chart_canvas.draw()

        except Exception as e:
            logger.warning(f"차트 업데이트 실패: {e}", exc_info=True)

    # ──────────────────────────────────────────────────────────────────────────
    # 이벤트 핸들러: 백테스트
    # ──────────────────────────────────────────────────────────────────────────

    def _on_bt_start(self):
        if self._bt_thread and self._bt_thread.is_alive():
            return

        self._bt_btn.config(state="disabled")
        self._bt_prog.start()

        sym    = self._bt_sym_var.get()
        period = self._bt_period_var.get()
        minscore = float(self._bt_min_score_var.get())

        self._bt_thread = threading.Thread(
            target=self._bt_worker,
            args=(sym, period, minscore),
            daemon=True,
        )
        self._bt_thread.start()

    def _bt_worker(self, sym: str, period: str, min_score: float):
        cfg  = self._cfg or BNFConfig()
        syms = getattr(self.settings.data, "symbols", [])

        if sym == "(전체 종목)":
            target_syms = syms
        else:
            # sym이 "삼성전자 (005930)" 형태일 경우 ticker로 역변환
            ticker = self._ticker_from_label(sym)
            target_syms = [ticker] if ticker in syms else syms[:1]

        results = {}
        for s in target_syms:
            df = self._symbol_dfs.get(s) or self._fetch_ohlcv(s, period)
            if df is not None and len(df) >= cfg.min_data_days:
                try:
                    res = backtest_bnf_signals(df, s, cfg=cfg, min_score=min_score)
                    results[s] = res
                except Exception as e:
                    logger.warning(f"[{s}] 백테스트 실패: {e}")

        # 민감도 분석 (첫 번째 종목)
        sens_df = None
        if target_syms:
            first = target_syms[0]
            df0   = self._symbol_dfs.get(first) or self._fetch_ohlcv(first, period)
            if df0 is not None and len(df0) >= cfg.min_data_days:
                try:
                    sens_df = sensitivity_analysis(df0, first, cfg)
                except Exception as e:
                    logger.warning(f"민감도 분석 실패: {e}")

        self.frame.after(0, lambda: self._bt_finished(results, sens_df))

    def _bt_finished(self, results: Dict[str, "BNFBacktestResult"], sens_df):
        self._bt_btn.config(state="normal")
        self._bt_prog.stop()

        # ── 요약 텍스트 ──────────────────────────────────────────────────────
        lines = ["=" * 58, " BNF 백테스트 결과 요약", "=" * 58]

        for sym, res in results.items():
            name_disp = self._sym_label(sym)
            lines.append(f"\n▶ {name_disp}")
            d = res.to_summary_dict()
            for k, v in d.items():
                if k != "symbol":
                    lines.append(f"   {k:<16} {v}")
            if res.regime_stats:
                lines.append("   [국면별 성과]")
                for regime, stat in res.regime_stats.items():
                    lines.append(
                        f"     {regime:<10}: "
                        f"n={stat['count']}  "
                        f"승률={stat['win_rate']*100:.1f}%  "
                        f"평균={stat['avg_ret']:.2f}%"
                    )
            if res.fp_patterns:
                lines.append("   [False Positive 패턴]")
                for fp in res.fp_patterns:
                    lines.append(f"     {fp}")

        self._bt_summary_text.config(state="normal")
        self._bt_summary_text.delete("1.0", "end")
        self._bt_summary_text.insert("1.0", "\n".join(lines))
        self._bt_summary_text.config(state="disabled")

        # ── 트레이드 목록 ─────────────────────────────────────────────────────
        for item in self._bt_trade_tree.get_children():
            self._bt_trade_tree.delete(item)

        for res in results.values():
            for t in res.trades:
                self._bt_trade_tree.insert(
                    "", "end",
                    values=(
                        t.signal_date, t.symbol,
                        f"{t.entry_price:,.0f}",
                        f"{t.bnf_score:.0f}",
                        t.label,
                        _fmt(t.ret_1d, ".2f"),
                        _fmt(t.ret_3d, ".2f"),
                        _fmt(t.ret_5d, ".2f"),
                        _fmt(t.ret_10d, ".2f"),
                        _fmt(t.mae_5d, ".2f"),
                        _fmt(t.mfe_5d, ".2f"),
                        "✓" if t.stop_hit_5d else "—",
                        t.market_regime,
                    ),
                    tags=(t.label,),
                )
            for label, color in _LABEL_COLOR.items():
                self._bt_trade_tree.tag_configure(label, foreground=color)

        # ── 민감도 분석 ───────────────────────────────────────────────────────
        self._bt_sens_text.config(state="normal")
        self._bt_sens_text.delete("1.0", "end")
        if sens_df is not None and not sens_df.empty:
            self._bt_sens_text.insert("1.0",
                "▶ min_score 민감도 분석\n\n" + sens_df.to_string()
            )
        else:
            self._bt_sens_text.insert("1.0", "민감도 분석 데이터 없음")
        self._bt_sens_text.config(state="disabled")

        n_total = sum(r.n_signals for r in results.values())
        self._status_var.set(
            f"백테스트 완료 — {len(results)}종목 / 총 {n_total}신호 "
            f"({datetime.now().strftime('%H:%M:%S')})"
        )

    # ──────────────────────────────────────────────────────────────────────────
    # 이벤트 핸들러: 설정 저장/초기화
    # ──────────────────────────────────────────────────────────────────────────

    def _on_save_settings(self):
        if not self._cfg:
            return
        v = self._cfg_vars
        try:
            # 가중치
            self._cfg.weights.trend     = float(v.get("w_trend",     tk.DoubleVar(value=0.20)).get())
            self._cfg.weights.deviation = float(v.get("w_deviation", tk.DoubleVar(value=0.15)).get())
            self._cfg.weights.volume    = float(v.get("w_volume",    tk.DoubleVar(value=0.20)).get())
            self._cfg.weights.reversal  = float(v.get("w_reversal",  tk.DoubleVar(value=0.25)).get())
            self._cfg.weights.market    = float(v.get("w_market",    tk.DoubleVar(value=0.10)).get())
            self._cfg.weights.risk      = float(v.get("w_risk",      tk.DoubleVar(value=0.10)).get())

            # 임계값
            self._cfg.thresholds.observe   = int(v.get("thr_observe",   tk.IntVar(value=30)).get())
            self._cfg.thresholds.candidate = int(v.get("thr_candidate", tk.IntVar(value=50)).get())
            self._cfg.thresholds.strong    = int(v.get("thr_strong",    tk.IntVar(value=70)).get())

            # 리스크
            self._cfg.risk.min_rr           = float(v.get("min_rr",    tk.DoubleVar(value=1.5)).get())
            self._cfg.risk.max_stop_pct     = float(v.get("max_stop",  tk.DoubleVar(value=8.0)).get())
            self._cfg.risk.atr_multiplier   = float(v.get("atr_mult",  tk.DoubleVar(value=2.0)).get())
            self._cfg.risk.target1_atr_mult = float(v.get("t1_atr",    tk.DoubleVar(value=3.0)).get())
            self._cfg.risk.target2_atr_mult = float(v.get("t2_atr",    tk.DoubleVar(value=6.0)).get())

            save_bnf_config(self._cfg)
            messagebox.showinfo("BNF 설정", "설정이 저장되었습니다.")
        except Exception as e:
            messagebox.showerror("설정 저장 실패", str(e))

    def _on_reset_settings(self):
        if messagebox.askyesno("BNF 설정", "기본값으로 초기화하시겠습니까?"):
            self._cfg = BNFConfig()
            save_bnf_config(self._cfg)
            messagebox.showinfo("BNF 설정", "기본값으로 복원되었습니다.\n설정 탭을 다시 열면 반영됩니다.")

    # ──────────────────────────────────────────────────────────────────────────
    # 외부 호출 인터페이스
    # ──────────────────────────────────────────────────────────────────────────

    # ──────────────────────────────────────────────────────────────────────────
    # 한글 종목명 헬퍼
    # ──────────────────────────────────────────────────────────────────────────

    def _get_name(self, sym: str) -> str:
        """종목코드 → 한글명 (캐시 적용, 없으면 ticker 반환)"""
        if sym not in self._name_cache:
            self._name_cache[sym] = _ks_get_name(sym)
        return self._name_cache[sym]

    def _sym_label(self, sym: str) -> str:
        """'삼성전자 (005930)' 형태 표시 문자열 생성"""
        name = self._get_name(sym)
        code = sym.split(".")[0]          # "005930.KS" → "005930"
        return f"{name} ({code})" if name != sym else sym

    def _ticker_from_label(self, label: str) -> str:
        """'삼성전자 (005930)' → '005930.KS' 역변환"""
        # 캐시 역매핑에서 먼저 조회
        if label in self._label_to_ticker:
            return self._label_to_ticker[label]
        # 코드 파싱 폴백: "(005930)" 추출 후 settings에서 매칭
        import re
        m = re.search(r'\(([^)]+)\)$', label)
        if m:
            code = m.group(1)
            syms = getattr(self.settings.data, "symbols", [])
            for s in syms:
                if s.split(".")[0] == code:
                    return s
        # 그대로 반환 (이미 ticker인 경우)
        return label

    def _build_sym_labels(self) -> List[str]:
        """현재 설정 종목 → 표시 라벨 목록 생성 & 역매핑 갱신"""
        syms = getattr(self.settings.data, "symbols", [])
        labels = []
        self._label_to_ticker.clear()
        for sym in syms:
            lbl = self._sym_label(sym)
            labels.append(lbl)
            self._label_to_ticker[lbl] = sym
        return labels

    def refresh_symbols(self):
        """설정에서 종목 목록이 변경될 때 호출"""
        labels = self._build_sym_labels()
        if hasattr(self, "_detail_sym_cb"):
            self._detail_sym_cb["values"] = labels
            if labels:
                self._detail_sym_cb.current(0)
        if hasattr(self, "_bt_sym_cb"):
            self._bt_sym_cb["values"] = ["(전체 종목)"] + labels
            self._bt_sym_cb.current(0)
