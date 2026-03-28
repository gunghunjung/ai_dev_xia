# gui/market_env_panel.py — 외부환경 실시간 모니터 패널
"""
시장 외부환경을 직관적으로 보여주는 전용 UI 패널

구성:
  (1) 시장 분위기 게이지    — 악재 ←──── 중립 ────→ 호재  (색상 + 애니메이션)
  (2) 섹터별 영향 히트맵    — 금융 / 에너지 / 반도체 / 헬스케어
  (3) 이벤트 리스트         — 색상 + 강도 표시 + 정렬
  (4) 이벤트 상세 + 예측 영향 설명
  (5) 자동 갱신 (30분 주기)
"""
from __future__ import annotations

import datetime
import os
import sys
import threading
import tkinter as tk
from tkinter import ttk
import logging
from typing import Callable, List, Optional

logger = logging.getLogger("quant.gui.market")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 외부환경 분석 모듈 (optional)
try:
    from features.external_env import (
        ExternalEnvFeatureEngineer, NewsEventCategorizer,
        EventCategory, CATEGORY_WEIGHTS, StructuredEvent,
        get_analyzer as get_nlp_analyzer,
    )
    _EXT_ENV_AVAILABLE = True
except ImportError:
    _EXT_ENV_AVAILABLE = False
sys.path.insert(0, BASE_DIR)

# ──────────────────────────────────────────────────────────────────────────────
# 색상 테마
# ──────────────────────────────────────────────────────────────────────────────
_C = {
    "bg":          "#1e1e2e",
    "bg2":         "#181825",
    "bg3":         "#11111b",
    "border":      "#313244",
    "text":        "#cdd6f4",
    "subtext":     "#9399b2",
    "bull":        "#89dceb",   # 호재: 청록
    "bull_strong": "#74c7ec",   # 강한 호재: 파랑
    "bear":        "#f38ba8",   # 악재: 핑크-레드
    "bear_strong": "#e64553",   # 강한 악재: 진한 빨강
    "neutral":     "#6c7086",   # 중립: 회색
    "uncert":      "#f9e2af",   # 불확실: 황색
    "gauge_bg":    "#313244",
    "green":       "#a6e3a1",
    "yellow":      "#f9e2af",
    "red":         "#f38ba8",
}

# 이벤트 타입별 한국어 레이블 + 아이콘
_TYPE_META = {
    "FOMC":      {"label": "🏦 연준/FOMC",  "color": "#cba6f7"},
    "RATE":      {"label": "📊 금리",        "color": "#89b4fa"},
    "CPI":       {"label": "💰 물가(CPI)",   "color": "#f9e2af"},
    "WAR":       {"label": "⚔️ 전쟁/분쟁",  "color": "#f38ba8"},
    "POLICY":    {"label": "🏛️ 정책/규제",  "color": "#a6e3a1"},
    "EARNINGS":  {"label": "📈 기업실적",    "color": "#94e2d5"},
    "SUPPLY":    {"label": "🏭 공급망",      "color": "#fab387"},
    "FX":        {"label": "💱 환율",        "color": "#f5c2e7"},
    "COMMODITY": {"label": "🛢️ 원자재",     "color": "#eba0ac"},
    "UNKNOWN":   {"label": "📰 기타",        "color": "#6c7086"},
}

_SECTOR_META = {
    "FINANCE": {"icon": "🏦", "label": "금융"},
    "ENERGY":  {"icon": "⚡", "label": "에너지"},
    "TECH":    {"icon": "💻", "label": "반도체/IT"},
    "HEALTH":  {"icon": "🏥", "label": "헬스케어"},
}


def _source_badge(source: str) -> str:
    """소스 문자열 → 짧은 배지 텍스트 (Treeview 표시용)"""
    sl = (source or "").lower()
    if "google" in sl:
        return "🔍G"
    if "finnhub" in sl:
        return "📊F"
    if any(x in sl for x in ("gdelt", "collector")):
        return "🗞C"
    if any(x in sl for x in ("yahoo", "reuters", "marketwatch",
                               "연합", "한국경제", "매일경제", "investing")):
        return "📡R"
    return "📰"


# ──────────────────────────────────────────────────────────────────────────────
# 시장 분위기 게이지 위젯
# ──────────────────────────────────────────────────────────────────────────────

class MoodGauge(tk.Canvas):
    """
    악재 ←────────────── 중립 ──────────────→ 호재
    빨간색                                     파란색
    게이지 포인터 + 중앙값 수치 표시
    """

    def __init__(self, parent, width=600, height=60, **kw):
        super().__init__(parent, width=width, height=height,
                         bg=_C["bg2"], highlightthickness=0, **kw)
        self._score = 0.0     # -1 ~ +1
        self._anim_target = 0.0
        self._anim_current = 0.0
        self.bind("<Configure>", lambda e: self._draw())
        self._draw()
        self._animate()

    def set_score(self, score: float):
        """score: -1(극도 악재) ~ 0(중립) ~ +1(극도 호재)"""
        self._anim_target = max(-1.0, min(1.0, float(score)))

    def _animate(self):
        """부드러운 포인터 이동 애니메이션"""
        diff = self._anim_target - self._anim_current
        if abs(diff) > 0.005:
            self._anim_current += diff * 0.15
            self._draw()
        self.after(33, self._animate)

    def _draw(self):
        self.delete("all")
        W = self.winfo_width()  or 600
        H = self.winfo_height() or 60
        if W < 10:
            return

        pad = 20
        bar_y1, bar_y2 = H // 2 - 8, H // 2 + 8
        bar_w = W - pad * 2

        # ── 그라디언트 바 (세그먼트로 근사) ──────────────────────────────
        N = 120
        for i in range(N):
            t = i / N               # 0=왼쪽(악재) ~ 1=오른쪽(호재)
            # 빨강→회색→파랑 그라디언트
            if t < 0.5:
                r = int(243 - (243 - 108) * (t * 2))
                g = int(139 - (139 - 115) * (t * 2))
                b = int(168 - (168 - 134) * (t * 2))
            else:
                r = int(108 - (108 - 137) * ((t - 0.5) * 2))
                g = int(115 - (115 - 180) * ((t - 0.5) * 2))
                b = int(134 - (134 - 250) * ((t - 0.5) * 2))
            color = f"#{r:02x}{g:02x}{b:02x}"
            x1 = pad + int(bar_w * i / N)
            x2 = pad + int(bar_w * (i + 1) / N)
            self.create_rectangle(x1, bar_y1, x2, bar_y2, fill=color, outline="")

        # ── 라벨 ──────────────────────────────────────────────────────────
        self.create_text(pad, H // 2, text="악재", anchor="e",
                         fill=_C["bear"], font=("맑은 고딕", 8))
        self.create_text(W - pad, H // 2, text="호재", anchor="w",
                         fill=_C["bull"], font=("맑은 고딕", 8))
        self.create_text(W // 2, H // 2, text="중립",
                         fill=_C["neutral"], font=("맑은 고딕", 8))

        # ── 포인터 삼각형 ─────────────────────────────────────────────────
        t = (self._anim_current + 1) / 2  # [0, 1]
        px = pad + int(bar_w * t)

        # 포인터 색: 악재=빨강, 호재=파랑, 중립=흰색
        sc = self._anim_current
        if sc > 0.1:
            ptr_color = _C["bull_strong"]
        elif sc < -0.1:
            ptr_color = _C["bear_strong"]
        else:
            ptr_color = _C["text"]

        self.create_polygon(
            px - 7, bar_y1 - 3,
            px + 7, bar_y1 - 3,
            px, bar_y2 + 5,
            fill=ptr_color, outline=""
        )

        # ── 수치 표시 ──────────────────────────────────────────────────────
        pct = int(sc * 100)
        label = f"{'호재' if pct > 0 else '악재' if pct < 0 else '중립'} {abs(pct)}%"
        self.create_text(px, 10, text=label,
                         fill=ptr_color, font=("맑은 고딕", 9, "bold"), anchor="n")


# ──────────────────────────────────────────────────────────────────────────────
# 12카테고리 히트맵 위젯 (4열 × 3행 그리드)
# ──────────────────────────────────────────────────────────────────────────────

_CAT_META: dict = {
    "Macro":          {"icon": "📊", "label": "거시경제"},
    "MonetaryPolicy": {"icon": "🏦", "label": "통화정책"},
    "Geopolitics":    {"icon": "⚔",  "label": "지정학"},
    "Industry":       {"icon": "🏭", "label": "산업"},
    "Corporate":      {"icon": "📈", "label": "기업"},
    "Government":     {"icon": "🏛", "label": "정부/규제"},
    "Flow":           {"icon": "💰", "label": "수급"},
    "MarketEvent":    {"icon": "📉", "label": "시장이벤트"},
    "Technology":     {"icon": "💻", "label": "기술"},
    "Commodity":      {"icon": "🛢", "label": "원자재"},
    "FinancialMkt":   {"icon": "💳", "label": "금융시장"},
    "Sentiment":      {"icon": "😱", "label": "심리"},
}

class CategoryHeatmap(tk.Frame):
    """12개 EventCategory 영향 강도를 4열×3행 색상 그리드로 표시"""

    def __init__(self, parent, **kw):
        super().__init__(parent, bg=_C["bg2"], **kw)
        self._cells: dict = {}
        self._build()

    def _build(self):
        tk.Label(self, text="12카테고리 외부환경 현황",
                 bg=_C["bg2"], fg=_C["subtext"],
                 font=("맑은 고딕", 8)).pack(anchor="w", padx=6, pady=(3, 2))

        grid_fr = tk.Frame(self, bg=_C["bg2"])
        grid_fr.pack(fill="x", padx=4, pady=(0, 3))

        cats = list(_CAT_META.keys())     # 12개
        cols = 4
        for idx, cat_key in enumerate(cats):
            meta = _CAT_META[cat_key]
            r, c = divmod(idx, cols)

            cell = tk.Frame(grid_fr, bg=_C["bg3"],
                            highlightthickness=1,
                            highlightbackground=_C["border"])
            cell.grid(row=r, column=c, padx=2, pady=2, sticky="nsew")
            grid_fr.columnconfigure(c, weight=1)

            icon_lbl = tk.Label(cell, text=meta["icon"],
                                bg=_C["bg3"],
                                font=("Segoe UI Emoji", 13))
            icon_lbl.pack(pady=(4, 0))

            name_lbl = tk.Label(cell, text=meta["label"],
                                bg=_C["bg3"], fg=_C["subtext"],
                                font=("맑은 고딕", 7))
            name_lbl.pack()

            score_lbl = tk.Label(cell, text="─",
                                  bg=_C["bg3"], fg=_C["neutral"],
                                  font=("맑은 고딕", 8, "bold"))
            score_lbl.pack(pady=(0, 4))

            self._cells[cat_key] = {
                "frame": cell, "icon": icon_lbl,
                "name": name_lbl, "score": score_lbl,
            }

    def update_scores(self, scores: dict):
        """scores: {category_value_str: float}  범위 약 -2 ~ +2"""
        for cat_key, cell in self._cells.items():
            sc = scores.get(cat_key, 0.0)
            if sc > 0.05:
                bg   = self._tint(sc,  "bull")
                fg   = _C["bull_strong"] if sc > 0.8 else _C["bull"]
                text = f"+{sc:.2f}"
            elif sc < -0.05:
                bg   = self._tint(-sc, "bear")
                fg   = _C["bear_strong"] if sc < -0.8 else _C["bear"]
                text = f"{sc:.2f}"
            else:
                bg  = _C["bg3"]
                fg  = _C["neutral"]
                text = "─"

            for w in (cell["frame"], cell["icon"], cell["name"], cell["score"]):
                w.configure(bg=bg)
            cell["score"].configure(fg=fg, text=text)
            cell["frame"].configure(highlightbackground=fg)

    @staticmethod
    def _tint(v: float, side: str) -> str:
        s = max(0.0, min(1.0, v / 2.0))
        if side == "bull":
            return f"#{int(18+s*18):02x}{int(18+s*50):02x}{int(30+s*70):02x}"
        return f"#{int(18+s*70):02x}{int(18+s*8):02x}{int(30+s*8):02x}"


# ──────────────────────────────────────────────────────────────────────────────
# 12카테고리 바 차트 위젯
# ──────────────────────────────────────────────────────────────────────────────

_CAT_LABELS = {
    "Macro":          "거시경제",
    "MonetaryPolicy": "통화정책",
    "Geopolitics":    "지정학",
    "Industry":       "산업",
    "Corporate":      "기업",
    "Government":     "정부/규제",
    "Flow":           "수급",
    "MarketEvent":    "시장이벤트",
    "Technology":     "기술",
    "Commodity":      "원자재",
    "FinancialMkt":   "금융시장",
    "Sentiment":      "심리",
}

class CategoryBarChart(tk.Canvas):
    """
    12개 카테고리별 누적 점수를 수평 바 차트로 표시.
    양수=호재(파랑), 음수=악재(빨강)
    """

    def __init__(self, parent, **kw):
        kw.setdefault("bg", _C["bg2"])
        kw.setdefault("highlightthickness", 0)
        super().__init__(parent, **kw)
        self._scores: dict[str, float] = {}
        self.bind("<Configure>", lambda e: self._draw())

    def update_scores(self, scores: dict):
        """scores: {category_value_str: float} 범위 -2 ~ +2"""
        self._scores = dict(scores)
        self._draw()

    def _draw(self):
        self.delete("all")
        W = self.winfo_width()  or 400
        H = self.winfo_height() or 200
        if W < 40 or H < 40:
            return

        cats = list(_CAT_LABELS.keys())
        n = len(cats)
        row_h = H / n
        label_w = 70
        bar_max = W - label_w - 60   # 오른쪽 수치 공간
        bar_cx = label_w + bar_max // 2  # 중앙(0점) x좌표

        for i, cat_val in enumerate(cats):
            y_center = int(row_h * i + row_h / 2)
            score = self._scores.get(cat_val, 0.0)
            label = _CAT_LABELS.get(cat_val, cat_val)

            # 레이블
            self.create_text(label_w - 4, y_center, text=label,
                             anchor="e", fill=_C["subtext"],
                             font=("맑은 고딕", 7))

            # 중앙선
            self.create_line(bar_cx, y_center - row_h//2 + 2,
                             bar_cx, y_center + row_h//2 - 2,
                             fill=_C["border"], width=1)

            # 바
            half_max = bar_max // 2
            bar_len = int(abs(score) / 2.0 * half_max)
            bar_len = min(bar_len, half_max)

            if score > 0.01:
                color = _C["bull_strong"] if score > 1.0 else _C["bull"]
                x1, x2 = bar_cx, bar_cx + bar_len
            elif score < -0.01:
                color = _C["bear_strong"] if score < -1.0 else _C["bear"]
                x1, x2 = bar_cx - bar_len, bar_cx
            else:
                color = _C["border"]
                x1, x2 = bar_cx - 2, bar_cx + 2

            bar_y1 = y_center - max(int(row_h * 0.25), 2)
            bar_y2 = y_center + max(int(row_h * 0.25), 2)
            self.create_rectangle(x1, bar_y1, x2, bar_y2,
                                   fill=color, outline="")

            # 수치
            sc_text = f"{score:+.2f}" if abs(score) > 0.01 else "─"
            sc_color = (_C["bull"] if score > 0.01
                        else _C["bear"] if score < -0.01
                        else _C["neutral"])
            self.create_text(bar_cx + half_max + 4, y_center,
                             text=sc_text, anchor="w",
                             fill=sc_color, font=("Consolas", 7))


# ──────────────────────────────────────────────────────────────────────────────
# 메인 패널
# ──────────────────────────────────────────────────────────────────────────────

class MarketEnvPanel:
    """
    외부환경 실시간 모니터 패널

    사용법:
        panel = MarketEnvPanel(parent, settings)
        panel.refresh()          # 수동 갱신
        # → start_auto_refresh() 는 __init__ 에서 자동 호출
    """

    def __init__(self, parent, settings,
                 on_predict_explain: Optional[Callable] = None,
                 on_mood_update: Optional[Callable] = None):
        self.settings = settings
        self._events  = []
        self._structured_events: list = []
        self._on_explain     = on_predict_explain   # 이벤트→예측 연결 콜백
        self._on_mood_update = on_mood_update        # 메인창 기분바 콜백 (n_bull, n_bear, top_title)
        self._days_back_var: Optional[tk.IntVar] = None
        # 소스 선택 BooleanVar (다이얼로그에서 설정, 없으면 settings.news.news_sources 사용)
        self._source_vars: dict = {}

        # 외부환경 피처 엔지니어 (optional)
        self._feature_engineer = None
        if _EXT_ENV_AVAILABLE:
            try:
                cache_dir = os.path.join(BASE_DIR,
                                          getattr(settings.data, "cache_dir", "cache"))
                fe_cache = os.path.join(cache_dir, "ext_env_accumulator.json")
                self._feature_engineer = ExternalEnvFeatureEngineer(
                    cache_file=fe_cache
                )
            except Exception as e:
                logger.debug(f"ExternalEnvFeatureEngineer 초기화 실패: {e}")

        self._fetcher  = None   # RSS 수집기
        self._pipeline = None   # AI 분류 파이프라인 (통합 단일 경로)

        # 크롤링 콜백 참조 (interval 변경 시 재사용)
        self._on_collected_cb  = None
        self._on_classified_cb = None

        self.frame = ttk.Frame(parent)
        self._build()
        self._start_fetcher()

    # ──────────────────────────────────────────────────────────────────────────
    # 빌드
    # ──────────────────────────────────────────────────────────────────────────

    def _build(self):
        # ── 상단 고정 영역 ─────────────────────────────────────────────────
        top_fr = tk.Frame(self.frame, bg="#1a1a2e")
        top_fr.pack(fill="x", padx=0, pady=0)

        # 배너 행
        banner = tk.Frame(top_fr, bg="#1a1a2e", pady=5)
        banner.pack(fill="x", padx=6)
        tk.Label(banner,
                 text="🌐  외부환경 실시간 모니터",
                 bg="#1a1a2e", fg="#89b4fa",
                 font=("맑은 고딕", 10, "bold")).pack(side="left", padx=10)

        self._status_var = tk.StringVar(value="")
        tk.Label(banner, textvariable=self._status_var,
                 bg="#1a1a2e", fg=_C["subtext"],
                 font=("맑은 고딕", 8)).pack(side="right", padx=8)
        ttk.Button(banner, text="🔄",
                   command=self._on_refresh_click, width=3).pack(side="right", padx=2)
        _saved_days = getattr(self.settings.news, "news_display_days", 3650)
        self._days_back_var = tk.IntVar(value=_saved_days)
        tk.Label(banner, text="일",
                 bg="#1a1a2e", fg=_C["subtext"], font=("맑은 고딕", 8)).pack(side="right")
        days_spin = tk.Spinbox(banner, from_=1, to=36500,   # 최대 100년
                               textvariable=self._days_back_var,
                               width=5, bg="#1e1e2e", fg=_C["text"],
                               buttonbackground="#313244", font=("맑은 고딕", 9))
        days_spin.pack(side="right")
        days_spin.bind("<Return>",        lambda e: self._on_days_change())
        days_spin.bind("<FocusOut>",      lambda e: self._on_days_change())
        days_spin.bind("<<Increment>>",   lambda e: self.frame.after(50, self._on_days_change))
        days_spin.bind("<<Decrement>>",   lambda e: self.frame.after(50, self._on_days_change))
        tk.Label(banner, text="기간:",
                 bg="#1a1a2e", fg=_C["subtext"], font=("맑은 고딕", 8)).pack(side="right")

        # ── 수집 진행 바 (수집 중에만 표시) ──────────────────────────────
        self._collect_bar_fr = tk.Frame(top_fr, bg="#12122a", height=0)
        self._collect_bar_fr.pack(fill="x", padx=6)
        self._collect_bar_fr.pack_propagate(False)
        _cb_inner = tk.Frame(self._collect_bar_fr, bg="#12122a")
        _cb_inner.pack(fill="both", expand=True, padx=6, pady=2)
        self._collect_prog_var = tk.StringVar(value="")
        tk.Label(_cb_inner, textvariable=self._collect_prog_var,
                 bg="#12122a", fg="#89b4fa",
                 font=("맑은 고딕", 8), anchor="w").pack(side="left", fill="x", expand=True)
        self._collect_progress = ttk.Progressbar(_cb_inner, mode="indeterminate", length=200)
        self._collect_progress.pack(side="right", padx=(8, 0))

        # ── 소스 설정 버튼 ────────────────────────────────────────────────
        ttk.Button(banner, text="⚙ 소스",
                   command=self._show_source_dialog,
                   width=7).pack(side="right", padx=2)

        # ── 백필 버튼 (배너에 추가) ──────────────────────────────────────
        ttk.Button(banner, text="📥 백필",
                   command=self._show_backfill_dialog,
                   width=7).pack(side="right", padx=2)

        # ── AI 재분류 버튼 ────────────────────────────────────────────────
        ttk.Button(banner, text="🤖 AI재분류",
                   command=self._show_reclassify_dialog,
                   width=9).pack(side="right", padx=2)

        # 분위기 게이지 (항상 표시)
        gauge_fr = tk.Frame(top_fr, bg=_C["bg2"])
        gauge_fr.pack(fill="x", padx=6, pady=(0, 2))
        self._gauge = MoodGauge(gauge_fr, height=55)
        self._gauge.pack(fill="x", padx=4, pady=3)

        # 이벤트 카운터 + 분위기 요약
        cnt_fr = tk.Frame(top_fr, bg=_C["bg3"])
        cnt_fr.pack(fill="x", padx=6, pady=(0, 2))
        self._bull_cnt  = tk.Label(cnt_fr, text="호재: 0",
                                    bg=_C["bg3"], fg=_C["bull"],
                                    font=("맑은 고딕", 8, "bold"))
        self._bull_cnt.pack(side="left", padx=10, pady=3)
        self._bear_cnt  = tk.Label(cnt_fr, text="악재: 0",
                                    bg=_C["bg3"], fg=_C["bear"],
                                    font=("맑은 고딕", 8, "bold"))
        self._bear_cnt.pack(side="left", padx=8)
        self._unc_cnt   = tk.Label(cnt_fr, text="불확실: 0",
                                    bg=_C["bg3"], fg=_C["uncert"],
                                    font=("맑은 고딕", 8))
        self._unc_cnt.pack(side="left", padx=6)
        self._src_cnt = tk.Label(cnt_fr, text="",
                                  bg=_C["bg3"], fg=_C["subtext"],
                                  font=("맑은 고딕", 7))
        self._src_cnt.pack(side="right", padx=6)
        self._total_cnt = tk.Label(cnt_fr, text="총 0건",
                                    bg=_C["bg3"], fg=_C["subtext"],
                                    font=("맑은 고딕", 8))
        self._total_cnt.pack(side="right", padx=10)

        # ── 3탭 Notebook ──────────────────────────────────────────────────
        nb = ttk.Notebook(self.frame)
        nb.pack(fill="both", expand=True, padx=6, pady=(2, 4))

        # 탭 빌드 헬퍼 — 예외 발생 시 탭 자체는 유지, 오류 레이블 표시
        def _safe_add(text: str, builder, *args):
            fr = ttk.Frame(nb)
            nb.add(fr, text=text)
            try:
                builder(fr, *args)
            except Exception as _e:
                logger.error(f"탭 '{text}' 빌드 실패: {_e}", exc_info=True)
                tk.Label(fr, text=f"⚠ 탭 초기화 실패\n{_e}",
                         fg=_C["bear"], bg=_C["bg"],
                         font=("맑은 고딕", 9), justify="left").pack(pady=20, padx=10)

        _safe_add("📰 뉴스 이벤트",   self._build_tab_news)
        _safe_add("📊 카테고리 분석", self._build_tab_categories)
        _safe_add("🔮 시나리오 분석", self._build_tab_scenario)
        _safe_add("🕐 크롤링 현황",   self._build_tab_crawl)

    def _build_tab_news(self, parent):
        """탭1: 섹터히트맵 + 이벤트목록 + 이벤트상세/NLP 분석"""
        # 12카테고리 히트맵
        self._cat_heatmap = CategoryHeatmap(parent)
        self._cat_heatmap.pack(fill="x", padx=4, pady=(4, 2))

        # 메인 분할 (이벤트 목록 | 상세)
        pane = tk.PanedWindow(parent, orient="horizontal",
                              bg=_C["bg3"], sashwidth=4)
        pane.pack(fill="both", expand=True, padx=4, pady=(2, 4))

        # ── 이벤트 목록 (왼쪽) ──────────────────────────────────────────
        left_fr = ttk.LabelFrame(pane, text="이벤트 목록", padding=4)
        pane.add(left_fr, minsize=380)

        # 필터 바
        flt_fr = tk.Frame(left_fr, bg=_C["bg2"])
        flt_fr.pack(fill="x", pady=(0, 3))
        self._filter_var = tk.StringVar(value="전체")
        filters = ["전체", "호재", "악재", "불확실", "FOMC", "금리", "전쟁", "실적"]
        ttk.Label(flt_fr, text="필터:").pack(side="left", padx=4)
        flt_cb = ttk.Combobox(flt_fr, textvariable=self._filter_var,
                               values=filters, width=9, state="readonly")
        flt_cb.pack(side="left", padx=2)
        flt_cb.bind("<<ComboboxSelected>>", lambda e: self._apply_filter())
        self._sort_var = tk.StringVar(value="최신순")
        sort_cb = ttk.Combobox(flt_fr, textvariable=self._sort_var,
                                values=["최신순", "강도순", "관련도순"],
                                width=8, state="readonly")
        sort_cb.pack(side="left", padx=2)
        sort_cb.bind("<<ComboboxSelected>>", lambda e: self._apply_filter())

        # Treeview
        cols = ("감성", "카테고리", "제목", "강도", "소스", "시간")
        self._tree = ttk.Treeview(left_fr, columns=cols,
                                   show="headings", height=18)
        self._tree.heading("감성",    text="감성")
        self._tree.heading("카테고리", text="분류")
        self._tree.heading("제목",    text="제목")
        self._tree.heading("강도",    text="강도")
        self._tree.heading("소스",    text="소스")
        self._tree.heading("시간",    text="시간")
        self._tree.column("감성",    width=46,  anchor="center")
        self._tree.column("카테고리", width=105, anchor="center")
        self._tree.column("제목",    width=230, anchor="w")
        self._tree.column("강도",    width=52,  anchor="center")
        self._tree.column("소스",    width=70,  anchor="center")
        self._tree.column("시간",    width=72,  anchor="center")
        vsb = ttk.Scrollbar(left_fr, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=vsb.set)
        self._tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        self._tree.tag_configure("bull",        foreground=_C["bull"])
        self._tree.tag_configure("bull_strong", foreground=_C["bull_strong"],
                                  background="#1a2a3a")
        self._tree.tag_configure("bear",        foreground=_C["bear"])
        self._tree.tag_configure("bear_strong", foreground=_C["bear_strong"],
                                  background="#3a1a1a")
        self._tree.tag_configure("uncert",      foreground=_C["uncert"])
        self._tree.tag_configure("neutral",     foreground=_C["neutral"])
        self._tree.bind("<<TreeviewSelect>>", self._on_event_select)

        # ── 이벤트 상세 + NLP 분석 (오른쪽) ────────────────────────────
        right_fr = ttk.LabelFrame(pane, text="이벤트 상세 & 분석", padding=6)
        pane.add(right_fr, minsize=280)

        self._detail_title = tk.Label(right_fr, text="이벤트를 선택하세요",
                                       bg=_C["bg"], fg=_C["text"],
                                       font=("맑은 고딕", 9, "bold"),
                                       wraplength=270, justify="left")
        self._detail_title.pack(anchor="w", pady=(0, 4))

        # 속성 그리드
        grid = tk.Frame(right_fr, bg=_C["bg"])
        grid.pack(fill="x", pady=2)
        self._detail_vars = {}
        detail_fields = [
            ("분류",   "type"),
            ("감성",   "sentiment"),
            ("강도",   "intensity"),
            ("관련도", "relevance"),
            ("출처",   "source"),
            ("시각",   "time"),
        ]
        for i, (lbl, key) in enumerate(detail_fields):
            tk.Label(grid, text=f"{lbl}:", bg=_C["bg"], fg=_C["subtext"],
                     font=("맑은 고딕", 8), width=6, anchor="e").grid(
                row=i, column=0, sticky="e", padx=(0, 4), pady=1)
            var = tk.StringVar(value="─")
            tk.Label(grid, textvariable=var, bg=_C["bg"], fg=_C["text"],
                     font=("맑은 고딕", 8), anchor="w").grid(
                row=i, column=1, sticky="w", pady=1)
            self._detail_vars[key] = var

        # 본문
        tk.Label(right_fr, text="내용:", bg=_C["bg"], fg=_C["subtext"],
                 font=("맑은 고딕", 8)).pack(anchor="w", pady=(4, 0))
        self._detail_body = tk.Text(right_fr, height=4,
                                     bg=_C["bg2"], fg=_C["text"],
                                     font=("맑은 고딕", 8),
                                     wrap="word", relief="flat",
                                     state="disabled")
        self._detail_body.pack(fill="x", pady=2)

        # NLP + 예측 영향 분석
        tk.Label(right_fr, text="━━━ NLP / 예측 영향 분석 ━━━",
                 bg=_C["bg"], fg=_C["subtext"],
                 font=("맑은 고딕", 8)).pack(pady=(6, 2))
        self._impact_text = tk.Text(right_fr, height=10,
                                     bg=_C["bg3"], fg=_C["yellow"],
                                     font=("Consolas", 8),
                                     wrap="word", relief="flat",
                                     state="disabled")
        self._impact_text.pack(fill="both", expand=True)

    def _build_tab_categories(self, parent):
        """탭2: 12카테고리 바 차트 + 피처벡터 요약"""
        # 상단: 설명 + 갱신 버튼
        hdr = tk.Frame(parent, bg=_C["bg2"])
        hdr.pack(fill="x", padx=4, pady=(4, 0))
        tk.Label(hdr,
                 text="뉴스 이벤트를 12개 카테고리로 정량화 — 시간 감쇠 가중 점수",
                 bg=_C["bg2"], fg=_C["subtext"],
                 font=("맑은 고딕", 8)).pack(side="left", padx=8)
        ttk.Button(hdr, text="↻ 갱신", width=6,
                   command=self._update_cat_chart).pack(side="right", padx=6, pady=2)

        # 12카테고리 바 차트 (공간 크게 할당)
        self._cat_chart = CategoryBarChart(parent, height=240)
        self._cat_chart.pack(fill="both", expand=True, padx=4, pady=4)

        # 피처벡터 요약 (외부환경 32D 점수)
        feat_fr = ttk.LabelFrame(parent, text="외부환경 피처벡터 요약 (32D)", padding=4)
        feat_fr.pack(fill="x", padx=4, pady=(0, 4))

        self._feat_text = tk.Text(feat_fr, height=5,
                                   bg=_C["bg3"], fg=_C["text"],
                                   font=("Consolas", 7),
                                   wrap="none", relief="flat",
                                   state="disabled")
        vsb2 = ttk.Scrollbar(feat_fr, orient="vertical",
                               command=self._feat_text.yview)
        self._feat_text.configure(yscrollcommand=vsb2.set)
        self._feat_text.pack(side="left", fill="both", expand=True)
        vsb2.pack(side="right", fill="y")

    def _build_tab_scenario(self, parent):
        """탭3: 가상 이벤트 → 예측 수익률 시뮬레이션"""
        if not _EXT_ENV_AVAILABLE:
            tk.Label(parent,
                     text="⚠ features.external_env 모듈이 없어 시나리오 분석을 사용할 수 없습니다.",
                     fg=_C["uncert"], bg=_C["bg"],
                     font=("맑은 고딕", 9)).pack(pady=40)
            return

        # 입력 패널
        inp_fr = ttk.LabelFrame(parent, text="가상 이벤트 입력", padding=8)
        inp_fr.pack(fill="x", padx=6, pady=(6, 4))

        row0 = tk.Frame(inp_fr, bg=_C["bg"])
        row0.pack(fill="x", pady=2)
        tk.Label(row0, text="이벤트 유형:", bg=_C["bg"], fg=_C["subtext"],
                 font=("맑은 고딕", 8), width=10, anchor="e").pack(side="left")
        self._scen_type_var = tk.StringVar(value="RATE_DECISION")
        scen_types = ["RATE_DECISION", "CPI", "GDP", "EMPLOYMENT",
                      "WAR", "SANCTION", "EARNINGS", "MA", "IPO",
                      "REGULATION", "OIL", "FX", "VOLATILITY",
                      "AI_TECH", "SEMICONDUCTOR"]
        ttk.Combobox(row0, textvariable=self._scen_type_var,
                     values=scen_types, width=16,
                     state="readonly").pack(side="left", padx=4)

        row1 = tk.Frame(inp_fr, bg=_C["bg"])
        row1.pack(fill="x", pady=2)
        tk.Label(row1, text="방향:", bg=_C["bg"], fg=_C["subtext"],
                 font=("맑은 고딕", 8), width=10, anchor="e").pack(side="left")
        self._scen_dir_var = tk.StringVar(value="호재(Bullish)")
        ttk.Combobox(row1, textvariable=self._scen_dir_var,
                     values=["호재(Bullish)", "악재(Bearish)", "중립(Neutral)"],
                     width=14, state="readonly").pack(side="left", padx=4)

        row2 = tk.Frame(inp_fr, bg=_C["bg"])
        row2.pack(fill="x", pady=2)
        tk.Label(row2, text="강도 (0~1):", bg=_C["bg"], fg=_C["subtext"],
                 font=("맑은 고딕", 8), width=10, anchor="e").pack(side="left")
        self._scen_str_var = tk.DoubleVar(value=0.7)
        tk.Scale(row2, from_=0.0, to=1.0, resolution=0.05, orient="horizontal",
                 variable=self._scen_str_var, length=180,
                 bg=_C["bg"], fg=_C["text"], highlightthickness=0,
                 troughcolor=_C["bg3"], activebackground=_C["bull"]).pack(side="left", padx=4)
        tk.Label(row2, textvariable=self._scen_str_var,
                 bg=_C["bg"], fg=_C["text"],
                 font=("맑은 고딕", 8), width=4).pack(side="left")

        row3 = tk.Frame(inp_fr, bg=_C["bg"])
        row3.pack(fill="x", pady=2)
        tk.Label(row3, text="기본 수익률(%):", bg=_C["bg"], fg=_C["subtext"],
                 font=("맑은 고딕", 8), width=10, anchor="e").pack(side="left")
        self._scen_base_var = tk.DoubleVar(value=0.0)
        tk.Spinbox(row3, from_=-20.0, to=20.0, increment=0.5,
                   textvariable=self._scen_base_var,
                   format="%.1f", width=8,
                   bg=_C["bg"], fg=_C["text"],
                   buttonbackground=_C["bg3"],
                   font=("맑은 고딕", 9)).pack(side="left", padx=4)
        tk.Label(row3, text="% (현재 모델 예측 기반 수익률)",
                 bg=_C["bg"], fg=_C["subtext"],
                 font=("맑은 고딕", 7)).pack(side="left", padx=4)

        # 이벤트 수 (여러 개 동시 가정 가능)
        row4 = tk.Frame(inp_fr, bg=_C["bg"])
        row4.pack(fill="x", pady=2)
        tk.Label(row4, text="이벤트 수:", bg=_C["bg"], fg=_C["subtext"],
                 font=("맑은 고딕", 8), width=10, anchor="e").pack(side="left")
        self._scen_count_var = tk.IntVar(value=1)
        tk.Spinbox(row4, from_=1, to=5, textvariable=self._scen_count_var,
                   width=4, bg=_C["bg"], fg=_C["text"],
                   buttonbackground=_C["bg3"],
                   font=("맑은 고딕", 9)).pack(side="left", padx=4)
        tk.Label(row4, text="개 동시 발생 가정",
                 bg=_C["bg"], fg=_C["subtext"],
                 font=("맑은 고딕", 7)).pack(side="left", padx=4)

        # 실행 버튼
        btn_row = tk.Frame(inp_fr, bg=_C["bg"])
        btn_row.pack(fill="x", pady=(6, 2))
        ttk.Button(btn_row, text="▶  시나리오 시뮬레이션 실행",
                   command=self._run_scenario).pack(side="left", padx=4)
        ttk.Button(btn_row, text="⟳  현재 이벤트 기반 재설정",
                   command=self._reset_scenario).pack(side="left", padx=4)

        # 결과 출력
        res_fr = ttk.LabelFrame(parent, text="시뮬레이션 결과", padding=6)
        res_fr.pack(fill="both", expand=True, padx=6, pady=(0, 6))

        self._scen_result = tk.Text(res_fr, height=14,
                                     bg=_C["bg3"], fg=_C["text"],
                                     font=("Consolas", 9),
                                     wrap="word", relief="flat",
                                     state="disabled")
        vsb3 = ttk.Scrollbar(res_fr, orient="vertical",
                               command=self._scen_result.yview)
        self._scen_result.configure(yscrollcommand=vsb3.set)
        self._scen_result.pack(side="left", fill="both", expand=True)
        vsb3.pack(side="right", fill="y")
        # 결과 색상 태그
        self._scen_result.tag_configure("up",   foreground=_C["bull_strong"])
        self._scen_result.tag_configure("down",  foreground=_C["bear_strong"])
        self._scen_result.tag_configure("head",  foreground=_C["yellow"],
                                         font=("Consolas", 9, "bold"))
        self._scen_result.tag_configure("label", foreground=_C["subtext"])

    # ──────────────────────────────────────────────────────────────────────────
    # 탭4: 크롤링 현황
    # ──────────────────────────────────────────────────────────────────────────

    def _build_tab_crawl(self, parent):
        """탭4: 뉴스 크롤링 및 AI 분류 현황 제어 패널"""
        # ── 내부 상태 변수 (datetime + 간격) ─────────────────────────────
        self._crawl_rss_last_dt      = None   # datetime | None
        self._crawl_pipe_last_dt     = None
        self._crawl_rss_interval_sec = 60     # 초 단위 (기본 1분)
        self._crawl_pipe_interval_sec= 300    # 초 단위 (기본 5분)

        # ── StringVar 초기화 ──────────────────────────────────────────────
        self._crawl_rss_status_var   = tk.StringVar(value="상태: 초기화 중...")
        self._crawl_pipe_status_var  = tk.StringVar(value="상태: 초기화 중...")
        self._crawl_rss_interval_var = tk.StringVar(value="1분")
        self._crawl_pipe_interval_var= tk.StringVar(value="5분")
        self._crawl_rss_count_var    = tk.StringVar(value="누적 수집: -")
        self._crawl_pipe_count_var   = tk.StringVar(value="분류완료: - / 미분류: -")
        self._crawl_rss_last_var     = tk.StringVar(value="마지막 수집: -")
        self._crawl_pipe_last_var    = tk.StringVar(value="마지막 분류: -")
        self._crawl_rss_elapsed_var  = tk.StringVar(value="")
        self._crawl_pipe_elapsed_var = tk.StringVar(value="")
        self._crawl_rss_next_var     = tk.StringVar(value="다음 수집: -")
        self._crawl_pipe_next_var    = tk.StringVar(value="다음 분류: -")
        self._crawl_rss_bar_var      = tk.StringVar(value="")
        self._crawl_pipe_bar_var     = tk.StringVar(value="")
        self._crawl_action_var       = tk.StringVar(value="")

        BG = _C["bg2"]

        # ── 상단 즉시 실행 버튼 행 ────────────────────────────────────────
        top_fr = tk.Frame(parent, bg=BG)
        top_fr.pack(fill="x", padx=8, pady=(8, 4))

        ttk.Button(
            top_fr, text="⚡  지금 바로 수집+분류",
            command=self._manual_crawl_now,
        ).pack(side="left", padx=(0, 6))

        ttk.Button(
            top_fr, text="🔄  RSS만 수집",
            command=lambda: threading.Thread(
                target=self._manual_rss_only,
                daemon=True, name="manual-rss-only").start(),
        ).pack(side="left", padx=(0, 6))

        ttk.Button(
            top_fr, text="🤖  분류만 실행",
            command=lambda: threading.Thread(
                target=self._manual_pipe_only,
                daemon=True, name="manual-pipe-only").start(),
        ).pack(side="left", padx=(0, 8))

        tk.Label(top_fr, textvariable=self._crawl_action_var,
                 bg=BG, fg=_C["yellow"],
                 font=("맑은 고딕", 8)).pack(side="left")

        # ── 2열 레이아웃: RSS | Pipeline ──────────────────────────────────
        cols = tk.Frame(parent, bg=BG)
        cols.pack(fill="x", padx=8, pady=4)
        cols.columnconfigure(0, weight=1)
        cols.columnconfigure(1, weight=1)

        # ──────────────────────────────────────────────────────────────────
        # RSS 섹션
        # ──────────────────────────────────────────────────────────────────
        rss_fr = ttk.LabelFrame(cols, text="📡 RSS 자동 수집", padding=8)
        rss_fr.grid(row=0, column=0, sticky="nsew", padx=(0, 4))

        # 간격 선택
        r_int = tk.Frame(rss_fr, bg=_C["bg"])
        r_int.pack(fill="x", pady=(0, 2))
        tk.Label(r_int, text="수집 간격:",
                 bg=_C["bg"], fg=_C["subtext"],
                 font=("맑은 고딕", 8)).pack(side="left")
        rss_cb = ttk.Combobox(r_int, textvariable=self._crawl_rss_interval_var,
                               values=["1분", "5분", "10분", "30분"],
                               width=6, state="readonly")
        rss_cb.pack(side="left", padx=4)
        rss_cb.bind("<<ComboboxSelected>>",
                    lambda e: self.frame.after(0, self._change_rss_interval))

        # 상태
        tk.Label(rss_fr, textvariable=self._crawl_rss_status_var,
                 bg=_C["bg"], fg=_C["green"],
                 font=("맑은 고딕", 8)).pack(anchor="w", pady=(2, 0))

        # 마지막 수집 시각 + 경과 시간
        r_last_row = tk.Frame(rss_fr, bg=_C["bg"])
        r_last_row.pack(fill="x")
        tk.Label(r_last_row, textvariable=self._crawl_rss_last_var,
                 bg=_C["bg"], fg=_C["subtext"],
                 font=("맑은 고딕", 7)).pack(side="left")
        tk.Label(r_last_row, textvariable=self._crawl_rss_elapsed_var,
                 bg=_C["bg"], fg=_C["yellow"],
                 font=("맑은 고딕", 7)).pack(side="left", padx=(4, 0))

        # 진행 바 + 다음 수집까지 카운트다운
        tk.Label(rss_fr, textvariable=self._crawl_rss_bar_var,
                 bg=_C["bg"], fg="#89dceb",
                 font=("Consolas", 8)).pack(anchor="w")
        tk.Label(rss_fr, textvariable=self._crawl_rss_next_var,
                 bg=_C["bg"], fg=_C["bull"],
                 font=("맑은 고딕", 8, "bold")).pack(anchor="w")

        # 누적 수집 건수
        tk.Label(rss_fr, textvariable=self._crawl_rss_count_var,
                 bg=_C["bg"], fg=_C["text"],
                 font=("맑은 고딕", 8)).pack(anchor="w", pady=(4, 0))

        # ──────────────────────────────────────────────────────────────────
        # Pipeline 섹션
        # ──────────────────────────────────────────────────────────────────
        pipe_fr = ttk.LabelFrame(cols, text="🤖 AI 자동 분류", padding=8)
        pipe_fr.grid(row=0, column=1, sticky="nsew")

        # 간격 선택
        p_int = tk.Frame(pipe_fr, bg=_C["bg"])
        p_int.pack(fill="x", pady=(0, 2))
        tk.Label(p_int, text="분류 간격:",
                 bg=_C["bg"], fg=_C["subtext"],
                 font=("맑은 고딕", 8)).pack(side="left")
        pipe_cb = ttk.Combobox(p_int, textvariable=self._crawl_pipe_interval_var,
                                values=["5분", "10분", "30분", "1시간"],
                                width=6, state="readonly")
        pipe_cb.pack(side="left", padx=4)
        pipe_cb.bind("<<ComboboxSelected>>",
                     lambda e: self.frame.after(0, self._change_pipe_interval))

        # 상태
        tk.Label(pipe_fr, textvariable=self._crawl_pipe_status_var,
                 bg=_C["bg"], fg=_C["green"],
                 font=("맑은 고딕", 8)).pack(anchor="w", pady=(2, 0))

        # 마지막 분류 시각 + 경과 시간
        p_last_row = tk.Frame(pipe_fr, bg=_C["bg"])
        p_last_row.pack(fill="x")
        tk.Label(p_last_row, textvariable=self._crawl_pipe_last_var,
                 bg=_C["bg"], fg=_C["subtext"],
                 font=("맑은 고딕", 7)).pack(side="left")
        tk.Label(p_last_row, textvariable=self._crawl_pipe_elapsed_var,
                 bg=_C["bg"], fg=_C["yellow"],
                 font=("맑은 고딕", 7)).pack(side="left", padx=(4, 0))

        # 진행 바 + 다음 분류까지 카운트다운
        tk.Label(pipe_fr, textvariable=self._crawl_pipe_bar_var,
                 bg=_C["bg"], fg="#a6e3a1",
                 font=("Consolas", 8)).pack(anchor="w")
        tk.Label(pipe_fr, textvariable=self._crawl_pipe_next_var,
                 bg=_C["bg"], fg=_C["bull"],
                 font=("맑은 고딕", 8, "bold")).pack(anchor="w")

        # 분류 건수
        tk.Label(pipe_fr, textvariable=self._crawl_pipe_count_var,
                 bg=_C["bg"], fg=_C["text"],
                 font=("맑은 고딕", 8)).pack(anchor="w", pady=(4, 0))

        # ── 활동 로그 ──────────────────────────────────────────────────────
        log_fr = ttk.LabelFrame(parent, text="📋 활동 로그", padding=6)
        log_fr.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        self._crawl_log_text = tk.Text(
            log_fr, height=12,
            bg=_C["bg3"], fg=_C["text"],
            font=("Consolas", 8),
            wrap="word", relief="flat",
            state="disabled",
        )
        log_vsb = ttk.Scrollbar(log_fr, orient="vertical",
                                 command=self._crawl_log_text.yview)
        self._crawl_log_text.configure(yscrollcommand=log_vsb.set)
        self._crawl_log_text.pack(side="left", fill="both", expand=True)
        log_vsb.pack(side="right", fill="y")

        # 색상 태그
        self._crawl_log_text.tag_configure("ts",     foreground="#6c7086")
        self._crawl_log_text.tag_configure("rss",    foreground="#89dceb")
        self._crawl_log_text.tag_configure("pipe",   foreground="#a6e3a1")
        self._crawl_log_text.tag_configure("manual", foreground="#f9e2af")
        self._crawl_log_text.tag_configure("err",    foreground="#f38ba8")
        self._crawl_log_text.tag_configure("info",   foreground="#cdd6f4")

        # ── 1초 주기 카운트다운 타이머 시작 ──────────────────────────────
        self.frame.after(1000, self._crawl_tick)

    # ── 크롤링 헬퍼 메서드 ────────────────────────────────────────────────

    def _crawl_log(self, msg: str, tag: str = "info") -> None:
        """활동 로그에 타임스탬프 포함 메시지 추가 (thread-safe)."""
        def _append():
            try:
                w = self._crawl_log_text
                w.configure(state="normal")
                ts = datetime.datetime.now().strftime("%H:%M:%S")
                w.insert("end", f"[{ts}] ", "ts")
                w.insert("end", msg + "\n", tag)
                w.see("end")
                # 200줄 초과 시 앞부분 자동 삭제
                lines = int(w.index("end-1c").split(".")[0])
                if lines > 200:
                    w.delete("1.0", f"{lines - 200}.0")
                w.configure(state="disabled")
            except Exception:
                pass
        try:
            self.frame.after(0, _append)
        except Exception:
            pass

    # ─────────────────────────────────────────────────────────────────────────
    # 1초 주기 카운트다운 타이머
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _fmt_seconds(sec: float) -> str:
        """초 → 'MM분 SS초' 문자열. 1시간 이상이면 'HH:MM:SS'."""
        sec = max(0, int(sec))
        h, rem = divmod(sec, 3600)
        m, s   = divmod(rem, 60)
        if h:
            return f"{h}시간 {m:02d}분 {s:02d}초"
        if m:
            return f"{m}분 {s:02d}초"
        return f"{s}초"

    @staticmethod
    def _progress_bar(elapsed: float, total: float, width: int = 12) -> str:
        """
        elapsed/total 비율을 Unicode 블록 진행 바로 표현.
        예: '▓▓▓▓▓▓░░░░░░  50%'
        """
        ratio = min(1.0, elapsed / max(total, 1))
        filled = int(ratio * width)
        empty  = width - filled
        pct    = int(ratio * 100)
        return f"{'▓' * filled}{'░' * empty}  {pct:3d}%"

    def _crawl_tick(self) -> None:
        """1초마다 호출 — 경과시간/카운트다운/진행 바 업데이트."""
        try:
            if not self.frame.winfo_exists():
                return
        except Exception:
            return

        now = datetime.datetime.now()

        # ── RSS ──────────────────────────────────────────────────────────
        try:
            rss_sec = self._crawl_rss_interval_sec
            if self._crawl_rss_last_dt:
                elapsed_rss = (now - self._crawl_rss_last_dt).total_seconds()
                remaining   = max(0.0, rss_sec - elapsed_rss)
                self._crawl_rss_elapsed_var.set(
                    f"({self._fmt_seconds(elapsed_rss)} 전)")
                self._crawl_rss_bar_var.set(
                    self._progress_bar(elapsed_rss, rss_sec))
                if remaining > 0:
                    self._crawl_rss_next_var.set(
                        f"다음 수집: {self._fmt_seconds(remaining)} 후")
                else:
                    self._crawl_rss_next_var.set("다음 수집: 곧 실행...")
            else:
                self._crawl_rss_elapsed_var.set("")
                self._crawl_rss_bar_var.set("░" * 12 + "    0%")
                self._crawl_rss_next_var.set(
                    f"다음 수집: 초기화 대기 중 ({self._fmt_seconds(rss_sec)} 간격)")
        except Exception:
            pass

        # ── Pipeline ─────────────────────────────────────────────────────
        try:
            pipe_sec = self._crawl_pipe_interval_sec
            if self._crawl_pipe_last_dt:
                elapsed_pipe = (now - self._crawl_pipe_last_dt).total_seconds()
                remaining    = max(0.0, pipe_sec - elapsed_pipe)
                self._crawl_pipe_elapsed_var.set(
                    f"({self._fmt_seconds(elapsed_pipe)} 전)")
                self._crawl_pipe_bar_var.set(
                    self._progress_bar(elapsed_pipe, pipe_sec))
                if remaining > 0:
                    self._crawl_pipe_next_var.set(
                        f"다음 분류: {self._fmt_seconds(remaining)} 후")
                else:
                    self._crawl_pipe_next_var.set("다음 분류: 곧 실행...")
            else:
                self._crawl_pipe_elapsed_var.set("")
                self._crawl_pipe_bar_var.set("░" * 12 + "    0%")
                self._crawl_pipe_next_var.set(
                    f"다음 분류: 초기화 대기 중 ({self._fmt_seconds(pipe_sec)} 간격)")
        except Exception:
            pass

        # 다음 틱 예약
        try:
            self.frame.after(1000, self._crawl_tick)
        except Exception:
            pass

    def _crawl_set_rss_status(self, fetched: int = -1) -> None:
        """RSS 상태 StringVar 갱신 + 마지막 실행 datetime 기록 (thread-safe)."""
        snap_dt = datetime.datetime.now()

        def _upd():
            try:
                self._crawl_rss_last_dt = snap_dt          # 카운트다운 기준점 갱신
                self._crawl_rss_last_var.set(
                    f"마지막 수집: {snap_dt.strftime('%H:%M:%S')}")
                self._crawl_rss_status_var.set("상태: 실행 중 🟢")
                if fetched >= 0:
                    self._crawl_rss_count_var.set(f"누적 수집: {fetched:,}건")
            except Exception:
                pass
        try:
            self.frame.after(0, _upd)
        except Exception:
            pass

    def _crawl_set_pipe_status(self, stats: dict) -> None:
        """파이프라인 상태 StringVar 갱신 + 마지막 실행 datetime 기록 (thread-safe)."""
        snap_dt = datetime.datetime.now()
        ok      = stats.get("classified_ok", stats.get("classified", 0))
        p       = stats.get("pending", 0)

        def _upd():
            try:
                self._crawl_pipe_last_dt = snap_dt          # 카운트다운 기준점 갱신
                self._crawl_pipe_last_var.set(
                    f"마지막 분류: {snap_dt.strftime('%H:%M:%S')}")
                self._crawl_pipe_count_var.set(
                    f"분류완료: {ok:,}건 / 미분류: {p:,}건")
                self._crawl_pipe_status_var.set("상태: 실행 중 🟢")
            except Exception:
                pass
        try:
            self.frame.after(0, _upd)
        except Exception:
            pass

    def _manual_crawl_now(self) -> None:
        """수동 즉시 수집+분류 실행 (버튼 핸들러)."""
        threading.Thread(
            target=self._run_manual_full,
            daemon=True, name="manual-crawl-full",
        ).start()

    def _run_manual_full(self) -> None:
        """RSS 수집 → AI 분류 → UI 갱신 (백그라운드)."""
        try:
            self.frame.after(0, lambda: self._crawl_action_var.set("수집 중..."))
            self._crawl_log("=== 수동 수집+분류 시작 ===", "manual")

            # 1) RSS 수집
            if self._fetcher:
                self._fetcher.fetch_all(force_refresh=True)
                self._crawl_log("RSS 수집 완료", "rss")
                self._crawl_set_rss_status()
            else:
                self._crawl_log("RSS 수집기 미초기화 — 건너뜀", "err")

            # 2) AI 분류
            if self._pipeline:
                self.frame.after(0, lambda: self._crawl_action_var.set("AI 분류 중..."))
                stats = self._pipeline.classify_pending(batch=300)
                ok    = stats.get("classified_ok", stats.get("classified", 0))
                p     = stats.get("pending", 0)
                self._crawl_log(f"AI 분류 완료: 분류완료 {ok:,}건 / 미분류 {p:,}건", "pipe")
                self._crawl_set_pipe_status(stats)
            else:
                self._crawl_log("AI 분류 파이프라인 미초기화 — 건너뜀", "err")

            # 3) UI 갱신
            self._refresh_structured_events()
            self._crawl_log("=== 수집+분류 완료 ===", "manual")
        except Exception as exc:
            self._crawl_log(f"수동 수집 오류: {exc}", "err")
            logger.warning(f"[크롤링] 수동 수집 오류: {exc}")
        finally:
            self.frame.after(0, lambda: self._crawl_action_var.set(""))

    def _manual_rss_only(self) -> None:
        """RSS 수집만 실행 (백그라운드)."""
        try:
            self.frame.after(0, lambda: self._crawl_action_var.set("RSS 수집 중..."))
            self._crawl_log("RSS 수동 수집 시작", "rss")
            if self._fetcher:
                self._fetcher.fetch_all(force_refresh=True)
                self._crawl_log("RSS 수집 완료", "rss")
                self._crawl_set_rss_status()
            else:
                self._crawl_log("RSS 수집기 미초기화", "err")
        except Exception as exc:
            self._crawl_log(f"RSS 수집 오류: {exc}", "err")
        finally:
            self.frame.after(0, lambda: self._crawl_action_var.set(""))

    def _manual_pipe_only(self) -> None:
        """AI 분류만 실행 (백그라운드)."""
        try:
            self.frame.after(0, lambda: self._crawl_action_var.set("AI 분류 중..."))
            self._crawl_log("AI 분류 수동 실행 시작", "pipe")
            if self._pipeline:
                stats = self._pipeline.classify_pending(batch=300)
                ok    = stats.get("classified_ok", stats.get("classified", 0))
                p     = stats.get("pending", 0)
                self._crawl_log(f"AI 분류 완료: 분류완료 {ok:,}건 / 미분류 {p:,}건", "pipe")
                self._crawl_set_pipe_status(stats)
                self._refresh_structured_events()
            else:
                self._crawl_log("파이프라인 미초기화", "err")
        except Exception as exc:
            self._crawl_log(f"AI 분류 오류: {exc}", "err")
        finally:
            self.frame.after(0, lambda: self._crawl_action_var.set(""))

    def _change_rss_interval(self) -> None:
        """RSS 수집 간격 변경 및 재시작."""
        _map = {"1분": 1, "5분": 5, "10분": 10, "30분": 30}
        key     = self._crawl_rss_interval_var.get()
        minutes = _map.get(key, 1)
        # 카운트다운 타이머용 간격 업데이트
        self._crawl_rss_interval_sec = minutes * 60
        # 마지막 실행 기준점 초기화 (새 간격으로 재시작)
        self._crawl_rss_last_dt = datetime.datetime.now()
        if self._fetcher and self._on_collected_cb:
            self._fetcher.stop_auto_refresh()
            self._fetcher.start_auto_refresh(
                callback=self._on_collected_cb,
                interval_minutes=minutes,
            )
            self._crawl_rss_status_var.set(f"상태: 실행 중 🟢 ({key} 간격)")
            self._crawl_log(f"RSS 수집 간격 변경: {key}마다 자동 수집", "manual")
        else:
            self._crawl_log("RSS 수집기 미준비 — 초기화 완료 후 재시도하세요", "err")

    def _change_pipe_interval(self) -> None:
        """AI 분류 간격 변경 및 재시작."""
        _map = {"5분": 5, "10분": 10, "30분": 30, "1시간": 60}
        key     = self._crawl_pipe_interval_var.get()
        minutes = _map.get(key, 5)
        # 카운트다운 타이머용 간격 업데이트
        self._crawl_pipe_interval_sec = minutes * 60
        # 마지막 실행 기준점 초기화 (새 간격으로 재시작)
        self._crawl_pipe_last_dt = datetime.datetime.now()
        if self._pipeline and self._on_classified_cb:
            self._pipeline.stop()
            self._pipeline.start_auto(
                interval_min=minutes,
                on_done=self._on_classified_cb,
            )
            self._crawl_pipe_status_var.set(f"상태: 실행 중 🟢 ({key} 간격)")
            self._crawl_log(f"AI 분류 간격 변경: {key}마다 자동 분류", "manual")
        else:
            self._crawl_log("AI 파이프라인 미준비 — 초기화 완료 후 재시도하세요", "err")

    # ──────────────────────────────────────────────────────────────────────────
    # 공개 API
    # ──────────────────────────────────────────────────────────────────────────

    def refresh(self):
        """수동 갱신 트리거"""
        threading.Thread(target=self._fetch_and_update,
                         daemon=True, name="market-env-refresh").start()

    def get_macro_events(self):
        """현재 캐시된 이벤트 목록 반환 (predictor.py 등에서 사용)"""
        return list(self._events)

    def get_macro_feature(self):
        """
        현재 이벤트로 32차원 매크로 피처 벡터 계산.
        Returns: np.ndarray (32,)
        """
        try:
            from features.macro_features import MacroFeatureBuilder
            return MacroFeatureBuilder().build(self._events)
        except Exception:
            import numpy as np
            return np.zeros(32, dtype=np.float32)

    def get_external_feature(self, extended: bool = False):
        """
        외부환경 분석 모듈로 32D(기본) 또는 64D(extended) 피처 벡터 반환.
        Returns: np.ndarray
        """
        import numpy as np
        if self._feature_engineer is None:
            dim = 64 if extended else 32
            return np.zeros(dim, dtype=np.float32)
        try:
            return self._feature_engineer.get_features(extended=extended)
        except Exception:
            dim = 64 if extended else 32
            return np.zeros(dim, dtype=np.float32)

    def get_structured_events(self) -> list:
        """현재 캐시된 StructuredEvent 목록 반환"""
        return list(self._structured_events)

    # ──────────────────────────────────────────────────────────────────────────
    # 내부 메서드 — 데이터 수집
    # ──────────────────────────────────────────────────────────────────────────

    def _on_days_change(self):
        """뉴스 조회 기간 변경 → fetcher + DB 즉시 재로드 + 12카테고리 재분류"""
        days = self._days_back_var.get() if self._days_back_var else 7
        logger.debug(f"[UI] days_back 스핀박스 → {days}일")
        self._status_var.set(f"기간 변경 중... ({days}일)")

        # 설정 저장
        try:
            self.settings.news.news_display_days = days
            from config.settings import save_settings
            save_settings(self.settings)
        except Exception as e:
            logger.debug(f"days_back 설정 저장 실패: {e}")

        # ① 즉시: DB 기존 건수만 빠르게 표시 (분류 전)
        threading.Thread(target=lambda: self._check_db_coverage(days),
                         daemon=True, name="days-db-check").start()

        def _reload():
            # MacroEvent 갱신
            self._fetch_and_update()
            # 12카테고리 StructuredEvent DB 재분류 → 완료 후 stats 갱신
            self._refresh_structured_events(force_db=True)

        threading.Thread(target=_reload, daemon=True, name="days-change-refresh").start()

    def _check_db_coverage(self, days: int):
        """DB에 요청 기간의 실제 데이터가 있는지 확인하고 상태 표시"""
        try:
            from data.news_db import get_news_db
            import datetime as _dt
            db = get_news_db(getattr(self.settings.news, "db_path", None))
            since = _dt.datetime.now() - _dt.timedelta(days=days)
            rows = db.get_raw(since_dt=since, limit=1)
            # 가장 오래된 기사 날짜 확인
            oldest_row = db._conn().execute(
                "SELECT MIN(published_at) FROM news_raw"
            ).fetchone()
            oldest_str = (oldest_row[0] or "")[:10] if oldest_row else ""
            total_in_range = db._conn().execute(
                "SELECT COUNT(*) FROM news_raw WHERE published_at >= ?",
                (since.isoformat(),)
            ).fetchone()[0]

            if total_in_range == 0:
                msg = f"⚠ {days}일 범위 DB 데이터 없음 — 백필 필요 (DB 최초: {oldest_str or '없음'})"
            else:
                msg = f"DB: {days}일 범위 {total_in_range:,}건 (최초 {oldest_str})"

            if self.frame.winfo_exists():
                self.frame.after(0, lambda m=msg: self._status_var.set(m))
        except Exception:
            pass

    def _start_fetcher(self):
        """뉴스 수집 + AI 분류 파이프라인 시작 (통합 단일 경로)"""
        def _init():
            try:
                logger.info("[외부환경] _init() 시작")
                self.frame.after(0, lambda: self._status_var.set("DB에서 저장된 데이터 불러오는 중..."))

                # ① 즉시 DB에서 전체 분류 데이터 빠르게 로드 (날짜 필터 없음)
                try:
                    from data.news_db import get_news_db
                    from features.external_env.event_structure import StructuredEvent
                    db_path_early = getattr(self.settings.news, "db_path", None)
                    logger.info(f"[외부환경] ① DB 빠른 로드 시작 — db_path={db_path_early}")
                    db_early = get_news_db(db_path_early)

                    _conn = db_early._conn()
                    _raw_cnt = _conn.execute("SELECT COUNT(*) FROM news_raw").fetchone()[0]
                    _evt_cnt = _conn.execute("SELECT COUNT(*) FROM news_events").fetchone()[0]
                    logger.info(f"[외부환경]   news_raw={_raw_cnt:,}건, news_events={_evt_cnt:,}건")

                    # 날짜 필터 없이 최신순 5000건 (단일 쿼리, 즉시 반환)
                    logger.info("[외부환경]   get_events 호출 (전체, limit=5000)")
                    raw_rows = db_early.get_events(only_representative=False, limit=5000)
                    logger.info(f"[외부환경]   get_events 반환: {len(raw_rows)}행")

                    s_events = []
                    seen_e: set = set()
                    parse_err = 0
                    for row in raw_rows:
                        try:
                            eid = row.get("event_id", "")
                            if not eid or eid in seen_e:
                                continue
                            seen_e.add(eid)
                            s_events.append(StructuredEvent.from_dict({
                                "event_id":         eid,
                                "title":            row.get("title", ""),
                                "timestamp":        row.get("published_at", ""),
                                "categories":       row.get("categories", []),
                                "primary_cat":      row.get("primary_category", ""),
                                "event_type":       row.get("event_type", ""),
                                "impact_direction": row.get("impact_direction", 0),
                                "impact_strength":  float(row.get("impact_strength", 0.0)),
                                "confidence":       float(row.get("confidence", 0.5)),
                                "target_sectors":   row.get("target_sectors", []),
                                "duration":         row.get("duration", "short"),
                                "keywords":         row.get("keywords", []),
                                "sentiment_score":  float(row.get("sentiment_score", 0.0)),
                                "importance":       0.5,
                                "external_score":   float(row.get("computed_score", 0.0)),
                            }))
                        except Exception:
                            parse_err += 1
                    logger.info(f"[외부환경]   변환: {len(s_events)}건 성공 / {parse_err}건 실패")

                    if s_events:
                        self._structured_events = s_events
                        # ingest는 블로킹이므로 백그라운드 실행
                        if self._feature_engineer:
                            _fe = self._feature_engineer
                            _ev = s_events[:]
                            threading.Thread(
                                target=lambda: _fe.ingest(_ev),
                                daemon=True, name="fe-ingest-early"
                            ).start()
                        self._update_stats_label(0, len(s_events))  # days=0 → 전체
                        self.frame.after(0, self._update_cat_heatmap)
                        self.frame.after(0, self._update_cat_chart)
                        self.frame.after(0, lambda: self._update_ui([]))
                        self.frame.after(0, lambda n=len(s_events):
                            self._status_var.set(f"DB 로드 완료: {n:,}건 (파이프라인 초기화 중...)"))
                        logger.info(f"[외부환경]   UI 갱신 예약 완료 ({len(s_events):,}건)")
                    else:
                        logger.info("[외부환경]   news_events 비어있음")
                except Exception as e:
                    logger.info(f"[외부환경] ① DB 빠른 로드 실패: {e}", exc_info=True)

                logger.info("[외부환경] ② 파이프라인 초기화 시작")
                self.frame.after(0, lambda: self._status_var.set("뉴스 파이프라인 초기화 중..."))
                cache_dir = os.path.join(BASE_DIR, self.settings.data.cache_dir)
                db_path   = getattr(self.settings.news, "db_path", None)
                days      = self._days_back_var.get() if self._days_back_var else 7
                logger.info(f"[외부환경]   cache_dir={cache_dir}, db_path={db_path}, days={days}")

                # RSS 수집기
                logger.info("[외부환경]   NewsFetcher 생성 중...")
                from data.news_fetcher import NewsFetcher
                self._fetcher = NewsFetcher(cache_dir=cache_dir, days_back=days,
                                            db_path=db_path)
                logger.info("[외부환경]   NewsFetcher 생성 완료")

                # AI 분류 파이프라인
                logger.info("[외부환경]   get_pipeline 호출 중...")
                from data.news_pipeline import get_pipeline
                self._pipeline = get_pipeline(db_path)
                logger.info("[외부환경]   get_pipeline 완료")

                # 파이프라인 완료 콜백 → UI 갱신 + 크롤링 탭 상태 업데이트
                def _on_classified(stats):
                    ok  = stats.get("classified_ok", 0)
                    tot = stats.get("raw_total", 0)
                    p   = stats.get("pending", 0)
                    logger.info(f"[외부환경] _on_classified: 전체 {tot:,}건 / 완료 {ok:,}건 / 미분류 {p:,}건")
                    self._crawl_log(
                        f"[자동 분류] 완료 {ok:,}건 / 미분류 {p:,}건 (전체 {tot:,}건)",
                        "pipe",
                    )
                    self._crawl_set_pipe_status(stats)
                    self._refresh_structured_events()

                # RSS 수집 완료 콜백 → 즉시 파이프라인 트리거 + 크롤링 탭 로그
                def _on_collected():
                    logger.info("[외부환경] RSS 수집 완료 → pipeline-rss-trigger 실행")
                    # 누적 기사 수 읽기 (fast, 락 없음)
                    try:
                        cnt = len(getattr(self._fetcher, "_accumulated", {}))
                    except Exception:
                        cnt = -1
                    self._crawl_log(
                        f"[RSS 수집] 완료 — 누적 {cnt:,}건" if cnt >= 0
                        else "[RSS 수집] 완료",
                        "rss",
                    )
                    self._crawl_set_rss_status(fetched=cnt)
                    threading.Thread(
                        target=lambda: self._pipeline.classify_pending(batch=50),
                        daemon=True, name="pipeline-rss-trigger",
                    ).start()

                # 콜백을 인스턴스 속성으로 보관 (간격 변경 시 재사용)
                self._on_collected_cb  = _on_collected
                self._on_classified_cb = _on_classified

                # 1분마다 RSS 수집
                logger.info("[외부환경]   start_auto_refresh 시작 (1분 간격)")
                self._fetcher.start_auto_refresh(
                    callback=_on_collected, interval_minutes=1)

                # 5분마다 파이프라인 자동 분류
                logger.info("[외부환경]   pipeline.start_auto 시작 (5분 간격)")
                self._pipeline.start_auto(interval_min=5, on_done=_on_classified)

                # 크롤링 탭 초기 상태 표시 + 카운트다운 기준점 설정
                _init_dt = datetime.datetime.now()

                def _set_crawl_init():
                    try:
                        self._crawl_rss_last_dt       = _init_dt
                        self._crawl_pipe_last_dt      = _init_dt
                        self._crawl_rss_interval_sec  = 60
                        self._crawl_pipe_interval_sec = 300
                        self._crawl_rss_status_var.set("상태: 실행 중 🟢 (1분 간격)")
                        self._crawl_pipe_status_var.set("상태: 실행 중 🟢 (5분 간격)")
                        self._crawl_rss_last_var.set(
                            f"마지막 수집: {_init_dt.strftime('%H:%M:%S')}")
                        self._crawl_pipe_last_var.set(
                            f"마지막 분류: {_init_dt.strftime('%H:%M:%S')}")
                        self._crawl_log("파이프라인 초기화 완료 — 자동 수집 시작", "info")
                    except Exception:
                        pass

                self.frame.after(500, _set_crawl_init)

                # 초기 DB 상태 즉시 표시
                logger.info("[외부환경]   _refresh_structured_events 호출")
                self._refresh_structured_events()
                logger.info("[외부환경] _init() 완료")

            except Exception as e:
                logger.warning(f"파이프라인 초기화 실패: {e}")
                self._load_mock_data()

        threading.Thread(target=_init, daemon=True, name="pipeline-init").start()

    def _refresh_structured_events(self, force_db: bool = False):
        """news_events DB 전체 로드 → UI 갱신. 날짜 필터 없음, ingest 백그라운드."""
        try:
            logger.info(f"[외부환경] _refresh_structured_events 시작 (pipeline={'있음' if self._pipeline else '없음'})")

            # DB에서 직접 로드 — 날짜 필터 없이 최신 5000건
            from data.news_db import get_news_db
            from features.external_env.event_structure import StructuredEvent
            db_path  = getattr(self.settings.news, "db_path", None)
            db       = get_news_db(db_path)
            raw_rows = db.get_events(only_representative=False, limit=5000)
            logger.info(f"[외부환경]   get_events → {len(raw_rows)}행")

            s_events: list = []
            seen: set = set()
            for row in raw_rows:
                try:
                    eid = row.get("event_id", "")
                    if not eid or eid in seen:
                        continue
                    seen.add(eid)
                    s_events.append(StructuredEvent.from_dict({
                        "event_id":         eid,
                        "title":            row.get("title", ""),
                        "timestamp":        row.get("published_at", ""),
                        "categories":       row.get("categories", []),
                        "primary_cat":      row.get("primary_category", ""),
                        "event_type":       row.get("event_type", ""),
                        "impact_direction": row.get("impact_direction", 0),
                        "impact_strength":  float(row.get("impact_strength", 0.0)),
                        "confidence":       float(row.get("confidence", 0.5)),
                        "target_sectors":   row.get("target_sectors", []),
                        "duration":         row.get("duration", "short"),
                        "keywords":         row.get("keywords", []),
                        "sentiment_score":  float(row.get("sentiment_score", 0.0)),
                        "importance":       0.5,
                        "external_score":   float(row.get("computed_score", 0.0)),
                    }))
                except Exception:
                    pass

            if not s_events:
                logger.info("[외부환경]   비어있음 → mock fallback")
                try:
                    from data.news_fetcher import get_mock_structured_events
                    s_events = get_mock_structured_events()
                except Exception:
                    s_events = []

            self._structured_events = s_events

            # ingest는 블로킹이므로 백그라운드 실행
            if self._feature_engineer and s_events:
                _fe = self._feature_engineer
                _ev = s_events[:]
                threading.Thread(
                    target=lambda: _fe.ingest(_ev),
                    daemon=True, name="fe-ingest"
                ).start()

            self._update_stats_label(0, len(s_events))
            self.frame.after(0, self._update_cat_heatmap)
            self.frame.after(0, self._update_cat_chart)
            if not self._events:
                self.frame.after(0, lambda: self._update_ui([]))

            logger.info(f"[외부환경] _refresh_structured_events 완료: {len(s_events)}건")

        except Exception as e:
            logger.info(f"[외부환경] _refresh_structured_events 실패: {e}", exc_info=True)

    def _load_structured_from_db(self, days: int) -> list:
        """DB에서 AI 분류 이벤트 로드 (RSS + 백필 통합).

        우선순위:
          1) news_events (Layer 2) — 이미 AI 분류된 데이터 즉시 반환
          2) news_raw (Layer 1) 미분류 기사 → 12카테고리 분류 → news_events에 영구 저장
        """
        if not _EXT_ENV_AVAILABLE:
            return []
        try:
            import datetime as _dt
            from data.news_db import get_news_db
            from features.external_env.categorizer import NewsEventCategorizer
            from features.external_env.event_structure import StructuredEvent

            db_path = getattr(self.settings.news, "db_path", None)
            db = get_news_db(db_path)
            now = _dt.datetime.now()
            since = now - _dt.timedelta(days=days)

            def _segmented_get_events(since_dt, days_n):
                rows = []
                if days_n <= 7:
                    rows = db.get_events(since_dt=since_dt,
                                         only_representative=False, limit=2000)
                else:
                    seg = 7; d = days_n
                    while d > 0:
                        s_until = now - _dt.timedelta(days=d - seg)
                        s_since = now - _dt.timedelta(days=d)
                        if s_until > now:
                            s_until = now
                        rows.extend(db.get_events(
                            since_dt=s_since, until_dt=s_until,
                            only_representative=False, limit=500
                        ))
                        d -= seg
                return rows

            # ── Step 1: news_events에서 이미 AI 분류된 이벤트 로드 ─────────
            classified_rows = _segmented_get_events(since, days)
            events: list = []
            seen_ids: set = set()

            for row in classified_rows:
                try:
                    d_map = {
                        "event_id":        row.get("event_id", ""),
                        "title":           row.get("title", ""),
                        "timestamp":       row.get("published_at", ""),
                        "categories":      row.get("categories", []),
                        "primary_cat":     row.get("primary_category", ""),
                        "event_type":      row.get("event_type", ""),
                        "impact_direction": row.get("impact_direction", 0),
                        "impact_strength": row.get("impact_strength", 0.0),
                        "confidence":      row.get("confidence", 0.5),
                        "target_sectors":  row.get("target_sectors", []),
                        "duration":        row.get("duration", "short"),
                        "keywords":        row.get("keywords", []),
                        "sentiment_score": row.get("sentiment_score", 0.0),
                        "importance":      0.5,
                        "external_score":  row.get("computed_score", 0.0),
                    }
                    evt = StructuredEvent.from_dict(d_map)
                    if evt.event_id and evt.event_id not in seen_ids:
                        seen_ids.add(evt.event_id)
                        events.append(evt)
                except Exception:
                    pass

            # Step 2 제거: 분류는 NewsPipeline.classify_pending() 전용
            # 이 함수에서 직접 분류하면 event_id 충돌로 news_events 레코드가
            # 호출할 때마다 중복 생성됨 → 파이프라인에 위임

            logger.debug(
                f"[DB→StructuredEvent] {days}일: "
                f"기존분류 {len(classified_rows)}건 + "
                f"신규분류 {len(events) - len(classified_rows)}건 = 총 {len(events)}건"
            )
            return events
        except Exception as e:
            logger.debug(f"DB→StructuredEvent 변환 실패: {e}")
            return []

    def _update_stats_label(self, days: int, classified: int):
        """상태 레이블에 DB 현황 표시 (빠른 단일 스캔 쿼리)"""
        try:
            from data.news_db import get_news_db
            db_path = getattr(self.settings.news, "db_path", None)
            db   = get_news_db(db_path)
            conn = db._conn()

            raw_total = conn.execute("SELECT COUNT(*) FROM news_raw").fetchone()[0]
            classified_ok = conn.execute(
                "SELECT COUNT(DISTINCT raw_id) FROM news_events "
                "WHERE raw_id IS NOT NULL AND raw_id != ''"
            ).fetchone()[0]
            pending = max(0, raw_total - classified_ok)

            msg = (
                f"전체 {raw_total:,}건 │ 분류완료 {classified_ok:,}건 │ "
                f"미분류 {pending:,}건 │ 표시 {classified:,}건"
            )
            logger.info(f"[외부환경] stats: {msg}")
            if self.frame.winfo_exists():
                self.frame.after(0, lambda m=msg: self._status_var.set(m))
        except Exception as e:
            logger.info(f"[외부환경] _update_stats_label 실패: {e}", exc_info=True)

    def _load_mock_data(self):
        """오프라인 목업 데이터 로드"""
        try:
            from data.news_fetcher import get_mock_events, get_mock_structured_events
            events = get_mock_events()
            self._on_events_received(events)
            # StructuredEvent 목업도 로드
            s_events = get_mock_structured_events()
            if s_events:
                self._structured_events = s_events
                if self._feature_engineer:
                    self._feature_engineer.ingest(s_events)
                self.frame.after(0, self._update_cat_heatmap)
                self.frame.after(0, self._update_cat_chart)
        except Exception as e:
            logger.debug(f"목업 데이터 로드 실패: {e}")

    def _on_events_received(self, events):
        """백그라운드 스레드에서 이벤트 수신 → UI 업데이트"""
        # days_back 기준 필터링 (누적된 이벤트 중 기간 내 항목만)
        days = self._days_back_var.get() if self._days_back_var else 7
        cutoff = datetime.datetime.now() - datetime.timedelta(days=days)
        filtered = [e for e in events
                    if e.timestamp is None or
                    (e.timestamp.replace(tzinfo=None) if e.timestamp.tzinfo else e.timestamp) >= cutoff]
        self._events = filtered if filtered else events

        # 메인창 기분바 콜백
        if self._on_mood_update:
            try:
                from features.macro_features import SentimentLabel
                n_bull = sum(1 for e in self._events if e.sentiment == SentimentLabel.POSITIVE)
                n_bear = sum(1 for e in self._events if e.sentiment == SentimentLabel.NEGATIVE)
                top = next((e.title for e in self._events
                             if e.sentiment in (SentimentLabel.POSITIVE,
                                                SentimentLabel.NEGATIVE)), "")
                self._on_mood_update(n_bull, n_bear, top)
            except Exception:
                pass

        self.frame.after(0, lambda evts=self._events: self._update_ui(evts))

    def _fetch_and_update(self):
        """RSS 수집 + 파이프라인 분류 트리거 + UI 갱신 (통합 경로)"""
        try:
            days = self._days_back_var.get() if self._days_back_var else 7
            if self._fetcher:
                self._fetcher.set_days_back(days)
                self._fetcher.fetch_all(force_refresh=True)   # RSS → news_raw
            # 즉시 최근 기사 분류 (50건)
            if self._pipeline:
                self._pipeline.classify_pending(batch=50)
            # UI 갱신
            self._refresh_structured_events()
        except Exception as e:
            logger.debug(f"[fetch_and_update] 실패: {e}")
            self.frame.after(0, lambda: self._status_var.set(f"갱신 실패: {e}"))
            self._load_mock_data()

    def _on_refresh_click(self):
        self._status_var.set("새로고침 중...")
        self._show_progress("뉴스 수집 중...")
        threading.Thread(target=self._refresh_with_progress,
                         daemon=True, name="ext-env-refresh").start()

    def _refresh_with_progress(self):
        """진행바 포함 갱신 — 초기화된 fetcher + pipeline 직접 사용"""
        def _pg(msg: str):
            self.frame.after(0, lambda m=msg: self._collect_prog_var.set(m[:90]))

        try:
            # ① RSS 수집 (이미 초기화된 fetcher — 타임아웃 보장)
            if self._fetcher:
                _pg("RSS 피드 수집 중...")
                try:
                    self._fetcher.fetch_all(force_refresh=True)
                    _pg("RSS 수집 완료")
                except Exception as e:
                    _pg(f"RSS 수집 오류: {e}")

            # ② AI 분류 파이프라인 실행 (batch=100, 빠른 종료)
            classified = 0
            pending    = 0
            if self._pipeline:
                _pg("AI 분류 중...")
                try:
                    stats      = self._pipeline.classify_pending(batch=100, progress_cb=_pg)
                    classified = stats.get("classified", 0)
                    pending    = stats.get("pending", 0)
                except Exception as e:
                    _pg(f"분류 오류: {e}")

            # ③ UI 갱신 (stats_label이 정확한 수치 표시)
            self._refresh_structured_events()

        except Exception as e:
            self.frame.after(0, lambda: self._status_var.set(f"갱신 실패: {e}"))
        finally:
            self.frame.after(0, self._hide_progress)

    def _get_selected_sources(self) -> List[str]:
        """UI 소스 체크박스에서 선택된 소스 목록 반환"""
        src_vars = getattr(self, "_source_vars", {})
        if not src_vars:
            # 체크박스 미설정 시 settings에서 읽음
            return list(getattr(self.settings.news, "news_sources",
                                ["rss", "google"]))
        selected = [k for k, v in src_vars.items() if v.get()]
        return selected if selected else ["rss"]

    def _update_source_counts(self):
        """소스별 기사 수를 소스 카운트 레이블에 표시"""
        if not hasattr(self, "_src_cnt"):
            return
        try:
            days = self._days_back_var.get() if self._days_back_var else 7
            from data.news_service import get_news_service
            svc = get_news_service(
                settings=self.settings,
                cache_dir=os.path.join(BASE_DIR, self.settings.data.cache_dir),
            )
            since = datetime.datetime.now() - datetime.timedelta(days=days)
            summary = svc.get_source_summary(since_dt=since)
            parts = []
            icons = {"rss": "📡", "google": "🔍", "finnhub": "📊", "collector": "🗞"}
            for src, cnt in summary.items():
                if cnt > 0:
                    parts.append(f"{icons.get(src, '●')}{cnt}")
            self._src_cnt.configure(text="  ".join(parts))
        except Exception:
            pass

    def _show_source_dialog(self):
        """뉴스 소스 선택 다이얼로그"""
        dlg = tk.Toplevel(self.frame)
        dlg.title("뉴스 소스 설정")
        dlg.configure(bg="#1e1e2e")
        dlg.resizable(False, False)
        dlg.grab_set()

        tk.Label(dlg, text="수집할 뉴스 소스를 선택하세요",
                 bg="#1e1e2e", fg=_C["text"],
                 font=("맑은 고딕", 9, "bold")).pack(padx=20, pady=(14, 6))

        # 현재 선택 상태 (settings 또는 _source_vars)
        current = set(getattr(self.settings.news, "news_sources", ["rss", "google"]))

        src_frame = tk.Frame(dlg, bg="#1e1e2e")
        src_frame.pack(padx=20, pady=4, fill="x")

        _src_defs = [
            ("rss",       "📡 기본 RSS",      "Yahoo/Reuters/연합뉴스 등 7개 소스 (API 키 불필요)"),
            ("google",    "🔍 Google News",   "키워드 기반 RSS 검색 (API 키 불필요)"),
            ("finnhub",   "📊 Finnhub",       "종목별/시장 뉴스 API (API 키 필요)"),
            ("collector", "🗞 Collector",     "60+ RSS 소스 대량 수집"),
        ]

        dlg_vars: dict = {}
        for src_key, label, desc in _src_defs:
            row = tk.Frame(src_frame, bg="#1e1e2e")
            row.pack(fill="x", pady=3)
            var = tk.BooleanVar(value=(src_key in current))
            dlg_vars[src_key] = var
            tk.Checkbutton(row, text=label, variable=var,
                           bg="#1e1e2e", fg=_C["text"],
                           selectcolor="#313244",
                           activebackground="#1e1e2e",
                           font=("맑은 고딕", 9)).pack(side="left", padx=4)
            tk.Label(row, text=desc,
                     bg="#1e1e2e", fg=_C["subtext"],
                     font=("맑은 고딕", 7)).pack(side="left", padx=8)

        # Finnhub API 키 입력
        ttk.Separator(dlg, orient="horizontal").pack(fill="x", padx=20, pady=8)
        key_fr = tk.Frame(dlg, bg="#1e1e2e")
        key_fr.pack(padx=20, pady=(0, 6), fill="x")
        tk.Label(key_fr, text="Finnhub API 키:",
                 bg="#1e1e2e", fg=_C["text"],
                 font=("맑은 고딕", 9)).pack(side="left", padx=4)
        key_var = tk.StringVar(value=getattr(self.settings.news, "finnhub_api_key", ""))
        key_entry = tk.Entry(key_fr, textvariable=key_var,
                             width=30, bg="#181825", fg=_C["text"],
                             insertbackground=_C["text"],
                             font=("Consolas", 9), show="*")
        key_entry.pack(side="left", padx=6)
        tk.Label(key_fr, text="(finnhub.io 무료 발급)",
                 bg="#1e1e2e", fg=_C["subtext"],
                 font=("맑은 고딕", 7)).pack(side="left")

        # 버튼
        btn_fr = tk.Frame(dlg, bg="#1e1e2e")
        btn_fr.pack(pady=(4, 14))

        def _apply():
            selected = [k for k, v in dlg_vars.items() if v.get()]
            if not selected:
                selected = ["rss"]
            # settings에 저장
            self.settings.news.news_sources = selected
            self.settings.news.finnhub_api_key = key_var.get().strip()
            # _source_vars 갱신
            self._source_vars = {k: v for k, v in dlg_vars.items()}
            # settings 파일 저장
            try:
                from config.settings import save_settings
                save_settings(self.settings)
            except Exception:
                pass
            dlg.destroy()

        ttk.Button(btn_fr, text="✔ 적용", command=_apply, width=10).pack(side="left", padx=6)
        ttk.Button(btn_fr, text="취소", command=dlg.destroy, width=8).pack(side="left", padx=4)

    def _show_progress(self, msg: str = ""):
        """진행 바 표시"""
        try:
            if self._collect_bar_fr.winfo_exists():
                self._collect_prog_var.set(msg)
                self._collect_bar_fr.configure(height=24)
                self._collect_progress.start(12)
        except Exception:
            pass

    def _hide_progress(self):
        """진행 바 숨김 — 위젯 소멸 후 호출돼도 안전"""
        try:
            if self._collect_progress.winfo_exists():
                self._collect_progress.stop()
        except Exception:
            pass
        try:
            if self._collect_bar_fr.winfo_exists():
                self._collect_bar_fr.configure(height=0)
        except Exception:
            pass

    def _show_reclassify_dialog(self):
        """날짜 범위 지정 + 진행률 표시 AI 재분류 다이얼로그"""
        import datetime as _dt

        dlg = tk.Toplevel(self.frame)
        dlg.title("🤖 뉴스 AI 재분류")
        dlg.configure(bg="#1e1e2e")
        dlg.resizable(False, False)
        dlg.geometry("480x420")
        dlg.grab_set()

        # ── 날짜 범위 ──────────────────────────────────────────────────
        rng_fr = tk.LabelFrame(dlg, text=" 날짜 범위 ", bg="#1e1e2e",
                               fg=_C["yellow"], font=("맑은 고딕", 9, "bold"),
                               padx=10, pady=8)
        rng_fr.pack(fill="x", padx=14, pady=(12, 6))

        today = _dt.date.today()
        row0 = tk.Frame(rng_fr, bg="#1e1e2e")
        row0.pack(fill="x", pady=2)

        tk.Label(row0, text="시작일:", bg="#1e1e2e", fg=_C["text"],
                 font=("맑은 고딕", 9), width=6, anchor="e").pack(side="left")
        start_var = tk.StringVar(value=(today - _dt.timedelta(days=30)).isoformat())
        start_ent = tk.Entry(row0, textvariable=start_var, width=12,
                             bg="#181825", fg=_C["text"],
                             font=("맑은 고딕", 9), relief="flat")
        start_ent.pack(side="left", padx=(4, 12))

        tk.Label(row0, text="종료일:", bg="#1e1e2e", fg=_C["text"],
                 font=("맑은 고딕", 9), width=6, anchor="e").pack(side="left")
        end_var = tk.StringVar(value=today.isoformat())
        end_ent = tk.Entry(row0, textvariable=end_var, width=12,
                           bg="#181825", fg=_C["text"],
                           font=("맑은 고딕", 9), relief="flat")
        end_ent.pack(side="left", padx=4)

        # 빠른 기간 선택 버튼
        quick_fr = tk.Frame(rng_fr, bg="#1e1e2e")
        quick_fr.pack(fill="x", pady=(6, 2))
        tk.Label(quick_fr, text="빠른선택:", bg="#1e1e2e", fg=_C["subtext"],
                 font=("맑은 고딕", 8)).pack(side="left", padx=(0, 6))
        for label, days in [("7일", 7), ("30일", 30), ("3개월", 90),
                             ("1년", 365), ("전체", 3650)]:
            def _set(d=days):
                start_var.set((today - _dt.timedelta(days=d)).isoformat())
                end_var.set(today.isoformat())
                _update_count()
            ttk.Button(quick_fr, text=label, command=_set,
                       width=5).pack(side="left", padx=2)

        # ── 미분류 기사 수 표시 ────────────────────────────────────────
        count_var = tk.StringVar(value="  미분류 기사 수 조회 중...")
        count_lbl = tk.Label(rng_fr, textvariable=count_var,
                             bg="#1e1e2e", fg=_C["green"],
                             font=("맑은 고딕", 9, "bold"))
        count_lbl.pack(anchor="w", pady=(6, 0))

        def _update_count(*_):
            def _do():
                try:
                    from data.news_db import get_news_db
                    db = get_news_db(getattr(self.settings.news, "db_path", None))
                    s = _dt.datetime.fromisoformat(start_var.get())
                    e = _dt.datetime.fromisoformat(end_var.get() + "T23:59:59")
                    unc = db.count_unclassified_raw(since_dt=s, until_dt=e)
                    tot = db._conn().execute(
                        "SELECT COUNT(*) FROM news_raw WHERE published_at BETWEEN ? AND ?",
                        (s.isoformat(), e.isoformat())
                    ).fetchone()[0]
                    msg = f"  미분류: {unc:,}건 / 해당기간 원본: {tot:,}건"
                    if dlg.winfo_exists():
                        dlg.after(0, lambda m=msg: count_var.set(m))
                except Exception as ex:
                    if dlg.winfo_exists():
                        dlg.after(0, lambda m=f"  조회 오류: {ex}": count_var.set(m))
            threading.Thread(target=_do, daemon=True).start()

        start_ent.bind("<FocusOut>", _update_count)
        end_ent.bind("<FocusOut>",   _update_count)
        threading.Thread(target=_update_count, daemon=True).start()

        # ── 진행 현황 ──────────────────────────────────────────────────
        prog_fr = tk.LabelFrame(dlg, text=" 진행 현황 ", bg="#1e1e2e",
                                fg=_C["yellow"], font=("맑은 고딕", 9, "bold"),
                                padx=10, pady=8)
        prog_fr.pack(fill="x", padx=14, pady=6)

        prog_lbl_var = tk.StringVar(value="대기 중...")
        tk.Label(prog_fr, textvariable=prog_lbl_var,
                 bg="#1e1e2e", fg="#89b4fa",
                 font=("맑은 고딕", 9), anchor="w").pack(fill="x")

        prog_bar = ttk.Progressbar(prog_fr, mode="determinate",
                                   length=420, maximum=100)
        prog_bar.pack(fill="x", pady=(4, 2))

        saved_var = tk.StringVar(value="")
        tk.Label(prog_fr, textvariable=saved_var,
                 bg="#1e1e2e", fg=_C["green"],
                 font=("맑은 고딕", 8), anchor="w").pack(fill="x")

        # ── 로그 텍스트 ────────────────────────────────────────────────
        log_text = tk.Text(dlg, height=5, bg="#12122a", fg="#a6e3a1",
                           font=("Consolas", 8), wrap="word",
                           relief="flat", state="disabled")
        log_text.pack(fill="both", expand=True, padx=14, pady=(0, 6))

        def _log(msg: str):
            try:
                if dlg.winfo_exists():
                    def _do():
                        log_text.configure(state="normal")
                        log_text.insert("end", msg + "\n")
                        log_text.see("end")
                        log_text.configure(state="disabled")
                    dlg.after(0, _do)
            except Exception:
                pass

        # ── 버튼 ───────────────────────────────────────────────────────
        btn_fr = tk.Frame(dlg, bg="#1e1e2e")
        btn_fr.pack(pady=(0, 12))

        stop_event = threading.Event()

        def _run():
            start_btn.configure(state="disabled")
            stop_btn.configure(state="normal")
            stop_event.clear()
            prog_bar["value"] = 0
            saved_var.set("")

            def _worker():
                try:
                    import hashlib as _hashlib
                    from data.news_db import get_news_db
                    from features.external_env.categorizer import NewsEventCategorizer

                    db = get_news_db(getattr(self.settings.news, "db_path", None))

                    # 파이프라인 자동 분류 일시 중단 (동시 실행 방지)
                    _pipeline = getattr(self, "_pipeline", None)
                    if _pipeline:
                        _pipeline._stop.set()
                        import time as _time
                        _time.sleep(0.3)
                        _pipeline._stop.clear()
                    # 기존 중복/고아 레코드 먼저 정리 (이전 버전 버그로 생긴 중복 제거)
                    purged = db.purge_orphan_events()
                    if purged > 0:
                        _log(f"🧹 기존 중복/고아 레코드 {purged:,}건 정리 완료")

                    s_dt = _dt.datetime.fromisoformat(start_var.get())
                    e_dt = _dt.datetime.fromisoformat(end_var.get() + "T23:59:59")

                    # 전체 미분류 기사 수 먼저 확인
                    total = db.count_unclassified_raw(since_dt=s_dt, until_dt=e_dt)
                    if total == 0:
                        _log("✅ 해당 기간에 미분류 기사 없음 (이미 모두 분류됨)")
                        return

                    _log(f"📋 미분류 기사 {total:,}건 분류 시작...")
                    categorizer = NewsEventCategorizer()

                    # 배치 단위로 처리 (LIMIT 500씩)
                    BATCH = 500
                    done = 0
                    new_saved = 0

                    while not stop_event.is_set():
                        rows = db.get_unclassified_raw(
                            since_dt=s_dt, until_dt=e_dt, limit=BATCH
                        )
                        if not rows:
                            break

                        batch_dicts: list = []
                        for r in rows:
                            if stop_event.is_set():
                                break
                            try:
                                ts_str = r.get("published_at", "")
                                try:
                                    ts = _dt.datetime.fromisoformat(ts_str)
                                except Exception:
                                    ts = _dt.datetime.now()
                                if ts.tzinfo is not None:
                                    ts = ts.replace(tzinfo=None)

                                evt = categorizer.classify(
                                    title=r.get("title", ""),
                                    summary=r.get("summary", ""),
                                    url=r.get("url", ""),
                                    ts=ts,
                                )
                                raw_id = r.get("id", "")
                                # pipeline과 동일한 event_id 공식: MD5(raw_id)
                                # evt.event_id = MD5(title+ts) 는 실행마다 달라져
                                # INSERT OR REPLACE 가 중복 레코드를 생성함
                                event_id = _hashlib.md5(raw_id.encode()).hexdigest()[:12] if raw_id else _hashlib.md5(evt.title.encode()).hexdigest()[:12]
                                batch_dicts.append({
                                    "event_id":         event_id,
                                    "raw_id":           raw_id,
                                    "published_at":     ts.isoformat(),
                                    "title":            evt.title,
                                    "categories":       [c.value for c in evt.categories],
                                    "primary_category": evt.primary_cat.value if evt.primary_cat else "",
                                    "keywords":         evt.keywords,
                                    "sentiment_score":  evt.sentiment_score,
                                    "impact_direction": evt.impact_direction.value,
                                    "impact_strength":  evt.impact_strength,
                                    "confidence":       evt.confidence,
                                    "duration":         (evt.duration.value
                                                         if hasattr(evt.duration, "value")
                                                         else "short"),
                                    "target_scope":     getattr(evt, "target_market", "market"),
                                    "target_markets":   [],
                                    "target_sectors":   evt.target_sectors,
                                    "target_stocks":    getattr(evt, "target_tickers", []),
                                    "event_type":       evt.event_type,
                                    "cluster_id":       "",
                                    "is_representative": 1,
                                    "repeat_count":     1,
                                    "computed_score":   evt.external_score,
                                    "source_diversity": 0.0,
                                })
                            except Exception:
                                pass

                        if batch_dicts:
                            saved = db.insert_events_bulk(batch_dicts)
                            new_saved += saved
                            done += len(rows)

                        pct = min(int(done / max(total, 1) * 100), 100)
                        msg_lbl = f"분류 중... {done:,}/{total:,}건 ({pct}%)"
                        msg_sv  = f"  → 저장 완료: {new_saved:,}건"
                        if dlg.winfo_exists():
                            dlg.after(0, lambda v=pct:    prog_bar.configure(value=v))
                            dlg.after(0, lambda m=msg_lbl: prog_lbl_var.set(m))
                            dlg.after(0, lambda m=msg_sv:  saved_var.set(m))

                        if len(rows) < BATCH:
                            break  # 더 이상 미분류 없음

                    if stop_event.is_set():
                        _log(f"⏹ 중단됨 — {done:,}/{total:,}건 처리 / {new_saved:,}건 저장")
                    else:
                        _log(f"✅ 완료 — {done:,}건 처리 / {new_saved:,}건 신규 저장")
                        if dlg.winfo_exists():
                            dlg.after(0, lambda: prog_lbl_var.set(f"완료: {done:,}건 처리 / {new_saved:,}건 저장"))
                            dlg.after(0, lambda: prog_bar.configure(value=100))
                        # 패널 갱신
                        threading.Thread(
                            target=lambda: self._refresh_structured_events(force_db=True),
                            daemon=True
                        ).start()
                        _update_count()

                except Exception as e:
                    _log(f"오류: {e}")
                finally:
                    if dlg.winfo_exists():
                        dlg.after(0, lambda: start_btn.configure(state="normal"))
                        dlg.after(0, lambda: stop_btn.configure(state="disabled"))

            threading.Thread(target=_worker, daemon=True, name="reclassify").start()

        start_btn = ttk.Button(btn_fr, text="▶ 분류 시작", command=_run, width=12)
        start_btn.pack(side="left", padx=6)
        stop_btn = ttk.Button(btn_fr, text="⏹ 중단",
                              command=lambda: stop_event.set(),
                              width=8, state="disabled")
        stop_btn.pack(side="left", padx=4)
        ttk.Button(btn_fr, text="닫기", command=dlg.destroy, width=8).pack(side="left", padx=4)

    def _show_backfill_dialog(self):
        """월/년 단위 백필 다이얼로그 — DB 현황 + 이력 + AI 분류 결과 통합 표시"""
        dlg = tk.Toplevel(self.frame)
        dlg.title("📥 뉴스 DB 관리 — 백필 / 현황 / 이력")
        dlg.configure(bg="#1e1e2e")
        dlg.resizable(True, True)
        dlg.geometry("580x680")
        dlg.grab_set()

        nb = ttk.Notebook(dlg)
        nb.pack(fill="both", expand=True, padx=8, pady=8)

        # ════════════════════════════════════════════════════════
        # 탭 1: DB 현황
        # ════════════════════════════════════════════════════════
        tab_stat = ttk.Frame(nb)
        nb.add(tab_stat, text="📊 DB 현황")

        stat_text = tk.Text(tab_stat, height=22,
                             bg="#12122a", fg=_C["text"],
                             font=("Consolas", 9), wrap="none",
                             relief="flat", state="disabled")
        stat_text.pack(fill="both", expand=True, padx=6, pady=6)
        stat_text.tag_configure("head",  foreground=_C["yellow"],
                                font=("Consolas", 9, "bold"))
        stat_text.tag_configure("ok",    foreground=_C["green"])
        stat_text.tag_configure("warn",  foreground=_C["uncert"])
        stat_text.tag_configure("sub",   foreground=_C["subtext"])

        def _load_db_stats():
            try:
                from data.news_db import get_news_db
                db_path = getattr(self.settings.news, "db_path", None)
                db = get_news_db(db_path)
                s = db.get_summary()
                cov = db.get_backfill_coverage()
                log = db.get_backfill_log(limit=20)

                lines: list = []
                lines.append(("═══ DB 저장 현황 ══════════════════════════════\n", "head"))
                lines.append((f"  📁 파일: {db.db_path}\n", "sub"))
                lines.append((f"  💾 크기: {s.get('db_size_mb', 0):.1f} MB\n", "ok"))
                lines.append(("\n", "sub"))
                lines.append(("  [ Layer 1 ] 원본 기사 (news_raw)\n", "head"))
                lines.append((f"    총 {s.get('raw_articles', 0):,}건\n", "ok"))
                lines.append((f"    최초: {s.get('oldest_article','?')}  "
                               f"최신: {s.get('newest_article','?')}\n", "sub"))
                lines.append(("\n", "sub"))
                lines.append(("  [ Layer 2 ] AI 분류 이벤트 (news_events)\n", "head"))
                lines.append((f"    총 {s.get('structured_events', 0):,}건 "
                               f"(12카테고리 분류 완료)\n", "ok"))
                lines.append(("\n", "sub"))
                lines.append(("  [ Layer 3 ] 모델 특징 캐시 (news_feat_cache)\n", "head"))
                lines.append((f"    총 {s.get('feature_caches', 0):,}건\n", "sub"))

                lines.append(("\n═══ 백필 수집 이력 ════════════════════════════\n", "head"))
                if cov.get("total_periods", 0) == 0:
                    lines.append(("  ⚠  백필 미수행 — 과거 데이터 없음\n", "warn"))
                    lines.append(("     [백필] 탭에서 수집하세요\n", "sub"))
                else:
                    lines.append((f"  수집된 기간 수: {cov['total_periods']}개 구간\n", "ok"))
                    lines.append((f"  범위: {cov.get('earliest','?')} ~ "
                                  f"{cov.get('latest_end','?')}\n", "ok"))

                lines.append(("\n  최근 수집 기록 (최신 20개):\n", "head"))
                if log:
                    for r in log:
                        ins  = r.get("inserted", 0)
                        skip = r.get("skipped", 0)
                        tag  = "ok" if ins > 0 else "sub"
                        lines.append((
                            f"  {r['start_date']} ~ {r['end_date']}  "
                            f"신규:{ins:>5}건  중복:{skip:>5}건  "
                            f"({r.get('source','?')})\n", tag
                        ))
                else:
                    lines.append(("  없음\n", "sub"))

                stat_text.configure(state="normal")
                stat_text.delete("1.0", "end")
                for text, tag in lines:
                    stat_text.insert("end", text, tag)
                stat_text.configure(state="disabled")
            except Exception as e:
                stat_text.configure(state="normal")
                stat_text.delete("1.0", "end")
                stat_text.insert("end", f"DB 조회 오류: {e}")
                stat_text.configure(state="disabled")

        btn_row0 = tk.Frame(tab_stat, bg="#1e1e2e")
        btn_row0.pack(fill="x", padx=6, pady=(0, 4))
        ttk.Button(btn_row0, text="↻ 새로고침",
                   command=lambda: threading.Thread(
                       target=_load_db_stats, daemon=True).start(),
                   width=12).pack(side="left", padx=4)

        # 초기 로드
        threading.Thread(target=_load_db_stats, daemon=True,
                         name="db-stat-load").start()

        # ════════════════════════════════════════════════════════
        # 탭 2: 백필
        # ════════════════════════════════════════════════════════
        tab_fill = ttk.Frame(nb)
        nb.add(tab_fill, text="📥 백필 수집")

        tk.Label(tab_fill,
                 text="GDELT API — 과거 뉴스를 기간별로 수집해 DB에 영구 저장합니다.",
                 bg="#1e1e2e", fg=_C["subtext"],
                 font=("맑은 고딕", 9)).pack(padx=16, pady=(12, 2))
        tk.Label(tab_fill,
                 text="✔ 이미 수집된 기간은 자동 스킵  ✔ URL+제목 이중 중복 제거",
                 bg="#1e1e2e", fg=_C["green"],
                 font=("맑은 고딕", 8)).pack(padx=16)

        opt_fr = tk.Frame(tab_fill, bg="#1e1e2e")
        opt_fr.pack(padx=16, pady=10)

        tk.Label(opt_fr, text="수집 단위:", bg="#1e1e2e", fg=_C["text"],
                 font=("맑은 고딕", 9)).grid(row=0, column=0, sticky="e", padx=6)
        gran_var = tk.StringVar(value="month")
        for i, (label, val) in enumerate([("월 단위", "month"), ("연 단위", "year")]):
            tk.Radiobutton(opt_fr, text=label, variable=gran_var, value=val,
                           bg="#1e1e2e", fg=_C["text"],
                           selectcolor="#313244", activebackground="#1e1e2e",
                           font=("맑은 고딕", 9)).grid(row=0, column=i + 1, padx=4)

        tk.Label(opt_fr, text="기간:", bg="#1e1e2e", fg=_C["text"],
                 font=("맑은 고딕", 9)).grid(row=1, column=0, sticky="e",
                                              padx=6, pady=6)
        period_var = tk.IntVar(value=12)
        tk.Spinbox(opt_fr, from_=1, to=60, textvariable=period_var,
                   width=5, bg="#181825", fg=_C["text"],
                   buttonbackground="#313244",
                   font=("맑은 고딕", 9)).grid(row=1, column=1, sticky="w")
        tk.Label(opt_fr, text="개월 / 년 전부터",
                 bg="#1e1e2e", fg=_C["subtext"],
                 font=("맑은 고딕", 8)).grid(row=1, column=2, sticky="w", padx=4)

        # 진행 로그 텍스트
        prog_var = tk.StringVar(value="")
        prog_text = tk.Text(tab_fill, height=10,
                             bg="#12122a", fg="#89b4fa",
                             font=("맑은 고딕", 8), wrap="word",
                             relief="flat", state="disabled")
        prog_text.pack(fill="both", expand=True, padx=16, pady=(0, 4))

        prog_bar = ttk.Progressbar(tab_fill, mode="indeterminate", length=400)
        prog_bar.pack(padx=16, pady=(0, 8))

        btn_fr = tk.Frame(tab_fill, bg="#1e1e2e")
        btn_fr.pack(pady=(0, 12))

        def _dlg_safe(fn):
            try:
                if dlg.winfo_exists():
                    fn()
            except Exception:
                pass

        def _append_log(msg: str):
            try:
                if dlg.winfo_exists():
                    def _do():
                        prog_text.configure(state="normal")
                        prog_text.insert("end", msg + "\n")
                        prog_text.see("end")
                        prog_text.configure(state="disabled")
                    dlg.after(0, _do)
            except Exception:
                pass

        _collector_ref: list = []   # 실행 중 collector 참조 보관 (중단용)

        def _stop_backfill():
            if _collector_ref:
                try:
                    _collector_ref[0]._stop_event.set()
                    _append_log("[중단 요청] 현재 구간 완료 후 중지합니다...")
                    stop_btn.configure(state="disabled")
                except Exception:
                    pass

        def _run_backfill():
            gran = gran_var.get()
            n    = period_var.get()
            start_btn.configure(state="disabled")
            stop_btn.configure(state="normal")
            prog_bar.start(10)
            prog_text.configure(state="normal")
            prog_text.delete("1.0", "end")
            prog_text.configure(state="disabled")

            def _do():
                try:
                    from data.news_db import get_news_db
                    db = get_news_db(
                        getattr(self.settings.news, "db_path", None)
                    )
                    before = db.get_summary()
                    _append_log(
                        f"[시작] DB 현재 상태: "
                        f"원본 {before.get('raw_articles',0):,}건 / "
                        f"AI분류 {before.get('structured_events',0):,}건"
                    )

                    from data.news_collector import get_collector
                    collector = get_collector()
                    # stop_event 초기화 (이전 중단 상태 리셋)
                    collector._stop_event.clear()
                    _collector_ref.clear()
                    _collector_ref.append(collector)

                    if gran == "month":
                        stats = collector.backfill_months(
                            months_back=n, progress_cb=_append_log
                        )
                    else:
                        stats = collector.backfill_years(
                            years_back=n, progress_cb=_append_log
                        )

                    after = db.get_summary()
                    diff  = (after.get("raw_articles", 0) -
                             before.get("raw_articles", 0))
                    _append_log(
                        f"\n[완료] 신규 저장: {stats['total_inserted']:,}건 / "
                        f"중복 제외: {stats['total_skipped']:,}건 / "
                        f"스킵(기수집): {stats.get('periods',0) - stats.get('errors',0)}구간"
                    )
                    _append_log(
                        f"[DB 변화] {before.get('raw_articles',0):,}건 → "
                        f"{after.get('raw_articles',0):,}건 (+{diff:,}건)"
                    )
                    _append_log(
                        f"[DB 범위] {after.get('oldest_article','?')} ~ "
                        f"{after.get('newest_article','?')} / "
                        f"{after.get('db_size_mb',0):.1f} MB"
                    )

                    # 백필 완료 → 파이프라인으로 대량 분류 후 UI 갱신
                    def _classify_after_backfill():
                        if self._pipeline:
                            _append_log("\n[파이프라인] 백필 기사 AI 분류 시작...")
                            stats = self._pipeline.classify_pending(
                                batch=500, progress_cb=_append_log
                            )
                            _append_log(
                                f"[파이프라인] 분류 완료: {stats.get('classified',0)}건 "
                                f"/ 잔여 {stats.get('pending',0)}건"
                            )
                        self._refresh_structured_events()
                        threading.Thread(target=_load_db_stats, daemon=True).start()

                    threading.Thread(target=_classify_after_backfill,
                                     daemon=True, name="backfill-classify").start()
                except Exception as e:
                    _append_log(f"오류: {e}")
                finally:
                    _collector_ref.clear()
                    dlg.after(0, lambda: _dlg_safe(prog_bar.stop))
                    dlg.after(0, lambda: _dlg_safe(
                        lambda: start_btn.configure(state="normal")))
                    dlg.after(0, lambda: _dlg_safe(
                        lambda: stop_btn.configure(state="disabled")))

            threading.Thread(target=_do, daemon=True, name="backfill").start()

        start_btn = ttk.Button(btn_fr, text="▶ 백필 시작",
                               command=_run_backfill, width=14)
        start_btn.pack(side="left", padx=6)
        stop_btn = ttk.Button(btn_fr, text="■ 중단",
                              command=_stop_backfill, width=8, state="disabled")
        stop_btn.pack(side="left", padx=4)
        ttk.Button(btn_fr, text="닫기",
                   command=dlg.destroy, width=8).pack(side="left", padx=4)

    def _update_cat_heatmap(self):
        """12카테고리 히트맵 갱신 (Tab1) — StructuredEvent 우선, MacroEvent fallback"""
        if not hasattr(self, "_cat_heatmap"):
            return
        scores: dict[str, float] = {k: 0.0 for k in _CAT_META}

        if self._structured_events and _EXT_ENV_AVAILABLE:
            # StructuredEvent 기반 (정확한 카테고리 스코어)
            if self._feature_engineer:
                try:
                    cat_scores = self._feature_engineer.get_category_scores()
                    scores.update({cat.value: v for cat, v in cat_scores.items()})
                except Exception:
                    pass
            else:
                import math
                now = datetime.datetime.now()
                for evt in self._structured_events:
                    ts = evt.timestamp
                    if hasattr(ts, "tzinfo") and ts.tzinfo is not None:
                        ts = ts.replace(tzinfo=None)
                    dt_days = (now - ts).total_seconds() / 86400.0
                    decay = math.exp(-0.3 * max(dt_days, 0))
                    for cat in evt.categories:
                        key = cat.value
                        scores[key] = scores.get(key, 0.0) + evt.external_score * decay
        elif self._events:
            # MacroEvent fallback — tags → 카테고리 근사 매핑
            _tag_to_cat = {
                "FOMC": "MonetaryPolicy", "RATE": "MonetaryPolicy",
                "CPI": "Macro", "GDP": "Macro", "EMPLOYMENT": "Macro",
                "WAR": "Geopolitics", "SUPPLY": "Industry",
                "EARNINGS": "Corporate", "POLICY": "Government",
                "FX": "Commodity", "COMMODITY": "Commodity",
                "VOLATILITY": "Sentiment",
            }
            try:
                from features.macro_features import SentimentLabel
                import math
                for evt in self._events:
                    decay = math.exp(-0.1 * max(evt.age_hours / 24.0, 0))
                    sign = (1 if evt.sentiment == SentimentLabel.POSITIVE
                            else -1 if evt.sentiment == SentimentLabel.NEGATIVE
                            else 0)
                    sc = sign * evt.intensity * decay
                    for tag in evt.tags:
                        cat_key = _tag_to_cat.get(tag)
                        if cat_key:
                            scores[cat_key] = scores.get(cat_key, 0.0) + sc
            except Exception:
                pass

        self._cat_heatmap.update_scores(scores)

    def _update_cat_chart(self):
        """12카테고리 바 차트 + 피처벡터 요약 갱신"""
        if not hasattr(self, "_cat_chart"):
            return
        scores: dict[str, float] = {}
        if self._feature_engineer and _EXT_ENV_AVAILABLE:
            try:
                cat_scores = self._feature_engineer.get_category_scores()
                scores = {cat.value: v for cat, v in cat_scores.items()}
            except Exception:
                pass
        elif self._structured_events and _EXT_ENV_AVAILABLE:
            import math
            now = datetime.datetime.now()
            for evt in self._structured_events:
                ts = evt.timestamp
                if hasattr(ts, "tzinfo") and ts.tzinfo is not None:
                    ts = ts.replace(tzinfo=None)
                dt_days = (now - ts).total_seconds() / 86400.0
                decay = math.exp(-0.3 * max(dt_days, 0))
                for cat in evt.categories:
                    scores[cat.value] = scores.get(cat.value, 0.0) + \
                                        evt.external_score * decay
        self._cat_chart.update_scores(scores)
        # 피처벡터 요약 텍스트
        self._update_feat_summary()

    def _update_feat_summary(self):
        """피처벡터 요약 텍스트박스 갱신"""
        if not hasattr(self, "_feat_text"):
            return
        lines = []
        if self._feature_engineer and _EXT_ENV_AVAILABLE:
            try:
                feat = self._feature_engineer.get_features(extended=False)
                _names = [
                    "외부환경총점", "호재강도", "악재강도", "불확실성",
                    "거시경제", "통화정책", "지정학", "산업", "기업", "정부/규제",
                    "수급", "시장이벤트", "기술", "원자재", "금융시장", "심리",
                    "1일밀도", "3일밀도", "7일밀도", "14일밀도",
                    "1일중요도", "3일중요도", "7일중요도", "14일중요도",
                    "센티먼트트렌드", "최강이벤트", "이벤트다양성", "신뢰도",
                    "예비1", "예비2", "예비3", "예비4",
                ]
                for i, (name, val) in enumerate(zip(_names, feat)):
                    bar = ("▓" * int(abs(val) * 5)).ljust(5)
                    sign = "+" if val >= 0 else ""
                    lines.append(f"[{i:02d}] {name:<12} {sign}{val:+.3f}  {bar}")
            except Exception as e:
                lines.append(f"피처 계산 오류: {e}")
        else:
            lines.append("외부환경 모듈 미사용 — 뉴스 수집 후 표시됩니다.")
        self._feat_text.configure(state="normal")
        self._feat_text.delete("1.0", "end")
        self._feat_text.insert("end", "\n".join(lines))
        self._feat_text.configure(state="disabled")

    # ──────────────────────────────────────────────────────────────────────────
    # 내부 메서드 — UI 업데이트
    # ──────────────────────────────────────────────────────────────────────────

    def _update_ui(self, events):
        """이벤트 목록으로 UI 전체 갱신"""
        days = self._days_back_var.get() if self._days_back_var else 7
        now_str = datetime.datetime.now().strftime("%H:%M:%S")
        n_classified = len(self._structured_events) if self._structured_events else 0
        status_msg = (
            f"갱신: {now_str}  "
            f"원본:{len(events)}건  분류:{n_classified}건  ({days}일)"
        )
        self._status_var.set(status_msg)

        # ── 게이지 점수 계산 ────────────────────────────────────────────
        # StructuredEvent 기반 (더 정확한 감성 점수 사용)
        if self._structured_events and _EXT_ENV_AVAILABLE:
            import math
            import datetime as _dt
            now_dt = _dt.datetime.now()
            score_sum = 0.0
            weight_sum = 0.0
            for e in self._structured_events:
                # timezone-aware/naive 혼재 방지 — naive로 통일
                ts = e.timestamp
                if hasattr(ts, "tzinfo") and ts.tzinfo is not None:
                    ts = ts.replace(tzinfo=None)
                dt_h = (now_dt - ts).total_seconds() / 3600.0
                if dt_h < 0:
                    continue
                # 최근 24시간 가중치 2배
                w = (2.0 if dt_h < 24 else 1.0) * math.exp(-0.01 * dt_h)
                score_sum  += e.external_score * w
                weight_sum += w
            score = (score_sum / weight_sum) if weight_sum > 0 else 0.0
            score = max(-1.0, min(1.0, score))
            n_pos = sum(1 for e in self._structured_events
                        if getattr(e.impact_direction, "value", 0) > 0)
            n_neg = sum(1 for e in self._structured_events
                        if getattr(e.impact_direction, "value", 0) < 0)
            n_unc = len(self._structured_events) - n_pos - n_neg
        else:
            from features.macro_features import SentimentLabel
            window_hours = days * 24
            n_pos = sum(1 for e in events
                        if e.sentiment == SentimentLabel.POSITIVE
                        and e.age_hours <= window_hours)
            n_neg = sum(1 for e in events
                        if e.sentiment == SentimentLabel.NEGATIVE
                        and e.age_hours <= window_hours)
            n_unc = sum(1 for e in events
                        if e.sentiment == SentimentLabel.UNCERTAINTY
                        and e.age_hours <= window_hours)
            n_pos_w = sum((2 if e.age_hours < 24 else 1) for e in events
                          if e.sentiment == SentimentLabel.POSITIVE
                          and e.age_hours <= window_hours)
            n_neg_w = sum((2 if e.age_hours < 24 else 1) for e in events
                          if e.sentiment == SentimentLabel.NEGATIVE
                          and e.age_hours <= window_hours)
            total_w = max(n_pos_w + n_neg_w, 1)
            score   = (n_pos_w - n_neg_w) / total_w
        self._gauge.set_score(score)

        # ── 카운터 ───────────────────────────────────────────────────────
        # StructuredEvent가 있으면 그 수치를 우선 사용
        if self._structured_events and _EXT_ENV_AVAILABLE:
            s_evts = self._structured_events
            n_bull = sum(1 for e in s_evts
                         if getattr(e.impact_direction, "value", 0) > 0)
            n_bear = sum(1 for e in s_evts
                         if getattr(e.impact_direction, "value", 0) < 0)
            n_neut = len(s_evts) - n_bull - n_bear
            self._bull_cnt.configure(text=f"호재: {n_bull}건")
            self._bear_cnt.configure(text=f"악재: {n_bear}건")
            self._unc_cnt.configure(text=f"중립/불확실: {n_neut}건")
            self._total_cnt.configure(
                text=f"분류:{len(s_evts):,}건  원본:{len(events)}건 ({days}일)"
            )
        else:
            self._bull_cnt.configure(text=f"호재: {n_pos}건")
            self._bear_cnt.configure(text=f"악재: {n_neg}건")
            self._unc_cnt.configure(text=f"불확실: {n_unc}건")
            self._total_cnt.configure(text=f"총 {len(events)}건 ({days}일)")

        # 배경 분위기 색조 (게이지 점수에 따라 미묘하게 변화)
        self._tint_ui(score)

        # ── 12카테고리 히트맵 ────────────────────────────────────────────
        self._update_cat_heatmap()

        # ── 이벤트 목록 (StructuredEvent 우선 — 12카테고리 표시) ──────
        if self._structured_events and _EXT_ENV_AVAILABLE:
            self._populate_tree_structured(self._structured_events)
        else:
            self._populate_tree(events)

        # ── 소스별 카운트 ────────────────────────────────────────────────
        self._update_source_counts()

        # ── 카테고리 바 차트 갱신 ────────────────────────────────────────
        self._update_cat_chart()

    def _populate_tree(self, events):
        """이벤트 Treeview 채우기 (필터/정렬 적용)"""
        flt  = self._filter_var.get()
        sort = self._sort_var.get()

        from features.macro_features import SentimentLabel

        # 필터
        filtered = []
        for e in events:
            if flt == "전체":
                filtered.append(e)
            elif flt == "호재"   and e.sentiment == SentimentLabel.POSITIVE:
                filtered.append(e)
            elif flt == "악재"   and e.sentiment == SentimentLabel.NEGATIVE:
                filtered.append(e)
            elif flt == "불확실" and e.sentiment == SentimentLabel.UNCERTAINTY:
                filtered.append(e)
            elif flt in e.tags:
                filtered.append(e)

        # 정렬
        if sort == "강도순":
            filtered.sort(key=lambda e: e.intensity, reverse=True)
        elif sort == "관련도순":
            filtered.sort(key=lambda e: e.relevance, reverse=True)
        else:  # 최신순
            filtered.sort(key=lambda e: e.age_hours)

        # 삽입
        for row in self._tree.get_children():
            self._tree.delete(row)

        for evt in filtered[:80]:
            sent  = evt.sentiment.value
            emot  = ("📈" if sent == "positive"
                      else "📉" if sent == "negative"
                      else "❓" if sent == "uncertainty"
                      else "─")
            etype = _TYPE_META.get(evt.event_type.value, _TYPE_META["UNKNOWN"])
            bars  = "█" * int(evt.intensity * 5) + "░" * (5 - int(evt.intensity * 5))
            h     = evt.age_hours
            time_str = (f"{int(h*60)}분 전" if h < 1
                         else f"{int(h)}시간 전" if h < 24
                         else f"{int(h/24)}일 전")

            title_short = evt.title[:38] + ("…" if len(evt.title) > 38 else "")
            src_badge   = _source_badge(getattr(evt, "source", ""))

            tag = ("bull_strong" if sent == "positive" and evt.intensity > 0.5
                    else "bull"  if sent == "positive"
                    else "bear_strong" if sent == "negative" and evt.intensity > 0.5
                    else "bear"  if sent == "negative"
                    else "uncert" if sent == "uncertainty"
                    else "neutral")

            self._tree.insert("", "end",
                values=(emot, etype["label"], title_short, bars, src_badge, time_str),
                tags=(tag,),
                iid=str(id(evt)),  # 이벤트 객체 ID를 row ID로
            )

        # 원본 이벤트 인덱싱 저장 (선택 시 조회용)
        self._filtered_events = {str(id(e)): e for e in filtered[:80]}
        self._tree_mode = "macro"

    def _populate_tree_structured(self, s_events: list):
        """StructuredEvent 목록으로 Treeview 채우기 — 12카테고리 정확 표시"""
        flt  = self._filter_var.get()
        sort = self._sort_var.get()
        import datetime as _dt

        # 방향 → 문자열
        def _dir_str(evt) -> str:
            v = getattr(evt.impact_direction, "value", 0)
            return "positive" if v > 0 else "negative" if v < 0 else "neutral"

        # 필터
        filtered = []
        for e in s_events:
            d = _dir_str(e)
            if flt == "전체":
                filtered.append(e)
            elif flt == "호재"    and d == "positive":
                filtered.append(e)
            elif flt == "악재"    and d == "negative":
                filtered.append(e)
            elif flt == "불확실"  and d == "neutral":
                filtered.append(e)
            elif flt in (e.event_type or ""):
                filtered.append(e)

        # 정렬
        now = _dt.datetime.now()
        def _ts_naive(e):
            ts = e.timestamp
            if ts is None:
                return now
            if hasattr(ts, "tzinfo") and ts.tzinfo is not None:
                ts = ts.replace(tzinfo=None)
            return ts
        if sort == "강도순":
            filtered.sort(key=lambda e: e.impact_strength, reverse=True)
        elif sort == "관련도순":
            filtered.sort(key=lambda e: e.confidence, reverse=True)
        else:  # 최신순
            filtered.sort(key=lambda e: (now - _ts_naive(e)).total_seconds())

        # 삽입
        for row in self._tree.get_children():
            self._tree.delete(row)

        for evt in filtered[:200]:
            d = _dir_str(evt)
            emot = "📈" if d == "positive" else "📉" if d == "negative" else "─"

            # 12카테고리 레이블
            pcat = evt.primary_cat
            cat_label = (
                _CAT_LABELS.get(pcat.value, pcat.value) if pcat
                else "기타"
            )
            # 복수 카테고리가 있으면 1~2개 표시
            if evt.categories and len(evt.categories) > 1:
                extra = _CAT_LABELS.get(evt.categories[1].value,
                                        evt.categories[1].value)
                cat_label = f"{cat_label}+{extra}"

            bars = ("█" * int(evt.impact_strength * 5) +
                    "░" * (5 - int(evt.impact_strength * 5)))
            dt_sec = max((now - _ts_naive(evt)).total_seconds(), 0)
            h = dt_sec / 3600.0
            time_str = (f"{int(dt_sec/60)}분 전" if h < 1
                        else f"{int(h)}시간 전" if h < 24
                        else f"{int(h/24)}일 전")

            title_short = evt.title[:38] + ("…" if len(evt.title) > 38 else "")
            src_badge   = _source_badge(getattr(evt, "source_url", ""))

            str_v = evt.impact_strength
            tag = ("bull_strong" if d == "positive" and str_v > 0.5
                   else "bull"  if d == "positive"
                   else "bear_strong" if d == "negative" and str_v > 0.5
                   else "bear"  if d == "negative"
                   else "neutral")

            self._tree.insert("", "end",
                values=(emot, cat_label, title_short, bars, src_badge, time_str),
                tags=(tag,),
                iid=str(id(evt)),
            )

        self._filtered_events = {str(id(e)): e for e in filtered[:200]}
        self._tree_mode = "structured"

    def _apply_filter(self):
        if self._structured_events and _EXT_ENV_AVAILABLE:
            self._populate_tree_structured(self._structured_events)
        else:
            self._populate_tree(self._events)

    def _on_event_select(self, _event=None):
        """트리 행 선택 → 상세 패널 업데이트 (MacroEvent / StructuredEvent 공용)"""
        sel = self._tree.selection()
        if not sel:
            return
        iid = sel[0]
        evt = self._filtered_events.get(iid)
        if not evt:
            return

        self._detail_title.configure(text=evt.title)

        # ── StructuredEvent 경로 ─────────────────────────────────────
        tree_mode = getattr(self, "_tree_mode", "macro")
        if tree_mode == "structured" and _EXT_ENV_AVAILABLE:
            try:
                pcat    = getattr(evt, "primary_cat", None)
                cat_lbl = (_CAT_LABELS.get(pcat.value, pcat.value)
                           if pcat else "기타")
                dir_val = getattr(evt.impact_direction, "value", 0)
                dir_lbl = ("호재 📈" if dir_val > 0
                           else "악재 📉" if dir_val < 0 else "중립 ─")
                str_v   = getattr(evt, "impact_strength", 0.0)
                conf    = getattr(evt, "confidence", 0.0)
                dur     = getattr(evt, "duration", None)
                dur_map = {"short": "단기(~3일)",
                           "mid":   "중기(~2주)",
                           "long":  "장기(~3개월+)"}
                dur_lbl = dur_map.get(
                    getattr(dur, "value", "short"), "단기"
                )
                self._detail_vars["type"].set(cat_lbl)
                self._detail_vars["sentiment"].set(dir_lbl)
                self._detail_vars["intensity"].set(
                    f"{str_v:.0%}  " + "█" * int(str_v * 5)
                )
                self._detail_vars["relevance"].set(
                    f"{conf:.0%}  ({dur_lbl})"
                )
                src = getattr(evt, "source_url", "") or "─"
                self._detail_vars["source"].set(src[:50])
                self._detail_vars["time"].set(
                    evt.timestamp.strftime("%m/%d %H:%M")
                )
                self._detail_body.configure(state="normal")
                self._detail_body.delete("1.0", "end")
                self._detail_body.insert(
                    "end", getattr(evt, "summary", "") or "요약 없음"
                )
                self._detail_body.configure(state="disabled")

                # NLP/카테고리 분석 패널
                struct_info = self._get_structured_event_info_direct(evt)
                self._impact_text.configure(state="normal")
                self._impact_text.delete("1.0", "end")
                self._impact_text.insert("end", struct_info)
                self._impact_text.configure(state="disabled")
                return
            except Exception as e:
                logger.debug(f"StructuredEvent 상세 표시 실패: {e}")

        # ── MacroEvent 경로 (fallback) ────────────────────────────────
        try:
            etype = _TYPE_META.get(
                getattr(evt.event_type, "value", str(evt.event_type)),
                _TYPE_META["UNKNOWN"]
            )
            sent_map = {"positive": "호재 📈", "negative": "악재 📉",
                        "uncertainty": "불확실 ❓", "neutral": "중립 ─"}
            self._detail_vars["type"].set(etype["label"])
            self._detail_vars["sentiment"].set(
                sent_map.get(getattr(evt.sentiment, "value", ""), "─")
            )
            self._detail_vars["intensity"].set(
                f"{evt.intensity:.0%}  " + "█" * int(evt.intensity * 5)
            )
            self._detail_vars["relevance"].set(f"{evt.relevance:.0%}")
            self._detail_vars["source"].set(getattr(evt, "source", "") or "─")
            self._detail_vars["time"].set(
                evt.timestamp.strftime("%m/%d %H:%M")
            )
            self._detail_body.configure(state="normal")
            self._detail_body.delete("1.0", "end")
            self._detail_body.insert(
                "end", getattr(evt, "body", "") or "본문 없음"
            )
            self._detail_body.configure(state="disabled")

            impact = self._generate_impact_explanation(evt)
            struct_info = self._get_structured_event_info(evt)
            if struct_info:
                impact = struct_info + "\n\n" + impact
            self._impact_text.configure(state="normal")
            self._impact_text.delete("1.0", "end")
            self._impact_text.insert("end", impact)
            self._impact_text.configure(state="disabled")
        except Exception as e:
            logger.debug(f"MacroEvent 상세 표시 실패: {e}")

    def _get_structured_event_info_direct(self, evt) -> str:
        """StructuredEvent 객체에서 직접 NLP/카테고리 분석 요약 생성"""
        lines = ["[12카테고리 외부환경 분석]"]
        cats = [c.value for c in getattr(evt, "categories", [])[:4]]
        if cats:
            cat_labels = [_CAT_LABELS.get(c, c) for c in cats]
            lines.append(f"카테고리: {' / '.join(cat_labels)}")
        dir_val = getattr(evt.impact_direction, "value", 0)
        dir_map = {1: "호재(Bullish)", -1: "악재(Bearish)", 0: "중립(Neutral)"}
        str_v   = getattr(evt, "impact_strength", 0.0)
        lines.append(f"방향: {dir_map.get(dir_val, '중립')}  강도: {str_v:.0%}")
        dur = getattr(evt, "duration", None)
        dur_map = {"short": "단기(~3일)", "mid": "중기(~2주)", "long": "장기(~3개월+)"}
        lines.append(
            f"지속기간: {dur_map.get(getattr(dur,'value','short'), '단기')}"
        )
        sent = getattr(evt, "sentiment_score", 0.0)
        conf = getattr(evt, "confidence", 0.0)
        lines.append(f"NLP 감성 점수: {sent:+.3f}  신뢰도: {conf:.0%}")
        lines.append(f"외부환경 점수: {getattr(evt,'external_score',0.0):+.3f}")
        kws = getattr(evt, "keywords", [])
        if kws:
            lines.append(f"키워드: {', '.join(kws[:5])}")
        sectors = getattr(evt, "target_sectors", [])
        if sectors:
            lines.append(f"관련섹터: {', '.join(sectors)}")
        event_type = getattr(evt, "event_type", "")
        if event_type and event_type != "GENERAL":
            lines.append(f"이벤트유형: {event_type}")
        return "\n".join(lines)

    def _run_scenario(self):
        """시나리오 분석: 가상 이벤트 추가 → 예측 수익률 변화 계산"""
        if not _EXT_ENV_AVAILABLE or not hasattr(self, "_scen_result"):
            return
        try:
            from features.external_env import (
                StructuredEvent, EventCategory, ImpactDirection, EventDuration,
                CATEGORY_WEIGHTS,
            )
            import datetime as _dt

            # 입력 수집
            etype   = self._scen_type_var.get()
            dir_str = self._scen_dir_var.get()
            strength = float(self._scen_str_var.get())
            base_ret = float(self._scen_base_var.get())
            count    = int(self._scen_count_var.get())

            dir_map = {
                "호재(Bullish)": ImpactDirection.BULLISH,
                "악재(Bearish)": ImpactDirection.BEARISH,
                "중립(Neutral)": ImpactDirection.NEUTRAL,
            }
            direction = dir_map.get(dir_str, ImpactDirection.NEUTRAL)

            # 이벤트 유형 → 카테고리 추론
            type_cat_map = {
                "RATE_DECISION": EventCategory.MONETARY_POLICY,
                "CPI":           EventCategory.MACRO,
                "GDP":           EventCategory.MACRO,
                "EMPLOYMENT":    EventCategory.MACRO,
                "WAR":           EventCategory.GEOPOLITICS,
                "SANCTION":      EventCategory.GEOPOLITICS,
                "EARNINGS":      EventCategory.CORPORATE,
                "MA":            EventCategory.CORPORATE,
                "IPO":           EventCategory.MARKET_EVENT,
                "REGULATION":    EventCategory.GOVERNMENT,
                "OIL":           EventCategory.COMMODITY,
                "FX":            EventCategory.COMMODITY,
                "VOLATILITY":    EventCategory.SENTIMENT,
                "AI_TECH":       EventCategory.TECHNOLOGY,
                "SEMICONDUCTOR": EventCategory.TECHNOLOGY,
            }
            cat = type_cat_map.get(etype, EventCategory.MACRO)
            cat_weight = CATEGORY_WEIGHTS.get(cat, 1.0)

            # 가상 이벤트 생성
            hypo_events = []
            for _ in range(count):
                evt = StructuredEvent()
                evt.event_id       = f"scenario_{etype}"
                evt.title          = f"[가상] {etype} 이벤트"
                evt.timestamp      = _dt.datetime.now()
                evt.categories     = [cat]
                evt.primary_cat    = cat
                evt.event_type     = etype
                evt.impact_direction = direction
                evt.impact_strength  = strength
                evt.confidence       = 0.8
                evt.duration         = EventDuration.SHORT
                evt.sentiment_score  = direction.value * strength
                evt.importance       = strength
                evt.compute_score()
                hypo_events.append(evt)

            # 시뮬레이션 (feature_engineer가 있으면 사용, 없으면 직접 계산)
            if self._feature_engineer:
                result = self._feature_engineer.simulate_scenario(
                    hypo_events, base_ret
                )
            else:
                extra_score = sum(e.external_score for e in hypo_events)
                impact = extra_score * 2.0
                new_ret = base_ret + impact
                result = {
                    "base_return":       round(base_ret, 4),
                    "event_impact":      round(impact, 4),
                    "simulated_return":  round(new_ret, 4),
                    "direction":         "UP" if new_ret > 0 else "DOWN",
                    "events_added":      count,
                }

            # 결과 표시
            new_ret  = result["simulated_return"]
            impact   = result["event_impact"]
            dir_icon = "📈" if result["direction"] == "UP" else "📉"

            self._scen_result.configure(state="normal")
            self._scen_result.delete("1.0", "end")

            self._scen_result.insert("end",
                f"═══════════════════════════════════════\n", "head")
            self._scen_result.insert("end",
                f"  시나리오 시뮬레이션 결과\n", "head")
            self._scen_result.insert("end",
                f"═══════════════════════════════════════\n\n", "head")

            self._scen_result.insert("end", "▣ 입력 조건\n", "label")
            self._scen_result.insert("end",
                f"  이벤트 유형 : {etype}\n"
                f"  방향        : {dir_str}\n"
                f"  강도        : {strength:.0%}\n"
                f"  이벤트 수   : {count}건\n"
                f"  카테고리    : {cat.value}  (가중치 {cat_weight:.1f}×)\n\n")

            self._scen_result.insert("end", "▣ 계산 결과\n", "label")
            self._scen_result.insert("end",
                f"  기본 수익률     : {base_ret:+.2f}%\n")
            tag = "up" if impact >= 0 else "down"
            self._scen_result.insert("end",
                f"  이벤트 영향     : {impact:+.2f}%\n", tag)
            tag2 = "up" if new_ret >= 0 else "down"
            self._scen_result.insert("end",
                f"  시뮬레이션 수익률: {new_ret:+.2f}%  {dir_icon}\n\n", tag2)

            self._scen_result.insert("end", "▣ 이벤트 외부환경 점수\n", "label")
            for i, e in enumerate(hypo_events):
                self._scen_result.insert("end",
                    f"  이벤트[{i+1}]  external_score = {e.external_score:+.4f}\n")

            self._scen_result.insert("end",
                f"\n⚠  주의: 선형 모델 근사치 (±2%/point). "
                f"실제 모델은 비선형 반응을 포함합니다.\n", "label")

            self._scen_result.configure(state="disabled")

        except Exception as e:
            self._scen_result.configure(state="normal")
            self._scen_result.delete("1.0", "end")
            self._scen_result.insert("end", f"시뮬레이션 오류: {e}")
            self._scen_result.configure(state="disabled")

    def _reset_scenario(self):
        """현재 외부환경 점수 기반으로 기본 수익률 업데이트"""
        if not hasattr(self, "_scen_base_var"):
            return
        if self._feature_engineer and _EXT_ENV_AVAILABLE:
            try:
                feat = self._feature_engineer.get_features()
                total_score = float(feat[0])   # [0] = 전체 외부환경 점수
                approx_ret = round(total_score * 1.5, 2)
                self._scen_base_var.set(approx_ret)
            except Exception:
                pass

    def _get_structured_event_info(self, evt) -> str:
        """StructuredEvent가 있으면 NLP/카테고리 분석 요약 반환"""
        if not self._structured_events or not _EXT_ENV_AVAILABLE:
            return ""
        # 제목으로 매칭
        title = getattr(evt, "title", "")
        matched = next(
            (s for s in self._structured_events if s.title == title), None
        )
        if matched is None:
            return ""
        lines = ["[NLP + 12카테고리 외부환경 분석]"]
        # 카테고리
        cats = [c.value for c in matched.categories[:4]]
        if cats:
            lines.append(f"카테고리: {' / '.join(cats)}")
        # 방향
        dir_map = {1: "호재(Bullish)", -1: "악재(Bearish)", 0: "중립(Neutral)"}
        dir_val = getattr(matched.impact_direction, "value", 0)
        lines.append(f"방향: {dir_map.get(dir_val, '중립')}  강도: {matched.impact_strength:.0%}")
        # 지속기간
        dur_map = {"short": "단기(~3일)", "mid": "중기(~2주)", "long": "장기(~3개월+)"}
        dur_val = getattr(matched.duration, "value", "short")
        lines.append(f"지속기간: {dur_map.get(dur_val, dur_val)}")
        # 감성 점수
        lines.append(f"NLP 감성 점수: {matched.sentiment_score:+.3f}  신뢰도: {matched.confidence:.0%}")
        # 외부환경 점수
        lines.append(f"외부환경 점수: {matched.external_score:+.3f}")
        # 키워드
        if matched.keywords:
            lines.append(f"키워드: {', '.join(matched.keywords[:5])}")
        return "\n".join(lines)

    def _generate_impact_explanation(self, evt) -> str:
        """이벤트가 예측에 미치는 영향을 한국어로 설명"""
        from features.macro_features import SentimentLabel, MacroEventType

        lines = []
        etype = evt.event_type
        sent  = evt.sentiment
        intens = evt.intensity

        # ── 이벤트 유형별 설명 ────────────────────────────────────────
        effect_map = {
            MacroEventType.FOMC: {
                SentimentLabel.POSITIVE: "금리동결/인하 → 성장주·기술주 상승 기대\n→ 예측 모델: 전반적 상승 편향 반영",
                SentimentLabel.NEGATIVE: "금리인상 → 할인율 상승 → 성장주 밸류에이션 하락\n→ 예측 모델: 하락 리스크 가중",
                SentimentLabel.UNCERTAINTY: "FOMC 불확실성 → 변동성 확대 예상\n→ 모델 σ(불확실성) 증가 반영",
            },
            MacroEventType.RATE: {
                SentimentLabel.POSITIVE: "금리인하 기대 → 기업 자금조달 비용 감소\n→ 금융주·부동산 상승 반영",
                SentimentLabel.NEGATIVE: "금리인상 확정 → 대출금리 상승 → 소비 위축\n→ 내수 관련주 하향 조정",
            },
            MacroEventType.CPI: {
                SentimentLabel.POSITIVE: "CPI 하락/예상 하회 → 인플레 완화 → 금리인하 기대\n→ 성장주 상승 반영",
                SentimentLabel.NEGATIVE: "CPI 예상 상회 → 인플레 지속 → 추가 금리인상 우려\n→ 하락 리스크 반영",
            },
            MacroEventType.WAR: {
                SentimentLabel.NEGATIVE: "전쟁/분쟁 → 공급망 충격 + 원자재 가격 상승\n→ 방산주 상승 / 항공·관광 하락 반영\n→ 전체 시장 변동성 σ 대폭 증가",
            },
            MacroEventType.POLICY: {
                SentimentLabel.POSITIVE: "친기업 정책/규제완화 → 해당 섹터 상승\n→ 정책 수혜주 예측 상향 반영",
                SentimentLabel.NEGATIVE: "규제강화/세금인상 → 관련 기업 비용 증가\n→ 해당 섹터 예측 하향 반영",
            },
            MacroEventType.EARNINGS: {
                SentimentLabel.POSITIVE: "실적 호조 → 해당 기업 직접 반영\n→ 동일 섹터 동반 상승 기대",
                SentimentLabel.NEGATIVE: "실적 부진 → 해당 기업 하락\n→ 섹터 전반 하향 압력",
            },
            MacroEventType.SUPPLY: {
                SentimentLabel.NEGATIVE: "공급망 차질 → 원가 상승 + 생산 차질\n→ 제조업·반도체 하락 반영",
                SentimentLabel.POSITIVE: "공급망 정상화 → 원가 하락 기대\n→ 관련주 상승 반영",
            },
            MacroEventType.FX: {
                SentimentLabel.NEGATIVE: "원화 약세(환율↑) → 수입 물가 상승\n→ 수출주 수혜 / 내수주 하락",
                SentimentLabel.POSITIVE: "원화 강세(환율↓) → 수입 물가 안정\n→ 수입기업 수혜 / 수출주 부담",
            },
            MacroEventType.COMMODITY: {
                SentimentLabel.NEGATIVE: "원자재 급등 → 제조업 마진 압박\n→ 정유·소재주 상승 / 제조업 하락",
                SentimentLabel.POSITIVE: "원자재 하락 → 원가 절감 → 제조업 마진 개선",
            },
        }

        # 매핑된 설명 찾기
        type_effects = effect_map.get(etype, {})
        main_effect = type_effects.get(sent, "이 이벤트는 예측 모델에 직접 반영됩니다.")
        lines.append(f"[이벤트 분류: {etype.value} — {sent.value}]")
        lines.append("")
        lines.append(main_effect)
        lines.append("")

        # 강도 설명
        if intens >= 0.7:
            lines.append(f"⚡ 고강도 이벤트 (강도 {intens:.0%})")
            lines.append("→ 예측 신뢰도(σ) 크게 증가 / 방향성 신호 강화")
        elif intens >= 0.4:
            lines.append(f"📊 중강도 이벤트 (강도 {intens:.0%})")
            lines.append("→ 예측에 보통 수준 반영")
        else:
            lines.append(f"💤 저강도 이벤트 (강도 {intens:.0%})")
            lines.append("→ 예측에 미약하게 반영 (노이즈 수준)")

        # 섹터 관련도
        if evt.sectors:
            lines.append("")
            lines.append(f"영향 섹터: {', '.join(evt.sectors)}")

        # 래그 경고
        if evt.age_hours > 24:
            lines.append("")
            lines.append(f"⏱ 주의: {int(evt.age_hours)}시간 전 이벤트")
            lines.append("→ 시장이 이미 일부 반응했을 수 있음 (래그 효과)")

        return "\n".join(lines)

    def _tint_ui(self, score: float):
        """게이지 점수에 따라 메인 배경에 미묘한 색조 적용"""
        # 극단적인 경우에만 색조 표시
        if score > 0.3:
            tint = "#1e2030"   # 약한 파란 색조 (호재 분위기)
        elif score < -0.3:
            tint = "#2e1e1e"   # 약한 붉은 색조 (악재 분위기)
        else:
            tint = _C["bg"]    # 중립

        try:
            self.frame.configure(style="")
        except Exception:
            pass
