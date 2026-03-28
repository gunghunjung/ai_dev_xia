"""gui/chart_panel.py — Professional HTS-class chart panel (Tkinter Canvas + PIL acceleration)."""
from __future__ import annotations

import math
import os
import sys
import threading
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk

# ─── PIL/Pillow off-screen rendering (optional, speeds up canvas 5-10×) ───────
try:
    from PIL import Image as _PILImage, ImageDraw as _PILDraw, ImageFont as _PILFont
    from PIL.ImageTk import PhotoImage as _TkPhoto
    _PIL_OK = True

    def _mk_font(size: int) -> Any:
        for name in ("Consolas", "consola.ttf", "DejaVuSansMono.ttf", "LiberationMono-Regular.ttf"):
            try:
                return _PILFont.truetype(name, size)
            except Exception:
                pass
        return _PILFont.load_default()

    _PFONT8  = _mk_font(9)
    _PFONT7  = _mk_font(8)
except ImportError:
    _PIL_OK  = False
    _TkPhoto = None
    _PFONT8  = _PFONT7 = None


def _hex_to_rgb(h: str) -> tuple:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _hex_to_rgba(h: str, a: int = 255) -> tuple:
    return (*_hex_to_rgb(h), a)


def _pdash_line(draw: Any, x1: float, y1: float, x2: float, y2: float,
                fill: tuple, dash: tuple = (4, 3), width: int = 1) -> None:
    """PIL에는 dash 선이 없으므로 수동 구현."""
    dist = math.hypot(x2 - x1, y2 - y1)
    if dist < 1:
        return
    on_len, off_len = dash
    seg = on_len + off_len
    t = 0.0
    while t < dist:
        ta = min(t, dist)
        tb = min(t + on_len, dist)
        rx0 = x1 + (x2 - x1) * ta / dist
        ry0 = y1 + (y2 - y1) * ta / dist
        rx1 = x1 + (x2 - x1) * tb / dist
        ry1 = y1 + (y2 - y1) * tb / dist
        draw.line([(rx0, ry0), (rx1, ry1)], fill=fill, width=width)
        t += seg

# ─────────────────────────── colour palette ───────────────────────────────────
BG          = "#181825"
BG_PANEL    = "#1e1e2e"
BG_TOOLBAR  = "#11111b"
CANDLE_UP   = "#a6e3a1"
CANDLE_DOWN = "#f38ba8"
MA20_COL    = "#89b4fa"
MA60_COL    = "#f9e2af"
MA120_COL   = "#fab387"
BB_COL      = "#cba6f7"
BB_FILL     = "#cba6f7"        # stipple fill colour
VOL_UP      = "#a6e3a1"
VOL_DOWN    = "#f38ba8"
VOL_MA_COL  = "#89dceb"
RSI_COL     = "#89b4fa"
RSI_OB_COL  = "#f38ba8"
RSI_OS_COL  = "#a6e3a1"
MACD_COL    = "#89b4fa"
SIGNAL_COL  = "#f9e2af"
HIST_POS    = "#a6e3a1"
HIST_NEG    = "#f38ba8"
CROSS_COL   = "#585b70"
TEXT_COL    = "#cdd6f4"
AXIS_COL    = "#45475a"
TOOLTIP_BG  = "#313244"
TOOLTIP_FG  = "#cdd6f4"
ACTIVE_BTN  = "#45475a"
INACTIVE_BTN = "#11111b"

# ─────────────────────────── helpers ──────────────────────────────────────────

def _fmt_price(v: float) -> str:
    if abs(v) >= 1_000_000:
        return f"{v/1_000_000:.2f}M"
    if abs(v) >= 1_000:
        return f"{v:,.0f}"
    return f"{v:.2f}"


def _fmt_vol(v: float) -> str:
    if v >= 1_000_000:
        return f"{v/1_000_000:.1f}M"
    if v >= 1_000:
        return f"{v/1_000:.0f}K"
    return f"{v:.0f}"


# ─────────────────────────── indicator computation ────────────────────────────

def _calc_indicators(df: pd.DataFrame) -> dict[str, pd.Series | pd.DataFrame]:
    close = df["Close"].astype(float)
    high  = df["High"].astype(float)
    low   = df["Low"].astype(float)
    vol   = df["Volume"].astype(float)

    ind: dict[str, Any] = {}

    # Moving averages
    ind["MA20"]  = close.rolling(20).mean()
    ind["MA60"]  = close.rolling(60).mean()
    ind["MA120"] = close.rolling(120).mean()

    # Bollinger Bands (20, 2)
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std(ddof=0)
    ind["BB_MID"]   = bb_mid
    ind["BB_UPPER"] = bb_mid + 2 * bb_std
    ind["BB_LOWER"] = bb_mid - 2 * bb_std

    # RSI (14) — Wilder smoothing
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    ind["RSI"] = 100 - 100 / (1 + rs)

    # MACD (12, 26, 9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line   = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    ind["MACD"]        = macd_line
    ind["MACD_SIGNAL"] = signal_line
    ind["MACD_HIST"]   = macd_line - signal_line

    # OBV
    obv = (np.sign(close.diff()).fillna(0) * vol).cumsum()
    ind["OBV"] = obv

    # ATR (14)
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low  - close.shift()).abs()
    tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    ind["ATR"] = tr.ewm(com=13, adjust=False).mean()

    # Volume MA20
    ind["VOL_MA20"] = vol.rolling(20).mean()

    return ind


# ──────────────────────────────────────────────────────────────────────────────
# ChartPanel
# ──────────────────────────────────────────────────────────────────────────────

class ChartPanel:
    """
    Tkinter-Canvas 기반 전문 HTS 차트 패널.

    사용법:
        panel = ChartPanel(parent, settings={})
        panel.frame.pack(fill="both", expand=True)
        panel.load_symbol("005930", df)
    """

    # layout ratios (will be recalculated on resize)
    _PRICE_RATIO  = 0.58
    _VOL_RATIO    = 0.18
    _RSI_RATIO    = 0.12
    _MACD_RATIO   = 0.12

    _PAD_LEFT   = 10
    _PAD_RIGHT  = 70   # price axis
    _PAD_TOP    = 10
    _PAD_BOTTOM = 24   # date axis

    def __init__(self, parent: tk.Widget, settings: Any = None) -> None:
        self._settings = settings  # AppSettings or dict or None
        self._lock = threading.Lock()

        # PIL PhotoImage references — must keep alive to prevent GC
        self._chart_photo_refs: dict[str, Any] = {}

        # Portfolio symbol list (set externally via set_symbols)
        self._symbols: list[str] = []
        self._portfolio_var: tk.StringVar | None = None

        self._data: pd.DataFrame | None = None
        self._symbol: str = ""
        self._indicators: dict[str, Any] = {}

        # view state
        self._view_start: int = 0
        self._view_count: int = 80
        self._zoom_level: float = 1.0

        # crosshair
        self._crosshair_x: int | None = None
        self._crosshair_y: int | None = None
        self._drag_start: int | None = None
        self._drag_view_start: int | None = None

        # indicator toggles
        self._show: dict[str, bool] = {
            "MA20":    True,
            "MA60":    True,
            "MA120":   False,
            "BB":      True,
            "Volume":  True,
            "RSI":     True,
            "MACD":    False,
        }

        # active period button tracking
        self._active_period: str = "1Y"
        self._period_btns: dict[str, tk.Button] = {}
        self._ind_btns: dict[str, tk.Button]    = {}

        # build UI
        self.frame = ttk.Frame(parent)
        self.frame.configure(style="Chart.TFrame")
        self._build_styles()
        self._build_toolbar()
        self._build_canvases()
        self._bind_events()

    # ──────────────────────────────────────────────────────────────────────────
    # Styles
    # ──────────────────────────────────────────────────────────────────────────

    def _build_styles(self) -> None:
        style = ttk.Style()
        style.configure("Chart.TFrame",    background=BG)
        style.configure("Toolbar.TFrame",  background=BG_TOOLBAR)

    # ──────────────────────────────────────────────────────────────────────────
    # Toolbar
    # ──────────────────────────────────────────────────────────────────────────

    def _build_toolbar(self) -> None:
        # ── 첫 번째 행: 종목정보 / 기간 버튼 / 지표 토글 ──────────────────────
        tb = tk.Frame(self.frame, bg=BG_TOOLBAR, height=32)
        tb.pack(side="top", fill="x")
        tb.pack_propagate(False)

        # Symbol label
        self._sym_lbl = tk.Label(
            tb, text="─ 종목없음 ─",
            bg=BG_TOOLBAR, fg=TEXT_COL,
            font=("Consolas", 11, "bold"), padx=8,
        )
        self._sym_lbl.pack(side="left")

        # Separator
        tk.Frame(tb, bg=AXIS_COL, width=1).pack(side="left", fill="y", pady=4)

        # Period buttons
        for p in ("1M", "3M", "6M", "1Y", "3Y", "ALL"):
            btn = tk.Button(
                tb, text=p, width=4,
                bg=INACTIVE_BTN, fg=TEXT_COL,
                activebackground=ACTIVE_BTN, activeforeground=TEXT_COL,
                relief="flat", bd=0, padx=4,
                font=("Segoe UI", 9),
                command=lambda pv=p: self.set_period(pv),
            )
            btn.pack(side="left", padx=1, pady=4)
            self._period_btns[p] = btn
        self._highlight_period_btn(self._active_period)

        # Separator
        tk.Frame(tb, bg=AXIS_COL, width=1).pack(side="left", fill="y", pady=4)

        # Indicator toggle buttons
        ind_defs = [
            ("MA",    "MA20"),
            ("BB",    "BB"),
            ("RSI",   "RSI"),
            ("MACD",  "MACD"),
            ("거래량", "Volume"),
        ]
        for label, key in ind_defs:
            btn = tk.Button(
                tb, text=label, width=5,
                bg=ACTIVE_BTN if self._show.get(key, False) else INACTIVE_BTN,
                fg=TEXT_COL,
                activebackground=ACTIVE_BTN, activeforeground=TEXT_COL,
                relief="flat", bd=0, padx=4,
                font=("Segoe UI", 9),
                command=lambda k=key: self.toggle_indicator(k),
            )
            btn.pack(side="left", padx=1, pady=4)
            self._ind_btns[key] = btn

        # Fullscreen button (right side)
        tk.Button(
            tb, text="⛶", width=3,
            bg=INACTIVE_BTN, fg=TEXT_COL,
            activebackground=ACTIVE_BTN, activeforeground=TEXT_COL,
            relief="flat", bd=0,
            font=("Segoe UI", 11),
            command=self._toggle_fullscreen,
        ).pack(side="right", padx=4, pady=4)

        # ── 두 번째 행: 포트폴리오 종목 선택 / 전체 종목 검색 ─────────────────
        tb2 = tk.Frame(self.frame, bg=BG_TOOLBAR, height=28)
        tb2.pack(side="top", fill="x")
        tb2.pack_propagate(False)

        tk.Label(
            tb2, text="포트폴리오:",
            bg=BG_TOOLBAR, fg="#9399b2",
            font=("Segoe UI", 9), padx=6,
        ).pack(side="left")

        self._portfolio_var = tk.StringVar(value="─ 선택 ─")
        self._portfolio_cb = ttk.Combobox(
            tb2, textvariable=self._portfolio_var,
            state="readonly", width=28,
            font=("Segoe UI", 9),
        )
        self._portfolio_cb.pack(side="left", padx=4, pady=3)
        self._portfolio_cb.bind("<<ComboboxSelected>>", self._on_portfolio_select)

        tk.Frame(tb2, bg=AXIS_COL, width=1).pack(side="left", fill="y", pady=3)

        tk.Button(
            tb2, text="🔍 종목검색", padx=8,
            bg="#313244", fg=TEXT_COL,
            activebackground=ACTIVE_BTN, activeforeground=TEXT_COL,
            relief="flat", bd=0,
            font=("Segoe UI", 9),
            command=self._on_search_click,
        ).pack(side="left", padx=6, pady=3)

        # PIL 상태 표시
        pil_text = "⚡ PIL가속 ON" if _PIL_OK else "PIL없음(pip install pillow)"
        pil_col  = "#a6e3a1" if _PIL_OK else "#f38ba8"
        tk.Label(
            tb2, text=pil_text,
            bg=BG_TOOLBAR, fg=pil_col,
            font=("Segoe UI", 8), padx=8,
        ).pack(side="right")

    # ──────────────────────────────────────────────────────────────────────────
    # Canvas layout
    # ──────────────────────────────────────────────────────────────────────────

    def _build_canvases(self) -> None:
        self._chart_area = tk.Frame(self.frame, bg=BG)
        self._chart_area.pack(side="top", fill="both", expand=True)

        # Price canvas
        self._price_canvas = tk.Canvas(
            self._chart_area, bg=BG, highlightthickness=0, cursor="crosshair",
        )
        self._price_canvas.pack(side="top", fill="both", expand=True)

        # Volume canvas
        self._vol_frame = tk.Frame(self._chart_area, bg=BG)
        self._vol_frame.pack(side="top", fill="x")
        self._vol_canvas = tk.Canvas(
            self._vol_frame, bg=BG, highlightthickness=0,
            height=80, cursor="crosshair",
        )
        self._vol_canvas.pack(fill="both", expand=True)

        # RSI canvas
        self._rsi_frame = tk.Frame(self._chart_area, bg=BG)
        self._rsi_frame.pack(side="top", fill="x")
        self._rsi_canvas = tk.Canvas(
            self._rsi_frame, bg=BG, highlightthickness=0,
            height=70, cursor="crosshair",
        )
        self._rsi_canvas.pack(fill="both", expand=True)

        # MACD canvas (hidden by default)
        self._macd_frame = tk.Frame(self._chart_area, bg=BG)
        self._macd_canvas = tk.Canvas(
            self._macd_frame, bg=BG, highlightthickness=0,
            height=70, cursor="crosshair",
        )
        self._macd_canvas.pack(fill="both", expand=True)
        # start hidden
        if not self._show.get("MACD", False):
            self._macd_frame.pack_forget()

        if not self._show.get("RSI", True):
            self._rsi_frame.pack_forget()

        if not self._show.get("Volume", True):
            self._vol_frame.pack_forget()

        # Tooltip overlay (drawn directly on price canvas)
        self._tooltip_items: list[int] = []

    # ──────────────────────────────────────────────────────────────────────────
    # Event bindings
    # ──────────────────────────────────────────────────────────────────────────

    def _bind_events(self) -> None:
        for canvas in (self._price_canvas, self._vol_canvas,
                       self._rsi_canvas, self._macd_canvas):
            canvas.bind("<Motion>",           self.on_motion)
            canvas.bind("<B1-Motion>",         self.on_drag)
            canvas.bind("<ButtonPress-1>",     self._on_press)
            canvas.bind("<ButtonRelease-1>",   self._on_release)
            canvas.bind("<MouseWheel>",        self.on_mousewheel)
            canvas.bind("<Double-Button-1>",   self._reset_view)
            canvas.bind("<Configure>",         self._on_configure)
            canvas.bind("<Leave>",             self._on_leave)

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def load_symbol(self, symbol: str, df: pd.DataFrame) -> None:
        """데이터 로드 후 차트 재그리기."""
        if df is None or df.empty:
            return
        self._symbol = symbol
        self._data = df.copy()
        self._sym_lbl.configure(text=f"  {symbol}  ")
        self._calc_indicators()
        # default view: last 252 candles (1Y)
        self.set_period("1Y")

    def set_period(self, period_str: str) -> None:
        """기간 버튼으로 view_count / view_start 설정."""
        period_map = {
            "1M": 21, "3M": 63, "6M": 126,
            "1Y": 252, "3Y": 756,
        }
        if self._data is None:
            return
        n = len(self._data)
        if period_str == "ALL":
            count = n
        else:
            count = period_map.get(period_str, 252)
        self._view_count = min(count, n)
        self._view_start = max(0, n - self._view_count)
        self._active_period = period_str
        self._highlight_period_btn(period_str)
        self._redraw_all()

    def toggle_indicator(self, name: str) -> None:
        """인디케이터 표시/숨김 전환."""
        self._show[name] = not self._show.get(name, False)
        on = self._show[name]

        # Update button colour
        if name in self._ind_btns:
            self._ind_btns[name].configure(bg=ACTIVE_BTN if on else INACTIVE_BTN)

        # MA20 also controls MA toggle button state
        if name == "MA20":
            self._show["MA60"]  = on
            self._show["MA120"] = on

        # Show/hide sub-canvases
        if name == "Volume":
            if on:
                self._vol_frame.pack(side="top", fill="x",
                                     before=self._rsi_frame)
            else:
                self._vol_frame.pack_forget()

        elif name == "RSI":
            if on:
                self._rsi_frame.pack(side="top", fill="x")
            else:
                self._rsi_frame.pack_forget()

        elif name == "MACD":
            if on:
                self._macd_frame.pack(side="top", fill="x")
            else:
                self._macd_frame.pack_forget()

        self._redraw_all()

    # ──────────────────────────────────────────────────────────────────────────
    # Indicator calculation
    # ──────────────────────────────────────────────────────────────────────────

    def _calc_indicators(self) -> None:
        if self._data is None:
            return
        self._ind = _calc_indicators(self._data)

    # ──────────────────────────────────────────────────────────────────────────
    # Coordinate helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _x_from_idx(self, i: int, canvas_w: int, cw: int, pad_l: int) -> int:
        """데이터 인덱스 i → 캔버스 x 픽셀 (캔들 중심)."""
        rel = i - self._view_start
        return int(pad_l + rel * (cw + 1) + cw // 2)

    def _y_from_price(self, price: float,
                      p_min: float, p_max: float,
                      canvas_h: int,
                      pad_t: int, pad_b: int) -> int:
        usable = canvas_h - pad_t - pad_b
        if p_max == p_min:
            return pad_t + usable // 2
        ratio = (price - p_min) / (p_max - p_min)
        return int(pad_t + usable * (1.0 - ratio))

    def _candle_width(self, canvas_w: int) -> int:
        usable = canvas_w - self._PAD_LEFT - self._PAD_RIGHT
        cw = max(2, usable // max(1, self._view_count) - 1)
        return cw

    def _visible_slice(self) -> tuple[int, int]:
        if self._data is None:
            return 0, 0
        n = len(self._data)
        start = max(0, self._view_start)
        end   = min(n, self._view_start + self._view_count)
        return start, end

    def _price_range(self, start: int, end: int) -> tuple[float, float]:
        """가시 범위의 가격 min/max (BB 포함)."""
        df = self._data
        if df is None or start >= end:
            return 0.0, 1.0
        sl = df.iloc[start:end]
        lo = float(sl["Low"].min())
        hi = float(sl["High"].max())
        # include BB
        ind = getattr(self, "_ind", {})
        if self._show.get("BB") and "BB_UPPER" in ind:
            bb_sl = ind["BB_UPPER"].iloc[start:end].dropna()
            bl_sl = ind["BB_LOWER"].iloc[start:end].dropna()
            if not bb_sl.empty:
                hi = max(hi, float(bb_sl.max()))
            if not bl_sl.empty:
                lo = min(lo, float(bl_sl.min()))
        margin = (hi - lo) * 0.05
        return lo - margin, hi + margin

    # ──────────────────────────────────────────────────────────────────────────
    # Master redraw
    # ──────────────────────────────────────────────────────────────────────────

    def _redraw_all(self) -> None:
        """전체 차트 재렌더링 (줌·팬·데이터변경·리사이즈 시)."""
        if self._data is None:
            return
        self._draw_price_chart()
        if self._show.get("Volume"):
            self._draw_volume_bars()
        if self._show.get("RSI"):
            self._draw_rsi()
        if self._show.get("MACD"):
            self._draw_macd()
        # 크로스헤어는 항상 캔버스 레이어에 그림 (PIL 이미지 위에 overlay)
        if self._crosshair_x is not None:
            self._draw_crosshair(self._crosshair_x, self._crosshair_y or 0)

    # ──────────────────────────────────────────────────────────────────────────
    # PIL off-screen blit helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _pil_blit(self, canvas: tk.Canvas, img: Any, key: str) -> None:
        """PIL Image를 canvas에 단일 create_image로 블릿."""
        photo = _TkPhoto(img)
        canvas.delete("chart_img")
        canvas.create_image(0, 0, anchor="nw", image=photo, tags="chart_img")
        self._chart_photo_refs[key] = photo  # GC 방지

    def _pil_line_series(self, draw: Any, series: "pd.Series",
                         start: int, end: int,
                         cw: int, pad_l: int,
                         H: int, p_lo: float, p_hi: float,
                         pad_t: int, pad_b: int,
                         colour: tuple, width: int = 1) -> None:
        """PIL에서 MA/BB 등 선 시리즈를 꺾은선으로 그린다."""
        pts: list = []
        for i in range(start, end):
            v = series.iloc[i]
            if np.isnan(v):
                if len(pts) >= 2:
                    draw.line(pts, fill=colour, width=width)
                pts = []
                continue
            x = self._x_from_idx(i, 0, cw, pad_l)
            y = self._y_from_price(v, p_lo, p_hi, H, pad_t, pad_b)
            pts.append((x, y))
        if len(pts) >= 2:
            draw.line(pts, fill=colour, width=width)

    def _pil_sub_line(self, draw: Any, series: "pd.Series",
                      start: int, end: int,
                      cw: int, pad_l: int,
                      H: int, v_min: float, v_max: float,
                      pad_t: int, pad_b: int,
                      colour: tuple, width: int = 1) -> None:
        """PIL에서 sub-panel (RSI/MACD 등) 선 시리즈."""
        if v_max == v_min:
            return
        usable = H - pad_t - pad_b
        pts: list = []
        for i in range(start, end):
            v = series.iloc[i]
            if np.isnan(v):
                if len(pts) >= 2:
                    draw.line(pts, fill=colour, width=width)
                pts = []
                continue
            x     = self._x_from_idx(i, 0, cw, pad_l)
            ratio = (v - v_min) / (v_max - v_min)
            y     = int(pad_t + usable * (1.0 - ratio))
            pts.append((x, y))
        if len(pts) >= 2:
            draw.line(pts, fill=colour, width=width)

    # ──────────────────────────────────────────────────────────────────────────
    # Price chart drawing  (PIL → canvas fallback)
    # ──────────────────────────────────────────────────────────────────────────

    def _draw_price_chart(self) -> None:
        c = self._price_canvas
        W = c.winfo_width()
        H = c.winfo_height()
        if W < 50 or H < 50:
            return

        if _PIL_OK:
            self._draw_price_pil(c, W, H)
        else:
            self._draw_price_canvas(c, W, H)

    def _draw_price_pil(self, c: tk.Canvas, W: int, H: int) -> None:
        """PIL 오프스크린 렌더링 → PhotoImage 단일 블릿."""
        PAD_L = self._PAD_LEFT
        PAD_R = self._PAD_RIGHT
        PAD_T = self._PAD_TOP
        PAD_B = self._PAD_BOTTOM

        start, end = self._visible_slice()
        if start >= end:
            return

        cw = self._candle_width(W)
        df = self._data
        ind = getattr(self, "_ind", {})
        p_lo, p_hi = self._price_range(start, end)

        def px(price: float) -> int:
            return self._y_from_price(price, p_lo, p_hi, H, PAD_T, PAD_B)
        def cx(i: int) -> int:
            return self._x_from_idx(i, W, cw, PAD_L)

        # RGBA image
        img  = _PILImage.new("RGBA", (W, H), _hex_to_rgba(BG))
        draw = _PILDraw.Draw(img, "RGBA")

        # Grid lines (dashed)
        grid_col = _hex_to_rgb(AXIS_COL)
        steps = 5
        interval = (p_hi - p_lo) / steps if p_hi != p_lo else 1
        for k in range(steps + 1):
            price = p_lo + k * interval
            y = px(price)
            _pdash_line(draw, PAD_L, y, W - PAD_R, y, fill=grid_col, dash=(2, 4))

        # Recent 5-day highlight
        if end > start:
            x0 = cx(max(start, end - 5))
            draw.rectangle([x0, PAD_T, W - PAD_R, H - PAD_B],
                           fill=(0x28, 0x28, 0x40, 200))

        # BB band fill (semi-transparent)
        if self._show.get("BB") and "BB_UPPER" in ind:
            upper = ind["BB_UPPER"]
            lower = ind["BB_LOWER"]
            pts_top: list = []
            pts_bot: list = []
            for i in range(start, end):
                uv = upper.iloc[i];  lv = lower.iloc[i]
                if np.isnan(uv) or np.isnan(lv):
                    continue
                pts_top.append((cx(i), px(uv)))
                pts_bot.append((cx(i), px(lv)))
            if len(pts_top) >= 2:
                poly = pts_top + list(reversed(pts_bot))
                bb_fill_col = _hex_to_rgba(BB_FILL, 35)
                draw.polygon(poly, fill=bb_fill_col)

        # Candlesticks
        up_rgb   = _hex_to_rgb(CANDLE_UP)
        down_rgb = _hex_to_rgb(CANDLE_DOWN)
        for i in range(start, end):
            row = df.iloc[i]
            o   = float(row["Open"]);  h = float(row["High"])
            lo_ = float(row["Low"]);   cl = float(row["Close"])
            x   = cx(i)
            col = up_rgb if cl >= o else down_rgb
            # wick
            draw.line([(x, px(h)), (x, px(lo_))], fill=col, width=1)
            # body
            y1   = px(max(o, cl));  y2 = px(min(o, cl))
            if y2 <= y1:
                y2 = y1 + 1
            half = max(1, cw // 2)
            draw.rectangle([x - half, y1, x + half, y2], fill=col, outline=col)

        # MA lines
        ma_defs = [("MA20", MA20_COL, True), ("MA60", MA60_COL, True),
                   ("MA120", MA120_COL, False)]
        for key, col, _ in ma_defs:
            if key == "MA120":
                if not self._show.get("MA120", False):
                    continue
            elif not self._show.get(key, True):
                continue
            if key not in ind:
                continue
            self._pil_line_series(draw, ind[key], start, end, cw, PAD_L,
                                  H, p_lo, p_hi, PAD_T, PAD_B, _hex_to_rgb(col))

        # BB lines
        if self._show.get("BB"):
            for bk in ("BB_UPPER", "BB_MID", "BB_LOWER"):
                if bk in ind:
                    self._pil_line_series(draw, ind[bk], start, end, cw, PAD_L,
                                          H, p_lo, p_hi, PAD_T, PAD_B,
                                          _hex_to_rgb(BB_COL))

        # Price axis
        ax_x = W - PAD_R + 4
        draw.line([(W - PAD_R, PAD_T), (W - PAD_R, H - PAD_B)],
                  fill=_hex_to_rgb(AXIS_COL), width=1)
        text_col = _hex_to_rgb(TEXT_COL)
        for k in range(steps + 1):
            price = p_lo + k * interval
            y = px(price)
            draw.text((ax_x, y - 5), _fmt_price(price), fill=text_col, font=_PFONT8)

        # Date axis
        draw.line([(PAD_L, H - PAD_B), (W - PAD_R, H - PAD_B)],
                  fill=_hex_to_rgb(AXIS_COL), width=1)
        n_visible = end - start
        if n_visible <= 63:
            step = max(1, n_visible // 8)
        elif n_visible <= 252:
            step = max(1, n_visible // 10)
        else:
            step = max(1, n_visible // 12)
        prev_month = -1
        for i in range(start, end, step):
            ts = self._data.index[i]
            try:
                dt = pd.Timestamp(ts).date()
            except Exception:
                continue
            if dt.month != prev_month:
                label = dt.strftime("%y/%m")
                prev_month = dt.month
            else:
                label = dt.strftime("%d")
            x = cx(i)
            draw.line([(x, H - PAD_B), (x, H - PAD_B + 3)],
                      fill=_hex_to_rgb(AXIS_COL), width=1)
            draw.text((x - 10, H - PAD_B + 4), label, fill=text_col, font=_PFONT7)

        self._pil_blit(c, img.convert("RGB"), "price")

    def _draw_price_canvas(self, c: tk.Canvas, W: int, H: int) -> None:
        """캔버스 직접 드로잉 (PIL 없을 때 폴백)."""
        c.delete("all")
        PAD_L = self._PAD_LEFT
        PAD_R = self._PAD_RIGHT
        PAD_T = self._PAD_TOP
        PAD_B = self._PAD_BOTTOM

        start, end = self._visible_slice()
        if start >= end:
            return

        cw   = self._candle_width(W)
        df   = self._data
        ind  = getattr(self, "_ind", {})
        p_lo, p_hi = self._price_range(start, end)

        def px(price: float) -> int:
            return self._y_from_price(price, p_lo, p_hi, H, PAD_T, PAD_B)

        def cx(i: int) -> int:
            return self._x_from_idx(i, W, cw, PAD_L)

        # Background grid lines
        self._draw_hgrid(c, W, H, p_lo, p_hi, PAD_L, PAD_T, PAD_B)

        # Recent 5-day highlight (very subtle)
        if end > start:
            x0 = cx(max(start, end - 5))
            c.create_rectangle(x0, PAD_T, W - PAD_R, H - PAD_B,
                                fill="#282840", outline="")

        # BB band fill (stipple)
        if self._show.get("BB") and "BB_UPPER" in ind:
            self._draw_bb_fill(c, ind, start, end, cw, PAD_L, H, p_lo, p_hi, PAD_T, PAD_B)

        # Candlesticks
        for i in range(start, end):
            row  = df.iloc[i]
            o    = float(row["Open"])
            h    = float(row["High"])
            lo_  = float(row["Low"])
            cl   = float(row["Close"])
            x    = cx(i)
            up   = cl >= o
            col  = CANDLE_UP if up else CANDLE_DOWN

            # Wick
            c.create_line(x, px(h), x, px(lo_), fill=col, width=1)

            # Body
            y1 = px(max(o, cl))
            y2 = px(min(o, cl))
            if y2 <= y1:
                y2 = y1 + 1
            half = max(1, cw // 2)
            c.create_rectangle(x - half, y1, x + half, y2,
                                fill=col, outline=col)

        # MA lines
        ma_defs = [
            ("MA20",  MA20_COL,  True),
            ("MA60",  MA60_COL,  True),
            ("MA120", MA120_COL, False),
        ]
        for key, col, default_on in ma_defs:
            show_key = key if key == "MA120" else key
            if key == "MA120":
                if not self._show.get("MA120", False):
                    continue
            elif not self._show.get(key, default_on):
                continue
            if key not in ind:
                continue
            self._draw_line_series(c, ind[key], start, end, cw, PAD_L, H,
                                   p_lo, p_hi, PAD_T, PAD_B, col, width=1)

        # BB lines
        if self._show.get("BB"):
            for bk in ("BB_UPPER", "BB_MID", "BB_LOWER"):
                if bk in ind:
                    dash = (4, 2) if bk == "BB_MID" else ()
                    self._draw_line_series(c, ind[bk], start, end, cw, PAD_L, H,
                                           p_lo, p_hi, PAD_T, PAD_B, BB_COL,
                                           width=1, dash=dash)

        # Price axis
        self._draw_price_axis(c, W, H, p_lo, p_hi, PAD_T, PAD_B)
        # Date axis
        self._draw_date_axis(c, W, H, start, end, cw, PAD_L, PAD_B)

    # ──────────────────────────────────────────────────────────────────────────
    # BB fill
    # ──────────────────────────────────────────────────────────────────────────

    def _draw_bb_fill(self, c: tk.Canvas, ind: dict,
                      start: int, end: int,
                      cw: int, pad_l: int,
                      H: int, p_lo: float, p_hi: float,
                      pad_t: int, pad_b: int) -> None:
        upper = ind.get("BB_UPPER")
        lower = ind.get("BB_LOWER")
        if upper is None or lower is None:
            return

        pts_top: list[float] = []
        pts_bot: list[float] = []
        for i in range(start, end):
            uv = upper.iloc[i]
            lv = lower.iloc[i]
            if np.isnan(uv) or np.isnan(lv):
                continue
            x = self._x_from_idx(i, 0, cw, pad_l)
            pts_top.extend([x, self._y_from_price(uv, p_lo, p_hi, H, pad_t, pad_b)])
            pts_bot.extend([x, self._y_from_price(lv, p_lo, p_hi, H, pad_t, pad_b)])

        if len(pts_top) < 4:
            return

        # polygon = upper points forward + lower points backward
        poly = pts_top + list(reversed(pts_bot[::1]))
        # rebuild reversed lower correctly
        lower_pts = []
        for i in range(0, len(pts_bot), 2):
            lower_pts.append(pts_bot[i])
            lower_pts.append(pts_bot[i + 1])
        rev_lower = []
        for i in range(len(lower_pts) - 2, -1, -2):
            rev_lower.extend([lower_pts[i], lower_pts[i + 1]])
        poly_final = pts_top + rev_lower
        if len(poly_final) >= 4:
            c.create_polygon(poly_final, fill=BB_FILL,
                             stipple="gray12", outline="")

    # ──────────────────────────────────────────────────────────────────────────
    # Generic line series
    # ──────────────────────────────────────────────────────────────────────────

    def _draw_line_series(
        self, c: tk.Canvas, series: pd.Series,
        start: int, end: int,
        cw: int, pad_l: int,
        H: int, p_lo: float, p_hi: float,
        pad_t: int, pad_b: int,
        colour: str, width: int = 1,
        dash: tuple = (),
    ) -> None:
        pts: list[float] = []
        for i in range(start, end):
            v = series.iloc[i]
            if np.isnan(v):
                if len(pts) >= 4:
                    kw: dict = dict(fill=colour, width=width, smooth=True)
                    if dash:
                        kw["dash"] = dash
                    c.create_line(pts, **kw)
                pts = []
                continue
            x = self._x_from_idx(i, 0, cw, pad_l)
            y = self._y_from_price(v, p_lo, p_hi, H, pad_t, pad_b)
            pts.extend([x, y])
        if len(pts) >= 4:
            kw = dict(fill=colour, width=width, smooth=True)
            if dash:
                kw["dash"] = dash
            c.create_line(pts, **kw)

    # ──────────────────────────────────────────────────────────────────────────
    # Horizontal grid
    # ──────────────────────────────────────────────────────────────────────────

    def _draw_hgrid(self, c: tk.Canvas, W: int, H: int,
                    p_lo: float, p_hi: float,
                    pad_l: int, pad_t: int, pad_b: int) -> None:
        usable = H - pad_t - pad_b
        steps = 5
        interval = (p_hi - p_lo) / steps if p_hi != p_lo else 1
        for k in range(steps + 1):
            price = p_lo + k * interval
            y = self._y_from_price(price, p_lo, p_hi, H, pad_t, pad_b)
            c.create_line(pad_l, y, W - self._PAD_RIGHT, y,
                          fill=AXIS_COL, dash=(2, 4))

    # ──────────────────────────────────────────────────────────────────────────
    # Price axis
    # ──────────────────────────────────────────────────────────────────────────

    def _draw_price_axis(self, c: tk.Canvas, W: int, H: int,
                         p_lo: float, p_hi: float,
                         pad_t: int, pad_b: int) -> None:
        ax_x = W - self._PAD_RIGHT + 4
        steps = 5
        interval = (p_hi - p_lo) / steps if p_hi != p_lo else 1
        for k in range(steps + 1):
            price = p_lo + k * interval
            y = self._y_from_price(price, p_lo, p_hi, H, pad_t, pad_b)
            c.create_text(ax_x, y, text=_fmt_price(price),
                          fill=TEXT_COL, anchor="w",
                          font=("Consolas", 8))
        # axis line
        c.create_line(W - self._PAD_RIGHT, pad_t, W - self._PAD_RIGHT, H - pad_b,
                      fill=AXIS_COL)

    # ──────────────────────────────────────────────────────────────────────────
    # Date axis
    # ──────────────────────────────────────────────────────────────────────────

    def _draw_date_axis(self, c: tk.Canvas, W: int, H: int,
                        start: int, end: int,
                        cw: int, pad_l: int, pad_b: int) -> None:
        if self._data is None:
            return
        index = self._data.index
        n_visible = end - start
        # decide label frequency
        if n_visible <= 63:
            step = max(1, n_visible // 8)
        elif n_visible <= 252:
            step = max(1, n_visible // 10)
        else:
            step = max(1, n_visible // 12)

        y_text = H - pad_b + 4
        c.create_line(pad_l, H - pad_b, W - self._PAD_RIGHT, H - pad_b,
                      fill=AXIS_COL)
        prev_month = -1
        for i in range(start, end, step):
            ts = index[i]
            try:
                if hasattr(ts, "date"):
                    dt = ts.date()
                else:
                    dt = pd.Timestamp(ts).date()
            except Exception:
                continue
            # show month if changed
            if dt.month != prev_month:
                label = dt.strftime("%y/%m")
                prev_month = dt.month
            else:
                label = dt.strftime("%d")
            x = self._x_from_idx(i, W, cw, pad_l)
            c.create_line(x, H - pad_b, x, H - pad_b + 3, fill=AXIS_COL)
            c.create_text(x, y_text, text=label, fill=TEXT_COL,
                          font=("Consolas", 7), anchor="n")

    # ──────────────────────────────────────────────────────────────────────────
    # Volume bars  (PIL → canvas fallback)
    # ──────────────────────────────────────────────────────────────────────────

    def _draw_volume_bars(self) -> None:
        c = self._vol_canvas
        W = c.winfo_width()
        H = c.winfo_height()
        if W < 10 or H < 10 or self._data is None:
            return

        PAD_L = self._PAD_LEFT;  PAD_B = 16;  PAD_T = 4
        start, end = self._visible_slice()
        if start >= end:
            return

        df  = self._data
        ind = getattr(self, "_ind", {})
        cw  = self._candle_width(W)
        vols = df["Volume"].iloc[start:end].astype(float)
        v_max = vols.max()
        if v_max == 0:
            return
        usable_h = H - PAD_T - PAD_B

        def vy(vol: float) -> int:
            return H - PAD_B - int(vol / v_max * usable_h)

        if _PIL_OK:
            img  = _PILImage.new("RGB", (W, H), _hex_to_rgb(BG))
            draw = _PILDraw.Draw(img)
            up_c  = _hex_to_rgb(VOL_UP)
            dn_c  = _hex_to_rgb(VOL_DOWN)
            ax_c  = _hex_to_rgb(AXIS_COL)
            tc    = _hex_to_rgb(TEXT_COL)

            for i in range(start, end):
                row  = df.iloc[i]
                vol  = float(row["Volume"])
                col  = up_c if float(row["Close"]) >= float(row["Open"]) else dn_c
                x    = self._x_from_idx(i, W, cw, PAD_L)
                half = max(1, cw // 2)
                draw.rectangle([x - half, vy(vol), x + half, H - PAD_B], fill=col)

            if "VOL_MA20" in ind:
                self._pil_sub_line(draw, ind["VOL_MA20"], start, end, cw, PAD_L,
                                   H, 0, v_max, PAD_T, PAD_B, _hex_to_rgb(VOL_MA_COL))

            draw.line([(W - self._PAD_RIGHT, PAD_T), (W - self._PAD_RIGHT, H - PAD_B)], fill=ax_c)
            draw.line([(PAD_L, H - PAD_B), (W - self._PAD_RIGHT, H - PAD_B)], fill=ax_c)
            draw.text((PAD_L + 2, PAD_T + 2), "VOL", fill=tc, font=_PFONT7)
            self._pil_blit(c, img, "vol")
        else:
            c.delete("all")
            c.create_text(PAD_L + 2, PAD_T + 2, text="VOL", fill=TEXT_COL,
                          font=("Consolas", 7), anchor="nw")
            for i in range(start, end):
                row = df.iloc[i]
                vol = float(row["Volume"])
                col = VOL_UP if float(row["Close"]) >= float(row["Open"]) else VOL_DOWN
                x   = self._x_from_idx(i, W, cw, PAD_L)
                half = max(1, cw // 2)
                c.create_rectangle(x - half, vy(vol), x + half, H - PAD_B,
                                   fill=col, outline="")
            if "VOL_MA20" in ind:
                self._draw_sub_line(c, ind["VOL_MA20"], start, end, cw, PAD_L,
                                    H, 0, v_max, PAD_T, PAD_B, VOL_MA_COL)
            c.create_line(W - self._PAD_RIGHT, PAD_T, W - self._PAD_RIGHT, H - PAD_B, fill=AXIS_COL)
            c.create_line(PAD_L, H - PAD_B, W - self._PAD_RIGHT, H - PAD_B, fill=AXIS_COL)

    # ──────────────────────────────────────────────────────────────────────────
    # RSI  (PIL → canvas fallback)
    # ──────────────────────────────────────────────────────────────────────────

    def _draw_rsi(self) -> None:
        c = self._rsi_canvas
        W = c.winfo_width()
        H = c.winfo_height()
        if W < 10 or H < 10 or self._data is None:
            return

        PAD_L = self._PAD_LEFT;  PAD_B = 16;  PAD_T = 4
        start, end = self._visible_slice()
        ind = getattr(self, "_ind", {})
        cw  = self._candle_width(W)

        def ry(val: float) -> int:
            usable = H - PAD_T - PAD_B
            return int(PAD_T + usable * (1.0 - val / 100.0))

        y70 = ry(70);  y30 = ry(30)

        if _PIL_OK:
            img  = _PILImage.new("RGB", (W, H), _hex_to_rgb(BG))
            draw = _PILDraw.Draw(img)
            ax_c = _hex_to_rgb(AXIS_COL)
            tc   = _hex_to_rgb(TEXT_COL)

            draw.rectangle([PAD_L, PAD_T, W - self._PAD_RIGHT, y70],   fill=(0x2a, 0x1a, 0x1a))
            draw.rectangle([PAD_L, y30,  W - self._PAD_RIGHT, H - PAD_B], fill=(0x1a, 0x2a, 0x1a))

            for level, col_hex in ((70, RSI_OB_COL), (50, AXIS_COL), (30, RSI_OS_COL)):
                y = ry(level)
                col_r = _hex_to_rgb(col_hex)
                _pdash_line(draw, PAD_L, y, W - self._PAD_RIGHT, y, fill=col_r, dash=(3, 3))
                draw.text((W - self._PAD_RIGHT + 4, y - 5), str(level), fill=col_r, font=_PFONT7)

            if "RSI" in ind:
                self._pil_sub_line(draw, ind["RSI"], start, end, cw, PAD_L,
                                   H, 0, 100, PAD_T, PAD_B, _hex_to_rgb(RSI_COL))

            draw.text((PAD_L + 2, PAD_T + 2), "RSI(14)", fill=tc, font=_PFONT7)
            draw.line([(W - self._PAD_RIGHT, PAD_T), (W - self._PAD_RIGHT, H - PAD_B)], fill=ax_c)
            draw.line([(PAD_L, H - PAD_B), (W - self._PAD_RIGHT, H - PAD_B)], fill=ax_c)
            self._pil_blit(c, img, "rsi")
        else:
            c.delete("all")
            c.create_rectangle(PAD_L, PAD_T, W - self._PAD_RIGHT, y70, fill="#2a1a1a", outline="")
            c.create_rectangle(PAD_L, y30, W - self._PAD_RIGHT, H - PAD_B, fill="#1a2a1a", outline="")
            for level, col in ((70, RSI_OB_COL), (50, AXIS_COL), (30, RSI_OS_COL)):
                y = ry(level)
                c.create_line(PAD_L, y, W - self._PAD_RIGHT, y, fill=col, dash=(3, 3))
                c.create_text(W - self._PAD_RIGHT + 4, y, text=str(level), fill=col,
                              font=("Consolas", 7), anchor="w")
            if "RSI" in ind:
                self._draw_sub_line(c, ind["RSI"], start, end, cw, PAD_L,
                                    H, 0, 100, PAD_T, PAD_B, RSI_COL)
            c.create_text(PAD_L + 2, PAD_T + 2, text="RSI(14)", fill=TEXT_COL,
                          font=("Consolas", 7), anchor="nw")
            c.create_line(W - self._PAD_RIGHT, PAD_T, W - self._PAD_RIGHT, H - PAD_B, fill=AXIS_COL)
            c.create_line(PAD_L, H - PAD_B, W - self._PAD_RIGHT, H - PAD_B, fill=AXIS_COL)

    # ──────────────────────────────────────────────────────────────────────────
    # MACD  (PIL → canvas fallback)
    # ──────────────────────────────────────────────────────────────────────────

    def _draw_macd(self) -> None:
        c = self._macd_canvas
        W = c.winfo_width()
        H = c.winfo_height()
        if W < 10 or H < 10 or self._data is None:
            return

        PAD_L = self._PAD_LEFT;  PAD_B = 16;  PAD_T = 4
        start, end = self._visible_slice()
        ind = getattr(self, "_ind", {})
        cw  = self._candle_width(W)

        macd_s   = ind.get("MACD")
        signal_s = ind.get("MACD_SIGNAL")
        hist_s   = ind.get("MACD_HIST")
        if macd_s is None:
            return

        sl_macd   = macd_s.iloc[start:end].dropna()
        sl_signal = signal_s.iloc[start:end].dropna() if signal_s is not None else pd.Series(dtype=float)
        sl_hist   = hist_s.iloc[start:end].dropna()   if hist_s   is not None else pd.Series(dtype=float)

        combined = list(sl_macd) + list(sl_signal) + list(sl_hist)
        if not combined:
            return
        v_min = min(combined);  v_max = max(combined)
        if v_max == v_min:
            v_max = v_min + 0.001

        def my(val: float) -> int:
            usable = H - PAD_T - PAD_B
            ratio  = (val - v_min) / (v_max - v_min)
            return int(PAD_T + usable * (1.0 - ratio))

        y0 = my(0.0)

        if _PIL_OK:
            img  = _PILImage.new("RGB", (W, H), _hex_to_rgb(BG))
            draw = _PILDraw.Draw(img)
            ax_c = _hex_to_rgb(AXIS_COL)
            tc   = _hex_to_rgb(TEXT_COL)

            _pdash_line(draw, PAD_L, y0, W - self._PAD_RIGHT, y0, fill=ax_c, dash=(3, 3))

            if hist_s is not None:
                hp_c = _hex_to_rgb(HIST_POS);  hn_c = _hex_to_rgb(HIST_NEG)
                for i in range(start, end):
                    v = hist_s.iloc[i]
                    if np.isnan(v):
                        continue
                    x    = self._x_from_idx(i, W, cw, PAD_L)
                    half = max(1, cw // 2)
                    y_v  = my(v)
                    col  = hp_c if v >= 0 else hn_c
                    draw.rectangle([x - half, min(y_v, y0), x + half, max(y_v, y0)], fill=col)

            self._pil_sub_line(draw, macd_s, start, end, cw, PAD_L,
                               H, v_min, v_max, PAD_T, PAD_B, _hex_to_rgb(MACD_COL))
            if signal_s is not None:
                self._pil_sub_line(draw, signal_s, start, end, cw, PAD_L,
                                   H, v_min, v_max, PAD_T, PAD_B, _hex_to_rgb(SIGNAL_COL))

            draw.text((PAD_L + 2, PAD_T + 2), "MACD(12,26,9)", fill=tc, font=_PFONT7)
            draw.line([(W - self._PAD_RIGHT, PAD_T), (W - self._PAD_RIGHT, H - PAD_B)], fill=ax_c)
            draw.line([(PAD_L, H - PAD_B), (W - self._PAD_RIGHT, H - PAD_B)], fill=ax_c)
            self._pil_blit(c, img, "macd")
        else:
            c.delete("all")
            c.create_line(PAD_L, y0, W - self._PAD_RIGHT, y0, fill=AXIS_COL, dash=(3, 3))
            if hist_s is not None:
                for i in range(start, end):
                    v = hist_s.iloc[i]
                    if np.isnan(v):
                        continue
                    x    = self._x_from_idx(i, W, cw, PAD_L)
                    half = max(1, cw // 2)
                    y_v  = my(v)
                    col  = HIST_POS if v >= 0 else HIST_NEG
                    c.create_rectangle(x - half, min(y_v, y0), x + half, max(y_v, y0),
                                       fill=col, outline="")
            self._draw_sub_line(c, macd_s, start, end, cw, PAD_L,
                                H, v_min, v_max, PAD_T, PAD_B, MACD_COL)
            if signal_s is not None:
                self._draw_sub_line(c, signal_s, start, end, cw, PAD_L,
                                    H, v_min, v_max, PAD_T, PAD_B, SIGNAL_COL)
            c.create_text(PAD_L + 2, PAD_T + 2, text="MACD(12,26,9)", fill=TEXT_COL,
                          font=("Consolas", 7), anchor="nw")
            c.create_line(W - self._PAD_RIGHT, PAD_T, W - self._PAD_RIGHT, H - PAD_B, fill=AXIS_COL)
            c.create_line(PAD_L, H - PAD_B, W - self._PAD_RIGHT, H - PAD_B, fill=AXIS_COL)

    # ──────────────────────────────────────────────────────────────────────────
    # Generic sub-panel line (arbitrary y-scale)
    # ──────────────────────────────────────────────────────────────────────────

    def _draw_sub_line(
        self, c: tk.Canvas, series: pd.Series,
        start: int, end: int,
        cw: int, pad_l: int,
        H: int, v_min: float, v_max: float,
        pad_t: int, pad_b: int,
        colour: str,
    ) -> None:
        usable = H - pad_t - pad_b
        if v_max == v_min:
            return
        pts: list[float] = []
        for i in range(start, end):
            v = series.iloc[i]
            if np.isnan(v):
                if len(pts) >= 4:
                    c.create_line(pts, fill=colour, width=1, smooth=True)
                pts = []
                continue
            x     = self._x_from_idx(i, 0, cw, pad_l)
            ratio = (v - v_min) / (v_max - v_min)
            y     = int(pad_t + usable * (1.0 - ratio))
            pts.extend([x, y])
        if len(pts) >= 4:
            c.create_line(pts, fill=colour, width=1, smooth=True)

    # ──────────────────────────────────────────────────────────────────────────
    # Crosshair
    # ──────────────────────────────────────────────────────────────────────────

    def _draw_crosshair(self, x: int, y: int) -> None:
        """모든 캔버스에 십자선을 그리고 tooltip 박스를 표시한다."""
        canvases = [self._price_canvas]
        if self._show.get("Volume"):
            canvases.append(self._vol_canvas)
        if self._show.get("RSI"):
            canvases.append(self._rsi_canvas)
        if self._show.get("MACD"):
            canvases.append(self._macd_canvas)

        # Remove old crosshair items
        for cv in canvases:
            cv.delete("crosshair")

        # Draw vertical line on all canvases + horizontal only on price canvas
        for cv in canvases:
            W = cv.winfo_width()
            H = cv.winfo_height()
            if W < 10 or H < 10:
                continue
            cv.create_line(x, 0, x, H, fill=CROSS_COL, dash=(4, 3),
                           tags="crosshair")

        # Horizontal line only on price canvas
        pc = self._price_canvas
        PH = pc.winfo_height()
        pc.create_line(0, y, pc.winfo_width() - self._PAD_RIGHT, y,
                       fill=CROSS_COL, dash=(4, 3), tags="crosshair")

        # Tooltip
        self._draw_tooltip(x, y)

    def _draw_tooltip(self, x: int, y: int) -> None:
        c = self._price_canvas
        # Remove old tooltip
        c.delete("tooltip")

        if self._data is None:
            return

        # Find data index at x
        W  = c.winfo_width()
        cw = self._candle_width(W)
        start, end = self._visible_slice()
        idx = self._view_start + max(0, (x - self._PAD_LEFT) // (cw + 1))
        idx = max(start, min(end - 1, idx))

        row = self._data.iloc[idx]
        ind = getattr(self, "_ind", {})
        ts  = self._data.index[idx]
        try:
            date_str = pd.Timestamp(ts).strftime("%Y-%m-%d")
        except Exception:
            date_str = str(ts)[:10]

        lines: list[str] = [
            date_str,
            f"O: {_fmt_price(float(row['Open']))}",
            f"H: {_fmt_price(float(row['High']))}",
            f"L: {_fmt_price(float(row['Low']))}",
            f"C: {_fmt_price(float(row['Close']))}",
            f"V: {_fmt_vol(float(row['Volume']))}",
        ]
        if "RSI" in ind:
            v = ind["RSI"].iloc[idx]
            if not np.isnan(v):
                lines.append(f"RSI: {v:.1f}")
        if "MACD" in ind:
            v = ind["MACD"].iloc[idx]
            if not np.isnan(v):
                lines.append(f"MACD: {v:.3f}")

        # Tooltip box dimensions
        line_h = 13
        box_w  = 110
        box_h  = len(lines) * line_h + 8
        tx     = x + 10
        ty     = max(self._PAD_TOP, y - box_h // 2)
        if tx + box_w > W - self._PAD_RIGHT:
            tx = x - box_w - 4

        c.create_rectangle(tx, ty, tx + box_w, ty + box_h,
                           fill=TOOLTIP_BG, outline=AXIS_COL, tags="tooltip")
        for k, line in enumerate(lines):
            c.create_text(tx + 6, ty + 4 + k * line_h,
                          text=line, fill=TOOLTIP_FG,
                          font=("Consolas", 8), anchor="nw", tags="tooltip")

    # ──────────────────────────────────────────────────────────────────────────
    # Mouse events
    # ──────────────────────────────────────────────────────────────────────────

    def on_motion(self, event: tk.Event) -> None:
        self._crosshair_x = event.x
        self._crosshair_y = event.y
        # 크로스헤어만 갱신 — 전체 차트를 다시 그리지 않음 (매우 빠름)
        self._draw_crosshair_only(event.x, event.y)

    def on_drag(self, event: tk.Event) -> None:
        if self._drag_start is None or self._drag_view_start is None:
            return
        if self._data is None:
            return
        W  = self._price_canvas.winfo_width()
        cw = self._candle_width(W)
        dx = event.x - self._drag_start
        n  = len(self._data)
        delta = -dx // max(1, cw + 1)
        new_start = int(self._drag_view_start + delta)
        new_start = max(0, min(n - self._view_count, new_start))
        self._view_start = new_start
        self._crosshair_x = event.x
        self._crosshair_y = event.y
        self._redraw_all()

    def _on_press(self, event: tk.Event) -> None:
        self._drag_start      = event.x
        self._drag_view_start = self._view_start

    def _on_release(self, event: tk.Event) -> None:
        self._drag_start      = None
        self._drag_view_start = None

    def on_mousewheel(self, event: tk.Event) -> None:
        if self._data is None:
            return
        n = len(self._data)
        W = self._price_canvas.winfo_width()

        # delta: positive = zoom in (fewer candles), negative = zoom out
        if event.delta > 0:
            new_count = max(10, int(self._view_count * 0.85))
        else:
            new_count = min(n, int(self._view_count * 1.18))

        # Keep cursor position as pivot
        cw   = self._candle_width(W)
        cursor_idx = self._view_start + max(0, (event.x - self._PAD_LEFT) // max(1, cw + 1))
        ratio = (cursor_idx - self._view_start) / max(1, self._view_count)

        self._view_count = new_count
        new_start = int(cursor_idx - ratio * new_count)
        self._view_start = max(0, min(n - new_count, new_start))
        self._redraw_all()

    def _reset_view(self, event: tk.Event) -> None:
        self.set_period(self._active_period)

    def _on_configure(self, event: tk.Event) -> None:
        # 디바운스: 연속 resize 이벤트에서 마지막 50ms 후 한 번만 렌더링
        if hasattr(self, "_resize_job") and self._resize_job is not None:
            try:
                self.frame.after_cancel(self._resize_job)
            except Exception:
                pass
        self._resize_job = self.frame.after(50, self._redraw_all)

    def _on_leave(self, event: tk.Event) -> None:
        self._crosshair_x = None
        self._crosshair_y = None
        # 크로스헤어만 지움 (전체 재렌더링 없음)
        for cv in (self._price_canvas, self._vol_canvas,
                   self._rsi_canvas, self._macd_canvas):
            cv.delete("crosshair")
        self._price_canvas.delete("tooltip")

    # ──────────────────────────────────────────────────────────────────────────
    # Misc helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _highlight_period_btn(self, period: str) -> None:
        for p, btn in self._period_btns.items():
            btn.configure(bg=ACTIVE_BTN if p == period else INACTIVE_BTN)

    def _toggle_fullscreen(self) -> None:
        top = self.frame.winfo_toplevel()
        state = top.attributes("-fullscreen")
        top.attributes("-fullscreen", not state)

    # ──────────────────────────────────────────────────────────────────────────
    # Crosshair-only update (fast path, no full chart redraw)
    # ──────────────────────────────────────────────────────────────────────────

    def _draw_crosshair_only(self, x: int, y: int) -> None:
        """크로스헤어 + 툴팁만 갱신. PIL 이미지를 재렌더링하지 않아 매우 빠름."""
        canvases = [self._price_canvas]
        if self._show.get("Volume"):
            canvases.append(self._vol_canvas)
        if self._show.get("RSI"):
            canvases.append(self._rsi_canvas)
        if self._show.get("MACD"):
            canvases.append(self._macd_canvas)
        # 기존 크로스헤어/툴팁 삭제
        for cv in canvases:
            cv.delete("crosshair")
        self._price_canvas.delete("tooltip")
        # 새로 그리기
        self._draw_crosshair(x, y)

    # ──────────────────────────────────────────────────────────────────────────
    # Portfolio & Search toolbar callbacks
    # ──────────────────────────────────────────────────────────────────────────

    def set_symbols(self, symbols: list[str]) -> None:
        """포트폴리오 종목 목록을 드롭다운에 반영 — 반드시 '종목명 (코드)' 형태로 표시."""
        self._symbols = list(symbols)
        if self._portfolio_cb is None:
            return

        try:
            from data.korean_stocks import get_name as _get_name
        except Exception:
            _get_name = None  # type: ignore

        values: list[str] = []
        for s in symbols:
            code = s.split(".")[0]
            name = ""
            if _get_name is not None:
                try:
                    name = _get_name(s) or ""
                except Exception:
                    name = ""
            # 이름이 있으면 "삼성전자 (005930)", 없으면 "005930"
            if name and name != s and name != code:
                values.append(f"{name} ({code})")
            else:
                values.append(code)

        self._portfolio_cb["values"] = values
        if values:
            self._portfolio_var.set(values[0])

    def _on_portfolio_select(self, _event=None) -> None:
        """포트폴리오 드롭다운에서 종목 선택 → 데이터 로드."""
        if self._portfolio_var is None:
            return
        sel = self._portfolio_var.get()
        if not sel or sel == "─ 선택 ─":
            return
        # sel 형태: "005930  (005930.KS)" 또는 "005930"
        idx = self._portfolio_cb["values"].index(sel) if sel in self._portfolio_cb["values"] else -1
        if idx >= 0 and idx < len(self._symbols):
            ticker = self._symbols[idx]
        else:
            # 직접 파싱
            ticker = sel.split()[0]
        self._load_ticker_async(ticker)

    def _on_search_click(self) -> None:
        """전체 종목 검색 다이얼로그 열기."""
        try:
            from .stock_search_dialog import StockSearchDialog
        except ImportError:
            try:
                import importlib, os as _os, sys as _sys
                _gui_dir = _os.path.dirname(_os.path.abspath(__file__))
                _sys.path.insert(0, _os.path.dirname(_gui_dir))
                from gui.stock_search_dialog import StockSearchDialog
            except Exception:
                tk.messagebox.showerror("오류", "StockSearchDialog를 불러올 수 없습니다.")
                return

        def _on_add(tickers: list[str]) -> None:
            if tickers:
                self._load_ticker_async(tickers[0])

        StockSearchDialog(self.frame.winfo_toplevel(), on_add=_on_add, title="차트 종목 검색")

    def _load_ticker_async(self, ticker: str) -> None:
        """종목코드로 데이터를 비동기 로드 후 차트에 표시."""
        self._sym_lbl.configure(text=f"  {ticker}  ─ 로딩 중... ─")

        def _do() -> None:
            try:
                import os as _os
                import sys as _sys
                _here = _os.path.dirname(_os.path.abspath(__file__))
                _root = _os.path.dirname(_here)
                _sys.path.insert(0, _root)

                from data import DataLoader
                from config import load_settings

                # 설정에서 캐시 경로·기간 읽기
                try:
                    _cfg = self._settings if self._settings is not None else load_settings()
                    _cache = _os.path.join(_root, _cfg.data.cache_dir)
                    _period   = _cfg.data.period
                    _interval = _cfg.data.interval
                    _ttl      = _cfg.data.cache_ttl_hours
                except Exception:
                    _cache    = _os.path.join(_root, "outputs", "cache")
                    _period   = "5y"
                    _interval = "1d"
                    _ttl      = 23

                loader = DataLoader(_cache, _ttl)
                df = loader.load(ticker, _period, _interval)

                if df is not None and not df.empty:
                    try:
                        from data.korean_stocks import get_name
                        name = get_name(ticker)
                        disp = f"{name} ({ticker.split('.')[0]})" if name and name != ticker else ticker
                    except Exception:
                        disp = ticker
                    self.frame.after(0, lambda: self.load_symbol(disp, df))
                else:
                    self.frame.after(
                        0, lambda: self._sym_lbl.configure(text=f"  {ticker}  ─ 데이터 없음 ─")
                    )
            except Exception as e:
                self.frame.after(
                    0, lambda: self._sym_lbl.configure(text=f"  {ticker}  ─ 오류: {e} ─")
                )

        threading.Thread(target=_do, daemon=True).start()


# ──────────────────────────────────────────────────────────────────────────────
# Quick standalone test
# ──────────────────────────────────────────────────────────────────────────────

def _demo() -> None:
    """데모: 랜덤 OHLCV 데이터로 차트 패널 테스트."""
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2022-01-03", periods=n, freq="B")
    price = 50000.0
    closes = []
    for _ in range(n):
        price *= 1 + np.random.normal(0.0002, 0.015)
        closes.append(price)
    closes_arr = np.array(closes)
    df = pd.DataFrame({
        "Open":   closes_arr * (1 + np.random.uniform(-0.005, 0.005, n)),
        "High":   closes_arr * (1 + np.random.uniform(0.002, 0.015, n)),
        "Low":    closes_arr * (1 - np.random.uniform(0.002, 0.015, n)),
        "Close":  closes_arr,
        "Volume": np.random.randint(500_000, 5_000_000, n).astype(float),
    }, index=dates)

    root = tk.Tk()
    root.title("ChartPanel Demo")
    root.geometry("1280x800")
    root.configure(bg=BG)

    panel = ChartPanel(root, settings={})
    panel.frame.pack(fill="both", expand=True)
    panel.load_symbol("005930", df)

    root.mainloop()


if __name__ == "__main__":
    _demo()
