"""
ChartWidget — matplotlib 기반 주가 + 예측 + 시그널 차트
"""
from __future__ import annotations
import platform
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("QtAgg")

# ── 한글 폰트 설정 (OS별 자동 선택) ─────────────────────────────────────
def _setup_korean_font() -> None:
    import matplotlib.font_manager as fm
    _sys = platform.system()
    if _sys == "Windows":
        candidates = ["Malgun Gothic", "맑은 고딕", "Microsoft YaHei"]
    elif _sys == "Darwin":
        candidates = ["AppleGothic", "Apple SD Gothic Neo"]
    else:
        candidates = ["NanumGothic", "NanumBarunGothic", "UnDotum", "Baekmuk Gulim"]

    available = {f.name for f in fm.fontManager.ttflist}
    for font in candidates:
        if font in available:
            matplotlib.rcParams["font.family"] = font
            break
    matplotlib.rcParams["axes.unicode_minus"] = False   # 마이너스 부호 깨짐 방지

_setup_korean_font()

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtWidgets import QWidget, QVBoxLayout
from typing import Optional, List
from ..logger_config import get_logger

log = get_logger("gui.chart")

_DARK_BG    = "#0d1117"
_DARK_FG    = "#c9d1d9"
_PRED_COL   = "#58a6ff"
_CI_COL     = "#1f6feb"
_UP_COL     = "#3fb950"
_DN_COL     = "#f85149"
_EQ_COL     = "#e3b341"
_FUTURE_COL = "#f85149"   # 빨간색 — 미래 예측선


class ChartWidget(QWidget):
    def __init__(self, parent=None, dark_mode: bool = True) -> None:
        super().__init__(parent)
        self.dark_mode = dark_mode
        self._fig = Figure(figsize=(10, 6), facecolor=_DARK_BG if dark_mode else "white")
        self._canvas = FigureCanvas(self._fig)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas)

    def plot_price_and_prediction(
        self,
        price: pd.Series,
        pred_index: pd.DatetimeIndex,
        pred: np.ndarray,
        lower: Optional[np.ndarray] = None,
        upper: Optional[np.ndarray] = None,
        buy_dates: Optional[List] = None,
        sell_dates: Optional[List] = None,
        buy_prices: Optional[List] = None,
        sell_prices: Optional[List] = None,
        overlay_sma: Optional[pd.DataFrame] = None,
        title: str = "",
    ) -> None:
        self._fig.clear()
        bg = _DARK_BG if self.dark_mode else "white"
        fg = _DARK_FG if self.dark_mode else "black"

        ax = self._fig.add_subplot(111, facecolor=bg)
        ax.tick_params(colors=fg)
        for sp in ax.spines.values():
            sp.set_edgecolor("#30363d" if self.dark_mode else "#cccccc")
        ax.xaxis.label.set_color(fg)
        ax.yaxis.label.set_color(fg)
        ax.set_facecolor(bg)
        self._fig.set_facecolor(bg)

        # 실제 가격
        ax.plot(price.index, price.values, color=fg, linewidth=1.0,
                label="Actual", alpha=0.9)

        # 이동평균 overlay
        if overlay_sma is not None:
            colors = ["#f0883e", "#58a6ff", "#bc8cff"]
            for j, col in enumerate(overlay_sma.columns):
                ax.plot(overlay_sma.index, overlay_sma[col],
                        color=colors[j % len(colors)], linewidth=0.8,
                        alpha=0.7, label=col)

        # 예측선
        if len(pred) > 0:
            ax.plot(pred_index, pred, color=_PRED_COL, linewidth=1.5,
                    linestyle="--", label="Prediction")
            if lower is not None and upper is not None:
                ax.fill_between(pred_index, lower, upper,
                                alpha=0.2, color=_CI_COL, label="95% CI")

        # 매수/매도 마커
        if buy_dates:
            ax.scatter(buy_dates, buy_prices, marker="^", color=_UP_COL,
                       s=60, zorder=5, label="Buy")
        if sell_dates:
            ax.scatter(sell_dates, sell_prices, marker="v", color=_DN_COL,
                       s=60, zorder=5, label="Sell")

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        self._fig.autofmt_xdate()
        ax.legend(facecolor="#161b22" if self.dark_mode else "white",
                  labelcolor=fg, fontsize=8, loc="upper left")
        if title:
            ax.set_title(title, color=fg, fontsize=10)
        self._fig.tight_layout()
        self._canvas.draw_idle()   # draw_idle: Qt 이벤트 루프에 위임 → recursive repaint 방지

    def plot_prediction_result(
        self,
        price: pd.Series,
        pred_result: dict,
        target_type: str = "return",
        title: str = "",
        future: Optional[dict] = None,
    ) -> None:
        """
        target_type에 따라 차트 구성을 달리함.

        - "return" / "direction" : 2-subplot
            • 상단: 실제 종가 (close price)
            • 하단: 실제 수익률 vs 예측 수익률 + 95% CI
        - "close" : 1-subplot (기존 방식과 동일, 가격 단위끼리 비교)
        """
        self._fig.clear()
        bg = _DARK_BG if self.dark_mode else "white"
        fg = _DARK_FG if self.dark_mode else "black"
        grid_c = "#30363d" if self.dark_mode else "#cccccc"

        idx    = pred_result["dates"]
        actual = pred_result["actual"]   # returns 또는 close
        pred   = pred_result["pred"]
        lower  = pred_result.get("lower")
        upper  = pred_result.get("upper")

        def _style_ax(a):
            a.set_facecolor(bg)
            a.tick_params(colors=fg, labelsize=8)
            for sp in a.spines.values():
                sp.set_edgecolor(grid_c)
            a.xaxis.label.set_color(fg)
            a.yaxis.label.set_color(fg)
            a.grid(True, color=grid_c, linewidth=0.4, alpha=0.5)

        if target_type == "close":
            # ── 단일 서브플롯: 가격 단위 비교 ──────────────────────
            ax = self._fig.add_subplot(111, facecolor=bg)
            _style_ax(ax)
            ax.plot(price.index, price.values,
                    color=fg, linewidth=1.0, label="Close", alpha=0.9)
            ax.plot(idx, pred, color=_PRED_COL, linewidth=1.5,
                    linestyle="--", label="Predicted Close")
            if lower is not None and upper is not None:
                ax.fill_between(idx, lower, upper,
                                alpha=0.2, color=_CI_COL, label="95% CI")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            ax.legend(facecolor="#161b22" if self.dark_mode else "white",
                      labelcolor=fg, fontsize=8, loc="upper left")
            if title:
                ax.set_title(title, color=fg, fontsize=10)
            # 미래 예측 오버레이 (close 모드 — 단일 축)
            if future:
                self._draw_future_forecast(ax, None, future, fg)
        else:
            # ── 2-subplot: return / direction ─────────────────────
            # gridspec: 상단 60% / 하단 40%, 간격 좁게
            gs = self._fig.add_gridspec(2, 1, hspace=0.08, height_ratios=[3, 2])
            ax_top = self._fig.add_subplot(gs[0], facecolor=bg)
            ax_bot = self._fig.add_subplot(gs[1], facecolor=bg, sharex=ax_top)
            _style_ax(ax_top)
            _style_ax(ax_bot)

            # 상단: 종가 — 예측 인덱스 범위만 표시하면 화면이 너무 좁아지므로
            #        전체 price를 흐릿하게 + 예측 구간을 굵게
            ax_top.plot(price.index, price.values,
                        color=fg, linewidth=0.9, alpha=0.55, label="Close (전체)")
            if len(idx) > 0:
                # 예측 기간에 해당하는 price 구간 강조
                mask = price.index.isin(idx)
                ax_top.plot(price.index[mask], price.values[mask],
                            color=_EQ_COL, linewidth=1.4, alpha=0.9,
                            label="Close (예측기간)")
            ax_top.set_ylabel("종가", color=fg, fontsize=8)
            ax_top.tick_params(labelbottom=False)   # x축 레이블 숨김 (공유)
            ax_top.legend(facecolor="#161b22" if self.dark_mode else "white",
                          labelcolor=fg, fontsize=7, loc="upper left")
            if title:
                ax_top.set_title(title, color=fg, fontsize=10, pad=4)

            # 하단: 실제 수익률(회색) vs 예측 수익률(파란 점선) + CI
            ax_bot.plot(idx, actual, color=fg, linewidth=1.0,
                        alpha=0.7, label="Actual return")
            ax_bot.plot(idx, pred, color=_PRED_COL, linewidth=1.5,
                        linestyle="--", label="Predicted return")
            if lower is not None and upper is not None:
                ax_bot.fill_between(idx, lower, upper,
                                    alpha=0.25, color=_CI_COL, label="95% CI")
            ax_bot.axhline(0, color=grid_c, linewidth=0.6, linestyle=":")
            ax_bot.set_ylabel("수익률", color=fg, fontsize=8)
            ax_bot.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            ax_bot.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            ax_bot.legend(facecolor="#161b22" if self.dark_mode else "white",
                          labelcolor=fg, fontsize=7, loc="upper left")
            # 미래 예측 오버레이 (2-subplot 모드)
            if future:
                self._draw_future_forecast(ax_top, ax_bot, future, fg)

        self._fig.set_facecolor(bg)
        self._fig.autofmt_xdate(rotation=30, ha="right")
        if target_type == "close":
            self._fig.tight_layout()
        else:
            # tight_layout()은 sharex 공유 축과 충돌 → subplots_adjust로 수동 여백 지정
            self._fig.subplots_adjust(
                left=0.08, right=0.97, top=0.93, bottom=0.12, hspace=0.05
            )
        self._canvas.draw_idle()

    def _draw_future_forecast(self, ax_price, ax_return, future: dict, fg: str) -> None:
        """
        미래 1스텝 예측을 차트에 오버레이
        - ax_price : 상단 가격 축 (항상 그림)
        - ax_return: 하단 수익률 축 (return/direction 모드일 때만 non-None)
        """
        lc    = future["last_close"]
        pp    = future["pred_price"]
        lo    = future["lower_price"]
        hi    = future["upper_price"]
        t0    = future["last_date"]
        t1    = future["target_date"]
        ret   = future["pred_return"]
        hor   = future["horizon"]

        # ── 상단: 가격 차트 ──────────────────────────────────────
        # 현재가 → 예측가를 빨간 점선으로 연결
        ax_price.plot([t0, t1], [lc, pp],
                      color=_FUTURE_COL, linewidth=2.0,
                      linestyle="--", alpha=0.9,
                      label=f"미래 예측 (+{hor}일)")
        # 예측 가격 마커
        ax_price.scatter([t1], [pp],
                         color=_FUTURE_COL, s=80, zorder=6)
        # 95% CI 채움
        ax_price.fill_between([t0, t1],
                               [lc, lo], [lc, hi],
                               alpha=0.15, color=_FUTURE_COL)
        # 예측 가격 라벨 (소숫점 없이 콤마 형식)
        ax_price.annotate(
            f"  {pp:,.0f}원\n  ({ret:+.2%})",
            xy=(t1, pp),
            xytext=(6, 0), textcoords="offset points",
            color=_FUTURE_COL, fontsize=8,
            va="center",
        )
        # 현재 기준선 (세로 점선)
        ax_price.axvline(t0, color=_FUTURE_COL, linewidth=0.8,
                         linestyle=":", alpha=0.5)
        # 범례 갱신
        ax_price.legend(
            facecolor="#161b22" if self.dark_mode else "white",
            labelcolor=fg, fontsize=7, loc="upper left"
        )

        # ── 하단: 수익률 차트 ────────────────────────────────────
        if ax_return is not None:
            ax_return.scatter([t1], [ret],
                              color=_FUTURE_COL, s=80, zorder=6,
                              label=f"미래 수익률 ({ret:+.2%})")
            ax_return.axvline(t0, color=_FUTURE_COL, linewidth=0.8,
                              linestyle=":", alpha=0.5)
            ax_return.legend(
                facecolor="#161b22" if self.dark_mode else "white",
                labelcolor=fg, fontsize=7, loc="upper left"
            )

    def plot_equity_curve(
        self,
        equity: List[float],
        dates: List[str],
        title: str = "Equity Curve",
    ) -> None:
        self._fig.clear()
        bg = _DARK_BG if self.dark_mode else "white"
        fg = _DARK_FG if self.dark_mode else "black"

        ax = self._fig.add_subplot(111, facecolor=bg)
        ax.tick_params(colors=fg)
        self._fig.set_facecolor(bg)

        xs = pd.to_datetime(dates)
        eq = list(equity)
        # equity_curve는 루프 전 초기값 1개 + 루프 n개 = n+1개일 수 있음
        # dates는 n개 → 길이 맞추기 (초기값 제거)
        if len(eq) > len(xs):
            eq = eq[len(eq) - len(xs):]
        elif len(eq) < len(xs):
            xs = xs[len(xs) - len(eq):]
        ax.plot(xs, eq, color=_EQ_COL, linewidth=1.5)
        ax.fill_between(xs, min(eq), eq, alpha=0.1, color=_EQ_COL)
        ax.set_title(title, color=fg)
        self._fig.tight_layout()
        self._canvas.draw_idle()

    def clear(self) -> None:
        self._fig.clear()
        self._canvas.draw_idle()
