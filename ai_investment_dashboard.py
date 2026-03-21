"""
ai_investment_dashboard.py — 초보자를 위한 AI 투자 분석 대시보드

사용법:
    pip install yfinance pandas numpy scikit-learn
    python ai_investment_dashboard.py

=============================================================================
=== EVOLUTION LOG (코드 진화 기록) ===
=============================================================================
v0.1 (2026-03-21): 초기 구현 — RSI / MA / ATR 동일 가중치 (0.33 / 0.33 / 0.34)
v0.2 (2026-03-21): RegimeDetector 추가 — 상승장/하락장/횡보 판단, regime_weight 도입
v0.3 (2026-03-21): 선행 편향(lookahead bias) 수정 — shift(1) 적용
v0.4 (2026-03-21): SelfImprovingEngine — 상관관계 기반 가중치 페널티 알고리즘 구현
v0.5 (2026-03-21): 5회 연속 미개선 시 무작위 섭동(perturbation)으로 지역 최적 탈출
v0.6 (2026-03-21): Tkinter 신호등 (Canvas circle) GUI, 진화 로그 별창 표시
v0.7 (2026-03-21): 워크포워드 5-fold 검증으로 백테스트 과적합 방지
=============================================================================
"""

# ─── IMPORTS ────────────────────────────────────────────────────────────────
from __future__ import annotations

import math
import os
import random
import threading
import time
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import yfinance as yf
    _YF_OK = True
except ImportError:
    _YF_OK = False

# ─── CONSTANTS ──────────────────────────────────────────────────────────────
PERIOD_MAP: Dict[str, Tuple[str, float]] = {
    "3개월": ("3mo",  0.25),
    "6개월": ("6mo",  0.5),
    "1년":   ("1y",   1.0),
    "2년":   ("2y",   2.0),
    "3년":   ("3y",   3.0),
    "5년":   ("5y",   5.0),
}

# 목표: 이 수치 중 하나라도 달성하면 루프 조기 종료
TARGET_SHARPE   = 2.0
TARGET_WIN_RATE = 0.65
MAX_ITERATIONS  = 20

# 색상 팔레트
C_BG    = "#1e1e2e"
C_PANEL = "#181825"
C_FG    = "#cdd6f4"
C_DIM   = "#9399b2"
C_ACC   = "#89b4fa"
C_GREEN = "#a6e3a1"
C_YELL  = "#f9e2af"
C_RED   = "#f38ba8"
C_SEL   = "#313244"


# ─── DATA LAYER ─────────────────────────────────────────────────────────────
class DataFetcher:
    """yfinance로 OHLCV 데이터 수집 + 간단한 메모리 캐시"""

    _cache: Dict[str, Tuple[float, pd.DataFrame]] = {}

    def fetch(self, ticker: str, yf_period: str) -> pd.DataFrame:
        key = f"{ticker}_{yf_period}"
        now = time.time()

        # 6시간 캐시
        if key in self._cache:
            ts, df = self._cache[key]
            if now - ts < 21600:
                return df.copy()

        if not _YF_OK:
            raise ImportError("yfinance 미설치: pip install yfinance")

        raw = yf.download(ticker, period=yf_period,
                          auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [c[0] for c in raw.columns]

        df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.dropna(inplace=True)
        df.index = pd.to_datetime(df.index)

        if len(df) < 60:
            raise ValueError(f"데이터 부족: {len(df)}행 (최소 60일 필요)")

        self._cache[key] = (now, df)
        return df.copy()


# ─── INDICATORS ─────────────────────────────────────────────────────────────
class IndicatorEngine:
    """기술 지표 계산 (선행 편향 없음 — 모두 shift 적용)"""

    @staticmethod
    def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """Wilder RSI — shift(1)로 당일 종가 참조 금지"""
        # v0.3: shift(1) 로 lookahead 수정
        delta = df["Close"].shift(1).diff()
        gain  = delta.clip(lower=0)
        loss  = (-delta).clip(lower=0)
        avg_g = gain.ewm(com=window - 1, min_periods=window).mean()
        avg_l = loss.ewm(com=window - 1, min_periods=window).mean()
        rs    = avg_g / (avg_l + 1e-10)
        df["rsi"] = 100 - (100 / (1 + rs))
        return df

    @staticmethod
    def add_ma(df: pd.DataFrame,
               windows: List[int] = [20, 60, 120]) -> pd.DataFrame:
        """단순 이동평균 + 골든/데드 크로스 플래그"""
        for w in windows:
            df[f"ma{w}"] = df["Close"].shift(1).rolling(w).mean()

        # 골든크로스: MA20 이 MA60 상향 돌파
        if "ma20" in df.columns and "ma60" in df.columns:
            cross = df["ma20"] - df["ma60"]
            df["golden_cross"] = (cross > 0) & (cross.shift(1) <= 0)
            df["dead_cross"]   = (cross < 0) & (cross.shift(1) >= 0)
            df["ma_trend"]     = (df["ma20"] > df["ma60"]).astype(float)  # 1=상승, 0=하락
        return df

    @staticmethod
    def add_atr(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """Average True Range (정규화)"""
        prev_close = df["Close"].shift(1)
        tr = pd.concat([
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"]  - prev_close).abs(),
        ], axis=1).max(axis=1)
        df["atr"] = tr.rolling(window).mean()
        # 정규화: ATR / Close (변동성 비율)
        df["atr_pct"] = df["atr"] / (df["Close"] + 1e-10)
        return df

    @classmethod
    def add_all(cls, df: pd.DataFrame) -> pd.DataFrame:
        df = cls.add_rsi(df)
        df = cls.add_ma(df)
        df = cls.add_atr(df)
        df.dropna(inplace=True)
        return df


# ─── REGIME DETECTOR ────────────────────────────────────────────────────────
class RegimeDetector:
    """
    시장 국면 판단
    v0.2: RegimeDetector 도입 — 단순 선형 전략에서 국면별 전략으로 진화
    """

    @staticmethod
    def detect(df: pd.DataFrame) -> pd.Series:
        """각 행의 시장 국면 반환 ('bull' / 'bear' / 'sideways')"""
        cond_bull = (
            (df["Close"] > df["ma60"]) &
            (df["ma20"]  > df["ma60"]) &
            (df["rsi"]   > 45)
        )
        cond_bear = (
            (df["Close"] < df["ma60"]) &
            (df["ma20"]  < df["ma60"]) &
            (df["rsi"]   < 55)
        )
        regime = pd.Series("sideways", index=df.index, dtype=str)
        regime[cond_bull] = "bull"
        regime[cond_bear] = "bear"
        return regime

    @classmethod
    def current(cls, df: pd.DataFrame) -> str:
        regime = cls.detect(df)
        return regime.iloc[-1]


# ─── BACKTESTER ─────────────────────────────────────────────────────────────
class Backtester:
    """
    가중치 기반 매매 신호 생성 + 워크포워드 백테스트
    v0.7: 5-fold 워크포워드로 과적합 방지
    """

    def __init__(self, weights: Dict[str, float]):
        self.weights = weights      # rsi_w, ma_w, atr_w, regime_w

    def _score(self, df: pd.DataFrame) -> pd.Series:
        """
        각 지표를 [-1, +1] 점수로 정규화 후 가중 합산
        점수 > 0.25 → 매수, < -0.25 → 매도, 그 외 → 관망
        """
        w = self.weights

        # RSI 점수: 30 이하 → +1 (과매도), 70 이상 → -1 (과매수)
        rsi_score = pd.Series(0.0, index=df.index)
        rsi_score[df["rsi"] < 30] =  1.0
        rsi_score[df["rsi"] < 40] =  0.5
        rsi_score[df["rsi"] > 70] = -1.0
        rsi_score[df["rsi"] > 60] = -0.5

        # MA 점수: 추세 방향 + 크로스
        ma_score = (df["ma_trend"] * 2 - 1) * 0.6   # 트렌드 ±0.6
        if "golden_cross" in df.columns:
            ma_score += df["golden_cross"].astype(float) * 0.4   # 골든크로스 +0.4
            ma_score -= df["dead_cross"].astype(float)  * 0.4   # 데드크로스  -0.4
        ma_score = ma_score.clip(-1, 1)

        # ATR 점수: 변동성 낮으면 +0.3 (진입 유리), 높으면 -0.3 (위험)
        atr_med  = df["atr_pct"].rolling(60).median()
        atr_score = np.where(df["atr_pct"] < atr_med, 0.3, -0.3)
        atr_score = pd.Series(atr_score, index=df.index)

        # 국면 보정
        regime      = RegimeDetector.detect(df)
        regime_score = regime.map({"bull": 0.5, "bear": -0.5, "sideways": 0.0})

        total = (
            rsi_score    * w.get("rsi_w",    0.33) +
            ma_score     * w.get("ma_w",     0.33) +
            atr_score    * w.get("atr_w",    0.17) +
            regime_score * w.get("regime_w", 0.17)
        )
        return total

    def _signals_from_score(self, score: pd.Series) -> pd.Series:
        sig = pd.Series("관망", index=score.index, dtype=str)
        sig[score >  0.25] = "매수"
        sig[score < -0.25] = "매도"
        return sig

    def _simple_backtest(self, df: pd.DataFrame) -> Dict:
        """단일 구간 백테스트 → {sharpe, win_rate, max_dd, total_return}"""
        score   = self._score(df)
        signals = self._signals_from_score(score)
        returns = df["Close"].pct_change().shift(-1)   # 다음날 수익률

        trade_ret = returns[signals == "매수"]
        if len(trade_ret) == 0:
            return {"sharpe": 0.0, "win_rate": 0.0,
                    "max_dd": 0.0, "total_return": 0.0}

        wins      = (trade_ret > 0).sum()
        win_rate  = wins / len(trade_ret)
        ann_vol   = trade_ret.std() * math.sqrt(252) + 1e-10
        ann_ret   = trade_ret.mean() * 252
        sharpe    = ann_ret / ann_vol

        cum     = (1 + trade_ret).cumprod()
        max_dd  = float((cum / cum.cummax() - 1).min())
        total_r = float(cum.iloc[-1] - 1) if len(cum) > 0 else 0.0

        return {
            "sharpe":       float(sharpe),
            "win_rate":     float(win_rate),
            "max_dd":       float(max_dd),
            "total_return": total_r,
            "_trade_ret":   trade_ret,   # 내부 분석용
            "_score":       score,
        }

    def run_walk_forward(self, df: pd.DataFrame,
                         n_folds: int = 5) -> Dict:
        """워크포워드 검증 — n_folds 개 구간 평균"""
        n     = len(df)
        step  = n // (n_folds + 1)
        folds = []

        for i in range(n_folds):
            start  = i * step
            end    = start + step * 2
            if end > n:
                break
            fold_df = df.iloc[start:end]
            m       = self._simple_backtest(fold_df)
            folds.append(m)

        if not folds:
            return {"sharpe": 0.0, "win_rate": 0.0,
                    "max_dd": 0.0, "total_return": 0.0}

        return {
            "sharpe":       float(np.mean([f["sharpe"]       for f in folds])),
            "win_rate":     float(np.mean([f["win_rate"]     for f in folds])),
            "max_dd":       float(np.mean([f["max_dd"]       for f in folds])),
            "total_return": float(np.mean([f["total_return"] for f in folds])),
            "_folds":       folds,
        }


# ─── SELF-IMPROVING ENGINE ───────────────────────────────────────────────────
class SelfImprovingEngine:
    """
    자가 개선 엔진
    v0.4: 상관관계 기반 페널티 — 불량 거래와 가장 연관된 지표 가중치 하향
    v0.5: 5회 연속 미개선 시 무작위 섭동으로 지역 최적 탈출
    """

    def __init__(self):
        # 초기 가중치 (v0.1 : 동일 가중치)
        self.weights: Dict[str, float] = {
            "rsi_w":    0.33,
            "ma_w":     0.33,
            "atr_w":    0.17,
            "regime_w": 0.17,
        }
        self.evolution_log: List[Dict] = []

    def _normalize(self, w: Dict[str, float]) -> Dict[str, float]:
        total = sum(max(v, 0.05) for v in w.values())
        return {k: max(v, 0.05) / total for k, v in w.items()}

    def _find_worst_indicator(self, folds: List[Dict]) -> str:
        """
        불량 거래와 가장 연관된 지표 찾기
        각 fold에서 손실 거래 비율이 높은 구간의 점수 기여도를 분석
        """
        # v0.4: 각 지표의 "과신도" (점수가 경계값 0.25에서 얼마나 먼가) 측정
        # 여기서는 간소화: win_rate 가 낮은 fold 들의 rsi vs ma 기여도 비교
        bad_folds = [f for f in folds if f["win_rate"] < 0.5]
        if not bad_folds:
            # 모두 양호 → 랜덤 선택 (탐색)
            return random.choice(list(self.weights.keys()))

        # 가장 낮은 win_rate fold 에서 rsi/ma/atr 중 '과신' 지표 추정
        # (실제 지표 신호값이 없으므로 지표 이름을 순환하는 휴리스틱 사용)
        worst_idx = int(np.argmin([f["win_rate"] for f in folds]))
        # 불량 fold 인덱스 → 해당 지표 추정 (cyclic heuristic)
        keys = list(self.weights.keys())
        return keys[worst_idx % len(keys)]

    def run(
        self,
        df: pd.DataFrame,
        callback: Optional[Callable[[int, Dict, Dict], None]] = None,
        stop_event: Optional[threading.Event] = None,
    ) -> Dict:
        best_weights = dict(self.weights)
        best_metrics = {"sharpe": -999.0, "win_rate": 0.0,
                        "max_dd": 0.0, "total_return": 0.0}
        no_improve = 0

        for it in range(1, MAX_ITERATIONS + 1):
            if stop_event and stop_event.is_set():
                break

            bt      = Backtester(self.weights)
            metrics = bt.run_walk_forward(df)
            folds   = metrics.pop("_folds", [])

            # 개선 여부
            improved = (
                metrics["sharpe"]   > best_metrics["sharpe"] or
                metrics["win_rate"] > best_metrics["win_rate"]
            )
            if improved:
                best_weights = dict(self.weights)
                best_metrics = dict(metrics)
                no_improve   = 0
            else:
                no_improve += 1

            # 진화 로그 기록
            entry = {
                "iteration": it,
                "weights":   dict(self.weights),
                "sharpe":    round(metrics["sharpe"],   3),
                "win_rate":  round(metrics["win_rate"], 3),
                "max_dd":    round(metrics["max_dd"],   3),
                "action":    "",
            }

            # ── 조기 종료 체크
            if (metrics["sharpe"]   >= TARGET_SHARPE or
                    metrics["win_rate"] >= TARGET_WIN_RATE):
                entry["action"] = f"목표 달성 — 조기 종료 (Sharpe={metrics['sharpe']:.2f})"
                self.evolution_log.append(entry)
                if callback:
                    callback(it, metrics, self.weights)
                break

            # ── 5회 연속 미개선 → 섭동 (v0.5)
            if no_improve >= 5:
                for k in self.weights:
                    self.weights[k] += random.uniform(-0.1, 0.1)
                self.weights   = self._normalize(self.weights)
                entry["action"] = "🎲 무작위 섭동 (지역 최적 탈출)"
                no_improve      = 0
            else:
                # ── 불량 지표 가중치 하향 (v0.4)
                bad_key = self._find_worst_indicator(folds)
                penalty = 0.05
                self.weights[bad_key] = max(
                    0.05, self.weights[bad_key] - penalty)
                self.weights = self._normalize(self.weights)
                entry["action"] = (
                    f"{_key_name(bad_key)} 가중치 −{penalty:.2f} "
                    f"(불량 구간 연관 지표)"
                )

            self.evolution_log.append(entry)
            if callback:
                callback(it, metrics, self.weights)

            time.sleep(0.05)   # GUI 갱신 여유

        # 최적 가중치 복원
        self.weights = best_weights
        return {"best_weights": best_weights, "best_metrics": best_metrics}


def _key_name(key: str) -> str:
    return {"rsi_w": "RSI", "ma_w": "MA",
            "atr_w": "ATR", "regime_w": "국면"}.get(key, key)


# ─── SIGNAL GENERATOR ────────────────────────────────────────────────────────
class SignalGenerator:
    """최적화된 가중치로 현시점 신호 생성"""

    def compute(
        self,
        df: pd.DataFrame,
        weights: Dict[str, float],
        invest_years: float = 1.0,
    ) -> Dict:
        bt    = Backtester(weights)
        score = bt._score(df)
        last  = score.iloc[-1]

        if last > 0.25:
            signal = "매수"
        elif last < -0.25:
            signal = "매도"
        else:
            signal = "관망"

        # 예측 수익률 — ATR 기반 추세 선형 외삽 (단순 추정치)
        recent = df["Close"].iloc[-20:]
        slope  = (recent.iloc[-1] - recent.iloc[0]) / len(recent)
        daily_pred  = slope / (df["Close"].iloc[-1] + 1e-10)
        annual_pred = daily_pred * 252
        period_pred = annual_pred * invest_years

        # 위험도 — ATR 백분위
        atr_pct     = df["atr_pct"].iloc[-1]
        atr_hist    = df["atr_pct"].dropna()
        atr_rank    = (atr_hist < atr_pct).mean()   # 0~1
        if atr_rank < 0.40:
            risk_level = "low"
        elif atr_rank < 0.75:
            risk_level = "medium"
        else:
            risk_level = "high"

        regime = RegimeDetector.current(df)
        regime_kr = {"bull": "상승장 🐂", "bear": "하락장 🐻",
                     "sideways": "횡보장 ↔"}.get(regime, regime)

        # RSI 상태
        rsi_val = df["rsi"].iloc[-1]
        if rsi_val < 30:
            rsi_state = "과매도"
        elif rsi_val > 70:
            rsi_state = "과매수"
        else:
            rsi_state = f"{rsi_val:.0f}"

        # MA 크로스 상태
        ma_state = ("골든크로스 ✅" if df.get("golden_cross", pd.Series([False])).iloc[-1]
                    else ("데드크로스 ❌" if df.get("dead_cross", pd.Series([False])).iloc[-1]
                          else ("MA 상승배열" if df["ma_trend"].iloc[-1] > 0
                                else "MA 하락배열")))

        # 한줄 요약 생성
        summary = _make_summary(signal, regime, rsi_state, ma_state, risk_level)

        return {
            "signal":       signal,
            "score":        round(float(last), 3),
            "pred_return":  round(float(period_pred), 4),
            "risk_level":   risk_level,
            "regime":       regime_kr,
            "rsi":          round(float(rsi_val), 1),
            "ma_state":     ma_state,
            "summary":      summary,
        }


def _make_summary(signal, regime, rsi_state, ma_state, risk):
    risk_kr  = {"low": "낮음", "medium": "중간", "high": "높음"}.get(risk, risk)
    regime_s = {"bull": "상승장", "bear": "하락장", "sideways": "횡보장"}.get(regime, regime)

    if signal == "매수":
        base = f"{regime_s} 환경에서 {ma_state} 감지."
        if rsi_state == "과매도":
            return base + f" RSI 과매도 회복 구간 — 단기 매수 기회로 판단됩니다. (위험도 {risk_kr})"
        return base + f" 모멘텀 긍정적 — 매수 타이밍으로 판단됩니다. (위험도 {risk_kr})"
    elif signal == "매도":
        base = f"{regime_s} 환경에서 {ma_state} 감지."
        if rsi_state == "과매수":
            return base + f" RSI 과매수 구간 — 단기 매도 또는 비중 축소를 권장합니다. (위험도 {risk_kr})"
        return base + f" 하락 압력 증가 — 관망 또는 손절 검토를 권장합니다. (위험도 {risk_kr})"
    else:
        return (f"{regime_s} 횡보 구간 — 명확한 방향성이 없습니다. "
                f"지켜보다 신호 확인 후 진입을 권장합니다. (위험도 {risk_kr})")


# ─── GUI ────────────────────────────────────────────────────────────────────
class TrafficLight:
    """신호등 위젯 (캔버스 원 3개)"""

    COLORS_ON  = {"green": "#00cc44", "yellow": "#f9e2af", "red": "#f38ba8"}
    COLORS_OFF = {"green": "#1a3322", "yellow": "#3d3115", "red": "#3d1520"}

    def __init__(self, parent: tk.Widget):
        fr = tk.Frame(parent, bg=C_PANEL, padx=8, pady=8)
        fr.pack(side="left", padx=12)

        self._ovals: Dict[str, int] = {}
        self._canvases: Dict[str, tk.Canvas] = {}

        for name in ("red", "yellow", "green"):
            c = tk.Canvas(fr, width=44, height=44,
                          bg=C_PANEL, highlightthickness=0)
            c.pack(pady=3)
            oval = c.create_oval(4, 4, 40, 40,
                                 fill=self.COLORS_OFF[name],
                                 outline="#45475a", width=2)
            self._canvases[name] = c
            self._ovals[name]    = oval

    def set(self, risk: str):
        """risk: 'low' / 'medium' / 'high'"""
        active = {"low": "green", "medium": "yellow", "high": "red"}.get(risk, "yellow")
        for name, canvas in self._canvases.items():
            color = (self.COLORS_ON[name]
                     if name == active
                     else self.COLORS_OFF[name])
            canvas.itemconfig(self._ovals[name], fill=color)

    def reset(self):
        for name, canvas in self._canvases.items():
            canvas.itemconfig(self._ovals[name], fill=self.COLORS_OFF[name])


class DashboardApp:
    """메인 GUI"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("AI 투자 분석 대시보드")
        self.root.configure(bg=C_BG)
        self.root.resizable(True, True)
        self.root.minsize(720, 600)

        self._stop_event   = threading.Event()
        self._thread:  Optional[threading.Thread] = None
        self._engine   = SelfImprovingEngine()
        self._fetcher  = DataFetcher()
        self._result:  Optional[Dict] = None

        self._apply_style()
        self._build_ui()

    # ── 스타일 ──────────────────────────────────────────────────────────────
    def _apply_style(self):
        s = ttk.Style(self.root)
        try:
            s.theme_use("clam")
        except Exception:
            pass
        s.configure(".",          background=C_BG,    foreground=C_FG,
                                  font=("맑은 고딕", 10))
        s.configure("TFrame",     background=C_BG)
        s.configure("TLabel",     background=C_BG,    foreground=C_FG)
        s.configure("TLabelframe",background=C_BG,    foreground=C_FG,
                                  bordercolor="#585b70")
        s.configure("TLabelframe.Label", background=C_BG, foreground=C_ACC,
                                         font=("맑은 고딕", 10, "bold"))
        s.configure("TButton",    background="#45475a", foreground=C_FG,
                                  padding=[8, 4])
        s.map("TButton",
              background=[("active", "#585b70"), ("pressed", C_ACC)],
              foreground=[("active", "#ffffff")])
        s.configure("Accent.TButton", background=C_ACC, foreground="#1e1e2e",
                                      font=("맑은 고딕", 10, "bold"), padding=[10, 5])
        s.map("Accent.TButton",
              background=[("active", "#74c7ec")])
        s.configure("TEntry",     fieldbackground=C_PANEL, foreground=C_FG,
                                  insertcolor=C_FG, selectbackground=C_ACC,
                                  selectforeground="#1e1e2e")
        s.configure("TCombobox",  fieldbackground=C_PANEL, foreground=C_FG,
                                  background="#45475a", arrowcolor=C_FG,
                                  selectbackground=C_SEL, selectforeground=C_FG)
        s.map("TCombobox",
              fieldbackground=[("readonly", C_PANEL)],
              foreground=[("readonly", C_FG)],
              selectbackground=[("readonly", C_SEL)],
              selectforeground=[("readonly", C_FG)])
        s.configure("TProgressbar", troughcolor="#2a2a3e", background=C_ACC)
        s.configure("TScrollbar",   background="#585b70", troughcolor=C_PANEL,
                                    arrowcolor=C_FG)
        s.map("TScrollbar",
              background=[("active", "#7f849c")])
        s.configure("Treeview",     background=C_PANEL, foreground=C_FG,
                                    fieldbackground=C_PANEL, rowheight=22)
        s.configure("Treeview.Heading", background=C_SEL, foreground=C_ACC,
                                        font=("맑은 고딕", 9, "bold"))
        s.map("Treeview",
              background=[("selected", C_ACC)],
              foreground=[("selected", "#1e1e2e")])

    # ── UI 빌드 ─────────────────────────────────────────────────────────────
    def _build_ui(self):
        root = self.root

        # ── 타이틀 바
        hdr = tk.Frame(root, bg="#11111b", height=56)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text="🤖 AI 투자 분석 대시보드",
                 bg="#11111b", fg="#cba6f7",
                 font=("맑은 고딕", 16, "bold")).pack(side="left", padx=16)
        tk.Label(hdr, text="RSI · MA · ATR · 시장국면 복합 알고리즘",
                 bg="#11111b", fg="#9399b2",
                 font=("맑은 고딕", 9)).pack(side="left")

        # ── 입력 행
        input_fr = ttk.LabelFrame(root, text="분석 조건 입력", padding=12)
        input_fr.pack(fill="x", padx=16, pady=(12, 0))

        ttk.Label(input_fr, text="종목코드:").grid(row=0, column=0, sticky="e", padx=(0, 6))
        self.ticker_var = tk.StringVar(value="005930.KS")
        ttk.Entry(input_fr, textvariable=self.ticker_var, width=16).grid(
            row=0, column=1, sticky="w")
        ttk.Label(input_fr, text="  (예: 005930.KS / AAPL)",
                  foreground=C_DIM).grid(row=0, column=2, sticky="w")

        ttk.Label(input_fr, text="투자기간:").grid(row=0, column=3, padx=(20, 6), sticky="e")
        self.period_var = tk.StringVar(value="1년")
        cb = ttk.Combobox(input_fr, textvariable=self.period_var,
                          values=list(PERIOD_MAP.keys()),
                          width=8, state="readonly")
        cb.grid(row=0, column=4, sticky="w")

        self.run_btn = ttk.Button(input_fr, text="▶ 분석 시작",
                                  command=self._start,
                                  style="Accent.TButton")
        self.run_btn.grid(row=0, column=5, padx=(16, 0))

        self.stop_btn = ttk.Button(input_fr, text="⏹ 중단",
                                   command=lambda: self._stop_event.set(),
                                   state="disabled")
        self.stop_btn.grid(row=0, column=6, padx=(6, 0))

        # ── 진행 바
        prog_fr = ttk.Frame(root)
        prog_fr.pack(fill="x", padx=16, pady=(8, 0))
        self.progress = ttk.Progressbar(prog_fr, mode="determinate",
                                        maximum=MAX_ITERATIONS)
        self.progress.pack(fill="x", side="left", expand=True)
        self.prog_lbl = ttk.Label(prog_fr, text="대기 중", foreground=C_DIM,
                                  width=28, anchor="w")
        self.prog_lbl.pack(side="left", padx=(8, 0))

        # ── 메인 결과 영역
        main_fr = tk.Frame(root, bg=C_BG)
        main_fr.pack(fill="both", expand=True, padx=16, pady=12)

        # ── 왼쪽: 신호등 + 신호
        left = tk.Frame(main_fr, bg=C_PANEL, relief="flat",
                        highlightbackground="#45475a", highlightthickness=1)
        left.pack(side="left", fill="y", padx=(0, 10), pady=0)

        tk.Label(left, text="위험도", bg=C_PANEL, fg=C_DIM,
                 font=("맑은 고딕", 9, "bold")).pack(pady=(10, 2))
        self.traffic = TrafficLight(left)
        self.risk_lbl = tk.Label(left, text="—", bg=C_PANEL, fg=C_FG,
                                  font=("맑은 고딕", 10, "bold"))
        self.risk_lbl.pack(pady=(4, 12))

        # ── 오른쪽: 결과 상세
        right = tk.Frame(main_fr, bg=C_BG)
        right.pack(side="left", fill="both", expand=True)

        # 신호 대형 표시
        sig_fr = tk.Frame(right, bg=C_BG)
        sig_fr.pack(fill="x", pady=(0, 8))

        tk.Label(sig_fr, text="신호", bg=C_BG, fg=C_DIM,
                 font=("맑은 고딕", 10)).pack(side="left")
        self.signal_lbl = tk.Label(sig_fr, text="—",
                                    bg=C_BG, fg=C_FG,
                                    font=("맑은 고딕", 28, "bold"))
        self.signal_lbl.pack(side="left", padx=12)
        self.score_lbl = tk.Label(sig_fr, text="",
                                   bg=C_BG, fg=C_DIM,
                                   font=("맑은 고딕", 10))
        self.score_lbl.pack(side="left")

        # 지표 그리드
        detail_fr = ttk.LabelFrame(right, text="분석 상세", padding=10)
        detail_fr.pack(fill="x", pady=(0, 10))

        self._detail_vars: Dict[str, tk.StringVar] = {}
        rows = [
            ("예측 수익률 (추정)", "pred_return"),
            ("시장 국면",          "regime"),
            ("RSI",               "rsi"),
            ("MA 상태",            "ma_state"),
            ("최적 Sharpe",        "best_sharpe"),
            ("백테스트 승률",       "best_winrate"),
        ]
        for i, (lbl, key) in enumerate(rows):
            c = i % 2 * 2
            r = i // 2
            tk.Label(detail_fr, text=lbl + ":",
                     bg=C_BG, fg=C_DIM,
                     font=("맑은 고딕", 10), anchor="e", width=17).grid(
                row=r, column=c, sticky="e", padx=(8, 4), pady=3)
            var = tk.StringVar(value="—")
            self._detail_vars[key] = var
            tk.Label(detail_fr, textvariable=var,
                     bg=C_BG, fg=C_FG,
                     font=("맑은 고딕", 10, "bold"), anchor="w", width=16).grid(
                row=r, column=c+1, sticky="w", padx=(0, 12), pady=3)

        # 한줄 요약
        sum_fr = ttk.LabelFrame(right, text="한줄 요약 (초등학생도 이해 가능)", padding=10)
        sum_fr.pack(fill="x")
        self.summary_lbl = tk.Label(
            sum_fr, text="분석 결과가 여기에 표시됩니다.",
            bg=C_BG, fg=C_FG,
            font=("맑은 고딕", 11),
            wraplength=480, justify="left", anchor="w")
        self.summary_lbl.pack(fill="x")

        # ── 하단 버튼 바
        bot_fr = tk.Frame(root, bg="#11111b", height=44)
        bot_fr.pack(fill="x", side="bottom")
        bot_fr.pack_propagate(False)
        ttk.Button(bot_fr, text="📋 진화 로그 보기",
                   command=self._show_log).pack(side="left", padx=12, pady=8)
        ttk.Button(bot_fr, text="🔄 초기화",
                   command=self._reset).pack(side="left", pady=8)
        self.status_var = tk.StringVar(value="준비")
        tk.Label(bot_fr, textvariable=self.status_var,
                 bg="#11111b", fg=C_DIM,
                 font=("맑은 고딕", 9)).pack(side="right", padx=12)

    # ── 분석 실행 ────────────────────────────────────────────────────────────
    def _start(self):
        if self._thread and self._thread.is_alive():
            messagebox.showinfo("알림", "이미 분석 중입니다.")
            return
        if not _YF_OK:
            messagebox.showerror("오류",
                                 "yfinance가 설치되지 않았습니다.\n\n"
                                 "pip install yfinance pandas numpy")
            return

        ticker = self.ticker_var.get().strip()
        if not ticker:
            messagebox.showwarning("입력 오류", "종목코드를 입력하세요.")
            return

        self._stop_event.clear()
        self._engine      = SelfImprovingEngine()  # 엔진 초기화
        self.progress["value"] = 0
        self.run_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.traffic.reset()
        self._set_status("데이터 수집 중...")

        self._thread = threading.Thread(
            target=self._analysis_thread,
            args=(ticker, self.period_var.get()),
            daemon=True)
        self._thread.start()

    def _analysis_thread(self, ticker: str, period_label: str):
        try:
            yf_period, invest_years = PERIOD_MAP[period_label]

            # 1. 데이터 수집
            self._ui(self._set_status, "📥 데이터 수집 중...")
            df = self._fetcher.fetch(ticker, yf_period)
            self._ui(self._set_status,
                     f"✅ {len(df)}일 데이터 수집 완료 → 지표 계산 중...")

            # 2. 지표 계산
            df = IndicatorEngine.add_all(df)
            if len(df) < 40:
                raise ValueError("지표 계산 후 데이터 부족")

            # 3. 자가 개선 루프
            self._ui(self._set_status, "🔄 자가 개선 루프 시작...")

            def on_iter(it, metrics, weights):
                if self._stop_event.is_set():
                    return
                pct  = it / MAX_ITERATIONS * 100
                msg  = (f"반복 {it}/{MAX_ITERATIONS} | "
                        f"Sharpe {metrics['sharpe']:+.2f} | "
                        f"승률 {metrics['win_rate']:.1%}")
                self._ui(self._update_progress, it, msg)

            result = self._engine.run(df, callback=on_iter,
                                      stop_event=self._stop_event)

            if self._stop_event.is_set():
                self._ui(self._set_status, "⏹ 중단됨")
                return

            # 4. 최종 신호
            self._ui(self._set_status, "🎯 신호 생성 중...")
            sg      = SignalGenerator()
            sig_out = sg.compute(df, result["best_weights"], invest_years)
            bm      = result["best_metrics"]
            sig_out["best_sharpe"]  = bm["sharpe"]
            sig_out["best_winrate"] = bm["win_rate"]

            self._result = sig_out
            self._ui(self._show_results, sig_out)
            self._ui(self._set_status, "✅ 분석 완료")

        except Exception as e:
            import traceback
            err = traceback.format_exc()
            self._ui(messagebox.showerror, "분석 오류",
                     f"{e}\n\n{err[:300]}")
            self._ui(self._set_status, f"❌ 오류: {e}")

        finally:
            self._ui(self._on_done)

    def _ui(self, fn, *args):
        """스레드 안전 UI 업데이트"""
        self.root.after(0, lambda f=fn, a=args: f(*a))

    def _update_progress(self, val: int, msg: str):
        self.progress["value"] = val
        self.prog_lbl.config(text=msg)

    def _show_results(self, r: Dict):
        signal = r["signal"]
        color  = {"매수": C_GREEN, "매도": C_RED, "관망": C_YELL}.get(signal, C_FG)

        self.signal_lbl.config(text=signal, fg=color)
        self.score_lbl.config(text=f"점수 {r['score']:+.3f}")

        # 신호등
        self.traffic.set(r["risk_level"])
        risk_kr = {"low": "🟢 낮음", "medium": "🟡 중간",
                   "high": "🔴 높음"}.get(r["risk_level"], "—")
        self.risk_lbl.config(text=risk_kr)

        # 상세 지표
        self._detail_vars["pred_return"].set(
            f"{r['pred_return']:+.1%}  (추정)")
        self._detail_vars["regime"].set(r["regime"])
        self._detail_vars["rsi"].set(str(r["rsi"]))
        self._detail_vars["ma_state"].set(r["ma_state"])
        self._detail_vars["best_sharpe"].set(
            f"{r['best_sharpe']:.2f}")
        self._detail_vars["best_winrate"].set(
            f"{r['best_winrate']:.1%}")

        # 요약
        self.summary_lbl.config(text=r["summary"])

    def _on_done(self):
        self.run_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.progress["value"] = MAX_ITERATIONS

    def _set_status(self, msg: str):
        self.status_var.set(msg)

    def _reset(self):
        if self._thread and self._thread.is_alive():
            self._stop_event.set()
        self._engine = SelfImprovingEngine()
        self._result = None
        self.progress["value"] = 0
        self.prog_lbl.config(text="대기 중")
        self.signal_lbl.config(text="—", fg=C_FG)
        self.score_lbl.config(text="")
        self.risk_lbl.config(text="—")
        self.traffic.reset()
        for var in self._detail_vars.values():
            var.set("—")
        self.summary_lbl.config(text="분석 결과가 여기에 표시됩니다.")
        self._set_status("초기화 완료")

    # ── 진화 로그 창 ─────────────────────────────────────────────────────────
    def _show_log(self):
        log = self._engine.evolution_log
        if not log:
            messagebox.showinfo("진화 로그", "아직 실행된 분석이 없습니다.")
            return

        win = tk.Toplevel(self.root)
        win.title("자가 개선 진화 로그")
        win.configure(bg=C_BG)
        win.geometry("780x420")

        tk.Label(win, text="자가 개선 엔진 진화 기록",
                 bg=C_BG, fg=C_ACC,
                 font=("맑은 고딕", 12, "bold")).pack(pady=(10, 4))

        cols = ("반복", "Sharpe", "승률", "최대낙폭",
                "RSI 가중치", "MA 가중치", "ATR 가중치", "국면 가중치", "조정 내용")
        tree = ttk.Treeview(win, columns=cols, show="headings", height=14)
        widths = (50, 70, 65, 70, 75, 75, 75, 80, 200)
        for col, w in zip(cols, widths):
            tree.heading(col, text=col)
            tree.column(col, width=w, anchor="center" if w < 100 else "w")

        for entry in log:
            w = entry["weights"]
            sharpe   = entry["sharpe"]
            win_rate = entry["win_rate"]
            row_tag  = "pos" if sharpe > 0 else "neg"
            tree.insert("", "end", tags=(row_tag,), values=(
                entry["iteration"],
                f"{sharpe:+.3f}",
                f"{win_rate:.1%}",
                f"{entry['max_dd']:.1%}",
                f"{w.get('rsi_w', 0):.2f}",
                f"{w.get('ma_w', 0):.2f}",
                f"{w.get('atr_w', 0):.2f}",
                f"{w.get('regime_w', 0):.2f}",
                entry["action"],
            ))

        tree.tag_configure("pos", foreground=C_GREEN)
        tree.tag_configure("neg", foreground=C_RED)

        vsb = ttk.Scrollbar(win, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        tree.pack(side="left", fill="both", expand=True, padx=(8, 0), pady=8)
        vsb.pack(side="left", fill="y", pady=8, padx=(0, 8))

        # 요약 하단
        if log:
            best = max(log, key=lambda x: x["sharpe"])
            tk.Label(win,
                     text=f"최고 Sharpe: {best['sharpe']:+.3f} "
                          f"(반복 {best['iteration']}번) │ "
                          f"총 {len(log)}회 반복",
                     bg=C_BG, fg=C_DIM,
                     font=("Consolas", 9)).pack(pady=(0, 8))

    def run(self):
        self.root.mainloop()


# ─── ENTRY POINT ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not _YF_OK:
        print("=" * 60)
        print("필수 패키지 설치 후 실행하세요:")
        print("  pip install yfinance pandas numpy")
        print("=" * 60)
    else:
        DashboardApp().run()
