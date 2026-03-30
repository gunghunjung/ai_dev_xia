# backtest/bnf_backtester.py
# BNF 신호 백테스트 엔진 — Signal Quality Report
# ─────────────────────────────────────────────────────────────────────────────
# 기능:
#   1. BNF 신호 발생 이력 재현 (walk-forward / hold-out)
#   2. 신호 후 N일 수익률 계산 (1/3/5/10일)
#   3. MAE / MFE 측정
#   4. 승률·손익비·기대값·샤프 지표
#   5. 시장 국면별 성과 분리 (상승/하락/횡보)
#   6. False Positive 패턴 분석
#   7. 파라미터 민감도 분석
# ─────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any
import logging
import math
import numpy as np
import pandas as pd

logger = logging.getLogger("quant.bnf.backtest")

try:
    from config.bnf_config import BNFConfig, load_bnf_config
    from indicators.bnf_features import compute_bnf_features, extract_latest_row
    from strategies.bnf_signal_engine import score_bnf_buy_signal, BNFSignalResult
except ImportError as e:
    logger.error(f"BNF 백테스터 import 실패: {e}")
    raise


# ─────────────────────────────────────────────────────────────────────────────
# 결과 구조체
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BNFTradeRecord:
    """단일 신호 → 실제 결과 기록"""
    symbol:      str
    signal_date: str
    entry_price: float
    bnf_score:   float
    label:       str
    stop_price:  float
    target1:     float
    rr:          float

    # 결과 (N일 후)
    ret_1d:  float = np.nan
    ret_3d:  float = np.nan
    ret_5d:  float = np.nan
    ret_10d: float = np.nan

    # MAE (최대 불리 변동) / MFE (최대 유리 변동)
    mae_5d: float = np.nan   # 5일 내 최대 손실 (%)
    mfe_5d: float = np.nan   # 5일 내 최대 이익 (%)

    # 손절 히트 여부
    stop_hit_5d: bool = False

    # 시장 국면
    market_regime: str = "unknown"  # bull/bear/sideways


@dataclass
class BNFBacktestResult:
    """백테스트 전체 결과"""
    symbol:       str
    period_start: str
    period_end:   str
    n_signals:    int
    n_valid:      int

    # 기간별 성과
    win_rate_5d:   float = np.nan
    win_rate_10d:  float = np.nan
    avg_ret_5d:    float = np.nan
    avg_ret_10d:   float = np.nan
    median_ret_5d: float = np.nan

    # 손익비
    profit_factor: float = np.nan   # 평균 수익 / 평균 손실 절댓값

    # 리스크
    avg_mae_5d:    float = np.nan
    avg_mfe_5d:    float = np.nan
    avg_rr:        float = np.nan

    # 기대값 (EV = 승률×평균수익 - 패률×평균손실)
    expected_value_5d: float = np.nan

    # 국면별
    regime_stats: Dict[str, Any] = field(default_factory=dict)

    # False Positive 패턴
    fp_patterns: List[str] = field(default_factory=list)

    # 개별 기록
    trades: List[BNFTradeRecord] = field(default_factory=list)

    def to_summary_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "기간": f"{self.period_start} ~ {self.period_end}",
            "신호수": self.n_signals,
            "유효신호": self.n_valid,
            "승률(5일)": f"{self.win_rate_5d*100:.1f}%" if not math.isnan(self.win_rate_5d) else "-",
            "평균수익(5일)": f"{self.avg_ret_5d:.2f}%" if not math.isnan(self.avg_ret_5d) else "-",
            "기대값(5일)": f"{self.expected_value_5d:.2f}%" if not math.isnan(self.expected_value_5d) else "-",
            "평균R:R": f"{self.avg_rr:.2f}" if not math.isnan(self.avg_rr) else "-",
            "평균MAE(5일)": f"{self.avg_mae_5d:.2f}%" if not math.isnan(self.avg_mae_5d) else "-",
            "평균MFE(5일)": f"{self.avg_mfe_5d:.2f}%" if not math.isnan(self.avg_mfe_5d) else "-",
            "손익비": f"{self.profit_factor:.2f}" if not math.isnan(self.profit_factor) else "-",
        }


# ─────────────────────────────────────────────────────────────────────────────
# 시장 국면 판별
# ─────────────────────────────────────────────────────────────────────────────

def _classify_regime(
    idx: int,
    close_series: pd.Series,
    lookback: int = 20,
    bull_thresh: float = 3.0,
    bear_thresh: float = -3.0,
) -> str:
    if idx < lookback:
        return "unknown"
    start_price = close_series.iloc[idx - lookback]
    end_price   = close_series.iloc[idx]
    if start_price <= 0:
        return "unknown"
    chg = (end_price - start_price) / start_price * 100
    if chg > bull_thresh:
        return "bull"
    elif chg < bear_thresh:
        return "bear"
    return "sideways"


# ─────────────────────────────────────────────────────────────────────────────
# 단일 종목 백테스트
# ─────────────────────────────────────────────────────────────────────────────

def backtest_bnf_signals(
    df: pd.DataFrame,
    symbol: str = "",
    market_df: Optional[pd.DataFrame] = None,
    cfg: Optional[BNFConfig] = None,
    min_score: float = 50.0,
    train_ratio: float = 0.6,
) -> BNFBacktestResult:
    """
    단일 종목 시계열 백테스트.

    Parameters
    ----------
    df          : OHLCV DataFrame (DatetimeIndex)
    symbol      : 종목 코드
    market_df   : 시장 지수 OHLCV (Optional)
    cfg         : BNFConfig
    min_score   : 신호 발생 최소 점수
    train_ratio : train/hold-out 분할 비율 (0.6 = 앞 60% train, 뒤 40% test)

    Returns
    -------
    BNFBacktestResult
    """
    cfg = cfg or load_bnf_config()

    if df is None or len(df) < cfg.min_data_days + 10:
        return BNFBacktestResult(
            symbol=symbol, period_start="", period_end="",
            n_signals=0, n_valid=0,
        )

    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]
    n     = len(df)

    # hold-out 구간만 평가 (train 구간 = warm-up)
    split_idx = max(cfg.min_data_days, int(n * train_ratio))

    period_start = str(df.index[split_idx])[:10] if split_idx < n else ""
    period_end   = str(df.index[-1])[:10]

    trades: List[BNFTradeRecord] = []

    # ── 시점별 신호 생성 (look-ahead 없음: i까지만 사용) ─────────────────────
    for i in range(split_idx, n - 1):   # n-1: 내일 가격 필요
        hist_df = df.iloc[:i + 1].copy()
        mkt_hist = market_df.iloc[:i + 1].copy() if market_df is not None and len(market_df) > i else None

        try:
            sig: BNFSignalResult = score_bnf_buy_signal(
                hist_df, symbol, mkt_hist, cfg
            )
        except Exception as e:
            logger.debug(f"[{symbol}] i={i} 채점 실패: {e}")
            continue

        if not sig.valid or sig.bnf_score < min_score:
            continue

        entry_price = float(close.iloc[i])   # 신호 당일 종가 진입
        if entry_price <= 0:
            continue

        # ── N일 후 수익률 ────────────────────────────────────────────────────
        def _ret(j: int) -> float:
            idx_f = i + j
            if idx_f >= n:
                return np.nan
            return (close.iloc[idx_f] - entry_price) / entry_price * 100

        # ── MAE / MFE (5일 구간) ─────────────────────────────────────────────
        end5 = min(i + 5, n - 1)
        fut_lo  = float(low.iloc[i + 1:end5 + 1].min()) if end5 > i else entry_price
        fut_hi  = float(high.iloc[i + 1:end5 + 1].max()) if end5 > i else entry_price
        mae_5d  = (fut_lo - entry_price) / entry_price * 100
        mfe_5d  = (fut_hi - entry_price) / entry_price * 100

        stop_hit = (
            not math.isnan(sig.stop_price) and
            fut_lo <= sig.stop_price
        )

        regime = _classify_regime(i, close)

        trade = BNFTradeRecord(
            symbol      = symbol,
            signal_date = str(df.index[i])[:10],
            entry_price = entry_price,
            bnf_score   = sig.bnf_score,
            label       = sig.label,
            stop_price  = sig.stop_price if not math.isnan(sig.stop_price) else 0.0,
            target1     = sig.target1    if not math.isnan(sig.target1)    else 0.0,
            rr          = sig.reward_risk_ratio if not math.isnan(sig.reward_risk_ratio) else 0.0,
            ret_1d      = _ret(1),
            ret_3d      = _ret(3),
            ret_5d      = _ret(5),
            ret_10d     = _ret(10),
            mae_5d      = mae_5d,
            mfe_5d      = mfe_5d,
            stop_hit_5d = stop_hit,
            market_regime = regime,
        )
        trades.append(trade)

    # ── 통계 집계 ─────────────────────────────────────────────────────────────
    return _aggregate_backtest(symbol, period_start, period_end, trades)


# ─────────────────────────────────────────────────────────────────────────────
# 통계 집계
# ─────────────────────────────────────────────────────────────────────────────

def _aggregate_backtest(
    symbol: str,
    period_start: str,
    period_end: str,
    trades: List[BNFTradeRecord],
) -> BNFBacktestResult:
    n_signals = len(trades)

    res = BNFBacktestResult(
        symbol       = symbol,
        period_start = period_start,
        period_end   = period_end,
        n_signals    = n_signals,
        n_valid      = n_signals,
        trades       = trades,
    )

    if n_signals == 0:
        return res

    # ── 수익률 집계 ───────────────────────────────────────────────────────────
    rets_5  = [t.ret_5d  for t in trades if not math.isnan(t.ret_5d)]
    rets_10 = [t.ret_10d for t in trades if not math.isnan(t.ret_10d)]
    maes    = [t.mae_5d  for t in trades if not math.isnan(t.mae_5d)]
    mfes    = [t.mfe_5d  for t in trades if not math.isnan(t.mfe_5d)]
    rrs     = [t.rr      for t in trades if t.rr > 0]

    def safe_mean(lst):
        return float(np.mean(lst)) if lst else np.nan

    def safe_median(lst):
        return float(np.median(lst)) if lst else np.nan

    res.avg_ret_5d    = safe_mean(rets_5)
    res.avg_ret_10d   = safe_mean(rets_10)
    res.median_ret_5d = safe_median(rets_5)
    res.avg_mae_5d    = safe_mean(maes)
    res.avg_mfe_5d    = safe_mean(mfes)
    res.avg_rr        = safe_mean(rrs)

    if rets_5:
        wins_5 = [r for r in rets_5 if r > 0]
        loss_5 = [r for r in rets_5 if r <= 0]
        res.win_rate_5d = len(wins_5) / len(rets_5)

        avg_win  = safe_mean(wins_5)
        avg_loss = abs(safe_mean(loss_5)) if loss_5 else 0.001
        res.profit_factor = avg_win / avg_loss if avg_loss > 0 else np.nan

        # 기대값
        res.expected_value_5d = (
            res.win_rate_5d * avg_win -
            (1 - res.win_rate_5d) * abs(avg_loss)
        )

    if rets_10:
        res.win_rate_10d = len([r for r in rets_10 if r > 0]) / len(rets_10)

    # ── 국면별 분석 ───────────────────────────────────────────────────────────
    for regime in ("bull", "bear", "sideways"):
        sub = [t for t in trades if t.market_regime == regime]
        if not sub:
            continue
        sub_rets = [t.ret_5d for t in sub if not math.isnan(t.ret_5d)]
        if sub_rets:
            wins = [r for r in sub_rets if r > 0]
            res.regime_stats[regime] = {
                "count":    len(sub),
                "win_rate": len(wins) / len(sub_rets),
                "avg_ret":  float(np.mean(sub_rets)),
            }

    # ── False Positive 패턴 분석 ──────────────────────────────────────────────
    fp_trades = [t for t in trades if not math.isnan(t.ret_5d) and t.ret_5d < -3.0]
    if fp_trades:
        fp_scores = np.mean([t.bnf_score for t in fp_trades])
        fp_mkt    = [t.market_regime for t in fp_trades]
        bear_fp   = fp_mkt.count("bear") / len(fp_mkt) if fp_mkt else 0

        res.fp_patterns.append(
            f"손실 신호 {len(fp_trades)}건 "
            f"(평균점수 {fp_scores:.0f}, "
            f"하락장 비율 {bear_fp*100:.0f}%)"
        )
        if bear_fp > 0.5:
            res.fp_patterns.append("→ 하락장에서 손실 신호 집중 — 시장 필터 강화 권장")

    logger.info(
        f"[{symbol}] BNF백테스트 완료: {n_signals}신호 "
        f"승률{res.win_rate_5d*100:.1f}% 평균수익{res.avg_ret_5d:.2f}%"
        if not math.isnan(res.win_rate_5d) and not math.isnan(res.avg_ret_5d)
        else f"[{symbol}] BNF백테스트 완료: 유효신호 없음"
    )

    return res


# ─────────────────────────────────────────────────────────────────────────────
# 파라미터 민감도 분석
# ─────────────────────────────────────────────────────────────────────────────

def sensitivity_analysis(
    df: pd.DataFrame,
    symbol: str = "",
    base_cfg: Optional[BNFConfig] = None,
    param_grid: Optional[Dict[str, list]] = None,
) -> pd.DataFrame:
    """
    min_score 민감도 분석.
    각 임계값마다 백테스트 → 성과 비교 테이블.
    """
    base_cfg = base_cfg or load_bnf_config()

    if param_grid is None:
        param_grid = {"min_score": [30, 40, 50, 60, 70]}

    rows = []
    for score_thresh in param_grid.get("min_score", [50]):
        try:
            res = backtest_bnf_signals(df, symbol, cfg=base_cfg, min_score=float(score_thresh))
            row = res.to_summary_dict()
            row["min_score"] = score_thresh
            rows.append(row)
        except Exception as e:
            logger.warning(f"민감도 분석 실패 score={score_thresh}: {e}")

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).set_index("min_score")


# ─────────────────────────────────────────────────────────────────────────────
# 여러 종목 일괄 백테스트
# ─────────────────────────────────────────────────────────────────────────────

def batch_backtest(
    symbol_dfs: Dict[str, pd.DataFrame],
    market_df: Optional[pd.DataFrame] = None,
    cfg: Optional[BNFConfig] = None,
    min_score: float = 50.0,
) -> Dict[str, BNFBacktestResult]:
    """여러 종목 일괄 백테스트 → {symbol: BNFBacktestResult}"""
    cfg = cfg or load_bnf_config()
    results = {}
    for sym, df in symbol_dfs.items():
        try:
            results[sym] = backtest_bnf_signals(df, sym, market_df, cfg, min_score)
        except Exception as e:
            logger.warning(f"[{sym}] 배치 백테스트 실패: {e}")
    return results
