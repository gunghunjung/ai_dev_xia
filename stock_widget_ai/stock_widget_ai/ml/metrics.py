"""
평가 지표 — 회귀 / 분류 / 금융 실전 지표
────────────────────────────────────────
회귀    : MAE, RMSE, MAPE, SMAPE, R², DA
분류    : Accuracy, Precision, Recall, F1, AUC
금융    : Hit Ratio, Avg Return, Max Drawdown, Sharpe, Sortino, Calmar
불확실성: Calibration Error, PICP (prediction interval coverage)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


# ══════════════════════════════════════════════════════════════════════════
# 1. 회귀 지표
# ══════════════════════════════════════════════════════════════════════════
def regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    eps = 1e-9
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae  = float(np.mean(np.abs(y_true - y_pred)))
    mse  = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))

    # MAPE
    nonzero = np.abs(y_true) > eps
    mape = float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100) \
           if nonzero.any() else float("nan")

    # SMAPE
    denom = (np.abs(y_true) + np.abs(y_pred) + eps)
    smape = float(np.mean(2 * np.abs(y_true - y_pred) / denom) * 100)

    # R²
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    r2     = float(1 - ss_res / (ss_tot + eps))

    # Directional Accuracy
    da = directional_accuracy(y_true, y_pred)

    return {
        "mae": mae, "mse": mse, "rmse": rmse,
        "mape": mape, "smape": smape, "r2": r2, "da": da,
    }


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))


# ══════════════════════════════════════════════════════════════════════════
# 2. 분류 지표
# ══════════════════════════════════════════════════════════════════════════
def classification_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    y_true  = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred  = (y_score >= threshold).astype(int)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    acc  = (tp + tn) / (tp + tn + fp + fn + 1e-9)
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)

    # AUC (간단 구현 — trapezoidal)
    auc = _roc_auc(y_true, y_score)

    return {
        "accuracy": float(acc), "precision": float(prec),
        "recall":   float(rec), "f1": float(f1), "auc": float(auc),
    }


def _roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        from sklearn.metrics import roc_auc_score
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        pass
    # 수동 trapezoidal AUC
    desc_score_idx = np.argsort(-y_score)
    y_sorted = y_true[desc_score_idx]
    tp_cum = np.cumsum(y_sorted)
    fp_cum = np.cumsum(1 - y_sorted)
    tpr = tp_cum / (tp_cum[-1] + 1e-9)
    fpr = fp_cum / (fp_cum[-1] + 1e-9)
    return float(np.trapz(tpr, fpr))


# ══════════════════════════════════════════════════════════════════════════
# 3. 금융 실전 지표
# ══════════════════════════════════════════════════════════════════════════
def financial_metrics(
    strategy_returns: np.ndarray,
    benchmark_returns: Optional[np.ndarray] = None,
    risk_free_rate: float = 0.02,       # 연율화 무위험 이자율
    periods_per_year: int = 252,
) -> Dict[str, float]:
    """
    Parameters
    ----------
    strategy_returns  : 전략 일별 수익률 (소수점, e.g. 0.01 = 1%)
    benchmark_returns : 비교 기준 일별 수익률
    risk_free_rate    : 연간 무위험 이자율
    periods_per_year  : 연율화 계수 (일봉=252)
    """
    r = np.asarray(strategy_returns, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) == 0:
        return _empty_financial()

    rf_daily = risk_free_rate / periods_per_year

    # 누적 수익률
    cum_ret    = float(np.prod(1 + r) - 1)
    ann_ret    = float((1 + cum_ret) ** (periods_per_year / len(r)) - 1)
    ann_vol    = float(np.std(r, ddof=1) * np.sqrt(periods_per_year))

    # Sharpe
    excess     = r - rf_daily
    sharpe     = float(np.mean(excess) / (np.std(excess, ddof=1) + 1e-9) * np.sqrt(periods_per_year))

    # Sortino (하방 변동성만)
    downside   = r[r < rf_daily]
    sortino_vol = float(np.std(downside, ddof=1) * np.sqrt(periods_per_year)) if len(downside) > 1 else ann_vol
    sortino    = float(ann_ret / (sortino_vol + 1e-9))

    # Max Drawdown
    mdd, mdd_dur = max_drawdown(r)

    # Calmar
    calmar     = float(ann_ret / (abs(mdd) + 1e-9))

    # Hit Ratio (양수 수익률 비율)
    hit_ratio  = float(np.mean(r > 0))

    # 평균 일 수익률 (거래일)
    avg_daily  = float(np.mean(r))

    # 알파/베타 (벤치마크 있을 때)
    alpha, beta = 0.0, 1.0
    if benchmark_returns is not None:
        bm = np.asarray(benchmark_returns, dtype=float)
        n  = min(len(r), len(bm))
        if n > 10:
            alpha, beta = _alpha_beta(r[:n], bm[:n], rf_daily)

    return {
        "cumulative_return": cum_ret,
        "annualized_return": ann_ret,
        "annualized_volatility": ann_vol,
        "sharpe":     sharpe,
        "sortino":    sortino,
        "calmar":     calmar,
        "max_drawdown": mdd,
        "max_drawdown_duration": mdd_dur,
        "hit_ratio":  hit_ratio,
        "avg_daily_return": avg_daily,
        "alpha":      alpha,
        "beta":       beta,
    }


def max_drawdown(returns: np.ndarray) -> Tuple[float, int]:
    """
    Returns (max_drawdown, max_drawdown_duration_in_periods)
    """
    cum_curve = np.cumprod(1 + np.asarray(returns))
    peak      = np.maximum.accumulate(cum_curve)
    drawdown  = (cum_curve - peak) / (peak + 1e-9)
    mdd       = float(drawdown.min())

    # 최장 MDD 기간
    in_dd     = drawdown < 0
    max_dur   = 0
    cur_dur   = 0
    for v in in_dd:
        if v:
            cur_dur += 1
            max_dur = max(max_dur, cur_dur)
        else:
            cur_dur = 0

    return mdd, max_dur


def _alpha_beta(
    strategy: np.ndarray, benchmark: np.ndarray, rf_daily: float
) -> Tuple[float, float]:
    rs = strategy  - rf_daily
    rb = benchmark - rf_daily
    cov = np.cov(rs, rb, ddof=1)
    beta  = float(cov[0, 1] / (cov[1, 1] + 1e-9))
    alpha = float(np.mean(rs) - beta * np.mean(rb))
    return alpha * 252, beta   # 알파 연율화


def _empty_financial() -> Dict[str, float]:
    return {k: float("nan") for k in [
        "cumulative_return", "annualized_return", "annualized_volatility",
        "sharpe", "sortino", "calmar", "max_drawdown",
        "max_drawdown_duration", "hit_ratio", "avg_daily_return",
        "alpha", "beta",
    ]}


# ══════════════════════════════════════════════════════════════════════════
# 4. 불확실성 지표
# ══════════════════════════════════════════════════════════════════════════
def prediction_interval_coverage(
    y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> Dict[str, float]:
    """
    PICP (Prediction Interval Coverage Probability)
    MPIW (Mean Prediction Interval Width)
    """
    covered = (y_true >= lower) & (y_true <= upper)
    picp    = float(np.mean(covered))
    mpiw    = float(np.mean(upper - lower))
    return {"picp": picp, "mpiw": mpiw}


def quantile_calibration(
    y_true: np.ndarray,
    quantile_preds: Dict[float, np.ndarray],
) -> Dict[float, float]:
    """
    각 분위수에 대해 실제 커버리지를 계산.
    이상적 캘리브레이션: quantile q → coverage ≈ q
    """
    result = {}
    for q, preds in quantile_preds.items():
        coverage = float(np.mean(y_true <= preds))
        result[q] = coverage
    return result


# ══════════════════════════════════════════════════════════════════════════
# 5. 통합 요약
# ══════════════════════════════════════════════════════════════════════════
def full_metrics_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    strategy_returns: Optional[np.ndarray] = None,
    lower: Optional[np.ndarray] = None,
    upper: Optional[np.ndarray] = None,
    task: str = "regression",   # "regression" | "classification"
) -> Dict[str, float]:
    result: Dict[str, float] = {}

    if task == "regression":
        result.update(regression_metrics(y_true, y_pred))
    else:
        result.update(classification_metrics(y_true, y_pred))
        result.update(regression_metrics(y_true, y_pred))

    if strategy_returns is not None:
        result.update(financial_metrics(strategy_returns))

    if lower is not None and upper is not None:
        result.update(prediction_interval_coverage(y_true, lower, upper))

    return result
