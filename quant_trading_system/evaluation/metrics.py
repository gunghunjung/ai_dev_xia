# evaluation/metrics.py — 성과 평가 및 과적합 감지
from __future__ import annotations
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger("quant.eval")


class PerformanceEvaluator:
    """
    기관급 성과 평가 엔진

    평가 항목:
    1. 수익률: CAGR, 총 수익률
    2. 위험: 변동성, 최대낙폭, VaR, CVaR
    3. 위험조정: Sharpe, Sortino, Calmar
    4. 안정성: 롤링 Sharpe 변동성
    5. 과적합 감지: 훈련/테스트 성과 비율
    6. 레짐별 분석: 강세/약세장 구분 성과
    """

    def __init__(self, risk_free_rate: float = 0.035):  # 3.5% 한국 기준금리 근사 (engine.py와 통일)
        self.rf = risk_free_rate

    # ──────────────────────────────────────────────────
    # 전체 평가
    # ──────────────────────────────────────────────────

    def evaluate(
        self,
        equity: pd.Series,
        returns: pd.Series,
        benchmark: Optional[pd.Series] = None,
        train_metrics: Optional[Dict] = None,
    ) -> Dict:
        """종합 성과 평가"""
        if len(returns) < 5:
            return {"error": "데이터 부족"}

        result = {}
        result.update(self._return_metrics(equity, returns))
        result.update(self._risk_metrics(returns))
        result.update(self._risk_adj_metrics(returns))
        result.update(self._stability_metrics(returns))

        if benchmark is not None:
            result.update(self._benchmark_metrics(returns, benchmark))

        if train_metrics is not None:
            result.update(self._overfitting_metrics(result, train_metrics))

        return result

    # ──────────────────────────────────────────────────
    # 수익률 지표
    # ──────────────────────────────────────────────────

    def _return_metrics(self, equity: pd.Series,
                        returns: pd.Series) -> Dict:
        n = len(returns)
        years = n / 252

        total_ret = (equity.iloc[-1] / equity.iloc[0]) - 1 if len(equity) > 0 else 0
        cagr = (1 + total_ret) ** (1 / max(years, 0.01)) - 1

        # 월별 수익률
        monthly = returns.resample("M").apply(lambda x: (1+x).prod() - 1)

        return {
            "total_return": float(total_ret),
            "cagr": float(cagr),
            "annual_return_avg": float(returns.mean() * 252),
            "best_month": float(monthly.max()) if len(monthly) > 0 else 0,
            "worst_month": float(monthly.min()) if len(monthly) > 0 else 0,
            "positive_months_pct": float((monthly > 0).mean()) if len(monthly) > 0 else 0,
        }

    # ──────────────────────────────────────────────────
    # 위험 지표
    # ──────────────────────────────────────────────────

    def _risk_metrics(self, returns: pd.Series) -> Dict:
        ann_vol = returns.std() * np.sqrt(252)

        # 최대낙폭
        equity = (1 + returns).cumprod()
        rolling_max = equity.cummax()
        dd = (equity - rolling_max) / rolling_max
        max_dd = dd.min()
        avg_dd = dd[dd < 0].mean() if (dd < 0).any() else 0

        # 최대낙폭 지속 기간
        in_dd = dd < 0
        dd_duration = 0
        current_dd = 0
        for v in in_dd:
            if v:
                current_dd += 1
                dd_duration = max(dd_duration, current_dd)
            else:
                current_dd = 0

        # VaR / CVaR (95%)
        var95 = float(np.percentile(returns, 5))
        cvar95 = float(returns[returns <= var95].mean()) if (returns <= var95).any() else var95

        # 하방 변동성
        downside = returns[returns < self.rf / 252]
        downside_vol = downside.std() * np.sqrt(252) if len(downside) > 0 else ann_vol

        return {
            "annual_volatility": float(ann_vol),
            "max_drawdown": float(max_dd),
            "avg_drawdown": float(avg_dd),
            "max_drawdown_duration_days": int(dd_duration),
            "var_95": var95,
            "cvar_95": cvar95,
            "downside_vol": float(downside_vol),
        }

    # ──────────────────────────────────────────────────
    # 위험조정 수익률
    # ──────────────────────────────────────────────────

    def _risk_adj_metrics(self, returns: pd.Series) -> Dict:
        ann_ret = returns.mean() * 252
        ann_vol = returns.std() * np.sqrt(252)

        # Sharpe Ratio
        sharpe = (ann_ret - self.rf) / (ann_vol + 1e-10)

        # Sortino Ratio
        downside = returns[returns < self.rf / 252]
        sortino_vol = downside.std() * np.sqrt(252) if len(downside) > 0 else ann_vol
        sortino = (ann_ret - self.rf) / (sortino_vol + 1e-10)

        # 최대낙폭 기반 Calmar
        equity = (1 + returns).cumprod()
        rolling_max = equity.cummax()
        max_dd = ((equity - rolling_max) / rolling_max).min()
        calmar = ann_ret / (abs(max_dd) + 1e-10)

        # 승률 및 손익비
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        win_rate = len(wins) / max(len(returns), 1)
        profit_factor = (wins.sum() / abs(losses.sum())) if len(losses) > 0 else float("inf")
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = losses.mean() if len(losses) > 0 else 0

        return {
            "sharpe_ratio": float(sharpe),
            "sortino_ratio": float(sortino),
            "calmar_ratio": float(calmar),
            "win_rate": float(win_rate),
            "profit_factor": float(min(profit_factor, 99.0)),
            "avg_win": float(avg_win),
            "avg_loss": float(avg_loss),
            "payoff_ratio": float(abs(avg_win / avg_loss)) if avg_loss != 0 else 0,
        }

    # ──────────────────────────────────────────────────
    # 안정성 분석
    # ──────────────────────────────────────────────────

    def _stability_metrics(self, returns: pd.Series) -> Dict:
        """롤링 Sharpe 안정성"""
        window = min(60, len(returns) // 3)
        if window < 10:
            return {}

        roll_sharpe = returns.rolling(window).apply(
            lambda x: x.mean() / (x.std() + 1e-10) * np.sqrt(252)
        )
        valid = roll_sharpe.dropna()

        return {
            "rolling_sharpe_mean": float(valid.mean()) if len(valid) > 0 else 0,
            "rolling_sharpe_std": float(valid.std()) if len(valid) > 0 else 0,
            "rolling_sharpe_min": float(valid.min()) if len(valid) > 0 else 0,
            "pct_positive_sharpe": float((valid > 0).mean()) if len(valid) > 0 else 0,
        }

    # ──────────────────────────────────────────────────
    # 벤치마크 대비
    # ──────────────────────────────────────────────────

    def _benchmark_metrics(self, returns: pd.Series,
                           benchmark: pd.Series) -> Dict:
        bench = benchmark.reindex(returns.index).dropna()
        common = returns.reindex(bench.index).dropna()

        if len(common) < 10:
            return {}

        # 알파 / 베타
        cov = np.cov(common, bench)
        beta = cov[0, 1] / (cov[1, 1] + 1e-10)
        alpha = (common.mean() - beta * bench.mean()) * 252

        # 정보 비율
        excess = common - bench.reindex(common.index)
        ir = excess.mean() * 252 / (excess.std() * np.sqrt(252) + 1e-10)

        # 최대 상대 낙폭
        rel_equity = (1 + common).cumprod() / (1 + bench).cumprod()
        max_rel_dd = ((rel_equity - rel_equity.cummax()) / rel_equity.cummax()).min()

        return {
            "alpha": float(alpha),
            "beta": float(beta),
            "information_ratio": float(ir),
            "max_relative_drawdown": float(max_rel_dd),
            "correlation_to_benchmark": float(common.corr(bench)),
        }

    # ──────────────────────────────────────────────────
    # 과적합 감지
    # ──────────────────────────────────────────────────

    def _overfitting_metrics(self, test: Dict, train: Dict) -> Dict:
        """훈련 vs 테스트 성과 비율"""
        result = {}
        for k in ["sharpe_ratio", "cagr", "max_drawdown"]:
            tv = train.get(k, 0)
            tev = test.get(k, 0)
            if abs(tv) > 1e-6:
                ratio = tev / tv
                result[f"overfit_{k}_ratio"] = float(ratio)

        # 과적합 경고: 테스트 Sharpe가 훈련의 50% 미만
        train_s = train.get("sharpe_ratio", 0)
        test_s = test.get("sharpe_ratio", 0)
        if abs(train_s) > 0.1:
            decay = (train_s - test_s) / abs(train_s)
            result["oos_decay"] = float(decay)
            result["overfit_warning"] = decay > 0.5

        return result

    # ──────────────────────────────────────────────────
    # 리포트 생성
    # ──────────────────────────────────────────────────

    def format_report(self, metrics: Dict) -> str:
        """성과 지표 텍스트 리포트"""
        lines = ["=" * 50, "  성과 평가 리포트", "=" * 50]

        sections = [
            ("수익률", ["total_return", "cagr", "best_month", "worst_month"]),
            ("위험", ["annual_volatility", "max_drawdown", "var_95", "cvar_95"]),
            ("위험조정수익률", ["sharpe_ratio", "sortino_ratio", "calmar_ratio"]),
            ("거래", ["win_rate", "profit_factor", "payoff_ratio"]),
            ("벤치마크", ["alpha", "beta", "information_ratio"]),
            ("안정성", ["rolling_sharpe_mean", "rolling_sharpe_std"]),
        ]

        pct_keys = {"total_return", "cagr", "annual_volatility", "max_drawdown",
                    "var_95", "cvar_95", "best_month", "worst_month",
                    "win_rate", "alpha"}

        for section, keys in sections:
            section_lines = []
            for k in keys:
                if k in metrics:
                    v = metrics[k]
                    if k in pct_keys:
                        vstr = f"{v:.2%}"
                    else:
                        vstr = f"{v:.4f}"
                    section_lines.append(f"  {k}: {vstr}")
            if section_lines:
                lines.append(f"\n[{section}]")
                lines.extend(section_lines)

        if metrics.get("overfit_warning"):
            lines.append("\n⚠️  과적합 경고: OOS 성과 크게 저하")

        return "\n".join(lines)

    def compare_walk_forward(self, wf_results: List[Dict]) -> pd.DataFrame:
        """Walk-Forward 기간별 성과 비교"""
        if not wf_results:
            return pd.DataFrame()
        return pd.DataFrame(wf_results)
