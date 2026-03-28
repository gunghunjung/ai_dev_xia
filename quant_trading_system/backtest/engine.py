# backtest/engine.py — 기관급 백테스트 엔진 (Walk-Forward)
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Tuple
import numpy as np
import pandas as pd
from datetime import datetime

logger = logging.getLogger("quant.backtest")


@dataclass
class Trade:
    date: pd.Timestamp
    symbol: str
    action: str          # BUY / SELL
    shares: float
    price: float
    cost: float          # 거래비용 포함 총 금액
    commission: float


@dataclass
class BacktestResult:
    """백테스트 결과 컨테이너"""
    equity_curve: pd.Series          # 날짜 → 포트폴리오 가치
    daily_returns: pd.Series         # 일별 수익률
    trades: List[Trade]
    weights_history: pd.DataFrame    # 날짜 × 종목 비중
    metrics: Dict[str, float]
    walk_forward_results: List[Dict] = field(default_factory=list)


class BacktestEngine:
    """
    기관급 Walk-Forward 백테스트 엔진

    특징:
    - 거래비용 / 슬리피지 / 체결지연 반영
    - 미래 데이터 누수 완전 차단
    - Walk-forward expanding window 검증
    - 일별/주별/월별 리밸런싱 지원
    """

    def __init__(
        self,
        initial_capital: float = 100_000_000,
        transaction_cost: float = 0.0015,   # 편도 0.15%
        slippage: float = 0.0005,           # 0.05%
        execution_delay: int = 1,           # 1일 지연 체결
        rebalance_freq: str = "weekly",     # daily/weekly/monthly
        risk_free_rate: float = 0.035,      # 무위험수익률 (기본: 3.5% 한국 기준금리 근사)
    ):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.execution_delay = execution_delay
        self.rebalance_freq = rebalance_freq
        self.risk_free_rate = risk_free_rate

    # ──────────────────────────────────────────────────
    # 메인 백테스트
    # ──────────────────────────────────────────────────

    def run(
        self,
        price_df: pd.DataFrame,          # 날짜 × 종목 종가
        weights_fn: Callable,            # 날짜 → {symbol: weight} 함수
        benchmark: Optional[pd.Series] = None,
        progress_cb: Optional[Callable] = None,
    ) -> BacktestResult:
        """
        백테스트 실행
        Args:
            price_df:   종가 DataFrame (미래 데이터 포함 금지)
            weights_fn: fn(date, available_data) → {symbol: weight}
            benchmark:  벤치마크 수익률 시리즈
            progress_cb: 진행률 콜백 fn(pct: float, msg: str)
        """
        price_df = price_df.sort_index().ffill(limit=5)  # 최대 5일치 gap만 보간 (거래정지/휴일 방어)
        rebal_dates = self._get_rebal_dates(price_df.index)

        capital = self.initial_capital
        positions: Dict[str, float] = {}   # symbol → shares
        equity_history = []
        weights_history = []
        trades = []
        prev_weights: Dict[str, float] = {}
        n = len(rebal_dates)

        for i, date in enumerate(rebal_dates):
            if progress_cb:
                progress_cb(i / n * 100, f"백테스트 중... {date.date()}")

            # 현재까지의 데이터만 사용 (미래 누수 방지)
            hist_data = price_df[price_df.index <= date]

            # 포트폴리오 평가 (현재 날짜 기준)
            portfolio_value = self._calc_portfolio_value(
                positions, price_df, date, capital)

            # 비중 계산 (전날 데이터 사용 → 당일 체결 지연)
            try:
                target_weights = weights_fn(date, hist_data)
            except Exception as e:
                logger.warning(f"{date}: 비중 계산 실패 ({e}), 이전 비중 유지")
                target_weights = prev_weights

            # 리밸런싱 체결 (execution_delay일 후)
            # CRITICAL: exec_date가 가용 데이터 범위를 벗어나면 주문을 건너뜀.
            # 같은 날로 fallback 하면 '리밸런싱 신호 당일 바로 체결'이 되어
            # 미래 데이터 누수(look-ahead bias)가 발생한다.
            exec_date = self._next_trading_day(price_df.index, date,
                                               self.execution_delay)
            if exec_date is None or exec_date not in price_df.index:
                # 데이터 끝 지점 — 체결 불가, 주문 스킵 (bias-free)
                logger.debug(f"{date}: exec_date 가용 데이터 없음 → 주문 스킵")
                equity_history.append({"date": date, "value": portfolio_value})
                weights_history.append({"date": date, **prev_weights})
                continue

            new_trades, capital, positions = self._rebalance(
                positions=positions,
                target_weights=target_weights,
                price_df=price_df,
                exec_date=exec_date,
                portfolio_value=portfolio_value,
                capital=capital,
            )
            trades.extend(new_trades)

            equity_history.append({"date": date, "value": portfolio_value})
            weights_history.append({"date": date, **target_weights})
            prev_weights = target_weights

        # 최종 포트폴리오 가치
        last_date = price_df.index[-1]
        final_value = self._calc_portfolio_value(
            positions, price_df, last_date, capital)
        equity_history.append({"date": last_date, "value": final_value})

        equity_df = pd.DataFrame(equity_history).set_index("date")["value"]
        daily_ret = equity_df.pct_change().dropna()
        weights_df = pd.DataFrame(weights_history).set_index("date").fillna(0)

        metrics = self._compute_metrics(equity_df, daily_ret, benchmark)

        if progress_cb:
            progress_cb(100, "백테스트 완료")

        return BacktestResult(
            equity_curve=equity_df,
            daily_returns=daily_ret,
            trades=trades,
            weights_history=weights_df,
            metrics=metrics,
        )

    def run_walk_forward(
        self,
        price_df: pd.DataFrame,
        strategy_fn: Callable,          # fn(train_data) → weights_fn
        train_days: int = 504,
        test_days: int = 126,
        step_days: int = 63,
        progress_cb: Optional[Callable] = None,
    ) -> BacktestResult:
        """
        Walk-Forward 검증
        - 학습 윈도우로 전략 학습
        - 테스트 윈도우로 검증 (비중첩)
        - 슬라이딩 윈도우로 반복
        """
        dates = price_df.index
        n = len(dates)
        wf_results = []
        all_equity = []

        total_windows = max(1, (n - train_days - test_days) // step_days)
        win_count = 0

        pos = train_days
        while pos + test_days <= n:
            train_start = max(0, pos - train_days)
            train_end = pos
            test_start = pos
            test_end = min(n, pos + test_days)

            train_data = price_df.iloc[train_start:train_end]
            test_data = price_df.iloc[test_start:test_end]

            if len(train_data) < 50 or len(test_data) < 5:
                pos += step_days
                continue

            win_count += 1
            if progress_cb:
                pct = win_count / total_windows * 100
                progress_cb(pct, f"WF {win_count}/{total_windows}: "
                                 f"학습 {dates[train_start].date()}~"
                                 f"{dates[train_end-1].date()}")

            # 전략 학습 (학습 데이터 사용)
            try:
                weights_fn = strategy_fn(train_data)
            except Exception as e:
                logger.error(f"전략 학습 실패: {e}")
                pos += step_days
                continue

            # 테스트 기간 백테스트
            try:
                result = self.run(test_data, weights_fn)
                metrics = result.metrics
                metrics["train_start"] = str(dates[train_start].date())
                metrics["train_end"] = str(dates[train_end-1].date())
                metrics["test_start"] = str(dates[test_start].date())
                metrics["test_end"] = str(dates[test_end-1].date())
                wf_results.append(metrics)
                all_equity.append(result.equity_curve)
            except Exception as e:
                logger.error(f"WF 테스트 실패: {e}")

            pos += step_days

        # 전체 연결 equity curve
        if all_equity:
            combined_equity = self._chain_equity_curves(all_equity)
            daily_ret = combined_equity.pct_change().dropna()
            metrics = self._compute_metrics(combined_equity, daily_ret)
            metrics["n_windows"] = len(wf_results)

            # 과적합 경고: 평균 OOS Sharpe < 0
            if wf_results:
                test_sharpes = [w.get("sharpe_ratio", 0) or 0 for w in wf_results]
                metrics["overfit_warning"] = (
                    len(test_sharpes) > 0 and
                    sum(test_sharpes) / len(test_sharpes) < 0
                )

            result = BacktestResult(
                equity_curve=combined_equity,
                daily_returns=daily_ret,
                trades=[],
                weights_history=pd.DataFrame(),
                metrics=metrics,
                walk_forward_results=wf_results,
            )
        else:
            result = BacktestResult(
                equity_curve=pd.Series(dtype=float),
                daily_returns=pd.Series(dtype=float),
                trades=[],
                weights_history=pd.DataFrame(),
                metrics={},
                walk_forward_results=wf_results,
            )

        return result

    # ──────────────────────────────────────────────────
    # 내부 헬퍼
    # ──────────────────────────────────────────────────

    def _rebalance(
        self,
        positions: Dict[str, float],
        target_weights: Dict[str, float],
        price_df: pd.DataFrame,
        exec_date: pd.Timestamp,
        portfolio_value: float,
        capital: float,
    ) -> Tuple[List[Trade], float, Dict[str, float]]:
        """포트폴리오 리밸런싱 체결"""
        trades = []
        new_positions = dict(positions)

        for sym, target_w in target_weights.items():
            if sym not in price_df.columns:
                continue
            if exec_date not in price_df.index:
                continue

            price = price_df.loc[exec_date, sym]
            if pd.isna(price) or price <= 0:
                continue

            # 슬리피지 적용 (매수: 고, 매도: 저)
            current_shares = new_positions.get(sym, 0)
            target_value = portfolio_value * target_w
            target_shares = target_value / price

            delta_shares = target_shares - current_shares

            if abs(delta_shares) < 0.001:
                continue

            if delta_shares > 0:  # 매수
                exec_price = price * (1 + self.slippage)
                action = "BUY"
            else:  # 매도
                exec_price = price * (1 - self.slippage)
                action = "SELL"

            trade_value = abs(delta_shares) * exec_price
            commission = trade_value * self.transaction_cost
            total_cost = trade_value + commission if action == "BUY" \
                else trade_value - commission

            # 자본 부족 시 조정
            if action == "BUY" and total_cost > capital + 1:
                affordable_shares = capital / (exec_price * (1 + self.transaction_cost))
                delta_shares = min(delta_shares, affordable_shares)
                trade_value = delta_shares * exec_price
                commission = trade_value * self.transaction_cost
                total_cost = trade_value + commission

            if abs(delta_shares) < 0.001:
                continue

            new_positions[sym] = current_shares + delta_shares
            if action == "BUY":
                capital -= total_cost
            else:
                capital += trade_value - commission

            trades.append(Trade(
                date=exec_date,
                symbol=sym,
                action=action,
                shares=abs(delta_shares),
                price=exec_price,
                cost=total_cost,
                commission=commission,
            ))

        # 신호 없는 종목 청산
        for sym in list(new_positions.keys()):
            if sym not in target_weights and new_positions.get(sym, 0) > 0.001:
                if exec_date in price_df.index and sym in price_df.columns:
                    price = price_df.loc[exec_date, sym]
                    if not pd.isna(price) and price > 0:
                        exec_price = price * (1 - self.slippage)
                        shares = new_positions[sym]
                        trade_value = shares * exec_price
                        commission = trade_value * self.transaction_cost
                        capital += trade_value - commission
                        trades.append(Trade(
                            date=exec_date,
                            symbol=sym,
                            action="SELL",
                            shares=shares,
                            price=exec_price,
                            cost=trade_value,
                            commission=commission,
                        ))
                del new_positions[sym]

        return trades, capital, new_positions

    def _calc_portfolio_value(
        self,
        positions: Dict[str, float],
        price_df: pd.DataFrame,
        date: pd.Timestamp,
        cash: float,
    ) -> float:
        """포트폴리오 총 가치 계산"""
        value = cash
        for sym, shares in positions.items():
            if sym not in price_df.columns or shares == 0:
                continue
            idx = price_df.index.get_indexer([date], method="ffill")[0]
            # get_indexer가 -1 반환 시(날짜 미발견) 건너뜀
            if idx < 0 or idx >= len(price_df):
                continue
            price = price_df.iloc[idx][sym]
            if not pd.isna(price) and price > 0:
                value += shares * price
        return max(value, 0.0)  # 음수 방어

    def _get_rebal_dates(self, dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """리밸런싱 날짜 목록"""
        if self.rebalance_freq == "daily":
            return dates
        elif self.rebalance_freq == "weekly":
            # 각 주의 첫 거래일 (월요일이 없으면 화요일 등으로 자동 대체)
            s = pd.Series(dates, index=dates)
            weekly = s.resample("W-MON", closed="left", label="left").first()
            return pd.DatetimeIndex(weekly.dropna().values)
        elif self.rebalance_freq == "monthly":
            # 각 월의 첫 거래일
            s = pd.Series(dates, index=dates)
            monthly = s.resample("MS").first()  # Month Start
            return pd.DatetimeIndex(monthly.dropna().values)
        return dates

    def _next_trading_day(self, all_dates: pd.DatetimeIndex,
                          date: pd.Timestamp, n: int) -> Optional[pd.Timestamp]:
        """n 영업일 후 날짜"""
        idx = all_dates.searchsorted(date)
        target = idx + n
        if target < len(all_dates):
            return all_dates[target]
        return None

    def _chain_equity_curves(self, curves: List[pd.Series]) -> pd.Series:
        """여러 equity curve를 연결 (기준 정규화, 중복 날짜 제거)"""
        combined = []
        base = self.initial_capital
        prev_last_date = None
        for c in curves:
            if len(c) == 0:
                continue
            normalized = c / c.iloc[0] * base
            # 이전 윈도우 마지막 날짜와 겹치는 첫 날 제거 (pct_change=0 왜곡 방지)
            if prev_last_date is not None and normalized.index[0] == prev_last_date:
                normalized = normalized.iloc[1:]
            if len(normalized) == 0:
                continue
            combined.append(normalized)
            base = normalized.iloc[-1]
            prev_last_date = normalized.index[-1]
        if not combined:
            return pd.Series(dtype=float)
        result = pd.concat(combined)
        # 혹시 남은 중복 인덱스 제거 (안전장치)
        return result[~result.index.duplicated(keep="last")]

    def _compute_metrics(
        self,
        equity: pd.Series,
        returns: pd.Series,
        benchmark: Optional[pd.Series] = None,
    ) -> Dict[str, float]:
        """성과 지표 계산"""
        if len(equity) < 2 or len(returns) < 2:
            return {}

        # 기간
        n_days = len(returns)
        years = n_days / 252

        # 수익률
        total_ret = (equity.iloc[-1] / equity.iloc[0]) - 1
        cagr = (1 + total_ret) ** (1 / max(years, 0.01)) - 1

        # 위험
        ann_vol = returns.std() * np.sqrt(252)

        # 최대낙폭
        # ⚠️ 중복 타임스탬프 안전 처리: 정수 위치 기반으로 계산
        eq_vals    = equity.values
        roll_max   = np.maximum.accumulate(eq_vals)
        dd_arr     = (eq_vals - roll_max) / (roll_max + 1e-10)
        max_dd     = float(dd_arr.min())
        dd_end_pos = int(np.argmin(dd_arr))
        dd_start_pos = int(np.argmax(eq_vals[:dd_end_pos + 1]))

        # 타임스탬프 (원본 인덱스)
        try:
            dd_end   = equity.index[dd_end_pos]
            dd_start = equity.index[dd_start_pos]
        except IndexError:
            dd_end   = equity.index[-1]
            dd_start = equity.index[0]

        drawdown = pd.Series(dd_arr, index=equity.index)

        # Sharpe (무위험수익률 차감 — 기본값 3.5% 한국 기준금리 근사)
        risk_free = getattr(self, "risk_free_rate", 0.035)
        sharpe = (cagr - risk_free) / (ann_vol + 1e-10)

        # Sortino (하방 변동성 기준)
        downside = returns[returns < 0]
        sortino_vol = downside.std() * np.sqrt(252) if len(downside) > 0 else ann_vol
        sortino = cagr / (sortino_vol + 1e-10)

        # Calmar
        calmar = cagr / (abs(max_dd) + 1e-10)

        # 승률
        win_rate = (returns > 0).sum() / max(len(returns), 1)

        # 손익비 (평균 수익 / 평균 손실)
        wins   = returns[returns > 0]
        losses = returns[returns < 0]
        avg_win  = float(wins.mean())  if len(wins)   > 0 else 0.0
        avg_loss = float(losses.mean()) if len(losses) > 0 else -1e-10
        payoff_ratio = avg_win / (abs(avg_loss) + 1e-10)

        # VaR / CVaR (95%)
        var_95 = cvar_95 = None
        if len(returns) >= 20:
            var_95  = float(returns.quantile(0.05))
            cvar_95 = float(returns[returns <= returns.quantile(0.05)].mean())

        # 벤치마크 대비 지표
        metrics: dict = {
            "total_return":      float(total_ret),
            "cagr":              float(cagr),
            "annual_volatility": float(ann_vol),
            "max_drawdown":      float(max_dd),
            "sharpe_ratio":      float(sharpe),
            "sortino_ratio":     float(sortino),
            "calmar_ratio":      float(calmar),
            "win_rate":          float(win_rate),
            "payoff_ratio":      float(payoff_ratio),
            "n_days":            n_days,
        }
        if var_95 is not None:
            metrics["var_95"]  = var_95
            metrics["cvar_95"] = cvar_95

        if benchmark is not None and len(benchmark) > 0:
            bench_ret = benchmark.reindex(returns.index).dropna()
            if len(bench_ret) > 0:
                common = returns.reindex(bench_ret.index).dropna()
                bench_common = bench_ret.reindex(common.index).dropna()
                # 길이 맞추기 (dropna 후 길이가 다를 수 있음)
                common = common.reindex(bench_common.index).dropna()
                if len(common) > 1 and len(common) == len(bench_common):
                    cov_mat = np.cov(common.values, bench_common.values)
                    bench_var = bench_common.var()
                    if bench_var > 1e-10 and cov_mat.ndim == 2:
                        beta = float(cov_mat[0, 1] / bench_var)
                        alpha = cagr - beta * (bench_common.mean() * 252)
                        metrics["beta"] = beta
                        metrics["alpha"] = float(alpha)

                    # 정보비율 (IR): 초과수익 / 추적오차
                    active_ret = common - bench_common
                    tracking_error = active_ret.std() * np.sqrt(252)
                    if tracking_error > 1e-10:
                        excess_ann = active_ret.mean() * 252
                        metrics["information_ratio"] = float(
                            excess_ann / tracking_error)

        return metrics
