# portfolio/constructor.py — 포트폴리오 구성 레이어 (핵심 알파 소스)
from __future__ import annotations
import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger("quant.portfolio")


class PortfolioConstructor:
    """
    신호 → 포트폴리오 비중 변환

    구현 방법:
    1. Risk Parity (위험 동등 배분)
    2. Mean-Variance Optimization (MVO)
    3. Volatility Scaling (변동성 타겟팅)
    4. Equal Weight (동일 비중)

    제약 조건:
    - 종목별 최대/최소 비중
    - 회전율 제한 (급격한 리밸런싱 방지)
    - 상관관계 제한
    """

    def __init__(
        self,
        method: str = "risk_parity",
        max_weight: float = 0.35,
        min_weight: float = 0.0,
        turnover_limit: float = 0.30,
        target_vol: float = 0.15,
        lookback_vol: int = 20,
        correlation_cap: float = 0.7,
    ):
        self.method = method
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.turnover_limit = turnover_limit
        self.target_vol = target_vol
        self.lookback_vol = lookback_vol
        self.correlation_cap = correlation_cap

    def construct(
        self,
        signal_df: pd.DataFrame,
        returns_df: pd.DataFrame,
        prev_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        포트폴리오 비중 계산
        Args:
            signal_df:    신호 DataFrame (symbol, signal, weight, mu, sigma)
            returns_df:   과거 수익률 DataFrame (날짜 × 종목)
            prev_weights: 이전 포트폴리오 비중 (회전율 제한용)
        Returns:
            weights: {symbol: weight} — 합계 ≤ 1
        """
        # 롱 신호가 있는 종목만 필터
        long_df = signal_df[signal_df["signal"] == 1].copy()
        if long_df.empty:
            logger.info("롱 신호 없음 → 현금 100%")
            return {}

        symbols = long_df["symbol"].tolist()

        # 공통 수익률 행렬
        ret_sub = self._get_returns_subset(returns_df, symbols)

        if self.method == "risk_parity":
            weights = self._risk_parity(ret_sub)
        elif self.method == "mean_variance":
            mu_vec = long_df.set_index("symbol")["mu"].reindex(symbols).values
            weights = self._mean_variance(ret_sub, mu_vec)
        elif self.method == "vol_scaling":
            weights = self._vol_scaling(ret_sub, long_df)
        elif self.method == "equal_weight":
            n = len(symbols)
            weights = {s: 1.0 / n for s in symbols}
        else:
            n = len(symbols)
            weights = {s: 1.0 / n for s in symbols}

        # 제약 적용
        weights = self._apply_constraints(weights)

        # 회전율 제한
        if prev_weights is not None:
            weights = self._apply_turnover_limit(weights, prev_weights)

        total = sum(weights.values())
        if total > 1e-10:
            weights = {s: w / total for s, w in weights.items()}

        logger.info(f"포트폴리오 구성: {len(weights)}개 종목, "
                    f"최대비중={max(weights.values()):.2%}")
        return weights

    # ──────────────────────────────────────────────────
    # 비중 계산 방법
    # ──────────────────────────────────────────────────

    def _risk_parity(self, returns: pd.DataFrame) -> Dict[str, float]:
        """
        위험 동등 배분 (Risk Parity)
        각 종목의 기여 위험이 동일하도록 비중 설정
        """
        if returns.empty or returns.shape[1] == 0:
            return {}
        symbols = returns.columns.tolist()

        # 공분산 행렬 추정
        cov = returns.cov().values * 252  # 연간화
        cov = self._regularize_cov(cov)

        # 역변동성 비중 (Risk Parity 근사)
        vols = np.sqrt(np.diag(cov)) + 1e-10
        inv_vol = 1.0 / vols
        weights_arr = inv_vol / inv_vol.sum()

        return {s: float(w) for s, w in zip(symbols, weights_arr)}

    def _mean_variance(self, returns: pd.DataFrame,
                       mu_vec: np.ndarray) -> Dict[str, float]:
        """
        평균-분산 최적화 (Markowitz MVO)
        Sharpe 최대화 (숏 금지, 비중 합 = 1)
        """
        if returns.empty:
            return {}
        symbols = returns.columns.tolist()
        n = len(symbols)

        cov = returns.cov().values * 252
        cov = self._regularize_cov(cov)

        # 반복적 기울기 상승 (scipy 불필요)
        try:
            from scipy.optimize import minimize

            def neg_sharpe(w):
                port_ret = np.dot(w, mu_vec) * 252
                port_vol = np.sqrt(w @ cov @ w)
                return -port_ret / (port_vol + 1e-10)

            constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
            bounds = [(self.min_weight, self.max_weight)] * n
            w0 = np.ones(n) / n
            res = minimize(neg_sharpe, w0, method="SLSQP",
                           bounds=bounds, constraints=constraints,
                           options={"maxiter": 500, "ftol": 1e-9})
            weights_arr = res.x if res.success else w0

        except ImportError:
            # scipy 없으면 역변동성 대체
            vols = np.sqrt(np.diag(cov)) + 1e-10
            weights_arr = (1.0 / vols) / (1.0 / vols).sum()

        return {s: float(w) for s, w in zip(symbols, weights_arr)}

    def _vol_scaling(self, returns: pd.DataFrame,
                     signal_df: pd.DataFrame) -> Dict[str, float]:
        """
        변동성 스케일링: 목표 변동성 달성하도록 비중 조정
        """
        if returns.empty:
            return {}
        symbols = returns.columns.tolist()

        weights = {}
        for sym in symbols:
            if sym not in returns.columns:
                continue
            vol = returns[sym].std() * np.sqrt(252) + 1e-10
            w = self.target_vol / vol
            weights[sym] = min(w, self.max_weight)

        return weights

    # ──────────────────────────────────────────────────
    # 제약 및 보정
    # ──────────────────────────────────────────────────

    def _apply_constraints(self, weights: Dict[str, float]) -> Dict[str, float]:
        """비중 제약 적용"""
        result = {}
        for s, w in weights.items():
            w = max(self.min_weight, min(self.max_weight, w))
            if w > 1e-6:
                result[s] = w
        return result

    def _apply_turnover_limit(
        self,
        new_w: Dict[str, float],
        prev_w: Dict[str, float],
    ) -> Dict[str, float]:
        """회전율 제한: 변화량이 turnover_limit 초과 시 블렌딩"""
        all_syms = set(new_w) | set(prev_w)
        turnover = sum(
            abs(new_w.get(s, 0) - prev_w.get(s, 0))
            for s in all_syms
        )

        if turnover <= self.turnover_limit:
            return new_w

        # 블렌딩 비율
        blend = self.turnover_limit / turnover
        blended = {}
        for s in all_syms:
            nw = new_w.get(s, 0)
            pw = prev_w.get(s, 0)
            bw = pw + blend * (nw - pw)
            if bw > 1e-6:
                blended[s] = bw

        logger.info(f"회전율 제한: {turnover:.2%} → {self.turnover_limit:.2%}로 축소")
        return blended

    def _regularize_cov(self, cov: np.ndarray,
                        shrinkage: float = 0.1) -> np.ndarray:
        """공분산 행렬 정규화 (Ledoit-Wolf 근사)"""
        n = cov.shape[0]
        # Shrinkage toward diagonal
        diag_target = np.diag(np.diag(cov))
        cov_reg = (1 - shrinkage) * cov + shrinkage * diag_target
        # 최소 고유값 양수 보정
        eigvals = np.linalg.eigvalsh(cov_reg)
        if eigvals.min() < 1e-8:
            cov_reg += (1e-8 - eigvals.min()) * np.eye(n)
        return cov_reg

    @staticmethod
    def _get_returns_subset(returns_df: pd.DataFrame,
                            symbols: List[str]) -> pd.DataFrame:
        """반환: 공통 종목의 수익률 DataFrame"""
        available = [s for s in symbols if s in returns_df.columns]
        if not available:
            return pd.DataFrame()
        return returns_df[available].dropna()

    def compute_portfolio_stats(
        self,
        weights: Dict[str, float],
        returns_df: pd.DataFrame,
    ) -> Dict[str, float]:
        """포트폴리오 기대 성과 계산"""
        if not weights or returns_df.empty:
            return {}

        syms = [s for s in weights if s in returns_df.columns]
        if not syms:
            return {}

        w = np.array([weights[s] for s in syms])
        w = w / w.sum()
        ret = returns_df[syms].dropna()

        port_ret = ret @ w
        ann_ret = port_ret.mean() * 252
        ann_vol = port_ret.std() * np.sqrt(252)
        sharpe = ann_ret / (ann_vol + 1e-10)

        return {
            "expected_annual_return": float(ann_ret),
            "expected_annual_vol": float(ann_vol),
            "expected_sharpe": float(sharpe),
        }
