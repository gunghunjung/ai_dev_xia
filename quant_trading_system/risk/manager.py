# risk/manager.py — 리스크 관리 엔진
from __future__ import annotations
import logging
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger("quant.risk")


class MarketRegime:
    """시장 레짐 (강세/약세/중립)"""
    BULL = "강세장"
    BEAR = "약세장"
    NEUTRAL = "중립"
    HIGH_VOL = "고변동성"


class RiskManager:
    """
    동적 리스크 관리 엔진

    기능:
    1. 변동성 타겟팅: 포트폴리오 레버리지 조정
    2. 최대낙폭 제한: 한도 초과 시 익스포저 축소
    3. 시장 레짐 감지: 강세/약세/고변동성 구분
    4. 킬스위치: Sharpe 급락 시 전체 포지션 청산
    5. 상관관계 제한: 집중 리스크 방지
    """

    def __init__(
        self,
        vol_target: float = 0.15,
        max_drawdown_limit: float = 0.20,
        kill_switch_sharpe: float = -0.5,
        regime_detection: bool = True,
        vol_lookback: int = 20,
        correlation_cap: float = 0.7,
    ):
        self.vol_target = vol_target
        self.max_dd_limit = max_drawdown_limit
        self.kill_switch_sharpe = kill_switch_sharpe
        self.regime_detection = regime_detection
        self.vol_lookback = vol_lookback
        self.correlation_cap = correlation_cap

        self._kill_switch_active = False
        self._current_regime = MarketRegime.NEUTRAL
        self._peak_value = None

    def adjust_weights(
        self,
        weights: Dict[str, float],
        portfolio_returns: pd.Series,
        current_value: float,
        equity_curve: Optional[pd.Series] = None,
    ) -> Dict[str, float]:
        """
        리스크 조정된 포트폴리오 비중 반환
        Args:
            weights:           원본 포트폴리오 비중
            portfolio_returns: 과거 일별 수익률
            current_value:     현재 포트폴리오 가치
            equity_curve:      누적 equity curve
        Returns:
            조정된 비중 (익스포저 < 1 → 현금 보유)
        """
        if not weights:
            return {}

        # 킬스위치 확인
        if self._check_kill_switch(portfolio_returns):
            logger.warning("🚨 킬스위치 발동! 전체 포지션 청산")
            return {}

        # 최대낙폭 한도 확인
        if equity_curve is not None:
            dd = self._current_drawdown(equity_curve)
            if dd < -self.max_dd_limit:
                exposure = max(0, 1 - (-dd - self.max_dd_limit) * 2)
                logger.warning(f"낙폭 한도 초과: {dd:.2%} → 익스포저 {exposure:.0%}로 축소")
                return {s: w * exposure for s, w in weights.items()}

        # 레짐 감지
        if self.regime_detection and len(portfolio_returns) >= self.vol_lookback:
            self._current_regime = self._detect_regime(portfolio_returns)

        # 변동성 스케일링
        exposure = self._vol_scale(portfolio_returns)

        # 레짐별 익스포저 조정
        exposure = self._regime_adjust(exposure)

        if exposure < 1.0:
            logger.info(f"익스포저 조정: {exposure:.2%} (레짐: {self._current_regime})")

        return {s: w * exposure for s, w in weights.items()}

    def check_correlation(
        self,
        weights: Dict[str, float],
        returns_df: pd.DataFrame,
    ) -> Dict[str, float]:
        """상관관계 집중 방지"""
        syms = [s for s in weights if s in returns_df.columns]
        if len(syms) < 2:
            return weights

        corr = returns_df[syms].dropna().corr()
        adjusted = dict(weights)

        for i, s1 in enumerate(syms):
            for j, s2 in enumerate(syms):
                if i >= j:
                    continue
                if abs(corr.loc[s1, s2]) > self.correlation_cap:
                    # 더 작은 비중의 종목 비중 절반 축소
                    if adjusted.get(s1, 0) < adjusted.get(s2, 0):
                        adjusted[s1] = adjusted.get(s1, 0) * 0.5
                    else:
                        adjusted[s2] = adjusted.get(s2, 0) * 0.5

        return adjusted

    def get_regime(self) -> str:
        return self._current_regime

    def is_kill_switch_active(self) -> bool:
        return self._kill_switch_active

    def reset_kill_switch(self) -> None:
        self._kill_switch_active = False
        logger.info("킬스위치 해제")

    # ──────────────────────────────────────────────────
    # 내부 메서드
    # ──────────────────────────────────────────────────

    def _vol_scale(self, returns: pd.Series) -> float:
        """변동성 타겟팅: 목표 변동성 대비 현재 변동성 비율로 레버리지 조정"""
        if len(returns) < self.vol_lookback:
            return 1.0
        recent = returns.iloc[-self.vol_lookback:]
        realized_vol = recent.std() * np.sqrt(252)
        if realized_vol < 1e-6:
            return 1.0
        exposure = min(1.0, self.vol_target / realized_vol)
        return exposure

    def _detect_regime(self, returns: pd.Series) -> str:
        """HMM-lite 레짐 감지 (단순 휴리스틱)"""
        if len(returns) < 60:
            return MarketRegime.NEUTRAL

        short = returns.iloc[-20:].mean() * 252
        long = returns.iloc[-60:].mean() * 252
        vol = returns.iloc[-20:].std() * np.sqrt(252)

        if vol > 0.35:
            return MarketRegime.HIGH_VOL
        elif short > 0.05 and long > 0.0:
            return MarketRegime.BULL
        elif short < -0.05 and long < 0.0:
            return MarketRegime.BEAR
        else:
            return MarketRegime.NEUTRAL

    def _regime_adjust(self, exposure: float) -> float:
        """레짐별 익스포저 조정 계수"""
        adj = {
            MarketRegime.BULL: 1.0,
            MarketRegime.NEUTRAL: 0.9,
            MarketRegime.HIGH_VOL: 0.7,
            MarketRegime.BEAR: 0.5,
        }
        return exposure * adj.get(self._current_regime, 1.0)

    def _current_drawdown(self, equity: pd.Series) -> float:
        """현재 낙폭 계산"""
        if len(equity) < 2:
            return 0.0
        peak = equity.cummax().iloc[-1]
        current = equity.iloc[-1]
        return (current - peak) / (peak + 1e-10)

    def _check_kill_switch(self, returns: pd.Series) -> bool:
        """롤링 Sharpe 기반 킬스위치"""
        if self._kill_switch_active:
            return True
        if len(returns) < 60:
            return False
        recent = returns.iloc[-60:]
        roll_sharpe = recent.mean() * 252 / (recent.std() * np.sqrt(252) + 1e-10)
        if roll_sharpe < self.kill_switch_sharpe:
            self._kill_switch_active = True
            return True
        return False

    def compute_var(
        self,
        returns: pd.Series,
        weights: Dict[str, float],
        returns_df: pd.DataFrame,
        confidence: float = 0.95,
    ) -> Tuple[float, float]:
        """
        VaR / CVaR 계산 (역사적 시뮬레이션)
        Returns:
            (VaR, CVaR) — 1일 기준, 신뢰수준 95%
        """
        syms = [s for s in weights if s in returns_df.columns]
        if not syms:
            return 0.0, 0.0

        w = np.array([weights[s] for s in syms])
        w = w / w.sum()
        port_ret = (returns_df[syms].dropna() @ w)

        if len(port_ret) < 20:
            return 0.0, 0.0

        var = np.percentile(port_ret, (1 - confidence) * 100)
        cvar = port_ret[port_ret <= var].mean()
        return float(var), float(cvar)
