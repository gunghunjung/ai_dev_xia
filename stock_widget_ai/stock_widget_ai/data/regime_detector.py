"""
RegimeDetector — 시장 국면(Regime) 탐지
────────────────────────────────────────
지원하는 국면:
  0 = BULL_QUIET   (상승 저변동)
  1 = BULL_VOLATILE(상승 고변동)
  2 = BEAR_QUIET   (하락 저변동)
  3 = BEAR_VOLATILE(하락 고변동)
  4 = RANGING      (횡보)

탐지 방법:
  1. Rule-based (기본, 의존성 없음)
     - 수익률 부호 (추세 판단)
     - Realized Volatility z-score (변동성 판단)
     - ADX (추세 강도 판단)
  2. HMM-based (hmmlearn 설치 시 자동 활성화, fallback 지원)

⚠ 누수 방지: 모든 특성은 예측 시점 이전 정보만 사용.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from enum import IntEnum
from ..logger_config import get_logger

log = get_logger("data.regime")


# ── 국면 정의 ──────────────────────────────────────────────────────────────
class Regime(IntEnum):
    BULL_QUIET    = 0
    BULL_VOLATILE = 1
    BEAR_QUIET    = 2
    BEAR_VOLATILE = 3
    RANGING       = 4


REGIME_LABELS = {
    Regime.BULL_QUIET:    "상승 저변동 (Bull Quiet)",
    Regime.BULL_VOLATILE: "상승 고변동 (Bull Volatile)",
    Regime.BEAR_QUIET:    "하락 저변동 (Bear Quiet)",
    Regime.BEAR_VOLATILE: "하락 고변동 (Bear Volatile)",
    Regime.RANGING:       "횡보 (Ranging)",
}

REGIME_COLORS = {
    Regime.BULL_QUIET:    "#2ea043",   # green
    Regime.BULL_VOLATILE: "#d29922",   # yellow
    Regime.BEAR_QUIET:    "#da3633",   # red light
    Regime.BEAR_VOLATILE: "#8b0000",   # dark red
    Regime.RANGING:       "#8b949e",   # gray
}


# ── 메인 클래스 ────────────────────────────────────────────────────────────
class RegimeDetector:
    """
    Parameters
    ----------
    trend_window    : 추세 판단에 사용할 이동평균 기간 (일봉 기준)
    vol_window      : 변동성 rolling window
    vol_z_thresh    : 변동성 Z-score 임계값 (이 이상이면 고변동으로 판단)
    adx_thresh      : ADX 임계값 (이 이하이면 횡보로 판단)
    drift_lookback  : Drift detection 비교 기간 (일)
    """

    def __init__(
        self,
        trend_window: int = 60,
        vol_window: int = 20,
        vol_z_thresh: float = 1.0,
        adx_thresh: float = 20.0,
        drift_lookback: int = 60,
    ) -> None:
        self._tw  = trend_window
        self._vw  = vol_window
        self._vzt = vol_z_thresh
        self._adx = adx_thresh
        self._dl  = drift_lookback
        self._hmm_model = None
        self._try_load_hmm()

    # ── 공개 API ───────────────────────────────────────────────────────────
    def detect(self, df: pd.DataFrame) -> pd.Series:
        """
        df 에 'close', 'high', 'low', 'volume' 컬럼 필요.
        Returns pd.Series[int] — 각 시점의 Regime 번호.
        """
        feats = self._build_regime_features(df)
        if self._hmm_model is not None:
            regimes = self._hmm_detect(feats)
        else:
            regimes = self._rule_detect(feats)
        return pd.Series(regimes, index=df.index, name="regime", dtype=int)

    def current_regime(self, df: pd.DataFrame) -> Tuple[Regime, str, float]:
        """
        Returns (regime_enum, label_str, confidence 0~1)
        """
        series = self.detect(df)
        r = Regime(int(series.iloc[-1]))
        label = REGIME_LABELS.get(r, str(r))
        conf = self._regime_confidence(df)
        return r, label, conf

    def detect_drift(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        최근 30일 vs 이전 drift_lookback 일 분포 비교
        Returns dict with ks_stat, psi, is_drifted
        """
        ret = df["close"].pct_change().dropna()
        if len(ret) < self._dl + 30:
            return {"ks_stat": 0.0, "psi": 0.0, "is_drifted": False}

        recent   = ret.iloc[-30:].values
        baseline = ret.iloc[-(self._dl + 30):-30].values

        ks_stat = self._ks_statistic(baseline, recent)
        psi     = self._psi(baseline, recent)
        is_drifted = ks_stat > 0.15 or psi > 0.2

        log.debug(f"drift: ks={ks_stat:.3f}, psi={psi:.3f}, drifted={is_drifted}")
        return {
            "ks_stat":    float(ks_stat),
            "psi":        float(psi),
            "is_drifted": bool(is_drifted),
        }

    def regime_weights(
        self, regime: Regime, model_names: List[str]
    ) -> Dict[str, float]:
        """
        국면별 권장 모델 가중치 반환.
        추세장 → Transformer/TFT 강세
        횡보장 → GBM/TCN 강세
        고변동 → 앙상블 분산 줄이기 (균등)
        """
        n = len(model_names)
        default = {m: 1.0 / n for m in model_names}

        _boost  = lambda names, w: {m: (w if any(k in m.lower() for k in names) else (1-w*len([x for x in model_names if any(k in x.lower() for k in names)]))/(n-len([x for x in model_names if any(k in x.lower() for k in names)]) or 1)) for m in model_names}

        if regime in (Regime.BULL_QUIET, Regime.BEAR_QUIET):
            # 추세 명확 → 딥러닝 시계열 우선
            return self._weight_by_prefix(
                model_names,
                prefer=["transformer", "tft", "patchtst", "lstm"],
                prefer_w=0.20,
            )
        elif regime == Regime.RANGING:
            # 횡보 → GBM 우선
            return self._weight_by_prefix(
                model_names,
                prefer=["xgb", "lgbm", "catboost", "gbm"],
                prefer_w=0.25,
            )
        else:
            # 고변동 → 균등
            return default

    # ── 내부 메서드 ───────────────────────────────────────────────────────

    def _build_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """국면 판단에 필요한 특성 계산 (누수 없음)"""
        c  = df["close"]
        ret = c.pct_change()

        feats = pd.DataFrame(index=df.index)

        # 추세
        feats["trend_slope"] = (c / c.rolling(self._tw).mean() - 1)
        feats["mom20"]       = c.pct_change(20)
        feats["above_ma"]    = (c > c.rolling(self._tw).mean()).astype(float)

        # 변동성
        feats["rv"]          = ret.rolling(self._vw).std() * np.sqrt(252)
        rv_mean              = feats["rv"].rolling(252).mean()
        rv_std               = feats["rv"].rolling(252).std().replace(0, 1e-9)
        feats["rv_z"]        = (feats["rv"] - rv_mean) / rv_std

        # ADX (간단 구현)
        feats["adx"]         = self._calc_adx(df, 14)

        return feats.fillna(0)

    def _rule_detect(self, feats: pd.DataFrame) -> np.ndarray:
        """규칙 기반 국면 분류"""
        regimes = np.full(len(feats), Regime.RANGING, dtype=int)

        trend_up   = (feats["above_ma"] > 0) & (feats["mom20"] > 0.0)
        trend_dn   = (feats["above_ma"] == 0) & (feats["mom20"] < 0.0)
        high_vol   = feats["rv_z"] > self._vzt
        no_trend   = feats["adx"] < self._adx

        # RANGING 먼저
        regimes[no_trend.values] = Regime.RANGING

        # BULL
        bull_mask = trend_up & ~no_trend
        regimes[bull_mask.values & ~high_vol.values] = Regime.BULL_QUIET
        regimes[bull_mask.values & high_vol.values]  = Regime.BULL_VOLATILE

        # BEAR
        bear_mask = trend_dn & ~no_trend
        regimes[bear_mask.values & ~high_vol.values] = Regime.BEAR_QUIET
        regimes[bear_mask.values & high_vol.values]  = Regime.BEAR_VOLATILE

        return regimes

    def _hmm_detect(self, feats: pd.DataFrame) -> np.ndarray:
        """HMM 기반 상태 감지 (hmmlearn 필요)"""
        try:
            X = feats[["trend_slope", "rv_z", "adx"]].values
            X = np.nan_to_num(X, 0)
            states = self._hmm_model.predict(X)
            # HMM 상태를 Regime 으로 매핑 (변동성/추세로 재분류)
            return self._map_hmm_to_regime(states, feats)
        except Exception as e:
            log.warning(f"HMM 감지 실패, rule-based fallback: {e}")
            return self._rule_detect(feats)

    def _map_hmm_to_regime(
        self, states: np.ndarray, feats: pd.DataFrame
    ) -> np.ndarray:
        """HMM 상태 → Regime 번호 매핑 (사후 분류)"""
        regimes = np.full(len(states), Regime.RANGING, dtype=int)
        for s in np.unique(states):
            mask     = states == s
            mean_mom = feats.loc[mask, "mom20"].mean() if mask.any() else 0
            mean_rv  = feats.loc[mask, "rv_z"].mean()  if mask.any() else 0
            bull     = mean_mom > 0
            high_vol = mean_rv  > self._vzt
            if bull and not high_vol:
                regimes[mask] = Regime.BULL_QUIET
            elif bull and high_vol:
                regimes[mask] = Regime.BULL_VOLATILE
            elif not bull and not high_vol:
                regimes[mask] = Regime.BEAR_QUIET
            elif not bull and high_vol:
                regimes[mask] = Regime.BEAR_VOLATILE
        return regimes

    def _regime_confidence(self, df: pd.DataFrame) -> float:
        """최근 20일 국면 일관성으로 신뢰도 추정 (0~1)"""
        series = self.detect(df)
        last20 = series.iloc[-20:] if len(series) >= 20 else series
        if len(last20) == 0:
            return 0.5
        mode_count = last20.value_counts().iloc[0]
        return float(mode_count / len(last20))

    @staticmethod
    def _calc_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """ADX 간소 구현 (ta 라이브러리 없어도 동작)"""
        try:
            import ta
            return ta.trend.adx(df["high"], df["low"], df["close"], window=period).fillna(0)
        except Exception:
            pass
        # 수동 계산
        h  = df["high"]
        l  = df["low"]
        c  = df["close"]
        tr  = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        dmp = (h - h.shift()).clip(lower=0)
        dmn = (l.shift() - l).clip(lower=0)
        dmp[dmp <= dmn] = 0
        dmn[dmn <= dmp] = 0
        dip = (dmp.rolling(period).mean() / (atr + 1e-9)) * 100
        din = (dmn.rolling(period).mean() / (atr + 1e-9)) * 100
        dx  = ((dip - din).abs() / (dip + din + 1e-9)) * 100
        adx = dx.rolling(period).mean()
        return adx.fillna(0)

    def _try_load_hmm(self) -> None:
        try:
            from hmmlearn.hmm import GaussianHMM
            self._hmm_model = GaussianHMM(
                n_components=5, covariance_type="diag", n_iter=100, random_state=42
            )
            log.debug("hmmlearn 로드 성공 — HMM 국면 감지 활성화")
        except ImportError:
            self._hmm_model = None
            log.debug("hmmlearn 없음 — Rule-based 국면 감지 사용")

    def fit_hmm(self, df: pd.DataFrame) -> None:
        """HMM 모델을 데이터에 피팅 (선택적, hmmlearn 필요)"""
        if self._hmm_model is None:
            return
        feats = self._build_regime_features(df)
        X = feats[["trend_slope", "rv_z", "adx"]].values
        X = np.nan_to_num(X, 0)
        try:
            self._hmm_model.fit(X)
            log.info("HMM fitting 완료")
        except Exception as e:
            log.warning(f"HMM fitting 실패: {e}")
            self._hmm_model = None

    @staticmethod
    def _ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
        """Kolmogorov-Smirnov 통계량 (scipy 없이 구현)"""
        a_sorted = np.sort(a)
        b_sorted = np.sort(b)
        combined = np.concatenate([a_sorted, b_sorted])
        cdf_a = np.searchsorted(a_sorted, combined, side="right") / len(a_sorted)
        cdf_b = np.searchsorted(b_sorted, combined, side="right") / len(b_sorted)
        return float(np.max(np.abs(cdf_a - cdf_b)))

    @staticmethod
    def _psi(baseline: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
        """Population Stability Index"""
        bins = np.percentile(baseline, np.linspace(0, 100, n_bins + 1))
        bins[0]  = -np.inf
        bins[-1] = np.inf
        base_cnt = np.histogram(baseline, bins)[0]
        curr_cnt = np.histogram(current,  bins)[0]
        base_pct = (base_cnt + 1e-9) / (len(baseline) + n_bins * 1e-9)
        curr_pct = (curr_cnt + 1e-9) / (len(current)  + n_bins * 1e-9)
        return float(np.sum((base_pct - curr_pct) * np.log(base_pct / curr_pct)))

    @staticmethod
    def _weight_by_prefix(
        names: List[str], prefer: List[str], prefer_w: float
    ) -> Dict[str, float]:
        """prefer 키워드를 포함하는 모델에 prefer_w 가중치 부여, 나머지 균등 분배"""
        preferred = [n for n in names if any(k in n.lower() for k in prefer)]
        others    = [n for n in names if n not in preferred]
        if not preferred:
            n = len(names)
            return {m: 1.0 / n for m in names}
        w_each_p = prefer_w
        w_total_o = max(0.0, 1.0 - w_each_p * len(preferred))
        w_each_o = w_total_o / len(others) if others else 0.0
        result = {}
        for m in names:
            result[m] = w_each_p if m in preferred else w_each_o
        # 정규화
        total = sum(result.values())
        return {m: v / total for m, v in result.items()}
