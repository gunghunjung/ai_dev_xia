# strategies/bnf_signal_engine.py
# BNF 매수 타이밍 탐지기 — 채점 엔진 (규칙 기반 + 점수 기반 + Veto)
# ─────────────────────────────────────────────────────────────────────────────
# 설계 원칙:
#   1. 단순 RSI<30 같은 단일 지표 금지 — 멀티팩터 가중합
#   2. "반등 확인 없음" → 최대 49점 (매수 후보 미달성)
#   3. 과도한 단기 급등 → 추격 경고, 상한 49점
#   4. 손절/R:R 계산 없는 신호 무효
#   5. 시장 급락 당일 신호 무효
#   6. 이유 설명 필수 (explain_bnf_signal)
# ─────────────────────────────────────────────────────────────────────────────
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import logging
import math
import numpy as np
import pandas as pd

from config.bnf_config import BNFConfig, load_bnf_config
from indicators.bnf_features import (
    BNFFeatureRow,
    compute_bnf_features,
    extract_latest_row,
)

logger = logging.getLogger("quant.bnf.engine")


# ─────────────────────────────────────────────────────────────────────────────
# 결과 구조체
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BNFSignalResult:
    """BNF 채점 결과 (1 종목, 1 시점)"""
    symbol:  str = ""
    date:    str = ""      # 신호 발생 날짜 (YYYY-MM-DD)
    close:   float = np.nan

    # ── 하위 점수 (0~100 scale) ──────────────────────────────────────────────
    trend_score:     float = 0.0
    deviation_score: float = 0.0
    volume_score:    float = 0.0
    reversal_score:  float = 0.0
    market_score:    float = 0.0
    risk_score:      float = 0.0

    # ── 최종 점수 & 판정 ─────────────────────────────────────────────────────
    bnf_score:  float = 0.0   # 0~100
    label:      str   = "관심 없음"
    label_color: str  = "#6c7086"

    # ── 리스크 정보 ──────────────────────────────────────────────────────────
    stop_price:        float = np.nan
    stop_pct:          float = np.nan
    target1:           float = np.nan
    target2:           float = np.nan
    reward_risk_ratio: float = np.nan

    # ── Veto 적용 내역 ────────────────────────────────────────────────────────
    vetos: List[str] = field(default_factory=list)

    # ── 점수 근거 설명 ────────────────────────────────────────────────────────
    reasons: List[str] = field(default_factory=list)

    # ── 유효 여부 ─────────────────────────────────────────────────────────────
    valid: bool = True
    invalid_reason: str = ""

    def as_dict(self) -> dict:
        return {
            "symbol":       self.symbol,
            "date":         self.date,
            "close":        self.close,
            "bnf_score":    round(self.bnf_score, 1),
            "label":        self.label,
            "trend":        round(self.trend_score, 1),
            "deviation":    round(self.deviation_score, 1),
            "volume":       round(self.volume_score, 1),
            "reversal":     round(self.reversal_score, 1),
            "market":       round(self.market_score, 1),
            "risk_sc":      round(self.risk_score, 1),
            "stop_price":   round(self.stop_price, 0) if not math.isnan(self.stop_price) else None,
            "stop_pct":     round(self.stop_pct, 2)   if not math.isnan(self.stop_pct)   else None,
            "target1":      round(self.target1, 0)    if not math.isnan(self.target1)    else None,
            "target2":      round(self.target2, 0)    if not math.isnan(self.target2)    else None,
            "rr":           round(self.reward_risk_ratio, 2) if not math.isnan(self.reward_risk_ratio) else None,
            "vetos":        "; ".join(self.vetos),
            "reasons":      " | ".join(self.reasons),
            "valid":        self.valid,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 내부 유틸
# ─────────────────────────────────────────────────────────────────────────────

def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    if math.isnan(x):
        return 0.0
    return max(lo, min(hi, x))


def _isnan(x) -> bool:
    try:
        return math.isnan(float(x))
    except Exception:
        return True


# ─────────────────────────────────────────────────────────────────────────────
# A. 추세 / 위치 점수
# ─────────────────────────────────────────────────────────────────────────────

def _score_trend(r: BNFFeatureRow, cfg: BNFConfig) -> tuple[float, list]:
    """
    trend_score (0~100) 계산.
    BNF 원칙: 강한 상승추세 첫 눌림목/과매도 해소 가점.
              완전 하락추세 바닥권 → '반등 확인 전 대기' 유지.
              단기 과열 → 감점.
    """
    score = 50.0
    reasons = []

    # ── MA 정렬 ──────────────────────────────────────────────────────────────
    if not _isnan(r.ma5) and not _isnan(r.ma20) and not _isnan(r.ma60):
        if r.close > r.ma5 > r.ma20 > r.ma60:
            score += 15.0
            reasons.append("MA 정배열 (5>20>60)")
        elif r.ma5 < r.ma20 < r.ma60:
            score -= 15.0
            reasons.append("MA 역배열 — 하락추세")
        else:
            # 혼재: 수렴 또는 교차 구간
            reasons.append("MA 수렴/교차 구간")

    # ── MA20 기울기 ───────────────────────────────────────────────────────────
    if not _isnan(r.ma20_slope):
        if r.ma20_slope > 0.005:
            score += 8.0
            reasons.append("MA20 우상향")
        elif r.ma20_slope < -0.005:
            score -= 8.0
            reasons.append("MA20 우하향")

    # ── 레인지 내 위치 ────────────────────────────────────────────────────────
    p60 = r.pos_in_range60
    if not _isnan(p60):
        if p60 < 0.20:
            # 60일 범위 하단 → 바닥권 (반등 확인 전 중립)
            score += 5.0
            reasons.append(f"60일 레인지 하단 ({p60*100:.0f}%위치)")
        elif p60 > 0.85:
            score -= 10.0
            reasons.append(f"60일 레인지 상단 과열 ({p60*100:.0f}%위치)")

    # ── 단기 급등 감점 (추격 금지) ────────────────────────────────────────────
    if not _isnan(r.ret5) and r.ret5 > 10.0:
        score -= 18.0
        reasons.append(f"단기 5일 급등 {r.ret5:.1f}% → 추격 경고")
    elif not _isnan(r.ret3) and r.ret3 > 7.0:
        score -= 12.0
        reasons.append(f"단기 3일 급등 {r.ret3:.1f}%")

    # ── 눌림목 구간 (상승추세에서 MA20 근접) ─────────────────────────────────
    if not _isnan(r.dev_ma20_pct) and not _isnan(r.ma20_slope):
        if -5 < r.dev_ma20_pct < 0 and r.ma20_slope > 0:
            score += 10.0
            reasons.append(f"상승추세 눌림목 (MA20 대비 {r.dev_ma20_pct:.1f}%)")

    # ── Higher Low 형성 ───────────────────────────────────────────────────────
    if r.higher_low > 0:
        score += 7.0
        reasons.append("Higher Low 형성")

    return (_clamp(score), reasons)


# ─────────────────────────────────────────────────────────────────────────────
# B. 괴리율 점수
# ─────────────────────────────────────────────────────────────────────────────

def _score_deviation(r: BNFFeatureRow, cfg: BNFConfig) -> tuple[float, list]:
    score   = 50.0
    reasons = []
    dc = cfg.deviation

    dev = r.dev_ma25_pct
    z   = r.zscore_ma25

    if not _isnan(dev):
        if dev > dc.max_dev_ma25_pct:
            # 과도한 상승 이격 → 추격 금지
            penalty = min(40.0, (dev - dc.max_dev_ma25_pct) * 2.5)
            score -= penalty
            reasons.append(f"MA25 대비 +{dev:.1f}% 과도 이격 (추격 금지)")
        elif dev < dc.min_dev_ma25_pct:
            # 극단적 하락 이격 → 칼잡이 위험
            penalty = min(25.0, abs(dev - dc.min_dev_ma25_pct) * 1.5)
            score -= penalty
            reasons.append(f"MA25 대비 {dev:.1f}% 과도 이격 (칼잡이 위험)")
        elif -10 < dev < -3:
            # 적절한 이격 축소 구간
            score += 15.0
            reasons.append(f"MA25 대비 {dev:.1f}% — 적정 매수 이격 구간")
        elif -3 <= dev <= 0:
            score += 8.0
            reasons.append(f"MA25 근접 ({dev:.1f}%)")

    if not _isnan(z):
        if z < dc.zscore_buy_zone:
            score += 10.0
            reasons.append(f"Z-score {z:.2f} — 저평가 구간")
        elif z > dc.zscore_overbought:
            score -= 12.0
            reasons.append(f"Z-score {z:.2f} — 과매수")

    # VWAP 괴리
    dv = r.dev_vwap20_pct
    if not _isnan(dv):
        if -8 < dv < -2:
            score += 8.0
            reasons.append(f"VWAP 대비 {dv:.1f}% — 저평가")
        elif dv > 10:
            score -= 8.0
            reasons.append(f"VWAP 대비 +{dv:.1f}% — 고평가")

    return (_clamp(score), reasons)


# ─────────────────────────────────────────────────────────────────────────────
# C. 거래량 / 수급 점수
# ─────────────────────────────────────────────────────────────────────────────

def _score_volume(r: BNFFeatureRow, cfg: BNFConfig) -> tuple[float, list]:
    score   = 50.0
    reasons = []
    vc = cfg.volume

    # 유동성 미달 → 무조건 0
    if not _isnan(r.volume) and r.volume < vc.min_volume:
        return (0.0, ["유동성 미달 (거래량 부족)"])
    if not _isnan(r.amount) and r.amount < vc.min_amount_krw:
        return (0.0, [f"거래대금 미달 ({r.amount/1e8:.1f}억원)"])

    # 거래량 비율 (MA20 대비)
    vr = r.vol_ratio20
    if not _isnan(vr):
        if vr >= vc.surge_ratio:
            score += 20.0
            reasons.append(f"거래량 급증 (MA20 대비 {vr:.1f}배)")
        elif vr >= vc.confirm_ratio:
            score += 10.0
            reasons.append(f"거래량 증가 (MA20 대비 {vr:.1f}배)")
        elif vr < vc.dry_up_ratio:
            # 눌림목 구간 거래량 감소는 긍정 (이후 반전 기대)
            score += 5.0
            reasons.append(f"눌림목 거래량 감소 ({vr:.1f}배) — 소화 구간")

    # 수급 편향 (상승일 vs 하락일 거래량)
    vb = r.vol_bias
    if not _isnan(vb):
        if vb > 1.5:
            score += 12.0
            reasons.append(f"매수 수급 우세 (편향 {vb:.2f})")
        elif vb < 0.7:
            score -= 10.0
            reasons.append(f"매도 수급 우세 (편향 {vb:.2f})")

    # 거래량 급증 후 장대양봉 꼭지 → 감점
    if not _isnan(vr) and not _isnan(r.ret3):
        if vr > 3.0 and r.ret3 > 8.0:
            score -= 20.0
            reasons.append(f"거래량 폭발+급등 후 꼭지 경고 (3일 {r.ret3:.1f}%)")

    return (_clamp(score), reasons)


# ─────────────────────────────────────────────────────────────────────────────
# D. 반등 확인 점수
# ─────────────────────────────────────────────────────────────────────────────

def _score_reversal(r: BNFFeatureRow, cfg: BNFConfig) -> tuple[float, list]:
    """
    핵심 원칙: "떨어졌으니 싸다"가 아니라
               "떨어진 뒤 반등 시그널이 확인되었는가"를 채점.
    반등 확인 없음 → 최대 점수 상한이 veto로 49 제한됨.
    """
    score   = 30.0   # 반등 미확인 기본값 (낮게 시작)
    reasons = []
    cc = cfg.candle

    confirmed = False   # 반등 확인 플래그

    # 장대음봉 후 양봉 전환
    if r.recovery_after_bear > 0:
        score += 25.0
        confirmed = True
        reasons.append("장대음봉 후 양봉 전환 확인")

    # 연속 하락 후 첫 양봉
    if r.consec_down >= cc.streak_down_min and r.is_bullish > 0:
        score += 20.0
        confirmed = True
        reasons.append(f"연속 {r.consec_down}봉 하락 후 첫 양봉")

    # Higher Low + 전일 고가 돌파
    if r.higher_low > 0 and r.breaks_prev_high5 > 0:
        score += 15.0
        confirmed = True
        reasons.append("Higher Low + 전일 5일 고가 돌파")
    elif r.higher_low > 0:
        score += 8.0
        reasons.append("Higher Low 형성")

    # 하락 둔화
    if not _isnan(r.ret5) and not _isnan(r.ret10):
        # 최근 5일 수익률이 10일보다 덜 나쁘면 둔화
        if r.ret5 < 0 and r.ret10 < 0 and r.ret5 > r.ret10:
            score += 8.0
            reasons.append(f"하락 둔화 (5일 {r.ret5:.1f}%, 10일 {r.ret10:.1f}%)")

    # 하한가권 근접 + 긍정 캔들 (망치형)
    if r.lower_shadow_ratio > 0.5 and r.is_bullish > 0:
        score += 10.0
        confirmed = True
        reasons.append(f"망치형/도지 반등 캔들 (하단꼬리 {r.lower_shadow_ratio*100:.0f}%)")

    # 박스권 하단 반등
    if r.near_range_bottom > 0 and r.is_bullish > 0:
        score += 8.0
        confirmed = True
        reasons.append("60일 박스권 하단 반등 확인")

    if not confirmed:
        reasons.append("반등 확인 시그널 없음 — 관찰 대기")

    return (_clamp(score), reasons), confirmed


# ─────────────────────────────────────────────────────────────────────────────
# E. 시장 필터 점수
# ─────────────────────────────────────────────────────────────────────────────

def _score_market(market_row: Optional[BNFFeatureRow],
                  cfg: BNFConfig) -> tuple[float, list]:
    """
    시장 지수 필터. market_row가 None이면 중립(50점) 처리.
    """
    if market_row is None or not cfg.market_filter.enabled:
        return (50.0, ["시장 필터 비활성 — 중립"])

    score   = 50.0
    reasons = []
    mc = cfg.market_filter

    # 당일 지수 낙폭
    if not _isnan(market_row.ret1):
        if market_row.ret1 < -mc.max_drop_pct_day:
            score -= 30.0
            reasons.append(f"지수 당일 급락 {market_row.ret1:.1f}%")
        elif market_row.ret1 > 0.5:
            score += 10.0
            reasons.append(f"지수 상승 {market_row.ret1:.1f}%")

    # 지수 중기 추세
    if not _isnan(market_row.ma20_slope):
        if market_row.ma20_slope > 0.003:
            score += 10.0
            reasons.append("지수 MA20 우상향")
        elif market_row.ma20_slope < -0.003:
            score -= 15.0
            reasons.append("지수 MA20 우하향 (하락장)")

    # 지수 변동성 급등
    vr = market_row.vol_ratio20
    if not _isnan(vr) and vr > mc.vol_spike_ratio:
        score -= 10.0
        reasons.append(f"지수 거래량 급등 (불안정 {vr:.1f}배)")

    return (_clamp(score), reasons)


# ─────────────────────────────────────────────────────────────────────────────
# F. 리스크 / 손절 점수
# ─────────────────────────────────────────────────────────────────────────────

def _calc_risk(r: BNFFeatureRow, cfg: BNFConfig) -> tuple[float, float, float, float, list]:
    """
    손절가 / 목표가 / R:R 계산.
    Returns: (stop_price, stop_pct, target1, target2, notes)
    """
    close = r.close
    rc    = cfg.risk
    notes = []

    if _isnan(close) or close <= 0:
        return (np.nan, np.nan, np.nan, np.nan, ["현재가 없음"])

    # ── 손절가 결정 (ATR vs 스윙 저점 중 더 보수적인 값) ──────────────────
    stops = []

    # 1) ATR 기반
    if not _isnan(r.atr14):
        atr_stop = close - r.atr14 * rc.atr_multiplier
        stops.append(atr_stop)
        notes.append(f"ATR손절 {atr_stop:,.0f}원")

    # 2) 스윙 저점 기반
    if rc.use_swing_stop and not _isnan(r.swing_low):
        sw_stop = r.swing_low * 0.99   # 저점 1% 아래
        stops.append(sw_stop)
        notes.append(f"스윙저점 손절 {sw_stop:,.0f}원")

    if not stops:
        return (np.nan, np.nan, np.nan, np.nan, ["손절 계산 불가"])

    # 더 높은 손절가(보수적) 선택
    stop_price = max(stops)
    stop_price = max(stop_price, close * 0.01)   # 최소 1% 손절

    stop_pct = (close - stop_price) / close * 100

    # ── 목표가 ─────────────────────────────────────────────────────────────
    if not _isnan(r.atr14):
        target1 = close + r.atr14 * rc.target1_atr_mult
        target2 = close + r.atr14 * rc.target2_atr_mult
    else:
        target1 = close * (1 + stop_pct * rc.target1_atr_mult / 100)
        target2 = close * (1 + stop_pct * rc.target2_atr_mult / 100)

    # ── R:R ─────────────────────────────────────────────────────────────────
    reward = target1 - close
    risk   = close - stop_price
    rr     = reward / risk if risk > 0 else 0.0

    return (stop_price, stop_pct, target1, target2, notes)


def _score_risk(stop_pct: float, rr: float, cfg: BNFConfig) -> tuple[float, list]:
    score   = 50.0
    reasons = []
    rc = cfg.risk

    if _isnan(stop_pct) or _isnan(rr):
        return (0.0, ["리스크 계산 실패 — 신호 신뢰도 하락"])

    # 손절폭 평가
    if stop_pct > rc.max_stop_pct:
        score -= 30.0
        reasons.append(f"손절폭 {stop_pct:.1f}% 과대 (기준 {rc.max_stop_pct}%)")
    elif stop_pct < 2.0:
        score += 15.0
        reasons.append(f"손절폭 {stop_pct:.1f}% 양호")
    elif stop_pct < 5.0:
        score += 8.0
        reasons.append(f"손절폭 {stop_pct:.1f}%")

    # R:R 평가
    if rr >= 3.0:
        score += 25.0
        reasons.append(f"R:R = {rr:.1f} 우수")
    elif rr >= rc.min_rr:
        score += 12.0
        reasons.append(f"R:R = {rr:.1f} 양호")
    elif rr >= 1.0:
        reasons.append(f"R:R = {rr:.1f} 보통")
    else:
        score -= 25.0
        reasons.append(f"R:R = {rr:.1f} 불량 — 매수 후보 승격 불가")

    return (_clamp(score), reasons)


# ─────────────────────────────────────────────────────────────────────────────
# Veto 적용
# ─────────────────────────────────────────────────────────────────────────────

def _apply_veto(
    raw_score: float,
    reversal_confirmed: bool,
    r: BNFFeatureRow,
    stop_pct: float,
    rr: float,
    market_drop: float,
    cfg: BNFConfig,
) -> tuple[float, list]:
    """
    Veto 규칙 적용 — 최종 점수 상한 제한.

    규칙:
    1. 반등 확인 없음   → 최대 49
    2. 단기 급등 과열   → 최대 49
    3. 손절폭 과대      → 강한 매수 후보 금지 (최대 69)
    4. R:R 미달         → 매수 후보 금지 (최대 49)
    5. 시장 급락 당일   → 신호 무효 (0)
    6. 데이터 이상      → 신호 무효 (0)
    """
    vetos  = []
    score  = raw_score
    rc     = cfg.risk

    # V1. 시장 급락 당일
    if not _isnan(market_drop) and market_drop < -cfg.market_filter.max_drop_pct_day:
        vetos.append(f"V1:시장 급락 {market_drop:.1f}% — 신호 무효")
        return (0.0, vetos)

    # V2. 데이터 이상
    if not r.valid:
        vetos.append("V2:데이터 부족/이상 — 신호 무효")
        return (0.0, vetos)

    # V3. 유동성 미달
    if (not _isnan(r.volume) and r.volume < cfg.volume.min_volume):
        vetos.append("V3:유동성 미달 — 신호 무효")
        return (0.0, vetos)

    # V4. 반등 확인 없음 → 최대 49
    if not reversal_confirmed and score > 49:
        vetos.append("V4:반등 미확인 — 매수 후보 승격 차단 (상한 49)")
        score = min(score, 49.0)

    # V5. 단기 급등 과열 → 최대 49
    if not _isnan(r.ret5) and r.ret5 > 15.0:
        vetos.append(f"V5:단기 급등 과열 {r.ret5:.1f}% — 추격 금지 (상한 49)")
        score = min(score, 49.0)
    elif not _isnan(r.ret3) and r.ret3 > 10.0:
        vetos.append(f"V5:단기 급등 {r.ret3:.1f}% — 추격 주의 (상한 59)")
        score = min(score, 59.0)

    # V6. 손절폭 과대 → 강한 매수 후보 금지 (최대 69)
    if not _isnan(stop_pct) and stop_pct > rc.max_stop_pct:
        vetos.append(f"V6:손절폭 {stop_pct:.1f}% 과대 — 강한 매수 후보 차단 (상한 69)")
        score = min(score, 69.0)

    # V7. R:R 미달 → 매수 후보 금지 (최대 49)
    if not _isnan(rr) and rr < 1.0:
        vetos.append(f"V7:R:R={rr:.2f} 미달 — 매수 후보 승격 차단 (상한 49)")
        score = min(score, 49.0)

    return (_clamp(score), vetos)


# ─────────────────────────────────────────────────────────────────────────────
# 메인 채점 함수
# ─────────────────────────────────────────────────────────────────────────────

def score_bnf_buy_signal(
    df: pd.DataFrame,
    symbol: str = "",
    market_df: Optional[pd.DataFrame] = None,
    cfg: Optional[BNFConfig] = None,
) -> BNFSignalResult:
    """
    OHLCV DataFrame → BNFSignalResult.

    Parameters
    ----------
    df         : 종목 OHLCV DataFrame
    symbol     : 종목 코드 (로깅용)
    market_df  : 시장 지수 OHLCV (없으면 시장 필터 중립)
    cfg        : BNFConfig (None이면 기본값)
    """
    cfg = cfg or load_bnf_config()
    result = BNFSignalResult(symbol=symbol)

    # ── 데이터 검증 ────────────────────────────────────────────────────────────
    if df is None or len(df) < cfg.min_data_days:
        result.valid = False
        result.invalid_reason = f"데이터 부족 ({len(df) if df is not None else 0}행)"
        logger.debug(f"[{symbol}] {result.invalid_reason}")
        return result

    # ── 피처 계산 ──────────────────────────────────────────────────────────────
    try:
        feat_df = compute_bnf_features(df)
        r = extract_latest_row(feat_df)
    except Exception as e:
        result.valid = False
        result.invalid_reason = f"피처 계산 오류: {e}"
        logger.warning(f"[{symbol}] 피처 계산 실패: {e}", exc_info=True)
        return result

    if not r.valid:
        result.valid = False
        result.invalid_reason = "피처 유효성 실패 (NaN 과다)"
        return result

    # ── 날짜 ──────────────────────────────────────────────────────────────────
    try:
        result.date  = str(feat_df.index[-1])[:10]
        result.close = float(r.close) if not _isnan(r.close) else np.nan
    except Exception:
        pass

    # ── 시장 피처 ─────────────────────────────────────────────────────────────
    market_row = None
    market_drop = np.nan
    if market_df is not None and len(market_df) >= 20:
        try:
            mf = compute_bnf_features(market_df)
            market_row = extract_latest_row(mf)
            market_drop = market_row.ret1
        except Exception as e:
            logger.debug(f"시장 피처 계산 실패: {e}")

    # ── 리스크 계산 ───────────────────────────────────────────────────────────
    stop_price, stop_pct, target1, target2, risk_notes = _calc_risk(r, cfg)
    rr = (
        (target1 - r.close) / (r.close - stop_price)
        if not any(_isnan(x) for x in [target1, r.close, stop_price])
           and (r.close - stop_price) > 0
        else np.nan
    )

    result.stop_price        = stop_price
    result.stop_pct          = stop_pct
    result.target1           = target1
    result.target2           = target2
    result.reward_risk_ratio = rr

    # ── 하위 점수 채점 ────────────────────────────────────────────────────────
    w = cfg.weights

    ts, tr = _score_trend(r, cfg)
    ds, dr = _score_deviation(r, cfg)
    vs, vr = _score_volume(r, cfg)
    (rs, rr_reasons), reversal_confirmed = _score_reversal(r, cfg)
    ms, mr = _score_market(market_row, cfg)
    risk_s, risk_r = _score_risk(stop_pct, rr, cfg)

    result.trend_score     = ts
    result.deviation_score = ds
    result.volume_score    = vs
    result.reversal_score  = rs
    result.market_score    = ms
    result.risk_score      = risk_s

    # 가중합
    raw = (
        ts * w.trend +
        ds * w.deviation +
        vs * w.volume +
        rs * w.reversal +
        ms * w.market +
        risk_s * w.risk
    )

    # ── Veto 적용 ──────────────────────────────────────────────────────────────
    final_score, vetos = _apply_veto(
        raw, reversal_confirmed, r, stop_pct, rr, market_drop, cfg
    )

    result.bnf_score   = round(_clamp(final_score), 1)
    result.vetos       = vetos
    result.reasons     = (
        [f"[추세] {x}" for x in tr] +
        [f"[괴리] {x}" for x in dr] +
        [f"[거래량] {x}" for x in vr] +
        [f"[반등] {x}" for x in rr_reasons] +
        [f"[시장] {x}" for x in mr] +
        [f"[리스크] {x}" for x in risk_r + risk_notes]
    )
    result.label       = cfg.label(result.bnf_score)
    result.label_color = cfg.label_color(result.bnf_score)

    logger.debug(
        f"[{symbol}] score={result.bnf_score} label={result.label} "
        f"vetos={vetos} reversal={reversal_confirmed}"
    )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 설명 텍스트 생성
# ─────────────────────────────────────────────────────────────────────────────

def explain_bnf_signal(sig: BNFSignalResult, max_reasons: int = 6) -> str:
    """
    BNFSignalResult → 사람이 읽기 쉬운 근거 설명 문자열.
    """
    lines = [
        f"[{sig.label}]  BNF Score: {sig.bnf_score:.0f} / 100",
        f"종목: {sig.symbol}  현재가: {sig.close:,.0f}원" if not _isnan(sig.close) else "",
    ]
    if sig.stop_price and not _isnan(sig.stop_price):
        lines.append(
            f"손절가: {sig.stop_price:,.0f}원 ({sig.stop_pct:.1f}%)  "
            f"1차목표: {sig.target1:,.0f}원  R:R={sig.reward_risk_ratio:.1f}"
        )

    if sig.reasons:
        lines.append("─" * 40)
        shown = [r for r in sig.reasons if r][:max_reasons]
        lines += [f"  • {r}" for r in shown]

    if sig.vetos:
        lines.append("─" * 40)
        lines.append("⚠ Veto 적용:")
        for v in sig.vetos:
            lines.append(f"  ✗ {v}")

    return "\n".join(l for l in lines if l)


# ─────────────────────────────────────────────────────────────────────────────
# 여러 종목 일괄 스캔
# ─────────────────────────────────────────────────────────────────────────────

def scan_bnf_signals(
    symbol_dfs: Dict[str, pd.DataFrame],
    market_df: Optional[pd.DataFrame] = None,
    cfg: Optional[BNFConfig] = None,
    min_score: float = 30.0,
) -> List[BNFSignalResult]:
    """
    {symbol: ohlcv_df} 딕셔너리를 받아 일괄 채점.
    min_score 이상인 결과만 반환, 점수 내림차순 정렬.
    """
    cfg = cfg or load_bnf_config()
    results = []
    for sym, df in symbol_dfs.items():
        try:
            sig = score_bnf_buy_signal(df, sym, market_df, cfg)
            if sig.valid and sig.bnf_score >= min_score:
                results.append(sig)
        except Exception as e:
            logger.warning(f"[{sym}] 스캔 실패: {e}")

    results.sort(key=lambda x: x.bnf_score, reverse=True)
    return results
