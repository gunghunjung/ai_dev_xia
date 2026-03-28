# config/bnf_config.py — BNF 매수 타이밍 탐지기 전체 설정
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List
import json, os


# ─────────────────────────────────────────────────────────────────────────────
# 하위 설정 그룹
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BNFWeightConfig:
    """BNF 하위 점수 가중치  (합계 = 1.0 이어야 함)"""
    trend:     float = 0.20   # A. 추세/위치
    deviation: float = 0.15   # B. 괴리율
    volume:    float = 0.20   # C. 거래량/수급
    reversal:  float = 0.25   # D. 반등 확인
    market:    float = 0.10   # E. 장세 필터
    risk:      float = 0.10   # F. 리스크

    def total(self) -> float:
        return self.trend + self.deviation + self.volume + self.reversal + self.market + self.risk

    def is_valid(self) -> bool:
        return abs(self.total() - 1.0) < 0.01


@dataclass
class BNFThresholdConfig:
    """단계 판정 임계값"""
    observe:   int = 30   # 관찰 필요
    candidate: int = 50   # 매수 후보
    strong:    int = 70   # 강한 매수 후보


@dataclass
class BNFDeviationConfig:
    """괴리율 기준"""
    # MA25 기준 과도한 상승 → 추격 금지
    max_dev_ma25_pct:  float = 12.0   # 이 이상이면 감점·추격 경고
    # 반등 유효 구간 (너무 떨어진 종목은 '칼잡이 위험')
    min_dev_ma25_pct:  float = -20.0
    # Z-score 경계
    zscore_overbought: float =  2.0   # 과매수 경계
    zscore_buy_zone:   float = -1.0   # 매수 관심 구간 (이하)
    # VWAP 사용 여부
    use_vwap:          bool  = True


@dataclass
class BNFVolumeConfig:
    """거래량/수급 기준"""
    surge_ratio:    float = 1.5    # MA20 대비 급증 배수
    confirm_ratio:  float = 1.2    # 반등 확인 최소 배수
    dry_up_ratio:   float = 0.7    # 눌림목 감소 기준
    min_volume:     int   = 10_000  # 최소 거래량 (유동성 필터)
    min_amount_krw: int   = 100_000_000   # 최소 거래대금 (1억원)


@dataclass
class BNFRiskConfig:
    """리스크 / 손절 기준"""
    min_rr:           float = 1.5   # 최소 R:R 비율
    max_stop_pct:     float = 8.0   # 최대 허용 손절폭 (%)
    atr_multiplier:   float = 2.0   # ATR 기반 손절 배수
    target1_atr_mult: float = 3.0   # 1차 목표: ATR × 배수
    target2_atr_mult: float = 6.0   # 2차 목표: ATR × 배수
    use_swing_stop:   bool  = True   # 스윙 저점 기반 손절 병행


@dataclass
class BNFCandleConfig:
    """캔들 패턴 판정 기준"""
    big_candle_atr_mult:     float = 1.5   # 장대봉 기준 (ATR × 배수)
    engulf_body_ratio:       float = 0.6   # 장악형 몸통 비율
    reversal_confirm_bars:   int   = 2     # 반등 확인 최소 봉 수
    streak_down_min:         int   = 3     # 연속 하락 최소 봉 수
    hl_lookback:             int   = 20    # Higher Low 탐지 구간


@dataclass
class BNFMarketFilterConfig:
    """시장(지수) 필터 기준"""
    enabled:               bool  = True
    max_drop_pct_day:      float = 3.0    # 당일 지수 낙폭 제한 (%)
    vol_spike_ratio:       float = 2.0    # 지수 변동성 급등 배수
    bear_trend_days:       int   = 10     # 지수 하락 추세 판단 기간
    market_ticker:         str   = "^KS11"  # 기본 지수 (KOSPI)


@dataclass
class BNFConfig:
    """BNF 매수 타이밍 탐지기 마스터 설정"""
    weights:       BNFWeightConfig       = field(default_factory=BNFWeightConfig)
    thresholds:    BNFThresholdConfig    = field(default_factory=BNFThresholdConfig)
    deviation:     BNFDeviationConfig    = field(default_factory=BNFDeviationConfig)
    volume:        BNFVolumeConfig       = field(default_factory=BNFVolumeConfig)
    risk:          BNFRiskConfig         = field(default_factory=BNFRiskConfig)
    candle:        BNFCandleConfig       = field(default_factory=BNFCandleConfig)
    market_filter: BNFMarketFilterConfig = field(default_factory=BNFMarketFilterConfig)

    # 일반
    mode:           str = "swing"   # "swing" | "short"
    lookback_days:  int = 120       # 피처 계산 기준 기간
    min_data_days:  int = 60        # 최소 필요 데이터

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "BNFConfig":
        def _sub(klass, key):
            return klass(**{k: v for k, v in d.get(key, {}).items()
                            if k in klass.__dataclass_fields__})
        return cls(
            weights       = _sub(BNFWeightConfig,       "weights"),
            thresholds    = _sub(BNFThresholdConfig,    "thresholds"),
            deviation     = _sub(BNFDeviationConfig,    "deviation"),
            volume        = _sub(BNFVolumeConfig,       "volume"),
            risk          = _sub(BNFRiskConfig,         "risk"),
            candle        = _sub(BNFCandleConfig,       "candle"),
            market_filter = _sub(BNFMarketFilterConfig, "market_filter"),
            mode          = d.get("mode", "swing"),
            lookback_days = d.get("lookback_days", 120),
            min_data_days = d.get("min_data_days", 60),
        )

    def label(self, score: float) -> str:
        """점수 → 단계 레이블"""
        t = self.thresholds
        if score >= t.strong:    return "강한 매수 후보"
        if score >= t.candidate: return "매수 후보"
        if score >= t.observe:   return "관찰 필요"
        return "관심 없음"

    def label_color(self, score: float) -> str:
        t = self.thresholds
        if score >= t.strong:    return "#a6e3a1"   # green
        if score >= t.candidate: return "#f9e2af"   # yellow
        if score >= t.observe:   return "#89dceb"   # sky
        return "#6c7086"                             # dimmed


# ─────────────────────────────────────────────────────────────────────────────
# 파일 저장/불러오기
# ─────────────────────────────────────────────────────────────────────────────
_CFG_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs", "bnf_config.json")


def load_bnf_config(path: str = _CFG_PATH) -> BNFConfig:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return BNFConfig.from_dict(json.load(f))
        except Exception:
            pass
    return BNFConfig()


def save_bnf_config(cfg: BNFConfig, path: str = _CFG_PATH) -> None:
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, indent=2, ensure_ascii=False)
