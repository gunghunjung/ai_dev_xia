# config/settings.py — 전체 시스템 설정 (데이터클래스 기반)
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
import json
import os


@dataclass
class DataConfig:
    """데이터 수집 설정"""
    symbols: List[str] = field(default_factory=lambda: [
        "005930.KS",   # 삼성전자
        "000660.KS",   # SK하이닉스
        "035420.KS",   # NAVER
        "051910.KS",   # LG화학
        "006400.KS",   # 삼성SDI
    ])
    benchmark: str = "^KS11"          # KOSPI 지수
    period: str = "5y"                # 데이터 기간 (1y/2y/5y/max)
    interval: str = "1d"              # 데이터 주기 (1d/1wk)
    cache_dir: str = "cache"          # 캐시 저장 경로
    cache_ttl_hours: int = 23         # 캐시 유효시간 (시간)
    source: str = "yfinance"          # yfinance / pykrx
    use_cache: bool = True


@dataclass
class ROIConfig:
    """ROI(관심 구간) 감지 설정"""
    # 구간 길이
    segment_length: int = 30          # ROI 고정 길이 (캔들 수)
    lookahead: int = 5                # 레이블 미래 수익률 계산 기간

    # 감지 임계값
    vol_z_threshold: float = 1.5      # 변동성 z-score 임계값
    breakout_threshold: float = 2.0   # 가격 돌파 z-score 임계값
    volume_spike_threshold: float = 2.0  # 거래량 급증 z-score 임계값

    # 필터
    min_roi_spacing: int = 5          # ROI 최소 간격 (중복 방지)
    rolling_window: int = 20          # 롤링 통계 윈도우


@dataclass
class ModelConfig:
    """모델 아키텍처 및 학습 설정"""
    # CNN 인코더
    cnn_out_dim: int = 128            # CNN 출력 임베딩 차원
    cnn_channels: List[int] = field(default_factory=lambda: [32, 64, 128])
    cnn_kernel_size: int = 3

    # Transformer 인코더
    d_model: int = 256                # Transformer 임베딩 차원
    nhead: int = 8                    # 멀티헤드 어텐션 헤드 수
    num_encoder_layers: int = 4       # Transformer 레이어 수
    dim_feedforward: int = 512        # FFN 차원
    dropout: float = 0.1             # 드롭아웃 비율
    max_seq_len: int = 60             # 최대 시퀀스 길이

    # 학습 하이퍼파라미터
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 32
    epochs: int = 100
    patience: int = 15               # Early stopping 인내 epoch 수
    grad_clip: float = 1.0           # Gradient clipping

    # 하드웨어
    use_cuda: bool = True            # CUDA GPU 사용 여부
    num_workers: int = 0             # DataLoader workers (Windows=0)

    # 이미지 피처
    image_size: int = 64             # GAF/캔들스틱 이미지 크기


@dataclass
class PortfolioConfig:
    """포트폴리오 구성 설정"""
    method: str = "risk_parity"      # risk_parity / mean_variance / vol_scaling / equal_weight
    max_weight: float = 0.35         # 종목 최대 비중
    min_weight: float = 0.0          # 종목 최소 비중
    turnover_limit: float = 0.30     # 1회 리밸런싱 최대 회전율
    target_volatility: float = 0.15  # 목표 연간 변동성 (15%)
    rebalance_freq: str = "weekly"   # daily / weekly / monthly
    lookback_vol: int = 20           # 변동성 추정 윈도우


@dataclass
class BacktestConfig:
    """백테스트 설정"""
    initial_capital: float = 100_000_000   # 초기 자본 (1억원)
    transaction_cost: float = 0.0015       # 편도 거래비용 (0.15% = 증권사 수수료 + 세금)
    slippage: float = 0.0005              # 슬리피지 (0.05%)
    execution_delay: int = 1              # 체결 지연 (영업일)

    # Walk-forward 설정
    wf_train_days: int = 504              # 학습 윈도우 (2년)
    wf_test_days: int = 126               # 테스트 윈도우 (6개월)
    wf_step_days: int = 63               # 슬라이딩 스텝 (3개월)


@dataclass
class RiskConfig:
    """리스크 관리 설정"""
    vol_target: float = 0.15              # 목표 포트폴리오 변동성
    max_drawdown_limit: float = 0.20      # 최대 낙폭 한도 (20%)
    kill_switch_sharpe: float = -0.5     # Sharpe 킬스위치 임계값
    regime_detection: bool = True         # 시장 레짐 감지 사용 여부
    vol_lookback: int = 20               # 변동성 계산 윈도우
    correlation_cap: float = 0.7         # 상관관계 상한


@dataclass
class AppSettings:
    """전체 애플리케이션 설정"""
    data: DataConfig = field(default_factory=DataConfig)
    roi: ROIConfig = field(default_factory=ROIConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)

    # 경로 설정
    model_dir: str = "outputs/models"
    output_dir: str = "outputs"
    log_level: str = "INFO"

    # GUI 설정
    window_width: int = 1400
    window_height: int = 900
    theme: str = "light"


_SETTINGS_PATH = os.path.join(os.path.dirname(__file__), "..", "settings.json")


def _deep_update(base: dict, update: dict) -> dict:
    """딕셔너리 깊은 업데이트"""
    for k, v in update.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_settings(path: str = _SETTINGS_PATH) -> AppSettings:
    """JSON 파일에서 설정 로드"""
    defaults = asdict(AppSettings())
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                saved = json.load(f)
            _deep_update(defaults, saved)
        except Exception:
            pass

    s = AppSettings(
        data=DataConfig(**defaults["data"]),
        roi=ROIConfig(**defaults["roi"]),
        model=ModelConfig(**defaults["model"]),
        portfolio=PortfolioConfig(**defaults["portfolio"]),
        backtest=BacktestConfig(**defaults["backtest"]),
        risk=RiskConfig(**defaults["risk"]),
        model_dir=defaults["model_dir"],
        output_dir=defaults["output_dir"],
        log_level=defaults["log_level"],
        window_width=defaults["window_width"],
        window_height=defaults["window_height"],
        theme=defaults["theme"],
    )
    return s


def save_settings(settings: AppSettings, path: str = _SETTINGS_PATH) -> None:
    """설정을 JSON 파일로 저장"""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(settings), f, indent=2, ensure_ascii=False)
