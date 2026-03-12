"""
AppState dataclass — 모든 GUI 상태를 타입 안전하게 관리
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict
import copy

SCHEMA_VERSION = 3

# 파이프라인 단계 이름 목록 (순서 고정)
PIPELINE_STEPS = [
    "data_load",       # 1. 데이터 로드
    "feature_build",   # 2. 피처 생성
    "regime_detect",   # 3. 국면 탐지
    "model_train",     # 4. 모델 학습
    "walk_forward",    # 5. Walk-forward 검증
    "forecast",        # 6. 예측 (ForecastEngine)
    "conformal",       # 7. Conformal 보정
    "explain",         # 8. 설명 (SHAP)
    "backtest",        # 9. 백테스트
    "report",          # 10. 보고서 생성
]


@dataclass
class WindowState:
    width: int = 1400
    height: int = 900
    x: int = 80
    y: int = 80
    maximized: bool = False
    last_tab: str = "data"
    splitter_ratio: float = 0.35   # left panel ratio
    geometry_b64: str = ""         # Qt saveGeometry() → base64 (가장 신뢰성 높음)


@dataclass
class DataState:
    symbol: str = "AAPL"
    benchmark_symbol: str = "^GSPC"
    period: str = "5y"
    interval: str = "1d"
    data_source: str = "yfinance"
    cache_dir: str = "data"
    recent_symbols: List[str] = field(default_factory=lambda: ["AAPL", "MSFT", "NVDA"])


@dataclass
class ModelState:
    selected_model: str = "transformer"
    sequence_length: int = 60
    prediction_horizon: int = 5
    target_type: str = "return"          # "close" | "return" | "direction"
    feature_groups: List[str] = field(default_factory=lambda: ["price", "ta", "market"])
    epochs: int = 50
    batch_size: int = 64
    hidden_dim: int = 128
    num_layers: int = 3
    num_heads: int = 4
    dropout: float = 0.2
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    loss_type: str = "huber"             # "mse"|"mae"|"huber"|"quantile"
    optimizer: str = "adamw"            # "adam"|"adamw"|"rmsprop"
    scheduler: str = "onecycle"         # "cosine"|"onecycle"|"plateau"|"none"
    early_stopping_patience: int = 10
    validation_split: float = 0.15
    test_split: float = 0.1
    random_seed: int = 42
    device: str = "auto"                # "auto"|"cpu"|"cuda"
    use_augmentation: bool = True
    use_mixed_precision: bool = True
    mc_dropout_samples: int = 30


@dataclass
class ChartState:
    candlestick: bool = True
    show_prediction: bool = True
    show_confidence_interval: bool = True
    show_signals: bool = True
    dark_mode: bool = True
    overlay_indicators: List[str] = field(default_factory=lambda: ["SMA20", "SMA60"])
    auto_refresh: bool = False


@dataclass
class PipelineState:
    """파이프라인 실행 모드 및 단계별 상태"""
    mode: str = "lightweight"         # "lightweight" | "advanced"
    last_run_symbol: str = ""
    last_run_ts: str = ""
    last_result_path: str = ""
    # 각 단계 상태: "pending" | "running" | "done" | "error" | "skipped"
    step_status: Dict[str, str] = field(
        default_factory=lambda: {s: "pending" for s in PIPELINE_STEPS}
    )
    # Walk-forward 설정
    wf_n_splits: int = 5
    wf_gap: int = 20               # 학습-테스트 경계 격리 영업일
    wf_window: str = "expanding"   # "expanding" | "rolling"
    wf_min_train: int = 252        # 최소 학습 기간 (영업일)
    # Conformal 설정
    use_conformal: bool = True
    conformal_alpha: float = 0.1   # 90% 예측구간
    # Regime 앙상블 설정
    use_regime_ensemble: bool = True
    # Foundation model 설정 (Advanced)
    use_foundation_model: bool = False
    foundation_model: str = "chronos"  # "chronos"|"timesfm"|"timesgpt"


@dataclass
class BacktestState:
    initial_cash: float = 10_000_000.0
    fee: float = 0.00015
    slippage: float = 0.0005
    buy_threshold: float = 0.01
    sell_threshold: float = -0.01
    position_size: float = 1.0          # fraction of cash per trade
    max_position: float = 1.0
    holding_period: int = 0             # 0 = no forced exit
    use_confidence_filter: bool = True
    confidence_min: float = 0.6
    long_only: bool = True


@dataclass
class AppState:
    version: int = SCHEMA_VERSION
    window: WindowState   = field(default_factory=WindowState)
    data: DataState       = field(default_factory=DataState)
    model: ModelState     = field(default_factory=ModelState)
    chart: ChartState     = field(default_factory=ChartState)
    backtest: BacktestState = field(default_factory=BacktestState)
    pipeline: PipelineState = field(default_factory=PipelineState)
    log_level: str = "INFO"
    output_dir: str = "outputs"
    last_model_path: str = ""
    last_export_path: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AppState":
        """누락 키는 기본값으로 채우고, 버전 불일치는 merge"""
        state = cls()
        try:
            w = d.get("window", {})
            for k, v in w.items():
                if hasattr(state.window, k):
                    setattr(state.window, k, v)

            da = d.get("data", {})
            for k, v in da.items():
                if hasattr(state.data, k):
                    setattr(state.data, k, v)

            m = d.get("model", {})
            for k, v in m.items():
                if hasattr(state.model, k):
                    setattr(state.model, k, v)

            c = d.get("chart", {})
            for k, v in c.items():
                if hasattr(state.chart, k):
                    setattr(state.chart, k, v)

            b = d.get("backtest", {})
            for k, v in b.items():
                if hasattr(state.backtest, k):
                    setattr(state.backtest, k, v)

            p = d.get("pipeline", {})
            for k, v in p.items():
                if hasattr(state.pipeline, k):
                    setattr(state.pipeline, k, v)

            for k in ("log_level", "output_dir", "last_model_path", "last_export_path"):
                if k in d:
                    setattr(state, k, d[k])
        except Exception:
            pass
        return state

    def copy(self) -> "AppState":
        return AppState.from_dict(copy.deepcopy(self.to_dict()))
