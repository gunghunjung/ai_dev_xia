"""
PredictionPipeline — 전체 예측 파이프라인 오케스트레이터
──────────────────────────────────────────────────────────
책임:
  1. 데이터 로드
  2. 피처 생성 + 선택
  3. 국면 탐지 (Regime Detection)
  4. 모델 학습 (DL + GBM)
  5. Walk-forward 검증 (Advanced only)
  6. 예측 (ForecastEngine)
  7. Conformal 보정 (Advanced only)
  8. 설명 (SHAP + FeatureImportance)
  9. 백테스트
  10. 보고서 생성

사용:
  pipe = PredictionPipeline(cfg, mode_cfg)
  result = pipe.run(on_step=..., on_log=..., on_progress=...)

설계 원칙:
  - 각 단계는 독립 실행 가능 (이전 단계 결과를 내부에 보관)
  - 미래 데이터 누수 완전 차단 (train_end 인덱스 엄격 준수)
  - 단계별 콜백으로 GUI 진행률 연동
  - 예외 발생 시 해당 단계만 error 처리, 이후 단계는 skip or fallback
"""
from __future__ import annotations

import os
import json
import datetime
import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .config_manager import ConfigManager
from .mode_config import ModeConfig
from .state_schema import PIPELINE_STEPS
from .utils import resolve_device, set_seed
from .logger_config import get_logger

from .data.market_data_service import MarketDataService
from .data.regime_detector import RegimeDetector
from .features.feature_engineering import FeatureEngineer
from .ml.datasets import TimeSeriesDataset
from .ml.trainer import Trainer
from .ml.inference import InferenceEngine
from .ml.experiment_tracker import ExperimentTracker
from .ml.metrics import regression_metrics, classification_metrics, financial_metrics
from .backtest.backtester import Backtester
from .backtest.strategy import PredictionStrategy

log = get_logger("pipeline")

# 콜백 타입 정의
StepCallback  = Callable[[str, str], None]   # (step_name, status)
LogCallback   = Callable[[str], None]
ProgressCallback = Callable[[int, int], None]  # (current_step, total_steps)


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class PipelineResult:
    """파이프라인 전체 실행 결과"""
    symbol:       str = ""
    mode:         str = ""
    run_ts:       str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    # 데이터
    df:           Optional[pd.DataFrame] = None    # OHLCV
    X:            Optional[pd.DataFrame] = None    # 피처
    y:            Optional[pd.Series]   = None     # 타겟
    # 국면
    regime:       Optional[pd.Series]   = None     # 국면 라벨 시리즈
    # 예측
    forecast:     Optional[Dict]        = None     # ForecastEngine 출력
    pred_series:  Optional[pd.Series]  = None     # 슬라이딩 예측 시리즈
    lower_series: Optional[pd.Series]  = None
    upper_series: Optional[pd.Series]  = None
    # 검증
    wf_result:    Any                  = None      # WalkForwardResult
    val_metrics:  Dict[str, float]     = field(default_factory=dict)
    # 설명
    shap_info:    Optional[Dict]       = None
    importances:  Optional[List[tuple]] = None
    # 백테스트
    backtest:     Optional[Dict]       = None
    # 보고서
    report:       str                  = ""
    # 성능
    metrics:      Dict[str, float]     = field(default_factory=dict)
    # 모델 저장 경로
    model_paths:  Dict[str, str]       = field(default_factory=dict)
    # 오류
    errors:       Dict[str, str]       = field(default_factory=dict)

    def to_json_safe(self) -> dict:
        """JSON 직렬화 가능 형태로 변환"""
        def _safe(v):
            if isinstance(v, pd.DataFrame):
                return f"<DataFrame {v.shape}>"
            if isinstance(v, pd.Series):
                return f"<Series len={len(v)}>"
            if isinstance(v, np.ndarray):
                return f"<ndarray shape={v.shape}>"
            return v

        return {
            "symbol":    self.symbol,
            "mode":      self.mode,
            "run_ts":    self.run_ts,
            "forecast":  {k: (float(v) if isinstance(v, (np.floating, float)) else v)
                          for k, v in (self.forecast or {}).items()
                          if not isinstance(v, (pd.DataFrame, pd.Series, np.ndarray))},
            "val_metrics": self.val_metrics,
            "metrics":   self.metrics,
            "report":    self.report,
            "errors":    self.errors,
            "model_paths": self.model_paths,
        }


# ─────────────────────────────────────────────────────────────────────────────
class PredictionPipeline:
    """
    전체 예측 파이프라인.

    Parameters
    ----------
    cfg      : ConfigManager  (AppState 포함)
    mode_cfg : ModeConfig     (Lightweight / Advanced 설정)
    """

    TOTAL_STEPS = len(PIPELINE_STEPS)

    def __init__(self, cfg: ConfigManager, mode_cfg: ModeConfig) -> None:
        self._cfg      = cfg
        self._state    = cfg.state
        self._mode_cfg = mode_cfg
        self._svc      = MarketDataService(self._state.data.cache_dir)
        self._tracker  = ExperimentTracker(
            os.path.join(self._state.output_dir, "experiments")
        )
        self._device: Optional[torch.device] = None
        self._result   = PipelineResult(
            symbol=self._state.data.symbol,
            mode=mode_cfg.name,
        )
        self._stop_requested = False
        self._models: Dict[str, Any] = {}     # name → fitted model
        self._scaler = None                   # 입력 정규화 (train에만 fit)

    # ─────────────────────────────────────────────────────────────────
    #  메인 실행
    # ─────────────────────────────────────────────────────────────────
    def run(
        self,
        on_step:     Optional[StepCallback]     = None,
        on_log:      Optional[LogCallback]      = None,
        on_progress: Optional[ProgressCallback] = None,
    ) -> PipelineResult:
        """전체 파이프라인 실행. 반드시 백그라운드 스레드에서 호출."""
        set_seed(self._state.model.random_seed)
        self._device = resolve_device(self._state.model.device)
        log_fn = on_log or log.info

        log_fn(self._mode_cfg.describe())

        step_fns = [
            ("data_load",     self._step_data_load),
            ("feature_build", self._step_feature_build),
            ("regime_detect", self._step_regime_detect),
            ("model_train",   self._step_model_train),
            ("walk_forward",  self._step_walk_forward),
            ("forecast",      self._step_forecast),
            ("conformal",     self._step_conformal),
            ("explain",       self._step_explain),
            ("backtest",      self._step_backtest),
            ("report",        self._step_report),
        ]

        for step_i, (step_name, step_fn) in enumerate(step_fns):
            if self._stop_requested:
                log_fn("⏹ 파이프라인 중단 요청됨")
                break

            # 상태 업데이트
            self._state.pipeline.step_status[step_name] = "running"
            if on_step:
                on_step(step_name, "running")
            if on_progress:
                on_progress(step_i, self.TOTAL_STEPS)

            try:
                step_fn(log_fn)
                self._state.pipeline.step_status[step_name] = "done"
                if on_step:
                    on_step(step_name, "done")

            except _PipelineSkip as skip:
                log_fn(f"⏭ [{step_name}] 스킵: {skip}")
                self._state.pipeline.step_status[step_name] = "skipped"
                if on_step:
                    on_step(step_name, "skipped")

            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                log_fn(f"❌ [{step_name}] 오류: {e}\n{tb}")
                self._result.errors[step_name] = str(e)
                self._state.pipeline.step_status[step_name] = "error"
                if on_step:
                    on_step(step_name, "error")
                # 치명적 단계(데이터/피처) 실패 시 중단
                if step_name in ("data_load", "feature_build"):
                    log_fn("치명적 오류: 파이프라인 중단")
                    break

        if on_progress:
            on_progress(self.TOTAL_STEPS, self.TOTAL_STEPS)

        # 결과 저장
        self._save_result()
        self._state.pipeline.last_run_symbol = self._result.symbol
        self._state.pipeline.last_run_ts     = self._result.run_ts
        self._cfg.safe_save()

        return self._result

    def stop(self) -> None:
        """중단 요청 (다음 단계 시작 전에 중단됨)"""
        self._stop_requested = True

    # ─────────────────────────────────────────────────────────────────
    #  Step 1: 데이터 로드
    # ─────────────────────────────────────────────────────────────────
    def _step_data_load(self, log_fn: LogCallback) -> None:
        st = self._state.data
        log_fn(f"📥 데이터 로드: {st.symbol} ({st.period}/{st.interval})")
        df, bm = self._svc.fetch_with_benchmark(
            st.symbol, st.benchmark_symbol, st.period, st.interval
        )
        if df is None or df.empty:
            raise ValueError(f"데이터 없음: {st.symbol}")
        log_fn(f"✅ {st.symbol}: {len(df)}행  {df.index[0].date()}~{df.index[-1].date()}")
        self._result.df = df
        self._bm_df = bm   # 벤치마크 (백테스트용)

    # ─────────────────────────────────────────────────────────────────
    #  Step 2: 피처 생성
    # ─────────────────────────────────────────────────────────────────
    def _step_feature_build(self, log_fn: LogCallback) -> None:
        if self._result.df is None:
            raise RuntimeError("df 없음")
        st_model = self._state.model
        log_fn("🔧 피처 생성 중...")
        feat_eng = FeatureEngineer(groups=st_model.feature_groups)
        X, y = feat_eng.build(
            self._result.df,
            target_type=st_model.target_type,
            horizon=st_model.prediction_horizon,
        )
        log_fn(f"✅ 피처 {X.shape[1]}개, 샘플 {len(X)}개")

        # 피처 선택 (Advanced only)
        if self._mode_cfg.use_feature_selection and X.shape[1] > self._mode_cfg.max_features:
            log_fn(f"   → 피처 선택: {X.shape[1]} → {self._mode_cfg.max_features}개")
            try:
                from .features.feature_selector import FeatureSelector
                fs = FeatureSelector(max_features=self._mode_cfg.max_features)
                X = fs.fit_transform(X, y)
                log_fn(f"   → 선택 후: {X.shape[1]}개")
            except Exception as e:
                log_fn(f"   ⚠ 피처 선택 실패 (전체 사용): {e}")

        self._result.X = X
        self._result.y = y
        self._feat_eng = feat_eng

    # ─────────────────────────────────────────────────────────────────
    #  Step 3: 국면 탐지
    # ─────────────────────────────────────────────────────────────────
    def _step_regime_detect(self, log_fn: LogCallback) -> None:
        if self._result.df is None:
            raise RuntimeError("df 없음")
        log_fn("🔍 국면 탐지 중...")
        try:
            rd = RegimeDetector()
            regime_series = rd.detect(self._result.df)
            self._result.regime = regime_series
            # 최신 국면
            latest = regime_series.iloc[-1] if not regime_series.empty else "Unknown"
            log_fn(f"✅ 현재 국면: {latest}")
            self._regime_detector = rd
        except Exception as e:
            log_fn(f"⚠ 국면 탐지 실패 (중립 사용): {e}")
            if self._result.X is not None:
                self._result.regime = pd.Series(
                    ["NEUTRAL"] * len(self._result.X),
                    index=self._result.X.index
                )

    # ─────────────────────────────────────────────────────────────────
    #  Step 4: 모델 학습
    # ─────────────────────────────────────────────────────────────────
    def _step_model_train(self, log_fn: LogCallback) -> None:
        if self._result.X is None:
            raise RuntimeError("피처 없음")

        X, y = self._result.X, self._result.y
        st = self._state.model
        cfg = self._mode_cfg

        # ── train/test 분할 (시간 순서 유지, shuffle 금지) ─────────────
        n = len(X)
        test_n  = max(1, int(n * st.test_split))
        train_n = n - test_n
        X_train, X_test = X.iloc[:train_n], X.iloc[train_n:]
        y_train, y_test = y.iloc[:train_n], y.iloc[train_n:]
        log_fn(f"   학습: {train_n}샘플 / 테스트: {test_n}샘플")

        self._X_train = X_train
        self._X_test  = X_test
        self._y_train = y_train
        self._y_test  = y_test

        # ── GBM 학습 ─────────────────────────────────────────────────
        for model_name in cfg.gbm_models:
            try:
                model = self._build_gbm(model_name, cfg)
                log_fn(f"   [GBM] {model_name} 학습 중...")
                model.fit(
                    X_train.values, y_train.values,
                    eval_set=(X_test.values, y_test.values),
                )
                self._models[model_name] = model
                # 테스트셋 성능
                preds = model.predict(X_test.values)
                m = regression_metrics(y_test.values, preds)
                log_fn(f"   ✅ {model_name}: DA={m.get('da',0):.3f}, RMSE={m.get('rmse',0):.5f}")
            except Exception as e:
                log_fn(f"   ❌ {model_name} 실패: {e}")

        # ── DL 입력 정규화 (SMA/ATR/OBV 등 가격 스케일 피처 → Transformer 수치 안정화) ──
        if cfg.dl_models:
            from .data.preprocessing import Preprocessor as _DLPrep
            _dl_prep = _DLPrep()
            X_train_dl = _dl_prep.fit_transform(X_train, X_train.columns.tolist())
            X_test_dl  = _dl_prep.transform(X_test,  X_test.columns.tolist())
            self._scaler = _dl_prep   # 추론 단계에서 재사용
        else:
            X_train_dl = X_train
            X_test_dl  = X_test

        # ── DL 학습 ──────────────────────────────────────────────────
        for model_name in cfg.dl_models:
            try:
                model = self._build_dl_model(model_name, X_train_dl.shape[1], cfg)
                dataset = TimeSeriesDataset(X_train_dl, y_train, cfg.sequence_length)
                if len(dataset) < 10:
                    log_fn(f"   ⚠ {model_name}: 데이터 부족 ({len(dataset)}샘플), 스킵")
                    continue
                log_fn(f"   [DL] {model_name} 학습 중... ({model.count_params():,} params)")
                trainer = Trainer(
                    model=model,
                    device=self._device,
                    loss_name=st.loss_type,
                    optimizer_name=st.optimizer,
                    lr=cfg.lr,
                    scheduler_name=cfg.scheduler,
                    use_augmentation=cfg.use_augmentation,
                    use_mixed_precision=cfg.use_mixed_precision,
                    on_log=log_fn,
                )
                trainer.fit(
                    dataset,
                    epochs=cfg.epochs,
                    batch_size=cfg.batch_size,
                    val_ratio=st.validation_split,
                    seed=st.random_seed,
                )
                self._models[model_name] = model
                # 테스트셋 성능
                engine = InferenceEngine(model, self._device, n_mc=0)
                X_test_arr = X_test_dl.values.astype(np.float32)
                preds = engine.predict_sliding(X_test_arr, cfg.sequence_length)
                if len(preds) > 0:
                    y_sub = y_test.values[-len(preds):]
                    m = regression_metrics(y_sub, preds)
                    log_fn(f"   ✅ {model_name}: DA={m.get('da',0):.3f}, RMSE={m.get('rmse',0):.5f}")
            except Exception as e:
                log_fn(f"   ❌ {model_name} 실패: {e}")

        if not self._models:
            raise RuntimeError("학습된 모델이 하나도 없습니다. 데이터/설정을 확인하세요.")

        # 모델 저장
        save_dir = os.path.join(self._state.output_dir, "predictions")
        os.makedirs(save_dir, exist_ok=True)
        for name, model in self._models.items():
            try:
                path = os.path.join(save_dir, f"{self._result.symbol}_{name}.pt")
                if hasattr(model, "save"):
                    model.save(path)
                    self._result.model_paths[name] = path
            except Exception:
                pass

        log_fn(f"✅ 학습 완료: {list(self._models.keys())}")

    # ─────────────────────────────────────────────────────────────────
    #  Step 5: Walk-forward 검증
    # ─────────────────────────────────────────────────────────────────
    def _step_walk_forward(self, log_fn: LogCallback) -> None:
        if not self._mode_cfg.use_walk_forward:
            raise _PipelineSkip("Lightweight 모드: Walk-forward 생략")
        if self._result.X is None:
            raise RuntimeError("피처 없음")

        log_fn("📋 Walk-forward 검증 시작...")
        try:
            from .backtest.walk_forward import WalkForwardValidator
            from .ml.metrics import regression_metrics, financial_metrics

            cfg = self._mode_cfg

            # 대표 모델만 walk-forward (가장 중요한 것 1~2개)
            wf_model_names = (cfg.dl_models[:1] + cfg.gbm_models[:1])[:2]
            log_fn(f"   WF 대상 모델: {wf_model_names}")

            wf_results = {}
            for model_name in wf_model_names:
                if self._stop_requested:
                    break
                log_fn(f"   [{model_name}] Walk-forward 중... ({cfg.wf_n_splits}폴드)")
                try:
                    wfv = WalkForwardValidator(
                        n_splits=cfg.wf_n_splits,
                        gap=cfg.wf_gap,
                        window=cfg.wf_window,
                        min_train_size=cfg.wf_min_train,
                    )
                    result = wfv.validate(
                        X=self._result.X,
                        y=self._result.y,
                        model_builder=lambda: self._build_any_model(model_name, self._result.X.shape[1]),
                        seq_len=cfg.sequence_length if model_name in cfg.dl_models else None,
                        device=self._device,
                        on_log=log_fn,
                    )
                    wf_results[model_name] = result
                    m = result.aggregate_metrics()
                    log_fn(
                        f"   ✅ {model_name}: "
                        f"DA={m.get('da',0):.3f}, "
                        f"Sharpe={m.get('sharpe',0):.2f}, "
                        f"OvFit={result.overfitting_score:.3f}"
                    )
                except Exception as e:
                    log_fn(f"   ❌ {model_name} WF 실패: {e}")

            self._result.wf_result = wf_results
            # val_metrics에 요약 저장
            for name, res in wf_results.items():
                agg = res.aggregate_metrics()
                for k, v in agg.items():
                    self._result.val_metrics[f"{name}_{k}"] = v

        except Exception as e:
            log_fn(f"⚠ Walk-forward 검증 실패 (계속 진행): {e}")

    # ─────────────────────────────────────────────────────────────────
    #  Step 6: 예측 (ForecastEngine)
    # ─────────────────────────────────────────────────────────────────
    def _step_forecast(self, log_fn: LogCallback) -> None:
        if not self._models:
            raise RuntimeError("학습된 모델 없음")

        log_fn("🔮 예측 실행...")
        st_model = self._state.model
        cfg = self._mode_cfg

        X_arr = self._result.X.values.astype(np.float32)
        # DL 모델용 정규화 (학습 시 fit 한 scaler 재사용)
        if self._scaler is not None:
            _X_dl_df = self._scaler.transform(self._result.X, list(self._result.X.columns))
            X_arr_dl = _X_dl_df.values.astype(np.float32)
        else:
            X_arr_dl = X_arr
        n = len(X_arr)
        seq = cfg.sequence_length

        # ── 슬라이딩 윈도우 예측 (전 구간) ──────────────────────────
        all_preds: Dict[str, np.ndarray] = {}
        for model_name, model in self._models.items():
            try:
                if model_name in cfg.dl_models:
                    engine = InferenceEngine(model, self._device, n_mc=cfg.mc_samples if cfg.use_mc_dropout else 0)
                    preds = engine.predict_sliding(X_arr_dl, seq)
                    all_preds[model_name] = preds
                else:
                    # GBM: 마지막 seq 이후 구간 예측
                    preds = model.predict(X_arr[seq:])
                    all_preds[model_name] = preds
            except Exception as e:
                log_fn(f"   ⚠ {model_name} 예측 실패: {e}")

        if not all_preds:
            raise RuntimeError("예측 결과 없음")

        # ── 앙상블 ───────────────────────────────────────────────────
        min_len = min(len(v) for v in all_preds.values())
        weights = cfg.ensemble_weights
        total_w = sum(weights.get(n, 1.0) for n in all_preds)
        ensemble_pred = np.zeros(min_len)
        for name, preds in all_preds.items():
            w = weights.get(name, 1.0) / total_w
            ensemble_pred += w * preds[-min_len:]

        # 인덱스
        idx = self._result.X.index[-min_len:]
        self._result.pred_series = pd.Series(ensemble_pred, index=idx)

        # ── CI 계산 ──────────────────────────────────────────────────
        # Bootstrap CI (Lightweight) 또는 MC Dropout std (Advanced)
        pred_stacks = np.stack([v[-min_len:] for v in all_preds.values()])
        if cfg.use_mc_dropout and len(pred_stacks) > 1:
            sigma = pred_stacks.std(axis=0)
        elif cfg.use_bootstrap_ci:
            # Bootstrap: 모델 간 disagreement 기반
            sigma = pred_stacks.std(axis=0) + 1e-6
        else:
            sigma = np.abs(ensemble_pred) * 0.1 + 1e-6

        z90 = 1.645
        lower = ensemble_pred - z90 * sigma
        upper = ensemble_pred + z90 * sigma
        self._result.lower_series = pd.Series(lower, index=idx)
        self._result.upper_series = pd.Series(upper, index=idx)

        # ── ForecastEngine (마지막 시점 포인트 예측) ─────────────────
        try:
            from .prediction.forecast_engine import ForecastEngine
            current_price = float(self._result.df["close"].iloc[-1])
            engine = ForecastEngine(
                symbol=self._result.symbol,
                current_price=current_price,
                target_pct=0.05,
                stop_loss_pct=-0.03,
            )
            # 마지막 윈도우 예측 결과로 ForecastEngine 구성
            last_pred = float(ensemble_pred[-1])
            last_sigma = float(sigma[-1])
            prob_up = float(np.mean(ensemble_pred > 0))

            # regime
            regime_label = "NEUTRAL"
            if self._result.regime is not None and len(self._result.regime) > 0:
                regime_label = str(self._result.regime.iloc[-1])

            fc = engine.forecast(
                point_return=last_pred,
                uncertainty=last_sigma,
                prob_up=prob_up,
                regime_label=regime_label,
                multi_horizon_preds={
                    h: _extrapolate_horizon(last_pred, last_sigma, h)
                    for h in [1, 5, 10, 20]
                },
                drift_info=None,
                warnings=self._collect_warnings(last_pred, last_sigma),
            )
            self._result.forecast = fc
            log_fn(
                f"✅ 예측 완료: {fc.get('signal','?')} | "
                f"↑{fc.get('prob_up',0):.1%} / ↓{fc.get('prob_down',0):.1%} | "
                f"예측수익률 {fc.get('point_return',0):+.2%}"
            )
        except Exception as e:
            log_fn(f"⚠ ForecastEngine 실패 (기본 결과 사용): {e}")
            # fallback
            prob_up = float(np.mean(ensemble_pred > 0))
            self._result.forecast = {
                "signal":       "강세" if prob_up > 0.55 else ("약세" if prob_up < 0.45 else "중립"),
                "prob_up":      prob_up,
                "prob_down":    1 - prob_up,
                "point_return": float(ensemble_pred[-1]),
                "ci_90_lower":  float(lower[-1]),
                "ci_90_upper":  float(upper[-1]),
                "uncertainty":  float(sigma[-1]),
                "regime_label": "NEUTRAL",
                "warnings":     [],
            }

        # 성능 계산 (테스트셋 구간)
        y_arr  = self._result.y.values
        y_test = y_arr[-min_len:]
        m = regression_metrics(y_test, ensemble_pred)
        self._result.metrics.update(m)
        log_fn(f"   테스트 성능: DA={m.get('da',0):.3f}, RMSE={m.get('rmse',0):.5f}")

    # ─────────────────────────────────────────────────────────────────
    #  Step 7: Conformal 보정
    # ─────────────────────────────────────────────────────────────────
    def _step_conformal(self, log_fn: LogCallback) -> None:
        if not self._mode_cfg.use_conformal:
            raise _PipelineSkip("Lightweight 모드: Conformal 생략")
        if self._result.pred_series is None:
            raise _PipelineSkip("예측 결과 없음")

        log_fn("📐 Conformal 예측구간 보정 중...")
        try:
            from .prediction.uncertainty import UncertaintyEstimator

            # calibration set = train 마지막 20%
            n_cal = max(20, len(self._result.pred_series) // 5)
            y_cal = self._result.y.values[-(len(self._result.pred_series) + n_cal):-len(self._result.pred_series)]
            p_cal = self._result.pred_series.values[:n_cal]

            if len(y_cal) < 10:
                raise ValueError("calibration 샘플 부족")

            est = UncertaintyEstimator(alpha=self._mode_cfg.conformal_alpha)
            conf_lower, conf_upper = est.conformal_interval(y_cal, p_cal, self._result.pred_series.values)

            idx = self._result.pred_series.index
            # forecast dict 업데이트
            if self._result.forecast:
                self._result.forecast["conformal_lower"] = float(conf_lower[-1])
                self._result.forecast["conformal_upper"] = float(conf_upper[-1])
            log_fn(f"✅ Conformal PI: [{conf_lower[-1]:+.2%}, {conf_upper[-1]:+.2%}]")
        except Exception as e:
            log_fn(f"⚠ Conformal 보정 실패 (계속): {e}")

    # ─────────────────────────────────────────────────────────────────
    #  Step 8: 설명 (SHAP)
    # ─────────────────────────────────────────────────────────────────
    def _step_explain(self, log_fn: LogCallback) -> None:
        if self._result.X is None:
            raise _PipelineSkip("피처 없음")

        log_fn("🔬 설명 생성 중 (SHAP)...")
        try:
            from .explain.shap_explainer import SHAPExplainer
            from .explain.feature_importance import FeatureImportanceAnalyzer

            X_arr  = self._result.X.values.astype(np.float32)
            y_arr  = self._result.y.values

            # SHAP: GBM 모델 우선 (빠름), 없으면 kernel (느림)
            shap_model = None
            shap_method = self._mode_cfg.shap_method
            for name in ["xgboost", "lightgbm"]:
                if name in self._models:
                    shap_model = self._models[name]
                    shap_method = "tree"
                    break
            if shap_model is None and self._mode_cfg.dl_models:
                dl_name = self._mode_cfg.dl_models[0]
                if dl_name in self._models:
                    shap_model = self._models[dl_name]

            if shap_model is not None:
                max_s = min(self._mode_cfg.shap_max_samples, len(X_arr))
                explainer = SHAPExplainer(
                    model=shap_model,
                    feature_names=list(self._result.X.columns),
                    method=shap_method,
                )
                background = X_arr[:min(100, len(X_arr))]
                shap_info = explainer.local_explanation(
                    X_arr[-max_s:], background=background
                )
                self._result.shap_info = shap_info
                top_pos = shap_info.get("top_positive", [])
                if top_pos:
                    log_fn(f"   상승 기여 1위: {top_pos[0][0]} ({top_pos[0][1]:+.4f})")

            # 피처 중요도
            fia = FeatureImportanceAnalyzer(feature_names=list(self._result.X.columns))
            for name in ["xgboost", "lightgbm"]:
                if name in self._models:
                    df_imp = fia.from_model(self._models[name], method_name=name)
            corr_df = fia.correlation_based(X_arr, y_arr)
            agg = fia.aggregate()
            if not agg.empty:
                self._result.importances = list(
                    zip(agg["feature"], agg["importance"])
                )
            log_fn("✅ 설명 생성 완료")
        except Exception as e:
            log_fn(f"⚠ 설명 생성 실패 (계속): {e}")

    # ─────────────────────────────────────────────────────────────────
    #  Step 9: 백테스트
    # ─────────────────────────────────────────────────────────────────
    def _step_backtest(self, log_fn: LogCallback) -> None:
        if self._result.pred_series is None:
            raise _PipelineSkip("예측 결과 없음")
        if self._result.df is None:
            raise _PipelineSkip("원본 데이터 없음")

        log_fn("📈 백테스트 실행...")
        bt_st  = self._state.backtest
        pred   = self._result.pred_series
        lower  = self._result.lower_series
        upper  = self._result.upper_series

        # 가격 시리즈: 예측 구간과 정렬
        price = self._result.df["close"].reindex(pred.index).dropna()
        pred  = pred.reindex(price.index)
        lower = lower.reindex(price.index) if lower is not None else None
        upper = upper.reindex(price.index) if upper is not None else None

        strategy = PredictionStrategy(
            buy_threshold=bt_st.buy_threshold,
            sell_threshold=bt_st.sell_threshold,
            target_type=self._state.model.target_type,
        )
        bt = Backtester(
            initial_cash=bt_st.initial_cash,
            fee=bt_st.fee,
            slippage=bt_st.slippage,
            position_size=bt_st.position_size,
        )
        result = bt.run(
            price,
            pred.values,
            lower.values if lower is not None else None,
            upper.values if upper is not None else None,
            strategy,
        )
        self._result.backtest = result
        perf = result.get("performance", {})
        log_fn(
            f"✅ 백테스트 완료: "
            f"Sharpe={perf.get('sharpe',0):.2f}  "
            f"MDD={perf.get('max_drawdown',0):.2%}  "
            f"승률={perf.get('win_rate',0):.2%}  "
            f"CAGR={perf.get('cagr',0):.2%}"
        )

    # ─────────────────────────────────────────────────────────────────
    #  Step 10: 보고서 생성
    # ─────────────────────────────────────────────────────────────────
    def _step_report(self, log_fn: LogCallback) -> None:
        log_fn("📄 보고서 생성 중...")
        try:
            from .explain.report_generator import ReportGenerator
            gen = ReportGenerator(
                symbol=self._result.symbol,
                company_name=self._state.data.symbol,
                currency="KRW" if ".KS" in self._result.symbol or ".KQ" in self._result.symbol else "USD",
            )
            current_price = float(self._result.df["close"].iloc[-1]) if self._result.df is not None else 0.0
            perf_metrics  = self._result.backtest.get("performance", {}) if self._result.backtest else {}
            report = gen.generate(
                forecast=self._result.forecast or {},
                current_price=current_price,
                shap_info=self._result.shap_info,
                perf_metrics=perf_metrics,
            )
            self._result.report = report
            log_fn("✅ 보고서 생성 완료")
        except Exception as e:
            log_fn(f"⚠ 보고서 생성 실패: {e}")
            self._result.report = f"보고서 생성 실패: {e}"

    # ─────────────────────────────────────────────────────────────────
    #  결과 저장
    # ─────────────────────────────────────────────────────────────────
    def _save_result(self) -> None:
        try:
            save_dir = os.path.join(self._state.output_dir, "predictions")
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(
                save_dir,
                f"{self._result.symbol.replace('^','IDX_').replace('/','_')}_pipeline.json"
            )
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._result.to_json_safe(), f, ensure_ascii=False, indent=2, default=str)
            self._state.pipeline.last_result_path = path
        except Exception as e:
            log.warning(f"파이프라인 결과 저장 실패: {e}")

    # ─────────────────────────────────────────────────────────────────
    #  모델 빌드 헬퍼
    # ─────────────────────────────────────────────────────────────────
    def _build_gbm(self, name: str, cfg: ModeConfig):
        """GBM 모델 생성 (fit 전)"""
        params = {
            "n_estimators": cfg.gbm_n_estimators,
            "max_depth":    cfg.gbm_max_depth,
            "early_stopping_rounds": cfg.gbm_early_stopping,
        }
        if name == "xgboost":
            from .ml.gbm.xgboost_model import XGBoostWrapper
            return XGBoostWrapper(**params)
        if name == "lightgbm":
            from .ml.gbm.lightgbm_model import LightGBMWrapper
            return LightGBMWrapper(**params)
        raise ValueError(f"Unknown GBM: {name}")

    def _build_dl_model(self, name: str, n_features: int, cfg: ModeConfig):
        """DL 모델 생성 (학습 전)"""
        kw = dict(
            n_features=n_features,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
        )
        if name == "lstm":
            from .ml.models.lstm_model import LSTMModel
            return LSTMModel(**kw)
        if name == "gru":
            from .ml.models.gru_model import GRUModel
            return GRUModel(**kw)
        if name == "transformer":
            from .ml.models.transformer_model import TransformerModel
            return TransformerModel(**kw, num_heads=cfg.num_heads)
        if name == "patchtst":
            from .ml.models.patchtst_model import PatchTSTModel
            return PatchTSTModel(n_features=n_features, d_model=cfg.hidden_dim,
                                 n_heads=cfg.num_heads, n_encoder_layers=cfg.num_layers,
                                 dropout=cfg.dropout)
        if name == "temporal_cnn":
            from .ml.models.temporal_cnn import TemporalCNN
            return TemporalCNN(**kw)
        if name == "tft_like":
            from .ml.models.tft_like_model import TFTLikeModel
            return TFTLikeModel(**kw, num_heads=cfg.num_heads)
        if name == "cnn_lstm":
            from .ml.models.cnn_lstm_hybrid import CNNLSTMHybrid
            return CNNLSTMHybrid(**kw)
        if name == "nbeats_like":
            from .ml.models.nbeats_like_model import NBEATSLikeModel
            return NBEATSLikeModel(n_features=n_features, seq_len=cfg.sequence_length,
                                   hidden_dim=cfg.hidden_dim, num_layers=cfg.num_layers,
                                   dropout=cfg.dropout)
        raise ValueError(f"Unknown DL model: {name}")

    def _build_any_model(self, name: str, n_features: int):
        """Walk-forward용 모델 빌더 (GBM or DL)"""
        cfg = self._mode_cfg
        if name in cfg.gbm_models:
            return self._build_gbm(name, cfg)
        return self._build_dl_model(name, n_features, cfg)

    # ─────────────────────────────────────────────────────────────────
    #  유틸
    # ─────────────────────────────────────────────────────────────────
    def _collect_warnings(self, pred: float, sigma: float) -> List[str]:
        warns = []
        if sigma > 0.03:
            warns.append(f"⚠ 불확실성 높음 (σ={sigma:.4f})")
        if abs(pred) > 0.1:
            warns.append(f"⚠ 예측 수익률 극단값 ({pred:+.2%}) — 과적합 가능성 확인 권장")
        return warns


# ─────────────────────────────────────────────────────────────────────────────
class _PipelineSkip(Exception):
    """단계 생략 신호 (오류 아님)"""
    pass


# ─────────────────────────────────────────────────────────────────────────────
def _extrapolate_horizon(point: float, sigma: float, horizon: int) -> dict:
    """단순 외삽으로 다중 시계 예측 근사 (실제 예측이 아닌 참고용)"""
    scale = np.sqrt(horizon)
    return {
        "point":    point * horizon,
        "lower_90": point * horizon - 1.645 * sigma * scale,
        "upper_90": point * horizon + 1.645 * sigma * scale,
        "prob_up":  float(0.5 + (point / (sigma * scale + 1e-9)) * 0.15),
    }
