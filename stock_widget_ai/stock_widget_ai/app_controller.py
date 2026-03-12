"""
AppController — 데이터/ML/백테스트 파이프라인을 GUI와 연결하는 핵심 컨트롤러
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import torch
from typing import Optional, Dict, Callable

from .state_schema import AppState
from .config_manager import ConfigManager
from .utils import resolve_device, set_seed, ensure_dirs
from .logger_config import get_logger
from .data.market_data_service import MarketDataService
from .features.feature_engineering import FeatureEngineer
from .ml.datasets import TimeSeriesDataset
from .ml.trainer import Trainer
from .ml.inference import InferenceEngine
from .ml.experiment_tracker import ExperimentTracker
from .backtest.backtester import Backtester
from .backtest.strategy import PredictionStrategy

log = get_logger("controller")


def _build_model(name: str, n_features: int, seq_len: int, params: dict):
    from .ml.models.lstm_model       import LSTMModel
    from .ml.models.gru_model        import GRUModel
    from .ml.models.temporal_cnn     import TemporalCNN
    from .ml.models.transformer_model import TransformerModel
    from .ml.models.tft_like_model   import TFTLikeModel
    from .ml.models.nbeats_like_model import NBEATSLikeModel
    from .ml.models.cnn_lstm_hybrid  import CNNLSTMHybrid

    kw = dict(
        n_features=n_features,
        hidden_dim=params.get("hidden_dim", 128),
        num_layers=params.get("num_layers", 2),
        dropout=params.get("dropout", 0.2),
    )
    mapping = {
        "lstm":         lambda: LSTMModel(**kw),
        "gru":          lambda: GRUModel(**kw),
        "temporal_cnn": lambda: TemporalCNN(**kw),
        "transformer":  lambda: TransformerModel(**kw, num_heads=params.get("num_heads", 4)),
        "tft_like":     lambda: TFTLikeModel(**kw, num_heads=params.get("num_heads", 4)),
        "nbeats_like":  lambda: NBEATSLikeModel(
                            n_features=n_features, seq_len=seq_len, **{k: v for k, v in kw.items() if k != "n_features"}),
        "cnn_lstm":     lambda: CNNLSTMHybrid(**kw),
    }
    if name not in mapping:
        log.warning(f"Unknown model '{name}', fallback to lstm")
        name = "lstm"
    return mapping[name]()


class AppController:
    def __init__(self, cfg: ConfigManager) -> None:
        self._cfg    = cfg
        self._state  = cfg.state
        self._svc    = MarketDataService(self._state.data.cache_dir)
        self._tracker = ExperimentTracker(
            os.path.join(self._state.output_dir, "experiments")
        )
        self._df:    Optional[pd.DataFrame] = None
        self._X:     Optional[pd.DataFrame] = None
        self._y:     Optional[pd.Series]    = None
        self._feat:  Optional[FeatureEngineer] = None
        self._scaler = None   # DL 모델용 RobustScaler (train에서 fit, predict에서 transform)
        self._n_train: int = 0  # trainer의 temporal split 경계 (predict에서 test-only 메트릭에 사용)
        self._model  = None
        self._device: Optional[torch.device] = None
        self._pred_result: Optional[Dict]    = None
        self._bt_result:   Optional[Dict]    = None
        self._current_trainer: Optional[Trainer] = None
        ensure_dirs(self._state.output_dir,
                    os.path.join(self._state.output_dir, "predictions"),
                    os.path.join(self._state.output_dir, "backtests"))

    # ── 데이터 로드 ──────────────────────────────────────────────
    def load_data(
        self,
        symbol: str,
        benchmark: str,
        period: str,
        interval: str,
        force: bool = False,
        on_log: Optional[Callable[[str], None]] = None,
    ) -> pd.DataFrame:
        log_fn = on_log or log.info
        log_fn(f"데이터 로드: {symbol} ({period}/{interval})")
        self._df = self._svc.fetch(symbol, benchmark, period, interval, force)
        log_fn(f"완료: {len(self._df)}행 로드")
        return self._df

    # ── 피처 생성 ─────────────────────────────────────────────────
    def build_features(
        self,
        feature_groups=None,
        target_type: str = "return",
        horizon: int = 5,
        on_log=None,
    ):
        if self._df is None:
            raise RuntimeError("데이터를 먼저 로드하세요")
        log_fn = on_log or log.info
        log_fn("피처 생성 중...")
        self._feat = FeatureEngineer(groups=feature_groups)
        self._X, self._y = self._feat.build(self._df, target_type, horizon)
        log_fn(f"피처 {self._X.shape[1]}개, 샘플 {len(self._X)}개")
        return self._X, self._y   # 튜플 반환 — GUI의 _done 콜백에서 (X, y)로 언팩

    # ── 학습 ──────────────────────────────────────────────────────
    def train(
        self,
        params: dict,
        on_log=None,
        on_progress=None,
        on_epoch_end=None,
    ) -> Dict:
        if self._X is None:
            raise RuntimeError("피처를 먼저 생성하세요")

        log_fn   = on_log or log.info
        set_seed(self._state.model.random_seed)
        self._device = resolve_device(params.get("device", "auto"))

        seq_len = self._state.model.sequence_length

        # DL 모델 입력 정규화 — SMA/EMA/ATR/OBV 등 가격 스케일 피처를 RobustScaler로 정규화
        # (미정규화 입력은 Transformer LayerNorm에서 NaN → 학습 불가 원인)
        from .data.preprocessing import Preprocessor as _Prep
        _prep = _Prep()
        X_scaled = _prep.fit_transform(self._X, self._X.columns.tolist())
        self._scaler = _prep

        dataset = TimeSeriesDataset(X_scaled, self._y, seq_len)
        if len(dataset) < 10:
            raise ValueError(f"데이터 부족: {len(dataset)}샘플 (seq_len={seq_len})")

        model_name = params.get("model", "lstm")
        self._model = _build_model(
            model_name, dataset.n_features, seq_len, params
        ).to(self._device)
        log_fn(f"{model_name} 모델 파라미터: {self._model.count_params():,}")

        self._tracker.start(
            self._state.data.symbol, model_name, params
        )

        # epoch_end 콜백으로 progress 계산
        total_ep = params.get("epochs", 50)
        def _epoch_end(ep, tr, vl):
            pct = int(ep / total_ep * 100)
            if on_progress:
                on_progress(pct)
            if on_epoch_end:
                on_epoch_end(ep, tr, vl)

        trainer = Trainer(
            model=self._model,
            device=self._device,
            loss_name=params.get("loss", "huber"),
            optimizer_name=params.get("optimizer", "adamw"),
            lr=params.get("lr", 5e-4),
            scheduler_name=params.get("scheduler", "onecycle"),
            use_augmentation=params.get("augmentation", True),
            on_epoch_end=_epoch_end,
            on_log=log_fn,
        )
        self._current_trainer = trainer

        history = trainer.fit(
            dataset,
            epochs=total_ep,
            batch_size=params.get("batch_size", 64),
            val_ratio=self._state.model.validation_split,
            seed=self._state.model.random_seed,
        )
        self._tracker.log_training_history(history)
        # temporal split 경계 저장 (seq_len 오프셋 포함)
        self._n_train = seq_len + getattr(trainer, "n_train", 0)

        # 모델 저장
        model_path = os.path.join(
            self._state.output_dir, "predictions",
            f"{self._state.data.symbol}_{model_name}.pt"
        )
        self._model.save(model_path)
        self._state.last_model_path = model_path
        self._tracker.finish(model_path)
        self._cfg.safe_save()
        log_fn(f"모델 저장: {model_path}")
        return history

    def stop_training(self) -> None:
        if self._current_trainer:
            self._current_trainer.stop()

    # ── 예측 ──────────────────────────────────────────────────────
    def predict(self, on_log=None) -> Dict:
        if self._model is None:
            raise RuntimeError("먼저 학습하거나 모델을 불러오세요")
        if self._X is None:
            raise RuntimeError("피처를 먼저 생성하세요")

        log_fn = on_log or log.info
        log_fn("예측 실행...")

        seq_len = self._state.model.sequence_length
        mc      = self._state.model.mc_dropout_samples

        engine = InferenceEngine(self._model, self._device or torch.device("cpu"), mc)

        # 학습 시 fit한 스케일러로 동일하게 정규화 (데이터 누수 없음)
        if self._scaler is not None:
            _X_scaled = self._scaler.transform(self._X, self._X.columns.tolist())
        else:
            _X_scaled = self._X
        X_arr = _X_scaled.values.astype(np.float32)
        n = len(X_arr)
        preds, lowers, uppers = [], [], []
        for i in range(seq_len, n):
            window = X_arr[i - seq_len: i]
            out = engine.predict_window(window)
            preds.append(out["pred"])
            lowers.append(out["lower"])
            uppers.append(out["upper"])

        pred_arr  = np.array(preds)
        lower_arr = np.array(lowers)
        upper_arr = np.array(uppers)
        y_arr     = self._y.values[seq_len:]
        idx       = self._X.index[seq_len:]

        from .ml.metrics import regression_metrics, classification_metrics
        _metric_fn = (classification_metrics
                      if self._state.model.target_type == "direction"
                      else regression_metrics)

        # ── 전체 & 테스트 전용 메트릭 ────────────────────────────
        m_full = _metric_fn(y_arr, pred_arr)

        # temporal split 경계 이후가 테스트셋
        # self._n_train = seq_len + n_train_samples (trainer에서 저장)
        test_start = self._n_train - seq_len   # pred_arr 기준 offset
        if 0 < test_start < len(pred_arr):
            m_test = _metric_fn(y_arr[test_start:], pred_arr[test_start:])
            n_test = len(pred_arr) - test_start
        else:
            m_test = m_full
            n_test = len(pred_arr)

        # GUI 메트릭으로는 테스트셋 기준 사용 (더 정직한 수치)
        m = m_test
        self._tracker.log_metrics("test", m_test)

        # ── 콘솔 상세 로그 ────────────────────────────────────────
        log_fn("=" * 60)
        log_fn(f"[PREDICT] symbol={self._state.data.symbol}  "
               f"target={self._state.model.target_type}  "
               f"horizon={self._state.model.prediction_horizon}d")
        log_fn(f"[PREDICT] 전체 예측 샘플: {len(pred_arr)}  "
               f"(train: {test_start}, test: {n_test})")
        log_fn(f"[PREDICT] 전체 메트릭 (train+test, 참고용): "
               f"{ {k: round(float(v), 4) for k, v in m_full.items()} }")
        log_fn(f"[PREDICT] ★테스트 메트릭 (test only, {n_test}샘플): "
               f"{ {k: round(float(v), 4) for k, v in m_test.items()} }")
        log_fn(f"[PREDICT] actual  mean={float(y_arr.mean()):+.6f}  "
               f"std={float(y_arr.std()):.6f}  "
               f"min={float(y_arr.min()):+.6f}  max={float(y_arr.max()):+.6f}")
        log_fn(f"[PREDICT] pred    mean={float(pred_arr.mean()):+.6f}  "
               f"std={float(pred_arr.std()):.6f}  "
               f"min={float(pred_arr.min()):+.6f}  max={float(pred_arr.max()):+.6f}")
        # 마지막 5개 샘플 출력
        n_tail = min(5, len(pred_arr))
        log_fn(f"[PREDICT] 최근 {n_tail}개 (날짜 / actual / pred / CI):")
        for i in range(-n_tail, 0):
            log_fn(f"  {str(idx[i])[:10]}  "
                   f"actual={float(y_arr[i]):+.6f}  "
                   f"pred={float(pred_arr[i]):+.6f}  "
                   f"CI=[{float(lower_arr[i]):+.6f}, {float(upper_arr[i]):+.6f}]")
        log_fn("=" * 60)

        # ── 미래 1스텝 예측 (차트에 붉은 점선으로 표시) ─────────────
        future = self._calc_future_forecast(X_arr, seq_len, engine, log_fn)

        self._pred_result = {
            "dates": idx, "actual": y_arr,
            "pred": pred_arr, "lower": lower_arr, "upper": upper_arr,
            "metrics": m,
            "future": future,   # 미래 예측 (None이면 표시 안함)
        }
        return self._pred_result

    def _calc_future_forecast(self, X_arr, seq_len, engine, log_fn) -> Optional[Dict]:
        """마지막 seq_len 윈도우 → 다음 horizon일 후 가격 예측"""
        try:
            if len(X_arr) < seq_len or self._df is None:
                return None

            window     = X_arr[-seq_len:]
            out        = engine.predict_window(window)
            last_date  = self._X.index[-1]
            last_close = float(self._df["close"].iloc[-1])
            horizon    = self._state.model.prediction_horizon
            ttype      = self._state.model.target_type

            # pred → 가격으로 변환
            raw_pred  = float(out["pred"])
            raw_lower = float(out["lower"])
            raw_upper = float(out["upper"])

            if ttype == "return":
                pred_price  = last_close * (1.0 + raw_pred)
                lower_price = last_close * (1.0 + raw_lower)
                upper_price = last_close * (1.0 + raw_upper)
                pred_return = raw_pred
            elif ttype == "close":
                pred_price  = raw_pred
                lower_price = raw_lower
                upper_price = raw_upper
                pred_return = (pred_price - last_close) / max(last_close, 1e-9)
            else:   # direction: 확률 → 기대 수익률 ≈ (p-0.5)*4%
                pred_return = (raw_pred - 0.5) * 0.04
                pred_price  = last_close * (1.0 + pred_return)
                lower_price = last_close * (1.0 - 0.02)
                upper_price = last_close * (1.0 + 0.02)

            # 예측 목표일 (영업일 기준)
            target_date = pd.bdate_range(start=last_date, periods=horizon + 1)[-1]

            log_fn("─" * 60)
            log_fn(f"[FUTURE FORECAST] 기준일={str(last_date)[:10]}  "
                   f"예측일={str(target_date)[:10]}  ({horizon}거래일 후)")
            log_fn(f"[FUTURE FORECAST] 현재가={last_close:,.0f}원  "
                   f"예측가={pred_price:,.0f}원  "
                   f"예상수익률={pred_return:+.2%}")
            log_fn(f"[FUTURE FORECAST] 95%CI=[{lower_price:,.0f}, {upper_price:,.0f}]원")
            log_fn("─" * 60)

            return {
                "last_date":   last_date,
                "target_date": target_date,
                "last_close":  last_close,
                "pred_price":  pred_price,
                "lower_price": lower_price,
                "upper_price": upper_price,
                "pred_return": pred_return,
                "target_type": ttype,
                "horizon":     horizon,
            }
        except Exception as e:
            log_fn(f"[FUTURE FORECAST] 오류: {e}")
            return None

    # ── 백테스트 ─────────────────────────────────────────────────
    def backtest(self, bt_params: dict, on_log=None) -> Dict:
        if self._pred_result is None:
            raise RuntimeError("예측을 먼저 실행하세요")
        if self._df is None:
            raise RuntimeError("데이터가 없습니다")

        log_fn = on_log or log.info
        log_fn("백테스트 실행...")

        strategy = PredictionStrategy(
            buy_threshold=bt_params.get("buy_threshold", 0.01),
            sell_threshold=bt_params.get("sell_threshold", -0.01),
            target_type=self._state.model.target_type,
        )
        bt = Backtester(
            initial_cash=bt_params.get("initial_cash", 10_000_000),
            fee=bt_params.get("fee", 0.00015),
            slippage=bt_params.get("slippage", 0.0005),
            position_size=bt_params.get("position_size", 1.0),
        )

        seq_len    = self._state.model.sequence_length
        test_only  = bt_params.get("test_only", True)
        pred_full  = self._pred_result["pred"]
        lower_full = self._pred_result["lower"]
        upper_full = self._pred_result["upper"]

        # ── pred_result의 날짜 인덱스 — 날짜 기반 필터링에 사용 ──
        pred_dates = pd.DatetimeIndex(self._pred_result["dates"])
        # price를 pred_dates 기준으로 정렬 (X와 df 길이 차이 보정)
        price_aligned = self._df["close"].reindex(pred_dates)
        # reindex로 NaN이 생긴 날짜는 가장 가까운 유효 값으로 채움
        price_aligned = price_aligned.ffill().bfill()

        # ── 테스트 기간만 vs 전체 기간 선택 ───────────────────────
        test_offset = self._n_train - seq_len  # pred_arr 기준 오프셋
        if test_only and self._n_train > 0 and 0 < test_offset < len(pred_full):
            pred_use     = pred_full[test_offset:]
            lower_use    = lower_full[test_offset:]
            upper_use    = upper_full[test_offset:]
            dates_use    = pred_dates[test_offset:]
            price_use    = price_aligned.iloc[test_offset:]
            mode_base    = "테스트 기간 (out-of-sample)"
            log_fn(f"[BACKTEST] 테스트 기간 선택: {len(pred_use)}일 "
                   f"(전체 {len(pred_full)}일 중 마지막 {len(pred_use)}일)")
        else:
            pred_use  = pred_full
            lower_use = lower_full
            upper_use = upper_full
            dates_use = pred_dates
            price_use = price_aligned
            mode_base = "전체 기간 (in-sample 포함)"
            if test_only and self._n_train == 0:
                log_fn("[BACKTEST] ⚠️ 학습 후 예측을 먼저 실행하면 테스트 기간을 정확히 분리할 수 있습니다.")
            else:
                log_fn(f"[BACKTEST] 전체 기간 백테스트: {len(pred_use)}일")

        # ── 날짜 범위 필터 (다이얼로그에서 지정한 start/end) ─────
        start_date = bt_params.get("start_date")
        end_date   = bt_params.get("end_date")
        if start_date or end_date:
            mask = np.ones(len(pred_use), dtype=bool)
            if start_date:
                mask &= dates_use >= pd.Timestamp(start_date)
            if end_date:
                mask &= dates_use <= pd.Timestamp(end_date)
            n_before = int(mask.sum())
            pred_use  = pred_use[mask]
            lower_use = lower_use[mask]
            upper_use = upper_use[mask]
            price_use = price_use[mask]
            dates_use = dates_use[mask]
            log_fn(f"[BACKTEST] 날짜 필터 {start_date} ~ {end_date}: {n_before}일")
            if len(pred_use) == 0:
                raise ValueError(
                    f"선택 기간({start_date} ~ {end_date})에 예측 데이터가 없습니다.\n"
                    "기간을 다시 확인해 주세요."
                )

        mode_label = (f"{mode_base}: "
                      f"{str(dates_use[0])[:10]} ~ {str(dates_use[-1])[:10]} "
                      f"({len(pred_use)}일)")

        result = bt.run(
            price_use,   # 날짜 인덱스가 정렬된 Series
            pred_use,
            lower_use,
            upper_use,
            strategy,
        )
        result["mode"] = mode_label
        self._bt_result = result
        self._tracker.save_backtest(result)

        # ── 백테스트 상세 로그 ────────────────────────────────────
        perf = result["performance"]
        log_fn("=" * 60)
        log_fn(f"[BACKTEST] symbol={self._state.data.symbol}  "
               f"target={self._state.model.target_type}")
        log_fn(f"[BACKTEST] 총 수익률={perf['total_return']:.2%}  "
               f"CAGR={perf['cagr']:.2%}  "
               f"Sharpe={perf['sharpe']:.2f}  "
               f"Sortino={perf['sortino']:.2f}")
        log_fn(f"[BACKTEST] MDD={perf['max_drawdown']:.2%}  "
               f"승률={perf['win_rate']:.2%}  "
               f"거래 횟수={int(perf['n_trades'])}")
        trades = result.get("trades", [])
        if trades:
            log_fn(f"[BACKTEST] 거래 내역 ({len(trades)}건):")
            for i, t in enumerate(trades, 1):
                ep  = t.get("entry_price", 1) or 1
                xp  = t.get("exit_price", 0)
                pnl_pct = (xp - ep) / ep
                log_fn(f"  [{i:2d}] {t.get('entry','')} → {t.get('exit','')}  "
                       f"진입={ep:,.0f}  청산={xp:,.0f}  "
                       f"수익률={pnl_pct:+.2%}  "
                       f"PnL={t.get('pnl', 0):+,.0f}원")
        else:
            log_fn("[BACKTEST] ⚠️ 거래 없음 — threshold 설정 확인 필요")
        log_fn("=" * 60)

        return result

    # ── 편의 프로퍼티 ─────────────────────────────────────────────
    @property
    def df(self) -> Optional[pd.DataFrame]:
        return self._df

    @property
    def pred_result(self) -> Optional[Dict]:
        return self._pred_result

    @property
    def bt_result(self) -> Optional[Dict]:
        return self._bt_result

    @property
    def state(self) -> AppState:
        return self._state
