"""
WalkForwardValidator — Walk-Forward 교차 검증
──────────────────────────────────────────────
금융 시계열에 특화된 검증 방식:
  - 미래 데이터 누수 완전 방지
  - Expanding window / Rolling window 지원
  - 각 fold별 학습/검증 성능 + 백테스트 지표 산출

⚠ 중요: 항상 시간 순서를 지켜 train < validation 를 보장.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Callable, Tuple
from ..logger_config import get_logger
from ..ml.metrics import regression_metrics, financial_metrics, classification_metrics

log = get_logger("backtest.walk_forward")


class WalkForwardResult:
    """Walk-Forward 결과 컨테이너"""

    def __init__(self) -> None:
        self.folds: List[Dict[str, Any]] = []

    def add_fold(self, fold_result: Dict[str, Any]) -> None:
        self.folds.append(fold_result)

    @property
    def n_folds(self) -> int:
        return len(self.folds)

    def aggregate_metrics(self) -> Dict[str, float]:
        """폴드별 지표의 평균 ± 표준편차"""
        if not self.folds:
            return {}
        all_keys = set()
        for f in self.folds:
            all_keys.update(f.get("metrics", {}).keys())

        result = {}
        for key in all_keys:
            vals = [f["metrics"].get(key, float("nan")) for f in self.folds]
            vals = [v for v in vals if not np.isnan(v)]
            if vals:
                result[f"{key}_mean"] = float(np.mean(vals))
                result[f"{key}_std"]  = float(np.std(vals))
                result[f"{key}_min"]  = float(np.min(vals))
                result[f"{key}_max"]  = float(np.max(vals))
        return result

    def overfitting_score(self) -> float:
        """
        Train vs Validation 성능 괴리도.
        높을수록 과최적화 의심.
        (train_da - val_da) / (train_da + 1e-9)
        """
        diffs = []
        for f in self.folds:
            train_da = f.get("train_metrics", {}).get("da", float("nan"))
            val_da   = f.get("metrics", {}).get("da", float("nan"))
            if not (np.isnan(train_da) or np.isnan(val_da)):
                diffs.append(abs(train_da - val_da))
        return float(np.mean(diffs)) if diffs else float("nan")

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for i, f in enumerate(self.folds):
            row = {"fold": i + 1,
                   "train_start": f.get("train_start"),
                   "train_end":   f.get("train_end"),
                   "val_start":   f.get("val_start"),
                   "val_end":     f.get("val_end"),
                   "n_train":     f.get("n_train"),
                   "n_val":       f.get("n_val")}
            row.update(f.get("metrics", {}))
            rows.append(row)
        return pd.DataFrame(rows)

    def oof_predictions(self) -> np.ndarray:
        """전체 OOF 예측값 (시간 순서 연결)"""
        return np.concatenate([f["val_preds"] for f in self.folds if "val_preds" in f])

    def oof_true(self) -> np.ndarray:
        return np.concatenate([f["val_true"]  for f in self.folds if "val_true"  in f])

    def summary(self) -> str:
        agg = self.aggregate_metrics()
        lines = [
            f"Walk-Forward 결과 ({self.n_folds} 폴드)",
            f"  DA    : {agg.get('da_mean', float('nan')):.3f} ± {agg.get('da_std', float('nan')):.3f}",
            f"  RMSE  : {agg.get('rmse_mean', float('nan')):.5f} ± {agg.get('rmse_std', float('nan')):.5f}",
            f"  Sharpe: {agg.get('sharpe_mean', float('nan')):.2f} ± {agg.get('sharpe_std', float('nan')):.2f}",
            f"  과최적화 점수: {self.overfitting_score():.3f} (낮을수록 좋음)",
        ]
        return "\n".join(lines)


class WalkForwardValidator:
    """
    Parameters
    ----------
    n_splits    : 검증 폴드 수
    min_train   : 최소 학습 샘플 수
    val_size    : 각 검증 폴드 크기 (샘플 수 또는 비율)
    expanding   : True=expanding window, False=rolling window
    gap         : 학습 끝~검증 시작 사이 gap (leakage 방지, 단위: 샘플)
    task        : "regression" | "classification"
    """

    def __init__(
        self,
        n_splits:  int = 5,
        min_train: int = 252,   # 1년치
        val_size:  int = 63,    # 분기치
        expanding: bool = True,
        gap:       int = 0,
        task:      str = "regression",
    ) -> None:
        self._n       = n_splits
        self._min_tr  = min_train
        self._val_sz  = val_size
        self._expand  = expanding
        self._gap     = gap
        self._task    = task

    # ── 폴드 생성 ─────────────────────────────────────────────────────────
    def split(self, X: np.ndarray, y: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        (train_idx, val_idx) 쌍의 리스트 반환.
        항상 시간 순서 보장.
        """
        N = len(X)
        splits = []

        # val_size가 비율이면 샘플 수로 변환
        val_sz = int(self._val_sz * N) if self._val_sz < 1 else self._val_sz

        # 가능한 검증 윈도우 끝 위치 목록
        val_ends = np.linspace(
            self._min_train + val_sz + self._gap,
            N,
            self._n + 1,
        ).astype(int)[1:]   # 마지막 N개 폴드

        for val_end in val_ends:
            val_start = val_end - val_sz
            train_end = val_start - self._gap

            if train_end < self._min_train:
                continue

            if self._expand:
                train_start = 0
            else:
                train_start = max(0, train_end - self._min_train * 2)

            train_idx = np.arange(train_start, train_end)
            val_idx   = np.arange(val_start, val_end)
            splits.append((train_idx, val_idx))

        log.debug(f"WalkForward splits: {len(splits)}개 폴드 생성")
        return splits

    # ── 검증 실행 ─────────────────────────────────────────────────────────
    def validate(
        self,
        X:              np.ndarray,
        y:              np.ndarray,
        model_factory:  Callable[[], Any],   # () → 새 모델 인스턴스
        price_series:   Optional[np.ndarray] = None,   # 백테스트용 실제 가격
    ) -> WalkForwardResult:
        """
        Parameters
        ----------
        model_factory : 매 폴드마다 새 모델을 반환하는 callable.
            반환된 모델은 .fit(X_tr, y_tr, X_val, y_val) 인터페이스 필요.

        Returns
        -------
        WalkForwardResult
        """
        result = WalkForwardResult()
        splits = self.split(X, y)

        if not splits:
            log.warning("유효한 폴드 없음 — 데이터가 너무 적습니다")
            return result

        for fold_i, (tr_idx, val_idx) in enumerate(splits):
            log.info(
                f"Fold {fold_i+1}/{len(splits)}: "
                f"train[{tr_idx[0]}:{tr_idx[-1]}] val[{val_idx[0]}:{val_idx[-1]}]"
            )
            X_tr, y_tr  = X[tr_idx], y[tr_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            # 새 모델 생성 + 학습
            model = model_factory()
            try:
                history = model.fit(X_tr, y_tr, X_val, y_val)
            except TypeError:
                history = {}
                model.fit(X_tr, y_tr)

            # 검증셋 예측
            val_preds = np.asarray(model.predict(X_val), dtype=float).ravel()

            # 학습셋 예측 (과최적화 진단용)
            try:
                train_preds = np.asarray(model.predict(X_tr), dtype=float).ravel()
                train_metrics = regression_metrics(y_tr, train_preds)
            except Exception:
                train_metrics = {}

            # 검증 지표
            if self._task == "classification":
                metrics = classification_metrics(y_val, val_preds)
            else:
                metrics = regression_metrics(y_val, val_preds)

            # 백테스트 수익률 (가격 시리즈 있을 때)
            bt_metrics: Dict[str, float] = {}
            if price_series is not None:
                bt_metrics = self._quick_backtest(
                    price_series[val_idx], val_preds, y_val
                )
                metrics.update(bt_metrics)

            result.add_fold({
                "fold":          fold_i + 1,
                "train_start":   int(tr_idx[0]),
                "train_end":     int(tr_idx[-1]),
                "val_start":     int(val_idx[0]),
                "val_end":       int(val_idx[-1]),
                "n_train":       len(tr_idx),
                "n_val":         len(val_idx),
                "metrics":       metrics,
                "train_metrics": train_metrics,
                "val_preds":     val_preds,
                "val_true":      y_val.copy(),
                "history":       history,
            })

            log.info(
                f"  DA={metrics.get('da', float('nan')):.3f}  "
                f"RMSE={metrics.get('rmse', float('nan')):.5f}  "
                f"Sharpe={metrics.get('sharpe', float('nan')):.2f}"
            )

        log.info(f"\n{result.summary()}")
        return result

    # ── 빠른 백테스트 ─────────────────────────────────────────────────────
    @staticmethod
    def _quick_backtest(
        prices:   np.ndarray,
        preds:    np.ndarray,
        y_true:   np.ndarray,
        fee:      float = 0.00015,
        threshold: float = 0.005,
    ) -> Dict[str, float]:
        """
        신호 기반 단순 백테스트 (수수료 포함).
        preds > threshold → 매수, preds < -threshold → 매도
        """
        try:
            n = len(prices)
            ret = np.diff(prices) / (prices[:-1] + 1e-9)
            strategy_ret = np.zeros(n - 1)
            position = 0  # 0=현금, 1=롱

            for i in range(n - 1):
                signal = 1 if preds[i] > threshold else (0 if preds[i] < -threshold else position)
                if signal != position:
                    strategy_ret[i] -= fee   # 거래 비용
                strategy_ret[i] += signal * ret[i]
                position = signal

            return financial_metrics(strategy_ret)
        except Exception as e:
            log.warning(f"Quick backtest 실패: {e}")
            return {}

    @property
    def _min_train(self) -> int:
        return self._min_tr
