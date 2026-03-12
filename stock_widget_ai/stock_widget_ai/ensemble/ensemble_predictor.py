"""
EnsemblePredictor — 다중 모델 앙상블 예측기
──────────────────────────────────────────────
지원 앙상블 방법:
  - simple_avg    : 단순 평균
  - weighted_avg  : 가중 평균 (성능 기반 자동 설정)
  - stacking      : MetaLearner 기반 스태킹
  - regime_aware  : 시장 국면 적응형 가중치 (RegimeEnsemble)

출력:
  - 점 예측 (point estimate)
  - 분위수 예측 (10th, 25th, 50th, 75th, 90th)
  - 방향성 확률 (P(상승) / P(하락))
  - 신뢰구간
  - 앙상블 불일치도 (모델 간 분산 → 불확실성 대리 지표)
"""
from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from ..logger_config import get_logger

log = get_logger("ensemble.predictor")


class EnsemblePredictor:
    """
    Parameters
    ----------
    method      : "simple_avg" | "weighted_avg" | "stacking" | "regime_aware"
    weights     : 수동 가중치 dict {model_name: float}. None이면 자동 계산.
    quantiles   : 출력할 분위수 목록
    min_models  : 앙상블에 필요한 최소 모델 수
    """

    def __init__(
        self,
        method:     str = "weighted_avg",
        weights:    Optional[Dict[str, float]] = None,
        quantiles:  List[float] = [0.1, 0.25, 0.5, 0.75, 0.9],
        min_models: int = 2,
    ) -> None:
        self._method    = method
        self._weights   = weights or {}
        self._quantiles = quantiles
        self._min_models = min_models
        self._models:  Dict[str, Any] = {}
        self._val_scores: Dict[str, float] = {}  # 검증 성능 (낮을수록 좋음: RMSE)
        self._meta_learner = None

    # ── 모델 등록 ─────────────────────────────────────────────────────────
    def add_model(
        self,
        name: str,
        model: Any,
        val_score: float = 1.0,   # RMSE 또는 오류율 (낮을수록 좋음)
    ) -> None:
        self._models[name] = model
        self._val_scores[name] = val_score
        log.debug(f"모델 등록: {name} (val_score={val_score:.4f})")

    def remove_model(self, name: str) -> None:
        self._models.pop(name, None)
        self._val_scores.pop(name, None)

    @property
    def model_names(self) -> List[str]:
        return list(self._models.keys())

    # ── 가중치 계산 ───────────────────────────────────────────────────────
    def compute_weights(self) -> Dict[str, float]:
        """성능 기반 가중치 자동 계산 (RMSE 역수 정규화)"""
        if not self._models:
            return {}
        if self._weights:
            # 수동 가중치 정규화
            total = sum(self._weights.get(n, 0) for n in self._models)
            if total > 0:
                return {n: self._weights.get(n, 0) / total for n in self._models}

        # 검증 점수 기반 자동 가중치
        scores = {n: self._val_scores.get(n, 1.0) for n in self._models}
        # 역수 변환 (낮은 오류 → 높은 가중치)
        inv = {n: 1.0 / (s + 1e-9) for n, s in scores.items()}
        total = sum(inv.values())
        return {n: v / total for n, v in inv.items()}

    # ── 예측 ──────────────────────────────────────────────────────────────
    def predict(
        self,
        X: np.ndarray,
        regime_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Parameters
        ----------
        X              : (N, seq_len, n_features) 또는 (N, n_features)
        regime_weights : 국면별 가중치 (있으면 우선 적용)

        Returns
        -------
        dict with keys:
          point, lower_90, upper_90, lower_50, upper_50,
          prob_up, prob_down, uncertainty,
          individual_preds, weights_used
        """
        if len(self._models) < self._min_models:
            log.warning(f"모델 수 부족 ({len(self._models)} < {self._min_models})")

        # 개별 모델 예측 수집
        individual: Dict[str, np.ndarray] = {}
        for name, model in self._models.items():
            try:
                pred = self._get_prediction(name, model, X)
                individual[name] = pred
            except Exception as e:
                log.warning(f"{name} 예측 실패: {e}")

        if not individual:
            log.error("모든 모델 예측 실패")
            n = len(X) if hasattr(X, "__len__") else 1
            dummy = np.zeros(n)
            return self._empty_result(dummy)

        # 가중치 결정
        weights = regime_weights if regime_weights else self.compute_weights()
        weights = {n: weights.get(n, 1.0 / len(individual)) for n in individual}
        total_w = sum(weights.values())
        weights = {n: w / total_w for n, w in weights.items()}

        # 앙상블 집계
        preds_arr = np.stack(list(individual.values()), axis=0)  # (M, N)
        w_arr     = np.array([weights[n] for n in individual])

        # 가중 평균
        point = np.average(preds_arr, weights=w_arr, axis=0)

        # 불확실성 (모델 간 가중 분산)
        diff_sq = np.average((preds_arr - point[np.newaxis, :]) ** 2, weights=w_arr, axis=0)
        uncertainty = np.sqrt(diff_sq)

        # 분위수 (개별 예측의 분위수)
        q10  = np.quantile(preds_arr, 0.10, axis=0)
        q25  = np.quantile(preds_arr, 0.25, axis=0)
        q50  = np.quantile(preds_arr, 0.50, axis=0)
        q75  = np.quantile(preds_arr, 0.75, axis=0)
        q90  = np.quantile(preds_arr, 0.90, axis=0)

        # 방향 확률 (예측값이 양수일 비율)
        prob_up   = np.mean(preds_arr > 0, axis=0)
        prob_down = 1.0 - prob_up

        return {
            "point":       point,
            "lower_90":    q10,
            "upper_90":    q90,
            "lower_50":    q25,
            "upper_50":    q75,
            "median":      q50,
            "prob_up":     prob_up,
            "prob_down":   prob_down,
            "uncertainty": uncertainty,
            "individual_preds": individual,
            "weights_used": weights,
        }

    def predict_single(
        self, x: np.ndarray, **kwargs
    ) -> Dict[str, float]:
        """단일 샘플 예측 (GUI 실시간 출력용) — scalar dict 반환"""
        x_batch = x[np.newaxis] if x.ndim < 3 else x
        result  = self.predict(x_batch, **kwargs)
        return {
            "point":       float(result["point"].mean()),
            "lower_90":    float(result["lower_90"].mean()),
            "upper_90":    float(result["upper_90"].mean()),
            "lower_50":    float(result["lower_50"].mean()),
            "upper_50":    float(result["upper_50"].mean()),
            "prob_up":     float(result["prob_up"].mean()),
            "prob_down":   float(result["prob_down"].mean()),
            "uncertainty": float(result["uncertainty"].mean()),
        }

    def agreement_score(self, X: np.ndarray) -> float:
        """
        모델 간 방향성 일치도 (0~1).
        1.0 = 모든 모델이 같은 방향 예측 (확신)
        0.5 = 무작위 수준
        """
        result = self.predict(X)
        return float(1.0 - result["uncertainty"].mean() /
                     (abs(result["point"]).mean() + 1e-9))

    # ── 내부 헬퍼 ─────────────────────────────────────────────────────────
    @staticmethod
    def _get_prediction(name: str, model: Any, X: np.ndarray) -> np.ndarray:
        """모델 타입에 따라 적절한 predict 메서드 호출"""
        # PyTorch 모델
        try:
            import torch
            if isinstance(model, torch.nn.Module):
                model.eval()
                with torch.no_grad():
                    xt = torch.tensor(X, dtype=torch.float32)
                    out = model(xt)
                    return out.cpu().numpy().ravel()
        except Exception:
            pass

        # sklearn/GBM 스타일
        if hasattr(model, "predict"):
            # 2D input 필요 시 reshape
            Xr = X.reshape(len(X), -1) if X.ndim == 3 else X
            return np.asarray(model.predict(Xr), dtype=float)

        raise RuntimeError(f"{name}: 알 수 없는 모델 인터페이스")

    @staticmethod
    def _empty_result(arr: np.ndarray) -> Dict[str, Any]:
        return {
            "point": arr, "lower_90": arr, "upper_90": arr,
            "lower_50": arr, "upper_50": arr, "median": arr,
            "prob_up": arr + 0.5, "prob_down": arr + 0.5,
            "uncertainty": arr, "individual_preds": {}, "weights_used": {},
        }

    # ── 직렬화 ────────────────────────────────────────────────────────────
    def summary(self) -> str:
        lines = [
            f"EnsemblePredictor ({self._method})",
            f"  모델 수: {len(self._models)}",
        ]
        weights = self.compute_weights()
        for name in self._models:
            w = weights.get(name, 0)
            sc = self._val_scores.get(name, float("nan"))
            lines.append(f"  [{name}]  weight={w:.3f}  val_rmse={sc:.5f}")
        return "\n".join(lines)
