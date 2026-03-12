"""
SHAPExplainer — SHAP 기반 예측 설명
──────────────────────────────────────
지원 모델:
  - TreeExplainer : XGBoost, LightGBM, RandomForest (빠름, 정확)
  - GradientExplainer: PyTorch 딥러닝 모델 (GPU 지원)
  - KernelExplainer  : 범용 (느림, 임의 모델)
  - DeepExplainer    : 딥러닝 모델 대안

출력:
  - 피처별 SHAP 값
  - 예측에 가장 많이 기여한 피처 Top-K
  - 양수/음수 기여 피처 분리
  - 피처별 평균 |SHAP| (전역 중요도)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from ..logger_config import get_logger

log = get_logger("explain.shap")


class SHAPExplainer:
    """
    Parameters
    ----------
    model        : 설명할 모델 인스턴스
    feature_names: 피처 이름 목록
    model_type   : "tree" | "gradient" | "deep" | "kernel" | "auto"
    background_n : KernelExplainer 배경 샘플 수 (느림)
    """

    def __init__(
        self,
        model:         Any,
        feature_names: Optional[List[str]] = None,
        model_type:    str = "auto",
        background_n:  int = 100,
    ) -> None:
        self._model       = model
        self._feat_names  = feature_names or []
        self._mtype       = model_type
        self._bg_n        = background_n
        self._explainer:  Optional[Any] = None
        self._shap_vals:  Optional[np.ndarray] = None

    # ── Explainer 생성 ────────────────────────────────────────────────────
    def build_explainer(
        self, background: Optional[np.ndarray] = None
    ) -> "SHAPExplainer":
        """
        background: 배경 데이터 (KernelExplainer, GradientExplainer 필요)
        """
        try:
            import shap
        except ImportError:
            log.warning("shap 미설치. pip install shap 후 사용 가능")
            return self

        mtype = self._resolve_type()

        try:
            if mtype == "tree":
                self._explainer = shap.TreeExplainer(self._model)
                log.info("TreeExplainer 생성")
            elif mtype == "gradient":
                if background is None:
                    raise ValueError("GradientExplainer에 background 데이터 필요")
                import torch
                bg_tensor = torch.tensor(background[:self._bg_n], dtype=torch.float32)
                self._explainer = shap.GradientExplainer(self._model, bg_tensor)
                log.info("GradientExplainer 생성")
            elif mtype == "deep":
                if background is None:
                    raise ValueError("DeepExplainer에 background 데이터 필요")
                import torch
                bg_tensor = torch.tensor(background[:self._bg_n], dtype=torch.float32)
                self._explainer = shap.DeepExplainer(self._model, bg_tensor)
                log.info("DeepExplainer 생성")
            else:   # kernel
                if background is None:
                    raise ValueError("KernelExplainer에 background 데이터 필요")
                bg_summary = shap.kmeans(background, min(self._bg_n, 50))
                self._explainer = shap.KernelExplainer(
                    self._model.predict, bg_summary
                )
                log.info("KernelExplainer 생성")
        except Exception as e:
            log.warning(f"Explainer 생성 실패: {e}")

        return self

    # ── SHAP 계산 ─────────────────────────────────────────────────────────
    def explain(self, X: np.ndarray) -> Optional[np.ndarray]:
        """
        SHAP 값 계산.
        Returns shape (N, n_features) or None on failure.
        """
        if self._explainer is None:
            log.warning("Explainer가 초기화되지 않았습니다. build_explainer() 먼저 호출")
            return None
        try:
            import shap
            import torch

            mtype = self._resolve_type()
            if mtype in ("gradient", "deep"):
                xt = torch.tensor(X, dtype=torch.float32)
                vals = self._explainer.shap_values(xt)
            else:
                vals = self._explainer.shap_values(X)

            # 분류 모델 → 클래스 1의 SHAP 값만
            if isinstance(vals, list):
                vals = vals[-1] if len(vals) > 1 else vals[0]

            self._shap_vals = np.asarray(vals)
            log.info(f"SHAP 계산 완료: shape={self._shap_vals.shape}")
            return self._shap_vals

        except Exception as e:
            log.warning(f"SHAP 계산 실패: {e}")
            return None

    # ── 결과 분석 ─────────────────────────────────────────────────────────
    def global_importance(self) -> pd.DataFrame:
        """
        전역 피처 중요도 (평균 |SHAP|).
        """
        if self._shap_vals is None:
            return pd.DataFrame()
        vals = self._shap_vals
        if vals.ndim == 3:
            vals = vals.reshape(vals.shape[0], -1)
        mean_abs = np.mean(np.abs(vals), axis=0)

        names = self._feat_names
        if len(names) < len(mean_abs):
            names = [f"f{i}" for i in range(len(mean_abs))]

        df = pd.DataFrame({
            "feature":    names[:len(mean_abs)],
            "shap_mean_abs": mean_abs,
        }).sort_values("shap_mean_abs", ascending=False).reset_index(drop=True)
        df["rank"] = range(1, len(df) + 1)
        return df

    def local_explanation(
        self, sample_idx: int = -1, top_k: int = 10
    ) -> Dict[str, Any]:
        """
        단일 샘플의 SHAP 설명.
        sample_idx: -1 이면 마지막 샘플 (최신 예측).

        Returns
        -------
        dict with:
          top_positive   : 상승에 기여한 피처 목록
          top_negative   : 하락에 기여한 피처 목록
          base_value     : 평균 예측
          prediction     : 이 샘플의 예측값
          explanation_text: 사람이 읽을 수 있는 설명
        """
        if self._shap_vals is None:
            return {"error": "SHAP 값 없음"}

        vals = self._shap_vals
        if vals.ndim == 3:
            vals = vals.reshape(vals.shape[0], -1)

        idx = sample_idx if sample_idx >= 0 else len(vals) - 1
        sv  = vals[idx]

        names = self._feat_names
        if len(names) < len(sv):
            names = [f"f{i}" for i in range(len(sv))]

        # 기여도 정렬
        sorted_idx = np.argsort(np.abs(sv))[::-1]
        top_pos = [(names[i], float(sv[i])) for i in sorted_idx if sv[i] > 0][:top_k]
        top_neg = [(names[i], float(sv[i])) for i in sorted_idx if sv[i] < 0][:top_k]

        # 텍스트 설명 생성
        explanation = self._generate_text(top_pos, top_neg)

        return {
            "top_positive":    top_pos,
            "top_negative":    top_neg,
            "shap_values":     sv.tolist(),
            "feature_names":   names[:len(sv)],
            "explanation_text": explanation,
        }

    def top_features(self, top_k: int = 10) -> List[Tuple[str, float]]:
        """전역 중요도 Top-K (이름, 중요도) 목록"""
        df = self.global_importance()
        if df.empty:
            return []
        return [(row["feature"], row["shap_mean_abs"])
                for _, row in df.head(top_k).iterrows()]

    # ── 텍스트 생성 ───────────────────────────────────────────────────────
    @staticmethod
    def _generate_text(
        top_pos: List[Tuple[str, float]],
        top_neg: List[Tuple[str, float]],
    ) -> str:
        lines = []
        if top_pos:
            feat_list = ", ".join(f"{n} (+{v:.4f})" for n, v in top_pos[:3])
            lines.append(f"📈 상승 기여 주요 요인: {feat_list}")
        if top_neg:
            feat_list = ", ".join(f"{n} ({v:.4f})" for n, v in top_neg[:3])
            lines.append(f"📉 하락 기여 주요 요인: {feat_list}")
        if not lines:
            lines.append("설명 가능한 주요 요인 없음")
        lines.append("\n⚠ SHAP 값은 모델의 학습된 패턴 기반이며 인과관계를 의미하지 않습니다.")
        return "\n".join(lines)

    # ── 타입 추론 ─────────────────────────────────────────────────────────
    def _resolve_type(self) -> str:
        if self._mtype != "auto":
            return self._mtype
        try:
            import torch
            if isinstance(self._model, torch.nn.Module):
                return "gradient"
        except ImportError:
            pass
        if hasattr(self._model, "feature_importances_"):   # sklearn/GBM
            return "tree"
        return "kernel"
