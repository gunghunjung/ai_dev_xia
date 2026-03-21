# features/fusion.py — 멀티모달 피처 퓨전
from __future__ import annotations
import numpy as np
from typing import Tuple


class FeatureFusion:
    """
    CV 임베딩 + TS 피처 → 통합 특징 벡터

    퓨전 방식:
    - concatenate: 단순 연결 (기본값)
    - weighted: 가중 평균 결합
    """

    def __init__(self, method: str = "concatenate",
                 cv_weight: float = 0.5):
        self.method = method
        self.cv_weight = cv_weight
        self.ts_weight = 1 - cv_weight

    def fuse(self, cv_emb: np.ndarray, ts_feat: np.ndarray) -> np.ndarray:
        """
        Args:
            cv_emb:  (N, cv_dim)
            ts_feat: (N, ts_dim)
        Returns:
            fused: (N, cv_dim + ts_dim) or (N, d)
        """
        if self.method == "concatenate":
            return np.concatenate([cv_emb, ts_feat], axis=1)
        elif self.method == "weighted":
            # L2 정규화 후 가중 결합
            cv_n = cv_emb / (np.linalg.norm(cv_emb, axis=1, keepdims=True) + 1e-10)
            ts_n = ts_feat / (np.linalg.norm(ts_feat, axis=1, keepdims=True) + 1e-10)
            # 차원 맞추기 (짧은 쪽 패딩)
            if cv_n.shape[1] != ts_n.shape[1]:
                max_d = max(cv_n.shape[1], ts_n.shape[1])
                cv_n = np.pad(cv_n, ((0,0),(0, max_d-cv_n.shape[1])))
                ts_n = np.pad(ts_n, ((0,0),(0, max_d-ts_n.shape[1])))
            return self.cv_weight * cv_n + self.ts_weight * ts_n
        else:
            return np.concatenate([cv_emb, ts_feat], axis=1)

    @staticmethod
    def get_fused_dim(cv_dim: int, ts_dim: int, method: str = "concatenate") -> int:
        if method == "concatenate":
            return cv_dim + ts_dim
        elif method == "weighted":
            return max(cv_dim, ts_dim)
        return cv_dim + ts_dim
