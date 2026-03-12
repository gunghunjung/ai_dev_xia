"""
Inference Engine — 단일 모델 & MC Dropout 불확실성 추정
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from typing import Dict
from ..logger_config import get_logger

log = get_logger("ml.inference")


class InferenceEngine:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        mc_samples: int = 30,
    ) -> None:
        self.model      = model.to(device)
        self.device     = device
        self.mc_samples = mc_samples

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Dict[str, np.ndarray]:
        """
        Returns
        -------
        pred, lower, upper, std — all (N,) numpy arrays
        """
        self.model.eval()
        x = x.to(self.device)
        point = self.model(x).cpu().numpy()

        if self.mc_samples > 1:
            samples = self._mc_predict(x)
            std   = samples.std(axis=0)
            z     = 1.96
            lower = point - z * std
            upper = point + z * std
        else:
            std   = np.zeros_like(point)
            lower = point
            upper = point

        return {"pred": point, "lower": lower, "upper": upper, "std": std}

    def _mc_predict(self, x: torch.Tensor) -> np.ndarray:
        """MC Dropout: keep dropout active"""
        self.model.train()
        preds = []
        for _ in range(self.mc_samples):
            with torch.no_grad():
                preds.append(self.model(x).cpu().numpy())
        self.model.eval()
        return np.stack(preds)   # (mc_samples, N)

    def predict_window(
        self,
        X_seq: np.ndarray,   # (seq_len, n_features) — 최근 시퀀스 하나
    ) -> Dict[str, float]:
        x = torch.tensor(X_seq, dtype=torch.float32).unsqueeze(0)  # (1, T, F)
        out = self.predict(x)
        return {k: float(v[0]) for k, v in out.items()}
