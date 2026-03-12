"""
커스텀 Loss 함수
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class HuberLoss(nn.Module):
    def __init__(self, delta: float = 1.0) -> None:
        super().__init__()
        self.delta = delta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.huber_loss(pred, target, delta=self.delta)


class QuantileLoss(nn.Module):
    def __init__(self, quantiles=(0.1, 0.5, 0.9)) -> None:
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """preds: (B, n_quantiles), target: (B,)"""
        assert preds.shape[1] == len(self.quantiles)
        total = torch.zeros(1, device=preds.device)
        for i, q in enumerate(self.quantiles):
            e = target - preds[:, i]
            total = total + torch.max(q * e, (q - 1) * e).mean()
        return total / len(self.quantiles)


class DirectionLoss(nn.Module):
    """MSE + 방향 패널티"""
    def __init__(self, alpha: float = 0.5) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse = F.mse_loss(pred, target)
        sign_penalty = torch.mean((torch.sign(pred) != torch.sign(target)).float())
        return mse + self.alpha * sign_penalty


def get_loss(name: str) -> nn.Module:
    mapping = {
        "mse":       nn.MSELoss(),
        "mae":       nn.L1Loss(),
        "huber":     HuberLoss(),
        "bce":       nn.BCEWithLogitsLoss(),
        "ce":        nn.CrossEntropyLoss(),
        "quantile":  QuantileLoss(),
        "direction": DirectionLoss(),
    }
    if name not in mapping:
        raise ValueError(f"Unknown loss: {name}. Options: {list(mapping)}")
    return mapping[name]
