"""
시계열 데이터 증강
"""
from __future__ import annotations
import numpy as np
import torch


class TimeSeriesAugmenter:
    def __init__(
        self,
        noise_std: float = 0.005,
        scale_range: tuple = (0.95, 1.05),
        shift_max: int = 3,
        feat_drop_p: float = 0.05,
        enabled: bool = True,
    ) -> None:
        self.noise_std   = noise_std
        self.scale_range = scale_range
        self.shift_max   = shift_max
        self.feat_drop_p = feat_drop_p
        self.enabled     = enabled

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """x: (seq, features)"""
        if not self.enabled or not self.training_mode:
            return x
        x = x.clone()
        x = self._gaussian_noise(x)
        x = self._random_scale(x)
        x = self._feature_dropout(x)
        return x

    def gaussian_noise(self, x: torch.Tensor) -> torch.Tensor:
        return x + torch.randn_like(x) * self.noise_std

    def _gaussian_noise(self, x: torch.Tensor) -> torch.Tensor:
        if np.random.random() < 0.5:
            return x + torch.randn_like(x) * self.noise_std
        return x

    def _random_scale(self, x: torch.Tensor) -> torch.Tensor:
        if np.random.random() < 0.3:
            lo, hi = self.scale_range
            scale = np.random.uniform(lo, hi)
            return x * scale
        return x

    def _feature_dropout(self, x: torch.Tensor) -> torch.Tensor:
        if np.random.random() < 0.2:
            mask = torch.bernoulli(
                torch.full((x.shape[-1],), 1 - self.feat_drop_p)
            )
            return x * mask
        return x

    @property
    def training_mode(self) -> bool:
        return True   # 모델 train() 모드에서만 호출되므로 항상 True
