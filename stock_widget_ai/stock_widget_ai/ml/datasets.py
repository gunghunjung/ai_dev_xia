"""
시계열 슬라이딩 윈도우 Dataset
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Tuple


class TimeSeriesDataset(Dataset):
    """
    (seq_len, n_features) → scalar label 슬라이딩 윈도우
    """
    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        seq_len: int = 60,
    ) -> None:
        assert len(X) == len(y), "X/y 길이 불일치"
        self._X = X.values.astype(np.float32)
        self._y = y.values.astype(np.float32)
        self._seq = seq_len
        self._valid_idx = [
            i for i in range(seq_len, len(self._X))
            if not (np.isnan(self._X[i - seq_len:i]).any()
                    or np.isinf(self._X[i - seq_len:i]).any()
                    or np.isnan(self._y[i])
                    or np.isinf(self._y[i]))
        ]

    def __len__(self) -> int:
        return len(self._valid_idx)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        i = self._valid_idx[idx]
        x = torch.tensor(self._X[i - self._seq : i])   # (seq, F)
        y = torch.tensor(self._y[i])
        return x, y

    @property
    def n_features(self) -> int:
        return self._X.shape[1]
