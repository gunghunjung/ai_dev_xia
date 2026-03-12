"""
Trainer — 모델 종류/task 무관하게 재사용 가능한 학습 루프
"""
from __future__ import annotations
import os
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from typing import Callable, Dict, List, Optional
from .losses import get_loss
from .metrics import regression_metrics, classification_metrics
from .datasets import TimeSeriesDataset
from .augmentations import TimeSeriesAugmenter
from ..logger_config import get_logger

log = get_logger("ml.trainer")


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-5) -> None:
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best      = float("inf")
        self.should_stop = False

    def step(self, loss: float) -> bool:
        if loss < self.best - self.min_delta:
            self.best    = loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        loss_name: str = "huber",
        optimizer_name: str = "adamw",
        lr: float = 5e-4,
        weight_decay: float = 1e-4,
        grad_clip: float = 1.0,
        scheduler_name: str = "onecycle",
        early_stopping_patience: int = 10,
        use_mixed_precision: bool = True,
        use_augmentation: bool = True,
        on_epoch_end: Optional[Callable[[int, float, float], None]] = None,
        on_log: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.model    = model.to(device)
        self.device   = device
        self.loss_fn  = get_loss(loss_name)
        self.grad_clip = grad_clip
        self.on_epoch_end = on_epoch_end
        self.on_log  = on_log or (lambda s: log.info(s))
        self.aug     = TimeSeriesAugmenter(enabled=use_augmentation)
        self.use_amp = use_mixed_precision and device.type == "cuda"
        # torch.cuda.amp.GradScaler → torch.amp.GradScaler (PyTorch 2.4+ 권장)
        self.scaler  = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self._stop   = False

        # optimizer
        optim_cls = {
            "adam":    torch.optim.Adam,
            "adamw":   torch.optim.AdamW,
            "rmsprop": torch.optim.RMSprop,
        }.get(optimizer_name, torch.optim.AdamW)
        self.optimizer = optim_cls(model.parameters(), lr=lr, weight_decay=weight_decay)
        self._scheduler_name = scheduler_name
        self._lr = lr
        self._scheduler = None   # built in fit() after knowing n_steps

    def fit(
        self,
        dataset: TimeSeriesDataset,
        epochs: int = 50,
        batch_size: int = 64,
        val_ratio: float = 0.15,
        seed: int = 42,
    ) -> Dict[str, List[float]]:
        torch.manual_seed(seed)
        n_val   = max(1, int(len(dataset) * val_ratio))
        n_train = len(dataset) - n_val
        # ⚠️ random_split은 시계열에 부적합: 미래 데이터가 학습에 포함될 수 있음
        # → 시간순(temporal) 분할: 앞부분 n_train개 학습, 뒷부분 n_val개 검증
        indices  = list(range(len(dataset)))
        train_ds = Subset(dataset, indices[:n_train])
        val_ds   = Subset(dataset, indices[n_train:])
        self.n_train = n_train   # predict()에서 테스트 메트릭 계산에 사용
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True)
        val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

        self._scheduler = self._build_scheduler(len(train_dl), epochs)
        es = EarlyStopping(patience=10)
        best_val = float("inf")
        best_path = None
        history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

        for epoch in range(1, epochs + 1):
            if self._stop:
                self.on_log("학습 중단 요청")
                break

            t0 = time.time()
            tr_loss = self._train_epoch(train_dl)
            vl_loss = self._val_epoch(val_dl)
            elapsed = time.time() - t0

            history["train_loss"].append(tr_loss)
            history["val_loss"].append(vl_loss)

            msg = f"Epoch {epoch}/{epochs}  train={tr_loss:.6f}  val={vl_loss:.6f}  {elapsed:.1f}s"
            self.on_log(msg)
            if self.on_epoch_end:
                self.on_epoch_end(epoch, tr_loss, vl_loss)

            if vl_loss < best_val:
                best_val = vl_loss
                import tempfile
                best_path = os.path.join(tempfile.gettempdir(), "best_model_tmp.pt")
                torch.save(self.model.state_dict(), best_path)

            if es.step(vl_loss):
                self.on_log(f"Early stopping at epoch {epoch}")
                break

        # best weight 복원
        if best_path and os.path.exists(best_path):
            self.model.load_state_dict(torch.load(best_path, map_location=self.device, weights_only=True))
        return history

    def stop(self) -> None:
        self._stop = True

    def _train_epoch(self, loader) -> float:
        self.model.train()
        total   = 0.0
        n_valid = 0          # NaN/Inf skip 제외 유효 배치 수
        for x, y in loader:
            x = self.aug._gaussian_noise(x.to(self.device))
            y = y.to(self.device)
            self.optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=self.use_amp):
                pred = self.model(x)
                loss = self.loss_fn(pred, y)
            # NaN/Inf loss 배치는 건너뜀 (Inf 피처가 LayerNorm에서 NaN을 만드는 경우 방어)
            if not torch.isfinite(loss):
                self.optimizer.zero_grad()
                continue
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            # scaler.step() 이 실제로 optimizer.step()을 호출했는지 확인
            scale_before = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            # 스케일이 줄지 않았으면 optimizer.step()이 실행된 것 → scheduler.step()
            if self._scheduler and self.scaler.get_scale() >= scale_before:
                self._scheduler.step()
            total   += loss.item()
            n_valid += 1
        return total / max(1, n_valid)

    @torch.no_grad()
    def _val_epoch(self, loader) -> float:
        self.model.eval()
        total = 0.0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            pred = self.model(x)
            total += self.loss_fn(pred, y).item()
        return total / max(1, len(loader))

    def _build_scheduler(self, steps_per_epoch: int, epochs: int):
        total = steps_per_epoch * epochs
        name  = self._scheduler_name
        # OneCycleLR은 __init__ 에서 step()을 1회 내부 호출하므로
        # "lr_scheduler.step() before optimizer.step()" 경고를 억제
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            if name == "onecycle":
                return torch.optim.lr_scheduler.OneCycleLR(
                    self.optimizer, max_lr=self._lr * 3, total_steps=total,
                    pct_start=0.3, anneal_strategy="cos",
                )
            if name == "cosine":
                return torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=total,
                )
            if name == "plateau":
                return torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, patience=5, factor=0.5
                )
        return None
