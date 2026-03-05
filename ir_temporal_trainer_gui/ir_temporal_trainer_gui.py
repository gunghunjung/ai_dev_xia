import os
import json
import time
import math
import traceback
import threading
import queue
from collections import Counter
from dataclasses import dataclass, asdict, fields
from typing import List, Tuple, Optional, Dict

import cv2
import numpy as np

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# -----------------------------
# Config
# -----------------------------
VIDEO_EXTS = (".mp4", ".avi", ".mkv", ".mov")

@dataclass
class AppConfig:
    roi: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)
    fps_target: int = 10
    hist_bins: int = 16
    window_len: int = 64     # time steps per training sample
    stride: int = 32         # window stride
    smooth_sec: float = 0.5  # moving average seconds on features
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_augmentation: bool = True
    aug_noise_std: float = 0.01
    aug_time_shift: int = 4
    aug_feature_dropout: float = 0.1
    loss_name: str = "cross_entropy_ls"   # cross_entropy_ls | focal
    label_smoothing: float = 0.05
    focal_gamma: float = 2.0
    scheduler_name: str = "cosine"        # none | cosine | onecycle
    k_folds: int = 1
    use_hard_mining: bool = False
    hard_mining_ratio: float = 0.35
    use_type_oversampling: bool = False
    hard_type_keywords: str = ""
    type_oversample_factor: int = 2
    auto_threshold: bool = True
    infer_threshold: float = 0.5
    use_balance_features: bool = False
    balance_feature_gain: float = 1.0
    use_spike_features: bool = False
    spike_feature_gain: float = 1.0
    use_adaptive_pixel_smoothing: bool = False
    pixel_jitter_threshold: float = 0.08
    pixel_smoothing_kernel: int = 3
    use_tilt_detail_features: bool = False
    tilt_gain: float = 1.0
    tilt_loss_alpha: float = 0.0
    auto_hparam_search: bool = False
    hparam_trials: int = 8
    train_class_mode: str = "both"  # both | normal | defect


# -----------------------------
# Feature extraction
# -----------------------------
def list_videos(root: str) -> List[str]:
    out = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(VIDEO_EXTS):
                out.append(os.path.join(dirpath, fn))
    return sorted(out)

def infer_type_name(video_path: str, dataset_root: str, label: int) -> str:
    cls_name = "defect" if int(label) == 1 else "normal"
    cls_root = os.path.join(dataset_root, cls_name)
    rel = os.path.relpath(video_path, cls_root)
    parts = [p for p in rel.replace("\\", "/").split("/") if p not in (".", "")]
    if len(parts) >= 2:
        return parts[0].lower()
    stem = os.path.splitext(os.path.basename(video_path))[0].lower()
    return stem.split("_")[0] if "_" in stem else stem

def parse_item(it):
    # Supported shapes:
    # (X, y) or (X, y, type_name)
    if len(it) >= 3:
        return it[0], int(it[1]), str(it[2])
    return it[0], int(it[1]), "unknown"

def safe_mkdir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def moving_average(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x
    # x: T x D
    pad = k // 2
    xp = np.pad(x, ((pad, pad), (0, 0)), mode="edge")
    c = np.cumsum(xp, axis=0)
    y = (c[k:] - c[:-k]) / float(k)
    return y

def feature_layout(cfg: AppConfig) -> Dict[str, int]:
    d = 5 + int(cfg.hist_bins)
    if bool(cfg.use_balance_features):
        d += 10
    if bool(cfg.use_spike_features):
        d += 4
    tilt_score_idx = -1
    if bool(cfg.use_tilt_detail_features):
        # tilt detail vector size = 13, last element is tilt_score
        tilt_score_idx = d + 12
        d += 13
    return {"total_dim": d, "tilt_score_index": tilt_score_idx}

def compute_feature_vector(
    roi_gray: np.ndarray,
    prev_roi_gray: Optional[np.ndarray],
    hist_bins: int,
    use_balance_features: bool = False,
    balance_feature_gain: float = 1.0,
    use_spike_features: bool = False,
    spike_feature_gain: float = 1.0,
    use_tilt_detail_features: bool = False,
    tilt_gain: float = 1.0
) -> np.ndarray:
    # roi_gray: H x W (uint8)
    # Normalize to [0,1] float
    g = roi_gray.astype(np.float32) / 255.0

    # Basic stats
    mean = float(g.mean())
    std = float(g.std())
    p10 = float(np.percentile(g, 10))
    p90 = float(np.percentile(g, 90))

    # Change energy (frame diff)
    if prev_roi_gray is None:
        diff_energy = 0.0
    else:
        pg = prev_roi_gray.astype(np.float32) / 255.0
        diff_energy = float(np.abs(g - pg).mean())

    # Histogram (density)
    hist = cv2.calcHist([roi_gray], [0], None, [hist_bins], [0, 256]).reshape(-1).astype(np.float32)
    s = float(hist.sum()) + 1e-6
    hist = hist / s

    base_feat = np.concatenate(
        [np.array([diff_energy, mean, std, p10, p90], dtype=np.float32), hist],
        axis=0
    )

    feat = base_feat
    if use_balance_features:
        h, w = g.shape[:2]
        mass = g + 1e-6
        msum = float(mass.sum()) + 1e-6
        xx = np.arange(w, dtype=np.float32)[None, :]
        yy = np.arange(h, dtype=np.float32)[:, None]

        # Intensity-as-height balance descriptors
        cx = float((mass * xx).sum() / msum) / float(max(1, w - 1))
        cy = float((mass * yy).sum() / msum) / float(max(1, h - 1))

        w2 = max(1, w // 2)
        h2 = max(1, h // 2)
        left = float(mass[:, :w2].sum()); right = float(mass[:, w2:].sum())
        top = float(mass[:h2, :].sum()); bottom = float(mass[h2:, :].sum())
        lr_balance = (left - right) / (left + right + 1e-6)
        tb_balance = (top - bottom) / (top + bottom + 1e-6)

        gx0 = float(np.diff(g, axis=1).mean()) if w > 1 else 0.0
        gy0 = float(np.diff(g, axis=0).mean()) if h > 1 else 0.0

        q_ul = float(mass[:h2, :w2].sum()) / msum
        q_ur = float(mass[:h2, w2:].sum()) / msum
        q_ll = float(mass[h2:, :w2].sum()) / msum
        q_lr = float(mass[h2:, w2:].sum()) / msum

        bal = np.array([cx, cy, lr_balance, tb_balance, gx0, gy0, q_ul, q_ur, q_ll, q_lr], dtype=np.float32)
        gain = float(max(0.1, min(10.0, balance_feature_gain)))
        bal = bal * gain
        feat = np.concatenate([feat, bal], axis=0)

    if use_spike_features:
        if prev_roi_gray is None:
            spike = np.zeros((4,), dtype=np.float32)
        else:
            pg = prev_roi_gray.astype(np.float32) / 255.0
            d = np.abs(g - pg)
            # Descriptors for abrupt temporal jumps
            s_mean = float(d.mean())
            s_p95 = float(np.percentile(d, 95))
            s_max = float(d.max())
            s_ratio = float((d > 0.15).mean())
            spike = np.array([s_mean, s_p95, s_max, s_ratio], dtype=np.float32)
        sg = float(max(0.1, min(10.0, spike_feature_gain)))
        feat = np.concatenate([feat, spike * sg], axis=0)

    if use_tilt_detail_features:
        # Detailed tilt descriptors: global, local (quadrant), Sobel orientation and a normalized tilt score.
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        abs_gx = np.abs(gx)
        abs_gy = np.abs(gy)
        mgx = float(abs_gx.mean())
        mgy = float(abs_gy.mean())
        anis = abs(mgx - mgy)
        ratio = abs(mgx - mgy) / (mgx + mgy + 1e-6)

        h, w = g.shape[:2]
        h2 = max(1, h // 2)
        w2 = max(1, w // 2)
        quads = [
            (slice(0, h2), slice(0, w2)),
            (slice(0, h2), slice(w2, w)),
            (slice(h2, h), slice(0, w2)),
            (slice(h2, h), slice(w2, w)),
        ]
        q_feats = []
        for ys, xs in quads:
            q_feats.append(float(np.abs(gx[ys, xs]).mean()))
            q_feats.append(float(np.abs(gy[ys, xs]).mean()))

        sobel_mag = np.sqrt(abs_gx ** 2 + abs_gy ** 2)
        mag_mean = float(sobel_mag.mean())
        mag_p90 = float(np.percentile(sobel_mag, 90))
        tilt_score = float(min(1.0, ratio + 0.5 * min(1.0, mag_p90)))
        tilt_vec = np.array(
            [mgx, mgy, anis, ratio] + q_feats + [mag_mean, mag_p90, tilt_score],
            dtype=np.float32
        )
        tg = float(max(0.1, min(10.0, tilt_gain)))
        feat = np.concatenate([feat, tilt_vec * tg], axis=0)
    return feat

def extract_features_from_video(
    video_path: str,
    roi: Tuple[int, int, int, int],
    fps_target: int,
    hist_bins: int,
    smooth_sec: float,
    use_balance_features: bool = False,
    balance_feature_gain: float = 1.0,
    use_spike_features: bool = False,
    spike_feature_gain: float = 1.0,
    use_adaptive_pixel_smoothing: bool = False,
    pixel_jitter_threshold: float = 0.08,
    pixel_smoothing_kernel: int = 3,
    use_tilt_detail_features: bool = False,
    tilt_gain: float = 1.0,
    log_fn=None
) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    # Original FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-3:
        fps = 30.0

    step = max(int(round(fps / float(fps_target))), 1)
    if log_fn:
        log_fn(f"[extract] {os.path.basename(video_path)} fps={fps:.2f} -> step={step} (target={fps_target})")

    x, y, w, h = roi

    feats = []
    prev_roi = None
    idx = 0
    grabbed = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step != 0:
            idx += 1
            continue
        idx += 1
        grabbed += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        H, W = gray.shape[:2]
        # Clamp ROI to frame size
        xx = max(0, min(x, W - 1))
        yy = max(0, min(y, H - 1))
        ww = max(1, min(w, W - xx))
        hh = max(1, min(h, H - yy))
        roi_gray = gray[yy:yy+hh, xx:xx+ww]

        if use_adaptive_pixel_smoothing:
            rf = roi_gray.astype(np.float32) / 255.0
            # High-frequency residual energy as jitter score
            lp = cv2.GaussianBlur(rf, (3, 3), 0)
            jitter_score = float(np.mean(np.abs(rf - lp)))
            if jitter_score >= float(max(0.0, pixel_jitter_threshold)):
                k = int(max(1, pixel_smoothing_kernel))
                if k % 2 == 0:
                    k += 1
                k = min(k, 11)
                roi_gray = cv2.GaussianBlur(roi_gray, (k, k), 0)

        feat = compute_feature_vector(
            roi_gray,
            prev_roi,
            hist_bins,
            use_balance_features=use_balance_features,
            balance_feature_gain=balance_feature_gain,
            use_spike_features=use_spike_features,
            spike_feature_gain=spike_feature_gain,
            use_tilt_detail_features=use_tilt_detail_features,
            tilt_gain=tilt_gain
        )
        feats.append(feat)
        prev_roi = roi_gray

    cap.release()

    if len(feats) < 4:
        raise RuntimeError(f"Too few sampled frames ({len(feats)}) from {video_path}. Check fps/roi.")

    X = np.stack(feats, axis=0)  # T x D

    # Smoothing
    # k = smooth_sec * fps_target (approx)
    k = int(round(max(1.0, smooth_sec * float(fps_target))))
    X = moving_average(X, k)

    return X.astype(np.float32)

# -----------------------------
# Dataset builder
# -----------------------------
class WindowedTimeSeriesDataset(Dataset):
    def __init__(
        self,
        items: List[Tuple[np.ndarray, int]],
        window_len: int,
        stride: int,
        augment: bool = False,
        noise_std: float = 0.01,
        max_time_shift: int = 4,
        feature_dropout: float = 0.1
    ):
        """
        items: list of (X: T x D, label int [, type_name])
        Produces windows: D x window_len for 1D CNN
        """
        self.window_len = window_len
        self.stride = stride
        self.augment = augment
        self.noise_std = max(0.0, float(noise_std))
        self.max_time_shift = max(0, int(max_time_shift))
        self.feature_dropout = min(max(float(feature_dropout), 0.0), 0.95)
        self.samples: List[Tuple[np.ndarray, int, str]] = []
        for it in items:
            X, y, type_name = parse_item(it)
            T = X.shape[0]
            if T < window_len:
                continue
            for s in range(0, T - window_len + 1, stride):
                w = X[s:s+window_len]  # window_len x D
                self.samples.append((w, y, type_name))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        w, y, _ = self.samples[idx]
        if self.augment:
            w = w.copy()
            # Gaussian noise on all timesteps/features
            if self.noise_std > 0.0:
                n = np.random.normal(0.0, self.noise_std, size=w.shape).astype(np.float32)
                w = w + n

            # Random temporal roll shift
            if self.max_time_shift > 0:
                sh = np.random.randint(-self.max_time_shift, self.max_time_shift + 1)
                if sh != 0:
                    w = np.roll(w, shift=sh, axis=0)

            # Random feature masking (same mask across time in a window)
            if self.feature_dropout > 0.0:
                keep = (np.random.rand(1, w.shape[1]) >= self.feature_dropout).astype(np.float32)
                w = w * keep

        # To torch: (D, T)
        x = torch.from_numpy(w.T.copy())  # D x window_len
        y = torch.tensor(y, dtype=torch.long)
        return x, y

# -----------------------------
# Model (1D CNN)
# -----------------------------
class TemporalCNN(nn.Module):
    def __init__(self, in_ch: int, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool1d(1),  # -> (B, 128, 1)
        )
        self.head = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: (B, D, T)
        z = self.net(x).squeeze(-1)  # (B, 128)
        logits = self.head(z)
        return logits

# -----------------------------
# Train / Eval
# -----------------------------
def split_train_val(items: List[Tuple[np.ndarray, int]], val_ratio=0.2, seed=42):
    rng = np.random.RandomState(seed)
    idx = np.arange(len(items))
    rng.shuffle(idx)
    n_val = int(round(len(items) * val_ratio))
    val_idx = set(idx[:n_val].tolist())
    train, val = [], []
    for i, it in enumerate(items):
        (val if i in val_idx else train).append(it)
    return train, val

def build_stratified_kfold_splits(items: List[Tuple[np.ndarray, int]], k: int, seed=42):
    labels = np.array([int(parse_item(it)[1]) for it in items], dtype=np.int64)
    idx0 = np.where(labels == 0)[0]
    idx1 = np.where(labels == 1)[0]

    rng = np.random.RandomState(seed)
    rng.shuffle(idx0)
    rng.shuffle(idx1)

    folds0 = np.array_split(idx0, k)
    folds1 = np.array_split(idx1, k)

    splits = []
    for fi in range(k):
        val_idx = np.concatenate([folds0[fi], folds1[fi]]).astype(np.int64)
        val_set = set(val_idx.tolist())
        train_items, val_items = [], []
        for i, it in enumerate(items):
            (val_items if i in val_set else train_items).append(it)
        splits.append((train_items, val_items))
    return splits

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = float(gamma)
        self.weight = weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(logits, target, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        fl = ((1.0 - pt) ** self.gamma) * ce
        return fl.mean()

def _group_matrix_np(mat: np.ndarray, out_groups: int, in_groups: int) -> np.ndarray:
    o, i = mat.shape
    o_idx = np.array_split(np.arange(o), max(1, out_groups))
    i_idx = np.array_split(np.arange(i), max(1, in_groups))
    out = np.zeros((len(o_idx), len(i_idx)), dtype=np.float32)
    for oi, og in enumerate(o_idx):
        for ii, ig in enumerate(i_idx):
            out[oi, ii] = float(mat[np.ix_(og, ig)].mean())
    return out

def _param_grad_abs_mean(p: Optional[torch.Tensor]) -> float:
    if p is None or p.grad is None:
        return 0.0
    return float(p.grad.detach().abs().mean().item())

def build_nn_snapshot(
    model: nn.Module,
    in_nodes: int = 10,
    h1_nodes: int = 8,
    h2_nodes: int = 8,
    out_nodes: int = 2
) -> Dict[str, object]:
    st = model.state_dict()
    w1 = st["net.0.weight"].detach().cpu().numpy()  # 64 x in x k
    w2 = st["net.3.weight"].detach().cpu().numpy()  # 128 x 64 x k
    w3 = st["head.weight"].detach().cpu().numpy()   # 2 x 128
    m1 = np.mean(np.abs(w1), axis=2)               # 64 x in
    m2 = np.mean(np.abs(w2), axis=2)               # 128 x 64
    m3 = np.abs(w3)                                # 2 x 128
    g1 = _group_matrix_np(m1, h1_nodes, in_nodes)
    g2 = _group_matrix_np(m2, h2_nodes, h1_nodes)
    g3 = _group_matrix_np(m3, out_nodes, h2_nodes)
    return {
        "g1": g1,
        "g2": g2,
        "g3": g3,
        "grad_conv1": _param_grad_abs_mean(getattr(model.net[0], "weight", None)),
        "grad_conv2": _param_grad_abs_mean(getattr(model.net[3], "weight", None)),
        "grad_head": _param_grad_abs_mean(getattr(model.head, "weight", None)),
    }

def compute_sample_losses(
    logits: torch.Tensor,
    target: torch.Tensor,
    weights: Optional[torch.Tensor],
    loss_name: str,
    cfg: AppConfig
) -> torch.Tensor:
    if loss_name == "focal":
        ce = nn.functional.cross_entropy(logits, target, weight=weights, reduction="none")
        pt = torch.exp(-ce)
        return ((1.0 - pt) ** float(cfg.focal_gamma)) * ce
    return nn.functional.cross_entropy(
        logits,
        target,
        weight=weights,
        reduction="none",
        label_smoothing=float(cfg.label_smoothing),
    )

def reduce_losses(
    losses: torch.Tensor,
    use_hard_mining: bool,
    hard_ratio: float,
    sample_weights: Optional[torch.Tensor] = None
) -> torch.Tensor:
    if sample_weights is not None:
        losses = losses * sample_weights
    if not use_hard_mining or losses.numel() <= 1:
        return losses.mean()
    ratio = min(max(float(hard_ratio), 0.05), 1.0)
    k = max(1, int(round(losses.numel() * ratio)))
    top_vals, _ = torch.topk(losses, k=k, largest=True, sorted=False)
    return top_vals.mean()

def oversample_items_by_type(items, keywords: List[str], factor: int):
    if factor <= 1 or not keywords:
        return items, 0
    out = []
    extra = 0
    keys = [k.strip().lower() for k in keywords if k.strip()]
    for it in items:
        _, _, type_name = parse_item(it)
        out.append(it)
        t = type_name.lower()
        if any(k in t for k in keys):
            for _ in range(max(0, factor - 1)):
                out.append(it)
                extra += 1
    return out, extra

@torch.no_grad()
def find_best_threshold(model: nn.Module, loader: DataLoader, device: str):
    model.eval()
    ys, probs = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        p = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy().tolist()
        probs.extend(p)
        ys.extend(y.numpy().tolist())
    if len(ys) == 0:
        return 0.5, 0.0
    y_true = np.array(ys, dtype=np.int64)
    p_def = np.array(probs, dtype=np.float32)
    best_t = 0.5
    best_bacc = -1.0
    for t in np.arange(0.10, 0.901, 0.01):
        pred = (p_def >= t).astype(np.int64)
        tp = float(np.sum((pred == 1) & (y_true == 1)))
        tn = float(np.sum((pred == 0) & (y_true == 0)))
        fp = float(np.sum((pred == 1) & (y_true == 0)))
        fn = float(np.sum((pred == 0) & (y_true == 1)))
        tpr = tp / max(1.0, tp + fn)
        tnr = tn / max(1.0, tn + fp)
        bacc = 0.5 * (tpr + tnr)
        if bacc > best_bacc:
            best_bacc = bacc
            best_t = float(t)
    return best_t, float(best_bacc)

@torch.no_grad()
def eval_loader(
    model: nn.Module,
    loader: DataLoader,
    weights: Optional[torch.Tensor],
    loss_name: str,
    cfg: AppConfig,
    device: str
) -> Tuple[float, float]:
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        losses = compute_sample_losses(logits, y, weights=weights, loss_name=loss_name, cfg=cfg)
        loss = losses.mean()
        loss_sum += float(loss.item()) * x.size(0)
        pred = torch.argmax(logits, dim=1)
        correct += int((pred == y).sum().item())
        total += x.size(0)
    if total == 0:
        return 0.0, 0.0
    return loss_sum / total, correct / total

def train_model(
    items: List[Tuple[np.ndarray, int]],
    cfg: AppConfig,
    epochs: int,
    batch_size: int,
    lr: float,
    loss_name: str,
    scheduler_name: str,
    use_augmentation: bool,
    k_folds: int,
    progress_cb,
    lr_cb,
    metrics_cb,
    nn_cb,
    log_cb,
    stop_event: threading.Event
):
    device = cfg.device
    D = items[0][0].shape[1]
    layout = feature_layout(cfg)
    tilt_score_idx = int(layout["tilt_score_index"])
    train_t0 = time.time()
    labels = [int(parse_item(it)[1]) for it in items]
    cls_counts = np.bincount(np.array(labels, dtype=np.int64), minlength=2)
    min_class = int(cls_counts.min()) if cls_counts.size > 1 else 0
    k = int(max(1, k_folds))
    if k > 1:
        if min_class < 2:
            log_cb("K-Fold requested but at least one class has <2 videos. Falling back to single split.")
            k = 1
        elif k > min_class:
            log_cb(f"K-Fold={k} is too large for class counts {cls_counts.tolist()}. Using K-Fold={min_class}.")
            k = min_class

    if k > 1:
        splits = build_stratified_kfold_splits(items, k=k, seed=42)
        log_cb(f"Using Stratified {k}-Fold validation.")
    else:
        splits = [split_train_val(items, val_ratio=0.2, seed=42)]
        log_cb("Using single train/val split.")

    # Estimate progress across all folds
    total_steps = 0
    split_windows = []
    for tr_items, va_items in splits:
        ds_tr = WindowedTimeSeriesDataset(
            tr_items, cfg.window_len, cfg.stride,
            augment=use_augmentation,
            noise_std=cfg.aug_noise_std,
            max_time_shift=cfg.aug_time_shift,
            feature_dropout=cfg.aug_feature_dropout
        )
        ds_va = WindowedTimeSeriesDataset(va_items, cfg.window_len, cfg.stride, augment=False)
        split_windows.append((len(ds_tr), len(ds_va)))
        total_steps += max(1, len(ds_tr) // max(1, batch_size) + (1 if len(ds_tr) % max(1, batch_size) else 0)) * epochs

    if total_steps <= 0:
        total_steps = 1
    step_count = 0
    last_hb_t = time.time()
    debug_interval_epochs = max(1, epochs // 100)
    lr_warn_floor = 1e-8
    fold_scores = []
    overall_best_acc = -1.0
    overall_best_state = None
    overall_best_thr = 0.5
    hard_keys = [k_.strip().lower() for k_ in str(cfg.hard_type_keywords).split(",") if k_.strip()]

    for fold_idx, (train_items, val_items) in enumerate(splits, start=1):
        lr_cb({"event": "fold_start", "fold": fold_idx, "fold_total": len(splits)})
        if cfg.use_type_oversampling and hard_keys:
            train_items, extra_n = oversample_items_by_type(
                train_items,
                keywords=hard_keys,
                factor=max(2, int(cfg.type_oversample_factor))
            )
            if extra_n > 0:
                log_cb(f"[Fold {fold_idx}] type oversampling added {extra_n} video-items for keys={hard_keys}")

        ds_train = WindowedTimeSeriesDataset(
            train_items, cfg.window_len, cfg.stride,
            augment=use_augmentation,
            noise_std=cfg.aug_noise_std,
            max_time_shift=cfg.aug_time_shift,
            feature_dropout=cfg.aug_feature_dropout
        )
        ds_val = WindowedTimeSeriesDataset(val_items, cfg.window_len, cfg.stride, augment=False)
        if len(ds_train) == 0 or len(ds_val) == 0:
            raise RuntimeError(f"Not enough windows on fold {fold_idx}. train={len(ds_train)} val={len(ds_val)}.")

        ys = [int(y) for _, y, _ in ds_train.samples]
        c0 = max(1, sum(1 for y in ys if y == 0))
        c1 = max(1, sum(1 for y in ys if y == 1))
        w0 = 1.0 / c0
        w1 = 1.0 / c1
        weights = torch.tensor([w0, w1], dtype=torch.float32, device=device)

        loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)
        loader_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

        model = TemporalCNN(in_ch=D, num_classes=2).to(device)
        opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

        if scheduler_name == "onecycle":
            scheduler = optim.lr_scheduler.OneCycleLR(
                opt,
                max_lr=lr,
                epochs=epochs,
                steps_per_epoch=max(1, len(loader_train))
            )
        elif scheduler_name == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
        else:
            scheduler = None

        best_val_acc = -1.0
        best_state = None
        log_cb(f"[Fold {fold_idx}/{len(splits)}] train_windows={len(ds_train)} val_windows={len(ds_val)}")
        if nn_cb is not None:
            nn_cb({
                "event": "fold_start",
                "phase": "fold_start",
                "fold": fold_idx,
                "fold_total": len(splits),
                "epoch": 0,
                "epochs": epochs,
                "step": step_count,
                "total_steps": total_steps,
                "lr": float(opt.param_groups[0]["lr"]),
                "loss": 0.0,
                **build_nn_snapshot(model),
            })
        last_nn_t = time.time()

        for ep in range(1, epochs + 1):
            if stop_event.is_set():
                log_cb("Training stopped.")
                break

            ep_t0 = time.time()
            model.train()
            ep_loss = 0.0
            ep_total = 0
            ep_correct = 0
            ep_tilt_sum = 0.0
            ep_tilt_n = 0

            for x, y in loader_train:
                if stop_event.is_set():
                    break
                x = x.to(device)
                y = y.to(device)

                opt.zero_grad(set_to_none=True)
                logits = model(x)
                per_sample = compute_sample_losses(logits, y, weights=weights, loss_name=loss_name, cfg=cfg)
                sample_w = None
                if bool(cfg.use_tilt_detail_features) and float(cfg.tilt_loss_alpha) > 0.0 and 0 <= tilt_score_idx < x.size(1):
                    tilt_score = torch.clamp(x[:, tilt_score_idx, :].mean(dim=1), min=0.0, max=1.0)
                    sample_w = 1.0 + float(cfg.tilt_loss_alpha) * tilt_score
                    ep_tilt_sum += float(tilt_score.sum().item())
                    ep_tilt_n += int(tilt_score.numel())
                loss = reduce_losses(
                    per_sample,
                    use_hard_mining=bool(cfg.use_hard_mining),
                    hard_ratio=float(cfg.hard_mining_ratio),
                    sample_weights=sample_w
                )
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite loss detected at fold={fold_idx} epoch={ep} step={step_count+1}")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                opt.step()
                if scheduler is not None and scheduler_name == "onecycle":
                    scheduler.step()

                step_count += 1
                ep_loss += float(loss.item()) * x.size(0)
                ep_total += x.size(0)
                pred = torch.argmax(logits, dim=1)
                ep_correct += int((pred == y).sum().item())

                prog = step_count / max(1, total_steps)
                progress_cb(prog)

                now2 = time.time()
                if nn_cb is not None and (now2 - last_nn_t) >= 0.8:
                    nn_cb({
                        "event": "batch",
                        "phase": "update",
                        "fold": fold_idx,
                        "fold_total": len(splits),
                        "epoch": ep,
                        "epochs": epochs,
                        "step": step_count,
                        "total_steps": total_steps,
                        "lr": float(opt.param_groups[0]["lr"]),
                        "loss": float(loss.item()),
                        **build_nn_snapshot(model),
                    })
                    last_nn_t = now2

                now = time.time()
                if (now - last_hb_t) >= 5.0:
                    cur_lr = float(opt.param_groups[0]["lr"])
                    elapsed = now - train_t0
                    sps = step_count / max(1e-6, elapsed)
                    eta = (total_steps - step_count) / max(1e-6, sps)
                    log_cb(
                        f"[HB] fold={fold_idx}/{len(splits)} ep={ep}/{epochs} "
                        f"step={step_count}/{total_steps} lr={cur_lr:.10f} "
                        f"sps={sps:.2f} eta={eta/60.0:.1f}m"
                    )
                    last_hb_t = now

            if scheduler is not None and scheduler_name == "cosine":
                scheduler.step()

            train_loss = ep_loss / max(1, ep_total)
            train_acc = ep_correct / max(1, ep_total)
            tilt_mean = (ep_tilt_sum / max(1, ep_tilt_n)) if ep_tilt_n > 0 else 0.0
            val_loss, val_acc = eval_loader(model, loader_val, weights=weights, loss_name=loss_name, cfg=cfg, device=device)
            metrics_cb({
                "fold": fold_idx,
                "fold_total": len(splits),
                "epoch": ep,
                "epochs": epochs,
                "train_loss": float(train_loss),
                "train_acc": float(train_acc),
                "val_loss": float(val_loss),
                "val_acc": float(val_acc),
                "best_val_acc": float(max(best_val_acc, val_acc)),
                "lr": float(opt.param_groups[0]["lr"]),
                "tilt_mean": float(tilt_mean),
            })
            lr_cb({"event": "epoch", "epoch": ep, "epochs": epochs, "lr": float(opt.param_groups[0]["lr"])})
            if nn_cb is not None:
                nn_cb({
                    "event": "epoch_end",
                    "phase": "epoch_end",
                    "fold": fold_idx,
                    "fold_total": len(splits),
                    "epoch": ep,
                    "epochs": epochs,
                    "step": step_count,
                    "total_steps": total_steps,
                    "lr": float(opt.param_groups[0]["lr"]),
                    "loss": float(train_loss),
                    **build_nn_snapshot(model),
                })

            if (ep == 1) or (ep == epochs) or (ep % debug_interval_epochs == 0):
                cur_lr = float(opt.param_groups[0]["lr"])
                ep_sec = time.time() - ep_t0
                elapsed = time.time() - train_t0
                done_ratio = step_count / max(1, total_steps)
                eta_sec = (elapsed / max(1e-6, done_ratio)) * (1.0 - done_ratio) if done_ratio > 0 else 0.0
                log_cb(
                    f"[DBG] fold={fold_idx}/{len(splits)} ep={ep}/{epochs} "
                    f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
                    f"lr={cur_lr:.10f} tilt_mean={tilt_mean:.4f} ep_time={ep_sec:.2f}s eta={eta_sec/60.0:.1f}m"
                )
            if float(opt.param_groups[0]["lr"]) < lr_warn_floor:
                log_cb(
                    f"[WARN] LR is very small ({float(opt.param_groups[0]['lr']):.10f}) at "
                    f"fold={fold_idx} epoch={ep}. Training may appear stagnant."
                )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k_: v.detach().cpu().clone() for k_, v in model.state_dict().items()}

        best_thr = 0.5
        if bool(cfg.auto_threshold):
            best_thr, thr_bacc = find_best_threshold(model, loader_val, device=device)
            log_cb(f"[Fold {fold_idx}] auto-threshold={best_thr:.2f} (val_bal_acc={thr_bacc:.3f})")

        fold_scores.append(best_val_acc)
        if best_state is not None and best_val_acc > overall_best_acc:
            overall_best_acc = best_val_acc
            overall_best_state = best_state
            overall_best_thr = best_thr

        if stop_event.is_set():
            break

    if overall_best_state is None:
        raise RuntimeError("Training ended without a valid model state.")

    final_model = TemporalCNN(in_ch=D, num_classes=2).to(device)
    final_model.load_state_dict(overall_best_state)
    cfg.infer_threshold = float(overall_best_thr if cfg.auto_threshold else 0.5)
    mean_acc = float(np.mean(fold_scores)) if fold_scores else 0.0
    log_cb(f"K-Fold mean best_val_acc={mean_acc:.3f} over {len(fold_scores)} fold(s)")
    log_cb(f"Inference threshold set to {cfg.infer_threshold:.2f}")
    return final_model, mean_acc

def search_hparams(
    items: List[Tuple[np.ndarray, int]],
    base_cfg: AppConfig,
    base_epochs: int,
    base_batch_size: int,
    base_lr: float,
    trials: int,
    stop_event: threading.Event,
    log_cb,
):
    rng = np.random.RandomState(2026)
    trials = max(1, int(trials))
    tune_epochs = max(4, min(14, int(round(base_epochs * 0.35))))

    # Candidate space (lightweight random search)
    win_candidates = sorted(list({max(16, int(base_cfg.window_len * r)) for r in (0.5, 0.75, 1.0, 1.25)}))
    str_ratio = [0.25, 0.5, 0.75]
    loss_candidates = ["cross_entropy_ls", "focal"]
    sched_candidates = ["cosine", "onecycle", "none"]

    best = {
        "score": -1.0,
        "params": {
            "lr": float(base_lr),
            "window_len": int(base_cfg.window_len),
            "stride": int(base_cfg.stride),
            "loss_name": str(base_cfg.loss_name),
            "scheduler_name": str(base_cfg.scheduler_name),
        }
    }

    # First trial uses current settings as baseline.
    candidates = [best["params"].copy()]
    for _ in range(max(0, trials - 1)):
        wl = int(rng.choice(win_candidates))
        st = max(1, int(round(wl * float(rng.choice(str_ratio)))))
        cand = {
            "lr": float(10 ** rng.uniform(-4.2, -2.5)),
            "window_len": wl,
            "stride": st,
            "loss_name": str(rng.choice(loss_candidates)),
            "scheduler_name": str(rng.choice(sched_candidates)),
        }
        candidates.append(cand)

    log_cb(f"[HPO] start random-search trials={len(candidates)} tune_epochs={tune_epochs}")
    for i, cand in enumerate(candidates, start=1):
        if stop_event.is_set():
            log_cb("[HPO] stopped by user.")
            break
        cfg = AppConfig(**asdict(base_cfg))
        cfg.window_len = int(cand["window_len"])
        cfg.stride = int(cand["stride"])
        try:
            _, score = train_model(
                items=items,
                cfg=cfg,
                epochs=tune_epochs,
                batch_size=base_batch_size,
                lr=float(cand["lr"]),
                loss_name=str(cand["loss_name"]),
                scheduler_name=str(cand["scheduler_name"]),
                use_augmentation=cfg.use_augmentation,
                k_folds=cfg.k_folds,
                progress_cb=lambda _p: None,
                lr_cb=lambda _ev: None,
                metrics_cb=lambda _m: None,
                nn_cb=None,
                log_cb=lambda _s: None,  # keep HPO logs concise
                stop_event=stop_event,
            )
        except Exception as e:
            log_cb(f"[HPO] trial {i}/{len(candidates)} failed: {e}")
            continue

        log_cb(
            f"[HPO] trial {i}/{len(candidates)} score={score:.4f} "
            f"lr={cand['lr']:.6f} win={cand['window_len']} stride={cand['stride']} "
            f"loss={cand['loss_name']} sched={cand['scheduler_name']}"
        )
        if score > float(best["score"]):
            best["score"] = float(score)
            best["params"] = cand.copy()

    log_cb(
        f"[HPO] best score={best['score']:.4f} "
        f"lr={best['params']['lr']:.6f} win={best['params']['window_len']} stride={best['params']['stride']} "
        f"loss={best['params']['loss_name']} sched={best['params']['scheduler_name']}"
    )
    return best

# -----------------------------
# Inference
# -----------------------------
@torch.no_grad()
def infer_video(model: nn.Module, X: np.ndarray, cfg: AppConfig) -> Dict[str, float]:
    """
    X: T x D features
    Returns: dict with prob_defect, prob_normal, score_defect, verdict
    """
    device = cfg.device
    model.eval()

    T, D = X.shape
    if T < cfg.window_len:
        # Pad by edge
        pad = cfg.window_len - T
        Xp = np.pad(X, ((0, pad), (0, 0)), mode="edge")
        T = Xp.shape[0]
    else:
        Xp = X

    windows = []
    for s in range(0, T - cfg.window_len + 1, cfg.stride):
        w = Xp[s:s+cfg.window_len].T  # D x L
        windows.append(w)

    if not windows:
        w = Xp[:cfg.window_len].T
        windows = [w]

    xb = torch.from_numpy(np.stack(windows, axis=0)).float().to(device)  # B x D x L
    logits = model(xb)
    probs = torch.softmax(logits, dim=1)  # B x 2
    # average across windows
    p = probs.mean(dim=0).detach().cpu().numpy()
    p_normal = float(p[0])
    p_defect = float(p[1])
    thr = float(min(max(cfg.infer_threshold, 0.01), 0.99))
    verdict = "DEFECT" if p_defect >= thr else "NORMAL"
    return {
        "prob_normal": p_normal,
        "prob_defect": p_defect,
        "verdict": verdict,
        "num_windows": float(len(windows)),
        "threshold": thr,
    }


# ROI selector helper (replacement for cv2.selectROI)
def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def select_roi_with_seek(video_path: str, init_roi=None, win_name="ROI Select (A/D/J/L/Z/C, Space, Enter, R, ESC)"):
    """
    Returns: (x, y, w, h) or None
    - Red thick rectangle ROI
    - Frame seek (back/forward) + play/pause
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if fps <= 1e-3:
        fps = 30.0

    # Read first frame to get size
    ok, frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("Cannot read frames from video.")

    H, W = frame.shape[:2]
    cur = 0

    # ROI state in absolute coords (x1,y1,x2,y2)
    roi = None
    if init_roi is not None:
        x, y, w, h = init_roi
        roi = (x, y, x + w, y + h)

    dragging = False
    x0 = y0 = 0
    play = False
    last_tick = time.time()

    def set_pos(frame_idx: int):
        nonlocal cur
        if total > 0:
            frame_idx = _clamp(frame_idx, 0, total - 1)
        else:
            frame_idx = max(frame_idx, 0)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        cur = frame_idx

    def read_at_current():
        nonlocal frame
        ok2, fr = cap.read()
        if not ok2:
            return False, None
        return True, fr

    def on_mouse(event, x, y, flags, param):
        nonlocal dragging, x0, y0, roi
        if event == cv2.EVENT_LBUTTONDOWN:
            dragging = True
            x0, y0 = x, y
            roi = (x0, y0, x0, y0)
        elif event == cv2.EVENT_MOUSEMOVE and dragging:
            roi = (x0, y0, x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            dragging = False
            roi = (x0, y0, x, y)

    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win_name, on_mouse)

    # Reset to first frame already read
    set_pos(0)
    ok, frame = read_at_current()
    if not ok:
        cap.release()
        cv2.destroyWindow(win_name)
        return None

    # Helper: draw overlay
    def draw_overlay(fr):
        disp = fr.copy()

        # Draw ROI in thick red
        if roi is not None:
            x1, y1, x2, y2 = roi
            x1, x2 = sorted([int(x1), int(x2)])
            y1, y2 = sorted([int(y1), int(y2)])
            x1 = _clamp(x1, 0, W-1); x2 = _clamp(x2, 0, W-1)
            y1 = _clamp(y1, 0, H-1); y2 = _clamp(y2, 0, H-1)
            if x2 > x1 and y2 > y1:
                cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 0, 255), thickness=4)

        # HUD text
        t_sec = cur / fps
        if total > 0:
            dur = total / fps
            hud = f"Frame {cur}/{total-1}  Time {t_sec:.2f}s / {dur:.2f}s   Play={play}"
        else:
            hud = f"Frame {cur}  Time {t_sec:.2f}s   Play={play}"
        cv2.putText(disp, hud, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(disp, "Drag ROI | A/D:1f  J/L:30f  Z/C:300f  Space:Play  Enter:OK  R:Reset  ESC:Cancel",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return disp

    # Main loop
    while True:
        # Auto play
        if play:
            now = time.time()
            if (now - last_tick) >= (1.0 / fps):
                last_tick = now
                set_pos(cur + 1)
                ok, frame = read_at_current()
                if not ok:
                    play = False
                    # stay at last valid frame if possible
                    if total > 0:
                        set_pos(total - 1)
                        ok, frame = read_at_current()

        disp = draw_overlay(frame)
        cv2.imshow(win_name, disp)

        key = cv2.waitKey(10) & 0xFF

        if key == 27:  # ESC
            cap.release()
            cv2.destroyWindow(win_name)
            return None

        if key in (13, 10):  # Enter
            if roi is None:
                continue
            x1, y1, x2, y2 = roi
            x1, x2 = sorted([int(x1), int(x2)])
            y1, y2 = sorted([int(y1), int(y2)])
            x1 = _clamp(x1, 0, W-1); x2 = _clamp(x2, 0, W-1)
            y1 = _clamp(y1, 0, H-1); y2 = _clamp(y2, 0, H-1)
            if x2 <= x1 or y2 <= y1:
                continue
            cap.release()
            cv2.destroyWindow(win_name)
            return (x1, y1, x2 - x1, y2 - y1)

        if key == ord(' '):  # space
            play = not play
            last_tick = time.time()

        if key in (ord('r'), ord('R')):
            roi = None

        # Seek controls
        if key in (ord('a'), ord('A')):
            play = False
            set_pos(cur - 1)
            ok, frame = read_at_current()
        elif key in (ord('d'), ord('D')):
            play = False
            set_pos(cur + 1)
            ok, frame = read_at_current()
        elif key in (ord('j'), ord('J')):
            play = False
            set_pos(cur - 30)
            ok, frame = read_at_current()
        elif key in (ord('l'), ord('L')):
            play = False
            set_pos(cur + 30)
            ok, frame = read_at_current()
        elif key in (ord('z'), ord('Z')):
            play = False
            set_pos(cur - 300)
            ok, frame = read_at_current()
        elif key in (ord('c'), ord('C')):
            play = False
            set_pos(cur + 300)
            ok, frame = read_at_current()

# -----------------------------
# GUI
# -----------------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("IR Temporal Trainer (ROI Drag + Batch Train + Save/Load + Infer)")
        self.geometry("1160x740")
        try:
            self.state("zoomed")
        except Exception:
            pass

        self.cfg = AppConfig()
        self.model: Optional[nn.Module] = None
        self.model_path: Optional[str] = None

        self.msg_q: "queue.Queue[Tuple[str, object]]" = queue.Queue()
        self.stop_event = threading.Event()
        self.profiles_dir = os.path.join(os.getcwd(), "profiles")
        self.lr_history: List[Tuple[int, float]] = []
        self.lr_epochs_total: int = 1
        self.lr_fold_label: str = "-"
        self.lr_plot_planned_max: float = 1e-3
        self.lr_y_linear_bounds: Optional[Tuple[float, float]] = None
        self.lr_info_var = tk.StringVar(value="LR info: no data")
        self.nn_live: Optional[Dict[str, object]] = None
        self.nn_info_var = tk.StringVar(value="Network: (not initialized)")
        self.metric_vars: Dict[str, tk.StringVar] = {
            "fold": tk.StringVar(value="-"),
            "epoch": tk.StringVar(value="-"),
            "train_loss": tk.StringVar(value="-"),
            "train_acc": tk.StringVar(value="-"),
            "val_loss": tk.StringVar(value="-"),
            "val_acc": tk.StringVar(value="-"),
            "best_val_acc": tk.StringVar(value="-"),
            "lr": tk.StringVar(value="-"),
            "tilt_mean": tk.StringVar(value="-"),
            "threshold": tk.StringVar(value=f"{self.cfg.infer_threshold:.2f}"),
            "train_target": tk.StringVar(value=self._train_mode_text(self.cfg.train_class_mode)),
        }

        self._build_ui()
        self._refresh_profile_list()
        self.after(100, self._poll_queue)

    def _build_ui(self):
        # Neural network diagram (top)
        netf = ttk.Labelframe(self, text="Neural Network Structure")
        netf.pack(fill="x", padx=10, pady=(10, 6))
        self.nn_canvas = tk.Canvas(netf, width=1120, height=210, bg="#f8fafc", highlightthickness=1, highlightbackground="#cfd8e3")
        self.nn_canvas.pack(fill="x", padx=6, pady=6)
        ttk.Label(netf, textvariable=self.nn_info_var, anchor="w", justify="left").pack(fill="x", padx=6, pady=(0, 6))
        self.nn_canvas.bind("<Configure>", lambda _e: self._draw_nn_graph())

        top = ttk.Frame(self)
        top.pack(fill="x", padx=10, pady=(0, 8))
        top.grid_columnconfigure(0, weight=1)
        top.grid_columnconfigure(1, weight=1)
        top.grid_rowconfigure(0, weight=1)
        top.grid_rowconfigure(1, weight=1)
        self.var_lr_log_scale = tk.BooleanVar(value=True)

        left_col = ttk.Frame(top)
        left_col.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0, 6), pady=0)
        left_col.grid_columnconfigure(0, weight=3)
        left_col.grid_columnconfigure(1, weight=2)
        left_col.grid_rowconfigure(0, weight=1)
        left_col.grid_rowconfigure(1, weight=1)

        # ROI & dataset controls
        roi_frame = ttk.Labelframe(left_col, text="ROI / Dataset")
        roi_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 6), pady=(0, 6))
        roi_frame.grid_columnconfigure(0, weight=1)
        roi_frame.grid_columnconfigure(1, weight=1)

        ttk.Button(roi_frame, text="Set ROI (drag on a sample video)", command=self.on_set_roi).grid(row=0, column=0, padx=6, pady=6, sticky="w")
        self.roi_label = ttk.Label(roi_frame, text="ROI: (not set)")
        self.roi_label.grid(row=0, column=1, padx=6, pady=6, sticky="w")

        ttk.Button(roi_frame, text="Select Dataset Root Folder", command=self.on_select_dataset).grid(row=1, column=0, padx=6, pady=6, sticky="w")
        self.ds_label = ttk.Label(roi_frame, text="Dataset: (not set)")
        self.ds_label.grid(row=1, column=1, padx=6, pady=6, sticky="w")
        metf = ttk.Labelframe(roi_frame, text="Training Metrics")
        metf.grid(row=2, column=0, columnspan=2, padx=6, pady=6, sticky="ew")
        metf.grid_columnconfigure(0, weight=1)
        metf.grid_columnconfigure(1, weight=1)
        metric_rows = [
            ("Fold", "fold"),
            ("Epoch", "epoch"),
            ("Train Loss", "train_loss"),
            ("Train Acc", "train_acc"),
            ("Val Loss", "val_loss"),
            ("Val Acc", "val_acc"),
            ("Best Val Acc", "best_val_acc"),
            ("LR", "lr"),
            ("Tilt Mean", "tilt_mean"),
            ("Threshold", "threshold"),
            ("Train Target", "train_target"),
        ]
        for r, (title, key) in enumerate(metric_rows):
            ttk.Label(metf, text=title).grid(row=r, column=0, padx=8, pady=2, sticky="w")
            ttk.Label(metf, textvariable=self.metric_vars[key]).grid(row=r, column=1, padx=8, pady=2, sticky="e")

        # LR curve (small, beside ROI/metrics)
        lrf = ttk.Labelframe(left_col, text="Learning Rate Curve")
        lrf.grid(row=0, column=1, sticky="nsew", padx=(6, 0), pady=(0, 6))
        self.lr_canvas = tk.Canvas(lrf, width=320, height=180, bg="white", highlightthickness=1, highlightbackground="#cccccc")
        self.lr_canvas.pack(fill="both", expand=True, padx=6, pady=6)
        ttk.Checkbutton(
            lrf,
            text="log-scale y (for long training)",
            variable=self.var_lr_log_scale,
            command=self._draw_lr_plot
        ).pack(anchor="w", padx=6, pady=(0, 2))
        ttk.Label(lrf, textvariable=self.lr_info_var, anchor="w", justify="left").pack(fill="x", padx=6, pady=(0, 6))
        self._draw_lr_plot()

        # Params (scrollable)
        param_frame = ttk.Labelframe(top, text="Parameters")
        param_frame.grid(row=0, column=1, rowspan=2, sticky="nsew", padx=(6, 0), pady=0)
        param_canvas = tk.Canvas(param_frame, width=360, height=430, highlightthickness=0)
        param_canvas.pack(side="left", fill="both", expand=True, padx=(0, 2), pady=2)
        param_scroll = ttk.Scrollbar(param_frame, orient="vertical", command=param_canvas.yview)
        param_scroll.pack(side="right", fill="y", pady=2)
        param_canvas.configure(yscrollcommand=param_scroll.set)
        param_inner = ttk.Frame(param_canvas)
        param_win = param_canvas.create_window((0, 0), window=param_inner, anchor="nw")

        def _on_param_inner_config(_e):
            param_canvas.configure(scrollregion=param_canvas.bbox("all"))

        def _on_param_canvas_config(e):
            param_canvas.itemconfigure(param_win, width=e.width)

        param_inner.bind("<Configure>", _on_param_inner_config)
        param_canvas.bind("<Configure>", _on_param_canvas_config)

        self.var_fps = tk.IntVar(value=self.cfg.fps_target)
        self.var_bins = tk.IntVar(value=self.cfg.hist_bins)
        self.var_win = tk.IntVar(value=self.cfg.window_len)
        self.var_stride = tk.IntVar(value=self.cfg.stride)
        self.var_epochs = tk.IntVar(value=12)
        self.var_bs = tk.IntVar(value=64)
        self.var_lr = tk.DoubleVar(value=1e-3)
        self.var_use_aug = tk.BooleanVar(value=self.cfg.use_augmentation)
        self.var_loss = tk.StringVar(value=self.cfg.loss_name)
        self.var_sched = tk.StringVar(value=self.cfg.scheduler_name)
        self.var_train_class_mode = tk.StringVar(value=self.cfg.train_class_mode)
        self.var_kfold = tk.IntVar(value=self.cfg.k_folds)
        self.var_use_hard_mining = tk.BooleanVar(value=self.cfg.use_hard_mining)
        self.var_hard_ratio = tk.DoubleVar(value=self.cfg.hard_mining_ratio)
        self.var_use_type_oversampling = tk.BooleanVar(value=self.cfg.use_type_oversampling)
        self.var_hard_keywords = tk.StringVar(value=self.cfg.hard_type_keywords)
        self.var_type_os_factor = tk.IntVar(value=self.cfg.type_oversample_factor)
        self.var_auto_threshold = tk.BooleanVar(value=self.cfg.auto_threshold)
        self.var_use_balance = tk.BooleanVar(value=self.cfg.use_balance_features)
        self.var_balance_gain = tk.DoubleVar(value=self.cfg.balance_feature_gain)
        self.var_use_spike_feat = tk.BooleanVar(value=self.cfg.use_spike_features)
        self.var_spike_gain = tk.DoubleVar(value=self.cfg.spike_feature_gain)
        self.var_use_adapt_smooth = tk.BooleanVar(value=self.cfg.use_adaptive_pixel_smoothing)
        self.var_jitter_th = tk.DoubleVar(value=self.cfg.pixel_jitter_threshold)
        self.var_smooth_k = tk.IntVar(value=self.cfg.pixel_smoothing_kernel)
        self.var_use_tilt_detail = tk.BooleanVar(value=self.cfg.use_tilt_detail_features)
        self.var_tilt_gain = tk.DoubleVar(value=self.cfg.tilt_gain)
        self.var_tilt_loss_alpha = tk.DoubleVar(value=self.cfg.tilt_loss_alpha)
        self.var_auto_hpo = tk.BooleanVar(value=self.cfg.auto_hparam_search)
        self.var_hpo_trials = tk.IntVar(value=self.cfg.hparam_trials)

        for v in (
            self.var_bins,
            self.var_win,
            self.var_use_balance,
            self.var_balance_gain,
            self.var_use_spike_feat,
            self.var_spike_gain,
            self.var_use_tilt_detail,
            self.var_tilt_gain,
        ):
            v.trace_add("write", lambda *_args: self._draw_nn_graph())

        for cc in (0, 1, 2, 3):
            param_inner.grid_columnconfigure(cc, weight=1 if cc in (1, 3) else 0)

        def row2(r, l_label, l_var, r_label, r_var):
            ttk.Label(param_inner, text=l_label).grid(row=r, column=0, padx=6, pady=4, sticky="e")
            ttk.Entry(param_inner, textvariable=l_var, width=10).grid(row=r, column=1, padx=6, pady=4, sticky="w")
            ttk.Label(param_inner, text=r_label).grid(row=r, column=2, padx=6, pady=4, sticky="e")
            ttk.Entry(param_inner, textvariable=r_var, width=10).grid(row=r, column=3, padx=6, pady=4, sticky="w")

        row2(0, "fps_target", self.var_fps, "hist_bins", self.var_bins)
        row2(1, "window_len", self.var_win, "stride", self.var_stride)
        row2(2, "epochs", self.var_epochs, "batch_size", self.var_bs)
        row2(3, "lr", self.var_lr, "k_folds", self.var_kfold)

        ttk.Label(param_inner, text="loss").grid(row=4, column=0, padx=6, pady=4, sticky="e")
        ttk.Combobox(
            param_inner,
            textvariable=self.var_loss,
            values=("cross_entropy_ls", "focal"),
            state="readonly",
            width=14
        ).grid(row=4, column=1, padx=6, pady=4, sticky="w")
        ttk.Label(param_inner, text="scheduler").grid(row=4, column=2, padx=6, pady=4, sticky="e")
        ttk.Combobox(
            param_inner,
            textvariable=self.var_sched,
            values=("none", "cosine", "onecycle"),
            state="readonly",
            width=14
        ).grid(row=4, column=3, padx=6, pady=4, sticky="w")

        ttk.Checkbutton(param_inner, text="use_augmentation", variable=self.var_use_aug).grid(row=5, column=0, columnspan=2, padx=6, pady=4, sticky="w")
        ttk.Checkbutton(param_inner, text="use_hard_mining", variable=self.var_use_hard_mining).grid(row=5, column=2, columnspan=2, padx=6, pady=4, sticky="w")

        row2(6, "hard_ratio", self.var_hard_ratio, "type_os_factor", self.var_type_os_factor)

        ttk.Checkbutton(param_inner, text="use_type_oversampling", variable=self.var_use_type_oversampling).grid(row=7, column=0, columnspan=2, padx=6, pady=4, sticky="w")
        ttk.Checkbutton(param_inner, text="auto_threshold", variable=self.var_auto_threshold).grid(row=7, column=2, columnspan=2, padx=6, pady=4, sticky="w")

        ttk.Label(param_inner, text="hard_types(csv)").grid(row=8, column=0, padx=6, pady=4, sticky="e")
        ttk.Entry(param_inner, textvariable=self.var_hard_keywords, width=28).grid(row=8, column=1, columnspan=3, padx=6, pady=4, sticky="we")

        ttk.Checkbutton(param_inner, text="use_balance_features", variable=self.var_use_balance).grid(row=9, column=0, columnspan=2, padx=6, pady=4, sticky="w")
        ttk.Checkbutton(param_inner, text="use_spike_features", variable=self.var_use_spike_feat).grid(row=9, column=2, columnspan=2, padx=6, pady=4, sticky="w")

        row2(10, "balance_gain", self.var_balance_gain, "spike_gain", self.var_spike_gain)

        ttk.Checkbutton(param_inner, text="adaptive_pixel_smoothing", variable=self.var_use_adapt_smooth).grid(row=11, column=0, columnspan=2, padx=6, pady=4, sticky="w")
        row2(12, "jitter_threshold", self.var_jitter_th, "smooth_kernel", self.var_smooth_k)
        ttk.Checkbutton(param_inner, text="use_tilt_detail_features", variable=self.var_use_tilt_detail).grid(row=13, column=0, columnspan=2, padx=6, pady=4, sticky="w")
        row2(14, "tilt_gain", self.var_tilt_gain, "tilt_loss_alpha", self.var_tilt_loss_alpha)
        ttk.Checkbutton(param_inner, text="auto_hparam_search", variable=self.var_auto_hpo).grid(row=15, column=0, columnspan=2, padx=6, pady=4, sticky="w")
        ttk.Label(param_inner, text="hpo_trials").grid(row=15, column=2, padx=6, pady=4, sticky="e")
        ttk.Entry(param_inner, textvariable=self.var_hpo_trials, width=10).grid(row=15, column=3, padx=6, pady=4, sticky="w")
        ttk.Label(param_inner, text="train_class").grid(row=16, column=0, padx=6, pady=4, sticky="e")
        ttk.Combobox(
            param_inner,
            textvariable=self.var_train_class_mode,
            values=("both", "normal", "defect"),
            state="readonly",
            width=14
        ).grid(row=16, column=1, padx=6, pady=4, sticky="w")

        # Log box (below ROI, same width as ROI column)
        log_mid = ttk.Labelframe(left_col, text="Log / Status")
        log_mid.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=0, pady=(6, 0))
        self.txt = tk.Text(log_mid, height=7)
        self.txt.pack(fill="both", expand=True, padx=6, pady=6)

        # Profile manager
        prof = ttk.Labelframe(self, text="Profiles (JSON)")
        prof.pack(fill="x", padx=10, pady=(0, 6))

        left = ttk.Frame(prof)
        left.pack(side="left", fill="both", expand=True, padx=6, pady=6)
        self.profile_list = tk.Listbox(left, height=6, exportselection=False)
        self.profile_list.pack(side="left", fill="both", expand=True)
        scroll = ttk.Scrollbar(left, orient="vertical", command=self.profile_list.yview)
        scroll.pack(side="right", fill="y")
        self.profile_list.config(yscrollcommand=scroll.set)

        right = ttk.Frame(prof)
        right.pack(side="left", fill="y", padx=6, pady=6)
        ttk.Button(right, text="Save Current", command=self.on_profile_save).pack(fill="x", pady=2)
        ttk.Button(right, text="Load Selected", command=self.on_profile_load).pack(fill="x", pady=2)
        ttk.Button(right, text="Delete Selected", command=self.on_profile_delete).pack(fill="x", pady=2)
        ttk.Button(right, text="Refresh List", command=self._refresh_profile_list).pack(fill="x", pady=2)

        # Train/Save/Load/Infer controls
        mid = ttk.Frame(self)
        mid.pack(fill="x", padx=10)

        ttk.Button(mid, text="Train", command=self.on_train).pack(side="left", padx=6, pady=6)
        ttk.Button(mid, text="Reset Train Options", command=self.on_reset_train_options).pack(side="left", padx=6, pady=6)
        ttk.Button(mid, text="Help", command=self.on_show_help).pack(side="left", padx=6, pady=6)
        ttk.Button(mid, text="Stop", command=self.on_stop).pack(side="left", padx=6, pady=6)
        ttk.Button(mid, text="Save Model", command=self.on_save_model).pack(side="left", padx=6, pady=6)
        ttk.Button(mid, text="Load Model", command=self.on_load_model).pack(side="left", padx=6, pady=6)
        ttk.Button(mid, text="Infer New Video", command=self.on_infer).pack(side="left", padx=6, pady=6)
        ttk.Button(mid, text="Infer Folder", command=self.on_infer_folder).pack(side="left", padx=6, pady=6)

        self.progress = ttk.Progressbar(mid, length=320, mode="determinate")
        self.progress.pack(side="right", padx=6, pady=6)
        self.progress["value"] = 0

        self.dataset_root: Optional[str] = None
        self._draw_nn_graph()

    def _estimate_input_channels(self) -> int:
        tmp = AppConfig(**asdict(self.cfg))
        tmp.hist_bins = max(1, self._safe_int_var(self.var_bins, self.cfg.hist_bins))
        tmp.use_balance_features = bool(self.var_use_balance.get()) if hasattr(self, "var_use_balance") else tmp.use_balance_features
        tmp.use_spike_features = bool(self.var_use_spike_feat.get()) if hasattr(self, "var_use_spike_feat") else tmp.use_spike_features
        tmp.use_tilt_detail_features = bool(self.var_use_tilt_detail.get()) if hasattr(self, "var_use_tilt_detail") else tmp.use_tilt_detail_features
        return int(feature_layout(tmp)["total_dim"])

    def _safe_int_var(self, var: tk.Variable, default: int) -> int:
        try:
            return int(var.get())
        except Exception:
            return int(default)

    def _safe_float_var(self, var: tk.Variable, default: float) -> float:
        try:
            return float(var.get())
        except Exception:
            return float(default)

    def _train_mode_text(self, mode: Optional[str] = None) -> str:
        m = str(mode if mode is not None else self.cfg.train_class_mode).strip().lower()
        if m == "normal":
            return "normal only"
        if m == "defect":
            return "defect only"
        return "normal + defect"

    def _group_matrix(self, mat: np.ndarray, out_groups: int, in_groups: int) -> np.ndarray:
        # mat: out x in
        o, i = mat.shape
        o_idx = np.array_split(np.arange(o), max(1, out_groups))
        i_idx = np.array_split(np.arange(i), max(1, in_groups))
        out = np.zeros((len(o_idx), len(i_idx)), dtype=np.float32)
        for oi, og in enumerate(o_idx):
            for ii, ig in enumerate(i_idx):
                out[oi, ii] = float(mat[np.ix_(og, ig)].mean())
        return out

    def _build_weight_graph(self, in_nodes: int, h1_nodes: int, h2_nodes: int, out_nodes: int):
        # Returns three adjacency matrices: h1 x in, h2 x h1, out x h2
        if self.nn_live is not None:
            try:
                g1 = np.array(self.nn_live.get("g1"), dtype=np.float32)
                g2 = np.array(self.nn_live.get("g2"), dtype=np.float32)
                g3 = np.array(self.nn_live.get("g3"), dtype=np.float32)
                if g1.shape == (h1_nodes, in_nodes) and g2.shape == (h2_nodes, h1_nodes) and g3.shape == (out_nodes, h2_nodes):
                    return g1, g2, g3, "weights=live"
            except Exception:
                pass

        if self.model is not None:
            try:
                st = self.model.state_dict()
                w1 = st["net.0.weight"].detach().cpu().numpy()  # 64 x in x k
                w2 = st["net.3.weight"].detach().cpu().numpy()  # 128 x 64 x k
                w3 = st["head.weight"].detach().cpu().numpy()   # 2 x 128
                m1 = np.mean(np.abs(w1), axis=2)               # 64 x in
                m2 = np.mean(np.abs(w2), axis=2)               # 128 x 64
                m3 = np.abs(w3)                                # 2 x 128
                g1 = self._group_matrix(m1, h1_nodes, in_nodes)
                g2 = self._group_matrix(m2, h2_nodes, h1_nodes)
                g3 = self._group_matrix(m3, out_nodes, h2_nodes)
                return g1, g2, g3, "weights=actual"
            except Exception:
                pass

        # Fallback schematic weights when model is not available.
        rng = np.random.RandomState(7 + int(in_nodes))
        g1 = rng.uniform(0.2, 1.0, size=(h1_nodes, in_nodes)).astype(np.float32)
        g2 = rng.uniform(0.2, 1.0, size=(h2_nodes, h1_nodes)).astype(np.float32)
        g3 = rng.uniform(0.2, 1.0, size=(out_nodes, h2_nodes)).astype(np.float32)
        return g1, g2, g3, "weights=example"

    def _draw_nn_graph(self):
        if not hasattr(self, "nn_canvas"):
            return
        c = self.nn_canvas
        c.delete("all")
        w = max(200, int(c.winfo_width()) if c.winfo_width() > 1 else 1120)
        h = max(160, int(c.winfo_height()) if c.winfo_height() > 1 else 210)
        c.create_rectangle(0, 0, w, h, fill="#f8fafc", outline="")

        in_ch = self._estimate_input_channels()
        win = max(4, self._safe_int_var(self.var_win, self.cfg.window_len))
        balance_on = bool(self.var_use_balance.get())
        gain = self._safe_float_var(self.var_balance_gain, self.cfg.balance_feature_gain)
        spike_on = bool(self.var_use_spike_feat.get()) if hasattr(self, "var_use_spike_feat") else False
        spike_gain = self._safe_float_var(self.var_spike_gain, self.cfg.spike_feature_gain) if hasattr(self, "var_spike_gain") else 1.0

        phase = "idle"
        live_txt = "phase=idle"
        if self.nn_live is not None:
            try:
                phase = str(self.nn_live.get("phase", "idle"))
                live_txt = (
                    f"phase={phase}"
                    f" fold={self.nn_live.get('fold', '-')}/{self.nn_live.get('fold_total', '-')}"
                    f" ep={self.nn_live.get('epoch', '-')}/{self.nn_live.get('epochs', '-')}"
                    f" step={self.nn_live.get('step', '-')}/{self.nn_live.get('total_steps', '-')}"
                    f" loss={float(self.nn_live.get('loss', 0.0)):.4f}"
                    f" lr={self._fmt_lr(float(self.nn_live.get('lr', 0.0)))}"
                )
            except Exception:
                pass

        if phase == "update":
            accent = "#4f7cff"
        elif phase == "epoch_end":
            accent = "#8b5cf6"
        elif phase == "fold_start":
            accent = "#1f9d7a"
        else:
            accent = "#6b7280"

        blocks = [
            (f"Input\n(B,{in_ch},{win})", "#e8f2ff"),
            ("Conv1D 64\nk=5", "#f3f4f6"),
            ("BatchNorm + ReLU", "#f3f4f6"),
            ("Conv1D 128\nk=5", "#f3f4f6"),
            ("BatchNorm + ReLU", "#f3f4f6"),
            ("MaxPool1D\nk=2", "#f3f4f6"),
            ("Conv1D 128\nk=3", "#f3f4f6"),
            ("BatchNorm + ReLU", "#f3f4f6"),
            ("AdaptiveAvgPool1D(1)", "#f3f4f6"),
            ("Linear(128 -> 2)\n[NORMAL, DEFECT]", "#eaf8ea"),
        ]
        n = len(blocks)
        margin = 20
        gap = 10
        bw = max(88, int((w - margin * 2 - gap * (n - 1)) / n))
        bh = min(88, h - 64)
        y = (h - bh) // 2 + 2

        x = margin
        centers = []
        for i, (txt, fill) in enumerate(blocks):
            c.create_rectangle(x + 2, y + 2, x + bw + 2, y + bh + 2, outline="", fill="#d9e1ea")
            edge = accent if phase != "idle" and i in (1, 3, 6, 9) else "#8f98a3"
            c.create_rectangle(x, y, x + bw, y + bh, outline=edge, width=2 if edge == accent else 1, fill=fill)
            c.create_text(x + bw / 2, y + bh / 2, text=txt, fill="#2f3945", justify="center")
            centers.append((x + bw / 2, y + bh / 2))
            x += bw + gap

        for i in range(len(centers) - 1):
            x1, y1 = centers[i]
            x2, y2 = centers[i + 1]
            col = accent if phase != "idle" and i % 2 == 0 else "#7d8794"
            c.create_line(x1 + bw / 2 - 8, y1, x2 - bw / 2 + 8, y2, arrow="last", fill=col, width=2)

        bal_txt = "ON" if balance_on else "OFF"
        c.create_text(10, 8, anchor="nw", text=live_txt, fill="#37485a")
        c.create_text(10, h - 8, anchor="sw", text=f"Balance: {bal_txt} (gain={gain:.2f}) | Spike: {'ON' if spike_on else 'OFF'} (gain={spike_gain:.2f})", fill="#44586f")
        c.create_text(w - 10, h - 8, anchor="se", text=f"TemporalCNN block diagram | phase={phase}", fill="#637487")
        self.nn_info_var.set(
            f"Input channels={in_ch} (base=5+hist_bins={self._safe_int_var(self.var_bins, self.cfg.hist_bins)}"
            f"{'+10 balance' if balance_on else ''}{'+4 spike' if spike_on else ''}"
            f"{'+13 tilt' if (hasattr(self, 'var_use_tilt_detail') and bool(self.var_use_tilt_detail.get())) else ''}), window_len={win}"
        )

    def log(self, s: str):
        self.txt.insert("end", s + "\n")
        self.txt.see("end")

    def _ts(self) -> str:
        return time.strftime("%H:%M:%S")

    def dlog(self, s: str):
        line = f"[DBG {self._ts()}] {s}"
        # Immediate visibility in UI thread + queued visibility for worker thread usage.
        try:
            self.log(line)
        except Exception:
            pass
        self.msg_q.put(("log", line))

    def _poll_queue(self):
        try:
            while True:
                kind, payload = self.msg_q.get_nowait()
                try:
                    if kind == "log":
                        self.log(str(payload))
                    elif kind == "progress":
                        p = float(payload)
                        self.progress["value"] = max(0.0, min(100.0, p * 100.0))
                    elif kind == "lr_reset":
                        self.lr_history = []
                        if isinstance(payload, dict):
                            self.lr_epochs_total = max(1, int(payload.get("epochs", int(self.var_epochs.get()))))
                            self.lr_plot_planned_max = max(1e-12, float(payload.get("lr", self._safe_float_var(self.var_lr, 1e-3))))
                        else:
                            self.lr_epochs_total = max(1, int(payload) if payload is not None else int(self.var_epochs.get()))
                            self.lr_plot_planned_max = max(1e-12, self._safe_float_var(self.var_lr, 1e-3))
                        self.lr_y_linear_bounds = None
                        self.lr_fold_label = "-"
                        self.nn_live = None
                        self._draw_lr_plot()
                        self._reset_metrics()
                        self._draw_nn_graph()
                    elif kind == "lr_event":
                        ev = payload if isinstance(payload, dict) else {}
                        et = str(ev.get("event", ""))
                        if et == "fold_start":
                            f = int(ev.get("fold", 0))
                            ft = int(ev.get("fold_total", 0))
                            self.lr_fold_label = f"{f}/{ft}" if f > 0 and ft > 0 else "-"
                            self.lr_history = []
                        elif et == "epoch":
                            ep = int(ev.get("epoch", 0))
                            ept = int(ev.get("epochs", self.lr_epochs_total))
                            lr = float(ev.get("lr", 0.0))
                            self.lr_epochs_total = max(1, ept)
                            if ep > 0:
                                self.lr_history = [x for x in self.lr_history if int(x[0]) != ep]
                                self.lr_history.append((ep, lr))
                                self.lr_history.sort(key=lambda t: t[0])
                        self._draw_lr_plot()
                    elif kind == "metrics":
                        self._update_metrics(payload if isinstance(payload, dict) else {})
                    elif kind == "nn_snapshot":
                        self.nn_live = payload if isinstance(payload, dict) else None
                        self._draw_nn_graph()
                    elif kind == "metrics_threshold":
                        self.metric_vars["threshold"].set(f"{float(payload):.2f}")
                    elif kind == "train_target":
                        self.metric_vars["train_target"].set(self._train_mode_text(str(payload)))
                    elif kind == "train_done":
                        model, best_acc = payload
                        self.model = model
                        self.metric_vars["best_val_acc"].set(f"{float(best_acc):.4f}")
                        self._draw_nn_graph()
                        self.msg_q.put(("log", f"[DONE] Training finished. best_val_acc={best_acc:.3f}  device={self.cfg.device}"))
                    elif kind == "error":
                        messagebox.showerror("Error", str(payload))
                        self.msg_q.put(("log", f"[ERROR] {payload}"))
                except Exception as inner_e:
                    self.log(f"[POLL-ERROR] kind={kind} payload={payload} err={inner_e}")
                    self.log(traceback.format_exc().strip())
        except queue.Empty:
            pass
        except Exception as outer_e:
            self.log(f"[POLL-FATAL] {outer_e}")
            self.log(traceback.format_exc().strip())
        self.after(100, self._poll_queue)

    def _reset_metrics(self):
        self.metric_vars["fold"].set("-")
        self.metric_vars["epoch"].set("-")
        self.metric_vars["train_loss"].set("-")
        self.metric_vars["train_acc"].set("-")
        self.metric_vars["val_loss"].set("-")
        self.metric_vars["val_acc"].set("-")
        self.metric_vars["best_val_acc"].set("-")
        self.metric_vars["lr"].set("-")
        self.metric_vars["tilt_mean"].set("-")
        self.metric_vars["threshold"].set(f"{self.cfg.infer_threshold:.2f}")
        self.metric_vars["train_target"].set(self._train_mode_text(self.cfg.train_class_mode))

    def _update_metrics(self, m: Dict[str, object]):
        f = int(m.get("fold", 0))
        ft = int(m.get("fold_total", 0))
        ep = int(m.get("epoch", 0))
        ept = int(m.get("epochs", 0))
        self.metric_vars["fold"].set(f"{f}/{ft}" if f > 0 and ft > 0 else "-")
        self.metric_vars["epoch"].set(f"{ep}/{ept}" if ep > 0 and ept > 0 else "-")
        if "train_loss" in m:
            self.metric_vars["train_loss"].set(f"{float(m['train_loss']):.4f}")
        if "train_acc" in m:
            self.metric_vars["train_acc"].set(f"{float(m['train_acc']):.4f}")
        if "val_loss" in m:
            self.metric_vars["val_loss"].set(f"{float(m['val_loss']):.4f}")
        if "val_acc" in m:
            self.metric_vars["val_acc"].set(f"{float(m['val_acc']):.4f}")
        if "best_val_acc" in m:
            self.metric_vars["best_val_acc"].set(f"{float(m['best_val_acc']):.4f}")
        if "lr" in m:
            self.metric_vars["lr"].set(self._fmt_lr(float(m["lr"])))
        if "tilt_mean" in m:
            self.metric_vars["tilt_mean"].set(f"{float(m['tilt_mean']):.4f}")

    def _fmt_lr(self, v: float) -> str:
        v = float(max(0.0, v))
        if v >= 0.1:
            return f"{v:.4f}"
        if v >= 0.01:
            return f"{v:.5f}"
        if v >= 0.001:
            return f"{v:.6f}"
        if v >= 0.0001:
            return f"{v:.7f}"
        return f"{v:.8f}"

    def _draw_lr_plot(self):
        if not hasattr(self, "lr_canvas"):
            return
        sched_name = self.cfg.scheduler_name
        if hasattr(self, "var_sched"):
            try:
                sched_name = str(self.var_sched.get())
            except Exception:
                sched_name = self.cfg.scheduler_name
        c = self.lr_canvas
        c.delete("all")
        w = max(50, int(c.winfo_width()) if c.winfo_width() > 1 else 520)
        h = max(50, int(c.winfo_height()) if c.winfo_height() > 1 else 300)
        # Reserve bigger top/bottom header areas so labels never overlap plot.
        # Intentionally shrink plot area left/right to avoid any overlap with labels.
        left, top, right, bottom = 86, 44, w - 86, h - 58
        c.create_rectangle(left, top, right, bottom, outline="#cccccc")
        c.create_text(12, 10, text="LR vs Epoch", anchor="nw", fill="#333333")
        c.create_text(w - 12, h - 10, text=f"Fold {self.lr_fold_label} | {('log' if bool(self.var_lr_log_scale.get()) else 'linear')}",
                      anchor="se", fill="#555555")

        if len(self.lr_history) == 0:
            c.create_text((left + right) // 2, (top + bottom) // 2, text="No data", fill="#888888")
            self.lr_info_var.set(f"LR info: no data | scheduler={sched_name} | x=epoch, y=learning rate")
            return

        data = sorted(self.lr_history, key=lambda t: int(t[0]))
        xs = np.array([float(p[0]) for p in data], dtype=np.float32)
        ys = np.array([float(p[1]) for p in data], dtype=np.float32)
        x0, x1 = 1.0, float(max(1, self.lr_epochs_total))
        y_data_min, y_data_max = float(ys.min()), float(ys.max())
        if self.lr_y_linear_bounds is None:
            y_hi_init = max(float(self.lr_plot_planned_max), y_data_max, 1e-12)
            self.lr_y_linear_bounds = (0.0, y_hi_init * 1.05)
        y0_lin, y1_lin = self.lr_y_linear_bounds
        if y_data_max > y1_lin:
            y1_lin = y_data_max * 1.05
            self.lr_y_linear_bounds = (y0_lin, y1_lin)

        log_scale = bool(self.var_lr_log_scale.get())
        if log_scale:
            # Keep log range fixed per run and avoid log(0).
            y_floor = max(1e-12, y1_lin * 1e-6)
            ys_plot = np.log10(np.maximum(ys, y_floor))
            py0 = float(np.log10(y_floor))
            py1 = float(np.log10(max(y1_lin, y_floor * 10.0)))
        else:
            ys_plot = ys
            py0, py1 = float(y0_lin), float(y1_lin)

        # Grid + ticks
        for i in range(5):
            yy = top + i * (bottom - top) / 4.0
            c.create_line(left, yy, right, yy, fill="#efefef")
            yv = py1 - i * (py1 - py0) / 4.0
            if log_scale:
                lv = 10.0 ** float(yv)
                c.create_text(left - 6, yy, text=self._fmt_lr(float(lv)), anchor="e", fill="#666666")
            else:
                c.create_text(left - 6, yy, text=self._fmt_lr(float(yv)), anchor="e", fill="#666666")
        x_ticks = [1, max(1, int(round(x1 * 0.33))), max(1, int(round(x1 * 0.66))), int(x1)]
        x_ticks = sorted(list(dict.fromkeys(x_ticks)))
        for xv in x_ticks:
            xx = left + (float(xv) - x0) / max(1e-9, (x1 - x0)) * (right - left)
            c.create_line(xx, top, xx, bottom, fill="#f4f4f4")
            c.create_text(xx, bottom + 14, text=f"{int(xv)}", anchor="n", fill="#666666")
        c.create_text((left + right) / 2.0, h - 24, text="epoch", anchor="s", fill="#555555")
        c.create_text(8, (top + bottom) / 2.0, text="LR", anchor="w", fill="#555555")

        pts = []
        for xv, yv in zip(xs, ys_plot):
            px = left + (xv - x0) / max(1e-9, (x1 - x0)) * (right - left)
            py = bottom - (yv - py0) / (py1 - py0) * (bottom - top)
            pts.extend([float(px), float(py)])
        if len(pts) >= 4:
            c.create_line(*pts, fill="#1f77b4", width=2, smooth=True)
        for xv, yv in zip(xs, ys_plot):
            px = left + (xv - x0) / max(1e-9, (x1 - x0)) * (right - left)
            py = bottom - (yv - py0) / (py1 - py0) * (bottom - top)
            c.create_oval(px - 2.5, py - 2.5, px + 2.5, py + 2.5, fill="#1f77b4", outline="#1f77b4")
        # Mark current point
        last_x, last_y = float(xs[-1]), float(ys[-1])
        last_y_plot = float(ys_plot[-1])
        lx = left + (last_x - x0) / max(1e-9, (x1 - x0)) * (right - left)
        ly = bottom - (last_y_plot - py0) / (py1 - py0) * (bottom - top)
        c.create_oval(lx - 3, ly - 3, lx + 3, ly + 3, fill="#d62728", outline="#d62728")
        c.create_text(lx + 6, ly - 6, text=self._fmt_lr(last_y), anchor="sw", fill="#d62728")

        start_lr = float(ys[0])
        self.lr_info_var.set(
            "LR info: "
            f"scheduler={sched_name} | start={self._fmt_lr(start_lr)} | current={self._fmt_lr(last_y)} | "
            f"data_min={self._fmt_lr(y_data_min)} | data_max={self._fmt_lr(y_data_max)} | "
            f"plot_max={self._fmt_lr(y1_lin)} | points={len(self.lr_history)} | "
            f"scale={'log' if log_scale else 'linear'} | x=epoch, y=learning rate"
        )

    def on_reset_train_options(self):
        dcfg = AppConfig()
        self.var_fps.set(dcfg.fps_target)
        self.var_bins.set(dcfg.hist_bins)
        self.var_win.set(dcfg.window_len)
        self.var_stride.set(dcfg.stride)
        self.var_epochs.set(12)
        self.var_bs.set(64)
        self.var_lr.set(1e-3)
        self.var_use_aug.set(bool(dcfg.use_augmentation))
        self.var_loss.set(dcfg.loss_name)
        self.var_sched.set(dcfg.scheduler_name)
        self.var_kfold.set(int(dcfg.k_folds))
        self.var_use_hard_mining.set(bool(dcfg.use_hard_mining))
        self.var_hard_ratio.set(float(dcfg.hard_mining_ratio))
        self.var_use_type_oversampling.set(bool(dcfg.use_type_oversampling))
        self.var_hard_keywords.set(str(dcfg.hard_type_keywords))
        self.var_type_os_factor.set(int(dcfg.type_oversample_factor))
        self.var_auto_threshold.set(bool(dcfg.auto_threshold))
        self.var_use_balance.set(bool(dcfg.use_balance_features))
        self.var_balance_gain.set(float(dcfg.balance_feature_gain))
        self.var_use_spike_feat.set(bool(dcfg.use_spike_features))
        self.var_spike_gain.set(float(dcfg.spike_feature_gain))
        self.var_use_adapt_smooth.set(bool(dcfg.use_adaptive_pixel_smoothing))
        self.var_jitter_th.set(float(dcfg.pixel_jitter_threshold))
        self.var_smooth_k.set(int(dcfg.pixel_smoothing_kernel))
        self.var_use_tilt_detail.set(bool(dcfg.use_tilt_detail_features))
        self.var_tilt_gain.set(float(dcfg.tilt_gain))
        self.var_tilt_loss_alpha.set(float(dcfg.tilt_loss_alpha))
        self.var_auto_hpo.set(bool(dcfg.auto_hparam_search))
        self.var_hpo_trials.set(int(dcfg.hparam_trials))
        self.var_train_class_mode.set(str(dcfg.train_class_mode))
        self.metric_vars["train_target"].set(self._train_mode_text(dcfg.train_class_mode))
        self.msg_q.put(("log", "Training options reset to defaults."))

    def _apply_cfg_to_ui(self):
        self.var_fps.set(self.cfg.fps_target)
        self.var_bins.set(self.cfg.hist_bins)
        self.var_win.set(self.cfg.window_len)
        self.var_stride.set(self.cfg.stride)
        self.var_use_aug.set(bool(self.cfg.use_augmentation))
        self.var_loss.set(self.cfg.loss_name)
        self.var_sched.set(self.cfg.scheduler_name)
        self.var_train_class_mode.set(str(self.cfg.train_class_mode))
        self.var_kfold.set(int(self.cfg.k_folds))
        self.var_use_hard_mining.set(bool(self.cfg.use_hard_mining))
        self.var_hard_ratio.set(float(self.cfg.hard_mining_ratio))
        self.var_use_type_oversampling.set(bool(self.cfg.use_type_oversampling))
        self.var_hard_keywords.set(str(self.cfg.hard_type_keywords))
        self.var_type_os_factor.set(int(self.cfg.type_oversample_factor))
        self.var_auto_threshold.set(bool(self.cfg.auto_threshold))
        self.var_use_balance.set(bool(self.cfg.use_balance_features))
        self.var_balance_gain.set(float(self.cfg.balance_feature_gain))
        self.var_use_spike_feat.set(bool(self.cfg.use_spike_features))
        self.var_spike_gain.set(float(self.cfg.spike_feature_gain))
        self.var_use_adapt_smooth.set(bool(self.cfg.use_adaptive_pixel_smoothing))
        self.var_jitter_th.set(float(self.cfg.pixel_jitter_threshold))
        self.var_smooth_k.set(int(self.cfg.pixel_smoothing_kernel))
        self.var_use_tilt_detail.set(bool(self.cfg.use_tilt_detail_features))
        self.var_tilt_gain.set(float(self.cfg.tilt_gain))
        self.var_tilt_loss_alpha.set(float(self.cfg.tilt_loss_alpha))
        self.var_auto_hpo.set(bool(self.cfg.auto_hparam_search))
        self.var_hpo_trials.set(int(self.cfg.hparam_trials))
        self.metric_vars["train_target"].set(self._train_mode_text(self.cfg.train_class_mode))
        if self.cfg.roi is not None:
            x, y, w, h = self.cfg.roi
            self.roi_label.config(text=f"ROI: x={x}, y={y}, w={w}, h={h}")
        else:
            self.roi_label.config(text="ROI: (not set)")

    def _safe_profile_name(self, name: str) -> str:
        # Keep Korean/Unicode letters and digits too.
        cleaned = "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in name.strip())
        while "__" in cleaned:
            cleaned = cleaned.replace("__", "_")
        return cleaned.strip("_")

    def _extract_tail_token(self, stem: str) -> str:
        # Example: "SKR0fk430gfk304g.吏꾨룞" -> "吏꾨룞"
        parts = [p.strip() for p in stem.split(".") if p.strip()]
        if len(parts) >= 2:
            return parts[-1]
        return stem.strip()

    def _derive_common_train_name(self) -> str:
        # Build a default name from common suffix of training file names (without extension).
        stems: List[str] = []
        mode = str(self.var_train_class_mode.get()).strip().lower() if hasattr(self, "var_train_class_mode") else str(self.cfg.train_class_mode).strip().lower()
        cls_targets = ("normal", "defect") if mode == "both" else (mode,)
        if self.dataset_root:
            for cls in cls_targets:
                cls_dir = os.path.join(self.dataset_root, cls)
                if os.path.isdir(cls_dir):
                    vids = list_videos(cls_dir)
                    stems.extend(os.path.splitext(os.path.basename(v))[0] for v in vids)
        stems = [s.strip() for s in stems if s.strip()]
        if stems:
            # 1) Prefer suffix token after last dot in file stem
            tail_tokens = [self._extract_tail_token(s) for s in stems]
            tail_tokens = [t for t in tail_tokens if t]
            if tail_tokens:
                # If one token dominates, use it (e.g., 吏꾨룞/怨좎삩)
                tok, cnt = Counter(tail_tokens).most_common(1)[0]
                if cnt >= 2:
                    safe_tok = self._safe_profile_name(tok)
                    if len(safe_tok) >= 1:
                        return safe_tok
                # Else fallback to common suffix among tokens
                rev_tok = [t[::-1] for t in tail_tokens]
                common_tok_suffix = os.path.commonprefix(rev_tok)[::-1].strip(" _-.")
                if len(common_tok_suffix) >= 1:
                    safe_tok_suffix = self._safe_profile_name(common_tok_suffix)
                    if len(safe_tok_suffix) >= 1:
                        return safe_tok_suffix

            # 2) Fallback: common suffix among full stems
            rev = [s[::-1] for s in stems]
            common_suffix = os.path.commonprefix(rev)[::-1].strip(" _-.")
            safe_common = self._safe_profile_name(common_suffix)
            if len(safe_common) >= 1:
                return safe_common
        if self.dataset_root:
            ds_name = os.path.basename(os.path.normpath(self.dataset_root))
            safe_ds = self._safe_profile_name(ds_name)
            if safe_ds:
                return safe_ds
        return "train"

    def _build_option_compact_suffix(self) -> str:
        # Compact but exhaustive option signature for profile names.
        ui_epochs = self._safe_int_var(self.var_epochs, 12) if hasattr(self, "var_epochs") else 12
        ui_bs = self._safe_int_var(self.var_bs, 64) if hasattr(self, "var_bs") else 64
        ui_lr = self._safe_float_var(self.var_lr, 1e-3) if hasattr(self, "var_lr") else 1e-3
        c = self.cfg
        parts = [
            f"fps{int(c.fps_target)}",
            f"hb{int(c.hist_bins)}",
            f"wl{int(c.window_len)}",
            f"st{int(c.stride)}",
            f"ep{int(ui_epochs)}",
            f"bs{int(ui_bs)}",
            f"lr{ui_lr:.6f}",
            f"ls{c.loss_name}",
            f"sch{c.scheduler_name}",
            f"tcm{self._safe_profile_name(str(c.train_class_mode))}",
            f"kf{int(c.k_folds)}",
            f"aug{int(bool(c.use_augmentation))}",
            f"hm{int(bool(c.use_hard_mining))}",
            f"hr{float(c.hard_mining_ratio):.2f}",
            f"tos{int(bool(c.use_type_oversampling))}",
            f"toF{int(c.type_oversample_factor)}",
            f"toK{self._safe_profile_name(str(c.hard_type_keywords)) or 'none'}",
            f"ath{int(bool(c.auto_threshold))}",
            f"thr{float(c.infer_threshold):.2f}",
            f"bal{int(bool(c.use_balance_features))}",
            f"bg{float(c.balance_feature_gain):.2f}",
            f"spk{int(bool(c.use_spike_features))}",
            f"sg{float(c.spike_feature_gain):.2f}",
            f"aps{int(bool(c.use_adaptive_pixel_smoothing))}",
            f"jt{float(c.pixel_jitter_threshold):.3f}",
            f"sk{int(c.pixel_smoothing_kernel)}",
            f"til{int(bool(c.use_tilt_detail_features))}",
            f"tg{float(c.tilt_gain):.2f}",
            f"ta{float(c.tilt_loss_alpha):.2f}",
            f"hpo{int(bool(c.auto_hparam_search))}",
            f"ht{int(c.hparam_trials)}",
        ]
        out = "_".join(parts).replace(".", "p")
        out = self._safe_profile_name(out)
        # Keep filename length practical on Windows.
        return out[:180] if len(out) > 180 else out

    def _profile_display_text(self, name: str, payload: Optional[Dict[str, object]] = None) -> str:
        # List view text: "name | key options..."
        if payload is None:
            return name
        cfg = payload.get("cfg", {}) if isinstance(payload, dict) else {}
        if not isinstance(cfg, dict):
            return name
        ui = payload.get("ui", {}) if isinstance(payload, dict) else {}
        ep = ui.get("epochs", "-") if isinstance(ui, dict) else "-"
        bs = ui.get("batch_size", "-") if isinstance(ui, dict) else "-"
        lr = ui.get("lr", "-") if isinstance(ui, dict) else "-"
        opts = (
            f"ep={ep},bs={bs},lr={lr},"
            f"loss={cfg.get('loss_name','-')},sch={cfg.get('scheduler_name','-')},"
            f"target={cfg.get('train_class_mode','both')},"
            f"kf={cfg.get('k_folds','-')},bal={cfg.get('use_balance_features','-')},"
            f"spk={cfg.get('use_spike_features','-')},tilt={cfg.get('use_tilt_detail_features','-')}"
        )
        return f"{name} | {opts}"

    def _profile_path(self, name: str) -> str:
        return os.path.join(self.profiles_dir, f"{name}.json")

    def _selected_profile_name(self) -> Optional[str]:
        sel = self.profile_list.curselection()
        if not sel:
            return None
        v = self.profile_list.get(sel[0]).strip()
        if not v:
            return None
        # list item format: "<name> | ..."
        if " | " in v:
            v = v.split(" | ", 1)[0].strip()
        return v if v else None

    def _refresh_profile_list(self):
        safe_mkdir(self.profiles_dir)
        files = [f for f in os.listdir(self.profiles_dir) if f.lower().endswith(".json")]
        names = sorted(os.path.splitext(f)[0] for f in files)
        self.profile_list.delete(0, "end")
        for n in names:
            p = self._profile_path(n)
            payload = None
            try:
                with open(p, "r", encoding="utf-8") as f:
                    payload = json.load(f)
            except Exception:
                payload = None
            self.profile_list.insert("end", self._profile_display_text(n, payload))
        self.msg_q.put(("log", f"Profiles loaded: {len(names)}"))

    def on_profile_save(self):
        self._sync_cfg_from_ui()
        base = self._derive_common_train_name()
        opt = self._build_option_compact_suffix()
        default_name = f"{base}_{opt}_{time.strftime('%Y%m%d_%H%M%S')}"
        name = simpledialog.askstring("Save Profile", "Profile name:", initialvalue=default_name, parent=self)
        if not name:
            return
        safe_name = self._safe_profile_name(name)
        if not safe_name:
            messagebox.showerror("Error", "Invalid profile name.")
            return
        safe_mkdir(self.profiles_dir)
        path = self._profile_path(safe_name)
        payload = {
            "profile_name": safe_name,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_root": self.dataset_root,
            "model_path": self.model_path,
            "ui": {
                "epochs": int(self.var_epochs.get()) if hasattr(self, "var_epochs") else 12,
                "batch_size": int(self.var_bs.get()) if hasattr(self, "var_bs") else 64,
                "lr": float(self.var_lr.get()) if hasattr(self, "var_lr") else 1e-3,
            },
            "cfg": asdict(self.cfg),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        self._refresh_profile_list()
        self.msg_q.put(("log", f"Profile saved: {path}"))

    def on_profile_load(self):
        name = self._selected_profile_name()
        if not name:
            messagebox.showerror("Error", "Select a profile first.")
            return
        path = self._profile_path(name)
        if not os.path.isfile(path):
            messagebox.showerror("Error", f"Profile file not found: {path}")
            self._refresh_profile_list()
            return
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        cfg_dict = payload.get("cfg", {})
        valid = {f.name for f in fields(AppConfig)}
        cfg_dict = {k: v for k, v in cfg_dict.items() if k in valid}
        if "roi" in cfg_dict and isinstance(cfg_dict["roi"], list):
            cfg_dict["roi"] = tuple(cfg_dict["roi"])
        self.cfg = AppConfig(**cfg_dict)
        self._apply_cfg_to_ui()

        self.dataset_root = payload.get("dataset_root", None)
        if self.dataset_root:
            self.ds_label.config(text=f"Dataset: {self.dataset_root}")
        else:
            self.ds_label.config(text="Dataset: (not set)")
        self.model_path = payload.get("model_path", None)
        self.msg_q.put(("log", f"Profile loaded: {path}"))

    def on_profile_delete(self):
        name = self._selected_profile_name()
        if not name:
            messagebox.showerror("Error", "Select a profile first.")
            return
        path = self._profile_path(name)
        if not os.path.isfile(path):
            self._refresh_profile_list()
            return
        ok = messagebox.askyesno("Delete Profile", f"Delete profile '{name}'?")
        if not ok:
            return
        os.remove(path)
        self._refresh_profile_list()
        self.msg_q.put(("log", f"Profile deleted: {path}"))

    def on_set_roi(self):
        path = filedialog.askopenfilename(
            title="Pick a sample video for ROI selection",
            filetypes=[("Video", "*.mp4 *.avi *.mkv *.mov"), ("All", "*.*")]
        )
        if not path:
            return

        try:
            r = select_roi_with_seek(path, init_roi=self.cfg.roi)
            if r is None:
                self.msg_q.put(("log", "ROI selection cancelled."))
                return

            x, y, w, h = map(int, r)
            self.cfg.roi = (x, y, w, h)
            self.roi_label.config(text=f"ROI: x={x}, y={y}, w={w}, h={h}")
            self.msg_q.put(("log", f"ROI set: {self.cfg.roi}"))

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def on_select_dataset(self):
        root = filedialog.askdirectory(title="Select dataset root folder (normal/defect subfolders)")
        if not root:
            return
        n_dir = os.path.join(root, "normal")
        d_dir = os.path.join(root, "defect")
        mode = str(self.var_train_class_mode.get()).strip().lower() if hasattr(self, "var_train_class_mode") else "both"
        required = []
        if mode in ("both", "normal"):
            required.append(("normal", n_dir))
        if mode in ("both", "defect"):
            required.append(("defect", d_dir))
        missing = [name for name, p in required if not os.path.isdir(p)]
        if missing:
            messagebox.showerror("Error", f"Dataset root is missing required subfolder(s) for mode '{mode}': {', '.join(missing)}")
            return
        self.dataset_root = root
        self.ds_label.config(text=f"Dataset: {root}")
        self.msg_q.put(("log", f"Dataset root set: {root} (train_class={mode})"))

    def _sync_cfg_from_ui(self):
        self.cfg.fps_target = int(self.var_fps.get())
        self.cfg.hist_bins = int(self.var_bins.get())
        self.cfg.window_len = int(self.var_win.get())
        self.cfg.stride = int(self.var_stride.get())
        self.cfg.use_augmentation = bool(self.var_use_aug.get())
        self.cfg.loss_name = str(self.var_loss.get()).strip() or "cross_entropy_ls"
        self.cfg.scheduler_name = str(self.var_sched.get()).strip() or "cosine"
        m = str(self.var_train_class_mode.get()).strip().lower()
        self.cfg.train_class_mode = m if m in ("both", "normal", "defect") else "both"
        self.cfg.k_folds = max(1, int(self.var_kfold.get()))
        self.cfg.use_hard_mining = bool(self.var_use_hard_mining.get())
        self.cfg.hard_mining_ratio = min(max(float(self.var_hard_ratio.get()), 0.05), 1.0)
        self.cfg.use_type_oversampling = bool(self.var_use_type_oversampling.get())
        self.cfg.hard_type_keywords = str(self.var_hard_keywords.get()).strip()
        self.cfg.type_oversample_factor = max(2, int(self.var_type_os_factor.get()))
        self.cfg.auto_threshold = bool(self.var_auto_threshold.get())
        self.cfg.use_balance_features = bool(self.var_use_balance.get())
        self.cfg.balance_feature_gain = min(max(self._safe_float_var(self.var_balance_gain, 1.0), 0.1), 10.0)
        self.cfg.use_spike_features = bool(self.var_use_spike_feat.get())
        self.cfg.spike_feature_gain = min(max(self._safe_float_var(self.var_spike_gain, 1.0), 0.1), 10.0)
        self.cfg.use_adaptive_pixel_smoothing = bool(self.var_use_adapt_smooth.get())
        self.cfg.pixel_jitter_threshold = min(max(self._safe_float_var(self.var_jitter_th, 0.08), 0.0), 1.0)
        k = max(1, int(self._safe_int_var(self.var_smooth_k, 3)))
        if k % 2 == 0:
            k += 1
        self.cfg.pixel_smoothing_kernel = min(k, 11)
        self.cfg.use_tilt_detail_features = bool(self.var_use_tilt_detail.get())
        self.cfg.tilt_gain = min(max(self._safe_float_var(self.var_tilt_gain, 1.0), 0.1), 10.0)
        self.cfg.tilt_loss_alpha = min(max(self._safe_float_var(self.var_tilt_loss_alpha, 0.0), 0.0), 5.0)
        self.cfg.auto_hparam_search = bool(self.var_auto_hpo.get())
        self.cfg.hparam_trials = max(1, int(self._safe_int_var(self.var_hpo_trials, 8)))

    def on_stop(self):
        self.stop_event.set()
        self.msg_q.put(("log", "Stop requested..."))

    def on_show_help(self):
        import tkinter.font as tkfont

        win = tk.Toplevel(self)
        win.title("옵션 상세 도움말")
        win.geometry("980x760")
        try:
            win.transient(self)
        except Exception:
            pass

        outer = ttk.Frame(win)
        outer.pack(fill="both", expand=True, padx=10, pady=10)

        ctrl = ttk.Frame(outer)
        ctrl.pack(fill="x", pady=(0, 8))

        ttk.Label(ctrl, text="검색").pack(side="left")
        var_q = tk.StringVar()
        ent_q = ttk.Entry(ctrl, textvariable=var_q, width=22)
        ent_q.pack(side="left", padx=(4, 4))

        btn_prev = ttk.Button(ctrl, text="이전")
        btn_prev.pack(side="left")
        btn_next = ttk.Button(ctrl, text="다음")
        btn_next.pack(side="left", padx=(4, 0))

        ttk.Separator(ctrl, orient="vertical").pack(side="left", fill="y", padx=8)

        ttk.Label(ctrl, text="섹션").pack(side="left")
        var_section = tk.StringVar(value="전체")
        cmb_section = ttk.Combobox(ctrl, textvariable=var_section, state="readonly", width=24)
        cmb_section.pack(side="left", padx=(4, 4))
        btn_jump = ttk.Button(ctrl, text="이동")
        btn_jump.pack(side="left")

        var_section_only = tk.BooleanVar(value=False)
        chk_section_only = ttk.Checkbutton(ctrl, text="선택 섹션만 보기", variable=var_section_only)
        chk_section_only.pack(side="left", padx=(8, 0))

        ttk.Separator(ctrl, orient="vertical").pack(side="left", fill="y", padx=8)

        ttk.Label(ctrl, text="글자 크기").pack(side="left")
        var_font_size = tk.IntVar(value=11)
        spn_font = ttk.Spinbox(ctrl, from_=9, to=18, width=4, textvariable=var_font_size)
        spn_font.pack(side="left", padx=(4, 6))

        var_info = tk.StringVar(value="검색어를 입력하면 강조됩니다.")
        ttk.Label(ctrl, textvariable=var_info).pack(side="right")

        body = ttk.Frame(outer)
        body.pack(fill="both", expand=True)
        txt = tk.Text(body, wrap="word")
        yscroll = ttk.Scrollbar(body, orient="vertical", command=txt.yview)
        txt.configure(yscrollcommand=yscroll.set)
        txt.pack(side="left", fill="both", expand=True)
        yscroll.pack(side="right", fill="y")
        help_font = tkfont.Font(family="Malgun Gothic", size=var_font_size.get())
        txt.configure(font=help_font, spacing1=2, spacing2=2, spacing3=5)
        txt.tag_configure("find_hit", background="#ffe58a")
        txt.tag_configure("find_current", background="#ffcc33")

        help_text = """
[기본 사용 순서]
1) Set ROI 버튼으로 검사 영역을 먼저 지정합니다.

2) Select Dataset Root Folder 버튼으로 데이터셋 루트를 선택합니다.
   - 루트 아래에 normal, defect 폴더가 있어야 합니다.

3) Parameter 영역에서 학습 옵션을 설정합니다.

4) Train 버튼으로 학습을 시작합니다.

5) 결과 확인 후 Save Model / Save Current(옵션 저장)으로 재사용합니다.


[판단 로직(정상/불량)]
- 모델은 각 샘플의 defect 확률(prob_defect)을 계산합니다.
- 최종 판정은 prob_defect >= threshold 조건으로 결정됩니다.
- auto_threshold ON이면 검증 세트 기준으로 임계값(threshold)을 자동 탐색합니다.


[데이터/특징 옵션]
fps_target:
  영상에서 초당 몇 프레임을 사용할지 결정합니다.
  값이 높으면 시간 정보가 풍부해지지만 연산량이 증가합니다.

hist_bins:
  밝기 히스토그램 구간 수입니다.
  값이 높으면 세밀한 특징을 보지만 입력 차원이 커집니다.

window_len:
  시계열 윈도우 길이(프레임 수)입니다.
  길면 장기 패턴을 반영하고, 짧으면 빠른 변화에 민감합니다.

stride:
  윈도우 이동 간격입니다.
  작을수록 샘플 수가 증가합니다.

smooth_sec:
  시계열 이동평균 시간(초)입니다.
  노이즈를 줄이는 대신 급격한 변화를 완화할 수 있습니다.

use_balance_features:
  ROI의 상하/좌우 균형, 무게중심, 분면 에너지 특징을 추가합니다.

balance_gain:
  위 균형 특징의 반영 강도(가중)를 조정합니다.

use_spike_features:
  프레임 간 급변(spike) 관련 특징을 추가합니다.

spike_gain:
  spike 특징 반영 강도입니다.

use_tilt_detail_features:
  기울기 상세 특징(전역/분면/Sobel/tilt_score)을 추가합니다.

tilt_gain:
  기울기 상세 특징 반영 강도입니다.

tilt_loss_alpha:
  tilt_score가 큰 샘플의 학습 손실 가중치를 추가합니다.
  0이면 비활성, 값이 클수록 기울기 이상 샘플을 더 강하게 학습합니다.

adaptive_pixel_smoothing:
  픽셀 변동(jitter)이 큰 프레임에 조건부 평활화를 적용합니다.

jitter_threshold:
  jitter 감지 임계값(0~1)입니다.
  낮추면 더 자주 평활화가 걸립니다.

smooth_kernel:
  평활화 커널 크기(홀수 권장)입니다.


[학습 옵션]
epochs:
  전체 학습 반복 횟수입니다.

batch_size:
  배치 크기입니다. GPU 메모리 사용량에 직접 영향을 줍니다.

lr:
  기본 학습률입니다.

loss:
  cross_entropy_ls: 일반적으로 안정적, 라벨 스무딩 적용
  focal: 어려운 샘플에 집중

scheduler:
  none / cosine / onecycle

k_folds:
  1이면 단일 분할, 2 이상이면 Stratified K-Fold 검증입니다.

use_augmentation:
  학습 시 노이즈/시간 시프트/feature dropout 증강을 적용합니다.

use_hard_mining:
  손실이 큰 샘플(top loss)을 우선 학습합니다.

hard_ratio:
  hard mining에서 상위 손실 샘플 비율입니다.

use_type_oversampling:
  지정 타입 샘플을 복제해 학습 비중을 높입니다.

hard_types(csv):
  오버샘플링 대상 타입 키워드 목록(쉼표 구분)입니다.

type_os_factor:
  오버샘플링 배수입니다.

auto_hparam_search:
  학습 전 자동 하이퍼파라미터 탐색을 실행합니다.

hpo_trials:
  탐색 시도 횟수입니다.


[추론/임계값 옵션]
auto_threshold:
  검증 기반 임계값 자동 탐색 사용 여부.

infer_threshold:
  수동 임계값(자동 탐색 OFF일 때 주로 사용).


[시각화 영역]
Neural Network Structure:
  현재 입력 차원, 네트워크 블록 구성, 학습 상태(phase/epoch/step/loss/lr)를 표시합니다.

Learning Rate Curve:
  epoch 대비 learning rate 변화를 보여줍니다.
  장기 학습에서 작은 LR 변화를 보기 위해 로그 스케일 옵션을 사용할 수 있습니다.


[프로파일/모델 저장]
Save Current:
  현재 ROI, 경로, 파라미터 전체를 JSON으로 저장합니다.

Load Selected:
  저장된 프로파일을 불러와 UI 설정을 복원합니다.

Save Model / Load Model:
  모델 파라미터와 학습 설정(cfg/env)을 함께 저장/복원합니다.


[불량 miss가 많을 때 권장 튜닝]
1) use_tilt_detail_features ON, tilt_gain 1.5~3.0
2) tilt_loss_alpha 0.5~1.2
3) loss=focal, scheduler=cosine
4) auto_threshold ON
5) k_folds를 3~5로 증가
6) hard mining + type oversampling 병행


[빠른 시작 추천값(예시)]
- epochs: 80~150
- batch_size: 16~64 (GPU 메모리에 맞춤)
- lr: 1e-3 또는 3e-4
- loss: cross_entropy_ls부터 시작, 어려우면 focal 전환
- scheduler: cosine
"""
        lines = help_text.strip().splitlines()
        sections = {}
        cur = "전체"
        buf = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                if cur != "전체":
                    sections[cur] = "\n".join(buf).strip()
                cur = stripped
                buf = [line]
            else:
                buf.append(line)
        if cur != "전체":
            sections[cur] = "\n".join(buf).strip()

        cmb_section["values"] = ["전체"] + list(sections.keys())
        search_state = {"hits": [], "pos": -1}

        def _render_help_text():
            txt.configure(state="normal")
            txt.delete("1.0", "end")
            sel = var_section.get().strip()
            if var_section_only.get() and sel and sel != "전체" and sel in sections:
                payload = sections[sel]
            else:
                payload = help_text.strip()
            txt.insert("1.0", payload + "\n")
            txt.configure(state="disabled")
            _refresh_search()

        def _refresh_search(*_args):
            q = var_q.get().strip()
            txt.configure(state="normal")
            txt.tag_remove("find_hit", "1.0", "end")
            txt.tag_remove("find_current", "1.0", "end")
            txt.configure(state="disabled")
            search_state["hits"] = []
            search_state["pos"] = -1
            if not q:
                var_info.set("검색어를 입력하면 강조됩니다.")
                return
            start = "1.0"
            while True:
                idx = txt.search(q, start, stopindex="end", nocase=1)
                if not idx:
                    break
                end = f"{idx}+{len(q)}c"
                search_state["hits"].append((idx, end))
                start = end
            txt.configure(state="normal")
            for s, e in search_state["hits"]:
                txt.tag_add("find_hit", s, e)
            txt.configure(state="disabled")
            n = len(search_state["hits"])
            var_info.set(f"검색 결과: {n}개")
            if n > 0:
                _go_hit(1)

        def _go_hit(step):
            n = len(search_state["hits"])
            if n <= 0:
                return
            search_state["pos"] = (search_state["pos"] + step) % n
            s, e = search_state["hits"][search_state["pos"]]
            txt.configure(state="normal")
            txt.tag_remove("find_current", "1.0", "end")
            txt.tag_add("find_current", s, e)
            txt.mark_set("insert", s)
            txt.see(s)
            txt.configure(state="disabled")
            var_info.set(f"검색 결과: {n}개 ({search_state['pos'] + 1}/{n})")

        def _jump_section(*_args):
            sel = var_section.get().strip()
            if not sel or sel == "전체":
                txt.see("1.0")
                return
            if var_section_only.get():
                _render_help_text()
                txt.see("1.0")
                return
            idx = txt.search(sel, "1.0", stopindex="end")
            if idx:
                txt.mark_set("insert", idx)
                txt.see(idx)

        def _apply_font_size(*_args):
            try:
                size = int(var_font_size.get())
            except Exception:
                size = 11
            size = max(9, min(18, size))
            help_font.configure(size=size)

        btn_prev.configure(command=lambda: _go_hit(-1))
        btn_next.configure(command=lambda: _go_hit(1))
        btn_jump.configure(command=_jump_section)
        var_q.trace_add("write", _refresh_search)
        var_section.trace_add("write", _jump_section)
        var_section_only.trace_add("write", lambda *_args: _render_help_text())
        var_font_size.trace_add("write", _apply_font_size)
        ent_q.bind("<Return>", lambda _e: _go_hit(1))

        _render_help_text()
        _apply_font_size()

    def on_train(self):
        self.log(f"[DBG {self._ts()}] Train button clicked.")
        self.dlog("Train button clicked.")
        self.dlog(f"Pre-check state: roi={self.cfg.roi} dataset_root={self.dataset_root}")
        if self.cfg.roi is None:
            messagebox.showerror("Error", "ROI is not set. Click 'Set ROI' first.")
            self.dlog("Train aborted: ROI not set.")
            return
        if self.dataset_root is None:
            messagebox.showerror("Error", "Dataset root is not set.")
            self.dlog("Train aborted: dataset root not set.")
            return

        try:
            self._sync_cfg_from_ui()
            epochs = int(self.var_epochs.get())
            bs = int(self.var_bs.get())
            lr = float(self.var_lr.get())
        except Exception as e:
            self.dlog(f"Train aborted: invalid UI parameter. {e}")
            self.dlog(traceback.format_exc().strip())
            messagebox.showerror("Error", f"Invalid training parameter: {e}")
            return

        self.dlog(
            f"Parsed params: epochs={epochs} batch_size={bs} lr={lr} "
            f"fps={self.cfg.fps_target} bins={self.cfg.hist_bins} win={self.cfg.window_len} stride={self.cfg.stride} "
            f"loss={self.cfg.loss_name} sched={self.cfg.scheduler_name} kfold={self.cfg.k_folds} "
            f"train_class={self.cfg.train_class_mode} "
            f"aug={self.cfg.use_augmentation} hard_mining={self.cfg.use_hard_mining} "
            f"type_os={self.cfg.use_type_oversampling} balance={self.cfg.use_balance_features} "
            f"balance_gain={self.cfg.balance_feature_gain:.2f} "
            f"tilt_detail={self.cfg.use_tilt_detail_features} tilt_gain={self.cfg.tilt_gain:.2f} "
            f"tilt_loss_alpha={self.cfg.tilt_loss_alpha:.2f} auto_hpo={self.cfg.auto_hparam_search} "
            f"hpo_trials={self.cfg.hparam_trials}"
        )

        self.stop_event.clear()
        self.progress["value"] = 0
        self.msg_q.put(("lr_reset", {"epochs": epochs, "lr": lr}))
        self.msg_q.put(("metrics_threshold", self.cfg.infer_threshold))
        self.msg_q.put(("train_target", self.cfg.train_class_mode))
        self.dlog("Training worker thread starting.")

        def worker():
            try:
                t0 = time.time()
                self.msg_q.put(("log", f"[DBG {self._ts()}] worker started"))
                self.msg_q.put(("log", f"Device: {self.cfg.device}"))
                self.msg_q.put(("log", "Scanning dataset videos..."))

                n_videos = list_videos(os.path.join(self.dataset_root, "normal"))
                d_videos = list_videos(os.path.join(self.dataset_root, "defect"))
                self.msg_q.put(("log", f"[DBG {self._ts()}] scan done in {time.time()-t0:.2f}s"))
                mode = str(self.cfg.train_class_mode).strip().lower()
                if mode not in ("both", "normal", "defect"):
                    mode = "both"
                if mode == "both":
                    if len(n_videos) == 0 or len(d_videos) == 0:
                        raise RuntimeError(f"Need videos in both normal/defect for mode=both. normal={len(n_videos)} defect={len(d_videos)}")
                    all_videos = [(p, 0) for p in n_videos] + [(p, 1) for p in d_videos]
                elif mode == "normal":
                    if len(n_videos) == 0:
                        raise RuntimeError("No videos found in normal folder for mode=normal.")
                    all_videos = [(p, 0) for p in n_videos]
                else:
                    if len(d_videos) == 0:
                        raise RuntimeError("No videos found in defect folder for mode=defect.")
                    all_videos = [(p, 1) for p in d_videos]

                self.msg_q.put(("log", f"Found normal={len(n_videos)} defect={len(d_videos)} videos | selected={len(all_videos)} (mode={mode})"))
                self.msg_q.put(("train_target", mode))
                self.msg_q.put(("log", "Extracting features (batch)..."))

                items: List[Tuple[np.ndarray, int, str]] = []

                for i, (vp, lab) in enumerate(all_videos, start=1):
                    if self.stop_event.is_set():
                        self.msg_q.put(("log", f"[DBG {self._ts()}] stop detected during feature extraction at item {i}"))
                        break
                    t_vid = time.time()
                    self.msg_q.put(("log", f"[DBG {self._ts()}] feature start {i}/{len(all_videos)} file={os.path.basename(vp)}"))
                    X = extract_features_from_video(
                        vp,
                        roi=self.cfg.roi,
                        fps_target=self.cfg.fps_target,
                        hist_bins=self.cfg.hist_bins,
                        smooth_sec=self.cfg.smooth_sec,
                        use_balance_features=self.cfg.use_balance_features,
                        balance_feature_gain=self.cfg.balance_feature_gain,
                        use_spike_features=self.cfg.use_spike_features,
                        spike_feature_gain=self.cfg.spike_feature_gain,
                        use_adaptive_pixel_smoothing=self.cfg.use_adaptive_pixel_smoothing,
                        pixel_jitter_threshold=self.cfg.pixel_jitter_threshold,
                        pixel_smoothing_kernel=self.cfg.pixel_smoothing_kernel,
                        use_tilt_detail_features=self.cfg.use_tilt_detail_features,
                        tilt_gain=self.cfg.tilt_gain,
                        log_fn=lambda s: self.msg_q.put(("log", s))
                    )
                    type_name = infer_type_name(vp, self.dataset_root, lab)
                    items.append((X, lab, type_name))
                    self.msg_q.put(("log", f"[{i}/{len(all_videos)}] features: T={X.shape[0]} D={X.shape[1]} label={'defect' if lab==1 else 'normal'} type={type_name}"))
                    self.msg_q.put(("log", f"[DBG {self._ts()}] feature done {i}/{len(all_videos)} took {time.time()-t_vid:.2f}s"))

                if len(items) < 4:
                    raise RuntimeError("Too few videos processed. Check stop/paths.")

                self.msg_q.put(("log", "Training model..."))
                self.msg_q.put(("log", f"Options: train_class={self.cfg.train_class_mode} aug={self.cfg.use_augmentation} hard_mining={self.cfg.use_hard_mining} "
                                       f"type_oversampling={self.cfg.use_type_oversampling} auto_threshold={self.cfg.auto_threshold} "
                                       f"balance_features={self.cfg.use_balance_features} balance_gain={self.cfg.balance_feature_gain:.2f} "
                                       f"spike_features={self.cfg.use_spike_features} spike_gain={self.cfg.spike_feature_gain:.2f} "
                                       f"adaptive_smoothing={self.cfg.use_adaptive_pixel_smoothing} jitter_th={self.cfg.pixel_jitter_threshold:.3f} "
                                       f"smooth_k={self.cfg.pixel_smoothing_kernel} "
                                       f"tilt_detail={self.cfg.use_tilt_detail_features} tilt_gain={self.cfg.tilt_gain:.2f} "
                                       f"tilt_loss_alpha={self.cfg.tilt_loss_alpha:.2f}"))

                train_lr = lr
                train_loss_name = self.cfg.loss_name
                train_scheduler_name = self.cfg.scheduler_name
                if self.cfg.auto_hparam_search:
                    best = search_hparams(
                        items=items,
                        base_cfg=self.cfg,
                        base_epochs=epochs,
                        base_batch_size=bs,
                        base_lr=lr,
                        trials=self.cfg.hparam_trials,
                        stop_event=self.stop_event,
                        log_cb=lambda s: self.msg_q.put(("log", s)),
                    )
                    bp = best["params"]
                    train_lr = float(bp["lr"])
                    train_loss_name = str(bp["loss_name"])
                    train_scheduler_name = str(bp["scheduler_name"])
                    self.cfg.window_len = int(bp["window_len"])
                    self.cfg.stride = int(bp["stride"])
                    self.msg_q.put(("log",
                                    f"[HPO] apply best -> lr={train_lr:.6f}, window_len={self.cfg.window_len}, "
                                    f"stride={self.cfg.stride}, loss={train_loss_name}, scheduler={train_scheduler_name}"))
                    self.msg_q.put(("log", "[HPO] best parameters are used for this training run."))

                self.msg_q.put(("log", f"[DBG {self._ts()}] calling train_model with items={len(items)}"))
                t_train = time.time()
                model, best_acc = train_model(
                    items=items,
                    cfg=self.cfg,
                    epochs=epochs,
                    batch_size=bs,
                    lr=train_lr,
                    loss_name=train_loss_name,
                    scheduler_name=train_scheduler_name,
                    use_augmentation=self.cfg.use_augmentation,
                    k_folds=self.cfg.k_folds,
                    progress_cb=lambda p: self.msg_q.put(("progress", p)),
                    lr_cb=lambda ev: self.msg_q.put(("lr_event", ev)),
                    metrics_cb=lambda m: self.msg_q.put(("metrics", m)),
                    nn_cb=lambda s: self.msg_q.put(("nn_snapshot", s)),
                    log_cb=lambda s: self.msg_q.put(("log", s)),
                    stop_event=self.stop_event
                )
                self.msg_q.put(("log", f"[DBG {self._ts()}] train_model returned in {time.time()-t_train:.2f}s best_acc={best_acc:.4f}"))
                self.msg_q.put(("metrics_threshold", self.cfg.infer_threshold))
                self.msg_q.put(("train_done", (model, best_acc)))
                self.msg_q.put(("log", f"[DBG {self._ts()}] worker finished total={time.time()-t0:.2f}s"))

            except Exception as e:
                self.msg_q.put(("log", f"[DBG {self._ts()}] worker exception: {e}"))
                self.msg_q.put(("log", traceback.format_exc().strip()))
                self.msg_q.put(("error", str(e)))

        threading.Thread(target=worker, daemon=True).start()

    def on_save_model(self):
        if self.model is None:
            messagebox.showerror("Error", "No trained model in memory. Train or Load first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save model",
            defaultextension=".pt",
            filetypes=[("PyTorch Model", "*.pt")]
        )
        if not path:
            return
        self._sync_cfg_from_ui()

        env_state = {
            "dataset_root": self.dataset_root,
            "ui": {
                "epochs": int(self.var_epochs.get()),
                "batch_size": int(self.var_bs.get()),
                "lr": float(self.var_lr.get()),
                "train_class_mode": str(self.var_train_class_mode.get()) if hasattr(self, "var_train_class_mode") else str(self.cfg.train_class_mode),
                "lr_log_scale": bool(self.var_lr_log_scale.get()) if hasattr(self, "var_lr_log_scale") else True,
            },
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        payload = {
            "cfg": asdict(self.cfg),
            "state_dict": self.model.state_dict(),
            "env": env_state,
        }
        torch.save(payload, path)
        self.model_path = path
        self.msg_q.put(("log", f"Model saved: {path}"))
        self.msg_q.put(("log", f"Saved env: dataset_root={self.dataset_root} epochs={env_state['ui']['epochs']} "
                               f"bs={env_state['ui']['batch_size']} lr={env_state['ui']['lr']} "
                               f"train_class={env_state['ui'].get('train_class_mode', self.cfg.train_class_mode)}"))

    def on_load_model(self):
        path = filedialog.askopenfilename(title="Load model", filetypes=[("PyTorch Model", "*.pt")])
        if not path:
            return
        payload = torch.load(path, map_location="cpu")
        cfg_dict = payload.get("cfg", None)
        if cfg_dict is None:
            messagebox.showerror("Error", "Invalid model file (missing cfg).")
            return

        # Restore cfg
        valid = {f.name for f in fields(AppConfig)}
        cfg_dict = {k: v for k, v in cfg_dict.items() if k in valid}
        if "roi" in cfg_dict and isinstance(cfg_dict["roi"], list):
            cfg_dict["roi"] = tuple(cfg_dict["roi"])
        self.cfg = AppConfig(**cfg_dict)
        self._apply_cfg_to_ui()

        # Build model
        # Need input D: infer from saved weights
        state = payload["state_dict"]
        # conv1 weight shape: (64, D, 5)
        w = state["net.0.weight"]
        D = int(w.shape[1])
        model = TemporalCNN(in_ch=D, num_classes=2)
        model.load_state_dict(state)
        model.to(self.cfg.device)

        self.model = model
        self.model_path = path
        self._draw_nn_graph()

        # Restore environment (optional for backward compatibility)
        env = payload.get("env", {})
        if isinstance(env, dict):
            ds = env.get("dataset_root", None)
            if isinstance(ds, str) and ds:
                self.dataset_root = ds
                self.ds_label.config(text=f"Dataset: {ds}")
            ui = env.get("ui", {})
            if isinstance(ui, dict):
                if "epochs" in ui:
                    try:
                        self.var_epochs.set(int(ui["epochs"]))
                    except Exception:
                        pass
                if "batch_size" in ui:
                    try:
                        self.var_bs.set(int(ui["batch_size"]))
                    except Exception:
                        pass
                if "lr" in ui:
                    try:
                        self.var_lr.set(float(ui["lr"]))
                    except Exception:
                        pass
                if "lr_log_scale" in ui and hasattr(self, "var_lr_log_scale"):
                    try:
                        self.var_lr_log_scale.set(bool(ui["lr_log_scale"]))
                    except Exception:
                        pass
                if "train_class_mode" in ui and hasattr(self, "var_train_class_mode"):
                    try:
                        self.var_train_class_mode.set(str(ui["train_class_mode"]))
                    except Exception:
                        pass

        self.metric_vars["train_target"].set(self._train_mode_text(self.cfg.train_class_mode))

        self.msg_q.put(("log", f"Model loaded: {path} (device={self.cfg.device})"))
        self.msg_q.put(("log", f"Restored cfg: fps={self.cfg.fps_target}, bins={self.cfg.hist_bins}, win={self.cfg.window_len}, stride={self.cfg.stride}, roi={self.cfg.roi}, train_class={self.cfg.train_class_mode}"))
        if isinstance(env, dict):
            self.msg_q.put(("log", f"Restored env: dataset_root={self.dataset_root} epochs={self.var_epochs.get()} "
                                   f"bs={self.var_bs.get()} lr={self.var_lr.get()} train_class={self.var_train_class_mode.get() if hasattr(self, 'var_train_class_mode') else self.cfg.train_class_mode}"))

    def on_infer(self):
        if self.model is None:
            messagebox.showerror("Error", "Load or Train a model first.")
            return
        if self.cfg.roi is None:
            messagebox.showerror("Error", "ROI is not set in config/model.")
            return
        paths = filedialog.askopenfilenames(
            title="Select video(s) for inference",
            filetypes=[("Video", "*.mp4 *.avi *.mkv *.mov"), ("All", "*.*")]
        )
        if not paths:
            return
        self._infer_paths(list(paths), source_label="files")

    def on_infer_folder(self):
        if self.model is None:
            messagebox.showerror("Error", "Load or Train a model first.")
            return
        if self.cfg.roi is None:
            messagebox.showerror("Error", "ROI is not set in config/model.")
            return
        root = filedialog.askdirectory(title="Select folder for batch inference")
        if not root:
            return
        paths = list_videos(root)
        if not paths:
            messagebox.showerror("Error", "No video files found in selected folder.")
            return
        self._infer_paths(paths, source_label=f"folder={root}")

    def _infer_paths(self, paths: List[str], source_label: str = ""):
        if self.model is None:
            messagebox.showerror("Error", "Load or Train a model first.")
            return
        if self.cfg.roi is None:
            messagebox.showerror("Error", "ROI is not set in config/model.")
            return
        cleaned = []
        for p in paths:
            if isinstance(p, str) and p and p.lower().endswith(VIDEO_EXTS) and os.path.isfile(p):
                cleaned.append(p)
        if not cleaned:
            messagebox.showerror("Error", "No valid video file to infer.")
            return

        def worker():
            try:
                self.msg_q.put(("log", f"[infer] start batch: n={len(cleaned)} {source_label}"))
                rows = []
                n_defect = 0
                n_normal = 0
                for i, path in enumerate(cleaned, start=1):
                    self.msg_q.put(("log", f"[infer {i}/{len(cleaned)}] extracting features: {os.path.basename(path)}"))
                    X = extract_features_from_video(
                        path,
                        roi=self.cfg.roi,
                        fps_target=self.cfg.fps_target,
                        hist_bins=self.cfg.hist_bins,
                        smooth_sec=self.cfg.smooth_sec,
                        use_balance_features=self.cfg.use_balance_features,
                        balance_feature_gain=self.cfg.balance_feature_gain,
                        use_spike_features=self.cfg.use_spike_features,
                        spike_feature_gain=self.cfg.spike_feature_gain,
                        use_adaptive_pixel_smoothing=self.cfg.use_adaptive_pixel_smoothing,
                        pixel_jitter_threshold=self.cfg.pixel_jitter_threshold,
                        pixel_smoothing_kernel=self.cfg.pixel_smoothing_kernel,
                        use_tilt_detail_features=self.cfg.use_tilt_detail_features,
                        tilt_gain=self.cfg.tilt_gain,
                        log_fn=lambda s: self.msg_q.put(("log", s))
                    )
                    res = infer_video(self.model, X, self.cfg)
                    verdict = str(res["verdict"]).upper()
                    if verdict == "DEFECT":
                        n_defect += 1
                    else:
                        n_normal += 1
                    self.msg_q.put(("log", f"[infer {i}/{len(cleaned)}] {os.path.basename(path)} -> {verdict} "
                                           f"(def={res['prob_defect']:.3f}, nor={res['prob_normal']:.3f}, thr={res['threshold']:.2f}, win={int(res['num_windows'])})"))
                    rows.append(
                        f"{i}. {os.path.basename(path)} -> {verdict} | "
                        f"def={res['prob_defect']:.3f}, nor={res['prob_normal']:.3f}, thr={res['threshold']:.2f}, win={int(res['num_windows'])}"
                    )

                preview_n = 20
                preview = "\n".join(rows[:preview_n])
                if len(rows) > preview_n:
                    preview += f"\n... ({len(rows) - preview_n} more)"
                messagebox.showinfo(
                    "Inference Batch Result",
                    f"Source: {source_label or '-'}\n"
                    f"Total: {len(cleaned)}\n"
                    f"DEFECT: {n_defect}\n"
                    f"NORMAL: {n_normal}\n\n"
                    f"{preview}"
                )
            except Exception as e:
                self.msg_q.put(("error", str(e)))

        threading.Thread(target=worker, daemon=True).start()


if __name__ == "__main__":
    try:
        app = App()
        app.mainloop()
    except KeyboardInterrupt:
        # Allow clean stop from terminal without traceback noise.
        pass

