"""
공통 유틸리티
"""
from __future__ import annotations
import os
import torch
from typing import Optional
from .logger_config import get_logger

log = get_logger("utils")


def resolve_device(pref: str = "auto") -> torch.device:
    if pref == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        log.warning("CUDA 요청했으나 사용 불가 → CPU 폴백")
        return torch.device("cpu")
    if pref == "cpu":
        return torch.device("cpu")
    # auto
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        log.info(f"GPU 감지: {torch.cuda.get_device_name(0)}")
    else:
        dev = torch.device("cpu")
        log.info("CPU 모드")
    return dev


def ensure_dirs(*paths: str) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)


def clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def set_seed(seed: int) -> None:
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
