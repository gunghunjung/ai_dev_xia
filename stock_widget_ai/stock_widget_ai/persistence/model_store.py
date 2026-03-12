"""
Model Store — 모델 저장/불러오기 유틸리티
"""
from __future__ import annotations
import os
import torch
from typing import Optional
from ..logger_config import get_logger

log = get_logger("persistence.model_store")


def save_model(model: torch.nn.Module, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(model.state_dict(), path)
    log.info(f"모델 저장: {path}")


def load_model(model: torch.nn.Module, path: str, device: Optional[str] = "cpu") -> torch.nn.Module:
    if not os.path.exists(path):
        raise FileNotFoundError(f"모델 파일 없음: {path}")
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=False)
    log.info(f"모델 로드: {path}")
    return model
