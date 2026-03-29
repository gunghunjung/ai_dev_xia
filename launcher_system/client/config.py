from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path


# 스크립트 위치 기준 절대경로 — CWD와 무관하게 항상 같은 위치
_BASE_DIR = Path(__file__).resolve().parent
_CONFIG_FILE = _BASE_DIR / "launcher_client.json"
_DEFAULT_APPS_DIR = str(_BASE_DIR / "apps")


@dataclass
class ClientConfig:
    server_url: str = ""
    token: str = ""
    apps_dir: str = _DEFAULT_APPS_DIR
    last_connected: bool = False


def load_config() -> ClientConfig:
    """launcher_client.json 로드, 없으면 기본값."""
    if _CONFIG_FILE.exists():
        try:
            with open(_CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            cfg = ClientConfig()
            cfg.server_url     = data.get("server_url", cfg.server_url)
            cfg.token          = data.get("token", cfg.token)
            cfg.apps_dir       = data.get("apps_dir", cfg.apps_dir)
            cfg.last_connected = bool(data.get("last_connected", cfg.last_connected))
            return cfg
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"[config] Warning: failed to load config ({e}), using defaults.")
    return ClientConfig()


def save_config(cfg: ClientConfig) -> None:
    """launcher_client.json 저장."""
    try:
        with open(_CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)
    except OSError as e:
        print(f"[config] Warning: failed to save config ({e})")
