from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class ServerConfig:
    root_folder: str = ""
    host: str = "0.0.0.0"
    port: int = 8765
    token: str = "launcher-token"


_CONFIG_FILE = "launcher_server.json"


def load_config() -> ServerConfig:
    """Load config from launcher_server.json in CWD, or return defaults."""
    config_path = Path(os.getcwd()) / _CONFIG_FILE
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            cfg = ServerConfig()
            cfg.root_folder = data.get("root_folder", cfg.root_folder)
            cfg.host = data.get("host", cfg.host)
            cfg.port = int(data.get("port", cfg.port))
            cfg.token = data.get("token", cfg.token)
            return cfg
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"[config] Warning: failed to load config ({e}), using defaults.")
    return ServerConfig()


def save_config(cfg: ServerConfig) -> None:
    """Save config to launcher_server.json in CWD."""
    config_path = Path(os.getcwd()) / _CONFIG_FILE
    try:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)
    except OSError as e:
        print(f"[config] Warning: failed to save config ({e})")
