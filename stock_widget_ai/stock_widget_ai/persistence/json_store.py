"""
JSON Store — 범용 key-value JSON 저장소
"""
from __future__ import annotations
import json
import os
from typing import Any


class JsonStore:
    def __init__(self, path: str) -> None:
        self._path = path
        self._data: dict = {}
        self._load()

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value

    def save(self) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self._path)), exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False, default=str)

    def _load(self) -> None:
        if os.path.exists(self._path):
            try:
                with open(self._path, encoding="utf-8") as f:
                    self._data = json.load(f)
            except Exception:
                self._data = {}
