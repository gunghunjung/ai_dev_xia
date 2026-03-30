from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Callable, Optional

from watchdog.events import FileSystemEventHandler, FileSystemEvent
from watchdog.observers import Observer

from models import AppMeta


_ICON_NAMES  = ("icon.png", "icon.jpg", "icon.ico")
_TITLE_NAMES = ("title.jpg", "title.jpeg", "title.png")
_ENTRY_EXTENSIONS = (".exe", ".py", ".bat")
_METADATA_FILE = "metadata.json"


def _compute_folder_hash(folder: Path) -> str:
    """Compute SHA256 hash of all file contents in a folder, sorted by name."""
    hasher = hashlib.sha256()
    try:
        files = sorted(folder.rglob("*"))
        for f in files:
            if f.is_file() and f.name != _METADATA_FILE:
                try:
                    hasher.update(f.name.encode("utf-8"))
                    hasher.update(f.read_bytes())
                except OSError:
                    pass
    except OSError:
        pass
    return hasher.hexdigest()


def _compute_folder_size(folder: Path) -> int:
    """Return total size of all files in folder."""
    total = 0
    try:
        for f in folder.rglob("*"):
            if f.is_file():
                try:
                    total += f.stat().st_size
                except OSError:
                    pass
    except OSError:
        pass
    return total


def _find_entry(folder: Path) -> str:
    """Find first .exe, .py, or .bat in folder."""
    for ext in _ENTRY_EXTENSIONS:
        for f in folder.iterdir():
            if f.is_file() and f.suffix.lower() == ext:
                return f.name
    return ""


def _find_icon(folder: Path) -> Optional[str]:
    """Find icon file in folder."""
    for name in _ICON_NAMES:
        if (folder / name).exists():
            return name
    return None


def _find_title_image(folder: Path) -> Optional[str]:
    """Find title.jpg / title.jpeg / title.png in folder."""
    for name in _TITLE_NAMES:
        if (folder / name).exists():
            return name
    return None


def _folder_to_id(folder_name: str) -> str:
    """Convert folder name to app id."""
    return folder_name.lower().replace(" ", "_")


def _load_or_create_metadata(folder: Path, app_id: str) -> dict:
    """Load metadata.json or generate defaults, auto-creating the file if missing."""
    meta_path = folder / _METADATA_FILE
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Validate required keys exist (fill missing ones)
            data.setdefault("id", app_id)
            data.setdefault("name", folder.name)
            data.setdefault("version", "1.0.0")
            data.setdefault("description", f"{folder.name} application")
            return data
        except (json.JSONDecodeError, OSError) as e:
            print(f"[scanner] Warning: corrupt metadata.json in {folder} ({e}), regenerating.")

    # Auto-generate defaults
    entry = _find_entry(folder)
    icon = _find_icon(folder)
    data = {
        "id": app_id,
        "name": folder.name,
        "version": "1.0.0",
        "description": f"{folder.name} application",
        "entry": entry,
        "icon": icon,
    }
    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except OSError as e:
        print(f"[scanner] Warning: could not write metadata.json to {folder} ({e})")
    return data


class _WatchHandler(FileSystemEventHandler):
    def __init__(self, callback: Callable[[], None]) -> None:
        super().__init__()
        self._callback = callback
        self._last_call = 0.0
        self._debounce = 1.0  # seconds

    def on_any_event(self, event: FileSystemEvent) -> None:
        now = time.time()
        if now - self._last_call > self._debounce:
            self._last_call = now
            try:
                self._callback()
            except Exception as e:
                print(f"[scanner] Watch callback error: {e}")


class AppScanner:
    def __init__(self, root: str) -> None:
        self._root = Path(root)
        self._registry: dict[str, AppMeta] = {}
        self._observer: Optional[Observer] = None
        self._root.mkdir(parents=True, exist_ok=True)
        self._registry = self.scan()

    def scan(self) -> dict[str, AppMeta]:
        """Scan root folder and return dict[app_id, AppMeta]."""
        registry: dict[str, AppMeta] = {}
        try:
            entries = list(self._root.iterdir())
        except OSError as e:
            print(f"[scanner] Cannot read root folder ({e})")
            return registry

        for folder in entries:
            if not folder.is_dir():
                continue
            app_id = _folder_to_id(folder.name)
            try:
                meta_data = _load_or_create_metadata(folder, app_id)
                entry = meta_data.get("entry") or _find_entry(folder)
                icon = meta_data.get("icon") or _find_icon(folder)
                title_image = meta_data.get("title_image") or _find_title_image(folder)
                size_bytes = _compute_folder_size(folder)
                hash_sha256 = _compute_folder_hash(folder)
                try:
                    mtime = folder.stat().st_mtime
                except OSError:
                    mtime = time.time()

                # 실행파일명(확장자 제거) 우선, 없으면 폴더명
                display_name = Path(entry).stem if entry else folder.name

                app = AppMeta(
                    id=app_id,
                    name=display_name,
                    version=meta_data.get("version", "1.0.0"),
                    description=meta_data.get("description", f"{folder.name} application"),
                    entry=entry,
                    icon=icon,
                    title_image=title_image,
                    size_bytes=size_bytes,
                    hash_sha256=hash_sha256,
                    updated_at=mtime,
                )
                registry[app_id] = app
            except Exception as e:
                print(f"[scanner] Error scanning {folder}: {e}")

        self._registry = registry
        return registry

    def get_app(self, app_id: str) -> Optional[AppMeta]:
        return self._registry.get(app_id)

    def get_all(self) -> list[AppMeta]:
        return list(self._registry.values())

    def get_app_path(self, app_id: str) -> Path:
        """Return path to app folder. Reconstructs folder name from id."""
        # Try to find the actual folder matching this id
        try:
            for folder in self._root.iterdir():
                if folder.is_dir() and _folder_to_id(folder.name) == app_id:
                    return folder
        except OSError:
            pass
        # Fallback: return expected path
        return self._root / app_id

    def start_watch(self, callback: Callable[[], None]) -> None:
        """Start watchdog to monitor root folder."""
        if self._observer is not None:
            return
        handler = _WatchHandler(callback)
        self._observer = Observer()
        self._observer.schedule(handler, str(self._root), recursive=True)
        self._observer.start()
        print(f"[scanner] Watching {self._root} for changes...")

    def stop_watch(self) -> None:
        """Stop watchdog observer."""
        if self._observer is not None:
            try:
                self._observer.stop()
                self._observer.join(timeout=5)
            except Exception as e:
                print(f"[scanner] Error stopping watcher: {e}")
            finally:
                self._observer = None
            print("[scanner] File watcher stopped.")
