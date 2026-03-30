from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional


_VERSION_FILE = ".version"


class AppManager:
    """Manages installed applications on the local filesystem."""

    def __init__(self, apps_dir: str) -> None:
        self._apps_dir = Path(apps_dir)
        self._apps_dir.mkdir(parents=True, exist_ok=True)

    @property
    def apps_dir(self) -> Path:
        return self._apps_dir

    def is_installed(self, app_id: str) -> bool:
        """Return True if the app folder exists."""
        return (self._apps_dir / app_id).is_dir()

    def get_installed_version(self, app_id: str) -> Optional[str]:
        """Read .version file from app folder, or return None."""
        version_file = self._apps_dir / app_id / _VERSION_FILE
        if version_file.exists():
            try:
                return version_file.read_text(encoding="utf-8").strip()
            except OSError:
                pass
        return None

    def save_version(self, app_id: str, version: str) -> None:
        """Write version string to .version file in app folder."""
        app_folder = self._apps_dir / app_id
        app_folder.mkdir(parents=True, exist_ok=True)
        version_file = app_folder / _VERSION_FILE
        try:
            version_file.write_text(version.strip(), encoding="utf-8")
        except OSError as e:
            print(f"[app_manager] Warning: could not save version for {app_id}: {e}")

    def get_entry_path(self, app_id: str, entry: str) -> Path:
        """기대 경로 먼저 확인, 없으면 앱 폴더 전체 재귀 탐색."""
        expected = self._apps_dir / app_id / entry
        if expected.exists():
            return expected
        # 중첩 폴더에 풀린 경우 재귀 탐색으로 찾기
        app_folder = self._apps_dir / app_id
        filename = Path(entry).name
        for found in app_folder.rglob(filename):
            if found.is_file():
                print(f"[app_manager] Entry found at: {found}")
                return found
        return expected   # 없어도 기대 경로 반환 (오류 메시지용)

    def launch(self, app_id: str, entry: str) -> bool:
        """
        Launch the entry file using subprocess.Popen.
        Returns True on success, False on failure.
        """
        entry_path = self.get_entry_path(app_id, entry)
        if not entry_path.exists():
            print(f"[app_manager] Entry file not found: {entry_path}")
            return False
        try:
            suffix = entry_path.suffix.lower()
            if suffix == ".py":
                proc = subprocess.Popen(
                    [sys.executable, str(entry_path)],
                    cwd=str(entry_path.parent),
                    creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0,
                )
            elif suffix == ".exe":
                proc = subprocess.Popen(
                    [str(entry_path)],
                    cwd=str(entry_path.parent),
                    creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0,
                )
            elif suffix == ".bat":
                proc = subprocess.Popen(
                    [str(entry_path)],
                    cwd=str(entry_path.parent),
                    shell=True,
                )
            else:
                # Generic: try to open with OS default
                proc = subprocess.Popen(
                    [str(entry_path)],
                    cwd=str(entry_path.parent),
                    shell=True,
                )
            print(f"[app_manager] Launched {app_id} (PID {proc.pid})")
            return True
        except OSError as e:
            print(f"[app_manager] Failed to launch {app_id}: {e}")
            return False

    def uninstall(self, app_id: str) -> None:
        """Remove the app folder entirely."""
        app_folder = self._apps_dir / app_id
        if app_folder.exists():
            try:
                shutil.rmtree(str(app_folder))
                print(f"[app_manager] Uninstalled {app_id}")
            except OSError as e:
                print(f"[app_manager] Failed to uninstall {app_id}: {e}")
                raise
        else:
            print(f"[app_manager] App {app_id} not found, nothing to uninstall.")
