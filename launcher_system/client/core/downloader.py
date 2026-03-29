from __future__ import annotations

import io
import zipfile
from pathlib import Path

import requests
from PyQt6.QtCore import QThread, pyqtSignal

from core.api_client import ApiClient


class Downloader(QThread):
    """QThread that downloads an app ZIP from the server and extracts it."""

    # Signals
    progress = pyqtSignal(int)          # 0–100 percent
    finished = pyqtSignal(str, str)     # app_id, local_path
    error = pyqtSignal(str, str)        # app_id, error_message

    def __init__(self, api_client: ApiClient, app_id: str, dest_dir: str) -> None:
        super().__init__()
        self._api_client = api_client
        self._app_id = app_id
        self._dest_dir = Path(dest_dir)
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        """Download and extract the app. Emits progress/finished/error."""
        try:
            resp = self._api_client.get_download_stream(self._app_id)

            # Determine total size from Content-Length header
            total_size = int(resp.headers.get("Content-Length", 0))
            buf = io.BytesIO()
            downloaded = 0
            chunk_size = 65536  # 64 KB

            for chunk in resp.iter_content(chunk_size=chunk_size):
                if self._cancelled:
                    self.error.emit(self._app_id, "Download cancelled by user")
                    return
                if chunk:
                    buf.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = int(downloaded * 100 / total_size)
                        self.progress.emit(min(pct, 99))

            # Extract ZIP
            self.progress.emit(99)
            buf.seek(0)
            extract_path = self._dest_dir / self._app_id
            extract_path.mkdir(parents=True, exist_ok=True)

            with zipfile.ZipFile(buf, "r") as zf:
                # 서버가 폴더명 없이 내용물만 ZIP으로 보내므로 그대로 풀면 됨
                for member in zf.namelist():
                    if self._cancelled:
                        self.error.emit(self._app_id, "Extraction cancelled by user")
                        return
                    if member.endswith("/"):
                        continue   # 디렉토리 엔트리 스킵
                    target = extract_path / member
                    target.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        with zf.open(member) as src, open(target, "wb") as dst:
                            dst.write(src.read())
                    except (KeyError, OSError) as e:
                        print(f"[downloader] Warning: could not extract {member}: {e}")

            self.progress.emit(100)
            self.finished.emit(self._app_id, str(extract_path))

        except requests.RequestException as e:
            self.error.emit(self._app_id, f"Network error: {e}")
        except zipfile.BadZipFile as e:
            self.error.emit(self._app_id, f"Invalid ZIP archive: {e}")
        except OSError as e:
            self.error.emit(self._app_id, f"File system error: {e}")
        except Exception as e:
            self.error.emit(self._app_id, f"Unexpected error: {e}")
