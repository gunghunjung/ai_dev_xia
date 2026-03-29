from __future__ import annotations

import io
import zipfile
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from api.auth import verify_token

router = APIRouter(prefix="/download", tags=["download"])

# Module-level scanner reference — set by main.py at startup
_scanner = None


def set_scanner(scanner) -> None:
    """Called by main.py to inject scanner into this module."""
    global _scanner
    _scanner = scanner


def _zip_folder(folder: Path) -> tuple[bytes, int]:
    """폴더 내용물만 ZIP — 최상위 폴더명 없이 파일 바로 들어감.

    예) AshfallProtocol/AshfallProtocol.exe  →  ZIP: AshfallProtocol.exe
    → 클라이언트: apps/<app_id>/AshfallProtocol.exe
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(folder.rglob("*")):
            if file_path.is_file():
                arcname = file_path.relative_to(folder)  # 폴더명 제외
                try:
                    zf.write(file_path, arcname=str(arcname))
                except OSError as e:
                    print(f"[download] Warning: could not add {file_path} to zip: {e}")
    data = buf.getvalue()
    return data, len(data)


@router.get("/{app_id}")
async def download_app(
    app_id: str,
    _token: str = Depends(verify_token),
) -> StreamingResponse:
    """Stream a ZIP archive of the app folder for download."""
    if _scanner is None:
        raise HTTPException(status_code=503, detail="Scanner not initialized")
    app = _scanner.get_app(app_id)
    if app is None:
        raise HTTPException(status_code=404, detail=f"App '{app_id}' not found")

    app_path = _scanner.get_app_path(app_id)
    if not app_path.exists() or not app_path.is_dir():
        raise HTTPException(status_code=404, detail=f"App folder not found on disk")

    try:
        zip_data, zip_size = _zip_folder(app_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create archive: {e}")

    def _iter_zip():
        chunk_size = 64 * 1024  # 64 KB chunks
        offset = 0
        while offset < len(zip_data):
            yield zip_data[offset : offset + chunk_size]
            offset += chunk_size

    headers = {
        "Content-Disposition": f'attachment; filename="{app_id}.zip"',
        "Content-Length": str(zip_size),
    }
    return StreamingResponse(
        _iter_zip(),
        media_type="application/zip",
        headers=headers,
    )
