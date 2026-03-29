from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse

from api.auth import verify_token
from models import AppList, AppMeta, VersionCheckResult

router = APIRouter(prefix="/apps", tags=["apps"])

# Module-level scanner reference — set by main.py at startup
_scanner = None


def set_scanner(scanner) -> None:
    """Called by main.py to inject scanner into this module."""
    global _scanner
    _scanner = scanner


@router.get("", response_model=AppList)
async def list_apps(_token: str = Depends(verify_token)) -> AppList:
    """Return list of all available apps."""
    if _scanner is None:
        raise HTTPException(status_code=503, detail="Scanner not initialized")
    apps = _scanner.get_all()
    return AppList(apps=apps, total=len(apps))


@router.get("/{app_id}", response_model=AppMeta)
async def get_app(app_id: str, _token: str = Depends(verify_token)) -> AppMeta:
    """Return metadata for a single app."""
    if _scanner is None:
        raise HTTPException(status_code=503, detail="Scanner not initialized")
    app = _scanner.get_app(app_id)
    if app is None:
        raise HTTPException(status_code=404, detail=f"App '{app_id}' not found")
    return app


@router.get("/{app_id}/icon")
async def get_app_icon(app_id: str, _token: str = Depends(verify_token)) -> FileResponse:
    """Return icon file for an app, or 404 if not found."""
    if _scanner is None:
        raise HTTPException(status_code=503, detail="Scanner not initialized")
    app = _scanner.get_app(app_id)
    if app is None:
        raise HTTPException(status_code=404, detail=f"App '{app_id}' not found")
    if not app.icon:
        raise HTTPException(status_code=404, detail=f"App '{app_id}' has no icon")
    app_path = _scanner.get_app_path(app_id)
    icon_path = app_path / app.icon
    if not icon_path.exists():
        raise HTTPException(status_code=404, detail="Icon file not found on disk")
    return FileResponse(str(icon_path))


@router.get("/{app_id}/title")
async def get_app_title_image(app_id: str, _token: str = Depends(verify_token)) -> FileResponse:
    """Return title image (title.jpg) for an app, or 404 if not found."""
    if _scanner is None:
        raise HTTPException(status_code=503, detail="Scanner not initialized")
    app = _scanner.get_app(app_id)
    if app is None:
        raise HTTPException(status_code=404, detail=f"App '{app_id}' not found")
    if not app.title_image:
        raise HTTPException(status_code=404, detail=f"App '{app_id}' has no title image")
    app_path = _scanner.get_app_path(app_id)
    title_path = app_path / app.title_image
    if not title_path.exists():
        raise HTTPException(status_code=404, detail="Title image not found on disk")
    return FileResponse(str(title_path))


# Version check is outside the /apps prefix — registered separately
version_router = APIRouter(tags=["version"])


@version_router.get("/version-check", response_model=VersionCheckResult)
async def version_check(
    id: str = Query(..., description="App ID"),
    version: str = Query(..., description="Installed version"),
    _token: str = Depends(verify_token),
) -> VersionCheckResult:
    """Check if an installed version is up to date."""
    if _scanner is None:
        raise HTTPException(status_code=503, detail="Scanner not initialized")
    app = _scanner.get_app(id)
    if app is None:
        raise HTTPException(status_code=404, detail=f"App '{id}' not found")

    def _version_tuple(v: str) -> tuple[int, ...]:
        try:
            return tuple(int(x) for x in v.strip().split("."))
        except ValueError:
            return (0,)

    update_available = _version_tuple(app.version) > _version_tuple(version)
    return VersionCheckResult(
        id=id,
        current=version,
        latest=app.version,
        update_available=update_available,
    )
