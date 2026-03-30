from __future__ import annotations

from typing import Optional

import requests


class ApiClient:
    """HTTP client for the Launcher Server REST API."""

    def __init__(self, base_url: str, token: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._token = token
        self._session = requests.Session()
        self._session.headers.update({"Authorization": f"Bearer {token}"})

    @property
    def base_url(self) -> str:
        return self._base_url

    @property
    def token(self) -> str:
        return self._token

    def _url(self, path: str) -> str:
        return f"{self._base_url}{path}"

    def get_apps(self) -> list[dict]:
        """GET /apps → list of app metadata dicts."""
        resp = self._session.get(self._url("/apps"), timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data.get("apps", [])

    def get_app(self, app_id: str) -> dict:
        """GET /apps/{id} → app metadata dict."""
        resp = self._session.get(self._url(f"/apps/{app_id}"), timeout=10)
        resp.raise_for_status()
        return resp.json()

    def get_icon(self, app_id: str) -> Optional[bytes]:
        """GET /apps/{id}/icon → raw bytes, or None on error."""
        try:
            resp = self._session.get(self._url(f"/apps/{app_id}/icon"), timeout=10)
            if resp.status_code == 200:
                return resp.content
            return None
        except requests.RequestException:
            return None

    def get_title_image(self, app_id: str) -> Optional[bytes]:
        """GET /apps/{id}/title → raw bytes of title.jpg, or None on error."""
        try:
            resp = self._session.get(self._url(f"/apps/{app_id}/title"), timeout=10)
            if resp.status_code == 200:
                return resp.content
            return None
        except requests.RequestException:
            return None

    def version_check(self, app_id: str, version: str) -> dict:
        """GET /version-check?id=X&version=Y → VersionCheckResult dict."""
        resp = self._session.get(
            self._url("/version-check"),
            params={"id": app_id, "version": version},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()

    def download_url(self, app_id: str) -> str:
        """Return full URL for /download/{id}."""
        return self._url(f"/download/{app_id}")

    def test_connection(self) -> bool:
        """Simple GET /apps; returns True if status 200."""
        try:
            resp = self._session.get(self._url("/apps"), timeout=5)
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def get_download_stream(self, app_id: str) -> requests.Response:
        """Return a streaming Response for downloading an app ZIP."""
        resp = self._session.get(
            self.download_url(app_id),
            stream=True,
            timeout=60,
        )
        resp.raise_for_status()
        return resp
