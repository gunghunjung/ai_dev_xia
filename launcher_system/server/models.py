from __future__ import annotations

from typing import Optional
from pydantic import BaseModel


class AppMeta(BaseModel):
    id: str
    name: str
    version: str
    description: str
    entry: str
    icon: Optional[str] = None
    title_image: Optional[str] = None   # title.jpg / title.png
    size_bytes: int
    hash_sha256: str
    updated_at: float


class AppList(BaseModel):
    apps: list[AppMeta]
    total: int


class VersionCheckResult(BaseModel):
    id: str
    current: str
    latest: str
    update_available: bool
