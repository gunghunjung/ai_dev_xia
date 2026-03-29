from __future__ import annotations

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

# Module-level config reference — set by main.py at startup
_config = None

security = HTTPBearer()


def set_config(cfg) -> None:
    """Called by main.py to inject config into this module."""
    global _config
    _config = cfg


def verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    """FastAPI dependency: verify Bearer token against server config."""
    if _config is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server not configured",
        )
    if credentials.credentials != _config.token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials
