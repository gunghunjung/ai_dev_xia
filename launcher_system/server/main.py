from __future__ import annotations

import argparse
import asyncio
import json
import sys
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from config import ServerConfig, load_config, save_config
from scanner import AppScanner
import api.auth as auth_module
import api.apps as apps_module
import api.download as download_module
from api.apps import router as apps_router, version_router
from api.download import router as download_router


# ─── Globals ────────────────────────────────────────────────────────────────
_config: ServerConfig = ServerConfig()
_scanner: AppScanner | None = None
_ws_clients: list[WebSocket] = []


# ─── WebSocket broadcast ─────────────────────────────────────────────────────
async def _broadcast_update() -> None:
    """Send updated app list to all connected WebSocket clients."""
    if not _ws_clients or _scanner is None:
        return
    apps_data = [app.model_dump() for app in _scanner.get_all()]
    message = json.dumps({"event": "update", "apps": apps_data})
    disconnected: list[WebSocket] = []
    for ws in list(_ws_clients):
        try:
            await ws.send_text(message)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        if ws in _ws_clients:
            _ws_clients.remove(ws)


def _on_fs_change() -> None:
    """Called by watchdog on file system change; re-scans and broadcasts."""
    if _scanner is None:
        return
    print("[server] File system change detected, re-scanning...")
    _scanner.scan()
    # Schedule coroutine in the running event loop (thread-safe)
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.run_coroutine_threadsafe(_broadcast_update(), loop)
    except RuntimeError:
        pass


# ─── Lifespan ────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _scanner, _config

    # Inject config and scanner into sub-modules
    auth_module.set_config(_config)

    root = _config.root_folder or "./apps"
    if not _config.root_folder:
        print("[server] No root_folder configured — using './apps' as default.")
        print("[server] To configure, create launcher_server.json or use --root CLI arg.")

    print(f"[server] Scanning app folder: {root}")
    _scanner = AppScanner(root)
    apps_module.set_scanner(_scanner)
    download_module.set_scanner(_scanner)

    n = len(_scanner.get_all())
    print(f"[server] Found {n} app(s).")

    _scanner.start_watch(_on_fs_change)

    yield

    # Cleanup
    if _scanner:
        _scanner.stop_watch()
    print("[server] Shutdown complete.")


# ─── App factory ─────────────────────────────────────────────────────────────
def create_app() -> FastAPI:
    app = FastAPI(title="Launcher Server", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(apps_router)
    app.include_router(version_router)
    app.include_router(download_router)

    @app.get("/")
    async def root() -> dict[str, Any]:
        return {"service": "Launcher Server", "status": "running"}

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        _ws_clients.append(websocket)
        print(f"[ws] Client connected. Total: {len(_ws_clients)}")
        try:
            while True:
                # Keep connection alive; we only push from server side
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30)
                # Echo ping/pong if client sends "ping"
                if data.strip().lower() == "ping":
                    await websocket.send_text("pong")
        except asyncio.TimeoutError:
            # Send keepalive ping
            try:
                await websocket.send_text(json.dumps({"event": "ping"}))
            except Exception:
                pass
        except WebSocketDisconnect:
            pass
        except Exception as e:
            print(f"[ws] Error: {e}")
        finally:
            if websocket in _ws_clients:
                _ws_clients.remove(websocket)
            print(f"[ws] Client disconnected. Total: {len(_ws_clients)}")

    return app


# ─── CLI entry point ─────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launcher Server")
    parser.add_argument("--root", type=str, default=None, help="App root folder path")
    parser.add_argument("--port", type=int, default=None, help="Server port")
    parser.add_argument("--token", type=str, default=None, help="Auth token")
    parser.add_argument("--host", type=str, default=None, help="Bind host")
    return parser.parse_args()


def main() -> None:
    global _config
    args = parse_args()
    _config = load_config()

    # Override with CLI args
    if args.root is not None:
        _config.root_folder = args.root
    if args.port is not None:
        _config.port = args.port
    if args.token is not None:
        _config.token = args.token
    if args.host is not None:
        _config.host = args.host

    save_config(_config)

    # Print startup info
    print("=" * 60)
    print("  Launcher Server")
    print("=" * 60)
    print(f"  URL   : http://{_config.host}:{_config.port}")
    print(f"  Token : {_config.token}")
    print(f"  Root  : {_config.root_folder or './apps (default)'}")
    print("=" * 60)

    app = create_app()
    uvicorn.run(
        app,
        host=_config.host,
        port=_config.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
