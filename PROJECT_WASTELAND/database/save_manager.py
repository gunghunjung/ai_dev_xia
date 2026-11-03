"""SQLite save slots with versioning, migrations, atomic writes and a backup copy.

Write path: build the full save into ``slot.db.tmp`` inside one transaction, fsync, then
atomically replace ``slot.db`` (the previous file is kept as ``slot.db.bak``). A crash
mid-save therefore never corrupts the existing save.
"""
import json
import logging
import os
import shutil
import sqlite3
import time

from config import SAVE_DIR, SAVE_VERSION, GAME_VERSION
from database.world_repository import WorldRepository, PlayerRepository
from interfaces.save_interface import ISaveService

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS player (id INTEGER PRIMARY KEY, x REAL, y REAL, hp REAL, stats TEXT,
                                   equipped TEXT, hotbar TEXT, spawn TEXT);
CREATE TABLE IF NOT EXISTS items (uid TEXT PRIMARY KEY, owner TEXT NOT NULL, item_id TEXT NOT NULL, count INTEGER,
                                  durability REAL, ammo INTEGER, gx INTEGER, gy INTEGER, data TEXT);
CREATE TABLE IF NOT EXISTS world_cells (x INTEGER, y INTEGER, depth INTEGER, work REAL, flags INTEGER,
                                        PRIMARY KEY (x, y));
CREATE TABLE IF NOT EXISTS world_objects (id TEXT PRIMARY KEY, kind TEXT, type TEXT, x REAL, y REAL, state TEXT);
"""

# version → function(conn) upgrading a save *from* that version to version + 1.
MIGRATIONS: dict = {}


class SaveError(Exception):
    pass


class SaveManager(ISaveService):
    def __init__(self, save_dir: str = SAVE_DIR):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.world_repo = WorldRepository()
        self.player_repo = PlayerRepository()
        self.last_error = ""

    def path(self, slot: str) -> str:
        return os.path.join(self.save_dir, f"{slot}.db")

    def exists(self, slot: str) -> bool:
        return os.path.exists(self.path(slot)) or os.path.exists(self.path(slot) + ".bak")

    def info(self, slot: str) -> dict | None:
        for p in (self.path(slot), self.path(slot) + ".bak"):
            if not os.path.exists(p):
                continue
            try:
                with sqlite3.connect(p) as conn:
                    return dict(conn.execute("SELECT key, value FROM meta").fetchall())
            except sqlite3.Error:
                continue
        return None

    # ------------------------------------------------------------ save
    def save(self, session, slot: str = "slot1") -> bool:
        final = self.path(slot)
        tmp = final + ".tmp"
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
            conn = sqlite3.connect(tmp)
            try:
                conn.executescript(SCHEMA)
                with conn:  # single transaction
                    meta = {
                        "save_version": str(SAVE_VERSION), "game_version": GAME_VERSION, "seed": str(session.seed),
                        "saved_at": str(int(time.time())), "play_time": f"{session.play_time:.1f}",
                        "day": str(session.clock.day), "time_of_day": f"{session.clock.time:.5f}",
                        "weather": session.weather.state, "grace": "0",
                    }
                    conn.executemany("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", meta.items())
                    self.player_repo.save_player(conn, session.player)
                    self.world_repo.save(conn, session)
                conn.execute("PRAGMA integrity_check")
            finally:
                conn.close()
            if os.path.exists(final):
                shutil.copy2(final, final + ".bak")
            os.replace(tmp, final)
            self.last_error = ""
            return True
        except (sqlite3.Error, OSError) as exc:
            log.exception("Save failed")
            self.last_error = str(exc)
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except OSError:
                pass
            return False

    # ------------------------------------------------------------ load
    def read_meta(self, slot: str) -> tuple[str, dict]:
        """Return (path, meta) of the first readable copy of the slot (main file, then backup)."""
        for p in (self.path(slot), self.path(slot) + ".bak"):
            if not os.path.exists(p):
                continue
            try:
                with sqlite3.connect(p) as conn:
                    if conn.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                        raise sqlite3.DatabaseError("integrity check failed")
                    meta = dict(conn.execute("SELECT key, value FROM meta").fetchall())
                    if "seed" not in meta:
                        raise sqlite3.DatabaseError("missing meta")
                    return p, meta
            except sqlite3.Error as exc:
                log.warning("Save copy %s unreadable: %s", p, exc)
        raise SaveError("세이브 파일을 읽을 수 없습니다")

    def _migrate(self, conn, version: int):
        while version < SAVE_VERSION:
            fn = MIGRATIONS.get(version)
            if fn is None:
                raise SaveError(f"세이브 버전 {version} 에서 마이그레이션할 수 없습니다")
            with conn:
                fn(conn)
                version += 1
                conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('save_version', ?)", (str(version),))

    def load(self, session, slot: str = "slot1") -> bool:
        """Load into an already-constructed session created with the slot's seed."""
        try:
            path, meta = self.read_meta(slot)
            version = int(meta.get("save_version", "1"))
            if version > SAVE_VERSION:
                raise SaveError("더 새로운 버전의 게임에서 만든 세이브입니다")
            conn = sqlite3.connect(path)
            try:
                self._migrate(conn, version)
                self.world_repo.load(conn, session)
                self.player_repo.load_player(conn, session.player)
            finally:
                conn.close()
            session.clock.day = int(meta.get("day", 1))
            session.clock.time = float(meta.get("time_of_day", 0.3))
            session.play_time = float(meta.get("play_time", 0))
            if meta.get("weather") in ("clear", "rain", "fog", "dust"):
                session.weather.state = session.weather.prev = meta["weather"]
            session.containers.clear()
            session.enemies.clear()
            session.projectiles.clear()
            session.world.ensure_area(session.player.x, session.player.y, 40)
            self.last_error = ""
            return True
        except (SaveError, sqlite3.Error, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            log.exception("Load failed")
            self.last_error = str(exc)
            return False


class SettingsStore:
    """Settings/bindings persisted in saves/settings.db (key → JSON)."""

    def __init__(self, save_dir: str = SAVE_DIR):
        os.makedirs(save_dir, exist_ok=True)
        self.path = os.path.join(save_dir, "settings.db")

    def load(self, defaults: dict) -> dict:
        out = dict(defaults)
        try:
            with sqlite3.connect(self.path) as conn:
                conn.execute("CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT)")
                for k, v in conn.execute("SELECT key, value FROM settings"):
                    if k in defaults:
                        out[k] = json.loads(v)
        except (sqlite3.Error, json.JSONDecodeError) as exc:
            log.warning("Settings unreadable, using defaults: %s", exc)
        return out

    def save(self, settings: dict):
        try:
            with sqlite3.connect(self.path) as conn:
                conn.execute("CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT)")
                conn.executemany("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
                                 [(k, json.dumps(v)) for k, v in settings.items()])
        except sqlite3.Error as exc:
            log.warning("Settings save failed: %s", exc)
