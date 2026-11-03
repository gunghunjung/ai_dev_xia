"""Global constants and default settings for PROJECT WASTELAND."""
import os
import sys

GAME_TITLE = "PROJECT WASTELAND"
GAME_VERSION = "0.1.0"
SAVE_VERSION = 1


def _base_dir() -> str:
    # PyInstaller unpacks bundled data next to sys._MEIPASS; saves must live beside the exe.
    if getattr(sys, "frozen", False):
        return getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def _user_dir() -> str:
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


BASE_DIR = _base_dir()
DATA_DIR = os.path.join(BASE_DIR, "data")
ASSET_DIR = os.path.join(BASE_DIR, "assets")
SAVE_DIR = os.path.join(_user_dir(), "saves")

SCREEN_W = 1280
SCREEN_H = 720
FIXED_DT = 1.0 / 60.0
MAX_FRAME_TIME = 0.25

# Isometric tile footprint in pixels (2:1 diamond).
TILE_W = 64
TILE_H = 32
HALF_W = TILE_W // 2
HALF_H = TILE_H // 2
DEPTH_PX = 7  # screen pixels per dug layer

WORLD_SIZE = 160
CHUNK_SIZE = 16
MAX_DIG_DEPTH = 3

# Gameplay timing (real seconds).
DAY_LENGTH = 960.0          # full day/night cycle
START_TIME_OF_DAY = 0.30    # 0=midnight, 0.25=dawn, 0.5=noon
AUTOSAVE_INTERVAL = 300.0

PLAYER_INV_W = 8
PLAYER_INV_H = 6
PLAYER_CARRY_CAPACITY = 28.0
HOTBAR_SIZE = 5
INTERACT_RANGE = 1.6
DIG_REACH = 1.9

DEFAULT_SETTINGS = {
    "master_volume": 0.8,
    "sfx_volume": 0.9,
    "screen_shake": 1.0,
    "hitstop": 1.0,
    "vibration": 0.8,
    "aim_assist": 0.5,
    "aim_assist_enabled": True,
    "stick_deadzone": 0.22,
    "aim_sensitivity": 1.0,
    "show_fps": False,
    "fullscreen": False,
    "damage_numbers": True,
    "bindings": None,  # None → defaults from data/input_profiles.json
}
