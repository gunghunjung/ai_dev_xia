import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame  # noqa: E402
import pytest  # noqa: E402

from config import DEFAULT_SETTINGS  # noqa: E402


@pytest.fixture(scope="session")
def data():
    pygame.init()
    from engine.asset_manager import AssetManager
    from game.session import GameData
    return GameData(AssetManager())


@pytest.fixture
def audio():
    from engine.audio_manager import AudioManager
    return AudioManager(dict(DEFAULT_SETTINGS))


@pytest.fixture
def session(data, audio):
    from game.session import GameSession
    s = GameSession(data, dict(DEFAULT_SETTINGS), audio, seed=4242)
    s.enemies.clear()
    s.spawner.timer = 1e9
    return s
