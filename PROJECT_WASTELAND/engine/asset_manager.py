"""Central asset cache.

Every image is requested by a key such as ``"terrain/sand_0"``. If ``assets/terrain/sand_0.png``
exists it is loaded (once) and used; otherwise a procedural placeholder generator supplies the
surface. Replacing placeholder art is therefore a matter of dropping PNG files into ``assets/``.
"""
import json
import logging
import os

import pygame

from config import ASSET_DIR, DATA_DIR

log = logging.getLogger(__name__)

_FONT_CANDIDATES = [
    os.path.join(ASSET_DIR, "fonts", "main.ttf"),
    "C:/Windows/Fonts/malgunbd.ttf",
    "C:/Windows/Fonts/malgun.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
]


class AssetManager:
    def __init__(self):
        self._images: dict[str, pygame.Surface] = {}
        self._anchored: dict[str, tuple] = {}
        self._fonts: dict[tuple, pygame.font.Font] = {}
        self._json: dict[str, object] = {}
        self._font_path = next((p for p in _FONT_CANDIDATES if os.path.exists(p)), None)
        self.missing_files: set[str] = set()

    # ---------------------------------------------------------------- data
    def load_json(self, name: str):
        if name not in self._json:
            path = os.path.join(DATA_DIR, name)
            try:
                with open(path, encoding="utf-8") as fh:
                    self._json[name] = json.load(fh)
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"데이터 파일을 읽을 수 없습니다: {path} ({exc})") from exc
        return self._json[name]

    # ---------------------------------------------------------------- images
    def _try_file(self, key: str):
        path = os.path.join(ASSET_DIR, key + ".png")
        if not os.path.exists(path):
            return None
        try:
            surf = pygame.image.load(path)
            return surf.convert_alpha() if pygame.display.get_surface() else surf
        except pygame.error as exc:
            log.warning("Failed to load %s: %s — using placeholder", path, exc)
            self.missing_files.add(path)
            return None

    def image(self, key: str, generator=None) -> pygame.Surface:
        surf = self._images.get(key)
        if surf is None:
            surf = self._try_file(key)
            if surf is None:
                if generator is None:
                    surf = pygame.Surface((16, 16), pygame.SRCALPHA)
                    surf.fill((255, 0, 255, 255))
                else:
                    surf = generator()
            self._images[key] = surf
        return surf

    def anchored(self, key: str, generator):
        """Image with a ground anchor point: returns (surface, anchor_x, anchor_y).

        Generators return the same tuple. File overrides use the bottom-center as anchor
        unless a ``<key>.json`` sidecar provides ``{"anchor": [x, y]}``.
        """
        item = self._anchored.get(key)
        if item is None:
            surf = self._try_file(key)
            if surf is not None:
                ax, ay = surf.get_width() // 2, surf.get_height() - 8
                side = os.path.join(ASSET_DIR, key + ".json")
                if os.path.exists(side):
                    try:
                        with open(side, encoding="utf-8") as fh:
                            ax, ay = json.load(fh)["anchor"]
                    except (OSError, KeyError, ValueError):
                        pass
                item = (surf, ax, ay)
            else:
                item = generator()
            self._anchored[key] = item
        return item

    def cache_size(self) -> int:
        return len(self._images) + len(self._anchored)

    # ---------------------------------------------------------------- fonts
    def font(self, size: int, bold: bool = False) -> pygame.font.Font:
        key = (size, bold)
        f = self._fonts.get(key)
        if f is None:
            try:
                if self._font_path:
                    f = pygame.font.Font(self._font_path, size)
                    f.set_bold(bold and "bd" not in self._font_path.lower() and "bold" not in self._font_path.lower())
                else:
                    f = pygame.font.SysFont("malgungothic,applesdgothicneo,nanumgothic,arial", size, bold=bold)
            except Exception as exc:
                log.warning("Font load failed (%s); using default font", exc)
                f = pygame.font.Font(None, size + 6)
            self._fonts[key] = f
        return f
