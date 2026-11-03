"""Isometric projection and player-tracking camera with trauma-based shake."""
import math
import random

from config import HALF_W, HALF_H, SCREEN_W, SCREEN_H


def iso(x: float, y: float) -> tuple[float, float]:
    """World (tile units) → unscrolled screen pixels."""
    return (x - y) * HALF_W, (x + y) * HALF_H


def inv_iso(sx: float, sy: float) -> tuple[float, float]:
    """Unscrolled screen pixels → world (ground plane)."""
    a = sx / HALF_W
    b = sy / HALF_H
    return (a + b) * 0.5, (b - a) * 0.5


def screen_dir_to_world(dx: float, dy: float) -> tuple[float, float]:
    """Convert a screen-space direction (e.g. WASD) to a normalized world direction."""
    if dx == 0 and dy == 0:
        return 0.0, 0.0
    wx, wy = inv_iso(dx * HALF_W, dy * HALF_W)
    length = math.hypot(wx, wy)
    return wx / length, wy / length


class Camera:
    def __init__(self, width: int = SCREEN_W, height: int = SCREEN_H):
        self.width = width
        self.height = height
        self.x = 0.0  # world focus
        self.y = 0.0
        self.trauma = 0.0
        self.shake_scale = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.lead_x = 0.0
        self.lead_y = 0.0

    def resize(self, width: int, height: int):
        self.width, self.height = width, height

    def snap(self, x: float, y: float):
        self.x, self.y = x, y

    def follow(self, tx: float, ty: float, dt: float, lead=(0.0, 0.0)):
        # Look slightly toward the aim direction so the player sees where they shoot.
        self.lead_x += (lead[0] - self.lead_x) * min(1.0, dt * 4)
        self.lead_y += (lead[1] - self.lead_y) * min(1.0, dt * 4)
        gx, gy = tx + self.lead_x, ty + self.lead_y
        k = min(1.0, dt * 8.0)
        self.x += (gx - self.x) * k
        self.y += (gy - self.y) * k

    def add_trauma(self, amount: float):
        self.trauma = min(1.0, self.trauma + amount * self.shake_scale)

    def update_shake(self, dt: float):
        if self.trauma > 0:
            s = self.trauma * self.trauma
            self.offset_x = random.uniform(-1, 1) * 14 * s
            self.offset_y = random.uniform(-1, 1) * 10 * s
            self.trauma = max(0.0, self.trauma - dt * 1.6)
        else:
            self.offset_x = self.offset_y = 0.0

    def _origin(self):
        cx, cy = iso(self.x, self.y)
        return cx - self.width / 2 - self.offset_x, cy - self.height / 2 - self.offset_y

    def world_to_screen(self, x: float, y: float, z: float = 0.0) -> tuple[float, float]:
        sx, sy = iso(x, y)
        ox, oy = self._origin()
        return sx - ox, sy - oy - z

    def screen_to_world(self, sx: float, sy: float) -> tuple[float, float]:
        ox, oy = self._origin()
        return inv_iso(sx + ox, sy + oy)

    def visible_tile_bounds(self, margin: int = 2):
        """Return (min_x, min_y, max_x, max_y) of tiles that may intersect the screen."""
        corners = [self.screen_to_world(0, 0), self.screen_to_world(self.width, 0),
                   self.screen_to_world(0, self.height), self.screen_to_world(self.width, self.height)]
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        return (int(math.floor(min(xs))) - margin, int(math.floor(min(ys))) - margin,
                int(math.ceil(max(xs))) + margin, int(math.ceil(max(ys))) + margin)

    def on_screen(self, sx: float, sy: float, margin: float = 96) -> bool:
        return -margin <= sx <= self.width + margin and -margin <= sy <= self.height + margin
