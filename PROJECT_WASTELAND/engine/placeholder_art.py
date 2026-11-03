"""Procedural cartoon-style placeholder art.

Everything here is a *fallback* used when no PNG exists under assets/. The style aims for a
hand-painted wasteland look: mottled fills, dark ink outlines, warm highlights.
"""
import math
import random
import zlib

import pygame

from config import TILE_W, TILE_H, HALF_W, HALF_H, DEPTH_PX

INK = (38, 30, 26)


def _seed(*parts) -> int:
    return zlib.crc32(repr(parts).encode())


def _shade(c, k):
    return tuple(max(0, min(255, int(v * k))) for v in c[:3])


def _mix(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def diamond_points(cx, cy, hw=HALF_W, hh=HALF_H):
    return [(cx, cy - hh), (cx + hw, cy), (cx, cy + hh), (cx - hw, cy)]


# ---------------------------------------------------------------- terrain
def make_tile(tdef: dict, variant: int) -> pygame.Surface:
    rng = random.Random(_seed(tdef["id"], variant))
    w, h = TILE_W, TILE_H
    base = tuple(tdef["color"])
    accent = tuple(tdef["accent"])
    tex = pygame.Surface((w, h), pygame.SRCALPHA)
    tex.fill((*base, 255))
    # Mottled paint blotches.
    for _ in range(4):
        col = _mix(base, accent, rng.uniform(0.25, 0.55))
        r = rng.randint(9, 16)
        pygame.draw.ellipse(tex, (*col, 70), (rng.randint(-8, w - 8), rng.randint(-4, h - 4), r * 2, r))
    for _ in range(5):
        col = _shade(base, rng.uniform(1.05, 1.15))
        pygame.draw.ellipse(tex, (*col, 60), (rng.randint(0, w - 8), rng.randint(0, h - 4), rng.randint(6, 14), rng.randint(3, 6)))
    tid = tdef["id"]
    if tid in ("asphalt", "concrete", "clay", "rock"):
        for _ in range(2 if tid != "rock" else 3):
            x, y = rng.randint(8, w - 8), rng.randint(4, h - 4)
            pts = [(x, y)]
            for _ in range(3):
                x += rng.randint(-8, 8)
                y += rng.randint(-3, 3)
                pts.append((x, y))
            pygame.draw.lines(tex, (*_shade(base, 0.62), 200), False, pts, 1)
    if tid in ("gravel", "rubble", "rock", "sand", "dirt"):
        n = {"gravel": 7, "rubble": 5, "rock": 3, "sand": 2, "dirt": 2}[tid]
        for _ in range(n):
            x, y = rng.randint(4, w - 6), rng.randint(3, h - 4)
            s = rng.randint(1, 3)
            pygame.draw.ellipse(tex, (*_shade(accent, 0.8), 255), (x, y, s * 2 + 1, s + 1))
            pygame.draw.ellipse(tex, (*_shade(base, 1.2), 255), (x, y, s + 1, 1))
    if tid == "asphalt" and variant == 0:
        pygame.draw.line(tex, (200, 180, 90, 160), (w // 2 - 12, h // 2 - 6), (w // 2 + 12, h // 2 + 6), 2)
    if tid == "dry_grass":
        for _ in range(9):
            x, y = rng.randint(6, w - 6), rng.randint(6, h - 2)
            col = _shade(accent, rng.uniform(0.75, 1.1))
            for dx in (-2, 0, 2):
                pygame.draw.line(tex, (*col, 255), (x, y), (x + dx, y - rng.randint(3, 6)), 1)
    if tid == "toxic":
        for _ in range(2):
            x, y = rng.randint(8, w - 20), rng.randint(6, h - 10)
            pygame.draw.ellipse(tex, (150, 210, 60, 150), (x, y, rng.randint(8, 14), rng.randint(4, 6)))
    if tid == "sand":
        for _ in range(3):
            y = rng.randint(6, h - 6)
            x = rng.randint(4, w - 24)
            pygame.draw.arc(tex, (*_shade(base, 1.1), 160), (x, y, 20, 6), 0.3, 2.8, 1)

    mask = pygame.Surface((w, h), pygame.SRCALPHA)
    pygame.draw.polygon(mask, (255, 255, 255, 255), diamond_points(w / 2, h / 2))
    tex.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
    # Faint lower-edge shadow sells the painted-tile depth without a hard grid.
    pygame.draw.line(tex, (*_shade(base, 0.86), 90), (0, h // 2), (w // 2, h - 1), 1)
    pygame.draw.line(tex, (*_shade(base, 0.9), 70), (w // 2, h - 1), (w - 1, h // 2), 1)
    return tex


def make_hole(tdef: dict, depth: int) -> pygame.Surface:
    """Dug-hole overlay drawn on top of the tile (same footprint)."""
    w, h = TILE_W, TILE_H
    s = pygame.Surface((w, h + 4), pygame.SRCALPHA)
    base = tuple(tdef["color"])
    cx, cy = w / 2, h / 2
    mouth_k = 0.62 + 0.08 * depth
    mouth = diamond_points(cx, cy, HALF_W * mouth_k, HALF_H * mouth_k)
    # Spoil ring: loose dirt kicked out of the hole.
    ring = diamond_points(cx, cy, HALF_W * min(0.98, mouth_k + 0.18), HALF_H * min(0.98, mouth_k + 0.18))
    pygame.draw.polygon(s, (*_shade(base, 0.9), 255), ring)
    rng = random.Random(depth * 31 + len(tdef["id"]))
    for _ in range(6 + depth * 2):
        a = rng.uniform(0, math.tau)
        rx = cx + math.cos(a) * HALF_W * (mouth_k + 0.1)
        ry = cy + math.sin(a) * HALF_H * (mouth_k + 0.1)
        pygame.draw.ellipse(s, (*_shade(base, rng.uniform(0.75, 1.1)), 255), (rx - 3, ry - 2, 6, 3))
    wall_dark = _shade(base, 0.30 - depth * 0.04)
    wall_light = _shade(base, 0.50 - depth * 0.06)
    floor = _shade(base, 0.38 - depth * 0.07)
    pygame.draw.polygon(s, (*wall_dark, 255), mouth)
    fk = mouth_k * (0.75 - depth * 0.08)
    fy = cy + depth * DEPTH_PX * 0.55
    floor_pts = diamond_points(cx, fy, HALF_W * fk, HALF_H * fk)
    # Back walls (visible far faces).
    pygame.draw.polygon(s, (*wall_light, 255), [mouth[3], mouth[0], floor_pts[0], floor_pts[3]])
    pygame.draw.polygon(s, (*_shade(wall_light, 0.85), 255), [mouth[0], mouth[1], floor_pts[1], floor_pts[0]])
    pygame.draw.polygon(s, (*floor, 255), floor_pts)
    # Strata lines on the far walls.
    for i in range(1, depth + 1):
        t = i / (depth + 1)
        a = (mouth[3][0] + (floor_pts[3][0] - mouth[3][0]) * t, mouth[3][1] + (floor_pts[3][1] - mouth[3][1]) * t)
        b = (mouth[0][0] + (floor_pts[0][0] - mouth[0][0]) * t, mouth[0][1] + (floor_pts[0][1] - mouth[0][1]) * t)
        c = (mouth[1][0] + (floor_pts[1][0] - mouth[1][0]) * t, mouth[1][1] + (floor_pts[1][1] - mouth[1][1]) * t)
        pygame.draw.lines(s, (*_shade(wall_dark, 0.8), 255), False, [a, b, c], 1)
    # Clip everything below the mouth so the hole stays inside its tile.
    clip = pygame.Surface(s.get_size(), pygame.SRCALPHA)
    pygame.draw.polygon(clip, (255, 255, 255, 255), ring)
    s.blit(clip, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
    pygame.draw.polygon(s, (*INK, 160), mouth, 1)
    return s


def make_cache_clue() -> pygame.Surface:
    """Rust stain + faint glint: surface hint that something is buried below."""
    s = pygame.Surface((TILE_W, TILE_H), pygame.SRCALPHA)
    rng = random.Random(7)
    for _ in range(6):
        x, y = rng.randint(18, 40), rng.randint(9, 20)
        pygame.draw.ellipse(s, (130, 70, 40, 90), (x, y, rng.randint(5, 10), rng.randint(3, 5)))
    pygame.draw.line(s, (90, 60, 40, 220), (28, 16), (36, 14), 2)
    pygame.draw.circle(s, (255, 240, 200, 200), (36, 14), 1)
    return s


def make_target_marker(color=(255, 230, 140)) -> pygame.Surface:
    s = pygame.Surface((TILE_W, TILE_H), pygame.SRCALPHA)
    pts = diamond_points(TILE_W / 2, TILE_H / 2, HALF_W - 2, HALF_H - 1)
    pygame.draw.polygon(s, (*color, 50), pts)
    pygame.draw.polygon(s, (*color, 220), pts, 2)
    return s


# ---------------------------------------------------------------- iso shapes
def iso_box(surf, cx, cy, w, d, h, color, outline=INK):
    """Draw an isometric box. (cx, cy) is ground center in pixels, w/d in world tiles, h in px."""
    def p(x, y, z=0):
        return (cx + (x - y) * HALF_W, cy + (x + y) * HALF_H - z)
    x0, x1, y0, y1 = -w / 2, w / 2, -d / 2, d / 2
    top = [p(x0, y0, h), p(x1, y0, h), p(x1, y1, h), p(x0, y1, h)]
    left = [p(x0, y1, 0), p(x1, y1, 0), p(x1, y1, h), p(x0, y1, h)]
    right = [p(x1, y0, 0), p(x1, y1, 0), p(x1, y1, h), p(x1, y0, h)]
    pygame.draw.polygon(surf, _shade(color, 0.72), left)
    pygame.draw.polygon(surf, _shade(color, 0.55), right)
    pygame.draw.polygon(surf, _shade(color, 1.08), top)
    for poly in (left, right, top):
        pygame.draw.polygon(surf, outline, poly, 2)
    return top, left, right


def _canvas(w, h):
    return pygame.Surface((w, h), pygame.SRCALPHA)


def _shadow(surf, cx, cy, rx, ry, alpha=70):
    sh = _canvas(rx * 2, ry * 2)
    pygame.draw.ellipse(sh, (0, 0, 0, alpha), sh.get_rect())
    surf.blit(sh, (cx - rx, cy - ry))


def _blob(surf, color, rect, outline=True):
    pygame.draw.ellipse(surf, color, rect)
    if outline:
        pygame.draw.ellipse(surf, INK, rect, 2)


def make_object(art: str, variant: int = 0):
    """Return (surface, anchor_x, anchor_y)."""
    rng = random.Random(_seed(art, variant))
    fn = _OBJECT_ART.get(art)
    if fn is None:
        s = _canvas(40, 40)
        _blob(s, (200, 0, 200), (8, 8, 24, 24))
        return s, 20, 30
    return fn(rng)


def _stone_pile(rng):
    s = _canvas(48, 32)
    _shadow(s, 24, 22, 16, 6)
    for i in range(3):
        x, y = 10 + i * 9 + rng.randint(-2, 2), 12 + rng.randint(-2, 3)
        c = _shade((132, 126, 116), rng.uniform(0.85, 1.15))
        _blob(s, c, (x, y, 14, 10))
        pygame.draw.arc(s, _shade(c, 1.35), (x + 2, y + 1, 9, 6), 0.6, 2.4, 2)
    return s, 24, 22


def _branch_pile(rng):
    s = _canvas(52, 30)
    _shadow(s, 26, 20, 18, 5, 55)
    for _ in range(3):
        x0, y0 = rng.randint(6, 18), rng.randint(10, 22)
        x1, y1 = rng.randint(32, 46), rng.randint(8, 22)
        pygame.draw.line(s, INK, (x0, y0), (x1, y1), 5)
        pygame.draw.line(s, (128, 88, 52), (x0, y0), (x1, y1), 3)
        mx, my = (x0 + x1) // 2, (y0 + y1) // 2
        pygame.draw.line(s, (128, 88, 52), (mx, my), (mx + 5, my - 6), 2)
    return s, 26, 20


def _bush(rng):
    s = _canvas(56, 48)
    _shadow(s, 28, 38, 20, 6)
    for _ in range(14):
        a = rng.uniform(math.pi * 1.05, math.pi * 1.95)
        ln = rng.randint(14, 24)
        x0, y0 = 28 + rng.randint(-6, 6), 38
        x1, y1 = x0 + math.cos(a) * ln, y0 + math.sin(a) * ln
        col = _shade((150, 132, 70), rng.uniform(0.8, 1.15))
        pygame.draw.line(s, INK, (x0, y0), (x1, y1), 3)
        pygame.draw.line(s, col, (x0, y0), (x1, y1), 1)
    for _ in range(4):
        x, y = rng.randint(14, 40), rng.randint(14, 32)
        pygame.draw.circle(s, (110, 140, 70), (x, y), 3)
    return s, 28, 38


def _dead_tree(rng):
    s = _canvas(90, 120)
    _shadow(s, 45, 108, 22, 7)
    trunk = (96, 72, 54)
    pygame.draw.polygon(s, INK, [(38, 110), (52, 110), (49, 40), (42, 40)])
    pygame.draw.polygon(s, trunk, [(40, 108), (50, 108), (47, 42), (44, 42)])

    def branch(x, y, ang, ln, width):
        if ln < 6 or width < 1:
            return
        x2, y2 = x + math.cos(ang) * ln, y + math.sin(ang) * ln
        pygame.draw.line(s, INK, (x, y), (x2, y2), width + 2)
        pygame.draw.line(s, trunk, (x, y), (x2, y2), width)
        for _ in range(2):
            branch(x2, y2, ang + rng.uniform(-0.7, 0.7), ln * rng.uniform(0.55, 0.75), width - 1)
    branch(45, 50, -math.pi / 2 - 0.5, 26, 4)
    branch(45, 56, -math.pi / 2 + 0.6, 24, 4)
    branch(45, 44, -math.pi / 2, 22, 4)
    pygame.draw.line(s, _shade(trunk, 1.3), (44, 104), (46, 50), 1)
    return s, 45, 108


def _boulder(rng):
    s = _canvas(72, 60)
    _shadow(s, 36, 48, 28, 8)
    col = (128, 118, 108)
    pts = [(10, 46), (8, 30), (22, 12), (44, 8), (62, 20), (66, 40), (56, 50), (24, 52)]
    pygame.draw.polygon(s, col, pts)
    pygame.draw.polygon(s, _shade(col, 1.25), [(22, 14), (44, 10), (58, 20), (36, 24)])
    pygame.draw.polygon(s, _shade(col, 0.75), [(56, 50), (66, 40), (62, 22), (48, 34)])
    pygame.draw.polygon(s, INK, pts, 3)
    pygame.draw.line(s, _shade(col, 0.6), (30, 30), (40, 40), 2)
    return s, 36, 48


def _scrap_pile(rng):
    s = _canvas(70, 50)
    _shadow(s, 35, 38, 26, 8)
    pygame.draw.ellipse(s, (110, 96, 84), (8, 24, 54, 20))
    for _ in range(8):
        x, y = rng.randint(10, 52), rng.randint(14, 32)
        col = rng.choice([(150, 100, 70), (120, 124, 130), (90, 130, 160), (170, 160, 140)])
        pts = [(x, y), (x + rng.randint(6, 14), y - rng.randint(2, 8)), (x + rng.randint(8, 16), y + rng.randint(2, 6))]
        pygame.draw.polygon(s, col, pts)
        pygame.draw.polygon(s, INK, pts, 2)
    pygame.draw.line(s, INK, (20, 20), (44, 10), 4)
    pygame.draw.line(s, (130, 130, 136), (20, 20), (44, 10), 2)
    return s, 35, 38


def _car(rng, color=(150, 80, 60), police=False):
    s = _canvas(150, 110)
    cx, cy = 75, 72
    _shadow(s, cx, cy + 4, 60, 18)
    iso_box(s, cx, cy, 2.2, 1.1, 18, color)
    top, _, _ = iso_box(s, cx - 4, cy - 18, 1.2, 0.9, 14, _shade(color, 1.05))
    pygame.draw.polygon(s, (60, 80, 90), [top[0], top[1], (top[1][0], top[1][1] + 6), (top[0][0], top[0][1] + 6)])
    for wx, wy in ((-0.7, 0.55), (0.7, 0.55)):
        px, py = cx + (wx - wy) * HALF_W, cy + (wx + wy) * HALF_H
        _blob(s, (30, 30, 32), (px - 8, py - 6, 16, 12))
    if police:
        pygame.draw.rect(s, (40, 70, 200), (cx - 18, cy - 44, 10, 5))
        pygame.draw.rect(s, (220, 50, 50), (cx - 8, cy - 44, 10, 5))
        pygame.draw.rect(s, INK, (cx - 18, cy - 44, 20, 5), 1)
    for _ in range(5):
        x, y = rng.randint(30, 120), rng.randint(40, 80)
        pygame.draw.circle(s, (110, 60, 30, 200), (x, y), rng.randint(2, 4))
    return s, cx, cy


def _military_crate(rng):
    s = _canvas(64, 60)
    _shadow(s, 32, 44, 24, 8)
    top, left, right = iso_box(s, 32, 44, 0.8, 0.6, 20, (100, 112, 70))
    pygame.draw.line(s, (220, 220, 200), left[3], left[1], 2)
    return s, 32, 44


def _skeleton(rng):
    s = _canvas(54, 34)
    bone = (226, 218, 196)
    pygame.draw.circle(s, INK, (14, 16), 7)
    pygame.draw.circle(s, bone, (14, 16), 5)
    pygame.draw.circle(s, INK, (12, 15), 1)
    pygame.draw.line(s, INK, (19, 18), (40, 22), 5)
    pygame.draw.line(s, bone, (19, 18), (40, 22), 3)
    for i in range(3):
        pygame.draw.line(s, bone, (24 + i * 5, 14), (26 + i * 5, 26), 2)
    pygame.draw.line(s, bone, (40, 22), (50, 16), 2)
    pygame.draw.line(s, bone, (40, 22), (50, 28), 2)
    return s, 27, 22


def _dumpster(rng):
    s = _canvas(80, 70)
    _shadow(s, 40, 50, 30, 9)
    iso_box(s, 40, 50, 1.0, 0.7, 26, (60, 110, 70))
    return s, 40, 50


def _puddle(rng):
    s = _canvas(80, 40)
    pygame.draw.ellipse(s, (96, 84, 60, 255), (4, 8, 72, 26))
    pygame.draw.ellipse(s, (80, 110, 120, 255), (10, 11, 60, 20))
    pygame.draw.ellipse(s, (130, 160, 170, 200), (20, 14, 22, 6))
    return s, 40, 21


def _ruin_wall(rng):
    s = _canvas(80, 110)
    cx, cy = 40, 86
    _shadow(s, cx, cy, 34, 10)
    h = rng.randint(34, 62)
    top, left, right = iso_box(s, cx, cy, 1.0, 0.45, h, (150, 132, 116))
    for row in range(4, h, 9):
        pygame.draw.line(s, _shade((150, 132, 116), 0.6), (left[0][0] + 3, left[0][1] - row), (left[1][0] - 2, left[1][1] - row), 1)
    # Jagged broken top edge.
    pygame.draw.polygon(s, (0, 0, 0, 0), [(top[0][0], top[0][1] - 4), (top[0][0] + 14, top[0][1] + 6), (top[0][0] + 6, top[0][1] - 4)])
    return s, cx, cy


def _gas_pump(rng):
    s = _canvas(50, 80)
    _shadow(s, 25, 64, 16, 6)
    iso_box(s, 25, 64, 0.45, 0.35, 42, (190, 60, 50))
    pygame.draw.rect(s, (230, 230, 220), (18, 26, 10, 8))
    pygame.draw.rect(s, INK, (18, 26, 10, 8), 1)
    return s, 25, 64


def _toxic_barrel(rng):
    s = _canvas(40, 54)
    _shadow(s, 20, 44, 14, 5)
    pygame.draw.rect(s, INK, (7, 12, 26, 34), border_radius=5)
    pygame.draw.rect(s, (210, 180, 40), (9, 14, 22, 30), border_radius=4)
    pygame.draw.ellipse(s, (230, 200, 60), (8, 8, 24, 10))
    pygame.draw.ellipse(s, INK, (8, 8, 24, 10), 2)
    pygame.draw.circle(s, INK, (20, 30), 5, 2)
    pygame.draw.ellipse(s, (130, 200, 50), (4, 40, 18, 8))
    return s, 20, 44


def _campfire(rng):
    s = _canvas(56, 40)
    _shadow(s, 28, 26, 20, 6, 50)
    for i in range(8):
        a = i / 8 * math.tau
        x, y = 28 + math.cos(a) * 15, 24 + math.sin(a) * 7
        _blob(s, _shade((130, 124, 116), rng.uniform(0.85, 1.15)), (x - 5, y - 4, 10, 8))
    pygame.draw.line(s, INK, (18, 26), (38, 20), 5)
    pygame.draw.line(s, (110, 76, 46), (18, 26), (38, 20), 3)
    pygame.draw.line(s, INK, (20, 20), (38, 28), 5)
    pygame.draw.line(s, (110, 76, 46), (20, 20), (38, 28), 3)
    return s, 28, 26


def _bedroll(rng):
    s = _canvas(80, 44)
    pts = [(10, 22), (40, 6), (70, 20), (40, 38)]
    pygame.draw.polygon(s, (110, 90, 140), pts)
    pygame.draw.polygon(s, INK, pts, 2)
    pygame.draw.polygon(s, (150, 130, 180), [(10, 22), (22, 16), (52, 30), (40, 38)])
    pygame.draw.ellipse(s, (210, 200, 180), (14, 14, 18, 9))
    pygame.draw.ellipse(s, INK, (14, 14, 18, 9), 1)
    return s, 40, 24


def _storage_box(rng):
    s = _canvas(64, 60)
    _shadow(s, 32, 44, 24, 8)
    top, left, right = iso_box(s, 32, 44, 0.75, 0.55, 22, (150, 110, 70))
    pygame.draw.line(s, _shade((150, 110, 70), 0.5), left[3], left[2], 3)
    return s, 32, 44


def _workbench(rng):
    s = _canvas(96, 72)
    cx, cy = 48, 54
    _shadow(s, cx, cy, 36, 10)
    for lx, ly in ((-0.5, -0.25), (0.5, -0.25), (-0.5, 0.25), (0.5, 0.25)):
        px, py = cx + (lx - ly) * HALF_W, cy + (lx + ly) * HALF_H
        pygame.draw.line(s, INK, (px, py), (px, py - 22), 5)
        pygame.draw.line(s, (100, 72, 46), (px, py), (px, py - 22), 3)
    iso_box(s, cx, cy - 22, 1.2, 0.65, 6, (140, 100, 62))
    pygame.draw.line(s, (150, 150, 160), (cx - 6, cy - 32), (cx + 10, cy - 38), 3)
    pygame.draw.rect(s, (90, 90, 96), (cx + 10, cy - 40, 8, 5))
    return s, cx, cy


def _water_purifier(rng):
    s = _canvas(56, 80)
    _shadow(s, 28, 64, 18, 6)
    pygame.draw.rect(s, INK, (14, 20, 28, 44), border_radius=6)
    pygame.draw.rect(s, (90, 140, 190), (16, 22, 24, 40), border_radius=5)
    pygame.draw.rect(s, (160, 200, 230), (20, 28, 6, 24), border_radius=3)
    pygame.draw.rect(s, (120, 120, 126), (12, 14, 32, 8))
    pygame.draw.rect(s, INK, (12, 14, 32, 8), 2)
    pygame.draw.line(s, INK, (42, 54), (50, 60), 3)
    return s, 28, 64


def _barricade(rng):
    s = _canvas(90, 70)
    cx, cy = 45, 52
    _shadow(s, cx, cy, 34, 8)
    for i in range(5):
        x = cx - 30 + i * 14
        y = cy - 6 + (i - 2) * 2
        pygame.draw.line(s, INK, (x, y), (x + rng.randint(-3, 3), y - 34), 8)
        pygame.draw.line(s, (130, 92, 58), (x, y), (x + rng.randint(-3, 3), y - 34), 6)
    pygame.draw.line(s, INK, (cx - 34, cy - 16), (cx + 34, cy - 26), 7)
    pygame.draw.line(s, (110, 80, 50), (cx - 34, cy - 16), (cx + 34, cy - 26), 5)
    return s, cx, cy


_OBJECT_ART = {
    "stone_pile": _stone_pile, "branch_pile": _branch_pile, "bush": _bush, "dead_tree": _dead_tree,
    "boulder": _boulder, "scrap_pile": _scrap_pile, "car_wreck": lambda r: _car(r, r.choice([(150, 80, 60), (90, 110, 140), (170, 150, 90)])),
    "police_car": lambda r: _car(r, (220, 220, 214), police=True), "military_crate": _military_crate,
    "skeleton": _skeleton, "dumpster": _dumpster, "puddle": _puddle, "ruin_wall": _ruin_wall, "gas_pump": _gas_pump,
    "toxic_barrel": _toxic_barrel, "campfire": _campfire, "bedroll": _bedroll, "storage_box": _storage_box,
    "workbench": _workbench, "water_purifier": _water_purifier, "barricade": _barricade,
}

FLAT_ART = {"puddle", "bedroll", "skeleton"}


# ---------------------------------------------------------------- item icons
def make_icon(icon: dict, w_cells: int, h_cells: int, cell: int) -> pygame.Surface:
    W, H = w_cells * cell, h_cells * cell
    s = _canvas(W, H)
    col = tuple(icon.get("color", (200, 200, 200)))
    shape = icon.get("shape", "box")
    cx, cy = W / 2, H / 2
    m = min(W, H)
    hi = _shade(col, 1.35)
    lo = _shade(col, 0.7)

    def line(a, b, width, c):
        pygame.draw.line(s, INK, a, b, width + 3)
        pygame.draw.line(s, c, a, b, width)

    if shape in ("stone", "scrap"):
        r = m * 0.3
        pts = [(cx - r, cy + r * 0.4), (cx - r * 0.6, cy - r * 0.6), (cx + r * 0.3, cy - r * 0.8), (cx + r, cy - r * 0.1), (cx + r * 0.6, cy + r * 0.7), (cx - r * 0.3, cy + r * 0.8)]
        if shape == "scrap":
            pts = pts[::2] + [(cx + r * 0.9, cy + r * 0.9)]
        pygame.draw.polygon(s, col, pts)
        pygame.draw.polygon(s, INK, pts, 2)
        pygame.draw.line(s, hi, pts[1], pts[2], 2)
    elif shape in ("stick", "pipe"):
        a, b = (W * 0.25, H * 0.85), (W * 0.75, H * 0.15)
        line(a, b, 5 if shape == "stick" else 6, col)
        if shape == "stick":
            line((cx, cy), (cx + m * 0.18, cy + m * 0.02), 2, col)
    elif shape == "fiber":
        for i in range(4):
            x = cx - m * 0.2 + i * m * 0.13
            pygame.draw.arc(s, col, (x - 6, cy - m * 0.3, 14, m * 0.6), -1.2, 1.2, 3)
    elif shape == "log":
        r = pygame.Rect(0, 0, m * 0.7, m * 0.38)
        r.center = (cx, cy)
        pygame.draw.rect(s, col, r, border_radius=8)
        pygame.draw.rect(s, INK, r, 2, border_radius=8)
        pygame.draw.ellipse(s, hi, (r.right - 12, r.top + 2, 10, r.height - 4))
        pygame.draw.ellipse(s, INK, (r.right - 12, r.top + 2, 10, r.height - 4), 1)
    elif shape == "leaf":
        pygame.draw.ellipse(s, col, (cx - m * 0.3, cy - m * 0.18, m * 0.6, m * 0.36))
        pygame.draw.ellipse(s, INK, (cx - m * 0.3, cy - m * 0.18, m * 0.6, m * 0.36), 2)
        pygame.draw.line(s, lo, (cx - m * 0.3, cy), (cx + m * 0.3, cy), 2)
    elif shape in ("plate", "box", "chip", "battery"):
        r = pygame.Rect(0, 0, m * 0.62, m * 0.5 if shape != "battery" else m * 0.36)
        r.center = (cx, cy)
        pygame.draw.rect(s, col, r, border_radius=4)
        pygame.draw.rect(s, INK, r, 2, border_radius=4)
        if shape == "chip":
            for i in range(4):
                pygame.draw.line(s, (200, 200, 120), (r.left + 4 + i * 6, r.top), (r.left + 4 + i * 6, r.top - 4), 2)
        if shape == "battery":
            pygame.draw.rect(s, (200, 170, 60), (r.right, cy - 3, 4, 6))
            pygame.draw.rect(s, (200, 170, 60), (r.left + 3, r.top + 3, r.width * 0.4, r.height - 6))
        if shape == "box":
            pygame.draw.line(s, lo, (r.left, cy), (r.right, cy), 2)
    elif shape == "coil":
        for i in range(3):
            pygame.draw.circle(s, col, (cx - 6 + i * 6, cy), m * 0.2, 3)
        pygame.draw.circle(s, INK, (cx, cy), m * 0.28, 1)
    elif shape == "shard":
        pts = [(cx - m * 0.22, cy + m * 0.2), (cx - m * 0.05, cy - m * 0.3), (cx + m * 0.25, cy + m * 0.05), (cx + m * 0.05, cy + m * 0.28)]
        pygame.draw.polygon(s, col, pts)
        pygame.draw.polygon(s, INK, pts, 2)
        pygame.draw.line(s, (255, 255, 255), pts[1], (cx, cy), 1)
    elif shape == "cloth":
        pts = [(cx - m * 0.3, cy - m * 0.2), (cx + m * 0.28, cy - m * 0.25), (cx + m * 0.3, cy + m * 0.22), (cx - m * 0.25, cy + m * 0.25)]
        pygame.draw.polygon(s, col, pts)
        pygame.draw.polygon(s, INK, pts, 2)
        pygame.draw.line(s, lo, (cx - m * 0.2, cy), (cx + m * 0.2, cy - m * 0.05), 2)
    elif shape == "bottle":
        r = pygame.Rect(0, 0, m * 0.32, m * 0.55)
        r.center = (cx, cy + m * 0.06)
        pygame.draw.rect(s, col, r, border_radius=6)
        pygame.draw.rect(s, INK, r, 2, border_radius=6)
        pygame.draw.rect(s, (60, 60, 70), (cx - 4, r.top - 7, 8, 8))
        pygame.draw.rect(s, (255, 255, 255), (r.left + 3, r.top + 5, 3, r.height * 0.5))
    elif shape == "can":
        r = pygame.Rect(0, 0, m * 0.44, m * 0.5)
        r.center = (cx, cy)
        pygame.draw.rect(s, (190, 190, 196), r, border_radius=3)
        pygame.draw.rect(s, col, (r.left, r.top + 6, r.width, r.height - 12))
        pygame.draw.rect(s, INK, r, 2, border_radius=3)
    elif shape == "meat":
        pygame.draw.ellipse(s, col, (cx - m * 0.28, cy - m * 0.2, m * 0.5, m * 0.4))
        pygame.draw.ellipse(s, INK, (cx - m * 0.28, cy - m * 0.2, m * 0.5, m * 0.4), 2)
        line((cx + m * 0.15, cy + m * 0.05), (cx + m * 0.32, cy + m * 0.22), 3, (240, 230, 210))
    elif shape == "bar":
        r = pygame.Rect(0, 0, m * 0.7, m * 0.3)
        r.center = (cx, cy)
        pygame.draw.rect(s, col, r, border_radius=3)
        pygame.draw.rect(s, INK, r, 2, border_radius=3)
        pygame.draw.rect(s, hi, (r.left + 6, r.top + 4, r.width * 0.4, 4))
    elif shape == "pill":
        for i, c in enumerate(((230, 80, 80), (240, 240, 250))):
            pygame.draw.circle(s, c, (cx - 5 + i * 10, cy), m * 0.14)
            pygame.draw.circle(s, INK, (cx - 5 + i * 10, cy), m * 0.14, 2)
    elif shape == "gear":
        r = m * 0.25
        for i in range(8):
            a = i / 8 * math.tau
            pygame.draw.circle(s, INK, (cx + math.cos(a) * r, cy + math.sin(a) * r), 5)
        pygame.draw.circle(s, col, (cx, cy), r)
        pygame.draw.circle(s, INK, (cx, cy), r, 2)
        pygame.draw.circle(s, (40, 40, 40), (cx, cy), r * 0.35)
    elif shape in ("bullet", "shell"):
        for i in range(3):
            x = cx - 8 + i * 8
            body = col if shape == "bullet" else (190, 50, 40)
            pygame.draw.rect(s, body, (x - 3, cy - 4, 6, 14))
            pygame.draw.rect(s, INK, (x - 3, cy - 4, 6, 14), 1)
            tip = (200, 130, 70) if shape == "bullet" else (210, 180, 80)
            pygame.draw.ellipse(s, tip, (x - 3, cy - 10, 6, 10))
            pygame.draw.ellipse(s, INK, (x - 3, cy - 10, 6, 10), 1)
    elif shape == "axe":
        line((W * 0.5, H * 0.9), (W * 0.5, H * 0.12), 4, (120, 86, 52))
        head = [(W * 0.5, H * 0.12), (W * 0.9, H * 0.08), (W * 0.95, H * 0.32), (W * 0.5, H * 0.28)]
        pygame.draw.polygon(s, col, head)
        pygame.draw.polygon(s, INK, head, 2)
    elif shape == "shovel":
        line((W * 0.5, H * 0.08), (W * 0.5, H * 0.65), 4, (120, 86, 52))
        blade = [(W * 0.25, H * 0.65), (W * 0.75, H * 0.65), (W * 0.68, H * 0.88), (W * 0.5, H * 0.95), (W * 0.32, H * 0.88)]
        pygame.draw.polygon(s, col, blade)
        pygame.draw.polygon(s, INK, blade, 2)
    elif shape == "hammer":
        line((W * 0.5, H * 0.92), (W * 0.5, H * 0.2), 4, (120, 86, 52))
        r = pygame.Rect(W * 0.18, H * 0.06, W * 0.64, H * 0.2)
        pygame.draw.rect(s, col, r)
        pygame.draw.rect(s, INK, r, 2)
    elif shape == "knife":
        line((W * 0.2, H * 0.8), (W * 0.42, H * 0.58), 4, (60, 50, 44))
        blade = [(W * 0.42, H * 0.58), (W * 0.85, H * 0.15), (W * 0.52, H * 0.66)]
        pygame.draw.polygon(s, col, blade)
        pygame.draw.polygon(s, INK, blade, 2)
    elif shape in ("pistol", "rifle"):
        if shape == "pistol":
            body = [(W * 0.1, H * 0.3), (W * 0.85, H * 0.3), (W * 0.85, H * 0.5), (W * 0.45, H * 0.5), (W * 0.38, H * 0.85), (W * 0.22, H * 0.85), (W * 0.26, H * 0.5), (W * 0.1, H * 0.5)]
        else:
            body = [(W * 0.03, H * 0.4), (W * 0.95, H * 0.4), (W * 0.95, H * 0.52), (W * 0.4, H * 0.55), (W * 0.32, H * 0.75), (W * 0.05, H * 0.75)]
        pygame.draw.polygon(s, col, body)
        pygame.draw.polygon(s, INK, body, 2)
        pygame.draw.line(s, hi, body[0], body[1], 2)
    elif shape == "shirt":
        pts = [(cx - m * 0.35, cy - m * 0.25), (cx - m * 0.12, cy - m * 0.35), (cx + m * 0.12, cy - m * 0.35), (cx + m * 0.35, cy - m * 0.25), (cx + m * 0.28, cy - m * 0.08), (cx + m * 0.2, cy - m * 0.12), (cx + m * 0.2, cy + m * 0.38), (cx - m * 0.2, cy + m * 0.38), (cx - m * 0.2, cy - m * 0.12), (cx - m * 0.28, cy - m * 0.08)]
        pygame.draw.polygon(s, col, pts)
        pygame.draw.polygon(s, INK, pts, 2)
    elif shape == "fire":
        for i in range(3):
            pygame.draw.line(s, INK, (cx - 12, cy + 10 - i), (cx + 12, cy + 4 + i * 3), 5)
        pygame.draw.polygon(s, (240, 150, 40), [(cx - 9, cy + 6), (cx, cy - m * 0.35), (cx + 9, cy + 6)])
        pygame.draw.polygon(s, (255, 230, 120), [(cx - 4, cy + 6), (cx, cy - m * 0.12), (cx + 4, cy + 6)])
    elif shape in ("book", "tag"):
        r = pygame.Rect(0, 0, m * 0.5, m * 0.6 if shape == "book" else m * 0.3)
        r.center = (cx, cy)
        pygame.draw.rect(s, col, r, border_radius=3)
        pygame.draw.rect(s, INK, r, 2, border_radius=3)
        if shape == "book":
            pygame.draw.line(s, lo, (r.left + 5, r.top), (r.left + 5, r.bottom), 2)
    else:
        pygame.draw.circle(s, col, (cx, cy), m * 0.3)
        pygame.draw.circle(s, INK, (cx, cy), m * 0.3, 2)
    return s


def make_light(radius: int, color=(255, 200, 140)) -> pygame.Surface:
    """Radial gradient for additive light maps."""
    s = pygame.Surface((radius * 2, radius * 2))
    s.fill((0, 0, 0))
    steps = 24
    for i in range(steps, 0, -1):
        t = i / steps
        k = (1 - t) ** 1.6
        c = tuple(int(v * k) for v in color)
        pygame.draw.ellipse(s, c, (radius - radius * t, radius - radius * t * 0.6, radius * 2 * t, radius * 1.2 * t))
    return s
