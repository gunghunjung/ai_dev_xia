"""Runtime-drawn cartoon characters (placeholder until 8-direction sprite sheets exist).

Characters are drawn into a small canvas each frame so that facing, walk cycle, weapon
swing, hit flash and death fall all come from game state. The renderer only needs the
feet position on screen.
"""
import math

import pygame

INK = (38, 30, 26)
CANVAS_W, CANVAS_H = 96, 96
FEET_X, FEET_Y = 48, 80

# weapon look: (length, width, handle color, head color, head kind)
WEAPON_LOOK = {
    None: None,
    "fists": None,
    "branch": (22, 3, (120, 84, 50), None, None),
    "stick": (28, 4, (140, 100, 60), None, None),
    "stone_axe": (24, 3, (120, 86, 52), (130, 124, 114), "axe"),
    "axe": (26, 3, (120, 86, 52), (180, 186, 196), "axe"),
    "crude_shovel": (28, 3, (120, 86, 52), (130, 124, 114), "blade"),
    "shovel": (30, 3, (120, 86, 52), (160, 166, 176), "blade"),
    "pipe": (30, 4, (124, 128, 136), None, None),
    "knife": (14, 3, (60, 50, 44), (200, 206, 216), "knife"),
    "sledgehammer": (30, 4, (120, 86, 52), (90, 94, 100), "hammer"),
    "pistol": (14, 5, (60, 62, 70), None, "gun"),
    "revolver": (16, 5, (90, 80, 70), None, "gun"),
    "shotgun": (30, 5, (110, 70, 50), None, "gun"),
    "rifle": (34, 5, (70, 80, 60), None, "gun"),
}


def _outline_circle(s, color, center, r):
    pygame.draw.circle(s, INK, center, r + 2)
    pygame.draw.circle(s, color, center, r)


def _shade(c, k):
    return tuple(max(0, min(255, int(v * k))) for v in c[:3])


def _draw_weapon(s, hand, angle, item_id, extend=0.0):
    look = WEAPON_LOOK.get(item_id)
    if not look:
        return
    length, width, handle, head, kind = look
    length += extend
    ca, sa = math.cos(angle), math.sin(angle)
    hx, hy = hand
    if kind == "gun":
        tip = (hx + ca * length, hy + sa * length * 0.8)
        pygame.draw.line(s, INK, hand, tip, width + 4)
        pygame.draw.line(s, handle, hand, tip, width)
        pygame.draw.circle(s, INK, (int(hx - sa * 3), int(hy + ca * 3)), 3)
        return
    tip = (hx + ca * length, hy + sa * length * 0.8)
    pygame.draw.line(s, INK, hand, tip, width + 3)
    pygame.draw.line(s, handle, hand, tip, width)
    if kind == "axe" and head:
        px, py = -sa, ca
        pts = [tip, (tip[0] + px * 9 - ca * 2, tip[1] + py * 9 - sa * 2), (tip[0] + px * 8 - ca * 9, tip[1] + py * 8 - sa * 9),
               (tip[0] - ca * 6, tip[1] - sa * 6)]
        pygame.draw.polygon(s, head, pts)
        pygame.draw.polygon(s, INK, pts, 2)
    elif kind == "blade" and head:
        px, py = -sa, ca
        pts = [(tip[0] + px * 6, tip[1] + py * 6), (tip[0] - px * 6, tip[1] - py * 6),
               (tip[0] - px * 4 + ca * 10, tip[1] - py * 4 + sa * 10), (tip[0] + px * 4 + ca * 10, tip[1] + py * 4 + sa * 10)]
        pygame.draw.polygon(s, head, pts)
        pygame.draw.polygon(s, INK, pts, 2)
    elif kind == "knife" and head:
        end = (tip[0] + ca * 10, tip[1] + sa * 8)
        pygame.draw.line(s, INK, tip, end, 6)
        pygame.draw.line(s, head, tip, end, 3)
    elif kind == "hammer" and head:
        px, py = -sa, ca
        pts = [(tip[0] + px * 9 - ca * 4, tip[1] + py * 9 - sa * 4), (tip[0] + px * 9 + ca * 6, tip[1] + py * 9 + sa * 6),
               (tip[0] - px * 9 + ca * 6, tip[1] - py * 9 + sa * 6), (tip[0] - px * 9 - ca * 4, tip[1] - py * 9 - sa * 4)]
        pygame.draw.polygon(s, head, pts)
        pygame.draw.polygon(s, INK, pts, 2)


def swing_offset(style: str, t: float, arc_deg: float) -> tuple[float, float]:
    """Return (angle offset in radians, extension px) for an attack animation at progress t∈[0,1]."""
    arc = math.radians(arc_deg)
    if style == "stab" or style == "jab":
        ext = math.sin(min(1.0, t) * math.pi) * (14 if style == "stab" else 10)
        return 0.0, ext
    # Anticipation → fast swing → follow-through.
    if t < 0.25:
        k = -0.5 - 0.15 * (t / 0.25)
    elif t < 0.6:
        u = (t - 0.25) / 0.35
        k = -0.65 + 1.3 * (1 - (1 - u) ** 3)
    else:
        k = 0.65 - 0.25 * ((t - 0.6) / 0.4)
    if style == "smash" or style == "chop":
        k *= 1.15
    return k * arc, 0.0


def draw_humanoid(look: dict, aim_angle: float, walk_phase: float, moving: bool, weapon_id,
                  attack_t=None, swing_style="swing", arc=90, flash=0.0, recoil=0.0) -> pygame.Surface:
    """aim_angle is a *screen-space* angle in radians."""
    s = pygame.Surface((CANVAS_W, CANVAS_H), pygame.SRCALPHA)
    body = tuple(look.get("body", (130, 120, 100)))
    skin = tuple(look.get("skin", (220, 180, 150)))
    accent = tuple(look.get("accent", (90, 70, 50)))
    height = float(look.get("height", 1.0))
    hunch = float(look.get("hunch", 0.0))
    ca, sa = math.cos(aim_angle), math.sin(aim_angle)
    facing_back = sa < -0.35
    bob = abs(math.sin(walk_phase)) * 2 if moving else 0
    fx, fy = FEET_X, FEET_Y
    leg = math.sin(walk_phase) * 5 if moving else 0
    th = int(22 * height)
    # Legs
    for side, off in ((-1, leg), (1, -leg)):
        x = fx + side * 5
        pygame.draw.line(s, INK, (x, fy - 14), (x + off * 0.6, fy - 1), 7)
        pygame.draw.line(s, _shade(body, 0.6), (x, fy - 14), (x + off * 0.6, fy - 1), 4)
        pygame.draw.ellipse(s, INK, (x + off * 0.6 - 4, fy - 4, 9, 5))
    torso_top = fy - 14 - th - bob
    lean = hunch * 7
    torso = pygame.Rect(0, 0, 22, th + 4)
    torso.midbottom = (fx + ca * lean, fy - 11 - bob)
    head_c = (int(fx + ca * (lean + 2) * 1.2), int(torso_top - 6 + hunch * 4))
    shoulder = (fx + ca * lean + ca * 4, torso_top + 8)
    angle = aim_angle
    extend = 0.0
    if attack_t is not None:
        off, extend = swing_offset(swing_style, attack_t, arc)
        angle += off
    angle -= recoil
    hand = (shoulder[0] + math.cos(angle) * 11, shoulder[1] + math.sin(angle) * 8)

    def arm_and_weapon():
        pygame.draw.line(s, INK, shoulder, hand, 7)
        pygame.draw.line(s, body, shoulder, hand, 4)
        _draw_weapon(s, hand, angle, weapon_id, extend)
        _outline_circle(s, skin, (int(hand[0]), int(hand[1])), 3)

    if facing_back:
        arm_and_weapon()
    pygame.draw.rect(s, INK, torso.inflate(4, 4), border_radius=9)
    pygame.draw.rect(s, body, torso, border_radius=8)
    pygame.draw.line(s, _shade(body, 1.25), (torso.left + 5, torso.top + 4), (torso.left + 5, torso.bottom - 6), 2)
    pygame.draw.line(s, accent, (torso.left + 2, torso.bottom - 8), (torso.right - 2, torso.bottom - 8), 3)
    if facing_back:
        # Backpack strap / bedroll when seen from behind.
        pygame.draw.rect(s, INK, (torso.left + 3, torso.top + 4, 16, 12), border_radius=3)
        pygame.draw.rect(s, _shade(accent, 1.2), (torso.left + 4, torso.top + 5, 14, 10), border_radius=3)
    _outline_circle(s, skin, head_c, 9)
    if facing_back:
        pygame.draw.circle(s, _shade(accent, 0.8), head_c, 8)
    else:
        ex = ca * 4
        for d in (-3, 3):
            pygame.draw.circle(s, INK, (int(head_c[0] + ex + d), int(head_c[1] + 1)), 2)
        pygame.draw.arc(s, _shade(accent, 0.8), (head_c[0] - 9, head_c[1] - 11, 18, 12), 0.2, 2.9, 4)
    if not facing_back:
        arm_and_weapon()
    if flash > 0:
        _apply_flash(s, flash)
    return s


def draw_beast(look: dict, aim_angle: float, walk_phase: float, moving: bool, attack_t=None, flash=0.0) -> pygame.Surface:
    s = pygame.Surface((CANVAS_W, CANVAS_H), pygame.SRCALPHA)
    body = tuple(look.get("body", (120, 86, 70)))
    skin = tuple(look.get("skin", (160, 110, 90)))
    accent = tuple(look.get("accent", (200, 60, 60)))
    ca, sa = math.cos(aim_angle), math.sin(aim_angle) * 0.6
    fx, fy = FEET_X, FEET_Y - 8
    lunge = math.sin(min(1.0, attack_t) * math.pi) * 10 if attack_t is not None else 0
    pts = [(fx - ca * 14 + ca * lunge * 0.3, fy - sa * 14), (fx + ca * lunge * 0.6, fy), (fx + ca * (14 + lunge), fy + sa * (14 + lunge))]
    legs = []
    for i, (px, py) in enumerate(pts[:2]):
        ph = walk_phase + i * math.pi
        off = math.sin(ph) * 4 if moving else 0
        legs.append(((px - 3 + off, py + 4), (px - 3 + off, py + 14)))
        legs.append(((px + 3 - off, py + 4), (px + 3 - off, py + 14)))
    for a, b in legs:
        pygame.draw.line(s, INK, a, b, 5)
        pygame.draw.line(s, _shade(body, 0.7), a, b, 3)
    order = pts if sa >= 0 else list(reversed(pts))
    for i, (px, py) in enumerate(order):
        r = 10 if i != len(order) // 2 else 11
        _outline_circle(s, body, (int(px), int(py)), r)
    head = pts[2]
    pygame.draw.line(s, _shade(body, 1.3), (pts[0][0], pts[0][1] - 8), (pts[1][0], pts[1][1] - 10), 3)
    # Spiky mutated back.
    for i in range(3):
        bx = pts[0][0] + (pts[1][0] - pts[0][0]) * i / 2
        by = pts[0][1] + (pts[1][1] - pts[0][1]) * i / 2 - 9
        pygame.draw.polygon(s, accent, [(bx - 3, by + 2), (bx, by - 6), (bx + 3, by + 2)])
    hx, hy = head[0] + ca * 6, head[1] - 4
    _outline_circle(s, skin, (int(hx), int(hy)), 7)
    if sa > -0.3:
        pygame.draw.circle(s, accent, (int(hx + ca * 3 - 2), int(hy - 1)), 2)
        pygame.draw.circle(s, accent, (int(hx + ca * 3 + 2), int(hy - 1)), 2)
    if attack_t is not None:
        pygame.draw.arc(s, (240, 240, 230), (hx - 6 + ca * 4, hy - 2, 12, 8), 3.4, 6.0, 2)
    if flash > 0:
        _apply_flash(s, flash)
    return s


def draw_small(look: dict, aim_angle: float, walk_phase: float, moving: bool, flash=0.0) -> pygame.Surface:
    s = pygame.Surface((CANVAS_W, CANVAS_H), pygame.SRCALPHA)
    body = tuple(look.get("body", (176, 150, 116)))
    accent = tuple(look.get("accent", (240, 230, 220)))
    ca = math.cos(aim_angle)
    hop = abs(math.sin(walk_phase * 1.5)) * 6 if moving else 0
    cx, cy = FEET_X, FEET_Y - 6 - hop
    pygame.draw.ellipse(s, INK, (cx - 10, cy - 7, 20, 14))
    pygame.draw.ellipse(s, body, (cx - 8, cy - 5, 16, 10))
    hx = cx + ca * 8
    _outline_circle(s, body, (int(hx), int(cy - 6)), 5)
    for d in (-2, 2):
        pygame.draw.line(s, INK, (hx + d, cy - 10), (hx + d * 2, cy - 22), 4)
        pygame.draw.line(s, body, (hx + d, cy - 10), (hx + d * 2, cy - 21), 2)
    pygame.draw.circle(s, accent, (int(cx - ca * 9), int(cy - 1)), 3)
    pygame.draw.circle(s, INK, (int(hx + ca * 2), int(cy - 7)), 1)
    if flash > 0:
        _apply_flash(s, flash)
    return s


def _apply_flash(s: pygame.Surface, amount: float):
    v = int(255 * min(1.0, amount))
    s.fill((v, v, v, 0), special_flags=pygame.BLEND_RGBA_ADD)


def death_pose(s: pygame.Surface, t: float, direction: float) -> pygame.Surface:
    """Rotate the character to the ground and fade, t∈[0,1+]."""
    ang = -90 * min(1.0, t * 2.2) * (1 if math.cos(direction) >= 0 else -1)
    out = pygame.transform.rotate(s, ang)
    if t > 0.6:
        out.set_alpha(max(0, int(255 * (1 - (t - 0.6) / 1.4))))
    return out
