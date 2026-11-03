"""Shared UI drawing helpers in a worn, hand-painted wasteland style."""
import pygame

PANEL = (34, 29, 25)
PANEL_LIGHT = (52, 44, 37)
BORDER = (126, 100, 72)
BORDER_DARK = (18, 14, 12)
ACCENT = (232, 172, 82)
TEXT = (238, 230, 212)
DIM = (162, 150, 130)
BAD = (232, 96, 78)
GOOD = (150, 214, 120)
RARITY = {"common": (170, 160, 140), "uncommon": (120, 190, 110), "rare": (100, 160, 240), "epic": (200, 120, 240)}


class UIContext:
    """Font + text cache shared by all UI widgets."""

    def __init__(self, assets):
        self.assets = assets
        self._cache: dict = {}

    def text(self, s: str, size: int = 16, color=TEXT, bold: bool = False, shadow: bool = True) -> pygame.Surface:
        key = (s, size, color, bold, shadow)
        surf = self._cache.get(key)
        if surf is None:
            font = self.assets.font(size, bold)
            base = font.render(s, True, color)
            if shadow:
                surf = pygame.Surface((base.get_width() + 2, base.get_height() + 2), pygame.SRCALPHA)
                surf.blit(font.render(s, True, (10, 8, 6)), (2, 2))
                surf.blit(base, (0, 0))
            else:
                surf = base
            if len(self._cache) > 1500:
                self._cache.clear()
            self._cache[key] = surf
        return surf

    def wrap(self, s: str, size: int, width: int) -> list[str]:
        font = self.assets.font(size)
        lines, line = [], ""
        for word in s.replace("\n", " \n ").split(" "):
            if word == "\n":
                lines.append(line)
                line = ""
                continue
            test = (line + " " + word).strip()
            if font.size(test)[0] > width and line:
                lines.append(line)
                line = word
            else:
                line = test
        if line:
            lines.append(line)
        return lines


def panel(surf: pygame.Surface, rect, title: str | None = None, ui: UIContext | None = None, alpha: int = 236):
    rect = pygame.Rect(rect)
    s = pygame.Surface(rect.size, pygame.SRCALPHA)
    pygame.draw.rect(s, (*PANEL, alpha), s.get_rect(), border_radius=10)
    # Painted inner bevel.
    pygame.draw.rect(s, (*PANEL_LIGHT, alpha), s.get_rect().inflate(-6, -6), 2, border_radius=8)
    surf.blit(s, rect.topleft)
    pygame.draw.rect(surf, BORDER_DARK, rect.inflate(4, 4), 3, border_radius=12)
    pygame.draw.rect(surf, BORDER, rect, 2, border_radius=10)
    if title and ui:
        t = ui.text(title, 20, ACCENT, True)
        tab = pygame.Rect(rect.x + 14, rect.y - 14, t.get_width() + 20, 28)
        pygame.draw.rect(surf, BORDER_DARK, tab.inflate(4, 4), border_radius=8)
        pygame.draw.rect(surf, (70, 54, 40), tab, border_radius=7)
        surf.blit(t, (tab.x + 10, tab.y + 3))


def bar(surf, rect, frac: float, color, bg=(26, 22, 20), border=True, segments: int = 0):
    rect = pygame.Rect(rect)
    frac = max(0.0, min(1.0, frac))
    pygame.draw.rect(surf, bg, rect, border_radius=4)
    if frac > 0:
        fill = rect.inflate(-4, -4)
        fill.width = max(1, int(fill.width * frac))
        pygame.draw.rect(surf, color, fill, border_radius=3)
        hi = fill.copy()
        hi.height = max(1, fill.height // 3)
        pygame.draw.rect(surf, tuple(min(255, int(c * 1.25)) for c in color), hi, border_radius=3)
    if segments:
        for i in range(1, segments):
            x = rect.x + rect.width * i // segments
            pygame.draw.line(surf, bg, (x, rect.y + 2), (x, rect.bottom - 3), 1)
    if border:
        pygame.draw.rect(surf, BORDER_DARK, rect, 2, border_radius=4)


def slot(surf, rect, selected=False, hovered=False, color=(46, 40, 34)):
    rect = pygame.Rect(rect)
    pygame.draw.rect(surf, color, rect, border_radius=6)
    pygame.draw.rect(surf, (24, 20, 17), rect, 2, border_radius=6)
    if hovered:
        pygame.draw.rect(surf, (200, 180, 140), rect, 2, border_radius=6)
    if selected:
        pygame.draw.rect(surf, ACCENT, rect.inflate(2, 2), 3, border_radius=7)


class NavRepeater:
    """Turns held directional UI actions into repeated steps (for pads and arrow keys)."""

    def __init__(self, delay: float = 0.32, rate: float = 0.09):
        self.delay, self.rate = delay, rate
        self.timer = 0.0
        self.last = (0, 0)

    def step(self, inp, dt: float) -> tuple[int, int]:
        dx = (1 if inp.held("ui_right") else 0) - (1 if inp.held("ui_left") else 0)
        dy = (1 if inp.held("ui_down") else 0) - (1 if inp.held("ui_up") else 0)
        cur = (dx, dy)
        if cur == (0, 0):
            self.last = cur
            return 0, 0
        if cur != self.last:
            self.last = cur
            self.timer = self.delay
            return cur
        self.timer -= dt
        if self.timer <= 0:
            self.timer = self.rate
            return cur
        return 0, 0
