"""Device-aware button prompts: Xbox glyphs for pads, keycaps for keyboard/mouse."""
import pygame

from engine.input_manager import key_label, pad_button_label

PAD_COLORS = {"A": (96, 180, 70), "B": (220, 70, 60), "X": (70, 130, 220), "Y": (230, 190, 50)}


class ButtonPrompts:
    def __init__(self, ui, inp):
        self.ui = ui
        self.inp = inp
        self._cache: dict = {}

    def glyph(self, action: str, device: str | None = None) -> pygame.Surface:
        device = device or self.inp.last_device
        if device == "pad" and not self.inp.has_pad:
            device = "kbm"
        if device == "pad":
            names = self.inp.pad_bindings.get(action) or ["?"]
            return self.pad_glyph(names[0])
        codes = self.inp.kb_bindings.get(action) or []
        return self.key_glyph(key_label(codes[0]) if codes else "-")

    def pad_glyph(self, name: str) -> pygame.Surface:
        label = pad_button_label(name, self.inp.active_pad.name if self.inp.active_pad else "")
        key = ("pad", name, label)
        s = self._cache.get(key)
        if s is not None:
            return s
        ink = (20, 16, 14)
        if name in PAD_COLORS:
            s = pygame.Surface((26, 26), pygame.SRCALPHA)
            pygame.draw.circle(s, ink, (13, 13), 13)
            pygame.draw.circle(s, PAD_COLORS[name], (13, 13), 11)
            t = self.ui.text(name, 14, (255, 255, 255), True, False)
            s.blit(t, (13 - t.get_width() // 2, 13 - t.get_height() // 2))
        elif name.startswith("DPAD"):
            s = pygame.Surface((26, 26), pygame.SRCALPHA)
            pygame.draw.rect(s, ink, (9, 1, 8, 24), border_radius=2)
            pygame.draw.rect(s, ink, (1, 9, 24, 8), border_radius=2)
            pygame.draw.rect(s, (90, 90, 96), (10, 2, 6, 22))
            pygame.draw.rect(s, (90, 90, 96), (2, 10, 22, 6))
            d = name.split("_")[1]
            hl = {"UP": (10, 2, 6, 8), "DOWN": (10, 16, 6, 8), "LEFT": (2, 10, 8, 6), "RIGHT": (16, 10, 8, 6)}[d]
            pygame.draw.rect(s, (240, 220, 140), hl)
        else:
            t = self.ui.text(label, 12, (240, 236, 226), True, False)
            w = max(30, t.get_width() + 12)
            s = pygame.Surface((w, 22), pygame.SRCALPHA)
            pygame.draw.rect(s, ink, s.get_rect(), border_radius=8)
            pygame.draw.rect(s, (70, 72, 80), s.get_rect().inflate(-4, -4), border_radius=7)
            s.blit(t, (w // 2 - t.get_width() // 2, 11 - t.get_height() // 2))
        self._cache[key] = s
        return s

    def key_glyph(self, label: str) -> pygame.Surface:
        key = ("kb", label)
        s = self._cache.get(key)
        if s is not None:
            return s
        t = self.ui.text(label, 13, (30, 26, 22), True, False)
        w = max(24, t.get_width() + 12)
        s = pygame.Surface((w, 24), pygame.SRCALPHA)
        pygame.draw.rect(s, (20, 16, 14), s.get_rect(), border_radius=5)
        pygame.draw.rect(s, (226, 216, 196), (2, 2, w - 4, 18), border_radius=4)
        pygame.draw.rect(s, (180, 168, 148), (2, 18, w - 4, 4), border_radius=3)
        s.blit(t, (w // 2 - t.get_width() // 2, 11 - t.get_height() // 2))
        self._cache[key] = s
        return s

    def prompt(self, action: str, text: str, color=(238, 230, 212), size: int = 15) -> pygame.Surface:
        g = self.glyph(action)
        t = self.ui.text(text, size, color, True)
        s = pygame.Surface((g.get_width() + 6 + t.get_width(), max(g.get_height(), t.get_height())), pygame.SRCALPHA)
        s.blit(g, (0, (s.get_height() - g.get_height()) // 2))
        s.blit(t, (g.get_width() + 6, (s.get_height() - t.get_height()) // 2))
        return s

    def row(self, items: list[tuple[str, str]], gap: int = 16) -> pygame.Surface:
        parts = [self.prompt(a, t) for a, t in items]
        w = sum(p.get_width() for p in parts) + gap * max(0, len(parts) - 1)
        h = max((p.get_height() for p in parts), default=1)
        s = pygame.Surface((max(1, w), h), pygame.SRCALPHA)
        x = 0
        for p in parts:
            s.blit(p, (x, (h - p.get_height()) // 2))
            x += p.get_width() + gap
        return s
