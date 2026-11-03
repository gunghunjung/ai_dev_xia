"""Tabbed survival menu (inventory / crafting / status) shown over the live world."""
import pygame

from ui.character_status import StatusUI
from ui.crafting_ui import CraftingUI
from ui.inventory_ui import InventoryUI
from ui.widgets import TEXT, ACCENT

TABS = [("inventory", "인벤토리"), ("crafting", "제작"), ("status", "상태")]


class GameMenu:
    def __init__(self, ui, prompts, assets):
        self.ui = ui
        self.prompts = prompts
        self.pages = {"inventory": InventoryUI(ui, prompts, assets),
                      "crafting": CraftingUI(ui, prompts, assets),
                      "status": StatusUI(ui, prompts, assets)}
        self.tab = "inventory"
        self.is_open = False
        self.session = None
        self.tab_rects = {}

    def open(self, session, tab="inventory", container=None):
        self.session = session
        self.is_open = True
        self.tab = tab
        self.pages["inventory"].open(session, container)
        self.pages["crafting"].open(session)
        self.pages["status"].open(session)

    def close(self):
        for p in self.pages.values():
            p.close()
        self.is_open = False

    def switch(self, delta: int):
        keys = [k for k, _ in TABS]
        self.tab = keys[(keys.index(self.tab) + delta) % len(keys)]
        self.session.audio.play("ui_move", 0.4)

    def handle(self, inp, events, dt):
        page = self.pages[self.tab]
        busy = self.tab == "inventory" and (page.context is not None or page.held is not None)
        for e in events:
            if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                for key, r in self.tab_rects.items():
                    if r.collidepoint(e.pos):
                        self.tab = key
                        return None
        if not busy:
            if inp.pressed("ui_tab_prev"):
                self.switch(-1)
                return None
            if inp.pressed("ui_tab_next"):
                self.switch(1)
                return None
            if inp.pressed("inventory") or inp.pressed("crafting") and self.tab == "crafting":
                self.close()
                return "close"
        res = page.handle(inp, events, dt)
        if res == "close":
            self.close()
        return res

    def draw(self, screen, inp):
        W, H = screen.get_size()
        dim = pygame.Surface((W, H), pygame.SRCALPHA)
        dim.fill((10, 8, 6, 165))
        screen.blit(dim, (0, 0))
        x = W // 2 - 220
        self.tab_rects = {}
        g1 = self.prompts.glyph("ui_tab_prev")
        screen.blit(g1, (x - g1.get_width() - 10, 22))
        for key, label in TABS:
            sel = key == self.tab
            t = self.ui.text(label, 20, (20, 16, 14) if sel else TEXT, True, False)
            r = pygame.Rect(x, 18, 140, 34)
            pygame.draw.rect(screen, ACCENT if sel else (52, 44, 37), r, border_radius=10)
            pygame.draw.rect(screen, (18, 14, 12), r, 2, border_radius=10)
            screen.blit(t, (r.centerx - t.get_width() // 2, r.centery - t.get_height() // 2))
            self.tab_rects[key] = r
            x += 150
        screen.blit(self.prompts.glyph("ui_tab_next"), (x, 22))
        area = pygame.Rect(0, 70, W, H - 90)
        self.pages[self.tab].draw(screen, area, inp)
