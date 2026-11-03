"""Title, pause and death menus (fully navigable with mouse, keyboard or pad)."""
import math

import pygame

from ui.widgets import panel, NavRepeater, TEXT, DIM, ACCENT


class VerticalMenu:
    def __init__(self, ui, prompts, options: list[tuple[str, str]]):
        self.ui = ui
        self.prompts = prompts
        self.options = options
        self.index = 0
        self.nav = NavRepeater()
        self.rects = []
        self.disabled: set[str] = set()

    def handle(self, inp, events, dt):
        for e in events:
            if e.type == pygame.MOUSEMOTION:
                for i, r in enumerate(self.rects):
                    if r.collidepoint(e.pos):
                        self.index = i
            elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                for i, r in enumerate(self.rects):
                    if r.collidepoint(e.pos) and self.options[i][0] not in self.disabled:
                        return self.options[i][0]
        _, dy = self.nav.step(inp, dt)
        if dy:
            self.index = (self.index + dy) % len(self.options)
        if inp.pressed("ui_confirm"):
            key = self.options[self.index][0]
            if key not in self.disabled:
                return key
        if inp.pressed("ui_back"):
            return "back"
        return None

    def draw(self, screen, cx, y, width=320):
        self.rects = []
        for i, (key, label) in enumerate(self.options):
            r = pygame.Rect(cx - width // 2, y + i * 54, width, 46)
            self.rects.append(r)
            sel = i == self.index
            off = key in self.disabled
            pygame.draw.rect(screen, (18, 14, 12), r.inflate(4, 4), border_radius=10)
            pygame.draw.rect(screen, (110, 80, 50) if sel else (52, 44, 37), r, border_radius=9)
            if sel:
                pygame.draw.rect(screen, ACCENT, r, 2, border_radius=9)
            t = self.ui.text(label, 20, DIM if off else TEXT, True)
            screen.blit(t, (r.centerx - t.get_width() // 2, r.centery - t.get_height() // 2))


def draw_title(screen, ui, t, subtitle=True):
    W, H = screen.get_size()
    # Painted dusk sky gradient + ruined skyline silhouette.
    for i in range(0, H, 4):
        k = i / H
        col = (int(60 + 150 * k), int(40 + 80 * k), int(46 + 30 * k))
        pygame.draw.rect(screen, col, (0, i, W, 4))
    pygame.draw.circle(screen, (250, 190, 110), (int(W * 0.72), int(H * 0.62)), 90)
    base = int(H * 0.68)
    x = 0
    rng = 7
    while x < W:
        rng = (rng * 1103515245 + 12345) & 0x7FFFFFFF
        w = 40 + rng % 90
        h = 40 + (rng >> 8) % 160
        pts = [(x, base), (x, base - h), (x + w * 0.4, base - h - (rng % 20)), (x + w * 0.7, base - h + 18), (x + w, base - h + 6), (x + w, base)]
        pygame.draw.polygon(screen, (40, 28, 30), pts)
        x += w + (rng >> 4) % 30
    pygame.draw.rect(screen, (58, 40, 34), (0, base, W, H - base))
    for i in range(40):
        dx = (i * 97 + t * 30) % W
        pygame.draw.line(screen, (120, 80, 60), (dx, base + 20 + (i * 37) % 200), (dx + 30, base + 22 + (i * 37) % 200), 2)
    title = ui.text("PROJECT WASTELAND", 64, (246, 214, 150), True)
    wob = math.sin(t * 1.3) * 2
    screen.blit(title, (W // 2 - title.get_width() // 2, 70 + wob))
    if subtitle:
        sub = ui.text("황무지에서 홀로 살아남아라", 20, (230, 200, 170), True)
        screen.blit(sub, (W // 2 - sub.get_width() // 2, 150))


class MainMenu:
    def __init__(self, ui, prompts, has_save: bool):
        self.ui = ui
        self.prompts = prompts
        self.menu = VerticalMenu(ui, prompts, [("continue", "이어하기"), ("new", "새 게임"), ("settings", "설정"), ("quit", "종료")])
        if not has_save:
            self.menu.disabled.add("continue")
            self.menu.index = 1
        self.t = 0.0
        self.note = ""

    def handle(self, inp, events, dt):
        self.t += dt
        res = self.menu.handle(inp, events, dt)
        return None if res == "back" else res

    def draw(self, screen, inp):
        W, H = screen.get_size()
        draw_title(screen, self.ui, self.t)
        self.menu.draw(screen, W // 2, 260)
        if self.note:
            t = self.ui.text(self.note, 16, (255, 160, 130), True)
            screen.blit(t, (W // 2 - t.get_width() // 2, 230))
        dev = "Xbox 패드" if inp.last_device == "pad" else "키보드 / 마우스"
        t = self.ui.text(f"입력 장치: {dev}" + ("  · 패드 연결됨" if inp.has_pad else ""), 14, DIM)
        screen.blit(t, (20, H - 30))
        hint = self.prompts.row([("ui_confirm", "선택")])
        screen.blit(hint, (W - hint.get_width() - 20, H - 34))


class PauseMenu:
    def __init__(self, ui, prompts):
        self.ui = ui
        self.prompts = prompts
        self.menu = VerticalMenu(ui, prompts, [("resume", "계속하기"), ("save", "저장하기"), ("load", "불러오기"),
                                               ("settings", "설정"), ("title", "메인 메뉴"), ("quit", "게임 종료")])
        self.note = ""
        self.note_t = 0.0

    def handle(self, inp, events, dt):
        self.note_t = max(0.0, self.note_t - dt)
        res = self.menu.handle(inp, events, dt)
        if res == "back" or inp.pressed("pause"):
            return "resume"
        return res

    def draw(self, screen, inp):
        W, H = screen.get_size()
        dim = pygame.Surface((W, H), pygame.SRCALPHA)
        dim.fill((10, 8, 6, 180))
        screen.blit(dim, (0, 0))
        r = pygame.Rect(W // 2 - 200, 120, 400, 400)
        panel(screen, r, "일시 정지", self.ui)
        self.menu.draw(screen, W // 2, r.y + 40)
        if self.note_t > 0:
            t = self.ui.text(self.note, 16, ACCENT, True)
            screen.blit(t, (W // 2 - t.get_width() // 2, r.bottom + 14))


class DeathScreen:
    def __init__(self, ui, prompts, stats, has_save: bool):
        self.ui = ui
        self.prompts = prompts
        self.stats = stats
        self.menu = VerticalMenu(ui, prompts, [("load", "마지막 저장 불러오기"), ("new", "새 게임"), ("title", "메인 메뉴")])
        if not has_save:
            self.menu.disabled.add("load")
            self.menu.index = 1
        self.t = 0.0

    def handle(self, inp, events, dt):
        self.t += dt
        if self.t < 1.0:
            return None
        res = self.menu.handle(inp, events, dt)
        return None if res == "back" else res

    def draw(self, screen, inp):
        W, H = screen.get_size()
        dim = pygame.Surface((W, H), pygame.SRCALPHA)
        dim.fill((30, 4, 4, min(210, int(self.t * 160))))
        screen.blit(dim, (0, 0))
        t = self.ui.text("당신은 황무지에 쓰러졌다", 40, (230, 90, 80), True)
        screen.blit(t, (W // 2 - t.get_width() // 2, 140))
        s = self.ui.text(f"생존 {self.stats.days_survived:.1f}일 · 처치 {self.stats.kills}", 20, TEXT, True)
        screen.blit(s, (W // 2 - s.get_width() // 2, 200))
        if self.t >= 1.0:
            self.menu.draw(screen, W // 2, 270, 380)
