"""Settings screen: feedback intensity, audio, controls and button remapping."""
import pygame

from engine.input_manager import key_label, pad_button_label
from ui.widgets import panel, bar, NavRepeater, TEXT, DIM, ACCENT

SLIDERS = [
    ("master_volume", "전체 음량", 0.0, 1.0, 0.1),
    ("sfx_volume", "효과음", 0.0, 1.0, 0.1),
    ("screen_shake", "화면 흔들림", 0.0, 1.5, 0.1),
    ("hitstop", "히트스톱", 0.0, 1.5, 0.1),
    ("vibration", "진동 세기", 0.0, 1.0, 0.1),
    ("aim_assist", "조준 보정 세기", 0.0, 1.0, 0.1),
    ("stick_deadzone", "스틱 데드존", 0.05, 0.5, 0.01),
    ("aim_sensitivity", "조준 감도", 0.3, 2.0, 0.1),
]
TOGGLES = [
    ("aim_assist_enabled", "조준 보정 (패드)"),
    ("damage_numbers", "피해 숫자 표시"),
    ("show_fps", "성능 정보 (F3)"),
    ("fullscreen", "전체 화면"),
]


class SettingsUI:
    def __init__(self, ui, prompts, assets, inp, settings, on_change):
        self.ui = ui
        self.prompts = prompts
        self.inp = inp
        self.settings = settings
        self.on_change = on_change
        self.nav = NavRepeater()
        self.index = 0
        self.scroll = 0
        self.capturing = None
        self.bind_device = "kbm"
        self.rows = []

    def items(self):
        out = [("slider", s) for s in SLIDERS] + [("toggle", t) for t in TOGGLES]
        out.append(("device", None))
        for action in self.inp.profiles["rebindable"]:
            if self.bind_device == "pad" and action in ("reload", "dig", "crafting", "slot_5"):
                continue
            if self.bind_device == "kbm" and action == "tool":
                continue
            out.append(("bind", action))
        out.append(("reset", None))
        out.append(("back", None))
        return out

    def handle(self, inp, events, dt):
        if self.capturing is not None:
            code = inp.captured
            if code is not None:
                inp.end_capture()
                if code != pygame.K_ESCAPE and code != "MENU":
                    inp.rebind(self.capturing, self.bind_device, code)
                    self.settings["bindings"] = inp.export_bindings()
                    self.on_change()
                self.capturing = None
            return None
        items = self.items()
        for e in events:
            if e.type == pygame.MOUSEMOTION:
                for i, r in enumerate(self.rows):
                    if r.collidepoint(e.pos):
                        self.index = self.scroll + i
            elif e.type == pygame.MOUSEBUTTONDOWN and e.button in (1, 3):
                for i, r in enumerate(self.rows):
                    if r.collidepoint(e.pos):
                        self.index = self.scroll + i
                        kind = items[self.index][0]
                        if kind == "slider":
                            self._adjust(items[self.index], 1 if e.button == 1 else -1)
                        else:
                            res = self._activate(items[self.index])
                            if res:
                                return res
            elif e.type == pygame.MOUSEWHEEL:
                self.index = max(0, min(len(items) - 1, self.index - e.y))
        dx, dy = self.nav.step(inp, dt)
        if dy:
            self.index = (self.index + dy) % len(items)
        if dx:
            self._adjust(items[self.index], dx)
        if inp.pressed("ui_confirm"):
            res = self._activate(items[self.index])
            if res:
                return res
        if inp.pressed("ui_back"):
            return "close"
        return None

    def _adjust(self, item, d):
        kind, data = item
        if kind == "slider":
            key, _, lo, hi, step = data
            v = round(min(hi, max(lo, float(self.settings.get(key, lo)) + d * step)), 3)
            self.settings[key] = v
            self.on_change()
        elif kind == "toggle":
            self.settings[data[0]] = not self.settings.get(data[0], False)
            self.on_change()
        elif kind == "device":
            self.bind_device = "pad" if self.bind_device == "kbm" else "kbm"

    def _activate(self, item):
        kind, data = item
        if kind in ("toggle", "device"):
            self._adjust(item, 1)
        elif kind == "bind":
            self.capturing = data
            self.inp.begin_capture(self.bind_device)
        elif kind == "reset":
            self.inp.reset_bindings()
            self.settings["bindings"] = None
            self.on_change()
        elif kind == "back":
            return "close"
        return None

    def draw(self, screen):
        W, H = screen.get_size()
        dim = pygame.Surface((W, H), pygame.SRCALPHA)
        dim.fill((10, 8, 6, 200))
        screen.blit(dim, (0, 0))
        r = pygame.Rect(W // 2 - 380, 50, 760, H - 100)
        panel(screen, r, "설정", self.ui)
        items = self.items()
        row_h = 34
        visible = (r.height - 70) // row_h
        if self.index < self.scroll:
            self.scroll = self.index
        if self.index >= self.scroll + visible:
            self.scroll = self.index - visible + 1
        self.rows = []
        y = r.y + 24
        names = self.inp.profiles["action_names"]
        for i, (kind, data) in enumerate(items[self.scroll:self.scroll + visible]):
            idx = self.scroll + i
            row = pygame.Rect(r.x + 20, y, r.width - 40, row_h - 4)
            self.rows.append(row)
            if idx == self.index:
                pygame.draw.rect(screen, (88, 68, 46), row, border_radius=6)
            if kind == "slider":
                key, label, lo, hi, _ = data
                v = float(self.settings.get(key, lo))
                screen.blit(self.ui.text(label, 16, TEXT, True), (row.x + 12, row.y + 4))
                bar(screen, (row.x + 300, row.y + 8, 300, 14), (v - lo) / (hi - lo), ACCENT)
                screen.blit(self.ui.text(f"{v:.2f}", 15, DIM, True), (row.x + 620, row.y + 4))
            elif kind == "toggle":
                key, label = data
                on = bool(self.settings.get(key, False))
                screen.blit(self.ui.text(label, 16, TEXT, True), (row.x + 12, row.y + 4))
                screen.blit(self.ui.text("켜짐" if on else "꺼짐", 16, ACCENT if on else DIM, True), (row.x + 300, row.y + 4))
            elif kind == "device":
                screen.blit(self.ui.text("버튼 매핑 대상", 16, ACCENT, True), (row.x + 12, row.y + 4))
                screen.blit(self.ui.text("◀ " + ("키보드/마우스" if self.bind_device == "kbm" else "Xbox 패드") + " ▶", 16, TEXT, True), (row.x + 300, row.y + 4))
            elif kind == "bind":
                screen.blit(self.ui.text(names.get(data, data), 16, TEXT, True), (row.x + 32, row.y + 4))
                if self.capturing == data:
                    msg = "키를 누르세요... (Esc 취소)" if self.bind_device == "kbm" else f"버튼을 누르세요... ({pad_button_label('MENU', self.inp.active_pad.name if self.inp.active_pad else '')} 취소)"
                    screen.blit(self.ui.text(msg, 15, ACCENT, True), (row.x + 300, row.y + 4))
                else:
                    if self.bind_device == "pad":
                        g = self.prompts.pad_glyph((self.inp.pad_bindings.get(data) or ["-"])[0])
                    else:
                        codes = self.inp.kb_bindings.get(data) or []
                        g = self.prompts.key_glyph(key_label(codes[0]) if codes else "-")
                    screen.blit(g, (row.x + 300, row.y + 2))
            elif kind == "reset":
                screen.blit(self.ui.text("버튼 매핑 초기화", 16, TEXT, True), (row.x + 12, row.y + 4))
            elif kind == "back":
                screen.blit(self.ui.text("돌아가기", 16, ACCENT, True), (row.x + 12, row.y + 4))
            y += row_h
        hint = self.prompts.row([("ui_confirm", "선택"), ("ui_left", "조절"), ("ui_back", "뒤로")])
        screen.blit(hint, (r.x + 20, r.bottom - 34))
