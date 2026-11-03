"""Crafting and repair screen (recipes from data/recipes.json)."""
import pygame

from engine import placeholder_art as art
from ui.widgets import panel, bar, NavRepeater, TEXT, DIM, ACCENT, BAD, GOOD

ROW_H = 46


class CraftingUI:
    def __init__(self, ui, prompts, assets):
        self.ui = ui
        self.prompts = prompts
        self.assets = assets
        self.session = None
        self.mode = "craft"     # craft | repair
        self.index = 0
        self.scroll = 0
        self.nav = NavRepeater()
        self.rows: list[pygame.Rect] = []
        self.button = None
        self.mode_rects = {}
        self.msg = ""
        self.msg_t = 0.0

    def open(self, session):
        self.session = session
        self.index = 0
        self.scroll = 0

    def close(self):
        pass

    def icon(self, item_id, size=36):
        idef = self.session.db.item(item_id)
        return self.assets.image(f"ui/icons/{item_id}_{size}", lambda: art.make_icon(idef.get("icon", {}), 1, 1, size))

    def entries(self):
        s = self.session
        if self.mode == "repair":
            return s.crafting.repairables()
        stations = s.crafting.stations_near(s.player.x, s.player.y)
        inv = s.player.inventory

        def order(r):
            ok, _ = s.crafting.can_craft(r, inv, stations)
            return (0 if ok else 1 if not r.get("station") or r["station"] in stations else 2)
        return sorted(s.crafting.recipes, key=order)

    def activate(self):
        s = self.session
        ents = self.entries()
        if not ents:
            return
        e = ents[min(self.index, len(ents) - 1)]
        if self.mode == "repair":
            ok, msg = s.crafting.repair(e)
        else:
            ok, msg = s.crafting.start(e["id"])
        if msg:
            self.msg, self.msg_t = msg, 2.0
        if not ok:
            s.audio.play("dry_fire", 0.4)

    def handle(self, inp, events, dt):
        self.msg_t = max(0.0, self.msg_t - dt)
        ents = self.entries()
        for e in events:
            if e.type == pygame.MOUSEMOTION:
                for i, r in enumerate(self.rows):
                    if r.collidepoint(e.pos):
                        self.index = self.scroll + i
            elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                for mode, r in self.mode_rects.items():
                    if r.collidepoint(e.pos):
                        self.mode, self.index, self.scroll = mode, 0, 0
                        return None
                if self.button and self.button.collidepoint(e.pos):
                    self.activate()
                for i, r in enumerate(self.rows):
                    if r.collidepoint(e.pos):
                        self.index = self.scroll + i
            elif e.type == pygame.MOUSEWHEEL:
                self.scroll = max(0, self.scroll - e.y)
        dx, dy = self.nav.step(inp, dt)
        if dy and ents:
            self.index = (self.index + dy) % len(ents)
            self.session.audio.play("ui_move", 0.3)
        if inp.pressed("ui_menu"):
            self.mode = "repair" if self.mode == "craft" else "craft"
            self.index = self.scroll = 0
        if inp.pressed("ui_confirm"):
            self.activate()
        if inp.pressed("ui_back"):
            return "close"
        return None

    def draw(self, screen, area, inp):
        s = self.session
        ui = self.ui
        p = s.player
        stations = s.crafting.stations_near(p.x, p.y)
        left = pygame.Rect(area.x + 60, area.y + 40, 470, area.height - 90)
        right = pygame.Rect(left.right + 30, left.y, area.right - left.right - 90, left.height)
        panel(screen, left, "제작" if self.mode == "craft" else "수리", ui)
        # Mode toggle chips
        self.mode_rects = {}
        x = left.x + 120
        for mode, label in (("craft", "제작"), ("repair", "수리")):
            t = ui.text(label, 14, (20, 16, 14) if self.mode == mode else TEXT, True, False)
            r = pygame.Rect(x, left.y - 12, t.get_width() + 20, 24)
            pygame.draw.rect(screen, ACCENT if self.mode == mode else (60, 50, 42), r, border_radius=10)
            screen.blit(t, (r.x + 10, r.y + 3))
            self.mode_rects[mode] = r
            x = r.right + 6
        g = self.prompts.glyph("ui_menu")
        screen.blit(g, (x + 4, left.y - 12))
        st_names = [s.resources["structures"][k]["name"] for k in sorted(stations)] or ["없음"]
        screen.blit(ui.text("근처 시설: " + ", ".join(st_names), 13, DIM), (left.x + 16, left.y + 18))
        ents = self.entries()
        visible = max(1, (left.height - 60) // ROW_H)
        self.index = min(self.index, max(0, len(ents) - 1))
        if self.index < self.scroll:
            self.scroll = self.index
        if self.index >= self.scroll + visible:
            self.scroll = self.index - visible + 1
        self.scroll = min(self.scroll, max(0, len(ents) - visible))
        self.rows = []
        y = left.y + 44
        if not ents:
            screen.blit(ui.text("수리할 장비가 없다" if self.mode == "repair" else "레시피 없음", 15, DIM), (left.x + 20, y + 10))
        for i, e in enumerate(ents[self.scroll:self.scroll + visible]):
            idx = self.scroll + i
            r = pygame.Rect(left.x + 10, y, left.width - 20, ROW_H - 4)
            self.rows.append(r)
            sel = idx == self.index
            pygame.draw.rect(screen, (88, 68, 46) if sel else (44, 38, 32), r, border_radius=6)
            if sel:
                pygame.draw.rect(screen, ACCENT, r, 2, border_radius=6)
            if self.mode == "craft":
                item_id, count = e["result"]
                ok, why = s.crafting.can_craft(e, p.inventory, stations)
                col = TEXT if ok else DIM
                screen.blit(self.icon(item_id), (r.x + 6, r.y + 3))
                screen.blit(ui.text(f"{s.db.name(item_id)}" + (f" x{count}" if count > 1 else ""), 16, col, True), (r.x + 50, r.y + 4))
                sub = "제작 가능" if ok else why
                screen.blit(ui.text(sub, 12, GOOD if ok else BAD if why == "재료 부족" else DIM), (r.x + 50, r.y + 24))
            else:
                idef = s.db.item(e.item_id)
                screen.blit(self.icon(e.item_id), (r.x + 6, r.y + 3))
                screen.blit(ui.text(idef["name"], 16, TEXT, True), (r.x + 50, r.y + 4))
                bar(screen, (r.x + 50, r.y + 26, 160, 10), e.durability / idef["durability"], ACCENT)
            y += ROW_H

        panel(screen, right, "상세", ui)
        self.button = None
        if ents:
            e = ents[self.index]
            yy = right.y + 24
            if self.mode == "craft":
                item_id, count = e["result"]
                idef = s.db.item(item_id)
                screen.blit(self.icon(item_id, 72), (right.x + 16, yy))
                screen.blit(ui.text(idef["name"], 22, TEXT, True), (right.x + 100, yy + 4))
                ly = yy + 36
                for line in ui.wrap(idef.get("desc", ""), 14, right.width - 120):
                    screen.blit(ui.text(line, 14, DIM), (right.x + 100, ly))
                    ly += 19
                yy += 100
                screen.blit(ui.text("재료", 16, ACCENT, True), (right.x + 16, yy))
                yy += 28
                for ing, need in e["ingredients"]:
                    have = p.inventory.count(ing)
                    screen.blit(self.icon(ing, 30), (right.x + 18, yy))
                    col = GOOD if have >= need else BAD
                    screen.blit(ui.text(f"{s.db.name(ing)}", 15, TEXT, True), (right.x + 56, yy + 4))
                    t = ui.text(f"{have}/{need}", 15, col, True)
                    screen.blit(t, (right.right - t.get_width() - 20, yy + 4))
                    yy += 36
                if e.get("station"):
                    sname = s.resources["structures"][e["station"]]["name"]
                    have = e["station"] in stations
                    screen.blit(ui.text(f"필요 시설: {sname}", 14, GOOD if have else BAD, True), (right.x + 16, yy + 6))
                    yy += 28
                screen.blit(ui.text(f"제작 시간 {e.get('time', 1):.1f}초", 13, DIM), (right.x + 16, yy + 6))
                label = "제작"
            else:
                idef = s.db.item(e.item_id)
                rep = idef["repair"]
                screen.blit(self.icon(e.item_id, 72), (right.x + 16, yy))
                screen.blit(ui.text(idef["name"], 22, TEXT, True), (right.x + 100, yy + 4))
                screen.blit(ui.text(f"내구도 {int(e.durability)}/{idef['durability']}", 15, TEXT), (right.x + 100, yy + 38))
                yy += 100
                have = p.inventory.count(rep["item"])
                screen.blit(ui.text(f"필요: {s.db.name(rep['item'])} {have}/{rep['count']}", 15, GOOD if have >= rep["count"] else BAD, True), (right.x + 16, yy))
                screen.blit(ui.text(f"회복량 +{rep['amount']}", 14, DIM), (right.x + 16, yy + 26))
                if rep.get("station"):
                    sname = s.resources["structures"][rep["station"]]["name"]
                    screen.blit(ui.text(f"필요 시설: {sname}", 14, GOOD if rep["station"] in stations else BAD, True), (right.x + 16, yy + 50))
                label = "수리"
            btn = pygame.Rect(right.x + 16, right.bottom - 70, right.width - 32, 48)
            self.button = btn
            act = s.crafting.active
            pygame.draw.rect(screen, (96, 74, 46), btn, border_radius=8)
            if act:
                frac = act["t"] / act["dur"]
                pygame.draw.rect(screen, (160, 120, 60), (btn.x, btn.y, btn.width * frac, btn.height), border_radius=8)
                label = "제작 중..."
            pygame.draw.rect(screen, ACCENT, btn, 2, border_radius=8)
            pr = self.prompts.prompt("ui_confirm", label, size=18)
            screen.blit(pr, (btn.centerx - pr.get_width() // 2, btn.centery - pr.get_height() // 2))
        if self.msg_t > 0:
            t = ui.text(self.msg, 15, ACCENT, True)
            screen.blit(t, (right.x + 16, right.bottom - 100))
