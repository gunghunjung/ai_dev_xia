"""Grid inventory screen: drag & drop, split, merge, equip, hotbar, drop, container transfer.

Works identically with mouse (drag/click/right-click), keyboard (arrows + keys) and gamepad
(cursor + A/X/Y/B). Every action is also reachable through the context menu.
"""
import pygame

from engine import placeholder_art as art
from ui.widgets import panel, slot, bar, NavRepeater, TEXT, DIM, ACCENT, BAD, GOOD, RARITY

CELL = 50


class InventoryUI:
    def __init__(self, ui, prompts, assets):
        self.ui = ui
        self.prompts = prompts
        self.assets = assets
        self.session = None
        self.container = None       # world object
        self.cursor = ["player", 0, 0]
        self.held = None            # {"inv": GridInventory, "uid": str}
        self.context = None
        self.nav = NavRepeater()
        self.rects = {}
        self.drag_from = None
        self.mouse_cell = None
        self.flash_msg = ""
        self.flash_t = 0.0

    def open(self, session, container=None):
        self.session = session
        self.container = container
        self.cursor = ["player", 0, 0]
        self.held = None
        self.context = None

    def close(self):
        self.held = None
        self.context = None
        if self.session is not None:
            self.session.sync_containers()

    # ------------------------------------------------------------ helpers
    def inv(self, name):
        s = self.session
        if name == "player":
            return s.player.inventory
        if name == "container" and self.container is not None:
            return s.container_inventory(self.container)
        return None

    def grid_names(self):
        return ["equip", "player"] + (["container"] if self.container is not None else [])

    def item_icon(self, item_id: str, w: int, h: int):
        idef = self.session.db.item(item_id)
        return self.assets.image(f"ui/icons/{item_id}_grid{w}x{h}", lambda: art.make_icon(idef.get("icon", {}), w, h, CELL))

    def stack_at_cursor(self):
        name, x, y = self.cursor
        if name == "equip":
            p = self.session.player
            return (p.equipped_stack() if x == 0 else p.body), None
        inv = self.inv(name)
        return (inv.stack_at(x, y) if inv else None), inv

    def _say(self, text):
        self.flash_msg = text
        self.flash_t = 2.0

    # ------------------------------------------------------------ actions
    def pick(self):
        st, inv = self.stack_at_cursor()
        if st is None or inv is None:
            return
        self.held = {"inv": inv, "uid": st.uid}
        self.session.audio.play("ui_move", 0.5)

    def place(self):
        if not self.held:
            return
        name, x, y = self.cursor
        if name == "equip":
            src = self.held["inv"]
            st = src.stacks.get(self.held["uid"])
            if st is not None and src is self.session.player.inventory:
                self.session.equip(st.uid)
            self.held = None
            return
        dst = self.inv(name)
        src = self.held["inv"]
        st = src.stacks.get(self.held["uid"])
        if st is None or dst is None:
            self.held = None
            return
        target = dst.stack_at(x, y)
        ok = False
        if target is not None and target is not st and target.item_id == st.item_id and self.session.db.max_stack(st.item_id) > 1:
            cap = self.session.db.max_stack(st.item_id)
            moved = min(cap - target.count, st.count)
            if moved > 0:
                target.count += moved
                st.count -= moved
                if st.count <= 0:
                    src.remove_uid(st.uid)
                src._touch()
                dst._touch()
                ok = True
        elif src is dst:
            ok = dst.move(st.uid, x, y)
        elif dst.can_place(st.item_id, x, y):
            src.remove_uid(st.uid)
            ok = dst.place(st, x, y)
            self._after_transfer(src, st)
        if ok:
            self.held = None
            self.session.audio.play("ui_confirm", 0.4)
        else:
            self.session.audio.play("dry_fire", 0.3)

    def _after_transfer(self, src, st):
        p = self.session.player
        if src is p.inventory:
            if p.equipped_uid == st.uid:
                p.equipped_uid = None
            p.clean_hotbar()

    def quick_transfer(self, st, inv):
        if self.container is None or st is None or inv is None:
            return
        p = self.session.player
        dst = self.inv("container") if inv is p.inventory else p.inventory
        inv.remove_uid(st.uid)
        if dst.insert_stack(st):
            self._after_transfer(inv, st)
            self.session.audio.play("ui_move", 0.5)
        else:
            inv.insert_stack(st)
            self._say("공간이 부족하다")

    def split(self):
        st, inv = self.stack_at_cursor()
        if st is None or inv is None or st.count < 2:
            return
        if inv.split(st.uid, st.count // 2) is None:
            self._say("나눌 공간이 없다")
        else:
            self.session.audio.play("ui_move", 0.5)

    def assign_hotbar(self, uid, slot_i):
        p = self.session.player
        for i, u in enumerate(p.hotbar):
            if u == uid:
                p.hotbar[i] = None
        p.hotbar[slot_i] = uid
        self._say(f"핫바 {slot_i + 1}번에 지정")

    def open_context(self):
        st, inv = self.stack_at_cursor()
        if st is None:
            return
        s = self.session
        p = s.player
        idef = s.db.item(st.item_id)
        opts = []
        if self.cursor[0] == "equip":
            opts.append(("unequip", "해제"))
        elif inv is p.inventory:
            if idef.get("place"):
                opts.append(("use", "설치"))
            elif idef.get("equip") == "hand":
                opts.append(("use", "해제" if p.equipped_uid == st.uid else "장착"))
            elif idef.get("equip") == "body":
                opts.append(("use", "착용"))
            elif idef.get("use"):
                opts.append(("use", "읽기" if idef["use"].get("lore") else "사용"))
            opts.append(("hotbar", "핫바 지정"))
            if st.count > 1:
                opts.append(("split", "나누기"))
            if self.container is not None:
                opts.append(("transfer", "상자로 옮기기"))
            opts.append(("drop", "버리기"))
        else:
            opts.append(("transfer", "가져오기"))
            if st.count > 1:
                opts.append(("split", "나누기"))
        self.context = {"uid": st.uid, "inv": inv, "options": opts, "index": 0, "slot": 0}

    def run_context(self):
        c = self.context
        key = c["options"][c["index"]][0]
        s = self.session
        p = s.player
        st = c["inv"].stacks.get(c["uid"]) if c["inv"] else None
        self.context = None
        if key == "unequip":
            if self.cursor[1] == 0:
                s.equip(None)
            else:
                s.unequip_body()
            return
        if st is None:
            return
        if key == "use":
            s.use_stack(st.uid)
            if p.placing is not None:
                return "close"
        elif key == "hotbar":
            self.assign_hotbar(st.uid, c["slot"])
        elif key == "split":
            self.split()
        elif key == "transfer":
            self.quick_transfer(st, c["inv"])
        elif key == "drop":
            s.drop_stack(st.uid)
            s.audio.play("ui_move", 0.5)

    # ------------------------------------------------------------ input
    def _cell_from_mouse(self, pos):
        for name in self.grid_names():
            r = self.rects.get(name)
            if r is None or not r.collidepoint(pos):
                continue
            if name == "equip":
                for i, er in enumerate(self.rects.get("equip_slots", [])):
                    if er.collidepoint(pos):
                        return [name, i, 0]
                return None
            return [name, (pos[0] - r.x) // CELL, (pos[1] - r.y) // CELL]
        return None

    def handle(self, inp, events, dt):
        if self.session is None:
            return None
        self.flash_t = max(0.0, self.flash_t - dt)
        if self.context is not None:
            return self._handle_context(inp, events, dt)
        mods = pygame.key.get_mods()
        for e in events:
            if e.type == pygame.MOUSEMOTION:
                cell = self._cell_from_mouse(e.pos)
                self.mouse_cell = cell
                if cell:
                    self.cursor = cell
            elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                cell = self._cell_from_mouse(e.pos)
                if cell is None:
                    if self.held and not self.rects["frame"].collidepoint(e.pos):
                        st = self.held["inv"].remove_uid(self.held["uid"])
                        if st:
                            self._after_transfer(self.held["inv"], st)
                            self.session.drop_item(st)
                        self.held = None
                    continue
                self.cursor = cell
                if mods & pygame.KMOD_SHIFT:
                    st, inv = self.stack_at_cursor()
                    self.quick_transfer(st, inv)
                elif self.held:
                    self.place()
                else:
                    self.pick()
                    self.drag_from = list(cell)
            elif e.type == pygame.MOUSEBUTTONUP and e.button == 1:
                cell = self._cell_from_mouse(e.pos)
                if self.held and self.drag_from and cell and cell != self.drag_from:
                    self.cursor = cell
                    self.place()
                self.drag_from = None
            elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 3:
                cell = self._cell_from_mouse(e.pos)
                if cell:
                    self.cursor = cell
                    self.held = None
                    self.open_context()
            elif e.type == pygame.KEYDOWN and pygame.K_1 <= e.key <= pygame.K_5:
                st, inv = self.stack_at_cursor()
                if st is not None and inv is self.session.player.inventory:
                    self.assign_hotbar(st.uid, e.key - pygame.K_1)
        dx, dy = self.nav.step(inp, dt)
        if dx or dy:
            self._move_cursor(dx, dy)
        if inp.pressed("ui_confirm"):
            if self.held:
                self.place()
            else:
                self.pick()
        if inp.pressed("ui_split"):
            self.split()
        if inp.pressed("ui_menu"):
            self.held = None
            self.open_context()
        if inp.pressed("ui_back"):
            if self.held:
                self.held = None
            else:
                return "close"
        return None

    def _move_cursor(self, dx, dy):
        name, x, y = self.cursor
        names = self.grid_names()
        if name == "equip":
            if dy:
                x = max(0, min(1, x + dy))
            if dx > 0:
                name, x, y = "player", 0, 0
        else:
            inv = self.inv(name)
            nx, ny = x + dx, y + dy
            if nx < 0:
                idx = names.index(name)
                if idx > 0:
                    prev = names[idx - 1]
                    if prev == "equip":
                        name, x, y = "equip", 0, 0
                    else:
                        name, x, y = prev, self.inv(prev).w - 1, min(y, self.inv(prev).h - 1)
                    self.cursor = [name, x, y]
                    self.session.audio.play("ui_move", 0.3)
                    return
                nx = 0
            if nx >= inv.w:
                idx = names.index(name)
                if idx + 1 < len(names):
                    nxt = names[idx + 1]
                    self.cursor = [nxt, 0, min(y, self.inv(nxt).h - 1)]
                    self.session.audio.play("ui_move", 0.3)
                    return
                nx = inv.w - 1
            x, y = nx, max(0, min(inv.h - 1, ny))
        self.cursor = [name, x, y]
        self.session.audio.play("ui_move", 0.3)

    def _handle_context(self, inp, events, dt):
        c = self.context
        for e in events:
            if e.type == pygame.MOUSEMOTION:
                for i, r in enumerate(self.rects.get("ctx", [])):
                    if r.collidepoint(e.pos):
                        c["index"] = i
            elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                for i, r in enumerate(self.rects.get("ctx", [])):
                    if r.collidepoint(e.pos):
                        c["index"] = i
                        if c["options"][i][0] == "hotbar":
                            rel = (e.pos[0] - r.x - 90) // 26
                            if 0 <= rel < 5:
                                c["slot"] = rel
                        return self.run_context()
                self.context = None
                return None
            elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 3:
                self.context = None
                return None
        dx, dy = self.nav.step(inp, dt)
        if dy:
            c["index"] = (c["index"] + dy) % len(c["options"])
        if dx and c["options"][c["index"]][0] == "hotbar":
            c["slot"] = (c["slot"] + dx) % 5
        if inp.pressed("ui_confirm"):
            return self.run_context()
        if inp.pressed("ui_back") or inp.pressed("ui_menu"):
            self.context = None
        return None

    # ------------------------------------------------------------ drawing
    def draw(self, screen, area: pygame.Rect, inp):
        s = self.session
        p = s.player
        db = s.db
        ui = self.ui
        inv = p.inventory
        gw, gh = inv.w * CELL, inv.h * CELL
        cont = self.inv("container")
        total_w = 170 + gw + 40 + (cont.w * CELL + 30 if cont else 300)
        x0 = area.centerx - total_w // 2
        y0 = area.y + 30
        frame = pygame.Rect(x0 - 20, y0 - 20, total_w + 40, gh + 170)
        self.rects["frame"] = frame

        # Equipment column
        eq = pygame.Rect(x0, y0, 150, gh)
        panel(screen, eq, "장비", ui)
        self.rects["equip"] = eq
        slots = []
        for i, (label, st) in enumerate((("손", p.equipped_stack()), ("몸", p.body))):
            r = pygame.Rect(eq.x + 25, eq.y + 34 + i * 120, 100, 100)
            slots.append(r)
            sel = self.cursor[0] == "equip" and self.cursor[1] == i
            slot(screen, r, selected=sel)
            screen.blit(ui.text(label, 13, DIM), (r.x, r.y - 18))
            if st is not None:
                icon = self.assets.image(f"ui/icons/{st.item_id}_eq", lambda st=st: art.make_icon(db.item(st.item_id).get("icon", {}), 1, 1, 90))
                screen.blit(icon, (r.x + 5, r.y + 5))
        self.rects["equip_slots"] = slots
        wy = eq.y + 280
        w = p.carry_weight()
        cap = s.carry_capacity()
        screen.blit(ui.text(f"무게 {w:.1f}/{cap:.0f}kg", 13, BAD if w > cap else TEXT, True), (eq.x + 12, wy))
        bar(screen, (eq.x + 12, wy + 20, 126, 10), w / cap, BAD if w > cap else ACCENT)

        # Player grid
        g = pygame.Rect(eq.right + 20, y0, gw, gh)
        self._draw_grid(screen, g, inv, "player", "배낭")
        self.rects["player"] = g

        # Right side: container or details
        if cont:
            c = pygame.Rect(g.right + 30, y0, cont.w * CELL, cont.h * CELL)
            self._draw_grid(screen, c, cont, "container", self.container.name)
            self.rects["container"] = c
            det = pygame.Rect(c.x, c.bottom + 30, c.width, gh - c.height - 30)
            if det.height > 80:
                self._draw_details(screen, det)
        else:
            self.rects.pop("container", None)
            det = pygame.Rect(g.right + 30, y0, 300, gh)
            self._draw_details(screen, det)

        # Hotbar preview
        hb = pygame.Rect(g.x, g.bottom + 30, gw, 60)
        for i, uid in enumerate(p.hotbar):
            r = pygame.Rect(hb.x + i * 58, hb.y, 52, 52)
            slot(screen, r, selected=uid is not None and uid == p.equipped_uid)
            st = inv.stacks.get(uid) if uid else None
            if st:
                icon = self.assets.image(f"ui/icons/{st.item_id}_42", lambda st=st: art.make_icon(db.item(st.item_id).get("icon", {}), 1, 1, 42))
                screen.blit(icon, (r.x + 5, r.y + 5))
            screen.blit(ui.text(str(i + 1), 12, ACCENT, True), (r.x + 3, r.y + 1))
        hint = "1~5: 핫바 지정 · Shift+클릭: 빠른 이동 · 우클릭: 메뉴" if inp.last_device == "kbm" else ""
        if hint:
            screen.blit(ui.text(hint, 12, DIM), (hb.x + 300, hb.y + 18))

        # Held item follows cursor
        if self.held:
            st = self.held["inv"].stacks.get(self.held["uid"])
            if st:
                w_, h_ = db.size(st.item_id)
                icon = self.item_icon(st.item_id, w_, h_)
                if inp.last_device == "kbm":
                    mx, my = inp.mouse_pos
                    pos = (mx - CELL // 2, my - CELL // 2)
                else:
                    r = self._cursor_rect()
                    pos = (r.x + 8, r.y - 8) if r else (0, 0)
                ghost = icon.copy()
                ghost.set_alpha(210)
                screen.blit(ghost, pos)

        if self.context is not None:
            self._draw_context(screen)
        if self.flash_t > 0:
            t = ui.text(self.flash_msg, 15, ACCENT, True)
            screen.blit(t, (frame.centerx - t.get_width() // 2, frame.bottom - 32))

    def _cursor_rect(self):
        name, x, y = self.cursor
        if name == "equip":
            sl = self.rects.get("equip_slots")
            return sl[x] if sl else None
        r = self.rects.get(name)
        return pygame.Rect(r.x + x * CELL, r.y + y * CELL, CELL, CELL) if r else None

    def _draw_grid(self, screen, rect, inv, name, title):
        s = self.session
        db = s.db
        panel(screen, rect.inflate(16, 16), title, self.ui)
        for gx in range(inv.w):
            for gy in range(inv.h):
                r = pygame.Rect(rect.x + gx * CELL + 2, rect.y + gy * CELL + 2, CELL - 4, CELL - 4)
                pygame.draw.rect(screen, (44, 38, 32), r, border_radius=4)
                pygame.draw.rect(screen, (28, 24, 20), r, 1, border_radius=4)
        p = s.player
        for st, (gx, gy) in inv.items():
            w, h = db.size(st.item_id)
            r = pygame.Rect(rect.x + gx * CELL + 2, rect.y + gy * CELL + 2, w * CELL - 4, h * CELL - 4)
            idef = db.item(st.item_id)
            rc = RARITY.get(idef.get("rarity", "common"), (150, 140, 120))
            pygame.draw.rect(screen, tuple(int(c * 0.35) for c in rc), r, border_radius=5)
            pygame.draw.rect(screen, tuple(int(c * 0.7) for c in rc), r, 1, border_radius=5)
            icon = self.item_icon(st.item_id, w, h)
            if self.held and self.held["uid"] == st.uid:
                icon = icon.copy()
                icon.set_alpha(80)
            screen.blit(icon, (r.x - 2, r.y - 2))
            if st.count > 1:
                t = self.ui.text(str(st.count), 13, TEXT, True)
                screen.blit(t, (r.right - t.get_width() - 3, r.bottom - 17))
            maxd = idef.get("durability")
            if st.durability is not None and maxd:
                frac = st.durability / maxd
                pygame.draw.rect(screen, (20, 16, 14), (r.x + 3, r.bottom - 6, r.width - 6, 4))
                pygame.draw.rect(screen, GOOD if frac > 0.5 else ACCENT if frac > 0.2 else BAD, (r.x + 3, r.bottom - 6, (r.width - 6) * frac, 4))
            if st.uid == p.equipped_uid:
                pygame.draw.rect(screen, ACCENT, r, 2, border_radius=5)
                screen.blit(self.ui.text("E", 12, ACCENT, True), (r.x + 3, r.y + 1))
            if st.uid in p.hotbar:
                screen.blit(self.ui.text(str(p.hotbar.index(st.uid) + 1), 11, (200, 220, 255), True), (r.right - 10, r.y + 1))
        if self.cursor[0] == name:
            cx, cy = self.cursor[1], self.cursor[2]
            st = inv.stack_at(cx, cy)
            if st is not None and not self.held:
                gx, gy = inv.pos[st.uid]
                w, h = db.size(st.item_id)
                r = pygame.Rect(rect.x + gx * CELL, rect.y + gy * CELL, w * CELL, h * CELL)
            else:
                w, h = (db.size(self.held_stack().item_id) if self.held and self.held_stack() else (1, 1))
                r = pygame.Rect(rect.x + cx * CELL, rect.y + cy * CELL, w * CELL, h * CELL)
                if self.held and self.held_stack():
                    ok = inv.can_place(self.held_stack().item_id, cx, cy, self.held["uid"]) or (st is not None and st.item_id == self.held_stack().item_id)
                    pygame.draw.rect(screen, GOOD if ok else BAD, r, 3, border_radius=6)
            pygame.draw.rect(screen, ACCENT, r, 2, border_radius=6)

    def held_stack(self):
        if not self.held:
            return None
        return self.held["inv"].stacks.get(self.held["uid"])

    def _draw_details(self, screen, rect):
        s = self.session
        panel(screen, rect, "정보", self.ui)
        st, _ = self.stack_at_cursor()
        if st is None:
            screen.blit(self.ui.text("아이템을 선택하세요", 14, DIM), (rect.x + 16, rect.y + 24))
            return
        idef = s.db.item(st.item_id)
        y = rect.y + 22
        rc = RARITY.get(idef.get("rarity", "common"), TEXT)
        screen.blit(self.ui.text(idef["name"], 20, rc, True), (rect.x + 16, y))
        y += 32
        for line in self.ui.wrap(idef.get("desc", ""), 14, rect.width - 32):
            screen.blit(self.ui.text(line, 14, TEXT), (rect.x + 16, y))
            y += 20
        y += 6
        info = [f"무게 {idef.get('weight', 0) * st.count:.2f}kg", f"수량 {st.count}/{idef.get('max_stack', 1)}"]
        if st.durability is not None and idef.get("durability"):
            info.append(f"내구도 {int(st.durability)}/{idef['durability']}")
        w = s.db.weapon(idef["weapon"]) if idef.get("weapon") else None
        if w:
            info.append(f"피해 {w['damage']}" + (f" x{w['pellets']}" if w.get("pellets", 1) > 1 else "") + f" · 사거리 {w['range']}")
            if w["type"] == "ranged":
                info.append(f"장탄 {st.ammo}/{w['mag_size']} · {s.db.name(w['ammo'])}")
        tool = idef.get("tool")
        if tool and "dig_power" in tool:
            info.append(f"굴착력 {tool['dig_power']} · 최대 깊이 {tool['max_depth']}")
        use = idef.get("use", {})
        eff = []
        for k, label in (("hunger", "허기"), ("thirst", "갈증"), ("health", "체력"), ("infection", "감염")):
            if k in use:
                eff.append(f"{label} {use[k]:+d}")
        if use.get("stop_bleeding"):
            eff.append("지혈")
        if eff:
            info.append(" · ".join(eff))
        if idef.get("contamination"):
            info.append("오염됨 — 감염 위험")
        for line in info:
            screen.blit(self.ui.text(line, 14, DIM, True), (rect.x + 16, y))
            y += 20
        y = rect.bottom - 64
        rows = [("ui_confirm", "집기/놓기"), ("ui_menu", "메뉴")]
        if st.count > 1:
            rows.append(("ui_split", "나누기"))
        r1 = self.prompts.row(rows, 12)
        screen.blit(r1, (rect.x + 14, y))
        screen.blit(self.prompts.row([("ui_back", "닫기")], 12), (rect.x + 14, y + 28))

    def _draw_context(self, screen):
        c = self.context
        anchor = self._cursor_rect() or pygame.Rect(400, 300, 10, 10)
        w = 250
        h = 36 * len(c["options"]) + 12
        r = pygame.Rect(anchor.right + 6, anchor.y, w, h)
        sw, sh = screen.get_size()
        if r.right > sw - 10:
            r.right = anchor.x - 6
        if r.bottom > sh - 10:
            r.bottom = sh - 10
        panel(screen, r, alpha=250)
        rects = []
        for i, (key, label) in enumerate(c["options"]):
            row = pygame.Rect(r.x + 6, r.y + 6 + i * 36, w - 12, 32)
            rects.append(row)
            if i == c["index"]:
                pygame.draw.rect(screen, (90, 70, 46), row, border_radius=6)
            screen.blit(self.ui.text(label, 16, TEXT, True), (row.x + 10, row.y + 6))
            if key == "hotbar":
                for k in range(5):
                    b = pygame.Rect(row.x + 90 + k * 26, row.y + 4, 22, 24)
                    pygame.draw.rect(screen, ACCENT if k == c["slot"] else (60, 50, 42), b, border_radius=4)
                    t = self.ui.text(str(k + 1), 13, (20, 16, 14) if k == c["slot"] else TEXT, True, False)
                    screen.blit(t, (b.centerx - t.get_width() // 2, b.y + 3))
        self.rects["ctx"] = rects
