"""In-game HUD: vitals, survival meters, hotbar, weapon panel, crosshair, prompts, warnings."""
import math

import pygame

from engine import placeholder_art as art
from ui.widgets import panel, bar, slot, TEXT, DIM, ACCENT, BAD, GOOD

HOTBAR_PAD_GLYPH = ["slot_1", "slot_2", "slot_3", "slot_4", "slot_5"]


class HUD:
    def __init__(self, ui, prompts, assets):
        self.ui = ui
        self.prompts = prompts
        self.assets = assets
        self.time = 0.0
        self.toast = None
        self.toast_t = 0.0
        self._vignette = {}

    def icon(self, db, item_id, size):
        idef = db.item(item_id)
        return self.assets.image(f"ui/icons/{item_id}_{size}", lambda: art.make_icon(idef.get("icon", {}), 1, 1, size))

    def show_toast(self, text: str):
        self.toast = text
        self.toast_t = 3.0

    # ------------------------------------------------------------------
    def draw(self, screen, session, camera, inp, dt, show_debug=False, profiler=None, menu_open=False):
        self.time += dt
        W, H = screen.get_size()
        p = session.player
        self._vignette_fx(screen, session)
        self._vitals(screen, session)
        self._survival(screen, session, H)
        self._hotbar(screen, session, W, H, inp)
        self._weapon(screen, session, W, H)
        self._clock(screen, session, W)
        self._messages(screen, session, W)
        if not menu_open and not p.dead:
            self._threats(screen, session, camera, W, H)
            self._interaction_prompt(screen, session, camera)
            self._crosshair(screen, session, camera, inp)
        if self.toast_t > 0:
            self.toast_t -= dt
            t = self.ui.text(self.toast, 18, ACCENT, True)
            r = pygame.Rect(0, 0, t.get_width() + 30, 38)
            r.midtop = (W // 2, 70)
            panel(screen, r)
            screen.blit(t, (r.x + 15, r.y + 8))
        if session.lore_popup:
            self._lore(screen, session, W, H)
        if show_debug and profiler is not None:
            self._debug(screen, session, profiler)

    # ------------------------------------------------------------------
    def _vignette_fx(self, screen, session):
        p = session.player
        k = max(session.damage_flash * 0.8, (1 - p.hp / p.max_hp) * 0.9 if p.hp < 35 else 0)
        if p.hp < 35:
            k = max(k, (0.35 + 0.15 * math.sin(self.time * 5)) * (1 - p.hp / 35))
        if k <= 0.02:
            return
        W, H = screen.get_size()
        v = self._vignette.get((W, H))
        if v is None:
            v = pygame.Surface((W, H), pygame.SRCALPHA)
            for i in range(30):
                t = i / 30
                a = int(200 * (1 - t) ** 2)
                pygame.draw.rect(v, (170, 20, 16, a), (W * t * 0.25, H * t * 0.25, W * (1 - t * 0.5), H * (1 - t * 0.5)), int(max(2, W * 0.01)))
            self._vignette[(W, H)] = v
        v2 = v.copy()
        v2.set_alpha(int(255 * min(1.0, k)))
        screen.blit(v2, (0, 0))

    def _vitals(self, screen, session):
        p = session.player
        st = p.stats
        r = pygame.Rect(16, 16, 300, 74)
        panel(screen, r, alpha=200)
        screen.blit(self.ui.text("체력", 14, TEXT, True), (r.x + 12, r.y + 10))
        hp_col = (210, 64, 56) if p.hp > 30 else (255, 60, 50)
        bar(screen, (r.x + 60, r.y + 10, 190, 20), p.hp / p.max_hp, hp_col, segments=10)
        screen.blit(self.ui.text(f"{int(p.hp)}", 14, TEXT, True), (r.x + 258, r.y + 11))
        screen.blit(self.ui.text("기력", 14, TEXT, True), (r.x + 12, r.y + 40))
        bar(screen, (r.x + 60, r.y + 42, 190 * st.max_stamina / 100, 14), st.stamina / max(1, st.max_stamina), (226, 190, 70))
        if st.max_stamina < 100:
            pygame.draw.rect(screen, (60, 50, 44), (r.x + 60 + 190 * st.max_stamina / 100, r.y + 44, 190 * (1 - st.max_stamina / 100), 10))
        screen.blit(self.ui.text(f"{int(st.stamina)}", 13, DIM, True), (r.x + 258, r.y + 40))
        x = r.right + 12
        for name, col in session.survival.warnings:
            t = self.ui.text(name, 14, (20, 16, 14), True, False)
            chip = pygame.Rect(x, r.y + 6, t.get_width() + 16, 24)
            pulse = 0.85 + 0.15 * math.sin(self.time * 6)
            pygame.draw.rect(screen, tuple(int(c * pulse) for c in col), chip, border_radius=12)
            pygame.draw.rect(screen, (20, 16, 14), chip, 2, border_radius=12)
            screen.blit(t, (chip.x + 8, chip.y + 3))
            x = chip.right + 6

    def _meter(self, screen, x, y, label, frac, color, value_text):
        screen.blit(self.ui.text(label, 13, TEXT, True), (x, y))
        bar(screen, (x + 46, y + 2, 120, 14), frac, color)
        screen.blit(self.ui.text(value_text, 13, DIM, True), (x + 172, y))

    def _survival(self, screen, session, H):
        st = session.player.stats
        r = pygame.Rect(16, H - 150, 244, 134)
        panel(screen, r, alpha=200)
        x, y = r.x + 12, r.y + 12
        self._meter(screen, x, y, "허기", st.hunger / 100, (214, 150, 70), f"{int(st.hunger)}")
        self._meter(screen, x, y + 24, "갈증", st.thirst / 100, (84, 160, 226), f"{int(st.thirst)}")
        self._meter(screen, x, y + 48, "피로", 1 - st.fatigue / 100, (170, 150, 210), f"{int(100 - st.fatigue)}")
        t = st.body_temp
        tcol = (120, 180, 255) if t < 35.8 else (255, 120, 80) if t > 38.0 else GOOD
        screen.blit(self.ui.text("체온", 13, TEXT, True), (x, y + 74))
        frac = (t - 33.0) / (41.0 - 33.0)
        bar(screen, (x + 46, y + 76, 120, 14), frac, tcol)
        mid = x + 46 + 120 * (37 - 33) / 8
        pygame.draw.line(screen, (240, 240, 240), (mid, y + 76), (mid, y + 89), 1)
        screen.blit(self.ui.text(f"{t:.1f}°", 13, tcol, True), (x + 172, y + 74))
        amb = session.survival.feels_temp
        fire = " (모닥불)" if session.survival.near_fire else ""
        screen.blit(self.ui.text(f"체감 {amb:.0f}°C{fire}", 12, DIM), (x, y + 96))
        if st.infection > 0:
            screen.blit(self.ui.text(f"감염 {int(st.infection)}%", 12, (150, 210, 70), True), (x + 120, y + 96))

    def _hotbar(self, screen, session, W, H, inp):
        p = session.player
        n = len(p.hotbar)
        size = 54
        gap = 8
        total = n * size + (n - 1) * gap
        x0 = W // 2 - total // 2
        y = H - size - 18
        back = pygame.Rect(x0 - 10, y - 10, total + 20, size + 20)
        panel(screen, back, alpha=170)
        for i, uid in enumerate(p.hotbar):
            r = pygame.Rect(x0 + i * (size + gap), y, size, size)
            st = p.inventory.stacks.get(uid) if uid else None
            equipped = st is not None and uid == p.equipped_uid
            slot(screen, r, selected=equipped)
            if st is not None:
                icon = self.icon(session.db, st.item_id, 42)
                screen.blit(icon, (r.x + 6, r.y + 6))
                total_count = p.inventory.count(st.item_id) if session.db.max_stack(st.item_id) > 1 else 0
                if total_count > 1:
                    t = self.ui.text(str(total_count), 13, TEXT, True)
                    screen.blit(t, (r.right - t.get_width() - 4, r.bottom - 18))
                maxd = session.db.item(st.item_id).get("durability")
                if st.durability is not None and maxd:
                    frac = st.durability / maxd
                    col = GOOD if frac > 0.5 else ACCENT if frac > 0.2 else BAD
                    pygame.draw.rect(screen, (20, 16, 14), (r.x + 5, r.bottom - 7, size - 10, 4))
                    pygame.draw.rect(screen, col, (r.x + 5, r.bottom - 7, (size - 10) * frac, 4))
            if inp.last_device == "pad" and inp.has_pad and i < 4:
                g = self.prompts.glyph(HOTBAR_PAD_GLYPH[i], "pad")
                g = pygame.transform.smoothscale(g, (18, 18))
                screen.blit(g, (r.x - 4, r.y - 8))
            else:
                t = self.ui.text(str(i + 1), 13, ACCENT, True)
                screen.blit(t, (r.x + 4, r.y + 2))
        if p.placing is not None:
            t = self.prompts.row([("attack", "설치"), ("aim", "취소")])
            screen.blit(t, (W // 2 - t.get_width() // 2, back.y - 32))

    def _weapon(self, screen, session, W, H):
        p = session.player
        r = pygame.Rect(W - 276, H - 112, 260, 96)
        panel(screen, r, alpha=200)
        st = p.equipped_stack()
        weapon = p.weapon()
        if st is None:
            screen.blit(self.ui.text("맨손", 20, TEXT, True), (r.x + 16, r.y + 14))
            screen.blit(self.ui.text("근접 · 약함", 13, DIM), (r.x + 16, r.y + 46))
            return
        idef = session.db.item(st.item_id)
        icon = self.assets.image(f"ui/icons/{st.item_id}_big", lambda: art.make_icon(idef.get("icon", {}), 1, 1, 64))
        screen.blit(icon, (r.x + 10, r.y + 16))
        screen.blit(self.ui.text(idef["name"], 18, TEXT, True), (r.x + 82, r.y + 10))
        if weapon["type"] == "ranged":
            reserve = p.inventory.count(weapon["ammo"])
            col = BAD if st.ammo == 0 else TEXT
            big = self.ui.text(f"{st.ammo}", 30, col, True)
            screen.blit(big, (r.x + 82, r.y + 36))
            screen.blit(self.ui.text(f"/ {weapon['mag_size']}   예비 {reserve}", 14, DIM, True), (r.x + 86 + big.get_width(), r.y + 50))
            if st.ammo == 0 and reserve > 0 and p.reload_t <= 0:
                t = self.prompts.prompt("reload" if self.prompts.inp.last_device == "kbm" else "tool", "재장전", BAD, 13)
                screen.blit(t, (r.x + 82, r.y - 24))
        else:
            tool = idef.get("tool")
            kind = "도구 · 근접" if tool else "근접 무기"
            screen.blit(self.ui.text(kind, 13, DIM), (r.x + 82, r.y + 40))
        maxd = idef.get("durability")
        if st.durability is not None and maxd:
            frac = st.durability / maxd
            bar(screen, (r.x + 82, r.y + 72, 160, 12), frac, GOOD if frac > 0.5 else ACCENT if frac > 0.2 else BAD)

    def _clock(self, screen, session, W):
        r = pygame.Rect(W - 236, 16, 220, 82)
        panel(screen, r, alpha=200)
        night = session.is_night()
        screen.blit(self.ui.text(session.clock.clock_text(), 18, (170, 180, 255) if night else ACCENT, True), (r.x + 14, r.y + 10))
        region = session.world.region_at(session.player.x, session.player.y)
        rname = session.terrain_db.regions.get(region, {}).get("name", region)
        screen.blit(self.ui.text(f"{rname} · {session.weather.name}", 14, TEXT), (r.x + 14, r.y + 36))
        screen.blit(self.ui.text("소음", 12, DIM), (r.x + 14, r.y + 58))
        n = session.noise.player_noise
        bar(screen, (r.x + 50, r.y + 60, 150, 10), n, (230, 90, 70) if n > 0.6 else (220, 180, 90), border=False)

    def _messages(self, screen, session, W):
        y = 110
        for m in session.messages[-5:]:
            a = min(1.0, m["t"] / 0.6)
            t = self.ui.text(m["text"], 20 if m["big"] else 16, m["color"], True)
            if a < 1:
                t = t.copy()
                t.set_alpha(int(255 * a))
            screen.blit(t, (W // 2 - t.get_width() // 2, y))
            y += t.get_height() + 4

    def _interaction_prompt(self, screen, session, camera):
        p = session.player
        if p.action is not None or p.placing is not None:
            return
        obj = session.find_interactable()
        if obj is None:
            return
        sx, sy = camera.world_to_screen(obj.x, obj.y)
        label = session.interaction_label(obj)
        pr = self.prompts.prompt("interact", label, size=15)
        r = pygame.Rect(0, 0, pr.get_width() + 16, pr.get_height() + 10)
        r.midbottom = (sx, sy - 40 - obj.radius * 40)
        panel(screen, r, alpha=210)
        screen.blit(pr, (r.x + 8, r.y + 5))

    def _crosshair(self, screen, session, camera, inp):
        p = session.player
        if inp.last_device == "kbm" or not inp.has_pad:
            cx, cy = inp.mouse_pos
        else:
            d = 3.5
            cx, cy = camera.world_to_screen(p.x + math.cos(p.aim_angle) * d, p.y + math.sin(p.aim_angle) * d, 20)
        weapon = p.weapon()
        col = (255, 240, 200)
        if weapon["type"] == "ranged":
            spread = (weapon["aim_spread"] if p.aiming else weapon["spread"]) + p.spread_bloom
            r = int(8 + spread * 2.2)
            for a in (0, math.pi / 2, math.pi, math.pi * 1.5):
                x1, y1 = cx + math.cos(a) * r, cy + math.sin(a) * r
                x2, y2 = cx + math.cos(a) * (r + 7), cy + math.sin(a) * (r + 7)
                pygame.draw.line(screen, (20, 16, 14), (x1, y1), (x2, y2), 4)
                pygame.draw.line(screen, col, (x1, y1), (x2, y2), 2)
            pygame.draw.circle(screen, col, (int(cx), int(cy)), 2)
        else:
            pygame.draw.circle(screen, (20, 16, 14), (int(cx), int(cy)), 8, 4)
            pygame.draw.circle(screen, col, (int(cx), int(cy)), 8, 2)
        # Aim-direction tick from the character (helps on pads).
        if inp.last_device == "pad":
            px, py = camera.world_to_screen(p.x, p.y, 20)
            pygame.draw.line(screen, (255, 230, 160), (px + math.cos(p.aim_screen_angle) * 30, py + math.sin(p.aim_screen_angle) * 30),
                             (px + math.cos(p.aim_screen_angle) * 44, py + math.sin(p.aim_screen_angle) * 44), 3)

    def _threats(self, screen, session, camera, W, H):
        p = session.player
        chasing = [e for e in session.enemies if not e.dead and e.state in ("chase", "attack") and e.defn.get("kind") == "hostile"]
        if chasing:
            t = self.ui.text(f"! 추적당하는 중 ({len(chasing)})", 16, BAD, True)
            screen.blit(t, (W // 2 - t.get_width() // 2, 76))
        for e in chasing:
            sx, sy = camera.world_to_screen(e.x, e.y, 20)
            if 0 <= sx <= W and 0 <= sy <= H:
                continue
            cx, cy = W / 2, H / 2
            ang = math.atan2(sy - cy, sx - cx)
            ex = cx + math.cos(ang) * (min(W, H) / 2 - 40)
            ey = cy + math.sin(ang) * (min(W, H) / 2 - 40)
            pts = [(ex + math.cos(ang) * 14, ey + math.sin(ang) * 14),
                   (ex + math.cos(ang + 2.5) * 10, ey + math.sin(ang + 2.5) * 10),
                   (ex + math.cos(ang - 2.5) * 10, ey + math.sin(ang - 2.5) * 10)]
            pygame.draw.polygon(screen, (230, 70, 60), pts)
            pygame.draw.polygon(screen, (20, 16, 14), pts, 2)
        if session.last_damage_angle is not None and session.damage_flash > 0.1:
            from engine.renderer import _screen_angle
            a = _screen_angle(session.last_damage_angle + math.pi)
            px, py = camera.world_to_screen(p.x, p.y, 24)
            rect = pygame.Rect(0, 0, 120, 120)
            rect.center = (px, py)
            pygame.draw.arc(screen, (255, 60, 50), rect, -a - 0.5, -a + 0.5, 5)

    def _lore(self, screen, session, W, H):
        r = pygame.Rect(0, 0, 560, 260)
        r.center = (W // 2, H // 2)
        panel(screen, r, "낡은 일지", self.ui)
        y = r.y + 30
        for line in self.ui.wrap(session.lore_popup, 17, r.width - 50):
            screen.blit(self.ui.text(line, 17, (230, 220, 196)), (r.x + 25, y))
            y += 26
        t = self.prompts.prompt("ui_confirm", "닫기")
        screen.blit(t, (r.right - t.get_width() - 20, r.bottom - 36))

    def _debug(self, screen, session, prof):
        lines = [
            f"FPS {prof.fps():.0f}",
            f"update {prof.avg_ms('update'):.2f} ms   render {prof.avg_ms('render'):.2f} ms   ui {prof.avg_ms('ui'):.2f} ms",
            f"enemies {sum(1 for e in session.enemies if not e.dead)}/{len(session.enemies)}  particles {session.effects.particle_count()}",
            f"chunks {len(session.world.chunks)}  objects(near) {len(session.world.objects_near(session.player.x, session.player.y, 20))}",
            f"tiles {prof.counters.get('tiles', 0)}  sprites {prof.counters.get('sprites', 0)}  projectiles {len(session.projectiles.active())}",
            f"pos {session.player.x:.1f},{session.player.y:.1f}  seed {session.seed}",
        ]
        r = pygame.Rect(16, 100, 470, 18 * len(lines) + 16)
        s = pygame.Surface(r.size, pygame.SRCALPHA)
        s.fill((0, 0, 0, 160))
        screen.blit(s, r.topleft)
        for i, line in enumerate(lines):
            screen.blit(self.ui.text(line, 13, (180, 255, 180), False), (r.x + 8, r.y + 8 + i * 18))
