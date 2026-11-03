"""World rendering: ground tiles, holes, props, characters, effects, lighting and weather.

The renderer only *reads* session state; it never mutates gameplay data.
"""
import math
import random

import pygame

from config import TILE_W, TILE_H, HALF_W, HALF_H, DEPTH_PX, CHUNK_SIZE
from engine import placeholder_art as art
from engine import character_art as chars
from world.chunks import FLAG_CACHE, FLAG_CACHE_FOUND

FLAT = art.FLAT_ART
PICKUP_ICON = 26


class Renderer:
    def __init__(self, assets, camera):
        self.assets = assets
        self.camera = camera
        self._text_cache: dict = {}
        self._light_cache: dict = {}
        self._fog_cache: dict = {}
        self.rain = [[random.uniform(0, 1), random.uniform(0, 1), random.uniform(0.6, 1.0)] for _ in range(220)]
        self.dust = [[random.uniform(0, 1), random.uniform(0, 1), random.uniform(0.5, 1.0)] for _ in range(120)]
        self.light_map = None
        self.time = 0.0
        self.stats = {"tiles": 0, "sprites": 0}

    # ------------------------------------------------------------ cached resources
    def tile(self, tdb, tidx: int, variant: int):
        tdef = tdb.get(tidx)
        return self.assets.image(f"terrain/{tdef['id']}_{variant}", lambda: art.make_tile(tdef, variant))

    def hole(self, tdb, tidx: int, depth: int):
        tdef = tdb.get(tidx)
        return self.assets.image(f"terrain/{tdef['id']}_hole{depth}", lambda: art.make_hole(tdef, depth))

    def obj_sprite(self, art_id: str, variant: int):
        return self.assets.anchored(f"objects/{art_id}_{variant}", lambda: art.make_object(art_id, variant))

    def icon(self, db, item_id: str, size: int):
        idef = db.item(item_id)
        return self.assets.image(f"ui/icons/{item_id}_{size}",
                                 lambda: art.make_icon(idef.get("icon", {}), 1, 1, size))

    def text(self, s: str, size: int, color, outline=True):
        key = (s, size, color, outline)
        surf = self._text_cache.get(key)
        if surf is None:
            font = self.assets.font(size, True)
            base = font.render(s, True, color)
            if outline:
                o = font.render(s, True, (20, 16, 14))
                surf = pygame.Surface((base.get_width() + 2, base.get_height() + 2), pygame.SRCALPHA)
                for dx, dy in ((0, 1), (2, 1), (1, 0), (1, 2)):
                    surf.blit(o, (dx, dy))
                surf.blit(base, (1, 1))
            else:
                surf = base
            if len(self._text_cache) > 400:
                self._text_cache.clear()
            self._text_cache[key] = surf
        return surf

    def light(self, radius: int, color):
        key = (radius // 8 * 8, color)
        s = self._light_cache.get(key)
        if s is None:
            s = art.make_light(max(8, key[0]), color)
            self._light_cache[key] = s
        return s

    # ------------------------------------------------------------ main
    def draw(self, screen: pygame.Surface, session, dt: float):
        self.time += dt
        cam = self.camera
        world = session.world
        tdb = session.terrain_db
        W, H = screen.get_size()
        screen.fill((24, 20, 18))
        x0, y0, x1, y1 = cam.visible_tile_bounds()
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(world.size - 1, x1), min(world.size - 1, y1)

        # ---------------- ground
        tiles = 0
        clue = self.assets.image("effects/cache_clue", art.make_cache_clue)
        chunks = world.chunks
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                sx, sy = cam.world_to_screen(x, y)
                if sy > H or sy + TILE_H + 8 < 0 or sx + HALF_W < 0 or sx - HALF_W > W:
                    continue
                ch = chunks.get((x // CHUNK_SIZE, y // CHUNK_SIZE))
                if ch is None:
                    continue
                i = (y % CHUNK_SIZE) * CHUNK_SIZE + (x % CHUNK_SIZE)
                tidx = ch.terrain[i]
                screen.blit(self.tile(tdb, tidx, (x * 7 + y * 13) % 3), (sx - HALF_W, sy))
                d = ch.depth[i]
                if d:
                    screen.blit(self.hole(tdb, tidx, d), (sx - HALF_W, sy))
                f = ch.flags[i]
                if f & FLAG_CACHE and not f & FLAG_CACHE_FOUND:
                    screen.blit(clue, (sx - HALF_W, sy))
                tiles += 1
        self.stats["tiles"] = tiles

        # ---------------- decals & flat props
        for d in session.effects.decals:
            sx, sy = cam.world_to_screen(d.x, d.y)
            if not cam.on_screen(sx, sy):
                continue
            a = min(1.0, d.life / 5.0)
            r = int(10 * d.size)
            surf = pygame.Surface((r * 3, r * 2), pygame.SRCALPHA)
            rng = random.Random(d.seed)
            for _ in range(4):
                pygame.draw.ellipse(surf, (*d.color, int(170 * a)), (rng.randint(0, r), rng.randint(0, r // 2 + 1), r + rng.randint(0, r), r // 2 + rng.randint(1, r // 2 + 1)))
            screen.blit(surf, (sx - r * 1.5, sy - r))
        objs = list(world.objects_in_rect(x0 - 1, y0 - 1, x1 + 1, y1 + 1))
        for o in objs:
            a = o.defn.get("art")
            if a in FLAT:
                surf, ax, ay = self.obj_sprite(a, o.variant)
                sx, sy = cam.world_to_screen(o.x, o.y)
                screen.blit(surf, (sx - ax, sy - ay))

        self._draw_markers(screen, session)

        # ---------------- sorted sprites
        sprites = []
        for o in objs:
            if o.defn.get("art") in FLAT:
                continue
            sprites.append((o.x + o.y, 0, o))
        for e in session.enemies:
            if x0 - 2 <= e.x <= x1 + 2 and y0 - 2 <= e.y <= y1 + 2:
                sprites.append((e.x + e.y + (-0.5 if e.dead else 0), 1, e))
        p = session.player
        sprites.append((p.x + p.y, 2, p))
        sprites.sort(key=lambda s: s[0])
        self.stats["sprites"] = len(sprites)
        for _, kind, obj in sprites:
            if kind == 0:
                self._draw_object(screen, session, obj)
            elif kind == 1:
                self._draw_enemy(screen, session, obj)
            else:
                self._draw_player(screen, session, obj)

        self._draw_effects(screen, session)
        self._draw_lighting(screen, session)
        self._draw_weather(screen, session, dt)

    # ------------------------------------------------------------ markers
    def _draw_markers(self, screen, session):
        cam = self.camera
        p = session.player
        tgt = session.digging.target
        if tgt is not None and p.placing is None:
            col = (255, 230, 140) if session.digging.target_ok else (230, 90, 70)
            marker = self.assets.image(f"effects/marker_{col}", lambda: art.make_target_marker(col))
            sx, sy = cam.world_to_screen(*tgt)
            screen.blit(marker, (sx - HALF_W, sy))
            if p.digging:
                prog = session.digging.progress
                cx, cy = cam.world_to_screen(tgt[0] + 0.5, tgt[1] + 0.5)
                rect = pygame.Rect(0, 0, 34, 34)
                rect.center = (cx, cy - 26)
                pygame.draw.circle(screen, (20, 16, 14), rect.center, 18)
                pygame.draw.arc(screen, (90, 80, 70), rect, 0, math.tau, 4)
                if prog > 0:
                    pygame.draw.arc(screen, (255, 210, 110), rect, math.pi / 2, math.pi / 2 + math.tau * prog, 4)
                info = session.world.cell(*tgt)
                if info:
                    label = self.text(session.terrain_db.layer_name(info.depth + 1), 13, (240, 230, 210))
                    screen.blit(label, (cx - label.get_width() // 2, cy - 56))
        if p.placing is not None and p.placing.get("cell"):
            cell = p.placing["cell"]
            ok = p.placing.get("ok")
            sdef = session.resources["structures"][p.placing["struct"]]
            surf, ax, ay = self.obj_sprite(sdef["art"], 0)
            ghost = surf.copy()
            ghost.fill((120, 255, 140, 150) if ok else (255, 110, 100, 150), special_flags=pygame.BLEND_RGBA_MULT)
            col = (140, 255, 150) if ok else (255, 110, 100)
            marker = self.assets.image(f"effects/marker_{col}", lambda: art.make_target_marker(col))
            sx, sy = cam.world_to_screen(*cell)
            screen.blit(marker, (sx - HALF_W, sy))
            gx, gy = cam.world_to_screen(cell[0] + 0.5, cell[1] + 0.5)
            screen.blit(ghost, (gx - ax, gy - ay))
        # interaction highlight
        if p.action is None and not p.dead and p.placing is None:
            obj = session.find_interactable()
            if obj is not None:
                sx, sy = cam.world_to_screen(obj.x, obj.y)
                pulse = 0.5 + 0.5 * math.sin(self.time * 6)
                r = max(14, int(obj.radius * 50))
                pygame.draw.ellipse(screen, (255, 230, 150), (sx - r, sy - r // 2, r * 2, r), 2 if pulse > 0.5 else 1)

    # ------------------------------------------------------------ objects
    def _draw_object(self, screen, session, o):
        cam = self.camera
        sx, sy = cam.world_to_screen(o.x, o.y)
        if not cam.on_screen(sx, sy, 160):
            return
        if o.kind == "pickup":
            bob = math.sin(o.anim * 3 + o.x) * 2
            self._shadow(screen, sx, sy, 18, 8, 70)
            icon = self.icon(session.db, o.stack.item_id, PICKUP_ICON)
            glow = 0.5 + 0.5 * math.sin(o.anim * 4)
            pygame.draw.circle(screen, (255, 240, 180), (int(sx), int(sy - 14 - o.z * 10 + bob)), int(13 + glow * 2), 1)
            screen.blit(icon, (sx - PICKUP_ICON // 2, sy - 14 - PICKUP_ICON // 2 - o.z * 10 + bob))
            return
        art_id = o.defn.get("art", o.otype)
        surf, ax, ay = self.obj_sprite(art_id, o.variant)
        if o.uses == 0 and o.defn.get("interact") == "search":
            # Already searched: desaturate slightly so players can read the world state.
            key = f"objects/{art_id}_{o.variant}_searched"
            surf = self.assets.image(key, lambda: _dim(surf))
        screen.blit(surf, (sx - ax, sy - ay))
        if o.otype == "campfire" and o.data.get("fuel", 0) > 0:
            t = self.time * 9 + o.x
            fuel_k = min(1.0, o.data["fuel"] / 200 + 0.4)
            for i, (col, h, w) in enumerate((((240, 120, 30), 26, 12), ((255, 190, 60), 18, 8), ((255, 245, 180), 10, 4))):
                fh = h * fuel_k * (0.85 + 0.15 * math.sin(t + i * 1.7))
                sway = math.sin(t * 0.7 + i) * 2
                pts = [(sx - w, sy - 6), (sx + sway, sy - 6 - fh), (sx + w, sy - 6)]
                pygame.draw.polygon(screen, col, pts)
        if o.kind == "structure" and o.otype == "barricade" and o.hp < o.defn.get("hp", 150):
            frac = max(0.0, o.hp / o.defn.get("hp", 150))
            pygame.draw.rect(screen, (20, 16, 14), (sx - 16, sy - 54, 32, 5))
            pygame.draw.rect(screen, (200, 160, 80), (sx - 15, sy - 53, 30 * frac, 3))

    # ------------------------------------------------------------ characters
    def _shadow(self, screen, sx, sy, w=26, h=10, alpha=80):
        sh = self.assets.image(f"effects/shadow_{w}_{h}_{alpha}", lambda: _ellipse_surface(w, h, alpha))
        screen.blit(sh, (sx - w // 2, sy - h // 2))

    def _draw_player(self, screen, session, p):
        cam = self.camera
        depth = session.world.depth_at(p.x, p.y)
        sx, sy = cam.world_to_screen(p.x, p.y)
        sy += depth * DEPTH_PX * 0.6
        self._shadow(screen, sx, sy, 28, 11)
        attack_t, style, arc = None, "swing", 90
        weapon_id = p.equipped_item_id()
        if p.attack_anim is not None:
            attack_t, style, arc = p.attack_anim["t"], p.attack_anim["style"], p.attack_anim["arc"]
            weapon_id = p.attack_anim["visual"] if p.attack_anim["visual"] is not None else weapon_id
            if p.weapon()["type"] == "ranged" and p.attack_anim["visual"] is None:
                weapon_id = p.equipped_item_id()
        elif p.digging:
            attack_t = (self.time * 1.8) % 1.0
            style, arc = "chop", 70
        elif p.action is not None:
            attack_t = (self.time * 1.5) % 1.0 if p.action["kind"] in ("chop",) else None
            style, arc = "chop", 60
        screen_angle = p.aim_screen_angle if not p.digging and p.action is None else _screen_angle(p.facing)
        surf = chars.draw_humanoid(Player_LOOK(p), screen_angle, p.walk_phase, p.moving, weapon_id, attack_t, style, arc,
                                   p.flash, p.recoil)
        if p.dead:
            surf = chars.death_pose(surf, p.death_t + 0.5, p.facing)
            screen.blit(surf, (sx - surf.get_width() // 2, sy - surf.get_height() // 2 - 8))
            return
        if p.dodge_t > 0:
            surf.set_alpha(170)
        if p.action is not None:
            sy += math.sin(self.time * 10) * 1.5
        screen.blit(surf, (sx - chars.FEET_X, sy - chars.FEET_Y))
        if p.reload_t > 0:
            frac = 1 - p.reload_t / max(0.01, p.reload_total)
            pygame.draw.rect(screen, (20, 16, 14), (sx - 20, sy - 76, 40, 6))
            pygame.draw.rect(screen, (240, 210, 120), (sx - 19, sy - 75, 38 * frac, 4))
        if p.action is not None:
            a = p.action
            frac = min(1.0, a["t"] / a["dur"])
            rect = pygame.Rect(0, 0, 30, 30)
            rect.center = (sx, sy - 88)
            pygame.draw.circle(screen, (20, 16, 14), rect.center, 16)
            pygame.draw.arc(screen, (180, 230, 160), rect, math.pi / 2, math.pi / 2 + math.tau * frac, 4)

    def _draw_enemy(self, screen, session, e):
        cam = self.camera
        sx, sy = cam.world_to_screen(e.x, e.y)
        if not cam.on_screen(sx, sy, 100):
            return
        sy += session.world.depth_at(e.x, e.y) * DEPTH_PX * 0.6
        look = e.defn.get("look", {})
        shape = look.get("shape", "human")
        ang = _screen_angle(e.facing)
        attack_t = e.attack_anim
        if attack_t is None and e.state == "attack":
            attack_t = 0.12  # wind-up pose telegraphs the strike
        if not e.dead:
            self._shadow(screen, sx, sy, 26 if shape != "small" else 16, 10 if shape != "small" else 6)
        if shape == "beast":
            surf = chars.draw_beast(look, ang, e.walk_phase, e.moving, attack_t, e.flash)
        elif shape == "small":
            surf = chars.draw_small(look, ang, e.walk_phase, e.moving, e.flash)
        else:
            surf = chars.draw_humanoid(look, ang, e.walk_phase, e.moving, look.get("weapon"), attack_t, "swing", 90, e.flash)
        if e.dead:
            surf = chars.death_pose(surf, e.death_t, e.facing)
            screen.blit(surf, (sx - surf.get_width() // 2, sy - surf.get_height() // 2 - 6))
            return
        if e.state == "attack" and e.attack_anim is None:
            # Red wind-up glint so the player can react with a dodge.
            pygame.draw.circle(screen, (255, 80, 60), (int(sx), int(sy - 70 * look.get("height", 1))), 4)
        screen.blit(surf, (sx - chars.FEET_X, sy - chars.FEET_Y))
        top = sy - 20 - 58 * look.get("height", 1.0)
        if e.hp < e.max_hp:
            frac = max(0.0, e.hp / e.max_hp)
            pygame.draw.rect(screen, (20, 16, 14), (sx - 18, top - 4, 36, 6))
            pygame.draw.rect(screen, (220, 70, 60), (sx - 17, top - 3, 34 * frac, 4))
        if e.alert > 0 and e.defn.get("kind") == "hostile":
            t = self.text("!" if e.state == "chase" else "?", 20, (255, 210, 80))
            screen.blit(t, (sx - t.get_width() // 2, top - 28))

    # ------------------------------------------------------------ effects
    def _draw_effects(self, screen, session):
        cam = self.camera
        fx = session.effects
        for t in fx.tracers:
            a = cam.world_to_screen(t.x0, t.y0, 20)
            b = cam.world_to_screen(t.x1, t.y1, 20)
            k = t.life / t.max_life
            pygame.draw.line(screen, t.color, a, b, max(1, int(3 * k)))
        for p in fx.particles:
            if not p.alive:
                continue
            sx, sy = cam.world_to_screen(p.x, p.y, p.z)
            if not cam.on_screen(sx, sy, 10):
                continue
            k = p.life / p.max_life
            if p.kind == "smoke":
                r = int(p.size * (1.6 - k))
                s = pygame.Surface((r * 2 + 2, r * 2 + 2), pygame.SRCALPHA)
                pygame.draw.circle(s, (*p.color, int(110 * k)), (r + 1, r + 1), r)
                screen.blit(s, (sx - r, sy - r))
            elif p.kind == "spark":
                ex = sx - p.vx * 3
                ey = sy - (p.vx + p.vy) * 1.2 + p.vz * 0.02
                pygame.draw.line(screen, p.color, (sx, sy), (ex, ey), max(1, int(p.size * k)))
            elif p.kind == "chunk":
                r = max(1, int(p.size))
                pygame.draw.rect(screen, p.color, (sx - r, sy - r, r * 2, r * 2))
            else:
                pygame.draw.circle(screen, p.color, (int(sx), int(sy)), max(1, int(p.size * (0.4 + 0.6 * k))))
        for tr in fx.trails:
            k = tr.life / tr.max_life
            if tr.kind == "stab":
                a = cam.world_to_screen(tr.x + math.cos(tr.angle) * 0.3, tr.y + math.sin(tr.angle) * 0.3, 24)
                b = cam.world_to_screen(tr.x + math.cos(tr.angle) * tr.radius, tr.y + math.sin(tr.angle) * tr.radius, 24)
                pygame.draw.line(screen, tr.color, a, b, max(1, int(5 * k)))
                continue
            pts_outer, pts_inner = [], []
            n = 10
            sweep = 1 - k
            for i in range(n + 1):
                u = i / n
                ang = tr.angle - tr.arc / 2 + tr.arc * u * min(1.0, 0.4 + sweep * 2)
                ro = tr.radius
                ri = tr.radius * (0.55 + 0.25 * u)
                pts_outer.append(cam.world_to_screen(tr.x + math.cos(ang) * ro, tr.y + math.sin(ang) * ro, 22))
                pts_inner.append(cam.world_to_screen(tr.x + math.cos(ang) * ri, tr.y + math.sin(ang) * ri, 22))
            poly = pts_outer + pts_inner[::-1]
            if len(poly) >= 3:
                pygame.draw.polygon(screen, _fade(tr.color, k), poly)
                pygame.draw.lines(screen, (255, 255, 255), False, pts_outer, max(1, int(3 * k)))
        for t in fx.texts:
            sx, sy = cam.world_to_screen(t.x, t.y, t.z + 30)
            k = t.life / t.max_life
            surf = self.text(t.text, int(18 * t.scale), t.color)
            if k < 0.4:
                surf = surf.copy()
                surf.set_alpha(int(255 * k / 0.4))
            screen.blit(surf, (sx - surf.get_width() // 2, sy))
        for f in fx.flashes:
            sx, sy = cam.world_to_screen(f.x, f.y, 22)
            k = f.life / f.max_life
            r = int(10 + 10 * k)
            pygame.draw.circle(screen, f.color, (int(sx), int(sy)), r)
            pygame.draw.circle(screen, (255, 255, 240), (int(sx), int(sy)), r // 2)

    # ------------------------------------------------------------ lighting / weather
    def _draw_lighting(self, screen, session):
        ambient = session.clock.ambient()
        if session.weather.intensity("dust") > 0:
            k = session.weather.intensity("dust")
            ambient = tuple(int(a * (1 - 0.25 * k) + b * 0.25 * k) for a, b in zip(ambient, (230, 160, 90)))
        if session.weather.intensity("rain") > 0:
            k = session.weather.intensity("rain") * 0.18
            ambient = tuple(int(a * (1 - k) + b * k) for a, b in zip(ambient, (90, 100, 120)))
        if min(ambient) >= 250 and not session.effects.flashes:
            return
        cam = self.camera
        size = screen.get_size()
        if self.light_map is None or self.light_map.get_size() != size:
            self.light_map = pygame.Surface(size)
        lm = self.light_map
        lm.fill(ambient)
        p = session.player
        night_k = 1.0 - session.clock.light_level()
        if night_k > 0.2:
            sx, sy = cam.world_to_screen(p.x, p.y, 20)
            g = self.light(110, (90, 90, 100))
            lm.blit(g, (sx - g.get_width() // 2, sy - g.get_height() // 2), special_flags=pygame.BLEND_RGB_ADD)
        for fire in session.world_system.campfires_near(p.x, p.y, 26):
            sx, sy = cam.world_to_screen(fire.x, fire.y, 10)
            flick = 1.0 + 0.06 * math.sin(self.time * 13 + fire.x) + 0.04 * math.sin(self.time * 31)
            r = int(fire.defn.get("light", 6) * 32 * flick)
            g = self.light(r, (255, 170, 90))
            lm.blit(g, (sx - g.get_width() // 2, sy - g.get_height() // 2), special_flags=pygame.BLEND_RGB_ADD)
        for f in session.effects.flashes:
            sx, sy = cam.world_to_screen(f.x, f.y, 20)
            g = self.light(int(f.radius * 40), (255, 210, 150))
            lm.blit(g, (sx - g.get_width() // 2, sy - g.get_height() // 2), special_flags=pygame.BLEND_RGB_ADD)
        screen.blit(lm, (0, 0), special_flags=pygame.BLEND_RGB_MULT)

    def _fog_overlay(self, size, color):
        key = (size, color)
        s = self._fog_cache.get(key)
        if s is None:
            w, h = size
            s = pygame.Surface((w, h), pygame.SRCALPHA)
            steps = 18
            s.fill((*color, 225))
            for i in range(steps, 0, -1):
                t = i / steps
                a = int(225 * t ** 1.4)
                rw, rh = w * 0.95 * t, h * 1.05 * t
                pygame.draw.ellipse(s, (*color, a), (w / 2 - rw / 2, h / 2 - rh / 2, rw, rh))
            self._fog_cache[key] = s
        return s

    def _draw_weather(self, screen, session, dt):
        W, H = screen.get_size()
        wx = session.weather
        fog = wx.intensity("fog")
        dust = wx.intensity("dust")
        rain = wx.intensity("rain")
        p = session.player
        psx, psy = self.camera.world_to_screen(p.x, p.y, 20)
        if fog > 0.02:
            ov = self._fog_overlay((W, H), (196, 198, 196))
            ov2 = ov.copy()
            ov2.set_alpha(int(220 * fog))
            screen.blit(ov2, (psx - W / 2, psy - H / 2))
        if dust > 0.02:
            ov = self._fog_overlay((W, H), (196, 150, 96))
            ov2 = ov.copy()
            ov2.set_alpha(int(210 * dust))
            screen.blit(ov2, (psx - W / 2, psy - H / 2))
            for d in self.dust:
                d[0] = (d[0] + dt * 0.5 * d[2]) % 1.0
                d[1] = (d[1] + dt * 0.08 * d[2]) % 1.0
                x, y = d[0] * W, d[1] * H
                pygame.draw.line(screen, (220, 180, 120), (x, y), (x - 24 * d[2], y - 4), 1)
        if rain > 0.02:
            n = int(len(self.rain) * rain)
            col = (170, 190, 215)
            for d in self.rain[:n]:
                d[1] = (d[1] + dt * 1.6 * d[2]) % 1.0
                d[0] = (d[0] + dt * 0.12) % 1.0
                x, y = d[0] * W, d[1] * H
                pygame.draw.line(screen, col, (x, y), (x - 4, y + 14 * d[2]), 1)
        if session.sleep_fade > 0:
            s = pygame.Surface((W, H))
            s.fill((0, 0, 0))
            s.set_alpha(int(255 * min(1.0, session.sleep_fade)))
            screen.blit(s, (0, 0))


def Player_LOOK(p):
    look = dict(p.LOOK)
    if p.body is not None:
        col = p.db.item(p.body.item_id).get("icon", {}).get("color")
        if col:
            look["body"] = tuple(col)
    return look


def _screen_angle(world_angle: float) -> float:
    sx = (math.cos(world_angle) - math.sin(world_angle)) * HALF_W
    sy = (math.cos(world_angle) + math.sin(world_angle)) * HALF_H
    return math.atan2(sy, sx)


def _fade(color, k):
    return tuple(int(c * (0.35 + 0.4 * k)) for c in color[:3])


def _dim(surf):
    s = surf.copy()
    s.fill((175, 170, 165, 255), special_flags=pygame.BLEND_RGBA_MULT)
    return s


def _ellipse_surface(w, h, alpha):
    s = pygame.Surface((w, h), pygame.SRCALPHA)
    pygame.draw.ellipse(s, (0, 0, 0, alpha), s.get_rect())
    return s
