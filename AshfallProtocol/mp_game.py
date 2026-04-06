"""
Ashfall Protocol – Multiplayer Client-Side Game
================================================
Renders the authoritative server state received via NetworkClient.

Fixes vs initial version
-------------------------
* Weapon selection is now tracked by KEYDOWN events (edge-triggered),
  not by key-state each frame → no more forced weapon-0 every tick.
* Input packet always carries the current selected weapon index.
"""

import math
import time

import pygame

from asset_loader import AssetLoader
from config import (ACCENT_COLOR, BG_COLOR, DANGER_COLOR, GRID_COLOR,
                    HEAL_COLOR, PANEL_COLOR, RUBBLE_COLOR, TEXT_COLOR,
                    WIDTH, HEIGHT, WORLD_WIDTH, WORLD_HEIGHT,
                    CAMERA_LERP, FPS, TITLE, vec)
from game import OBSTACLE_DEFS, DECOR_POINTS
from network.client import NetworkClient

_SNAPSHOT_DT  = 1.0 / 20.0   # expected server broadcast interval

_MY_COLOR     = (120, 205, 214)
_OTHER_COLOR  = (205, 160, 80)
_DEAD_COLOR   = (80, 80, 80)

_ITEM_COLORS  = {'hp': HEAL_COLOR, 'ammo': ACCENT_COLOR}
_BOSS_TYPES   = {'BossEnemy'}

_WEAPON_NAMES = [
    'Rifle', 'Shotgun', 'SMG',
    'Sniper', 'Grenade', 'Energy', 'Chain', 'Cryo',
]


# ─── Camera ──────────────────────────────────────────────────────────────────

class _Camera:
    def __init__(self):
        self.pos    = vec(WORLD_WIDTH / 2, WORLD_HEIGHT / 2)
        self.offset = vec()

    def update(self, target):
        self.pos += (target - self.pos) * CAMERA_LERP
        desired = self.pos - vec(WIDTH / 2, HEIGHT / 2)
        desired.x = max(0, min(WORLD_WIDTH  - WIDTH,  desired.x))
        desired.y = max(0, min(WORLD_HEIGHT - HEIGHT, desired.y))
        self.offset = desired

    def w2s(self, pos) -> tuple:
        return int(pos[0] - self.offset.x), int(pos[1] - self.offset.y)

    def s2w(self, pos) -> pygame.math.Vector2:
        return vec(pos[0] + self.offset.x, pos[1] + self.offset.y)


# ─── State interpolation ─────────────────────────────────────────────────────

def _lerp(a, b, t):
    return a + (b - a) * t


def _interp_state(prev, curr, alpha):
    if prev is None or alpha >= 1.0:
        return curr

    def interp_list(pl, cl, key):
        pm = {e[key]: e for e in pl}
        out = []
        for c in cl:
            p = pm.get(c[key])
            if p:
                ic = dict(c)
                ic['x'] = _lerp(p['x'], c['x'], alpha)
                ic['y'] = _lerp(p['y'], c['y'], alpha)
                out.append(ic)
            else:
                out.append(c)
        return out

    return {
        **curr,
        'players': interp_list(prev.get('players', []),
                               curr.get('players', []), 'pid'),
        'enemies': interp_list(prev.get('enemies', []),
                               curr.get('enemies', []), 'id'),
        'bullets': interp_list(prev.get('bullets', []),
                               curr.get('bullets', []), 'id'),
    }


# ─── MPClientGame ────────────────────────────────────────────────────────────

class MPClientGame:
    def __init__(self, client: NetworkClient):
        self.client   = client
        self.my_pid   = client.my_pid
        self.running  = True

        self.screen   = pygame.display.get_surface()
        self.clock    = pygame.time.Clock()
        self.assets   = AssetLoader()
        self.camera   = _Camera()

        self.font_hud  = self.assets.font(16, bold=True)
        self.font_name = self.assets.font(13, bold=True)
        self.font_wave = self.assets.font(24, bold=True)
        self.font_over = self.assets.font(36, bold=True)
        self.font_sm   = self.assets.font(14)

        self.obstacles = [pygame.Rect(*d) for d in OBSTACLE_DEFS]

        # ── Weapon tracking (event-driven, not key-state) ────────────────
        self._weapon_idx = 0        # tracks the player's currently selected weapon

        # ── Wave flash ────────────────────────────────────────────────────
        self._last_wave  = 1
        self._wave_flash = 0.0

    # ── Main loop ────────────────────────────────────────────────────────────

    def run(self):
        pygame.display.set_caption(f'{TITLE} – Multiplayer')
        while self.running:
            dt = min(self.clock.tick(FPS) / 1000.0, 0.05)
            self._handle_events()
            self._send_input()
            state = self._get_interp_state()
            if state:
                self._update_camera(state)
                self._update_wave_flash(state, dt)
            self._draw(state)
        self.client.disconnect()
        pygame.display.set_caption(TITLE)

    # ── Events ───────────────────────────────────────────────────────────────

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                # Weapon select: KEYDOWN events only (edge-triggered)
                for i, key in enumerate([pygame.K_1, pygame.K_2, pygame.K_3,
                                         pygame.K_4, pygame.K_5, pygame.K_6,
                                         pygame.K_7, pygame.K_8]):
                    if event.key == key:
                        self._weapon_idx = i
                        break

    # ── Input ────────────────────────────────────────────────────────────────

    def _send_input(self):
        keys   = pygame.key.get_pressed()
        mouse  = pygame.mouse.get_pressed()
        mpos   = pygame.mouse.get_pos()
        world  = self.camera.s2w(mpos)

        self.client.send_input({
            'up':     bool(keys[pygame.K_w]),
            'down':   bool(keys[pygame.K_s]),
            'left':   bool(keys[pygame.K_a]),
            'right':  bool(keys[pygame.K_d]),
            'mx':     round(world.x, 1),
            'my':     round(world.y, 1),
            'shoot':  bool(mouse[0]),
            'dash':   bool(keys[pygame.K_SPACE]),
            'reload': bool(keys[pygame.K_r]),
            'weapon': self._weapon_idx,   # ← always the current selection
        })

    # ── Interpolation ─────────────────────────────────────────────────────────

    def _get_interp_state(self):
        prev, curr, t = self.client.latest_state()
        if curr is None:
            return None
        alpha = min(1.0, (time.monotonic() - t) / _SNAPSHOT_DT)
        return _interp_state(prev, curr, alpha)

    # ── Camera ───────────────────────────────────────────────────────────────

    def _update_camera(self, state):
        me = self._my(state)
        if me:
            self.camera.update(vec(me['x'], me['y']))

    def _my(self, state) -> dict | None:
        for p in state.get('players', []):
            if p['pid'] == self.my_pid:
                return p
        return None

    # ── Wave flash ────────────────────────────────────────────────────────────

    def _update_wave_flash(self, state, dt):
        w = state.get('wave', 1)
        if w != self._last_wave:
            self._wave_flash = 2.5
            self._last_wave  = w
        self._wave_flash = max(0.0, self._wave_flash - dt)

    # ── Drawing ──────────────────────────────────────────────────────────────

    def _draw(self, state):
        self._draw_bg()
        if state is None:
            self._draw_connecting()
            pygame.display.flip()
            return
        self._draw_barrels(state.get('barrels', []))
        self._draw_items(state.get('items', []))
        self._draw_bullets(state.get('bullets', []))
        self._draw_enemies(state.get('enemies', []))
        self._draw_players(state.get('players', []))
        me = self._my(state)
        self._draw_hud(state, me)
        self._draw_wave_flash(state)
        self._draw_boss_bar(state.get('enemies', []))
        self._draw_scoreboard(state.get('players', []))
        self._draw_weapon_bar(me)
        pygame.display.flip()

    def _draw_bg(self):
        self.screen.fill(BG_COLOR)
        grid = 64
        ox   = int(self.camera.offset.x // grid) * grid
        oy   = int(self.camera.offset.y // grid) * grid
        for x in range(ox, int(self.camera.offset.x + WIDTH) + grid, grid):
            pygame.draw.line(self.screen, GRID_COLOR,
                             (x - self.camera.offset.x, 0),
                             (x - self.camera.offset.x, HEIGHT))
        for y in range(oy, int(self.camera.offset.y + HEIGHT) + grid, grid):
            pygame.draw.line(self.screen, GRID_COLOR,
                             (0, y - self.camera.offset.y),
                             (WIDTH, y - self.camera.offset.y))
        for (x, y) in DECOR_POINTS:
            pos = self.camera.w2s((x, y))
            if -60 < pos[0] < WIDTH + 60 and -60 < pos[1] < HEIGHT + 60:
                pygame.draw.circle(self.screen, (44, 40, 36), pos, 32)
                pygame.draw.circle(self.screen, (60, 54, 46), pos, 14)
        for ob in self.obstacles:
            r = pygame.Rect(ob.x - self.camera.offset.x,
                            ob.y - self.camera.offset.y,
                            ob.width, ob.height)
            pygame.draw.rect(self.screen, RUBBLE_COLOR, r, border_radius=8)
            pygame.draw.rect(self.screen, (85, 78, 70), r, 3, border_radius=8)
            pygame.draw.line(self.screen, (90, 85, 78),
                             (r.left + 4, r.top + 3),
                             (r.right - 4, r.top + 3), 2)

    def _draw_barrels(self, barrels):
        for b in barrels:
            pos = self.camera.w2s((b['x'], b['y']))
            r   = 14
            pygame.draw.circle(self.screen, (200, 130, 50), pos, r)
            pygame.draw.circle(self.screen, (160, 90, 30),  pos, r, 3)
            pygame.draw.line(self.screen, (100, 50, 20),
                             (pos[0] - r + 3, pos[1] - r + 3),
                             (pos[0] + r - 3, pos[1] + r - 3), 2)
            pygame.draw.line(self.screen, (100, 50, 20),
                             (pos[0] + r - 3, pos[1] - r + 3),
                             (pos[0] - r + 3, pos[1] + r - 3), 2)

    def _draw_items(self, items):
        for item in items:
            pos = self.camera.w2s((item['x'], item['y']))
            col = _ITEM_COLORS.get(item['kind'], ACCENT_COLOR)
            pygame.draw.circle(self.screen, col,           pos, 8)
            pygame.draw.circle(self.screen, (255, 255, 255), pos, 8, 2)

    def _draw_bullets(self, bullets):
        for b in bullets:
            pos = self.camera.w2s((b['x'], b['y']))
            col = tuple(b['color']) if b.get('color') else (220, 200, 100)
            r   = max(2, int(b.get('r', 4)))
            vx, vy = b.get('vx', 0), b.get('vy', 0)
            spd = math.hypot(vx, vy)
            if spd > 0:
                tl = min(22, int(spd * 0.02))
                tx = int(-vx / spd * tl)
                ty = int(-vy / spd * tl)
                dim = tuple(max(0, c - 80) for c in col)
                pygame.draw.line(self.screen, dim, pos, (pos[0] + tx, pos[1] + ty), max(1, r - 1))
            pygame.draw.circle(self.screen, col, pos, r)

    def _draw_enemies(self, enemies):
        for e in enemies:
            pos   = self.camera.w2s((e['x'], e['y']))
            col   = tuple(e['color']) if e.get('color') else (219, 71, 71)
            r     = int(e.get('r', 18))
            hp    = e.get('hp', 0)
            mhp   = e.get('mhp', 1)
            pygame.draw.circle(self.screen, col, pos, r)
            if e.get('type') in _BOSS_TYPES:
                pygame.draw.circle(self.screen, (220, 80, 80), pos, r + 5, 3)
            ratio = max(0.0, hp / max(mhp, 1))
            bw    = r * 2
            bx, by = pos[0] - r, pos[1] - r - 10
            pygame.draw.rect(self.screen, (30, 30, 30), (bx, by, bw, 5))
            pygame.draw.rect(self.screen, (200, 64, 64),
                             (bx, by, int(bw * ratio), 5))

    def _draw_players(self, players):
        for p in players:
            is_me  = p['pid'] == self.my_pid
            pos    = self.camera.w2s((p['x'], p['y']))
            dead   = p.get('dead', False)
            dash   = p.get('dash', False)

            if dead:
                col = _DEAD_COLOR
            elif dash:
                col = (180, 255, 255)
            elif is_me:
                col = _MY_COLOR
            else:
                col = _OTHER_COLOR

            r = 18
            pygame.draw.circle(self.screen, col, pos, r)
            pygame.draw.circle(self.screen, (36, 38, 44), pos, r - 6)

            ax, ay = p.get('ax', 1.0), p.get('ay', 0.0)
            tip    = (pos[0] + int(ax * 26), pos[1] + int(ay * 26))
            pygame.draw.line(self.screen, (70, 70, 70), pos, tip, 6)
            pygame.draw.line(self.screen, (160, 160, 160), pos, tip, 2)

            if dash and is_me:
                ghost = (pos[0] - int(ax * 16), pos[1] - int(ay * 16))
                pygame.draw.circle(self.screen, (80, 200, 220), ghost, r - 4, 2)

            name   = p.get('name', '?')
            label  = f'[{name}]' if is_me else name
            nc     = ACCENT_COLOR if is_me else (200, 190, 170)
            ns     = self.font_name.render(label, True, nc)
            self.screen.blit(ns, ns.get_rect(center=(pos[0], pos[1] - r - 14)))

            if not dead:
                hp, mhp = p.get('hp', 100), p.get('mhp', 100)
                ratio   = max(0.0, hp / max(mhp, 1))
                bw = 40
                bx = pos[0] - bw // 2
                by = pos[1] - r - 28
                pygame.draw.rect(self.screen, (30, 30, 30), (bx, by, bw, 4))
                hc = HEAL_COLOR if ratio > 0.3 else DANGER_COLOR
                pygame.draw.rect(self.screen, hc, (bx, by, int(bw * ratio), 4))

    # ── HUD ──────────────────────────────────────────────────────────────────

    def _draw_hud(self, state, me):
        pad = 10
        # Top-right: wave / score / kills / ping
        lines = [
            (f'Wave  {state.get("wave", 1)}',  ACCENT_COLOR),
            (f'Score {state.get("score", 0)}',  TEXT_COLOR),
            (f'Kills {state.get("kills", 0)}',  TEXT_COLOR),
            (f'Ping  {self.client.ping_ms} ms', (100, 95, 90)),
        ]
        y = pad
        for text, col in lines:
            s = self.font_hud.render(text, True, col)
            self.screen.blit(s, s.get_rect(topright=(WIDTH - pad, y)))
            y += 21

        if not me:
            return

        # HP bar (top-left)
        hp, mhp = me.get('hp', 0), me.get('mhp', 100)
        ratio   = max(0.0, hp / max(mhp, 1))
        bw, bh  = 200, 18
        bx, by  = pad, pad
        pygame.draw.rect(self.screen, (40, 35, 30),
                         (bx, by, bw, bh), border_radius=4)
        hc = HEAL_COLOR if ratio > 0.3 else DANGER_COLOR
        pygame.draw.rect(self.screen, hc,
                         (bx, by, int(bw * ratio), bh), border_radius=4)
        pygame.draw.rect(self.screen, (80, 75, 70),
                         (bx, by, bw, bh), 2, border_radius=4)
        ht = self.font_hud.render(f'{int(hp)} / {mhp}', True, (230, 222, 205))
        self.screen.blit(ht, ht.get_rect(midleft=(bx + 8, by + bh // 2)))

        # Ammo
        ammo = me.get('ammo', 0)
        as_  = self.font_hud.render(f'Ammo  {ammo}', True, TEXT_COLOR)
        self.screen.blit(as_, (pad, by + bh + 6))

        # Dead overlay
        if me.get('dead', False):
            ov = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            ov.fill((0, 0, 0, 130))
            self.screen.blit(ov, (0, 0))
            ds = self.font_over.render('YOU DIED', True, DANGER_COLOR)
            self.screen.blit(ds, ds.get_rect(center=(WIDTH // 2, HEIGHT // 2)))
            hs = self.font_hud.render('Waiting for next wave…', True, TEXT_COLOR)
            self.screen.blit(hs, hs.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 50)))

    def _draw_wave_flash(self, state):
        if self._wave_flash <= 0:
            return
        alpha = int(min(1.0, self._wave_flash) * 200)
        ov    = pygame.Surface((WIDTH, 60), pygame.SRCALPHA)
        ov.fill((0, 0, 0, alpha // 3))
        self.screen.blit(ov, (0, HEIGHT // 2 - 30))
        ws    = self.font_wave.render(
            f'Wave {state.get("wave", 1)} Begin!', True, ACCENT_COLOR)
        ws.set_alpha(alpha)
        self.screen.blit(ws, ws.get_rect(center=(WIDTH // 2, HEIGHT // 2)))

    def _draw_boss_bar(self, enemies):
        bosses = [e for e in enemies if e.get('type') in _BOSS_TYPES]
        if not bosses:
            return
        boss  = bosses[0]
        ratio = max(0.0, boss.get('hp', 0) / max(boss.get('mhp', 1), 1))
        bw, bh = 400, 20
        bx = (WIDTH - bw) // 2
        by = HEIGHT - 54
        pygame.draw.rect(self.screen, (30, 30, 30), (bx, by, bw, bh), border_radius=6)
        pygame.draw.rect(self.screen, DANGER_COLOR,
                         (bx, by, int(bw * ratio), bh), border_radius=6)
        pygame.draw.rect(self.screen, (180, 60, 60), (bx, by, bw, bh), 2, border_radius=6)
        ls = self.font_sm.render('BOSS', True, (255, 200, 200))
        self.screen.blit(ls, ls.get_rect(center=(WIDTH // 2, by - 14)))

    def _draw_scoreboard(self, players):
        x = 10
        y = HEIGHT - 20 - len(players) * 20
        for p in players:
            hp   = int(p.get('hp', 0))
            dead = p.get('dead', False)
            mark = '✕' if dead else '♥'
            col  = ACCENT_COLOR if p['pid'] == self.my_pid else TEXT_COLOR
            line = f"{mark} {p['name'][:12]:<12} HP:{hp:3}"
            s    = self.font_sm.render(line, True, col)
            self.screen.blit(s, (x, y))
            y += 20

    def _draw_weapon_bar(self, me):
        """Bottom-center weapon strip showing selected weapon."""
        if not me:
            return
        wi   = me.get('wi', 0)
        n    = len(_WEAPON_NAMES)
        w, h = 90, 32
        pad  = 4
        total = n * w + (n - 1) * pad
        sx   = (WIDTH - total) // 2
        sy   = HEIGHT - h - 6

        for i, wname in enumerate(_WEAPON_NAMES):
            rx  = sx + i * (w + pad)
            sel = (i == wi)
            bg  = (55, 50, 44) if sel else (22, 20, 18)
            brd = ACCENT_COLOR if sel else (55, 50, 46)
            r   = pygame.Rect(rx, sy, w, h)
            pygame.draw.rect(self.screen, bg, r, border_radius=5)
            pygame.draw.rect(self.screen, brd, r, 2, border_radius=5)
            key_s = self.font_sm.render(f'{i+1}', True, (100, 95, 90) if not sel else ACCENT_COLOR)
            self.screen.blit(key_s, (rx + 5, sy + 4))
            name_s = self.font_sm.render(wname, True, TEXT_COLOR if sel else (120, 115, 110))
            self.screen.blit(name_s, name_s.get_rect(center=(rx + w // 2, sy + h // 2 + 2)))

    def _draw_connecting(self):
        msg = self.font_wave.render('Waiting for server state…', True, ACCENT_COLOR)
        self.screen.blit(msg, msg.get_rect(center=(WIDTH // 2, HEIGHT // 2)))
        hint = self.font_sm.render('Press ESC to disconnect', True, TEXT_COLOR)
        self.screen.blit(hint, hint.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 40)))
