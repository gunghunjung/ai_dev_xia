"""
Ashfall Protocol – Multiplayer Lobby  (Room-List Edition)
==========================================================
Commercial-style flow — no manual IP entry.

Flow
----
 [Multiplayer] button in MainMenu → run_lobby()

  ┌─────────────────────────────────┐
  │   MULTIPLAYER                   │
  │                                 │
  │  [ Create Room ]                │
  │                                 │
  │  ── Available Rooms ──          │
  │  ┌─────────────────────────┐    │
  │  │  My Cool Room   1/8  ✓  │ ◄──click → join immediately
  │  │  Alice's Game   3/8     │    │
  │  └─────────────────────────┘    │
  │  [ Refresh ]    [ ← Back ]      │
  └─────────────────────────────────┘

  "Create Room" sub-screen
  ┌──────────────────────────────────┐
  │  Room Name: [_____________]      │
  │  Your Name: [_____________]      │
  │  [ Create & Host ]  [ ← Back ]  │
  └──────────────────────────────────┘

  After clicking a room → name prompt → connect & show waiting room

Returns
-------
  (NetworkClient, GameServer|None)   on game start
  (None, None)                       on back / error
"""

import time
import socket

import pygame

from asset_loader import AssetLoader
from config import (ACCENT_COLOR, BG_COLOR, DANGER_COLOR, HEAL_COLOR,
                    PANEL_COLOR, TEXT_COLOR, WIDTH, HEIGHT)
from network.client import NetworkClient
from network.discovery import LanDiscovery
from network.server import DEFAULT_PORT, GameServer


# ── Text input widget ────────────────────────────────────────────────────────

class _TextInput:
    def __init__(self, rect, placeholder, font, max_len=30):
        self.rect        = rect
        self.placeholder = placeholder
        self.font        = font
        self.max_len     = max_len
        self.text        = ''
        self.active      = False

    def handle(self, event) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.active = self.rect.collidepoint(event.pos)
            return self.active
        if not self.active:
            return False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            elif event.key not in (pygame.K_RETURN, pygame.K_ESCAPE, pygame.K_TAB):
                if len(self.text) < self.max_len and event.unicode.isprintable():
                    self.text += event.unicode
            return True
        return False

    def draw(self, surface):
        bg  = (42, 38, 34) if self.active else (26, 22, 20)
        brd = ACCENT_COLOR if self.active else (70, 65, 60)
        pygame.draw.rect(surface, bg,  self.rect, border_radius=6)
        pygame.draw.rect(surface, brd, self.rect, 2, border_radius=6)

        disp  = self.text if self.text else self.placeholder
        col   = TEXT_COLOR if self.text else (100, 95, 90)
        surf  = self.font.render(disp, True, col)
        surface.blit(surf, surf.get_rect(midleft=(self.rect.x + 10, self.rect.centery)))

        if self.active and (time.time() % 1.0) < 0.5:
            tw = self.font.size(self.text)[0]
            cx = self.rect.x + 10 + tw + 2
            pygame.draw.line(surface, TEXT_COLOR,
                             (cx, self.rect.top + 8), (cx, self.rect.bottom - 8), 2)


# ── Button helper ─────────────────────────────────────────────────────────────

def _draw_btn(surface, rect, label, font, hovered,
              base_col=(45, 40, 35), border_col=None,
              text_col=None):
    if border_col is None:
        border_col = ACCENT_COLOR if hovered else (70, 65, 60)
    if text_col is None:
        text_col = (255, 255, 255) if hovered else TEXT_COLOR
    pygame.draw.rect(surface, base_col if hovered else PANEL_COLOR,
                     rect, border_radius=8)
    pygame.draw.rect(surface, border_col, rect, 2, border_radius=8)
    s = font.render(label, True, text_col)
    surface.blit(s, s.get_rect(center=rect.center))


# ── Name prompt (shared by join & create) ────────────────────────────────────

def _prompt_name(screen, assets, clock,
                 title: str, initial: str = '') -> str | None:
    """Prompt player for their display name. Returns name or None."""
    f_title = assets.font(28, bold=True)
    f_input = assets.font(20)
    f_btn   = assets.font(20, bold=True)
    f_sm    = assets.font(14)

    cx, cy = WIDTH // 2, HEIGHT // 2
    inp    = _TextInput(pygame.Rect(cx - 150, cy - 22, 300, 44), 'Enter name…', f_input, 20)
    inp.text   = initial
    inp.active = True
    btn_ok   = pygame.Rect(cx - 80, cy + 50, 160, 42)
    btn_back = pygame.Rect(cx - 60, cy + 108, 120, 34)

    while True:
        clock.tick(60)
        mx, my = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return None
                if event.key == pygame.K_RETURN and inp.text.strip():
                    return inp.text.strip()
            inp.handle(event)
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if btn_ok.collidepoint(mx, my) and inp.text.strip():
                    return inp.text.strip()
                if btn_back.collidepoint(mx, my):
                    return None

        screen.fill(BG_COLOR)
        ts = f_title.render(title, True, ACCENT_COLOR)
        screen.blit(ts, ts.get_rect(center=(cx, cy - 80)))
        inp.draw(screen)
        _draw_btn(screen, btn_ok, 'Confirm', f_btn, btn_ok.collidepoint(mx, my))
        _draw_btn(screen, btn_back, '← Back', f_sm,
                  btn_back.collidepoint(mx, my),
                  text_col=(180, 80, 80) if btn_back.collidepoint(mx, my) else (130, 70, 70))
        pygame.display.flip()


# ── Create Room screen ────────────────────────────────────────────────────────

def _screen_create_room(screen, assets, clock) -> tuple | None:
    """
    Returns (room_name, player_name) or None if cancelled.
    """
    f_title = assets.font(30, bold=True)
    f_label = assets.font(16, bold=True)
    f_input = assets.font(20)
    f_btn   = assets.font(20, bold=True)
    f_sm    = assets.font(14)

    cx, cy  = WIDTH // 2, HEIGHT // 2
    room_in = _TextInput(pygame.Rect(cx - 180, cy - 60, 360, 42),
                         'My Ashfall Room', f_input, 30)
    name_in = _TextInput(pygame.Rect(cx - 180, cy + 10, 360, 42),
                         'Player name…', f_input, 20)
    room_in.active = True

    btn_create = pygame.Rect(cx - 130, cy + 90, 260, 46)
    btn_back   = pygame.Rect(cx - 60,  cy + 152, 120, 34)

    while True:
        clock.tick(60)
        mx, my = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return None
            room_in.handle(event)
            name_in.handle(event)
            if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                if room_in.active and room_in.text.strip():
                    name_in.active = True
                    room_in.active = False
                elif name_in.text.strip() and room_in.text.strip():
                    return room_in.text.strip(), name_in.text.strip()
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if btn_create.collidepoint(mx, my):
                    rn = room_in.text.strip() or 'My Room'
                    nn = name_in.text.strip()
                    if nn:
                        return rn, nn
                if btn_back.collidepoint(mx, my):
                    return None

        screen.fill(BG_COLOR)
        ts = f_title.render('Create Room', True, ACCENT_COLOR)
        screen.blit(ts, ts.get_rect(center=(cx, cy - 140)))

        rl = f_label.render('Room Name', True, (150, 140, 130))
        screen.blit(rl, (cx - 180, cy - 84))
        room_in.draw(screen)

        nl = f_label.render('Your Name', True, (150, 140, 130))
        screen.blit(nl, (cx - 180, cy - 14))
        name_in.draw(screen)

        can_create = bool(room_in.text.strip() and name_in.text.strip())
        hov_c = btn_create.collidepoint(mx, my)
        col_c = HEAL_COLOR if can_create else (60, 90, 60)
        pygame.draw.rect(screen, (35, 50, 35) if hov_c else PANEL_COLOR,
                         btn_create, border_radius=8)
        pygame.draw.rect(screen, col_c, btn_create, 2, border_radius=8)
        cs = f_btn.render('🎮  Create & Host', True, col_c)
        screen.blit(cs, cs.get_rect(center=btn_create.center))

        _draw_btn(screen, btn_back, '← Back', f_sm,
                  btn_back.collidepoint(mx, my),
                  text_col=(180, 80, 80) if btn_back.collidepoint(mx, my) else (130, 70, 70))
        pygame.display.flip()


# ── Waiting room (after connect, before game start) ──────────────────────────

def _screen_waiting_room(screen, assets, clock,
                          client: NetworkClient,
                          server: 'GameServer | None',
                          is_host: bool) -> bool:
    """Show lobby player list. Returns True when game starts, False on back."""
    f_title  = assets.font(28, bold=True)
    f_player = assets.font(20, bold=True)
    f_sm     = assets.font(15)
    f_xs     = assets.font(13)

    cx = WIDTH // 2
    game_started   = [False]
    disconnected   = [False]

    client.on_game_start  = lambda:   game_started.__setitem__(0, True)
    client.on_disconnect  = lambda _: disconnected.__setitem__(0, True)

    btn_ready  = pygame.Rect(cx - 110, HEIGHT - 145, 220, 44)
    btn_start  = pygame.Rect(cx - 110, HEIGHT - 88,  220, 44)
    btn_leave  = pygame.Rect(cx - 60,  HEIGHT - 34,  120, 28)

    while True:
        clock.tick(30)
        if game_started[0]:
            return True
        if disconnected[0]:
            return False

        mx, my = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if btn_ready.collidepoint(mx, my):
                    client.send_ready()
                if is_host and btn_start.collidepoint(mx, my):
                    if server:
                        server.begin_game()
                if btn_leave.collidepoint(mx, my):
                    return False

        screen.fill(BG_COLOR)

        # Title + room info
        ts = f_title.render('Waiting Room', True, ACCENT_COLOR)
        screen.blit(ts, ts.get_rect(center=(cx, 48)))

        # Subtitle (host IP)
        if is_host:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(('8.8.8.8', 80))
                ip = s.getsockname()[0]
                s.close()
            except Exception:
                ip = 'localhost'
            tip = f_xs.render(f'Broadcasting on LAN – others see this room automatically',
                               True, (120, 115, 110))
            screen.blit(tip, tip.get_rect(center=(cx, 78)))

        # Player list panel
        lobby    = client.lobby
        panel_h  = max(60, len(lobby) * 52 + 24)
        panel    = pygame.Rect(cx - 300, 100, 600, panel_h)
        pygame.draw.rect(screen, PANEL_COLOR, panel, border_radius=10)
        pygame.draw.rect(screen, (70, 65, 60), panel, 2, border_radius=10)

        hdr = f_xs.render('PLAYER', True, (130, 120, 110))
        screen.blit(hdr, (panel.x + 16, panel.y + 8))
        rhdr = f_xs.render('STATUS', True, (130, 120, 110))
        screen.blit(rhdr, rhdr.get_rect(midright=(panel.right - 16, panel.y + 16)))

        for i, p in enumerate(lobby):
            y     = panel.y + 24 + i * 52
            is_me = p['pid'] == client.my_pid
            nc    = ACCENT_COLOR if is_me else TEXT_COLOR
            prefix = '▶ ' if is_me else '   '
            ns    = f_player.render(prefix + p['name'], True, nc)
            screen.blit(ns, (panel.x + 16, y + 6))
            rc    = HEAL_COLOR if p['ready'] else (160, 80, 80)
            rl    = 'READY ✓' if p['ready'] else 'NOT READY'
            rs    = f_sm.render(rl, True, rc)
            screen.blit(rs, rs.get_rect(midright=(panel.right - 16, y + 18)))
            if i < len(lobby) - 1:
                pygame.draw.line(screen, (50, 45, 42),
                                 (panel.x + 8, y + 52 - 4),
                                 (panel.right - 8, y + 52 - 4))

        # Ping
        ping_s = f_xs.render(f'Ping: {client.ping_ms} ms', True, (100, 95, 90))
        screen.blit(ping_s, (WIDTH - 110, 10))

        # Ready button
        h1 = btn_ready.collidepoint(mx, my)
        pygame.draw.rect(screen, (42, 38, 34) if h1 else PANEL_COLOR,
                         btn_ready, border_radius=8)
        pygame.draw.rect(screen, ACCENT_COLOR, btn_ready, 2, border_radius=8)
        rs2 = f_sm.render('Toggle Ready', True, TEXT_COLOR)
        screen.blit(rs2, rs2.get_rect(center=btn_ready.center))

        # Start button (host only)
        if is_host:
            all_ready = all(p['ready'] for p in lobby) if lobby else False
            h2  = btn_start.collidepoint(mx, my)
            sc  = HEAL_COLOR if all_ready else (70, 100, 70)
            pygame.draw.rect(screen, (35, 50, 35) if h2 else PANEL_COLOR,
                             btn_start, border_radius=8)
            pygame.draw.rect(screen, sc, btn_start, 2, border_radius=8)
            ss = f_sm.render('▶  Start Game', True, sc)
            screen.blit(ss, ss.get_rect(center=btn_start.center))

        # Leave
        h3 = btn_leave.collidepoint(mx, my)
        pygame.draw.rect(screen, PANEL_COLOR, btn_leave, border_radius=6)
        pygame.draw.rect(screen, (80, 70, 60), btn_leave, 1, border_radius=6)
        ls = f_xs.render('← Leave', True, (180, 80, 80) if h3 else (120, 70, 70))
        screen.blit(ls, ls.get_rect(center=btn_leave.center))

        pygame.display.flip()


# ── Room list (main multiplayer screen) ──────────────────────────────────────

def run_lobby(screen: pygame.Surface,
              assets: AssetLoader,
              clock: pygame.time.Clock):
    """
    Entry point.  Returns (NetworkClient, GameServer|None) on success,
    or (None, None) on back / error.
    """
    f_title   = assets.font(34, bold=True)
    f_sub     = assets.font(15)
    f_room    = assets.font(20, bold=True)
    f_detail  = assets.font(14)
    f_btn     = assets.font(20, bold=True)
    f_xs      = assets.font(13)

    # Start LAN discovery
    disc = LanDiscovery()
    disc.start()

    # Remember last player name across sessions
    _last_name = ['']

    # Scroll state for room list
    scroll_y   = [0]

    btn_create  = pygame.Rect(WIDTH - 220, 90, 200, 46)
    btn_refresh = pygame.Rect(WIDTH - 220, 148, 200, 36)
    btn_back    = pygame.Rect(20, HEIGHT - 46, 120, 34)

    CARD_H   = 72
    LIST_TOP = 90
    LIST_BOT = HEIGHT - 70
    LIST_W   = WIDTH - 260
    CARD_PAD = 8

    connecting_msg = [None]  # None | 'connecting' | 'failed'

    def do_connect(room: dict, player_name: str):
        """Attempt to connect to the given room. Returns (client, None) or (None, None)."""
        client = NetworkClient(room['host'], room['port'], player_name)
        connecting_msg[0] = 'connecting'
        _render_connecting(screen, assets, room['room'])
        pygame.display.flip()

        ok = client.connect()
        if not ok:
            client.disconnect()
            connecting_msg[0] = 'failed'
            return None, None
        client.start()
        connecting_msg[0] = None
        return client, None

    def _render_connecting(surface, a, room_name):
        surface.fill(BG_COLOR)
        s = a.font(24, bold=True).render(f'Connecting to "{room_name}"…', True, ACCENT_COLOR)
        surface.blit(s, s.get_rect(center=(WIDTH // 2, HEIGHT // 2)))

    while True:
        clock.tick(30)
        rooms  = disc.get_rooms()
        mx, my = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                disc.stop()
                return None, None
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                disc.stop()
                return None, None
            if event.type == pygame.MOUSEWHEEL:
                scroll_y[0] = max(0, scroll_y[0] - event.y * 20)

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                # Create Room
                if btn_create.collidepoint(mx, my):
                    disc.stop()
                    result = _screen_create_room(screen, assets, clock)
                    if result:
                        room_name, player_name = result
                        _last_name[0] = player_name
                        try:
                            server = GameServer(port=DEFAULT_PORT,
                                                room_name=room_name,
                                                friendly_fire=True)
                            server.launch()
                        except OSError as e:
                            _show_error(screen, assets, clock, f'Port {DEFAULT_PORT} busy: {e}')
                            disc = LanDiscovery()
                            disc.start()
                            continue
                        client = NetworkClient('127.0.0.1', DEFAULT_PORT, player_name)
                        _render_connecting(screen, assets, room_name)
                        pygame.display.flip()
                        if client.connect():
                            client.start()
                            started = _screen_waiting_room(
                                screen, assets, clock, client, server, is_host=True)
                            if started:
                                return client, server
                        client.disconnect()
                        server.stop()
                    disc = LanDiscovery()
                    disc.start()

                # Refresh
                if btn_refresh.collidepoint(mx, my):
                    disc.stop()
                    disc = LanDiscovery()
                    disc.start()
                    scroll_y[0] = 0

                # Back
                if btn_back.collidepoint(mx, my):
                    disc.stop()
                    return None, None

                # Room card click
                visible_h = LIST_BOT - LIST_TOP
                for i, room in enumerate(rooms):
                    card_y = LIST_TOP + i * (CARD_H + CARD_PAD) - scroll_y[0]
                    if card_y < LIST_TOP - CARD_H or card_y > LIST_BOT:
                        continue
                    card = pygame.Rect(10, card_y, LIST_W - 10, CARD_H)
                    if card.collidepoint(mx, my) and not room.get('is_full', False):
                        name = _prompt_name(screen, assets, clock,
                                            f"Join  \"{room['room']}\"",
                                            _last_name[0])
                        if name:
                            _last_name[0] = name
                            client, _ = do_connect(room, name)
                            if client:
                                started = _screen_waiting_room(
                                    screen, assets, clock, client,
                                    server=None, is_host=False)
                                if started:
                                    disc.stop()
                                    return client, None
                                client.disconnect()
                            else:
                                _show_error(screen, assets, clock,
                                            'Could not connect. Room may be full or closed.')
                        disc.stop()
                        disc = LanDiscovery()
                        disc.start()
                        break

        # ── Draw ──────────────────────────────────────────────────────────
        screen.fill(BG_COLOR)

        # Grid
        for x in range(0, WIDTH, 64):
            pygame.draw.line(screen, (34, 30, 28), (x, 0), (x, HEIGHT))
        for y in range(0, HEIGHT, 64):
            pygame.draw.line(screen, (34, 30, 28), (0, y), (WIDTH, y))

        # Title
        ts = f_title.render('Multiplayer', True, ACCENT_COLOR)
        screen.blit(ts, (16, 18))
        sub = f_sub.render('Searching for rooms on your LAN…', True, (120, 115, 110))
        screen.blit(sub, (18, 58))

        # Right-side buttons
        h_c = btn_create.collidepoint(mx, my)
        pygame.draw.rect(screen, (42, 38, 34) if h_c else PANEL_COLOR,
                         btn_create, border_radius=8)
        pygame.draw.rect(screen, HEAL_COLOR if h_c else (70, 100, 70),
                         btn_create, 2, border_radius=8)
        cs = f_btn.render('+ Create Room', True, HEAL_COLOR if h_c else (100, 150, 100))
        screen.blit(cs, cs.get_rect(center=btn_create.center))

        h_r = btn_refresh.collidepoint(mx, my)
        pygame.draw.rect(screen, (38, 34, 30) if h_r else PANEL_COLOR,
                         btn_refresh, border_radius=6)
        pygame.draw.rect(screen, ACCENT_COLOR if h_r else (70, 65, 60),
                         btn_refresh, 2, border_radius=6)
        rs = f_detail.render('⟳  Refresh', True, TEXT_COLOR)
        screen.blit(rs, rs.get_rect(center=btn_refresh.center))

        # Divider
        pygame.draw.line(screen, (55, 50, 46), (10, 84), (LIST_W, 84))

        # Scrollable room list
        clip = pygame.Rect(10, LIST_TOP, LIST_W, LIST_BOT - LIST_TOP)
        screen.set_clip(clip)

        if not rooms:
            ems = f_room.render('No rooms found on LAN', True, (90, 85, 80))
            screen.blit(ems, ems.get_rect(center=(LIST_W // 2, (LIST_TOP + LIST_BOT) // 2)))
            hint = f_detail.render('Create a room or ask a friend to host', True, (70, 65, 62))
            screen.blit(hint, hint.get_rect(center=(LIST_W // 2,
                                                      (LIST_TOP + LIST_BOT) // 2 + 30)))
        else:
            for i, room in enumerate(rooms):
                card_y = LIST_TOP + i * (CARD_H + CARD_PAD) - scroll_y[0]
                if card_y < LIST_TOP - CARD_H or card_y > LIST_BOT:
                    continue
                card   = pygame.Rect(10, card_y, LIST_W - 10, CARD_H)
                hov    = card.collidepoint(mx, my)
                is_full = room['players'] >= room['max_players']

                # Card background
                card_bg = (40, 36, 32) if hov and not is_full else (28, 25, 22)
                pygame.draw.rect(screen, card_bg, card, border_radius=10)
                brd = ACCENT_COLOR if hov and not is_full else (55, 50, 46)
                pygame.draw.rect(screen, brd, card, 2, border_radius=10)

                # Room name
                name_col = (220, 210, 190) if not is_full else (100, 95, 90)
                ns = f_room.render(room['room'], True, name_col)
                screen.blit(ns, (card.x + 16, card.y + 10))

                # Player count badge
                cnt_text = f"{room['players']}/{room['max_players']}"
                cnt_col  = DANGER_COLOR if is_full else HEAL_COLOR
                cs2 = f_detail.render(cnt_text, True, cnt_col)
                screen.blit(cs2, (card.x + 16, card.y + 40))

                # Friendly fire tag
                ff_text = '⚔ FF ON' if room.get('friendly_fire', True) else '🛡 FF OFF'
                ff_col  = (200, 100, 80) if room.get('friendly_fire') else (80, 160, 120)
                ffs = f_detail.render(ff_text, True, ff_col)
                screen.blit(ffs, (card.x + 80, card.y + 40))

                # Join hint
                if hov and not is_full:
                    hs = f_detail.render('▶  Click to Join', True, ACCENT_COLOR)
                    screen.blit(hs, hs.get_rect(midright=(card.right - 14, card.centery)))
                elif is_full:
                    hs = f_detail.render('FULL', True, DANGER_COLOR)
                    screen.blit(hs, hs.get_rect(midright=(card.right - 14, card.centery)))

        screen.set_clip(None)

        # Separator line right of list
        pygame.draw.line(screen, (55, 50, 46),
                         (LIST_W, LIST_TOP - 4), (LIST_W, LIST_BOT))

        # Room count
        rc = f_xs.render(f'{len(rooms)} room(s) found', True, (100, 95, 90))
        screen.blit(rc, (10, LIST_BOT + 6))

        # Back button
        h_b = btn_back.collidepoint(mx, my)
        pygame.draw.rect(screen, PANEL_COLOR, btn_back, border_radius=6)
        pygame.draw.rect(screen, (80, 70, 60), btn_back, 1, border_radius=6)
        bk = f_detail.render('← Back', True, (180, 80, 80) if h_b else (120, 70, 70))
        screen.blit(bk, bk.get_rect(center=btn_back.center))

        pygame.display.flip()


def _show_error(screen, assets, clock, msg: str, dur: float = 2.5):
    f = assets.font(20, bold=True)
    t0 = time.time()
    while time.time() - t0 < dur:
        clock.tick(30)
        for ev in pygame.event.get():
            if ev.type in (pygame.QUIT, pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN):
                return
        screen.fill(BG_COLOR)
        s = f.render(msg, True, DANGER_COLOR)
        screen.blit(s, s.get_rect(center=(WIDTH // 2, HEIGHT // 2)))
        pygame.display.flip()
