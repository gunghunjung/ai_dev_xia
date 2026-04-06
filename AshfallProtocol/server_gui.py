"""
Ashfall Protocol – Server Manager GUI
=======================================
Standalone graphical server manager.
Build → AshfallProtocol_Server.exe

Panels
------
  ┌──────────────────────────────────────────┐
  │ ASHFALL PROTOCOL  SERVER MANAGER         │
  ├──── Config ──────────────────────────────┤
  │  Room Name  [________________]           │
  │  Port [7654]  Max [8]  FF [ON]           │
  │  [ ▶ Start Server ]                      │
  ├──── Status ──────────────────────────────┤
  │  ● ONLINE  2 / 8 players                 │
  ├──── Players ─────────────────────────────┤
  │  PID  Name            Ready   [Kick]     │
  │  1    Alice           READY   [✕]        │
  │  2    Bob             –       [✕]        │
  │  [ ▶ Start Game ]  [ ■ Stop Server ]     │
  ├──── Log ─────────────────────────────────┤
  │  [12:00] Server started on port 7654     │
  │  [12:01] Alice joined                    │
  └──────────────────────────────────────────┘
"""

import sys
import time
import threading
from datetime import datetime

import pygame

# ── Window constants ──────────────────────────────────────────────────────────
W, H      = 700, 660
FPS       = 30
TITLE     = 'Ashfall Protocol – Server Manager'

# ── Colours (dark theme matching the game) ────────────────────────────────────
C_BG      = (18, 16, 14)
C_PANEL   = (28, 25, 22)
C_BORDER  = (55, 50, 45)
C_TEXT    = (230, 222, 205)
C_DIM     = (120, 115, 108)
C_ACCENT  = (255, 179, 71)
C_GREEN   = (73, 208, 130)
C_RED     = (219, 71, 71)
C_ORANGE  = (230, 120, 50)

# ── Section heights ───────────────────────────────────────────────────────────
HDR_H     = 52
CFG_H     = 148
STA_H     = 38
PLY_H     = 200
LOG_H     = H - HDR_H - CFG_H - STA_H - PLY_H   # remaining

CFG_Y     = HDR_H
STA_Y     = CFG_Y + CFG_H
PLY_Y     = STA_Y + STA_H
LOG_Y     = PLY_Y + PLY_H


# ─── Text input ──────────────────────────────────────────────────────────────

class _Input:
    def __init__(self, rect, text, font, max_len=40, numeric=False):
        self.rect    = rect
        self.text    = text
        self.font    = font
        self.max_len = max_len
        self.numeric = numeric
        self.active  = False

    def handle(self, ev) -> bool:
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            self.active = self.rect.collidepoint(ev.pos)
            return self.active
        if not self.active:
            return False
        if ev.type == pygame.KEYDOWN:
            if ev.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            elif ev.key not in (pygame.K_RETURN, pygame.K_ESCAPE, pygame.K_TAB):
                ch = ev.unicode
                if len(self.text) < self.max_len and ch.isprintable():
                    if self.numeric and not ch.isdigit():
                        pass
                    else:
                        self.text += ch
            return True
        return False

    def int_value(self, default):
        try:
            return int(self.text)
        except ValueError:
            return default

    def draw(self, surface):
        bg  = (38, 34, 30) if self.active else C_PANEL
        brd = C_ACCENT if self.active else C_BORDER
        pygame.draw.rect(surface, bg,  self.rect, border_radius=5)
        pygame.draw.rect(surface, brd, self.rect, 2, border_radius=5)
        t = self.font.render(self.text or ' ', True, C_TEXT)
        surface.blit(t, t.get_rect(midleft=(self.rect.x + 8, self.rect.centery)))
        if self.active and (time.time() % 1.0) < 0.5:
            tw = self.font.size(self.text)[0]
            cx = self.rect.x + 8 + tw + 1
            pygame.draw.line(surface, C_TEXT,
                             (cx, self.rect.top + 6), (cx, self.rect.bottom - 6), 2)


# ─── Toggle button ────────────────────────────────────────────────────────────

class _Toggle:
    def __init__(self, rect, font, value=True):
        self.rect  = rect
        self.font  = font
        self.value = value

    def handle(self, ev) -> bool:
        if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
            if self.rect.collidepoint(ev.pos):
                self.value = not self.value
                return True
        return False

    def draw(self, surface):
        col = C_GREEN if self.value else C_RED
        pygame.draw.rect(surface, col, self.rect, border_radius=5)
        lbl = 'FF  ON' if self.value else 'FF OFF'
        s = self.font.render(lbl, True, (255, 255, 255))
        surface.blit(s, s.get_rect(center=self.rect.center))


# ─── Button ───────────────────────────────────────────────────────────────────

def _btn(surface, rect, label, font, enabled=True, color=None, mx=0, my=0):
    if color is None:
        color = C_ACCENT
    hov = rect.collidepoint(mx, my) and enabled
    bg  = (50, 44, 38) if hov else C_PANEL
    brd = color if enabled else C_BORDER
    pygame.draw.rect(surface, bg,  rect, border_radius=7)
    pygame.draw.rect(surface, brd, rect, 2, border_radius=7)
    tc  = color if enabled else C_DIM
    s   = font.render(label, True, tc)
    surface.blit(s, s.get_rect(center=rect.center))
    return hov


def _clicked(ev, rect):
    return (ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1
            and rect.collidepoint(ev.pos))


# ─── Section divider ─────────────────────────────────────────────────────────

def _section(surface, y, h, label, font_sm):
    r = pygame.Rect(0, y, W, h)
    pygame.draw.rect(surface, C_PANEL, r)
    pygame.draw.line(surface, C_BORDER, (0, y), (W, y))
    pygame.draw.line(surface, C_BORDER, (0, y + h - 1), (W, y + h - 1))
    ls = font_sm.render(f'  {label}', True, C_DIM)
    surface.blit(ls, (10, y + 5))


# ─── Log panel ────────────────────────────────────────────────────────────────

class _Log:
    MAX = 200

    def __init__(self):
        self._lines  = []
        self._scroll = 0
        self._lock   = threading.Lock()

    def add(self, msg: str):
        ts = datetime.now().strftime('%H:%M:%S')
        with self._lock:
            self._lines.append(f'[{ts}] {msg}')
            if len(self._lines) > self.MAX:
                self._lines.pop(0)
            self._scroll = 0   # auto-scroll to bottom

    def scroll(self, delta):
        with self._lock:
            self._scroll = max(0, self._scroll + delta)

    def draw(self, surface, rect, font):
        pygame.draw.rect(surface, C_BG, rect)
        with self._lock:
            lines = self._lines[:]
        line_h = font.get_linesize()
        visible = (rect.height - 4) // line_h
        start   = max(0, len(lines) - visible - self._scroll)
        end     = start + visible
        for i, line in enumerate(lines[start:end]):
            col = C_GREEN if 'joined' in line else \
                  C_RED   if 'kicked' in line or 'ERROR' in line else \
                  C_ORANGE if 'started' in line or 'stopped' in line else \
                  C_DIM
            s = font.render(line, True, col)
            surface.blit(s, (rect.x + 6, rect.y + 4 + i * line_h))


# ─── Main server GUI ──────────────────────────────────────────────────────────

def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption(TITLE)
    clock  = pygame.time.Clock()

    def make_font(size, bold=False):
        return pygame.font.SysFont(None, size, bold=bold)

    f_title = make_font(28, bold=True)
    f_head  = make_font(18, bold=True)
    f_body  = make_font(17)
    f_sm    = make_font(15)
    f_log   = make_font(14)

    # ── Widgets ──────────────────────────────────────────────────────────────
    pad = 14

    inp_room = _Input(pygame.Rect(pad + 90, CFG_Y + 14, 340, 32),
                      'Ashfall Server', f_body, max_len=30)
    inp_port = _Input(pygame.Rect(pad + 52, CFG_Y + 60, 80, 30),
                      '7654', f_body, max_len=5, numeric=True)
    inp_max  = _Input(pygame.Rect(pad + 220, CFG_Y + 60, 50, 30),
                      '8', f_body, max_len=2, numeric=True)
    tog_ff   = _Toggle(pygame.Rect(pad + 320, CFG_Y + 60, 80, 30), f_sm, value=True)

    btn_start_srv  = pygame.Rect(pad, CFG_Y + 106, 180, 34)
    btn_start_game = pygame.Rect(pad, PLY_Y + PLY_H - 44, 180, 34)
    btn_stop_srv   = pygame.Rect(pad + 194, PLY_Y + PLY_H - 44, 180, 34)

    log = _Log()

    # ── Server state ─────────────────────────────────────────────────────────
    server      = None
    srv_running = False

    def start_server():
        nonlocal server, srv_running
        from network.server import GameServer, DEFAULT_PORT
        port     = inp_port.int_value(DEFAULT_PORT)
        max_p    = max(1, min(8, inp_max.int_value(8)))
        room     = inp_room.text.strip() or 'Ashfall Server'
        try:
            server = GameServer(port=port, max_players=max_p,
                                friendly_fire=tog_ff.value,
                                room_name=room)
            server._log_callback = log.add   # optional hook
            server.launch()
            srv_running = True
            log.add(f'Server started  ·  Room: "{room}"  ·  Port: {port}')
        except OSError as e:
            log.add(f'ERROR: cannot bind port {port}: {e}')

    def stop_server():
        nonlocal server, srv_running
        if server:
            server.stop()
            server = None
        srv_running = False
        log.add('Server stopped.')

    def do_start_game():
        if server and srv_running:
            server.begin_game()
            log.add('Game started!')

    def do_kick(pid):
        if not server:
            return
        with server._lock:
            target = next(
                (r for r in server.clients.values() if r.pid == pid), None)
            if target:
                name = target.name
                server._on_disconnect(target.addr)
                log.add(f'Kicked PID {pid} ({name})')

    # ── Player list scroll ────────────────────────────────────────────────────
    ply_scroll = [0]
    PLY_ROW_H  = 36
    PLY_LIST_Y = PLY_Y + 26
    PLY_LIST_H = PLY_H - 70

    # Track kick-button rects per frame
    kick_rects = {}

    # ── Main loop ─────────────────────────────────────────────────────────────
    running = True
    while running:
        clock.tick(FPS)
        mx, my = pygame.mouse.get_pos()

        # Read server state
        players = server.get_lobby_snapshot() if srv_running and server else []
        n_ply   = len(players)
        max_ply = server.max_players if server else int(inp_max.int_value(8))

        # ── Events ───────────────────────────────────────────────────────────
        kick_rects_this = {}
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False

            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                running = False

            if ev.type == pygame.MOUSEWHEEL:
                if LOG_Y < my < H:
                    log.scroll(-ev.y)
                if PLY_LIST_Y < my < PLY_LIST_Y + PLY_LIST_H:
                    ply_scroll[0] = max(0, ply_scroll[0] - ev.y)

            if not srv_running:
                inp_room.handle(ev)
                inp_port.handle(ev)
                inp_max.handle(ev)
                tog_ff.handle(ev)
                if _clicked(ev, btn_start_srv):
                    start_server()
            else:
                if _clicked(ev, btn_stop_srv):
                    stop_server()
                if _clicked(ev, btn_start_game) and not (server and server.game_started):
                    do_start_game()
                # Kick buttons (built per frame below)
                for pid, rect in kick_rects.items():
                    if _clicked(ev, rect):
                        do_kick(pid)

        # ── Draw ─────────────────────────────────────────────────────────────
        screen.fill(C_BG)

        # Header
        hdr = pygame.Rect(0, 0, W, HDR_H)
        pygame.draw.rect(screen, (22, 20, 18), hdr)
        pygame.draw.line(screen, C_BORDER, (0, HDR_H - 1), (W, HDR_H - 1))
        ts = f_title.render('ASHFALL PROTOCOL', True, C_ACCENT)
        screen.blit(ts, (pad, 12))
        sm = f_sm.render('Server Manager', True, C_DIM)
        screen.blit(sm, sm.get_rect(midright=(W - pad, HDR_H // 2)))

        # ── Config section ────────────────────────────────────────────────────
        _section(screen, CFG_Y, CFG_H, 'CONFIGURATION', f_sm)

        if not srv_running:
            # Room name
            rl = f_sm.render('Room Name', True, C_DIM)
            screen.blit(rl, (pad, CFG_Y + 22))
            inp_room.draw(screen)
            # Port
            pl = f_sm.render('Port', True, C_DIM)
            screen.blit(pl, (pad, CFG_Y + 68))
            inp_port.draw(screen)
            # Max
            ml = f_sm.render('Max', True, C_DIM)
            screen.blit(ml, (pad + 143, CFG_Y + 68))
            inp_max.draw(screen)
            # FF toggle
            tog_ff.draw(screen)
            # Start server button
            _btn(screen, btn_start_srv, '▶  Start Server', f_body,
                 enabled=True, color=C_GREEN, mx=mx, my=my)
        else:
            # Server is running – show read-only config
            info = (f'Room: {server.room_name if server else "–"}   '
                    f'Port: {server.port if server else "–"}   '
                    f'FF: {"ON" if (server and server.friendly_fire) else "OFF"}')
            is_  = f_body.render(info, True, C_TEXT)
            screen.blit(is_, (pad, CFG_Y + 40))
            hint = f_sm.render('Stop the server to change settings.', True, C_DIM)
            screen.blit(hint, (pad, CFG_Y + 75))

        # ── Status bar ────────────────────────────────────────────────────────
        sta_r = pygame.Rect(0, STA_Y, W, STA_H)
        pygame.draw.rect(screen, (24, 22, 20), sta_r)
        pygame.draw.line(screen, C_BORDER, (0, STA_Y), (W, STA_Y))

        if srv_running:
            dot_col = C_GREEN if not (server and server.game_started) else C_ORANGE
            dot_lbl = ('● ONLINE' if not (server and server.game_started)
                       else '● GAME IN PROGRESS')
            dot_s = f_head.render(dot_lbl, True, dot_col)
            screen.blit(dot_s, (pad, STA_Y + 10))
            cnt_s = f_body.render(f'{n_ply} / {max_ply} players', True, C_TEXT)
            screen.blit(cnt_s, cnt_s.get_rect(midright=(W - pad, STA_Y + STA_H // 2)))
        else:
            off_s = f_head.render('○ OFFLINE', True, C_DIM)
            screen.blit(off_s, (pad, STA_Y + 10))

        # ── Players section ───────────────────────────────────────────────────
        _section(screen, PLY_Y, PLY_H, f'PLAYERS  ({n_ply} / {max_ply})', f_sm)

        # Column headers
        hdr_y = PLY_Y + 20
        screen.blit(f_sm.render('PID', True, C_DIM), (pad + 2, hdr_y))
        screen.blit(f_sm.render('Name', True, C_DIM), (pad + 52, hdr_y))
        screen.blit(f_sm.render('Status', True, C_DIM), (pad + 280, hdr_y))

        pygame.draw.line(screen, C_BORDER,
                         (pad, hdr_y + 18), (W - pad, hdr_y + 18))

        # Player rows
        list_rect = pygame.Rect(0, PLY_LIST_Y, W, PLY_LIST_H)
        screen.set_clip(list_rect)

        kick_rects_this = {}
        visible_rows = PLY_LIST_H // PLY_ROW_H
        start_idx = min(ply_scroll[0], max(0, n_ply - visible_rows))
        ply_scroll[0] = start_idx

        for i, p in enumerate(players[start_idx:start_idx + visible_rows + 1]):
            ry = PLY_LIST_Y + i * PLY_ROW_H

            # Row background (alternate)
            if i % 2 == 0:
                pygame.draw.rect(screen, (26, 23, 20),
                                 (0, ry, W, PLY_ROW_H))

            # PID
            pid_s = f_body.render(str(p['pid']), True, C_ACCENT)
            screen.blit(pid_s, (pad + 4, ry + 8))

            # Name
            name_s = f_body.render(p['name'][:22], True, C_TEXT)
            screen.blit(name_s, (pad + 52, ry + 8))

            # Ready status
            ready   = p.get('ready', False)
            rc      = C_GREEN if ready else C_DIM
            rl_text = 'READY ✓' if ready else '–'
            rs      = f_sm.render(rl_text, True, rc)
            screen.blit(rs, (pad + 280, ry + 10))

            # Kick button
            if srv_running:
                kick_r = pygame.Rect(W - pad - 54, ry + 6, 50, 24)
                kick_hov = kick_r.collidepoint(mx, my)
                pygame.draw.rect(screen, (55, 25, 25) if kick_hov else C_PANEL,
                                 kick_r, border_radius=5)
                pygame.draw.rect(screen, C_RED if kick_hov else (80, 40, 40),
                                 kick_r, 2, border_radius=5)
                ks = f_sm.render('Kick', True, C_RED if kick_hov else (160, 80, 80))
                screen.blit(ks, ks.get_rect(center=kick_r.center))
                kick_rects_this[p['pid']] = kick_r

        kick_rects = kick_rects_this
        screen.set_clip(None)

        # Scroll indicator
        if n_ply > visible_rows:
            pygame.draw.circle(screen, C_DIM,
                               (W - 8, PLY_LIST_Y + PLY_LIST_H // 2), 4)

        # Control buttons (bottom of players section)
        if srv_running:
            game_started = server.game_started if server else False
            _btn(screen, btn_start_game,
                 '▶  Start Game', f_body,
                 enabled=(n_ply > 0 and not game_started),
                 color=C_GREEN, mx=mx, my=my)
            _btn(screen, btn_stop_srv,
                 '■  Stop Server', f_body,
                 enabled=True, color=C_RED, mx=mx, my=my)

        # ── Log section ───────────────────────────────────────────────────────
        pygame.draw.line(screen, C_BORDER, (0, LOG_Y), (W, LOG_Y))
        log_label = f_sm.render('  LOG', True, C_DIM)
        screen.blit(log_label, (0, LOG_Y + 4))
        log_rect = pygame.Rect(0, LOG_Y + 22, W, H - LOG_Y - 22)
        log.draw(screen, log_rect, f_log)

        pygame.display.flip()

    # Cleanup
    if server:
        server.stop()
    pygame.quit()


if __name__ == '__main__':
    main()
