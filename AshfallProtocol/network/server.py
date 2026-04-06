"""
Ashfall Protocol – Authoritative Game Server
=============================================
Runs in a daemon thread.  Manages all game logic for multiplayer sessions.

Key fixes vs initial version
-----------------------------
* Bullet.color → computed via b._get_color() (Bullet has no .color attr)
* _loop wrapped in try/except so a single tick error never kills the thread
* LAN discovery: broadcasts ANNOUNCE every 2s so clients auto-detect the room
* room_name parameter added for display in the room list
"""

import json
import random
import socket
import threading
import time

import pygame

from config import WAVE_BREAK_TIME, WORLD_HEIGHT, WORLD_WIDTH, vec
from enemy import (BossEnemy, Barrel, DropItem, EnemySpawner,
                   SupportEnemy, SuicideEnemy)
from game import BARREL_POSITIONS, OBSTACLE_DEFS
from network.protocol import MsgType, pack, unpack
from player import Player

# ── Ports ───────────────────────────────────────────────────────────────────
DEFAULT_PORT      = 7654
DISCOVERY_PORT    = 7655   # UDP broadcast for LAN room discovery

# ── Tuning ──────────────────────────────────────────────────────────────────
SERVER_TICK_RATE  = 30
BROADCAST_RATE    = 20
ANNOUNCE_INTERVAL = 2.0
CLIENT_TIMEOUT    = 10.0
MAX_PLAYERS       = 8

EXPLOSIVE_RADIUS  = 80
EXPLOSIVE_DAMAGE  = 40
SHOCK_RADIUS      = 120
_TRAIT_EXPLOSIVE  = 'explosive'
_TRAIT_SHOCK      = 'shock'

_SPAWN_OFFSETS = [
    (  0, -100), ( 100,    0), (   0,  100), (-100,    0),
    (-70,  -70), (  70,  -70), (  70,   70), ( -70,   70),
]


# ─── Per-client record ──────────────────────────────────────────────────────

class ClientRecord:
    def __init__(self, pid: int, name: str, addr: tuple):
        self.pid        = pid
        self.name       = name
        self.addr       = addr
        self.player     = None
        self.last_input = {
            'up': False, 'down': False,
            'left': False, 'right': False,
            'mx': WORLD_WIDTH / 2, 'my': WORLD_HEIGHT / 2,
            'shoot': False, 'dash': False,
            'reload': False, 'weapon': 0,
        }
        self.last_seen  = time.time()
        self.ready      = False
        self.kills      = 0
        self.score      = 0


# ─── Headless game proxy ─────────────────────────────────────────────────────

class _GameProxy:
    """Duck-typed Game object required by EnemySpawner.update()."""

    def __init__(self, sg: 'ServerGame'):
        self._sg             = sg
        self.wave_break_time = WAVE_BREAK_TIME

    @property
    def enemies(self):
        return self._sg.enemies

    @property
    def player(self):
        return self._sg._nearest_player(vec(WORLD_WIDTH / 2, WORLD_HEIGHT / 2))


# ─── Server-side game state ──────────────────────────────────────────────────

class ServerGame:
    def __init__(self, clients: dict, friendly_fire: bool):
        self.clients        = clients
        self.friendly_fire  = friendly_fire
        self.obstacles      = [pygame.Rect(*d) for d in OBSTACLE_DEFS]
        self.barrels        = [Barrel(p) for p in BARREL_POSITIONS]
        self.enemies        = []
        self.bullets        = []
        self.items          = []
        self.spawner        = EnemySpawner()
        self.spawner.start_wave()
        self.kills          = 0
        self.score          = 0

        self._next_bullet_id = 0
        self._enemy_id_map   = {}
        self._next_eid       = 1

        cx, cy = WORLD_WIDTH / 2, WORLD_HEIGHT / 2
        for i, rec in enumerate(clients.values()):
            ox, oy = _SPAWN_OFFSETS[i % len(_SPAWN_OFFSETS)]
            rec.player = Player((cx + ox, cy + oy))

        self._proxy = _GameProxy(self)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _eid(self, enemy) -> int:
        oid = id(enemy)
        if oid not in self._enemy_id_map:
            self._enemy_id_map[oid] = self._next_eid
            self._next_eid += 1
        return self._enemy_id_map[oid]

    def _nearest_player(self, pos):
        best, best_dist = None, float('inf')
        for rec in self.clients.values():
            p = rec.player
            if p and not p.dead:
                d = pos.distance_to(p.pos)
                if d < best_dist:
                    best_dist = d
                    best = p
        return best

    # ── Main tick ────────────────────────────────────────────────────────────

    def tick(self, dt: float):
        world_rect = pygame.Rect(0, 0, WORLD_WIDTH, WORLD_HEIGHT)

        # 1. Apply inputs → update players
        for rec in self.clients.values():
            p = rec.player
            if not p or p.dead:
                continue
            inp = rec.last_input
            p.switch_weapon(int(inp.get('weapon', 0)))
            p.update(
                dt,
                {k: inp[k] for k in ('up', 'down', 'left', 'right')},
                vec(inp['mx'], inp['my']),
                self.obstacles,
            )
            if inp.get('reload') and not p.weapon.stats.heat_based:
                p.weapon.begin_reload()
            if (not p.weapon.stats.heat_based
                    and p.weapon.ammo <= 0
                    and not p.weapon.is_reloading):
                p.weapon.begin_reload()
            if inp.get('dash'):
                p.start_dash()
            if inp.get('shoot'):
                new_bullets = p.weapon.try_fire(p.pos, p.aim_dir)
                for b in new_bullets:
                    b._net_id    = self._next_bullet_id
                    b._owner_pid = rec.pid
                    self._next_bullet_id += 1
                self.bullets.extend(new_bullets)

        # 2. Spawner
        self.spawner.update(dt, self._proxy)

        # 3. Bullets
        for b in self.bullets:
            b.update(dt, world_rect)
        self.bullets = [b for b in self.bullets if not b.dead]

        # 4. Enemy AI
        for e in self.enemies:
            target = self._nearest_player(e.pos)
            if target is None:
                continue
            allies = self.enemies if isinstance(e, SupportEnemy) else None
            e.update(dt, target, self.obstacles, self.bullets, allies)
            if not e.dead:
                for rec in self.clients.values():
                    p = rec.player
                    if p and not p.dead and e.try_attack(p):
                        p.take_damage(e.damage)

        # 5. Items
        for item in self.items:
            item.update(dt)
        self.items = [i for i in self.items if not i.dead]

        # 6. Collision
        self._resolve_bullet_hits()
        self._resolve_barrel_hits()
        self._resolve_item_pickups()
        self._cleanup_dead_enemies()

    # ── Collision ────────────────────────────────────────────────────────────

    def _resolve_bullet_hits(self):
        for bullet in self.bullets:
            if bullet.dead:
                continue
            br = pygame.Rect(
                int(bullet.pos.x - bullet.radius),
                int(bullet.pos.y - bullet.radius),
                bullet.radius * 2, bullet.radius * 2,
            )
            if any(br.colliderect(ob) for ob in self.obstacles):
                if bullet.trait and bullet.trait.type.value == _TRAIT_EXPLOSIVE:
                    self._trigger_explosion(bullet.pos)
                bullet.dead = True
                continue

            if bullet.owner == 'player':
                for enemy in self.enemies:
                    if enemy.dead or id(enemy) in bullet.hit_enemies:
                        continue
                    if bullet.pos.distance_to(enemy.pos) <= bullet.radius + enemy.radius:
                        enemy.take_damage(bullet.damage, bullet.velocity)
                        if bullet.trait:
                            tv = bullet.trait.type.value
                            if tv in ('fire', 'ice', 'bleed'):
                                enemy.apply_status(bullet.trait)
                            elif tv == _TRAIT_SHOCK:
                                self._trigger_shock(bullet, enemy)
                            elif tv == _TRAIT_EXPLOSIVE:
                                self._trigger_explosion(bullet.pos)
                                bullet.hit_enemies.add(id(enemy))
                                bullet.dead = True
                                break
                        if bullet.pierce:
                            bullet.hit_enemies.add(id(enemy))
                        else:
                            bullet.dead = True
                            break

                if self.friendly_fire and not bullet.dead:
                    owner_pid = getattr(bullet, '_owner_pid', None)
                    for rec in self.clients.values():
                        p = rec.player
                        if not p or p.dead or rec.pid == owner_pid:
                            continue
                        if bullet.pos.distance_to(p.pos) <= bullet.radius + p.radius:
                            p.take_damage(bullet.damage)
                            bullet.dead = True
                            break

            elif bullet.owner == 'enemy':
                for rec in self.clients.values():
                    p = rec.player
                    if not p or p.dead:
                        continue
                    if bullet.pos.distance_to(p.pos) <= bullet.radius + p.radius:
                        p.take_damage(bullet.damage)
                        bullet.dead = True
                        break

    def _trigger_explosion(self, pos):
        for e in self.enemies:
            if not e.dead and e.pos.distance_to(pos) <= EXPLOSIVE_RADIUS:
                e.take_damage(EXPLOSIVE_DAMAGE, e.pos - pos)
        for rec in self.clients.values():
            p = rec.player
            if p and not p.dead and p.pos.distance_to(pos) <= EXPLOSIVE_RADIUS:
                p.take_damage(25)

    def _trigger_shock(self, bullet, hit_enemy):
        chain_dmg = int(bullet.damage * 0.5)
        closest, closest_d = None, SHOCK_RADIUS + 1
        for other in self.enemies:
            if other is hit_enemy or other.dead:
                continue
            d = hit_enemy.pos.distance_to(other.pos)
            if d <= SHOCK_RADIUS and d < closest_d:
                closest_d = d
                closest = other
        if closest:
            closest.take_damage(chain_dmg)

    def _resolve_barrel_hits(self):
        for barrel in self.barrels:
            if barrel.dead:
                continue
            for bullet in self.bullets:
                if not bullet.dead:
                    if bullet.pos.distance_to(barrel.pos) <= bullet.radius + barrel.radius:
                        barrel.take_damage(bullet.damage)
                        bullet.dead = True
            if barrel.exploded:
                for e in self.enemies:
                    if e.pos.distance_to(barrel.pos) <= 100:
                        e.take_damage(60, e.pos - barrel.pos)
                for rec in self.clients.values():
                    p = rec.player
                    if p and not p.dead and p.pos.distance_to(barrel.pos) <= 100:
                        p.take_damage(35)
        self.barrels = [b for b in self.barrels if not b.dead]

    def _resolve_item_pickups(self):
        for item in self.items:
            if item.dead:
                continue
            for rec in self.clients.values():
                p = rec.player
                if p and not p.dead:
                    if p.pos.distance_to(item.pos) <= p.radius + item.size + 4:
                        item.apply(p)
                        break

    def _cleanup_dead_enemies(self):
        alive = []
        for enemy in self.enemies:
            if enemy.dead:
                self.kills += 1
                self.score += enemy.score
                r = random.random()
                if r < 0.15:
                    self.items.append(DropItem(enemy.pos, 'hp'))
                elif r < 0.30:
                    self.items.append(DropItem(enemy.pos, 'ammo'))
                self._enemy_id_map.pop(id(enemy), None)
                if isinstance(enemy, SuicideEnemy) and enemy.exploded:
                    self._trigger_explosion(enemy.pos)
            else:
                alive.append(enemy)
        self.enemies = alive

    # ── Serialisation ────────────────────────────────────────────────────────

    def serialize_state(self, seq: int) -> dict:
        players = []
        for rec in self.clients.values():
            p = rec.player
            if not p:
                continue
            players.append({
                'pid':  rec.pid,
                'name': rec.name,
                'x':    round(p.pos.x, 1),
                'y':    round(p.pos.y, 1),
                'ax':   round(p.aim_dir.x, 3),
                'ay':   round(p.aim_dir.y, 3),
                'hp':   p.hp,
                'mhp':  p.max_hp,
                'wi':   p.weapon_idx,
                'ammo': p.weapon.ammo,
                'dead': p.dead,
                'dash': p.dash_timer > 0,
            })

        enemies = []
        for e in self.enemies:
            enemies.append({
                'id':    self._eid(e),
                'type':  type(e).__name__,
                'x':     round(e.pos.x, 1),
                'y':     round(e.pos.y, 1),
                'hp':    round(e.hp, 1),
                'mhp':   e.max_hp,
                'color': list(e.color),
                'r':     e.radius,
            })

        bullets = []
        for b in self.bullets:
            # *** FIX: Bullet has no .color attr → call _get_color() ***
            try:
                color = list(b._get_color())
            except Exception:
                color = [220, 200, 100]
            bullets.append({
                'id':    getattr(b, '_net_id', 0),
                'x':     round(b.pos.x, 1),
                'y':     round(b.pos.y, 1),
                'vx':    round(b.velocity.x, 1),
                'vy':    round(b.velocity.y, 1),
                'r':     b.radius,
                'owner': b.owner,
                'color': color,
            })

        items = []
        for item in self.items:
            items.append({
                'x':    round(item.pos.x, 1),
                'y':    round(item.pos.y, 1),
                'kind': item.kind,
            })

        barrels = []
        for barrel in self.barrels:
            barrels.append({
                'x': round(barrel.pos.x, 1),
                'y': round(barrel.pos.y, 1),
            })

        return {
            'seq':     seq,
            'players': players,
            'enemies': enemies,
            'bullets': bullets,
            'items':   items,
            'barrels': barrels,
            'wave':    self.spawner.wave,
            'score':   self.score,
            'kills':   self.kills,
        }


# ─── UDP Game Server ─────────────────────────────────────────────────────────

class GameServer:
    """
    Authoritative UDP game server.

    Usage
    -----
        srv = GameServer(port=7654, room_name='My Room')
        srv.launch()
        # … host presses Start …
        srv.begin_game()
    """

    def __init__(self, port: int = DEFAULT_PORT,
                 max_players: int = MAX_PLAYERS,
                 friendly_fire: bool = True,
                 room_name: str = 'Ashfall Room'):
        self.port          = port
        self.max_players   = max_players
        self.friendly_fire = friendly_fire
        self.room_name     = room_name

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(('0.0.0.0', port))
        self.sock.settimeout(0.001)

        # Separate broadcast socket for LAN discovery
        self._bcast_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._bcast_sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self._bcast_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        self.clients      = {}
        self.next_pid     = 1
        self.game_started = False
        self.running      = True
        self.sg           = None
        self.state_seq    = 0
        self._lock        = threading.Lock()

    # ── Public API ───────────────────────────────────────────────────────────

    def launch(self) -> threading.Thread:
        t = threading.Thread(target=self._loop, daemon=True, name='GameServer')
        t.start()
        return t

    def begin_game(self):
        with self._lock:
            if not self.game_started and self.clients:
                self._start_game()

    def get_lobby_snapshot(self) -> list:
        with self._lock:
            return [{'pid': r.pid, 'name': r.name, 'ready': r.ready}
                    for r in self.clients.values()]

    def stop(self):
        self.running = False
        try:
            self.sock.close()
        except Exception:
            pass
        try:
            self._bcast_sock.close()
        except Exception:
            pass

    # ── Internal loop ────────────────────────────────────────────────────────

    def _loop(self):
        tick_dt      = 1.0 / SERVER_TICK_RATE
        bcast_dt     = 1.0 / BROADCAST_RATE
        announce_dt  = ANNOUNCE_INTERVAL
        last_tick    = time.time()
        last_bcast   = time.time()
        last_announce = time.time()

        while self.running:
            # ── recv ────────────────────────────────────────────────────────
            try:
                data, addr = self.sock.recvfrom(65535)
                msg_type, payload = unpack(data)
                if msg_type is not None:
                    self._handle(addr, msg_type, payload)
            except (socket.timeout, BlockingIOError, OSError):
                pass
            except Exception:
                pass  # never let a bad packet crash the loop

            now = time.time()

            # ── game tick ───────────────────────────────────────────────────
            if now - last_tick >= tick_dt:
                dt = min(now - last_tick, 0.05)
                if self.game_started and self.sg:
                    try:
                        with self._lock:
                            self.sg.tick(dt)
                    except Exception as e:
                        print(f'[Server] tick error: {e}')
                last_tick = now

            # ── state broadcast ──────────────────────────────────────────────
            if now - last_bcast >= bcast_dt:
                if self.game_started and self.sg:
                    try:
                        with self._lock:
                            self._broadcast_state()
                    except Exception as e:
                        print(f'[Server] broadcast error: {e}')
                last_bcast = now

            # ── LAN discovery announce ───────────────────────────────────────
            if now - last_announce >= announce_dt:
                if not self.game_started:
                    self._send_announce()
                last_announce = now

    # ── LAN discovery ────────────────────────────────────────────────────────

    def _send_announce(self):
        """Broadcast room info so LAN clients can auto-discover this server."""
        try:
            info = json.dumps({
                'room':    self.room_name,
                'port':    self.port,
                'players': len(self.clients),
                'max':     self.max_players,
                'ff':      self.friendly_fire,
            }, separators=(',', ':')).encode()
            self._bcast_sock.sendto(info, ('<broadcast>', DISCOVERY_PORT))
        except Exception:
            pass

    # ── Message dispatch ─────────────────────────────────────────────────────

    def _handle(self, addr, msg_type: int, payload: dict):
        with self._lock:
            if msg_type == MsgType.JOIN:
                self._on_join(addr, payload)
            elif msg_type == MsgType.INPUT:
                self._on_input(addr, payload)
            elif msg_type == MsgType.READY:
                self._on_ready(addr, payload)
            elif msg_type == MsgType.PING:
                self._send(addr, MsgType.PONG, {'t': payload.get('t', 0)})
            elif msg_type == MsgType.DISCONNECT:
                self._on_disconnect(addr)

    def _send(self, addr, msg_type: int, data: dict):
        try:
            self.sock.sendto(pack(msg_type, data), addr)
        except Exception:
            pass

    def _broadcast(self, msg_type: int, data: dict):
        raw = pack(msg_type, data)
        for rec in self.clients.values():
            try:
                self.sock.sendto(raw, rec.addr)
            except Exception:
                pass

    # ── Lobby handlers ───────────────────────────────────────────────────────

    def _on_join(self, addr, payload: dict):
        if addr in self.clients:
            rec = self.clients[addr]
            self._send(addr, MsgType.WELCOME, {
                'pid': rec.pid, 'players': self._lobby_list(), 'ff': self.friendly_fire,
            })
            return
        if len(self.clients) >= self.max_players:
            self._send(addr, MsgType.DISCONNECT, {'reason': 'server_full'})
            return
        if self.game_started:
            self._send(addr, MsgType.DISCONNECT, {'reason': 'game_in_progress'})
            return
        pid  = self.next_pid
        name = str(payload.get('name', f'Player{pid}'))[:20]
        self.next_pid += 1
        rec  = ClientRecord(pid, name, addr)
        self.clients[addr] = rec
        self._send(addr, MsgType.WELCOME, {
            'pid': pid, 'players': self._lobby_list(), 'ff': self.friendly_fire,
        })
        self._broadcast(MsgType.LOBBY, {'players': self._lobby_list()})

    def _on_input(self, addr, payload: dict):
        rec = self.clients.get(addr)
        if rec:
            rec.last_seen = time.time()
            rec.last_input.update(payload)

    def _on_ready(self, addr, payload: dict):
        rec = self.clients.get(addr)
        if rec:
            rec.ready = not rec.ready
            self._broadcast(MsgType.LOBBY, {'players': self._lobby_list()})

    def _on_disconnect(self, addr):
        self.clients.pop(addr, None)
        self._broadcast(MsgType.LOBBY, {'players': self._lobby_list()})

    def _start_game(self):
        self.game_started = True
        self.sg = ServerGame(self.clients, self.friendly_fire)
        self._broadcast(MsgType.START, {})

    def _broadcast_state(self):
        state = self.sg.serialize_state(self.state_seq)
        self.state_seq += 1
        self._broadcast(MsgType.STATE, state)

    def _lobby_list(self) -> list:
        return [{'pid': r.pid, 'name': r.name, 'ready': r.ready}
                for r in self.clients.values()]
