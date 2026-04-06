"""
Ashfall Protocol – LAN Room Discovery
======================================
Listens for UDP broadcast ANNOUNCE packets from GameServer instances on the
local network.  No IP configuration needed – just scan and show the list.

Usage
-----
    disc = LanDiscovery()
    disc.start()
    # in UI loop:
    rooms = disc.get_rooms()   # list of room dicts, auto-expires stale entries
    disc.stop()
"""

import json
import socket
import threading
import time

DISCOVERY_PORT   = 7655
ROOM_TTL         = 6.0    # remove room if not seen for N seconds


class RoomInfo:
    __slots__ = ('room', 'host', 'port', 'players', 'max_players',
                 'friendly_fire', 'last_seen', 'ping_ms')

    def __init__(self, data: dict, host_ip: str):
        self.room          = data.get('room', 'Unnamed Room')
        self.host          = host_ip
        self.port          = int(data.get('port', 7654))
        self.players       = int(data.get('players', 0))
        self.max_players   = int(data.get('max', 8))
        self.friendly_fire = bool(data.get('ff', True))
        self.last_seen     = time.time()
        self.ping_ms       = -1   # measured separately

    def key(self):
        return (self.host, self.port)

    def is_full(self):
        return self.players >= self.max_players

    def as_dict(self) -> dict:
        return {
            'room':          self.room,
            'host':          self.host,
            'port':          self.port,
            'players':       self.players,
            'max_players':   self.max_players,
            'friendly_fire': self.friendly_fire,
            'ping_ms':       self.ping_ms,
        }


class LanDiscovery:
    """
    Passive LAN room scanner.
    Binds to DISCOVERY_PORT and collects ANNOUNCE broadcasts from GameServers.
    """

    def __init__(self):
        self._rooms: dict = {}   # key → RoomInfo
        self._lock        = threading.Lock()
        self._running     = False
        self._thread      = None
        self._sock        = None

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def start(self):
        if self._running:
            return
        self._running = True
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            # Bind to broadcast-receive (empty string = all interfaces)
            self._sock.bind(('', DISCOVERY_PORT))
            self._sock.settimeout(0.5)
        except OSError as e:
            print(f'[Discovery] Could not bind port {DISCOVERY_PORT}: {e}')
            self._running = False
            return
        self._thread = threading.Thread(
            target=self._listen, daemon=True, name='LanDiscovery')
        self._thread.start()

    def stop(self):
        self._running = False
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass

    def get_rooms(self) -> list:
        """Return list of currently known rooms (stale entries auto-removed)."""
        now = time.time()
        with self._lock:
            stale = [k for k, r in self._rooms.items()
                     if now - r.last_seen > ROOM_TTL]
            for k in stale:
                del self._rooms[k]
            return [r.as_dict() for r in self._rooms.values()]

    # ── Background listener ──────────────────────────────────────────────────

    def _listen(self):
        while self._running:
            try:
                data, (host_ip, _) = self._sock.recvfrom(1024)
                info = json.loads(data.decode('utf-8'))
                room = RoomInfo(info, host_ip)
                with self._lock:
                    self._rooms[room.key()] = room
            except socket.timeout:
                pass
            except (json.JSONDecodeError, KeyError, ValueError):
                pass
            except OSError:
                break   # socket closed
            except Exception:
                pass
