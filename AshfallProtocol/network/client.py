"""
Ashfall Protocol – Network Client
===================================
Handles all UDP communication between a game client and the server.
Runs a background receiver thread so the main game loop is never blocked.

Usage
-----
    client = NetworkClient('192.168.1.10', 7654, 'Alice')
    client.connect()         # sends JOIN, waits for WELCOME
    client.start()           # starts background recv thread

    # In game loop:
    client.send_input(input_dict)
    state = client.latest_state()   # latest STATE snapshot or None
    events = client.drain_events()  # list of EVENT payloads since last call

    client.disconnect()
"""

import socket
import threading
import time

from network.protocol import MsgType, pack, unpack

CONNECT_TIMEOUT  = 5.0    # sec – max wait for WELCOME
RECV_BUFFER      = 65535
PING_INTERVAL    = 2.0    # sec


class NetworkClient:
    """UDP client for AshfallProtocol multiplayer."""

    def __init__(self, host: str, port: int, player_name: str):
        self.host        = host
        self.port        = port
        self.name        = player_name
        self.server_addr = (host, port)

        self.sock        = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(0.001)

        # State filled in after WELCOME
        self.my_pid      = None          # int
        self.lobby       = []            # list of {pid, name, ready}
        self.friendly_fire = True

        # Game runtime state
        self._state_lock  = threading.Lock()
        self._latest_state = None        # last STATE payload
        self._prev_state   = None        # second-to-last STATE (for interpolation)
        self._state_time   = 0.0         # monotonic time when latest_state arrived

        self._events_lock = threading.Lock()
        self._events      = []           # buffered EVENT payloads

        # Lobby callbacks
        self.on_lobby_update = None      # callable(lobby_list)
        self.on_game_start   = None      # callable()
        self.on_disconnect   = None      # callable(reason)

        self.connected   = False
        self.game_active = False
        self.running     = False
        self._thread     = None
        self._ping_sent  = 0.0
        self.ping_ms     = 0

    # ── Connection lifecycle ─────────────────────────────────────────────────

    def connect(self) -> bool:
        """
        Send JOIN and wait up to CONNECT_TIMEOUT seconds for WELCOME.
        Returns True on success, False on timeout.
        """
        self._send(MsgType.JOIN, {'name': self.name})
        deadline = time.time() + CONNECT_TIMEOUT
        self.sock.settimeout(0.1)
        while time.time() < deadline:
            try:
                data, _ = self.sock.recvfrom(RECV_BUFFER)
                msg_type, payload = unpack(data)
                if msg_type == MsgType.WELCOME:
                    self.my_pid       = payload['pid']
                    self.lobby        = payload.get('players', [])
                    self.friendly_fire = payload.get('ff', True)
                    self.connected    = True
                    self.sock.settimeout(0.001)
                    return True
                elif msg_type == MsgType.DISCONNECT:
                    return False
            except socket.timeout:
                # Retry JOIN
                self._send(MsgType.JOIN, {'name': self.name})
        return False

    def start(self):
        """Start background receive thread (call after connect())."""
        self.running = True
        self._thread = threading.Thread(
            target=self._recv_loop, daemon=True, name='NetworkClient')
        self._thread.start()

    def disconnect(self):
        """Gracefully close the connection."""
        self.running = False
        if self.connected:
            self._send(MsgType.DISCONNECT, {})
        try:
            self.sock.close()
        except Exception:
            pass
        self.connected = False

    # ── Game-loop API ────────────────────────────────────────────────────────

    def send_input(self, inp: dict):
        """Send player input to server.  Call every frame."""
        self._send(MsgType.INPUT, inp)

    def send_ready(self):
        """Toggle ready state in lobby."""
        self._send(MsgType.READY, {})

    def send_start(self):
        """Host requests game start (server ignores if not host by default)."""
        # The server's begin_game() is called externally for the host process.
        # This is a no-op for remote clients; kept for protocol symmetry.
        pass

    def latest_state(self):
        """Return (prev_state, latest_state, state_time) for interpolation."""
        with self._state_lock:
            return self._prev_state, self._latest_state, self._state_time

    def drain_events(self) -> list:
        """Return and clear buffered EVENT payloads."""
        with self._events_lock:
            evts = self._events[:]
            self._events.clear()
            return evts

    # ── Background recv loop ─────────────────────────────────────────────────

    def _recv_loop(self):
        while self.running:
            self._recv_once()
            self._maybe_ping()

    def _recv_once(self):
        try:
            data, _ = self.sock.recvfrom(RECV_BUFFER)
            msg_type, payload = unpack(data)
            if msg_type is None:
                return
            self._dispatch(msg_type, payload)
        except (socket.timeout, BlockingIOError, OSError):
            pass
        except Exception:
            pass

    def _dispatch(self, msg_type: int, payload: dict):
        if msg_type == MsgType.STATE:
            with self._state_lock:
                self._prev_state   = self._latest_state
                self._latest_state = payload
                self._state_time   = time.monotonic()

        elif msg_type == MsgType.LOBBY:
            self.lobby = payload.get('players', [])
            if self.on_lobby_update:
                self.on_lobby_update(self.lobby)

        elif msg_type == MsgType.START:
            self.game_active = True
            if self.on_game_start:
                self.on_game_start()

        elif msg_type == MsgType.EVENT:
            with self._events_lock:
                self._events.append(payload)

        elif msg_type == MsgType.PONG:
            sent_t = payload.get('t', 0)
            self.ping_ms = int((time.time() - sent_t) * 1000)

        elif msg_type == MsgType.DISCONNECT:
            self.connected = False
            reason = payload.get('reason', 'unknown')
            if self.on_disconnect:
                self.on_disconnect(reason)

    def _maybe_ping(self):
        now = time.time()
        if now - self._ping_sent >= PING_INTERVAL:
            self._send(MsgType.PING, {'t': now})
            self._ping_sent = now

    def _send(self, msg_type: int, data: dict):
        try:
            self.sock.sendto(pack(msg_type, data), self.server_addr)
        except Exception:
            pass
