"""
Ashfall Protocol – Dedicated Server
=====================================
Run this on any machine (PC, VM, cloud) to host a persistent server.
Players connect automatically via LAN discovery – no IP sharing needed for LAN.

Usage
-----
    python server_standalone.py
    python server_standalone.py --room "My Clan Server" --port 7654 --max 8
    python server_standalone.py --no-ff          # disable friendly fire

Commands (interactive, type & press Enter)
------------------------------------------
    start   – start the game immediately
    status  – print current player list
    kick N  – kick player with PID N
    quit    – shut down the server

Requirements
------------
    pip install pygame     (for pygame.Vector2 used in game logic)
"""

import argparse
import sys
import time
import threading


def _parse_args():
    p = argparse.ArgumentParser(
        description='Ashfall Protocol Dedicated Server',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--room',    default='Ashfall Server', help='Room name shown in room list')
    p.add_argument('--port',    type=int, default=7654,   help='UDP port')
    p.add_argument('--max',     type=int, default=8,      help='Max players')
    p.add_argument('--no-ff',   action='store_true',      help='Disable friendly fire')
    return p.parse_args()


def _print_status(server):
    players = server.get_lobby_snapshot()
    if not players:
        print('[Server] No players connected.')
    else:
        print(f'[Server] {len(players)} player(s):')
        for p in players:
            ready = 'READY' if p['ready'] else 'not ready'
            print(f'         PID {p["pid"]:>2}  {p["name"]:<20}  {ready}')


def _command_loop(server):
    """Interactive command loop – runs in the main thread."""
    print('[Server] Commands: start | status | quit')
    while server.running:
        try:
            line = input().strip().lower()
        except (EOFError, KeyboardInterrupt):
            break

        if not line:
            continue
        if line == 'start':
            if server.game_started:
                print('[Server] Game already started.')
            else:
                players = server.get_lobby_snapshot()
                if not players:
                    print('[Server] No players connected yet.')
                else:
                    server.begin_game()
                    print(f'[Server] Game started with {len(players)} player(s)!')
        elif line == 'status':
            _print_status(server)
        elif line.startswith('kick '):
            try:
                pid = int(line.split()[1])
                with server._lock:
                    target = next(
                        (r for r in server.clients.values() if r.pid == pid), None)
                    if target:
                        server._on_disconnect(target.addr)
                        print(f'[Server] Kicked PID {pid}.')
                    else:
                        print(f'[Server] No player with PID {pid}.')
            except (IndexError, ValueError):
                print('[Server] Usage: kick <pid>')
        elif line in ('quit', 'exit', 'q'):
            break
        else:
            print('[Server] Unknown command.')

    server.stop()
    print('[Server] Stopped.')


def main():
    args = _parse_args()

    # pygame must be initialised for pygame.Vector2 used in game logic
    import os
    os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
    os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
    import pygame
    pygame.init()

    from network.server import GameServer

    try:
        server = GameServer(
            port          = args.port,
            max_players   = args.max,
            friendly_fire = not args.no_ff,
            room_name     = args.room,
        )
    except OSError as e:
        print(f'[Server] ERROR: Cannot bind port {args.port}: {e}')
        sys.exit(1)

    server.launch()

    print('=' * 50)
    print(' Ashfall Protocol – Dedicated Server')
    print('=' * 50)
    print(f' Room:          {args.room}')
    print(f' Port:          {args.port}')
    print(f' Max players:   {args.max}')
    print(f' Friendly fire: {not args.no_ff}')
    print('=' * 50)
    print(' Players on the same LAN will see this room')
    print(' automatically in the Multiplayer screen.')
    print('=' * 50)

    # Status ticker in background
    def _ticker():
        while server.running:
            time.sleep(10)
            players = server.get_lobby_snapshot()
            if players:
                print(f'[Server] {len(players)} player(s) connected.')
    threading.Thread(target=_ticker, daemon=True).start()

    _command_loop(server)


if __name__ == '__main__':
    main()
