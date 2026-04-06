"""
Ashfall Protocol – Entry Point
================================
Boots pygame, shows the main menu, then routes to:
  - Single Player → Game() (original, unchanged)
  - Multiplayer   → Lobby → MPClientGame
"""

import pygame
from asset_loader import AssetLoader
from config import FPS, HEIGHT, TITLE, WIDTH


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(TITLE)
    assets = AssetLoader()
    clock  = pygame.time.Clock()

    while True:
        # ── Main menu ────────────────────────────────────────────────────────
        from menu import run_main_menu
        choice = run_main_menu(screen, assets)

        if choice == 'quit':
            break

        elif choice == 'singleplayer':
            # Original game – completely unchanged
            from game import Game
            Game().run()
            # After game exits (quit / ESC from game-over), return to menu

        elif choice == 'multiplayer':
            from lobby import run_lobby
            client, server = run_lobby(screen, assets, clock)
            if client is not None:
                from mp_game import MPClientGame
                MPClientGame(client).run()
                # Clean-up after game session
                client.disconnect()
                if server is not None:
                    server.stop()

    pygame.quit()


if __name__ == '__main__':
    main()
