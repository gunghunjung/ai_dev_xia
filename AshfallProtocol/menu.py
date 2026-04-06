"""
Ashfall Protocol – Main Menu
==============================
Shown at startup.  Player selects Single Player or Multiplayer.
Returns a string: 'singleplayer' | 'multiplayer' | 'quit'
"""

import pygame
from config import (ACCENT_COLOR, BG_COLOR, DANGER_COLOR, PANEL_COLOR,
                    TEXT_COLOR, WIDTH, HEIGHT, TITLE)
from asset_loader import AssetLoader


def run_main_menu(screen: pygame.Surface, assets: AssetLoader) -> str:
    """
    Block until the player makes a selection.
    Returns 'singleplayer', 'multiplayer', or 'quit'.
    """
    clock   = pygame.time.Clock()
    font_t  = assets.font(52, bold=True)
    font_s  = assets.font(28, bold=True)
    font_sm = assets.font(16, bold=False)

    buttons = [
        ('singleplayer', 'Single Player',   (WIDTH // 2, HEIGHT // 2 - 30)),
        ('multiplayer',  'Multiplayer',      (WIDTH // 2, HEIGHT // 2 + 50)),
        ('quit',         'Quit',             (WIDTH // 2, HEIGHT // 2 + 130)),
    ]
    btn_w, btn_h = 320, 48

    while True:
        clock.tick(60)
        mx, my = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return 'quit'
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return 'quit'
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for action, label, (bx, by) in buttons:
                    rect = pygame.Rect(bx - btn_w // 2, by - btn_h // 2, btn_w, btn_h)
                    if rect.collidepoint(mx, my):
                        return action

        # ── Draw ──────────────────────────────────────────────────────────
        screen.fill(BG_COLOR)

        # Decorative grid
        for x in range(0, WIDTH, 64):
            pygame.draw.line(screen, (38, 34, 30), (x, 0), (x, HEIGHT))
        for y in range(0, HEIGHT, 64):
            pygame.draw.line(screen, (38, 34, 30), (0, y), (WIDTH, y))

        # Title
        title_surf = font_t.render(TITLE, True, ACCENT_COLOR)
        screen.blit(title_surf, title_surf.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 150)))

        subtitle = font_sm.render('Select Game Mode', True, (150, 140, 130))
        screen.blit(subtitle, subtitle.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 90)))

        # Buttons
        for action, label, (bx, by) in buttons:
            rect = pygame.Rect(bx - btn_w // 2, by - btn_h // 2, btn_w, btn_h)
            hovered = rect.collidepoint(mx, my)

            # Background panel
            bg_col = (45, 40, 35) if hovered else PANEL_COLOR
            pygame.draw.rect(screen, bg_col, rect, border_radius=8)

            # Border
            border_col = ACCENT_COLOR if hovered else (80, 70, 60)
            pygame.draw.rect(screen, border_col, rect, 2, border_radius=8)

            # Label
            label_col = (255, 255, 255) if hovered else TEXT_COLOR
            if action == 'quit':
                label_col = DANGER_COLOR if hovered else (180, 80, 80)
            lsurf = font_s.render(label, True, label_col)
            screen.blit(lsurf, lsurf.get_rect(center=(bx, by)))

        # Version hint
        hint = font_sm.render('v1.0  –  LAN Multiplayer supported', True, (80, 75, 70))
        screen.blit(hint, hint.get_rect(center=(WIDTH // 2, HEIGHT - 30)))

        pygame.display.flip()
