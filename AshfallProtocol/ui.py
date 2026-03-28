import pygame
from config import ACCENT_COLOR, DANGER_COLOR, HEAL_COLOR, PANEL_COLOR, TEXT_COLOR, WIDTH, HEIGHT


class UI:
    def __init__(self, assets):
        self.assets = assets
        self.font_big   = assets.font(42, bold=True)
        self.font_med   = assets.font(24, bold=True)
        self.font_small = assets.font(18)
        self.font_tiny  = assets.font(14)
        self.wave_flash = 0.0  # 웨이브 클리어 메시지 타이머

    def notify_wave_clear(self):
        self.wave_flash = 1.8

    def draw_bar(self, surface, x, y, w, h, ratio, fill_color, label, bg=None):
        bg = bg or PANEL_COLOR
        pygame.draw.rect(surface, bg, (x, y, w, h), border_radius=6)
        pygame.draw.rect(surface, (55, 55, 55), (x, y, w, h), 2, border_radius=6)
        inner = max(0, min(w - 4, int((w - 4) * ratio)))
        pygame.draw.rect(surface, fill_color, (x + 2, y + 2, inner, h - 4), border_radius=5)
        text = self.font_tiny.render(label, True, TEXT_COLOR)
        surface.blit(text, (x + 8, y + h // 2 - text.get_height() // 2))

    def update(self, dt):
        if self.wave_flash > 0:
            self.wave_flash -= dt

    def draw(self, surface, game):
        p = game.player
        self._draw_hud(surface, game)
        self._draw_weapon_selector(surface, p)
        if self.wave_flash > 0 and not p.dead:
            self._draw_wave_clear(surface, game.spawner.wave - 1)
        if game.spawner.wave % 5 == 0 and any(
                hasattr(e, 'shoot_cooldown') and e.max_hp >= 500 for e in game.enemies):
            self._draw_boss_bar(surface, game)
        if game.paused and not p.dead:
            self._overlay_center(surface, 'PAUSED', 'ESC to resume  |  Q to quit')
        if p.dead:
            self._overlay_center(surface, 'YOU DIED',
                                 f'KILLS: {game.kills}  SCORE: {game.score}  WAVE: {game.spawner.wave}  |  ENTER restart  ESC quit',
                                 danger=True)

    def _draw_hud(self, surface, game):
        p = game.player
        # HP
        self.draw_bar(surface, 18, 18, 240, 26,
                      p.hp / p.max_hp, HEAL_COLOR, f'HP  {p.hp}/{p.max_hp}')
        # AMMO
        ammo_ratio = p.weapon.ammo / p.weapon.stats.magazine_size
        ammo_color = ACCENT_COLOR if not p.weapon.is_reloading else (160, 160, 160)
        ammo_label = 'RELOADING...' if p.weapon.is_reloading else f'{p.weapon.ammo}/{p.weapon.stats.magazine_size}'
        self.draw_bar(surface, 18, 52, 240, 22, ammo_ratio, ammo_color, ammo_label)

        # Wave / Score / Kills — 우상단
        wave_s  = self.font_med.render(f'WAVE  {game.spawner.wave}', True, TEXT_COLOR)
        score_s = self.font_med.render(f'SCORE {game.score}', True, ACCENT_COLOR)
        kills_s = self.font_small.render(f'KILLS {game.kills}', True, TEXT_COLOR)
        surface.blit(wave_s,  (WIDTH - wave_s.get_width()  - 18, 14))
        surface.blit(score_s, (WIDTH - score_s.get_width() - 18, 42))
        surface.blit(kills_s, (WIDTH - kills_s.get_width() - 18, 70))

        # 대시 쿨 (게이지)
        dash_ratio = 1.0 - min(1.0, p.dash_cooldown / 0.9)
        self.draw_bar(surface, 18, 82, 100, 10, dash_ratio,
                      (100, 200, 255), 'DASH', bg=(18, 18, 18))

    def _draw_weapon_selector(self, surface, player):
        wx, wy = 18, HEIGHT - 70
        for i, w in enumerate(player.weapons):
            active = (i == player.weapon_idx)
            box = pygame.Rect(wx + i * 100, wy, 92, 50)
            bg = (45, 45, 45) if active else (22, 22, 22)
            pygame.draw.rect(surface, bg, box, border_radius=6)
            pygame.draw.rect(surface, ACCENT_COLOR if active else (55, 55, 55), box, 2, border_radius=6)
            name_s = self.font_tiny.render(w.stats.name, True, TEXT_COLOR)
            ammo_s = self.font_small.render(f'{w.ammo}/{w.stats.magazine_size}', True,
                                            ACCENT_COLOR if not w.is_reloading else (140, 140, 140))
            key_s = self.font_tiny.render(str(i + 1), True, (130, 130, 130))
            surface.blit(key_s,   (box.x + 5,  box.y + 4))
            surface.blit(name_s,  (box.x + 6,  box.y + 16))
            surface.blit(ammo_s,  (box.x + 6,  box.y + 30))

    def _draw_wave_clear(self, surface, wave):
        alpha = min(1.0, self.wave_flash)
        msg = f'WAVE {wave} CLEAR!'
        img = self.font_big.render(msg, True, ACCENT_COLOR)
        img.set_alpha(int(255 * alpha))
        surface.blit(img, img.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 60)))

    def _draw_boss_bar(self, surface, game):
        bosses = [e for e in game.enemies if hasattr(e, 'shoot_cooldown') and e.max_hp >= 500]
        if not bosses:
            return
        boss = bosses[0]
        bw = 500
        bx = (WIDTH - bw) // 2
        by = HEIGHT - 50
        self.draw_bar(surface, bx, by, bw, 28,
                      boss.hp / boss.max_hp, DANGER_COLOR, f'BOSS  {boss.hp}/{boss.max_hp}')
        phase_s = self.font_tiny.render(f'PHASE {boss.phase}', True, (255, 200, 200))
        surface.blit(phase_s, (bx + bw + 8, by + 8))

    def _overlay_center(self, surface, title, subtitle, danger=False):
        overlay = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 170))
        surface.blit(overlay, (0, 0))
        t = self.font_big.render(title, True, DANGER_COLOR if danger else TEXT_COLOR)
        s = self.font_small.render(subtitle, True, (180, 180, 180))
        cx, cy = surface.get_width() // 2, surface.get_height() // 2
        surface.blit(t, t.get_rect(center=(cx, cy - 28)))
        surface.blit(s, s.get_rect(center=(cx, cy + 22)))
