import math
import pygame
from config import ACCENT_COLOR, vec


# trait 타입별 색상 (weapon.py에서 import하면 순환참조 위험 → 직접 정의)
_TRAIT_COLORS = {
    'fire':      (255, 120, 30),
    'ice':       (120, 200, 255),
    'shock':     (200, 180, 255),
    'bleed':     (180, 30,  80),
    'explosive': (255, 200, 60),
    'pierce':    (200, 255, 180),
}


class Bullet:
    def __init__(self, pos, velocity, damage, life, radius,
                 owner='player', trait=None, pierce=False):
        self.pos = vec(pos)
        self.velocity = vec(velocity)
        self.damage = damage
        self.life = life
        self.radius = radius
        self.owner = owner
        self.dead = False

        # trait / pierce
        self.trait = trait          # WeaponTrait or None
        self.pierce = pierce        # 관통 여부
        self.hit_enemies = set()    # 관통 시 중복 히트 방지 (id 저장)

    def update(self, dt, world_rect):
        self.life -= dt
        self.pos += self.velocity * dt
        if self.life <= 0:
            self.dead = True
        if not world_rect.collidepoint(self.pos.x, self.pos.y):
            self.dead = True

    def _get_color(self):
        """trait 종류에 따라 총알 색상 결정"""
        if self.owner != 'player':
            return (255, 90, 90)
        if self.trait is not None:
            return _TRAIT_COLORS.get(self.trait.type.value, ACCENT_COLOR)
        return ACCENT_COLOR

    def _get_tail_length(self):
        """bullet_speed에 비례한 꼬리 길이 (최소 6, 최대 28)"""
        speed = self.velocity.length() if self.velocity.length_squared() > 0 else 0
        return max(6, min(28, int(speed * 0.015)))

    def draw(self, surface, camera):
        color = self._get_color()

        if self.owner == 'player':
            tail_color = tuple(max(0, min(255, int(c * 0.65))) for c in color)
        else:
            tail_color = (200, 60, 60)

        screen_pos = camera.world_to_screen(self.pos)
        pygame.draw.circle(surface, color, screen_pos, self.radius)

        # 꼬리
        if self.velocity.length_squared() > 0:
            tail_len = self._get_tail_length()
            tail_world = self.pos - self.velocity.normalize() * tail_len
            tail_screen = camera.world_to_screen(tail_world)
            pygame.draw.line(surface, tail_color, tail_screen, screen_pos, 2)

        # EXPLOSIVE 총알 외곽 링 강조
        if self.trait is not None and self.trait.type.value == 'explosive':
            pygame.draw.circle(surface, (255, 240, 100), screen_pos, self.radius + 2, 1)

        # PIERCE 총알 빛나는 테두리
        if self.pierce and self.owner == 'player':
            pygame.draw.circle(surface, (230, 255, 200), screen_pos, self.radius + 1, 1)
