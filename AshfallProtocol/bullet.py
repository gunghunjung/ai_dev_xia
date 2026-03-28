import pygame
from config import ACCENT_COLOR, vec


class Bullet:
    def __init__(self, pos, velocity, damage, life, radius, owner='player'):
        self.pos = vec(pos)
        self.velocity = vec(velocity)
        self.damage = damage
        self.life = life
        self.radius = radius
        self.owner = owner
        self.dead = False

    def update(self, dt, world_rect):
        self.life -= dt
        self.pos += self.velocity * dt
        if self.life <= 0:
            self.dead = True
        if not world_rect.collidepoint(self.pos.x, self.pos.y):
            self.dead = True

    def draw(self, surface, camera):
        screen_pos = camera.world_to_screen(self.pos)
        if self.owner == 'player':
            color = ACCENT_COLOR
            tail_color = (255, 235, 180)
        else:
            color = (255, 90, 90)
            tail_color = (200, 60, 60)
        pygame.draw.circle(surface, color, screen_pos, self.radius)
        if self.velocity.length_squared() > 0:
            tail = self.pos - self.velocity.normalize() * 10
            pygame.draw.line(surface, tail_color, camera.world_to_screen(tail), screen_pos, 2)
