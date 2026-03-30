import math
import random
import pygame
from config import SCREEN_SHAKE_DECAY, vec


class Particle:
    def __init__(self, pos, velocity, color, radius, life):
        self.pos = vec(pos)
        self.velocity = vec(velocity)
        self.color = color
        self.radius = radius
        self.life = life
        self.max_life = life

    def update(self, dt):
        self.life -= dt
        self.pos += self.velocity * dt
        self.velocity *= 0.88
        self.radius = max(1, self.radius - dt * 7)

    @property
    def dead(self):
        return self.life <= 0

    def draw(self, surface, camera):
        alpha_scale = max(0.0, self.life / self.max_life)
        color = tuple(max(0, min(255, int(c * alpha_scale))) for c in self.color)
        pygame.draw.circle(surface, color, camera.world_to_screen(self.pos), max(1, int(self.radius)))


class FlashText:
    def __init__(self, text, pos, color, life=0.45):
        self.text = text
        self.pos = vec(pos)
        self.color = color
        self.life = life
        self.max_life = life

    def update(self, dt):
        self.life -= dt
        self.pos.y -= 30 * dt

    @property
    def dead(self):
        return self.life <= 0

    def draw(self, surface, camera, font):
        alpha = max(0.0, self.life / self.max_life)
        img = font.render(self.text, True, self.color)
        img.set_alpha(int(255 * alpha))
        rect = img.get_rect(center=camera.world_to_screen(self.pos))
        surface.blit(img, rect)


class EffectManager:
    def __init__(self):
        self.particles = []
        self.texts = []
        self.shake_power = 0.0
        self.shake_time = 0.0

    def spawn_hit(self, pos, base_color=(255, 110, 90), count=8):
        for _ in range(count):
            angle = random.uniform(0, math.tau)
            speed = random.uniform(90, 280)
            vel = vec(math.cos(angle), math.sin(angle)) * speed
            self.particles.append(
                Particle(pos, vel, base_color, random.uniform(2, 5), random.uniform(0.18, 0.40))
            )

    def spawn_death(self, pos, base_color=(200, 70, 50), count=18):
        for _ in range(count):
            angle = random.uniform(0, math.tau)
            speed = random.uniform(150, 380)
            vel = vec(math.cos(angle), math.sin(angle)) * speed
            self.particles.append(
                Particle(pos, vel, base_color, random.uniform(3, 6), random.uniform(0.35, 0.70))
            )

    def spawn_explosion(self, pos, count=32):
        colors = [(255, 200, 50), (255, 130, 30), (220, 60, 30), (180, 180, 180)]
        for _ in range(count):
            angle = random.uniform(0, math.tau)
            speed = random.uniform(80, 500)
            vel = vec(math.cos(angle), math.sin(angle)) * speed
            color = random.choice(colors)
            self.particles.append(
                Particle(pos, vel, color, random.uniform(3, 8), random.uniform(0.3, 0.8))
            )

    def add_damage_text(self, value, pos, color=(255, 220, 160)):
        self.texts.append(FlashText(str(value), pos, color))

    def add_shake(self, power=6.0, duration=0.13):
        self.shake_power = max(self.shake_power, power)
        self.shake_time = max(self.shake_time, duration)

    def update(self, dt):
        for p in self.particles:
            p.update(dt)
        for t in self.texts:
            t.update(dt)
        self.particles = [p for p in self.particles if not p.dead]
        self.texts = [t for t in self.texts if not t.dead]
        if self.shake_time > 0:
            self.shake_time -= dt
            self.shake_power = max(0.0, self.shake_power - SCREEN_SHAKE_DECAY * dt)
        else:
            self.shake_power = 0.0

    def current_shake_offset(self):
        if self.shake_power <= 0:
            return vec(0, 0)
        return vec(
            random.uniform(-self.shake_power, self.shake_power),
            random.uniform(-self.shake_power, self.shake_power)
        )

    def draw(self, surface, camera, small_font):
        for p in self.particles:
            p.draw(surface, camera)
        for t in self.texts:
            t.draw(surface, camera, small_font)
