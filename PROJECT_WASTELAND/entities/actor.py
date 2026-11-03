"""Shared base for anything that moves, takes hits and dies."""
import math


class Actor:
    def __init__(self, x: float, y: float, radius: float, max_hp: float):
        self.x = x
        self.y = y
        self.radius = radius
        self.max_hp = max_hp
        self.hp = max_hp
        self.kx = 0.0  # knockback velocity (world units / s)
        self.ky = 0.0
        self.facing = 0.0  # world-space angle
        self.walk_phase = 0.0
        self.moving = False
        self.flash = 0.0
        self.stun = 0.0
        self.dead = False
        self.death_t = 0.0
        self.mass = 1.0
        self.z = 0.0

    @property
    def alive(self) -> bool:
        return not self.dead

    def apply_knockback(self, angle: float, force: float):
        k = force / max(0.2, self.mass)
        self.kx += math.cos(angle) * k
        self.ky += math.sin(angle) * k

    def tick_common(self, dt: float):
        if self.flash > 0:
            self.flash = max(0.0, self.flash - dt * 6)
        if self.stun > 0:
            self.stun = max(0.0, self.stun - dt)
        # Knockback decays quickly (friction).
        decay = math.exp(-dt * 9.0)
        self.kx *= decay
        self.ky *= decay
        if abs(self.kx) < 0.01:
            self.kx = 0.0
        if abs(self.ky) < 0.01:
            self.ky = 0.0
        if self.dead:
            self.death_t += dt

    def dist_to(self, other) -> float:
        return math.hypot(other.x - self.x, other.y - self.y)

    def angle_to(self, other) -> float:
        return math.atan2(other.y - self.y, other.x - self.x)
