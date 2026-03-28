import math
import random
from dataclasses import dataclass
from config import BULLET_LIFETIME, BULLET_RADIUS, BULLET_SPEED, vec
from bullet import Bullet


@dataclass
class WeaponStats:
    name: str
    damage: int
    fire_interval: float
    magazine_size: int
    reload_time: float
    bullet_speed: float = BULLET_SPEED
    bullet_life: float = BULLET_LIFETIME
    bullet_radius: int = BULLET_RADIUS
    spread_deg: float = 2.0
    recoil: float = 4.5
    pellets: int = 1        # 샷건용


# 무기 프리셋
WEAPONS = {
    'rifle': WeaponStats(
        name='Assault Rifle',
        damage=18,
        fire_interval=0.11,
        magazine_size=30,
        reload_time=1.2,
        spread_deg=1.8,
        recoil=3.5,
    ),
    'shotgun': WeaponStats(
        name='Shotgun',
        damage=14,
        fire_interval=0.65,
        magazine_size=8,
        reload_time=1.8,
        spread_deg=14.0,
        recoil=9.0,
        pellets=7,
        bullet_speed=780,
        bullet_life=0.35,
    ),
    'smg': WeaponStats(
        name='SMG',
        damage=10,
        fire_interval=0.07,
        magazine_size=40,
        reload_time=1.0,
        spread_deg=4.5,
        recoil=2.0,
    ),
}


class Weapon:
    def __init__(self, stats: WeaponStats):
        self.stats = stats
        self.cooldown = 0.0
        self.ammo = stats.magazine_size
        self.reload_timer = 0.0

    @property
    def is_reloading(self):
        return self.reload_timer > 0

    def update(self, dt):
        if self.cooldown > 0:
            self.cooldown -= dt
        if self.reload_timer > 0:
            self.reload_timer -= dt
            if self.reload_timer <= 0:
                self.ammo = self.stats.magazine_size

    def begin_reload(self):
        if self.is_reloading:
            return False
        if self.ammo >= self.stats.magazine_size:
            return False
        self.reload_timer = self.stats.reload_time
        return True

    def can_fire(self):
        return self.cooldown <= 0 and not self.is_reloading and self.ammo > 0

    def try_fire(self, owner_pos, direction):
        if not self.can_fire():
            return []
        self.cooldown = self.stats.fire_interval
        self.ammo -= 1

        dir_vec = vec(direction)
        if dir_vec.length_squared() == 0:
            dir_vec = vec(1, 0)
        dir_vec = dir_vec.normalize()
        base_angle = math.atan2(dir_vec.y, dir_vec.x)
        spread = math.radians(self.stats.spread_deg)

        bullets = []
        for _ in range(self.stats.pellets):
            a = base_angle + random.uniform(-spread, spread)
            d = vec(math.cos(a), math.sin(a))
            vel = d * self.stats.bullet_speed
            bullets.append(
                Bullet(owner_pos + d * 26, vel, self.stats.damage,
                       self.stats.bullet_life, self.stats.bullet_radius, owner='player')
            )
        return bullets
