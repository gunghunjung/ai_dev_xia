import math
import random
from dataclasses import dataclass, field
from enum import Enum
from config import BULLET_LIFETIME, BULLET_RADIUS, BULLET_SPEED, vec
from bullet import Bullet


# ─── Trait 시스템 ──────────────────────────────────────────────────────────────

class TraitType(Enum):
    FIRE      = 'fire'       # DoT 화염 (초당 5 데미지, 3초)
    ICE       = 'ice'        # 이동속도 50% 감소, 2초
    SHOCK     = 'shock'      # 연쇄 (반경 120 내 적 1명에게 50% 데미지 전달)
    BLEED     = 'bleed'      # DoT 출혈 (초당 3 데미지, 5초)
    PIERCE    = 'pierce'     # 관통 (총알이 적을 통과)
    EXPLOSIVE = 'explosive'  # 충돌 시 반경 80 폭발, 40 데미지


@dataclass
class WeaponTrait:
    type: TraitType
    chance: float = 1.0   # 발동 확률 (0~1)


# ─── WeaponStats ───────────────────────────────────────────────────────────────

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
    pellets: int = 1                                    # 샷건용
    traits: list = field(default_factory=list)          # WeaponTrait 리스트
    heat_based: bool = False                            # True면 탄약 대신 과열 시스템


# ─── 무기 프리셋 ───────────────────────────────────────────────────────────────

WEAPONS = {
    # 기존 3종
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

    # 신규 5종
    'sniper': WeaponStats(
        name='Sniper Rifle',
        damage=95,
        fire_interval=1.6,
        magazine_size=5,
        reload_time=2.4,
        bullet_speed=1800,
        spread_deg=0.3,
        recoil=12.0,
        pellets=1,
        bullet_life=1.5,
        bullet_radius=5,
        traits=[WeaponTrait(TraitType.PIERCE, 1.0)],   # 항상 관통
    ),
    'grenade': WeaponStats(
        name='Grenade Launcher',
        damage=60,
        fire_interval=0.9,
        magazine_size=4,
        reload_time=2.0,
        bullet_speed=380,
        spread_deg=1.0,
        recoil=8.0,
        pellets=1,
        bullet_life=2.0,
        bullet_radius=8,
        traits=[WeaponTrait(TraitType.EXPLOSIVE, 1.0)],
    ),
    'energy': WeaponStats(
        name='Energy Rifle',
        damage=22,
        fire_interval=0.09,
        magazine_size=60,
        reload_time=2.2,
        bullet_speed=1100,
        spread_deg=2.5,
        recoil=1.5,
        pellets=1,
        heat_based=True,
        traits=[WeaponTrait(TraitType.FIRE, 0.25)],    # 25% 확률 화염
    ),
    'chain': WeaponStats(
        name='Chain Lightning',
        damage=28,
        fire_interval=0.45,
        magazine_size=18,
        reload_time=1.6,
        bullet_speed=700,
        spread_deg=3.0,
        recoil=4.0,
        pellets=1,
        traits=[WeaponTrait(TraitType.SHOCK, 1.0)],
    ),
    'cryo_smg': WeaponStats(
        name='Cryo SMG',
        damage=8,
        fire_interval=0.08,
        magazine_size=45,
        reload_time=1.1,
        bullet_speed=820,
        spread_deg=5.0,
        recoil=1.8,
        pellets=1,
        traits=[WeaponTrait(TraitType.ICE, 0.20)],     # 20% 확률 냉기
    ),
}


# ─── Weapon 클래스 ─────────────────────────────────────────────────────────────

class Weapon:
    def __init__(self, stats: WeaponStats):
        self.stats = stats
        self.cooldown = 0.0
        self.ammo = stats.magazine_size
        self.reload_timer = 0.0

        # 과열 시스템 (heat_based 무기)
        self.heat = 0.0             # 0~100
        self.overheated = False
        self.overheat_cooldown = 0.0  # 강제 냉각 타이머

    @property
    def is_reloading(self):
        return self.reload_timer > 0

    def update(self, dt):
        if self.cooldown > 0:
            self.cooldown -= dt

        if self.stats.heat_based:
            self._update_heat(dt)
        else:
            if self.reload_timer > 0:
                self.reload_timer -= dt
                if self.reload_timer <= 0:
                    self.ammo = self.stats.magazine_size

    def _update_heat(self, dt):
        """과열 시스템 업데이트"""
        if self.overheated:
            # 과열 상태: 강제 2초 냉각
            self.overheat_cooldown -= dt
            self.heat -= 50 * dt
            if self.heat < 0:
                self.heat = 0
            if self.overheat_cooldown <= 0:
                self.overheated = False
                self.overheat_cooldown = 0.0
        else:
            # 자연 냉각 (비발사 시)
            if self.heat > 0:
                self.heat = max(0.0, self.heat - 15 * dt)

    def begin_reload(self):
        if self.stats.heat_based:
            return False   # heat_based는 reload 없음
        if self.is_reloading:
            return False
        if self.ammo >= self.stats.magazine_size:
            return False
        self.reload_timer = self.stats.reload_time
        return True

    def can_fire(self):
        if self.stats.heat_based:
            return self.cooldown <= 0 and not self.overheated
        return self.cooldown <= 0 and not self.is_reloading and self.ammo > 0

    def try_fire(self, owner_pos, direction):
        if not self.can_fire():
            return []
        self.cooldown = self.stats.fire_interval

        if self.stats.heat_based:
            self.heat = min(100.0, self.heat + 8.0)
            if self.heat >= 100.0:
                self.heat = 100.0
                self.overheated = True
                self.overheat_cooldown = 2.0
        else:
            self.ammo -= 1

        dir_vec = vec(direction)
        if dir_vec.length_squared() == 0:
            dir_vec = vec(1, 0)
        dir_vec = dir_vec.normalize()
        base_angle = math.atan2(dir_vec.y, dir_vec.x)
        spread = math.radians(self.stats.spread_deg)

        # trait 발동 결정
        active_trait = None
        has_pierce = False
        for t in self.stats.traits:
            if random.random() < t.chance:
                active_trait = t
                if t.type == TraitType.PIERCE:
                    has_pierce = True
                break

        bullets = []
        for _ in range(self.stats.pellets):
            a = base_angle + random.uniform(-spread, spread)
            d = vec(math.cos(a), math.sin(a))
            vel = d * self.stats.bullet_speed
            bullets.append(
                Bullet(
                    owner_pos + d * 26,
                    vel,
                    self.stats.damage,
                    self.stats.bullet_life,
                    self.stats.bullet_radius,
                    owner='player',
                    trait=active_trait,
                    pierce=has_pierce,
                )
            )

        # ammo 자동 재장전 트리거 (non-heat 무기)
        if not self.stats.heat_based and self.ammo <= 0 and not self.is_reloading:
            self.begin_reload()

        return bullets
