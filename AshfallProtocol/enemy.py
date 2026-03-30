import math
import random
import pygame
from config import WORLD_HEIGHT, WORLD_WIDTH, vec
from bullet import Bullet


# ─── Base ───────────────────────────────────────────────────────────────────

class EnemyBase:
    def __init__(self, pos, hp, speed, radius, damage, color, score=10):
        self.pos = vec(pos)
        self.velocity = vec()
        self.hp = hp
        self.max_hp = hp
        self.speed = speed
        self.radius = radius
        self.damage = damage
        self.color = color
        self.score = score
        self.dead = False
        self.hit_flash = 0.0
        self.attack_cooldown = 0.0
        self.knockback = vec()

    def update(self, dt, player, obstacles, bullets_out=None):
        self.attack_cooldown = max(0.0, self.attack_cooldown - dt)
        self.hit_flash = max(0.0, self.hit_flash - dt * 5)
        self._move_toward_player(dt, player, obstacles)

    def _move_toward_player(self, dt, player, obstacles):
        direction = player.pos - self.pos
        if direction.length_squared() > 0:
            direction = direction.normalize()
        desired = direction * self.speed + self.knockback
        self.knockback *= 0.80
        self.velocity = desired
        self.pos.x += self.velocity.x * dt
        self._resolve_collision(obstacles, axis='x')
        self.pos.y += self.velocity.y * dt
        self._resolve_collision(obstacles, axis='y')
        self.pos.x = max(self.radius, min(WORLD_WIDTH  - self.radius, self.pos.x))
        self.pos.y = max(self.radius, min(WORLD_HEIGHT - self.radius, self.pos.y))

    def _resolve_collision(self, obstacles, axis='x'):
        rect = self.rect
        for ob in obstacles:
            if rect.colliderect(ob):
                if axis == 'x':
                    if self.velocity.x > 0: self.pos.x = ob.left  - self.radius
                    else:                   self.pos.x = ob.right + self.radius
                else:
                    if self.velocity.y > 0: self.pos.y = ob.top    - self.radius
                    else:                   self.pos.y = ob.bottom + self.radius
                rect = self.rect

    @property
    def rect(self):
        return pygame.Rect(
            int(self.pos.x - self.radius), int(self.pos.y - self.radius),
            self.radius * 2, self.radius * 2
        )

    def take_damage(self, amount, knockback_dir=None):
        self.hp -= amount
        self.hit_flash = 1.0
        if knockback_dir is not None and knockback_dir.length_squared() > 0:
            self.knockback += knockback_dir.normalize() * 110
        if self.hp <= 0:
            self.dead = True

    def try_attack(self, player):
        if self.attack_cooldown > 0:
            return False
        reach = self.radius + player.radius + 6
        if self.pos.distance_to(player.pos) <= reach:
            self.attack_cooldown = 0.75
            return True
        return False

    def draw(self, surface, camera):
        c = (255, 200, 200) if self.hit_flash > 0.5 else self.color
        pos = camera.world_to_screen(self.pos)
        pygame.draw.circle(surface, c, pos, self.radius)
        # HP 바
        hp_ratio = max(0.0, self.hp / self.max_hp)
        bar_w = self.radius * 2
        bar_rect = pygame.Rect(pos[0] - self.radius, pos[1] - self.radius - 10, bar_w, 5)
        pygame.draw.rect(surface, (30, 30, 30), bar_rect)
        pygame.draw.rect(surface, (200, 64, 64),
                         (bar_rect.x, bar_rect.y, int(bar_w * hp_ratio), 5))


# ─── Runner (빠르고 약함) ────────────────────────────────────────────────────

class RunnerEnemy(EnemyBase):
    def __init__(self, pos):
        super().__init__(pos, hp=28, speed=160, radius=14, damage=10,
                         color=(205, 104, 74), score=10)

    def draw(self, surface, camera):
        super().draw(surface, camera)
        # 뾰족한 표시 (방향 삼각형)
        pos = camera.world_to_screen(self.pos)
        if self.velocity.length_squared() > 0:
            d = self.velocity.normalize()
            tip = (pos[0] + int(d.x * self.radius), pos[1] + int(d.y * self.radius))
            pygame.draw.circle(surface, (255, 180, 100), tip, 4)


# ─── Brute (느리고 탱커) ────────────────────────────────────────────────────

class BruteEnemy(EnemyBase):
    def __init__(self, pos):
        super().__init__(pos, hp=90, speed=75, radius=24, damage=22,
                         color=(108, 145, 91), score=25)


# ─── Shooter (원거리) ───────────────────────────────────────────────────────

class ShooterEnemy(EnemyBase):
    def __init__(self, pos):
        super().__init__(pos, hp=40, speed=60, radius=16, damage=8,
                         color=(150, 100, 200), score=20)
        self.shoot_cooldown = 0.0
        self.preferred_dist = 260

    def update(self, dt, player, obstacles, bullets_out=None):
        self.attack_cooldown = max(0.0, self.attack_cooldown - dt)
        self.hit_flash = max(0.0, self.hit_flash - dt * 5)
        self.shoot_cooldown = max(0.0, self.shoot_cooldown - dt)
        self._move_toward_player(dt, player, obstacles)

        # 일정 거리 유지하면서 사격
        dist = self.pos.distance_to(player.pos)
        if self.shoot_cooldown <= 0 and dist < 400 and bullets_out is not None:
            self.shoot_cooldown = 1.4
            direction = player.pos - self.pos
            if direction.length_squared() > 0:
                direction = direction.normalize()
            vel = direction * 420
            bullets_out.append(
                Bullet(self.pos + direction * (self.radius + 8), vel,
                       self.damage, 1.2, 5, owner='enemy')
            )

    def _move_toward_player(self, dt, player, obstacles):
        dist = self.pos.distance_to(player.pos)
        direction = player.pos - self.pos
        if direction.length_squared() > 0:
            direction = direction.normalize()
        # 너무 가까우면 물러남
        if dist < self.preferred_dist - 30:
            direction = -direction
        elif dist < self.preferred_dist + 30:
            direction = vec(0, 0)
        desired = direction * self.speed + self.knockback
        self.knockback *= 0.80
        self.velocity = desired
        self.pos.x += self.velocity.x * dt
        self._resolve_collision(obstacles, axis='x')
        self.pos.y += self.velocity.y * dt
        self._resolve_collision(obstacles, axis='y')
        self.pos.x = max(self.radius, min(WORLD_WIDTH  - self.radius, self.pos.x))
        self.pos.y = max(self.radius, min(WORLD_HEIGHT - self.radius, self.pos.y))


# ─── Boss ────────────────────────────────────────────────────────────────────

class BossEnemy(EnemyBase):
    def __init__(self, pos):
        super().__init__(pos, hp=600, speed=95, radius=40, damage=28,
                         color=(160, 40, 40), score=200)
        self.shoot_cooldown = 0.0
        self.phase = 1  # 체력 50% 이하에서 phase 2

    def update(self, dt, player, obstacles, bullets_out=None):
        self.attack_cooldown = max(0.0, self.attack_cooldown - dt)
        self.hit_flash = max(0.0, self.hit_flash - dt * 5)
        self.shoot_cooldown = max(0.0, self.shoot_cooldown - dt)

        hp_ratio = self.hp / self.max_hp
        self.phase = 1 if hp_ratio > 0.5 else 2
        speed_mult = 1.4 if self.phase == 2 else 1.0
        self.speed = int(95 * speed_mult)

        self._move_toward_player(dt, player, obstacles)

        # 방사형 사격
        if self.shoot_cooldown <= 0 and bullets_out is not None:
            self.shoot_cooldown = 0.8 if self.phase == 2 else 1.4
            shots = 8 if self.phase == 2 else 5
            for i in range(shots):
                angle = (math.tau / shots) * i
                d = vec(math.cos(angle), math.sin(angle))
                vel = d * 360
                bullets_out.append(
                    Bullet(self.pos + d * (self.radius + 8), vel,
                           self.damage // 2, 1.8, 6, owner='enemy')
                )

    def draw(self, surface, camera):
        pos = camera.world_to_screen(self.pos)
        # 외곽 링
        ring_color = (220, 80, 80) if self.phase == 2 else (160, 40, 40)
        pygame.draw.circle(surface, ring_color, pos, self.radius + 5, 3)
        super().draw(surface, camera)
        # BOSS 텍스트
        # (UI에서 별도 처리)


# ─── 드럼통 ───────────────────────────────────────────────────────────────────

class Barrel:
    """폭발 드럼통 — 적/총알에 맞으면 폭발"""
    def __init__(self, pos):
        self.pos = vec(pos)
        self.radius = 16
        self.hp = 30
        self.dead = False
        self.exploded = False

    @property
    def rect(self):
        return pygame.Rect(
            int(self.pos.x - self.radius), int(self.pos.y - self.radius),
            self.radius * 2, self.radius * 2
        )

    def take_damage(self, amount):
        self.hp -= amount
        if self.hp <= 0 and not self.exploded:
            self.exploded = True
            self.dead = True

    def draw(self, surface, camera):
        pos = camera.world_to_screen(self.pos)
        pygame.draw.circle(surface, (200, 130, 50), pos, self.radius)
        pygame.draw.circle(surface, (160, 90, 30), pos, self.radius, 3)
        # X 표시
        r = self.radius - 5
        pygame.draw.line(surface, (100, 50, 20),
                         (pos[0]-r, pos[1]-r), (pos[0]+r, pos[1]+r), 2)
        pygame.draw.line(surface, (100, 50, 20),
                         (pos[0]+r, pos[1]-r), (pos[0]-r, pos[1]+r), 2)


# ─── 아이템 드랍 ─────────────────────────────────────────────────────────────

class DropItem:
    TYPES = {
        'hp':     {'color': (73, 208, 130), 'label': '+HP',    'size': 12},
        'ammo':   {'color': (255, 200, 60),  'label': '+AMMO',  'size': 10},
    }

    def __init__(self, pos, item_type='hp'):
        self.pos = vec(pos)
        self.item_type = item_type
        self.dead = False
        self.life = 8.0  # 8초 후 사라짐
        info = self.TYPES[item_type]
        self.color = info['color']
        self.label = info['label']
        self.size = info['size']

    @property
    def rect(self):
        s = self.size
        return pygame.Rect(int(self.pos.x - s), int(self.pos.y - s), s*2, s*2)

    def update(self, dt):
        self.life -= dt
        if self.life <= 0:
            self.dead = True

    def apply(self, player):
        if self.item_type == 'hp':
            player.hp = min(player.max_hp, player.hp + 25)
        elif self.item_type == 'ammo':
            for w in player.weapons:
                w.ammo = w.stats.magazine_size
        self.dead = True

    def draw(self, surface, camera):
        pos = camera.world_to_screen(self.pos)
        alpha = min(1.0, self.life * 0.5)
        pygame.draw.circle(surface, self.color, pos, self.size)
        pygame.draw.circle(surface, (255, 255, 255), pos, self.size, 2)


# ─── Spawner ─────────────────────────────────────────────────────────────────

class EnemySpawner:
    def __init__(self):
        self.wave = 1
        self.to_spawn = 0
        self.spawn_timer = 0.0
        self.break_timer = 0.0
        self.wave_cleared = False
        self.boss_spawned = False

    def start_wave(self):
        self.to_spawn = 4 + self.wave * 2
        self.spawn_timer = 0.5
        self.wave_cleared = False
        # 보스 웨이브 (5, 10, 15 ...)
        if self.wave % 5 == 0:
            self.boss_spawned = False

    def update(self, dt, game):
        if self.to_spawn <= 0 and not game.enemies:
            self.break_timer += dt
            if not self.wave_cleared:
                self.wave_cleared = True
            if self.break_timer >= game.wave_break_time:
                self.break_timer = 0.0
                self.wave += 1
                self.start_wave()
        else:
            self.break_timer = 0.0

        if self.to_spawn > 0:
            self.spawn_timer -= dt
            if self.spawn_timer <= 0:
                self.spawn_timer = max(0.15, 0.55 - self.wave * 0.015)
                self.to_spawn -= 1
                game.enemies.append(
                    self._make_enemy(game.player.pos, game.obstacles)
                )
                # 보스 웨이브
                if self.wave % 5 == 0 and not self.boss_spawned and self.to_spawn == 0:
                    self.boss_spawned = True
                    bp = self._random_border_pos()
                    game.enemies.append(BossEnemy(bp))

    def _make_enemy(self, player_pos, obstacles):
        pos = self._random_border_pos()
        r = random.random()
        wave = self.wave
        if wave >= 6 and r < 0.18:
            return ShooterEnemy(pos)
        elif wave >= 3 and r < 0.35:
            return BruteEnemy(pos)
        else:
            return RunnerEnemy(pos)

    def _random_border_pos(self):
        margin = 180
        side = random.randint(0, 3)
        if side == 0: return vec(random.randint(0, WORLD_WIDTH), -margin)
        if side == 1: return vec(random.randint(0, WORLD_WIDTH), WORLD_HEIGHT + margin)
        if side == 2: return vec(-margin, random.randint(0, WORLD_HEIGHT))
        return vec(WORLD_WIDTH + margin, random.randint(0, WORLD_HEIGHT))
