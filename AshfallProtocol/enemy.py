import math
import random
import pygame
from config import WORLD_HEIGHT, WORLD_WIDTH, vec
from bullet import Bullet


# ─── 상태이상 타입 (weapon.py TraitType과 동일 값 — 순환참조 없이 문자열 비교) ─

_TRAIT_FIRE      = 'fire'
_TRAIT_ICE       = 'ice'
_TRAIT_BLEED     = 'bleed'
_TRAIT_SHOCK     = 'shock'
_TRAIT_EXPLOSIVE = 'explosive'
_TRAIT_PIERCE    = 'pierce'


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

        # 상태이상 딕셔너리
        # 예: {'fire': {'damage': 5, 'timer': 3.0}, 'ice': {'timer': 2.0, 'speed_mult': 0.4}}
        self.status = {}

    # ── 상태이상 API ──────────────────────────────────────────────────────

    def apply_status(self, trait):
        """WeaponTrait 객체를 받아 상태이상 적용"""
        t = trait.type.value
        if t == _TRAIT_FIRE:
            self.status['fire'] = {'damage': 5, 'timer': 3.0}
        elif t == _TRAIT_ICE:
            self.status['ice'] = {'timer': 2.0, 'speed_mult': 0.4}
        elif t == _TRAIT_BLEED:
            self.status['bleed'] = {'damage': 3, 'timer': 5.0}
        # SHOCK / EXPLOSIVE / PIERCE는 즉발 효과로 game.py에서 처리

    def _update_status(self, dt):
        for key in list(self.status.keys()):
            s = self.status[key]
            s['timer'] -= dt
            if key in ('fire', 'bleed'):
                dmg = s['damage'] * dt
                self.hp -= dmg
                if self.hp <= 0:
                    self.dead = True
            if s['timer'] <= 0:
                del self.status[key]

    def get_speed_mult(self):
        if 'ice' in self.status:
            return self.status['ice']['speed_mult']
        return 1.0

    # ── 업데이트 ──────────────────────────────────────────────────────────

    def update(self, dt, player, obstacles, bullets_out=None, allies=None):
        self.attack_cooldown = max(0.0, self.attack_cooldown - dt)
        self.hit_flash = max(0.0, self.hit_flash - dt * 5)
        self._update_status(dt)
        if not self.dead:
            self._move_toward_player(dt, player, obstacles)

    def _move_toward_player(self, dt, player, obstacles):
        direction = player.pos - self.pos
        if direction.length_squared() > 0:
            direction = direction.normalize()
        effective_speed = self.speed * self.get_speed_mult()
        desired = direction * effective_speed + self.knockback
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

    # ── 렌더링 ────────────────────────────────────────────────────────────

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

        # 상태이상 표시
        self._draw_status(surface, camera, pos)

    def _draw_status(self, surface, camera, pos):
        """상태이상 시각 효과"""
        t = pygame.time.get_ticks() / 1000.0

        # 화염: 원 주변 주황 점 4개 회전
        if 'fire' in self.status:
            for i in range(4):
                angle = t * 3.0 + (math.tau / 4) * i
                dx = int(math.cos(angle) * (self.radius + 5))
                dy = int(math.sin(angle) * (self.radius + 5))
                pygame.draw.circle(surface, (255, 140, 30),
                                   (pos[0] + dx, pos[1] + dy), 3)

        # 냉기: 외곽 하늘색 링
        if 'ice' in self.status:
            pygame.draw.circle(surface, (120, 200, 255), pos, self.radius + 4, 2)

        # 출혈: 원 주변 빨간 점 표시
        if 'bleed' in self.status:
            for i in range(3):
                angle = t * -2.0 + (math.tau / 3) * i
                dx = int(math.cos(angle) * (self.radius + 4))
                dy = int(math.sin(angle) * (self.radius + 4))
                pygame.draw.circle(surface, (200, 20, 50),
                                   (pos[0] + dx, pos[1] + dy), 3)


# ─── Runner (빠르고 약함) ────────────────────────────────────────────────────

class RunnerEnemy(EnemyBase):
    def __init__(self, pos):
        super().__init__(pos, hp=28, speed=160, radius=14, damage=10,
                         color=(205, 104, 74), score=10)

    def draw(self, surface, camera):
        super().draw(surface, camera)
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

    def update(self, dt, player, obstacles, bullets_out=None, allies=None):
        self.attack_cooldown = max(0.0, self.attack_cooldown - dt)
        self.hit_flash = max(0.0, self.hit_flash - dt * 5)
        self.shoot_cooldown = max(0.0, self.shoot_cooldown - dt)
        self._update_status(dt)
        if not self.dead:
            self._move_toward_player(dt, player, obstacles)

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
        if dist < self.preferred_dist - 30:
            direction = -direction
        elif dist < self.preferred_dist + 30:
            direction = vec(0, 0)
        effective_speed = self.speed * self.get_speed_mult()
        desired = direction * effective_speed + self.knockback
        self.knockback *= 0.80
        self.velocity = desired
        self.pos.x += self.velocity.x * dt
        self._resolve_collision(obstacles, axis='x')
        self.pos.y += self.velocity.y * dt
        self._resolve_collision(obstacles, axis='y')
        self.pos.x = max(self.radius, min(WORLD_WIDTH  - self.radius, self.pos.x))
        self.pos.y = max(self.radius, min(WORLD_HEIGHT - self.radius, self.pos.y))


# ─── Boss ─────────────────────────────────────────────────────────────────────

class BossEnemy(EnemyBase):
    def __init__(self, pos):
        super().__init__(pos, hp=600, speed=95, radius=40, damage=28,
                         color=(160, 40, 40), score=200)
        self.shoot_cooldown = 0.0
        self.phase = 1

    def update(self, dt, player, obstacles, bullets_out=None, allies=None):
        self.attack_cooldown = max(0.0, self.attack_cooldown - dt)
        self.hit_flash = max(0.0, self.hit_flash - dt * 5)
        self.shoot_cooldown = max(0.0, self.shoot_cooldown - dt)
        self._update_status(dt)

        hp_ratio = self.hp / self.max_hp
        self.phase = 1 if hp_ratio > 0.5 else 2
        speed_mult = 1.4 if self.phase == 2 else 1.0
        self.speed = int(95 * speed_mult)

        if not self.dead:
            self._move_toward_player(dt, player, obstacles)

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
        ring_color = (220, 80, 80) if self.phase == 2 else (160, 40, 40)
        pygame.draw.circle(surface, ring_color, pos, self.radius + 5, 3)
        super().draw(surface, camera)


# ─── SuicideEnemy (자폭형) ────────────────────────────────────────────────────

class SuicideEnemy(EnemyBase):
    EXPLODE_TRIGGER_DIST = 16   # 이 거리 이하에서 폭발
    BLINK_WARN_DIST      = 70   # 깜빡임 경고 거리

    def __init__(self, pos):
        super().__init__(pos, hp=20, speed=220, radius=13, damage=0,
                         color=(220, 80, 50), score=8)
        self.will_explode = False
        self.exploded = False
        self.blink_timer = 0.0

    def update(self, dt, player, obstacles, bullets_out=None, allies=None):
        self.attack_cooldown = max(0.0, self.attack_cooldown - dt)
        self.hit_flash = max(0.0, self.hit_flash - dt * 5)
        self._update_status(dt)
        if self.dead:
            return

        self._move_toward_player(dt, player, obstacles)

        dist = self.pos.distance_to(player.pos)
        # 폭발 경고 깜빡임
        if dist <= self.BLINK_WARN_DIST:
            self.blink_timer += dt * 8
        else:
            self.blink_timer = max(0.0, self.blink_timer - dt * 4)

        # 폭발 트리거
        if dist <= self.EXPLODE_TRIGGER_DIST + self.radius + player.radius:
            self.exploded = True
            self.dead = True

    def draw(self, surface, camera):
        # 깜빡임
        blink_on = int(self.blink_timer) % 2 == 0
        c = (255, 240, 60) if (self.blink_timer > 0 and blink_on) else (
            (255, 200, 200) if self.hit_flash > 0.5 else self.color
        )
        pos = camera.world_to_screen(self.pos)
        pygame.draw.circle(surface, c, pos, self.radius)

        hp_ratio = max(0.0, self.hp / self.max_hp)
        bar_w = self.radius * 2
        bar_rect = pygame.Rect(pos[0] - self.radius, pos[1] - self.radius - 10, bar_w, 5)
        pygame.draw.rect(surface, (30, 30, 30), bar_rect)
        pygame.draw.rect(surface, (200, 64, 64),
                         (bar_rect.x, bar_rect.y, int(bar_w * hp_ratio), 5))
        self._draw_status(surface, camera, pos)

        # 경고 원 표시
        if self.blink_timer > 0 and blink_on:
            pygame.draw.circle(surface, (255, 200, 60), pos, 22, 1)


# ─── TeleporterEnemy (순간이동) ───────────────────────────────────────────────

class TeleporterEnemy(EnemyBase):
    def __init__(self, pos):
        super().__init__(pos, hp=35, speed=80, radius=15, damage=12,
                         color=(160, 60, 220), score=22)
        self.teleport_cooldown = 3.0
        self.ghost_timer = 0.0

    def update(self, dt, player, obstacles, bullets_out=None, allies=None):
        self.attack_cooldown = max(0.0, self.attack_cooldown - dt)
        self.hit_flash = max(0.0, self.hit_flash - dt * 5)
        self.ghost_timer = max(0.0, self.ghost_timer - dt)
        self._update_status(dt)
        if self.dead:
            return

        self._move_toward_player(dt, player, obstacles)

        self.teleport_cooldown -= dt
        if self.teleport_cooldown <= 0:
            self._teleport(player)
            self.teleport_cooldown = 3.0

    def _teleport(self, player):
        angle = random.uniform(0, math.tau)
        dist = random.uniform(150, 250)
        self.pos = vec(
            player.pos.x + math.cos(angle) * dist,
            player.pos.y + math.sin(angle) * dist,
        )
        self.pos.x = max(self.radius, min(WORLD_WIDTH  - self.radius, self.pos.x))
        self.pos.y = max(self.radius, min(WORLD_HEIGHT - self.radius, self.pos.y))
        self.ghost_timer = 1.0

    def draw(self, surface, camera):
        # 순간이동 직후 밝게 (반투명 효과)
        if self.ghost_timer > 0:
            bright = min(1.0, self.ghost_timer)
            c = tuple(min(255, int(v + (255 - v) * bright * 0.6)) for v in self.color)
        elif self.hit_flash > 0.5:
            c = (255, 200, 200)
        else:
            c = self.color

        pos = camera.world_to_screen(self.pos)
        pygame.draw.circle(surface, c, pos, self.radius)
        if self.ghost_timer > 0:
            pygame.draw.circle(surface, (200, 140, 255), pos, self.radius + 3, 1)

        hp_ratio = max(0.0, self.hp / self.max_hp)
        bar_w = self.radius * 2
        bar_rect = pygame.Rect(pos[0] - self.radius, pos[1] - self.radius - 10, bar_w, 5)
        pygame.draw.rect(surface, (30, 30, 30), bar_rect)
        pygame.draw.rect(surface, (200, 64, 64),
                         (bar_rect.x, bar_rect.y, int(bar_w * hp_ratio), 5))
        self._draw_status(surface, camera, pos)


# ─── ShieldEnemy (방패) ───────────────────────────────────────────────────────

class ShieldEnemy(EnemyBase):
    SHIELD_ARC_HALF = 120  # 전면 방어각 (도)

    def __init__(self, pos):
        super().__init__(pos, hp=55, speed=65, radius=20, damage=18,
                         color=(80, 120, 200), score=30)
        self.shield_hp = 60
        self.shield_active = True

    def take_damage(self, amount, knockback_dir=None):
        """방패 활성화 시 전면 공격은 방패 HP 감소"""
        if self.shield_active and knockback_dir is not None and \
           knockback_dir.length_squared() > 0:
            # 적의 진행 방향 벡터 (플레이어→적 방향이 공격 방향)
            # knockback_dir은 총알 velocity → 정면(-120~+120도) 판별
            if self.velocity.length_squared() > 0:
                front = self.velocity.normalize()
            else:
                front = vec(1, 0)
            # 공격이 앞쪽에서 왔는지 확인
            attack_dir = knockback_dir.normalize()
            dot = front.dot(attack_dir)
            angle_deg = math.degrees(math.acos(max(-1, min(1, dot))))
            if angle_deg <= self.SHIELD_ARC_HALF:
                # 방패가 막음
                self.shield_hp -= amount
                self.hit_flash = 1.0
                if self.shield_hp <= 0:
                    self.shield_hp = 0
                    self.shield_active = False
                if knockback_dir is not None and knockback_dir.length_squared() > 0:
                    self.knockback += knockback_dir.normalize() * 40  # 방패 반동 감소
                return
        # 방패 무효 or 측면/후면 → 본체 데미지
        super().take_damage(amount, knockback_dir)

    def draw(self, surface, camera):
        super().draw(surface, camera)
        if not self.shield_active:
            return
        pos = camera.world_to_screen(self.pos)
        # 방패 반원 아크 (이동 방향 앞쪽)
        if self.velocity.length_squared() > 0:
            front_angle = math.degrees(math.atan2(self.velocity.y, self.velocity.x))
        else:
            front_angle = 0
        start_angle = math.radians(front_angle - self.SHIELD_ARC_HALF)
        end_angle   = math.radians(front_angle + self.SHIELD_ARC_HALF)
        shield_r = self.radius + 6
        shield_ratio = max(0.0, self.shield_hp / 60)
        shield_color = (
            int(80 + 120 * shield_ratio),
            int(120 + 80 * shield_ratio),
            220,
        )
        rect = pygame.Rect(pos[0] - shield_r, pos[1] - shield_r,
                           shield_r * 2, shield_r * 2)
        pygame.draw.arc(surface, shield_color, rect, -end_angle, -start_angle, 3)


# ─── SniperEnemy (저격) ───────────────────────────────────────────────────────

class SniperEnemy(EnemyBase):
    SNIPE_RANGE       = 500
    AIM_DURATION      = 1.5
    FIRE_COOLDOWN     = 2.5
    BULLET_SPEED      = 600
    BULLET_DAMAGE     = 22

    def __init__(self, pos):
        super().__init__(pos, hp=30, speed=40, radius=14, damage=22,
                         color=(200, 170, 80), score=28)
        self.aim_timer    = 0.0
        self.aim_target   = None
        self.fire_cooldown = 0.0
        self.is_aiming    = False

    def update(self, dt, player, obstacles, bullets_out=None, allies=None):
        self.attack_cooldown = max(0.0, self.attack_cooldown - dt)
        self.hit_flash = max(0.0, self.hit_flash - dt * 5)
        self._update_status(dt)
        if self.dead:
            return

        dist = self.pos.distance_to(player.pos)
        self.fire_cooldown = max(0.0, self.fire_cooldown - dt)

        if dist >= self.SNIPE_RANGE and self.fire_cooldown <= 0:
            # 조준 시작
            if not self.is_aiming:
                self.is_aiming = True
                self.aim_timer = self.AIM_DURATION
                self.aim_target = vec(player.pos)

            self.aim_timer -= dt
            if self.aim_timer <= 0:
                # 발사
                self.is_aiming = False
                self.fire_cooldown = self.FIRE_COOLDOWN
                if bullets_out is not None and self.aim_target is not None:
                    direction = self.aim_target - self.pos
                    if direction.length_squared() > 0:
                        direction = direction.normalize()
                    vel = direction * self.BULLET_SPEED
                    bullets_out.append(
                        Bullet(self.pos + direction * (self.radius + 8), vel,
                               self.BULLET_DAMAGE, 2.0, 5, owner='enemy')
                    )
                self.aim_target = None
        else:
            self.is_aiming = False
            self.aim_timer = 0.0
            # 근거리면 평범하게 접근
            if dist < self.SNIPE_RANGE:
                self._move_toward_player(dt, player, obstacles)

    def draw(self, surface, camera):
        super().draw(surface, camera)
        # 레이저 조준선
        if self.is_aiming and self.aim_target is not None:
            start = camera.world_to_screen(self.pos)
            end   = camera.world_to_screen(self.aim_target)
            # 투명도 효과 → 점선으로 대체 (aim 진행률에 따라 밝기)
            aim_progress = 1.0 - (self.aim_timer / self.AIM_DURATION)
            r = int(200 + 55 * aim_progress)
            pygame.draw.line(surface, (r, 40, 40), start, end, 1)
            # 조준점
            pygame.draw.circle(surface, (255, 60, 60), end, 4, 1)


# ─── SupportEnemy (힐러) ─────────────────────────────────────────────────────

class SupportEnemy(EnemyBase):
    HEAL_RANGE  = 250
    HEAL_RATE   = 15   # 초당 HP

    def __init__(self, pos):
        super().__init__(pos, hp=45, speed=55, radius=16, damage=8,
                         color=(80, 200, 120), score=35)
        self.heal_target = None
        self.heal_line_timer = 0.0

    def update(self, dt, player, obstacles, bullets_out=None, allies=None):
        self.attack_cooldown = max(0.0, self.attack_cooldown - dt)
        self.hit_flash = max(0.0, self.hit_flash - dt * 5)
        self.heal_line_timer = max(0.0, self.heal_line_timer - dt)
        self._update_status(dt)
        if self.dead:
            return

        self.heal_target = None
        # 범위 내 가장 HP 낮은 아군 찾기
        if allies:
            best = None
            best_hp_ratio = 1.1
            for ally in allies:
                if ally is self or ally.dead:
                    continue
                if self.pos.distance_to(ally.pos) <= self.HEAL_RANGE:
                    ratio = ally.hp / ally.max_hp
                    if ratio < best_hp_ratio:
                        best_hp_ratio = ratio
                        best = ally
            if best is not None:
                self.heal_target = best
                best.hp = min(best.max_hp, best.hp + self.HEAL_RATE * dt)
                self.heal_line_timer = 0.2

        # 이동: 힐 대상 없으면 플레이어 접근, 있으면 도주
        if self.heal_target is not None:
            # 플레이어에게서 멀어짐
            direction = self.pos - player.pos
            if direction.length_squared() > 0:
                direction = direction.normalize()
            effective_speed = self.speed * self.get_speed_mult()
            desired = direction * effective_speed + self.knockback
            self.knockback *= 0.80
            self.velocity = desired
            self.pos.x += self.velocity.x * dt
            self._resolve_collision(obstacles, axis='x')
            self.pos.y += self.velocity.y * dt
            self._resolve_collision(obstacles, axis='y')
            self.pos.x = max(self.radius, min(WORLD_WIDTH  - self.radius, self.pos.x))
            self.pos.y = max(self.radius, min(WORLD_HEIGHT - self.radius, self.pos.y))
        else:
            self._move_toward_player(dt, player, obstacles)

    def draw(self, surface, camera):
        super().draw(surface, camera)
        # 힐 선 표시
        if self.heal_target is not None and self.heal_line_timer > 0:
            start = camera.world_to_screen(self.pos)
            end   = camera.world_to_screen(self.heal_target.pos)
            pygame.draw.line(surface, (80, 230, 130), start, end, 2)
            pygame.draw.circle(surface, (80, 255, 140), end, 5, 1)


# ─── EliteRunner ─────────────────────────────────────────────────────────────

class EliteRunner(RunnerEnemy):
    """RunnerEnemy 상속 — 스탯 강화 + BLEED 근접 공격"""

    def __init__(self, pos):
        super().__init__(pos)
        self.hp     = self.hp * 2
        self.max_hp = self.hp
        self.speed  = int(self.speed * 1.3)
        self.score  = self.score * 3
        self.color  = (255, 140, 40)
        self.damage = int(self.damage * 1.2)

    def try_attack(self, player):
        """근접 공격 시 BLEED 상태이상 적용"""
        attacked = super().try_attack(player)
        if attacked:
            # player에 bleed 적용을 위해 특별 플래그 사용
            # (player는 EnemyBase가 아니므로 apply_status 없음 → game.py에서 처리)
            self._pending_bleed = True
        return attacked

    def update(self, dt, player, obstacles, bullets_out=None, allies=None):
        self._pending_bleed = False
        super().update(dt, player, obstacles, bullets_out, allies)


# ─── EliteBrute ──────────────────────────────────────────────────────────────

class EliteBrute(BruteEnemy):
    """BruteEnemy 상속 — 스탯 강화 + 분노 메카닉"""

    def __init__(self, pos):
        super().__init__(pos)
        self.hp     = int(self.hp * 1.5)
        self.max_hp = self.hp
        self._base_speed  = int(self.speed * 1.2)
        self._base_damage = int(self.damage * 1.0)
        self.speed  = self._base_speed
        self.damage = self._base_damage
        self.score  = self.score * 3
        self.color  = (60, 200, 80)
        self.enraged = False

    def update(self, dt, player, obstacles, bullets_out=None, allies=None):
        # 분노: HP 30% 이하
        if self.hp / self.max_hp <= 0.3:
            if not self.enraged:
                self.enraged = True
            self.speed  = int(self._base_speed * 2)
            self.damage = int(self._base_damage * 1.5)
        else:
            self.enraged = False
            self.speed  = self._base_speed
            self.damage = self._base_damage

        super().update(dt, player, obstacles, bullets_out, allies)

    def draw(self, surface, camera):
        super().draw(surface, camera)
        if self.enraged:
            pos = camera.world_to_screen(self.pos)
            t = pygame.time.get_ticks() / 1000.0
            pulse = int(abs(math.sin(t * 6)) * 40)
            pygame.draw.circle(surface, (200 + pulse, 100, 40), pos,
                               self.radius + 4, 2)


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
        r = self.radius - 5
        pygame.draw.line(surface, (100, 50, 20),
                         (pos[0]-r, pos[1]-r), (pos[0]+r, pos[1]+r), 2)
        pygame.draw.line(surface, (100, 50, 20),
                         (pos[0]+r, pos[1]-r), (pos[0]-r, pos[1]+r), 2)


# ─── 아이템 드랍 ─────────────────────────────────────────────────────────────

class DropItem:
    TYPES = {
        'hp':   {'color': (73, 208, 130), 'label': '+HP',   'size': 12},
        'ammo': {'color': (255, 200, 60),  'label': '+AMMO', 'size': 10},
    }

    def __init__(self, pos, item_type='hp'):
        self.pos = vec(pos)
        self.item_type = item_type
        self.dead = False
        self.life = 8.0
        info = self.TYPES[item_type]
        self.color = info['color']
        self.label = info['label']
        self.size  = info['size']

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
                if not w.stats.heat_based:
                    w.ammo = w.stats.magazine_size
        self.dead = True

    def draw(self, surface, camera):
        pos = camera.world_to_screen(self.pos)
        pygame.draw.circle(surface, self.color, pos, self.size)
        pygame.draw.circle(surface, (255, 255, 255), pos, self.size, 2)


# ─── 웨이브 설정 ──────────────────────────────────────────────────────────────

class WaveConfig:
    """웨이브 구성 정의"""
    def __init__(self, squads, event=None):
        self.squads = squads   # list of (EnemyClass, count)
        self.event  = event    # 'boss'|'elite_rush'|'swarm'|'mixed'|None


WAVE_TEMPLATES = [
    WaveConfig([(RunnerEnemy, 6)]),                                                               # 1
    WaveConfig([(RunnerEnemy, 5), (BruteEnemy, 2)]),                                              # 2
    WaveConfig([(RunnerEnemy, 4), (ShooterEnemy, 2), (BruteEnemy, 1)]),                           # 3
    WaveConfig([(SuicideEnemy, 4), (RunnerEnemy, 3), (BruteEnemy, 2)]),                           # 4
    WaveConfig([(RunnerEnemy, 4), (ShooterEnemy, 2)], event='boss'),                              # 5
    WaveConfig([(TeleporterEnemy, 2), (RunnerEnemy, 4), (ShooterEnemy, 2)]),                      # 6
    WaveConfig([(ShieldEnemy, 2), (BruteEnemy, 2), (SuicideEnemy, 3)]),                          # 7
    WaveConfig([(SniperEnemy, 2), (SuicideEnemy, 4), (RunnerEnemy, 4)]),                          # 8
    WaveConfig([(SupportEnemy, 1), (BruteEnemy, 3), (ShooterEnemy, 3), (RunnerEnemy, 2)]),        # 9
    WaveConfig([(EliteRunner, 3), (EliteBrute, 1)], event='boss'),                               # 10
]


# ─── Spawner ─────────────────────────────────────────────────────────────────

class EnemySpawner:
    def __init__(self):
        self.wave          = 1
        self.spawn_queue   = []   # list of EnemyClass to spawn
        self.spawn_timer   = 0.0
        self.break_timer   = 0.0
        self.wave_cleared  = False
        self.boss_spawned  = False

    def start_wave(self):
        self.wave_cleared = False
        self.boss_spawned = False
        self.spawn_queue  = []

        config = self._get_wave_config()
        scale  = self._get_scale()

        for cls, count in config.squads:
            scaled = count + scale
            self.spawn_queue.extend([cls] * scaled)
        random.shuffle(self.spawn_queue)

        self.spawn_timer = 0.5
        self._pending_event = config.event

    def _get_wave_config(self):
        idx = (self.wave - 1) % len(WAVE_TEMPLATES)
        return WAVE_TEMPLATES[idx]

    def _get_scale(self):
        """wave 11+ 부터 카운트 보너스"""
        if self.wave > 10:
            return (self.wave - 10) // 2
        return 0

    def update(self, dt, game):
        # 모든 적 소환 완료 + 화면에 적 없음 → 다음 웨이브 준비
        if not self.spawn_queue and not game.enemies:
            # 보스 이벤트 처리
            if hasattr(self, '_pending_event') and self._pending_event == 'boss' \
               and not self.boss_spawned:
                self.boss_spawned = True
                bp = self._random_border_pos()
                game.enemies.append(BossEnemy(bp))
                # 보스가 남아있으면 다음 웨이브 이동 안 함
                return

            self.break_timer += dt
            if not self.wave_cleared:
                self.wave_cleared = True
            if self.break_timer >= game.wave_break_time:
                self.break_timer = 0.0
                self.wave += 1
                self.start_wave()
        else:
            self.break_timer = 0.0

        # 대기 중인 적 소환
        if self.spawn_queue:
            self.spawn_timer -= dt
            if self.spawn_timer <= 0:
                interval = random.uniform(0.4, 0.8)
                self.spawn_timer = interval
                cls = self.spawn_queue.pop(0)
                pos = self._random_border_pos()
                game.enemies.append(cls(pos))

                # elite_rush 이벤트
                if hasattr(self, '_pending_event') and \
                   self._pending_event == 'elite_rush' and not self.spawn_queue:
                    count = random.randint(3, 5)
                    for _ in range(count):
                        ep = self._random_border_pos()
                        e = random.choice([EliteRunner, EliteBrute])(ep)
                        game.enemies.append(e)
                    self._pending_event = None

                # wave 15+ 엘리트 추가
                if self.wave >= 15 and random.random() < 0.12:
                    ep = self._random_border_pos()
                    game.enemies.append(random.choice([EliteRunner, EliteBrute])(ep))

                # wave 20+ 보스 동시
                if self.wave >= 20 and not self.boss_spawned \
                   and not self.spawn_queue:
                    self.boss_spawned = True
                    bp = self._random_border_pos()
                    game.enemies.append(BossEnemy(bp))

    def _random_border_pos(self):
        margin = 180
        side = random.randint(0, 3)
        if side == 0: return vec(random.randint(0, WORLD_WIDTH), -margin)
        if side == 1: return vec(random.randint(0, WORLD_WIDTH), WORLD_HEIGHT + margin)
        if side == 2: return vec(-margin, random.randint(0, WORLD_HEIGHT))
        return vec(WORLD_WIDTH + margin, random.randint(0, WORLD_HEIGHT))
