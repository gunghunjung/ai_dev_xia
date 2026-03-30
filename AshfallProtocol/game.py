import random
import pygame
from asset_loader import AssetLoader
from config import (BG_COLOR, CAMERA_LERP, FPS, GRID_COLOR, HEIGHT,
                    RUBBLE_COLOR, TITLE, WAVE_BREAK_TIME, WIDTH,
                    WORLD_HEIGHT, WORLD_WIDTH, vec)
from effects import EffectManager
from enemy import EnemySpawner, Barrel, DropItem, BossEnemy
from player import Player
from ui import UI


class Camera:
    def __init__(self):
        self.pos = vec(WORLD_WIDTH / 2, WORLD_HEIGHT / 2)
        self.offset = vec()

    def update(self, target, shake_offset):
        self.pos += (target - self.pos) * CAMERA_LERP
        desired = self.pos - vec(WIDTH / 2, HEIGHT / 2)
        desired.x = max(0, min(WORLD_WIDTH  - WIDTH,  desired.x))
        desired.y = max(0, min(WORLD_HEIGHT - HEIGHT, desired.y))
        self.offset = desired + shake_offset

    def world_to_screen(self, pos):
        return int(pos.x - self.offset.x), int(pos.y - self.offset.y)

    def screen_to_world(self, pos):
        return vec(pos[0] + self.offset.x, pos[1] + self.offset.y)


# ── 맵 장애물 ──────────────────────────────────────────────────────────────

OBSTACLE_DEFS = [
    # (x, y, w, h)
    (340,  270,  220, 90),
    (780,  190,  130, 220),
    (1260, 510,  260, 110),
    (530,  960,  180, 160),
    (1030, 1040, 290, 130),
    (1710, 760,  180, 260),
    (1820, 270,  250, 100),
    (200,  700,  120, 300),
    (900,  700,  350, 80),
    (1500, 1200, 220, 180),
]

BARREL_POSITIONS = [
    (460, 315), (860, 290), (1350, 555),
    (620, 1010), (1120, 1090), (1800, 820),
]

DECOR_POINTS = [
    (180, 160), (620, 550), (920, 850), (1400, 300),
    (2100, 1210), (1900, 540), (320, 1280), (1580, 1380),
    (700, 1350), (1200, 200), (2200, 400), (100, 1000),
]


class Game:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption(TITLE)
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.assets = AssetLoader()
        self.ui = UI(self.assets)
        self.small_font = self.assets.font(17, bold=True)
        self.camera = Camera()
        self.running = True
        self.wave_break_time = WAVE_BREAK_TIME
        self.restart()

    # ── 초기화 ────────────────────────────────────────────────────────────

    def restart(self):
        self.player = Player((WORLD_WIDTH / 2, WORLD_HEIGHT / 2))
        self.effects = EffectManager()
        self.bullets = []
        self.enemies = []
        self.items   = []
        self.obstacles = [pygame.Rect(*d) for d in OBSTACLE_DEFS]
        self.barrels   = [Barrel(p) for p in BARREL_POSITIONS]
        self.spawner = EnemySpawner()
        self.spawner.start_wave()
        self.kills = 0
        self.score = 0
        self.paused = False
        self.prev_wave = 1
        self.ui.wave_flash = 0.0

    # ── 메인 루프 ─────────────────────────────────────────────────────────

    def run(self):
        while self.running:
            dt = min(self.clock.tick(FPS) / 1000.0, 0.05)
            self._handle_events()
            if not self.paused and not self.player.dead:
                self._update(dt)
            self.ui.update(dt)
            self._draw()
        pygame.quit()

    # ── 이벤트 ────────────────────────────────────────────────────────────

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if self.player.dead:
                        self.running = False
                    else:
                        self.paused = not self.paused

                elif event.key == pygame.K_q and self.paused:
                    self.running = False

                elif not self.paused and not self.player.dead:
                    if event.key == pygame.K_r:
                        self.player.weapon.begin_reload()
                    elif event.key == pygame.K_SPACE:
                        if self.player.start_dash():
                            self.effects.add_shake(2.5, 0.05)
                    elif event.key == pygame.K_1:
                        self.player.switch_weapon(0)
                    elif event.key == pygame.K_2:
                        self.player.switch_weapon(1)
                    elif event.key == pygame.K_3:
                        self.player.switch_weapon(2)

                elif event.key == pygame.K_RETURN and self.player.dead:
                    self.restart()

    def _input_state(self):
        keys = pygame.key.get_pressed()
        return {
            'up':    keys[pygame.K_w],
            'down':  keys[pygame.K_s],
            'left':  keys[pygame.K_a],
            'right': keys[pygame.K_d],
        }

    # ── 업데이트 ──────────────────────────────────────────────────────────

    def _update(self, dt):
        mouse_world = self.camera.screen_to_world(pygame.mouse.get_pos())
        self.player.update(dt, self._input_state(), mouse_world, self.obstacles)

        # 자동 재장전
        if self.player.weapon.ammo <= 0 and not self.player.weapon.is_reloading:
            self.player.weapon.begin_reload()

        # 사격
        if pygame.mouse.get_pressed()[0]:
            new_bullets = self.player.weapon.try_fire(self.player.pos, self.player.aim_dir)
            if new_bullets:
                self.bullets.extend(new_bullets)
                self.effects.add_shake(self.player.weapon.stats.recoil, 0.06)

        # 웨이브 클리어 알림
        if self.spawner.wave != self.prev_wave:
            self.ui.notify_wave_clear()
            self.prev_wave = self.spawner.wave

        self.spawner.update(dt, self)

        # 총알 업데이트
        world_rect = pygame.Rect(0, 0, WORLD_WIDTH, WORLD_HEIGHT)
        for b in self.bullets:
            b.update(dt, world_rect)
        self.bullets = [b for b in self.bullets if not b.dead]

        # 적 업데이트
        for e in self.enemies:
            e.update(dt, self.player, self.obstacles, self.bullets)
            if e.try_attack(self.player):
                if self.player.take_damage(e.damage):
                    self.effects.spawn_hit(self.player.pos, (200, 230, 255), 12)
                    self.effects.add_shake(9.0, 0.16)

        # 아이템 업데이트
        for item in self.items:
            item.update(dt)
        self.items = [i for i in self.items if not i.dead]

        self._resolve_bullet_hits()
        self._resolve_barrel_hits()
        self._resolve_item_pickups()
        self._cleanup_dead_enemies()

        self.effects.update(dt)
        self.camera.update(self.player.pos, self.effects.current_shake_offset())

    # ── 충돌 처리 ─────────────────────────────────────────────────────────

    def _resolve_bullet_hits(self):
        for bullet in self.bullets:
            if bullet.dead:
                continue
            br = pygame.Rect(int(bullet.pos.x - bullet.radius),
                             int(bullet.pos.y - bullet.radius),
                             bullet.radius * 2, bullet.radius * 2)
            # 장애물 충돌
            if any(br.colliderect(ob) for ob in self.obstacles):
                bullet.dead = True
                self.effects.spawn_hit(bullet.pos, (140, 130, 120), 4)
                continue

            if bullet.owner == 'player':
                # 적 충돌
                for enemy in self.enemies:
                    if enemy.dead:
                        continue
                    if bullet.pos.distance_to(enemy.pos) <= bullet.radius + enemy.radius:
                        enemy.take_damage(bullet.damage, bullet.velocity)
                        self.effects.spawn_hit(bullet.pos)
                        self.effects.add_damage_text(bullet.damage, enemy.pos)
                        self.effects.add_shake(3.0, 0.04)
                        bullet.dead = True
                        break
            elif bullet.owner == 'enemy':
                # 플레이어 충돌
                if bullet.pos.distance_to(self.player.pos) <= bullet.radius + self.player.radius:
                    if self.player.take_damage(bullet.damage):
                        self.effects.spawn_hit(self.player.pos, (200, 230, 255), 8)
                        self.effects.add_shake(6.0, 0.10)
                    bullet.dead = True

    def _resolve_barrel_hits(self):
        for barrel in self.barrels:
            if barrel.dead:
                continue
            # 총알 충돌
            for bullet in self.bullets:
                if bullet.dead:
                    continue
                if bullet.pos.distance_to(barrel.pos) <= bullet.radius + barrel.radius:
                    barrel.take_damage(bullet.damage)
                    bullet.dead = True
            # 폭발
            if barrel.exploded:
                self.effects.spawn_explosion(barrel.pos)
                self.effects.add_shake(15.0, 0.28)
                # 범위 데미지
                for enemy in self.enemies:
                    if enemy.pos.distance_to(barrel.pos) <= 100:
                        enemy.take_damage(60, enemy.pos - barrel.pos)
                if self.player.pos.distance_to(barrel.pos) <= 100:
                    self.player.take_damage(35)
        self.barrels = [b for b in self.barrels if not b.dead]

    def _resolve_item_pickups(self):
        for item in self.items:
            if item.dead:
                continue
            if self.player.pos.distance_to(item.pos) <= self.player.radius + item.size + 4:
                item.apply(self.player)

    def _cleanup_dead_enemies(self):
        alive = []
        for enemy in self.enemies:
            if enemy.dead:
                self.kills += 1
                self.score += enemy.score
                self.effects.spawn_death(enemy.pos, enemy.color)
                # 아이템 드랍
                r = random.random()
                if r < 0.15:
                    self.items.append(DropItem(enemy.pos, 'hp'))
                elif r < 0.30:
                    self.items.append(DropItem(enemy.pos, 'ammo'))
            else:
                alive.append(enemy)
        self.enemies = alive

    # ── 렌더링 ────────────────────────────────────────────────────────────

    def _draw_background(self):
        self.screen.fill(BG_COLOR)
        # 그리드
        grid = 64
        ox = int(self.camera.offset.x // grid) * grid
        oy = int(self.camera.offset.y // grid) * grid
        for x in range(ox, int(self.camera.offset.x + WIDTH) + grid, grid):
            sx = x - self.camera.offset.x
            pygame.draw.line(self.screen, GRID_COLOR, (sx, 0), (sx, HEIGHT))
        for y in range(oy, int(self.camera.offset.y + HEIGHT) + grid, grid):
            sy = y - self.camera.offset.y
            pygame.draw.line(self.screen, GRID_COLOR, (0, sy), (WIDTH, sy))
        # 장식 잔해 (원형 크레이터)
        for (x, y) in DECOR_POINTS:
            pos = self.camera.world_to_screen(vec(x, y))
            if -60 < pos[0] < WIDTH+60 and -60 < pos[1] < HEIGHT+60:
                pygame.draw.circle(self.screen, (44, 40, 36), pos, 32)
                pygame.draw.circle(self.screen, (60, 54, 46), pos, 14)
        # 장애물
        for ob in self.obstacles:
            r = pygame.Rect(ob.x - self.camera.offset.x, ob.y - self.camera.offset.y,
                            ob.width, ob.height)
            pygame.draw.rect(self.screen, RUBBLE_COLOR, r, border_radius=8)
            pygame.draw.rect(self.screen, (85, 78, 70), r, 3, border_radius=8)
            # 그림자 느낌 하이라이트
            pygame.draw.line(self.screen, (90, 85, 78),
                             (r.left+4, r.top+3), (r.right-4, r.top+3), 2)

    def _draw(self):
        self._draw_background()
        # 드럼통
        for barrel in self.barrels:
            barrel.draw(self.screen, self.camera)
        # 아이템
        for item in self.items:
            item.draw(self.screen, self.camera)
        # 총알
        for b in self.bullets:
            b.draw(self.screen, self.camera)
        # 적
        for e in self.enemies:
            e.draw(self.screen, self.camera)
        # 플레이어
        self.player.draw(self.screen, self.camera)
        # 이펙트
        self.effects.draw(self.screen, self.camera, self.small_font)
        # UI
        self.ui.draw(self.screen, self)
        pygame.display.flip()
