import pygame
from config import PLAYER_IFRAME, PLAYER_MAX_HP, PLAYER_RADIUS, PLAYER_SPEED, WORLD_HEIGHT, WORLD_WIDTH, vec
from weapon import Weapon, WEAPONS

DASH_SPEED      = 680
DASH_DURATION   = 0.13
DASH_COOLDOWN   = 0.9


class Player:
    def __init__(self, pos):
        self.pos = vec(pos)
        self.velocity = vec()
        self.speed = PLAYER_SPEED
        self.radius = PLAYER_RADIUS
        self.max_hp = PLAYER_MAX_HP
        self.hp = PLAYER_MAX_HP
        self.dead = False
        self.hit_timer = 0.0
        self.aim_dir = vec(1, 0)

        # 무기
        self.weapons = [
            Weapon(WEAPONS['rifle']),
            Weapon(WEAPONS['shotgun']),
            Weapon(WEAPONS['smg']),
        ]
        self.weapon_idx = 0

        # 대시
        self.dash_timer = 0.0
        self.dash_cooldown = 0.0
        self.dash_dir = vec()

        # 킬 카운트/스코어는 game에서 관리
        self.kills = 0

    @property
    def weapon(self):
        return self.weapons[self.weapon_idx]

    @property
    def rect(self):
        return pygame.Rect(
            int(self.pos.x - self.radius), int(self.pos.y - self.radius),
            self.radius * 2, self.radius * 2
        )

    def switch_weapon(self, idx):
        if 0 <= idx < len(self.weapons):
            self.weapon_idx = idx

    def update(self, dt, input_state, mouse_world, obstacles):
        self.weapon.update(dt)
        self.hit_timer = max(0.0, self.hit_timer - dt)

        # 대시 쿨
        if self.dash_cooldown > 0:
            self.dash_cooldown -= dt
        if self.dash_timer > 0:
            self.dash_timer -= dt

        # 이동
        move = vec()
        if input_state['up']:    move.y -= 1
        if input_state['down']:  move.y += 1
        if input_state['left']:  move.x -= 1
        if input_state['right']: move.x += 1
        if move.length_squared() > 0:
            move = move.normalize()

        # 대시 중이면 대시 방향으로 빠르게
        if self.dash_timer > 0:
            self.velocity = self.dash_dir * DASH_SPEED
        else:
            self.velocity = move * self.speed

        # 조준
        look = mouse_world - self.pos
        if look.length_squared() > 0:
            self.aim_dir = look.normalize()

        self.pos.x += self.velocity.x * dt
        self._resolve_collision(obstacles, axis='x')
        self.pos.y += self.velocity.y * dt
        self._resolve_collision(obstacles, axis='y')
        self.pos.x = max(self.radius, min(WORLD_WIDTH - self.radius, self.pos.x))
        self.pos.y = max(self.radius, min(WORLD_HEIGHT - self.radius, self.pos.y))

    def start_dash(self):
        if self.dash_cooldown > 0 or self.dash_timer > 0:
            return False
        self.dash_dir = self.velocity.normalize() if self.velocity.length_squared() > 0 else self.aim_dir
        self.dash_timer = DASH_DURATION
        self.dash_cooldown = DASH_COOLDOWN
        return True

    def _resolve_collision(self, obstacles, axis='x'):
        rect = self.rect
        for obstacle in obstacles:
            if rect.colliderect(obstacle):
                if axis == 'x':
                    if self.velocity.x > 0:
                        self.pos.x = obstacle.left - self.radius
                    elif self.velocity.x < 0:
                        self.pos.x = obstacle.right + self.radius
                else:
                    if self.velocity.y > 0:
                        self.pos.y = obstacle.top - self.radius
                    elif self.velocity.y < 0:
                        self.pos.y = obstacle.bottom + self.radius
                rect = self.rect

    def take_damage(self, amount):
        if self.hit_timer > 0 or self.dead:
            return False
        if self.dash_timer > 0:  # 대시 중 무적
            return False
        self.hp -= amount
        self.hit_timer = PLAYER_IFRAME
        if self.hp <= 0:
            self.hp = 0
            self.dead = True
        return True

    def draw(self, surface, camera):
        is_dashing = self.dash_timer > 0
        body_color = (180, 255, 255) if is_dashing else (
            (255, 245, 245) if self.hit_timer > 0 else (120, 205, 214)
        )
        screen_pos = camera.world_to_screen(self.pos)
        # 몸통
        pygame.draw.circle(surface, body_color, screen_pos, self.radius)
        pygame.draw.circle(surface, (36, 38, 44), screen_pos, self.radius - 6)
        # 총구
        gun_tip = self.pos + self.aim_dir * 26
        pygame.draw.line(surface, (70, 70, 70), screen_pos, camera.world_to_screen(gun_tip), 6)
        pygame.draw.line(surface, (160, 160, 160), screen_pos, camera.world_to_screen(gun_tip), 2)
        # 대시 잔상
        if is_dashing:
            ghost = self.pos - self.dash_dir * 16
            pygame.draw.circle(surface, (80, 200, 220), camera.world_to_screen(ghost), self.radius - 4, 2)
