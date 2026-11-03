"""Hostile creatures with a sight + hearing driven state machine.

States: idle, patrol, investigate, chase, attack, hit, stunned, flee, dead.
"""
import math
import random

from entities.actor import Actor

IDLE, PATROL, INVESTIGATE, CHASE, ATTACK, HIT, STUNNED, FLEE, DEAD = (
    "idle", "patrol", "investigate", "chase", "attack", "hit", "stunned", "flee", "dead")


class Enemy(Actor):
    _ids = 0

    def __init__(self, etype: str, defn: dict, x: float, y: float):
        super().__init__(x, y, defn.get("radius", 0.3), defn["hp"])
        Enemy._ids += 1
        self.id = Enemy._ids
        self.etype = etype
        self.defn = defn
        self.mass = defn.get("mass", 1.0)
        self.state = IDLE
        self.state_t = random.uniform(0.5, 2.5)
        self.home = (x, y)
        self.target = (x, y)
        self.last_seen = None
        self.lost_t = 0.0
        self.attack_cd = random.uniform(0.3, 1.0)
        self.windup = 0.0
        self.attack_anim = None
        self.think_t = random.uniform(0, 0.2)
        self.vocal_t = random.uniform(3, 8)
        self.speed = 0.0
        self.has_fled = False
        self.alert = 0.0  # "!" indicator timer
        self.rng = random.Random(self.id * 7919)
        self.dropped = False
        self.data_bleed = 0.0

    @property
    def name(self):
        return self.defn["name"]

    def set_state(self, state: str, t: float = 0.0):
        self.state = state
        self.state_t = t

    # ------------------------------------------------------------ perception
    def can_see(self, session) -> bool:
        p = session.player
        if p.dead:
            return False
        d = self.dist_to(p)
        sight = self.defn["sight"] * session.visibility_mult()
        if p.sprinting:
            sight *= 1.25
        if session.world.depth_at(p.x, p.y) >= 2:
            sight *= 0.6  # crouched in a deep hole
        if d > sight:
            return False
        if d < 2.0:
            return True
        # Rough line of sight against blocking props (walls, cars).
        steps = int(d / 0.7)
        for i in range(1, steps):
            t = i / steps
            sx, sy = self.x + (p.x - self.x) * t, self.y + (p.y - self.y) * t
            for o in session.world.objects_near(sx, sy, 0.6):
                if o.blocking and o.radius >= 0.45 and math.hypot(o.x - sx, o.y - sy) < o.radius:
                    return False
        return True

    def hear(self, x: float, y: float, radius: float, kind: str):
        if self.dead:
            return
        if math.hypot(x - self.x, y - self.y) > radius * self.defn.get("hearing", 1.0):
            return
        if self.state in (IDLE, PATROL, INVESTIGATE):
            err = 1.5
            self.target = (x + self.rng.uniform(-err, err), y + self.rng.uniform(-err, err))
            self.set_state(INVESTIGATE, 6.0)
            self.alert = 0.8

    # ------------------------------------------------------------ AI
    def walk_speed(self, session) -> float:
        base = self.defn["speed"]
        if session.is_night():
            base *= self.defn.get("night_speed_mult", 1.0)
        return base

    def chase_speed(self, session) -> float:
        base = self.defn["chase_speed"]
        if session.is_night():
            base *= self.defn.get("night_speed_mult", 1.0)
        return base

    def think(self, session, dt: float):
        p = session.player
        st = self.state
        sees = self.can_see(session)
        hp_frac = self.hp / self.max_hp
        if st in (IDLE, PATROL, INVESTIGATE) and sees:
            self.set_state(CHASE)
            self.alert = 1.2
            session.audio.play(self.defn.get("sound"), 0.7, (self.x, self.y))
        if st in (CHASE, HIT) and not self.has_fled and hp_frac <= self.defn.get("flee_hp", 0.0):
            self.has_fled = True
            self.set_state(FLEE, 6.0)
        if st == CHASE:
            if sees:
                self.last_seen = (p.x, p.y)
                self.lost_t = 0.0
            else:
                self.lost_t += dt
                if self.lost_t > 4.0:
                    if self.last_seen:
                        self.target = self.last_seen
                    self.set_state(INVESTIGATE, 6.0)

    def update(self, session, dt: float):
        self.tick_common(dt)
        if self.dead:
            return
        self.attack_cd = max(0.0, self.attack_cd - dt)
        self.alert = max(0.0, self.alert - dt)
        self.think_t -= dt
        if self.think_t <= 0:
            far = self.dist_to(session.player) > 24
            interval = 0.25 if far else 0.1
            self.think(session, interval)
            self.think_t = interval
        self.vocal_t -= dt
        if self.vocal_t <= 0:
            self.vocal_t = self.rng.uniform(5, 12)
            if self.state in (CHASE, INVESTIGATE, IDLE):
                session.audio.play(self.defn.get("sound"), 0.35, (self.x, self.y))
        mx = my = 0.0
        speed = 0.0
        st = self.state
        p = session.player
        if self.stun > 0:
            self.state = STUNNED if st != HIT else HIT
        if st == STUNNED:
            if self.stun <= 0:
                self.set_state(CHASE)
        elif st == HIT:
            self.state_t -= dt
            if self.state_t <= 0:
                self.set_state(FLEE if self.has_fled and self.hp / self.max_hp <= self.defn.get("flee_hp", 0) else CHASE, 5.0)
        elif st == IDLE:
            self.state_t -= dt
            if self.state_t <= 0:
                a = self.rng.uniform(0, math.tau)
                r = self.rng.uniform(1.5, 5.0)
                self.target = (self.home[0] + math.cos(a) * r, self.home[1] + math.sin(a) * r)
                self.set_state(PATROL, 8.0)
        elif st in (PATROL, INVESTIGATE):
            self.state_t -= dt
            dx, dy = self.target[0] - self.x, self.target[1] - self.y
            d = math.hypot(dx, dy)
            if d < 0.4 or self.state_t <= 0:
                self.set_state(IDLE, self.rng.uniform(1.5, 4.0))
            else:
                mx, my = dx / d, dy / d
                speed = self.walk_speed(session) * (1.15 if st == INVESTIGATE else 0.7)
        elif st == CHASE:
            dx, dy = p.x - self.x, p.y - self.y
            d = math.hypot(dx, dy) or 1e-6
            reach = self.defn["attack_range"] + p.radius
            if d <= reach and self.attack_cd <= 0 and not p.dead:
                self.set_state(ATTACK)
                self.windup = self.defn["attack_windup"]
                self.facing = math.atan2(dy, dx)
            elif d > 0.6 * reach:
                mx, my = dx / d, dy / d
                speed = self.chase_speed(session)
        elif st == ATTACK:
            self.windup -= dt
            if self.windup <= 0 and self.attack_anim is None:
                self.attack_anim = 0.0
                session.combat.enemy_strike(self)
                self.attack_cd = self.defn["attack_cooldown"]
            if self.attack_anim is not None:
                self.attack_anim += dt / 0.3
                if self.attack_anim >= 1.0:
                    self.attack_anim = None
                    self.set_state(CHASE)
        elif st == FLEE:
            self.state_t -= dt
            dx, dy = self.x - p.x, self.y - p.y
            d = math.hypot(dx, dy) or 1e-6
            if self.state_t <= 0 or d > self.defn["sight"] * 1.8:
                self.home = (self.x, self.y)
                self.set_state(IDLE, 2.0)
            else:
                mx, my = dx / d, dy / d
                speed = self.chase_speed(session) * 1.05

        self.moving = speed > 0 and self.stun <= 0
        if self.moving:
            self.facing = math.atan2(my, mx)
            self.walk_phase += dt * speed * 4.5
        elif st == CHASE:
            self.facing = self.angle_to(p)
        vx = mx * speed + self.kx
        vy = my * speed + self.ky
        if vx or vy:
            moved = session.move_actor(self, vx * dt, vy * dt)
            if self.moving and not moved:
                # Blocked: sidestep around the obstacle.
                side = 1 if (self.id % 2) else -1
                ang = self.facing + side * 1.2
                session.move_actor(self, math.cos(ang) * speed * dt, math.sin(ang) * speed * dt)
