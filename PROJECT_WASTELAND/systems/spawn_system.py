"""Keeps a region- and time-appropriate creature population around the player."""
import math

from entities.enemy import Enemy, CHASE
from entities.wildlife import Wildlife

SPAWN_MIN, SPAWN_MAX = 17.0, 27.0
DESPAWN_DIST = 42.0
MAX_ENEMIES = 32
SAFE_START = 75.0  # seconds of grace near the spawn point at the start of a new game


class SpawnSystem:
    def __init__(self, session, enemy_data: dict):
        self.session = session
        self.defs = {k: v for k, v in enemy_data.items() if k != "spawn_tables"}
        self.tables = enemy_data["spawn_tables"]
        self.timer = 1.0
        self.elapsed = 0.0

    def make(self, etype: str, x: float, y: float) -> Enemy:
        d = self.defs[etype]
        cls = Wildlife if d.get("kind") == "wildlife" else Enemy
        return cls(etype, d, x, y)

    def target_population(self, region: str) -> float:
        t = self.tables.get(region, self.tables["wasteland"])
        area = math.pi * SPAWN_MAX * SPAWN_MAX
        n = t["density"] * area
        if self.session.is_night():
            n *= 1.6
        return n

    def update(self, dt: float):
        s = self.session
        self.elapsed += dt
        self.timer -= dt
        p = s.player
        alive = []
        for e in s.enemies:
            d = math.hypot(e.x - p.x, e.y - p.y)
            if e.dead and e.death_t > 25:
                continue
            if d > DESPAWN_DIST and e.state != CHASE:
                continue
            alive.append(e)
        s.enemies = alive
        if self.timer > 0 or p.dead:
            return
        self.timer = 2.0
        near = sum(1 for e in alive if not e.dead and math.hypot(e.x - p.x, e.y - p.y) < SPAWN_MAX + 4)
        region = s.world.region_at(p.x, p.y)
        target = self.target_population(region)
        if s.new_game_grace and self.elapsed < SAFE_START:
            target *= 0.35
        if near >= target or len(alive) >= MAX_ENEMIES:
            return
        self.spawn_one()

    def spawn_one(self, min_d=SPAWN_MIN, max_d=SPAWN_MAX) -> Enemy | None:
        s = self.session
        p = s.player
        for _ in range(12):
            a = s.rng.uniform(0, math.tau)
            d = s.rng.uniform(min_d, max_d)
            x, y = p.x + math.cos(a) * d, p.y + math.sin(a) * d
            if not s.world.in_bounds(int(x), int(y)) or x < 2 or y < 2 or x > s.world.size - 2 or y > s.world.size - 2:
                continue
            if s.world.blocking_near(x, y, 0.6) and any(math.hypot(o.x - x, o.y - y) < o.radius + 0.5 for o in s.world.blocking_near(x, y, 0.6)):
                continue
            region = s.world.region_at(x, y)
            table = self.tables.get(region, self.tables["wasteland"])["night" if s.is_night() else "day"]
            total = sum(w for _, w in table)
            r = s.rng.uniform(0, total)
            acc = 0
            for etype, w in table:
                acc += w
                if r <= acc:
                    e = self.make(etype, x, y)
                    s.enemies.append(e)
                    return e
        return None

    def populate_initial(self):
        s = self.session
        for _ in range(4):
            self.spawn_one(16, 30)
