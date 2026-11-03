"""Bullets with object pooling."""


class Projectile:
    __slots__ = ("x", "y", "px", "py", "vx", "vy", "damage", "range_left", "weapon", "alive", "owner", "trail")

    def __init__(self):
        self.alive = False

    def reset(self, x, y, vx, vy, damage, max_range, weapon, owner):
        self.x = self.px = x
        self.y = self.py = y
        self.vx, self.vy = vx, vy
        self.damage = damage
        self.range_left = max_range
        self.weapon = weapon
        self.owner = owner
        self.alive = True
        self.trail = 0.0
        return self


class ProjectilePool:
    def __init__(self, size: int = 128):
        self.items = [Projectile() for _ in range(size)]

    def spawn(self, *args) -> Projectile | None:
        for p in self.items:
            if not p.alive:
                return p.reset(*args)
        # Pool exhausted: grow rather than drop the shot.
        p = Projectile()
        self.items.append(p)
        return p.reset(*args)

    def active(self):
        return [p for p in self.items if p.alive]

    def clear(self):
        for p in self.items:
            p.alive = False
