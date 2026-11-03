"""Pooled visual effects: particles, floating text, slash trails, flashes and ground decals."""
import math
import random


class Particle:
    __slots__ = ("x", "y", "z", "vx", "vy", "vz", "life", "max_life", "color", "size", "gravity", "kind", "alive")

    def __init__(self):
        self.alive = False


class FloatText:
    __slots__ = ("x", "y", "z", "text", "color", "life", "max_life", "scale")

    def __init__(self, x, y, text, color, life=0.9, scale=1.0):
        self.x, self.y, self.z = x, y, 30.0
        self.text, self.color = text, color
        self.life = self.max_life = life
        self.scale = scale


class Trail:
    __slots__ = ("x", "y", "angle", "arc", "radius", "life", "max_life", "color", "kind")

    def __init__(self, x, y, angle, arc, radius, life, color, kind="slash"):
        self.x, self.y, self.angle, self.arc, self.radius = x, y, angle, arc, radius
        self.life = self.max_life = life
        self.color = color
        self.kind = kind


class Flash:
    __slots__ = ("x", "y", "life", "max_life", "radius", "color", "angle")

    def __init__(self, x, y, life, radius, color, angle=0.0):
        self.x, self.y, self.life, self.max_life, self.radius, self.color, self.angle = x, y, life, life, radius, color, angle


class Decal:
    __slots__ = ("x", "y", "life", "size", "color", "seed")

    def __init__(self, x, y, life, size, color, seed):
        self.x, self.y, self.life, self.size, self.color, self.seed = x, y, life, size, color, seed


class Tracer:
    __slots__ = ("x0", "y0", "x1", "y1", "life", "max_life", "color")

    def __init__(self, x0, y0, x1, y1, life, color):
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1
        self.life = self.max_life = life
        self.color = color


class EffectsSystem:
    def __init__(self, max_particles: int = 900):
        self.particles = [Particle() for _ in range(max_particles)]
        self._cursor = 0
        self.texts: list[FloatText] = []
        self.trails: list[Trail] = []
        self.flashes: list[Flash] = []
        self.decals: list[Decal] = []
        self.tracers: list[Tracer] = []
        self.rng = random.Random()
        self.damage_numbers = True

    def _next(self) -> Particle:
        # Ring buffer: when full, the oldest particle is recycled.
        p = self.particles[self._cursor]
        self._cursor = (self._cursor + 1) % len(self.particles)
        return p

    def burst(self, x, y, color, n=8, speed=3.0, z=18.0, vz=(40, 120), gravity=260.0, size=(2, 4), life=(0.3, 0.7),
              kind="dot", angle=None, spread=math.pi):
        r = self.rng
        for _ in range(n):
            p = self._next()
            a = (angle + r.uniform(-spread, spread)) if angle is not None else r.uniform(0, math.tau)
            sp = r.uniform(speed * 0.4, speed)
            p.x, p.y, p.z = x, y, z
            p.vx, p.vy = math.cos(a) * sp, math.sin(a) * sp
            p.vz = r.uniform(*vz)
            p.life = p.max_life = r.uniform(*life)
            p.color = color if not isinstance(color, list) else r.choice(color)
            p.size = r.uniform(*size)
            p.gravity = gravity
            p.kind = kind
            p.alive = True

    # --- presets -----------------------------------------------------------
    def blood(self, x, y, angle, amount=1.0, color=(150, 30, 30)):
        self.burst(x, y, [color, (110, 20, 20), (180, 50, 40)], int(10 * amount), 3.5, 20, (30, 140), 300, (2, 4.5), (0.3, 0.6), "dot", angle, 0.9)
        if self.rng.random() < 0.8:
            self.decals.append(Decal(x + math.cos(angle) * 0.4, y + math.sin(angle) * 0.4, 25.0, self.rng.uniform(0.5, 1.0) * amount, color, self.rng.randint(0, 999)))

    def sparks(self, x, y, angle=None, n=10):
        self.burst(x, y, [(255, 230, 140), (255, 190, 80), (255, 255, 220)], n, 6.0, 18, (20, 90), 200, (1.5, 2.5), (0.15, 0.35), "spark", angle, 1.2)

    def dirt(self, x, y, color, n=8, z=4):
        dark = tuple(int(c * 0.75) for c in color)
        self.burst(x, y, [color, dark], n, 2.5, z, (60, 160), 380, (2.5, 5), (0.35, 0.7), "chunk")

    def dust(self, x, y, n=4):
        self.burst(x, y, [(200, 186, 160), (170, 158, 136)], n, 0.8, 2, (5, 20), -10, (4, 8), (0.4, 0.8), "smoke")

    def smoke(self, x, y, z=20):
        self.burst(x, y, [(90, 86, 82), (120, 116, 110)], 1, 0.3, z, (20, 35), -8, (5, 9), (1.0, 1.8), "smoke")

    def embers(self, x, y):
        self.burst(x, y, [(255, 180, 60), (255, 120, 30)], 1, 0.4, 14, (30, 60), -20, (1.5, 2.5), (0.6, 1.1), "spark")

    def text(self, x, y, text, color=(255, 255, 255), life=0.9, scale=1.0, damage=False):
        if damage and not self.damage_numbers:
            return
        self.texts.append(FloatText(x + self.rng.uniform(-0.15, 0.15), y, text, color, life, scale))

    def slash(self, x, y, angle, arc, radius, color=(255, 255, 255), life=0.16, kind="slash"):
        self.trails.append(Trail(x, y, angle, arc, radius, life, color, kind))

    def flash(self, x, y, radius=3.0, color=(255, 210, 140), life=0.07, angle=0.0):
        self.flashes.append(Flash(x, y, life, radius, color, angle))

    def tracer(self, x0, y0, x1, y1, color=(255, 230, 150), life=0.06):
        self.tracers.append(Tracer(x0, y0, x1, y1, life, color))

    # --- update ------------------------------------------------------------
    def update(self, dt: float):
        for p in self.particles:
            if not p.alive:
                continue
            p.life -= dt
            if p.life <= 0:
                p.alive = False
                continue
            p.x += p.vx * dt
            p.y += p.vy * dt
            p.vz -= p.gravity * dt
            p.z += p.vz * dt
            if p.z < 0:
                p.z = 0
                p.vz = -p.vz * 0.3
                p.vx *= 0.5
                p.vy *= 0.5
            drag = math.exp(-dt * 2.0)
            p.vx *= drag
            p.vy *= drag
        for lst in (self.texts, self.trails, self.flashes, self.tracers):
            for e in lst:
                e.life -= dt
        for t in self.texts:
            t.z += dt * 40
        self.texts = [t for t in self.texts if t.life > 0]
        self.trails = [t for t in self.trails if t.life > 0]
        self.flashes = [f for f in self.flashes if f.life > 0]
        self.tracers = [t for t in self.tracers if t.life > 0]
        for d in self.decals:
            d.life -= dt
        if len(self.decals) > 120:
            self.decals = self.decals[-120:]
        self.decals = [d for d in self.decals if d.life > 0]

    def particle_count(self) -> int:
        return sum(1 for p in self.particles if p.alive)
