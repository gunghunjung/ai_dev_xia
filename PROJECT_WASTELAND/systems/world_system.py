"""Day/night clock and per-tick world object upkeep (campfires, dropped items)."""
import math

from config import DAY_LENGTH, START_TIME_OF_DAY


def _lerp_col(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


# (time_of_day, ambient light colour)
_AMBIENT_KEYS = [
    (0.00, (52, 58, 96)),
    (0.20, (60, 64, 104)),
    (0.26, (200, 150, 130)),
    (0.32, (255, 244, 230)),
    (0.68, (255, 246, 232)),
    (0.75, (236, 160, 120)),
    (0.80, (88, 80, 120)),
    (1.00, (52, 58, 96)),
]


class WorldClock:
    def __init__(self, time_of_day: float = START_TIME_OF_DAY, day: int = 1):
        self.time = time_of_day
        self.day = day

    def update(self, dt: float):
        self.time += dt / DAY_LENGTH
        if self.time >= 1.0:
            self.time -= 1.0
            self.day += 1

    def advance_hours(self, hours: float):
        self.update(hours / 24.0 * DAY_LENGTH)

    @property
    def hour(self) -> float:
        return self.time * 24.0

    def is_night(self) -> bool:
        return self.time < 0.23 or self.time > 0.79

    def ambient(self) -> tuple[int, int, int]:
        t = self.time
        for (t0, c0), (t1, c1) in zip(_AMBIENT_KEYS, _AMBIENT_KEYS[1:]):
            if t0 <= t <= t1:
                return _lerp_col(c0, c1, (t - t0) / (t1 - t0) if t1 > t0 else 0)
        return _AMBIENT_KEYS[0][1]

    def light_level(self) -> float:
        c = self.ambient()
        return sum(c) / 765.0

    def base_temperature(self) -> float:
        """Ambient air temperature in °C (cold nights, hot afternoons)."""
        return 17.0 + 13.0 * math.sin((self.time - 0.33) * math.tau)

    def clock_text(self) -> str:
        h = int(self.hour)
        m = int((self.hour - h) * 60)
        return f"{self.day}일차 {h:02d}:{m:02d}"


class WorldSystem:
    def __init__(self, session):
        self.session = session
        self.fx_t = 0.0

    def update(self, dt: float):
        s = self.session
        s.clock.update(dt)
        self.fx_t -= dt
        emit = self.fx_t <= 0
        if emit:
            self.fx_t = 0.08
        p = s.player
        for o in s.world.objects_near(p.x, p.y, 22):
            if o.kind == "structure" and o.otype == "campfire":
                fuel = o.data.get("fuel", 0)
                if fuel > 0:
                    o.data["fuel"] = max(0.0, fuel - dt)
                    o.anim += dt
                    if emit:
                        s.effects.embers(o.x, o.y)
                        if s.effects.rng.random() < 0.25:
                            s.effects.smoke(o.x, o.y, 26)
            elif o.kind == "pickup":
                if o.vz or o.z > 0:
                    o.vz -= 22 * dt
                    o.z = max(0.0, o.z + o.vz * dt * 6)
                    if o.z <= 0 and o.vz < 0:
                        o.vz = 0.0
                o.no_auto = max(0.0, o.no_auto - dt)
                o.anim += dt

    def campfires_near(self, x: float, y: float, radius: float):
        return [o for o in self.session.world.objects_near(x, y, radius)
                if o.kind == "structure" and o.otype == "campfire" and o.data.get("fuel", 0) > 0]
