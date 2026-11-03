"""Weather state machine: clear / rain / fog / dust storm with smooth intensity blending."""
import random

WEATHER_INFO = {
    "clear": {"name": "맑음", "vis": 1.0, "temp": 0.0, "noise": 1.0, "duration": (150, 320)},
    "rain": {"name": "비", "vis": 0.8, "temp": -6.0, "noise": 0.7, "duration": (70, 150)},
    "fog": {"name": "안개", "vis": 0.55, "temp": -3.0, "noise": 1.0, "duration": (60, 130)},
    "dust": {"name": "먼지 폭풍", "vis": 0.5, "temp": 4.0, "noise": 0.6, "duration": (45, 100)},
}
_NEXT = {"clear": [("rain", 3), ("fog", 2), ("dust", 2), ("clear", 3)],
         "rain": [("clear", 6), ("fog", 2)], "fog": [("clear", 6), ("rain", 1)], "dust": [("clear", 7), ("fog", 1)]}


class WeatherSystem:
    def __init__(self, seed: int = 0, state: str = "clear"):
        self.rng = random.Random(seed ^ 0xBEEF)
        self.state = state
        self.prev = state
        self.blend = 1.0          # 0 → prev, 1 → state
        self.timer = self.rng.uniform(*WEATHER_INFO[state]["duration"])
        self.wetness = 0.0

    def set(self, state: str):
        self.prev = self.state
        self.state = state
        self.blend = 0.0
        self.timer = self.rng.uniform(*WEATHER_INFO[state]["duration"])

    def update(self, dt: float):
        self.timer -= dt
        if self.timer <= 0:
            options = _NEXT[self.state]
            total = sum(w for _, w in options)
            r = self.rng.uniform(0, total)
            acc = 0
            for name, w in options:
                acc += w
                if r <= acc:
                    self.set(name)
                    break
        if self.blend < 1.0:
            self.blend = min(1.0, self.blend + dt / 8.0)
        target_wet = 1.0 if self.state == "rain" else 0.0
        self.wetness += (target_wet - self.wetness) * min(1.0, dt * 0.1)

    def intensity(self, kind: str) -> float:
        v = 0.0
        if self.state == kind:
            v += self.blend
        if self.prev == kind:
            v += 1.0 - self.blend
        return min(1.0, v)

    def _mix(self, key: str) -> float:
        a = WEATHER_INFO[self.prev][key]
        b = WEATHER_INFO[self.state][key]
        return a + (b - a) * self.blend

    def visibility(self) -> float:
        return self._mix("vis")

    def temp_offset(self) -> float:
        return self._mix("temp")

    def noise_mult(self) -> float:
        return self._mix("noise")

    @property
    def name(self) -> str:
        return WEATHER_INFO[self.state]["name"]
