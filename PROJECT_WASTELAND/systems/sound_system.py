"""Noise propagation: gameplay sounds that creatures can hear (separate from audio playback)."""
import math

NOISE_RADIUS = {
    "walk": 2.5,
    "sprint": 6.0,
    "dig": 8.0,
    "melee": 5.0,
    "chop": 7.0,
    "gunshot": 26.0,
    "explosion": 35.0,
    "search": 4.5,
}


class NoiseSystem:
    def __init__(self, session):
        self.session = session
        self.player_noise = 0.0   # 0..1 for the HUD noise meter
        self.events: list = []    # (x, y, radius, life) for debug / HUD rings

    def emit(self, x: float, y: float, kind: str, radius: float | None = None, from_player: bool = True):
        r = radius if radius is not None else NOISE_RADIUS.get(kind, 4.0)
        weather = self.session.weather
        r *= weather.noise_mult()
        for e in self.session.enemies:
            if not e.dead and abs(e.x - x) < r * 1.6 and abs(e.y - y) < r * 1.6:
                e.hear(x, y, r, kind)
        if from_player:
            self.player_noise = max(self.player_noise, min(1.0, r / 26.0))
            self.events.append([x, y, r, 0.6])

    def update(self, dt: float):
        self.player_noise = max(0.0, self.player_noise - dt * 0.8)
        for ev in self.events:
            ev[3] -= dt
        self.events = [e for e in self.events if e[3] > 0]

    @staticmethod
    def distance(a, b) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])
