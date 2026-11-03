"""Lightweight frame profiler for the F3 debug overlay."""
import time
from collections import deque


class Profiler:
    def __init__(self, window: int = 60):
        self.samples: dict[str, deque] = {}
        self.counters: dict[str, int] = {}
        self._starts: dict[str, float] = {}
        self.window = window
        self.frame_times: deque = deque(maxlen=window)

    def begin(self, name: str):
        self._starts[name] = time.perf_counter()

    def end(self, name: str):
        start = self._starts.pop(name, None)
        if start is None:
            return
        dq = self.samples.setdefault(name, deque(maxlen=self.window))
        dq.append((time.perf_counter() - start) * 1000.0)

    def frame(self, dt: float):
        self.frame_times.append(dt)

    def avg_ms(self, name: str) -> float:
        dq = self.samples.get(name)
        return sum(dq) / len(dq) if dq else 0.0

    def fps(self) -> float:
        if not self.frame_times:
            return 0.0
        avg = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / avg if avg > 0 else 0.0

    def count(self, name: str, value: int):
        self.counters[name] = value
