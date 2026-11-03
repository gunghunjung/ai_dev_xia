"""Broad-phase spatial hashing and circle collision helpers (world/tile units)."""
import math
from collections import defaultdict


class SpatialHash:
    """Buckets objects with .x/.y by integer cell. Rebuilt each tick for moving actors."""

    def __init__(self, cell: float = 2.0):
        self.cell = cell
        self.buckets: dict[tuple[int, int], list] = defaultdict(list)

    def clear(self):
        self.buckets.clear()

    def _key(self, x, y):
        return int(x // self.cell), int(y // self.cell)

    def insert(self, obj):
        self.buckets[self._key(obj.x, obj.y)].append(obj)

    def remove(self, obj):
        bucket = self.buckets.get(self._key(obj.x, obj.y))
        if bucket and obj in bucket:
            bucket.remove(obj)

    def query(self, x: float, y: float, r: float):
        c = self.cell
        x0, x1 = int((x - r) // c), int((x + r) // c)
        y0, y1 = int((y - r) // c), int((y + r) // c)
        b = self.buckets
        for cx in range(x0, x1 + 1):
            for cy in range(y0, y1 + 1):
                bucket = b.get((cx, cy))
                if bucket:
                    yield from bucket

    def __len__(self):
        return sum(len(v) for v in self.buckets.values())


def resolve_circle(ax: float, ay: float, ar: float, bx: float, by: float, br: float):
    """Push circle A out of circle B. Returns new (ax, ay) or None if not overlapping."""
    dx, dy = ax - bx, ay - by
    d2 = dx * dx + dy * dy
    rr = ar + br
    if d2 >= rr * rr:
        return None
    d = math.sqrt(d2)
    if d < 1e-6:
        return ax + rr, ay
    push = rr - d
    return ax + dx / d * push, ay + dy / d * push


def angle_diff(a: float, b: float) -> float:
    d = (a - b + math.pi) % (2 * math.pi) - math.pi
    return d


def segment_circle_hit(x0, y0, x1, y1, cx, cy, r):
    """Return param t∈[0,1] of first intersection of segment with circle, or None."""
    dx, dy = x1 - x0, y1 - y0
    fx, fy = x0 - cx, y0 - cy
    a = dx * dx + dy * dy
    if a < 1e-12:
        return 0.0 if fx * fx + fy * fy <= r * r else None
    b = 2 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - r * r
    disc = b * b - 4 * a * c
    if disc < 0:
        return None
    disc = math.sqrt(disc)
    t1 = (-b - disc) / (2 * a)
    t2 = (-b + disc) / (2 * a)
    if 0 <= t1 <= 1:
        return t1
    if 0 <= t2 <= 1:
        return t2
    if t1 < 0 < t2:
        return 0.0
    return None
