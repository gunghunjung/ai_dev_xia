"""Chunk storage: compact per-cell arrays for a CHUNK_SIZE² block of the world."""
import array

from config import CHUNK_SIZE

FLAG_CACHE = 1          # a buried cache exists in this cell
FLAG_CACHE_FOUND = 2    # cache already dug up
FLAG_ROAD = 4
FLAG_BUILDING = 8


class Chunk:
    __slots__ = ("cx", "cy", "terrain", "depth", "region", "flags", "work", "objects", "modified", "generated_at")

    def __init__(self, cx: int, cy: int):
        n = CHUNK_SIZE * CHUNK_SIZE
        self.cx = cx
        self.cy = cy
        self.terrain = bytearray(n)
        self.depth = bytearray(n)
        self.region = bytearray(n)
        self.flags = bytearray(n)
        self.work = array.array("f", [0.0] * n)  # progress into the next layer
        self.objects: list = []
        self.modified: set[int] = set()
        self.generated_at = 0.0

    @staticmethod
    def local_index(x: int, y: int) -> int:
        return (y % CHUNK_SIZE) * CHUNK_SIZE + (x % CHUNK_SIZE)
