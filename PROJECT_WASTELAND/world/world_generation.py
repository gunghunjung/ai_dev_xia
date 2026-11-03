"""Deterministic world generation.

A world is fully described by its seed. The *plan* (landmarks, roads, buildings, special
props) is computed once; chunks (terrain + props) are generated lazily on demand.
"""
import math
import random
from collections import defaultdict

from config import WORLD_SIZE, CHUNK_SIZE
from world.chunks import Chunk, FLAG_CACHE, FLAG_ROAD, FLAG_BUILDING
from world.terrain_layers import cell_rng


def _h(ix: int, iy: int, seed: int) -> float:
    n = (ix * 374761393 + iy * 668265263 + seed * 2246822519) & 0xFFFFFFFF
    n = ((n ^ (n >> 13)) * 1274126177) & 0xFFFFFFFF
    return ((n ^ (n >> 16)) & 0xFFFFFFFF) / 4294967295.0


def value_noise(x: float, y: float, seed: int) -> float:
    ix, iy = math.floor(x), math.floor(y)
    fx, fy = x - ix, y - iy
    u = fx * fx * (3 - 2 * fx)
    v = fy * fy * (3 - 2 * fy)
    a = _h(ix, iy, seed)
    b = _h(ix + 1, iy, seed)
    c = _h(ix, iy + 1, seed)
    d = _h(ix + 1, iy + 1, seed)
    top = a + (b - a) * u
    bot = c + (d - c) * u
    return top + (bot - top) * v


def fbm(x: float, y: float, seed: int, octaves: int = 3) -> float:
    total, amp, freq, norm = 0.0, 1.0, 1.0, 0.0
    for o in range(octaves):
        total += value_noise(x * freq, y * freq, seed + o * 1013) * amp
        norm += amp
        amp *= 0.5
        freq *= 2.0
    return total / norm


class WorldGenerator:
    def __init__(self, seed: int, terrain_db, resources: dict, size: int = WORLD_SIZE):
        self.seed = seed
        self.tdb = terrain_db
        self.res = resources
        self.size = size
        self.spawn = (size / 2 + 0.5, size / 2 + 0.5)
        self.landmarks: list[dict] = []
        self.roads: set[tuple[int, int]] = set()
        self.floor: dict[tuple[int, int], str] = {}
        self.building_cells: set[tuple[int, int]] = set()
        self.special: dict[tuple[int, int], list] = defaultdict(list)  # chunk → [(otype, x, y, id)]
        self.exclude: set[tuple[int, int]] = set()
        self._special_count = 0
        self._plan()

    # ------------------------------------------------------------ planning
    def _add_special(self, otype: str, x: float, y: float):
        cell = (int(x), int(y))
        if not (1 <= cell[0] < self.size - 1 and 1 <= cell[1] < self.size - 1):
            return
        oid = f"s{self._special_count}"
        self._special_count += 1
        self.special[(cell[0] // CHUNK_SIZE, cell[1] // CHUNK_SIZE)].append((otype, x, y, oid))
        self.exclude.add(cell)

    def _road(self, x0, y0, x1, y1, width=1):
        rng = random.Random(self.seed + int(x1 * 31 + y1))
        # Two-segment wobble so roads don't look ruler-straight.
        mx = (x0 + x1) / 2 + rng.uniform(-6, 6)
        my = (y0 + y1) / 2 + rng.uniform(-6, 6)
        for (ax, ay), (bx, by) in (((x0, y0), (mx, my)), ((mx, my), (x1, y1))):
            steps = int(max(abs(bx - ax), abs(by - ay)) * 2) + 1
            for i in range(steps + 1):
                t = i / steps
                cx, cy = ax + (bx - ax) * t, ay + (by - ay) * t
                for dx in range(-width, width + 1):
                    for dy in range(-width, width + 1):
                        if abs(dx) + abs(dy) <= width:
                            self.roads.add((int(cx) + dx, int(cy) + dy))

    def _plan(self):
        rng = random.Random(self.seed)
        sx, sy = self.spawn
        for spec in self.tdb.landmarks:
            for _ in range(80):
                ang = rng.uniform(0, math.tau)
                dist = rng.uniform(spec["min_dist"], spec["max_dist"])
                r = rng.uniform(*spec["radius"])
                x, y = sx + math.cos(ang) * dist, sy + math.sin(ang) * dist
                if not (r + 4 < x < self.size - r - 4 and r + 4 < y < self.size - r - 4):
                    continue
                if any(math.hypot(x - o["x"], y - o["y"]) < r + o["r"] + 8 for o in self.landmarks):
                    continue
                self.landmarks.append({"region": spec["region"], "x": x, "y": y, "r": r})
                break
        for lm in self.landmarks:
            self._road(sx, sy, lm["x"], lm["y"], width=1)
        for lm in self.landmarks:
            builder = {"ruins": self._plan_ruins, "gas_station": self._plan_gas_station,
                       "military": self._plan_military}.get(lm["region"])
            if builder:
                builder(lm, random.Random(self.seed ^ int(lm["x"] * 1000 + lm["y"])))
        self._plan_spawn_area(rng)

    def _building(self, x0, y0, w, h, rng, gap_side=None, decay=0.3, interior=("scrap_pile", "skeleton")):
        x1, y1 = x0 + w - 1, y0 + h - 1
        door = rng.choice(["n", "s", "e", "w"]) if gap_side is None else gap_side
        for x in range(x0, x1 + 1):
            for y in range(y0, y1 + 1):
                self.building_cells.add((x, y))
                edge = x in (x0, x1) or y in (y0, y1)
                if not edge:
                    self.floor[(x, y)] = "rubble"
                    continue
                mid_x, mid_y = (x0 + x1) // 2, (y0 + y1) // 2
                is_door = ((door == "n" and y == y0 and abs(x - mid_x) <= 1) or (door == "s" and y == y1 and abs(x - mid_x) <= 1)
                           or (door == "w" and x == x0 and abs(y - mid_y) <= 1) or (door == "e" and x == x1 and abs(y - mid_y) <= 1))
                self.floor[(x, y)] = "concrete"
                if is_door or rng.random() < decay:
                    self.exclude.add((x, y))
                    continue
                self._add_special("ruin_wall", x + 0.5, y + 0.5)
        for _ in range(rng.randint(1, 2)):
            ix, iy = rng.randint(x0 + 1, x1 - 1), rng.randint(y0 + 1, y1 - 1)
            if (ix, iy) not in self.exclude:
                self._add_special(rng.choice(interior), ix + 0.5, iy + 0.5)

    def _plan_ruins(self, lm, rng):
        cx, cy, r = int(lm["x"]), int(lm["y"]), lm["r"]
        for off in (-6, 6):
            self._road(cx - r, cy + off, cx + r, cy + off, 0)
            self._road(cx + off, cy - r, cx + off, cy + r, 0)
        placed = []
        for _ in range(40):
            w, h = rng.randint(4, 7), rng.randint(4, 7)
            x0 = rng.randint(int(cx - r * 0.8), int(cx + r * 0.8) - w)
            y0 = rng.randint(int(cy - r * 0.8), int(cy + r * 0.8) - h)
            rect = (x0 - 1, y0 - 1, x0 + w + 1, y0 + h + 1)
            if any(not (rect[2] < o[0] or rect[0] > o[2] or rect[3] < o[1] or rect[1] > o[3]) for o in placed):
                continue
            if any((x, y) in self.roads for x in range(x0, x0 + w) for y in range(y0, y0 + h)):
                continue
            placed.append(rect)
            self._building(x0, y0, w, h, rng, decay=0.35, interior=("scrap_pile", "skeleton", "dumpster", "scrap_pile"))
            if len(placed) >= 9:
                break
        for _ in range(4):
            self._add_special("car_wreck", cx + rng.uniform(-r, r) * 0.7, cy + rng.choice((-6, 6)) + 0.5)

    def _plan_gas_station(self, lm, rng):
        cx, cy = int(lm["x"]), int(lm["y"])
        for x in range(cx - 4, cx + 5):
            for y in range(cy - 2, cy + 3):
                self.floor[(x, y)] = "concrete"
                self.exclude.add((x, y))
        for i in range(3):
            self._add_special("gas_pump", cx - 2 + i * 2 + 0.5, cy + 0.5)
        self._building(cx - 3, cy - 7, 5, 4, rng, gap_side="s", decay=0.2, interior=("dumpster", "scrap_pile"))
        for _ in range(3):
            self._add_special("car_wreck", cx + rng.uniform(-6, 6), cy + rng.uniform(3.5, 6))

    def _plan_military(self, lm, rng):
        cx, cy, r = int(lm["x"]), int(lm["y"]), lm["r"]
        half = int(r * 0.7)
        sx, sy = self.spawn
        gate = "w" if abs(sx - cx) > abs(sy - cy) and sx < cx else "e" if abs(sx - cx) > abs(sy - cy) else ("n" if sy < cy else "s")
        self._building(cx - half, cy - half, half * 2, half * 2, rng, gap_side=gate, decay=0.12, interior=("military_crate",))
        for x in range(cx - 3, cx + 4):
            for y in range(cy - 3, cy + 4):
                self.floor[(x, y)] = "concrete"
        for _ in range(6):
            x = cx + rng.uniform(-half + 1.5, half - 1.5)
            y = cy + rng.uniform(-half + 1.5, half - 1.5)
            if (int(x), int(y)) not in self.exclude:
                self._add_special("military_crate", x, y)

    def _plan_spawn_area(self, rng):
        sx, sy = self.spawn
        for x in range(int(sx) - 3, int(sx) + 4):
            for y in range(int(sy) - 3, int(sy) + 4):
                self.exclude.add((x, y))

        def ring(otype, dmin, dmax, n):
            for _ in range(n):
                for _try in range(30):
                    a = rng.uniform(0, math.tau)
                    d = rng.uniform(dmin, dmax)
                    x, y = sx + math.cos(a) * d, sy + math.sin(a) * d
                    if (int(x), int(y)) not in self.exclude and (int(x), int(y)) not in self.roads:
                        self._add_special(otype, x, y)
                        break
        ring("stone_pile", 2.5, 5, 3)
        ring("branch_pile", 2.5, 5, 3)
        ring("dry_bush", 3, 6, 2)
        ring("dead_tree", 5, 8, 2)
        ring("puddle", 7, 10, 1)
        ring("skeleton", 9, 12, 1)
        # The wrecked patrol car is the only gun near spawn; it must be found, not given.
        gas = next((l for l in self.landmarks if l["region"] == "gas_station"), None)
        target = gas or (self.landmarks[0] if self.landmarks else {"x": sx + 20, "y": sy})
        ang = math.atan2(target["y"] - sy, target["x"] - sx)
        d = rng.uniform(15, 19)
        px, py = sx + math.cos(ang) * d - math.sin(ang) * 2.2, sy + math.sin(ang) * d + math.cos(ang) * 2.2
        self._add_special("police_car", px, py)

    # ------------------------------------------------------------ queries
    def region_at(self, x: int, y: int) -> str:
        best = "wasteland"
        best_d = 1e9
        for lm in self.landmarks:
            d = math.hypot(x + 0.5 - lm["x"], y + 0.5 - lm["y"])
            wobble = 0.85 + 0.3 * fbm(x * 0.15, y * 0.15, self.seed + 77, 2)
            if d < lm["r"] * wobble and d < best_d:
                best, best_d = lm["region"], d
        return best

    def terrain_at(self, x: int, y: int, region: str) -> str:
        fl = self.floor.get((x, y))
        if fl:
            return fl
        if (x, y) in self.roads:
            return "asphalt"
        reg = self.tdb.regions[region]
        n2 = fbm(x * 0.06 + 300, y * 0.06 + 300, self.seed + 5, 3)
        if n2 > reg.get("rock_threshold", 2.0) - 0.06 * (region == "wasteland"):
            return "rock"
        n = fbm(x * 0.09, y * 0.09, self.seed, 3)
        u = min(0.999, max(0.0, (n - 0.28) / 0.44))
        acc = 0.0
        total = sum(w for _, w in reg["terrain"])
        for name, w in reg["terrain"]:
            acc += w / total
            if u <= acc:
                return name
        return reg["terrain"][-1][0]

    # ------------------------------------------------------------ chunk generation
    def generate_chunk(self, cx: int, cy: int):
        ch = Chunk(cx, cy)
        objects = []
        tdb = self.tdb
        density = self.res["object_density"]
        sx, sy = self.spawn
        for ly in range(CHUNK_SIZE):
            for lx in range(CHUNK_SIZE):
                x, y = cx * CHUNK_SIZE + lx, cy * CHUNK_SIZE + ly
                i = ly * CHUNK_SIZE + lx
                if not (0 <= x < self.size and 0 <= y < self.size):
                    ch.terrain[i] = tdb.tid("rock")
                    continue
                region = self.region_at(x, y)
                tname = self.terrain_at(x, y, region)
                tidx = tdb.tid(tname)
                ch.terrain[i] = tidx
                ch.region[i] = tdb.region_index[region]
                flags = 0
                if (x, y) in self.roads:
                    flags |= FLAG_ROAD
                if (x, y) in self.building_cells:
                    flags |= FLAG_BUILDING
                rng = cell_rng(self.seed, x, y, 7)
                tdef = tdb.get(tidx)
                reg = tdb.regions[region]
                if (not flags and tdef["diggable"] and tdef["max_depth"] >= reg["cache_depth"]
                        and rng.random() < reg["cache_chance"]):
                    flags |= FLAG_CACHE
                ch.flags[i] = flags
                if (x, y) in self.exclude or flags & FLAG_BUILDING or x in (0, self.size - 1) or y in (0, self.size - 1):
                    continue
                r = rng.random()
                acc = 0.0
                for otype, dens in density.get(region, []):
                    acc += dens
                    if r < acc:
                        odef = self.res["objects"][otype]
                        if flags & FLAG_ROAD and odef.get("blocking") and otype != "car_wreck":
                            break
                        if odef.get("blocking") and math.hypot(x - sx, y - sy) < 4:
                            break
                        objects.append((otype, x + rng.uniform(0.25, 0.75), y + rng.uniform(0.25, 0.75), f"o{x}_{y}"))
                        break
        objects.extend(self.special.get((cx, cy), []))
        return ch, objects
