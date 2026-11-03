"""World state: chunk cache, cells, objects and structures (local IWorldService implementation)."""
import math
from collections import defaultdict
from dataclasses import dataclass

from config import CHUNK_SIZE, WORLD_SIZE
from entities.world_object import WorldObject
from interfaces.world_interface import IWorldService
from world.chunks import Chunk, FLAG_CACHE, FLAG_CACHE_FOUND
from world.terrain_layers import DepositTable
from world.world_generation import WorldGenerator


@dataclass
class CellInfo:
    x: int
    y: int
    terrain: int
    depth: int
    region: int
    flags: int
    work: float


class WorldManager(IWorldService):
    def __init__(self, seed: int, terrain_db, resources: dict, items_db, size: int = WORLD_SIZE):
        self.seed = seed
        self.tdb = terrain_db
        self.res = resources
        self.items_db = items_db
        self.size = size
        self.gen = WorldGenerator(seed, terrain_db, resources, size)
        self.deposits = DepositTable(resources, terrain_db)
        self.chunks: dict[tuple[int, int], Chunk] = {}
        self.obj_grid: dict[tuple[int, int], list[WorldObject]] = defaultdict(list)
        self.objects_by_id: dict[str, WorldObject] = {}
        # Saved state applied when a chunk is generated (persisted world changes).
        self.cell_overrides: dict[tuple[int, int], tuple[int, float, int]] = {}
        self.object_overrides: dict[str, dict] = {}
        self.dynamic_count = 0
        self.dirty_cells: set[tuple[int, int]] = set()

    # ------------------------------------------------------------ chunks
    def chunk(self, cx: int, cy: int) -> Chunk:
        ch = self.chunks.get((cx, cy))
        if ch is None:
            ch, objs = self.gen.generate_chunk(cx, cy)
            self.chunks[(cx, cy)] = ch
            for otype, x, y, oid in objs:
                odef = self.res["objects"][otype]
                obj = WorldObject(oid, "object", otype, x, y, odef)
                st = self.object_overrides.get(oid)
                if st is not None:
                    obj.apply_state(st)
                if not obj.removed:
                    self._index(obj)
                else:
                    self.objects_by_id[oid] = obj  # keep for saving
            base_x, base_y = cx * CHUNK_SIZE, cy * CHUNK_SIZE
            for (x, y), (depth, work, flags) in self.cell_overrides.items():
                if base_x <= x < base_x + CHUNK_SIZE and base_y <= y < base_y + CHUNK_SIZE:
                    i = Chunk.local_index(x, y)
                    ch.depth[i] = depth
                    ch.work[i] = work
                    ch.flags[i] = flags
                    ch.modified.add(i)
        return ch

    def ensure_area(self, x: float, y: float, radius: float, budget: int = 99):
        made = 0
        r = int(radius // CHUNK_SIZE) + 1
        ccx, ccy = int(x) // CHUNK_SIZE, int(y) // CHUNK_SIZE
        n = self.size // CHUNK_SIZE + (1 if self.size % CHUNK_SIZE else 0)
        for cy in range(max(0, ccy - r), min(n, ccy + r + 1)):
            for cx in range(max(0, ccx - r), min(n, ccx + r + 1)):
                if (cx, cy) not in self.chunks:
                    self.chunk(cx, cy)
                    made += 1
                    if made >= budget:
                        return made
        return made

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.size and 0 <= y < self.size

    def cell(self, x: int, y: int) -> CellInfo | None:
        if not self.in_bounds(x, y):
            return None
        ch = self.chunk(x // CHUNK_SIZE, y // CHUNK_SIZE)
        i = Chunk.local_index(x, y)
        return CellInfo(x, y, ch.terrain[i], ch.depth[i], ch.region[i], ch.flags[i], ch.work[i])

    def depth_at(self, x: float, y: float) -> int:
        ix, iy = int(x), int(y)
        if not self.in_bounds(ix, iy):
            return 0
        ch = self.chunks.get((ix // CHUNK_SIZE, iy // CHUNK_SIZE))
        return ch.depth[Chunk.local_index(ix, iy)] if ch else 0

    def terrain_def_at(self, x: float, y: float) -> dict:
        c = self.cell(int(x), int(y))
        return self.tdb.get(c.terrain) if c else self.tdb.get(0)

    def region_at(self, x: float, y: float) -> str:
        c = self.cell(int(x), int(y))
        return self.tdb.region_name(c.region) if c else "wasteland"

    def set_work(self, x: int, y: int, work: float):
        ch = self.chunk(x // CHUNK_SIZE, y // CHUNK_SIZE)
        i = Chunk.local_index(x, y)
        ch.work[i] = work
        ch.modified.add(i)

    # ------------------------------------------------------------ digging (authoritative)
    def complete_dig_layer(self, x: int, y: int) -> list:
        """Advance the cell one layer deeper and return what was unearthed."""
        ch = self.chunk(x // CHUNK_SIZE, y // CHUNK_SIZE)
        i = Chunk.local_index(x, y)
        depth = ch.depth[i] + 1
        ch.depth[i] = depth
        ch.work[i] = 0.0
        ch.modified.add(i)
        self.dirty_cells.add((x, y))
        region = self.tdb.region_name(ch.region[i])
        found = self.deposits.layer_deposit(self.seed, x, y, depth, region, ch.terrain[i])
        cache = False
        if ch.flags[i] & FLAG_CACHE and not ch.flags[i] & FLAG_CACHE_FOUND and depth >= self.tdb.regions[region]["cache_depth"]:
            ch.flags[i] |= FLAG_CACHE_FOUND
            found = found + self.deposits.cache_contents(self.seed, x, y, region)
            cache = True
        return found, cache

    def preview_deposit(self, x: int, y: int) -> bool:
        """Does the next layer contain something? Drives the 'shovel hits something' clue."""
        c = self.cell(x, y)
        if c is None:
            return False
        region = self.tdb.region_name(c.region)
        nxt = c.depth + 1
        if c.flags & FLAG_CACHE and not c.flags & FLAG_CACHE_FOUND and nxt >= self.tdb.regions[region]["cache_depth"]:
            return True
        if nxt > 3:
            return False
        return bool(self.deposits.layer_deposit(self.seed, x, y, nxt, region, c.terrain))

    def modified_cells(self):
        for (cx, cy), ch in self.chunks.items():
            for i in ch.modified:
                x = cx * CHUNK_SIZE + i % CHUNK_SIZE
                y = cy * CHUNK_SIZE + i // CHUNK_SIZE
                yield x, y, ch.depth[i], float(ch.work[i]), ch.flags[i]
        loaded = set(self.chunks)
        for (x, y), (d, w, f) in self.cell_overrides.items():
            if (x // CHUNK_SIZE, y // CHUNK_SIZE) not in loaded:
                yield x, y, d, w, f

    # ------------------------------------------------------------ objects
    def _index(self, obj: WorldObject):
        self.obj_grid[obj.cell].append(obj)
        self.objects_by_id[obj.id] = obj

    def objects_near(self, x: float, y: float, radius: float) -> list:
        out = []
        r = int(math.ceil(radius)) + 1
        ix, iy = int(x), int(y)
        r2 = radius * radius
        for cx in range(ix - r, ix + r + 1):
            for cy in range(iy - r, iy + r + 1):
                for o in self.obj_grid.get((cx, cy), ()):
                    if not o.removed and (o.x - x) ** 2 + (o.y - y) ** 2 <= r2:
                        out.append(o)
        return out

    def objects_in_rect(self, x0: int, y0: int, x1: int, y1: int):
        grid = self.obj_grid
        for cx in range(x0, x1 + 1):
            for cy in range(y0, y1 + 1):
                lst = grid.get((cx, cy))
                if lst:
                    for o in lst:
                        if not o.removed:
                            yield o

    def blocking_near(self, x: float, y: float, radius: float):
        return [o for o in self.objects_near(x, y, radius + 1.0) if o.blocking]

    def cell_has_blocker(self, x: int, y: int) -> bool:
        for o in self.objects_near(x + 0.5, y + 0.5, 1.0):
            if o.blocking and int(o.x) == x and int(o.y) == y:
                return True
        return False

    def remove_object(self, obj: WorldObject) -> None:
        obj.removed = True
        lst = self.obj_grid.get(obj.cell)
        if lst and obj in lst:
            lst.remove(obj)
        if obj.kind == "pickup" or (obj.kind == "structure" and not obj.generated):
            self.objects_by_id.pop(obj.id, None)

    def _new_dynamic_id(self, prefix: str) -> str:
        self.dynamic_count += 1
        return f"{prefix}{self.dynamic_count}"

    def place_structure(self, struct_type: str, x: float, y: float, oid: str | None = None) -> WorldObject:
        sdef = self.res["structures"][struct_type]
        obj = WorldObject(oid or self._new_dynamic_id("b"), "structure", struct_type, x, y, sdef, generated=False)
        obj.uses = -1
        if "fuel" in sdef:
            obj.data.setdefault("fuel", sdef["fuel"])
        self._index(obj)
        return obj

    def spawn_pickup(self, stack, x: float, y: float, pop: bool = True, oid: str | None = None) -> WorldObject:
        idef = self.items_db.item(stack.item_id)
        obj = WorldObject(oid or self._new_dynamic_id("p"), "pickup", stack.item_id, x, y,
                          {"radius": 0.2, "item_name": idef["name"]}, generated=False)
        obj.uses = -1
        obj.stack = stack
        if pop:
            obj.z = 0.0
            obj.vz = 4.5
            obj.no_auto = 0.5
        self._index(obj)
        return obj

    def move_object(self, obj: WorldObject, x: float, y: float):
        old = obj.cell
        obj.x, obj.y = x, y
        if obj.cell != old:
            lst = self.obj_grid.get(old)
            if lst and obj in lst:
                lst.remove(obj)
            self.obj_grid[obj.cell].append(obj)

    def all_live_objects(self):
        for lst in self.obj_grid.values():
            for o in lst:
                if not o.removed:
                    yield o

    def persistent_objects(self):
        """Objects whose state differs from generation or that were created at runtime."""
        for oid, o in self.objects_by_id.items():
            if not o.generated or o.removed or o.uses != o.defn.get("uses", 1) or o.data:
                yield o
        loaded_ids = set(self.objects_by_id)
        for oid, st in self.object_overrides.items():
            if oid not in loaded_ids and st.get("kind", "object") == "object":
                yield oid, st

    def free_for_structure(self, x: int, y: int) -> bool:
        c = self.cell(x, y)
        if c is None or c.depth > 0:
            return False
        if self.tdb.get(c.terrain)["id"] == "rock":
            return False
        for o in self.objects_near(x + 0.5, y + 0.5, 0.9):
            if o.kind != "pickup" and (o.blocking or o.kind == "structure" or int(o.x) == x and int(o.y) == y):
                return False
        return True
