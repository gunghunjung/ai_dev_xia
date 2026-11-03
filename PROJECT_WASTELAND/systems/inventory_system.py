"""Grid inventory with item sizes, stacking, splitting, merging and weight."""
from interfaces.inventory_interface import IInventoryService
from systems.resource_system import ItemStack, ItemDatabase


class GridInventory(IInventoryService):
    def __init__(self, db: ItemDatabase, width: int, height: int, name: str = "inventory"):
        self.db = db
        self.w = width
        self.h = height
        self.name = name
        self.stacks: dict[str, ItemStack] = {}
        self.pos: dict[str, tuple[int, int]] = {}
        self.version = 0  # bumped on every change so UIs/caches can refresh

    # ------------------------------------------------------------ geometry
    def _cells(self, uid: str):
        x, y = self.pos[uid]
        w, h = self.db.size(self.stacks[uid].item_id)
        for dx in range(w):
            for dy in range(h):
                yield x + dx, y + dy

    def occupancy(self) -> dict[tuple[int, int], str]:
        occ = {}
        for uid in self.stacks:
            for c in self._cells(uid):
                occ[c] = uid
        return occ

    def can_place(self, item_id: str, x: int, y: int, ignore_uid: str | None = None) -> bool:
        w, h = self.db.size(item_id)
        if x < 0 or y < 0 or x + w > self.w or y + h > self.h:
            return False
        occ = self.occupancy()
        for dx in range(w):
            for dy in range(h):
                other = occ.get((x + dx, y + dy))
                if other is not None and other != ignore_uid:
                    return False
        return True

    def find_space(self, item_id: str):
        w, h = self.db.size(item_id)
        occ = self.occupancy()
        for y in range(self.h - h + 1):
            for x in range(self.w - w + 1):
                if all((x + dx, y + dy) not in occ for dx in range(w) for dy in range(h)):
                    return x, y
        return None

    def stack_at(self, x: int, y: int) -> ItemStack | None:
        uid = self.occupancy().get((x, y))
        return self.stacks.get(uid) if uid else None

    # ------------------------------------------------------------ mutation
    def _touch(self):
        self.version += 1

    def place(self, stack: ItemStack, x: int, y: int) -> bool:
        if not self.can_place(stack.item_id, x, y, stack.uid):
            return False
        self.stacks[stack.uid] = stack
        self.pos[stack.uid] = (x, y)
        self._touch()
        return True

    def insert_stack(self, stack: ItemStack) -> bool:
        """Place a whole stack: merge into same items first, then free space."""
        if stack.durability is None or self.db.max_stack(stack.item_id) > 1:
            self._merge_into_existing(stack)
            if stack.count <= 0:
                return True
        spot = self.find_space(stack.item_id)
        if spot is None:
            return False
        return self.place(stack, *spot)

    def _merge_into_existing(self, stack: ItemStack):
        cap = self.db.max_stack(stack.item_id)
        for other in self.stacks.values():
            if other is stack or other.item_id != stack.item_id or other.count >= cap:
                continue
            moved = min(cap - other.count, stack.count)
            other.count += moved
            stack.count -= moved
            self._touch()
            if stack.count <= 0:
                return

    def add_item(self, item_id: str, count: int, durability: float | None = None) -> int:
        remaining = count
        cap = self.db.max_stack(item_id)
        if cap > 1:
            probe = ItemStack(item_id, remaining)
            self._merge_into_existing(probe)
            remaining = probe.count
        while remaining > 0:
            spot = self.find_space(item_id)
            if spot is None:
                break
            n = min(cap, remaining)
            st = self.db.new_stack(item_id, n, durability)
            self.place(st, *spot)
            remaining -= n
        return remaining

    def remove_uid(self, uid: str) -> ItemStack | None:
        st = self.stacks.pop(uid, None)
        self.pos.pop(uid, None)
        if st is not None:
            self._touch()
        return st

    def remove_item(self, item_id: str, count: int) -> bool:
        if self.count(item_id) < count:
            return False
        # Consume from the smallest stacks first to keep the grid tidy.
        for st in sorted([s for s in self.stacks.values() if s.item_id == item_id], key=lambda s: s.count):
            take = min(st.count, count)
            st.count -= take
            count -= take
            if st.count <= 0:
                self.remove_uid(st.uid)
            if count == 0:
                break
        self._touch()
        return True

    def consume_one(self, uid: str) -> bool:
        st = self.stacks.get(uid)
        if st is None:
            return False
        st.count -= 1
        if st.count <= 0:
            self.remove_uid(uid)
        self._touch()
        return True

    def count(self, item_id: str) -> int:
        return sum(s.count for s in self.stacks.values() if s.item_id == item_id)

    def find_first(self, item_id: str) -> ItemStack | None:
        for s in self.stacks.values():
            if s.item_id == item_id:
                return s
        return None

    def move(self, uid: str, x: int, y: int) -> bool:
        st = self.stacks.get(uid)
        if st is None or not self.can_place(st.item_id, x, y, uid):
            return False
        self.pos[uid] = (x, y)
        self._touch()
        return True

    def split(self, uid: str, amount: int):
        st = self.stacks.get(uid)
        if st is None or amount <= 0 or amount >= st.count:
            return None
        spot = self.find_space(st.item_id)
        if spot is None:
            return None
        new = ItemStack(st.item_id, amount, st.durability)
        st.count -= amount
        self.place(new, *spot)
        return new

    def merge(self, src_uid: str, dst_uid: str) -> bool:
        src, dst = self.stacks.get(src_uid), self.stacks.get(dst_uid)
        if not src or not dst or src is dst or src.item_id != dst.item_id:
            return False
        cap = self.db.max_stack(dst.item_id)
        moved = min(cap - dst.count, src.count)
        if moved <= 0:
            return False
        dst.count += moved
        src.count -= moved
        if src.count <= 0:
            self.remove_uid(src_uid)
        self._touch()
        return True

    # ------------------------------------------------------------ queries
    def total_weight(self) -> float:
        return sum(self.db.weight(s) for s in self.stacks.values())

    def items(self):
        return [(s, self.pos[s.uid]) for s in self.stacks.values()]

    def clear(self):
        self.stacks.clear()
        self.pos.clear()
        self._touch()

    def to_list(self) -> list[dict]:
        out = []
        for s, (x, y) in self.items():
            d = s.to_dict()
            d["gx"], d["gy"] = x, y
            out.append(d)
        return out

    def load_list(self, rows: list[dict]):
        self.clear()
        for d in rows:
            if not self.db.exists(d["item_id"]):
                continue  # item removed from data files: drop silently instead of crashing
            st = ItemStack.from_dict(d)
            if not self.place(st, d.get("gx", 0), d.get("gy", 0)):
                self.insert_stack(st)
