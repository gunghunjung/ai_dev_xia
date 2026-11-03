"""Item / weapon definitions loaded from data files, and deterministic loot rolling."""
import itertools
import random
import time

from interfaces.inventory_interface import IItemDatabase

_uid_counter = itertools.count(1)
_uid_prefix = f"{int(time.time()) % 100000:05d}"


def new_uid() -> str:
    return f"{_uid_prefix}-{next(_uid_counter)}"


class ItemStack:
    __slots__ = ("uid", "item_id", "count", "durability", "ammo", "data")

    def __init__(self, item_id: str, count: int = 1, durability=None, ammo: int = 0, uid: str | None = None, data=None):
        self.uid = uid or new_uid()
        self.item_id = item_id
        self.count = count
        self.durability = durability
        self.ammo = ammo
        self.data = data or {}

    def to_dict(self) -> dict:
        return {"uid": self.uid, "item_id": self.item_id, "count": self.count,
                "durability": self.durability, "ammo": self.ammo, "data": self.data}

    @classmethod
    def from_dict(cls, d: dict) -> "ItemStack":
        return cls(d["item_id"], d.get("count", 1), d.get("durability"), d.get("ammo", 0), d.get("uid"), d.get("data"))

    def __repr__(self):
        return f"ItemStack({self.item_id} x{self.count})"


class ItemDatabase(IItemDatabase):
    def __init__(self, items: dict, weapons: dict):
        self.items = items
        self.weapons = weapons
        for item_id, d in items.items():
            if "weapon" in d and d["weapon"] not in weapons:
                raise ValueError(f"item {item_id} references unknown weapon {d['weapon']}")

    def item(self, item_id: str) -> dict:
        try:
            return self.items[item_id]
        except KeyError:
            raise KeyError(f"unknown item id: {item_id}") from None

    def exists(self, item_id: str) -> bool:
        return item_id in self.items

    def weapon(self, weapon_id: str):
        return self.weapons.get(weapon_id)

    def weapon_for_item(self, item_id: str | None):
        if item_id is None:
            return self.weapons["fists"]
        w = self.items.get(item_id, {}).get("weapon")
        return self.weapons.get(w) if w else self.weapons["fists"]

    def size(self, item_id: str) -> tuple[int, int]:
        w, h = self.item(item_id).get("size", [1, 1])
        return w, h

    def max_stack(self, item_id: str) -> int:
        return self.item(item_id).get("max_stack", 1)

    def weight(self, stack: ItemStack) -> float:
        return self.item(stack.item_id).get("weight", 0.0) * stack.count

    def name(self, item_id: str) -> str:
        return self.item(item_id).get("name", item_id)

    def new_stack(self, item_id: str, count: int = 1, durability=None) -> ItemStack:
        d = self.item(item_id)
        if durability is None and d.get("durability"):
            durability = float(d["durability"])
        return ItemStack(item_id, count, durability, 0)


def roll_table(table: list, rng: random.Random, rolls: int = 1) -> list[tuple[str, int]]:
    """Weighted table rows: [item_id, weight, min, max] or ['none', weight]."""
    out: dict[str, int] = {}
    total = sum(row[1] for row in table)
    if total <= 0:
        return []
    for _ in range(rolls):
        r = rng.uniform(0, total)
        acc = 0.0
        for row in table:
            acc += row[1]
            if r <= acc:
                if row[0] != "none":
                    n = rng.randint(row[2], row[3]) if len(row) >= 4 else 1
                    out[row[0]] = out.get(row[0], 0) + n
                break
    return list(out.items())


def roll_chances(rows: list, rng: random.Random) -> list[tuple[str, int]]:
    """Independent-chance rows: [item_id, chance, min, max]."""
    out = []
    for item_id, chance, lo, hi in rows:
        if rng.random() < chance:
            out.append((item_id, rng.randint(lo, hi)))
    return out
