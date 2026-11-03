"""Static world objects: natural resources, searchable props, player structures and dropped items."""
import zlib


class WorldObject:
    __slots__ = ("id", "kind", "otype", "x", "y", "defn", "uses", "variant", "radius", "blocking",
                 "removed", "data", "stack", "z", "vz", "no_auto", "hp", "anim", "generated")

    def __init__(self, oid: str, kind: str, otype: str, x: float, y: float, defn: dict, generated: bool = True):
        self.id = oid
        self.kind = kind            # "object" | "structure" | "pickup"
        self.otype = otype
        self.x = x
        self.y = y
        self.defn = defn
        self.uses = defn.get("uses", 1)
        self.variant = zlib.crc32(oid.encode()) % 3
        self.radius = defn.get("radius", 0.3)
        self.blocking = defn.get("blocking", False)
        self.removed = False
        self.data: dict = {}
        self.stack = None
        self.z = 0.0
        self.vz = 0.0
        self.no_auto = 0.0
        self.hp = defn.get("hp", 0)
        self.anim = 0.0
        self.generated = generated

    @property
    def name(self) -> str:
        if self.kind == "pickup" and self.stack is not None:
            return self.defn.get("item_name", "아이템")
        return self.defn.get("name", self.otype)

    @property
    def interact(self):
        if self.kind == "pickup":
            return "pickup"
        if self.uses == 0 and self.defn.get("interact") in ("gather", "chop", "search"):
            return None
        return self.defn.get("interact")

    @property
    def cell(self) -> tuple[int, int]:
        return int(self.x), int(self.y)

    def state(self) -> dict:
        d = {"uses": self.uses, "removed": self.removed, "data": self.data}
        if self.hp:
            d["hp"] = self.hp
        if self.stack is not None:
            d["stack"] = self.stack.to_dict()
        return d

    def apply_state(self, state: dict):
        self.uses = state.get("uses", self.uses)
        self.removed = state.get("removed", False)
        self.data = state.get("data", {}) or {}
        if "hp" in state:
            self.hp = state["hp"]
