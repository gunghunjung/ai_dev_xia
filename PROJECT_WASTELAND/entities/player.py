"""Player character state. Input interpretation lives in systems/player_controller.py."""
from dataclasses import dataclass, asdict, field

from config import PLAYER_INV_W, PLAYER_INV_H, HOTBAR_SIZE
from entities.actor import Actor
from systems.inventory_system import GridInventory


@dataclass
class SurvivalStats:
    hunger: float = 85.0       # 100 = full
    thirst: float = 80.0       # 100 = hydrated
    stamina: float = 100.0
    body_temp: float = 37.0
    fatigue: float = 10.0      # 100 = exhausted
    bleeding: float = 0.0      # 0..3 severity
    injury: float = 0.0        # 0..100
    infection: float = 0.0     # 0..100
    max_stamina: float = 100.0
    days_survived: float = 0.0
    kills: int = 0
    exposure: float = 0.0      # contamination exposure accumulator
    flags: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)

    def load(self, d: dict):
        for k, v in d.items():
            if hasattr(self, k):
                setattr(self, k, v)


class Player(Actor):
    LOOK = {"body": (128, 116, 92), "skin": (226, 184, 148), "accent": (96, 72, 52), "height": 1.0, "hunch": 0.0}

    def __init__(self, x: float, y: float, db):
        super().__init__(x, y, radius=0.28, max_hp=100.0)
        self.db = db
        self.inventory = GridInventory(db, PLAYER_INV_W, PLAYER_INV_H, "player")
        self.equipped_uid: str | None = None
        self.body = None                     # ItemStack of worn clothing
        self.hotbar: list[str | None] = [None] * HOTBAR_SIZE
        self.stats = SurvivalStats()
        self.aim_angle = 0.0                 # world angle
        self.aim_point = (x + 1, y)
        self.aim_screen_angle = 0.0
        self.aiming = False
        self.sprinting = False
        self.sprint_toggle = False
        # Action timers
        self.attack_cd = 0.0
        self.attack_anim = None              # dict(t, dur, style, arc, weapon)
        self.dodge_t = 0.0
        self.dodge_cd = 0.0
        self.dodge_dir = (0.0, 0.0)
        self.iframes = 0.0
        self.reload_t = 0.0
        self.reload_total = 0.0
        self.recoil = 0.0
        self.spread_bloom = 0.0
        self.dig_target = None               # (x, y)
        self.digging = False
        self.dig_sound_t = 0.0
        self.dig_noise_t = 0.0
        self.dig_hint_shown = False
        self.action = None                   # timed interaction dict
        self.placing = None                  # dict(uid, struct)
        self.step_t = 0.0
        self.noise_t = 0.0
        self.hurt_cd = 0.0
        self.regen_t = 0.0
        self.mass = 1.0
        self.spawn_point = (x, y)

    # ------------------------------------------------------------ equipment
    def equipped_stack(self):
        if self.equipped_uid is None:
            return None
        st = self.inventory.stacks.get(self.equipped_uid)
        if st is None:
            self.equipped_uid = None
        return st

    def equipped_item_id(self):
        st = self.equipped_stack()
        return st.item_id if st else None

    def weapon(self) -> dict:
        return self.db.weapon_for_item(self.equipped_item_id())

    def tool(self) -> dict | None:
        st = self.equipped_stack()
        if st is None:
            return None
        return self.db.item(st.item_id).get("tool")

    def carry_weight(self) -> float:
        w = self.inventory.total_weight()
        if self.body is not None:
            w += self.db.weight(self.body)
        return w

    def insulation(self) -> float:
        if self.body is None:
            return 0.0
        return self.db.item(self.body.item_id).get("armor", {}).get("insulation", 0.0)

    def defense(self) -> float:
        if self.body is None:
            return 0.0
        return self.db.item(self.body.item_id).get("armor", {}).get("defense", 0.0)

    def clean_hotbar(self):
        for i, uid in enumerate(self.hotbar):
            if uid is not None and uid not in self.inventory.stacks:
                # Consumables: re-point the slot at another stack of the same item if present.
                self.hotbar[i] = None

    def auto_hotbar(self, stack):
        idef = self.db.item(stack.item_id)
        if not (idef.get("equip") == "hand" or idef.get("place")):
            return
        if idef.get("category") == "natural":
            return  # branches are equippable but would flood the bar
        if stack.uid in self.hotbar:
            return
        for i, uid in enumerate(self.hotbar):
            if uid is None:
                self.hotbar[i] = stack.uid
                return

    @property
    def busy(self) -> bool:
        return self.action is not None or self.digging or self.dodge_t > 0
