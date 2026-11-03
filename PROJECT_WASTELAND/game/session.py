"""GameSession: the complete simulation state of one playthrough (no rendering code).

Systems talk to each other through the session's small helper API (message, give_item,
damage_player, hitstop, ...). This keeps every system unit-testable headlessly.
"""
import math
import random

from config import (AUTOSAVE_INTERVAL, INTERACT_RANGE, PLAYER_CARRY_CAPACITY, WORLD_SIZE)
from engine.collision import resolve_circle, SpatialHash, angle_diff
from entities.player import Player
from entities.projectile import ProjectilePool
from interfaces.network_interface import LocalNetworkService
from systems.combat_system import CombatSystem
from systems.crafting_system import CraftingSystem
from systems.digging_system import DiggingSystem
from systems.effects_system import EffectsSystem
from systems.inventory_system import GridInventory
from systems.player_controller import PlayerController
from systems.resource_system import ItemDatabase, roll_table
from systems.sound_system import NoiseSystem
from systems.spawn_system import SpawnSystem
from systems.survival_system import SurvivalSystem
from systems.weather_system import WeatherSystem
from systems.world_system import WorldClock, WorldSystem
from world.terrain import TerrainDB
from world.world_manager import WorldManager


class GameData:
    """All static data files, loaded once and shared between sessions."""

    def __init__(self, assets):
        self.items = assets.load_json("items.json")
        self.weapons = assets.load_json("weapons.json")
        self.resources = assets.load_json("resources.json")
        self.terrain = assets.load_json("terrain.json")
        self.enemies = assets.load_json("enemies.json")
        self.recipes = assets.load_json("recipes.json")
        self.lore = assets.load_json("lore.json")
        self.input_profiles = assets.load_json("input_profiles.json")
        self.db = ItemDatabase(self.items, self.weapons)
        self.terrain_db = TerrainDB(self.terrain)
        self._validate()

    def _validate(self):
        for r in self.recipes:
            for item_id, _ in r["ingredients"] + [r["result"]]:
                self.db.item(item_id)
        for table in self.resources["loot_tables"].values():
            for row in table:
                if row[0] != "none":
                    self.db.item(row[0])
        for region in self.resources["dig_loot"].values():
            for table in region.values():
                for row in table:
                    if row[0] != "none":
                        self.db.item(row[0])


class GameSession:
    def __init__(self, data: GameData, settings: dict, audio, seed: int | None = None, new_game: bool = True):
        self.data = data
        self.db = data.db
        self.terrain_db = data.terrain_db
        self.resources = data.resources
        self.lore = data.lore
        self.settings = settings
        self.audio = audio
        self.input = None  # set by the scene (for rumble)
        self.seed = seed if seed is not None else random.randrange(1, 2 ** 31)
        self.rng = random.Random(self.seed ^ 0x5EED)
        self.network = LocalNetworkService()
        self.world = WorldManager(self.seed, self.terrain_db, self.resources, self.db, WORLD_SIZE)
        sx, sy = self.world.gen.spawn
        self.player = Player(sx, sy, self.db)
        self.enemies: list = []
        self.projectiles = ProjectilePool()
        self.effects = EffectsSystem()
        self.effects.damage_numbers = settings.get("damage_numbers", True)
        self.clock = WorldClock()
        self.weather = WeatherSystem(self.seed)
        self.noise = NoiseSystem(self)
        self.world_system = WorldSystem(self)
        self.survival = SurvivalSystem(self)
        self.crafting = CraftingSystem(self, data.recipes)
        self.digging = DiggingSystem(self)
        self.combat = CombatSystem(self)
        self.spawner = SpawnSystem(self, data.enemies)
        self.controller = PlayerController(self)
        self.actor_hash = SpatialHash(2.0)
        self.messages: list[dict] = []
        self.hitstop_t = 0.0
        self.pending_trauma = 0.0
        self.ui_request = None       # ("inventory", container_obj) / ("crafting", None)
        self.lore_popup = None
        self.sleep_fade = 0.0
        self.autosave_t = AUTOSAVE_INTERVAL
        self.autosave_requested = False
        self.play_time = 0.0
        self.new_game_grace = new_game
        self.last_damage_angle = None
        self.damage_flash = 0.0
        self.containers: dict[str, GridInventory] = {}
        self.world.ensure_area(sx, sy, 40)
        if new_game:
            self._starting_kit()
            self.spawner.populate_initial()

    def _starting_kit(self):
        # Nothing but old clothes: everything else must be found or made.
        self.player.body = self.db.new_stack("old_clothes")

    # ------------------------------------------------------------ helpers used by systems
    def message(self, text: str, color=(235, 230, 215), big: bool = False):
        self.messages.append({"text": text, "color": color, "t": 4.0 if big else 3.0, "big": big})
        if len(self.messages) > 6:
            self.messages = self.messages[-6:]

    def hitstop(self, t: float):
        scale = float(self.settings.get("hitstop", 1.0))
        self.hitstop_t = max(self.hitstop_t, min(0.2, t * scale))

    def camera_trauma(self, amount: float):
        self.pending_trauma += amount

    def input_rumble(self, strength: float, duration: float = 0.12):
        if self.input is not None:
            self.input.rumble(strength, duration)

    def is_night(self) -> bool:
        return self.clock.is_night()

    def visibility_mult(self) -> float:
        v = self.weather.visibility()
        if self.is_night():
            v *= 0.6
        return v

    def carry_capacity(self) -> float:
        return PLAYER_CARRY_CAPACITY

    def give_item(self, item_id: str, count: int, announce: bool = True):
        p = self.player
        left = p.inventory.add_item(item_id, count)
        got = count - left
        if got > 0:
            st = p.inventory.find_first(item_id)
            if st is not None:
                p.auto_hotbar(st)
            if announce:
                self.message(f"+{got} {self.db.name(item_id)}", (220, 240, 200))
        if left > 0:
            self.drop_item(self.db.new_stack(item_id, left))
            self.message("가방이 가득 찼다 — 바닥에 떨어뜨렸다", (255, 170, 120))

    def drop_item(self, stack, near_player: bool = True):
        p = self.player
        a = self.rng.uniform(0, math.tau)
        obj = self.world.spawn_pickup(stack, p.x + math.cos(a) * 0.5, p.y + math.sin(a) * 0.5)
        obj.no_auto = 3.0
        return obj

    def collect_pickup(self, obj) -> bool:
        p = self.player
        st = obj.stack
        if st is None:
            self.world.remove_object(obj)
            return True
        if not p.inventory.insert_stack(st):
            if obj.no_auto <= 0:
                self.message("가방에 공간이 없다", (255, 170, 120))
                obj.no_auto = 2.0
            return False
        self.world.remove_object(obj)
        self.audio.play("pickup", 0.6)
        self.message(f"+{st.count} {self.db.name(st.item_id)}" if st.count > 0 else self.db.name(st.item_id), (220, 240, 200))
        merged = p.inventory.stacks.get(st.uid) or p.inventory.find_first(st.item_id)
        if merged is not None:
            p.auto_hotbar(merged)
        return True

    def wear_equipped(self, amount: float):
        p = self.player
        st = p.equipped_stack()
        if st is None or st.durability is None:
            return
        st.durability -= amount
        if st.durability <= 0:
            name = self.db.name(st.item_id)
            p.inventory.remove_uid(st.uid)
            p.equipped_uid = None
            p.clean_hotbar()
            self.message(f"{name}이(가) 망가졌다!", (255, 120, 90), big=True)
            self.audio.play("hit_metal", 0.6)

    def damage_player(self, amount: float, source: str = "", angle: float | None = None, feedback: bool = True):
        p = self.player
        if p.dead:
            return
        p.hp -= amount
        if feedback:
            p.flash = 1.0
            self.damage_flash = 1.0
            self.last_damage_angle = angle
            self.camera_trauma(0.35)
            self.input_rumble(0.7, 0.2)
            self.audio.play("hurt", 0.8)
            self.effects.blood(p.x, p.y, angle if angle is not None else 0.0, 0.8)
            self.effects.text(p.x, p.y, f"-{int(round(amount))}", (255, 90, 80), 0.9, 1.1)
            self.hitstop(0.05)
            p.digging = False
            if p.action is not None:
                self.cancel_action()
        if p.hp <= 0:
            p.hp = 0
            p.dead = True
            p.digging = False
            p.action = None
            self.audio.play("death", 0.9)
            self.message("당신은 쓰러졌다...", (255, 80, 80), big=True)

    # ------------------------------------------------------------ physics
    def move_actor(self, actor, dx: float, dy: float) -> bool:
        ox, oy = actor.x, actor.y
        nx, ny = ox + dx, oy + dy
        r = actor.radius
        for o in self.world.blocking_near(nx, ny, r + 0.6):
            res = resolve_circle(nx, ny, r, o.x, o.y, o.radius)
            if res is not None:
                nx, ny = res
        size = self.world.size
        nx = max(0.5, min(size - 0.5, nx))
        ny = max(0.5, min(size - 0.5, ny))
        actor.x, actor.y = nx, ny
        want = math.hypot(dx, dy)
        got = math.hypot(nx - ox, ny - oy)
        return want < 1e-6 or got > want * 0.35

    def _separate_actors(self):
        h = self.actor_hash
        h.clear()
        live = [e for e in self.enemies if not e.dead]
        for e in live:
            h.insert(e)
        p = self.player
        for e in live:
            for o in h.query(e.x, e.y, 1.0):
                if o is e or o.id < e.id:
                    continue
                res = resolve_circle(e.x, e.y, e.radius, o.x, o.y, o.radius)
                if res is not None:
                    mx, my = (res[0] - e.x) * 0.5, (res[1] - e.y) * 0.5
                    e.x += mx
                    e.y += my
                    o.x -= mx
                    o.y -= my
            if not p.dead and p.dodge_t <= 0:
                res = resolve_circle(p.x, p.y, p.radius, e.x, e.y, e.radius)
                if res is not None:
                    mx, my = (res[0] - p.x) * 0.5, (res[1] - p.y) * 0.5
                    p.x += mx
                    p.y += my
                    e.x -= mx
                    e.y -= my

    # ------------------------------------------------------------ equipment & items
    def equip(self, uid: str | None):
        p = self.player
        if uid is None or p.equipped_uid == uid:
            p.equipped_uid = None
            p.reload_t = 0.0
            return
        st = p.inventory.stacks.get(uid)
        if st is None:
            return
        idef = self.db.item(st.item_id)
        if idef.get("equip") == "body":
            old = p.body
            p.inventory.remove_uid(uid)
            p.body = st
            if old is not None and not p.inventory.insert_stack(old):
                self.drop_item(old)
            self.message(f"{idef['name']} 착용")
            return
        if idef.get("equip") != "hand":
            return
        p.equipped_uid = uid
        p.reload_t = 0.0
        p.attack_anim = None
        self.audio.play("reload", 0.3)

    def unequip_body(self):
        p = self.player
        if p.body is None:
            return
        if p.inventory.insert_stack(p.body):
            p.body = None

    def use_stack(self, uid: str):
        p = self.player
        st = p.inventory.stacks.get(uid)
        if st is None:
            return
        idef = self.db.item(st.item_id)
        if idef.get("place"):
            self.start_placement(uid)
        elif idef.get("equip"):
            self.equip(uid)
        elif idef.get("use"):
            ok, msg = self.survival.apply_use(st)
            if msg:
                self.message(msg, (210, 235, 200))
            p.clean_hotbar()

    def activate_slot(self, i: int):
        p = self.player
        uid = p.hotbar[i] if i < len(p.hotbar) else None
        if uid is None or uid not in p.inventory.stacks:
            p.hotbar[i] = None if i < len(p.hotbar) else None
            # Empty slot 1 doubles as "bare hands".
            if p.equipped_uid is not None and i == 0:
                self.equip(None)
            return
        self.use_stack(uid)

    def cycle_weapon(self):
        p = self.player
        cands = [None] + [u for u in p.hotbar if u and u in p.inventory.stacks
                          and self.db.item(p.inventory.stacks[u].item_id).get("equip") == "hand"]
        if len(cands) <= 1:
            return
        try:
            i = cands.index(p.equipped_uid)
        except ValueError:
            i = 0
        nxt = cands[(i + 1) % len(cands)]
        if nxt is None:
            self.equip(None)
        else:
            self.equip(nxt)
        self.message("맨손" if nxt is None else self.db.name(p.inventory.stacks[nxt].item_id))

    def drop_stack(self, uid: str):
        p = self.player
        st = p.inventory.remove_uid(uid)
        if st is None:
            return
        if p.equipped_uid == uid:
            p.equipped_uid = None
        p.clean_hotbar()
        self.drop_item(st)

    # ------------------------------------------------------------ structures
    def start_placement(self, uid: str):
        p = self.player
        st = p.inventory.stacks.get(uid)
        if st is None:
            return
        p.placing = {"uid": uid, "struct": self.db.item(st.item_id)["place"], "cell": None, "ok": False}
        self.message("설치 위치를 고르세요 (공격: 설치 / 조준: 취소)", (240, 230, 180))

    def place_structure(self):
        p = self.player
        pl = p.placing
        if not pl or not pl.get("cell"):
            return
        st = p.inventory.stacks.get(pl["uid"])
        if st is None:
            p.placing = None
            return
        cx, cy = pl["cell"]
        obj = self.world.place_structure(pl["struct"], cx + 0.5, cy + 0.5)
        p.inventory.consume_one(st.uid)
        p.clean_hotbar()
        p.placing = None
        self.effects.dust(obj.x, obj.y, 8)
        self.audio.play("craft", 0.7)
        self.message(f"{obj.name} 설치", (200, 240, 170))
        if obj.otype == "bedroll":
            p.spawn_point = (obj.x, obj.y)

    def container_inventory(self, obj) -> GridInventory:
        inv = self.containers.get(obj.id)
        if inv is None:
            w, h = obj.defn.get("grid", [6, 4])
            inv = GridInventory(self.db, w, h, obj.id)
            if obj.data.get("items"):
                inv.load_list(obj.data["items"])
            self.containers[obj.id] = inv
        return inv

    def sync_containers(self):
        for oid, inv in self.containers.items():
            obj = self.world.objects_by_id.get(oid)
            if obj is not None:
                obj.data["items"] = inv.to_list()

    # ------------------------------------------------------------ interaction
    def find_interactable(self):
        p = self.player
        best, best_score = None, 1e9
        ax, ay = p.aim_point
        look = math.atan2(ay - p.y, ax - p.x)
        for o in self.world.objects_near(p.x, p.y, INTERACT_RANGE + 1.0):
            if o.interact is None:
                continue
            d = math.hypot(o.x - p.x, o.y - p.y) - o.radius
            if d > INTERACT_RANGE:
                continue
            facing = abs(angle_diff(math.atan2(o.y - p.y, o.x - p.x), look))
            score = d + facing * 0.35 - (0.6 if o.kind == "pickup" else 0)
            if math.hypot(o.x - ax, o.y - ay) < o.radius + 0.5:
                score -= 1.5  # the thing under the cursor wins
            if score < best_score:
                best, best_score = o, score
        return best

    def interaction_label(self, obj) -> str:
        it = obj.interact
        p = self.player
        if it == "pickup":
            return f"줍기: {obj.name}" + (f" x{obj.stack.count}" if obj.stack and obj.stack.count > 1 else "")
        if it == "gather":
            return f"채집: {obj.name}"
        if it == "chop":
            tool = p.tool()
            return f"{'벌목' if tool and tool.get('chop') else '가지 꺾기'}: {obj.name}"
        if it == "search":
            return f"수색: {obj.name}"
        if it == "water":
            return "물 담기" if p.inventory.count("empty_bottle") else "물 마시기 (오염)"
        if it == "station":
            if obj.otype == "campfire" and obj.data.get("fuel", 0) < 300 and p.inventory.count("branch"):
                return "장작 넣기"
            return f"{obj.name} 사용"
        if it == "container":
            return f"열기: {obj.name}"
        if it == "sleep":
            return "잠자기"
        return obj.name

    def begin_interaction(self, obj):
        p = self.player
        it = obj.interact
        if it == "pickup":
            self.collect_pickup(obj)
            return
        if it in ("gather", "chop", "search", "water"):
            dur = obj.defn.get("time", 1.0)
            if it == "chop":
                tool = p.tool()
                chop = tool.get("chop", 0) if tool else 0
                dur = obj.defn["time"] / chop if chop else obj.defn.get("hand_time", 2.0)
            p.action = {"obj": obj, "kind": it, "t": 0.0, "dur": dur, "fx": 0.0}
            p.facing = math.atan2(obj.y - p.y, obj.x - p.x)
            if it == "search":
                self.noise.emit(p.x, p.y, "search", obj.defn.get("noise", 4))
            return
        if it == "station":
            if obj.otype == "campfire" and obj.data.get("fuel", 0) < 300 and p.inventory.count("branch"):
                p.inventory.remove_item("branch", 1)
                obj.data["fuel"] = min(900.0, obj.data.get("fuel", 0) + obj.defn.get("fuel_per_item", 180))
                self.audio.play("fire_crackle", 0.7)
                self.message("모닥불에 장작을 넣었다")
                return
            self.ui_request = ("crafting", obj)
            return
        if it == "container":
            self.ui_request = ("inventory", obj)
            return
        if it == "sleep":
            self.try_sleep(obj)

    def cancel_action(self):
        self.player.action = None

    def progress_action(self, dt: float):
        p = self.player
        a = p.action
        obj = a["obj"]
        if obj.removed or math.hypot(obj.x - p.x, obj.y - p.y) > INTERACT_RANGE + obj.radius + 0.4:
            p.action = None
            return
        a["t"] += dt
        a["fx"] -= dt
        if a["fx"] <= 0:
            a["fx"] = 0.4
            if a["kind"] == "chop":
                self.audio.play("hit_sharp" if p.tool() else "hit_blunt", 0.6)
                self.effects.burst(obj.x, obj.y, (150, 110, 70), 4, 2.0, 30, (30, 90), 300, (2, 3.5), (0.3, 0.6), "chunk")
                self.noise.emit(obj.x, obj.y, "chop", obj.defn.get("noise", 6))
                self.camera_trauma(0.05)
            elif a["kind"] in ("gather", "search"):
                self.effects.dust(obj.x, obj.y, 1)
        if a["t"] >= a["dur"]:
            p.action = None
            self.complete_interaction(obj, a["kind"])

    def complete_interaction(self, obj, kind: str):
        p = self.player
        found: list[tuple[str, int]] = []
        if kind == "water":
            if p.inventory.count("empty_bottle"):
                p.inventory.remove_item("empty_bottle", 1)
                self.give_item("dirty_water", 1)
            else:
                p.stats.thirst = min(100.0, p.stats.thirst + 12)
                if self.rng.random() < 0.5:
                    p.stats.infection = min(100.0, p.stats.infection + 6)
                self.message("흙탕물을 마셨다. 찝찝하다...", (200, 200, 160))
            self.audio.play("pickup", 0.4)
            return
        if kind == "gather":
            from systems.resource_system import roll_chances
            found = roll_chances(obj.defn.get("yield", []), self.rng)
        elif kind == "chop":
            from systems.resource_system import roll_chances
            tool = p.tool()
            if tool and tool.get("chop"):
                found = roll_chances(obj.defn.get("yield", []), self.rng)
                self.wear_equipped(1.0)
            else:
                found = roll_chances(obj.defn.get("hand_yield", []), self.rng)
        elif kind == "search":
            if obj.defn.get("special"):
                found = [(i, a) for i, a, _ in self.resources["special_loot"][obj.defn["special"]]]
            else:
                found = roll_table(self.resources["loot_tables"][obj.defn["loot"]], self.rng, obj.defn.get("rolls", 1))
            if obj.defn.get("contaminate"):
                p.stats.infection = min(100.0, p.stats.infection + obj.defn["contaminate"] * self.rng.random())
            if not found:
                self.message(f"{obj.name}: 쓸 만한 게 없다", (190, 180, 160))
        for item_id, n in found:
            self.give_item(item_id, n)
        if found:
            self.audio.play("pickup", 0.6)
        if obj.uses > 0:
            obj.uses -= 1
            if obj.uses == 0 and not obj.defn.get("keep_when_empty"):
                self.world.remove_object(obj)
                self.effects.dust(obj.x, obj.y, 5)

    def try_sleep(self, obj):
        p = self.player
        for e in self.enemies:
            if not e.dead and e.defn.get("kind") == "hostile" and e.dist_to(p) < 14:
                self.message("주변이 위험해서 잠들 수 없다", (255, 150, 120))
                return
        if p.stats.fatigue < 25:
            self.message("아직 졸리지 않다")
            return
        hours = 7.0 if self.is_night() else 3.0
        self.clock.advance_hours(hours)
        secs = hours / 24.0 * 960.0
        st = p.stats
        st.hunger = max(0.0, st.hunger - 0.4 * 0.065 * secs)
        st.thirst = max(0.0, st.thirst - 0.4 * 0.095 * secs)
        st.fatigue = 0.0
        st.stamina = st.max_stamina = 100.0
        p.hp = min(p.max_hp, p.hp + hours * 3)
        p.spawn_point = (obj.x, obj.y)
        self.enemies = [e for e in self.enemies if e.dist_to(p) > 20]
        self.sleep_fade = 1.5
        self.autosave_requested = True
        self.message(f"{int(hours)}시간 잠을 잤다. ({self.clock.clock_text()})", (200, 210, 255), big=True)

    def show_lore(self, text: str):
        self.lore_popup = text

    # ------------------------------------------------------------ tick
    def update(self, dt: float, inp=None, camera=None, ui_blocking: bool = False):
        self.play_time += dt
        for m in self.messages:
            m["t"] -= dt
        self.messages = [m for m in self.messages if m["t"] > 0]
        self.damage_flash = max(0.0, self.damage_flash - dt * 2.5)
        self.sleep_fade = max(0.0, self.sleep_fade - dt)
        if self.hitstop_t > 0:
            self.hitstop_t -= dt
            self.effects.update(dt * 0.15)
            return
        p = self.player
        self.audio.listener = (p.x, p.y)
        self.world.ensure_area(p.x, p.y, 36, budget=1)
        self.weather.update(dt)
        self.world_system.update(dt)
        self.controller.update(dt, inp, camera, ui_blocking)
        for e in self.enemies:
            e.update(self, dt)
        self._separate_actors()
        self.combat.update_projectiles(dt)
        self.combat.update_enemy_bleed(dt)
        self.crafting.update(dt)
        self.survival.update(dt, self.controller.activity)
        self.noise.update(dt)
        self.spawner.update(dt)
        self._auto_pickup()
        self.effects.update(dt)
        if not p.dead:
            self.autosave_t -= dt
            if self.autosave_t <= 0:
                self.autosave_t = AUTOSAVE_INTERVAL
                self.autosave_requested = True

    def _auto_pickup(self):
        p = self.player
        if p.dead:
            return
        for o in self.world.objects_near(p.x, p.y, 0.7):
            if o.kind == "pickup" and o.no_auto <= 0 and o.z <= 0:
                if not self.collect_pickup(o):
                    o.no_auto = 3.0
