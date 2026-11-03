"""Recipe-driven crafting, station detection and equipment repair."""

STATION_RADIUS = 3.0


class CraftingSystem:
    def __init__(self, session, recipes: list):
        self.session = session
        self.recipes = recipes
        self.by_id = {r["id"]: r for r in recipes}
        self.active = None  # {"recipe", "t", "dur"}

    def stations_near(self, x: float, y: float) -> set[str]:
        out = set()
        for o in self.session.world.objects_near(x, y, STATION_RADIUS):
            if o.kind == "structure":
                st = o.defn.get("station")
                if st == "campfire" and o.data.get("fuel", 0) <= 0:
                    continue
                if st:
                    out.add(st)
        return out

    def missing(self, recipe: dict, inv) -> list[tuple[str, int, int]]:
        out = []
        for item_id, need in recipe["ingredients"]:
            have = inv.count(item_id)
            if have < need:
                out.append((item_id, have, need))
        return out

    def can_craft(self, recipe: dict, inv, stations: set) -> tuple[bool, str]:
        st = recipe.get("station")
        if st and st not in stations:
            name = self.session.resources["structures"].get(st, {}).get("name", st)
            return False, f"{name} 근처에서만 제작 가능"
        miss = self.missing(recipe, inv)
        if miss:
            return False, "재료 부족"
        return True, ""

    def start(self, recipe_id: str) -> tuple[bool, str]:
        s = self.session
        p = s.player
        recipe = self.by_id[recipe_id]
        ok, why = self.can_craft(recipe, p.inventory, self.stations_near(p.x, p.y))
        if not ok:
            return False, why
        if self.active:
            return False, "이미 제작 중이다"
        self.active = {"recipe": recipe, "t": 0.0, "dur": recipe.get("time", 1.0)}
        s.audio.play("craft", 0.6)
        return True, ""

    def cancel(self):
        self.active = None

    def update(self, dt: float):
        if not self.active:
            return
        self.active["t"] += dt
        if self.active["t"] >= self.active["dur"]:
            recipe = self.active["recipe"]
            self.active = None
            self.finish(recipe)

    def finish(self, recipe: dict) -> bool:
        s = self.session
        p = s.player
        ok, why = self.can_craft(recipe, p.inventory, self.stations_near(p.x, p.y))
        if not ok:
            s.message(why, (255, 120, 100))
            return False
        for item_id, need in recipe["ingredients"]:
            p.inventory.remove_item(item_id, need)
        item_id, count = recipe["result"]
        s.give_item(item_id, count, announce=False)
        s.message(f"제작 완료: {s.db.name(item_id)} x{count}", (180, 240, 160))
        s.audio.play("ui_confirm", 0.6)
        p.clean_hotbar()
        return True

    # ------------------------------------------------------------ repair
    def repairables(self):
        p = self.session.player
        out = []
        for st, _ in p.inventory.items():
            idef = self.session.db.item(st.item_id)
            if idef.get("repair") and st.durability is not None and st.durability < idef.get("durability", 0):
                out.append(st)
        if p.body is not None:
            idef = self.session.db.item(p.body.item_id)
            if idef.get("repair") and p.body.durability is not None and p.body.durability < idef.get("durability", 0):
                out.append(p.body)
        return out

    def can_repair(self, stack) -> tuple[bool, str]:
        s = self.session
        p = s.player
        rep = s.db.item(stack.item_id)["repair"]
        if rep.get("station") and rep["station"] not in self.stations_near(p.x, p.y):
            return False, "작업대가 필요하다"
        if p.inventory.count(rep["item"]) < rep["count"]:
            return False, f"{s.db.name(rep['item'])} {rep['count']}개 필요"
        return True, ""

    def repair(self, stack) -> tuple[bool, str]:
        ok, why = self.can_repair(stack)
        if not ok:
            return False, why
        s = self.session
        idef = s.db.item(stack.item_id)
        rep = idef["repair"]
        s.player.inventory.remove_item(rep["item"], rep["count"])
        stack.durability = min(float(idef["durability"]), stack.durability + rep["amount"])
        s.audio.play("craft", 0.7)
        return True, f"{idef['name']} 수리 완료"
