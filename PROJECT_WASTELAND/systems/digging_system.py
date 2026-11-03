"""Terrain excavation: tool/soil/depth based work, stamina, noise, durability and finds."""
import math

from config import DIG_REACH

DIG_STAMINA_PER_SEC = 7.0
DIG_NOISE_INTERVAL = 0.7
DIG_SOUND_INTERVAL = 0.42


class DiggingSystem:
    def __init__(self, session):
        self.session = session
        self.target = None          # (x, y) cell currently targeted (for HUD highlight)
        self.target_ok = False
        self.target_reason = ""
        self.progress = 0.0         # 0..1 of the current layer
        self.layer_total = 1.0

    def tool_profile(self, player) -> dict:
        tool = player.tool()
        if tool and "dig_power" in tool:
            return {"dig_power": tool["dig_power"], "max_depth": tool["max_depth"], "soft_only": False, "hands": False}
        h = self.session.terrain_db.hand
        return {"dig_power": h["dig_power"], "max_depth": h["max_depth"], "soft_only": h["soft_only"], "hands": True}

    def pick_target(self, player, aim_world, use_cursor: bool):
        """Choose the cell to dig: the cursor cell if within reach, else one step along the aim."""
        if use_cursor and aim_world is not None:
            cx, cy = int(math.floor(aim_world[0])), int(math.floor(aim_world[1]))
            if math.hypot(cx + 0.5 - player.x, cy + 0.5 - player.y) <= DIG_REACH:
                return cx, cy
        a = player.aim_angle
        tx, ty = player.x + math.cos(a) * 0.95, player.y + math.sin(a) * 0.95
        return int(math.floor(tx)), int(math.floor(ty))

    def evaluate(self, cell) -> tuple[bool, str]:
        s = self.session
        info = s.world.cell(*cell)
        if info is None:
            return False, "월드 경계"
        ok, why = s.terrain_db.can_dig(info.terrain, info.depth, self.tool_profile(s.player))
        if not ok:
            return ok, why
        if s.world.cell_has_blocker(*cell):
            return False, "무언가가 가로막고 있다"
        for o in s.world.objects_near(cell[0] + 0.5, cell[1] + 0.5, 0.8):
            if o.kind == "structure" and o.cell == cell:
                return False, "구조물이 있다"
        return True, ""

    def update_target(self, cell):
        if cell != self.target:
            self.target = cell
            self.progress = 0.0
            info = self.session.world.cell(*cell) if cell else None
            if info:
                self.layer_total = self.session.terrain_db.layer_work(info.terrain, min(3, info.depth + 1))
                self.progress = min(0.99, info.work / self.layer_total) if self.layer_total > 0 else 0.0
            self.session.player.dig_hint_shown = False
        if cell is not None:
            self.target_ok, self.target_reason = self.evaluate(cell)

    def dig_tick(self, dt: float) -> bool:
        """Advance digging on the current target. Returns False if digging cannot continue."""
        s = self.session
        p = s.player
        cell = self.target
        if cell is None or not self.target_ok:
            return False
        info = s.world.cell(*cell)
        prof = self.tool_profile(p)
        power = prof["dig_power"]
        st = p.stats
        if st.stamina <= 1:
            power *= 0.4
        st.stamina = max(0.0, st.stamina - DIG_STAMINA_PER_SEC * dt)
        total = s.terrain_db.layer_work(info.terrain, info.depth + 1)
        self.layer_total = total
        work = info.work + power * dt
        cx, cy = cell[0] + 0.5, cell[1] + 0.5
        tdef = s.terrain_db.get(info.terrain)
        p.dig_sound_t -= dt
        if p.dig_sound_t <= 0:
            p.dig_sound_t = DIG_SOUND_INTERVAL / max(0.5, power)
            s.audio.play("dig", 0.55)
            s.effects.dirt(cx, cy, tuple(tdef["color"]), n=6, z=6)
            s.camera_trauma(0.04)
        p.dig_noise_t -= dt
        if p.dig_noise_t <= 0:
            p.dig_noise_t = DIG_NOISE_INTERVAL
            s.noise.emit(cx, cy, "dig", 7.0 + info.depth * 1.5)
        self.progress = min(1.0, work / total)
        if self.progress > 0.6 and not p.dig_hint_shown:
            p.dig_hint_shown = True
            if s.world.preview_deposit(*cell):
                s.message("삽 끝에 단단한 무언가가 걸린다...", (255, 220, 140))
                s.effects.sparks(cx, cy, n=4)
        if work >= total:
            self._complete_layer(cell, info, prof)
            return False
        s.world.set_work(cell[0], cell[1], work)
        return True

    def _complete_layer(self, cell, info, prof):
        s = self.session
        p = s.player
        found, cache = s.world.complete_dig_layer(*cell)
        depth = info.depth + 1
        cx, cy = cell[0] + 0.5, cell[1] + 0.5
        tdef = s.terrain_db.get(info.terrain)
        s.effects.dirt(cx, cy, tuple(tdef["color"]), n=16, z=8)
        s.camera_trauma(0.12)
        s.input_rumble(0.35, 0.12)
        if not prof["hands"]:
            s.wear_equipped(1.0)
        layer_name = s.terrain_db.layer_name(depth)
        if found:
            s.audio.play("dig_find", 0.8)
            s.effects.sparks(cx, cy, n=12)
            for item_id, count in found:
                stack = s.db.new_stack(item_id, count)
                s.world.spawn_pickup(stack, cx + s.rng.uniform(-0.3, 0.3), cy + s.rng.uniform(-0.3, 0.3))
            names = ", ".join(f"{s.db.name(i)} x{c}" for i, c in found)
            if cache:
                s.message("매장된 은닉물을 발견했다!", (255, 210, 90), big=True)
            s.message(f"{layer_name}: {names}", (240, 220, 160))
        else:
            s.message(f"{layer_name}까지 팠다. 아무것도 없다.", (190, 180, 160))
        self.target = None  # force re-evaluation (layer table changes)
        self.progress = 0.0
        p.dig_hint_shown = False
