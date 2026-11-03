"""Turns logical input actions into player intents (move, aim, attack, dig, interact...)."""
import math

from engine.camera import screen_dir_to_world, iso
from engine.collision import angle_diff

BASE_SPEED = 3.3
SPRINT_MULT = 1.6
DODGE_SPEED = 9.0
DODGE_TIME = 0.26
DODGE_COST = 18.0
AIM_PLANE_Z = 22


class PlayerController:
    def __init__(self, session):
        self.session = session
        self.reason_cd = 0.0
        self.activity = "idle"
        self.aim_world = None

    # ------------------------------------------------------------ aim
    def _update_aim(self, inp, camera):
        s = self.session
        p = s.player
        if inp.last_device == "kbm" or not inp.has_pad:
            mx, my = camera.screen_to_world(*inp.mouse_pos)
            # Players put the cursor on a creature's body, not its feet: read the cursor on the
            # chest-height plane so shots go where the crosshair is.
            cx, cy = camera.screen_to_world(inp.mouse_pos[0], inp.mouse_pos[1] + AIM_PLANE_Z)
            p.aim_angle = math.atan2(cy - p.y, cx - p.x)
            self.aim_world = (mx, my)  # ground point: used for dig / build targeting
            self.use_cursor = True
        else:
            if inp.aim_stick_active:
                wx, wy = screen_dir_to_world(*inp.aim_vec)
                p.aim_angle = math.atan2(wy, wx)
            if s.settings.get("aim_assist_enabled", True):
                p.aim_angle = self._aim_assist(p.aim_angle)
            self.aim_world = (p.x + math.cos(p.aim_angle) * 4.5, p.y + math.sin(p.aim_angle) * 4.5)
            self.use_cursor = False
        p.aim_point = self.aim_world
        sx, sy = iso(math.cos(p.aim_angle), math.sin(p.aim_angle))
        p.aim_screen_angle = math.atan2(sy, sx)

    def _aim_assist(self, angle: float) -> float:
        s = self.session
        p = s.player
        strength = float(s.settings.get("aim_assist", 0.5))
        if strength <= 0:
            return angle
        best, best_diff = None, math.radians(20)
        for e in s.enemies:
            if e.dead or e.defn.get("kind") != "hostile":
                continue
            d = p.dist_to(e)
            if d > 10:
                continue
            diff = angle_diff(p.angle_to(e), angle)
            if abs(diff) < best_diff:
                best, best_diff = diff, abs(diff)
        if best is None:
            return angle
        return angle + best * min(1.0, strength) * 0.6

    # ------------------------------------------------------------ main
    def update(self, dt: float, inp, camera, ui_blocking: bool):
        s = self.session
        p = s.player
        self.activity = "idle"
        if p.dead:
            return
        self.reason_cd = max(0.0, self.reason_cd - dt)
        p.attack_cd = max(0.0, p.attack_cd - dt)
        p.dodge_cd = max(0.0, p.dodge_cd - dt)
        p.iframes = max(0.0, p.iframes - dt)
        p.hurt_cd = max(0.0, p.hurt_cd - dt)
        p.recoil = max(0.0, p.recoil - dt * 2.5)
        p.spread_bloom = max(0.0, p.spread_bloom - dt * 14)
        p.tick_common(dt)
        s.combat.update_player_attack(p, dt)
        s.combat.update_reload(p, dt)
        if inp is None:
            return
        self._update_aim(inp, camera)
        p.aiming = inp.held("aim") and not ui_blocking

        # ---------------- movement intent
        mvx, mvy = (0.0, 0.0) if ui_blocking else inp.move_vector()
        mag = min(1.0, math.hypot(mvx, mvy))
        wx, wy = screen_dir_to_world(mvx, mvy)
        moving = mag > 0.05
        if inp.pressed("sprint") and inp.last_device == "pad":
            p.sprint_toggle = not p.sprint_toggle
        if not moving:
            p.sprint_toggle = False
        want_sprint = inp.held("sprint") if inp.last_device == "kbm" else (p.sprint_toggle or inp.held("sprint"))
        p.sprinting = (moving and want_sprint and not p.aiming and s.survival.can_sprint(p)
                       and p.dodge_t <= 0 and not p.digging)

        if not ui_blocking:
            self._handle_actions(dt, inp, moving)

        # ---------------- velocity
        vx = vy = 0.0
        if p.dodge_t > 0:
            p.dodge_t -= dt
            vx, vy = p.dodge_dir[0] * DODGE_SPEED, p.dodge_dir[1] * DODGE_SPEED
            if s.rng.random() < 0.5:
                s.effects.dust(p.x, p.y, 1)
            self.activity = "sprint"
        elif moving and not p.digging and p.action is None:
            speed = BASE_SPEED * mag
            if p.sprinting:
                speed *= SPRINT_MULT
                p.stats.stamina = max(0.0, p.stats.stamina - s.survival.sprint_cost(p) * dt)
                self.activity = "sprint"
            else:
                self.activity = "walk"
            if p.aiming:
                speed *= 0.55
            if p.attack_anim is not None:
                speed *= 0.6
            speed *= s.survival.speed_mult(p)
            speed *= s.world.terrain_def_at(p.x, p.y).get("speed", 1.0)
            if s.world.depth_at(p.x, p.y) > 0:
                speed *= 0.72
            vx, vy = wx * speed, wy * speed
            p.walk_phase += dt * speed * 4.2
            p.step_t -= dt
            if p.step_t <= 0:
                p.step_t = 0.27 if p.sprinting else 0.4
                s.noise.emit(p.x, p.y, "sprint" if p.sprinting else "walk")
                if p.sprinting:
                    s.effects.dust(p.x, p.y, 2)
        if p.digging:
            self.activity = "dig"
        if p.attack_anim is not None:
            self.activity = "fight"
        p.moving = abs(vx) + abs(vy) > 0.01
        if p.attack_anim is None and not p.digging:
            p.facing = p.aim_angle
        tvx, tvy = vx + p.kx, vy + p.ky
        if tvx or tvy:
            s.move_actor(p, tvx * dt, tvy * dt)

    # ------------------------------------------------------------ actions
    def _reason(self, text: str):
        if self.reason_cd <= 0:
            self.session.message(text, (230, 190, 150))
            self.reason_cd = 1.2

    def _handle_actions(self, dt, inp, moving):
        s = self.session
        p = s.player
        weapon = p.weapon()
        gun = weapon["type"] == "ranged"

        # Timed interaction: moving cancels it.
        if p.action is not None:
            if moving or inp.pressed("dodge"):
                s.cancel_action()
            else:
                s.progress_action(dt)
                return

        if inp.pressed("dodge") and p.placing is None and p.dodge_cd <= 0 and p.stats.stamina >= DODGE_COST * 0.5:
            mvx, mvy = inp.move_vector()
            if abs(mvx) + abs(mvy) > 0.1:
                dx, dy = screen_dir_to_world(mvx, mvy)
            else:
                dx, dy = -math.cos(p.aim_angle), -math.sin(p.aim_angle)
            p.dodge_dir = (dx, dy)
            p.dodge_t = DODGE_TIME
            p.iframes = DODGE_TIME + 0.06
            p.dodge_cd = 0.55
            p.stats.stamina = max(0.0, p.stats.stamina - DODGE_COST)
            p.digging = False
            s.audio.play("dodge", 0.5)
            return

        if p.placing is not None:
            self._placement(inp)
            return

        if inp.pressed("switch_weapon"):
            s.cycle_weapon()
        for i in range(5):
            if inp.pressed(f"slot_{i + 1}"):
                s.activate_slot(i)
        if inp.pressed("aim_assist_toggle"):
            s.settings["aim_assist_enabled"] = not s.settings.get("aim_assist_enabled", True)
            s.message("조준 보정 " + ("켜짐" if s.settings["aim_assist_enabled"] else "꺼짐"))

        if inp.pressed("reload") or (inp.pressed("tool") and gun):
            s.combat.start_reload(p)

        if gun:
            if inp.pressed("attack"):
                s.combat.try_fire(p)
        elif inp.held("attack") and not p.digging:
            s.combat.start_melee(p)
        if inp.pressed("melee"):
            s.combat.start_melee(p)

        # Digging (hold).
        dig_held = inp.held("dig") or (inp.held("tool") and not gun)
        tool = p.tool()
        show_target = dig_held or (tool is not None and "dig_power" in tool)
        if show_target:
            cell = s.digging.pick_target(p, self.aim_world, self.use_cursor)
            s.digging.update_target(cell)
        else:
            s.digging.target = None
        if dig_held and p.attack_anim is None and p.reload_t <= 0:
            if s.digging.target_ok:
                if moving:
                    p.digging = False
                else:
                    p.digging = True
                    tx, ty = s.digging.target[0] + 0.5, s.digging.target[1] + 0.5
                    p.facing = math.atan2(ty - p.y, tx - p.x)
                    p.aim_angle = p.facing
                    if not s.digging.dig_tick(dt):
                        p.digging = False
            else:
                p.digging = False
                if inp.pressed("dig") or inp.pressed("tool"):
                    self._reason(s.digging.target_reason)
        else:
            p.digging = False

        if inp.pressed("interact"):
            obj = s.find_interactable()
            if obj is not None:
                s.begin_interaction(obj)

    def _placement(self, inp):
        s = self.session
        p = s.player
        cell = s.digging.pick_target(p, self.aim_world, self.use_cursor)
        p.placing["cell"] = cell
        p.placing["ok"] = s.world.free_for_structure(*cell) and math.hypot(cell[0] + 0.5 - p.x, cell[1] + 0.5 - p.y) > 0.6
        if inp.pressed("aim") or inp.pressed("dodge"):
            p.placing = None
            s.message("설치 취소")
            return
        if inp.pressed("attack") or inp.pressed("interact"):
            if p.placing["ok"]:
                s.place_structure()
            else:
                self._reason("여기에는 설치할 수 없다")
