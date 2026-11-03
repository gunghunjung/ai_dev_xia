"""Melee, firearms, projectiles, damage resolution and hit feedback."""
import math

from engine.collision import angle_diff, segment_circle_hit
from entities.enemy import ATTACK, HIT, DEAD, STUNNED
from interfaces.combat_interface import ICombatService, DamageResult

MELEE_IMPACT_T = 0.38


class LocalCombatService(ICombatService):
    """Authoritative damage math. A network build would run this on the server."""

    def __init__(self, rng):
        self.rng = rng

    def resolve_hit(self, attacker, target, weapon: dict, multiplier: float = 1.0, defense: float = 0.0) -> DamageResult:
        dmg = weapon["damage"] * multiplier * self.rng.uniform(0.9, 1.1)
        dmg = max(1.0, dmg - defense)
        killed = target.hp - dmg <= 0
        bleed = self.rng.random() < weapon.get("bleed", 0.0)
        return DamageResult(dmg, killed, weapon.get("knockback", 3.0) * multiplier, weapon.get("stun", 0.2), bleed)


class CombatSystem:
    def __init__(self, session):
        self.session = session
        self.service = LocalCombatService(session.rng)

    # ------------------------------------------------------------ player melee
    def melee_weapon_for(self, player) -> tuple[dict, str | None]:
        w = player.weapon()
        if w["type"] == "ranged":
            return self.session.db.weapon(w.get("melee", "fists")), None
        return w, player.equipped_item_id()

    def start_melee(self, player) -> bool:
        s = self.session
        if player.attack_cd > 0 or player.attack_anim is not None or player.reload_t > 0:
            return False
        weapon, visual = self.melee_weapon_for(player)
        tired = player.stats.stamina < weapon["stamina"]
        player.stats.stamina = max(0.0, player.stats.stamina - weapon["stamina"])
        dur = max(0.22, weapon["cooldown"] * 0.8) * (1.25 if tired else 1.0)
        player.attack_anim = {"t": 0.0, "dur": dur, "style": weapon.get("swing", "swing"), "arc": weapon["arc"],
                              "weapon": weapon, "visual": visual, "hit": False, "mult": 0.6 if tired else 1.0,
                              "angle": player.aim_angle}
        player.attack_cd = weapon["cooldown"] * (1.3 if tired else 1.0)
        s.audio.play(weapon.get("swing_sound", "swing_light"), 0.7)
        s.noise.emit(player.x, player.y, "melee", weapon.get("noise", 5))
        return True

    def update_player_attack(self, player, dt: float):
        anim = player.attack_anim
        if anim is None:
            return
        anim["t"] += dt / anim["dur"]
        if not anim["hit"] and anim["t"] >= MELEE_IMPACT_T:
            anim["hit"] = True
            self._melee_impact(player, anim)
        if anim["t"] >= 1.0:
            player.attack_anim = None

    def _melee_impact(self, player, anim):
        s = self.session
        weapon = anim["weapon"]
        ang = anim["angle"]
        reach = weapon["range"]
        half_arc = math.radians(weapon["arc"]) / 2
        s.effects.slash(player.x, player.y, ang, math.radians(weapon["arc"]), reach,
                        (255, 250, 230) if weapon.get("sound") != "hit_sharp" else (255, 235, 200),
                        kind="stab" if anim["style"] in ("stab", "jab") else "slash")
        hits = 0
        strongest = 0.0
        for e in s.enemies:
            if e.dead:
                continue
            d = player.dist_to(e)
            if d > reach + e.radius:
                continue
            a = player.angle_to(e)
            if d > 0.45 and abs(angle_diff(a, ang)) > half_arc + 0.15:
                continue
            res = self.service.resolve_hit(player, e, weapon, anim["mult"])
            self.apply_damage_to_enemy(e, res, a, weapon, melee=True)
            hits += 1
            strongest = max(strongest, res.amount)
        if hits:
            s.hitstop(weapon.get("hitstop", 0.05) * (1.0 + 0.15 * (hits - 1)))
            s.camera_trauma(weapon.get("shake", 0.2))
            s.input_rumble(weapon.get("rumble", 0.4), 0.12)
            s.audio.play(weapon.get("sound", "hit_blunt"), 0.9)
            if anim["visual"] is not None:
                s.wear_equipped(1.0)
                if weapon.get("break_chance") and s.rng.random() < weapon["break_chance"]:
                    st = player.equipped_stack()
                    if st:
                        player.inventory.consume_one(st.uid)
                        s.message(f"{s.db.name(st.item_id)}이(가) 부러졌다", (230, 160, 120))

    # ------------------------------------------------------------ firearms
    def try_fire(self, player) -> bool:
        s = self.session
        weapon = player.weapon()
        st = player.equipped_stack()
        if weapon["type"] != "ranged" or st is None:
            return False
        if player.attack_cd > 0 or player.reload_t > 0:
            return False
        if st.ammo <= 0:
            s.audio.play("dry_fire", 0.8)
            player.attack_cd = 0.3
            if not self.start_reload(player):
                s.message("탄약이 없다", (255, 150, 120))
            return False
        st.ammo -= 1
        player.attack_cd = weapon["cooldown"]
        spread_deg = (weapon["aim_spread"] if player.aiming else weapon["spread"]) + player.spread_bloom
        if st.durability is not None and st.durability < s.db.item(st.item_id).get("durability", 100) * 0.25:
            spread_deg *= 1.6  # worn guns are inaccurate
        mx = player.x + math.cos(player.aim_angle) * 0.55
        my = player.y + math.sin(player.aim_angle) * 0.55
        for _ in range(weapon.get("pellets", 1)):
            a = player.aim_angle + math.radians(s.rng.uniform(-spread_deg, spread_deg) * 0.5)
            sp = weapon["bullet_speed"]
            s.projectiles.spawn(mx, my, math.cos(a) * sp, math.sin(a) * sp, weapon["damage"], weapon["range"], weapon, "player")
        player.spread_bloom = min(12.0, player.spread_bloom + weapon["recoil"] * 10)
        player.recoil = weapon["recoil"]
        s.effects.flash(mx, my, 4.5, tuple(weapon.get("muzzle", (255, 220, 140))), 0.06, player.aim_angle)
        s.effects.burst(mx, my, [(255, 230, 150), (255, 180, 80)], 6, 5.0, 20, (10, 60), 100, (1.5, 3), (0.06, 0.14), "spark", player.aim_angle, 0.35)
        side = player.aim_angle + math.pi / 2
        s.effects.burst(player.x, player.y, (220, 180, 80), 1, 2.0, 22, (60, 90), 300, (1.5, 2), (0.6, 0.9), "chunk", side, 0.3)
        s.camera_trauma(weapon.get("shake", 0.25))
        s.input_rumble(weapon.get("rumble", 0.5), 0.1)
        s.audio.play(weapon.get("sound", "pistol"), 1.0)
        s.noise.emit(player.x, player.y, "gunshot", weapon.get("noise", 26))
        s.wear_equipped(0.5)
        return True

    def start_reload(self, player) -> bool:
        s = self.session
        weapon = player.weapon()
        st = player.equipped_stack()
        if weapon["type"] != "ranged" or st is None or player.reload_t > 0:
            return False
        if st.ammo >= weapon["mag_size"]:
            return False
        if player.inventory.count(weapon["ammo"]) <= 0:
            s.message(f"{s.db.name(weapon['ammo'])}이(가) 없다", (255, 150, 120))
            return False
        player.reload_t = player.reload_total = weapon["reload_time"]
        s.audio.play("reload", 0.7)
        return True

    def update_reload(self, player, dt: float):
        if player.reload_t <= 0:
            return
        st = player.equipped_stack()
        weapon = player.weapon()
        if st is None or weapon["type"] != "ranged":
            player.reload_t = 0.0
            return
        player.reload_t -= dt
        if player.reload_t <= 0:
            player.reload_t = 0.0
            need = weapon["mag_size"] - st.ammo
            have = player.inventory.count(weapon["ammo"])
            n = min(need, have)
            if n > 0:
                player.inventory.remove_item(weapon["ammo"], n)
                st.ammo += n
                self.session.audio.play("reload", 0.5)

    # ------------------------------------------------------------ projectiles
    def update_projectiles(self, dt: float):
        s = self.session
        for pr in s.projectiles.active():
            pr.px, pr.py = pr.x, pr.y
            nx, ny = pr.x + pr.vx * dt, pr.y + pr.vy * dt
            step = math.hypot(nx - pr.x, ny - pr.y)
            best_t, best = None, None
            for e in s.enemies:
                if e.dead or abs(e.x - pr.x) > step + 1 or abs(e.y - pr.y) > step + 1:
                    continue
                t = segment_circle_hit(pr.x, pr.y, nx, ny, e.x, e.y, e.radius + 0.06)
                if t is not None and (best_t is None or t < best_t):
                    best_t, best = t, e
            for o in s.world.objects_near((pr.x + nx) / 2, (pr.y + ny) / 2, step / 2 + 1.2):
                if not o.blocking or o.radius < 0.3:
                    continue
                t = segment_circle_hit(pr.x, pr.y, nx, ny, o.x, o.y, o.radius)
                if t is not None and (best_t is None or t < best_t):
                    best_t, best = t, o
            if best is not None:
                hx, hy = pr.x + (nx - pr.x) * best_t, pr.y + (ny - pr.y) * best_t
                s.effects.tracer(pr.x, pr.y, hx, hy)
                pr.alive = False
                ang = math.atan2(pr.vy, pr.vx)
                if hasattr(best, "etype"):
                    w = pr.weapon
                    falloff = max(0.5, 1.0 - (w["range"] - pr.range_left) / w["range"] * 0.5)
                    res = self.service.resolve_hit(None, best, w, falloff)
                    self.apply_damage_to_enemy(best, res, ang, w, melee=False)
                    s.hitstop(w.get("hitstop", 0.04))
                else:
                    s.effects.sparks(hx, hy, ang + math.pi, 8)
                    s.audio.play("bullet_hit", 0.5, (hx, hy))
                continue
            s.effects.tracer(pr.x, pr.y, nx, ny)
            pr.x, pr.y = nx, ny
            pr.range_left -= step
            if pr.range_left <= 0 or not s.world.in_bounds(int(nx), int(ny)):
                pr.alive = False
                s.effects.dust(nx, ny, 3)

    # ------------------------------------------------------------ damage
    def apply_damage_to_enemy(self, e, res: DamageResult, angle: float, weapon: dict, melee: bool):
        s = self.session
        e.hp -= res.amount
        e.flash = 1.0
        e.apply_knockback(angle, res.knockback)
        big = res.amount >= 20
        color = (130, 150, 60) if e.etype == "infected" else (160, 30, 30)
        s.effects.blood(e.x, e.y, angle, 1.4 if big else 0.9, color)
        if not melee:
            s.effects.sparks(e.x, e.y, angle + math.pi, 4)
        s.effects.text(e.x, e.y, str(int(round(res.amount))), (255, 240, 120) if big else (255, 255, 255), 0.8, 1.3 if big else 1.0, damage=True)
        if res.bleed:
            e.data_bleed = 3.0
        if e.hp <= 0:
            self.kill(e, angle)
            return
        e.stun = max(e.stun, res.stun)
        if e.state not in (DEAD,):
            if e.state == ATTACK and res.stun < 0.3:
                pass  # light hits don't interrupt a committed attack
            else:
                e.attack_anim = None
                e.set_state(HIT, 0.25 + res.stun)
        if e.state in ("idle", "patrol", "investigate"):
            e.set_state(HIT, 0.25)

    def kill(self, e, angle: float):
        s = self.session
        e.dead = True
        e.hp = 0
        e.state = DEAD
        e.facing = angle
        s.audio.play("death", 0.6, (e.x, e.y))
        s.effects.blood(e.x, e.y, angle, 2.0, (130, 150, 60) if e.etype == "infected" else (150, 30, 30))
        s.camera_trauma(0.15)
        if e.defn.get("kind") == "hostile":
            s.player.stats.kills += 1
        if not e.dropped:
            e.dropped = True
            from systems.resource_system import roll_chances
            for item_id, count in roll_chances(e.defn.get("drops", []), s.rng):
                s.world.spawn_pickup(s.db.new_stack(item_id, count), e.x + s.rng.uniform(-0.2, 0.2), e.y + s.rng.uniform(-0.2, 0.2))

    def update_enemy_bleed(self, dt: float):
        s = self.session
        for e in s.enemies:
            b = getattr(e, "data_bleed", 0.0)
            if b > 0 and not e.dead:
                e.data_bleed = b - dt
                e.hp -= 2.5 * dt
                if s.rng.random() < dt * 6:
                    s.effects.burst(e.x, e.y, (140, 30, 30), 1, 0.5, 14, (0, 10), 200, (1.5, 2.5), (0.3, 0.5))
                if e.hp <= 0:
                    self.kill(e, e.facing)

    def enemy_strike(self, e):
        s = self.session
        p = s.player
        if p.dead:
            return
        d = e.dist_to(p)
        a = e.angle_to(p)
        reach = e.defn["attack_range"] + p.radius + 0.15
        s.audio.play(e.defn.get("sound"), 0.5, (e.x, e.y))
        if d > reach or abs(angle_diff(a, e.facing)) > math.radians(75):
            return
        if p.iframes > 0:
            s.effects.text(p.x, p.y, "회피!", (180, 230, 255), 0.6)
            return
        dmg = max(1.0, e.defn["damage"] * s.rng.uniform(0.85, 1.15) - p.defense())
        s.damage_player(dmg, source=e.etype, angle=a)
        p.apply_knockback(a, 3.5)
        st = p.stats
        if s.rng.random() < e.defn.get("bleed_chance", 0):
            st.bleeding = min(3.0, st.bleeding + 1.0)
            s.message("출혈이 시작됐다!", (230, 80, 80))
        if s.rng.random() < e.defn.get("infect_chance", 0):
            st.infection = min(100.0, st.infection + e.defn.get("infect_amount", 10))
        if dmg >= 10:
            st.injury = min(100.0, st.injury + dmg * 0.6)
