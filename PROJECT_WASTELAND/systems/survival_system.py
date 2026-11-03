"""Hunger, thirst, stamina, temperature, fatigue, bleeding, injury, infection.

Rates are per real second. Each condition feeds back into gameplay (regen, speed, damage)
and surfaces as a HUD warning — the player should feel pressure, not micromanagement.
"""
import zlib

HUNGER_RATE = 0.065
THIRST_RATE = 0.095
FATIGUE_RATE = 0.045


class SurvivalSystem:
    def __init__(self, session):
        self.session = session
        self.warnings: list[tuple[str, tuple]] = []
        self.ambient_temp = 20.0
        self.feels_temp = 20.0
        self.near_fire = False

    # ------------------------------------------------------------ derived values
    def speed_mult(self, player) -> float:
        st = player.stats
        m = 1.0
        if st.injury > 70:
            m *= 0.72
        elif st.injury > 40:
            m *= 0.86
        if st.thirst < 15:
            m *= 0.9
        if st.fatigue > 85:
            m *= 0.9
        if st.body_temp < 35.0:
            m *= 0.88
        cap = self.session.carry_capacity()
        w = player.carry_weight()
        if w > cap:
            m *= max(0.45, 1.0 - (w - cap) / cap)
        return m

    def can_sprint(self, player) -> bool:
        st = player.stats
        return st.stamina > 5 and st.thirst > 3 and player.carry_weight() <= self.session.carry_capacity() * 1.5

    def sprint_cost(self, player) -> float:
        cost = 16.0
        if player.stats.thirst < 25:
            cost *= 1.5  # dehydration makes running inefficient
        return cost

    # ------------------------------------------------------------ tick
    def update(self, dt: float, activity: str):
        s = self.session
        p = s.player
        st = p.stats
        if p.dead:
            return
        act_h = {"sprint": 1.6, "dig": 1.5, "fight": 1.3}.get(activity, 1.0)
        act_t = {"sprint": 1.8, "dig": 1.6, "fight": 1.3}.get(activity, 1.0)
        if st.body_temp > 38.5:
            act_t *= 1.7
        st.hunger = max(0.0, st.hunger - HUNGER_RATE * act_h * dt)
        st.thirst = max(0.0, st.thirst - THIRST_RATE * act_t * dt)
        st.fatigue = min(100.0, st.fatigue + FATIGUE_RATE * (1.5 if activity in ("sprint", "dig") else 1.0) * dt)
        st.days_survived += dt / 960.0

        # Temperature ----------------------------------------------------
        region = s.world.region_at(p.x, p.y)
        reg_temp = s.terrain_db.regions.get(region, {}).get("temp", 0) * 3.0
        self.ambient_temp = s.clock.base_temperature() + s.weather.temp_offset() + reg_temp - s.weather.wetness * 4.0
        fire_bonus = 0.0
        self.near_fire = False
        for fire in s.world_system.campfires_near(p.x, p.y, 4.5):
            d = ((fire.x - p.x) ** 2 + (fire.y - p.y) ** 2) ** 0.5
            fire_bonus = max(fire_bonus, 18.0 * max(0.0, 1.0 - d / 4.5))
            self.near_fire = True
        self.feels_temp = self.ambient_temp + p.insulation() * 14.0 + fire_bonus + (4.0 if activity == "sprint" else 0.0)
        target = 37.0 + (self.feels_temp - 21.0) * 0.06
        if st.infection >= 50:
            target += 1.6  # fever
        target = max(33.0, min(40.8, target))
        st.body_temp += (target - st.body_temp) * min(1.0, 0.025 * dt)

        # Stamina --------------------------------------------------------
        st.max_stamina = max(30.0, 100.0 - max(0.0, st.fatigue - 50) * 0.7 - st.injury * 0.25)
        regen = 20.0
        if st.thirst < 25:
            regen *= 0.5
        if st.hunger < 25:
            regen *= 0.65
        if st.fatigue > 75:
            regen *= 0.65
        if st.body_temp < 35.5:
            regen *= 0.6
        if activity in ("idle", "walk"):
            st.stamina = min(st.max_stamina, st.stamina + regen * dt * (1.0 if activity == "idle" else 0.7))
        st.stamina = min(st.stamina, st.max_stamina)

        # Health ---------------------------------------------------------
        dmg = 0.0
        if st.hunger <= 0:
            dmg += 0.4
        if st.thirst <= 0:
            dmg += 0.6
        if st.body_temp < 34.5 or st.body_temp > 39.8:
            dmg += 0.2
        if st.bleeding > 0:
            dmg += st.bleeding * 0.55
            st.bleeding = max(0.0, st.bleeding - 0.012 * dt)
        if st.infection >= 85:
            dmg += 0.4
        elif st.infection >= 50:
            dmg += 0.15
        if dmg > 0:
            s.damage_player(dmg * dt, source="survival", feedback=False)
        elif st.hunger > 50 and st.thirst > 50 and st.infection < 30:
            p.hp = min(p.max_hp, p.hp + 0.18 * dt)

        # Injury / infection --------------------------------------------
        if st.hunger > 40 and st.thirst > 40:
            st.injury = max(0.0, st.injury - 0.05 * dt)
        tdef = s.world.terrain_def_at(p.x, p.y)
        contaminated = tdef.get("contaminated") or s.terrain_db.regions.get(region, {}).get("contamination", 0) > 0
        if contaminated:
            st.infection = min(100.0, st.infection + 0.10 * dt * (2.0 if st.bleeding > 0 else 1.0))
        elif st.infection > 0:
            if st.infection < 25 and st.hunger > 40 and st.thirst > 40:
                st.infection = max(0.0, st.infection - 0.03 * dt)
            elif st.infection >= 25:
                st.infection = min(100.0, st.infection + 0.035 * dt)

        self._warnings(st)

    def _warnings(self, st):
        w = []
        if st.thirst < 20:
            w.append(("탈수" if st.thirst > 0 else "심각한 탈수", (90, 170, 230)))
        if st.hunger < 20:
            w.append(("배고픔" if st.hunger > 0 else "굶주림", (230, 170, 70)))
        if st.body_temp < 35.5:
            w.append(("저체온", (130, 190, 255)))
        elif st.body_temp > 38.5:
            w.append(("고열" if st.infection >= 50 else "열사병 위험", (255, 120, 80)))
        if st.bleeding > 0:
            w.append(("출혈", (230, 50, 50)))
        if st.infection >= 25:
            w.append(("감염", (150, 210, 70)))
        if st.injury >= 40:
            w.append(("부상", (220, 140, 140)))
        if st.fatigue >= 80:
            w.append(("탈진", (180, 160, 210)))
        self.warnings = w

    # ------------------------------------------------------------ item use
    def apply_use(self, stack) -> tuple[bool, str]:
        s = self.session
        p = s.player
        idef = s.db.item(stack.item_id)
        use = idef.get("use")
        if not use:
            return False, ""
        st = p.stats
        if use.get("lore"):
            pages = s.lore
            page = pages[zlib.crc32(stack.uid.encode()) % len(pages)] if pages else "..."
            s.show_lore(page)
            return False, "일지를 읽었다"
        msgs = []
        if "hunger" in use:
            st.hunger = min(100.0, st.hunger + use["hunger"])
        if "thirst" in use:
            st.thirst = min(100.0, st.thirst + use["thirst"])
        if "health" in use:
            p.hp = min(p.max_hp, p.hp + use["health"])
        if "injury" in use:
            st.injury = max(0.0, st.injury + use["injury"])
        if use.get("stop_bleeding"):
            if st.bleeding > 0:
                msgs.append("출혈이 멈췄다")
            st.bleeding = 0.0
        if "infection" in use:
            amt = use["infection"]
            if amt > 0:
                # Contaminated food/water is a risk, not a guarantee.
                if s.rng.random() < 0.6:
                    st.infection = min(100.0, st.infection + amt)
                    msgs.append("속이 좋지 않다...")
            else:
                st.infection = max(0.0, st.infection + amt)
        p.inventory.consume_one(stack.uid)
        if idef.get("returns"):
            left = p.inventory.add_item(idef["returns"], 1)
            if left:
                s.drop_item(s.db.new_stack(idef["returns"], 1))
        s.audio.play("pickup", 0.5)
        return True, (f"{idef['name']} 사용" + (" — " + ", ".join(msgs) if msgs else ""))
