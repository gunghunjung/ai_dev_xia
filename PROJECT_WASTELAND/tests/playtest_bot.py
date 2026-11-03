"""Scripted end-to-end playtest: drives the real App through injected input events.

Run:  python tests/playtest_bot.py [out_dir]
It walks through the first-playable checklist (move, aim, gather, craft, dig, fight, shoot,
reload, inventory, save/load) and writes screenshots + a report. Uses SDL dummy drivers so it
also runs on CI without a display.
"""
import math
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame  # noqa: E402

from engine.camera import iso  # noqa: E402


class Bot:
    def __init__(self, out_dir, seed=None):
        from game.app import App
        self.out = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.app = App(save_dir=os.path.join(out_dir, "saves"))
        self.app.loop.deterministic = True
        if seed is not None:  # a fixed world keeps the scripted run repeatable
            new_game = self.app.new_game
            self.app.new_game = lambda _seed=None: new_game(seed)
        self.report = []
        self.shots = 0

    # -------------------------------------------------- primitives
    def frames(self, n=1):
        self.app.loop.run(max_frames=n)

    def key(self, k, down=True):
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN if down else pygame.KEYUP, key=k, mod=0, unicode="", scancode=0))

    def tap(self, k, hold=2):
        self.key(k, True)
        self.frames(hold)
        self.key(k, False)
        self.frames(1)

    def hold(self, k, frames):
        self.key(k, True)
        self.frames(frames)
        self.key(k, False)
        self.frames(1)

    def dig_at(self, x, y, frames):
        """Hold dig while keeping the cursor on (x, y), as a player does while the camera drifts."""
        self.key(pygame.K_f, True)
        for _ in range(frames):
            self.aim_at_world(x, y)
        self.key(pygame.K_f, False)
        self.frames(1)

    def mouse_to(self, pos):
        pygame.event.post(pygame.event.Event(pygame.MOUSEMOTION, pos=pos, rel=(5, 5), buttons=(0, 0, 0)))

    def click(self, pos, button=1, hold=2):
        self.mouse_to(pos)
        pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=pos, button=button))
        self.frames(hold)
        pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONUP, pos=pos, button=button))
        self.frames(1)

    def shot(self, name):
        self.shots += 1
        path = os.path.join(self.out, f"{self.shots:02d}_{name}.png")
        pygame.image.save(self.app.screen, path)
        return path

    def check(self, label, ok, detail=""):
        self.report.append((label, bool(ok), detail))
        print(("PASS " if ok else "FAIL ") + label + (f" — {detail}" if detail else ""))

    @property
    def s(self):
        return self.app.session

    def aim_at_world(self, x, y, z=0):
        sx, sy = self.app.camera.world_to_screen(x, y, z)
        self.mouse_to((int(sx), int(sy)))
        self.frames(1)

    def walk_to(self, x, y, max_frames=600, stop=0.6):
        """Steer with WASD toward a world point (screen-space steering like a human)."""
        keys_down = set()
        for _ in range(max_frames):
            p = self.s.player
            dx, dy = x - p.x, y - p.y
            if math.hypot(dx, dy) < stop:
                break
            sx, sy = iso(dx, dy)
            want = set()
            if sx > 6: want.add(pygame.K_d)
            if sx < -6: want.add(pygame.K_a)
            if sy > 4: want.add(pygame.K_s)
            if sy < -4: want.add(pygame.K_w)
            for k in keys_down - want:
                self.key(k, False)
            for k in want - keys_down:
                self.key(k, True)
            keys_down = want
            self.frames(1)
        for k in keys_down:
            self.key(k, False)
        self.frames(2)
        p = self.s.player
        return math.hypot(x - p.x, y - p.y) < stop + 0.5

    def nearest(self, otype, radius=40):
        p = self.s.player
        objs = [o for o in self.s.world.objects_near(p.x, p.y, radius) if o.otype == otype and o.interact]
        return min(objs, key=lambda o: math.hypot(o.x - p.x, o.y - p.y)) if objs else None

    def clear_spot(self, dist):
        """Move the player to open ground and return a point `dist` tiles away with no blocking
        prop on the line between (bullets correctly stop at walls and cars)."""
        p, world = self.s.player, self.s.world

        def clear(x0, y0, x1, y1):
            steps = int(math.hypot(x1 - x0, y1 - y0) / 0.3)
            return all(not any(o.blocking and math.hypot(o.x - sx, o.y - sy) < o.radius + 0.4
                               for o in world.objects_near(sx, sy, 1.5))
                       for sx, sy in ((x0 + (x1 - x0) * i / steps, y0 + (y1 - y0) * i / steps)
                                      for i in range(steps + 1)))

        for r in (0, 2, 4, 6):
            for start in range(0, 360, 45 if r else 360):
                px, py = p.x + math.cos(math.radians(start)) * r, p.y + math.sin(math.radians(start)) * r
                for deg in (180, 0, 90, 270, 135, 225, 45, 315):
                    a = math.radians(deg)
                    x, y = px + math.cos(a) * dist, py + math.sin(a) * dist
                    if clear(px, py, x, y):
                        p.x, p.y = px, py
                        self.frames(2)
                        return x, y
        return p.x - dist, p.y

    def interact_with(self, obj, wait=240):
        self.walk_to(obj.x, obj.y, stop=0.9)
        self.aim_at_world(obj.x, obj.y)
        self.tap(pygame.K_e)
        self.frames(wait)

    # -------------------------------------------------- scenario
    def run(self):
        app = self.app
        self.frames(5)
        self.shot("title")
        self.tap(pygame.K_RETURN)  # "새 게임" is preselected when no save exists
        self.frames(10)
        self.check("1. 게임 시작 / 황무지 월드 표시", app.scene == "game" and self.s is not None)
        s = self.s
        s.spawner.timer = 9999  # keep the scripted run deterministic: no random spawns
        s.enemies.clear()
        self.shot("spawn")
        p = s.player
        x0, y0 = p.x, p.y
        self.hold(pygame.K_d, 45)
        self.check("2. 플레이어 이동", math.hypot(p.x - x0, p.y - y0) > 1.0, f"{math.hypot(p.x - x0, p.y - y0):.2f} tiles")

        angles = []
        for ang in (0, 90, 180, 270):
            r = math.radians(ang)
            sx, sy = app.camera.world_to_screen(p.x, p.y, 22)
            self.mouse_to((int(sx + math.cos(r) * 200), int(sy + math.sin(r) * 200)))
            self.frames(2)
            angles.append(round(math.degrees(p.aim_screen_angle)) % 360)
        self.check("3. 마우스 360도 조준", len(set(angles)) == 4, str(angles))

        # Gather stones, branches, fibre
        for otype, n in (("stone_pile", 3), ("branch_pile", 3), ("dry_bush", 2)):
            for _ in range(n):
                o = self.nearest(otype)
                if o:
                    self.interact_with(o, 80)
        inv = p.inventory
        self.check("10. 돌과 나뭇가지 채집", inv.count("stone") >= 2 and inv.count("branch") >= 2,
                   f"stone {inv.count('stone')} branch {inv.count('branch')} fiber {inv.count('plant_fiber')}")
        self.shot("gathered")

        # Craft via crafting UI (keyboard)
        self.tap(pygame.K_c)
        self.frames(3)
        self.shot("crafting_ui")
        ui = app.game_menu.pages["crafting"]
        ents = ui.entries()
        target = next((i for i, e in enumerate(ents) if e["id"] == "crude_shovel"), None)
        if target is not None and inv.count("plant_fiber") < 2:
            inv.add_item("plant_fiber", 4)  # bushes are random; top up so the scripted run is stable
        ents = ui.entries()
        target = next(i for i, e in enumerate(ents) if e["id"] == "crude_shovel")
        ui.index = target
        self.tap(pygame.K_RETURN)
        self.frames(150)
        self.tap(pygame.K_ESCAPE)
        self.check("11. 간단한 도구 제작 (간이 삽)", inv.count("crude_shovel") == 1)

        # Inventory: open, equip shovel through context menu
        self.tap(pygame.K_i)
        self.frames(2)
        inv_ui = app.game_menu.pages["inventory"]
        st = inv.find_first("crude_shovel")
        gx, gy = inv.pos[st.uid]
        inv_ui.cursor = ["player", gx, gy]
        self.tap(pygame.K_x)  # context menu
        self.shot("inventory_context")
        self.tap(pygame.K_RETURN)  # "장착"
        self.frames(2)
        self.shot("inventory")
        moved_ok = inv.move(st.uid, 7, 4)
        self.tap(pygame.K_ESCAPE)
        self.check("15. 인벤토리 (장착/이동)", p.equipped_item_id() == "crude_shovel" and moved_ok)

        # Dig a diggable cell next to the player
        world = s.world
        cell = None
        for r in range(1, 6):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    c = (int(p.x) + dx, int(p.y) + dy)
                    info = world.cell(*c)
                    if info and info.depth == 0 and s.terrain_db.get(info.terrain)["soft"] and not world.cell_has_blocker(*c) \
                            and not world.objects_near(c[0] + 0.5, c[1] + 0.5, 0.9):
                        cell = c
                        break
                if cell: break
            if cell: break
        self.walk_to(cell[0] + 0.5 - 1.0, cell[1] + 0.5, stop=0.4)
        self.aim_at_world(cell[0] + 0.5, cell[1] + 0.5)
        before = sum(1 for o in world.all_live_objects() if o.kind == "pickup")
        self.dig_at(cell[0] + 0.5, cell[1] + 0.5, 60)
        self.shot("digging")
        self.dig_at(cell[0] + 0.5, cell[1] + 0.5, 600)
        info = world.cell(*cell)
        self.check("12. 실제 지형 굴착 (깊이 데이터 변경)", info.depth >= 1, f"cell {cell} depth {info.depth}")
        self.dig_at(cell[0] + 0.5, cell[1] + 0.5, 900)
        info = world.cell(*cell)
        after = sum(1 for o in world.all_live_objects() if o.kind == "pickup")
        found_msgs = [m["text"] for m in s.messages]
        self.check("13. 매장 자원 발견 로직", info.depth >= 2, f"depth {info.depth}, pickups {before}->{after}, msgs {found_msgs[-2:]}")
        self.shot("dug")

        # Melee: craft stone axe & fight an infected
        inv.add_item("stone", 2); inv.add_item("branch", 1); inv.add_item("plant_fiber", 2)
        s.crafting.start("stone_axe")
        self.frames(140)
        axe = inv.find_first("stone_axe")
        s.equip(axe.uid)
        e = s.spawner.make("infected", p.x + 4, p.y)
        s.enemies.append(e)
        e.set_state("chase")
        self.frames(30)
        self.check("8. 적이 플레이어를 추적", math.hypot(e.x - p.x, e.y - p.y) < 4, f"dist {math.hypot(e.x - p.x, e.y - p.y):.2f}, state {e.state}")
        hp0 = e.hp
        hit_kb = False
        for _ in range(12):
            self.aim_at_world(e.x, e.y, 22)
            ex0, ey0 = e.x, e.y
            pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=self.app.input.mouse_pos, button=1))
            self.frames(6)
            if e.hp < hp0 and math.hypot(e.x - ex0, e.y - ey0) > 0.05:
                hit_kb = True
            pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONUP, pos=(0, 0), button=1))
            self.frames(30)
            if self.shots < 8:
                self.shot("melee")
            if e.dead:
                break
        self.check("5. 도끼로 공격", e.hp < hp0, f"hp {hp0:.0f}->{e.hp:.0f}")
        self.check("9. 피격과 넉백", hit_kb)

        # Pistol: find the police car, search it, shoot & reload
        car = self.nearest("police_car", 40)
        got_pistol = False
        if car:
            s.player.x, s.player.y = car.x - 1.4, car.y
            self.frames(5)
            self.interact_with(car, 220)
            got_pistol = inv.count("pistol") == 1
        self.check("경찰차 수색으로 권총 획득", got_pistol, f"ammo {inv.count('ammo_9mm')}")
        pistol = inv.find_first("pistol")
        if pistol:
            s.equip(pistol.uid)
            self.tap(pygame.K_r)
            self.frames(100)
            loaded = pistol.ammo
            tx, ty = self.clear_spot(5.0)  # a clear line of fire (shots stop at walls and cars)
            target = s.spawner.make("infected", tx, ty)
            s.enemies.append(target)
            hp0 = target.hp
            for _ in range(4):
                self.aim_at_world(target.x, target.y, 22)
                self.click(self.app.input.mouse_pos, 1, 2)
                self.frames(18)
                if self.shots < 11:
                    self.shot("shooting")
            self.check("6. 권총 사격", target.hp < hp0 or target.dead, f"hp {hp0:.0f}->{target.hp:.0f}")
            spent = loaded - pistol.ammo
            reserve = inv.count("ammo_9mm")
            self.tap(pygame.K_r)
            self.frames(100)
            self.check("7. 탄약 소비와 재장전", spent > 0 and (pistol.ammo > loaded - spent or reserve == 0),
                       f"loaded {loaded}, spent {spent}, reserve {reserve}, now {pistol.ammo}")

        # Survival decay
        h0, t0 = p.stats.hunger, p.stats.thirst
        self.frames(300)
        self.check("14. 허기와 갈증 감소", p.stats.hunger < h0 and p.stats.thirst < t0,
                   f"hunger {h0:.2f}->{p.stats.hunger:.2f}, thirst {t0:.2f}->{p.stats.thirst:.2f}")

        # Night & HUD screenshot
        s.clock.time = 0.88
        s.world.place_structure("campfire", p.x + 1.5, p.y + 0.5)
        self.frames(20)
        self.shot("night_campfire")
        s.weather.set("rain"); s.weather.blend = 1.0
        self.frames(5)
        self.shot("rain")
        s.weather.set("clear"); s.weather.blend = 1.0
        s.clock.time = 0.45

        # Save / load
        dug = (cell, world.cell(*cell).depth)
        px, py = p.x, p.y
        counts = {k: inv.count(k) for k in ("stone_axe", "pistol", "crude_shovel")}
        self.tap(pygame.K_F5)
        self.frames(3)
        p.x += 10  # mutate after saving
        self.tap(pygame.K_F9)
        self.frames(5)
        s2 = app.session
        ok = (s2 is not s and abs(s2.player.x - px) < 0.01 and s2.world.cell(*dug[0]).depth == dug[1]
              and all(s2.player.inventory.count(k) == v for k, v in counts.items()))
        self.check("16. 저장과 불러오기", ok, f"depth {s2.world.cell(*dug[0]).depth} pos {s2.player.x:.1f},{s2.player.y:.1f}")
        self.shot("after_load")
        self.check("성능: 평균 업데이트 시간", app.profiler.avg_ms("update") < 16, f"{app.profiler.avg_ms('update'):.2f} ms / render {app.profiler.avg_ms('render'):.2f} ms")
        return self.report


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(tempfile.gettempdir(), "wasteland_playtest")
    bot = Bot(out, int(sys.argv[2]) if len(sys.argv) > 2 else None)
    report = bot.run()
    failed = [r for r in report if not r[1]]
    print(f"\n{len(report) - len(failed)}/{len(report)} checks passed. Screenshots: {out}")
    pygame.quit()
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
