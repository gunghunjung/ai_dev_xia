import math


def soft_cell_near(s):
    p = s.player
    for r in range(1, 8):
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                c = (int(p.x) + dx, int(p.y) + dy)
                info = s.world.cell(*c)
                if (info.depth == 0 and s.terrain_db.get(info.terrain)["soft"]
                        and not s.world.objects_near(c[0] + .5, c[1] + .5, 1.0)):
                    return c


def test_start_has_nothing_but_clothes(session):
    p = session.player
    assert p.body is not None and p.body.item_id == "old_clothes"
    assert not p.inventory.stacks and p.equipped_uid is None


def test_crafting_consumes_and_requires_station(session):
    s = session
    inv = s.player.inventory
    inv.add_item("stone", 2)
    inv.add_item("branch", 1)
    inv.add_item("plant_fiber", 2)
    ok, _ = s.crafting.start("stone_axe")
    assert ok
    for _ in range(200):
        s.crafting.update(1 / 60)
    assert inv.count("stone_axe") == 1 and inv.count("stone") == 0
    inv.add_item("scrap_metal", 3)
    inv.add_item("wood", 2)
    ok, why = s.crafting.start("shovel")
    assert not ok and "작업대" in why
    s.world.place_structure("workbench", s.player.x + 1, s.player.y)
    assert s.crafting.start("shovel")[0]


def test_repair_needs_materials(session):
    s = session
    inv = s.player.inventory
    inv.add_item("stick", 1)
    st = inv.find_first("stick")
    st.durability = 10
    assert not s.crafting.repair(st)[0]
    inv.add_item("branch", 1)
    assert s.crafting.repair(st)[0] and st.durability == 35 and inv.count("branch") == 0


def test_digging_with_tool_reveals_layers(session):
    s = session
    p = s.player
    p.inventory.add_item("crude_shovel", 1)
    s.equip(p.inventory.find_first("crude_shovel").uid)
    cell = soft_cell_near(s)
    p.x, p.y = cell[0] - 0.5, cell[1] + 0.5
    s.digging.update_target(cell)
    assert s.digging.target_ok
    for _ in range(60 * 30):
        if not s.digging.dig_tick(1 / 60):
            s.digging.update_target(cell)
            if not s.digging.target_ok:
                break
    assert s.world.cell(*cell).depth == 2           # crude shovel max depth
    assert p.inventory.find_first("crude_shovel").durability < 60


def test_hands_cannot_dig_deep(session):
    s = session
    cell = soft_cell_near(s)
    s.digging.update_target(cell)
    assert s.digging.target_ok
    s.world.complete_dig_layer(*cell)
    s.digging.target = None
    s.digging.update_target(cell)
    assert not s.digging.target_ok


def test_melee_hits_in_arc_with_knockback(session):
    s = session
    p = s.player
    front = s.spawner.make("infected", p.x + 1.0, p.y)
    behind = s.spawner.make("infected", p.x - 1.0, p.y)
    s.enemies += [front, behind]
    p.aim_angle = 0.0
    assert s.combat.start_melee(p)
    for _ in range(20):
        s.combat.update_player_attack(p, 1 / 60)
    assert front.hp < front.max_hp and front.kx > 0
    assert behind.hp == behind.max_hp


def test_pistol_fire_consumes_ammo_and_hits(session):
    s = session
    p = s.player
    p.inventory.add_item("pistol", 1)
    p.inventory.add_item("ammo_9mm", 10)
    gun = p.inventory.find_first("pistol")
    s.equip(gun.uid)
    assert s.combat.start_reload(p)
    for _ in range(120):
        s.combat.update_reload(p, 1 / 60)
    assert gun.ammo == 8 and p.inventory.count("ammo_9mm") == 2
    e = s.spawner.make("infected", p.x + 4, p.y)
    s.enemies.append(e)
    p.aim_angle = 0.0
    p.aiming = True
    assert s.combat.try_fire(p)
    for _ in range(30):
        s.combat.update_projectiles(1 / 60)
    assert gun.ammo == 7 and e.hp < e.max_hp


def test_enemy_hears_gunshot(session):
    s = session
    p = s.player
    e = s.spawner.make("infected", p.x + 18, p.y)
    s.enemies.append(e)
    s.noise.emit(p.x, p.y, "gunshot", 26)
    assert e.state == "investigate"


def test_enemy_attack_damages_player_unless_dodging(session):
    s = session
    p = s.player
    e = s.spawner.make("raider", p.x + 0.8, p.y)
    s.enemies.append(e)
    e.facing = math.pi
    hp = p.hp
    p.iframes = 0.3
    s.combat.enemy_strike(e)
    assert p.hp == hp
    p.iframes = 0
    s.combat.enemy_strike(e)
    assert p.hp < hp


def test_survival_decay_and_effects(session):
    s = session
    st = s.player.stats
    h, t = st.hunger, st.thirst
    for _ in range(600):
        s.survival.update(1 / 60, "sprint")
    assert st.hunger < h and st.thirst < t
    st.bleeding = 2.0
    hp = s.player.hp
    for _ in range(120):
        s.survival.update(1 / 60, "idle")
    assert s.player.hp < hp
    s.player.inventory.add_item("bandage", 1)
    s.use_stack(s.player.inventory.find_first("bandage").uid)
    assert st.bleeding == 0


def test_dirty_water_returns_bottle(session):
    s = session
    inv = s.player.inventory
    inv.add_item("dirty_water", 1)
    st = s.player.stats
    st.thirst = 10
    s.use_stack(inv.find_first("dirty_water").uid)
    assert st.thirst > 10 and inv.count("empty_bottle") == 1


def test_gather_and_search_interactions(session):
    s = session
    p = s.player
    pile = next(o for o in s.world.objects_near(p.x, p.y, 12) if o.otype == "stone_pile")
    p.x, p.y = pile.x - 0.6, pile.y
    s.begin_interaction(pile)
    for _ in range(60):
        if p.action:
            s.progress_action(1 / 60)
    assert p.inventory.count("stone") >= 1 and pile.removed
    car = next(o for o in s.world.objects_near(p.x, p.y, 40) if o.otype == "police_car")
    p.x, p.y = car.x - 1.2, car.y
    s.begin_interaction(car)
    for _ in range(240):
        if p.action:
            s.progress_action(1 / 60)
    assert p.inventory.count("pistol") == 1 and car.uses == 0


def test_structure_placement_and_campfire_station(session):
    s = session
    p = s.player
    p.inventory.add_item("campfire", 1)
    s.use_stack(p.inventory.find_first("campfire").uid)
    assert p.placing is not None
    p.placing["cell"] = (int(p.x) + 2, int(p.y))
    s.place_structure()
    assert p.inventory.count("campfire") == 0
    assert s.world_system.campfires_near(p.x, p.y, 5)
    assert "campfire" in s.crafting.stations_near(p.x, p.y)


def test_simulation_runs_with_spawns(session):
    s = session
    s.spawner.timer = 0
    for _ in range(60 * 30):
        s.update(1 / 60)
    assert s.enemies
    assert not math.isnan(s.player.x)
