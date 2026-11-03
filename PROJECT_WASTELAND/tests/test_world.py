from world.world_manager import WorldManager


def make(data, seed):
    return WorldManager(seed, data.terrain_db, data.resources, data.db)


def test_generation_is_deterministic(data):
    a, b = make(data, 77), make(data, 77)
    for x, y in ((10, 10), (80, 80), (120, 40), (33, 140)):
        ca, cb = a.cell(x, y), b.cell(x, y)
        assert (ca.terrain, ca.region, ca.flags) == (cb.terrain, cb.region, cb.flags)
    ida = sorted(o.id for o in a.objects_near(80, 80, 20))
    idb = sorted(o.id for o in b.objects_near(80, 80, 20))
    assert ida == idb and ida


def test_all_regions_present_and_spawn_resources(data):
    w = make(data, 5)
    assert {lm["region"] for lm in w.gen.landmarks} == {"ruins", "gas_station", "military", "contaminated"}
    sx, sy = w.gen.spawn
    w.ensure_area(sx, sy, 30)
    kinds = {o.otype for o in w.objects_near(sx, sy, 12)}
    assert {"stone_pile", "branch_pile", "puddle"} <= kinds
    assert any(o.otype == "police_car" for o in w.objects_near(sx, sy, 25))


def test_deposits_are_deterministic_and_regional(data):
    w = make(data, 9)
    seq = [w.deposits.layer_deposit(9, 50, 60, d, "military", 1) for d in (1, 2, 3)]
    assert seq == [w.deposits.layer_deposit(9, 50, 60, d, "military", 1) for d in (1, 2, 3)]
    found = set()
    for x in range(200):
        for item, _ in w.deposits.layer_deposit(9, x, 3, 3, "military", 1):
            found.add(item)
    assert found & {"ammo_rifle", "ammo_shell", "military_ration", "gun_parts"}
    shallow = {i for x in range(200) for i, _ in w.deposits.layer_deposit(9, x, 3, 1, "wasteland", 1)}
    assert "ammo_rifle" not in shallow


def test_dig_rules(data):
    tdb = data.terrain_db
    hands = {"dig_power": 0.35, "max_depth": 1, "soft_only": True}
    shovel = {"dig_power": 1.0, "max_depth": 2}
    assert tdb.can_dig(tdb.tid("sand"), 0, hands)[0]
    assert not tdb.can_dig(tdb.tid("clay"), 0, hands)[0]          # too hard for bare hands
    assert not tdb.can_dig(tdb.tid("sand"), 1, hands)[0]          # tool depth limit
    assert tdb.can_dig(tdb.tid("sand"), 1, shovel)[0]
    assert not tdb.can_dig(tdb.tid("concrete"), 0, shovel)[0]
    assert not tdb.can_dig(tdb.tid("asphalt"), 0, {"dig_power": 0.6, "max_depth": 1})[0]
    assert tdb.layer_work(tdb.tid("clay"), 2) > tdb.layer_work(tdb.tid("sand"), 2)


def test_complete_dig_layer_changes_cell(data):
    w = make(data, 11)
    w.complete_dig_layer(70, 70)
    assert w.cell(70, 70).depth == 1
    assert (70, 70, 1) in {(x, y, d) for x, y, d, _, _ in w.modified_cells()}
