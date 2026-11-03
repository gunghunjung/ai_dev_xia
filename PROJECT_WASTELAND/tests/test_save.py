import sqlite3

from config import DEFAULT_SETTINGS
from database.save_manager import SaveManager


def fresh(data, audio, seed):
    from game.session import GameSession
    return GameSession(data, dict(DEFAULT_SETTINGS), audio, seed=seed, new_game=False)


def test_save_load_roundtrip(tmp_path, session, data, audio):
    s = session
    p = s.player
    p.inventory.add_item("pipe", 1)
    p.inventory.add_item("stone", 7)
    pipe = p.inventory.find_first("pipe")
    pipe.durability = 33
    s.equip(pipe.uid)
    p.hotbar[0] = pipe.uid
    p.stats.hunger = 42
    s.world.complete_dig_layer(70, 71)
    s.world.complete_dig_layer(70, 71)
    pile = next(o for o in s.world.objects_near(p.x, p.y, 10) if o.otype == "stone_pile")
    s.world.remove_object(pile)
    box = s.world.place_structure("storage_box", p.x + 2, p.y)
    s.container_inventory(box).add_item("canned_food", 2)
    s.world.place_structure("campfire", p.x - 2, p.y)
    s.clock.time = 0.61
    sm = SaveManager(str(tmp_path))
    assert sm.save(s, "t")

    s2 = fresh(data, audio, s.seed)
    assert sm.load(s2, "t")
    p2 = s2.player
    assert p2.inventory.count("stone") == 7
    pipe2 = p2.inventory.find_first("pipe")
    assert pipe2.durability == 33 and p2.equipped_uid == pipe2.uid and p2.hotbar[0] == pipe2.uid
    assert abs(p2.stats.hunger - 42) < 1e-6
    assert s2.world.cell(70, 71).depth == 2
    assert s2.world.objects_by_id[pile.id].removed
    boxes = [o for o in s2.world.objects_near(p.x, p.y, 4) if o.otype == "storage_box"]
    assert boxes and s2.container_inventory(boxes[0]).count("canned_food") == 2
    assert any(o.otype == "campfire" for o in s2.world.objects_near(p.x, p.y, 4))
    assert abs(s2.clock.time - 0.61) < 1e-4

    # Re-saving a loaded game must keep changes in chunks that were never visited this time.
    far = (5, 150)
    s2.world.complete_dig_layer(*far)
    assert sm.save(s2, "t")
    s3 = fresh(data, audio, s.seed)
    assert sm.load(s3, "t")
    assert s3.world.cell(70, 71).depth == 2 and s3.world.cell(*far).depth == 1


def test_corrupted_save_falls_back_to_backup(tmp_path, session, data, audio):
    sm = SaveManager(str(tmp_path))
    session.player.stats.hunger = 55
    assert sm.save(session, "c")
    session.player.stats.hunger = 20
    assert sm.save(session, "c")                     # previous save becomes .bak
    with open(sm.path("c"), "wb") as fh:
        fh.write(b"not a database at all" * 50)
    s2 = fresh(data, audio, session.seed)
    assert sm.load(s2, "c")
    assert abs(s2.player.stats.hunger - 55) < 1e-6


def test_newer_save_version_rejected(tmp_path, session, data, audio):
    sm = SaveManager(str(tmp_path))
    assert sm.save(session, "v")
    with sqlite3.connect(sm.path("v")) as conn:
        conn.execute("UPDATE meta SET value = ? WHERE key = 'save_version'", ("999",))
    s2 = fresh(data, audio, session.seed)
    assert not sm.load(s2, "v")
