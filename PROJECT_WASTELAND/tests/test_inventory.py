from systems.inventory_system import GridInventory


def test_stacking_and_overflow(data):
    inv = GridInventory(data.db, 2, 1)
    assert inv.add_item("stone", 15) == 0          # 10 + 5 in two cells
    assert inv.count("stone") == 15
    assert inv.add_item("stone", 10) == 5          # second stack tops up to 10, rest does not fit
    assert inv.count("stone") == 20


def test_item_sizes_block_cells(data):
    inv = GridInventory(data.db, 3, 3)
    inv.add_item("stone_axe", 1)                   # 1x2
    st = inv.find_first("stone_axe")
    assert inv.pos[st.uid] == (0, 0)
    assert inv.stack_at(0, 1) is st
    assert not inv.can_place("pistol", 0, 1)       # 2x1 overlapping the axe
    assert inv.can_place("pistol", 1, 0)
    assert not inv.can_place("pistol", 2, 0)       # out of bounds


def test_split_merge_move_remove(data):
    inv = GridInventory(data.db, 4, 4)
    inv.add_item("branch", 8)
    st = inv.find_first("branch")
    new = inv.split(st.uid, 3)
    assert new is not None and st.count == 5 and new.count == 3
    assert inv.move(new.uid, 3, 3)
    assert inv.merge(new.uid, st.uid) and st.count == 8 and new.uid not in inv.stacks
    assert inv.remove_item("branch", 5) and inv.count("branch") == 3
    assert not inv.remove_item("branch", 4)


def test_weight_and_durability(data):
    inv = GridInventory(data.db, 4, 4)
    inv.add_item("stone", 4)
    inv.add_item("pipe", 1)
    assert abs(inv.total_weight() - (4 * 0.5 + 1.6)) < 1e-6
    assert inv.find_first("pipe").durability == data.items["pipe"]["durability"]


def test_serialization_roundtrip(data):
    inv = GridInventory(data.db, 4, 4)
    inv.add_item("canned_food", 3)
    inv.add_item("pistol", 1)
    rows = inv.to_list()
    other = GridInventory(data.db, 4, 4)
    other.load_list(rows)
    assert other.count("canned_food") == 3 and other.count("pistol") == 1
