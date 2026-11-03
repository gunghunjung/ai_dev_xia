"""Persistence of world changes (dug cells, object state, structures, containers) and the player."""
import json

from interfaces.player_interface import IPlayerRepository
from systems.resource_system import ItemStack


class WorldRepository:
    def save(self, conn, session):
        world = session.world
        session.sync_containers()
        conn.execute("DELETE FROM world_cells")
        conn.executemany("INSERT INTO world_cells (x, y, depth, work, flags) VALUES (?, ?, ?, ?, ?)",
                         list(world.modified_cells()))
        conn.execute("DELETE FROM world_objects")
        rows = []
        seen = set()
        for oid, o in world.objects_by_id.items():
            changed = (not o.generated or o.removed or o.uses != o.defn.get("uses", 1) or o.data)
            if not changed:
                continue
            if o.kind == "pickup" and o.removed:
                continue
            seen.add(oid)
            rows.append((oid, o.kind, o.otype, o.x, o.y, json.dumps(o.state(), ensure_ascii=False)))
        # Overrides for chunks that were never loaded this session must survive a re-save.
        for oid, st in world.object_overrides.items():
            if oid not in seen and oid not in world.objects_by_id:
                rows.append((oid, st.get("kind", "object"), st.get("otype", ""), st.get("x", 0), st.get("y", 0),
                             json.dumps(st, ensure_ascii=False)))
        conn.executemany("INSERT INTO world_objects (id, kind, type, x, y, state) VALUES (?, ?, ?, ?, ?, ?)", rows)

    def load(self, conn, session):
        world = session.world
        world.cell_overrides = {(x, y): (d, w, f) for x, y, d, w, f in
                                conn.execute("SELECT x, y, depth, work, flags FROM world_cells")}
        world.object_overrides = {}
        dynamic = []
        max_dyn = 0
        for oid, kind, otype, x, y, state in conn.execute("SELECT id, kind, type, x, y, state FROM world_objects"):
            st = json.loads(state)
            if kind == "object":
                st.update({"kind": kind, "otype": otype, "x": x, "y": y})
                world.object_overrides[oid] = st
            else:
                dynamic.append((oid, kind, otype, x, y, st))
            if oid[:1] in ("b", "p") and oid[1:].isdigit():
                max_dyn = max(max_dyn, int(oid[1:]))
        world.dynamic_count = max_dyn
        world.chunks.clear()
        world.obj_grid.clear()
        world.objects_by_id.clear()
        for oid, kind, otype, x, y, st in dynamic:
            if kind == "structure" and otype in session.resources["structures"]:
                obj = world.place_structure(otype, x, y, oid=oid)
                obj.apply_state(st)
            elif kind == "pickup" and st.get("stack") and session.db.exists(st["stack"]["item_id"]):
                world.spawn_pickup(ItemStack.from_dict(st["stack"]), x, y, pop=False, oid=oid)


class PlayerRepository(IPlayerRepository):
    def save_player(self, conn, player) -> None:
        conn.execute("DELETE FROM player")
        conn.execute("DELETE FROM items WHERE owner IN ('player', 'body')")
        conn.execute(
            "INSERT INTO player (id, x, y, hp, stats, equipped, hotbar, spawn) VALUES (1, ?, ?, ?, ?, ?, ?, ?)",
            (player.x, player.y, player.hp, json.dumps(player.stats.to_dict()), player.equipped_uid,
             json.dumps(player.hotbar), json.dumps(player.spawn_point)))
        rows = [(d["uid"], "player", d["item_id"], d["count"], d["durability"], d["ammo"], d["gx"], d["gy"],
                 json.dumps(d["data"])) for d in player.inventory.to_list()]
        if player.body is not None:
            b = player.body.to_dict()
            rows.append((b["uid"], "body", b["item_id"], b["count"], b["durability"], b["ammo"], 0, 0, json.dumps(b["data"])))
        conn.executemany("INSERT INTO items (uid, owner, item_id, count, durability, ammo, gx, gy, data) "
                         "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)

    def load_player(self, conn, player) -> bool:
        row = conn.execute("SELECT x, y, hp, stats, equipped, hotbar, spawn FROM player WHERE id = 1").fetchone()
        if row is None:
            return False
        x, y, hp, stats, equipped, hotbar, spawn = row
        player.x, player.y, player.hp = x, y, hp
        player.stats.load(json.loads(stats))
        inv_rows = []
        player.body = None
        for uid, owner, item_id, count, dur, ammo, gx, gy, data in conn.execute(
                "SELECT uid, owner, item_id, count, durability, ammo, gx, gy, data FROM items WHERE owner IN ('player', 'body')"):
            d = {"uid": uid, "item_id": item_id, "count": count, "durability": dur, "ammo": ammo, "gx": gx, "gy": gy,
                 "data": json.loads(data or "{}")}
            if owner == "body":
                if player.db.exists(item_id):
                    player.body = ItemStack.from_dict(d)
            else:
                inv_rows.append(d)
        player.inventory.load_list(inv_rows)
        player.equipped_uid = equipped if equipped in player.inventory.stacks else None
        hb = json.loads(hotbar)
        player.hotbar = [u if u in player.inventory.stacks else None for u in hb] + [None] * max(0, len(player.hotbar) - len(hb))
        player.hotbar = player.hotbar[:5]
        player.spawn_point = tuple(json.loads(spawn)) if spawn else (x, y)
        return True
