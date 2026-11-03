"""Terrain type database (data/terrain.json)."""

REGION_ORDER = ["wasteland", "ruins", "gas_station", "military", "contaminated"]


class TerrainDB:
    def __init__(self, data: dict):
        self.data = data
        self.types: list[dict] = data["terrain_types"]
        self.index = {t["id"]: i for i, t in enumerate(self.types)}
        self.layers = {l["depth"]: l for l in data["layers"]}
        self.hand = data["hand_digging"]
        self.regions = data["regions"]
        self.region_order = [r for r in REGION_ORDER if r in self.regions]
        self.region_index = {r: i for i, r in enumerate(self.region_order)}
        self.landmarks = data.get("landmarks", [])

    def tid(self, name: str) -> int:
        return self.index[name]

    def get(self, idx: int) -> dict:
        return self.types[idx]

    def region_name(self, idx: int) -> str:
        return self.region_order[idx]

    def layer_work(self, terrain_idx: int, target_depth: int) -> float:
        """Total work units needed to dig *into* target_depth (1..3)."""
        layer = self.layers[target_depth]
        return layer["work"] * layer.get("time_mult", 1.0) * self.types[terrain_idx]["hardness"]

    def layer_name(self, depth: int) -> str:
        if depth <= 0:
            return "지표면"
        return self.layers[depth]["name"]

    def can_dig(self, terrain_idx: int, current_depth: int, tool: dict) -> tuple[bool, str]:
        """tool: {'dig_power', 'max_depth', 'soft_only'(opt)}"""
        t = self.types[terrain_idx]
        if not t["diggable"]:
            return False, f"{t['name']}은(는) 팔 수 없다"
        if current_depth >= t["max_depth"]:
            return False, "더 깊이 팔 수 없는 지층이다"
        if current_depth >= tool["max_depth"]:
            return False, "이 도구로는 더 깊이 팔 수 없다"
        if tool.get("soft_only") and not t["soft"]:
            return False, "맨손으로 파기엔 땅이 너무 단단하다"
        if tool["dig_power"] < t.get("min_power", 0):
            return False, "더 튼튼한 도구가 필요하다"
        return True, ""
