"""Buried-resource logic.

Deposits are a pure function of (world seed, cell, layer, region): the same cell always
holds the same thing, and regions/layers follow data/resources.json → dig_loot. This keeps
discoveries logical (ruins give scrap & electronics, military gives ammo) and lets a future
server reproduce results without storing the whole underground.
"""
import random

from systems.resource_system import roll_table


def cell_rng(seed: int, x: int, y: int, salt: int) -> random.Random:
    h = (x * 73856093) ^ (y * 19349663) ^ (salt * 83492791) ^ (seed * 2654435761)
    return random.Random(h & 0xFFFFFFFFFFFF)


class DepositTable:
    def __init__(self, resources: dict, terrain_db):
        self.dig_loot = resources["dig_loot"]
        self.terrain_db = terrain_db

    def layer_deposit(self, seed: int, x: int, y: int, depth: int, region: str, terrain_idx: int) -> list[tuple[str, int]]:
        table = self.dig_loot.get(region, self.dig_loot["wasteland"]).get(str(depth))
        rng = cell_rng(seed, x, y, depth)
        found = roll_table(table, rng) if table else []
        bonus = self.terrain_db.get(terrain_idx).get("bonus")
        if bonus and rng.random() < bonus[1]:
            found.append((bonus[0], 1))
        return found

    def cache_contents(self, seed: int, x: int, y: int, region: str) -> list[tuple[str, int]]:
        table = self.dig_loot.get(region, self.dig_loot["wasteland"]).get("cache", [])
        rng = cell_rng(seed, x, y, 99)
        return roll_table(table, rng, rolls=3)
