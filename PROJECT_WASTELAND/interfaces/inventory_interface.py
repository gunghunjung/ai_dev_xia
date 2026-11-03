"""Inventory and item-database contracts (server-authority ready)."""
from abc import ABC, abstractmethod


class IItemDatabase(ABC):
    @abstractmethod
    def item(self, item_id: str) -> dict: ...

    @abstractmethod
    def weapon(self, weapon_id: str) -> dict | None: ...

    @abstractmethod
    def new_stack(self, item_id: str, count: int = 1): ...


class IInventoryService(ABC):
    """All inventory mutations go through this service so a network implementation can
    let the server validate and confirm them."""

    @abstractmethod
    def add_item(self, item_id: str, count: int, durability: float | None = None) -> int:
        """Returns the amount that did not fit."""

    @abstractmethod
    def remove_item(self, item_id: str, count: int) -> bool: ...

    @abstractmethod
    def count(self, item_id: str) -> int: ...

    @abstractmethod
    def move(self, uid: str, x: int, y: int) -> bool: ...

    @abstractmethod
    def split(self, uid: str, amount: int): ...
