"""World service contract. Local implementation: world.world_manager.WorldManager."""
from abc import ABC, abstractmethod


class IWorldService(ABC):
    @abstractmethod
    def cell(self, x: int, y: int):
        """Return CellInfo for a world cell."""

    @abstractmethod
    def complete_dig_layer(self, x: int, y: int) -> list:
        """Authoritatively advance a cell one layer deeper. Returns found [(item_id, count)]."""

    @abstractmethod
    def objects_near(self, x: float, y: float, radius: float) -> list: ...

    @abstractmethod
    def place_structure(self, struct_type: str, x: float, y: float): ...

    @abstractmethod
    def remove_object(self, obj) -> None: ...
