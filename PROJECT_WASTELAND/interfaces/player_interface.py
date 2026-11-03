"""Player persistence contract."""
from abc import ABC, abstractmethod


class IPlayerRepository(ABC):
    @abstractmethod
    def save_player(self, conn, player) -> None: ...

    @abstractmethod
    def load_player(self, conn, player) -> bool: ...
