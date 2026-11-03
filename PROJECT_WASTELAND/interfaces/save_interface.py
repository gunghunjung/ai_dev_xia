"""Save service contract."""
from abc import ABC, abstractmethod


class ISaveService(ABC):
    @abstractmethod
    def save(self, session, slot: str) -> bool: ...

    @abstractmethod
    def load(self, session, slot: str) -> bool: ...

    @abstractmethod
    def exists(self, slot: str) -> bool: ...
