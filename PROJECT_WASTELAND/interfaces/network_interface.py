"""Network contract for a future MMO conversion.

The single-player build uses ``LocalNetworkService`` which is authoritative and offline.
A future client implementation would forward intents (dig, attack, craft, move item) to a
server and apply confirmed results instead of deciding them locally.
"""
from abc import ABC, abstractmethod


class INetworkService(ABC):
    @abstractmethod
    def is_authoritative(self) -> bool: ...

    @abstractmethod
    def send_intent(self, kind: str, payload: dict) -> None: ...

    @abstractmethod
    def poll(self) -> list: ...


class LocalNetworkService(INetworkService):
    """Offline loopback: this process is the authority, intents are recorded for debugging."""

    def __init__(self, keep: int = 64):
        self.keep = keep
        self.history: list = []

    def is_authoritative(self) -> bool:
        return True

    def send_intent(self, kind: str, payload: dict) -> None:
        self.history.append((kind, payload))
        if len(self.history) > self.keep:
            del self.history[0]

    def poll(self) -> list:
        return []
