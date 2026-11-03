"""Combat resolution contract. The client predicts effects; the service decides damage."""
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class DamageResult:
    amount: float
    killed: bool
    knockback: float
    stun: float
    bleed: bool = False
    infect: float = 0.0


class ICombatService(ABC):
    @abstractmethod
    def resolve_hit(self, attacker, target, weapon: dict, multiplier: float = 1.0) -> DamageResult: ...
