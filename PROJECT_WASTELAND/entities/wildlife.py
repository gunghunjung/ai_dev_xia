"""Passive animals: graze, and bolt away from the player or loud noises."""
import math

from entities.enemy import Enemy, IDLE, PATROL, FLEE


class Wildlife(Enemy):
    def think(self, session, dt: float):
        p = session.player
        if self.state == FLEE or p.dead:
            return
        d = self.dist_to(p)
        # Walking slowly lets a hunter get closer than sprinting does.
        if d < 3.2 or (p.sprinting and d < self.defn["sight"] * session.visibility_mult()):
            self.set_state(FLEE, 4.0)

    def hear(self, x: float, y: float, radius: float, kind: str):
        if self.dead:
            return
        if math.hypot(x - self.x, y - self.y) <= radius * self.defn.get("hearing", 1.0):
            self.set_state(FLEE, 4.0)

    def update(self, session, dt: float):
        if self.state not in (IDLE, PATROL, FLEE) and not self.dead and self.stun <= 0:
            self.set_state(FLEE, 4.0)
        super().update(session, dt)
