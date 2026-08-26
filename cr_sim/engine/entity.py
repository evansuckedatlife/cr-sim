"""Battlefield entities.

Everything that occupies a position and can be interacted with is an
:class:`Entity`: troops, buildings, towers, projectiles and area effects. They
share an id, an owner, a position in subtiles and a lifecycle, which is what the
tick loop iterates over.

Two performance choices, made here because they are hard to retrofit:

*   ``__slots__`` everywhere. A three-minute battle is 10,800 ticks and can hold
    dozens of entities, so this is millions of attribute reads per battle; slots
    remove the per-instance dict and cut both memory and lookup cost.
*   Entities are never removed from the list mid-tick. Death sets a flag and the
    sweep happens at a defined point in the tick, so iteration order stays
    stable and every phase within a tick sees the same population. Mutating the
    list while phases iterate it is the classic source of order-dependent,
    irreproducible bugs.
"""

from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .specs import UnitSpec

__all__ = ["Team", "EntityKind", "EntityState", "Entity", "next_entity_id"]


class Team(IntEnum):
    """Which side an entity belongs to. Blue defends the low-y end."""

    BLUE = 0
    RED = 1

    @property
    def opponent(self) -> "Team":
        return Team.RED if self is Team.BLUE else Team.BLUE


class EntityKind(IntEnum):
    TROOP = 0
    BUILDING = 1
    TOWER = 2
    PROJECTILE = 3
    AREA_EFFECT = 4


class EntityState(IntEnum):
    """Lifecycle of an entity within a battle.

    ``DEPLOYING`` is a real state, not a formality: a card spends its deploy
    time on the board untargetable and unable to act, and the length of that
    window is what makes surprise placements punishable.
    """

    DEPLOYING = 0
    IDLE = 1
    MOVING = 2
    ATTACKING = 3
    DYING = 4
    DEAD = 5


_next_id = 0


def next_entity_id() -> int:
    """Monotonic entity id.

    Ids are assigned in spawn order and never reused, so they give every phase a
    stable, deterministic tiebreak when two entities are otherwise equivalent --
    which target to pick, which of two simultaneous deaths resolves first.
    """
    global _next_id
    _next_id += 1
    return _next_id


def reset_entity_ids() -> None:
    """Reset the id counter. Called when a battle starts so runs are comparable."""
    global _next_id
    _next_id = 0


class Entity:
    """Base for anything that exists on the battlefield."""

    __slots__ = (
        "id",
        "kind",
        "team",
        "spec",
        "x",
        "y",
        "hitpoints",
        "max_hitpoints",
        "shield",
        "state",
        "state_ticks",
        "spawn_tick",
        "deploy_ticks_left",
        "target_id",
        "collision_radius",
        "mass",
        "flying",
        "dead",
        "lifetime_left",
    )

    def __init__(
        self,
        *,
        kind: EntityKind,
        team: Team,
        x: int,
        y: int,
        hitpoints: int,
        spec: "UnitSpec | None" = None,
        spawn_tick: int = 0,
        deploy_ticks: int = 0,
        collision_radius: int = 0,
        mass: int = 0,
        flying: bool = False,
        shield: int = 0,
        lifetime_ticks: int = 0,
    ) -> None:
        self.id = next_entity_id()
        self.kind = kind
        self.team = team
        self.spec = spec
        self.x = x
        self.y = y
        self.hitpoints = hitpoints
        self.max_hitpoints = hitpoints
        self.shield = shield
        self.state = EntityState.DEPLOYING if deploy_ticks > 0 else EntityState.IDLE
        self.state_ticks = 0
        self.spawn_tick = spawn_tick
        self.deploy_ticks_left = deploy_ticks
        self.target_id = 0
        self.collision_radius = collision_radius
        self.mass = mass
        self.flying = flying
        self.dead = False
        #: Ticks until this entity expires on its own. Spawned buildings are
        #: temporary -- a Cannon lives 30 seconds whether or not anything
        #: attacks it -- so the timer is part of the entity, not of combat.
        #: Zero means "permanent" (troops, towers).
        self.lifetime_left = lifetime_ticks

    # ------------------------------------------------------------- lifecycle

    @property
    def is_deploying(self) -> bool:
        return self.deploy_ticks_left > 0

    @property
    def is_alive(self) -> bool:
        return not self.dead

    @property
    def is_targetable(self) -> bool:
        """A unit cannot be hit while still deploying."""
        return not self.dead and self.deploy_ticks_left <= 0

    def tick_deploy(self) -> bool:
        """Advance the deploy timer. Returns True on the tick it completes."""
        if self.deploy_ticks_left <= 0:
            return False
        self.deploy_ticks_left -= 1
        if self.deploy_ticks_left == 0:
            self.state = EntityState.IDLE
            self.state_ticks = 0
            return True
        return False

    def tick_lifetime(self) -> bool:
        """Count down an expiry timer. Returns True on the tick it runs out."""
        if self.lifetime_left <= 0:
            return False
        self.lifetime_left -= 1
        if self.lifetime_left == 0:
            self.kill()
            return True
        return False

    def set_state(self, state: EntityState) -> None:
        if self.state is not state:
            self.state = state
            self.state_ticks = 0

    # ---------------------------------------------------------------- damage

    def apply_damage(self, amount: int) -> int:
        """Deal ``amount``, shield first. Returns damage actually absorbed.

        Shields (Dark Prince, Guards, Royal Recruits) soak whole hits before
        hitpoints are touched; overflow does **not** carry through to the body in
        Clash Royale, which is why a big hit into a small shield is wasted.
        """
        if self.dead or amount <= 0:
            return 0
        dealt = 0
        if self.shield > 0:
            absorbed = min(self.shield, amount)
            self.shield -= absorbed
            dealt += absorbed
            # Remaining damage is discarded, not carried into hitpoints.
            amount = 0
        if amount > 0:
            absorbed = min(self.hitpoints, amount)
            self.hitpoints -= absorbed
            dealt += absorbed
        if self.hitpoints <= 0:
            self.hitpoints = 0
            self.state = EntityState.DYING
        return dealt

    def heal(self, amount: int) -> int:
        if self.dead or amount <= 0:
            return 0
        before = self.hitpoints
        self.hitpoints = min(self.max_hitpoints, self.hitpoints + amount)
        return self.hitpoints - before

    def kill(self) -> None:
        self.hitpoints = 0
        self.state = EntityState.DYING

    # ------------------------------------------------------------- debugging

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        name = getattr(self.spec, "name", self.kind.name)
        return (
            f"<{name} #{self.id} {self.team.name} "
            f"({self.x},{self.y}) hp={self.hitpoints}/{self.max_hitpoints} "
            f"{self.state.name}>"
        )
