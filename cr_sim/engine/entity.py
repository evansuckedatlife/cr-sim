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

from .buffs import apply_delta

if TYPE_CHECKING:  # pragma: no cover
    from .specs import UnitSpec

__all__ = [
    "Team", "EntityKind", "EntityState", "Entity", "next_entity_id",
    "entity_id_cursor", "restore_entity_ids",
]


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


def entity_id_cursor() -> int:
    """Where the id counter currently stands.

    Paired with :func:`restore_entity_ids` so speculative work -- a branch
    played forward and thrown away -- can hand back the ids it burned. The
    counter is module-level, so without this a projection would consume ids
    the live battle was going to use, and the match would run differently for
    having been *asked about*. Replay determinism is the whole foundation
    here, and that would quietly undermine it.
    """
    return _next_id


def restore_entity_ids(cursor: int) -> None:
    """Wind the id counter back to ``cursor``.

    Only sound when everything allocated since is unreachable; reviving an id
    that a live entity still holds would collide in ``Battle._by_id_map``.
    """
    global _next_id
    _next_id = cursor


def reset_entity_ids() -> None:
    """Reset the id counter. Called when a battle starts so runs are comparable."""
    global _next_id
    _next_id = 0


#: Every slot a class carries, base classes first, collected once per class
#: and cached -- ``__slots__`` is per-class, not inherited automatically, so
#: a subclass (``Projectile``, ``AreaEffect``, ...) needs its own plus every
#: ancestor's. Used by :meth:`Entity.__deepcopy__`, which has to work for all
#: of them, not just the base class.
_slots_cache: dict[type, tuple[str, ...]] = {}


def _slots_for(cls: type) -> tuple[str, ...]:
    found = _slots_cache.get(cls)
    if found is None:
        collected: list[str] = []
        for klass in reversed(cls.__mro__):
            collected.extend(getattr(klass, "__slots__", ()))
        found = tuple(collected)
        _slots_cache[cls] = found
    return found


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
        "buffs",
        "is_clone",
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
        #: Timed status effects. Created lazily -- most entities never carry
        #: one, and a battle holds hundreds of entities.
        self.buffs = None
        #: Produced by a Clone spell. Kept because the game's own
        #: CLONE_CLONED_UNITS global is False: a second Clone cast over the
        #: first one's output must not double it again, or two 3-elixir spells
        #: would quadruple a push.
        self.is_clone = False

    # ------------------------------------------------------------- lifecycle

    @property
    def is_deploying(self) -> bool:
        return self.deploy_ticks_left > 0

    @property
    def is_alive(self) -> bool:
        return not self.dead

    @property
    def is_targetable(self) -> bool:
        """A unit cannot be hit while still deploying, and a bomb never can."""
        if self.dead or self.deploy_ticks_left > 0:
            return False
        # A live bomb is scenery with a countdown. Nothing in the game can
        # shoot a Giant Skeleton's bomb out of the air, and letting anything
        # try would also let it be killed early, cancelling the blast.
        return self.spec is None or not self.spec.is_fuse

    @property
    def is_acquirable(self) -> bool:
        """Whether an enemy can *choose* this entity as its target.

        Stricter than :attr:`is_targetable`, and the gap between the two is
        deliberate: an invisible Royal Ghost cannot be picked as a target, but
        a Fireball dropped on the tile it happens to be standing on still
        burns it. Splash and area effects therefore keep testing
        ``is_targetable``, and only target *selection* tests this.

        Collapsing the two would make invisibility a blanket immunity, which
        would let a Royal Ghost walk through Poison untouched.
        """
        if not self.is_targetable:
            return False
        return self.buffs is None or not self.buffs.is_invisible()

    def clone(self) -> "Entity":
        """A copy carrying its own mutable state, sharing its spec.

        Entities refer to each other by ``target_id`` rather than by object,
        which is what makes this a flat copy: nothing here points at another
        entity, so a cloned battle needs no reference fixing. ``spec`` is
        immutable and shared; ``buffs`` is not, and gets its own copy.
        """
        copy = type(self).__new__(type(self))
        # Every slot down the MRO, and the concrete class rather than the base:
        # iterating Entity.__slots__ and constructing an Entity silently dropped
        # whatever a Projectile or AreaEffect adds. Safe only for as long as
        # nothing cloned one, which is not a property worth relying on.
        for slot in _slots_for(type(self)):
            setattr(copy, slot, getattr(self, slot))
        if self.buffs is not None:
            copy.buffs = self.buffs.clone()
        return copy

    def __deepcopy__(self, memo: dict) -> "Entity":
        """Fast path for ``copy.deepcopy`` -- what :meth:`Battle.clone` uses.

        The generic deepcopy machinery reconstructs every entity through
        ``__reduce_ex__``, which for a ``__slots__`` class means pickling
        each slot into a state dict and then deep-copying *that dict*,
        recursing through :mod:`copy`'s dispatch for every attribute even
        when it is a plain ``int``. Measured on a mid-match clone this was
        the largest single cost: thousands of entities' worth of attribute
        traversal for objects with no internal graph to speak of.

        Unlike :meth:`clone`, this covers every concrete subclass
        (``Projectile``, ``RollingProjectile``, ``AreaEffect``), each of
        which adds its own ``__slots__`` on top of :class:`Entity`'s --
        ``clone`` only ever runs on troops and buildings (the Clone spell's
        targets), so it does not need to.
        """
        cls = type(self)
        copy_obj = cls.__new__(cls)
        for slot in _slots_for(cls):
            setattr(copy_obj, slot, getattr(self, slot))
        if self.buffs is not None:
            copy_obj.buffs = self.buffs.clone()
        # The one mutable container slot outside `buffs`: RollingProjectile
        # and AreaEffect both track ids they have already struck, mutated
        # in place with `.add()`. Aliasing it would let a branch's hits mark
        # the origin's projectile as having already struck someone, or vice
        # versa. `spec`/`pspec`/`aspec` are deliberately left aliased above --
        # they are immutable and already shared via Battle._SHARED.
        struck = getattr(self, "struck", None)
        if struck is not None:
            copy_obj.struck = set(struck)
        return copy_obj

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
        """Move to a new state, unless this entity is already dying.

        Death is terminal and must survive the rest of the tick. Without this
        guard a unit that takes a fatal hit early in a tick can overwrite its
        own DYING state when it acts later in the same tick -- the death sweep
        then never sees it and it fights on forever at zero hitpoints. Ordering
        hazards like that are why deaths are swept at a fixed point rather than
        applied immediately.
        """
        if self.state in (EntityState.DYING, EntityState.DEAD):
            return
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
        if self.buffs is not None:
            # Reduction is a property of whoever is being hit, so it belongs at
            # the one point every source of damage passes through -- a Monk's
            # shield has to blunt a Fireball and a sword swing alike. Applying
            # it per call site would mean five places to keep in step.
            taken = self.buffs.damage_taken_multiplier()
            if taken:
                amount = apply_delta(amount, taken)
                if amount <= 0:
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
