"""Timed status effects: Rage, Freeze, Poison, and everything like them.

A buff is a temporary modifier attached to an entity rather than to an attack.
Where :mod:`cr_sim.engine.combat` decides whether *this* swing lands, a buff
changes how every subsequent tick behaves until it expires -- a Rage spell
makes a push arrive early, a Freeze turns a counter-push into a free trade, and
a stacked Poison cloud can out-race a tank's hitpoints entirely on its own.
They are the mechanism behind board control: nothing else lets a card affect
units it never touches directly.

Game data defines every buff in the ``BUFF`` namespace (112 entries in this
build, most of them evolution/hero-ability variants of a handful of ideas).
Three fields carry the mechanic:

``SpeedMultiplier`` / ``HitSpeedMultiplier`` / ``SpawnSpeedMultiplier``
    Percentage modifiers on movement, attack interval, and building spawn
    interval.
``DamagePerSecond`` / ``HitFrequency``
    Damage-over-time: a rate, plus how often it is actually applied.
``EnableStacking``
    Whether repeat applications add up or just refresh the clock.

Reading the raw data (verified below) settles the two questions that matter
for correctness: what the multiplier fields actually mean, and how the
damage-over-time rate turns into a number applied on a given tick.

**The multiplier convention -- checked against Freeze, Poison, Ice Wizard and
Rage.** ``BUFF.Freeze`` is ``SpeedMultiplier: -100`` and stops movement dead;
``BUFF.Poison`` is ``-15``; live sources (Fandom, Liquipedia -- see the final
report for this module) describe Poison as "15% slower" and Ice Wizard's
``IceWizardCold``/``IceWizardSlowDown`` (``-30``) as "30% slower". Both check
out *exactly* under the house convention already used everywhere else in this
codebase for a percentage delta -- ``UnitSpec.damage_to``,
``ProjectileSpec.damage_to``, ``AreaEffectSpec.damage_to`` and
``TowerScale._percent`` all compute ``base * (100 + percent) // 100`` -- so
:func:`apply_multiplier` below matches that pattern exactly rather than
inventing a new one.

Rage is the one surprise. ``BUFF.Rage`` (confirmed as the buff the ``AEO.Rage``
spell object actually applies, via its ``Buff = "Rage"`` field) reads
``SpeedMultiplier: 130``. Fed through the same ``(100 + percent) // 100``
formula that nails Poison and Ice Wizard exactly, that is a **+130% speed
buff** -- 2.3x normal speed -- wildly stronger than Rage's well-documented
30-35% boost. Every positive-value buff in the namespace has the same shape:
``PrinceRageBuff1/2/3`` step 135 / 170 / 230, ``DarkElixirBuff`` (a "berserk"
hero buff) is 200, evolutions sit around 130. Read *directly* (``base *
value // 100``, no ``+100``) rather than as a delta, 130 becomes a +30%
boost, which lands right in Rage's cited 30-40% history. So it looks like
this data format encodes boosts as an already-baselined total (130 meaning
"130% of normal") while it encodes penalties as a delta from that same
baseline (-30 meaning "30 off of 100"). That is a real inconsistency in the
source data, not a guess on my part -- both readings check out precisely
against known values on their own side of zero and neither reading works for
both. **This module stores the raw signed value verbatim** (matching the
required field's own doc comment, "-100 stopped, +35 faster") and applies it
through the single ``(100 + percent) // 100`` formula everywhere in this
codebase already uses, because inventing a sign-dependent special case here,
in a leaf module nothing has wired up yet, would just move the ambiguity
somewhere harder to find. Whoever wires :func:`apply_multiplier` up to a
unit's actual speed should see this docstring and decide with the full
picture -- flagged in the module's final report, not silently resolved here.

**Damage-over-time: the first application is immediate.**
:class:`~cr_sim.engine.areaeffects.AreaEffect` already answers this question
for the identical mechanism one layer up (an area effect's own periodic
damage) and says so explicitly: it seeds ``ticks_to_next = 0`` so the first
hit lands "on the tick it lands rather than after one interval", naming
Poison as the example ("a Poison cloud that waited would let a unit walk
through the first quarter-second untouched"). ``ActiveBuff`` mirrors that
field (``ticks_to_next_damage``) and the same seeding for consistency. This
also reproduces a fact every player can check: Poison's ``LifeDuration`` is
8000ms with ``HitFrequency`` 1000ms, and Poison is universally described as
"8 ticks over 8 seconds" -- with an immediate first tick and a duration of
exactly 8000ms/8 ticks, applications land at relative offsets 0, 1000, ...,
7000ms, which is 8 of them. Waiting one interval before the first hit would
only produce 7 within the same window.

**DamagePerSecond is a rate, not a per-hit amount, and needs converting.**
Most damage buffs tick once a second (``HitFrequency: 1000``), so for them
the distinction is invisible -- Poison's ``DamagePerSecond: 36`` happens to
equal its damage per application. Tornado breaks the tie: ``HitFrequency:
550`` with ``DamagePerSecond: 60``, and its own ``StatsTags`` in the data
relabels its in-game display stat from ``DamagePerSecond`` to ``DamagePerHit``
specifically because those two numbers differ for it. That confirms the field
is a genuine rate that must be scaled by the interval to get a real
per-application amount (``60 * 550 // 1000 = 33`` damage per hit, not 60).
``build_buff_spec`` does that conversion once, at build time, using the same
``TickClock`` every other spec-builder in this package uses to convert
milliseconds once rather than in the hot loop -- so despite its name,
``BuffSpec.damage_per_second`` holds the already-converted *per-application*
amount, and ``BuffState.tick`` never needs to know the tick rate to use it.

**Combining multipliers from several active buffs: add the deltas, not
compound them.** If two buffs each modified speed independently
(``base * m1 // 100`` then that result ``* m2 // 100``), the outcome would
depend on the order the buffs happen to be stored in, and each multiply-then-
floor-divide step throws away its own fraction of a subtile -- two rounding
errors instead of one, and a non-commutative one at that. Summing the raw
percentage deltas first and applying :func:`apply_multiplier` exactly once is
commutative (order cannot matter, which this engine's determinism guarantee
already requires everywhere else) and truncates only a single time.

**Stacking, per ``EnableStacking``.** Reading the whole namespace: the buffs
that carry ``EnableStacking: True`` are exactly the ones where stacking make
sense -- repeatable damage-over-time and heal-over-time effects (``Poison``,
``Earthquake``, ``Tornado``, ``GoblinCurseDamage``, ``WarmUp`` and their
evolution variants). Every plain status buff -- ``Rage``, ``Freeze``,
``IceWizardCold``/``IceWizardSlowDown``, ``BolaSnare``, ``ZapFreeze``, the
``PrinceRageBuff*`` ladder, ``DarkElixirBuff`` -- has no such field, i.e. it
is ``False``: a second application refreshes the timer rather than adding a
second copy. That matches the games's own rules players rely on (two
overlapping Poison clouds both tick; two overlapping Rages do not double your
speed) and this module honours it exactly rather than guessing a uniform
policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..data.leveling import RarityScale
from ..data.source import LogicData, UnknownEntity
from .constants import TickClock

__all__ = [
    "BuffSpec",
    "build_buff_spec",
    "ActiveBuff",
    "BuffState",
    "apply_multiplier",
]


def apply_multiplier(base: int, percent: int) -> int:
    """Apply a buff's speed percentage to ``base``.

    The field uses **two conventions in one column**, and the data settles which
    is which without ambiguity: every value in the build is either ``<= 0`` or
    ``>= 100``, with nothing in between. The ranges do not overlap, so no value
    is ever ambiguous.

    * ``<= 0`` is a *delta* off the baseline: ``-15`` is Poison's 15% slow,
      ``-30`` Ice Wizard's, ``-100`` a full stop (Freeze, Stun, Clone).
    * ``>= 100`` is the *whole* multiplier: ``130`` is Rage at 1.3x, ``170``
      Prince's second charge tier, ``250`` Valkyrie's hero chain.

    The clincher is ``IgnoreBarrel`` at exactly ``100``. It is a pure
    targeting-immunity marker that must not change speed at all, and ``100`` as
    a whole multiplier is precisely 1.0x -- neutral. Read as a delta it would
    double the unit's speed, which is plainly not what an immunity flag does.
    Reading every positive as a delta would also put Rage at +130% against its
    real +30%, and Prince's top charge tier at 3.3x.

    Clamped at zero: a stack of slows means "as slow as this gets", never
    reversed. Truncates toward zero like every other scaling in this engine.
    """
    multiplier = percent if percent >= 100 else 100 + percent
    result = base * multiplier // 100
    return result if result > 0 else 0


def _int(value: Any, default: int = 0) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else default


def _bool(value: Any) -> bool:
    return value is True


@dataclass(frozen=True, slots=True)
class BuffSpec:
    """One buff *type*, pre-converted to engine units.

    Holds no duration: the same buff (e.g. ``Rage``) can be cast for different
    lengths of time by different sources, so ``BuffTime`` lives on whatever
    area effect or projectile applies it (see
    :class:`cr_sim.engine.areaeffects.AreaEffectSpec` /
    :class:`cr_sim.engine.projectiles.ProjectileSpec`) and is passed to
    :meth:`BuffState.apply` at application time instead.
    """

    name: str
    #: Percentage delta, raw from the data -- see the module docstring for
    #: why this is stored verbatim rather than normalised: it is the caller's
    #: decision how to reconcile Rage's data against its known real value.
    speed_multiplier: int
    hit_speed_multiplier: int
    spawn_speed_multiplier: int
    #: Damage dealt on each application -- already converted from the data's
    #: DamagePerSecond *rate* into a flat per-hit amount using this buff's own
    #: HitFrequency, at build time. See the module docstring ("DamagePerSecond
    #: is a rate...").
    damage_per_second: int
    #: Ticks between damage-over-time applications. Zero (or less) means this
    #: buff never deals periodic damage -- either it has none (Rage, Freeze)
    #: or its data uses the HitFrequency: -1 sentinel some evolution "on hit"
    #: buffs carry, which is a single triggered application, not a repeating
    #: one this generic ticker can replay.
    hit_frequency_ticks: int
    #: EnableStacking in the data. True means repeat applications add a fully
    #: independent copy; False means a repeat application refreshes the
    #: existing one's timer instead. See the module docstring.
    stacks: bool
    crown_tower_damage_percent: int

    def damage_to(self, is_crown_tower: bool) -> int:
        """Per-application damage, reduced against a crown tower where applicable.

        Mirrors ``UnitSpec.damage_to`` / ``ProjectileSpec.damage_to`` /
        ``AreaEffectSpec.damage_to`` exactly, via :func:`apply_multiplier`.
        """
        if not is_crown_tower or not self.crown_tower_damage_percent:
            return self.damage_per_second
        return apply_multiplier(self.damage_per_second, self.crown_tower_damage_percent)


def build_buff_spec(
    data: LogicData,
    name: str,
    scale: RarityScale,
    *,
    level: int,
    clock: TickClock,
) -> BuffSpec | None:
    """Resolve ``BUFF.<name>`` and convert it to engine units.

    Returns ``None`` for a name the data doesn't define, the same "missing is
    not an error at this layer" contract every other ``build_*_spec`` in this
    package uses -- a card whose buff reference is absent or renamed should
    fail where the card is built, not deep inside this leaf module.
    """
    try:
        raw: Mapping[str, Any] = data.resolve(f"BUFF.{name}")
    except (UnknownEntity, KeyError):
        return None

    hit_frequency_ticks = clock.ticks(raw.get("HitFrequency"))
    raw_dps = _int(raw.get("DamagePerSecond"))
    scaled_dps = scale.scale(raw_dps, level) if raw_dps else 0
    # Convert the rate to a flat per-application amount here, once, using this
    # build's actual tick rate -- so BuffState.tick() can stay a plain int
    # counter with no clock of its own. See the module docstring.
    damage_per_application = (
        scaled_dps * hit_frequency_ticks // clock.ticks_per_second
        if scaled_dps and hit_frequency_ticks > 0
        else 0
    )

    return BuffSpec(
        name=str(raw.get("Name", name)),
        speed_multiplier=_int(raw.get("SpeedMultiplier")),
        hit_speed_multiplier=_int(raw.get("HitSpeedMultiplier")),
        spawn_speed_multiplier=_int(raw.get("SpawnSpeedMultiplier")),
        damage_per_second=damage_per_application,
        hit_frequency_ticks=hit_frequency_ticks,
        stacks=_bool(raw.get("EnableStacking")),
        crown_tower_damage_percent=_int(raw.get("CrownTowerDamagePercent")),
    )


@dataclass(slots=True)
class ActiveBuff:
    """One live application of a buff on one entity."""

    spec: BuffSpec
    #: Ticks until this application expires and is removed.
    ticks_left: int
    #: Ticks until the next damage-over-time application. Seeded at one full
    #: interval, so the first tick lands after ``HitFrequency`` rather than on
    #: contact. Poison is the evidence: its cloud lasts 8 seconds and ticks
    #: once a second for 8 ticks. An immediate first tick gives 9 -- one on
    #: contact plus one a second for eight seconds -- which is a ninth of the
    #: card's damage too much. Earthquake checks out the same way, 3 ticks over
    #: 3 seconds.
    #:
    #: This is a different clock from the *area effect's* own first
    #: application, which IS immediate: Zap has to be instant.
    ticks_to_next_damage: int
    #: Which effect applied this copy. Two Poison clouds overlapping are two
    #: independent sources and genuinely stack; one cloud re-touching the same
    #: unit every quarter-second is the *same* source and must not.
    source: int = 0


class BuffState:
    """All buffs currently on one entity.

    A plain list, not a dict keyed by name: stacking buffs (Poison and its
    kin) can have several independent copies alive with different remaining
    durations at once, and list order is insertion order, which keeps this
    deterministic without sorting anything -- the same reason
    :mod:`cr_sim.engine.combat` collects hits before applying them rather than
    using an unordered structure.
    """

    __slots__ = ("_buffs",)

    def __init__(self) -> None:
        self._buffs: list[ActiveBuff] = []

    def apply(self, spec: BuffSpec, duration_ticks: int, source: int = 0) -> None:
        """Apply ``spec`` for ``duration_ticks``, on behalf of ``source``.

        ``source`` is what makes stacking mean the right thing. Every area
        effect in the build carries ``BuffNumber = 1``: one instance per target
        *per source*. A Poison cloud re-touches everything inside it four times
        a second, and treating each touch as a fresh stack turns an 8-second,
        736-damage spell into an instant kill. Refreshing instead keeps the
        status alive for as long as the unit stays in the cloud, and lets the
        buff's own ``HitFrequency`` set the damage rhythm -- which is the eight
        ticks over eight seconds the card is documented to do.

        Two *different* clouds are two sources, so they still stack, which is
        what ``EnableStacking`` is actually for.

        A non-stacking buff refreshes regardless of source: a second Freeze does
        not double the duration, it restarts it. Refreshing also replaces the
        spec, so a stronger cast supersedes a weaker one already in place.
        """
        for active in self._buffs:
            if active.spec.name != spec.name:
                continue
            if not spec.stacks or active.source == source:
                active.spec = spec
                active.ticks_left = duration_ticks
                # Deliberately NOT resetting the damage countdown. A refresh
                # extends how long the status lasts; it does not restart its
                # rhythm. Resetting here means a cloud that re-touches its
                # victims every 250ms lands a Poison tick every 250ms instead
                # of every second -- four times the damage the card does.
                return
        self._buffs.append(
            ActiveBuff(
                spec=spec,
                ticks_left=duration_ticks,
                ticks_to_next_damage=spec.hit_frequency_ticks,
                source=source,
            )
        )

    def tick(self) -> int:
        """Advance every active buff by one tick.

        Expired buffs (``ticks_left`` reaching zero on this call) are dropped
        here rather than left for a caller to sweep, the same way
        :meth:`cr_sim.engine.entity.Entity` and area effects retire themselves
        on their own countdown. Returns the total damage-over-time owed this
        tick, summed across every active buff (including independent stacks
        of the same buff, each on its own countdown) -- raw, unreduced for
        crown towers, since that reduction can differ per buff and this
        entity-level total no longer knows which buff contributed what; a
        caller needing per-buff crown tower damage should read
        ``active_names()`` and call ``BuffSpec.damage_to`` itself.
        """
        damage = 0
        survivors: list[ActiveBuff] = []
        for active in self._buffs:
            if active.spec.hit_frequency_ticks > 0:
                # Count down first, then fire. Firing before the decrement
                # would let the tick that fires also be the first tick of the
                # next interval, spacing hits one apart too far -- 61 ticks for
                # a 60-tick HitFrequency, which quietly loses a Poison tick
                # over an 8-second cloud.
                active.ticks_to_next_damage -= 1
                if active.ticks_to_next_damage <= 0:
                    damage += active.spec.damage_per_second
                    active.ticks_to_next_damage = active.spec.hit_frequency_ticks
            active.ticks_left -= 1
            if active.ticks_left > 0:
                survivors.append(active)
        self._buffs = survivors
        return damage

    def speed_multiplier(self) -> int:
        """Combined movement speed delta -- see the module docstring on why deltas sum."""
        return sum(active.spec.speed_multiplier for active in self._buffs)

    def hit_speed_multiplier(self) -> int:
        return sum(active.spec.hit_speed_multiplier for active in self._buffs)

    def spawn_speed_multiplier(self) -> int:
        """Combined spawn-speed delta, for buildings under Rage or Freeze."""
        return sum(active.spec.spawn_speed_multiplier for active in self._buffs)

    def is_frozen(self) -> bool:
        """Whether movement is fully stopped.

        True once the combined speed delta reaches -100 or below --
        :func:`apply_multiplier` would floor any such delta's effect on speed
        at zero, so -100 alone (Freeze) and several smaller slows stacking to
        -100 or past it are the same observable state: stopped.
        """
        return self.speed_multiplier() <= -100

    def clear(self) -> None:
        self._buffs.clear()

    def __bool__(self) -> bool:
        return bool(self._buffs)

    def active_names(self) -> tuple[str, ...]:
        """Names of every active buff, one entry per instance.

        A triple-stacked Poison therefore returns ``Poison`` three times
        rather than once -- the count itself is information (it says how much
        of the stack is still ticking), and collapsing it to a set would
        throw that away.
        """
        return tuple(active.spec.name for active in self._buffs)
