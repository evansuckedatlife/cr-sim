"""Casting a spell.

A spell card is not one mechanism but a choice between four, and which one a
card uses is the difference between how it plays:

*direct projectile*
    Fireball, Rocket, Snowball. A shot flies to the chosen point and detonates.
    It can be dodged by anything that leaves before it lands, which is why
    committing a Rocket at a moving push is a prediction.
*area effect*
    Zap, Freeze, Poison, Tornado. A cloud is placed and keeps touching whatever
    is inside it. Poison for eight seconds, Zap for one tick.
*projectile that becomes an area effect*
    Arrows, Lightning. The shot travels, and what it leaves is the actual
    payload.
*a delivery vehicle*
    Goblin Barrel and Royal Delivery are troops in a wrapper: the projectile's
    only job is to arrive somewhere and unpack.

Two of them are also **waved**: Arrows fires three volleys 200ms apart rather
than one, so a unit that leaves the area between volleys takes less than the
full amount. That is why Arrows' damage looks inconsistent in play, and why its
listed 122 is per volley rather than per cast.

Spells are cast at a *point*, never at a unit. Nothing about a spell is bound
to whoever was standing there when you pressed the button.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..data.cards import Card
from .constants import TickClock

__all__ = ["SpellPlan", "plan_spell"]


@dataclass(frozen=True, slots=True)
class SpellPlan:
    """What a cast actually puts on the board, resolved once at build time."""

    card: str
    #: A projectile launched at the target point, if any.
    projectile: str | None
    #: An area effect placed at the target point, if any. When a projectile is
    #: also present the area effect is created where the projectile lands.
    area_effect: str | None
    #: Units the cast deploys directly (Rage's bottle, Goblin Barrel's goblins).
    summon_character: str | None
    summon_count: int
    #: Volleys, and the gap between them. One volley is the normal case.
    waves: int
    wave_interval_ticks: int
    #: Radius the card itself advertises, used when the payload does not carry
    #: one of its own.
    radius: int

    @property
    def is_waved(self) -> bool:
        return self.waves > 1

    @property
    def does_nothing(self) -> bool:
        """A cast with no payload the engine can currently express.

        True for the cards whose whole behaviour lives in the ACTION graph --
        Graveyard, Clone, Mirror, Vines. Reported rather than silently ignored
        so it is obvious which spells are still inert.
        """
        return not (self.projectile or self.area_effect or self.summon_character)


def _int(value: Any, default: int = 0) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else default


def _str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def plan_spell(card: Card, clock: TickClock | None = None) -> SpellPlan:
    """Work out what casting ``card`` should produce.

    Read off the card row rather than guessed: the payload fields are on the
    spell itself, and following them is what keeps Goblin Barrel a troop
    delivery rather than a damage spell.
    """
    clock = clock or TickClock()
    row: Mapping[str, Any] = card.raw or {}

    summon = card.summon_character or _str(row.get("SummonCharacter"))
    return SpellPlan(
        card=card.name,
        projectile=card.projectile or _str(row.get("Projectile")),
        area_effect=card.area_effect_object or _str(row.get("AreaEffectObject")),
        summon_character=summon,
        summon_count=max(1, _int(row.get("SummonNumber"), card.summon_count or 1)),
        waves=max(1, _int(row.get("ProjectileWaves"), 1)),
        wave_interval_ticks=clock.ticks(row.get("ProjectileWaveInterval")),
        radius=_int(row.get("Radius")),
    )
