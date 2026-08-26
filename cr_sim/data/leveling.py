"""Card level scaling.

Every scalable stat in Clash Royale is derived from a single base value by a
lookup in a shared multiplier table -- there are no per-level stat arrays in the
game files, so this table *is* the progression.

``rarities.csv`` supplies two things per rarity:

``PowerLevelMultiplier``
    The power ladder ``(110, 121, 133, 146, 160, 176, 193, 212, 233, 256, ...)``.
    Every rarity carries an *identical prefix* of one shared ladder, each
    truncated to roughly its own level count -- so the per-rarity copies are
    not independently usable.  A Champion sits at power index 10 at level 1 and
    reaches 15 at level 6, well past the 9 entries stored on its own row, so the
    longest available copy is used as the single shared ladder.
``RelativeLevel``
    The rarity's starting offset: Common 0, Rare 2, Epic 5, Legendary 8,
    Champion 10.  A Rare at level 1 therefore has the same power as a Common at
    level 3, which is the familiar in-game rule.

So the index into the ladder is ``RelativeLevel + level - 1``, with index 0
meaning "unscaled" (an implicit 100).  Values truncate toward zero.

Verified against live values: Knight (Common, base 690 HP / 79 damage) at level
11 -> index 10 -> multiplier 256 -> 1766 HP and 202 damage, which is exactly
what the game shows at tournament standard.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .source import LogicData

__all__ = ["LevelTable", "RarityScale", "SCALED_STATS", "build_level_table"]

#: Stats that scale with card level.  Everything else (speed, range, hit speed,
#: radii, deploy time) is level-invariant in Clash Royale.
SCALED_STATS = frozenset(
    {
        "Hitpoints",
        "Damage",
        "DamageSpecial",
        "DeathDamage",
        "AreaDamage",
        "ShieldHitpoints",
        "DashDamage",
        "DashingDamage",
        "SpawnDamage",
        "VariableDamage2",
        "VariableDamage3",
        "ReflectedAttackDamage",
        "HealPerSecond",
        "InstantHeal",
        "InstantDamage",
        "AttackStateDamage",
    }
)


@dataclass(frozen=True, slots=True)
class RarityScale:
    """One rarity's slice of the shared power ladder."""

    name: str
    level_count: int
    relative_level: int
    tournament_level_index: int
    #: Multipliers indexed by *absolute power index*; index 0 is the implicit 100.
    multipliers: tuple[int, ...]

    @property
    def tournament_level(self) -> int:
        """The internal level this rarity sits at under tournament standard."""
        return self.tournament_level_index + 1

    def power_index(self, level: int) -> int:
        return self.relative_level + level - 1

    # -- internal vs displayed levels -------------------------------------
    #
    # The game shows every card on one 1..15 scale regardless of rarity, so a
    # P.E.K.K.A the UI calls "level 11" is internally an Epic at level 6.  The
    # displayed number is just the power index plus one, which is why every
    # rarity's tournament standard displays as 11.

    def display_level(self, level: int) -> int:
        """Internal level -> the level the game shows the player."""
        return self.power_index(level) + 1

    def internal_level(self, display_level: int) -> int:
        """The level the game shows -> this rarity's internal level."""
        return display_level - self.relative_level

    def scale_display(self, base: int, display_level: int) -> int:
        return self.scale(base, self.internal_level(display_level))

    def multiplier(self, level: int) -> int:
        index = self.power_index(level)
        if index < 0:
            raise ValueError(f"{self.name} level {level} is below the ladder")
        if index >= len(self.multipliers):
            raise ValueError(
                f"{self.name} level {level} (power index {index}) exceeds the "
                f"{len(self.multipliers) - 1}-entry ladder"
            )
        return self.multipliers[index]

    def scale(self, base: int, level: int) -> int:
        """Scale ``base`` to ``level``, truncating toward zero as the game does."""
        return base * self.multiplier(level) // 100


@dataclass(frozen=True, slots=True)
class LevelTable:
    """All rarities' scaling, keyed by rarity name."""

    rarities: Mapping[str, RarityScale]

    def __getitem__(self, rarity: str) -> RarityScale:
        return self.rarities[rarity]

    def get(self, rarity: str | None, default: str = "Common") -> RarityScale:
        if rarity and rarity in self.rarities:
            return self.rarities[rarity]
        return self.rarities[default]

    def scale(self, base: int, rarity: str | None, level: int) -> int:
        return self.get(rarity).scale(base, level)

    def scale_display(self, base: int, rarity: str | None, display_level: int) -> int:
        """Scale using the level the game displays, which is rarity-independent."""
        return self.get(rarity).scale_display(base, display_level)

    def tournament_level(self, rarity: str | None) -> int:
        return self.get(rarity).tournament_level


def build_level_table(data: LogicData) -> LevelTable:
    """Read ``rarities.csv`` into a :class:`LevelTable`."""
    table = data.tables.get("rarities")
    if table is None:
        raise KeyError("rarities.csv is missing from this build")

    ladders = {r.name: tuple(r.array("PowerLevelMultiplier")) for r in table}
    ladders = {name: values for name, values in ladders.items() if values}
    if not ladders:
        raise ValueError("rarities.csv contained no PowerLevelMultiplier ladders")

    # A rarity's own ladder is stored only as far as roughly its own level count,
    # which is short of where the high-offset rarities actually reach: a Champion
    # starts at power index 10 and hits 15 at level 6, but only 9 entries sit on
    # its row.  The longest ladder in the file fills the tail.
    #
    # These are *not* always the same numbers.  In the 2023 build Rare diverges
    # from Common at index 13 (340 vs 339), so a rarity's own values always win
    # where it has them; the longest ladder is only an extension.
    longest = max(ladders.values(), key=len)

    rarities: dict[str, RarityScale] = {}
    for record in table:
        own = ladders.get(record.name)
        if not own:
            continue
        extended = own + longest[len(own) :]
        rarities[record.name] = RarityScale(
            name=record.name,
            level_count=record.scalar("LevelCount", len(own)),
            relative_level=record.scalar("RelativeLevel", 0),
            tournament_level_index=record.scalar("TournamentLevelIndex", 0),
            # Index 0 is "no scaling"; the file's first entry is index 1.
            multipliers=(100,) + extended,
        )
    return LevelTable(rarities=rarities)
