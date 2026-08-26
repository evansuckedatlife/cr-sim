"""The playable card layer.

A *card* is what sits in a deck and costs elixir; a *character* is what appears
on the battlefield.  They are separate tables and the distinction matters:
``Rarity`` and ``ManaCost`` live on the card, while ``Hitpoints`` and ``Damage``
live on the character the card summons.  ``Goblins`` is one 2-elixir Common card
that summons three ``Goblin_Stab`` characters.

Cards come from five tables:

===================== ============================================
``spells_characters``  troop cards
``spells_buildings``   building cards
``spells_other``       spell cards (and a few that summon a carrier)
``spells_evolved``     Evolution variants of the above
``spells_hero_form``   hero-form variants used by an event mode
===================== ============================================

Only the first three make up the standard deck pool; the other two are kept
because Evolutions are real cards and the hero forms share the same schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterator, Mapping

from .leveling import LevelTable
from .source import LogicData, UnknownEntity

__all__ = ["Card", "CardKind", "CardRegistry", "build_card_registry"]


class CardKind(str, Enum):
    TROOP = "troop"
    BUILDING = "building"
    SPELL = "spell"


#: source table -> (kind, is_evolution, is_hero_form)
_CARD_TABLES: dict[str, tuple[CardKind, bool, bool]] = {
    "spells_characters": (CardKind.TROOP, False, False),
    "spells_buildings": (CardKind.BUILDING, False, False),
    "spells_other": (CardKind.SPELL, False, False),
    "spells_evolved": (CardKind.TROOP, True, False),
    "spells_hero_form": (CardKind.TROOP, False, True),
}

#: Tables whose cards can appear in an ordinary battle deck.
STANDARD_TABLES = ("spells_characters", "spells_buildings", "spells_other")


@dataclass(frozen=True, slots=True)
class Card:
    """One deck-slot entry."""

    name: str
    kind: CardKind
    mana_cost: int
    rarity: str
    source_table: str
    is_evolution: bool = False
    is_hero_form: bool = False
    not_in_use: bool = False
    not_visible: bool = False

    # What the card puts on the battlefield.  Exactly one of these paths is
    # normally taken, but a few cards (Rage, Goblin Barrel) do both.
    summon_character: str | None = None
    summon_count: int = 1
    summon_character_second: str | None = None
    summon_count_second: int = 0
    summon_radius: int = 0
    summon_deploy_delay: int = 0
    projectile: str | None = None
    area_effect_object: str | None = None

    unlock_arena: str | None = None
    tid: str | None = None
    raw: Mapping[str, Any] | None = None

    @property
    def evolution_base(self) -> str | None:
        """For an Evolution card, the name of the card it evolves."""
        if not self.is_evolution:
            return None
        return self.name.removesuffix("_EV1").removesuffix("_EV2")

    @property
    def is_playable(self) -> bool:
        return not self.not_in_use and not self.not_visible and self.mana_cost > 0

    def summons(self) -> tuple[tuple[str, int], ...]:
        """(character, count) pairs this card deploys."""
        out: list[tuple[str, int]] = []
        if self.summon_character:
            out.append((self.summon_character, max(1, self.summon_count)))
        if self.summon_character_second and self.summon_count_second:
            out.append((self.summon_character_second, self.summon_count_second))
        return tuple(out)


@dataclass(frozen=True, slots=True)
class CardRegistry:
    cards: tuple[Card, ...]
    by_name: Mapping[str, Card]

    def __iter__(self) -> Iterator[Card]:
        return iter(self.cards)

    def __len__(self) -> int:
        return len(self.cards)

    def __getitem__(self, name: str) -> Card:
        return self.by_name[name]

    def get(self, name: str) -> Card | None:
        return self.by_name.get(name)

    def standard(self) -> tuple[Card, ...]:
        """Playable, non-Evolution, non-hero-form cards -- the normal deck pool."""
        return tuple(
            c
            for c in self.cards
            if c.source_table in STANDARD_TABLES and c.is_playable
        )

    def evolutions(self) -> tuple[Card, ...]:
        return tuple(c for c in self.cards if c.is_evolution and c.is_playable)

    def of_kind(self, kind: CardKind) -> tuple[Card, ...]:
        return tuple(c for c in self.standard() if c.kind is kind)


def _int(value: Any, default: int = 0) -> int:
    return value if isinstance(value, int) else default


def build_card_registry(data: LogicData) -> CardRegistry:
    """Read every ``spells_*`` table into a :class:`CardRegistry`."""
    cards: list[Card] = []
    for table_name, (kind, is_evo, is_hero) in _CARD_TABLES.items():
        table = data.tables.get(table_name)
        if table is None:
            continue
        for record in table:
            mana = record.scalar("ManaCost")
            if mana is None:
                continue
            cards.append(
                Card(
                    name=record.name,
                    kind=kind,
                    mana_cost=mana,
                    rarity=record.scalar("Rarity", "Common"),
                    source_table=table_name,
                    is_evolution=is_evo,
                    is_hero_form=is_hero,
                    not_in_use=bool(record.scalar("NotInUse", False)),
                    not_visible=bool(record.scalar("NotVisible", False)),
                    summon_character=record.scalar("SummonCharacter"),
                    summon_count=_int(record.scalar("SummonNumber"), 1),
                    summon_character_second=record.scalar("SummonCharacterSecond"),
                    summon_count_second=_int(record.scalar("SummonCharacterSecondCount"), 0),
                    summon_radius=_int(record.scalar("SummonRadius"), 0),
                    summon_deploy_delay=_int(record.scalar("SummonDeployDelay"), 0),
                    projectile=record.scalar("Projectile"),
                    area_effect_object=record.scalar("AreaEffectObject"),
                    unlock_arena=record.scalar("UnlockArena"),
                    tid=record.scalar("TID"),
                    raw=None,
                )
            )

    by_name: dict[str, Card] = {}
    for card in cards:
        by_name.setdefault(card.name, card)
    return CardRegistry(cards=tuple(cards), by_name=by_name)


#: The level the game displays for tournament standard, for every rarity.
TOURNAMENT_DISPLAY_LEVEL = 11


def _spell_payload(data: LogicData, scale, level: int, card: Card, summary: dict[str, Any]) -> None:
    """Fill in a spell's damage/radius by following projectile -> area effect."""
    projectile: Mapping[str, Any] = {}
    if card.projectile:
        try:
            projectile = data.resolve(f"PROJECTILE.{card.projectile}")
        except UnknownEntity:
            summary["error"] = f"unknown projectile {card.projectile!r}"
            return
        summary["projectile"] = card.projectile

    area_name = card.area_effect_object or projectile.get("AreaEffectObject")
    area: Mapping[str, Any] = {}
    if isinstance(area_name, str):
        try:
            area = data.resolve(f"AEO.{area_name}")
            summary["area_effect"] = area_name
        except UnknownEntity:
            area = {}

    damage = projectile.get("Damage")
    if isinstance(damage, int):
        summary["damage"] = scale.scale(damage, level)
    radius = projectile.get("Radius") or area.get("Radius")
    if isinstance(radius, int):
        summary["radius"] = radius

    # Area effects tick damage over a lifetime instead of hitting once.
    tick_damage = area.get("Damage")
    if isinstance(tick_damage, int):
        summary["area_damage_per_tick"] = scale.scale(tick_damage, level)
    for key, field_name in (
        ("area_duration", "LifeDuration"),
        ("area_hit_frequency", "HitFrequency"),
        ("crown_tower_damage_percent", "CrownTowerDamagePercent"),
    ):
        value = area.get(field_name, projectile.get(field_name))
        if value is not None:
            summary[key] = value


def card_stat_summary(
    data: LogicData,
    levels: LevelTable,
    card: Card,
    display_level: int = TOURNAMENT_DISPLAY_LEVEL,
) -> dict[str, Any]:
    """Scaled headline stats for a card, for validation and display.

    ``display_level`` is the rarity-independent level the game shows (1..15), so
    11 is tournament standard for every card.

    Damage for ranged units lives on the projectile, not the character, so the
    projectile is followed when the character itself has none.
    """
    scale = levels.get(card.rarity)
    level = scale.internal_level(display_level)

    summary: dict[str, Any] = {
        "card": card.name,
        "kind": card.kind.value,
        "rarity": card.rarity,
        "elixir": card.mana_cost,
        "level": level,
        "display_level": display_level,
    }

    summons = card.summons()
    if not summons:
        # A pure spell carries its payload on a projectile, which in turn may
        # hand off to an area-effect object (Poison, Tornado, Graveyard).
        _spell_payload(data, scale, level, card, summary)
        return summary

    character_name, count = summons[0]
    summary["character"] = character_name
    summary["count"] = count
    try:
        character = data.resolve(character_name)
    except UnknownEntity:
        summary["error"] = f"unknown character {character_name!r}"
        return summary

    hitpoints = character.get("Hitpoints")
    if isinstance(hitpoints, int):
        summary["hitpoints"] = scale.scale(hitpoints, level)

    damage = character.get("Damage")
    projectile_name = character.get("Projectile")
    if not isinstance(damage, int) and isinstance(projectile_name, str):
        try:
            projectile = data.resolve(f"PROJECTILE.{projectile_name}")
        except UnknownEntity:
            projectile = {}
        damage = projectile.get("Damage")
        summary["damage_from_projectile"] = projectile_name
    if isinstance(damage, int):
        summary["damage"] = scale.scale(damage, level)

    for key, field in (
        ("hit_speed", "HitSpeed"),
        ("load_time", "LoadTime"),
        ("range", "Range"),
        ("speed", "Speed"),
        ("sight_range", "SightRange"),
        ("deploy_time", "DeployTime"),
        ("collision_radius", "CollisionRadius"),
        ("mass", "Mass"),
    ):
        value = character.get(field)
        if value is not None:
            summary[key] = value

    if summary.get("hit_speed") and summary.get("damage"):
        summary["dps"] = round(summary["damage"] * 1000 / summary["hit_speed"])
    return summary
