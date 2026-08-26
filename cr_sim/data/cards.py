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
    #: Reworked multi-unit cards (Three Musketeers) name their units explicitly
    #: and give each a spawn offset in milli-tiles instead of a radius.
    summon_characters_list: tuple[str, ...] = ()
    summon_offsets: tuple[tuple[int, int], ...] = ()
    projectile: str | None = None
    area_effect_object: str | None = None

    unlock_arena: str | None = None
    tid: str | None = None
    #: Set when the card has no explicit summon field but a character of the
    #: same name exists -- the game's implicit convention.
    implicit_character: str | None = None
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
        """(character, count) pairs this card deploys.

        Three spellings exist and are tried in order: an explicit character
        list (Three Musketeers), the classic ``SummonCharacter`` field, and --
        when neither is set -- the character sharing the card's own name, which
        is how Ice Wizard and Electro Wizard are wired.
        """
        if self.summon_characters_list:
            return tuple((name, 1) for name in self.summon_characters_list)
        out: list[tuple[str, int]] = []
        if self.summon_character:
            out.append((self.summon_character, max(1, self.summon_count)))
        if self.summon_character_second and self.summon_count_second:
            out.append((self.summon_character_second, self.summon_count_second))
        if not out and self.implicit_character:
            out.append((self.implicit_character, max(1, self.summon_count)))
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


#: source table -> the namespace its TOML overlay lands in
_TABLE_NAMESPACE = {
    "spells_characters": "SPELL_CHARACTER",
    "spells_buildings": "SPELL_BUILDING",
    "spells_other": "SPELL_OTHER",
    "spells_evolved": "SPELL_EVOLVED",
    "spells_hero_form": "SPELL_HERO",
}


def build_card_registry(data: LogicData) -> CardRegistry:
    """Read every ``spells_*`` table into a :class:`CardRegistry`.

    Cards are read through :meth:`LogicData.resolve` rather than straight off
    the CSV row, because several have migrated into the matching ``.toml``
    overlay -- Three Musketeers' unit list only exists there.
    """
    cards: list[Card] = []
    for table_name, (kind, is_evo, is_hero) in _CARD_TABLES.items():
        namespace = _TABLE_NAMESPACE[table_name]
        if table_name not in data.tables and not data.namespace(namespace):
            continue
        for name in data.names(namespace):
            row = data.resolve(f"{namespace}.{name}")
            mana = row.get("ManaCost")
            if not isinstance(mana, int):
                continue

            summon_list = tuple(_strs(row.get("SummonCharactersList")))
            offsets_x = _ints(row.get("SummonCharactersOffsetsX"))
            offsets_y = _ints(row.get("SummonCharactersOffsetsY"))
            offsets = tuple(zip(offsets_x, offsets_y)) if offsets_x and offsets_y else ()

            explicit = row.get("SummonCharacter") or summon_list
            implicit = None
            if not explicit and _character_exists(data, name):
                implicit = name

            cards.append(
                Card(
                    name=name,
                    kind=kind,
                    mana_cost=mana,
                    rarity=row.get("Rarity") or "Common",
                    source_table=table_name,
                    is_evolution=is_evo,
                    is_hero_form=is_hero,
                    not_in_use=bool(row.get("NotInUse", False)),
                    not_visible=bool(row.get("NotVisible", False)),
                    summon_character=row.get("SummonCharacter"),
                    summon_count=_int(row.get("SummonNumber"), 1),
                    summon_character_second=row.get("SummonCharacterSecond"),
                    summon_count_second=_int(row.get("SummonCharacterSecondCount"), 0),
                    summon_radius=_int(row.get("SummonRadius"), 0),
                    summon_deploy_delay=_int(row.get("SummonDeployDelay"), 0),
                    summon_characters_list=summon_list,
                    summon_offsets=offsets,
                    projectile=row.get("Projectile"),
                    area_effect_object=row.get("AreaEffectObject"),
                    unlock_arena=row.get("UnlockArena"),
                    tid=row.get("TID"),
                    implicit_character=implicit,
                    raw=row,
                )
            )

    by_name: dict[str, Card] = {}
    for card in cards:
        by_name.setdefault(card.name, card)
    return CardRegistry(cards=tuple(cards), by_name=by_name)


def _character_exists(data: LogicData, name: str) -> bool:
    for namespace in ("CHARACTER", "BUILDING", "EXT"):
        if name in data.namespace(namespace) or name in data.names(namespace):
            return True
    return False


def _strs(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [v for v in value if isinstance(v, str)]
    return []


def _ints(value: Any) -> list[int]:
    if isinstance(value, int):
        return [value]
    if isinstance(value, (list, tuple)):
        return [v for v in value if isinstance(v, int)]
    return []


#: The level the game displays for tournament standard, for every rarity.
TOURNAMENT_DISPLAY_LEVEL = 11


def _projectile_damage(data: LogicData, name: Any) -> int | None:
    if not isinstance(name, str):
        return None
    try:
        projectile = data.resolve(f"PROJECTILE.{name}")
    except UnknownEntity:
        return None
    damage = projectile.get("Damage")
    return damage if isinstance(damage, int) else None


def _character_damage(data: LogicData, character: Mapping[str, Any]) -> tuple[int | None, str | None]:
    """Find a character's per-hit damage, wherever the build happens to keep it.

    Four places, in precedence order:

    1. ``Damage`` on the character (melee units).
    2. ``CustomFirstProjectile`` -- Princess fires this for damage while her
       plain ``Projectile`` is a visual-only "Deco" round.
    3. ``Projectile`` (ordinary ranged units).
    4. ``AttackSequenceList`` -- reworked multi-swing units such as Berserker
       and Three Musketeers carry per-swing damage or a per-swing projectile.
    """
    damage = character.get("Damage")
    if isinstance(damage, int):
        return damage, "character"

    for field_name, label in (
        ("CustomFirstProjectile", "custom_first_projectile"),
        ("Projectile", "projectile"),
    ):
        found = _projectile_damage(data, character.get(field_name))
        if found is not None:
            return found, f"{label}:{character[field_name]}"

    sequence = character.get("AttackSequenceList")
    if isinstance(sequence, (list, tuple)):
        for step in sequence:
            if not isinstance(step, Mapping):
                continue
            step_damage = step.get("Damage")
            if isinstance(step_damage, int):
                return step_damage, "attack_sequence"
            found = _projectile_damage(data, step.get("Projectile"))
            if found is not None:
                return found, f"attack_sequence_projectile:{step['Projectile']}"
    return None, None


def _note_multishot(source: Mapping[str, Any], summary: dict[str, Any]) -> None:
    """Flag entities that fire more than one projectile per attack.

    This matters because the ``damage`` reported everywhere else is the damage
    of *one* projectile.  Hunter fires 10 pellets from one shot, so his 84 is
    per-pellet and a point-blank hit is worth roughly ten times that; Arrows
    lands three separate waves 200ms apart, so a unit that walks out of the
    radius takes fewer of them.  Leaving these implicit would make the numbers
    look wrong later for reasons nobody could see.
    """
    for key, field_name in (
        ("multiple_projectiles", "MultipleProjectiles"),
        ("group_projectiles", "GroupProjectiles"),
        ("projectile_waves", "ProjectileWaves"),
        ("projectile_wave_interval", "ProjectileWaveInterval"),
        ("area_damage_radius", "AreaDamageRadius"),
    ):
        value = source.get(field_name)
        if value is not None:
            summary[key] = value
    if summary.get("multiple_projectiles") or summary.get("projectile_waves"):
        summary["damage_is_per_projectile"] = True


def _resolve_opt(data: LogicData, namespace: str, name: Any) -> Mapping[str, Any]:
    if not isinstance(name, str):
        return {}
    try:
        return data.resolve(f"{namespace}.{name}")
    except UnknownEntity:
        return {}


def _spell_payload(data: LogicData, scale, level: int, card: Card, summary: dict[str, Any]) -> None:
    """Fill in a spell's payload by walking projectile -> area effect -> buff.

    Spells keep their numbers in three different places depending on how they
    work, and a spell can chain through all of them:

    * **direct hit** -- damage on the projectile (Rocket, Fireball) or on a
      one-shot area effect (Zap, Freeze);
    * **indirect** -- the area effect fires its own projectile (Lightning);
    * **over time** -- no damage at all on either, just a buff that carries
      ``DamagePerSecond`` (Poison, Tornado).
    """
    card_row = card.raw or {}
    projectile = _resolve_opt(data, "PROJECTILE", card.projectile)
    if projectile:
        summary["projectile"] = card.projectile

    area_name = card.area_effect_object or projectile.get("AreaEffectObject")
    area = _resolve_opt(data, "AEO", area_name)
    if area:
        summary["area_effect"] = area_name

    # An area effect may itself launch the projectile that does the damage.
    area_projectile = _resolve_opt(data, "PROJECTILE", area.get("Projectile"))

    # The Log's thrown projectile carries no damage; the rolling one it spawns
    # does, so follow the SpawnProjectile chain.
    chain: list[tuple[Mapping[str, Any], str]] = []
    seen: set[str] = set()
    current, label = projectile, "projectile"
    while current:
        chain.append((current, label))
        spawn = current.get("SpawnProjectile")
        if not isinstance(spawn, str) or spawn in seen:
            break
        seen.add(spawn)
        current, label = _resolve_opt(data, "PROJECTILE", spawn), f"spawn_projectile:{spawn}"

    for source, source_label in [*chain, (area, "area_effect"), (area_projectile, "area_projectile")]:
        damage = source.get("Damage")
        if isinstance(damage, int):
            summary["damage"] = scale.scale(damage, level)
            summary["damage_source"] = source_label
            break

    # A spell projectile can also carry troops (Goblin Barrel, Royal Delivery).
    for source, _label in chain:
        spawned = source.get("SpawnCharacter")
        if isinstance(spawned, str):
            summary["spawns_character"] = spawned
            summary["spawns_count"] = _int(source.get("SpawnCharacterCount"), 1)
            break

    radius = card_row.get("Radius") or projectile.get("Radius") or area.get("Radius")
    if isinstance(radius, int):
        summary["radius"] = radius

    _note_multishot(card_row, summary)

    # Damage-over-time lives on the buff the area effect applies.
    buff = _resolve_opt(data, "BUFF", area.get("Buff") or projectile.get("TargetBuff"))
    if buff:
        summary["buff"] = area.get("Buff") or projectile.get("TargetBuff")
        dps = buff.get("DamagePerSecond")
        if isinstance(dps, int):
            summary["damage_per_second"] = scale.scale(dps, level)
        for key, field_name in (
            ("buff_speed_multiplier", "SpeedMultiplier"),
            ("buff_hit_speed_multiplier", "HitSpeedMultiplier"),
            ("buff_hit_frequency", "HitFrequency"),
        ):
            if buff.get(field_name) is not None:
                summary[key] = buff[field_name]

    for key, field_name in (
        ("duration", "LifeDuration"),
        ("hit_frequency", "HitSpeed"),
        ("buff_time", "BuffTime"),
        ("crown_tower_damage_percent", "CrownTowerDamagePercent"),
    ):
        value = area.get(field_name, projectile.get(field_name))
        if value is not None:
            summary[key] = value

    # Graveyard, Vines and Clone define what they do entirely in the ACTION
    # graph rather than in stat fields.  Record which action drives them so the
    # cards needing the action interpreter are visible rather than looking blank.
    for field_name in ("OnStartingAction", "OnHitAction"):
        action = area.get(field_name) or projectile.get(field_name)
        if isinstance(action, str):
            summary["action"] = action
            break


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
    # Total bodies deployed, across a squad of mixed unit types (Rascals) or an
    # explicit unit list (Three Musketeers).
    summary["count"] = sum(n for _name, n in summons)
    if len(summons) > 1:
        summary["squad"] = [{"character": n, "count": c} for n, c in summons]
    try:
        character = data.resolve(character_name)
    except UnknownEntity:
        summary["error"] = f"unknown character {character_name!r}"
        return summary

    hitpoints = character.get("Hitpoints")
    if isinstance(hitpoints, int):
        summary["hitpoints"] = scale.scale(hitpoints, level)

    damage, source = _character_damage(data, character)
    if isinstance(damage, int):
        summary["damage"] = scale.scale(damage, level)
        if source:
            summary["damage_source"] = source

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

    _note_multishot(character, summary)

    if summary.get("hit_speed") and summary.get("damage"):
        summary["dps"] = round(summary["damage"] * 1000 / summary["hit_speed"])
    return summary
