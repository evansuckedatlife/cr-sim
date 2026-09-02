"""A card described by what it does, not by which card it is.

The card-conditioned policy head (:class:`cr_sim.train.nets.FactoredHead`)
reads a one-hot out of the observation and multiplies it by a learned matrix.
That matrix has one free column per vocabulary entry, and the vocabulary is
*this episode's deck union* -- rebuilt per environment by
:func:`cr_sim.api.encoding.build_encoding_config`. So column ``i`` means
whatever ``vocab[i]`` happens to be, and swapping in a different deck silently
reuses the Knight's learned column for whatever card sorts into position 4.
Nothing errors; the head is simply conditioned on the wrong card.

This module is the alternative: turn a card into a fixed-length vector of its
own statistics -- hitpoints, damage per second, reach, speed, what it targets,
what it leaves behind -- so the head learns "slow, tanky, melee, ground-only
goes at the bridge" rather than "card 4 goes at the bridge". A card the agent
has never seen then gets a description for free, because the description is
computed from the card's data rather than looked up in a table of cards that
happened to be in the training decks.

**The boundary is worth stating, and it is narrower than it looks.** What this
buys is an identity-free *head*. It does not make the network as a whole
identity-free, and two separate things stop it:

*   ``_vector_length`` makes the observation's width a function of
    ``vocab_size``, so a deck pair whose union is a different *size* changes
    ``vector.0.weight`` and fails strict loading on the trunk before the head
    is ever reached. Same-size decks only.
*   **The trunk still sees card identity.** The observation carries ten card
    one-hots -- four hand slots plus the next card, per side
    (:func:`cr_sim.api.encoding.encode_observation`) -- at offsets indexed by
    ``vocab``, and every one of them feeds ``ActorCritic.vector``. Measured
    on an 8-card mirror deck: permuting the vocabulary consistently across all
    ten one-hot blocks *and* the stat table -- a pure relabelling, identical
    cards in identical slots -- leaves the head's conditioning invariant to
    4e-07 and moves the trunk's features by 56% relative L2. 80 of the 102
    columns of ``vector.0.weight`` are per-vocab-index one-hot columns, all of
    them receive gradient, and training will memorise identity there exactly
    as the old 32x8 table did.

So: swapping in an unseen 8-card mirror deck loads and runs, and the *head*
conditions correctly on cards it has never seen. The trunk does not, and
whether that is enough is an empirical question this module does not settle.
Dropping the identity one-hots from the observation is the change that would,
and it is a change to the observation, not to this file.

Where the numbers come from
---------------------------

:func:`cr_sim.engine.specs.spec_for_card`, not the raw rows and not
:func:`cr_sim.data.cards.card_stat_summary`, for anything about a deployed
unit. Two reasons, both measured over the 122 standard cards:

*   ``Damage`` is on only 51 of the 102 summoning cards in the raw data,
    because a ranged unit keeps its damage on its projectile.
    ``build_unit_spec`` follows the projectile chain; a raw read does not, and
    a Musketeer with zero damage is a card that does not fight.
*   ``AttacksAir`` is absent from ``card_stat_summary`` entirely. It exists
    only on :class:`~cr_sim.engine.specs.UnitSpec`.

And the ``UnitSpec`` is not the end of it either -- see
:func:`_on_hit_payload`. Splash and the on-hit buff live on the *projectile*
for a ranged unit: of the 25 standard cards with any splash radius, 17 carry
it only there, and every stun, slow and heal in the game is either a
projectile's ``TargetBuff`` or the unit's own ``buff_on_damage``.

``_HP_NORM``, ``_DPS_NORM``, ``_REACH_NORM``, ``_COUNT_NORM`` and
``MAX_ELIXIR`` are the encoder's own, so a card's hitpoint feature reads on
the same scale as the hitpoint mass in the observation grid the same network
sees. The rest are this module's, and the rule for those is that 1.0 is the
build's own measured maximum: a norm a real card exceeds silently clips the
most extreme card in the game onto the same value as a merely large one, which
is the failure ``_BURST_NORM``, ``_TOTAL_DPS_NORM`` and ``_SIGHT_NORM`` exist
to avoid.

Four things that are deliberate and look like bugs
--------------------------------------------------

**Every block carries a validity flag, and a missing field is a hard zero
behind it.** ``has_unit`` gates the unit block and ``has_payload`` the spell
block. A spell zero-filled into the unit block reads as "a unit with no
health", which is a different claim from "not a unit", and the network has no
way to tell them apart. Two fields where zero is the *meaning* rather than a
gap: ``Speed`` and ``Mass`` are absent for exactly the 13 building-entity
cards, and "does not move" is the correct value -- disambiguated by
``bldg_entity``, not by imputing a median.

**Damage lies about six real threats,** which is why the five death/spawn
flags exist. ``damage == 0`` on ten cards and only Elixir Collector and Goblin
Drill are genuinely harmless: Barbarian Hut, Goblin Hut, Tombstone and Goblin
Cage carry their payload in ``SpawnCharacter`` or an ``OnStartingAction``
graph, Skeleton Balloon and Suspicious Bush in a death spawn, Royal Delivery
in ``DeathDamage`` plus ``DeathSpawnCharacter``. Without those flags a hut
reads as a harmless 0-DPS box.

**Rage is the card the unit path lies about worst,** and it is why the fuse
fallback exists. ``Rage.summons()`` returns ``RageBottle``: a 2-hitpoint fuse
whose entire attribute set is ``DeployTime``, ``IsBuilding`` and
``DeathAreaEffect``. Its ``UnitSpec`` is hitpoints 2, damage 0, speed 0, range
0 -- a bottle, not a spell. So a card whose specs are *all* fuses is read
through its fuse's death payload instead, resolving ``AEO.<name>`` and then
``BUFF.<name>``. Royal Delivery takes the same route and comes out as 384
damage in 3 tiles plus a spawn. The namespace prefix on the buff is not
optional: resolving the bare name returns an empty mapping and every *other*
field of the card still looks right.

**A variant card comes out as a spell with a full unit block,** which is
Merge Maiden and nothing else in the standard pool. It summons nothing itself;
it names the forms it can turn into, and ``battle.play_card`` deploys whichever
the elixir on hand pays for. Block A stays the card in hand's -- six elixir is
what the player pays, and the game data does call it a spell -- while the unit
block describes the richest form, the one it becomes at full elixir. Read
literally instead, it is ``mana`` plus ``is_spell`` and forty-five zeros: a
six-elixir card the head is told puts nothing on the board.

The level and the clock are pinned, not read from the environment
-----------------------------------------------------------------

``CARD_FEATURE_LEVEL`` is a module constant. The browser server never sets a
card level and the trainer may, so a table that followed ``env.level`` would
condition the same checkpoint on different numbers in two places -- an entire
class of train/serve skew, bought for nothing. Rarity scaling was verified
rarity-uniform (the scale factor is identical across Common, Rare, Epic,
Legendary and Champion at every display level), so the level only rescales the
whole table roughly uniformly and the head reads relative stats anyway.

``TickClock(60)`` is pinned defensively, and the measurement says it is
currently unfalsifiable: across all 122 cards and all four tick-valued fields
below, **zero diverge between 20 and 60 TPS** once divided by
``ticks_per_second``. There is deliberately no test asserting that -- no
mutation could kill it, and this codebase's signature failure is a green test
over nothing.

Validated
---------

Over all 122 ``registry.standard()`` cards: no card raises, no feature is
clipped by ``_clip`` or ``_signed_clip``, and the 122 vectors are **pairwise
distinct** -- no collisions. ``is_mirror`` is a dimension because Mirror has
no payload, no unit and no variant of any kind, and would otherwise be an
all-zero vector indistinguishable from an empty hand slot.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from ..engine.constants import MAX_ELIXIR, TickClock
from ..engine.entity import EntityKind
from ..engine.fixed import SUBTILES_PER_TILE
from ..engine.projectiles import build_projectile_spec
from ..engine.specs import UnitSpec, spec_for_card
from .cards import Card, CardKind, CardRegistry, card_stat_summary
from .leveling import LevelTable
from .source import LogicData, UnknownEntity

__all__ = [
    "CARD_FEATURE_NAMES", "CARD_FEATURE_COUNT", "CARD_FEATURE_LEVEL",
    "CARD_FEATURE_CLOCK", "card_feature_vector", "card_feature_table",
]

#: The displayed level every card in the table is scaled to. A constant, not
#: ``env.level`` -- see the module docstring.
CARD_FEATURE_LEVEL = 11

#: The tick rate the table's timing fields are built at. Pinned so a 20 TPS
#: training run and a 60 TPS verification run describe a card identically;
#: measured to make no difference to any of the 122 cards.
CARD_FEATURE_CLOCK = TickClock(60)

# Normalisation constants, mirrored from cr_sim.api.encoding so a card's
# hitpoint feature reads on the same scale as its hitpoint mass in the grid
# the same network sees. Duplicated rather than imported to keep the data
# layer free of the encoder, and pinned against it by a test.
_HP_NORM = 6000.0
_DPS_NORM = 800.0
_REACH_NORM = 12.0
_COUNT_NORM = 4.0

_SUBTILES = float(SUBTILES_PER_TILE)
#: Bodies that saturate the swarm feature. ``log1p`` rather than ``n / 4``:
#: linear normalisation pinned eight swarm cards at 1.0 and could not separate
#: Goblins from Skeleton Army (15).
_MAX_BODIES = 15.0
#: Splash radius in tiles that saturates. The build's widest is Princess at
#: 2.5, so nothing sits on the ceiling and there is room above the pool for a
#: card this table has never seen.
_SPLASH_NORM = 3.0
#: Tiles per minute that saturates the speed feature. 120 is "very fast".
_SPEED_NORM = 120.0
#: Seconds of deploy or windup that saturates. Four is beyond the longest:
#: X-Bow and Mortar take 3.5s to deploy and Mortar 3.7s to reload.
_WINDUP_NORM = 4.0
#: Collision radius in tiles that saturates. 1.0 is the build's own maximum,
#: which is why seven cards sit exactly at it rather than being clipped.
_COLLISION_NORM = 1.0
#: A single swing's damage that saturates. Zap Machine's 1331 at display level
#: 11 is the build's maximum and the next is P.E.K.K.A's 842, so 1000 -- what
#: this was -- put the hardest-hitting card in the game on a ceiling and threw
#: away the 489 points of daylight between the two.
_BURST_NORM = 1400.0
#: A card's summed damage per second that saturates. Skeleton Army's fifteen
#: bodies come to 1104.5 against Battle Ram's 715 for second place, and 800 --
#: the single-unit ``_DPS_NORM`` this used to reuse -- clipped the one card
#: whose whole argument is the total.
_TOTAL_DPS_NORM = 1200.0
#: Aggro radius in tiles that saturates. Its own constant rather than
#: ``_REACH_NORM``: Goblin Cage sees 20 tiles, which is nearly twice the
#: longest *attack* range and is the whole reason the card pulls a push.
_SIGHT_NORM = 20.0
#: A spell's damage that saturates. Above a Rocket at tournament standard.
_PAYLOAD_DAMAGE_NORM = 1500.0
#: A spell's radius in tiles that saturates.
_PAYLOAD_RADIUS_NORM = 6.0
#: Seconds a spell's area lasts that saturates.
_PAYLOAD_DURATION_NORM = 10.0
#: Seconds a spell's buff lasts that saturates.
_PAYLOAD_BUFF_NORM = 5.0
#: Percentage that saturates a buff multiplier. **150, not 100.** Slows run
#: -100..-15, but ``BUFF.Rage`` carries ``SpeedMultiplier: 130`` and
#: ``HitSpeedMultiplier: 130``, so dividing by 100 would clip Rage flat
#: against the ceiling and lose the only positive value in the column.
_BUFF_MULTIPLIER_NORM = 150.0

#: Every feature, in order. The encoder's first layer is indexed by this, so
#: the order is part of the contract: inserting a name in the middle silently
#: redefines every weight after it.
CARD_FEATURE_NAMES: tuple[str, ...] = (
    # -- block A: the card itself, valid for every card
    "mana", "is_troop", "is_building", "is_spell", "enemy_side_ok",
    "is_mirror", "bodies",
    # -- block B: the deployed unit, gated by has_unit
    "has_unit", "hp", "total_hp", "dps", "total_dps", "damage",
    "attack_range", "sight_range", "splash", "speed", "deploy", "load",
    "collision", "air", "ground", "flying", "only_bldgs", "bldg_entity",
    "kamikaze", "death_spawn", "death_blast", "spawner", "charge", "action",
    "on_hit_buff", "on_hit_slow", "on_hit_hitspeed", "on_hit_dps", "heals",
    # -- block C: the spell or area payload, gated by has_payload
    "has_payload", "p_damage", "p_radius", "p_duration", "p_dps",
    "p_speed_mult", "p_hitspeed_mult", "p_buff_time", "p_crown", "p_spawns",
    "p_action",
)

CARD_FEATURE_COUNT = len(CARD_FEATURE_NAMES)

#: The gated blocks, as slices, so a test can assert the gating by name
#: rather than by an index a reader has to count out by hand.
UNIT_BLOCK = slice(CARD_FEATURE_NAMES.index("hp"),
                   CARD_FEATURE_NAMES.index("has_payload"))
PAYLOAD_BLOCK = slice(CARD_FEATURE_NAMES.index("p_damage"), CARD_FEATURE_COUNT)

#: Summary keys that describe the card rather than its payload. A pure spell
#: with none of the others -- Mirror, Merge Maiden -- has no payload at all
#: and must not be given a ``has_payload`` of 1 for carrying its own rarity.
_NON_PAYLOAD_KEYS = frozenset(
    {"card", "kind", "rarity", "elixir", "level", "display_level"})


def _clip(value: float) -> float:
    """Into ``[0, 1]``.

    Bounded on purpose: the encoder's first layer is a tanh, and an unbounded
    input from a card outside the training set would put its hidden
    activations somewhere the second layer has never been.
    """
    return max(0.0, min(1.0, value))


def _signed_clip(value: float) -> float:
    """Into ``[-1, 1]``, for the columns where the sign is the meaning -- a
    slow is negative and a rage positive on the same axis."""
    return max(-1.0, min(1.0, value))


def _area_payload(
    data: LogicData, scale: Any, level: int, name: str
) -> dict[str, Any]:
    """A fuse's death area effect, resolved ``AEO.<name>`` then ``BUFF.<name>``.

    The namespace prefixes are not optional. ``data.resolve("Rage")`` raises
    :class:`~cr_sim.data.source.UnknownEntity` and, caught, leaves an empty
    payload with every other field of the card still looking correct -- which
    is exactly how a spell that does nothing passes a smoke test.
    """
    out: dict[str, Any] = {}
    try:
        area = data.resolve(f"AEO.{name}")
    except (KeyError, UnknownEntity):
        return out
    if area.get("Radius") is not None:
        out["radius"] = area["Radius"]
    if area.get("LifeDuration") is not None:
        out["duration"] = area["LifeDuration"]
    if isinstance(area.get("Damage"), int):
        out["damage"] = scale.scale(area["Damage"], level)
    if area.get("BuffTime") is not None:
        out["buff_time"] = area["BuffTime"]
    if area.get("CrownTowerDamagePercent") is not None:
        out["crown_tower_damage_percent"] = area["CrownTowerDamagePercent"]
    buff_name = area.get("Buff")
    if isinstance(buff_name, str):
        try:
            buff = data.resolve(f"BUFF.{buff_name}")
        except (KeyError, UnknownEntity):
            buff = {}
        if isinstance(buff.get("DamagePerSecond"), int):
            out["damage_per_second"] = scale.scale(buff["DamagePerSecond"], level)
        for key, field in (("buff_speed_multiplier", "SpeedMultiplier"),
                           ("buff_hit_speed_multiplier", "HitSpeedMultiplier")):
            if buff.get(field) is not None:
                out[key] = buff[field]
        # Last link in the crown-tower chain, and the only one a
        # damage-over-time payload has: all of Poison's damage is its buff's,
        # and so is all of its -77. ``cards._spell_payload`` falls back the
        # same way, so both routes give the same card the same answer.
        if ("crown_tower_damage_percent" not in out
                and buff.get("CrownTowerDamagePercent") is not None):
            out["crown_tower_damage_percent"] = buff["CrownTowerDamagePercent"]
    return out


def _on_hit_payload(
    data: LogicData, levels: LevelTable, unit: UnitSpec
) -> dict[str, Any]:
    """What a unit's hits do besides damage, and how wide they land.

    Block B reads the *character* row, and for a ranged unit the character row
    is not where the card is. Two things live on the projectile instead:

    *   **Splash.** ``UnitSpec.area_damage_radius`` is the melee swing's
        radius -- :mod:`cr_sim.engine.battle` applies it only when nothing was
        launched. A ranged splash unit carries its radius on the projectile
        instead, and 17 of the 25 standard cards with a splash radius carry it
        *only* there: Wizard, Baby Dragon, Bowler, Bomb Tower, Witch, Mortar,
        Bomber, Fire Spirits and nine more all read as single-target without
        this, while the eight that keep it on the character -- Valkyrie, Mega
        Knight, Dark Prince, Princess and the rest -- read correctly either
        way. For a head choosing a *tile*, "does this hit a group" is the most
        placement-relevant bit there is.
    *   **The on-hit buff.** Ice Spirit's whole card is ``TargetBuff: Freeze``
        on its projectile, and Heal Spirit's is ``SpawnAreaEffectObject``
        resolving to ``HealPerSecond: 157``. Without them the two are equal on
        every behavioural feature -- one freezes a push and the other heals
        one, and the vector said they were the same card twice.

    ``buff_on_damage`` is read alongside, because Electro Wizard puts the same
    stun on the *unit* rather than on his shot.

    Scaled on the unit's own rarity ladder at its own internal level, which is
    what :func:`cr_sim.engine.specs.build_unit_spec` scaled the unit with.
    """
    out: dict[str, Any] = {}
    scale = levels.get(unit.rarity)
    names: list[str] = []
    if unit.buff_on_damage:
        names.append(unit.buff_on_damage)
    if unit.projectile:
        shot = build_projectile_spec(
            data, unit.projectile, scale, level=unit.level, clock=CARD_FEATURE_CLOCK)
        if shot is not None:
            out["splash_radius"] = shot.radius
            if shot.target_buff:
                names.append(shot.target_buff)
            if shot.area_effect:
                try:
                    area = data.resolve(f"AEO.{shot.area_effect}")
                except (KeyError, UnknownEntity):
                    area = {}
                spawned = area.get("Buff")
                if isinstance(spawned, str):
                    names.append(spawned)
    for name in names:
        try:
            buff = data.resolve(f"BUFF.{name}")
        except (KeyError, UnknownEntity):
            continue
        # Mother Witch's VoodooCurse is the reason this flag is separate from
        # the four numbers below: it carries no multiplier and no damage at
        # all, only ``DeathSpawn``, so on the numbers alone her defining
        # property is indistinguishable from having no on-hit effect.
        out["buff"] = True
        # Strongest wins where a unit carries more than one. Keyed on absolute
        # value so a slow and a rage cannot cancel each other to nothing.
        for key, field in (("speed_multiplier", "SpeedMultiplier"),
                           ("hit_speed_multiplier", "HitSpeedMultiplier")):
            value = buff.get(field)
            if isinstance(value, int) and abs(value) > abs(out.get(key, 0)):
                out[key] = value
        for key, field in (("damage_per_second", "DamagePerSecond"),
                           ("heal_per_second", "HealPerSecond")):
            value = buff.get(field)
            if isinstance(value, int) and value:
                out[key] = max(out.get(key, 0), scale.scale(value, unit.level))
    return out


def _payload_for(
    data: LogicData, levels: LevelTable, card: Card, fuses: Sequence[Any]
) -> Mapping[str, Any]:
    """The card's spell or area payload, by whichever of the two routes applies.

    A pure spell -- nothing in ``summons()`` -- has its payload walked for it
    by :func:`~cr_sim.data.cards.card_stat_summary`, which follows projectile
    to area effect to buff. A card that summons only *fuses*, which is exactly
    Rage and Royal Delivery, has to be read through the fuse's death payload
    instead, because the fuse's own spec describes the delivery vehicle.
    """
    if not card.summons():
        summary = card_stat_summary(
            data, levels, card, display_level=CARD_FEATURE_LEVEL)
        payload = {k: v for k, v in summary.items() if k not in _NON_PAYLOAD_KEYS}
        # Graveyard, Vines and Clone define what they do in an ACTION graph and
        # leave every stat column empty; the card row's own Action is the only
        # sign they do anything at all.
        if "action" not in payload and (card.raw or {}).get("Action"):
            payload["action"] = True
        return payload
    if fuses:
        scale = levels.get(card.rarity)
        level = scale.internal_level(CARD_FEATURE_LEVEL)
        fuse = fuses[0]
        payload: dict[str, Any] = {}
        if fuse.death_damage:
            payload["damage"] = fuse.death_damage
        if fuse.death_damage_radius:
            # Back into the milli-tiles the spell route reports radii in, so
            # both routes feed the same normalisation.
            payload["radius"] = int(fuse.death_damage_radius / _SUBTILES * 1000)
        if fuse.death_spawn_character:
            payload["spawns_count"] = max(1, fuse.death_spawn_count)
        if fuse.death_area_effect:
            payload.update(
                _area_payload(data, scale, level, fuse.death_area_effect))
        if fuse.lifetime_ticks:
            payload.setdefault(
                "duration",
                int(fuse.lifetime_ticks * 1000 / fuse.ticks_per_second))
        return payload
    return {}


def card_feature_vector(
    data: LogicData, levels: LevelTable, registry: CardRegistry, card: Card
) -> tuple[float, ...]:
    """One card as :data:`CARD_FEATURE_COUNT` numbers in ``[-1, 1]``.

    **Card-local, always.** Every constant this divides by is fixed and named
    above; nothing is normalised against the other cards in a deck. That is
    load-bearing rather than stylistic. A z-score or a min/max taken over a
    vocabulary would give the same card a different vector depending on which
    cards it was drawn alongside, which destroys the entire generalisation
    claim while looking like an improvement.

    ``registry`` is here for the one card that describes itself by naming
    another: see the variant resolution below.
    """
    values = [0.0] * CARD_FEATURE_COUNT
    at = CARD_FEATURE_NAMES.index

    # -- block A: true of every card ---------------------------------------
    values[at("mana")] = _clip(card.mana_cost / MAX_ELIXIR)
    values[at("is_troop")] = float(card.kind is CardKind.TROOP)
    values[at("is_building")] = float(card.kind is CardKind.BUILDING)
    values[at("is_spell")] = float(card.kind is CardKind.SPELL)
    values[at("enemy_side_ok")] = float(card.can_deploy_on_enemy_side)
    values[at("is_mirror")] = float(card.is_mirror)

    # A variant card summons nothing itself; it names the forms it can turn
    # into and the engine deploys whichever the elixir on hand pays for
    # (``battle.play_card``). Merge Maiden is the whole set: six elixir, no
    # ``summons()``, and read literally it comes out as ``mana`` plus
    # ``is_spell`` and forty-five zeros -- a six-elixir card the head is told
    # puts nothing on the board, two bits away from an empty hand slot.
    # The richest form is the one it is played as at full elixir, so that is
    # the one described. Block A stays the *outer* card's: the cost, the
    # deploy restriction and the mirror flag are properties of the card in
    # hand, not of the body it becomes.
    spec_card = card
    if card.variants and not card.summons():
        spec_card = registry.get(card.variants[0][1])
        if spec_card is None:
            raise KeyError(
                f"{card.name} deploys its variant {card.variants[0][1]!r}, "
                "which the registry does not have. A zero row here is a card "
                "the head believes is nothing.")

    specs = spec_for_card(data, levels, spec_card,
                          display_level=CARD_FEATURE_LEVEL,
                          clock=CARD_FEATURE_CLOCK)
    live = [spec for spec in specs if not spec.is_fuse]
    fuses = [spec for spec in specs if spec.is_fuse]
    values[at("bodies")] = _clip(math.log1p(len(specs)) / math.log1p(_MAX_BODIES))

    # -- block B: the deployed unit ----------------------------------------
    # Gated on ``live`` and not on ``specs``. Rage and Royal Delivery summon
    # only a fuse, and reading that as the unit fills this whole block with a
    # bottle's 2 hitpoints, zero speed and zero range.
    if live:
        unit = live[0]
        tps = float(unit.ticks_per_second)
        values[at("has_unit")] = 1.0
        # Shield folded into hitpoints as effective health: one fewer column
        # that is zero for 119 of 122 cards, and a truer number for the three.
        values[at("hp")] = _clip(
            (unit.hitpoints + unit.shield_hitpoints) / _HP_NORM)
        values[at("total_hp")] = _clip(
            sum(s.hitpoints + s.shield_hitpoints for s in live) / _HP_NORM)
        values[at("dps")] = _clip(unit.damage_per_second / _DPS_NORM)
        values[at("total_dps")] = _clip(
            sum(s.damage_per_second for s in live) / _TOTAL_DPS_NORM)
        # Burst per swing, which damage-per-second alone loses: one huge slow
        # hit and a stream of small fast ones can share a rate and want
        # opposite tiles. Zap Machine swings for 1331 at 332 per second;
        # nothing else in the pool hits for even two thirds of that.
        values[at("damage")] = _clip(unit.damage / _BURST_NORM)
        values[at("attack_range")] = _clip(
            unit.attack_range / _SUBTILES / _REACH_NORM)
        values[at("sight_range")] = _clip(
            unit.sight_range / _SUBTILES / _SIGHT_NORM)
        on_hit = _on_hit_payload(data, levels, unit)
        # The larger of the melee swing's radius and the projectile's. Not the
        # projectile's alone: Princess carries 2.5 tiles on the character and
        # 2.0 on the shot, and taking the shot's would shrink her.
        values[at("splash")] = _clip(
            max(unit.area_damage_radius, on_hit.get("splash_radius", 0))
            / _SUBTILES / _SPLASH_NORM)
        values[at("speed")] = _clip(unit.speed / _SPEED_NORM)
        values[at("deploy")] = _clip(unit.deploy_ticks / tps / _WINDUP_NORM)
        values[at("load")] = _clip(unit.load_time_ticks / tps / _WINDUP_NORM)
        values[at("collision")] = _clip(
            unit.collision_radius / _SUBTILES / _COLLISION_NORM)
        values[at("air")] = float(unit.attacks_air)
        values[at("ground")] = float(unit.attacks_ground)
        values[at("flying")] = float(unit.flying)
        values[at("only_bldgs")] = float(unit.target_only_buildings)
        values[at("bldg_entity")] = float(unit.kind is EntityKind.BUILDING)
        values[at("kamikaze")] = float(unit.kamikaze)
        # The five flags that stop a hut reading as a harmless 0-DPS box.
        values[at("death_spawn")] = float(
            any(s.death_spawn_character for s in live))
        values[at("death_blast")] = float(any(
            s.death_damage or s.death_area_effect or s.death_spawn_projectile
            for s in live))
        values[at("spawner")] = float(any(s.spawn_character for s in live))
        values[at("charge")] = float(any(s.charge_range for s in live))
        # Doubles as "these stat columns are empty on purpose".
        values[at("action")] = float(any(s.on_starting_action for s in live))
        # What the hits do besides damage. Without these five, Ice Spirit
        # (freeze) and Heal Spirit (heal) are equal on every behavioural
        # column, and so are Electro Spirit and a plain chip-damage troop.
        values[at("on_hit_buff")] = float(bool(on_hit.get("buff")))
        values[at("on_hit_slow")] = _signed_clip(
            on_hit.get("speed_multiplier", 0) / _BUFF_MULTIPLIER_NORM)
        values[at("on_hit_hitspeed")] = _signed_clip(
            on_hit.get("hit_speed_multiplier", 0) / _BUFF_MULTIPLIER_NORM)
        values[at("on_hit_dps")] = _clip(
            on_hit.get("damage_per_second", 0) / _DPS_NORM)
        values[at("heals")] = _clip(
            on_hit.get("heal_per_second", 0) / _DPS_NORM)

    # -- block C: the spell or area payload --------------------------------
    payload = _payload_for(data, levels, spec_card, fuses)
    if payload:
        values[at("has_payload")] = 1.0
        values[at("p_damage")] = _clip(
            payload.get("damage", 0) / _PAYLOAD_DAMAGE_NORM)
        # Milli-tiles to tiles.
        values[at("p_radius")] = _clip(
            payload.get("radius", 0) / 1000.0 / _PAYLOAD_RADIUS_NORM)
        values[at("p_duration")] = _clip(
            payload.get("duration", 0) / 1000.0 / _PAYLOAD_DURATION_NORM)
        values[at("p_dps")] = _clip(
            payload.get("damage_per_second", 0) / _DPS_NORM)
        values[at("p_speed_mult")] = _signed_clip(
            payload.get("buff_speed_multiplier", 0) / _BUFF_MULTIPLIER_NORM)
        values[at("p_hitspeed_mult")] = _signed_clip(
            payload.get("buff_hit_speed_multiplier", 0) / _BUFF_MULTIPLIER_NORM)
        values[at("p_buff_time")] = _clip(
            payload.get("buff_time", 0) / 1000.0 / _PAYLOAD_BUFF_NORM)
        values[at("p_crown")] = _signed_clip(
            payload.get("crown_tower_damage_percent", 0) / 100.0)
        values[at("p_spawns")] = _clip(
            payload.get("spawns_count", 0) / _COUNT_NORM)
        values[at("p_action")] = float(bool(payload.get("action")))

    return tuple(values)


def card_feature_table(
    data: LogicData,
    levels: LevelTable,
    registry: CardRegistry,
    names: Sequence[str],
) -> tuple[tuple[float, ...], ...]:
    """One row per name, **in the order given**.

    The order is the contract. ``names`` must be the encoding config's own
    ``vocab``, because row ``i`` is what the head multiplies slot ``i``'s
    one-hot bit by -- and that bit is set by
    ``cr_sim.api.encoding._card_features`` from ``vocab.index(...)``. Build
    the table from anything else, in any other order, and the head is
    conditioned on the wrong card while still training to a lower loss.

    Raises ``KeyError`` for a name the registry does not have, rather than
    emitting a zero row: a deck naming a card that does not exist is a bug in
    the caller, and a zero row is a card the head believes is nothing.
    """
    return tuple(
        card_feature_vector(data, levels, registry, registry[name])
        for name in names)
