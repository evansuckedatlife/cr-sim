"""M5 gate: spells, area effects and buffs.

A spell is only worth anything if it can be put where the enemy is. That sounds
obvious, and it is exactly what was broken: placement ignored the card's own
``CanDeployOnEnemySide`` flag, so a Fireball could not be cast past the river --
the only place anyone would ever cast one -- and every spell landed on empty
grass in its owner's half.

The damage figures asserted here are the ones in ``reference/anchors.json``,
verified against the live game. They are end-to-end: cast the card, run the
battle, read the victim's hitpoints.
"""

from __future__ import annotations

import pytest

from cr_sim.data.cards import build_card_registry
from cr_sim.data.leveling import build_level_table
from cr_sim.data.source import LogicData
from cr_sim.engine.battle import Battle, BattleConfig
from cr_sim.engine.entity import Entity, EntityKind, Team
from cr_sim.engine.fixed import tiles, to_tiles
from cr_sim.engine.specs import build_unit_spec

from .test_data_pipeline import BUILD


@pytest.fixture(scope="module")
def world():
    data = LogicData.load(BUILD)
    return data, build_level_table(data), build_card_registry(data)


def _battle(world, card, *, towers=False):
    data, levels, registry = world
    battle = Battle(
        data, levels, registry,
        BattleConfig(seed=1, blue_deck=(card,) * 8, red_deck=("Knight",) * 8),
    )
    if not towers:
        battle.entities = [e for e in battle.entities if e.kind is not EntityKind.TOWER]
        battle._towers = {Team.BLUE: [], Team.RED: []}
    battle.players[Team.BLUE].elixir.add(10)
    return battle


def _victim(battle, world, x, y, *, hitpoints=99_999, unit="Knight"):
    data, levels, _registry = world
    spec = build_unit_spec(data, levels, unit, level=11, rarity="Common", clock=battle.clock)
    entity = Entity(
        kind=spec.kind, team=Team.RED, x=tiles(x), y=tiles(y),
        hitpoints=hitpoints, spec=spec,
        collision_radius=spec.collision_radius, mass=spec.mass, flying=spec.flying,
    )
    entity.max_hitpoints = hitpoints
    battle._register(entity)
    return entity


def _cast_and_measure(world, card, *, ticks=900, unit="Knight"):
    battle = _battle(world, card)
    victim = _victim(battle, world, 9, 12, unit=unit)
    assert battle.play_card(Team.BLUE, card, tiles(9), tiles(12)), f"{card} would not cast"
    for _ in range(ticks):
        battle.step()
    return victim.max_hitpoints - victim.hitpoints


# ---------------------------------------------------------------- placement


def test_spells_can_be_cast_on_the_enemy_half(world):
    """The bug this suite exists for: a Fireball must reach past the river."""
    for card in ("Fireball", "Zap", "Rocket", "Arrows", "Poison"):
        battle = _battle(world, card, towers=True)
        assert battle.play_card(Team.BLUE, card, tiles(9), tiles(25)), (
            f"{card} could not be cast on the enemy half"
        )


def test_troops_still_cannot_be_deployed_on_the_enemy_half(world):
    """Opening placement up for spells must not open it up for everything."""
    for card in ("Knight", "Musketeer", "Giant"):
        battle = _battle(world, card, towers=True)
        battle.players[Team.BLUE].elixir.add(10)
        assert not battle.play_card(Team.BLUE, card, tiles(9), tiles(25)), (
            f"{card} was deployed in enemy territory"
        )


def test_tunnelling_troops_may_be_deployed_anywhere(world):
    """Miner and Goblin Drill carry the same flag as spells, and should."""
    for card in ("Miner", "GoblinDrill"):
        battle = _battle(world, card, towers=True)
        battle.players[Team.BLUE].elixir.add(10)
        assert battle.play_card(Team.BLUE, card, tiles(9), tiles(25)), card


def test_spells_may_be_cast_over_the_river_but_troops_may_not(world):
    """Area spells cover water; nothing can stand on it."""
    battle = _battle(world, "Fireball", towers=True)
    assert battle.play_card(Team.BLUE, "Fireball", tiles(9), tiles(16))

    battle = _battle(world, "Knight", towers=True)
    battle.players[Team.BLUE].elixir.add(10)
    assert not battle.play_card(Team.BLUE, "Knight", tiles(9), tiles(16))


def test_the_log_cannot_be_cast_into_enemy_territory(world):
    """It rolls out from your side, so it is placed on your side.

    A card-by-card flag rather than a blanket "spells go anywhere" rule, which
    is why this one has to be checked separately.
    """
    battle = _battle(world, "Log", towers=True)
    assert not battle.play_card(Team.BLUE, "Log", tiles(9), tiles(25))
    assert battle.play_card(Team.BLUE, "Log", tiles(9), tiles(14))


# ------------------------------------------------------------------ damage


@pytest.mark.parametrize(
    "card,expected",
    [
        ("Zap", 192),
        ("Fireball", 688),
        ("Rocket", 1484),
        ("Snowball", 179),
        ("Log", 268),
        ("Freeze", 148),
        ("Arrows", 366),   # three waves of 122
        ("Poison", 736),   # 92 a second for eight seconds
    ],
)
def test_spell_damage_end_to_end(world, card, expected):
    """Cast it at a unit and count the damage. These are the anchored values."""
    assert _cast_and_measure(world, card) == expected


def test_arrows_damage_is_three_separate_volleys(world):
    """Not one hit of 366: three of 122, 200ms apart.

    The distinction is the whole reason Arrows' damage looks inconsistent -- a
    unit that leaves between volleys takes fewer of them.
    """
    battle = _battle(world, "Arrows")
    victim = _victim(battle, world, 9, 12)
    battle.play_card(Team.BLUE, "Arrows", tiles(9), tiles(12))
    for _ in range(900):
        battle.step()
    hits = [e for e in battle.damage_log if e.target_id == victim.id]
    assert len(hits) == 3, f"{len(hits)} volleys landed"
    assert {h.amount for h in hits} == {122}
    gaps = sorted(b.tick - a.tick for a, b in zip(hits, hits[1:]))
    assert all(g >= 10 for g in gaps), f"volleys were not separated: {gaps}"


def test_poison_ticks_once_a_second_for_eight_seconds(world):
    """736 total is 8 ticks of 92, not one lump."""
    battle = _battle(world, "Poison")
    victim = _victim(battle, world, 9, 12)
    battle.play_card(Team.BLUE, "Poison", tiles(9), tiles(12))
    for _ in range(900):
        battle.step()
    hits = [e for e in battle.damage_log if e.target_id == victim.id]
    assert len(hits) == 8, f"{len(hits)} poison ticks"
    assert {h.amount for h in hits} == {92}


# ------------------------------------------------------------------- buffs


def test_freeze_stops_a_unit_moving(world):
    """Freeze deals almost nothing; it buys time. That is the mechanic."""
    battle = _battle(world, "Freeze", towers=True)
    victim = _victim(battle, world, 9, 20, hitpoints=99_999)
    for _ in range(120):  # let it start walking
        battle.step()
    battle.play_card(Team.BLUE, "Freeze", victim.x, victim.y)
    for _ in range(20):
        battle.step()
    assert victim.buffs is not None and victim.buffs.is_frozen()

    held = (victim.x, victim.y)
    for _ in range(120):
        battle.step()
    assert (victim.x, victim.y) == held, "a frozen unit moved"


def test_poison_slows_as_well_as_damages(world):
    """Its -15 speed is a real part of the card, not just flavour."""
    battle = _battle(world, "Poison", towers=True)
    victim = _victim(battle, world, 9, 20, hitpoints=99_999)
    for _ in range(120):
        battle.step()
    battle.play_card(Team.BLUE, "Poison", victim.x, victim.y)
    for _ in range(30):
        battle.step()
    assert victim.buffs is not None
    assert victim.buffs.speed_multiplier() < 0, "Poison applied no slow"


def test_an_area_effect_only_touches_what_is_inside_it_now(world):
    """Walk out of a Poison cloud and the ticks stop.

    Membership is re-evaluated on every application rather than captured when
    the spell was cast; binding victims at cast time would turn every lingering
    spell into a delayed burst.
    """
    battle = _battle(world, "Poison")
    inside = _victim(battle, world, 9, 12)
    outside = _victim(battle, world, 9, 20)  # well clear of a 3.5 tile radius
    battle.play_card(Team.BLUE, "Poison", tiles(9), tiles(12))
    for _ in range(900):
        battle.step()
    assert inside.hitpoints < inside.max_hitpoints
    assert outside.hitpoints == outside.max_hitpoints


# --------------------------------------------------------------- lightning


def test_lightning_strikes_the_biggest_targets_and_not_the_same_one_twice(world):
    """Its bolts pick the largest units, one each.

    That is why Lightning answers a tank and its support and is wasted on a
    swarm. Striking one unit repeatedly would make it a single-target nuke.
    """
    battle = _battle(world, "Lightning")
    big = _victim(battle, world, 9, 12, hitpoints=9_999)
    mid = _victim(battle, world, 9.7, 12, hitpoints=5_000)
    small = _victim(battle, world, 10.4, 12, hitpoints=300, unit="Skeleton")

    battle.play_card(Team.BLUE, "Lightning", tiles(9.5), tiles(12))
    for _ in range(400):
        battle.step()

    assert big.hitpoints < big.max_hitpoints
    assert mid.hitpoints < mid.max_hitpoints
    # Each was struck once, not one of them repeatedly.
    for victim in (big, mid):
        hits = [e for e in battle.damage_log if e.target_id == victim.id]
        assert len(hits) == 1, f"struck {len(hits)} times"
    assert small.hitpoints < small.max_hitpoints
