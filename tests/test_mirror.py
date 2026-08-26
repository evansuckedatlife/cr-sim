"""Mirror.

Mirror is a reference, not a card. It has no units, no damage and no radius of
its own, and every number on its row describes the wrapper rather than what it
puts on the board: ``ManaCost`` is 1, which is what it would cost if it
deployed nothing, and ``CanDeployOnEnemySide`` is true, which is not true of
most of what it copies.

So the cost and the placement rules both have to come from the copied card. A
version that read Mirror's own row would let you drop a mirrored Golem in the
enemy half for one elixir.

The parts that *are* in the build are used: ``MIRROR_LEVEL_OFFSET`` is 1, so
the copy is a level higher, and ``OmitFromStartingHand`` keeps it out of the
opening hand -- with nothing yet played it would be a dead card.
"""

from __future__ import annotations

import pytest

from cr_sim.data.cards import build_card_registry
from cr_sim.data.leveling import build_level_table
from cr_sim.data.source import LogicData
from cr_sim.engine.battle import MIRROR_EXTRA_ELIXIR, Battle, BattleConfig
from cr_sim.engine.entity import EntityKind, Team
from cr_sim.engine.fixed import tiles

from .test_data_pipeline import BUILD

DECK = ("Mirror", "Knight", "Musketeer", "Cannon",
        "Log", "Fireball", "Goblins", "Skeletons")


@pytest.fixture(scope="module")
def world():
    data = LogicData.load(BUILD)
    return data, build_level_table(data), build_card_registry(data)


@pytest.fixture
def battle(world):
    data, levels, registry = world
    b = Battle(
        data, levels, registry,
        BattleConfig(seed=1, blue_deck=DECK, red_deck=("Knight",) * 8),
    )
    b.entities = [e for e in b.entities if e.kind is not EntityKind.TOWER]
    b._towers = {Team.BLUE: [], Team.RED: []}
    return b


def _force(battle, name):
    player = battle.players[Team.BLUE]
    player.elixir.add(10)
    player.cycle.remove(name)
    player.cycle.insert(0, name)


def _units(battle, name):
    return [e for e in battle.entities if not e.dead and e.spec is not None
            and e.spec.name == name]


# ---------------------------------------------------------------- the flags


def test_mirror_is_flagged_rather_than_special_cased_by_name(world):
    _data, _levels, registry = world
    mirror = registry.get("Mirror")
    assert mirror.is_mirror and mirror.omit_from_starting_hand
    assert not registry.get("Knight").is_mirror


def test_mirror_is_never_dealt_into_the_opening_hand(world):
    """With nothing played yet it has nothing to copy."""
    data, levels, registry = world
    for seed in range(30):
        b = Battle(
            data, levels, registry,
            BattleConfig(seed=seed, blue_deck=DECK, red_deck=DECK),
        )
        assert "Mirror" not in b.players[Team.BLUE].hand, f"seed {seed}"


# ----------------------------------------------------------------- the copy


def test_mirror_with_nothing_to_copy_is_refused(battle):
    _force(battle, "Mirror")
    assert not battle.play_card(Team.BLUE, "Mirror", tiles(9), tiles(12))


def test_mirror_replays_the_last_card(battle):
    _force(battle, "Knight")
    assert battle.play_card(Team.BLUE, "Knight", tiles(9), tiles(12))
    for _ in range(90):
        battle.step()
    assert len(_units(battle, "Knight")) == 1

    _force(battle, "Mirror")
    assert battle.play_card(Team.BLUE, "Mirror", tiles(11), tiles(12))
    for _ in range(90):
        battle.step()
    assert len(_units(battle, "Knight")) == 2


def test_the_copy_is_a_level_higher(battle):
    """MIRROR_LEVEL_OFFSET is 1, and it is the whole reason to play the card."""
    _force(battle, "Knight")
    battle.play_card(Team.BLUE, "Knight", tiles(9), tiles(12))
    for _ in range(90):
        battle.step()
    original = _units(battle, "Knight")[0]

    _force(battle, "Mirror")
    battle.play_card(Team.BLUE, "Mirror", tiles(11), tiles(12))
    for _ in range(90):
        battle.step()
    copy = next(e for e in _units(battle, "Knight") if e.id != original.id)
    assert copy.spec.hitpoints > original.spec.hitpoints
    assert copy.spec.damage > original.spec.damage


def test_mirror_costs_the_copied_card_plus_one(battle):
    """Not its own ManaCost of 1, which is what it would cost to deploy nothing."""
    _force(battle, "Knight")
    battle.play_card(Team.BLUE, "Knight", tiles(9), tiles(12))
    knight = battle.registry.get("Knight")

    _force(battle, "Mirror")
    before = battle.players[Team.BLUE].elixir.units
    assert battle.play_card(Team.BLUE, "Mirror", tiles(11), tiles(12))
    spent = before - battle.players[Team.BLUE].elixir.units
    assert spent == knight.mana_cost + MIRROR_EXTRA_ELIXIR


def test_mirror_is_refused_when_the_copy_is_unaffordable(battle):
    """The price comes from the copy, so affordability has to as well."""
    _force(battle, "Knight")
    battle.play_card(Team.BLUE, "Knight", tiles(9), tiles(12))

    player = battle.players[Team.BLUE]
    player.cycle.remove("Mirror")
    player.cycle.insert(0, "Mirror")
    player.elixir.amount = 0
    assert not battle.play_card(Team.BLUE, "Mirror", tiles(11), tiles(12))


def test_mirroring_does_not_become_the_next_thing_to_mirror(battle):
    """Otherwise a second Mirror would copy the first and copy nothing."""
    _force(battle, "Knight")
    battle.play_card(Team.BLUE, "Knight", tiles(9), tiles(12))
    _force(battle, "Mirror")
    battle.play_card(Team.BLUE, "Mirror", tiles(11), tiles(12))
    assert battle.players[Team.BLUE].last_played == "Knight"

    _force(battle, "Mirror")
    assert battle.play_card(Team.BLUE, "Mirror", tiles(13), tiles(12))
    for _ in range(90):
        battle.step()
    assert len(_units(battle, "Knight")) == 3


def test_a_mirrored_card_keeps_its_own_placement_rules(battle):
    """Mirror's row says it may be played anywhere. What it copies usually may not.

    Reading the wrapper's rules would let a mirrored Knight be dropped in the
    enemy half for four elixir.
    """
    _force(battle, "Knight")
    battle.play_card(Team.BLUE, "Knight", tiles(9), tiles(12))
    _force(battle, "Mirror")
    assert not battle.play_card(Team.BLUE, "Mirror", tiles(9), tiles(26))


# ------------------------------------------------------------- variant cards


def test_merge_maiden_is_read_as_a_variant_card(world):
    """Its cost is not on the card. It is the trigger of the form afforded.

    ``Options`` lists a mounted form at six elixir and a foot form at three,
    so what the card *is* depends on what you can pay when you play it.
    """
    _data, _levels, registry = world
    card = registry.get("MergeMaiden")
    assert card.variants == ((6, "MergeMaiden_Mounted"), (3, "MergeMaiden_Normal"))
    # Richest first, so resolution is "the best one you can afford" rather
    # than a search.
    costs = [c for c, _ in card.variants]
    assert costs == sorted(costs, reverse=True)


def test_no_other_card_has_variants(world):
    """Pinned so a second one shows up as a failure rather than a mystery."""
    _data, _levels, registry = world
    assert [c.name for c in registry.cards if c.variants] == ["MergeMaiden"]


@pytest.mark.parametrize(
    "elixir,expected_cost,form",
    [(9, 6, "MergeMaiden_Mounted"), (4, 3, "MergeMaiden_Normal")],
)
def test_the_form_follows_the_elixir(world, elixir, expected_cost, form):
    data, levels, registry = world
    deck = ("MergeMaiden", "Knight", "Musketeer", "Cannon",
            "Log", "Fireball", "Goblins", "Skeletons")
    b = Battle(
        data, levels, registry,
        BattleConfig(seed=1, blue_deck=deck, red_deck=("Knight",) * 8),
    )
    b.entities = [e for e in b.entities if e.kind is not EntityKind.TOWER]
    b._towers = {Team.BLUE: [], Team.RED: []}
    player = b.players[Team.BLUE]
    player.cycle.remove("MergeMaiden")
    player.cycle.insert(0, "MergeMaiden")
    player.elixir.amount = 0
    player.elixir.add(elixir)

    before = player.elixir.units
    assert b.play_card(Team.BLUE, "MergeMaiden", tiles(9), tiles(12))
    assert before - player.elixir.units == expected_cost
    for _ in range(120):
        b.step()
    deployed = [e.spec.name for e in b.entities
                if not e.dead and e.spec is not None and "Maiden" in e.spec.name]
    assert deployed == [form]


def test_a_variant_card_is_refused_below_its_cheapest_form(world):
    data, levels, registry = world
    deck = ("MergeMaiden", "Knight", "Musketeer", "Cannon",
            "Log", "Fireball", "Goblins", "Skeletons")
    b = Battle(
        data, levels, registry,
        BattleConfig(seed=1, blue_deck=deck, red_deck=("Knight",) * 8),
    )
    player = b.players[Team.BLUE]
    player.cycle.remove("MergeMaiden")
    player.cycle.insert(0, "MergeMaiden")
    player.elixir.amount = 0
    player.elixir.add(2)
    assert not b.play_card(Team.BLUE, "MergeMaiden", tiles(9), tiles(12))
