"""M6: the ACTION graph interpreter.

This build keeps its modern behaviour in a scripting language rather than in
stat columns. The reworked Graveyard's ``AEO`` row carries a radius and a
lifetime and nothing else; the reworked Goblin Hut's ``SpawnCharacter`` is the
empty string. Both cards do exactly nothing without an interpreter, and no
amount of reading their stat fields would tell you otherwise -- which is why
the coverage assertion at the bottom of this file matters more than any single
case: it is what stops a card going quietly inert.
"""

from __future__ import annotations

import math

import pytest

from cr_sim.data.cards import build_card_registry
from cr_sim.data.leveling import build_level_table
from cr_sim.data.source import LogicData
from cr_sim.engine.actions import (
    ActionContext,
    ExpressionError,
    evaluate_expression,
)
from cr_sim.engine.battle import CLONE_HITPOINTS, Battle, BattleConfig
from cr_sim.engine.entity import EntityKind, Team
from cr_sim.engine.fixed import tiles, to_tiles

from .test_data_pipeline import BUILD


@pytest.fixture(scope="module")
def world():
    data = LogicData.load(BUILD)
    return data, build_level_table(data), build_card_registry(data)


def _scope(team=Team.BLUE, x=9.0, y=20.0):
    return ActionContext(team=team, x=tiles(x), y=tiles(y)).expression_scope()


def _battle(world, card):
    data, levels, registry = world
    battle = Battle(
        data, levels, registry,
        BattleConfig(seed=5, blue_deck=(card,) * 8, red_deck=("Knight",) * 8),
    )
    battle.entities = [e for e in battle.entities if e.kind is not EntityKind.TOWER]
    battle._towers = {Team.BLUE: [], Team.RED: []}
    battle.players[Team.BLUE].elixir.add(10)
    return battle


def _arrivals(battle, name, ticks):
    """Ticks on which a new entity of this name appeared."""
    seen, out = set(), []
    for tick in range(ticks):
        battle.step()
        for entity in battle.entities:
            if entity.spec is not None and name in entity.spec.name and entity.id not in seen:
                seen.add(entity.id)
                out.append(tick)
    return out


# ------------------------------------------------------------- expressions


def test_arithmetic_follows_normal_precedence():
    assert evaluate_expression("1 + 2 * 3", {}) == 7
    assert evaluate_expression("(1 + 2) * 3", {}) == 9


def test_c_style_operators_are_understood():
    """The files use ``&&``, ``||`` and ``!``, which Python does not."""
    scope = {"a": True, "b": False}
    assert evaluate_expression("a && !b", scope) is True
    assert evaluate_expression("b || a", scope) is True
    assert evaluate_expression("!(1 == 2)", scope) is True
    assert evaluate_expression("1 != 2", scope) is True


def test_select_is_the_ternary():
    assert evaluate_expression("select(1 > 0, 5, 9)", {}) == 5
    assert evaluate_expression("select(1 < 0, 5, 9)", {}) == 9


def test_team_y_direction_mirrors_for_red():
    """An offset authored from blue's point of view has to work for red.

    Without the mirror every action would place its spawns behind a red caster
    instead of in front of it.
    """
    expression = "y - (3500 * team_y_direction(team_index))"
    blue = evaluate_expression(expression, _scope(team=Team.BLUE))
    red = evaluate_expression(expression, _scope(team=Team.RED))
    assert blue == 20_000 - 3500
    assert red == 20_000 + 3500


def test_position_offsets_mirror_across_the_map():
    """``select(x > map_width / 2, -1, 1)`` flips an offset by lane."""
    expression = "x + (-2500 * select(x > (map_width / 2), -1, 1))"
    left = evaluate_expression(expression, _scope(x=4.0))
    right = evaluate_expression(expression, _scope(x=14.0))
    assert left == 4000 - 2500
    assert right == 14_000 + 2500


def test_an_unknown_name_is_an_error_rather_than_zero():
    """A mistyped or newly added function must surface, not evaluate to nothing.

    Defaulting an unreadable name to zero is how a spell silently starts
    landing at the origin.
    """
    with pytest.raises(ExpressionError):
        evaluate_expression("mystery_function(1)", {})
    with pytest.raises(ExpressionError):
        evaluate_expression("undefined_name + 1", {})


def test_expressions_cannot_execute_code():
    """These strings come out of a data file and are never handed to eval."""
    for hostile in ("__import__('os').system('echo hi')", "[].__class__", "lambda: 1"):
        with pytest.raises(ExpressionError):
            evaluate_expression(hostile, {})


# -------------------------------------------------------------- graveyard


def test_graveyard_spawns_twelve_skeletons_on_its_authored_schedule(world):
    """2200, 2700, 3300 ... 8200ms, straight out of ``SubActionsDelay``.

    There is no rate field anywhere that produces this. The card's whole
    character -- a trickle you have to answer over time rather than a burst you
    can swat once -- is in that array.
    """
    battle = _battle(world, "Graveyard")
    assert battle.play_card(Team.BLUE, "Graveyard", tiles(9), tiles(20))
    arrivals = _arrivals(battle, "Skeleton", 700)

    assert len(arrivals) == 12, f"{len(arrivals)} skeletons"
    expected = [2200, 2700, 3300, 3800, 4400, 4900, 5500, 6000, 6500, 7100, 7600, 8200]
    for got, want_ms in zip(arrivals, expected):
        assert abs(got - want_ms * 60 // 1000) <= 2, f"{got} ticks vs {want_ms}ms"


def test_graveyard_skeletons_ring_the_cast_point(world):
    """They surround a tower rather than stacking on the tile you tapped.

    The ring is only in the position expressions -- 2500 and 3500 milli-tile
    offsets mirrored by lane and team. A spawn at the bare cast point would put
    all twelve inside one splash.
    """
    battle = _battle(world, "Graveyard")
    battle.play_card(Team.BLUE, "Graveyard", tiles(9), tiles(20))
    for _ in range(700):
        battle.step()

    skeletons = [
        e for e in battle.entities + battle.graveyard
        if e.spec is not None and e.spec.name == "Skeleton"
    ]
    assert skeletons
    spread = [
        math.hypot(to_tiles(e.x) - 9.0, to_tiles(e.y) - 20.0) for e in skeletons
    ]
    assert min(spread) > 2.0, "skeletons landed on the cast point"
    assert max(spread) < 5.0, "skeletons landed outside the card's radius"


def test_graveyard_needs_no_unsupported_nodes(world):
    battle = _battle(world, "Graveyard")
    battle.play_card(Team.BLUE, "Graveyard", tiles(9), tiles(20))
    for _ in range(700):
        battle.step()
    assert dict(battle.actions.unsupported) == {}


# ------------------------------------------------------------------- huts


def test_the_reworked_goblin_hut_produces_spear_goblins(world):
    """Its stat columns are empty; the cycle is entirely an OnStartingAction.

    ``BUILDING.GoblinHut_Rework`` carries ``SpawnCharacter`` as the empty
    string, so the ordinary spawner path finds nothing and the building sits
    inert for its whole 30-second life.
    """
    battle = _battle(world, "GoblinHut")
    assert battle.play_card(Team.BLUE, "GoblinHut", tiles(9), tiles(12))
    arrivals = _arrivals(battle, "SpearGoblin", 60 * 40)

    assert len(arrivals) >= 10, f"only {len(arrivals)} spear goblins"
    # ActionDelay 1000 before the first, then SpawnInterval 2200.
    assert 55 <= arrivals[0] <= 70, f"first at {arrivals[0]} ticks, expected ~60"
    gaps = [b - a for a, b in zip(arrivals, arrivals[1:])]
    assert all(128 <= g <= 137 for g in gaps[:5]), f"gaps were {gaps[:5]}"


def test_a_destroyed_spawner_stops_producing(world):
    """The queue is abandoned when its instigator dies.

    A self-rescheduling spawner whose source is gone would otherwise keep
    producing for the rest of the match out of a building that is not there.
    """
    battle = _battle(world, "GoblinHut")
    battle.play_card(Team.BLUE, "GoblinHut", tiles(9), tiles(12))
    for _ in range(60 * 8):
        battle.step()

    hut = next(e for e in battle.entities if e.kind is EntityKind.BUILDING)
    hut.hitpoints = 1
    hut.apply_damage(1)
    for _ in range(30):
        battle.step()

    before = len([
        e for e in battle.entities + battle.graveyard
        if e.spec is not None and "SpearGoblin" in e.spec.name
    ])
    for _ in range(60 * 12):
        battle.step()
    after = len([
        e for e in battle.entities + battle.graveyard
        if e.spec is not None and "SpearGoblin" in e.spec.name
    ])
    assert after == before, "a dead hut kept spawning"


# --------------------------------------------------------------- coverage


def test_no_playable_card_hits_an_unsupported_action_node(world):
    """The coverage gate.

    A card whose behaviour lives in an unimplemented node type does nothing at
    all, and does it silently. This walks the whole playable pool and fails
    with the node names rather than letting a card go quietly inert.

    The known remainder is listed explicitly: these are one-off classes for
    single cards, each of which is its own piece of work, and the point of
    pinning them is that the list only ever shrinks.
    """
    data, levels, registry = world
    known = {
        "ActionCounter",                             # Ronin's parry
        "ActionGiantBufferCollectFriends",           # Giant Buffer, an event card
        "ActionRunActionAtHealth",                   # fires below a health threshold
        "ActionRunActionListOnObjectsInShapeWithPrio",  # needs the Shape definitions
        "ActionSkeletonBarrelPopBalloon",            # the Skeleton Balloon evolution
        "ActionTargetIndicatorAttack",               # Goblin Machine's second attack
    }

    seen: set[str] = set()
    for card in registry.standard():
        battle = Battle(
            data, levels, registry,
            BattleConfig(seed=1, blue_deck=(card.name,) * 8, red_deck=("Knight",) * 8),
        )
        battle.players[Team.BLUE].elixir.add(10)
        if not battle.play_card(Team.BLUE, card.name, tiles(9), tiles(12)):
            battle.play_card(Team.BLUE, card.name, tiles(9), tiles(20))
        for _ in range(240):
            battle.step()
        seen.update(battle.actions.unsupported)

    assert seen <= known, f"new unsupported action nodes: {sorted(seen - known)}"


# ------------------------------------------------------------------- clone


def test_clone_duplicates_a_friendly_troop(world):
    """The spell acts per unit, not at a point.

    It arrives through the area effect's ``OnHitAction`` with the touched unit
    as its source, which is the only way a spell that duplicates *each* troop
    it covers can work. Running it once at the centre would duplicate nothing.
    """
    # The unit is placed directly rather than played: only four of a deck's
    # eight cards are in hand at a time, so playing a specific one depends on
    # the shuffle. What is under test is the cloning, not the draw.
    battle = _battle(world, "Clone")
    battle._spawn_units(
        team=Team.BLUE, character="DarkPrince", count=1,
        x=tiles(9), y=tiles(12), rarity="Epic",
    )
    for _ in range(120):
        battle.step()

    battle.players[Team.BLUE].elixir.add(10)
    assert battle.play_card(Team.BLUE, "Clone", tiles(9), tiles(12))
    for _ in range(90):
        battle.step()

    princes = [
        e for e in battle.entities
        if e.spec is not None and e.spec.name == "DarkPrince" and not e.dead
    ]
    assert len(princes) == 2, f"{len(princes)} Dark Princes after a Clone"

    clone = next(e for e in princes if e.is_clone)
    original = next(e for e in princes if not e.is_clone)
    assert clone.hitpoints == CLONE_HITPOINTS
    # The shield is preserved and the body is not, which is the entire reason
    # cloning a Dark Prince is worth three elixir.
    assert clone.shield == original.shield > 0
    assert (clone.x, clone.y) != (original.x, original.y), "the clone was stacked"


def test_a_clone_cannot_itself_be_cloned(world):
    """CLONE_CLONED_UNITS is False.

    Two 3-elixir spells would otherwise quadruple a push rather than double it.
    """
    battle = _battle(world, "Clone")
    battle._spawn_units(
        team=Team.BLUE, character="Knight", count=1,
        x=tiles(9), y=tiles(12), rarity="Common",
    )
    for _ in range(120):
        battle.step()

    for _ in range(2):
        battle.players[Team.BLUE].elixir.add(10)
        battle.play_card(Team.BLUE, "Clone", tiles(9), tiles(12))
        for _ in range(90):
            battle.step()

    knights = [
        e for e in battle.entities
        if e.spec is not None and e.spec.name == "Knight" and not e.dead
    ]
    assert len(knights) == 3, f"{len(knights)} Knights; two Clones should give three"
