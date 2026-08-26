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


# ------------------------------------------------------ run action at health


def _placed_knight(battle, hitpoints=1000):
    """A plain Knight dropped straight on the board, for exercising a
    synthetic ``ActionRunActionAtHealth`` node against a known hitpoints
    total rather than whatever a levelled real card happens to have."""
    from cr_sim.engine.entity import Entity
    from cr_sim.engine.specs import build_unit_spec

    spec = build_unit_spec(
        battle.data, battle.levels, "Knight", level=11, rarity="Common", clock=battle.clock
    )
    unit = Entity(
        kind=spec.kind, team=Team.BLUE, x=tiles(9), y=tiles(12), hitpoints=hitpoints,
        spec=spec, collision_radius=spec.collision_radius, mass=spec.mass, flying=spec.flying,
    )
    unit.max_hitpoints = hitpoints
    unit.deploy_ticks_left = 0
    battle._register(unit)
    return unit


def test_action_run_action_at_health_does_not_fire_above_its_threshold(world):
    """The primitive behind GoblinGiant's evolution, MovingCannon and Goblin
    Demolisher, driven directly rather than through any one card's chain.

    A full-health unit must not trigger a 50%-health action, and the watch
    itself must not report the node as unsupported -- both the graph node and
    the polling it does are load-bearing, not decorative.
    """
    battle = _battle(world, "Knight")
    unit = _placed_knight(battle)
    inline = {
        "ClassType": "ActionRunActionAtHealth",
        "Actions": [{"ClassType": "ActionKill"}],
        "HealthPercentages": [50],
    }
    ctx = ActionContext(team=Team.BLUE, x=unit.x, y=unit.y, source=unit)
    battle.actions.run(inline, ctx, battle.tick)
    for _ in range(60):
        battle.step()
    assert not unit.dead, "fired without the hitpoints ever crossing the threshold"
    assert dict(battle.actions.unsupported) == {}


def test_action_run_action_at_health_fires_a_threshold_at_most_once(world):
    """Crossing the same threshold repeatedly must not refire it.

    Real combat does not hold hitpoints still at exactly the cutoff -- a unit
    takes a series of separate hits that can nick it below 50% more than
    once. ``ActionRunActionAtHealth`` has to treat the threshold as "has this
    ever been true", not "is this true on this particular tick", or a unit
    that takes several hits around the line would run its transformation
    several times over.
    """
    battle = _battle(world, "Knight")
    unit = _placed_knight(battle)
    inline = {
        "ClassType": "ActionRunActionAtHealth",
        "Actions": [{"ClassType": "ActionSpawn", "SpawnType": "CharacterType",
                     "SpawnData": "Skeleton"}],
        "HealthPercentages": [50],
    }
    ctx = ActionContext(team=Team.BLUE, x=unit.x, y=unit.y, source=unit)
    battle.actions.run(inline, ctx, battle.tick)

    for _ in range(4):
        unit.hitpoints = 400  # 40%: below the line
        for _ in range(5):
            battle.step()
        unit.hitpoints = 900  # 90%: back above it
        for _ in range(5):
            battle.step()

    skeletons = [e for e in battle.entities if e.spec is not None and e.spec.name == "Skeleton"]
    assert len(skeletons) == 1, f"{len(skeletons)} skeletons for one threshold, crossed four times"


def test_action_run_action_at_health_fires_each_listed_threshold_independently(world):
    """``Actions`` and ``HealthPercentages`` are parallel arrays, not a single
    pair -- MovingCannon and the Goblin Giant evolution both list several
    actions that should fire at their own percentage, not all together.
    """
    battle = _battle(world, "Knight")
    unit = _placed_knight(battle)
    inline = {
        "ClassType": "ActionRunActionAtHealth",
        "Actions": [
            {"ClassType": "ActionSetVariable", "Variable": "crossed_high", "Value": 1},
            {"ClassType": "ActionSetVariable", "Variable": "crossed_low", "Value": 1},
        ],
        "HealthPercentages": [75, 10],
    }
    ctx = ActionContext(team=Team.BLUE, x=unit.x, y=unit.y, source=unit)
    battle.actions.run(inline, ctx, battle.tick)

    unit.hitpoints = 500  # 50%: past the 75% line, nowhere near the 10% one
    for _ in range(10):
        battle.step()
    assert ctx.variables.get("crossed_high") == 1
    assert "crossed_low" not in ctx.variables

    unit.hitpoints = 50  # 5%: past both now
    for _ in range(10):
        battle.step()
    assert ctx.variables.get("crossed_low") == 1


def test_goblin_giant_evolution_drops_no_goblins_above_half_health(world):
    """The evolution's whole spawn cycle is gated on the health watcher.

    ``GoblinGiant_EV1``'s two ``SpearGoblinGiant`` riders come from its
    ordinary ``SpawnCharacter``/``SpawnNumber`` fields and arrive on deploy
    regardless; the free Goblin trickle is the evolution's actual new
    behaviour, and it must not start before the giant is hurt.
    """
    battle = _battle(world, "GoblinGiant_EV1")
    assert battle.play_card(Team.BLUE, "GoblinGiant_EV1", tiles(9), tiles(12))
    for _ in range(300):
        battle.step()
    assert not [
        e for e in battle.entities if e.spec is not None and e.spec.name == "Goblin"
    ], "goblins dropped before the giant crossed 50% health"
    assert dict(battle.actions.unsupported) == {}


def test_goblin_giant_evolution_drops_goblins_once_it_crosses_half_health(world):
    """Its ``OnStartingAction`` is ``GoblinGiant_EV1_trigger_at_health``: at
    50% hitpoints it starts an ``ActionInterval`` that spawns a Goblin every
    2200ms for the rest of the fight. There is no other path in this build
    that puts Goblins on the board for this evolution.
    """
    battle = _battle(world, "GoblinGiant_EV1")
    assert battle.play_card(Team.BLUE, "GoblinGiant_EV1", tiles(9), tiles(12))
    for _ in range(60):
        battle.step()
    giant = next(
        e for e in battle.entities if e.spec is not None and e.spec.name == "GoblinGiant"
    )
    giant.hitpoints = giant.max_hitpoints // 4

    goblins = _arrivals(battle, "Goblin", 700)
    assert len(goblins) >= 2, f"only {len(goblins)} goblins from the evolution's trickle"
    assert dict(battle.actions.unsupported) == {}


# ---------------------------------------------------------------- furnace


def test_firespirit_hut_launches_its_spirits_ahead_of_itself_not_on_top_of_it(world):
    """``Furnace_rework_spawn_forward`` carries ``MirroredY = 3``: three whole
    tiles, not three milli-tiles.

    ``RelativeX``/``RelativeY`` and ``Mirrored X``/``Y`` are plain integers
    rather than an expression, and unlike ``XPositionExpression`` they are not
    written in the file's usual milli-tile unit -- Fire Spirits visibly launch
    a few tiles clear of the building, not overlapping its own footprint. Read
    as milli-tiles the offset rounds away to nothing and every spirit spawns
    stacked on the Furnace instead.
    """
    battle = _battle(world, "FirespiritHut")
    assert battle.play_card(Team.BLUE, "FirespiritHut", tiles(9), tiles(12))
    for _ in range(180):
        battle.step()

    hut = next(
        e for e in battle.entities if e.spec is not None and e.spec.name == "Furnace_rework"
    )
    spirits = [e for e in battle.entities if e.spec is not None and e.spec.name == "FireSpirits"]
    assert spirits, "the Furnace never launched a Fire Spirit"
    for spirit in spirits:
        gap = math.hypot(to_tiles(spirit.x) - to_tiles(hut.x), to_tiles(spirit.y) - to_tiles(hut.y))
        assert gap > 1.5, f"a Fire Spirit landed {gap:.2f} tiles from the Furnace"
    assert dict(battle.actions.unsupported) == {}


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
        "ActionRunActionListOnObjectsInShapeWithPrio",  # needs the Shape definitions
        "ActionSkeletonBarrelPopBalloon",            # the Skeleton Balloon evolution
        "ActionTargetIndicatorAttack",               # Goblin Machine's second attack
        # GoblinDemolisher and MovingCannon both trigger a full character
        # transformation at 50% health (ActionRunActionAtHealth, implemented)
        # into a different CHARACTER definition entirely -- new stats, new
        # targeting rules, in MovingCannon's case a different EntityKind. This
        # scenario deals no damage, so it never reaches that threshold; see
        # test_goblin_demolisher_attempts_its_transformation_at_half_health for
        # where it is exercised and what remains unimplemented.
        "ActionChangeGameObjectData",
        "ActionTaunt",                               # nested under the same transformation
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


# ------------------------------------------------------- inline actions


def test_an_action_may_be_written_inline_rather_than_referenced(world):
    """Nineteen actions in this build are objects, not names.

    Goblin Curse's buffs and Dark Magic's whole effect are among them, and a
    reader that only accepted names dropped every one silently.
    """
    battle = _battle(world, "GoblinCurse")
    inline = {
        "ClassType": "ActionGroup",
        "SubActionsDelay": [0],
        "SubActions": [{"ClassType": "ActionSpawn", "SpawnType": "CharacterType",
                        "SpawnData": "Skeleton"}],
    }
    before = len(battle.entities)
    battle.actions.run(
        inline, ActionContext(team=Team.BLUE, x=tiles(9), y=tiles(12)), battle.tick
    )
    for _ in range(120):
        battle.step()
    assert len(battle.entities) > before
    assert dict(battle.actions.unsupported) == {}


# --------------------------------------------------------- goblin curse


def _cursed(world, hitpoints=99_999):
    from cr_sim.engine.entity import Entity
    from cr_sim.engine.specs import build_unit_spec

    data, levels, _ = world
    battle = _battle(world, "GoblinCurse")
    player = battle.players[Team.BLUE]
    player.elixir.add(10)
    player.cycle.remove("GoblinCurse")
    player.cycle.insert(0, "GoblinCurse")

    spec = build_unit_spec(data, levels, "Knight", level=11, rarity="Common",
                           clock=battle.clock)
    foe = Entity(
        kind=spec.kind, team=Team.RED, x=tiles(9), y=tiles(12), hitpoints=hitpoints,
        spec=spec, collision_radius=spec.collision_radius, mass=spec.mass,
        flying=spec.flying,
    )
    foe.max_hitpoints = hitpoints
    foe.deploy_ticks_left = 0
    battle._register(foe)
    assert battle.play_card(Team.BLUE, "GoblinCurse", tiles(9), tiles(12))
    return battle, foe


def test_goblin_curse_applies_its_curse_and_its_damage(world):
    """The cloud hits nothing itself -- HitsAir and HitsGround are both false.

    Its whole effect is an action chain: place a second area effect, and have
    that put two buffs on what it touches.
    """
    battle, foe = _cursed(world)
    for _ in range(300):
        battle.step()
    assert foe.buffs is not None
    names = set(foe.buffs.active_names())
    assert {"GoblinCurse", "GoblinCurseDamage"} <= names, names
    assert foe.hitpoints < foe.max_hitpoints, "the curse dealt no damage"
    assert foe.buffs.speed_multiplier() < 0, "the curse did not slow"


def test_what_dies_under_the_curse_comes_back_on_your_side(world):
    """The card's entire point, and it is carried by the buff rather than by
    the dying unit -- DeathSpawnIsEnemy means the spawn belongs to whoever
    applied it, not to whoever died."""
    battle, foe = _cursed(world, hitpoints=400)
    for _ in range(120):
        battle.step()
    foe.hitpoints = 1
    foe.apply_damage(1)
    for _ in range(120):
        battle.step()

    goblins = [
        e for e in battle.entities
        if not e.dead and e.spec is not None and e.spec.name == "Goblin"
    ]
    assert len(goblins) == 1, "nothing came back"
    assert goblins[0].team is Team.BLUE, "it came back on the wrong side"


def test_an_uncursed_death_leaves_nothing_behind(world):
    """So the spawn is the curse's doing and not the Knight's."""
    from cr_sim.engine.entity import Entity
    from cr_sim.engine.specs import build_unit_spec

    data, levels, _ = world
    battle = _battle(world, "GoblinCurse")
    spec = build_unit_spec(data, levels, "Knight", level=11, rarity="Common",
                           clock=battle.clock)
    foe = Entity(
        kind=spec.kind, team=Team.RED, x=tiles(9), y=tiles(12), hitpoints=400,
        spec=spec, collision_radius=spec.collision_radius, mass=spec.mass,
        flying=spec.flying,
    )
    foe.max_hitpoints = 400
    foe.deploy_ticks_left = 0
    battle._register(foe)
    foe.hitpoints = 1
    foe.apply_damage(1)
    for _ in range(120):
        battle.step()
    assert not [e for e in battle.entities
                if not e.dead and e.spec is not None and e.spec.name == "Goblin"]
