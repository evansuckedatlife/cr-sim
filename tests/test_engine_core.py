"""M1 gate: the deterministic core.

The claim this module defends is that a battle is a pure function of
``(seed, configuration, commands)``. Everything here exists to make that
falsifiable: exact fixed-point arithmetic, a reproducible RNG, tick conversion
that agrees across tick rates, and a state hash sensitive enough to catch a
single subtile of drift.
"""

from __future__ import annotations

import pytest

from cr_sim.data.cards import build_card_registry
from cr_sim.data.leveling import build_level_table
from cr_sim.data.source import LogicData
from cr_sim.engine.constants import TRAINING_TPS, VERIFICATION_TPS, TickClock
from cr_sim.engine.elixir import ElixirBar, build_timeline
from cr_sim.engine.entity import (
    Entity,
    EntityKind,
    EntityState,
    Team,
    reset_entity_ids,
)
from cr_sim.engine.fixed import (
    SUBTILES_PER_MILLI_TILE,
    SUBTILES_PER_TILE,
    distance,
    half_tiles,
    milli_tiles,
    point_along,
    tiles,
    to_tiles,
    within_range,
)
from cr_sim.engine.rng import Rng
from cr_sim.engine.specs import SpecError, spec_for_card
from cr_sim.replay import Command, Replay, compare_hashes, state_hash

from .test_data_pipeline import BUILD


@pytest.fixture(scope="module")
def data():
    return LogicData.load(BUILD)


@pytest.fixture(scope="module")
def levels(data):
    return build_level_table(data)


@pytest.fixture(scope="module")
def registry(data):
    return build_card_registry(data)


# ------------------------------------------------------------- fixed point


def test_subtile_makes_both_game_units_exact():
    """The choice of 18000 is what keeps every conversion integral."""
    assert SUBTILES_PER_TILE % 1000 == 0
    assert SUBTILES_PER_MILLI_TILE * 1000 == SUBTILES_PER_TILE
    # A milli-tile value from the files converts with no remainder.
    assert milli_tiles(1200) == 21_600
    assert to_tiles(milli_tiles(1200)) == 1.2
    # spawn_groups.toml half-tiles convert with no remainder too.
    assert half_tiles(7) == 63_000


@pytest.mark.parametrize("tps", [VERIFICATION_TPS, TRAINING_TPS])
@pytest.mark.parametrize("speed", [45, 60, 90, 120])
def test_every_speed_tier_is_exact_at_both_tick_rates(tps, speed):
    """No speed may leave a remainder, at either rate, or units drift."""
    clock = TickClock(tps)
    assert clock.is_exact_for(speed), f"speed {speed} is lossy at {tps} TPS"
    per_tick = clock.subtiles_per_tick(speed)
    # A minute of ticks must cover exactly `speed` tiles.
    assert per_tick * 60 * tps == speed * SUBTILES_PER_TILE


def test_position_is_derived_not_accumulated():
    """Per-tick accumulation drifts; deriving from a running total does not.

    Adding a truncated step vector each tick loses up to a subtile per tick, so
    a unit walking a diagonal for ten seconds ends up measurably short. The
    error here must stay bounded regardless of how long the walk is.
    """
    ax, ay = 0, 0
    bx, by = tiles(10), tiles(3)
    segment = distance(ax, ay, bx, by)
    step = 300  # a Medium unit at 60 TPS

    naive_x = naive_y = 0
    travelled = 0
    for _ in range(600):
        travelled += step
        naive_x += (bx - ax) * step // segment
        naive_y += (by - ay) * step // segment

    exact_x, exact_y = point_along(ax, ay, bx, by, travelled, segment)
    naive_drift = abs(exact_x - naive_x) + abs(exact_y - naive_y)
    assert naive_drift > 100, "expected the naive approach to visibly drift"

    # The derived position stays within one subtile per axis of the true point.
    ratio_x = (bx - ax) * travelled / segment
    assert abs(exact_x - ratio_x) <= 1


def test_point_along_clamps_at_the_endpoints():
    ax, ay, bx, by = 0, 0, tiles(4), 0
    segment = distance(ax, ay, bx, by)
    assert point_along(ax, ay, bx, by, 0, segment) == (ax, ay)
    assert point_along(ax, ay, bx, by, segment, segment) == (bx, by)
    assert point_along(ax, ay, bx, by, segment * 2, segment) == (bx, by)


def test_within_range_matches_true_distance():
    reach = milli_tiles(1200)
    assert within_range(0, 0, reach, 0, reach)
    assert not within_range(0, 0, reach + 1, 0, reach)


# -------------------------------------------------------------------- rng


def test_same_seed_same_stream():
    a, b = Rng(12345), Rng(12345)
    assert [a.next_u32() for _ in range(50)] == [b.next_u32() for _ in range(50)]


def test_different_seeds_diverge():
    assert Rng(1).next_u32() != Rng(2).next_u32()


def test_below_is_unbiased():
    """Rejection sampling, because modulo bias would teach an agent falsehoods."""
    rng = Rng(7)
    counts = [0] * 6
    for _ in range(60_000):
        counts[rng.below(6)] += 1
    for count in counts:
        assert abs(count - 10_000) < 500, counts


def test_named_streams_are_independent_and_stable():
    """Adding a draw in one subsystem must not shift another's stream."""
    parent = Rng(99)
    deck_first = parent.stream("deck").next_u32()
    spawn_first = parent.stream("spawn").next_u32()
    assert deck_first != spawn_first
    # Re-deriving gives the same stream regardless of what else has drawn.
    parent.stream("spawn").next_u32()
    assert Rng(99).stream("deck").next_u32() == deck_first


def test_rng_state_round_trips():
    rng = Rng(5)
    [rng.next_u32() for _ in range(10)]
    saved = rng.state()
    expected = [rng.next_u32() for _ in range(5)]
    rng.restore(saved)
    assert [rng.next_u32() for _ in range(5)] == expected


# ----------------------------------------------------------------- entity


def test_deploy_window_blocks_targeting():
    """A deploying unit is on the board but cannot be hit -- this is real."""
    reset_entity_ids()
    unit = Entity(
        kind=EntityKind.TROOP, team=Team.BLUE, x=0, y=0, hitpoints=690, deploy_ticks=60
    )
    assert unit.state is EntityState.DEPLOYING
    assert not unit.is_targetable
    for _ in range(59):
        assert unit.tick_deploy() is False
    assert unit.tick_deploy() is True
    assert unit.state is EntityState.IDLE
    assert unit.is_targetable


def test_shield_absorbs_the_whole_hit_without_overflow():
    """A big hit into a small shield is wasted -- why Zap only strips Guards."""
    reset_entity_ids()
    unit = Entity(
        kind=EntityKind.TROOP, team=Team.BLUE, x=0, y=0, hitpoints=690, shield=50
    )
    dealt = unit.apply_damage(500)
    assert dealt == 50
    assert unit.shield == 0
    assert unit.hitpoints == 690, "damage must not carry through the shield"
    unit.apply_damage(500)
    assert unit.hitpoints == 190


def test_entity_ids_are_monotonic_for_stable_tiebreaks():
    reset_entity_ids()
    first = Entity(kind=EntityKind.TROOP, team=Team.BLUE, x=0, y=0, hitpoints=1)
    second = Entity(kind=EntityKind.TROOP, team=Team.RED, x=0, y=0, hitpoints=1)
    assert second.id > first.id


# ------------------------------------------------------------------ ticks


def test_tick_conversion_agrees_across_rates(registry, data, levels):
    """The same card must have the same millisecond timings at 60 and 20 TPS."""
    fast, slow = TickClock(60), TickClock(20)
    for name in ("Knight", "Pekka", "Musketeer", "Giant"):
        card = registry[name]
        a = spec_for_card(data, levels, card, clock=fast)[0]
        b = spec_for_card(data, levels, card, clock=slow)[0]
        assert fast.milliseconds(a.hit_speed_ticks) == slow.milliseconds(b.hit_speed_ticks)
        assert fast.milliseconds(a.deploy_ticks) == slow.milliseconds(b.deploy_ticks)


def test_tick_conversion_rounds_rather_than_truncates():
    """A whole tick of windup decides close interactions, so round half-up."""
    clock = TickClock(20)
    assert clock.ticks(350) == 7  # 7.0 exactly
    assert clock.ticks(370) == 7  # 7.4 -> 7
    assert clock.ticks(380) == 8  # 7.6 -> 8


def test_knight_walks_one_tile_per_second(data, levels, registry):
    """Speed 60 is 'Medium', which is exactly one tile per second."""
    spec = spec_for_card(data, levels, registry["Knight"], clock=TickClock(60))[0]
    assert spec.speed == 60
    assert spec.speed_per_tick * 60 == SUBTILES_PER_TILE


# ------------------------------------------------------------------ specs


def test_specs_carry_engine_units_not_file_units(data, levels, registry):
    spec = spec_for_card(data, levels, registry["Knight"], clock=TickClock(60))[0]
    assert spec.hitpoints == 1766
    assert spec.damage == 202
    assert spec.hit_speed_ticks == 72  # 1200ms at 60 TPS
    assert spec.attack_range == milli_tiles(1200)
    assert spec.is_melee


def test_every_troop_and_building_card_builds_a_spec(data, levels, registry):
    """A card the engine cannot turn into units is a card it cannot play."""
    failures = []
    for card in registry.standard():
        if card.kind.value == "spell":
            continue
        try:
            if not spec_for_card(data, levels, card):
                failures.append(f"{card.name}: produced no units")
        except SpecError as exc:
            failures.append(f"{card.name}: {exc}")
    assert failures == [], failures


def test_spell_carriers_are_rejected_loudly(data, levels, registry):
    """Rage and Royal Delivery deploy an invulnerable carrier, not a unit.

    These must fail as specs rather than silently become zero-hitpoint troops;
    they get their own entity type in M5. Pinned so a *third* such card shows up
    as a test failure rather than a mystery.
    """
    carriers = []
    for card in registry.standard():
        try:
            spec_for_card(data, levels, card)
        except SpecError:
            carriers.append(card.name)
    assert set(carriers) == {"Rage", "RoyalDelivery"}, carriers


def test_crown_tower_damage_reduction_has_the_right_sign(data, levels, registry):
    """``CrownTowerDamagePercent`` is a negative delta: -75 leaves 25% going through.

    Getting this sign backwards would turn every spell into a win condition, so
    it is checked both ways: a unit with no reduction must be unaffected.
    """
    import dataclasses

    knight = spec_for_card(data, levels, registry["Knight"], clock=TickClock(60))[0]
    assert knight.crown_tower_damage_percent == 0
    assert knight.damage_to(is_crown_tower=True) == knight.damage

    fireball_like = dataclasses.replace(knight, damage=688, crown_tower_damage_percent=-75)
    assert fireball_like.damage_to(is_crown_tower=False) == 688
    assert fireball_like.damage_to(is_crown_tower=True) == 172  # 25% of 688


# ---------------------------------------------------------------- elixir


@pytest.mark.parametrize("tps", [VERIFICATION_TPS, TRAINING_TPS])
def test_match_structure_matches_the_game(data, tps):
    timeline = build_timeline(data, clock=TickClock(tps))
    assert timeline.regulation_ticks == 180 * tps
    assert timeline.overtime_ticks == 120 * tps
    assert timeline.starting_elixir == 6
    assert [s.multiplier_tenths for s in timeline.segments] == [10, 20, 30]


def test_first_elixir_arrives_at_exactly_2800ms(data):
    timeline = build_timeline(data, clock=TickClock(60))
    bar = ElixirBar(timeline)
    assert bar.units == 6
    for tick in range(168):  # 2800ms at 60 TPS
        bar.regenerate(tick)
    assert bar.units == 7


def test_elixir_rate_segments_span_regulation_and_overtime(data):
    """2x starts a minute before regulation ends and runs into overtime.

    Treating "double elixir" as an overtime property would be wrong at both
    ends, so the segment boundaries are asserted explicitly.
    """
    clock = TickClock(60)
    timeline = build_timeline(data, clock=clock)
    assert timeline.segment_at(clock.seconds_to_ticks(119)).multiplier == 1
    assert timeline.segment_at(clock.seconds_to_ticks(121)).multiplier == 2
    # Still 2x after regulation ends at 180s.
    assert timeline.segment_at(clock.seconds_to_ticks(200)).multiplier == 2
    assert timeline.segment_at(clock.seconds_to_ticks(250)).multiplier == 3
    assert timeline.is_overtime(clock.seconds_to_ticks(200))


def test_elixir_caps_at_ten(data):
    timeline = build_timeline(data, clock=TickClock(60))
    bar = ElixirBar(timeline)
    for tick in range(timeline.total_ticks):
        bar.regenerate(tick)
    assert bar.units == 10


def test_spending_requires_whole_elixir(data):
    timeline = build_timeline(data, clock=TickClock(60))
    bar = ElixirBar(timeline)
    assert bar.units == 6
    assert not bar.can_afford(7)
    assert bar.spend(4)
    assert bar.units == 2
    assert not bar.spend(3)


# ---------------------------------------------------------- state hashing


def _two_identical_worlds():
    worlds = []
    for _ in range(2):
        reset_entity_ids()
        worlds.append(
            [
                Entity(kind=EntityKind.TROOP, team=Team.BLUE, x=1000, y=2000, hitpoints=690),
                Entity(kind=EntityKind.TROOP, team=Team.RED, x=5000, y=9000, hitpoints=720),
            ]
        )
    return worlds


def test_identical_runs_hash_identically():
    left, right = _two_identical_worlds()
    assert [state_hash(t, left) for t in range(5)] == [state_hash(t, right) for t in range(5)]


def test_hash_detects_a_single_subtile_of_drift():
    """The gate is only worth having if it is this sensitive."""
    left, right = _two_identical_worlds()
    right[1].x += 1
    a = [state_hash(t, left) for t in range(5)]
    b = [state_hash(t, right) for t in range(5)]
    assert compare_hashes(a, b) == 0


def test_compare_hashes_reports_the_diverging_tick():
    left, right = _two_identical_worlds()
    a = [state_hash(t, left) for t in range(5)]
    right[0].hitpoints -= 1
    b = a[:3] + [state_hash(t, right) for t in range(3, 5)]
    assert compare_hashes(a, b) == 3


def test_replay_round_trips(tmp_path):
    replay = Replay(seed=42, ticks_per_second=60, decks={"blue": ["Knight"]})
    replay.add(Command(tick=10, team=0, card="Knight", x=9000, y=5000))
    replay.hashes = [1, 2, 3]
    restored = Replay.load(replay.save(tmp_path / "r.json"))
    assert restored.seed == 42
    assert restored.commands == replay.commands
    assert restored.hashes == replay.hashes
    assert restored.by_tick()[10][0].card == "Knight"
