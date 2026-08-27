"""Reach and Princess Tower support, checked against the engine itself.

The interesting assertions here are the ones that would have passed against
the broken model. Every early version of :mod:`cr_sim.data.engagement` gave a
plausible-looking answer for these pairs and a wrong one, so each test names
the specific way it used to fail.
"""

from __future__ import annotations

import pytest

from cr_sim.data.cards import build_card_registry
from cr_sim.data.engagement import (
    can_hit,
    engagement_delays,
    free_hits,
    hits_with_tower,
    opening_gap,
    resolve_duel,
    tower_matrix,
)
from cr_sim.data.interactions import build_profiles, simulate_duel
from cr_sim.data.leveling import build_level_table
from cr_sim.data.source import LogicData

from .test_data_pipeline import BUILD

TILE = 18000


@pytest.fixture(scope="module")
def world():
    data = LogicData.load(BUILD)
    levels = build_level_table(data)
    registry = build_card_registry(data)
    return data, levels, registry


@pytest.fixture(scope="module")
def profiles(world):
    data, levels, registry = world
    return build_profiles(data, levels, registry)


def fight(profiles, a, b, **kwargs):
    defenses, attacks, _ = profiles
    return resolve_duel(defenses[a], attacks[a], defenses[b], attacks[b], **kwargs)


# ------------------------------------------------------------------ reach --


def test_reach_buys_free_hits(profiles):
    """A Musketeer hits a Knight several times before he can swing back.

    The number a bare time-to-kill comparison implies is zero, because it
    starts both clocks together.
    """
    _, attacks, _ = profiles
    assert attacks["Musketeer"].attack_range > attacks["Knight"].attack_range
    assert free_hits(attacks["Musketeer"], attacks["Knight"]) >= 4
    result = fight(profiles, "Musketeer", "Knight")
    assert result.winner == "a"
    assert result.head_start >= 4
    assert result.b_hits > 0, "the Knight does arrive; this is not a clean win"


def test_head_start_belongs_to_the_winner(profiles):
    """Mini P.E.K.K.A wins with no head start of her own.

    An earlier version reported ``head_start`` off side A unconditionally, so
    every fight B won came back as zero whether or not B had shot first.
    """
    result = fight(profiles, "Musketeer", "MiniPekka")
    assert result.winner == "b"
    assert result.head_start == 0
    assert result.a_hits > 0, "the Musketeer got shots off before dying"


def test_starting_distance_changes_the_winner(profiles):
    """The same two cards, two distances, two different outcomes.

    This is the whole reason reach belongs in the model: dropped on top of a
    Musketeer the Knight wins, walked at her from range he does not.
    """
    _, attacks, _ = profiles
    far = fight(profiles, "Musketeer", "Knight")
    near = fight(profiles, "Musketeer", "Knight",
                 engage_gap=attacks["Knight"].attack_range)
    assert far.winner == "a"
    assert near.winner == "b"


def test_both_units_approach_before_either_fires(profiles):
    """Opened beyond either reach, both walk, and the closing speed is the sum.

    Modelling only the shorter-ranged unit as moving would put the first shot
    much later than it lands.
    """
    _, attacks, _ = profiles
    musketeer, knight = attacks["Musketeer"], attacks["Knight"]
    gap = musketeer.attack_range + 3 * TILE
    a_delay, b_delay = engagement_delays(musketeer, knight, gap)
    assert a_delay > 0, "the Musketeer has to walk in too"
    assert b_delay > a_delay, "the Knight still arrives later"
    combined = musketeer.speed_per_tick + knight.speed_per_tick
    assert a_delay == pytest.approx(3 * TILE / combined, rel=0.05)


def test_a_building_out_of_reach_never_fires(profiles):
    """An out-ranged building cannot walk into range, so it never answers."""
    defenses, attacks, _ = profiles
    xbow, cannon = attacks["Xbow"], attacks["Cannon"]
    assert xbow.attack_range > cannon.attack_range
    assert cannon.speed_per_tick == 0
    assert free_hits(xbow, cannon) is None


# -------------------------------------------------------------- targeting --


def test_building_only_attacker_cannot_fight_a_troop(profiles):
    """Skeletons beat a Golem because he can never select one.

    Not "does reduced damage" -- he has no way to target them at all, and a
    pure hitpoints-over-damage matrix confidently reports a fight that cannot
    happen.
    """
    defenses, attacks, _ = profiles
    assert not can_hit(defenses["Skeletons"], attacks["Golem"])
    result = fight(profiles, "Skeletons", "Golem")
    assert result.winner == "a"
    assert result.b_hits == 0
    assert result.clean


def test_ground_attacker_cannot_reach_a_flyer(profiles):
    defenses, attacks, _ = profiles
    assert not can_hit(defenses["Minions"], attacks["Knight"])
    assert can_hit(defenses["Minions"], attacks["Archer"])


# ---------------------------------------------------------------- damage --


def test_kamikaze_lands_exactly_one_hit(profiles):
    """An Ice Spirit dies delivering its hit, so it cannot grind anything down.

    Read as a repeating attacker it "wins" against a Giant by landing 37
    unanswered hits, which is how the first version of this module had it.
    """
    defenses, attacks, _ = profiles
    assert attacks["IceSpirits"].kamikaze
    result = fight(profiles, "IceSpirits", "Giant")
    assert result.a_hits == 1
    assert result.winner != "a"


def test_shield_overflow_is_wasted(profiles):
    """A big hit into a small shield does not carry through to the body.

    Modelling a shield as extra hitpoints would let one Mini P.E.K.K.A swing
    both break the shield and kill what is behind it.
    """
    defenses, attacks, _ = profiles
    guards = defenses.get("SkeletonWarriors")
    if guards is None or guards.shield_hitpoints <= 0:
        pytest.skip("no shielded defender in this build")
    big = attacks["MiniPekka"]
    assert big.damage > guards.shield_hitpoints + guards.hitpoints, (
        "this test only means something if one hit would otherwise kill outright")
    assist = hits_with_tower(guards, big, attacks["PrincessTower"])
    assert assist.alone >= 2, "the shield must cost a separate hit"


# ----------------------------------------------------------------- tower --


def test_tower_reduces_what_the_troop_must_do(profiles):
    """A Princess Tower firing alongside cuts the troop's hit count."""
    defenses, attacks, _ = profiles
    assist = hits_with_tower(defenses["Giant"], attacks["Musketeer"],
                             attacks["PrincessTower"])
    assert assist.alone > assist.with_tower > 0
    assert assist.saved == assist.alone - assist.with_tower
    assert assist.tower_alone > 0


def test_tower_can_finish_a_small_unit_alone(profiles):
    """Against Skeletons the tower needs no help at all."""
    defenses, attacks, _ = profiles
    assist = hits_with_tower(defenses["Skeletons"], attacks["Knight"],
                             attacks["PrincessTower"])
    assert assist.tower_alone == 1
    assert assist.with_tower == 0, "the tower kills it before the Knight swings"


def test_fireball_plus_tower_kills_a_musketeer(profiles):
    """The most-quoted tower interaction in the game, from the game's own tables.

    A Fireball lands a little short of a Musketeer on its own; one tower hit
    covers the difference. An earlier version of this module reported ``None``
    for it, because it counted hits off the *cast* timeline and a spell casts
    once -- which silently deleted every two-cast threshold from the matrix,
    this one included.
    """
    defenses, attacks, _ = profiles
    fireball, musketeer = attacks["Fireball"], defenses["Musketeer"]
    assert fireball.damage < musketeer.hitpoints, "short on its own, or there is no interaction"
    assist = hits_with_tower(musketeer, fireball, attacks["PrincessTower"])
    assert assist.alone == 2
    assert assist.with_tower == 1
    assert assist.tower_hits == 1


def test_tower_hits_distinguish_a_finish_from_a_grind(profiles):
    """One troop hit plus the tower is not one claim, it is two.

    Fireball needs a single tower hit to finish a Musketeer; Zap needs the
    tower to keep firing for several seconds. Both are "1" in the troop
    column, and reporting only that would flatten them together.
    """
    defenses, attacks, _ = profiles
    tower = attacks["PrincessTower"]
    quick = hits_with_tower(defenses["Musketeer"], attacks["Fireball"], tower)
    slow = hits_with_tower(defenses["Witch"], attacks["Zap"], tower)
    assert quick.with_tower == slow.with_tower == 1
    assert slow.tower_hits > quick.tower_hits * 4


def test_tower_matrix_excludes_towers_and_itself(profiles):
    defenses, attacks, _ = profiles
    results = tower_matrix(defenses, attacks)
    assert results
    assert not any(defenses[d].is_tower for d, _ in results)
    assert not any(a == "PrincessTower" for _, a in results)


# ------------------------------------------------------------ vs the engine --


@pytest.mark.parametrize("a,b", [
    ("Musketeer", "Knight"),
    ("MiniPekka", "Knight"),
    ("Archer", "Skeletons"),
    ("Giant", "Musketeer"),
])
def test_agrees_with_a_real_battle(world, profiles, a, b):
    """The arithmetic and the engine pick the same winner.

    Opened where both units can see each other. Past that the duel harness
    stops being an oracle: with the towers removed a unit with nothing in
    sight has nothing to walk toward, so it stands still and is shot down
    without moving -- Mini P.E.K.K.A sees 5.5 tiles and a Musketeer shoots
    6.0, which is enough to manufacture a disagreement out of nothing.
    """
    data, levels, registry = world
    defenses, attacks, _ = profiles
    sight = min(attacks[a].sight_range, attacks[b].sight_range)
    engage = min(sight, opening_gap(attacks[a], attacks[b]))
    mine = resolve_duel(defenses[a], attacks[a], defenses[b], attacks[b],
                        engage_gap=engage)
    centre = (engage + defenses[a].collision_radius
              + defenses[b].collision_radius) / TILE
    sim = simulate_duel(data, levels, registry, b, a, gap=centre)
    assert sim is not None
    expected = "attacker" if mine.winner == "a" else "defender"
    assert sim.winner == expected
