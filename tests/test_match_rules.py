"""M4 gate: how a match still level at the end of regulation gets resolved.

Elixir, deploy legality, ordinary crown scoring and the King's instant win are
already covered elsewhere (test_combat.py, test_arena_and_battle.py). What
was missing until this gate is the resolution ladder for a match that reaches
the end of regulation without a winner:

    regulation ends level -> overtime (sudden death) -> still level ->
    tiebreaker on tower damage -> still exactly level -> draw

Reaching overtime for real takes three minutes of simulated ticks -- far too
slow for a test suite that must stay fast. Every test here instead drives
``battle.tick`` and tower/crown state directly to the moment that matters and
steps once, the same way test_combat.py kills towers directly with
``.kill()`` rather than fighting a match down to them.
"""

from __future__ import annotations

import pytest

from cr_sim.data.cards import build_card_registry
from cr_sim.data.leveling import build_level_table
from cr_sim.data.source import LogicData
from cr_sim.engine.battle import Battle, BattleConfig
from cr_sim.engine.entity import EntityState, Team

from .test_data_pipeline import BUILD

DECK = ("Knight",) * 8


@pytest.fixture(scope="module")
def world():
    data = LogicData.load(BUILD)
    return data, build_level_table(data), build_card_registry(data)


def _battle(world, *, seed=1):
    """A full battle -- real towers on both sides, nothing else on the board."""
    data, levels, registry = world
    return Battle(
        data, levels, registry,
        BattleConfig(seed=seed, blue_deck=DECK, red_deck=DECK),
    )


def _princesses(battle, team):
    return [t for t in battle._towers[team] if "King" not in t.spec.name]


# ------------------------------------------------------------ regulation end


def test_regulation_ends_immediately_when_crowns_differ(world):
    """A crown lead at the final tick of regulation wins outright -- no overtime."""
    battle = _battle(world)
    battle.tick = battle.timeline.regulation_ticks - 1
    battle.players[Team.BLUE].crowns = 1
    battle.step()
    assert battle.finished
    assert battle.result.reason == "regulation"
    assert battle.result.winner is Team.BLUE
    assert battle.result.blue_crowns == 1 and battle.result.red_crowns == 0


def test_level_match_plays_on_into_overtime(world):
    """Level on crowns at the buzzer means the match keeps going, not ending.

    This is the case the whole ladder exists for: regulation alone cannot
    settle it.
    """
    battle = _battle(world)
    battle.tick = battle.timeline.regulation_ticks - 1
    assert not battle.in_overtime
    battle.step()
    assert not battle.finished, "a level match ended instead of going to overtime"
    assert battle.in_overtime


def test_in_overtime_reflects_the_clock_not_a_flag(world):
    """``in_overtime`` is a live read of the tick, so it works under direct control too."""
    battle = _battle(world)
    assert not battle.in_overtime
    battle.tick = battle.timeline.regulation_ticks
    assert battle.in_overtime
    battle.tick = battle.timeline.regulation_ticks - 1
    assert not battle.in_overtime


# --------------------------------------------------------------- sudden death


def test_first_tower_in_overtime_wins_instantly(world):
    """Sudden death: the very first tower destroyed ends it, Princess or not.

    Two crowns' worth of a lead does not matter here -- only being first.
    """
    battle = _battle(world)
    battle.tick = battle.timeline.regulation_ticks
    assert battle.in_overtime
    victim = _princesses(battle, Team.RED)[0]
    victim.kill()
    battle.step()
    assert battle.finished
    assert battle.result.reason == "sudden death"
    assert battle.result.winner is Team.BLUE


def test_the_same_kill_one_tick_earlier_is_only_a_crown(world):
    """The identical event, one tick before the buzzer, does not end the match.

    This is what makes sudden death a genuinely different rule from ordinary
    crown-scoring, rather than a relabeling of it: the period it happens in
    is what matters, not the event itself.
    """
    battle = _battle(world)
    battle.tick = battle.timeline.regulation_ticks - 5
    victim = _princesses(battle, Team.RED)[0]
    victim.kill()
    battle.step()
    assert not battle.finished
    assert battle.players[Team.BLUE].crowns == 1


def test_king_destroyed_in_overtime_keeps_its_own_reason(world):
    """Destroying the King is always an instant win; overtime doesn't relabel it as sudden death."""
    battle = _battle(world)
    battle.tick = battle.timeline.regulation_ticks
    king = battle._king(Team.RED)
    king.kill()
    battle.step()
    assert battle.finished
    assert battle.result.reason == "king tower destroyed"
    assert battle.result.winner is Team.BLUE


def test_third_crown_in_overtime_is_still_three_crowns(world):
    """Reaching three crowns keeps its own reason too, even during overtime."""
    battle = _battle(world)
    battle.tick = battle.timeline.regulation_ticks
    battle.players[Team.BLUE].crowns = 2
    battle.players[Team.RED].crowns = 2
    victim = _princesses(battle, Team.RED)[0]
    victim.kill()
    battle.step()
    assert battle.finished
    assert battle.result.reason == "three crowns"
    assert battle.result.winner is Team.BLUE


# ------------------------------------------------------------ the tiebreaker


def test_overtime_expiring_level_falls_to_the_tower_damage_tiebreaker(world):
    """No tower falls in the whole of overtime: whoever chipped harder wins."""
    battle = _battle(world)
    battle.tick = battle.timeline.total_ticks - 1
    blue_tower = _princesses(battle, Team.BLUE)[0]
    red_tower = _princesses(battle, Team.RED)[0]
    blue_tower.hitpoints = blue_tower.max_hitpoints - 100  # lightly chipped
    red_tower.hitpoints = red_tower.max_hitpoints - 500    # chipped harder
    battle.step()
    assert battle.finished
    assert battle.result.reason == "tiebreaker"
    # Red took more damage -> Blue is the side that did more damage -> Blue wins.
    assert battle.result.winner is Team.BLUE


def test_tiebreaker_compares_percentage_not_raw_hitpoints(world):
    """Comparing raw hitpoints instead of a share of max would get this backwards.

    Blue's tower has more hitpoints left in absolute terms but has lost half
    its (artificially large) pool; Red's has fewer hitpoints left but has
    barely been touched. Percentage-wise Red is the less-damaged side, so
    Red -- not Blue -- must win.
    """
    battle = _battle(world)
    battle.tick = battle.timeline.total_ticks - 1
    blue_tower = _princesses(battle, Team.BLUE)[0]
    red_tower = _princesses(battle, Team.RED)[0]
    blue_tower.max_hitpoints = 4000
    blue_tower.hitpoints = 2000  # 50% remaining, 2000 raw hp
    red_tower.max_hitpoints = 1000
    red_tower.hitpoints = 900  # 90% remaining, only 900 raw hp
    assert red_tower.hitpoints < blue_tower.hitpoints, "test setup should favour blue on raw hp"
    battle.step()
    assert battle.result.reason == "tiebreaker"
    assert battle.result.winner is Team.RED


def test_tiebreaker_uses_each_sides_worst_tower(world):
    """A side has two Princess Towers; only the more-damaged one should count."""
    battle = _battle(world)
    battle.tick = battle.timeline.total_ticks - 1
    blue_towers = _princesses(battle, Team.BLUE)
    red_towers = _princesses(battle, Team.RED)
    assert len(blue_towers) == 2 and len(red_towers) == 2

    blue_towers[0].hitpoints = blue_towers[0].max_hitpoints  # untouched
    blue_towers[1].hitpoints = 1  # nearly destroyed
    for tower in red_towers:
        tower.hitpoints = tower.max_hitpoints - 10  # both only lightly chipped

    battle.step()
    assert battle.result.reason == "tiebreaker"
    assert battle.result.winner is Team.RED, "blue's worst tower should decide it, not its best"


def test_exact_tie_on_the_tiebreaker_is_a_genuine_draw(world):
    """Equal proportional damage on both sides is a draw, not a coin flip."""
    battle = _battle(world)
    battle.tick = battle.timeline.total_ticks - 1
    blue_tower = _princesses(battle, Team.BLUE)[0]
    red_tower = _princesses(battle, Team.RED)[0]
    blue_tower.hitpoints = blue_tower.max_hitpoints // 2
    red_tower.hitpoints = red_tower.max_hitpoints // 2
    battle.step()
    assert battle.finished
    assert battle.result.reason == "draw"
    assert battle.result.winner is None
    assert battle.result.blue_crowns == battle.result.red_crowns


def test_a_princess_tower_already_lost_earlier_still_counts_at_the_tiebreaker(world):
    """1-1 on crowns from regulation is still level; the surviving towers decide it.

    Each side can enter overtime already down one Princess Tower (destroyed
    during regulation) and still be level on crowns. The destroyed tower is
    unambiguously the worst tower on its side -- zero is the lowest
    percentage there is -- so it must be the one compared, not the survivor.
    """
    battle = _battle(world)
    battle.tick = battle.timeline.total_ticks - 1
    battle.players[Team.BLUE].crowns = 1
    battle.players[Team.RED].crowns = 1

    blue_towers = _princesses(battle, Team.BLUE)
    # Already dead *before* this tick starts -- not killed by it. Using
    # .kill() here would let resolve_deaths score it THIS tick, which is
    # exactly the sudden-death case the other tests cover, not this one.
    blue_towers[0].hitpoints = 0
    blue_towers[0].dead = True
    blue_towers[0].state = EntityState.DEAD
    blue_towers[1].hitpoints = blue_towers[1].max_hitpoints  # the survivor, untouched

    red_towers = _princesses(battle, Team.RED)
    for tower in red_towers:
        tower.hitpoints = tower.max_hitpoints - 5  # both alive, barely chipped

    battle.step()
    assert battle.finished
    assert battle.result.reason == "tiebreaker"
    assert battle.result.winner is Team.RED, "blue's destroyed tower should still be its worst"


# ------------------------------------------------------------------ via run()


def test_run_surfaces_the_regulation_result_not_a_time_fallback(world):
    """run() must report the real reason, not fall back to the generic 'time' result."""
    battle = _battle(world)
    battle.tick = battle.timeline.regulation_ticks - 1
    battle.players[Team.RED].crowns = 1
    result = battle.run()
    assert result is battle.result
    assert result.reason == "regulation"
    assert result.winner is Team.RED


def test_run_reaches_the_tiebreaker_when_overtime_runs_all_the_way_out(world):
    """run()'s own stop condition must still land on the tiebreaker, not stop short."""
    battle = _battle(world)
    battle.tick = battle.timeline.total_ticks - 1
    blue_tower = _princesses(battle, Team.BLUE)[0]
    red_tower = _princesses(battle, Team.RED)[0]
    blue_tower.hitpoints = blue_tower.max_hitpoints
    red_tower.hitpoints = red_tower.max_hitpoints - 200
    result = battle.run()
    assert result.reason == "tiebreaker"
    assert result.winner is Team.BLUE
    # The result is decided while processing overtime's last tick, before the
    # tick counter's end-of-step increment -- the same convention every other
    # phase-decided reason uses (three crowns, king destroyed, ...).
    assert result.ticks == battle.timeline.total_ticks - 1
    assert battle.tick == battle.timeline.total_ticks, "run() stopped short of the clock"
