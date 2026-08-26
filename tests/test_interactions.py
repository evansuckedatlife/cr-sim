"""Interaction breakpoints — the seed of verification gate #2.

Individual stat values can each look plausible while the pipeline is still
wrong; what pins them down is that *relationships between* cards come out
right.  "P.E.K.K.A one-shots a Musketeer" is a fact about Clash Royale that
every player knows, and it is only true for one of the two candidate damage
numbers -- so it settles the question that no stat website could.

These are deliberately stat-level: no engine required, so they run from M0
onward.  The full matrix (who beats whom in a real fight, tower hit counts,
travel timings) needs the simulator and arrives with M2/M3.

Everything here is at displayed level 11 -- tournament standard, equal level
for both sides, which is how these interactions are always quoted.
"""

from __future__ import annotations

import pytest

from cr_sim.data.cards import build_card_registry, card_stat_summary
from cr_sim.data.leveling import build_level_table
from cr_sim.data.source import LogicData

from .test_data_pipeline import BUILD


@pytest.fixture(scope="module")
def stats():
    data = LogicData.load(BUILD)
    levels = build_level_table(data)
    registry = build_card_registry(data)

    def get(card_name: str) -> dict:
        return card_stat_summary(data, levels, registry[card_name])

    return get


def hp(stats, name: str) -> int:
    value = stats(name)["hitpoints"]
    assert isinstance(value, int)
    return value


def dmg(stats, name: str) -> int:
    value = stats(name)["damage"]
    assert isinstance(value, int)
    return value


# --------------------------------------------------------- one-shot breakpoints


@pytest.mark.parametrize("attacker,victim", [("Pekka", "Musketeer"), ("Pekka", "Wizard")])
def test_pekka_one_shots_squishy_ranged_units(stats, attacker, victim):
    """This is what settles the P.E.K.K.A damage question.

    The extracted build gives 842 damage; some public stat sites say 510.
    A Musketeer has 721 hitpoints at the same level, and P.E.K.K.A one-shotting
    a Musketeer is one of the most familiar interactions in the game -- which
    510 cannot produce and 842 can.
    """
    assert dmg(stats, attacker) >= hp(stats, victim)


def test_mini_pekka_one_shots_a_musketeer(stats):
    assert dmg(stats, "MiniPekka") >= hp(stats, "Musketeer")


def test_pekka_does_not_one_shot_a_knight(stats):
    """The breakpoint has to cut both ways or it proves nothing."""
    assert dmg(stats, "Pekka") < hp(stats, "Knight")


# --------------------------------------------------------------- spell breakpoints


def test_fireball_alone_does_not_kill_a_musketeer(stats):
    """The classic near-miss: she survives a same-level Fireball."""
    assert dmg(stats, "Fireball") < hp(stats, "Musketeer")


def test_fireball_plus_zap_kills_a_musketeer(stats):
    """...and the classic combo that finishes her."""
    assert dmg(stats, "Fireball") + dmg(stats, "Zap") >= hp(stats, "Musketeer")


@pytest.mark.parametrize("victim", ["Skeletons", "Bats"])
def test_zap_clears_one_hitpoint_swarms(stats, victim):
    assert dmg(stats, "Zap") >= hp(stats, victim)


def test_zap_does_not_kill_goblins(stats):
    """Zap leaves Goblins alive; this is why the Log exists."""
    assert dmg(stats, "Zap") < hp(stats, "Goblins")


def test_log_kills_goblins(stats):
    assert dmg(stats, "Log") >= hp(stats, "Goblins")


def test_rocket_kills_a_wizard(stats):
    assert dmg(stats, "Rocket") >= hp(stats, "Wizard")


# ------------------------------------------------------- multi-projectile care


@pytest.mark.parametrize("card_name", ["Arrows", "Hunter", "Princess"])
def test_multi_projectile_damage_is_flagged(stats, card_name):
    """Damage on these is per projectile, and must not be read as a single hit.

    Hunter fires ten pellets from one shot and Arrows lands three separate
    waves, so comparing their bare ``damage`` against a hitpoint total would be
    wrong in opposite directions.  The flag keeps that mistake visible.
    """
    summary = stats(card_name)
    assert summary.get("damage_is_per_projectile") is True
    assert summary.get("multiple_projectiles") or summary.get("projectile_waves")
