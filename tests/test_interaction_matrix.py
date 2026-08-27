"""Verification gate #2: the interaction matrix.

The stat gate (``test_data_pipeline.py``) and the breakpoint tests
(``test_interactions.py``) pin individual numbers. This file is about whether
those numbers *combine* right -- whether a shield really costs the attacker an
extra hit, whether a spell's reduced tower damage has the right sign, whether
arithmetic and an actual simulated duel agree when nothing should make them
disagree.

Two things are deliberately absent:

* **No agreement floor against ``reference/hits_to_kill.csv``.** That sheet is
  roughly a year old (see ``reference/hits_to_kill.md``), Clash Royale
  rebalances constantly, and a floor here would be a test of how much the
  game has changed, not of whether the engine is right -- something you would
  have to keep loosening for reasons that have nothing to do with a bug. The
  CLI's ``interactions`` command reports that comparison; nothing here gates
  on it.
* **No re-litigating of already-anchored numbers.** Musketeer's hitpoints and
  Knight's damage are pinned in ``test_data_pipeline.py``; this file is about
  the arithmetic built on top of them.

What *is* asserted here is deliberately narrow: specific, well-understood
interactions confirmed against the *current* build (not the stale sheet), and
the one invariant that should never break regardless of what balance patch
lands -- that for a plain melee duel with no ranged/splash/charge advantage,
naive arithmetic and an actual simulated fight agree on the winner.
"""

from __future__ import annotations

import pytest

from cr_sim.data.cards import build_card_registry
from cr_sim.data.interactions import (
    NAME_MAP,
    _attack_key,
    _defense_key,
    build_profiles,
    compute_hits,
    compute_matrix,
    predicted_winner,
    simulate_duel,
)
from cr_sim.data.leveling import build_level_table
from cr_sim.data.source import LogicData

from .test_data_pipeline import BUILD


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


@pytest.fixture(scope="module")
def matrix(profiles):
    defenses, attacks, _labels = profiles
    return compute_matrix(defenses, attacks)


def hits(matrix, defender: str, attacker: str) -> int | None:
    result = matrix.get((defender, attacker))
    return result.hits if result else None


# ------------------------------------------------------- name-map integrity


def test_every_mapped_name_resolves_to_a_usable_profile(profiles):
    """A typo in ``NAME_MAP`` fails silently otherwise.

    The mapping is explicit by design (no fuzzy matching -- see the module
    docstring in ``cr_sim/data/interactions.py``), which means nothing catches
    a stale or misspelled character/card name except actually resolving every
    entry. A ``Ref`` that resolves to neither a defense nor an attack profile
    is dead weight at best and a silently-wrong pairing at worst.
    """
    defenses, attacks, _labels = profiles
    dead = [
        name
        for name, ref in NAME_MAP.items()
        if _defense_key(ref) not in defenses and _attack_key(ref) not in attacks
    ]
    assert dead == []


def test_the_standard_pool_mostly_resolves(profiles):
    """A regression that silently shrinks the resolvable pool should be loud.

    Not a claim about any specific count -- just that the pipeline still
    turns the large majority of playable cards into usable combat profiles,
    the way it does today (118 defenders / 113 attackers out of 122 standard
    cards + a handful of towers/sub-characters/evolutions).
    """
    defenses, attacks, _labels = profiles
    assert len(defenses) > 100
    assert len(attacks) > 100
    assert "Knight" in defenses and "Knight" in attacks


# --------------------------------------------------- well-understood, current


def test_fireball_alone_does_not_kill_a_musketeer(matrix):
    """The classic near-miss, re-derived through the matrix rather than raw stats."""
    assert hits(matrix, "Musketeer", "Fireball") == 2


def test_knight_one_shots_a_skeleton(matrix):
    assert hits(matrix, "Skeletons", "Knight") == 1


def test_arrows_clears_bats_and_minions(matrix):
    """Arrows' listed damage is one of three waves; the matrix must use the
    three-wave total (as ``test_interactions.py`` already established) or
    this would read as needing more casts than it actually does.
    """
    assert hits(matrix, "Bats", "Arrows") == 1
    assert hits(matrix, "Minions", "Arrows") == 1


def test_pekka_one_shots_a_musketeer_but_not_a_knight(matrix):
    """The breakpoint that originally settled the P.E.K.K.A damage question
    (see ``test_interactions.py``), re-checked here because it is exactly the
    kind of fact a shield/tower-percent regression could quietly break.
    """
    assert hits(matrix, "Musketeer", "Pekka") == 1
    assert hits(matrix, "Knight", "Pekka") is not None and hits(matrix, "Knight", "Pekka") > 1


# ---------------------------------------------------------------- shields


def test_a_shield_costs_a_whole_extra_hit_not_partial_damage(profiles):
    """Direct check of ``entity.py``'s documented shield rule: a shield
    absorbs a full hit and discards the overflow, it does not let a big hit
    punch through into the body. Guards' shield (256 at tournament standard)
    against Knight's 202 damage must cost a full second hit, not be treated
    as "202 of the shield's 256 gone, 46 carries over".
    """
    defenses, attacks, _labels = profiles
    shield = defenses["SkeletonWarriors#shield"]
    knight = attacks["Knight"]
    result = compute_hits(shield, knight)
    assert result is not None
    assert result.hits == 2  # ceil(256/202), not 1


# ------------------------------------------------------------ crown towers


def test_crown_tower_damage_percent_only_applies_to_towers(profiles):
    """Fireball's reduced tower damage must not leak onto an ordinary target,
    and must apply once it is a tower on the receiving end.
    """
    defenses, attacks, _labels = profiles
    fireball = attacks["Fireball"]
    musketeer_defense = defenses["Musketeer"]
    tower_defense = defenses["PrincessTower"]

    against_musketeer = compute_hits(musketeer_defense, fireball)
    against_tower = compute_hits(tower_defense, fireball)
    assert against_musketeer is not None and against_tower is not None

    # Full damage against a troop, a quarter of it against a tower (-75%).
    full_dmg = fireball.damage
    reduced_dmg = full_dmg * (100 + fireball.crown_tower_damage_percent) // 100
    assert reduced_dmg < full_dmg
    assert against_tower.hits == -(-tower_defense.hitpoints // reduced_dmg)


# --------------------------------------------------------------------- ramps


def test_inferno_tower_ramp_needs_far_fewer_hits_than_its_first_stage_implies():
    """Inferno Tower's damage is 17 for ~2s, then 62, then 331 (see
    ``specs.py``'s docstring). A naive ``ceil(hp / 17)`` against a tanky
    target would wildly overstate how long it survives; the ramp must be
    honoured hit-by-hit against elapsed time, not averaged or ignored.
    """
    from cr_sim.data.interactions import AttackProfile, DefenseProfile

    # A stand-in "tank": 5000 hp, big enough that the ramp matters.
    tank = DefenseProfile(key="tank", hitpoints=5000, shield_hitpoints=0, flying=False)
    inferno = AttackProfile(
        key="inferno",
        damage=17,
        crown_tower_damage_percent=0,
        attacks_ground=True,
        attacks_air=True,
        hit_speed_ticks=24,  # 400ms at 60 ticks/s
        load_time_ticks=72,
        variable_damage=(17, 62, 331),
        variable_damage_ticks=(120, 120),  # 2000ms each at 60 ticks/s
    )
    naive_hits = -(-5000 // 17)  # what ignoring the ramp would say: 295 hits
    result = compute_hits(tank, inferno)
    assert result is not None
    assert result.hits < naive_hits
    # by hand: 5 hits at 17 (85), then 5 hits at 62 (395, running total 480),
    # then the rest at 331 -- (5000-480) / 331 = 13.66 -> 14 more hits, 24 total
    assert result.hits == 24


# --------------------------------------------------- computed vs simulated


#: Plain melee, similar range, no charge/splash/multi-target advantage on
#: either side -- pairs where arithmetic's "whoever's time-to-kill is
#: shorter" guess has nothing to be wrong about. If this ever disagrees with
#: an actual duel, something in the basic attack cycle broke, not a modelling
#: nuance arithmetic was never meant to capture.
_SIMPLE_MELEE_PAIRS = [
    ("Knight", "MiniPekka"),
    ("Barbarians", "Valkyrie"),
    ("Skeletons", "Knight"),
]


@pytest.mark.parametrize("defender,attacker", _SIMPLE_MELEE_PAIRS)
def test_computed_and_simulated_agree_on_simple_melee_duels(world, matrix, defender, attacker):
    """The one invariant a stat gate and a simulator must never disagree on.

    Deploy time, closing distance and retargeting are real and are exactly
    what the simulated pass exists to catch -- but for two plain melee units
    placed close together with no range, splash or charge mismatch, none of
    that should be enough to flip who wins. If it does, the attack cycle
    itself (not a mechanic) is the suspect.
    """
    data, levels, registry = world
    sim = simulate_duel(data, levels, registry, defender, attacker, gap=1.0)
    assert sim is not None
    assert sim.winner in ("attacker", "defender")

    guess = predicted_winner(matrix, defender, attacker)
    assert guess == sim.winner, (
        f"arithmetic predicted {guess}, the duel gave {sim.winner} "
        f"({attacker} vs {defender}, {sim.ticks} ticks, {sim.hits_landed} hits landed)"
    )
