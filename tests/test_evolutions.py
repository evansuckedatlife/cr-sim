"""M7: evolutions.

An evolution is not a different card in the deck. It is the same slot, the same
elixir, and the same place in the cycle -- and every so often it deploys
something else. Evo Barbarians summons ``Barbarian_EV1``, which has the same
hitpoints, damage and speed as an ordinary Barbarian and gains rage after one
hit. That is the whole difference and none of it shows up in a stat comparison
of the two cards.

Two things in the data make this easy to implement wrongly.

``EvolvedSpells`` names the evolved card, but sixteen of the fifty-eight links
in this build point at ``_hero`` variants, which are a different feature. They
are told apart by the cycle count: a real evolution carries one, a hero form
does not.

The evolved unit reports the base unit's ``Name``. ``Barbarian_EV1`` calls
itself ``Barbarian``, exactly as the Graveyard's skeleton calls itself
``Skeleton``. Checking which unit was deployed by name silently passes whether
the evolution fired or not, which is how the first version of this looked
correct while doing nothing.
"""

from __future__ import annotations

import pytest

from cr_sim.data.cards import build_card_registry
from cr_sim.data.leveling import build_level_table
from cr_sim.data.source import LogicData
from cr_sim.engine.battle import Battle, BattleConfig
from cr_sim.engine.entity import EntityKind, Team
from cr_sim.engine.fixed import tiles
from cr_sim.engine.specs import build_unit_spec

from .test_data_pipeline import BUILD

DECK = ("Barbarians", "Knight", "Musketeer", "Cannon",
        "Log", "Fireball", "Goblins", "Skeletons")


@pytest.fixture(scope="module")
def world():
    data = LogicData.load(BUILD)
    return data, build_level_table(data), build_card_registry(data)


def _battle(world, deck=DECK):
    data, levels, registry = world
    battle = Battle(
        data, levels, registry,
        BattleConfig(seed=1, blue_deck=deck, red_deck=("Knight",) * 8),
    )
    battle.entities = [e for e in battle.entities if e.kind is not EntityKind.TOWER]
    battle._towers = {Team.BLUE: [], Team.RED: []}
    return battle


def _play(battle, name):
    """Force a card to the front of the cycle and play it, returning the units."""
    player = battle.players[Team.BLUE]
    player.elixir.add(10)
    player.cycle.remove(name)
    player.cycle.insert(0, name)
    before = {e.id for e in battle.entities}
    assert battle.play_card(Team.BLUE, name, tiles(9), tiles(12)), name
    for _ in range(90):
        battle.step()
    return [e for e in battle.entities if e.id not in before and e.spec is not None]


# --------------------------------------------------------------- the links


def test_evolutions_are_told_apart_from_hero_forms(world):
    """Both are named by ``EvolvedSpells``; only one has a cycle count.

    Treating the hero links as evolutions would hand half the roster a second
    card it never had.
    """
    _data, _levels, registry = world
    evolutions = [c for c in registry.standard() if c.evolution]
    assert len(evolutions) > 30, f"only {len(evolutions)} evolutions found"
    assert all(c.evolution_cycles in (1, 2) for c in evolutions)

    hero_only = [
        c for c in registry.standard()
        if (c.raw or {}).get("EvolvedSpells") and not c.evolution
    ]
    assert hero_only, "no hero-form links were filtered out at all"


def test_an_evolution_costs_the_same_as_its_base(world):
    """It is the same deck slot, not a more expensive card."""
    _data, _levels, registry = world
    for card in registry.standard():
        if not card.evolution:
            continue
        evolved = registry.get(card.evolution)
        assert evolved is not None
        assert evolved.mana_cost == card.mana_cost, card.name


def test_the_evolved_unit_reports_the_base_units_name(world):
    """Which is why nothing here identifies a deployment by name.

    ``Barbarian_EV1`` calls itself ``Barbarian``. A test that checked the name
    would pass whether the evolution fired or not.
    """
    data, levels, _registry = world
    base = build_unit_spec(data, levels, "Barbarian", level=11, rarity="Common")
    evolved = build_unit_spec(data, levels, "Barbarian_EV1", level=11, rarity="Common")
    assert evolved.name == base.name
    # The difference is behavioural, not statistical.
    assert evolved.hitpoints == base.hitpoints
    assert evolved.buff_after_hits and not base.buff_after_hits


# --------------------------------------------------------------- the cycle


def test_the_first_play_of_an_evolution_is_evolved(world):
    """A match begins with the slot charged."""
    battle = _battle(world)
    units = _play(battle, "Barbarians")
    assert units and any(u.spec.buff_after_hits for u in units)


def test_a_one_cycle_evolution_alternates(world):
    """One cycle means every second play, two means every third."""
    battle = _battle(world)
    seen = []
    for _ in range(4):
        units = _play(battle, "Barbarians")
        seen.append(any(u.spec.buff_after_hits for u in units))
    assert seen == [True, False, True, False], seen


def test_a_two_cycle_evolution_takes_longer(world):
    """Evo Archers is two cycles: evolved, plain, plain, evolved."""
    _data, _levels, registry = world
    archers = registry.get("Archer")
    if archers is None or archers.evolution_cycles != 2:
        pytest.skip("Archer is not a two-cycle evolution in this build")

    deck = ("Archer", "Knight", "Musketeer", "Cannon",
            "Log", "Fireball", "Goblins", "Skeletons")
    battle = _battle(world, deck)
    player = battle.players[Team.BLUE]
    ready = []
    for _ in range(5):
        ready.append(player.evolution_ready(archers))
        _play(battle, "Archer")
    assert ready == [True, False, False, True, False], ready


def test_spending_an_evolution_resets_its_charge(world):
    battle = _battle(world)
    player = battle.players[Team.BLUE]
    card = battle.registry.get("Barbarians")
    assert player.evolution_ready(card)
    _play(battle, "Barbarians")
    assert not player.evolution_ready(card)


def test_a_card_with_no_evolution_is_never_ready(world):
    """Most of the roster has one, so the card here is chosen deliberately.

    Bowler carries an ``EvolvedSpells`` link to a *hero* form and no cycle
    count, which is exactly the case the filter has to reject.
    """
    deck = ("Bowler", "Knight", "Musketeer", "Cannon",
            "Log", "Fireball", "Goblins", "Skeletons")
    battle = _battle(world, deck)
    player = battle.players[Team.BLUE]
    bowler = battle.registry.get("Bowler")
    assert bowler.evolution is None and bowler.evolution_cycles == 0
    assert not player.evolution_ready(bowler)
    assert _play(battle, "Bowler"), "an ordinary card stopped deploying"


def test_each_side_charges_its_own_evolutions(world):
    """The counter belongs to the player, not to the card."""
    data, levels, registry = world
    battle = Battle(
        data, levels, registry,
        BattleConfig(seed=1, blue_deck=DECK, red_deck=DECK),
    )
    card = registry.get("Barbarians")
    blue, red = battle.players[Team.BLUE], battle.players[Team.RED]
    blue.spend_evolution(card)
    assert not blue.evolution_ready(card)
    assert red.evolution_ready(card), "spending blue's evolution spent red's"
