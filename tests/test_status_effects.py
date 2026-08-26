"""M5: the buffs that do something other than damage.

Damage-over-time was the easy half of the buff system -- a number on a timer.
This is the other half, where a buff changes what a unit *is* rather than how
much health it has: dragged by a Tornado, healed by a Heal Spirit, unable to be
targeted at all, or stunned into missing its next swing.

Each of these is the defining property of the card it belongs to. A Tornado
that does not pull is a 33-damage spell nobody would play, and a Royal Ghost
that can be targeted is a worse Mini P.E.K.K.A.
"""

from __future__ import annotations

import pytest

from cr_sim.data.cards import build_card_registry
from cr_sim.data.leveling import build_level_table
from cr_sim.data.source import LogicData
from cr_sim.engine.battle import Battle, BattleConfig
from cr_sim.engine.buffs import apply_delta, as_delta
from cr_sim.engine.entity import Entity, EntityKind, Team
from cr_sim.engine.fixed import tiles, to_tiles
from cr_sim.engine.specs import build_unit_spec

from .test_data_pipeline import BUILD


@pytest.fixture(scope="module")
def world():
    data = LogicData.load(BUILD)
    return data, build_level_table(data), build_card_registry(data)


def _battle(world, card, *, red=("Knight",) * 8):
    """A bare arena: no towers, so nothing interferes with what is measured."""
    data, levels, registry = world
    battle = Battle(
        data, levels, registry,
        BattleConfig(seed=1, blue_deck=(card,) * 8, red_deck=red),
    )
    battle.entities = [e for e in battle.entities if e.kind is not EntityKind.TOWER]
    battle._towers = {Team.BLUE: [], Team.RED: []}
    battle.players[Team.BLUE].elixir.add(10)
    return battle


def _spawn(battle, world, unit, team, x, y, *, rarity="Common", hitpoints=None):
    data, levels, _ = world
    spec = build_unit_spec(data, levels, unit, level=11, rarity=rarity, clock=battle.clock)
    entity = Entity(
        kind=spec.kind, team=team, x=tiles(x), y=tiles(y),
        hitpoints=hitpoints or spec.hitpoints, spec=spec,
        collision_radius=spec.collision_radius, mass=spec.mass, flying=spec.flying,
    )
    entity.max_hitpoints = entity.hitpoints
    entity.deploy_ticks_left = 0
    battle._register(entity)
    return entity


# ------------------------------------------------------------------ tornado


def test_tornado_drags_a_unit_to_its_centre(world):
    """The card's entire purpose. 360 tiles/minute, per reference/anchors.json."""
    battle = _battle(world, "Tornado")
    victim = _spawn(battle, world, "Knight", Team.RED, 9, 12, hitpoints=99_999)
    battle.play_card(Team.BLUE, "Tornado", tiles(9), tiles(16))  # four tiles away
    for _ in range(80):
        battle.step()
    assert to_tiles(victim.y) == pytest.approx(16.0, abs=0.05), "not pulled to the eye"


def test_tornado_pulls_a_unit_that_is_standing_still_to_fight(world):
    """A pull is a force, not a decision -- so it has to move a unit mid-attack.

    This is what the card is played for: dragging a committed push off its
    target and into the King Tower. A unit locked in combat does not move under
    its own power at all, so folding the pull into the movement phase would
    skip precisely the case that matters.
    """
    battle = _battle(world, "Tornado")
    attacker = _spawn(battle, world, "Knight", Team.RED, 9, 12, hitpoints=99_999)
    _spawn(battle, world, "Knight", Team.BLUE, 9, 12.6, hitpoints=99_999)
    for _ in range(60):  # let them lock together and start swinging
        battle.step()
    held = to_tiles(attacker.y)
    battle.play_card(Team.BLUE, "Tornado", tiles(9), tiles(16))
    for _ in range(60):
        battle.step()
    assert to_tiles(attacker.y) > held + 1.0, "a fighting unit was not pulled"


def test_tornado_does_not_drag_buildings(world):
    """Nothing in the game moves a Cannon."""
    battle = _battle(world, "Tornado", red=("Cannon",) * 8)
    battle.players[Team.RED].elixir.add(10)
    assert battle.play_card(Team.RED, "Cannon", tiles(9), tiles(20))
    cannon = next(e for e in battle.entities if e.kind is EntityKind.BUILDING)
    where = (cannon.x, cannon.y)
    battle.play_card(Team.BLUE, "Tornado", tiles(9), tiles(16))
    for _ in range(80):
        battle.step()
    assert (cannon.x, cannon.y) == where


# --------------------------------------------------------------------- heal


def test_heal_spirit_restores_hitpoints_in_four_applications(world):
    """Healing is damage-over-time with the sign flipped, on the same clock.

    HealSpiritBuff is a 250ms HitFrequency over a 1000ms BuffTime, so four
    top-ups rather than one lump -- the same structure as Poison's eight ticks.
    """
    battle = _battle(world, "Heal")
    ally = _spawn(battle, world, "Knight", Team.BLUE, 9, 12.5)
    ally.max_hitpoints = 40_000
    ally.hitpoints = 20_000
    _spawn(battle, world, "Knight", Team.RED, 9, 14)  # something for it to fly at

    assert battle.play_card(Team.BLUE, "Heal", tiles(9), tiles(12.5))
    previous, gains = ally.hitpoints, []
    for _ in range(400):
        battle.step()
        if ally.hitpoints > previous:
            gains.append(ally.hitpoints - previous)
        previous = ally.hitpoints

    assert len(gains) == 4, f"{len(gains)} heal applications"
    assert len(set(gains)) == 1, f"uneven applications: {gains}"


def test_healing_cannot_take_a_unit_past_full(world):
    battle = _battle(world, "Heal")
    ally = _spawn(battle, world, "Knight", Team.BLUE, 9, 12.5)
    ally.hitpoints = ally.max_hitpoints - 1
    _spawn(battle, world, "Knight", Team.RED, 9, 14)
    battle.play_card(Team.BLUE, "Heal", tiles(9), tiles(12.5))
    for _ in range(400):
        battle.step()
    assert ally.hitpoints <= ally.max_hitpoints


# ------------------------------------------------------------- invisibility


def test_royal_ghost_becomes_untargetable_when_it_stops_attacking(world):
    """BuffWhenNotAttackingTime is 2000ms -- 120 ticks."""
    battle = _battle(world, "Knight")
    ghost = _spawn(battle, world, "Ghost", Team.RED, 9, 12, rarity="Legendary")
    for _ in range(60):
        battle.step()
    assert ghost.is_acquirable, "went invisible early"
    for _ in range(70):
        battle.step()
    assert not ghost.is_acquirable, "never went invisible"


def test_an_invisible_unit_is_still_burned_by_a_spell(world):
    """Invisibility hides you from *targeting*, not from an area of effect.

    The distinction is the reason ``is_acquirable`` is separate from
    ``is_targetable``: collapsing them would make Royal Ghost immune to Poison,
    turning a repositioning tool into blanket invulnerability.
    """
    battle = _battle(world, "Fireball")
    ghost = _spawn(battle, world, "Ghost", Team.RED, 9, 12, rarity="Legendary")
    for _ in range(130):
        battle.step()
    assert not ghost.is_acquirable, "test needs it invisible"

    before = ghost.hitpoints
    battle.play_card(Team.BLUE, "Fireball", ghost.x, ghost.y)
    for _ in range(200):
        battle.step()
    assert ghost.hitpoints < before, "a spell failed to hit an invisible unit"


def test_nothing_targets_an_invisible_unit(world):
    """The buff is applied directly rather than waited for.

    A Royal Ghost left to its own devices closes and attacks, which makes it
    visible again -- so letting the card drive this would test the approach
    rather than the targeting rule. Applying ``Invisibility`` to a stationary
    dummy isolates the one thing being asserted.
    """
    battle = _battle(world, "Knight")
    dummy = _spawn(battle, world, "Knight", Team.RED, 9, 14, hitpoints=99_999)
    hunter = _spawn(battle, world, "Musketeer", Team.BLUE, 9, 12, hitpoints=99_999)
    spec = battle._buff_spec("Invisibility", "Common", 11)
    assert spec is not None and spec.invisible
    battle._apply_buff(dummy, spec, 600, source=dummy.id)

    for _ in range(300):
        battle.step()
        assert hunter.target_id != dummy.id, "acquired an invisible target"
    assert dummy.hitpoints == dummy.max_hitpoints, "an invisible unit was shot"


# -------------------------------------------------------------------- stuns


def test_electro_wizard_stuns_what_he_hits(world):
    """His hits carry ZapFreeze for 500ms -- a -100 speed multiplier, i.e. a stun.

    This is the mechanic behind his signature use: the stun interrupts an
    Inferno Tower's damage ramp.
    """
    battle = _battle(world, "Knight")
    wizard = _spawn(battle, world, "ElectroWizard", Team.BLUE, 9, 12, rarity="Legendary")
    victim = _spawn(battle, world, "Knight", Team.RED, 9, 14, hitpoints=99_999)
    assert wizard.spec.buff_on_damage == "ZapFreeze"

    stunned = 0
    for _ in range(400):
        battle.step()
        if victim.buffs is not None and victim.buffs.is_frozen():
            stunned += 1
    assert stunned > 0, "his hits never stunned"
    assert "ZapFreeze" in (victim.buffs.active_names() if victim.buffs else ())


# ------------------------------------------------------- the delta convention


def test_a_neutral_hundred_does_not_cancel_a_freeze():
    """IgnoreBarrel is exactly 100 -- a 1.0x marker, not a doubling.

    Summed raw against Freeze's -100 the two cancelled and the unit kept
    walking, so a targeting-immunity flag cured Freeze. As deltas they are 0
    and -100, and it stays stopped.
    """
    assert as_delta(100) == 0
    assert as_delta(-100) == -100
    assert as_delta(130) == 30
    assert apply_delta(1000, as_delta(100) + as_delta(-100)) == 0


def test_two_stacked_boosts_do_not_cancel_each_other():
    """+30% twice is 1.6x, not 1.0x.

    Routing summed deltas through the raw-value reader would see 100 and call
    it neutral, silently deleting both buffs.
    """
    assert apply_delta(1000, as_delta(130) + as_delta(130)) == 1600
