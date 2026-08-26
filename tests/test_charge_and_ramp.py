"""M6: the two mechanics where a swing is not worth its listed damage.

A charge and a damage ramp are opposite answers to the same question -- how a
unit earns more than its ``Damage`` column. The Prince earns it by covering
ground; the Inferno earns it by holding a target. Both are why the cards exist,
and both are invisible in a stat table.

They also share a failure mode: each is undone by a stun, and that is what a
500ms Electro Wizard zap is actually for.
"""

from __future__ import annotations

import pytest

from cr_sim.data.cards import build_card_registry
from cr_sim.data.leveling import build_level_table
from cr_sim.data.source import LogicData
from cr_sim.engine.battle import Battle, BattleConfig
from cr_sim.engine.combat import AttackState, ramp_damage
from cr_sim.engine.entity import Entity, EntityKind, Team
from cr_sim.engine.fixed import tiles, to_tiles
from cr_sim.engine.specs import build_unit_spec

from .test_data_pipeline import BUILD


@pytest.fixture(scope="module")
def world():
    data = LogicData.load(BUILD)
    return data, build_level_table(data), build_card_registry(data)


def _battle(world, card, *, towers=False):
    data, levels, registry = world
    battle = Battle(
        data, levels, registry,
        BattleConfig(seed=1, blue_deck=(card,) * 8, red_deck=("Knight",) * 8),
    )
    if not towers:
        battle.entities = [e for e in battle.entities if e.kind is not EntityKind.TOWER]
        battle._towers = {Team.BLUE: [], Team.RED: []}
    battle.players[Team.BLUE].elixir.add(10)
    return battle


def _dummy(battle, world, x, y, *, unit="Knight", team=Team.RED):
    data, levels, _ = world
    spec = build_unit_spec(data, levels, unit, level=11, rarity="Common", clock=battle.clock)
    entity = Entity(
        kind=spec.kind, team=team, x=tiles(x), y=tiles(y), hitpoints=999_999, spec=spec,
        collision_radius=spec.collision_radius, mass=spec.mass, flying=spec.flying,
    )
    entity.max_hitpoints = 999_999
    entity.deploy_ticks_left = 0
    battle._register(entity)
    return entity


def _hits_on(battle, victim, ticks):
    seen, out = set(), []
    for _ in range(ticks):
        battle.step()
        for event in battle.damage_log:
            key = (event.tick, event.attacker_id, event.target_id)
            if event.target_id == victim.id and key not in seen:
                seen.add(key)
                out.append(event.amount)
    return out


# ------------------------------------------------------------------ charge


@pytest.mark.parametrize("card", ["Prince", "DarkPrince"])
def test_a_charger_doubles_its_first_hit_after_a_run_up(world, card):
    """DamageSpecial is exactly twice Damage for every charger in the build.

    Blocking a Prince before he connects is worth a whole card, and that is
    only true if the charge hit is real.
    """
    battle = _battle(world, card)
    victim = _dummy(battle, world, 9, 18)
    assert battle.play_card(Team.BLUE, card, tiles(9), tiles(12))
    hits = _hits_on(battle, victim, 60 * 25)

    assert len(hits) >= 2, f"only {len(hits)} hits"
    # Not exactly double the *scaled* figure: DamageSpecial is its own base
    # value put through the level ladder, so each is truncated once
    # independently and the pair can differ by a point. Prince at level 11 is
    # 783 against 391, where doubling the scaled number would give 782.
    assert hits[0] - hits[1] * 2 in (0, 1), (
        f"first hit {hits[0]} against follow-ups of {hits[1]}"
    )


@pytest.mark.parametrize("card", ["Prince", "DarkPrince"])
def test_a_charger_placed_in_contact_does_not_get_the_charge_hit(world, card):
    """The run-up is the cost. Without it there is no bonus.

    This is the half that makes the mechanic a decision rather than a stat: a
    Prince dropped straight onto a defender is just an expensive melee unit.
    """
    battle = _battle(world, card)
    victim = _dummy(battle, world, 9, 12.6)
    assert battle.play_card(Team.BLUE, card, tiles(9), tiles(12))
    hits = _hits_on(battle, victim, 60 * 15)

    assert hits, "never attacked"
    assert len(set(hits)) == 1, f"a charge hit landed without a run-up: {hits[:4]}"


def test_connecting_spends_the_charge(world):
    """One charge hit per run-up, not a permanent damage buff."""
    battle = _battle(world, "Prince")
    victim = _dummy(battle, world, 9, 18)
    battle.play_card(Team.BLUE, "Prince", tiles(9), tiles(12))
    hits = _hits_on(battle, victim, 60 * 25)
    assert hits.count(max(hits)) == 1, f"charge hit landed more than once: {hits[:6]}"


def test_a_charger_moves_faster_once_charged(world):
    """ChargeSpeedMultiplier is 200 -- it closes the ground you were counting on.

    Towers are left in: a Prince with nothing in sight and no tower to walk
    toward simply stands still, and never builds a charge to measure.
    """
    battle = _battle(world, "Prince", towers=True)
    battle.play_card(Team.BLUE, "Prince", tiles(3.5), tiles(12))
    for _ in range(90):
        battle.step()
    prince = next(
        e for e in battle.entities if e.team is Team.BLUE and e.kind is EntityKind.TROOP
    )

    def travelled(ticks):
        start = prince.y
        for _ in range(ticks):
            battle.step()
        return to_tiles(prince.y - start)

    walking = travelled(30)
    for _ in range(600):
        if battle._is_charged(prince, prince.spec):
            break
        battle.step()
    else:
        pytest.fail("the Prince never built a charge")
    galloping = travelled(30)
    assert galloping > walking * 1.5, f"walked {walking:.2f}t, galloped {galloping:.2f}t"


def test_a_stun_costs_a_charger_its_run_up(world):
    battle = _battle(world, "Prince", towers=True)
    battle.play_card(Team.BLUE, "Prince", tiles(3.5), tiles(12))
    prince = None
    for _ in range(600):
        battle.step()
        if prince is None:
            prince = next(
                (e for e in battle.entities
                 if e.team is Team.BLUE and e.kind is EntityKind.TROOP),
                None,
            )
        if prince is not None and battle._is_charged(prince, prince.spec):
            break
    else:
        pytest.fail("the Prince never built a charge")

    spec = battle._buff_spec("ZapFreeze", "Common", 11)
    battle._apply_buff(prince, spec, 30, source=0)
    battle.step()
    assert not battle._is_charged(prince, prince.spec), "a stunned Prince kept his charge"


def test_the_battle_ram_charges_a_tower_and_is_spent(world):
    """Its charge hit, its kamikaze death and its two Barbarians are one card."""
    battle = _battle(world, "BattleRam", towers=True)
    assert battle.play_card(Team.BLUE, "BattleRam", tiles(3.5), tiles(12))

    ram_hits, barbarians = [], 0
    for _ in range(60 * 40):
        battle.step()
        for event in battle.damage_log:
            if event.tick != battle.tick - 1:
                continue
            source = battle._entity(event.attacker_id)
            if source is not None and source.spec is not None:
                if source.spec.name == "BattleRam":
                    ram_hits.append(event.amount)
        barbarians = max(barbarians, len([
            e for e in battle.entities
            if e.spec is not None and e.spec.name == "Barbarian"
        ]))

    assert len(ram_hits) == 1, f"the ram hit {len(ram_hits)} times; it is spent on contact"
    assert barbarians == 2, f"{barbarians} Barbarians"


# -------------------------------------------------------------------- ramp


def test_the_ramp_ladder_reads_off_the_spec(world):
    """Stage one is the unit's own Damage; VariableDamage2/3 are the steps."""
    data, levels, _ = world
    spec = build_unit_spec(data, levels, "InfernoTower", level=11, rarity="Rare",
                           kind=EntityKind.BUILDING)
    assert len(spec.variable_damage) == 3
    first, second, third = spec.variable_damage
    assert first < second < third
    assert third > first * 15, "the escalation is the card"

    assert ramp_damage(spec, 0) == first
    assert ramp_damage(spec, spec.variable_damage_ticks[0] - 1) == first
    assert ramp_damage(spec, spec.variable_damage_ticks[0]) == second
    assert ramp_damage(spec, sum(spec.variable_damage_ticks)) == third
    assert ramp_damage(spec, 99_999) == third, "the ladder has a top"


def test_a_unit_with_no_ladder_has_no_ramp(world):
    data, levels, _ = world
    spec = build_unit_spec(data, levels, "Knight", level=11, rarity="Common")
    assert ramp_damage(spec, 10_000) is None


def test_an_inferno_tower_escalates_while_it_holds_one_target(world):
    """Two 2000ms steps: four seconds from tickle to melting a tank."""
    battle = _battle(world, "InfernoTower")
    golem = _dummy(battle, world, 9, 13, unit="Golem")
    assert battle.play_card(Team.BLUE, "InfernoTower", tiles(9), tiles(12))
    hits = _hits_on(battle, golem, 60 * 10)

    assert hits, "the tower never fired"
    steps = [a for a, b in zip(hits, hits[1:]) if a != b]
    assert len(set(hits)) == 3, f"expected three damage stages, saw {sorted(set(hits))}"
    assert hits[0] == min(hits) and hits[-1] == max(hits), "the ramp went the wrong way"
    assert steps == sorted(steps), "damage did not increase monotonically"


def test_retargeting_sends_the_ramp_back_to_the_start(world):
    """The clock belongs to the target, not to the tower.

    An Inferno that kept its stage across targets would melt a whole push
    instead of one tank, which is the opposite of how the card plays.
    """
    state = AttackState()
    data, levels, _ = world
    spec = build_unit_spec(data, levels, "InfernoTower", level=11, rarity="Rare",
                           kind=EntityKind.BUILDING)
    state.engage(spec, target_id=1)
    state.locked_ticks = 500
    assert ramp_damage(spec, state.locked_ticks) == spec.variable_damage[-1]

    state.engage(spec, target_id=2)
    assert state.locked_ticks == 0
    assert ramp_damage(spec, state.locked_ticks) == spec.variable_damage[0]


def test_a_stun_resets_the_ramp_without_dropping_the_target(world):
    """This is precisely what a 500ms Electro Wizard zap buys.

    Losing the target instead would be a different and weaker effect -- the
    tower would re-acquire and pay only its load time. Restarting the burn is
    what makes the reset worth a card.
    """
    battle = _battle(world, "InfernoTower")
    golem = _dummy(battle, world, 9, 13, unit="Golem")
    battle.play_card(Team.BLUE, "InfernoTower", tiles(9), tiles(12))
    for _ in range(60 * 8):
        battle.step()

    tower = next(e for e in battle.entities if e.team is Team.BLUE and e.kind is EntityKind.BUILDING)
    state = battle._attacks[tower.id]
    assert ramp_damage(tower.spec, state.locked_ticks) == tower.spec.variable_damage[-1]
    held = state.target_id

    stun = battle._buff_spec("ZapFreeze", "Common", 11)
    battle._apply_buff(tower, stun, 30, source=0)
    battle.step()

    assert state.locked_ticks == 0, "the stun did not reset the ramp"
    assert state.target_id == held, "the stun dropped the target instead of the ramp"


# ------------------------------------------------- reflected and earned buffs


def test_hitting_an_electro_giant_stuns_you(world):
    """Its ReflectedAttackBuff is a stun, so attacking it is itself a cost.

    The card punishes the defence rather than out-damaging it, and that is
    entirely in the reflect -- its damage is unremarkable for the elixir.
    """
    data, levels, registry = world
    battle = Battle(
        data, levels, registry,
        BattleConfig(seed=1, blue_deck=("Knight",) * 8, red_deck=("ElectroGiant",) * 8),
    )
    battle.entities = [e for e in battle.entities if e.kind is not EntityKind.TOWER]
    battle._towers = {Team.BLUE: [], Team.RED: []}
    battle.players[Team.RED].elixir.add(10)
    knight = _dummy(battle, world, 9, 20.6, team=Team.BLUE)
    assert battle.play_card(Team.RED, "ElectroGiant", tiles(9), tiles(21))

    stunned = 0
    for _ in range(60 * 20):
        battle.step()
        if knight.buffs is not None and knight.buffs.is_frozen():
            stunned += 1
    assert stunned > 0, "attacking an Electro Giant did not stun"


def test_the_after_hits_ladder_is_read_off_the_data(world):
    """Prince's escalating rage: 2 / 4 / 6 hits for 6000 / 4000 / 2000ms.

    No standard card carries this yet -- every user is an evolution or event
    variant, which arrive in M7 -- so the ladder is asserted on the spec
    directly. Pinned here so the mechanism is known to be wired before the
    cards that need it exist.
    """
    data, levels, _ = world
    spec = build_unit_spec(data, levels, "PrinceBuff", level=11, rarity="Epic")
    assert spec.buff_after_hits == (
        "PrinceRageBuff1", "PrinceRageBuff2", "PrinceRageBuff3",
    )
    assert spec.buff_after_hits_count == (2, 4, 6)
    # Later tiers are shorter, so the rage escalates in strength but not in
    # uptime -- it rewards being left alone rather than snowballing forever.
    assert list(spec.buff_after_hits_ticks) == sorted(
        spec.buff_after_hits_ticks, reverse=True
    )


def test_a_single_value_field_reads_as_a_one_entry_ladder(world):
    """These fields are lists for some units and bare values for others.

    Barbarian's evolution carries one entry where Prince carries three, and a
    reader that assumed a list would drop it silently.
    """
    data, levels, _ = world
    spec = build_unit_spec(data, levels, "Barbarian_EV1", level=11, rarity="Common")
    assert len(spec.buff_after_hits) == 1
    assert len(spec.buff_after_hits_count) == 1
