"""M2 gate: targeting, the attack cycle, and towers.

The tests that matter here are *outcomes*, not mechanics in isolation. Whether
a Knight beats a Musketeer is a fact about Clash Royale that thousands of
players know; if the simulator disagrees, something in targeting, range, timing
or damage is wrong and it does not much matter which.

Duels are fought away from any tower so the result isolates the two units.
"""

from __future__ import annotations

import pytest

from cr_sim.data.cards import build_card_registry
from cr_sim.data.leveling import build_level_table
from cr_sim.data.source import LogicData
from cr_sim.engine.battle import KING_ACTIVATION_MS, Battle, BattleConfig
from cr_sim.engine.combat import AttackState
from cr_sim.engine.entity import Entity, EntityKind, Team
from cr_sim.engine.fixed import milli_tiles, tiles, to_tiles
from cr_sim.engine.specs import build_unit_spec
from cr_sim.engine.targeting import can_target, gap_between, in_attack_range

from .test_data_pipeline import BUILD

RARITY = {
    "Knight": "Common", "Musketeer": "Rare", "MiniPekka": "Rare", "Pekka": "Epic",
    "Giant": "Rare", "Valkyrie": "Rare", "Barbarian": "Common", "Skeleton": "Common",
    "Archer": "Common", "Minion": "Common", "Bomber": "Common", "Golem": "Epic",
    "HogRider": "Rare", "BabyDragon": "Epic",
}


@pytest.fixture(scope="module")
def world():
    data = LogicData.load(BUILD)
    return data, build_level_table(data), build_card_registry(data)


def _empty_battle(world, *, seed=1):
    """A battle with the towers removed, so duels isolate the two units."""
    data, levels, registry = world
    battle = Battle(
        data, levels, registry,
        BattleConfig(seed=seed, blue_deck=("Knight",), red_deck=("Knight",)),
    )
    battle.entities = [e for e in battle.entities if e.kind is not EntityKind.TOWER]
    battle._towers = {Team.BLUE: [], Team.RED: []}
    return battle


def _spawn(battle, world, name, team, x, y):
    data, levels, _registry = world
    rarity = RARITY.get(name, "Common")
    scale = levels.get(rarity)
    spec = build_unit_spec(
        data, levels, name,
        level=scale.internal_level(11), rarity=rarity, clock=battle.clock,
    )
    entity = Entity(
        kind=spec.kind, team=team, x=tiles(x), y=tiles(y),
        hitpoints=spec.hitpoints, spec=spec,
        collision_radius=spec.collision_radius, mass=spec.mass,
        flying=spec.flying, shield=spec.shield_hitpoints,
    )
    battle.entities.append(entity)
    return entity


def duel(world, blue_name, red_name, *, gap=2.0, limit=6000):
    battle = _empty_battle(world)
    blue = _spawn(battle, world, blue_name, Team.BLUE, 9, 10.0)
    red = _spawn(battle, world, red_name, Team.RED, 9, 10.0 + gap)
    for _ in range(limit):
        battle.step()
        if blue.dead or red.dead:
            break
    return blue, red, battle


# ------------------------------------------------------------------ outcomes


@pytest.mark.parametrize(
    "winner,loser",
    [
        ("Knight", "Musketeer"),
        ("MiniPekka", "Knight"),
        ("Pekka", "Musketeer"),
        ("Valkyrie", "Barbarian"),
        ("Knight", "Skeleton"),
        ("Knight", "Archer"),
    ],
)
def test_known_duel_outcomes(world, winner, loser):
    """Match-ups every player knows. Getting these wrong means something upstream is."""
    blue, red, _battle = duel(world, winner, loser)
    assert red.dead and not blue.dead, (
        f"{winner} should beat {loser}; got {winner} {blue.hitpoints}hp, {loser} {red.hitpoints}hp"
    )


@pytest.mark.parametrize("name", ["Knight", "Musketeer", "Barbarian"])
def test_a_mirror_match_is_a_draw(world, name):
    """Identical units placed symmetrically must destroy each other.

    A winner here would mean the outcome depends on entity list order -- damage
    landing inline rather than simultaneously within a tick.
    """
    blue, red, _battle = duel(world, name, name)
    assert blue.dead and red.dead, f"{blue.hitpoints} vs {red.hitpoints}"


def test_pekka_one_shots_a_musketeer(world):
    """842 damage against 721 hitpoints -- one swing, and it settles the damage path."""
    blue, red, battle = duel(world, "Pekka", "Musketeer")
    assert red.dead
    lethal = [e for e in battle.damage_log if e.lethal and e.target_id == red.id]
    assert len(lethal) == 1
    hits_on_musketeer = [e for e in battle.damage_log if e.target_id == red.id]
    assert len(hits_on_musketeer) == 1, "took more than one swing"


# ------------------------------------------------------------------ targeting


def test_building_targeting_troops_cannot_see_troops(world):
    """A Giant does not *prefer* buildings -- troops are invisible to it.

    This is why a Giant walks through a Musketeer without breaking stride, and
    why it needs no aggro rules to keep going.
    """
    battle = _empty_battle(world)
    giant = _spawn(battle, world, "Giant", Team.BLUE, 9, 10.0)
    musketeer = _spawn(battle, world, "Musketeer", Team.RED, 9, 11.0)
    assert giant.spec.target_only_buildings
    assert not can_target(giant.spec, giant, musketeer)

    for _ in range(300):
        battle.step()
    assert giant.target_id != musketeer.id
    assert musketeer.hitpoints == musketeer.max_hitpoints, "Giant attacked a troop"


def test_ground_only_units_cannot_hit_flyers(world):
    battle = _empty_battle(world)
    knight = _spawn(battle, world, "Knight", Team.BLUE, 9, 10.0)
    minion = _spawn(battle, world, "Minion", Team.RED, 9, 10.6)
    assert not knight.spec.attacks_air
    assert not can_target(knight.spec, knight, minion)
    for _ in range(400):
        battle.step()
    assert minion.hitpoints == minion.max_hitpoints


def test_flyers_can_be_hit_by_air_capable_units(world):
    battle = _empty_battle(world)
    musketeer = _spawn(battle, world, "Musketeer", Team.BLUE, 9, 10.0)
    minion = _spawn(battle, world, "Minion", Team.RED, 9, 12.0)
    assert musketeer.spec.attacks_air
    for _ in range(600):
        battle.step()
        if minion.dead:
            break
    assert minion.dead


def test_deploying_units_cannot_be_targeted(world):
    battle = _empty_battle(world)
    knight = _spawn(battle, world, "Knight", Team.BLUE, 9, 10.0)
    victim = _spawn(battle, world, "Musketeer", Team.RED, 9, 10.6)
    victim.deploy_ticks_left = 60
    assert not can_target(knight.spec, knight, victim)


def test_range_is_measured_between_hitboxes(world):
    """A larger unit is reachable from further away, because its hitbox is nearer.

    Comparing centre-to-centre distances instead would make every big unit
    harder to reach than it actually is.
    """
    battle = _empty_battle(world)
    attacker = _spawn(battle, world, "Knight", Team.BLUE, 9, 10.0)
    small = _spawn(battle, world, "Skeleton", Team.RED, 9, 12.0)
    large = _spawn(battle, world, "Golem", Team.RED, 9, 12.0)
    assert large.collision_radius > small.collision_radius
    assert gap_between(attacker, large) < gap_between(attacker, small)


# -------------------------------------------------------------- attack timing


def test_first_hit_waits_the_load_time_not_the_hit_speed(world):
    """Knight winds up 700ms before its first swing, then 1200ms between swings.

    A unit repeatedly forced to re-engage never gets past the windup, which is
    the whole reason distraction works.
    """
    data, levels, _registry = world
    battle = _empty_battle(world)
    attacker = _spawn(battle, world, "Knight", Team.BLUE, 9, 10.0)
    victim = _spawn(battle, world, "Knight", Team.RED, 9, 10.9)
    assert attacker.spec.load_time_ticks == 42  # 700ms at 60 TPS
    assert attacker.spec.hit_speed_ticks == 72  # 1200ms

    first = second = None
    for _ in range(400):
        battle.step()
        hits = [e for e in battle.damage_log if e.attacker_id == attacker.id]
        if first is None and hits:
            first = hits[0].tick
        if first is not None and len(hits) >= 2:
            second = hits[1].tick
            break
    assert first == 41, f"first hit at tick {first}, expected the load time"
    assert second - first == 72, f"gap {second - first}, expected the hit speed"


def test_switching_target_restarts_the_windup(world):
    """Being pulled onto a new target costs the first-hit delay again."""
    state = AttackState()
    battle = _empty_battle(world)
    knight = _spawn(battle, world, "Knight", Team.BLUE, 9, 10.0)
    state.engage(knight.spec, target_id=5)
    assert state.cooldown == knight.spec.load_time_ticks
    state.cooldown = 3
    state.engage(knight.spec, target_id=9)  # new target
    assert state.cooldown == knight.spec.load_time_ticks


def test_a_unit_killed_mid_tick_does_not_keep_fighting(world):
    """Death must survive the rest of the tick.

    A fatally-hit unit that acts later in the same tick used to overwrite its
    own DYING state, so the death sweep never saw it and it fought on at zero
    hitpoints forever.
    """
    from cr_sim.engine.entity import EntityState

    battle = _empty_battle(world)
    victim = _spawn(battle, world, "Knight", Team.BLUE, 9, 10.0)
    victim.apply_damage(victim.hitpoints)
    assert victim.state is EntityState.DYING
    victim.set_state(EntityState.ATTACKING)
    assert victim.state is EntityState.DYING, "a dead unit came back to life"


# ------------------------------------------------------------------- towers


def test_a_lone_musketeer_loses_to_a_princess_tower(world):
    """She out-ranges nothing here: the tower wins the trade and she chips it."""
    data, levels, registry = world
    battle = Battle(
        data, levels, registry,
        BattleConfig(seed=3, blue_deck=("Musketeer",) * 8, red_deck=("Musketeer",) * 8),
    )
    musketeer = _spawn(battle, world, "Musketeer", Team.BLUE, 3.5, 14.0)
    tower = next(
        e for e in battle.entities
        if e.kind is EntityKind.TOWER and e.team is Team.RED
        and "Princess" in e.spec.name and abs(to_tiles(e.x) - 3.5) < 1
    )
    for _ in range(3000):
        battle.step()
        if musketeer.dead or tower.dead:
            break
    assert musketeer.dead, "a lone Musketeer should not survive a Princess Tower"
    assert tower.hitpoints < tower.max_hitpoints, "she should have chipped it"
    assert not tower.dead


def test_king_tower_is_inert_until_provoked(world):
    """The King sits out until it is damaged or loses a Princess Tower.

    That is why chip damage onto the King is a genuine commitment, and why
    taking a tower changes the defensive geometry of that whole side.
    """
    data, levels, registry = world
    battle = Battle(
        data, levels, registry,
        BattleConfig(seed=4, blue_deck=("Knight",) * 8, red_deck=("Knight",) * 8),
    )
    assert not battle._king_active[Team.RED]
    king = battle._king(Team.RED)

    # Walk a unit deep into range of the King but not of a Princess Tower.
    intruder = _spawn(battle, world, "Musketeer", Team.BLUE, 9, 24.0)
    for _ in range(120):
        battle.step()
    assert king.target_id == 0, "an inert King acquired a target"

    # Damaging the King starts it waking -- but not instantly.
    king.apply_damage(1)
    battle.step()
    assert not battle._king_active[Team.RED], "the King woke with no delay"
    for _ in range(battle.clock.ticks(KING_ACTIVATION_MS) + 2):
        battle.step()
    assert battle._king_active[Team.RED]
    assert intruder is not None


def test_king_activation_takes_3300ms(world):
    """The wake-up delay is why a tower trade can finish before the King fires.

    This build dropped the old KING_ACTIVATE_TIME_MS global; the duration now
    lives in the King Tower's action graph as an ActionWithDuration of 3300ms.
    """
    data, levels, registry = world
    battle = Battle(
        data, levels, registry,
        BattleConfig(seed=8, blue_deck=("Knight",) * 8, red_deck=("Knight",) * 8),
    )
    battle._king(Team.RED).apply_damage(1)
    woke = None
    for _ in range(600):
        battle.step()
        if battle._king_active[Team.RED]:
            woke = battle.tick
            break
    expected = battle.clock.ticks(KING_ACTIVATION_MS)
    assert woke is not None
    assert abs(woke - expected) <= 2, f"woke at {woke}, expected about {expected}"


def test_losing_a_princess_tower_wakes_the_king(world):
    data, levels, registry = world
    battle = Battle(
        data, levels, registry,
        BattleConfig(seed=5, blue_deck=("Knight",) * 8, red_deck=("Knight",) * 8),
    )
    princess = next(
        e for e in battle._towers[Team.RED] if "King" not in e.spec.name
    )
    assert not battle._king_active[Team.RED]
    princess.kill()
    for _ in range(battle.clock.ticks(KING_ACTIVATION_MS) + 3):
        battle.step()
    assert battle._king_active[Team.RED]


def test_destroying_a_tower_scores_a_crown(world):
    data, levels, registry = world
    battle = Battle(
        data, levels, registry,
        BattleConfig(seed=6, blue_deck=("Knight",) * 8, red_deck=("Knight",) * 8),
    )
    assert battle.players[Team.BLUE].crowns == 0
    next(e for e in battle._towers[Team.RED] if "King" not in e.spec.name).kill()
    battle.step()
    assert battle.players[Team.BLUE].crowns == 1


def test_destroying_the_king_ends_the_battle(world):
    data, levels, registry = world
    battle = Battle(
        data, levels, registry,
        BattleConfig(seed=7, blue_deck=("Knight",) * 8, red_deck=("Knight",) * 8),
    )
    battle._king(Team.RED).kill()
    battle.step()
    assert battle.finished
    assert battle.result.winner is Team.BLUE


# ---------------------------------------------------------------- determinism


def test_combat_is_deterministic(world):
    """Fights must not introduce any order or timing dependence."""
    from cr_sim.replay import compare_hashes

    runs = []
    for _ in range(2):
        battle = _empty_battle(world)
        _spawn(battle, world, "Knight", Team.BLUE, 9, 10.0)
        _spawn(battle, world, "Musketeer", Team.RED, 9, 12.0)
        runs.append([(battle.step(), battle.hash())[1] for _ in range(600)])
    assert compare_hashes(runs[0], runs[1]) is None


def test_crown_tower_damage_reduction_applies_in_combat(world):
    """A spell-like reduction must reach real damage application, not just the spec."""
    import dataclasses

    battle = _empty_battle(world)
    attacker = _spawn(battle, world, "Knight", Team.BLUE, 9, 10.0)
    assert attacker.spec.damage_to(is_crown_tower=True) == attacker.spec.damage
    reduced = dataclasses.replace(attacker.spec, crown_tower_damage_percent=-75)
    assert reduced.damage_to(is_crown_tower=True) == reduced.damage // 4


# ------------------------------------------------- swarms and kamikaze units


@pytest.mark.parametrize(
    "card,expected",
    [("Skeletons", 3), ("Goblins", 4), ("Barbarians", 5), ("MinionHorde", 6),
     ("GoblinGang", 6), ("SkeletonArmy", 15), ("ThreeMusketeers", 3), ("Rascals", 3)],
)
def test_swarm_cards_deploy_their_full_count(world, card, expected):
    data, levels, registry = world
    battle = Battle(
        data, levels, registry,
        BattleConfig(seed=1, blue_deck=(card,) * 8, red_deck=(card,) * 8),
    )
    battle.players[Team.BLUE].elixir.add(10)
    before = len(battle.entities)
    assert battle.play_card(Team.BLUE, card, tiles(9), tiles(10))
    assert len(battle.entities) - before == expected


def test_swarm_units_spread_instead_of_stacking(world):
    """SummonRadius rings a swarm out; stacked units would share one point.

    Beyond looking wrong, perfectly overlapping units mean every splash hits
    the entire group, which would badly distort spell value.
    """
    data, levels, registry = world
    battle = Battle(
        data, levels, registry,
        BattleConfig(seed=1, blue_deck=("Skeletons",) * 8, red_deck=("Knight",) * 8),
    )
    battle.players[Team.BLUE].elixir.add(10)
    battle.play_card(Team.BLUE, "Skeletons", tiles(9), tiles(10))
    skeletons = [e for e in battle.entities if e.spec and e.spec.name == "Skeleton"]
    assert len(skeletons) == 3
    positions = {(e.x, e.y) for e in skeletons}
    assert len(positions) == 3, "all three spawned on the same point"
    # And they sit roughly on the card's SummonRadius (700 milli-tiles).
    for entity in skeletons:
        offset = ((entity.x - tiles(9)) ** 2 + (entity.y - tiles(10)) ** 2) ** 0.5
        assert abs(offset - milli_tiles(700)) < milli_tiles(60)


def test_swarm_units_deploy_staggered(world):
    """SummonDeployDelay makes a swarm arrive in sequence, not all at once."""
    data, levels, registry = world
    battle = Battle(
        data, levels, registry,
        BattleConfig(seed=1, blue_deck=("GoblinGang",) * 8, red_deck=("Knight",) * 8),
    )
    battle.players[Team.BLUE].elixir.add(10)
    battle.play_card(Team.BLUE, "GoblinGang", tiles(9), tiles(10))
    goblins = [e for e in battle.entities if e.spec and "Goblin" in e.spec.name]
    delays = sorted(e.deploy_ticks_left for e in goblins)
    assert len(set(delays)) > 1, "every unit deployed on the same tick"
    assert delays == sorted(delays)


@pytest.mark.parametrize("name", ["IceSpirits", "FireSpirits"])
def test_kamikaze_units_are_consumed_by_their_attack(world, name):
    """Ice Spirit and Fire Spirits land one hit and die.

    Without this they keep swinging forever, turning a one-shot utility card
    into a permanent damage dealer.
    """
    battle = _empty_battle(world)
    bomber = _spawn(battle, world, name, Team.BLUE, 9, 10.0)
    victim = _spawn(battle, world, "Knight", Team.RED, 9, 11.0)
    assert bomber.spec.kamikaze

    for _ in range(1200):
        battle.step()
        if bomber.dead:
            break
    assert bomber.dead, "kamikaze unit survived its own attack"
    hits = [e for e in battle.damage_log if e.attacker_id == bomber.id]
    assert len(hits) == 1, f"landed {len(hits)} hits, expected exactly 1"
    assert victim.hitpoints < victim.max_hitpoints


def test_wall_breakers_need_a_building_to_detonate_on(world):
    """Wall Breakers are kamikaze *and* building-targeting.

    Against a troop they have no valid target at all, so they never detonate --
    they simply run past. Pairing the two rules is the whole card.
    """
    battle = _empty_battle(world)
    breaker = _spawn(battle, world, "Wallbreaker", Team.BLUE, 9, 10.0)
    knight = _spawn(battle, world, "Knight", Team.RED, 9, 11.0)
    assert breaker.spec.kamikaze and breaker.spec.target_only_buildings
    assert not can_target(breaker.spec, breaker, knight)

    for _ in range(600):
        battle.step()
    # It may well be killed by the Knight -- what matters is that it never
    # detonated, because it never had a target.
    hits = [e for e in battle.damage_log if e.attacker_id == breaker.id]
    assert hits == [], "detonated on a troop it cannot even see"
    assert knight.hitpoints == knight.max_hitpoints


def test_wall_breakers_detonate_on_a_tower(world):
    data, levels, registry = world
    battle = Battle(
        data, levels, registry,
        BattleConfig(seed=9, blue_deck=("Wallbreakers",) * 8, red_deck=("Knight",) * 8),
    )
    breaker = _spawn(battle, world, "Wallbreaker", Team.BLUE, 3.5, 24.0)
    tower = next(
        e for e in battle._towers[Team.RED]
        if "King" not in e.spec.name and abs(to_tiles(e.x) - 3.5) < 1
    )
    before = tower.hitpoints
    for _ in range(1200):
        battle.step()
        if breaker.dead:
            break
    assert breaker.dead
    assert tower.hitpoints < before, "did not damage the tower it blew up on"
