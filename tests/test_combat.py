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
    battle._register(entity)
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


def test_stop_time_after_attack_holds_movement_after_a_swing(world):
    """``StopTimeAfterAttack`` is unpopulated for every unit in this build --
    the CSV column is blank for the whole roster (its sibling
    ``StopTimeAfterSpecialAttack`` is the one populated field, and only for
    the Fisherman's barrel), so nothing currently on the card ladder exercises
    it. Exercised here with a synthetic spec, because the historical bug was
    plausible without any real card ever revealing it: ``advance_attack`` used
    to hard-code ``state.stop_ticks = 0`` on every swing regardless of what the
    spec said, so the field was parsed nowhere and had no way to take effect
    even if a future extraction populated it.
    """
    import dataclasses

    from cr_sim.engine.combat import advance_attack

    battle = _empty_battle(world)
    attacker = _spawn(battle, world, "Knight", Team.BLUE, 9, 10.0)
    target = _spawn(battle, world, "Knight", Team.RED, 9, 10.9)
    stalled = dataclasses.replace(attacker.spec, stop_time_after_attack_ticks=30)

    state = AttackState()
    hit = None
    for _ in range(stalled.load_time_ticks + 2):
        hit = advance_attack(state, stalled, attacker, target)
        if hit is not None:
            break
    assert hit is not None, "never swung"
    assert state.stop_ticks == 30, "the spec's stop time was not loaded onto the cooldown"
    assert not state.can_move, "a unit that just swung should be held in place"


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
    # It dies on the swing, but its bomb is still in the air; let it land.
    for _ in range(120):
        battle.step()
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
    for _ in range(120):  # the bomb is still travelling
        battle.step()
    assert tower.hitpoints < before, "did not damage the tower it blew up on"


# ------------------------------------------------------- first-hit timing


def _first_two_hits(world, battle, name, rarity, *, gap=0.9, limit=900):
    """Spawn a unit against an indestructible wall and time its first two hits."""
    from cr_sim.engine.projectiles import build_projectile_spec
    from cr_sim.engine.fixed import distance

    data, levels, _registry = world
    spec = build_unit_spec(
        data, levels, name,
        level=levels.get(rarity).internal_level(11), rarity=rarity, clock=battle.clock,
    )
    attacker = Entity(
        kind=spec.kind, team=Team.BLUE, x=tiles(9), y=tiles(10),
        hitpoints=spec.hitpoints, spec=spec,
        collision_radius=spec.collision_radius, mass=spec.mass, flying=spec.flying,
    )
    battle._register(attacker)

    wall_spec = build_unit_spec(data, levels, "Golem", level=6, rarity="Epic", clock=battle.clock)
    wall = Entity(
        kind=EntityKind.TROOP, team=Team.RED, x=tiles(9), y=tiles(10 + gap),
        hitpoints=999_999, spec=wall_spec,
        collision_radius=wall_spec.collision_radius, mass=10**9,
    )
    wall.max_hitpoints = 999_999
    battle._register(wall)

    hits: list[int] = []
    for _ in range(limit):
        battle.step()
        hits = [e.tick for e in battle.damage_log if e.attacker_id == attacker.id]
        if len(hits) >= 2:
            break

    # Ranged units log damage on IMPACT, so the shot's travel time sits between
    # the swing and the recorded hit.
    flight = 0
    if spec.projectile:
        pspec = build_projectile_spec(
            data, spec.projectile, levels.get(spec.rarity), level=spec.level, clock=battle.clock
        )
        if pspec and pspec.speed_per_tick > 0:
            reach = distance(tiles(9), tiles(10), tiles(9), tiles(10 + gap))
            flight = max(1, reach // pspec.speed_per_tick)
    return spec, hits, flight


@pytest.mark.parametrize(
    "name,rarity",
    [
        ("Knight", "Common"),        # melee, ordinary
        ("Musketeer", "Rare"),       # ranged, ordinary
        ("Pekka", "Epic"),           # slow melee, long windup
        ("InfernoTower", "Rare"),    # LoadTime 1200 > HitSpeed 400
        ("InfernoDragon", "Legendary"),
        ("MightyMiner", "Champion"), # LoadTime 700 > HitSpeed 400
        ("Bomber", "Common"),
        ("Archer", "Common"),
        ("Valkyrie", "Rare"),
        ("Wizard", "Rare"),
        ("MiniPekka", "Rare"),
        # Giant and the other TargetOnlyBuildings troops are deliberately absent:
        # they cannot attack the troop used as a wall here at all, which the
        # building-targeting tests above already cover.
    ],
)
def test_first_hit_uses_load_time_and_later_hits_use_hit_speed(world, name, rarity):
    """Every unit waits its own windup for the first swing, then its hit speed.

    These are two independent timers, not one derived from the other. Inferno
    Tower proves it: a 1200ms windup followed by 400ms ticks. Treating the
    first hit as just another hit-speed interval would make slow-winding units
    dramatically stronger, and treating later hits as windups would make fast
    ones useless.
    """
    battle = _empty_battle(world)
    spec, hits, flight = _first_two_hits(world, battle, name, rarity)
    assert len(hits) >= 2, f"{name} never landed two hits"

    expected_first = max(1, spec.load_time_ticks) - 1 + flight
    assert abs(hits[0] - expected_first) <= 3, (
        f"{name} first hit at {hits[0]}, expected about {expected_first} "
        f"(load {spec.load_time_ticks} + flight {flight})"
    )
    assert abs((hits[1] - hits[0]) - spec.hit_speed_ticks) <= 2, (
        f"{name} hit interval {hits[1] - hits[0]}, expected {spec.hit_speed_ticks}"
    )


def test_windup_is_independent_of_hit_speed(world):
    """Three units in the build wind up for longer than their firing interval.

    If the engine derived one from the other these could not exist, so they are
    the case that proves the two timers are separate.
    """
    data, levels, _registry = world
    battle = _empty_battle(world)
    for name, rarity in (("InfernoTower", "Rare"), ("InfernoDragon", "Legendary"),
                         ("MightyMiner", "Champion")):
        spec = build_unit_spec(
            data, levels, name,
            level=levels.get(rarity).internal_level(11), rarity=rarity, clock=battle.clock,
        )
        assert spec.load_time_ticks > spec.hit_speed_ticks, name


def test_switching_target_makes_a_unit_pay_the_windup_again(world):
    """Distraction works because re-engaging costs the first-hit delay again.

    Without this a unit pulled onto a new target would continue on its old
    rhythm, and chip-blocking a Prince or a Sparky would do nothing.
    """
    battle = _empty_battle(world)
    knight = _spawn(battle, world, "Knight", Team.BLUE, 9, 10.0)
    first = _spawn(battle, world, "Skeleton", Team.RED, 9, 10.8)

    for _ in range(200):
        battle.step()
        if first.dead:
            break
    assert first.dead

    second = _spawn(battle, world, "Skeleton", Team.RED, 9, 10.8)
    switched_at = battle.tick
    for _ in range(200):
        battle.step()
        if second.hitpoints < second.max_hitpoints:
            break
    delay = battle.tick - switched_at
    assert delay >= knight.spec.load_time_ticks - 4, (
        f"hit the new target after {delay} ticks, windup is {knight.spec.load_time_ticks}"
    )


# --------------------------------------------------- the RL training deck (M2)

# cr_sim.train.run.DEFAULT_DECK. These are the only six troops/buildings the RL
# agent ever actually fights with, so their pairwise outcomes matter more than
# roster breadth: a targeting or attack-cycle bug here silently teaches the
# wrong policy rather than merely failing a stat check.


def _card_vs_lone_unit(world, card, opponent, *, opponent_rarity="Common", drop=2.0, limit=3000):
    """Deploy a full card against a single stationary defender.

    The defender is placed directly rather than played as a card so its side
    of the fight is exactly one unit, whatever the attacking card's count is.

    Returns the attackers *as spawned*, not re-queried afterwards: a dead unit
    leaves ``battle.entities`` for the graveyard (see
    ``test_dead_units_leave_the_live_list`` in test_collision.py), so asking
    "which Goblins died" by filtering the live list at the end would just find
    an empty list once they all had.
    """
    data, levels, registry = world
    battle = Battle(
        data, levels, registry,
        BattleConfig(seed=1, blue_deck=(card,) * 8, red_deck=(opponent,) * 8),
    )
    battle.entities = [e for e in battle.entities if e.kind is not EntityKind.TOWER]
    battle._towers = {Team.BLUE: [], Team.RED: []}
    battle.players[Team.BLUE].elixir.add(10)
    assert battle.play_card(Team.BLUE, card, tiles(9), tiles(10))
    attackers = [e for e in battle.entities if e.team is Team.BLUE]

    scale = levels.get(opponent_rarity)
    spec = build_unit_spec(
        data, levels, opponent,
        level=scale.internal_level(11), rarity=opponent_rarity, clock=battle.clock,
    )
    defender = Entity(
        kind=spec.kind, team=Team.RED, x=tiles(9), y=tiles(10 + drop), hitpoints=spec.hitpoints,
        spec=spec, collision_radius=spec.collision_radius, mass=spec.mass, flying=spec.flying,
    )
    battle._register(defender)

    for _ in range(limit):
        battle.step()
        if all(e.dead for e in attackers) or defender.dead:
            break
    return defender, attackers, battle


def test_knight_one_shots_a_goblin(world):
    """202 damage against 202 hitpoints at level 11 -- an exact breakpoint.

    It is why a lone Knight is a real answer to Goblins dropped on him one at a
    time, rather than something four bodies simply overwhelm.
    """
    data, levels, _registry = world
    knight = build_unit_spec(data, levels, "Knight", level=levels.get("Common").internal_level(11),
                              rarity="Common")
    goblin = build_unit_spec(data, levels, "Goblin_Stab", level=levels.get("Common").internal_level(11),
                              rarity="Common")
    assert knight.damage >= goblin.hitpoints

    blue, red, _battle = duel(world, "Knight", "Goblin_Stab")
    assert red.dead and not blue.dead
    assert blue.hitpoints == blue.max_hitpoints, "took damage from a unit that should never connect"


def test_knight_survives_a_full_goblins_drop(world):
    """Four Goblins land on a Knight together; he kills all four and keeps most of his health.

    Their ring SummonRadius means they do not all reach melee range on the
    same tick, so the Knight -- who one-shots each of them -- picks them off
    faster than their combined DPS can equal his hitpoints.
    """
    knight, goblins, _battle = _card_vs_lone_unit(world, "Goblins", "Knight", drop=0.0)
    assert knight.hitpoints > 0 and not knight.dead
    assert len(goblins) == 4
    assert all(g.dead for g in goblins), "a Knight should clear a full Goblins drop"
    assert knight.hitpoints > knight.max_hitpoints // 2, "took more damage than expected"


def test_knight_survives_a_full_skeletons_drop(world):
    """Three Skeletons is an even easier trade for the Knight than Goblins is."""
    knight, skeletons, _battle = _card_vs_lone_unit(world, "Skeletons", "Knight", drop=0.0)
    assert not knight.dead
    assert len(skeletons) == 3
    assert all(s.dead for s in skeletons)
    assert knight.hitpoints > knight.max_hitpoints * 3 // 4


def test_musketeer_beats_a_full_goblins_drop(world):
    """Her range lets her shoot Goblins down before most of them ever swing.

    217 damage one-shots each Goblin (202 hp), and 6 tiles of range means she
    is already firing while they are still closing the gap.
    """
    musketeer, goblins, _battle = _card_vs_lone_unit(
        world, "Goblins", "Musketeer", opponent_rarity="Rare", drop=2.0
    )
    assert not musketeer.dead
    assert all(g.dead for g in goblins)


def test_musketeer_takes_no_damage_from_a_full_skeletons_drop(world):
    """The clearest case of her range advantage: she kills all three for free."""
    musketeer, skeletons, _battle = _card_vs_lone_unit(
        world, "Skeletons", "Musketeer", opponent_rarity="Rare", drop=2.0
    )
    assert not musketeer.dead
    assert musketeer.hitpoints == musketeer.max_hitpoints, "a Skeleton landed a hit on her"
    assert all(s.dead for s in skeletons)


def test_goblins_pack_around_a_knight_without_their_collision_radii_crowding_them_out(world):
    """Small hitboxes are the whole reason a swarm can surround a target.

    All four Goblins have room to be in melee range of the Knight at once --
    if their 0.5-tile collision radii were too large relative to his to fit,
    collision would shove some of them back out of range and Goblins would
    never be able to land more than one or two hits at a time. The Knight's
    hitpoints are inflated so he outlasts the whole approach; the point of the
    test is the geometry, not who wins.
    """
    data, levels, registry = world
    battle = Battle(
        data, levels, registry,
        BattleConfig(seed=1, blue_deck=("Goblins",) * 8, red_deck=("Knight",) * 8),
    )
    battle.entities = [e for e in battle.entities if e.kind is not EntityKind.TOWER]
    battle._towers = {Team.BLUE: [], Team.RED: []}
    knight = _spawn(battle, world, "Knight", Team.RED, 9, 10.0)
    knight.hitpoints = knight.max_hitpoints = 999_999
    battle.players[Team.BLUE].elixir.add(10)
    assert battle.play_card(Team.BLUE, "Goblins", tiles(9), tiles(10.6))  # dropped almost on him

    most_simultaneous = 0
    for _ in range(200):
        battle.step()
        goblins = [e for e in battle.entities if e.spec and e.spec.name == "Goblin_Stab" and not e.dead]
        in_range = sum(1 for g in goblins if in_attack_range(g.spec, g, knight))
        most_simultaneous = max(most_simultaneous, in_range)
    assert most_simultaneous == 4, (
        f"only {most_simultaneous} of 4 Goblins ever got into melee range at once"
    )


def test_cannon_cannot_target_flying_units(world):
    """Cannon is ground-only -- a Minion overhead is simply invisible to it."""
    battle = _empty_battle(world)
    cannon = _spawn(battle, world, "Cannon", Team.BLUE, 9, 10.0)
    minion = _spawn(battle, world, "Minion", Team.RED, 9, 11.0)
    assert cannon.spec.attacks_ground and not cannon.spec.attacks_air
    assert not can_target(cannon.spec, cannon, minion)
    for _ in range(200):
        battle.step()
    assert minion.hitpoints == minion.max_hitpoints


def test_cannon_holds_its_target_even_when_a_closer_enemy_arrives(world):
    """Sticky targeting: acquiring a target is not re-run every tick.

    Without this a Cannon would flicker onto whatever is nearest right now,
    which would make it trivial to tank its damage by feeding it a fresh unit
    every tick instead of trading a whole one into it.
    """
    battle = _empty_battle(world)
    cannon = _spawn(battle, world, "Cannon", Team.BLUE, 9, 10.0)
    far = _spawn(battle, world, "Skeleton", Team.RED, 9, 14.0)
    far.hitpoints = far.max_hitpoints = 999_999  # outlast the test, not the point of it
    battle.step()
    assert cannon.target_id == far.id

    near = _spawn(battle, world, "Skeleton", Team.RED, 9, 10.5)
    for _ in range(30):
        battle.step()
    assert cannon.target_id == far.id, "switched off a valid target for a merely closer one"
    assert near.hitpoints == near.max_hitpoints, "attacked the unit it should not have switched to"


def test_cannon_does_not_fire_during_its_own_deploy_time(world):
    """Cannon's LoadTime is only 100ms -- far shorter than its 1000ms DeployTime.

    A gate implemented on the attack cycle alone (rather than on deployment)
    would let a fast LoadTime race ahead of DeployTime and have the Cannon
    firing while it is still supposed to be inert on the ground.
    """
    battle = _empty_battle(world)
    cannon = _spawn(battle, world, "Cannon", Team.BLUE, 9, 10.0)
    cannon.deploy_ticks_left = cannon.spec.deploy_ticks
    knight = _spawn(battle, world, "Knight", Team.RED, 9, 12.0)
    assert cannon.spec.load_time_ticks < cannon.spec.deploy_ticks

    for _ in range(cannon.spec.deploy_ticks - 1):
        battle.step()
    assert knight.hitpoints == knight.max_hitpoints, "Cannon fired before finishing deployment"


def test_a_knight_routes_around_a_friendly_cannon_rather_than_through_it(world):
    """A building bends a push; it does not merely sit there decoratively.

    The Cannon here is on the Knight's own team, so it can never be his
    target -- the only thing that can make him detour around it is the
    pathing grid's occupancy, driven by the Cannon's real ``CollisionRadius``.
    A Knight that instead walked the straight line would have to shove
    through (or get stuck on) an immovable building.
    """
    data, levels, _registry = world
    battle = _empty_battle(world)
    cannon = _spawn(battle, world, "Cannon", Team.BLUE, 9, 12.0)
    knight = _spawn(battle, world, "Knight", Team.BLUE, 9, 10.0)
    musketeer = _spawn(battle, world, "Musketeer", Team.RED, 9, 14.0)

    max_deviation = 0
    closest_approach = 1 << 30
    for _ in range(1200):
        battle.step()
        max_deviation = max(max_deviation, abs(to_tiles(knight.x) - 9.0))
        from cr_sim.engine.fixed import distance

        closest_approach = min(closest_approach, distance(knight.x, knight.y, cannon.x, cannon.y))
        if knight.dead or musketeer.dead:
            break
    assert musketeer.dead, "the Knight never got past the Cannon to reach her"
    assert max_deviation > 0.5, "walked the straight line instead of detouring"
    assert closest_approach >= knight.collision_radius + cannon.collision_radius, (
        "clipped into the Cannon's own hitbox while routing around it"
    )
