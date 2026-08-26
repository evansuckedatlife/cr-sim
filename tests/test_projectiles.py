"""Projectiles: flight time, splash, homing, and cleanup.

Before this milestone every ranged attack resolved the instant the swing was
decided -- a Musketeer's arrow and a Mortar shell landed on the same tick they
were loosed. That is wrong in a way that matters: instant resolution deletes
dodging, deletes "the shell is already committed to where the push *was*", and
makes a Mortar mechanically identical to a Musketeer except for its numbers.

These tests check the things instant resolution could never have gotten right:
that damage genuinely waits for a shot to cross the gap, that a projectile is
a real entity while it travels, that splash punishes a clump from one shot
while a single-target arrow does not, that homing re-aims and non-homing does
not, that a shot in flight survives its target dying, and that none of this
leaks entities or desyncs the simulation.

COPY the ``world`` / ``_empty_battle`` / ``_spawn`` fixtures from
``test_combat.py`` rather than importing them: this file owns its own duels so
it isolates projectile behaviour the same way M2's tests isolate targeting.
"""

from __future__ import annotations

import pytest

from cr_sim.data.cards import build_card_registry
from cr_sim.data.leveling import build_level_table
from cr_sim.data.source import LogicData
from cr_sim.engine.battle import Battle, BattleConfig
from cr_sim.engine.constants import TickClock
from cr_sim.engine.entity import Entity, EntityKind, Team
from cr_sim.engine.fixed import distance, tiles
from cr_sim.engine.projectiles import Projectile, build_projectile_spec, flight_ticks
from cr_sim.engine.specs import build_unit_spec
from cr_sim.engine.targeting import can_target

from .test_data_pipeline import BUILD

RARITY = {
    "Knight": "Common",
    "Musketeer": "Rare",
    "Bomber": "Common",
    "Skeleton": "Common",
    "FireSpirits": "Common",
}


@pytest.fixture(scope="module")
def world():
    data = LogicData.load(BUILD)
    return data, build_level_table(data), build_card_registry(data)


def _empty_battle(world, *, seed=1):
    """A battle with the towers removed, so duels isolate the units in play."""
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


def _live_projectiles(battle):
    return [e for e in battle.entities if e.kind is EntityKind.PROJECTILE]


# --------------------------------------------------------- delayed resolution


def test_ranged_damage_waits_for_the_shot_to_arrive(world):
    """A Musketeer's first hit must land well after her load time alone.

    Under the old instant-resolution model the arrow and the swing were the
    same event, so the first hit would land at ``load_time_ticks - 1`` exactly
    like a melee unit's. It must instead wait for the shot to cross the gap --
    the whole reason a target can duck out from under it.
    """
    battle = _empty_battle(world)
    musketeer = _spawn(battle, world, "Musketeer", Team.BLUE, 9, 10.0)
    victim = _spawn(battle, world, "Musketeer", Team.RED, 9, 15.5)

    first = None
    for _ in range(200):
        battle.step()
        hits = [e for e in battle.damage_log if e.attacker_id == musketeer.id]
        if hits:
            first = hits[0].tick
            break

    assert first is not None, "the shot never landed"
    assert first > musketeer.spec.load_time_ticks, (
        "damage must not land on the load tick alone -- the shot still has to travel"
    )

    pspec = battle._projectile_spec("MusketeerProjectile", "Rare", musketeer.spec.level)
    expected_flight = flight_ticks(
        distance(musketeer.x, musketeer.y, victim.x, victim.y), pspec.speed, battle.clock
    )
    # A melee swing of this load time would land at load_time_ticks - 1; the
    # shot should land that many ticks later, plus roughly its flight time.
    expected = musketeer.spec.load_time_ticks - 1 + expected_flight
    assert abs(first - expected) <= 2, (
        f"first hit at {first}, expected about {expected} (load + flight)"
    )


def test_melee_units_deal_damage_on_the_swing_with_no_projectile(world):
    """A Knight has no ``projectile`` and must keep hitting instantly.

    This is the control for the delay test above: if melee damage also waited
    for something to arrive, a Knight's whole timing model would be wrong.
    """
    battle = _empty_battle(world)
    attacker = _spawn(battle, world, "Knight", Team.BLUE, 9, 10.0)
    _spawn(battle, world, "Knight", Team.RED, 9, 10.9)
    assert attacker.spec.projectile is None

    first = None
    for _ in range(100):
        battle.step()
        assert not _live_projectiles(battle), (
            "a melee swing must never create a projectile entity"
        )
        hits = [e for e in battle.damage_log if e.attacker_id == attacker.id]
        if hits:
            first = hits[0].tick
            break

    assert first == attacker.spec.load_time_ticks - 1, (
        "melee damage must land the instant the swing completes, not later"
    )


# ------------------------------------------------------- the shot as an entity


def test_projectile_entity_exists_between_swing_and_impact(world):
    """A shot in flight is a real entity on the board, not a deferred number.

    It has to be, so it shows up in the state hash and the replay viewer --
    and so a splash shot can be resolved from *where it actually is* rather
    than from the attacker's position or the target's.
    """
    battle = _empty_battle(world)
    musketeer = _spawn(battle, world, "Musketeer", Team.BLUE, 9, 10.0)
    _spawn(battle, world, "Knight", Team.RED, 9, 15.5)

    launch_tick = None
    seen_in_flight = False
    impact_tick = None
    for _ in range(200):
        battle.step()
        if _live_projectiles(battle):
            if launch_tick is None:
                launch_tick = battle.tick
            seen_in_flight = True
        hits = [e for e in battle.damage_log if e.attacker_id == musketeer.id]
        if hits:
            impact_tick = hits[0].tick
            break

    assert launch_tick is not None, "no shot was ever launched"
    assert seen_in_flight, "the shot must exist as an entity while it travels"
    assert impact_tick is not None and impact_tick > launch_tick, (
        "flight must take measurable time"
    )

    for _ in range(10):
        battle.step()
    assert not _live_projectiles(battle), "a landed shot must not linger on the board"


# ------------------------------------------------------------------- splash


def test_splash_projectile_damages_the_whole_clump(world):
    """A Bomber's shell hits everything near where it lands, not just its target.

    Splash punishing a clumped push (and single-target not doing the same) is
    the entire reason a player spreads troops out against a Bomber.
    """
    battle = _empty_battle(world)
    bomber = _spawn(battle, world, "Bomber", Team.BLUE, 9, 10.0)
    assert bomber.spec.projectile == "BombSkeletonProjectile"
    skeletons = [
        _spawn(battle, world, "Skeleton", Team.RED, 9, 13.5),
        _spawn(battle, world, "Skeleton", Team.RED, 9.3, 13.5),
        _spawn(battle, world, "Skeleton", Team.RED, 9, 13.8),
    ]

    for _ in range(250):
        battle.step()
        if all(s.dead for s in skeletons):
            break

    assert all(s.dead for s in skeletons), "one splash shell should down the whole clump"
    bomber_hits = [e for e in battle.damage_log if e.attacker_id == bomber.id]
    assert len(bomber_hits) == 3, "expected exactly one hit per skeleton, from one shell"
    assert len({e.tick for e in bomber_hits}) == 1, "all three must be hit on the same tick"


def test_single_target_projectile_only_damages_its_target(world):
    """A Musketeer's arrow hits the one unit it was aimed at, nothing beside it.

    Placed next to the splash test above, this is what makes spreading out
    only a counter to splash -- it does nothing against a single-target
    attacker, who just picks the nearest body regardless.
    """
    battle = _empty_battle(world)
    musketeer = _spawn(battle, world, "Musketeer", Team.BLUE, 9, 10.0)
    assert musketeer.spec.projectile == "MusketeerProjectile"
    near = _spawn(battle, world, "Skeleton", Team.RED, 9, 12.5)
    far_left = _spawn(battle, world, "Skeleton", Team.RED, 15, 12.5)
    far_right = _spawn(battle, world, "Skeleton", Team.RED, 3, 12.5)

    for _ in range(150):
        battle.step()
        if near.dead:
            break

    assert near.dead, "the targeted skeleton should have been killed"
    assert far_left.hitpoints == far_left.max_hitpoints, "splash leaked to a bystander"
    assert far_right.hitpoints == far_right.max_hitpoints, "splash leaked to a bystander"


# ------------------------------------------------------------------- homing


def test_homing_shot_still_hits_a_target_that_has_moved(world):
    """A homing shot re-aims every tick, so a target cannot simply step out of it.

    Fire Spirits' bomb is homing *and* splash, so this also proves the blast
    goes off at the target's current position, not the point it was fired at
    -- a non-homing shot committed to the original aim point could never have
    reached a target that relocated across the arena.
    """
    battle = _empty_battle(world)
    caster = _spawn(battle, world, "FireSpirits", Team.BLUE, 9, 10.0)
    victim = _spawn(battle, world, "Knight", Team.RED, 9, 11.5)
    pspec = battle._projectile_spec("FireSpiritsProjectile", "Common", caster.spec.level)
    assert pspec.homing

    shot = None
    for _ in range(30):
        battle.step()
        shots = _live_projectiles(battle)
        if shots:
            shot = shots[0]
            break
    assert shot is not None
    original_aim = (shot.target_x, shot.target_y)

    # Relocate the target well clear of the original aim point -- far enough
    # that a shot committed to that point could never reach it.
    victim.x, victim.y = tiles(15), tiles(25)

    for _ in range(400):
        battle.step()
        if victim.hitpoints < victim.max_hitpoints:
            break

    assert victim.hitpoints < victim.max_hitpoints, "a homing shot must follow its target"
    assert (shot.target_x, shot.target_y) != original_aim, (
        "the shot must have re-aimed at the target's new position"
    )


# -------------------------------------------------------- death mid-flight


def test_splash_shot_still_detonates_where_aimed_if_the_target_dies_first(world):
    """A Bomber's shell keeps going even if the skeleton it was aimed at dies first.

    The attacker has already committed the shot; it is not recalled. A
    bystander standing where the target was still eats the blast.
    """
    battle = _empty_battle(world)
    bomber = _spawn(battle, world, "Bomber", Team.BLUE, 9, 10.0)
    target = _spawn(battle, world, "Skeleton", Team.RED, 9, 13.0)
    bystander = _spawn(battle, world, "Skeleton", Team.RED, 9, 13.0)

    for _ in range(150):
        battle.step()
        if _live_projectiles(battle):
            break
    assert _live_projectiles(battle), "the shell never launched"

    target.apply_damage(target.hitpoints)  # dies mid-flight, before the shell lands

    for _ in range(150):
        battle.step()  # must not raise

    assert bystander.dead, "the shell should still have gone off where it was aimed"


def test_single_target_shot_does_not_crash_if_the_target_dies_in_flight(world):
    """An arrow aimed at a target that dies mid-flight simply fizzles.

    There is nothing left to redirect a single-target shot to, so it must
    resolve to nothing rather than raise or silently damage whoever replaced
    the target on that tile.
    """
    battle = _empty_battle(world)
    musketeer = _spawn(battle, world, "Musketeer", Team.BLUE, 9, 10.0)
    victim = _spawn(battle, world, "Musketeer", Team.RED, 9, 14.0)

    for _ in range(60):
        battle.step()
        if any(e.owner_id == musketeer.id for e in _live_projectiles(battle)):
            break
    hits_before = len([e for e in battle.damage_log if e.target_id == victim.id])

    victim.kill()  # dies mid-flight, before the arrow lands

    for _ in range(60):
        battle.step()  # must not raise

    hits_after = len([e for e in battle.damage_log if e.target_id == victim.id])
    assert hits_after == hits_before, "a shot aimed at a corpse must not deal damage"


# --------------------------------------------------- targetability & collision


def test_projectiles_are_not_targetable(world):
    """Nothing can select a shot in flight as a target -- there is no shooting
    down an arrow in Clash Royale."""
    battle = _empty_battle(world)
    _spawn(battle, world, "Musketeer", Team.BLUE, 9, 10.0)
    victim = _spawn(battle, world, "Knight", Team.RED, 9, 12.0)

    shot = None
    for _ in range(60):
        battle.step()
        shots = _live_projectiles(battle)
        if shots:
            shot = shots[0]
            break
    assert shot is not None
    assert not can_target(victim.spec, victim, shot)


def test_projectiles_do_not_participate_in_collision(world):
    """Shots pass over everything -- a unit never detours around an arrow.

    Without this, a projectile parked in a unit's path would shove it aside
    like any other body, which would make ranged fire an obstacle course.
    """
    battle = _empty_battle(world)
    knight = _spawn(battle, world, "Knight", Team.BLUE, 9, 10.0)
    dummy_target = _spawn(battle, world, "Knight", Team.RED, 9, 20.0)
    pspec = battle._projectile_spec("MusketeerProjectile", "Rare", 1)
    # Park the shot exactly on top of the Knight -- the deepest overlap two
    # entities can have.
    shot = Projectile(
        pspec=pspec, team=Team.RED, x=knight.x, y=knight.y,
        target=dummy_target, owner_id=dummy_target.id, spawn_tick=battle.tick,
    )
    battle._register(shot)

    before = (knight.x, knight.y)
    battle._phase_resolve_collisions()
    assert (knight.x, knight.y) == before, "a projectile must never push a unit aside"


# --------------------------------------------------------------- flight_ticks


def test_flight_ticks_matches_the_documented_reference_shots(world):
    """A slow Mortar lob and a near-continuous X-Bow bolt corroborate the
    Speed-is-tiles-per-minute reading, both over the 11.5-tile range they
    actually fire at."""
    data, levels, _registry = world
    clock = TickClock()
    scale = levels.get("Common")
    level = scale.internal_level(11)

    mortar = build_projectile_spec(data, "MortarProjectile", scale, level=level, clock=clock)
    xbow = build_projectile_spec(data, "xbow_projectile", scale, level=level, clock=clock)
    assert mortar.speed == 300
    assert xbow.speed == 1600

    mortar_ticks = flight_ticks(tiles(11.5), mortar.speed, clock)
    xbow_ticks = flight_ticks(tiles(11.5), xbow.speed, clock)
    mortar_ms = mortar_ticks * 1000 / clock.ticks_per_second
    xbow_ms = xbow_ticks * 1000 / clock.ticks_per_second

    tick_ms = 1000 / clock.ticks_per_second
    assert abs(mortar_ms - 2300) <= 2 * tick_ms, f"Mortar flight {mortar_ms}ms, expected ~2300ms"
    assert abs(xbow_ms - 417) <= 2 * tick_ms, f"X-Bow flight {xbow_ms}ms, expected ~417ms"


# ------------------------------------------------------------------- determinism


def test_projectile_combat_is_deterministic(world):
    """Two identical ranged duels must produce byte-identical state every tick.

    Projectiles add a second population of entities with their own re-aiming
    logic; nothing about tracking or resolving them may depend on iteration
    order or timing outside the tick loop.
    """
    from cr_sim.replay import compare_hashes

    runs = []
    for _ in range(2):
        battle = _empty_battle(world)
        _spawn(battle, world, "Musketeer", Team.BLUE, 9, 10.0)
        _spawn(battle, world, "Musketeer", Team.RED, 9, 15.5)
        runs.append([(battle.step(), battle.hash())[1] for _ in range(400)])
    assert compare_hashes(runs[0], runs[1]) is None


# --------------------------------------------------------------------- cleanup


def test_projectiles_do_not_accumulate(world):
    """A long, sustained ranged fight must never build up an unbounded number
    of shots in flight.

    At most one shot per shooter can be airborne at once (each unit reloads
    before firing again), so the live count is capped by the number of
    shooters, however many volleys have been fired over the whole fight.
    """
    battle = _empty_battle(world)
    shooters = [
        _spawn(battle, world, "Musketeer", Team.BLUE, 3 + 3 * i, 10.0)
        for i in range(3)
    ]
    for i in range(3):
        _spawn(battle, world, "Knight", Team.RED, 3 + 3 * i, 15.5)

    max_concurrent = 0
    for _ in range(500):
        battle.step()
        max_concurrent = max(max_concurrent, len(_live_projectiles(battle)))

    assert max_concurrent <= len(shooters), (
        f"never more than one shot per shooter should be in flight, got {max_concurrent}"
    )
    # Many more shots were fired over the whole fight than were ever airborne
    # at once -- proof that spent shots are actually swept, not merely capped.
    assert len(battle.graveyard) > len(shooters)
