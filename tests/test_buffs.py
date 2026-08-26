"""Buff system: timed status effects, read directly against real game data.

These tests are about mechanics, not code shape: a Poison cloud really does
tick 8 times over its 8 second life with the first hit landing immediately, a
non-stacking buff really does refresh instead of piling up, and the raw
numbers this module reads for Freeze/Poison/Ice Wizard really do match the
percentages every Clash Royale player knows. See the module docstring in
``cr_sim/engine/buffs.py`` for the evidence behind each decision this file
pins down.
"""

from __future__ import annotations

from cr_sim.data.leveling import build_level_table
from cr_sim.data.source import LogicData
from cr_sim.engine.buffs import (
    ActiveBuff,
    BuffState,
    apply_multiplier,
    build_buff_spec,
)
from cr_sim.engine.constants import TickClock

from .test_data_pipeline import BUILD


def _world():
    data = LogicData.load(BUILD)
    levels = build_level_table(data)
    return data, levels


DATA, LEVELS = _world()
CLOCK = TickClock()
# Common at internal level 1 has power index 0, whose multiplier is the
# implicit "100" (no scaling) -- see RarityScale.multiplier -- so specs built
# with this scale/level pair carry the data's raw numbers unchanged, which
# makes them directly comparable to the values read straight from BUFF.*.
COMMON = LEVELS.get("Common")
UNSCALED_LEVEL = 1


def spec(name: str):
    built = build_buff_spec(DATA, name, COMMON, level=UNSCALED_LEVEL, clock=CLOCK)
    assert built is not None, f"BUFF.{name} did not resolve"
    return built


# --------------------------------------------------------- reading real data


def test_freeze_stops_movement_and_everything_else():
    """Freeze is -100 across the board: speed, hit speed, and spawn speed all halt."""
    f = spec("Freeze")
    assert f.speed_multiplier == -100
    assert f.hit_speed_multiplier == -100
    assert f.spawn_speed_multiplier == -100
    assert f.stacks is False


def test_poison_matches_its_well_known_15_percent_slow():
    """Poison's SpeedMultiplier (-15) matches the "15% slower" every source cites."""
    p = spec("Poison")
    assert p.speed_multiplier == -15
    assert apply_multiplier(1000, p.speed_multiplier) == 850  # 15% slower, exactly
    assert p.stacks is True, "Poison clouds are the textbook example of a stacking buff"


def test_ice_wizard_matches_its_well_known_30_percent_slow():
    """Both Ice Wizard buffs (on-hit and lingering cold) read -30, matching the card's 30% slow."""
    for name in ("IceWizardCold", "IceWizardSlowDown"):
        w = spec(name)
        assert w.speed_multiplier == -30
        assert w.hit_speed_multiplier == -30
        assert apply_multiplier(100, w.speed_multiplier) == 70


def test_rage_is_read_verbatim_from_the_data_despite_the_anomaly():
    """Pins the surprising raw value down -- see the module docstring's evidence.

    Rage's real, well-documented effect is a 30-40% speed and attack speed
    boost, but ``BUFF.Rage`` (confirmed as what the Rage spell's AEO actually
    applies) reads +130, which the same delta formula that nails Poison and
    Ice Wizard exactly would turn into +130% -- not the real card. This
    module stores the value verbatim rather than silently "fixing" it; this
    test exists so nobody "corrects" the reader without re-reading why.
    """
    r = spec("Rage")
    assert r.speed_multiplier == 130
    assert r.hit_speed_multiplier == 130
    assert r.spawn_speed_multiplier == 130
    assert r.stacks is False


def test_unknown_buff_name_returns_none():
    assert build_buff_spec(DATA, "NoSuchBuff", COMMON, level=UNSCALED_LEVEL, clock=CLOCK) is None


# -------------------------------------------------------- apply_multiplier


def test_apply_multiplier_matches_the_house_delta_convention():
    """base * (100 + percent) // 100, the same formula used by every damage_to() in this codebase."""
    assert apply_multiplier(1000, 0) == 1000
    assert apply_multiplier(1000, 30) == 1300
    assert apply_multiplier(1000, -15) == 850
    assert apply_multiplier(1000, -100) == 0


def test_apply_multiplier_never_goes_negative():
    """A slow stronger than -100% must floor at zero, not reverse."""
    assert apply_multiplier(1000, -250) == 0
    assert apply_multiplier(0, -50) == 0


# ---------------------------------------------------- damage-over-time timing


def test_damage_over_time_starts_after_one_interval_not_on_contact():
    """Poison's first tick lands a second after it is applied, not immediately.

    This is the arithmetic that settles it. A Poison cloud lives 8 seconds and
    re-touches whatever is inside it four times a second, refreshing the status
    rather than stacking it. With the damage rhythm starting one interval after
    application, a unit that stands in the whole cloud takes exactly 8 ticks --
    the documented behaviour, verified end to end in test_spells.py at 736
    damage. Starting on contact instead yields 9, an extra ninth of the card.

    The area effect's *own* first application is a different clock and IS
    immediate; Zap has to be instant.
    """
    p = spec("Poison")
    state = BuffState()
    state.apply(p, duration_ticks=p.hit_frequency_ticks * 8)

    assert state.tick() == 0, "damage landed on the contact tick"
    for _ in range(p.hit_frequency_ticks - 2):
        assert state.tick() == 0
    assert state.tick() == p.damage_per_second, "no damage after one full interval"


def test_damage_over_time_repeats_on_its_own_interval():
    """One tick per HitFrequency for as long as the status lasts."""
    p = spec("Poison")
    state = BuffState()
    # Kept alive well past its own duration, the way a cloud refreshes it.
    state.apply(p, duration_ticks=p.hit_frequency_ticks * 8 + 5)
    hits = [i for i in range(p.hit_frequency_ticks * 8) if state.tick() > 0]
    assert len(hits) == 8, hits
    # Evenly spaced, one interval apart.
    gaps = {b - a for a, b in zip(hits, hits[1:])}
    assert gaps == {p.hit_frequency_ticks}, gaps


def test_poison_stacks_two_independent_copies():
    """Two Poison clouds stack; one cloud re-touching you does not.

    Stacking is per *source*, because every area effect in the build carries
    ``BuffNumber = 1``. A single cloud re-applies to everything inside it four
    times a second, and treating each touch as a new stack turns an 8-second
    spell into an instant kill -- so a repeat from the same source refreshes.
    Two genuinely separate clouds are two sources and do stack, which is what
    ``EnableStacking`` is for.
    """
    p = spec("Poison")
    state = BuffState()
    duration = p.hit_frequency_ticks * 8

    state.apply(p, duration_ticks=duration, source=1)
    state.apply(p, duration_ticks=duration, source=1)  # same cloud, re-touch
    assert state.active_names() == ("Poison",), "one cloud stacked with itself"

    state.apply(p, duration_ticks=duration, source=2)  # a second cloud
    assert state.active_names() == ("Poison", "Poison")

    # Damage lands one interval after application, not on contact.
    total = sum(state.tick() for _ in range(p.hit_frequency_ticks))
    assert total == p.damage_per_second * 2, "two stacked clouds must both tick"


def test_rage_refreshes_instead_of_stacking():
    """A second Rage application replaces the first rather than doubling the speed boost."""
    r = spec("Rage")
    state = BuffState()
    state.apply(r, duration_ticks=100)
    state.apply(r, duration_ticks=100)
    assert state.active_names() == ("Rage",), "non-stacking buff produced two copies"
    assert state.speed_multiplier() == r.speed_multiplier, "speed boost must not have doubled"


def test_reapplying_a_refreshing_buff_resets_its_remaining_duration():
    freeze = spec("Freeze")
    state = BuffState()
    state.apply(freeze, duration_ticks=10)
    for _ in range(8):
        state.tick()
    state.apply(freeze, duration_ticks=10)  # re-frozen before the first application expired
    for _ in range(9):
        state.tick()
    assert state, "refreshed duration expired early"
    state.tick()
    assert not state, "refreshed duration should have expired on the 10th tick after reapplying"


# -------------------------------------------------------- combined multipliers


def test_combined_multiplier_is_the_sum_of_active_deltas():
    """Two simultaneous slows add their percentage deltas rather than compounding."""
    poison = spec("Poison")  # -15
    ice = spec("IceWizardCold")  # -30
    state = BuffState()
    state.apply(poison, duration_ticks=480)
    state.apply(ice, duration_ticks=30)
    assert state.speed_multiplier() == poison.speed_multiplier + ice.speed_multiplier == -45


def test_combining_multipliers_is_order_independent():
    """Applying the same two buffs in the opposite order must give an identical combined value.

    Summing raw deltas before applying apply_multiplier once is commutative;
    chaining two multiply-then-floor-divide steps would not be, and this
    engine's determinism guarantee requires order not to matter anywhere.
    """
    poison, ice = spec("Poison"), spec("IceWizardCold")

    forward = BuffState()
    forward.apply(poison, duration_ticks=480)
    forward.apply(ice, duration_ticks=30)

    backward = BuffState()
    backward.apply(ice, duration_ticks=30)
    backward.apply(poison, duration_ticks=480)

    assert forward.speed_multiplier() == backward.speed_multiplier()
    assert apply_multiplier(1000, forward.speed_multiplier()) == apply_multiplier(
        1000, backward.speed_multiplier()
    )


def test_stacked_slows_reaching_100_percent_count_as_frozen():
    """Two -50 slows combine to -100, which must read as fully stopped, same as a real Freeze."""
    earthquake = spec("Earthquake")  # -50
    assert earthquake.speed_multiplier == -50
    state = BuffState()
    state.apply(earthquake, duration_ticks=60, source=1)
    assert not state.is_frozen(), "one Earthquake alone should not read as frozen"
    # A second, separate Earthquake. Same source would refresh rather than
    # stack, which is what stops a single cloud slowing a unit to a halt.
    state.apply(earthquake, duration_ticks=60, source=2)
    assert state.is_frozen(), "two stacked -50 slows must combine to a full stop"


def test_freeze_reads_as_frozen():
    state = BuffState()
    state.apply(spec("Freeze"), duration_ticks=240)
    assert state.is_frozen()
    assert apply_multiplier(1000, state.speed_multiplier()) == 0


# --------------------------------------------------------- lifecycle & bookkeeping


def test_buff_state_is_falsy_when_empty_and_truthy_when_active():
    state = BuffState()
    assert not state
    state.apply(spec("Rage"), duration_ticks=60)
    assert state


def test_clear_removes_every_active_buff():
    state = BuffState()
    state.apply(spec("Poison"), duration_ticks=480)
    state.apply(spec("Rage"), duration_ticks=60)
    state.clear()
    assert not state
    assert state.active_names() == ()
    assert state.speed_multiplier() == 0
    assert state.tick() == 0


def test_expired_buffs_are_swept_on_the_tick_their_duration_ends():
    state = BuffState()
    state.apply(spec("Rage"), duration_ticks=3)
    assert state.speed_multiplier() == 130
    state.tick()
    state.tick()
    assert state, "buff expired a tick early"
    state.tick()
    assert not state, "buff outlived its duration"
    assert state.speed_multiplier() == 0


def test_active_buff_is_a_plain_mutable_record():
    """ActiveBuff itself, independent of BuffState, behaves as documented."""
    r = spec("Rage")
    active = ActiveBuff(spec=r, ticks_left=5, ticks_to_next_damage=0)
    assert active.spec is r
    active.ticks_left -= 1
    assert active.ticks_left == 4
