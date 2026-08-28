"""The card-stat vector, pinned to numbers that did not come out of it.

:mod:`cr_sim.data.card_features` turns a card into the row a stat-conditioned
policy head reads instead of a per-card embedding column. Everything about it
is a claim about *content* -- "this column is the hitpoints", "this column says
the card hits a group" -- and a test that builds its expectation by calling the
same function compares the function against itself: it can only see a row
landing in the wrong order, never a column that is wrong or missing. That is
not a hypothetical -- an earlier eight-test draft whose one content check was
written that way stayed entirely green while ``hp``, ``air``, ``mana`` and
``speed`` were each hard-zeroed in turn, which is a vector that cannot tell a
Skeleton from a P.E.K.K.A on health.

So the numbers below are written out. 1766 is a Knight's hitpoints at display
level 11 in this build, and it is here as 1766 rather than as
``spec.hitpoints``.
"""

from __future__ import annotations

import pytest

from cr_sim.api import encoding
from cr_sim.data import card_features as cf
from cr_sim.data.card_features import (
    CARD_FEATURE_COUNT, CARD_FEATURE_NAMES, CARD_FEATURE_LEVEL,
    card_feature_table, card_feature_vector,
)
from cr_sim.data.cards import build_card_registry
from cr_sim.data.leveling import build_level_table
from cr_sim.data.source import LogicData

from .test_data_pipeline import BUILD


@pytest.fixture(scope="module")
def world():
    data = LogicData.load(BUILD)
    return data, build_level_table(data), build_card_registry(data)


@pytest.fixture(scope="module")
def feature(world):
    data, levels, registry = world

    def read(card_name: str, name: str) -> float:
        vector = card_feature_vector(data, levels, registry, registry[card_name])
        return vector[CARD_FEATURE_NAMES.index(name)]

    return read


# ------------------------------------------------------- anchored to the game


#: ``(card, hitpoints, damage per second, attacks air)`` at display level 11,
#: transcribed from the build rather than recomputed from the module under
#: test. The whole point is that these do not move when it does.
ANCHORS = (
    ("Knight", 1766, 168.33333333333334, False),
    ("Musketeer", 721, 217.0, True),
    ("Pekka", 3760, 467.77777777777777, False),
)


@pytest.mark.parametrize("card_name,hitpoints,dps,attacks_air", ANCHORS)
def test_the_headline_columns_carry_the_card_s_real_numbers(
    feature, card_name, hitpoints, dps, attacks_air
):
    """Hard-zeroing ``hp`` is the mutation this exists for: without it a
    Skeleton and a P.E.K.K.A are equally healthy and nothing goes red."""
    assert feature(card_name, "hp") * cf._HP_NORM == pytest.approx(hitpoints)
    assert feature(card_name, "dps") * cf._DPS_NORM == pytest.approx(dps)
    assert feature(card_name, "air") == float(attacks_air)


def test_the_geometry_columns_are_in_tiles(feature):
    """A Knight reaches 1.2 tiles and a Musketeer 6.0. Read in subtiles the two
    are 21600 and 108000, which clip to 1.0 and stop being different."""
    assert feature("Knight", "attack_range") * cf._REACH_NORM == pytest.approx(1.2)
    assert feature("Musketeer", "attack_range") * cf._REACH_NORM == pytest.approx(6.0)
    assert feature("Pekka", "collision") * cf._COLLISION_NORM == pytest.approx(0.75)
    assert feature("Knight", "speed") * cf._SPEED_NORM == pytest.approx(60.0)
    assert feature("Mortar", "speed") == 0.0, "a building does not move"


def test_elixir_is_the_cost_the_card_is_played_at(feature):
    from cr_sim.engine.constants import MAX_ELIXIR

    assert feature("Knight", "mana") * MAX_ELIXIR == pytest.approx(3)
    assert feature("Pekka", "mana") * MAX_ELIXIR == pytest.approx(7)


def test_the_shared_normalisations_are_the_encoder_s_own():
    """The module says these are duplicated from :mod:`cr_sim.api.encoding`
    "and pinned against it by a test". This is that test: a card's hitpoint
    feature has to read on the same scale as the hitpoint mass in the grid the
    same network sees, and a silent divergence would put the two inputs on
    different scales with nothing to show for it."""
    assert cf._HP_NORM == encoding.HP_NORM
    assert cf._DPS_NORM == encoding.DPS_NORM
    assert cf._REACH_NORM == encoding.REACH_NORM
    assert cf._COUNT_NORM == encoding.COUNT_NORM


# ------------------------------------------------- what lives on the projectile


def test_a_ranged_splash_unit_is_not_encoded_as_single_target(feature):
    """``UnitSpec.area_damage_radius`` is the *melee* swing's radius -- the
    engine applies it only when nothing was launched. A Wizard's splash is his
    projectile's 1.5 tiles, and reading the character row alone made him
    identical to a Musketeer on the one column that decides a tile."""
    assert feature("Wizard", "splash") * cf._SPLASH_NORM == pytest.approx(1.5)
    assert feature("BombTower", "splash") * cf._SPLASH_NORM == pytest.approx(1.5)
    assert feature("FireSpirits", "splash") * cf._SPLASH_NORM == pytest.approx(2.3)
    assert feature("Musketeer", "splash") == 0.0, "a single-target shot"
    assert feature("Knight", "splash") == 0.0, "a single-target swing"


def test_the_melee_radius_still_wins_where_it_is_the_larger(feature):
    """Eight of the 25 cards with a splash radius keep it on the character,
    and taking the projectile's unconditionally would lose them. Princess
    carries 2.5 tiles on the character against her shot's 2.0; Valkyrie's
    360-degree swing has no projectile to read at all."""
    assert feature("Princess", "splash") * cf._SPLASH_NORM == pytest.approx(2.5)
    assert feature("Valkyrie", "splash") > 0.0
    assert feature("MegaKnight", "splash") > 0.0


def test_ice_spirit_and_heal_spirit_are_not_the_same_card(world):
    """One freezes a push and the other heals one, and both put their whole
    card on the projectile: ``TargetBuff: Freeze`` against
    ``SpawnAreaEffectObject`` resolving to ``HealPerSecond: 157``. Read off the
    character row alone the two differ on ``is_troop`` and ``is_spell`` and on
    nothing else -- in an all-troop deck the rows would be bit-identical."""
    data, levels, registry = world
    ice = card_feature_vector(data, levels, registry, registry["IceSpirits"])
    heal = card_feature_vector(data, levels, registry, registry["Heal"])

    behavioural = [
        index for index, name in enumerate(CARD_FEATURE_NAMES)
        if name not in ("is_troop", "is_building", "is_spell")
    ]
    differ = [CARD_FEATURE_NAMES[i] for i in behavioural if ice[i] != heal[i]]
    assert differ, "the two cards are the same vector on every behavioural column"

    at = CARD_FEATURE_NAMES.index
    assert ice[at("on_hit_slow")] < 0.0 and ice[at("heals")] == 0.0
    assert heal[at("heals")] > 0.0 and heal[at("on_hit_slow")] == 0.0
    assert heal[at("heals")] * cf._DPS_NORM == pytest.approx(401.0)


def test_a_stun_on_the_unit_reads_the_same_as_one_on_the_shot(feature):
    """Electro Wizard carries ``ZapFreeze`` on ``buff_on_damage`` rather than
    on his projectile. Same effect, different field, and reading only the
    projectile leaves the card that resets an Inferno Tower looking inert."""
    for card_name in ("ElectroWizard", "ElectroSpirit", "IceSpirits"):
        assert feature(card_name, "on_hit_slow") == pytest.approx(
            -100 / cf._BUFF_MULTIPLIER_NORM), card_name
        assert feature(card_name, "on_hit_hitspeed") == pytest.approx(
            -100 / cf._BUFF_MULTIPLIER_NORM), card_name
    assert feature("IceWizard", "on_hit_slow") == pytest.approx(
        -30 / cf._BUFF_MULTIPLIER_NORM)
    assert feature("Knight", "on_hit_slow") == 0.0


def test_a_curse_that_carries_no_numbers_still_reads_as_an_on_hit_effect(feature):
    """``BUFF.VoodooCurse`` has no multiplier and no damage at all -- only
    ``DeathSpawn`` -- so on the numeric columns alone Mother Witch's defining
    property is indistinguishable from having no on-hit effect."""
    assert feature("WitchMother", "on_hit_buff") == 1.0
    assert feature("WitchMother", "on_hit_slow") == 0.0
    assert feature("WitchMother", "on_hit_dps") == 0.0
    assert feature("Knight", "on_hit_buff") == 0.0


def test_an_on_hit_burn_reaches_the_damage_over_time_column(feature):
    """The evolved Firecracker's sparks are ``DamagePerSecond`` on a buff, not
    ``Damage`` on the shot, and are the only card in the build that fills this
    column."""
    assert feature("Firecracker_EV1", "on_hit_dps") > 0.0
    assert feature("Firecracker", "on_hit_dps") == 0.0


# ------------------------------------------------------------- variant cards


def test_a_variant_card_is_described_by_the_form_it_deploys(world):
    """Merge Maiden summons nothing itself: it names the forms it can turn
    into and the engine deploys whichever the elixir pays for. Read literally
    it is ``mana`` plus ``is_spell`` and forty-five zeros -- a six-elixir card
    the head is told puts nothing on the board, two bits away from an empty
    hand slot."""
    data, levels, registry = world
    maiden = card_feature_vector(data, levels, registry, registry["MergeMaiden"])
    mounted = card_feature_vector(
        data, levels, registry, registry["MergeMaiden_Mounted"])
    at = CARD_FEATURE_NAMES.index

    assert maiden[at("has_unit")] == 1.0
    assert maiden[at("hp")] * cf._HP_NORM == pytest.approx(1121)
    assert maiden[at("dps")] * cf._DPS_NORM == pytest.approx(220.71428571428572)
    assert maiden[at("flying")] == 1.0
    assert maiden[at("bodies")] > 0.0

    # Everything about the body is the mounted form's, exactly.
    for name in CARD_FEATURE_NAMES[at("bodies"):]:
        assert maiden[at(name)] == mounted[at(name)], name
    # Block A stays the card in hand's: six elixir is what the player pays,
    # and the game data calls the card a spell whatever it deploys.
    assert maiden[at("mana")] == pytest.approx(0.6)
    assert (maiden[at("is_spell")], maiden[at("is_troop")]) == (1.0, 0.0)
    assert (mounted[at("is_spell")], mounted[at("is_troop")]) == (0.0, 1.0)


# ------------------------------------------------------------- spell payloads


def test_a_damage_over_time_spell_reports_its_crown_tower_reduction(feature):
    """Poison, Tornado and Earthquake carry every point of their damage on a
    buff, and the reduction with it. ``0.0`` -- what they read before the
    payload chain reached the buff -- is the same number a spell with no
    reduction at all gets, so the head was told Poison hits a tower for full
    damage-per-second when it hits for 23% of it."""
    assert feature("Poison", "p_crown") == pytest.approx(-0.77)
    assert feature("Tornado", "p_crown") == pytest.approx(-0.70)
    assert feature("Earthquake", "p_crown") == pytest.approx(-0.40)
    # Unchanged, and the reason the fallback is a fallback: these three carry
    # the percentage on the object that deals the damage.
    assert feature("Fireball", "p_crown") == pytest.approx(-0.75)
    assert feature("Log", "p_crown") == pytest.approx(-0.87)
    assert feature("Arrows", "p_crown") == pytest.approx(-0.80)


def test_the_fuse_route_reads_a_crown_reduction_off_the_buff_too(world):
    """The fuse route resolves ``AEO.<name>`` and then ``BUFF.<name>`` itself
    rather than going through ``card_stat_summary``, so it needs the same last
    link in the crown-tower chain. No fuse card in the build carries one
    today, which is exactly why it is asserted on the helper: an area effect
    whose damage is all on its buff -- Poison's is -- must not report "full
    damage to towers" through this route while the spell route says -77."""
    data, levels, registry = world
    scale = levels.get(registry["Poison"].rarity)
    payload = cf._area_payload(
        data, scale, scale.internal_level(CARD_FEATURE_LEVEL), "Poison")
    assert payload["crown_tower_damage_percent"] == -77
    assert payload["damage_per_second"] > 0


def test_the_fuse_route_still_reads_rage_and_royal_delivery(feature):
    """Both summon only a fuse, and reading that as the unit describes the
    bottle rather than the spell."""
    assert feature("Rage", "has_unit") == 0.0
    assert feature("Rage", "p_speed_mult") == pytest.approx(130 / cf._BUFF_MULTIPLIER_NORM)
    assert feature("RoyalDelivery", "p_damage") * cf._PAYLOAD_DAMAGE_NORM == pytest.approx(384)
    assert feature("RoyalDelivery", "p_spawns") > 0.0


# --------------------------------------------------------- the whole card pool


def test_no_standard_card_saturates_a_normalisation(world):
    """The module's rule is that 1.0 is the build's own maximum. A norm a real
    card exceeds silently puts the most extreme card in the game on the same
    value as a merely large one: Zap Machine's 1331-damage swing, Goblin
    Cage's 20-tile aggro radius -- the whole reason it pulls a push -- and
    Skeleton Army's 1104 summed damage per second each sat on a ceiling."""
    data, levels, registry = world
    # With the clips removed a saturating column reports the quotient it
    # actually computed. Asserting on the clipped output cannot see this at
    # all: 1.331 and 1.000 are both 1.0 by the time the caller has them.
    identity = lambda value: value  # noqa: E731
    monkeypatched = (cf._clip, cf._signed_clip)
    cf._clip = cf._signed_clip = identity
    try:
        over = [
            (card.name, CARD_FEATURE_NAMES[index], value)
            for card in registry.standard()
            for index, value in enumerate(
                card_feature_vector(data, levels, registry, card))
            if value > 1.0 or value < -1.0
        ]
    finally:
        cf._clip, cf._signed_clip = monkeypatched
    assert over == []

    # And the three that used to be past it are still the largest, so the
    # norms are the build's maxima rather than arbitrarily large numbers.
    at = CARD_FEATURE_NAMES.index
    for column, card_name, quotient in (
        ("damage", "ZapMachine", 1331 / cf._BURST_NORM),
        ("sight_range", "GoblinCage", 20.0 / cf._SIGHT_NORM),
        ("total_dps", "SkeletonArmy", 1104.5454545454545 / cf._TOTAL_DPS_NORM),
    ):
        value = card_feature_vector(
            data, levels, registry, registry[card_name])[at(column)]
        assert value == pytest.approx(quotient), column
        assert 0.75 < value <= 1.0, (
            f"{card_name}'s {column} is {value}: a norm this far above the "
            "build's maximum wastes the top of the column")


def test_every_standard_card_gets_a_distinct_vector(world):
    """The module's own validation claim. A collision is two cards the head
    cannot tell apart at all, and it is what ``is_mirror`` exists to prevent:
    Mirror has no payload, no unit and no variant, and would otherwise be an
    all-zero row indistinguishable from an empty hand slot."""
    data, levels, registry = world
    seen: dict[tuple[float, ...], str] = {}
    for card in registry.standard():
        vector = card_feature_vector(data, levels, registry, card)
        assert len(vector) == CARD_FEATURE_COUNT
        assert vector not in seen, f"{card.name} collides with {seen.get(vector)}"
        seen[vector] = card.name
    assert len(seen) == 122


def test_an_empty_hand_slot_is_the_one_all_zero_row(world):
    """The observation encodes an empty slot as zeros, so no real card may be
    zeros -- and Mirror is the card that nearly is."""
    data, levels, registry = world
    zeros = (0.0,) * CARD_FEATURE_COUNT
    for card in registry.standard():
        assert card_feature_vector(data, levels, registry, card) != zeros, card.name


# ------------------------------------------------------------------ the table


def test_the_table_is_one_row_per_name_in_the_order_given(world):
    """Row ``i`` is what slot ``i``'s one-hot bit selects, and that bit is set
    from ``vocab.index(...)``. Built in any other order the head trains to a
    lower loss while conditioned on the wrong cards."""
    data, levels, registry = world
    names = ("Musketeer", "Knight", "Fireball")
    table = card_feature_table(data, levels, registry, names)
    assert len(table) == len(names)
    for row, name in zip(table, names):
        assert row == card_feature_vector(data, levels, registry, registry[name])
    # Order, not just membership: the reversed table is a different table.
    assert table != card_feature_table(data, levels, registry, names[::-1])


def test_a_card_the_registry_does_not_have_raises(world):
    """Rather than a zero row, which is a card the head believes is nothing."""
    data, levels, registry = world
    with pytest.raises(KeyError):
        card_feature_table(data, levels, registry, ("Knight", "NotACard"))


def test_the_table_is_plain_floats_so_it_survives_the_worker_pickle(world):
    """``NetConfig`` is frozen and hashable and is pickled into every spawned
    worker's config. A numpy array here breaks the auto ``__hash__``."""
    import pickle

    data, levels, registry = world
    table = card_feature_table(data, levels, registry, ("Knight", "Musketeer"))
    assert isinstance(table, tuple)
    for row in table:
        assert isinstance(row, tuple)
        assert all(type(value) is float for value in row)
    assert hash(table)
    assert pickle.loads(pickle.dumps(table)) == table


def test_the_level_the_table_is_built_at_is_pinned(world):
    """A table that followed ``env.level`` would describe the same card with
    different numbers in the trainer and in the browser server, which never
    sets one."""
    assert CARD_FEATURE_LEVEL == 11
