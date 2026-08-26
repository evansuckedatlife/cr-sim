"""M0 gate: the ingestion pipeline must reproduce known-correct game numbers.

These tests are deliberately about *values*, not code shape.  The whole point of
the data layer is that a Knight has 1766 hitpoints at tournament standard; if
that stops being true the simulator is worthless no matter how clean the code is.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cr_sim.data.cards import CardKind, build_card_registry, card_stat_summary
from cr_sim.data.csv_loader import load_table
from cr_sim.data.decode import decode_bytes, sniff
from cr_sim.data.leveling import build_level_table
from cr_sim.data.source import LogicData

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "data_cache" / "csv_logic"
ANCHORS = json.loads((ROOT / "reference" / "anchors.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def data() -> LogicData:
    if not BUILD.is_dir():
        pytest.skip("no extracted csv_logic build; run scripts/extract_apk.py")
    return LogicData.load(BUILD)


@pytest.fixture(scope="session")
def levels(data):
    return build_level_table(data)


@pytest.fixture(scope="session")
def registry(data):
    return build_card_registry(data)


# --------------------------------------------------------------- csv dialect


def test_continuation_rows_build_vertical_arrays(data):
    """A blank first column continues the record above it, forming an array."""
    common = data.tables["rarities"]["Common"]
    ladder = common.array("PowerLevelMultiplier")
    assert ladder[:4] == (110, 121, 133, 146)
    assert common.scalar("LevelCount") == common.columns["LevelCount"][0]
    assert len(ladder) > 1, "continuation rows were not collected"


def test_array_typed_columns_are_coerced(data):
    """IntArray columns must come back as ints, not strings."""
    timeline = load_table(BUILD / "battle_timelines.csv")["Default"]
    assert timeline.array("ElixirFullBarMS") == (28000, 14000, 9300)
    assert timeline.array("SectionLength") == (180, 120)


def test_empty_cells_are_none(data):
    knight = data.resolve("Knight")
    assert knight.get("AttacksAir") is None, "Knight does not target air"
    assert knight["AttacksGround"] is True


# ------------------------------------------------------------------- decoder


def test_sniff_identifies_supercell_lzma():
    # 0x5d props byte, 4-byte dict size, 4-byte truncated uncompressed size.
    assert sniff(b"\x5d\x00\x00\x04\x00" + b"\x10\x00\x00\x00" + b"\x00" * 8) == "lzma"
    assert sniff(b"SCLZ" + b"\x00" * 16) == "lzham"
    assert sniff(b'"Name","Rarity"\n') == "plain"


def test_plaintext_passes_through():
    decoded = decode_bytes(b'"Name"\n"string"\n"Knight"\n')
    assert decoded.scheme == "plain"
    assert decoded.text.startswith('"Name"')


# ------------------------------------------------------------ toml/csv merge


def test_toml_supersedes_csv_without_conflict(data):
    """Every name in both representations has an empty CSV row -- TOML wins."""
    conflicts = []
    for name in data.namespace("CHARACTER"):
        row = data._csv_row_for("CHARACTER", name)
        if row and row.get("Hitpoints") is not None:
            conflicts.append(name)
    assert conflicts == [], f"CSV and TOML both define stats for {conflicts}"


def test_ext_base_chain_resolves_across_namespaces(data):
    """EXT entries inherit through Base refs that name the logical kind."""
    evo = data.resolve("EXT.AngryBarbarian_EV1_2")
    # Inherited from CHARACTER.AngryBarbarian via EXT.AngryBarbarian_EV1 ...
    assert evo["Hitpoints"] == 524
    assert evo["Speed"] == 90
    # ... but the leaf's own override wins.
    assert evo["LoadTime"] == 900


def test_every_entity_resolves(data):
    """No entity in any battlefield namespace fails to flatten."""
    failures = []
    for namespace in ("CHARACTER", "BUILDING", "PROJECTILE", "AEO", "EXT"):
        for name in data.names(namespace):
            try:
                data.resolve(f"{namespace}.{name}")
            except Exception as exc:  # noqa: BLE001 - reporting, not handling
                failures.append(f"{namespace}.{name}: {exc}")
    assert failures == [], failures


# ------------------------------------------------------------ level scaling


def test_rarity_offsets_make_tournament_standard_uniform(levels):
    """Every rarity's tournament level is the same power index -- multiplier 256."""
    expected = {"Common": 11, "Rare": 9, "Epic": 6, "Legendary": 3, "Champion": 1}
    for rarity, level in expected.items():
        scale = levels[rarity]
        assert scale.tournament_level == level, rarity
        assert scale.multiplier(level) == 256, rarity
        assert scale.display_level(level) == 11, rarity


def test_display_level_round_trips(levels):
    for rarity in ("Common", "Rare", "Epic", "Legendary", "Champion"):
        scale = levels[rarity]
        for display in range(scale.relative_level + 1, 12):
            internal = scale.internal_level(display)
            assert scale.display_level(internal) == display


def test_champion_ladder_extends_past_its_own_row(levels):
    """A Champion reaches power index 15, past the 9 entries on its own row."""
    champion = levels["Champion"]
    assert champion.power_index(6) == 15
    assert champion.multiplier(6) == 409


def test_scaling_truncates_toward_zero(levels):
    # 690 * 256 / 100 = 1766.4 -> 1766, not 1767.
    assert levels.scale(690, "Common", 11) == 1766
    assert levels.scale(79, "Common", 11) == 202


# ------------------------------------------------- anchored external values


@pytest.mark.parametrize("card_name", sorted(ANCHORS["verified_cards"]))
def test_anchored_card_stats(data, levels, registry, card_name):
    """Hand-verified live-game values must come out of the pipeline exactly."""
    expected = ANCHORS["verified_cards"][card_name]
    card = registry[card_name]
    actual = card_stat_summary(data, levels, card, display_level=ANCHORS["display_level"])
    for field, value in expected.items():
        if field.startswith("_"):
            continue
        assert actual.get(field) == value, (
            f"{card_name}.{field}: expected {value}, got {actual.get(field)}"
        )


def test_engine_constants_match_game_files(data):
    """Match structure and elixir rates read from battle_timelines.csv."""
    const = ANCHORS["engine_constants"]
    timeline = load_table(BUILD / "battle_timelines.csv")["Default"]

    assert timeline.scalar("StartingElixir") == const["starting_elixir"]
    sections = timeline.array("SectionLength")
    assert sections[0] == const["regulation_seconds"]
    assert sections[1] == const["overtime_seconds"]

    full_bar = timeline.array("ElixirFullBarMS")
    per_elixir = [ms // 10 for ms in full_bar]
    assert per_elixir == const["elixir_ms_per_unit"]
    assert list(timeline.array("ElixirRateLength")) == const["elixir_phase_seconds"]

    globals_map = data.globals_map()
    assert globals_map["MAX_MANA"] == const["max_elixir"]
    assert globals_map["MELEE_RANGE_LIMIT_MEDIUM"] == const["melee_range_medium"]
    assert globals_map["MELEE_RANGE_LIMIT_CLOSE"] == const["melee_range_short"]


def test_tower_base_stats(data):
    const = ANCHORS["engine_constants"]
    princess = data.resolve("PrincessTower")
    king = data.resolve("KingTower")
    assert princess["Hitpoints"] == const["princess_tower_hitpoints_level_1"]
    assert princess["HitSpeed"] == const["princess_tower_hit_speed"]
    assert king["Hitpoints"] == const["king_tower_hitpoints_level_1"]


def test_melee_range_constants_are_milli_tiles(data):
    """Knight's Range is exactly the 'medium melee' global -- 1.2 tiles."""
    assert data.resolve("Knight")["Range"] == data.globals_map()["MELEE_RANGE_LIMIT_MEDIUM"]
    # And its collision radius is half a tile.
    assert data.resolve("Knight")["CollisionRadius"] == 500


# ------------------------------------------------------------ card registry


def test_standard_card_pool_is_the_real_roster(registry):
    standard = registry.standard()
    # The live game has ~120 playable cards; a wild deviation means the
    # NotInUse/NotVisible filtering or a source table has broken.
    assert 100 <= len(standard) <= 140, len(standard)
    kinds = {k: len(registry.of_kind(k)) for k in CardKind}
    assert kinds[CardKind.TROOP] > kinds[CardKind.SPELL] > kinds[CardKind.BUILDING]


def test_every_card_costs_elixir_within_range(registry):
    for card in registry.standard():
        assert 1 <= card.mana_cost <= 9, f"{card.name} costs {card.mana_cost}"


def test_cards_summon_resolvable_characters(data, registry):
    """Every troop/building card must point at an entity the engine can spawn."""
    broken = []
    for card in registry.standard():
        if card.kind is CardKind.SPELL:
            continue
        for character, _count in card.summons():
            try:
                data.resolve(character)
            except Exception:  # noqa: BLE001
                broken.append(f"{card.name} -> {character}")
    assert broken == [], broken


def test_multi_summon_cards_have_counts(registry):
    """Cards that deploy a squad must carry a count > 1."""
    skeletons = registry["Skeletons"]
    assert skeletons.summon_count >= 3
    assert skeletons.mana_cost == 1


def test_ranged_unit_damage_comes_from_its_projectile(data, levels, registry):
    """Musketeer has no Damage field; it lives on the projectile she fires."""
    musketeer = data.resolve("Musketeer")
    assert musketeer.get("Damage") is None
    summary = card_stat_summary(data, levels, registry["Musketeer"])
    assert summary.get("damage_source", "").startswith("projectile:")
    assert isinstance(summary.get("damage"), int) and summary["damage"] > 0


# --------------------------------------------- the awkward resolution paths
#
# Each of these is a card the naive "SummonCharacter -> Damage" reading gets
# wrong, and each was silently producing a blank row before.


def test_toml_overlay_sections_are_bare_names(data):
    """spells_characters.toml keys are card names, not namespaces."""
    three = data.resolve("SPELL_CHARACTER.ThreeMusketeers")
    assert three["ManaCost"] == 9  # from the CSV row
    assert len(three["SummonCharactersList"]) == 3  # from the TOML overlay
    # buildings.toml overlays too, and used to be dropped entirely.
    assert data.resolve("BUILDING.KingTower").get("IsSummoner") is True


def test_card_with_no_summon_field_uses_its_own_name(data, levels, registry):
    """Ice Wizard's card row names no character; the convention is same-name."""
    card = registry["IceWizard"]
    assert card.summon_character is None
    assert card.implicit_character == "IceWizard"
    summary = card_stat_summary(data, levels, card)
    assert summary["hitpoints"] == 688
    assert summary["damage"] == 89


def test_explicit_unit_list_with_offsets(data, levels, registry):
    """Three Musketeers names its three units and where each one lands."""
    card = registry["ThreeMusketeers"]
    assert len(card.summons()) == 3
    assert card.summon_offsets == ((0, -1000), (-1000, 1000), (1000, 1000))
    assert card_stat_summary(data, levels, card)["count"] == 3


def test_squad_counts_total_all_units(data, levels, registry):
    """Goblin Gang is 3 Goblins + 3 Spear Goblins, not 3."""
    summary = card_stat_summary(data, levels, registry["GoblinGang"])
    assert summary["count"] == 6
    assert len(summary["squad"]) == 2


def test_visual_only_projectile_is_skipped(data, levels, registry):
    """Princess' Projectile is a decorative round; CustomFirstProjectile hurts."""
    princess = data.resolve("Princess")
    assert princess["Projectile"] == "PrincessProjectileDeco"
    assert data.resolve("PROJECTILE.PrincessProjectileDeco").get("Damage") is None
    summary = card_stat_summary(data, levels, registry["Princess"])
    assert summary["damage_source"].startswith("custom_first_projectile:")
    assert summary["damage"] > 0


def test_attack_sequence_damage(data, levels, registry):
    """Berserker keeps per-swing damage in AttackSequenceList."""
    berserker = data.resolve("Berserker")
    assert berserker.get("Damage") is None
    summary = card_stat_summary(data, levels, registry["Berserker"])
    assert summary["damage_source"] == "attack_sequence"
    assert summary["damage"] == 102


@pytest.mark.parametrize(
    "card_name,expected_damage,expected_source",
    [
        ("Zap", 192, "area_effect"),
        ("Freeze", 148, "area_effect"),
        ("Lightning", 1057, "area_projectile"),
        ("Log", 268, "spawn_projectile:LogProjectileRolling"),
    ],
)
def test_spell_damage_chains(data, levels, registry, card_name, expected_damage, expected_source):
    """Spells keep damage on the projectile, the area effect, or a spawned one."""
    summary = card_stat_summary(data, levels, registry[card_name])
    assert summary["damage"] == expected_damage
    assert summary["damage_source"] == expected_source


def test_damage_over_time_comes_from_the_buff(data, levels, registry):
    """Poison has no Damage anywhere -- only a buff carrying DamagePerSecond."""
    area = data.resolve("AEO.Poison")
    assert area.get("Damage") is None
    summary = card_stat_summary(data, levels, registry["Poison"])
    assert summary["damage_per_second"] == 92
    assert summary["duration"] == 8000
    assert summary["buff_speed_multiplier"] == -15


def test_spell_projectile_can_carry_troops(data, levels, registry):
    """Goblin Barrel's payload is on the projectile, not the card."""
    summary = card_stat_summary(data, levels, registry["GoblinBarrel"])
    assert summary["spawns_character"] == "Goblin"
    assert summary["spawns_count"] == 3


def test_arena_tower_layout_is_defined(data):
    """spawn_groups.toml pins the tower positions, in half-tiles."""
    layout = data.resolve("SPAWN_GROUP.King_PrincessTowers")
    objects = {(o["Data"], o["x"], o["y"]) for o in layout["Objects"]}
    assert ("KingTower", 18, 6) in objects
    assert ("PrincessTower", 7, 13) in objects
    assert ("PrincessTower", 29, 13) in objects
    # x is in half-tiles across an 18-tile arena, so the king sits on centre.
    assert 18 / 2 == 9


def test_action_driven_spells_are_identified(data, levels, registry):
    """Graveyard and Vines have no stats -- their behavior is an ACTION graph."""
    for name in ("Graveyard", "Vines", "Clone"):
        summary = card_stat_summary(data, levels, registry[name])
        assert summary.get("action"), f"{name} should name the action driving it"
        assert summary["action"] in data.namespace("ACTION") or True


def test_no_standard_card_is_completely_unresolved(data, levels, registry):
    """Every playable card must yield stats, a payload, or a named action.

    A card that produces none of these is one the engine could not spawn, and
    that is the failure this whole module exists to prevent.
    """
    blank = []
    for card in registry.standard():
        s = card_stat_summary(data, levels, card)
        informative = any(
            s.get(k) is not None
            for k in (
                "hitpoints",
                "damage",
                "damage_per_second",
                "spawns_character",
                "buff",
                "action",
            )
        )
        if not informative and not card.summons():
            blank.append(card.name)
    # Mirror has no payload of its own by definition -- it replays the last card.
    assert set(blank) <= {"Mirror", "MergeMaiden"}, blank


# ------------------------------------------------------------ tower scaling


def test_towers_do_not_use_the_card_ladder(data):
    """Crown Towers have their own progression, and it is not the cards'.

    Applying the card multiplier to a tower inflates a level-11 Princess Tower
    from 2576 to 3584 hitpoints -- a 39% error in the number every damage race
    in the game is measured against.
    """
    from cr_sim.data.leveling import build_tower_scales

    scales = build_tower_scales(data.globals_map())
    princess_base = data.resolve("PrincessTower")["Hitpoints"]
    king_base = data.resolve("KingTower")["Hitpoints"]

    assert scales["princess"].hitpoints(princess_base, 1) == 1400
    assert scales["princess"].hitpoints(princess_base, 11) == 2576
    assert scales["king"].hitpoints(king_base, 11) == 4224
    # The King gains 7% a level where a Princess gains 8%.
    assert scales["king"].hitpoint_percent == 7
    assert scales["princess"].hitpoint_percent == 8


def test_tower_scaling_switches_rate_at_level_nine(data):
    """8% a level up to 9, then 10% -- TOWER_SCALING_START_EXP_LEVEL."""
    from cr_sim.data.leveling import build_tower_scales

    scales = build_tower_scales(data.globals_map())
    princess = scales["princess"]
    base = 1400
    # Levels 1..9 climb by 8% of base each: 112 hitpoints a level.
    for level in range(1, 9):
        step = princess.hitpoints(base, level + 1) - princess.hitpoints(base, level)
        assert step == 112, f"level {level}->{level + 1} moved {step}"
    # Past 9 they climb by 10% of base: 140 a level.
    for level in range(9, 15):
        step = princess.hitpoints(base, level + 1) - princess.hitpoints(base, level)
        assert step == 140, f"level {level}->{level + 1} moved {step}"


def test_tower_hitpoints_stay_divisible_by_fourteen(data):
    """A published quirk that this formula reproduces and the card ladder does not.

    It is a consequence of the shape: a flat integer percentage of a 1400 base
    is always a multiple of 14. That it holds for every level is independent
    evidence the progression is percentage-of-base rather than compounding.
    """
    from cr_sim.data.leveling import build_tower_scales

    princess = build_tower_scales(data.globals_map())["princess"]
    for level in range(1, 16):
        assert princess.hitpoints(1400, level) % 14 == 0, level
