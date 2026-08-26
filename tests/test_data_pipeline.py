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
    assert summary.get("damage_from_projectile")
    assert isinstance(summary.get("damage"), int) and summary["damage"] > 0
