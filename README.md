# cr-sim

A mechanically exact Clash Royale battle simulator — every troop, building and
spell, real collisions, real targeting, real elixir, real towers. No ladder, no
shop, no monetization. It exists to be a training environment for a machine
learning agent, so the requirements are: **correct, deterministic, and fast
enough to train against.**

Full plan: `../.claude/plans/i-want-to-build-purrfect-kernighan.md`.

## Status

| Milestone | State | Notes |
|-----------|-------|-------|
| **M0 — data pipeline** | ✅ | APK decoder, csv_logic + TOML ingestion, `EXT` inheritance, level scaling, 122-card registry, stat gate. **27 tests.** |
| M1 — fixed-point core, arena, tick loop | ⬜ next | |
| M2 — targeting, attacks, projectiles, towers | ⬜ | |
| M3 — pathing, collision, pushback | ⬜ | |
| M4 — elixir, deck cycle, win conditions | ⬜ | |
| M5 — spells, area effects, buffs | ⬜ | |
| M6 — all generic cards + special mechanics | ⬜ | |
| M7 — champions, evolutions, oddballs | ⬜ | |
| M8 — Gymnasium env, self-play, vec env | ⬜ | |

## What M0 established

The data layer is the foundation for accuracy, and the game's own files are the
only source that carries the fields interactions actually depend on —
`CollisionRadius`, `Mass`, `LoadTime`, `RetargetEachTick`, `StopTimeAfterAttack`,
`DeathSpawn*`. Wiki stats do not have these.

**Units**, confirmed from the files themselves:

| Quantity | Unit | Evidence |
|---|---|---|
| Distance | milli-tiles (1/1000 tile) | Knight `Range=1200` == `MELEE_RANGE_LIMIT_MEDIUM`; `CollisionRadius=500` = half a tile |
| Speed | tiles per minute | Knight `Speed=60` → 1.0 tile/s |
| Time | milliseconds | `HitSpeed=1200`, `DeployTime=1000` |

**Two representations, one merge rule.** Modern builds moved logic out of the
wide CSVs into per-entity TOML (`[CHARACTER.Knight]`, `[BUILDING.Cannon]`,
`[PROJECTILE.*]`, `[AEO.*]`, `[BUFF.*]`, `[ABILITY.*]`, `[ACTION.*]`). Every name
present in both has an *empty* CSV row, so TOML simply wins — verified as a test,
not assumed. `EXT` entries add inheritance (`Base = "CHARACTER.Foo"`, chainable)
and carry the Evolutions and champion ability forms.

**Level scaling** is a table, not a formula. `rarities.csv` gives a power ladder
(110, 121, 133, 146, 160, 176, 193, …) plus a per-rarity `RelativeLevel` offset
(Common 0, Rare 2, Epic 5, Legendary 8, Champion 10). The index is
`RelativeLevel + level - 1`, index 0 means unscaled, and values truncate toward
zero. The game displays every rarity on one 1–15 scale, so displayed level 11
is tournament standard for everything.

Two traps here, both caught by tests rather than assumed away:
- A rarity's stored ladder is too short for itself — a Champion reaches power
  index 15 but only 9 entries sit on its row, so the tail comes from the longest
  ladder in the file.
- Those ladders are **not** always identical across rarities. In the 2023 build
  Rare diverges from Common at index 13 (340 vs 339). A rarity's own values
  always win where it has them.

**Verified against live values** — this is the gate, in `reference/anchors.json`:

| Card | Computed | Live |
|---|---|---|
| Knight | 1766 HP, 202 dmg, 168 DPS | ✅ exact |
| P.E.K.K.A | 3760 HP | ✅ exact |
| Archer Queen | 1000 HP | ✅ exact |
| Rocket | 1484 dmg | ✅ exact |
| Fireball | 688 dmg, 2.5-tile radius | ✅ exact |
| Arrows | 122 dmg | ✅ exact |
| Snowball | 179 dmg | ✅ exact |

Match structure comes straight out of `battle_timelines.csv` and matches live
play exactly: 6 starting elixir, 180s regulation + 120s overtime, and elixir at
2800 / 1400 / 930 ms per unit across 120s / 120s / 60s phases.

## Data provenance

Extracted from a real APK, not a public dump:

```
Clash Royale build 150535029  ->  install_time_asset_pack.apk
  assets/csv_logic/  ->  349 files, all Supercell-LZMA, all decoded
```

Supercell packs these with a **truncated 9-byte LZMA header** (5 property bytes +
a 4-byte size, where the standard container wants 8), so `data/decode.py`
reassembles the stream before handing it to `lzma`. `SCLZ`/LZHAM and plaintext
are handled too.

`data_cache/` is gitignored — the game files are not redistributed. Re-extract
with:

```bash
python scripts/extract_apk.py "<dir containing the APK splits>"
```

## Use

```bash
python -m cr_sim.cli ingest              # what got loaded
python -m cr_sim.cli cards               # the 122-card playable pool
python -m cr_sim.cli cards --kind spell --level 14
python -m cr_sim.cli card Knight         # full resolved stats for one card
python -m cr_sim.cli validate            # the stat gate + open questions
python -m pytest                         # 27 tests
```

`freeze` re-cuts the regression baseline (`reference/card_stats.json`). Note the
deliberate split: **`anchors.json` is external truth and is never generated**
(that would make the gate circular), while `card_stats.json` is a generated
baseline whose only job is to make a new APK's balance changes visible instead
of silent.

## Layout

```
cr_sim/
  data/
    decode.py      Supercell LZMA/LZHAM/plaintext decoder
    csv_loader.py  csv_logic dialect: header row, type row, continuation-row arrays
    source.py      CSV+TOML merge, EXT Base-chain inheritance, globals
    leveling.py    the power ladder and rarity offsets
    cards.py       card registry (card != character) and stat summaries
    validate.py    the stat gate
  cli.py
scripts/extract_apk.py
reference/  anchors.json (external truth) + card_stats.json (generated baseline)
tests/
```
