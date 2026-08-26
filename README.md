# cr-sim

A mechanically exact Clash Royale battle simulator — every troop, building and
spell, real collisions, real targeting, real elixir, real towers. No ladder, no
shop, no monetization. It exists to be a training environment for a machine
learning agent, so the requirements are: **correct, deterministic, and fast
enough to train against.**

Full plan: `../.claude/plans/i-want-to-build-purrfect-kernighan.md`.

## Status

**255 tests.** Battles run end to end: units deploy, route, fight, die, towers
fall, and matches resolve on crowns, sudden death or a tiebreaker.

| Milestone | State | What landed |
|-----------|-------|-------------|
| **M0 — data pipeline** | ✅ | APK decoder, csv_logic + TOML ingestion, `EXT` inheritance and merge operators, level scaling, 122-card registry, stat gate. Every playable card resolves. |
| **M1 — deterministic core** | ✅ | Integer subtile geometry, PCG32 RNG, entities, elixir from the real timeline, arena from the shipped tilemap, bridge routing, 12-phase tick loop, state hashing. |
| **M2 — combat** | ✅ | Targeting (sight, filters, sticky targets), attack cycle (load/hit-speed, simultaneous resolution), damage, Crown Towers with their own scaling, King activation with its 3300ms delay, kamikaze units, swarm spread, and projectiles that actually fly. |
| **M3 — collision & pathing** | 🟡 | Circle collision, mass-weighted pushback, immovable tanks, derived swarm packing, and a spatial index. Weighted-grid pathfinding (the `PATHFINDING_*` costs) is still waypoint-only. |
| **M4 — match rules** | ✅ | Elixir, deck cycle, deploy legality, crowns, king-destruction, regulation end, overtime sudden death, and a percentage-based tiebreaker. |
| **M5 — spells & buffs** | 🟡 | Spells cast and land, area effects tick, buffs slow/freeze/rage, damage-over-time, Arrows' waves. The ACTION-driven spells (Graveyard, Clone, Vines, Mirror) still do nothing. |
| **M6 — the full roster** | ⬜ | Charge/dash, death spawns, spawners, shields, ramp damage — plus an **ACTION interpreter** |
| **M7 — champions & evolutions** | ⬜ | Abilities, Evolutions, Mirror/Clone/Graveyard |
| **M8 — ML interface** | ⬜ | Gymnasium env, observation/action encoding, self-play, multiprocess vec env |
| **M9 — vectorized port** | ⬜ | Optional; scalar core as the correctness oracle |

### Known gaps, stated plainly

These are things the engine does **not** do yet. None are bugs; all are
scheduled, and each is listed because a simulator that hides its gaps is worse
than one that names them.

- **Five spells still do nothing**: Graveyard, Clone, Vines, Mirror, Merge
  Maiden. Every one is defined in the ACTION graph rather than in stat fields,
  so they need the interpreter that M6 brings.
- **The Log does not roll.** Its damage lands where the throw lands rather than
  sweeping down the lane, which understates it against a spread push.
- **Pathfinding is waypoints, not a search.** Ground units route through the
  nearer bridge and otherwise steer straight. The `PATHFINDING_*` costs the
  game ships (`DEFAULT=8`, `ROAD=5`, `WATER=7`, `BLOCKED=50`, `BUILDING=50`)
  are not used, so units do not flow around a building the way they should.
- **Shields, invisibility and damage reduction** are read from the data but not
  applied.
- **Sparky's `LoadFirstHit` is not implemented.** One entity in the build sets
  it, and it is why a stun resets her charge.
- **Area-effect objects are inert.** The `AEO` layer (Poison's cloud, Tornado's
  pull, Graveyard's spawner) is loaded but nothing ticks it.
- **Performance is the constraint on M8.** ~5.7× real-time at 60 TPS, ~12-15×
  at 20 TPS. Enough to verify against, not enough to train against on its own:
  that needs the 20 TPS mode, all 8 cores, and eventually the vectorized port.

### Projectiles fly

Ranged attacks no longer resolve on the swing. A unit commits a projectile and
the damage arrives when the projectile does, which restores a whole layer of
the game — dodging, over-committing, and shots landing where a push *was*.

`Speed` is tiles per minute, the same unit as a character's. Three flight times
that are recognisable in play corroborate the reading:

| projectile | Speed | over its range | flight |
|---|---|---|---|
| Mortar shell | 300 | 11.5 tiles | 2300ms — a slow visible lob |
| X-Bow bolt | 1600 | 11.5 tiles | 417ms — near-continuous fire |
| King Tower | 1000 | 7.0 tiles | 417ms — a quick flat shot |

Homing shots re-aim from where they currently are; non-homing ones commit to
the point they were fired at, which is what makes a Mortar a prediction rather
than a guarantee. Splash resolves where the shot *landed*, so a Bomber still
punishes a clump when its intended target dies mid-flight.

### Open questions

Tracked in `reference/anchors.json` and printed by `cr-sim validate`. Three have
been closed with evidence; two remain:

| id | status | why it matters |
|---|---|---|
| `pekka-damage` | ✅ resolved | 842, not the 510 public sources list — settled by the one-shot breakpoint against a 721 hitpoint Musketeer |
| `tower-hp-scaling` | ✅ resolved | Towers use their own progression, not the card ladder; 39% error avoided |
| `building-collision-shape` | open, **high** | Blocks M3. The King Tower's footprint is a 3×3 square but the only per-entity extent in the data is a scalar `CollisionRadius` |
| `arrows-effective-damage` | ✅ resolved | Per wave. 3 waves x 122 = 366, which is exactly what clears Minions, Goblins and Princess |
| `bridge-width` | open, low | This build says 2 tiles everywhere; public sources say 3 for some arenas |

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
| Zap / Freeze / Lightning / Log | 192 / 148 / 1057 / 268 | ✅ exact |
| Poison | 92 dmg/sec over 8s | ✅ exact |
| Ice Wizard / Electro Wizard | 688 HP 89 dmg / 714 HP | ✅ exact |

**Interaction breakpoints** (`tests/test_interactions.py`) cross-check the same
numbers a second way — relationships between cards, which is what actually
proves a damage pipeline. All at equal level:

- P.E.K.K.A one-shots a Musketeer and a Wizard, but **not** a Knight
- Mini P.E.K.K.A one-shots a Musketeer
- Fireball alone leaves a Musketeer alive; Fireball + Zap kills her
- Zap clears Skeletons and Bats but **not** Goblins; the Log clears Goblins

This is how the one genuinely ambiguous number got settled. Public stat sites
listed P.E.K.K.A at 510 damage where the extracted build says 842; a same-level
Musketeer has 721 hitpoints, so only 842 reproduces the one-shot everyone knows
— and it still correctly fails to one-shot a Knight. The public figure is stale.

Match structure comes straight out of `battle_timelines.csv` and matches live
play exactly: 6 starting elixir, 180s regulation + 120s overtime, and elixir at
2800 / 1400 / 930 ms per unit across 120s / 120s / 60s phases.

### Where the numbers actually hide

A card's stats are not in one place, and the naive "`SummonCharacter` →
`Damage`" reading silently produces a blank row for a third of the roster.
Every one of these paths is exercised by a test:

| Path | Card that needs it |
|---|---|
| `Damage` on the character | Knight |
| `Projectile` → `Damage` | Musketeer |
| `CustomFirstProjectile` (plain `Projectile` is a visual-only "Deco" round) | Princess |
| `AttackSequenceList[].Damage` (multi-swing rework) | Berserker |
| No summon field at all → character of the **same name** | Ice Wizard, Electro Wizard |
| `SummonCharactersList` + per-unit offsets | Three Musketeers |
| Area effect → `Damage` | Zap, Freeze |
| Area effect → `Projectile` → `Damage` | Lightning |
| `SpawnProjectile` chain | The Log |
| Buff → `DamagePerSecond` | Poison, Tornado |
| `SpawnCharacter` on the spell's *projectile* | Goblin Barrel |
| `OnStartingAction` → the ACTION graph | Graveyard, Vines, Clone |

That last row is the important one for later milestones: this build ships **828
`ACTION` definitions**, a declarative behavior-graph language with expression
conditions (`target_in_range(x) && target_is_ground`) driving branching attacks,
evolutions and champion abilities. 18 characters and 4 spells hook into it.
M6/M7 will need an interpreter for it rather than hand-coded special cases.

The arena layout is data too — `SPAWN_GROUP.King_PrincessTowers` places the King
Tower at (18, 6) and Princess Towers at (7, 13) and (29, 13), in half-tiles
across the 18×32 arena. M1 does not have to guess the geometry.

## The arena is data, not a guess

Terrain comes from the game's own `tilemaps/tilemap.csv` — a 36×64 grid of
**half-tiles** covering the 18×32 board. Each cell is a bitfield, and every
value in the shipped file decomposes cleanly into known flags with **no leftover
bits**, which is what confirms the reading:

| bit | meaning |
|---|---|
| 1 / 2 | left / right lane road (`PATHFINDING_ROAD_COST=5` vs `DEFAULT=8`) |
| 16 | blocked — out-of-play margins and tower footprints |
| 32 | river; bridges are simply the gaps in it |
| 256 | bridge centre line |

What falls out, all verified: the river spans **y 15→17** (two tiles, centred on
16); two bridges **two tiles wide centred on x 3.5 and 14.5** — exactly the
Princess Tower positions, so towers sit in line with their bridge. The map holds
both sides; `spawn_groups.toml` lists one and the other mirrors through
`y → 64 − y`.

Watch the two coordinate systems: `spawn_groups.toml` positions are half-tile
grid *lines* (King Tower `x=18` → tile 9.0, dead centre), while tilemap cells
are *spans*. Reading the tower's x as a cell index puts it at 9.25 and slightly
off-centre forever.

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
python -m cr_sim.cli arena --map         # terrain, towers, deploy zones
python -m cr_sim.cli battle --html r.html   # run a match, write a replay
python -m pytest                         # 255 tests
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
