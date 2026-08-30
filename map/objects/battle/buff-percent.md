---
type: object
cluster: battle
universe: live
status: verified
verified: 2026-08-30
commit: dc47f51
entity: cr_sim/engine/buffs.py
---

# Buff percentages

Timed status effects — Rage, Freeze, Poison and everything like them — and the
three functions that read the percentage column. The most dangerous trio in the
engine.

## Why this shape

A buff is a modifier attached to an *entity* rather than to an attack. Where
[attack-state.md](attack-state.md) decides whether this swing lands, a buff
changes every subsequent tick until it expires. Nothing else in the game lets a
card affect units it never touches.

The column that carries the mechanic uses **two conventions at once**, and the
data settles which is which without ambiguity: every value in the build is
either `<= 0` or `>= 100`, with nothing in between, so no value is ever
ambiguous. `<= 0` is a delta off 100 (Poison `-15`, Freeze `-100`); `>= 100` is
the whole multiplier (Rage `130` = 1.3x, Prince's second charge tier `170`).

The clincher is `IgnoreBarrel` at exactly `100` — a pure targeting-immunity
marker that must not change speed at all. As a whole multiplier that is 1.0x,
neutral. As a delta it would double the unit's speed, which is plainly not what
an immunity flag does. And read as deltas, Rage would be +130% against its real
+30%.

The three functions are **not interchangeable**, which is why they are three:

- `apply_multiplier(base, raw)` reads a *raw data value* under both conventions.
- `as_delta(raw)` normalises to a delta so values can be **summed** — `130`
  means "1.3x in total", so adding two raw values gives 2.6x instead of 1.6x,
  and each contributes a whole extra baseline. It also fixes a case that was
  quietly wrong: summed raw, `IgnoreBarrel` (100) against a Freeze (-100)
  cancelled to zero and the unit kept walking — a targeting-immunity flag
  curing Freeze.
- `apply_delta(base, summed)` takes an already-normalised sum, where 0 is
  neutral. **Routing a summed delta through `apply_multiplier` is silently
  wrong at exactly one point**: two Rages stacking to `+100` would be read as
  the whole multiplier 100, i.e. 1.0x, cancelling both instead of doubling.

All three clamp at zero: a stack of slows means "as slow as this gets", never
reversed.

## Shape

- `apply_multiplier` (`:145`), `apply_delta` (`:173`), `as_delta` (`:191`).
- `BuffSpec` / `build_buff_spec` / `buff_spec_from_row` — the data side;
  `ActiveBuff`, `BuffState`, `BuffTick` — the per-entity side, created lazily
  on `Entity.buffs` because most entities never carry one.
- **Count, stated precisely.** The module docstring says "112 entries in this
  build". That is the TOML `BUFF` namespace. `LogicData.names("BUFF")` — which
  is what a lookup actually searches, TOML unioned with the legacy
  `character_buffs.csv` rows — gives **157**. Both numbers are right and they
  measure different things; the docstring is not wrong, it is narrower than it
  reads.
- The docstring's long "Rage is the one surprise" passage (`:39-61`) reaches a
  *different* conclusion from the code below it: it says the module stores the
  raw value and applies one `(100 + percent)` formula everywhere. **Code
  wins** — `apply_multiplier` at `:145` implements the two-convention split.
  The docstring is the earlier decision, kept ([`../../_meta/overrides.md`](../../_meta/overrides.md), row 6).

Citations: `cr_sim/engine/buffs.py:145`, `:173`, `:191`, `:221`, `:328`,
`:425`, `:439`, `:462`; `cr_sim/engine/entity.py:203` (`buffs` lazily `None`);
`cr_sim/engine/combat.py:156-159` (`apply_delta` on the dealing side).
Verified 2026-08-30 against `main` @ `dc47f51`.

## Connected to

- **owns:** `BuffSpec`, `BuffState`, `ActiveBuff`, `BuffTick`.
- **owned-by:** [../build/logic-data.md](../build/logic-data.md) — the `BUFF`
  namespace plus `character_buffs.csv`.
- **joins:** [entity.md](entity.md) (`Entity.buffs`, and the one slot both copy
  paths give a real copy), [attack-state.md](attack-state.md),
  [battle.md](battle.md) (`update_buffs` and `update_conditional_buffs` are two
  separate phases), [action-interpreter.md](action-interpreter.md) — most
  modern buffs are applied by an `ACTION`, not a column.
- **looks-like-but-is-not:** `CrownTowerDamagePercent`, which is also a signed
  percentage and is **not** on this convention — it is always a negative delta
  applied `damage*(100+p)//100` (`cr_sim/engine/specs.py:293-301`). Do not route it
  through `apply_multiplier`.

## If you change this

- **Hits:** every speed, hit-speed and spawn-speed modifier, so Rage, Freeze,
  slows, charge tiers and every evolution buff at once; `Entity.apply_damage`
  and `damage_for`, which both call `apply_delta`; the status-effect tests,
  which are the only place the sign convention is asserted.
- **Does not hit:** damage-over-time. `DamagePerSecond` / `HitFrequency` are a
  rate and a cadence, not a percentage, and they do not pass through any of the
  three functions — a Poison that ticks for the wrong *amount* is a different
  bug from a Poison that slows by the wrong *fraction*. Nor does it hit the
  tower ladder, which uses a structurally identical `100 + percent` formula for
  an unrelated reason ([../build/tower-ladder.md](../build/tower-ladder.md)).

## Surfaces

| Surface | Role |
|---|---|
| `tests/test_buffs.py`, `tests/test_status_effects.py` | pin the conventions |
| `reference/anchors.json` → `tornado-attract-unit` (resolved) | external check on a percentage-shaped field |
| nothing outside `cr_sim/engine/` calls the trio | — |

## See

- Source: `cr_sim/engine/buffs.py`
