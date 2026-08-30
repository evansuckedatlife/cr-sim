---
type: object
cluster: battle
universe: live
status: verified
verified: 2026-08-30
commit: dc47f51
entity: cr_sim/engine/elixir.py
---

# Elixir and the timeline

`BattleTimeline` is the match's shape — how long it runs and how fast elixir
arrives; `ElixirBar` is one player's supply, in fixed-point thousandths. Both
come out of `battle_timelines.csv`.

## Why this shape

**They are one schedule, not two.** The rate changes are a sequence of segments
that runs across regulation *and* overtime without resetting, and the segments
do **not** line up with the period boundaries: the `Default` timeline is 120s
at 2800ms, 120s at 1400ms, 60s at 930ms, against 180s of regulation and 120s of
overtime. So the 2x segment starts a minute *before* regulation ends and
continues a minute into overtime. Hard-coding "double elixir in overtime" gets
it wrong in both directions.

**Fixed-point thousandths with a carried remainder.** `ELIXIR_PRECISION = 1000`
divides all three rates evenly, and `regenerate` uses `divmod` on a carried
remainder so integer truncation loses nothing across a three-minute match. No
float ever touches a spend decision — `ElixirBar.exact` is a float and is
display only, read by the viewer and by `elixir_advantage`.

`can_afford` compares whole `units`, so a bar at 3.999 cannot pay for a
4-elixir card. That is deliberate and is what makes affordability a step
function the action mask can express.

## Shape

- `ElixirSegment(start_tick, end_tick, ms_per_elixir, multiplier_tenths)`.
  `ElixirFullBarMS` is the time to fill all ten; one elixir is a tenth.
- `BattleTimeline` — `regulation_ticks`, `overtime_ticks`, `starting_elixir`,
  `segments`, `clock`; `total_ticks`, `segment_at`, `ticks_per_elixir`,
  `is_overtime`.
- `ElixirBar` — `amount` (thousandths), `_remainder`; `units`, `exact`,
  `regenerate`, `can_afford`, `spend`, `add`. Capped at
  `MAX_ELIXIR * ELIXIR_PRECISION`, and the remainder is cleared at the cap.
- `build_timeline(data, name="Default", clock=None)` — sums `SectionLength` by
  `SectionType` to split regulation from overtime.
- `BattleTimeline.elixir_gain_per_tick` (`:75`) is a **ghost**: a second,
  simpler regeneration formula. `ElixirBar.regenerate` uses `divmod` on the
  remainder instead and never calls it. Do not "unify" them — the divmod
  version is the one that does not lose elixir.

Citations: `cr_sim/engine/elixir.py:38`, `:52`, `:75`, `:92`, `:113-124`,
`:143`, `:1-24`; `cr_sim/engine/constants.py:38`, `:39`;
`cr_sim/engine/battle.py:331` (built once per battle), `:853-855`.
Verified 2026-08-30 against `main` @ `dc47f51`.

## Connected to

- **owns:** `ElixirSegment`, `ElixirBar`, `build_timeline`.
- **owned-by:** [../build/logic-data.md](../build/logic-data.md)
  (`battle_timelines.csv`) and
  [../build/tick-clock.md](../build/tick-clock.md) (every boundary is
  `seconds_to_ticks`).
- **joins:** [battle.md](battle.md) — `timeline` is in `_SHARED`,
  `regenerate_elixir` is phase 1 of 20, `Battle.run`'s default limit is
  `timeline.total_ticks`; [state-hash.md](state-hash.md) — both bars' `amount`
  are the `extra` folded into `Battle.hash`; [lookahead.md](lookahead.md)
  (`elixir_advantage` reads `exact` off the live board, never the projection).
- **looks-like-but-is-not:** `Battle.in_overtime`, which re-reads
  `timeline.is_overtime(self.tick)` on every access rather than caching a flag,
  so it stays right even when a test pokes the clock directly.

## If you change this

- **Hits:** every affordability decision and therefore the action mask
  (`cr_sim/api/encoding.py:586`) and `play_card` — the two must agree; the
  elixir channels of the observation; `Battle.hash`, since both amounts are
  folded; and the reward's `elixir_trade` term (`cr_sim/api/reward.py`).
- **Does not hit:** match length. `regulation_ticks` / `overtime_ticks` come
  from `SectionLength`, a different column from the elixir segments, and
  editing a rate does not shorten a match. The obvious wrong move is to reason
  about "overtime" from the segment list — they are misaligned on purpose, and
  `is_overtime` is the only correct test.

## Surfaces

| Surface | Role |
|---|---|
| `Battle._capture_frame` → the viewer | reads `exact` |
| `cr_sim/api/encoding.py`, `cr_sim/api/reward.py` | read |
| `cr_sim/play/session.py` | reads |
| `reference/anchors.json` → `elixir_ms_per_unit`, `elixir_phase_seconds`, `starting_elixir` | pins |

## See

- Source: `cr_sim/engine/elixir.py`
