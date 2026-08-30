---
type: object
cluster: battle
universe: live
status: verified
verified: 2026-08-30
commit: dc47f51
entity: cr_sim/engine/lookahead.py
---

# Projection

What the board is worth if nobody plays another card, computed by playing a
clone forward. An *exact* evaluation, not an estimate.

## Why this shape

The engine is deterministic, so from any position there is exactly one answer
to "what happens if both players stop playing" — not a distribution, a fact.
That makes a projection a measurement rather than a prediction, and it costs a
clone plus some ticks instead of a trained network.

Which matters because the learned critic was the weak link: on its own training
distribution it explained six per cent of the variance in returns, so PPO's
advantages were mostly noise. A projection needs no training and cannot be
miscalibrated.

It is deliberately **narrow**. Elixir in hand, cards in cycle and everything
either player might do next are invisible to it. It answers only "what is
already committed to the board" — which is the question "do I need to respond
right now" turns on. `elixir_advantage` is read from the live board, not from
the projection, precisely because a projection plays forward without either
side spending, so both bars fill to the cap and the projected difference is
always zero.

Three things make it *invisible*, and each was a real leak:

1. `_is_quiet` skips the clone entirely when only untouched towers stand — the
   common case between pushes, worth more than any tuning of the simulation
   that follows.
2. The branch's `config` is **replaced** with a `record_frames=False` copy.
   `clone` empties the frame list, but the switch lives on the config, which a
   branch *shares* — so a projection taken while a replay was recording went on
   capturing frames and threw the lot away.
3. `entity_id_cursor()` / `restore_entity_ids()` bracket the run, in a
   `finally`. Without it, asking what happens next changes what happens next
   ([entity-ids.md](entity-ids.md)).

## Shape

- `Projection` — frozen: both crown counts, both tower hitpoint totals and
  fractions, `ticks`, `decided`.
- `project(battle, horizon_ticks=None)` — `None` runs to the end of the match
  (~200ms from mid-match); a few seconds of horizon costs a tenth of that.
- `_read` walks `entities` **and** `graveyard`, because a destroyed tower must
  still be counted, and does it as two loops rather than one concatenated tuple
  — building a tuple of every corpse is work proportional to match length, on
  the function that *is* the whole cost when the board is quiet.
- `committed_value(...)` — crowns dominate; `tower_weight` separates equal
  crowns; `elixir_weight` is the one term that makes it usable as a reward
  potential, because spending lowers the value immediately.

Citations: `cr_sim/engine/lookahead.py:31`, `:56-71`, `:75-105`, `:109`,
`:122-131`, `:135-152`, `:155`, `:169`.
Verified 2026-08-30 against `main` @ `dc47f51`.

## Connected to

- **owns:** `Projection`, `elixir_advantage`, `committed_value`.
- **owned-by:** [battle-clone.md](battle-clone.md) — a projection *is* a clone
  plus a loop.
- **joins:** [entity-ids.md](entity-ids.md), [battle-config.md](battle-config.md)
  (`record_frames`), [elixir.md](elixir.md) (`ElixirBar.exact`, a float, is
  display-only and this is a display-grade read).
- **looks-like-but-is-not:** the learned value head. `committed_value` is a
  potential over board state; the critic predicts returns. Bug 2 — a demo set
  whose value targets came from one reward while every fine-tune ran another —
  is about the critic, not about this
  ([../measurement/demonstrations.md](../measurement/demonstrations.md)).

## If you change this

- **Hits:** the projected reward (`cr_sim/api/reward.py`, the `projected` mode)
  and therefore every return, advantage and lift measured under it; the
  scripted proposer (`cr_sim/train/scripted.py`), which projects per candidate;
  throughput, because `_is_quiet` is the difference between a clone per
  decision and none.
- **Does not hit:** the observation. Nothing projected reaches the agent's
  input — a projection is a *reward* term, so changing it changes what the
  agent is paid, not what it sees, and no `check_observation` will notice. Nor
  does it hit the gamma-correct potential in `docs/training.md:483-504`: that
  plan was written and reverted and is a **ghost** — do not re-derive it from
  that paragraph.

## Surfaces

| Surface | Role |
|---|---|
| `cr_sim/api/reward.py` (`ProjectedReward`) | reads |
| `cr_sim/train/scripted.py` | reads |
| `tests/test_lookahead.py` | pins isolation, id restoration and the frame switch |

## See

- Source: `cr_sim/engine/lookahead.py`
