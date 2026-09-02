---
type: object
cluster: interface
universe: live
status: verified
verified: 2026-08-30
commit: dc47f51
entity: cr_sim/train/evaluate.py
---

# check_observation — and what it cannot see

The one guard between a checkpoint and an environment that does not match it. It
compares `ObservationFeatures`, including `version`, and raises. **It never
touches `vocab`,** and no model artefact records a deck, so the vocabulary half
of the promise has no guard at all. This card exists to make that gap
un-missable.

## Why this shape

Changing the observation invalidates every checkpoint that predates the change.
Nine channels of weights do not fit a thirteen-channel network, and that surfaces
as a size-mismatch on `conv.0.weight` — which says nothing about the cause.
Worse, where the channel *count* happens to match while the channels mean
different things, nothing fails and the policy simply plays badly. So a run
records a **name**, and this compares names before any weight is loaded.

A checkpoint carrying no `observation` key predates the field and is assumed v1,
which is what it is.

## Shape

- `check_observation(payload, env)` — `parse_observation(payload["observation"])`
  vs `env.encoding.features`; `ObservationFeatures.__eq__` is the whole test.
- Two callers: `load_policy` (`cr_sim/train/evaluate.py:125`), which every evaluation and
  ladder path goes through, and `cr_sim/train/ladder.py:214`.
- **What records an observation:** `best.pt`, `checkpoint.pt`, `final.pt`,
  `cloned.pt` (all record `"observation"`), `runs/*/config.json`
  (`"observation"` plus the resolved `"observation_channels"` name list),
  `Demonstrations.observation`, and `VecEnvConfig.observation`.
- **What records the deck:** `runs/*/config.json` `"deck"` only — and it writes
  the module literal `DEFAULT_DECK`, not the deck read off the env. **No model
  artefact records it.** All four checkpoint payloads are `state_dict`, `head`,
  `observation` and metrics.

Citations: `cr_sim/train/evaluate.py:89-113`, `:125`; `cr_sim/train/ladder.py:214`;
`cr_sim/train/run.py:868-885`, `:891-895`, `:1053` (the three payloads),
`scripts/clone_policy.py:288-302` (the fourth), `cr_sim/train/run.py:606`,
`:628-629` (`config.json`), `cr_sim/api/vec.py:109-113`.

## Connected to

- **owns:** nothing.
- **owned-by:** [`observation-features.md`](./observation-features.md) — this
  guard is exactly as strong as that dataclass's `__eq__`.
- **joins:** [`encoding-config.md`](./encoding-config.md) (the half it does not
  cover); `Demonstrations` and the promotion probe in the **measurement**
  cluster — index rows, cards stub.
- **looks-like-but-is-not:** strict `load_state_dict`. It looks like a second
  guard and covers a different, narrower thing: a channel-count change, a head
  change, a vocabulary **size** change. It cannot see a same-width, different-
  meaning change of any kind.

## If you change this

- **Hits:** every evaluation and ladder load, since both go through
  `load_policy`; `cr_sim/train/run.py:502`'s claim that loading the anchors routes them
  through this guard.
- **Does not hit:** the **training loop's own weight loads**. The obvious
  assumption — "the guard runs wherever a checkpoint is opened" — is wrong at
  three places. `--init-from` compares `head` and never `observation`
  (`cr_sim/train/run.py:684-689`). `_load_reference`, the KL trust-region anchor, calls
  `net_config_for` and `load_state_dict` directly with no check
  (`cr_sim/train/run.py:396-398`). `cr_sim/play/policy.py:86-91` does not check either — it
  builds its encoding *from* the recorded name, which is a different and stricter
  arrangement for that path but means the browser opponent never compares
  anything.

## The three gaps, filed as gaps

1. **No artefact records the deck.** A checkpoint moved out of its run directory
   carries nothing about what its 80 vocabulary-keyed columns meant
   ([`observation-vector.md`](./observation-vector.md)). Two same-size decks pass
   this guard, pass strict loading, and mean different things column for column.
   Nothing in `tests/` compares a vocabulary across a load.
2. **The name is still a declaration in one place.** `Demonstrations.observation`
   is stamped from a parameter, and `scripts/make_demos.py` passes the literal
   `"v1"` on its single-variant branch rather than reading it off the env it
   built. The branch is currently guarded by `names == ["v1"]`, so the
   declaration is true today and is true by coincidence of two literals agreeing.
   The multi-variant branch takes the name from the variant's own key and does
   not have this shape.
3. **Empty is not v1.** `Demonstrations.observation` defaults to `""` for a shard
   written before the field existed, and the docstring is explicit that this
   "must not be silently treated as" v1. Nothing enforces that; the field is a
   string and any consumer may coerce it.

Citations: `scripts/make_demos.py:357-372` (**working-tree lines** — see
[`../../CONTEXT.md`](../../CONTEXT.md)), `cr_sim/train/clone.py:68-78`,
`:275-282`.

## Surfaces

| Surface | Role |
|---|---|
| `cr_sim/train/evaluate.py:125` (`load_policy`) | the only path most loads take |
| `cr_sim/train/ladder.py:214` | reads |
| `cr_sim/train/run.py:396-398` (`_load_reference`), `:674-695` (`--init-from`) | **do not** call it |
| `cr_sim/play/policy.py:86-91` | does not call it; rebuilds the encoding from the recorded name instead |
| `scripts/run_ladder.py:14` | documents "one ladder per observation" and relies on this |
| humans reading `runs/*/config.json` | the only place a deck is written down at all |

## See

- Source: `cr_sim/train/evaluate.py`
- As-built: `docs/training.md`, section "Which observation changes helped"

*Verified 2026-08-30 against `main` @ `dc47f51`. `train/evaluate.py`,
`train/ladder.py`, `train/clone.py` and `scripts/make_demos.py` are in the
uncommitted working tree: `evaluate.py` and `ladder.py` carry no line shift at
the points cited, `clone.py` shifts only past line 290, and every
`make_demos.py` line here is a working-tree number.*
