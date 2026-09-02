---
type: object
cluster: measurement
universe: live
status: verified
verified: 2026-08-30
commit: dc47f51
entity: cr_sim/train/run.py
---

# Checkpoint

A `.pt` file holding weights plus everything needed to rebuild the network they
belong to and to say what chose them. Two writers with different payloads: the
training checkpoints (`best.pt` / `checkpoint.pt` / `final.pt`) and the clone's
`cloned.pt`.

Verified 2026-08-30 against the working tree at `dc47f51` (nine files
uncommitted — `../../CONTEXT.md`, Verification basis).

## Why this shape

Shapes are **not** stored. They are a property of the environment, and taking
them from the env is what makes a mismatch fail loudly at load rather than
quietly score a policy against an observation it was never trained on
(`cr_sim/train/evaluate.py:116-124`). What *is* stored is everything the
environment cannot re-derive: which head the parameters fit, which observation
they were trained on, and — for a promoted checkpoint — which statistic
promoted it.

`promoted_on` exists because a checkpoint chosen on Elo and one chosen on lift
are not comparable, and the file used to say `rolling_lift` whichever it was
(`cr_sim/train/run.py:877-881`). `head` exists because a factored head's
parameters do not fit a flat one and the failure is a shape error about a
tensor nobody can place (`cr_sim/train/evaluate.py:126-130`).

## Shape

Training (`cr_sim/train/run.py`):

- `best.pt` `:868-885` — `state_dict`, `head`, `observation`, `stats`, the
  rolling statistic under its own key, `promoted_on`, `window`. Written only
  when the rolling mean of `_BEST_WINDOW` readings (`:47`) beats the best so
  far, never on a single reading `:851-866`.
- `checkpoint.pt` `:891-905` — the only one carrying `optimiser`. Adam's
  moment estimates are most of what a long run has learned about its own
  gradients.
- `final.pt` `:1053-1054` — `state_dict`, `head`, `observation`, and nothing
  else. Chosen by nothing at all.

Identity, when the file does not say (`cr_sim/train/ladder.py`):

- `HEAD_BY_PARAMETERS` `:98` / `head_for_parameters` `:106` — which head has
  this many weights. **Raises rather than falling back to "flat"**: a
  checkpoint loaded into the wrong head does not raise, it plays badly, and a
  rating built on it rates a network nobody built `:107-124`. The table is v1
  counts only — 22 of the 42 checkpoints on disk record neither `head` nor
  `observation`, so this table is what identifies them — and a count matching
  nothing is either a different observation or the pre-`ActorCritic`
  generation `:92-97`.
- `player_from_checkpoint` `:240` reads head and observation off the payload
  where recorded and infers the head from the parameter count where not,
  stamping `head_source` `:150` so a reader can see which ratings rest on an
  inference. `default_player_name` `:226` names the *weights*, not the
  directory — `_GENERIC_STEMS` `:223` is why
  `checkpoints/headablate-{flat,factored}.pt` do not both arrive as
  "checkpoints" and merge into one entrant. `parse_player` `:271` is the
  command-line form.

Clone (`scripts/clone_policy.py:288-301`) — `state_dict`, `observation`
(**from `--observation`, not from `data.observation`**), `proposer`,
`demo_meta`, `targets`, `pass_weight`, `head`, `clone`.

Readers: `load_policy` `cr_sim/train/evaluate.py:116` runs `check_observation` `:89` and
builds from `payload.get("head", "flat")`; `Player.load`
`cr_sim/train/ladder.py:193` does the same for a ladder entrant;
`_load_reference` `cr_sim/train/run.py:387` and `--init-from` `cr_sim/train/run.py:684-689` refuse a head
mismatch before training starts.

## Connected to

- **owns:** the `observation` and `head` strings every later run trusts.
- **owned-by:** [`run-directory.md`](run-directory.md) — the directory these
  sit in beside `config.json`, `metrics.jsonl` and `verdict.json`.
- **joins:** [`demonstrations.md`](demonstrations.md) (the clone's critic is
  the demonstrations' `value` column); [`ladder.md`](ladder.md) (a checkpoint
  is a `Player`, and `Player.ref` is what a metrics row records);
  [`lift.md`](lift.md) (what `best.pt` was selected on).
- **looks-like-but-is-not:** the promotion statistic itself. `best.pt` carries
  `rolling_lift` *or* `rolling_ladder_elo`, never both, and `promoted_on` says
  which — a reader that assumes `rolling_lift` reads an Elo as a lift.

## If you change this

- **Hits:** `load_policy`, `Player.load` and `_load_reference` all read
  `payload.get("head", ...)` and `payload.get("observation", "v1")` with a
  default that means *predates the field*. Adding a required key makes every
  checkpoint on disk unloadable; adding an optional one with a default silently
  labels old files. `check_observation` compares parsed features, not strings
  (`cr_sim/train/evaluate.py:105-107`), so a rename of a variant is a load failure, not a
  mislabel.
- **Does not hit:** the network's shapes. Those come from the environment at
  `net_config_for(env, head=...)` every time, so a checkpoint written under one
  deck loads cleanly into an environment built with another — same
  `vocab_size`, same `vector_size`, strict load succeeds, `check_observation`
  passes. The obvious next assumption, that recording `observation` makes a
  checkpoint self-describing, is wrong: **no artefact here records the deck.**
  See [`../interface/check-observation.md`](../interface/check-observation.md),
  and — for the second `DEFAULT_DECK` a checkpoint can silently meet —
  [`../surfaces/play-server.md`](../surfaces/play-server.md), which owns
  `cr_sim/play/server.py:46` and the three other restatements beside it.

## Surfaces

| Surface | Role |
|---|---|
| `cr_sim.train.run` | writes three; reads one on `--resume` / `--init-from` |
| `scripts/clone_policy.py` | writes `cloned.pt` |
| `scripts/run_ladder.py`, `cr_sim/train/ladder.py` | read as `Player` weights |
| `cr_sim/play/policy.py` | reads one to put a policy behind the browser opponent |
| `checkpoints/` at the repo root | a human-curated shelf of promoted weights |

## See

- Source: `cr_sim/train/run.py`, `cr_sim/train/ladder.py`
