---
type: object
cluster: measurement
universe: live
status: verified
verified: 2026-08-30
commit: dc47f51
entity: cr_sim/train/run.py
---

# Run directory

`runs/<name>/` — the four files a measurement leaves behind: `config.json`
(what was asked for), `metrics.jsonl` (one row per update), `verdict.json`
(the result) and the checkpoints. Not a type; a layout that six writers agree
on.

Verified 2026-08-30 against the working tree at `dc47f51` (nine files
uncommitted — `../../CONTEXT.md`, Verification basis).

## Why this shape

A run that takes hours and prints to a terminal loses everything when the
terminal closes, and the questions worth asking afterwards need the whole
series (`cr_sim/train/run.py:8-12`). Beyond that, the layout is a **contract
with the progress page**: anything that produces a number and does not write
one of these is invisible, which is why the clone and the ladder — neither of
which is a training run — both write a `config.json` and a two-row
`metrics.jsonl` (`scripts/clone_policy.py:378-390`, `:414-417`).

`config.json`'s key set is a **budget**, not a bag. `watch.py` pairs two runs
for A/B only while their key sets differ by at most four, so every field added
here is spent out of that allowance, and a field is nested rather than added
flat for that reason alone (`cr_sim/train/run.py:635-651`). Deleting a key is worse than
adding one: it makes every new run unpairable with every old one.

## Shape

- Directory and files: `cr_sim/train/run.py:515-517`, opened for append on `--resume`
  `:733`.
- `config.json` `cr_sim/train/run.py:605-654`. Records the deck, arena, reward,
  `tower_level`, self-play cadence, `init_from`, `resumed`, the observation
  and its channel list, `eval_opponent` read off a real evaluation environment
  `:634`, `probe` `:638`, `reward_schedule` with
  `shaping_is_inert: args.reward != "simple"` `:646-649`, and `eval_reward` as
  the pinned constant `:652`.
- `metrics.jsonl` — [`metrics-row.md`](metrics-row.md). Every writer routes
  through `check_lift_is_named` **except one**: `cr_sim/train/run.py:913`,
  `scripts/clone_policy.py:413`, `scripts/run_ladder.py:310`, `:337`,
  `scripts/evaluate_vs_expert.py:224`, `scripts/measure_expert.py:233`,
  `cr_sim/train/selfplay.py:548`, `cr_sim/train/ladder.py:837` — and
  `scripts/register_job.py:50`, which does not, as `cr_sim/train/watch.py:526-528`
  says in source. The row is
  written **after** the evaluation, not before, or the eval fields never reach
  the file (`cr_sim/train/run.py:908-913`); and once, not twice — writing on the way in as
  well read as two trainers racing on one file.
- Every row carries `reward_weights`, the weights *pushed* at that update, on
  every row and not only the ones that moved: a schedule in `config.json` plus
  a `--resume` from a different step does not reconstruct it (`cr_sim/train/run.py:774-779`).
- `verdict.json` — [`verdict.md`](verdict.md).
- Registration: `scripts/register_job.py:35` puts a process on the progress
  page when it does not register itself.

**The contract the watcher enumerates is two files, not four.** A directory is
a run to `cr_sim/train/watch.py` when it holds a `metrics.jsonl`, found by
`rglob` rather than `iterdir` so a sweep's nested variants are visible
(`cr_sim/train/watch.py:4647-4649`); `verdict.json` and `summary.json` are
reached by two *separate* `rglob`s (`:4499`, `:4516`), which is the only way a
verdict in a directory with no metrics file is visible at all. Five programs
write these directories with five disjoint key sets, one of which
(`scripts/register_job.py:50`) skips the naming guard.

**Not every directory under `runs/` satisfies the contract**, and how many do is
not recorded here: it changes between one reading and the next, and a
checked-in count of generated data is wrong by the afternoon
([`../../_meta/schema.md`](../../_meta/schema.md), "What may not go in a card").
Say "every run directory" and mean the contract.

## Connected to

- **owns:** the four filenames and the promise that all four describe one run.
- **owned-by:** nothing — this is the top of the measurement cluster.
- **joins:** [`verdict.md`](verdict.md), [`checkpoint.md`](checkpoint.md),
  [`lift.md`](lift.md) (the row field `eval_lift_sd`),
  [`reward-schedule.md`](reward-schedule.md) (`reward_schedule` and the
  per-row `reward_weights`), and
  [`../surfaces/progress-page.md`](../surfaces/progress-page.md), which owns
  what the watcher reads out of this directory and how deep it looks.
- **looks-like-but-is-not:** `data_cache/demos*/`. Also an artefact directory,
  also stamped with provenance, but nothing on the progress page reads it and
  its provenance lives inside the `.npz` rather than beside it.

## If you change this

- **Hits:** the four other `metrics.jsonl` writers, because they hand-build a
  row rather than sharing a schema — a new required field has to be added five
  times. And `config.json`'s A/B pairing: one added key is one quarter of the
  budget `watch.py` allows.
- **Does not hit:** anything already on disk. `report.py` and `watch.py` read
  with `.get(...)` throughout and treat a missing key as *predates the field*,
  so a new key does not invalidate old runs — and that is exactly why a
  removed key is the dangerous move, not an added one. Nor does adding a
  `config.json` field reach a worker: `VecEnvConfig` is a separate object with
  its own field list, which is bug 1's shape and is still live for
  `skip_forced` (see [`../interface/crsim-env.md`](../interface/crsim-env.md)).

## Surfaces

| Surface | Role |
|---|---|
| `cr_sim/train/watch.py` | the progress page; the only view of the project in one place |
| `cr_sim/train/report.py` | the comparison table |
| `scripts/register_job.py` | writes a run row for a process that is not a training run |
| a human running `--resume` | reads `checkpoint.pt` and appends to `metrics.jsonl` |

## See

- Source: `cr_sim/train/run.py`
