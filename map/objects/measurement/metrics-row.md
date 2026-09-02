---
type: object
cluster: measurement
universe: live
status: verified
verified: 2026-08-30
commit: dc47f51
entity: cr_sim/train/run.py
---

# MetricsRow

One JSON object per line of `runs/<name>/metrics.jsonl` — the run's only
time series. There is no `MetricsRow` type; a row is whatever dict reaches
`check_lift_is_named` (`cr_sim/train/selfplay.py:55`) on its way to the file.

## Why this shape

Open, append-only, and guarded at exactly one point.

**Open**, because a row is assembled in four layers and no layer knows the
others: PPO's update returns the losses (`cr_sim/train/ppo.py:565-567`), the
training loop adds the counters and the diagnostics (`cr_sim/train/ppo.py:398-434`),
`run.record` stamps `reward_weights` on every row rather than only where they
moved (`cr_sim/train/run.py:769-779`), and the probes `stats.update(...)` their own dicts in
(`cr_sim/train/run.py:803`, `:812`). A closed schema would have to be edited in four files
to add a number, and the point of this file is that a number can be added.

**Guarded at one point**, because the thing that must never happen is not a
missing key — it is an *unlabelled* one. A lift is meaningless without both the
opponent it was measured against and the reward it was counted in, and this
project has already spent three rounds of invalid comparisons and one retracted
headline finding that out. So one function refuses, and every writer of a
metrics row goes through it, which makes the omission unrecordable rather than
merely discouraged (`cr_sim/train/selfplay.py:55-160`).

The subtlety that makes the guard non-trivial: **one row genuinely carries
several measurements against several different opponents.** `run.py` merges the
random-control probe, the self-play ancestor ladder and the rating ladder into
one dict, and a single `eval_opponent` cannot name all three — measured on a
smoke run, where the ancestor's score arrived on a row saying it had been played
against the rating ladder's anchors. Hence `SCORED_FAMILIES`
(`cr_sim/train/selfplay.py:52`): each family looks for its own `<prefix>opponent` first and
falls back to `eval_opponent` only where the row carries a single measurement.

## Shape

- Written line by line: `print(json.dumps(check_lift_is_named(stats)),
  file=stream)` — `cr_sim/train/run.py:908-914`, flushed each row so a run that
  dies at hour three keeps hour two.
- Opened `"a"` on `--resume` and `"w"` otherwise (`cr_sim/train/run.py:733`); truncating
  would delete the hours being recovered.
- Written **after** the evaluation, never before: writing on the way in as well
  emitted each update twice, once without the eval fields
  (`cr_sim/train/run.py:781-784`, `:909-912`).
- The guard, five refusals in `check_lift_is_named` (`cr_sim/train/selfplay.py:55`):
  `eval_lift_sd` with no `eval_opponent` (`:111`); `eval_lift_sd` with no
  `eval_reward` (`:118`); a `SCORED_FAMILIES` score or Elo with neither its own
  opponent nor `eval_opponent` (`:133`); such a row naming the opponent's kind
  but not `<prefix>opponent_ref` — "pool" is not an opponent (`:142`);
  `ladder_elo` with no `ladder_pinned`, because the same battles pinned at
  +382 and pinned at 0 came out 377 points apart (`:148`). This corrects the
  index line, which said four.
- Writers, all through the guard except one: `cr_sim/train/run.py:913`;
  `scripts/clone_policy.py:413`; `scripts/run_ladder.py:310`, `:337`;
  `scripts/evaluate_vs_expert.py:224`; `scripts/measure_expert.py:233`; and the
  probes that hand rows up — `ladder_probe` (`cr_sim/train/ladder.py:837`),
  `ancestor_probe` (`cr_sim/train/selfplay.py:548`).
- **The exception: `scripts/register_job.py:50` writes rows with no guard at
  all**, and `cr_sim/train/watch.py:526-528` names it as such. That is why
  `_mode_of` must never default an unrecognised arm label to greedy.
- `scripts/clone_policy.py:414-416` writes the *same row twice*, at
  `updates` 1 and 2, because a clone is a single result and the page needs two
  points to draw a flat series.
- Reading: `read_metrics` (`cr_sim/train/watch.py:40-58`) drops a half-written final line as
  normal rather than an error, and skips anything that will not parse.
- Deduplication is **adjacent-only** (`_repeats`, `cr_sim/train/watch.py:623-642`). Keying a
  dict on `updates` across the whole file collapsed a resume's replayed numbers
  onto their pre-resume namesakes and deleted 816 real training battles. A
  counter that goes backwards marks a new segment and both rows are kept.
- A run counts as evaluated if it carries `eval_lift_sd` **or** `ladder_elo`
  (`_is_evaluation`, `cr_sim/train/watch.py:62-74`) — two families, never merged into one
  number.
- `report.collect` withholds `mean_lift` and `best_lift` entirely when a run
  carries more than one `(eval_opponent, eval_reward)` pair
  (`cr_sim/train/report.py:84-94`, `:123-127`), and keeps the raw readings so
  the page can say how many it is declining to average.

Verified 2026-08-30 against `main` @ `dc47f51`. `cr_sim/train/ladder.py` is
modified in the working tree (`../../CONTEXT.md`); `run.py`, `ppo.py`,
`selfplay.py`, `watch.py`, `report.py` and the four scripts are clean at that
commit.

## Connected to

- **owns:** `eval_opponent`, `eval_reward`, `ladder_pinned`,
  `<prefix>opponent_ref` — the labels that make a number comparable.
- **owned-by:** [`run-directory.md`](run-directory.md). This file is what makes
  a directory a run.
- **joins:** [`verdict.md`](verdict.md) — the same two refusals, on the file
  that outlives the run; `measurement/lift.md` — the row is where a lift is
  recorded, not where it is computed; `measurement/reward-schedule.md` —
  `reward_weights` is the target pushed at this update, not a per-battle fact
  (`cr_sim/train/run.py:773-779`); `measurement/self-play.md` and `measurement/ladder.md` —
  the probes that contribute the eval families; `surfaces/progress-page.md`.
- **looks-like-but-is-not:** [`config-json.md`](config-json.md). Both sit in the
  run directory and both carry `eval_opponent`, but the config's is a
  once-written *intention* and the row's is read off the environment that
  actually played (`opponent_name`, `cr_sim/train/selfplay.py:188`). Also not
  `runs/*/matches.jsonl`, which is a per-battle log nothing enumerates.

## If you change this

- **Hits:** `check_lift_is_named` (`cr_sim/train/selfplay.py:55`) — a new measurement family
  needs its own entry in `SCORED_FAMILIES` (`:52`) or it is exempt from the
  guard by accident, which is exactly how `ancestor_probe` went unnamed for its
  whole life. `read_metrics` and `_repeats` (`cr_sim/train/watch.py:40`, `:623`) if the new
  key is a counter, because `_repeats` decides resume-vs-rewrite from four
  named counters. `report.collect` (`cr_sim/train/report.py:54`) and `_is_evaluation`
  (`cr_sim/train/watch.py:62`) if it is a measurement. Every `metrics.jsonl` already on
  disk, including the ones in `.claude/worktrees/*/runs/`.
- **Does not hit:** `measurement/ppo.md`. The obvious next stop is `PPOConfig`
  or the training loop, because most of the row's keys are produced there — but
  nothing in `ppo.py` reads a row back, the stats dict is write-only from its
  side, and adding a key to `PPOConfig` puts it in
  [`config-json.md`](config-json.md), not here. A field that must be on both is
  a field written twice from two places, which is the shape of bug 1.

## Surfaces

| Surface | Role |
|---|---|
| `cr_sim/train/run.py` | writes, one row per update |
| `scripts/{clone_policy,run_ladder,evaluate_vs_expert,measure_expert}.py` | write, through the guard |
| `scripts/register_job.py` | writes, **not** through the guard |
| `cr_sim/train/watch.py` | reads — every chart, counter and A/B on the progress page |
| `cr_sim/train/report.py`, `notify.py`, `bot.py` | read |
| humans | `tail -1 runs/<name>/metrics.jsonl` |

## See

- Source: `cr_sim/train/selfplay.py:44-160`, `cr_sim/train/run.py:769-914`,
  `cr_sim/train/watch.py:40-74`, `:623-642`
- As-built: root `CLAUDE.md`
