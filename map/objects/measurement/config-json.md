---
type: object
cluster: measurement
universe: live
status: verified
verified: 2026-08-30
commit: dc47f51
entity: cr_sim/train/run.py
---

# config.json

The one-shot record of what a run was asked to be, written into
`runs/<name>/` before the first update. No type, no schema, no validator —
four writers agree on a filename and on almost nothing else.

## Why this shape

Written once, and that is load-bearing twice over. It dates the start of the
run, which is how every listing is ordered (`_started_at`,
`cr_sim/train/watch.py:230-246`). And it is a *statement of intent recorded
before anything ran*, which is the only reason bug 1 was ever findable: the
file said `tower_level: 5` while every `--workers` rollout trained at 11.

Its key set is not free. `watch.py` pairs two runs into an A/B only while their
config key sets differ in at most four names (`_AB_MAX_DIFF`,
`cr_sim/train/watch.py:1799-1803`; applied at `:1974`), so **every field added here is spent
out of that budget** and the comments at `cr_sim/train/run.py:635-651` account for the spend
in place. Four rather than one because a head change drags its own KL reference
and a free-text note along with it. Deleting a key is worse than adding one: it
changes the key set and makes every new run unpairable with every old one,
which is why `shaping` stays where it is and `shaping_is_inert` was nested
inside `reward_schedule` instead (`cr_sim/train/run.py:639-649`).

Only the *names* of differing keys ever reach the page — one of them holds an
absolute Windows path (`_config_diff`, `cr_sim/train/watch.py:1812-1822`;
`_config_of`, `:4460-4470`).

## Shape

- Written at `cr_sim/train/run.py:605-654`, as
  `{**asdict(config), ...}` — so the whole of `PPOConfig`
  (`cr_sim/train/ppo.py:54-94`) is in the file, and a field added to `PPOConfig`
  silently lands here and costs a slot in the A/B budget.
- Fields worth naming, because something reads them: `tower_level` (`:609`) —
  the field bug 1 turned on; `deck` (`:606`), the only artefact in the tree that
  records it; `observation` and `observation_channels` (`:628-629`);
  `eval_opponent`, read off a real evaluation environment rather than asserted
  (`:634`); `eval_reward`, the pinned probe scale (`:652`); `probe` (`:638`);
  `reward_schedule` with `shaping_is_inert` nested (`:646-649`).
- **`workers` is not a key.** `num_envs` is, via `PPOConfig`, but nothing in the
  file says whether the environments were built by `_env()` or by
  `VecEnvConfig`. That is bug 1's axis, still unrecorded: the file cannot
  distinguish a run that took the local path from one that took the worker
  path. See `interface/vec-env-config.md` and `interface/crsim-env.md`.
- Four writers, four disjoint key sets:
  - `cr_sim/train/run.py:605` — a training run, ~40 keys, no `kind`.
  - `scripts/clone_policy.py:378-389` — twelve keys plus a prose `note`; a flat
    series, because a clone does not learn over time.
  - `scripts/run_ladder.py:461-484` — `kind: "job"`, `mode`, `observation`,
    and a `note` that says in prose why `ladder.json` is invisible on the page.
  - `scripts/measure_expert.py:162-178` — records `tower_level` and a `note`
    explaining which earlier number it replaces.
  - Plus `scripts/register_job.py:54-58` — three keys: `note`, `kind`,
    `registered_at`. Nothing else.
- Two keys are read across all of them and the rest is per-writer:
  `note` (`_note_of`, `cr_sim/train/watch.py:268-285`) and `kind` (`_kind_of`,
  `cr_sim/train/watch.py:4443-4457`). `kind == "job"` is the exact job/model split the
  census, every counter and the ranking rest on — name prefixes are not exact,
  since `bench-*` and `agent-*` are jobs while `probe-*` and `ab-*` are models.
- Every read is defensive on purpose. A config that is valid JSON but is not an
  object has no `.get`, and a non-UTF-8 file raises `UnicodeDecodeError` rather
  than `OSError`; either used to kill the watcher and freeze every served page
  (`_note_of`'s docstring, `cr_sim/train/watch.py:276-285`; `_read_json`, `:4472-4483`).
  `report._config` returns `{}` on a decode error (`cr_sim/train/report.py:34-42`).

Verified 2026-08-30 against `main` @ `dc47f51`. Every file cited here is clean
at that commit.

## Connected to

- **owns:** `deck` and `tower_level` — no other artefact in this cluster
  records either.
- **owned-by:** [`run-directory.md`](run-directory.md).
- **joins:** `measurement/ppo.md` — `PPOConfig` is spliced in whole;
  `measurement/reward-schedule.md` — `reward_schedule` is the resolved
  endpoints, not the flags; [`verdict.md`](verdict.md) and
  [`metrics-row.md`](metrics-row.md), which carry their own `eval_opponent` and
  `eval_reward` read off the environment that actually played;
  `surfaces/progress-page.md` — the only reader of the A/B budget.
- **looks-like-but-is-not:** the run's command line. `resumed` and `init_from`
  are recorded (`cr_sim/train/run.py:624-625`), but `--resume` appends to an existing
  directory and **rewrites this file**, so a resumed run's config is the second
  invocation's, not the first's.

## If you change this

- **Hits:** the A/B pairing budget (`cr_sim/train/watch.py:1799-1803`, `:1974`) — one new
  key makes every existing run one step further from pairable, and five make
  them unpairable. `_config_diff` (`cr_sim/train/watch.py:1812`) if the value could be
  machine-specific. `report._config` (`cr_sim/train/report.py:34`) and everything downstream of
  `record` in `report.collect` (`cr_sim/train/report.py:114-127`). `_kind_of` (`cr_sim/train/watch.py:4443`) if
  the key is `kind`. Every `config.json` on disk, which becomes the old key
  set — and unlike a metrics row, there is only ever one per run.
- **Does not hit:** [`metrics-row.md`](metrics-row.md). The obvious next stop
  is the metrics row, because both files carry `eval_opponent` and
  `eval_reward` and a reader compares them — but nothing copies between them.
  The row's fields come from `opponent_name`/`reward_name` on the live
  environment (`cr_sim/train/selfplay.py:163`, `:188`) and the config's are written once at
  startup. They can disagree, they have no guard that they agree, and the
  disagreement is a real signal rather than a bug in either file.

## Surfaces

| Surface | Role |
|---|---|
| `cr_sim/train/run.py` | writes, once, before the first update |
| `scripts/{clone_policy,run_ladder,measure_expert,register_job}.py` | write, four other key sets |
| `cr_sim/train/watch.py` | reads — ordering, the note, the job/model split, the A/B diff |
| `cr_sim/train/report.py` | reads |
| humans | `cat runs/<name>/config.json` — the first thing anyone opens when a number looks wrong |

## See

- Source: `cr_sim/train/run.py:590-654`, `cr_sim/train/watch.py:1799-1822`,
  `:4443-4483`
- As-built: `docs/training.md`
