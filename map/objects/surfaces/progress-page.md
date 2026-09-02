---
type: object
cluster: surfaces
universe: live
status: verified
verified: 2026-08-30
commit: dc47f51
entity: cr_sim/train/watch.py
---

# Progress page

`cr_sim/train/watch.py` — the served page, and the only place the state of this
project is visible in one view. `cr_sim/train/report.py` aggregates for it,
`scripts/register_job.py` writes entries into it, and
`cr_sim/train/{notify,bot}.py` are the two paths that were meant to replace it.

## Why this shape

A separate process, deliberately: it must not be able to slow a run down, crash
it, or hold a lock on the file it reads (`cr_sim/train/watch.py:8-11`). It draws
the **lift**, not the return, first and largest, because the trainer's own
return is measured while exploring and runs about eighteen points optimistic
against a paired-seed control (`:12-18`).

**The rule that has no enforcement anywhere in the tree.** Python does not
reload an edited module in a running process, so after any change to
`cr_sim/train/watch.py` the served page silently keeps rendering the old code. A
fresh timestamp proves nothing — the stale process rewrites `progress.html`
every 15 seconds. Verify by grepping the generated page for a string only the
new code emits. It is written in the root `CLAUDE.md` ("The watcher runs stale
code") and nowhere in the source, which is why `read_ladder`
(`cr_sim/train/watch.py:775-800`) is landed **dark on purpose** and says so in
its own docstring.

## Shape

What this file hardcodes about the rest of the tree, and where each lands:

- **`_run_roots`** (`cr_sim/train/watch.py:4427-4441`) reads `ROOT/"runs"` **and
  every `ROOT/.claude/worktrees/*/runs`** — the page shows three other
  checkouts' run directories as if they were this one's, on purpose, because
  the experiments actually moving are usually a directory away. See
  [`worktree-shadows.md`](worktree-shadows.md).
- **`rglob`, not `iterdir`** (`:4649`): a sweep nests its variants one level
  deeper, so run discovery is depth-agnostic. Verdicts and summaries are found
  by a separate `rglob` (`:4499`, `:4516`), which is the only way a verdict in a
  directory with no metrics file is reachable at all.
- **`_kind_of`** (`:4443-4459`) splits job from model on `config.json`'s
  `kind == "job"`, which `scripts/register_job.py:56` writes and no trainer
  writes at all. Name prefixes are **not** exact and the docstring says so:
  `bench-*` and `agent-*` are jobs, `probe-*` and `ab-*` are models.
- **The four-key A/B budget** (`_AB_MAX_DIFF = 4`, `:1803`, applied `:1974`) and
  names-only config diffing (`:1812`), because one config value is an absolute
  Windows path. A new `config.json` key is spent out of that budget — see
  [`../measurement/config-json.md`](../measurement/config-json.md).
- **The default output** `ROOT/"progress.html"` (`:4662-4663`), which is where
  the `progress.html` and `progress.json` at the repo root come from.
- **`report.collect`** withholds `mean_lift` and `best_lift` **entirely** when a
  run mixed scales (`cr_sim/train/report.py:82-99`), which is bug 6 enforced at
  the aggregation layer rather than argued about.

Citations: `cr_sim/train/watch.py:150`, `:526-528`, `:775-800`, `:1803`,
`:1812`, `:1974`, `:4427-4441`, `:4443-4459`, `:4499`, `:4516`, `:4649`,
`:4662-4663`; `cr_sim/train/report.py:54`, `:82-99`, `:318`;
`scripts/register_job.py:35`, `:56`; root `CLAUDE.md:8-35`, `:48-54`.
Verified 2026-08-30 against `main` @ `dc47f51`.

## Connected to

- **owns:** the page, `progress.html`, `progress.json`.
- **owned-by:** [`../measurement/run-directory.md`](../measurement/run-directory.md)
  — the two-file contract is what it enumerates.
- **joins:** [`../measurement/metrics-row.md`](../measurement/metrics-row.md),
  [`../measurement/config-json.md`](../measurement/config-json.md),
  [`../measurement/verdict.md`](../measurement/verdict.md),
  [`../measurement/ladder.md`](../measurement/ladder.md),
  [`worktree-shadows.md`](worktree-shadows.md).
- **looks-like-but-is-not:** `cr_sim/train/notify.py` and
  `cr_sim/train/bot.py`. Both were written to replace this page from a phone
  and neither did; both are **leftover**, still tested
  (`tests/test_notify.py`, `tests/test_bot.py`), and nothing in the training
  loop imports the bot. Reading either as the current reporting path is the
  mistake.

## If you change this

- **Hits:** nothing in `cr_sim/` — it is a pure reader. It hits the *running
  watcher*, which is not restarted and therefore does not change, and that is
  the whole trap. `tests/test_watch.py` and `tests/test_report.py` quote dozens
  of `runs/<name>/...` literals as data, so they encode the naming scheme even
  though they touch no filesystem.
- **Does not hit:** any number. The page computes `best_lift = max` over rows it
  did not write, so a change here cannot make a run better or worse — the
  obvious wrong reading of a page that suddenly shows a different figure is
  that the run moved. It is far more likely the key set did.

## Surfaces

| Surface | Role |
|---|---|
| a person, at `http://localhost:8899` | the only reader that matters |
| `runs/**` and `.claude/worktrees/*/runs/**` | reads |
| `scripts/register_job.py` | writes entries in (root `CLAUDE.md:8-35`) |
| `tests/test_watch.py`, `tests/test_report.py` | pin the naming scheme |

## See

- Source: `cr_sim/train/watch.py`, `cr_sim/train/report.py`,
  `scripts/register_job.py`
