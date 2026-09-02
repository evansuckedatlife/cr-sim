# What points IN — the consumers outside the tree

[`CONTEXT.md`](CONTEXT.md) answers *I am changing X, what inside this tree
moves*. This file answers the other direction, and it is the one the tree cannot
tell you: these consumers hold paths **into** `cr_sim/` and into its generated
data, nothing in `cr_sim/` imports them, so no card names them until someone
goes looking. They break silently for exactly that reason.

Its own file rather than a section of the hub, because it answers a different
question and no forward entry needs it loaded. Every row lands on a card, and
the card is where the detail lives — a row here is a pointer.

---

## From `tests/`

| Points in at | What it hardcodes | Lands on |
|---|---|---|
| `tests/test_data_pipeline.py:22` | `ROOT/"data_cache"/"csv_logic"` — **and 33 test files import `BUILD` from it**, of which 30 have no skip of their own. Only this file's own `data` fixture skips cleanly (`:28-29`); the rest call `LogicData.load(BUILD)` in their own `world` fixture, so a moved `data_cache` is ~30 files of errors, not skips | [`../objects/build/logic-data.md`](../objects/build/logic-data.md) |
| `tests/test_data_pipeline.py:23` | `ROOT/"reference"/"anchors.json"`, read at **import** time with no guard — a collection-time dependency for the whole suite | [`../objects/build/anchors.md`](../objects/build/anchors.md) |
| `tests/test_measurement.py:925` | the literal `"checkpoints/headablate-factored.pt"`, passed to `evaluate_vs_expert.main` and resolved against the **process cwd**. `checkpoints/` is gitignored generated data and there is no skip | [`../objects/measurement/checkpoint.md`](../objects/measurement/checkpoint.md) |
| `tests/test_measurement.py:920` | `parents[1]/"scripts"` on `sys.path`, then `import evaluate_vs_expert` — a test that imports a script by filesystem position | [`../processes/evaluate-against-a-control.md`](../processes/evaluate-against-a-control.md) |
| `tests/test_interaction_matrix.py:12-13` | `reference/hits_to_kill.csv` and its `.md`, deliberately with **no** agreement floor, because the sheet is a year old | [`../objects/build/validation-gates.md`](../objects/build/validation-gates.md) |
| `tests/test_spells.py:9`, `tests/test_status_effects.py:66` | damage and speed figures **hand-transcribed** from `reference/anchors.json`, with the source named in a comment rather than read. `tests/test_mumu_geometry.py:24-29` does the opposite and reads from `cr_sim.engine.arena` | [`../objects/build/anchors.md`](../objects/build/anchors.md) |
| `tests/test_train.py:940-951` | `expert_iterate`'s two command builders, parsed back through the targets' own parsers | [`../processes/expert-iterate.md`](../processes/expert-iterate.md) |
| `tests/test_ladder.py`, `tests/test_watch.py`, `tests/test_report.py` | dozens of `runs/<name>/...` string literals used as **data** — checkpoint refs, verdict paths, ladder sources. They touch no filesystem, but they encode the naming scheme, so a rename of the run-directory convention breaks them | [`../objects/measurement/run-directory.md`](../objects/measurement/run-directory.md) |

## From `scripts/`

| Points in at | What it hardcodes | Lands on |
|---|---|---|
| every script's `sys.path.insert(0, parents[1])` — `scripts/make_demos.py:27`, `scripts/clone_policy.py:31`, `scripts/run_ladder.py:44`, `scripts/evaluate_vs_expert.py:41`, `scripts/evaluate_checkpoints.py:13`, `scripts/expert_iterate.py:42-43` | `cr_sim` is imported **by repo position**, not by install. Move a script one directory deeper and it imports a different `cr_sim` or none | [`../processes/CONTEXT.md`](../processes/CONTEXT.md) — each verb's card |
| `scripts/make_demos.py:70` `data_cache/demos`; `scripts/clone_policy.py:160-161` `data_cache/demos` → `runs/cloned`; `scripts/run_ladder.py:174` `runs`; `scripts/expert_iterate.py:100-101` `ROOT/runs`, `ROOT/data_cache` | the generated-data layout, in argparse defaults | [`../objects/measurement/run-directory.md`](../objects/measurement/run-directory.md) |
| `scripts/expert_iterate.py:137`, `:150`, `:176` | the *filenames* `scripts/make_demos.py`, `scripts/clone_policy.py`, `scripts/run_ladder.py`, invoked through `subprocess.run` with `cwd=ROOT` | [`../processes/expert-iterate.md`](../processes/expert-iterate.md) |
| `scripts/register_job.py:38`, `:50-58` | `ROOT/"runs"/<name>/{metrics.jsonl,config.json}` — the two-file contract the watcher enumerates, written by the one caller that never calls `check_lift_is_named` | [`../objects/surfaces/progress-page.md`](../objects/surfaces/progress-page.md) |

## From the progress watcher

Every row here lands on one card:
[`../objects/surfaces/progress-page.md`](../objects/surfaces/progress-page.md),
which owns `_run_roots`, `_kind_of`, the two `rglob` depths, the four-key A/B
budget and the default output path — and the rule, written only in the root
`CLAUDE.md`, that the running watcher holds the module it was started with.

The one row that lands somewhere else:
`cr_sim/train/watch.py:4499`, `:4516` reach verdicts and summaries by
`rglob("*verdict*.json")` / `rglob("summary.json")`, which is the **only** way a
verdict in a directory with no metrics file is visible at all —
[`../objects/measurement/verdict.md`](../objects/measurement/verdict.md).

## From the root `CLAUDE.md` — rules with no enforcement in the tree

| The rule | What it hardcodes | Lands on |
|---|---|---|
| "Every living python process belongs on the progress page" (`:8-35`) | a literal `scripts/register_job.py` command line, and the claim that `runs/<name>/{metrics.jsonl,config.json}` "is all the watcher enumerates" | [`../objects/measurement/run-directory.md`](../objects/measurement/run-directory.md) |
| "The watcher runs stale code" (`:48-54`) | that a `cr_sim/train/watch.py` edit is inert until the watcher is restarted, and that a fresh timestamp proves nothing | [`../objects/surfaces/progress-page.md`](../objects/surfaces/progress-page.md) |
| the 0.062 sd noise floor (`:67-77`) | `runs/sampled-noise-floor/noise.json`, four specific lifts, and a warning that the older ±0.02–0.04 figure is 4x too small — "anything sized off it, battle counts especially, is off by 4x" | [`../objects/measurement/lift-callers.md`](../objects/measurement/lift-callers.md) |
| "A lift also needs the reward it was counted in" (`:78-84`) | the two literal scale strings `simple:shaping=0.01` and `projected:tower=1,elixir=0.3,horizon_seconds=3` | [`quoting-a-result.md`](quoting-a-result.md) |
| "Point `--out` at a file for every evaluation worth citing" (`:88-90`) | the `+1.813` transcription error against the real `+1.623` | [`quoting-a-result.md`](quoting-a-result.md) |
| `runs/`, `data_cache/`, `checkpoints/` are gitignored, and 194 MB exists only in a worktree (`:114-116`) | that the tests in the first table depend on data git does not carry | [`../objects/surfaces/worktree-shadows.md`](../objects/surfaces/worktree-shadows.md) |
| the map routing row (`:6`) | `map/CLAUDE.md`. This is the **only** edit this map makes outside `map/` | this file |

## From the registered jobs under `runs/`

Read-only history, and *evidence*: `runs/agent-ladder-v1` holds the row where a
whole-graph Elo sat beside four different `eval_opponent`s;
`runs/audit-ladder-greedy` holds the verdict where the worst-rated entrant's
lift was reported as the top player's; `runs/iter-1/cloned.pt` records
`targets: 'hard'` over a shard collected for soft targets. Several tests quote
them by name and by figure.

The shape of a run directory — the two-file contract, the `rglob` depth, the
five writers — is on
[`../objects/measurement/run-directory.md`](../objects/measurement/run-directory.md),
and no count of what is on disk is recorded anywhere in this map: those drift
between one reading and the next.

**What this means for a change:** a rename of a metrics key, a config key or the
run-directory convention breaks no test that reads these directories — nothing
does. It breaks the *comparability* of every recorded result, and the only place
that shows up is the page. Lands on
[`../objects/measurement/run-directory.md`](../objects/measurement/run-directory.md)
and [`../objects/measurement/metrics-row.md`](../objects/measurement/metrics-row.md).

## From `cr_sim/play/` and `cr_sim/mumu/` — inside the package, outside the loop

Nothing in the training loop imports these, so no `interface` or `measurement`
card would ever name them. Both now have cards, and the detail lives there:

- [`../objects/surfaces/play-server.md`](../objects/surfaces/play-server.md) —
  the second `DEFAULT_DECK`, a fourth `--tower-level` defaulting to 11, a
  hand-restated `NetConfig`, an `nvec` literal beside the constant it spells
  out, and a `PolicyOpponent` that builds a named generator and then samples off
  torch's global stream.
- [`../objects/surfaces/mumu.md`](../objects/surfaces/mumu.md) — the emulator
  bridge, and the only path that settles an open anchor against a real client.
- [`../objects/surfaces/cli.md`](../objects/surfaces/cli.md) — nine subcommands,
  the only consumer of `cr_sim/data/interactions.py` and
  `cr_sim/data/engagement.py`, and the `--level` flags that do exist.

## From the worktrees

`.claude/worktrees/` holds three agent checkouts, each with its own
`cr_sim/api/encoding.py` from a 9- or 13-channel era, plus a fourth partial copy
under `runs/_diag/`. **Any `grep -rn` from the repo root returns these first**,
and `cr_sim/train/watch.py:4427-4441` reads their `runs/` on purpose. They are
not the subject tree, and no citation in this map may resolve into one —
[`../objects/surfaces/worktree-shadows.md`](../objects/surfaces/worktree-shadows.md),
enforced by `map/_meta/check.py`.

Never `git worktree remove --force` — it follows Windows directory junctions and
has already destroyed the real `data_cache` once (root `CLAUDE.md:110-111`).
