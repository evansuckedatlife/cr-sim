# processes/ — the verb shelf

Six movements actually run in this repo. Each has a card here; nothing else
does, and that is a rule rather than an accident (`../_meta/schema.md`).

A card here answers *what moves, in what order, and what comes out with it*.
It does not restate what the nouns are — those are in `../objects/`, reached by
link. If a step needs a paragraph about a type, the paragraph belongs on that
type's card and the step cites it.

## The six, in the order they compose

| Verb | Card | Entry point | Produces |
|---|---|---|---|
| collect demonstrations | [`collect-demonstrations.md`](collect-demonstrations.md) | `scripts/make_demos.py:266` | `data_cache/<set>/shard-NN.npz` |
| clone | [`clone.md`](clone.md) | `scripts/clone_policy.py:219` | `runs/<name>/cloned.pt` + the four-file run directory |
| fine-tune | [`fine-tune.md`](fine-tune.md) | `python -m cr_sim.train.run` → `cr_sim/train/run.py:510` | `runs/<name>/{best,checkpoint,final}.pt` + the run directory |
| evaluate against a control | [`evaluate-against-a-control.md`](evaluate-against-a-control.md) | `cr_sim/train/evaluate.py:692` and three `scripts/` | `verdict.json` |
| rate on the ladder | [`rate-on-the-ladder.md`](rate-on-the-ladder.md) | `scripts/run_ladder.py:143`; in-run at `cr_sim/train/ladder.py:737` | `ladder.json`, `arms.json` |
| expert-iterate | [`expert-iterate.md`](expert-iterate.md) | `scripts/expert_iterate.py:156` | a round's demos, clone and rating; `runs/expert-iteration.json` |

The sixth is a driver over the first, second and fifth. It invents no
measurement of its own and says so in its own docstring
(`scripts/expert_iterate.py:21-26`) — which is why it is a verb here and not a
seventh kind of thing.

## Why there is no seventh

Several things in `scripts/` look like movements and are not. Each is a
**leftover** or a one-shot, filed in
[`../objects/surfaces/_index.md`](../objects/surfaces/_index.md),
and none earns a card:

- `cr_sim/soak.py` — a hundred thousand matches with no policy in them. It
  measures the engine, not an agent. No artefact any other verb consumes.
- `scripts/bench_engine.py`, `scripts/measure_sampled_noise.py` — profiling and
  a one-time constant. `measure_sampled_noise` ran once and produced the 0.062
  sd figure now quoted in the root `CLAUDE.md` and at `cr_sim/train/evaluate.py:159-169`.
  A constant is not a repeated movement.
- `scripts/extract_apk.py`, `scripts/extract_icons.py` — the one-time build
  extraction that produced `data_cache/csv_logic`. Ran once, in 2025.
- `scripts/evaluate_decks.py`, `scripts/summarize_decks.py` — a closed question
  about `FactoredStatsHead`, answered in `runs/agent-card-stat-encoder`.
- `scripts/register_job.py` — writes two files so the watcher can see a
  process. It moves nothing; it is a *surface*
  ([`../objects/surfaces/progress-page.md`](../objects/surfaces/progress-page.md)),
  and it is where [`../effects/points-in.md`](../effects/points-in.md) lands
  hardest, because it is the one writer of a metrics row that does not go
  through `check_lift_is_named` (`scripts/register_job.py:50` — named as the
  exception in source at `cr_sim/train/watch.py:526-528`).

## The loop, stated once

The expert is a one-ply search over the exact engine, not a network. It plays;
`collect` writes down every decision where more than one action was legal. A
clone fits those decisions and inherits their **value column** as its critic.
PPO fine-tunes the clone. A ladder rates the result against fixed anchors.
Expert iteration then makes the clone the *proposer* for the next collection,
which is the only arrow of AlphaZero's operator this project was missing
(`scripts/expert_iterate.py:5-11`).

Two things travel with an artefact through every one of those steps and are
the reason five of the six historical bugs were invisible: **the reward its
numbers are denominated in**, and **the environment they were produced in**.
Every card here has a step that says where each is stamped, and a *Does not
hit* that names where it is not.

## Rules for a card in here

- Copy `../_templates/process.md`. Input → Movement → Output, then numbered
  steps, each with a `path:line` **rooted at the repo root**, then the
  `## Citations` block. That block is not optional: `status: verified` needs a
  date, a commit and citations (`../_meta/schema.md`), and all six cards here
  shipped without one while the template had no slot for it.
- `consumes` / `produces` are **links to object cards**, not prose. If the
  noun has no card, link the index line's cluster and say so.
- Steps cite; they do not paraphrase source comments. Where a docstring and
  the code disagree, the code wins and the step says which docstring it is
  overriding.
- Never restate a unit convention or a name collision — `../CONTEXT.md` owns
  those once.
- **Hits / Does not hit is first-order.** "Does not hit" names the obvious next
  file that is the *wrong* one.
