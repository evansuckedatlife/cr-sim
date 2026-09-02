---
type: process
status: verified
verified: 2026-08-30
commit: dc47f51
consumes: [objects/measurement/checkpoint.md, objects/measurement/ladder.md, objects/measurement/search-bot.md]
produces: [objects/measurement/ladder.md, objects/measurement/verdict.md, objects/measurement/run-directory.md]
---

# rate-on-the-ladder

Play a star of mirrored pairings, fit one Bradley-Terry rating over the graph
with `random` pinned at 0, and write the table beside the per-player lifts that
put it back on the historical scale.

Offline: `scripts/run_ladder.py:143`. In-run: `--probe ladder`, which builds
`ladder_probe` (`cr_sim/train/ladder.py:737`) at `cr_sim/train/run.py:969-978`.

## Input → Movement → Output

In: one or more `--entrant`s, one or more `--anchor`s, an episode count and a
mode. Movement: every entrant plays every anchor in both directions over the
same seed block; the results are folded into one edge per unordered pair and
fitted; then each net player is separately measured against the shared random
control. Out: `ladder.json` (the native table), `arms.json` (the lifts),
`verdict.json`, `metrics.jsonl`, `config.json`.

## Why this shape

**A star, not a round robin.** 33 loadable checkpoints is 528 pairings at four
minutes each — 35 hours. Against four fixed anchors it is 132 pairings, and the
rating is transitive, so the star answers the same question provided the graph
stays connected (`scripts/run_ladder.py:8-12`).

**Mirrored, and folded.** A rating is a property of the match-up, not of the
colour, so both directions go into one edge (`cr_sim/train/ladder.py:657-658`).

**The prior is not decoration.** The expert beats the random control 100-0, and
an unregularised Bradley-Terry fit diverges on that edge — the maximum is at
infinity. `N(0, prior_sd)` makes the mode finite and its Hessian gives every
player a standard error (`cr_sim/train/ladder.py:609-617`).

**Something must be pinned or the scale does not exist.** `fit_ratings` raises
when the anchor is absent from the graph and nothing else is pinned
(`:636-651`): measured, a roster with no `random` in it rated a checkpoint at
-71 Elo and printed it as losing to a uniform random agent, where the same
weights on the same seeds rate +200 in a ladder that contains random.

**An Elo is not a lift and they never share a field.** `ladder.json` holds
`elo`; `arms.json` holds `lift`; `verdict.json` flattens a lift **only when
there is exactly one arm to flatten** (`scripts/run_ladder.py:404-425`), because
`arms[0]` is one player's lift and the headline Elo is usually somebody else's.
`runs/agent-ladder-v1` is the file where that went wrong.

## Steps

1. Parse and refuse three ways: no entrant or no anchor (`:188-190`); two
   players sharing a name, which merges two entrants into one rated row
   (`:193-199`); players trained on different observations, which cannot play
   each other at all because the environment encodes one observation for both
   sides (`:201-207`).
2. Build the task list — the star, plus any explicit `--pairing` (`:217-224`).
   Tasks are plain data: checkpoints are re-read from disk in the worker rather
   than shipped as weights (`:91-97`).
3. Enforce an equal branch budget over **every** task, not only the explicit
   ones, and only where the claim needs it — both sides search and at least one
   is guided (`:235-239`; `check_equal_branch_budget` at
   `cr_sim/train/proposal.py:204`). A guided bot that quietly took sixteen
   branches against fourteen would win on the budget alone.
4. Play each pairing (`play`, `:91-125`), single-process or across a
   `ProcessPoolExecutor` (`:247-259`). Each worker sets `torch.set_num_threads(1)`
   (`:104`) — a batch-of-one forward gets nothing from more.
5. `play_pairing` → two `_play_direction`s (`cr_sim/train/ladder.py:562`, `:514`). Each
   direction derives its stream arithmetically from the battles and the mode
   (`_stream_seed`, `:501-511`), never from `hash()`, which is salted per
   process. The opponent is handed to the *factory* (`:522`), and the label is
   read back **off the environment** (`:525`).
6. A direction's score is `(wins + 0.5 * draws) / n` over crown differences
   (`:550-559`). **Crowns, not returns** — which is why a rating needs no
   reward.
7. `fit_ratings(pairings, prior_sd=..., anchor="random")` (`:262`;
   `cr_sim/train/ladder.py:604`).
8. Write `ladder.json` (`:272-297`), recording the arena — `tower_level` at
   `:279` — so `--ladder-ratings` can refuse a table fitted on another game
   (`cr_sim/train/run.py:480-486`).
9. Write metrics rows: one per **pairing direction** carrying that direction's
   own score and nothing else (`:307-327`), then one per **player** for the
   rating, which names the whole roster as what it was measured against
   (`:329-357`). Both go through `check_lift_is_named` (`:310`, `:337`), which
   has a second door for score rows and a third for `ladder_elo` without
   `ladder_pinned` (`cr_sim/train/selfplay.py:129-159`).
10. Unless `--no-arms`, play every net player against the shared random control
    on the same seeds through `evaluate_paired(..., modes=(args.mode,))`
    (`:359-402`) — this is [`evaluate-against-a-control.md`](evaluate-against-a-control.md)
    in single-arm mode, and it is the bridge back to every historical number.
11. Write `verdict.json` (`:426-453`) and `config.json` with `kind: "job"`
    (`:461-484`).

## The in-run variant

`--probe ladder` rates the live policy against loaded anchors every
`--eval-every` updates. Two guards exist only on this path:

- `_ladder_anchors` (`cr_sim/train/run.py:490-507`) calls `Player.load`, which the in-run
  path did not: `--ladder-anchor checkpoints/headablate-flat.pt` trained to the
  first evaluation and died in `FrozenOpponent._snapshot` twenty updates in with
  no evaluation ever written. Loading also routes every anchor through
  `check_observation`.
- `_ladder_ratings` (`cr_sim/train/run.py:445-487`) refuses a table whose `mode`,
  `observation` or `tower_level` disagrees with the run. `tower_level` is checked
  **only where the file records it** (`:458-461`, `:480`) — tables written before
  that field existed cannot answer, and refusing them would make the flag
  unusable against every table on this machine.

## If you change this

- **Hits:** [`objects/measurement/ladder.md`](../objects/measurement/ladder.md)
  — `Player`, `Direction`, `Pairing`, `Rating` and the two files.
  [`objects/measurement/checkpoint.md`](../objects/measurement/checkpoint.md) —
  `parse_player` (`cr_sim/train/ladder.py:271`) and `player_from_checkpoint` (`:240`)
  identify a checkpoint's head by **parameter count** (`HEAD_BY_PARAMETERS`,
  `:98`) for the 22-of-42 checkpoints recording neither `head` nor
  `observation`; any change to `ActorCritic`'s parameter count silently
  invalidates that table.
  `cr_sim/train/run.py:445-487` and `:490-507`, the in-run consumers of `ladder.json`.
  `scripts/expert_iterate.py:176-185`, which builds this command line.
- **Does not hit:** the reward. A rating is fitted on crowns
  (`cr_sim/train/ladder.py:550-559`), so `--reward`, `--shaping`, `--tower-weight` and the
  whole of [`objects/measurement/reward-schedule.md`](../objects/measurement/reward-schedule.md)
  move every lift in `arms.json` and move **no** Elo in `ladder.json`. That
  asymmetry is why `write_verdict` keys its `eval_reward` clause to the lift keys
  and not to the file (`cr_sim/train/evaluate.py:477-480`), and why `run_ladder.py` writes
  `eval_reward` only when an arm was actually played (`:434`).

## A file nobody renders

`ladder.json` is **invisible on the progress page**, deliberately. Its reader,
`cr_sim.train.watch.read_ladder` (`cr_sim/train/watch.py:775`, path at `:811`),
is landed and dark — the running watcher holds the older module and must not be
restarted, so a `watch.py` edit is inert until somebody does (root `CLAUDE.md`,
"The watcher runs stale code"). Its only callers are `tests/test_ladder.py`. The
run's own `config.json` note says so in prose (`scripts/run_ladder.py:476-480`) rather
than letting a reader assume the page shows everything.

## Surfaces

| Surface | Role |
|---|---|
| a human at a shell | `--workers 8`, about an hour for a real ladder |
| `scripts/expert_iterate.py` | writes this command line (`:176-185`) |
| `cr_sim/train/run.py` | reads `ladder.json` through `--ladder-ratings` |
| `cr_sim/train/watch.py` | renders `arms.json`'s shape; `read_ladder` is dark |
| `cr_sim/train/report.py` | `rated(...)`, and `ladder_ratings_source` (`tests/test_report.py:120-126`) |
| `tests/test_ladder.py`, `tests/test_measurement.py`, `tests/test_proposal.py` | hold the refusals, the fit and the branch budget |

## Citations

Every `path:line` in the steps above is rooted at the repo root and was
resolved against the tree, together with the AST extent of the symbol it
is named beside (`../_meta/check.py`).

Verified 2026-08-30 against `main` @ `dc47f51`, **working tree**: `cr_sim/train/ladder.py`, `scripts/run_ladder.py`
are among the nine files uncommitted at that commit, so their line
numbers are working-tree numbers and were re-checked here rather than
inherited (`../CONTEXT.md`, Verification basis).

## See

- Objects: [`ladder.md`](../objects/measurement/ladder.md),
  [`verdict.md`](../objects/measurement/verdict.md),
  [`checkpoint.md`](../objects/measurement/checkpoint.md),
  [`search-bot.md`](../objects/measurement/search-bot.md)
- Source: `scripts/run_ladder.py`, `cr_sim/train/ladder.py:501-839`
