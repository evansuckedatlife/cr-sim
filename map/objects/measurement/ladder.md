---
type: object
cluster: measurement
universe: live
status: verified
verified: 2026-08-30
commit: dc47f51
entity: cr_sim/train/ladder.py
---

# Ladder

A rating fitted over mirrored pairings — `Player`, `Direction`, `Pairing`,
`Rating`, `fit_ratings` — written offline as `runs/<name>/ladder.json` by
`scripts/run_ladder.py`, and produced in-run by `ladder_probe` behind
`--probe ladder`.

Verified 2026-08-30 against the working tree at `dc47f51` (`ladder.py` is among
the nine uncommitted files — `../../CONTEXT.md`).

## Why this shape

The lift is used up. Everything this project has built lands in a band 0.024 sd
wide against a control the expert beats 100-0, and the metric's own resolution
at the n anyone runs is several times wider than the range it is asked to
separate (`cr_sim/train/ladder.py:3-14`). A rating is **transitive**, so a
policy is scored relative to the expert without ever playing it — which is the
whole affordability argument: three cheap anchors is 4.9% of a run, facing the
expert directly is 73% (`:746-752`).

Two structural commitments, both measured rather than assumed:

- **Both directions, always, swapping which side is *controlled*.**
  `CRSimEnv.step` applies the controlled side's action and then calls
  `_opponent_move()` inside every frame-skip window, so the controlled side
  moves first all match. Swapping colour alone would not cancel that; swapping
  the controlled side cancels colour and move order together, and is what makes
  a self-pairing come out at exactly 0.5 rather than approximately
  (`:30-41`, `:571-577`).
- **Greedy and sampled are separate arms throughout** — separate pairings,
  ratings and tables, never averaged (`:43-46`).

## Shape

- `Player` `:129`, with `ref` `:161` — the `ladder_opponent_ref` a row records.
  "pool" is not an opponent; a path is. For a guided search player the
  proposer is *part of which player this is*, stamped from the **effective**
  candidate count, not the requested one (`:170-190`). `load` `:193` routes
  every anchor through `check_observation`.
- `Direction` `:303`, `Pairing` `:332`, `Rating` `:372`.
- `play_pairing` `:562`; `_play_direction` `:514`, which reads the faced
  opponent off the environment `:523-525`, never off the argument.
- `_stream_seed` `:501` — keyed on the **mode itself**, derived arithmetically.
- `fit_ratings` `:604` — Bradley-Terry MAP, draws as half a win each way,
  Gaussian prior. The prior is not decoration: the expert's 100-0 edge makes
  an unregularised likelihood diverge, and the Hessian is what gives every
  player a standard error `:611-618`. `anchor` is held at 0 because
  `random = 0` is the scale every recorded number sits on `:619-623`; a fit
  with the anchor absent and nothing else pinned is **refused** `:636-651`.
- `ladder_probe` `:737` — refuses an anchor the pinned table does not name
  `:772-783`, because pinning it at 0 silently puts it level with a uniform
  random agent and shifts every `ladder_elo` with no field saying so. Emits
  `ladder_pinned` and `ladder_ratings_source` so a reading traces to the table
  that set its scale `:825-828`. Wired at `cr_sim/train/run.py:972-978`.
- `_ladder_ratings` `cr_sim/train/run.py:445` refuses a table whose `mode`
  `:465`, `observation` `:474` or recorded `tower_level` `:481` disagrees with
  the run. The level is checked **only where the file records it** `:458-462`
  — tables written before that field cannot answer the question.
- `runs/<name>/ladder.json` `scripts/run_ladder.py:272`, recording mode,
  observation, `prior_sd` and `tower_level` `:279`.
  `runs/<name>/arms.json` `:401` carries **lifts only, never Elo** `:389-393`;
  a lift is flattened into the verdict only when exactly one arm exists
  `:414-425`, because the file's headline Elo is the top-rated player's and
  `arms[0]` is usually somebody else.
- `check_equal_branch_budget` lives in **`cr_sim/train/proposal.py:204`**, not
  in `ladder.py`, and `scripts/run_ladder.py:236-239` applies it only where
  both sides are search players and at least one is guided — an unguided
  search at a different budget is a legitimate rung.

## Connected to

- **owns:** `ladder.json`, `arms.json`, and the `ladder_*` metrics family.
- **owned-by:** [`verdict.md`](verdict.md) — `check_lift_is_named` demands
  `ladder_opponent`, `ladder_opponent_ref` and `ladder_pinned`; `write_verdict`
  refuses a file carrying both a rating and an unattributed lift.
- **joins:** [`checkpoint.md`](checkpoint.md) (a `Player` is a checkpoint plus
  its head and observation); [`search-bot.md`](search-bot.md) (search anchors
  and `check_equal_branch_budget`); [`self-play.md`](self-play.md)
  (`FrozenOpponent` is what a net player plays as);
  [`random-streams.md`](random-streams.md).
- **looks-like-but-is-not:** the self-play `ancestor_probe`. It also reports a
  score against a past self and also calls itself a ladder in the printout
  (`cr_sim/train/run.py:806-808`), but it fits nothing, is not transitive, and
  writes the `ancestor_` family. Its opponent is an integer age against a pool
  that evicts from the middle.

## If you change this

- **Hits:** every `ladder.json` on disk becomes a table the in-run probe may
  refuse, because `_ladder_ratings` compares recorded fields to the running
  configuration and `ladder_probe` compares anchor **names** to the table's
  keys. A rename of a player is enough: `parse_player('runs/x/cloned.pt')` is
  `'x:cloned'` while `'clone=runs/x/cloned.pt'` is `'clone'`, and the second
  matches nothing.
- **Does not hit:** the lift, and this is the one to get right. Elo is fitted
  on **crowns**, which no reward touches, so a ladder needs no `eval_reward`
  and a change to the rating cannot move any recorded lift
  (`cr_sim/train/evaluate.py:477-480`, `scripts/run_ladder.py:429-434`). The obvious next move —
  plotting a rating and a lift on one axis because both went up — is the
  conflation `write_verdict:526-545` exists to refuse.

## Surfaces

| Surface | Role |
|---|---|
| `scripts/run_ladder.py` | writes `ladder.json`, `arms.json`, `verdict.json`, `metrics.jsonl` |
| `cr_sim.train.run --probe ladder` | reads a table, writes `ladder_*` rows, promotes on `rolling_ladder_elo` |
| `cr_sim/train/watch.py` | `read_ladder` `:775` is the reader, **landed dark on purpose** — `ladder.json` is invisible on the progress page |
| `scripts/expert_iterate.py` | rates each round through `run_ladder.py` as a subprocess |

## See

- Source: `cr_sim/train/ladder.py`
